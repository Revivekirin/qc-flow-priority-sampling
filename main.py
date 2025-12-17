import glob
import json
import os
import random
import time

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets
from envs.robomimic_utils import is_robomimic_env
from evaluation import evaluate
from log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb
from utils.flax_utils import save_agent
from utils.datasets import Dataset, ReplayBuffer, PriorityTrajectorySampler

from cluster_vis import build_traj_features, visualize_traj_clusters, select_k_by_value_homogeneity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

FLAGS = flags.FLAGS

flags.DEFINE_string("run_group", "Debug", "Run group.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("env_name", "cube-triple-play-singletask-task2-v0", "Environment.")
flags.DEFINE_string("save_dir", "exp/", "Save directory.")

flags.DEFINE_integer("offline_steps", 1000000, "Offline RL steps.")
flags.DEFINE_integer("online_steps", 1000000, "Online RL steps.")
flags.DEFINE_integer("buffer_size", 2000000, "Replay buffer size.")
flags.DEFINE_integer("log_interval", 5000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 100000, "Evaluation interval.")
flags.DEFINE_integer("save_interval", 100000, "Save interval.")
flags.DEFINE_integer("start_training", 5000, "When online training begins.")

flags.DEFINE_integer("utd_ratio", 1, "Update-to-data ratio.")
flags.DEFINE_float("discount", 0.99, "Discount factor.")

flags.DEFINE_integer("eval_episodes", 50, "Evaluation episodes.")
flags.DEFINE_integer("video_episodes", 0, "Video episodes.")
flags.DEFINE_integer("video_frame_skip", 3, "Video frame skip.")

config_flags.DEFINE_config_file("agent", "agents/acfql.py", lock_config=False)

flags.DEFINE_float("dataset_proportion", 1.0, "Percentage of dataset to load.")
flags.DEFINE_integer("dataset_replace_interval", 1000, "Large dataset rotation interval (OGBench).")
flags.DEFINE_string("ogbench_dataset_dir", None, "OGBench dataset directory.")

flags.DEFINE_integer("horizon_length", 5, "Action chunk horizon.")
flags.DEFINE_bool("sparse", False, "Sparse reward flag.")

flags.DEFINE_string('entity', 'entity', 'wandb entity')

flags.DEFINE_bool("use_ptr_backward", True, "Use PTR trajectory-priority sampling.")
flags.DEFINE_bool("use_ptr_online_priority", True, "Update PTR priorities online.")
flags.DEFINE_bool("save_all_online_states", False, "Save trajectory states.")
flags.DEFINE_bool("backward", True, "PTR backward sampling.")
flags.DEFINE_float("beta", 0.5, "Beta for PTR sarsa target critic weighted target.")
flags.DEFINE_bool("use_weighted_target", False, "Use PTR sarsa target critic weighted target.")
flags.DEFINE_string("metric", "success_binary", "PTR priority metric")
flags.DEFINE_integer("ptr_warmup_steps", 20000, "Number of online steps before enabling PTR sampling.")

flags.DEFINE_bool("cluster_sampler", False, "Cluster Sampler (kmeans)")


class LoggingHelper:
    def __init__(self, csv_loggers, wandb_logger):
        self.csv_loggers = csv_loggers
        self.wandb_logger = wandb_logger

    def log(self, data, prefix, step, wandb_step=None):
        if not data:
            return
        self.csv_loggers[prefix].log(data, step)
        if wandb_step is None:
            wandb_step = step
        self.wandb_logger.log(
            {f"{prefix}/{k}": v for k, v in data.items()},
            step=wandb_step,
        )


def sample_sequence_PTR(
    dataset,
    ptr_sampler,
    batch_size,
    seq_len,
    discount,
    backward=True,
    log_prefix=None,
    global_step=None,
):
    traj_ids = ptr_sampler.sample_trajectory_indices(batch_size)
    boundaries = ptr_sampler.trajectory_boundaries

    start_idxs = np.empty(batch_size, dtype=np.int64)

    rel_starts = []
    traj_success_flags = []
    seg_returns = []
    last_rewards = []
    sampled_priors = []

    for i, tid in enumerate(traj_ids):
        start, end = boundaries[tid]
        traj_len = end - start + 1
        if traj_len < seq_len:
            start_idxs[i] = np.random.randint(0, dataset.size - seq_len + 1)
            continue
        if backward:
            if traj_len >= seq_len:
                s = end - seq_len + 1
            else:
                s = start
        else:
            max_start = min(end - seq_len + 1, dataset.size - seq_len)
            s = np.random.randint(start, max_start + 1) if max_start >= start else start

        start_idxs[i] = s

        if log_prefix is not None:
            denom = max(1, traj_len - seq_len + 1)
            rel_pos = (s - start) / denom
            rel_starts.append(rel_pos)

            if hasattr(ptr_sampler, "success_flags"):
                traj_success_flags.append(float(ptr_sampler.success_flags[tid]))

            seg_r = dataset["rewards"][s : min(s + seq_len, end + 1)]
            seg_returns.append(float(seg_r.sum()))
            last_rewards.append(float(seg_r[-1]))

            if ptr_sampler.priorities is not None:
                sampled_priors.append(float(ptr_sampler.priorities[tid]))

    batch = dataset.sample_sequence_from_start_idxs(start_idxs, seq_len, discount)

    batch = dict(batch)
    batch["traj_ids"] = np.asarray(traj_ids, dtype=np.int32)

    if log_prefix is not None and global_step is not None and len(rel_starts) > 0:
        log_dict = {
            f"{log_prefix}/rel_start_mean": float(np.mean(rel_starts)),
            f"{log_prefix}/rel_start_std": float(np.std(rel_starts)),
            f"{log_prefix}/seg_return_mean": float(np.mean(seg_returns)),
            f"{log_prefix}/seg_return_std": float(np.std(seg_returns)),
            f"{log_prefix}/r_last_mean": float(np.mean(last_rewards)),
        }
        if len(traj_success_flags) > 0:
            log_dict[f"{log_prefix}/success_traj_ratio"] = float(np.mean(traj_success_flags))
        if len(sampled_priors) > 0:
            log_dict[f"{log_prefix}/prior_mean"] = float(np.mean(sampled_priors))
            log_dict[f"{log_prefix}/prior_std"] = float(np.std(sampled_priors))

        wandb.log(log_dict, step=global_step)

    return batch


class ClusterBalancedSampler:
    def __init__(self, cluster_ids, trajectory_boundaries):
        self.cluster_ids = np.array(cluster_ids)
        self.boundaries = trajectory_boundaries

        self.clusters = {}
        for cid in np.unique(cluster_ids):
            idxs = np.where(cluster_ids == cid)[0]
            self.clusters[cid] = idxs

        self.K = len(self.clusters)

    def sample_traj_ids(self, batch_size):
        per_cluster = batch_size // self.K
        remainder = batch_size % self.K

        traj_ids = []
        for cid in range(self.K):
            idxs = self.clusters[cid]
            chosen = np.random.choice(idxs, per_cluster, replace=True)
            traj_ids.extend(chosen)

        flat_ids = np.concatenate(list(self.clusters.values()))
        if remainder > 0:
            extra = np.random.choice(flat_ids, remainder, replace=True)
            traj_ids.extend(extra)

        return np.array(traj_ids, dtype=np.int32)

    def sample_sequence(self, dataset, batch_size, seq_len, discount, return_ids=True):
        traj_ids = self.sample_traj_ids(batch_size)
        starts = []
        for tid in traj_ids:
            s, e = self.boundaries[tid]
            max_s = e - seq_len + 1
            if max_s < s:
                starts.append(s)
            else:
                starts.append(np.random.randint(s, max_s + 1))

        batch = dataset.sample_sequence_from_start_idxs(starts, seq_len, discount)
        if return_ids:
            batch = dict(batch)
            batch["traj_ids"] = traj_ids
            batch["cluster_ids"] = self.cluster_ids[traj_ids]
        return batch


def main(_):
    exp_name = get_exp_name(FLAGS.seed)
    run = setup_wandb(entity=FLAGS.entity, project="qc", group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(
        FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name
    )
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    with open(os.path.join(FLAGS.save_dir, "flags.json"), "w") as f:
        json.dump(get_flag_dict(), f)

    config = FLAGS.agent
    config["horizon_length"] = FLAGS.horizon_length
    config["use_weighted_target"] = FLAGS.use_weighted_target
    config["beta"] = FLAGS.beta

    if FLAGS.ogbench_dataset_dir:
        dataset_paths = sorted(
            f for f in glob.glob(f"{FLAGS.ogbench_dataset_dir}/*.npz")
            if "-val.npz" not in f
        )
        env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
            FLAGS.env_name,
            dataset_path=dataset_paths[0],
            compact_dataset=False,
        )
    else:
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name)

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    online_rng, _ = jax.random.split(jax.random.PRNGKey(FLAGS.seed))

    def process_train_dataset(ds):
        ds = Dataset.create(**ds)

        if FLAGS.dataset_proportion < 1.0:
            newN = int(len(ds["masks"]) * FLAGS.dataset_proportion)
            ds = Dataset.create(**{k: v[:newN] for k, v in ds.items()})

        if is_robomimic_env(FLAGS.env_name):
            rew = ds["rewards"] - 1.0
            d2 = dict(ds)
            d2["rewards"] = rew
            ds = Dataset.create(**d2)

        if FLAGS.sparse:
            rew = (ds["rewards"] != 0.0) * -1.0
            d2 = dict(ds)
            d2["rewards"] = rew
            ds = Dataset.create(**d2)

        return ds

    train_dataset = process_train_dataset(train_dataset)

    d = dict(train_dataset)
    if "is_success" not in d:
        d["is_success"] = np.zeros_like(d["terminals"], dtype=np.float32)

    replay_buffer = ReplayBuffer.create_from_initial_dataset(
        d,
        size=max(FLAGS.buffer_size, train_dataset.size + 1),
    )

    (terminal_locs,) = np.nonzero(train_dataset["terminals"] > 0)
    initial_locs = np.concatenate([[0], terminal_locs[:-1] + 1])
    trajectory_boundaries = list(zip(initial_locs, terminal_locs))

    num_traj = len(trajectory_boundaries)

    traj_returns = []
    for (s, e) in trajectory_boundaries:
        r = train_dataset["rewards"][s : e + 1]
        traj_returns.append(float(r.sum()))
    traj_returns = np.asarray(traj_returns, dtype=np.float64)

    features, lengths = build_traj_features(
        train_dataset,
        trajectory_boundaries,
        obs_key="observations",
        obs_slice=None,
        k_keyframes=20,
    )

    ptr_sampler = None
    if FLAGS.use_ptr_backward:
        ptr_sampler = PriorityTrajectorySampler(
            trajectory_boundaries=trajectory_boundaries,
            rewards_source=replay_buffer["rewards"],
            success_source=replay_buffer["is_success"],
            metric=FLAGS.metric,
            temperature=1.0,
        )

    ptr_sampler.num_offline_traj = len(trajectory_boundaries)

    online_ep_start = replay_buffer.pointer

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features).astype(np.float64)
    pca_2d = PCA(n_components=2, random_state=0)
    X_pca = pca_2d.fit_transform(X_scaled)

    k_candidates = list(range(5, 13))
    K = select_k_by_value_homogeneity(X_scaled, traj_returns, k_candidates)
    print(f"Selected K for clustering: {K}")
    print("Selected K:", K)

    kmeans = KMeans(n_clusters=K, random_state=0)
    cluster_ids = kmeans.fit_predict(X_scaled)
    visualize_traj_clusters(
        X_pca,
        cluster_ids,
        lengths,
        save_prefix=f"{FLAGS.env_name}_kmeans_{K}"
    )
    print(f"kmeans pngs are saved!")

    cluster_stats = []
    for cid in range(K):
        idxs = np.where(cluster_ids == cid)[0]
        if len(idxs) == 0:
            continue

        lengths_c = lengths[idxs]
        returns_c = traj_returns[idxs]
        cluster_stats.append(
            {
                "cid": cid,
                "n": int(len(idxs)),
                "len_mean": float(lengths_c.mean()),
                "len_std": float(lengths_c.std()),
                "ret_mean": float(returns_c.mean()),
            }
        )
    columns = ["cid", "n", "len_mean", "len_std", "ret_mean"]
    data = [[d[c] for c in columns] for d in cluster_stats]

    wandb.log({"cluster/stats": wandb.Table(columns=columns, data=data)})

    cluster_sampler = ClusterBalancedSampler(
        cluster_ids=cluster_ids,
        trajectory_boundaries=trajectory_boundaries
    )

    example_batch = train_dataset.sample(1)
    obs_example = example_batch["observations"]
    act_example = example_batch["actions"]
    agent_class = agents[config["agent_name"]]
    agent = agent_class.create(
        FLAGS.seed,
        obs_example,
        act_example,
        config,
    )

    prefixes = ["eval", "env"]
    if FLAGS.offline_steps > 0:
        prefixes.append("offline_agent")
    if FLAGS.online_steps > 0:
        prefixes.append("online_agent")

    loggers = {
        p: CsvLogger(os.path.join(FLAGS.save_dir, f"{p}.csv")) for p in prefixes
    }
    logger = LoggingHelper(loggers, wandb)

    # Offline RL
    for step in tqdm.tqdm(range(1, FLAGS.offline_steps + 1)):
        if FLAGS.cluster_sampler:
            batch = cluster_sampler.sample_sequence(
                train_dataset,
                config["batch_size"],
                FLAGS.horizon_length,
                FLAGS.discount,
                return_ids=True
            )
        else:
            batch = train_dataset.sample_sequence(
                config["batch_size"],
                FLAGS.horizon_length,
                FLAGS.discount,
            )

        if FLAGS.cluster_sampler and "cluster_ids" in batch:
            cids = np.asarray(batch["cluster_ids"])
            counts = np.bincount(cids, minlength=cluster_sampler.K)
            probs = counts / np.maximum(1, counts.sum())

            eps = 1e-12
            entropy = float(-(probs * np.log(probs + eps)).sum())
            kl_u = float(
                (probs * (np.log(probs + eps) - np.log(1.0 / cluster_sampler.K))).sum()
            )
            ess = float(1.0 / (np.sum(probs ** 2) + eps))

            wandb.log(
                {
                    "cluster/offline/prob_entropy": entropy,
                    "cluster/offline/kl_to_uniform": kl_u,
                    "cluster/offline/ess": ess,
                    "cluster/offline/count_min": int(counts.min()),
                    "cluster/offline/count_max": int(counts.max()),
                },
                step=step,
            )

            for k in range(cluster_sampler.K):
                wandb.log(
                    {f"cluster/offline/prob_{k}": float(probs[k])},
                    step=step,
                )

        agent, info = agent.update(batch)

        if step % FLAGS.log_interval == 0:
            logger.log(info, "offline_agent", step)

        if step % FLAGS.eval_interval == 0 or step == FLAGS.offline_steps:
            eval_info, _, _ = evaluate(
                agent=agent,
                env=eval_env,
                action_dim=act_example.shape[-1],
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            logger.log(eval_info, "eval", step)

    # Online RL
    d = dict(train_dataset)
    if "is_success" not in d:
        d["is_success"] = np.zeros_like(d["terminals"], dtype=np.float32)
    for k, v in d.items():
        a = np.asarray(v)
        print(k, "dtype", a.dtype, "ndim", a.ndim, "shape", getattr(a, "shape", None))
        if a.ndim == 0:
            print("  >>> [BAD] scalar detected:", k, a)

    buffer_size = max(FLAGS.buffer_size, len(d["rewards"]) + 1)
    replay_buffer = ReplayBuffer.create_from_initial_dataset(d, size=buffer_size)

    ob, _ = env.reset()
    action_queue = []
    current_traj = []

    online_episode_count = 0
    online_success_count = 0

    save_states = FLAGS.save_all_online_states
    if save_states:
        from collections import defaultdict
        state_log = defaultdict(list)
        online_start_time = time.time()

    for step in tqdm.tqdm(range(1, FLAGS.online_steps + 1)):
        global_step = FLAGS.offline_steps + step
        online_rng, key = jax.random.split(online_rng)

        if not action_queue:
            a = agent.sample_actions(observations=ob, rng=key)
            chunk = np.array(a).reshape(-1, act_example.shape[-1])
            action_queue = list(chunk)

        action = action_queue.pop(0)

        next_ob, r, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        if is_robomimic_env(FLAGS.env_name):
            r = r - 1.0

        if FLAGS.sparse:
            assert r <= 0.0
            r = -1.0 if r != 0.0 else 0.0

        env_info = {k: v for k, v in info.items() if k.startswith("distance")}
        logger.log(env_info, "env", step=global_step)

        if save_states:
            state = env.get_state()
            state_log["steps"].append(step)
            state_log["obs"].append(np.copy(next_ob))
            state_log["qpos"].append(np.copy(state["qpos"]))
            state_log["qvel"].append(np.copy(state["qvel"]))
            if "button_states" in state:
                state_log["button_states"].append(np.copy(state["button_states"]))

        if "success" in info and isinstance(info["success"], (int, float, bool, np.number)):
            is_success = float(info["success"])
        elif "is_success" in info:
            s = info["is_success"]
            if isinstance(s, dict):
                is_success = float(any(bool(v) for v in s.values()))
            else:
                is_success = float(s)
        else:
            is_success = 0.0

        transition = dict(
            observations=ob,
            actions=action,
            rewards=r,
            terminals=float(done),
            masks=1.0 - float(terminated),
            next_observations=next_ob,
            is_success=is_success,
        )

        replay_buffer.add_transition(transition)
        current_traj.append(transition)
        ob = next_ob

        if done:
            online_episode_count += 1

            if is_success > 0.5:
                online_success_count += 1

            if FLAGS.use_ptr_online_priority and ptr_sampler is not None:
                ep_end = (replay_buffer.pointer - 1) % replay_buffer.max_size

                if ep_end >= online_ep_start:
                    ptr_sampler.trajectory_boundaries.append((online_ep_start, ep_end))

                online_ep_start = replay_buffer.pointer

                ptr_sampler.rewards_source = replay_buffer["rewards"]
                if hasattr(ptr_sampler, "success_source"):
                    ptr_sampler.success_source = replay_buffer["is_success"]

                ptr_sampler._compute_basic_stats()
                ptr_sampler.compute_priorities()

            current_traj = []
            ob, _ = env.reset()
            action_queue = []

        if step >= FLAGS.start_training:
            use_ptr_now = (
                FLAGS.use_ptr_backward
                and ptr_sampler is not None
                and step >= FLAGS.ptr_warmup_steps
            )

            if use_ptr_now:
                batch = sample_sequence_PTR(
                    replay_buffer,
                    ptr_sampler,
                    config["batch_size"] * FLAGS.utd_ratio,
                    FLAGS.horizon_length,
                    FLAGS.discount,
                    backward=FLAGS.backward,
                    log_prefix="ptr/sample_online",
                    global_step=global_step,
                )
            else:
                batch = replay_buffer.sample_sequence(
                    config["batch_size"] * FLAGS.utd_ratio,
                    FLAGS.horizon_length,
                    FLAGS.discount,
                )

            batch = jax.tree.map(
                lambda x: x.reshape((FLAGS.utd_ratio, config["batch_size"]) + x.shape[1:]),
                batch,
            )
            agent, info = agent.batch_update(batch)

            if (
                use_ptr_now
                and FLAGS.use_ptr_online_priority
                and ptr_sampler is not None
                and FLAGS.metric == "td_error_rank"
                and "traj_ids" in batch
            ):
                td_err = np.asarray(info["critic/td_error_per_sample"])
                traj_ids = np.asarray(batch["traj_ids"])

                if td_err.ndim == 3:
                    td_err = np.max(np.abs(td_err), axis=-1)
                else:
                    td_err = np.abs(td_err)

                ptr_sampler.update_td_error_from_batch(
                    traj_ids=traj_ids.reshape(-1),
                    td_errors=td_err.reshape(-1),
                    ema_beta=0.9,
                )
                ptr_sampler.compute_priorities()

            logger.log(info, "online_agent", step=global_step)

        if step % FLAGS.eval_interval == 0:
            eval_info, _, _ = evaluate(
                agent=agent,
                env=eval_env,
                action_dim=act_example.shape[-1],
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            logger.log(eval_info, "eval", global_step)

        if FLAGS.save_interval > 0 and step % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, global_step)

    if save_states:
        end_time = time.time()
        c_data = {
            "steps": np.array(state_log["steps"]),
            "qpos": np.stack(state_log["qpos"], axis=0),
            "qvel": np.stack(state_log["qvel"], axis=0),
            "obs": np.stack(state_log["obs"], axis=0),
            "online_time": end_time - online_start_time,
        }
        if len(state_log["button_states"]) != 0:
            c_data["button_states"] = np.stack(state_log["button_states"], axis=0)
        np.savez(os.path.join(FLAGS.save_dir, "data.npz"), **c_data)

    for csv_logger in logger.csv_loggers.values():
        csv_logger.close()

    with open(os.path.join(FLAGS.save_dir, "token.tk"), "w") as f:
        f.write(run.url)


if __name__ == "__main__":
    app.run(main)