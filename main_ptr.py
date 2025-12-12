# main_ptr.py
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


FLAGS = flags.FLAGS

# ============================================================================
# FLAGS
# ============================================================================
flags.DEFINE_string("run_group", "Debug", "Run group.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("env_name", "cube-triple-play-singletask-task2-v0", "Environment.")
flags.DEFINE_string("save_dir", "exp/", "Save directory.")

flags.DEFINE_integer("offline_steps", 1000000, "Offline RL steps.")
flags.DEFINE_integer("online_steps", 1000000, "Online RL steps.")
flags.DEFINE_integer("buffer_size", 2000000, "Replay buffer size.")
flags.DEFINE_integer("log_interval", 5000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 100000, "Evaluation interval.")
flags.DEFINE_integer("save_interval", -1, "Save interval.")
flags.DEFINE_integer("start_training", 5000, "When online training begins.")

flags.DEFINE_integer("utd_ratio", 1, "Update-to-data ratio.")
flags.DEFINE_float("discount", 0.99, "Discount factor.")

flags.DEFINE_integer("eval_episodes", 50, "Evaluation episodes.")
flags.DEFINE_integer("video_episodes", 0, "Video episodes.")
flags.DEFINE_integer("video_frame_skip", 3, "Video frame skip.")

config_flags.DEFINE_config_file(
    "agent", "agents/acfql.py", lock_config=False
)

flags.DEFINE_float("dataset_proportion", 1.0, "Percentage of dataset to load.")
flags.DEFINE_integer(
    "dataset_replace_interval",
    1000,
    "Large dataset rotation interval (OGBench).",
)
flags.DEFINE_string("ogbench_dataset_dir", None, "OGBench dataset directory.")

flags.DEFINE_integer("horizon_length", 5, "Action chunk horizon.")
flags.DEFINE_bool("sparse", False, "Sparse reward flag.")

# PTR
flags.DEFINE_bool("use_ptr_backward", True, "Use PTR trajectory-priority sampling.")
flags.DEFINE_bool("use_ptr_online_priority", True, "Update PTR priorities online.")

flags.DEFINE_bool("save_all_online_states", False, "Save trajectory states.")

flags.DEFINE_bool("backward", True, "PTR backward sampling.")
flags.DEFINE_float("beta", 0.5, "Beta for PTR sarsa target critic weighted target.")
flags.DEFINE_bool("use_weighted_target", False, "Use PTR sarsa target critic weighted target.")
flags.DEFINE_string("metric", "success_binary", "PTR priority metric.")



# ============================================================================
# Logging helper
# ============================================================================

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


# ============================================================================
# PTR sequence sampling
# ============================================================================

# def sample_sequence_PTR(dataset, ptr_sampler, batch_size, sequence_length, discount):
#     """
#     dataset: Dataset or ReplayBuffer (둘 다 Dataset 상속)
#     ptr_sampler: PriorityTrajectorySampler
#     """
#     traj_ids = ptr_sampler.sample_trajectory_indices(batch_size)
#     boundaries = ptr_sampler.trajectory_boundaries

#     start_idxs = np.empty(batch_size, dtype=np.int64)
#     max_start_allowed = dataset.size - sequence_length

#     for i, tid in enumerate(traj_ids):
#         start, end = boundaries[tid]
#         # 시퀀스 길이 고려한 max_start
#         max_start = min(end, max_start_allowed)
#         if max_start < start:
#             s = start  # traj가 매우 짧거나 끝에 붙어 있으면 그냥 시작에서 패딩
#         else:
#             s = np.random.randint(start, max_start + 1)
#         start_idxs[i] = s

#     # 여기서부터는 기존 sample_sequence와 완전 동일한 로직 사용
#     return dataset.sample_sequence_from_start_idxs(
#         start_idxs, sequence_length, discount
#     )

def sample_sequence_PTR(dataset, ptr_sampler, batch_size, sequence_length, discount, backward=True):
    traj_ids = ptr_sampler.sample_trajectory_indices(batch_size)
    boundaries = ptr_sampler.trajectory_boundaries
    
    start_idxs = np.empty(batch_size, dtype=np.int64)
    
    for i, tid in enumerate(traj_ids):
        start, end = boundaries[tid]
        traj_len = end - start + 1
        
        if backward:
            # PTR: Always sample from END
            if traj_len >= sequence_length:
                s = end - sequence_length + 1  # Start from END!
            else:
                s = start
        else:
            # Random start (baseline)
            max_start = min(end - sequence_length + 1, dataset.size - sequence_length)
            s = np.random.randint(start, max_start + 1) if max_start >= start else start
        
        start_idxs[i] = s
    
    return dataset.sample_sequence_from_start_idxs(
        start_idxs, sequence_length, discount
    )




# ============================================================================
# MAIN
# ============================================================================

def main(_):
    exp_name = get_exp_name(FLAGS.seed)
    run = setup_wandb(project="qc", group=FLAGS.run_group, name=exp_name)

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

    # ----------------------------
    # Env & dataset
    # ----------------------------
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
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
            FLAGS.env_name
        )

    # ----------------------------
    # RNG & seeding
    # ----------------------------
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    online_rng, _ = jax.random.split(jax.random.PRNGKey(FLAGS.seed))

    # ----------------------------
    # Process offline dataset
    # ----------------------------
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

    # ----------------------------
    # Build ReplayBuffer (offline + online)
    # ----------------------------
    replay_buffer = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset),
        size=max(FLAGS.buffer_size, train_dataset.size + 1),
    )

    # ----------------------------
    # PTR initialization
    # ----------------------------
    terminal_locs = np.nonzero(train_dataset["terminals"] > 0)[0]
    initial_locs = np.concatenate([[0], terminal_locs[:-1] + 1])
    trajectory_boundaries = list(zip(initial_locs, terminal_locs))

    ptr_sampler = None

    #TODO: different PRIORITY_METRICS --- IGNORE ---

    PRIORITY_METRICS = [
        "uniform",          # Baseline (no priority)
        "success_binary",   # Current (simple)
        "avg_reward",       # PTR baseline
        "uqm_reward",       # PTR best (Table 3)
        "uhm_reward",       # PTR alternative
        "min_reward",       # PTR defensive
    ]

    if FLAGS.use_ptr_backward:
        ptr_sampler = PriorityTrajectorySampler(
            trajectory_boundaries=trajectory_boundaries,
            rewards_source=train_dataset["rewards"],
            metric=FLAGS.metric,
            temperature=1.0,
        )

    # ----------------------------
    # Agent initialization
    # ----------------------------
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

    # ----------------------------
    # Logging setup
    # ----------------------------
    prefixes = ["eval", "env"]
    if FLAGS.offline_steps > 0:
        prefixes.append("offline_agent")
    if FLAGS.online_steps > 0:
        prefixes.append("online_agent")

    loggers = {
        p: CsvLogger(os.path.join(FLAGS.save_dir, f"{p}.csv")) for p in prefixes
    }
    logger = LoggingHelper(loggers, wandb)

    # ============================================================================
    # OFFLINE RL
    # ============================================================================
    for step in tqdm.tqdm(range(1, FLAGS.offline_steps + 1)):
        # if FLAGS.use_ptr_backward and ptr_sampler:
        #     batch = sample_sequence_PTR(
        #         train_dataset, ptr_sampler,
        #         config["batch_size"],
        #         FLAGS.horizon_length,
        #         FLAGS.discount,
        #         backward=True,
        #     )
        # else:
        batch = train_dataset.sample_sequence(
                config["batch_size"],
                FLAGS.horizon_length,
                FLAGS.discount,
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

    # ============================================================================
    # ONLINE RL
    # ============================================================================
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

        # ----------------------------
        # Action chunking
        # ----------------------------
        if not action_queue:
            a = agent.sample_actions(observations=ob, rng=key)
            chunk = np.array(a).reshape(-1, act_example.shape[-1])
            action_queue = list(chunk)

        action = action_queue.pop(0)

        # ----------------------------
        # Env step
        # ----------------------------
        next_ob, r, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        if is_robomimic_env(FLAGS.env_name):
            r = r - 1.0

        if FLAGS.sparse:
            assert r <= 0.0
            r = -1.0 if r != 0.0 else 0.0

        # log distances, etc.
        env_info = {
            k: v for k, v in info.items() if k.startswith("distance")
        }
        logger.log(env_info, "env", step=global_step)

        # Save raw states if requested
        if save_states:
            state = env.get_state()
            state_log["steps"].append(step)
            state_log["obs"].append(np.copy(next_ob))
            state_log["qpos"].append(np.copy(state["qpos"]))
            state_log["qvel"].append(np.copy(state["qvel"]))
            if "button_states" in state:
                state_log["button_states"].append(np.copy(state["button_states"]))

        transition = dict(
            observations=ob,
            actions=action,
            rewards=r,
            terminals=float(done),
            masks=1.0 - float(terminated),
            next_observations=next_ob,
        )

        replay_buffer.add_transition(transition)
        current_traj.append(transition)
        ob = next_ob

        # ----------------------------
        # End of episode
        # ----------------------------
        if done:
            online_episode_count += 1
            success = any(t["rewards"] >= -0.5 for t in current_traj)
            if success:
                online_success_count += 1

            if FLAGS.use_ptr_online_priority and ptr_sampler is not None:
                N = replay_buffer.size
                rewards_source = replay_buffer["rewards"][:N]
                new_terminal_locs = np.nonzero(
                    replay_buffer["terminals"][:N] > 0
                )[0]
                new_initial_locs = np.concatenate(
                    [[0], new_terminal_locs[:-1] + 1]
                )
                new_boundaries = list(zip(new_initial_locs, new_terminal_locs))
                ptr_sampler.update_online(
                    rewards_source=rewards_source,
                    trajectory_boundaries=new_boundaries,
                )

            current_traj = []
            ob, _ = env.reset()
            action_queue = []

        # ----------------------------
        # Online training
        # ----------------------------
        if step >= FLAGS.start_training:
            if FLAGS.use_ptr_backward and ptr_sampler:
                batch = sample_sequence_PTR(
                    replay_buffer, ptr_sampler,
                    config["batch_size"] * FLAGS.utd_ratio,
                    FLAGS.horizon_length,
                    FLAGS.discount,
                    backward=FLAGS.backward,
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

            logger.log(info, "online_agent", step=global_step)

        # ----------------------------
        # Periodic evaluation
        # ----------------------------
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

        # ----------------------------
        # Periodic saving
        # ----------------------------
        if FLAGS.save_interval > 0 and step % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, global_step)

    # ============================================================================
    # Finalization
    # ============================================================================
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
            c_data["button_states"] = np.stack(
                state_log["button_states"], axis=0
            )
        np.savez(os.path.join(FLAGS.save_dir, "data.npz"), **c_data)

    for csv_logger in logger.csv_loggers.values():
        csv_logger.close()

    with open(os.path.join(FLAGS.save_dir, "token.tk"), "w") as f:
        f.write(run.url)


if __name__ == "__main__":
    app.run(main)
