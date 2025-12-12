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
from utils.datasets_logging import Dataset, ReplayBuffer, PriorityTrajectorySampler

from cluster_vis import build_traj_features
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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
flags.DEFINE_integer("save_interval", 100000, "Save interval.")
flags.DEFINE_integer("start_training", 5000, "When online training begins.")

flags.DEFINE_integer("utd_ratio", 1, "Update-to-data ratio.")
flags.DEFINE_float("discount", 0.99, "Discount factor.")

flags.DEFINE_integer("eval_episodes", 50, "Evaluation episodes.")
flags.DEFINE_integer("video_episodes", 0, "Video episodes.")
flags.DEFINE_integer("video_frame_skip", 3, "Video frame skip.")

config_flags.DEFINE_config_file(
    "agent", "agents/acfql_logging.py", lock_config=False
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

# Cluster sampler
flags.DEFINE_bool("cluster_sampler", False, "Cluster Sampler (kmeans)")
flags.DEFINE_bool("cluster_use_curriculum", True, "Use easy/medium/hard curriculum")
flags.DEFINE_bool("cluster_use_internal_priority", False, "Use priority within clusters")
flags.DEFINE_integer("cluster_curriculum_steps", 200000, "Steps to complete curriculum")
flags.DEFINE_integer("cluster_min_size", 10, "Minimum trajectories per cluster")

flags.DEFINE_integer("ptr_warmup_steps", 20000,
                     "Number of online steps before enabling PTR sampling.")

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
# K selection + easy/medium/hard auto-split
# ============================================================================

def select_optimal_K(X_scaled, possible_K, min_cluster_size=10):
    """
    K를 자동으로 선택: 모든 클러스터의 최소 크기를 최대화
    
    Args:
        X_scaled: 정규화된 feature matrix (n_traj, n_features)
        possible_K: 시도할 K 값들의 리스트
        min_cluster_size: 최소 허용 cluster size
    
    Returns:
        best_K: 선택된 K
        best_kmeans: 해당 K의 KMeans 모델
        cluster_ids: 클러스터 할당
    """
    best_K = None
    best_min_size = -1
    best_kmeans = None
    best_cluster_ids = None
    
    print("\n=== K Selection ===")
    for K in possible_K:
        kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)
        cluster_ids = kmeans.fit_predict(X_scaled)
        
        counts = np.bincount(cluster_ids, minlength=K)
        min_size = counts.min()
        max_size = counts.max()
        avg_size = counts.mean()
        
        print(f"K={K}: min={min_size}, max={max_size}, avg={avg_size:.1f}, std={counts.std():.1f}")
        
        # 조건: min_size가 threshold 이상이면서, 가장 큰 min_size
        if min_size >= min_cluster_size and min_size > best_min_size:
            best_min_size = min_size
            best_K = K
            best_kmeans = kmeans
            best_cluster_ids = cluster_ids
    
    # fallback: 아무것도 조건을 만족 안 하면 가장 큰 min_size
    if best_K is None:
        for K in possible_K:
            kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)
            cluster_ids = kmeans.fit_predict(X_scaled)
            counts = np.bincount(cluster_ids, minlength=K)
            min_size = counts.min()
            
            if min_size > best_min_size:
                best_min_size = min_size
                best_K = K
                best_kmeans = kmeans
                best_cluster_ids = cluster_ids
    
    print(f"\n✓ Selected K={best_K} with min_cluster_size={best_min_size}\n")
    return best_K, best_kmeans, best_cluster_ids


def split_clusters_by_difficulty(cluster_stats):
    """
    클러스터를 return mean 기준으로 easy/medium/hard로 자동 분할
    
    Args:
        cluster_stats: list of dict with keys ['cid', 'n', 'len_mean', 'ret_mean']
    
    Returns:
        easy_clusters: list of cluster IDs (high return)
        medium_clusters: list of cluster IDs (medium return)
        hard_clusters: list of cluster IDs (low return)
    """
    if len(cluster_stats) == 0:
        return [], [], []
    
    ret_means = np.array([cs["ret_mean"] for cs in cluster_stats])
    cids = np.array([cs["cid"] for cs in cluster_stats])
    K = len(cluster_stats)
    
    # K가 3 이하면 단순 분할
    if K <= 3:
        sorted_idx = np.argsort(ret_means)
        if K == 1:
            easy_clusters = cids.tolist()
            medium_clusters = []
            hard_clusters = []
        elif K == 2:
            hard_clusters = [cids[sorted_idx[0]]]
            medium_clusters = []
            easy_clusters = [cids[sorted_idx[1]]]
        else:  # K == 3
            hard_clusters = [cids[sorted_idx[0]]]
            medium_clusters = [cids[sorted_idx[1]]]
            easy_clusters = [cids[sorted_idx[2]]]
    else:
        # quantile로 3등분
        q33 = np.quantile(ret_means, 1/3)
        q67 = np.quantile(ret_means, 2/3)
        
        hard_mask = ret_means < q33
        medium_mask = (ret_means >= q33) & (ret_means < q67)
        easy_mask = ret_means >= q67
        
        hard_clusters = cids[hard_mask].tolist()
        medium_clusters = cids[medium_mask].tolist()
        easy_clusters = cids[easy_mask].tolist()
    
    print("\n=== Difficulty Groups ===")
    print(f"Easy clusters (high return, n={len(easy_clusters)}): {easy_clusters}")
    if len(easy_clusters) > 0:
        easy_rets = ret_means[[cids.tolist().index(c) for c in easy_clusters]]
        print(f"  Return range: [{easy_rets.min():.3f}, {easy_rets.max():.3f}]")
    
    print(f"Medium clusters (mid return, n={len(medium_clusters)}): {medium_clusters}")
    if len(medium_clusters) > 0:
        med_rets = ret_means[[cids.tolist().index(c) for c in medium_clusters]]
        print(f"  Return range: [{med_rets.min():.3f}, {med_rets.max():.3f}]")
    
    print(f"Hard clusters (low return, n={len(hard_clusters)}): {hard_clusters}")
    if len(hard_clusters) > 0:
        hard_rets = ret_means[[cids.tolist().index(c) for c in hard_clusters]]
        print(f"  Return range: [{hard_rets.min():.3f}, {hard_rets.max():.3f}]")
    print()
    
    return easy_clusters, medium_clusters, hard_clusters


# ============================================================================
# ClusterBalancedSampler (통합 버전)
# ============================================================================

class ClusterBalancedSampler:
    """
    통합된 Cluster-Balanced Sampler:
    - 클러스터 간 균등 or curriculum (easy/medium/hard)
    - 클러스터 내부 uniform or priority
    """
    def __init__(
        self,
        cluster_ids,
        trajectory_boundaries,
        priorities=None,
        use_curriculum=False,
        easy_clusters=None,
        medium_clusters=None,
        hard_clusters=None,
        T_curr=200000,
        use_internal_priority=False,
        priority_temp_schedule=True,
    ):
        """
        Args:
            cluster_ids: (num_traj,) cluster assignment
            trajectory_boundaries: list of (start, end)
            priorities: (num_traj,) priority scores (optional)
            use_curriculum: curriculum learning with easy/medium/hard
            easy/medium/hard_clusters: cluster IDs for each difficulty
            T_curr: curriculum total steps
            use_internal_priority: use priority within clusters
            priority_temp_schedule: gradually increase priority influence
        """
        self.cluster_ids = np.array(cluster_ids)
        self.boundaries = trajectory_boundaries
        self.num_traj = len(trajectory_boundaries)
        
        # cluster → trajectory mapping
        self.K = int(self.cluster_ids.max()) + 1
        self.cluster_to_trajs = {cid: [] for cid in range(self.K)}
        for tid, cid in enumerate(cluster_ids):
            self.cluster_to_trajs[cid].append(tid)
        self.cluster_to_trajs = {
            cid: np.array(tids, dtype=np.int32)
            for cid, tids in self.cluster_to_trajs.items()
        }
        
        # curriculum settings
        self.use_curriculum = use_curriculum
        self.easy_clusters = easy_clusters if easy_clusters else []
        self.medium_clusters = medium_clusters if medium_clusters else []
        self.hard_clusters = hard_clusters if hard_clusters else []
        self.T_curr = T_curr
        
        # priority settings
        self.use_internal_priority = use_internal_priority
        self.priority_temp_schedule = priority_temp_schedule
        if priorities is None:
            priorities = np.ones(self.num_traj, dtype=np.float32)
        self.priorities = np.asarray(priorities, dtype=np.float32)

    def get_cluster_weights(self, step):
        """클러스터별 샘플링 비율 계산"""
        if not self.use_curriculum:
            # 균등 분배
            w = np.ones(self.K, dtype=np.float32) / self.K
            return w
        
        # curriculum: 초반에는 easy 많이, 후반에는 균등
        alpha = np.clip(step / self.T_curr, 0.0, 1.0)
        
        w_easy0, w_med0, w_hard0 = 0.7, 0.2, 0.1
        w_easy1, w_med1, w_hard1 = 1/3, 1/3, 1/3
        
        w_easy = (1 - alpha) * w_easy0 + alpha * w_easy1
        w_med = (1 - alpha) * w_med0 + alpha * w_med1
        w_hard = (1 - alpha) * w_hard0 + alpha * w_hard1
        
        w = np.zeros(self.K, dtype=np.float32)
        
        # 각 그룹에 weight 할당
        n_groups = 0
        if len(self.easy_clusters) > 0:
            w[self.easy_clusters] = w_easy / len(self.easy_clusters)
            n_groups += 1
        if len(self.medium_clusters) > 0:
            w[self.medium_clusters] = w_med / len(self.medium_clusters)
            n_groups += 1
        if len(self.hard_clusters) > 0:
            w[self.hard_clusters] = w_hard / len(self.hard_clusters)
            n_groups += 1
        
        # 빈 그룹이 있으면 그룹들에 재분배
        if n_groups < 3:
            # 최소한 하나의 그룹은 있어야 함
            if n_groups == 0:
                # fallback: 모든 클러스터 균등
                w = np.ones(self.K, dtype=np.float32) / self.K
            else:
                # 정규화만
                w /= w.sum()
        else:
            # 정규화
            w /= w.sum()
        
        return w

    def sample_traj_ids(self, batch_size, global_step=0):
        """trajectory ID 샘플링"""
        w_cluster = self.get_cluster_weights(global_step)
        chosen_clusters = np.random.choice(self.K, size=batch_size, p=w_cluster)
        
        traj_ids = np.empty(batch_size, dtype=np.int32)
        
        # priority temperature (초기에는 거의 uniform)
        if self.use_internal_priority and self.priority_temp_schedule:
            alpha = np.clip(global_step / self.T_curr, 0.0, 1.0)
            priority_temp = 0.5 + 0.5 * alpha  # 0.5 -> 1.0
        else:
            priority_temp = 1.0
        
        for i, cid in enumerate(chosen_clusters):
            trajs_c = self.cluster_to_trajs[cid]
            if len(trajs_c) == 0:
                traj_ids[i] = np.random.randint(0, self.num_traj)
                continue
            
            if not self.use_internal_priority:
                # uniform within cluster
                traj_ids[i] = np.random.choice(trajs_c)
            else:
                # priority within cluster
                pri_c = self.priorities[trajs_c]
                if pri_c.sum() <= 0:
                    p = None
                else:
                    # temperature scaling
                    pri_c_t = pri_c ** priority_temp
                    p = pri_c_t / pri_c_t.sum()
                traj_ids[i] = np.random.choice(trajs_c, p=p)
        
        return traj_ids

    def sample_sequence(self, dataset, batch_size, seq_len, discount, global_step=0):
        """sequence 샘플링"""
        traj_ids = self.sample_traj_ids(batch_size, global_step)
        starts = []
        
        for tid in traj_ids:
            s, e = self.boundaries[tid]
            max_s = e - seq_len + 1
            if max_s < s:
                starts.append(s)
            else:
                starts.append(np.random.randint(s, max_s + 1))
        
        return dataset.sample_sequence_from_start_idxs(starts, seq_len, discount)


# ============================================================================
# PTR sequence sampling
# ============================================================================

def sample_sequence_PTR(
    dataset,
    ptr_sampler,
    batch_size,
    sequence_length,
    discount,
    backward=True,
    log_prefix=None,
    global_step=None,
):
    traj_ids = ptr_sampler.sample_trajectory_indices(batch_size)
    boundaries = ptr_sampler.trajectory_boundaries
    
    start_idxs = np.empty(batch_size, dtype=np.int64)

    # 통계 수집용 리스트
    rel_starts = []
    traj_success_flags = []
    seg_returns = []
    last_rewards = []
    sampled_priors = []

    for i, tid in enumerate(traj_ids):
        start, end = boundaries[tid]
        traj_len = end - start + 1
        if backward:
            if traj_len >= sequence_length:
                s = end - sequence_length + 1 
            else:
                s = start
        else:
            max_start = min(end - sequence_length + 1, dataset.size - sequence_length)
            s = np.random.randint(start, max_start + 1) if max_start >= start else start
        
        start_idxs[i] = s

        # ===== 로깅용 통계 =====
        if log_prefix is not None:
            denom = max(1, traj_len - sequence_length + 1)
            rel_pos = (s - start) / denom
            rel_starts.append(rel_pos)

            if hasattr(ptr_sampler, "success_flags"):
                traj_success_flags.append(float(ptr_sampler.success_flags[tid]))

            seg_r = dataset["rewards"][s : min(s + sequence_length, end + 1)]
            seg_returns.append(float(seg_r.sum()))
            last_rewards.append(float(seg_r[-1]))

            if ptr_sampler.priorities is not None:
                sampled_priors.append(float(ptr_sampler.priorities[tid]))

    # ====== 실제 batch 구성 ======
    batch = dataset.sample_sequence_from_start_idxs(
        start_idxs, sequence_length, discount
    )

    batch = dict(batch)
    batch["traj_ids"] = np.asarray(traj_ids, dtype=np.int32)

    # ====== wandb 로깅 ======
    if log_prefix is not None and global_step is not None and len(rel_starts) > 0:
        log_dict = {
            f"{log_prefix}/rel_start_mean": float(np.mean(rel_starts)),
            f"{log_prefix}/rel_start_std": float(np.std(rel_starts)),
            f"{log_prefix}/seg_return_mean": float(np.mean(seg_returns)),
            f"{log_prefix}/seg_return_std": float(np.std(seg_returns)),
            f"{log_prefix}/r_last_mean": float(np.mean(last_rewards)),
        }
        if len(traj_success_flags) > 0:
            log_dict[f"{log_prefix}/success_traj_ratio"] = float(
                np.mean(traj_success_flags)
            )
        if len(sampled_priors) > 0:
            log_dict[f"{log_prefix}/prior_mean"] = float(np.mean(sampled_priors))
            log_dict[f"{log_prefix}/prior_std"] = float(np.std(sampled_priors))

        wandb.log(log_dict, step=global_step)

    return batch


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
    # Trajectory boundaries & returns
    # ----------------------------
    (terminal_locs,) = np.nonzero(train_dataset["terminals"] > 0)
    initial_locs = np.concatenate([[0], terminal_locs[:-1] + 1])
    trajectory_boundaries = list(zip(initial_locs, terminal_locs))
    num_traj = len(trajectory_boundaries)

    traj_returns = []
    for (s, e) in trajectory_boundaries:
        r = train_dataset["rewards"][s : e + 1]
        traj_returns.append(float(r.sum()))
    traj_returns = np.asarray(traj_returns, dtype=np.float64)

    # ----------------------------
    # Clustering with auto K selection
    # ----------------------------
    cluster_sampler = None
    
    if FLAGS.cluster_sampler:
        print("\n" + "="*60)
        print("Building Cluster-Balanced Sampler")
        print("="*60)
        
        # 1) Build features
        features, lengths = build_traj_features(
            train_dataset,
            trajectory_boundaries,
            obs_key="observations",
            obs_slice=None,
            k_keyframes=20,
        )
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features).astype(np.float64)
        
        # 2) Auto K selection
        possible_K = [4, 5, 6, 7, 8, 9, 10]
        K, kmeans, cluster_ids = select_optimal_K(
            X_scaled,
            possible_K,
            min_cluster_size=FLAGS.cluster_min_size
        )
        
        # 3) Cluster statistics
        cluster_stats = []
        for cid in range(K):
            idxs = np.where(cluster_ids == cid)[0]
            if len(idxs) == 0:
                continue
            
            lengths_c = lengths[idxs]
            returns_c = traj_returns[idxs]
            cluster_stats.append({
                "cid": cid,
                "n": int(len(idxs)),
                "len_mean": float(lengths_c.mean()),
                "len_std": float(lengths_c.std()),
                "ret_mean": float(returns_c.mean()),
                "ret_std": float(returns_c.std()),
            })
        
        # Log cluster statistics
        print("\n=== Cluster Statistics ===")
        for cs in cluster_stats:
            print(f"Cluster {cs['cid']}: n={cs['n']}, "
                  f"len={cs['len_mean']:.1f}±{cs['len_std']:.1f}, "
                  f"ret={cs['ret_mean']:.3f}±{cs['ret_std']:.3f}")
        
        wandb.log({
            "cluster/K": K,
            "cluster/num_trajectories": num_traj,
        })
        
        # 4) Auto split easy/medium/hard
        easy_clusters, medium_clusters, hard_clusters = None, None, None
        if FLAGS.cluster_use_curriculum:
            easy_clusters, medium_clusters, hard_clusters = split_clusters_by_difficulty(
                cluster_stats
            )
            
            wandb.log({
                "cluster/n_easy": len(easy_clusters),
                "cluster/n_medium": len(medium_clusters),
                "cluster/n_hard": len(hard_clusters),
            })
        
        # 5) Create sampler
        cluster_sampler = ClusterBalancedSampler(
            cluster_ids=cluster_ids,
            trajectory_boundaries=trajectory_boundaries,
            priorities=None,  # uniform within cluster (기본값)
            use_curriculum=FLAGS.cluster_use_curriculum,
            easy_clusters=easy_clusters,
            medium_clusters=medium_clusters,
            hard_clusters=hard_clusters,
            T_curr=FLAGS.cluster_curriculum_steps,
            use_internal_priority=FLAGS.cluster_use_internal_priority,
            priority_temp_schedule=True,
        )
        
        print(f"\n✓ Cluster sampler created:")
        print(f"  - K={K}")
        print(f"  - Curriculum: {FLAGS.cluster_use_curriculum}")
        print(f"  - Internal priority: {FLAGS.cluster_use_internal_priority}")
        print("="*60 + "\n")

    # ----------------------------
    # PTR initialization
    # ----------------------------
    ptr_sampler = None
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
        if FLAGS.cluster_sampler and cluster_sampler is not None:
            batch = cluster_sampler.sample_sequence(
                train_dataset,
                config["batch_size"],
                FLAGS.horizon_length,
                FLAGS.discount,
                global_step=step,
            )
        else:
            batch = train_dataset.sample_sequence(
                config["batch_size"],
                FLAGS.horizon_length,
                FLAGS.discount,
            )

        agent, info = agent.update(batch)

        if step % FLAGS.log_interval == 0:
            logger.log(info, "offline_agent", step)
            
            # Log curriculum progress
            if FLAGS.cluster_sampler and FLAGS.cluster_use_curriculum and cluster_sampler is not None:
                w_cluster = cluster_sampler.get_cluster_weights(step)
                alpha = np.clip(step / FLAGS.cluster_curriculum_steps, 0.0, 1.0)
                
                wandb.log({
                    "cluster/curriculum_alpha": alpha,
                    "cluster/easy_weight": float(w_cluster[easy_clusters].sum()) if easy_clusters else 0,
                    "cluster/medium_weight": float(w_cluster[medium_clusters].sum()) if medium_clusters else 0,
                    "cluster/hard_weight": float(w_cluster[hard_clusters].sum()) if hard_clusters else 0,
                }, step=step)

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

        env_info = {
            k: v for k, v in info.items() if k.startswith("distance")
        }
        logger.log(env_info, "env", step=global_step)

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

            # TD-error rank priority 업데이트
            if (
                use_ptr_now                           
                and FLAGS.use_ptr_online_priority
                and ptr_sampler is not None
                and FLAGS.metric == "td_error_rank"
                and "traj_ids" in batch              
            ):
                td_err = np.asarray(info["critic/td_error_per_sample"])
                traj_ids = np.asarray(batch["traj_ids"])
                
                td_err_flat = td_err.reshape(-1)
                traj_ids_flat = traj_ids.reshape(-1)

                ptr_sampler.update_td_error_from_batch(
                    traj_ids=traj_ids_flat,
                    td_errors=td_err_flat,
                    ema_beta=0.9,
                )
            
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

# import glob
# import json
# import os
# import random
# import time

# import jax
# import jax.numpy as jnp
# import numpy as np
# import tqdm
# import wandb
# from absl import app, flags
# from ml_collections import config_flags

# from agents import agents
# from envs.env_utils import make_env_and_datasets
# from envs.ogbench_utils import make_ogbench_env_and_datasets
# from envs.robomimic_utils import is_robomimic_env
# from evaluation import evaluate
# from log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb
# from utils.flax_utils import save_agent
# from utils.datasets_logging import Dataset, ReplayBuffer, PriorityTrajectorySampler

# from cluster_vis import build_traj_features
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans

# FLAGS = flags.FLAGS

# # ============================================================================
# # FLAGS
# # ============================================================================
# flags.DEFINE_string("run_group", "Debug", "Run group.")
# flags.DEFINE_integer("seed", 0, "Random seed.")
# flags.DEFINE_string("env_name", "cube-triple-play-singletask-task2-v0", "Environment.")
# flags.DEFINE_string("save_dir", "exp/", "Save directory.")

# flags.DEFINE_integer("offline_steps", 1000000, "Offline RL steps.")
# flags.DEFINE_integer("online_steps", 1000000, "Online RL steps.")
# flags.DEFINE_integer("buffer_size", 2000000, "Replay buffer size.")
# flags.DEFINE_integer("log_interval", 5000, "Logging interval.")
# flags.DEFINE_integer("eval_interval", 100000, "Evaluation interval.")
# flags.DEFINE_integer("save_interval", 100000, "Save interval.")
# flags.DEFINE_integer("start_training", 5000, "When online training begins.")

# flags.DEFINE_integer("utd_ratio", 1, "Update-to-data ratio.")
# flags.DEFINE_float("discount", 0.99, "Discount factor.")

# flags.DEFINE_integer("eval_episodes", 50, "Evaluation episodes.")
# flags.DEFINE_integer("video_episodes", 0, "Video episodes.")
# flags.DEFINE_integer("video_frame_skip", 3, "Video frame skip.")

# config_flags.DEFINE_config_file(
#     "agent", "agents/acfql_logging.py", lock_config=False
# )

# flags.DEFINE_float("dataset_proportion", 1.0, "Percentage of dataset to load.")
# flags.DEFINE_integer(
#     "dataset_replace_interval",
#     1000,
#     "Large dataset rotation interval (OGBench).",
# )
# flags.DEFINE_string("ogbench_dataset_dir", None, "OGBench dataset directory.")

# flags.DEFINE_integer("horizon_length", 5, "Action chunk horizon.")
# flags.DEFINE_bool("sparse", False, "Sparse reward flag.")

# # PTR
# flags.DEFINE_bool("use_ptr_backward", True, "Use PTR trajectory-priority sampling.")
# flags.DEFINE_bool("use_ptr_online_priority", True, "Update PTR priorities online.")

# flags.DEFINE_bool("save_all_online_states", False, "Save trajectory states.")

# flags.DEFINE_bool("backward", True, "PTR backward sampling.")
# flags.DEFINE_float("beta", 0.5, "Beta for PTR sarsa target critic weighted target.")
# flags.DEFINE_bool("use_weighted_target", False, "Use PTR sarsa target critic weighted target.")
# flags.DEFINE_string("metric", "success_binary", "PTR priority metric.")

# flags.DEFINE_bool("cluster_sampler", False, "Cluster Sampler (kmeans)")
# flags.DEFINE_integer("ptr_warmup_steps", 20000,
#                      "Number of online steps before enabling PTR sampling.")

# # ============================================================================
# # Logging helper
# # ============================================================================

# class LoggingHelper:
#     def __init__(self, csv_loggers, wandb_logger):
#         self.csv_loggers = csv_loggers
#         self.wandb_logger = wandb_logger

#     def log(self, data, prefix, step, wandb_step=None):
#         if not data:
#             return
#         self.csv_loggers[prefix].log(data, step)
#         if wandb_step is None:
#             wandb_step = step
#         self.wandb_logger.log(
#             {f"{prefix}/{k}": v for k, v in data.items()},
#             step=wandb_step,
#         )


# # ============================================================================
# # PTR sequence sampling
# # ============================================================================

# # def sample_sequence_PTR(dataset, ptr_sampler, batch_size, sequence_length, discount):
# #     """
# #     dataset: Dataset or ReplayBuffer (둘 다 Dataset 상속)
# #     ptr_sampler: PriorityTrajectorySampler
# #     """
# #     traj_ids = ptr_sampler.sample_trajectory_indices(batch_size)
# #     boundaries = ptr_sampler.trajectory_boundaries

# #     start_idxs = np.empty(batch_size, dtype=np.int64)
# #     max_start_allowed = dataset.size - sequence_length

# #     for i, tid in enumerate(traj_ids):
# #         start, end = boundaries[tid]
# #         # 시퀀스 길이 고려한 max_start
# #         max_start = min(end, max_start_allowed)
# #         if max_start < start:
# #             s = start  # traj가 매우 짧거나 끝에 붙어 있으면 그냥 시작에서 패딩
# #         else:
# #             s = np.random.randint(start, max_start + 1)
# #         start_idxs[i] = s

# #     # 여기서부터는 기존 sample_sequence와 완전 동일한 로직 사용
# #     return dataset.sample_sequence_from_start_idxs(
# #         start_idxs, sequence_length, discount
# #     )

# class ClusterBalancedSampler:
#     def __init__(self, cluster_ids, trajectory_boundaries):
#         self.cluster_ids = np.array(cluster_ids)
#         self.boundaries = trajectory_boundaries
        
#         # cluster → trajectories list
#         self.clusters = {}
#         for cid in np.unique(cluster_ids):
#             idxs = np.where(cluster_ids == cid)[0]
#             self.clusters[cid] = idxs
    
#         self.K = len(self.clusters)

#     def sample_traj_ids(self, batch_size):
#         # 균등하게 cluster 배분
#         per_cluster = batch_size // self.K
#         remainder = batch_size % self.K

#         traj_ids = []
#         for cid in range(self.K):
#             idxs = self.clusters[cid]
#             chosen = np.random.choice(idxs, per_cluster, replace=True)
#             traj_ids.extend(chosen)

#         # remainder는 전체 cluster에서 랜덤하게 선택
#         flat_ids = np.concatenate(list(self.clusters.values()))
#         if remainder > 0:
#             extra = np.random.choice(flat_ids, remainder, replace=True)
#             traj_ids.extend(extra)

#         return np.array(traj_ids, dtype=np.int32)

#     def sample_sequence(self, dataset, batch_size, seq_len, discount):
#         traj_ids = self.sample_traj_ids(batch_size)
#         starts = []
#         for tid in traj_ids:
#             s, e = self.boundaries[tid]
#             max_s = e - seq_len + 1
#             if max_s < s:
#                 starts.append(s)
#             else:
#                 starts.append(np.random.randint(s, max_s + 1))

#         return dataset.sample_sequence_from_start_idxs(starts, seq_len, discount)


# def sample_sequence_PTR(
#     dataset,
#     ptr_sampler,
#     batch_size,
#     sequence_length,
#     discount,
#     backward=True,
#     log_prefix=None,
#     global_step=None,
# ):
#     traj_ids = ptr_sampler.sample_trajectory_indices(batch_size)
#     boundaries = ptr_sampler.trajectory_boundaries
    
#     start_idxs = np.empty(batch_size, dtype=np.int64)

#     # 통계 수집용 리스트
#     rel_starts = []
#     traj_success_flags = []
#     seg_returns = []
#     last_rewards = []
#     sampled_priors = []

#     for i, tid in enumerate(traj_ids):
#         start, end = boundaries[tid]
#         traj_len = end - start + 1
#         if backward:
#             if traj_len >= sequence_length:
#                 s = end - sequence_length + 1 
#             else:
#                 s = start
#         else:
#             max_start = min(end - sequence_length + 1, dataset.size - sequence_length)
#             s = np.random.randint(start, max_start + 1) if max_start >= start else start
        
#         start_idxs[i] = s

#         # ===== 로깅용 통계 =====
#         if log_prefix is not None:
#             denom = max(1, traj_len - sequence_length + 1)
#             rel_pos = (s - start) / denom
#             rel_starts.append(rel_pos)

#             if hasattr(ptr_sampler, "success_flags"):
#                 traj_success_flags.append(float(ptr_sampler.success_flags[tid]))

#             seg_r = dataset["rewards"][s : min(s + sequence_length, end + 1)]
#             seg_returns.append(float(seg_r.sum()))
#             last_rewards.append(float(seg_r[-1]))

#             if ptr_sampler.priorities is not None:
#                 sampled_priors.append(float(ptr_sampler.priorities[tid]))

#     # ====== 실제 batch 구성 ======
#     batch = dataset.sample_sequence_from_start_idxs(
#         start_idxs, sequence_length, discount
#     )

#     # dataset이 dict-like 이라고 가정
#     batch = dict(batch)
#     batch["traj_ids"] = np.asarray(traj_ids, dtype=np.int32)

#     # ====== wandb 로깅 ======
#     if log_prefix is not None and global_step is not None and len(rel_starts) > 0:
#         import wandb
#         log_dict = {
#             f"{log_prefix}/rel_start_mean": float(np.mean(rel_starts)),
#             f"{log_prefix}/rel_start_std": float(np.std(rel_starts)),
#             f"{log_prefix}/seg_return_mean": float(np.mean(seg_returns)),
#             f"{log_prefix}/seg_return_std": float(np.std(seg_returns)),
#             f"{log_prefix}/r_last_mean": float(np.mean(last_rewards)),
#         }
#         if len(traj_success_flags) > 0:
#             log_dict[f"{log_prefix}/success_traj_ratio"] = float(
#                 np.mean(traj_success_flags)
#             )
#         if len(sampled_priors) > 0:
#             log_dict[f"{log_prefix}/prior_mean"] = float(np.mean(sampled_priors))
#             log_dict[f"{log_prefix}/prior_std"] = float(np.std(sampled_priors))

#         wandb.log(log_dict, step=global_step)

#     return batch

# # ============================================================================
# # MAIN
# # ============================================================================

# def main(_):
#     exp_name = get_exp_name(FLAGS.seed)
#     run = setup_wandb(project="qc", group=FLAGS.run_group, name=exp_name)

#     FLAGS.save_dir = os.path.join(
#         FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name
#     )
#     os.makedirs(FLAGS.save_dir, exist_ok=True)

#     with open(os.path.join(FLAGS.save_dir, "flags.json"), "w") as f:
#         json.dump(get_flag_dict(), f)

#     config = FLAGS.agent
#     config["horizon_length"] = FLAGS.horizon_length
#     config["use_weighted_target"] = FLAGS.use_weighted_target 
#     config["beta"] = FLAGS.beta

#     # ----------------------------
#     # Env & dataset
#     # ----------------------------
#     if FLAGS.ogbench_dataset_dir:
#         dataset_paths = sorted(
#             f for f in glob.glob(f"{FLAGS.ogbench_dataset_dir}/*.npz")
#             if "-val.npz" not in f
#         )
#         env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
#             FLAGS.env_name,
#             dataset_path=dataset_paths[0],
#             compact_dataset=False,
#         )
#     else:
#         env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
#             FLAGS.env_name
#         )

#     # ----------------------------
#     # RNG & seeding
#     # ----------------------------
#     random.seed(FLAGS.seed)
#     np.random.seed(FLAGS.seed)
#     online_rng, _ = jax.random.split(jax.random.PRNGKey(FLAGS.seed))

#     # ----------------------------
#     # Process offline dataset
#     # ----------------------------
#     def process_train_dataset(ds):
#         ds = Dataset.create(**ds)

#         if FLAGS.dataset_proportion < 1.0:
#             newN = int(len(ds["masks"]) * FLAGS.dataset_proportion)
#             ds = Dataset.create(**{k: v[:newN] for k, v in ds.items()})

#         if is_robomimic_env(FLAGS.env_name):
#             rew = ds["rewards"] - 1.0
#             d2 = dict(ds)
#             d2["rewards"] = rew
#             ds = Dataset.create(**d2)

#         if FLAGS.sparse:
#             rew = (ds["rewards"] != 0.0) * -1.0
#             d2 = dict(ds)
#             d2["rewards"] = rew
#             ds = Dataset.create(**d2)

#         return ds

#     train_dataset = process_train_dataset(train_dataset)

#     # ----------------------------
#     # Build ReplayBuffer (offline + online)
#     # ----------------------------
#     replay_buffer = ReplayBuffer.create_from_initial_dataset(
#         dict(train_dataset),
#         size=max(FLAGS.buffer_size, train_dataset.size + 1),
#     )

#     # ----------------------------
#     # PTR initialization
#     # ----------------------------
#     # (1) trajectory boundary 계산
#     (terminal_locs,) = np.nonzero(train_dataset["terminals"] > 0)
#     initial_locs = np.concatenate([[0], terminal_locs[:-1] + 1])
#     trajectory_boundaries = list(zip(initial_locs, terminal_locs))

#     num_traj = len(trajectory_boundaries)

#     # (2) traj별 return 계산
#     traj_returns = []
#     for (s, e) in trajectory_boundaries:
#         # rewards[s:e+1] 합
#         r = train_dataset["rewards"][s : e + 1]
#         traj_returns.append(float(r.sum()))
#     traj_returns = np.asarray(traj_returns, dtype=np.float64)

#     # (3) traj feature + KMeans 클러스터링
#     features, lengths = build_traj_features(
#         train_dataset,
#         trajectory_boundaries,
#         obs_key="observations",
#         obs_slice=None,     # 또는 slice(0, 3)
#         k_keyframes=20,
#     )

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(features).astype(np.float64)

#     K = 7   # TODO: elbow plot으로 선택
#     kmeans = KMeans(n_clusters=K, random_state=0)
#     cluster_ids = kmeans.fit_predict(X_scaled)

#     # (4) cluster 통계 (길이/return 평균 등)
#     cluster_stats = []
#     for cid in range(K):
#         idxs = np.where(cluster_ids == cid)[0]
#         if len(idxs) == 0:
#             continue

#         lengths_c = lengths[idxs]
#         returns_c = traj_returns[idxs]
#         cluster_stats.append(
#             {
#                 "cid": cid,
#                 "n": int(len(idxs)),
#                 "len_mean": float(lengths_c.mean()),
#                 "len_std": float(lengths_c.std()),
#                 "ret_mean": float(returns_c.mean()),
#             }
#         )

#     # (5) PTR sampler 생성 (trajectory-level priority)
#     ptr_sampler = None
#     if FLAGS.use_ptr_backward:
#         ptr_sampler = PriorityTrajectorySampler(
#             trajectory_boundaries=trajectory_boundaries,
#             rewards_source=train_dataset["rewards"],
#             metric=FLAGS.metric,
#             temperature=1.0,
#         )

#     #TODO: different PRIORITY_METRICS --- IGNORE ---

#     PRIORITY_METRICS = [
#         "uniform",          # Baseline (no priority)
#         "success_binary",   # Current (simple)
#         "avg_reward",       # PTR baseline
#         "uqm_reward",       # PTR best (Table 3)
#         "uhm_reward",       # PTR alternative
#         "min_reward",       # PTR defensive
#     ]

#     # ----------------------------
#     # Agent initialization
#     # ----------------------------
#     example_batch = train_dataset.sample(1)
#     obs_example = example_batch["observations"]
#     act_example = example_batch["actions"]
#     agent_class = agents[config["agent_name"]]
#     agent = agent_class.create(
#         FLAGS.seed,
#         obs_example,
#         act_example,
#         config,
#     )

#     # ----------------------------
#     # Logging setup
#     # ----------------------------
#     prefixes = ["eval", "env"]
#     if FLAGS.offline_steps > 0:
#         prefixes.append("offline_agent")
#     if FLAGS.online_steps > 0:
#         prefixes.append("online_agent")

#     loggers = {
#         p: CsvLogger(os.path.join(FLAGS.save_dir, f"{p}.csv")) for p in prefixes
#     }
#     logger = LoggingHelper(loggers, wandb)

#     # Initialize Custersampler
#     cluster_sampler = ClusterBalancedSampler(
#         cluster_ids=cluster_ids,
#         trajectory_boundaries=trajectory_boundaries
#     )


#     # ============================================================================
#     # OFFLINE RL
#     # ============================================================================
#     for step in tqdm.tqdm(range(1, FLAGS.offline_steps + 1)):
#         # if FLAGS.use_ptr_backward and ptr_sampler:
#         #     batch = sample_sequence_PTR(
#         #         train_dataset, ptr_sampler,
#         #         config["batch_size"],
#         #         FLAGS.horizon_length,
#         #         FLAGS.discount,
#         #         backward=True,
#         #     )
#         # else:
#         if FLAGS.cluster_sampler:
#             batch = cluster_sampler.sample_sequence(
#                 train_dataset,
#                 config["batch_size"],
#                 FLAGS.horizon_length,
#                 FLAGS.discount,
#             )
#         else:
#             batch = train_dataset.sample_sequence(
#                     config["batch_size"],
#                     FLAGS.horizon_length,
#                     FLAGS.discount,
#                 )

#         agent, info = agent.update(batch)

#         if step % FLAGS.log_interval == 0:
#             logger.log(info, "offline_agent", step)

#         if step % FLAGS.eval_interval == 0 or step == FLAGS.offline_steps:
#             eval_info, _, _ = evaluate(
#                 agent=agent,
#                 env=eval_env,
#                 action_dim=act_example.shape[-1],
#                 num_eval_episodes=FLAGS.eval_episodes,
#                 num_video_episodes=FLAGS.video_episodes,
#                 video_frame_skip=FLAGS.video_frame_skip,
#             )
#             logger.log(eval_info, "eval", step)

#     # ============================================================================
#     # ONLINE RL
#     # ============================================================================
#     ob, _ = env.reset()
#     action_queue = []
#     current_traj = []

#     online_episode_count = 0
#     online_success_count = 0

#     save_states = FLAGS.save_all_online_states
#     if save_states:
#         from collections import defaultdict
#         state_log = defaultdict(list)
#         online_start_time = time.time()

#     for step in tqdm.tqdm(range(1, FLAGS.online_steps + 1)):
#         global_step = FLAGS.offline_steps + step
#         online_rng, key = jax.random.split(online_rng)

#         # ----------------------------
#         # Action chunking
#         # ----------------------------
#         if not action_queue:
#             a = agent.sample_actions(observations=ob, rng=key)
#             chunk = np.array(a).reshape(-1, act_example.shape[-1])
#             action_queue = list(chunk)

#         action = action_queue.pop(0)

#         # ----------------------------
#         # Env step
#         # ----------------------------
#         next_ob, r, terminated, truncated, info = env.step(action)
#         done = bool(terminated or truncated)

#         if is_robomimic_env(FLAGS.env_name):
#             r = r - 1.0

#         if FLAGS.sparse:
#             assert r <= 0.0
#             r = -1.0 if r != 0.0 else 0.0

#         # log distances, etc.
#         env_info = {
#             k: v for k, v in info.items() if k.startswith("distance")
#         }
#         logger.log(env_info, "env", step=global_step)

#         # Save raw states if requested
#         if save_states:
#             state = env.get_state()
#             state_log["steps"].append(step)
#             state_log["obs"].append(np.copy(next_ob))
#             state_log["qpos"].append(np.copy(state["qpos"]))
#             state_log["qvel"].append(np.copy(state["qvel"]))
#             if "button_states" in state:
#                 state_log["button_states"].append(np.copy(state["button_states"]))

#         transition = dict(
#             observations=ob,
#             actions=action,
#             rewards=r,
#             terminals=float(done),
#             masks=1.0 - float(terminated),
#             next_observations=next_ob,
#         )

#         replay_buffer.add_transition(transition)
#         current_traj.append(transition)
#         ob = next_ob

#         # ----------------------------
#         # End of episode
#         # ----------------------------
#         if done:
#             online_episode_count += 1
#             success = any(t["rewards"] >= -0.5 for t in current_traj)
#             if success:
#                 online_success_count += 1

#             if FLAGS.use_ptr_online_priority and ptr_sampler is not None:
#                 N = replay_buffer.size
#                 rewards_source = replay_buffer["rewards"][:N]
#                 new_terminal_locs = np.nonzero(
#                     replay_buffer["terminals"][:N] > 0
#                 )[0]
#                 new_initial_locs = np.concatenate(
#                     [[0], new_terminal_locs[:-1] + 1]
#                 )
#                 new_boundaries = list(zip(new_initial_locs, new_terminal_locs))
#                 ptr_sampler.update_online(
#                     rewards_source=rewards_source,
#                     trajectory_boundaries=new_boundaries,
#                 )

#             current_traj = []
#             ob, _ = env.reset()
#             action_queue = []

#         # ----------------------------
#         # Online training
#         # ----------------------------
#         if step >= FLAGS.start_training:
#             use_ptr_now = (
#                 FLAGS.use_ptr_backward
#                 and ptr_sampler is not None
#                 and step >= FLAGS.ptr_warmup_steps
#             )

#             if use_ptr_now:
#                 batch = sample_sequence_PTR(
#                     replay_buffer,
#                     ptr_sampler,
#                     config["batch_size"] * FLAGS.utd_ratio,
#                     FLAGS.horizon_length,
#                     FLAGS.discount,
#                     backward=FLAGS.backward,
#                     log_prefix="ptr/sample_online",
#                     global_step=global_step,
#                 )
#             else:
#                 batch = replay_buffer.sample_sequence(
#                     config["batch_size"] * FLAGS.utd_ratio,
#                     FLAGS.horizon_length,
#                     FLAGS.discount,
#                 )

#             batch = jax.tree.map(
#                 lambda x: x.reshape((FLAGS.utd_ratio, config["batch_size"]) + x.shape[1:]),
#                 batch,
#             )
#             agent, info = agent.batch_update(batch)

#             # ======== TD-error rank priority 업데이트 (online 전용) ========
#             if (
#                 use_ptr_now                           
#                 and FLAGS.use_ptr_online_priority
#                 and ptr_sampler is not None
#                 and FLAGS.metric == "td_error_rank"
#                 and "traj_ids" in batch              
#             ):
#                 # info에서 TD-error per sample 꺼내기 (utd_ratio, batch_size)
#                 td_err = np.asarray(info["critic/td_error_per_sample"])

#                 # batch에서 traj_ids 가져오기 (같은 shape 가정)
#                 traj_ids = np.asarray(batch["traj_ids"])

#                 # 둘을 1D로 펴서 업데이트
#                 td_err_flat = td_err.reshape(-1)
#                 traj_ids_flat = traj_ids.reshape(-1)

#                 ptr_sampler.update_td_error_from_batch(
#                     traj_ids=traj_ids_flat,
#                     td_errors=td_err_flat,
#                     ema_beta=0.9,
#                 )
#             logger.log(info, "online_agent", step=global_step)

#         # ----------------------------
#         # Periodic evaluation
#         # ----------------------------
#         if step % FLAGS.eval_interval == 0:
#             eval_info, _, _ = evaluate(
#                 agent=agent,
#                 env=eval_env,
#                 action_dim=act_example.shape[-1],
#                 num_eval_episodes=FLAGS.eval_episodes,
#                 num_video_episodes=FLAGS.video_episodes,
#                 video_frame_skip=FLAGS.video_frame_skip,
#             )
#             logger.log(eval_info, "eval", global_step)


#         # ----------------------------
#         # Periodic saving
#         # ----------------------------
#         if FLAGS.save_interval > 0 and step % FLAGS.save_interval == 0:
#             save_agent(agent, FLAGS.save_dir, global_step)

#     # ============================================================================
#     # Finalization
#     # ============================================================================
#     if save_states:
#         end_time = time.time()
#         c_data = {
#             "steps": np.array(state_log["steps"]),
#             "qpos": np.stack(state_log["qpos"], axis=0),
#             "qvel": np.stack(state_log["qvel"], axis=0),
#             "obs": np.stack(state_log["obs"], axis=0),
#             "online_time": end_time - online_start_time,
#         }
#         if len(state_log["button_states"]) != 0:
#             c_data["button_states"] = np.stack(
#                 state_log["button_states"], axis=0
#             )
#         np.savez(os.path.join(FLAGS.save_dir, "data.npz"), **c_data)

#     for csv_logger in logger.csv_loggers.values():
#         csv_logger.close()

#     with open(os.path.join(FLAGS.save_dir, "token.tk"), "w") as f:
#         f.write(run.url)


# if __name__ == "__main__":
#     app.run(main)
