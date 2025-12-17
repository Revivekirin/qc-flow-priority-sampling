from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the size (length) of the dataset along time dimension."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


# ============================================================================
# Image Augmentation Helper
# ============================================================================
@partial(jax.jit, static_argnames=("padding",))
def random_crop(img, crop_from, padding):
    """Randomly crop an image with padding."""
    padded_img = jnp.pad(
        img,
        ((padding, padding), (padding, padding), (0, 0)),
        mode="edge",
    )
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=("padding",))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


# ============================================================================
# Dataset
# ============================================================================
class Dataset(FrozenDict):
    """Static dataset (offline)."""

    @classmethod
    def create(cls, freeze=True, **fields):
        """
        Create a dataset from numpy arrays.

        Keys must at least include 'observations', 'actions', 'rewards',
        'terminals', 'masks', 'next_observations'.
        """
        data = fields
        assert "observations" in data
        if freeze:
            # Make underlying numpy arrays read-only to catch accidental writes.
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)

        # options configured externally
        self.frame_stack = None
        self.p_aug = None
        self.return_next_actions = False

        # episode boundaries (for frame stacking etc.)
        self.terminal_locs = np.nonzero(self["terminals"] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    # Basic sampling
    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices from [0, self.size)."""
        return np.random.randint(self.size, size=num_idxs)

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if self.return_next_actions:
            # WARNING: incorrect at very end-of-trajectory, use with care.
            result["next_actions"] = self._dict["actions"][
                np.minimum(idxs + 1, self.size - 1)
            ]
        return result

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate(
            [crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1
        )
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding))
                if len(arr.shape) == 4
                else arr,
                batch[key],
            )

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of single-step transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)

        # Optional frame stacking
        if self.frame_stack is not None:
            initial_state_idxs = self.initial_locs[
                np.searchsorted(self.initial_locs, idxs, side="right") - 1
            ]
            obs = []
            next_obs = []
            for i in reversed(range(self.frame_stack)):
                cur_idxs = np.maximum(idxs - i, initial_state_idxs)
                obs.append(
                    jax.tree_util.tree_map(
                        lambda arr: arr[cur_idxs], self["observations"]
                    )
                )
                if i != self.frame_stack - 1:
                    next_obs.append(
                        jax.tree_util.tree_map(
                            lambda arr: arr[cur_idxs], self["observations"]
                        )
                    )
            next_obs.append(
                jax.tree_util.tree_map(
                    lambda arr: arr[idxs], self["next_observations"]
                )
            )

            batch["observations"] = jax.tree_util.tree_map(
                lambda *args: np.concatenate(args, axis=-1), *obs
            )
            batch["next_observations"] = jax.tree_util.tree_map(
                lambda *args: np.concatenate(args, axis=-1), *next_obs
            )

        # Optional augmentation
        if self.p_aug is not None and np.random.rand() < self.p_aug:
            self.augment(batch, ["observations", "next_observations"])

        return batch

    def _sample_sequence_core(self, start_idxs, sequence_length, discount):
        """
        Shared implementation used by:
          - sample_sequence (uniform)
          - sample_sequence_given_start_indices (PTR)

        start_idxs: np.ndarray of shape (B,)
        """
        start_idxs = np.asarray(start_idxs)
        assert start_idxs.ndim == 1
        batch_size = start_idxs.shape[0]

        # Build all indices: (B, H)
        all_idxs = start_idxs[:, None] + np.arange(sequence_length)[None, :]
        all_idxs = np.clip(all_idxs, 0, self.size - 1)
        flat_idxs = all_idxs.reshape(-1)  

        # Fetch raw arrays
        obs_all = self["observations"][flat_idxs].reshape(
            batch_size, sequence_length, *self["observations"].shape[1:]
        )
        next_obs_all = self["next_observations"][flat_idxs].reshape(
            batch_size, sequence_length, *self["next_observations"].shape[1:]
        )
        act_all = self["actions"][flat_idxs].reshape(
            batch_size, sequence_length, *self["actions"].shape[1:]
        )
        rew_all = self["rewards"][flat_idxs].reshape(
            batch_size, sequence_length, *self["rewards"].shape[1:]
        )
        mask_all = self["masks"][flat_idxs].reshape(
            batch_size, sequence_length, *self["masks"].shape[1:]
        )
        term_all = self["terminals"][flat_idxs].reshape(
            batch_size, sequence_length, *self["terminals"].shape[1:]
        )

        # next_actions: one-step shifted
        next_flat_idxs = np.minimum(flat_idxs + 1, self.size - 1)
        next_act_all = self["actions"][next_flat_idxs].reshape(
            batch_size, sequence_length, *self["actions"].shape[1:]
        )

        # cumulative discounted rewards, masks, terminals, valid
        rewards = np.zeros((batch_size, sequence_length), dtype=float)
        masks = np.ones((batch_size, sequence_length), dtype=float)
        terminals = np.zeros((batch_size, sequence_length), dtype=float)
        valid = np.ones((batch_size, sequence_length), dtype=float)

        rewards[:, 0] = rew_all[:, 0].squeeze()
        masks[:, 0] = mask_all[:, 0].squeeze()
        terminals[:, 0] = term_all[:, 0].squeeze()

        discount_powers = discount ** np.arange(sequence_length)
        for i in range(1, sequence_length):
            rewards[:, i] = (
                rewards[:, i - 1]
                + rew_all[:, i].squeeze() * discount_powers[i]
            )
            masks[:, i] = np.minimum(masks[:, i - 1], mask_all[:, i].squeeze())
            terminals[:, i] = np.maximum(
                terminals[:, i - 1], term_all[:, i].squeeze()
            )
            valid[:, i] = 1.0 - terminals[:, i - 1]

        if obs_all.ndim == 5:
            # (B, T, H, W, C) -> (B, H, W, T, C)
            full_obs = obs_all.transpose(0, 2, 3, 1, 4)
            full_next_obs = next_obs_all.transpose(0, 2, 3, 1, 4)
        else:
            full_obs = obs_all
            full_next_obs = next_obs_all

        data_obs = self["observations"][start_idxs].copy()

        return dict(
            observations=data_obs,            
            full_observations=full_obs,      
            actions=act_all,                 
            masks=masks,                     
            rewards=rewards,                 
            terminals=terminals,             
            valid=valid,                     
            next_observations=full_next_obs, 
            next_actions=next_act_all,       
        )

    def sample_sequence(self, batch_size, sequence_length, discount):
        idxs = np.random.randint(self.size - sequence_length + 1, size=batch_size)
        return self.sample_sequence_from_start_idxs(idxs, sequence_length, discount)

    def sample_sequence_from_start_idxs(self, start_idxs, sequence_length, discount):
        start_idxs = np.asarray(start_idxs)
        batch_size = start_idxs.shape[0]

        data = {k: v[start_idxs] for k, v in self.items()}

        all_idxs = start_idxs[:, None] + np.arange(sequence_length)[None, :]
        max_idx = self.size - 1
        all_idxs = np.clip(all_idxs, 0, max_idx)
        flat_idxs = all_idxs.reshape(-1)

        obs = self["observations"]
        nxt_obs = self["next_observations"]
        act = self["actions"]
        rew = self["rewards"]
        msk = self["masks"]
        term = self["terminals"]

        # (B, H, ...)
        batch_observations = obs[flat_idxs].reshape(
            batch_size, sequence_length, *obs.shape[1:]
        )
        batch_next_observations = nxt_obs[flat_idxs].reshape(
            batch_size, sequence_length, *nxt_obs.shape[1:]
        )
        batch_actions = act[flat_idxs].reshape(
            batch_size, sequence_length, *act.shape[1:]
        )
        batch_rewards = rew[flat_idxs].reshape(
            batch_size, sequence_length, *rew.shape[1:]
        )
        batch_masks = msk[flat_idxs].reshape(
            batch_size, sequence_length, *msk.shape[1:]
        )
        batch_terminals = term[flat_idxs].reshape(
            batch_size, sequence_length, *term.shape[1:]
        )

        next_action_idxs = np.minimum(flat_idxs + 1, max_idx)
        batch_next_actions = act[next_action_idxs].reshape(
            batch_size, sequence_length, *act.shape[1:]
        )
        rewards = np.zeros((batch_size, sequence_length), dtype=float)
        masks = np.ones((batch_size, sequence_length), dtype=float)
        terminals = np.zeros((batch_size, sequence_length), dtype=float)
        valid = np.ones((batch_size, sequence_length), dtype=float)

        rewards[:, 0] = batch_rewards[:, 0].squeeze()
        masks[:, 0] = batch_masks[:, 0].squeeze()
        terminals[:, 0] = batch_terminals[:, 0].squeeze()

        discount_powers = discount ** np.arange(sequence_length)
        for i in range(1, sequence_length):
            rewards[:, i] = (
                rewards[:, i - 1]
                + batch_rewards[:, i].squeeze() * discount_powers[i]
            )
            masks[:, i] = np.minimum(
                masks[:, i - 1], batch_masks[:, i].squeeze()
            )
            terminals[:, i] = np.maximum(
                terminals[:, i - 1], batch_terminals[:, i].squeeze()
            )
            valid[:, i] = 1.0 - terminals[:, i - 1]

        if len(batch_observations.shape) == 5: #image
            # (B, H, H, W, C) → (B, H_img, W_img, H, C)
            full_observations = batch_observations.transpose(0, 2, 3, 1, 4)
            full_next_observations = batch_next_observations.transpose(0, 2, 3, 1, 4)
        else: #state
            full_observations = batch_observations
            full_next_observations = batch_next_observations

        return dict(
            observations=data["observations"].copy(), 
            full_observations=full_observations,      
            actions=batch_actions,                     
            masks=masks,
            rewards=rewards,
            terminals=terminals,
            valid=valid,
            next_observations=full_next_observations,
            next_actions=batch_next_actions,
        )



# ============================================================================
# ReplayBuffer
# ============================================================================
class ReplayBuffer(Dataset):
    """Replay buffer that extends Dataset with add() / clear()."""

    @classmethod
    def create(cls, transition, size):
        """
        Create empty replay buffer given an example transition dict.

        transition: dict of single-step numpy arrays.
        size: max buffer size.
        """
        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """
        Create replay buffer and pre-fill with an initial offline dataset.
        """
        def create_buffer(init_buffer):
            init_buffer = np.asarray(init_buffer)
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_size = get_size(self._dict)
        # will be overwritten by create_from_initial_dataset
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """Add a single transition to the replay buffer."""
        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        # self.size = max(self.size, self.pointer)
        self.size = min(self.max_size, self.size + 1)


    def clear(self):
        """Clear the replay buffer."""
        self.size = 0
        self.pointer = 0


# ============================================================================
# History Utility
# ============================================================================
def add_history(dataset, history_length):
    size = dataset.size
    (terminal_locs,) = np.nonzero(dataset["terminals"] > 0)
    initial_locs = np.concatenate([[0], terminal_locs[:-1] + 1])
    assert terminal_locs[-1] == size - 1

    idxs = np.arange(size)
    initial_state_idxs = initial_locs[
        np.searchsorted(initial_locs, idxs, side="right") - 1
    ]
    obs_rets = []
    acts_rets = []
    for i in reversed(range(1, history_length)):
        cur_idxs = np.maximum(idxs - i, initial_state_idxs)
        outside = (idxs - i < initial_state_idxs)[..., None]
        obs_rets.append(
            jax.tree_util.tree_map(
                lambda arr: arr[cur_idxs] * (~outside)
                + jnp.zeros_like(arr[cur_idxs]) * outside,
                dataset["observations"],
            )
        )
        acts_rets.append(
            jax.tree_util.tree_map(
                lambda arr: arr[cur_idxs] * (~outside)
                + jnp.zeros_like(arr[cur_idxs]) * outside,
                dataset["actions"],
            )
        )

    observation_history, action_history = jax.tree_util.tree_map(
        lambda *args: np.stack(args, axis=-2), *obs_rets
    ), jax.tree_util.tree_map(
        lambda *args: np.stack(args, axis=-2), *acts_rets
    )

    dataset = Dataset(
        dataset.copy(
            dict(
                observation_history=observation_history,
                action_history=action_history,
            )
        )
    )
    return dataset

# ============================================================================
# PriorityTrajectorySampler
# ============================================================================
class PriorityTrajectorySampler:
    """
    Trajectory-level priority sampler with:
    - reward-based metrics
    - success-based metrics
    - TD-error rank-based priority
    - eps-uniform mixing for stability
    """

    def __init__(
        self,
        trajectory_boundaries,
        rewards_source,
        metric="success_binary",
        success_source=None,   
        temperature=1.0,
        alpha_rank=0.7,      # rank exponent (lower → more uniform)
        eps_uniform=0.05,    # exploration mixing
    ):
        self.trajectory_boundaries = trajectory_boundaries
        self.rewards_source = np.asarray(rewards_source)
        self.metric = metric
        self.temperature = temperature

        self.alpha_rank = alpha_rank
        self.eps_uniform = eps_uniform

        self.priorities = None
        self.success_flags = None
        self.rewards_per_traj = None

        # TD-error statistics
        self.td_error_mean = None  
        self.num_offline_traj = 0
        self.rewards_source = np.asarray(rewards_source)
        self.success_source = None if success_source is None else np.asarray(success_source)


        # Build basic statistics (success flags, rewards per traj)
        self._compute_basic_stats()

    def normalize_td_errors(self, td_abs):
        """
        td_abs: np.ndarray (N,)  abs(td_error)
        returns: normalized abs(td_error)
        """
        td_abs = np.asarray(td_abs, dtype=float)

        m = td_abs.mean()
        v = ((td_abs - m) ** 2).mean()

        b = self.td_scale_beta
        self.td_scale_mean = b * self.td_scale_mean + (1 - b) * m
        self.td_scale_var  = b * self.td_scale_var  + (1 - b) * v

        std = np.sqrt(max(self.td_scale_var, 0.0)) + self.td_scale_eps
        return td_abs / std
    
    # Build reward per trajectory + success flags
    def _compute_basic_stats(self):
        self.success_flags = []
        self.rewards_per_traj = []

        for start, end in self.trajectory_boundaries:
            r = self.rewards_source[start:end+1]
            self.rewards_per_traj.append(r)

            if self.success_source is not None:
                s = self.success_source[start:end+1]
                self.success_flags.append(np.any(s > 0.5))
            else:
                self.success_flags.append(np.any(r >= -0.5))  # fallback for robomimic


        # for start, end in self.trajectory_boundaries:
        #     r = self.rewards_source[start:end+1]
        #     self.rewards_per_traj.append(r)
        #     self.success_flags.append(np.any(r >= -0.5))

        self.success_flags = np.asarray(self.success_flags, dtype=bool)

        # Maintain existing TD-error
        n_traj = len(self.trajectory_boundaries)

        if self.td_error_mean is None:
            # First initialization
            self.td_error_mean = np.zeros(n_traj, dtype=float)
        else:
            # Extend if #traj increased
            old_len = len(self.td_error_mean)
            if n_traj > old_len:
                new_arr = np.zeros(n_traj, dtype=float)
                new_arr[:old_len] = self.td_error_mean
                self.td_error_mean = new_arr
            # If equal, keep as is

    # Update TD-error (EMA Tracking)
    def update_td_error_from_batch(self, traj_ids, td_errors, ema_beta=0.9):
        traj_ids = np.asarray(traj_ids, dtype=int)
        td_errors = np.asarray(td_errors, dtype=float)

        # Expand if needed
        if self.td_error_mean is None:
            self.td_error_mean = np.zeros(len(self.trajectory_boundaries), dtype=float)

        for tid, delta in zip(traj_ids, td_errors):
            old = self.td_error_mean[tid]
            if old == 0.0:
                new_val = float(delta)
            else:
                new_val = ema_beta * old + (1.0 - ema_beta) * abs(delta)
            self.td_error_mean[tid] = new_val

    # score -> rank-based priority
    def _scores_to_rank_based_priorities(self, scores):
        n = len(scores)

        # All same → uniform
        if np.all(scores == scores[0]):
            return np.ones(n) / n

        # large score → rank 0
        order = np.argsort(-scores)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(n)

        # w_i = 1 / rank^alpha
        w = 1.0 / ((ranks + 1) ** self.alpha_rank)
        w_sum = w.sum()
        base_p = w / w_sum

        # eps-uniform mixture
        if self.eps_uniform > 0:
            uniform = np.ones(n) / n
            p = (1 - self.eps_uniform) * base_p + self.eps_uniform * uniform
        else:
            p = base_p

        return p
    


    # compute trajectory-level priority distribution
    def compute_priorities(self, log_to_wandb=False):
        n = len(self.trajectory_boundaries)
        scores = np.zeros(n, dtype=float)

        # compute score per trajectory
        for i, rewards in enumerate(self.rewards_per_traj):
            rewards = np.atleast_1d(rewards)

            if self.metric == "success_binary":
                scores[i] = 2.0 if self.success_flags[i] else 0.1

            elif self.metric == "avg_reward":
                scores[i] = float(np.mean(rewards))

            elif self.metric == "uqm_reward":   # Upper quartile mean
                sorted_r = np.sort(rewards)
                q75 = int(len(sorted_r) * 0.75)
                scores[i] = float(np.mean(sorted_r[q75:]))

            elif self.metric == "uhm_reward":   # Upper half mean
                sorted_r = np.sort(rewards)
                mid = len(sorted_r) // 2
                scores[i] = float(np.mean(sorted_r[mid:]))

            elif self.metric == "min_reward":
                scores[i] = float(np.min(rewards))

            elif self.metric == "uniform":
                scores[i] = 1.0   # 어차피 나중에 uniform 처리

            elif self.metric == "td_error_rank":
                scores[i] = float(self.td_error_mean[i])

            else:
                raise ValueError(f"Unknown metric: {self.metric}")

        # ---------------------------
        # 2) uniform이면 그냥 균등 분포
        # ---------------------------
        if self.metric == "uniform":
            priorities = np.ones(n) / n
            self.priorities = priorities
            return {"priorities": priorities}

        # ---------------------------
        # 3) reward metric일 때: 음수 shift + temperature
        #    (td_error_rank는 건드리지 X)
        # ---------------------------
        if self.metric != "td_error_rank":
            if np.min(scores) < 0:
                scores = scores - np.min(scores) + 1e-6

            if self.temperature != 1.0:
                # temperature는 순위 보존하는 monotonic transform
                scores = scores ** (1.0 / self.temperature)

        # ---------------------------
        # 4) 최종 priority = rank 기반 (공통)
        # ---------------------------
        priorities = self._scores_to_rank_based_priorities(scores)
        self.priorities = priorities

        return {"priorities": priorities}


    def sample_trajectory_indices(self, batch_size):
        if self.priorities is None:
            self.compute_priorities()

        return np.random.choice(
            len(self.trajectory_boundaries),
            size=batch_size,
            p=self.priorities,
        )

    def sample_start_indices(self, batch_size, sequence_length):
        if self.priorities is None:
            self.compute_priorities()

        traj_ids = np.random.choice(
            len(self.trajectory_boundaries),
            size=batch_size,
            p=self.priorities,
        )

        start_idxs = np.zeros(batch_size, dtype=int)

        for i, tid in enumerate(traj_ids):
            start, end = self.trajectory_boundaries[tid]
            length = end - start + 1
            if length > sequence_length:
                s = np.random.randint(start, end - sequence_length + 2)
            else:
                s = start
            start_idxs[i] = s

        return start_idxs

    # ================================================================
    # Update sampler when online buffer grows
    # ================================================================
    def update_online(self, rewards_source, trajectory_boundaries, success_source=None):
        self.rewards_source = np.asarray(rewards_source)
        self.trajectory_boundaries = trajectory_boundaries

        if success_source is not None:
            self.success_source = np.asarray(success_source)

        self._compute_basic_stats()
        self.compute_priorities()



# def get_group_weights(step, T_curr):
#     # 0 ~ 1 사이 curriculum 진도
#     alpha = np.clip(step / T_curr, 0.0, 1.0)
    
#     # 초반 비율
#     w_easy0, w_med0, w_hard0 = 0.7, 0.2, 0.1
#     # 후반(완료) 비율
#     w_easy1, w_med1, w_hard1 = 1/3, 1/3, 1/3

#     w_easy  = (1-alpha) * w_easy0  + alpha * w_easy1
#     w_med   = (1-alpha) * w_med0   + alpha * w_med1
#     w_hard  = (1-alpha) * w_hard0  + alpha * w_hard1
#     return w_easy, w_med, w_hard


# def get_cluster_weights(step, T_curr, K,
#                         easy_clusters, medium_clusters, hard_clusters):
#     w_easy, w_med, w_hard = get_group_weights(step, T_curr)
#     w = np.zeros(K, dtype=np.float32)

#     if len(easy_clusters) > 0:
#         w[easy_clusters] = w_easy / len(easy_clusters)
#     if len(medium_clusters) > 0:
#         w[medium_clusters] = w_med / len(medium_clusters)
#     if len(hard_clusters) > 0:
#         w[hard_clusters] = w_hard / len(hard_clusters)

#     w /= w.sum()
#     return w



# class ClusterBalancedSampler:
#     def __init__(self,
#                  trajectory_boundaries,
#                  cluster_ids,
#                  priorities=None,
#                  easy_clusters=None,
#                  medium_clusters=None,
#                  hard_clusters=None,
#                  T_curr=2_00_000):
#         self.trajectory_boundaries = trajectory_boundaries
#         self.cluster_ids = np.asarray(cluster_ids, dtype=np.int32)
#         self.num_traj = len(trajectory_boundaries)
#         self.K = int(self.cluster_ids.max()) + 1

#         self.cluster_to_trajs = {cid: [] for cid in range(self.K)}
#         for tid, cid in enumerate(self.cluster_ids):
#             self.cluster_to_trajs[cid].append(tid)
#         self.cluster_to_trajs = {cid: np.array(tids, dtype=np.int32)
#                                  for cid, tids in self.cluster_to_trajs.items()}

#         # priority (e.g. avg_return, td-error rank 등)
#         if priorities is None:
#             priorities = np.ones(self.num_traj, dtype=np.float32)
#         self.priorities = np.asarray(priorities, dtype=np.float32)

#         # 난이도 그룹
#         self.easy_clusters = list(easy_clusters)
#         self.medium_clusters = list(medium_clusters)
#         self.hard_clusters = list(hard_clusters)
#         self.T_curr = T_curr

#     def get_cluster_weights(self, step):
#         return get_cluster_weights(
#             step, self.T_curr, self.K,
#             self.easy_clusters, self.medium_clusters, self.hard_clusters
#         )

#     def sample_trajectory_indices(self, batch_size, global_step):
#         # 1) cluster weights
#         w_cluster = self.get_cluster_weights(global_step)

#         # 2) 각 샘플마다 cluster 먼저 뽑기
#         chosen_clusters = np.random.choice(
#             self.K, size=batch_size, p=w_cluster
#         )

#         traj_idxs = np.empty(batch_size, dtype=np.int32)

#         for i, cid in enumerate(chosen_clusters):
#             trajs_c = self.cluster_to_trajs[cid]
#             if len(trajs_c) == 0:
#                 # 비어있으면 fallback: 전체 uniform
#                 traj_idxs[i] = np.random.randint(0, self.num_traj)
#                 continue

#             # --- cluster 내부 priority ---
#             pri_c = self.priorities[trajs_c]
#             if pri_c.sum() <= 0:
#                 p = None  # uniform
#             else:
#                 p = pri_c / pri_c.sum()
#             traj_idxs[i] = np.random.choice(trajs_c, p=p)

#         return traj_idxs
