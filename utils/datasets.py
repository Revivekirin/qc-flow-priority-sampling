# utils/datasets.py
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the size (length) of the dataset along time dimension."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


# ----------------------------
# Image augmentation helpers
# ----------------------------

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


# ----------------------------
# Dataset
# ----------------------------

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

    # ----------------------------
    # Basic sampling
    # ----------------------------

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

    # ----------------------------
    # Sequence sampling (uniform)
    # ----------------------------

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
        flat_idxs = all_idxs.reshape(-1)  # (B*H,)

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

        # Reorganize observation tensors:
        # - For visual: (B, H, W, H_len, C) (time = second last axis)
        # - For state: (B, H, obs_dim)
        if obs_all.ndim == 5:
            # (B, T, H, W, C) -> (B, H, W, T, C)
            full_obs = obs_all.transpose(0, 2, 3, 1, 4)
            full_next_obs = next_obs_all.transpose(0, 2, 3, 1, 4)
        else:
            # (B, T, obs_dim)
            full_obs = obs_all
            full_next_obs = next_obs_all

        # "observations" key: current-time step obs (t=0)
        # 원래 코드가 data['observations'].copy()를 써서
        # (B, obs_dim) 만 넘겼으니 그 형식을 유지
        data_obs = self["observations"][start_idxs].copy()

        return dict(
            observations=data_obs,            # (B, obs_dim)
            full_observations=full_obs,      # sequence
            actions=act_all,                 # (B, H, act_dim)
            masks=masks,                     # (B, H)
            rewards=rewards,                 # (B, H)
            terminals=terminals,             # (B, H)
            valid=valid,                     # (B, H)
            next_observations=full_next_obs, # sequence
            next_actions=next_act_all,       # (B, H, act_dim)
        )

    def sample_sequence(self, batch_size, sequence_length, discount):
        # 기존처럼 uniform random start index 선택
        idxs = np.random.randint(self.size - sequence_length + 1, size=batch_size)
        return self.sample_sequence_from_start_idxs(idxs, sequence_length, discount)

    def sample_sequence_from_start_idxs(self, start_idxs, sequence_length, discount):
        """
        주어진 start index 배열(start_idxs)에 대해
        기존 sample_sequence와 동일한 포맷의 batch를 만든다.
        """
        start_idxs = np.asarray(start_idxs)
        batch_size = start_idxs.shape[0]

        # (batch_size,) → 기존 sample_sequence와 동일하게 'data' 구성
        data = {k: v[start_idxs] for k, v in self.items()}

        # (batch_size, sequence_length) 의 전역 인덱스
        all_idxs = start_idxs[:, None] + np.arange(sequence_length)[None, :]
        max_idx = self.size - 1
        all_idxs = np.clip(all_idxs, 0, max_idx)
        flat_idxs = all_idxs.reshape(-1)

        # 이하 내용은 기존 sample_sequence 로직과 동일하게 맞춘다
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

        # next_actions도 동일하게 구성
        next_action_idxs = np.minimum(flat_idxs + 1, max_idx)
        batch_next_actions = act[next_action_idxs].reshape(
            batch_size, sequence_length, *act.shape[1:]
        )

        # 누적 reward / mask / terminal / valid 계산
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

        # obs shape 정리 (이미지 vs state)
        if len(batch_observations.shape) == 5:
            # (B, H, H, W, C) → (B, H_img, W_img, H, C)
            full_observations = batch_observations.transpose(0, 2, 3, 1, 4)
            full_next_observations = batch_next_observations.transpose(0, 2, 3, 1, 4)
        else:
            full_observations = batch_observations
            full_next_observations = batch_next_observations

        return dict(
            observations=data["observations"].copy(),  # (B, obs_dim)
            full_observations=full_observations,       # (B, H, obs_dim) or image형식
            actions=batch_actions,                     # (B, H, act_dim)
            masks=masks,
            rewards=rewards,
            terminals=terminals,
            valid=valid,
            next_observations=full_next_observations,
            next_actions=batch_next_actions,
        )



# ----------------------------
# ReplayBuffer
# ----------------------------

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
        self.size = max(self.size, self.pointer)

    def clear(self):
        """Clear the replay buffer."""
        self.size = 0
        self.pointer = 0


# ----------------------------
# History utility (unchanged)
# ----------------------------

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


# ----------------------------
# PriorityTrajectorySampler
# ----------------------------

class PriorityTrajectorySampler:
    """
    Trajectory-level priority sampler. It NEVER stores transitions itself,
    only uses:
      - trajectory_boundaries (list of (start, end))
      - rewards_source: 1D or 2D rewards array
    and returns start indices for sequences.
    """

    def __init__(
        self,
        trajectory_boundaries,
        rewards_source,
        metric="success_binary",
        temperature=1.0,
    ):
        """
        trajectory_boundaries : list[(start_idx, end_idx)]
        rewards_source        : numpy array of shape (N,) or (N,1) from dataset/replay_buffer["rewards"]
        """
        self.trajectory_boundaries = trajectory_boundaries
        self.rewards_source = np.asarray(rewards_source)
        self.metric = metric
        self.temperature = temperature

        self.priorities = None
        self.success_flags = None
        self.rewards_per_traj = None

        self._compute_basic_stats()

    def _compute_basic_stats(self):
        """Precompute reward arrays for each trajectory."""
        self.success_flags = []
        self.rewards_per_traj = []

        for start, end in self.trajectory_boundaries:
            rewards = self.rewards_source[start : end + 1]
            # squeeze last dim if (N,1)
            rewards = np.asarray(rewards).squeeze()

            is_success = np.any(rewards >= -0.5)
            self.success_flags.append(is_success)
            self.rewards_per_traj.append(rewards.copy())

    def compute_priorities(self):
        """PTR 논문 Section 5.2 기반 metrics"""
        n = len(self.trajectory_boundaries)
        scores = np.zeros(n, dtype=float)
        
        for i, rewards in enumerate(self.rewards_per_traj):
            rewards = np.atleast_1d(rewards)

            # ===== Quality-based (sparse reward 최적) =====
            if self.metric == "success_binary":
                scores[i] = 2.0 if self.success_flags[i] else 0.1
            
            elif self.metric == "avg_reward":
                # PTR baseline
                scores[i] = float(np.mean(rewards))
            
            elif self.metric == "uqm_reward":
                # PTR best for sparse (Table 3)
                sorted_r = np.sort(rewards)
                q75_idx = int(len(sorted_r) * 0.75)
                scores[i] = float(np.mean(sorted_r[q75_idx:]))  # Top 25%
            
            elif self.metric == "uhm_reward":
                # Upper Half Mean
                sorted_r = np.sort(rewards)
                q50_idx = len(sorted_r) // 2
                scores[i] = float(np.mean(sorted_r[q50_idx:]))
            
            elif self.metric == "min_reward":
                # Filter out low-quality trajectories
                scores[i] = float(np.min(rewards))
            
            # ===== Uniform baseline =====
            elif self.metric == "uniform":
                scores[i] = 1.0
            
            else:
                raise ValueError(f"Unknown metric: {self.metric}")
       

        if self.metric == "uniform":
            self.priorities = np.ones_like(scores) / len(scores)
            return

        if not np.isfinite(scores).all() or np.allclose(scores, scores[0]):
            self.priorities = np.ones_like(scores) / len(scores)
            return

        # rank 기반 priority 계산
        #   - score가 클수록 좋은 trajectory라고 가정
        #   - 가장 큰 score → rank 1 (최상위)
        order = np.argsort(-scores)                # 내림차순 정렬된 index
        ranks = np.empty_like(order, dtype=float)  # ranks[i] = 그 trajectory의 순위 (1=best)
        ranks[order] = np.arange(1, n + 1)

        # temperature를 PTR의 alpha처럼 사용 (alpha > 0)
        alpha = 1.0 / max(self.temperature, 1e-6)  # 온도 해석을 반대로 쓰고 싶으면 여기 조정

        # rank가 낮을수록(=좋을수록) priority가 크도록
        # p_i ∝ 1 / rank_i^alpha
        p = 1.0 / (ranks ** alpha)

        # 5) normalize
        p_sum = p.sum()
        if p_sum <= 0 or not np.isfinite(p_sum):
            self.priorities = np.ones_like(p) / len(p)
        else:
            self.priorities = p / p_sum
        
        # # Handle negative scores (important for sparse reward!)
        # if np.min(scores) < 0:
        #     scores = scores - np.min(scores) + 1e-6
        
        # # Temperature scaling
        # if self.temperature != 1.0:
        #     scores = scores ** (1.0 / self.temperature)
        
        # # Normalize
        # scores_sum = scores.sum()
        # if scores_sum <= 0:
        #     self.priorities = np.ones_like(scores) / len(scores)
        # else:
            # self.priorities = scores / scores_sum

    def sample_trajectory_indices(self, batch_size):
        """(optional) sample just trajectory indices."""
        if self.priorities is None:
            self.compute_priorities()

        return np.random.choice(
            len(self.trajectory_boundaries),
            size=batch_size,
            p=self.priorities,
        )

    def sample_start_indices(self, batch_size, sequence_length):
        """
        Sample start indices for sequences of length H, using
        trajectory-level priorities.
        """
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

    def update_online(self, rewards_source, trajectory_boundaries):
        """
        Call this after replay_buffer has grown during online RL.
        """
        self.rewards_source = np.asarray(rewards_source)
        self.trajectory_boundaries = trajectory_boundaries
        self._compute_basic_stats()
        self.compute_priorities()
