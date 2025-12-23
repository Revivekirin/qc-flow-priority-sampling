"""
PARS-ACFQL: Action Chunking Flow Q-Learning with PARS
Extends ACFQL with reward scaling and infeasible action penalization
"""

from typing import Any, Dict, Tuple
import jax
import jax.numpy as jnp
import ml_collections
from agents.acfql import ACFQLAgent


class PARSACFQLAgent(ACFQLAgent):
    """PARS agent extending ACFQL.
    
    Key additions to ACFQL:
    1. Reward scaling (RS) with layer normalization (LN)
    2. Penalizing infeasible actions (PA)
    3. Compatible with action chunking and flow matching
    """

    def critic_loss(self, batch: Dict, grad_params: Any, rng: jax.random.PRNGKey) -> Tuple[jnp.ndarray, Dict]:
        """Compute PARS-ACFQL critic loss.
        
        L_Total = L_TD + α * L_PA
        
        where:
        - L_TD: ACFQL TD loss with reward scaling
        - L_PA: Penalty for infeasible actions
        """
        # ===== Prepare actions (handle action chunking) =====
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(
                batch["actions"],
                (batch["actions"].shape[0], -1),
            )
        else:
            batch_actions = batch["actions"][..., 0, :]

        batch_size, action_dim = batch_actions.shape

        # ===== Sample next actions from policy =====
        rng, sample_rng, infeasible_rng = jax.random.split(rng, 3)
        next_actions_policy = self.sample_actions(
            batch['next_observations'][..., -1, :],
            rng=sample_rng
        )
        next_qs_policy = self.network.select('target_critic')(
            batch['next_observations'][..., -1, :],
            actions=next_actions_policy
        )

        # ===== Weighted target (SARSA + Policy) =====
        use_weighted_target = self.config.get('use_weighted_target', False)
        
        q_policy = 0.0
        q_sarsa = 0.0
        q_next_sarsa_mean = 0.0

        if use_weighted_target:
            beta = self.config.get('beta', 0.5)

            if self.config["action_chunking"]:
                next_actions_traj = batch['next_actions'].reshape(
                    batch['next_actions'].shape[0], -1
                )
            else:
                next_actions_traj = batch['next_actions'][..., 0, :]

            next_qs_sarsa = self.network.select('target_critic')(
                batch['next_observations'][..., -1, :],
                actions=next_actions_traj,
            )

            if self.config['q_agg'] == 'min':
                q_policy = next_qs_policy.min(axis=0)
                q_sarsa = next_qs_sarsa.min(axis=0)
            else:
                q_policy = next_qs_policy.mean(axis=0)
                q_sarsa = next_qs_sarsa.mean(axis=0)

            next_q = (1.0 - beta) * q_sarsa + beta * q_policy
            q_next_sarsa_mean = next_qs_sarsa.mean()
        else:
            if self.config['q_agg'] == 'min':
                next_q = next_qs_policy.min(axis=0)
            else:
                next_q = next_qs_policy.mean(axis=0)

        # ===== TD Loss =====
        scaled_rewards = self.config['reward_scale'] * batch['rewards'][..., -1]
        gamma_H = self.config['discount'] ** self.config["horizon_length"]
        target_q = scaled_rewards + gamma_H * batch['masks'][..., -1] * next_q
        target_q = jax.lax.stop_gradient(target_q)

        # Current Q-values (ID actions)
        q = self.network.select('critic')(
            batch['observations'],
            actions=batch_actions,
            params=grad_params
        )
        # q shape: (num_critics, batch_size)

        td_error = q - target_q
        td_error_abs = jnp.abs(td_error)
        td_loss = (jnp.square(td_error) * batch['valid'][..., -1]).mean()

        # ===== PA Loss =====
        batch_size, action_dim = batch_actions.shape
        infeasible_actions = self._sample_infeasible_actions(
            infeasible_rng, batch_size, action_dim
        )
        
        infeasible_q = self.network.select('critic')(
            batch['observations'],
            actions=infeasible_actions,
            params=grad_params
        )
        # infeasible_q shape: (num_critics, batch_size)
        
        q_min = self.config['q_min']
        pa_loss = jnp.square(infeasible_q - q_min).mean()
        
        critic_loss = td_loss + self.config['alpha_pa'] * pa_loss

        # ===== Q Geometry Analysis =====
        
        # 1. ID Q-values (from dataset actions)
        q_id_per_critic = q  # (num_critics, batch_size)
        q_id_flat = q_id_per_critic.flatten()  # All Q values
        
        # 2. Policy Q-values (from learned policy)
        if use_weighted_target:
            q_policy_per_critic = next_qs_policy  # (num_critics, batch_size)
        else:
            q_policy_per_critic = next_qs_policy
        q_policy_flat = q_policy_per_critic.flatten()
        
        # 3. Infeasible Q-values
        q_inf_flat = infeasible_q.flatten()
        
        # ===== Percentile Analysis =====
        def compute_percentiles(values):
            """Compute key percentiles of Q-value distribution"""
            return {
                'min': jnp.min(values),
                'p01': jnp.percentile(values, 1),
                'p05': jnp.percentile(values, 5),
                'p10': jnp.percentile(values, 10),
                'p25': jnp.percentile(values, 25),
                'p50': jnp.percentile(values, 50),  # median
                'p75': jnp.percentile(values, 75),
                'p90': jnp.percentile(values, 90),
                'p95': jnp.percentile(values, 95),
                'p99': jnp.percentile(values, 99),
                'max': jnp.max(values),
                'mean': jnp.mean(values),
                'std': jnp.std(values),
            }
        
        q_id_percentiles = compute_percentiles(q_id_flat)
        q_policy_percentiles = compute_percentiles(q_policy_flat)
        q_inf_percentiles = compute_percentiles(q_inf_flat)
        
        # ===== Range and Overlap Analysis =====
        
        # Q-value ranges
        q_id_range = q_id_percentiles['max'] - q_id_percentiles['min']
        q_policy_range = q_policy_percentiles['max'] - q_policy_percentiles['min']
        q_inf_range = q_inf_percentiles['max'] - q_inf_percentiles['min']
        
        # Overlap detection
        # Does infeasible Q overlap with ID Q?
        overlap_inf_id = (q_inf_percentiles['max'] > q_id_percentiles['min']) & \
                        (q_inf_percentiles['min'] < q_id_percentiles['max'])
        
        # Does infeasible Q overlap with Q_min?
        overlap_inf_qmin = (q_inf_percentiles['max'] > q_min) | \
                        (q_inf_percentiles['min'] < q_min)
        
        # ===== Violation Analysis =====
        
        # ID Q violations (should be rare)
        q_id_below_qmin = (q_id_flat < q_min).mean()
        q_id_negative = (q_id_flat < 0).mean()
        
        # Infeasible Q violations (target: should be ~0 at Q_min)
        q_inf_below_qmin = (q_inf_flat < q_min).mean()
        q_inf_above_qmin = (q_inf_flat > q_min).mean()
        q_inf_negative = (q_inf_flat < 0).mean()
        
        # Policy Q sanity checks
        q_policy_negative = (q_policy_flat < 0).mean()
        q_policy_below_qmin = (q_policy_flat < q_min).mean()
        
        # ===== Separation Metrics =====
        
        # Gap between distributions (median-based)
        gap_id_inf_median = q_id_percentiles['p50'] - q_inf_percentiles['p50']
        gap_id_inf_mean = q_id_percentiles['mean'] - q_inf_percentiles['mean']
        
        # Separation quality (higher = better)
        # How many std deviations apart are the means?
        pooled_std = jnp.sqrt(
            (q_id_percentiles['std']**2 + q_inf_percentiles['std']**2) / 2
        )
        separation_score = gap_id_inf_mean / (pooled_std + 1e-6)
        
        # ===== Histogram Bins (for visualization) =====
        
        # Compute histogram of Q values
        num_bins = 50
        q_all_min = jnp.min(jnp.concatenate([q_id_flat, q_inf_flat]))
        q_all_max = jnp.max(jnp.concatenate([q_id_flat, q_inf_flat]))
        
        def compute_histogram(values, bins=num_bins):
            hist, bin_edges = jnp.histogram(
                values,
                bins=bins,
                range=(q_all_min, q_all_max)
            )
            return hist, bin_edges
        
        q_id_hist, q_id_bins = compute_histogram(q_id_flat)
        q_inf_hist, q_inf_bins = compute_histogram(q_inf_flat)
        
        # ===== Comprehensive Info Dict =====
        
        info = {
            # Losses
            'critic_loss': critic_loss,
            'td_loss': td_loss,
            'pa_loss': pa_loss,
            
            # ===== Q_ID Geometry =====
            'q_id/mean': q_id_percentiles['mean'],
            'q_id/std': q_id_percentiles['std'],
            'q_id/median': q_id_percentiles['p50'],
            'q_id/min': q_id_percentiles['min'],
            'q_id/max': q_id_percentiles['max'],
            'q_id/p25': q_id_percentiles['p25'],
            'q_id/p75': q_id_percentiles['p75'],
            'q_id/p95': q_id_percentiles['p95'],
            'q_id/range': q_id_range,
            'q_id/below_qmin_ratio': q_id_below_qmin,
            'q_id/negative_ratio': q_id_negative,
            
            # ===== Q_Policy Geometry =====
            'q_policy/mean': q_policy_percentiles['mean'],
            'q_policy/std': q_policy_percentiles['std'],
            'q_policy/median': q_policy_percentiles['p50'],
            'q_policy/min': q_policy_percentiles['min'],
            'q_policy/max': q_policy_percentiles['max'],
            'q_policy/p25': q_policy_percentiles['p25'],
            'q_policy/p75': q_policy_percentiles['p75'],
            'q_policy/range': q_policy_range,
            'q_policy/below_qmin_ratio': q_policy_below_qmin,
            'q_policy/negative_ratio': q_policy_negative,
            
            # ===== Q_Infeasible Geometry =====
            'q_inf/mean': q_inf_percentiles['mean'],
            'q_inf/std': q_inf_percentiles['std'],
            'q_inf/median': q_inf_percentiles['p50'],
            'q_inf/min': q_inf_percentiles['min'],
            'q_inf/max': q_inf_percentiles['max'],
            'q_inf/p25': q_inf_percentiles['p25'],
            'q_inf/p75': q_inf_percentiles['p75'],
            'q_inf/p01': q_inf_percentiles['p01'],
            'q_inf/p99': q_inf_percentiles['p99'],
            'q_inf/range': q_inf_range,
            'q_inf/below_qmin_ratio': q_inf_below_qmin,
            'q_inf/above_qmin_ratio': q_inf_above_qmin,
            'q_inf/negative_ratio': q_inf_negative,
            
            # ===== Gaps and Separation =====
            'gap/id_inf_mean': gap_id_inf_mean,
            'gap/id_inf_median': gap_id_inf_median,
            'gap/separation_score': separation_score,
            
            # ===== Overlap Flags =====
            'overlap/inf_id': jnp.asarray(overlap_inf_id, dtype=jnp.float32),
            'overlap/inf_qmin': jnp.asarray(overlap_inf_qmin, dtype=jnp.float32),
            
            # ===== Reference Values =====
            'ref/q_min': q_min,
            'ref/reward_scale': self.config['reward_scale'],
            
            # ===== Histograms (for custom plotting) =====
            'hist/q_id': q_id_hist,
            'hist/q_inf': q_inf_hist,
            'hist/bins': q_id_bins,
            
            # ===== ACFQL-specific (기존 유지) =====
            'q_policy_acfql': q_policy.mean() if use_weighted_target else 0.0,
            'q_sarsa': q_sarsa.mean() if use_weighted_target else 0.0,
            'target_q_mean': target_q.mean(),
            'target_q_std': target_q.std(),
            'td_error_mean': td_error.mean(),
            'td_error_abs_mean': td_error_abs.mean(),
            'td_error_abs_max': td_error_abs.max(),
            'q_next_policy_mean': next_qs_policy.mean(),
            'q_next_sarsa_mean': q_next_sarsa_mean,
            'td_error_per_sample': td_error_abs,
            'use_weighted_target': jnp.array(float(use_weighted_target)),
            'sarsa_beta': jnp.array(self.config.get('beta', 0.5) if use_weighted_target else 0.0),
        }

        return critic_loss, info

        # # ===== PARS: Reward Scaling =====
        # scaled_rewards = self.config['reward_scale'] * batch['rewards'][..., -1]
        
        # # TD target with reward scaling
        # gamma_H = self.config['discount'] ** self.config["horizon_length"]
        # target_q = scaled_rewards + gamma_H * batch['masks'][..., -1] * next_q
        # target_q = jax.lax.stop_gradient(target_q)

        # # Current Q-values
        # q = self.network.select('critic')(
        #     batch['observations'],
        #     actions=batch_actions,
        #     params=grad_params
        # )

        # # TD error and loss
        # td_error = q - target_q
        # td_error_abs = jnp.abs(td_error)
        # td_loss = (jnp.square(td_error) * batch['valid'][..., -1]).mean()

        # # ===== PARS: Penalizing Infeasible Actions =====
        
        # # Sample infeasible actions
        # infeasible_actions = self._sample_infeasible_actions(
        #     infeasible_rng,
        #     batch_size,
        #     action_dim
        # )
        
        # # Q-values for infeasible actions
        # infeasible_q = self.network.select('critic')(
        #     batch['observations'],
        #     actions=infeasible_actions,
        #     params=grad_params
        # )
        
        # # PA loss: Bidirectional penalty to Q_min
        # q_min = self.config['q_min']
        # pa_loss = jnp.square(infeasible_q - q_min).mean()
        
        # # Total critic loss
        # critic_loss = td_loss + self.config['alpha_pa'] * pa_loss

        # # ===== Statistics =====
        
        # # ID Q statistics
        # q_mean = q.mean()
        # q_std = q.std()
        
        # # Infeasible Q statistics
        # infeasible_q_mean = infeasible_q.mean()
        # infeasible_q_std = infeasible_q.std()
        # infeasible_q_min = infeasible_q.min()
        # infeasible_q_max = infeasible_q.max()
        
        # # Violations and gaps
        # violations = (infeasible_q < q_min).mean()
        # q_gap_raw = q_mean - infeasible_q_mean
        # q_gap_true = q_mean - jnp.maximum(infeasible_q_mean, q_min)
        
        # info = {
        #     'critic_loss': critic_loss,
        #     'td_loss': td_loss,
        #     'pa_loss': pa_loss,
            
        #     # ID Q statistics
        #     'q_mean': q_mean,
        #     'q_std': q_std,
            
        #     # Infeasible Q statistics
        #     'infeasible_q_mean': infeasible_q_mean,
        #     'infeasible_q_std': infeasible_q_std,
        #     'infeasible_q_min': infeasible_q_min,
        #     'infeasible_q_max': infeasible_q_max,
            
        #     # Gaps and violations
        #     'q_gap_raw': q_gap_raw,
        #     'q_gap_true': q_gap_true,
        #     'q_min_violations': violations,
            
        #     # ACFQL-specific
        #     'q_policy': q_policy.mean() if use_weighted_target else 0.0,
        #     'q_sarsa': q_sarsa.mean() if use_weighted_target else 0.0,
        #     'target_q_mean': target_q.mean(),
        #     'target_q_std': target_q.std(),
        #     'td_error_mean': td_error.mean(),
        #     'td_error_abs_mean': td_error_abs.mean(),
        #     'td_error_abs_max': td_error_abs.max(),
        #     'q_next_policy_mean': next_qs_policy.mean(),
        #     'q_next_sarsa_mean': q_next_sarsa_mean,
        #     'td_error_per_sample': td_error_abs,
        #     'use_weighted_target': jnp.array(float(use_weighted_target)),
        #     'sarsa_beta': jnp.array(self.config.get('beta', 0.5) if use_weighted_target else 0.0),
        # }

        # return critic_loss, info

    def _sample_infeasible_actions(
        self,
        rng: jax.random.PRNGKey,
        batch_size: int,
        action_dim: int
    ) -> jnp.ndarray:
        """Sample infeasible actions from the infeasible region.
        
        For action space [-1, 1]^n, infeasible region is defined as:
        A_I = [-2L, -L] ∪ [L, 2L] where L > 1
        
        Args:
            rng: Random key
            batch_size: Number of actions to sample
            action_dim: Action dimension (including chunking)
            
        Returns:
            Infeasible actions of shape (batch_size, action_dim)
        """
        L = self.config.get('L_infeasible', 1000.0)
        
        # Sample uniformly in [-1, 1]
        actions = jax.random.uniform(
            rng,
            shape=(batch_size, action_dim),
            minval=-1.0,
            maxval=1.0,
        )
        
        # Map to infeasible region
        infeasible_actions = jnp.where(
            actions < 0,
            (actions - 1) * L,  # Negative side: [-2L, -L]
            (actions + 1) * L,  # Positive side: [L, 2L]
        )
        
        return infeasible_actions

    @classmethod
    def create(
        cls,
        seed: int,
        ex_observations: jnp.ndarray,
        ex_actions: jnp.ndarray,
        config: ml_collections.ConfigDict,
    ) -> 'PARSACFQLAgent':
        """Create a new PARS-ACFQL agent.
        
        Adds PARS-specific configuration on top of ACFQL.
        """
        # Compute Q_min for penalization
        min_reward = config.get('min_reward', 0.0)
        horizon_length = config.get('horizon_length', 1)
        
        # Q_min with horizon-adjusted discount
        gamma_H = config['discount'] ** horizon_length
        q_min = config['reward_scale'] * min_reward / (1 - gamma_H)
        config['q_min'] = q_min
        
        # Force layer normalization for critic (essential for PARS)
        config['layer_norm'] = True
        
        # print(f"\n{'='*60}")
        # print(f"Creating PARS-ACFQL Agent")
        # print(f"{'='*60}")
        # print(f"Reward scale: {config['reward_scale']}")
        # print(f"Alpha_PA (PA weight): {config['alpha_pa']}")
        # print(f"Alpha (distill weight): {config['alpha']}")
        # print(f"Q_min: {q_min:.3f}")
        # print(f"L_infeasible: {config.get('L_infeasible', 1000.0)}")
        # print(f"Horizon length: {horizon_length}")
        # print(f"Action chunking: {config['action_chunking']}")
        # print(f"Actor type: {config['actor_type']}")
        # print(f"Layer Norm: {config['layer_norm']}")
        # print(f"{'='*60}\n")
        
        # Create using parent ACFQL
        return super().create(seed, ex_observations, ex_actions, config)


# agents/pars_acfql.py 끝에 추가

def get_config() -> ml_collections.ConfigDict:
    """Get default PARS-ACFQL configuration."""
    config = ml_collections.ConfigDict(
        dict(
            agent_name='pars',
            
            # Observation and action dimensions
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            
            # Learning
            lr=3e-4,
            batch_size=256,
            discount=0.99,
            tau=0.005,
            
            # PARS hyperparameters
            reward_scale=1000.0,  # Reward scaling
            alpha_pa=0.1,  # PA loss weight
            min_reward=0.0,
            L_infeasible=1000.0,
            
            # ACFQL hyperparameters
            alpha=100.0,  # Distillation weight
            q_agg='mean',
            num_qs=2,
            
            # Flow matching
            flow_steps=10,
            use_fourier_features=False,
            fourier_feature_dim=64,
            
            # Action chunking
            horizon_length=ml_collections.config_dict.placeholder(int),
            action_chunking=True,
            
            # Actor type
            actor_type="distill-ddpg",
            actor_num_samples=32,
            
            # Networks
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,  # PARS requires this
            actor_layer_norm=False,
            
            # Weighted target
            use_weighted_target=False,
            beta=0.5,
            
            # Regularization
            weight_decay=0.0,
            normalize_q_loss=False,
            
            # Encoder
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
    return config