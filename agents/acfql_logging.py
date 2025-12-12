import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value


class ACFQLAgent_LOG(flax.struct.PyTreeNode):
    """Flow Q-learning (FQL) agent with action chunking. 
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    
    def critic_loss(self, batch, grad_params, rng):
        """FQL critic loss with optional PTR weighted target + 로깅용 통계 추가"""
        
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(
                batch["actions"],
                (batch["actions"].shape[0], -1),
            )
        else:
            batch_actions = batch["actions"][..., 0, :]
        
        # ===== Standard target (policy's action) =====
        rng, sample_rng = jax.random.split(rng)
        next_actions_policy = self.sample_actions(
            batch['next_observations'][..., -1, :], 
            rng=sample_rng
        )
        next_qs_policy = self.network.select('target_critic')(
            batch['next_observations'][..., -1, :], 
            actions=next_actions_policy
        )
        
        # ===== PTR: SARSA target (trajectory's actual next action) =====
        use_weighted_target = self.config.get('use_weighted_target', False)

        # 로깅용 초기값 (use_weighted_target=False일 때도 info 키는 항상 존재하도록)
        q_policy = 0.0
        q_sarsa = 0.0
        q_next_sarsa_mean = 0.0

        if use_weighted_target:
            beta = self.config.get('beta', 0.5)

            if self.config["action_chunking"]:
                # next_actions: (B, H, act_dim)  →  (B, H*act_dim)
                next_actions_traj = batch['next_actions'].reshape(
                    batch['next_actions'].shape[0], -1
                )
            else:
                # non-chunk: 첫 step만 사용
                next_actions_traj = batch['next_actions'][..., 0, :]

            next_qs_sarsa = self.network.select('target_critic')(
                batch['next_observations'][..., -1, :],
                actions=next_actions_traj,
            )

            # Q ensemble aggregation
            if self.config['q_agg'] == 'min':
                q_policy = next_qs_policy.min(axis=0)
                q_sarsa = next_qs_sarsa.min(axis=0)
            else:
                q_policy = next_qs_policy.mean(axis=0)
                q_sarsa = next_qs_sarsa.mean(axis=0)

            # Equation (4): weighted combination
            next_q = (1.0 - beta) * q_sarsa + beta * q_policy

            # 로깅용 통계
            q_next_sarsa_mean = next_qs_sarsa.mean()
        else:
            if self.config['q_agg'] == 'min':
                next_q = next_qs_policy.min(axis=0)
            else:
                next_q = next_qs_policy.mean(axis=0)
        
        # TD target
        gamma_H = self.config['discount'] ** self.config["horizon_length"]
        target_q = batch['rewards'][..., -1] + gamma_H * batch['masks'][..., -1] * next_q
        
        # Q prediction
        q = self.network.select('critic')(
            batch['observations'], 
            actions=batch_actions, 
            params=grad_params
        )
        
        td_error = q - target_q
        td_error_abs = jnp.abs(td_error)

        critic_loss = (jnp.square(td_error) * batch['valid'][..., -1]).mean()
        
        return critic_loss, {
            # 원래 있던 것
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_policy': q_policy.mean() if use_weighted_target else 0.0,
            'q_sarsa': q_sarsa.mean() if use_weighted_target else 0.0,

            # ==== 추가 로깅: SARSA vs non-SARSA 비교 / TD 통계 ====
            'q_std': q.std(),
            'target_q_mean': target_q.mean(),
            'target_q_std': target_q.std(),
            'td_error_mean': td_error.mean(),
            'td_error_abs_mean': td_error_abs.mean(),
            'td_error_abs_max': td_error_abs.max(),

            # next-Q 통계 (policy vs SARSA)
            'q_next_policy_mean': next_qs_policy.mean(),
            'q_next_sarsa_mean': q_next_sarsa_mean,

            'td_error_per_sample': td_error_abs,

            # weighted target 사용 여부 / beta 값
            'use_weighted_target': jnp.array(float(use_weighted_target)),
            'sarsa_beta': jnp.array(self.config.get('beta', 0.5) if use_weighted_target else 0.0),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss + 로깅용 통계 추가."""
        if self.config["action_chunking"]:
            # fold in horizon_length together with action_dim
            batch_actions = jnp.reshape(
                batch["actions"],
                (batch["actions"].shape[0], -1),
            )
        else:
            # take the first one
            batch_actions = batch["actions"][..., 0, :]
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(
            batch['observations'], x_t, t, params=grad_params
        )

        # only bc on the valid chunk indices
        if self.config["action_chunking"]:
            bc_flow_loss = jnp.mean(
                jnp.reshape(
                    (pred - vel) ** 2,
                    (batch_size, self.config["horizon_length"], self.config["action_dim"]),
                ) * batch["valid"][..., None]
            )
        else:
            bc_flow_loss = jnp.mean(jnp.square(pred - vel))

        # 로깅용 초기값
        q_loss = jnp.zeros(())
        q_value_mean = jnp.zeros(())
        action_norm_mean = jnp.zeros(())
        
        if self.config["actor_type"] == "distill-ddpg":
            # Distillation loss.
            rng, noise_rng = jax.random.split(rng)
            noises = jax.random.normal(noise_rng, (batch_size, action_dim))
            target_flow_actions = self.compute_flow_actions(
                batch['observations'], noises=noises
            )
            actor_actions = self.network.select('actor_onestep_flow')(
                batch['observations'], noises, params=grad_params
            )
            distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)
            
            # Q loss.
            actor_actions = jnp.clip(actor_actions, -1, 1)

            qs = self.network.select('critic')(batch['observations'], actions=actor_actions)
            q = jnp.mean(qs, axis=0)
            q_loss = -q.mean()

            # 로깅용 추가 통계: actor가 보는 Q와 action norm
            q_value_mean = q.mean()
            action_norm_mean = jnp.linalg.norm(actor_actions, axis=-1).mean()
        else:
            distill_loss = jnp.zeros(())
            # q_loss, q_value_mean, action_norm_mean은 위에서 0으로 초기화

        # Total loss.
        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,

            # ==== 추가 로깅: actor가 만드는 action / Q 구조 파악 ====
            'q_loss': q_loss,
            'q_value_mean': q_value_mean,
            'actor_action_norm_mean': action_norm_mean,
            'bc_batch_action_norm_mean': jnp.linalg.norm(batch_actions, axis=-1).mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(agent, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)
    
    @jax.jit
    def batch_update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        # update_size = batch["observations"].shape[0]
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)
    
    @jax.jit
    def sample_actions(self, observations, rng=None):
        # 1) rng 기본값
        if rng is None:
            rng = self.rng

        # 2) 단일 obs인지 확인해서 batch dim 붙이기
        ob_dims = tuple(self.config['ob_dims'])
        added_batch = False

        if observations.ndim == len(ob_dims):
            observations = observations[None, ...]  # (1, obs_dim...) 으로 만들기
            added_batch = True

        batch_size = observations.shape[0]
        action_dim = self.config['action_dim'] * (
            self.config['horizon_length'] if self.config["action_chunking"] else 1
        )

        if self.config["actor_type"] == "distill-ddpg":
            noises = jax.random.normal(
                rng,
                (batch_size, action_dim)
            )
            actions = self.network.select('actor_onestep_flow')(observations, noises)
            actions = jnp.clip(actions, -1, 1)

        elif self.config["actor_type"] == "best-of-n":
            noises = jax.random.normal(
                rng,
                (batch_size, self.config["actor_num_samples"], action_dim)
            )
            obs_rep = jnp.repeat(
                observations[:, None, :],
                self.config["actor_num_samples"],
                axis=1
            )

            actions = self.compute_flow_actions(obs_rep, noises)
            actions = jnp.clip(actions, -1, 1)

            qs = self.network.select("critic")(obs_rep, actions)
            if self.config["q_agg"] == "mean":
                q = qs.mean(axis=0)
            else:
                q = qs.min(axis=0)
            idx = jnp.argmax(q, axis=-1)
            actions = actions[jnp.arange(batch_size), idx]

        # 3) single obs였으면 다시 batch 축 제거
        if added_batch:
            actions = actions[0]

        return actions

    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)
        actions = noises
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(
                observations, actions, t, is_encoded=True
            )
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        # ob_dims = ex_observations.shape
        if ex_observations.ndim > 1:
            # (obs_dim,) or (H, W, C)
            ob_dims = ex_observations.shape[1:]
        else:
            ob_dims = ex_observations.shape

        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate(
                [ex_actions] * config["horizon_length"], axis=-1
            )
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

        # Define networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            encoder=encoders.get('critic'),
        )

        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
            use_fourier_features=config["use_fourier_features"],
            fourier_feature_dim=config["fourier_feature_dim"],
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
        )

        network_info = dict(
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, full_actions, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, full_actions)),
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
        )
        if encoders.get('actor_bc_flow') is not None:
            # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_bc_flow_encoder'] = (
                encoders.get('actor_bc_flow'),
                (ex_observations,),
            )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        if config["weight_decay"] > 0.0:
            network_tx = optax.adamw(
                learning_rate=config['lr'],
                weight_decay=config["weight_decay"],
            )
        else:
            network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params

        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():

    config = ml_collections.ConfigDict(
        dict(
            agent_name='acfql_log',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='mean',  # Aggregation method for target Q values.
            alpha=100.0,  # BC coefficient (need to be tuned for each environment).
            num_qs=2,  # critic ensemble size
            flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            horizon_length=ml_collections.config_dict.placeholder(int),  # will be set
            action_chunking=True,  # False means n-step return
            actor_type="distill-ddpg",
            actor_num_samples=32,  # for actor_type="best-of-n" only
            use_fourier_features=False,
            fourier_feature_dim=64,
            weight_decay=0.0,

            use_weighted_target=False,  # whether to use PTR weighted target
            beta=0.5,  # for PTR weighted target
        )
    )
    return config
