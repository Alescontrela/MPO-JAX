from functools import partial
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
from jax.random import PRNGKey
import optax
from flax import struct
from flax.training.train_state import TrainState
from scipy.optimize import minimize

from jax_rl.buffers import ReplayBuffer
from jax_rl.models import Constant, GaussianPolicy, DoubleCritic
from jax_rl.utils import double_mse, gaussian_likelihood, kl_mvg_diag

treemap = jax.tree_util.tree_map
sg = lambda x: treemap(jax.lax.stop_gradient, x)


@partial(jax.jit, static_argnums=(2, 3))
def get_action(ts: TrainState, state: jnp.ndarray, sample: bool,
               num_samples: int=1, rng: Optional[PRNGKey]=None
               ) -> jnp.ndarray:
  mu, log_sig = ts.apply_fn({'params': ts.params}, state, MPO=True)
  if not sample:
      return mu, log_sig
  else:
      if num_samples == 1:
        return (mu + jax.random.normal(rng, mu.shape) * jnp.exp(log_sig))
      else:
        batch_size, action_dim = mu.shape
        sample_shape = (batch_size, num_samples, action_dim)
        mu = jnp.expand_dims(mu, axis=1)
        sig = jnp.expand_dims(jnp.exp(log_sig), axis=1)
        return (mu + sig * jax.random.normal(rng, sample_shape))


@partial(jax.jit, static_argnums=1)
def dual(Q1: jnp.ndarray, eps_eta: float, temp: jnp.ndarray) -> float:
    """
    Dual function of the non-parametric variational distribution using samples.
    g(η) = η*ε + η \\mean \\log (\\mean \\exp(Q(a, s)/η))
    """
    out = temp * (
        eps_eta
        + jnp.mean(jax.scipy.special.logsumexp(Q1 / temp, axis=1))
        - jnp.log(Q1.shape[1]))
    return out.sum()


class MPO(struct.PyTreeNode):

    rng: PRNGKey
    actor: TrainState
    actor_target: TrainState
    critic: TrainState
    critic_target: TrainState
    mu_lagrange: TrainState
    sig_lagrange: TrainState
    temp: float

    discount: float = struct.field(pytree_node=False)
    eps_eta: float = struct.field(pytree_node=False)
    eps_mu: float = struct.field(pytree_node=False)
    eps_sig: float = struct.field(pytree_node=False)
    max_action: float = struct.field(pytree_node=False)
    target_freq: int = struct.field(pytree_node=False)
    state_dim: int = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False)
    total_it: int

    @classmethod
    def create(
          cls, state_dim: int, action_dim: int, max_action: float,
          discount: float = 0.99, lr: float = 3e-4, eps_eta: float = 0.1,
          eps_mu: float = 5e-4, eps_sig: float = 1e-5, target_freq: int = 250,
          seed: int = 0):
        rng = PRNGKey(seed)
        rng, actor_key = jax.random.split(rng)

        actor_def = GaussianPolicy(action_dim=action_dim, max_action=max_action)
        actor_params = actor_def.init(
            actor_key, jnp.ones((1, state_dim), jnp.float32))["params"]
        actor_gradient_transform = optax.chain(
            optax.clip_by_global_norm(40.0),
            optax.scale_by_adam(),
            optax.scale(-lr))
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=actor_gradient_transform) 
        actor_target = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None))
        
        rng, critic_key = jax.random.split(rng)
        critic_input_dim = [(1, state_dim), (1, action_dim)]
        init_batch = [jnp.ones(shape, jnp.float32) for shape in critic_input_dim]
        critic_def = DoubleCritic()
        critic_params = critic_def.init(critic_key, *init_batch)["params"]
        critic_gradient_transform = optax.chain(
            optax.clip_by_global_norm(40.0),
            optax.scale_by_adam(),
            optax.scale(-lr))
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=critic_gradient_transform)
        critic_target = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None))
        
        rng, mu_key, sig_key = jax.random.split(rng, 3)
        mu_lagrange_def = Constant(start_value=1.0, absolute=True)
        mu_lagrange = TrainState.create(
            apply_fn=mu_lagrange_def.apply,
            params=mu_lagrange_def.init(mu_key)["params"],
            tx=optax.adam(learning_rate=lr))

        sig_lagrange_def = Constant(start_value=100.0, absolute=True)
        sig_lagrange = TrainState.create(
            apply_fn=sig_lagrange_def.apply,
            params=sig_lagrange_def.init(sig_key)["params"],
            tx=optax.adam(learning_rate=lr))

        return cls(
            rng=rng, actor=actor, actor_target=actor_target,
            critic=critic, critic_target=critic_target,
            mu_lagrange=mu_lagrange, sig_lagrange=sig_lagrange,
            temp=1.0, discount=discount, eps_eta=eps_eta,
            eps_mu=eps_mu, eps_sig=eps_sig, max_action=max_action,
            target_freq=target_freq, state_dim=state_dim,
            action_dim=action_dim, total_it=0)

    def select_action(self, state: jnp.ndarray) -> jnp.ndarray:
        mu, _ = get_action(
            self.actor, state.reshape(1, -1), sample=False)
        return mu.flatten()

    def sample_action(self, state: jnp.ndarray) -> jnp.ndarray:
        rng, key = jax.random.split(rng)
        actions = get_action(
            self.actor, state.reshape(1, -1), sample=True, rng=key).flatten()
        return self.replace(rng=rng), actions

    @jax.jit
    def update_critic(
          self, state: jnp.ndarray, action: jnp.ndarray,
          next_state: jnp.ndarray, reward: jnp.ndarray, not_done: jnp.ndarray
          ) -> struct.PyTreeNode:

        rng, key = jax.random.split(self.rng)
        next_action = get_action(self.actor_target, next_state, sample=True, rng=key)
        next_action = self.max_action * jnp.tanh(next_action)
        target_Q1, target_Q2 = self.critic_target.apply_fn(
            {'params': self.critic_target.params}, next_state, next_action, False)
        target_Q = jnp.minimum(target_Q1, target_Q2)
        target_Q = reward + not_done * self.discount * target_Q
        target_Q = sg(target_Q)

        def loss_fn(critic_params):
            current_Q1, current_Q2 = self.critic.apply_fn(
                {'params': critic_params}, state, action, False)
            critic_loss = double_mse(current_Q1, current_Q2, target_Q)
            return critic_loss.mean()

        grads = jax.grad(loss_fn)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)
        return self.replace(critic=critic, rng=rng)

    @partial(jax.jit, static_argnames=["action_sample_size", "batch_size"])
    def sample_actions_and_evaluate(
          self, state: jnp.ndarray, action_sample_size: int, batch_size: int
          ) -> Tuple[struct.PyTreeNode, jnp.ndarray, jnp.ndarray]:
        """
        To build our nonparametric policy, q(s, a), we sample `action_sample_size`
        actions from each policy in the batch and evaluate their Q-values.
        """
        rng, key = jax.random.split(self.rng)
        # get the policy distribution for each state and sample `action_sample_size`
        sampled_actions = get_action(
            self.actor_target, state, sample=True, 
            num_samples=action_sample_size, rng=key)
        assert sampled_actions.shape == (
            batch_size, action_sample_size, self.action_dim)
        sampled_actions = sampled_actions.reshape(
            (batch_size * action_sample_size, self.action_dim))
        sampled_actions = sg(sampled_actions)
        states_repeated = jnp.repeat(state, action_sample_size, axis=0)

        # evaluate each of the sampled actions at their corresponding state
        # we keep the `sampled_actions` array unnquashed because we need to calcuate
        # the log probabilities using it, but we pass the squashed actions to the critic
        Q1 = self.critic_target.apply_fn(
            {'params': self.critic_target.params},
            states_repeated, self.max_action * jnp.tanh(sampled_actions), True)
        Q1 = Q1.reshape((batch_size, action_sample_size))
        Q1 = sg(Q1)
        return self.replace(rng=rng), Q1, sampled_actions

    def update_temp_multi_step(self, Q: jnp.ndarray) -> struct.PyTreeNode:
        new_agent = self
        jac = jax.grad(dual, argnums=2)
        jac = partial(jac, Q, new_agent.eps_eta)

        # use nonconvex optimizer to minimize the dual of the temperature parameter
        # we have direct access to the jacobian function with jax so we can take
        # advantage of it here
        this_dual = partial(dual, Q, new_agent.eps_eta)
        bounds = [(1e-6, None)]
        res = minimize(this_dual, new_agent.temp, jac=jac, method="SLSQP", bounds=bounds)
        temp = jax.lax.stop_gradient(res.x)
        return new_agent.replace(temp=temp)

    def e_step(
          self, state: jnp.ndarray, action_sample_size: int, batch_size: int):
        """
        The 'E-step' from the MPO paper. We calculate our weights, which correspond
        to the relative likelihood of obtaining the maximum reward for each of the
        sampled actions. We also take steps on our temperature parameter, which
        induces diversity in the weights.
        """

        new_agent = self
        new_agent, Q1, sampled_actions = new_agent.sample_actions_and_evaluate(
            state, action_sample_size, batch_size)

        jac = jax.grad(dual, argnums=2)
        jac = partial(jac, Q1, new_agent.eps_eta)

        # use nonconvex optimizer to minimize the dual of the temperature parameter
        # we have direct access to the jacobian function with jax so we can take
        # advantage of it here
        this_dual = partial(dual, Q1, new_agent.eps_eta)
        bounds = [(1e-6, None)]
        res = minimize(this_dual, new_agent.temp, jac=jac, method="SLSQP", bounds=bounds)
        temp = jax.lax.stop_gradient(res.x)

        # calculate the sample-based q distribution. we can think of these weights
        # as the relative likelihood of each of the sampled actions giving us the
        # maximum score when taken at the corresponding state.
        weights = jax.nn.softmax(Q1 / temp, axis=1)
        weights = jax.lax.stop_gradient(weights)
        weights = jnp.expand_dims(weights, axis=-1)

        sampled_actions = sampled_actions.reshape(
            (batch_size, action_sample_size, new_agent.action_dim))

        return new_agent.replace(temp=temp), weights, sampled_actions

    @jax.jit
    def m_step(
          self, state: jnp.ndarray, weights: jnp.ndarray,
          sampled_actions: jnp.ndarray) -> struct.PyTreeNode:
        """
        The 'M-step' from the MPO paper. We optimize our policy network to maximize
        the lower bound on the probablility of obtaining the maximum reward given
        that we act according to our policy (i.e. weighted according to our sampled actions).
        """

        def loss_fn(mu_lagrange_ts, sig_lagrange_ts, actor_params):
            # get the distribution of the actor network (current policy)
            mu, log_sig = self.actor.apply_fn({'params': actor_params}, state, MPO=True)
            sig = jnp.exp(log_sig)
            # get the distribution of the target network (old policy)
            target_mu, target_log_sig = get_action(self.actor_target, state, False)
            target_mu = jax.lax.stop_gradient(target_mu)
            target_log_sig = jax.lax.stop_gradient(target_log_sig)
            target_sig = jnp.exp(target_log_sig)

            # get the log likelihooods of the sampled actions according to the
            # decoupled distributions. described in section 4.2.1 of
            # Relative Entropy Regularized Policy Iteration
            # this ensures that the nonparametric policy won't collapse to give
            # a probability of 1 to the best action, which is a risk when we use
            # the on-policy distribution to calculate the likelihood.
            actor_log_prob = gaussian_likelihood(sampled_actions, target_mu, log_sig)
            actor_log_prob += gaussian_likelihood(sampled_actions, mu, target_log_sig)

            mu_kl = kl_mvg_diag(target_mu, target_sig, mu, target_sig).mean()
            sig_kl = kl_mvg_diag(target_mu, target_sig, target_mu, sig).mean()

            def mu_lagrange_step(mu_lagrange_ts, reg):
                def loss_fn(mu_lagrange_params):
                    return jnp.sum(mu_lagrange_ts.apply_fn(
                        {'params': mu_lagrange_params}) * reg)
                grad = jax.grad(loss_fn)(mu_lagrange_ts.params)
                return mu_lagrange_ts.apply_gradients(grads=grad)

            mu_lagrange_ts = mu_lagrange_step(
                mu_lagrange_ts, self.eps_mu - jax.lax.stop_gradient(mu_kl))

            def sig_lagrange_step(sig_lagrange_ts, reg):
                def loss_fn(sig_lagrange_params):
                    return jnp.sum(sig_lagrange_ts.apply_fn(
                        {'params': sig_lagrange_params}) * reg)
                grad = jax.grad(loss_fn)(sig_lagrange_ts.params)
                return sig_lagrange_ts.apply_gradients(grads=grad)

            sig_lagrange_ts = sig_lagrange_step(
                sig_lagrange_ts, self.eps_sig - jax.lax.stop_gradient(sig_kl))

            # maximize the log likelihood, regularized by the divergence between
            # the target policy and the current policy. the goal here is to fit
            # the parametric policy to have the minimum divergence with the nonparametric
            # distribution based on the sampled actions.
            actor_loss = -(actor_log_prob * weights).sum(axis=1).mean()
            actor_loss -= jax.lax.stop_gradient(
                mu_lagrange_ts.apply_fn({'params': mu_lagrange_ts.params})
            ) * (self.eps_mu - mu_kl)
            actor_loss -= jax.lax.stop_gradient(
                sig_lagrange_ts.apply_fn({'params': sig_lagrange_ts.params})
            ) * (self.eps_sig - sig_kl)
            return actor_loss.mean(), (mu_lagrange_ts, sig_lagrange_ts)

        grad, (mu_lagrange, sig_lagrange) = jax.grad(
            partial(loss_fn, self.mu_lagrange, self.sig_lagrange), has_aux=True
        )(self.actor.params)
        actor = self.actor.apply_gradients(grads=grad)
        return self.replace(actor=actor, mu_lagrange=mu_lagrange, sig_lagrange=sig_lagrange)

    def update(
        self, replay_buffer: ReplayBuffer, batch_size: int, action_sample_size: int
    ):
        new_agent = self
        new_agent = new_agent.replace(total_it=new_agent.total_it + 1)

        rng, buffer_key = jax.random.split(new_agent.rng)
        new_agent = new_agent.replace(rng=rng)
        buffer_out = replay_buffer.sample(buffer_key, batch_size)
        state, action, next_state, reward, not_done = buffer_out

        # Optimize critic.
        new_agent = new_agent.update_critic(
            state, action, next_state, reward, not_done)

        # Expectation step.
        new_agent, weights, sampled_actions = new_agent.e_step(
            state, action_sample_size, batch_size)

        # Maximization step.
        new_agent = new_agent.m_step(state, weights, sampled_actions)

        if new_agent.total_it % new_agent.target_freq == 0:
          new_agent = new_agent.replace(
              actor_target=new_agent.actor_target.replace(
                  params=new_agent.actor.params),
              critic_target=new_agent.critic_target.replace(
                  params=new_agent.critic.params))

        return new_agent
