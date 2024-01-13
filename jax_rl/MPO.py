from functools import partial
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from scipy.optimize import minimize

from jax_rl.buffers import ReplayBuffer
from jax_rl.models import Constant, GaussianPolicy, DoubleCritic
from jax_rl.utils import double_mse, gaussian_likelihood, kl_mvg_diag

treemap = jax.tree_util.tree_map


@partial(jax.jit, static_argnums=(2, 3))
def get_action(ts: train_state.TrainState, state: jnp.ndarray, sample: bool,
               num_samples: int=1, rng: Optional[jax.random.PRNGKey]=None
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


@partial(jax.jit, static_argnums=(2, 3, 5, 6))
def sample_actions_and_evaluate(
      actor_target_ts: train_state.TrainState,
      critic_target_ts: train_state.TrainState,
      max_action: float, action_dim: int, state: jnp.ndarray, batch_size: int,
      action_sample_size: int, rng: jax.random.PRNGKey
      ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    To build our nonparametric policy, q(s, a), we sample `action_sample_size`
    actions from each policy in the batch and evaluate their Q-values.
    """
    # get the policy distribution for each state and sample `action_sample_size`
    sampled_actions = get_action(actor_target_ts, state, sample=True, 
                                 num_samples=action_sample_size, rng=rng)
    assert sampled_actions.shape == (batch_size, action_sample_size, action_dim)
    sampled_actions = sampled_actions.reshape(
        (batch_size * action_sample_size, action_dim))
    sampled_actions = jax.lax.stop_gradient(sampled_actions)
    states_repeated = jnp.repeat(state, action_sample_size, axis=0)

    # evaluate each of the sampled actions at their corresponding state
    # we keep the `sampled_actions` array unnquashed because we need to calcuate
    # the log probabilities using it, but we pass the squashed actions to the critic
    Q1 = critic_target_ts.apply_fn(
        {'params': critic_target_ts.params},
        states_repeated, max_action * jnp.tanh(sampled_actions), True)
    Q1 = Q1.reshape((batch_size, action_sample_size))
    Q1 = jax.lax.stop_gradient(Q1)
    return Q1, sampled_actions


def e_step(
      actor_target_ts: train_state.TrainState,
      critic_target_ts: train_state.TrainState,
      max_action: float, action_dim: int, temp: float, eps_eta: float,
      state: jnp.ndarray, batch_size: int, action_sample_size: int,
      rng: jax.random.PRNGKey) -> Tuple[float, jnp.ndarray, jnp.ndarray]:
    """
    The 'E-step' from the MPO paper. We calculate our weights, which correspond
    to the relative likelihood of obtaining the maximum reward for each of the
    sampled actions. We also take steps on our temperature parameter, which
    induces diversity in the weights.
    """
    Q1, sampled_actions = sample_actions_and_evaluate(
        actor_target_ts, critic_target_ts, max_action, action_dim,
        state, batch_size, action_sample_size, rng)

    jac = jax.grad(dual, argnums=2)
    jac = partial(jac, Q1, eps_eta)

    # use nonconvex optimizer to minimize the dual of the temperature parameter
    # we have direct access to the jacobian function with jax so we can take
    # advantage of it here
    this_dual = partial(dual, Q1, eps_eta)
    bounds = [(1e-6, None)]
    res = minimize(this_dual, temp, jac=jac, method="SLSQP", bounds=bounds)
    temp = jax.lax.stop_gradient(res.x)

    # calculate the sample-based q distribution. we can think of these weights
    # as the relative likelihood of each of the sampled actions giving us the
    # maximum score when taken at the corresponding state.
    weights = jax.nn.softmax(Q1 / temp, axis=1)
    weights = jax.lax.stop_gradient(weights)
    weights = jnp.expand_dims(weights, axis=-1)

    return temp, weights, sampled_actions


@partial(jax.jit, static_argnums=(4, 5))
def m_step(
      actor_ts: train_state.TrainState,
      actor_target_ts: train_state.TrainState,
      mu_lagrange_ts: train_state.TrainState,
      sig_lagrange_ts: train_state.TrainState,
      eps_mu: float, eps_sig: float, state: jnp.ndarray, weights: jnp.ndarray,
      sampled_actions: jnp.ndarray
      ) -> Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]:
    """
    The 'M-step' from the MPO paper. We optimize our policy network to maximize
    the lower bound on the probablility of obtaining the maximum reward given
    that we act according to our policy (i.e. weighted according to our sampled actions).
    """

    def loss_fn(mu_lagrange_ts, sig_lagrange_ts, actor_params):
        # get the distribution of the actor network (current policy)
        mu, log_sig = actor_ts.apply_fn({'params': actor_params}, state, MPO=True)
        sig = jnp.exp(log_sig)
        # get the distribution of the target network (old policy)
        target_mu, target_log_sig = get_action(actor_target_ts, state, False)
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
            mu_lagrange_ts, eps_mu - jax.lax.stop_gradient(mu_kl))

        def sig_lagrange_step(sig_lagrange_ts, reg):
            def loss_fn(sig_lagrange_params):
                return jnp.sum(sig_lagrange_ts.apply_fn(
                    {'params': sig_lagrange_params}) * reg)
            grad = jax.grad(loss_fn)(sig_lagrange_ts.params)
            return sig_lagrange_ts.apply_gradients(grads=grad)

        sig_lagrange_ts = sig_lagrange_step(
            sig_lagrange_ts, eps_sig - jax.lax.stop_gradient(sig_kl))

        # maximize the log likelihood, regularized by the divergence between
        # the target policy and the current policy. the goal here is to fit
        # the parametric policy to have the minimum divergence with the nonparametric
        # distribution based on the sampled actions.
        actor_loss = -(actor_log_prob * weights).sum(axis=1).mean()
        actor_loss -= jax.lax.stop_gradient(
            mu_lagrange_ts.apply_fn({'params': mu_lagrange_ts.params})
        ) * (eps_mu - mu_kl)
        actor_loss -= jax.lax.stop_gradient(
            sig_lagrange_ts.apply_fn({'params': sig_lagrange_ts.params})
        ) * (eps_sig - sig_kl)
        return actor_loss.mean(), (mu_lagrange_ts, sig_lagrange_ts)

    grad, (mu_lagrange_ts, sig_lagrange_ts) = jax.grad(
        partial(loss_fn, mu_lagrange_ts, sig_lagrange_ts), has_aux=True
    )(actor_ts.params)
    actor_ts = actor_ts.apply_gradients(grads=grad)
    return actor_ts, mu_lagrange_ts, sig_lagrange_ts


@partial(jax.jit, static_argnums=(5, 6))
def get_td_target(
      actor_target_ts: train_state.TrainState,
      critic_target_ts: train_state.TrainState,
      next_state: jnp.ndarray, reward: jnp.ndarray, not_done: jnp.ndarray,
      max_action: float, discount: float, rng: jax.random.PRNGKey) -> jnp.ndarray:
    next_action = get_action(actor_target_ts, next_state, sample=True, rng=rng)
    next_action = max_action * jnp.tanh(next_action)
    target_Q1, target_Q2 = critic_target_ts.apply_fn(
        {'params': critic_target_ts.params}, next_state, next_action, False)
    target_Q = jnp.minimum(target_Q1, target_Q2)
    target_Q = reward + not_done * discount * target_Q
    return target_Q


@jax.jit
def critic_step(
      critic_ts: train_state.TrainState, state: jnp.ndarray,
      action: jnp.ndarray, target_Q: jnp.ndarray
      ) -> train_state.TrainState:
    def loss_fn(critic_params):
        current_Q1, current_Q2 = critic_ts.apply_fn(
            {'params': critic_params}, state, action, False)
        critic_loss = double_mse(current_Q1, current_Q2, target_Q)
        return critic_loss.mean()

    grad = jax.grad(loss_fn)(critic_ts.params)
    return critic_ts.apply_gradients(grads=grad)


class MPO:
    def __init__(
          self, state_dim: int, action_dim: int, max_action: float,
          discount: float = 0.99, lr: float = 3e-4, eps_eta: float = 0.1,
          eps_mu: float = 5e-4, eps_sig: float = 1e-5, target_freq: int = 250,
          seed: int = 0):
        self.rng = jax.random.PRNGKey(seed)
        self.rng, actor_key = jax.random.split(self.rng)

        policy = GaussianPolicy(action_dim=action_dim, max_action=max_action)
        actor_variables = policy.init(actor_key,
                                      jnp.ones((1, state_dim), jnp.float32))
        actor_gradient_transform = optax.chain(
            optax.clip_by_global_norm(40.0),
            optax.scale_by_adam(),
            optax.scale(-lr))
        self.actor_ts = train_state.TrainState.create(
            apply_fn=policy.apply,
            params=actor_variables["params"],
            tx=actor_gradient_transform) 
        self.actor_target_ts = train_state.TrainState.create(
            apply_fn=policy.apply,
            params=actor_variables["params"],
            tx=optax.GradientTransformation(lambda _: None, lambda _: None))
        
        self.rng, critic_key = jax.random.split(self.rng)
        critic_input_dim = [(1, state_dim), (1, action_dim)]
        init_batch = [jnp.ones(shape, jnp.float32) for shape in critic_input_dim]
        critic = DoubleCritic()
        critic_variables = critic.init(critic_key, *init_batch)
        critic_gradient_transform = optax.chain(
            optax.clip_by_global_norm(40.0),
            optax.scale_by_adam(),
            optax.scale(-lr))
        self.critic_ts = train_state.TrainState.create(
            apply_fn=critic.apply,
            params=critic_variables["params"],
            tx=critic_gradient_transform)
        self.critic_target_ts = train_state.TrainState.create(
            apply_fn=critic.apply,
            params=critic_variables["params"],
            tx=optax.GradientTransformation(lambda _: None, lambda _: None))
        
        self.rng, mu_key, sig_key = jax.random.split(self.rng, 3)
        mu_lagrange = Constant(start_value=1.0, absolute=True)
        self.mu_lagrange_ts = train_state.TrainState.create(
            apply_fn=mu_lagrange.apply,
            params=mu_lagrange.init(mu_key)["params"],
            tx=optax.adam(learning_rate=lr))

        sig_lagrange = Constant(start_value=100.0, absolute=True)
        self.sig_lagrange_ts = train_state.TrainState.create(
            apply_fn=sig_lagrange.apply,
            params=sig_lagrange.init(sig_key)["params"],
            tx=optax.adam(learning_rate=lr))

        self.temp = 1.0
        self.eps_eta = eps_eta
        self.eps_mu = eps_mu
        self.eps_sig = eps_sig

        self.max_action = max_action
        self.discount = discount
        self.target_freq = target_freq

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.total_it = 0

    def select_action(self, state: jnp.ndarray) -> jnp.ndarray:
        mu, _ = get_action(
            self.actor_ts, state.reshape(1, -1), sample=False)
        return mu.flatten()

    def sample_action(self, rng: jax.random.PRNGKey, state: jnp.ndarray) -> jnp.ndarray:
        return get_action(
            self.actor_ts, state.reshape(1, -1), sample=True, rng=rng).flatten()

    def train(
        self, replay_buffer: ReplayBuffer, batch_size: int, action_sample_size: int
    ):
        self.total_it += 1

        self.rng, buffer_key = jax.random.split(self.rng)
        buffer_out = replay_buffer.sample(buffer_key, batch_size)
        state, action, next_state, reward, not_done = buffer_out

        # Optimize critic.
        self.rng, td_key = jax.random.split(self.rng)
        target_Q = jax.lax.stop_gradient(get_td_target(
            self.actor_target_ts, self.critic_target_ts, next_state, reward,
            not_done, self.max_action, self.discount, td_key))
        self.critic_ts = critic_step(self.critic_ts, state, action, target_Q)

        # Expectation step.
        self.rng, e_step_key = jax.random.split(self.rng)
        self.temp, weights, sampled_actions = e_step(
            self.actor_target_ts, self.critic_target_ts, self.max_action,
            self.action_dim, self.temp, self.eps_eta, state, batch_size,
            action_sample_size, e_step_key)

        sampled_actions = sampled_actions.reshape(
            (batch_size, action_sample_size, self.action_dim))

        # Maximization step.
        (
            self.actor_ts,
            self.mu_lagrange_ts,
            self.sig_lagrange_ts,
        ) = m_step(self.actor_ts, self.actor_target_ts, self.mu_lagrange_ts,
                   self.sig_lagrange_ts, self.eps_mu, self.eps_sig, state,
                   weights, sampled_actions)

        if self.total_it % self.target_freq == 0:
            self.actor_target_ts = self.actor_target_ts.replace(params=self.actor_ts.params)
            self.critic_target_ts = self.critic_target_ts.replace(params=self.critic_ts.params)
