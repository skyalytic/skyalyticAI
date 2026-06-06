"""
Active Inference Agent - Continuous Gaussian Implementation

Implements the active inference framework based on the Free Energy
Principle (Friston, 2010) in continuous state spaces. The agent
maintains Gaussian beliefs about hidden states and selects actions
that minimize expected free energy, balancing:

1. Epistemic (information-seeking) value: actions that reduce
   uncertainty about hidden states
2. Pragmatic (goal-seeking) value: actions that lead to preferred
   outcomes

Mathematical formulation (continuous):
    Belief: q(s) = N(mu, Sigma)
    Transition: p(s'|s,a) = N(mu_trans(s,a), Sigma_trans(s,a))
    Observation: p(o|s) = N(mu_obs(s), Sigma_obs(s))

    Variational free energy:
    F = E_q[-log p(o|s)] - E_q[-log q(s)]
      = 0.5 * (log|Sigma| + (mu-mu_prior)^T Sigma^{-1} (mu-mu_prior)
        + (o-mu_obs)^T Sigma_obs^{-1} (o-mu_obs)) + const

    Expected free energy:
    G(a) = E_q[H[q(s'|o,a)]] - E_q[log p(o)]
         = -epistemic_value - pragmatic_value

This continuous formulation scales to arbitrary state dimensions
unlike the discrete Dirichlet-based version which requires O(n^3)
storage for n discrete states.

Key advantages over discrete active inference:
- Scales to high-dimensional continuous state spaces
- Neural network parameterized transition/observation models
- Compatible with GPU batch processing
- No O(n^3) memory bottleneck
- Supports gradient-based learning of internal models
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    pass


def _kaiming_init(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((rows, cols)) * np.sqrt(2.0 / cols)


class _MLP:
    """
    Multi-layer perceptron with configurable depth.

    Parameters
    ----------
    input_dim : int
    hidden_dims : list of int
    output_dim : int
    learning_rate : float
    rng : numpy Generator
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        learning_rate: float,
        rng: np.random.Generator,
    ) -> None:
        self.learning_rate = learning_rate
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            W = _kaiming_init(dims[i + 1], dims[i], rng)
            b = np.zeros(dims[i + 1], dtype=np.float64)
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for i in range(len(self.weights) - 1):
            x = np.tanh(self.weights[i] @ x + self.biases[i])
        x = self.weights[-1] @ x + self.biases[-1]
        return x

    def forward_batch(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return self.forward(x)
        for i in range(len(self.weights) - 1):
            x = np.tanh(x @ self.weights[i].T + self.biases[i])
        x = x @ self.weights[-1].T + self.biases[-1]
        return x

    def backward(
        self,
        x_input: np.ndarray,
        grad_output: np.ndarray,
    ) -> np.ndarray:
        activations = [x_input.copy()]
        pre_activations = []
        x = x_input.copy()

        for i in range(len(self.weights) - 1):
            z = self.weights[i] @ x + self.biases[i]
            pre_activations.append(z)
            x = np.tanh(z)
            activations.append(x.copy())

        z = self.weights[-1] @ x + self.biases[-1]
        pre_activations.append(z)
        activations.append(z.copy())

        grad = grad_output
        for i in range(len(self.weights) - 1, -1, -1):
            if i < len(self.weights) - 1:
                grad = grad * (1.0 - np.tanh(pre_activations[i]) ** 2)

            dW = np.outer(grad, activations[i])
            db = grad.copy()

            grad = self.weights[i].T @ grad

            flat_grad = np.concatenate([dW.flatten(), db.flatten()])
            flat_norm = np.linalg.norm(flat_grad)
            if flat_norm > 1.0:
                scale = 1.0 / flat_norm
                dW = dW * scale
                db = db * scale

            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db

        return grad

    def get_params(self) -> Dict[str, np.ndarray]:
        params = {}
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            params[f"W{i}"] = W.copy()
            params[f"b{i}"] = b.copy()
        return params

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        for i in range(len(self.weights)):
            self.weights[i] = params[f"W{i}"].copy()
            self.biases[i] = params[f"b{i}"].copy()


if _TORCH_AVAILABLE:

    class _MLP_Torch(torch.nn.Module):
        """
        PyTorch MLP mirroring _MLP for GPU acceleration.

        Uses the same architecture (hidden layers with tanh, linear output)
        and supports weight sync from the numpy _MLP.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int],
            output_dim: int,
            device: Any = None,
        ) -> None:
            super().__init__()
            dims = [input_dim] + hidden_dims + [output_dim]
            self.layers = torch.nn.ModuleList()
            for i in range(len(dims) - 1):
                self.layers.append(
                    torch.nn.Linear(dims[i], dims[i + 1], dtype=torch.float64)
                )
            self.device = device
            if device is not None:
                self.to(device)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for i in range(len(self.layers) - 1):
                x = torch.tanh(self.layers[i](x))
            x = self.layers[-1](x)
            return x

        def sync_from_numpy(
            self,
            weights: List[np.ndarray],
            biases: List[np.ndarray],
        ) -> None:
            """Sync weights from numpy _MLP to this torch model."""
            with torch.no_grad():
                for i, layer in enumerate(self.layers):
                    layer.weight.copy_(
                        torch.as_tensor(
                            weights[i], dtype=torch.float64, device=self.device
                        )
                    )
                    layer.bias.copy_(
                        torch.as_tensor(
                            biases[i], dtype=torch.float64, device=self.device
                        )
                    )


class ActiveInferenceAgent:
    """
    Continuous Active Inference Agent with Gaussian beliefs.

    The agent operates in a continuous state-action space and maintains:
    - Gaussian belief q(s) = N(mu, Sigma) over hidden states
    - Neural network parameterized transition model
    - Neural network parameterized observation model
    - Gaussian prior preferences over observations

    This replaces the discrete Dirichlet-based implementation with
    a scalable continuous version that supports arbitrary state
    dimensions and GPU batch processing.

    Parameters
    ----------
    state_dim : int
        Dimension of continuous hidden state space.
    obs_dim : int
        Dimension of continuous observation space.
    n_actions : int
        Number of available discrete actions.
    hidden_dim : int
        Hidden dimension for internal neural networks.
    n_hidden_layers : int
        Number of hidden layers in transition/observation networks.
    planning_horizon : int
        Number of time steps to look ahead when evaluating policies.
    temperature : float
        Softmax temperature for action selection.
    learning_rate : float
        Learning rate for internal model updates.
    belief_lr : float
        Learning rate for variational belief updating.
    n_belief_steps : int
        Number of gradient steps for belief updating.
    action_precision : float
        Precision of action selection.
    device : Any
        PyTorch device for GPU acceleration.
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 128,
        n_hidden_layers: int = 2,
        planning_horizon: int = 1,
        temperature: float = 1.0,
        learning_rate: float = 0.001,
        belief_lr: float = 0.01,
        n_belief_steps: int = 5,
        action_precision: float = 1.0,
        device: Any = None,
    ) -> None:
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if obs_dim <= 0:
            raise ValueError(f"obs_dim must be positive, got {obs_dim}")
        if n_actions <= 0:
            raise ValueError(f"n_actions must be positive, got {n_actions}")

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.planning_horizon = planning_horizon
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.belief_lr = belief_lr
        self.n_belief_steps = n_belief_steps
        self.action_precision = action_precision
        self.device = device

        self._use_gpu = (
            _TORCH_AVAILABLE
            and device is not None
            and str(device) == "cuda"
        )

        rng = np.random.default_rng(42)

        self.belief_mu = np.zeros(state_dim, dtype=np.float64)
        self.belief_Sigma = np.eye(state_dim, dtype=np.float64)

        self.transition_net = _MLP(
            input_dim=state_dim + n_actions,
            hidden_dims=[hidden_dim] * n_hidden_layers,
            output_dim=state_dim * 2,
            learning_rate=learning_rate,
            rng=rng,
        )

        self.observation_net = _MLP(
            input_dim=state_dim,
            hidden_dims=[hidden_dim] * n_hidden_layers,
            output_dim=obs_dim * 2,
            learning_rate=learning_rate,
            rng=rng,
        )

        self.preference_mu = np.zeros(obs_dim, dtype=np.float64)
        self.preference_Sigma = np.eye(obs_dim, dtype=np.float64)

        self._prior_mu = np.zeros(state_dim, dtype=np.float64)
        self._prior_Sigma = np.eye(state_dim, dtype=np.float64)

        self.history: List[Dict[str, Any]] = []
        self.step_count: int = 0

        # GPU models (created only when GPU is available)
        self._trans_net_torch: Optional[Any] = None
        self._obs_net_torch: Optional[Any] = None
        if self._use_gpu:
            self._trans_net_torch = _MLP_Torch(
                input_dim=state_dim + n_actions,
                hidden_dims=[hidden_dim] * n_hidden_layers,
                output_dim=state_dim * 2,
                device=device,
            )
            self._obs_net_torch = _MLP_Torch(
                input_dim=state_dim,
                hidden_dims=[hidden_dim] * n_hidden_layers,
                output_dim=obs_dim * 2,
                device=device,
            )
            self._sync_torch_weights()

    def perceive(self, observation: np.ndarray) -> np.ndarray:
        """
        Update beliefs based on a new observation using variational inference.

        Minimizes variational free energy:
        F = D_KL[q(s) || p(s)] - E_q[log p(o|s)]

        For Gaussian beliefs, this reduces to gradient descent on
        the belief parameters (mu, Sigma) given the observation.

        Parameters
        ----------
        observation : np.ndarray
            Observation vector, shape (obs_dim,) or (state_dim,) for
            direct state observation.

        Returns
        -------
        np.ndarray
            Updated belief mean, shape (state_dim,).
        """
        if self._use_gpu:
            return self._perceive_gpu(observation)

        observation = np.asarray(observation, dtype=np.float64)

        if observation.shape[0] == self.state_dim and self.state_dim < self.obs_dim:
            obs_state = np.zeros(self.obs_dim, dtype=np.float64)
            obs_state[:observation.shape[0]] = observation
        elif observation.shape[0] == self.state_dim:
            obs_state = observation
        elif observation.shape[0] == self.obs_dim:
            obs_state = observation
        else:
            obs_state = observation[:self.state_dim]

        for _ in range(self.n_belief_steps):
            pred_obs = self.observation_net.forward(self.belief_mu)
            pred_mu = pred_obs[:self.obs_dim]
            pred_logvar = pred_obs[self.obs_dim:]

            pred_logvar = np.clip(pred_logvar, -10, 10)
            pred_var = np.exp(pred_logvar)

            obs_residual = obs_state[:self.obs_dim] - pred_mu
            obs_grad = obs_residual / (pred_var + 1e-8)

            # Compute Jacobian of observation network using finite differences
            # so that grad_mu = J_obs^T @ obs_grad (chain rule through obs network)
            eps_jac = 1e-5
            J_obs = np.zeros((self.obs_dim, self.state_dim), dtype=np.float64)
            for j in range(self.state_dim):
                e_j = np.zeros(self.state_dim, dtype=np.float64)
                e_j[j] = eps_jac
                obs_plus = self.observation_net.forward(self.belief_mu + e_j)[:self.obs_dim]
                obs_minus = self.observation_net.forward(self.belief_mu - e_j)[:self.obs_dim]
                J_obs[:, j] = (obs_plus - obs_minus) / (2.0 * eps_jac)

            grad_mu = J_obs.T @ obs_grad

            prior_residual = self.belief_mu - self._prior_mu
            grad_mu -= np.linalg.solve(self._prior_Sigma, prior_residual)

            grad_norm = np.linalg.norm(grad_mu)
            if grad_norm > 1.0:
                grad_mu = grad_mu * (1.0 / grad_norm)
            self.belief_mu += self.belief_lr * grad_mu

            belief_norm = np.linalg.norm(self.belief_mu)
            if belief_norm > 10.0:
                self.belief_mu = self.belief_mu * (10.0 / belief_norm)

            # Bayesian posterior covariance update:
            # Sigma_posterior = (J_obs^T @ Sigma_obs^{-1} @ J_obs + Sigma_prior^{-1})^{-1}
            Sigma_obs_inv = np.diag(1.0 / np.clip(pred_var, 1e-8, 100.0))
            try:
                Sigma_prior_inv = np.linalg.inv(self._prior_Sigma)
                posterior_precision = J_obs.T @ Sigma_obs_inv @ J_obs + Sigma_prior_inv
                Sigma_posterior = np.linalg.inv(posterior_precision)
                Sigma_posterior = 0.5 * (Sigma_posterior + Sigma_posterior.T)
            except np.linalg.LinAlgError:
                Sigma_posterior = self._prior_Sigma
            self.belief_Sigma = Sigma_posterior

        return self.belief_mu.copy()

    def predict_transition(
        self, state: np.ndarray, action: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict next state distribution given current state and action.

        Parameters
        ----------
        state : np.ndarray
            Current state, shape (state_dim,).
        action : int
            Action index.

        Returns
        -------
        next_mu : np.ndarray
            Predicted next state mean, shape (state_dim,).
        next_Sigma : np.ndarray
            Predicted next state covariance, shape (state_dim, state_dim).
        """
        action_vec = np.zeros(self.n_actions, dtype=np.float64)
        action_vec[action] = 1.0
        sa = np.concatenate([state, action_vec])

        output = self.transition_net.forward(sa)
        next_mu = output[:self.state_dim]
        next_logvar = output[self.state_dim:]
        next_logvar = np.clip(next_logvar, -10, 10)
        next_var = np.exp(next_logvar)
        next_Sigma = np.diag(np.clip(next_var, 1e-8, 100.0))

        return next_mu, next_Sigma

    def predict_observation(
        self, state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict observation distribution given state.

        Parameters
        ----------
        state : np.ndarray
            State vector, shape (state_dim,).

        Returns
        -------
        obs_mu : np.ndarray
            Predicted observation mean, shape (obs_dim,).
        obs_Sigma : np.ndarray
            Predicted observation covariance, shape (obs_dim, obs_dim).
        """
        output = self.observation_net.forward(state)
        obs_mu = output[:self.obs_dim]
        obs_logvar = output[self.obs_dim:]
        obs_logvar = np.clip(obs_logvar, -10, 10)
        obs_var = np.exp(obs_logvar)
        obs_Sigma = np.diag(np.clip(obs_var, 1e-8, 100.0))

        return obs_mu, obs_Sigma

    def expected_free_energy(
        self, action: int
    ) -> Tuple[float, float, float]:
        """
        Compute expected free energy for a single action.

        G(a) = -epistemic_value - pragmatic_value

        Epistemic value (information gain):
            E_q[D_KL[q(s'|o,a) || q(s'|a)]]
            Approximated by the expected reduction in belief entropy.

        Pragmatic value (goal seeking):
            E_q[log p(o)]
            Approximated by the negative KL divergence between
            predicted observations and preferences.

        Parameters
        ----------
        action : int
            Action index to evaluate.

        Returns
        -------
        quality : float
            Total quality (negative expected free energy).
        epistemic_value : float
            Expected information gain.
        pragmatic_value : float
            Expected preference satisfaction.
        """
        next_mu, next_Sigma = self.predict_transition(self.belief_mu, action)

        obs_mu, obs_Sigma = self.predict_observation(next_mu)

        epistemic_value = self._compute_epistemic_value(next_mu, next_Sigma)

        pragmatic_value = self._compute_pragmatic_value(obs_mu, obs_Sigma)

        quality = epistemic_value + pragmatic_value

        return quality, epistemic_value, pragmatic_value

    def _compute_epistemic_value(
        self,
        next_mu: np.ndarray,
        next_Sigma: np.ndarray,
    ) -> float:
        """
        Compute epistemic value as expected information gain.

        In continuous state spaces, this is the mutual information:
            I(s'; o) = H[p(s'|a)] - E_o[H[q(s'|o,a)]]

        Under Gaussian assumptions with a linearised observation model:
            Prior:       p(s'|a) = N(next_mu, next_Sigma)
            Posterior:   q(s'|o,a) ≈ N(next_mu, Sigma_posterior)
            where Sigma_posterior = (next_Sigma^{-1} + J^T Sigma_obs^{-1} J)^{-1}
            and J is the Jacobian of the observation model.

        The information gain is then:
            I = 0.5 * log(det(next_Sigma) / det(Sigma_posterior))

        Parameters
        ----------
        next_mu : np.ndarray
            Predicted next state mean.
        next_Sigma : np.ndarray
            Predicted next state covariance (transition prior).

        Returns
        -------
        float
            Epistemic value (non-negative).
        """
        eps = 1e-16

        # Prior entropy from transition prior (next_Sigma), not current belief
        sign_prior, logdet_prior = np.linalg.slogdet(next_Sigma)
        if sign_prior <= 0:
            jitter = 1e-6
            for _ in range(5):
                sign_prior, logdet_prior = np.linalg.slogdet(
                    next_Sigma + jitter * np.eye(self.state_dim, dtype=np.float64)
                )
                if sign_prior > 0:
                    break
                jitter *= 10
        prior_entropy = 0.5 * logdet_prior + 0.5 * self.state_dim * np.log(2.0 * np.pi * np.e)

        # Approximate posterior by incorporating observation precision
        # via Bayesian update with linearised observation model
        obs_output = self.observation_net.forward(next_mu)
        obs_logvar = obs_output[self.obs_dim:]
        obs_logvar = np.clip(obs_logvar, -10, 10)
        obs_var = np.exp(obs_logvar)
        obs_Sigma = np.diag(np.clip(obs_var, 1e-8, 100.0))

        # Finite-difference Jacobian: J[i,j] = d(obs_i)/d(state_j)
        eps_jac = 1e-5
        J_obs = np.zeros((self.obs_dim, self.state_dim), dtype=np.float64)
        for j in range(self.state_dim):
            e_j = np.zeros(self.state_dim, dtype=np.float64)
            e_j[j] = eps_jac
            obs_plus = self.observation_net.forward(next_mu + e_j)[:self.obs_dim]
            obs_minus = self.observation_net.forward(next_mu - e_j)[:self.obs_dim]
            J_obs[:, j] = (obs_plus - obs_minus) / (2.0 * eps_jac)

        # Bayesian update: posterior_precision = prior_precision + J^T obs_Sigma^{-1} J
        try:
            next_Sigma_inv = np.linalg.inv(next_Sigma)
            obs_Sigma_inv = np.linalg.inv(obs_Sigma)
            posterior_precision = next_Sigma_inv + J_obs.T @ obs_Sigma_inv @ J_obs
            Sigma_posterior = np.linalg.inv(posterior_precision)
            # Ensure symmetry
            Sigma_posterior = 0.5 * (Sigma_posterior + Sigma_posterior.T)
        except np.linalg.LinAlgError:
            Sigma_posterior = next_Sigma

        sign_post, logdet_post = np.linalg.slogdet(Sigma_posterior)
        if sign_post <= 0:
            jitter = 1e-6
            for _ in range(5):
                sign_post, logdet_post = np.linalg.slogdet(
                    Sigma_posterior + jitter * np.eye(self.state_dim, dtype=np.float64)
                )
                if sign_post > 0:
                    break
                jitter *= 10
        posterior_entropy = 0.5 * logdet_post + 0.5 * self.state_dim * np.log(2.0 * np.pi * np.e)

        info_gain = prior_entropy - posterior_entropy
        return max(0.0, info_gain)

    def _compute_pragmatic_value(
        self,
        obs_mu: np.ndarray,
        obs_Sigma: np.ndarray,
    ) -> float:
        """
        Compute pragmatic value as preference satisfaction.

        In continuous space, this is the negative KL divergence
        between predicted observations and preferences:
            -D_KL[N(obs_mu, obs_Sigma) || N(pref_mu, pref_Sigma)]

        Parameters
        ----------
        obs_mu : np.ndarray
            Predicted observation mean.
        obs_Sigma : np.ndarray
            Predicted observation covariance.

        Returns
        -------
        float
            Pragmatic value.
        """
        d = min(obs_mu.shape[0], self.preference_mu.shape[0])

        obs_mu_d = obs_mu[:d]
        pref_mu_d = self.preference_mu[:d]
        obs_Sigma_d = obs_Sigma[:d, :d]
        pref_Sigma_d = self.preference_Sigma[:d, :d]

        diff = obs_mu_d - pref_mu_d

        try:
            pref_Sigma_inv = np.linalg.inv(pref_Sigma_d)
        except np.linalg.LinAlgError:
            pref_Sigma_inv = np.eye(d)

        try:
            log_det_pref = np.linalg.slogdet(pref_Sigma_d)[1]
            log_det_obs = np.linalg.slogdet(obs_Sigma_d)[1]
        except np.linalg.LinAlgError:
            log_det_pref = 0.0
            log_det_obs = 0.0

        kl = 0.5 * (
            np.trace(pref_Sigma_inv @ obs_Sigma_d)
            + diff @ pref_Sigma_inv @ diff
            - d
            + log_det_pref
            - log_det_obs
        )

        return -kl

    # ------------------------------------------------------------------
    # GPU-accelerated methods
    # ------------------------------------------------------------------

    def _sync_torch_weights(self) -> None:
        """Sync numpy MLP weights to torch models for GPU computation."""
        if not self._use_gpu or self._trans_net_torch is None:
            return
        self._trans_net_torch.sync_from_numpy(
            self.transition_net.weights, self.transition_net.biases
        )
        self._obs_net_torch.sync_from_numpy(
            self.observation_net.weights, self.observation_net.biases
        )

    def _perceive_gpu(self, observation: np.ndarray) -> np.ndarray:
        """
        GPU-accelerated belief update using torch autograd for Jacobian.

        Mirrors the CPU perceive() logic but uses torch tensors and
        torch.autograd.functional.jacobian instead of finite differences.
        """
        observation = np.asarray(observation, dtype=np.float64)

        if observation.shape[0] == self.state_dim and self.state_dim < self.obs_dim:
            obs_state = np.zeros(self.obs_dim, dtype=np.float64)
            obs_state[:observation.shape[0]] = observation
        elif observation.shape[0] == self.state_dim:
            obs_state = observation
        elif observation.shape[0] == self.obs_dim:
            obs_state = observation
        else:
            obs_state = observation[:self.state_dim]

        self._sync_torch_weights()

        obs_t = torch.as_tensor(
            obs_state[:self.obs_dim], dtype=torch.float64, device=self.device
        )
        belief_t = torch.as_tensor(
            self.belief_mu, dtype=torch.float64, device=self.device
        )
        prior_mu_t = torch.as_tensor(
            self._prior_mu, dtype=torch.float64, device=self.device
        )
        prior_Sigma_t = torch.as_tensor(
            self._prior_Sigma, dtype=torch.float64, device=self.device
        )

        for _ in range(self.n_belief_steps):
            # Forward pass through observation net
            pred_obs = self._obs_net_torch(belief_t)
            pred_mu = pred_obs[:self.obs_dim]
            pred_logvar = pred_obs[self.obs_dim:]
            pred_logvar = torch.clamp(pred_logvar, -10, 10)
            pred_var = torch.exp(pred_logvar)

            obs_residual = obs_t - pred_mu
            obs_grad = obs_residual / (pred_var + 1e-8)

            # Compute Jacobian via torch autograd (much more efficient
            # than finite differences)
            def _obs_fn(s: "torch.Tensor") -> "torch.Tensor":
                return self._obs_net_torch(s)[:self.obs_dim]

            J_obs = torch.autograd.functional.jacobian(_obs_fn, belief_t)

            grad_mu = J_obs.T @ obs_grad

            prior_residual = belief_t - prior_mu_t
            grad_mu -= torch.linalg.solve(prior_Sigma_t, prior_residual)

            grad_norm = torch.linalg.norm(grad_mu)
            if grad_norm > 1.0:
                grad_mu = grad_mu * (1.0 / grad_norm)
            belief_t = (belief_t + self.belief_lr * grad_mu).detach()

            belief_norm = torch.linalg.norm(belief_t)
            if belief_norm > 10.0:
                belief_t = belief_t * (10.0 / belief_norm)

            # Bayesian posterior covariance update:
            # Sigma_posterior = (J_obs^T @ Sigma_obs^{-1} @ J_obs + Sigma_prior^{-1})^{-1}
            Sigma_obs_inv = torch.diag(1.0 / torch.clamp(pred_var, min=1e-8, max=100.0))
            try:
                prior_Sigma_inv = torch.linalg.inv(prior_Sigma_t)
                posterior_precision = J_obs.T @ Sigma_obs_inv @ J_obs + prior_Sigma_inv
                Sigma_posterior_t = torch.linalg.inv(posterior_precision)
                Sigma_posterior_t = 0.5 * (Sigma_posterior_t + Sigma_posterior_t.T)
            except RuntimeError:
                Sigma_posterior_t = prior_Sigma_t

        self.belief_mu = belief_t.detach().cpu().numpy()
        self.belief_Sigma = Sigma_posterior_t.detach().cpu().numpy()

        return self.belief_mu.copy()

    def _compute_epistemic_value_gpu(
        self,
        next_mu_t: "torch.Tensor",
        next_Sigma_t: "torch.Tensor",
    ) -> float:
        """
        GPU-accelerated epistemic value computation.

        Uses torch.linalg.slogdet for numerically stable
        log-determinant and torch.autograd.functional.jacobian
        for the observation model Jacobian.
        """
        eps = 1e-16

        # Prior entropy from transition prior
        sign_prior, logdet_prior = torch.linalg.slogdet(next_Sigma_t)
        if sign_prior <= 0:
            jitter = 1e-6
            eye_t = torch.eye(self.state_dim, dtype=torch.float64, device=self.device)
            for _ in range(5):
                sign_prior, logdet_prior = torch.linalg.slogdet(
                    next_Sigma_t + jitter * eye_t
                )
                if sign_prior > 0:
                    break
                jitter *= 10
        prior_entropy = (
            0.5 * logdet_prior
            + 0.5 * self.state_dim * np.log(2.0 * np.pi * np.e)
        )

        # Observation model at next_mu
        obs_output = self._obs_net_torch(next_mu_t)
        obs_logvar = obs_output[self.obs_dim:]
        obs_logvar = torch.clamp(obs_logvar, -10, 10)
        obs_var = torch.exp(obs_logvar)
        obs_Sigma_t = torch.diag(torch.clamp(obs_var, min=1e-8, max=100.0))

        # Jacobian via torch autograd
        def _obs_fn(s: "torch.Tensor") -> "torch.Tensor":
            return self._obs_net_torch(s)[:self.obs_dim]

        J_obs = torch.autograd.functional.jacobian(_obs_fn, next_mu_t)

        # Bayesian update: posterior_precision = prior_precision + J^T obs_Sigma^{-1} J
        try:
            next_Sigma_inv = torch.linalg.inv(next_Sigma_t)
            obs_Sigma_inv = torch.linalg.inv(obs_Sigma_t)
            posterior_precision = next_Sigma_inv + J_obs.T @ obs_Sigma_inv @ J_obs
            Sigma_posterior = torch.linalg.inv(posterior_precision)
            Sigma_posterior = 0.5 * (Sigma_posterior + Sigma_posterior.T)
        except RuntimeError:
            Sigma_posterior = next_Sigma_t

        sign_post, logdet_post = torch.linalg.slogdet(Sigma_posterior)
        if sign_post <= 0:
            jitter = 1e-6
            eye_t = torch.eye(self.state_dim, dtype=torch.float64, device=self.device)
            for _ in range(5):
                sign_post, logdet_post = torch.linalg.slogdet(
                    Sigma_posterior + jitter * eye_t
                )
                if sign_post > 0:
                    break
                jitter *= 10
        posterior_entropy = (
            0.5 * logdet_post
            + 0.5 * self.state_dim * np.log(2.0 * np.pi * np.e)
        )

        info_gain = prior_entropy - posterior_entropy
        return max(0.0, info_gain.item())

    def _compute_pragmatic_value_gpu(
        self,
        obs_mu_t: "torch.Tensor",
        obs_Sigma_t: "torch.Tensor",
    ) -> float:
        """
        GPU-accelerated pragmatic value computation.

        Uses torch for KL divergence between predicted observations
        and preferences.
        """
        d = min(obs_mu_t.shape[0], self.preference_mu.shape[0])

        obs_mu_d = obs_mu_t[:d]
        pref_mu_d = torch.as_tensor(
            self.preference_mu[:d], dtype=torch.float64, device=self.device
        )
        obs_Sigma_d = obs_Sigma_t[:d, :d]
        pref_Sigma_d = torch.as_tensor(
            self.preference_Sigma[:d, :d], dtype=torch.float64, device=self.device
        )

        diff = obs_mu_d - pref_mu_d

        try:
            pref_Sigma_inv = torch.linalg.inv(pref_Sigma_d)
        except RuntimeError:
            pref_Sigma_inv = torch.eye(
                d, dtype=torch.float64, device=self.device
            )

        log_det_pref = torch.linalg.slogdet(pref_Sigma_d)[1]
        log_det_obs = torch.linalg.slogdet(obs_Sigma_d)[1]

        kl = 0.5 * (
            torch.trace(pref_Sigma_inv @ obs_Sigma_d)
            + diff @ pref_Sigma_inv @ diff
            - d
            + log_det_pref
            - log_det_obs
        )

        return -kl.item()

    def _select_action_gpu(self) -> Tuple[int, np.ndarray, np.ndarray]:
        """GPU-accelerated action selection."""
        self._sync_torch_weights()

        if self.planning_horizon == 1:
            return self._select_action_single_step_gpu()

        return self._select_action_multi_step_gpu()

    def _select_action_single_step_gpu(
        self,
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """GPU-accelerated single-step action selection."""
        qualities = np.zeros(self.n_actions, dtype=np.float64)
        epistemic_values = np.zeros(self.n_actions, dtype=np.float64)
        pragmatic_values = np.zeros(self.n_actions, dtype=np.float64)

        for a in range(self.n_actions):
            action_vec = np.zeros(self.n_actions, dtype=np.float64)
            action_vec[a] = 1.0
            sa = np.concatenate([self.belief_mu, action_vec])
            sa_t = torch.as_tensor(sa, dtype=torch.float64, device=self.device)

            # Transition prediction on GPU
            trans_output = self._trans_net_torch(sa_t)
            next_mu_t = trans_output[:self.state_dim]
            next_logvar_t = trans_output[self.state_dim:]
            next_logvar_t = torch.clamp(next_logvar_t, -10, 10)
            next_var_t = torch.exp(next_logvar_t)
            next_Sigma_t = torch.diag(
                torch.clamp(next_var_t, min=1e-8, max=100.0)
            )

            # Observation prediction on GPU
            obs_output = self._obs_net_torch(next_mu_t)
            obs_mu_t = obs_output[:self.obs_dim]
            obs_logvar_t = obs_output[self.obs_dim:]
            obs_logvar_t = torch.clamp(obs_logvar_t, -10, 10)
            obs_var_t = torch.exp(obs_logvar_t)
            obs_Sigma_t = torch.diag(
                torch.clamp(obs_var_t, min=1e-8, max=100.0)
            )

            epistemic_values[a] = self._compute_epistemic_value_gpu(
                next_mu_t, next_Sigma_t
            )
            pragmatic_values[a] = self._compute_pragmatic_value_gpu(
                obs_mu_t, obs_Sigma_t
            )
            qualities[a] = epistemic_values[a] + pragmatic_values[a]

        scaled = qualities * self.action_precision / max(self.temperature, 1e-8)
        scaled -= np.max(scaled)
        action_probs = np.exp(scaled)
        action_probs /= action_probs.sum()

        action = int(np.random.choice(self.n_actions, p=action_probs))

        return action, qualities, action_probs

    def _select_action_multi_step_gpu(
        self,
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """GPU-accelerated multi-step action selection with open-loop rollout."""
        n_policies = self.n_actions ** self.planning_horizon

        if n_policies > 256:
            return self._select_action_single_step_gpu()

        policy_qualities = np.zeros(n_policies, dtype=np.float64)
        policy_sequences: List[List[int]] = []

        for pi_idx in range(n_policies):
            sequence = []
            temp = pi_idx
            for _ in range(self.planning_horizon):
                sequence.append(temp % self.n_actions)
                temp //= self.n_actions
            sequence.reverse()
            policy_sequences.append(sequence)

            cumulative_quality = 0.0
            current_mu_t = torch.as_tensor(
                self.belief_mu, dtype=torch.float64, device=self.device
            )
            current_Sigma_t = torch.as_tensor(
                self.belief_Sigma, dtype=torch.float64, device=self.device
            )

            for t, act in enumerate(sequence):
                action_vec = np.zeros(self.n_actions, dtype=np.float64)
                action_vec[act] = 1.0
                current_mu_np = current_mu_t.detach().cpu().numpy()
                sa = np.concatenate([current_mu_np, action_vec])
                sa_t = torch.as_tensor(sa, dtype=torch.float64, device=self.device)

                trans_output = self._trans_net_torch(sa_t)
                next_mu_t = trans_output[:self.state_dim]
                next_logvar_t = trans_output[self.state_dim:]
                next_logvar_t = torch.clamp(next_logvar_t, -10, 10)
                next_var_t = torch.exp(next_logvar_t)
                next_Sigma_t = torch.diag(
                    torch.clamp(next_var_t, min=1e-8, max=100.0)
                )

                # Compute Jacobian of transition model for covariance propagation
                def _trans_fn(s: "torch.Tensor") -> "torch.Tensor":
                    sa_inner = torch.cat([s, torch.as_tensor(
                        action_vec, dtype=torch.float64, device=self.device
                    )])
                    return self._trans_net_torch(sa_inner)[:self.state_dim]

                J_trans = torch.autograd.functional.jacobian(
                    _trans_fn, current_mu_t.detach()
                )

                # Propagate covariance: Sigma' = J @ Sigma @ J^T + Sigma_trans
                current_Sigma_t = J_trans @ current_Sigma_t @ J_trans.T + next_Sigma_t

                obs_output = self._obs_net_torch(next_mu_t)
                obs_mu_t = obs_output[:self.obs_dim]
                obs_logvar_t = obs_output[self.obs_dim:]
                obs_logvar_t = torch.clamp(obs_logvar_t, -10, 10)
                obs_var_t = torch.exp(obs_logvar_t)
                obs_Sigma_t = torch.diag(
                    torch.clamp(obs_var_t, min=1e-8, max=100.0)
                )

                epistemic = self._compute_epistemic_value_gpu(
                    next_mu_t, current_Sigma_t
                )
                pragmatic = self._compute_pragmatic_value_gpu(
                    obs_mu_t, obs_Sigma_t
                )

                discount = 1.0 / (1.0 + t)
                cumulative_quality += discount * (epistemic + pragmatic)

                current_mu_t = next_mu_t

            policy_qualities[pi_idx] = cumulative_quality

        action_qualities = np.full(self.n_actions, -np.inf, dtype=np.float64)
        for pi_idx, sequence in enumerate(policy_sequences):
            first_action = sequence[0]
            action_qualities[first_action] = max(
                action_qualities[first_action], policy_qualities[pi_idx]
            )

        action_qualities = np.where(
            np.isinf(action_qualities), 0.0, action_qualities
        )

        scaled = action_qualities * self.action_precision / max(self.temperature, 1e-8)
        scaled -= np.max(scaled)
        action_probs = np.exp(scaled)
        action_probs /= action_probs.sum()

        action = int(np.random.choice(self.n_actions, p=action_probs))

        return action, action_qualities, action_probs

    def select_action(self) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Select an action based on expected free energy.

        If planning_horizon > 1, evaluates multi-step action sequences
        by recursively predicting future states and accumulating
        expected free energy over the planning horizon.

        Returns
        -------
        action : int
            Selected action index.
        qualities : np.ndarray
            Quality values for each action.
        action_probs : np.ndarray
            Probability of selecting each action.
        """
        if self._use_gpu:
            return self._select_action_gpu()

        if self.planning_horizon == 1:
            return self._select_action_single_step()

        return self._select_action_multi_step()

    def _select_action_single_step(self) -> Tuple[int, np.ndarray, np.ndarray]:
        """Select action using single-step expected free energy."""
        qualities = np.zeros(self.n_actions, dtype=np.float64)
        epistemic_values = np.zeros(self.n_actions, dtype=np.float64)
        pragmatic_values = np.zeros(self.n_actions, dtype=np.float64)

        for a in range(self.n_actions):
            q, ev, pv = self.expected_free_energy(a)
            qualities[a] = q
            epistemic_values[a] = ev
            pragmatic_values[a] = pv

        scaled = qualities * self.action_precision / max(self.temperature, 1e-8)
        scaled -= np.max(scaled)
        action_probs = np.exp(scaled)
        action_probs /= action_probs.sum()

        action = int(np.random.choice(self.n_actions, p=action_probs))

        return action, qualities, action_probs

    def _select_action_multi_step(self) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Select action using multi-step planning with open-loop rollout.

        Evaluates action sequences up to planning_horizon steps.
        For computational tractability, limits the number of
        evaluated policies and uses open-loop prediction.
        """
        n_policies = self.n_actions ** self.planning_horizon

        if n_policies > 256:
            return self._select_action_single_step()

        policy_qualities = np.zeros(n_policies, dtype=np.float64)
        policy_sequences: List[List[int]] = []

        for pi_idx in range(n_policies):
            sequence = []
            temp = pi_idx
            for _ in range(self.planning_horizon):
                sequence.append(temp % self.n_actions)
                temp //= self.n_actions
            sequence.reverse()
            policy_sequences.append(sequence)

            cumulative_quality = 0.0
            current_mu = self.belief_mu.copy()
            current_Sigma = self.belief_Sigma.copy()

            for t, act in enumerate(sequence):
                next_mu, next_Sigma = self.predict_transition(current_mu, act)
                obs_mu, obs_Sigma = self.predict_observation(next_mu)

                # Compute Jacobian of transition model for covariance propagation
                eps_jac = 1e-5
                J_trans = np.zeros((self.state_dim, self.state_dim), dtype=np.float64)
                action_vec = np.zeros(self.n_actions, dtype=np.float64)
                action_vec[act] = 1.0
                for j in range(self.state_dim):
                    e_j = np.zeros(self.state_dim, dtype=np.float64)
                    e_j[j] = eps_jac
                    sa_plus = np.concatenate([current_mu + e_j, action_vec])
                    sa_minus = np.concatenate([current_mu - e_j, action_vec])
                    trans_plus = self.transition_net.forward(sa_plus)[:self.state_dim]
                    trans_minus = self.transition_net.forward(sa_minus)[:self.state_dim]
                    J_trans[:, j] = (trans_plus - trans_minus) / (2.0 * eps_jac)

                # Propagate covariance: Sigma' = J @ Sigma @ J^T + Sigma_trans
                current_Sigma = J_trans @ current_Sigma @ J_trans.T + next_Sigma

                epistemic = self._compute_epistemic_value(next_mu, current_Sigma)
                pragmatic = self._compute_pragmatic_value(obs_mu, obs_Sigma)

                discount = 1.0 / (1.0 + t)
                cumulative_quality += discount * (epistemic + pragmatic)

                current_mu = next_mu

            policy_qualities[pi_idx] = cumulative_quality

        action_qualities = np.full(self.n_actions, -np.inf, dtype=np.float64)
        for pi_idx, sequence in enumerate(policy_sequences):
            first_action = sequence[0]
            action_qualities[first_action] = max(
                action_qualities[first_action], policy_qualities[pi_idx]
            )

        action_qualities = np.where(
            np.isinf(action_qualities), 0.0, action_qualities
        )

        scaled = action_qualities * self.action_precision / max(self.temperature, 1e-8)
        scaled -= np.max(scaled)
        action_probs = np.exp(scaled)
        action_probs /= action_probs.sum()

        action = int(np.random.choice(self.n_actions, p=action_probs))

        return action, action_qualities, action_probs

    def learn_transition(
        self,
        prev_state: np.ndarray,
        action: int,
        next_state: np.ndarray,
    ) -> Dict[str, float]:
        """
        Update the transition model based on observed state transition.

        Uses gradient descent on the prediction error of the
        transition network, minimizing:
            L = ||next_state - mu_trans(prev_state, action)||^2
              + kl_weight * D_KL[q(z) || p(z)]

        Parameters
        ----------
        prev_state : np.ndarray
            Previous state, shape (state_dim,).
        action : int
            Action taken.
        next_state : np.ndarray
            Observed next state, shape (state_dim,).

        Returns
        -------
        dict
            Learning metrics including loss and gradient norm.
        """
        prev_state = np.asarray(prev_state, dtype=np.float64)
        next_state = np.asarray(next_state, dtype=np.float64)

        if prev_state.shape[0] != self.state_dim:
            prev_state = prev_state[:self.state_dim]
            if prev_state.shape[0] < self.state_dim:
                padded = np.zeros(self.state_dim, dtype=np.float64)
                padded[:prev_state.shape[0]] = prev_state
                prev_state = padded

        if next_state.shape[0] != self.state_dim:
            next_state = next_state[:self.state_dim]
            if next_state.shape[0] < self.state_dim:
                padded = np.zeros(self.state_dim, dtype=np.float64)
                padded[:next_state.shape[0]] = next_state
                next_state = padded

        action_vec = np.zeros(self.n_actions, dtype=np.float64)
        action_vec[action] = 1.0
        sa = np.concatenate([prev_state, action_vec])

        output = self.transition_net.forward(sa)
        pred_mu = output[:self.state_dim]
        pred_logvar = output[self.state_dim:]
        pred_logvar = np.clip(pred_logvar, -10, 10)
        pred_var = np.exp(pred_logvar)

        residual = next_state - pred_mu
        nll = 0.5 * np.sum(residual ** 2 / (pred_var + 1e-8) + pred_logvar)

        grad_output = np.zeros(self.state_dim * 2, dtype=np.float64)
        grad_output[:self.state_dim] = -residual / (pred_var + 1e-8)
        grad_output[self.state_dim:] = 0.5 * (1.0 - residual ** 2 / (pred_var + 1e-8))
        grad_norm = np.linalg.norm(grad_output)
        if grad_norm > 1.0:
            grad_output = grad_output * (1.0 / grad_norm)

        self.transition_net.backward(sa, grad_output)

        old_prior_mu = self._prior_mu.copy()
        self._prior_mu = 0.99 * self._prior_mu + 0.01 * prev_state
        self._prior_Sigma = 0.99 * self._prior_Sigma + 0.01 * np.outer(
            prev_state - old_prior_mu, prev_state - old_prior_mu
        )

        return {
            "transition_loss": float(nll),
            "prediction_error": float(np.linalg.norm(residual)),
        }

    def learn_observation(
        self,
        state: np.ndarray,
        observation: np.ndarray,
    ) -> Dict[str, float]:
        """
        Update the observation model based on state-observation pair.

        Parameters
        ----------
        state : np.ndarray
            State vector, shape (state_dim,).
        observation : np.ndarray
            Observation vector, shape (obs_dim,).

        Returns
        -------
        dict
            Learning metrics.
        """
        state = np.asarray(state, dtype=np.float64)
        observation = np.asarray(observation, dtype=np.float64)

        if state.shape[0] != self.state_dim:
            state = state[:self.state_dim]
            if state.shape[0] < self.state_dim:
                padded = np.zeros(self.state_dim, dtype=np.float64)
                padded[:state.shape[0]] = state
                state = padded

        if observation.shape[0] != self.obs_dim:
            observation = observation[:self.obs_dim]
            if observation.shape[0] < self.obs_dim:
                padded = np.zeros(self.obs_dim, dtype=np.float64)
                padded[:observation.shape[0]] = observation
                observation = padded

        output = self.observation_net.forward(state)
        pred_mu = output[:self.obs_dim]
        pred_logvar = output[self.obs_dim:]
        pred_logvar = np.clip(pred_logvar, -10, 10)
        pred_var = np.exp(pred_logvar)

        residual = observation - pred_mu
        nll = 0.5 * np.sum(residual ** 2 / (pred_var + 1e-8) + pred_logvar)

        grad_output = np.zeros(self.obs_dim * 2, dtype=np.float64)
        grad_output[:self.obs_dim] = -residual / (pred_var + 1e-8)
        grad_output[self.obs_dim:] = 0.5 * (1.0 - residual ** 2 / (pred_var + 1e-8))
        grad_norm = np.linalg.norm(grad_output)
        if grad_norm > 1.0:
            grad_output = grad_output * (1.0 / grad_norm)

        self.observation_net.backward(state, grad_output)

        return {
            "observation_loss": float(nll),
            "prediction_error": float(np.linalg.norm(residual)),
        }

    def step(self, observation: np.ndarray) -> int:
        """
        Execute one complete perception-action cycle.

        1. Perceive: Update beliefs based on observation
        2. Select: Choose action based on expected free energy

        Parameters
        ----------
        observation : np.ndarray
            Current observation vector.

        Returns
        -------
        int
            Selected action index.
        """
        self.perceive(observation)
        action, qualities, action_probs = self.select_action()

        self.history.append({
            "step": self.step_count,
            "belief_mu": self.belief_mu.copy(),
            "action": action,
            "qualities": qualities.copy(),
            "action_probs": action_probs.copy(),
        })

        self.step_count += 1
        return action

    def get_belief_entropy(self) -> float:
        """
        Compute the entropy of the current Gaussian belief.

        Returns
        -------
        float
            Belief entropy. Higher values indicate more uncertainty.
        """
        sign, logdet = np.linalg.slogdet(self.belief_Sigma)
        if sign <= 0:
            logdet = 0.0
        return 0.5 * (self.state_dim * np.log(2.0 * np.pi * np.e) + logdet)

    def set_preferences(
        self,
        preference_mu: np.ndarray,
        preference_Sigma: Optional[np.ndarray] = None,
    ) -> None:
        """
        Set observation preferences as a Gaussian distribution.

        Parameters
        ----------
        preference_mu : np.ndarray
            Preferred observation mean, shape (obs_dim,).
        preference_Sigma : np.ndarray or None
            Preferred observation covariance. If None, uses identity.
        """
        preference_mu = np.asarray(preference_mu, dtype=np.float64)
        if preference_mu.shape[0] != self.obs_dim:
            padded = np.zeros(self.obs_dim, dtype=np.float64)
            n = min(preference_mu.shape[0], self.obs_dim)
            padded[:n] = preference_mu[:n]
            preference_mu = padded

        self.preference_mu = preference_mu
        if preference_Sigma is not None:
            self.preference_Sigma = np.asarray(preference_Sigma, dtype=np.float64)
        else:
            self.preference_Sigma = np.eye(self.obs_dim, dtype=np.float64)

    def reset(self) -> None:
        """Reset the agent to initial state."""
        self.belief_mu = np.zeros(self.state_dim, dtype=np.float64)
        self.belief_Sigma = np.eye(self.state_dim, dtype=np.float64)
        self.history = []
        self.step_count = 0

    def state_dict(self) -> Dict[str, Any]:
        """Return the agent state for serialization.

        All values are returned as numpy arrays for portability,
        even when GPU mode is active.
        """
        state: Dict[str, Any] = {
            "belief_mu": self.belief_mu.copy(),
            "belief_Sigma": self.belief_Sigma.copy(),
            "transition_net_params": self.transition_net.get_params(),
            "observation_net_params": self.observation_net.get_params(),
            "preference_mu": self.preference_mu.copy(),
            "preference_Sigma": self.preference_Sigma.copy(),
            "prior_mu": self._prior_mu.copy(),
            "prior_Sigma": self._prior_Sigma.copy(),
            "step_count": self.step_count,
        }
        # Convert any torch tensors to numpy for portability
        if _TORCH_AVAILABLE:
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.detach().cpu().numpy()
                elif isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, torch.Tensor):
                            value[k] = v.detach().cpu().numpy()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load agent state from a dictionary.

        Handles conversion from torch tensors to numpy when needed,
        and syncs torch models after loading if GPU mode is active.
        """
        def _to_numpy(v: Any) -> np.ndarray:
            if _TORCH_AVAILABLE and isinstance(v, torch.Tensor):
                return v.detach().cpu().numpy().astype(np.float64)
            return np.asarray(v, dtype=np.float64)

        self.belief_mu = _to_numpy(state["belief_mu"])
        self.belief_Sigma = _to_numpy(state["belief_Sigma"])

        trans_params = state["transition_net_params"]
        obs_params = state["observation_net_params"]

        # Convert torch tensors to numpy if present
        for params in [trans_params, obs_params]:
            for k in list(params.keys()):
                params[k] = _to_numpy(params[k])

        self.transition_net.set_params(trans_params)
        self.observation_net.set_params(obs_params)

        self.preference_mu = _to_numpy(state["preference_mu"])
        self.preference_Sigma = _to_numpy(state["preference_Sigma"])
        self._prior_mu = _to_numpy(state["prior_mu"])
        self._prior_Sigma = _to_numpy(state["prior_Sigma"])
        self.step_count = state["step_count"]

        # Sync torch models if using GPU
        if self._use_gpu:
            self._sync_torch_weights()

    def __repr__(self) -> str:
        return (
            f"ActiveInferenceAgent(state_dim={self.state_dim}, "
            f"obs_dim={self.obs_dim}, "
            f"n_actions={self.n_actions}, "
            f"horizon={self.planning_horizon})"
        )
