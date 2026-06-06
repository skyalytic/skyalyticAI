"""
World Model - Deep Recurrent State Space Model for Planning and Imagination

Implements a world model that learns an internal representation of
environment dynamics. Given an observation and action, the model can:
1. Encode observations into latent states
2. Predict next latent states given actions
3. Decode latent states back to observation space
4. Imagine future trajectories without environment interaction

Industrial-grade improvements:
- Deep MLP encoder/decoder/dynamics (configurable depth, default 4 layers)
- Batch training support for GPU utilization
- Analytical backpropagation through all components
- KL divergence regularization for latent space
- Gradient clipping for numerical stability
- Optional GPU acceleration via PyTorch
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    pass


class _DeepMLP:
    """
    Deep MLP with configurable number of layers.

    Parameters
    ----------
    input_dim : int
    hidden_dims : list of int
    output_dim : int
    learning_rate : float
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        learning_rate: float = 0.001,
    ) -> None:
        self.learning_rate = learning_rate
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        dims = [input_dim] + list(hidden_dims) + [output_dim]
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i])
            W = np.random.randn(dims[i + 1], dims[i]) * scale
            b = np.zeros(dims[i + 1], dtype=np.float64)
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for i in range(len(self.weights) - 1):
            x = np.tanh(self.weights[i] @ x + self.biases[i])
        x = self.weights[-1] @ x + self.biases[-1]
        return x

    def forward_with_activations(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        activations = [x.copy()]
        pre_activations = []

        for i in range(len(self.weights) - 1):
            z = self.weights[i] @ x + self.biases[i]
            pre_activations.append(z)
            x = np.tanh(z)
            activations.append(x.copy())

        z = self.weights[-1] @ x + self.biases[-1]
        pre_activations.append(z)
        activations.append(z.copy())

        return z, activations, pre_activations

    def backward(
        self,
        grad_output: np.ndarray,
        activations: List[np.ndarray],
        pre_activations: List[np.ndarray],
        clip_value: float = 1.0,
    ) -> np.ndarray:
        grad = grad_output
        for i in range(len(self.weights) - 1, -1, -1):
            if i < len(self.weights) - 1:
                grad = grad * (1.0 - np.tanh(pre_activations[i]) ** 2)

            dW = np.outer(grad, activations[i])
            db = grad.copy()

            grad = self.weights[i].T @ grad

            if clip_value > 0:
                flat_grad = np.concatenate([dW.flatten(), db.flatten()])
                flat_norm = np.linalg.norm(flat_grad)
                if flat_norm > clip_value:
                    scale = clip_value / flat_norm
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

    class _DeepMLP_Torch(nn.Module):
        """
        PyTorch-based Deep MLP that mirrors _DeepMLP.

        Uses PyTorch layers and autograd for GPU-accelerated training.
        Architecture: n_layers Linear layers with tanh activation,
        no activation on the output layer.

        Parameters
        ----------
        input_dim : int
        hidden_dims : list of int
        output_dim : int
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int],
            output_dim: int,
        ) -> None:
            super().__init__()
            dims = [input_dim] + list(hidden_dims) + [output_dim]
            self.linear_layers = nn.ModuleList()
            for i in range(len(dims) - 1):
                self.linear_layers.append(nn.Linear(dims[i], dims[i + 1]))
            self._init_weights(dims)

        def _init_weights(self, dims: List[int]) -> None:
            for i, layer in enumerate(self.linear_layers):
                scale = math.sqrt(2.0 / dims[i])
                nn.init.normal_(layer.weight, std=scale)
                nn.init.zeros_(layer.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for i, layer in enumerate(self.linear_layers[:-1]):
                x = torch.tanh(layer(x))
            x = self.linear_layers[-1](x)
            return x

        def copy_from_numpy(self, mlp: _DeepMLP) -> None:
            """Copy weights from a numpy _DeepMLP into this torch module."""
            with torch.no_grad():
                for i, layer in enumerate(self.linear_layers):
                    layer.weight.copy_(
                        torch.tensor(mlp.weights[i], dtype=torch.float32)
                    )
                    layer.bias.copy_(
                        torch.tensor(mlp.biases[i], dtype=torch.float32)
                    )

        def copy_to_numpy(self, mlp: _DeepMLP) -> None:
            """Copy weights from this torch module into a numpy _DeepMLP."""
            for i, layer in enumerate(self.linear_layers):
                mlp.weights[i] = layer.weight.detach().cpu().numpy().astype(np.float64)
                mlp.biases[i] = layer.bias.detach().cpu().numpy().astype(np.float64)


class WorldModel:
    """
    Deep World Model with encoder, dynamics, and decoder components.

    Architecture:
        Encoder: obs -> deep MLP -> (mu, logvar) -> z (VAE)
        Dynamics: (z, action) -> deep MLP -> z_next
        Decoder: z -> deep MLP -> obs_reconstructed
        Reward: z -> deep MLP -> reward

    Training minimizes:
        L = recon_loss_obs + recon_loss_next + reward_loss + kl_weight * kl_divergence

    Parameters
    ----------
    obs_dim : int
        Observation dimension.
    state_dim : int
        Latent state dimension.
    action_dim : int
        Action dimension.
    hidden_dim : int
        Hidden layer dimension for all sub-networks.
    n_layers : int
        Number of hidden layers in each sub-network.
    learning_rate : float
        Learning rate for all components.
    kl_weight : float
        Weight for KL divergence regularization term.
    grad_clip_value : float
        Maximum gradient norm for gradient clipping.
    device : Any
        PyTorch device for GPU acceleration.
    """

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        learning_rate: float = 0.001,
        kl_weight: float = 0.01,
        grad_clip_value: float = 1.0,
        device: Any = None,
    ) -> None:
        if obs_dim <= 0:
            raise ValueError(f"obs_dim must be positive, got {obs_dim}")
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")

        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight
        self.grad_clip_value = grad_clip_value

        hidden_dims = [hidden_dim] * n_layers

        self.encoder_trunk = _DeepMLP(
            input_dim=obs_dim,
            hidden_dims=hidden_dims,
            output_dim=hidden_dim,
            learning_rate=learning_rate,
        )

        enc_head_scale = np.sqrt(2.0 / hidden_dim)
        self.enc_W_mu = np.random.randn(state_dim, hidden_dim) * enc_head_scale
        self.enc_b_mu = np.zeros(state_dim, dtype=np.float64)
        self.enc_W_logvar = np.random.randn(state_dim, hidden_dim) * enc_head_scale
        self.enc_b_logvar = np.zeros(state_dim, dtype=np.float64)

        self.dynamics_net = _DeepMLP(
            input_dim=state_dim + action_dim,
            hidden_dims=hidden_dims,
            output_dim=state_dim,
            learning_rate=learning_rate,
        )

        self.decoder_net = _DeepMLP(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=obs_dim,
            learning_rate=learning_rate,
        )

        self.reward_net = _DeepMLP(
            input_dim=state_dim,
            hidden_dims=[hidden_dim] * max(1, n_layers - 1),
            output_dim=1,
            learning_rate=learning_rate,
        )

        self.loss_history: List[Dict[str, float]] = []

        self.device = device
        self._use_gpu = (
            _TORCH_AVAILABLE
            and device is not None
            and str(device) == "cuda"
        )

        # GPU model placeholders (initialized lazily if GPU is used)
        self._encoder_trunk_torch: Optional[_DeepMLP_Torch] = None
        self._dynamics_net_torch: Optional[_DeepMLP_Torch] = None
        self._decoder_net_torch: Optional[_DeepMLP_Torch] = None
        self._reward_net_torch: Optional[_DeepMLP_Torch] = None
        self._enc_W_mu_torch: Optional[nn.Parameter] = None
        self._enc_b_mu_torch: Optional[nn.Parameter] = None
        self._enc_W_logvar_torch: Optional[nn.Parameter] = None
        self._enc_b_logvar_torch: Optional[nn.Parameter] = None
        self._optimizer: Optional[torch.optim.Adam] = None

        if self._use_gpu:
            self._init_gpu_models()

    # ------------------------------------------------------------------
    # GPU model management
    # ------------------------------------------------------------------

    def _init_gpu_models(self) -> None:
        """Initialize PyTorch models and copy weights from numpy models."""
        if not _TORCH_AVAILABLE:
            return

        hidden_dims = [self.hidden_dim] * self.n_layers
        reward_hidden_dims = [self.hidden_dim] * max(1, self.n_layers - 1)

        self._encoder_trunk_torch = _DeepMLP_Torch(
            input_dim=self.obs_dim,
            hidden_dims=hidden_dims,
            output_dim=self.hidden_dim,
        )
        self._dynamics_net_torch = _DeepMLP_Torch(
            input_dim=self.state_dim + self.action_dim,
            hidden_dims=hidden_dims,
            output_dim=self.state_dim,
        )
        self._decoder_net_torch = _DeepMLP_Torch(
            input_dim=self.state_dim,
            hidden_dims=hidden_dims,
            output_dim=self.obs_dim,
        )
        self._reward_net_torch = _DeepMLP_Torch(
            input_dim=self.state_dim,
            hidden_dims=reward_hidden_dims,
            output_dim=1,
        )

        # Encoder head parameters
        self._enc_W_mu_torch = nn.Parameter(
            torch.tensor(self.enc_W_mu, dtype=torch.float32)
        )
        self._enc_b_mu_torch = nn.Parameter(
            torch.tensor(self.enc_b_mu, dtype=torch.float32)
        )
        self._enc_W_logvar_torch = nn.Parameter(
            torch.tensor(self.enc_W_logvar, dtype=torch.float32)
        )
        self._enc_b_logvar_torch = nn.Parameter(
            torch.tensor(self.enc_b_logvar, dtype=torch.float32)
        )

        # Copy numpy weights into torch models
        self._sync_numpy_to_torch()

        # Move everything to device
        self._encoder_trunk_torch.to(self.device)
        self._dynamics_net_torch.to(self.device)
        self._decoder_net_torch.to(self.device)
        self._reward_net_torch.to(self.device)
        # Encoder head parameters need a module container to be moved with .to()
        # Instead, we move them manually
        self._enc_W_mu_torch.data = self._enc_W_mu_torch.data.to(self.device)
        self._enc_b_mu_torch.data = self._enc_b_mu_torch.data.to(self.device)
        self._enc_W_logvar_torch.data = self._enc_W_logvar_torch.data.to(self.device)
        self._enc_b_logvar_torch.data = self._enc_b_logvar_torch.data.to(self.device)

        # Build optimizer over all parameters
        all_params: List[nn.Parameter] = []
        all_params += list(self._encoder_trunk_torch.parameters())
        all_params += list(self._dynamics_net_torch.parameters())
        all_params += list(self._decoder_net_torch.parameters())
        all_params += list(self._reward_net_torch.parameters())
        all_params += [
            self._enc_W_mu_torch,
            self._enc_b_mu_torch,
            self._enc_W_logvar_torch,
            self._enc_b_logvar_torch,
        ]
        self._optimizer = torch.optim.Adam(all_params, lr=self.learning_rate)

    def _sync_numpy_to_torch(self) -> None:
        """Copy weights from numpy models into torch models."""
        if not self._use_gpu:
            return
        self._encoder_trunk_torch.copy_from_numpy(self.encoder_trunk)
        self._dynamics_net_torch.copy_from_numpy(self.dynamics_net)
        self._decoder_net_torch.copy_from_numpy(self.decoder_net)
        self._reward_net_torch.copy_from_numpy(self.reward_net)
        with torch.no_grad():
            self._enc_W_mu_torch.copy_(
                torch.tensor(self.enc_W_mu, dtype=torch.float32, device=self.device)
            )
            self._enc_b_mu_torch.copy_(
                torch.tensor(self.enc_b_mu, dtype=torch.float32, device=self.device)
            )
            self._enc_W_logvar_torch.copy_(
                torch.tensor(self.enc_W_logvar, dtype=torch.float32, device=self.device)
            )
            self._enc_b_logvar_torch.copy_(
                torch.tensor(self.enc_b_logvar, dtype=torch.float32, device=self.device)
            )

    def _sync_torch_to_numpy(self) -> None:
        """Copy weights from torch models back into numpy models."""
        if not self._use_gpu:
            return
        self._encoder_trunk_torch.copy_to_numpy(self.encoder_trunk)
        self._dynamics_net_torch.copy_to_numpy(self.dynamics_net)
        self._decoder_net_torch.copy_to_numpy(self.decoder_net)
        self._reward_net_torch.copy_to_numpy(self.reward_net)
        self.enc_W_mu = self._enc_W_mu_torch.detach().cpu().numpy().astype(np.float64)
        self.enc_b_mu = self._enc_b_mu_torch.detach().cpu().numpy().astype(np.float64)
        self.enc_W_logvar = self._enc_W_logvar_torch.detach().cpu().numpy().astype(np.float64)
        self.enc_b_logvar = self._enc_b_logvar_torch.detach().cpu().numpy().astype(np.float64)

    # ------------------------------------------------------------------
    # Static / helper methods (unchanged)
    # ------------------------------------------------------------------

    @staticmethod
    def _tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def _clip_gradient(self, grad: np.ndarray) -> np.ndarray:
        if self.grad_clip_value <= 0:
            return grad
        norm = np.linalg.norm(grad)
        if norm > self.grad_clip_value:
            grad = grad * (self.grad_clip_value / norm)
        return grad

    @staticmethod
    def _compute_gradient_without_update(
        net: _DeepMLP,
        grad_output: np.ndarray,
        activations: List[np.ndarray],
        pre_activations: List[np.ndarray],
    ) -> np.ndarray:
        """
        Compute gradient through a _DeepMLP without updating weights.

        This is used when the same network needs to be backpropagated
        through multiple times in a single training step. The first
        pass computes only the input gradient (for chaining), while
        the second pass computes and applies the weight updates.
        """
        grad = grad_output
        for i in range(len(net.weights) - 1, -1, -1):
            if i < len(net.weights) - 1:
                grad = grad * (1.0 - np.tanh(pre_activations[i]) ** 2)
            grad = net.weights[i].T @ grad
        return grad

    # ------------------------------------------------------------------
    # Inference methods (unchanged – always use numpy)
    # ------------------------------------------------------------------

    def encode(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Encode observation into latent state.

        Parameters
        ----------
        obs : np.ndarray
            Observation vector, shape (obs_dim,).

        Returns
        -------
        z : np.ndarray
            Sampled latent state, shape (state_dim,).
        mu : np.ndarray
            Mean of latent distribution, shape (state_dim,).
        logvar : np.ndarray
            Log variance of latent distribution, shape (state_dim,).
        """
        h = self.encoder_trunk.forward(obs)
        mu = self.enc_W_mu @ h + self.enc_b_mu
        logvar = self.enc_W_logvar @ h + self.enc_b_logvar
        logvar = np.clip(logvar, -10, 10)

        std = np.exp(0.5 * logvar)
        eps = np.random.standard_normal(self.state_dim)
        z = mu + std * eps

        return z, mu, logvar

    def encode_deterministic(self, obs: np.ndarray) -> np.ndarray:
        """
        Encode observation into latent state (deterministic, using mean).
        """
        h = self.encoder_trunk.forward(obs)
        mu = self.enc_W_mu @ h + self.enc_b_mu
        return mu

    def predict_next_state(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Predict next latent state given current state and action.
        """
        sa = np.concatenate([state, action])
        return self.dynamics_net.forward(sa)

    def decode(self, state: np.ndarray) -> np.ndarray:
        """
        Decode latent state into observation space.
        """
        return self.decoder_net.forward(state)

    def predict_reward(self, state: np.ndarray) -> float:
        """
        Predict expected reward from latent state.
        """
        return float(self.reward_net.forward(state)[0])

    def imagine_trajectory(
        self,
        start_obs: np.ndarray,
        action_sequence: List[np.ndarray],
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Imagine a future trajectory given starting observation and actions.
        """
        if deterministic:
            state = self.encode_deterministic(start_obs)
        else:
            state, _, _ = self.encode(start_obs)

        states = [state.copy()]
        observations = [self.decode(state).copy()]

        for action in action_sequence:
            state = self.predict_next_state(state, action)
            obs = self.decode(state)
            states.append(state.copy())
            observations.append(obs.copy())

        return np.array(states), np.array(observations)

    # ------------------------------------------------------------------
    # Training – dispatch
    # ------------------------------------------------------------------

    def train_step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        reward: Optional[float] = None,
    ) -> Dict[str, float]:
        obs = np.asarray(obs, dtype=np.float64)
        action = np.asarray(action, dtype=np.float64)
        next_obs = np.asarray(next_obs, dtype=np.float64)

        if self._use_gpu:
            return self._train_step_gpu(obs, action, next_obs, reward)
        return self._train_step_cpu(obs, action, next_obs, reward)

    # ------------------------------------------------------------------
    # Training – CPU path (unchanged)
    # ------------------------------------------------------------------

    def _train_step_cpu(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        reward: Optional[float],
    ) -> Dict[str, float]:
        enc_h, enc_activations, enc_pre = self.encoder_trunk.forward_with_activations(obs)
        mu = self.enc_W_mu @ enc_h + self.enc_b_mu
        logvar = self.enc_W_logvar @ enc_h + self.enc_b_logvar
        logvar = np.clip(logvar, -10, 10)
        std = np.exp(0.5 * logvar)
        eps = np.random.standard_normal(self.state_dim)
        z = mu + std * eps

        obs_recon, dec_activations, dec_pre = self.decoder_net.forward_with_activations(z)

        sa = np.concatenate([z, action])
        z_next_pred, dyn_activations, dyn_pre = self.dynamics_net.forward_with_activations(sa)

        next_obs_pred, dec_next_activations, dec_next_pre = self.decoder_net.forward_with_activations(z_next_pred)

        recon_loss_obs = float(np.mean((obs_recon - obs) ** 2))
        recon_loss_next = float(np.mean((next_obs_pred - next_obs) ** 2))
        kl_loss = float(-0.5 * np.sum(1 + logvar - mu ** 2 - np.exp(logvar)))

        reward_pred_raw, reward_activations, reward_pre = self.reward_net.forward_with_activations(z)
        reward_pred = float(reward_pred_raw[0])
        reward_loss = 0.0
        if reward is not None:
            reward_loss = float((reward_pred - reward) ** 2)

        total_loss = recon_loss_obs + recon_loss_next + reward_loss + self.kl_weight * kl_loss

        lr = self.learning_rate
        clip = self.grad_clip_value

        d_next_obs_pred = 2.0 * (next_obs_pred - next_obs) / max(next_obs.size, 1)

        d_z_next_from_dec = self._compute_gradient_without_update(
            self.decoder_net, d_next_obs_pred, dec_next_activations, dec_next_pre
        )

        d_z_from_dyn = self._compute_gradient_without_update(
            self.dynamics_net, d_z_next_from_dec, dyn_activations, dyn_pre
        )[:self.state_dim]

        d_obs_recon = 2.0 * (obs_recon - obs) / max(obs.size, 1)

        # Save decoder weights before backward passes to accumulate gradients
        # from both the obs_recon and next_obs_pred loss paths
        dec_weights_orig = [w.copy() for w in self.decoder_net.weights]
        dec_biases_orig = [b.copy() for b in self.decoder_net.biases]

        # Backward through decoder for obs_recon path
        d_z_from_obs = self.decoder_net.backward(d_obs_recon, dec_activations, dec_pre, clip)

        # Save weight deltas from obs_recon path
        dec_delta_w_obs = [
            self.decoder_net.weights[i] - dec_weights_orig[i]
            for i in range(len(self.decoder_net.weights))
        ]
        dec_delta_b_obs = [
            self.decoder_net.biases[i] - dec_biases_orig[i]
            for i in range(len(self.decoder_net.biases))
        ]

        # Restore original weights so next_obs_pred backward uses correct weights
        for i in range(len(self.decoder_net.weights)):
            self.decoder_net.weights[i] = dec_weights_orig[i].copy()
            self.decoder_net.biases[i] = dec_biases_orig[i].copy()

        # Backward through decoder for next_obs_pred path
        self.decoder_net.backward(d_next_obs_pred, dec_next_activations, dec_next_pre, clip)

        # Accumulate weight updates from both paths
        for i in range(len(self.decoder_net.weights)):
            self.decoder_net.weights[i] += dec_delta_w_obs[i]
            self.decoder_net.biases[i] += dec_delta_b_obs[i]

        d_z_from_reward = np.zeros_like(d_z_from_obs)
        if reward is not None:
            d_reward_pred = np.array([2.0 * (reward_pred - reward)])
            d_z_from_reward = self.reward_net.backward(d_reward_pred, reward_activations, reward_pre, clip)

        d_z_total = d_z_from_obs + d_z_from_dyn + d_z_from_reward

        d_mu = d_z_total + self.kl_weight * mu
        d_logvar = d_z_total * 0.5 * eps * std + self.kl_weight * 0.5 * (np.exp(logvar) - 1)

        d_enc_W_mu = np.outer(d_mu, enc_h)
        d_enc_b_mu = d_mu
        d_enc_W_logvar = np.outer(d_logvar, enc_h)
        d_enc_b_logvar = d_logvar

        d_enc_h = d_mu @ self.enc_W_mu + d_logvar @ self.enc_W_logvar

        self.encoder_trunk.backward(d_enc_h, enc_activations, enc_pre, clip)

        self.enc_W_mu -= lr * self._clip_gradient(d_enc_W_mu)
        self.enc_b_mu -= lr * self._clip_gradient(d_enc_b_mu)
        self.enc_W_logvar -= lr * self._clip_gradient(d_enc_W_logvar)
        self.enc_b_logvar -= lr * self._clip_gradient(d_enc_b_logvar)

        self.dynamics_net.backward(d_z_next_from_dec, dyn_activations, dyn_pre, clip)

        self.loss_history.append({
            "total": total_loss,
            "recon_obs": recon_loss_obs,
            "recon_next": recon_loss_next,
            "kl": kl_loss,
            "reward": reward_loss,
        })

        return {
            "total": total_loss,
            "recon_obs": recon_loss_obs,
            "recon_next": recon_loss_next,
            "kl": kl_loss,
            "reward": reward_loss,
        }

    # ------------------------------------------------------------------
    # Training – GPU path
    # ------------------------------------------------------------------

    def _train_step_gpu(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        reward: Optional[float],
    ) -> Dict[str, float]:
        """Single-sample GPU training step using PyTorch autograd."""
        # Convert inputs to tensors with batch dimension
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_t = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        # --- Forward pass ---
        # Encoder trunk
        enc_h = self._encoder_trunk_torch(obs_t)  # (1, hidden_dim)
        mu = torch.nn.functional.linear(enc_h, self._enc_W_mu_torch, self._enc_b_mu_torch)  # (1, state_dim)
        logvar = torch.nn.functional.linear(enc_h, self._enc_W_logvar_torch, self._enc_b_logvar_torch)
        logvar = torch.clamp(logvar, -10, 10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        z = mu + std * eps

        # Decoder – obs reconstruction
        obs_recon = self._decoder_net_torch(z)  # (1, obs_dim)

        # Dynamics
        sa = torch.cat([z, action_t], dim=-1)  # (1, state_dim + action_dim)
        z_next_pred = self._dynamics_net_torch(sa)  # (1, state_dim)

        # Decoder – next obs prediction
        next_obs_pred = self._decoder_net_torch(z_next_pred)  # (1, obs_dim)

        # Reward
        reward_pred = self._reward_net_torch(z)  # (1, 1)

        # --- Losses ---
        recon_loss_obs = torch.mean((obs_recon - obs_t) ** 2)
        recon_loss_next = torch.mean((next_obs_pred - next_obs_t) ** 2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar))

        reward_loss = torch.tensor(0.0, device=self.device)
        if reward is not None:
            reward_t = torch.tensor(reward, dtype=torch.float32, device=self.device)
            reward_loss = (reward_pred[0, 0] - reward_t) ** 2

        total_loss = recon_loss_obs + recon_loss_next + reward_loss + self.kl_weight * kl_loss

        # --- Backward pass ---
        self._optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self._encoder_trunk_torch.parameters())
            + list(self._dynamics_net_torch.parameters())
            + list(self._decoder_net_torch.parameters())
            + list(self._reward_net_torch.parameters())
            + [
                self._enc_W_mu_torch,
                self._enc_b_mu_torch,
                self._enc_W_logvar_torch,
                self._enc_b_logvar_torch,
            ],
            self.grad_clip_value,
        )
        self._optimizer.step()

        # Sync torch weights back to numpy for inference / state_dict
        self._sync_torch_to_numpy()

        result = {
            "total": total_loss.item(),
            "recon_obs": recon_loss_obs.item(),
            "recon_next": recon_loss_next.item(),
            "kl": kl_loss.item(),
            "reward": reward_loss.item(),
        }

        self.loss_history.append(result)
        return result

    def _train_step_batch_gpu(
        self,
        obs_batch: np.ndarray,
        action_batch: np.ndarray,
        next_obs_batch: np.ndarray,
        reward_batch: Optional[np.ndarray],
    ) -> Dict[str, float]:
        """Real batch GPU training step – processes all samples together."""
        obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
        action_t = torch.tensor(action_batch, dtype=torch.float32, device=self.device)
        next_obs_t = torch.tensor(next_obs_batch, dtype=torch.float32, device=self.device)

        batch_size = obs_t.shape[0]

        # --- Forward pass ---
        enc_h = self._encoder_trunk_torch(obs_t)  # (B, hidden_dim)
        mu = torch.nn.functional.linear(enc_h, self._enc_W_mu_torch, self._enc_b_mu_torch)
        logvar = torch.nn.functional.linear(enc_h, self._enc_W_logvar_torch, self._enc_b_logvar_torch)
        logvar = torch.clamp(logvar, -10, 10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        z = mu + std * eps

        obs_recon = self._decoder_net_torch(z)
        sa = torch.cat([z, action_t], dim=-1)
        z_next_pred = self._dynamics_net_torch(sa)
        next_obs_pred = self._decoder_net_torch(z_next_pred)
        reward_pred = self._reward_net_torch(z)  # (B, 1)

        # --- Losses (averaged over batch) ---
        recon_loss_obs = torch.mean((obs_recon - obs_t) ** 2)
        recon_loss_next = torch.mean((next_obs_pred - next_obs_t) ** 2)
        # KL: sum over state_dim, mean over batch
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar), dim=-1)
        )

        reward_loss = torch.tensor(0.0, device=self.device)
        if reward_batch is not None:
            reward_t = torch.tensor(reward_batch, dtype=torch.float32, device=self.device).unsqueeze(1)
            reward_loss = torch.mean((reward_pred - reward_t) ** 2)

        total_loss = recon_loss_obs + recon_loss_next + reward_loss + self.kl_weight * kl_loss

        # --- Backward pass ---
        self._optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self._encoder_trunk_torch.parameters())
            + list(self._dynamics_net_torch.parameters())
            + list(self._decoder_net_torch.parameters())
            + list(self._reward_net_torch.parameters())
            + [
                self._enc_W_mu_torch,
                self._enc_b_mu_torch,
                self._enc_W_logvar_torch,
                self._enc_b_logvar_torch,
            ],
            self.grad_clip_value,
        )
        self._optimizer.step()

        self._sync_torch_to_numpy()

        result = {
            "total": total_loss.item(),
            "recon_obs": recon_loss_obs.item(),
            "recon_next": recon_loss_next.item(),
            "kl": kl_loss.item(),
            "reward": reward_loss.item(),
        }

        self.loss_history.append(result)
        return {"total": result["total"], "batch_size": batch_size}

    # ------------------------------------------------------------------
    # Batch training – dispatch
    # ------------------------------------------------------------------

    def train_step_batch(
        self,
        obs_batch: np.ndarray,
        action_batch: np.ndarray,
        next_obs_batch: np.ndarray,
        reward_batch: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Batch training step.

        Parameters
        ----------
        obs_batch : np.ndarray
            Batch of observations, shape (batch_size, obs_dim).
        action_batch : np.ndarray
            Batch of actions, shape (batch_size, action_dim).
        next_obs_batch : np.ndarray
            Batch of next observations, shape (batch_size, obs_dim).
        reward_batch : np.ndarray or None
            Batch of rewards, shape (batch_size,).

        Returns
        -------
        dict
            Average loss metrics across the batch.
        """
        obs_batch = np.asarray(obs_batch, dtype=np.float64)
        action_batch = np.asarray(action_batch, dtype=np.float64)
        next_obs_batch = np.asarray(next_obs_batch, dtype=np.float64)

        if obs_batch.ndim == 1:
            obs_batch = obs_batch[np.newaxis, :]
        if action_batch.ndim == 1:
            action_batch = action_batch[np.newaxis, :]
        if next_obs_batch.ndim == 1:
            next_obs_batch = next_obs_batch[np.newaxis, :]

        if self._use_gpu:
            return self._train_step_batch_gpu(
                obs_batch, action_batch, next_obs_batch, reward_batch
            )

        # CPU fallback: loop over samples
        batch_size = obs_batch.shape[0]
        total_losses = []

        for b in range(batch_size):
            r = float(reward_batch[b]) if reward_batch is not None else None
            result = self.train_step(obs_batch[b], action_batch[b], next_obs_batch[b], r)
            total_losses.append(result["total"])

        return {
            "total": float(np.mean(total_losses)),
            "batch_size": batch_size,
        }

    # ------------------------------------------------------------------
    # Reset / state
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset loss history."""
        self.loss_history = []

    def state_dict(self) -> Dict[str, Any]:
        """Return the model state for serialization."""
        # Sync torch -> numpy first so GPU-trained weights are captured
        if self._use_gpu:
            self._sync_torch_to_numpy()

        state = {
            "enc_W_mu": self.enc_W_mu.copy(),
            "enc_b_mu": self.enc_b_mu.copy(),
            "enc_W_logvar": self.enc_W_logvar.copy(),
            "enc_b_logvar": self.enc_b_logvar.copy(),
            "encoder_trunk": self.encoder_trunk.get_params(),
            "dynamics_net": self.dynamics_net.get_params(),
            "decoder_net": self.decoder_net.get_params(),
            "reward_net": self.reward_net.get_params(),
        }
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load model state from a dictionary."""
        self.enc_W_mu = state["enc_W_mu"].copy()
        self.enc_b_mu = state["enc_b_mu"].copy()
        self.enc_W_logvar = state["enc_W_logvar"].copy()
        self.enc_b_logvar = state["enc_b_logvar"].copy()
        self.encoder_trunk.set_params(state["encoder_trunk"])
        self.dynamics_net.set_params(state["dynamics_net"])
        self.decoder_net.set_params(state["decoder_net"])
        self.reward_net.set_params(state["reward_net"])

        # Sync numpy -> torch so GPU models pick up the new weights
        if self._use_gpu:
            self._sync_numpy_to_torch()

    def __repr__(self) -> str:
        return (
            f"WorldModel(obs_dim={self.obs_dim}, state_dim={self.state_dim}, "
            f"action_dim={self.action_dim}, hidden_dim={self.hidden_dim}, "
            f"n_layers={self.n_layers})"
        )
