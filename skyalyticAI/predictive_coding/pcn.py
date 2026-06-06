"""
Predictive Coding Network (PCN)

Hierarchical predictive coding network implementing the free energy
principle. The network consists of multiple layers that iteratively
perform:

1. Inference (perception): Update internal states to minimize
   prediction errors given fixed weights.
2. Learning (memory): Update weights to improve predictions
   given fixed states.

This implementation properly handles:
- Arbitrary network depth
- Precision-weighted error propagation
- Convergence detection during inference
- Both linear and nonlinear generative models
- Online (single-sample) and batch learning
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from skyalyticAI.predictive_coding.pcn_layer import PCNLayer


class PredictiveCodingNetwork:
    """
    Hierarchical Predictive Coding Network.

    Architecture:
        Input (observation) -> Layer 0 -> Layer 1 -> ... -> Layer L (top)

    Each layer l receives:
    - Bottom-up: prediction error from layer l-1
    - Top-down: prediction from layer l+1

    The network alternates between inference (updating states) and
    learning (updating weights) to minimize the total free energy.

    Parameters
    ----------
    layer_sizes : list of int
        Sizes of each layer from bottom to top.
        E.g., [input_dim, hidden1_dim, hidden2_dim, ...]
    sigmas : list of float or None
        Precision parameters for each layer. If None, all layers
        use sigma=1.0. The first element corresponds to the
        precision of the input (bottom) layer.
    learning_rate : float
        Weight learning rate.
    state_learning_rate : float
        State inference learning rate.
    n_inference_steps : int
        Default number of inference iterations per observation.
    inference_tolerance : float
        Convergence tolerance for inference. If the maximum state
        change between iterations is below this value, inference
        stops early.
    nonlinear : bool
        If True, use tanh nonlinearity in generative models.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        sigmas: Optional[List[float]] = None,
        learning_rate: float = 0.01,
        state_learning_rate: float = 0.05,
        n_inference_steps: int = 20,
        inference_tolerance: float = 1e-5,
        nonlinear: bool = False,
        sparse: bool = False,
        synapses_per_neuron: int = 5000,
        top_prior_sigma: float = 1.0,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError(
                "layer_sizes must have at least 2 elements (input + one hidden)"
            )
        for i, s in enumerate(layer_sizes):
            if s <= 0:
                raise ValueError(
                    f"layer_sizes[{i}] must be positive, got {s}"
                )

        self.layer_sizes = list(layer_sizes)
        self.n_layers = len(layer_sizes) - 1
        self.n_inference_steps = n_inference_steps
        self.inference_tolerance = inference_tolerance
        self.top_prior_sigma = top_prior_sigma

        if sigmas is not None:
            if len(sigmas) != self.n_layers:
                raise ValueError(
                    f"sigmas must have {self.n_layers} elements, "
                    f"got {len(sigmas)}"
                )
            self.sigmas = list(sigmas)
        else:
            self.sigmas = [1.0] * self.n_layers

        f, f_deriv = None, None
        if nonlinear:
            f = _tanh_predict
            f_deriv = _tanh_predict_deriv

        self.sparse = sparse
        self.synapses_per_neuron = synapses_per_neuron
        self.layers: List[PCNLayer] = []
        for i in range(self.n_layers):
            dim_below = layer_sizes[i]
            dim = layer_sizes[i + 1]
            self.layers.append(
                PCNLayer(
                    dim_below=dim_below,
                    dim=dim,
                    sigma=self.sigmas[i],
                    f=f,
                    f_deriv=f_deriv,
                    learning_rate=learning_rate,
                    state_learning_rate=state_learning_rate,
                    sparse=sparse,
                    synapses_per_neuron=synapses_per_neuron,
                )
            )

        self.input_state: np.ndarray = np.zeros(layer_sizes[0], dtype=np.float64)
        self.prediction_errors: List[Optional[np.ndarray]] = [None] * self.n_layers
        self._inference_converged: bool = False

    def infer(
        self,
        observation: np.ndarray,
        n_steps: Optional[int] = None,
        inference_lr: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Perform iterative inference to update all hidden states.

        Given an observation at the bottom layer, iteratively update
        all hidden layer states to minimize the total free energy.

        The inference procedure:
        1. Fix the bottom layer state to the observation
        2. Compute prediction errors bottom-up
        3. Update states top-down using precision-weighted errors
        4. Repeat until convergence or max iterations

        Parameters
        ----------
        observation : np.ndarray
            Input observation, shape (layer_sizes[0],).
        n_steps : int or None
            Number of inference iterations. If None, uses default.
        inference_lr : float or None
            State learning rate for this inference. If None, uses
            the layer's default.

        Returns
        -------
        dict
            Dictionary containing:
            - 'converged': whether inference converged early
            - 'n_steps': actual number of steps taken
            - 'total_error': sum of squared prediction errors
            - 'errors': list of prediction errors per layer
        """
        observation = np.asarray(observation, dtype=np.float64)
        if observation.shape != (self.layer_sizes[0],):
            raise ValueError(
                f"observation shape must be ({self.layer_sizes[0]},), "
                f"got {observation.shape}"
            )

        self.input_state = observation.copy()

        if n_steps is None:
            n_steps = self.n_inference_steps

        self._inference_converged = False

        for step in range(n_steps):
            max_change = 0.0

            for l in range(self.n_layers):
                if l == 0:
                    obs_below = self.input_state
                else:
                    obs_below = self.layers[l - 1].x

                error = self.layers[l].compute_error(obs_below)
                self.prediction_errors[l] = error

            for l in range(self.n_layers - 1, -1, -1):
                if l == self.n_layers - 1:
                    prediction_from_above = np.zeros(self.layer_sizes[-1])
                    sigma_above = self.top_prior_sigma
                else:
                    prediction_from_above = self.layers[l + 1].predict()
                    sigma_above = self.layers[l + 1].sigma

                error_from_below = self.prediction_errors[l]
                if error_from_below is None:
                    continue

                old_x = self.layers[l].x.copy()

                if inference_lr is not None:
                    old_lr = self.layers[l].state_learning_rate
                    self.layers[l].state_learning_rate = inference_lr
                    self.layers[l].inference_step(
                        prediction_from_above, error_from_below, sigma_above
                    )
                    self.layers[l].state_learning_rate = old_lr
                else:
                    self.layers[l].inference_step(
                        prediction_from_above, error_from_below, sigma_above
                    )

                change = np.max(np.abs(self.layers[l].x - old_x))
                max_change = max(max_change, change)

            if max_change < self.inference_tolerance:
                self._inference_converged = True
                break

        total_error = 0.0
        for err in self.prediction_errors:
            if err is not None:
                total_error += float(np.sum(err ** 2))

        return {
            "converged": self._inference_converged,
            "n_steps": step + 1 if n_steps > 0 else 0,
            "total_error": total_error,
            "errors": [e.copy() if e is not None else None for e in self.prediction_errors],
        }

    def learn(self) -> Dict[str, Any]:
        """
        Update all weights based on current prediction errors.

        This should be called after inference to update the generative
        model weights. Each layer's weights are updated to minimize
        the prediction error at the layer below.

        Returns
        -------
        dict
            Dictionary containing:
            - 'weight_updates': list of weight update magnitudes per layer
            - 'total_update': sum of all weight update magnitudes
        """
        weight_updates = []

        for l in range(self.n_layers):
            if self.prediction_errors[l] is None:
                weight_updates.append(0.0)
                continue

            if l == 0:
                obs_below = self.input_state
            else:
                obs_below = self.layers[l - 1].x

            error = self.prediction_errors[l]
            update_mag = self.layers[l].learning_step(error, self.layers[l].x)
            weight_updates.append(update_mag)

        return {
            "weight_updates": weight_updates,
            "total_update": sum(weight_updates),
        }

    def infer_batch(
        self,
        observations: np.ndarray,
        n_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform batch inference on multiple observations.

        Each observation is processed independently through the
        same network weights but with separate state trajectories.

        Parameters
        ----------
        observations : np.ndarray
            Batch of observations, shape (batch_size, layer_sizes[0]).
        n_steps : int or None
            Number of inference iterations per observation.

        Returns
        -------
        dict
            Batch inference results including errors and states.
        """
        observations = np.asarray(observations, dtype=np.float64)
        if observations.ndim == 1:
            observations = observations[np.newaxis, :]
        if observations.ndim != 2:
            raise ValueError(
                f"observations must be 1D or 2D, got {observations.ndim}D"
            )

        batch_size = observations.shape[0]

        saved_input = self.input_state.copy()
        saved_errors = [e.copy() if e is not None else None for e in self.prediction_errors]
        saved_states = [layer.x.copy() for layer in self.layers]

        batch_errors = np.zeros(batch_size, dtype=np.float64)
        batch_states = np.zeros(
            (batch_size, self.layers[0].dim), dtype=np.float64
        )

        for b in range(batch_size):
            # Reset layer states and prediction errors before each observation's inference
            for layer in self.layers:
                layer.x = np.zeros(layer.dim, dtype=np.float64)
            self.prediction_errors = [None] * self.n_layers
            self.input_state = np.zeros(self.layer_sizes[0], dtype=np.float64)

            result = self.infer(observations[b], n_steps=n_steps)
            batch_errors[b] = result["total_error"]
            batch_states[b] = self.layers[0].x.copy() if self.n_layers > 0 else self.input_state.copy()

        self.input_state = saved_input
        self.prediction_errors = saved_errors
        for l_idx, layer in enumerate(self.layers):
            layer.x = saved_states[l_idx]

        return {
            "total_errors": batch_errors,
            "states": batch_states,
            "mean_error": float(np.mean(batch_errors)),
        }

    def learn_batch(
        self,
        observations: np.ndarray,
        n_infer_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform batch learning: infer then update weights.

        Processes each observation through inference, accumulates
        weight updates across the batch, then applies the averaged
        update once.

        Parameters
        ----------
        observations : np.ndarray
            Batch of observations, shape (batch_size, layer_sizes[0]).
        n_infer_steps : int or None
            Number of inference iterations.

        Returns
        -------
        dict
            Batch learning metrics.
        """
        observations = np.asarray(observations, dtype=np.float64)
        if observations.ndim == 1:
            observations = observations[np.newaxis, :]

        batch_size = observations.shape[0]

        # Save initial weights
        saved_W = [layer.W.copy() for layer in self.layers]
        saved_b = [layer.b.copy() for layer in self.layers]

        # Accumulate weight changes across the batch
        acc_dW: list = [None] * self.n_layers
        acc_db = [np.zeros_like(layer.b) for layer in self.layers]

        for b in range(batch_size):
            # Reset weights to initial for each sample (batch learning)
            for l, layer in enumerate(self.layers):
                layer.W = saved_W[l].copy()
                layer.b = saved_b[l].copy()

            # Reset layer states and prediction errors before each sample's inference
            for layer in self.layers:
                layer.x = np.zeros(layer.dim, dtype=np.float64)
            self.prediction_errors = [None] * self.n_layers
            self.input_state = np.zeros(self.layer_sizes[0], dtype=np.float64)

            self.infer(observations[b], n_steps=n_infer_steps)
            self.learn()

            # Accumulate weight changes from this sample
            for l, layer in enumerate(self.layers):
                dW = layer.W - saved_W[l]
                db = layer.b - saved_b[l]
                if acc_dW[l] is None:
                    acc_dW[l] = dW.copy()
                else:
                    acc_dW[l] = acc_dW[l] + dW
                acc_db[l] = acc_db[l] + db

        # Apply averaged updates to original weights
        total_update = 0.0
        for l, layer in enumerate(self.layers):
            avg_dW = acc_dW[l] / batch_size if acc_dW[l] is not None else 0
            avg_db = acc_db[l] / batch_size
            layer.W = saved_W[l] + avg_dW
            layer.b = saved_b[l] + avg_db
            if acc_dW[l] is not None:
                dW_arr = acc_dW[l].toarray() if hasattr(acc_dW[l], 'toarray') else acc_dW[l]
                total_update += float(np.linalg.norm(dW_arr / batch_size))

        return {
            "total_update": total_update / max(self.n_layers, 1),
            "batch_size": batch_size,
        }

    def predict_next(self) -> np.ndarray:
        """
        Generate a prediction for the next observation.

        Uses the current top-layer state to generate a top-down
        prediction through all layers.

        Returns
        -------
        np.ndarray
            Predicted observation at the bottom layer, shape (layer_sizes[0],).
        """
        if self.n_layers == 1:
            return self.layers[0].predict()

        current = self.layers[-1].x
        for l in range(self.n_layers - 1, -1, -1):
            current = self.layers[l].predict(current)

        return current

    def get_free_energy(self) -> float:
        """
        Compute the total free energy of the network.

        F = sum_l sigma_l * ||x_l - f_l(x_{l+1})||^2 / 2
            + top_prior_sigma * ||x_top||^2 / 2

        Returns
        -------
        float
            Total free energy.
        """
        total = 0.0
        for l in range(self.n_layers):
            if l == 0:
                obs_below = self.input_state
            else:
                obs_below = self.layers[l - 1].x

            error = obs_below - self.layers[l].predict()
            total += self.layers[l].sigma * float(np.sum(error ** 2)) / 2.0

        # Top-layer prior term: top_prior_sigma * ||x_top||^2 / 2
        top_x = self.layers[-1].x
        total += self.top_prior_sigma * float(np.sum(top_x ** 2)) / 2.0

        return total

    def get_prediction_uncertainty(self) -> float:
        """
        Estimate prediction uncertainty based on recent errors.

        Returns
        -------
        float
            Mean squared prediction error across all layers.
        """
        total = 0.0
        count = 0
        for err in self.prediction_errors:
            if err is not None:
                total += float(np.mean(err ** 2))
                count += 1
        return total / max(count, 1)

    def reset(self) -> None:
        """Reset all layer states."""
        for layer in self.layers:
            layer.reset()
        self.input_state = np.zeros(self.layer_sizes[0], dtype=np.float64)
        self.prediction_errors = [None] * self.n_layers

    def state_dict(self) -> Dict[str, Any]:
        """Return the network state for serialization."""
        return {
            "layers": [l.state_dict() for l in self.layers],
            "input_state": self.input_state.copy(),
            "layer_sizes": self.layer_sizes,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load network state from a dictionary."""
        for layer, l_state in zip(self.layers, state["layers"]):
            layer.load_state_dict(l_state)
        self.input_state = state["input_state"].copy()

    def __repr__(self) -> str:
        return (
            f"PredictiveCodingNetwork(layers={self.layer_sizes}, "
            f"n_layers={self.n_layers})"
        )


def _tanh_predict(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Nonlinear generative function using tanh with bias."""
    return np.tanh(W @ x + b)


def _tanh_predict_deriv(x: np.ndarray, W: np.ndarray, b: Optional[np.ndarray] = None) -> np.ndarray:
    """Derivative of tanh generative function w.r.t. x."""
    from scipy import sparse as scipy_sparse
    if b is None:
        y = W @ x
    else:
        y = W @ x + b
    if scipy_sparse.issparse(W):
        return W.multiply((1 - np.tanh(y) ** 2)[:, np.newaxis])
    else:
        return W * (1 - np.tanh(y) ** 2)[:, np.newaxis]
