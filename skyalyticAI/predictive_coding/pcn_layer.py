"""
Predictive Coding Network Layer

Single layer of a hierarchical predictive coding network.

Implements the core computation of predictive coding:
- Top-down prediction generation from higher layer states
- Bottom-up prediction error computation
- Iterative state inference to minimize prediction errors
- Weight learning to improve predictions

Mathematical formulation:
    Free energy: F = sum_l ||x_l - f_l(x_{l+1})||^2 / (2 * sigma_l^2)

    Inference (perception): minimize F w.r.t. states x_l
    Learning (memory): minimize F w.r.t. weights W_l
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from scipy import sparse as scipy_sparse


class PCNLayer:
    """
    Single layer of a predictive coding network.

    Each layer maintains:
    - A state vector x (the representation at this level)
    - A generative weight matrix W (maps from this layer's state
      to the layer below's prediction)
    - A precision parameter sigma (controls the sensitivity to
      prediction errors at this level)

    Parameters
    ----------
    dim_below : int
        Dimension of the layer below (input dimension for prediction).
    dim : int
        Dimension of this layer's state.
    sigma : float
        Precision (inverse variance) for prediction errors at this
        level. Higher values mean this layer's errors are weighted
        more heavily. Must be positive.
    f : callable or None
        Generative function f(x) that maps this layer's state to
        a prediction for the layer below. If None, uses linear
        mapping: f(x) = W @ x.
    f_deriv : callable or None
        Derivative of the generative function. Required if f is
        provided and is nonlinear. If None and f is None, uses W.
    learning_rate : float
        Learning rate for weight updates.
    state_learning_rate : float
        Learning rate for state inference updates.
    """

    def __init__(
        self,
        dim_below: int,
        dim: int,
        sigma: float = 1.0,
        f: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        f_deriv: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        learning_rate: float = 0.01,
        state_learning_rate: float = 0.05,
        sparse: bool = False,
        synapses_per_neuron: int = 5000,
    ) -> None:
        if dim_below <= 0:
            raise ValueError(f"dim_below must be positive, got {dim_below}")
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        self.dim_below = dim_below
        self.dim = dim
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.state_learning_rate = state_learning_rate
        self.sparse = sparse
        self.synapses_per_neuron = synapses_per_neuron

        if self.sparse:
            n_synapses = min(self.synapses_per_neuron, self.dim)
            rows = np.repeat(np.arange(dim_below), n_synapses)
            cols = np.concatenate([
                np.random.choice(dim, size=n_synapses, replace=False)
                for _ in range(dim_below)
            ])
            data = np.random.randn(dim_below * n_synapses) * np.sqrt(
                2.0 / (dim_below + dim)
            )
            self.W = scipy_sparse.csr_matrix(
                (data, (rows, cols)), shape=(dim_below, dim)
            )
        else:
            self.W = np.random.randn(dim_below, dim) * np.sqrt(2.0 / (dim_below + dim))
        self.b = np.zeros(dim_below, dtype=np.float64)

        self.f = f
        self.f_deriv = f_deriv

        self.x = np.zeros(dim, dtype=np.float64)
        self.prediction_error_below: Optional[np.ndarray] = None

    def predict(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        if x is None:
            x = self.x
        if self.f is not None:
            result = self.f(x, self.W, self.b)
            if self.sparse:
                return np.asarray(result).ravel()
            return result
        result = self.W @ x + self.b
        if self.sparse:
            return np.asarray(result).ravel()
        return result

    def compute_prediction_deriv(self, x: Optional[np.ndarray] = None):
        if x is None:
            x = self.x
        if self.f_deriv is not None:
            if self.sparse:
                J = self.f_deriv(x, self.W, self.b)
                if not scipy_sparse.issparse(J):
                    J = scipy_sparse.csr_matrix(J)
                return J
            return self.f_deriv(x, self.W, self.b)
        return self.W

    def inference_step(
        self,
        prediction_from_above: np.ndarray,
        error_from_below: np.ndarray,
        sigma_above: float = 1.0,
    ) -> np.ndarray:
        """
        Perform one step of state inference.

        Updates the state x to minimize the free energy, which is
        the sum of:
        1. The prediction error from above: ||x - prediction_from_above||^2 * sigma_above
        2. The prediction error propagated from below: W^T @ error_from_below * sigma_below

        The gradient of free energy w.r.t. x is:
            dF/dx = sigma_above * (x - prediction_from_above)
                    - sigma * J^T @ error_from_below

        Parameters
        ----------
        prediction_from_above : np.ndarray
            Prediction received from the layer above, shape (dim,).
        error_from_below : np.ndarray
            Prediction error from the layer below, shape (dim_below,).
        sigma_above : float
            Precision of the prediction from above.

        Returns
        -------
        np.ndarray
            Updated state vector, shape (dim,).
        """
        prediction_from_above = np.asarray(prediction_from_above, dtype=np.float64)
        error_from_below = np.asarray(error_from_below, dtype=np.float64)

        if prediction_from_above.shape != (self.dim,):
            raise ValueError(
                f"prediction_from_above shape must be ({self.dim},), "
                f"got {prediction_from_above.shape}"
            )
        if error_from_below.shape != (self.dim_below,):
            raise ValueError(
                f"error_from_below shape must be ({self.dim_below},), "
                f"got {error_from_below.shape}"
            )

        df_dx = self.compute_prediction_deriv()

        propagated_error = df_dx.T @ error_from_below
        if self.sparse:
            propagated_error = np.asarray(propagated_error).ravel()

        grad = (
            sigma_above * (self.x - prediction_from_above)
            - self.sigma * propagated_error
        )

        grad_norm = np.linalg.norm(grad)
        if grad_norm > 10.0:
            grad = grad * (10.0 / grad_norm)

        self.x -= self.state_learning_rate * grad

        return self.x.copy()

    def learning_step(
        self,
        error_below: np.ndarray,
        x: Optional[np.ndarray] = None,
    ) -> float:
        """
        Update weights to minimize prediction error.

        The gradient of free energy w.r.t. W is:
            dF/dW = -sigma * error_below * x^T

        For linear prediction, this gives the outer product update.

        Parameters
        ----------
        error_below : np.ndarray
            Prediction error at the layer below, shape (dim_below,).
        x : np.ndarray or None
            State vector. If None, uses current internal state.

        Returns
        -------
        float
            Weight update magnitude (Frobenius norm).
        """
        error_below = np.asarray(error_below, dtype=np.float64)
        if x is None:
            x = self.x

        if self.f is None:
            dW = self.sigma * np.outer(error_below, x)
            db = self.sigma * error_below
        else:
            prediction = self.predict(x)
            if self.f_deriv is not None:
                J = self.compute_prediction_deriv(x)
                if self.sparse and scipy_sparse.issparse(J):
                    J_dense = np.asarray(J.todense())
                    W_dense = np.asarray(self.W.todense())
                else:
                    J_dense = J
                    W_dense = self.W
                f_deriv_elem = np.sum(J_dense * W_dense, axis=1) / (np.sum(W_dense ** 2, axis=1) + 1e-10)
            else:
                f_deriv_elem = 1.0 - prediction ** 2
            dW = self.sigma * np.outer(error_below * f_deriv_elem, x)
            db = self.sigma * error_below * f_deriv_elem

        flat_grad = np.concatenate([dW.ravel(), db.ravel()])
        grad_norm = np.linalg.norm(flat_grad)
        if grad_norm > 10.0:
            scale = 10.0 / grad_norm
            dW = dW * scale
            db = db * scale

        if self.sparse:
            W_coo = self.W.tocoo()
            update_values = dW[W_coo.row, W_coo.col]
            dW_sparse = scipy_sparse.coo_matrix(
                (update_values, (W_coo.row, W_coo.col)),
                shape=(self.dim_below, self.dim),
            ).tocsr()
            self.W = self.W + self.learning_rate * dW_sparse
        else:
            self.W += self.learning_rate * dW
        self.b += self.learning_rate * db

        return float(np.linalg.norm(self.learning_rate * self.sigma * np.outer(error_below, x)))

    def compute_error(
        self,
        observation_below: np.ndarray,
    ) -> np.ndarray:
        """
        Compute prediction error for the layer below.

        Parameters
        ----------
        observation_below : np.ndarray
            Actual state (observation) at the layer below.

        Returns
        -------
        np.ndarray
            Prediction error: observation - prediction, shape (dim_below,).
        """
        prediction = self.predict()
        self.prediction_error_below = observation_below - prediction
        return self.prediction_error_below.copy()

    def reset(self) -> None:
        """Reset the layer state."""
        self.x = np.zeros(self.dim, dtype=np.float64)
        self.prediction_error_below = None

    def state_dict(self) -> Dict[str, Any]:
        """Return the layer state for serialization."""
        if self.sparse:
            W_serialized = {
                "data": self.W.data.copy(),
                "indices": self.W.indices.copy(),
                "indptr": self.W.indptr.copy(),
                "shape": self.W.shape,
            }
        else:
            W_serialized = self.W.copy()
        return {
            "W": W_serialized,
            "b": self.b.copy(),
            "x": self.x.copy(),
            "sigma": self.sigma,
            "dim_below": self.dim_below,
            "dim": self.dim,
            "sparse": self.sparse,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load layer state from a dictionary."""
        W_state = state["W"]
        if isinstance(W_state, dict) and "data" in W_state:
            self.W = scipy_sparse.csr_matrix(
                (W_state["data"], W_state["indices"], W_state["indptr"]),
                shape=W_state["shape"],
            )
        else:
            self.W = W_state.copy()
        self.b = state["b"].copy()
        self.x = state["x"].copy()
        if "sigma" in state:
            self.sigma = float(state["sigma"])
        if "dim_below" in state:
            self.dim_below = int(state["dim_below"])
        if "dim" in state:
            self.dim = int(state["dim"])

    def __repr__(self) -> str:
        sparse_info = f", sparse={self.sparse}" if self.sparse else ""
        return (
            f"PCNLayer(dim_below={self.dim_below}, dim={self.dim}, "
            f"sigma={self.sigma}{sparse_info})"
        )
