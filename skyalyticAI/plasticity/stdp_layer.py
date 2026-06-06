"""
STDP Learning Layer

Applies STDP learning rules across a full connectivity matrix
between pre- and post-synaptic neuron populations.

Supports:
- Vectorized trace updates for efficiency
- Multiple STDP variants
- Weight normalization options
- Batch processing
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy import sparse as sp

from skyalyticAI.neurons.sparse_connectivity import SparseConnectivity
from skyalyticAI.plasticity.stdp import STDPSynapse, STDPVariant


class STDPLayer:
    """
    STDP learning layer connecting two populations of neurons.

    Maintains a matrix of STDPSynapse objects and provides vectorized
    update operations for efficient simulation.

    Parameters
    ----------
    pre_dim : int
        Number of pre-synaptic neurons.
    post_dim : int
        Number of post-synaptic neurons.
    A_plus : float
        LTP amplitude for all synapses.
    A_minus : float
        LTD amplitude for all synapses.
    tau_plus : float
        Pre-synaptic trace decay time constant (ms).
    tau_minus : float
        Post-synaptic trace decay time constant (ms).
    w_min : float
        Minimum weight.
    w_max : float
        Maximum weight.
    variant : STDPVariant or str
        STDP update variant.
    learning_rate : float
        Global learning rate multiplier.
    weight_init : str
        Weight initialization: 'uniform', 'normal', 'glorot'.
    normalize_weights : bool
        Whether to normalize weights after each update so that
        the sum of weights into each post-synaptic neuron is
        constrained. This prevents unbounded potentiation.
    norm_target : float
        Target sum for weight normalization per post-synaptic neuron.
    """

    def __init__(
        self,
        pre_dim: int,
        post_dim: int,
        A_plus: float = 0.01,
        A_minus: float = 0.012,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        w_min: float = 0.0,
        w_max: float = 1.0,
        variant: STDPVariant | str = STDPVariant.ADDITIVE,
        learning_rate: float = 1.0,
        weight_init: str = "glorot",
        normalize_weights: bool = False,
        norm_target: Optional[float] = None,
        sparse_connectivity: bool = False,
        synapses_per_neuron: int = 7000,
    ) -> None:
        if pre_dim <= 0:
            raise ValueError(f"pre_dim must be positive, got {pre_dim}")
        if post_dim <= 0:
            raise ValueError(f"post_dim must be positive, got {post_dim}")

        self.pre_dim = pre_dim
        self.post_dim = post_dim
        self.normalize_weights = normalize_weights
        self.norm_target = norm_target if norm_target is not None else pre_dim * 0.5
        self.sparse_connectivity = sparse_connectivity
        self.synapses_per_neuron = synapses_per_neuron

        if isinstance(variant, str):
            variant = STDPVariant(variant.lower())
        self.variant = variant

        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.w_min = w_min
        self.w_max = w_max
        self.learning_rate = learning_rate

        if self.sparse_connectivity:
            self._sparse_conn = SparseConnectivity(
                n_pre=pre_dim,
                n_post=post_dim,
                synapses_per_neuron=synapses_per_neuron,
                weight_init=weight_init,
            )
            self.W = self._sparse_conn.W
        else:
            self._sparse_conn = None
            self.W = self._init_weights(pre_dim, post_dim, weight_init, w_min, w_max)

        self.trace_pre = np.zeros(pre_dim, dtype=np.float64)
        self.trace_post = np.zeros(post_dim, dtype=np.float64)

        self.decay_pre = np.exp(-1.0 / tau_plus)
        self.decay_post = np.exp(-1.0 / tau_minus)

    @staticmethod
    def _init_weights(
        pre_dim: int,
        post_dim: int,
        method: str,
        w_min: float,
        w_max: float,
    ) -> np.ndarray:
        """Initialize weight matrix."""
        rng = np.random.default_rng()

        if method == "uniform":
            W = rng.uniform(w_min, w_max, (post_dim, pre_dim))
        elif method == "normal":
            std = (w_max - w_min) / 4.0
            mean = (w_max + w_min) / 2.0
            W = rng.normal(mean, std, (post_dim, pre_dim))
            W = np.clip(W, w_min, w_max)
        elif method == "glorot":
            std = np.sqrt(2.0 / (pre_dim + post_dim))
            W = rng.standard_normal((post_dim, pre_dim)) * std
            W = np.clip(W, w_min, w_max)
        else:
            raise ValueError(f"Unknown weight_init method: {method}")

        return W.astype(np.float64)

    def update(
        self,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        dt: float = 1.0,
    ) -> float:
        """
        Update all synapses based on pre- and post-synaptic spike events.

        Uses vectorized operations for efficiency. The eligibility traces
        are maintained as vectors and updated in bulk.

        Parameters
        ----------
        pre_spikes : np.ndarray
            Pre-synaptic spike vector of shape (pre_dim,).
            Values should be 0 or 1 (binary spikes).
        post_spikes : np.ndarray
            Post-synaptic spike vector of shape (post_dim,).
            Values should be 0 or 1 (binary spikes).
        dt : float
            Time step duration in milliseconds.

        Returns
        -------
        float
            Mean absolute weight change across all synapses.
        """
        pre_spikes = np.asarray(pre_spikes, dtype=np.float64)
        post_spikes = np.asarray(post_spikes, dtype=np.float64)

        if pre_spikes.shape != (self.pre_dim,):
            raise ValueError(
                f"pre_spikes shape must be ({self.pre_dim},), "
                f"got {pre_spikes.shape}"
            )
        if post_spikes.shape != (self.post_dim,):
            raise ValueError(
                f"post_spikes shape must be ({self.post_dim},), "
                f"got {post_spikes.shape}"
            )
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        decay_pre_dt = np.exp(-dt / self.tau_plus)
        decay_post_dt = np.exp(-dt / self.tau_minus)

        self.trace_pre *= decay_pre_dt
        self.trace_post *= decay_post_dt

        if self.sparse_connectivity:
            return self._update_sparse(pre_spikes, post_spikes)

        # --- Dense path ---
        W_old = self.W.copy()

        # Compute both LTP and LTD based on original weights before applying
        ltp_delta = np.zeros_like(self.W)
        ltd_delta = np.zeros_like(self.W)

        if np.any(post_spikes > 0):
            ltp_input = np.outer(post_spikes, self.trace_pre) * self.A_plus
            ltp_input *= self.learning_rate
            ltp_delta = self._scale_ltp(ltp_input)

        if np.any(pre_spikes > 0):
            ltd_input = np.outer(self.trace_post, pre_spikes) * self.A_minus
            ltd_input *= self.learning_rate
            ltd_delta = self._scale_ltd(ltd_input)

        self.W += ltp_delta - ltd_delta

        self.trace_pre += pre_spikes
        self.trace_post += post_spikes

        self.W = np.clip(self.W, self.w_min, self.w_max)

        if self.normalize_weights:
            self._normalize()

        return float(np.mean(np.abs(self.W - W_old)))

    def _update_sparse(
        self,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
    ) -> float:
        """Update synapses using sparse weight matrix (only existing connections)."""
        W_coo = self.W.tocoo()
        W_old_data = W_coo.data.copy()

        rows = W_coo.row
        cols = W_coo.col
        data = W_coo.data

        # Compute both LTP and LTD based on original weights before applying
        ltp_delta = np.zeros_like(data)
        ltd_delta = np.zeros_like(data)

        # LTP: post fires → strengthen connections from traced pre neurons
        if np.any(post_spikes > 0):
            ltp = (post_spikes[rows] * self.trace_pre[cols]) * self.A_plus * self.learning_rate
            ltp_delta = self._scale_ltp_sparse(ltp, data)

        # LTD: pre fires → weaken connections to traced post neurons
        if np.any(pre_spikes > 0):
            ltd = (self.trace_post[rows] * pre_spikes[cols]) * self.A_minus * self.learning_rate
            ltd_delta = self._scale_ltd_sparse(ltd, data)

        data += ltp_delta - ltd_delta

        self.trace_pre += pre_spikes
        self.trace_post += post_spikes

        # Clip weights
        np.clip(data, self.w_min, self.w_max, out=data)

        # Reconstruct sparse matrix
        self.W = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(self.post_dim, self.pre_dim),
            dtype=np.float64,
        )
        if self._sparse_conn is not None:
            self._sparse_conn.W = self.W

        if self.normalize_weights:
            self._normalize_sparse()

        return float(np.mean(np.abs(data - W_old_data)))

    def _scale_ltp(self, dw: np.ndarray) -> np.ndarray:
        """Scale LTP update according to the STDP variant."""
        if self.variant == STDPVariant.ADDITIVE:
            return dw
        elif self.variant == STDPVariant.MULTIPLICATIVE:
            return dw * (self.w_max - self.W)
        elif self.variant == STDPVariant.WEIGHT_DEPENDENT:
            proximity = (self.w_max - self.W) / (self.w_max - self.w_min + 1e-10)
            return dw * proximity
        return dw

    def _scale_ltd(self, dw: np.ndarray) -> np.ndarray:
        """Scale LTD update according to the STDP variant."""
        if self.variant == STDPVariant.ADDITIVE:
            return dw
        elif self.variant == STDPVariant.MULTIPLICATIVE:
            return dw * (self.W - self.w_min)
        elif self.variant == STDPVariant.WEIGHT_DEPENDENT:
            proximity = (self.W - self.w_min) / (self.w_max - self.w_min + 1e-10)
            return dw * proximity
        return dw

    def _scale_ltp_sparse(self, dw: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Scale LTP update for sparse mode (operates on flat data arrays)."""
        if self.variant == STDPVariant.ADDITIVE:
            return dw
        elif self.variant == STDPVariant.MULTIPLICATIVE:
            return dw * (self.w_max - weights)
        elif self.variant == STDPVariant.WEIGHT_DEPENDENT:
            proximity = (self.w_max - weights) / (self.w_max - self.w_min + 1e-10)
            return dw * proximity
        return dw

    def _scale_ltd_sparse(self, dw: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Scale LTD update for sparse mode (operates on flat data arrays)."""
        if self.variant == STDPVariant.ADDITIVE:
            return dw
        elif self.variant == STDPVariant.MULTIPLICATIVE:
            return dw * (weights - self.w_min)
        elif self.variant == STDPVariant.WEIGHT_DEPENDENT:
            proximity = (weights - self.w_min) / (self.w_max - self.w_min + 1e-10)
            return dw * proximity
        return dw

    def _normalize(self) -> None:
        """
        Normalize weights so that each post-synaptic neuron's
        incoming weight sum equals the target.
        """
        row_sums = self.W.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        self.W = self.W * (self.norm_target / row_sums)
        self.W = np.clip(self.W, self.w_min, self.w_max)

    def _normalize_sparse(self) -> None:
        """Normalize weights for sparse mode so each post-synaptic neuron's
        incoming weight sum equals the target."""
        W_csr = self.W.tocsr()
        for i in range(self.post_dim):
            start, end = W_csr.indptr[i], W_csr.indptr[i + 1]
            row_sum = W_csr.data[start:end].sum()
            if row_sum > 1e-10:
                W_csr.data[start:end] *= self.norm_target / row_sum
            np.clip(W_csr.data[start:end], self.w_min, self.w_max, out=W_csr.data[start:end])
        self.W = W_csr
        if self._sparse_conn is not None:
            self._sparse_conn.W = self.W

    def forward(self, pre_spikes: np.ndarray) -> np.ndarray:
        """
        Compute post-synaptic currents (without STDP update).

        Parameters
        ----------
        pre_spikes : np.ndarray
            Pre-synaptic spike vector of shape (pre_dim,).

        Returns
        -------
        np.ndarray
            Post-synaptic currents of shape (post_dim,).
        """
        pre_spikes = np.asarray(pre_spikes, dtype=np.float64)
        return self.W.dot(pre_spikes)

    def reset(self) -> None:
        """Reset all traces to zero."""
        self.trace_pre = np.zeros(self.pre_dim, dtype=np.float64)
        self.trace_post = np.zeros(self.post_dim, dtype=np.float64)
        if self.sparse_connectivity and self._sparse_conn is not None:
            self.W = self._sparse_conn.W

    def state_dict(self) -> Dict[str, Any]:
        """Return the layer state for serialization."""
        state: Dict[str, Any] = {
            "trace_pre": self.trace_pre.copy(),
            "trace_post": self.trace_post.copy(),
            "pre_dim": self.pre_dim,
            "post_dim": self.post_dim,
            "sparse_connectivity": self.sparse_connectivity,
        }
        if self.sparse_connectivity:
            W_coo = self.W.tocoo()
            state["W_row"] = W_coo.row.copy()
            state["W_col"] = W_coo.col.copy()
            state["W_data"] = W_coo.data.copy()
            state["synapses_per_neuron"] = self.synapses_per_neuron
        else:
            state["W"] = self.W.copy()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load layer state from a dictionary."""
        self.sparse_connectivity = state.get("sparse_connectivity", False)
        self.trace_pre = state["trace_pre"].copy()
        self.trace_post = state["trace_post"].copy()
        if self.sparse_connectivity:
            self.W = sp.csr_matrix(
                (state["W_data"], (state["W_row"], state["W_col"])),
                shape=(self.post_dim, self.pre_dim),
                dtype=np.float64,
            )
            self.synapses_per_neuron = state.get("synapses_per_neuron", self.synapses_per_neuron)
            if self._sparse_conn is not None:
                self._sparse_conn.W = self.W
        else:
            self.W = state["W"].copy()

    def __repr__(self) -> str:
        sparse_info = ""
        if self.sparse_connectivity:
            nnz = self.W.nnz if sp.issparse(self.W) else np.count_nonzero(self.W)
            sparse_info = f", sparse=True, synapses/neuron={self.synapses_per_neuron}, nnz={nnz}"
        return (
            f"STDPLayer(pre_dim={self.pre_dim}, post_dim={self.post_dim}, "
            f"variant={self.variant.value}{sparse_info})"
        )
