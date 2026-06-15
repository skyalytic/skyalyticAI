"""
Structural Self-Evolution Module

Implements the self-evolution mechanism for NIEA, enabling the network
to autonomously modify its own architecture during training. This is
the seventh core module's structural evolution component.

Key mechanisms:
1. Neurogenesis: Create new neurons when existing capacity is insufficient
2. Synaptic Pruning: Remove weak/unused connections to improve efficiency
3. Structural Plasticity: Rewire connections based on activity correlation
4. Module Growth: Expand module dimensions when performance plateaus
5. Evolutionary Pressure: Use fitness signals to guide structural changes

This module operates on a slower timescale than learning (epochs, not steps).
Structural changes are applied periodically based on performance metrics.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class StructuralEvolution:
    """
    Structural Self-Evolution for NIEA modules.

    Monitors module performance and applies structural changes
    when performance plateaus or specific conditions are met.

    Parameters
    ----------
    pruning_threshold : float
        Weight magnitude below which connections are pruned.
    neurogenesis_threshold : float
        Performance threshold below which new neurons are added.
    growth_rate : int
        Number of neurons to add during neurogenesis.
    max_prune_fraction : float
        Maximum fraction of connections that can be pruned per cycle.
    evolution_interval : int
        Number of training steps between evolution cycles.
    activity_tracking_window : int
        Window size for tracking neuron activity statistics.
    """

    def __init__(
        self,
        pruning_threshold: float = 0.01,
        neurogenesis_threshold: float = 0.1,
        growth_rate: int = 4,
        max_prune_fraction: float = 0.1,
        evolution_interval: int = 500,
        activity_tracking_window: int = 100,
    ) -> None:
        if pruning_threshold < 0:
            raise ValueError("pruning_threshold must be non-negative")
        if growth_rate <= 0:
            raise ValueError("growth_rate must be positive")
        if not 0 < max_prune_fraction <= 1:
            raise ValueError("max_prune_fraction must be in (0, 1]")
        if evolution_interval <= 0:
            raise ValueError("evolution_interval must be positive")

        self.pruning_threshold = pruning_threshold
        self.neurogenesis_threshold = neurogenesis_threshold
        self.growth_rate = growth_rate
        self.max_prune_fraction = max_prune_fraction
        self.evolution_interval = evolution_interval
        self.activity_tracking_window = activity_tracking_window

        self._step_counter: int = 0
        self._evolution_history: List[Dict[str, Any]] = []
        self._performance_history: List[float] = []
        self._neuron_activity: Dict[str, np.ndarray] = {}
        self._weight_history: Dict[str, List[float]] = {}

    def should_evolve(self) -> bool:
        """Check if an evolution cycle should be triggered."""
        self._step_counter += 1
        return self._step_counter >= self.evolution_interval

    def record_performance(self, performance: float) -> None:
        """
        Record a performance measurement for plateau detection.

        Parameters
        ----------
        performance : float
            Current performance metric (higher is better).
        """
        self._performance_history.append(performance)
        if len(self._performance_history) > 1000:
            self._performance_history = self._performance_history[-500:]

    def record_neuron_activity(
        self, module_name: str, activity: np.ndarray
    ) -> None:
        """
        Record neuron activity for a module.

        Parameters
        ----------
        module_name : str
            Name of the module.
        activity : np.ndarray
            Neuron activation values, shape (n_neurons,).
        """
        activity = np.asarray(activity, dtype=np.float64).flatten()
        if module_name not in self._neuron_activity:
            self._neuron_activity[module_name] = np.zeros(
                self.activity_tracking_window, dtype=np.float64
            )
        self._neuron_activity[module_name] = np.roll(
            self._neuron_activity[module_name], -1
        )
        self._neuron_activity[module_name][-1] = float(np.mean(activity))

    def detect_plateau(self, window: int = 50) -> bool:
        """
        Detect if performance has plateaued.

        A plateau is detected when the performance improvement
        over the recent window is below a small threshold.

        Parameters
        ----------
        window : int
            Number of recent measurements to check.

        Returns
        -------
        bool
            True if performance has plateaued.
        """
        if len(self._performance_history) < 2 * window:
            return False

        recent = self._performance_history[-window:]
        previous = self._performance_history[-2 * window : -window]

        recent_mean = np.mean(recent)
        previous_mean = np.mean(previous)

        if abs(previous_mean) < 1e-10:
            return False

        improvement = (recent_mean - previous_mean) / (abs(previous_mean) + 1e-10)
        return improvement < self.neurogenesis_threshold

    def prune_weights(
        self, weight_matrix: np.ndarray, bias: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """
        Prune weak connections from a weight matrix.

        Removes connections whose magnitude is below the pruning
        threshold, up to a maximum fraction of total connections.

        Parameters
        ----------
        weight_matrix : np.ndarray
            Weight matrix of shape (out_dim, in_dim).
        bias : np.ndarray or None
            Bias vector of shape (out_dim,).

        Returns
        -------
        weight_matrix : np.ndarray
            Pruned weight matrix.
        bias : np.ndarray or None
            Pruned bias vector (dead neurons removed).
        info : dict
            Pruning statistics.
        """
        W = weight_matrix.copy()

        # 处理稀疏矩阵：转为稠密进行剪枝
        is_sparse = hasattr(W, 'toarray')
        if is_sparse:
            W = W.toarray()

        abs_W = np.abs(W)
        total_connections = W.size

        mask = abs_W < self.pruning_threshold
        n_prunable = int(np.sum(mask))
        max_prune = int(total_connections * self.max_prune_fraction)

        if n_prunable > max_prune:
            flat_abs = abs_W.flatten()
            sorted_indices = np.argsort(flat_abs)
            prune_mask = np.zeros(flat_abs.shape, dtype=bool)
            prune_mask[sorted_indices[:max_prune]] = True
            mask = prune_mask.reshape(abs_W.shape)

        W[mask] = 0.0

        n_pruned = int(np.sum(mask))
        sparsity = float(np.sum(W == 0)) / W.size

        info = {
            "n_pruned": n_pruned,
            "sparsity": sparsity,
            "original_connections": total_connections,
        }

        if bias is not None:
            dead_neurons = np.all(W == 0, axis=1)
            bias_out = bias.copy()
            bias_out[dead_neurons] = 0.0
            info["n_dead_neurons"] = int(np.sum(dead_neurons))
            return W, bias_out, info

        return W, bias, info

    def grow_neurons(
        self,
        weight_matrix: np.ndarray,
        bias: Optional[np.ndarray] = None,
        n_new: Optional[int] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """
        Add new neurons to a weight matrix (neurogenesis).

        .. deprecated::
            grow_neurons() adds rows to the weight matrix, changing the
            output dimension. Downstream layers still expect the original
            dimension, so this method requires manual dimension
            synchronization across all dependent modules. Use pruning only
            in evolve_module() unless you can guarantee consistency.

        New neurons are initialized with small random weights
        connected to the most active input neurons.

        Parameters
        ----------
        weight_matrix : np.ndarray
            Weight matrix of shape (out_dim, in_dim).
        bias : np.ndarray or None
            Bias vector of shape (out_dim,).
        n_new : int or None
            Number of new neurons to add. If None, uses growth_rate.

        Returns
        -------
        weight_matrix : np.ndarray
            Expanded weight matrix of shape (out_dim + n_new, in_dim).
        bias : np.ndarray or None
            Expanded bias vector.
        info : dict
            Growth statistics.
        """
        import warnings
        warnings.warn(
            "grow_neurons() changes output dimensions and requires manual "
            "synchronization with downstream layers. Prefer pruning-only "
            "evolution via evolve_module().",
            DeprecationWarning,
            stacklevel=2,
        )

        if n_new is None:
            n_new = self.growth_rate

        out_dim, in_dim = weight_matrix.shape

        new_weights = np.random.randn(n_new, in_dim) * np.sqrt(
            2.0 / (in_dim + n_new)
        )
        new_W = np.vstack([weight_matrix, new_weights])

        new_bias = None
        if bias is not None:
            new_bias = np.zeros(out_dim + n_new, dtype=np.float64)
            new_bias[:out_dim] = bias

        info = {
            "n_new_neurons": n_new,
            "old_dim": out_dim,
            "new_dim": out_dim + n_new,
        }

        return new_W, new_bias, info

    def rewire_connections(
        self,
        weight_matrix: np.ndarray,
        activity_correlation: np.ndarray,
        rewiring_fraction: float = 0.05,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Rewire connections based on activity correlation.

        Connections between highly correlated neurons are strengthened,
        while connections between anti-correlated neurons are weakened.

        Parameters
        ----------
        weight_matrix : np.ndarray
            Weight matrix of shape (out_dim, in_dim).
        activity_correlation : np.ndarray
            Correlation matrix of shape (in_dim, in_dim).
        rewiring_fraction : float
            Fraction of connections to rewire.

        Returns
        -------
        weight_matrix : np.ndarray
            Rewired weight matrix.
        info : dict
            Rewiring statistics.
        """
        W = weight_matrix.copy()
        out_dim, in_dim = W.shape

        n_rewire = max(1, int(W.size * rewiring_fraction))

        abs_W = np.abs(W)
        weak_indices = np.argsort(abs_W.flatten())[:n_rewire]

        n_rewired = 0
        for idx in weak_indices:
            i = idx // in_dim
            j = idx % in_dim

            if activity_correlation.shape[0] > j:
                correlations = activity_correlation[j]
                valid_len = min(len(correlations), in_dim)
                corr_abs = np.abs(correlations[:valid_len]).copy()
                if j < valid_len:
                    corr_abs[j] = -1.0  # Exclude self-correlation
                best_partner = int(np.argmax(corr_abs))
                if best_partner != j:
                    transferred = W[i, j] * 0.5
                    W[i, best_partner] += transferred
                    W[i, j] = 0.0
                    # Clip to prevent unbounded growth
                    W[i, best_partner] = np.clip(W[i, best_partner], -2.0, 2.0)
                    n_rewired += 1

        info = {
            "n_rewired": n_rewired,
            "rewiring_fraction": rewiring_fraction,
        }

        return W, info

    def evolve_module(
        self,
        module_name: str,
        weight_matrix: np.ndarray,
        bias: Optional[np.ndarray] = None,
        activity_correlation: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """
        Apply a full evolution cycle to a module.

        1. Prune weak connections
        2. If plateaued, grow new neurons
        3. If correlation data available, rewire connections

        Parameters
        ----------
        module_name : str
            Name of the module being evolved.
        weight_matrix : np.ndarray
            Current weight matrix.
        bias : np.ndarray or None
            Current bias vector.
        activity_correlation : np.ndarray or None
            Activity correlation matrix for rewiring.

        Returns
        -------
        weight_matrix : np.ndarray
            Evolved weight matrix.
        bias : np.ndarray or None
            Evolved bias vector.
        info : dict
            Complete evolution statistics.
        """
        self._step_counter = 0

        W, b, prune_info = self.prune_weights(weight_matrix, bias)
        info = {"prune": prune_info}

        # NOTE: grow_neurons() is NOT called here because it changes
        # output dimensions, which breaks downstream layer compatibility.
        # Use pruning only; neurogenesis requires manual dimension sync.

        if activity_correlation is not None:
            W, rewire_info = self.rewire_connections(W, activity_correlation)
            info["rewire"] = rewire_info

        info["module"] = module_name
        self._evolution_history.append(info)
        if len(self._evolution_history) > 1000:
            self._evolution_history = self._evolution_history[-500:]

        return W, b, info

    def get_evolution_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all evolution operations.

        Returns
        -------
        dict
            Summary statistics.
        """
        total_pruned = sum(
            e.get("prune", {}).get("n_pruned", 0)
            for e in self._evolution_history
        )
        total_grown = sum(
            e.get("grow", {}).get("n_new_neurons", 0)
            for e in self._evolution_history
        )
        total_rewired = sum(
            e.get("rewire", {}).get("n_rewired", 0)
            for e in self._evolution_history
        )

        return {
            "total_evolution_cycles": len(self._evolution_history),
            "total_pruned_connections": total_pruned,
            "total_new_neurons": total_grown,
            "total_rewired_connections": total_rewired,
            "current_step": self._step_counter,
            "performance_trend": (
                "improving"
                if len(self._performance_history) > 10
                and self._performance_history[-1]
                > np.mean(self._performance_history[-10:])
                else "plateaued"
                if len(self._performance_history) > 10
                else "insufficient_data"
            ),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Return evolution state for serialization."""
        return {
            "step_counter": self._step_counter,
            "performance_history": list(self._performance_history[-500:]),
            "evolution_history": self._evolution_history[-50:],
            "pruning_threshold": self.pruning_threshold,
            "neurogenesis_threshold": self.neurogenesis_threshold,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load evolution state from a dictionary."""
        self._step_counter = state.get("step_counter", 0)
        self._performance_history = state.get("performance_history", [])
        self._evolution_history = state.get("evolution_history", [])
        if "pruning_threshold" in state:
            self.pruning_threshold = state["pruning_threshold"]
        if "neurogenesis_threshold" in state:
            self.neurogenesis_threshold = state["neurogenesis_threshold"]
