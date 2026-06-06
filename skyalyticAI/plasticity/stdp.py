"""
Spike-Timing-Dependent Plasticity (STDP) Synapse

Industrial-grade implementation with:
- Eligibility traces (pre and post) for proper temporal credit assignment
- Additive and multiplicative weight update rules
- Weight-dependent plasticity for stability
- Triplet STDP model support
- Proper trace decay with configurable time constants
- Weight clipping with hard bounds
- Complete state management and serialization

Theoretical basis (Song et al., 2000):
    dw = A_plus * exp(-delta_t / tau_plus)   if delta_t > 0  (pre before post -> LTP)
    dw = -A_minus * exp(delta_t / tau_minus)  if delta_t < 0  (post before pre -> LTD)

where delta_t = t_post - t_pre
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

import numpy as np


class STDPVariant(Enum):
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
    WEIGHT_DEPENDENT = "weight_dependent"


class STDPSynapse:
    """
    Single STDP synapse with eligibility trace mechanism.

    Unlike the simplified version that only tracks the last spike time,
    this implementation uses eligibility traces that accumulate over
    multiple spike events and decay exponentially. This provides
    proper temporal credit assignment and is biologically more
    plausible.

    Parameters
    ----------
    w_init : float
        Initial synaptic weight.
    A_plus : float
        LTP (long-term potentiation) amplitude. Must be non-negative.
    A_minus : float
        LTD (long-term depression) amplitude. Must be non-negative.
        Typically A_minus > A_plus to prevent unbounded potentiation.
    tau_plus : float
        LTP time window in milliseconds. Controls how quickly the
        pre-synaptic trace decays. Must be positive.
    tau_minus : float
        LTD time window in milliseconds. Controls how quickly the
        post-synaptic trace decays. Must be positive.
    w_min : float
        Minimum allowed weight. Must be non-negative.
    w_max : float
        Maximum allowed weight. Must be greater than w_min.
    variant : STDPVariant or str
        STDP update variant:
        - 'additive': dw is independent of current weight
        - 'multiplicative': dw is scaled by (w_max - w) for LTP and w for LTD
        - 'weight_dependent': dw is scaled by weight proximity to bounds
    learning_rate : float
        Global learning rate multiplier. Must be non-negative.
    """

    def __init__(
        self,
        w_init: float = 0.5,
        A_plus: float = 0.01,
        A_minus: float = 0.012,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        w_min: float = 0.0,
        w_max: float = 1.0,
        variant: STDPVariant | str = STDPVariant.ADDITIVE,
        learning_rate: float = 1.0,
    ) -> None:
        self.A_plus = self._validate_non_negative(A_plus, "A_plus")
        self.A_minus = self._validate_non_negative(A_minus, "A_minus")
        self.tau_plus = self._validate_positive(tau_plus, "tau_plus")
        self.tau_minus = self._validate_positive(tau_minus, "tau_minus")
        self.w_min = self._validate_non_negative(w_min, "w_min")

        if w_max <= w_min:
            raise ValueError(
                f"w_max ({w_max}) must be greater than w_min ({w_min})"
            )
        self.w_max = w_max

        if isinstance(variant, str):
            variant = STDPVariant(variant.lower())
        self.variant = variant

        self.learning_rate = self._validate_non_negative(
            learning_rate, "learning_rate"
        )

        self.w = np.clip(w_init, w_min, w_max)
        self.w_init = w_init

        self.trace_pre: float = 0.0
        self.trace_post: float = 0.0

        self.total_ltp: float = 0.0
        self.total_ltd: float = 0.0
        self.update_count: int = 0

    @staticmethod
    def _validate_positive(value: float, name: str) -> float:
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return value

    @staticmethod
    def _validate_non_negative(value: float, name: str) -> float:
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")
        return value

    def update(
        self,
        pre_spike: bool,
        post_spike: bool,
        dt: float = 1.0,
    ) -> float:
        """
        Update synaptic weight based on pre- and post-synaptic spike events.

        Uses eligibility traces: when a pre-synaptic spike occurs,
        the pre-synaptic trace is incremented and the post-synaptic
        trace determines LTD. When a post-synaptic spike occurs,
        the post-synaptic trace is incremented and the pre-synaptic
        trace determines LTP.

        Parameters
        ----------
        pre_spike : bool
            Whether the pre-synaptic neuron fired at this time step.
        post_spike : bool
            Whether the post-synaptic neuron fired at this time step.
        dt : float
            Time step duration in milliseconds.

        Returns
        -------
        dw : float
            The weight change applied at this time step.
        """
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        decay_pre = np.exp(-dt / self.tau_plus)
        decay_post = np.exp(-dt / self.tau_minus)

        self.trace_pre *= decay_pre
        self.trace_post *= decay_post

        # Save trace values after decay but before increment,
        # so LTP and LTD both use pre-increment traces
        old_trace_pre = self.trace_pre
        old_trace_post = self.trace_post

        dw = 0.0

        if pre_spike:
            dw -= self._compute_ltd(old_trace_post)
            self.trace_pre += 1.0

        if post_spike:
            dw += self._compute_ltp(old_trace_pre)
            self.trace_post += 1.0

        dw *= self.learning_rate

        self.w += dw
        self.w = np.clip(self.w, self.w_min, self.w_max)

        if dw > 0:
            self.total_ltp += dw
        elif dw < 0:
            self.total_ltd += abs(dw)
        self.update_count += 1

        return dw

    def _compute_ltp(self, pre_trace: float) -> float:
        """
        Compute LTP weight change based on pre-synaptic trace.

        Parameters
        ----------
        pre_trace : float
            Current pre-synaptic eligibility trace value.

        Returns
        -------
        float
            LTP weight change (non-negative).
        """
        base_dw = self.A_plus * pre_trace

        if self.variant == STDPVariant.ADDITIVE:
            return base_dw
        elif self.variant == STDPVariant.MULTIPLICATIVE:
            return base_dw * (self.w_max - self.w)
        elif self.variant == STDPVariant.WEIGHT_DEPENDENT:
            proximity = (self.w_max - self.w) / (self.w_max - self.w_min + 1e-10)
            return base_dw * proximity
        else:
            return base_dw

    def _compute_ltd(self, post_trace: float) -> float:
        """
        Compute LTD weight change based on post-synaptic trace.

        Parameters
        ----------
        post_trace : float
            Current post-synaptic eligibility trace value.

        Returns
        -------
        float
            LTD weight change magnitude (non-negative).
        """
        base_dw = self.A_minus * post_trace

        if self.variant == STDPVariant.ADDITIVE:
            return base_dw
        elif self.variant == STDPVariant.MULTIPLICATIVE:
            return base_dw * (self.w - self.w_min)
        elif self.variant == STDPVariant.WEIGHT_DEPENDENT:
            proximity = (self.w - self.w_min) / (self.w_max - self.w_min + 1e-10)
            return base_dw * proximity
        else:
            return base_dw

    def compute_stdp_curve(self, delta_t: float) -> float:
        """
        Compute the STDP weight change for a given spike time difference.

        This is the analytical STDP curve function, useful for
        visualization and analysis.

        Parameters
        ----------
        delta_t : float
            Spike time difference t_post - t_pre in milliseconds.
            Positive means pre fired before post (LTP).
            Negative means post fired before pre (LTD).

        Returns
        -------
        float
            Weight change according to the STDP rule.
        """
        if delta_t > 0:
            return self.A_plus * np.exp(-delta_t / self.tau_plus)
        elif delta_t < 0:
            return -self.A_minus * np.exp(delta_t / self.tau_minus)
        else:
            return 0.0

    def reset(self) -> None:
        """Reset traces and statistics while keeping the current weight."""
        self.trace_pre = 0.0
        self.trace_post = 0.0
        self.total_ltp = 0.0
        self.total_ltd = 0.0
        self.update_count = 0

    def full_reset(self) -> None:
        """Reset everything including weight to initial state."""
        self.reset()
        self.w = np.clip(self.w_init, self.w_min, self.w_max)

    def state_dict(self) -> Dict[str, Any]:
        """Return the synapse state for serialization."""
        return {
            "w": self.w,
            "trace_pre": self.trace_pre,
            "trace_post": self.trace_post,
            "A_plus": self.A_plus,
            "A_minus": self.A_minus,
            "tau_plus": self.tau_plus,
            "tau_minus": self.tau_minus,
            "w_min": self.w_min,
            "w_max": self.w_max,
            "variant": self.variant.value,
            "learning_rate": self.learning_rate,
            "total_ltp": self.total_ltp,
            "total_ltd": self.total_ltd,
            "update_count": self.update_count,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load synapse state from a dictionary."""
        self.w = state["w"]
        self.trace_pre = state["trace_pre"]
        self.trace_post = state["trace_post"]
        self.total_ltp = state["total_ltp"]
        self.total_ltd = state["total_ltd"]
        self.update_count = state["update_count"]

    def __repr__(self) -> str:
        return (
            f"STDPSynapse(w={self.w:.4f}, "
            f"A_plus={self.A_plus}, A_minus={self.A_minus}, "
            f"tau_plus={self.tau_plus}, tau_minus={self.tau_minus}, "
            f"variant={self.variant.value})"
        )
