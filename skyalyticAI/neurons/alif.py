"""
Adaptive Leaky Integrate-and-Fire (ALIF) Neuron Model

Extends the LIF model with an adaptive threshold mechanism that
models spike-frequency adaptation observed in biological neurons.

The adaptive threshold increases with each spike and decays back
to the baseline, allowing the neuron to adapt its firing rate
to sustained input -- a key property for temporal processing.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from skyalyticAI.neurons.lif import (
    IntegrationMethod,
    LIFNeuron,
    ResetMechanism,
)


class ALIFNeuron(LIFNeuron):
    """
    Adaptive Leaky Integrate-and-Fire neuron model.

    Extends LIF with an adaptive threshold:
        tau_w * dw/dt = -w
        V_threshold_adaptive = V_threshold + beta * w

    When a spike occurs:
        w <- w + 1

    The adaptation variable w increases with each spike and decays
    exponentially, causing the effective threshold to rise temporarily.
    This produces spike-frequency adaptation: the neuron fires rapidly
    at stimulus onset, then slows down even if input persists.

    Parameters
    ----------
    tau_m : float
        Membrane time constant in milliseconds.
    v_threshold : float
        Baseline spike threshold voltage.
    v_reset : float
        Reset voltage after spike emission.
    v_rest : float
        Resting membrane potential.
    resistance : float
        Membrane resistance in megaohms.
    refractory_period : float
        Absolute refractory period in milliseconds.
    reset_mechanism : ResetMechanism or str
        Reset mechanism after spike.
    integration_method : IntegrationMethod or str
        Numerical integration method.
    tau_s : float or None
        Synaptic time constant. If None, no synaptic filtering.
    tau_w : float
        Adaptation time constant in milliseconds. Controls how quickly
        the adaptation variable decays back to zero. Must be positive.
    beta : float
        Adaptation coupling strength. Determines how much each spike
        raises the effective threshold. Must be non-negative.
    """

    def __init__(
        self,
        tau_m: float = 20.0,
        v_threshold: float = -50.0,
        v_reset: float = -70.0,
        v_rest: float = -65.0,
        resistance: float = 10.0,
        refractory_period: float = 2.0,
        reset_mechanism: ResetMechanism | str = ResetMechanism.HARD,
        integration_method: IntegrationMethod | str = IntegrationMethod.EXPONENTIAL,
        tau_s: Optional[float] = None,
        tau_w: float = 100.0,
        beta: float = 0.05,
    ) -> None:
        super().__init__(
            tau_m=tau_m,
            v_threshold=v_threshold,
            v_reset=v_reset,
            v_rest=v_rest,
            resistance=resistance,
            refractory_period=refractory_period,
            reset_mechanism=reset_mechanism,
            integration_method=integration_method,
            tau_s=tau_s,
        )
        self.tau_w = self._validate_positive(tau_w, "tau_w")
        self.beta = self._validate_non_negative(beta, "beta")
        self.w: float = 0.0

    @property
    def adaptive_threshold(self) -> float:
        """Current effective threshold including adaptation."""
        return self.v_threshold + self.beta * self.w

    def step(self, current: float, dt: float = 1.0) -> Tuple[float, bool]:
        """
        Advance the neuron state by one time step.

        The adaptive threshold is used instead of the baseline threshold
        for spike generation. After a spike, the adaptation variable w
        is incremented, raising the effective threshold for subsequent
        spikes.

        Parameters
        ----------
        current : float
            Input current at this time step (in nanoamperes).
        dt : float
            Time step duration in milliseconds.

        Returns
        -------
        voltage : float
            Membrane potential after the time step.
        spike : bool
            Whether the neuron emitted a spike.
        """
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        if self.refractory_timer > 0:
            self.refractory_timer = max(0.0, self.refractory_timer - dt)
            self.v = self.v_reset
            self.i_syn = 0.0
            self.w = self._decay_adaptation(self.w, dt)
            return self.v, False

        if self.tau_s is not None:
            self.i_syn = self._update_synaptic_current(self.i_syn, current, dt)
            effective_current = self.i_syn
        else:
            effective_current = current

        if self.integration_method == IntegrationMethod.EXPONENTIAL:
            self.v = self._exponential_euler_step(self.v, effective_current, dt)
        else:
            self.v = self._euler_step(self.v, effective_current, dt)

        self.w = self._decay_adaptation(self.w, dt)

        effective_threshold = self.adaptive_threshold
        spike = self.v >= effective_threshold
        if spike:
            if self.reset_mechanism == ResetMechanism.HARD:
                self.v = self.v_reset
            else:
                self.v = self.v - effective_threshold
            self.refractory_timer = self.refractory_period
            self.w += 1.0
            self.spike_count += 1

        return self.v, spike

    def _decay_adaptation(self, w: float, dt: float) -> float:
        """
        Exponential decay of adaptation variable.

        w(t+dt) = w(t) * exp(-dt/tau_w)
        """
        return w * np.exp(-dt / self.tau_w)

    def reset(self) -> None:
        """Reset neuron to resting state, including adaptation."""
        super().reset()
        self.w = 0.0

    def state_dict(self) -> Dict[str, Any]:
        """Return the current state as a dictionary for serialization."""
        state = super().state_dict()
        state.update({
            "w": self.w,
            "tau_w": self.tau_w,
            "beta": self.beta,
        })
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state from a dictionary."""
        super().load_state_dict(state)
        self.w = state["w"]

    def __repr__(self) -> str:
        return (
            f"ALIFNeuron(tau_m={self.tau_m}, v_threshold={self.v_threshold}, "
            f"v_reset={self.v_reset}, v_rest={self.v_rest}, "
            f"resistance={self.resistance}, "
            f"refractory_period={self.refractory_period}, "
            f"reset={self.reset_mechanism.value}, "
            f"integration={self.integration_method.value}, "
            f"tau_w={self.tau_w}, beta={self.beta})"
        )
