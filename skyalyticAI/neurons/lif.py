"""
Leaky Integrate-and-Fire (LIF) Neuron Model

Industrial-grade implementation with:
- Exponential Euler integration for numerical stability
- Refractory period support
- Hard and soft reset mechanisms
- Synaptic current filtering (optional)
- Batch processing support
- Complete state management and serialization
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np


class ResetMechanism(Enum):
    HARD = "hard"
    SOFT = "soft"


class IntegrationMethod(Enum):
    EULER = "euler"
    EXPONENTIAL = "exponential"


class LIFNeuron:
    """
    Leaky Integrate-and-Fire neuron model.

    Implements the LIF differential equation:
        tau_m * dV/dt = -(V - V_rest) + R * I(t)

    When V >= V_threshold, a spike is emitted and V is reset according
    to the configured reset mechanism.

    Parameters
    ----------
    tau_m : float
        Membrane time constant in milliseconds. Controls how quickly
        the membrane potential decays toward rest. Must be positive.
    v_threshold : float
        Spike threshold voltage. When membrane potential reaches this
        value, a spike is emitted. Must be greater than v_reset.
    v_reset : float
        Reset voltage after spike emission. For hard reset, the membrane
        potential is set to this value. For soft reset, the threshold
        is subtracted from the membrane potential.
    v_rest : float
        Resting membrane potential. The membrane potential decays toward
        this value in the absence of input.
    resistance : float
        Membrane resistance in megaohms. Scales the input current to
        voltage. Must be non-negative.
    refractory_period : float
        Absolute refractory period in milliseconds. During this time
        after a spike, the neuron cannot fire again. Must be non-negative.
    reset_mechanism : ResetMechanism or str
        Reset mechanism after spike. 'hard' sets V = v_reset,
        'soft' sets V = V - v_threshold (preserves sub-threshold dynamics).
    integration_method : IntegrationMethod or str
        Numerical integration method. 'euler' uses forward Euler,
        'exponential' uses exact exponential integration (more stable).
    tau_s : float or None
        Synaptic time constant in milliseconds. If provided, input current
        is filtered through a first-order low-pass filter before being
        applied to the membrane. Must be positive if provided.
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
    ) -> None:
        self.tau_m = self._validate_positive(tau_m, "tau_m")
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.resistance = self._validate_non_negative(resistance, "resistance")
        self.refractory_period = self._validate_non_negative(
            refractory_period, "refractory_period"
        )

        if isinstance(reset_mechanism, str):
            reset_mechanism = ResetMechanism(reset_mechanism.lower())
        self.reset_mechanism = reset_mechanism

        if isinstance(integration_method, str):
            integration_method = IntegrationMethod(integration_method.lower())
        self.integration_method = integration_method

        if tau_s is not None:
            self.tau_s = self._validate_positive(tau_s, "tau_s")
        else:
            self.tau_s = None

        if v_threshold <= v_reset:
            raise ValueError(
                f"v_threshold ({v_threshold}) must be greater than v_reset ({v_reset})"
            )

        self.v: float = v_rest
        self.i_syn: float = 0.0
        self.refractory_timer: float = 0.0
        self.spike_count: int = 0

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

    def step(self, current: float, dt: float = 1.0) -> Tuple[float, bool]:
        """
        Advance the neuron state by one time step.

        Parameters
        ----------
        current : float
            Input current at this time step (in nanoamperes).
        dt : float
            Time step duration in milliseconds. Must be positive.

        Returns
        -------
        voltage : float
            Membrane potential after the time step.
        spike : bool
            Whether the neuron emitted a spike during this time step.
        """
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        if self.refractory_timer > 0:
            self.refractory_timer = max(0.0, self.refractory_timer - dt)
            self.v = self.v_reset
            self.i_syn = 0.0
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

        spike = bool(self.v >= self.v_threshold)
        if spike:
            if self.reset_mechanism == ResetMechanism.HARD:
                self.v = self.v_reset
            else:
                self.v = self.v - self.v_threshold
            self.refractory_timer = self.refractory_period
            self.spike_count += 1

        return self.v, spike

    def forward(
        self, currents: np.ndarray, dt: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a sequence of input currents.

        Parameters
        ----------
        currents : np.ndarray
            1D array of input currents over time.
        dt : float
            Time step duration in milliseconds.

        Returns
        -------
        voltages : np.ndarray
            Membrane potential at each time step.
        spikes : np.ndarray
            Boolean array indicating spike events.
        """
        currents = np.asarray(currents, dtype=np.float64)
        if currents.ndim != 1:
            raise ValueError(f"currents must be 1D, got {currents.ndim}D")

        n_steps = len(currents)
        voltages = np.zeros(n_steps, dtype=np.float64)
        spikes = np.zeros(n_steps, dtype=bool)

        for t in range(n_steps):
            v, s = self.step(float(currents[t]), dt)
            voltages[t] = v
            spikes[t] = s

        return voltages, spikes

    def _euler_step(self, v: float, current: float, dt: float) -> float:
        """Forward Euler integration step."""
        dv = (-(v - self.v_rest) + self.resistance * current) * (dt / self.tau_m)
        return v + dv

    def _exponential_euler_step(self, v: float, current: float, dt: float) -> float:
        """
        Exponential Euler integration step.

        Solves the ODE exactly for constant input current over the
        time step, providing superior numerical stability.

        V(t+dt) = V_rest + (V(t) - V_rest) * exp(-dt/tau_m)
                  + R * I * (1 - exp(-dt/tau_m))
        """
        decay = np.exp(-dt / self.tau_m)
        v_inf = self.v_rest + self.resistance * current
        return v_inf + (v - v_inf) * decay

    def _update_synaptic_current(
        self, i_syn: float, current: float, dt: float
    ) -> float:
        """
        Update synaptic current with first-order low-pass filter.

        tau_s * dI/dt = -I + current
        """
        decay = np.exp(-dt / self.tau_s)
        return current + (i_syn - current) * decay

    def reset(self) -> None:
        """Reset neuron to resting state."""
        self.v = self.v_rest
        self.i_syn = 0.0
        self.refractory_timer = 0.0
        self.spike_count = 0

    def state_dict(self) -> Dict[str, Any]:
        """Return the current state as a dictionary for serialization."""
        return {
            "v": self.v,
            "i_syn": self.i_syn,
            "refractory_timer": self.refractory_timer,
            "spike_count": self.spike_count,
            "tau_m": self.tau_m,
            "v_threshold": self.v_threshold,
            "v_reset": self.v_reset,
            "v_rest": self.v_rest,
            "resistance": self.resistance,
            "refractory_period": self.refractory_period,
            "reset_mechanism": self.reset_mechanism.value,
            "integration_method": self.integration_method.value,
            "tau_s": self.tau_s,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state from a dictionary."""
        self.v = state["v"]
        self.i_syn = state["i_syn"]
        self.refractory_timer = state["refractory_timer"]
        self.spike_count = state["spike_count"]

    def __repr__(self) -> str:
        return (
            f"LIFNeuron(tau_m={self.tau_m}, v_threshold={self.v_threshold}, "
            f"v_reset={self.v_reset}, v_rest={self.v_rest}, "
            f"resistance={self.resistance}, "
            f"refractory_period={self.refractory_period}, "
            f"reset={self.reset_mechanism.value}, "
            f"integration={self.integration_method.value})"
        )
