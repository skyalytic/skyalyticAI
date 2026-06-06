"""
Spiking Neural Network Layer

A layer of LIF/ALIF neurons with weight matrix connectivity.
Supports rate coding input, temporal coding, and batch processing.
When PyTorch is available, supports GPU-accelerated forward pass.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

from skyalyticAI.neurons.lif import LIFNeuron, IntegrationMethod, ResetMechanism
from skyalyticAI.neurons.sparse_connectivity import SparseConnectivity

_TORCH_AVAILABLE = False
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass


class SNNLayer:
    """
    A layer of spiking neurons with full connectivity.

    Implements a population of LIF or ALIF neurons that receive
    input through a weight matrix, process it through membrane
    dynamics, and produce spike outputs.

    Parameters
    ----------
    input_dim : int
        Number of input features (pre-synaptic neurons).
    output_dim : int
        Number of neurons in this layer.
    neuron_type : type
        Neuron class to use (LIFNeuron or ALIFNeuron).
    neuron_params : dict or None
        Parameters passed to each neuron constructor. If None,
        default parameters are used.
    weight_init : str
        Weight initialization method. Supported: 'normal', 'uniform',
        'xavier_normal', 'xavier_uniform', 'kaiming_normal'.
    weight_scale : float
        Scale factor for weight initialization.
    use_bias : bool
        Whether to include bias current for each neuron.
    sparse_connectivity : bool
        If True, use SparseConnectivity (scipy.sparse.csr_matrix)
        instead of dense numpy array for the weight matrix.
    synapses_per_neuron : int
        Number of incoming synapses per post-synaptic neuron when
        sparse_connectivity=True. Default 7000 (matching human brain).
    """

    SUPPORTED_WEIGHT_INIT = {
        "normal", "uniform", "xavier_normal", "xavier_uniform", "kaiming_normal"
    }

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        neuron_type: Type[LIFNeuron] = LIFNeuron,
        neuron_params: Optional[Dict[str, Any]] = None,
        weight_init: str = "xavier_normal",
        weight_scale: float = 1.0,
        use_bias: bool = True,
        device: Any = None,
        sparse_connectivity: bool = False,
        synapses_per_neuron: int = 7000,
    ) -> None:
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if weight_init not in self.SUPPORTED_WEIGHT_INIT:
            raise ValueError(
                f"weight_init must be one of {self.SUPPORTED_WEIGHT_INIT}, "
                f"got '{weight_init}'"
            )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neuron_type = neuron_type
        self.use_bias = use_bias
        self.device = device
        self.sparse = sparse_connectivity
        self.synapses_per_neuron = synapses_per_neuron

        if self.sparse:
            self._sparse_conn = SparseConnectivity(
                n_pre=input_dim,
                n_post=output_dim,
                synapses_per_neuron=synapses_per_neuron,
                weight_init="glorot" if weight_init in ("xavier_normal", "xavier_uniform") else weight_init,
                weight_scale=weight_scale,
            )
            self.W = self._sparse_conn.W
        else:
            self._sparse_conn = None
            self.W = self._initialize_weights(
                input_dim, output_dim, weight_init, weight_scale
            )

        if use_bias:
            self.bias = np.zeros(output_dim, dtype=np.float64)
        else:
            self.bias = None

        params = neuron_params.copy() if neuron_params else {}
        self.neurons: List[LIFNeuron] = []
        for _ in range(output_dim):
            self.neurons.append(neuron_type(**params))

        self._use_gpu = (
            _TORCH_AVAILABLE
            and device is not None
            and str(device) == "cuda"
        )

        self._init_neuron_state()

    def _init_neuron_state(self) -> None:
        """Initialize vectorized neuron state arrays for GPU forward pass."""
        if not self.neurons:
            return
        n0 = self.neurons[0]
        self._v = np.full(self.output_dim, n0.v_rest, dtype=np.float64)
        self._refractory = np.zeros(self.output_dim, dtype=np.float64)
        self._spike_count = np.zeros(self.output_dim, dtype=np.float64)
        self._total_current = np.zeros(self.output_dim, dtype=np.float64)

        if hasattr(n0, "w"):
            self._w = np.zeros(self.output_dim, dtype=np.float64)
        else:
            self._w = None

    def _sync_from_neurons(self) -> None:
        """Sync vectorized state from individual neuron objects."""
        for j, neuron in enumerate(self.neurons):
            self._v[j] = neuron.v
            self._refractory[j] = neuron.refractory_timer
            self._spike_count[j] = neuron.spike_count
            self._total_current[j] = neuron.i_syn
            if self._w is not None and hasattr(neuron, "w"):
                self._w[j] = neuron.w

    def _sync_to_neurons(self) -> None:
        """Sync vectorized state back to individual neuron objects."""
        for j, neuron in enumerate(self.neurons):
            neuron.v = float(self._v[j])
            neuron.refractory_timer = float(self._refractory[j])
            neuron.spike_count = int(self._spike_count[j])
            neuron.i_syn = float(self._total_current[j])
            if self._w is not None and hasattr(neuron, "w"):
                neuron.w = float(self._w[j])

    def _initialize_weights(
        self,
        input_dim: int,
        output_dim: int,
        method: str,
        scale: float,
    ) -> np.ndarray:
        """
        Initialize weight matrix using the specified method.

        Parameters
        ----------
        input_dim : int
            Number of input units.
        output_dim : int
            Number of output units.
        method : str
            Initialization method name.
        scale : float
            Scale factor.

        Returns
        -------
        np.ndarray
            Weight matrix of shape (output_dim, input_dim).
        """
        rng = np.random.default_rng()

        if method == "normal":
            W = rng.standard_normal((output_dim, input_dim)) * scale
        elif method == "uniform":
            limit = scale / np.sqrt(input_dim)
            W = rng.uniform(-limit, limit, (output_dim, input_dim))
        elif method == "xavier_normal":
            std = scale * np.sqrt(2.0 / (input_dim + output_dim))
            W = rng.standard_normal((output_dim, input_dim)) * std
        elif method == "xavier_uniform":
            limit = scale * np.sqrt(6.0 / (input_dim + output_dim))
            W = rng.uniform(-limit, limit, (output_dim, input_dim))
        elif method == "kaiming_normal":
            std = scale * np.sqrt(2.0 / input_dim)
            W = rng.standard_normal((output_dim, input_dim)) * std
        else:
            raise ValueError(f"Unknown initialization method: {method}")

        return W.astype(np.float64)

    def forward(
        self,
        spike_train: np.ndarray,
        dt: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a spike train through the layer.

        When GPU is available and device is set to CUDA, uses
        vectorized PyTorch computation for acceleration.
        Otherwise, uses the standard per-neuron loop.

        Parameters
        ----------
        spike_train : np.ndarray
            Input spike train of shape (n_steps, input_dim) or (input_dim,).
            Values are typically 0 or 1 (binary spikes) but can be
            floating point (rate-coded).
        dt : float
            Time step duration in milliseconds.

        Returns
        -------
        output_spikes : np.ndarray
            Output spike train of shape (n_steps, output_dim).
        voltages : np.ndarray
            Membrane potentials of shape (n_steps, output_dim).
        """
        spike_train = np.asarray(spike_train, dtype=np.float64)

        if spike_train.ndim == 1:
            spike_train = spike_train.reshape(1, -1)
        elif spike_train.ndim != 2:
            raise ValueError(
                f"spike_train must be 1D or 2D, got {spike_train.ndim}D"
            )

        if spike_train.shape[1] != self.input_dim:
            raise ValueError(
                f"spike_train has {spike_train.shape[1]} features, "
                f"expected {self.input_dim}"
            )

        if self._use_gpu and not self.sparse:
            return self._forward_gpu(spike_train, dt)

        return self._forward_cpu(spike_train, dt)

    def _forward_cpu(
        self,
        spike_train: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Standard CPU forward pass using per-neuron loop."""
        n_steps = spike_train.shape[0]
        output_spikes = np.zeros((n_steps, self.output_dim), dtype=np.float64)
        voltages = np.zeros((n_steps, self.output_dim), dtype=np.float64)

        for t in range(n_steps):
            current = self.W.dot(spike_train[t])
            if self.bias is not None:
                current = current + self.bias

            for j, neuron in enumerate(self.neurons):
                v, spike = neuron.step(float(current[j]), dt)
                output_spikes[t, j] = float(spike)
                voltages[t, j] = v

        return output_spikes, voltages

    def _forward_gpu(
        self,
        spike_train: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        GPU-accelerated forward pass using vectorized PyTorch computation.

        Instead of iterating over individual neurons, computes all
        neuron updates simultaneously using matrix operations on GPU.
        Supports both LIF and ALIF neuron types with proper state
        maintenance between forward calls.
        """
        n0 = self.neurons[0]
        is_alif = hasattr(n0, "w") and hasattr(n0, "beta") and hasattr(n0, "tau_w")
        has_tau_s = hasattr(n0, "tau_s") and n0.tau_s is not None

        self._sync_from_neurons()

        W_t = torch.from_numpy(self.W).float().to(self.device)
        spike_t = torch.from_numpy(spike_train).float().to(self.device)
        bias_t = (
            torch.from_numpy(self.bias).float().to(self.device)
            if self.bias is not None
            else None
        )

        v = torch.from_numpy(self._v).float().to(self.device)
        v_rest = n0.v_rest
        v_threshold = n0.v_threshold
        v_reset = n0.v_reset
        resistance = n0.resistance
        tau_m = n0.tau_m
        refractory_period = n0.refractory_period
        reset_mechanism = n0.reset_mechanism

        decay = torch.exp(torch.tensor(-dt / tau_m, device=self.device))

        refractory = torch.from_numpy(self._refractory).float().to(self.device)

        i_syn = torch.from_numpy(self._total_current).float().to(self.device)

        tau_s_decay = None
        if has_tau_s:
            tau_s_decay = torch.exp(torch.tensor(-dt / n0.tau_s, device=self.device))

        w_adapt = None
        w_decay = None
        beta = None
        if is_alif:
            w_adapt = torch.from_numpy(self._w).float().to(self.device)
            w_decay_val = np.exp(-dt / n0.tau_w)
            w_decay = torch.tensor(w_decay_val, dtype=torch.float32, device=self.device)
            beta = n0.beta

        n_steps = spike_train.shape[0]
        output_spikes = torch.zeros((n_steps, self.output_dim), dtype=torch.float32, device=self.device)
        voltages = torch.zeros((n_steps, self.output_dim), dtype=torch.float32, device=self.device)

        for t in range(n_steps):
            current = torch.matmul(spike_t[t], W_t.T)
            if bias_t is not None:
                current = current + bias_t

            in_refractory = refractory > 0

            if has_tau_s:
                i_syn = current + (i_syn - current) * tau_s_decay
                effective_current = torch.where(in_refractory, torch.zeros_like(i_syn), i_syn)
            else:
                effective_current = torch.where(in_refractory, torch.zeros_like(current), current)

            active = ~in_refractory

            v_inf = v_rest + resistance * effective_current
            v = torch.where(active, v_inf + (v - v_inf) * decay, torch.tensor(v_reset, dtype=torch.float32, device=self.device))

            if is_alif:
                w_adapt = w_adapt * w_decay
                effective_threshold = v_threshold + beta * w_adapt
            else:
                effective_threshold = v_threshold

            spike = (v >= effective_threshold) & active

            if reset_mechanism == ResetMechanism.HARD:
                v = torch.where(spike, torch.tensor(v_reset, dtype=torch.float32, device=self.device), v)
            else:
                v = torch.where(spike, v - effective_threshold, v)

            refractory = torch.where(spike, torch.tensor(refractory_period, dtype=torch.float32, device=self.device), torch.clamp(refractory - dt, min=0.0))

            i_syn = torch.where(spike, torch.zeros_like(i_syn), i_syn)

            if is_alif:
                w_adapt = torch.where(spike, w_adapt + 1.0, w_adapt)

            output_spikes[t] = spike.float()
            voltages[t] = v

        result_spikes = output_spikes.cpu().numpy().astype(np.float64)
        result_voltages = voltages.cpu().numpy().astype(np.float64)

        self._v = result_voltages[-1]
        self._refractory = refractory.cpu().numpy().astype(np.float64)
        self._total_current = i_syn.cpu().numpy().astype(np.float64)
        if is_alif:
            self._w = w_adapt.cpu().numpy().astype(np.float64)

        self._sync_to_neurons()

        for j in range(self.output_dim):
            spike_count_this_step = int(np.sum(result_spikes[:, j] > 0.5))
            self.neurons[j].spike_count += spike_count_this_step

        return result_spikes, result_voltages

    def forward_batch(
        self,
        spike_trains: np.ndarray,
        dt: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a batch of spike trains through the layer.

        Each sample in the batch is processed independently with
        fresh neuron states (no carry-over between batch items).

        Parameters
        ----------
        spike_trains : np.ndarray
            Batch of spike trains, shape (batch_size, n_steps, input_dim).
        dt : float
            Time step duration in milliseconds.

        Returns
        -------
        output_spikes : np.ndarray
            Batch of output spike trains, shape (batch_size, n_steps, output_dim).
        hidden_states : np.ndarray
            Batch of hidden states (mean spike rates), shape (batch_size, output_dim).
        """
        spike_trains = np.asarray(spike_trains, dtype=np.float64)
        if spike_trains.ndim == 2:
            spike_trains = spike_trains[np.newaxis, :, :]
        if spike_trains.ndim != 3:
            raise ValueError(
                f"spike_trains must be 2D or 3D, got {spike_trains.ndim}D"
            )

        batch_size = spike_trains.shape[0]
        n_steps = spike_trains.shape[1]

        if self._use_gpu and _TORCH_AVAILABLE and not self.sparse:
            return self._forward_batch_gpu(spike_trains, dt)

        all_output = np.zeros(
            (batch_size, n_steps, self.output_dim), dtype=np.float64
        )
        all_hidden = np.zeros((batch_size, self.output_dim), dtype=np.float64)

        saved_v = self._v.copy()
        saved_refractory = self._refractory.copy()
        saved_total_current = self._total_current.copy()
        saved_w = self._w.copy() if self._w is not None else None
        # Save individual neuron object states
        saved_neuron_states = []
        for j in range(self.output_dim):
            saved_neuron_states.append({
                'v': self.neurons[j].v,
                'refractory_timer': self.neurons[j].refractory_timer,
                'i_syn': self.neurons[j].i_syn,
                'w': getattr(self.neurons[j], 'w', 0.0),
            })

        for b in range(batch_size):
            self._v = np.full(self.output_dim, self.neurons[0].v_rest, dtype=np.float64)
            self._refractory = np.zeros(self.output_dim, dtype=np.float64)
            self._total_current = np.zeros(self.output_dim, dtype=np.float64)
            if self._w is not None:
                self._w = np.zeros(self.output_dim, dtype=np.float64)
            # Reset individual neuron objects
            for j in range(self.output_dim):
                self.neurons[j].v = self.neurons[j].v_rest
                self.neurons[j].refractory_timer = 0.0
                self.neurons[j].i_syn = 0.0
                if hasattr(self.neurons[j], 'w'):
                    self.neurons[j].w = 0.0

            output, _ = self.forward(spike_trains[b], dt)
            all_output[b] = output
            all_hidden[b] = np.mean(output, axis=0)

        self._v = saved_v
        self._refractory = saved_refractory
        self._total_current = saved_total_current
        if saved_w is not None:
            self._w = saved_w
        # Restore individual neuron object states
        for j in range(self.output_dim):
            self.neurons[j].v = saved_neuron_states[j]['v']
            self.neurons[j].refractory_timer = saved_neuron_states[j]['refractory_timer']
            self.neurons[j].i_syn = saved_neuron_states[j]['i_syn']
            if hasattr(self.neurons[j], 'w'):
                self.neurons[j].w = saved_neuron_states[j]['w']

        return all_output, all_hidden

    def _forward_batch_gpu(
        self,
        spike_trains: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """GPU-accelerated batch forward pass."""
        batch_size, n_steps, _ = spike_trains.shape

        n0 = self.neurons[0]
        is_alif = hasattr(n0, "w") and hasattr(n0, "beta")
        v_rest = n0.v_rest
        v_threshold = n0.v_threshold
        v_reset = n0.v_reset
        resistance = n0.resistance
        tau_m = n0.tau_m
        refractory_period = n0.refractory_period
        reset_mechanism = n0.reset_mechanism
        decay_val = np.exp(-dt / tau_m)

        W_t = torch.from_numpy(self.W).float().to(self.device)
        bias_t = (
            torch.from_numpy(self.bias).float().to(self.device)
            if self.bias is not None else None
        )
        spike_t = torch.from_numpy(spike_trains).float().to(self.device)

        v = torch.full((batch_size, self.output_dim), v_rest, dtype=torch.float32, device=self.device)
        refractory = torch.zeros(batch_size, self.output_dim, dtype=torch.float32, device=self.device)
        i_syn = torch.zeros(batch_size, self.output_dim, dtype=torch.float32, device=self.device)
        decay = torch.tensor(decay_val, dtype=torch.float32, device=self.device)

        w_adapt = None
        w_decay = None
        beta = None
        if is_alif:
            w_adapt = torch.zeros(batch_size, self.output_dim, dtype=torch.float32, device=self.device)
            w_decay_val = np.exp(-dt / n0.tau_w)
            w_decay = torch.tensor(w_decay_val, dtype=torch.float32, device=self.device)
            beta = n0.beta

        output_spikes = torch.zeros(
            (batch_size, n_steps, self.output_dim), dtype=torch.float32, device=self.device
        )

        for t in range(n_steps):
            current = torch.matmul(spike_t[:, t, :], W_t.T)
            if bias_t is not None:
                current = current + bias_t

            in_refractory = refractory > 0
            active = ~in_refractory

            effective_current = torch.where(in_refractory, torch.zeros_like(current), current)

            v_inf = v_rest + resistance * effective_current
            v = torch.where(
                active,
                v_inf + (v - v_inf) * decay,
                torch.tensor(v_reset, dtype=torch.float32, device=self.device),
            )

            if is_alif:
                w_adapt = w_adapt * w_decay
                effective_threshold = v_threshold + beta * w_adapt
            else:
                effective_threshold = v_threshold

            spike = (v >= effective_threshold) & active

            if reset_mechanism == ResetMechanism.HARD:
                v = torch.where(spike, torch.tensor(v_reset, dtype=torch.float32, device=self.device), v)
            else:
                v = torch.where(spike, v - effective_threshold, v)

            refractory = torch.where(
                spike,
                torch.tensor(refractory_period, dtype=torch.float32, device=self.device),
                torch.clamp(refractory - dt, min=0.0),
            )

            i_syn = torch.where(spike, torch.zeros_like(i_syn), i_syn)

            if is_alif:
                w_adapt = torch.where(spike, w_adapt + 1.0, w_adapt)

            output_spikes[:, t, :] = spike.float()

        result_spikes = output_spikes.cpu().numpy().astype(np.float64)
        hidden_states = np.mean(result_spikes, axis=1)

        return result_spikes, hidden_states

    def step(
        self,
        input_spikes: np.ndarray,
        dt: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single time step of input spikes.

        Parameters
        ----------
        input_spikes : np.ndarray
            Input spike vector of shape (input_dim,).
        dt : float
            Time step duration in milliseconds.

        Returns
        -------
        output_spikes : np.ndarray
            Output spike vector of shape (output_dim,).
        voltages : np.ndarray
            Membrane potentials of shape (output_dim,).
        """
        input_spikes = np.asarray(input_spikes, dtype=np.float64)
        if input_spikes.shape != (self.input_dim,):
            raise ValueError(
                f"input_spikes shape must be ({self.input_dim},), "
                f"got {input_spikes.shape}"
            )

        current = self.W.dot(input_spikes)
        if self.bias is not None:
            current = current + self.bias

        output_spikes = np.zeros(self.output_dim, dtype=np.float64)
        voltages = np.zeros(self.output_dim, dtype=np.float64)

        for j, neuron in enumerate(self.neurons):
            v, spike = neuron.step(float(current[j]), dt)
            output_spikes[j] = float(spike)
            voltages[j] = v

        return output_spikes, voltages

    def reset(self) -> None:
        """Reset all neurons in the layer to resting state."""
        for neuron in self.neurons:
            neuron.reset()

    def get_spike_rates(self) -> np.ndarray:
        """
        Get the current spike rate of each neuron.

        Returns
        -------
        np.ndarray
            Array of spike counts for each neuron.
        """
        return np.array([n.spike_count for n in self.neurons], dtype=np.float64)

    def state_dict(self) -> Dict[str, Any]:
        """Return the layer state for serialization."""
        state = {
            "bias": self.bias.copy() if self.bias is not None else None,
            "neurons": [n.state_dict() for n in self.neurons],
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "sparse": self.sparse,
        }
        if self.sparse and self._sparse_conn is not None:
            state["sparse_conn"] = self._sparse_conn.state_dict()
        else:
            state["W"] = self.W.copy()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load layer state from a dictionary."""
        is_sparse = state.get("sparse", False)
        if is_sparse and "sparse_conn" in state:
            if self._sparse_conn is not None:
                self._sparse_conn.load_state_dict(state["sparse_conn"])
            else:
                self._sparse_conn = SparseConnectivity(
                    n_pre=state["sparse_conn"]["n_pre"],
                    n_post=state["sparse_conn"]["n_post"],
                    synapses_per_neuron=state["sparse_conn"]["synapses_per_neuron"],
                )
                self._sparse_conn.load_state_dict(state["sparse_conn"])
            self.W = self._sparse_conn.W
            self.sparse = True
        elif "W" in state:
            self.W = state["W"].copy()
            self.sparse = False
            self._sparse_conn = None
        if state["bias"] is not None:
            self.bias = state["bias"].copy()
        else:
            self.bias = None
        if "neurons" in state:
            for neuron, n_state in zip(self.neurons, state["neurons"]):
                neuron.load_state_dict(n_state)

    def __repr__(self) -> str:
        sparse_info = f", sparse=True, synapses_per_neuron={self.synapses_per_neuron}" if self.sparse else ""
        return (
            f"SNNLayer(input_dim={self.input_dim}, "
            f"output_dim={self.output_dim}, "
            f"neuron_type={self.neuron_type.__name__}{sparse_info})"
        )
