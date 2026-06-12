"""
NIEA Brain - Neural Isomorphic Evolutionary Architecture

Complete integration of all seven core modules into a unified
brain model that can perceive, think, learn, and develop.

Architecture:
    1. Spiking Neurons (LIF/ALIF) - Event-driven computing primitives
    2. STDP Plasticity - Local, online, unsupervised learning
    3. Predictive Coding (PCN) - Hierarchical world model
    4. Active Inference - Intrinsic motivation (curiosity)
    5. HDC Memory - Episodic memory and concept storage
    6. World Model - Recurrent prediction for planning/imagination
    7. Metacognition - Self-awareness and uncertainty estimation

The brain operates in a continuous perceive-think-learn cycle:
    1. Perceive: Convert sensory input to internal representation
    2. Think: Evaluate knowledge, compute curiosity, select action
    3. Learn: Update all plastic components based on experience
    4. Develop: Progress through developmental stages

Key differences from the theory document's simplified integration:
- Uses full industrial-grade sub-modules (not simplified versions)
- Proper spike encoding with Poisson rate coding
- Correct STDP with eligibility traces
- Proper PCN inference with convergence detection
- Full active inference with Dirichlet priors
- HDC memory with correct binding/unbinding
- World model with analytical gradients
- Metacognition with proper calibration
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from skyalyticAI.neurons.snn_layer import SNNLayer
from skyalyticAI.neurons.alif import ALIFNeuron
from skyalyticAI.neurons.sparse_connectivity import BrainScaleConfig, SparseConnectivity
from skyalyticAI.plasticity.stdp_layer import STDPLayer
from skyalyticAI.predictive_coding.pcn import PredictiveCodingNetwork
from skyalyticAI.active_inference.agent import ActiveInferenceAgent
from skyalyticAI.memory.hdc import HDCMemory
from skyalyticAI.memory.consolidation import ComplementaryMemorySystem
from skyalyticAI.world_model.world_model import WorldModel
from skyalyticAI.metacognition.metacognition import MetacognitiveModule
from skyalyticAI.perception.visual_encoder import VisualEncoder
from skyalyticAI.perception.audio_encoder import AudioEncoder
from skyalyticAI.perception.multimodal_fusion import MultimodalFusion
from skyalyticAI.consciousness import GlobalWorkspace
from skyalyticAI.evolution import StructuralEvolution
from skyalyticAI.language.language_head import LanguageHead


class DevelopmentStage:
    SENSORIMOTOR = "sensorimotor"
    KINDERGARTEN = "kindergarten"
    PRIMARY = "primary"
    MIDDLE = "middle"
    HIGH = "high"
    UNDERGRADUATE = "undergraduate"
    MASTER = "master"
    PHD = "phd"

    STAGES = [
        SENSORIMOTOR,
        KINDERGARTEN,
        PRIMARY,
        MIDDLE,
        HIGH,
        UNDERGRADUATE,
        MASTER,
        PHD,
    ]

    @staticmethod
    def get_stage(experience_steps: int) -> str:
        if experience_steps < 500:
            return DevelopmentStage.SENSORIMOTOR
        elif experience_steps < 2000:
            return DevelopmentStage.KINDERGARTEN
        elif experience_steps < 5000:
            return DevelopmentStage.PRIMARY
        elif experience_steps < 10000:
            return DevelopmentStage.MIDDLE
        elif experience_steps < 20000:
            return DevelopmentStage.HIGH
        elif experience_steps < 50000:
            return DevelopmentStage.UNDERGRADUATE
        elif experience_steps < 80000:
            return DevelopmentStage.MASTER
        else:
            return DevelopmentStage.PHD


class BrainScalePresets:
    """人脑规模预设方案，用户可按需选择或自定义参数覆盖。"""

    # -- 小规模：个人电脑 / 笔记本（CPU 或 小显存GPU）--
    SMALL = {
        "hidden_dim": 256,
        "hd_dim": 10000,
        "pcn_hidden_dim": 128,
        "world_model_hidden_dim": 256,
        "ai_hidden_dim": 128,
        "n_observations": 256,
        "spike_encoding_steps": 20,
        "synapses_per_neuron": 100,
    }

    # -- 中规模：单卡GPU（RTX 3090 / 4090 / A10 等 24GB显存）--
    MEDIUM = {
        "hidden_dim": 2048,
        "hd_dim": 32000,
        "pcn_hidden_dim": 1024,
        "world_model_hidden_dim": 1024,
        "ai_hidden_dim": 1024,
        "n_observations": 2048,
        "spike_encoding_steps": 50,
        "synapses_per_neuron": 1000,
    }

    # -- 大规模：多卡GPU（A100 80GB x 4~8 卡）--
    LARGE = {
        "hidden_dim": 8192,
        "hd_dim": 128000,
        "pcn_hidden_dim": 4096,
        "world_model_hidden_dim": 4096,
        "ai_hidden_dim": 4096,
        "n_observations": 8192,
        "spike_encoding_steps": 80,
        "synapses_per_neuron": 3000,
    }

    # -- 超大规模：GPU集群（A100/H100 x 32+ 卡，分布式训练）--
    XLARGE = {
        "hidden_dim": 65536,
        "hd_dim": 1000000,
        "pcn_hidden_dim": 32768,
        "world_model_hidden_dim": 32768,
        "ai_hidden_dim": 32768,
        "n_observations": 65536,
        "spike_encoding_steps": 100,
        "synapses_per_neuron": 5000,
    }

    # -- 人脑规模：需要神经形态芯片或超大规模集群 --
    HUMAN = {
        "hidden_dim": 16_000_000_000,      # 大脑皮层160亿神经元
        "hd_dim": 15_000_000,               # 海马体1500万
        "pcn_hidden_dim": 8_000_000_000,    # 感觉皮层80亿
        "world_model_hidden_dim": 2_000_000_000,  # 前额叶20亿
        "ai_hidden_dim": 1_000_000_000,     # 基底节+前额叶10亿
        "n_observations": 16_000_000_000,
        "spike_encoding_steps": 100,
        "synapses_per_neuron": 7000,        # 人脑每神经元平均突触数
    }

    @classmethod
    def get(cls, preset: str) -> dict:
        presets = {
            "small": cls.SMALL,
            "medium": cls.MEDIUM,
            "large": cls.LARGE,
            "xlarge": cls.XLARGE,
            "human": cls.HUMAN,
        }
        key = preset.lower()
        if key not in presets:
            raise ValueError(
                f"Unknown preset '{preset}', available: {list(presets.keys())}"
            )
        return presets[key].copy()


class NIEABrain:
    """
    NIEA Brain - Complete brain model integrating all seven modules.

    Parameters
    ----------
    input_dim : int
        Dimension of sensory input.
    hidden_dim : int
        Dimension of hidden representations in SNN and PCN.
    action_dim : int
        Number of available discrete actions.
    n_observations : int
        Number of discrete observations for active inference.
    hd_dim : int
        Dimension of HDC memory vectors.
    pcn_hidden_dim : int
        Hidden dimension in the PCN.
    world_model_hidden_dim : int
        Hidden dimension in the world model.
    development_stages : bool
        Whether to simulate developmental stage transitions.
    spike_encoding_steps : int
        Number of time steps for spike encoding of each input.
    surprise_threshold : float
        Threshold for deciding whether to store an experience
        in long-term memory (based on prediction error magnitude).
    consolidation_interval : int
        Number of experiences between memory consolidation steps.
    consolidation_batch_size : int
        Number of experiences to replay during consolidation.
    brain_scale : str or bool
        Preset scale configuration. Options:
        - "small"  : PC/laptop (CPU or small GPU)
        - "medium" : Single GPU (RTX 3090/4090, 24GB VRAM)
        - "large"  : Multi-GPU (A100 80GB x 4~8)
        - "xlarge" : GPU cluster (A100/H100 x 32+)
        - "human"  : Full human brain scale (requires neuromorphic hardware)
        - False    : No preset, use explicit parameters
        - True     : Equivalent to "human" (backward compatible)
        User-provided explicit parameters always override preset values.
    sparse : bool
        Whether to use sparse connectivity (recommended for large scale).
    synapses_per_neuron : int
        Number of synapses per neuron (only used when sparse=True).
    """

    # 可被 brain_scale 预设覆盖的参数名
    _PRESET_OVERRIDABLE = {
        "hidden_dim", "hd_dim", "pcn_hidden_dim",
        "world_model_hidden_dim", "ai_hidden_dim",
        "n_observations", "spike_encoding_steps", "synapses_per_neuron",
    }

    _SENTINEL = object()  # 用于检测用户是否显式传参

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 256,
        action_dim: int = 4,
        n_observations: int = 10,
        hd_dim: int = 10000,
        pcn_hidden_dim: int = 128,
        world_model_hidden_dim: int = 256,
        world_model_n_layers: int = 4,
        n_snn_layers: int = 3,
        ai_hidden_dim: int = 128,
        development_stages: bool = True,
        spike_encoding_steps: int = 20,
        surprise_threshold: float = 0.5,
        consolidation_interval: int = 100,
        consolidation_batch_size: int = 20,
        visual_encoder: Optional[VisualEncoder] = None,
        audio_encoder: Optional[AudioEncoder] = None,
        multimodal_fusion: Optional[MultimodalFusion] = None,
        device: Any = None,
        language_vocab_size: Optional[int] = None,
        sparse: bool = False,
        synapses_per_neuron: int = 7000,
        brain_scale: Union[bool, str] = False,
        # 以下参数用于覆盖预设值，仅传入时生效
        override_hidden_dim: int = _SENTINEL,
        override_hd_dim: int = _SENTINEL,
        override_pcn_hidden_dim: int = _SENTINEL,
        override_world_model_hidden_dim: int = _SENTINEL,
        override_ai_hidden_dim: int = _SENTINEL,
        override_n_observations: int = _SENTINEL,
        override_spike_encoding_steps: int = _SENTINEL,
        override_synapses_per_neuron: int = _SENTINEL,
    ) -> None:
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")

        # -- brain_scale 预设机制 --
        # 优先级：override_xxx > brain_scale预设 > 普通参数默认值
        if brain_scale is True:
            brain_scale = "human"
        if isinstance(brain_scale, str):
            preset = BrainScalePresets.get(brain_scale)
            # 用预设值覆盖默认值
            hidden_dim = preset["hidden_dim"]
            hd_dim = preset["hd_dim"]
            pcn_hidden_dim = preset["pcn_hidden_dim"]
            world_model_hidden_dim = preset["world_model_hidden_dim"]
            ai_hidden_dim = preset["ai_hidden_dim"]
            n_observations = preset["n_observations"]
            spike_encoding_steps = preset["spike_encoding_steps"]
            synapses_per_neuron = preset["synapses_per_neuron"]
            self.brain_scale = brain_scale
            if brain_scale in ("large", "xlarge", "human"):
                sparse = True
        else:
            self.brain_scale = False

        # override 参数优先级最高，覆盖预设值
        if override_hidden_dim is not self._SENTINEL:
            hidden_dim = override_hidden_dim
        if override_hd_dim is not self._SENTINEL:
            hd_dim = override_hd_dim
        if override_pcn_hidden_dim is not self._SENTINEL:
            pcn_hidden_dim = override_pcn_hidden_dim
        if override_world_model_hidden_dim is not self._SENTINEL:
            world_model_hidden_dim = override_world_model_hidden_dim
        if override_ai_hidden_dim is not self._SENTINEL:
            ai_hidden_dim = override_ai_hidden_dim
        if override_n_observations is not self._SENTINEL:
            n_observations = override_n_observations
        if override_spike_encoding_steps is not self._SENTINEL:
            spike_encoding_steps = override_spike_encoding_steps
        if override_synapses_per_neuron is not self._SENTINEL:
            synapses_per_neuron = override_synapses_per_neuron

        self.sparse = sparse
        self.synapses_per_neuron = synapses_per_neuron

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.n_observations = n_observations
        self.development_stages = development_stages
        self.spike_encoding_steps = spike_encoding_steps
        self.surprise_threshold = surprise_threshold
        if surprise_threshold < 0:
            self.surprise_threshold = max(0.01, 0.5 / np.sqrt(hidden_dim))
        self.consolidation_interval = consolidation_interval
        self.consolidation_batch_size = consolidation_batch_size
        self.n_snn_layers = n_snn_layers

        self.age: int = 0
        self.stage: str = DevelopmentStage.SENSORIMOTOR
        # 人类升学阶段（与 HumanGrowthWorld.school_stage 同步）
        self.school_stage: str = "sensorimotor"

        if language_vocab_size is not None and language_vocab_size > 0:
            self.language_head: Optional[LanguageHead] = LanguageHead(
                hidden_dim=hidden_dim,
                vocab_size=language_vocab_size,
            )
            # 多模态任务头：ASR/OCR（与语言头同构，独立参数）
            self.asr_head: Optional[LanguageHead] = LanguageHead(
                hidden_dim=hidden_dim,
                vocab_size=language_vocab_size,
            )
            self.ocr_head: Optional[LanguageHead] = LanguageHead(
                hidden_dim=hidden_dim,
                vocab_size=language_vocab_size,
            )
        else:
            self.language_head = None
            self.asr_head = None
            self.ocr_head = None

        snn_dims = [input_dim]
        if n_snn_layers == 1:
            snn_dims.append(hidden_dim)
        elif n_snn_layers == 2:
            snn_dims.append(hidden_dim * 2)
            snn_dims.append(hidden_dim)
        else:
            for i in range(n_snn_layers - 1):
                scale = 2.0 - (i / max(n_snn_layers - 1, 1))
                snn_dims.append(int(hidden_dim * scale))
            snn_dims.append(hidden_dim)

        neuron_params = {
            "tau_m": 10.0,
            "v_threshold": 0.5,
            "v_reset": 0.0,
            "v_rest": 0.0,
            "resistance": 1.0,
            "refractory_period": 1.0,
            "tau_w": 50.0,
            "beta": 0.02,
        }

        self.snn_layers: List[SNNLayer] = []
        for i in range(len(snn_dims) - 1):
            layer_weight_scale = 2.0 * (1.0 + i * 0.5)
            layer_neuron_params = dict(neuron_params)
            layer_neuron_params["v_threshold"] = 0.5 / (1.0 + i * 0.3)
            layer_neuron_params["resistance"] = 1.0 * (1.0 + i * 0.5)
            self.snn_layers.append(SNNLayer(
                input_dim=snn_dims[i],
                output_dim=snn_dims[i + 1],
                neuron_type=ALIFNeuron,
                neuron_params=layer_neuron_params,
                weight_init="kaiming_normal",
                weight_scale=layer_weight_scale,
                device=device,
                sparse_connectivity=sparse,
                synapses_per_neuron=synapses_per_neuron,
            ))

        self.snn_layer = self.snn_layers[-1]

        self.stdp_layer = STDPLayer(
            pre_dim=hidden_dim,
            post_dim=hidden_dim,
            A_plus=0.01,
            A_minus=0.012,
            tau_plus=20.0,
            tau_minus=20.0,
            variant="additive",
            weight_init="glorot",
            sparse_connectivity=sparse,
            synapses_per_neuron=synapses_per_neuron,
        )

        self.pcn = PredictiveCodingNetwork(
            layer_sizes=[hidden_dim, pcn_hidden_dim, pcn_hidden_dim // 2, action_dim],
            learning_rate=0.01,
            state_learning_rate=0.05,
            n_inference_steps=20,
            inference_tolerance=1e-5,
            sparse=sparse,
            synapses_per_neuron=max(5000, synapses_per_neuron // 2),
        )

        self.active_inference = ActiveInferenceAgent(
            state_dim=hidden_dim,
            obs_dim=hidden_dim,
            n_actions=action_dim,
            hidden_dim=ai_hidden_dim,
            n_hidden_layers=2,
            planning_horizon=1,
            temperature=0.5,
            learning_rate=0.001,
            device=device,
        )

        self.hd_memory = HDCMemory(
            dim=hd_dim,
            vector_type="bipolar",
            seed=42,
        )

        _is_large_scale = isinstance(brain_scale, str) and brain_scale in ("large", "xlarge", "human")
        hippocampal_capacity = 15_000_000 if _is_large_scale else 2000
        cortical_capacity = 16_000_000_000 if _is_large_scale else 20000
        self.complementary_memory = ComplementaryMemorySystem(
            dim=hidden_dim,
            hippocampal_capacity=hippocampal_capacity,
            cortical_capacity=cortical_capacity,
        )

        self.world_model = WorldModel(
            obs_dim=input_dim,
            state_dim=hidden_dim,
            action_dim=action_dim,
            hidden_dim=world_model_hidden_dim,
            n_layers=world_model_n_layers,
            learning_rate=0.001,
            device=device,
        )

        self.metacognition = MetacognitiveModule(
            input_dim=hidden_dim + 3,
            hidden_dim=64,
            memory_size=2000,
        )

        self.experience_buffer: deque = deque(maxlen=100_000_000 if _is_large_scale else 10000)
        self._consolidation_counter: int = 0
        self._last_broadcast = None
        self._input_running_mean = np.zeros(input_dim)
        self._input_running_M2 = np.zeros(input_dim)
        self._input_running_count = 0

        self.state_to_obs_W = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + input_dim))
        self.state_to_obs_b = np.zeros(input_dim, dtype=np.float64)

        self.visual_encoder = visual_encoder
        self.audio_encoder = audio_encoder
        self.multimodal_fusion = multimodal_fusion

        self._perception_projection_W: Optional[np.ndarray] = None
        self._perception_projection_input_dim: Optional[int] = None

        self._saved_pcn_state: Optional[Dict[str, Any]] = None

        self.global_workspace = GlobalWorkspace(
            workspace_dim=hidden_dim,
            n_modules=7,
        )

        self.structural_evolution = StructuralEvolution(
            evolution_interval=500,
        )

        self.stats: Dict[str, List[float]] = {
            "prediction_errors": [],
            "curiosity_levels": [],
            "knowledge_growth": [],
            "rewards": [],
            "confidences": [],
        }

    def _encode_to_spikes(self, sensory_input: np.ndarray) -> np.ndarray:
        """
        Convert continuous sensory input to spike train using
        Poisson rate coding.

        Each input dimension is converted to a firing rate, and
        spikes are generated probabilistically at each time step.

        Parameters
        ----------
        sensory_input : np.ndarray
            Sensory input vector, shape (input_dim,). Values should
            be in [0, 1] for proper rate coding.

        Returns
        -------
        np.ndarray
            Spike train of shape (spike_encoding_steps, input_dim).
        """
        sensory_input = np.asarray(sensory_input, dtype=np.float64)

        # Use running mean and std for normalization (Welford's online algorithm)
        self._input_running_count += 1
        alpha = 1.0 / self._input_running_count
        diff_old = sensory_input - self._input_running_mean
        self._input_running_mean += alpha * diff_old
        diff_new = sensory_input - self._input_running_mean
        self._input_running_M2 += diff_old * diff_new
        variance = self._input_running_M2 / self._input_running_count
        self._input_running_std = np.sqrt(variance + 1e-8)

        normalized = (sensory_input - self._input_running_mean) / (self._input_running_std + 1e-8)
        normalized = np.clip(normalized, -3.0, 3.0) / 3.0  # Map to roughly [0, 1] range
        normalized = (normalized + 1.0) / 2.0  # Shift to [0, 1]

        rates = np.clip(normalized, 0.0, 1.0) * 100.0
        probs = np.clip(rates / 1000.0 * (1000.0 / self.spike_encoding_steps), 0.0, 1.0)

        spike_train = np.zeros(
            (self.spike_encoding_steps, self.input_dim), dtype=np.float64
        )
        for t in range(self.spike_encoding_steps):
            spike_train[t] = (np.random.random(self.input_dim) < probs).astype(
                np.float64
            )

        return spike_train

    def perceive(
        self, sensory_input: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perceive: Convert sensory input to internal representation.

        Pipeline:
        1. Spike encoding (Poisson rate coding)
        2. SNN processing (feature extraction)
        3. PCN prediction (predict next state)

        Parameters
        ----------
        sensory_input : np.ndarray
            Raw sensory input, shape (input_dim,).

        Returns
        -------
        hidden_state : np.ndarray
            Internal representation, shape (hidden_dim,).
        prediction : np.ndarray
            PCN prediction for the current state.
        prediction_error : np.ndarray
            Difference between actual and predicted state.
        """
        sensory_input = np.asarray(sensory_input, dtype=np.float64)

        spike_train = self._encode_to_spikes(sensory_input)

        current_spikes = spike_train
        for snn_layer in self.snn_layers:
            output_spikes, _ = snn_layer.forward(current_spikes)
            current_spikes = output_spikes

        hidden_state = np.mean(current_spikes, axis=0)

        recurrent_current = self.stdp_layer.forward(hidden_state)
        hidden_state = hidden_state + 0.1 * np.tanh(recurrent_current)

        if hidden_state.shape[0] != self.hidden_dim:
            padded = np.zeros(self.hidden_dim, dtype=np.float64)
            n = min(hidden_state.shape[0], self.hidden_dim)
            padded[:n] = hidden_state[:n]
            hidden_state = padded

        inference_result = self.pcn.infer(hidden_state)

        self._saved_pcn_state = {
            "prediction_errors": [
                e.copy() if e is not None else None
                for e in self.pcn.prediction_errors
            ],
            "input_state": self.pcn.input_state.copy(),
            "layer_states": [layer.x.copy() for layer in self.pcn.layers],
        }

        if self.pcn.layer_sizes[0] == self.hidden_dim:
            prediction = self.pcn.predict_next()
        else:
            prediction = self.pcn.layers[0].predict()

        prediction_error = hidden_state - prediction

        return hidden_state, prediction, prediction_error

    def perceive_multimodal(
        self,
        visual: Optional[np.ndarray] = None,
        audio: Optional[np.ndarray] = None,
        raw_observation: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perceive multimodal input through the perception pipeline.

        Pipeline:
        1. Encode each modality (visual, audio) into feature vectors
        2. Fuse modalities using attention-weighted combination
        3. Feed fused representation through SNN + PCN

        Parameters
        ----------
        visual : np.ndarray or None
            Visual input (image array).
        audio : np.ndarray or None
            Audio input (waveform array).
        raw_observation : np.ndarray or None
            Raw observation vector (bypasses perception pipeline).

        Returns
        -------
        hidden_state : np.ndarray
        prediction : np.ndarray
        prediction_error : np.ndarray
        """
        features: Dict[str, np.ndarray] = {}

        if visual is not None and self.visual_encoder is not None:
            features["visual"] = self.visual_encoder.encode(visual)

        if audio is not None and self.audio_encoder is not None:
            features["audio"] = self.audio_encoder.encode(audio)

        if features and self.multimodal_fusion is not None:
            fused = self.multimodal_fusion.fuse(features)
            if raw_observation is not None:
                raw = np.asarray(raw_observation, dtype=np.float64)
                # 融合多模态特征与原始观测，各占一半维度
                half = self.input_dim // 2
                sensory_input = np.zeros(self.input_dim, dtype=np.float64)
                sensory_input[:min(half, fused.shape[0])] = fused[:min(half, fused.shape[0])]
                sensory_input[half:half + min(self.input_dim - half, raw.shape[0])] = raw[:min(self.input_dim - half, raw.shape[0])]
            else:
                sensory_input = fused
        elif raw_observation is not None:
            sensory_input = np.asarray(raw_observation, dtype=np.float64)
        elif features:
            modality_values = list(features.values())
            if len(modality_values) == 1:
                sensory_input = modality_values[0]
            else:
                min_dim = min(v.shape[0] for v in modality_values)
                stacked = np.stack([v[:min_dim] for v in modality_values])
                sensory_input = np.mean(stacked, axis=0)
        else:
            raise ValueError(
                "No input provided. Supply visual, audio, or raw_observation."
            )

        if sensory_input.shape[0] != self.input_dim:
            if sensory_input.shape[0] < self.input_dim:
                padded = np.zeros(self.input_dim, dtype=np.float64)
                padded[:sensory_input.shape[0]] = sensory_input
                sensory_input = padded
            else:
                if (
                    self._perception_projection_W is None
                    or self._perception_projection_input_dim != sensory_input.shape[0]
                ):
                    self._perception_projection_input_dim = sensory_input.shape[0]
                    self._perception_projection_W = (
                        np.random.randn(self.input_dim, sensory_input.shape[0])
                        * np.sqrt(2.0 / (sensory_input.shape[0] + self.input_dim))
                    )
                sensory_input = self._perception_projection_W @ sensory_input

        return self.perceive(sensory_input)

    def speak(self, hidden_state: np.ndarray) -> int:
        """语言输出：由内部状态生成下一个字符索引。"""
        if self.language_head is None:
            raise RuntimeError("未启用语言头，请创建大脑时传入 language_vocab_size")
        return self.language_head.sample(hidden_state)

    def learn_speech(
        self, hidden_state: np.ndarray, target_index: int, reward: float
    ) -> Dict[str, float]:
        """根据对错反馈学习说话。"""
        if self.language_head is None:
            return {}
        sign = 1.0 if reward > 0 else -1.0
        return self.language_head.learn(hidden_state, target_index, sign)

    def asr_decode(self, hidden_state: np.ndarray) -> int:
        """ASR 头：从隐藏状态预测下一字符索引。"""
        if self.asr_head is None:
            return self.speak(hidden_state)
        return self.asr_head.sample(hidden_state)

    def ocr_decode(self, hidden_state: np.ndarray) -> int:
        """OCR 头：从隐藏状态预测下一字符索引。"""
        if self.ocr_head is None:
            return self.speak(hidden_state)
        return self.ocr_head.sample(hidden_state)

    def learn_asr(self, hidden_state: np.ndarray, target_index: int, reward: float) -> Dict[str, float]:
        """ASR 监督学习（字符级）。"""
        if self.asr_head is None:
            return {}
        sign = 1.0 if reward > 0 else -1.0
        out = self.asr_head.learn(hidden_state, target_index, sign)
        return {"asr_loss": out.get("speech_loss", 0.0), "asr_target_prob": out.get("target_prob", 0.0)}

    def learn_ocr(self, hidden_state: np.ndarray, target_index: int, reward: float) -> Dict[str, float]:
        """OCR 监督学习（字符级）。"""
        if self.ocr_head is None:
            return {}
        sign = 1.0 if reward > 0 else -1.0
        out = self.ocr_head.learn(hidden_state, target_index, sign)
        return {"ocr_loss": out.get("speech_loss", 0.0), "ocr_target_prob": out.get("target_prob", 0.0)}

    def set_school_stage(self, stage: str) -> None:
        """与类人世界的升学阶段对齐。"""
        self.school_stage = stage
        if stage in DevelopmentStage.STAGES:
            self.stage = stage

    def think(
        self,
        hidden_state: np.ndarray,
        external_reward: float = 0.0,
        prefer_speech: bool = False,
    ) -> Dict[str, Any]:
        """
        Think: Internal cognitive process.

        Pipeline:
        1. Metacognitive evaluation (how certain am I?)
        2. Curiosity computation (what should I explore?)
        3. Action selection (active inference)
        4. Memory retrieval (relevant past experiences)

        Parameters
        ----------
        hidden_state : np.ndarray
            Current internal representation.
        external_reward : float
            External reward signal (if any).

        Returns
        -------
        dict
            Dictionary containing action, confidence, curiosity,
            and memory context.
        """
        meta_features = np.array([external_reward, self.pcn.get_prediction_uncertainty(), float(self.age) / 10000.0])
        # 拷贝 hidden_state 避免原地修改污染调用者
        hidden_state = hidden_state.copy()
        meta_input = np.concatenate([hidden_state, meta_features])
        if meta_input.shape[0] < self.metacognition.input_dim:
            padded = np.zeros(self.metacognition.input_dim, dtype=np.float64)
            padded[:meta_input.shape[0]] = meta_input
            meta_input = padded
        elif meta_input.shape[0] > self.metacognition.input_dim:
            state_part = hidden_state[:self.metacognition.input_dim - 3]
            meta_input = np.concatenate([state_part, meta_features])

        meta_result = self.metacognition.forward(meta_input)
        confidence = meta_result["confidence"]

        curiosity = self.pcn.get_prediction_uncertainty()

        self.active_inference.perceive(hidden_state)
        action, qualities, action_probs = self.active_inference.select_action()

        if prefer_speech and self.language_head is not None:
            action = self.speak(hidden_state)

        memory_context = self._query_memory(hidden_state)

        imagined_outcomes = self._imagine_action_outcomes(hidden_state)

        preferred_action = self._select_action_from_imagination(
            imagined_outcomes, confidence, curiosity
        )
        if preferred_action is not None:
            action = preferred_action

        self.global_workspace.submit_bid(0, float(np.mean(np.abs(hidden_state))), hidden_state)
        pcn_err = self.pcn.prediction_errors[-1] if self.pcn.prediction_errors and self.pcn.prediction_errors[-1] is not None else np.zeros(self.pcn.layer_sizes[self.pcn.n_layers - 1])
        self.global_workspace.submit_bid(1, float(np.mean(np.abs(pcn_err))), pcn_err[:self.global_workspace.workspace_dim] if pcn_err.shape[0] >= self.global_workspace.workspace_dim else np.pad(pcn_err, (0, max(0, self.global_workspace.workspace_dim - pcn_err.shape[0]))))
        self.global_workspace.submit_bid(2, float(curiosity), np.full(self.global_workspace.workspace_dim, curiosity))
        self.global_workspace.submit_bid(3, float(confidence), np.full(self.global_workspace.workspace_dim, confidence))
        self.global_workspace.submit_bid(4, float(len(memory_context)) / max(1, self.hd_memory.dim) if memory_context else 0.0, np.zeros(self.global_workspace.workspace_dim))
        self.global_workspace.submit_bid(5, float(external_reward), np.full(self.global_workspace.workspace_dim, external_reward))
        self.global_workspace.submit_bid(6, float(np.mean(np.abs(hidden_state))), hidden_state)

        gw_result = self.global_workspace.compete()

        broadcast = self.global_workspace.get_broadcast()
        consciousness_level = self.global_workspace.get_consciousness_level()

        if gw_result["is_ignited"] and broadcast is not None:
            broadcast_norm = np.linalg.norm(broadcast)
            if broadcast_norm > 1e-10:
                broadcast_signal = 0.1 * broadcast / broadcast_norm
                if broadcast_signal.shape[0] == hidden_state.shape[0]:
                    hidden_state += broadcast_signal
                    # 同步更新内部状态，使广播信号影响后续 perceive/learn
                    self._last_broadcast = broadcast_signal.copy()
        else:
            self._last_broadcast = None

        return {
            "action": action,
            "confidence": confidence,
            "curiosity": curiosity,
            "memory_context": memory_context,
            "meta_result": meta_result,
            "imagined_outcomes": imagined_outcomes,
            "consciousness_level": consciousness_level,
            "is_ignited": gw_result["is_ignited"],
            "gw_winner": gw_result["winner"],
        }

    def learn(
        self,
        hidden_state: np.ndarray,
        action: int,
        next_hidden_state: np.ndarray,
        reward: float,
        prediction_error: np.ndarray,
        env_reward: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Learn: Update all plastic components.

        Pipeline:
        1. STDP unsupervised learning (based on spike timing)
        2. PCN prediction learning (minimize prediction error)
        3. World model learning (minimize reconstruction error)
        4. HDC memory consolidation (store important experiences)
        5. Metacognitive calibration (improve self-assessment)

        Parameters
        ----------
        hidden_state : np.ndarray
            State before action.
        action : int
            Action taken.
        next_hidden_state : np.ndarray
            State after action.
        reward : float
            Reward received.
        prediction_error : np.ndarray
            Prediction error from the perceive step.

        Returns
        -------
        dict
            Dictionary containing learning metrics.
        """
        pre_spikes = (hidden_state > np.mean(hidden_state)).astype(np.float64)
        post_spikes = (next_hidden_state > np.mean(next_hidden_state)).astype(np.float64)
        stdp_update = self.stdp_layer.update(pre_spikes, post_spikes)

        if self._saved_pcn_state is not None:
            current_errors = self.pcn.prediction_errors
            current_input = self.pcn.input_state
            current_layer_states = [layer.x.copy() for layer in self.pcn.layers]

            self.pcn.prediction_errors = self._saved_pcn_state["prediction_errors"]
            self.pcn.input_state = self._saved_pcn_state["input_state"]
            for l_idx, layer in enumerate(self.pcn.layers):
                layer.x = self._saved_pcn_state["layer_states"][l_idx].copy()

            pcn_result = self.pcn.learn()

            self.pcn.prediction_errors = current_errors
            self.pcn.input_state = current_input
            for l_idx, layer in enumerate(self.pcn.layers):
                layer.x = current_layer_states[l_idx]

            self._saved_pcn_state = None
        else:
            pcn_result = self.pcn.learn()

        action_vec = np.zeros(self.action_dim, dtype=np.float64)
        action_vec[action % self.action_dim] = 1.0
        obs = self.state_to_obs_W @ hidden_state + self.state_to_obs_b
        next_obs = self.state_to_obs_W @ next_hidden_state + self.state_to_obs_b

        wm_reward = env_reward if env_reward is not None else reward
        wm_loss = self.world_model.train_step(obs, action_vec, next_obs, reward=wm_reward)

        self.active_inference.learn_transition(hidden_state, action, next_hidden_state)
        self.active_inference.learn_observation(hidden_state, hidden_state)

        surprise = float(np.linalg.norm(prediction_error))
        if surprise > self.surprise_threshold or reward > 0:
            self._store_to_memory(hidden_state, action, next_hidden_state, reward, surprise)

        self.complementary_memory.store(hidden_state, next_hidden_state)

        self.structural_evolution.record_performance(1.0 - surprise)
        self.structural_evolution.record_neuron_activity("snn", hidden_state)

        if self.structural_evolution.should_evolve():
            if self.structural_evolution.detect_plateau():
                self._evolve_structure()

        meta_features = np.array([reward, self.pcn.get_prediction_uncertainty(), float(self.age) / 10000.0])
        meta_input = np.concatenate([hidden_state, meta_features])
        if meta_input.shape[0] < self.metacognition.input_dim:
            padded = np.zeros(self.metacognition.input_dim, dtype=np.float64)
            padded[:meta_input.shape[0]] = meta_input
            meta_input = padded
        elif meta_input.shape[0] > self.metacognition.input_dim:
            state_part = hidden_state[:self.metacognition.input_dim - 3]
            meta_input = np.concatenate([state_part, meta_features])

        confidence = self.metacognition.forward(meta_input)["confidence"]
        actual_outcome = 1.0 if reward > 0 else 0.0
        self.metacognition.update_meta_knowledge(
            meta_input, confidence, actual_outcome, reward
        )

        self.experience_buffer.append({
            "state": hidden_state.copy(),
            "action": action,
            "next_state": next_hidden_state.copy(),
            "reward": reward,
        })

        self._consolidation_counter += 1
        if self._consolidation_counter >= self.consolidation_interval:
            self._consolidate_memory()
            self._consolidation_counter = 0

        self.stats["prediction_errors"].append(surprise)
        self.stats["curiosity_levels"].append(self.pcn.get_prediction_uncertainty())
        self.stats["knowledge_growth"].append(len(self.hd_memory.item_memory))
        self.stats["rewards"].append(reward)
        self.stats["confidences"].append(confidence)
        # Trim stats to prevent unbounded memory growth
        _MAX_STATS = 10000
        for key in self.stats:
            if len(self.stats[key]) > _MAX_STATS:
                self.stats[key] = self.stats[key][-_MAX_STATS // 2:]

        return {
            "stdp_update": stdp_update,
            "pcn_loss": pcn_result["total_update"],
            "wm_loss": wm_loss["total"],
            "surprise": surprise,
        }

    def develop(self) -> None:
        """
        Advance developmental stage based on experience.
        """
        self.age += 1
        if self.development_stages:
            self.stage = DevelopmentStage.get_stage(self.age)

    def _discretize_observation(self, state: np.ndarray) -> int:
        """Convert continuous state to discrete observation index."""
        idx = int(np.argmax(state[:self.n_observations]))
        return min(idx, self.n_observations - 1)

    def _imagine_action_outcomes(
        self, hidden_state: np.ndarray
    ) -> Dict[int, Dict[str, Any]]:
        obs = self.state_to_obs_W @ hidden_state + self.state_to_obs_b
        outcomes = {}
        for a in range(self.action_dim):
            action_vec = np.zeros(self.action_dim, dtype=np.float64)
            action_vec[a] = 1.0
            z = self.world_model.encode_deterministic(obs)
            z_next = self.world_model.predict_next_state(z, action_vec)
            obs_next = self.world_model.decode(z_next)
            reward_pred = self.world_model.predict_reward(z_next)
            outcomes[a] = {
                "obs": obs_next,
                "reward": reward_pred,
            }
        return outcomes

    def _evolve_structure(self) -> None:
        """
        Apply structural evolution to the brain's architecture.

        Currently only applies synaptic pruning (weight zeroing)
        which does not change tensor dimensions. Neurogenesis
        (adding neurons) requires synchronized dimension updates
        across all downstream modules and is deferred to a
        future architecture version.
        """
        for snn_layer in self.snn_layers:
            W, b, info = self.structural_evolution.prune_weights(
                snn_layer.W, snn_layer.bias
            )
            # 如果原来是稀疏的，将剪枝后的稠密矩阵转回稀疏格式
            if snn_layer._sparse_conn is not None:
                from scipy import sparse as sp
                W = sp.csr_matrix(W)
            snn_layer.W = W
            if b is not None:
                snn_layer.bias = b
            # 同步稀疏连接矩阵
            if snn_layer._sparse_conn is not None:
                snn_layer._sparse_conn.W = W.copy()
                # 同步稀疏元数据
                if hasattr(W, 'nnz'):
                    snn_layer._sparse_conn.n_synapses = W.nnz
                    snn_layer._sparse_conn.sparsity = 1.0 - (W.nnz / max(W.shape[0] * W.shape[1], 1))
        self.structural_evolution._step_counter = 0

    def _select_action_from_imagination(
        self,
        imagined_outcomes: Dict[int, Dict[str, Any]],
        confidence: float,
        curiosity: float,
    ) -> Optional[int]:
        if self.age < 200:
            return None

        if not imagined_outcomes:
            return None

        wm_loss = self.world_model.loss_history
        if len(wm_loss) < 10:
            return None

        recent_loss = np.mean([l["total"] for l in wm_loss[-10:]])
        if recent_loss > 2.0:
            return None

        imagination_weight = min(1.0, confidence) * (1.0 - min(1.0, curiosity))
        if imagination_weight < 0.3:
            return None

        action_scores = np.zeros(self.action_dim, dtype=np.float64)
        for a, outcome in imagined_outcomes.items():
            action_scores[a] = outcome["reward"]

        if np.max(action_scores) - np.min(action_scores) < 1e-10:
            return None

        return int(np.argmax(action_scores))

    def _query_memory(self, state: np.ndarray, top_k: int = 3) -> List[Any]:
        """Query HDC memory for relevant experiences."""
        state_vec = np.sign(state[:self.hd_memory.dim])
        if state_vec.shape[0] < self.hd_memory.dim:
            padded = np.zeros(self.hd_memory.dim, dtype=np.float64)
            padded[:state_vec.shape[0]] = state_vec
            state_vec = padded
        results = self.hd_memory.retrieve(state_vec, top_k=top_k)
        return results

    def _store_to_memory(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        reward: float,
        surprise: float,
    ) -> None:
        """
        Store a complete experience in HDC memory.

        Stores:
        - State vector
        - Action vector
        - Next state vector
        - Reward signal
        - Surprise level

        Associations:
        - state -> action (what action was taken)
        - state -> next_state (what happened)
        - state -> reward (what reward was received)
        - state -> surprise (how surprising was this)
        """
        state_key = "exp_{}".format(self.age)
        action_key = "act_{}".format(action)
        next_state_key = "nexp_{}".format(self.age)
        reward_key = "rew_{}".format(int(reward > 0))
        surprise_key = "sur_{}".format(min(int(surprise * 10), 9))

        if state_key not in self.hd_memory.item_memory:
            state_vec = np.sign(state[:self.hd_memory.dim])
            if state_vec.shape[0] < self.hd_memory.dim:
                padded = np.zeros(self.hd_memory.dim, dtype=np.float64)
                padded[:state_vec.shape[0]] = state_vec
                state_vec = padded
            self.hd_memory.add_concept(state_key, state_vec)

        if action_key not in self.hd_memory.item_memory:
            action_vec = np.zeros(self.hd_memory.dim, dtype=np.float64)
            action_vec[action % self.hd_memory.dim] = 1.0
            self.hd_memory.add_concept(action_key, action_vec)

        if next_state_key not in self.hd_memory.item_memory:
            next_state_vec = np.sign(next_state[:self.hd_memory.dim])
            if next_state_vec.shape[0] < self.hd_memory.dim:
                padded = np.zeros(self.hd_memory.dim, dtype=np.float64)
                padded[:next_state_vec.shape[0]] = next_state_vec
                next_state_vec = padded
            self.hd_memory.add_concept(next_state_key, next_state_vec)

        if reward_key not in self.hd_memory.item_memory:
            reward_vec = np.zeros(self.hd_memory.dim, dtype=np.float64)
            reward_vec[0] = 1.0 if reward > 0 else -1.0
            self.hd_memory.add_concept(reward_key, reward_vec)

        if surprise_key not in self.hd_memory.item_memory:
            surprise_vec = np.zeros(self.hd_memory.dim, dtype=np.float64)
            surprise_vec[int(surprise * 10 * self.hd_memory.dim / 10) % self.hd_memory.dim] = 1.0
            self.hd_memory.add_concept(surprise_key, surprise_vec)

        self.hd_memory.store_association(state_key, action_key)
        self.hd_memory.store_association(state_key, next_state_key)
        self.hd_memory.store_association(state_key, reward_key)
        self.hd_memory.store_association(state_key, surprise_key)

    def _consolidate_memory(self) -> None:
        """
        Memory consolidation (simulating sleep/replay).

        Replays recent experiences to strengthen memories and
        improve the world model, PCN, and complementary memory.
        """
        if len(self.experience_buffer) < 10:
            return

        # Save current PCN state to prevent consolidation from polluting it
        saved_pcn_errors = [e.copy() if e is not None else None for e in self.pcn.prediction_errors]
        saved_pcn_input = self.pcn.input_state.copy()
        saved_pcn_layer_states = [layer.x.copy() for layer in self.pcn.layers]

        batch_size = min(
            self.consolidation_batch_size, len(self.experience_buffer)
        )
        indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in indices]

        for exp in batch:
            state = exp["state"]
            action = exp["action"]
            next_state = exp["next_state"]

            if state.shape[0] != self.state_to_obs_W.shape[1]:
                continue

            action_vec = np.zeros(self.action_dim, dtype=np.float64)
            action_vec[action % self.action_dim] = 1.0

            obs = self.state_to_obs_W @ state + self.state_to_obs_b
            next_obs = self.state_to_obs_W @ next_state + self.state_to_obs_b
            reward = exp.get("reward", 0.0)

            self.world_model.train_step(obs, action_vec, next_obs, reward=reward)

            if state.shape[0] == self.pcn.layer_sizes[0]:
                self.pcn.infer(state)
                self.pcn.learn()

        cms_result = self.complementary_memory.consolidate()

        # Restore PCN state
        self.pcn.prediction_errors = saved_pcn_errors
        self.pcn.input_state = saved_pcn_input
        for l_idx, layer in enumerate(self.pcn.layers):
            layer.x = saved_pcn_layer_states[l_idx]

    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current brain state.

        Returns
        -------
        dict
            Dictionary containing age, stage, knowledge size,
            prediction error, curiosity, and calibration metrics.
        """
        avg_pred_error = (
            np.mean(self.stats["prediction_errors"][-100:])
            if self.stats["prediction_errors"]
            else 0.0
        )
        avg_curiosity = (
            np.mean(self.stats["curiosity_levels"][-100:])
            if self.stats["curiosity_levels"]
            else 0.0
        )
        avg_reward = (
            np.mean(self.stats["rewards"][-100:])
            if self.stats["rewards"]
            else 0.0
        )

        return {
            "age": self.age,
            "stage": self.stage,
            "knowledge_vectors": len(self.hd_memory.item_memory),
            "avg_prediction_error": avg_pred_error,
            "curiosity": avg_curiosity,
            "avg_reward": avg_reward,
            "confidence_calibration": self.metacognition.calibration_score,
            "experience_buffer_size": len(self.experience_buffer),
        }

    def reset(self) -> None:
        """Reset the entire brain to initial state."""
        for snn_layer in self.snn_layers:
            snn_layer.reset()
        self.stdp_layer.reset()
        self.pcn.reset()
        self.active_inference.reset()
        self.hd_memory.reset()
        self.complementary_memory.reset()
        self.world_model.reset()
        self.metacognition.reset()
        self.global_workspace.reset()
        self.structural_evolution._step_counter = 0
        self.structural_evolution._performance_history = []
        self.structural_evolution._evolution_history = []
        self.structural_evolution._neuron_activity = {}
        self.experience_buffer.clear()
        self._consolidation_counter = 0
        self.age = 0
        self.stage = DevelopmentStage.SENSORIMOTOR
        self.school_stage = "sensorimotor"
        self._saved_pcn_state = None
        self._last_broadcast = None
        self._input_running_mean = np.zeros(self.input_dim)
        self._input_running_M2 = np.zeros(self.input_dim)
        self._input_running_count = 0
        self._perception_projection_W = None
        self._perception_projection_input_dim = None
        # Keep state_to_obs_W/b to stay consistent with trained world model
        self.stats = {
            "prediction_errors": [],
            "curiosity_levels": [],
            "knowledge_growth": [],
            "rewards": [],
            "confidences": [],
        }

    def reset_episode(self) -> None:
        """
        Reset per-episode state without clearing learned knowledge.

        Resets SNN neuron states and PCN prediction errors for
        a new episode, but preserves all learned weights, HDC
        memory, world model, and metacognitive calibration.
        """
        for snn_layer in self.snn_layers:
            snn_layer.reset()
        self.pcn.prediction_errors = [None] * self.pcn.n_layers
        self._saved_pcn_state = None
        self.global_workspace.reset()

    def state_dict(self) -> Dict[str, Any]:
        """Return the brain state for serialization."""
        return {
            "snn_layers": [l.state_dict() for l in self.snn_layers],
            "stdp_layer": self.stdp_layer.state_dict(),
            "pcn": self.pcn.state_dict(),
            "active_inference": self.active_inference.state_dict(),
            "hd_memory": self.hd_memory.state_dict(),
            "world_model": self.world_model.state_dict(),
            "metacognition": self.metacognition.state_dict(),
            "state_to_obs_W": self.state_to_obs_W.copy(),
            "state_to_obs_b": self.state_to_obs_b.copy(),
            "age": self.age,
            "stage": self.stage,
            "surprise_threshold": self.surprise_threshold,
            "experience_buffer": list(self.experience_buffer),
            "consolidation_counter": self._consolidation_counter,
            "stats": {k: list(v) for k, v in self.stats.items()},
            "perception_projection_W": self._perception_projection_W.copy() if self._perception_projection_W is not None else None,
            "perception_projection_input_dim": self._perception_projection_input_dim,
            "global_workspace": self.global_workspace.state_dict(),
            "structural_evolution": self.structural_evolution.state_dict(),
            "complementary_memory": self.complementary_memory.state_dict(),
            "school_stage": self.school_stage,
            "language_head": (
                self.language_head.state_dict() if self.language_head else None
            ),
            "asr_head": (
                self.asr_head.state_dict() if self.asr_head else None
            ),
            "ocr_head": (
                self.ocr_head.state_dict() if self.ocr_head else None
            ),
            "sparse": self.sparse,
            "brain_scale": self.brain_scale,
            "_input_running_mean": self._input_running_mean.copy() if hasattr(self, '_input_running_mean') else None,
            "_input_running_M2": self._input_running_M2.copy() if hasattr(self, '_input_running_M2') else None,
            "_input_running_count": self._input_running_count if hasattr(self, '_input_running_count') else None,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load brain state from a dictionary."""
        if "snn_layers" in state:
            for i, snn_state in enumerate(state["snn_layers"]):
                if i < len(self.snn_layers):
                    self.snn_layers[i].load_state_dict(snn_state)
        elif "snn_layer" in state:
            self.snn_layers[-1].load_state_dict(state["snn_layer"])
        self.stdp_layer.load_state_dict(state["stdp_layer"])
        self.pcn.load_state_dict(state["pcn"])
        self.active_inference.load_state_dict(state["active_inference"])
        self.hd_memory.load_state_dict(state["hd_memory"])
        self.world_model.load_state_dict(state["world_model"])
        self.metacognition.load_state_dict(state["metacognition"])
        if "state_to_obs_W" in state:
            self.state_to_obs_W = state["state_to_obs_W"].copy()
        if "state_to_obs_b" in state:
            self.state_to_obs_b = state["state_to_obs_b"].copy()
        self.age = state["age"]
        self.stage = state["stage"]
        if "surprise_threshold" in state:
            self.surprise_threshold = state["surprise_threshold"]
        if "brain_scale" in state:
            self.brain_scale = state["brain_scale"]
        if "experience_buffer" in state:
            _is_large_scale = isinstance(self.brain_scale, str) and self.brain_scale in ("large", "xlarge", "human")
            buffer_maxlen = 100_000_000 if _is_large_scale else 10000
            self.experience_buffer = deque(state["experience_buffer"], maxlen=buffer_maxlen)
        if "consolidation_counter" in state:
            self._consolidation_counter = state["consolidation_counter"]
        if "stats" in state:
            self.stats = {k: list(v) for k, v in state["stats"].items()}
        if "perception_projection_W" in state and state["perception_projection_W"] is not None:
            self._perception_projection_W = state["perception_projection_W"].copy()
            self._perception_projection_input_dim = state.get("perception_projection_input_dim")
        if "global_workspace" in state:
            self.global_workspace.load_state_dict(state["global_workspace"])
        if "structural_evolution" in state:
            self.structural_evolution.load_state_dict(state["structural_evolution"])
        if "complementary_memory" in state:
            self.complementary_memory.load_state_dict(state["complementary_memory"])
        if "school_stage" in state:
            self.school_stage = state["school_stage"]
        if state.get("language_head") and self.language_head is not None:
            self.language_head.load_state_dict(state["language_head"])
        if state.get("asr_head") and self.asr_head is not None:
            self.asr_head.load_state_dict(state["asr_head"])
        if state.get("ocr_head") and self.ocr_head is not None:
            self.ocr_head.load_state_dict(state["ocr_head"])
        if "sparse" in state:
            self.sparse = state["sparse"]
        if "_input_running_mean" in state and state["_input_running_mean"] is not None:
            self._input_running_mean = state["_input_running_mean"].copy()
        if "_input_running_M2" in state and state["_input_running_M2"] is not None:
            self._input_running_M2 = state["_input_running_M2"].copy()
        if "_input_running_count" in state and state["_input_running_count"] is not None:
            self._input_running_count = state["_input_running_count"]

    def get_brain_stats(self) -> Dict[str, Any]:
        """返回人脑规模统计信息"""
        stats = self.get_state_summary()
        stats["sparse"] = self.sparse
        stats["brain_scale"] = self.brain_scale
        if self.sparse:
            for i, layer in enumerate(self.snn_layers):
                if hasattr(layer, '_sparse_conn') and layer._sparse_conn is not None:
                    stats[f"snn_layer_{i}_synapses"] = layer._sparse_conn.n_synapses
                    stats[f"snn_layer_{i}_sparsity"] = layer._sparse_conn.sparsity
        return stats

    def __repr__(self) -> str:
        return (
            f"NIEABrain(input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"action_dim={self.action_dim}, "
            f"stage={self.stage}, age={self.age})"
        )
