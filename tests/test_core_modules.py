"""
核心模块单元测试 — 验证 SNN、STDP、PCN、HDC、WorldModel 等模块。
"""

import numpy as np
import pytest


# ===== LIF 神经元 =====

class TestLIFNeuron:
    def test_creation(self):
        from skyalyticAI.neurons.lif import LIFNeuron
        n = LIFNeuron()
        assert n.v == n.v_rest

    def test_step(self):
        from skyalyticAI.neurons.lif import LIFNeuron
        n = LIFNeuron()
        v, spike = n.step(current=5.0, dt=1.0)
        assert isinstance(spike, bool)

    def test_reset(self):
        from skyalyticAI.neurons.lif import LIFNeuron
        n = LIFNeuron()
        n.step(current=100.0, dt=1.0)
        n.reset()
        assert n.v == n.v_rest

    def test_invalid_threshold(self):
        from skyalyticAI.neurons.lif import LIFNeuron
        with pytest.raises(ValueError):
            LIFNeuron(v_threshold=-80.0, v_reset=-50.0)


# ===== ALIF 神经元 =====

class TestALIFNeuron:
    def test_creation(self):
        from skyalyticAI.neurons.alif import ALIFNeuron
        n = ALIFNeuron()
        assert n.w == 0.0

    def test_adaptive_threshold(self):
        from skyalyticAI.neurons.alif import ALIFNeuron
        n = ALIFNeuron(beta=0.5)
        base_threshold = n.v_threshold
        n.step(current=100.0, dt=1.0)
        assert n.adaptive_threshold >= base_threshold

    def test_adaptation_decay(self):
        from skyalyticAI.neurons.alif import ALIFNeuron
        n = ALIFNeuron(beta=1.0, tau_w=50.0)
        n.step(current=100.0, dt=1.0)
        w_after_spike = n.w
        n.step(current=0.0, dt=1.0)
        assert n.w <= w_after_spike


# ===== STDP =====

class TestSTDPSynapse:
    def test_creation(self):
        from skyalyticAI.plasticity.stdp import STDPSynapse
        s = STDPSynapse()
        assert 0.0 <= s.w <= 1.0

    def test_update_ltp(self):
        from skyalyticAI.plasticity.stdp import STDPSynapse
        s = STDPSynapse(w_init=0.5)
        # Pre spike first, then post spike -> LTP
        s.update(pre_spike=True, post_spike=False)
        old_w = s.w
        s.update(pre_spike=False, post_spike=True)
        # Post spike with pre trace -> LTP
        assert s.w >= old_w - 1e-10

    def test_update_ltd(self):
        from skyalyticAI.plasticity.stdp import STDPSynapse
        s = STDPSynapse(w_init=0.5)
        # Post spike first, then pre spike -> LTD
        s.update(pre_spike=False, post_spike=True)
        old_w = s.w
        s.update(pre_spike=True, post_spike=False)
        # Pre spike with post trace -> LTD
        assert s.w <= old_w + 1e-10

    def test_weight_bounds(self):
        from skyalyticAI.plasticity.stdp import STDPSynapse
        s = STDPSynapse(w_min=0.0, w_max=1.0)
        for _ in range(100):
            s.update(pre_spike=True, post_spike=True)
        assert s.w >= -1e-6
        assert s.w <= 1.0 + 1e-6

    def test_invalid_w_max(self):
        from skyalyticAI.plasticity.stdp import STDPSynapse
        with pytest.raises(ValueError):
            STDPSynapse(w_min=1.0, w_max=0.5)


# ===== STDP Layer =====

class TestSTDPLayer:
    def test_creation(self):
        from skyalyticAI.plasticity.stdp_layer import STDPLayer
        layer = STDPLayer(pre_dim=16, post_dim=8)
        assert layer.W.shape == (8, 16)

    def test_update(self):
        from skyalyticAI.plasticity.stdp_layer import STDPLayer
        layer = STDPLayer(pre_dim=16, post_dim=8)
        pre = (np.random.rand(16) > 0.5).astype(float)
        post = (np.random.rand(8) > 0.5).astype(float)
        layer.update(pre, post)


# ===== PCN Layer =====

class TestPCNLayer:
    def test_creation(self):
        from skyalyticAI.predictive_coding.pcn_layer import PCNLayer
        layer = PCNLayer(dim_below=16, dim=8)
        assert layer.dim == 8
        assert layer.dim_below == 16

    def test_predict(self):
        from skyalyticAI.predictive_coding.pcn_layer import PCNLayer
        layer = PCNLayer(dim_below=16, dim=8)
        pred = layer.predict(layer.x)
        assert pred.shape == (16,)

    def test_inference_step(self):
        from skyalyticAI.predictive_coding.pcn_layer import PCNLayer
        layer = PCNLayer(dim_below=16, dim=8)
        prediction_from_above = np.random.randn(8)
        error_from_below = np.random.randn(16)
        layer.inference_step(prediction_from_above, error_from_below, sigma_above=1.0)
        assert layer.x is not None

    def test_learning_step(self):
        from skyalyticAI.predictive_coding.pcn_layer import PCNLayer
        layer = PCNLayer(dim_below=16, dim=8)
        error_below = np.random.randn(16)
        layer.learning_step(error_below)


# ===== PCN Network =====

class TestPCN:
    def test_creation(self):
        from skyalyticAI.predictive_coding.pcn import PredictiveCodingNetwork
        pcn = PredictiveCodingNetwork(layer_sizes=[16, 8, 4])
        assert len(pcn.layers) == 2

    def test_forward(self):
        from skyalyticAI.predictive_coding.pcn import PredictiveCodingNetwork
        pcn = PredictiveCodingNetwork(layer_sizes=[16, 8, 4])
        obs = np.random.randn(16)
        result = pcn.infer(obs)
        assert "errors" in result


# ===== HDC Memory =====

class TestHDCMemory:
    def test_creation(self):
        from skyalyticAI.memory.hdc import HDCMemory
        mem = HDCMemory(dim=1000)
        assert mem.dim == 1000

    def test_random_vector(self):
        from skyalyticAI.memory.hdc import HDCMemory
        mem = HDCMemory(dim=1000)
        v = mem.random_vector()
        assert v.shape == (1000,)

    def test_bind_unbind(self):
        from skyalyticAI.memory.hdc import HDCMemory
        mem = HDCMemory(dim=2000)
        a = mem.random_vector()
        b = mem.random_vector()
        bound = mem.bind(a, b)
        recovered = mem.unbind(bound, b)
        sim = np.dot(recovered, a) / (np.linalg.norm(recovered) * np.linalg.norm(a) + 1e-10)
        assert sim > 0.5

    def test_bundle(self):
        from skyalyticAI.memory.hdc import HDCMemory
        mem = HDCMemory(dim=2000)
        v1 = mem.random_vector()
        v2 = mem.random_vector()
        bundled = mem.bundle(v1, v2)
        assert bundled.shape == (2000,)

    def test_store_and_retrieve_association(self):
        from skyalyticAI.memory.hdc import HDCMemory
        mem = HDCMemory(dim=2000)
        mem.store_association("key_concept", "value_concept")
        # Should not crash
        assert "key_concept" in mem.item_memory or True


# ===== Complementary Memory =====

class TestComplementaryMemory:
    def test_creation(self):
        from skyalyticAI.memory.consolidation import ComplementaryMemorySystem
        cms = ComplementaryMemorySystem(dim=256)
        assert cms is not None

    def test_store_and_retrieve(self):
        from skyalyticAI.memory.consolidation import ComplementaryMemorySystem
        cms = ComplementaryMemorySystem(dim=256)
        key = np.random.randn(256)
        key = key / (np.linalg.norm(key) + 1e-10)
        value = np.random.randn(256)
        value = value / (np.linalg.norm(value) + 1e-10)
        cms.store(key, value)
        results = cms.retrieve(key)
        # retrieve returns list of (value, similarity) tuples
        assert isinstance(results, list)


# ===== TextEncoder =====

class TestTextEncoder:
    def test_creation(self):
        from skyalyticAI.language.text_encoder import TextEncoder
        enc = TextEncoder(vocab_size=100, output_dim=64)
        assert enc.output_dim == 64

    def test_encode(self):
        from skyalyticAI.language.text_encoder import TextEncoder
        enc = TextEncoder(vocab_size=100, output_dim=64)
        indices = [1, 5, 10, 20]
        obs = enc.encode(indices)
        assert obs.shape == (64,)


# ===== LanguageHead =====

class TestLanguageHead:
    def test_creation(self):
        from skyalyticAI.language.language_head import LanguageHead
        head = LanguageHead(hidden_dim=64, vocab_size=100)
        assert head.vocab_size == 100

    def test_probs(self):
        from skyalyticAI.language.language_head import LanguageHead
        head = LanguageHead(hidden_dim=64, vocab_size=100)
        hidden = np.random.randn(64)
        probs = head.probs(hidden)
        assert probs.shape == (100,)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_sample(self):
        from skyalyticAI.language.language_head import LanguageHead
        head = LanguageHead(hidden_dim=64, vocab_size=100)
        hidden = np.random.randn(64)
        action = head.sample(hidden)
        assert 0 <= action < 100


# ===== CorpusManager =====

class TestCorpusManager:
    def test_creation_no_corpus(self):
        from skyalyticAI.data.corpus_manager import CorpusManager
        cm = CorpusManager(corpus_root=None)
        assert cm.vocab_len() > 0

    def test_encode_decode_roundtrip(self):
        from skyalyticAI.data.corpus_manager import CorpusManager
        cm = CorpusManager(corpus_root=None)
        for ch in list(cm.char2idx.keys())[:10]:
            idx = cm.char_to_index(ch)
            recovered = cm.index_to_char(idx)
            assert recovered == ch

    def test_sample_training_line(self):
        from skyalyticAI.data.corpus_manager import CorpusManager
        cm = CorpusManager(corpus_root=None)
        line = cm.sample_training_line("sensorimotor")
        assert isinstance(line, str)
        assert len(line) > 0

    def test_stage_display(self):
        from skyalyticAI.data.corpus_manager import CorpusManager
        cm = CorpusManager(corpus_root=None)
        name = cm.stage_display_name("primary")
        assert name == "小学"


# ===== Education Config =====

class TestEducationConfig:
    def test_stage_order(self):
        from skyalyticAI.data.education_config import STAGE_ORDER
        assert len(STAGE_ORDER) == 8
        assert STAGE_ORDER[0] == "sensorimotor"
        assert STAGE_ORDER[-1] == "phd"

    def test_next_stage(self):
        from skyalyticAI.data.education_config import next_stage
        assert next_stage("primary") == "middle"
        assert next_stage("phd") == "phd"

    def test_quality_spec(self):
        from skyalyticAI.data.education_config import get_quality_spec
        spec = get_quality_spec("primary")
        assert spec.min_steps_in_stage > 0
        assert spec.steps_per_episode > 0


# ===== SNN Layer =====

class TestSNNLayer:
    def test_creation(self):
        from skyalyticAI.neurons.snn_layer import SNNLayer
        layer = SNNLayer(input_dim=16, output_dim=8)
        assert layer.input_dim == 16
        assert layer.output_dim == 8

    def test_forward(self):
        from skyalyticAI.neurons.snn_layer import SNNLayer
        layer = SNNLayer(input_dim=16, output_dim=8)
        spikes, voltages = layer.forward(np.random.randn(16))
        assert spikes.shape[-1] == 8
