"""
NIEA: Neural Isomorphic Evolutionary Architecture

A brain-inspired AI framework implementing four theoretical pillars
and seven core functional modules:

Four Theoretical Pillars:
1. SNN Substrate - Event-driven computing primitives (LIF/ALIF)
2. Predictive Coding Learning - Hierarchical free energy minimization (PCN)
3. Complementary Memory System - Hippocampal-cortical dual architecture (HDC + CMS)
4. Global Workspace Consciousness Emergence - Competitive broadcasting (GW)

Seven Core Functional Modules:
1. Spike Encoder - Rate/temporal coding for sensory input
2. SNN Computing Core - LIF/ALIF neurons with STDP plasticity
3. Predictive Coding Network (PCN) - Hierarchical prediction and error correction
4. Hyperdimensional Memory System (HDC) - Episodic and semantic memory
5. Active Inference Engine - Intrinsic motivation and curiosity-driven exploration
6. World Model (WM) - Recurrent state space model for planning and imagination
7. Metacognition & Structural Self-Evolution - Self-awareness and architecture adaptation
"""

__version__ = "0.4.0"

from skyalyticAI.neurons import LIFNeuron, ALIFNeuron, SNNLayer
from skyalyticAI.plasticity import STDPSynapse, STDPLayer
from skyalyticAI.predictive_coding import PCNLayer, PredictiveCodingNetwork
from skyalyticAI.active_inference import ActiveInferenceAgent
from skyalyticAI.memory import HDCMemory, ComplementaryMemorySystem, HippocampalStore, CorticalStore
from skyalyticAI.world_model import WorldModel
from skyalyticAI.metacognition import MetacognitiveModule
from skyalyticAI.brain import NIEABrain, BrainScalePresets
from skyalyticAI.perception import VisualEncoder, AudioEncoder, MultimodalFusion
try:
    from skyalyticAI.env import Environment, GridWorldEnv, HumanGrowthWorld, TextWorldEnv, ExamWorld
except ImportError:
    Environment = GridWorldEnv = HumanGrowthWorld = TextWorldEnv = ExamWorld = None
from skyalyticAI.training import NIEATrainer, HumanGrowthTrainer
from skyalyticAI.language import TextEncoder, LanguageHead
from skyalyticAI.data.corpus_manager import CorpusManager
from skyalyticAI.data import Dataset, NIEADataLoader, ExperienceReplayDataset, MultimodalDataset
from skyalyticAI.gpu import GPUBatchProcessor, is_gpu_available, get_device
from skyalyticAI.consciousness import GlobalWorkspace
from skyalyticAI.evolution import StructuralEvolution

__all__ = [
    "LIFNeuron",
    "ALIFNeuron",
    "SNNLayer",
    "STDPSynapse",
    "STDPLayer",
    "PCNLayer",
    "PredictiveCodingNetwork",
    "ActiveInferenceAgent",
    "HDCMemory",
    "ComplementaryMemorySystem",
    "HippocampalStore",
    "CorticalStore",
    "WorldModel",
    "MetacognitiveModule",
    "NIEABrain",
    "BrainScalePresets",
    "VisualEncoder",
    "AudioEncoder",
    "MultimodalFusion",
    "Environment",
    "GridWorldEnv",
    "HumanGrowthWorld",
    "TextWorldEnv",
    "ExamWorld",
    "NIEATrainer",
    "HumanGrowthTrainer",
    "TextEncoder",
    "LanguageHead",
    "CorpusManager",
    "Dataset",
    "NIEADataLoader",
    "ExperienceReplayDataset",
    "MultimodalDataset",
    "GPUBatchProcessor",
    "is_gpu_available",
    "get_device",
    "GlobalWorkspace",
    "StructuralEvolution",
]
