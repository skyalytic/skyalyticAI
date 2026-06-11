"""
集成测试 — 验证端到端训练管道。
"""

import numpy as np
import pytest

from skyalyticAI.brain import NIEABrain
from skyalyticAI.env.grid_world import GridWorldEnv
from skyalyticAI.env.curriculum_world import HumanGrowthWorld
from skyalyticAI.training.trainer import NIEATrainer


class TestGridWorldTraining:
    """验证 GridWorld 环境下的完整训练循环。"""

    def test_single_episode(self):
        env = GridWorldEnv(width=5, height=5, n_obstacles=2, seed=0)
        brain = NIEABrain(
            input_dim=env.get_observation_dim(),
            hidden_dim=32,
            action_dim=4,
            n_observations=16,
            hd_dim=500,
            pcn_hidden_dim=16,
            world_model_hidden_dim=16,
            device="cpu",
        )
        trainer = NIEATrainer(
            brain=brain,
            env=env,
            max_episodes=1,
            max_steps_per_episode=10,
            checkpoint_dir=None,
        )
        summary = trainer.train()
        assert summary["total_episodes"] == 1
        assert summary["total_steps"] > 0

    def test_three_episodes(self):
        env = GridWorldEnv(width=5, height=5, seed=0)
        brain = NIEABrain(
            input_dim=env.get_observation_dim(),
            hidden_dim=32,
            action_dim=4,
            n_observations=16,
            hd_dim=500,
            pcn_hidden_dim=16,
            world_model_hidden_dim=16,
            device="cpu",
        )
        trainer = NIEATrainer(
            brain=brain,
            env=env,
            max_episodes=3,
            max_steps_per_episode=10,
            checkpoint_dir=None,
        )
        summary = trainer.train()
        assert summary["total_episodes"] == 3


class TestHumanGrowthTraining:
    """验证类人成长训练循环。"""

    def test_single_episode(self):
        env = HumanGrowthWorld(observation_dim=64, seed=0)
        brain = NIEABrain(
            input_dim=64,
            hidden_dim=32,
            action_dim=env.get_action_dim(),
            n_observations=16,
            hd_dim=500,
            pcn_hidden_dim=16,
            world_model_hidden_dim=16,
            language_vocab_size=env.get_action_dim(),
            device="cpu",
        )
        trainer = NIEATrainer(
            brain=brain,
            env=env,
            max_episodes=1,
            max_steps_per_episode=10,
            checkpoint_dir=None,
        )
        summary = trainer.train()
        assert summary["total_episodes"] == 1


class TestImportChain:
    """验证所有导入链完整。"""

    def test_import_env(self):
        from skyalyticAI.env import Environment, GridWorldEnv, HumanGrowthWorld
        assert Environment is not None
        assert GridWorldEnv is not None
        assert HumanGrowthWorld is not None

    def test_import_social_classroom(self):
        from skyalyticAI.env import SocialClassroomWorld
        assert SocialClassroomWorld is not None

    def test_import_top_level(self):
        import skyalyticAI
        assert skyalyticAI.__version__ == "0.3.1"

    def test_import_brain(self):
        from skyalyticAI.brain import NIEABrain, BrainScalePresets
        assert NIEABrain is not None

    def test_import_all_modules(self):
        from skyalyticAI.neurons import LIFNeuron, ALIFNeuron, SNNLayer
        from skyalyticAI.plasticity import STDPSynapse, STDPLayer
        from skyalyticAI.predictive_coding import PCNLayer, PredictiveCodingNetwork
        from skyalyticAI.active_inference import ActiveInferenceAgent
        from skyalyticAI.memory import HDCMemory, ComplementaryMemorySystem
        from skyalyticAI.world_model import WorldModel
        from skyalyticAI.metacognition import MetacognitiveModule
        from skyalyticAI.consciousness import GlobalWorkspace
        from skyalyticAI.evolution import StructuralEvolution
        from skyalyticAI.language import TextEncoder, LanguageHead
        from skyalyticAI.perception import VisualEncoder, AudioEncoder, MultimodalFusion
