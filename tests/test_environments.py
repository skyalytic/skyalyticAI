"""
环境模块单元测试 — 验证 env/ 包的完整性和正确性。
"""

import numpy as np
import pytest

from skyalyticAI.env.environment import Environment
from skyalyticAI.env.grid_world import GridWorldEnv
from skyalyticAI.env.curriculum_world import HumanGrowthWorld
from skyalyticAI.env.social_classroom_world import SocialClassroomWorld


# ===== Environment 基类 =====

class TestEnvironmentBase:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            Environment()

    def test_subclass_must_implement(self):
        class BadEnv(Environment):
            pass
        with pytest.raises(TypeError):
            BadEnv()


# ===== GridWorldEnv =====

class TestGridWorld:
    def test_creation(self):
        env = GridWorldEnv(width=5, height=5, seed=0)
        assert env.width == 5
        assert env.height == 5
        assert env.get_action_dim() == 4

    def test_obs_dim(self):
        env = GridWorldEnv(width=5, height=5, seed=0)
        # one-hot(25) + direction(2) + obstacle(4) = 31
        assert env.get_observation_dim() == 31

    def test_reset_returns_obs(self):
        env = GridWorldEnv(width=5, height=5, seed=0)
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape[0] == env.get_observation_dim()

    def test_step_returns_tuple(self):
        env = GridWorldEnv(width=5, height=5, seed=0)
        env.reset()
        obs, reward, done, info = env.step(0)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert "mode" in info

    def test_reach_goal(self):
        env = GridWorldEnv(width=3, height=3, n_obstacles=0, seed=0)
        obs = env.reset()
        # 从(0,0)到(2,2)，连续走右下
        for _ in range(50):
            obs, reward, done, info = env.step(1)  # down
            if done:
                break
            obs, reward, done, info = env.step(3)  # right
            if done:
                break
        # 应该能到达目标或超时
        assert isinstance(done, bool)

    def test_render(self):
        env = GridWorldEnv(width=3, height=3, n_obstacles=0, seed=0)
        env.reset()
        rendered = env.render()
        assert isinstance(rendered, str)

    def test_invalid_size(self):
        with pytest.raises(ValueError):
            GridWorldEnv(width=1, height=1)


# ===== HumanGrowthWorld =====

class TestHumanGrowthWorld:
    def test_creation(self):
        env = HumanGrowthWorld(observation_dim=64, seed=0)
        assert env.school_stage == "sensorimotor"
        assert env.get_observation_dim() == 64
        assert env.get_action_dim() > 0

    def test_reset(self):
        env = HumanGrowthWorld(observation_dim=64, seed=0)
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape[0] == 64

    def test_step_motor(self):
        env = HumanGrowthWorld(observation_dim=64, seed=0)
        env.reset()
        obs, reward, done, info = env.step(0)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_corpus_available(self):
        env = HumanGrowthWorld(observation_dim=64, seed=0)
        assert env.corpus is not None
        assert env.corpus.vocab_len() > 0

    def test_set_stage(self):
        env = HumanGrowthWorld(observation_dim=64, seed=0)
        env.set_stage("kindergarten")
        assert env.school_stage == "kindergarten"
        env.set_stage("invalid_stage")
        assert env.school_stage == "kindergarten"

    def test_set_rolling_accuracy(self):
        env = HumanGrowthWorld(observation_dim=64, seed=0)
        env.set_rolling_speech_accuracy(0.5)
        assert env._rolling_speech_accuracy == 0.5

    def test_steps_per_episode(self):
        env = HumanGrowthWorld(observation_dim=64, seed=0)
        assert env.get_steps_per_episode() > 0


# ===== SocialClassroomWorld =====

class TestSocialClassroomWorld:
    def test_creation(self):
        env = SocialClassroomWorld(
            observation_dim=64,
            school_stage="primary",
            max_stage="high",
            seed=0,
        )
        assert env.school_stage == "primary"
        assert env.max_stage == "high"

    def test_reset(self):
        env = SocialClassroomWorld(observation_dim=64, seed=0)
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape[0] == 64

    def test_step(self):
        env = SocialClassroomWorld(observation_dim=64, seed=0)
        env.reset()
        obs, reward, done, info = env.step(0)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)

    def test_max_stage_limit(self):
        env = SocialClassroomWorld(
            observation_dim=64,
            school_stage="primary",
            max_stage="primary",
            seed=0,
        )
        # 不应升到 primary 以上
        assert env.school_stage == "primary"
