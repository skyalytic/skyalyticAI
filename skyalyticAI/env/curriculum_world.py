"""
类人成长世界 — 0~3 岁感知运动 + 上学多科 + 升学考试。

核心机制：
1) 学段升学线：sensorimotor -> kindergarten -> primary -> ... -> phd
2) 双活动模式：motor（迷宫探索）+ reading（文本阅读/考试）
3) 升学三重门槛：步数 + 说话准确率 + 考试间隔
4) 科目轮换：每学段多科并行，防偏科
5) 考试系统：ExamSuite 多题型轮换

与 HumanGrowthTrainer 的接口：
  - school_stage: 当前学段
  - corpus: CorpusManager 实例
  - _current_subject: 当前科目
  - _activity: 当前活动模式
  - _episodes_since_exam: 距上次考试的回合数
  - set_rolling_speech_accuracy(acc)
  - set_rolling_subject_accuracy(acc_dict)
  - get_steps_per_episode()
  - set_stage(stage)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from skyalyticAI.data.corpus_manager import CorpusManager
from skyalyticAI.data.education_config import (
    NON_SCHOOL_EXAM_STAGES,
    STAGE_ORDER,
    get_quality_spec,
    next_stage,
    subjects_for_stage,
)
from skyalyticAI.env.environment import Environment
from skyalyticAI.env.grid_world import GridWorldEnv
from skyalyticAI.exams.exam_suite import ExamSuite
from skyalyticAI.language.text_encoder import TextEncoder


class Activity(str, Enum):
    MOTOR = "motor"
    READING = "reading"
    EXAM = "exam"


class HumanGrowthWorld(Environment):
    """类人成长训练环境 — 模拟 0~3 岁到博士的完整教育路径。"""

    def __init__(
        self,
        corpus_root: Optional[str] = None,
        observation_dim: int = 128,
        exam_every_n_episodes: int = 5,
        exam_pass_accuracy: float = 0.55,
        seed: int = 42,
    ) -> None:
        self.observation_dim = observation_dim
        self.exam_every_n_episodes = max(1, exam_every_n_episodes)
        self.exam_pass_accuracy = exam_pass_accuracy
        self.rng = np.random.default_rng(seed)

        # 语料
        self.corpus = CorpusManager(corpus_root=corpus_root, seed=seed)
        self.vocab_size = max(self.corpus.vocab_len(), 32)
        self.text_encoder = TextEncoder(
            vocab_size=self.vocab_size,
            output_dim=observation_dim,
        )

        # 学段
        self.school_stage: str = "sensorimotor"
        self._spec = get_quality_spec(self.school_stage)

        # 迷宫子环境（感知运动期使用）
        self._grid_world = GridWorldEnv(width=6, height=6, n_obstacles=4, seed=seed)

        # 考试套件
        self._exam_suite = ExamSuite(
            corpus=self.corpus,
            stage=self.school_stage,
            observation_dim=observation_dim,
            seed=seed,
        )

        # 状态
        self._activity: Activity = Activity.MOTOR
        self._current_subject: Optional[str] = None
        self._steps_in_stage: int = 0
        self._episodes_since_exam: int = 0
        self._rolling_speech_accuracy: float = 0.0
        self._rolling_subject_accuracy: Dict[str, float] = {}
        self._step_in_episode: int = 0

        # 文本阅读状态
        self._text_line: str = ""
        self._text_indices: List[int] = []
        self._text_pos: int = 0
        self._text_correct: int = 0
        self._text_total: int = 0

        # 考试状态
        self._in_exam: bool = False

    # ----- 训练器兼容接口 -----

    def set_rolling_speech_accuracy(self, acc: float) -> None:
        self._rolling_speech_accuracy = float(acc)

    def set_rolling_subject_accuracy(self, subject_acc: Dict[str, float]) -> None:
        self._rolling_subject_accuracy = dict(subject_acc)

    def get_steps_per_episode(self) -> int:
        return self._spec.steps_per_episode

    def set_stage(self, stage: str) -> None:
        if stage not in STAGE_ORDER:
            return
        self.school_stage = stage
        self._spec = get_quality_spec(stage)
        self._steps_in_stage = 0
        self._episodes_since_exam = 0
        self._exam_suite.set_stage(stage)

    # ----- 内部逻辑 -----

    def _pick_activity(self) -> Activity:
        """根据学段配置选择当前活动。"""
        if self._in_exam:
            return Activity.EXAM

        spec = self._spec
        r = self.rng.random()
        if r < spec.motor_ratio:
            return Activity.MOTOR
        else:
            return Activity.READING

    def _pick_subject(self) -> str:
        """选择当前科目。"""
        subjects = subjects_for_stage(self.school_stage)
        if not subjects:
            return "通识"
        if self._rolling_subject_accuracy:
            # 偏向薄弱科目
            weak = [s for s in subjects if self._rolling_subject_accuracy.get(s, 1.0) < 0.5]
            if weak and self.rng.random() < 0.6:
                return str(self.rng.choice(weak))
        return str(self.rng.choice(subjects))

    def _should_take_exam(self) -> bool:
        """判断是否应该参加考试。"""
        if self.school_stage in NON_SCHOOL_EXAM_STAGES:
            return False
        if not self._spec.allows_subject_exam:
            return False
        if self._episodes_since_exam < self._spec.min_episodes_between_exams:
            return False
        if self._steps_in_stage < self._spec.min_steps_in_stage:
            return False
        if self._rolling_speech_accuracy < self._spec.min_rolling_speech_accuracy:
            return False
        return True

    def _try_promote(self) -> bool:
        """尝试升学。"""
        nxt = next_stage(self.school_stage)
        if nxt == self.school_stage:
            return False
        self.school_stage = nxt
        self._spec = get_quality_spec(nxt)
        self._steps_in_stage = 0
        self._episodes_since_exam = 0
        self._exam_suite.set_stage(nxt)
        return True

    # ----- 环境主接口 -----

    def reset(self) -> np.ndarray:
        self._episodes_since_exam += 1
        self._spec = get_quality_spec(self.school_stage)
        self._step_in_episode = 0

        # 判断是否进入考试模式
        if self._should_take_exam():
            self._in_exam = True
            self._activity = Activity.EXAM
            return self._exam_obs()
        else:
            self._in_exam = False
            self._activity = self._pick_activity()
            if self._activity == Activity.MOTOR:
                obs = self._grid_world.reset()
                return self._project_obs(obs)
            else:
                return self._start_reading()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self._step_in_episode += 1
        self._steps_in_stage += 1

        if self._activity == Activity.MOTOR:
            return self._step_motor(action)
        elif self._activity == Activity.READING:
            return self._step_reading(action)
        else:
            return self._step_exam(action)

    # ----- 迷宫模式 -----

    def _step_motor(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, reward, done, info = self._grid_world.step(action)
        info["mode"] = "motor"
        info["activity"] = "motor"
        info["school_stage"] = self.school_stage

        # 迷宫观测可能维度不同，投影到 observation_dim
        if isinstance(obs, np.ndarray) and obs.shape[0] != self.observation_dim:
            obs = self._project_obs(obs)

        # 回合结束条件
        if self._step_in_episode >= self._spec.steps_per_episode:
            done = True

        return obs, reward, done, info

    # ----- 阅读模式 -----

    def _start_reading(self) -> np.ndarray:
        self._current_subject = self._pick_subject()
        self._text_line = self.corpus.sample_training_line(
            self.school_stage, self._current_subject
        )
        self._text_indices = self.corpus.encode_char_indices(self._text_line)
        if len(self._text_indices) < 2:
            self._text_indices = [0, 0]
        self._text_pos = 0
        self._text_correct = 0
        self._text_total = 0
        return self._reading_obs()

    def _reading_obs(self) -> np.ndarray:
        context = self._text_indices[:self._text_pos]
        obs = self.text_encoder.encode(context)
        if obs.shape[0] < self.observation_dim:
            pad = np.zeros(self.observation_dim, dtype=np.float64)
            pad[:obs.shape[0]] = obs
            obs = pad
        elif obs.shape[0] > self.observation_dim:
            obs = obs[:self.observation_dim]
        return obs

    def _step_reading(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        action = int(action) % self.vocab_size
        target = self._text_indices[self._text_pos + 1] if self._text_pos + 1 < len(self._text_indices) else 0
        ok = action == target
        self._text_total += 1
        if ok:
            self._text_correct += 1
        reward = 1.0 if ok else -0.1
        self._text_pos += 1

        line_done = self._text_pos >= len(self._text_indices) - 1
        done = line_done or self._step_in_episode >= self._spec.steps_per_episode

        # 在行切换前保存当前步骤的上下文（奖励/正确性属于旧行）
        accuracy = self._text_correct / max(self._text_total, 1)
        current_subject = self._current_subject

        if line_done and not done:
            # 继续读下一行（通过 _start_reading 以支持子类覆写）
            self._start_reading()

        info: Dict[str, Any] = {
            "mode": "text",
            "activity": "reading",
            "school_stage": self.school_stage,
            "subject": current_subject,
            "correct": ok,
            "target_char": self.corpus.index_to_char(target),
            "spoken_char": self.corpus.index_to_char(action),
            "accuracy": accuracy,
        }
        return self._reading_obs(), reward, done, info

    # ----- 考试模式 -----

    def _exam_obs(self) -> np.ndarray:
        return self._exam_suite.reset()

    def _step_exam(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, reward, done, info = self._exam_suite.step(action)
        info["mode"] = "exam"
        info["activity"] = "exam"
        info["school_stage"] = self.school_stage

        if done:
            self._in_exam = False
            self._episodes_since_exam = 0
            passed = self._exam_suite.passed()
            info["passed"] = passed
            if passed:
                promoted = self._try_promote()
                info["promoted"] = promoted
            else:
                info["promoted"] = False

        return obs, reward, done, info

    # ----- 工具方法 -----

    def _project_obs(self, obs: np.ndarray) -> np.ndarray:
        """将任意维度观测投影到 observation_dim。"""
        if obs.shape[0] == self.observation_dim:
            return obs
        result = np.zeros(self.observation_dim, dtype=np.float64)
        n = min(obs.shape[0], self.observation_dim)
        result[:n] = obs[:n]
        return result

    def get_observation_dim(self) -> int:
        return self.observation_dim

    def get_action_dim(self) -> int:
        return self.vocab_size

    def render(self):
        if self._activity == Activity.MOTOR:
            return self._grid_world.render()
        elif self._activity == Activity.READING:
            return self._text_line[:self._text_pos] + "|" + self._text_line[self._text_pos:]
        else:
            return f"考试中... 准确率: {self._exam_suite.composite_accuracy():.1%}"
