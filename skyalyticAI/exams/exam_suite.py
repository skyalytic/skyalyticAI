"""
考试套件 — 轮换基础题、阅读理解、多步推理。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np

from skyalyticAI.data.corpus_manager import CorpusManager
from skyalyticAI.data.education_config import LIGHT_EXAM_STAGES, NON_SCHOOL_EXAM_STAGES, get_quality_spec
from skyalyticAI.env.environment import Environment
from skyalyticAI.exams.char_prediction_exam import CharPredictionExam
from skyalyticAI.exams.reading_comprehension_exam import ReadingComprehensionExam
from skyalyticAI.exams.multi_step_reasoning_exam import MultiStepReasoningExam


class ExamType(str, Enum):
    CHAR = "char_prediction"
    READING = "reading_comprehension"
    REASONING = "multi_step_reasoning"


class ExamSuite(Environment):
    """按学段组合多种题型，统一 Environment 接口。"""

    def __init__(
        self,
        corpus: CorpusManager,
        stage: str,
        observation_dim: int = 128,
        seed: int = 0,
    ) -> None:
        self.corpus = corpus
        self.stage = stage
        self.observation_dim = observation_dim
        self.rng = np.random.default_rng(seed)
        spec = get_quality_spec(stage)
        self._char = CharPredictionExam(
            corpus, stage, observation_dim,
            n_questions=spec.exam_questions,
            pass_accuracy=spec.exam_pass_accuracy if spec.exam_pass_accuracy > 0 else 0.5,
            seed=seed,
        )
        self._reading = ReadingComprehensionExam(
            corpus, stage, observation_dim,
            n_questions=max(8, spec.exam_questions // 2),
            pass_accuracy=max(0.45, (spec.exam_pass_accuracy if spec.exam_pass_accuracy > 0 else 0.5) - 0.05),
            seed=seed + 1,
        )
        self._reasoning = MultiStepReasoningExam(
            corpus, stage, observation_dim,
            n_questions=max(6, spec.exam_questions // 3),
            pass_accuracy=max(0.42, (spec.exam_pass_accuracy if spec.exam_pass_accuracy > 0 else 0.5) - 0.07),
            seed=seed + 2,
        )
        self._active = self._char
        self._exam_type = ExamType.CHAR
        self._scores: Dict[str, float] = {}

    def set_stage(self, stage: str) -> None:
        """切换学段并重建所有子考试。"""
        self.stage = stage
        spec = get_quality_spec(stage)
        seed = int(self.rng.integers(0, 2**31))
        self._char = CharPredictionExam(
            self.corpus, stage, self.observation_dim,
            n_questions=spec.exam_questions,
            pass_accuracy=spec.exam_pass_accuracy if spec.exam_pass_accuracy > 0 else 0.5,
            seed=seed,
        )
        self._reading = ReadingComprehensionExam(
            self.corpus, stage, self.observation_dim,
            n_questions=max(8, spec.exam_questions // 2),
            pass_accuracy=max(0.45, (spec.exam_pass_accuracy if spec.exam_pass_accuracy > 0 else 0.5) - 0.05),
            seed=seed + 1,
        )
        self._reasoning = MultiStepReasoningExam(
            self.corpus, stage, self.observation_dim,
            n_questions=max(6, spec.exam_questions // 3),
            pass_accuracy=max(0.42, (spec.exam_pass_accuracy if spec.exam_pass_accuracy > 0 else 0.5) - 0.07),
            seed=seed + 2,
        )

    def _pick_exam(self) -> None:
        if self.stage in NON_SCHOOL_EXAM_STAGES:
            self._active = self._char
            self._exam_type = ExamType.CHAR
            return
        r = self.rng.random()
        if self.stage in LIGHT_EXAM_STAGES:
            self._active = self._reading if r < 0.6 else self._char
        elif r < 0.34:
            self._active = self._char
        elif r < 0.67:
            self._active = self._reading
        else:
            self._active = self._reasoning
        if self._active is self._char:
            self._exam_type = ExamType.CHAR
        elif self._active is self._reading:
            self._exam_type = ExamType.READING
        else:
            self._exam_type = ExamType.REASONING

    def reset(self) -> np.ndarray:
        self._pick_exam()
        self._scores = {}
        return self._active.reset()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, reward, done, info = self._active.step(action)
        info = dict(info)
        info["exam_type"] = self._exam_type.value
        if done:
            self._scores[self._exam_type.value] = float(info.get("accuracy", 0.0))
        return obs, reward, done, info

    def composite_accuracy(self) -> float:
        if not self._scores:
            return self._active.accuracy()
        return float(np.mean(list(self._scores.values())))

    def passed(self) -> bool:
        if self.stage in NON_SCHOOL_EXAM_STAGES:
            return False
        spec = get_quality_spec(self.stage)
        return self.composite_accuracy() >= (spec.exam_pass_accuracy if spec.exam_pass_accuracy > 0 else 0.5)

    def get_observation_dim(self) -> int:
        return self.observation_dim

    def get_action_dim(self) -> int:
        return self.corpus.vocab_len()

    def render(self):
        return None
