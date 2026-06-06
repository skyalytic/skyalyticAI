"""
训练质量门控 — 确保每学段步数与准确率达标后才允许升学考试。

避免「没学会就考试」的玩具级训练节奏。
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional

from skyalyticAI.data.education_config import StageQualitySpec, get_quality_spec


class TrainingQualityGate:
    """
    跟踪当前学段训练量与说话准确率，决定是否允许升学考。

    Parameters
    ----------
    speech_window : int
        滚动窗口大小（用于计算近期说话准确率）。
    """

    def __init__(self, speech_window: int = 500) -> None:
        self.speech_window = max(50, speech_window)
        self._steps_in_stage: int = 0
        self._episodes_in_stage: int = 0
        self._episodes_since_exam: int = 0
        self._speech_results: Deque[int] = deque(maxlen=self.speech_window)

    def reset_stage_counters(self) -> None:
        """升学后重置本学段计数。"""
        self._steps_in_stage = 0
        self._episodes_in_stage = 0
        self._episodes_since_exam = 0

    def record_step(self, correct: Optional[bool] = None) -> None:
        self._steps_in_stage += 1
        if correct is not None:
            self._speech_results.append(1 if correct else 0)

    def record_episode_end(self) -> None:
        self._episodes_in_stage += 1
        self._episodes_since_exam += 1

    def rolling_speech_accuracy(self) -> float:
        if not self._speech_results:
            return 0.0
        return sum(self._speech_results) / len(self._speech_results)

    def can_take_exam(self, stage: str) -> bool:
        spec = get_quality_spec(stage)
        if self._steps_in_stage < spec.min_steps_in_stage:
            return False
        if self._episodes_since_exam < spec.min_episodes_between_exams:
            return False
        if self.rolling_speech_accuracy() < spec.min_rolling_speech_accuracy:
            return False
        return True

    def steps_per_episode(self, stage: str) -> int:
        return get_quality_spec(stage).steps_per_episode

    def exam_questions(self, stage: str) -> int:
        return get_quality_spec(stage).exam_questions

    def exam_pass_accuracy(self, stage: str) -> float:
        return get_quality_spec(stage).exam_pass_accuracy

    def motor_ratio(self, stage: str) -> float:
        return get_quality_spec(stage).motor_ratio

    def status(self, stage: str) -> Dict[str, float]:
        spec = get_quality_spec(stage)
        return {
            "steps_in_stage": float(self._steps_in_stage),
            "min_steps_required": float(spec.min_steps_in_stage),
            "speech_accuracy": self.rolling_speech_accuracy(),
            "min_speech_required": spec.min_rolling_speech_accuracy,
            "episodes_since_exam": float(self._episodes_since_exam),
            "min_episodes_between_exams": float(spec.min_episodes_between_exams),
            "exam_ready": float(self.can_take_exam(stage)),
        }
