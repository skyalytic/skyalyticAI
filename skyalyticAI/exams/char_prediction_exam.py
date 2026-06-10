"""
字符预测考试 — 基础题型（下一字预测），对应识字与语言流畅度。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from skyalyticAI.data.corpus_manager import CorpusManager
from skyalyticAI.env.environment import Environment
from skyalyticAI.language.text_encoder import TextEncoder


class CharPredictionExam(Environment):
    """与 ExamWorld 等价的字符级升学考试。"""

    def __init__(
        self,
        corpus: CorpusManager,
        stage: str,
        observation_dim: int = 128,
        n_questions: int = 20,
        pass_accuracy: float = 0.55,
        seed: int = 0,
    ) -> None:
        self.corpus = corpus
        self.stage = stage
        self.observation_dim = observation_dim
        self.n_questions = n_questions if n_questions > 0 else 0
        self.pass_accuracy = pass_accuracy
        self.rng = np.random.default_rng(seed)
        self.vocab_size = corpus.vocab_len()
        self.text_encoder = TextEncoder(
            vocab_size=max(self.vocab_size, 32),
            output_dim=observation_dim,
        )
        self._lines: List[str] = []
        self._line_idx = 0
        self._indices: List[int] = []
        self._pos = 0
        self._context: List[int] = []
        self._correct = 0
        self._total = 0
        self._finished = False

    def reset(self) -> np.ndarray:
        self._lines = self.corpus.get_exam_lines(self.stage, self.n_questions)
        self._line_idx = 0
        self._correct = 0
        self._total = 0
        self._finished = False
        return self._start_line()

    def _start_line(self) -> np.ndarray:
        if self._line_idx >= len(self._lines):
            self._finished = True
            return np.zeros(self.observation_dim, dtype=np.float64)
        line = self._lines[self._line_idx]
        self._indices = self.corpus.encode_char_indices(line)
        if len(self._indices) < 2:
            self._indices = [0, 0]
        self._pos = 0
        self._context = []
        return self._obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self._finished:
            return (
                np.zeros(self.observation_dim, dtype=np.float64),
                0.0,
                True,
                self._info(done=True),
            )

        target = self._indices[self._pos + 1] if self._pos + 1 < len(self._indices) else 0
        ok = int(action) % self.vocab_size == target
        self._total += 1
        if ok:
            self._correct += 1
        reward = 1.0 if ok else -0.1
        self._context.append(self._indices[self._pos])
        self._pos += 1
        line_done = self._pos >= len(self._indices) - 1
        if line_done:
            self._line_idx += 1
            obs = self._start_line()
            done = self._finished
        else:
            obs = self._obs()
            done = False
        return obs, reward, done, self._info(done=done, correct=ok, target=target)

    def _info(self, done: bool, correct: bool = False, target: int = 0) -> Dict[str, Any]:
        return {
            "mode": "exam",
            "exam_type": "char_prediction",
            "correct": correct,
            "target_char": self.corpus.index_to_char(target),
            "accuracy": self.accuracy(),
            "school_stage": self.stage,
            "passed": self.passed() if done else False,
        }

    def accuracy(self) -> float:
        return self._correct / max(self._total, 1)

    def passed(self) -> bool:
        return self.accuracy() >= self.pass_accuracy

    def _obs(self) -> np.ndarray:
        obs = self.text_encoder.encode(self._context)
        if obs.shape[0] < self.observation_dim:
            pad = np.zeros(self.observation_dim, dtype=np.float64)
            pad[: obs.shape[0]] = obs
            obs = pad
        elif obs.shape[0] > self.observation_dim:
            obs = obs[:self.observation_dim]
        return obs

    def get_observation_dim(self) -> int:
        return self.observation_dim

    def get_action_dim(self) -> int:
        return self.vocab_size

    def render(self):
        return None
