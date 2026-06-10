"""
阅读理解考试 — 给定短文上下文，预测答案句中的下一字（高阶语言理解）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from skyalyticAI.data.corpus_manager import CorpusManager
from skyalyticAI.env.environment import Environment
from skyalyticAI.language.text_encoder import TextEncoder
from skyalyticAI.npc.teacher_npc import TeacherNPC


class ReadingComprehensionExam(Environment):
    def __init__(
        self,
        corpus: CorpusManager,
        stage: str,
        observation_dim: int = 128,
        n_questions: int = 15,
        pass_accuracy: float = 0.50,
        seed: int = 0,
        exam_bank_path: Optional[Path] = None,
    ) -> None:
        self.corpus = corpus
        self.stage = stage
        self.observation_dim = observation_dim
        self.n_questions = max(1, n_questions)
        self.pass_accuracy = pass_accuracy
        self.rng = np.random.default_rng(seed)
        self.vocab_size = corpus.vocab_len()
        self.text_encoder = TextEncoder(
            vocab_size=max(self.vocab_size, 32),
            output_dim=observation_dim,
            context_len=16,
        )
        self._items = self._load_items(exam_bank_path)
        self._idx = 0
        self._passage_ctx: List[int] = []
        self._answer_ctx: List[int] = []
        self._answer_indices: List[int] = []
        self._pos = 0
        self._correct = 0
        self._total = 0
        self._finished = False

    def _load_items(self, path: Optional[Path]) -> List[Dict[str, str]]:
        root = path or (self.corpus.corpus_root.parent / "exams")
        f = root / self.stage / "reading_comprehension.jsonl"
        if f.is_file():
            items = []
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    items.append(json.loads(line))
            if items:
                return items
        return self._synthetic_items()

    def _synthetic_items(self) -> List[Dict[str, str]]:
        # 优先使用 NPC 老师生成阅读理解题（不依赖静态课件文件）
        teacher = TeacherNPC(seed=int(self.rng.integers(0, 10_000)))
        items = []
        for _ in range(self.n_questions):
            it = teacher.make_reading_item(self.stage, None)
            items.append({"passage": it.passage, "question": it.question, "answer": it.answer})
        return items

    def reset(self) -> np.ndarray:
        self.rng.shuffle(self._items)
        self._idx = 0
        self._correct = 0
        self._total = 0
        self._finished = False
        return self._start_item()

    def _start_item(self) -> np.ndarray:
        if self._idx >= min(self.n_questions, len(self._items)):
            self._finished = True
            return np.zeros(self.observation_dim, dtype=np.float64)
        item = self._items[self._idx]
        passage = item.get("passage", "") + item.get("question", "")
        answer = item.get("answer", "是")
        self._passage_ctx = self.corpus.encode_char_indices(passage)
        self._answer_indices = self.corpus.encode_char_indices(answer)
        if len(self._answer_indices) < 1:
            self._answer_indices = [0]
        self._answer_ctx = list(self._passage_ctx[-8:])
        self._pos = 0
        return self._obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self._finished:
            return (
                np.zeros(self.observation_dim, dtype=np.float64),
                0.0,
                True,
                {"mode": "exam", "exam_type": "reading_comprehension", "passed": self.passed()},
            )

        target = self._answer_indices[self._pos] if self._pos < len(self._answer_indices) else 0
        ok = int(action) % self.vocab_size == target
        self._total += 1
        if ok:
            self._correct += 1
        reward = 1.0 if ok else -0.15
        self._answer_ctx.append(self._answer_indices[self._pos] if self._pos < len(self._answer_indices) else 0)
        self._pos += 1
        item_done = self._pos >= len(self._answer_indices)
        if item_done:
            self._idx += 1
            obs = self._start_item()
            done = self._finished
        else:
            obs = self._obs()
            done = False
        return obs, reward, done, {
            "mode": "exam",
            "exam_type": "reading_comprehension",
            "correct": ok,
            "target_char": self.corpus.index_to_char(target),
            "accuracy": self.accuracy(),
            "passed": self.passed() if done else False,
        }

    def accuracy(self) -> float:
        return self._correct / max(self._total, 1)

    def passed(self) -> bool:
        return self.accuracy() >= self.pass_accuracy

    def _obs(self) -> np.ndarray:
        ctx = self._passage_ctx + self._answer_ctx
        obs = self.text_encoder.encode(ctx)
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
