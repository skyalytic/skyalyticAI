"""
多步推理考试 — 链式推导，逐步预测每一步的答案字符（高阶推理）。
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


class MultiStepReasoningExam(Environment):
    def __init__(
        self,
        corpus: CorpusManager,
        stage: str,
        observation_dim: int = 128,
        n_questions: int = 10,
        pass_accuracy: float = 0.48,
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
            context_len=20,
        )
        self._chains = self._load_chains(exam_bank_path)
        self._chain_idx = 0
        self._step_idx = 0
        self._chain_ctx: List[int] = []
        self._current_step_text = ""
        self._step_indices: List[int] = []
        self._pos = 0
        self._correct = 0
        self._total = 0
        self._finished = False

    def _load_chains(self, path: Optional[Path]) -> List[List[Dict[str, str]]]:
        root = path or (self.corpus.corpus_root.parent / "exams")
        f = root / self.stage / "multi_step_reasoning.jsonl"
        if f.is_file():
            chains = []
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    chains.append(json.loads(line))
            if chains:
                return chains
        return self._synthetic_chains()

    def _synthetic_chains(self) -> List[List[Dict[str, str]]]:
        teacher = TeacherNPC(seed=int(self.rng.integers(0, 10_000)))
        chains: List[List[Dict[str, str]]] = []
        for _ in range(self.n_questions):
            steps = teacher.make_reasoning_chain(self.stage, None)
            chains.append([{"prompt": s.prompt, "answer": s.answer} for s in steps])
        return chains

    def reset(self) -> np.ndarray:
        self.rng.shuffle(self._chains)
        self._chain_idx = 0
        self._correct = 0
        self._total = 0
        self._finished = False
        return self._start_chain()

    def _start_chain(self) -> np.ndarray:
        if self._chain_idx >= min(self.n_questions, len(self._chains)):
            self._finished = True
            return np.zeros(self.observation_dim, dtype=np.float64)
        chain = self._chains[self._chain_idx]
        self._chain_ctx = []
        self._step_idx = 0
        return self._start_step(chain)

    def _start_step(self, chain: List[Dict[str, str]]) -> np.ndarray:
        if self._step_idx >= len(chain):
            self._chain_idx += 1
            return self._start_chain()
        step = chain[self._step_idx]
        prompt = step.get("prompt", "")
        ans = step.get("answer", "0")
        self._current_step_text = prompt
        self._step_indices = self.corpus.encode_char_indices(ans)
        if not self._step_indices:
            self._step_indices = [0]
        self._chain_ctx.extend(self.corpus.encode_char_indices(prompt))
        self._pos = 0
        return self._obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self._finished:
            return (
                np.zeros(self.observation_dim, dtype=np.float64),
                0.0,
                True,
                {"mode": "exam", "exam_type": "multi_step_reasoning", "passed": self.passed()},
            )

        target = self._step_indices[self._pos] if self._pos < len(self._step_indices) else 0
        ok = int(action) % self.vocab_size == target
        self._total += 1
        if ok:
            self._correct += 1
        reward = 1.2 if ok else -0.2
        self._chain_ctx.append(self._step_indices[self._pos] if self._pos < len(self._step_indices) else 0)
        self._pos += 1
        step_done = self._pos >= len(self._step_indices)
        if step_done:
            self._step_idx += 1
            chain = self._chains[self._chain_idx]
            if self._step_idx >= len(chain):
                self._chain_idx += 1
                obs = self._start_chain()
            else:
                obs = self._start_step(chain)
            done = self._finished
        else:
            obs = self._obs()
            done = False
        return obs, reward, done, {
            "mode": "exam",
            "exam_type": "multi_step_reasoning",
            "correct": ok,
            "target_char": self.corpus.index_to_char(target),
            "accuracy": self.accuracy(),
            "reasoning_step": self._step_idx,
            "passed": self.passed() if done else False,
        }

    def accuracy(self) -> float:
        return self._correct / max(self._total, 1)

    def passed(self) -> bool:
        return self.accuracy() >= self.pass_accuracy

    def _obs(self) -> np.ndarray:
        obs = self.text_encoder.encode(self._chain_ctx[-24:])
        if obs.shape[0] < self.observation_dim:
            pad = np.zeros(self.observation_dim, dtype=np.float64)
            pad[: obs.shape[0]] = obs
            obs = pad
        return obs

    def get_observation_dim(self) -> int:
        return self.observation_dim

    def get_action_dim(self) -> int:
        return self.vocab_size

    def render(self):
        return None
