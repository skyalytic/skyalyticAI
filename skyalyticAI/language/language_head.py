"""
语言输出头 — 把内部状态变成「说出来的字」（下一个字符）。

通过环境反馈（对错、奖励）在线学习，与主动推理选动作并行。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


class LanguageHead:
    """
    线性层 + softmax：hidden -> 词表概率。

    Parameters
    ----------
    hidden_dim : int
        隐藏状态维度。
    vocab_size : int
        字符词表大小。
    learning_rate : float
        在线学习率。
    temperature : float
        softmax 温度。
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        learning_rate: float = 0.05,
        temperature: float = 1.0,
    ) -> None:
        if hidden_dim <= 0 or vocab_size <= 0:
            raise ValueError("hidden_dim 与 vocab_size 须为正")

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.temperature = max(temperature, 1e-6)

        self.W = np.random.randn(vocab_size, hidden_dim) * np.sqrt(
            2.0 / (hidden_dim + vocab_size)
        )
        self.b = np.zeros(vocab_size, dtype=np.float64)
        self._last_hidden: Optional[np.ndarray] = None

    def _align_hidden(self, hidden: np.ndarray) -> np.ndarray:
        hidden = np.asarray(hidden, dtype=np.float64).flatten()
        if hidden.shape[0] == self.hidden_dim:
            return hidden
        out = np.zeros(self.hidden_dim, dtype=np.float64)
        n = min(hidden.shape[0], self.hidden_dim)
        out[:n] = hidden[:n]
        return out

    def logits(self, hidden: np.ndarray) -> np.ndarray:
        h = self._align_hidden(hidden)
        return self.W @ h + self.b

    def probs(self, hidden: np.ndarray) -> np.ndarray:
        z = self.logits(hidden) / self.temperature
        z -= np.max(z)
        e = np.exp(z)
        p = e / np.sum(e)
        p = p / p.sum()  # ensure strict normalization
        self._last_hidden = self._align_hidden(hidden)
        return p

    def sample(self, hidden: np.ndarray, rng: Optional[np.random.Generator] = None) -> int:
        p = self.probs(hidden)
        if rng is None:
            rng = np.random.default_rng()
        return int(rng.choice(self.vocab_size, p=p))

    def argmax(self, hidden: np.ndarray) -> int:
        return int(np.argmax(self.logits(hidden)))

    def learn(self, hidden: np.ndarray, target_index: int, reward: float) -> Dict[str, float]:
        """根据对错奖励更新说话权重。"""
        if target_index < 0 or target_index >= self.vocab_size:
            raise ValueError("target_index 超出词表")

        h = self._align_hidden(hidden)
        p = self.probs(h)
        target = np.zeros(self.vocab_size, dtype=np.float64)
        target[target_index] = 1.0
        # Always move toward target; magnitude proportional to |reward|
        # (target_index is the correct answer regardless of reward sign)
        scale = self.learning_rate * abs(float(reward)) / self.temperature
        err = target - p
        self.W += scale * np.outer(err, h)
        self.b += scale * err
        self.W = np.clip(self.W, -5.0, 5.0)
        self.b = np.clip(self.b, -5.0, 5.0)
        return {"speech_loss": float(np.linalg.norm(err)), "target_prob": float(p[target_index])}

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"W": self.W.copy(), "b": self.b.copy()}

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        self.W = state["W"].copy()
        self.b = state["b"].copy()
