"""
文本编码器 — 把字符上下文变成大脑可读的观测向量（相当于「听/读进脑」）。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class TextEncoder:
    """
    将最近若干个字符索引编码为固定维观测。

    Parameters
    ----------
    vocab_size : int
        词表大小。
    output_dim : int
        输出维度，应与 NIEABrain.input_dim 一致。
    context_len : int
        上下文长度（最近几个字）。
    seed : int
        随机种子，保证同一字符投影稳定。
    """

    def __init__(
        self,
        vocab_size: int,
        output_dim: int,
        context_len: int = 12,
        seed: int = 42,
    ) -> None:
        if vocab_size <= 0 or output_dim <= 0 or context_len <= 0:
            raise ValueError("vocab_size、output_dim、context_len 须为正")

        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.context_len = context_len

        rng = np.random.default_rng(seed)
        block = max(1, output_dim // (context_len + 1))
        self._char_proj = rng.standard_normal((vocab_size, block)) * np.sqrt(
            2.0 / vocab_size
        )
        self._pos_proj = rng.standard_normal((context_len + 1, block)) * 0.1

    def encode(self, context_indices: Sequence[int]) -> np.ndarray:
        """编码字符索引序列为观测向量。"""
        obs = np.zeros(self.output_dim, dtype=np.float64)
        if not context_indices:
            return obs

        window = list(context_indices)[-self.context_len :]
        block = self._char_proj.shape[1]
        offset = 0
        for pos, idx in enumerate(window):
            if idx < 0 or idx >= self.vocab_size:
                continue
            chunk = self._char_proj[idx] + self._pos_proj[pos]
            end = min(offset + block, self.output_dim)
            n = end - offset
            obs[offset:end] += chunk[:n]
            offset = end
            if offset >= self.output_dim:
                break

        norm = np.linalg.norm(obs)
        if norm > 1e-10:
            obs /= norm
        return obs
