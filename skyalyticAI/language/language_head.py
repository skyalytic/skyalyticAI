"""
语言输出头 — 把内部状态变成「说出来的字」（下一个字符）。

通过环境反馈（对错、奖励）在线学习，与主动推理选动作并行。

v2: 皮层化语言头 — 多层解码网络（概念层→字符层）+ scheduled sampling。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


class LanguageHead:
    """
    皮层化语言头：hidden → 概念层 → 字符概率。

    架构：
        Layer 1: hidden_dim → lang_hidden_dim (tanh)  ← 概念解码层
        Layer 2: lang_hidden_dim → vocab_size (linear) ← 字符输出层

    与单层线性头相比：
    - 概念层引入非线性，能学习更复杂的 hidden→character 映射
    - 深度解码允许网络学习语法树的中间表示
    - scheduled sampling 支持从教师强制逐步过渡到自主生成

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
    lang_hidden_dim : int
        概念层维度（默认=hidden_dim，设为0时退化为单层线性头）。
    teacher_forcing_rate : float
        Scheduled sampling：训练时使用目标字符（而非模型预测）的概率。
        1.0 = 完全教师强制（默认，早期训练）
        0.0 = 完全自主生成（后期训练）
        建议随训练进展从 1.0 逐步降到 0.5。
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        learning_rate: float = 0.05,
        temperature: float = 1.0,
        lang_hidden_dim: Optional[int] = None,
        teacher_forcing_rate: float = 1.0,
    ) -> None:
        if hidden_dim <= 0 or vocab_size <= 0:
            raise ValueError("hidden_dim 与 vocab_size 须为正")

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.temperature = max(temperature, 1e-6)
        self.teacher_forcing_rate = max(0.0, min(1.0, teacher_forcing_rate))

        # 概念层维度：None 或 0 时退化为单层线性头（向后兼容）
        if lang_hidden_dim is None or lang_hidden_dim <= 0:
            self.lang_hidden_dim = hidden_dim
            self._use_deep = False
        else:
            self.lang_hidden_dim = lang_hidden_dim
            self._use_deep = True

        # 输出层权重（与旧版兼容）
        self.W = np.random.randn(vocab_size, self.lang_hidden_dim) * np.sqrt(
            2.0 / (self.lang_hidden_dim + vocab_size)
        )
        self.b = np.zeros(vocab_size, dtype=np.float64)

        if self._use_deep:
            # 概念层权重：hidden_dim → lang_hidden_dim
            self.W1 = np.random.randn(self.lang_hidden_dim, hidden_dim) * np.sqrt(
                2.0 / (hidden_dim + self.lang_hidden_dim)
            )
            self.b1 = np.zeros(self.lang_hidden_dim, dtype=np.float64)

        self._last_hidden: Optional[np.ndarray] = None

    def _align_hidden(self, hidden: np.ndarray) -> np.ndarray:
        hidden = np.asarray(hidden, dtype=np.float64).flatten()
        if hidden.shape[0] == self.hidden_dim:
            return hidden
        out = np.zeros(self.hidden_dim, dtype=np.float64)
        n = min(hidden.shape[0], self.hidden_dim)
        out[:n] = hidden[:n]
        return out

    def _concept_forward(self, hidden: np.ndarray) -> np.ndarray:
        """概念层前向传播：hidden → 概念表征（tanh）"""
        if self._use_deep:
            return np.tanh(self.W1 @ hidden + self.b1)
        return hidden

    def logits(self, hidden: np.ndarray) -> np.ndarray:
        h = self._align_hidden(hidden)
        concept = self._concept_forward(h)
        return self.W @ concept + self.b

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
        """
        根据对错奖励更新说话权重（多层反向传播 + scheduled sampling）。

        Scheduled Sampling（单步架构适配版）：
        - 以概率 teacher_forcing_rate 执行教师强制（朝真实目标学习，正常学习率）
        - 以概率 (1 - teacher_forcing_rate) 执行自主探索（朝模型预测学习，较小学习率）
        - 训练初期 teacher_forcing_rate=1.0（完全教师强制，快速收敛）
        - 训练后期逐步降低（brain.develop 中调用 set_teacher_forcing_rate）
          让模型有更多自主空间，减少对教师信号的过度依赖

        与 RNN 版 scheduled sampling 的对应：
        - RNN 版：用模型预测作为下一步输入（解决序列生成的暴露偏差）
        - 单步版：用模型预测作为伪目标进行学习（让模型自主探索时仍能学习）
        两者都实现了"逐步减少对教师的依赖"这一核心思想。
        """
        if target_index < 0 or target_index >= self.vocab_size:
            raise ValueError("target_index 超出词表")

        h = self._align_hidden(hidden)
        concept = self._concept_forward(h)
        p = self.probs(h)

        # Scheduled sampling：以概率 teacher_forcing_rate 执行教师强制
        if np.random.random() >= self.teacher_forcing_rate:
            # 自主探索：用模型预测的字符作为伪目标
            # 置信度门控：只有模型预测有明确偏好时才用伪目标
            # 避免早期预测接近均匀分布时引入表征漂移
            max_prob = float(np.max(p))
            random_prob = 1.0 / self.vocab_size  # 随机猜测的概率
            # 阈值 = 3倍随机概率（vocab=10时阈值=0.3）
            # 低于阈值说明模型还不自信，回退到教师强制
            if max_prob > 3.0 * random_prob:
                pseudo_target_idx = int(np.argmax(p))
                target = np.zeros(self.vocab_size, dtype=np.float64)
                target[pseudo_target_idx] = 1.0
                # 自主探索时用较小的 reward（弱监督），避免模式崩溃
                reward_mag = 0.1
            else:
                # 置信度不足，回退到教师强制（避免表征漂移）
                target = np.zeros(self.vocab_size, dtype=np.float64)
                target[target_index] = 1.0
                reward_mag = max(abs(float(reward)), 0.1)
        else:
            # 教师强制：朝真实目标方向更新
            target = np.zeros(self.vocab_size, dtype=np.float64)
            target[target_index] = 1.0
            # 保证最小学习率：reward=0 时仍能学习（训练初期常见 reward=0）
            reward_mag = max(abs(float(reward)), 0.1)

        scale = self.learning_rate * reward_mag / self.temperature
        err = target - p  # (vocab_size,)

        # 保存更新前的 W（用于概念层梯度的正确反向传播）
        W_old = self.W.copy()

        # 输出层梯度
        dW_out = scale * np.outer(err, concept)  # (vocab, lang_hidden)
        db_out = scale * err

        self.W += dW_out
        self.b += db_out

        if self._use_deep:
            # 概念层梯度（用更新前的 W 反传，保证梯度正确性）
            d_concept = W_old.T @ err * scale  # (lang_hidden,)
            # tanh 导数
            d_concept *= (1.0 - concept ** 2)

            dW1 = np.outer(d_concept, h)  # (lang_hidden, hidden)
            db1 = d_concept

            self.W1 += dW1
            self.b1 += db1

            self.W1 = np.clip(self.W1, -5.0, 5.0)
            self.b1 = np.clip(self.b1, -5.0, 5.0)

        self.W = np.clip(self.W, -5.0, 5.0)
        self.b = np.clip(self.b, -5.0, 5.0)

        return {"speech_loss": float(np.linalg.norm(err)), "target_prob": float(p[target_index])}

    def set_teacher_forcing_rate(self, rate: float) -> None:
        """Scheduled sampling：调整教师强制概率。"""
        self.teacher_forcing_rate = max(0.0, min(1.0, rate))

    def state_dict(self) -> Dict[str, np.ndarray]:
        state = {
            "W": self.W.copy(),
            "b": self.b.copy(),
            "teacher_forcing_rate": self.teacher_forcing_rate,
            "temperature": self.temperature,
            "_use_deep": self._use_deep,
            "lang_hidden_dim": self.lang_hidden_dim,
        }
        if self._use_deep:
            state["W1"] = self.W1.copy()
            state["b1"] = self.b1.copy()
        return state

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        self.W = state["W"].copy()
        self.b = state["b"].copy()
        self.teacher_forcing_rate = float(state.get("teacher_forcing_rate", 1.0))
        self.temperature = float(state.get("temperature", 1.0))
        self._use_deep = bool(state.get("_use_deep", False))
        self.lang_hidden_dim = int(state.get("lang_hidden_dim", self.hidden_dim))
        if self._use_deep and "W1" in state:
            self.W1 = state["W1"].copy()
            self.b1 = state["b1"].copy()
