"""
Cochlea Encoder - 耳蜗前端编码器（生物可解释，无预训练，不学习）。

模拟从声波到听神经脉冲的完整耳蜗通路：

    声压波形
      -> 基底膜频率分解（ERB 尺度分布的 gammatone 滤波器组，
         Glasberg & Moore 1990 的耳蜗位置-频率映射）
      -> 毛细胞换能（半波整流 + 低通包络 + 立方根压缩，
         模拟内毛细胞的非线性输入-输出特性）
      -> 听神经发放（伯努利速率编码，等价于强度调制的泊松过程）

理论立场（fix 119）：
- 耳蜗是固定的感觉器官硬件，本身不学习——学习发生在听觉皮层（SNN/STDP/PCN）。
  因此本编码器没有任何可学习参数，adapt() 恒返回 0。
- 不依赖任何预训练模型（Whisper 等），保持全链路神经同构。
- encode() 返回平均发放率向量（与 MultimodalFusion 接口兼容）；
  encode_spikes() 返回完整脉冲序列（供 SNN 直接消费）。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy import signal as _sig


class CochleaEncoder:
    """
    耳蜗编码器：真实声波 -> 听神经脉冲。

    Parameters
    ----------
    n_fibers : int
        听神经纤维数（频率通道数），按 ERB 尺度分布于 [f_min, f_max]。
    f_min : float
        最低中心频率（Hz）。
    f_max : float
        最高中心频率（Hz，自动限制在 Nyquist 以下）。
    sample_rate : int
        内部工作采样率（Hz）；输入波形若采样率不同会自动重采样。
    envelope_cutoff : float
        毛细胞包络低通截止频率（Hz）。
    spike_steps : int
        脉冲编码时间步数。
    output_dim : int
        encode() 输出向量维度。
    stochastic : bool
        True=随机发放（生物现实）；False=确定性速率（便于调试）。
    seed : int or None
        随机种子。
    """

    def __init__(
        self,
        n_fibers: int = 64,
        f_min: float = 100.0,
        f_max: float = 8000.0,
        sample_rate: int = 16000,
        envelope_cutoff: float = 800.0,
        spike_steps: int = 20,
        output_dim: int = 128,
        stochastic: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        if n_fibers <= 0:
            raise ValueError("n_fibers must be positive")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if spike_steps <= 0:
            raise ValueError("spike_steps must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")

        self.n_fibers = n_fibers
        self.f_min = f_min
        self.f_max = min(f_max, sample_rate * 0.45)
        self.sample_rate = sample_rate
        self.envelope_cutoff = envelope_cutoff
        self.spike_steps = spike_steps
        self.output_dim = output_dim
        self.stochastic = stochastic
        self.rng = np.random.default_rng(seed)

        self.center_freqs = self._erb_space(f_min, self.f_max, n_fibers)
        self._filters = self._design_filterbank()
        self._env_sos = _sig.butter(
            2, min(envelope_cutoff, sample_rate * 0.45), btype="low", fs=sample_rate, output="sos"
        )

        self._last_rates = np.zeros(n_fibers, dtype=np.float64)
        self._last_spikes = np.zeros((n_fibers, spike_steps), dtype=np.float64)

    # ----- 基底膜 -----
    @staticmethod
    def _erb_space(f_min: float, f_max: float, n: int) -> np.ndarray:
        """ERB 尺度等距分布中心频率（Glasberg & Moore 耳蜗位置-频率映射）。"""
        def to_erb(f: float) -> float:
            return 21.4 * np.log10(4.37e-3 * f + 1.0)

        def to_hz(e: float) -> float:
            return (10.0 ** (e / 21.4) - 1.0) / 4.37e-3

        erbs = np.linspace(to_erb(f_min), to_erb(f_max), n)
        return to_hz(erbs)

    def _design_filterbank(self):
        """gammatone 滤波器组（耳蜗滤波的标准模型）；不可用时退化为 Butterworth 带通。"""
        bank = []
        has_gamma = hasattr(_sig, "gammatone")
        for fc in self.center_freqs:
            if has_gamma:
                b, a = _sig.gammatone(fc, "iir", fs=self.sample_rate)
                bank.append(("ba", (b, a)))
            else:
                bw = max(fc * 0.25, 20.0)
                lo = max(fc - bw / 2.0, 10.0)
                hi = min(fc + bw / 2.0, self.sample_rate * 0.49)
                sos = _sig.butter(2, [lo, hi], btype="band", fs=self.sample_rate, output="sos")
                bank.append(("sos", sos))
        return bank

    def _basilar_membrane(self, wave: np.ndarray) -> np.ndarray:
        """波形 -> (n_fibers, n_samples) 各频率通道的基底膜振动。"""
        out = np.zeros((self.n_fibers, wave.shape[0]), dtype=np.float64)
        for i, (kind, coef) in enumerate(self._filters):
            if kind == "ba":
                b, a = coef
                out[i] = _sig.lfilter(b, a, wave)
            else:
                out[i] = _sig.sosfilt(coef, wave)
        return out

    def _hair_cells(self, channels: np.ndarray) -> np.ndarray:
        """毛细胞换能：半波整流 -> 包络低通 -> 立方根压缩 -> 每纤维平均发放率。"""
        rectified = np.maximum(channels, 0.0)
        envelope = _sig.sosfilt(self._env_sos, rectified, axis=1)
        compressed = np.cbrt(np.maximum(envelope, 0.0))
        rates = compressed.mean(axis=1)
        peak = float(rates.max())
        if peak > 1e-12:
            rates = rates / peak
        return np.clip(rates, 0.0, 1.0)

    def _resample(self, wave: np.ndarray, input_sample_rate: int) -> np.ndarray:
        """线性插值重采样到内部工作采样率。"""
        if input_sample_rate == self.sample_rate or wave.shape[0] == 0:
            return wave
        duration = wave.shape[0] / float(input_sample_rate)
        n_target = max(1, int(round(duration * self.sample_rate)))
        x_old = np.linspace(0.0, duration, num=wave.shape[0], endpoint=False)
        x_new = np.linspace(0.0, duration, num=n_target, endpoint=False)
        return np.interp(x_new, x_old, wave)

    # ----- 听神经发放 -----
    def encode_spikes(self, audio: np.ndarray, input_sample_rate: Optional[int] = None) -> np.ndarray:
        """
        声波 -> 脉冲序列，shape (n_fibers, spike_steps)。

        Parameters
        ----------
        audio : np.ndarray
            1D 声压波形（[-1,1] 或任意 PCM 范围，自动归一化）。
        input_sample_rate : int or None
            输入波形采样率；None 表示与内部 sample_rate 一致。
        """
        wave = np.asarray(audio, dtype=np.float64).flatten()
        if wave.size == 0:
            wave = np.zeros(self.sample_rate // 10, dtype=np.float64)
        peak = float(np.abs(wave).max())
        if peak > 1.0:
            wave = wave / peak
        if input_sample_rate is not None:
            wave = self._resample(wave, int(input_sample_rate))

        channels = self._basilar_membrane(wave)
        rates = self._hair_cells(channels)

        if self.stochastic:
            spikes = (self.rng.random((self.n_fibers, self.spike_steps)) < rates[:, None]).astype(np.float64)
        else:
            spikes = np.repeat(rates[:, None], self.spike_steps, axis=1)
        self._last_rates = rates
        self._last_spikes = spikes
        return spikes

    def encode(self, audio: np.ndarray, input_sample_rate: Optional[int] = None) -> np.ndarray:
        """声波 -> 平均发放率向量，shape (output_dim,)（与 MultimodalFusion 兼容）。"""
        self.encode_spikes(audio, input_sample_rate)
        rates = self._last_rates
        n = rates.shape[0]
        if n == self.output_dim:
            return rates.astype(np.float64)
        idx = np.floor(np.linspace(0, n, self.output_dim + 1)).astype(int)
        out = np.zeros(self.output_dim, dtype=np.float64)
        for i in range(self.output_dim):
            lo, hi = idx[i], max(idx[i + 1], idx[i] + 1)
            hi = min(hi, n)
            out[i] = rates[lo:hi].mean()
        return out

    def adapt(
        self,
        audio: np.ndarray,
        target: Optional[np.ndarray] = None,
        reconstruction: Optional[np.ndarray] = None,
    ) -> float:
        """耳蜗不学习（学习在皮层）。保留接口兼容，恒返回 0。"""
        return 0.0

    def state_dict(self) -> Dict[str, Any]:
        """前端为固定硬件，仅保存配置（无可学习参数）。"""
        return {
            "n_fibers": self.n_fibers,
            "f_min": self.f_min,
            "f_max": self.f_max,
            "sample_rate": self.sample_rate,
            "output_dim": self.output_dim,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """无可学习参数，仅校验配置一致性。"""
        if int(state.get("output_dim", self.output_dim)) != self.output_dim:
            raise ValueError("CochleaEncoder output_dim mismatch with checkpoint")

    def __repr__(self) -> str:
        return (
            f"CochleaEncoder(fibers={self.n_fibers}, "
            f"freqs=[{self.f_min:.0f},{self.f_max:.0f}]Hz, "
            f"steps={self.spike_steps}, output_dim={self.output_dim})"
        )
