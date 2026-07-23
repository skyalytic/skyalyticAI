"""
Retina Encoder - 视网膜前端编码器（生物可解释，无预训练，不学习）。

模拟从光子到神经节细胞脉冲的完整视网膜通路：

    像素亮度
      -> 光感受器适应（Weber 归一化 + Naka-Rushton 压缩）
      -> 双极细胞空间汇总（感受野面积平均，任意分辨率输入）
      -> 神经节细胞中心-周边拮抗（DoG：ON/OFF 双通道）
      -> 脉冲发放（伯努利速率编码，等价于强度调制的泊松过程）

理论立场（fix 119）：
- 视网膜是固定的感觉器官硬件，本身不学习——学习发生在皮层（SNN/STDP/PCN）。
  因此本编码器没有任何可学习参数，adapt() 恒返回 0。
- 不依赖任何预训练深度网络（CNN/ViT），保持 NIEA 从底层数值到上层认知的
  全链路神经同构：输出可以直接作为脉冲序列送入 SNN。
- encode() 返回平均发放率向量（与 MultimodalFusion 接口兼容）；
  encode_spikes() 返回完整脉冲序列（供 SNN 直接消费）。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class RetinaEncoder:
    """
    视网膜编码器：真实图像 -> 神经节细胞脉冲。

    Parameters
    ----------
    ganglion_rows : int
        神经节细胞感受野网格行数（垂直分辨率）。
    ganglion_cols : int
        神经节细胞感受野网格列数（水平分辨率）。
    on_off : bool
        是否使用 ON/OFF 双通道（真实视网膜约各半）。
    spike_steps : int
        脉冲编码时间步数。
    output_dim : int
        encode() 输出向量维度（面积重采样自感受野网格）。
    stochastic : bool
        True=随机发放（生物现实，含神经噪声）；False=确定性速率（便于调试）。
    seed : int or None
        随机种子。
    """

    def __init__(
        self,
        ganglion_rows: int = 16,
        ganglion_cols: int = 16,
        on_off: bool = True,
        spike_steps: int = 20,
        output_dim: int = 128,
        stochastic: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        if ganglion_rows <= 0 or ganglion_cols <= 0:
            raise ValueError("ganglion grid must be positive")
        if spike_steps <= 0:
            raise ValueError("spike_steps must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")

        self.ganglion_rows = ganglion_rows
        self.ganglion_cols = ganglion_cols
        self.on_off = on_off
        self.spike_steps = spike_steps
        self.output_dim = output_dim
        self.stochastic = stochastic
        self.rng = np.random.default_rng(seed)

        self.n_receptors = ganglion_rows * ganglion_cols * (2 if on_off else 1)
        self._last_rates = np.zeros(self.n_receptors, dtype=np.float64)
        self._last_spikes = np.zeros((self.n_receptors, spike_steps), dtype=np.float64)

    # ----- 光感受器 + 双极细胞 -----
    @staticmethod
    def _to_luminance(image: np.ndarray) -> np.ndarray:
        """任意图像 -> [0,1] 亮度图。RGB 用 Rec.601 加权（人眼视锥敏感度）。"""
        img = np.asarray(image, dtype=np.float64)
        if img.ndim == 3:
            if img.shape[0] in (1, 3) and img.shape[0] <= img.shape[2]:
                img = img.transpose(1, 2, 0)
            if img.shape[2] >= 3:
                img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
            else:
                img = img[..., 0]
        if img.size == 0:
            return np.zeros((1, 1), dtype=np.float64)
        if img.max() > 1.0:
            img = img / 255.0
        return np.clip(img, 0.0, 1.0)

    @staticmethod
    def _area_resize(x: np.ndarray, rows: int, cols: int) -> np.ndarray:
        """面积平均重采样到 (rows, cols)——双极细胞对感受野内光感受器的空间汇总。"""
        h, w = x.shape
        ri = np.floor(np.linspace(0, h, rows + 1)).astype(int)
        ci = np.floor(np.linspace(0, w, cols + 1)).astype(int)
        ri[0], ci[0] = 0, 0
        ri[-1], ci[-1] = h, w
        starts_r = np.unique(np.clip(ri[:-1], 0, h - 1))
        starts_c = np.unique(np.clip(ci[:-1], 0, w - 1))
        summed = np.add.reduceat(np.add.reduceat(x, starts_r, axis=0), starts_c, axis=1)
        counts_r = np.diff(np.concatenate([starts_r, [h]])).astype(np.float64)
        counts_c = np.diff(np.concatenate([starts_c, [w]])).astype(np.float64)
        counts = np.maximum(counts_r[:, None] * counts_c[None, :], 1.0)
        out = summed / counts
        if out.shape != (rows, cols):
            fixed = np.zeros((rows, cols), dtype=np.float64)
            r = min(rows, out.shape[0])
            c = min(cols, out.shape[1])
            fixed[:r, :c] = out[:r, :c]
            out = fixed
        return out

    @staticmethod
    def _gaussian_kernel1d(sigma: float) -> np.ndarray:
        radius = max(1, int(3.0 * sigma))
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
        return k / k.sum()

    @classmethod
    def _dog(cls, grid: np.ndarray, sigma_c: float = 0.8, sigma_s: float = 2.4) -> np.ndarray:
        """中心-周边拮抗（DoG），numpy 可分离卷积实现。"""
        kc = cls._gaussian_kernel1d(sigma_c)
        ks = cls._gaussian_kernel1d(sigma_s)

        def sep_conv(img: np.ndarray, k: np.ndarray) -> np.ndarray:
            pad = len(k) // 2
            p = np.pad(img, ((pad, pad), (0, 0)), mode="reflect")
            tmp = np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 1, p)
            p2 = np.pad(tmp, ((0, 0), (pad, pad)), mode="reflect")
            return np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 0, p2)

        return sep_conv(grid, kc) - sep_conv(grid, ks)

    def _phototransduce(self, image: np.ndarray) -> np.ndarray:
        """像素 -> 神经节细胞发放率（感受野网格, 或 2x网格 for ON/OFF）。"""
        lum = self._to_luminance(image)
        grid = self._area_resize(lum, self.ganglion_rows, self.ganglion_cols)

        # 光适应：Weber 归一化（除以平均亮度）+ Naka-Rushton 压缩 R = L/(1+L)
        mean_lum = float(grid.mean())
        adapted = grid / (mean_lum + 1e-6)
        compressed = adapted / (1.0 + adapted)

        # 神经节细胞：中心-周边拮抗 -> ON/OFF 通道
        dog = self._dog(compressed)
        if self.on_off:
            rates = np.concatenate([np.maximum(dog, 0.0).ravel(), np.maximum(-dog, 0.0).ravel()])
        else:
            rates = np.maximum(dog, 0.0).ravel()

        peak = float(rates.max())
        if peak > 1e-9:
            rates = rates / peak
        # 极低对比度（均匀画面）时退化为绝对亮度编码，保证有信号
        if rates.max() < 1e-6:
            rates = compressed.ravel()
            if self.on_off:
                rates = np.concatenate([rates, np.zeros_like(rates)])
        return np.clip(rates, 0.0, 1.0)

    # ----- 脉冲发放 -----
    def encode_spikes(self, image: np.ndarray) -> np.ndarray:
        """
        图像 -> 脉冲序列，shape (n_receptors, spike_steps)。

        伯努利速率编码：每个时间步以概率=发放率产生脉冲，
        等价于强度调制的非齐次泊松过程（视网膜神经节细胞的标准模型）。
        """
        rates = self._phototransduce(image)
        if self.stochastic:
            spikes = (self.rng.random((rates.shape[0], self.spike_steps)) < rates[:, None]).astype(np.float64)
        else:
            spikes = np.repeat(rates[:, None], self.spike_steps, axis=1)
        self._last_rates = rates
        self._last_spikes = spikes
        return spikes

    def encode(self, image: np.ndarray) -> np.ndarray:
        """
        图像 -> 平均发放率向量，shape (output_dim,)（与 MultimodalFusion 兼容）。

        感受野发放率经一维面积重采样到 output_dim。
        """
        self.encode_spikes(image)
        rates = self._last_rates
        n = rates.shape[0]
        if n == self.output_dim:
            return rates.astype(np.float64)
        # 一维面积重采样
        idx = np.floor(np.linspace(0, n, self.output_dim + 1)).astype(int)
        out = np.zeros(self.output_dim, dtype=np.float64)
        for i in range(self.output_dim):
            lo, hi = idx[i], max(idx[i + 1], idx[i] + 1)
            hi = min(hi, n)
            out[i] = rates[lo:hi].mean()
        return out

    def adapt(
        self,
        image: np.ndarray,
        target: Optional[np.ndarray] = None,
        reconstruction: Optional[np.ndarray] = None,
    ) -> float:
        """视网膜不学习（学习在皮层）。保留接口兼容，恒返回 0。"""
        return 0.0

    def state_dict(self) -> Dict[str, Any]:
        """前端为固定硬件，仅保存配置（无可学习参数）。"""
        return {
            "ganglion_rows": self.ganglion_rows,
            "ganglion_cols": self.ganglion_cols,
            "on_off": self.on_off,
            "spike_steps": self.spike_steps,
            "output_dim": self.output_dim,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """无可学习参数，仅校验配置一致性。"""
        if int(state.get("output_dim", self.output_dim)) != self.output_dim:
            raise ValueError("RetinaEncoder output_dim mismatch with checkpoint")

    def __repr__(self) -> str:
        return (
            f"RetinaEncoder(grid=({self.ganglion_rows},{self.ganglion_cols}), "
            f"on_off={self.on_off}, receptors={self.n_receptors}, "
            f"steps={self.spike_steps}, output_dim={self.output_dim})"
        )
