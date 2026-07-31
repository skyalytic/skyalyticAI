"""
SNN 层后端工厂

根据硬件后端字符串创建对应的 SNNLayer 实例。
- cuda / npu / mlu / xpu : 复用现有 ``SNNLayer``（PyTorch 设备无关）
- cpu                    : 复用现有 ``SNNLayer``（device=None）
- loihi                  : 占位类，实例化即抛出 ``NotImplementedError``

使用工厂模式屏蔽后端差异，调用方只需传入 ``detect_backend()`` 的结果。
"""

from __future__ import annotations

from typing import Any

from skyalyticAI.neurons.snn_layer import SNNLayer

# PyTorch 设备无关后端：均复用 SNNLayer，仅 device 不同
_TORCH_DEVICE_BACKENDS = ("cuda", "npu", "mlu", "xpu")


class LoihiSNNLayer:
    """
    Loihi 神经形态后端的 SNN 层占位类。

    实际实现需要 Intel NxSDK。在未安装 nxSDK 的环境下，
    实例化即抛出 ``NotImplementedError``，避免误用。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Loihi backend requires nxsdk")

    def forward(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("Loihi backend requires nxsdk")

    def step(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("Loihi backend requires nxsdk")

    def reset(self) -> None:
        raise NotImplementedError("Loihi backend requires nxsdk")


def create_snn_layer(
    backend: str,
    input_dim: int,
    output_dim: int,
    **kwargs: Any,
) -> Any:
    """
    根据后端字符串创建对应的 SNNLayer（或占位类）实例。

    Parameters
    ----------
    backend : str
        后端名称：``"loihi"``、``"cuda"``、``"npu"``、``"mlu"``、
        ``"xpu"``、``"cpu"``（大小写不敏感）。
    input_dim : int
        输入维度（前突触神经元数量）。
    output_dim : int
        输出维度（本层神经元数量）。
    **kwargs
        传递给 ``SNNLayer`` 的其他参数（如 ``neuron_type``、
        ``weight_init``、``weight_scale`` 等）。

    Returns
    -------
    SNNLayer or LoihiSNNLayer
        对应后端的 SNN 层实例。Loihi 后端会抛出
        ``NotImplementedError``。

    Notes
    -----
    - ``cuda`` / ``npu`` / ``mlu`` / ``xpu`` 后端会构造对应的
      ``torch.device``，PyTorch 不可用时自动降级到 CPU。
    - ``cpu`` 后端使用 ``device=None``。
    - 若 ``kwargs`` 中显式传入 ``device``，会覆盖工厂推导的设备。
    """
    backend = backend.lower()

    if backend == "loihi":
        # Loihi 需要专用 SDK，占位类在实例化时报错
        return LoihiSNNLayer(input_dim=input_dim, output_dim=output_dim, **kwargs)

    if backend == "cpu":
        # CPU 后端：device=None，使用 SNNLayer 默认 CPU 路径
        kwargs.pop("device", None)
        return SNNLayer(
            input_dim=input_dim,
            output_dim=output_dim,
            device=None,
            **kwargs,
        )

    if backend in _TORCH_DEVICE_BACKENDS:
        # PyTorch 设备无关后端：构造对应 device
        device = None
        try:
            import torch
            try:
                device = torch.device(backend)
            except (RuntimeError, AssertionError):
                # 后端虽被检测到但当前无法构造 device，降级到 CPU
                device = None
        except ImportError:
            # torch 未安装，降级到 CPU
            device = None

        # 允许调用方显式覆盖 device
        device = kwargs.pop("device", device)
        return SNNLayer(
            input_dim=input_dim,
            output_dim=output_dim,
            device=device,
            **kwargs,
        )

    # 未知后端：降级到 CPU
    kwargs.pop("device", None)
    return SNNLayer(
        input_dim=input_dim,
        output_dim=output_dim,
        device=None,
        **kwargs,
    )
