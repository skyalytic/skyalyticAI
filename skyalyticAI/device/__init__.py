"""
设备后端抽象层

提供多后端硬件检测（Loihi / CUDA / NPU / MLU / XPU / CPU）与
SNN 层工厂模式。当对应依赖未安装时自动降级到 CPU。

Usage:
    from skyalyticAI.device import (
        detect_backend, get_device, get_backend_info,
        is_backend_available, create_snn_layer,
    )

    backend = detect_backend()         # 例如 "cuda"
    device = get_device()              # 例如 torch.device("cuda")
    info = get_backend_info()          # 例如 "CUDA: ... (24.0 GB)"
    layer = create_snn_layer(backend, input_dim=784, output_dim=128)
"""

from skyalyticAI.device.detector import (
    detect_backend,
    get_device,
    get_backend_info,
    is_backend_available,
)
from skyalyticAI.device.backend_factory import create_snn_layer

__all__ = [
    "detect_backend",
    "get_device",
    "get_backend_info",
    "is_backend_available",
    "create_snn_layer",
]
