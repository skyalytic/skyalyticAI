"""
多后端硬件检测模块

按优先级检测可用的计算后端：Loihi > CUDA > NPU > MLU > XPU > CPU
每个后端的检测独立进行，缺失依赖时自动跳过。
当 PyTorch 未安装时，整体降级到 CPU。

支持的后端：
    - loihi : Intel Loihi 神经形态芯片（通过 nxSDK）
    - cuda  : NVIDIA GPU（通过 torch.cuda）
    - npu   : 华为昇腾 NPU（通过 torch_npu）
    - mlu   : 寒武纪 MLU（通过 torch_mlu）
    - xpu   : Intel GPU/FPGA（通过 torch.xpu / intel-extension-for-pytorch）
    - cpu   : 通用 CPU（始终可用）
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

# 检测 torch 是否可用（torch 未安装时本模块仍可 import，整体降级到 CPU）
_TORCH_AVAILABLE = False
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore

# 后端优先级顺序（高 -> 低）
_BACKEND_PRIORITY = ("loihi", "cuda", "npu", "mlu", "xpu", "cpu")

# 检测结果缓存：只检测一次
_detected_backend: Optional[str] = None


def _check_loihi() -> bool:
    """检测 Intel Loihi 神经形态芯片（通过 nxSDK）。"""
    try:
        import nxsdk  # noqa: F401
        return True
    except ImportError:
        return False
    except Exception:
        # nxSDK 安装但环境异常（如缺少硬件）也视为不可用
        return False


def _check_cuda() -> bool:
    """检测 NVIDIA CUDA（通过 torch.cuda）。"""
    if not _TORCH_AVAILABLE:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _check_npu() -> bool:
    """检测华为昇腾 NPU（通过 torch_npu，会向 torch 注入 torch.npu）。"""
    if not _TORCH_AVAILABLE:
        return False
    try:
        import torch_npu  # noqa: F401
        return bool(torch.npu.is_available())
    except ImportError:
        return False
    except Exception:
        return False


def _check_mlu() -> bool:
    """检测寒武纪 MLU（通过 torch_mlu，会向 torch 注入 torch.mlu）。"""
    if not _TORCH_AVAILABLE:
        return False
    try:
        import torch_mlu  # noqa: F401
        return bool(torch.mlu.is_available())
    except ImportError:
        return False
    except Exception:
        return False


def _check_xpu() -> bool:
    """检测 Intel XPU（torch.xpu 内置或通过 intel-extension-for-pytorch）。"""
    if not _TORCH_AVAILABLE:
        return False
    # 较新版本 PyTorch 内置 torch.xpu；旧版本需要 intel-extension-for-pytorch
    try:
        return bool(torch.xpu.is_available())
    except AttributeError:
        try:
            import intel_extension_for_pytorch  # noqa: F401
            return bool(torch.xpu.is_available())
        except ImportError:
            return False
        except Exception:
            return False
    except Exception:
        return False


def _check_cpu() -> bool:
    """CPU 始终可用。"""
    return True


# 各后端名称 -> 检测函数的映射
_BACKEND_CHECKERS: Dict[str, Callable[[], bool]] = {
    "loihi": _check_loihi,
    "cuda": _check_cuda,
    "npu": _check_npu,
    "mlu": _check_mlu,
    "xpu": _check_xpu,
    "cpu": _check_cpu,
}


def detect_backend() -> str:
    """
    按优先级检测可用的计算后端，结果会被缓存（只检测一次）。

    优先级（从高到低）：Loihi > CUDA > NPU > MLU > XPU > CPU

    Returns
    -------
    str
        检测到的后端名称，取值之一：
        ``"loihi"``、``"cuda"``、``"npu"``、``"mlu"``、``"xpu"``、``"cpu"``。
    """
    global _detected_backend
    if _detected_backend is not None:
        return _detected_backend

    for backend in _BACKEND_PRIORITY:
        checker = _BACKEND_CHECKERS.get(backend)
        if checker is not None and checker():
            _detected_backend = backend
            return backend

    # 理论上不会走到这里（CPU 始终可用），作兜底保护
    _detected_backend = "cpu"
    return "cpu"


def get_device() -> Any:
    """
    返回检测到的后端对应的 ``torch.device``。

    若 PyTorch 未安装，返回 ``None``。
    Loihi 后端不走 torch 设备抽象，返回 ``torch.device("cpu")``
    便于在 CPU 上准备输入张量（Loihi 自身通过 nxSDK 执行）。

    Returns
    -------
    torch.device or None
        对应设备的 torch.device，或 None（torch 未安装时）。
    """
    if not _TORCH_AVAILABLE:
        return None

    backend = detect_backend()

    if backend == "cpu":
        return torch.device("cpu")

    if backend == "loihi":
        # Loihi 不通过 torch 设备抽象，张量放 CPU 即可
        return torch.device("cpu")

    try:
        return torch.device(backend)
    except (RuntimeError, AssertionError):
        # 后端虽被检测到但当前无法构造 device，降级到 CPU
        return torch.device("cpu")


def get_backend_info() -> str:
    """
    返回人类可读的硬件信息字符串。

    Returns
    -------
    str
        例如 ``"CUDA: NVIDIA GeForce RTX 4090 (24.0 GB)"``、
        ``"NPU (Ascend): Ascend910A"`` 或 ``"CPU"``。
    """
    backend = detect_backend()

    if backend == "loihi":
        try:
            import nxsdk
            version = getattr(nxsdk, "__version__", "unknown")
            return "Loihi (Intel NxSDK {})".format(version)
        except ImportError:
            return "Loihi (Intel NxSDK)"
        except Exception:
            return "Loihi (Intel NxSDK)"

    if not _TORCH_AVAILABLE:
        return "PyTorch not installed - CPU only"

    if backend == "cuda":
        try:
            name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return "CUDA: {} ({:.1f} GB)".format(name, memory)
        except Exception as e:
            return "CUDA (info unavailable: {})".format(e)

    if backend == "npu":
        try:
            import torch_npu  # noqa: F401
            name = torch.npu.get_device_name(0)
            return "NPU (Ascend): {}".format(name)
        except Exception:
            return "NPU (Ascend)"

    if backend == "mlu":
        try:
            import torch_mlu  # noqa: F401
            name = torch.mlu.get_device_name(0)
            return "MLU (Cambricon): {}".format(name)
        except Exception:
            return "MLU (Cambricon)"

    if backend == "xpu":
        try:
            name = torch.xpu.get_device_name(0)
            return "XPU (Intel): {}".format(name)
        except Exception:
            return "XPU (Intel)"

    return "CPU"


def is_backend_available(name: str) -> bool:
    """
    检查特定后端是否可用。

    该方法直接调用对应后端的检测函数，不受 ``detect_backend`` 缓存影响，
    可用于查询非最高优先级后端的可用性。

    Parameters
    ----------
    name : str
        后端名称：``"loihi"``、``"cuda"``、``"npu"``、``"mlu"``、
        ``"xpu"``、``"cpu"``（大小写不敏感）。

    Returns
    -------
    bool
        该后端当前是否可用。
    """
    checker = _BACKEND_CHECKERS.get(name.lower())
    if checker is None:
        return False
    return checker()
