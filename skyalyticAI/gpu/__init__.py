"""
GPU Acceleration Backend

Provides optional GPU acceleration for NIEA modules when PyTorch
is available. Falls back to NumPy when PyTorch is not installed.

This module provides:
1. Device management (auto-select GPU/CPU)
2. Tensor conversion utilities
3. GPU-accelerated operations for core computations
4. Batch processing support for GPU

Usage:
    from skyalyticAI.gpu import get_device, to_tensor, to_numpy, GPUBatchProcessor

    device = get_device()
    processor = GPUBatchProcessor(device)

    # Batch matrix multiply on GPU
    results = processor.batch_matmul(weights_batch, inputs_batch)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_TORCH_AVAILABLE = False
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass


def is_gpu_available() -> bool:
    """Check if PyTorch with CUDA is available."""
    if not _TORCH_AVAILABLE:
        return False
    return torch.cuda.is_available()


def get_device() -> Any:
    """
    Get the best available compute device.

    Returns
    -------
    device
        PyTorch device if available, None otherwise.
    """
    if not _TORCH_AVAILABLE:
        return None
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_tensor(array: np.ndarray, device: Any = None) -> Any:
    """
    Convert NumPy array to PyTorch tensor.

    Parameters
    ----------
    array : np.ndarray
        Input array.
    device : device or None
        Target device.

    Returns
    -------
    torch.Tensor or np.ndarray
        Tensor on the specified device, or original array
        if PyTorch is not available.
    """
    if not _TORCH_AVAILABLE:
        return array
    tensor = torch.from_numpy(array).float()
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def to_numpy(tensor: Any) -> np.ndarray:
    """
    Convert PyTorch tensor to NumPy array.

    Parameters
    ----------
    tensor : torch.Tensor or np.ndarray
        Input tensor.

    Returns
    -------
    np.ndarray
        NumPy array.
    """
    if not _TORCH_AVAILABLE:
        return np.asarray(tensor, dtype=np.float64)
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy().astype(np.float64)
    return np.asarray(tensor, dtype=np.float64)


def get_gpu_info() -> str:
    """Get GPU information string."""
    if not _TORCH_AVAILABLE:
        return "PyTorch not installed - GPU acceleration unavailable"
    if not torch.cuda.is_available():
        return "CUDA not available - using CPU"
    name = torch.cuda.get_device_name(0)
    memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return "GPU: {} ({:.1f} GB)".format(name, memory)


class GPUBatchProcessor:
    """
    GPU-accelerated batch processor for NIEA core operations.

    Provides GPU implementations of the most computationally
    expensive operations in the NIEA pipeline:
    - Batch matrix multiplication (SNN forward, PCN inference)
    - Batch outer product (STDP weight updates)
    - Batch convolution (Visual encoder)
    - Batch FFT (Audio encoder)

    When PyTorch is not available or GPU is not detected,
    all operations fall back to NumPy CPU implementations.

    Parameters
    ----------
    device : torch.device or None
        Target device. If None, auto-selects.
    """

    def __init__(self, device: Any = None) -> None:
        if device is None:
            self.device = get_device()
        else:
            self.device = device

        self._use_gpu = (
            _TORCH_AVAILABLE
            and self.device is not None
            and str(self.device) == "cuda"
        )

    def batch_matmul(
        self, W: np.ndarray, inputs: np.ndarray
    ) -> np.ndarray:
        """
        Batch matrix multiplication: outputs = W @ inputs.

        Parameters
        ----------
        W : np.ndarray
            Weight matrix of shape (out_dim, in_dim).
        inputs : np.ndarray
            Input batch of shape (batch_size, in_dim) or (in_dim,).

        Returns
        -------
        np.ndarray
            Output of shape (batch_size, out_dim) or (out_dim,).
        """
        if self._use_gpu:
            W_t = to_tensor(W, self.device)
            x_t = to_tensor(inputs, self.device)
            if x_t.ndim == 1:
                result = W_t @ x_t
            else:
                result = x_t @ W_t.T
            return to_numpy(result)

        W = np.asarray(W, dtype=np.float64)
        inputs = np.asarray(inputs, dtype=np.float64)
        if inputs.ndim == 1:
            return W @ inputs
        return inputs @ W.T

    def batch_outer_product(
        self, a: np.ndarray, b: np.ndarray
    ) -> np.ndarray:
        """
        Batch outer product: result[i] = outer(a[i], b[i]).

        Parameters
        ----------
        a : np.ndarray
            First batch of shape (batch_size, m).
        b : np.ndarray
            Second batch of shape (batch_size, n).

        Returns
        -------
        np.ndarray
            Outer products of shape (batch_size, m, n).
        """
        if self._use_gpu:
            a_t = to_tensor(a, self.device)
            b_t = to_tensor(b, self.device)
            if a_t.ndim == 1:
                a_t = a_t.unsqueeze(0)
            if b_t.ndim == 1:
                b_t = b_t.unsqueeze(0)
            result = torch.bmm(
                a_t.unsqueeze(2), b_t.unsqueeze(1)
            )
            return to_numpy(result)

        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.ndim == 1:
            return np.outer(a, b)[np.newaxis]
        return np.einsum("bi,bj->bij", a, b)

    def batch_conv2d(
        self,
        images: np.ndarray,
        filters: np.ndarray,
    ) -> np.ndarray:
        """
        Batch 2D convolution using PyTorch conv2d when available.

        Parameters
        ----------
        images : np.ndarray
            Input images of shape (batch, channels, height, width).
        filters : np.ndarray
            Convolutional filters of shape (n_filters, channels, f_h, f_w).

        Returns
        -------
        np.ndarray
            Feature maps of shape (batch, n_filters, out_h, out_w).
        """
        if self._use_gpu:
            images_t = to_tensor(images, self.device)
            filters_t = to_tensor(filters, self.device)
            result = torch.nn.functional.conv2d(images_t, filters_t)
            return to_numpy(result)

        images = np.asarray(images, dtype=np.float64)
        filters = np.asarray(filters, dtype=np.float64)

        batch_size = images.shape[0]
        n_filters, n_ch, f_h, f_w = filters.shape
        _, _, h, w = images.shape
        out_h = h - f_h + 1
        out_w = w - f_w + 1

        output = np.zeros((batch_size, n_filters, out_h, out_w), dtype=np.float64)
        for b in range(batch_size):
            for f in range(n_filters):
                for i in range(out_h):
                    for j in range(out_w):
                        patch = images[b, :, i:i + f_h, j:j + f_w]
                        output[b, f, i, j] = np.sum(patch * filters[f])

        return output

    def batch_fft(
        self, signals: np.ndarray
    ) -> np.ndarray:
        """
        Batch FFT using PyTorch when available.

        Parameters
        ----------
        signals : np.ndarray
            Input signals of shape (batch_size, signal_length).

        Returns
        -------
        np.ndarray
            Complex FFT output of shape (batch_size, signal_length//2 + 1).
        """
        if self._use_gpu:
            signals_t = to_tensor(signals, self.device)
            result = torch.fft.rfft(signals_t, dim=-1)
            return to_numpy(result.real), to_numpy(result.imag)

        signals = np.asarray(signals, dtype=np.float64)
        spectrum = np.fft.rfft(signals, axis=-1)
        return spectrum.real, spectrum.imag

    def batch_softmax(
        self, logits: np.ndarray, temperature: float = 1.0, axis: int = -1
    ) -> np.ndarray:
        """
        Batch softmax with temperature.

        Parameters
        ----------
        logits : np.ndarray
            Input logits of shape (batch_size, n_classes) or (n_classes,).
        temperature : float
            Softmax temperature.
        axis : int
            Axis along which to compute softmax.

        Returns
        -------
        np.ndarray
            Softmax probabilities.
        """
        if self._use_gpu:
            logits_t = to_tensor(logits, self.device)
            result = torch.softmax(logits_t / temperature, dim=axis)
            return to_numpy(result)

        logits = np.asarray(logits, dtype=np.float64)
        scaled = logits / temperature
        shifted = scaled - np.max(scaled, axis=axis, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=axis, keepdims=True)

    def transfer_weights_to_gpu(
        self, weight_dict: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Transfer a dictionary of weight arrays to GPU.

        Parameters
        ----------
        weight_dict : dict
            Dictionary mapping names to numpy arrays.

        Returns
        -------
        dict
            Dictionary mapping names to GPU tensors (or original arrays).
        """
        if not self._use_gpu:
            return weight_dict

        gpu_dict = {}
        for name, array in weight_dict.items():
            if isinstance(array, np.ndarray):
                gpu_dict[name] = to_tensor(array, self.device)
            else:
                gpu_dict[name] = array
        return gpu_dict

    def transfer_weights_from_gpu(
        self, weight_dict: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Transfer a dictionary of weights from GPU back to CPU.

        Parameters
        ----------
        weight_dict : dict
            Dictionary mapping names to tensors or arrays.

        Returns
        -------
        dict
            Dictionary mapping names to numpy arrays.
        """
        cpu_dict = {}
        for name, value in weight_dict.items():
            cpu_dict[name] = to_numpy(value) if _TORCH_AVAILABLE and isinstance(value, torch.Tensor) else np.asarray(value, dtype=np.float64)
        return cpu_dict

    def __repr__(self) -> str:
        device_str = str(self.device) if self.device else "cpu"
        gpu_status = "GPU" if self._use_gpu else "CPU"
        return "GPUBatchProcessor(device={}, backend={})".format(device_str, gpu_status)
