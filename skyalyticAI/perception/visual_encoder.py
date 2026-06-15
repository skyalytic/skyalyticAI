"""
Visual Encoder - Converts raw image data into neural representations.

Implements a convolutional feature extraction pipeline that converts
2D image arrays into 1D feature vectors suitable for the NIEABrain's
spiking neural network input.

Pipeline:
    1. Normalize pixel values to [0, 1]
    2. Apply learnable convolutional filters (Gabor-like edge detection)
    3. Apply ReLU activation
    4. Spatial pooling (average pooling)
    5. Flatten to 1D feature vector
    6. Project to target output dimension

The convolutional filters are initialized with Gabor-like patterns
for edge detection, which is biologically motivated (simple cells
in V1 cortex respond to oriented edges).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


class VisualEncoder:
    """
    Visual encoder that converts images to neural feature vectors.

    Parameters
    ----------
    image_height : int
        Height of input images in pixels.
    image_width : int
        Width of input images in pixels.
    n_channels : int
        Number of color channels (1 for grayscale, 3 for RGB).
    n_filters : int
        Number of convolutional filters (orientations).
    filter_size : int
        Size of each convolutional filter (must be odd).
    pool_size : int
        Spatial pooling window size.
    output_dim : int
        Dimension of the output feature vector.
    learning_rate : float
        Learning rate for filter adaptation.
    """

    def __init__(
        self,
        image_height: int = 28,
        image_width: int = 28,
        n_channels: int = 1,
        n_filters: int = 8,
        filter_size: int = 5,
        pool_size: int = 2,
        output_dim: int = 64,
        learning_rate: float = 0.001,
    ) -> None:
        if image_height <= 0 or image_width <= 0:
            raise ValueError("Image dimensions must be positive")
        if n_filters <= 0:
            raise ValueError("n_filters must be positive")
        if filter_size <= 0 or filter_size % 2 == 0:
            raise ValueError("filter_size must be positive and odd")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")

        self.image_height = image_height
        self.image_width = image_width
        self.n_channels = n_channels
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        self.filters = self._init_gabor_filters(n_filters, n_channels, filter_size)

        conv_h = image_height - filter_size + 1
        conv_w = image_width - filter_size + 1
        pool_h = conv_h // pool_size
        pool_w = conv_w // pool_size
        self.flat_dim = n_filters * pool_h * pool_w

        scale = np.sqrt(2.0 / (self.flat_dim + output_dim))
        self.projection_W = np.random.randn(output_dim, self.flat_dim) * scale
        self.projection_b = np.zeros(output_dim, dtype=np.float64)
        self._last_flat = np.zeros(self.flat_dim, dtype=np.float64)

    def _init_gabor_filters(
        self, n_filters: int, n_channels: int, size: int
    ) -> np.ndarray:
        """
        Initialize convolutional filters with Gabor-like patterns.

        Gabor filters model the orientation-selective simple cells
        found in the primary visual cortex (V1). Each filter responds
        to edges at a specific orientation.

        Parameters
        ----------
        n_filters : int
            Number of filters.
        n_channels : int
            Number of input channels.
        size : int
            Filter spatial size.

        Returns
        -------
        np.ndarray
            Filter bank of shape (n_filters, n_channels, size, size).
        """
        filters = np.zeros((n_filters, n_channels, size, size), dtype=np.float64)
        half = size // 2

        for i in range(n_filters):
            theta = i * np.pi / n_filters
            sigma = max(size / 4.0, 1.0)
            lambd = max(size / 2.0, 1.0)
            psi = 0.0

            for y in range(size):
                for x in range(size):
                    x_rot = (x - half) * np.cos(theta) + (y - half) * np.sin(theta)
                    y_rot = -(x - half) * np.sin(theta) + (y - half) * np.cos(theta)
                    gaussian = np.exp(-(x_rot ** 2 + y_rot ** 2) / (2 * sigma ** 2))
                    sinusoid = np.cos(2 * np.pi * x_rot / lambd + psi)
                    filters[i, :, y, x] = gaussian * sinusoid

            filter_norm = np.linalg.norm(filters[i])
            if filter_norm > 1e-10:
                filters[i] /= filter_norm

        return filters

    def _convolve(
        self, image: np.ndarray, filter_bank: np.ndarray
    ) -> np.ndarray:
        """
        Apply 2D convolution with multiple filters.

        Parameters
        ----------
        image : np.ndarray
            Input image of shape (n_channels, height, width).
        filter_bank : np.ndarray
            Filters of shape (n_filters, n_channels, f_height, f_width).

        Returns
        -------
        np.ndarray
            Feature maps of shape (n_filters, out_height, out_width).
        """
        n_filters, n_ch, f_h, f_w = filter_bank.shape
        _, h, w = image.shape
        out_h = h - f_h + 1
        out_w = w - f_w + 1

        output = np.zeros((n_filters, out_h, out_w), dtype=np.float64)
        for f in range(n_filters):
            for i in range(out_h):
                for j in range(out_w):
                    patch = image[:, i:i + f_h, j:j + f_w]
                    output[f, i, j] = np.sum(patch * filter_bank[f])

        return output

    def _pool(self, feature_maps: np.ndarray) -> np.ndarray:
        """
        Apply average pooling to feature maps.

        Parameters
        ----------
        feature_maps : np.ndarray
            Feature maps of shape (n_filters, height, width).

        Returns
        -------
        np.ndarray
            Pooled feature maps of shape (n_filters, height//pool_size, width//pool_size).
        """
        n_filters, h, w = feature_maps.shape
        pool_h = h // self.pool_size
        pool_w = w // self.pool_size

        output = np.zeros((n_filters, pool_h, pool_w), dtype=np.float64)
        for f in range(n_filters):
            for i in range(pool_h):
                for j in range(pool_w):
                    region = feature_maps[
                        f,
                        i * self.pool_size:(i + 1) * self.pool_size,
                        j * self.pool_size:(j + 1) * self.pool_size,
                    ]
                    output[f, i, j] = np.mean(region)

        return output

    def encode(self, image: np.ndarray) -> np.ndarray:
        """
        Encode an image into a feature vector.

        Parameters
        ----------
        image : np.ndarray
            Input image. Can be:
            - 2D array (height, width) for grayscale
            - 3D array (height, width, channels) for color
            Pixel values should be in [0, 255] or [0, 1].

        Returns
        -------
        np.ndarray
            Feature vector of shape (output_dim,).
        """
        image = np.asarray(image, dtype=np.float64)

        if image.ndim == 2:
            image = image[np.newaxis, :, :]
        elif image.ndim == 3:
            if image.shape[2] == self.n_channels:
                image = image.transpose(2, 0, 1)
            elif image.shape[0] != self.n_channels:
                raise ValueError(
                    f"Image channel dimension mismatch: expected {self.n_channels}, "
                    f"got shape {image.shape}"
                )

        if image.shape[1] != self.image_height or image.shape[2] != self.image_width:
            raise ValueError(
                f"Image size mismatch: expected ({self.image_height}, {self.image_width}), "
                f"got ({image.shape[1]}, {image.shape[2]})"
            )

        if image.max() > 1.0:
            image = image / 255.0
        image = np.clip(image, 0.0, 1.0)

        feature_maps = self._convolve(image, self.filters)
        feature_maps = np.maximum(0, feature_maps)
        pooled = self._pool(feature_maps)
        flat = pooled.flatten()

        if flat.shape[0] != self.flat_dim:
            padded = np.zeros(self.flat_dim, dtype=np.float64)
            n = min(flat.shape[0], self.flat_dim)
            padded[:n] = flat[:n]
            flat = padded

        output = self.projection_W @ flat + self.projection_b
        self._last_flat = flat
        return output

    def adapt(
        self,
        image: np.ndarray,
        target: Optional[np.ndarray] = None,
        reconstruction: Optional[np.ndarray] = None,
    ) -> float:
        """
        Adapt the projection layer based on reconstruction error.

        When a target or reconstruction is provided, updates the
        projection weights to minimize the error. This enables
        the encoder to learn better representations over time.

        Parameters
        ----------
        image : np.ndarray
            Input image.
        target : np.ndarray or None
            Target feature vector. If None, uses self-supervised
            reconstruction from the brain's prediction.
        reconstruction : np.ndarray or None
            Reconstructed feature vector from the brain.

        Returns
        -------
        float
            Reconstruction error magnitude.
        """
        features = self.encode(image)

        if target is not None:
            error = features - target
            d_W = np.outer(error, self._last_flat)
            self.projection_W -= self.learning_rate * d_W
            self.projection_b -= self.learning_rate * error
            return float(np.linalg.norm(error))

        if reconstruction is not None:
            error = features - reconstruction
            d_W = np.outer(error, self._last_flat)
            self.projection_W -= self.learning_rate * d_W
            self.projection_b -= self.learning_rate * error
            return float(np.linalg.norm(error))

        return 0.0

    def state_dict(self) -> Dict[str, Any]:
        """Return encoder state for serialization."""
        return {
            "filters": self.filters.copy(),
            "projection_W": self.projection_W.copy(),
            "projection_b": self.projection_b.copy(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load encoder state from a dictionary."""
        self.filters = state["filters"].copy()
        self.projection_W = state["projection_W"].copy()
        self.projection_b = state["projection_b"].copy()

    def __repr__(self) -> str:
        return (
            f"VisualEncoder(image=({self.image_height},{self.image_width}), "
            f"channels={self.n_channels}, filters={self.n_filters}, "
            f"output_dim={self.output_dim})"
        )
