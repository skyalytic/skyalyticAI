"""
Audio Encoder - Converts raw audio signals into neural representations.

Implements a cochlear-inspired feature extraction pipeline that converts
1D audio waveforms into spectral feature vectors suitable for the
NIEABrain's spiking neural network input.

Pipeline:
    1. Pre-emphasis (high-frequency boost)
    2. Framing and windowing (Hamming window)
    3. FFT-based power spectrum computation
    4. Mel filterbank application (cochlear modeling)
    5. Log compression
    6. Projection to target output dimension

The Mel filterbank models the frequency selectivity of the cochlea:
lower frequencies are resolved with higher resolution, matching
human auditory perception.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class AudioEncoder:
    """
    Audio encoder that converts sound waveforms to feature vectors.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate in Hz.
    n_mels : int
        Number of Mel filterbank channels.
    fft_size : int
        FFT window size.
    hop_length : int
        Number of samples between successive frames.
    output_dim : int
        Dimension of the output feature vector.
    learning_rate : float
        Learning rate for projection adaptation.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 26,
        fft_size: int = 512,
        hop_length: int = 160,
        output_dim: int = 64,
        learning_rate: float = 0.001,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if n_mels <= 0:
            raise ValueError(f"n_mels must be positive, got {n_mels}")
        if fft_size <= 0:
            raise ValueError(f"fft_size must be positive, got {fft_size}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        self.mel_filterbank = self._create_mel_filterbank(
            sample_rate, fft_size, n_mels
        )

        scale = np.sqrt(2.0 / (n_mels + output_dim))
        self.projection_W = np.random.randn(output_dim, n_mels) * scale
        self.projection_b = np.zeros(output_dim, dtype=np.float64)

    def _create_mel_filterbank(
        self, sample_rate: int, fft_size: int, n_mels: int
    ) -> np.ndarray:
        """
        Create a Mel-spaced filterbank matrix.

        The Mel scale models human pitch perception: equal distances
        on the Mel scale sound equally distant to humans. Lower
        frequencies have finer resolution.

        Parameters
        ----------
        sample_rate : int
            Audio sample rate.
        fft_size : int
            FFT size.
        n_mels : int
            Number of Mel filters.

        Returns
        -------
        np.ndarray
            Mel filterbank of shape (n_mels, fft_size // 2 + 1).
        """
        low_freq = 0
        high_freq = sample_rate / 2.0
        low_mel = self._hz_to_mel(low_freq)
        high_mel = self._hz_to_mel(high_freq)

        mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)

        bin_points = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)

        n_fft_bins = fft_size // 2 + 1
        filterbank = np.zeros((n_mels, n_fft_bins), dtype=np.float64)

        for i in range(n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]

            for j in range(left, center):
                if j < n_fft_bins and center > left:
                    filterbank[i, j] = (j - left) / (center - left)

            for j in range(center, right):
                if j < n_fft_bins and right > center:
                    filterbank[i, j] = (right - j) / (right - center)

        return filterbank

    @staticmethod
    def _hz_to_mel(hz: float) -> float:
        """Convert frequency in Hz to Mel scale."""
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel: float) -> float:
        """Convert Mel scale to frequency in Hz."""
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def encode(self, audio: np.ndarray) -> np.ndarray:
        """
        Encode an audio waveform into a feature vector.

        Parameters
        ----------
        audio : np.ndarray
            1D audio waveform array. Values should be in [-1, 1]
            for normalized audio or any range for raw PCM.

        Returns
        -------
        np.ndarray
            Feature vector of shape (output_dim,).
        """
        audio = np.asarray(audio, dtype=np.float64).flatten()

        if len(audio) < self.fft_size:
            audio = np.pad(audio, (0, self.fft_size - len(audio)))

        pre_emphasis = 0.97
        emphasized = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

        n_frames = 1 + (len(emphasized) - self.fft_size) // self.hop_length
        n_frames = max(n_frames, 1)

        window = np.hamming(self.fft_size)

        mel_features = np.zeros((n_frames, self.n_mels), dtype=np.float64)

        for i in range(n_frames):
            start = i * self.hop_length
            frame = emphasized[start:start + self.fft_size]

            if len(frame) < self.fft_size:
                frame = np.pad(frame, (0, self.fft_size - len(frame)))

            frame = frame * window

            spectrum = np.fft.rfft(frame, n=self.fft_size)
            power_spectrum = np.abs(spectrum) ** 2 / self.fft_size

            mel_spectrum = self.mel_filterbank @ power_spectrum
            mel_spectrum = np.maximum(mel_spectrum, 1e-10)
            mel_features[i] = np.log(mel_spectrum)

        avg_mel = np.mean(mel_features, axis=0)

        self._last_input = avg_mel
        output = self.projection_W @ avg_mel + self.projection_b
        return output

    def adapt(
        self,
        audio: np.ndarray,
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
        audio : np.ndarray
            Input audio waveform.
        target : np.ndarray or None
            Target feature vector.
        reconstruction : np.ndarray or None
            Reconstructed feature vector from the brain.

        Returns
        -------
        float
            Reconstruction error magnitude.
        """
        features = self.encode(audio)

        if target is not None:
            error = features - target
            d_W = np.outer(error, self._last_input)
            self.projection_W -= self.learning_rate * d_W
            self.projection_b -= self.learning_rate * error
            return float(np.linalg.norm(error))

        return 0.0

    def state_dict(self) -> Dict[str, Any]:
        """Return encoder state for serialization."""
        return {
            "mel_filterbank": self.mel_filterbank.copy(),
            "projection_W": self.projection_W.copy(),
            "projection_b": self.projection_b.copy(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load encoder state from a dictionary."""
        self.mel_filterbank = state["mel_filterbank"].copy()
        self.projection_W = state["projection_W"].copy()
        self.projection_b = state["projection_b"].copy()

    def __repr__(self) -> str:
        return (
            f"AudioEncoder(sample_rate={self.sample_rate}, "
            f"n_mels={self.n_mels}, output_dim={self.output_dim})"
        )
