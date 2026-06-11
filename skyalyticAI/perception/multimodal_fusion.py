"""
Multimodal Fusion - Combines features from multiple sensory modalities.

Implements attention-based fusion that learns to weight different
sensory inputs based on their reliability and relevance. This models
the brain's ability to integrate visual, auditory, and other sensory
information into a unified percept.

The fusion mechanism:
1. Projects each modality's features to a common dimension
2. Computes attention weights based on feature reliability
3. Produces a weighted combination as the fused representation
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class MultimodalFusion:
    """
    Multimodal fusion module for combining multiple sensory inputs.

    Parameters
    ----------
    modality_dims : dict
        Mapping from modality name to feature dimension.
        E.g., {"visual": 64, "audio": 64}
    output_dim : int
        Dimension of the fused output vector.
    learning_rate : float
        Learning rate for attention weight adaptation.
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        output_dim: int = 64,
        learning_rate: float = 0.01,
    ) -> None:
        if not modality_dims:
            raise ValueError("modality_dims must not be empty")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")

        self.modality_dims = dict(modality_dims)
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.modalities = list(modality_dims.keys())

        self.projections: Dict[str, np.ndarray] = {}
        self.biases: Dict[str, np.ndarray] = {}
        for name, dim in modality_dims.items():
            scale = np.sqrt(2.0 / (dim + output_dim))
            self.projections[name] = np.random.randn(output_dim, dim) * scale
            self.biases[name] = np.zeros(output_dim, dtype=np.float64)

        self.attention_W = np.random.randn(len(self.modalities)) * 0.1

    def fuse(
        self, features: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Fuse features from multiple modalities.

        Parameters
        ----------
        features : dict
            Mapping from modality name to feature vector.
            E.g., {"visual": np.array([...]), "audio": np.array([...])}

        Returns
        -------
        np.ndarray
            Fused feature vector of shape (output_dim,).
        """
        projected = {}
        for name in self.modalities:
            if name in features:
                feat = np.asarray(features[name], dtype=np.float64)
                expected_dim = self.modality_dims[name]
                if feat.shape[0] != expected_dim:
                    if feat.shape[0] < expected_dim:
                        padded = np.zeros(expected_dim, dtype=np.float64)
                        padded[:feat.shape[0]] = feat
                        feat = padded
                    else:
                        feat = feat[:expected_dim]
                projected[name] = self.projections[name] @ feat + self.biases[name]

        if not projected:
            return np.zeros(self.output_dim, dtype=np.float64)

        raw_weights = np.array([
            float(self.attention_W[i])
            for i, name in enumerate(self.modalities) if name in projected
        ])
        weights = np.exp(raw_weights - np.max(raw_weights))
        weights /= weights.sum()

        result = np.zeros(self.output_dim, dtype=np.float64)
        for idx, (name, proj) in enumerate(projected.items()):
            result += weights[idx] * proj

        return result

    def update_attention(
        self,
        features: Dict[str, np.ndarray],
        prediction_error: float,
    ) -> None:
        """
        Update attention weights based on prediction error.

        Modalities that contribute to lower prediction error
        receive higher attention weights over time.

        Parameters
        ----------
        features : dict
            Current modality features.
        prediction_error : float
            Current prediction error magnitude.
        """
        for i, name in enumerate(self.modalities):
            if name in features:
                feat = np.asarray(features[name], dtype=np.float64)
                reliability = 1.0 / (1.0 + np.var(feat) + 1e-10)
                self.attention_W[i] += self.learning_rate * reliability * (1.0 - prediction_error)
        self.attention_W = np.clip(self.attention_W, -5.0, 5.0)

    def state_dict(self) -> Dict[str, Any]:
        """Return fusion state for serialization."""
        return {
            "projections": {k: v.copy() for k, v in self.projections.items()},
            "biases": {k: v.copy() for k, v in self.biases.items()},
            "attention_W": self.attention_W.copy(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load fusion state from a dictionary."""
        self.projections = {k: v.copy() for k, v in state["projections"].items()}
        self.biases = {k: v.copy() for k, v in state["biases"].items()}
        self.attention_W = state["attention_W"].copy()

    def __repr__(self) -> str:
        return (
            f"MultimodalFusion(modalities={self.modalities}, "
            f"output_dim={self.output_dim})"
        )
