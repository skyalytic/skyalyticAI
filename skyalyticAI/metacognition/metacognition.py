"""
Metacognitive Module - Self-Awareness and Uncertainty Estimation

Implements a metacognitive system that monitors and regulates
the learning process. The module provides:

1. Confidence estimation: evaluates how certain the system is
   about its predictions
2. Learning rate adaptation: adjusts learning speed based on
   uncertainty and performance
3. Knowledge boundary assessment: determines what the system
   knows vs. doesn't know
4. Attention allocation: suggests which inputs deserve more
   processing resources
5. Calibration: improves the accuracy of self-assessment over time

The metacognitive module uses a neural network that learns to
predict its own performance, enabling the system to:
- Know what it knows (high confidence + high accuracy)
- Know what it doesn't know (low confidence + uncertain)
- Adaptively allocate computational resources
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np


class MetacognitiveModule:
    """
    Metacognitive Module for self-awareness and adaptive learning.

    Uses a two-layer neural network to predict confidence and
    learning rate adjustment based on the current state of
    the underlying learning system. The module is trained
    online using the actual prediction outcomes as supervision.

    Parameters
    ----------
    input_dim : int
        Dimension of the input state vector (from lower-level
        modules such as prediction errors, belief entropy, etc.).
    hidden_dim : int
        Number of hidden units in the metacognitive network.
    memory_size : int
        Capacity of the experience replay buffer for meta-learning.
    confidence_lr : float
        Learning rate for confidence prediction network.
    calibration_lr : float
        Learning rate for the calibration adjustment.
    grad_clip_value : float
        Maximum gradient norm for gradient clipping.
    initial_confidence : float
        Initial confidence value before any training.
    confidence_smoothing : float
        Exponential smoothing factor for confidence estimates.
        Must be in (0, 1). Higher values give more weight to
        recent observations.
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 32,
        memory_size: int = 500,
        confidence_lr: float = 0.01,
        calibration_lr: float = 0.005,
        grad_clip_value: float = 1.0,
        initial_confidence: float = 0.5,
        confidence_smoothing: float = 0.9,
    ) -> None:
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if memory_size <= 0:
            raise ValueError(f"memory_size must be positive, got {memory_size}")
        if not 0 < confidence_smoothing < 1:
            raise ValueError(
                f"confidence_smoothing must be in (0, 1), "
                f"got {confidence_smoothing}"
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.confidence_lr = confidence_lr
        self.calibration_lr = calibration_lr
        self.grad_clip_value = grad_clip_value
        self.confidence_smoothing = confidence_smoothing

        scale_w1 = np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W1 = np.random.randn(hidden_dim, input_dim) * scale_w1
        self.b1 = np.zeros(hidden_dim, dtype=np.float64)

        scale_w2 = np.sqrt(2.0 / (hidden_dim + 3))
        self.W2 = np.random.randn(3, hidden_dim) * scale_w2
        self.b2 = np.zeros(3, dtype=np.float64)

        self.memory: deque = deque(maxlen=memory_size)

        self._smoothed_confidence: float = initial_confidence
        self._smoothed_lr_factor: float = 1.0

        self.confidence_history: list = []
        self.actual_error_history: list = []
        self.calibration_history: list = []
        self.calibration_score: float = 0.5

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function."""
        x = np.clip(x, -500, 500)
        pos = x >= 0
        result = np.zeros_like(x)
        result[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        exp_x = np.exp(x[~pos])
        result[~pos] = exp_x / (1.0 + exp_x)
        return result

    @staticmethod
    def _sigmoid_deriv(s: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid given sigmoid output."""
        return s * (1.0 - s)

    @staticmethod
    def _softplus(x: np.ndarray) -> np.ndarray:
        """Numerically stable softplus: log(1 + exp(x))."""
        return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -500, 20))))

    def _clip_gradient(self, grad: np.ndarray) -> np.ndarray:
        """Clip gradient by norm."""
        if self.grad_clip_value <= 0:
            return grad
        norm = np.linalg.norm(grad)
        if norm > self.grad_clip_value:
            grad = grad * (self.grad_clip_value / norm)
        return grad

    def forward(self, state_vector: np.ndarray) -> Dict[str, float]:
        """
        Forward pass: estimate confidence and learning rate factor.

        Parameters
        ----------
        state_vector : np.ndarray
            Input state from lower-level modules, shape (input_dim,).
            Typically includes prediction errors, belief entropy,
            recent reward statistics, etc.

        Returns
        -------
        dict
            Dictionary containing:
            - 'confidence': estimated confidence (0-1)
            - 'lr_factor': learning rate adjustment factor (>0)
            - 'exploration_value': exploration tendency (0-1)
        """
        state_vector = np.asarray(state_vector, dtype=np.float64)
        if state_vector.shape != (self.input_dim,):
            raise ValueError(
                f"state_vector shape must be ({self.input_dim},), "
                f"got {state_vector.shape}"
            )

        h_pre = self.W1 @ state_vector + self.b1
        h = np.tanh(h_pre)

        output_pre = self.W2 @ h + self.b2

        confidence = float(self._sigmoid(output_pre[0:1])[0])
        lr_factor = float(self._softplus(output_pre[1:2])[0])
        exploration = float(self._sigmoid(output_pre[2:3])[0])

        self._smoothed_confidence = (
            self.confidence_smoothing * self._smoothed_confidence
            + (1 - self.confidence_smoothing) * confidence
        )
        self._smoothed_lr_factor = (
            self.confidence_smoothing * self._smoothed_lr_factor
            + (1 - self.confidence_smoothing) * lr_factor
        )

        return {
            "confidence": confidence,
            "lr_factor": lr_factor,
            "exploration_value": exploration,
            "smoothed_confidence": self._smoothed_confidence,
            "smoothed_lr_factor": self._smoothed_lr_factor,
        }

    def evaluate_knowledge_boundary(
        self,
        state_vector: np.ndarray,
        task_difficulty: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Evaluate the knowledge boundary: what is known vs. unknown.

        Parameters
        ----------
        state_vector : np.ndarray
            Current state from lower-level modules.
        task_difficulty : float
            Estimated task difficulty in [0, 1]. Higher difficulty
            raises the confidence threshold for claiming knowledge.

        Returns
        -------
        dict
            Dictionary containing:
            - 'knows': whether the system believes it knows the answer
            - 'should_explore': whether the system should explore
            - 'confidence': confidence level
            - 'exploration_value': exploration tendency
            - 'effective_threshold': the confidence threshold used
        """
        if not 0 <= task_difficulty <= 1:
            raise ValueError(
                f"task_difficulty must be in [0, 1], got {task_difficulty}"
            )

        result = self.forward(state_vector)
        confidence = result["confidence"]
        exploration = result["exploration_value"]

        effective_threshold = 0.5 + 0.3 * task_difficulty

        knows = confidence > effective_threshold
        should_explore = confidence < 0.7 or exploration > 0.5

        return {
            "knows": knows,
            "should_explore": should_explore,
            "confidence": confidence,
            "exploration_value": exploration,
            "effective_threshold": effective_threshold,
        }

    def update_meta_knowledge(
        self,
        state_vector: np.ndarray,
        predicted_confidence: float,
        actual_outcome: float,
        learning_outcome: float,
    ) -> Dict[str, float]:
        """
        Update metacognitive knowledge based on actual performance.

        This is the core learning mechanism: the module compares
        its predicted confidence with the actual outcome and
        adjusts its parameters to improve calibration.

        Parameters
        ----------
        state_vector : np.ndarray
            State at the time of prediction.
        predicted_confidence : float
            Confidence that was predicted.
        actual_outcome : float
            Actual performance (0 = incorrect, 1 = correct,
            or continuous in [0, 1]).
        learning_outcome : float
            How much the underlying system improved from this
            experience. Positive = improvement, negative = degradation.

        Returns
        -------
        dict
            Dictionary containing:
            - 'calibration_error': difference between predicted
              confidence and actual outcome
            - 'calibration_score': current calibration quality
        """
        state_vector = np.asarray(state_vector, dtype=np.float64)
        actual_outcome = float(np.clip(actual_outcome, 0, 1))

        self.memory.append({
            "state": state_vector.copy(),
            "pred_conf": predicted_confidence,
            "actual": actual_outcome,
            "learning": learning_outcome,
        })

        calibration_error = predicted_confidence - actual_outcome
        self.calibration_history.append(calibration_error)
        self.confidence_history.append(predicted_confidence)
        self.actual_error_history.append(1.0 - actual_outcome)

        if len(self.memory) >= 10:
            self._meta_learning_step()

        if len(self.calibration_history) > 0:
            recent = self.calibration_history[-min(50, len(self.calibration_history)):]
            avg_calibration = np.mean(np.abs(recent))
            self.calibration_score = 1.0 / (1.0 + avg_calibration)

        return {
            "calibration_error": calibration_error,
            "calibration_score": self.calibration_score,
        }

    def _meta_learning_step(self) -> None:
        """
        Perform one meta-learning update using recent experiences.

        Uses analytical backpropagation through the metacognitive
        network to improve confidence prediction accuracy.
        """
        batch_size = min(10, len(self.memory))
        batch = list(self.memory)[-batch_size:]

        dW1_accum = np.zeros_like(self.W1)
        db1_accum = np.zeros_like(self.b1)
        dW2_accum = np.zeros_like(self.W2)
        db2_accum = np.zeros_like(self.b2)

        for exp in batch:
            state = exp["state"]
            actual = exp["actual"]

            h_pre = self.W1 @ state + self.b1
            h = np.tanh(h_pre)
            output_pre = self.W2 @ h + self.b2

            conf = float(self._sigmoid(output_pre[0:1])[0])

            d_conf = conf - actual
            d_conf_sigmoid = d_conf * self._sigmoid_deriv(
                self._sigmoid(output_pre[0:1])
            )[0]

            d_output = np.zeros(3, dtype=np.float64)
            d_output[0] = d_conf_sigmoid
            # lr_factor gradient: encourage higher lr when prediction error is high
            softplus_deriv = float(self._sigmoid(output_pre[1:2])[0])
            d_output[1] = d_conf_sigmoid * 0.5 * (1.0 - conf) * softplus_deriv
            # exploration gradient: encourage exploration when surprise is high
            exploration_val = float(self._sigmoid(output_pre[2:3])[0])
            exploration_deriv = exploration_val * (1.0 - exploration_val)
            d_output[2] = d_conf_sigmoid * 0.3 * (1.0 - conf) * exploration_deriv

            dW2_accum += np.outer(d_output, h)
            db2_accum += d_output

            dh = self.W2.T @ d_output
            dh_pre = dh * (1 - h ** 2)

            dW1_accum += np.outer(dh_pre, state)
            db1_accum += dh_pre

        n = len(batch)
        self.W1 -= self.confidence_lr * self._clip_gradient(dW1_accum / n)
        self.b1 -= self.confidence_lr * self._clip_gradient(db1_accum / n)
        self.W2 -= self.confidence_lr * self._clip_gradient(dW2_accum / n)
        self.b2 -= self.confidence_lr * self._clip_gradient(db2_accum / n)

    def suggest_attention(self, input_features: np.ndarray) -> np.ndarray:
        """
        Suggest attention allocation across input features.

        Features with higher uncertainty (lower confidence) receive
        more attention, encouraging the system to focus on what
        it doesn't know.

        Parameters
        ----------
        input_features : np.ndarray
            Feature values from the input, shape (input_dim,).

        Returns
        -------
        np.ndarray
            Attention weights, shape (input_dim,). Sums to 1.
        """
        input_features = np.asarray(input_features, dtype=np.float64)
        if input_features.shape != (self.input_dim,):
            raise ValueError(
                f"input_features shape must be ({self.input_dim},), "
                f"got {input_features.shape}"
            )

        uncertainty = 1.0 - np.abs(input_features) / (
            np.max(np.abs(input_features)) + 1e-10
        )
        attention = np.exp(uncertainty)
        attention /= attention.sum()

        return attention

    def get_metacognitive_state(self) -> Dict[str, Any]:
        """
        Get a summary of the current metacognitive state.

        Returns
        -------
        dict
            Dictionary containing calibration quality, memory size,
            and recent confidence statistics.
        """
        recent_conf = (
            np.mean(self.confidence_history[-20:])
            if self.confidence_history
            else self._smoothed_confidence
        )
        recent_error = (
            np.mean(self.actual_error_history[-20:])
            if self.actual_error_history
            else 0.5
        )

        return {
            "calibration_score": self.calibration_score,
            "memory_size": len(self.memory),
            "recent_confidence": recent_conf,
            "recent_error_rate": recent_error,
            "smoothed_confidence": self._smoothed_confidence,
            "smoothed_lr_factor": self._smoothed_lr_factor,
        }

    def reset(self) -> None:
        """Reset the metacognitive module."""
        self.memory.clear()
        self.confidence_history = []
        self.actual_error_history = []
        self.calibration_history = []
        self.calibration_score = 0.5
        self._smoothed_confidence = 0.5
        self._smoothed_lr_factor = 1.0

    def state_dict(self) -> Dict[str, Any]:
        """Return the module state for serialization."""
        return {
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2": self.W2.copy(),
            "b2": self.b2.copy(),
            "calibration_score": self.calibration_score,
            "smoothed_confidence": self._smoothed_confidence,
            "smoothed_lr_factor": self._smoothed_lr_factor,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load module state from a dictionary."""
        self.W1 = state["W1"].copy()
        self.b1 = state["b1"].copy()
        self.W2 = state["W2"].copy()
        self.b2 = state["b2"].copy()
        self.calibration_score = state["calibration_score"]
        self._smoothed_confidence = state["smoothed_confidence"]
        self._smoothed_lr_factor = state["smoothed_lr_factor"]

    def __repr__(self) -> str:
        return (
            f"MetacognitiveModule(input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"calibration={self.calibration_score:.3f})"
        )
