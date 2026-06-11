"""
Complementary Memory System - Hippocampal-Cortical Dual Architecture

Implements the complementary memory system with two interacting stores:
1. Hippocampal Store (fast learning, limited capacity):
   - Rapid encoding of new experiences
   - Pattern separation (orthogonalization)
   - Short-term retention
   - Drives memory consolidation

2. Cortical Store (slow learning, large capacity):
   - Gradual integration of consolidated memories
   - Pattern completion (generalization)
   - Long-term retention
   - Semantic knowledge extraction

The consolidation process transfers memories from hippocampal to cortical
store during "sleep" periods, mimicking the biological memory consolidation
process. This implements the third theoretical pillar of NIEA.

This module complements the existing HDC memory system by providing
the dual-store architecture that HDC alone lacks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class HippocampalStore:
    """
    Fast-learning, limited-capacity memory store (hippocampal analog).

    Features:
    - Rapid one-shot encoding of experiences
    - Pattern separation via random projection
    - Temporary storage with decay
    - Drives consolidation to cortical store

    Parameters
    ----------
    dim : int
        Dimensionality of memory vectors.
    capacity : int
        Maximum number of stored memories.
    decay_rate : float
        Rate at which memory strength decays per access.
    separation_strength : float
        Strength of pattern separation (random projection).
    """

    def __init__(
        self,
        dim: int = 256,
        capacity: int = 500,
        decay_rate: float = 0.001,
        separation_strength: float = 0.5,
        rehearsal_rate: float = 0.01,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.dim = dim
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.separation_strength = separation_strength
        self.rehearsal_rate = rehearsal_rate

        self._memories: List[np.ndarray] = []
        self._keys: List[np.ndarray] = []
        self._strengths: List[float] = []
        self._ages: List[int] = []
        self._consolidated: List[bool] = []

        self._separation_matrix = np.random.randn(dim, dim) * np.sqrt(
            1.0 / dim
        )

        self._ptr: int = 0
        self._size: int = 0

    def _pattern_separation(self, vector: np.ndarray) -> np.ndarray:
        """
        Apply pattern separation to reduce interference.

        Uses random projection followed by sparsification to
        make similar inputs more distinct.

        Parameters
        ----------
        vector : np.ndarray
            Input vector, shape (dim,).

        Returns
        -------
        np.ndarray
            Separated vector, shape (dim,).
        """
        separated = (
            1.0 - self.separation_strength
        ) * vector + self.separation_strength * (
            self._separation_matrix @ vector
        )
        return separated / (np.linalg.norm(separated) + 1e-10)

    def encode(
        self,
        key: np.ndarray,
        value: np.ndarray,
    ) -> int:
        """
        Rapidly encode a new memory.

        Parameters
        ----------
        key : np.ndarray
            Memory key (e.g., state vector), shape (dim,).
        value : np.ndarray
            Memory value (e.g., next_state + reward), shape (dim,).

        Returns
        -------
        int
            Memory index.
        """
        key = np.asarray(key, dtype=np.float64).flatten()
        value = np.asarray(value, dtype=np.float64).flatten()

        if key.shape[0] < self.dim:
            padded = np.zeros(self.dim, dtype=np.float64)
            padded[:key.shape[0]] = key
            key = padded
        elif key.shape[0] > self.dim:
            key = key[:self.dim]

        if value.shape[0] < self.dim:
            padded = np.zeros(self.dim, dtype=np.float64)
            padded[:value.shape[0]] = value
            value = padded
        elif value.shape[0] > self.dim:
            value = value[:self.dim]

        separated_key = self._pattern_separation(key)

        if self._size < self.capacity:
            self._keys.append(separated_key.copy())
            self._memories.append(value.copy())
            self._strengths.append(1.0)
            self._ages.append(0)
            self._consolidated.append(False)
            idx = self._size
            self._size += 1
        else:
            idx = int(np.argmin(self._strengths))
            self._keys[idx] = separated_key.copy()
            self._memories[idx] = value.copy()
            self._strengths[idx] = 1.0
            self._ages[idx] = 0
            self._consolidated[idx] = False

        return idx

    def retrieve(
        self, query: np.ndarray, k: int = 5
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Retrieve memories by key similarity.

        Parameters
        ----------
        query : np.ndarray
            Query key, shape (dim,).
        k : int
            Number of nearest neighbors to retrieve.

        Returns
        -------
        list of (value, similarity) tuples
            Retrieved memories sorted by similarity.
        """
        if self._size == 0:
            return []

        query = np.asarray(query, dtype=np.float64).flatten()
        if query.shape[0] < self.dim:
            padded = np.zeros(self.dim, dtype=np.float64)
            padded[:query.shape[0]] = query
            query = padded
        elif query.shape[0] > self.dim:
            query = query[:self.dim]

        separated_query = self._pattern_separation(query)

        keys_matrix = np.array(self._keys[:self._size])
        similarities = keys_matrix @ separated_query

        k = min(k, self._size)
        top_indices = np.argsort(similarities)[-k:][::-1]

        results = []
        for idx in top_indices:
            self._strengths[idx] = min(1.0, self._strengths[idx] + self.rehearsal_rate)
            self._ages[idx] += 1
            results.append(
                (self._memories[idx].copy(), float(similarities[idx]))
            )

        return results

    def get_consolidation_candidates(
        self, min_age: int = 10, min_strength: float = 0.3
    ) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        """
        Get memories ready for consolidation to cortical store.

        Parameters
        ----------
        min_age : int
            Minimum age (accesses) for consolidation eligibility.
        min_strength : float
            Minimum strength for consolidation eligibility.

        Returns
        -------
        list of (index, key, value) tuples
            Memories ready for consolidation.
        """
        candidates = []
        for i in range(self._size):
            if (
                self._ages[i] >= min_age
                and self._strengths[i] >= min_strength
                and not self._consolidated[i]
            ):
                candidates.append(
                    (i, self._keys[i].copy(), self._memories[i].copy())
                )
        return candidates

    def mark_consolidated(self, index: int) -> None:
        """Mark a memory as consolidated and reduce its strength."""
        if 0 <= index < self._size:
            self._strengths[index] *= 0.5
            self._ages[index] = 0
            self._consolidated[index] = True

    def decay_all(self) -> None:
        """Apply time-based decay to all memories."""
        for i in range(self._size):
            self._strengths[i] -= self.decay_rate
            self._ages[i] += 1

    @property
    def size(self) -> int:
        """Current number of stored memories."""
        return self._size

    @property
    def utilization(self) -> float:
        """Fraction of capacity used."""
        return self._size / self.capacity


class CorticalStore:
    """
    Slow-learning, large-capacity memory store (cortical analog).

    Features:
    - Gradual integration of consolidated memories
    - Pattern completion for partial cues
    - Semantic generalization across experiences
    - Long-term stable storage

    Parameters
    ----------
    dim : int
        Dimensionality of memory vectors.
    capacity : int
        Maximum number of stored memories.
    learning_rate : float
        Rate at which new memories are integrated.
    completion_threshold : float
        Similarity threshold for pattern completion.
    """

    def __init__(
        self,
        dim: int = 256,
        capacity: int = 5000,
        learning_rate: float = 0.1,
        completion_threshold: float = 0.5,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.dim = dim
        self.capacity = capacity
        self.learning_rate = learning_rate
        self.completion_threshold = completion_threshold

        self._keys: List[np.ndarray] = []
        self._values: List[np.ndarray] = []
        self._counts: List[int] = []
        self._size: int = 0

    def consolidate(
        self,
        key: np.ndarray,
        value: np.ndarray,
    ) -> None:
        """
        Consolidate a memory from hippocampal store.

        If a similar key already exists, the value is updated
        using a weighted average (gradual integration).
        If not, a new entry is created.

        Parameters
        ----------
        key : np.ndarray
            Memory key, shape (dim,).
        value : np.ndarray
            Memory value, shape (dim,).
        """
        key = np.asarray(key, dtype=np.float64).flatten()
        value = np.asarray(value, dtype=np.float64).flatten()

        if key.shape[0] < self.dim:
            padded = np.zeros(self.dim, dtype=np.float64)
            padded[:key.shape[0]] = key
            key = padded
        elif key.shape[0] > self.dim:
            key = key[:self.dim]

        if value.shape[0] < self.dim:
            padded = np.zeros(self.dim, dtype=np.float64)
            padded[:value.shape[0]] = value
            value = padded
        elif value.shape[0] > self.dim:
            value = value[:self.dim]

        if self._size > 0:
            keys_matrix = np.array(self._keys)
            similarities = keys_matrix @ key
            best_idx = int(np.argmax(similarities))
            best_sim = float(similarities[best_idx])

            if best_sim >= self.completion_threshold:
                lr = self.learning_rate / (1.0 + self._counts[best_idx] * 0.01)
                self._values[best_idx] = (
                    1.0 - lr
                ) * self._values[best_idx] + lr * value
                self._keys[best_idx] = (
                    1.0 - lr * 0.01
                ) * self._keys[best_idx] + lr * 0.01 * key
                norm = np.linalg.norm(self._keys[best_idx])
                if norm > 1e-10:
                    self._keys[best_idx] /= norm
                self._counts[best_idx] += 1
                return

        if self._size < self.capacity:
            norm = np.linalg.norm(key)
            if norm > 1e-10:
                key = key / norm
            self._keys.append(key.copy())
            self._values.append(value.copy())
            self._counts.append(1)
            self._size += 1
        else:
            norm = np.linalg.norm(key)
            if norm > 1e-10:
                key = key / norm
            least_used = int(np.argmin(self._counts))
            self._keys[least_used] = key.copy()
            self._values[least_used] = value.copy()
            self._counts[least_used] = 1

    def pattern_completion(
        self, partial_key: np.ndarray, k: int = 3
    ) -> Optional[np.ndarray]:
        """
        Complete a partial cue using stored patterns.

        Parameters
        ----------
        partial_key : np.ndarray
            Partial or noisy key, shape (dim,).
        k : int
            Number of nearest neighbors to combine.

        Returns
        -------
        np.ndarray or None
            Completed pattern, or None if no good match.
        """
        if self._size == 0:
            return None

        partial_key = np.asarray(partial_key, dtype=np.float64).flatten()
        if partial_key.shape[0] < self.dim:
            padded = np.zeros(self.dim, dtype=np.float64)
            padded[:partial_key.shape[0]] = partial_key
            partial_key = padded
        elif partial_key.shape[0] > self.dim:
            partial_key = partial_key[:self.dim]

        norm = np.linalg.norm(partial_key)
        if norm > 1e-10:
            partial_key = partial_key / norm

        keys_matrix = np.array(self._keys)
        similarities = keys_matrix @ partial_key

        k = min(k, self._size)
        top_indices = np.argsort(similarities)[-k:][::-1]

        if float(similarities[top_indices[0]]) < self.completion_threshold * 0.5:
            return None

        weights = np.array(
            [max(0, float(similarities[i])) for i in top_indices]
        )
        weight_sum = np.sum(weights)
        if weight_sum < 1e-10:
            return None
        weights /= weight_sum

        completed = np.zeros(self.dim, dtype=np.float64)
        for i, idx in enumerate(top_indices):
            completed += weights[i] * self._values[idx]

        return completed

    def retrieve_semantic(
        self, query: np.ndarray, k: int = 5
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Retrieve semantically related memories.

        Parameters
        ----------
        query : np.ndarray
            Query vector, shape (dim,).
        k : int
            Number of results.

        Returns
        -------
        list of (value, similarity) tuples
        """
        if self._size == 0:
            return []

        query = np.asarray(query, dtype=np.float64).flatten()
        if query.shape[0] < self.dim:
            padded = np.zeros(self.dim, dtype=np.float64)
            padded[:query.shape[0]] = query
            query = padded
        elif query.shape[0] > self.dim:
            query = query[:self.dim]

        norm = np.linalg.norm(query)
        if norm > 1e-10:
            query = query / norm

        keys_matrix = np.array(self._keys)
        similarities = keys_matrix @ query

        k = min(k, self._size)
        top_indices = np.argsort(similarities)[-k:][::-1]

        return [
            (self._values[idx].copy(), float(similarities[idx]))
            for idx in top_indices
        ]

    @property
    def size(self) -> int:
        """Current number of stored memories."""
        return self._size


class ComplementaryMemorySystem:
    """
    Dual-store complementary memory system.

    Combines hippocampal (fast) and cortical (slow) stores with
    an automatic consolidation mechanism that transfers memories
    from fast to slow store during rest periods.

    This implements the complementary learning systems theory
    (McClelland, McNaughton & O'Reilly, 1995).

    Parameters
    ----------
    dim : int
        Dimensionality of memory vectors.
    hippocampal_capacity : int
        Capacity of the fast store.
    cortical_capacity : int
        Capacity of the slow store.
    consolidation_rate : int
        Number of memories to consolidate per cycle.
    consolidation_min_age : int
        Minimum hippocampal age for consolidation.
    """

    def __init__(
        self,
        dim: int = 256,
        hippocampal_capacity: int = 500,
        cortical_capacity: int = 5000,
        consolidation_rate: int = 10,
        consolidation_min_age: int = 10,
    ) -> None:
        self.dim = dim
        self.consolidation_rate = consolidation_rate
        self.consolidation_min_age = consolidation_min_age

        self.hippocampal = HippocampalStore(
            dim=dim, capacity=hippocampal_capacity
        )
        self.cortical = CorticalStore(dim=dim, capacity=cortical_capacity)

        self._consolidation_count: int = 0

    def store(
        self, key: np.ndarray, value: np.ndarray
    ) -> None:
        """
        Store a new memory (goes to hippocampal store first).

        Parameters
        ----------
        key : np.ndarray
            Memory key.
        value : np.ndarray
            Memory value.
        """
        self.hippocampal.encode(key, value)

    def retrieve(
        self, key: np.ndarray, k: int = 5
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Retrieve memories from both stores.

        Searches hippocampal store first (more recent, precise),
        then cortical store (older, generalized).

        Parameters
        ----------
        key : np.ndarray
            Query key.
        k : int
            Total number of results.

        Returns
        -------
        list of (value, similarity) tuples
            Combined results from both stores.
        """
        k_hippo = max(1, k // 2)
        k_cortical = k - k_hippo

        hippo_results = self.hippocampal.retrieve(key, k=k_hippo)
        cortical_results = self.cortical.retrieve_semantic(key, k=k_cortical)

        combined = hippo_results + cortical_results
        combined.sort(key=lambda x: x[1], reverse=True)

        return combined[:k]

    def complete_pattern(self, partial_key: np.ndarray) -> Optional[np.ndarray]:
        """
        Complete a partial pattern using cortical store.

        Parameters
        ----------
        partial_key : np.ndarray
            Partial or noisy key.

        Returns
        -------
        np.ndarray or None
            Completed pattern.
        """
        return self.cortical.pattern_completion(partial_key)

    def consolidate(self) -> Dict[str, int]:
        """
        Run one consolidation cycle.

        Transfers eligible memories from hippocampal to cortical store.
        This should be called during "sleep" or rest periods.

        Returns
        -------
        dict
            Consolidation statistics.
        """
        candidates = self.hippocampal.get_consolidation_candidates(
            min_age=self.consolidation_min_age
        )

        n_consolidated = 0
        for idx, key, value in candidates[:self.consolidation_rate]:
            self.cortical.consolidate(key, value)
            self.hippocampal.mark_consolidated(idx)
            n_consolidated += 1

        self.hippocampal.decay_all()
        self._consolidation_count += 1

        return {
            "n_consolidated": n_consolidated,
            "n_candidates": len(candidates),
            "hippocampal_size": self.hippocampal.size,
            "cortical_size": self.cortical.size,
            "consolidation_cycles": self._consolidation_count,
        }

    def state_dict(self) -> Dict[str, Any]:
        """Return the memory system state for serialization."""
        return {
            "hippocampal_keys": [k.copy() for k in self.hippocampal._keys],
            "hippocampal_values": [v.copy() for v in self.hippocampal._memories],
            "hippocampal_strengths": list(self.hippocampal._strengths),
            "hippocampal_ages": list(self.hippocampal._ages),
            "hippocampal_consolidated": list(self.hippocampal._consolidated),
            "hippocampal_separation_matrix": self.hippocampal._separation_matrix.copy(),
            "cortical_keys": [k.copy() for k in self.cortical._keys],
            "cortical_values": [v.copy() for v in self.cortical._values],
            "cortical_counts": list(self.cortical._counts),
            "consolidation_count": self._consolidation_count,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load memory system state from a dictionary."""
        if "hippocampal_keys" in state:
            self.hippocampal._keys = [k.copy() for k in state["hippocampal_keys"]]
            self.hippocampal._memories = [v.copy() for v in state["hippocampal_values"]]
            self.hippocampal._strengths = list(state["hippocampal_strengths"])
            self.hippocampal._ages = list(state["hippocampal_ages"])
            self.hippocampal._consolidated = list(state.get("hippocampal_consolidated", [False] * len(self.hippocampal._keys)))
            self.hippocampal._size = min(len(self.hippocampal._keys), self.hippocampal.capacity)
            if "hippocampal_separation_matrix" in state:
                self.hippocampal._separation_matrix = state["hippocampal_separation_matrix"].copy()
        if "cortical_keys" in state:
            self.cortical._keys = [k.copy() for k in state["cortical_keys"]]
            self.cortical._values = [v.copy() for v in state["cortical_values"]]
            self.cortical._counts = list(state["cortical_counts"])
            self.cortical._size = min(len(self.cortical._keys), self.cortical.capacity)
        if "consolidation_count" in state:
            self._consolidation_count = state["consolidation_count"]

    def reset(self) -> None:
        """Clear all memories, returning to initial empty state."""
        self.hippocampal._keys.clear()
        self.hippocampal._memories.clear()
        self.hippocampal._strengths.clear()
        self.hippocampal._ages.clear()
        self.hippocampal._consolidated.clear()
        self.hippocampal._size = 0
        self.hippocampal._ptr = 0
        self.cortical._keys.clear()
        self.cortical._values.clear()
        self.cortical._counts.clear()
        self.cortical._size = 0
        self._consolidation_count = 0
