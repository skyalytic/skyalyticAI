"""
Hyperdimensional Computing (HDC) Memory System

Implements a brain-inspired memory system using high-dimensional
random vectors (typically D > 10000). The key insight is that
in high-dimensional spaces, random vectors are nearly orthogonal,
enabling robust associative memory operations.

Core operations:
1. Bundling (addition): represents sets/superpositions
   A + B -> vector similar to both A and B
2. Binding (multiplication/XOR): represents associations/pairs
   A * B -> vector dissimilar to both A and B
3. Permutation (cyclic shift): represents sequences/positions
   rho(A) -> vector dissimilar to A

Key properties:
- Noise robustness: corrupted vectors can still be retrieved
- Holographic: information is distributed across all dimensions
- Scalable: operations preserve dimensionality

Bug fix from theory document:
- store_association now correctly uses bind(key, value) instead
  of bind(key, key)
- Added proper cleanup memory for associative recall
- Added proper unbinding for key-value retrieval
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class VectorType(Enum):
    BIPOLAR = "bipolar"
    BINARY = "binary"
    REAL = "real"


class HDCMemory:
    """
    Hyperdimensional Computing Memory System.

    Provides a complete associative memory system using high-dimensional
    vectors, supporting concept storage, key-value associations,
    episodic (sequence) memory, and robust retrieval with cleanup.

    Parameters
    ----------
    dim : int
        Dimensionality of hypervectors. Typical values are 1000-10000.
        Higher dimensions provide better orthogonality and noise
        robustness but require more memory and computation.
    vector_type : VectorType or str
        Type of hypervectors to use:
        - 'bipolar': elements are {-1, +1}, best for binding via
          element-wise multiplication
        - 'binary': elements are {0, 1}, binding via XOR
        - 'real': elements are continuous, less common
    seed : int or None
        Random seed for reproducibility.
    similarity_threshold : float
        Minimum cosine similarity for a successful retrieval.
        Must be in [0, 1].
    """

    def __init__(
        self,
        dim: int = 10000,
        vector_type: VectorType | str = VectorType.BIPOLAR,
        seed: Optional[int] = None,
        similarity_threshold: float = 0.1,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if not 0 <= similarity_threshold <= 1:
            raise ValueError(
                f"similarity_threshold must be in [0, 1], "
                f"got {similarity_threshold}"
            )

        self.dim = dim
        if isinstance(vector_type, str):
            vector_type = VectorType(vector_type.lower())
        self.vector_type = vector_type
        self.similarity_threshold = similarity_threshold

        self.rng = np.random.default_rng(seed)

        self.item_memory: Dict[str, np.ndarray] = {}
        self.associative_memory: Dict[str, np.ndarray] = {}
        self.episodic_memory: List[Dict[str, Any]] = []

        self._permutation: Optional[np.ndarray] = None

    def random_vector(self) -> np.ndarray:
        """
        Generate a random hypervector.

        Returns
        -------
        np.ndarray
            Random vector of shape (dim,) with the configured type.
        """
        if self.vector_type == VectorType.BIPOLAR:
            return self.rng.choice([-1, 1], size=self.dim).astype(np.float64)
        elif self.vector_type == VectorType.BINARY:
            return self.rng.choice([0, 1], size=self.dim).astype(np.float64)
        else:
            return self.rng.standard_normal(self.dim).astype(np.float64)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Parameters
        ----------
        a, b : np.ndarray
            Vectors to compare.

        Returns
        -------
        float
            Cosine similarity in [-1, 1].
        """
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-16 or norm_b < 1e-16:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def bundle(self, *vectors: np.ndarray) -> np.ndarray:
        """
        Bundle (superpose) multiple vectors via element-wise addition.

        The result is similar to each input vector, representing a
        set or superposition of concepts. For bipolar vectors, the
        result is thresholded back to bipolar.

        Parameters
        ----------
        *vectors : np.ndarray
            Vectors to bundle. Each must have shape (dim,).

        Returns
        -------
        np.ndarray
            Bundled vector of shape (dim,).
        """
        if len(vectors) == 0:
            raise ValueError("At least one vector must be provided")

        result = np.zeros(self.dim, dtype=np.float64)
        for v in vectors:
            v = np.asarray(v, dtype=np.float64)
            if v.shape != (self.dim,):
                raise ValueError(
                    f"Vector shape must be ({self.dim},), got {v.shape}"
                )
            result += v

        if self.vector_type == VectorType.BIPOLAR:
            result = np.sign(result)
            zero_mask = result == 0
            if np.any(zero_mask):
                result[zero_mask] = self.rng.choice([-1, 1], size=int(np.sum(zero_mask)))
            return result
        elif self.vector_type == VectorType.BINARY:
            threshold = len(vectors) / 2.0
            result_bin = (result >= threshold).astype(np.float64)
            # For exact ties (even number of vectors), randomly resolve
            tie_mask = result == threshold
            if np.any(tie_mask):
                result_bin[tie_mask] = self.rng.choice([0.0, 1.0], size=int(np.sum(tie_mask)))
            return result_bin
        else:
            return result / len(vectors)

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Bind two vectors to create an association.

        The result is dissimilar to both inputs, representing a
        key-value pair. Binding is self-inverse: bind(bind(a, b), b) = a.

        Parameters
        ----------
        a, b : np.ndarray
            Vectors to bind.

        Returns
        -------
        np.ndarray
            Bound vector of shape (dim,).
        """
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        if a.shape != (self.dim,):
            raise ValueError(f"Vector a shape must be ({self.dim},), got {a.shape}")
        if b.shape != (self.dim,):
            raise ValueError(f"Vector b shape must be ({self.dim},), got {b.shape}")

        if self.vector_type == VectorType.BIPOLAR:
            return a * b
        elif self.vector_type == VectorType.BINARY:
            return np.logical_xor(a.astype(bool), b.astype(bool)).astype(np.float64)
        else:
            return a * b

    def unbind(self, bound: np.ndarray, key: np.ndarray) -> np.ndarray:
        """
        Unbind a vector to retrieve the associated value.

        For bipolar/binary vectors, unbinding is the same as binding
        (self-inverse property): unbind(bind(a, b), b) = a.

        Parameters
        ----------
        bound : np.ndarray
            Previously bound vector.
        key : np.ndarray
            Key vector used in the original binding.

        Returns
        -------
        np.ndarray
            Unbound vector (approximation of the original value).
        """
        return self.bind(bound, key)

    def permute(self, v: np.ndarray, shift: int = 1) -> np.ndarray:
        """
        Permute (cyclically shift) a vector.

        The result is dissimilar to the original, useful for
        encoding position in sequences.

        Parameters
        ----------
        v : np.ndarray
            Vector to permute.
        shift : int
            Number of positions to shift. Positive shifts left.

        Returns
        -------
        np.ndarray
            Permuted vector.
        """
        v = np.asarray(v, dtype=np.float64)
        if v.shape != (self.dim,):
            raise ValueError(f"Vector shape must be ({self.dim},), got {v.shape}")
        return np.roll(v, shift)

    def add_concept(self, name: str, vector: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Add a concept to the item memory.

        Parameters
        ----------
        name : str
            Concept name/identifier.
        vector : np.ndarray or None
            Hypervector for this concept. If None, a random
            vector is generated.

        Returns
        -------
        np.ndarray
            The concept's hypervector.
        """
        if vector is None:
            vector = self.random_vector()
        else:
            vector = np.asarray(vector, dtype=np.float64)
            if vector.shape != (self.dim,):
                raise ValueError(
                    f"Vector shape must be ({self.dim},), got {vector.shape}"
                )

        self.item_memory[name] = vector
        return vector

    def get_concept(self, name: str) -> Optional[np.ndarray]:
        """
        Retrieve a concept vector by name.

        Parameters
        ----------
        name : str
            Concept name.

        Returns
        -------
        np.ndarray or None
            The concept's hypervector, or None if not found.
        """
        return self.item_memory.get(name)

    def store_association(self, key_name: str, value_name: str) -> np.ndarray:
        """
        Store a key-value association in associative memory.

        The association is stored by binding the key and value vectors.
        To retrieve the value given the key, use retrieve_association().

        Bug fix: The original theory document used bind(key, key) which
        is incorrect. This implementation correctly uses bind(key, value).

        Parameters
        ----------
        key_name : str
            Name of the key concept.
        value_name : str
            Name of the value concept.

        Returns
        -------
        np.ndarray
            The bound key-value vector.
        """
        if key_name not in self.item_memory:
            self.add_concept(key_name)
        if value_name not in self.item_memory:
            self.add_concept(value_name)

        key_vec = self.item_memory[key_name]
        value_vec = self.item_memory[value_name]

        bound = self.bind(key_vec, value_vec)
        self.associative_memory[key_name] = bound

        return bound

    def retrieve_association(self, key_name: str) -> Optional[Tuple[str, float]]:
        """
        Retrieve the value associated with a key.

        Unbinds the stored association with the key, then finds
        the most similar concept in the item memory (cleanup).

        Parameters
        ----------
        key_name : str
            Name of the key concept.

        Returns
        -------
        tuple or None
            (value_name, similarity) if found, None otherwise.
        """
        if key_name not in self.associative_memory:
            return None
        if key_name not in self.item_memory:
            return None

        bound = self.associative_memory[key_name]
        key_vec = self.item_memory[key_name]

        unbound = self.unbind(bound, key_vec)

        result = self.retrieve(unbound, top_k=1)
        if result and result[0][1] >= self.similarity_threshold:
            return result[0]
        return None

    def retrieve(
        self, query_vector: np.ndarray, top_k: int = 1
    ) -> List[Tuple[str, float]]:
        """
        Retrieve the most similar concepts from item memory.

        Performs a cleanup operation by finding the nearest
        neighbors in the item memory.

        Parameters
        ----------
        query_vector : np.ndarray
            Query vector.
        top_k : int
            Number of results to return.

        Returns
        -------
        list of (name, similarity) tuples
            Top-k most similar concepts, sorted by similarity
            in descending order.
        """
        query_vector = np.asarray(query_vector, dtype=np.float64)
        if query_vector.shape != (self.dim,):
            raise ValueError(
                f"Query shape must be ({self.dim},), got {query_vector.shape}"
            )

        similarities = []
        for name, vec in self.item_memory.items():
            sim = self.cosine_similarity(query_vector, vec)
            similarities.append((name, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def store_episode(self, sequence: List[str]) -> np.ndarray:
        """
        Store an episodic (sequence) memory.

        Encodes the sequence using role-filler binding:
        each item at position i is bound with a position vector
        (created by permuting a base vector i times), then all
        position-encoded items are bundled together.

        Parameters
        ----------
        sequence : list of str
            Ordered list of concept names forming the episode.

        Returns
        -------
        np.ndarray
            Episode hypervector.
        """
        if len(sequence) == 0:
            raise ValueError("Sequence must not be empty")

        for item in sequence:
            if item not in self.item_memory:
                self.add_concept(item)

        position_vector = self._get_position_vector()

        episode_vector = np.zeros(self.dim, dtype=np.float64)

        for i, item_name in enumerate(sequence):
            item_vec = self.item_memory[item_name]
            position_encoded = self.bind(self.permute(position_vector, i), item_vec)
            episode_vector += position_encoded

        if self.vector_type == VectorType.BIPOLAR:
            episode_vector = np.sign(episode_vector)
            zero_mask = episode_vector == 0
            if np.any(zero_mask):
                episode_vector[zero_mask] = self.rng.choice([-1, 1], size=int(np.sum(zero_mask)))
        elif self.vector_type == VectorType.BINARY:
            threshold = len(sequence) / 2.0
            tie_mask = episode_vector == threshold
            episode_vector = (episode_vector >= threshold).astype(np.float64)
            if np.any(tie_mask):
                episode_vector[tie_mask] = self.rng.choice([0.0, 1.0], size=int(np.sum(tie_mask)))

        self.episodic_memory.append({
            "sequence": list(sequence),
            "vector": episode_vector.copy(),
        })
        if len(self.episodic_memory) > 10000:
            self.episodic_memory = self.episodic_memory[-5000:]

        return episode_vector

    def query_episode(
        self, partial_sequence: List[str], n_positions: int = 0
    ) -> Optional[Tuple[List[str], float]]:
        """
        Query episodic memory with a partial sequence.

        Encodes the partial sequence the same way as store_episode
        and finds the most similar stored episode.

        Parameters
        ----------
        partial_sequence : list of str
            Partial sequence to use as query.
        n_positions : int
            Number of positions to consider. If 0, uses the
            length of the partial sequence.

        Returns
        -------
        tuple or None
            (matched_sequence, similarity) if a match is found
            above the similarity threshold, None otherwise.
        """
        if len(partial_sequence) == 0:
            return None

        for item in partial_sequence:
            if item not in self.item_memory:
                return None

        position_vector = self._get_position_vector()

        query_vec = np.zeros(self.dim, dtype=np.float64)
        n_pos = n_positions if n_positions > 0 else len(partial_sequence)

        for i in range(min(len(partial_sequence), n_pos)):
            item_vec = self.item_memory[partial_sequence[i]]
            position_encoded = self.bind(self.permute(position_vector, i), item_vec)
            query_vec += position_encoded

        if self.vector_type == VectorType.BIPOLAR:
            query_vec = np.sign(query_vec)
            zero_mask = query_vec == 0
            if np.any(zero_mask):
                query_vec[zero_mask] = self.rng.choice([-1, 1], size=int(np.sum(zero_mask)))
        elif self.vector_type == VectorType.BINARY:
            n_actual = min(len(partial_sequence), n_pos)
            threshold = n_actual / 2.0
            tie_mask = query_vec == threshold
            query_vec = (query_vec >= threshold).astype(np.float64)
            if np.any(tie_mask):
                query_vec[tie_mask] = self.rng.choice([0.0, 1.0], size=int(np.sum(tie_mask)))

        best_match = None
        best_sim = -1.0

        for ep in self.episodic_memory:
            sim = self.cosine_similarity(query_vec, ep["vector"])
            if sim > best_sim:
                best_sim = sim
                best_match = ep

        if best_match is not None and best_sim >= self.similarity_threshold:
            return best_match["sequence"], best_sim

        return None

    def _get_position_vector(self) -> np.ndarray:
        """Get or create the position vector used for sequence encoding."""
        if self._permutation is None:
            self._permutation = self.random_vector()
        return self._permutation

    def reset(self) -> None:
        """Clear all memories."""
        self.item_memory = {}
        self.associative_memory = {}
        self.episodic_memory = []
        self._permutation = None

    def state_dict(self) -> Dict[str, Any]:
        """Return the memory state for serialization."""
        return {
            "item_memory": dict(self.item_memory),
            "associative_memory": dict(self.associative_memory),
            "episodic_memory": list(self.episodic_memory),
            "dim": self.dim,
            "vector_type": self.vector_type.value,
            "similarity_threshold": self.similarity_threshold,
            "_permutation": self._permutation.copy() if self._permutation is not None else None,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load memory state from a dictionary."""
        if "dim" in state:
            self.dim = state["dim"]
        if "vector_type" in state:
            self.vector_type = VectorType(state["vector_type"])
        if "similarity_threshold" in state:
            self.similarity_threshold = state["similarity_threshold"]
        self.item_memory = {k: v.copy() for k, v in state["item_memory"].items()}
        self.associative_memory = {k: v.copy() for k, v in state["associative_memory"].items()}
        self.episodic_memory = [
            {"sequence": list(ep["sequence"]), "vector": ep["vector"].copy()}
            for ep in state["episodic_memory"]
        ]
        if len(self.episodic_memory) > 10000:
            self.episodic_memory = self.episodic_memory[-5000:]
        self._permutation = state["_permutation"].copy() if state.get("_permutation") is not None else None

    def __repr__(self) -> str:
        return (
            f"HDCMemory(dim={self.dim}, "
            f"type={self.vector_type.value}, "
            f"concepts={len(self.item_memory)}, "
            f"associations={len(self.associative_memory)}, "
            f"episodes={len(self.episodic_memory)})"
        )
