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
        max_item_memory: int = 80000,
        max_associative_memory: int = 50000,
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
        # 图结构记忆：存储 (source, relation, target) 三元组
        self.graph_edges: Dict[str, np.ndarray] = {}
        # 图边数量上限（与 episodic_memory 同策略，防止无限增长导致检索变慢）
        self.max_graph_edges: int = 5000
        # item_memory 和 associative_memory 容量上限（防止 RAM 爆炸）
        # 理论依据：海马体容量有限，重要记忆已巩固到皮层（world_model/PCN）
        self.max_item_memory: int = max_item_memory
        self.max_associative_memory: int = max_associative_memory
        # 概念重要性缓存：state_key -> surprise 等级 (0.0~9.0)
        # 理论依据：海马体对高 surprise（情绪/惊奇）经验加权保留，符合生物机制
        # 用于淘汰时保护高 surprise 的 exp_ 概念，避免重要低频专业知识被误删
        self._concept_importance: Dict[str, float] = {}
        # 按 relation 分桶的图边索引（存储时维护，检索时零字符串解析）
        # relation_name -> [(edge_key, target_name), ...]
        self._edges_by_relation: Dict[str, List[Tuple[str, str]]] = {}

        self._permutation: Optional[np.ndarray] = None
        # item_memory 矩阵缓存（预分配 + 增量填充，dirty 标记追踪失效）
        # 存储 (names, matrix, norms, count) 四元组，matrix 预分配 2x 容量
        self._item_matrix_cache: Optional[Tuple[List[str], np.ndarray, np.ndarray, int]] = None
        self._item_matrix_dirty: bool = True
        # 概念覆盖标记：若 add_concept 覆盖已存在概念，强制全量重建
        self._item_overwritten: bool = False

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

        # 检测覆盖：若概念已存在且向量不同，标记全量重建
        if name in self.item_memory:
            old_vec = self.item_memory[name]
            if not np.array_equal(old_vec, vector):
                self._item_overwritten = True
        self.item_memory[name] = vector
        self._item_matrix_dirty = True

        # 重要性加权淘汰：超出上限 1000 条后，按 surprise 等级升序淘汰 exp_/nexp_ 概念
        # 保留 act_/rew_/sur_ 等常用概念（数量少且反复使用）
        # 理论依据：海马体对高 surprise（情绪/惊奇）经验加权保留，避免重要低频专业知识被误删
        # nexp_ 默认 importance=0 先淘汰；若其重要，作为 state_key 时自有高 importance 被保护
        if len(self.item_memory) > self.max_item_memory + 1000:
            evict_count = len(self.item_memory) - self.max_item_memory
            # 收集候选淘汰概念及其重要性（低 surprise 先淘汰）
            candidates = []
            for name in self.item_memory.keys():
                if name.startswith(("exp_", "nexp_")):
                    importance = self._concept_importance.get(name, 0.0)
                    candidates.append((name, importance))
            # 按重要性升序排序（importance 相同时保持插入序，近似 FIFO）
            candidates.sort(key=lambda x: x[1])
            evicted = 0
            for name, _ in candidates:
                if evicted >= evict_count:
                    break
                del self.item_memory[name]
                # 清理重要性缓存，避免孤儿条目累积
                self._concept_importance.pop(name, None)
                evicted += 1
            if evicted > 0:
                self._item_matrix_dirty = True
                self._item_overwritten = True  # 强制全量重建缓存

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
        # 用局部变量保存向量，避免 add_concept 触发淘汰导致 KeyError
        # 理论依据：bind 只需向量本身，不依赖 item_memory 中的存活；
        # 关联键被淘汰后 retrieve 返回 None（海马体自然遗忘）
        key_vec = self.item_memory.get(key_name)
        if key_vec is None:
            key_vec = self.add_concept(key_name)
        value_vec = self.item_memory.get(value_name)
        if value_vec is None:
            value_vec = self.add_concept(value_name)

        bound = self.bind(key_vec, value_vec)
        self.associative_memory[key_name] = bound

        # 重要性加权：解析 surprise 关联，缓存 state 的 surprise 等级
        # 理论依据：高 surprise = 高信息量，海马体对惊奇经验加权保留（情绪/惊奇机制）
        # 关联格式：key="exp_{hex}__surprise" -> value="sur_{0-9}"
        if key_name.endswith("__surprise") and value_name.startswith("sur_"):
            state_key = key_name[:-len("__surprise")]
            try:
                surprise_level = float(value_name[4:])
                # 同一 state 多次经历，保留最高 surprise（重要经验难忘）
                if surprise_level > self._concept_importance.get(state_key, -1.0):
                    self._concept_importance[state_key] = surprise_level
            except (ValueError, IndexError):
                pass

        # FIFO 淘汰：所有 key 均为 "exp_*__action" 等一次性状态关联，可安全淘汰
        # 理论依据：海马体容量有限，重要关联已通过巩固存入皮层
        while len(self.associative_memory) > self.max_associative_memory:
            oldest_key = next(iter(self.associative_memory))
            del self.associative_memory[oldest_key]

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

    def _build_item_matrix(self) -> Tuple[List[str], np.ndarray, np.ndarray, int]:
        """构建 item_memory 的矩阵缓存（预分配 + 增量填充）。

        预分配策略：matrix 分配 2x 容量，新概念直接填入空行，
        避免每次 vstack 复制整个矩阵（1.6GB → 0 复制）。
        返回 (names, matrix, norms, count)，仅前 count 行有效。
        """
        if not self._item_matrix_dirty:
            return self._item_matrix_cache

        # 覆盖或无缓存：全量重建（预分配 2x 容量）
        if self._item_overwritten or self._item_matrix_cache is None:
            names = list(self.item_memory.keys())
            count = len(names)
            capacity = max(1024, count * 2)
            matrix = np.zeros((capacity, self.dim), dtype=np.float64)
            norms = np.zeros(capacity, dtype=np.float64)
            for i, name in enumerate(names):
                vec = self.item_memory[name]
                matrix[i] = vec
                norms[i] = np.sqrt(np.dot(vec, vec))
            self._item_matrix_cache = (names, matrix, norms, count)
            self._item_overwritten = False
            self._item_matrix_dirty = False
            return self._item_matrix_cache

        # 增量追加：找出新概念
        old_names, old_matrix, old_norms, old_count = self._item_matrix_cache
        old_name_set = set(old_names)
        new_names = [n for n in self.item_memory.keys() if n not in old_name_set]

        if not new_names:
            self._item_matrix_dirty = False
            return self._item_matrix_cache

        # 安全检查：概念数减少（删除），回退全量重建
        if len(self.item_memory) < old_count:
            names = list(self.item_memory.keys())
            count = len(names)
            capacity = max(1024, count * 2)
            matrix = np.zeros((capacity, self.dim), dtype=np.float64)
            norms = np.zeros(capacity, dtype=np.float64)
            for i, name in enumerate(names):
                vec = self.item_memory[name]
                matrix[i] = vec
                norms[i] = np.sqrt(np.dot(vec, vec))
            self._item_matrix_cache = (names, matrix, norms, count)
            self._item_matrix_dirty = False
            return self._item_matrix_cache

        # 检查容量，不足则扩容（2x）
        capacity = old_matrix.shape[0]
        needed = old_count + len(new_names)
        if needed > capacity:
            new_capacity = max(needed * 2, capacity * 2)
            new_matrix = np.zeros((new_capacity, self.dim), dtype=np.float64)
            new_norms = np.zeros(new_capacity, dtype=np.float64)
            new_matrix[:old_count] = old_matrix[:old_count]
            new_norms[:old_count] = old_norms[:old_count]
            old_matrix = new_matrix
            old_norms = new_norms

        # 填入新概念（零拷贝：直接写入预分配行）
        for i, name in enumerate(new_names):
            idx = old_count + i
            vec = self.item_memory[name]
            old_matrix[idx] = vec
            old_norms[idx] = np.sqrt(np.dot(vec, vec))

        names = old_names + new_names
        count = old_count + len(new_names)
        self._item_matrix_cache = (names, old_matrix, old_norms, count)
        self._item_matrix_dirty = False
        return self._item_matrix_cache

    def retrieve(
        self, query_vector: np.ndarray, top_k: int = 1
    ) -> List[Tuple[str, float]]:
        """
        Retrieve the most similar concepts from item memory.

        Performs a cleanup operation by finding the nearest
        neighbors in the item memory.

        向量化实现：将 item_memory 堆叠成矩阵，用一次矩阵乘法
        计算所有余弦相似度（语义与原 Python 循环完全一致，
        BLAS 加速 10-100 倍）。

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
        if not self.item_memory:
            return []

        names, matrix, norms, count = self._build_item_matrix()
        # 批量余弦相似度：sims = (M @ q) / (||M|| * ||q||)
        # 仅使用前 count 行（预分配矩阵的有效部分）
        q_norm = np.linalg.norm(query_vector)
        if q_norm < 1e-16:
            return []
        m = matrix[:count]
        n = norms[:count]
        sims = (m @ query_vector) / (n * q_norm + 1e-16)  # (count,)

        # top_k 选择
        if top_k >= count:
            top_idx = np.argsort(-sims)
        else:
            # argpartition 选 top_k，再局部排序
            top_idx = np.argpartition(-sims, top_k)[:top_k]
            top_idx = top_idx[np.argsort(-sims[top_idx])]

        return [(names[i], float(sims[i])) for i in top_idx]

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

    # ------------------------------------------------------------------
    # Graph-structured memory (knowledge graph via HDC)
    # ------------------------------------------------------------------

    def store_graph_edge(
        self,
        source: str,
        relation: str,
        target: str,
    ) -> np.ndarray:
        """
        存储知识图谱三元组 (source, relation, target)。

        编码方式：edge_vec = bind(bind(source_vec, relation_vec), target_vec)
        这样可以通过 unbind 进行图查询：
        - 给定 source + relation，查询 target
        - 给定 source + target，查询 relation

        Parameters
        ----------
        source : str
            源概念名。
        relation : str
            关系名（如 "is_a", "causes", "part_of"）。
        target : str
            目标概念名。

        Returns
        -------
        np.ndarray
            图边的超维向量。
        """
        for name in (source, relation, target):
            if name not in self.item_memory:
                self.add_concept(name)

        s_vec = self.item_memory[source]
        r_vec = self.item_memory[relation]
        t_vec = self.item_memory[target]

        # 三元组编码：bind(bind(s, r), t)
        edge_vec = self.bind(self.bind(s_vec, r_vec), t_vec)

        edge_key = f"{source}|{relation}|{target}"
        is_new = edge_key not in self.graph_edges
        # 图边上限：仅新边且超容量时淘汰最旧的（避免覆盖时误删）
        if is_new and len(self.graph_edges) >= self.max_graph_edges:
            oldest_key = next(iter(self.graph_edges))
            del self.graph_edges[oldest_key]
            # 同步从 _edges_by_relation 索引中移除
            old_parts = oldest_key.split("|")
            if len(old_parts) == 3:
                old_rel = old_parts[1]
                if old_rel in self._edges_by_relation:
                    self._edges_by_relation[old_rel] = [
                        (k, t) for (k, t) in self._edges_by_relation[old_rel]
                        if k != oldest_key
                    ]
                    if not self._edges_by_relation[old_rel]:
                        del self._edges_by_relation[old_rel]
        self.graph_edges[edge_key] = edge_vec.copy()
        # 仅新边时添加索引条目（避免重复三元组产生重复索引）
        if is_new:
            if relation not in self._edges_by_relation:
                self._edges_by_relation[relation] = []
            self._edges_by_relation[relation].append((edge_key, target))

        return edge_vec

    def query_graph(
        self,
        source: str,
        relation: str,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        图查询：给定 (source, relation)，检索最可能的 target。

        纯 HDC 联想检索（无字符串解析）：
        1. 构造查询向量 query_vec = bind(source_vec, relation_vec)
        2. 对每条图边做 unbind(edge_vec, query_vec) 得到 target 近似
        3. 在 item_memory 中找最相似的 target
        4. 按相似度排序返回

        向量化实现：将所有图边堆叠成矩阵，用一次矩阵乘法计算
        所有 unbind 结果与所有 item_memory 的相似度（语义与原
        Python 循环完全一致，BLAS 加速 100 倍+）。

        纯HDC实现的优势：
        - 不依赖字符串解析，完全基于超维向量的联想检索
        - 支持模糊查询（source/relation 不完全匹配也能工作）
        - 体现 HDC 的核心理论：bind/unbind 的自逆性

        Parameters
        ----------
        source : str
            源概念名。
        relation : str
            关系名。
        top_k : int
            返回前 k 个结果。

        Returns
        -------
        list of (target_name, similarity) tuples
        """
        if source not in self.item_memory or relation not in self.item_memory:
            return []
        if not self.graph_edges:
            return []

        s_vec = self.item_memory[source]
        r_vec = self.item_memory[relation]
        query_vec = self.bind(s_vec, r_vec)  # (D,)

        # ---- 用存储时维护的索引获取该 relation 的图边 ----
        # 纯 HDC 原则：检索时零字符串解析，索引在 store_graph_edge 时已构建
        # 这将 E 从全部图边(最多 5000)缩减到该 relation 下的子集
        indexed = self._edges_by_relation.get(relation, [])
        if not indexed:
            return []

        edge_keys = [k for (k, _) in indexed]
        edge_matrix = np.array(
            [self.graph_edges[k] for k in edge_keys], dtype=np.float64
        )  # (E_f, D)

        # 批量 unbind：edge_matrix * query_vec → t_approx_matrix
        if self.vector_type == VectorType.BINARY:
            t_approx_matrix = np.logical_xor(
                edge_matrix.astype(bool), query_vec[np.newaxis, :].astype(bool)
            ).astype(np.float64)
        else:
            t_approx_matrix = edge_matrix * query_vec[np.newaxis, :]  # (E_f, D)

        # ---- 缩小 cleanup 范围：只与该 relation 下的 target 概念比较 ----
        # target 名从存储时索引获取（非运行时字符串解析）
        # N 从全部 item_memory(~20000) 缩减到目标概念子集
        target_names: List[str] = []
        seen: set = set()
        for (_, t) in indexed:
            if t not in seen and t in self.item_memory:
                seen.add(t)
                target_names.append(t)
        if not target_names:
            return []

        target_matrix = np.array(
            [self.item_memory[t] for t in target_names], dtype=np.float64
        )  # (N_f, D)

        # 相似度矩阵: (E_f, N_f) — 纯 HDC cosine similarity
        t_norms = np.linalg.norm(t_approx_matrix, axis=1)  # (E_f,)
        target_norms = np.linalg.norm(target_matrix, axis=1)  # (N_f,)
        denom = np.outer(t_norms, target_norms) + 1e-16  # (E_f, N_f)
        sims_matrix = (t_approx_matrix @ target_matrix.T) / denom  # (E_f, N_f)

        # 对每条边取 top-1（最相似的 target）
        best_indices = np.argmax(sims_matrix, axis=1)  # (E_f,)
        best_sims = sims_matrix[np.arange(len(edge_keys)), best_indices]  # (E_f,)

        # 按相似度排序，取 top_k
        sorted_idx = np.argsort(-best_sims)[:top_k]
        results = [
            (target_names[best_indices[i]], float(best_sims[i])) for i in sorted_idx
        ]

        return results

    def graph_walk(
        self,
        start_concept: str,
        relations: List[str],
    ) -> List[Tuple[str, str, str, float]]:
        """
        沿知识图谱进行多关系多跳心智游走（multi-hop reasoning）。

        支持异质关系串联，例如：
            graph_walk("苹果", ["is_a", "contains"])
            → 苹果 --is_a--> 水果 --contains--> 维生素

        这实现了理论声明中的：
        "苹果->是一种->水果->含有->维生素" 多 relation 串联推理。

        Parameters
        ----------
        start_concept : str
            起始概念名。
        relations : list of str
            每步要沿的关系类型序列（支持不同关系串联）。

        Returns
        -------
        list of (relation, current_concept, next_concept, similarity) tuples
            游走路径，每步记录用到的关系、当前概念、下一概念、相似度。
        """
        path: List[Tuple[str, str, str, float]] = []
        current = start_concept

        for relation in relations:
            if current not in self.item_memory:
                break
            if relation not in self.item_memory:
                break
            results = self.query_graph(current, relation, top_k=1)
            if not results:
                break
            next_concept, sim = results[0]
            if next_concept == current or sim < self.similarity_threshold:
                break
            path.append((relation, current, next_concept, sim))
            current = next_concept

        return path

    def reset(self) -> None:
        """Clear all memories."""
        self.item_memory = {}
        self.associative_memory = {}
        self.episodic_memory = []
        self.graph_edges = {}
        self._edges_by_relation = {}
        self._permutation = None
        self._item_matrix_cache = None
        self._item_matrix_dirty = True
        self._item_overwritten = False
        self._concept_importance = {}

    def state_dict(self) -> Dict[str, Any]:
        """Return the memory state for serialization."""
        return {
            "item_memory": dict(self.item_memory),
            "associative_memory": dict(self.associative_memory),
            "episodic_memory": list(self.episodic_memory),
            "graph_edges": dict(self.graph_edges),
            "dim": self.dim,
            "vector_type": self.vector_type.value,
            "similarity_threshold": self.similarity_threshold,
            "_permutation": self._permutation.copy() if self._permutation is not None else None,
            "_concept_importance": dict(self._concept_importance),
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
        self.graph_edges = {
            k: v.copy() for k, v in state.get("graph_edges", {}).items()
        }
        self._permutation = state["_permutation"].copy() if state.get("_permutation") is not None else None
        # 从 graph_edges 重建 relation 索引
        self._edges_by_relation: Dict[str, List[Tuple[str, str]]] = {}
        for edge_key in self.graph_edges:
            parts = edge_key.split("|")
            if len(parts) == 3:
                _, rel, tgt = parts
                if rel not in self._edges_by_relation:
                    self._edges_by_relation[rel] = []
                self._edges_by_relation[rel].append((edge_key, tgt))
        # 缓存失效，下次 retrieve 时重建
        self._item_matrix_cache = None
        self._item_matrix_dirty = True
        self._item_overwritten = False
        # 恢复概念重要性缓存（旧 checkpoint 兼容：无此字段时置空，后续新经验自然填充）
        saved_importance = state.get("_concept_importance")
        if saved_importance is not None:
            self._concept_importance = dict(saved_importance)
        else:
            # 旧 checkpoint 无重要性缓存：置空，首次淘汰将按近似 FIFO（importance 全 0）
            # 之后新经验会通过 store_association 正常记录 importance，逐步过渡到加权淘汰
            self._concept_importance = {}

    def __repr__(self) -> str:
        return (
            f"HDCMemory(dim={self.dim}, "
            f"type={self.vector_type.value}, "
            f"concepts={len(self.item_memory)}, "
            f"associations={len(self.associative_memory)}, "
            f"episodes={len(self.episodic_memory)})"
        )
