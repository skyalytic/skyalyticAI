"""
稀疏连接基础设施 —— 人脑规模稀疏神经网络

参考：
- Digital Twin Brain (Fudan, Nature Computational Science 2024)
- CORTEX (Fugaku Supercomputer, 2024)
- GeNN Sparse SNN Framework (Sussex/Jülich)
- SpikingBrain (CAS, 2025)

人脑关键参数：
- 860亿神经元
- 每神经元~7000突触（稀疏连接，非全连接）
- 脉冲率1-10Hz（事件驱动）
"""

import numpy as np
from scipy import sparse
from typing import Optional, Tuple, Dict, Any


# 人脑参数常量
BRAIN_NEURONS = 86_000_000_000        # 860亿神经元
BRAIN_SYNAPSES_PER_NEURON = 7_000     # 每神经元平均突触数
BRAIN_TOTAL_SYNAPSES = 600_000_000_000_000  # 600万亿突触
BRAIN_FIRING_RATE_HZ = 7.0            # 平均放电率7Hz
BRAIN_SPARSITY = 1.0 - (BRAIN_SYNAPSES_PER_NEURON / BRAIN_NEURONS)  # ~99.99999%稀疏


class SparseConnectivity:
    """
    稀疏连接矩阵管理器

    使用CSR(Compressed Sparse Row)格式存储突触连接，
    每个神经元只连接到有限数量的其他神经元（~7000），
    匹配人脑的稀疏连接模式。

    内存估算：
    - 860亿神经元 × 7000突触/神经元 = 602万亿突触
    - CSR格式：每突触约8字节(权重) + 4字节(列索引) = 12字节
    - 总计：~7.2 PB（需要分布式存储）
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        synapses_per_neuron: int = 7000,
        weight_init: str = "glorot",
        weight_scale: float = 1.0,
        seed: Optional[int] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            n_pre: 突触前神经元数
            n_post: 突触后神经元数
            synapses_per_neuron: 每个突触后神经元的入度突触数
            weight_init: 权重初始化方式
            weight_scale: 权重缩放
            seed: 随机种子
            device: 计算设备
        """
        self.n_pre = n_pre
        self.n_post = n_post
        self.synapses_per_neuron = min(synapses_per_neuron, n_pre)
        self.weight_init = weight_init
        self.weight_scale = weight_scale
        self.seed = seed
        self.device = device

        # 初始化稀疏权重矩阵
        self.W = self._init_sparse_weights()

        # 统计信息
        self.n_synapses = self.W.nnz
        self.sparsity = 1.0 - (self.n_synapses / (n_pre * n_post))

    def _init_sparse_weights(self) -> sparse.csr_matrix:
        """初始化稀疏权重矩阵（CSR格式）"""
        rng = np.random.default_rng(self.seed)
        k = self.synapses_per_neuron

        # 为每个突触后神经元随机选择k个突触前神经元
        rows = []
        cols = []
        data = []

        for post_idx in range(self.n_post):
            # 随机选择k个突触前神经元（无重复）
            pre_indices = rng.choice(self.n_pre, size=k, replace=False)

            # 初始化权重
            if self.weight_init == "glorot":
                scale = np.sqrt(2.0 / (k + self.n_post)) * self.weight_scale
                weights = rng.normal(0, scale, size=k)
            elif self.weight_init == "kaiming_normal":
                scale = np.sqrt(2.0 / k) * self.weight_scale
                weights = rng.normal(0, scale, size=k)
            elif self.weight_init == "uniform":
                limit = np.sqrt(6.0 / (k + self.n_post)) * self.weight_scale
                weights = rng.uniform(-limit, limit, size=k)
            else:
                weights = rng.normal(0, 0.01, size=k) * self.weight_scale

            rows.extend([post_idx] * k)
            cols.extend(pre_indices.tolist())
            data.extend(weights.tolist())

        W = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(self.n_post, self.n_pre),
            dtype=np.float64,
        )
        return W

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        稀疏矩阵-向量乘法：y = W @ x

        利用CSR格式的高效spMV操作，
        只计算存在的突触连接。
        """
        return self.W.dot(x)

    def update_weights(self, row_indices: np.ndarray, col_indices: np.ndarray,
                       delta_w: np.ndarray) -> None:
        """
        稀疏权重更新（只更新已有连接）

        Args:
            row_indices: 突触后神经元索引
            col_indices: 突触前神经元索引
            delta_w: 权重变化量
        """
        # 转换为COO格式便于修改
        W_coo = self.W.tocoo()

        # 构建查找字典
        existing = {}
        for idx in range(W_coo.nnz):
            existing[(W_coo.row[idx], W_coo.col[idx])] = idx

        # 更新已有连接的权重
        for i in range(len(row_indices)):
            key = (row_indices[i], col_indices[i])
            if key in existing:
                W_coo.data[existing[key]] += delta_w[i]

        self.W = W_coo.tocsr()

    def prune_weights(self, threshold: float = 0.01) -> int:
        """
        突触修剪：移除权重绝对值低于阈值的连接
        （模拟人脑的突触修剪过程）

        Returns:
            pruned_count: 被修剪的突触数
        """
        W_coo = self.W.tocoo()
        mask = np.abs(W_coo.data) >= threshold
        pruned_count = np.sum(~mask)

        self.W = sparse.csr_matrix(
            (W_coo.data[mask], (W_coo.row[mask], W_coo.col[mask])),
            shape=(self.n_post, self.n_pre),
            dtype=np.float64,
        )
        self.n_synapses = self.W.nnz
        self.sparsity = 1.0 - (self.n_synapses / (self.n_pre * self.n_post))
        return int(pruned_count)

    def grow_synapses(self, n_new: int, rng: Optional[np.random.Generator] = None) -> int:
        """
        神经发生/突触生成：添加新的突触连接
        （模拟人脑的神经发生过程）

        Args:
            n_new: 新增突触数
            rng: 随机数生成器

        Returns:
            grown_count: 实际新增的突触数
        """
        if rng is None:
            rng = np.random.default_rng()

        W_coo = self.W.tocoo()
        existing_set = set(zip(W_coo.row.tolist(), W_coo.col.tolist()))

        new_rows = []
        new_cols = []
        new_data = []

        attempts = 0
        max_attempts = n_new * 10

        while len(new_rows) < n_new and attempts < max_attempts:
            r = rng.integers(0, self.n_post)
            c = rng.integers(0, self.n_pre)
            if (r, c) not in existing_set:
                existing_set.add((r, c))
                new_rows.append(r)
                new_cols.append(c)
                new_data.append(rng.normal(0, 0.01))
            attempts += 1

        if new_rows:
            all_rows = np.concatenate([W_coo.row, new_rows])
            all_cols = np.concatenate([W_coo.col, new_cols])
            all_data = np.concatenate([W_coo.data, new_data])

            self.W = sparse.csr_matrix(
                (all_data, (all_rows, all_cols)),
                shape=(self.n_post, self.n_pre),
                dtype=np.float64,
            )
            self.n_synapses = self.W.nnz
            self.sparsity = 1.0 - (self.n_synapses / (self.n_pre * self.n_post))

        return len(new_rows)

    def state_dict(self) -> Dict[str, Any]:
        """序列化稀疏连接状态"""
        W_coo = self.W.tocoo()
        return {
            "n_pre": self.n_pre,
            "n_post": self.n_post,
            "synapses_per_neuron": self.synapses_per_neuron,
            "W_row": W_coo.row.copy(),
            "W_col": W_coo.col.copy(),
            "W_data": W_coo.data.copy(),
            "weight_init": self.weight_init,
            "weight_scale": self.weight_scale,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """从字典加载稀疏连接状态"""
        self.n_pre = state["n_pre"]
        self.n_post = state["n_post"]
        self.synapses_per_neuron = state["synapses_per_neuron"]
        self.W = sparse.csr_matrix(
            (state["W_data"], (state["W_row"], state["W_col"])),
            shape=(self.n_post, self.n_pre),
            dtype=np.float64,
        )
        self.n_synapses = self.W.nnz
        self.sparsity = 1.0 - (self.n_synapses / (self.n_pre * self.n_post))

    def get_stats(self) -> Dict[str, Any]:
        """获取连接统计信息"""
        W_coo = self.W.tocoo()
        return {
            "n_pre": self.n_pre,
            "n_post": self.n_post,
            "n_synapses": self.n_synapses,
            "sparsity": self.sparsity,
            "mean_weight": float(np.mean(np.abs(W_coo.data))),
            "max_weight": float(np.max(np.abs(W_coo.data))),
            "min_weight": float(np.min(np.abs(W_coo.data))),
        }


class BrainScaleConfig:
    """
    人脑规模配置

    根据人脑各区域参数设置对应的网络规模。
    参考：Digital Twin Brain (Fudan, 2024)
    """

    # 人脑各区域神经元数量（近似值）
    CEREBRAL_CORTEX = 16_000_000_000       # 大脑皮层 ~160亿
    CEREBELLUM = 69_000_000_000            # 小脑 ~690亿
    BASAL_GANGLIA = 1_000_000_000          # 基底节 ~10亿
    THALAMUS = 1_000_000_000               # 丘脑 ~10亿
    HIPPOCAMPUS = 15_000_000               # 海马体 ~1500万
    AMYGDALA = 10_000_000                  # 杏仁核 ~1000万
    BRAINSTEM = 100_000_000                # 脑干 ~1亿

    # NIEA模块到人脑区域的映射
    MODULE_MAPPING = {
        "snn_core": {
            "brain_region": "cerebral_cortex",
            "neurons": 16_000_000_000,
            "synapses_per_neuron": 7_000,
            "description": "SNN计算核心 → 大脑皮层",
        },
        "stdp": {
            "brain_region": "cerebral_cortex_synapses",
            "neurons": 16_000_000_000,
            "synapses_per_neuron": 7_000,
            "description": "STDP可塑性 → 皮层突触",
        },
        "pcn": {
            "brain_region": "predictive_cortex",
            "neurons": 8_000_000_000,
            "synapses_per_neuron": 5_000,
            "description": "预测编码 → 感觉皮层预测通路",
        },
        "world_model": {
            "brain_region": "prefrontal_cortex",
            "neurons": 2_000_000_000,
            "synapses_per_neuron": 10_000,
            "description": "世界模型 → 前额叶",
        },
        "active_inference": {
            "brain_region": "basal_ganglia_prefrontal",
            "neurons": 1_000_000_000,
            "synapses_per_neuron": 8_000,
            "description": "主动推理 → 基底节+前额叶",
        },
        "hippocampus": {
            "brain_region": "hippocampus",
            "neurons": 15_000_000,
            "synapses_per_neuron": 20_000,
            "description": "HDC+CMS → 海马体",
        },
        "visual": {
            "brain_region": "visual_cortex",
            "neurons": 140_000_000,
            "synapses_per_neuron": 5_000,
            "description": "视觉编码 → 视觉皮层(V1-V5)",
        },
        "auditory": {
            "brain_region": "auditory_cortex",
            "neurons": 100_000_000,
            "synapses_per_neuron": 5_000,
            "description": "听觉编码 → 听觉皮层",
        },
        "language": {
            "brain_region": "broca_wernicke",
            "neurons": 50_000_000,
            "synapses_per_neuron": 10_000,
            "description": "语言头 → 布罗卡区+韦尼克区",
        },
        "metacognition": {
            "brain_region": "dlpfc",
            "neurons": 1_000_000_000,
            "synapses_per_neuron": 8_000,
            "description": "元认知 → 背外侧前额叶",
        },
        "consciousness": {
            "brain_region": "thalamocortical",
            "neurons": 10_000_000_000,
            "synapses_per_neuron": 7_000,
            "description": "全局工作空间 → 丘脑-皮层系统",
        },
    }

    @classmethod
    def get_config(cls, module_name: str) -> Dict[str, Any]:
        """获取指定模块的人脑规模配置"""
        if module_name not in cls.MODULE_MAPPING:
            raise ValueError(f"未知模块: {module_name}, 可选: {list(cls.MODULE_MAPPING.keys())}")
        return cls.MODULE_MAPPING[module_name]

    @classmethod
    def get_total_neurons(cls) -> int:
        """获取总神经元数"""
        return sum(m["neurons"] for m in cls.MODULE_MAPPING.values())

    @classmethod
    def get_total_synapses(cls) -> int:
        """获取总突触数"""
        return sum(m["neurons"] * m["synapses_per_neuron"] for m in cls.MODULE_MAPPING.values())

    @classmethod
    def estimate_memory(cls) -> Dict[str, float]:
        """估算所需内存"""
        total_synapses = cls.get_total_synapses()
        # CSR格式: 每突触约12字节(8字节权重 + 4字节列索引)
        synapse_memory_gb = total_synapses * 12 / (1024**3)
        # 神经元状态: 每神经元约64字节(膜电位、阈值、自适应变量等)
        total_neurons = cls.get_total_neurons()
        neuron_memory_gb = total_neurons * 64 / (1024**3)
        total_gb = synapse_memory_gb + neuron_memory_gb

        return {
            "synapse_memory_tb": synapse_memory_gb / 1024,
            "neuron_memory_tb": neuron_memory_gb / 1024,
            "total_memory_tb": total_gb / 1024,
            "total_neurons": total_neurons,
            "total_synapses": total_synapses,
        }
