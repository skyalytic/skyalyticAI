# SkyalyticAI API 参考

> pip install skyalyticai 后可通过 `from skyalyticAI.xxx import Xxx` 调用

---

## 1. 核心大脑 — `skyalyticAI.brain`

### class DevelopmentStage
发育阶段枚举（sensorimotor → phd）
- `get_stage(experience_steps) -> str` 根据步数返回阶段

### class BrainScalePresets
大脑规模预设（small/medium/large/xlarge/human）
- `get(preset) -> dict` 返回预设配置

### class NIEABrain
核心大脑类，集成 SNN + PCN + HDC + 主动推理 + 元认知 + 全局工作空间

**感知与思考**
- `perceive(observation) -> (hidden, prediction_error, surprise)` 感知输入
- `perceive_multimodal(visual, audio, raw_observation) -> (fused, pred_error, surprise)` 多模态感知
- `think(hidden, action) -> (new_hidden, intrinsic_reward)` 思考一步
- `speak(hidden_state) -> int` 说话（输出字符索引）
- `asr_decode(hidden_state) -> int` 语音识别解码
- `ocr_decode(hidden_state) -> int` 文字识别解码

**学习**
- `learn(hidden, action, next_hidden, reward, prediction_error, env_reward) -> dict` 综合学习
- `learn_speech(hidden_state, target_index, reward) -> dict` 语言学习（fix 118 教师强制）
- `learn_asr(hidden_state, target_index, reward) -> dict` ASR 学习
- `learn_ocr(hidden_state, target_index, reward) -> dict` OCR 学习
- `develop() -> None` 结构发育（突触修剪/神经元生长）

**记忆检索**
- `query_memory_action(state) -> int or None` 查询 HDC 记忆中最相似状态的关联动作（论文第4章 HDC 检索准确率评估）

**阶段控制**
- `set_school_stage(stage) -> None` 设置学段

**状态管理**
- `reset() / reset_episode() -> None` 重置
- `state_dict() -> dict` 序列化
- `load_state_dict(state) -> None` 反序列化
- `get_state_summary() / get_brain_stats() -> dict` 状态摘要
- `save_normalization_state() / restore_normalization_state(state)` 归一化状态

---

## 2. 神经元模型 — `skyalyticAI.neurons`

### class LIFNeuron (`lif.py`)
LIF 脉冲神经元
- `step(current, dt) -> (voltage, spike)` 单步更新
- `forward(inputs) -> spikes` 前向传播
- `reset() / state_dict() / load_state_dict(state)`

### class ALIFNeuron (`alif.py`)
自适应 LIF 神经元（继承 LIFNeuron）

### class SNNLayer (`snn_layer.py`)
SNN 网络层
- `forward(x) -> spikes` 前向传播
- `reset() / state_dict() / load_state_dict(state)`

### class SparseConnectivity (`sparse_connectivity.py`)
稀疏连接矩阵
- `get_dense() -> np.ndarray` 转稠密
- `apply(weights) -> np.ndarray` 应用稀疏掩码

### class BrainScaleConfig (`sparse_connectivity.py`)
大脑规模配置

---

## 3. 可塑性 — `skyalyticAI.plasticity`

### class STDPSynapse (`stdp.py`)
STDP 突触可塑性
- `update(pre_spike, post_spike, weight) -> float` 更新权重
- `reset() / state_dict() / load_state_dict(state)`

### class STDPLayer (`stdp_layer.py`)
STDP 网络层
- `forward(x) -> spikes` 前向传播
- `update_stdp(pre_spikes, post_spikes) -> None` STDP 更新
- `reset() / state_dict() / load_state_dict(state)`

### class STDPVariant (`stdp.py`)
STDP 变体枚举（STANDARD, ANTI_STDP, etc.）

---

## 4. 预测编码 — `skyalyticAI.predictive_coding`

### class PCNLayer (`pcn_layer.py`)
预测编码网络层
- `forward(x) -> (prediction, error)` 前向传播
- `update(x, target, lr) -> float` 更新权重
- `reset() / state_dict() / load_state_dict(state)`

### class PredictiveCodingNetwork (`pcn.py`)
预测编码网络（多层 PCN）
- `forward(x) -> (prediction, errors)` 前向传播
- `compute_surprise(x) -> float` 计算预测误差（surprise）
- `learn(x, target, lr) -> dict` 学习
- `reset() / state_dict() / load_state_dict(state)`

---

## 5. 记忆系统 — `skyalyticAI.memory`

### class HDCMemory (`hdc.py`)
超维计算记忆（海马体快速编码）
- `random_vector() -> np.ndarray` 生成随机超维向量
- `add_concept(name, vector) -> np.ndarray` 添加概念
- `get_concept(name) -> np.ndarray` 获取概念
- `store_association(key_name, value_name) -> np.ndarray` 存储关联
- `retrieve_association(key_name) -> (value_name, similarity)` 检索关联
- `retrieve(query, top_k) -> List[Tuple]` 检索最相似
- `store_episode(sequence) -> np.ndarray` 存储情节
- `query_episode(partial) -> List` 查询情节
- `bundle(*vectors) / bind(a, b) / unbind(bound, key) / permute(v, shift)` 超维运算
- `reset() / state_dict() / load_state_dict(state)`

### class HippocampalStore (`consolidation.py`)
海马体存储（快速编码、短期保持）
- `encode(pattern) -> np.ndarray` 编码模式
- `retrieve(query, top_k) -> List` 检索
- `get_consolidation_candidates() -> List` 获取待巩固项
- `mark_consolidated(index) -> None` 标记已巩固
- `decay_all() -> None` 衰减
- `size() / utilization() -> int / float`

### class CorticalStore (`consolidation.py`)
皮层存储（慢速巩固、长期保持）
- `consolidate(hippocampal_store) -> dict` 从海马体巩固
- `pattern_completion(partial) -> np.ndarray` 模式补全
- `retrieve_semantic(query) -> np.ndarray` 语义检索
- `size() -> int`

### class ComplementaryMemorySystem (`consolidation.py`)
互补记忆系统（CMS = 海马体 + 皮层）
- `store(key, pattern) -> None` 存储
- `retrieve(query) -> np.ndarray` 检索
- `complete_pattern(partial_key) -> np.ndarray` 模式补全
- `consolidate() -> dict` 巩固
- `reset() / state_dict() / load_state_dict(state)`

---

## 6. 感知编码器 — `skyalyticAI.perception`

### class RetinaEncoder (`retina_encoder.py`)
视网膜编码器（零参数，fix 119）
- `encode(image) -> np.ndarray` 图像 → 脉冲编码
- `encode_spikes(image) -> np.ndarray` 图像 → 脉冲序列

### class CochleaEncoder (`cochlea_encoder.py`)
耳蜗编码器（零参数，fix 119）
- `encode(audio, input_sample_rate) -> np.ndarray` 声波 → 平均发放率
- `encode_spikes(audio, input_sample_rate) -> np.ndarray` 声波 → 脉冲序列
- `_resample(wave, input_sample_rate) -> np.ndarray` 自动重采样到 16kHz

### class AudioEncoder (`audio_encoder.py`)
音频编码器（封装 CochleaEncoder）
- `encode(audio) -> np.ndarray` 音频 → 特征向量
- `state_dict() / load_state_dict(state)`

### class VisualEncoder (`visual_encoder.py`)
视觉编码器（封装 RetinaEncoder）
- `encode(image) -> np.ndarray` 图像 → 特征向量
- `state_dict() / load_state_dict(state)`

### class MultimodalFusion (`multimodal_fusion.py`)
多模态融合
- `fuse(features: Dict[str, np.ndarray]) -> np.ndarray` 融合多模态特征
- `state_dict() / load_state_dict(state)`

---

## 7. 意识 — `skyalyticAI.consciousness`

### class GlobalWorkspace (`__init__.py`)
全局工作空间（注意力竞争 + 广播）
- `submit_bid(module_name, bid_value, output) -> None` 提交竞价
- `compete() -> dict` 竞争胜出者
- `learn(winner_idx, module_output, broadcast, lr) -> None` 学习
- `get_broadcast() -> np.ndarray` 获取广播信号
- `get_consciousness_level() -> float` 意识水平
- `reset() / reset_all() / state_dict() / load_state_dict(state)`

---

## 8. 主动推理 — `skyalyticAI.active_inference`

### class ActiveInferenceAgent (`agent.py`)
主动推理智能体（预测编码 + 自由能最小化）
- `perceive(observation) -> np.ndarray` 感知
- `predict_transition(state, action) -> np.ndarray` 预测状态转移
- `predict_observation(state) -> np.ndarray` 预测观测
- `expected_free_energy(state, action) -> float` 期望自由能
- `select_action() -> (action, belief, posterior)` 选择动作
- `learn_transition(state, action, next_state) -> dict` 学习转移模型
- `learn_observation(state, observation) -> dict` 学习观测模型
- `step(observation) -> int` 一步交互
- `get_belief_entropy() -> float` 信念熵
- `set_preferences(preferences) -> None` 设置偏好
- `reset() / reset_all() / state_dict() / load_state_dict(state)`

---

## 9. 世界模型 — `skyalyticAI.world_model`

### class WorldModel (`world_model.py`)
世界模型（状态编码 + 转移预测 + 奖励预测）
- `encode(obs) -> (state, mu, logvar)` 编码观测
- `encode_deterministic(obs) -> np.ndarray` 确定性编码
- `predict_next_state(state, action) -> np.ndarray` 预测下一状态
- `decode(state) -> np.ndarray` 解码观测
- `predict_reward(state) -> float` 预测奖励
- `imagine_trajectory(state, actions) -> List` 想象轨迹
- `train_step(batch) -> dict` 训练一步
- `train_step_batch(batch) -> dict` 批量训练
- `reset() / state_dict() / load_state_dict(state)`

---

## 10. 元认知 — `skyalyticAI.metacognition`

### class MetacognitiveModule (`metacognition.py`)
元认知模块（知识边界评估 + 注意力建议）
- `forward(state_vector) -> dict` 前向传播
- `evaluate_knowledge_boundary(state) -> dict` 评估知识边界
- `update_meta_knowledge(state, outcome) -> None` 更新元知识
- `suggest_attention(input_features) -> np.ndarray` 注意力建议
- `get_metacognitive_state() -> dict` 元认知状态
- `reset() / reset_all() / state_dict() / load_state_dict(state)`

---

## 11. 结构进化 — `skyalyticAI.evolution`

### class StructuralEvolution (`__init__.py`)
结构自进化（突触修剪 + 神经元生长 + 连接重连）
- `should_evolve() -> bool` 是否应进化
- `record_performance(performance) -> None` 记录性能
- `record_neuron_activity(activity) -> None` 记录神经元活动
- `detect_plateau(window) -> bool` 检测性能平台期
- `prune_weights(weights, threshold) -> np.ndarray` 修剪权重
- `grow_neurons(layer, n_new) -> None` 生长神经元
- `rewire_connections(connectivity) -> None` 重连连接
- `evolve_module(module) -> dict` 进化模块
- `get_evolution_summary() -> dict` 进化摘要
- `state_dict() / load_state_dict(state)`

---

## 12. 语言 — `skyalyticAI.language`

### class LanguageHead (`language_head.py`)
语言头（布罗卡区/韦尼克区，fix 118 教师强制）
- `forward(hidden_state) -> np.ndarray` 前向传播（输出概率分布）
- `learn(hidden_state, target_index, reward) -> dict` 学习（教师强制）
- `state_dict() / load_state_dict(state)`

### class TextEncoder (`text_encoder.py`)
文本编码器
- `encode(context_indices) -> np.ndarray` 编码上下文

---

## 13. 环境 — `skyalyticAI.env`

### class Environment (`environment.py`) — ABC
环境基类
- `reset() -> Any` 重置
- `step(action) -> (obs, reward, done, info)` 一步交互
- `get_observation_dim() -> int` 观测维度
- `get_action_dim() -> int` 动作维度
- `render() / close() / seed(seed) / unwrapped()`

### class HumanGrowthWorld (`curriculum_world.py`)
类人成长环境（迷宫 + 读书 + 考试）
- `set_stage(stage) -> None` 设置学段
- `get_steps_per_episode() -> int` 每 episode 步数
- `set_rolling_speech_accuracy(acc) / set_rolling_subject_accuracy(subject_acc)`
- `reset() / step(action) / get_observation_dim() / get_action_dim() / render()`

### class SocialClassroomWorld (`social_classroom_world.py`)
社会课堂环境（多角色持续对话）
- 继承 HumanGrowthWorld

### class GridWorldEnv (`grid_world.py`)
网格世界环境
- `reset() / step(action) / get_observation_dim() / get_action_dim() / render()`

---

## 14. 社会模拟器 — `skyalyticAI.society`

### class SocietySimWorld (`sim_world.py`)
工业级社会模拟器（多模态 + 多智能体 + 课程体系）
- `reset() -> dict` 重置
- `step(action) -> (obs, reward, done, info)` 一步交互（fix 120 多轮问答）
- `get_observation_dim() / get_action_dim() -> int`
- `set_stage(stage) -> None` 设置学段

**构造参数（均可调）**：
- `corpus_root` 语料库路径
- `observation_dim=128` 观测维度
- `school_stage="sensorimotor"` 起始学段
- `max_stage="undergraduate"` 封顶学段
- `image_size=28` 图像尺寸
- `audio_len=96000` 音频长度（6秒@16kHz，可调大）
- `real_perception=True` 是否启用真实感知
- `seed` 随机种子

### class SpeechSynthesizer (`speech_synth.py`)
TTS 语音合成器（pyttsx3，子进程隔离）
- `synthesize(text) -> (wave, sample_rate) or None` 合成语音
- `_check_available() -> bool` 检查可用性

### class DaySlot (`sim_world.py`)
时段枚举（MORNING_HOME, SCHOOL_CLASS, ...）

### class SocietyState (`sim_world.py`)
社会状态

---

## 15. NPC 老师 — `skyalyticAI.npc`

### class TeacherNPC (`teacher_npc.py`)
NPC 老师（多角色、多学段、课程生成）
- `pick_persona(stage, subject) -> dict` 选择角色
- `sample_subjects(stage, default) -> List[str]` 抽取科目
- `sample_teaching_line(stage, subject) -> str` 采样教学文本
- `make_reading_item(stage, subject) -> ReadingItem` 生成阅读理解题
- `make_reasoning_chain(stage, subject) -> List[ReasoningStep]` 生成推理链
- `bootstrap_vocab_text() -> str` 引导词表文本

### class TeacherService (`teacher_service.py`)
LLM API 服务（DeepSeek 等，有降级）
- `from_env() -> TeacherService or None` 从环境变量创建
- `chat(system, user, temperature, max_tokens) -> str` 调用 LLM

### class Persona (`persona_registry.py`)
NPC 角色定义
- `build_personas() -> List[Persona]` 构建角色列表
- `persona_to_dict(p) -> dict` 转字典

### class ReadingItem / ReasoningStep (`teacher_npc.py`)
阅读理解题 / 推理步骤数据结构

---

## 16. 数据管理 — `skyalyticAI.data`

### class CorpusManager (`corpus_manager.py`)
语料库管理器
- `vocab_len() -> int` 词表大小
- `index_to_char(idx) / char_to_index(char)` 索引转换
- `get_exam_lines(stage) -> List[str]` 获取考试数据
- `list_subjects(stage) -> List[str]` 列出科目
- `stats() -> dict` 语料统计

### class StageQualitySpec (`education_config.py`)
学段质量规格
- `get_quality_spec(stage) -> StageQualitySpec` 获取规格
- `next_stage(current) -> str` 下一学段
- `is_school_stage(stage) -> bool` 是否是学段
- `core_subjects(stage) / subjects_for_stage(stage) -> List[str]`
- `core_subject_min_accuracy(stage, subject) -> float`

### class Dataset / NIEADataLoader / ExperienceReplayDataset / MultimodalDataset (`__init__.py`)
数据集基类 / 数据加载器 / 经验回放 / 多模态数据集
- `Dataset`: `get_observation_dim() / get_action_dim() / get_modality_keys()`
- `NIEADataLoader`: `epoch -> int`
- `ExperienceReplayDataset`: `add(transition) / add_transition(...) / sample_batch(n) / clear()`
- `MultimodalDataset`: `add_sample(...) / state_dict() / load_state_dict(state)`

---

## 17. 考试系统 — `skyalyticAI.exams`

### class ExamSuite (`exam_suite.py`)
考试套件（3 类题型集成）
- `reset() / step(action) / get_observation_dim() / get_action_dim() / render()`

### class CharPredictionExam (`char_prediction_exam.py`)
字符预测考试
- `reset() / step(action) / accuracy() / passed() / get_observation_dim() / get_action_dim()`

### class ReadingComprehensionExam (`reading_comprehension_exam.py`)
阅读理解考试
- `reset() / step(action) / accuracy() / passed() / get_observation_dim() / get_action_dim()`

### class MultiStepReasoningExam (`multi_step_reasoning_exam.py`)
多步推理考试
- `reset() / step(action) / accuracy() / passed() / get_observation_dim() / get_action_dim()`

### class ExamType (`exam_suite.py`)
考试类型枚举

---

## 18. 训练器 — `skyalyticAI.training`

### class NIEATrainer (`trainer.py`)
基础训练器
- `train() -> dict` 训练
- `_run_episode(episode) -> dict` 跑一个 episode
- `_save_checkpoint(path) / _load_checkpoint(path)` 存取 checkpoint

### class HumanGrowthTrainer (`human_growth_trainer.py`)
类人成长训练器（继承 NIEATrainer）
- `train() -> dict` 训练（含升学/遗忘回测逻辑）
- `_run_episode(episode) -> dict` 跑一个 episode（fix 120 多轮问答）

### class TrainingQualityGate (`training_quality.py`)
训练质量门控
- 检查 `rolling_speech_accuracy / forgetting_threshold` 等

### class AcceptanceReportBuilder (`acceptance_report.py`)
验收报告生成器
- `evaluate_stage_exam(trainer, stage) -> (exam_acc, hdc_acc)` 评估学段考试与 HDC 检索准确率（返回元组）
- `evaluate_retention(trainer, stage) -> List[RetentionResult]` 评估遗忘
- `should_rollback_promotion(retention) -> bool` 是否回滚升学
- `generate_report(trainer) -> dict` 生成完整验收报告

### class RetentionResult (`acceptance_report.py`)
遗忘回测结果
- `accuracy: float` 考试准确率
- `historical_best: float` 历史最佳
- `forgetting: float` 遗忘量
- `hdc_retrieval_accuracy: float` HDC 检索准确率（论文第4章数据点）

### 文本指标 (`text_metrics.py`)
- `edit_distance(a, b) -> int` 编辑距离
- `cer(pred, target) -> float` 字符错误率
- `wer(pred, target) -> float` 词错误率

---

## 19. GPU 加速 — `skyalyticAI.gpu`

### 函数 (`__init__.py`)
- `is_gpu_available() -> bool` GPU 是否可用
- `get_device() -> Any` 获取设备
- `to_tensor(array, device) -> Any` NumPy → Tensor
- `to_numpy(tensor) -> np.ndarray` Tensor → NumPy
- `get_gpu_info() -> str` GPU 信息

### class GPUBatchProcessor (`__init__.py`)
GPU 批处理器
- `batch_matmul(a, b) / batch_outer_product(a, b)` 矩阵运算
- `batch_conv2d(x, kernel) / batch_fft(x)` 卷积/FFT
- `batch_softmax(x) -> np.ndarray` Softmax
- `transfer_weights_to_gpu(weights) / transfer_weights_from_gpu()` 权重传输

---

## 20. 设备后端抽象 — `skyalyticAI.device`

多后端硬件检测（Loihi / CUDA / NPU / MLU / XPU / CPU）与 SNN 层工厂模式。当对应依赖未安装时自动降级到 CPU。

### 函数 (`detector.py`)
- `detect_backend() -> str` 检测最优后端（返回 "loihi"/"cuda"/"npu"/"mlu"/"xpu"/"cpu"）
- `get_device() -> Any` 获取对应后端的 device 对象（如 `torch.device("cuda")`）
- `get_backend_info() -> str` 后端可读描述（如 "CUDA: NVIDIA RTX 5070 (11.9 GB)"）
- `is_backend_available(name) -> bool` 指定后端是否可用

### 函数 (`backend_factory.py`)
- `create_snn_layer(backend, input_dim, output_dim, **kwargs) -> SNNLayer or LoihiSNNLayer` 根据后端创建 SNN 层实例（Loihi 后端抛 `NotImplementedError`）

### class LoihiSNNLayer (`backend_factory.py`)
Loihi 神经形态芯片 SNN 层占位类（Intel Loihi 未实际接入时使用）
- `forward(*args, **kwargs)` / `step(*args, **kwargs)` / `reset()`

---

## 快速参考：可调参数汇总

| 参数 | 所在类 | 默认值 | 说明 |
|------|--------|--------|------|
| `brain_scale` | NIEABrain | "small" | 大脑规模预设 |
| `hidden_dim` | BrainScalePresets | 256(small) | 隐藏层维度 |
| `hd_dim` | BrainScalePresets | 10000 | HDC 超维维度 |
| `audio_len` | SocietySimWorld | 96000 | 音频长度（6秒@16kHz） |
| `image_size` | SocietySimWorld | 28 | 图像尺寸 |
| `observation_dim` | SocietySimWorld | 128 | 观测维度 |
| `episodes` | run_growth_training.py | 5000 | 训练回合数 |
| `steps_per_episode` | run_growth_training.py | 220 | 每回合步数 |
| `checkpoint_interval` | run_growth_training.py | 20 | 存档间隔 |
| `forgetting_threshold` | HumanGrowthTrainer | 0.05 | 遗忘阈值 |
| `sample_rate` | CochleaEncoder | 16000 | 耳蜗工作采样率 |
| `n_fibers` | CochleaEncoder | 64 | 听神经纤维数 |
| `spike_steps` | CochleaEncoder | 20 | 脉冲时间步 |
| `vocab_size` | CorpusManager | 512 | 词表大小 |
| `exam_holdout_ratio` | CorpusManager | 0.12 | 考试集比例 |
