# NIEA - Neural Isomorphic Evolutionary Architecture

基于人脑认知架构的通用人工智能框架，实现从感知运动到博士级别的类人成长训练路径。

## 项目简介

NIEA 是一个脑启发AI框架，融合四大理论支柱与七大功能模块：

**四大理论支柱：**

1. SNN基底 -- 生物神经元的事件驱动计算基元（LIF/ALIF + STDP）
2. 预测编码学习 -- 层次化自由能最小化的世界模型（PCN）
3. 互补记忆系统 -- 海马体快速编码 + 皮层慢速巩固的双系统架构（HDC + CMS）
4. 全局工作空间意识涌现 -- 模块竞争广播与意识点火机制（Global Workspace）

**七大功能模块：**

1. 脉冲编码器 -- 速率/时间编码，将连续信号转换为脉冲序列（Welford在线归一化）
2. SNN计算核心 -- LIF/ALIF神经元 + STDP赫布可塑性
3. 预测编码网络（PCN） -- 层次化预测与误差校正
4. 超维记忆系统（HDC） -- 关联记忆、情景记忆、概念存储
5. 主动推理引擎 -- 期望自由能最小化、好奇心驱动探索（连续高斯变分推断）
6. 世界模型（WM） -- VAE世界模型 + 奖励预测 + 想象规划
7. 元认知与结构自进化模块 -- 不确定性估计 + 神经发生/突触修剪/结构可塑性

**类人成长训练路径：**

0~3岁感知运动期 -> 幼儿园 -> 小学 -> 初中 -> 高中 -> 本科 -> 硕士 -> 博士

## 环境要求

- Python >= 3.9
- numpy >= 1.22.0
- scipy >= 1.9.0

**可选依赖（按需安装）：**

| 依赖 | 用途 | 安装命令 |
|------|------|---------|
| torch >= 1.12 | GPU加速（SNN层、世界模型、主动推理） | `pip install torch` |
| pytest | 运行单元测试 | `pip install pytest` |
| openai | AI教师NPC（DeepSeek等） | `pip install openai` |

**安装：**

```bash
# 从 PyPI 安装
pip install SkyalyticAI

# 或从源码安装
pip install -e .
```

## 快速开始

### 1. 最简创建

```python
from skyalyticAI import NIEABrain

brain = NIEABrain(input_dim=10, action_dim=4)
```

### 2. 使用预设方案

```python
from skyalyticAI import NIEABrain

# 小规模（个人电脑）
brain = NIEABrain(input_dim=10, action_dim=4, brain_scale="small")

# 中规模（单卡GPU）
brain = NIEABrain(input_dim=10, action_dim=4, brain_scale="medium")

# 大规模（多卡GPU）
brain = NIEABrain(input_dim=10, action_dim=4, brain_scale="large")

# 超大规模（GPU集群）
brain = NIEABrain(input_dim=10, action_dim=4, brain_scale="xlarge")

# 人脑规模（神经形态芯片/超大规模集群）
brain = NIEABrain(input_dim=10, action_dim=4, brain_scale="human")
```

### 3. 预设 + 自定义覆盖

```python
# 使用large预设，但覆盖hidden_dim
brain = NIEABrain(
    input_dim=10,
    action_dim=4,
    brain_scale="large",
    override_hidden_dim=16384,  # 覆盖预设的8192
)
```

### 4. 完全自定义参数

```python
brain = NIEABrain(
    input_dim=128,
    hidden_dim=2048,
    action_dim=10,
    n_observations=2048,
    hd_dim=32000,
    pcn_hidden_dim=1024,
    world_model_hidden_dim=1024,
    ai_hidden_dim=1024,
    sparse=True,
    synapses_per_neuron=1000,
)
```

### 5. 启动训练

```bash
python scripts/run_growth_training.py
```

训练脚本会交互式选择成长线（human/social/society）、起始学段等。

### 6. AI教师NPC配置

通过环境变量配置外部AI教师服务（如DeepSeek）：

```bash
# Linux/macOS
export NIEA_TEACHER_API_BASE=https://api.deepseek.com
export NIEA_TEACHER_API_KEY=your-api-key
export NIEA_TEACHER_MODEL=deepseek-v4-flash

# Windows PowerShell
$env:NIEA_TEACHER_API_BASE = "https://api.deepseek.com"
$env:NIEA_TEACHER_API_KEY = "your-api-key"
$env:NIEA_TEACHER_MODEL = "deepseek-v4-flash"
```

## 参数说明

### brain_scale 预设方案

| 预设 | 适用机型 | hidden_dim | hd_dim | pcn_hidden | wm_hidden | ai_hidden | n_obs | spike_steps | syn/neuron | 稀疏连接 | 记忆容量 | 估算内存 |
|------|---------|-----------|--------|------------|-----------|-----------|-------|-------------|------------|---------|---------|---------|
| small | PC/笔记本（CPU） | 256 | 10,000 | 128 | 256 | 128 | 256 | 20 | 100 | 否 | 2K/20K | ~50MB |
| medium | 单卡GPU（RTX 3090/4090, 24GB） | 2,048 | 32,000 | 1,024 | 1,024 | 1,024 | 2,048 | 50 | 1,000 | 否 | 2K/20K | ~2GB |
| large | 多卡GPU（A100 80GB x 4~8） | 8,192 | 128,000 | 4,096 | 4,096 | 4,096 | 8,192 | 80 | 3,000 | 是 | 15M/16B | ~32GB |
| xlarge | GPU集群（A100/H100 x 32+） | 65,536 | 1,000,000 | 32,768 | 32,768 | 32,768 | 65,536 | 100 | 5,000 | 是 | 15M/16B | ~1TB |
| human | 神经形态芯片/超大规模集群 | 16B | 15M | 8B | 2B | 1B | 16B | 100 | 7,000 | 是 | 15M/16B | ~4,000TB |

> **记忆容量**列格式为"海马体容量/皮层容量"。small/medium 预设使用小容量（2K/20K），large/xlarge/human 使用大容量（15M/16B）。

### NIEABrain 完整参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| input_dim | int | 10 | 感官输入维度 |
| hidden_dim | int | 256 | SNN和PCN的隐藏层维度 |
| action_dim | int | 4 | 离散动作数量 |
| n_observations | int | 10 | 主动推理的离散观测数 |
| hd_dim | int | 10000 | HDC记忆向量维度 |
| pcn_hidden_dim | int | 128 | 预测编码网络隐藏维度 |
| world_model_hidden_dim | int | 256 | 世界模型隐藏维度 |
| world_model_n_layers | int | 4 | 世界模型层数 |
| n_snn_layers | int | 3 | SNN层数 |
| ai_hidden_dim | int | 128 | 主动推理引擎隐藏维度 |
| development_stages | bool | True | 是否模拟发展阶段转换 |
| spike_encoding_steps | int | 20 | 脉冲编码时间步数 |
| surprise_threshold | float | 0.5 | 惊喜阈值（记忆存储触发）。默认0.5会被自动缩放为 0.5/sqrt(hidden_dim) |
| consolidation_interval | int | 100 | 记忆巩固间隔步数 |
| consolidation_batch_size | int | 20 | 记忆巩固批量大小 |
| sparse | bool | False | 是否使用稀疏连接 |
| synapses_per_neuron | int | 7000 | 每神经元突触数（稀疏模式） |
| brain_scale | str/bool | False | 预设方案：small/medium/large/xlarge/human/True/False |
| language_vocab_size | int | None | 语言头词表大小（同时创建ASR/OCR头） |
| visual_encoder | VisualEncoder | None | 自定义视觉编码器实例 |
| audio_encoder | AudioEncoder | None | 自定义听觉编码器实例 |
| multimodal_fusion | MultimodalFusion | None | 自定义多模态融合实例 |
| device | str | None | 计算设备（cuda/cpu） |

### override 参数（覆盖预设值）

当使用 brain_scale 预设时，可通过 override 参数覆盖预设中的特定值：

| override 参数 | 覆盖的预设参数 |
|--------------|--------------|
| override_hidden_dim | hidden_dim |
| override_hd_dim | hd_dim |
| override_pcn_hidden_dim | pcn_hidden_dim |
| override_world_model_hidden_dim | world_model_hidden_dim |
| override_ai_hidden_dim | ai_hidden_dim |
| override_n_observations | n_observations |
| override_spike_encoding_steps | spike_encoding_steps |
| override_synapses_per_neuron | synapses_per_neuron |

**优先级：override_xxx > brain_scale预设 > 参数默认值**

### 各机型推荐配置

**个人电脑（8~16GB内存，无GPU）：**

```python
brain = NIEABrain(input_dim=10, action_dim=4, brain_scale="small")
```

**游戏本（16~32GB内存，RTX 3060/4060, 8~12GB显存）：**

```python
brain = NIEABrain(
    input_dim=128, action_dim=10,
    brain_scale="small",
    override_hidden_dim=512,
    override_synapses_per_neuron=200,
)
```

**工作站（64GB内存，RTX 3090/4090, 24GB显存）：**

```python
brain = NIEABrain(input_dim=128, action_dim=10, brain_scale="medium")
```

**服务器（A100 80GB x 4~8）：**

```python
brain = NIEABrain(input_dim=256, action_dim=20, brain_scale="large")
```

**GPU集群（A100/H100 x 32+）：**

```python
brain = NIEABrain(input_dim=512, action_dim=50, brain_scale="xlarge")
```

## 项目结构

```
skyalyticAI/
  skyalyticAI/                      # 核心包
    neurons/                     # 神经元模型（LIF, ALIF, SNN层, 稀疏连接）
    plasticity/                  # 突触可塑性（STDP）
    predictive_coding/           # 预测编码网络（PCN）
    memory/                      # 记忆系统（HDC, 巩固）
    active_inference/            # 主动推理引擎
    world_model/                 # 世界模型（VAE）
    metacognition/               # 元认知模块
    consciousness/               # 全局工作空间（意识涌现）
    evolution/                   # 结构自进化
    language/                    # 语言头（文本编码, 语言生成, ASR, OCR）
    perception/                  # 感知编码器（视觉, 听觉, 多模态融合）
    society/                     # 社会模拟世界
    npc/                         # AI教师NPC
    training/                    # 训练器（基础训练, 类人成长训练）
    exams/                       # 考试系统
    data/                        # 数据管理
    gpu/                         # GPU加速工具
    brain.py                     # NIEABrain 主类 + BrainScalePresets
  scripts/
    run_growth_training.py       # 一键训练启动器
    seed_curriculum_corpus.py    # 语料种子脚本
    validate_industrial_ready.py # 工业级验证脚本
    fetch_public_corpus.py       # 公开语料获取
    collect_open_knowledge_cn.py # 中文开放知识收集
  理论.md                        # 完整理论文档
```

## 注意事项

1. **内存需求**：brain_scale="human" 需要约4,000TB内存，仅适用于神经形态芯片或超大规模分布式集群。普通机器请使用 small/medium 预设。记忆容量按预设级别分配：small/medium 使用小容量（海马体2K/皮层20K），large/xlarge/human 使用大容量（海马体15M/皮层16B）。

2. **GPU加速**：以下模块支持PyTorch CUDA加速（需安装 `torch>=1.12`）：
   - SNN层：前向传播和批量前向传播（稀疏连接模式除外）
   - 世界模型：训练、批量训练（PyTorch autograd + Adam优化器）
   - 主动推理：感知（Jacobian计算）、认知/实用价值计算、动作选择
   - GPU工具：`GPUBatchProcessor` 提供批量矩阵乘法、外积、卷积、FFT、softmax等GPU操作

   通过 `device="cuda"` 参数启用GPU加速。稀疏连接模式（sparse=True）使用scipy CSR格式，在CPU上运行。large及以上预设自动启用稀疏连接。无GPU时自动回退到CPU（numpy）路径。

3. **AI教师NPC**：社会模拟模式（society）需要配置外部AI教师服务。通过环境变量设置API密钥，代码中不会硬编码任何密钥。

4. **训练数据**：本框架不依赖大规模预训练语料。训练通过AI教师NPC交互进行，模拟人类从0岁开始的学习过程。

5. **Checkpoint**：训练过程自动保存checkpoint，支持断点续训。checkpoint保存在 `checkpoints/` 目录。checkpoint包含运行统计量（Welford归一化参数），确保断点续训时输入归一化行为一致。

6. **发展阶段**：brain.development_stages=True 时，Brain会根据经验步数自动推进发展阶段（sensorimotor -> phd）。

7. **稀疏连接**：当 brain_scale 为 large/xlarge/human 时，自动启用稀疏连接（sparse=True），每个神经元仅连接有限数量的其他神经元，匹配人脑的稀疏连接模式。

8. **向后兼容**：brain_scale=True 等同于 brain_scale="human"。

9. **输入归一化**：脉冲编码器使用Welford在线算法计算输入的运行均值和方差，实现自适应归一化。归一化参数随checkpoint保存和恢复。

10. **语言功能**：设置 `language_vocab_size` 后，Brain自动创建语言头（speech）、ASR头（语音识别）和OCR头（文字识别），分别通过 `learn_speech()`、`learn_asr()`、`learn_ocr()` 训练。

## 开源协议

本项目采用 **CC BY-NC-SA 4.0**（知识共享 署名-非商业性使用-相同方式共享 4.0 国际）协议。

- 开源：源代码完全开放
- 可修改：允许修改和衍生
- 可下载：允许自由下载和分发
- 可提交审查：欢迎提交Issue和Pull Request
- 不可商用：禁止商业使用

详见 [LICENSE](LICENSE) 文件。

## 贡献

欢迎提交Issue和Pull Request参与贡献。
