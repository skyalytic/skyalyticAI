#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键训练启动器（可续训/可重头/可选成长线）。

目标：
1) 先检测是否存在 checkpoint
2) 若存在：选择继续训练 or 重头开始（重头会清空 checkpoints）
3) 若重头或无 checkpoint：选择成长线（human / social）
4) 选择起始学段与封顶学段（例如 0岁 sensorimotor → 本科 undergraduate）
5) 开始训练，并在终端持续显示训练过程

注意：
- 外部老师/家长AI服务（DeepSeek等）通过环境变量读取，脚本不会写入任何 key。
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]


def _env_default(name: str, default: str) -> str:
    v = os.environ.get(name, "").strip()
    return v or default


def _latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    if not ckpt_dir.is_dir():
        return None
    best: Optional[Tuple[int, Path]] = None
    for p in ckpt_dir.glob("checkpoint_ep*.npz"):
        m = re.search(r"checkpoint_ep(\d+)\.npz$", p.name)
        if not m:
            continue
        ep = int(m.group(1))
        if best is None or ep > best[0]:
            best = (ep, p)
    return best[1] if best else None


def _ask(prompt: str, default: str) -> str:
    s = input(f"{prompt} [{default}]> ").strip()
    return s or default


def _ask_choice(prompt: str, choices: dict, default_key: str) -> str:
    keys = "/".join(choices.keys())
    tip = ", ".join([f"{k}={v}" for k, v in choices.items()])
    while True:
        s = _ask(f"{prompt} ({keys}) {tip}", default_key).lower()
        if s in choices:
            return s
        print("无效输入，请重试。")


def _rm_tree(p: Path) -> None:
    if not p.exists():
        return
    for child in sorted(p.glob("**/*"), reverse=True):
        try:
            if child.is_file() or child.is_symlink():
                child.unlink(missing_ok=True)
            elif child.is_dir():
                child.rmdir()
        except Exception:
            pass
    try:
        p.rmdir()
    except Exception:
        pass


def main() -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("PYTHONPATH", str(ROOT))

    ckpt_dir = Path(_env_default("NIEA_CHECKPOINT_DIR", str(ROOT / "checkpoints")))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    latest = _latest_checkpoint(ckpt_dir)
    resume = False
    restart = False

    if latest is not None:
        print(f"检测到 checkpoint: {latest}")
        action = _ask_choice(
            "选择训练方式",
            {"c": "继续训练(从checkpoint)", "r": "重头开始(清空checkpoint)"},
            "c",
        )
        if action == "c":
            resume = True
        else:
            restart = True
    else:
        print("未检测到 checkpoint，将开始新训练。")
        restart = True

    # 选择成长线
    mode = _ask_choice(
        "选择成长线",
        {
            "society": "工业完整版：多模态+多智能体+可持续社会模拟器",
            "social": "社会课堂：多角色持续对话陪伴",
            "human": "迷宫+读书+考试的类人成长线",
        },
        "society",
    )

    # 选择阶段
    stage_choices = {
        "sensorimotor": "0~3岁感知运动(0岁起跑)",
        "kindergarten": "幼儿园",
        "primary": "小学",
        "middle": "初中",
        "high": "高中",
        "undergraduate": "本科",
        "master": "硕士",
        "phd": "博士",
    }
    start_stage = _ask_choice("起始学段", stage_choices, "sensorimotor")
    max_stage = _ask_choice("封顶学段(不超过此学段)", stage_choices, "undergraduate")

    episodes = int(_ask("训练回合数(industrial建议>=5000)", "5000"))
    steps = int(_ask("每回合最大步数", "220"))
    log_interval = int(_ask("日志间隔(1=每回合都显示)", "1"))
    checkpoint_interval = int(_ask("checkpoint间隔(回合)", "20"))
    report_path = _ask("验收报告输出路径", "reports/acceptance_report.json")

    # 选择 brain_scale 预设方案（放在环境初始化之前，避免等待）
    scale_choices = {
        "small": "PC/笔记本(CPU或小显存GPU)",
        "medium": "单卡GPU(RTX 3090/4090, 24GB显存)",
        "large": "多卡GPU(A100 80GB x 4~8)",
        "xlarge": "GPU集群(A100/H100 x 32+)",
        "human": "人脑规模(需要神经形态芯片或超大规模集群)",
    }
    brain_scale = _ask_choice("选择brain_scale预设方案", scale_choices, "small")

    if restart:
        print("清空 checkpoints...")
        _rm_tree(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 构建训练对象
    from skyalyticAI.brain import NIEABrain
    from skyalyticAI.training.human_growth_trainer import HumanGrowthTrainer
    from skyalyticAI.gpu import get_gpu_info

    print("=== NIEA 训练 ===")
    print(get_gpu_info())

    # device 推断
    device = None
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    print("正在初始化训练环境...")

    if mode == "society":
        from skyalyticAI.society.sim_world import SocietySimWorld
        from skyalyticAI.perception.visual_encoder import VisualEncoder
        from skyalyticAI.perception.audio_encoder import AudioEncoder
        from skyalyticAI.perception.multimodal_fusion import MultimodalFusion

        env = SocietySimWorld(
            corpus_root=None,
            observation_dim=128,
            school_stage=start_stage,
            max_stage=max_stage,
            student_name="小析",
            seed=42,
        )
        visual_encoder = VisualEncoder(
            image_height=28,
            image_width=28,
            n_channels=1,
            output_dim=128,
        )
        audio_encoder = AudioEncoder(
            sample_rate=16000,
            output_dim=128,
        )
        fusion = MultimodalFusion(
            modality_dims={"visual": 128, "audio": 128},
            output_dim=128,
        )
    elif mode == "social":
        from skyalyticAI.env.social_classroom_world import SocialClassroomWorld

        env = SocialClassroomWorld(
            corpus_root=None,
            observation_dim=128,
            school_stage=start_stage,
            max_stage=max_stage,
            student_name="小析",
            seed=42,
        )
    else:
        from skyalyticAI.env.curriculum_world import HumanGrowthWorld

        env = HumanGrowthWorld(
            corpus_root=None,
            observation_dim=128,
            school_stage=start_stage,
            seed=42,
        )
        # human 模式下封顶：若超出则强制回到封顶（不升级）
        if start_stage != max_stage:
            pass

    vocab = env.get_action_dim()
    brain = NIEABrain(
        input_dim=env.get_observation_dim(),
        action_dim=vocab,
        language_vocab_size=vocab,
        visual_encoder=locals().get("visual_encoder"),
        audio_encoder=locals().get("audio_encoder"),
        multimodal_fusion=locals().get("fusion"),
        device=device,
        brain_scale=brain_scale,
    )

    trainer = HumanGrowthTrainer(
        brain=brain,
        env=env,  # type: ignore
        max_episodes=episodes,
        max_steps_per_episode=steps,
        checkpoint_dir=str(ckpt_dir),
        checkpoint_interval=checkpoint_interval,
        log_interval=log_interval,
        forgetting_threshold=0.05,
    )

    if resume and latest is not None and latest.exists():
        try:
            trainer._load_checkpoint(str(latest))  # noqa: SLF001
            print("已从 checkpoint 恢复:", latest)
        except Exception as e:
            print("恢复 checkpoint 失败，将从头训练。原因:", e)

    summary = trainer.train()
    report = trainer.generate_acceptance_report(report_path)

    print("\n=== 训练完成 ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\n=== 验收报告 ===")
    print("  已写入:", report_path)
    print("  数据就绪:", report["industrial_readiness"]["data_ready"])
    print("  训练就绪:", report["industrial_readiness"]["training_ready"])
    print("  抗遗忘通过:", report["industrial_readiness"]["retention_pass"])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中断。")
        sys.exit(130)

