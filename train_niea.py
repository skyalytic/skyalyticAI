#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NIEA 类人成长训练入口。

用法示例：
  python train_niea.py --mode human --episodes 200
  python train_niea.py --mode grid --episodes 50
  python scripts/fetch_public_corpus.py   # 抓取公版读物到 data/corpus/

正版教材：将 TXT/MD 放入 data/corpus/01_primary 等文件夹即可自动加载。
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
  # Windows 下 PyTorch 与 MKL 的 OpenMP 冲突规避
  os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

  parser = argparse.ArgumentParser(description="NIEA 训练")
  parser.add_argument(
    "--mode",
    choices=["human", "grid", "social", "society"],
    default="human",
    help="human=类人世界(迷宫+读书+考试); social=社会课堂(家长/老师陪伴); society=完整社会模拟器; grid=仅迷宫",
  )
  parser.add_argument("--episodes", type=int, default=100, help="训练回合数")
  parser.add_argument("--steps", type=int, default=80, help="每回合最大步数")
  parser.add_argument("--obs-dim", type=int, default=128, help="观测维度")
  parser.add_argument("--hidden-dim", type=int, default=128, help="隐藏层维度")
  parser.add_argument("--corpus", type=str, default=None, help="语料根目录")
  parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="检查点目录")
  parser.add_argument("--checkpoint-interval", type=int, default=20, help="每 N 回合存盘")
  parser.add_argument("--log-interval", type=int, default=5, help="日志间隔")
  parser.add_argument("--seed", type=int, default=42, help="随机种子")
  parser.add_argument("--exam-every", type=int, default=5, help="每 N 回合一次升学考试")
  parser.add_argument("--pass-accuracy", type=float, default=0.55, help="考试及格准确率")
  parser.add_argument("--device", type=str, default=None, help="cuda 或 cpu")
  parser.add_argument("--forgetting-threshold", type=float, default=0.05, help="遗忘回滚阈值")
  parser.add_argument("--report-path", type=str, default="reports/acceptance_report.json", help="验收报告输出路径")
  parser.add_argument("--industrial", action="store_true", help="工业长跑预设（高步数与更严门槛）")
  args = parser.parse_args()

  import numpy as np
  from skyalyticAI.brain import NIEABrain
  from skyalyticAI.env.grid_world import GridWorldEnv
  from skyalyticAI.env.curriculum_world import HumanGrowthWorld
  from skyalyticAI.training.trainer import NIEATrainer
  from skyalyticAI.training.human_growth_trainer import HumanGrowthTrainer
  from skyalyticAI.gpu import get_gpu_info

  print("=== NIEA 训练 ===")
  print(get_gpu_info())

  device = args.device
  if device is None:
    try:
      import torch
      device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
      device = "cpu"

  if args.mode == "human":
    if args.industrial:
      # 工业长跑默认值（若用户显式传参，参数优先）
      args.episodes = max(args.episodes, 5000)
      args.steps = max(args.steps, 220)
      args.checkpoint_interval = max(10, min(args.checkpoint_interval, 100))

    env = HumanGrowthWorld(
      corpus_root=args.corpus,
      observation_dim=args.obs_dim,
      exam_every_n_episodes=args.exam_every,
      exam_pass_accuracy=args.pass_accuracy,
      seed=args.seed,
    )
    vocab = env.get_action_dim()
    brain = NIEABrain(
      input_dim=args.obs_dim,
      hidden_dim=args.hidden_dim,
      action_dim=vocab,
      n_observations=min(args.obs_dim, 64),
      hd_dim=4000,
      pcn_hidden_dim=64,
      world_model_hidden_dim=64,
      language_vocab_size=vocab,
      device=device,
    )
    trainer = HumanGrowthTrainer(
      brain=brain,
      env=env,
      max_episodes=args.episodes,
      max_steps_per_episode=args.steps,
      checkpoint_dir=args.checkpoint_dir,
      checkpoint_interval=args.checkpoint_interval,
      log_interval=args.log_interval,
      forgetting_threshold=args.forgetting_threshold,
    )
    print("环境: 类人成长世界 | 词表: {} | 语料: {}".format(
      vocab, env.corpus.corpus_root
    ))
  elif args.mode == "social":
    if args.industrial:
      args.episodes = max(args.episodes, 5000)
      args.steps = max(args.steps, 220)
      args.checkpoint_interval = max(10, min(args.checkpoint_interval, 100))

    from skyalyticAI.env.social_classroom_world import SocialClassroomWorld

    env = SocialClassroomWorld(
      corpus_root=args.corpus,
      observation_dim=args.obs_dim,
      school_stage="undergraduate",
      max_stage="undergraduate",
      student_name="小析",
      seed=args.seed,
    )
    vocab = env.get_action_dim()
    brain = NIEABrain(
      input_dim=args.obs_dim,
      hidden_dim=args.hidden_dim,
      action_dim=vocab,
      n_observations=min(args.obs_dim, 64),
      hd_dim=4000,
      pcn_hidden_dim=64,
      world_model_hidden_dim=64,
      language_vocab_size=vocab,
      device=device,
    )
    # 复用 HumanGrowthTrainer 的“说话学习 + 抗遗忘报告”
    trainer = HumanGrowthTrainer(
      brain=brain,
      env=env,  # type: ignore
      max_episodes=args.episodes,
      max_steps_per_episode=args.steps,
      checkpoint_dir=args.checkpoint_dir,
      checkpoint_interval=args.checkpoint_interval,
      log_interval=args.log_interval,
      forgetting_threshold=args.forgetting_threshold,
    )
    print("环境: 社会课堂(家长/老师陪伴) | 词表: {} | 语料(root可为空): {}".format(
      vocab, getattr(getattr(env, "corpus", None), "corpus_root", None)
    ))
  elif args.mode == "society":
    if args.industrial:
      args.episodes = max(args.episodes, 5000)
      args.steps = max(args.steps, 220)
      args.checkpoint_interval = max(10, min(args.checkpoint_interval, 100))

    from skyalyticAI.society.sim_world import SocietySimWorld
    from skyalyticAI.perception.visual_encoder import VisualEncoder
    from skyalyticAI.perception.audio_encoder import AudioEncoder
    from skyalyticAI.perception.multimodal_fusion import MultimodalFusion

    env = SocietySimWorld(
      corpus_root=args.corpus,
      observation_dim=args.obs_dim,
      school_stage="sensorimotor",
      max_stage="undergraduate",
      student_name="小析",
      seed=args.seed,
    )
    vocab = env.get_action_dim()
    visual_encoder = VisualEncoder(
      image_height=28,
      image_width=28,
      n_channels=1,
      output_dim=args.obs_dim,
    )
    audio_encoder = AudioEncoder(
      sample_rate=16000,
      output_dim=args.obs_dim,
    )
    fusion = MultimodalFusion(
      modality_dims={"visual": args.obs_dim, "audio": args.obs_dim},
      output_dim=args.obs_dim,
    )
    brain = NIEABrain(
      input_dim=args.obs_dim,
      hidden_dim=args.hidden_dim,
      action_dim=vocab,
      n_observations=min(args.obs_dim, 64),
      hd_dim=4000,
      pcn_hidden_dim=64,
      world_model_hidden_dim=64,
      language_vocab_size=vocab,
      visual_encoder=visual_encoder,
      audio_encoder=audio_encoder,
      multimodal_fusion=fusion,
      device=device,
    )
    trainer = HumanGrowthTrainer(
      brain=brain,
      env=env,  # type: ignore
      max_episodes=args.episodes,
      max_steps_per_episode=args.steps,
      checkpoint_dir=args.checkpoint_dir,
      checkpoint_interval=args.checkpoint_interval,
      log_interval=args.log_interval,
      forgetting_threshold=args.forgetting_threshold,
    )
    print("环境: 完整社会模拟器(多模态+多智能体+长期关系) | 词表: {}".format(vocab))
  else:
    env = GridWorldEnv(width=8, height=8, seed=args.seed)
    brain = NIEABrain(
      input_dim=env.get_observation_dim(),
      hidden_dim=args.hidden_dim,
      action_dim=4,
      device=device,
    )
    trainer = NIEATrainer(
      brain=brain,
      env=env,
      max_episodes=args.episodes,
      max_steps_per_episode=args.steps,
      checkpoint_dir=args.checkpoint_dir,
      checkpoint_interval=args.checkpoint_interval,
      log_interval=args.log_interval,
    )
    print("环境: 迷宫 | 观测维度:", env.get_observation_dim())

  summary = trainer.train()
  print("\n=== 训练完成 ===")
  for k, v in summary.items():
    print("  {}: {}".format(k, v))

  if args.mode in ("human", "social", "society"):
    report = trainer.generate_acceptance_report(args.report_path)
    print("\n=== 验收报告 ===")
    print("  已写入:", args.report_path)
    print("  数据就绪:", report["industrial_readiness"]["data_ready"])
    print("  训练就绪:", report["industrial_readiness"]["training_ready"])
    print("  抗遗忘通过:", report["industrial_readiness"]["retention_pass"])


if __name__ == "__main__":
  main()
