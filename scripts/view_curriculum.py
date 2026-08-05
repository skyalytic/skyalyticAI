"""查看 NPC 老师教了什么（API 增强语料）。"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skyalyticAI.data.corpus_manager import CorpusManager

# 初始化（模板语料，秒级完成）
cm = CorpusManager(vocab_size=512, seed=42)

print("=== 模板语料（初始化生成）===")
for stage in ["sensorimotor", "kindergarten", "primary"]:
    lines = cm._train_by_stage.get(stage, [])
    print(f"\n[{stage}] 模板语料 {len(lines)} 句:")
    for i, line in enumerate(lines[:5]):
        print(f"  {i+1}. {line}")
    if len(lines) > 5:
        print(f"  ... 共 {len(lines)} 句")

# 触发 API 增强（如果配置了 API key）
print("\n=== API 增强语料 ===")
for stage in ["kindergarten", "primary"]:
    print(f"\n[{stage}] 调用 API 增强...")
    cm._load_stage_curriculum(stage)
    for subj in cm.list_subjects(stage):
        lines = cm._train_by_key.get((stage, subj), [])
        api_lines = lines[120:] if len(lines) > 120 else lines  # 模板 120 句之后是 API 生成的
        if api_lines:
            print(f"\n  [{subj}] API 生成 {len(api_lines)} 句:")
            for i, line in enumerate(api_lines[:10]):
                print(f"    {i+1}. {line}")
            if len(api_lines) > 10:
                print(f"    ... 共 {len(api_lines)} 句")
