#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重新生成 learning_path.json：
- 扫描 data/corpus 下所有现有 .txt 文件
- 按学段基础难度 + 文件长度启发式分配 difficulty(1-5)
- 保留原有 learning_path.json 中的人工难度分配（若文件仍存在）
- 删除指向已不存在文件的条目
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"
LP_FILE = CORPUS / "learning_path.json"

# 学段基础难度偏移（学段越高级，整体难度越大）
STAGE_BASE_DIFFICULTY = {
    "00_sensorimotor": 1,
    "01_kindergarten": 1,
    "02_primary": 2,
    "03_middle": 3,
    "04_high": 4,
    "05_undergraduate": 5,
}

# 学段目录名 -> 学段key（与 education_config.STAGE_DIR_MAP 一致）
STAGE_DIRS = {
    "00_sensorimotor": "sensorimotor",
    "01_kindergarten": "kindergarten",
    "02_primary": "primary",
    "03_middle": "middle",
    "04_high": "high",
    "05_undergraduate": "undergraduate",
    "06_master": "master",
    "07_phd": "phd",
}


def load_existing_lp() -> dict:
    """加载现有的 learning_path.json。"""
    if not LP_FILE.is_file():
        return {}
    try:
        return json.loads(LP_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def compute_difficulty(stage_dir: str, length: int, existing_diff: int | None = None) -> int:
    """计算文件难度(1-5)。
    优先使用人工分配的难度；否则基于学段基础难度 + 文件长度启发式。
    """
    if existing_diff is not None and 1 <= existing_diff <= 5:
        return existing_diff
    base = STAGE_BASE_DIFFICULTY.get(stage_dir, 3)
    # 文件长度启发式：越长难度越高
    if length < 500:
        adj = 0
    elif length < 2000:
        adj = 0
    elif length < 10000:
        adj = 1
    elif length < 30000:
        adj = 1
    else:
        adj = 2
    return max(1, min(5, base + adj - 1))  # 限制在 1-5 范围


def main() -> None:
    print("=" * 70, flush=True)
    print("重新生成 learning_path.json", flush=True)
    print("=" * 70, flush=True)

    existing_lp = load_existing_lp()
    print(f"加载现有 learning_path.json：{len(existing_lp)} 个学段", flush=True)

    new_lp: dict = {}
    total_files = 0
    preserved = 0
    regenerated = 0

    for stage_dir in sorted(STAGE_DIRS.keys()):
        stage_path = CORPUS / stage_dir
        if not stage_path.is_dir():
            continue

        new_lp[stage_dir] = {}
        existing_stage = existing_lp.get(stage_dir, {})

        for subject_dir in sorted(stage_path.iterdir()):
            if not subject_dir.is_dir():
                continue
            subject = subject_dir.name
            files = sorted([f for f in subject_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt"])
            if not files:
                continue

            existing_subject: list[dict] = existing_stage.get(subject, [])
            existing_by_name = {item["file"]: item for item in existing_subject if "file" in item}

            entries = []
            for order, f in enumerate(files, 1):
                filename = f.name
                try:
                    length = len(f.read_text(encoding="utf-8", errors="ignore"))
                except OSError:
                    length = 0

                existing_entry = existing_by_name.get(filename)
                if existing_entry and "difficulty" in existing_entry:
                    diff = compute_difficulty(stage_dir, length, int(existing_entry["difficulty"]))
                    preserved += 1
                else:
                    diff = compute_difficulty(stage_dir, length, None)
                    regenerated += 1

                entries.append({"file": filename, "difficulty": diff, "order": order})
                total_files += 1

            new_lp[stage_dir][subject] = entries

    # 写入
    LP_FILE.write_text(
        json.dumps(new_lp, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n完成汇总：", flush=True)
    print(f"  总文件数：{total_files}", flush=True)
    print(f"  保留人工难度：{preserved}", flush=True)
    print(f"  重新分配难度：{regenerated}", flush=True)
    print(f"  输出：{LP_FILE}", flush=True)

    # 按学段统计
    print(f"\n按学段统计：", flush=True)
    for stage_dir, subjects in new_lp.items():
        file_count = sum(len(entries) for entries in subjects.values())
        subject_count = len(subjects)
        print(f"  {stage_dir}: {subject_count} 科目, {file_count} 文件", flush=True)


if __name__ == "__main__":
    main()
