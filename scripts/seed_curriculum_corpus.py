#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成结构化课程语料（占位教学文本），用于大规模训练前的数据管线打底。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from skyalyticAI.data.education_config import (
    DEFAULT_MAJORS,
    HIGH_SUBJECTS,
    KINDERGARTEN_SUBJECTS,
    MIDDLE_SUBJECTS,
    PRIMARY_SUBJECTS,
)


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"


STAGE_SUBJECTS: Dict[str, List[str]] = {
    "01_kindergarten": KINDERGARTEN_SUBJECTS,
    "02_primary": PRIMARY_SUBJECTS,
    "03_middle": MIDDLE_SUBJECTS,
    "04_high": HIGH_SUBJECTS,
}

UNIVERSITY_STAGES = ["05_undergraduate", "06_master", "07_phd"]


def _write_file(path: Path, stage: str, subject: str, idx: int) -> None:
    text = (
        f"学段：{stage}\n"
        f"科目/专业：{subject}\n"
        f"课次：{idx}\n"
        "目标：理解概念、完成例题、复述核心知识。\n"
        "阅读材料：\n"
        f"1) {subject}基础定义与背景。\n"
        f"2) {subject}典型问题拆解与多步推理。\n"
        f"3) {subject}阅读理解训练与归纳总结。\n"
        "练习：请写出关键术语并解释其因果关系。\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    total = 0
    for stage, subjects in STAGE_SUBJECTS.items():
        for subject in subjects:
            for i in range(1, 121):
                fp = CORPUS / stage / subject / f"{subject}_lesson_{i:04d}.txt"
                _write_file(fp, stage, subject, i)
                total += 1

    for stage in UNIVERSITY_STAGES:
        for major in DEFAULT_MAJORS:
            for i in range(1, 121):
                fp = CORPUS / stage / major / f"{major}_module_{i:04d}.txt"
                _write_file(fp, stage, major, i)
                total += 1

    print(f"完成写入课程占位语料: {total} 文件")


if __name__ == "__main__":
    main()

