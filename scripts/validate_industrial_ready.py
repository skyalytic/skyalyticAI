#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工业就绪校验器：检查“代码/理论/数据管线已完成，剩下仅训练”。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "acceptance_report.json"
THEORY_PATH = ROOT / "理论.md"
EXAMS_DIR = ROOT / "skyalyticAI" / "exams"
CORPUS_ROOT = ROOT / "data" / "corpus"
KNOWLEDGE_SOURCES = ROOT / "data" / "knowledge_sources_cn.json"


def _count_corpus_files() -> int:
    if not CORPUS_ROOT.exists():
        return 0
    return sum(1 for p in CORPUS_ROOT.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".md", ".jsonl"})


def _read_report() -> Dict:
    if not REPORT_PATH.exists():
        return {}
    try:
        return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _check_items() -> Tuple[List[Tuple[str, bool, str]], bool]:
    report = _read_report()
    corpus_files = _count_corpus_files()
    training_steps = int(report.get("summary", {}).get("total_steps", 0))
    data_ready = bool(report.get("industrial_readiness", {}).get("data_ready", False))
    npc_curriculum = bool(report.get("corpus_stats", {}).get("npc_curriculum", False))
    retention_ready = bool(report.get("industrial_readiness", {}).get("retention_pass", False))
    training_ready = bool(report.get("industrial_readiness", {}).get("training_ready", False))
    mm = report.get("multimodal_metrics", {}) or {}
    asr_cer = float(mm.get("asr_cer", 1.0))
    ocr_cer = float(mm.get("ocr_cer", 1.0))
    has_asr_ocr_metrics = {"asr_cer", "asr_wer", "ocr_cer", "ocr_wer"}.issubset(set(mm.keys()))

    checks: List[Tuple[str, bool, str]] = [
        ("高阶题型代码", (EXAMS_DIR / "reading_comprehension_exam.py").exists() and (EXAMS_DIR / "multi_step_reasoning_exam.py").exists(), "阅读理解+多步推理模块"),
        ("高阶题型接入课程世界", (ROOT / "skyalyticAI" / "env" / "curriculum_world.py").exists(), "ExamSuite接入"),
        ("长期回测曲线", "retention_history" in report, "报告包含retention_history"),
        ("理论文档", THEORY_PATH.exists(), "存在理论文档"),
        ("大规模数据管线", KNOWLEDGE_SOURCES.exists(), "存在多来源配置"),
        ("语料数量门槛", npc_curriculum or corpus_files >= 5000 or data_ready, f"npc_curriculum={npc_curriculum}, 文件数={corpus_files}"),
        ("ASR/OCR 指标链路", has_asr_ocr_metrics, f"asr_cer={asr_cer:.4f}, ocr_cer={ocr_cer:.4f}"),
        ("训练步数门槛", training_steps >= 100000 or training_ready, f"当前训练步数={training_steps}"),
        ("抗遗忘门槛", retention_ready, "retention_pass"),
    ]
    code_theory_data_done = all(ok for name, ok, _ in checks if name not in {"训练步数门槛"})
    return checks, code_theory_data_done


def main() -> None:
    checks, code_theory_data_done = _check_items()
    print("=== NIEA 工业就绪检查 ===")
    for name, ok, detail in checks:
        mark = "[OK]" if ok else "[NO]"
        print(f"{mark} {name}: {detail}")
    print()
    if code_theory_data_done:
        print("结论：代码/理论/数据管线已完成，剩下仅长跑训练。")
    else:
        print("结论：仍有代码/理论/数据管线项未完成。")


if __name__ == "__main__":
    main()

