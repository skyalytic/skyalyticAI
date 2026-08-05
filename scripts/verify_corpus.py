#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速验证修复效果：检查繁体字残留、维基残留、长度分布。"""
from __future__ import annotations

import re
from pathlib import Path

from opencc import OpenCC

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"

cc = OpenCC("t2s")

# 繁体特征字
TRAD_CHARS = set("國學會說來過開關時這個對於後從為經動點業產電裡內業務兩個們麼們來說過發展進這樣現實際環境問題還係業發")

WIKI_SECTIONS = ["参见", "参考文献", "外部链接", "延伸阅读", "来源", "注释", "书目", "参考资料"]
LATEX_PATTERN = re.compile(r"\{\\displaystyle|\\frac\{|\\sqrt\{|\\sum_|\\int_|\\lim_")
CITATION_PATTERN = re.compile(r"\[\d+\]")
DISAMBIG_PATTERN = re.compile(r"可以指[:：]|消歧义")


def main() -> None:
    print("=" * 70, flush=True)
    print("语料库修复效果验证", flush=True)
    print("=" * 70, flush=True)

    stats = {
        "total": 0,
        "trad_heavy": 0,  # 繁体密度 >= 5%
        "trad_medium": 0,  # 1-5%
        "trad_light": 0,  # < 1%
        "trad_clean": 0,  # 0
        "wiki_section": 0,
        "latex": 0,
        "citation": 0,
        "disambig": 0,
        "too_short": 0,  # < 100
        "too_long": 0,  # > 80000
    }

    length_dist = {"<100": 0, "100-1000": 0, "1000-10000": 0, "10000-50000": 0, ">50000": 0}

    for path in sorted(CORPUS.rglob("*.txt")):
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        stats["total"] += 1
        length = len(content)

        # 长度分布
        if length < 100:
            length_dist["<100"] += 1
            stats["too_short"] += 1
        elif length < 1000:
            length_dist["100-1000"] += 1
        elif length < 10000:
            length_dist["1000-10000"] += 1
        elif length < 50000:
            length_dist["10000-50000"] += 1
        else:
            length_dist[">50000"] += 1
            stats["too_long"] += 1

        # 繁体检测（opencc 差异）
        if length > 0:
            converted = cc.convert(content)
            diff_count = sum(1 for a, b in zip(content, converted) if a != b)
            density = diff_count / length * 100
            if density == 0:
                stats["trad_clean"] += 1
            elif density < 1:
                stats["trad_light"] += 1
            elif density < 5:
                stats["trad_medium"] += 1
            else:
                stats["trad_heavy"] += 1

        # 维基残留检测
        for section in WIKI_SECTIONS:
            if re.search(rf"^#+\s*{re.escape(section)}", content, re.MULTILINE):
                stats["wiki_section"] += 1
                break

        if LATEX_PATTERN.search(content):
            stats["latex"] += 1

        if CITATION_PATTERN.search(content):
            stats["citation"] += 1

        if length < 200 and DISAMBIG_PATTERN.search(content):
            stats["disambig"] += 1

    print(f"\n总文件数：{stats['total']}", flush=True)

    print(f"\n--- 繁体字残留 ---", flush=True)
    print(f"  干净（0%）：{stats['trad_clean']} ({stats['trad_clean']*100/stats['total']:.1f}%)", flush=True)
    print(f"  轻度（<1%）：{stats['trad_light']} ({stats['trad_light']*100/stats['total']:.1f}%)", flush=True)
    print(f"  中度（1-5%）：{stats['trad_medium']} ({stats['trad_medium']*100/stats['total']:.1f}%)", flush=True)
    print(f"  重度（≥5%）：{stats['trad_heavy']} ({stats['trad_heavy']*100/stats['total']:.1f}%)", flush=True)

    print(f"\n--- 维基残留 ---", flush=True)
    print(f"  章节残留：{stats['wiki_section']}", flush=True)
    print(f"  LaTeX残留：{stats['latex']}", flush=True)
    print(f"  引用标记：{stats['citation']}", flush=True)
    print(f"  消歧义页：{stats['disambig']}", flush=True)

    print(f"\n--- 长度分布 ---", flush=True)
    for k, v in length_dist.items():
        print(f"  {k}: {v}", flush=True)

    print(f"\n--- 异常文件 ---", flush=True)
    print(f"  过短（<100）：{stats['too_short']}", flush=True)
    print(f"  过长（>50000）：{stats['too_long']}", flush=True)

    # 评估
    print(f"\n--- 总体评估 ---", flush=True)
    issues = []
    if stats["trad_heavy"] > 0:
        issues.append(f"仍有 {stats['trad_heavy']} 个文件繁体重度污染")
    if stats["wiki_section"] > 0:
        issues.append(f"仍有 {stats['wiki_section']} 个文件有维基章节残留")
    if stats["latex"] > 0:
        issues.append(f"仍有 {stats['latex']} 个文件有 LaTeX 残留")
    if stats["disambig"] > 0:
        issues.append(f"仍有 {stats['disambig']} 个消歧义页")
    if stats["too_short"] > 0:
        issues.append(f"仍有 {stats['too_short']} 个过短文件")

    if not issues:
        print("✅ 所有主要问题已修复，可以开始训练", flush=True)
    else:
        print("⚠️ 仍有问题：", flush=True)
        for issue in issues:
            print(f"  - {issue}", flush=True)


if __name__ == "__main__":
    main()
