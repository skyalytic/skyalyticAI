#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
语料库综合修复脚本：
1. 繁简转换（opencc t2s）
2. 维基百科残留清理（章节/LaTeX/引用标记/模板）
3. 删除过短文件（<100字符）和消歧义页
4. 去重（保留每组中最长的一份）

跳过 01_kindergarten（已干净）。
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from opencc import OpenCC

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"
AUDIT = ROOT / "_corpus_audit_report.json"

# 初始化繁简转换器（通用繁→简）
cc = OpenCC("t2s")

# 维基百科残留章节标题（出现这些行时，删除该行及其后所有内容直到下一个 # 标题或文件末尾）
WIKI_SECTIONS = [
    "参见", "参考文献", "外部链接", "延伸阅读", "来源", "注释",
    "书目", "参考资料", "相关条目", "相关条目与外部链接",
    "参考资料与注释", "注脚", "文献", "进一步阅读",
    "外部连结", "外部参考", "外部资源", "延伸阅读与外部链接",
    "引用", "引文", "引用来源", "引用与注释",
]

# 维基模板文字（整行删除）
WIKI_TEMPLATE_PATTERNS = [
    r"^\s*此条目.*?$",
    r"^\s*本条目.*?$",
    r"^\s*维基百科.*?提醒.*?$",
    r"^\s*按.*?此处.*?$",
    r"^\s*请协助补充.*?$",
    r"^\s*请协助改善.*?$",
    r"^\s*需要扩充.*?$",
    r"^\s*需要专家关注.*?$",
    r"^\s*此页面.*?$",
    r"^\s*本页.*?$",
    r"^\s*这是一个.*?条目.*?$",
    r"^\s*关于.*?，详见.*?$",
    r"^\s*关于.*?，请见.*?$",
    r"^\s*同名的其他条目.*?$",
    r"^\s*消歧义.*?$",
    r"^\s*这不是.*?条目.*?$",
]

# LaTeX/公式残留
LATEX_PATTERNS = [
    r"\{\\displaystyle[^}]*\}",   # {\displaystyle ...}
    r"\\\[[^\]]*\\\]",            # \[ ... \]
    r"\\\([^\)]*\\\)",            # \( ... \)
    r"\\frac\{[^}]*\}\{[^}]*\}",  # \frac{a}{b}
    r"\\sqrt\{[^}]*\}",           # \sqrt{a}
    r"\\sum[_^]\{[^}]*\}",        # \sum_{...}
    r"\\int[_^]\{[^}]*\}",        # \int_{...}
    r"\\lim[_^]\{[^}]*\}",        # \lim_{...}
    r"\\[a-zA-Z]+\{[^}]*\}",      # 通用 \xxx{...}
]

# 引用标记：[1] [12] [citation needed] 等
CITATION_PATTERN = re.compile(r"\[(\d+|citation needed|来源请求|注 \d+|a-zA-Z]+)\]")

# 维基内部链接残留：[[xxx]] [[xxx|yyy]]
WIKI_LINK_PATTERN = re.compile(r"\[\[([^\]|]+)(\|([^\]]+))?\]\]")

# HTML 残留
HTML_PATTERN = re.compile(r"<[^>]+>")


def clean_wikipedia_residue(text: str) -> str:
    """清理维基百科残留：章节、LaTeX、引用、模板、链接、HTML。"""
    lines = text.split("\n")
    cleaned_lines = []
    skip_until_next_heading = False

    for line in lines:
        stripped = line.strip()

        # 检测章节标题（## xxx 或 ### xxx 或 无 # 但单独成行的中文标题）
        # 简单策略：以 # 开头视为标题
        is_heading = stripped.startswith("#")

        if is_heading:
            # 提取标题文本（去掉 # 前缀）
            heading_text = stripped.lstrip("#").strip()
            # 如果是维基残留章节，跳过该行及后续内容直到下一个标题
            if any(heading_text == s or heading_text.startswith(s) for s in WIKI_SECTIONS):
                skip_until_next_heading = True
                continue
            else:
                skip_until_next_heading = False
                cleaned_lines.append(line)
                continue

        if skip_until_next_heading:
            continue

        # 删除维基模板文字（整行匹配）
        if any(re.match(p, stripped, re.IGNORECASE) for p in WIKI_TEMPLATE_PATTERNS):
            continue

        # 删除"可以指"开头的消歧义行
        if re.match(r"^.{0,20}可以指[:：]", stripped):
            continue

        # 清理 LaTeX 公式（替换为占位符，避免完全丢失数学语义）
        for p in LATEX_PATTERNS:
            line = re.sub(p, "〔公式〕", line)

        # 清理引用标记 [1] [citation needed] 等
        line = CITATION_PATTERN.sub("", line)

        # 清理维基链接 [[xxx]] -> xxx, [[xxx|yyy]] -> yyy
        def _replace_wikilink(m: re.Match) -> str:
            return m.group(3) if m.group(3) else m.group(1)
        line = WIKI_LINK_PATTERN.sub(_replace_wikilink, line)

        # 清理 HTML 标签
        line = HTML_PATTERN.sub("", line)

        # 清理多余空行（连续空行压缩为单行）
        if stripped == "" and cleaned_lines and cleaned_lines[-1].strip() == "":
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # 清理末尾多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip() + "\n"

    return text


def convert_traditional(text: str) -> str:
    """繁体转简体。"""
    return cc.convert(text)


def is_disambiguation(content: str, length: int) -> bool:
    """检测是否为消歧义页。"""
    if length >= 200:
        return False
    # 长度短且包含"可以指"或"消歧义"或"可能指"
    if "可以指" in content or "消歧义" in content or "可能指" in content:
        return True
    # 长度极短且包含"可能是指"
    if length < 100 and "是指" in content:
        return True
    return False


def load_audit_report() -> dict:
    """加载审计报告。"""
    if not AUDIT.is_file():
        return {}
    try:
        return json.loads(AUDIT.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def main() -> None:
    print("=" * 70, flush=True)
    print("语料库综合修复脚本", flush=True)
    print("=" * 70, flush=True)

    if not CORPUS.is_dir():
        print(f"错误：语料目录不存在 {CORPUS}", flush=True)
        sys.exit(1)

    audit = load_audit_report()

    # 加载待删除文件清单（过短文件 + 消歧义页）
    too_short_paths = set()
    if audit:
        for item in audit.get("too_short_files", []):
            too_short_paths.add(Path(CORPUS) / item["relpath"])

    # 加载重复文件清单（保留每组中最长的一份）
    duplicate_to_delete: list[Path] = []
    if audit:
        for group in audit.get("duplicate_groups", []):
            paths = [Path(CORPUS) / p for p in group["paths"]]
            # 选出最长的一份保留，其他删除
            valid = [(p, p.stat().st_size if p.is_file() else 0) for p in paths]
            valid.sort(key=lambda x: x[1], reverse=True)
            for p, _ in valid[1:]:  # 保留最长的，删除其余
                duplicate_to_delete.append(p)

    print(f"\n[计划] 繁简转换 + 维基清理：全部非 kindergarten 文件", flush=True)
    print(f"[计划] 删除过短文件：{len(too_short_paths)} 个", flush=True)
    print(f"[计划] 删除重复文件：{len(duplicate_to_delete)} 个", flush=True)

    # 统计
    stats = {
        "total_processed": 0,
        "traditional_converted": 0,
        "wiki_cleaned": 0,
        "disambiguation_deleted": 0,
        "too_short_deleted": 0,
        "duplicate_deleted": 0,
        "errors": 0,
    }

    # 第1阶段：遍历所有 .txt 文件，做繁简转换 + 维基清理
    print("\n[阶段1] 繁简转换 + 维基残留清理...", flush=True)
    t0 = time.time()

    all_txt_files = sorted(CORPUS.rglob("*.txt"))
    print(f"  发现 {len(all_txt_files)} 个 .txt 文件", flush=True)

    for i, path in enumerate(all_txt_files, 1):
        try:
            # 跳过 01_kindergarten（已干净）
            rel = path.relative_to(CORPUS)
            if rel.parts and rel.parts[0] == "01_kindergarten":
                stats["total_processed"] += 1
                continue

            content = path.read_text(encoding="utf-8", errors="ignore")
            original_len = len(content)

            # 1. 维基残留清理
            cleaned = clean_wikipedia_residue(content)
            if len(cleaned) != original_len:
                stats["wiki_cleaned"] += 1

            # 2. 繁简转换
            converted = convert_traditional(cleaned)
            if converted != cleaned:
                stats["traditional_converted"] += 1

            # 写回
            if converted != content:
                path.write_text(converted, encoding="utf-8")

            stats["total_processed"] += 1

            if i % 100 == 0:
                print(f"  进度：{i}/{len(all_txt_files)} ({time.time()-t0:.1f}s)", flush=True)

        except Exception as e:
            stats["errors"] += 1
            print(f"  错误处理 {path}: {e}", flush=True)

    print(f"  阶段1完成：{time.time()-t0:.1f}s，处理 {stats['total_processed']} 文件", flush=True)
    print(f"    繁简转换：{stats['traditional_converted']} 文件", flush=True)
    print(f"    维基清理：{stats['wiki_cleaned']} 文件", flush=True)

    # 第2阶段：删除过短文件和消歧义页
    print("\n[阶段2] 删除过短文件和消歧义页...", flush=True)
    t0 = time.time()

    for path in too_short_paths:
        try:
            if not path.is_file():
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            length = len(content)
            if is_disambiguation(content, length) or length < 100:
                path.unlink()
                stats["too_short_deleted"] += 1
                if is_disambiguation(content, length):
                    stats["disambiguation_deleted"] += 1
                print(f"  删除：{path.relative_to(CORPUS)} (len={length})", flush=True)
        except Exception as e:
            stats["errors"] += 1
            print(f"  错误删除 {path}: {e}", flush=True)

    # 额外扫描：检测所有文件中的消歧义页（清理后可能仍存在）
    print("  额外扫描消歧义页...", flush=True)
    for path in sorted(CORPUS.rglob("*.txt")):
        try:
            rel = path.relative_to(CORPUS)
            if rel.parts and rel.parts[0] == "01_kindergarten":
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            length = len(content)
            if is_disambiguation(content, length):
                path.unlink()
                stats["disambiguation_deleted"] += 1
                print(f"  删除消歧义：{rel} (len={length})", flush=True)
        except Exception:
            pass

    print(f"  阶段2完成：{time.time()-t0:.1f}s", flush=True)
    print(f"    过短删除：{stats['too_short_deleted']}", flush=True)
    print(f"    消歧义删除：{stats['disambiguation_deleted']}", flush=True)

    # 第3阶段：去重
    print("\n[阶段3] 去重（保留每组中最长的一份）...", flush=True)
    t0 = time.time()

    for path in duplicate_to_delete:
        try:
            if path.is_file():
                rel = path.relative_to(CORPUS)
                size = path.stat().st_size
                path.unlink()
                stats["duplicate_deleted"] += 1
                print(f"  删除重复：{rel} (size={size})", flush=True)
        except Exception as e:
            stats["errors"] += 1
            print(f"  错误删除 {path}: {e}", flush=True)

    print(f"  阶段3完成：{time.time()-t0:.1f}s", flush=True)

    # 汇总
    print("\n" + "=" * 70, flush=True)
    print("修复完成汇总", flush=True)
    print("=" * 70, flush=True)
    print(f"总处理文件：{stats['total_processed']}", flush=True)
    print(f"繁简转换：{stats['traditional_converted']}", flush=True)
    print(f"维基清理：{stats['wiki_cleaned']}", flush=True)
    print(f"删除过短：{stats['too_short_deleted']}", flush=True)
    print(f"删除消歧义：{stats['disambiguation_deleted']}", flush=True)
    print(f"删除重复：{stats['duplicate_deleted']}", flush=True)
    print(f"错误数：{stats['errors']}", flush=True)

    # 统计剩余文件数
    remaining = sum(1 for _ in CORPUS.rglob("*.txt"))
    print(f"\n剩余文件数：{remaining}", flush=True)


if __name__ == "__main__":
    main()
