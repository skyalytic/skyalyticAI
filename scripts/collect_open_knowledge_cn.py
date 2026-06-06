#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
开放知识采集器（中国路径语料）。

说明：
1) 仅建议抓取公版/开放授权/公开课程文本
2) 不自动抓取受版权保护教材全文
3) 用户可在 data/knowledge_sources_cn.json 中继续扩展来源
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
SOURCES_PATH = ROOT / "data" / "knowledge_sources_cn.json"
CORPUS_ROOT = ROOT / "data" / "corpus"


def fetch_text(url: str, timeout: int = 45) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "NIEA-collector/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    for enc in ("utf-8", "gb18030", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def strip_html(text: str) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def save_text(stage_dir: str, subject: str, filename: str, content: str) -> Path:
    folder = CORPUS_ROOT / stage_dir / subject
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / filename
    out.write_text(content, encoding="utf-8")
    return out


def main() -> None:
    if not SOURCES_PATH.exists():
        print("未找到来源清单:", SOURCES_PATH, file=sys.stderr)
        sys.exit(1)
    sources = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    ok = 0
    failed = 0
    for item in sources.get("sources", []):
        stage_dir = item.get("stage_dir")
        subject = item.get("subject", "通识")
        url = item.get("url")
        filename = item.get("filename", "source.txt")
        html = bool(item.get("is_html", True))
        if not stage_dir or not url:
            continue
        try:
            body = fetch_text(url)
            text = strip_html(body) if html else body
            # 控制单源大小，避免噪声过大
            text = text[: item.get("max_chars", 120000)]
            path = save_text(stage_dir, subject, filename, text)
            ok += 1
            print("OK ", path)
        except Exception as e:
            failed += 1
            print("FAIL", url, e, file=sys.stderr)
    print(f"完成: success={ok}, failed={failed}")


if __name__ == "__main__":
    main()

