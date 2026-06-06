#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
抓取公版读物到 data/corpus/（合法免费来源，非盗版教材）。

来源：Project Gutenberg 等公版书。中小学正版课本请自行购买/放入对应文件夹。
"""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

# 公版书 URL（英文经典，作小学英语阅读补充；中文课本请本地放入）
GUTENBERG_SAMPLES = [
    (
        "01_primary",
        "alice.txt",
        "https://www.gutenberg.org/cache/epub/11/pg11.txt",
    ),
    (
        "02_middle",
        "sherlock_excerpt.txt",
        "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
    ),
]

ROOT = Path(__file__).resolve().parents[1] / "data" / "corpus"


def fetch_url(url: str, timeout: int = 60) -> str:
  req = urllib.request.Request(url, headers={"User-Agent": "NIEA-corpus-fetch/1.0"})
  with urllib.request.urlopen(req, timeout=timeout) as resp:
    raw = resp.read()
  for enc in ("utf-8", "latin-1"):
    try:
      return raw.decode(enc)
    except UnicodeDecodeError:
      continue
  return raw.decode("utf-8", errors="ignore")


def trim_gutenberg(text: str, max_chars: int = 80000) -> str:
  text = re.sub(r"\r\n?", "\n", text)
  start = re.search(r"\*\*\* START OF.*?\*\*\*", text)
  end = re.search(r"\*\*\* END OF.*?\*\*\*", text)
  if start and end:
    text = text[start.end() : end.start()]
  return text[:max_chars]


def main() -> None:
  for subdir, filename, url in GUTENBERG_SAMPLES:
    folder = ROOT / subdir
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / filename
    print("下载 {} -> {}".format(url, out))
    try:
      body = trim_gutenberg(fetch_url(url))
      out.write_text(body, encoding="utf-8")
      print("  完成, {} 字符".format(len(body)))
    except Exception as e:
      print("  失败:", e, file=sys.stderr)

  print("\n中文启蒙/课本：请将 TXT 放入")
  for name in ("00_infant", "01_primary", "02_middle", "03_high", "04_university"):
    print("  ", ROOT / name)


if __name__ == "__main__":
  main()
