"""
文本任务指标：CER / WER / 准确率。
"""

from __future__ import annotations

from typing import List


def edit_distance(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m]


def cer(pred: str, target: str) -> float:
    t = list(target)
    if not t:
        return 0.0
    return edit_distance(list(pred), t) / max(len(t), 1)


def wer(pred: str, target: str) -> float:
    p = pred.split()
    t = target.split()
    if not t:
        return 0.0
    return edit_distance(p, t) / max(len(t), 1)

