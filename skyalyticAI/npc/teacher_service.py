"""
外部教师服务（可插拔）：让 NPC 家长/老师真正“内置 AI”。

支持：
- OpenAI 兼容 Chat Completions 接口（可对接云 API / 本地 LM Studio / 其它兼容网关）

通过环境变量配置：
- NIEA_TEACHER_API_BASE: 例如 http://localhost:1234/v1 或 https://api.deepseek.com
- NIEA_TEACHER_API_KEY: 例如 sk-xxx（本地服务可留空）
- NIEA_TEACHER_MODEL: 例如 gpt-4o-mini / qwen2.5 / llama3.1 等
- NIEA_TEACHER_API_PATH: 默认 /chat/completions，一般无需修改
- NIEA_TEACHER_AUTO_VERSION: 默认 "1"，当 API_BASE 未含版本路径时自动补 /v1
  设为 "0" 可禁用自动补全（适用于自定义网关或非标准路径）

说明：
- 该模块只负责“生成老师话语/题干/答案”，不引入新依赖
- 若未配置或请求失败，调用方应回退到规则生成
- 自动补全规则：仅当 URL 路径部分为空或仅为 "/" 时，追加 "/v1"；
  若用户已填写 /v1 /v2 /v4 等版本路径，则保持原样不改动
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


def _normalize_api_base(api_base: str, auto_version: bool = True) -> str:
    """规范化 API base URL。

    通用规则（不硬编码任何服务商）：
    - 若 URL 路径为空或仅为 "/"，且 auto_version=True，则追加 "/v1"
    - 若 URL 已包含任何路径（如 /v1 /v2 /v4 /openai/v1），保持原样
    - 去除末尾多余的 "/"

    示例:
      https://api.deepseek.com        -> https://api.deepseek.com/v1
      https://api.deepseek.com/v4     -> https://api.deepseek.com/v4  (保持)
      https://api.openai.com          -> https://api.openai.com/v1
      https://api.openai.com/v1       -> https://api.openai.com/v1    (保持)
      http://localhost:1234           -> http://localhost:1234/v1
      http://localhost:1234/v1        -> http://localhost:1234/v1     (保持)
    """
    base = api_base.rstrip("/")
    if not auto_version:
        return base
    parsed = urllib.parse.urlparse(base)
    path = parsed.path.strip("/")
    if not path:
        # 路径为空，自动补 /v1
        return base + "/v1"
    return base


@dataclass
class TeacherServiceConfig:
    api_base: str
    api_key: str
    model: str
    timeout_s: int = 45
    min_interval_s: float = 0.2  # 简单限流，避免疯狂请求
    api_path: str = "/chat/completions"


class TeacherService:
    def __init__(self, cfg: TeacherServiceConfig) -> None:
        self.cfg = cfg
        self._last_call_t = 0.0

    @staticmethod
    def from_env() -> Optional["TeacherService"]:
        api_base = os.environ.get("NIEA_TEACHER_API_BASE", "").strip()
        model = os.environ.get("NIEA_TEACHER_MODEL", "").strip()
        if not api_base or not model:
            return None
        api_key = os.environ.get("NIEA_TEACHER_API_KEY", "").strip()
        api_path = os.environ.get("NIEA_TEACHER_API_PATH", "").strip() or "/chat/completions"
        # 通用自动补全：仅当 URL 无路径时补 /v1，已含 /v1 /v2 /v4 等则保持原样
        auto_version = os.environ.get("NIEA_TEACHER_AUTO_VERSION", "1").strip() not in ("0", "false", "False", "no", "No")
        api_base = _normalize_api_base(api_base, auto_version=auto_version)
        return TeacherService(
            TeacherServiceConfig(
                api_base=api_base,
                api_key=api_key,
                model=model,
                api_path=api_path,
            )
        )

    def _sleep_if_needed(self) -> None:
        now = time.time()
        dt = now - self._last_call_t
        if dt < self.cfg.min_interval_s:
            time.sleep(self.cfg.min_interval_s - dt)
        self._last_call_t = time.time()

    def chat(self, system: str, user: str, temperature: float = 0.6, max_tokens: int = 200) -> str:
        self._sleep_if_needed()
        url = self.cfg.api_base.rstrip("/") + self.cfg.api_path
        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = "Bearer " + self.cfg.api_key

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                body = ""
            raise RuntimeError(f"TeacherService HTTPError: {e.code} {e.reason} {body[:500]}") from e
        except Exception as e:
            raise RuntimeError(f"TeacherService request failed: {e}") from e

        obj = json.loads(raw)
        choices = obj.get("choices") or []
        if not choices:
            raise RuntimeError("TeacherService empty choices")
        msg = choices[0].get("message") or {}
        content = (msg.get("content") or "").strip()
        if not content:
            raise RuntimeError("TeacherService empty content")
        return content

