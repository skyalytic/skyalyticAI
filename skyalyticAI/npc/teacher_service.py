"""
外部教师服务（可插拔）：让 NPC 家长/老师真正“内置 AI”。

支持：
- OpenAI 兼容 Chat Completions 接口（可对接云 API / 本地 LM Studio / 其它兼容网关）

通过环境变量配置：
- NIEA_TEACHER_API_BASE: 例如 http://localhost:1234/v1
- NIEA_TEACHER_API_KEY: 例如 sk-xxx（本地服务可留空）
- NIEA_TEACHER_MODEL: 例如 gpt-4o-mini / qwen2.5 / llama3.1 等

说明：
- 该模块只负责“生成老师话语/题干/答案”，不引入新依赖
- 若未配置或请求失败，调用方应回退到规则生成
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


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
        # 兼容用户只填 https://api.deepseek.com （自动补 /v1）
        if api_base.rstrip("/").endswith("api.deepseek.com"):
            if not api_base.rstrip("/").endswith("/v1"):
                api_base = api_base.rstrip("/") + "/v1"
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

