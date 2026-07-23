"""
Speech Synthesizer - 真实语音合成（老师的"声带"）。

理论定位（fix 119）：
- 语音合成属于**输出侧**人工装置（老师的发声器官），老师可以是机器——
  婴儿的母亲也不必是生物神经元构成的。合法非生物。
- 学习者的感知侧（耳蜗编码器）保持严格生物可解释，不接触任何文本，
  只能从真实声波中自底向上学习。

实现：pyttsx3（Windows SAPI5，离线，无需 API），wav 文件按文本哈希缓存。

注意：pyttsx3 在同一进程内第二次调用 runAndWait() 存在死锁缺陷
（COM/SAPI 事件循环状态不可重入）。因此每次合成都派生独立子进程执行，
配合超时保护——进程级隔离彻底规避该问题。缓存命中时零开销。
任何失败（无 pyttsx3、无中文语音、超时、COM 错误）都返回 None，由调用方回退。
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import wave
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# 子进程内执行的合成脚本：argv[1]=文本, argv[2]=输出wav路径, argv[3]=语速
_SYNTH_SCRIPT = r"""
import sys
import pyttsx3

text, out_path, rate = sys.argv[1], sys.argv[2], int(sys.argv[3])
engine = pyttsx3.init()
engine.setProperty("rate", rate)
try:
    for v in engine.getProperty("voices") or []:
        name = (getattr(v, "name", "") or "").lower()
        langs = str(getattr(v, "languages", "") or "").lower()
        vid = (getattr(v, "id", "") or "").lower()
        if "zh" in langs or "chinese" in name or "huihui" in name or "kangkang" in name or "yaoyao" in name or "zh" in vid:
            engine.setProperty("voice", v.id)
            break
except Exception:
    pass
engine.save_to_file(text, out_path)
engine.runAndWait()
"""


class SpeechSynthesizer:
    """
    文本 -> 真实语音波形。

    Parameters
    ----------
    cache_dir : str
        wav 缓存目录（按文本 MD5 命名，重复文本零开销）。
    rate : int
        语速（词/分钟），儿童教学场景宜慢。
    timeout : int
        单次合成的子进程超时（秒）。
    """

    def __init__(self, cache_dir: str = "assets/audio/tts_cache", rate: int = 150, timeout: int = 60) -> None:
        self.cache_dir = Path(cache_dir)
        self.rate = rate
        self.timeout = timeout
        self._available: Optional[bool] = None

    def _check_available(self) -> bool:
        if self._available is None:
            try:
                import pyttsx3  # noqa: F401

                self._available = True
            except Exception:
                self._available = False
        return self._available

    @staticmethod
    def _load_wav(path: Path) -> Optional[Tuple[np.ndarray, int]]:
        """读取 wav -> (float波形[-1,1], 采样率)，多声道取均值。"""
        try:
            with wave.open(str(path), "rb") as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
            if sampwidth == 2:
                data = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0
            elif sampwidth == 4:
                data = np.frombuffer(frames, dtype=np.int32).astype(np.float64) / 2147483648.0
            elif sampwidth == 1:
                data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
            else:
                return None
            if n_channels > 1:
                data = data.reshape(-1, n_channels).mean(axis=1)
            return np.clip(data, -1.0, 1.0), framerate
        except Exception:
            return None

    def synthesize(self, text: str) -> Optional[Tuple[np.ndarray, int]]:
        """
        文本 -> (波形, 采样率)。命中缓存直接读 wav；失败返回 None。
        """
        if not text or not self._check_available():
            return None
        key = hashlib.md5(f"{self.rate}|{text}".encode("utf-8")).hexdigest()[:16]
        wav_path = self.cache_dir / f"{key}.wav"

        if wav_path.exists():
            return self._load_wav(wav_path)

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [sys.executable, "-c", _SYNTH_SCRIPT, text, str(wav_path), str(self.rate)],
                timeout=self.timeout,
                capture_output=True,
                check=False,
            )
        except Exception:
            return None

        if not wav_path.exists():
            return None
        return self._load_wav(wav_path)
