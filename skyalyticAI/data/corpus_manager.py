"""
语料管理器 — 人生阶段 ×（科目 | 专业）双维度加载。

目录示例::
    00_sensorimotor/          # 0~3 岁：儿歌、父母话（学说话），无课本
    01_kindergarten/语言/...
    02_primary/语文/  02_primary/数学/  02_primary/英语/
    05_undergraduate/计算机/...
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from skyalyticAI.data.education_config import (
    DEFAULT_MAJORS,
    STAGE_DIR_MAP,
    STAGE_DISPLAY,
    STAGE_ORDER,
    UNIVERSITY_STAGES,
    subjects_for_stage,
)
from skyalyticAI.npc.teacher_npc import TeacherNPC

# (阶段, 科目或专业) -> 句子
TrainKey = Tuple[str, str]


class CorpusManager:
    def __init__(
        self,
        corpus_root: Optional[Union[str, Path]] = None,
        vocab_size: int = 512,
        exam_holdout_ratio: float = 0.12,
        seed: int = 42,
    ) -> None:
        if vocab_size < 64:
            raise ValueError("工业级训练建议 vocab_size >= 64")
        if not 0.0 < exam_holdout_ratio < 1.0:
            raise ValueError("exam_holdout_ratio 须在 (0, 1) 之间")

        self.corpus_root = Path(corpus_root or _default_corpus_root())
        self.vocab_size = vocab_size
        self.exam_holdout_ratio = exam_holdout_ratio
        self.rng = np.random.default_rng(seed)

        self.char2idx: Dict[str, int] = {}
        self.idx2char: Dict[int, str] = {}
        self._train_by_stage: Dict[str, List[str]] = {}
        self._exam_by_stage: Dict[str, List[str]] = {}
        self._train_by_key: Dict[TrainKey, List[str]] = {}
        self._keys_by_stage: Dict[str, List[str]] = defaultdict(list)
        self._file_count: int = 0
        self._npc_curriculum: bool = False
        self._teacher: Optional[TeacherNPC] = None
        # 已用 API 增强过的学段集合（避免重复调用 API 限流）
        self._loaded_stages: set = set()

        self._load_all()

    def _load_all(self) -> None:
        all_chars: List[str] = []
        raw_by_stage: Dict[str, List[str]] = {s: [] for s in STAGE_ORDER}
        key_buckets: Dict[TrainKey, List[str]] = defaultdict(list)

        if not self.corpus_root.is_dir():
            self.corpus_root.mkdir(parents=True, exist_ok=True)

        for entry in sorted(self.corpus_root.iterdir()):
            if not entry.is_dir():
                continue
            stage_key = STAGE_DIR_MAP.get(entry.name)
            if stage_key is None:
                continue

            for path in sorted(entry.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in (".txt", ".md"):
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                self._file_count += 1
                lines = _split_into_lines(text)
                raw_by_stage[stage_key].extend(lines)
                all_chars.extend(list(text))

                rel = path.relative_to(entry)
                if len(rel.parts) >= 2:
                    label = rel.parts[0]
                    key_buckets[(stage_key, label)].extend(lines)

        # 无静态语料：启用 NPC 家长/老师动态课程（完全不依赖 data/corpus 文件）
        if not all_chars:
            self._npc_curriculum = True
            self._teacher = TeacherNPC(seed=int(self.rng.integers(0, 10_000)))
            # fix 122 懒加载：初始化时临时禁用 API，用模板快速生成全部学段语料。
            # 原设计在初始化时调 API 生成 10320 句，限流 10RPM 需 17 小时。
            # 改为：初始化用模板秒级生成（构建完整词表），教学时 API 按需增强。
            _saved_service = self._teacher.service
            self._teacher.service = None
            for stage in STAGE_ORDER:
                subjects = subjects_for_stage(stage)
                if stage in UNIVERSITY_STAGES and not subjects:
                    subjects = list(DEFAULT_MAJORS)
                if not subjects:
                    subjects = ["通识"]
                # 每科生成少量课堂互动句子，作为训练与考试切分的母集合
                for subj in subjects:
                    for _ in range(120):
                        line = self._teacher.sample_teaching_line(stage, subj)
                        raw_by_stage[stage].append(line)
                        key_buckets[(stage, subj)].append(line)
                        all_chars.extend(list(line))
            self._teacher.service = _saved_service  # 恢复，教学时可用 API
            all_chars.extend(list(self._teacher.bootstrap_vocab_text()))

        self._build_vocab(all_chars)

        for stage in STAGE_ORDER:
            lines = raw_by_stage.get(stage, [])
            if not lines and stage != "sensorimotor":
                prev_i = STAGE_ORDER.index(stage) - 1
                if prev_i >= 0:
                    prev = STAGE_ORDER[prev_i]
                    lines = list(self._train_by_stage.get(prev, []))[:200]
            train, exam = _split_train_exam(lines, self.exam_holdout_ratio, self.rng)
            self._train_by_stage[stage] = train
            self._exam_by_stage[stage] = exam

        for (stage, label), lines in key_buckets.items():
            if not lines:
                continue
            train, _ = _split_train_exam(lines, self.exam_holdout_ratio, self.rng)
            self._train_by_key[(stage, label)] = train
            if label not in self._keys_by_stage[stage]:
                self._keys_by_stage[stage].append(label)

        for stage in STAGE_ORDER:
            if stage in UNIVERSITY_STAGES and not self._keys_by_stage[stage]:
                for m in DEFAULT_MAJORS[:8]:
                    self._keys_by_stage[stage].append(m)
            subs = subjects_for_stage(stage)
            for s in subs:
                if s not in self._keys_by_stage[stage]:
                    self._keys_by_stage[stage].append(s)
            if self._npc_curriculum and not self._keys_by_stage[stage]:
                self._keys_by_stage[stage].append("通识")

    def _load_stage_curriculum(self, stage: str) -> None:
        """升学时按需用 API 增强当前学段语料（懒加载）。

        初始化时已用模板生成全部学段语料（构建词表，秒级完成），
        此处用 API 生成少量更丰富的教学句子补充到训练/考试池。
        若 API 不可用则静默跳过（模板语料已足够训练）。
        每个学段只增强一次，避免重复调用 API 限流。
        """
        if stage not in STAGE_ORDER:
            return
        if stage in self._loaded_stages:
            return
        # 标记已加载，无论 API 是否可用都避免重复尝试
        self._loaded_stages.add(stage)
        if not self._npc_curriculum or self._teacher is None:
            return
        if self._teacher.service is None:
            return

        subjects = subjects_for_stage(stage)
        if stage in UNIVERSITY_STAGES and not subjects:
            subjects = list(DEFAULT_MAJORS)
        if not subjects:
            subjects = ["通识"]

        import time as _time
        _t0 = _time.time()
        print(
            f"[Corpus] 升学增强 {stage}：{len(subjects)}科 × 20句/科 "
            f"= {len(subjects) * 20} 次 API 调用...",
            flush=True,
        )

        # 每科生成少量 API 增强句子（避免限流）
        batch_per_subject = 20
        new_train: List[str] = []
        new_exam: List[str] = []
        for subj in subjects:
            for _ in range(batch_per_subject):
                try:
                    line = self._teacher.sample_teaching_line(stage, subj)
                except Exception:
                    continue
                if not line:
                    continue
                if self.rng.random() < 0.8:
                    self._train_by_key.setdefault((stage, subj), []).append(line)
                    new_train.append(line)
                else:
                    new_exam.append(line)

        if new_train:
            self._train_by_stage.setdefault(stage, []).extend(new_train)
        if new_exam:
            self._exam_by_stage.setdefault(stage, []).extend(new_exam)

        print(
            f"[Corpus] {stage} 增强完成：+{len(new_train)}训练句 "
            f"+{len(new_exam)}考试句，用时{_time.time()-_t0:.1f}s",
            flush=True,
        )

    def _build_vocab(self, chars: Sequence[str]) -> None:
        counter = Counter(chars)
        specials = [
            "\n", " ", "，", "。", "？", "！", "、", "；", "：",
            "\u201c", "\u201d", "（", "）", "《", "》", ".", ",", "?", "!",
        ]
        ordered: List[str] = []
        for ch in specials:
            if ch in counter and ch not in ordered:
                ordered.append(ch)
        for ch, _ in counter.most_common():
            if ch not in ordered:
                ordered.append(ch)
            if len(ordered) >= self.vocab_size - 1:
                break
        self.char2idx = {ch: i for i, ch in enumerate(ordered)}
        # 确保 "?" 有映射（用于未知字符回退），如果不在词表中则追加到末尾
        if "?" not in self.char2idx and ordered:
            self.char2idx["?"] = len(ordered)
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}

    def vocab_len(self) -> int:
        return max(len(self.char2idx), 32)

    def char_to_index(self, ch: str) -> int:
        return self.char2idx.get(ch, self.char2idx.get("?", 0))

    def index_to_char(self, idx: int) -> str:
        return self.idx2char.get(idx, "?")

    def encode_char_indices(self, text: str) -> List[int]:
        return [self.char_to_index(c) for c in text]

    def list_subjects(self, stage: str) -> List[str]:
        return list(self._keys_by_stage.get(stage, []))

    def sample_subject(self, stage: str) -> Optional[str]:
        keys = self._keys_by_stage.get(stage, [])
        if not keys:
            return None
        weights = [
            max(len(self._train_by_key.get((stage, k), [])), 1) for k in keys
        ]
        w = np.array(weights, dtype=np.float64)
        w /= w.sum()
        return str(self.rng.choice(keys, p=w))

    def sample_training_line(self, stage: str, subject: Optional[str] = None) -> str:
        # 优先从缓存池取（初始化模板 + 升学时 API 增强的语料），
        # 避免每步训练都调 API 导致限流
        if subject and (stage, subject) in self._train_by_key:
            pool = self._train_by_key[(stage, subject)]
            if pool:
                return self.rng.choice(pool)
        if stage in UNIVERSITY_STAGES or subjects_for_stage(stage):
            subj = subject or self.sample_subject(stage)
            if subj and (stage, subj) in self._train_by_key:
                pool = self._train_by_key[(stage, subj)]
                if pool:
                    return self.rng.choice(pool)
        pool = self._train_by_stage.get(stage, [])
        if pool:
            return self.rng.choice(pool)
        # 池为空时回退到 NPC API 实时生成
        if self._npc_curriculum and self._teacher is not None:
            return self._teacher.sample_teaching_line(stage, subject)
        return self.rng.choice(_builtin_sensorimotor_lines())

    def get_exam_lines(self, stage: str, n: int = 20) -> List[str]:
        if n <= 0:
            return []
        # 优先从缓存考试池取（初始化模板 + API 增强的语料）
        pool = self._exam_by_stage.get(stage, [])
        if pool:
            n_use = min(n, len(pool))
            idx = self.rng.choice(len(pool), size=n_use, replace=False)
            out = [pool[int(i)] for i in idx]
            if len(out) >= n:
                return out
            # 不够则补充训练池
            train_pool = self._train_by_stage.get(stage, [])
            need = n - len(out)
            if train_pool:
                idx2 = self.rng.choice(len(train_pool), size=min(need, len(train_pool)), replace=False)
                out.extend(train_pool[int(i)] for i in idx2)
            return out
        # 考试池为空：回退到 NPC API 实时生成
        if self._npc_curriculum and self._teacher is not None:
            out = []
            subjects = self.list_subjects(stage) or ["通识"]
            for _ in range(n):
                subj = str(self.rng.choice(subjects))
                out.append(self._teacher.sample_teaching_line(stage, subj))
            return out
        pool = self._train_by_stage.get(stage, _builtin_sensorimotor_lines())
        n_use = min(n, len(pool))
        if n_use <= 0:
            return []
        idx = self.rng.choice(len(pool), size=n_use, replace=False)
        return [pool[int(i)] for i in idx]

    def stage_display_name(self, stage: str) -> str:
        return STAGE_DISPLAY.get(stage, stage)

    def corpus_stats(self) -> Dict[str, object]:
        return {
            "root": str(self.corpus_root),
            "files": self._file_count,
            "vocab": self.vocab_len(),
            "stages": {s: len(self._train_by_stage.get(s, [])) for s in STAGE_ORDER},
            "subjects": {s: self.list_subjects(s) for s in STAGE_ORDER},
            "npc_curriculum": self._npc_curriculum,
        }


def _default_corpus_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "corpus"


def _split_into_lines(text: str) -> List[str]:
    text = re.sub(r"\r\n?", "\n", text)
    parts = re.split(r"[\n。！？；]+", text)
    lines = [p.strip() for p in parts if len(p.strip()) >= 2]
    return lines if lines else [text.strip()[:500]]


def _split_train_exam(
    lines: List[str], ratio: float, rng: np.random.Generator
) -> Tuple[List[str], List[str]]:
    if len(lines) < 4:
        return lines, list(lines)
    idx = np.arange(len(lines))
    rng.shuffle(idx)
    n_exam = max(1, int(len(lines) * ratio))
    exam_idx = set(idx[:n_exam].tolist())
    train = [lines[i] for i in range(len(lines)) if i not in exam_idx]
    exam = [lines[i] for i in exam_idx]
    return train, exam


def _builtin_sensorimotor_lines() -> List[str]:
    """0~3 岁学语：短句、叠词、父母话，不是课本。"""
    return [
        "妈妈",
        "爸爸",
        "抱抱",
        "呀呀",
        "哇哇",
        "吃吃",
        "睡睡",
        "走走",
        "看看",
        "宝宝要",
        "妈妈抱",
        "爸爸好",
        "呀呀学语",
    ]
