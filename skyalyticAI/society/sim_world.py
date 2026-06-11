"""
工业级社会模拟器（多模态、多智能体、可持续日程）。

特性：
1) 多智能体：家长/老师/同学/管理角色共存
2) 多模态：视觉(2D图)、听觉(波形)、文本上下文同时提供
3) 可持续：按“天-时段”推进，伴随复杂事件与长期关系更新
4) 与现有训练器兼容：保留 school_stage / subject / target_char 等字段
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from skyalyticAI.data.corpus_manager import CorpusManager
from skyalyticAI.data.education_config import STAGE_ORDER, core_subjects, get_quality_spec, next_stage
from skyalyticAI.env.environment import Environment
from skyalyticAI.env.curriculum_world import Activity
from skyalyticAI.language.text_encoder import TextEncoder
from skyalyticAI.npc.teacher_npc import TeacherNPC


class DaySlot(str, Enum):
    MORNING_HOME = "morning_home"
    SCHOOL_CLASS = "school_class"
    SCHOOL_BREAK = "school_break"
    AFTERNOON_ACTIVITY = "afternoon_activity"
    EVENING_STUDY = "evening_study"
    NIGHT_REFLECTION = "night_reflection"


@dataclass
class SocietyState:
    day: int
    slot: DaySlot
    school_stage: str
    subject: str
    actor_role: str
    actor_style: str
    event: str
    prompt_text: str
    target_answer: str
    answer_indices: List[int]
    pos: int
    correct: int
    total: int


class SocietySimWorld(Environment):
    def __init__(
        self,
        corpus_root: Optional[str] = None,
        observation_dim: int = 128,
        school_stage: str = "sensorimotor",
        max_stage: str = "undergraduate",
        student_name: str = "小析",
        image_size: int = 28,
        audio_len: int = 16000,
        seed: Optional[int] = None,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.observation_dim = observation_dim
        self.school_stage = school_stage if school_stage in STAGE_ORDER else "sensorimotor"
        self.max_stage = max_stage if max_stage in STAGE_ORDER else "undergraduate"
        self.student_name = student_name
        self.image_size = image_size
        self.audio_len = audio_len

        self.corpus = CorpusManager(corpus_root=corpus_root, seed=seed)
        self.vocab_size = max(self.corpus.vocab_len(), 32)
        self.teacher = TeacherNPC(seed=(seed or 0) + 123)
        self.teacher.student_name = student_name
        self.text_encoder = TextEncoder(vocab_size=self.vocab_size, output_dim=observation_dim, context_len=32)

        self._spec = get_quality_spec(self.school_stage)
        self._steps_in_stage = 0
        self._episodes_since_exam = 0
        self._day = 0
        self._slot_idx = 0
        self._state: Optional[SocietyState] = None
        self._ctx_indices: List[int] = []
        self._current_subject: Optional[str] = None

        # 长期关系图（-1~1）：与各角色关系亲密度
        self.relationships: Dict[str, float] = {}
        self._init_relationships()

        # 兼容 HumanGrowthTrainer 的 _activity 属性
        self._activity: Optional[Any] = None

    def _init_relationships(self) -> None:
        for p in self.teacher.personas:
            self.relationships[p["id"]] = 0.0

    # ----- 训练器兼容接口 -----
    def set_rolling_speech_accuracy(self, acc: float) -> None:
        pass

    def set_rolling_subject_accuracy(self, subject_acc: Dict[str, float]) -> None:
        pass

    def get_quality_spec(self):
        return get_quality_spec(self.school_stage)

    def get_steps_per_episode(self) -> int:
        return self._spec.steps_per_episode

    def set_stage(self, stage: str) -> None:
        if stage not in STAGE_ORDER:
            return
        self.school_stage = stage
        self._spec = get_quality_spec(stage)
        self._steps_in_stage = 0
        self._episodes_since_exam = 0

    # ----- 社会事件与角色 -----
    def _pick_slot(self) -> DaySlot:
        slots = list(DaySlot)
        slot = slots[self._slot_idx % len(slots)]
        self._slot_idx += 1
        if self._slot_idx % len(slots) == 0:
            self._day += 1
        return slot

    def _pick_subject(self, slot: DaySlot) -> str:
        if slot in (DaySlot.MORNING_HOME, DaySlot.SCHOOL_BREAK, DaySlot.NIGHT_REFLECTION):
            return "通识"
        core = core_subjects(self.school_stage)
        if core and self.rng.random() < 0.5:
            return str(self.rng.choice(core))
        return self.corpus.sample_subject(self.school_stage) or "通识"

    def _pick_event(self, slot: DaySlot) -> str:
        events = {
            DaySlot.MORNING_HOME: ["起床拖延", "早餐沟通", "出门准备"],
            DaySlot.SCHOOL_CLASS: ["课堂提问", "随堂测验", "板书讲解"],
            DaySlot.SCHOOL_BREAK: ["同伴冲突", "合作讨论", "课间放松"],
            DaySlot.AFTERNOON_ACTIVITY: ["体育训练", "社团活动", "实验实践"],
            DaySlot.EVENING_STUDY: ["作业复盘", "错题订正", "专题训练"],
            DaySlot.NIGHT_REFLECTION: ["日记反思", "家长复盘", "情绪整理"],
        }
        return str(self.rng.choice(events[slot]))

    def _build_prompt(self, slot: DaySlot, subject: str, event: str) -> Tuple[str, str, str]:
        persona = self.teacher.pick_persona(self.school_stage, subject)
        actor_role = persona["role"]
        actor_style = persona["style"]
        base = self.teacher.sample_teaching_line(self.school_stage, subject)
        prompt = (
            f"[第{self._day + 1}天/{slot.value}] {actor_role}({actor_style})："
            f"{base} 当前事件：{event}。请小析回应。"
        )
        return prompt, actor_role, actor_style

    def _target_answer(self, slot: DaySlot, subject: str, event: str) -> str:
        if slot == DaySlot.SCHOOL_CLASS:
            if subject in ("数学", "物理", "化学", "高等数学"):
                return "我先读题再列条件推结论"
            if subject in ("马克思主义", "道德与法治", "政治"):
                return "我会用观点依据结论作答"
            return "我先概括主旨再解释理由"
        if slot == DaySlot.SCHOOL_BREAK:
            return "我先沟通再合作解决问题"
        if slot == DaySlot.NIGHT_REFLECTION:
            return "今天我学到并会复盘改错"
        if slot == DaySlot.MORNING_HOME:
            return "我会按计划出发并保持专注"
        if slot == DaySlot.AFTERNOON_ACTIVITY:
            return "我会先热身再训练并复盘"
        return "我会完成作业并订正错题"

    # ----- 多模态观测 -----
    def _make_visual(self, slot: DaySlot, event: str) -> np.ndarray:
        img = np.zeros((self.image_size, self.image_size), dtype=np.float64)
        # 用 slot / event hash 生成可重复图样（模拟视觉场景变化）
        seed_val = (hash(slot.value + event) % 10_000) / 10_000.0
        x = int(seed_val * (self.image_size - 1))
        y = int((1.0 - seed_val) * (self.image_size - 1))
        img[max(0, y - 2): min(self.image_size, y + 3), max(0, x - 2): min(self.image_size, x + 3)] = 1.0
        img += self.rng.random((self.image_size, self.image_size)) * 0.05
        img = np.clip(img, 0.0, 1.0)
        return img

    def _make_audio(self, text: str) -> np.ndarray:
        # 轻量“语音”模拟：根据文本hash生成多频正弦叠加
        t = np.linspace(0, 1.0, self.audio_len, endpoint=False)
        h = abs(hash(text)) % 1000
        f1 = 180 + (h % 200)
        f2 = 320 + (h % 180)
        wave = 0.5 * np.sin(2 * np.pi * f1 * t) + 0.3 * np.sin(2 * np.pi * f2 * t)
        wave += 0.02 * self.rng.standard_normal(self.audio_len)
        return np.clip(wave, -1.0, 1.0).astype(np.float64)

    def _obs_dict(self) -> Dict[str, Any]:
        assert self._state is not None
        raw = self.text_encoder.encode(self._ctx_indices[-32:])
        return {
            "visual": self._make_visual(self._state.slot, self._state.event),
            "audio": self._make_audio(self._state.prompt_text),
            "raw_observation": raw,
        }

    # ----- 环境主循环 -----
    def reset(self) -> Dict[str, Any]:
        self._episodes_since_exam += 1
        self._spec = get_quality_spec(self.school_stage)
        self._activity = Activity.READING  # 社会课堂始终为阅读模式，启用语言头
        slot = self._pick_slot()
        subject = self._pick_subject(slot)
        self._current_subject = subject
        event = self._pick_event(slot)
        prompt, actor_role, actor_style = self._build_prompt(slot, subject, event)
        target = self._target_answer(slot, subject, event)
        ans_idx = self.corpus.encode_char_indices(target)
        if not ans_idx:
            ans_idx = [0]

        self._state = SocietyState(
            day=self._day,
            slot=slot,
            school_stage=self.school_stage,
            subject=subject,
            actor_role=actor_role,
            actor_style=actor_style,
            event=event,
            prompt_text=prompt,
            target_answer=target,
            answer_indices=ans_idx,
            pos=0,
            correct=0,
            total=0,
        )
        self._ctx_indices = self.corpus.encode_char_indices(prompt)
        return self._obs_dict()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self._state is None:
            return self.reset(), 0.0, False, {"mode": "society"}

        action = int(action) % self.vocab_size
        target = self._state.answer_indices[self._state.pos] if self._state.pos < len(self._state.answer_indices) else 0
        ok = action == target
        self._state.total += 1
        if ok:
            self._state.correct += 1
        self._state.pos += 1
        self._steps_in_stage += 1
        self._ctx_indices.append(action)
        if len(self._ctx_indices) > 1000:
            self._ctx_indices = self._ctx_indices[-500:]

        # 关系更新：答对提升“当前角色体验”，答错轻微下降
        persona = self.teacher.pick_persona(self.school_stage, self._state.subject)
        pid = persona["id"]
        delta = 0.01 if ok else -0.004
        self.relationships[pid] = float(np.clip(self.relationships.get(pid, 0.0) + delta, -1.0, 1.0))

        reward = 1.0 if ok else -0.2
        done = self._state.pos >= len(self._state.answer_indices)

        info: Dict[str, Any] = {
            "mode": "society",
            "activity": "reading",
            "school_stage": self.school_stage,
            "subject": self._state.subject,
            "slot": self._state.slot.value,
            "event": self._state.event,
            "actor_role": self._state.actor_role,
            "actor_style": self._state.actor_style,
            "teacher_text": self._state.prompt_text,
            "target_text": self._state.target_answer,
            "correct": ok,
            "target_char": self.corpus.index_to_char(target),
            "spoken_char": self.corpus.index_to_char(action),
            "relationship": self.relationships.get(pid, 0.0),
        }

        if done:
            acc = self._state.correct / max(self._state.total, 1)
            info["accuracy"] = acc
            promoted = False
            if (
                self._steps_in_stage >= self._spec.min_steps_in_stage
                and acc >= max(0.45, self._spec.min_rolling_speech_accuracy)
                and self._episodes_since_exam >= self._spec.min_episodes_between_exams
                and self._spec.allows_subject_exam
            ):
                promoted = self._promote_stage()
            info["promoted"] = promoted
            info["passed"] = promoted

        return self._obs_dict(), reward, done, info

    def _promote_stage(self) -> bool:
        nxt = next_stage(self.school_stage)
        if STAGE_ORDER.index(nxt) > STAGE_ORDER.index(self.max_stage):
            return False
        if nxt != self.school_stage:
            self.school_stage = nxt
            self._steps_in_stage = 0
            self._episodes_since_exam = 0
            self._spec = get_quality_spec(self.school_stage)
            return True
        return False

    def get_observation_dim(self) -> int:
        return self.observation_dim

    def get_action_dim(self) -> int:
        return self.vocab_size

    def render(self):
        return None

