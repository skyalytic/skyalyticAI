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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from skyalyticAI.data.corpus_manager import CorpusManager
from skyalyticAI.data.education_config import STAGE_ORDER, core_subjects, get_quality_spec, next_stage
from skyalyticAI.env.environment import Environment
from skyalyticAI.env.curriculum_world import Activity
from skyalyticAI.language.text_encoder import TextEncoder
from skyalyticAI.npc.teacher_npc import TeacherNPC
from skyalyticAI.society.speech_synth import SpeechSynthesizer


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
        audio_len: int = 96000,
        assets_dir: Optional[str] = "assets",
        real_perception: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        import time as _time
        _t0 = _time.time()
        self.rng = np.random.default_rng(seed)
        self.observation_dim = observation_dim
        self.school_stage = school_stage if school_stage in STAGE_ORDER else "sensorimotor"
        self.max_stage = max_stage if max_stage in STAGE_ORDER else "undergraduate"
        self.student_name = student_name
        self.image_size = image_size
        self.audio_len = audio_len
        print(f"[SocietySim] 基础属性: {_time.time()-_t0:.1f}s", flush=True)

        _t0 = _time.time()
        self.corpus = CorpusManager(corpus_root=corpus_root, seed=seed)
        print(f"[SocietySim] CorpusManager 完成: {_time.time()-_t0:.1f}s", flush=True)

        self.vocab_size = max(self.corpus.vocab_len(), 32)

        _t0 = _time.time()
        self.teacher = TeacherNPC(seed=(seed or 0) + 123)
        print(f"[SocietySim] TeacherNPC 完成: {_time.time()-_t0:.1f}s", flush=True)

        self.teacher.student_name = student_name
        _t0 = _time.time()
        self.text_encoder = TextEncoder(vocab_size=self.vocab_size, output_dim=observation_dim, context_len=32)
        print(f"[SocietySim] TextEncoder 完成: {_time.time()-_t0:.1f}s", flush=True)

        self._spec = get_quality_spec(self.school_stage)
        self._steps_in_stage = 0
        self._episodes_since_exam = 0
        # 滚动说话准确率（由 trainer 在每 episode 开头通过 set_rolling_speech_accuracy 传入）
        # 用于升学判定，确保"持续达标"而非"偶然波动"
        self._rolling_speech_acc: float = 0.0
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

        # fix 119 教师强制开关：训练时上下文回喂目标字（听正确示范），
        # 评估时置 False 回喂模型自己的输出
        self.teacher_forcing = True

        # fix 119 真实感知资产：assets/images/ 放真实照片（{slot}_{event}.jpg 等命名），
        # TTS 真实语音缓存在 assets/audio/tts_cache/
        self.assets_dir = assets_dir
        self.real_perception = real_perception
        self.speech_synth: Optional[SpeechSynthesizer] = None
        if real_perception:
            cache = str(Path(assets_dir) / "audio" / "tts_cache") if assets_dir else "assets/audio/tts_cache"
            self.speech_synth = SpeechSynthesizer(cache_dir=cache)

    def _init_relationships(self) -> None:
        for p in self.teacher.personas:
            self.relationships[p["id"]] = 0.0

    # ----- 训练器兼容接口 -----
    def set_rolling_speech_accuracy(self, acc: float) -> None:
        self._rolling_speech_acc = acc

    def set_rolling_subject_accuracy(self, subject_acc: Dict[str, float]) -> None:
        pass

    def get_quality_spec(self):
        return get_quality_spec(self.school_stage)

    def get_steps_per_episode(self) -> int:
        return self._spec.steps_per_episode

    def set_stage(self, stage: str) -> None:
        if stage not in STAGE_ORDER:
            return
        old_stage = self.school_stage
        self.school_stage = stage
        self._spec = get_quality_spec(stage)
        self._steps_in_stage = 0
        self._episodes_since_exam = 0
        # 升学时按需用 API 增强新学段语料（懒加载，每学段只增强一次）
        if stage != old_stage:
            self.corpus._load_stage_curriculum(stage)

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
        if subject == "说话":
            return "我会跟着老师学说话"
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
    def _load_visual_asset(self, slot: DaySlot, event: str) -> Optional[np.ndarray]:
        """真实照片优先：assets/images/{slot}_{event}.* -> {slot}.* -> default.*。"""
        if not self.real_perception or not self.assets_dir:
            return None
        try:
            from PIL import Image
        except Exception:
            return None
        img_dir = Path(self.assets_dir) / "images"
        if not img_dir.is_dir():
            return None
        for stem in (f"{slot.value}_{event}", slot.value, "default"):
            for ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
                path = img_dir / f"{stem}{ext}"
                if path.is_file():
                    try:
                        with Image.open(path) as im:
                            return np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
                    except Exception:
                        continue
        return None

    def _render_scene_card(self, slot: DaySlot, event: str) -> Optional[np.ndarray]:
        """无真实照片时的结构化场景卡：PIL 渲染的真实光学图像（含边缘/区域/颜色结构，非哈希点）。"""
        try:
            from PIL import Image, ImageDraw
        except Exception:
            return None
        try:
            size = 224
            palette = {
                DaySlot.MORNING_HOME: ((255, 214, 170), (120, 85, 60)),
                DaySlot.SCHOOL_CLASS: ((180, 220, 255), (70, 110, 160)),
                DaySlot.SCHOOL_BREAK: ((200, 240, 200), (60, 120, 70)),
                DaySlot.AFTERNOON_ACTIVITY: ((255, 200, 120), (160, 90, 40)),
                DaySlot.EVENING_STUDY: ((150, 160, 210), (50, 55, 90)),
                DaySlot.NIGHT_REFLECTION: ((40, 50, 90), (15, 20, 40)),
            }
            sky, ground = palette.get(slot, ((200, 200, 200), (100, 100, 100)))
            img = Image.new("RGB", (size, size), sky)
            draw = ImageDraw.Draw(img)
            horizon = int(size * 0.7)
            for y in range(horizon):  # 天空竖直渐变
                t = y / max(horizon, 1)
                c = tuple(int(sky[i] * (0.7 + 0.3 * t)) for i in range(3))
                draw.line([(0, y), (size, y)], fill=c)
            draw.rectangle([0, horizon, size, size], fill=ground)
            # 事件相关的确定性几何主体（真实边缘与区域结构）
            h = int.from_bytes((slot.value + event).encode("utf-8"), "little")
            cx, cy = 40 + h % (size - 80), 60 + (h // 7) % max(size // 2, 1)
            r = 18 + (h // 13) % 26
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 250, 230), outline=(60, 60, 60), width=2)
            bx, bw, bh = 30 + (h // 3) % 120, 50 + (h // 5) % 60, 30 + (h // 11) % 50
            draw.rectangle([bx, horizon - bh, bx + bw, horizon], fill=(120, 80, 50), outline=(40, 25, 15), width=2)
            for i in range(3):
                x0 = (h // (17 * (i + 1))) % (size - 20) + 10
                hh = 15 + (h // (23 + i)) % 35
                draw.rectangle([x0, horizon - hh, x0 + 8, horizon], fill=(50, 90 + (h >> i) % 80, 50))
            return np.asarray(img, dtype=np.float64) / 255.0
        except Exception:
            return None

    def _make_visual(self, slot: DaySlot, event: str) -> np.ndarray:
        # fix 119 真实感知优先：真实照片 > 结构化场景卡 > 旧哈希点阵（最后兜底）
        real = self._load_visual_asset(slot, event)
        if real is not None:
            return real
        card = self._render_scene_card(slot, event)
        if card is not None:
            return card
        img = np.zeros((self.image_size, self.image_size), dtype=np.float64)
        # 修复：使用确定性哈希（避免 PYTHONHASHSEED 导致跨进程不可复现）
        seed_val = (int.from_bytes((slot.value + event).encode("utf-8"), "little") % 10_000) / 10_000.0
        x = int(seed_val * (self.image_size - 1))
        y = int((1.0 - seed_val) * (self.image_size - 1))
        img[max(0, y - 2): min(self.image_size, y + 3), max(0, x - 2): min(self.image_size, x + 3)] = 1.0
        img += self.rng.random((self.image_size, self.image_size)) * 0.05
        img = np.clip(img, 0.0, 1.0)
        return img

    def _make_audio(self, text: str) -> np.ndarray:
        # fix 121 类人听觉：不限时长连续处理，耳蜗编码器自适应压缩到固定维度。
        # 人耳不限时长——声波连续进入耳蜗，毛细胞以~10-30ms窗口提取频谱，
        # 听神经发放脉冲后由皮层整合。CochleaEncoder 的 spike_steps=20 相当于
        # 皮层整合窗口，把任意时长音频压缩到 (n_fibers, spike_steps) 固定维度。
        # 因此无需截断/补零到固定长度——短音频原样返回，长音频完整保留。
        if self.speech_synth is not None:
            try:
                out = self.speech_synth.synthesize(text)
                if out is not None:
                    wave, sr = out
                    if wave.shape[0] > 0:
                        # 安全上限：60秒，避免极端长音频导致内存问题
                        max_samples = int(60.0 * sr)
                        if wave.shape[0] > max_samples:
                            wave = wave[:max_samples]
                        if sr != 16000:  # 统一到16k，与耳蜗编码器工作采样率一致
                            duration = wave.shape[0] / float(sr)
                            n = max(1, int(duration * 16000))
                            wave = np.interp(
                                np.linspace(0.0, duration, n, endpoint=False),
                                np.linspace(0.0, duration, wave.shape[0], endpoint=False),
                                wave,
                            )
                        return wave  # 变长返回，CochleaEncoder 自动处理
            except Exception:
                pass  # TTS 失败时回退到正弦模拟
        # 轻量"语音"模拟：根据文本hash生成多频正弦叠加
        t = np.linspace(0, 1.0, self.audio_len, endpoint=False)
        # 修复：使用确定性哈希
        h = int.from_bytes(text.encode("utf-8"), "little") % 1000
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
        # fix 119 课程阶梯：感知运动期只练单一模板（先学说一句话），
        # 幼儿园固定课堂场景、开放全部科目模板，小学起开放全部场景模板。
        if self.school_stage == "sensorimotor":
            slot = DaySlot.SCHOOL_CLASS
            subject = "说话"
        elif self.school_stage == "kindergarten":
            slot = DaySlot.SCHOOL_CLASS
            subject = self._pick_subject(slot)
        else:
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

    def _start_new_round(self) -> None:
        """Episode 内启动新一轮独立问答（fix 120）。

        一轮问答答完后不结束 episode，而是重新生成独立的 prompt+target，
        brain state 与上下文跨轮保持（同一节课持续学习）。
        消除简单重复拼接导致的周期性虚假学习信号，
        一个 episode = 一节课内多轮师生问答，符合类人学习理论。
        """
        if self._state is None:
            return
        if self.school_stage == "sensorimotor":
            slot = DaySlot.SCHOOL_CLASS
            subject = "说话"
        elif self.school_stage == "kindergarten":
            slot = DaySlot.SCHOOL_CLASS
            subject = self._pick_subject(slot)
        else:
            slot = self._pick_slot()
            subject = self._pick_subject(slot)
        self._current_subject = subject
        event = self._pick_event(slot)
        prompt, actor_role, actor_style = self._build_prompt(slot, subject, event)
        target = self._target_answer(slot, subject, event)
        ans_idx = self.corpus.encode_char_indices(target) or [0]
        st = self._state
        st.slot, st.school_stage, st.subject = slot, self.school_stage, subject
        st.actor_role, st.actor_style, st.event = actor_role, actor_style, event
        st.prompt_text, st.target_answer, st.answer_indices = prompt, target, ans_idx
        st.pos = 0  # 新一轮从答案首字开始；correct/total 累计以统计轮内准确率
        self._ctx_indices.extend(self.corpus.encode_char_indices(prompt))
        if len(self._ctx_indices) > 1000:
            self._ctx_indices = self._ctx_indices[-500:]

    def _check_promotion(self, acc: float) -> bool:
        # 使用 trainer 维护的滚动准确率（episode 级更新），确保升学是"持续达标"而非"偶然波动"
        # acc 参数保留兼容签名，内部统一用 rolling_acc 判定
        rolling_acc = self._rolling_speech_acc
        # 非考试学段（sensorimotor）：步数+滚动说话率达标即可自动升学
        # 理论依据：0~3岁感知运动期无笔试，按发展里程碑（步数+咿呀学语准确率）升学
        if not self._spec.allows_subject_exam:
            if (self._steps_in_stage >= self._spec.min_steps_in_stage
                and rolling_acc >= self._spec.min_rolling_speech_accuracy):
                return self._promote_stage()
            return False
        # 考试学段（幼儿园及以上）：步数+滚动准确率+考试间隔+允许考试
        # 阈值直接取配置的 min_rolling_speech_accuracy，不拔高（各学段配置已合理）
        if not (
            self._steps_in_stage >= self._spec.min_steps_in_stage
            and rolling_acc >= self._spec.min_rolling_speech_accuracy
            and self._episodes_since_exam >= self._spec.min_episodes_between_exams
            and self._spec.allows_subject_exam
        ):
            return False
        return self._promote_stage()

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
        # fix 119 教师强制回喂：训练时把目标字（正确示范）追加进上下文，
        # 而非模型自己说错的字——等价于婴儿听大人的正确发音，而非只听自己的咿呀。
        feedback_char = target if self.teacher_forcing else action
        self._ctx_indices.append(feedback_char)
        if len(self._ctx_indices) > 1000:
            self._ctx_indices = self._ctx_indices[-500:]

        # 关系更新：答对提升“当前角色体验”，答错轻微下降
        persona = self.teacher.pick_persona(self.school_stage, self._state.subject)
        pid = persona["id"]
        delta = 0.01 if ok else -0.004
        self.relationships[pid] = float(np.clip(self.relationships.get(pid, 0.0) + delta, -1.0, 1.0))

        reward = 1.0 if ok else -0.2
        round_done = self._state.pos >= len(self._state.answer_indices)

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

        if round_done:
            # fix 120：一轮问答答完后不结束 episode，而是启动新一轮独立问答。
            # 升学检查在每轮结束时做（基于累计步数+累计准确率），episode 长度由
            # trainer 的 steps_per_episode 控制，使每 episode 跑满预算、步数达标。
            acc = self._state.correct / max(self._state.total, 1)
            info["accuracy"] = acc
            promoted = self._check_promotion(acc)
            info["promoted"] = promoted
            info["passed"] = promoted
            self._start_new_round()

        # done 恒为 False：episode 由 trainer 的 max_steps 终止，使每 episode 跑满预算
        return self._obs_dict(), reward, False, info

    def _promote_stage(self) -> bool:
        nxt = next_stage(self.school_stage)
        if STAGE_ORDER.index(nxt) > STAGE_ORDER.index(self.max_stage):
            return False
        if nxt != self.school_stage:
            self.school_stage = nxt
            self._steps_in_stage = 0
            self._episodes_since_exam = 0
            self._spec = get_quality_spec(self.school_stage)
            # 升学时按需用 API 增强新学段语料（懒加载）
            self.corpus._load_stage_curriculum(nxt)
            return True
        return False

    def get_observation_dim(self) -> int:
        return self.observation_dim

    def get_action_dim(self) -> int:
        return self.vocab_size

    def render(self):
        return None

