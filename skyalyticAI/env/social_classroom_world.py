"""
社会课堂世界 — 家长/老师陪伴的社会化学习环境。

继承 HumanGrowthWorld 的升学线、考试系统，
增加 NPC 互动和社会关系层。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from skyalyticAI.data.education_config import STAGE_ORDER, next_stage
from skyalyticAI.env.curriculum_world import Activity, HumanGrowthWorld
from skyalyticAI.npc.teacher_npc import TeacherNPC


class SocialClassroomWorld(HumanGrowthWorld):
    """社会课堂 — 在 NPC 陪伴下学习的成长环境。"""

    def __init__(
        self,
        corpus_root: Optional[str] = None,
        observation_dim: int = 128,
        school_stage: str = "sensorimotor",
        max_stage: str = "undergraduate",
        student_name: str = "小析",
        seed: int = 42,
    ) -> None:
        super().__init__(
            corpus_root=corpus_root,
            observation_dim=observation_dim,
            seed=seed,
        )
        # 覆盖学段（父类默认 sensorimotor）
        if school_stage in STAGE_ORDER:
            self.school_stage = school_stage
            from skyalyticAI.data.education_config import get_quality_spec
            self._spec = get_quality_spec(school_stage)
            self._exam_suite.set_stage(school_stage)

        self.max_stage = max_stage if max_stage in STAGE_ORDER else "undergraduate"

        # NPC 教师
        self._teacher = TeacherNPC(seed=seed + 100)
        self._teacher.student_name = student_name
        self.student_name = student_name

        # 社会特有状态
        self._current_persona: Optional[Dict[str, Any]] = None
        self._social_prompt: str = ""
        self._day_count: int = 0
        self._interaction_count: int = 0

    # ----- 社会化扩展 -----

    def _pick_persona(self) -> Dict[str, Any]:
        """选择当前互动的 NPC 角色。"""
        subject = self._current_subject or "通识"
        return self._teacher.pick_persona(self.school_stage, subject)

    def _generate_social_reading(self) -> str:
        """生成带社会上下文的教学文本。"""
        persona = self._pick_persona()
        self._current_persona = persona
        subject = self._current_subject or "通识"
        base_line = self._teacher.sample_teaching_line(self.school_stage, subject)
        role = persona.get("role", "老师")
        style = persona.get("style", "温和")
        return f"{role}({style})：{base_line}"

    def _start_reading(self) -> np.ndarray:
        """覆写：阅读模式加入社会上下文。"""
        self._current_subject = self._pick_subject()
        self._text_line = self._generate_social_reading()
        self._text_indices = self.corpus.encode_char_indices(self._text_line)
        if len(self._text_indices) < 2:
            self._text_indices = [0, 0]
        self._text_pos = 0
        self._text_correct = 0
        self._text_total = 0
        return self._reading_obs()

    def _step_reading(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """覆写：阅读步骤加入 NPC 信息。"""
        # 保存当前 persona（super()._step_reading 可能触发 _start_reading 更新 persona）
        old_persona = self._current_persona
        obs, reward, done, info = super()._step_reading(action)
        if old_persona:
            info["actor_role"] = old_persona.get("role", "")
            info["actor_style"] = old_persona.get("style", "")
            info["student_name"] = self.student_name
        return obs, reward, done, info

    def _try_promote(self) -> bool:
        """覆写：升学受 max_stage 限制。"""
        nxt = next_stage(self.school_stage)
        if STAGE_ORDER.index(nxt) > STAGE_ORDER.index(self.max_stage):
            return False
        if nxt == self.school_stage:
            return False
        self.school_stage = nxt
        from skyalyticAI.data.education_config import get_quality_spec
        self._spec = get_quality_spec(nxt)
        self._steps_in_stage = 0
        self._episodes_since_exam = 0
        self._exam_suite.set_stage(nxt)
        return True
