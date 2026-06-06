"""
中国教育路径（工业级配置）。

两层含义（不要混）：
1. 人生阶段 school_stage：按年龄/学历升学（含 0~3 岁感知运动期，不含「外语学段」）
2. 课程 subject：在同一学段内并行（语文、数学、英语…）；大学再叠加专业 major

0~3 岁（sensorimotor）：学走路、抓握、听声、咿呀学语 —— 以迷宫+极简语音为主，不考升学笔试。
幼儿园起（kindergarten~phd）：正式上学路径，多科轮换，英语是科目不是学段。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

# 人生升学线（与 data/corpus 顶层目录一一对应）
STAGE_ORDER: List[str] = [
    "sensorimotor",   # 00  0~3 岁：走路、说话、感知运动（非学校教育）
    "kindergarten",   # 01  幼儿园
    "primary",        # 02  小学
    "middle",         # 03  初中
    "high",           # 04  高中
    "undergraduate",  # 05  本科
    "master",         # 06  硕士
    "phd",            # 07  博士
]

STAGE_DIR_MAP: Dict[str, str] = {
    "00_sensorimotor": "sensorimotor",
    "00_infant": "sensorimotor",       # 兼容旧目录名
    "01_kindergarten": "kindergarten",
    "02_primary": "primary",
    "03_middle": "middle",
    "04_high": "high",
    "05_undergraduate": "undergraduate",
    "04_undergraduate": "undergraduate",
    "04_university": "undergraduate",
    "06_master": "master",
    "05_master": "master",
    "07_phd": "phd",
    "06_phd": "phd",
}

STAGE_DISPLAY: Dict[str, str] = {
    "sensorimotor": "0~3岁感知运动",
    "kindergarten": "幼儿园",
    "primary": "小学",
    "middle": "初中",
    "high": "高中",
    "undergraduate": "本科",
    "master": "硕士",
    "phd": "博士",
}

# 义务教育阶段核心科目（含英语课 —— 中国课程标准）
PRIMARY_SUBJECTS: List[str] = ["语文", "数学", "英语", "科学", "道德与法治"]
MIDDLE_SUBJECTS: List[str] = [
    "语文", "数学", "英语", "物理", "化学", "生物", "历史", "地理", "道德与法治",
]
HIGH_SUBJECTS: List[str] = MIDDLE_SUBJECTS + ["信息技术"]

# 幼儿园以启蒙为主
KINDERGARTEN_SUBJECTS: List[str] = ["语言", "数学启蒙", "英语启蒙", "艺术", "健康"]

# 本科公共课（中国大学普遍存在：思政/英语/数学/计算机基础等）
UNDERGRADUATE_PUBLIC_SUBJECTS: List[str] = [
    "马克思主义",
    "大学英语",
    "高等数学",
    "计算机基础",
    "体育",
    "心理健康",
]

# 大学阶段用专业文件夹；下列为默认专业名（用户可增删）
DEFAULT_MAJORS: List[str] = [
    "哲学", "经济学", "法学", "教育学", "文学", "历史学",
    "理学", "工学", "农学", "医学", "管理学", "艺术学",
    "计算机", "数学", "物理", "化学", "生物", "中文",
    "英语", "日语", "自动化", "电子信息", "机械", "土木", "临床医学",
]

UNIVERSITY_STAGES = frozenset({"undergraduate", "master", "phd"})

# 非学校教育阶段：不参加高考式升学，或仅做发育检查
NON_SCHOOL_EXAM_STAGES = frozenset({"sensorimotor"})
LIGHT_EXAM_STAGES = frozenset({"kindergarten"})

# 升学核心科目（用于防止偏科）
CORE_SUBJECTS_BY_STAGE: Dict[str, List[str]] = {
    "kindergarten": ["语言"],
    "primary": ["语文", "数学", "英语"],
    "middle": ["语文", "数学", "英语"],
    "high": ["语文", "数学", "英语"],
}

# 核心科目最低滚动准确率门槛（可按阶段调整）
CORE_SUBJECT_MIN_ACCURACY: Dict[str, Dict[str, float]] = {
    "kindergarten": {"语言": 0.20},
    "primary": {"语文": 0.30, "数学": 0.28, "英语": 0.25},
    "middle": {"语文": 0.38, "数学": 0.36, "英语": 0.34},
    "high": {"语文": 0.45, "数学": 0.43, "英语": 0.40},
}


def core_subjects(stage: str) -> List[str]:
    return CORE_SUBJECTS_BY_STAGE.get(stage, [])


def core_subject_min_accuracy(stage: str, subject: str) -> float:
    return CORE_SUBJECT_MIN_ACCURACY.get(stage, {}).get(subject, 0.0)


def subjects_for_stage(stage: str) -> List[str]:
    """返回该学段应轮换的课程列表。"""
    if stage == "sensorimotor":
        return []  # 0~3 岁无正式科目，只有感知运动+学语
    if stage == "kindergarten":
        return KINDERGARTEN_SUBJECTS
    if stage == "primary":
        return PRIMARY_SUBJECTS
    if stage == "middle":
        return MIDDLE_SUBJECTS
    if stage == "high":
        return HIGH_SUBJECTS
    if stage == "undergraduate":
        return UNDERGRADUATE_PUBLIC_SUBJECTS
    return []


@dataclass(frozen=True)
class StageQualitySpec:
    """升学前最低训练量（步数够、说话准再升学）。"""

    min_steps_in_stage: int
    min_rolling_speech_accuracy: float
    exam_pass_accuracy: float
    steps_per_episode: int
    exam_questions: int
    min_episodes_between_exams: int
    motor_ratio: float
    reading_ratio: float
    allows_subject_exam: bool


STAGE_QUALITY: Dict[str, StageQualitySpec] = {
    # 0~3 岁：走路探索为主，少量学语；无笔试升学
    "sensorimotor": StageQualitySpec(
        min_steps_in_stage=600,
        min_rolling_speech_accuracy=0.12,
        exam_pass_accuracy=0.0,
        steps_per_episode=80,
        exam_questions=0,
        min_episodes_between_exams=9999,
        motor_ratio=0.85,
        reading_ratio=0.15,
        allows_subject_exam=False,
    ),
    "kindergarten": StageQualitySpec(
        min_steps_in_stage=1500,
        min_rolling_speech_accuracy=0.22,
        exam_pass_accuracy=0.45,
        steps_per_episode=70,
        exam_questions=10,
        min_episodes_between_exams=12,
        motor_ratio=0.45,
        reading_ratio=0.55,
        allows_subject_exam=True,
    ),
    "primary": StageQualitySpec(
        min_steps_in_stage=4000,
        min_rolling_speech_accuracy=0.35,
        exam_pass_accuracy=0.55,
        steps_per_episode=90,
        exam_questions=25,
        min_episodes_between_exams=10,
        motor_ratio=0.25,
        reading_ratio=0.75,
        allows_subject_exam=True,
    ),
    "middle": StageQualitySpec(
        min_steps_in_stage=10000,
        min_rolling_speech_accuracy=0.45,
        exam_pass_accuracy=0.58,
        steps_per_episode=110,
        exam_questions=30,
        min_episodes_between_exams=12,
        motor_ratio=0.15,
        reading_ratio=0.85,
        allows_subject_exam=True,
    ),
    "high": StageQualitySpec(
        min_steps_in_stage=18000,
        min_rolling_speech_accuracy=0.52,
        exam_pass_accuracy=0.60,
        steps_per_episode=130,
        exam_questions=35,
        min_episodes_between_exams=15,
        motor_ratio=0.10,
        reading_ratio=0.90,
        allows_subject_exam=True,
    ),
    "undergraduate": StageQualitySpec(
        min_steps_in_stage=45000,
        min_rolling_speech_accuracy=0.58,
        exam_pass_accuracy=0.62,
        steps_per_episode=160,
        exam_questions=40,
        min_episodes_between_exams=20,
        motor_ratio=0.05,
        reading_ratio=0.95,
        allows_subject_exam=True,
    ),
    "master": StageQualitySpec(
        min_steps_in_stage=75000,
        min_rolling_speech_accuracy=0.65,
        exam_pass_accuracy=0.65,
        steps_per_episode=190,
        exam_questions=45,
        min_episodes_between_exams=25,
        motor_ratio=0.03,
        reading_ratio=0.97,
        allows_subject_exam=True,
    ),
    "phd": StageQualitySpec(
        min_steps_in_stage=120000,
        min_rolling_speech_accuracy=0.72,
        exam_pass_accuracy=0.68,
        steps_per_episode=220,
        exam_questions=50,
        min_episodes_between_exams=30,
        motor_ratio=0.02,
        reading_ratio=0.98,
        allows_subject_exam=True,
    ),
}


def get_quality_spec(stage: str) -> StageQualitySpec:
    return STAGE_QUALITY.get(stage, STAGE_QUALITY["sensorimotor"])


def next_stage(current: str) -> str:
    if current not in STAGE_ORDER:
        return STAGE_ORDER[0]
    i = STAGE_ORDER.index(current)
    if i + 1 < len(STAGE_ORDER):
        return STAGE_ORDER[i + 1]
    return current


def is_school_stage(stage: str) -> bool:
    return stage not in NON_SCHOOL_EXAM_STAGES
