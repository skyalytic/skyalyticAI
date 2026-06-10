"""
NIEA 训练环境包。

提供所有训练环境的统一入口：
- Environment: 环境基类
- GridWorldEnv: 迷宫导航（感知运动期）
- HumanGrowthWorld: 类人成长世界（0~3岁到博士）
- SocialClassroomWorld: 社会课堂（NPC陪伴学习）
- TextWorldEnv: 纯文本环境（别名 HumanGrowthWorld）
- ExamWorld: 考试环境（别名 ExamSuite）
"""

from skyalyticAI.env.environment import Environment
from skyalyticAI.env.grid_world import GridWorldEnv
from skyalyticAI.env.curriculum_world import HumanGrowthWorld
from skyalyticAI.env.social_classroom_world import SocialClassroomWorld

# 别名：兼容 __init__.py 中的导入
TextWorldEnv = HumanGrowthWorld

# ExamWorld 别名：延迟导入避免循环依赖
def __getattr__(name: str):
    if name == "ExamWorld":
        from skyalyticAI.exams.exam_suite import ExamSuite
        return ExamSuite
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Environment",
    "GridWorldEnv",
    "HumanGrowthWorld",
    "SocialClassroomWorld",
    "TextWorldEnv",
    "ExamWorld",
]
