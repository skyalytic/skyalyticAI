"""
NPC 家长/老师：按学段/科目动态生成“讲解-提问-纠错”教学语料与题型素材。

设计目标：
- 不依赖 data/corpus 静态文件
- 训练语料来自“上课互动”，而非堆无意义课件
- 输出以短句为主，便于字符级学习逐步提升
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Dict, List, Optional, Tuple

import numpy as np

from skyalyticAI.npc.teacher_service import TeacherService
from skyalyticAI.npc.persona_registry import build_personas, persona_to_dict

@dataclass(frozen=True)
class ReadingItem:
    passage: str
    question: str
    answer: str


@dataclass(frozen=True)
class ReasoningStep:
    prompt: str
    answer: str


class TeacherNPC:
    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.service = TeacherService.from_env()
        self.student_name = "小析"
        # 全生命周期身份库（不再是少量写死）
        self.personas: List[Dict[str, str]] = [persona_to_dict(p) for p in build_personas()]

    def _normalize_subject(self, subject: str) -> str:
        s = (subject or "").strip()
        # 中国语境常见别名对齐
        if s in ("政治", "思政", "思想政治"):
            return "政治"
        if s in ("道法", "道德法治", "道德与法治"):
            return "道德与法治"
        if s in ("马原", "毛概", "思修", "形势与政策", "马克思主义理论"):
            return "马克思主义"
        return s

    def pick_persona(self, stage: str, subject: Optional[str]) -> Dict[str, str]:
        subj = self._normalize_subject(subject or "")
        # 优先：同学段 + 同学科 的老师/教授
        def _match_subj(p_subj: Optional[str]) -> bool:
            if p_subj is None:
                return True
            ps = self._normalize_subject(p_subj)
            # 让“政治老师”也能覆盖道法/政治两套写法
            if subj in ("政治", "道德与法治") and ps in ("政治", "道德与法治"):
                return True
            return ps == subj

        candidates = [p for p in self.personas if (p.get("stage") in (None, stage)) and _match_subj(p.get("subject"))]
        if candidates:
            return candidates[int(self.rng.integers(0, len(candidates)))]
        # 退化：同学段任意
        stage_only = [p for p in self.personas if p.get("stage") in (None, stage)]
        if stage_only:
            return stage_only[int(self.rng.integers(0, len(stage_only)))]
        # 最后：全局任意
        return self.personas[int(self.rng.integers(0, len(self.personas)))]

    def _tone(self, stage: str) -> str:
        if stage in ("sensorimotor", "kindergarten"):
            return "温柔、短句、重复"
        if stage in ("primary", "middle"):
            return "清晰、分步骤、举例"
        if stage in ("high",):
            return "严谨、强调条件与推导"
        return "学术、定义-定理-例子-反例"

    def sample_subjects(self, stage: str, default_subjects: List[str]) -> List[str]:
        # 不做额外决策：直接使用配置里的科目/专业列表
        return list(default_subjects)

    def sample_teaching_line(self, stage: str, subject: Optional[str] = None) -> str:
        """生成一条课堂互动文本（讲解或提问或纠错）。"""
        subj = subject or "通识"
        tone = self._tone(stage)
        persona = self.pick_persona(stage, subj)
        if self.service is not None:
            system = (
                "你是一个陪伴式NPC，长期在真实生活/课堂中陪伴学生成长。"
                "你有明确身份与性格，不要自称AI。"
                "你要用简短口语化中文，一步步教学：讲解→提问→纠错→再练。"
                "输出只要一到两句，不要列表，不要长篇。"
            )
            user = (
                f"学生名字={self.student_name}。"
                f"你的身份={persona['role']}，性格={persona['style']}。"
                f"学段={stage}，科目/专业={subj}，教学风格={tone}。"
                "请对学生说话并带一个小问题或小练习。"
            )
            try:
                return self.service.chat(system, user, temperature=0.7, max_tokens=120)
            except Exception:
                # 失败则回退规则生成
                pass
        # 课堂脚本模板（强调“家长/老师”角色）
        templates = [
            f"老师：今天学{subj}。我会一步步讲，你跟着读。",
            f"老师：先给定义。{subj}里，关键是把概念说清楚。",
            f"老师：举个例子。请用一句话复述这个例子。",
            f"老师：我问你：为什么会这样？请按步骤回答。",
            f"老师：如果你答错了没关系，我们一起改正。",
            f"家长：做得好。再练一次，把错误的地方纠正。",
            f"老师：训练风格是{tone}。先读题，再找条件。",
            f"老师：请把“条件→结论”的链条写出来。",
            f"老师：现在我给你一个小测：请回答“是什么/为什么/怎么做”。",
        ]

        # 少量可控“知识点”词汇，避免纯空话
        nuggets = {
            "语文": ["主旨", "人物", "时间", "地点", "因果", "修辞"],
            "数学": ["已知", "求", "等于", "因此", "所以", "验证"],
            "英语": ["subject", "verb", "meaning", "because", "therefore"],
            "物理": ["力", "速度", "加速度", "能量", "守恒"],
            "化学": ["元素", "化学式", "反应", "守恒"],
            "生物": ["细胞", "遗传", "变异", "适应"],
            "历史": ["年代", "事件", "原因", "影响"],
            "地理": ["气候", "地形", "人口", "资源"],
            "道德与法治": ["权利", "义务", "规则", "责任"],
            "信息技术": ["算法", "数据", "程序", "调试"],
            "计算机": ["算法", "复杂度", "系统", "网络", "数据库"],
            "法学": ["权利", "义务", "合同", "侵权", "证据"],
            "医学": ["症状", "诊断", "治疗", "预防"],
        }
        words = nuggets.get(subj, ["概念", "条件", "步骤", "结论", "例子"])
        w1, w2 = self.rng.choice(words, size=2, replace=True)
        base = self.rng.choice(templates)
        # 拼成更像“教学”的一句
        return f"{persona['role']}：{base} 关键词：{w1}，{w2}。"

    def make_reading_item(self, stage: str, subject: Optional[str] = None) -> ReadingItem:
        subj = subject or "通识"
        if self.service is not None:
            system = (
                "你是中国课堂老师NPC。请生成阅读理解题，面向对应学段。"
                "必须输出严格JSON：{\"passage\":\"...\",\"question\":\"...\",\"answer\":\"...\"}。"
                "passage<=60字，answer<=10字。"
            )
            user = f"学段={stage}，科目/专业={subj}。生成1道题。"
            try:
                text = self.service.chat(system, user, temperature=0.6, max_tokens=220)
                obj = json.loads(text)
                return ReadingItem(
                    passage=str(obj.get("passage", ""))[:120],
                    question=str(obj.get("question", ""))[:80],
                    answer=str(obj.get("answer", ""))[:30],
                )
            except Exception:
                pass
        # 短文 + 问题 + 答案（答案尽量短，适合字符级监督）
        passage_templates = [
            f"{subj}课上，老师讲“条件和结论”。同学先找条件，再推出结论。",
            f"今天学{subj}。先读题，再画出关键字，再写出步骤。",
            f"家长陪读{subj}。孩子先复述要点，再做练习，最后纠错。",
        ]
        passage = str(self.rng.choice(passage_templates))
        question = "问题：这段话强调的学习顺序是什么？"
        answer = "读题找条件推结论"
        if stage in ("sensorimotor", "kindergarten"):
            question = "问题：先做什么？"
            answer = "先读题"
        return ReadingItem(passage=passage, question=question, answer=answer)

    def make_reasoning_chain(self, stage: str, subject: Optional[str] = None) -> List[ReasoningStep]:
        subj = subject or "数学"
        if self.service is not None:
            system = (
                "你是中国课堂老师NPC。请生成多步推理题链。"
                "必须输出严格JSON数组，每个元素形如：{\"prompt\":\"...\",\"answer\":\"...\"}。"
                "至少2步，prompt<=20字，answer<=6字。"
            )
            user = f"学段={stage}，科目/专业={subj}。生成1条多步推理链。"
            try:
                text = self.service.chat(system, user, temperature=0.6, max_tokens=260)
                arr = json.loads(text)
                steps: List[ReasoningStep] = []
                for it in (arr or []):
                    steps.append(
                        ReasoningStep(
                            prompt=str(it.get("prompt", ""))[:60],
                            answer=str(it.get("answer", ""))[:20],
                        )
                    )
                if len(steps) >= 2:
                    return steps
            except Exception:
                pass
        # 多步推理：保证>=2步
        if subj in ("数学", "科学", "物理", "化学", "计算机"):
            chain = [
                ReasoningStep(prompt="已知3加2等于", answer="5"),
                ReasoningStep(prompt="再用5乘2等于", answer="10"),
                ReasoningStep(prompt="最后10减3等于", answer="7"),
            ]
        elif subj in ("语文", "历史", "地理"):
            chain = [
                ReasoningStep(prompt="先找时间地点人物，再概括为", answer="主旨"),
                ReasoningStep(prompt="主旨需要用一句话表达，叫做", answer="概括"),
            ]
        else:
            chain = [
                ReasoningStep(prompt="先写条件，再写结论。条件叫", answer="已知"),
                ReasoningStep(prompt="结论前常用词是", answer="因此"),
            ]
        if stage in ("sensorimotor", "kindergarten"):
            chain = chain[:2]
        return chain

    def bootstrap_vocab_text(self) -> str:
        """用于词表构建的基础覆盖字符集合。"""
        base = (
            "老师家长学生今天我们一步步学读题条件结论为什么怎么做"
            "语文数学英语科学物理化学历史地理法治信息技术计算机法学医学"
            "加减乘除等于因此所以因为如果那么正确错误改正"
            "0123456789"
            "。，！？：；（）《》"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        )
        return base

