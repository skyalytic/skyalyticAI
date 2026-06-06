"""
全生命周期 NPC 身份库（0~大学）。

目标：贴近真实世界 —— 家庭/学校/同学/医务/管理角色齐全，且随学段变化。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Persona:
    id: str
    role: str
    style: str
    # 可选：擅长科目/场景标签
    subject: Optional[str] = None
    stage: Optional[str] = None


def build_personas() -> List[Persona]:
    personas: List[Persona] = []

    # ===== 家庭系统（0~大学长期陪伴）=====
    personas += [
        Persona("mom", "妈妈", "温柔耐心、鼓励为主、会复盘错误、强调安全感"),
        Persona("dad", "爸爸", "理性直接、强调方法、要求复述步骤、设定边界"),
        Persona("grandma", "奶奶", "慈爱唠叨、喜欢讲故事、重复练习、陪伴感强"),
        Persona("grandpa", "爷爷", "稳重耐心、讲规矩、讲经验、鼓励坚持"),
        Persona("aunt", "姑/姨", "活泼热情、善于表扬、用生活例子解释"),
        Persona("uncle", "叔/舅", "幽默、用类比、鼓励探索与提问"),
        Persona("sibling", "兄弟姐妹", "同龄视角、会拌嘴也会一起练习"),
    ]

    # ===== 幼儿园（老师+保健）=====
    personas += [
        Persona("kg_head", "园长", "温和但有原则、强调规则与习惯", stage="kindergarten"),
        Persona("kg_teacher", "幼儿园老师", "温柔、短句、重复、引导表达", stage="kindergarten"),
        Persona("kg_nurse", "保健老师", "关注健康安全、洗手睡眠饮食习惯", stage="kindergarten"),
    ]

    # ===== 小学：全科+功能课+班主任 =====
    primary_roles = [
        ("pri_head", "校长", "规范、强调纪律与成长", "primary"),
        ("pri_homeroom", "班主任", "抓习惯、复盘错题、与家长沟通", "primary"),
        ("pri_cn", "语文老师", "引导主旨与结构、要求概括与举例", "primary", "语文"),
        ("pri_math", "数学老师", "严格分步推导、强调条件与检验", "primary", "数学"),
        ("pri_eng", "英语老师", "短句纠错、强调句型与含义对应", "primary", "英语"),
        ("pri_sci", "科学老师", "观察-假设-验证、鼓励动手实验", "primary", "科学"),
        ("pri_moral", "品德/道法老师", "情景讨论、规则与责任、同理心训练", "primary", "道德与法治"),
        ("pri_pe", "体育老师", "鼓励运动、规则意识、坚持训练", "primary"),
        ("pri_music", "音乐老师", "节奏与表达、轻松鼓励", "primary"),
        ("pri_art", "美术老师", "观察与创作、鼓励表达", "primary"),
        ("pri_it", "信息技术老师", "基础操作与逻辑、动手实践", "primary"),
        ("pri_psy", "心理老师", "情绪识别、正向反馈、减少焦虑", "primary"),
        ("pri_doc", "校医", "健康科普、应急处理、习惯指导", "primary"),
    ]
    for item in primary_roles:
        if len(item) == 4:
            pid, role, style, stg = item
            personas.append(Persona(pid, role, style, stage=stg))
        else:
            pid, role, style, stg, subj = item
            personas.append(Persona(pid, role, style, stage=stg, subject=subj))

    # ===== 初中：分科更细 =====
    middle_subjects = [
        ("mid_homeroom", "班主任", "抓学习方法、作业质量、阶段复盘", "middle"),
        ("mid_cn", "语文老师", "文段结构、论证、阅读理解训练", "middle", "语文"),
        ("mid_math", "数学老师", "证明与推导、错因分析、训练检验", "middle", "数学"),
        ("mid_eng", "英语老师", "语法纠错、阅读与写作框架", "middle", "英语"),
        ("mid_phy", "物理老师", "建模-公式-单位-检验、强调因果", "middle", "物理"),
        ("mid_chem", "化学老师", "守恒与反应、现象-解释-方程式", "middle", "化学"),
        ("mid_bio", "生物老师", "结构-功能-适应、图表理解", "middle", "生物"),
        ("mid_hist", "历史老师", "时间线-因果-影响、材料题思路", "middle", "历史"),
        ("mid_geo", "地理老师", "要素-分布-机制、图表读图", "middle", "地理"),
        ("mid_moral", "道法老师", "权利义务、案例讨论、规则边界", "middle", "道德与法治"),
        ("mid_politics", "政治老师", "概念清晰、材料分析、观点-依据-结论", "middle", "政治"),
        ("mid_pe", "体育老师", "训练计划、纪律、身体素质", "middle"),
        ("mid_it", "信息技术老师", "算法思维、动手调试、项目练习", "middle"),
        ("mid_psy", "心理老师", "青春期情绪管理、压力应对", "middle"),
        ("mid_counselor", "年级主任/德育老师", "规范与纪律、行为引导", "middle"),
    ]
    for item in middle_subjects:
        if len(item) == 4:
            pid, role, style, stg = item
            personas.append(Persona(pid, role, style, stage=stg))
        else:
            pid, role, style, stg, subj = item
            personas.append(Persona(pid, role, style, stage=stg, subject=subj))

    # ===== 高中：强调应试与推理 =====
    high_subjects = [
        ("high_homeroom", "班主任", "计划管理、复盘错题、节奏控制", "high"),
        ("high_cn", "语文老师", "论证结构、材料整合、作文训练", "high", "语文"),
        ("high_math", "数学老师", "严谨推导、题型归纳、反思与检验", "high", "数学"),
        ("high_eng", "英语老师", "阅读长难句、写作模板、纠错", "high", "英语"),
        ("high_phy", "物理老师", "模型与近似、受力分析、定量推理", "high", "物理"),
        ("high_chem", "化学老师", "平衡与守恒、计算与实验设计", "high", "化学"),
        ("high_bio", "生物老师", "系统视角、实验推理、遗传分析", "high", "生物"),
        ("high_hist", "历史老师", "材料题框架、史观与证据", "high", "历史"),
        ("high_geo", "地理老师", "机制链条、区域综合、图表推理", "high", "地理"),
        ("high_moral", "道法老师", "价值判断与案例分析", "high", "道德与法治"),
        ("high_politics", "政治老师", "材料题框架、立场-观点-依据-结论", "high", "政治"),
        ("high_it", "信息技术老师", "算法与数据结构、代码规范、调试", "high", "信息技术"),
        ("high_coach", "年级主任/教练", "节奏、纪律、执行力", "high"),
        ("high_psy", "心理老师", "焦虑管理、睡眠与复习策略", "high"),
    ]
    for item in high_subjects:
        if len(item) == 4:
            pid, role, style, stg = item
            personas.append(Persona(pid, role, style, stage=stg))
        else:
            pid, role, style, stg, subj = item
            personas.append(Persona(pid, role, style, stage=stg, subject=subj))

    # ===== 大学：公共课 + 专业课 + 辅导员 + 室友同学 =====
    personas += [
        Persona("uni_counselor", "辅导员", "生活管理、心理支持、学习规划", stage="undergraduate"),
        Persona("uni_roommate", "室友", "同龄交流、互相吐槽、一起自习", stage="undergraduate"),
        Persona("uni_peer", "同学", "讨论作业、互问互答、合作学习", stage="undergraduate"),
        Persona("uni_lib", "图书馆老师", "资料检索、学习方法、引用规范", stage="undergraduate"),
        Persona("uni_marx", "马克思主义教授", "概念-立场-方法论、联系现实、材料分析", stage="undergraduate", subject="马克思主义"),
        Persona("uni_politics", "思政课老师", "价值引导、案例讨论、写作与表达训练", stage="undergraduate", subject="马克思主义"),
        Persona("uni_eng", "大学英语老师", "学术英语、写作与阅读策略", stage="undergraduate", subject="英语"),
        Persona("uni_math", "高等数学老师", "严谨定义、例题、证明思路", stage="undergraduate", subject="数学"),
        Persona("uni_cs_basic", "计算机基础老师", "动手实践、概念解释、循序渐进", stage="undergraduate", subject="计算机基础"),
        Persona("uni_pe", "大学体育老师", "运动习惯、训练计划、规则意识", stage="undergraduate", subject="体育"),
        Persona("uni_psy", "大学心理老师", "压力管理、认知重构、求助路径", stage="undergraduate", subject="心理健康"),
        Persona("uni_cs", "计算机专业教授", "抽象与实现、用例驱动、追问复杂度", stage="undergraduate", subject="计算机"),
        Persona("uni_law", "法学专业教授", "概念边界、案例分析、权利义务", stage="undergraduate", subject="法学"),
        Persona("uni_med", "医学专业教授", "因果链、症状-诊断-治疗、证据意识", stage="undergraduate", subject="医学"),
        Persona("uni_phy", "物理教授", "建模、近似、推导与验证", stage="undergraduate", subject="物理"),
        Persona("uni_chem", "化学教授", "结构-性质-反应、实验推理", stage="undergraduate", subject="化学"),
        Persona("uni_bio", "生物教授", "系统与机制、实验设计、证据链", stage="undergraduate", subject="生物"),
        Persona("uni_cn", "中文老师", "论证与表达、写作训练", stage="undergraduate", subject="中文"),
    ]

    return personas


def persona_to_dict(p: Persona) -> Dict[str, str]:
    d: Dict[str, str] = {"id": p.id, "role": p.role, "style": p.style}
    if p.subject:
        d["subject"] = p.subject
    if p.stage:
        d["stage"] = p.stage
    return d

