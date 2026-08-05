#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全科知识语料下载器（小学到本科）。

数据来源：
- 维基百科中文 REST API（CC BY-SA 3.0）
- 维基文库（公版文本）
- 古腾堡计划（公有领域）

合法合规：仅下载开放授权内容，不抓取受版权保护的教材全文。
输出目录：data/corpus/<学段>/<科目>/<条目>.txt
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = ROOT / "data" / "corpus"

# User-Agent 必须包含联系方式（维基百科政策）
UA = "SkyalyticAI-Corpus-Downloader/1.0 (educational research; contact: dev@skyalytic.ai)"

# 学段目录映射（与 education_config.STAGE_DIR_MAP 一致）
STAGE_DIRS = {
    "kindergarten": "01_kindergarten",
    "primary": "02_primary",
    "middle": "03_middle",
    "high": "04_high",
    "undergraduate": "05_undergraduate",
}

# 每科每学段的知识点维基百科条目清单
# 格式: (学段, 科目) -> [条目列表]
KNOWLEDGE_POINTS: Dict[Tuple[str, str], List[str]] = {
    # ============= 小学 =============
    ("primary", "语文"): [
        "汉字", "汉语拼音", "普通话", "成语", "寓言", "唐诗", "宋词",
        "三字经", "百家姓", "千字文", "论语", "西游记", "三国演义",
    ],
    ("primary", "数学"): [
        "加法", "减法", "乘法", "除法", "分数", "小数", "百分数",
        "几何学", "三角形", "正方形", "圆形", "面积", "体积",
        "方程", "代数", "数轴",
    ],
    ("primary", "英语"): [
        "英文字母", "英语发音", "英语语法", "英语词汇",
        "英语时态", "英语会话",
    ],
    ("primary", "科学"): [
        "自然", "动物", "植物", "昆虫", "鸟类", "哺乳动物",
        "地球", "太阳系", "宇宙", "天气", "季节",
        "物质", "能量", "力", "光", "声音", "电",
    ],
    ("primary", "道德与法治"): [
        "道德", "品德", "家庭", "学校", "社会",
        "公民", "权利", "义务", "法律", "规则",
    ],

    # ============= 初中 =============
    ("middle", "语文"): [
        "文言文", "现代文学", "鲁迅", "朱自清", "老舍",
        "修辞", "比喻", "拟人", "夸张", "排比",
        "记叙文", "说明文", "议论文", "应用文",
        "诗经", "楚辞", "汉赋", "唐诗", "宋词", "元曲",
        "明清小说", "红楼梦", "水浒传",
    ],
    ("middle", "数学"): [
        "代数", "方程", "一元一次方程", "二元一次方程",
        "不等式", "函数", "一次函数", "二次函数",
        "几何学", "平面几何", "三角形", "四边形", "圆",
        "相似", "全等", "勾股定理", "三角函数",
        "统计", "概率",
    ],
    ("middle", "英语"): [
        "英语语法", "英语时态", "英语语态",
        "英语从句", "英语词汇", "英语写作",
    ],
    ("middle", "物理"): [
        "力学", "运动学", "牛顿运动定律", "力", "质量",
        "速度", "加速度", "重力", "摩擦力", "浮力",
        "压强", "功", "能量", "动能", "势能",
        "热学", "温度", "热量",
        "电学", "电流", "电压", "电阻", "欧姆定律",
        "磁学", "电磁感应",
        "光学", "光的反射", "光的折射",
        "声学", "声音",
    ],
    ("middle", "化学"): [
        "化学", "化学元素", "原子", "分子", "离子",
        "化学键", "共价键", "离子键",
        "化学反应", "氧化还原反应", "酸碱反应",
        "化学式", "化学方程式",
        "氧气", "氢气", "二氧化碳", "水",
        "酸", "碱", "盐",
        "有机化学", "无机化学",
    ],
    ("middle", "生物"): [
        "生物学", "细胞", "细胞膜", "细胞核", "细胞质",
        "组织", "器官", "系统",
        "光合作用", "呼吸作用",
        "遗传", "DNA", "基因", "变异",
        "进化", "达尔文",
        "生态学", "生态系统", "食物链",
        "植物", "动物", "微生物",
        "人体", "消化系统", "循环系统", "神经系统",
    ],
    ("middle", "历史"): [
        "中国历史", "中国古代史", "夏朝", "商朝", "周朝",
        "秦朝", "汉朝", "唐朝", "宋朝", "元朝", "明朝", "清朝",
        "中国近代史", "鸦片战争", "辛亥革命",
        "世界历史", "古埃及", "古希腊", "古罗马",
        "文艺复兴", "工业革命", "第一次世界大战", "第二次世界大战",
    ],
    ("middle", "地理"): [
        "地理学", "自然地理", "人文地理",
        "地球", "经纬度", "时区",
        "地形", "山地", "平原", "高原", "盆地", "丘陵",
        "气候", "季风", "降水",
        "河流", "长江", "黄河", "湖泊", "海洋",
        "中国地理", "中国行政区划",
        "世界地理", "亚洲", "欧洲", "非洲", "美洲", "大洋洲",
    ],
    ("middle", "道德与法治"): [
        "政治", "国家", "政府", "宪法",
        "法律", "刑法", "民法", "行政法",
        "公民", "权利", "义务",
        "社会", "家庭", "学校",
        "道德", "伦理", "价值观",
    ],

    # ============= 高中 =============
    ("high", "语文"): [
        "中国古代文学", "中国现代文学", "中国当代文学",
        "外国文学", "莎士比亚", "托尔斯泰",
        "诗歌", "小说", "散文", "戏剧",
        "文学理论", "文学批评",
        "修辞学", "古汉语",
        "文言文阅读", "现代文阅读", "写作",
    ],
    ("high", "数学"): [
        "集合", "逻辑", "函数", "指数函数", "对数函数", "幂函数",
        "三角函数", "正弦", "余弦", "正切",
        "数列", "等差数列", "等比数列",
        "不等式",
        "立体几何", "空间向量",
        "解析几何", "直线", "圆锥曲线", "椭圆", "双曲线", "抛物线",
        "导数", "微积分",
        "概率", "统计", "随机变量",
        "复数",
    ],
    ("high", "英语"): [
        "英语语法", "英语词汇", "英语阅读",
        "英语写作", "英语听力", "英语口语",
        "英语翻译",
    ],
    ("high", "物理"): [
        "力学", "运动学", "动力学",
        "牛顿运动定律", "万有引力定律",
        "动量", "角动量",
        "功", "能", "机械能守恒",
        "简谐运动", "波", "声波", "电磁波",
        "热学", "热力学", "理想气体",
        "电磁学", "电场", "磁场", "电磁感应",
        "交流电", "电磁波",
        "光学", "几何光学", "物理光学",
        "干涉", "衍射", "偏振",
        "原子物理", "量子力学", "相对论",
    ],
    ("high", "化学"): [
        "化学", "无机化学", "有机化学",
        "物质结构", "原子结构", "元素周期表",
        "化学键", "分子结构",
        "化学反应原理", "化学平衡", "电离平衡",
        "电化学", "原电池", "电解池",
        "氧化还原反应",
        "元素化学", "碱金属", "卤素", "氧族元素", "氮族元素", "碳族元素",
        "有机化合物", "烃", "醇", "醛", "酸", "酯",
        "高分子化合物",
    ],
    ("high", "生物"): [
        "分子生物学", "蛋白质", "核酸", "DNA", "RNA",
        "细胞生物学", "细胞分裂", "有丝分裂", "减数分裂",
        "遗传学", "孟德尔遗传定律", "基因",
        "生物进化", "现代生物进化理论",
        "生态学", "种群", "群落", "生态系统",
        "人体生理", "神经调节", "体液调节", "免疫",
        "植物生理", "光合作用", "植物激素",
        "生物技术", "基因工程", "细胞工程",
    ],
    ("high", "历史"): [
        "中国近代史", "中国现代史",
        "鸦片战争", "太平天国", "洋务运动", "戊戌变法",
        "辛亥革命", "五四运动", "抗日战争", "解放战争",
        "中华人民共和国",
        "世界近代史", "世界现代史",
        "文艺复兴", "宗教改革", "启蒙运动",
        "英国革命", "美国独立战争", "法国大革命",
        "工业革命", "俄国十月革命",
        "第一次世界大战", "第二次世界大战", "冷战",
    ],
    ("high", "地理"): [
        "自然地理", "人文地理", "经济地理",
        "地球运动", "地球公转", "地球自转",
        "大气", "天气", "气候",
        "水文", "洋流",
        "地貌", "外力作用", "内力作用",
        "自然资源", "能源",
        "人口", "城市", "工业", "农业",
        "区域地理", "中国地理", "世界地理",
        "可持续发展",
    ],
    ("high", "道德与法治"): [
        "政治学", "政治制度", "民主", "共和",
        "经济学", "市场经济", "宏观调控",
        "哲学", "马克思主义哲学", "辩证唯物主义", "历史唯物主义",
        "伦理学", "道德哲学",
        "法学", "宪法学", "法理学",
        "社会学",
    ],
    ("high", "信息技术"): [
        "计算机", "计算机科学", "计算机硬件", "计算机软件",
        "操作系统", "Windows", "Linux",
        "办公软件", "文字处理", "电子表格",
        "编程语言", "Python", "C语言", "Java",
        "数据结构", "算法",
        "数据库", "SQL",
        "网络", "互联网", "TCP/IP", "HTTP",
        "网页设计", "HTML", "CSS", "JavaScript",
        "多媒体", "图像处理",
        "人工智能",
    ],

    # ============= 本科公共课 =============
    ("undergraduate", "马克思主义"): [
        "马克思主义", "卡尔·马克思", "弗里德里希·恩格斯",
        "辩证唯物主义", "历史唯物主义",
        "政治经济学", "资本论", "剩余价值",
        "科学社会主义",
        "列宁主义", "毛泽东思想", "中国特色社会主义",
    ],
    ("undergraduate", "大学英语"): [
        "英语", "英语语法", "英语写作",
        "英语翻译", "口译", "笔译",
        "英美文学", "英国文学", "美国文学",
        "英语国家概况",
        "学术论文写作",
    ],
    ("undergraduate", "高等数学"): [
        "微积分", "极限", "连续",
        "导数", "微分", "中值定理",
        "不定积分", "定积分", "广义积分",
        "多元微积分", "偏导数", "多重积分",
        "级数", "幂级数", "泰勒级数", "傅里叶级数",
        "常微分方程",
        "线性代数", "矩阵", "行列式", "线性方程组", "向量空间", "特征值",
        "概率论", "随机变量", "概率分布", "大数定律", "中心极限定理",
        "数理统计", "假设检验",
    ],
    ("undergraduate", "计算机基础"): [
        "计算机科学", "计算理论", "计算复杂性",
        "数据结构", "数组", "链表", "栈", "队列", "树", "图", "哈希表",
        "算法", "排序算法", "搜索算法", "动态规划", "贪心算法", "分治法",
        "操作系统", "进程", "线程", "内存管理", "文件系统",
        "计算机网络", "网络协议", "TCP/IP", "HTTP", "DNS",
        "数据库", "关系数据库", "SQL", "事务", "索引",
        "编程范式", "面向对象编程", "函数式编程",
        "软件工程", "设计模式",
        "人工智能", "机器学习", "深度学习", "神经网络",
    ],
    ("undergraduate", "体育"): [
        "体育运动", "体育教育",
        "田径", "篮球", "足球", "排球", "乒乓球", "羽毛球", "网球",
        "游泳", "体操", "武术",
        "运动生理学", "运动训练",
    ],
    ("undergraduate", "心理健康"): [
        "心理学", "普通心理学", "认知心理学", "发展心理学",
        "社会心理学", "人格心理学",
        "心理健康", "心理咨询", "心理治疗",
        "情绪", "压力", "焦虑", "抑郁",
        "积极心理学",
    ],

    # ============= 本科专业（精选核心条目）=============
    ("undergraduate", "哲学"): ["哲学", "形而上学", "认识论", "伦理学", "逻辑学", "美学", "中国哲学", "西方哲学"],
    ("undergraduate", "经济学"): ["经济学", "微观经济学", "宏观经济学", "计量经济学", "行为经济学", "发展经济学"],
    ("undergraduate", "法学"): ["法学", "法理学", "宪法学", "民法学", "刑法学", "行政法学", "国际法"],
    ("undergraduate", "教育学"): ["教育学", "教育心理学", "课程论", "教学论", "教育史", "比较教育学"],
    ("undergraduate", "文学"): ["文学", "文学理论", "文学批评", "比较文学", "中国古代文学", "中国现当代文学"],
    ("undergraduate", "历史学"): ["历史学", "史学理论", "中国史", "世界史", "考古学"],
    ("undergraduate", "理学"): ["理学", "数学", "物理学", "化学", "生物学", "天文学", "地球科学"],
    ("undergraduate", "工学"): ["工学", "工程学", "机械工程", "电气工程", "化学工程", "土木工程"],
    ("undergraduate", "农学"): ["农学", "园艺学", "植物保护", "畜牧学", "兽医学", "林学"],
    ("undergraduate", "医学"): ["医学", "基础医学", "临床医学", "预防医学", "中医学", "药学"],
    ("undergraduate", "管理学"): ["管理学", "工商管理", "公共管理", "会计学", "市场营销", "人力资源管理"],
    ("undergraduate", "艺术学"): ["艺术", "美术", "音乐", "戏剧", "电影", "设计", "艺术史"],
    ("undergraduate", "计算机"): [
        "计算机科学", "编程语言理论", "计算机组成", "操作系统",
        "计算机网络", "数据库", "编译原理",
        "人工智能", "机器学习", "深度学习", "自然语言处理", "计算机视觉",
        "分布式系统", "云计算", "信息安全", "密码学",
    ],
    ("undergraduate", "数学"): [
        "数学", "数学分析", "高等代数", "解析几何",
        "抽象代数", "群论", "环论", "域论",
        "拓扑学", "泛函分析", "微分几何",
        "数论", "组合数学",
        "概率论", "数理统计", "随机过程",
        "数值分析", "最优化",
    ],
    ("undergraduate", "物理"): [
        "物理学", "经典力学", "分析力学", "拉格朗日力学", "哈密顿力学",
        "电动力学", "电磁学", "麦克斯韦方程组",
        "热力学", "统计力学",
        "量子力学", "薛定谔方程",
        "光学", "原子物理", "核物理", "粒子物理",
        "凝聚态物理", "固体物理",
        "天体物理", "宇宙学",
        "相对论", "狭义相对论", "广义相对论",
    ],
    ("undergraduate", "化学"): [
        "化学", "无机化学", "有机化学", "分析化学", "物理化学",
        "结构化学", "量子化学",
        "高分子化学", "材料化学",
        "生物化学", "化学生物学",
        "热力学", "化学动力学", "电化学", "表面化学",
    ],
    ("undergraduate", "生物"): [
        "生物学", "分子生物学", "细胞生物学", "遗传学", "生物化学",
        "微生物学", "病毒学",
        "植物学", "动物学", "生理学", "神经科学",
        "生态学", "生物多样性",
        "进化生物学", "发育生物学",
        "免疫学", "生物技术", "基因工程",
    ],
    ("undergraduate", "中文"): [
        "中国语言文学", "现代汉语", "古代汉语", "语言学",
        "汉字学", "音韵学", "训诂学",
        "中国古代文学", "中国现代文学", "中国当代文学",
        "文学理论", "文学批评",
    ],
    ("undergraduate", "英语"): [
        "英语语言文学", "英语语言学", "英语词汇学", "英语语法学",
        "英美文学", "英国文学史", "美国文学史",
        "翻译学", "翻译理论", "口译", "笔译",
        "跨文化交际",
    ],
    ("undergraduate", "日语"): [
        "日语", "日语语法", "日语词汇",
        "日本文学", "日本文化",
        "翻译",
    ],
    ("undergraduate", "自动化"): [
        "自动化", "控制理论", "自动控制原理",
        "经典控制理论", "现代控制理论",
        "反馈", "PID控制器",
        "机器人学", "机器人",
        "信号与系统", "数字信号处理",
    ],
    ("undergraduate", "电子信息"): [
        "电子工程", "信息工程",
        "电路", "模拟电路", "数字电路",
        "信号处理", "数字信号处理", "图像处理",
        "通信原理", "通信工程",
        "电磁场与电磁波", "微波工程",
        "天线", "射频工程",
    ],
    ("undergraduate", "机械"): [
        "机械工程", "机械设计", "机械原理",
        "材料力学", "理论力学", "流体力学",
        "热力学", "传热学",
        "制造工程", "机械制造", "数控机床",
        "机器人学", "机电一体化",
    ],
    ("undergraduate", "土木"): [
        "土木工程", "结构工程", "结构力学",
        "建筑材料", "混凝土", "钢结构",
        "土力学", "岩土工程",
        "水利工程", "桥梁工程", "隧道工程",
        "建筑工程", "城市规划",
    ],
    ("undergraduate", "临床医学"): [
        "临床医学", "内科学", "外科学", "妇产科学", "儿科学",
        "诊断学", "病理学", "病理生理学",
        "药理学", "药物治疗学",
        "影像学", "医学影像",
        "神经病学", "精神病学",
        "传染病学", "流行病学",
    ],
}


def fetch_wikipedia_extract(title: str, timeout: int = 30) -> Optional[str]:
    """通过维基百科 REST API 获取条目纯文本摘要。

    使用 action=query&prop=extracts&explaintext 接口，返回纯文本而非 HTML。
    """
    api = "https://zh.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "exsectionformat": "plain",
        "redirects": "1",
        "format": "json",
        "titles": title,
    }
    url = api + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="ignore"))

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None
    page = next(iter(pages.values()))
    # 处理重定向
    if "redirects" in data.get("query", {}):
        for r in data["query"]["redirects"]:
            if r.get("from") == title:
                title = r.get("to", title)
                return fetch_wikipedia_extract(title)
    extract = page.get("extract")
    if not extract or extract.strip() == "":
        return None
    return extract


def fetch_wikisource_text(title: str, timeout: int = 30) -> Optional[str]:
    """从维基文库获取公版文本。"""
    api = "https://zh.wikisource.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "format": "json",
        "titles": title,
    }
    url = api + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return None
        page = next(iter(pages.values()))
        extract = page.get("extract")
        if not extract:
            return None
        return extract
    except Exception:
        return None


def sanitize_filename(name: str) -> str:
    """清理文件名，去除非法字符。"""
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    return name.strip()[:80]


def save_stage_subject(stage: str, subject: str, title: str, content: str, max_chars: int = 80000) -> Path:
    """保存到 data/corpus/<学段目录>/<科目>/<标题>.txt"""
    stage_dir = STAGE_DIRS.get(stage)
    if not stage_dir:
        raise ValueError(f"未知学段: {stage}")
    folder = CORPUS_ROOT / stage_dir / subject
    folder.mkdir(parents=True, exist_ok=True)
    fname = sanitize_filename(title) + ".txt"
    out = folder / fname
    # 加上标题行
    text = f"# {title}\n\n{content[:max_chars]}"
    out.write_text(text, encoding="utf-8")
    return out


def download_all(sleep_sec: float = 1.0, max_retries: int = 2) -> Tuple[int, int, int]:
    """下载全部知识点。

    返回 (success_count, fail_count, total_chars)
    """
    success = 0
    fail = 0
    total_chars = 0
    total_items = sum(len(v) for v in KNOWLEDGE_POINTS.values())
    idx = 0

    for (stage, subject), titles in KNOWLEDGE_POINTS.items():
        stage_dir = STAGE_DIRS.get(stage, stage)
        print(f"\n=== [{stage_dir}/{subject}] 共 {len(titles)} 个条目 ===", flush=True)

        for title in titles:
            idx += 1
            # 已存在则跳过
            out_path = CORPUS_ROOT / stage_dir / subject / (sanitize_filename(title) + ".txt")
            if out_path.exists() and out_path.stat().st_size > 200:
                success += 1
                total_chars += out_path.stat().st_size
                print(f"  [{idx}/{total_items}] SKIP (exists): {title}", flush=True)
                continue

            print(f"  [{idx}/{total_items}] {title} ...", end=" ", flush=True)

            content = None
            for attempt in range(max_retries + 1):
                try:
                    content = fetch_wikipedia_extract(title)
                    if content:
                        break
                except Exception as e:
                    if attempt < max_retries:
                        time.sleep(2.0)
                    else:
                        print(f"FAIL ({e})", flush=True)

            if content:
                try:
                    saved = save_stage_subject(stage, subject, title, content)
                    success += 1
                    total_chars += len(content)
                    print(f"OK ({len(content)} chars)", flush=True)
                except Exception as e:
                    fail += 1
                    print(f"SAVE-FAIL ({e})", flush=True)
            else:
                fail += 1
                print("NO-CONTENT", flush=True)

            # 限速，避免被维基百科封禁
            time.sleep(sleep_sec)

    return success, fail, total_chars


def main() -> None:
    print("=" * 60, flush=True)
    print("全科知识语料下载器", flush=True)
    print(f"来源: 维基百科中文 (CC BY-SA 3.0)", flush=True)
    print(f"输出: {CORPUS_ROOT}", flush=True)
    total = sum(len(v) for v in KNOWLEDGE_POINTS.values())
    stages = len(KNOWLEDGE_POINTS)
    print(f"范围: {stages} 个(学段×科目)组合, 共 {total} 个条目", flush=True)
    print("=" * 60, flush=True)

    t0 = time.time()
    success, fail, total_chars = download_all(sleep_sec=1.0, max_retries=2)
    elapsed = time.time() - t0

    print("\n" + "=" * 60, flush=True)
    print(f"下载完成", flush=True)
    print(f"  成功: {success} 个条目", flush=True)
    print(f"  失败: {fail} 个条目", flush=True)
    print(f"  总字符数: {total_chars:,}", flush=True)
    print(f"  总用时: {elapsed/60:.1f} 分钟", flush=True)
    print(f"  输出目录: {CORPUS_ROOT}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
