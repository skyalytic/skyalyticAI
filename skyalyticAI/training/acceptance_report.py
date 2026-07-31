"""
工业级训练验收与遗忘测试。

功能：
1) 训练验收报告（数据覆盖、阶段进展、准确率）
2) 遗忘测试（旧学段回测）
3) 可供训练器调用的升学保护判定
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import numpy as np

from skyalyticAI.data.education_config import STAGE_ORDER, get_quality_spec
from skyalyticAI.exams.exam_suite import ExamSuite


@dataclass
class RetentionResult:
    stage: str
    accuracy: float
    historical_best: float
    forgetting: float
    hdc_retrieval_accuracy: float = 0.0


class AcceptanceReportBuilder:
    """构建训练验收报告与遗忘评估。"""

    def __init__(self, forgetting_threshold: float = 0.05) -> None:
        self.forgetting_threshold = forgetting_threshold
        self.best_stage_accuracy: Dict[str, float] = {}
        self.latest_stage_accuracy: Dict[str, float] = {}
        self.retention_history: List[Dict[str, Any]] = []

    def evaluate_stage_exam(
        self,
        trainer: Any,
        stage: str,
        n_questions: Optional[int] = None,
    ) -> tuple:
        """评估指定学段考试准确率与 HDC 检索准确率。

        返回 (exam_accuracy, hdc_retrieval_accuracy)。
        """
        env = trainer.human_env
        spec = get_quality_spec(stage)
        eval_env = ExamSuite(
            corpus=env.corpus,
            stage=stage,
            observation_dim=env.get_observation_dim(),
            seed=1234,
        )
        obs = eval_env.reset()
        total = 0
        correct = 0
        hdc_total = 0
        hdc_correct = 0
        # 评估会调用 perceive，从而更新 Welford 输入归一化统计量；
        # 保存以便评估后恢复，避免考试数据污染训练分布。
        norm_state = trainer.brain.save_normalization_state()
        try:
            for _ in range(max(10, spec.steps_per_episode)):
                hidden, _, _ = trainer._perceive_observation(obs)
                action = trainer.brain.speak(hidden) if trainer.brain.language_head else 0
                # HDC 检索准确率：查询记忆中最相似状态的关联动作
                hdc_action = trainer.brain.query_memory_action(hidden)
                obs, _, done, info = eval_env.step(action)
                if "correct" in info:
                    total += 1
                    if info["correct"]:
                        correct += 1
                    # HDC 检索评估：仅当记忆中有数据时计数
                    if hdc_action is not None:
                        hdc_total += 1
                        target_char = info.get("target_char", "")
                        if target_char:
                            tid = env.corpus.char_to_index(target_char)
                            if hdc_action == tid:
                                hdc_correct += 1
                if done:
                    break
        finally:
            trainer.brain.restore_normalization_state(norm_state)
        exam_acc = float(correct / max(total, 1))
        hdc_acc = float(hdc_correct / max(hdc_total, 1)) if hdc_total > 0 else 0.0
        return (exam_acc, hdc_acc)

    def evaluate_retention(self, trainer: Any, up_to_stage: str) -> List[RetentionResult]:
        results: List[RetentionResult] = []
        if up_to_stage not in STAGE_ORDER:
            return results
        idx = STAGE_ORDER.index(up_to_stage)
        for stage in STAGE_ORDER[: idx + 1]:
            if stage == "sensorimotor":
                continue
            exam_acc, hdc_acc = self.evaluate_stage_exam(trainer, stage)
            best = max(self.best_stage_accuracy.get(stage, 0.0), exam_acc)
            self.best_stage_accuracy[stage] = best
            self.latest_stage_accuracy[stage] = exam_acc
            results.append(
                RetentionResult(
                    stage=stage,
                    accuracy=exam_acc,
                    historical_best=best,
                    forgetting=max(0.0, best - exam_acc),
                    hdc_retrieval_accuracy=hdc_acc,
                )
            )
        self.retention_history.append(
            {
                "step": int(getattr(trainer, "total_steps", 0)),
                "stage": up_to_stage,
                "results": [asdict(x) for x in results],
            }
        )
        return results

    def should_rollback_promotion(self, retention: List[RetentionResult]) -> bool:
        for r in retention:
            if r.forgetting > self.forgetting_threshold:
                return True
        return False

    def build_report(self, trainer: Any, summary: Dict[str, Any]) -> Dict[str, Any]:
        env = trainer.human_env
        corpus_stats = env.corpus.corpus_stats()
        retention = self.evaluate_retention(trainer, env.school_stage)
        retention_dict = [asdict(x) for x in retention]
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "school_stage": env.school_stage,
            "school_name": env.corpus.stage_display_name(env.school_stage),
            "quality_gate": {
                "rolling_speech_accuracy": trainer._rolling_speech_accuracy(),
                "rolling_subject_accuracy": trainer._rolling_subject_accuracy(),
                "forgetting_threshold": self.forgetting_threshold,
            },
            "corpus_stats": corpus_stats,
            "retention": retention_dict,
            "retention_history": self.retention_history,
            "industrial_readiness": {
                "data_ready": (corpus_stats.get("files", 0) >= 5000) or bool(corpus_stats.get("npc_curriculum")),
                "training_ready": summary.get("total_steps", 0) >= 100000,
                "retention_pass": not self.should_rollback_promotion(retention),
            },
        }
        if hasattr(trainer, "multimodal_metrics"):
            try:
                report["multimodal_metrics"] = trainer.multimodal_metrics()
            except Exception:
                report["multimodal_metrics"] = {}
        return report

    def save_report(self, report: Dict[str, Any], path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

