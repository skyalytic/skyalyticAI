"""
类人成长训练器 — 0~3 岁感知运动 + 上学多科 + 升学考试。
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional

import numpy as np

from skyalyticAI.brain import NIEABrain
from skyalyticAI.env.curriculum_world import HumanGrowthWorld
from skyalyticAI.training.acceptance_report import AcceptanceReportBuilder
from skyalyticAI.training.text_metrics import cer, wer
from skyalyticAI.training.trainer import NIEATrainer


class HumanGrowthTrainer(NIEATrainer):
    def __init__(
        self,
        brain: NIEABrain,
        env: HumanGrowthWorld,
        speech_window: int = 200,
        forgetting_threshold: float = 0.05,
        **kwargs: Any,
    ) -> None:
        super().__init__(brain=brain, env=env, **kwargs)
        self.human_env: HumanGrowthWorld = env
        self._speech_window = speech_window
        self._speech_history: Deque[float] = deque(maxlen=speech_window)
        # 近期各科目准确率（防止偏科）
        self._subject_history: Dict[str, Deque[float]] = {}
        self._exams_passed: int = 0
        self._promotions: int = 0
        self._report_builder = AcceptanceReportBuilder(
            forgetting_threshold=forgetting_threshold
        )
        self._last_report: Optional[Dict[str, Any]] = None
        self._retention_eval_interval = 50
        self._asr_cer_hist: Deque[float] = deque(maxlen=speech_window)
        self._asr_wer_hist: Deque[float] = deque(maxlen=speech_window)
        self._ocr_cer_hist: Deque[float] = deque(maxlen=speech_window)
        self._ocr_wer_hist: Deque[float] = deque(maxlen=speech_window)

    def _rolling_speech_accuracy(self) -> float:
        if not self._speech_history:
            return 0.0
        return float(np.mean(self._speech_history))

    def _rolling_subject_accuracy(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for subj, dq in self._subject_history.items():
            if dq:
                out[subj] = float(np.mean(dq))
        return out

    def _run_episode(self, episode: int) -> Dict[str, Any]:
        self.human_env.set_rolling_speech_accuracy(self._rolling_speech_accuracy())
        self.human_env.set_rolling_subject_accuracy(self._rolling_subject_accuracy())
        max_steps = self.human_env.get_steps_per_episode()

        obs = self.env.reset()
        self.brain.reset_episode()
        self.brain.set_school_stage(self.human_env.school_stage)

        total_reward = 0.0
        total_surprise = 0.0
        steps = 0
        prev_env_reward = 0.0
        speech_correct = 0
        speech_n = 0
        old_stage = self.human_env.school_stage

        for step in range(max_steps):
            hidden, prediction, prediction_error = self._perceive_observation(obs)

            activity = getattr(self.human_env, "_activity", None)
            activity_val = activity.value if activity is not None else "motor"
            prefer_speech = activity_val in ("reading", "exam")

            thought = self.brain.think(
                hidden,
                external_reward=prev_env_reward,
                prefer_speech=prefer_speech,
            )
            action = thought["action"]

            next_obs, env_reward, done, info = self.env.step(action)

            if info.get("mode") in ("text", "exam") or info.get("activity") in (
                "reading",
                "exam",
            ):
                target_ch = info.get("target_char", "")
                if target_ch:
                    tid = self.human_env.corpus.char_to_index(target_ch)
                    self.brain.learn_speech(hidden, tid, env_reward)
                    speech_n += 1
                    if info.get("correct"):
                        speech_correct += 1
                        self._speech_history.append(1.0)
                    else:
                        self._speech_history.append(0.0)
                    subj = info.get("subject")
                    if subj:
                        if subj not in self._subject_history:
                            self._subject_history[subj] = deque(
                                maxlen=self._speech_window
                            )
                        self._subject_history[subj].append(
                            1.0 if info.get("correct") else 0.0
                        )

            # ASR/OCR 任务头联合训练与指标统计（多模态世界会提供 target_text）
            if info.get("mode") == "society":
                target_text = str(info.get("target_text", "") or "")
                if target_text:
                    # 使用当前目标字符做逐字符监督
                    tid = self.human_env.corpus.char_to_index(info.get("target_char", ""))
                    rew = 1.0 if info.get("correct") else -1.0
                    self.brain.learn_asr(hidden, tid, rew)
                    self.brain.learn_ocr(hidden, tid, rew)

                    # 以单字符预测构造轻量 CER/WER 统计
                    pred_asr = self.human_env.corpus.index_to_char(self.brain.asr_decode(hidden))
                    pred_ocr = self.human_env.corpus.index_to_char(self.brain.ocr_decode(hidden))
                    self._asr_cer_hist.append(cer(pred_asr, info.get("target_char", "")))
                    self._ocr_cer_hist.append(cer(pred_ocr, info.get("target_char", "")))
                    self._asr_wer_hist.append(wer(pred_asr, info.get("target_char", "")))
                    self._ocr_wer_hist.append(wer(pred_ocr, info.get("target_char", "")))

            intrinsic_reward = 0.0
            if self.reward_shaping:
                surprise = float(np.linalg.norm(prediction_error))
                intrinsic_reward = self.curiosity_weight * surprise
                total_surprise += surprise

            next_hidden, _, _ = self._perceive_observation(next_obs)
            self.brain.learn(
                hidden,
                action,
                next_hidden,
                env_reward + intrinsic_reward,
                prediction_error,
                env_reward=env_reward,
            )
            self.brain.develop()
            self.brain.set_school_stage(self.human_env.school_stage)

            if done and info.get("passed"):
                self._exams_passed += 1
            if self.human_env.school_stage != old_stage:
                new_stage = self.human_env.school_stage
                retention = self._report_builder.evaluate_retention(self, old_stage)
                if self._report_builder.should_rollback_promotion(retention):
                    # 遗忘超阈值：回滚升级
                    self.human_env.set_stage(old_stage)
                    self.brain.set_school_stage(old_stage)
                else:
                    self._promotions += 1
                    old_stage = new_stage
                    self.human_env._episodes_since_exam = 0

            total_reward += env_reward
            prev_env_reward = env_reward
            steps += 1
            self.total_steps += 1
            obs = next_obs
            if done:
                break

        if (episode + 1) % self._retention_eval_interval == 0:
            self._report_builder.evaluate_retention(self, self.human_env.school_stage)

        return {
            "total_reward": total_reward,
            "steps": steps,
            "avg_surprise": total_surprise / max(steps, 1),
            "speech_accuracy": speech_correct / max(speech_n, 1),
            "school_stage": self.human_env.school_stage,
        }

    def _log_progress(self, episode: int) -> None:
        window = min(self.early_stop_window, len(self.episode_rewards))
        avg_reward = np.mean(self.episode_rewards[-window:])
        brain_summary = self.brain.get_state_summary()
        subj = self.human_env._current_subject or "-"
        print(
            "Episode {}/{} | 奖励: {:.2f} | 阶段: {} ({}) | 科目: {} | "
            "知识: {} | 说话: {:.1%} | 升学: {}".format(
                episode + 1,
                self.max_episodes,
                avg_reward,
                self.human_env.school_stage,
                self.human_env.corpus.stage_display_name(self.human_env.school_stage),
                subj,
                brain_summary["knowledge_vectors"],
                self._rolling_speech_accuracy(),
                self._promotions,
            )
        )

    def train(self) -> Dict[str, Any]:
        summary = super().train()
        self._last_report = self._report_builder.build_report(self, summary)
        return summary

    def multimodal_metrics(self) -> Dict[str, float]:
        """返回 ASR/OCR 指标（滚动窗口）。"""
        return {
            "asr_cer": float(np.mean(self._asr_cer_hist)) if self._asr_cer_hist else 1.0,
            "asr_wer": float(np.mean(self._asr_wer_hist)) if self._asr_wer_hist else 1.0,
            "ocr_cer": float(np.mean(self._ocr_cer_hist)) if self._ocr_cer_hist else 1.0,
            "ocr_wer": float(np.mean(self._ocr_wer_hist)) if self._ocr_wer_hist else 1.0,
        }

    def generate_acceptance_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        summary = {
            "total_episodes": len(self.episode_rewards),
            "total_steps": self.total_steps,
            "final_avg_reward": float(np.mean(self.episode_rewards[-50:])) if self.episode_rewards else 0.0,
            "brain_stage": self.human_env.school_stage,
        }
        report = self._report_builder.build_report(self, summary)
        self._last_report = report
        if output_path:
            self._report_builder.save_report(report, output_path)
        return report
