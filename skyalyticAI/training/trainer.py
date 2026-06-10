"""
NIEA Trainer - Complete Training Loop

Implements the full training pipeline for the NIEABrain,
connecting it with an environment and managing the
perceive-think-learn-develop cycle.

Training flow per step:
    1. Environment provides observation
    2. Brain perceives (encode -> SNN -> PCN)
    3. Brain thinks (metacognition -> curiosity -> action selection)
    4. Environment receives action, returns reward + next observation
    5. Brain learns (STDP + PCN + world model + HDC + metacognition)
    6. Brain develops (stage progression)
    7. Log metrics

Supports:
- Episode-based training with automatic reset
- Checkpoint saving and loading
- Metric logging and progress reporting
- Early stopping based on performance
- Curriculum learning (progressive difficulty)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from skyalyticAI.brain import NIEABrain
from skyalyticAI.env.environment import Environment


class NIEATrainer:
    """
    Trainer for the NIEABrain in an environment.

    Parameters
    ----------
    brain : NIEABrain
        The brain instance to train.
    env : Environment
        The training environment.
    max_episodes : int
        Maximum number of training episodes.
    max_steps_per_episode : int
        Maximum steps per episode (overrides env max_steps).
    checkpoint_dir : str or None
        Directory for saving checkpoints. If None, no checkpoints.
    checkpoint_interval : int
        Save checkpoint every N episodes.
    log_interval : int
        Print progress every N episodes.
    early_stop_reward : float or None
        If set, stop training when average reward exceeds this value.
    early_stop_window : int
        Number of recent episodes to average for early stopping.
    reward_shaping : bool
        Whether to apply intrinsic reward shaping (curiosity bonus).
    curiosity_weight : float
        Weight for curiosity-based intrinsic reward.
    """

    def __init__(
        self,
        brain: NIEABrain,
        env: Environment,
        max_episodes: int = 1000,
        max_steps_per_episode: int = 200,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 100,
        log_interval: int = 10,
        early_stop_reward: Optional[float] = None,
        early_stop_window: int = 50,
        reward_shaping: bool = True,
        curiosity_weight: float = 0.1,
    ) -> None:
        self.brain = brain
        self.env = env
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.log_interval = log_interval
        self.early_stop_reward = early_stop_reward
        self.early_stop_window = early_stop_window
        self.reward_shaping = reward_shaping
        self.curiosity_weight = curiosity_weight

        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)

        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_surprises: List[float] = []
        self.training_log: List[Dict[str, Any]] = []
        self.total_steps: int = 0

    def train(self) -> Dict[str, Any]:
        """
        Run the complete training loop.

        Returns
        -------
        dict
            Training summary with final metrics.
        """
        start_time = time.time()

        for episode in range(self.max_episodes):
            episode_result = self._run_episode(episode)

            self.episode_rewards.append(episode_result["total_reward"])
            self.episode_lengths.append(episode_result["steps"])
            self.episode_surprises.append(episode_result["avg_surprise"])

            self.training_log.append({
                "episode": episode,
                "reward": episode_result["total_reward"],
                "steps": episode_result["steps"],
                "avg_surprise": episode_result["avg_surprise"],
                "brain_age": self.brain.age,
                "brain_stage": self.brain.stage,
                "knowledge_size": len(self.brain.hd_memory.item_memory),
            })

            if self.log_interval > 0 and (episode + 1) % self.log_interval == 0:
                self._log_progress(episode)

            if self.checkpoint_dir and (episode + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(episode)

            if self._should_stop_early():
                break

        elapsed = time.time() - start_time

        summary = {
            "total_episodes": len(self.episode_rewards),
            "total_steps": self.total_steps,
            "elapsed_seconds": elapsed,
            "final_avg_reward": np.mean(self.episode_rewards[-self.early_stop_window:]),
            "best_reward": max(self.episode_rewards) if self.episode_rewards else 0.0,
            "brain_age": self.brain.age,
            "brain_stage": self.brain.stage,
        }

        return summary

    def _run_episode(self, episode: int) -> Dict[str, Any]:
        """
        Run a single training episode.

        Parameters
        ----------
        episode : int
            Episode index.

        Returns
        -------
        dict
            Episode metrics.
        """
        obs = self.env.reset()
        self.brain.reset_episode()

        total_reward = 0.0
        total_surprise = 0.0
        steps = 0
        prev_env_reward = 0.0

        for step in range(self.max_steps_per_episode):
            hidden, prediction, prediction_error = self._perceive_observation(obs)

            thought = self.brain.think(hidden, external_reward=prev_env_reward)
            action = thought["action"]

            next_obs, env_reward, done, info = self.env.step(action)

            intrinsic_reward = 0.0
            if self.reward_shaping:
                surprise = float(np.linalg.norm(prediction_error))
                intrinsic_reward = self.curiosity_weight * surprise
                total_surprise += surprise

            total_reward_step = env_reward + intrinsic_reward

            next_hidden, _, _ = self._perceive_observation(next_obs)

            learn_result = self.brain.learn(
                hidden, action, next_hidden, total_reward_step, prediction_error,
                env_reward=env_reward,
            )
            self.brain.develop()

            total_reward += env_reward
            prev_env_reward = env_reward
            steps += 1
            self.total_steps += 1

            obs = next_obs

            if done:
                break

        avg_surprise = total_surprise / max(steps, 1)

        return {
            "total_reward": total_reward,
            "steps": steps,
            "avg_surprise": avg_surprise,
        }

    def _perceive_observation(
        self, obs: Union[np.ndarray, Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        支持多模态观测：
        - 传统：np.ndarray
        - 多模态：dict，可包含 visual/audio/raw_observation
        """
        if isinstance(obs, dict):
            visual = obs.get("visual")
            audio = obs.get("audio")
            raw = obs.get("raw_observation")
            # 允许缺项：由 brain.perceive_multimodal 内部处理
            return self.brain.perceive_multimodal(visual=visual, audio=audio, raw_observation=raw)
        return self.brain.perceive(obs)

    def _should_stop_early(self) -> bool:
        """Check if early stopping criteria are met."""
        if self.early_stop_reward is None:
            return False
        if len(self.episode_rewards) < self.early_stop_window:
            return False
        recent_avg = np.mean(self.episode_rewards[-self.early_stop_window:])
        return recent_avg >= self.early_stop_reward

    def _log_progress(self, episode: int) -> None:
        """Log training progress."""
        window = min(self.early_stop_window, len(self.episode_rewards))
        recent_rewards = self.episode_rewards[-window:]
        avg_reward = np.mean(recent_rewards)
        avg_steps = np.mean(self.episode_lengths[-window:])

        brain_summary = self.brain.get_state_summary()

        print(
            f"Episode {episode + 1}/{self.max_episodes} | "
            f"Avg Reward: {avg_reward:.2f} | "
            f"Avg Steps: {avg_steps:.1f} | "
            f"Stage: {brain_summary['stage']} | "
            f"Age: {brain_summary['age']} | "
            f"Knowledge: {brain_summary['knowledge_vectors']} | "
            f"Pred Error: {brain_summary['avg_prediction_error']:.4f}"
        )

    def _save_checkpoint(self, episode: int) -> None:
        """Save a training checkpoint."""
        if self.checkpoint_dir is None:
            return

        checkpoint = {
            "episode": episode,
            "total_steps": self.total_steps,
            "brain_state": self.brain.state_dict(),
            "episode_rewards": self.episode_rewards[-100:],
            "episode_lengths": self.episode_lengths[-100:],
        }

        path = os.path.join(
            self.checkpoint_dir, "checkpoint_ep{}.pkl".format(episode + 1)
        )

        import pickle
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _flatten_dict(
        self, d: Dict[str, Any], out: Dict[str, Any], prefix: str
    ) -> None:
        """Flatten a nested dictionary for np.savez serialization."""
        for key, value in d.items():
            full_key = "{}_{}".format(prefix, key) if prefix else key
            if isinstance(value, np.ndarray):
                out[full_key] = value
            elif isinstance(value, (int, float, bool)):
                out[full_key] = np.array(value)
            elif isinstance(value, dict):
                self._flatten_dict(value, out, prefix=full_key)
            elif isinstance(value, list):
                if len(value) > 0 and isinstance(value[0], (int, float, bool)):
                    out[full_key] = np.array(value)
                elif len(value) > 0 and isinstance(value[0], dict):
                    for i, item in enumerate(value):
                        self._flatten_dict(item, out, prefix="{}_{}".format(full_key, i))
            elif isinstance(value, str):
                out[full_key] = np.array(value)

    def _load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load a training checkpoint from a pkl file."""
        import pickle
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)

        episode = checkpoint["episode"]
        total_steps = checkpoint["total_steps"]

        self.total_steps = total_steps

        brain_state = checkpoint.get("brain_state")
        if brain_state is not None:
            self.brain.load_state_dict(brain_state)

        return {"episode": episode, "total_steps": total_steps}

    def _reconstruct_brain_state(self, data: Any) -> Optional[Dict[str, Any]]:
        """Reconstruct brain state dict from flattened checkpoint data."""
        try:
            if "checkpoint" in data.files:
                checkpoint = data["checkpoint"].item()
                return checkpoint.get("brain_state")

            state = {}

            snn_layer_indices = set()
            for key in data.files:
                if key.startswith("brain_state_snn_layers_"):
                    parts = key[len("brain_state_snn_layers_"):].split("_", 1)
                    if parts[0].isdigit():
                        snn_layer_indices.add(int(parts[0]))
            if snn_layer_indices:
                snn_layers_list = []
                for idx in sorted(snn_layer_indices):
                    layer_data = {}
                    prefix = "brain_state_snn_layers_{}_".format(idx)
                    for key in data.files:
                        if key.startswith(prefix):
                            sub_key = key[len(prefix):]
                            layer_data[sub_key] = data[key]
                    if layer_data:
                        snn_layers_list.append(layer_data)
                if snn_layers_list:
                    state["snn_layers"] = snn_layers_list

            if "brain_state_snn_layer" in data.files or any(
                k.startswith("brain_state_snn_layer_") for k in data.files
            ):
                if "snn_layers" not in state:
                    snn_layer_data = {}
                    for key in data.files:
                        if key.startswith("brain_state_snn_layer_"):
                            sub_key = key[len("brain_state_snn_layer_"):]
                            snn_layer_data[sub_key] = data[key]
                    if snn_layer_data:
                        state["snn_layers"] = [snn_layer_data]

            simple_modules = [
                "stdp_layer", "pcn",
                "active_inference", "hd_memory",
                "world_model", "metacognition",
                "global_workspace", "structural_evolution",
                "complementary_memory",
            ]
            for module_name in simple_modules:
                prefix = "brain_state_{}".format(module_name)
                module_data = {}
                for key in data.files:
                    if key.startswith(prefix + "_"):
                        sub_key = key[len(prefix) + 1:]
                        module_data[sub_key] = data[key]
                if module_data:
                    state[module_name] = module_data

            if "brain_state_age" in data.files:
                state["age"] = int(data["brain_state_age"])
            if "brain_state_stage" in data.files:
                state["stage"] = str(data["brain_state_stage"])

            if "brain_state_state_to_obs_W" in data.files:
                state["state_to_obs_W"] = data["brain_state_state_to_obs_W"]
            if "brain_state_state_to_obs_b" in data.files:
                state["state_to_obs_b"] = data["brain_state_state_to_obs_b"]

            if "brain_state_consolidation_counter" in data.files:
                state["consolidation_counter"] = int(data["brain_state_consolidation_counter"])

            if "brain_state_perception_projection_W" in data.files:
                state["perception_projection_W"] = data["brain_state_perception_projection_W"]
            if "brain_state_perception_projection_input_dim" in data.files:
                state["perception_projection_input_dim"] = int(data["brain_state_perception_projection_input_dim"])

            return state if state else None
        except Exception:
            return None

    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the brain without learning.

        Parameters
        ----------
        n_episodes : int
            Number of evaluation episodes.

        Returns
        -------
        dict
            Evaluation metrics.
        """
        rewards = []
        steps_list = []

        for _ in range(n_episodes):
            obs = self.env.reset()
            self.brain.reset_episode()
            total_reward = 0.0
            steps = 0

            for step in range(self.max_steps_per_episode):
                hidden, _, _ = self._perceive_observation(obs)
                thought = self.brain.think(hidden)
                action = thought["action"]
                obs, reward, done, _ = self.env.step(action)
                total_reward += reward
                steps += 1
                if done:
                    break

            rewards.append(total_reward)
            steps_list.append(steps)

        return {
            "avg_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "avg_steps": np.mean(steps_list),
            "min_reward": min(rewards),
            "max_reward": max(rewards),
        }

    def __repr__(self) -> str:
        return (
            f"NIEATrainer(brain={self.brain}, env={self.env}, "
            f"max_episodes={self.max_episodes})"
        )
