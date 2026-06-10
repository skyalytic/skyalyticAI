"""
环境基类 — 所有训练环境的统一接口。

遵循 OpenAI Gym 风格：
  reset() -> observation
  step(action) -> (observation, reward, done, info)
  get_observation_dim() -> int
  get_action_dim() -> int
  render() -> Any
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np


class Environment(ABC):
    """所有 NIEA 训练环境的抽象基类。"""

    @abstractmethod
    def reset(self) -> Any:
        """重置环境，返回初始观测。

        Returns
        -------
        observation : np.ndarray or dict
            初始观测（单模态为 ndarray，多模态为 dict）。
        """
        ...

    @abstractmethod
    def step(self, action: int) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """执行一步交互。

        Parameters
        ----------
        action : int
            智能体选择的动作（离散动作空间）。

        Returns
        -------
        observation : np.ndarray or dict
            新观测。
        reward : float
            环境返回的标量奖励。
        done : bool
            当前回合是否结束。
        info : dict
            附加信息（如 mode, subject, target_char 等）。
        """
        ...

    @abstractmethod
    def get_observation_dim(self) -> int:
        """返回观测向量维度。"""
        ...

    @abstractmethod
    def get_action_dim(self) -> int:
        """返回动作空间大小。"""
        ...

    def render(self) -> Any:
        """可选的可视化方法。默认无操作。"""
        return None

    def close(self) -> None:
        """清理资源。"""
        pass

    def seed(self, seed: int) -> None:
        """设置随机种子。子类可覆写。"""
        self._seed = seed

    @property
    def unwrapped(self) -> "Environment":
        """返回未包装的环境。"""
        return self
