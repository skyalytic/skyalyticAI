"""
迷宫网格世界 — 感知运动期（0~3岁）的探索环境。

智能体在 2D 网格中导航，寻找目标位置。
观测：one-hot 位置编码 + 目标方向向量。
动作：0=上 1=下 2=左 3=右。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from skyalyticAI.env.environment import Environment


class GridWorldEnv(Environment):
    """2D 迷宫导航环境，用于感知运动期训练。"""

    # 动作定义：上、下、左、右
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    N_ACTIONS = 4

    def __init__(
        self,
        width: int = 8,
        height: int = 8,
        n_obstacles: int = 6,
        max_steps: int = 200,
        seed: int = 0,
    ) -> None:
        if width < 3 or height < 3:
            raise ValueError("迷宫尺寸至少 3x3")
        self.width = width
        self.height = height
        self.n_obstacles = min(n_obstacles, (width - 2) * (height - 2) - 2)
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self._agent_pos = np.array([0, 0])
        self._goal_pos = np.array([height - 1, width - 1])
        self._obstacles: List[np.ndarray] = []
        self._step_count = 0
        self._done = False
        self._generate_maze()

    def _generate_maze(self) -> None:
        """随机生成障碍物位置。"""
        self._obstacles = []
        occupied = {(0, 0), (self.height - 1, self.width - 1)}
        attempts = 0
        while len(self._obstacles) < self.n_obstacles and attempts < self.n_obstacles * 10:
            r = int(self.rng.integers(1, self.height - 1))
            c = int(self.rng.integers(1, self.width - 1))
            if (r, c) not in occupied:
                self._obstacles.append(np.array([r, c]))
                occupied.add((r, c))
            attempts += 1

    def _is_obstacle(self, pos: np.ndarray) -> bool:
        for obs in self._obstacles:
            if np.array_equal(pos, obs):
                return True
        return False

    def _obs_vector(self) -> np.ndarray:
        """构建观测向量：one-hot 位置 + 目标方向。"""
        # one-hot 位置编码
        pos_onehot = np.zeros(self.height * self.width, dtype=np.float64)
        idx = self._agent_pos[0] * self.width + self._agent_pos[1]
        pos_onehot[idx] = 1.0

        # 目标方向（归一化差向量）
        direction = self._goal_pos.astype(np.float64) - self._agent_pos.astype(np.float64)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction /= norm

        # 障碍物邻近指示（4个方向是否有障碍）
        obstacle_nearby = np.zeros(4, dtype=np.float64)
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for i, (dr, dc) in enumerate(deltas):
            neighbor = self._agent_pos + np.array([dr, dc])
            if self._is_obstacle(neighbor):
                obstacle_nearby[i] = 1.0

        return np.concatenate([pos_onehot, direction, obstacle_nearby])

    def reset(self) -> np.ndarray:
        self._agent_pos = np.array([0, 0])
        self._goal_pos = np.array([self.height - 1, self.width - 1])
        self._step_count = 0
        self._done = False
        self._generate_maze()
        return self._obs_vector()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self._done:
            return self._obs_vector(), 0.0, True, {"mode": "motor"}

        action = int(action) % self.N_ACTIONS
        deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dr, dc = deltas[action]
        new_pos = self._agent_pos + np.array([dr, dc])

        old_pos = self._agent_pos.copy()
        # 边界检查
        if (0 <= new_pos[0] < self.height
                and 0 <= new_pos[1] < self.width
                and not self._is_obstacle(new_pos)):
            self._agent_pos = new_pos

        self._step_count += 1

        # 到达目标
        reached = np.array_equal(self._agent_pos, self._goal_pos)
        if reached:
            self._done = True
            reward = 10.0
        elif self._step_count >= self.max_steps:
            self._done = True
            reward = -0.5
        else:
            # 距离变近给小正奖励，变远给小负奖励
            old_dist = np.linalg.norm(self._goal_pos - old_pos)
            new_dist = np.linalg.norm(self._goal_pos - self._agent_pos)
            reward = 0.1 if new_dist < old_dist else -0.05

        info: Dict[str, Any] = {
            "mode": "motor",
            "activity": "motor",
            "agent_pos": self._agent_pos.tolist(),
            "goal_pos": self._goal_pos.tolist(),
            "steps": self._step_count,
            "reached_goal": reached,
        }
        return self._obs_vector(), reward, self._done, info

    def get_observation_dim(self) -> int:
        # one-hot (H*W) + 方向 (2) + 障碍邻近 (4)
        return self.height * self.width + 2 + 4

    def get_action_dim(self) -> int:
        return self.N_ACTIONS

    def render(self) -> Any:
        """ASCII 渲染。"""
        grid = np.full((self.height, self.width), ".", dtype=str)
        for obs in self._obstacles:
            grid[obs[0], obs[1]] = "#"
        grid[self._goal_pos[0], self._goal_pos[1]] = "G"
        grid[self._agent_pos[0], self._agent_pos[1]] = "A"
        return "\n".join(" ".join(row) for row in grid)
