"""
Global Workspace - Consciousness Emergence Mechanism

Implements the Global Workspace Theory (GWT) for consciousness emergence
in the NIEA architecture. The global workspace is a shared communication
channel where specialized modules compete for access. When a module's
output exceeds a dynamic threshold, it "wins" access to the workspace
and broadcasts its information to all other modules.

Key mechanisms:
1. Module competition: Each module produces a "bid" (activation level)
2. Winner selection: The highest bid above threshold gains workspace access
3. Global broadcast: The winner's output is broadcast to all modules
4. Ignition: When multiple modules co-activate, a phase transition occurs
   (consciousness "ignition") where the workspace becomes globally accessible
5. Dynamic threshold: The access threshold adapts based on recent activity

This is the fourth theoretical pillar of NIEA:
- SNN Substrate (neurons/lif.py, neurons/snn_layer.py)
- Predictive Coding Learning (predictive_coding/pcn.py)
- Complementary Memory System (memory/hdc.py + memory/consolidation.py)
- Global Workspace Consciousness Emergence (consciousness/global_workspace.py)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class GlobalWorkspace:
    """
    Global Workspace for consciousness emergence.

    Implements a competitive broadcasting mechanism where specialized
    neural modules compete for access to a shared workspace. The winner
    broadcasts its information globally, enabling information integration
    across otherwise independent modules.

    Parameters
    ----------
    workspace_dim : int
        Dimensionality of the global workspace representation.
    n_modules : int
        Number of competing modules.
    competition_threshold : float
        Minimum activation for a module to compete for access.
    ignition_threshold : float
        Co-activation level required for consciousness ignition.
    broadcast_strength : float
        Strength of the global broadcast signal.
    threshold_adaptation_rate : float
        Rate at which the competition threshold adapts.
    """

    def __init__(
        self,
        workspace_dim: int = 64,
        n_modules: int = 7,
        competition_threshold: float = 0.3,
        ignition_threshold: float = 0.6,
        broadcast_strength: float = 0.5,
        threshold_adaptation_rate: float = 0.01,
    ) -> None:
        if workspace_dim <= 0:
            raise ValueError("workspace_dim must be positive")
        if n_modules <= 0:
            raise ValueError("n_modules must be positive")

        self.workspace_dim = workspace_dim
        self.n_modules = n_modules
        self.competition_threshold = competition_threshold
        self.ignition_threshold = ignition_threshold
        self.broadcast_strength = broadcast_strength
        self.threshold_adaptation_rate = threshold_adaptation_rate

        self.workspace_state = np.zeros(workspace_dim, dtype=np.float64)
        self.module_bids = np.zeros(n_modules, dtype=np.float64)
        self.module_outputs: List[Optional[np.ndarray]] = [None] * n_modules
        self.winner_index: Optional[int] = None
        self.is_ignited = False
        self.ignition_history: List[bool] = []

        self.access_W = [
            np.random.randn(workspace_dim, workspace_dim)
            * np.sqrt(2.0 / (workspace_dim + workspace_dim))
            for _ in range(n_modules)
        ]

        self._recent_winners: List[int] = []
        self._max_recent = 100

    def submit_bid(
        self,
        module_index: int,
        activation: float,
        module_output: np.ndarray,
    ) -> None:
        """
        Submit a module's bid for workspace access.

        Parameters
        ----------
        module_index : int
            Index of the competing module.
        activation : float
            Activation level (bid strength). Higher = more likely to win.
        module_output : np.ndarray
            The module's output representation to broadcast if it wins.
        """
        if module_index < 0 or module_index >= self.n_modules:
            raise ValueError(
                "module_index must be in [0, {}), got {}".format(
                    self.n_modules, module_index
                )
            )

        self.module_bids[module_index] = float(activation)
        output = np.asarray(module_output, dtype=np.float64).flatten()
        if output.shape[0] != self.workspace_dim:
            if output.shape[0] < self.workspace_dim:
                padded = np.zeros(self.workspace_dim, dtype=np.float64)
                padded[:output.shape[0]] = output
                output = padded
            else:
                output = output[:self.workspace_dim]
        self.module_outputs[module_index] = output

    def compete(self) -> Dict[str, Any]:
        """
        Run the competition process and determine workspace access.

        Steps:
        1. Filter modules above competition threshold
        2. Select the module with highest bid as winner
        3. Check for ignition (co-activation of multiple modules)
        4. Broadcast winner's output to workspace
        5. Adapt competition threshold

        Returns
        -------
        dict
            Competition results containing:
            - 'winner': index of winning module (or None)
            - 'is_ignited': whether consciousness ignition occurred
            - 'broadcast': the broadcast workspace state
            - 'bids': all module bids
            - 'n_competing': number of modules above threshold
        """
        competing = self.module_bids >= self.competition_threshold
        n_competing = int(np.sum(competing))

        if n_competing == 0:
            self.winner_index = None
            self.is_ignited = False
            self.ignition_history.append(False)
            return {
                "winner": None,
                "is_ignited": False,
                "broadcast": self.workspace_state.copy(),
                "bids": self.module_bids.copy(),
                "n_competing": 0,
            }

        sorted_indices = np.argsort(self.module_bids)[::-1]
        winner_idx = int(sorted_indices[0])
        self.winner_index = winner_idx

        top_bids = self.module_bids[competing]
        n_competing_true = int(np.sum(competing))
        if n_competing_true >= 2:
            top_bids = self.module_bids[competing]
            coactivation = float(np.mean(top_bids))
            self.is_ignited = coactivation >= self.ignition_threshold
        else:
            self.is_ignited = False

        self.ignition_history.append(self.is_ignited)

        winner_output = self.module_outputs[winner_idx]
        if winner_output is not None:
            projected = self.access_W[winner_idx] @ winner_output

            if self.is_ignited:
                competing_indices = np.where(competing)[0]
                for idx in competing_indices:
                    if idx == winner_idx:
                        continue
                    other_output = self.module_outputs[idx]
                    if other_output is not None:
                        other_proj = self.access_W[idx] @ other_output
                        weight = self.module_bids[idx] / (
                            self.module_bids[winner_idx] + 1e-10
                        )
                        projected += weight * other_proj

            projected = np.tanh(projected)

            self.workspace_state = (
                1.0 - self.broadcast_strength
            ) * self.workspace_state + self.broadcast_strength * projected

        self._adapt_threshold(n_competing)

        self._recent_winners.append(winner_idx)
        if len(self._recent_winners) > self._max_recent:
            self._recent_winners.pop(0)

        # Learn: update access_W for the winning module
        winner_output = self.module_outputs[winner_idx]
        if winner_output is not None:
            self.learn(winner_idx, winner_output, projected)

        return {
            "winner": winner_idx,
            "is_ignited": self.is_ignited,
            "broadcast": self.workspace_state.copy(),
            "bids": self.module_bids.copy(),
            "n_competing": n_competing,
        }

    def learn(self, winner_idx: int, module_output: np.ndarray, broadcast: np.ndarray, lr: float = 0.001) -> None:
        """
        Update access_W for the winning module using a Hebbian-like rule.

        When a module wins the competition, strengthen its access weights
        based on the correlation between the broadcast signal and its output.

        Parameters
        ----------
        winner_idx : int
            Index of the winning module.
        module_output : np.ndarray
            Output of the winning module.
        broadcast : np.ndarray
            Current workspace broadcast signal.
        lr : float
            Learning rate for the Hebbian update.
        """
        if winner_idx < len(self.access_W):
            self.access_W[winner_idx] += lr * np.outer(broadcast, module_output)
            # Clip to prevent unbounded growth
            self.access_W[winner_idx] = np.clip(self.access_W[winner_idx], -2.0, 2.0)

    def _adapt_threshold(self, n_competing: int) -> None:
        """
        Adapt the competition threshold based on recent activity.

        If too many modules are competing, raise the threshold.
        If too few are competing, lower it.
        """
        target_competing = max(1, self.n_modules // 3)
        error = n_competing - target_competing
        self.competition_threshold += (
            self.threshold_adaptation_rate * error
        )
        self.competition_threshold = np.clip(
            self.competition_threshold, 0.1, 0.9
        )

    def get_broadcast(self) -> np.ndarray:
        """
        Get the current global broadcast signal.

        Returns
        -------
        np.ndarray
            The workspace state broadcast to all modules, shape (workspace_dim,).
        """
        return self.workspace_state.copy()

    def get_consciousness_level(self) -> float:
        """
        Estimate the current level of consciousness.

        Based on:
        - Recent ignition frequency
        - Workspace activation level
        - Diversity of winning modules

        Returns
        -------
        float
            Consciousness level in [0, 1].
        """
        if len(self.ignition_history) == 0:
            return 0.0

        recent_window = min(50, len(self.ignition_history))
        ignition_rate = np.mean(self.ignition_history[-recent_window:])

        workspace_activation = float(np.mean(np.abs(self.workspace_state)))

        if len(self._recent_winners) > 10:
            recent_set = set(self._recent_winners[-50:])
            diversity = len(recent_set) / self.n_modules
        else:
            diversity = 0.0

        consciousness = (
            0.4 * ignition_rate
            + 0.3 * min(1.0, workspace_activation)
            + 0.3 * diversity
        )

        return float(np.clip(consciousness, 0.0, 1.0))

    def reset(self) -> None:
        """Reset the global workspace state."""
        self.workspace_state = np.zeros(self.workspace_dim, dtype=np.float64)
        self.module_bids = np.zeros(self.n_modules, dtype=np.float64)
        self.module_outputs = [None] * self.n_modules
        self.winner_index = None
        self.is_ignited = False
        self.ignition_history = []
        self._recent_winners = []

    def state_dict(self) -> Dict[str, Any]:
        """Return the workspace state for serialization."""
        return {
            "workspace_state": self.workspace_state.copy(),
            "access_W": [W.copy() for W in self.access_W],
            "competition_threshold": self.competition_threshold,
            "ignition_history": list(self.ignition_history[-100:]),
            "recent_winners": list(self._recent_winners[-100:]),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load workspace state from a dictionary."""
        self.workspace_state = state["workspace_state"].copy()
        if "access_W" in state:
            for i, W in enumerate(state["access_W"]):
                if i < self.n_modules:
                    self.access_W[i] = W.copy()
        if "competition_threshold" in state:
            self.competition_threshold = float(state["competition_threshold"])
        if "ignition_history" in state:
            self.ignition_history = list(state["ignition_history"])
        if "recent_winners" in state:
            self._recent_winners = list(state["recent_winners"])
