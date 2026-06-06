"""
Data Pipeline for NIEA Training

Provides dataset abstractions and data loading utilities for
training the NIEABrain with real-world data. Supports:
- Custom dataset interfaces
- Batch data loading with shuffling
- Multimodal data (images, audio, sensor data)
- Data preprocessing and augmentation
- Integration with the perception pipeline
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np


class Dataset(ABC):
    """
    Abstract base class for NIEA datasets.

    All datasets must implement:
    - __len__: Return the number of samples
    - __getitem__: Return a single sample by index
    - get_observation_dim: Return observation dimensionality
    - get_action_dim: Return number of available actions (0 for unsupervised)
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        ...

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """
        Get a single sample by index.

        Parameters
        ----------
        index : int
            Sample index.

        Returns
        -------
        dict
            Dictionary containing:
            - 'observation': np.ndarray of shape (obs_dim,)
            - 'action': np.ndarray of shape (action_dim,) (optional)
            - 'reward': float (optional)
            - 'next_observation': np.ndarray of shape (obs_dim,) (optional)
            - 'done': bool (optional)
            - Any additional modality data
        """
        ...

    @abstractmethod
    def get_observation_dim(self) -> int:
        """Return the dimensionality of observations."""
        ...

    def get_action_dim(self) -> int:
        """Return the number of available actions (0 for unsupervised)."""
        return 0

    def get_modality_keys(self) -> List[str]:
        """Return the list of available modality keys in samples."""
        return ["observation"]


class NIEADataLoader:
    """
    Data loader for NIEA training with batching and shuffling.

    Provides an iterable interface over a dataset, yielding
    batches of samples suitable for training the NIEABrain.

    Parameters
    ----------
    dataset : Dataset
        The dataset to load from.
    batch_size : int
        Number of samples per batch. Must be positive.
    shuffle : bool
        Whether to shuffle the data at each epoch.
    drop_last : bool
        Whether to drop the last incomplete batch.
    seed : int or None
        Random seed for reproducibility of shuffling.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive, got {}".format(batch_size))

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

        self._indices: np.ndarray = np.arange(len(dataset))
        self._position: int = 0
        self._epoch: int = 0

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        """Iterate over batches."""
        self._reset()
        while self._position < len(self.dataset):
            batch = self._get_batch()
            if batch is not None:
                yield batch

    def _reset(self) -> None:
        """Reset for a new epoch."""
        self._indices = np.arange(len(self.dataset))
        if self.shuffle:
            self.rng.shuffle(self._indices)
        self._position = 0
        self._epoch += 1

    def _get_batch(self) -> Optional[Dict[str, np.ndarray]]:
        """Get the next batch."""
        end = min(self._position + self.batch_size, len(self.dataset))
        if self.drop_last and (end - self._position) < self.batch_size:
            self._position = len(self.dataset)
            return None

        batch_indices = self._indices[self._position:end]
        self._position = end

        if len(batch_indices) == 0:
            return None

        samples = [self.dataset[int(i)] for i in batch_indices]

        batch = {}
        for key in samples[0].keys():
            values = [s[key] for s in samples]
            if isinstance(values[0], np.ndarray):
                batch[key] = np.stack(values, axis=0)
            elif isinstance(values[0], (int, float)):
                batch[key] = np.array(values, dtype=np.float64)
            elif isinstance(values[0], bool):
                batch[key] = np.array(values, dtype=bool)
            else:
                batch[key] = values

        batch["_batch_size"] = len(batch_indices)
        batch["_epoch"] = self._epoch

        return batch

    @property
    def epoch(self) -> int:
        """Current epoch number."""
        return self._epoch


class ExperienceReplayDataset(Dataset):
    """
    Dataset backed by an experience replay buffer.

    Stores transitions (observation, action, reward, next_observation, done)
    and provides random access for batch training.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    obs_dim : int
        Observation dimensionality.
    action_dim : int
        Action dimensionality.
    """

    def __init__(
        self,
        capacity: int = 10000,
        obs_dim: int = 10,
        action_dim: int = 4,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if obs_dim <= 0:
            raise ValueError("obs_dim must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")

        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self._observations = np.zeros((capacity, obs_dim), dtype=np.float64)
        self._actions = np.zeros((capacity, action_dim), dtype=np.float64)
        self._rewards = np.zeros(capacity, dtype=np.float64)
        self._next_observations = np.zeros((capacity, obs_dim), dtype=np.float64)
        self._dones = np.zeros(capacity, dtype=bool)

        self._size: int = 0
        self._ptr: int = 0

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        """
        Add a transition to the buffer.

        Parameters
        ----------
        observation : np.ndarray
            Current observation, shape (obs_dim,).
        action : np.ndarray
            Action taken, shape (action_dim,).
        reward : float
            Reward received.
        next_observation : np.ndarray
            Next observation, shape (obs_dim,).
        done : bool
            Whether the episode ended.
        """
        observation = np.asarray(observation, dtype=np.float64).flatten()
        action = np.asarray(action, dtype=np.float64).flatten()
        next_observation = np.asarray(next_observation, dtype=np.float64).flatten()

        if observation.shape[0] != self.obs_dim:
            raise ValueError(
                "observation shape mismatch: expected ({},), got {}".format(
                    self.obs_dim, observation.shape
                )
            )
        if action.shape[0] != self.action_dim:
            raise ValueError(
                "action shape mismatch: expected ({},), got {}".format(
                    self.action_dim, action.shape
                )
            )

        self._observations[self._ptr] = observation
        self._actions[self._ptr] = action
        self._rewards[self._ptr] = reward
        self._next_observations[self._ptr] = next_observation
        self._dones[self._ptr] = done

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_transition(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        """
        Add a transition with discrete action index.

        Parameters
        ----------
        observation : np.ndarray
            Current observation.
        action : int
            Discrete action index.
        reward : float
            Reward received.
        next_observation : np.ndarray
            Next observation.
        done : bool
            Whether the episode ended.
        """
        action_vec = np.zeros(self.action_dim, dtype=np.float64)
        if 0 <= action < self.action_dim:
            action_vec[action] = 1.0
        self.add(observation, action_vec, reward, next_observation, done)

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        if index < 0 or index >= self._size:
            raise IndexError("index {} out of range [0, {})".format(index, self._size))

        return {
            "observation": self._observations[index].copy(),
            "action": self._actions[index].copy(),
            "reward": self._rewards[index],
            "next_observation": self._next_observations[index].copy(),
            "done": self._dones[index],
        }

    def get_observation_dim(self) -> int:
        return self.obs_dim

    def get_action_dim(self) -> int:
        return self.action_dim

    def sample_batch(
        self, batch_size: int, rng: Optional[np.random.Generator] = None
    ) -> Dict[str, np.ndarray]:
        """
        Sample a random batch of transitions.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.
        rng : np.random.Generator or None
            Random number generator.

        Returns
        -------
        dict
            Batch of transitions with stacked arrays.
        """
        if batch_size > self._size:
            raise ValueError(
                "batch_size {} exceeds buffer size {}".format(batch_size, self._size)
            )

        if rng is None:
            rng = np.random.default_rng()

        indices = rng.choice(self._size, size=batch_size, replace=False)

        return {
            "observation": self._observations[indices].copy(),
            "action": self._actions[indices].copy(),
            "reward": self._rewards[indices].copy(),
            "next_observation": self._next_observations[indices].copy(),
            "done": self._dones[indices].copy(),
        }

    def clear(self) -> None:
        """Clear all stored transitions."""
        self._size = 0
        self._ptr = 0

    def state_dict(self) -> Dict[str, Any]:
        """Return the buffer state for serialization."""
        return {
            "observations": self._observations[:self._size].copy(),
            "actions": self._actions[:self._size].copy(),
            "rewards": self._rewards[:self._size].copy(),
            "next_observations": self._next_observations[:self._size].copy(),
            "dones": self._dones[:self._size].copy(),
            "size": self._size,
            "ptr": self._ptr,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load buffer state from a dictionary."""
        size = state["size"]
        self._observations[:size] = state["observations"]
        self._actions[:size] = state["actions"]
        self._rewards[:size] = state["rewards"]
        self._next_observations[:size] = state["next_observations"]
        self._dones[:size] = state["dones"]
        self._size = size
        self._ptr = state["ptr"]


class MultimodalDataset(Dataset):
    """
    Dataset for multimodal data (e.g., paired image + audio).

    Each sample contains data from multiple modalities, which
    are processed by the perception pipeline before being fed
    to the NIEABrain.

    Parameters
    ----------
    modality_configs : dict
        Mapping from modality name to configuration dict.
        Each config must have 'dim' (feature dimension).
        Example: {"visual": {"dim": 64}, "audio": {"dim": 64}}
    capacity : int
        Maximum number of samples.
    action_dim : int
        Number of actions (0 for unsupervised).
    """

    def __init__(
        self,
        modality_configs: Dict[str, Dict[str, Any]],
        capacity: int = 10000,
        action_dim: int = 0,
    ) -> None:
        if not modality_configs:
            raise ValueError("modality_configs must not be empty")
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.modality_configs = modality_configs
        self.capacity = capacity
        self.action_dim = action_dim
        self.modalities = list(modality_configs.keys())

        self._data: Dict[str, np.ndarray] = {}
        for name, config in modality_configs.items():
            dim = config["dim"]
            self._data[name] = np.zeros((capacity, dim), dtype=np.float64)

        self._actions = np.zeros((capacity, max(action_dim, 1)), dtype=np.float64)
        self._rewards = np.zeros(capacity, dtype=np.float64)

        self._size: int = 0
        self._ptr: int = 0

    def add_sample(
        self,
        modalities: Dict[str, np.ndarray],
        action: Optional[np.ndarray] = None,
        reward: float = 0.0,
    ) -> None:
        """
        Add a multimodal sample.

        Parameters
        ----------
        modalities : dict
            Mapping from modality name to feature vector.
        action : np.ndarray or None
            Action vector (if applicable).
        reward : float
            Reward value.
        """
        for name, features in modalities.items():
            if name not in self._data:
                raise ValueError("Unknown modality: {}".format(name))
            features = np.asarray(features, dtype=np.float64).flatten()
            expected_dim = self.modality_configs[name]["dim"]
            if features.shape[0] != expected_dim:
                if features.shape[0] < expected_dim:
                    padded = np.zeros(expected_dim, dtype=np.float64)
                    padded[:features.shape[0]] = features
                    features = padded
                else:
                    features = features[:expected_dim]
            self._data[name][self._ptr] = features

        if action is not None and self.action_dim > 0:
            action = np.asarray(action, dtype=np.float64).flatten()
            self._actions[self._ptr, :action.shape[0]] = action

        self._rewards[self._ptr] = reward

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        if index < 0 or index >= self._size:
            raise IndexError("index {} out of range [0, {})".format(index, self._size))

        sample = {}
        for name in self.modalities:
            sample[name] = self._data[name][index].copy()

        if self.action_dim > 0:
            sample["action"] = self._actions[index, :self.action_dim].copy()
        sample["reward"] = self._rewards[index]

        return sample

    def get_observation_dim(self) -> int:
        total = 0
        for config in self.modality_configs.values():
            total += config["dim"]
        return total

    def get_action_dim(self) -> int:
        return self.action_dim

    def get_modality_keys(self) -> List[str]:
        return list(self.modalities)

    def state_dict(self) -> Dict[str, Any]:
        """Return the dataset state for serialization."""
        return {
            "data": {k: v[:self._size].copy() for k, v in self._data.items()},
            "actions": self._actions[:self._size].copy(),
            "rewards": self._rewards[:self._size].copy(),
            "size": self._size,
            "ptr": self._ptr,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load dataset state from a dictionary."""
        size = state["size"]
        for name in self.modalities:
            self._data[name][:size] = state["data"][name]
        self._actions[:size] = state["actions"]
        self._rewards[:size] = state["rewards"]
        self._size = size
        self._ptr = state["ptr"]
