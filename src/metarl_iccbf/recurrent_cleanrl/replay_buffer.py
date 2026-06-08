"""Sequence replay buffer for recurrent off-policy learning (SAC + Mamba2).

Stores fixed-length trajectory chunks with burn-in context prefix.
Each stored chunk has shape ``(B + L, ...)``:

- ``[0:B]``   — burn-in region used to warm up Mamba2 states (no loss)
- ``[B:B+L]`` — training region used for Q / policy loss computation

The :class:`_ChunkCollector` accumulates per-env transitions and emits
completed chunks to the replay buffer with sliding-window overlap.
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Sequence Replay Buffer
# ---------------------------------------------------------------------------

class SequenceReplayBuffer:
    """Fixed-size circular buffer of trajectory chunks for recurrent SAC.

    Parameters
    ----------
    capacity : int
        Maximum number of chunks to store.
    chunk_len : int
        Training sequence length *L*.
    burn_in : int
        Burn-in prefix length *B*.
    obs_shape : tuple[int, ...]
    act_shape : tuple[int, ...]
    """

    def __init__(
        self,
        capacity: int,
        chunk_len: int,
        burn_in: int,
        obs_shape: tuple[int, ...],
        act_shape: tuple[int, ...],
    ):
        self.capacity = capacity
        self.chunk_len = chunk_len
        self.burn_in = burn_in
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.total_len = burn_in + chunk_len

        # Circular buffer storage
        self.obs = np.zeros(
            (capacity, self.total_len, *obs_shape), dtype=np.float32
        )
        self.actions = np.zeros(
            (capacity, self.total_len, *act_shape), dtype=np.float32
        )
        self.rewards = np.zeros(
            (capacity, self.total_len), dtype=np.float32
        )
        self.next_obs = np.zeros(
            (capacity, self.total_len, *obs_shape), dtype=np.float32
        )
        self.episode_starts = np.zeros(
            (capacity, self.total_len), dtype=np.float32
        )
        self.dones = np.zeros(
            (capacity, self.total_len), dtype=np.float32
        )

        self._pos = 0
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        episode_starts: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Store one trajectory chunk of shape ``(B+L, ...)``."""
        idx = self._pos % self.capacity
        self.obs[idx] = obs
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.next_obs[idx] = next_obs
        self.episode_starts[idx] = episode_starts
        self.dones[idx] = dones

        self._pos += 1
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Sample a random batch of chunks.

        Returns
        -------
        dict with keys mapping to tensors of shape ``(batch, B+L, ...)``.
        The caller splits ``[0:B]`` (burn-in) from ``[B:B+L]`` (training).
        """
        indices = np.random.randint(0, self._size, size=batch_size)

        return {
            "obs": torch.as_tensor(
                self.obs[indices], dtype=torch.float32, device=device
            ),
            "actions": torch.as_tensor(
                self.actions[indices], dtype=torch.float32, device=device
            ),
            "rewards": torch.as_tensor(
                self.rewards[indices], dtype=torch.float32, device=device
            ),
            "next_obs": torch.as_tensor(
                self.next_obs[indices], dtype=torch.float32, device=device
            ),
            "episode_starts": torch.as_tensor(
                self.episode_starts[indices], dtype=torch.float32, device=device
            ),
            "dones": torch.as_tensor(
                self.dones[indices], dtype=torch.float32, device=device
            ),
        }


# ---------------------------------------------------------------------------
# Chunk Collector
# ---------------------------------------------------------------------------

class ChunkCollector:
    """Accumulates per-env transitions and emits ``(B+L)`` chunks.

    For each environment, maintains a rolling window of size ``B + L``.
    When the window fills, the chunk is stored and the window slides:
    the last ``B`` steps become the start of the next window.

    Parameters
    ----------
    n_envs : int
    chunk_len : int
        Training sequence length *L*.
    burn_in : int
        Burn-in prefix length *B*.
    obs_shape : tuple[int, ...]
    act_shape : tuple[int, ...]
    """

    def __init__(
        self,
        n_envs: int,
        chunk_len: int,
        burn_in: int,
        obs_shape: tuple[int, ...],
        act_shape: tuple[int, ...],
    ):
        self.n_envs = n_envs
        self.chunk_len = chunk_len
        self.burn_in = burn_in
        self.total_len = burn_in + chunk_len

        self._obs = np.zeros(
            (n_envs, self.total_len, *obs_shape), dtype=np.float32
        )
        self._actions = np.zeros(
            (n_envs, self.total_len, *act_shape), dtype=np.float32
        )
        self._rewards = np.zeros(
            (n_envs, self.total_len), dtype=np.float32
        )
        self._next_obs = np.zeros(
            (n_envs, self.total_len, *obs_shape), dtype=np.float32
        )
        self._episode_starts = np.zeros(
            (n_envs, self.total_len), dtype=np.float32
        )
        self._dones = np.zeros(
            (n_envs, self.total_len), dtype=np.float32
        )

        self._cursors = np.zeros(n_envs, dtype=int)

        # On first chunk, burn-in is all episode_starts=1 (empty context)
        self._episode_starts[:, :burn_in] = 1.0
        self._cursors[:] = burn_in

    def add_step(
        self,
        env_idx: int,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        episode_start: float,
        done: float,
    ) -> bool:
        """Add one transition for one environment.

        Returns ``True`` if a chunk is ready (call :meth:`get_chunk`).
        """
        t = self._cursors[env_idx]
        self._obs[env_idx, t] = obs
        self._actions[env_idx, t] = action
        self._rewards[env_idx, t] = reward
        self._next_obs[env_idx, t] = next_obs
        self._episode_starts[env_idx, t] = episode_start
        self._dones[env_idx, t] = done
        self._cursors[env_idx] = t + 1
        return self._cursors[env_idx] >= self.total_len

    def get_chunk(self, env_idx: int) -> tuple[
        np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray,
    ]:
        """Retrieve completed chunk and slide window.

        Returns copies of ``(obs, actions, rewards, next_obs, episode_starts, dones)``,
        each of shape ``(B+L, ...)``.
        """
        chunk = (
            self._obs[env_idx].copy(),
            self._actions[env_idx].copy(),
            self._rewards[env_idx].copy(),
            self._next_obs[env_idx].copy(),
            self._episode_starts[env_idx].copy(),
            self._dones[env_idx].copy(),
        )
        # Slide: last B steps become new burn-in
        B = self.burn_in
        self._obs[env_idx, :B] = self._obs[env_idx, -B:]
        self._actions[env_idx, :B] = self._actions[env_idx, -B:]
        self._rewards[env_idx, :B] = self._rewards[env_idx, -B:]
        self._next_obs[env_idx, :B] = self._next_obs[env_idx, -B:]
        self._episode_starts[env_idx, :B] = self._episode_starts[env_idx, -B:]
        self._dones[env_idx, :B] = self._dones[env_idx, -B:]
        self._cursors[env_idx] = B
        return chunk
