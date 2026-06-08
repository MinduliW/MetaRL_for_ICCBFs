"""Rollout buffer for recurrent PPO with context-overlap burn-in.

Stores ``(n_steps, n_envs, ...)`` transitions without any Mamba state
tensors.  Training always starts the parallel scan from zero state;
context observations from the previous rollout serve as burn-in so the
scan has "warm" states by the time it reaches actual training positions.

Minibatches preserve temporal order: we shuffle *environments*, not
timesteps.
"""

from __future__ import annotations

from typing import Generator, Optional

import numpy as np
import torch


class RolloutBuffer:
    """Fixed-size rollout buffer for recurrent PPO.

    Parameters
    ----------
    n_steps : int
        Rollout length per environment.
    n_envs : int
        Number of parallel environments.
    obs_shape : tuple[int, ...]
        Observation shape (excluding batch dimensions).
    act_shape : tuple[int, ...]
        Action shape.
    gamma, gae_lambda : float
        GAE hyperparameters.
    burn_in : int
        Number of context-overlap steps prepended during training.
    """

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        obs_shape: tuple[int, ...],
        act_shape: tuple[int, ...],
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        burn_in: int = 0,
        n_objectives: int = 0,
    ):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.burn_in = burn_in
        self.n_objectives = n_objectives

        # ---- Main storage (n_steps, n_envs, ...) ----
        self.obs = np.zeros((n_steps, n_envs, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs, *act_shape), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)

        # Computed after rollout
        self.advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.returns = np.zeros((n_steps, n_envs), dtype=np.float32)

        # ---- Preference weights (multi_body MORL only) ----
        # Shape (n_steps, n_envs, n_objectives); None when not used.
        if n_objectives > 0:
            self.weights = np.zeros((n_steps, n_envs, n_objectives), dtype=np.float32)
        else:
            self.weights = None

        # ---- Burn-in context from previous rollout ----
        if burn_in > 0:
            self.context_obs = np.zeros(
                (burn_in, n_envs, *obs_shape), dtype=np.float32
            )
            self.context_episode_starts = np.ones(
                (burn_in, n_envs), dtype=np.float32
            )
            if n_objectives > 0:
                self.context_weights = np.zeros(
                    (burn_in, n_envs, n_objectives), dtype=np.float32
                )
            else:
                self.context_weights = None
        else:
            self.context_obs = None
            self.context_episode_starts = None
            self.context_weights = None

        self.pos = 0

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the write cursor for a new rollout."""
        self.pos = 0

    def set_context(
        self,
        context_obs: np.ndarray,
        context_episode_starts: np.ndarray,
        context_weights: Optional[np.ndarray] = None,
    ) -> None:
        """Store burn-in context from the previous rollout.

        Called once before :meth:`add` begins for a new rollout.
        """
        if self.burn_in > 0 and self.context_obs is not None and self.context_episode_starts is not None:
            self.context_obs[:] = context_obs
            self.context_episode_starts[:] = context_episode_starts
            if context_weights is not None and self.context_weights is not None:
                self.context_weights[:] = context_weights

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray,
        w: Optional[np.ndarray] = None,
    ) -> None:
        """Record one timestep across all environments."""
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.episode_starts[self.pos] = episode_start
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        if w is not None and self.weights is not None:
            self.weights[self.pos] = w
        self.pos += 1

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------

    def compute_returns_and_advantage(
        self,
        last_values: np.ndarray,
        last_dones: np.ndarray,
    ) -> None:
        """Generalized Advantage Estimation.

        Parameters
        ----------
        last_values : (n_envs,)
            V(s_{T}) bootstrap values.
        last_dones : (n_envs,)
            Whether the last step was terminal (1.0 = done).
        """
        last_gae = np.zeros(self.n_envs, dtype=np.float32)

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[t + 1]
                next_values = self.values[t + 1]

            delta = (
                self.rewards[t]
                + self.gamma * next_values * next_non_terminal
                - self.values[t]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    # ------------------------------------------------------------------
    # Minibatch iteration
    # ------------------------------------------------------------------

    def get(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Generator[dict[str, torch.Tensor], None, None]:
        """Yield sequential minibatches (shuffle envs, keep time order).

        ``batch_size`` is the number of *environments* per minibatch.
        If ``batch_size >= n_envs``, one minibatch with all envs.

        Each yielded dict has keys:
            obs             (B, T, *obs_shape)
            actions         (B, T, *act_shape)
            episode_starts  (B, T)
            old_values      (B, T)
            old_log_probs   (B, T)
            advantages      (B, T)
            returns         (B, T)
            context_obs            (B, K, *obs_shape)   [if burn_in > 0]
            context_episode_starts (B, K)               [if burn_in > 0]
        """
        env_indices = np.random.permutation(self.n_envs)

        # Split into minibatches of `batch_size` environments
        start = 0
        while start < self.n_envs:
            end = min(start + batch_size, self.n_envs)
            mb_inds = env_indices[start:end]
            start = end

            # Slice: (n_steps, mb_size, ...) → transpose → (mb_size, n_steps, ...)
            mb = {
                "obs": self._to_tensor(self.obs[:, mb_inds].transpose(1, 0, 2), device),
                "actions": self._to_tensor(
                    self.actions[:, mb_inds].transpose(1, 0, 2), device
                ),
                "episode_starts": self._to_tensor(
                    self.episode_starts[:, mb_inds].T, device
                ),
                "old_values": self._to_tensor(
                    self.values[:, mb_inds].T, device
                ),
                "old_log_probs": self._to_tensor(
                    self.log_probs[:, mb_inds].T, device
                ),
                "advantages": self._to_tensor(
                    self.advantages[:, mb_inds].T, device
                ),
                "returns": self._to_tensor(
                    self.returns[:, mb_inds].T, device
                ),
            }

            if self.burn_in > 0:
                mb["context_obs"] = self._to_tensor(
                    self.context_obs[:, mb_inds].transpose(1, 0, 2), device
                )
                mb["context_episode_starts"] = self._to_tensor(
                    self.context_episode_starts[:, mb_inds].T, device
                )
                if self.context_weights is not None:
                    mb["context_weights"] = self._to_tensor(
                        self.context_weights[:, mb_inds].transpose(1, 0, 2), device
                    )

            if self.weights is not None:
                mb["weights"] = self._to_tensor(
                    self.weights[:, mb_inds].transpose(1, 0, 2), device
                )

            yield mb

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
        return torch.as_tensor(arr.copy(), dtype=torch.float32, device=device)
