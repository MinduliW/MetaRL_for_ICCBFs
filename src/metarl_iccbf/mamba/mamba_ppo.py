"""Mamba2-backed RecurrentPPO.

Thin subclass of ``sb3_contrib.RecurrentPPO`` that accepts
:class:`Mamba2ActorCriticPolicy` as its policy class.  The only override is
``_setup_model`` which removes the ``isinstance(policy,
RecurrentActorCriticPolicy)`` check (our policy inherits from
``ActorCriticPolicy`` directly to avoid creating unused LSTM layers).
"""

from __future__ import annotations

from typing import ClassVar, cast

import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import get_schedule_fn

from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.buffers import (
    RecurrentDictRolloutBuffer,
    RecurrentRolloutBuffer,
)
from sb3_contrib.common.recurrent.type_aliases import RNNStates

from metarl_iccbf.mamba.mamba_policy import Mamba2ActorCriticPolicy


class Mamba2PPO(RecurrentPPO):
    """RecurrentPPO variant that uses a Mamba2 SSM instead of an LSTM."""

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MambaMlpPolicy": Mamba2ActorCriticPolicy,
    }

    def _setup_model(self) -> None:
        """Identical to ``RecurrentPPO._setup_model`` but accepts
        :class:`Mamba2ActorCriticPolicy` (which does not subclass
        ``RecurrentActorCriticPolicy``)."""

        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = (
            RecurrentDictRolloutBuffer
            if isinstance(self.observation_space, spaces.Dict)
            else RecurrentRolloutBuffer
        )

        policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,
        )
        self.policy: Mamba2ActorCriticPolicy = cast(Mamba2ActorCriticPolicy, policy.to(self.device))

        # Read buffer dimensions from the fake LSTM attrs on the policy
        lstm = self.policy.lstm_actor
        single_hidden_state_shape = (lstm.num_layers, self.n_envs, lstm.hidden_size)

        self._last_lstm_states = RNNStates(
            (
                th.zeros(single_hidden_state_shape, device=self.device),
                th.zeros(single_hidden_state_shape, device=self.device),
            ),
            (
                th.zeros(single_hidden_state_shape, device=self.device),
                th.zeros(single_hidden_state_shape, device=self.device),
            ),
        )

        hidden_state_buffer_shape = (
            self.n_steps,
            lstm.num_layers,
            self.n_envs,
            lstm.hidden_size,
        )

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            hidden_state_buffer_shape,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert (
                    self.clip_range_vf > 0
                ), "`clip_range_vf` must be positive, pass `None` to deactivate vf clipping"
            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)
