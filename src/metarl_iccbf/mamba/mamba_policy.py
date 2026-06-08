"""Mamba2-based recurrent actor-critic policy for RecurrentPPO.

Drop-in replacement for sb3_contrib's MlpLstmPolicy that uses a Mamba2 SSM
(from the ``mamba-ssm`` library) instead of an LSTM.  All RecurrentPPO training
infrastructure (rollout buffer, PPO loss, etc.) is reused without modification
by packing the Mamba2 states into the existing ``(h, c)`` tuple format.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    MlpExtractor,
)
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from mamba_ssm.modules.mamba2 import Mamba2
from sb3_contrib.common.recurrent.type_aliases import RNNStates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeLSTMAttrs:
    """Minimal stand-in so ``RecurrentPPO._setup_model`` can read
    ``.num_layers`` and ``.hidden_size`` from ``policy.lstm_actor``."""

    def __init__(self, num_layers: int, hidden_size: int):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # _process_sequence uses lstm.input_size for reshaping; we store it
        # here so existing code that might inspect it doesn't crash,
        # but our override never reads this attribute.
        self.input_size = hidden_size


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class Mamba2ActorCriticPolicy(ActorCriticPolicy):
    """Recurrent actor-critic policy backed by Mamba2 instead of LSTM.

    Designed to be used with ``Mamba2PPO`` (a thin subclass of
    ``RecurrentPPO``).  Accepts the same ``policy_kwargs`` as the stock
    ``RecurrentActorCriticPolicy`` plus additional Mamba2 hyper-parameters.

    Parameters
    ----------
    mamba_d_model : int
        Mamba2 model dimension (analogous to ``lstm_hidden_size``).  Must be a
        multiple of ``mamba_headdim``.
    mamba_d_state : int
        SSM state expansion factor.
    mamba_d_conv : int
        Causal-conv1d kernel width.
    mamba_expand : int
        Block expansion factor (``d_inner = d_model * expand``).
    mamba_headdim : int
        Per-head dimension.
    """

    # Make this class pass ``isinstance(..., RecurrentActorCriticPolicy)``
    # checks inside RecurrentPPO without importing the actual LSTM base class
    # at the *class* level.  We register the check via __init_subclass__ below.

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        # --- LSTM compat params (accepted for API compatibility) ---
        lstm_hidden_size: int = 64,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        lstm_kwargs: Optional[dict[str, Any]] = None,
        # --- Mamba2-specific ---
        mamba_d_model: int = 64,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_headdim: int = 64,
    ):
        # lstm_output_dim is read by _build_mlp_extractor (called inside
        # ActorCriticPolicy.__init__) to set the MLP extractor input size.
        self.lstm_output_dim = mamba_d_model

        # ---- grandparent init (builds features extractor + MLP + heads) ----
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

        # ---- store config ----
        self.shared_lstm = shared_lstm
        self.enable_critic_lstm = enable_critic_lstm
        self.mamba_d_model = mamba_d_model
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_headdim = mamba_headdim

        assert not (shared_lstm and enable_critic_lstm), (
            "Choose between shared recurrence, separate critic recurrence, or no critic recurrence."
        )
        assert not (shared_lstm and not self.share_features_extractor), (
            "If the features extractor is not shared, the recurrence cannot be shared."
        )

        # ---- derived dimensions ----
        d_inner = mamba_d_model * mamba_expand
        d_ssm = d_inner
        nheads = d_ssm // mamba_headdim
        ngroups = 1
        conv_dim = d_ssm + 2 * ngroups * mamba_d_state

        self._conv_state_shape = (conv_dim, mamba_d_conv)
        self._ssm_state_shape = (nheads, mamba_headdim, mamba_d_state)
        self._conv_flat = conv_dim * mamba_d_conv
        self._ssm_flat = nheads * mamba_headdim * mamba_d_state
        self._state_buffer_dim = max(self._conv_flat, self._ssm_flat)

        # ---- actor Mamba2 ----
        self.proj_actor = nn.Linear(self.features_dim, mamba_d_model)
        self.mamba_actor = Mamba2(
            d_model=mamba_d_model,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            headdim=mamba_headdim,
            ngroups=ngroups,
        )

        # ---- critic ----
        self.critic = None
        self.lstm_critic = None  # truthy sentinel checked by parent code

        if not (shared_lstm or enable_critic_lstm):
            self.critic = nn.Linear(self.features_dim, mamba_d_model)

        self.proj_critic: Optional[nn.Linear] = None
        self.mamba_critic: Optional[Mamba2] = None

        if enable_critic_lstm:
            self.proj_critic = nn.Linear(self.features_dim, mamba_d_model)
            self.mamba_critic = Mamba2(
                d_model=mamba_d_model,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                headdim=mamba_headdim,
                ngroups=ngroups,
            )
            self.lstm_critic = True  # truthy so parent checks pass

        # ---- compatibility shims for RecurrentPPO._setup_model ----
        self.lstm_hidden_state_shape = (1, 1, self._state_buffer_dim)
        self.lstm_actor = _FakeLSTMAttrs(
            num_layers=1, hidden_size=self._state_buffer_dim
        )

        # ---- rebuild optimizer with *all* new parameters ----
        # The parent class optimizer was created before the Mamba2 modules,
        # so we need to rebuild it to include the new parameters.
        self.optimizer = self.optimizer_class(
            self.parameters(), **self.optimizer_kwargs
        )

    # ------------------------------------------------------------------
    # MLP extractor override (same as RecurrentActorCriticPolicy)
    # ------------------------------------------------------------------

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MlpExtractor(
            self.lstm_output_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # State packing / unpacking
    # ------------------------------------------------------------------

    def _pack_states(
        self,
        conv_state: th.Tensor,
        ssm_state: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Flatten Mamba2 states into ``(h, c)`` buffer format.

        Returns two tensors of shape ``(1, batch, state_buffer_dim)``.
        """
        batch = conv_state.shape[0]
        device = conv_state.device
        dtype = conv_state.dtype

        h = th.zeros(1, batch, self._state_buffer_dim, device=device, dtype=dtype)
        c = th.zeros(1, batch, self._state_buffer_dim, device=device, dtype=dtype)

        h[0, :, : self._conv_flat] = conv_state.reshape(batch, -1)
        c[0, :, : self._ssm_flat] = ssm_state.reshape(batch, -1)
        return h, c

    def _unpack_states(
        self,
        packed: tuple[th.Tensor, th.Tensor],
        batch_size: int,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Reverse of :meth:`_pack_states`."""
        h, c = packed  # each (1, batch, buffer_dim)
        conv_flat = h[0, :batch_size, : self._conv_flat]
        ssm_flat = c[0, :batch_size, : self._ssm_flat]
        conv_state = conv_flat.reshape(batch_size, *self._conv_state_shape)
        ssm_state = ssm_flat.reshape(batch_size, *self._ssm_state_shape)
        return conv_state, ssm_state

    # ------------------------------------------------------------------
    # Sequence processing (replaces _process_sequence)
    # ------------------------------------------------------------------

    def _process_sequence_mamba(
        self,
        features: th.Tensor,
        packed_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        mamba: Mamba2,
        proj: nn.Linear,
    ) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        """Step-by-step Mamba2 processing with episode-boundary resets.

        Mirrors the role of ``RecurrentActorCriticPolicy._process_sequence``
        but uses ``mamba.step()`` for each timestep.
        """
        n_seq = packed_states[0].shape[1]

        # Unpack running states
        conv_state, ssm_state = self._unpack_states(packed_states, n_seq)

        # Reshape to (max_len, n_seq, feat_dim)
        features_seq = features.reshape((n_seq, -1, features.shape[-1])).swapaxes(0, 1)
        episode_starts_seq = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

        outputs: list[th.Tensor] = []
        for t in range(features_seq.shape[0]):
            feat_t = features_seq[t]          # (n_seq, feat_dim)
            ep_start_t = episode_starts_seq[t]  # (n_seq,)

            # Reset states on episode boundaries
            mask = (1.0 - ep_start_t)
            conv_state = mask.view(n_seq, 1, 1) * conv_state
            ssm_state = mask.view(n_seq, 1, 1, 1) * ssm_state

            # Project features → d_model and call mamba.step
            projected = proj(feat_t)  # (n_seq, d_model)
            out, conv_state, ssm_state = mamba.step(
                projected.unsqueeze(1), conv_state, ssm_state
            )
            outputs.append(out.squeeze(1))  # (n_seq, d_model)

        # (max_len, n_seq, d_model) → (n_seq * max_len, d_model)
        output = th.stack(outputs, dim=0)
        output = th.flatten(output.transpose(0, 1), start_dim=0, end_dim=1)

        packed_out = self._pack_states(conv_state, ssm_state)
        return output, packed_out

    def _process_sequence_mamba_scan(
        self,
        features: th.Tensor,
        packed_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        mamba: Mamba2,
        proj: nn.Linear,
    ) -> th.Tensor:
        """Parallel-scan Mamba2 processing for training.

        Uses ``mamba.forward()`` with ``seq_idx`` to handle episode boundaries
        in a single fused kernel call, instead of stepping through each timestep.
        Does not return final states (not needed during PPO training).
        """
        n_seq = packed_states[0].shape[1]
        max_len = features.shape[0] // n_seq

        # Reshape to (n_seq, max_len, feat_dim)
        features_batch = features.reshape(n_seq, max_len, -1)
        episode_starts_batch = episode_starts.reshape(n_seq, max_len)

        # Build seq_idx for episode-boundary resets.  Only pass it to the
        # kernel when there are actual mid-sequence resets, because the
        # causal_conv1d seq_idx code path has a CUDA bug that causes
        # intermittent illegal-memory-access errors (causal_conv1d 1.6.0).
        episode_starts_batch = episode_starts_batch.clone()
        episode_starts_batch[:, 0] = 1.0
        seq_idx = episode_starts_batch.cumsum(dim=1).int() - 1  # (n_seq, max_len)
        has_boundaries = seq_idx.max() > 0

        # Project features → d_model, then run parallel scan
        projected = proj(features_batch)           # (n_seq, max_len, d_model)
        if has_boundaries:
            out = mamba(projected, seq_idx=seq_idx)
        else:
            out = mamba(projected)

        # Flatten back to (n_seq * max_len, d_model)
        return out.reshape(n_seq * max_len, -1)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        obs: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, RNNStates]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features

        latent_pi, states_pi = self._process_sequence_mamba(
            pi_features, lstm_states.pi, episode_starts,
            self.mamba_actor, self.proj_actor,
        )

        if self.mamba_critic is not None:
            latent_vf, states_vf = self._process_sequence_mamba(
                vf_features, lstm_states.vf, episode_starts,
                self.mamba_critic, self.proj_critic,
            )
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
            states_vf = (states_pi[0].detach(), states_pi[1].detach())
        else:
            latent_vf = self.critic(vf_features)
            states_vf = states_pi

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob, RNNStates(states_pi, states_vf)

    # ------------------------------------------------------------------
    # evaluate_actions  (called during PPO training)
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features

        latent_pi = self._process_sequence_mamba_scan(
            pi_features, lstm_states.pi, episode_starts,
            self.mamba_actor, self.proj_actor,
        )
        if self.mamba_critic is not None:
            latent_vf = self._process_sequence_mamba_scan(
                vf_features, lstm_states.vf, episode_starts,
                self.mamba_critic, self.proj_critic,
            )
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(vf_features)

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    # ------------------------------------------------------------------
    # get_distribution / predict_values / _predict
    # ------------------------------------------------------------------

    def get_distribution(
        self,
        obs: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> tuple[Distribution, tuple[th.Tensor, th.Tensor]]:
        features = super(ActorCriticPolicy, self).extract_features(
            obs, self.pi_features_extractor
        )
        latent_pi, lstm_states = self._process_sequence_mamba(
            features, lstm_states, episode_starts,
            self.mamba_actor, self.proj_actor,
        )
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        return self._get_action_dist_from_latent(latent_pi), lstm_states

    def predict_values(
        self,
        obs: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> th.Tensor:
        features = super(ActorCriticPolicy, self).extract_features(
            obs, self.vf_features_extractor
        )
        if self.mamba_critic is not None:
            latent_vf, _ = self._process_sequence_mamba(
                features, lstm_states, episode_starts,
                self.mamba_critic, self.proj_critic,
            )
        elif self.shared_lstm:
            latent_pi, _ = self._process_sequence_mamba(
                features, lstm_states, episode_starts,
                self.mamba_actor, self.proj_actor,
            )
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(features)

        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        return self.value_net(latent_vf)

    def _predict(
        self,
        observation: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> tuple[th.Tensor, tuple[th.Tensor, ...]]:
        distribution, lstm_states = self.get_distribution(
            observation, lstm_states, episode_starts
        )
        return distribution.get_actions(deterministic=deterministic), lstm_states

    # ------------------------------------------------------------------
    # predict  (user-facing, numpy ↔ torch conversion)
    # ------------------------------------------------------------------

    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, np.ndarray]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        if isinstance(observation, dict):
            n_envs = observation[next(iter(observation.keys()))].shape[0]
        else:
            n_envs = observation.shape[0]

        if state is None:
            state_array = np.concatenate(
                [np.zeros(self.lstm_hidden_state_shape) for _ in range(n_envs)],
                axis=1,
            )
            state = (state_array, state_array)

        if episode_start is None:
            episode_start = np.array([False for _ in range(n_envs)])

        with th.no_grad():
            states = (
                th.tensor(state[0], dtype=th.float32, device=self.device),
                th.tensor(state[1], dtype=th.float32, device=self.device),
            )
            episode_starts = th.tensor(
                episode_start, dtype=th.float32, device=self.device
            )
            actions, states = self._predict(
                observation,
                lstm_states=states,
                episode_starts=episode_starts,
                deterministic=deterministic,
            )
            states = (states[0].cpu().numpy(), states[1].cpu().numpy())

        actions = actions.cpu().numpy()

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                actions = self.unscale_action(actions)
            else:
                actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, states
