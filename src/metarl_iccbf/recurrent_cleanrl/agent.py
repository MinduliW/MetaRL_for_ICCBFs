"""Recurrent actor-critics for CleanRL-style PPO.

Provides three backends with identical external interfaces:
- ``Mamba2ActorCritic`` — Mamba2 SSM recurrence (parallel scan for training)
- ``GRUActorCritic``    — standard GRU recurrence (sequential loop)
- ``LSTMActorCritic``   — standard LSTM recurrence (sequential loop)

All expose:
- ``step()``               — single-timestep inference for rollout collection
- ``evaluate_actions()``   — batched sequence evaluation for PPO training
- ``get_value()``          — single-step value prediction (GAE bootstrap)
- ``deterministic_step()`` — like ``step()`` but returns the action mean
- ``zero_env_states()``    — zero states for a specific environment index
- ``initial_states()``     — zero-initialized recurrent states
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Normal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_mlp(
    input_dim: int,
    hidden_sizes: list[int],
    activation_fn: type[nn.Module],
) -> nn.Sequential:
    """Build a simple MLP (no output activation)."""
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(activation_fn())
        prev = h
    return nn.Sequential(*layers)


def _ortho_init(module: nn.Module, gain: float = 1.0) -> None:
    """Apply orthogonal initialization to a Linear layer."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# Mamba2 Actor-Critic
# ---------------------------------------------------------------------------

class Mamba2ActorCritic(nn.Module):
    """Mamba2-backed actor-critic with separate actor/critic recurrence.

    Parameters
    ----------
    obs_dim : int
        Observation dimensionality (flattened).  When ``morl_arch="concat"``
        this includes the w dimensions; when ``morl_arch="multi_body"`` this
        is the raw obs dim (w is passed separately to ``step()`` etc.).
    act_dim : int
        Action dimensionality.
    mamba_d_model, mamba_d_state, mamba_d_conv, mamba_expand, mamba_headdim :
        Mamba2 hyper-parameters.
    net_arch : dict
        ``{"pi": [h1, h2, ...], "vf": [h1, h2, ...]}`` for the MLP heads.
    activation_fn : type[nn.Module]
        Activation class (e.g. ``nn.Tanh``).
    ortho_init : bool
        Apply SB3-style orthogonal initialization.
    log_std_init : float
        Initial value for the learnable log standard deviation.
    morl_arch : str
        ``"concat"`` (default) or ``"multi_body"`` (MOPPO paper eq. 10).
    n_objectives : int
        K — number of preference dimensions.  Only used when
        ``morl_arch="multi_body"``.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        *,
        mamba_d_model: int = 64,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_headdim: int = 64,
        net_arch: Optional[dict[str, list[int]]] = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        log_std_init: float = -1.6094,
        morl_arch: str = "concat",
        n_objectives: int = 2,
    ):
        super().__init__()

        from mamba_ssm.modules.mamba2 import Mamba2

        assert morl_arch in ("concat", "multi_body"), (
            f"morl_arch must be 'concat' or 'multi_body', got {morl_arch!r}"
        )

        if net_arch is None:
            net_arch = {"pi": [64, 64, 64], "vf": [64, 64, 64]}

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.mamba_d_model = mamba_d_model
        self.morl_arch = morl_arch
        self.n_objectives = n_objectives

        # ---- Derived Mamba2 dimensions (same as mamba_policy.py:150-161) ----
        d_inner = mamba_d_model * mamba_expand
        d_ssm = d_inner
        nheads = d_ssm // mamba_headdim
        ngroups = 1
        conv_dim = d_ssm + 2 * ngroups * mamba_d_state

        self._conv_state_shape = (conv_dim, mamba_d_conv)
        self._ssm_state_shape = (nheads, mamba_headdim, mamba_d_state)

        # ---- Input projections (architecture-dependent) ----
        if morl_arch == "multi_body":
            # K separate Linear+ReLU projections; w is the interpolation weight
            self.body_proj_actor = nn.ModuleList(
                [nn.Linear(obs_dim, mamba_d_model) for _ in range(n_objectives)]
            )
            self.body_proj_critic = nn.ModuleList(
                [nn.Linear(obs_dim, mamba_d_model) for _ in range(n_objectives)]
            )
            self.proj_actor = None  # type: ignore[assignment]
            self.proj_critic = None  # type: ignore[assignment]
        else:
            self.proj_actor = nn.Linear(obs_dim, mamba_d_model)
            self.proj_critic = nn.Linear(obs_dim, mamba_d_model)
            self.body_proj_actor = None  # type: ignore[assignment]
            self.body_proj_critic = None  # type: ignore[assignment]

        # ---- Actor recurrence ----
        self.mamba_actor = Mamba2(
            d_model=mamba_d_model,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            headdim=mamba_headdim,
            ngroups=ngroups,
        )

        # ---- Critic recurrence (separate) ----
        self.mamba_critic = Mamba2(
            d_model=mamba_d_model,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            headdim=mamba_headdim,
            ngroups=ngroups,
        )

        # ---- MLP heads ----
        self.mlp_actor = _build_mlp(mamba_d_model, net_arch["pi"], activation_fn)
        self.mlp_critic = _build_mlp(mamba_d_model, net_arch["vf"], activation_fn)

        pi_out_dim = net_arch["pi"][-1] if net_arch["pi"] else mamba_d_model
        vf_out_dim = net_arch["vf"][-1] if net_arch["vf"] else mamba_d_model

        self.action_mean = nn.Linear(pi_out_dim, act_dim)
        self.value_head = nn.Linear(vf_out_dim, 1)

        # ---- Learnable log_std ----
        self.log_std = nn.Parameter(torch.full((act_dim,), log_std_init))

        # ---- Orthogonal initialization ----
        if ortho_init:
            gain = math.sqrt(2.0)
            if morl_arch == "multi_body":
                for body in [self.body_proj_actor, self.body_proj_critic]:
                    for m in body:
                        _ortho_init(m, gain=1.0)
            else:
                for m in [self.proj_actor, self.proj_critic]:
                    _ortho_init(m, gain=1.0)
            for m in self.mlp_actor.modules():
                _ortho_init(m, gain=gain)
            for m in self.mlp_critic.modules():
                _ortho_init(m, gain=gain)
            _ortho_init(self.action_mean, gain=0.01)
            _ortho_init(self.value_head, gain=1.0)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def conv_state_shape(self) -> tuple[int, ...]:
        return self._conv_state_shape

    @property
    def ssm_state_shape(self) -> tuple[int, ...]:
        return self._ssm_state_shape

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def initial_states(
        self, n_envs: int, device: torch.device
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        """Return zero-initialized ``(actor_states, critic_states)``.

        Each is ``(conv_state, ssm_state)`` in native Mamba2 shapes.
        """
        def _zeros(shape):
            return torch.zeros(n_envs, *shape, device=device)

        actor = (_zeros(self._conv_state_shape), _zeros(self._ssm_state_shape))
        critic = (_zeros(self._conv_state_shape), _zeros(self._ssm_state_shape))
        return actor, critic

    @staticmethod
    def zero_env_states(states: tuple[torch.Tensor, torch.Tensor], env_idx: int) -> None:
        """Zero-out recurrent states for a single environment."""
        states[0][env_idx].zero_()
        states[1][env_idx].zero_()

    # ------------------------------------------------------------------
    # Single-step inference (rollout collection)
    # ------------------------------------------------------------------

    def step(
        self,
        obs: torch.Tensor,
        actor_states: tuple[torch.Tensor, torch.Tensor],
        critic_states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
        w: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        """One-step inference using ``mamba.step()``.

        Parameters
        ----------
        obs : (n_envs, obs_dim)
            When ``morl_arch="multi_body"``, raw obs (no w appended).
        actor_states, critic_states : (conv_state, ssm_state)
        episode_starts : (n_envs,)   1.0 when a new episode just started
        w : (n_envs, K)  preference weights — required when morl_arch="multi_body"

        Returns
        -------
        action, value, log_prob, new_actor_states, new_critic_states
        """
        actor_states = self._reset_states(actor_states, episode_starts)
        critic_states = self._reset_states(critic_states, episode_starts)

        # --- Actor ---
        if self.morl_arch == "multi_body":
            a_proj = self._multi_body_proj(obs, w, self.body_proj_actor)
        else:
            a_proj = self.proj_actor(obs)                      # (B, d_model)
        a_out, a_conv, a_ssm = self.mamba_actor.step(
            a_proj.unsqueeze(1), actor_states[0], actor_states[1],
        )
        a_out = a_out.squeeze(1)                               # (B, d_model)
        a_latent = self.mlp_actor(a_out)
        mean = self.action_mean(a_latent)
        std = self.log_std.clamp(min=-4.0).exp().expand_as(mean)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        # --- Critic ---
        if self.morl_arch == "multi_body":
            c_proj = self._multi_body_proj(obs, w, self.body_proj_critic)
        else:
            c_proj = self.proj_critic(obs)
        c_out, c_conv, c_ssm = self.mamba_critic.step(
            c_proj.unsqueeze(1), critic_states[0], critic_states[1],
        )
        c_out = c_out.squeeze(1)
        c_latent = self.mlp_critic(c_out)
        value = self.value_head(c_latent).squeeze(-1)          # (B,)

        return (
            action,
            value,
            log_prob,
            (a_conv, a_ssm),
            (c_conv, c_ssm),
        )

    # ------------------------------------------------------------------
    # Deterministic single-step (evaluation / predict)
    # ------------------------------------------------------------------

    def deterministic_step(
        self,
        obs: torch.Tensor,
        actor_states: tuple[torch.Tensor, torch.Tensor],
        critic_states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
        w: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        """Like ``step()`` but returns the action mean (no sampling).

        Parameters
        ----------
        w : (n_envs, K)  required when morl_arch="multi_body"

        Returns
        -------
        action_mean, new_actor_states, new_critic_states
        """
        actor_states = self._reset_states(actor_states, episode_starts)
        critic_states = self._reset_states(critic_states, episode_starts)

        # --- Actor ---
        if self.morl_arch == "multi_body":
            a_proj = self._multi_body_proj(obs, w, self.body_proj_actor)
        else:
            a_proj = self.proj_actor(obs)
        a_out, a_conv, a_ssm = self.mamba_actor.step(
            a_proj.unsqueeze(1), actor_states[0], actor_states[1],
        )
        a_out = a_out.squeeze(1)
        a_latent = self.mlp_actor(a_out)
        action = self.action_mean(a_latent)

        # --- Critic (keep states in sync) ---
        if self.morl_arch == "multi_body":
            c_proj = self._multi_body_proj(obs, w, self.body_proj_critic)
        else:
            c_proj = self.proj_critic(obs)
        _, c_conv, c_ssm = self.mamba_critic.step(
            c_proj.unsqueeze(1), critic_states[0], critic_states[1],
        )

        return action, (a_conv, a_ssm), (c_conv, c_ssm)

    # ------------------------------------------------------------------
    # Batch sequence processing (PPO training)
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        episode_starts: torch.Tensor,
        context_obs: Optional[torch.Tensor] = None,
        context_episode_starts: Optional[torch.Tensor] = None,
        w: Optional[torch.Tensor] = None,
        context_w: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions on batched sequences using parallel scan.

        Parameters
        ----------
        obs : (B, T, obs_dim)
            When ``morl_arch="multi_body"``, raw obs (no w appended).
        actions : (B, T, act_dim)
        episode_starts : (B, T)
        context_obs : (B, K, obs_dim)  optional burn-in context
        context_episode_starts : (B, K)
        w : (B, T, n_objectives)  preference weights (multi_body only)
        context_w : (B, K, n_objectives)  burn-in weights (multi_body only)

        Returns
        -------
        values : (B, T)
        log_probs : (B, T)
        entropy : (B, T)
        """
        # --- Prepend burn-in context ---
        if context_obs is not None:
            K = context_obs.shape[1]
            full_obs = torch.cat([context_obs, obs], dim=1)
            full_ep = torch.cat([context_episode_starts, episode_starts], dim=1)
            if self.morl_arch == "multi_body" and w is not None:
                _ctx_w = context_w if context_w is not None else torch.zeros(
                    obs.shape[0], K, self.n_objectives, device=obs.device, dtype=obs.dtype
                )
                full_w = torch.cat([_ctx_w, w], dim=1)
            else:
                full_w = None
        else:
            K = 0
            full_obs = obs
            full_ep = episode_starts
            full_w = w if self.morl_arch == "multi_body" else None

        # --- Actor parallel scan ---
        a_out = self._scan(
            full_obs, full_ep, self.mamba_actor,
            proj=self.proj_actor if self.morl_arch == "concat" else None,
            body_list=self.body_proj_actor if self.morl_arch == "multi_body" else None,
            w=full_w,
        )
        if K > 0:
            a_out = a_out[:, K:, :]                            # discard burn-in
        a_latent = self.mlp_actor(a_out)
        mean = self.action_mean(a_latent)                      # (B, T, act_dim)
        std = self.log_std.clamp(min=-4.0).exp().expand_as(mean)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)         # (B, T)
        entropy = dist.entropy().sum(dim=-1)                   # (B, T)

        # --- Critic parallel scan ---
        c_out = self._scan(
            full_obs, full_ep, self.mamba_critic,
            proj=self.proj_critic if self.morl_arch == "concat" else None,
            body_list=self.body_proj_critic if self.morl_arch == "multi_body" else None,
            w=full_w,
        )
        if K > 0:
            c_out = c_out[:, K:, :]
        c_latent = self.mlp_critic(c_out)
        values = self.value_head(c_latent).squeeze(-1)         # (B, T)

        return values, log_probs, entropy

    # ------------------------------------------------------------------
    # Single-step value prediction (for GAE bootstrap)
    # ------------------------------------------------------------------

    def get_value(
        self,
        obs: torch.Tensor,
        critic_states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
        w: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Single-step value prediction.

        Parameters
        ----------
        w : (B, K)  required when morl_arch="multi_body"

        Returns
        -------
        value : (B,)
        new_critic_states : (conv_state, ssm_state)
        """
        critic_states = self._reset_states(critic_states, episode_starts)
        if self.morl_arch == "multi_body":
            c_proj = self._multi_body_proj(obs, w, self.body_proj_critic)
        else:
            c_proj = self.proj_critic(obs)
        c_out, c_conv, c_ssm = self.mamba_critic.step(
            c_proj.unsqueeze(1), critic_states[0], critic_states[1],
        )
        c_out = c_out.squeeze(1)
        c_latent = self.mlp_critic(c_out)
        value = self.value_head(c_latent).squeeze(-1)
        return value, (c_conv, c_ssm)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reset_states(
        states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero-out states where a new episode started."""
        mask = (1.0 - episode_starts)
        conv = states[0] * mask.view(-1, 1, 1)
        ssm = states[1] * mask.view(-1, 1, 1, 1)
        return conv, ssm

    def _multi_body_proj(
        self,
        obs: torch.Tensor,
        w: torch.Tensor,
        body_list: nn.ModuleList,
    ) -> torch.Tensor:
        """Weighted sum of K per-objective projections (MOPPO paper eq. 10).

        Parameters
        ----------
        obs : (..., obs_dim)
        w   : (..., K)
        body_list : ModuleList of K Linear layers
        """
        return sum(
            w[..., i : i + 1] * torch.relu(body_list[i](obs))
            for i in range(len(body_list))
        )  # type: ignore[return-value]

    def _scan(
        self,
        obs: torch.Tensor,
        episode_starts: torch.Tensor,
        mamba: nn.Module,
        proj: Optional[nn.Linear] = None,
        body_list: Optional[nn.ModuleList] = None,
        w: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run ``mamba.forward()`` with ``seq_idx`` for episode boundaries.

        Parameters
        ----------
        obs : (B, L, obs_dim)   where L = K + T (or just T if no context)
        episode_starts : (B, L)
        proj : Linear for "concat" mode (mutually exclusive with body_list)
        body_list : ModuleList for "multi_body" mode
        w : (B, L, n_objectives)  required when body_list is not None

        Returns
        -------
        out : (B, L, d_model)
        """
        ep = episode_starts.clone()
        ep[:, 0] = 1.0                                        # mandatory reset

        seq_idx = ep.cumsum(dim=1).int() - 1                   # (B, L)
        has_boundaries = seq_idx.max() > 0

        if body_list is not None:
            projected = self._multi_body_proj(obs, w, body_list)
        else:
            projected = proj(obs)                               # (B, L, d_model)

        # Only pass seq_idx when there are actual mid-sequence resets,
        # because causal_conv1d v1.6.0 has a CUDA bug in the seq_idx
        # code path that causes intermittent illegal-memory-access errors.
        if has_boundaries:
            out = mamba(projected, seq_idx=seq_idx)
        else:
            out = mamba(projected)

        return out                                              # (B, L, d_model)


# ---------------------------------------------------------------------------
# GRU Actor-Critic
# ---------------------------------------------------------------------------

class GRUActorCritic(nn.Module):
    """GRU-backed actor-critic with separate actor/critic recurrence.

    Drop-in replacement for ``Mamba2ActorCritic`` — same external interface.

    Parameters
    ----------
    obs_dim : int
        Observation dimensionality (flattened).
    act_dim : int
        Action dimensionality.
    hidden_size : int
        GRU hidden state size.
    net_arch : dict
        ``{"pi": [h1, h2, ...], "vf": [h1, h2, ...]}`` for the MLP heads.
    activation_fn : type[nn.Module]
        Activation class (e.g. ``nn.Tanh``).
    ortho_init : bool
        Apply SB3-style orthogonal initialization.
    log_std_init : float
        Initial value for the learnable log standard deviation.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        *,
        hidden_size: int = 64,
        net_arch: Optional[dict[str, list[int]]] = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        log_std_init: float = -1.6094,
    ):
        super().__init__()

        if net_arch is None:
            net_arch = {"pi": [64, 64, 64], "vf": [64, 64, 64]}

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size

        # ---- Actor recurrence ----
        self.proj_actor = nn.Linear(obs_dim, hidden_size)
        self.gru_actor = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # ---- Critic recurrence (separate) ----
        self.proj_critic = nn.Linear(obs_dim, hidden_size)
        self.gru_critic = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # ---- MLP heads ----
        self.mlp_actor = _build_mlp(hidden_size, net_arch["pi"], activation_fn)
        self.mlp_critic = _build_mlp(hidden_size, net_arch["vf"], activation_fn)

        pi_out_dim = net_arch["pi"][-1] if net_arch["pi"] else hidden_size
        vf_out_dim = net_arch["vf"][-1] if net_arch["vf"] else hidden_size

        self.action_mean = nn.Linear(pi_out_dim, act_dim)
        self.value_head = nn.Linear(vf_out_dim, 1)

        # ---- Learnable log_std ----
        self.log_std = nn.Parameter(torch.full((act_dim,), log_std_init))

        # ---- Orthogonal initialization ----
        if ortho_init:
            gain = math.sqrt(2.0)
            for m in [self.proj_actor, self.proj_critic]:
                _ortho_init(m, gain=1.0)
            for m in self.mlp_actor.modules():
                _ortho_init(m, gain=gain)
            for m in self.mlp_critic.modules():
                _ortho_init(m, gain=gain)
            _ortho_init(self.action_mean, gain=0.01)
            _ortho_init(self.value_head, gain=1.0)

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def initial_states(
        self, n_envs: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-initialized ``(actor_h, critic_h)``.

        Each is shape ``(1, n_envs, hidden_size)`` — nn.GRU convention.
        """
        actor_h = torch.zeros(1, n_envs, self.hidden_size, device=device)
        critic_h = torch.zeros(1, n_envs, self.hidden_size, device=device)
        return actor_h, critic_h

    @staticmethod
    def zero_env_states(states: torch.Tensor, env_idx: int) -> None:
        """Zero-out recurrent states for a single environment."""
        states[:, env_idx, :].zero_()

    # ------------------------------------------------------------------
    # Single-step inference (rollout collection)
    # ------------------------------------------------------------------

    def step(
        self,
        obs: torch.Tensor,
        actor_states: torch.Tensor,
        critic_states: torch.Tensor,
        episode_starts: torch.Tensor,
        w: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """One-step inference using GRU.

        Parameters
        ----------
        obs : (n_envs, obs_dim)
        actor_states, critic_states : (1, n_envs, hidden_size)
        episode_starts : (n_envs,)   1.0 when a new episode just started

        Returns
        -------
        action, value, log_prob, new_actor_states, new_critic_states
        """
        actor_states = self._reset_states(actor_states, episode_starts)
        critic_states = self._reset_states(critic_states, episode_starts)

        # --- Actor ---
        a_proj = self.proj_actor(obs).unsqueeze(1)             # (B, 1, hidden)
        a_out, actor_states = self.gru_actor(a_proj, actor_states)
        a_out = a_out.squeeze(1)                               # (B, hidden)
        a_latent = self.mlp_actor(a_out)
        mean = self.action_mean(a_latent)
        std = self.log_std.clamp(min=-4.0).exp().expand_as(mean)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        # --- Critic ---
        c_proj = self.proj_critic(obs).unsqueeze(1)
        c_out, critic_states = self.gru_critic(c_proj, critic_states)
        c_out = c_out.squeeze(1)
        c_latent = self.mlp_critic(c_out)
        value = self.value_head(c_latent).squeeze(-1)          # (B,)

        return action, value, log_prob, actor_states, critic_states

    # ------------------------------------------------------------------
    # Deterministic single-step (evaluation / predict)
    # ------------------------------------------------------------------

    def deterministic_step(
        self,
        obs: torch.Tensor,
        actor_states: torch.Tensor,
        critic_states: torch.Tensor,
        episode_starts: torch.Tensor,
        w: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Like ``step()`` but returns the action mean (no sampling).

        Returns
        -------
        action_mean, new_actor_states, new_critic_states
        """
        actor_states = self._reset_states(actor_states, episode_starts)
        critic_states = self._reset_states(critic_states, episode_starts)

        # --- Actor ---
        a_proj = self.proj_actor(obs).unsqueeze(1)
        a_out, actor_states = self.gru_actor(a_proj, actor_states)
        a_out = a_out.squeeze(1)
        a_latent = self.mlp_actor(a_out)
        action = self.action_mean(a_latent)

        # --- Critic (keep states in sync) ---
        c_proj = self.proj_critic(obs).unsqueeze(1)
        _, critic_states = self.gru_critic(c_proj, critic_states)

        return action, actor_states, critic_states

    # ------------------------------------------------------------------
    # Batch sequence processing (PPO training)
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        episode_starts: torch.Tensor,
        context_obs: Optional[torch.Tensor] = None,
        context_episode_starts: Optional[torch.Tensor] = None,
        w: Optional[torch.Tensor] = None,
        context_w: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions on batched sequences.

        Parameters
        ----------
        obs : (B, T, obs_dim)
        actions : (B, T, act_dim)
        episode_starts : (B, T)
        context_obs : (B, K, obs_dim)  optional burn-in context
        context_episode_starts : (B, K)

        Returns
        -------
        values : (B, T)
        log_probs : (B, T)
        entropy : (B, T)
        """
        # --- Prepend burn-in context ---
        if context_obs is not None:
            K = context_obs.shape[1]
            full_obs = torch.cat([context_obs, obs], dim=1)
            full_ep = torch.cat([context_episode_starts, episode_starts], dim=1)
        else:
            K = 0
            full_obs = obs
            full_ep = episode_starts

        # --- Actor ---
        a_out = self._scan(full_obs, full_ep, self.gru_actor, self.proj_actor)
        if K > 0:
            a_out = a_out[:, K:, :]
        a_latent = self.mlp_actor(a_out)
        mean = self.action_mean(a_latent)                      # (B, T, act_dim)
        std = self.log_std.clamp(min=-4.0).exp().expand_as(mean)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)         # (B, T)
        entropy = dist.entropy().sum(dim=-1)                   # (B, T)

        # --- Critic ---
        c_out = self._scan(full_obs, full_ep, self.gru_critic, self.proj_critic)
        if K > 0:
            c_out = c_out[:, K:, :]
        c_latent = self.mlp_critic(c_out)
        values = self.value_head(c_latent).squeeze(-1)         # (B, T)

        return values, log_probs, entropy

    # ------------------------------------------------------------------
    # Single-step value prediction (for GAE bootstrap)
    # ------------------------------------------------------------------

    def get_value(
        self,
        obs: torch.Tensor,
        critic_states: torch.Tensor,
        episode_starts: torch.Tensor,
        w: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-step value prediction.

        Returns
        -------
        value : (B,)
        new_critic_states : (1, B, hidden_size)
        """
        critic_states = self._reset_states(critic_states, episode_starts)
        c_proj = self.proj_critic(obs).unsqueeze(1)
        c_out, critic_states = self.gru_critic(c_proj, critic_states)
        c_out = c_out.squeeze(1)
        c_latent = self.mlp_critic(c_out)
        value = self.value_head(c_latent).squeeze(-1)
        return value, critic_states

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reset_states(
        states: torch.Tensor,
        episode_starts: torch.Tensor,
    ) -> torch.Tensor:
        """Zero-out hidden states where a new episode started.

        Parameters
        ----------
        states : (1, B, hidden_size)
        episode_starts : (B,)
        """
        mask = (1.0 - episode_starts).unsqueeze(0).unsqueeze(-1)  # (1, B, 1)
        return states * mask

    def _scan(
        self,
        obs: torch.Tensor,
        episode_starts: torch.Tensor,
        gru: nn.GRU,
        proj: nn.Linear,
    ) -> torch.Tensor:
        """Run GRU over a sequence, resetting hidden state at episode boundaries.

        Parameters
        ----------
        obs : (B, L, obs_dim)
        episode_starts : (B, L)

        Returns
        -------
        out : (B, L, hidden_size)
        """
        B, T, _ = obs.shape
        projected = proj(obs)                                   # (B, T, hidden)
        h = torch.zeros(1, B, self.hidden_size, device=obs.device)
        outputs = []
        for t in range(T):
            mask = (1.0 - episode_starts[:, t]).unsqueeze(0).unsqueeze(-1)
            h = h * mask
            out_t, h = gru(projected[:, t : t + 1, :], h)
            outputs.append(out_t.squeeze(1))
        return torch.stack(outputs, dim=1)                      # (B, T, hidden)


# ---------------------------------------------------------------------------
# LSTM Actor-Critic
# ---------------------------------------------------------------------------

class LSTMActorCritic(nn.Module):
    """LSTM-backed actor-critic with separate actor/critic recurrence.

    Drop-in replacement for ``GRUActorCritic`` — same external interface,
    except hidden states are ``(h, c)`` tuples instead of single tensors.

    Parameters
    ----------
    obs_dim : int
        Observation dimensionality (flattened).
    act_dim : int
        Action dimensionality.
    hidden_size : int
        LSTM hidden state size.
    net_arch : dict
        ``{"pi": [h1, h2, ...], "vf": [h1, h2, ...]}`` for the MLP heads.
    activation_fn : type[nn.Module]
        Activation class (e.g. ``nn.Tanh``).
    ortho_init : bool
        Apply SB3-style orthogonal initialization.
    log_std_init : float
        Initial value for the learnable log standard deviation.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        *,
        hidden_size: int = 64,
        net_arch: Optional[dict[str, list[int]]] = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        log_std_init: float = -1.6094,
    ):
        super().__init__()

        if net_arch is None:
            net_arch = {"pi": [64, 64, 64], "vf": [64, 64, 64]}

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size

        # ---- Actor recurrence ----
        self.proj_actor = nn.Linear(obs_dim, hidden_size)
        self.lstm_actor = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # ---- Critic recurrence (separate) ----
        self.proj_critic = nn.Linear(obs_dim, hidden_size)
        self.lstm_critic = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # ---- MLP heads ----
        self.mlp_actor = _build_mlp(hidden_size, net_arch["pi"], activation_fn)
        self.mlp_critic = _build_mlp(hidden_size, net_arch["vf"], activation_fn)

        pi_out_dim = net_arch["pi"][-1] if net_arch["pi"] else hidden_size
        vf_out_dim = net_arch["vf"][-1] if net_arch["vf"] else hidden_size

        self.action_mean = nn.Linear(pi_out_dim, act_dim)
        self.value_head = nn.Linear(vf_out_dim, 1)

        # ---- Learnable log_std ----
        self.log_std = nn.Parameter(torch.full((act_dim,), log_std_init))

        # ---- Orthogonal initialization ----
        if ortho_init:
            gain = math.sqrt(2.0)
            for m in [self.proj_actor, self.proj_critic]:
                _ortho_init(m, gain=1.0)
            for m in self.mlp_actor.modules():
                _ortho_init(m, gain=gain)
            for m in self.mlp_critic.modules():
                _ortho_init(m, gain=gain)
            _ortho_init(self.action_mean, gain=0.01)
            _ortho_init(self.value_head, gain=1.0)

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def initial_states(
        self, n_envs: int, device: torch.device
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """Return zero-initialized ``(actor_states, critic_states)``.

        Each is ``(h, c)`` where both have shape ``(1, n_envs, hidden_size)``.
        """
        actor_h = torch.zeros(1, n_envs, self.hidden_size, device=device)
        actor_c = torch.zeros(1, n_envs, self.hidden_size, device=device)
        critic_h = torch.zeros(1, n_envs, self.hidden_size, device=device)
        critic_c = torch.zeros(1, n_envs, self.hidden_size, device=device)
        return (actor_h, actor_c), (critic_h, critic_c)

    @staticmethod
    def zero_env_states(
        states: tuple[torch.Tensor, torch.Tensor], env_idx: int
    ) -> None:
        """Zero-out recurrent states for a single environment."""
        h, c = states
        h[:, env_idx, :].zero_()
        c[:, env_idx, :].zero_()

    # ------------------------------------------------------------------
    # Single-step inference (rollout collection)
    # ------------------------------------------------------------------

    def step(
        self,
        obs: torch.Tensor,
        actor_states: tuple[torch.Tensor, torch.Tensor],
        critic_states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
        w: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        """One-step inference using LSTM.

        Parameters
        ----------
        obs : (n_envs, obs_dim)
        actor_states, critic_states : (h, c) each (1, n_envs, hidden_size)
        episode_starts : (n_envs,)   1.0 when a new episode just started

        Returns
        -------
        action, value, log_prob, new_actor_states, new_critic_states
        """
        actor_states = self._reset_states(actor_states, episode_starts)
        critic_states = self._reset_states(critic_states, episode_starts)

        # --- Actor ---
        a_proj = self.proj_actor(obs).unsqueeze(1)             # (B, 1, hidden)
        a_out, (ah_new, ac_new) = self.lstm_actor(a_proj, actor_states)
        a_out = a_out.squeeze(1)                               # (B, hidden)
        a_latent = self.mlp_actor(a_out)
        mean = self.action_mean(a_latent)
        std = self.log_std.clamp(min=-4.0).exp().expand_as(mean)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        # --- Critic ---
        c_proj = self.proj_critic(obs).unsqueeze(1)
        c_out, (ch_new, cc_new) = self.lstm_critic(c_proj, critic_states)
        c_out = c_out.squeeze(1)
        c_latent = self.mlp_critic(c_out)
        value = self.value_head(c_latent).squeeze(-1)          # (B,)

        return action, value, log_prob, (ah_new, ac_new), (ch_new, cc_new)

    # ------------------------------------------------------------------
    # Deterministic single-step (evaluation / predict)
    # ------------------------------------------------------------------

    def deterministic_step(
        self,
        obs: torch.Tensor,
        actor_states: tuple[torch.Tensor, torch.Tensor],
        critic_states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
        w: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        """Like ``step()`` but returns the action mean (no sampling).

        Returns
        -------
        action_mean, new_actor_states, new_critic_states
        """
        actor_states = self._reset_states(actor_states, episode_starts)
        critic_states = self._reset_states(critic_states, episode_starts)

        # --- Actor ---
        a_proj = self.proj_actor(obs).unsqueeze(1)
        a_out, (ah_new, ac_new) = self.lstm_actor(a_proj, actor_states)
        a_out = a_out.squeeze(1)
        a_latent = self.mlp_actor(a_out)
        action = self.action_mean(a_latent)

        # --- Critic (keep states in sync) ---
        c_proj = self.proj_critic(obs).unsqueeze(1)
        _, (ch_new, cc_new) = self.lstm_critic(c_proj, critic_states)

        return action, (ah_new, ac_new), (ch_new, cc_new)

    # ------------------------------------------------------------------
    # Batch sequence processing (PPO training)
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        episode_starts: torch.Tensor,
        context_obs: Optional[torch.Tensor] = None,
        context_episode_starts: Optional[torch.Tensor] = None,
        w: Optional[torch.Tensor] = None,
        context_w: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions on batched sequences.

        Parameters
        ----------
        obs : (B, T, obs_dim)
        actions : (B, T, act_dim)
        episode_starts : (B, T)
        context_obs : (B, K, obs_dim)  optional burn-in context
        context_episode_starts : (B, K)

        Returns
        -------
        values : (B, T)
        log_probs : (B, T)
        entropy : (B, T)
        """
        # --- Prepend burn-in context ---
        if context_obs is not None:
            K = context_obs.shape[1]
            full_obs = torch.cat([context_obs, obs], dim=1)
            full_ep = torch.cat([context_episode_starts, episode_starts], dim=1)
        else:
            K = 0
            full_obs = obs
            full_ep = episode_starts

        # --- Actor ---
        a_out = self._scan(full_obs, full_ep, self.lstm_actor, self.proj_actor)
        if K > 0:
            a_out = a_out[:, K:, :]
        a_latent = self.mlp_actor(a_out)
        mean = self.action_mean(a_latent)                      # (B, T, act_dim)
        std = self.log_std.clamp(min=-4.0).exp().expand_as(mean)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)         # (B, T)
        entropy = dist.entropy().sum(dim=-1)                   # (B, T)

        # --- Critic ---
        c_out = self._scan(full_obs, full_ep, self.lstm_critic, self.proj_critic)
        if K > 0:
            c_out = c_out[:, K:, :]
        c_latent = self.mlp_critic(c_out)
        values = self.value_head(c_latent).squeeze(-1)         # (B, T)

        return values, log_probs, entropy

    # ------------------------------------------------------------------
    # Single-step value prediction (for GAE bootstrap)
    # ------------------------------------------------------------------

    def get_value(
        self,
        obs: torch.Tensor,
        critic_states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
        w: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Single-step value prediction.

        Returns
        -------
        value : (B,)
        new_critic_states : (h, c)
        """
        critic_states = self._reset_states(critic_states, episode_starts)
        c_proj = self.proj_critic(obs).unsqueeze(1)
        c_out, (ch_new, cc_new) = self.lstm_critic(c_proj, critic_states)
        c_out = c_out.squeeze(1)
        c_latent = self.mlp_critic(c_out)
        value = self.value_head(c_latent).squeeze(-1)
        return value, (ch_new, cc_new)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reset_states(
        states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero-out hidden states where a new episode started.

        Parameters
        ----------
        states : (h, c) each (1, B, hidden_size)
        episode_starts : (B,)
        """
        mask = (1.0 - episode_starts).unsqueeze(0).unsqueeze(-1)  # (1, B, 1)
        return (states[0] * mask, states[1] * mask)

    def _scan(
        self,
        obs: torch.Tensor,
        episode_starts: torch.Tensor,
        lstm: nn.LSTM,
        proj: nn.Linear,
    ) -> torch.Tensor:
        """Run LSTM over a sequence, resetting hidden state at episode boundaries.

        Parameters
        ----------
        obs : (B, L, obs_dim)
        episode_starts : (B, L)

        Returns
        -------
        out : (B, L, hidden_size)
        """
        B, T, _ = obs.shape
        projected = proj(obs)                                   # (B, T, hidden)
        h = torch.zeros(1, B, self.hidden_size, device=obs.device)
        c = torch.zeros(1, B, self.hidden_size, device=obs.device)
        outputs = []
        for t in range(T):
            mask = (1.0 - episode_starts[:, t]).unsqueeze(0).unsqueeze(-1)
            h = h * mask
            c = c * mask
            out_t, (h, c) = lstm(projected[:, t : t + 1, :], (h, c))
            outputs.append(out_t.squeeze(1))
        return torch.stack(outputs, dim=1)                      # (B, T, hidden)
