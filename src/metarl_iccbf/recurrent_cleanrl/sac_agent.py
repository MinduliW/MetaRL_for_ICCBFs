"""Recurrent SAC agents: Mamba2, GRU, and LSTM backends.

Separate module from ``agent.py`` because SAC requires fundamentally different
network heads:
- Squashed Gaussian actor (tanh-transformed, state-dependent log_std)
- Twin Q-networks (obs+action → scalar) instead of value function
- Target networks (EMA-averaged copies of critic)
"""

from __future__ import annotations

from itertools import chain
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Normal

from .agent import _build_mlp, _ortho_init


# ---------------------------------------------------------------------------
# Squashed Gaussian
# ---------------------------------------------------------------------------

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class SquashedGaussian:
    """Tanh-squashed normal distribution with correct log_prob.

    Not a ``torch.distributions`` subclass because we need custom
    reparameterized sampling with the tanh correction.
    """

    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor):
        self.mean = mean.clamp(-10.0, 10.0)
        self.log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        self.std = self.log_std.exp()
        self._normal = Normal(self.mean, self.std)

    def rsample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterized sample with log_prob (includes tanh correction).

        Returns
        -------
        action : Tensor   tanh(u), values in (-1, 1)
        log_prob : Tensor  sum over action dims, shape (...,)
        """
        u = self._normal.rsample()
        action = torch.tanh(u)
        log_prob = self._normal.log_prob(u) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        return action, log_prob

    def sample_deterministic(self) -> torch.Tensor:
        """Deterministic action = tanh(mean)."""
        return torch.tanh(self.mean)

    def entropy(self) -> torch.Tensor:
        """Approximate entropy (Gaussian entropy, ignoring tanh correction)."""
        return self._normal.entropy().sum(dim=-1)


# ---------------------------------------------------------------------------
# Mamba2 SAC Agent
# ---------------------------------------------------------------------------

class Mamba2SACAgent(nn.Module):
    """Mamba2-backed SAC agent with squashed Gaussian actor and twin Q-critics.

    The twin Q-networks share a single Mamba2 backbone (``mamba_critic``) with
    separate MLP heads, reducing memory from 5 to 3 Mamba2 blocks total.

    Parameters
    ----------
    obs_dim : int
    act_dim : int
    mamba_d_model, mamba_d_state, mamba_d_conv, mamba_expand, mamba_headdim :
        Mamba2 hyper-parameters.
    net_arch : dict
        ``{"pi": [h1, h2, ...], "qf": [h1, h2, ...]}`` for the MLP heads.
    activation_fn : type[nn.Module]
        Activation class (default ``nn.ReLU`` — standard for SAC).
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
        activation_fn: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()

        from mamba_ssm.modules.mamba2 import Mamba2  # noqa: F401  # lazy: avoid import at module level

        if net_arch is None:
            net_arch = {"pi": [256, 256], "qf": [256, 256]}

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.mamba_d_model = mamba_d_model

        # ---- Derived Mamba2 dimensions ----
        d_inner = mamba_d_model * mamba_expand
        d_ssm = d_inner
        nheads = d_ssm // mamba_headdim
        ngroups = 1
        conv_dim = d_ssm + 2 * ngroups * mamba_d_state

        self._conv_state_shape = (conv_dim, mamba_d_conv)
        self._ssm_state_shape = (nheads, mamba_headdim, mamba_d_state)

        mamba_kwargs = dict(
            d_model=mamba_d_model,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            headdim=mamba_headdim,
            ngroups=ngroups,
        )

        # ======== Actor ========
        self.proj_actor = nn.Linear(obs_dim, mamba_d_model)
        self.mamba_actor = Mamba2(**mamba_kwargs)
        self.mlp_actor = _build_mlp(mamba_d_model, net_arch["pi"], activation_fn)
        pi_out_dim = net_arch["pi"][-1] if net_arch["pi"] else mamba_d_model
        self.action_mean = nn.Linear(pi_out_dim, act_dim)
        self.action_log_std = nn.Linear(pi_out_dim, act_dim)

        # ======== Twin Q-critics (shared Mamba2 backbone) ========
        self.proj_critic = nn.Linear(obs_dim, mamba_d_model)
        self.mamba_critic = Mamba2(**mamba_kwargs)

        qf_input_dim = mamba_d_model + act_dim
        self.mlp_qf1 = _build_mlp(qf_input_dim, net_arch["qf"], activation_fn)
        self.mlp_qf2 = _build_mlp(qf_input_dim, net_arch["qf"], activation_fn)
        qf_out_dim = net_arch["qf"][-1] if net_arch["qf"] else qf_input_dim
        self.qf1_head = nn.Linear(qf_out_dim, 1)
        self.qf2_head = nn.Linear(qf_out_dim, 1)

        # ---- Initialization ----
        for m in [self.proj_actor, self.proj_critic]:
            _ortho_init(m, gain=1.0)
        _ortho_init(self.action_mean, gain=0.01)
        _ortho_init(self.action_log_std, gain=0.01)

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
    # Backend-agnostic helpers (used by RecurrentSAC)
    # ------------------------------------------------------------------

    def actor_parameters(self):
        """Iterator over all actor parameters."""
        return chain(
            self.proj_actor.parameters(),
            self.mamba_actor.parameters(),
            self.mlp_actor.parameters(),
            self.action_mean.parameters(),
            self.action_log_std.parameters(),
        )

    def critic_parameters(self):
        """Iterator over all online critic parameters."""
        return chain(
            self.proj_critic.parameters(),
            self.mamba_critic.parameters(),
            self.mlp_qf1.parameters(),
            self.mlp_qf2.parameters(),
            self.qf1_head.parameters(),
            self.qf2_head.parameters(),
        )

    def target_critic_source_modules(self) -> dict[str, nn.Module]:
        """Modules to deepcopy for the target critic network."""
        return {
            "proj": self.proj_critic,
            "recurrent": self.mamba_critic,
            "mlp_qf1": self.mlp_qf1,
            "mlp_qf2": self.mlp_qf2,
            "qf1_head": self.qf1_head,
            "qf2_head": self.qf2_head,
        }

    @staticmethod
    def zero_env_actor_states(states, env_idx: int) -> None:
        """Zero-out actor recurrent states for a single environment."""
        # Mamba2 states: (conv_state, ssm_state) tuple
        states[0][env_idx].zero_()
        states[1][env_idx].zero_()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def initial_actor_states(
        self, n_envs: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero-initialized ``(conv_state, ssm_state)`` for actor."""
        return (
            torch.zeros(n_envs, *self._conv_state_shape, device=device),
            torch.zeros(n_envs, *self._ssm_state_shape, device=device),
        )

    def initial_critic_states(
        self, n_envs: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero-initialized ``(conv_state, ssm_state)`` for critic."""
        return (
            torch.zeros(n_envs, *self._conv_state_shape, device=device),
            torch.zeros(n_envs, *self._ssm_state_shape, device=device),
        )

    @staticmethod
    def _reset_states(
        states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero-out states where a new episode started."""
        mask = 1.0 - episode_starts
        conv = states[0] * mask.view(-1, 1, 1)
        ssm = states[1] * mask.view(-1, 1, 1, 1)
        return conv, ssm

    # ------------------------------------------------------------------
    # Parallel scan helper
    # ------------------------------------------------------------------

    def _scan(
        self,
        obs: torch.Tensor,
        episode_starts: torch.Tensor,
        mamba: Mamba2,
        proj: nn.Linear,
    ) -> torch.Tensor:
        """Run ``mamba.forward()`` with ``seq_idx`` for episode boundaries.

        Parameters
        ----------
        obs : (B, L, obs_dim)
        episode_starts : (B, L)

        Returns
        -------
        out : (B, L, d_model)
        """
        ep = episode_starts.clone()
        ep[:, 0] = 1.0

        seq_idx = (ep.cumsum(dim=1).int() - 1) % 2
        has_boundaries = seq_idx.max() > 0

        projected = proj(obs)

        # Clamp inputs to Mamba2 to prevent extreme activations from
        # triggering illegal memory access in the selective-scan CUDA kernel.
        # SAC's double-scan backward (actor → critic) can drive weights to
        # extremes that PPO's milder policy-gradient updates avoid.
        projected = projected.clamp(-50.0, 50.0)

        if has_boundaries:
            out = mamba(projected, seq_idx=seq_idx)
        else:
            out = mamba(projected)

        return out

    # ------------------------------------------------------------------
    # Single-step actor inference (for environment interaction)
    # ------------------------------------------------------------------

    def step_actor(
        self,
        obs: torch.Tensor,
        actor_states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Single-step actor for data collection.

        Returns
        -------
        action : (n_envs, act_dim)    in (-1, 1) due to tanh
        log_prob : (n_envs,)
        new_actor_states : (conv_state, ssm_state)
        """
        actor_states = self._reset_states(actor_states, episode_starts)

        a_proj = self.proj_actor(obs)
        a_out, a_conv, a_ssm = self.mamba_actor.step(
            a_proj.unsqueeze(1), actor_states[0], actor_states[1],
        )
        a_out = a_out.squeeze(1)
        a_latent = self.mlp_actor(a_out)
        mean = self.action_mean(a_latent)
        log_std = self.action_log_std(a_latent)

        dist = SquashedGaussian(mean, log_std)
        if deterministic:
            action = dist.sample_deterministic()
            log_prob = torch.zeros(obs.shape[0], device=obs.device)
        else:
            action, log_prob = dist.rsample()

        return action, log_prob, (a_conv, a_ssm)

    # ------------------------------------------------------------------
    # Batch actor forward (for policy loss)
    # ------------------------------------------------------------------

    def forward_actor(
        self,
        obs: torch.Tensor,
        episode_starts: torch.Tensor,
        context_obs: Optional[torch.Tensor] = None,
        context_episode_starts: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch actor forward using parallel scan.

        Returns
        -------
        actions : (B, T, act_dim)    reparameterized samples in (-1, 1)
        log_probs : (B, T)
        """
        if context_obs is not None:
            K = context_obs.shape[1]
            full_obs = torch.cat([context_obs, obs], dim=1)
            full_ep = torch.cat([context_episode_starts, episode_starts], dim=1)
        else:
            K = 0
            full_obs = obs
            full_ep = episode_starts

        a_out = self._scan(full_obs, full_ep, self.mamba_actor, self.proj_actor)
        if K > 0:
            a_out = a_out[:, K:, :]

        a_latent = self.mlp_actor(a_out)
        mean = self.action_mean(a_latent)
        log_std = self.action_log_std(a_latent)
        dist = SquashedGaussian(mean, log_std)
        actions, log_probs = dist.rsample()
        return actions, log_probs

    # ------------------------------------------------------------------
    # Batch Q-value computation (for critic loss)
    # ------------------------------------------------------------------

    def forward_critic(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        episode_starts: torch.Tensor,
        context_obs: Optional[torch.Tensor] = None,
        context_episode_starts: Optional[torch.Tensor] = None,
        target_modules: Optional[nn.ModuleDict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Q1, Q2 values for (obs_sequence, actions).

        Pass *target_modules* (a ``nn.ModuleDict``) to use target-network
        components instead of the online critic.

        Returns
        -------
        q1 : (B, T)
        q2 : (B, T)
        """
        if target_modules is not None:
            _recurrent = target_modules["recurrent"]
            _proj = target_modules["proj"]
            _mlp1 = target_modules["mlp_qf1"]
            _mlp2 = target_modules["mlp_qf2"]
            _head1 = target_modules["qf1_head"]
            _head2 = target_modules["qf2_head"]
        else:
            _recurrent = self.mamba_critic
            _proj = self.proj_critic
            _mlp1 = self.mlp_qf1
            _mlp2 = self.mlp_qf2
            _head1 = self.qf1_head
            _head2 = self.qf2_head

        if context_obs is not None:
            K = context_obs.shape[1]
            full_obs = torch.cat([context_obs, obs], dim=1)
            full_ep = torch.cat([context_episode_starts, episode_starts], dim=1)
        else:
            K = 0
            full_obs = obs
            full_ep = episode_starts

        c_out = self._scan(full_obs, full_ep, _recurrent, _proj)
        if K > 0:
            c_out = c_out[:, K:, :]

        qf_input = torch.cat([c_out, actions], dim=-1)

        q1 = _head1(_mlp1(qf_input)).squeeze(-1)
        q2 = _head2(_mlp2(qf_input)).squeeze(-1)

        return q1, q2


# ---------------------------------------------------------------------------
# GRU SAC Agent
# ---------------------------------------------------------------------------

class GRUSACAgent(nn.Module):
    """GRU-backed SAC agent with squashed Gaussian actor and twin Q-critics.

    Drop-in replacement for ``Mamba2SACAgent`` — same external interface.

    Parameters
    ----------
    obs_dim : int
    act_dim : int
    hidden_size : int
        GRU hidden state size.
    net_arch : dict
        ``{"pi": [h1, h2, ...], "qf": [h1, h2, ...]}`` for the MLP heads.
    activation_fn : type[nn.Module]
        Activation class (default ``nn.ReLU`` — standard for SAC).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        *,
        hidden_size: int = 64,
        net_arch: Optional[dict[str, list[int]]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()

        if net_arch is None:
            net_arch = {"pi": [256, 256], "qf": [256, 256]}

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size

        # ======== Actor ========
        self.proj_actor = nn.Linear(obs_dim, hidden_size)
        self.gru_actor = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.mlp_actor = _build_mlp(hidden_size, net_arch["pi"], activation_fn)
        pi_out_dim = net_arch["pi"][-1] if net_arch["pi"] else hidden_size
        self.action_mean = nn.Linear(pi_out_dim, act_dim)
        self.action_log_std = nn.Linear(pi_out_dim, act_dim)

        # ======== Twin Q-critics (shared GRU backbone) ========
        self.proj_critic = nn.Linear(obs_dim, hidden_size)
        self.gru_critic = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        qf_input_dim = hidden_size + act_dim
        self.mlp_qf1 = _build_mlp(qf_input_dim, net_arch["qf"], activation_fn)
        self.mlp_qf2 = _build_mlp(qf_input_dim, net_arch["qf"], activation_fn)
        qf_out_dim = net_arch["qf"][-1] if net_arch["qf"] else qf_input_dim
        self.qf1_head = nn.Linear(qf_out_dim, 1)
        self.qf2_head = nn.Linear(qf_out_dim, 1)

        # ---- Initialization ----
        for m in [self.proj_actor, self.proj_critic]:
            _ortho_init(m, gain=1.0)
        _ortho_init(self.action_mean, gain=0.01)
        _ortho_init(self.action_log_std, gain=0.01)

    # ------------------------------------------------------------------
    # Backend-agnostic helpers (used by RecurrentSAC)
    # ------------------------------------------------------------------

    def actor_parameters(self):
        """Iterator over all actor parameters."""
        return chain(
            self.proj_actor.parameters(),
            self.gru_actor.parameters(),
            self.mlp_actor.parameters(),
            self.action_mean.parameters(),
            self.action_log_std.parameters(),
        )

    def critic_parameters(self):
        """Iterator over all online critic parameters."""
        return chain(
            self.proj_critic.parameters(),
            self.gru_critic.parameters(),
            self.mlp_qf1.parameters(),
            self.mlp_qf2.parameters(),
            self.qf1_head.parameters(),
            self.qf2_head.parameters(),
        )

    def target_critic_source_modules(self) -> dict[str, nn.Module]:
        """Modules to deepcopy for the target critic network."""
        return {
            "proj": self.proj_critic,
            "recurrent": self.gru_critic,
            "mlp_qf1": self.mlp_qf1,
            "mlp_qf2": self.mlp_qf2,
            "qf1_head": self.qf1_head,
            "qf2_head": self.qf2_head,
        }

    @staticmethod
    def zero_env_actor_states(states, env_idx: int) -> None:
        """Zero-out actor recurrent states for a single environment."""
        # GRU states: (1, n_envs, hidden_size) tensor
        states[:, env_idx, :].zero_()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def initial_actor_states(
        self, n_envs: int, device: torch.device
    ) -> torch.Tensor:
        """Zero-initialized ``(1, n_envs, hidden_size)`` for actor."""
        return torch.zeros(1, n_envs, self.hidden_size, device=device)

    def initial_critic_states(
        self, n_envs: int, device: torch.device
    ) -> torch.Tensor:
        """Zero-initialized ``(1, n_envs, hidden_size)`` for critic."""
        return torch.zeros(1, n_envs, self.hidden_size, device=device)

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

    # ------------------------------------------------------------------
    # Sequential scan helper
    # ------------------------------------------------------------------

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
        projected = proj(obs)                                    # (B, T, hidden)
        h = torch.zeros(1, B, self.hidden_size, device=obs.device)
        outputs = []
        for t in range(T):
            mask = (1.0 - episode_starts[:, t]).unsqueeze(0).unsqueeze(-1)
            h = h * mask
            out_t, h = gru(projected[:, t : t + 1, :], h)
            outputs.append(out_t.squeeze(1))
        return torch.stack(outputs, dim=1)                       # (B, T, hidden)

    # ------------------------------------------------------------------
    # Single-step actor inference (for environment interaction)
    # ------------------------------------------------------------------

    def step_actor(
        self,
        obs: torch.Tensor,
        actor_states: torch.Tensor,
        episode_starts: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step actor for data collection.

        Returns
        -------
        action : (n_envs, act_dim)    in (-1, 1) due to tanh
        log_prob : (n_envs,)
        new_actor_states : (1, n_envs, hidden_size)
        """
        actor_states = self._reset_states(actor_states, episode_starts)

        a_proj = self.proj_actor(obs).unsqueeze(1)          # (B, 1, hidden)
        a_out, actor_states = self.gru_actor(a_proj, actor_states)
        a_out = a_out.squeeze(1)                            # (B, hidden)
        a_latent = self.mlp_actor(a_out)
        mean = self.action_mean(a_latent)
        log_std = self.action_log_std(a_latent)

        dist = SquashedGaussian(mean, log_std)
        if deterministic:
            action = dist.sample_deterministic()
            log_prob = torch.zeros(obs.shape[0], device=obs.device)
        else:
            action, log_prob = dist.rsample()

        return action, log_prob, actor_states

    # ------------------------------------------------------------------
    # Batch actor forward (for policy loss)
    # ------------------------------------------------------------------

    def forward_actor(
        self,
        obs: torch.Tensor,
        episode_starts: torch.Tensor,
        context_obs: Optional[torch.Tensor] = None,
        context_episode_starts: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch actor forward using sequential GRU scan.

        Returns
        -------
        actions : (B, T, act_dim)    reparameterized samples in (-1, 1)
        log_probs : (B, T)
        """
        if context_obs is not None:
            K = context_obs.shape[1]
            full_obs = torch.cat([context_obs, obs], dim=1)
            full_ep = torch.cat([context_episode_starts, episode_starts], dim=1)
        else:
            K = 0
            full_obs = obs
            full_ep = episode_starts

        a_out = self._scan(full_obs, full_ep, self.gru_actor, self.proj_actor)
        if K > 0:
            a_out = a_out[:, K:, :]

        a_latent = self.mlp_actor(a_out)
        mean = self.action_mean(a_latent)
        log_std = self.action_log_std(a_latent)
        dist = SquashedGaussian(mean, log_std)
        actions, log_probs = dist.rsample()
        return actions, log_probs

    # ------------------------------------------------------------------
    # Batch Q-value computation (for critic loss)
    # ------------------------------------------------------------------

    def forward_critic(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        episode_starts: torch.Tensor,
        context_obs: Optional[torch.Tensor] = None,
        context_episode_starts: Optional[torch.Tensor] = None,
        target_modules: Optional[nn.ModuleDict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Q1, Q2 values for (obs_sequence, actions).

        Pass *target_modules* (a ``nn.ModuleDict``) to use target-network
        components instead of the online critic.

        Returns
        -------
        q1 : (B, T)
        q2 : (B, T)
        """
        if target_modules is not None:
            _recurrent = target_modules["recurrent"]
            _proj = target_modules["proj"]
            _mlp1 = target_modules["mlp_qf1"]
            _mlp2 = target_modules["mlp_qf2"]
            _head1 = target_modules["qf1_head"]
            _head2 = target_modules["qf2_head"]
        else:
            _recurrent = self.gru_critic
            _proj = self.proj_critic
            _mlp1 = self.mlp_qf1
            _mlp2 = self.mlp_qf2
            _head1 = self.qf1_head
            _head2 = self.qf2_head

        if context_obs is not None:
            K = context_obs.shape[1]
            full_obs = torch.cat([context_obs, obs], dim=1)
            full_ep = torch.cat([context_episode_starts, episode_starts], dim=1)
        else:
            K = 0
            full_obs = obs
            full_ep = episode_starts

        c_out = self._scan(full_obs, full_ep, _recurrent, _proj)
        if K > 0:
            c_out = c_out[:, K:, :]

        qf_input = torch.cat([c_out, actions], dim=-1)

        q1 = _head1(_mlp1(qf_input)).squeeze(-1)
        q2 = _head2(_mlp2(qf_input)).squeeze(-1)

        return q1, q2


# ---------------------------------------------------------------------------
# LSTM backend
# ---------------------------------------------------------------------------

class LSTMSACAgent(nn.Module):
    """LSTM-backed SAC agent with squashed Gaussian actor and twin Q-critics.

    Drop-in replacement for ``Mamba2SACAgent`` / ``GRUSACAgent`` — same
    external interface.  The key difference from ``GRUSACAgent`` is that LSTM
    carries both hidden state *h* and cell state *c*, so all state-management
    helpers operate on ``(h, c)`` tuples.

    Parameters
    ----------
    obs_dim : int
    act_dim : int
    hidden_size : int
        LSTM hidden state size.
    net_arch : dict
        ``{"pi": [h1, h2, ...], "qf": [h1, h2, ...]}`` for the MLP heads.
    activation_fn : type[nn.Module]
        Activation class (default ``nn.ReLU`` — standard for SAC).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        *,
        hidden_size: int = 64,
        net_arch: Optional[dict[str, list[int]]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()

        if net_arch is None:
            net_arch = {"pi": [256, 256], "qf": [256, 256]}

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size

        # ======== Actor ========
        self.proj_actor = nn.Linear(obs_dim, hidden_size)
        self.lstm_actor = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.mlp_actor = _build_mlp(hidden_size, net_arch["pi"], activation_fn)
        pi_out_dim = net_arch["pi"][-1] if net_arch["pi"] else hidden_size
        self.action_mean = nn.Linear(pi_out_dim, act_dim)
        self.action_log_std = nn.Linear(pi_out_dim, act_dim)

        # ======== Twin Q-critics (shared LSTM backbone) ========
        self.proj_critic = nn.Linear(obs_dim, hidden_size)
        self.lstm_critic = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        qf_input_dim = hidden_size + act_dim
        self.mlp_qf1 = _build_mlp(qf_input_dim, net_arch["qf"], activation_fn)
        self.mlp_qf2 = _build_mlp(qf_input_dim, net_arch["qf"], activation_fn)
        qf_out_dim = net_arch["qf"][-1] if net_arch["qf"] else qf_input_dim
        self.qf1_head = nn.Linear(qf_out_dim, 1)
        self.qf2_head = nn.Linear(qf_out_dim, 1)

        # ---- Initialization ----
        for m in [self.proj_actor, self.proj_critic]:
            _ortho_init(m, gain=1.0)
        _ortho_init(self.action_mean, gain=0.01)
        _ortho_init(self.action_log_std, gain=0.01)

    # ------------------------------------------------------------------
    # Backend-agnostic helpers (used by RecurrentSAC)
    # ------------------------------------------------------------------

    def actor_parameters(self):
        """Iterator over all actor parameters."""
        return chain(
            self.proj_actor.parameters(),
            self.lstm_actor.parameters(),
            self.mlp_actor.parameters(),
            self.action_mean.parameters(),
            self.action_log_std.parameters(),
        )

    def critic_parameters(self):
        """Iterator over all online critic parameters."""
        return chain(
            self.proj_critic.parameters(),
            self.lstm_critic.parameters(),
            self.mlp_qf1.parameters(),
            self.mlp_qf2.parameters(),
            self.qf1_head.parameters(),
            self.qf2_head.parameters(),
        )

    def target_critic_source_modules(self) -> dict[str, nn.Module]:
        """Modules to deepcopy for the target critic network."""
        return {
            "proj": self.proj_critic,
            "recurrent": self.lstm_critic,
            "mlp_qf1": self.mlp_qf1,
            "mlp_qf2": self.mlp_qf2,
            "qf1_head": self.qf1_head,
            "qf2_head": self.qf2_head,
        }

    @staticmethod
    def zero_env_actor_states(
        states: tuple[torch.Tensor, torch.Tensor], env_idx: int
    ) -> None:
        """Zero-out actor recurrent states for a single environment."""
        h, c = states
        h[:, env_idx, :].zero_()
        c[:, env_idx, :].zero_()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def initial_actor_states(
        self, n_envs: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero-initialized ``(h, c)`` tuple for actor."""
        h = torch.zeros(1, n_envs, self.hidden_size, device=device)
        c = torch.zeros(1, n_envs, self.hidden_size, device=device)
        return (h, c)

    def initial_critic_states(
        self, n_envs: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero-initialized ``(h, c)`` tuple for critic."""
        h = torch.zeros(1, n_envs, self.hidden_size, device=device)
        c = torch.zeros(1, n_envs, self.hidden_size, device=device)
        return (h, c)

    @staticmethod
    def _reset_states(
        states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero-out hidden and cell states where a new episode started.

        Parameters
        ----------
        states : (h, c) each ``(1, B, hidden_size)``
        episode_starts : (B,)
        """
        mask = (1.0 - episode_starts).unsqueeze(0).unsqueeze(-1)  # (1, B, 1)
        return (states[0] * mask, states[1] * mask)

    # ------------------------------------------------------------------
    # Sequential scan helper
    # ------------------------------------------------------------------

    def _scan(
        self,
        obs: torch.Tensor,
        episode_starts: torch.Tensor,
        lstm: nn.LSTM,
        proj: nn.Linear,
    ) -> torch.Tensor:
        """Run LSTM over a sequence, resetting states at episode boundaries.

        Parameters
        ----------
        obs : (B, L, obs_dim)
        episode_starts : (B, L)

        Returns
        -------
        out : (B, L, hidden_size)
        """
        B, T, _ = obs.shape
        projected = proj(obs)                                    # (B, T, hidden)
        h = torch.zeros(1, B, self.hidden_size, device=obs.device)
        c = torch.zeros(1, B, self.hidden_size, device=obs.device)
        outputs = []
        for t in range(T):
            mask = (1.0 - episode_starts[:, t]).unsqueeze(0).unsqueeze(-1)
            h = h * mask
            c = c * mask
            out_t, (h, c) = lstm(projected[:, t : t + 1, :], (h, c))
            outputs.append(out_t.squeeze(1))
        return torch.stack(outputs, dim=1)                       # (B, T, hidden)

    # ------------------------------------------------------------------
    # Single-step actor inference (for environment interaction)
    # ------------------------------------------------------------------

    def step_actor(
        self,
        obs: torch.Tensor,
        actor_states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Single-step actor for data collection.

        Returns
        -------
        action : (n_envs, act_dim)    in (-1, 1) due to tanh
        log_prob : (n_envs,)
        new_actor_states : (h, c) tuple
        """
        actor_states = self._reset_states(actor_states, episode_starts)

        a_proj = self.proj_actor(obs).unsqueeze(1)          # (B, 1, hidden)
        a_out, (h_new, c_new) = self.lstm_actor(a_proj, actor_states)
        a_out = a_out.squeeze(1)                            # (B, hidden)
        a_latent = self.mlp_actor(a_out)
        mean = self.action_mean(a_latent)
        log_std = self.action_log_std(a_latent)

        dist = SquashedGaussian(mean, log_std)
        if deterministic:
            action = dist.sample_deterministic()
            log_prob = torch.zeros(obs.shape[0], device=obs.device)
        else:
            action, log_prob = dist.rsample()

        return action, log_prob, (h_new, c_new)

    # ------------------------------------------------------------------
    # Batch actor forward (for policy loss)
    # ------------------------------------------------------------------

    def forward_actor(
        self,
        obs: torch.Tensor,
        episode_starts: torch.Tensor,
        context_obs: Optional[torch.Tensor] = None,
        context_episode_starts: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch actor forward using sequential LSTM scan.

        Returns
        -------
        actions : (B, T, act_dim)    reparameterized samples in (-1, 1)
        log_probs : (B, T)
        """
        if context_obs is not None:
            K = context_obs.shape[1]
            full_obs = torch.cat([context_obs, obs], dim=1)
            full_ep = torch.cat([context_episode_starts, episode_starts], dim=1)
        else:
            K = 0
            full_obs = obs
            full_ep = episode_starts

        a_out = self._scan(full_obs, full_ep, self.lstm_actor, self.proj_actor)
        if K > 0:
            a_out = a_out[:, K:, :]

        a_latent = self.mlp_actor(a_out)
        mean = self.action_mean(a_latent)
        log_std = self.action_log_std(a_latent)
        dist = SquashedGaussian(mean, log_std)
        actions, log_probs = dist.rsample()
        return actions, log_probs

    # ------------------------------------------------------------------
    # Batch Q-value computation (for critic loss)
    # ------------------------------------------------------------------

    def forward_critic(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        episode_starts: torch.Tensor,
        context_obs: Optional[torch.Tensor] = None,
        context_episode_starts: Optional[torch.Tensor] = None,
        target_modules: Optional[nn.ModuleDict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Q1, Q2 values for (obs_sequence, actions).

        Pass *target_modules* (a ``nn.ModuleDict``) to use target-network
        components instead of the online critic.

        Returns
        -------
        q1 : (B, T)
        q2 : (B, T)
        """
        if target_modules is not None:
            _recurrent = target_modules["recurrent"]
            _proj = target_modules["proj"]
            _mlp1 = target_modules["mlp_qf1"]
            _mlp2 = target_modules["mlp_qf2"]
            _head1 = target_modules["qf1_head"]
            _head2 = target_modules["qf2_head"]
        else:
            _recurrent = self.lstm_critic
            _proj = self.proj_critic
            _mlp1 = self.mlp_qf1
            _mlp2 = self.mlp_qf2
            _head1 = self.qf1_head
            _head2 = self.qf2_head

        if context_obs is not None:
            K = context_obs.shape[1]
            full_obs = torch.cat([context_obs, obs], dim=1)
            full_ep = torch.cat([context_episode_starts, episode_starts], dim=1)
        else:
            K = 0
            full_obs = obs
            full_ep = episode_starts

        c_out = self._scan(full_obs, full_ep, _recurrent, _proj)
        if K > 0:
            c_out = c_out[:, K:, :]

        qf_input = torch.cat([c_out, actions], dim=-1)

        q1 = _head1(_mlp1(qf_input)).squeeze(-1)
        q2 = _head2(_mlp2(qf_input)).squeeze(-1)

        return q1, q2
