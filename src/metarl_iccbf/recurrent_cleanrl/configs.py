"""Per-environment hyperparameter presets for Mamba2 and GRU.

Each factory returns a frozen config dataclass with tuned defaults for
the target scenario.  Training scripts import the relevant preset and use it
as the source of default values (CLI flags can still override individual
fields).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------
# Mamba2 configs
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Mamba2Config:
    """All Mamba2 architecture + PPO training hyperparameters."""

    # -- Mamba2 SSM --
    mamba_d_model: int
    mamba_d_state: int
    mamba_d_conv: int
    mamba_expand: int
    mamba_headdim: int

    # -- MLP heads --
    layers: int
    nodes: int

    # -- RL / PPO --
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    target_kl: Optional[float]
    n_epochs: int
    batch_size: int
    n_steps: int
    burn_in: int
    approx_episode_len: int
    log_std_init: float


def cruise_control_config() -> Mamba2Config:
    """Obs 2D, action 4D, ~200-step episodes."""
    return Mamba2Config(
        mamba_d_model=32,
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_headdim=32,
        layers=2,
        nodes=64,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        target_kl=0.05,
        n_epochs=10,
        batch_size=14,
        n_steps=200,
        burn_in=40,
        approx_episode_len=200,
        log_std_init=math.log(0.5),
    )


def docking_config() -> Mamba2Config:
    """Obs 5D, action 4D, ~100-step episodes, 6 hidden task params."""
    return Mamba2Config(
        mamba_d_model=32,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_headdim=32,
        layers=2,
        nodes=128,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        target_kl=0.05,
        n_epochs=10,
        batch_size=14,
        n_steps=100,
        burn_in=20,
        approx_episode_len=100,
        log_std_init=math.log(0.5),
    )


def inspection_config() -> Mamba2Config:
    """Obs 11D, action 3-12D, ~1224-step episodes, 3 CBF constraints."""
    return Mamba2Config(
        mamba_d_model=64,
        mamba_d_state=32,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_headdim=64,
        layers=3,
        nodes=256,
        learning_rate=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        target_kl=0.05,
        n_epochs=10,
        batch_size=64,
        n_steps=612,
        burn_in=100,
        approx_episode_len=400,
        log_std_init=math.log(0.2),
    )


# ---------------------------------------------------------------------
# GRU configs
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class GRUConfig:
    """GRU architecture + PPO training hyperparameters."""

    # -- GRU --
    hidden_size: int

    # -- MLP heads --
    layers: int
    nodes: int

    # -- RL / PPO --
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    target_kl: Optional[float]
    n_epochs: int
    batch_size: int
    n_steps: int
    burn_in: int
    approx_episode_len: int
    log_std_init: float


def cruise_control_gru_config() -> GRUConfig:
    """GRU variant — hidden_size matches Mamba d_model for fair comparison."""
    return GRUConfig(
        hidden_size=32,
        layers=2,
        nodes=64,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        target_kl=0.05,
        n_epochs=10,
        batch_size=14,
        n_steps=200,
        burn_in=40,
        approx_episode_len=200,
        log_std_init=math.log(0.5),
    )


def cruise_control_lstm_ppo_config() -> GRUConfig:
    """LSTM PPO variant — hyperparams matched to cruise_control_gru_config for fair comparison."""
    return GRUConfig(
        hidden_size=32,
        layers=2,
        nodes=64,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        target_kl=0.05,
        n_epochs=10,
        batch_size=14,
        n_steps=200,
        burn_in=40,
        approx_episode_len=200,
        log_std_init=math.log(0.5),
    )


def docking_gru_config() -> GRUConfig:
    """GRU variant — hyperparams matched to LSTM (train_DCRNN) for fair comparison."""
    return GRUConfig(
        hidden_size=64,
        layers=4,
        nodes=64,
        learning_rate=5e-5,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        target_kl=0.02,
        n_epochs=10,
        batch_size=64,
        n_steps=80,
        burn_in=20,
        approx_episode_len=100,
        log_std_init=math.log(0.5),
    )


def docking_lstm_config() -> GRUConfig:
    """LSTM PPO variant — same hyperparams as docking_gru_config (matched to train_DCRNN)."""
    return GRUConfig(
        hidden_size=64,
        layers=4,
        nodes=64,
        learning_rate=5e-5,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        target_kl=0.02,
        n_epochs=10,
        batch_size=64,
        n_steps=80,
        burn_in=20,
        approx_episode_len=100,
        log_std_init=math.log(0.5),
    )


def inspection_lstm_ppo_config() -> GRUConfig:
    """CleanRL LSTM PPO — hyperparams matched to the legacy SB3 mainRNNnoisy.py config.

    SB3 used: lr=5e-5 (decreasing), layers=4, nodes=256, lstm_hidden=256,
    n_steps=int(64*50*8/64)=400, batch_size=64, gamma=0.99, clip=0.2,
    ent_coef=0.01, log_std=log(0.2).
    """
    return GRUConfig(
        hidden_size=256,
        layers=4,
        nodes=256,
        learning_rate=5e-5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        target_kl=0.05,
        n_epochs=10,
        batch_size=64,
        n_steps=400,
        burn_in=100,
        approx_episode_len=400,
        log_std_init=math.log(0.2),
    )


def inspection_gru_config() -> GRUConfig:
    """GRU variant — hyperparams matched to LSTM (train_inspection RNN) for fair comparison."""
    return GRUConfig(
        hidden_size=256,
        layers=3,
        nodes=256,
        learning_rate=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        target_kl=0.05,
        n_epochs=10,
        batch_size=64,
        n_steps=612,
        burn_in=100,
        approx_episode_len=400,
        log_std_init=math.log(0.2),
    )


# ---------------------------------------------------------------------
# Mamba2 SAC configs
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Mamba2SACConfig:
    """Mamba2 architecture + SAC training hyperparameters."""

    # -- Mamba2 SSM --
    mamba_d_model: int
    mamba_d_state: int
    mamba_d_conv: int
    mamba_expand: int
    mamba_headdim: int

    # -- MLP heads --
    layers: int
    nodes: int

    # -- SAC --
    learning_rate: float
    gamma: float
    tau: float
    target_entropy: str  # "auto" or a float string
    ent_coef_min: float
    buffer_size: int
    chunk_len: int
    burn_in: int
    batch_size: int
    learning_starts: int
    train_freq: int
    gradient_steps: int
    approx_episode_len: int


def cruise_control_sac_config() -> Mamba2SACConfig:
    """SAC variant for cruise control — Obs 2D, action 4D, ~200-step episodes."""
    return Mamba2SACConfig(
        mamba_d_model=32,
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_headdim=32,
        layers=2,
        nodes=64,
        learning_rate=3e-4,
        gamma=0.995,
        tau=0.005,
        target_entropy="auto",
        ent_coef_min=0.05,
        buffer_size=100_000,
        chunk_len=200,
        burn_in=40,
        batch_size=64,
        learning_starts=5_000,
        train_freq=200,
        gradient_steps=50,
        approx_episode_len=200,
    )


def docking_sac_config() -> Mamba2SACConfig:
    """SAC variant for docking — Obs 5D, action 4D, ~100-step episodes."""
    return Mamba2SACConfig(
        mamba_d_model=32,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_headdim=32,
        layers=2,
        nodes=128,
        learning_rate=3e-4,
        gamma=0.995,
        tau=0.005,
        target_entropy="auto",
        ent_coef_min=0.05,
        buffer_size=100_000,
        chunk_len=100,
        burn_in=20,
        batch_size=64,
        learning_starts=5_000,
        train_freq=100,
        gradient_steps=50,
        approx_episode_len=100,
    )


def inspection_sac_config() -> Mamba2SACConfig:
    """SAC variant for inspection — Obs 11D, action 3-12D, ~1224-step episodes."""
    return Mamba2SACConfig(
        mamba_d_model=64,
        mamba_d_state=32,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_headdim=64,
        layers=3,
        nodes=256,
        learning_rate=1e-4,
        gamma=0.99,
        tau=0.005,
        target_entropy="auto",
        ent_coef_min=0.05,
        buffer_size=50_000,
        chunk_len=200,
        burn_in=100,
        batch_size=64,
        learning_starts=5000,
        train_freq=200,
        gradient_steps=50,
        approx_episode_len=400,
    )


# ---------------------------------------------------------------------
# GRU SAC configs
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class GRUSACConfig:
    """GRU architecture + SAC training hyperparameters."""

    # -- GRU --
    hidden_size: int

    # -- MLP heads --
    layers: int
    nodes: int

    # -- SAC --
    learning_rate: float
    gamma: float
    tau: float
    target_entropy: str  # "auto" or a float string
    ent_coef_min: float
    buffer_size: int
    chunk_len: int
    burn_in: int
    batch_size: int
    learning_starts: int
    train_freq: int
    gradient_steps: int
    approx_episode_len: int


def cruise_control_gru_sac_config() -> GRUSACConfig:
    """GRU SAC for cruise control — hidden_size matched to GRU PPO config."""
    return GRUSACConfig(
        hidden_size=32,
        layers=2,
        nodes=64,
        learning_rate=3e-4,
        gamma=0.995,
        tau=0.005,
        target_entropy="auto",
        ent_coef_min=0.05,
        buffer_size=100_000,
        chunk_len=200,
        burn_in=40,
        batch_size=64,
        learning_starts=5_000,
        train_freq=200,
        gradient_steps=50,
        approx_episode_len=200,
    )


def docking_gru_sac_config() -> GRUSACConfig:
    """GRU SAC for docking — hidden_size matched to GRU PPO config."""
    return GRUSACConfig(
        hidden_size=64,
        layers=2,
        nodes=128,
        learning_rate=3e-4,
        gamma=0.995,
        tau=0.005,
        target_entropy="auto",
        ent_coef_min=0.05,
        buffer_size=100_000,
        chunk_len=100,
        burn_in=20,
        batch_size=64,
        learning_starts=5_000,
        train_freq=100,
        gradient_steps=50,
        approx_episode_len=100,
    )


def inspection_gru_sac_config() -> GRUSACConfig:
    """GRU SAC for inspection — hidden_size matched to GRU PPO config."""
    return GRUSACConfig(
        hidden_size=256,
        layers=3,
        nodes=256,
        learning_rate=1e-4,
        gamma=0.99,
        tau=0.005,
        target_entropy="auto",
        ent_coef_min=0.05,
        buffer_size=50_000,
        chunk_len=200,
        burn_in=100,
        batch_size=64,
        learning_starts=5000,
        train_freq=200,
        gradient_steps=50,
        approx_episode_len=400,
    )


# ---------------------------------------------------------------------
# LSTM SAC configs
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class LSTMSACConfig:
    """LSTM architecture + SAC training hyperparameters."""

    # -- LSTM --
    hidden_size: int

    # -- MLP heads --
    layers: int
    nodes: int

    # -- SAC --
    learning_rate: float
    gamma: float
    tau: float
    target_entropy: str  # "auto" or a float string
    ent_coef_min: float
    buffer_size: int
    chunk_len: int
    burn_in: int
    batch_size: int
    learning_starts: int
    train_freq: int
    gradient_steps: int
    approx_episode_len: int


def cruise_control_lstm_sac_config() -> LSTMSACConfig:
    """LSTM SAC for cruise control."""
    return LSTMSACConfig(
        hidden_size=32,
        layers=2,
        nodes=64,
        learning_rate=3e-4,
        gamma=0.995,
        tau=0.005,
        target_entropy="auto",
        ent_coef_min=0.05,
        buffer_size=100_000,
        chunk_len=200,
        burn_in=40,
        batch_size=64,
        learning_starts=5_000,
        train_freq=200,
        gradient_steps=50,
        approx_episode_len=200,
    )


def docking_lstm_sac_config() -> LSTMSACConfig:
    """LSTM SAC for docking."""
    return LSTMSACConfig(
        hidden_size=64,
        layers=2,
        nodes=128,
        learning_rate=3e-4,
        gamma=0.995,
        tau=0.005,
        target_entropy="auto",
        ent_coef_min=0.05,
        buffer_size=100_000,
        chunk_len=100,
        burn_in=20,
        batch_size=64,
        learning_starts=5_000,
        train_freq=100,
        gradient_steps=50,
        approx_episode_len=100,
    )


def inspection_lstm_sac_config() -> LSTMSACConfig:
    """LSTM SAC for inspection."""
    return LSTMSACConfig(
        hidden_size=256,
        layers=3,
        nodes=256,
        learning_rate=1e-4,
        gamma=0.99,
        tau=0.005,
        target_entropy="auto",
        ent_coef_min=0.05,
        buffer_size=50_000,
        chunk_len=200,
        burn_in=100,
        batch_size=64,
        learning_starts=5000,
        train_freq=200,
        gradient_steps=50,
        approx_episode_len=400,
    )
