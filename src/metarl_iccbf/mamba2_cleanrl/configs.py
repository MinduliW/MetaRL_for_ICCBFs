"""Per-environment Mamba2 + RL hyperparameter presets.

Each factory returns a frozen :class:`Mamba2Config` with tuned defaults for
the target scenario.  Training scripts import the relevant preset and use it
as the source of default values (CLI flags can still override individual
fields).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


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


# ---------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------

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
    """Obs 11D, action 3–12D, ~1224-step episodes, 3 CBF constraints."""
    return Mamba2Config(
        mamba_d_model=64,
        mamba_d_state=32,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_headdim=64,
        layers=3,
        nodes=256,
        learning_rate=5e-5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        target_kl=0.02,
        n_epochs=10,
        batch_size=64,
        n_steps=612,
        burn_in=100,
        approx_episode_len=400,
        log_std_init=math.log(0.2),
    )


def inspection_morl_config() -> Mamba2Config:
    """MORL inspection: obs (11+2)D, preference-conditioned Mamba2 PPO.

    Obs is augmented by 2 preference weights w ∈ Δ¹ before the encoder.
    Slightly larger d_model to absorb the extra conditioning signal.
    """
    return Mamba2Config(
        mamba_d_model=64,
        mamba_d_state=32,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_headdim=64,
        layers=3,
        nodes=256,
        learning_rate=5e-5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        target_kl=0.02,
        n_epochs=10,
        batch_size=64,
        n_steps=612,
        burn_in=100,
        approx_episode_len=400,
        log_std_init=math.log(0.2),
    )


def docking_morl_config() -> Mamba2Config:
    """MORL docking: obs (5+2)D, preference-conditioned Mamba2 PPO.

    Obs is augmented by 2 preference weights w ∈ Δ¹ before the encoder.
    """
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
