# train_DCRNN.py
"""
Docking case trainer (LSTM RecurrentPPO) mirroring the cruise-control train_CCRNN.py style.

This file keeps the same conceptual knobs as the cruise-control version:
- MLP depth/width + activation
- LSTM hidden size / layers / shared LSTM / critic LSTM
- LR schedule (constant vs decreasing)
- VecEnv parallelism via SubprocVecEnv

Environment: RLCBFcontrol in RLCBF.py (docking case).
"""

import gc
import warnings
import multiprocessing
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from torch.optim import Adam
from torch.nn.modules import activation

from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent import MlpLstmPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from .RLCBF import RLCBFcontrol

# Optional: DA init for DA-based margins
try:
    from daceypy import DA  # noqa: F401
    _HAS_DA = True
except Exception:
    _HAS_DA = False


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

class ConstantSchedule:
    def __init__(self, value: float):
        self.value = float(value)

    def __call__(self, progress_remaining: float) -> float:
        return self.value


class LinearSchedule:
    def __init__(self, initial: float, min_value: float):
        self.initial = float(initial)
        self.min_value = float(min_value)

    def __call__(self, progress_remaining: float) -> float:
        return max(self.initial * float(progress_remaining), self.min_value)


def make_env(rank: int, dt: float, deterministic: bool, seed: int,
             init_da: bool, da_order: int, da_vars: int):
    def _init():
        env = RLCBFcontrol(dt=dt, deterministic=deterministic)
        if init_da and _HAS_DA:
            DA.init(int(da_order), int(da_vars))
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        return Monitor(env)
    return _init


# ---------------------------------------------------------------------
# Training function (LSTM only)
# ---------------------------------------------------------------------

def train_docking_lstm(
    *,
    dt: float = 0.5,
    seed: int = 123,

    # ---- Network architecture ----
    layers: int = 4,
    nodes: int = 64,
    activation_fn=activation.Tanh,

    # ---- LSTM parameters ----
    lstm_hidden_size: int = 64,
    n_lstm_layers: int = 1,
    shared_lstm: bool = False,
    enable_critic_lstm: bool = True,

    # ---- RL hyperparameters ----
    total_episodes: int = 200_000,
    approx_episode_len: int = 100,
    learning_rate: float = 5e-5,
    lr_type: Literal["C", "D"] = "C",
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    clip_range: float = 0.1,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    use_sde: bool = True,
    target_kl: Optional[float] = 0.02,
    n_epochs: int = 10,
    batch_size: int = 64,
    n_steps: int = 80,   # must be divisible by num_env

    # ---- Parallelism ----
    num_env: int = 64,

    # ---- DA init (if your margins use DA) ----
    init_da: bool = True,
    da_order: int = 4,
    da_vars: int = 2,

    # ---- Control ----
    trainON: bool = True,
    trainLoad: bool = False,

    # ---- Logging ----
    root_log_dir: str = "TrainedModels",
):
    """
    Train a recurrent (LSTM) ICCBF meta-policy for docking.
    Returns (model, log_dir).
    """

    multiprocessing.freeze_support()
    warnings.filterwarnings("ignore", category=UserWarning)
    gc.collect()

    if not trainON:
        num_env = 1

    # --------------------------------------------------
    # Learning-rate schedule
    # --------------------------------------------------
    if lr_type == "D":
        lr_schedule = LinearSchedule(learning_rate, learning_rate / 100)
    else:
        lr_schedule = ConstantSchedule(learning_rate)

    # --------------------------------------------------
    # Environments
    # --------------------------------------------------
    train_env = SubprocVecEnv([
        make_env(i, dt, deterministic=False, seed=seed,
                 init_da=init_da, da_order=da_order, da_vars=da_vars)
        for i in range(num_env)
    ], start_method="spawn")

    eval_env = RLCBFcontrol(dt=dt, deterministic=True)

    # --------------------------------------------------
    # Policy kwargs
    # --------------------------------------------------
    std = 0.2
    base_policy_kwargs = dict(
        activation_fn=activation_fn,
        ortho_init=True,
        log_std_init=float(np.log(std)),
        share_features_extractor=False,
        optimizer_class=Adam,
        net_arch=[dict(pi=[nodes] * layers, vf=[nodes] * layers)],
    )

    lstm_kwargs = dict(
        lstm_hidden_size=int(lstm_hidden_size),
        n_lstm_layers=int(n_lstm_layers),
        shared_lstm=bool(shared_lstm),
        enable_critic_lstm=bool(enable_critic_lstm),
    )

    policy_kwargs = {**base_policy_kwargs, **lstm_kwargs}

    # --------------------------------------------------
    # Naming + logging
    # --------------------------------------------------
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    arch_str = f"RNN_L{layers}_N{nodes}_H{lstm_hidden_size}"
    hyper_str = f"lr{learning_rate:g}_g{gamma}_gae{gae_lambda}_ent{ent_coef}"
    training_name = f"MetaDocking_RNN_{arch_str}_{hyper_str}_{time_str}"

    log_dir = Path(root_log_dir) / training_name
    log_dir.mkdir(parents=True, exist_ok=True)

    total_timesteps = int(total_episodes * approx_episode_len)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = RecurrentPPO(
        MlpLstmPolicy,
        train_env,
        verbose=1,
        learning_rate=lr_schedule,
        tensorboard_log=str(root_log_dir),
        normalize_advantage=True,
        n_steps=int(n_steps),
        batch_size=int(batch_size),
        n_epochs=int(n_epochs),
        gamma=float(gamma),
        gae_lambda=float(gae_lambda),
        clip_range=float(clip_range),
        ent_coef=float(ent_coef),
        vf_coef=float(vf_coef),
        use_sde=bool(use_sde),
        target_kl=target_kl,
        policy_kwargs=policy_kwargs,
        seed=int(seed),
    )

    if trainLoad:
        best_path = log_dir / "best_model.zip"
        if best_path.exists():
            model = RecurrentPPO.load(
                best_path,
                env=train_env,
                tensorboard_log=str(root_log_dir),
            )

    # --------------------------------------------------
    # Train
    # --------------------------------------------------
    if trainON:
        callback = EvalCallback(
            eval_env,
            n_eval_episodes=10,
            eval_freq=int(n_steps),
            best_model_save_path=log_dir,
            log_path=log_dir,
            deterministic=True,
            verbose=1,
        )

        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
        )

        model.save(log_dir / "final_model.zip")

    return model, log_dir


if __name__ == "__main__":
    model, log_dir = train_docking_lstm(
        dt=0.5,
        num_env=64,
        trainON=True,
    )
    print("Saved to:", log_dir)
