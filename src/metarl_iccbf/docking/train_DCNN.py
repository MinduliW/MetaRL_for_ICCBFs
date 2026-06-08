# train_DCNN.py
"""
Docking case trainer (feed-forward PPO or RecurrentPPO) mirroring the cruise-control
train_CCNN.py style.

Key differences vs cruise-control:
- Environment import points to the docking env (RLCBFcontrol in RLCBF.py).
- Optional DA.init() hook is kept (many docking ICCBF/DA-margin stacks need it).
"""

import gc
import warnings
import multiprocessing
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from torch.optim import Adam
from torch.nn import Tanh, ReLU, ELU

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

# Docking env (your current docking setup)
from .RLCBF import RLCBFcontrol

# Optional: differential algebra initialisation (safe to keep behind a flag)
try:
    from daceypy import DA  # noqa: F401
    _HAS_DA = True
except Exception:
    _HAS_DA = False


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

ACTIVATION_MAP = {
    "tanh": Tanh,
    "relu": ReLU,
    "elu": ELU,
}


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


def make_env(dt: float, deterministic: bool, rank: int, seed: int,
             init_da: bool, da_order: int, da_vars: int):
    """
    Factory for SubprocVecEnv. Must build a *fresh* env instance per process.
    """
    def _init():
        env = RLCBFcontrol(dt=dt, deterministic=deterministic)
        if init_da and _HAS_DA:
            # DA.init(order, num_variables)
            DA.init(int(da_order), int(da_vars))
        env.action_space.seed(seed + rank)
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init


# ---------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------

def train_docking(
    *,
    dt: float = 0.5,
    total_episodes: int = 200_000,
    approx_episode_len: int = 100,
    seed: int = 123,

    # ---- Architecture ----
    layers: int = 4,
    nodes: int = 64,
    activation_fn: Literal["tanh", "relu", "elu"] = "tanh",
    policy_type: Literal["PPO", "RNN"] = "PPO",

    # ---- RL hyperparameters ----
    learning_rate: float = 5e-5,
    lr_type: Literal["C", "D"] = "C",
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    clip_range: float = 0.1,
    ent_coef: float = 0.01,
    n_epochs: int = 10,
    batch_size: int = 64,
    n_steps: int = 80,
    target_kl: Optional[float] = 0.02,
    vf_coef: float = 0.5,
    use_sde: bool = True,

    # ---- Environment parallelism ----
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
    Train a docking ICCBF meta-policy with configurable network architecture.

    Returns:
        (model, log_dir)
    """

    multiprocessing.freeze_support()
    warnings.filterwarnings("ignore", category=UserWarning)
    gc.collect()

    # --------------------------------------------------
    # Activation + architecture
    # --------------------------------------------------
    act_cls = ACTIVATION_MAP[activation_fn.lower()]
    net_arch = [dict(pi=[nodes] * layers, vf=[nodes] * layers)]

    # SB3 expects float for log_std_init
    std = 0.2
    policy_kwargs = dict(
        activation_fn=act_cls,
        net_arch=net_arch,
        optimizer_class=Adam,
        ortho_init=True,
        log_std_init=float(np.log(std)),
        share_features_extractor=False,
    )

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
    if not trainON:
        num_env = 1

    train_env = SubprocVecEnv([
        make_env(dt, deterministic=False, rank=i, seed=seed,
                 init_da=init_da, da_order=da_order, da_vars=da_vars)
        for i in range(num_env)
    ], start_method="spawn")

    eval_env = RLCBFcontrol(dt=dt, deterministic=True)

    # --------------------------------------------------
    # Naming + logging
    # --------------------------------------------------
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    arch_str = f"{policy_type}_L{layers}_N{nodes}_{activation_fn}"
    hyper_str = f"lr{learning_rate:g}_g{gamma}_gae{gae_lambda}_ent{ent_coef}"
    training_name = f"MetaDocking_{arch_str}_{hyper_str}_{time_str}"

    log_dir = Path(root_log_dir) / training_name
    log_dir.mkdir(parents=True, exist_ok=True)

    total_timesteps = int(total_episodes * approx_episode_len)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    if policy_type == "PPO":
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=lr_schedule,
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
            tensorboard_log=str(root_log_dir),
            verbose=1,
            seed=int(seed),
        )
    else:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            train_env,
            learning_rate=lr_schedule,
            n_steps=int(n_steps),
            batch_size=int(batch_size),
            n_epochs=int(n_epochs),
            gamma=float(gamma),
            gae_lambda=float(gae_lambda),
            clip_range=float(clip_range),
            ent_coef=float(ent_coef),
            vf_coef=float(vf_coef),
            use_sde=bool(use_sde),
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(root_log_dir),
            verbose=1,
            seed=int(seed),
        )

    # --------------------------------------------------
    # Load pretrained
    # --------------------------------------------------
    if trainLoad:
        # Load best_model.zip if it exists; fall back gracefully
        best_path = log_dir / "best_model.zip"
        if best_path.exists():
            if policy_type == "PPO":
                model = PPO.load(best_path, env=train_env)
            else:
                model = RecurrentPPO.load(best_path, env=train_env)

    # --------------------------------------------------
    # Train
    # --------------------------------------------------
    if trainON:
        callback = EvalCallback(
            eval_env,
            best_model_save_path=log_dir,
            log_path=log_dir,
            eval_freq=int(n_steps),   # note: vec env => eval every n_steps * num_env timesteps
            n_eval_episodes=10,
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
    # Minimal example run (edit as needed)
    model, log_dir = train_docking(
        dt=0.5,
        policy_type="PPO",
        num_env=64,
        total_episodes=200_000,
        approx_episode_len=100,
        trainON=True,
        trainLoad=False,
    )
    print("Saved to:", log_dir)
