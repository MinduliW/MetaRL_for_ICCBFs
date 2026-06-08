"""
inspection/train_DCNN.py

Inspection case trainer (feed-forward PPO or LSTM RecurrentPPO), mirroring the
docking trainer style (train_DCNN.py) but pointing to the Inspection environment.

Defaults are chosen to match your existing Inspection scripts:
- mainNNCBFtune.py (MLP/PPO): lr=1e-4, lr_type='D', gamma=0.99, clip=0.2, ent=0.01,
  n_epochs=10, batch_size=64, n_steps_factor=50 -> n_steps=400, layers=4, nodes=256,
  use_sde=True, enable_param_randomisation=True, enableNoise=False, enableCBFtunning=True, dvWeight=10.0
- mainRNNnoisy.py (RNN/RecurrentPPO): lr=5e-5, lr_type='D', same PPO params, use_sde=False,
  plus LSTM hidden=256, n_lstm_layers=1, shared_lstm=False, enable_critic_lstm=True

You can override any of these via function arguments.
"""

from __future__ import annotations

import gc
import warnings
import multiprocessing
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from torch.optim import Adam
from torch.nn import Tanh, ReLU, ELU

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

# Inspection env
from inspection.inspectionEnvNoisy import InspectionEnv

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


def make_env(
    *,
    dt: float,
    deterministic: bool,
    rank: int,
    seed: int,
    enable_param_randomisation: bool,
    enableNoise: bool,
    enableCBFtunning: bool,
    dvWeight: float,
    init_da: bool,
    da_order: int,
    da_vars: int,
):
    """
    Factory for (Subproc|Dummy)VecEnv. Must build a *fresh* env instance per worker.
    """
    def _init():
        env = InspectionEnv(
            dt=dt,
            enable_param_randomisation=enable_param_randomisation,
            enableNoise=enableNoise,
            enableCBFtunning=enableCBFtunning,
            dvWeight=dvWeight,
        )
        if deterministic:
            # If your env exposes a deterministic toggle, set it here.
            # Otherwise, leave as-is (your existing scripts use separate eval env instances).
            pass

        if init_da and _HAS_DA:
            DA.init(int(da_order), int(da_vars))

        env.action_space.seed(seed + rank)
        env.reset(seed=seed + rank)
        return Monitor(env)

    return _init


# ---------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------

def train_inspection(
    *,
    dt: float = 10.0,

    # total timesteps computed as total_episodes * approx_episode_len
    total_episodes: int = 100_000,
    approx_episode_len: int = 400,
    seed: int = 123,

    # ---- Architecture ----
    layers: int = 4,
    nodes: int = 256,
    activation_fn: Literal["tanh", "relu", "elu"] = "tanh",
    policy_type: Literal["PPO", "RNN"] = "PPO",

    # ---- RL hyperparameters ----
    learning_rate: float = 1e-4,                    # MLP default; RNN often uses 5e-5
    lr_type: Literal["C", "D"] = "D",
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    n_epochs: int = 10,
    batch_size: int = 64,
    n_steps: Optional[int] = None,                  # if None, computed from n_steps_factor
    n_steps_factor: int = 50,                       # matches your scripts -> 400 when batch_size=64
    target_kl: Optional[float] = None,              # your scripts do not set target_kl
    vf_coef: float = 0.5,
    use_sde: bool = True,                           # MLP default; RNN default in your script is False

    # ---- RNN-specific ----
    lstm_hidden_size: int = 256,
    n_lstm_layers: int = 1,
    shared_lstm: bool = False,
    enable_critic_lstm: bool = True,

    # ---- Environment knobs (match InspectionEnvNoisy) ----
    enable_param_randomisation: bool = True,
    enableNoise: bool = False,
    enableCBFtunning: bool = True,
    dvWeight: float = 10.0,

    # ---- Environment parallelism ----
    num_env: int = 64,

    # ---- DA init (if your margins use DA) ----
    init_da: bool = True,
    da_order: int = 4,
    da_vars: int = 3,

    # ---- Control ----
    trainON: bool = True,
    trainLoad: bool = False,

    # ---- Logging ----
    root_log_dir: str = "TrainedModels",
):
    """
    Train an Inspection ICCBF meta-policy with configurable architecture.

    Returns:
        (model, log_dir)
    """
    multiprocessing.freeze_support()
    warnings.filterwarnings("ignore", category=UserWarning)
    gc.collect()

    # --------------------------------------------------
    # Rollout length
    # --------------------------------------------------
    if n_steps is None:
        n_steps = int(batch_size * n_steps_factor * 8 / 64)

    # If not training, force single env
    if not trainON:
        num_env = 1

    # --------------------------------------------------
    # Activation + architecture
    # --------------------------------------------------
    act_cls = ACTIVATION_MAP[activation_fn.lower()]
    net_arch = [dict(pi=[nodes] * layers, vf=[nodes] * layers)]

    std = 0.2
    base_policy_kwargs = dict(
        activation_fn=act_cls,
        net_arch=net_arch,
        optimizer_class=Adam,
        ortho_init=True,
        log_std_init=float(np.log(std)),
        share_features_extractor=False,
    )

    # RNN additions (only used for RecurrentPPO)
    rnn_policy_kwargs = dict(
        lstm_hidden_size=int(lstm_hidden_size),
        n_lstm_layers=int(n_lstm_layers),
        shared_lstm=bool(shared_lstm),
        enable_critic_lstm=bool(enable_critic_lstm),
    )

    # --------------------------------------------------
    # Learning-rate schedule
    # --------------------------------------------------
    if lr_type == "D":
        lr_schedule = LinearSchedule(learning_rate, learning_rate / 100)
    else:
        lr_schedule = ConstantSchedule(learning_rate)

    # --------------------------------------------------
    # Environments (train + eval)
    # --------------------------------------------------
    if num_env > 1:
        train_env = SubprocVecEnv(
            [
                make_env(
                    dt=dt,
                    deterministic=False,
                    rank=i,
                    seed=seed,
                    enable_param_randomisation=enable_param_randomisation,
                    enableNoise=enableNoise,
                    enableCBFtunning=enableCBFtunning,
                    dvWeight=dvWeight,
                    init_da=init_da,
                    da_order=da_order,
                    da_vars=da_vars,
                )
                for i in range(num_env)
            ],
            start_method="spawn",
        )
    else:
        # single-process training (use DummyVecEnv for compatibility)
        train_env = DummyVecEnv(
            [
                make_env(
                    dt=dt,
                    deterministic=False,
                    rank=0,
                    seed=seed,
                    enable_param_randomisation=enable_param_randomisation,
                    enableNoise=enableNoise,
                    enableCBFtunning=enableCBFtunning,
                    dvWeight=dvWeight,
                    init_da=init_da,
                    da_order=da_order,
                    da_vars=da_vars,
                )
            ]
        )

    eval_env = InspectionEnv(
        dt=dt,
        enable_param_randomisation=enable_param_randomisation,
        enableNoise=False,                 # keep eval deterministic-ish by default
        enableCBFtunning=enableCBFtunning,
        dvWeight=dvWeight,
    )

    # --------------------------------------------------
    # Naming + logging
    # --------------------------------------------------
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    arch_str = f"{policy_type}_L{layers}_N{nodes}_{activation_fn}"
    hyper_str = f"lr{learning_rate:g}{lr_type}_g{gamma}_gae{gae_lambda}_ent{ent_coef}_clip{clip_range}"
    env_str = f"PR{int(enable_param_randomisation)}_N{int(enableNoise)}_CBF{int(enableCBFtunning)}_dv{dvWeight:g}"
    training_name = f"MetaInspection_{arch_str}_{hyper_str}_{env_str}_{time_str}"

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
            policy_kwargs=base_policy_kwargs,
            tensorboard_log=str(root_log_dir),
            verbose=1,
            seed=int(seed),
        )
    else:
        # If user didn't override, follow your script default for recurrent
        if use_sde is True:
            # your mainRNNnoisy uses use_sde=False
            pass

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
            target_kl=target_kl,
            policy_kwargs={**base_policy_kwargs, **rnn_policy_kwargs},
            tensorboard_log=str(root_log_dir),
            verbose=1,
            seed=int(seed),
        )

    # --------------------------------------------------
    # Load pretrained
    # --------------------------------------------------
    if trainLoad:
        best_path = log_dir / "best_model.zip"
        if best_path.exists():
            if policy_type == "PPO":
                model = PPO.load(best_path, env=train_env, tensorboard_log=str(root_log_dir))
            else:
                model = RecurrentPPO.load(best_path, env=train_env, tensorboard_log=str(root_log_dir))

    # --------------------------------------------------
    # Train
    # --------------------------------------------------
    if trainON:
        callback = EvalCallback(
            eval_env,
            best_model_save_path=log_dir,
            log_path=log_dir,
            eval_freq=int(n_steps),   # vec env => eval every n_steps * num_env timesteps
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
    # Minimal sanity run
    model, log_dir = train_inspection(
        dt=10.0,
        policy_type="PPO",
        total_episodes=100_000,
        approx_episode_len=400,
        num_env=64,
        trainON=True,
        trainLoad=False,
    )
    print("Saved to:", log_dir)
