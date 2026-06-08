from .ppo import RecurrentPPO, Mamba2PPO
from .agent import Mamba2ActorCritic, GRUActorCritic

# Lazy imports to avoid pulling in mamba_ssm (CUDA) at import time
def __getattr__(name):
    if name == "train_recurrent":
        from .train import train_recurrent
        return train_recurrent
    if name in ("Mamba2SAC", "RecurrentSAC"):
        from .sac import RecurrentSAC
        return RecurrentSAC
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
