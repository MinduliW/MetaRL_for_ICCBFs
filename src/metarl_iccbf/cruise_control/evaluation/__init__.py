# Lazy imports: avoid loading heavy torch/SB3 deps at package import time,
# which would trigger them in every spawned worker process.

def __getattr__(name):
    if name == "evaluate_serial":
        from metarl_iccbf.cruise_control.evaluation.eval_serial import evaluate_serial
        return evaluate_serial
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["evaluate_serial"]
