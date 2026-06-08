"""Backward-compat shim – old checkpoints were pickled with references to this module.

Everything now lives in ``metarl_iccbf.inspection.training.train``.
"""

from metarl_iccbf.inspection.training.train import (  # noqa: F401
    ConstantSchedule,
    LinearSchedule,
    _make_env,
    train_inspection,
)
