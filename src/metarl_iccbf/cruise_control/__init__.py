def __getattr__(name):
    if name == "RLCBFcontrol":
        from metarl_iccbf.cruise_control.envs.rlcbf_env import RLCBFcontrol
        return RLCBFcontrol
    if name == "ICCBF":
        from metarl_iccbf.cruise_control.iccbf import ICCBF
        return ICCBF
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
