#!/usr/bin/env python3
"""
validate_mc_da_bounds_boundary_cruise.py

Cruise-control analogue of the docking boundary-slack validator.

For each Monte-Carlo episode e and time index t (with stride):
  1) Center x0 := states[e,t,:] = [d, v].
  2) Estimate per-episode physical parameters (v0, m, ulim) from stored trajectories
     unless overridden by CLI flags.
  3) Build conservative DA interval bounds over a local box B(x0, half_width) for:
        - hICCBF (the 2nd-layer ICCBF barrier, a.k.a. b2)
        - LfhICCBF (drift Lie derivative of hICCBF)
        - LghICCBF (control Lie derivative of hICCBF)
     accounting conservatively for the u_inf branching through max-abs enclosures.
  4) Sample M points on the boundary of the box (random vertices by default).
  5) Evaluate the "true" quantities at those boundary points (using the same
     closed-form expressions as in your Cruise Control environment).
  6) Compute boundary slack to the DA bounds.

Outputs:
  - mc_bounds_slack_data.mat
  - mc_bounds_worstcase_boundary_slack.png

Run:
  python validate_mc_da_bounds_boundary_cruise.py --mat NoiseMetaICCBFNN.mat --out mc_bounds_out --M 10 --stride 5

Notes:
  - This script assumes the .mat has at least: states (E,T,2), tvec_full (T,),
    uOpts (E,T) or (E,T,1), and actionStore (E,T,4) where actionStore[...,0:2]
    are already scaled (as in your saved logs).
  - If your saved actionStore contains the *unscaled* actions, pass --actions_unscaled
    so the script applies the same scaling as env.step() before using a1,a2.
"""

import os
import argparse
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import math

from daceypy import DA


# -----------------------------
# Robust MAT loader
# -----------------------------
def _pick_first_existing(dct, names):
    for n in names:
        if n in dct:
            return n, dct[n]
    return None, None


def load_rollouts(mat_path: str):
    D = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    k_states, states = _pick_first_existing(D, ["states", "X", "x", "traj_states", "state_hist"])
    if states is None:
        raise KeyError(f"Could not find states in {mat_path}. Tried: states, X, x, traj_states, state_hist")
    states = np.asarray(states, dtype=float)
    if states.ndim != 3 or states.shape[-1] != 2:
        raise ValueError(f"Expected states shape (E,T,2) (or equivalent), got {states.shape}")

    k_t, tvec = _pick_first_existing(D, ["tvec_full", "t", "time", "ts", "Tvec", "time_vec"])
    if tvec is None:
        tvec = np.arange(states.shape[1], dtype=float)
    else:
        tvec = np.asarray(tvec, dtype=float).reshape(-1,)
        if tvec.size != states.shape[1]:
            # last resort
            tvec = np.arange(states.shape[1], dtype=float)

    k_u, uOpts = _pick_first_existing(D, ["uOpts", "u", "u_store", "u_hist", "uOpt"])
    if uOpts is None:
        raise KeyError(f"Could not find uOpts in {mat_path}. Needed for parameter estimation and/or ulim. Tried: uOpts, u, u_store, u_hist, uOpt")
    uOpts = np.asarray(uOpts, dtype=float)
    if uOpts.ndim == 3 and uOpts.shape[-1] == 1:
        uOpts = uOpts[..., 0]
    if uOpts.ndim != 2:
        raise ValueError(f"Expected uOpts shape (E,T) or (E,T,1), got {uOpts.shape}")

    k_a, actionStore = _pick_first_existing(D, ["actionStore", "actions", "a_store", "policy_actions"])
    if actionStore is None:
        raise KeyError(f"Could not find actionStore in {mat_path}. Tried: actionStore, actions, a_store, policy_actions")
    actionStore = np.asarray(actionStore, dtype=float)
    if actionStore.ndim != 3 or actionStore.shape[-1] < 2:
        raise ValueError(f"Expected actionStore shape (E,T,>=2), got {actionStore.shape}")

    return states, tvec, uOpts, actionStore


# -----------------------------
# DA interval helpers
# -----------------------------
def interval_maxabs(I):
    for a, b in [("lb", "ub"), ("lower", "upper"), ("l", "u"), ("inf", "sup")]:
        if hasattr(I, a) and hasattr(I, b):
            return max(abs(float(getattr(I, a))), abs(float(getattr(I, b))))
    return max(abs(float(I[0])), abs(float(I[1])))


def bound_over_box(poly: DA, half_width: np.ndarray):
    """
    Bound a DA polynomial over a local box around the expansion point:
      d = d0 + δd, v = v0 + δv,  with δd in [-hw_d, hw_d], δv in [-hw_v, hw_v]
    Achieved by scaling DA variables to [-1,1]^2 before calling bound().
    """
    hw = np.asarray(half_width, dtype=float).reshape(2,)
    p = poly.scaleVariable(1, float(hw[0])).scaleVariable(2, float(hw[1]))
    I = p.bound()
    # Return conservative [lb,ub]
    if hasattr(I, "lb") and hasattr(I, "ub"):
        return float(I.lb), float(I.ub)
    # fallback indexing
    return float(I[0]), float(I[1])


def maxabs_over_box(poly: DA, half_width: np.ndarray):
    lb, ub = bound_over_box(poly, half_width)
    return max(abs(lb), abs(ub))


# -----------------------------
# Cruise-control ICCBF (true eval) — vectorised
# -----------------------------
def cruise_iccbf_true_batch(X: np.ndarray,
                            a1: np.ndarray,
                            a2: np.ndarray,
                            f0: float,
                            f1: float,
                            f2: float,
                            m: float,
                            g0: float,
                            v0_front: float,
                            ulim: float,
                            eps: float = 1e-8):
    """
    Vectorised "true" evaluation matching the closed-form implementation in RLCBFcontrol.getICCBFvars(...).

    X: (N,2) with columns [d, v]
    a1, a2: (N,) scalar gains used in k1 and k2
    Returns dict: hICCBF (=b2), LfhICCBF, LghICCBF, plus Lgb1 (for u_inf sign)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"X must be (N,2), got {X.shape}")
    d = X[:, 0]
    v = X[:, 1]

    a1 = np.asarray(a1, dtype=float).reshape(-1,)
    a2 = np.asarray(a2, dtype=float).reshape(-1,)
    if a1.size != d.size or a2.size != d.size:
        raise ValueError("a1 and a2 must have same length as X batch")

    # Dynamics
    F = f0 + f1 * v + f2 * (v ** 2)
    Fv = f1 + 2.0 * f2 * v
    d_dot = v0_front - v
    v_dot = -(F / m)  # drift only for Lie derivatives

    # h and Lf h, Lg h
    h = d - 1.8 * v
    Lfh = d_dot + 1.8 * (F / m)
    Lgh = -1.8 * g0

    # b1 = Lfh + Lgh*ulim + a1*h   (matches your DA construction)
    b1 = Lfh + Lgh * ulim + a1 * h

    # Derivatives of b1
    A = 1.8 / m
    # b1_d = a1
    b1_d  = a1
    # b1_v = d/dv [Lfh] + d/dv [a1*h]  (Lgh*ulim constant)
    # Lfh_v = -1 + 1.8/m * (f1 + 2 f2 v)
    b1_v  = (-1.0 + A * (f1 + 2.0 * f2 * v)) + a1 * (-1.8)

    # Second derivatives needed for b2 derivatives
    b1_dd = np.zeros_like(v)
    b1_dv = np.zeros_like(v)
    b1_vd = np.zeros_like(v)
    b1_vv = 2.0 * A * f2 + np.zeros_like(v)  # derivative of (-1 + A(f1+2f2 v)) is 2A f2 ; a1*(-1.8) constant

    # Lg b1
    Lgb1 = g0 * b1_v

    # Worst-case u_inf per point
    u1inf = np.where(Lgb1 > 0.0, -abs(ulim), abs(ulim))

    # Lf b1
    Lfb1 = b1_d * d_dot + b1_v * v_dot

    # k2(b1) = a2 * sqrt(b1) if b1>0 else 0  (matches getICCBFvars_dace)
    b1_pos = np.maximum(b1, 0.0)
    k2 = a2 * np.sqrt(b1_pos)

    # Need derivatives of k2 where b1>0. Use safe epsilon.
    mask = b1 > 0.0
    denom = np.sqrt(np.maximum(b1, eps))
    # k2_d = a2 * 0.5 * b1^{-1/2} * b1_d
    k2_d = np.zeros_like(v)
    k2_v = np.zeros_like(v)
    k2_d[mask] = a2[mask] * 0.5 * (b1_d[mask] / denom[mask])
    k2_v[mask] = a2[mask] * 0.5 * (b1_v[mask] / denom[mask])

    # Derivatives of Lg b1 (since Lgb1 = g0*b1_v)
    Lgb1_d = g0 * b1_vd
    Lgb1_v = g0 * b1_vv

    # Derivatives of Lf b1
    # d_dot_d=0, d_dot_v=-1
    # v_dot_d=0, v_dot_v=-Fv/m
    Lfb1_d = b1_dd * d_dot + b1_vd * v_dot
    Lfb1_v = b1_dv * d_dot - b1_d + b1_vv * v_dot - b1_v * (Fv / m)

    # b2 derivatives
    b2_d = Lfb1_d + Lgb1_d * u1inf + k2_d
    b2_v = Lfb1_v + Lgb1_v * u1inf + k2_v

    # b2 itself
    b2 = Lfb1 + Lgb1 * u1inf + k2

    LghICCBF = g0 * b2_v
    LfhICCBF = b2_d * d_dot + b2_v * v_dot

    return {
        "hICCBF": b2,
        "LfhICCBF": LfhICCBF,
        "LghICCBF": LghICCBF,
        "Lgb1": Lgb1,
        "b1": b1,
        "u1inf": u1inf,
    }


# -----------------------------
# Conservative DA bounds at a single x0
# -----------------------------
def conservative_bounds_at_x0(center: np.ndarray,
                              half_width: np.ndarray,
                              a1: float,
                              a2: float,
                              f0: float,
                              f1: float,
                              f2: float,
                              m: float,
                              g0: float,
                              v0_front: float,
                              ulim: float,
                              da_order: int,
                              eps_zero: float = 1e-16):
    """
    Build conservative bounds for hICCBF, LfhICCBF, LghICCBF over the local box.

    Key idea:
      b2 = (Lfb1 + k2) + (Lgb1) * u_inf
    where u_inf is sign-selected; conservatively enclose with maxabs(Lgb1)*ulim.
    The same idea is used for Lgh and Lfh via the u_inf-dependent derivative terms.

    Returns:
      dict with bounds tuples and branch_info flags.
    """
    center = np.asarray(center, dtype=float).reshape(2,)
    hw = np.asarray(half_width, dtype=float).reshape(2,)
    d0, v0 = float(center[0]), float(center[1])

    DA.init(int(da_order), 2)
    dd = DA(1)
    dv = DA(2)
    d = d0 + dd
    v = v0 + dv

    # Dynamics (drift)
    F = f0 + f1 * v + f2 * (v * v)
    d_dot = v0_front - v
    v_dot0 = -(F / m)

    # h, Lf h, Lg h
    h = d - 1.8 * v
    Lfh = d_dot + 1.8 * (F / m)
    Lgh = -1.8 * g0

    # b1
    b1 = Lfh + Lgh * ulim + float(a1) * h
    b1_d = b1.deriv(1)
    b1_v = b1.deriv(2)

    # Lg b1
    Lgb1 = g0 * b1_v
    Lgb1_d = Lgb1.deriv(1)
    Lgb1_v = Lgb1.deriv(2)

    # Lf b1
    Lfb1 = b1_d * d_dot + b1_v * v_dot0
    Lfb1_d = Lfb1.deriv(1)
    Lfb1_v = Lfb1.deriv(2)

    # k2: follow your DA freezing: if b1(x0)>=0, include a2*sqrt(b1), else 0
    b1_x0 = float(b1.eval([0.0, 0.0]))
    if b1_x0 >= 0.0:
        k2 = float(a2) * (b1 ** 0.5)
    else:
        k2 = DA(0.0)

    # b2_no_u and its derivatives
    b2_no_u = Lfb1 + k2
    b2no_lb, b2no_ub = bound_over_box(b2_no_u, hw)

    # conservative u_inf enclosure for b2
    maxabs_Lgb1 = maxabs_over_box(Lgb1, hw)
    b2_lb = b2no_lb - abs(ulim) * maxabs_Lgb1
    b2_ub = b2no_ub + abs(ulim) * maxabs_Lgb1

    # LghICCBF:
    # b2_v = (b2_no_u)_v + (Lgb1_v)*u_inf
    b2no_v = b2_no_u.deriv(2)
    Lgh_no_u = g0 * b2no_v
    Lghno_lb, Lghno_ub = bound_over_box(Lgh_no_u, hw)

    maxabs_Lgb1_v = maxabs_over_box(Lgb1_v, hw)
    # u_inf term contributes g0 * Lgb1_v * u_inf
    Lgh_lb = Lghno_lb - abs(g0) * abs(ulim) * maxabs_Lgb1_v
    Lgh_ub = Lghno_ub + abs(g0) * abs(ulim) * maxabs_Lgb1_v

    # LfhICCBF:
    # b2_d = (b2_no_u)_d + (Lgb1_d)*u_inf
    # b2_v = (b2_no_u)_v + (Lgb1_v)*u_inf
    b2no_d = b2_no_u.deriv(1)

    # drift Lie derivative without u_inf
    Lfh_no_u = b2no_d * d_dot + b2no_v * v_dot0
    Lfhno_lb, Lfhno_ub = bound_over_box(Lfh_no_u, hw)

    # u_inf-dependent contribution:
    term_u = (Lgb1_d * d_dot + Lgb1_v * v_dot0)  # multiplies u_inf
    maxabs_term_u = maxabs_over_box(term_u, hw)

    Lfh_lb = Lfhno_lb - abs(ulim) * maxabs_term_u
    Lfh_ub = Lfhno_ub + abs(ulim) * maxabs_term_u

    # Branch diagnostics
    Lgb1_lb, Lgb1_ub = bound_over_box(Lgb1, hw)
    b1_lb, b1_ub = bound_over_box(b1, hw)

    branch_info = {
        "Lgb1_interval": (float(Lgb1_lb), float(Lgb1_ub)),
        "b1_interval": (float(b1_lb), float(b1_ub)),
        "crosses_Lgb1_zero": bool((float(Lgb1_lb) <= eps_zero) and (float(Lgb1_ub) >= -eps_zero)),
        "crosses_b1_zero": bool((float(b1_lb) <= eps_zero) and (float(b1_ub) >= -eps_zero)),
        "b1_x0": float(b1_x0),
    }

    return {
        "hICCBF": (float(b2_lb), float(b2_ub)),
        "LfhICCBF": (float(Lfh_lb), float(Lfh_ub)),
        "LghICCBF": (float(Lgh_lb), float(Lgh_ub)),
        "branch_info": branch_info,
    }


def slack_to_interval(vals: np.ndarray, lb: float, ub: float):
    vals = np.asarray(vals, dtype=float)
    return np.minimum(vals - float(lb), float(ub) - vals)


# -----------------------------
# Boundary sampling (2D vertices)
# -----------------------------
def sample_box_boundary_vertices(x0: np.ndarray, half_width: np.ndarray, M: int, rng: np.random.Generator):
    x0 = np.asarray(x0, float).reshape(2,)
    hw = np.asarray(half_width, float).reshape(2,)
    signs = rng.choice([-1.0, +1.0], size=(int(M), 2))
    return x0[None, :] + signs * hw[None, :]


# -----------------------------
# Per-episode parameter estimation
# -----------------------------
def estimate_episode_params(states_e: np.ndarray,
                            u_e: np.ndarray,
                            tvec: np.ndarray,
                            f0: float,
                            f1: float,
                            f2: float,
                            g0: float,
                            default_v0: float,
                            default_m: float,
                            default_ulim: float):
    """
    Estimate (v0_front, m, ulim) from trajectory data.
    This is useful because your env reset meta-randomises parameters but doesn't
    store them in the .mat.

    - ulim: max |u|
    - v0_front: median(d_dot + v)
    - m: median( -F / (v_dot - g0*u) )
    """
    d = states_e[:, 0]
    v = states_e[:, 1]
    u = np.asarray(u_e, dtype=float).reshape(-1,)

    # dt from time vector (robust)
    dt = float(np.median(np.diff(tvec)))
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.1

    # derivatives
    d_dot = np.gradient(d, dt)
    v_dot = np.gradient(v, dt)

    # ulim estimate
    ulim_est = float(np.nanmax(np.abs(u)))
    if not np.isfinite(ulim_est) or ulim_est <= 1e-9:
        ulim_est = float(default_ulim)

    # v0 estimate (front-car speed)
    v0_est = np.median(d_dot + v)
    if not np.isfinite(v0_est):
        v0_est = float(default_v0)

    # m estimate
    F = f0 + f1 * v + f2 * (v ** 2)
    denom = (v_dot - g0 * u)
    mask = np.abs(denom) > 1e-3  # avoid division noise
    if np.any(mask):
        m_samples = (-F[mask] / denom[mask])
        # keep physically plausible positive masses
        m_samples = m_samples[np.isfinite(m_samples) & (m_samples > 100.0) & (m_samples < 1e6)]
        if m_samples.size > 20:
            m_est = float(np.median(m_samples))
        else:
            m_est = float(default_m)
    else:
        m_est = float(default_m)

    return v0_est, m_est, ulim_est


# -----------------------------
# Main routine
# -----------------------------
def validate_mc_boundary(mat_path: str,
                         out_dir: str,
                         half_width: np.ndarray,
                         da_order: int,
                         stride: int,
                         M: int,
                         seed: int,
                         eps_zero: float,
                         actions_unscaled: bool,
                         use_param_estimates: bool,
                         v0_nom: float,
                         m_nom: float,
                         ulim_nom: float,
                         f0: float,
                         f1: float,
                         f2: float,
                         g0: float):
    os.makedirs(out_dir, exist_ok=True)

    states, tvec, uOpts, actionStore = load_rollouts(mat_path)
    E, T, _ = states.shape

    tidx = np.arange(0, T, int(stride))
    tplot = tvec[tidx]

    hw = np.asarray(half_width, dtype=float).reshape(2,)
    rng = np.random.default_rng(int(seed))

    # Extract actions -> a1,a2
    acts = actionStore[..., :2].copy()  # (E,T,2)
    if actions_unscaled:
        # env.step scaling:
        # action[0] = 10*(a+1)/2 ; action[1] = 10*(a+1)/2
        acts[..., 0] = 10.0 * (acts[..., 0] + 1.0) / 2.0
        acts[..., 1] = 10.0 * (acts[..., 1] + 1.0) / 2.0

    # Storage
    slack_h_boundary  = np.full((E, len(tidx)), np.nan, dtype=float)
    slack_Lf_boundary = np.full((E, len(tidx)), np.nan, dtype=float)
    slack_Lg_boundary = np.full((E, len(tidx)), np.nan, dtype=float)

    slack_h_centre  = np.full((E, len(tidx)), np.nan, dtype=float)
    slack_Lf_centre = np.full((E, len(tidx)), np.nan, dtype=float)
    slack_Lg_centre = np.full((E, len(tidx)), np.nan, dtype=float)

    crosses_Lgb1 = np.zeros((E, len(tidx)), dtype=bool)
    crosses_b1   = np.zeros((E, len(tidx)), dtype=bool)

    # Pre-estimate episode params if requested
    episode_params = []
    if use_param_estimates:
        for e in range(E):
            v0_e, m_e, ulim_e = estimate_episode_params(states[e, :, :], uOpts[e, :], tvec, f0, f1, f2, g0, v0_nom, m_nom, ulim_nom)
            episode_params.append((v0_e, m_e, ulim_e))
    else:
        episode_params = [(v0_nom, m_nom, ulim_nom)] * E

    # Main loops
    for j, k in enumerate(tidx):
        Xk = states[:, k, :]         # (E,2)
        a1k = acts[:, k, 0]          # (E,)
        a2k = acts[:, k, 1]          # (E,)

        for e in range(E):
            x0 = Xk[e, :]
            a1 = float(a1k[e])
            a2 = float(a2k[e])

            v0_front, m_e, ulim_e = episode_params[e]

            # Conservative DA bounds for this box
            bnd = conservative_bounds_at_x0(
                center=x0,
                half_width=hw,
                a1=a1,
                a2=a2,
                f0=f0, f1=f1, f2=f2,
                m=m_e, g0=g0, v0_front=v0_front,
                ulim=ulim_e,
                da_order=da_order,
                eps_zero=eps_zero
            )

            lh_lb, lh_ub = bnd["hICCBF"]
            lLf_lb, lLf_ub = bnd["LfhICCBF"]
            lLg_lb, lLg_ub = bnd["LghICCBF"]

            # Centre true values (single point)
            trC = cruise_iccbf_true_batch(
                X=x0.reshape(1, 2),
                a1=np.array([a1], dtype=float),
                a2=np.array([a2], dtype=float),
                f0=f0, f1=f1, f2=f2,
                m=m_e, g0=g0, v0_front=v0_front,
                ulim=ulim_e
            )

            hC  = float(trC["hICCBF"][0])
            LfC = float(trC["LfhICCBF"][0])
            LgC = float(trC["LghICCBF"][0])

            slack_h_centre[e, j]  = float(min(hC - lh_lb,  lh_ub - hC))
            slack_Lf_centre[e, j] = float(min(LfC - lLf_lb, lLf_ub - LfC))
            slack_Lg_centre[e, j] = float(min(LgC - lLg_lb, lLg_ub - LgC))

            # Boundary points (vertices)
            pts = sample_box_boundary_vertices(x0, hw, M=M, rng=rng)  # (M,2)
            trB = cruise_iccbf_true_batch(
                X=pts,
                a1=np.full(M, a1, dtype=float),
                a2=np.full(M, a2, dtype=float),
                f0=f0, f1=f1, f2=f2,
                m=m_e, g0=g0, v0_front=v0_front,
                ulim=ulim_e
            )

            sh  = slack_to_interval(trB["hICCBF"],   lh_lb,  lh_ub)
            sLf = slack_to_interval(trB["LfhICCBF"], lLf_lb, lLf_ub)
            sLg = slack_to_interval(trB["LghICCBF"], lLg_lb, lLg_ub)

            slack_h_boundary[e, j]  = float(np.min(sh))
            slack_Lf_boundary[e, j] = float(np.min(sLf))
            slack_Lg_boundary[e, j] = float(np.min(sLg))

            bi = bnd["branch_info"]
            crosses_Lgb1[e, j] = bool(bi["crosses_Lgb1_zero"])
            crosses_b1[e, j]   = bool(bi["crosses_b1_zero"])

        if (j + 1) % max(1, len(tidx) // 10) == 0:
            print(f"[progress] checked time index {j+1}/{len(tidx)}")

    # Worst-case over episodes
    wc_h  = np.nanmin(slack_h_boundary, axis=0)
    wc_Lf = np.nanmin(slack_Lf_boundary, axis=0)
    wc_Lg = np.nanmin(slack_Lg_boundary, axis=0)

    # Summary
    def _summ(slack, name):
        slack = np.asarray(slack, float)
        return (name, float(np.nanmin(slack)), int(np.sum(slack < 0.0)), float(np.mean(slack < 0.0)))

    summary = [
        _summ(slack_h_boundary, "hICCBF"),
        _summ(slack_Lf_boundary, "LfhICCBF"),
        _summ(slack_Lg_boundary, "LghICCBF"),
    ]

    print("\n=== Cruise boundary-sampled conservative DA-bound validation (slack<0 => violation) ===")
    print(f"File: {mat_path}")
    print(f"Episodes={E}, steps={T}, stride={stride}, checked_steps={len(tidx)}, M_boundary={M}")
    print(f"half_width={hw.tolist()}, da_order={da_order}, seed={seed}")
    print(f"use_param_estimates={use_param_estimates} (v0,m,ulim estimated from trajectories)")
    for (nm, mn, nv, frac) in summary:
        print(f"{nm:>8s}: min_slack={mn:+.3e}  violations={nv}  frac={frac:.3e}")

    # Plot (3x1)
    fig, axs = plt.subplots(3, 1, figsize=(11.0, 8.0), sharex=True)
    series = [
        ("hICCBF: worst-case boundary slack over episodes", wc_h),
        ("LfhICCBF: worst-case boundary slack over episodes", wc_Lf),
        ("LghICCBF: worst-case boundary slack over episodes", wc_Lg),
    ]
    for ax, (title, s) in zip(axs, series):
        ax.plot(tplot, s, linewidth=1.5)
        ax.axhline(0.0, linestyle="--", linewidth=1.0)
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
        ax.set_ylabel("min slack (boundary)")
    axs[-1].set_xlabel("time")

    fig.tight_layout()
    fig_path = os.path.join(out_dir, "mc_bounds_worstcase_boundary_slack.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    out_mat = {
        "t": tplot.reshape(-1, 1),
        "stride": int(stride),
        "half_width": hw.reshape(1, -1),
        "da_order": int(da_order),
        "eps_zero": float(eps_zero),
        "M_boundary": int(M),
        "seed_boundary": int(seed),
        "actions_unscaled": bool(actions_unscaled),
        "use_param_estimates": bool(use_param_estimates),

        "slack_h_boundary": slack_h_boundary,
        "slack_Lfh_boundary": slack_Lf_boundary,
        "slack_Lgh_boundary": slack_Lg_boundary,

        "slack_h_centre": slack_h_centre,
        "slack_Lfh_centre": slack_Lf_centre,
        "slack_Lgh_centre": slack_Lg_centre,

        "crosses_Lgb1_zero": crosses_Lgb1,
        "crosses_b1_zero": crosses_b1,
    }
    mat_out_path = os.path.join(out_dir, "mc_bounds_slack_data.mat")
    sio.savemat(mat_out_path, out_mat, do_compression=True)

    print(f"\nSaved plot : {fig_path}")
    print(f"Saved .mat : {mat_out_path}")
    return fig_path, mat_out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mat", type=str, required=True, help="Path to .mat containing cruise rollouts (states,uOpts,actionStore,tvec_full)")
    p.add_argument("--out", type=str, default="mc_bounds_out", help="Output directory")
    p.add_argument("--stride", type=int, default=5, help="Check every stride-th time step (increase for speed)")
    p.add_argument("--da_order", type=int, default=4, help="DA truncation order (DA.init(order, 2))")
    p.add_argument("--eps_zero", type=float, default=1e-16)
    p.add_argument("--M", type=int, default=10, help="Boundary points (vertices) per (episode,time)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for boundary sampling")
    p.add_argument("--hw", type=float, nargs=2, default=[2.0, 2.0],
                   help="Half-width box around each x0: [Δd, Δv]")

    # Action scaling
    p.add_argument("--actions_unscaled", action="store_true",
                   help="If set, actionStore contains raw [-1,1] actions; apply env.step scaling to get a1,a2.")

    # Parameter estimation
    p.add_argument("--no_param_estimates", action="store_true",
                   help="Disable per-episode parameter estimation; use nominal v0,m,ulim instead.")

    # Nominals (used if estimation disabled, or as fallbacks)
    p.add_argument("--v0_nom", type=float, default=13.89, help="Nominal front-car speed v0")
    p.add_argument("--m_nom", type=float, default=1650.0, help="Nominal mass m")
    p.add_argument("--ulim_nom", type=float, default=0.25, help="Nominal control limit ulim")

    # Model constants (your defaults)
    p.add_argument("--f0", type=float, default=0.1)
    p.add_argument("--f1", type=float, default=5.0)
    p.add_argument("--f2", type=float, default=0.25)
    p.add_argument("--g0", type=float, default=9.81)

    args = p.parse_args()

    validate_mc_boundary(
        mat_path=args.mat,
        out_dir=args.out,
        half_width=np.array(args.hw, dtype=float),
        da_order=int(args.da_order),
        stride=int(args.stride),
        M=int(args.M),
        seed=int(args.seed),
        eps_zero=float(args.eps_zero),
        actions_unscaled=bool(args.actions_unscaled),
        use_param_estimates=not bool(args.no_param_estimates),
        v0_nom=float(args.v0_nom),
        m_nom=float(args.m_nom),
        ulim_nom=float(args.ulim_nom),
        f0=float(args.f0),
        f1=float(args.f1),
        f2=float(args.f2),
        g0=float(args.g0),
    )


if __name__ == "__main__":
    main()
