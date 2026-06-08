"""Quick diagnostic: print per-step reward component statistics over a few episodes
using the actual trained model and CBF tuning enabled — matching the Pareto eval setup."""

import numpy as np
import torch
from metarl_iccbf.inspection.inspectionEnvNoisy import InspectionEnv
from metarl_iccbf.recurrent_cleanrl.ppo import RecurrentPPO

CKPT = "/home/aposadasn/MetaRL_for_ICCBFs/outputs/inspection/Mamba2TunedICCBF_Inspection_20260424_115940/final_model.zip"
BANK = "/home/aposadasn/MetaRL_for_ICCBFs/outputs/data/inspection/inspection_episode_bank.npz"
N_EPISODES = 5

bank  = np.load(BANK, allow_pickle=True)
model = RecurrentPPO.load(CKPT, device="cuda")
model.agent.eval()

env = InspectionEnv(dt=10.0, morl=True, enableCBFtunning=True, enable_param_randomisation=False)

# Fixed weight matching Pareto sweep midpoint
w_np = np.array([0.6, 0.2, 0.2], dtype=np.float32)
w_t  = torch.as_tensor(w_np[None], dtype=torch.float32, device="cuda")

coverage_vals, fuel_vals, safety_vals, raw_min_h_vals = [], [], [], []

for ep in range(N_EPISODES):
    meta = dict(
        m=float(bank["m_vec"][ep]),     R_D=float(bank["R_D_vec"][ep]),
        R_C=float(bank["R_C_vec"][ep]), U_MAX=float(bank["U_MAX_vec"][ep]),
        R_MAX=float(bank["R_MAX_vec"][ep]), r=float(bank["r_vec"][ep]),
    )
    for k, v in meta.items():
        setattr(env, k, v)
    if hasattr(env, "_rebuild_models"):
        env._rebuild_models()

    env.state      = np.asarray(bank["x0s"][ep], dtype=np.float64).copy()
    env.steps_done = 0
    env.inspected  = np.zeros(env.N_POINTS, dtype=bool)
    x_meas = env._noisy_measurement_state(env.state, env.np_random)
    env.inspected |= env.obs_model.get_observation(x_meas)["visible_mask"]
    obs = env._get_obs_vector()

    actor_states, critic_states = model.agent.initial_states(1, "cuda")
    ep_starts = np.ones(1, dtype=np.float32)
    done = False
    step = 0

    while not done:
        obs_t = torch.as_tensor(obs[None], dtype=torch.float32, device="cuda")
        ep_t  = torch.as_tensor(ep_starts, dtype=torch.float32, device="cuda")
        with torch.no_grad():
            action, actor_states, critic_states = model.agent.deterministic_step(
                obs_t, actor_states, critic_states, ep_t, w=w_t,
            )
        action_np = np.clip(action.squeeze(0).cpu().numpy(),
                            env.action_space.low, env.action_space.high)
        obs, r_vec, terminated, truncated, info = env.step(action_np)
        coverage_vals.append(float(r_vec[0]))
        fuel_vals.append(float(r_vec[1]))
        safety_vals.append(float(r_vec[2]))
        raw_min_h_vals.append(float(env.last_min_h))
        done = bool(terminated or truncated)
        ep_starts = np.zeros(1, dtype=np.float32)
        step += 1

    print(f"  ep {ep+1}: {step} steps, inspected={info['num_inspected']}")

coverage_vals  = np.array(coverage_vals)
fuel_vals      = np.array(fuel_vals)
safety_vals    = np.array(safety_vals)
raw_min_h_vals = np.array(raw_min_h_vals)

def stats(name, arr):
    print(f"  {name:20s}  min={arr.min():.4f}  max={arr.max():.4f}  "
          f"mean={arr.mean():.4f}  std={arr.std():.4f}  "
          f"nonzero={np.count_nonzero(arr)}/{len(arr)}")

print("\n--- Per-step reward component stats (trained agent, CBFtunning=True) ---")
stats("coverage  r[0]", coverage_vals)
stats("fuel      r[1]", fuel_vals)
stats("safety    r[2]", safety_vals)
stats("raw min_h      ", raw_min_h_vals)
print(f"\n  Cumulative per episode (~{len(coverage_vals)//N_EPISODES} steps):")
n = len(coverage_vals) // N_EPISODES
print(f"  coverage  sum≈{coverage_vals.sum()/N_EPISODES:.2f}")
print(f"  fuel      sum≈{fuel_vals.sum()/N_EPISODES:.2f}")
print(f"  safety    sum≈{safety_vals.sum()/N_EPISODES:.2f}")
print(f"  raw min_h sum≈{raw_min_h_vals.sum()/N_EPISODES:.2f}")
