"""Pareto front evaluation for MORL-trained Mamba2PPO policies.

Sweeps preference weights w over the 2-objective simplex, runs deterministic
rollouts for each weight, and returns the resulting reward vectors to trace
the Pareto front.

Usage::

    from metarl_iccbf.recurrent_cleanrl.pareto_eval import sweep_pareto_front

    results = sweep_pareto_front(
        model,
        env_fn=lambda: InspectionEnv(dt=10.0, morl=True),
        n_weights=20,
        n_episodes_per_weight=5,
        wandb_run=wandb.run,
    )
    # results["weights"]       (n_weights, 2)
    # results["mean_r_vec"]    (n_weights, 2)  mean reward per objective
    # results["all_r_vecs"]    list of (n_eps, 2) arrays per weight
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import torch

try:
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
except ImportError as e:
    raise ImportError("gymnasium and stable-baselines3 are required") from e


def sweep_pareto_front(
    model: Any,
    env_fn: Callable[[], gym.Env],
    *,
    n_weights: int = 20,
    n_episodes_per_weight: int = 5,
    weight_lo: float = 0.0,
    weight_hi: float = 1.0,
    wandb_run: Optional[Any] = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Sweep preference weights over the 2-objective Pareto simplex.

    Parameters
    ----------
    model : Mamba2PPO
        A trained MORL model (``model.morl`` must be ``True``).
    env_fn : callable
        Zero-argument factory returning a mo-gymnasium-compatible env
        (``step()`` returns ``r_vec`` of shape ``(2,)``).
    n_weights : int
        Number of evenly-spaced weights to evaluate.
    n_episodes_per_weight : int
        Episodes to average per weight point.
    weight_lo, weight_hi : float
        Range for w_0 (w_1 = 1 − w_0).
    wandb_run : optional W&B run
        If provided, logs the Pareto scatter table and a plot artifact.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys:
        ``weights``     (n_weights, 2) float32 array
        ``mean_r_vec``  (n_weights, 2) mean reward vector per weight
        ``all_r_vecs``  list[np.ndarray]  per-episode reward vectors per weight
    """
    assert getattr(model, "morl", False), "model.morl must be True"
    assert model.morl_n_objectives == 2, "sweep_pareto_front supports 2 objectives only"

    device = model.device
    agent = model.agent

    w0_vals = np.linspace(weight_lo, weight_hi, n_weights, dtype=np.float32)
    weights = np.stack([w0_vals, 1.0 - w0_vals], axis=-1)  # (n_weights, 2)

    mean_r_vecs: list[np.ndarray] = []
    all_r_vecs: list[np.ndarray] = []

    for wi, w in enumerate(weights):
        ep_r_vecs: list[np.ndarray] = []

        for _ in range(n_episodes_per_weight):
            env = env_fn()
            obs, _ = env.reset()
            actor_states, critic_states = agent.initial_states(1, device)
            episode_starts = np.ones(1, dtype=np.float32)

            ep_r_vec = np.zeros(2, dtype=np.float32)
            done = False

            agent.eval()
            while not done:
                # Augment obs with preference weight
                obs_aug = np.concatenate([obs[None], w[None]], axis=-1)  # (1, obs_dim+2)
                obs_t = torch.as_tensor(obs_aug, dtype=torch.float32, device=device)
                ep_t = torch.as_tensor(episode_starts, dtype=torch.float32, device=device)

                with torch.no_grad():
                    actor_states = agent._reset_states(actor_states, ep_t)
                    critic_states = agent._reset_states(critic_states, ep_t)

                    a_proj = agent.proj_actor(obs_t)
                    a_out, a_conv, a_ssm = agent.mamba_actor.step(
                        a_proj.unsqueeze(1), actor_states[0], actor_states[1],
                    )
                    a_out = a_out.squeeze(1)
                    a_latent = agent.mlp_actor(a_out)
                    action = agent.action_mean(a_latent)
                    actor_states = (a_conv, a_ssm)

                    c_proj = agent.proj_critic(obs_t)
                    _, c_conv, c_ssm = agent.mamba_critic.step(
                        c_proj.unsqueeze(1), critic_states[0], critic_states[1],
                    )
                    critic_states = (c_conv, c_ssm)

                action_np = action.squeeze(0).cpu().numpy()
                if hasattr(env, "action_space"):
                    action_np = np.clip(
                        action_np, env.action_space.low, env.action_space.high,
                    )

                obs, r_vec, terminated, truncated, _ = env.step(action_np)
                ep_r_vec += np.asarray(r_vec, dtype=np.float32)
                done = bool(terminated or truncated)
                episode_starts = np.ones(1, dtype=np.float32) if done else np.zeros(1, dtype=np.float32)

            ep_r_vecs.append(ep_r_vec)

        ep_arr = np.stack(ep_r_vecs)          # (n_episodes, 2)
        all_r_vecs.append(ep_arr)
        mean_r_vecs.append(ep_arr.mean(axis=0))

        if verbose:
            m = mean_r_vecs[-1]
            print(
                f"  w=[{w[0]:.2f},{w[1]:.2f}]  "
                f"fuel={m[0]:.3f}  safety={m[1]:.3f}"
            )

    mean_r_arr = np.stack(mean_r_vecs)  # (n_weights, 2)

    results = {
        "weights": weights,
        "mean_r_vec": mean_r_arr,
        "all_r_vecs": all_r_vecs,
    }

    # ---- W&B logging ----
    if wandb_run is not None:
        try:
            import wandb

            # Scatter table: one row per weight point
            cols = ["w_fuel", "w_safety", "mean_fuel_reward", "mean_safety_reward"]
            table = wandb.Table(columns=cols)
            for wi in range(n_weights):
                table.add_data(
                    float(weights[wi, 0]),
                    float(weights[wi, 1]),
                    float(mean_r_arr[wi, 0]),
                    float(mean_r_arr[wi, 1]),
                )
            wandb_run.log({"pareto/front_table": table})

            # Custom scatter plot
            wandb_run.log({
                "pareto/fuel_vs_safety": wandb.plot.scatter(
                    table,
                    x="mean_fuel_reward",
                    y="mean_safety_reward",
                    title="Pareto Front: Fuel vs Safety",
                ),
            })
        except Exception as exc:
            print(f"[pareto_eval] W&B logging failed: {exc}")

    return results
