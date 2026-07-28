"""
Case C — clean 4-way comparison for the continuous El Farol bar problem.

Runs four configurations side-by-side:

       |  Centralized critic  |  Decentralized critic
  -----+----------------------+----------------------
  WGF  |  WMATRPO + Q(s, a)   |  WMATRPO + Q_i(s, a_i)
  IPPO |  IPPO    + Q(s, a)   |  IPPO    + Q_i(s, a_i)   ← paper's IPPO baseline

For each cell we report:
  - peak / final mean reward,
  - distance to capacity (|Σ a_i - C|),
  - bifurcation profile (n_high, n_low, n_mid),
  - the empirical 'split' (e.g. 7-3).

This isolates the two confounded dimensions of Paper 2's Case C: information
(centralized vs decentralized) and algorithm (flow-based vs PPO-clipped).

Usage:
    python -m wmatrpo.scripts.case_c
    python -m wmatrpo.scripts.case_c --n-iterations 1500 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import yaml

from wmatrpo.envs.elfarol import ElFarolEnv, ElFarolConfig
from wmatrpo.policy import GaussianPolicy, PolicyConfig
from wmatrpo.critic import CentralizedCritic, DecentralizedCritic, CriticConfig
from wmatrpo.dual_solver import DualSolver, DualSolverConfig
from wmatrpo.caatr import CAATR, CAATRConfig
from wmatrpo.algorithm import WMATRPO, WMATRPOConfig
from wmatrpo.ippo import IPPO, IPPOConfig
from wmatrpo.utils import set_seed


# -----------------------------------------------------------------------------
# Build a single algorithm/critic configuration
# -----------------------------------------------------------------------------
def build_setup(env: ElFarolEnv, algorithm: str, centralized: bool,
                policy_cfg: PolicyConfig, seed: int = 0,
                batch_size: int = 30, n_iters: int = 1500):
    """
    algorithm : 'wgf' or 'ippo'
    centralized : True → CentralizedCritic, False → DecentralizedCritic
    """
    set_seed(seed)

    critic_cfg = CriticConfig(
        state_dim=env.state_dim, n_agents=env.n_agents,
        hidden=64, n_layers=2, lr=3e-3, n_update_epochs=4,
    )
    critic = (CentralizedCritic(critic_cfg) if centralized
              else DecentralizedCritic(critic_cfg))

    if algorithm == "wgf":
        dual_cfg = DualSolverConfig(
            lambda_min=1e-4, lambda_max=50.0,
            action_low=env.action_low, action_high=env.action_high,
            cost_type="l2",
        )
        dual = DualSolver(dual_cfg)
        caatr_cfg = CAATRConfig(C=0.05, epsilon_base=1e-8,
                                epsilon_max=0.5, fallback_delta=0.05)
        caatr = CAATR(caatr_cfg, env.n_agents)
        alg_cfg = WMATRPOConfig(batch_size=batch_size,
                                n_agents=env.n_agents, seed=seed)
        algo = WMATRPO.from_config(
            env=env, alg_cfg=alg_cfg, policy_cfg=policy_cfg,
            critic=critic, dual_solver=dual, caatr=caatr,
        )
    elif algorithm == "ippo":
        ippo_cfg = IPPOConfig(batch_size=batch_size,
                              n_agents=env.n_agents,
                              clip_eps=0.2,
                              n_policy_epochs=4,
                              policy_lr=3e-4,
                              seed=seed)
        algo = IPPO.from_config(env=env, ippo_cfg=ippo_cfg,
                                policy_cfg=policy_cfg, critic=critic)
    else:
        raise ValueError(f"unknown algorithm: {algorithm}")
    return algo


def run_one(env: ElFarolEnv, algorithm: str, centralized: bool,
            seed: int, n_iters: int, batch_size: int) -> dict:
    policy_cfg = PolicyConfig(
        init_mean=0.5, init_std=0.2, std_min=0.02, std_max=0.4,
        action_low=env.action_low, action_high=env.action_high,
    )
    algo = build_setup(env, algorithm, centralized, policy_cfg,
                       seed=seed, batch_size=batch_size, n_iters=n_iters)
    label = f"{algorithm}-{'centralized' if centralized else 'decentralized'}"
    print(f"\n=== {label}  (seed={seed}) ===")
    for it in range(n_iters):
        info = algo.step()
        if (it + 1) % max(1, n_iters // 6) == 0 or it == 0:
            means = [float(p.mean.item()) for p in algo.policies]
            print(f"  iter {it+1:4d} | reward {info['avg_reward']:+.4f} | "
                  f"means [{', '.join(f'{m:.2f}' for m in means)}]")

    final_means = torch.tensor([float(p.mean.item()) for p in algo.policies])
    bifurcation = env.is_bifurcated(final_means)
    peak_reward = float(np.max(algo.reward_history))
    final_reward = float(algo.reward_history[-1])
    terminal_total = float(final_means.sum().item())
    return {
        "label": label,
        "algorithm": algorithm,
        "centralized": centralized,
        "seed": seed,
        "final_means": final_means.tolist(),
        "terminal_total": terminal_total,
        "shortfall": bifurcation["shortfall"],
        "n_high": bifurcation["n_high"],
        "n_low": bifurcation["n_low"],
        "n_mid": bifurcation["n_mid"],
        "split": bifurcation["split"],
        "peak_reward": peak_reward,
        "final_reward": final_reward,
    }


# -----------------------------------------------------------------------------
# Sweep all four cells
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--capacity", type=float, default=6.0)
    parser.add_argument("--n-iterations", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--out", default="runs/case_c")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = ElFarolEnv(ElFarolConfig(n_agents=args.n_agents,
                                   capacity=args.capacity))

    cells = [
        ("wgf",  True),
        ("wgf",  False),
        ("ippo", True),
        ("ippo", False),
    ]

    rows = []
    for algorithm, centralized in cells:
        for seed in args.seeds:
            row = run_one(env, algorithm, centralized,
                          seed=seed, n_iters=args.n_iterations,
                          batch_size=args.batch_size)
            rows.append(row)

    df = pd.DataFrame(rows)
    raw_path = out_dir / "case_c_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"\nRaw results saved to {raw_path}")

    # ---- aggregate table (mean ± std across seeds) ----
    summary = (df.groupby(["algorithm", "centralized"])
                 .agg(terminal_total_mean=("terminal_total", "mean"),
                      terminal_total_std=("terminal_total", "std"),
                      shortfall_mean=("shortfall", "mean"),
                      shortfall_std=("shortfall", "std"),
                      n_high_mean=("n_high", "mean"),
                      n_low_mean=("n_low", "mean"),
                      n_mid_mean=("n_mid", "mean"),
                      peak_reward_mean=("peak_reward", "mean"),
                      final_reward_mean=("final_reward", "mean"))
                 .round(3))
    summary_path = out_dir / "case_c_summary.csv"
    summary.to_csv(summary_path)

    print("\n=== Case C — 4-way comparison summary ===")
    print(summary.to_string())
    print(f"\nFull summary saved to {summary_path}")

    # ---- terse interpretation ----
    print("\n=== Interpretation ===")
    print(f"  Target: Σ a_i = {args.capacity} (paper claims 7-3 split for N=10, C=6)")
    print()
    for algorithm, centralized in cells:
        sub = df[(df["algorithm"] == algorithm) & (df["centralized"] == centralized)]
        label = f"{algorithm}-{'central' if centralized else 'decentral'}"
        print(f"  {label:18s} | "
              f"Σ={sub['terminal_total'].mean():.2f}±{sub['terminal_total'].std():.2f}  "
              f"shortfall={sub['shortfall'].mean():.3f}  "
              f"split avg n_high={sub['n_high'].mean():.1f}, "
              f"n_low={sub['n_low'].mean():.1f}")
    print()
    print("If WGF-decentral significantly UNDERPERFORMS WGF-central, then info matters.")
    print("If IPPO-central significantly OUTPERFORMS IPPO-decentral, then info matters here too.")
    print("If all four cells achieve Σ ≈ C with bifurcation, the task structure carries the result.")


if __name__ == "__main__":
    main()
