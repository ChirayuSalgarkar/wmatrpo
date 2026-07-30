"""
Head-to-head baseline comparison (Reviewer R2.4).

Runs W-MATRPO, IPPO, MAPPO, and HAPPO on the n-agent differential game and
produces a comparison table: final distance to global optimum, peak reward, and
final actions, averaged over seeds. This is the deliverable that answers the
"no comparison against MAPPO/HAPPO/IPPO" reviewer demand.

All four algorithms use the SAME env, SAME batch size, SAME critic architecture
(centralized), and SAME policy initialization — so the comparison isolates the
policy-update rule:
  - W-MATRPO : Wasserstein dual trust region + CAATR
  - HAPPO    : sequential + IS correction + PPO clip  (the closest KL-based cousin)
  - MAPPO    : concurrent + PPO clip + centralized critic
  - IPPO     : concurrent + PPO clip + centralized critic (here; use build_critic
               centralized=False for the fully-decentralized variant)

Usage:
    python -m wmatrpo.scripts.baselines --agent-counts 3 5 7 9 --seeds 0 1 2
    python -m wmatrpo.scripts.baselines --agent-counts 3 --n-iterations 1000 --quick
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from wmatrpo.env import DifferentialGameEnv, DifferentialGameConfig
from wmatrpo.policy import GaussianPolicy, PolicyConfig
from wmatrpo.critic import CentralizedCritic, CriticConfig
from wmatrpo.dual_solver import DualSolver, DualSolverConfig
from wmatrpo.caatr import CAATR, CAATRConfig
from wmatrpo.allocation import AllocationStrategy, AllocationConfig
from wmatrpo.algorithm import WMATRPO, WMATRPOConfig
from wmatrpo.ippo import IPPO, IPPOConfig
from wmatrpo.mappo import MAPPO, MAPPOConfig
from wmatrpo.happo import HAPPO, HAPPOConfig
from wmatrpo.hatrpo import HATRPO, HATRPOConfig
from wmatrpo.utils import set_seed


REPO_ROOT = Path(__file__).resolve().parents[2]

# CAATR constant per agent count (paper Table 2)
CAATR_C = {2: 0.02, 3: 0.02, 5: 0.02, 7: 0.10, 9: 0.15}


def make_env(n_agents: int) -> DifferentialGameEnv:
    variances = [1.0] * n_agents
    if n_agents >= 2:
        variances[1] = 3.0
    return DifferentialGameEnv(DifferentialGameConfig(
        n_agents=n_agents, variances=variances, linear_bias=0.0,
    ))


def make_policy_cfg(env) -> PolicyConfig:
    return PolicyConfig(init_mean=1.5, init_std=0.5, std_min=0.1, std_max=1.0,
                        action_low=env.action_low, action_high=env.action_high)


def make_critic(env) -> CentralizedCritic:
    return CentralizedCritic(CriticConfig(
        state_dim=env.state_dim, n_agents=env.n_agents,
        hidden=64, n_layers=2, lr=3e-3, n_update_epochs=8,
    ))


def build_algo(name: str, env, seed: int, batch_size: int):
    set_seed(seed)
    policy_cfg = make_policy_cfg(env)
    critic = make_critic(env)

    if name == "wmatrpo":
        dual = DualSolver(DualSolverConfig(
            lambda_min=1e-4, lambda_max=50.0,
            action_low=env.action_low, action_high=env.action_high, cost_type="l2"))
        caatr = CAATR(CAATRConfig(C=CAATR_C.get(env.n_agents, 0.02),
                                  epsilon_base=1e-8, epsilon_max=0.5,
                                  fallback_delta=0.1), env.n_agents)
        return WMATRPO.from_config(
            env=env, alg_cfg=WMATRPOConfig(batch_size=batch_size,
                                           n_agents=env.n_agents, seed=seed),
            policy_cfg=policy_cfg, critic=critic, dual_solver=dual, caatr=caatr)
    elif name == "wmatrpo_fixed":
        # W-MATRPO with a FIXED (non-adaptive) trust-region radius instead of CAATR.
        # Same Wasserstein dual solve, critic, and init as `wmatrpo`; the only
        # difference is the radius rule (fixed budget vs coordination-aware adaptive).
        # This isolates CAATR's marginal contribution on top of the W-MATRPO geometry.
        dual = DualSolver(DualSolverConfig(
            lambda_min=1e-4, lambda_max=50.0,
            action_low=env.action_low, action_high=env.action_high, cost_type="l2"))
        fixed = AllocationStrategy(
            AllocationConfig(method="fixed", epsilon_total=0.1), env.n_agents)
        return WMATRPO.from_config(
            env=env, alg_cfg=WMATRPOConfig(batch_size=batch_size,
                                           n_agents=env.n_agents, seed=seed),
            policy_cfg=policy_cfg, critic=critic, dual_solver=dual, caatr=fixed)
    elif name == "ippo":
        return IPPO.from_config(env, IPPOConfig(batch_size=batch_size,
                                                n_agents=env.n_agents, seed=seed),
                                policy_cfg, critic)
    elif name == "mappo":
        return MAPPO.from_config(env, MAPPOConfig(batch_size=batch_size,
                                                  n_agents=env.n_agents, seed=seed),
                                 policy_cfg, critic)
    elif name == "happo":
        return HAPPO.from_config(env, HAPPOConfig(batch_size=batch_size,
                                                  n_agents=env.n_agents, seed=seed),
                                 policy_cfg, critic)
    elif name == "hatrpo":
        # Standard HATRPO: KL trust region, fixed radius δ (Kuba 2021).
        return HATRPO.from_config(
            env, HATRPOConfig(batch_size=batch_size, n_agents=env.n_agents,
                              delta=0.01, seed=seed, use_caatr=False),
            policy_cfg, critic)
    elif name == "hatrpo_caatr":
        # HATRPO with CAATR-adaptive per-agent radii (same C schedule as W-MATRPO).
        caatr = CAATR(CAATRConfig(C=CAATR_C.get(env.n_agents, 0.02),
                                  epsilon_base=1e-8, epsilon_max=0.5,
                                  fallback_delta=0.1), env.n_agents)
        return HATRPO.from_config(
            env, HATRPOConfig(batch_size=batch_size, n_agents=env.n_agents,
                              delta=0.01, seed=seed, use_caatr=True),
            policy_cfg, critic, caatr=caatr)
    else:
        raise ValueError(f"unknown algorithm: {name}")


def run_one(name: str, env, seed: int, n_iters: int, batch_size: int) -> dict:
    algo = build_algo(name, env, seed, batch_size)
    for _ in range(n_iters):
        algo.step()
    final_means = np.array([float(p.mean.item()) for p in algo.policies])
    global_opt = np.full(env.n_agents, env.cfg.global_optimum)
    dist_global = float(np.linalg.norm(final_means - global_opt))
    peak_reward = float(np.max(algo.reward_history))
    final_reward = float(algo.reward_history[-1])
    return {
        "algorithm": name,
        "n_agents": env.n_agents,
        "seed": seed,
        "distance_to_global": dist_global,
        "peak_reward": peak_reward,
        "final_reward": final_reward,
        "final_means": final_means.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-counts", nargs="+", type=int, default=[3, 5, 7, 9])
    parser.add_argument("--algorithms", nargs="+",
                        default=["wmatrpo", "happo", "mappo", "ippo"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-iterations", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--quick", action="store_true",
                        help="shortcut: 800 iterations, seeds [0]")
    parser.add_argument("--out", default="runs/baselines")
    args = parser.parse_args()

    if args.quick:
        args.n_iterations = 800
        args.seeds = [0]

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for n in args.agent_counts:
        env = make_env(n)
        for name in args.algorithms:
            for seed in args.seeds:
                print(f"[{name:8s}] N={n} seed={seed} ... ", end="", flush=True)
                row = run_one(name, env, seed, args.n_iterations, args.batch_size)
                rows.append(row)
                print(f"dist_global={row['distance_to_global']:.3f}  "
                      f"peak_reward={row['peak_reward']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "baselines_raw.csv", index=False)

    # aggregate: mean ± std across seeds, per (algorithm, n_agents)
    agg = (df.groupby(["n_agents", "algorithm"])
             .agg(dist_mean=("distance_to_global", "mean"),
                  dist_std=("distance_to_global", "std"),
                  peak_mean=("peak_reward", "mean"),
                  peak_std=("peak_reward", "std"))
             .round(3))
    agg.to_csv(out_dir / "baselines_summary.csv")

    print("\n=== Baseline comparison (distance to global optimum ↓, lower is better) ===")
    print(agg.to_string())
    print(f"\nRaw: {out_dir/'baselines_raw.csv'}")
    print(f"Summary: {out_dir/'baselines_summary.csv'}")

    # pivot table view: rows = n_agents, cols = algorithm, cell = dist mean
    pivot = df.pivot_table(index="n_agents", columns="algorithm",
                           values="distance_to_global", aggfunc="mean").round(3)
    print("\n=== Distance-to-global pivot (mean over seeds) ===")
    print(pivot.to_string())
    pivot.to_csv(out_dir / "baselines_pivot.csv")


if __name__ == "__main__":
    main()
