"""
Basin-gap ablation with allocation-strategy comparison (paper Table 5).

Sweeps basin_gap_factor k ∈ {0.5, 1.0, 1.5, 2.0} on the 2-agent differential
game, comparing four trust-region-radius rules on the SAME W-MATRPO dual solve,
critic, env, and initialization:

    fixed     — full shared budget to every agent (standard TRPO radius)
    greedy    — advantage/drift-scored budget split
    weighted  — water-filling on positive advantages
    caatr     — coordination-aware adaptive radius (the paper's proposal)

Only the radius rule changes between columns, so the comparison isolates the
allocation strategy — the controlled-ablation standard used for the R2.4 table.
Every cell is averaged over seeds and written with mean±s.d. so Table 5 traces
to seed-controlled code.

Usage:
    python -m wmatrpo.scripts.basingap_compare --ks 0.5 1.0 1.5 2.0 --seeds 0 1 2
    python -m wmatrpo.scripts.basingap_compare --ks 1.0 --seeds 0 --n-iterations 500
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from wmatrpo.env import DifferentialGameEnv, DifferentialGameConfig
from wmatrpo.policy import PolicyConfig
from wmatrpo.critic import CentralizedCritic, CriticConfig
from wmatrpo.dual_solver import DualSolver, DualSolverConfig
from wmatrpo.caatr import CAATR, CAATRConfig
from wmatrpo.allocation import AllocationStrategy, AllocationConfig
from wmatrpo.algorithm import WMATRPO, WMATRPOConfig
from wmatrpo.utils import set_seed

REPO_ROOT = Path(__file__).resolve().parents[2]

CAATR_C_2AGENT = 0.02
EPSILON_TOTAL = 0.1     # shared budget for fixed/greedy/weighted (paper δ)


def build(strategy: str, k: float, seed: int, batch_size: int) -> tuple:
    set_seed(seed)
    env = DifferentialGameEnv(DifferentialGameConfig(
        n_agents=2, variances=[1.0, 3.0], basin_gap_factor=float(k)))
    pcfg = PolicyConfig(init_mean=1.5, init_std=0.5, std_min=0.05, std_max=3.0,
                        action_low=env.action_low, action_high=env.action_high)
    critic = CentralizedCritic(CriticConfig(
        state_dim=env.state_dim, n_agents=2, hidden=64, n_layers=2,
        lr=3e-3, n_update_epochs=8))
    dual = DualSolver(DualSolverConfig(
        lambda_min=1e-4, lambda_max=50.0,
        action_low=env.action_low, action_high=env.action_high, cost_type="l2"))
    if strategy == "caatr":
        radius = CAATR(CAATRConfig(C=CAATR_C_2AGENT, epsilon_base=1e-8,
                                   epsilon_max=0.5, fallback_delta=0.1), 2)
    else:
        radius = AllocationStrategy(
            AllocationConfig(method=strategy, epsilon_total=EPSILON_TOTAL), 2)
    alg = WMATRPO.from_config(
        env=env, alg_cfg=WMATRPOConfig(batch_size=batch_size, n_agents=2, seed=seed),
        policy_cfg=pcfg, critic=critic, dual_solver=dual, caatr=radius)
    return alg, env


def run_one(strategy: str, k: float, seed: int, n_iters: int, batch_size: int) -> dict:
    alg, env = build(strategy, k, seed, batch_size)
    for _ in range(n_iters):
        alg.step()
    final_means = np.array([float(p.mean.item()) for p in alg.policies])
    dist = float(np.linalg.norm(final_means - env.cfg.global_optimum))
    return {
        "strategy": strategy, "k": k, "seed": seed,
        "distance_to_global": dist,
        "peak_reward": float(np.max(alg.reward_history)),
        "final_reward": float(alg.reward_history[-1]),
        "final_means": final_means.tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ks", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0])
    p.add_argument("--strategies", nargs="+",
                   default=["fixed", "greedy", "weighted", "caatr"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--n-iterations", type=int, default=2500)
    p.add_argument("--batch-size", type=int, default=30)
    p.add_argument("--out", default="runs/basingap_compare")
    args = p.parse_args()

    out = (REPO_ROOT / args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    for k in args.ks:
        for strat in args.strategies:
            for seed in args.seeds:
                print(f"=== k={k} strategy={strat} seed={seed} ===", flush=True)
                rows.append(run_one(strat, k, seed, args.n_iterations, args.batch_size))
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "basingap_raw.csv", index=False)

    g = raw.groupby(["k", "strategy"]).agg(
        dist_mean=("distance_to_global", "mean"),
        dist_std=("distance_to_global", "std"),
        peak_mean=("peak_reward", "mean"),
        final_reward_mean=("final_reward", "mean"),
    ).reset_index()
    g.to_csv(out / "basingap_summary.csv", index=False)
    print("\n", g.to_string(index=False))
    print(f"\nwrote {out/'basingap_raw.csv'} and {out/'basingap_summary.csv'}")


if __name__ == "__main__":
    main()
