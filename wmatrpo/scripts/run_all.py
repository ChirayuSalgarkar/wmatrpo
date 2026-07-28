"""
Run all agent-count configurations and produce a Table 3-style summary.

For each n_agents ∈ {3, 5, 7, 9}, runs a single seed (default 0) using the
matching `configs/diffgame_${n}agent.yaml`. Writes one run directory per N
and a `runs/table3_summary.csv` with final actions and distance to global.

Usage:
    python -m wmatrpo.scripts.run_all
    python -m wmatrpo.scripts.run_all --seeds 0 1 2
    python -m wmatrpo.scripts.run_all --agent-counts 2 3 5
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd
import yaml

from wmatrpo.scripts.train import build_from_config
from wmatrpo.utils import load_yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-counts", nargs="+", type=int, default=[3, 5, 7, 9])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--n-iterations", type=int, default=None,
                        help="override n_iterations in configs (default: use config value)")
    args = parser.parse_args()

    rows = []
    for n in args.agent_counts:
        cfg_path = REPO_ROOT / "configs" / f"diffgame_{n}agent.yaml"
        if not cfg_path.exists():
            print(f"skipping n={n}: no config at {cfg_path}")
            continue
        for seed in args.seeds:
            cfg = load_yaml(cfg_path)
            cfg.setdefault("algorithm", {})["seed"] = seed
            cfg.setdefault("trainer", {})["run_dir"] = f"runs/diffgame_{n}agent_seed{seed}"
            if args.n_iterations is not None:
                cfg.setdefault("trainer", {})["n_iterations"] = args.n_iterations

            # write patched config so Trainer can dump it
            tmp = REPO_ROOT / f"runs/_tmp_n{n}_seed{seed}.yaml"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w") as f:
                yaml.safe_dump(cfg, f)

            print(f"\n=== n_agents={n}, seed={seed} ===")
            trainer = build_from_config(tmp)
            summary = trainer.train()
            rows.append({
                "n_agents": n,
                "seed": seed,
                **{f"final_a{i}": v for i, v in enumerate(summary["final_means"])},
                "distance_to_global": summary["distance_to_global"],
                "distance_to_local": summary["distance_to_local"],
                "peak_reward": summary["peak_reward"],
                "terminal_reward": summary["terminal_reward_at_mean_action"],
            })

    df = pd.DataFrame(rows)
    out = REPO_ROOT / "runs" / "table3_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\nWrote raw summary to {out}")
    if len(args.seeds) > 1:
        agg = df.groupby("n_agents").agg({
            "distance_to_global": ["mean", "std"],
            "peak_reward": ["mean", "std"],
        })
        print("\nAggregate across seeds:")
        print(agg.to_string())
    else:
        print("\nSummary table (one seed per N):")
        print(df[["n_agents", "distance_to_global", "peak_reward"]].to_string(index=False))


if __name__ == "__main__":
    main()
