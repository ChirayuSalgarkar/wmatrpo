"""
Basin-gap ablation runner (Table 4 in the paper).

Sweeps basin_gap_factor k ∈ {0.5, 1.0, 1.5, 2.0} on a 2-agent setup, with the
same algorithm/critic/CAATR settings as `diffgame_2agent.yaml`. Writes one
run directory per k and produces a `basingap_summary.csv` aggregating the
final-distance numbers.

Usage:
    python -m wmatrpo.scripts.basingap --config configs/basingap.yaml
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd

from wmatrpo.scripts.train import build_from_config
from wmatrpo.utils import load_yaml, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ks", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)
    base_run_dir = Path(base_cfg.get("trainer", {}).get("run_dir", "runs/basingap"))
    base_run_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for k in args.ks:
        for seed in args.seeds:
            cfg = copy.deepcopy(base_cfg)
            cfg.setdefault("env", {})["basin_gap_factor"] = float(k)
            cfg.setdefault("algorithm", {})["seed"] = int(seed)
            cfg.setdefault("trainer", {})["run_dir"] = str(base_run_dir / f"k={k}_seed={seed}")

            # write a temp config file (Trainer expects to dump a copy)
            tmp_cfg_path = base_run_dir / f"k={k}_seed={seed}_config.yaml"
            import yaml
            with open(tmp_cfg_path, "w") as f:
                yaml.safe_dump(cfg, f)

            print(f"\n=== basin_gap_factor={k}, seed={seed} ===")
            trainer = build_from_config(tmp_cfg_path)
            summary = trainer.train()
            rows.append({
                "k": k, "seed": seed,
                **{f"final_a{i}": v for i, v in enumerate(summary["final_means"])},
                "distance_to_global": summary["distance_to_global"],
                "peak_reward": summary["peak_reward"],
            })

    df = pd.DataFrame(rows)
    out = base_run_dir / "basingap_summary.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote summary to {out}")
    # print averaged table
    agg = df.groupby("k").agg({
        "distance_to_global": ["mean", "std"],
        "peak_reward": ["mean", "std"],
    })
    print("\nAggregate (mean ± std across seeds):")
    print(agg.to_string())


if __name__ == "__main__":
    main()
