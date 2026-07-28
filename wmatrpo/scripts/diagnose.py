"""
Load CSV outputs from a training run and produce a diagnostic figure.

The figure has six panels:
  (a) per-agent policy mean over iterations         → shows convergence / oscillation
  (b) per-agent policy std over iterations          → shows whether σ is collapsing or exploding
  (c) average reward + running maximum              → shows whether peaks are sustained
  (d) per-agent W₁ drift                            → measures how much each policy moved per step
  (e) per-agent CAATR adaptive δ                    → shows the trust-region response
  (f) per-agent dual variable λ*                    → high λ* = active constraint, very high = unstable

Usage:
    python -m wmatrpo.scripts.diagnose --run runs/diffgame_3agent
    python -m wmatrpo.scripts.diagnose --run runs/diffgame_3agent --out my_diagnostic.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="path to run directory")
    parser.add_argument("--out", default=None, help="output PNG path")
    parser.add_argument("--ema", type=int, default=50,
                        help="EMA window for smoothed overlays (0 = off)")
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.exists():
        raise FileNotFoundError(f"run directory not found: {run_dir}")

    # ---- load ----
    traj = pd.read_csv(run_dir / "trajectory.csv")
    rewards = pd.read_csv(run_dir / "rewards.csv")
    drift = pd.read_csv(run_dir / "drift.csv")
    deltas = pd.read_csv(run_dir / "deltas.csv")
    lambdas = pd.read_csv(run_dir / "lambdas.csv")
    summary = json.loads((run_dir / "summary.json").read_text())

    n_agents = summary["n_agents"]
    target_global = 5.0
    target_local = 1.0

    print(f"Loaded run: {run_dir.resolve()}")
    print(f"  iterations: {len(rewards)}")
    print(f"  n_agents:   {n_agents}")
    print(f"  final dist to global: {summary['distance_to_global']:.3f}")
    print(f"  peak reward:          {summary['peak_reward']:.4f}")

    # ---- plot ----
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    colors = plt.cm.tab10.colors

    # (a) means
    ax = axes[0, 0]
    for i in range(n_agents):
        ax.plot(traj["iteration"], traj[f"agent{i}_mean"],
                color=colors[i % 10], alpha=0.7, lw=0.8, label=f"agent {i}")
        if args.ema > 1:
            ema = traj[f"agent{i}_mean"].ewm(span=args.ema).mean()
            ax.plot(traj["iteration"], ema, color=colors[i % 10], lw=1.8)
    ax.axhline(target_global, color="green", ls="--", lw=1, alpha=0.6, label="global=5")
    ax.axhline(target_local, color="red", ls="--", lw=1, alpha=0.6, label="local=1")
    ax.set_ylabel(r"policy mean $\mu_i$")
    ax.set_title("(a) Per-agent policy means")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # (b) stds
    ax = axes[0, 1]
    for i in range(n_agents):
        ax.plot(traj["iteration"], traj[f"agent{i}_std"],
                color=colors[i % 10], alpha=0.8, lw=1, label=f"agent {i}")
    ax.set_ylabel(r"policy std $\sigma_i$")
    ax.set_title("(b) Per-agent policy stds")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)

    # (c) reward
    ax = axes[1, 0]
    r = rewards["avg_reward"].values
    iters = rewards["iteration"].values
    ax.plot(iters, r, color="C0", alpha=0.3, lw=0.7, label="per-iter")
    if args.ema > 1:
        ema = pd.Series(r).ewm(span=args.ema).mean()
        ax.plot(iters, ema, color="C0", lw=1.8, label=f"EMA-{args.ema}")
    running_max = np.maximum.accumulate(r)
    ax.plot(iters, running_max, color="green", ls="--", lw=1.5, label="running max")
    ax.set_ylabel("mean reward")
    ax.set_title("(c) Mean reward per iteration")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    # (d) drift
    ax = axes[1, 1]
    for i in range(n_agents):
        col = f"agent{i}_drift"
        if col in drift.columns:
            ax.plot(drift["iteration"], drift[col],
                    color=colors[i % 10], alpha=0.7, lw=0.8, label=f"agent {i}")
            if args.ema > 1:
                ema = drift[col].ewm(span=args.ema).mean()
                ax.plot(drift["iteration"], ema, color=colors[i % 10], lw=1.5)
    ax.set_ylabel(r"$W_1(\pi_i^{t-1}, \pi_i^{t})$")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_title("(d) Per-agent W₁ drift (symlog)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)

    # (e) deltas
    ax = axes[2, 0]
    for i in range(n_agents):
        col = f"agent{i}_delta"
        if col in deltas.columns:
            ax.plot(deltas["iteration"], deltas[col],
                    color=colors[i % 10], alpha=0.7, lw=0.8, label=f"agent {i}")
            if args.ema > 1:
                ema = deltas[col].ewm(span=args.ema).mean()
                ax.plot(deltas["iteration"], ema, color=colors[i % 10], lw=1.5)
    ax.set_ylabel(r"trust radius $\delta_i$")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_title("(e) CAATR adaptive trust radii (log)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=8, ncol=2)

    # (f) lambdas
    ax = axes[2, 1]
    for i in range(n_agents):
        col = f"agent{i}_lambda"
        if col in lambdas.columns:
            ax.plot(lambdas["iteration"], lambdas[col],
                    color=colors[i % 10], alpha=0.7, lw=0.8, label=f"agent {i}")
            if args.ema > 1:
                ema = lambdas[col].ewm(span=args.ema).mean()
                ax.plot(lambdas["iteration"], ema, color=colors[i % 10], lw=1.5)
    ax.set_ylabel(r"dual variable $\lambda_i^*$")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_title("(f) Per-agent dual variable λ* (log)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=8, ncol=2)

    fig.suptitle(f"Diagnostic: {run_dir.name}  —  "
                 f"final dist to global = {summary['distance_to_global']:.2f}, "
                 f"peak reward = {summary['peak_reward']:.3f}",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = Path(args.out) if args.out else run_dir / "diagnose.png"
    fig.savefig(out, dpi=140)
    print(f"\nSaved diagnostic figure -> {out}")

    # quick numerical summary
    print("\n--- numerical summary ---")
    print(f"reward  final = {r[-1]:+.4f}   max = {running_max[-1]:+.4f}   "
          f"max - final = {running_max[-1] - r[-1]:+.4f}")
    if "agent0_drift" in drift.columns:
        last_drifts = [drift[f"agent{i}_drift"].iloc[-1] for i in range(n_agents)]
        print(f"drift   final (per agent): {[f'{d:.4f}' for d in last_drifts]}")
        # check whether drift has been decreasing or staying flat
        early = drift["agent0_drift"].iloc[:200].mean()
        late = drift["agent0_drift"].iloc[-200:].mean()
        print(f"drift   agent-0 early-200 mean = {early:.4f},  late-200 mean = {late:.4f}")
    if "agent0_lambda" in lambdas.columns:
        late_lambda = [lambdas[f"agent{i}_lambda"].iloc[-200:].mean() for i in range(n_agents)]
        print(f"λ*      late-200 mean (per agent): {[f'{l:.3f}' for l in late_lambda]}")


if __name__ == "__main__":
    main()
