"""
Training loop + CSV logging.

Owns the outer for-loop over iterations (Algorithm 1 line 2), delegates
each iteration to `WMATRPO.step()`, and writes outputs to a `runs/<name>/`
directory.

Files written per run:
  - trajectory.csv  — per-iteration agent means and stds
  - rewards.csv     — per-iteration mean reward
  - drift.csv       — per-iteration per-agent W₁ drift
  - deltas.csv      — per-iteration per-agent adaptive δ
  - lambdas.csv     — per-iteration per-agent λ*
  - summary.json    — final summary stats
  - config.yaml     — exact config dump
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
import time

import numpy as np
import pandas as pd
import torch
import yaml


@dataclass
class TrainerConfig:
    n_iterations: int = 4000
    log_every: int = 200
    run_dir: str = "runs/default"
    save_outputs: bool = True


class Trainer:
    def __init__(self, algorithm, cfg: TrainerConfig, full_config_dump: Optional[dict] = None):
        self.algorithm = algorithm
        self.cfg = cfg
        self.full_config_dump = full_config_dump or {}
        self.run_dir = Path(cfg.run_dir)
        if cfg.save_outputs:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            if self.full_config_dump:
                with open(self.run_dir / "config.yaml", "w") as f:
                    yaml.safe_dump(self.full_config_dump, f, sort_keys=False)

    def train(self) -> dict:
        t0 = time.time()
        env = self.algorithm.env
        for it in range(self.cfg.n_iterations):
            info = self.algorithm.step()
            if (it + 1) % self.cfg.log_every == 0 or it == 0:
                means = [float(p.mean.item()) for p in self.algorithm.policies]
                msg = (f"iter {it+1:5d} | reward {info['avg_reward']:+.4f} | "
                       f"actions {self._fmt(means)} | "
                       f"deltas {self._fmt(info['deltas'])}")
                print(msg)

        elapsed = time.time() - t0
        summary = self._finalize(elapsed)
        return summary

    # ------------------------------------------------------------------
    def _finalize(self, elapsed: float) -> dict:
        env = self.algorithm.env
        alg = self.algorithm
        final_means = [float(p.mean.item()) for p in alg.policies]
        final_stds = [float(p.std.item()) for p in alg.policies]
        global_opt = [env.cfg.global_optimum] * env.n_agents
        local_opt = [env.cfg.local_optimum] * env.n_agents
        dist_global = float(np.linalg.norm(np.array(final_means) - np.array(global_opt)))
        dist_local = float(np.linalg.norm(np.array(final_means) - np.array(local_opt)))
        peak_reward = float(np.max(alg.reward_history)) if alg.reward_history else 0.0
        terminal_reward = env.reward_at((float(np.mean(final_means))))

        summary = {
            "n_iterations": self.cfg.n_iterations,
            "n_agents": env.n_agents,
            "final_means": final_means,
            "final_stds": final_stds,
            "distance_to_global": dist_global,
            "distance_to_local": dist_local,
            "peak_reward": peak_reward,
            "terminal_reward_at_mean_action": float(terminal_reward),
            "elapsed_seconds": elapsed,
        }

        if not self.cfg.save_outputs:
            return summary

        # rewards.csv
        pd.DataFrame({
            "iteration": np.arange(len(alg.reward_history)),
            "avg_reward": alg.reward_history,
        }).to_csv(self.run_dir / "rewards.csv", index=False)

        # trajectory.csv
        traj_data = {"iteration": np.arange(len(alg.mean_history[0]))}
        for i in range(env.n_agents):
            traj_data[f"agent{i}_mean"] = alg.mean_history[i]
            traj_data[f"agent{i}_std"] = alg.std_history[i]
        pd.DataFrame(traj_data).to_csv(self.run_dir / "trajectory.csv", index=False)

        # drift.csv
        drift_data = {"iteration": np.arange(len(alg.caatr.drift_history[0]))}
        for i in range(env.n_agents):
            drift_data[f"agent{i}_drift"] = alg.caatr.drift_history[i]
        pd.DataFrame(drift_data).to_csv(self.run_dir / "drift.csv", index=False)

        # deltas.csv
        delta_data = {"iteration": np.arange(len(alg.delta_history[0]))}
        for i in range(env.n_agents):
            delta_data[f"agent{i}_delta"] = alg.delta_history[i]
        pd.DataFrame(delta_data).to_csv(self.run_dir / "deltas.csv", index=False)

        # lambdas.csv
        lam_data = {"iteration": np.arange(len(alg.lambda_history[0]))}
        for i in range(env.n_agents):
            lam_data[f"agent{i}_lambda"] = alg.lambda_history[i]
        pd.DataFrame(lam_data).to_csv(self.run_dir / "lambdas.csv", index=False)

        # summary.json
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nRun complete in {elapsed:.1f}s.")
        print(f"  Final actions:  {self._fmt(final_means)}")
        print(f"  Distance to global optimum (5,...,5): {dist_global:.4f}")
        print(f"  Distance to local optimum  (1,...,1): {dist_local:.4f}")
        print(f"  Peak reward across training: {peak_reward:.4f}")
        print(f"  Outputs saved to: {self.run_dir}")
        return summary

    @staticmethod
    def _fmt(xs):
        return "(" + ", ".join(f"{x:+.3f}" for x in xs) + ")"
