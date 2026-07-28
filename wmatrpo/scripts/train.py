"""
CLI entry point.

Usage:
    python -m wmatrpo.scripts.train --config configs/diffgame_3agent.yaml
    python -m wmatrpo.scripts.train --config configs/diffgame_3agent.yaml --seed 7
"""
from __future__ import annotations

import argparse
from pathlib import Path

from wmatrpo.env import DifferentialGameEnv, DifferentialGameConfig
from wmatrpo.policy import PolicyConfig
from wmatrpo.critic import CentralizedCritic, CriticConfig
from wmatrpo.dual_solver import DualSolver, DualSolverConfig
from wmatrpo.caatr import CAATR, CAATRConfig
from wmatrpo.algorithm import WMATRPO, WMATRPOConfig
from wmatrpo.trainer import Trainer, TrainerConfig
from wmatrpo.utils import set_seed, load_yaml, dataclass_from_dict


def build_from_config(path: str | Path, seed_override: int | None = None) -> Trainer:
    cfg_all = load_yaml(path)

    if seed_override is not None:
        cfg_all.setdefault("algorithm", {})["seed"] = seed_override

    set_seed(cfg_all.get("algorithm", {}).get("seed", 0))

    # --- env ---
    env_cfg = dataclass_from_dict(DifferentialGameConfig, cfg_all.get("env", {}))
    env = DifferentialGameEnv(env_cfg)

    # --- policy ---
    policy_cfg = dataclass_from_dict(PolicyConfig, cfg_all.get("policy", {}))
    policy_cfg.action_low = env.action_low
    policy_cfg.action_high = env.action_high

    # --- critic ---
    critic_dict = dict(cfg_all.get("critic", {}))
    critic_dict.setdefault("state_dim", env.state_dim)
    critic_dict.setdefault("n_agents", env.n_agents)
    critic_cfg = dataclass_from_dict(CriticConfig, critic_dict)
    critic = CentralizedCritic(critic_cfg)

    # --- dual solver ---
    ds_dict = dict(cfg_all.get("dual_solver", {}))
    ds_dict.setdefault("action_low", env.action_low)
    ds_dict.setdefault("action_high", env.action_high)
    ds_cfg = dataclass_from_dict(DualSolverConfig, ds_dict)
    dual_solver = DualSolver(ds_cfg)

    # --- CAATR ---
    caatr_cfg = dataclass_from_dict(CAATRConfig, cfg_all.get("caatr", {}))
    caatr = CAATR(caatr_cfg, env.n_agents)

    # --- algorithm ---
    alg_dict = dict(cfg_all.get("algorithm", {}))
    alg_dict.setdefault("n_agents", env.n_agents)
    alg_cfg = dataclass_from_dict(WMATRPOConfig, alg_dict)
    algorithm = WMATRPO.from_config(
        env=env, alg_cfg=alg_cfg, policy_cfg=policy_cfg,
        critic=critic, dual_solver=dual_solver, caatr=caatr,
    )

    # --- trainer ---
    trainer_cfg = dataclass_from_dict(TrainerConfig, cfg_all.get("trainer", {}))
    trainer = Trainer(algorithm, trainer_cfg, full_config_dump=cfg_all)
    return trainer


def main():
    parser = argparse.ArgumentParser(description="Train W-MATRPO on the differential game.")
    parser.add_argument("--config", required=True, help="path to YAML config")
    parser.add_argument("--seed", type=int, default=None, help="override RNG seed")
    args = parser.parse_args()
    trainer = build_from_config(args.config, seed_override=args.seed)
    trainer.train()


if __name__ == "__main__":
    main()
