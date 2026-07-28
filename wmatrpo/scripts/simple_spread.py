"""
Simple Spread (MPE) runner — SCAFFOLD ONLY.

Simple Spread is the canonical MARL continuous-control benchmark from the
MADDPG paper (Lowe et al. 2017) — 3 agents must cover 3 landmarks while
avoiding collisions. Used in IPPO, MAPPO, COMA, and most modern MARL
papers as the standard cooperative benchmark.

This script's job is to scaffold the integration; it currently raises with
a clear message about what still needs to be implemented. See
`wmatrpo/HIGHER_DIFFICULTY_ROADMAP.md` for the missing-pieces list.

Usage (once the missing pieces are in):
    pip install pettingzoo[mpe]
    python -m wmatrpo.scripts.simple_spread --algorithm ippo --n-iterations 1000
"""
from __future__ import annotations

import argparse
import sys

from wmatrpo.envs.pettingzoo_wrapper import make_simple_spread


MISSING_PIECES = """
Simple Spread requires three capabilities that wmatrpo doesn't yet support
(by design — the single-step stateless setting was scoped for the differential
game and El Farol). To enable Simple Spread:

  1. State-dependent Gaussian policy
       wmatrpo/policy.py: GaussianPolicy currently has scalar mean/log_std
       parameters. We need an NN head: obs → (μ_i(obs), log σ_i(obs)).
       ~30 lines. Use a 2-layer MLP with action_dim outputs for the mean
       and a separate scalar log_std parameter (or another MLP head).

  2. Episodic trajectory collection with GAE
       Currently algorithm.py collects a single-step batch via env.reward().
       Replace with rollout(env, policies, n_episodes) that calls
       env.reset()/env.step() until done, accumulates (obs, act, rew, next_obs),
       and computes GAE targets λ-returns for the critic.
       ~80 lines.

  3. Critic operates on flattened (obs, act) tuples per timestep
       The existing CentralizedCritic/DecentralizedCritic already work on
       (state_dim, n_agents*action_dim) — they're fine, just need
       state_dim to be the real per-agent observation dimension and
       action features to be the right shape. Minor refactor.

After (1), (2), (3) are in, the algorithms (WMATRPO, IPPO) plug in
unchanged; only the trainer loop and the env wrapper switch.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--n-iterations", type=int, default=1000)
    parser.add_argument("--algorithm", choices=["wgf", "ippo"], default="ippo")
    parser.add_argument("--max-cycles", type=int, default=25)
    args = parser.parse_args()

    # Build the env — this works (so you can verify pettingzoo is installed)
    print(f"Building MPE simple_spread with N={args.n_agents}, max_cycles={args.max_cycles}…")
    try:
        env = make_simple_spread(n_agents=args.n_agents, max_cycles=args.max_cycles)
        obs = env.reset(seed=0)
        print(f"  ✓ Env built. obs.shape={tuple(obs.shape)}, "
              f"action_low={env.action_low}, action_high={env.action_high}")
    except ImportError as e:
        print(f"  ✗ {e}")
        sys.exit(1)

    print()
    print("=" * 70)
    print("MISSING PIECES — implementation required before training works")
    print("=" * 70)
    print(MISSING_PIECES)
    print(f"See: HIGHER_DIFFICULTY_ROADMAP.md in the wmatrpo/ project root.")


if __name__ == "__main__":
    main()
