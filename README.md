# W-MATRPO — Faithful Reference Implementation

Reference implementation of W-MATRPO + CAATR from

> Salgarkar, Bailey, Alm, Baheri. *Wasserstein-Constrained Trust Region Optimization for Cooperative Multi-Agent Reinforcement Learning.*

Faithful to the paper's Algorithm 1 / Theorem 1 / Corollary 1. No GPU required; every
experiment runs on CPU.

## Install

```bash
pip install -r requirements.txt
```

## Verify the install (run first — seconds each)

```bash
python -m wmatrpo.tests.test_wasserstein        # W₁ between Gaussians
python -m wmatrpo.tests.test_dual_solver         # Theorem 1 dual solve
python -m wmatrpo.tests.test_hatrpo              # HATRPO KL constraint + local-opt trap
python -m wmatrpo.tests.test_allocation          # budget-allocation rules
python -m wmatrpo.tests.test_baselines_smoke     # all algorithms run and move the policy
```

## Reproduce the paper's tables

All sweeps are seeded (bit-for-bit reproducible) and write `*_raw.csv` (per seed) plus a
`*_summary.csv` (mean ± s.d.) into `--out`.

```bash
# Unified comparison — IPPO / MAPPO / HAPPO / HATRPO / HATRPO+CAATR /
# W-MATRPO (fixed radius) / W-MATRPO+CAATR, across N ∈ {3,5,7,9}
python -m wmatrpo.scripts.baselines \
    --algorithms ippo mappo happo hatrpo hatrpo_caatr wmatrpo_fixed wmatrpo \
    --agent-counts 3 5 7 9 --seeds 0 1 2 --n-iterations 4000 --out runs/main

# Basin-gap allocation ablation — Fixed / Greedy / Weighted / CAATR (2 agents)
python -m wmatrpo.scripts.basingap_compare \
    --ks 0.5 1.0 1.5 2.0 --strategies fixed greedy weighted caatr \
    --seeds 0 1 2 --n-iterations 2500 --out runs/basingap
```

## Single run + diagnostics

```bash
python -m wmatrpo.scripts.train --config configs/diffgame_3agent.yaml
```

Writes `runs/<name>/` with per-iteration `trajectory.csv`, `rewards.csv`, `drift.csv`,
`deltas.csv` (CAATR radii), the exact `config.yaml`, and a final `summary.json`.

## Algorithm selectors (for `scripts.baselines --algorithms ...`)

| selector | method |
|---|---|
| `wmatrpo` | W-MATRPO with CAATR (full proposed method) |
| `wmatrpo_fixed` | W-MATRPO with a fixed radius (CAATR ablation) |
| `hatrpo` / `hatrpo_caatr` | HATRPO with a fixed / CAATR-adaptive KL radius |
| `ippo` / `mappo` / `happo` | PPO-family baselines (matched setup) |

## Layout

```
wmatrpo/
  algorithm.py    Algorithm 1 (sequential + IS)
  dual_solver.py  Theorem 1 + Remark 1 + Corollary 1
  caatr.py        Algorithm 2 (CAATR adaptive radius)
  allocation.py   Fixed / Greedy / Weighted radius rules
  policy.py       Gaussian policy + W₁
  critic.py       centralized V / Q networks
  env.py          differential game
  hatrpo.py happo.py ippo.py mappo.py   comparison algorithms
  scripts/        baselines.py, basingap_compare.py, train.py
  tests/          the five checks above
```

Full step-by-step reproduction (expected numbers, runtimes, paper→code map) is in
`VERIFY_AND_RUN.md` in the revision bundle.
