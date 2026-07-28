# W-MATRPO — Faithful Reference Implementation

Clean reimplementation of the W-MATRPO + CAATR algorithm from

> Salgarkar, Bailey, Alm, Baheri. *Wasserstein-Constrained Trust Region Optimization for Cooperative Multi-Agent Reinforcement Learning.*

This implementation is **faithful to the paper's algorithm description**, as a corrective to the simplifications present in the original `SAILRIT/TRPOMARL` repo. The specific differences are listed in the table below. The reported baseline comparisons (reviewer R2.4) and the σ-inflation / non-convergence diagnosis (reviewer R2.3) were produced with this implementation; running the scripts below regenerates them from scratch.

## What's different from the original repo

| Component | Original repo | This implementation |
|---|---|---|
| Critic | EMA reward baseline | Centralized `V_φ(s)` + `Q_φ(s,a)` MLP networks (§3.3) |
| Dual problem (Theorem 1) | KDE-smoothed bootstrap | Closed-form for Gaussian + L₂ (Remark 1): `a* = a + ∇A/(2λ)` |
| Optimal policy (Corollary 1) | Moment-matching to Gaussian on 50-pt grid | Pushforward of samples through closed-form `a*`, then MLE-fit Gaussian |
| Importance sampling (eq. 15) | Not implemented | Implemented for sequential updates |
| Agent ordering (Alg. 1 line 5) | Fixed `0..N-1` | Random permutation per iteration |
| CAATR adaptive ε (eq. 14) | Hard-coded `1e-8` | `max(ε_base, min(ε_max, D_{-i}/10))` per spec |
| CAATR drift (eq. 13) | 1-step | 2-step (consecutive prior W₁ distances) |
| W₁ between Gaussians (eq. 31) | `|Δμ| + |Δσ|` (buggy) | Numerical integral of true W₁ (Vallender identity) |
| Transport cost | Mixed L₁ / L₂ across scripts | Single configurable cost, default L₂ for W₂² geometry |
| Codepath per N | Separate scripts | Single generic algorithm parameterized by config |

## Quick start

```bash
# from this directory
pip install -r requirements.txt

# run the 3-agent differential game
python -m wmatrpo.scripts.train --config configs/diffgame_3agent.yaml

# run the basin-gap ablation (Table 4 in paper)
python -m wmatrpo.scripts.basingap --config configs/basingap.yaml

# run all five agent counts (Table 3 in paper)
for n in 2 3 5 7 9; do
    python -m wmatrpo.scripts.train --config configs/diffgame_${n}agent.yaml
done

# head-to-head baselines: W-MATRPO vs IPPO vs MAPPO vs HAPPO  (reviewer R2.4)
python -m wmatrpo.tests.test_baselines_smoke          # verify all four run first
python -m wmatrpo.scripts.baselines --quick           # fast sanity sweep (3 agents, 1 seed)
python -m wmatrpo.scripts.baselines --agent-counts 3 5 7 9 --seeds 0 1 2   # full table
```

## Baselines (IPPO / MAPPO / HAPPO)

All three PPO-family baselines share the env, batch size, centralized critic, and
policy init with W-MATRPO, so the comparison isolates the policy-update rule:

- **IPPO** (`ippo.py`) — independent PPO; runs with a centralized or decentralized critic.
- **MAPPO** (`mappo.py`) — concurrent PPO with a centralized critic (Yu et al. 2021).
- **HAPPO** (`happo.py`) — sequential PPO with importance-sampling correction (Kuba et al. 2021); the closest KL-based cousin to W-MATRPO's sequential+IS structure, with PPO clip replacing the Wasserstein dual.

`scripts/baselines.py` produces `baselines_raw.csv`, `baselines_summary.csv`
(mean±std over seeds), and `baselines_pivot.csv` (distance-to-global, algorithm × N).

Outputs land in `runs/<run_name>/` with:
- `trajectory.csv` — per-iteration agent means
- `rewards.csv` — per-iteration mean reward
- `drift.csv` — per-iteration per-agent W₁ drift
- `deltas.csv` — per-iteration per-agent adaptive trust-region radius
- `config.yaml` — exact config used
- `summary.json` — final actions, distance to global optimum, peak reward

## Package structure

```
wmatrpo/
├── README.md                  # this file
├── requirements.txt           # pinned dependencies
├── configs/
│   ├── diffgame_2agent.yaml
│   ├── diffgame_3agent.yaml
│   ├── diffgame_5agent.yaml
│   ├── diffgame_7agent.yaml
│   ├── diffgame_9agent.yaml
│   └── basingap.yaml
└── wmatrpo/
    ├── __init__.py
    ├── env.py                 # DifferentialGameEnv (Eq. 30)
    ├── policy.py              # GaussianPolicy with closed-form W₁ / W₂
    ├── critic.py              # V and Q networks (small MLPs)
    ├── dual_solver.py         # Theorem 1 + Remark 1 + Corollary 1
    ├── caatr.py               # Algorithm 2
    ├── algorithm.py           # Algorithm 1 (Sequential + IS)
    ├── trainer.py             # train loop + logging
    ├── utils.py               # seeding, config loading
    └── scripts/
        ├── train.py           # CLI entry point
        └── basingap.py        # Table 4 ablation runner
```

## Tracing the paper to the code

Every load-bearing equation/algorithm in the paper maps to a specific file:

- Eq. 1 (objective): `env.py` (the `reward` function and `Trainer.train` loop)
- Eq. 2 (W₁ definition): `policy.py:GaussianPolicy.wasserstein_1`
- Eq. 3 (optimization setup): `algorithm.py:WMATRPO.step`
- Eq. 4 (dual representation), Eq. 5 (Φ_λ): `dual_solver.py:DualSolver.solve`
- Eq. 6 (Lagrangian), Remark 1 (closed form): `dual_solver.py:_closed_form_pushforward`
- Eq. 13 (CAATR radius), Eq. 14 (adaptive ε): `caatr.py:CAATR.compute_deltas`
- Eq. 15 (IS correction): `algorithm.py:WMATRPO._is_corrected_advantage`
- Eq. 16 (loss), Eqs. 17–18 (updates), Eq. 19 (critic TD): `algorithm.py` and `critic.py`
- Eq. 30 (differential game reward): `env.py:DifferentialGameEnv.reward`
- Corollary 1 (optimal policy form): `dual_solver.py:DualSolver._fit_gaussian_to_pushforward`
- Proposition 1 (surrogate bound): not a function — but the surrogate value is logged in `trainer.py`
- Algorithm 1 (W-MATRPO with CAATR): `algorithm.py:WMATRPO.step`
- Algorithm 2 (CAATR update): `caatr.py:CAATR.compute_deltas`

## Sanity checks (run before trusting results)

```bash
# verifies dual solver gets the right answer on a known 1D quadratic problem
python -m wmatrpo.tests.test_dual_solver

# verifies W₁ between Gaussians matches the corrected formula in the audit
python -m wmatrpo.tests.test_wasserstein
```

## Dependencies

- `torch>=2.0`
- `numpy>=1.24`
- `scipy>=1.10`
- `pyyaml>=6.0`
- `pandas>=2.0` (for CSV logging)

No GPU required; the differential-game tasks run on CPU in a few minutes per configuration.
