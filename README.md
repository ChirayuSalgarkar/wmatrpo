# W-MATRPO — Faithful Reference Implementation

Clean reimplementation of the W-MATRPO + CAATR algorithm from

> Salgarkar, Bailey, Alm, Baheri. *Wasserstein-Constrained Trust Region Optimization for Cooperative Multi-Agent Reinforcement Learning.*

This implementation is **faithful to the paper's algorithm description** as a corrective to the simplifications present in the original `SAILRIT/TRPOMARL` repo (see `paper1_code_audit.md` in the parent folder for details).

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
python -m wmatrpo.tests.test_baselines_smoke          # verify all algorithms run first
python -m wmatrpo.scripts.baselines --quick           # fast sanity sweep (3 agents, 1 seed)
python -m wmatrpo.scripts.baselines --agent-counts 3 5 7 9 --seeds 0 1 2   # full R2.4 table

# main-results table (Table 3): Standard HATRPO vs HATRPO+CAATR vs W-MATRPO+CAATR
python -m wmatrpo.scripts.baselines --algorithms hatrpo hatrpo_caatr wmatrpo \
    --agent-counts 3 5 7 9 --seeds 0 1 2 --out runs/table3

# basin-gap allocation comparison (Table 5): Fixed / Greedy / Weighted / +CAATR
python -m wmatrpo.scripts.basingap_compare --ks 0.5 1.0 1.5 2.0 --seeds 0 1 2
```

## Baselines and comparison algorithms

Every comparison algorithm shares the env, batch size, centralized critic, and
policy init with W-MATRPO, so each comparison isolates a single design choice.

**PPO-family baselines (Table 4 / reviewer R2.4):**

- **IPPO** (`ippo.py`) — independent PPO; runs with a centralized or decentralized critic.
- **MAPPO** (`mappo.py`) — concurrent PPO with a centralized critic (Yu et al. 2021/2022).
- **HAPPO** (`happo.py`) — sequential PPO with importance-sampling correction (Kuba et al. 2021); the closest PPO cousin to W-MATRPO's sequential+IS structure, with the PPO clip replacing the Wasserstein dual.

**KL-trust-region comparison (Table 3 / main results):**

- **HATRPO** (`hatrpo.py`) — Heterogeneous-Agent TRPO (Kuba et al. 2021): the *same* sequential+IS machinery as W-MATRPO, but each agent takes a natural-gradient step under a hard KL trust region (conjugate gradient + Fisher-vector products + backtracking line search) instead of the Wasserstein dual solve. Selected as `hatrpo` (fixed KL radius) or `hatrpo_caatr` (CAATR-adaptive radius) in `scripts/baselines.py`. This is the cleanest controlled contrast: swap the *geometry* of the trust region (KL vs Wasserstein) and hold everything else fixed. On the differential game the KL constraint keeps HATRPO trapped in the local optimum (distance ≈ 4√N), reproducing the paper's Standard-HATRPO / HATRPO+CAATR rows; W-MATRPO's Wasserstein constraint escapes.

**W-MATRPO variants (CAATR ablation):**

- **`wmatrpo`** — W-MATRPO with the coordination-aware adaptive trust region (CAATR); the full proposed method.
- **`wmatrpo_fixed`** — W-MATRPO with a **fixed** (non-adaptive) trust-region radius in place of CAATR. Identical Wasserstein dual solve, critic, and initialization; only the radius rule differs. Running both across N isolates CAATR's marginal contribution from the underlying Wasserstein geometry — i.e. it separates "does optimal-transport geometry help?" from "does making its radius adaptive help?". Both selectable in `scripts/baselines.py`.

**Trust-region allocation strategies (Table 5 / basin-gap ablation):**

- **`allocation.py`** — the Fixed / Greedy / Weighted budget-allocation rules ported faithfully from the original repo, re-expressed behind the same interface as CAATR so any of them drops into W-MATRPO's radius slot. `scripts/basingap_compare.py` sweeps k ∈ {0.5,1,1.5,2} × {fixed, greedy, weighted, caatr} on the 2-agent game and writes `basingap_summary.csv` (mean±s.d. over seeds).

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
    ├── caatr.py               # Algorithm 2 (Coordination-Aware Adaptive Trust Region)
    ├── allocation.py          # Fixed/Greedy/Weighted radius allocation (Table 5)
    ├── algorithm.py           # Algorithm 1 (Sequential + IS)
    ├── happo.py               # HAPPO baseline (Kuba 2021, PPO clip)
    ├── hatrpo.py              # HATRPO comparison (Kuba 2021, KL trust region)
    ├── ippo.py / mappo.py     # PPO-family baselines
    ├── trainer.py             # train loop + logging
    ├── utils.py               # seeding, config loading
    ├── scripts/
    │   ├── train.py           # CLI entry point
    │   ├── baselines.py       # Table 3 (hatrpo/hatrpo_caatr/wmatrpo) + Table 4 (R2.4) sweeps
    │   ├── basingap.py        # basin-gap runner (W-MATRPO+CAATR only)
    │   └── basingap_compare.py# Table 5 allocation-strategy comparison
    └── tests/
        ├── test_wasserstein.py
        ├── test_dual_solver.py
        ├── test_hatrpo.py         # KL-constraint + local-optimum-trap checks
        ├── test_allocation.py     # budget-conservation + ranking checks
        └── test_baselines_smoke.py
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
- Table 3 (main results — HATRPO / HATRPO+CAATR / W-MATRPO+CAATR): `hatrpo.py` + `scripts/baselines.py` (algorithms `hatrpo`, `hatrpo_caatr`, `wmatrpo`)
- Table 4 (R2.4 — W-MATRPO vs IPPO/MAPPO/HAPPO): `ippo.py`, `mappo.py`, `happo.py` + `scripts/baselines.py`
- Table 5 (basin-gap — Fixed/Greedy/Weighted/+CAATR): `allocation.py` + `scripts/basingap_compare.py`

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
