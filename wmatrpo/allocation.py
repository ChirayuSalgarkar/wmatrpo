"""
Trust-region-radius allocation strategies for the basin-gap ablation (paper
Table 5 / "Fixed, Greedy, Weighted, +CAATR").

These are the non-CAATR ways of setting each agent's per-step trust radius δ_i
from a shared budget ε_total. They are ported faithfully from the original
SAILRIT repo's `ottrpocomparisonforfixedgreedyweightedcaatr.py` (methods
`fixed`, `greedy`, `weighted`) and re-expressed behind the SAME interface as
`wmatrpo.caatr.CAATR` — `compute_deltas()` and `record_drifts()` — so any of
them can be dropped into the W-MATRPO algorithm in place of CAATR.

Strategies (verbatim logic from the original):
  * Fixed     : every agent gets the full budget ε_total (standard TRPO radius).
  * Greedy    : score_i = |mean advantage_i| / (last W₁ drift_i + 1e-8);
                allocate ε_total ∝ normalized scores. (HATRPO-G inspired.)
  * Weighted  : water-filling on positive mean advantages — allocate
                ε_total via a bisection on the water level λ so that
                Σ max(0, u_i/λ − 1e-4) = ε_total, then renormalize.
                (HATRPO-W inspired.)

Because Greedy/Weighted need per-agent mean advantages, the algorithm feeds them
in each step via `set_advantages(...)` before `compute_deltas()` is called; if
none have been supplied yet, all three fall back to a uniform / full-budget
split exactly as the original did.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class AllocationConfig:
    method: str = "fixed"          # "fixed" | "greedy" | "weighted"
    epsilon_total: float = 0.1     # shared trust-region budget (paper's ε_total)


class AllocationStrategy:
    """Drop-in replacement for CAATR with fixed/greedy/weighted budget splits."""

    def __init__(self, cfg: AllocationConfig, n_agents: int):
        if cfg.method not in ("fixed", "greedy", "weighted"):
            raise ValueError(f"unknown allocation method: {cfg.method}")
        self.cfg = cfg
        self.n_agents = n_agents
        self.drift_history: List[List[float]] = [[] for _ in range(n_agents)]
        self._avg_adv: Optional[np.ndarray] = None

    # --- hook the algorithm calls before compute_deltas() each step ---
    def set_advantages(self, avg_advantages: Sequence[float]) -> None:
        self._avg_adv = np.asarray(avg_advantages, dtype=float)

    def compute_deltas(self) -> List[float]:
        eps = self.cfg.epsilon_total
        n = self.n_agents

        if self.cfg.method == "fixed":
            return [eps] * n

        # greedy / weighted need mean advantages; fall back to uniform if absent
        if self._avg_adv is None:
            return [eps / n] * n
        avg_adv = self._avg_adv

        if self.cfg.method == "greedy":
            last_w = np.array([h[-1] if h else 0.0 for h in self.drift_history])
            scores = np.abs(avg_adv) / (last_w + 1e-8)
            scores = np.nan_to_num(scores, nan=0.0)
            if scores.sum() < 1e-8:
                return [eps / n] * n
            return (scores / scores.sum() * eps).tolist()

        # weighted — water-filling
        utilities = np.maximum(0.0, avg_adv)
        if utilities.sum() < 1e-8:
            return [eps / n] * n
        lam = float(utilities.max()) + 1e-6
        for _ in range(10):
            alloc = np.maximum(0.0, utilities / lam - 1e-4)
            total = float(alloc.sum())
            if abs(total - eps) < 1e-5:
                break
            if total < 1e-8:
                lam *= 0.5
            else:
                lam *= (total / eps)
        final = np.maximum(0.0, utilities / lam - 1e-4)
        if final.sum() > 1e-8:
            return (final / final.sum() * eps).tolist()
        return [eps / n] * n

    def record_drifts(self, drifts: Sequence[float]) -> None:
        if len(drifts) != self.n_agents:
            raise ValueError(f"got {len(drifts)} drifts but n_agents = {self.n_agents}")
        for i, d in enumerate(drifts):
            self.drift_history[i].append(float(d))

    def last_drifts(self) -> List[float]:
        return [h[-1] if h else 0.0 for h in self.drift_history]
