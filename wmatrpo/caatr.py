"""
Algorithm 2: Coordination-Aware Adaptive Trust Region (CAATR).

Per the paper:

    Eq. 13:  δ_i^(t) = C / (Σ_{j ≠ i} W₁(π_j^(t-1), π_j^(t-2)) + ε)

    Eq. 14:  ε = max(ε_base, min(ε_max, D_{-i} / 10))
             where  D_{-i} = Σ_{j ≠ i} W₁(π_j^(t-1), π_j^(t-2)).

The naming convention "D_{-i}" follows the paper exactly. The drift summed
is the W₁ between consecutive prior policies of each teammate j (i.e., one
step behind the current iteration).

This implementation keeps a per-agent drift buffer and returns δ_i for every
agent at the start of each iteration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class CAATRConfig:
    C: float = 0.02                  # paper Table 2: 0.02 (3,5), 0.10 (7), 0.15 (9)
    epsilon_base: float = 1e-8
    epsilon_max: float = 0.5
    fallback_delta: float = 0.1      # used until enough history exists


class CAATR:
    """Adaptive trust-region rule from Algorithm 2."""

    def __init__(self, cfg: CAATRConfig, n_agents: int):
        self.cfg = cfg
        self.n_agents = n_agents
        # drift_history[i] is a list of W₁(π_i^(t-1), π_i^(t-2)) values
        self.drift_history: List[List[float]] = [[] for _ in range(n_agents)]

    # ---------- core ----------
    def compute_deltas(self) -> List[float]:
        """
        Return list of δ_i, one per agent. Uses the *previous step's* drift
        — the W₁ between t-1 and t-2 policies — as eq. 13 specifies.

        If fewer than 2 drifts have been recorded, returns the fallback δ.
        """
        if any(len(h) < 1 for h in self.drift_history):
            return [self.cfg.fallback_delta] * self.n_agents

        # most recent recorded drift = W₁(π^(t-1), π^(t-2))
        last_drifts = [h[-1] for h in self.drift_history]

        deltas = []
        for i in range(self.n_agents):
            D_minus_i = sum(last_drifts[j] for j in range(self.n_agents) if j != i)
            epsilon = max(
                self.cfg.epsilon_base,
                min(self.cfg.epsilon_max, D_minus_i / 10.0),
            )
            delta_i = self.cfg.C / (D_minus_i + epsilon)
            deltas.append(delta_i)
        return deltas

    def record_drifts(self, drifts: Sequence[float]) -> None:
        """Append the latest per-agent W₁ drift after a full iteration."""
        if len(drifts) != self.n_agents:
            raise ValueError(
                f"got {len(drifts)} drifts but n_agents = {self.n_agents}"
            )
        for i, d in enumerate(drifts):
            self.drift_history[i].append(float(d))

    def last_drifts(self) -> List[float]:
        return [h[-1] if h else 0.0 for h in self.drift_history]
