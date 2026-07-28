"""
Continuous El Farol Bar Problem — Paper 2 §VIII.C.

Setup (per paper):
  - N agents (default 10)
  - Each agent submits a continuous bid a_i ∈ (0, 1) representing commitment
    to attend the bar.
  - Shared reward:   r = -(Σ_i a_i - C)²    where C = bar capacity (default 6.0).
  - Optimal joint strategy requires Σ a_i = C, achieved e.g. by k agents
    committing (a_i → 1) and (N-k) agents staying home (a_i → 0).
  - For N=10, C=6: a 7-committer / 3-stay-home split (or any split that
    sums to 6) maximizes reward.
  - Agents do not observe peer policies; coordination must emerge from
    learning dynamics alone.

The reward shape rewards symmetry-breaking: a monolithic "everyone bids 0.6"
strategy yields the SAME total Σ=6 and the SAME shared reward 0 as a 7-3 split.
But the 7-3 split has the property that small perturbations don't move Σ off C,
while the monolithic equilibrium is fragile. The interesting empirical question
is whether learning dynamics find the bifurcated solution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class ElFarolConfig:
    n_agents: int = 10
    capacity: float = 6.0
    action_low: float = 0.0
    action_high: float = 1.0
    state_dim: int = 1        # constant 1.0 observation


class ElFarolEnv:
    """Continuous El Farol bar problem environment."""

    def __init__(self, cfg: ElFarolConfig):
        self.cfg = cfg
        self.n_agents = cfg.n_agents
        self.state_dim = cfg.state_dim
        self.action_low = cfg.action_low
        self.action_high = cfg.action_high

        # convenience aliases — match the paper's variable naming
        self.C = float(cfg.capacity)

    # -------- reward (eq. 15 in paper) --------
    def reward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        actions: (B, N) or (N,)
        returns: (B,) or scalar  -- shared reward
        """
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        actions = actions.to(torch.float32)
        total = actions.sum(dim=-1)               # (B,)
        r = -((total - self.C) ** 2)               # (B,)
        if squeeze:
            r = r.squeeze(0)
        return r

    # -------- observation --------
    def initial_observation(self, batch_size: int = 1) -> torch.Tensor:
        return torch.ones((batch_size, self.state_dim), dtype=torch.float32)

    def clamp_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions.clamp(self.action_low, self.action_high)

    # -------- diagnostics --------
    def reward_at_uniform(self, action_value: float) -> float:
        """Reward when every agent bids the same value (monolithic equilibrium)."""
        a = torch.full((self.n_agents,), float(action_value), dtype=torch.float32)
        return float(self.reward(a).item())

    def is_bifurcated(self, actions: torch.Tensor, low_thresh: float = 0.25,
                      high_thresh: float = 0.75) -> dict:
        """
        Test whether a joint action exhibits role differentiation.

        Returns a dict with:
            n_high:   number of agents above high_thresh (committing)
            n_low:    number of agents below low_thresh (staying home)
            n_mid:    number of agents in (low_thresh, high_thresh)
            total:    Σ a_i
            shortfall: |Σ a_i - C|
            split:    'n_high - n_low' formatted as a string (e.g. '7-3')
        """
        a = actions.flatten()
        n_high = int((a >= high_thresh).sum().item())
        n_low = int((a <= low_thresh).sum().item())
        n_mid = int(self.n_agents - n_high - n_low)
        total = float(a.sum().item())
        return {
            "n_high": n_high,
            "n_low": n_low,
            "n_mid": n_mid,
            "total": total,
            "shortfall": abs(total - self.C),
            "split": f"{n_high}-{n_low}" + (f" (+{n_mid} mid)" if n_mid else ""),
        }


def make_paper_env(n_agents: int = 10, capacity: float = 6.0) -> ElFarolEnv:
    """Convenience constructor matching the paper's defaults."""
    return ElFarolEnv(ElFarolConfig(n_agents=n_agents, capacity=capacity))
