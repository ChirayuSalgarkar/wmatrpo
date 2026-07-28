"""
Differential-game environment.

Implements Eq. 30 from the paper:

    r(a_1, ..., a_n) = α_g · exp(-½ Σ_i (a_i - 5)² / σ_i²)
                    + α_l · exp(-½ Σ_i (a_i - 1)²)

with α_g = 10 / ((2π)^(n/2) · ∏√σ_i), α_l = 6.5 / (2π)^(n/2).

The basin-gap factor `k` (for Table 4 ablation, Eq. 32) scales α_l.

State is a fixed constant — the paper carries out its analysis statewise.
We expose a 1-dim "observation" of constant value 1.0 so that downstream
networks have a non-trivial input, but everything is invariant under state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import pi, sqrt
from typing import Optional

import numpy as np
import torch


@dataclass
class DifferentialGameConfig:
    """Configuration for the n-agent differential game (Eq. 30 / Eq. 32)."""
    n_agents: int = 2
    # Per-agent variance in the global-optimum Gaussian. Paper convention:
    #   σ_1 = σ_3 = 1, σ_2 = 3; others default to 1.
    variances: Optional[list] = None
    local_optimum: float = 1.0           # all agents' coords for local opt
    global_optimum: float = 5.0          # all agents' coords for global opt
    basin_gap_factor: float = 1.0        # k in Eq. 32; scales α_l
    linear_bias: float = 0.0             # extra linear-in-a_1 term (some scripts use 0.1)
    action_low: float = 0.0
    action_high: float = 7.0
    state_dim: int = 1                   # constant 1.0 observation

    def __post_init__(self):
        if self.variances is None:
            v = [1.0] * self.n_agents
            if self.n_agents >= 2:
                v[1] = 3.0   # σ_2 = 3, paper convention
            self.variances = v
        if len(self.variances) != self.n_agents:
            raise ValueError(
                f"variances has length {len(self.variances)} but n_agents is {self.n_agents}"
            )


class DifferentialGameEnv:
    """
    Stateless cooperative differential game on continuous actions in [0, 7]^N.
    """

    def __init__(self, cfg: DifferentialGameConfig):
        self.cfg = cfg
        self.n_agents = cfg.n_agents
        self.state_dim = cfg.state_dim
        self.action_low = cfg.action_low
        self.action_high = cfg.action_high

        v = torch.as_tensor(cfg.variances, dtype=torch.float64)
        self.register_variances(v)

        # Coefficients from Eq. 30
        n = self.n_agents
        self.global_coef = 10.0 / (((2 * pi) ** (n / 2)) * float(torch.sqrt(v.prod())))
        self.local_coef = (6.5 / ((2 * pi) ** (n / 2))) * cfg.basin_gap_factor

        # Optima vectors
        self.global_opt = torch.full((n,), cfg.global_optimum, dtype=torch.float64)
        self.local_opt = torch.full((n,), cfg.local_optimum, dtype=torch.float64)

    def register_variances(self, v: torch.Tensor):
        # Stored as buffer-style (move-with-device) for downstream tensors
        self._variances = v

    # -------- reward (Eq. 30) --------
    def reward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        actions: (B, N) or (N,)
        returns: (B,) or scalar
        """
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        actions = actions.to(torch.float64)
        var = self._variances.to(actions.device)
        glob = self.global_opt.to(actions.device)
        loc = self.local_opt.to(actions.device)

        # global Gaussian
        global_quad = ((actions - glob) ** 2 / var).sum(dim=-1)
        global_term = torch.exp(-0.5 * global_quad)

        # local Gaussian
        local_quad = ((actions - loc) ** 2).sum(dim=-1)
        local_term = torch.exp(-0.5 * local_quad)

        r = self.global_coef * global_term + self.local_coef * local_term
        if self.cfg.linear_bias != 0.0:
            r = r + self.cfg.linear_bias * actions[..., 0]

        if squeeze:
            r = r.squeeze(0)
        return r.to(torch.float32)

    # -------- observation --------
    def initial_observation(self, batch_size: int = 1) -> torch.Tensor:
        """Return constant observation tensor (B, state_dim)."""
        return torch.ones((batch_size, self.state_dim), dtype=torch.float32)

    # -------- helpers --------
    def clamp_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions.clamp(self.action_low, self.action_high)

    def reward_at(self, action_value: float) -> float:
        """Convenience: scalar reward at all-agents-take-the-same-action."""
        a = torch.full((self.n_agents,), float(action_value), dtype=torch.float64)
        return float(self.reward(a))


def make_paper_env(n_agents: int, basin_gap_factor: float = 1.0,
                   linear_bias: float = 0.0) -> DifferentialGameEnv:
    """Convenience constructor matching the paper's defaults."""
    return DifferentialGameEnv(DifferentialGameConfig(
        n_agents=n_agents,
        basin_gap_factor=basin_gap_factor,
        linear_bias=linear_bias,
    ))
