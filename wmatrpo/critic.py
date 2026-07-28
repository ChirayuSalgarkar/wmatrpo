"""
Critic networks.

We expose two flavors:

  - CentralizedCritic
      V_φ(s) and Q_φ(s, joint_action). Q sees the full joint action vector.
      This is the §3.3 specification in Paper 1 and the setup Paper 2's WGF
      experiment assumes.

  - DecentralizedCritic
      Per-agent V_i(s) and Q_i(s, a_i). Each agent has its own critic that
      sees only its own action. This is the IPPO baseline setup in Paper 2
      Case C and the natural control for "no joint information."

Both expose the same `advantage(s, a)` interface; the difference is what
each agent's advantage is computed from. The algorithms (WMATRPO, IPPO)
work with either.

For the differential-game / El Farol tasks (single-step, γ = 0), the TD
target reduces to the per-batch reward. We expose `gamma` for future
multi-step extensions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CriticConfig:
    state_dim: int = 1
    n_agents: int = 2
    hidden: int = 64
    n_layers: int = 2
    lr: float = 3e-3
    gamma: float = 0.0
    n_update_epochs: int = 8


def _mlp(in_dim: int, out_dim: int, hidden: int, n_layers: int) -> nn.Sequential:
    layers = []
    d = in_dim
    for _ in range(n_layers):
        layers.append(nn.Linear(d, hidden))
        layers.append(nn.Tanh())
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


# =============================================================================
# Centralized: V_φ(s), Q_φ(s, joint_action)
# =============================================================================
class CentralizedCritic(nn.Module):
    """
    V(s) and Q(s, joint_a). Single set of parameters shared across agents.
    """

    def __init__(self, cfg: CriticConfig):
        super().__init__()
        self.cfg = cfg
        self.v_net = _mlp(cfg.state_dim, 1, cfg.hidden, cfg.n_layers)
        self.q_net = _mlp(cfg.state_dim + cfg.n_agents, 1, cfg.hidden, cfg.n_layers)
        self.optim = torch.optim.Adam(self.parameters(), lr=cfg.lr)
        self.is_centralized = True

    def value(self, s: torch.Tensor) -> torch.Tensor:
        return self.v_net(s).squeeze(-1)

    def q_value(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.q_net(torch.cat([s, a], dim=-1)).squeeze(-1)

    def advantage(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """A(s, a) = Q(s, joint_a) - V(s). Same advantage seen by every agent."""
        return self.q_value(s, a) - self.value(s)

    def per_agent_advantage(self, s: torch.Tensor, a: torch.Tensor,
                            agent_id: int) -> torch.Tensor:
        """Same advantage for every agent (centralized critic)."""
        return self.advantage(s, a)

    def update(self, batch: dict) -> dict:
        s, a, r = batch["states"], batch["actions"], batch["rewards"]
        last = {}
        for _ in range(self.cfg.n_update_epochs):
            q_pred = self.q_value(s, a)
            v_pred = self.value(s)
            q_loss = F.mse_loss(q_pred, r)
            v_loss = F.mse_loss(v_pred, torch.full_like(v_pred, float(r.mean())))
            loss = q_loss + v_loss
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            last = {"q_loss": float(q_loss.item()), "v_loss": float(v_loss.item())}
        return last


# =============================================================================
# Decentralized: per-agent V_i(s), Q_i(s, a_i)
# =============================================================================
class DecentralizedCritic(nn.Module):
    """
    Per-agent V_i(s) and Q_i(s, a_i). No joint information.

    Each agent's "advantage" is computed from its own (local) Q and V; the
    cooperative reward is still observed (we use the joint shared reward
    as the regression target for every agent's Q), but each Q_i only sees
    a_i, not a_{-i}. This is the IPPO setup in Paper 2 Case C.
    """

    def __init__(self, cfg: CriticConfig):
        super().__init__()
        self.cfg = cfg
        self.is_centralized = False
        # one V network per agent; each one sees only s
        self.v_nets = nn.ModuleList([
            _mlp(cfg.state_dim, 1, cfg.hidden, cfg.n_layers)
            for _ in range(cfg.n_agents)
        ])
        # one Q network per agent; each one sees (s, a_i)
        self.q_nets = nn.ModuleList([
            _mlp(cfg.state_dim + 1, 1, cfg.hidden, cfg.n_layers)
            for _ in range(cfg.n_agents)
        ])
        self.optim = torch.optim.Adam(self.parameters(), lr=cfg.lr)

    def value(self, s: torch.Tensor, agent_id: int = 0) -> torch.Tensor:
        return self.v_nets[agent_id](s).squeeze(-1)

    def q_value(self, s: torch.Tensor, a_i: torch.Tensor, agent_id: int = 0
                ) -> torch.Tensor:
        a_i = a_i.unsqueeze(-1) if a_i.dim() == 1 else a_i
        return self.q_nets[agent_id](torch.cat([s, a_i], dim=-1)).squeeze(-1)

    def advantage(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        For interface compatibility with the dual solver: returns the
        per-agent advantage as a (B,) vector AVERAGED across agents.
        Use per_agent_advantage(s, a, i) for the actual per-agent signal.
        """
        adv = torch.stack([
            self.per_agent_advantage(s, a, i) for i in range(self.cfg.n_agents)
        ], dim=-1).mean(dim=-1)
        return adv

    def per_agent_advantage(self, s: torch.Tensor, a: torch.Tensor,
                            agent_id: int) -> torch.Tensor:
        """A_i(s, a_i) = Q_i(s, a_i) - V_i(s)."""
        return (self.q_value(s, a[:, agent_id], agent_id)
                - self.value(s, agent_id))

    def update(self, batch: dict) -> dict:
        s, a, r = batch["states"], batch["actions"], batch["rewards"]
        last = {}
        for _ in range(self.cfg.n_update_epochs):
            losses = []
            for i in range(self.cfg.n_agents):
                q_pred = self.q_value(s, a[:, i], i)
                v_pred = self.value(s, i)
                losses.append(F.mse_loss(q_pred, r))
                losses.append(F.mse_loss(v_pred, torch.full_like(v_pred, float(r.mean()))))
            loss = torch.stack(losses).sum()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            last = {"total_loss": float(loss.item())}
        return last


def build_critic(cfg: CriticConfig, centralized: bool = True):
    """Factory: returns a CentralizedCritic or DecentralizedCritic per the flag."""
    return CentralizedCritic(cfg) if centralized else DecentralizedCritic(cfg)
