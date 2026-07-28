"""
HAPPO — Heterogeneous-Agent PPO (Kuba et al., 2021, "Trust Region Policy
Optimisation in Multi-Agent Reinforcement Learning").

HAPPO is the sequential-update, importance-sampling-corrected multi-agent PPO
that carries a monotonic-improvement guarantee. It is the natural PPO-based
baseline against which W-MATRPO's Wasserstein trust region should be compared:
same sequential/IS structure, but KL-style PPO clipping instead of a Wasserstein
constraint.

Per iteration:
  1. Collect a joint batch; fit the centralized critic.
  2. Draw a random agent order σ.
  3. For k = 1..N, update agent i = σ(k):
       - cumulative IS weight from already-updated agents:
             w = ∏_{j ∈ U_k} π_j^new(a_j|s) / π_j^old(a_j|s)          (eq. 15 analogue)
       - corrected advantage:  M_i = w · A(s, a)
       - PPO-clip surrogate on agent i's own ratio r_i = π_i^new(a_i)/π_i^old(a_i):
             L_i = E[ min( r_i · M_i, clip(r_i, 1±ε) · M_i ) ]
       - gradient-ascend agent i's parameters on L_i.
       - add i to U_k.

This mirrors W-MATRPO's Algorithm 1 (random order + IS correction) but with the
PPO clip replacing the Wasserstein dual solve — which is exactly the controlled
comparison reviewers asked for.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch

from wmatrpo.policy import GaussianPolicy, PolicyConfig
from wmatrpo.critic import CentralizedCritic


@dataclass
class HAPPOConfig:
    batch_size: int = 30
    n_agents: int = 2
    clip_eps: float = 0.2
    n_policy_epochs: int = 4
    policy_lr: float = 3e-4
    seed: int = 0


class HAPPO:
    """Heterogeneous-Agent PPO with sequential updates and IS correction."""

    def __init__(self, env, policies: Sequence[GaussianPolicy],
                 critic: CentralizedCritic, cfg: HAPPOConfig):
        if not getattr(critic, "is_centralized", False):
            raise ValueError("HAPPO uses a centralized critic for the joint advantage.")
        self.env = env
        self.policies = list(policies)
        self.critic = critic
        self.cfg = cfg
        self.n_agents = env.n_agents

        self.optims = [
            torch.optim.Adam(p.parameters(), lr=cfg.policy_lr) for p in self.policies
        ]
        self.rng = np.random.default_rng(cfg.seed)

        self.iteration = 0
        self.reward_history: List[float] = []
        self.mean_history: List[List[float]] = [[float(p.mean.item())] for p in self.policies]
        self.std_history: List[List[float]] = [[float(p.std.item())] for p in self.policies]

    def step(self) -> dict:
        batch = self._collect_batch()
        critic_info = self.critic.update(batch)

        # freeze the pre-iteration policies (the "old" policies for IS ratios)
        old_policies = [p.snapshot() for p in self.policies]
        old_logps = [old_policies[i].log_prob(batch["actions"][:, i]).detach()
                     for i in range(self.n_agents)]

        with torch.no_grad():
            A = self.critic.advantage(batch["states"], batch["actions"])  # (B,)
            A = (A - A.mean()) / (A.std() + 1e-8)

        order = self.rng.permutation(self.n_agents).tolist()
        updated: List[int] = []

        for i in order:
            # cumulative IS weight from previously updated agents (eq. 15 analogue)
            with torch.no_grad():
                if updated:
                    log_w = torch.zeros_like(A)
                    for j in updated:
                        aj = batch["actions"][:, j]
                        log_w = log_w + self.policies[j].log_prob(aj) - old_logps[j]
                    log_w = torch.clamp(log_w, -10.0, 10.0)
                    w = torch.exp(log_w)
                else:
                    w = torch.ones_like(A)
                M_i = w * A                      # corrected advantage

            ai = batch["actions"][:, i]
            for _ in range(self.cfg.n_policy_epochs):
                new_logp = self.policies[i].log_prob(ai)
                ratio = torch.exp(new_logp - old_logps[i])
                surr1 = ratio * M_i
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps,
                                    1 + self.cfg.clip_eps) * M_i
                loss = -torch.min(surr1, surr2).mean()
                self.optims[i].zero_grad()
                loss.backward()
                self.optims[i].step()

            updated.append(i)

        self.iteration += 1
        avg_reward = float(batch["rewards"].mean().item())
        self.reward_history.append(avg_reward)
        for k in range(self.n_agents):
            self.mean_history[k].append(float(self.policies[k].mean.item()))
            self.std_history[k].append(float(self.policies[k].std.item()))

        return {"iteration": self.iteration, "avg_reward": avg_reward,
                "critic": critic_info, "agent_order": order}

    def _collect_batch(self) -> dict:
        B = self.cfg.batch_size
        with torch.no_grad():
            actions = torch.stack([p.sample(B) for p in self.policies], dim=-1)
            actions = self.env.clamp_actions(actions)
            states = self.env.initial_observation(B)
            rewards = self.env.reward(actions)
        return {"states": states, "actions": actions, "rewards": rewards}

    @classmethod
    def from_config(cls, env, happo_cfg: HAPPOConfig, policy_cfg: PolicyConfig, critic):
        policies = [GaussianPolicy(policy_cfg) for _ in range(env.n_agents)]
        return cls(env=env, policies=policies, critic=critic, cfg=happo_cfg)
