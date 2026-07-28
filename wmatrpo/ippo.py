"""
IPPO — Independent Proximal Policy Optimization.

Each agent maintains an independent Gaussian policy and updates it via PPO's
clipped surrogate objective. The critic can be either centralized
(CentralizedCritic, V(s) + Q(s, joint_a)) or decentralized (DecentralizedCritic,
per-agent V_i(s) + Q_i(s, a_i)).

The Paper 2 §VIII.C "IPPO" baseline corresponds to IPPO + DecentralizedCritic.
The Case C clean comparison also runs IPPO + CentralizedCritic.

PPO clipped surrogate (Schulman et al. 2017):

    ratio_i(a_i) = π_i^new(a_i | s) / π_i^old(a_i | s)
    L^CLIP = E[ min( ratio · A_i,  clip(ratio, 1-ε, 1+ε) · A_i ) ]

Updated parameters: each agent maximizes L^CLIP w.r.t. its own (μ_i, log σ_i)
for n_epochs of Adam steps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from wmatrpo.policy import GaussianPolicy, PolicyConfig
from wmatrpo.critic import CentralizedCritic, DecentralizedCritic


@dataclass
class IPPOConfig:
    batch_size: int = 30
    n_agents: int = 10
    clip_eps: float = 0.2
    n_policy_epochs: int = 4
    policy_lr: float = 3e-4
    seed: int = 0


class IPPO:
    """
    Independent PPO. Agents update concurrently using PPO clipping;
    critic (centralized or decentralized) supplies the advantage signal.
    """

    def __init__(
        self,
        env,
        policies: Sequence[GaussianPolicy],
        critic,
        cfg: IPPOConfig,
    ):
        self.env = env
        self.policies = list(policies)
        self.critic = critic
        self.cfg = cfg
        self.n_agents = env.n_agents

        # one Adam optimizer per agent (independence)
        self.optims = [
            torch.optim.Adam(p.parameters(), lr=cfg.policy_lr) for p in self.policies
        ]

        self.rng = np.random.default_rng(cfg.seed)

        # logging
        self.iteration = 0
        self.reward_history: List[float] = []
        self.mean_history: List[List[float]] = [[float(p.mean.item())] for p in self.policies]
        self.std_history: List[List[float]] = [[float(p.std.item())] for p in self.policies]

    # =========================================================================
    # one iteration
    # =========================================================================
    def step(self) -> dict:
        batch = self._collect_batch()
        critic_info = self.critic.update(batch)

        # snapshot old policies for IS ratio
        old_policies = [p.snapshot() for p in self.policies]
        old_logps = [old_policies[i].log_prob(batch["actions"][:, i]).detach()
                     for i in range(self.n_agents)]

        # per-agent advantage from critic
        with torch.no_grad():
            advantages = [
                self.critic.per_agent_advantage(batch["states"], batch["actions"], i)
                for i in range(self.n_agents)
            ]
            # standardize per-agent for stable PPO
            advantages = [
                (a - a.mean()) / (a.std() + 1e-8) for a in advantages
            ]

        # PPO update per agent
        for _ in range(self.cfg.n_policy_epochs):
            for i in range(self.n_agents):
                ai = batch["actions"][:, i]
                old_logp = old_logps[i]
                new_logp = self.policies[i].log_prob(ai)
                ratio = torch.exp(new_logp - old_logp)
                surr1 = ratio * advantages[i]
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps,
                                    1.0 + self.cfg.clip_eps) * advantages[i]
                loss = -torch.min(surr1, surr2).mean()
                self.optims[i].zero_grad()
                loss.backward()
                self.optims[i].step()

        # bookkeeping
        self.iteration += 1
        avg_reward = float(batch["rewards"].mean().item())
        self.reward_history.append(avg_reward)
        for i in range(self.n_agents):
            self.mean_history[i].append(float(self.policies[i].mean.item()))
            self.std_history[i].append(float(self.policies[i].std.item()))

        return {
            "iteration": self.iteration,
            "avg_reward": avg_reward,
            "critic": critic_info,
        }

    # =========================================================================
    # helpers
    # =========================================================================
    def _collect_batch(self) -> dict:
        B = self.cfg.batch_size
        with torch.no_grad():
            actions = torch.stack(
                [p.sample(B) for p in self.policies], dim=-1
            )
            actions = self.env.clamp_actions(actions)
            states = self.env.initial_observation(B)
            rewards = self.env.reward(actions)
        return {"states": states, "actions": actions, "rewards": rewards}

    # ----- convenience constructor -----
    @classmethod
    def from_config(cls, env, ippo_cfg: IPPOConfig, policy_cfg: PolicyConfig, critic):
        policies = [GaussianPolicy(policy_cfg) for _ in range(env.n_agents)]
        return cls(env=env, policies=policies, critic=critic, cfg=ippo_cfg)
