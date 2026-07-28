"""
MAPPO — Multi-Agent PPO (Yu et al., 2021, "The Surprising Effectiveness of PPO
in Cooperative Multi-Agent Games").

MAPPO = independent per-agent actors updated with the PPO clipped surrogate,
sharing a *centralized* value function that conditions on global information.
The distinction from IPPO is the critic: MAPPO uses a centralized value/critic;
IPPO uses per-agent local critics.

In this package the difference is made explicit by requiring a CentralizedCritic.
We also include the two MAPPO "tricks" that matter here:
  - advantage normalization (per update batch), and
  - PPO value clipping is not needed in the stateless single-step setting
    (γ = 0), so the critic is fit directly by CentralizedCritic.update().

For homogeneous teams MAPPO often shares actor parameters across agents; the
differential game is heterogeneous (asymmetric σ), so we keep independent actors,
which is the standard MAPPO choice for heterogeneous settings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch

from wmatrpo.policy import GaussianPolicy, PolicyConfig
from wmatrpo.critic import CentralizedCritic


@dataclass
class MAPPOConfig:
    batch_size: int = 30
    n_agents: int = 2
    clip_eps: float = 0.2
    n_policy_epochs: int = 4
    policy_lr: float = 3e-4
    seed: int = 0


class MAPPO:
    """Multi-Agent PPO with a centralized critic."""

    def __init__(self, env, policies: Sequence[GaussianPolicy],
                 critic: CentralizedCritic, cfg: MAPPOConfig):
        if not getattr(critic, "is_centralized", False):
            raise ValueError("MAPPO requires a CentralizedCritic "
                             "(a centralized value function is the defining feature).")
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

        old_policies = [p.snapshot() for p in self.policies]
        old_logps = [old_policies[i].log_prob(batch["actions"][:, i]).detach()
                     for i in range(self.n_agents)]

        # centralized advantage: same A(s, joint_a) signal for every agent
        with torch.no_grad():
            A = self.critic.advantage(batch["states"], batch["actions"])  # (B,)
            A = (A - A.mean()) / (A.std() + 1e-8)                          # normalize

        for _ in range(self.cfg.n_policy_epochs):
            for i in range(self.n_agents):
                ai = batch["actions"][:, i]
                new_logp = self.policies[i].log_prob(ai)
                ratio = torch.exp(new_logp - old_logps[i])
                surr1 = ratio * A
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps,
                                    1 + self.cfg.clip_eps) * A
                loss = -torch.min(surr1, surr2).mean()
                self.optims[i].zero_grad()
                loss.backward()
                self.optims[i].step()

        self.iteration += 1
        avg_reward = float(batch["rewards"].mean().item())
        self.reward_history.append(avg_reward)
        for i in range(self.n_agents):
            self.mean_history[i].append(float(self.policies[i].mean.item()))
            self.std_history[i].append(float(self.policies[i].std.item()))

        return {"iteration": self.iteration, "avg_reward": avg_reward, "critic": critic_info}

    def _collect_batch(self) -> dict:
        B = self.cfg.batch_size
        with torch.no_grad():
            actions = torch.stack([p.sample(B) for p in self.policies], dim=-1)
            actions = self.env.clamp_actions(actions)
            states = self.env.initial_observation(B)
            rewards = self.env.reward(actions)
        return {"states": states, "actions": actions, "rewards": rewards}

    @classmethod
    def from_config(cls, env, mappo_cfg: MAPPOConfig, policy_cfg: PolicyConfig, critic):
        policies = [GaussianPolicy(policy_cfg) for _ in range(env.n_agents)]
        return cls(env=env, policies=policies, critic=critic, cfg=mappo_cfg)
