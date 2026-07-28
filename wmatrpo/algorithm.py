"""
Algorithm 1: W-MATRPO with CAATR.

Implements the paper's main loop faithfully:

    Algorithm 1 W-MATRPO with CAATR
    Input: Initial policy π^(0), parameters C, ε_base, ε_max
     1: Initialize policy drift history
     2: for t = 0, 1, 2, ... do
     3:    Collect trajectories and compute joint advantage A^π^(t)
     4:    Compute trust-regions {δ_i^(t+1)} using Algorithm 2
     5:    Randomly order agents σ
     6:    Initialize U_k = ∅ (set of updated agents)
     7:    for k = 1 to N do
     8:       i ← σ(k)
     9:       Compute M_i(s, a) using U_k                        (eq. 15)
    10:       Solve dual problem for agent i: λ_i* ← argmin g(λ_i)
    11:       Update policy π_i^(t+1) with λ_i*
    12:       Apply importance sampling correction
    13:       U_k ← U_k ∪ {i}
    14:    end for
    15:    Update centralized critic parameters φ
    16:    Store policy drift Drift_j^(t) for all j
    17: end for

This file implements the for-loop body (steps 3–16) as `WMATRPO.step()`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
import torch

from wmatrpo.policy import GaussianPolicy, PolicyConfig
from wmatrpo.critic import CentralizedCritic
from wmatrpo.dual_solver import DualSolver
from wmatrpo.caatr import CAATR


@dataclass
class WMATRPOConfig:
    batch_size: int = 30
    n_agents: int = 2
    seed: int = 0


class WMATRPO:
    """
    Container that bundles policies, critic, CAATR, dual solver, and env,
    and exposes `step()` for one iteration of Algorithm 1.
    """

    def __init__(
        self,
        env,
        policies: Sequence[GaussianPolicy],
        critic: CentralizedCritic,
        dual_solver: DualSolver,
        caatr: CAATR,
        cfg: WMATRPOConfig,
    ):
        self.env = env
        self.policies = list(policies)
        self.critic = critic
        self.dual_solver = dual_solver
        self.caatr = caatr
        self.cfg = cfg

        self.n_agents = env.n_agents
        if len(self.policies) != self.n_agents:
            raise ValueError(
                f"expected {self.n_agents} policies, got {len(self.policies)}"
            )

        # Reproducibility
        self.rng = np.random.default_rng(cfg.seed)

        # History (for analysis / logging)
        self.iteration = 0
        self.reward_history: List[float] = []
        self.mean_history: List[List[float]] = [[float(p.mean.item())] for p in self.policies]
        self.std_history: List[List[float]] = [[float(p.std.item())] for p in self.policies]
        self.delta_history: List[List[float]] = [[] for _ in range(self.n_agents)]
        self.lambda_history: List[List[float]] = [[] for _ in range(self.n_agents)]

    # =========================================================================
    # One iteration of Algorithm 1
    # =========================================================================
    def step(self) -> dict:
        # --- Step 3: collect batch ---
        batch = self._collect_batch()

        # --- Step 15 (moved earlier to ensure fresh advantages): update critic ---
        critic_info = self.critic.update(batch)

        # --- Step 4: compute trust regions via CAATR ---
        deltas = self.caatr.compute_deltas()  # length N

        # --- Step 5: random agent ordering ---
        order = self.rng.permutation(self.n_agents).tolist()

        # --- Snapshot old policies (for IS, drift, and dual solver) ---
        old_policies = [p.snapshot() for p in self.policies]

        # --- Steps 7-14: sequential per-agent updates with IS correction ---
        updated_agents: List[int] = []
        info_per_agent: List[dict] = [{} for _ in range(self.n_agents)]
        for k_idx, agent_i in enumerate(order):
            # eq. 15: importance-corrected advantages over previously updated agents
            adv = self._is_corrected_advantage(batch, updated_agents, old_policies)
            solver_batch = {**batch, "advantages": adv}

            new_mean, new_std, lambda_star, info = self.dual_solver.solve(
                agent_id=agent_i,
                batch=solver_batch,
                critic=self.critic,
                policy_old=old_policies[agent_i],
                delta_i=float(deltas[agent_i]),
            )
            # Step 11: update policy — with an explicit backtracking cap on the
            # actual W₁ drift, in case the dual solve + Gaussian projection
            # exceeds δ. This is the same safeguard the original repo applied.
            new_mean, new_std = self._enforce_trust_region(
                old_policies[agent_i],
                new_mean,
                new_std,
                delta_i=float(deltas[agent_i]),
            )
            self.policies[agent_i].set_params(new_mean, new_std)
            updated_agents.append(agent_i)
            info_per_agent[agent_i] = info
            self.lambda_history[agent_i].append(lambda_star)

        # --- Step 16: compute and store policy drifts (W₁) ---
        drifts = []
        for i in range(self.n_agents):
            d = float(old_policies[i].wasserstein_1(self.policies[i]).detach().item())
            drifts.append(d)
        self.caatr.record_drifts(drifts)

        # --- Bookkeeping ---
        self.iteration += 1
        avg_reward = float(batch["rewards"].mean().item())
        self.reward_history.append(avg_reward)
        for i in range(self.n_agents):
            self.mean_history[i].append(float(self.policies[i].mean.item()))
            self.std_history[i].append(float(self.policies[i].std.item()))
            self.delta_history[i].append(float(deltas[i]))

        return {
            "iteration": self.iteration,
            "avg_reward": avg_reward,
            "deltas": [float(d) for d in deltas],
            "lambdas": [info_per_agent[i].get("lambda_star", 0.0) for i in range(self.n_agents)],
            "drifts": drifts,
            "critic": critic_info,
            "agent_order": order,
        }

    # =========================================================================
    # Helpers
    # =========================================================================
    def _collect_batch(self) -> dict:
        """Sample joint actions from current policies, compute rewards."""
        B = self.cfg.batch_size
        with torch.no_grad():
            actions = torch.stack(
                [p.sample(B) for p in self.policies], dim=-1
            )  # (B, N)
            actions = self.env.clamp_actions(actions)
            states = self.env.initial_observation(B)
            rewards = self.env.reward(actions)
        return {"states": states, "actions": actions, "rewards": rewards}

    def _enforce_trust_region(
        self,
        policy_old: GaussianPolicy,
        new_mean: float,
        new_std: float,
        delta_i: float,
    ) -> tuple:
        """
        Hard-cap the candidate (new_mean, new_std) so that the W₁ distance to
        policy_old does not exceed delta_i.

        For 1D Gaussians, W₁ along the parametric line
            (μ(α), σ(α)) = (μ_old + α Δμ, σ_old + α Δσ)
        is exactly linear in α (the closed form
            W₁ = 2|Δσ|·φ(Δμ/|Δσ|) + Δμ·(2Φ(Δμ/|Δσ|) − 1)
        is positively homogeneous in (Δμ, Δσ)). So a single linear backtrack
        α* = δ_i / W₁(candidate)  yields the unique policy on the segment
        with exact W₁ = δ_i.

        Same safeguard the original repo applied; we just make it exact instead
        of an O(δ/W₁) approximation.
        """
        cand = GaussianPolicy(policy_old.cfg)
        cand.set_params(new_mean, new_std)
        w1 = float(policy_old.wasserstein_1(cand).detach().item())
        if w1 <= delta_i:
            return new_mean, new_std

        mu_old = float(policy_old.mean.detach().item())
        sig_old = float(policy_old.std.detach().item())
        alpha = float(delta_i) / (w1 + 1e-12)
        return (
            mu_old + alpha * (new_mean - mu_old),
            sig_old + alpha * (new_std - sig_old),
        )

    def _is_corrected_advantage(
        self,
        batch: dict,
        updated_agents: Sequence[int],
        old_policies: Sequence[GaussianPolicy],
    ) -> torch.Tensor:
        """
        Eq. 15:  M_i(s, a) = A^π_old(s, a) · ∏_{k ∈ U_k} π_k^new(a_k|s) / π_k^old(a_k|s)
        """
        with torch.no_grad():
            A = self.critic.advantage(batch["states"], batch["actions"])
            if not updated_agents:
                return A
            log_ratio = torch.zeros_like(A)
            for k in updated_agents:
                ak = batch["actions"][:, k]
                log_ratio = log_ratio + self.policies[k].log_prob(ak) \
                                       - old_policies[k].log_prob(ak)
            # numerical-stable importance ratio
            log_ratio = torch.clamp(log_ratio, min=-10.0, max=10.0)
            ratio = torch.exp(log_ratio)
            return A * ratio

    # =========================================================================
    # Convenience constructors
    # =========================================================================
    @classmethod
    def from_config(cls, env, alg_cfg: WMATRPOConfig,
                    policy_cfg: PolicyConfig,
                    critic, dual_solver, caatr):
        policies = [GaussianPolicy(policy_cfg) for _ in range(env.n_agents)]
        return cls(env=env, policies=policies, critic=critic,
                   dual_solver=dual_solver, caatr=caatr, cfg=alg_cfg)
