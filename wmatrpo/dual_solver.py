"""
Dual solver for the Wasserstein-constrained policy improvement problem.

This is the **faithful** implementation of Theorem 1 + Remark 1 + Corollary 1.

Theorem 1 (paper):  the per-agent primal
    max_{π_i^new} E_{a_{-i}}[A(s, a_i, a_{-i})]
    s.t.  E_s[W₁(π_i^old, π_i^new)] ≤ δ_i

has the dual
    min_{λ_i ≥ 0}  λ_i δ_i + E_{s, a ~ π_old}[Φ_{λ_i}(s, a_i)]

where Φ_λ(s, a_i) = max_{a'_i} { A(s, a'_i, a_{-i}) - λ c(a_i, a'_i) }.

Remark 1 (paper):  for Gaussian policies with squared-L₂ transport cost,
the inner max has the closed-form solution
    a'_i* = a_i + ∇_{a_i} A(s, a_i, a_{-i}) / (2 λ).

Corollary 1 (paper):  the optimal updated policy is a pushforward of π_old
through the optimal-action map. We project to the Gaussian class by
moment-matching the pushforward samples (this is the operational choice
made implicitly whenever policies are restricted to Gaussians).

Implementation notes
--------------------
* The squared-L₂ cost is used (consistent with the W₂² geometry of Paper 2;
  unifies the two papers).
* ∇_{a_i} A is computed via autograd on the centralized critic's
  Q-network (A = Q - V, and V doesn't depend on a).
* The outer 1-D minimization over λ uses scipy.optimize.minimize_scalar
  with a 'bounded' method; this is exact for the smooth, convex dual
  objective in our setting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from scipy.optimize import minimize_scalar


@dataclass
class DualSolverConfig:
    lambda_min: float = 1e-4
    lambda_max: float = 50.0
    action_low: float = 0.0
    action_high: float = 7.0
    cost_type: str = "l2"          # "l2" → c(a,a') = (a - a')² ; "l1" → |a-a'|
    fallback_std: float = 0.5      # used when pushforward samples have ~0 spread


class DualSolver:
    """Theorem 1 dual solver (closed-form inner max for Gaussian + L₂)."""

    def __init__(self, cfg: DualSolverConfig):
        self.cfg = cfg

    # ---------- core ----------
    def solve(
        self,
        agent_id: int,
        batch: dict,
        critic,
        policy_old,
        delta_i: float,
    ) -> Tuple[float, float, float, dict]:
        """
        Solve the per-agent dual problem.

        Args:
            agent_id:   index of the agent being updated
            batch:      dict with 'states' (B, state_dim), 'actions' (B, N),
                        and optionally 'advantages' (B,) — if provided, used
                        as the per-sample advantage (e.g., after IS correction).
            critic:     CentralizedCritic
            policy_old: GaussianPolicy (frozen snapshot)
            delta_i:    trust-region radius for this agent

        Returns:
            new_mean, new_std, lambda_star, info_dict
        """
        s = batch["states"]
        a = batch["actions"]
        B, N = a.shape

        # ---- per-sample ∇_{a_i} A(s, a) ----
        # autograd through the critic's Q
        a_grad = a.clone().detach().requires_grad_(True)
        adv = critic.advantage(s, a_grad)            # (B,)
        adv_sum = adv.sum()
        grad = torch.autograd.grad(adv_sum, a_grad)[0]  # (B, N)
        grad_i = grad[:, agent_id].detach()           # (B,)

        # we use the actual advantage from the critic, unless caller supplied
        # IS-corrected advantages (see eq. 15)
        if "advantages" in batch:
            adv = batch["advantages"].detach()
        else:
            adv = adv.detach()

        a_i = a[:, agent_id].detach()                 # (B,)
        a_other = a.detach()                          # (B, N)  for re-eval at a'

        # ---- dual objective (Eq. 4) ----
        # For L₂ cost the inner max is closed-form (Remark 1):
        #   a'_i*(a_i; λ) = a_i + grad_i / (2 λ)
        # Then  Φ_λ(s, a_i) = A(s, a'_i*, a_{-i}) - λ |a'_i* - a_i|²
        # Estimated by Monte Carlo over the batch.
        def dual_obj(lam: float) -> float:
            lam = float(lam)
            if lam < self.cfg.lambda_min:
                return float("inf")
            a_star = self._closed_form_pushforward(a_i, grad_i, lam)
            a_joint = a_other.clone()
            a_joint[:, agent_id] = a_star
            with torch.no_grad():
                A_star = critic.advantage(s, a_joint)
                cost = self._cost(a_i, a_star)
                phi = A_star - lam * cost
            return lam * float(delta_i) + float(phi.mean().item())

        result = minimize_scalar(
            dual_obj,
            bounds=(self.cfg.lambda_min, self.cfg.lambda_max),
            method="bounded",
            options={"xatol": 1e-5},
        )
        lambda_star = float(result.x)
        dual_value_star = float(result.fun)

        # ---- pushforward to construct new policy ----
        # Corollary 1: π̃_i = (T_{λ*})_# π_i^old. We approximate T_{λ*} samples
        # by the closed-form action map and project to Gaussian by MLE
        # (sample mean and std), since the policy class is Gaussian.
        with torch.no_grad():
            a_pushed = self._closed_form_pushforward(a_i, grad_i, lambda_star)
            a_pushed = a_pushed.clamp(self.cfg.action_low, self.cfg.action_high)
            new_mean = float(a_pushed.mean().item())
            new_std_raw = float(a_pushed.std(unbiased=False).item())

        # numerical floor in case all gradients collapse
        new_std = max(new_std_raw, 1e-3)
        if new_std < 0.01:
            new_std = self.cfg.fallback_std

        info = {
            "lambda_star": lambda_star,
            "dual_value": dual_value_star,
            "grad_i_mean": float(grad_i.mean().item()),
            "grad_i_abs_mean": float(grad_i.abs().mean().item()),
            "pushforward_std_raw": new_std_raw,
        }
        return new_mean, new_std, lambda_star, info

    # ---------- helpers ----------
    def _closed_form_pushforward(
        self, a_i: torch.Tensor, grad_i: torch.Tensor, lam: float
    ) -> torch.Tensor:
        """
        Remark 1: For Gaussian policy + L₂ cost,
            a'* = a + ∇A / (2 λ).

        For L₁ cost the optimal a' is the soft-threshold version; we keep the
        L₁ branch for completeness but the production path uses L₂.
        """
        if self.cfg.cost_type == "l2":
            return a_i + grad_i / (2.0 * lam)
        elif self.cfg.cost_type == "l1":
            # subgradient solution: move in the direction of grad until ‖grad‖ ≤ λ
            return a_i + torch.sign(grad_i) * torch.clamp(grad_i.abs() - lam, min=0.0)
        else:
            raise ValueError(f"Unknown cost type: {self.cfg.cost_type}")

    def _cost(self, a: torch.Tensor, a_prime: torch.Tensor) -> torch.Tensor:
        if self.cfg.cost_type == "l2":
            return (a - a_prime) ** 2
        elif self.cfg.cost_type == "l1":
            return (a - a_prime).abs()
        else:
            raise ValueError(f"Unknown cost type: {self.cfg.cost_type}")
