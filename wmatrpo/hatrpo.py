"""
HATRPO — Heterogeneous-Agent Trust Region Policy Optimisation
(Kuba et al., 2021, "Trust Region Policy Optimisation in Multi-Agent
Reinforcement Learning", arXiv:2109.11251).

HATRPO is the KL-trust-region sibling of HAPPO: the *same* sequential-update +
importance-sampling machinery (random agent order, cumulative IS weight, eq. 15
analogue), but each agent's update is a natural-gradient step subject to a hard
KL constraint  KL(π_i^old ‖ π_i^new) ≤ δ  — solved by the classic TRPO recipe:

    1.  policy gradient        g = ∇_θ E[ M_i · log π_i(a_i) ]
    2.  natural gradient       x = H⁻¹ g   via conjugate gradient, where H is the
                               Fisher / KL-Hessian (computed through Fisher-vector
                               products, so no explicit matrix is formed)
    3.  step                   Δθ = sqrt(2δ / (xᵀ H x)) · x
    4.  backtracking line search to enforce KL ≤ δ AND surrogate improvement.

This is the faithful counterpart to W-MATRPO's Algorithm 1: replace the
Wasserstein dual solve with a KL trust region and you get HATRPO. It provides
the "Standard HATRPO" and (with CAATR radii) "HATRPO with CAATR" rows of the
paper's main results table, on the same matched setup as every other baseline.

The policy is the shared 1-D Gaussian (`wmatrpo.policy.GaussianPolicy`, two
learnable scalars μ and log σ), so the Fisher is 2×2; the CG machinery is
nonetheless the general TRPO one, so the module transfers unchanged to
NN-parameterised policies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch

from wmatrpo.policy import GaussianPolicy, PolicyConfig
from wmatrpo.critic import CentralizedCritic


@dataclass
class HATRPOConfig:
    batch_size: int = 30
    n_agents: int = 2
    delta: float = 0.01              # per-agent KL trust-region radius
    damping: float = 0.1             # Fisher-vector-product damping (numerical)
    cg_iters: int = 10               # conjugate-gradient iterations
    backtrack_iters: int = 10        # line-search steps
    backtrack_coeff: float = 0.8     # geometric shrink per line-search step
    seed: int = 0
    # CAATR coupling: if provided, per-agent δ_i overrides `delta` each step.
    use_caatr: bool = False


def _flat(tensors) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])


def _gaussian_kl(old: GaussianPolicy, new: GaussianPolicy) -> torch.Tensor:
    """KL( N(μ_o,σ_o²) ‖ N(μ_n,σ_n²) ), differentiable in the NEW policy params."""
    mu_o, s_o = old.mean.detach(), old.std.detach()
    mu_n, s_n = new.mean, new.std
    return (torch.log(s_n / s_o)
            + (s_o ** 2 + (mu_o - mu_n) ** 2) / (2.0 * s_n ** 2)
            - 0.5)


class HATRPO:
    """Heterogeneous-Agent TRPO with sequential updates and IS correction."""

    def __init__(self, env, policies: Sequence[GaussianPolicy],
                 critic: CentralizedCritic, cfg: HATRPOConfig,
                 caatr=None):
        if not getattr(critic, "is_centralized", False):
            raise ValueError("HATRPO uses a centralized critic for the joint advantage.")
        self.env = env
        self.policies = list(policies)
        self.critic = critic
        self.cfg = cfg
        self.caatr = caatr
        self.n_agents = env.n_agents
        self.rng = np.random.default_rng(cfg.seed)

        self.iteration = 0
        self.reward_history: List[float] = []
        self.mean_history: List[List[float]] = [[float(p.mean.item())] for p in self.policies]
        self.std_history: List[List[float]] = [[float(p.std.item())] for p in self.policies]
        self._prev_w1 = [0.0 for _ in range(self.n_agents)]  # for CAATR drift

    # -------- TRPO inner machinery (per agent) --------
    def _surrogate(self, policy: GaussianPolicy, old_policy: GaussianPolicy,
                   a_i: torch.Tensor, M_i: torch.Tensor) -> torch.Tensor:
        """IS surrogate  E[ (π_new/π_old) · M_i ]  for one agent."""
        logp_new = policy.log_prob(a_i)
        logp_old = old_policy.log_prob(a_i).detach()
        ratio = torch.exp(logp_new - logp_old)
        return (ratio * M_i).mean()

    def _fvp(self, policy: GaussianPolicy, old_policy: GaussianPolicy,
             a_i: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """Fisher-vector product via the KL Hessian (Pearlmutter trick)."""
        params = [policy.mean_param, policy.log_std_param]
        kl = _gaussian_kl(old_policy, policy).mean()
        grads = torch.autograd.grad(kl, params, create_graph=True)
        flat_grad = _flat(grads)
        gv = (flat_grad * vec).sum()
        hvp = torch.autograd.grad(gv, params, retain_graph=True)
        return _flat(hvp) + self.cfg.damping * vec

    def _conjugate_gradient(self, fvp_fn, b: torch.Tensor) -> torch.Tensor:
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rr = torch.dot(r, r)
        for _ in range(self.cfg.cg_iters):
            Ap = fvp_fn(p)
            alpha = rr / (torch.dot(p, Ap) + 1e-10)
            x = x + alpha * p
            r = r - alpha * Ap
            rr_new = torch.dot(r, r)
            if rr_new < 1e-10:
                break
            p = r + (rr_new / rr) * p
            rr = rr_new
        return x

    def _update_agent(self, i: int, a_i: torch.Tensor, M_i: torch.Tensor,
                      delta: float) -> None:
        policy = self.policies[i]
        old_policy = policy.snapshot()
        params = [policy.mean_param, policy.log_std_param]

        # 1. policy gradient of the surrogate (ascent → maximize)
        surr = self._surrogate(policy, old_policy, a_i, M_i)
        g = _flat(torch.autograd.grad(surr, params, retain_graph=True)).detach()
        if torch.allclose(g, torch.zeros_like(g)):
            return

        # 2. natural gradient  x = H⁻¹ g  (CG with Fisher-vector products)
        fvp_fn = lambda v: self._fvp(policy, old_policy, a_i, v)
        x = self._conjugate_gradient(fvp_fn, g).detach()
        xHx = torch.dot(x, fvp_fn(x).detach())
        if xHx <= 0:
            return
        step_scale = torch.sqrt(2.0 * delta / (xHx + 1e-10))
        full_step = step_scale * x

        # 3. backtracking line search: accept the largest step with KL ≤ δ and
        #    surrogate improvement.
        theta0 = _flat([p.detach().clone() for p in params])
        old_surr = surr.detach()
        accepted = False
        for j in range(self.cfg.backtrack_iters):
            step = (self.cfg.backtrack_coeff ** j) * full_step
            with torch.no_grad():
                new_theta = theta0 + step
                policy.mean_param.copy_(new_theta[0])
                policy.log_std_param.copy_(new_theta[1])
            with torch.no_grad():
                kl = _gaussian_kl(old_policy, policy).mean().item()
                new_surr = self._surrogate(policy, old_policy, a_i, M_i).item()
            if kl <= delta and new_surr >= old_surr:
                accepted = True
                break
        if not accepted:  # revert
            with torch.no_grad():
                policy.mean_param.copy_(theta0[0])
                policy.log_std_param.copy_(theta0[1])

    # -------- CAATR radius (matches caatr.py's δ_i = C / (Σ_{j≠i} W1_j + ε)) --------
    def _caatr_delta(self, i: int) -> float:
        # CAATR produces a *W₁* trust radius (δ = C/(Σ_{j≠i} W1_j + ε)), which can
        # be large when teammate drift is small. For HATRPO the radius is a *KL*
        # bound, so we clamp CAATR's adaptive value to a KL-stable range centered
        # on the fixed δ. The adaptivity (larger radius when teammates are stable,
        # smaller when they move) is preserved; only the scale is bounded.
        if self.caatr is None:
            return self.cfg.delta
        raw = float(self.caatr.compute_deltas()[i])
        return float(np.clip(raw, 1e-4, 10.0 * self.cfg.delta))

    def step(self) -> dict:
        cfg = self.cfg
        batch = self._collect_batch()
        critic_info = self.critic.update(batch)

        old_policies = [p.snapshot() for p in self.policies]
        old_logps = [old_policies[k].log_prob(batch["actions"][:, k]).detach()
                     for k in range(self.n_agents)]

        with torch.no_grad():
            A = self.critic.advantage(batch["states"], batch["actions"])
            A = (A - A.mean()) / (A.std() + 1e-8)

        order = self.rng.permutation(self.n_agents).tolist()
        updated: List[int] = []
        for i in order:
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
                M_i = w * A
            a_i = batch["actions"][:, i]
            delta_i = self._caatr_delta(i) if cfg.use_caatr else cfg.delta
            self._update_agent(i, a_i, M_i, delta_i)
            updated.append(i)

        # record W1 drift for CAATR bookkeeping
        if self.caatr is not None:
            drifts = [float(self.policies[k].wasserstein_1(old_policies[k]).item())
                      for k in range(self.n_agents)]
            self.caatr.record_drifts(drifts)

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
    def from_config(cls, env, hatrpo_cfg: HATRPOConfig, policy_cfg: PolicyConfig,
                    critic, caatr=None):
        policies = [GaussianPolicy(policy_cfg) for _ in range(env.n_agents)]
        return cls(env=env, policies=policies, critic=critic, cfg=hatrpo_cfg, caatr=caatr)
