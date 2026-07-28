"""
Sanity check for the dual solver against a closed-form problem.

We construct a synthetic 1-agent setup where the advantage function is exactly
A(a) = α (a - a*)  + β (i.e., locally linear in a). For a quadratic / linear
advantage and an L₂ trust region with radius δ, the optimal step toward the
gradient direction has the known closed form:

    new_mean = old_mean + sign(grad) · min(|grad|/(2λ*), √δ)

Specifically, for a constant gradient g (advantage of the form A(a) = g·a),
the dual problem reduces to:
    min_λ  λ δ + max_{a'} { g·a' - λ (a' - a)² }
The inner max gives a' = a + g/(2λ), and the dual objective becomes
    λ δ - g·a + g²/(4λ).
Differentiating w.r.t. λ: δ + (-g²/(4 λ²)) = 0 → λ* = |g|/(2√δ),
which gives a' - a = sign(g) · √δ. So the policy mean moves by exactly √δ
in the gradient direction.

This test mocks a critic that returns A(s, a) = g · a_agent (linear in agent 0's
action) and verifies that one dual-solver step moves the policy mean by √δ.

Run:
    python -m wmatrpo.tests.test_dual_solver
"""
from __future__ import annotations

from math import sqrt

import torch

from wmatrpo.policy import GaussianPolicy, PolicyConfig
from wmatrpo.dual_solver import DualSolver, DualSolverConfig


class LinearCritic:
    """A(s, a) = g · a[:, 0], V(s) = 0. Q(s, a) = g · a[:, 0]."""

    def __init__(self, g: float):
        self.g = float(g)

    def advantage(self, s, a):
        # use float64 for the test
        return self.g * a[..., 0].to(s.dtype)

    def value(self, s):
        return torch.zeros(s.shape[0], dtype=s.dtype, device=s.device)

    def q_value(self, s, a):
        return self.advantage(s, a)


def _run_case(g: float, delta: float, initial_mean: float = 1.5,
              initial_std: float = 0.5, B: int = 512):
    policy_cfg = PolicyConfig(init_mean=initial_mean, init_std=initial_std,
                              std_min=1e-3, std_max=10.0,
                              action_low=-1e6, action_high=1e6)
    policy = GaussianPolicy(policy_cfg)
    policy_old = policy.snapshot()

    ds = DualSolver(DualSolverConfig(
        lambda_min=1e-5, lambda_max=200.0,
        action_low=-1e6, action_high=1e6, cost_type="l2"))

    torch.manual_seed(0)
    a = torch.zeros((B, 1))
    a[:, 0] = policy.sample(B)
    batch = {
        "states": torch.ones((B, 1)),
        "actions": a,
        "advantages": LinearCritic(g).advantage(torch.ones(B, 1), a).detach(),
    }
    new_mean, new_std, lam_star, info = ds.solve(
        agent_id=0, batch=batch, critic=LinearCritic(g),
        policy_old=policy_old, delta_i=delta,
    )
    return new_mean, new_std, lam_star, info


def _check(case, g, delta, initial_mean=1.5, tol=0.08):
    new_mean, new_std, lam_star, info = _run_case(g, delta, initial_mean=initial_mean)
    expected_move = (1 if g > 0 else -1) * sqrt(delta)
    actual_move = new_mean - initial_mean
    err = abs(actual_move - expected_move)
    status = "PASS" if err < tol else "FAIL"
    print(f"  [{status}] g={g:+.2f}, δ={delta:.3f}  → moved {actual_move:+.4f} "
          f"(expected {expected_move:+.4f}, λ*={lam_star:.3f})")
    return status == "PASS"


def main():
    print("=== Sanity check: dual solver on linear advantage ===")
    print("Theory: for A(a)=g·a with L₂ cost and trust radius δ, ")
    print("optimal mean shift = sign(g)·√δ.\n")
    results = [
        _check("positive gradient, small δ",   g=+1.0, delta=0.04),
        _check("positive gradient, medium δ",  g=+1.0, delta=0.16),
        _check("negative gradient, small δ",   g=-1.0, delta=0.04),
        _check("positive gradient, large g",   g=+5.0, delta=0.04),
        _check("positive gradient, tiny g",    g=+0.1, delta=0.04),
    ]
    if all(results):
        print("\nDual solver passes linear-advantage check.")
    else:
        print("\nDual solver mismatch — investigate.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
