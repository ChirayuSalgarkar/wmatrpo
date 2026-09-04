"""
Regression test: the eq.-15 importance-sampling correction must actually
influence the update.

Background. An earlier version of `dual_solver.solve()` accepted a
pre-multiplied `advantages` entry, but computed the action gradient from the
raw critic *before* reading it and re-evaluated the critic inside the dual
objective, so the corrected advantage was assigned to a local variable and
never used. Varying it (including setting it to zero) produced a bit-identical
policy update and lambda*. This test pins the fixed behaviour.

Run:  python -m wmatrpo.tests.test_is_weights
"""
import torch
import numpy as np

from wmatrpo.dual_solver import DualSolver, DualSolverConfig
from wmatrpo.critic import CentralizedCritic, CriticConfig
from wmatrpo.policy import GaussianPolicy, PolicyConfig


def _setup(seed=0, N=3, B=30):
    torch.manual_seed(seed)
    np.random.seed(seed)
    critic = CentralizedCritic(CriticConfig(state_dim=1, n_agents=N))
    policy = GaussianPolicy(PolicyConfig())
    solver = DualSolver(DualSolverConfig())
    batch = {"states": torch.zeros(B, 1), "actions": torch.rand(B, N) * 7.0}
    return solver, batch, critic, policy, B


def test_is_weights_change_the_update():
    """Non-uniform IS weights must move the solution."""
    solver, batch, critic, policy, B = _setup()

    base = solver.solve(0, dict(batch), critic, policy, delta_i=0.04)

    torch.manual_seed(123)
    w = torch.exp(torch.randn(B) * 0.5)
    weighted = solver.solve(
        0, {**batch, "is_weights": w}, critic, policy, delta_i=0.04
    )

    moved = abs(weighted[0] - base[0]) + abs(weighted[2] - base[2])
    assert moved > 1e-6, (
        "IS weights did not influence the update: "
        f"base=(mean {base[0]:.8f}, lambda* {base[2]:.8f}) "
        f"weighted=(mean {weighted[0]:.8f}, lambda* {weighted[2]:.8f}). "
        "The eq.-15 correction is inert."
    )
    print(f"  [PASS] non-uniform IS weights move the update (delta={moved:.3e})")
    print(f"         base     mean={base[0]:.8f} lambda*={base[2]:.8f}")
    print(f"         weighted mean={weighted[0]:.8f} lambda*={weighted[2]:.8f}")


def test_uniform_weights_are_a_noop():
    """w == 1 must reproduce the unweighted path exactly (self-normalized)."""
    solver, batch, critic, policy, B = _setup()

    base = solver.solve(0, dict(batch), critic, policy, delta_i=0.04)
    ones = solver.solve(
        0, {**batch, "is_weights": torch.ones(B)}, critic, policy, delta_i=0.04
    )

    for name, x, y in zip(("mean", "std", "lambda*"), base[:3], ones[:3]):
        assert abs(x - y) < 1e-12, f"uniform weights changed {name}: {x} vs {y}"
    print("  [PASS] uniform weights (w=1) are an exact no-op")


def test_zero_weights_are_not_silently_ignored():
    """Degenerate all-zero weights must not reproduce the unweighted update."""
    solver, batch, critic, policy, B = _setup()

    base = solver.solve(0, dict(batch), critic, policy, delta_i=0.04)
    zeros = solver.solve(
        0, {**batch, "is_weights": torch.zeros(B)}, critic, policy, delta_i=0.04
    )

    same = (
        abs(base[0] - zeros[0]) < 1e-12
        and abs(base[2] - zeros[2]) < 1e-12
    )
    assert not same, (
        "all-zero IS weights produced an identical update — the correction "
        "is being discarded somewhere in the solve path."
    )
    print("  [PASS] all-zero weights change the update (not discarded)")


if __name__ == "__main__":
    print("=== Regression: eq.-15 IS correction reaches the update ===")
    test_is_weights_change_the_update()
    test_uniform_weights_are_a_noop()
    test_zero_weights_are_not_silently_ignored()
    print("\nAll IS-correction regression checks passed.")
