"""
Sanity checks for the faithful HATRPO implementation.

HATRPO (Kuba 2021) applies, per agent, a natural-gradient step under a hard KL
trust region  KL(π_old ‖ π_new) ≤ δ, solved by conjugate gradient +
backtracking line search. Two properties must hold:

  1. KL constraint respected: no per-agent update exceeds the radius δ (up to a
     small line-search tolerance). This is the defining guarantee of a trust
     region method — if it fails, the CG/line-search machinery is broken.

  2. Correct qualitative behavior on the differential game: with the KL trust
     region, HATRPO stays trapped in the LOCAL optimum (all-ones), reproducing
     the paper's "Standard HATRPO" row — distance ≈ 4√N to the global optimum.
     (This is the contrast the paper draws: the Wasserstein constraint escapes,
     the KL constraint does not.)

Run:
    python -m wmatrpo.tests.test_hatrpo
"""
from __future__ import annotations

import numpy as np
import torch

from wmatrpo.env import DifferentialGameEnv, DifferentialGameConfig
from wmatrpo.policy import PolicyConfig
from wmatrpo.critic import CentralizedCritic, CriticConfig
from wmatrpo.hatrpo import HATRPO, HATRPOConfig, _gaussian_kl


def _build(n_agents, delta, seed=0, use_caatr=False, caatr=None):
    torch.manual_seed(seed); np.random.seed(seed)
    variances = [1.0, 3.0] + [1.0] * (n_agents - 2)
    env = DifferentialGameEnv(DifferentialGameConfig(
        n_agents=n_agents, variances=variances[:n_agents]))
    pcfg = PolicyConfig(init_mean=1.5, init_std=0.5, std_min=0.1, std_max=1.0,
                        action_low=env.action_low, action_high=env.action_high)
    critic = CentralizedCritic(CriticConfig(
        state_dim=env.state_dim, n_agents=n_agents, hidden=64, n_layers=2,
        lr=3e-3, n_update_epochs=8))
    return HATRPO.from_config(
        env, HATRPOConfig(n_agents=n_agents, delta=delta, seed=seed,
                          use_caatr=use_caatr), pcfg, critic, caatr=caatr)


def test_kl_constraint():
    print("=== Sanity check: HATRPO respects the KL trust region ===")
    delta = 0.01
    alg = _build(n_agents=3, delta=delta, seed=0)
    tol = 1.5 * delta         # small slack for line-search granularity
    max_kl = 0.0
    n_viol = 0
    for _ in range(100):
        old = [p.snapshot() for p in alg.policies]
        alg.step()
        for k in range(3):
            kl = float(_gaussian_kl(old[k], alg.policies[k]).mean())
            max_kl = max(max_kl, kl)
            if kl > tol:
                n_viol += 1
    ok = n_viol == 0
    print(f"[{'PASS' if ok else 'FAIL'}] max realized KL = {max_kl:.5f} "
          f"(δ={delta}); violations>{tol:.3f} = {n_viol}")
    assert ok, f"KL constraint violated {n_viol} times (max KL {max_kl:.5f})"
    return ok


def test_local_optimum_trap():
    print("\n=== Sanity check: HATRPO converges to the LOCAL optimum (Table 3) ===")
    for n in (3, 5):
        alg = _build(n_agents=n, delta=0.01, seed=0)
        for _ in range(200):
            alg.step()
        means = np.array([float(p.mean.item()) for p in alg.policies])
        dist_local = float(np.linalg.norm(means - 1.0))
        dist_global = float(np.linalg.norm(means - 5.0))
        expect_local = 0.0
        near_local = dist_local < 0.6           # sits at the all-ones local opt
        print(f"[{'PASS' if near_local else 'FAIL'}] N={n}: dist_to_local={dist_local:.3f} "
              f"dist_to_global={dist_global:.3f} (4√N={4*np.sqrt(n):.3f})")
        assert near_local, f"N={n}: HATRPO not at local opt (dist_local {dist_local:.3f})"
    return True


if __name__ == "__main__":
    ok1 = test_kl_constraint()
    ok2 = test_local_optimum_trap()
    print("\nAll HATRPO checks passed." if (ok1 and ok2) else "\nSOME CHECKS FAILED.")
