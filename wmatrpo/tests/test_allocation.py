"""
Sanity checks for the trust-region allocation strategies (paper Table 5).

The three non-CAATR budget rules ported from the original SAILRIT repo must
satisfy their defining budget-conservation / ranking properties:

  1. Fixed    : every agent gets exactly the full budget ε_total.
  2. Greedy   : allocations are non-negative, sum to ε_total, and the agent with
                the higher |advantage|/drift score gets the larger share.
  3. Weighted : water-filling gives non-negative allocations summing to ε_total,
                and only positive-advantage agents receive budget.

These are checked directly on the AllocationStrategy object (no training loop),
so they isolate the allocation math from the dual solve.

Run:
    python -m wmatrpo.tests.test_allocation
"""
from __future__ import annotations

import numpy as np

from wmatrpo.allocation import AllocationStrategy, AllocationConfig

EPS = 0.1


def test_fixed():
    print("=== Sanity check: Fixed allocation ===")
    a = AllocationStrategy(AllocationConfig(method="fixed", epsilon_total=EPS), n_agents=3)
    d = a.compute_deltas()
    ok = all(abs(x - EPS) < 1e-9 for x in d)
    print(f"[{'PASS' if ok else 'FAIL'}] all agents get ε_total={EPS}: {[round(x,3) for x in d]}")
    assert ok
    return ok


def test_greedy():
    print("\n=== Sanity check: Greedy allocation ===")
    a = AllocationStrategy(AllocationConfig(method="greedy", epsilon_total=EPS), n_agents=3)
    a.record_drifts([0.1, 0.1, 0.1])              # equal drifts → score ∝ |advantage|
    a.set_advantages([2.0, 1.0, 0.0])             # agent 0 has the largest |adv|
    d = np.array(a.compute_deltas())
    conserves = abs(d.sum() - EPS) < 1e-6
    nonneg = (d >= 0).all()
    ranked = d[0] > d[1] > d[2]                   # 0 gets most, 2 gets least
    ok = conserves and nonneg and ranked
    print(f"[{'PASS' if ok else 'FAIL'}] alloc={np.round(d,4).tolist()} "
          f"sum={d.sum():.4f} (=ε_total) ranked(0>1>2)={ranked}")
    assert ok
    return ok


def test_weighted():
    print("\n=== Sanity check: Weighted (water-filling) allocation ===")
    a = AllocationStrategy(AllocationConfig(method="weighted", epsilon_total=EPS), n_agents=3)
    a.set_advantages([2.0, 1.0, -1.0])            # agent 2 has negative advantage
    d = np.array(a.compute_deltas())
    conserves = abs(d.sum() - EPS) < 1e-4
    nonneg = (d >= 0).all()
    neg_gets_zero = d[2] < 1e-6                   # negative-advantage agent gets none
    ok = conserves and nonneg and neg_gets_zero
    print(f"[{'PASS' if ok else 'FAIL'}] alloc={np.round(d,4).tolist()} "
          f"sum={d.sum():.4f} (=ε_total) neg_adv_agent_zero={neg_gets_zero}")
    assert ok
    return ok


def test_fallback():
    print("\n=== Sanity check: greedy/weighted fall back to uniform w/o advantages ===")
    for m in ("greedy", "weighted"):
        a = AllocationStrategy(AllocationConfig(method=m, epsilon_total=EPS), n_agents=4)
        d = np.array(a.compute_deltas())          # no set_advantages() called
        ok = np.allclose(d, EPS / 4)
        print(f"[{'PASS' if ok else 'FAIL'}] {m}: uniform ε/N fallback = {np.round(d,4).tolist()}")
        assert ok
    return True


if __name__ == "__main__":
    r = [test_fixed(), test_greedy(), test_weighted(), test_fallback()]
    print("\nAll allocation checks passed." if all(r) else "\nSOME CHECKS FAILED.")
