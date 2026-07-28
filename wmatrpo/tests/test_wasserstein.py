"""
Sanity check for the W₁ between Gaussians.

The original paper's eq. (31) used the formula |Δμ| + |Δσ|, which is wrong
in general (see `paper1_audit_w1_gaussian.md`). This test verifies our
`GaussianPolicy.wasserstein_1` uses the corrected Vallender integral form
and matches the analytical limits:

  - σ_1 = σ_2:    W₁ = |μ_1 - μ_2|
  - μ_1 = μ_2:    W₁ = √(2/π) · |σ_1 - σ_2|   (≈ 0.7979 · |Δσ|)

Run:
    python -m wmatrpo.tests.test_wasserstein
"""
from __future__ import annotations

from math import pi, sqrt

import torch

from wmatrpo.policy import GaussianPolicy, PolicyConfig


def _p(mean, std):
    cfg = PolicyConfig(init_mean=mean, init_std=std,
                       std_min=1e-6, std_max=100.0,
                       action_low=-1e6, action_high=1e6)
    return GaussianPolicy(cfg)


def _check(case, mu1, s1, mu2, s2, expected, tol=1e-3):
    a = _p(mu1, s1)
    b = _p(mu2, s2)
    w1 = float(a.wasserstein_1(b).item())
    err = abs(w1 - expected)
    status = "PASS" if err < tol else "FAIL"
    print(f"  [{status}] {case:40s}  W₁={w1:.5f}   expected={expected:.5f}   err={err:.2e}")
    return status == "PASS"


def main():
    print("=== Sanity check: W₁ between 1D Gaussians ===\n")
    sqrt_2_over_pi = sqrt(2.0 / pi)
    results = [
        _check("identical",                   0.0, 1.0, 0.0, 1.0, 0.0),
        _check("same σ, μ shift = 1",         0.0, 1.0, 1.0, 1.0, 1.0),
        _check("same σ, μ shift = 3.5",       1.5, 0.5, 5.0, 0.5, 3.5),
        _check("same μ, σ 1→2",               0.0, 1.0, 0.0, 2.0, sqrt_2_over_pi * 1.0),
        _check("same μ, σ 1→3",               0.0, 1.0, 0.0, 3.0, sqrt_2_over_pi * 2.0),
        _check("paper init μ, σ 0.5→0.6",     1.5, 0.5, 1.5, 0.6, sqrt_2_over_pi * 0.1),
    ]
    if all(results):
        print("\nAll W₁ checks passed.")
    else:
        print("\nSome W₁ checks failed — investigate.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
