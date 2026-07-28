"""
Per-agent Gaussian policy π_i(a_i | s) = N(μ_i, σ_i²).

For the differential-game task, the policy is stateless: μ_i and log σ_i are
learnable scalars. The class is structured so that extending to NN-parameterized
(state-dependent) policies is a matter of swapping the head.

Includes closed-form distances used by the algorithm:

    W₂²(N(μ_1, σ_1²), N(μ_2, σ_2²)) = (μ_1 - μ_2)² + (σ_1 - σ_2)²
    W₁(N(μ_1, σ_1²), N(μ_2, σ_2²))   = ∫₀¹ |(μ_1 - μ_2) + (σ_1 - σ_2) Φ⁻¹(t)| dt

The W₁ form is what the paper SHOULD have used (eq. 31 in the paper was buggy;
see `paper1_audit_w1_gaussian.md` in the parent folder).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt, pi, log
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


_LOG2PI = log(2.0 * pi)


@dataclass
class PolicyConfig:
    init_mean: float = 1.5
    init_std: float = 0.5
    std_min: float = 0.05
    std_max: float = 3.0
    action_low: float = 0.0
    action_high: float = 7.0


class GaussianPolicy(nn.Module):
    """Stateless 1D Gaussian policy for a single agent."""

    def __init__(self, cfg: Optional[PolicyConfig] = None):
        super().__init__()
        self.cfg = cfg or PolicyConfig()
        self.mean_param = nn.Parameter(torch.tensor(float(self.cfg.init_mean)))
        # Learnable log-std (unconstrained); std is exp + clamp.
        self.log_std_param = nn.Parameter(torch.tensor(float(np.log(self.cfg.init_std))))

    # -------- parameters --------
    @property
    def mean(self) -> torch.Tensor:
        return self.mean_param

    @property
    def std(self) -> torch.Tensor:
        return torch.clamp(torch.exp(self.log_std_param), self.cfg.std_min, self.cfg.std_max)

    def set_params(self, mean: float, std: float) -> None:
        """Hard-set μ and σ (used after dual-solver returns its solution)."""
        with torch.no_grad():
            mean_clamped = max(self.cfg.action_low + 1e-6,
                               min(self.cfg.action_high - 1e-6, float(mean)))
            std_clamped = max(self.cfg.std_min, min(self.cfg.std_max, float(std)))
            self.mean_param.copy_(torch.tensor(mean_clamped))
            self.log_std_param.copy_(torch.tensor(np.log(std_clamped)))

    # -------- sampling --------
    def sample(self, n: int) -> torch.Tensor:
        """Reparameterized samples: shape (n,)."""
        eps = torch.randn(n)
        a = self.mean + self.std * eps
        return a.clamp(self.cfg.action_low, self.cfg.action_high)

    def log_prob(self, a: torch.Tensor) -> torch.Tensor:
        """log π(a) for the unclamped Gaussian (clip-aware log-prob is approx)."""
        var = self.std ** 2
        return -0.5 * ((a - self.mean) ** 2 / var + _LOG2PI) - torch.log(self.std)

    # -------- distances --------
    def w2_squared(self, other: "GaussianPolicy") -> torch.Tensor:
        """W₂² in closed form for 1D Gaussians."""
        return (self.mean - other.mean) ** 2 + (self.std - other.std) ** 2

    def wasserstein_1(self, other: "GaussianPolicy") -> torch.Tensor:
        """
        W₁ between two 1D Gaussians in closed form.

        Starting from the Vallender / inverse-CDF integral
            W₁ = ∫₀¹ |Δμ + Δσ · Φ⁻¹(t)| dt
        and substituting z = Φ⁻¹(t), the integral becomes the expectation
            W₁ = E_{Z ~ N(0,1)} |Δμ + Δσ · Z|.
        For X = aZ + b with Z standard normal, this expectation has a
        well-known closed form (folded-normal mean):
            E|aZ + b| = 2 |a| · φ(b/|a|) + b · (2 Φ(b/|a|) − 1).
        We apply it with a = σ₁ − σ₂ and b = μ₁ − μ₂. The degenerate
        case σ₁ = σ₂ is handled separately (W₁ = |Δμ|).

        Fully differentiable in (μ, σ).
        """
        dmu = self.mean - other.mean
        dsigma = self.std - other.std
        dsigma_abs = dsigma.abs()

        # Safe denominator for the main branch
        eps = torch.tensor(1e-12, dtype=dmu.dtype, device=dmu.device)
        dsigma_safe = torch.maximum(dsigma_abs, eps)
        z = dmu / dsigma_safe

        sqrt2 = torch.tensor(2.0, dtype=z.dtype, device=z.device).sqrt()
        sqrt_2pi = torch.tensor(2.0 * pi, dtype=z.dtype, device=z.device).sqrt()
        phi_z = torch.exp(-0.5 * z * z) / sqrt_2pi
        Phi_z = 0.5 * (1.0 + torch.erf(z / sqrt2))

        main = 2.0 * dsigma_abs * phi_z + dmu * (2.0 * Phi_z - 1.0)
        degen = dmu.abs()  # σ₁ = σ₂ limit

        return torch.where(dsigma_abs > eps, main, degen)

    # -------- utility --------
    def snapshot(self) -> "GaussianPolicy":
        """
        Return a non-trainable clone with the current parameter values.
        Used to keep π_old around during one update iteration.
        """
        clone = GaussianPolicy(cfg=self.cfg)
        with torch.no_grad():
            clone.mean_param.copy_(self.mean_param)
            clone.log_std_param.copy_(self.log_std_param)
        for p in clone.parameters():
            p.requires_grad_(False)
        return clone

    def state_dict_minimal(self) -> dict:
        return {"mean": float(self.mean_param.item()),
                "std": float(self.std.item())}

    def __repr__(self):
        return (f"GaussianPolicy(mean={float(self.mean_param):.4f}, "
                f"std={float(self.std):.4f})")
