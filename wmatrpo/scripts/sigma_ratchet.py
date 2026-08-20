r"""
Generate the sigma-ratchet figure (reviewer point R2.3).

This script runs the faithful W-MATRPO algorithm on the 3-agent differential
game and instruments every dual solve to record, per agent-step, the old
policy standard deviation and the raw (pre-clamp) standard deviation of the
moment-matched Gaussian projection of the W2 pushforward. It then produces a
three-panel figure:

  (a) histogram of sigma_proj / sigma_old per dual solve  -> the mechanism:
      moment-matching a (generally bimodal, non-Gaussian) pushforward back to a
      single Gaussian inflates sigma on the majority of steps.
  (b) policy sigma trajectory for two caps (3.0 shipped, 1.0 mitigation)
      -> sigma ratchets up until it saturates whatever cap is set.
  (c) EMA-smoothed mean reward -> the consequence: reward peaks early then
      collapses as the policy becomes near-uniform noise.

Every annotated number (inflation fraction, median ratio, peak reward) is
computed from the run, not hard-coded.

Usage:
    python -m wmatrpo.scripts.sigma_ratchet
    python -m wmatrpo.scripts.sigma_ratchet --iters 1500 --seed 0 --out sigma_ratchet_R23.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wmatrpo.scripts.baselines import make_env
from wmatrpo.policy import GaussianPolicy, PolicyConfig
from wmatrpo.critic import CentralizedCritic, CriticConfig
from wmatrpo.dual_solver import DualSolver, DualSolverConfig
from wmatrpo.caatr import CAATR, CAATRConfig
from wmatrpo.algorithm import WMATRPO, WMATRPOConfig
from wmatrpo.utils import set_seed


def run(std_max: float, seed: int = 0, n: int = 3, iters: int = 1500) -> dict:
    """Run W-MATRPO once, capturing per-dual-solve sigma inflation."""
    set_seed(seed)
    env = make_env(n)
    pcfg = PolicyConfig(init_mean=1.5, init_std=0.5, std_min=0.1, std_max=std_max)
    pols = [GaussianPolicy(pcfg) for _ in range(n)]
    crit = CentralizedCritic(
        CriticConfig(n_agents=n, hidden=64, n_layers=2, lr=3e-3, n_update_epochs=8)
    )
    ds = DualSolver(DualSolverConfig(cost_type="l2"))
    ca = CAATR(
        CAATRConfig(C=0.02, epsilon_base=1e-8, epsilon_max=0.5, fallback_delta=0.1),
        n_agents=n,
    )
    algo = WMATRPO(env, pols, crit, ds, ca, WMATRPOConfig(batch_size=30, seed=seed))

    log = {"old": [], "raw": []}
    orig = ds.solve

    def wrapped(agent_id, batch, critic, policy_old, delta_i):
        old_std = float(policy_old.std.item())
        m, s, lam, info = orig(
            agent_id=agent_id,
            batch=batch,
            critic=critic,
            policy_old=policy_old,
            delta_i=delta_i,
        )
        # info["pushforward_std_raw"] is the moment-matched sigma BEFORE the
        # std_max clamp -- this is the honest measure of the ratchet.
        log["old"].append(old_std)
        log["raw"].append(info.get("pushforward_std_raw", s))
        return m, s, lam, info

    ds.solve = wrapped

    rew = []
    for _ in range(iters):
        algo.step()
        rew.append(float(algo.reward_history[-1]))

    return dict(
        std_hist=np.array(algo.std_history).tolist(),
        rew=rew,
        old=log["old"],
        raw=log["raw"],
    )


def make_figure(d: dict, out: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))

    def panel_letter(ax, letter):
        ax.text(
            -0.13, 1.02, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left",
        )

    # Panel (a): per-step raw projection inflation (the mechanism)
    raw = np.array(d["cap3"]["raw"])
    old = np.array(d["cap3"]["old"])
    ratio = raw / np.maximum(old, 1e-6)
    ax = axes[0]
    ax.hist(ratio, bins=60, range=(0.6, 1.8), color="#2b6cb0", alpha=0.85, edgecolor="none")
    ax.axvline(1.0, color="#c1272d", lw=1.6, ls="--",
               label=r"no change ($\sigma_{\mathrm{proj}}=\sigma_{\mathrm{old}}$)")
    ax.axvline(np.median(ratio), color="#1b1b1b", lw=1.4,
               label=f"median = {np.median(ratio):.3f}")
    ax.set_xlabel(r"$\sigma_{\mathrm{proj}}\,/\,\sigma_{\mathrm{old}}$ per dual solve")
    ax.set_ylabel("count (agent-steps)")
    frac = np.mean(ratio > 1.0)
    ax.set_title(f"Gaussian projection inflates $\\sigma$\non {frac:.0%} of steps", fontsize=8)
    ax.legend(frameon=False, fontsize=6.3, loc="upper right")

    # Panel (b): sigma trajectory, capped vs mitigation
    ax = axes[1]
    s3 = np.array(d["cap3"]["std_hist"])
    s1 = np.array(d["cap1"]["std_hist"])
    it = np.arange(s3.shape[1])
    for i in range(s3.shape[0]):
        ax.plot(it, s3[i], color="#c1272d", lw=1.2, alpha=0.8)
        ax.plot(it, s1[i], color="#2b6cb0", lw=1.2, alpha=0.8)
    ax.axhline(3.0, color="#c1272d", ls=":", lw=1.0)
    ax.axhline(1.0, color="#2b6cb0", ls=":", lw=1.0)
    ax.plot([], [], color="#c1272d", lw=1.5, label=r"$\sigma_{\max}=3.0$ (shipped run)")
    ax.plot([], [], color="#2b6cb0", lw=1.5, label=r"$\sigma_{\max}=1.0$ (mitigation)")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"policy $\sigma$ (per agent)")
    ax.set_title(r"$\sigma$ ratchets to whatever cap is set", fontsize=8)
    ax.legend(frameon=False, fontsize=6.3, loc="center right")

    # Panel (c): reward collapse under the ratchet
    ax = axes[2]

    def ema(x, w=25):
        x = np.array(x)
        k = np.ones(w) / w
        return np.convolve(x, k, mode="valid")

    r3 = d["cap3"]["rew"]
    ax.plot(ema(r3), color="#c1272d", lw=1.6, label="mean reward (EMA-25)")
    ax.axhline(np.max(r3), color="#1b1b1b", ls="--", lw=1.0, label=f"peak = {np.max(r3):.3f}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("mean batch reward")
    ax.set_title("Inflated $\\sigma$ $\\Rightarrow$ near-noise policy,\nreward collapses after early peak", fontsize=8)
    ax.legend(frameon=False, fontsize=6.3, loc="center right")

    for a, l in zip(axes, "abc"):
        panel_letter(a, l)
    fig.suptitle(
        r"The $\sigma$-ratchet (R2.3): moment-matched Gaussian projection of the "
        r"$W_2$ pushforward inflates $\sigma$ each step until it saturates the cap",
        fontsize=8.5, y=1.04,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=1500, help="iterations per run")
    ap.add_argument("--seed", type=int, default=0, help="random seed")
    ap.add_argument("--n", type=int, default=3, help="number of agents")
    ap.add_argument("--out", type=str, default="sigma_ratchet_R23.png", help="output PNG path")
    ap.add_argument("--data-out", type=str, default=None, help="optional JSON dump of the raw run data")
    args = ap.parse_args()

    data = {
        "cap3": run(std_max=3.0, seed=args.seed, n=args.n, iters=args.iters),
        "cap1": run(std_max=1.0, seed=args.seed, n=args.n, iters=args.iters),
    }
    if args.data_out:
        Path(args.data_out).write_text(json.dumps(data))
        print(f"wrote {args.data_out}")

    make_figure(data, args.out)


if __name__ == "__main__":
    main()
