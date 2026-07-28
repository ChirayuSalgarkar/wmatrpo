"""
Smoke test: verify IPPO, MAPPO, HAPPO, and W-MATRPO all run for a few
iterations on a small differential game without error, and that each moves
the policy away from initialization.

Run:
    python -m wmatrpo.tests.test_baselines_smoke
"""
from __future__ import annotations

import numpy as np

from wmatrpo.scripts.baselines import build_algo, make_env


def _run(name: str, n_agents: int = 3, n_iters: int = 60) -> dict:
    env = make_env(n_agents)
    algo = build_algo(name, env, seed=0, batch_size=30)
    init_means = np.array([float(p.mean.item()) for p in algo.policies])
    for _ in range(n_iters):
        algo.step()
    final_means = np.array([float(p.mean.item()) for p in algo.policies])
    moved = float(np.linalg.norm(final_means - init_means))
    return {"final_means": final_means, "moved": moved,
            "final_reward": algo.reward_history[-1]}


def main():
    print("=== Baseline smoke test (3 agents, 60 iterations each) ===\n")
    results = {}
    for name in ["ippo", "mappo", "happo", "wmatrpo"]:
        try:
            r = _run(name)
            ok = r["moved"] > 0.05
            flag = "PASS" if ok else "WEAK"
            print(f"  [{flag}] {name:8s} | moved {r['moved']:.3f} from init | "
                  f"final means [{', '.join(f'{m:.2f}' for m in r['final_means'])}]")
            results[name] = ok
        except Exception as e:
            print(f"  [ERROR] {name:8s} | {e!r}")
            results[name] = False

    print()
    if all(results.values()):
        print("All four algorithms run and move the policy. Baselines are wired correctly.")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"Investigate: {failed}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
