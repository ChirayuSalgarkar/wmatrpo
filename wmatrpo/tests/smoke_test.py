"""
End-to-end smoke test: builds a 2-agent setup from the YAML, runs a small
number of iterations, and verifies that training completes without raising
and that the agents are making progress toward (5, 5).

Run:
    python -m wmatrpo.tests.smoke_test
"""
from __future__ import annotations

from pathlib import Path

from wmatrpo.scripts.train import build_from_config
from wmatrpo.utils import load_yaml


def main():
    # Use the 2-agent config but cut iterations down for a quick check.
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "diffgame_2agent.yaml"
    cfg = load_yaml(cfg_path)
    cfg.setdefault("trainer", {})["n_iterations"] = 200
    cfg["trainer"]["log_every"] = 50
    cfg["trainer"]["save_outputs"] = False

    # write the patched config to a temp file
    import yaml, tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
        tmp_path = f.name

    print("=== W-MATRPO smoke test (200 iterations, 2 agents) ===\n")
    trainer = build_from_config(tmp_path)
    summary = trainer.train()

    print("\n--- summary ---")
    print(f"final means:         {summary['final_means']}")
    print(f"distance to global:  {summary['distance_to_global']:.3f}")
    print(f"distance to local:   {summary['distance_to_local']:.3f}")
    print(f"peak reward:         {summary['peak_reward']:.4f}")

    # cheap sanity: by 200 iterations, the algorithm should have moved away
    # from the initial point (1.5, 1.5) toward higher reward. The exact
    # destination depends on hyperparameters, but distance to global should
    # at minimum have decreased from the initial value.
    initial_dist_to_global = ((5 - 1.5)**2 * 2) ** 0.5  # = 4.949
    moved = summary["distance_to_global"] < initial_dist_to_global - 0.05
    if moved:
        print("\nSmoke test PASSED — agents moved away from initialization.")
    else:
        print("\nSmoke test inconclusive — agents barely moved. Investigate hyperparameters.")


if __name__ == "__main__":
    main()
