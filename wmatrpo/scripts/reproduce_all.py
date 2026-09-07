"""
Reproduce every reported result in one command.

Runs, in order:
  1. the test suite (fails fast if the eq.-15 IS correction is not live)
  2. the unified comparison table  (baselines.py)
  3. the basin-gap ablation table  (basingap_compare.py)
  4. the R2.3 variance-ratchet figure (sigma_ratchet.py)

Every stage writes into one timestamped directory under runs/ together with a
MANIFEST.json recording the commit, the environment, the exact argv of each
stage, and its wall time -- so any number in the paper traces to a run.

Usage:
    python -m wmatrpo.scripts.reproduce_all                    # 5 seeds, full
    python -m wmatrpo.scripts.reproduce_all --seeds 0 1 2      # 3 seeds
    python -m wmatrpo.scripts.reproduce_all --wmatrpo-only     # skip baselines
    python -m wmatrpo.scripts.reproduce_all --quick            # ~2 min smoke
    python -m wmatrpo.scripts.reproduce_all --skip ratchet     # drop a stage
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

TESTS = [
    "wmatrpo.tests.test_is_weights",
    "wmatrpo.tests.test_wasserstein",
    "wmatrpo.tests.test_dual_solver",
    "wmatrpo.tests.test_hatrpo",
    "wmatrpo.tests.test_allocation",
    "wmatrpo.tests.test_baselines_smoke",
]


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unavailable"


def _run(argv: list[str], log_path: Path) -> dict:
    """Run a stage, stream to console and tee to a log file."""
    print(f"\n$ {' '.join(argv)}", flush=True)
    t0 = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(argv, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        proc.wait()
    dt = time.time() - t0
    return {"argv": argv, "returncode": proc.returncode,
            "wall_seconds": round(dt, 1), "log": log_path.name}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--agent-counts", nargs="+", type=int, default=[3, 5, 7, 9])
    ap.add_argument("--n-iterations", type=int, default=4000,
                    help="iterations for the unified table")
    ap.add_argument("--basingap-iterations", type=int, default=2500)
    ap.add_argument("--batch-size", type=int, default=30)
    ap.add_argument("--wmatrpo-only", action="store_true",
                    help="only the two W-MATRPO variants (baseline rows are "
                         "unaffected by the IS fix -- they do not use DualSolver)")
    ap.add_argument("--quick", action="store_true",
                    help="smoke everything: 1 seed, N=3, 200/200 iterations")
    ap.add_argument("--skip", nargs="+", default=[],
                    choices=["tests", "unified", "basingap", "ratchet"])
    ap.add_argument("--out", default=None,
                    help="output dir under runs/ (default: reproduce_<UTC stamp>)")
    args = ap.parse_args()

    if args.quick:
        args.seeds = [0]
        args.agent_counts = [3]
        args.n_iterations = 200
        args.basingap_iterations = 200

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REPO_ROOT / "runs" / (args.out or f"reproduce_{stamp}")
    out.mkdir(parents=True, exist_ok=True)

    algos = (["wmatrpo", "wmatrpo_fixed"] if args.wmatrpo_only else
             ["wmatrpo", "wmatrpo_fixed", "hatrpo", "hatrpo_caatr",
              "happo", "mappo", "ippo"])

    manifest = {
        "started_utc": stamp,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": (None if _git("rev-parse", "HEAD") == "unavailable"
                      else bool(_git("status", "--porcelain"))),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "argv": sys.argv,
        "config": {"seeds": args.seeds, "agent_counts": args.agent_counts,
                   "n_iterations": args.n_iterations,
                   "basingap_iterations": args.basingap_iterations,
                   "batch_size": args.batch_size, "algorithms": algos,
                   "quick": args.quick, "skipped": args.skip},
        "stages": {},
    }
    try:
        import torch, numpy, scipy, pandas
        manifest["versions"] = {"torch": torch.__version__, "numpy": numpy.__version__,
                                "scipy": scipy.__version__, "pandas": pandas.__version__}
    except Exception:
        pass

    def save():
        (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    print("=" * 72)
    print(f"REPRODUCE ALL  ->  {out.relative_to(REPO_ROOT)}")
    print(f"commit {manifest['git_commit'][:12]}"
          f"{'  (WORKING TREE DIRTY)' if manifest['git_dirty'] is True else ''}")
    print(f"seeds {args.seeds}   agent counts {args.agent_counts}")
    print("=" * 72)
    save()

    failed: list[str] = []

    # ---- 1. tests -------------------------------------------------------
    if "tests" not in args.skip:
        print("\n" + "-" * 72 + "\n[1/4] test suite\n" + "-" * 72)
        for mod in TESTS:
            r = _run([sys.executable, "-m", mod], out / f"test_{mod.split('.')[-1]}.log")
            manifest["stages"][mod] = r
            save()
            if r["returncode"] != 0:
                failed.append(mod)
        if failed:
            print("\nFAILED tests: " + ", ".join(failed))
            print("Refusing to run the sweeps: fix the failures first.")
            manifest["aborted"] = "test failure"
            save()
            return 1
        print("\nall tests passed")

    # ---- 2. unified comparison table -----------------------------------
    if "unified" not in args.skip:
        print("\n" + "-" * 72 + "\n[2/4] unified comparison table\n" + "-" * 72)
        r = _run([sys.executable, "-m", "wmatrpo.scripts.baselines",
                  "--algorithms", *algos,
                  "--agent-counts", *map(str, args.agent_counts),
                  "--seeds", *map(str, args.seeds),
                  "--n-iterations", str(args.n_iterations),
                  "--batch-size", str(args.batch_size),
                  "--out", str((out / "unified").relative_to(REPO_ROOT))],
                 out / "unified.log")
        manifest["stages"]["unified"] = r
        save()
        if r["returncode"] != 0:
            failed.append("unified")

    # ---- 3. basin-gap ablation -----------------------------------------
    if "basingap" not in args.skip:
        print("\n" + "-" * 72 + "\n[3/4] basin-gap ablation\n" + "-" * 72)
        r = _run([sys.executable, "-m", "wmatrpo.scripts.basingap_compare",
                  "--strategies", "fixed", "greedy", "weighted", "caatr",
                  "--ks", "0.5", "1.0", "1.5", "2.0",
                  "--seeds", *map(str, args.seeds),
                  "--n-iterations", str(args.basingap_iterations),
                  "--batch-size", str(args.batch_size),
                  "--out", str((out / "basingap").relative_to(REPO_ROOT))],
                 out / "basingap.log")
        manifest["stages"]["basingap"] = r
        save()
        if r["returncode"] != 0:
            failed.append("basingap")

    # ---- 4. sigma-ratchet figure ---------------------------------------
    if "ratchet" not in args.skip:
        print("\n" + "-" * 72 + "\n[4/4] variance-ratchet figure (R2.3)\n" + "-" * 72)
        r = _run([sys.executable, "-m", "wmatrpo.scripts.sigma_ratchet"],
                 out / "ratchet.log")
        manifest["stages"]["ratchet"] = r
        save()
        if r["returncode"] != 0:
            failed.append("ratchet")

    total = sum(s.get("wall_seconds", 0) for s in manifest["stages"].values())
    manifest["total_wall_seconds"] = round(total, 1)
    manifest["failed_stages"] = failed
    save()

    print("\n" + "=" * 72)
    print(f"DONE in {total/3600:.2f} h -> {out.relative_to(REPO_ROOT)}")
    for name, s in manifest["stages"].items():
        flag = "ok  " if s["returncode"] == 0 else "FAIL"
        print(f"  [{flag}] {name:34s} {s['wall_seconds']:8.1f}s")
    print("\nKey outputs:")
    for p in ["unified/baselines_summary.csv", "unified/baselines_raw.csv",
              "basingap/basingap_summary.csv", "MANIFEST.json"]:
        f = out / p
        print(f"  {'✓' if f.exists() else '·'} {(out / p).relative_to(REPO_ROOT)}")
    print("=" * 72)
    if failed:
        print("FAILED stages: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
