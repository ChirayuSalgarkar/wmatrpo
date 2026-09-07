# Provenance: reproduce_20260907T035641Z

Source of every number in the manuscript's unified comparison table
(Table, label fig:rl_comparison_professional) and basin-gap table
(label tab:basin_gap_experiment), and of the escape-rate / peak-reward
figures quoted in the surrounding prose.

- Produced by: python -m wmatrpo.scripts.reproduce_all
  (7 algorithms x N in {3,5,7,9} x seeds 0-4 x 4000 iters, batch 30;
  basin-gap: 4 strategies x k in {0.5,1.0,1.5,2.0} x seeds 0-4 x 2500 iters)
- Machine: author's MacBook Pro (macOS 26.6.2 arm64, Python 3.9.6), 7.36 h
- Code: github.com/ChirayuSalgarkar/wmatrpo @ 9d5ae40163c1 (IS fix a5b41ff included)
- MANIFEST.json records per-stage argv, return codes, wall times, package versions.
- Full per-seed raws remain on the author's machine:
  ~/wmatrpo/runs/reproduce_20260907T035641Z/unified/baselines_raw.csv
  ~/wmatrpo/runs/reproduce_20260907T035641Z/basingap/basingap_raw.csv

Note: MANIFEST records git_dirty=true (uncommitted local edits at launch,
from the buffering hotfix having been pulled; see reproduce.log).
