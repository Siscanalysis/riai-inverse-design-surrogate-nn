"""Why the tolerance ladder is flat for the projecting methods (debug check, not a paper artifact).

Usage (from code/):  python check_tolerance.py

For each case and method it prints the distribution of the specification residual
r = |f_k(x) - c| / |c| over the points the method returned, how many of them fall in each
tolerance band, and how many are distinct designs (points closer than 1e-3 in the
span-normalized design space are counted once). A flat ladder is expected when every returned
point already sits far below the tightest tolerance; it would be a bug if instead the points
were duplicates, or if the residual were not being recomputed from the network.
"""

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import problems as P

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
BANDS = [(0.0, 1e-7), (1e-7, 1e-6), (1e-6, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2)]
SOURCES = {"repair": "out/sample/{case}_s{seed}.csv",
           "slsqp": "out/baselines/{case}_sample_slsqp_s{seed}.csv",
           "rejection": "out/baselines/{case}_sample_rejection_s{seed}.csv"}


def distinct(p, x, tolerance=1e-3):
    if len(x) < 2:
        return len(x)
    z = p.normalize(x)
    pairs = cKDTree(z).query_pairs(tolerance)
    duplicates = set()
    for i, j in pairs:
        if i not in duplicates:
            duplicates.add(j)
    return len(x) - len(duplicates)


def main(seed=1):
    for case in ("concrete", "steel"):
        p = P.load(case)
        print(f"\n=== {case} (seed {seed})")
        for method, pattern in SOURCES.items():
            path = HERE / pattern.format(case=case, seed=seed)
            if not path.exists():
                continue
            x, _ = P.read_rows(path, p)
            if len(x) == 0:
                continue
            y = p.evaluate(x)
            ok = p.feasible_except_spec(x, y)
            r = p.spec_residual(x, y)[ok]
            x = x[ok]
            counts = " ".join(f"{sum((r > lo) & (r <= hi)):>6}" for lo, hi in BANDS)
            print(f"  {method:10} returned {len(x):>6}  distinct {distinct(p, x):>6}   "
                  f"residual median {np.median(r):.2e}  p95 {np.percentile(r, 95):.2e}  max {r.max():.2e}")
            print(f"  {'':10} points per band (<=1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2): {counts}")


if __name__ == "__main__":
    main()
