"""Independent double-precision check of the probe output, the same gate the paper applies.

For each case and arm: re-evaluate every returned point with the Python mirror of the surrogate,
and report how many meet the specification at each tolerance and how many violate any other
constraint. The pinned build's own sample files are read for the same seeds as the reference.
"""
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent.parent      # .../code, whatever the checkout is called
sys.path.insert(0, str(CODE))
import problems  # noqa: E402

TOL = [1e-3, 1e-5, 1e-7]

print(f"{'case':9} {'arm':11} {'seed':>4} {'rows':>5} "
      + " ".join(f"{'<'+str(t):>9}" for t in TOL) + f" {'viol':>5} {'medres':>10}")
for case in ("steel", "concrete"):
    p = problems.load(case)
    for seed in (1, 2):
        for arm, path in (("pin", CODE / "out" / "sample" / f"{case}_s{seed}.csv"),
                          ("new fd", CODE / "out" / "probe" / f"{case}_fd_s{seed}.csv"),
                          ("new analyt", CODE / "out" / "probe" / f"{case}_analytical_s{seed}.csv")):
            if not path.exists():
                print(f"{case:9} {arm:11} {seed:>4} missing {path}")
                continue
            x, _ = problems.read_rows(path, p)
            # Re-evaluated from the inputs alone: the file's own outputs are never trusted.
            y = p.evaluate(x)
            res = p.spec_residual(x, y)
            ok = p.feasible_except_spec(x, y)
            print(f"{case:9} {arm:11} {seed:>4} {len(x):>5} "
                  + " ".join(f"{int(((res < t) & ok).sum()):>9}" for t in TOL)
                  + f" {int((~ok).sum()):>5} {np.median(res):>10.2e}")
