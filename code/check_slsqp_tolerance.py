"""How the requested tolerance changes the cost of the SLSQP baseline.

Usage (from code/):  python check_slsqp_tolerance.py [seeds]

The library repair accepts a point as soon as it meets the specification to a relative 1e-6
and stops; SLSQP, left to its own convergence, keeps going to about 1e-14. Charging it for
that extra accuracy makes the comparison of evaluations per solution unfair, and it also hides
the tolerance from the ladder: every point it returns passes every tolerance plotted.

This script runs the set-recovery task with the local solve stopped at several requested
tolerances, at the same budget, and reports how many solutions come back, what they cost and
what residual they actually carry.
"""

import sys
import time

import numpy as np
import pandas as pd

import baselines as B
import problems as P

TOLERANCES = [1e-3, 1e-5, 1e-6, 1e-7, None]
CASES = ("concrete", "steel")


def main(seeds=(1, 2, 3)):
    rows = []
    for case in CASES:
        p = P.load(case)
        space = B.Space(p)
        for seed in seeds:
            budget = B.budget_for(case, "sample", seed)
            for tolerance in TOLERANCES:
                rng = np.random.default_rng(1000*seed + B.METHOD_OFFSET["slsqp"])
                start = time.perf_counter()
                x, y, used = B.slsqp_nearest(p, space, budget, rng, None, stop_tolerance=tolerance)
                residual = p.spec_residual(x, y) if len(x) else np.array([np.nan])
                valid = int(((residual <= 1e-5) & p.feasible_except_spec(x, y)).sum()) if len(x) else 0
                rows.append({"case": case, "seed": seed, "tolerance": tolerance or "converged",
                             "returned": len(x), "valid_1e-5": valid,
                             "evals_per_solution": used/max(len(x), 1),
                             "median_residual": float(np.median(residual)),
                             "seconds": time.perf_counter() - start})
                print(f"{case:9} seed {seed} tol {str(tolerance):9} -> {len(x):5} points, "
                      f"{used/max(len(x),1):7.0f} evals each, median residual "
                      f"{np.median(residual):.1e}", flush=True)
    table = pd.DataFrame(rows)
    summary = table.groupby(["case", "tolerance"])[["returned", "valid_1e-5", "evals_per_solution",
                                                    "median_residual"]].median()
    print("\n" + summary.to_string())
    table.to_csv(B.OUT/"slsqp_tolerance.csv", index=False)
    print("\nwrote out/slsqp_tolerance.csv")


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]] or (1, 2, 3))
