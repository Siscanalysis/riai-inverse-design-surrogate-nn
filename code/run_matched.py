"""Objective formulation at the budget of the constraint formulation.

Usage (from code/):  python run_matched.py <case> [seeds]

With the specification as a Fixed objective there is no output constraint to repair
through the network, so a run with the default setting spends two to three orders of
magnitude fewer forward passes than the same solver with the specification as a
constraint. For a like-for-like comparison this script reruns the objective formulation
with its effort scaled by k = (forwards of the constraint run, or of the band run where the
case states one) / (forwards of the default
objective run), same case, task, solver and seed: DomainContraction draws k times more
points per iteration (at most DC_CAP times, so the matched DC runs may stay below the
budget; their actual forward passes are logged), GeneticResponse runs k times more
generations. The runs are logged
with formulation "objective-matched".
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EXE = HERE / "bin" / "inverse_cases.exe"
ITERATIONS, POINTS, ROUNDS = 10, 200, 50
DC_CAP = 20  # larger point counts make the front bookkeeping of DomainContraction impractical


def main(case: str, seeds):
    runs = pd.read_csv(HERE / "out" / f"runs_{case}.csv")
    solve = runs[runs["mode"] == "solve"]
    for seed in seeds:
        for task in sorted(solve["task"].unique()):
            for solver in ("dc", "ga"):
                sel = solve[(solve["seed"] == seed) & (solve["task"] == task) & (solve["solver"] == solver)]
                done = sel[sel["formulation"] == "objective-matched"]
                # Reference budget: the statement that admits solutions. On a catalogue case that
                # is the band, since the equality is unsatisfiable there whatever the budget.
                con = sel[sel["formulation"] == "band"]
                if con.empty:
                    con = sel[sel["formulation"] == "constraint"]
                obj = sel[sel["formulation"] == "objective"]
                if not done.empty or con.empty or obj.empty:
                    continue
                k = max(1, round(con["forwards"].iloc[0] / max(1, obj["forwards"].iloc[0])))
                iterations, points = (ITERATIONS, POINTS * min(k, DC_CAP)) if solver == "dc" else (ITERATIONS * k, POINTS)
                cmd = [str(EXE), "solve", case, solver, "objective-matched", task, str(seed),
                       str(iterations), str(points), str(ROUNDS)]
                print(f"k={k}:", " ".join(cmd[1:]), flush=True)
                out = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
                print("\n".join(l for l in (out.stdout + out.stderr).splitlines() if "local domains" not in l), flush=True)


if __name__ == "__main__":
    case = sys.argv[1]
    if len(sys.argv) > 2:
        a, _, b = sys.argv[2].partition("-")
        seeds = range(int(a), int(b or a) + 1)
    else:
        seeds = range(1, 11)
    main(case, seeds)
