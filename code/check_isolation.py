"""Is the band optimum isolated in the catalogue? The a-priori test of Section 7.3.

Usage (from code/):  python check_isolation.py       (after inverse_cases.exe problems)

The repair wins on a catalogue when rounding a relaxed solution has nowhere to land. Call the band
optimum ISOLATED when none of its catalogue neighbours, the admissible configurations reached by
moving one lever one step along its own catalogue, lies inside the band. If it is isolated, a
method that rounds at the end leaves the feasible region by construction and has to re-project; if
some neighbour is still inside, rounding can survive and no advantage should be expected.

The test costs one pass over the catalogue and needs no optimizer, which is the point: it says in
advance whether the repair matters. Audit finding Z-29 (2026-10-21) was that the manuscript quoted
this sweep with no script behind it; this is that script, and it writes out/isolation.json for
audit_numbers.py.

Everything is computed on the surrogate's predictions over the admissible catalogue, not on the
measured subset, because that is what the optimizers see.
"""

import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TARGETS_US = (4000.0, 6000.0, 8000.0, 10000.0, 12000.0, 16000.0, 20000.0, 30000.0)


def admissible(p):
    """Every catalogue point that satisfies the input-only constraints of the case."""
    catalogues = p.catalogues()
    if len(catalogues) != len(p.inputs):
        raise SystemExit("check_isolation.py expects every input to range over a catalogue")
    grid = np.array(list(itertools.product(*[catalogues[j] for j in range(len(p.inputs))])),
                    dtype=float)
    keep = np.ones(len(grid), bool)
    blank = np.zeros((len(grid), len(p.outputs)))
    for c in p.constraints:
        if c["condition"] in ("AllowedSet", "Integer"):
            continue
        value = p.value(c["expression"], grid, blank)
        if c["condition"] == "LessEqual":
            keep &= value <= c["values"][0] + 1e-9
        elif c["condition"] == "GreaterEqual":
            keep &= value >= c["values"][0] - 1e-9
        elif c["condition"] == "Between":
            keep &= (value >= c["values"][0] - 1e-9) & (value <= c["values"][-1] + 1e-9)
    return grid[keep], catalogues


def main() -> None:
    import problems

    p = problems.load("kernel")
    points, catalogues = admissible(p)
    outputs = p.evaluate(points)
    duration = np.power(10.0, outputs[:, p.outputs.index(p.specification)])
    objective = outputs[:, p.outputs.index("achieved_occupancy")]
    index = {tuple(row): i for i, row in enumerate(points)}
    band = p.band

    print(f"{len(points)} admissible configurations, band +-{band:.0%}")
    print(f"{'target us':>10} {'in band':>8} {'best obj':>9} {'nbrs in':>9}  verdict")
    sweep = {}
    for target in TARGETS_US:
        inside = np.abs(duration - target) / target <= band
        if not inside.any():
            sweep[str(int(target))] = {"in_band": 0}
            print(f"{target:>10.0f} {0:>8}  nothing inside the band")
            continue
        candidates = np.where(inside)[0]
        best = candidates[np.argmax(objective[candidates])]
        point = points[best]

        neighbours = still_inside = 0
        for j in range(len(p.inputs)):
            values = catalogues[j]
            at = int(np.where(values == point[j])[0][0])
            for step in (-1, 1):
                if not 0 <= at + step < len(values):
                    continue
                probe = point.copy()
                probe[j] = values[at + step]
                k = index.get(tuple(probe))
                if k is not None:                     # an inadmissible neighbour is not a landing
                    neighbours += 1
                    still_inside += bool(inside[k])

        isolated = still_inside == 0
        sweep[str(int(target))] = {
            "in_band": int(inside.sum()), "best_objective": round(float(objective[best]), 4),
            "neighbours": neighbours, "neighbours_in_band": still_inside, "isolated": isolated}
        print(f"{target:>10.0f} {int(inside.sum()):>8} {objective[best]:>9.4f} "
              f"{still_inside:>4}/{neighbours:<4}  {'ISOLATED' if isolated else 'rounding can survive'}")

    isolated = [t for t, v in sweep.items() if v.get("isolated")]
    result = {"admissible": int(len(points)), "band": band, "targets": sweep,
              "isolated_count": len(isolated), "targets_tried": len(TARGETS_US),
              "isolated_targets": [int(t) for t in isolated]}
    (HERE / "out" / "isolation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nisolated at {len(isolated)} of {len(TARGETS_US)} targets: "
          f"{', '.join(isolated)}\nwrote out/isolation.json")


if __name__ == "__main__":
    main()
