"""Error of each surrogate on its own specification response, near the requested value.

Usage (from code/):  python check_spec_error.py      (after prep_data.py and the training)

This is the number that bounds every physical claim in the paper: a tolerance tighter than the
surrogate's own error on the specified output measures consistency with the model, not accuracy
of the part. Audit finding B-13 (2026-10-20) was that the figures quoted in the manuscript had
no producing script, so this file supplies one and stores the result in out/spec_error.json for
audit_numbers.py to check against.

Definition. On the held-out (Testing) rows only, and only on those whose MEASURED specification
response lies inside a window around the requested value, the relative error is
|predicted - measured| / |measured|. The specification of the kernel case is stated on the base-10
logarithm of the duration, so both sides are converted back first and the error is relative in
microseconds, exactly as problems.spec_residual does for the returned designs.

The window is reported at several widths, because "near the target" has no canonical width and a
single figure could be an artefact of the one chosen. The manuscript states +-25 % (Section 5),
so that is the column it quotes; the others are there to show the figure is not an artefact of
that choice.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WINDOWS = (0.10, 0.20, 0.25, 0.30, None)   # None = every held-out row, for reference
CASES = ("concrete", "steel", "kernel")


def held_out(case: str) -> pd.DataFrame:
    table = pd.read_csv(HERE / "data" / f"{case}.csv")
    split = pd.read_csv(HERE / "data" / f"{case}_split.csv")
    # csv_row is 1-based over the file including its header, so row i of the table is i + 2.
    testing = set(split.loc[split["role"] == "Testing", "csv_row"].astype(int))
    keep = [i for i in range(len(table)) if (i + 2) in testing]
    return table.iloc[keep].reset_index(drop=True)


def main() -> None:
    import problems

    out = {}
    print(f"{'case':9} {'spec':18} {'window':>8} {'n':>5} {'median %':>9} {'p90 %':>8}")
    for case in CASES:
        p = problems.load(case)
        rows = held_out(case)
        x = rows[p.inputs].to_numpy(float)
        measured = rows[p.specification].to_numpy(float)
        predicted = p.evaluate(x)[:, p.outputs.index(p.specification)]

        if p.specification.startswith("log10_"):
            measured, predicted = np.power(10.0, measured), np.power(10.0, predicted)
            target = 10.0 ** p.target
        else:
            target = p.target

        error = np.abs(predicted - measured) / np.abs(measured)
        case_out = {}
        for w in WINDOWS:
            near = np.ones(len(measured), bool) if w is None \
                else np.abs(measured - target) / abs(target) <= w
            if near.sum() == 0:
                continue
            med = float(np.median(error[near]) * 100)
            p90 = float(np.percentile(error[near], 90) * 100)
            label = "all" if w is None else f"+-{w:.0%}"
            case_out[label] = {"n": int(near.sum()), "median_pct": med, "p90_pct": p90}
            print(f"{case:9} {p.specification:18} {label:>8} {int(near.sum()):>5} "
                  f"{med:>9.1f} {p90:>8.1f}")
        out[case] = {"specification": p.specification, "target": float(target),
                     "windows": case_out}

    (HERE / "out" / "spec_error.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote out/spec_error.json")


if __name__ == "__main__":
    main()
