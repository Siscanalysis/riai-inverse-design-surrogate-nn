"""Test fit of each surrogate on unseen groups, and its size relative to the data.

Usage (from code/):  python check_fit.py        (after prep_data.py and the training)

This produces the architecture and fit columns of the manuscript's case table, which audit finding
Z-28 (2026-10-21) found had no producing script: Neural Designer reported them, read_nd_report.py
only printed them, and they were transcribed by hand. Here they are recomputed from the committed
model files and the committed split, and stored in out/fit.json for audit_numbers.py to check.

R^2 is the true coefficient of determination, 1 - SSres/SStot, on the Testing rows only, whose
groups never appear in training. It is NOT the squared correlation that some tools report as
"determination": that one cannot go negative and flatters a biased fit.

Rows per weight is the number of rows in the table divided by the number of trained parameters of
the network, counted from the model itself (both weight matrices and both bias vectors).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CASES = ("concrete", "steel", "kernel")


def split_roles(case: str, rows: int) -> np.ndarray:
    """Role per table row, from data/<case>_split.csv (csv_row is 1-based over the file)."""
    table = pd.read_csv(HERE / "data" / f"{case}_split.csv")
    role = np.array(["unassigned"] * rows, dtype=object)
    for csv_row, name in zip(table["csv_row"].astype(int), table["role"]):
        index = csv_row - 2
        if 0 <= index < rows:
            role[index] = name
    return role


def true_r2(measured: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(np.sum((measured - predicted) ** 2))
    total = float(np.sum((measured - measured.mean()) ** 2))
    return float("nan") if total == 0 else 1.0 - residual / total


def main() -> None:
    import problems

    out = {}
    print(f"{'case':9} {'output':22} {'role':8} {'n':>5} {'true R2':>9}")
    for case in CASES:
        p = problems.load(case)
        path = HERE / "data" / f"{case}.csv"
        if not path.exists():
            print(f"{case:9} table absent (see DATA.md); skipped")
            continue
        table = pd.read_csv(path)
        role = split_roles(case, len(table))
        x = table[p.inputs].to_numpy(float)
        predicted = p.evaluate(x)

        s = p.surrogate
        parameters = int(s.W1.size + s.b1.size + s.W2.size + s.b2.size)
        case_out = {"rows": int(len(table)), "hidden": int(s.b1.size),
                    "parameters": parameters,
                    "rows_per_weight": round(len(table) / parameters, 2),
                    "r2_test": {}}

        for j, name in enumerate(p.outputs):
            if name not in table.columns:
                continue
            measured = table[name].to_numpy(float)
            for which in ("Testing",):
                mask = role == which
                if mask.sum() == 0:
                    continue
                value = true_r2(measured[mask], predicted[mask, j])
                case_out["r2_test"][name] = round(value, 4)
                print(f"{case:9} {name:22} {which:8} {int(mask.sum()):>5} {value:>9.4f}")
        print(f"{case:9} {'':22} {'':8} {'':>5} "
              f"hidden {case_out['hidden']}, {parameters} parameters, "
              f"{case_out['rows_per_weight']} rows/weight")
        out[case] = case_out

    (HERE / "out" / "fit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote out/fit.json")


if __name__ == "__main__":
    main()
