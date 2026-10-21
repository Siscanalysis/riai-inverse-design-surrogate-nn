"""What the kernel case can check that the continuous cases cannot.

Usage (from code/):  python check_kernel.py      (after the kernel runs)

Two references no other case in the study has:

  exact set     the design space is a catalogue of 4680 legal configurations, so every
                configuration the surrogate offers at a given tolerance can be enumerated.
                Recall is the share of that set a method returns, precision the share of what
                it returns that belongs to it. Counting solutions says nothing about whether a
                method found the ones that exist; this does.

  measurement   3311 of those 4680 configurations were actually run on the GPU, so a returned
                configuration can be compared against a measured duration instead of against
                the surrogate. The remaining 1369 pass the kernel's launch rules but were never
                measured, and nothing here can tell whether they compile.

Writes out/kernel_checks.json (read by audit_numbers.py).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import problems as P

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
CASE = "kernel"
TAUS = [1e-2, 1e-3, 1e-4, 1e-5]
TARGET_US = 8000.0


def key_of(x: np.ndarray) -> set:
    return set(map(tuple, np.rint(x).astype(int)))


def rows_of(path: Path, p):
    if not path.exists():
        return np.zeros((0, len(p.inputs))), np.zeros((0, len(p.outputs)))
    return P.read_rows(path, p)


def main():
    p = P.load(CASE)
    measured = pd.read_csv(HERE / "data" / "kernel_measured.csv")
    measured_key = {tuple(r) : d for r, d in
                    zip(np.rint(measured[p.inputs].to_numpy()).astype(int).tolist(),
                        measured["duration_us"].to_numpy(float))}

    x_all, y_all = rows_of(OUT / "baselines" / f"{CASE}_sample_enumeration_s1.csv", p)
    if not len(x_all):
        raise SystemExit("no enumeration reference: run the kernel baselines first")
    residual_all = p.spec_residual(x_all, y_all)
    feasible_all = p.feasible_except_spec(x_all, y_all)
    print(f"catalogue: {len(x_all)} legal configurations, "
          f"{sum(tuple(r) in measured_key for r in np.rint(x_all).astype(int).tolist())} of them measured")

    exact = {}
    for tau in TAUS:
        exact[tau] = key_of(x_all[(residual_all <= tau) & feasible_all])
        print(f"  exact solution set at tau {tau:g}: {len(exact[tau])} configurations")
    best_attainable = float(residual_all[feasible_all].min())
    print(f"  closest the catalogue can come to the target: {best_attainable:.3e} "
          f"({best_attainable * TARGET_US:.1f} us)")
    # How wide the band has to be before it holds anything: the recipe a practitioner needs, and
    # the reason a strict equality cannot work here whatever the solver does.
    sorted_residual = np.sort(residual_all[feasible_all])
    needed = {n: float(sorted_residual[n - 1]) for n in (1, 5, 10, 25, 50) if n <= len(sorted_residual)}
    for n, width in needed.items():
        print(f"  smallest band holding {n:3} configurations: +-{width:.2e} "
              f"({width * TARGET_US:.1f} us)")

    runs = pd.read_csv(OUT / f"baselines_runs_{CASE}.csv")
    library = pd.read_csv(OUT / f"runs_{CASE}.csv")
    rows = []
    for seed in range(1, 11):
        sets = {}
        for _, r in library[(library["mode"] == "sample") & (library["seed"] == seed)].iterrows():
            # One row per statement: the equality (which returns nothing here) and the band.
            suffix = "" if r["formulation"] == "constraint" else f"_{r['formulation']}"
            label = "repair" if r["formulation"] == "constraint" else f"repair_{r['formulation']}"
            sets[label] = OUT / "sample" / f"{CASE}{suffix}_s{seed}.csv"
        for method in runs[(runs["task"] == "sample") & (runs["seed"] == seed)]["method"].unique():
            sets[method] = OUT / "baselines" / f"{CASE}_sample_{method}_s{seed}.csv"
        for method, path in sets.items():
            x, y = rows_of(path, p)
            if not len(x):
                continue
            residual, feasible = p.spec_residual(x, y), p.feasible_except_spec(x, y)
            keys = np.rint(x).astype(int).tolist()
            in_measured = np.array([tuple(k) in measured_key for k in keys])
            measured_residual = np.array([abs(measured_key[tuple(k)] - TARGET_US) / TARGET_US
                                          for k in keys if tuple(k) in measured_key])
            row = {"method": method, "seed": seed, "returned": len(x),
                   "distinct": len(key_of(x)), "measured_share": float(in_measured.mean()),
                   # Did the method reach the closest the catalogue allows, or stop short of it?
                   "best_residual": float(residual[feasible].min()) if feasible.any() else np.nan,
                   "best_attainable": best_attainable}
            for tau in TAUS:
                found = key_of(x[(residual <= tau) & feasible])
                row[f"recall_{tau:g}"] = len(found & exact[tau]) / max(len(exact[tau]), 1)
                row[f"precision_{tau:g}"] = len(found & exact[tau]) / max(len(found), 1)
                row[f"valid_{tau:g}"] = len(found)
            # Of the returned points that were really measured, how far the measurement is
            # from the target: the only check in the study that does not go through a surrogate.
            row["measured_residual_median"] = float(np.median(measured_residual)) if len(measured_residual) else np.nan
            row["measured_within_1pct"] = float(np.mean(measured_residual <= 1e-2)) if len(measured_residual) else np.nan
            rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "kernel_checks.csv", index=False)
    columns = ["returned", "distinct", "measured_share", "best_residual", "measured_residual_median",
               "measured_within_1pct"] + [f"{k}_{t:g}" for t in TAUS for k in ("valid", "recall", "precision")]
    summary = table.groupby("method")[columns].median()
    print("\nmedian over seeds:")
    print(summary.to_string(float_format=lambda v: f"{v:.4g}"))

    payload = {"exact_set_sizes": {f"{t:g}": len(exact[t]) for t in TAUS},
               "best_attainable": best_attainable,
               "band_for_n_configurations": {str(n): w for n, w in needed.items()},
               "catalogue": int(len(x_all)),
               "measured_in_catalogue": int(sum(tuple(r) in measured_key
                                                for r in np.rint(x_all).astype(int).tolist())),
               "target_us": TARGET_US,
               "per_method": {m: {c: (None if pd.isna(v) else float(v)) for c, v in row.items()}
                              for m, row in summary.iterrows()}}
    (OUT / "kernel_checks.json").write_text(json.dumps(payload, indent=2))
    print("\nwrote out/kernel_checks.json and out/kernel_checks.csv")


if __name__ == "__main__":
    main()
