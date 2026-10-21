"""Physical plausibility and data support of the recovered concrete designs.

Usage (from code/):  python check_plausibility.py      (after the concrete runs)

The problem already forces the absolute volume balance (Section 4), so this script checks what
the optimizer was NOT asked to satisfy, using only quantities derived from the design itself:

  sand     fine / (fine + coarse) aggregate, by mass
  sp       water reducer as a percentage of binder (cement + fly ash)
  paste    binder + water volume as a fraction of the mix volume
  volume   sum of component volumes [L/m3], reported to confirm the constraint holds (within
           the checker's slack; the repair aims inside the band, SLSQP settles on it)

A design is *plausible* when sand, sp and paste all fall inside the range the plant itself uses
(5th-95th percentile of its 28-day mixes), and *supported* when it also lies closer to a
training mix than the 95th percentile of the mixes' own nearest-neighbour distance. The counts
per method, at one tolerance and one budget, say whether a method's advantage survives filters
it never optimized for.

Writes out/plausibility.json (read by audit_numbers.py).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import problems as P

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
DENSITY = {"Cement_kg_m3": 3.15, "FlyAsh_kg_m3": 2.30, "Water_kg_m3": 1.00,
           "FineAgg_kg_m3": 2.65, "CoarseAgg_kg_m3": 2.70, "WaterReducer_kg_m3": 1.10}
TAU = 1e-5
VOLUME_SLACK = 1.01  # the checker's own slack on the band, max(1e-2, 1e-3 |bound|)
SEEDS = range(1, 11)


def derived(frame: pd.DataFrame) -> pd.DataFrame:
    volume = sum(frame[name]/rho for name, rho in DENSITY.items())
    binder = frame["Cement_kg_m3"] + frame["FlyAsh_kg_m3"]
    return pd.DataFrame({
        "volume": volume,
        "sand": frame["FineAgg_kg_m3"]/(frame["FineAgg_kg_m3"] + frame["CoarseAgg_kg_m3"]),
        "sp": 100.0*frame["WaterReducer_kg_m3"]/binder,
        "paste": (binder/3.0 + frame["Water_kg_m3"])/volume,
    })


def valid_points(p, path: Path):
    if not path.exists():
        return None
    x, _ = P.read_rows(path, p)
    if len(x) == 0:
        return None
    y = p.evaluate(x)
    keep = p.feasible_except_spec(x, y) & (p.spec_residual(x, y) <= TAU)
    return pd.DataFrame(x[keep], columns=p.inputs)


def main():
    p = P.load("concrete")
    plant = pd.read_csv(HERE/"data"/"concrete.csv")
    plant = plant[plant["TestAge_days"] == 28]
    reference = derived(plant)
    band = {c: (float(reference[c].quantile(0.05)), float(reference[c].quantile(0.95)))
            for c in ("sand", "sp", "paste")}

    train = p.normalize(plant[p.inputs].to_numpy(float))
    tree = cKDTree(train)
    radius = float(np.percentile(tree.query(train, k=2)[0][:, 1], 95))

    runs = pd.read_csv(OUT/"runs_concrete.csv")
    baselines = pd.read_csv(OUT/"baselines_runs_concrete.csv")
    sources = {"repair": OUT/"sample"/"concrete_s{}.csv",
               "slsqp": OUT/"baselines"/"concrete_sample_slsqp_s{}.csv",
               "rejection": OUT/"baselines"/"concrete_sample_rejection_s{}.csv"}

    rows = []
    for seed in SEEDS:
        library = runs[(runs["mode"] == "sample") & (runs["seed"] == seed)]
        for method, pattern in sources.items():
            frame = valid_points(p, Path(str(pattern).format(seed)))
            if method == "repair":
                evaluations = int(library["forwards"].iloc[0])
            else:
                run = baselines[(baselines["task"] == "sample") & (baselines["method"] == method)
                                & (baselines["seed"] == seed)]
                evaluations = int(run["evaluations"].iloc[0]) if len(run) else 0
            if frame is None or len(frame) == 0:
                rows.append({"seed": seed, "method": method, "evaluations": evaluations,
                             "valid": 0, "plausible": 0, "supported": 0, "volume_ok": np.nan})
                continue
            values = derived(frame)
            plausible = np.ones(len(frame), bool)
            for column, (low, high) in band.items():
                plausible &= values[column].between(low, high).to_numpy()
            supported = tree.query(p.normalize(frame[p.inputs].to_numpy(float)))[0] <= radius
            rows.append({"seed": seed, "method": method, "evaluations": evaluations,
                         "valid": len(frame), "plausible": int(plausible.sum()),
                         "supported": int((plausible & supported).sum()),
                         "volume_ok": float(values["volume"].between(970 - VOLUME_SLACK, 1010 + VOLUME_SLACK).mean())})

    table = pd.DataFrame(rows)
    summary = table.groupby("method")[["evaluations", "valid", "plausible", "supported", "volume_ok"]].median()
    summary["% plausible"] = (100*summary["plausible"]/summary["valid"].replace(0, np.nan)).round(1)
    summary["% supported"] = (100*summary["supported"]/summary["valid"].replace(0, np.nan)).round(1)

    print(f"CONCRETE, every method at tau = {TAU:g}, same budget (medians of {len(list(SEEDS))} seeds)")
    print(f"plant bands: " + ", ".join(f"{c} [{lo:.3f}, {hi:.3f}]" for c, (lo, hi) in band.items()))
    print(summary.to_string())

    table.to_csv(OUT/"plausibility_concrete.csv", index=False)
    (OUT/"plausibility.json").write_text(json.dumps(
        {"bands": band, "support_radius": radius,
         "medians": {m: {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
                     for m, row in summary.iterrows()}}, indent=1))
    print("\nwrote out/plausibility.json and out/plausibility_concrete.csv")


if __name__ == "__main__":
    main()
