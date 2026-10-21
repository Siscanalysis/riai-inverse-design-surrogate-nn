"""Builds the numeric training tables and the a-priori data splits of the three cases.

Sources are the local mirrors in Desktop/benchmark_datasets (see ../README.md for the
original citations). Each case writes

    data/<case>.csv        numeric table, inputs first, targets last
    data/<case>_split.csv  csv_row,role   (role in Training / Validation / Testing)
    data/<case>_meta.json  input and target names, group column used for the split

Splits are grouped so that correlated rows never straddle training and testing:
concrete rows by mix design (one mix is tested at 7 and 28 days), CNC rows by recipe
(replicated measurements; kept for the record, not used as a case), steel rows
by alloy (one alloy is tested at several temperatures), HEA rows by composition.
The kernel case needs no grouping: each configuration was measured once, so its group is
the configuration itself and the split falls back to a split by row.
Groups are assigned 70/15/15 with a fixed seed.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# Where the raw source tables live. Set RAI_BENCHMARK_DATA to the root that holds the
# so_realdata/ tree; the default is the authors' workstation. The concrete and kernel tables are
# CC BY 4.0 and ship in data/; the NIMS steel table does not (see DATA.md), so steel() needs this.
BENCH_ROOT = Path(os.environ.get(
    "RAI_BENCHMARK_DATA", r"C:\Users\Artelnics\Desktop\benchmark_datasets")) / "so_realdata"
BENCH = BENCH_ROOT / "Q_materials_so"
BENCH_NAS = BENCH_ROOT / "K_nas"
DATA = HERE / "data"
SEED = 2026


def grouped_split(groups: pd.Series, seed: int = SEED, fractions=(0.70, 0.15, 0.15)) -> list[str]:
    rng = np.random.default_rng(seed)
    unique = groups.drop_duplicates().to_numpy()
    rng.shuffle(unique)
    n = len(unique)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    role_of = {}
    for i, g in enumerate(unique):
        role_of[g] = "Training" if i < n_train else ("Validation" if i < n_train + n_val else "Testing")
    return [role_of[g] for g in groups]


def write_case(name: str, table: pd.DataFrame, inputs: list[str], targets: list[str], groups: pd.Series,
               group_note: str) -> None:
    table = table[inputs + targets].astype(float).reset_index(drop=True)
    assert not table.isna().any().any(), f"{name}: missing values"
    roles = grouped_split(groups.reset_index(drop=True))
    table.to_csv(DATA / f"{name}.csv", index=False, float_format="%.6g")
    pd.DataFrame({"csv_row": np.arange(len(table)) + 2, "role": roles}).to_csv(DATA / f"{name}_split.csv", index=False)
    meta = {
        "inputs": inputs,
        "targets": targets,
        "rows": len(table),
        "groups": int(groups.nunique()),
        "group_note": group_note,
        "roles": {r: roles.count(r) for r in ("Training", "Validation", "Testing")},
        "minimums": table.min().to_dict(),
        "maximums": table.max().to_dict(),
    }
    (DATA / f"{name}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{name}: {len(table)} rows, {meta['groups']} groups, roles {meta['roles']}")


def concrete() -> None:
    raw = pd.read_csv(BENCH / "guangxi_concrete_strength_slump" / "data" / "guangxi_concrete_strength_slump.csv")
    inputs = ["Cement_kg_m3", "FineAgg_kg_m3", "CoarseAgg_kg_m3", "Water_kg_m3", "WaterReducer_kg_m3",
              "FlyAsh_kg_m3", "Accelerator_kg_m3", "SilicaFume_kg_m3", "TestAge_days"]
    targets = ["CompressiveStrength_MPa", "Slump_mm"]
    groups = raw[inputs[:-1]].astype(str).agg("|".join, axis=1)
    write_case("concrete", raw, inputs, targets, groups, "mix design (one mix tested at 7 and 28 days)")


def cnc() -> None:
    raw = pd.read_csv(BENCH / "cnc_turning_aisi304" / "data" / "cnc_turning_aisi304.csv")
    inputs = ["cutting_speed", "feed", "depth_of_cut", "tool_wear_VB"]
    targets = ["Ra", "F"]
    groups = raw[inputs].astype(str).agg("|".join, axis=1)
    write_case("cnc", raw, inputs, targets, groups, "recipe (vc, f, ap, VB): replicated measurements")


def hea() -> None:
    raw = pd.read_csv(BENCH / "hea_hardness_rao_wei" / "data" / "hea_hardness_10elt.csv")
    inputs = ["Al", "Co", "Cr", "Cu", "Fe", "Mn", "Mo", "Ni", "Ti", "V"]
    targets = ["HV"]
    groups = raw[inputs].round(4).astype(str).agg("|".join, axis=1)
    write_case("hea", raw, inputs, targets, groups, "composition (repeated alloys)")


def kernel() -> None:
    """CUDA convolution kernel auto-tuning: eight integer knobs on finite catalogues.

    Every row is one distinct configuration, measured once on one GPU, so there is no group
    to keep together and the split is by configuration (each its own group). The design space
    is a catalogue of 9216 points of which these 3311 are the feasible, measured ones, which
    lets the solution set of a specification be enumerated from the measurements themselves;
    kernel_measured.csv keeps the measured duration for that check.

    The duration spans 68x, so the network is trained on its base-10 logarithm; the analysis
    converts back and reports the residual as a relative error in microseconds.
    """
    raw = pd.read_csv(BENCH_NAS / "ktt_cuda_autotuning" / "data" / "source"
                      / "1070-conv_output.csv")
    raw = raw.rename(columns={"Computation duration (us)": "duration_us"})
    inputs = ["TBX", "TBY", "LOCAL", "WPTX", "WPTY", "VECTOR", "UNROLL_FACTOR", "PADDING"]
    measured = ["duration_us", "achieved_occupancy"]
    raw = raw[inputs + measured].apply(pd.to_numeric, errors="coerce").dropna()
    raw["log10_duration"] = np.log10(raw["duration_us"])
    raw[inputs + ["duration_us", "achieved_occupancy"]].to_csv(
        DATA / "kernel_measured.csv", index=False, float_format="%.8g")
    groups = raw[inputs].astype(str).agg("|".join, axis=1)
    assert groups.nunique() == len(raw), "a configuration was measured twice"
    write_case("kernel", raw, inputs, ["log10_duration", "achieved_occupancy"], groups,
               "configuration (each measured once; the split is therefore by row)")


def steel() -> None:
    raw = pd.read_csv(BENCH / "nims_lowalloy_steel" / "data" / "nims_lowalloy_steel.csv")
    inputs = ["C", "Si", "Mn", "P", "S", "Ni", "Cr", "Mo", "Cu", "V", "Al", "N", "Nb_Ta", "temperature_C"]
    targets = ["proof_stress_MPa", "tensile_strength_MPa", "elongation_pct", "reduction_area_pct"]
    write_case("steel", raw, inputs, targets, raw["alloy_code"], "alloy (tested at several temperatures)")


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    concrete()
    cnc()
    hea()
    steel()
    kernel()
