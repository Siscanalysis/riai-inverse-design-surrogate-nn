"""Metrics for every run (OpenNN and baselines), re-checked independently.

Usage (from code/):  python analyze.py

Every returned point is re-evaluated with the network (problems.py) and checked against
every constraint; the specification residual is r = |f(x) - c| / |c|.

Set recovery (task sample)
  valid(tau)     points with r <= tau that satisfy every other constraint
  evals/sol      surrogate forward passes per valid point at tau = 1e-5
  coverage       mean over free inputs of (range of the valid points) / (box span)
  spread         mean nearest-neighbour distance (span-normalized free inputs) of a random
                 subset of m valid points, m = smallest valid count (>= 20) among the methods
                 of that case and seed (capped at 200), so sets of different size compare
                 fairly; undefined for a method with fewer than m valid points
  data_dist      median distance of the valid points to the nearest training row (same
                 normalization); data_ref is the same statistic for the training rows among
                 themselves (leave-one-out)
  near_data      fraction of the valid points closer to a training row than the 95th
                 percentile of those leave-one-out training distances

Optimization (task single): best objective value among the returned points that satisfy
every constraint with r <= tau, for tau = 1e-4 (the tolerance given to the baselines) and
tau = 1e-5.

Fronts (task bi): hypervolume of the non-dominated valid points (r <= 1e-4), objectives
normalized per case to [0, 1] with the extremes of all valid points of all methods and
seeds, reference point (1.1, 1.1).

Writes out/metrics_<task>.csv (one row per run) and out/summary.json (medians and
quartiles over seeds, the numbers quoted in the manuscript).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import problems as P

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
CASES = ["concrete", "steel", "kernel"]  # HEA (cardinality) retired 2026-09-28, see notes/DECISIONS.md
TAUS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
SEEDS = range(1, 11)


def library_runs(case):
    return pd.read_csv(OUT / f"runs_{case}.csv")


def baseline_runs(case):
    f = OUT / f"baselines_runs_{case}.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["case", "task", "method", "seed", "evaluations"])


def load_points(p, path):
    if not path.exists():
        return np.zeros((0, len(p.inputs)))
    x, _ = P.read_rows(path, p)
    return x


def check(p, x):
    if len(x) == 0:
        return np.zeros(0), np.zeros(0, bool), np.zeros((0, len(p.outputs)))
    y = p.evaluate(x)
    return p.spec_residual(x, y), p.feasible_except_spec(x, y), y


def training_inputs(p):
    table = pd.read_csv(HERE / "data" / f"{p.name}.csv")
    x = table[p.inputs].to_numpy(float)
    fixed = {c["expression"]: c["values"][0] for c in p.constraints
             if c["condition"] == "Equal" and c["expression"] in p.inputs}
    for name, value in fixed.items():  # rows in the slice the problem designs for
        x = x[np.isclose(x[:, p.inputs.index(name)], value)]
    return x


def nn_spread(z, m, rng):
    if len(z) < max(m, 2) or m < 2:
        return np.nan
    z = z[rng.choice(len(z), m, replace=False)]
    d, _ = cKDTree(z).query(z, k=2)
    return float(d[:, 1].mean())


# ----------------------------------------------------------------------------- sample
def sample_metrics():
    rows = []
    for case in CASES:
        p = P.load(case)
        train = p.normalize(training_inputs(p))
        tree = cKDTree(train)
        d, _ = tree.query(train, k=2)
        data_ref = float(np.median(d[:, 1]))
        support_radius = float(np.percentile(d[:, 1], 95))
        lib, base = library_runs(case), baseline_runs(case)
        for seed in SEEDS:
            sets = {}
            for _, r in lib[(lib["mode"] == "sample") & (lib["seed"] == seed)].iterrows():
                # One row per statement of the specification: the equality, and the band where
                # the case declares one (the catalogue case states both).
                suffix = "" if r["formulation"] == "constraint" else f"_{r['formulation']}"
                label = "repair" if r["formulation"] == "constraint" else f"repair_{r['formulation']}"
                sets[label] = (load_points(p, OUT / "sample" / f"{case}{suffix}_s{seed}.csv"),
                               int(r["forwards"]))
            rows_here = base[(base["task"] == "sample") & (base["seed"] == seed)]
            for method in rows_here["method"].unique():          # whatever baselines this case ran
                b = rows_here[rows_here["method"] == method]
                sets[method] = (load_points(p, OUT / "baselines" / f"{case}_sample_{method}_s{seed}.csv"),
                                int(b["evaluations"].iloc[0]))
            if not sets:
                continue
            checked = {m: check(p, x) + (x, ev) for m, (x, ev) in sets.items()}
            # Reference tolerance for the geometry of the recovered set. A catalogue case cannot
            # reach 1e-5 at all, so there it is the band the problem states.
            tight = max(1e-5, p.band)
            valid_tight = {m: c[3][(c[0] <= tight) & c[1]] for m, c in checked.items()}
            sizes = [len(v) for v in valid_tight.values() if len(v) >= 20]
            m_common = min([200] + sizes)
            for method, (res, ok, y, x, ev) in checked.items():
                rng = np.random.default_rng(seed)
                row = {"case": case, "seed": seed, "method": method, "evaluations": ev, "returned": len(x),
                       "tight": tight, "max_residual": float(res[ok].max()) if ok.any() else np.nan}
                for tau in TAUS:
                    row[f"valid_{tau:g}"] = int(((res <= tau) & ok).sum())
                v = valid_tight[method]
                z = p.normalize(v)
                # On a finite catalogue several starts land on the same configuration, so the
                # distinct count is the one that means anything; on a continuum they coincide.
                row["distinct"] = len(np.unique(np.round(v, 6), axis=0)) if len(v) else 0
                row["evals_per_sol"] = ev / len(v) if len(v) else np.inf
                row["coverage"] = float(np.mean((z.max(0) - z.min(0)))) if len(v) > 1 else 0.0
                row["spread"] = nn_spread(z, m_common, rng)
                row["m_common"] = m_common
                row["data_dist"] = float(np.median(tree.query(z)[0])) if len(v) else np.nan
                row["data_ref"] = data_ref
                row["near_data"] = float(np.mean(tree.query(z)[0] <= support_radius)) if len(v) else np.nan
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "metrics_sample.csv", index=False)
    return df


# ----------------------------------------------------------------------------- single
def solve_files(case, task, seed):
    """(method, path, evaluations) of every optimization run of a case/task/seed."""
    items = []
    lib = library_runs(case)
    for _, r in lib[(lib["mode"] == "solve") & (lib["task"] == task) & (lib["seed"] == seed)].iterrows():
        items.append((f"{r['solver']}_{r['formulation']}",
                      OUT / "solve" / f"{case}_{r['solver']}_{r['formulation']}_{task}_s{seed}.csv", int(r["forwards"]),
                      r["status"]))
    base = baseline_runs(case)
    for _, r in base[(base["task"] == task) & (base["seed"] == seed)].iterrows():
        items.append((r["method"], OUT / "baselines" / f"{case}_{task}_{r['method']}_s{seed}.csv",
                      int(r["evaluations"]), "ok"))
    return items


def single_metrics():
    rows = []
    for case in CASES:
        p = P.load(case)
        for seed in SEEDS:
            for method, path, ev, status in solve_files(case, "single", seed):
                x = load_points(p, path)
                res, ok, y = check(p, x)
                row = {"case": case, "seed": seed, "method": method, "evaluations": ev, "returned": len(x),
                       "status": status}
                f = p.objective_values(x, y, p.single)[:, 0] if len(x) else np.zeros(0)
                sign = -1 if p.single[0]["sense"] == "Maximize" else 1
                # The two loose tolerances matter for a catalogue case, where nothing reaches
                # 1e-4 because the attainable responses are a finite set; the two tight ones are
                # what the continuous cases are quoted at.
                for tau in (1e-2, 1e-3, 1e-4, 1e-5):
                    good = (res <= tau) & ok
                    row[f"best_{tau:g}"] = float(sign * f[good].min()) if good.any() else np.nan
                row["min_residual"] = float(res[ok].min()) if ok.any() else np.nan
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "metrics_single.csv", index=False)
    return df


# ----------------------------------------------------------------------------- bi
def nondominated(F):
    keep = np.ones(len(F), bool)
    for i in range(len(F)):
        if keep[i]:
            dominated = np.all(F[i] <= F, axis=1) & np.any(F[i] < F, axis=1)
            keep &= ~dominated
    return F[keep]


def bi_metrics():
    from pymoo.indicators.hv import HV
    rows = []
    for case in CASES:
        p = P.load(case)
        if not p.bi:
            continue
        runs = []
        for seed in SEEDS:
            for method, path, ev, status in solve_files(case, "bi", seed):
                x = load_points(p, path)
                res, ok, y = check(p, x)
                good = (res <= 1e-4) & ok
                F = p.objective_values(x[good], y[good], p.bi) if good.any() else np.zeros((0, 2))
                ladder = {f"usable_{tau:g}": int(((res <= tau) & ok).sum())
                          for tau in (5e-2, 1e-2, 1e-3)}
                runs.append((seed, method, ev, len(x), F, status,
                             int(((res <= 1e-5) & ok).sum()), ladder))
        allF = np.vstack([r[4] for r in runs if len(r[4])])
        lo, hi = allF.min(0), allF.max(0)
        hv = HV(ref_point=np.array([1.1, 1.1]))
        for seed, method, ev, n, F, status, n_tight, ladder in runs:
            front = nondominated((F - lo) / np.maximum(hi - lo, 1e-12)) if len(F) else np.zeros((0, 2))
            row = {"case": case, "seed": seed, "method": method, "evaluations": ev, "returned": n,
                   "valid_1e-4": len(F), "valid_1e-5": n_tight, "front": len(front),
                   "hv": float(hv(front)) if len(front) else 0.0, "status": status}
            # How much of the front a designer could actually use, from a tolerance as loose as
            # the surrogate's own error down to the arithmetic: the hypervolume can look similar
            # while almost nothing on the front meets the specification.
            row.update(ladder)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "metrics_bi.csv", index=False)
    return df


def q(series):
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return None
    return {"median": float(s.median()), "q1": float(s.quantile(0.25)), "q3": float(s.quantile(0.75)),
            "min": float(s.min()), "max": float(s.max()), "n": int(len(s))}


def summarize(sample, single, bi):
    out = {}
    single_cols = ["evaluations", "returned", "best_0.01", "best_0.001", "best_0.0001",
                   "best_1e-05", "min_residual"]
    # Where the whole design space was enumerated, that run gives the exact optimum at each
    # tolerance, so every other method can be scored by how often it actually reaches it.
    reference = {}
    for case, g in single.groupby("case"):
        exact = g[g["method"] == "enumeration"]
        if not exact.empty:
            reference[case] = {c: float(exact[c].median()) for c in single_cols
                               if c.startswith("best_") and exact[c].notna().any()}
    for name, df, cols in (("sample", sample, [c for c in sample.columns if c not in ("case", "seed", "method")]),
                           ("single", single, single_cols),
                           ("bi", bi, ["evaluations", "returned", "usable_0.05", "usable_0.01",
                                       "usable_0.001", "valid_1e-4", "valid_1e-5", "front", "hv"])):
        out[name] = {}
        for (case, method), g in df.groupby(["case", "method"]):
            entry = {c: q(g[c]) for c in cols}
            if "status" in g:
                entry["failed_runs"] = int((g["status"] != "ok").sum())
            if name == "single":
                entry["success_1e-4"] = int(g["best_0.0001"].notna().sum())
                entry["success_1e-5"] = int(g["best_1e-05"].notna().sum())
                # best_* is already stored in the problem's own direction, so reaching the exact
                # optimum means matching the enumerated value.
                for col, best in reference.get(case, {}).items():
                    entry[f"optimum_{col}"] = int((np.abs(g[col] - best) <= 1e-4).sum())
                if reference.get(case):
                    entry["optimum_reference"] = reference[case]
            out[name].setdefault(case, {})[method] = entry
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    s = sample_metrics()
    o = single_metrics()
    b = bi_metrics()
    summarize(s, o, b)
    pd.set_option("display.width", 250)
    print(s.groupby(["case", "method"]).median(numeric_only=True).round(4).T)
    print(o.groupby(["case", "method"])[["evaluations", "best_0.0001", "best_1e-05"]].median().round(4))
    print(b.groupby(["case", "method"])[["evaluations", "valid_1e-4", "front", "hv"]].median().round(4))
