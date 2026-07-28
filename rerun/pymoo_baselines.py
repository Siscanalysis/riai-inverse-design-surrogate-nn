#!/usr/bin/env python3
"""
pymoo baselines (GA / DE / PSO / CMA-ES) on the SAME exported surrogate, SAME repair,
SAME budget as the C++ methods. Tasks:

  CNC  : minimize cutting force F subject to the fixed-Ra band |Ra - c| <= band
         (mixed-sense). Metric: best feasible F (lower is better).
  HEA  : find compositions on the simplex hitting the hardness band |HV - c| <= band.
         Repair = min-norm projection onto {sum(x)=1} cap box (same operator as ours).
  STEEL: room-temperature (T fixed at 27 C) inverse alloy design; fixed tensile-strength
         band |UTS - c| <= band, MAXIMIZE elongation (ductility). Metric: best feasible
         elongation (higher is better).

Budget = 12000 surrogate evaluations, enforced per-algorithm via an n_evals termination
(so CMA-ES's lambda != POP does not under/over-spend). Every algorithm evaluates REPAIRED
inputs, so all receive the same repair. Run:  python pymoo_baselines.py cnc|hea|steel [exact]
"""
import sys, os, csv, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BUDGET, POP = 12000, 100
NGEN = BUDGET // POP

# per-case configuration
#   csv   : data file (inputs are its first n_in columns)
#   n_in  : number of input columns
#   ok    : target-output index (the fixed output)
#   sec   : secondary-objective output index (None = pure fixed target)
#   sec_max: maximize (True) or minimize (False) the secondary
#   pct   : reachable-target percentile
#   fix   : list of (input_index, value) to hold fixed (e.g. steel temperature at 27 C)
#   simplex: sum(inputs)=1 repair
CASES = {
    "cnc":   dict(csv="cnc_multi_meanresp.csv",   n_in=4,  ok=0, sec=1,    sec_max=False, pct=55, fix=[],          simplex=False),
    "hea":   dict(csv="hea_hardness_10elt.csv",   n_in=10, ok=0, sec=None, sec_max=False, pct=60, fix=[],          simplex=True),
    "steel": dict(csv="steel_multi_fulldata.csv", n_in=14, ok=1, sec=2,    sec_max=True,  pct=55, fix=[(13, 27.0)], simplex=False),
}

def load_surrogate(case):
    spec = importlib.util.spec_from_file_location(case, os.path.join(HERE, "models", f"{case}_surrogate.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.NeuralNetwork()

def data_box(cfg):
    rows = list(csv.reader(open(os.path.join(HERE, "data", cfg["csv"]))))
    n_in = cfg["n_in"]
    X = np.array([[float(v) for v in r[:n_in]] for r in rows[1:]])
    lo, up = X.min(0), X.max(0)
    for idx, val in cfg["fix"]:            # pin fixed inputs (e.g. steel T=27) into the box
        lo[idx] = up[idx] = val
    return lo, up

def project_simplex_box(X, lo, up):
    """Min-norm projection onto {sum=1} cap [lo,up], iterative clip+redistribute (= affine repair)."""
    X = np.clip(X, lo, up)
    for _ in range(64):
        s = X.sum(1)
        e = s - 1.0
        done = np.abs(e) < 1e-9
        if done.all(): break
        active = np.where((e[:, None] > 0) & (X > lo + 1e-9) | (e[:, None] < 0) & (X < up - 1e-9), 1.0, 0.0)
        cnt = active.sum(1); cnt[cnt == 0] = 1
        X = np.clip(X - active * (e / cnt)[:, None], lo, up)
    return X

from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.optimize import minimize
from pymoo.termination import get_termination

def run(case):
    cfg = CASES[case]
    nn = load_surrogate(case)
    lo, up = data_box(cfg)
    simplex = cfg["simplex"]
    ok, ff, sec_max = cfg["ok"], cfg["sec"], cfg["sec_max"]
    n_in = len(lo)

    def repair_np(X):
        return project_simplex_box(X, lo, up) if simplex else np.clip(X, lo, up)
    def out(X):
        return np.array(nn.calculate_batch_output(np.asarray(X, float)))

    rng = np.random.default_rng(0)
    S = repair_np(lo + (up - lo) * rng.random((4000, n_in)))
    yv = np.sort(out(S)[:, ok])
    target = yv[int(len(yv) * cfg["pct"] / 100)]
    band = 1e-2 * (yv[-1] - yv[0])

    class Rep(Repair):
        def _do(self, problem, X, **kw): return repair_np(X)
    repair = Rep()

    class P(Problem):
        def __init__(self):
            super().__init__(n_var=n_in, n_obj=1, n_ieq_constr=1, xl=lo, xu=up)
        def _evaluate(self, X, o, *a, **k):
            Xr = repair_np(X)
            Y = out(Xr)
            g = np.abs(Y[:, ok] - target) - band            # <= 0 feasible (in band)
            if ff is None:
                f = np.abs(Y[:, ok] - target)               # pure fixed target: drive to band
            else:
                f = -Y[:, ff] if sec_max else Y[:, ff]      # optimize the secondary
            o["F"] = f.reshape(-1, 1)
            o["G"] = g.reshape(-1, 1)

    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.algorithms.soo.nonconvex.de import DE
    from pymoo.algorithms.soo.nonconvex.pso import PSO
    from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
    algos = {
        "GA":  GA(pop_size=POP, repair=repair, eliminate_duplicates=True),
        "DE":  DE(pop_size=POP),
        "PSO": PSO(pop_size=POP, repair=repair),
        "CMA-ES": CMAES(x0=(lo+up)/2, sigma=0.25, restarts=12, restart_from_best=True,
                        maxfevals=BUDGET, tolx=1e-12, tolfun=1e-12),
    }

    rows = []
    for name, alg in algos.items():
        try:
            res = minimize(P(), alg, get_termination("n_evals", BUDGET), seed=1, verbose=False)
        except Exception as e:
            rows.append((name, None, None, None, None, f"error:{e}")); print(f"  {name:8s} error: {e}"); continue
        n_eval = int(res.algorithm.evaluator.n_eval)
        X = res.pop.get("X") if res.pop is not None else res.X.reshape(1, -1)
        Xr = repair_np(X)
        Y = out(Xr)
        feas = np.abs(Y[:, ok] - target) <= band
        nfeas = int(feas.sum())
        if ff is not None and nfeas:
            best_sec = float(Y[feas, ff].max() if sec_max else Y[feas, ff].min())
        else:
            best_sec = None
        best_gap = float(np.abs(Y[:, ok] - target).min())
        simplex_resid = float(np.abs(Xr.sum(1) - 1).mean()) if simplex else 0.0
        rows.append((name, n_eval, nfeas, best_sec, best_gap, simplex_resid))
        extra = f"bestSec={best_sec:.3f}" if best_sec is not None else f"bestGap={best_gap:.3g}"
        print(f"  {name:8s} evals={n_eval:6d}  feas_in_pop={nfeas:3d}/{len(Xr)}  {extra}  simplexResid={simplex_resid:.2g}")

    with open(os.path.join(HERE, "out", f"pymoo_{case}.csv"), "w", newline="") as fp:
        w = csv.writer(fp); w.writerow(["algo", "n_eval", "feas_in_pop", "best_secondary", "best_gap", "simplex_resid"])
        for r in rows: w.writerow(r)
    sec_name = {None: "-"}.get(ff, cfg.get("sec"))
    print(f"  target={target:.4g} band=+/-{band:.4g}  budget={BUDGET}"
          f"  secondary_idx={ff} ({'max' if sec_max else 'min'})  -> out/pymoo_{case}.csv")


def run_exact(case):
    """Exact-target task: minimize |f_k(x) - target| (root finding), then measure how many
    final-population points fall AT the target for shrinking bands, AND their diversity."""
    cfg = CASES[case]
    nn = load_surrogate(case)
    lo, up = data_box(cfg)
    simplex = cfg["simplex"]
    ok = cfg["ok"]
    n_in = len(lo)
    span = np.maximum(up - lo, 1e-12)

    def repair_np(X):
        return project_simplex_box(X, lo, up) if simplex else np.clip(X, lo, up)
    def out(X):
        return np.array(nn.calculate_batch_output(np.asarray(X, float)))

    rng = np.random.default_rng(0)
    S = repair_np(lo + (up - lo) * rng.random((4000, n_in)))
    yv = np.sort(out(S)[:, ok])
    target = yv[int(len(yv) * cfg["pct"] / 100)]
    rng_out = float(yv[-1] - yv[0])

    def fill_pct(P):
        if len(P) < 2: return 0.0
        return float(((P.max(0) - P.min(0)) / span).mean() * 100.0)

    class Rep(Repair):
        def _do(self, problem, X, **kw): return repair_np(X)
    repair = Rep()

    class P(Problem):
        def __init__(self):
            super().__init__(n_var=n_in, n_obj=1, n_ieq_constr=0, xl=lo, xu=up)
        def _evaluate(self, X, o, *a, **k):
            o["F"] = np.abs(out(repair_np(X))[:, ok] - target).reshape(-1, 1)

    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.algorithms.soo.nonconvex.de import DE
    from pymoo.algorithms.soo.nonconvex.pso import PSO
    from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
    algos = {
        "GA":  GA(pop_size=POP, repair=repair, eliminate_duplicates=True),
        "DE":  DE(pop_size=POP),
        "PSO": PSO(pop_size=POP, repair=repair),
        "CMA-ES": CMAES(x0=(lo+up)/2, sigma=0.25, restarts=12, restart_from_best=True,
                        maxfevals=BUDGET, tolx=1e-12, tolfun=1e-12),
    }
    deltas = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    rows = []
    print(f"=== pymoo EXACT target ({case}): minimize |f-c|, target={target:.4g}, range={rng_out:.4g} ===")
    print(f"  {'algo':8s} {'bestGap':>10s}  " + "  ".join(f"n@{d:g}" for d in deltas) + f"   {'fill%@1e-3':>9s}")
    for name, alg in algos.items():
        try:
            res = minimize(P(), alg, get_termination("n_evals", BUDGET), seed=1, verbose=False)
        except Exception as e:
            print(f"  {name:8s} error: {e}"); rows.append([name, None] + [None]*len(deltas) + [None]); continue
        X = res.pop.get("X") if res.pop is not None else res.X.reshape(1, -1)
        Xr = repair_np(X)
        resid = np.abs(out(Xr)[:, ok] - target)
        best = float(resid.min())
        counts = [int((resid <= d * rng_out).sum()) for d in deltas]
        infill = Xr[resid <= 1e-3 * rng_out]
        fpct = fill_pct(infill)
        rows.append([name, best] + counts + [fpct])
        print(f"  {name:8s} {best:10.3g}  " + "  ".join(f"{c:4d}" for c in counts) + f"   {fpct:8.1f}")
    with open(os.path.join(HERE, "out", f"pymoo_{case}_exact.csv"), "w", newline="") as fp:
        w = csv.writer(fp); w.writerow(["algo", "best_gap"] + [f"n_within_{d:g}" for d in deltas] + ["fill_pct_at_1e-3"])
        for r in rows: w.writerow(r)
    print(f"  pop_size={POP} (returned population size); IDC returns the full manifold, not a converged pop.")
    print(f"  -> out/pymoo_{case}_exact.csv")

if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    case = sys.argv[1] if len(sys.argv) > 1 else "cnc"
    if len(sys.argv) > 2 and sys.argv[2] == "exact":
        run_exact(case)
    else:
        run(case)
