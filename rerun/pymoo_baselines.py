#!/usr/bin/env python3
"""
pymoo baselines (GA / DE / PSO / CMA-ES) on the SAME exported surrogate, SAME repair,
SAME budget as the C++ methods. Two tasks:

  CNC : minimize cutting force F subject to the fixed-Ra band |Ra - c| <= band
        (the mixed-sense optimization task). Metric: best feasible F (lower is better).
  HEA : find compositions on the simplex hitting the hardness band |HV - c| <= band.
        Repair = min-norm projection onto {sum(x)=1} cap box (same operator as ours).
        Metric: feasibility %, best |HV - c|.

Budget = 48000 surrogate evaluations (pop 100 x ~480 gens). Every algorithm evaluates
REPAIRED inputs, so all receive the same repair. Run:  python pymoo_baselines.py cnc|hea
"""
import sys, os, csv, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BUDGET, POP = 48000, 100
NGEN = BUDGET // POP

def load_surrogate(case):
    spec = importlib.util.spec_from_file_location(case, os.path.join(HERE, "models", f"{case}_surrogate.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.NeuralNetwork()

def data_box(case):
    csvf = {"cnc": "cnc_multi_meanresp.csv", "hea": "hea_hardness_10elt.csv"}[case]
    rows = list(csv.reader(open(os.path.join(HERE, "data", csvf))))
    hdr = rows[0]
    n_in = {"cnc": 4, "hea": 10}[case]
    X = np.array([[float(v) for v in r[:n_in]] for r in rows[1:]])
    return X.min(0), X.max(0)

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
    nn = load_surrogate(case)
    lo, up = data_box(case)
    simplex = (case == "hea")
    ok = 0                       # target output index: Ra (cnc) / HV (hea) is output 0
    ff = 1 if case == "cnc" else None
    n_in = len(lo)

    def out(X):
        return np.array(nn.calculate_batch_output(np.asarray(X, float)))

    # reachable target + band by sampling (matches the C++ protocol)
    rng = np.random.default_rng(0)
    S = lo + (up - lo) * rng.random((4000, n_in))
    if simplex: S = project_simplex_box(S, lo, up)
    yv = np.sort(out(S)[:, ok])
    pct = 55 if case == "cnc" else 60
    target = yv[int(len(yv) * pct / 100)]
    band = 1e-2 * (yv[-1] - yv[0])

    class Rep(Repair):
        def _do(self, problem, X, **kw):
            return project_simplex_box(X, lo, up) if simplex else np.clip(X, lo, up)
    repair = Rep()

    class P(Problem):
        def __init__(self):
            super().__init__(n_var=n_in, n_obj=1, n_ieq_constr=1, xl=lo, xu=up)
        def _evaluate(self, X, o, *a, **k):
            Xr = project_simplex_box(X, lo, up) if simplex else np.clip(X, lo, up)
            Y = out(Xr)
            g = np.abs(Y[:, ok] - target) - band            # <= 0 feasible (in band)
            f = Y[:, ff] if case == "cnc" else np.abs(Y[:, ok] - target)  # min F, or drive to band
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
        "CMA-ES": CMAES(x0=(lo+up)/2, sigma=0.25, restarts=0, maxfevals=BUDGET),
    }

    rows = []
    for name, alg in algos.items():
        try:
            res = minimize(P(), alg, get_termination("n_gen", NGEN), seed=1, verbose=False)
        except Exception as e:
            rows.append((name, None, None, None, f"error:{e}")); print(f"  {name:8s} error: {e}"); continue
        # gather all evaluated solutions, repair, filter to band, report best
        X = res.pop.get("X") if res.pop is not None else res.X.reshape(1, -1)
        Xr = project_simplex_box(X, lo, up) if simplex else np.clip(X, lo, up)
        Y = out(Xr)
        feas = np.abs(Y[:, ok] - target) <= band
        nfeas = int(feas.sum())
        best_F = float(Y[feas, ff].min()) if (case == "cnc" and nfeas) else None
        best_gap = float(np.abs(Y[:, ok] - target).min())
        simplex_resid = float(np.abs(Xr.sum(1) - 1).mean()) if simplex else 0.0
        rows.append((name, nfeas, best_F, best_gap, simplex_resid))
        extra = f"bestF={best_F:.2f}" if best_F is not None else f"bestGap={best_gap:.3g}"
        print(f"  {name:8s} feas_in_pop={nfeas:3d}/{len(Xr)}  {extra}  simplexResid={simplex_resid:.2g}")

    with open(os.path.join(HERE, "out", f"pymoo_{case}.csv"), "w", newline="") as fp:
        w = csv.writer(fp); w.writerow(["algo", "feas_in_pop", "best_F", "best_gap", "simplex_resid"])
        for r in rows: w.writerow(r)
    print(f"  target={target:.4g} band=+/-{band:.4g}  budget={BUDGET}  -> out/pymoo_{case}.csv")

if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    run(sys.argv[1] if len(sys.argv) > 1 else "cnc")
