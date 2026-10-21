"""Baselines for the three inverse-design problems, on the same surrogates and budgets.

Usage (from code/):  python baselines.py <concrete|hea|steel> <sample|single|bi|all> [seeds]

Every baseline states the problem from out/problems.json (problems.py), evaluates the
Python export of the same network (surrogates.py, float64) and is held to the surrogate
forward-pass budget of the OpenNN runs of the same case, task and seed (the largest of
them, out/runs_<case>.csv). Inputs pinned by an Equal constraint (age, temperature,
unused admixtures) are fixed for every method, as the library's domain narrowing does.

Set recovery (task "sample"), against the library's repair sampler:
  rejection   uniform candidates in the box; only candidates that already satisfy the
              input-only constraints are evaluated and charged. HEA candidates are drawn
              exactly on the simplex face of a random 5-element support (Dirichlet(1)),
              so the simplex and cardinality hold by construction.
  slsqp       multi-start SciPy SLSQP (float64): from each random start, the nearest point
              (span-normalized) satisfying the specification as an equality and every other
              constraint; gradients by finite differences, all evaluations charged. HEA
              starts on a random 5-element support and optimizes that support only.

Optimization (tasks "single", "bi"):
  ga, de, pso pymoo, population 100, termination at the budget. The specification is the
              constraint |f - c|/|c| - eps <= 0 (eps = 1e-4); other constraints are pymoo
              inequality constraints (feasibility-first). HEA candidates are projected onto
              the simplex within the box on their 5 largest fractions before evaluation.
  cmaes       pymoo CMA-ES with IPOP restarts; constraints as a static penalty.
  slsqp       multi-start SLSQP minimizing the objective under all constraints (single).
  nsga2       pymoo NSGA-II, population 100, same constraint handling (bi).

Outputs: out/baselines/<case>_<task>_<method>_s<seed>.csv (inputs + outputs of every
returned point) and one line per run in out/baselines_runs_<case>.csv.
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize as scipy_minimize

import problems as P

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
EPS = 1e-4
CARDINALITY = 5
# The library repair accepts a point when the specification holds to this relative tolerance
# and stops there; the SLSQP baseline is given the same rule so that neither pays for accuracy
# the specification never asked for (see check_slsqp_tolerance.py).
SLSQP_STOP = 1e-6
METHOD_OFFSET = {"rejection": 1, "slsqp": 2, "ga": 3, "de": 4, "pso": 5, "cmaes": 6, "nsga2": 7,
                 "slsqp_mo": 8, "slsqp_round": 9, "enumeration": 10}


# ----------------------------------------------------------------------------- helpers
class Budget(Exception):
    pass


class StopEarly(Exception):
    """Raised when an SLSQP iterate already meets the requested tolerance."""

    def __init__(self, w):
        self.w = w


class Space:
    """Free coordinates of a problem, with the pinned inputs filled back in."""

    def __init__(self, p: P.Problem):
        self.p = p
        self.fixed = {}
        for c in p.constraints:
            if c["condition"] == "Equal" and c["expression"] in p.inputs:
                self.fixed[p.inputs.index(c["expression"])] = c["values"][0]
        self.free = np.array([j for j in range(len(p.inputs)) if j not in self.fixed])
        lower, upper = p.lower.copy(), p.upper.copy()
        for c in p.constraints:  # bounds on a single input narrow the box, as in the library
            if c["expression"] in p.inputs and c["condition"] in ("Between", "GreaterEqual", "LessEqual"):
                j = p.inputs.index(c["expression"])
                if c["condition"] in ("Between", "GreaterEqual"):
                    lower[j] = max(lower[j], c["values"][0])
                if c["condition"] in ("Between", "LessEqual"):
                    upper[j] = min(upper[j], c["values"][-1])
        self.lower, self.upper = lower[self.free], upper[self.free]
        self.simplex = any(c["condition"] == "Equal" and "+" in c["expression"] and c["values"][0] == 1.0
                           for c in p.constraints if c["expression"] not in p.inputs)
        # Coordinates restricted to a finite catalogue, indexed in the free frame.
        position = {j: k for k, j in enumerate(self.free)}
        self.catalogues = {position[j]: v for j, v in p.catalogues().items() if j in position}
        self.discrete = bool(self.catalogues)

    def full(self, z: np.ndarray) -> np.ndarray:
        z = np.atleast_2d(z)
        x = np.empty((len(z), len(self.p.inputs)))
        x[:, self.free] = z
        for j, v in self.fixed.items():
            x[:, j] = v
        return x

    def snap(self, z: np.ndarray) -> np.ndarray:
        """Nearest catalogue value for every restricted coordinate; the others are left alone."""
        z = np.atleast_2d(z).astype(float).copy()
        for k, values in self.catalogues.items():
            z[:, k] = values[np.abs(z[:, k, None] - values[None, :]).argmin(axis=1)]
        return z

    def draw(self, rng, n: int) -> np.ndarray:
        """Uniform points: catalogue values where a coordinate is restricted, the interval else."""
        z = rng.uniform(self.lower, self.upper, size=(n, len(self.free)))
        for k, values in self.catalogues.items():
            z[:, k] = rng.choice(values, size=n)
        return z

    def grid(self) -> np.ndarray:
        """Every point of the catalogue (only for a space whose coordinates are all restricted)."""
        assert len(self.catalogues) == len(self.free), "the space has continuous coordinates"
        axes = [self.catalogues[k] for k in range(len(self.free))]
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, len(axes))


def project_simplex_box(v: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Euclidean projection of v onto {sum x = 1, lower <= x <= upper} (bisection on the shift)."""
    lo, hi = np.min(v - upper) - 1.0, np.max(v - lower) + 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if np.clip(v - mid, lower, upper).sum() > 1.0:
            lo = mid
        else:
            hi = mid
    return np.clip(v - 0.5 * (lo + hi), lower, upper)


def hea_repair(z: np.ndarray, space: Space) -> np.ndarray:
    """Keep the CARDINALITY largest fractions of each row, project them onto the simplex within
    the box (row-wise bisection on the shift, vectorized over rows)."""
    z = np.atleast_2d(z).astype(float)
    n = len(z)
    keep = np.zeros_like(z, dtype=bool)
    keep[np.arange(n)[:, None], np.argsort(-z, axis=1)[:, :CARDINALITY]] = True
    lower = np.where(keep, space.lower, 0.0)
    upper = np.where(keep, space.upper, 0.0)
    lo = np.min(z - upper, axis=1) - 1.0
    hi = np.max(z - lower, axis=1) + 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        over = np.clip(z - mid[:, None], lower, upper).sum(axis=1) > 1.0
        lo = np.where(over, mid, lo)
        hi = np.where(over, hi, mid)
    return np.clip(z - (0.5 * (lo + hi))[:, None], lower, upper)


def random_support_point(rng, space: Space) -> tuple[np.ndarray, np.ndarray]:
    support = rng.choice(len(space.free), CARDINALITY, replace=False)
    z = np.zeros(len(space.free))
    z[support] = rng.dirichlet(np.ones(CARDINALITY))
    return z, support


class Evaluator:
    """Budgeted, cached access to the surrogate for one run."""

    def __init__(self, p: P.Problem, budget: int):
        self.p, self.budget = p, budget
        self.start = p.surrogate.evaluations
        self.cache = {}

    @property
    def used(self) -> int:
        return self.p.surrogate.evaluations - self.start

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        if len(x) == 1:
            key = x.tobytes()
            if key in self.cache:
                return self.cache[key]
        if self.used + len(x) > self.budget:
            raise Budget()
        y = self.p.evaluate(x)
        if len(x) == 1:
            if len(self.cache) > 4096:
                self.cache.clear()
            self.cache[key] = y
        return y


def input_only(c) -> bool:
    return c["condition"] != "Cardinality"


def constraint_rows(p: P.Problem, x: np.ndarray, y: np.ndarray, include_simplex: bool) -> np.ndarray:
    """Normalized inequality constraints g <= 0 for the non-specification constraints."""
    cols = []
    for c in p.constraints:
        cond, vals = c["condition"], c["values"]
        if cond == "Cardinality" or (cond == "Equal" and c["expression"] in p.inputs):
            continue
        if cond in ("AllowedSet", "Integer"):
            continue  # a catalogue is met by rounding onto it, not by an inequality
        v = p.value(c["expression"], x, y)
        if cond == "Equal":
            if include_simplex:
                s = max(abs(vals[0]), 1.0)
                cols += [(v - vals[0]) / s - 1e-6, (vals[0] - v) / s - 1e-6]
        elif cond == "Between":
            s = max(abs(vals[0]), abs(vals[1]), 1.0)
            cols += [(vals[0] - v) / s, (v - vals[1]) / s]
        elif cond in ("LessEqual", "Less"):
            cols.append((v - vals[0]) / max(abs(vals[0]), 1e-12))
        elif cond in ("GreaterEqual", "Greater"):
            cols.append((vals[0] - v) / max(abs(vals[0]), 1e-12))
    return np.column_stack(cols) if cols else np.zeros((len(x), 0))


def spec_value(p: P.Problem, y: np.ndarray) -> np.ndarray:
    """Deviation from the specification, relative to the target and in the target's own units.

    A logarithmic specification is converted back first, so that the number means the same
    thing as the residual the analysis reports and as the band quoted in the problem."""
    value = y[:, p.outputs.index(p.specification)]
    target = p.target
    if p.specification.startswith("log10_"):
        value, target = np.power(10.0, value), 10.0 ** target
    return (value - target) / abs(target)


def spec_window(p: P.Problem) -> float:
    """Half-width the methods must bring the specification within: the band when the problem
    states one, and the relaxation the population methods are given otherwise."""
    return p.band if p.band > 0 else EPS


def budget_for(case: str, task: str, seed: int) -> int:
    runs = pd.concat([pd.read_csv(f) for f in OUT.glob(f"runs_{case}.csv")])
    mode = "sample" if task == "sample" else "solve"
    sel = runs[(runs["mode"] == mode) & (runs["seed"] == seed)]
    if mode == "solve":
        sel = sel[(sel["task"] == task) & (sel["formulation"] != "objective-matched")]
    if sel.empty:
        raise RuntimeError(f"no OpenNN run for {case} {task} seed {seed}")
    return int(sel["forwards"].max())


def save(case, task, method, seed, x, y, budget, used, seconds, note=""):
    folder = OUT / "baselines"
    folder.mkdir(parents=True, exist_ok=True)
    p = P.load(case) if not hasattr(save, "p") or save.p.name != case else save.p
    save.p = p
    pd.DataFrame(np.column_stack([x, y]) if len(x) else np.zeros((0, len(p.inputs) + len(p.outputs))),
                 columns=p.inputs + p.outputs).to_csv(folder / f"{case}_{task}_{method}_s{seed}.csv", index=False)
    log = OUT / f"baselines_runs_{case}.csv"
    fresh = not log.exists()
    with open(log, "a") as f:
        if fresh:
            f.write("case,task,method,seed,budget,evaluations,rows,seconds,note\n")
        f.write(f"{case},{task},{method},{seed},{budget},{used},{len(x)},{seconds:.3f},{note}\n")
    print(f"{case} {task} {method} s{seed}: {len(x)} rows, {used}/{budget} evaluations, {seconds:.1f} s {note}")


# ----------------------------------------------------------------------------- set recovery
def rejection(p, space, budget, rng):
    ev = Evaluator(p, budget)
    input_only_rows = [k for k, c in enumerate(p.constraints)
                       if c["condition"] != "Cardinality" and not any(n in c["expression"] for n in p.outputs)]
    xs, ys = [], []
    batch = 4096
    try:
        while ev.used < ev.budget:
            if space.simplex:
                # batch of random_support_point draws: a random 5-subset, Dirichlet(1) weights on it
                support = np.argsort(rng.random((batch, len(space.free))), axis=1)[:, :CARDINALITY]
                z = np.zeros((batch, len(space.free)))
                w = rng.exponential(size=(batch, CARDINALITY))
                z[np.arange(batch)[:, None], support] = w / w.sum(axis=1, keepdims=True)
                ok = np.all((z >= space.lower - 1e-12) & (z <= space.upper + 1e-12), axis=1)
            else:
                # On a catalogue the uniform draw is over the catalogue, which is what a random
                # search of a discrete space does; elsewhere it is uniform in the box.
                z = space.draw(rng, batch)
                viol = p.violations(space.full(z), np.zeros((batch, len(p.outputs))))
                ok = (viol[:, input_only_rows] <= 0).all(axis=1)
            z = z[ok][:ev.budget - ev.used]
            if len(z) == 0:
                continue
            x = space.full(z)
            y = ev(x)
            keep = p.feasible_except_spec(x, y) & (p.spec_residual(x, y) <= max(1e-2, p.band))
            xs.append(x[keep]); ys.append(y[keep])
    except Budget:
        pass
    x = np.vstack(xs) if xs else np.zeros((0, len(p.inputs)))
    y = np.vstack(ys) if ys else np.zeros((0, len(p.outputs)))
    return x, y, ev.used

def objective_scale(p, objectives):
    """Range of each objective over the measured table, to put a weighted sum on one scale.

    Taken from the data and not from the surrogate, so it costs no evaluations."""
    table = pd.read_csv(HERE / "data" / f"{p.name}.csv")
    x, y = table[p.inputs].to_numpy(float), table[p.outputs].to_numpy(float)
    g = p.objective_values(x, y, objectives)
    return np.maximum(g.max(0) - g.min(0), 1e-12)


def slsqp_nearest(p, space, budget, rng, objective=None, stop_tolerance=None, scalarize=False,
                  snap=False):
    """Multi-start SLSQP. objective None: nearest feasible point to the start; otherwise
    minimize the (minimization-form) objective. Returns every converged feasible point.

    scalarize turns a multiobjective task into one weighted sum per start, with the weights
    drawn uniformly from the simplex and each objective divided by its range in the data. It
    is the local-solver counterpart of NSGA-II: every start returns one point that satisfies
    the specification as an equality, and the front is the non-dominated subset of them.

    snap makes this the relax-and-round baseline of a catalogue problem: the solve runs on the
    continuous relaxation, which is the only thing a gradient method can do when the design
    variables are restricted to finite sets, and the solution is then rounded onto the
    catalogue and re-evaluated (charged). Rounded points are kept whatever their residual,
    because how far the rounding throws the specification is the measurement of interest; the
    relaxed points are returned separately so the two can be compared.

    Values and forward-difference Jacobians come from one batched surrogate call per point
    (1 row for the values, n more rows for the Jacobian), all charged to the budget.

    stop_tolerance gives the local solve the same acceptance rule as the library repair: the
    iteration stops as soon as the point satisfies the specification to that relative
    tolerance and the other constraints. Without it SLSQP runs to its own convergence and
    reaches a residual of about 1e-14, which is accuracy the specification never asked for and
    which the budget then pays for. It applies to the set-recovery task only; with an
    objective, stopping at the first feasible point would abandon the optimization."""
    ev = Evaluator(p, budget)
    span = np.maximum(space.upper - space.lower, 1e-12)
    xs, ys, relaxed = [], [], []
    scale = objective_scale(p, objective) if (objective is not None and scalarize) else None
    weight = np.ones(len(objective)) if scalarize else None

    def quantities(x, y):
        spec = spec_value(p, y)[:, None]
        cons = -constraint_rows(p, x, y, include_simplex=False)
        if objective is None:
            obj = np.zeros((len(x), 1))
        elif scalarize:
            obj = (p.objective_values(x, y, objective) / scale) @ weight
            obj = obj[:, None]
        else:
            obj = p.objective_values(x, y, objective)[:, :1]
        return spec, cons, obj

    try:
        while True:
            if scalarize:
                weight[:] = rng.dirichlet(np.ones(len(objective)))
            if space.simplex:
                z0, support = random_support_point(rng, space)
            else:
                z0, support = rng.uniform(space.lower, space.upper), np.arange(len(space.free))
            active = np.zeros(len(space.free), bool)
            active[support] = True
            h = 1e-7 * span[active]
            memo = {}

            def unpack(w):
                z = np.zeros(len(space.free)) if space.simplex else z0.copy()
                z[active] = w
                return z

            def at(w, jacobian=False):
                key = w.tobytes()
                entry = memo.get(key)
                if entry is None:
                    x = space.full(unpack(w))
                    entry = memo[key] = {"v": quantities(x, ev(x))}
                    if stop_tolerance is not None and objective is None:
                        spec, cons = entry["v"][0][0, 0], entry["v"][1][0]
                        window = max(stop_tolerance, p.band)   # a band asks for no more than itself
                        if abs(spec) <= window and (cons.size == 0 or cons.min() >= 0.0):
                            raise StopEarly(w.copy())
                if jacobian and "j" not in entry:
                    W = w + np.diag(h)
                    X = space.full(np.array([unpack(r) for r in W]))
                    V = quantities(X, ev(X))
                    entry["j"] = tuple((Vk - vk) / h[:, None] for Vk, vk in zip(V, entry["v"]))
                if len(memo) > 64:
                    memo.pop(next(iter(memo)))
                return entry

            if objective is None:
                fun = lambda w: float(np.sum(((w - z0[active]) / span[active]) ** 2))
                grad = lambda w: 2 * (w - z0[active]) / span[active] ** 2
            else:
                fun = lambda w: float(at(w)["v"][2][0, 0])
                grad = lambda w: at(w, True)["j"][2][:, 0]
            bounds = list(zip(space.lower[active], space.upper[active]))
            try:
                if p.band > 0:
                    # A band is an inequality pair, not an equality: the solve is asked to bring
                    # the response inside the window, which is all the statement requires.
                    cons = [{"type": "ineq",
                             "fun": lambda w: np.concatenate([p.band - at(w)["v"][0][0],
                                                              p.band + at(w)["v"][0][0]]),
                             "jac": lambda w: np.vstack([-at(w, True)["j"][0].T,
                                                         at(w, True)["j"][0].T])}]
                else:
                    cons = [{"type": "eq", "fun": lambda w: at(w)["v"][0][0],
                             "jac": lambda w: at(w, True)["j"][0].T}]
                if at(z0[active])["v"][1].shape[1]:
                    cons.append({"type": "ineq", "fun": lambda w: at(w)["v"][1][0],
                                 "jac": lambda w: at(w, True)["j"][1].T})
                if space.simplex:
                    cons.append({"type": "eq", "fun": lambda w: np.array([np.sum(w) - 1.0]),
                                 "jac": lambda w: np.ones((1, len(w)))})
                solution = scipy_minimize(fun, z0[active], jac=grad, method="SLSQP", bounds=bounds,
                                          constraints=cons, options={"ftol": 1e-12, "maxiter": 200}).x
            except StopEarly as early:
                # The start, or an iterate, already meets the requested tolerance: the library
                # repair would accept it here too, so the solve stops and pays nothing more.
                solution = early.w
            z = np.clip(unpack(solution), space.lower, space.upper)
            x = space.full(z)
            y = p.evaluate(x)  # re-check (not charged: SLSQP already evaluated its final iterate)
            p.surrogate.evaluations -= 1
            if snap:
                relaxed.append((x[0], y[0]))
                x = space.full(space.snap(z))
                y = ev(x)                       # a genuinely new point: charged
                if p.feasible_except_spec(x, y)[0]:
                    xs.append(x[0]); ys.append(y[0])
            elif p.feasible_except_spec(x, y)[0] and p.spec_residual(x, y)[0] <= max(1e-2, p.band):
                xs.append(x[0]); ys.append(y[0])
    except Budget:
        pass
    x = np.array(xs) if xs else np.zeros((0, len(p.inputs)))
    y = np.array(ys) if ys else np.zeros((0, len(p.outputs)))
    if snap:
        r = (np.array([a for a, _ in relaxed]), np.array([b for _, b in relaxed])) if relaxed else \
            (np.zeros((0, len(p.inputs))), np.zeros((0, len(p.outputs))))
        return x, y, ev.used, r
    return x, y, ev.used

# ----------------------------------------------------------------------------- pymoo
def pymoo_run(p, space, budget, seed, method, objectives):
    from pymoo.core.problem import Problem as PymooProblem
    from pymoo.core.repair import Repair
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    penalty = method == "cmaes"
    ev = Evaluator(p, budget)
    n_obj = len(objectives)

    def prepare(z):
        if space.simplex:
            return hea_repair(z, space)
        # A catalogue problem: the real-coded operators propose off-grid points, which are
        # rounded onto the catalogue before they are evaluated. This is how these algorithms
        # are run on a discrete space, and it keeps every evaluation a legal configuration.
        return space.snap(z) if space.discrete else z

    class Inverse(PymooProblem):
        def __init__(self):
            n_ieq = 0 if penalty else None
            probe = space.full(prepare(np.atleast_2d((space.lower + space.upper) / 2)))
            m = constraint_rows(p, probe, p.evaluate(probe), include_simplex=False).shape[1] + 1
            p.surrogate.evaluations -= 1
            super().__init__(n_var=len(space.free), n_obj=n_obj, n_ieq_constr=0 if penalty else m,
                             xl=space.lower, xu=space.upper)

        def _evaluate(self, z, out, *args, **kwargs):
            z = prepare(z)
            x = space.full(z)
            room = ev.budget - ev.used
            y = np.full((len(x), len(p.outputs)), np.nan)
            n = max(0, min(len(x), room))
            if n:
                y[:n] = ev(x[:n])
            if n < len(x):  # beyond the budget: the rows are never evaluated
                y[n:] = y[0] if n else 0.0
            F = p.objective_values(x, y, objectives)
            G = np.column_stack([np.abs(spec_value(p, y)) - spec_window(p),
                                 constraint_rows(p, x, y, include_simplex=False)])
            if n < len(x):
                F[n:] = 1e12
                G[n:] = 1e12
            if penalty:
                out["F"] = F + 1e3 * np.maximum(G, 0).sum(axis=1, keepdims=True)
            else:
                out["F"], out["G"] = F, G

    if method == "ga":
        from pymoo.algorithms.soo.nonconvex.ga import GA
        algorithm = GA(pop_size=100)
    elif method == "de":
        from pymoo.algorithms.soo.nonconvex.de import DE
        algorithm = DE(pop_size=100)
    elif method == "pso":
        from pymoo.algorithms.soo.nonconvex.pso import PSO
        algorithm = PSO(pop_size=100)
    elif method == "cmaes":
        from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
        algorithm = CMAES(sigma=0.3, restarts=10**6, incpopsize=2, restart_from_best=False)
    elif method == "nsga2":
        from pymoo.algorithms.moo.nsga2 import NSGA2
        algorithm = NSGA2(pop_size=100)
    else:
        raise ValueError(method)

    res = minimize(Inverse(), algorithm, get_termination("n_eval", budget), seed=seed, verbose=False)
    z = np.vstack([np.atleast_2d(res.pop.get("X"))] + ([np.atleast_2d(res.opt.get("X"))] if res.opt is not None else []))
    x = space.full(prepare(z))
    y = p.evaluate(x)
    p.surrogate.evaluations -= len(x)  # re-evaluation of already evaluated points, not charged
    return x, y, ev.used


# ----------------------------------------------------------------------------- exhaustive
def enumeration(p, space, budget):
    """Every legal point of the catalogue, evaluated once: the exact reference for a discrete
    case. Returns all of them with their predictions, so the analysis can read off the exact
    solution set at any tolerance. It is deterministic, hence seed-independent.

    On the kernel case this costs 4680 propagations, fewer than any optimizer's budget. That is
    a property of this catalogue, not of tuning spaces in general: the same authors' GEMM kernel
    has orders of magnitude more configurations. Here it buys a reference nothing else can give."""
    ev = Evaluator(p, max(budget, 10 ** 7))
    z = space.grid()
    x = space.full(z)
    legal = p.violations(x, np.zeros((len(x), len(p.outputs))))
    input_rows = [k for k, c in enumerate(p.constraints)
                  if not any(n in c["expression"] for n in p.outputs)]
    keep = (legal[:, input_rows] <= 0).all(axis=1)
    x = x[keep]
    y = ev(x)
    return x, y, ev.used


# ----------------------------------------------------------------------------- main
BAND_TAG = "-band"   # hyphen, so that a baseline never collides with a library label such as
                     # ga_band (the library's genetic solver with the band formulation)


def offset(method: str) -> int:
    """Seed offset of a method; a band variant gets its own stream."""
    base = method[:-len(BAND_TAG)] if method.endswith(BAND_TAG) else method
    return METHOD_OFFSET[base] + (100 if method.endswith(BAND_TAG) else 0)


def run(case: str, task: str, seeds, statement: str = "equal"):
    """statement 'equal' asks every method for the specification as an equality; 'band' asks for
    the window the problem declares. A catalogue case is run both ways: the first shows that the
    equality is unsatisfiable for every method and not only for the library, the second is the
    comparison that can actually be won."""
    p = P.load(case)
    declared_band = p.band
    if statement == "band" and declared_band <= 0:
        return
    p.band = declared_band if statement == "band" else 0.0
    tag = BAND_TAG if statement == "band" else ""
    space = Space(p)
    # On a catalogue problem the gradient method cannot honour the design variables, so it is
    # replaced by relax-and-round; the exhaustive reference is added once, and the population
    # methods stay, rounding onto the catalogue through prepare().
    local = "slsqp_round" if space.discrete else "slsqp"
    # The enumeration evaluates the whole catalogue and does not depend on how the specification
    # is stated, so it is run once, under the equality statement, and read by both.
    exhaustive = ["enumeration"] if (space.discrete and statement == "equal") else []
    if task == "sample":
        methods = ["rejection", local] + exhaustive
    elif task == "single":
        methods = ["ga", "de", "pso", "cmaes", local] + exhaustive
    else:
        if not p.bi:
            return
        methods = ["nsga2"] + ([] if space.discrete else ["slsqp_mo"])
    objectives = p.single if task == "single" else p.bi

    log = OUT / f"baselines_runs_{case}.csv"
    done = pd.read_csv(log) if log.exists() else pd.DataFrame(columns=["task", "method", "seed"])
    for seed in seeds:
        budget = budget_for(case, task, seed)
        for method in methods:
            logged = method + tag
            if ((done["task"] == task) & (done["method"] == logged) & (done["seed"] == seed)).any():
                continue  # resume: this run is already logged
            rng = np.random.default_rng(1000 * seed + offset(logged))
            t0 = time.perf_counter()
            if method == "rejection":
                x, y, used = rejection(p, space, budget, rng)
            elif method == "slsqp":
                x, y, used = slsqp_nearest(p, space, budget, rng,
                                          None if task == "sample" else objectives,
                                          stop_tolerance=SLSQP_STOP if task == "sample" else None)
            elif method == "slsqp_mo":
                x, y, used = slsqp_nearest(p, space, budget, rng, objectives, scalarize=True)
            elif method == "slsqp_round":
                x, y, used, relaxed = slsqp_nearest(
                    p, space, budget, rng, None if task == "sample" else objectives,
                    stop_tolerance=SLSQP_STOP if task == "sample" else None, snap=True)
                # The same solve also yields the points before rounding, kept for the comparison.
                save(case, task, "slsqp_relaxed" + tag, seed, relaxed[0], relaxed[1], budget, used,
                     time.perf_counter() - t0, note="off-catalogue")
            elif method == "enumeration":
                x, y, used = enumeration(p, space, budget)
            else:
                x, y, used = pymoo_run(p, space, budget, seed, method, objectives)
            save(case, task, logged, seed, x, y, budget, used, time.perf_counter() - t0)


if __name__ == "__main__":
    case, task = sys.argv[1], sys.argv[2]
    if len(sys.argv) > 3:
        a, _, b = sys.argv[3].partition("-")
        seeds = range(int(a), int(b or a) + 1)
    else:
        seeds = range(1, 11)
    for t in (["sample", "single", "bi"] if task == "all" else [task]):
        for statement in ("equal", "band"):
            run(case, t, seeds, statement)
