"""Independent statement and checking of the inverse-design problems.

Reads out/problems.json (written by `inverse_cases problems` from src/cases.h) so that the
Python side states exactly the problems given to OpenNN. Every point any method returns
is re-evaluated here with the Python export of the network (surrogates.py) and checked
constraint by constraint; nothing a solver reports about its own feasibility is trusted.

Tolerances. The specification residual is reported relative to the target,
r = |f(x) - c| / |c|. Other constraints are checked with the slack of the
library's own tests (bound_slack in tests/response_optimization/concrete_fixture.h):
max(1e-2, 1e-3 |bound|) on the expression value; the cardinality condition counts an
input as present when it exceeds 1e-6.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from surrogates import Surrogate

HERE = Path(__file__).resolve().parent
PRESENT = 1e-6
_COMPILED: dict = {}


def slack(bound: float) -> float:
    return max(1e-2, 1e-3 * abs(bound))


@dataclass
class Problem:
    name: str
    specification: str
    target: float
    band: float          # relative half-width of the band statement (0 when the case has none)
    inputs: list
    outputs: list
    lower: np.ndarray
    upper: np.ndarray
    constraints: list
    single: list
    bi: list
    surrogate: Surrogate = field(repr=False, default=None)

    # --- expressions -------------------------------------------------------------
    def env(self, x: np.ndarray, y: np.ndarray) -> dict:
        d = {n: x[:, j] for j, n in enumerate(self.inputs)}
        d.update({n: y[:, j] for j, n in enumerate(self.outputs)})
        return d

    def value(self, expression: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        code = _COMPILED.get(expression)
        if code is None:
            code = _COMPILED[expression] = compile(expression.replace("^", "**"), expression, "eval")
        return np.asarray(eval(code, {"__builtins__": {}, "np": np}, self.env(x, y)), dtype=float) * np.ones(len(x))

    # --- evaluation ---------------------------------------------------------------
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return self.surrogate(x)

    def spec_residual(self, x, y=None) -> np.ndarray:
        """Relative residual of the specification, in the units the specification is quoted in.

        When the network predicts the base-10 logarithm of the quantity (the kernel case, whose
        duration spans 68x), both sides are converted back first, so that the tolerance means the
        same thing in every case: a relative error on the physical quantity."""
        y = self.evaluate(x) if y is None else y
        value = y[:, self.outputs.index(self.specification)]
        target = self.target
        if self.specification.startswith("log10_"):
            value, target = np.power(10.0, value), 10.0 ** target
        return np.abs(value - target) / abs(target)

    def violations(self, x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Violation of every non-specification constraint and of the box, >0 means violated
        (already net of the checking slack). Returns an (n, m) array."""
        y = self.evaluate(x) if y is None else y
        cols = []
        for c in self.constraints:
            cond, vals = c["condition"], c["values"]
            if cond == "Cardinality":
                members = [n.strip() for n in c["expression"].split(";")]
                count = sum((np.abs(x[:, self.inputs.index(m)]) > PRESENT).astype(int) for m in members)
                cols.append(np.maximum(0, count - vals[0]))
                continue
            if cond in ("AllowedSet", "Integer"):
                # The value has to sit on the catalogue (or on an integer). The slack is far
                # below the spacing of every catalogue used here, so a wrong level cannot pass.
                v = self.value(c["expression"], x, y)
                allowed = np.asarray(vals, float) if cond == "AllowedSet" else np.round(v)
                distance = (np.abs(v[:, None] - allowed[None, :]).min(axis=1)
                            if cond == "AllowedSet" else np.abs(v - allowed))
                cols.append(np.maximum(0, distance - slack(0.0)))
                continue
            v = self.value(c["expression"], x, y)
            if cond == "Equal":
                cols.append(np.maximum(0, np.abs(v - vals[0]) - slack(vals[0])))
            elif cond == "Between":
                cols.append(np.maximum(0, np.maximum(vals[0] - v - slack(vals[0]), v - vals[1] - slack(vals[1]))))
            elif cond in ("LessEqual", "Less"):
                cols.append(np.maximum(0, v - vals[0] - slack(vals[0])))
            elif cond in ("GreaterEqual", "Greater"):
                cols.append(np.maximum(0, vals[0] - v - slack(vals[0])))
            else:
                raise ValueError(cond)
        span = np.maximum(self.upper - self.lower, 1e-12)
        box = np.maximum(0, np.maximum(self.lower - x, x - self.upper) - 1e-6 * span).max(axis=1)
        cols.append(box)
        return np.column_stack(cols)

    def feasible_except_spec(self, x, y=None) -> np.ndarray:
        return (self.violations(x, y) <= 0).all(axis=1)

    def objective_values(self, x, y, objectives) -> np.ndarray:
        """Objective values in minimization form (Maximize negated)."""
        return np.column_stack([
            (-1 if o["sense"] == "Maximize" else 1) * self.value(o["expression"], x, y) for o in objectives])

    # --- geometry -------------------------------------------------------------------
    def free_inputs(self) -> np.ndarray:
        """Inputs the constraints do not pin to one value."""
        pinned = {c["expression"] for c in self.constraints if c["condition"] == "Equal" and c["expression"] in self.inputs}
        return np.array([n not in pinned for n in self.inputs])

    def catalogues(self) -> dict:
        """{input index: sorted allowed values} for the inputs restricted to a finite set."""
        out = {}
        for c in self.constraints:
            if c["condition"] == "AllowedSet" and c["expression"] in self.inputs:
                out[self.inputs.index(c["expression"])] = np.sort(np.asarray(c["values"], float))
            elif c["condition"] == "Integer" and c["expression"] in self.inputs:
                j = self.inputs.index(c["expression"])
                out[j] = np.arange(np.ceil(self.lower[j]), np.floor(self.upper[j]) + 1)
        return out

    def design_box(self):
        """The box after the bounds on single inputs, as the library narrows it."""
        lower, upper = self.lower.copy(), self.upper.copy()
        for j, values in self.catalogues().items():
            # calculate_domain() narrows an AllowedSet input to the ends of its catalogue.
            lower[j], upper[j] = max(lower[j], values[0]), min(upper[j], values[-1])
        for c in self.constraints:
            if c["expression"] in self.inputs and c["condition"] in ("Between", "GreaterEqual", "LessEqual"):
                j = self.inputs.index(c["expression"])
                if c["condition"] in ("Between", "GreaterEqual"):
                    lower[j] = max(lower[j], c["values"][0])
                if c["condition"] in ("Between", "LessEqual"):
                    upper[j] = min(upper[j], c["values"][-1])
        return lower, upper

    def normalize(self, x: np.ndarray) -> np.ndarray:
        lower, upper = self.design_box()
        span = np.maximum(upper - lower, 1e-12)
        return ((x - lower) / span)[:, self.free_inputs()]


def load(name: str) -> Problem:
    spec = json.loads((HERE / "out" / "problems.json").read_text())[name]
    p = Problem(name=name, specification=spec["specification"], target=float(spec["target"]),
                band=float(spec.get("band", 0.0)),
                inputs=spec["inputs"], outputs=spec["outputs"], lower=np.array(spec["lower"], float),
                upper=np.array(spec["upper"], float), constraints=spec["constraints"],
                single=spec["single"], bi=spec["bi"])
    p.surrogate = Surrogate(name)
    return p


def read_rows(path: Path, p: Problem):
    table = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if table.size == 0:
        return np.zeros((0, len(p.inputs))), np.zeros((0, len(p.outputs)))
    return table[:, :len(p.inputs)], table[:, len(p.inputs):]
