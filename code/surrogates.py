"""Fast float64 access to the OpenNN networks, read from their Python exports.

models/<case>_surrogate.py is the OpenNN ModelExpression export of the trained network
(scaling -> tanh dense layer -> linear layer -> unscaling -> clamp). Evaluated as written it
handles one sample through hundreds of scalar operations, which dominates the cost of the
Python baselines. This module parses the same coefficients into matrices and evaluates a
batch with two matrix products. `check(case)` compares the result with the C++ predictions
written by train_surrogates (models/<case>_predictions.csv); the agreement is at float32
precision (the C++ network runs in single precision).

Every call is counted in `Surrogate.evaluations` (rows evaluated), which is how the Python
baselines are held to the forward-pass budget of the OpenNN runs.
"""

import re
from pathlib import Path

import numpy as np

MODELS = Path(__file__).resolve().parent / "models"
NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _terms(text: str, prefix: str) -> list[tuple[float, str]]:
    return [(float(w), name) for w, name in re.findall(rf"\(({NUM})\*{prefix}(\w+)\)", text)]


class Surrogate:
    def __init__(self, case: str):
        source = (MODELS / f"{case}_surrogate.py").read_text()
        self.input_names = eval(re.search(r"self\.input_names = (\[.*?\])", source).group(1))
        self.inputs_number = len(self.input_names)
        index = {n: j for j, n in enumerate(self.input_names)}

        self.scale = np.zeros(self.inputs_number)
        self.shift = np.zeros(self.inputs_number)
        for name, a, b in re.findall(rf"scaled_(\w+) = \1\*({NUM})({NUM})", source):
            self.scale[index[name]], self.shift[index[name]] = float(a), float(b)

        hidden = re.findall(rf"dense_layer_1_output_(\d+) = self\.Tanh\( ({NUM}) \+ (.*?) \)\n", source)
        self.b1 = np.zeros(len(hidden))
        self.W1 = np.zeros((len(hidden), self.inputs_number))
        for k, bias, body in hidden:
            self.b1[int(k)] = float(bias)
            for w, name in _terms(body, "scaled_"):
                self.W1[int(k), index[name]] = w

        output = re.findall(rf"approximation_layer_output_(\d+) = self\.Identity\( ({NUM}) \+ (.*?) \)\n", source)
        self.b2 = np.zeros(len(output))
        self.W2 = np.zeros((len(output), len(hidden)))
        for k, bias, body in output:
            self.b2[int(k)] = float(bias)
            for w, name in _terms(body, "dense_layer_1_output_"):
                self.W2[int(k), int(name)] = w

        self.out_scale = np.ones(len(output))
        self.out_shift = np.zeros(len(output))
        for k, s, m in re.findall(rf"unscaling_layer_output_(\d+)=approximation_layer_output_\1\*({NUM})({NUM})", source):
            self.out_scale[int(k)], self.out_shift[int(k)] = float(s), float(m)

        self.low = np.full(len(output), -np.inf)
        self.high = np.full(len(output), np.inf)
        for name, lo, k in re.findall(rf"(\w+) = max\(({NUM}), unscaling_layer_output_(\d+)\)", source):
            self.low[int(k)] = float(lo)
            hi = re.search(rf"{name} = min\(({NUM}), {name}\)", source)
            if hi:
                self.high[int(k)] = float(hi.group(1))
        self.evaluations = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        self.evaluations += x.shape[0]
        h = np.tanh((x * self.scale + self.shift) @ self.W1.T + self.b1)
        y = (h @ self.W2.T + self.b2) * self.out_scale + self.out_shift
        return np.clip(y, self.low, self.high)


def check(case: str) -> float:
    table = np.loadtxt(MODELS / f"{case}_predictions.csv", delimiter=",")
    model = Surrogate(case)
    x, y_cpp = table[:, :model.inputs_number], table[:, model.inputs_number:]
    return float(np.max(np.abs(model(x) - y_cpp) / np.maximum(1.0, np.abs(y_cpp))))


if __name__ == "__main__":
    for case in ("concrete", "hea", "steel"):
        print(f"{case}: max relative |python - C++| = {check(case):.2e}")
