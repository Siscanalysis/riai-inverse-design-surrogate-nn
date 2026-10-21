"""Converts a trained Neural Designer model into the OpenNN files the drivers load.

Usage (from code/):  python import_nd_network.py <concrete|steel|kernel> ...

A Neural Designer project (.ndm) carries, under "Network", the same serialization OpenNN
writes in its own model files, and the weights live in <project>_<data>_Params.bin. So the
conversion is a copy: {"Network": ...} into models/<case>_nd.json and the parameter file
beside it as models/<case>_nd.bin, which is what NeuralNetwork::load expects.

The conversion is only trusted after eval_network reloads the result with the pinned OpenNN
and reproduces Neural Designer's own reported fit.
"""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
TARGETS = {"concrete": 2, "steel": 4, "kernel": 2}


def convert(case: str) -> None:
    folder = HERE / "nd" / case
    model = json.loads((folder / f"{case}.ndm").read_text(encoding="utf-8"))
    network = model.get("Network")
    if not network:
        raise SystemExit(f"{case}: the project has no trained Network section")

    parameters = folder / f"{case}_{case}_Params.bin"
    if not parameters.exists():
        raise SystemExit(f"{case}: {parameters.name} not found")

    # Neural Designer takes the scaling descriptives from the training split and leaves the
    # output clamp open. The study needs the design box to be the range of the data and the
    # predictions bounded by the observed range, so both are set here exactly as
    # train_surrogates does. Only the bounds change: the mean and the standard deviation that
    # the forward pass uses are left untouched.
    table = pd.read_csv(HERE / "data" / f"{case}.csv")
    inputs_number = len(table.columns) - TARGETS[case]
    minimums = table.iloc[:, :inputs_number].min().tolist()
    maximums = table.iloc[:, :inputs_number].max().tolist()
    lower = table.iloc[:, inputs_number:].min().tolist()
    upper = table.iloc[:, inputs_number:].max().tolist()
    for item in network["Layers"]["Items"]:
        if "Scaling" in item:
            item["Scaling"]["Minimums"] = " ".join(f"{v:.9g}" for v in minimums) + " "
            item["Scaling"]["Maximums"] = " ".join(f"{v:.9g}" for v in maximums) + " "
        if "Clamping" in item:
            item["Clamping"]["LowerBounds"] = " ".join(f"{v:.9g}" for v in lower) + " "
            item["Clamping"]["UpperBounds"] = " ".join(f"{v:.9g}" for v in upper) + " "

    target = HERE / "models" / f"{case}_nd.json"
    target.write_text(json.dumps({"Network": network}, indent=1), encoding="utf-8")
    shutil.copy(parameters, HERE / "models" / f"{case}_nd.bin")
    print(f"{case}: wrote {target.name} + {target.stem}.bin ({parameters.stat().st_size} bytes of parameters)")


if __name__ == "__main__":
    for case in (sys.argv[1:] or ["concrete", "steel"]):
        convert(case)
