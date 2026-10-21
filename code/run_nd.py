"""Runs the Neural Designer engine, headless, over a project written by make_nd_project.py.

Usage (from code/):  python run_nd.py <concrete|steel> [task ...]

Each task is one engine call, `neuralengine <project>.ndm <task>.ndt MIGRATION`, the same way
Neural Designer's own integration tests drive it. The default sequence imports the data, lets
GrowingNeurons choose the hidden layer, trains, and runs the goodness-of-fit analysis whose
charts go into the .ndo report (open it in Neural Designer to export them).

Nothing in the Neural Designer tree is written to; the project folder here receives the model,
its parameters and the .ndo report.

This script is provenance, not a reproduction path. The engine it calls is a local build whose
version this repository does not record, and Neural Designer has changed since the surrogates were
trained, so a rerun is not expected to reproduce them bit for bit. The networks the study actually
uses are the OpenNN model files in models/, which are byte-identical to the imports this produced.
Point RAI_ND_ENGINE at another neuralengine.exe to use a different build.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ENGINE = Path(os.environ.get(
    "RAI_ND_ENGINE",
    r"C:\Users\Artelnics\Documents\neuraldesigner-build\mingw-release\bin\neuralengine.exe"))
LICENSE = "MIGRATION"
DEFAULT_TASKS = [
    ("DataSetTask", "Import data"),
    ("ModelSelectionTask", "Perform neuron selection"),
    ("TrainingStrategyTask", "Perform training"),
    ("TestingAnalysisTask", "Perform goodness-of-fit analysis"),
    ("TestingAnalysisTask", "Calculate errors"),
]


ROLE_CODE = {"Training": 0, "Validation": 1, "Testing": 2}


def pin_roles(case: str) -> None:
    """Puts the study's grouped split back into the project.

    Importing the data makes Neural Designer assign its own random 60/20/20 roles, which would
    put rows of the same mix (7 and 28 days) or the same alloy (several temperatures) on both
    sides of the split. Every task after this one reads the roles from the project file."""
    project = HERE / "nd" / case / f"{case}.ndm"
    model = json.loads(project.read_text(encoding="utf-8"))
    roles = pd.read_csv(HERE / "data" / f"{case}_split.csv")["role"].map(ROLE_CODE)
    model["Dataset"]["Samples"]["SampleRoles"] = " ".join(str(r) for r in roles)
    project.write_text(json.dumps(model, indent=4), encoding="utf-8")
    print(f"{case:9} {'pinned grouped split':28} {roles.value_counts().sort_index().to_dict()}", flush=True)


def run(case: str, tasks=None):
    folder = HERE / "nd" / case
    project = folder / f"{case}.ndm"
    if not project.exists():
        raise SystemExit(f"{project} not found; run make_nd_project.py first")

    for component, name in (tasks or DEFAULT_TASKS):
        task_file = folder / "_task.ndt"
        task_file.write_text(json.dumps({"Tasks": [{"Component": component, "Name": name}]}), encoding="utf-8")
        start = time.perf_counter()
        result = subprocess.run([str(ENGINE), str(project), str(task_file), LICENSE],
                                cwd=folder, capture_output=True, text=True, timeout=7200)
        output = (result.stdout + result.stderr).strip().splitlines()
        tail = " | ".join(line.strip() for line in output[-3:])
        print(f"{case:9} {name:28} rc={result.returncode} {time.perf_counter()-start:7.1f} s  {tail}", flush=True)
        if name == "Import data":
            pin_roles(case)
        if result.returncode != 0:
            for line in output[-15:]:
                print("    " + line)
            raise SystemExit(f"engine failed on '{name}'")


if __name__ == "__main__":
    case = sys.argv[1]
    names = sys.argv[2:]
    selected = [t for t in DEFAULT_TASKS if not names or t[1] in names]
    run(case, selected)
