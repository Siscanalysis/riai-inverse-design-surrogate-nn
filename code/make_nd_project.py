"""Builds a Neural Designer project (.ndm) for a case, from the same data and split as the study.

Usage (from code/):  python make_nd_project.py <concrete|steel>

Neural Designer reads a JSON project file plus its data CSV, and its engine can run the tasks
headlessly (see run_nd.py). The project written here carries:

  * the case's numeric table (data/<case>.csv, copied beside the project),
  * the same a priori grouped split as the study, as per-sample roles
    (0 training, 1 validation, 2 testing; prep_data.py wrote them),
  * the same training recipe (QuasiNewton on the mean squared error with an L2 weight of
    1e-3, early stopping on the validation set),
  * GrowingNeurons over 1..32 hidden neurons with five trials each, so Neural Designer chooses
    the architecture with the same effort the study gave its own selection (five seeds).

The structure follows the fixture project shipped with Neural Designer's integration tests,
which is the reference for the current schema. Nothing in the Neural Designer tree is modified.
"""

import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
# A Neural Designer project used as a template to clone. Set RAI_ND_FIXTURE to any .ndm; the
# default is the authors' workstation. Only needed to rebuild the projects under nd/, which are
# provenance rather than a reproduction path (see run_nd.py).
ND_FIXTURE = Path(os.environ.get(
    "RAI_ND_FIXTURE",
    r"C:\Users\Artelnics\Documents\neuraldesigner\neuraldesigner\tests\integration"
    r"\models\airfoilselfnoise\airfoilselfnoise.ndm"))
TARGETS = {"concrete": 2, "steel": 4, "kernel": 2}
ROLE_CODE = {"Training": 0, "Validation": 1, "Testing": 2}
# Upper end of the neuron search. The kernel case is capped lower: its cross-validated score is
# already flat from twelve neurons (0.972 against 0.988 at twenty-four on the specification
# response), so a wider layer would only spend rows per weight for nothing.
MAX_NEURONS = {"concrete": 32, "steel": 32, "kernel": 24}


def build(case: str) -> Path:
    table = pd.read_csv(HERE / "data" / f"{case}.csv")
    roles = pd.read_csv(HERE / "data" / f"{case}_split.csv")["role"].map(ROLE_CODE)
    targets_number = TARGETS[case]
    columns = list(table.columns)

    folder = HERE / "nd" / case
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(HERE / "data" / f"{case}.csv", folder / f"{case}.csv")

    model = json.loads(ND_FIXTURE.read_text(encoding="utf-8"))
    model["Description"] = f"RAI inverse design study, {case} surrogate"
    model["ModelMetaData"]["DescriptionText"] = model["Description"]
    model["ModelMetaData"]["Trained"] = False

    data_source = model["Dataset"]["DataSource"]
    data_source["Path"] = f"{case}.csv"
    data_source["Separator"] = "Comma"
    data_source["HasHeader"] = "1"

    model["Dataset"]["Variables"] = {
        "VariablesNumber": str(len(columns)),
        "Variable": [{"Name": name,
                      "Role": "Input" if i < len(columns) - targets_number else "Target",
                      "Type": "Numeric",
                      "Scaler": "MeanStandardDeviation"} for i, name in enumerate(columns)],
    }
    model["Dataset"]["Samples"] = {"SamplesNumber": str(len(table)),
                                   "SampleRoles": " ".join(str(r) for r in roles)}
    model["Dataset"]["MissingValues"] = {"MissingValuesNumber": "0"}
    preview = [",".join(columns)] + [",".join(f"{v:g}" for v in table.iloc[i]) for i in range(3)]
    model["Dataset"]["PreviewData"] = {"PreviewSize": str(len(preview)),
                                       "Row": [{"Item": str(i + 1), "Text": row}
                                               for i, row in enumerate(preview)]}

    training = model["Training"]
    training["Loss"]["Error"] = "MeanSquaredError"
    training["Loss"]["MeanSquaredError"]["RegularizationWeight"] = "0.001000"
    training["Loss"]["Regularization"]["RegularizationWeight"] = "0.001000"
    training["Optimizer"]["OptimizationMethod"] = "QuasiNewton"
    training["Optimizer"]["QuasiNewton"]["MaximumEpochsNumber"] = "5000"
    training["Optimizer"]["QuasiNewton"]["MaximumValidationFailures"] = "500"

    neurons = model["ModelSelection"]["NeuronSelection"]["GrowingNeurons"]
    neurons["MaximumNeurons"] = str(MAX_NEURONS[case])
    neurons["TrialsNumber"] = "5"

    # The engine rebuilds the network when the data are imported.
    model.pop("Network", None)

    path = folder / f"{case}.ndm"
    path.write_text(json.dumps(model, indent=4), encoding="utf-8")
    print(f"{case}: wrote {path} ({len(table)} samples, {len(columns) - targets_number} inputs, "
          f"{targets_number} targets, roles {roles.value_counts().to_dict()})")
    return path


if __name__ == "__main__":
    for case in (sys.argv[1:] or ["concrete", "steel"]):
        build(case)
