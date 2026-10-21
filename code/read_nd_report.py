"""Reads the tables of a Neural Designer report (.ndo) written by run_nd.py.

Usage (from code/):  python read_nd_report.py

Prints the architecture Neural Designer chose, its goodness-of-fit parameters and its testing
errors, so the study can quote them and compare them with its own evaluation of the same
network on the same split.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = ("concrete", "steel", "kernel")


def tables(report):
    for task in report.get("Tasks", []):
        for sub in task.get("Subtasks", []):
            if sub.get("Type") == "Table":
                yield task.get("Name", "?"), sub


def show(title, table, limit=8):
    names = str(table.get("RowNames", "")).split(";")
    data = str(table.get("Data", "")).split(";")
    print(f"  [{title}] {table.get('Title', '')}")
    for name, value in list(zip(names, data))[:limit]:
        print(f"     {name.strip():28} {value.strip()}")


def main():
    for case in CASES:
        report = json.loads((HERE / "nd" / case / f"{case}.ndo").read_text(encoding="utf-8"))
        model = json.loads((HERE / "nd" / case / f"{case}.ndm").read_text(encoding="utf-8"))
        print(f"=== {case}")
        for layer in model["Network"]["Layers"]["Items"]:
            for kind, body in layer.items():
                if kind == "Dense":
                    print(f"  dense {body.get('InputDimensions','').strip()} -> "
                          f"{body.get('OutputDimensions','').strip()} ({body.get('Activation')})")
        for task_name, table in tables(report):
            if "goodness" in table.get("Title", "").lower() or "rror" in table.get("Title", ""):
                show(task_name, table)
        print()


if __name__ == "__main__":
    main()
