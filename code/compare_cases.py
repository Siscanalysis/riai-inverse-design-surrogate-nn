"""Prints the per-case comparison of every task, from out/summary.json (reading aid, not a paper artifact)."""

import json
from pathlib import Path

S = json.loads((Path(__file__).resolve().parent / "out" / "summary.json").read_text())
CASES = [("concrete", "HORMIGON"), ("steel", "ACERO")]


def med(section, case, method, key, nd=2):
    entry = S[section].get(case, {}).get(method)
    if not entry or not entry.get(key):
        return "-"
    value = entry[key]["median"]
    return format(value, ",.0f") if abs(value) >= 1000 else format(value, "." + str(nd) + "f")


print("=== 1) SET RECOVERY (10 seeds, same budget)")
print(f"{'case':10} {'method':10} {'valid@1e-5':>11} {'evals/sol':>10} {'coverage':>9} {'spread':>8} {'near data':>10}")
for case, name in CASES:
    for method in ("repair", "slsqp", "rejection"):
        print(f"{name:10} {method:10} {med('sample', case, method, 'valid_1e-05'):>11} "
              f"{med('sample', case, method, 'evals_per_sol'):>10} {med('sample', case, method, 'coverage'):>9} "
              f"{med('sample', case, method, 'spread'):>8} {med('sample', case, method, 'near_data'):>10}")
    print()

print("=== 2) SCALAR TASK (best valid value at 1e-4 | seeds valid at 1e-5)")
print(f"{'method':22}" + "".join(f"{n:>22}" for _, n in CASES))
for method in ("dc_constraint", "ga_constraint", "dc_objective", "ga_objective",
               "dc_objective-matched", "ga_objective-matched", "slsqp", "ga", "de", "pso", "cmaes"):
    row = f"{method:22}"
    for case, _ in CASES:
        entry = S["single"].get(case, {}).get(method)
        cell = "-" if not entry else f"{med('single', case, method, 'best_0.0001', 3)} ({entry.get('success_1e-5', 0)}/10)"
        row += f"{cell:>22}"
    print(row)
print("\n  objectives: concrete = MIN cement [kg/m3] | HEA = MIN density [g/cm3] | steel = MAX elongation [%]")

print("\n=== 3) TWO-OBJECTIVE FRONT (valid points | hypervolume)")
print(f"{'method':22}" + "".join(f"{n:>20}" for _, n in CASES))
for method in ("dc_constraint", "ga_constraint", "dc_objective", "ga_objective",
               "dc_objective-matched", "ga_objective-matched", "nsga2"):
    row = f"{method:22}"
    for case, _ in CASES:
        entry = S["bi"].get(case, {}).get(method)
        cell = "-" if not entry else f"{med('bi', case, method, 'front')} | {med('bi', case, method, 'hv')}"
        row += f"{cell:>20}"
    print(row)
