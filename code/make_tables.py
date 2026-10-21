"""LaTeX tables of the manuscript, from out/summary.json and out/kernel_checks.json.

Usage (from code/):  python make_tables.py      (after analyze.py and check_kernel.py)

Writes ../paper/tables/*.tex so that no table entry is typed by hand; the prose quotes the same
values and audit_numbers.py checks them. Values are medians over the 10 seeds.

The tables follow the structure of the study. The two continuous cases answer how the
specification should be STATED, so they carry the two formulations of the library and the naive
alternative, rejection sampling, and no external optimizer: there the solver is the same in
either arm and would not be the variable under test. The catalogue case answers what happens
when the design variables rule a gradient method out as a direct method, so that is where the
external baselines belong. The exhaustive enumeration of the catalogue appears as the exact
reference in the text, never as a competing method: the case was chosen because its catalogue
can be enumerated, which is what makes the solution set known.
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "out" / "summary.json").read_text())
K = json.loads((HERE / "out" / "kernel_checks.json").read_text()) if \
    (HERE / "out" / "kernel_checks.json").exists() else {}
TABLES = HERE.parent / "paper" / "tables"
CONTINUOUS = [("concrete", "Hormigón"), ("steel", "Acero")]


def med(entry, key):
    v = entry.get(key) if entry else None
    return None if v is None else v["median"]


def fmt_int(x):
    """Counts, with one decimal kept below ten so that a median of 3.5 is not quoted as 4."""
    if x is None or not np.isfinite(x):
        return "-"
    if x < 10 and x != int(x):
        return f"{x:.1f}"
    return f"{x:,.0f}".replace(",", "\\,")


def fmt_evals(x):
    if x is None or not np.isfinite(x):
        return "-"
    return f"{x/1e3:.0f}\\,k" if x < 1e6 else f"{x/1e6:.2f}\\,M"


def fmt(x, nd=2):
    return "-" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def write(name, lines):
    (TABLES / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {name}")


# ----------------------------------------------------------------- continuous: set recovery
def tab_sets():
    names = {"repair": "Restricción", "rejection": "Rechazo"}
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Caso & Planteamiento & Evals. & \multicolumn{3}{c}{Válidas a $\tau$} "
             r"& Evals./sol. & Cobert. \\",
             r" & & & $10^{-3}$ & $10^{-5}$ & $10^{-7}$ & ($10^{-5}$) & \\", r"\midrule"]
    for case, title in CONTINUOUS:
        block = S["sample"].get(case, {})
        first = True
        for m in ("repair", "rejection"):
            e = block.get(m)
            if not e:
                continue
            lines.append(" & ".join([
                title if first else "", names[m], fmt_evals(med(e, "evaluations")),
                fmt_int(med(e, "valid_0.001")), fmt_int(med(e, "valid_1e-05")),
                fmt_int(med(e, "valid_1e-07")), fmt_int(med(e, "evals_per_sol")),
                fmt(med(e, "coverage"))]) + r" \\")
            first = False
        lines.append(r"\midrule" if case != CONTINUOUS[-1][0] else r"\bottomrule")
    lines.append(r"\end{tabular}")
    write("tab_sets.tex", lines)


# ----------------------------------------------------------------- continuous: the two statements
LIBRARY = {"dc_constraint": "DC, restricción", "ga_constraint": "GA, restricción",
           "dc_objective": "DC, objetivo", "ga_objective": "GA, objetivo",
           "dc_objective-matched": "DC, objetivo igualado",
           "ga_objective-matched": "GA, objetivo igualado",
           "dc_band": "DC, banda", "ga_band": "GA, banda"}
ORDER_LIBRARY = ["dc_constraint", "ga_constraint", "dc_objective", "ga_objective",
                 "dc_objective-matched", "ga_objective-matched"]
DECIMALS = {"concrete": 1, "steel": 2, "kernel": 4}


def tab_single():
    lines = [r"\begin{tabular}{l" + "rrr" * len(CONTINUOUS) + "}", r"\toprule"]
    lines.append(" & ".join(["Planteamiento"] +
                            [rf"\multicolumn{{3}}{{c}}{{{t}}}" for _, t in CONTINUOUS]) + r" \\")
    lines.append(" & " + " & ".join([r"Evals. & Mejor & $n_{5}$"] * len(CONTINUOUS)) + r" \\")
    lines.append(r"\midrule")
    for m in ORDER_LIBRARY:
        row = [LIBRARY[m]]
        for case, _ in CONTINUOUS:
            e = S["single"].get(case, {}).get(m)
            if not e:
                row += ["-", "-", "-"]
                continue
            ok, strict = e.get("success_1e-4", 0), e.get("success_1e-5", 0)
            cell = fmt(med(e, "best_0.0001"), DECIMALS[case]) + (f"$^{{{ok}}}$" if ok < 10 else "")
            row += [fmt_evals(med(e, "evaluations")), cell, f"{strict}"]
        lines.append(" & ".join(row) + r" \\")
        if m == "ga_constraint":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_single.tex", lines)


def tab_fronts():
    """Front size and how much of it meets the specification, per tolerance.

    The hypervolume alone hides the difference between the two statements: a front whose points
    mostly miss the specification can still score well, because the few that meet it dominate
    the indicator. The usable counts are the quantity a designer chooses from."""
    lines = [r"\begin{tabular}{l" + "rrrrr" * len(CONTINUOUS) + "}", r"\toprule",
             " & ".join(["Planteamiento"] + [rf"\multicolumn{{5}}{{c}}{{{t}}}" for _, t in CONTINUOUS]) + r" \\",
             " & " + " & ".join([r"Dev. & \multicolumn{3}{c}{Útiles a $\tau$} & HV"] * len(CONTINUOUS)) + r" \\",
             " & " + " & ".join([r" & $5\%$ & $1\%$ & $10^{-5}$ & "] * len(CONTINUOUS)) + r" \\",
             r"\midrule"]
    for m in ORDER_LIBRARY:
        row = [LIBRARY[m]]
        for case, _ in CONTINUOUS:
            e = S["bi"].get(case, {}).get(m)
            if not e:
                row += ["-"] * 5
                continue
            row += [fmt_int(med(e, "returned")), fmt_int(med(e, "usable_0.05")),
                    fmt_int(med(e, "usable_0.01")), fmt_int(med(e, "valid_1e-5")),
                    fmt(med(e, "hv"))]
        lines.append(" & ".join(row) + r" \\")
        if m == "ga_constraint":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_fronts.tex", lines)


# ----------------------------------------------------------------- catalogue: the attainable set
def tab_space():
    if not K:
        return
    bands, target = K["band_for_n_configurations"], K["target_us"]
    wanted = [k for k in ("1", "5", "10", "25", "50") if k in bands]

    def science(value):
        exponent = int(np.floor(np.log10(value)))
        return f"${value/10**exponent:.1f}\\cdot 10^{{{exponent}}}$"

    # Transposed: a row per requested number of solutions, so the table fits one column.
    lines = [r"\begin{tabular}{@{}rrr@{}}", r"\toprule",
             r"Configuraciones & \multicolumn{2}{c}{Semiancho mínimo} \\",
             r"buscadas & relativo & [$\mu$s] \\", r"\midrule"]
    lines += [f"{k} & {science(bands[k])} & {bands[k]*target:.1f}" + r" \\" for k in wanted]
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_space.tex", lines)


# ----------------------------------------------------------------- catalogue: the comparison
KERNEL_ROWS = [
    ("dc_constraint", "DC, igualdad", "biblioteca"),
    ("ga_constraint", "GA, igualdad", "biblioteca"),
    ("dc_band", "DC, banda", "biblioteca"),
    ("ga_band", "GA, banda", "biblioteca"),
    ("dc_objective-matched", "DC, objetivo igualado", "biblioteca"),
    ("ga_objective-matched", "GA, objetivo igualado", "biblioteca"),
    ("slsqp_round", "SLSQP redondeado, igualdad", "referencia"),
    ("slsqp_round-band", "SLSQP redondeado, banda", "referencia"),
    ("cmaes-band", "CMA-ES, banda", "referencia"),
    ("de-band", "DE, banda", "referencia"),
    ("ga-band", "GA (pymoo), banda", "referencia"),
    ("pso-band", "PSO, banda", "referencia"),
]


def tab_kernel():
    block = S["single"].get("kernel", {})
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Planteamiento & Evals. & Óptimo en & Mejor \\",
             r" & & $k/10$ semillas & ocupación \\", r"\midrule"]
    group = None
    for key, label, kind in KERNEL_ROWS:
        e = block.get(key)
        if not e:
            continue
        if group is not None and kind != group:
            lines.append(r"\midrule")
        group = kind
        hits = e.get("optimum_best_0.01", 0)
        best = med(e, "best_0.01")
        lines.append(" & ".join([label, fmt_evals(med(e, "evaluations")),
                                 f"{hits}" if best is not None else "0",
                                 fmt(best, 4) if best is not None else "sin solución"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_kernel.tex", lines)


if __name__ == "__main__":
    TABLES.mkdir(parents=True, exist_ok=True)
    tab_sets()
    tab_single()
    tab_fronts()
    tab_space()
    tab_kernel()
