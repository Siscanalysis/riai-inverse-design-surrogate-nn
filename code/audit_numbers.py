"""Checks the numbers quoted in the manuscript prose against the stored metrics (audit pass B).

Usage (from code/):  python audit_numbers.py        (after analyze.py, check_kernel.py and
                                                     check_spec_error.py)

Each claim names the manuscript section, the quoted value and where it is recomputed from. Any
mismatch prints as FAIL with both values. Sources:

    out/summary.json     analyze.py        medians over the 10 seeds, per case/task/method
    out/kernel_checks.json  check_kernel.py   the enumerated catalogue and the measured contrast
    out/spec_error.json  check_spec_error.py  each surrogate's error on its own specification
    out/fit.json         check_fit.py      the case table's architecture and test R^2 columns
    out/isolation.json   check_isolation.py   the a-priori isolation test of Section 7.3
    out/plausibility.json  check_plausibility.py  the concrete verisimilitude filter

Rewritten on 2026-10-20 (audit finding B-12): the previous claim set predated the Round 3 scope
cut and the Round 7 reruns, so 45 of its 59 claims failed against the current metrics and the
gate had stopped gating. Keep this file in step with the prose: a number that appears in the
manuscript and not here is a number nobody checks.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    path = HERE / "out" / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


S = load("summary.json")
K = load("kernel_checks.json")
E = load("spec_error.json")
F = load("fit.json")
I = load("isolation.json")
PLAUS = (load("plausibility.json") or {}).get("medians")


def m(section, case, method, key):
    """Median of a metric over the seeds."""
    entry = S[section][case].get(method)
    return None if entry is None or key not in entry else entry[key]["median"]


def count(section, case, method, key):
    """A per-method seed count, not a median."""
    entry = S[section][case].get(method)
    return None if entry is None else entry.get(key)


def spec(case, field, window="+-25%"):
    return E[case]["windows"][window][field] if E else None


def fit(case, key, output=None):
    if not F or case not in F:
        return None
    return F[case]["r2_test"].get(output) if output else F[case].get(key)


def isolation(target, key):
    return None if not I else I["targets"].get(str(target), {}).get(key)


CLAIMS = [
    # ---- Section 5, case table and protocol -------------------------------------------
    ("cases tab", "error del sustituto, hormigón", 7.6, spec("concrete", "median_pct"), 0.05),
    ("cases tab", "error del sustituto, acero", 4.6, spec("steel", "median_pct"), 0.05),
    ("cases tab", "error del sustituto, núcleo", 5.8, spec("kernel", "median_pct"), 0.05),
    ("cases tab", "neuronas ocultas, hormigón", 12, fit("concrete", "hidden"), 0),
    ("cases tab", "neuronas ocultas, acero", 16, fit("steel", "hidden"), 0),
    ("cases tab", "neuronas ocultas, núcleo", 24, fit("kernel", "hidden"), 0),
    ("cases tab", "filas por peso, hormigón", 9.6, fit("concrete", "rows_per_weight"), 0.05),
    ("cases tab", "filas por peso, acero", 3.0, fit("steel", "rows_per_weight"), 0.05),
    ("cases tab", "filas por peso, núcleo", 12.4, fit("kernel", "rows_per_weight"), 0.05),
    ("cases tab", "R2 especificación, hormigón", 0.85,
     fit("concrete", None, "CompressiveStrength_MPa"), 0.005),
    ("cases tab", "R2 especificación, acero", 0.87,
     fit("steel", None, "tensile_strength_MPa"), 0.005),
    ("cases tab", "R2 especificación, núcleo", 0.99, fit("kernel", None, "log10_duration"), 0.005),
    ("cases tab", "R2 asentamiento, hormigón", 0.54, fit("concrete", None, "Slump_mm"), 0.005),
    ("cases tab", "R2 otras salidas, acero (máx)", 0.85,
     fit("steel", None, "proof_stress_MPa"), 0.005),
    ("cases tab", "R2 otras salidas, acero (mín)", 0.74,
     fit("steel", None, "reduction_area_pct"), 0.005),
    ("cases tab", "R2 ocupación, núcleo", 0.88,
     fit("kernel", None, "achieved_occupancy"), 0.005),

    # ---- 6.1 set recovery --------------------------------------------------------------
    ("6.1", "rechazo válidas 1e-5 hormigón", 3.5, m("sample", "concrete", "rejection", "valid_1e-05"), 0.05),
    ("6.1", "rechazo válidas 1e-5 acero", 28, m("sample", "steel", "rejection", "valid_1e-05"), 0.5),
    ("6.1", "rechazo válidas 1e-3 hormigón", 290, m("sample", "concrete", "rejection", "valid_0.001"), 1),
    ("6.1", "rechazo válidas 1e-3 acero", 2552, m("sample", "steel", "rejection", "valid_0.001"), 1),
    ("6.1", "reparación válidas 1e-2 hormigón", 1697, m("sample", "concrete", "repair", "valid_0.01"), 0.5),
    ("6.1", "reparación válidas 1e-5 hormigón", 1697, m("sample", "concrete", "repair", "valid_1e-05"), 0.5),
    ("6.1", "reparación válidas 1e-2 acero", 168, m("sample", "steel", "repair", "valid_0.01"), 0.5),
    ("6.1", "reparación válidas 1e-5 acero", 168, m("sample", "steel", "repair", "valid_1e-05"), 0.5),
    ("6.1", "reparación válidas 1e-7 hormigón", 866, m("sample", "concrete", "repair", "valid_1e-07"), 0.5),
    ("6.1", "reparación válidas 1e-7 acero", 113, m("sample", "steel", "repair", "valid_1e-07"), 0.5),
    ("6.1", "reparación válidas 1e-8 hormigón", 103, m("sample", "concrete", "repair", "valid_1e-08"), 0.5),
    ("6.1", "reparación válidas 1e-8 acero", 16, m("sample", "steel", "repair", "valid_1e-08"), 0.5),
    ("6.1", "cobertura reparación hormigón", 0.96, m("sample", "concrete", "repair", "coverage"), 0.005),
    ("6.1", "cobertura reparación acero", 0.94, m("sample", "steel", "repair", "coverage"), 0.005),
    ("6.1", "cobertura rechazo hormigón", 0.52, m("sample", "concrete", "rejection", "coverage"), 0.005),

    # ---- 6.2 one design ----------------------------------------------------------------
    ("6.2", "DC objetivo, cemento", 240.7, m("single", "concrete", "dc_objective", "best_0.0001"), 0.05),
    ("6.2", "DC restricción, cemento", 234.0, m("single", "concrete", "dc_constraint", "best_0.0001"), 0.05),
    ("6.2", "GA O igualada, cemento", 234.0, m("single", "concrete", "ga_objective-matched", "best_0.0001"), 0.05),
    ("6.2", "DC objetivo, alargamiento", 35.1, m("single", "steel", "dc_objective", "best_0.0001"), 0.05),
    ("6.2", "DC restricción, alargamiento", 37.7, m("single", "steel", "dc_constraint", "best_0.0001"), 0.05),
    ("6.2", "GA O igualada, alargamiento", 38.9, m("single", "steel", "ga_objective-matched", "best_0.0001"), 0.05),

    # ---- 6.3 fronts --------------------------------------------------------------------
    ("6.3", "GA R útiles 5 % hormigón", 100, m("bi", "concrete", "ga_constraint", "usable_0.05"), 0.5),
    ("6.3", "DC R devueltos hormigón", 38, m("bi", "concrete", "dc_constraint", "returned"), 0.5),
    ("6.3", "GA R útiles 5 % acero", 41.5, m("bi", "steel", "ga_constraint", "usable_0.05"), 0.05),
    ("6.3", "DC R devueltos acero", 13, m("bi", "steel", "dc_constraint", "returned"), 0.5),
    ("6.3", "O útiles 5 %, mínimo", 21, m("bi", "concrete", "ga_objective-matched", "usable_0.05"), 0.5),
    ("6.3", "O útiles 5 %, máximo", 41, m("bi", "steel", "dc_objective-matched", "usable_0.05"), 0.5),
    ("6.3", "O útiles 1 %, mínimo", 9.5, m("bi", "concrete", "ga_objective-matched", "usable_0.01"), 0.05),
    ("6.3", "O útiles 1 %, máximo", 24, m("bi", "steel", "dc_objective-matched", "usable_0.01"), 0.5),
    ("6.3", "O válidas 1e-5, máximo", 2.5, m("bi", "steel", "ga_objective-matched", "valid_1e-5"), 0.05),
    ("6.3", "HV O igualada hormigón", 1.11, m("bi", "concrete", "ga_objective-matched", "hv"), 0.005),
    ("6.3", "HV R hormigón", 1.19, m("bi", "concrete", "ga_constraint", "hv"), 0.005),
    ("6.3", "O igualada válidas 1e-5 hormigón", 1.5, m("bi", "concrete", "ga_objective-matched", "valid_1e-5"), 0.05),

    # ---- 6.4 verisimilitude ------------------------------------------------------------
    ("6.4", "válidos hormigón", 1697, (PLAUS or {}).get("repair", {}).get("valid"), 0.5),
    ("6.4", "verosímiles hormigón", 1277, (PLAUS or {}).get("repair", {}).get("plausible"), 0.5),
    ("6.4", "soportados hormigón", 171, (PLAUS or {}).get("repair", {}).get("supported"), 0.6),
    ("6.4", "% verosímiles", 75, (PLAUS or {}).get("repair", {}).get("% plausible"), 0.5),
    ("6.4", "% soportados", 10, (PLAUS or {}).get("repair", {}).get("% supported"), 0.5),

    # ---- 6.5 cost against a local method -----------------------------------------------
    ("6.5", "SLSQP válidas hormigón", 2898, m("sample", "concrete", "slsqp", "valid_1e-05"), 1),
    ("6.5", "SLSQP evals/sol hormigón", 53.3, m("sample", "concrete", "slsqp", "evals_per_sol"), 0.05),
    ("6.5", "reparación evals/sol hormigón", 90.2, m("sample", "concrete", "repair", "evals_per_sol"), 0.05),
    ("6.5", "cobertura SLSQP hormigón", 0.961, m("sample", "concrete", "slsqp", "coverage"), 0.001),
    ("6.5", "cobertura reparación hormigón", 0.963, m("sample", "concrete", "repair", "coverage"), 0.001),
    ("6.5", "GA R propagaciones hormigón", 87919, m("single", "concrete", "ga_constraint", "evaluations"), 500),
    ("6.5", "SLSQP propagaciones hormigón", 164525, m("single", "concrete", "slsqp", "evaluations"), 500),
    ("6.5", "HV SLSQP escalarizado hormigón", 1.187, m("bi", "concrete", "slsqp_mo", "hv"), 0.001),
    ("6.5", "SLSQP escalarizado puntos hormigón", 872, m("bi", "concrete", "slsqp_mo", "valid_1e-5"), 1),
    ("6.5", "SLSQP válidas acero", 1439, m("sample", "steel", "slsqp", "valid_1e-05"), 1),
    ("6.5", "SLSQP evals/sol acero", 294, m("sample", "steel", "slsqp", "evals_per_sol"), 1),
    ("6.5", "reparación evals/sol acero", 2507, m("sample", "steel", "repair", "evals_per_sol"), 1),
    ("6.5", "HV SLSQP escalarizado acero", 1.194, m("bi", "steel", "slsqp_mo", "hv"), 0.001),
    ("6.5", "HV GA R acero", 1.073, m("bi", "steel", "ga_constraint", "hv"), 0.001),

    # ---- 7.1 the attainable set --------------------------------------------------------
    ("7.1", "configuraciones admisibles", 4680, (K or {}).get("catalogue"), 0.5),
    ("7.1", "dentro del 1 %", 63, (K or {}).get("exact_set_sizes", {}).get("0.01"), 0.5),
    ("7.1", "dentro de 1e-3", 5, (K or {}).get("exact_set_sizes", {}).get("0.001"), 0.5),
    ("7.1", "dentro de 1e-4", 0, (K or {}).get("exact_set_sizes", {}).get("0.0001"), 0.5),
    ("7.1", "residuo del más próximo", 4.674e-4, (K or {}).get("best_attainable"), 2e-6),
    ("7.1", "banda para 1 configuración", 4.7e-4, (K or {}).get("band_for_n_configurations", {}).get("1"), 2e-5),
    ("7.1", "banda para 10", 1.7e-3, (K or {}).get("band_for_n_configurations", {}).get("10"), 1e-4),
    ("7.1", "banda para 50", 7.8e-3, (K or {}).get("band_for_n_configurations", {}).get("50"), 1e-4),

    # ---- 7.2 the statements and the baselines ------------------------------------------
    ("7.2", "óptimo del catálogo", 0.8292, m("single", "kernel", "enumeration", "best_0.01"), 0.0005),
    ("7.2", "GA banda propagaciones", 535334, m("single", "kernel", "ga_band", "evaluations"), 2000),
    ("7.2", "DC banda propagaciones", 1492810, m("single", "kernel", "dc_band", "evaluations"), 5000),
    ("7.2", "DC O igualada propagaciones", 79841, m("single", "kernel", "dc_objective-matched", "evaluations"), 500),
    ("7.2", "CMA-ES banda propagaciones", 2186751, m("single", "kernel", "cmaes-band", "evaluations"), 5000),

    # ---- 7.3 the a-priori isolation test ------------------------------------------------
    ("7.3", "objetivos con óptimo aislado", 5, (I or {}).get("isolated_count"), 0),
    ("7.3", "objetivos probados", 8, (I or {}).get("targets_tried"), 0),
    ("7.3", "vecinos dentro de la banda a 8000", 0, isolation(8000, "neighbours_in_band"), 0),
    ("7.3", "vecinos admisibles del óptimo a 8000", 8, isolation(8000, "neighbours"), 0),
    ("7.3", "configuraciones en la banda a 8000", 63, isolation(8000, "in_band"), 0),
    ("7.3", "vecinos dentro de la banda a 12000", 1, isolation(12000, "neighbours_in_band"), 0),
    ("7.3", "vecinos dentro de la banda a 16000", 1, isolation(16000, "neighbours_in_band"), 0),
    ("7.3", "vecinos dentro de la banda a 30000", 1, isolation(30000, "neighbours_in_band"), 0),

    # ---- 7.3 set recovery on the catalogue ---------------------------------------------
    ("7.3", "reparación banda distintas", 36, m("sample", "kernel", "repair_band", "distinct"), 0.5),
    ("7.3", "SLSQP redondeado banda distintas", 42.5, m("sample", "kernel", "slsqp_round-band", "distinct"), 0.05),
    ("7.3", "muestreo uniforme distintas", 63, m("sample", "kernel", "rejection-band", "distinct"), 0.5),

    # ---- 7.4 contrast with the measurements --------------------------------------------
    ("7.4", "% del catálogo medido", 70.7,
     100.0 * (K or {}).get("measured_in_catalogue", 0) / max((K or {}).get("catalogue", 1), 1), 0.05),
    ("7.4", "residuo medido mediano %", 3.5,
     100.0 * (K or {}).get("per_method", {}).get("repair_band", {}).get("measured_residual_median", 0), 0.1),
]

SEED_COUNTS = [
    ("7.2", "GA banda, óptimo en k/10", 10, count("single", "kernel", "ga_band", "optimum_best_0.01")),
    ("7.2", "DC banda, óptimo en k/10", 10, count("single", "kernel", "dc_band", "optimum_best_0.01")),
    ("7.2", "DC O igualada, óptimo en k/10", 9, count("single", "kernel", "dc_objective-matched", "optimum_best_0.01")),
    ("7.2", "SLSQP redondeado igualdad", 0, count("single", "kernel", "slsqp_round", "optimum_best_0.01")),
    ("7.2", "SLSQP redondeado banda", 0, count("single", "kernel", "slsqp_round-band", "optimum_best_0.01")),
    ("7.2", "SLSQP relajado igualdad", 0, count("single", "kernel", "slsqp_relaxed", "optimum_best_0.01")),
    ("7.2", "SLSQP relajado banda", 0, count("single", "kernel", "slsqp_relaxed-band", "optimum_best_0.01")),
    ("7.2", "CMA-ES banda", 10, count("single", "kernel", "cmaes-band", "optimum_best_0.01")),
    ("7.2", "DE banda", 5, count("single", "kernel", "de-band", "optimum_best_0.01")),
    ("7.2", "GA pymoo banda", 0, count("single", "kernel", "ga-band", "optimum_best_0.01")),
    ("7.2", "PSO banda", 0, count("single", "kernel", "pso-band", "optimum_best_0.01")),
    ("7.2", "DC igualdad (insatisfacible)", 0, count("single", "kernel", "dc_constraint", "optimum_best_0.01")),
    ("7.2", "GA igualdad (insatisfacible)", 0, count("single", "kernel", "ga_constraint", "optimum_best_0.01")),
]


def main() -> int:
    bad = missing = 0
    for where, what, quoted, computed, tol in CLAIMS:
        if computed is None:
            missing += 1
            print(f"MISS {where:10s} {what:38s} quoted {quoted}, no metric found")
            continue
        ok = abs(quoted - computed) <= tol
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {where:10s} {what:38s} quoted {quoted} vs computed {computed}")
    for where, what, quoted, computed in SEED_COUNTS:
        if computed is None:
            missing += 1
            print(f"MISS {where:10s} {what:38s} quoted {quoted}, no metric found")
            continue
        ok = quoted == computed
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {where:10s} {what:38s} quoted {quoted} vs computed {computed}")
    total = len(CLAIMS) + len(SEED_COUNTS)
    print(f"\n{total} claims, {bad} mismatch(es), {missing} without a metric")
    return 1 if (bad or missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
