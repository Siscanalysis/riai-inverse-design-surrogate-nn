# RIAI rerun — results log

Build: VS2022 (bundled cmake) linking prebuilt OpenNN `build-ninja/opennn.lib`.
`cmake -S . -B build -G "Visual Studio 17 2022" -A x64 && cmake --build build --config Release`
Run each exe from `rerun/` (drivers use relative `models/`, `out/`).

IMPORTANT: every `*.json` net needs its sibling `*.bin` (same basename) — that is where
the weights live. Copying only the `.json` loads uninitialized weights (constant output).

## Surrogates (all-numeric)
| case | net (models/) | arch | quality |
|---|---|---|---|
| A CNC | cnc_multi.json (+.bin) | 4→32→2 | full-data, skip-gn (GN collapsed → constant; skip-gn fixes). Mean R²≈0.84 over {Ra,F}; F≈0.99, Ra≈0.69 |
| B HEA | hea_10elt.json (+.bin) | 10→37→1 | test R²≈0.87 (370 rows, 10-elt simplex) |
| B HEA alt | hea_6elt.json (+.bin) | 6→2→1 | test R²≈0.90 (236 rows, AlCoCrCuFeNi) |

## Case A — CNC mixed-sense (fix Ra / min F / max MRR)
- Net sanity: feed 0.07→Ra 0.40, 0.10→0.76, 0.13→1.22 (varies correctly).
- Reachable Ra ≈ [0.13, 1.25] µm; equality target = 0.62 µm (55th pct, interior).
- RUN1 (FIX Ra): manifold = 3400 recipes, Ra held in [0.605, 0.638] (band ≈ target).
  - F trade-off [136, 429] N (MINIMIZE); MRR trade-off [7.6, 28.4] (MAXIMIZE) — genuine tension.
  - get_robust_point balance 0/0.5/1 → distinct rows (sensRa 0.60–0.80).
- RUN2 (FIX Ra + MIN F): min-F recipe F=77.8 N at x=[vc 387, f 0.090, ap 0.25, VB 0].
- RUN3 (FIX Ra + FIX F multi-target): 5100 intersection recipes.
- CSVs: out/cnc_manifold.csv, out/cnc_fixRa_minF.csv, out/cnc_multitarget.csv, out/cnc_domain.csv

## Case B — HEA fixed-HV simplex manifold
- Reachable HV ≈ [-195, 1086]; target = 546 HV (60th pct).
- Simplex sum=1 enforced EXACTLY: max |Σx − 1| = 0.0000 over the manifold.
- Manifold = 170 alloys (feasible ratio ~0.008 — tight equality; oversampling escalates, OK).
- Diversity: fraction fills 44–99% of domain (rich, high-dim manifold).
- Robustness: most-robust alloy sens 454 vs least ~912; get_robust_point balance 1 → row 32.
- CSV: out/hea_manifold.csv
- Minor: one V fraction dips to −0.006 (box lower bound slightly <0); physically ≈0.
  TODO if wanted: add x_i ≥ 0 formula constraints to clamp.

## Baselines (honest: SAME repair given to the baselines)  -> out/baselines_{cnc,hea}.csv
Equal-ish eval budget (48000), identical feasibility test (output band + simplex).

| method | repair | CNC feas% | CNC meanS/minS | HEA feas% | HEA simplex resid | HEA meanS/minS |
|---|---|---|---|---|---|---|
| reject_norepair | no | 7.8% | 0.62 / 0.055 | 0% | 0.99 | - |
| reject_repair   | yes | 22.9% | 0.75 / 0.077 | 3.0% | 0.000 | 1092 / 319 |
| ours_iterative  | yes | 5.7% | 0.44 / 0.185 | ~6 pts | 0.000 | 771 / 477 |

Honest findings:
1. Repair is the decisive enabler: without it the HEA simplex is unsolvable by sampling
   (0% feasible, residual 0.99); with the SAME repair it is satisfied exactly (0.000).
2. For pure single-target COVERAGE, one-shot repaired rejection covers as well or better
   than our iterative contraction (which converges/concentrates, not covers).
3. Our method's real value: on-average more robust returned set (CNC meanS 0.44 vs 0.75),
   MIXED-SENSE (min F / max MRR along the fixed-Ra band, which rejection does not optimize),
   and robustness SELECTION: from the same repaired feasible set, get_robust_point picks
   minS (CNC 0.077 vs typical 0.75 ~10x; HEA 319 vs 1092 ~3.4x).

Notes: ours trips OpenNN's feasibility-ratio guard on the tight simplex unless
set_min_feasible_ratio is lowered (done, 1e-4). NSGA-II with the same repair is the
remaining baseline (future work); expected to behave like a more sample-efficient
repaired sampler.

## Manuscript
paper/manuscript.pdf (10 pages) - includes the honest baselines section (Sec 5) with
Table + fig_baselines_feasibility.pdf + fig_baselines_robustness.pdf.
Draft for review BEFORE sending the title email to jlcalvo@udc.es.
