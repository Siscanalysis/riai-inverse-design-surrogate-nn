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

## Baselines (honest: SAME repair given to ALL baselines)  -> out/baselines_{cnc,hea}.csv
Equal eval budget (24000; halved to expose eval-efficiency), identical feasibility test
(output band + simplex). 4 methods. feas% = feasible / TRUE evals.

| method | repair | CNC feas% | CNC fill% | CNC minS | HEA feas% | HEA simplex | HEA minS |
|---|---|---|---|---|---|---|---|
| reject_norepair | no  | 7.9%  | 99.9%  | 0.055 | 0%   | 0.98  | -   |
| reject_repair   | yes | 22.9% | 99.97% | 0.082 | 3.1% | 0.000 | 311 |
| ours_iterative  | yes | 5.9%  | 90.4%  | 0.141 | 6 pts| 0.000 | 770 |
| nsga2_repair    | yes | 0.4%  | 7.1%   | 0.039 | 73pts| 0.000 | 115 |

Honest findings (none is "we dominate"):
1. Repair is the decisive enabler: without it the HEA simplex is unsolvable by sampling
   (0% feasible, residual 0.99); with the SAME repair it is satisfied EXACTLY (0.000) by all.
   The repair operator is shared machinery from the companion (new_IDC), NOT novel here.
2. Coverage <-> robustness trade-off: reject_repair covers best (fill 99.98%, highest feas);
   NSGA-II (optimizes S) finds the MOST ROBUST design (minS 0.022 CNC / 111 HEA) but covers
   worst (fill 32%/4%); ours (contraction) is in between.
3. Our value = the UNIFIED inverse-problem framework: recover the whole fixed-target manifold
   + mixed senses (min F / max MRR along the band) + robustness selection, in one derivative-
   free pass. Baselines each solve only one axis (coverage OR a robust optimum).

NSGA-II: real-coded, SBX + polynomial mutation, fast-non-dominated-sort + crowding,
bi-objective (closeness, S), same repair (closed-form simplex projection) each generation.

### Fairness (true evals via get_evaluations_used)  [answers "same #evals?"]
All methods use <= 24000 TRUE surrogate forwards (repair evals counted within budget):
reject 24000, NSGA-II 24000, ours 18000 (converges early on the tolerance -> FEWER, not more).

### Mixed-sense optimization (CNC: fix Ra + minimize F)  -> out/baselines_cnc_mixedsense.csv, out/pymoo_cnc.csv
The pure fixed-target is the WORST case for ours (nothing to optimize). With a secondary
objective (min F), ours (an optimizer) beats a sampler and ties the best metaheuristics,
at fewer evals -- and the gap to sampling GROWS as the budget shrinks:
| method | best feasible F [N] @24k | @48k |
| ours      | **78.0 @18k evals** | 77.9 @36k |   <- barely changes (converges)
| DE        | 78.0 | 78.0 |
| PSO       | 78.0 | 78.0 |
| GA        | 87.5 | 87.5 |
| rejection (sampler) | 97.6 | 88.5 |   <- degrades most at low budget
| CMA-ES    | 114.8 | -- |   <- feasible (with repair+restarts) but worst objective

### pymoo panel with the SAME repair (GA/DE/PSO/CMA-ES)  -> out/pymoo_{cnc,hea}.csv
- BUDGET FAIRNESS: n_evals termination gives every algo exactly 24000 evals (recorded in
  the n_eval column). CMA-ES uses IPOP restarts (restarts=12) to spend the budget instead
  of self-terminating at ~66 evals; without restarts it collapses early.
- HEA simplex: with the same repair, ALL FOUR (GA/DE/PSO/CMA-ES) satisfy sum=1 EXACTLY
  and hit the HV band. Repair is the enabler; without it (reject_norepair) the simplex is
  infeasible (0%). CMA-ES is feasible but the weakest optimizer on CNC min-F (114.8).
- Surrogates exported to Python via OpenNN ModelExpression (models/*_surrogate.py);
  pymoo 0.6.1.6. Run: `python pymoo_baselines.py cnc|hea`.

## Build note (refactor-immune)
The upstream OpenNN refactor moved HEAD (5598487e3 -> 454232b74) and rebuilt/reorganized
the lib+headers mid-work. The companion is now SELF-CONTAINED: rerun/_vendor/ holds the
5598487e3 headers (tracked) + a lib built at that commit via `git worktree` + Eigen
(both gitignored). CMake points only at _vendor; nothing depends on the live source.

## Manuscript
paper/manuscript.pdf (10 pages) - includes the honest baselines section (Sec 5) with
Table + fig_baselines_feasibility.pdf + fig_baselines_robustness.pdf.
Draft for review BEFORE sending the title email to jlcalvo@udc.es.
