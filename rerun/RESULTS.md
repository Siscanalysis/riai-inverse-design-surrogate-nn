# RIAI rerun — results log

> **SEEDED (2026-07-27):** all C++ drivers now call `opennn::set_seed(1)` at the top of main
> => baselines + case_a are bit-deterministic (only wall-clock varies). Every manuscript number
> was re-locked to the seed=1 run; the authoritative locked values live in
> `../AUDIT_DECISIONS.md` (LOCKED NUMBERS). Two consequences worth flagging:
> (1) the HEA extrapolation "win" REVERSED under the seed (ours d2train 0.253 vs reject 0.220,
>     because ours recovers only ~3 HEA points => noise); the manuscript now DROPS that claim.
> (2) `case_b_hea.exe` crash FIXED (2026-07-27): the HEA simplex's low feasible ratio triggered
>     unbounded adaptive oversampling into a lib crash path. Fix in case_b: set_min_feasible_ratio(1e-4)
>     + set_evaluations_number(48000) + set_max_total_evaluations(500000). Case B now seeded + crash-free:
>     **183 alloys, 39-99% fill, HV~538, |sum-1|=0** (was 170/44-100% from the crashing oversample).
> References added (strictly necessary, web-verified): pymoo (Blank&Deb 2020), GA (Goldberg 1989),
>     DE (Storn&Price 1997), PSO (Kennedy&Eberhart 1995), CMA-ES (Hansen&Ostermeier 2001), OpenNN.
> The tables below still show the ORIGINAL unseeded numbers for history; trust AUDIT_DECISIONS.md.


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
  - get_robust_point balance 0/0.5/1 → distinct rows. wallMargin (centrality, 1=centre,
    0=on a wall) = 0.685 / 0.387 / 0.072: the pure-robust pick (a=1) sits AT the domain
    edge (margin 0.07) where the surrogate extrapolates/saturates; a=0.5 and a=0 pull the
    recipe back into the well-sampled interior (0.39, 0.69). => prefer a moderate alpha as
    a hedge against surrogate extrapolation. (sensRa printed is Ra-only; get_robust_point
    optimizes the FULL-output Jacobian, so don't tie S to sensRa.) Added to manuscript
    Sec 2 (robustez) + Sec 3 (Caso A).
- RUN2 (FIX Ra + MIN F): min-F recipe F=77.8 N at x=[vc 387, f 0.090, ap 0.25, VB 0].
- RUN3 (FIX Ra + FIX F multi-target): 5100 intersection recipes.
- CSVs: out/cnc_manifold.csv, out/cnc_fixRa_minF.csv, out/cnc_multitarget.csv, out/cnc_domain.csv

## Case B — HEA fixed-HV simplex manifold
- Reachable HV ≈ [-195, 1086]; target = 546 HV (60th pct).
- Simplex sum=1 enforced EXACTLY: max |Σx − 1| = 0.0000 over the manifold.
- Manifold = 170 alloys (feasible ratio ~0.008 — tight equality; oversampling escalates, OK).
- Diversity: fraction fills 44–99% of domain (rich, high-dim manifold).
- Robustness: most-robust alloy sens 454 vs least ~912; get_robust_point balance 1 → row 32.
  wallMargin: a=1 row32 margin 0.005 (on a simplex edge, sens 454); a=0/0.5 row167 margin
  0.142 (sens 912). On the simplex margins are inherently small (many fractions near 0), so
  the extrapolation-hedge argument is cleaner on CNC; HEA shown qualitatively.
- CSV: out/hea_manifold.csv
- Minor: one V fraction dips to −0.006 (box lower bound slightly <0); physically ≈0.
  TODO if wanted: add x_i ≥ 0 formula constraints to clamp.

## Baselines (honest: SAME repair given to ALL baselines)  -> out/baselines_{cnc,hea}.csv
Equal HARD eval budget = 12000 for EVERY method (ours included), identical feasibility test
(output band + simplex). 4 methods. feas% = feasible / TRUE evals.
KEY FAIRNESS FIX: ours now uses iterations=6 (not 8). Because min_iterations=4 in the lib,
the earliest early-stop is i=5, which IS the last iteration when max_iterations=6 -> ours
CANNOT stop before spending the whole budget => uses exactly 12000, same as everyone.
(Previously iters=8 -> stopped at i=5 -> only 18000; that "fewer evals" optics is now gone.)

NSGA-II DROPPED (2026-07-24): once we agreed to keep the comparison SCALAR (min F), NSGA-II
just duplicated the pymoo GA and, as an MO (min F/max MRR) optimizer, couldn't be matched
apple-to-apple by IDC (MRR is a product of inputs, NOT a network output, so IDC's
set_objective can't express "max MRR"). So NSGA-II + its repair_x + the MO front script were
removed. Robustness is post-hoc selection for ALL remaining methods (nobody optimizes S).

| method | repair | CNC feas% | CNC fill% | CNC minS | HEA feas% | HEA simplex | HEA minS |
|---|---|---|---|---|---|---|---|
| reject_norepair | no  | 7.6%  | 99.8%  | 0.055 | 0%    | 0.98  | -    |
| reject_repair   | yes | 23.1% | 99.96% | 0.094 | 358pts| 6e-6  | 259  |
| ours_iterative  | yes | 4.2%  | 78.6%  | 0.278 | 4 pts | 8e-8  | 1400 |

Honest findings (none is "we dominate"):
1. Repair is the decisive enabler: without it the HEA simplex is unsolvable by sampling
   (0% feasible, residual 0.99); with the SAME repair it is satisfied EXACTLY (0.000) by all.
   The repair operator is shared machinery from the companion (new_IDC), NOT novel here.
2. Robustness is POST-HOC for all (nobody optimizes S). Best min S tracks coverage: broad
   sampling (reject) contains the flattest points (minS 0.055 CNC); ours concentrates on the
   feasible region (79% fill) and its best S is higher. So coverage & robustness-of-best-point
   go together; the real differentiator is extrapolation (finding 3).
3. EXTRAPOLATION (the real differentiator, out/feasible_*.csv + analyze_extrapolation.py):
   wall-margin (dist to box wall) + dist-to-nearest-training-point per feasible point.
   Now clean (no NSGA-II): OURS is best on BOTH axes -- closest to data (0.148 vs reject
   0.18-0.19) AND fewest edge points (6% vs 22-31%). HEA (simplex => all on walls, use
   dist-to-data): ours 0.16 best, reject 0.22. Claim = "ours stays near-data AND interior".
4. Our value = the UNIFIED inverse-problem framework: recover the whole fixed-target manifold
   + mixed sense (min F on the band) + robustness selection, in one derivative-free pass. On
   the scalar objective ours ties the best metaheuristics (DE/PSO 78.0 N).

### Fairness (true evals via get_evaluations_used)  [answers "same #evals?"]
ALL methods use EXACTLY 12000 TRUE surrogate forwards (repair evals counted within budget):
reject 12000, ours 12000 (iters=6 spends the whole budget). Literal parity.

### Exact target, NO tolerance interval (CNC only)  -> out/baselines_cnc_exact.csv
Shows IDC handles an EXACT equality spec without a user-chosen tolerance. The band is only a
COLLECTION device; sweep it toward zero (rel_tol delta = 1e-2..1e-6) and count solutions
returned AT the target, same 12000-eval budget. HONEST framing pivot: the right metric is NOT
"best single residual" -- Ra is effectively feed-dominated (near-1D), so a uniform sampler's
LUCKY best point already lands ~9e-6 away, comparable to ours. The decisive metric is HOW MANY
solutions AT the target each method returns: hitting a measure-zero target by chance is
vanishingly unlikely, so the rejection SET collapses as the band shrinks, while contraction
concentrates the whole budget on the target and stays populated.
| delta (rel_tol) | band | reject #sol | ours #sol |
| 1e-2 | 1.2e-2 | 920 | 1700 |
| 1e-3 | 1.2e-3 | 103 | 1700 |
| 1e-4 | 1.2e-4 | 12  | 1700 |
| 1e-5 | 1.2e-5 | 2   | 1700 |
| 1e-6 | 1.2e-6 | **0** | **877** |  <- rejection empty; ours still returns the exact-value set
Reject best single residual floor ~9e-6 (constant, density-limited); ours residual -> ~2e-7.

ALL baselines on the exact target (added per review)  -> out/pymoo_cnc_exact.csv
`python pymoo_baselines.py cnc exact` : GA/DE/PSO/CMA-ES minimize |Ra-c| (root finding). At a
near-exact band delta=1e-4, count DISTINCT solutions returned AT the target + their diversity
(fill%). Reveals manifold-vs-point-vs-nothing:
| method | #sol @1e-4 | fill% (diversity) | best resid |
| ours (contraction) | **1700** | 44% | 6e-7 |
| rejection          | 12   | 77% | 9e-6 |  (collapses to 0 at 1e-6)
| DE                 | 28   | 90% | 4e-8 |
| PSO                | 83   | 34% | 2e-8 |
| GA                 | 100* | **0%** | **2e-13** |  *100 copies of ONE point
| CMA-ES             | 63   | 3%  | 1e-6 |
HONEST NUANCE (both directions):
 - IDC uniquely returns the exact-value MANIFOLD (1700 diverse, still 877 at band 1e-6).
 - Rejection: set collapses (measure-zero target can't be hit by chance).
 - Metaheuristics: return only their <=100 pop; GA/CMA-ES CONVERGE TO ONE POINT (fill~0),
   DE/PSO keep some spread but thin at tight bands. A scalar optimizer is a point-finder, not
   a set-recovery method.
 - IDC is NOT the most precise single point: GA (float64 root finder) hits |Ra-c|~2e-13 vs
   IDC ~2e-7 (float32 floor of the OpenNN net). Our claim is the exact-value SET, not the
   tightest point. Say so explicitly.
NOTE: numbers are ONE representative run (sampling stochastic); trends robust.

SIMPLEX (affine) EXACTNESS -- investigated per review. From baselines_hea.csv simplex_resid +
pymoo_hea.csv: |sum x - 1| = reject_norepair 0.98 (no repair), reject_repair 6e-6, ours 8.4e-8,
pymoo GA/DE/PSO/CMA-ES ~1e-17. FINDING: the affine simplex constraint is satisfied to MACHINE
PRECISION by EVERY repaired method -- ours 8.4e-8 is the float32 floor (OpenNN is single
precision; float32 eps ~1.2e-7), pymoo hits 1e-17 in float64. So simplex exactness = the SHARED
affine repair, NOT a unique IDC power (and pymoo is even tighter in double precision). IDC's
DISTINCTIVE exactness is the OUTPUT equality target via contraction (the CNC box result above).
On the HEA OUTPUT target at equal 12000 budget IDC does NOT dominate: it recovers only 4 alloys
(vs reject_repair 358, GA |HV-c|~1e-10) -- the tight 10-D simplex manifold is starved at 2000
samples/iter (case_b uses 48000 evals -> 170 alloys). Honest: no exact-target win on HEA.
Manuscript: Sec 2.2 (band = collection device) + Sec 5 tab:exact (all baselines, delta=1e-4) +
Sec 5 Finding-1 affine-exactness clarification + intro/concl.
HEA exact-target SWEEP is DELIBERATELY SKIPPED in baselines.cpp (if !simplex guard).

### Mixed-sense optimization (CNC: fix Ra + minimize F)  -> out/baselines_cnc_mixedsense.csv, out/pymoo_cnc.csv
The pure fixed-target is the WORST case for ours (nothing to optimize). With a secondary
objective (min F), ours ties the best general-purpose optimizers and beats GA, sampling and
CMA-ES -- all at the SAME 12000-eval budget (no "fewer evals" caveat anymore):
| method | best feasible F [N] @12000 evals |
| ours      | **78.0** |  <- ties the best optimizers (top group)
| DE        | 78.0 |
| PSO       | 78.0 |
| rejection (sampler) | 86.6 |
| GA        | 87.5 |
| CMA-ES    | 114.8 |  <- feasible (with repair+restarts) but worst objective
NOTE: dropped the earlier "gap grows as budget shrinks" claim -- rejection's best-F is
noisy across budgets (86.6 @12k vs 97.6 @24k was sampling luck, not a trend). Stay humble:
ours is in the top group and clearly beats GA/sampler/CMA-ES; it does NOT beat DE/PSO.

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
paper/manuscript.pdf - includes the honest baselines section (Sec 5) with
tab:base + tab:mixed + fig_extrapolation_cnc.pdf, all at the 12000-eval budget. Baselines =
reject +/- repair, ours, and the pymoo panel (GA/DE/PSO/CMA-ES) for the scalar objective;
NSGA-II dropped (see note above). Draft for review BEFORE sending the title email.
FORMAT (2026-07-27): now in the official RIAI camera-ready format (elsarticle 5p +
ecrc_RIAI.sty, two-column CRC, Elsevier/SD/RIAI logo header, "Copyright (c) 2026 CEA.").
The two-column format compresses the same content to 6 pages (vs 12 in the old single-column
preprint draft - same content, denser layout). RIAI package files (ecrc_RIAI.sty,
elsarticle.cls, elsarticle-harv.bst, RIAIlogo.pdf, SDlogo-5p.pdf, elsevier-logo-5p.pdf) are
committed under paper/ and pulled from the RIAI template bundle. TODO before submit: real
author emails (COMPLETAR@ placeholders) + volume/firstpage when assigned.
