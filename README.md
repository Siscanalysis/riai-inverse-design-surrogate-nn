# Inverse design with mixed objectives and equality constraints on surrogate neural networks

Reproducibility repository for the RIAI (Revista Iberoamericana de Automática e
Informática Industrial) article *"Problemas inversos con objetivos mixtos y
restricciones de igualdad sobre redes neuronales sustitutas"* (S. Scala, R. López),
an extension of the SCI 2026 symposium paper *"Diseño industrial óptimo con redes
neuronales"*.

**This repo contains only the reproducible experiments** — the code, data, trained
surrogates, and the scripts that regenerate every figure, table and number in the
article. The manuscript itself is not included here.

Inverse design is treated as recovering the **feasible solution manifold** defined by
an equality (fixed-target) constraint on a trained surrogate, with **mixed-sense**
objectives (minimize / maximize / fixed on the same problem) and a **robustness +
centrality** selection over the manifold.

## Layout
```
rerun/                reproducible experiments
  case_a_cnc.cpp        Case A - CNC turning AISI H13 (fix Ra / min F; MRR as trade-off)
  case_b_hea.cpp        Case B - HEA hardness on the composition simplex (fix HV)
  case_c_steel.cpp      Case C - low-alloy steel (fix UTS at T=27C; ductility vs cost Pareto;
                          nonlinear weldability CEV + affine cost repair)
  train_steel.cpp       trains the Case C surrogate -> models/steel_multi.{json,bin}
  baselines.cpp         honest baselines (reject +/- repair, ours) + exact-target sweep [cnc|hea|steel]
  pymoo_baselines.py    metaheuristic panel (GA/DE/PSO/CMA-ES), scalar + exact [cnc|hea|steel]
  analyze_extrapolation.py   extrapolation analysis -> figs/fig_extrapolation_*.pdf
  gen_figs.py           manifold figures -> figs/*.pdf
  models/               trained surrogates: each <name>.json REQUIRES its <name>.bin sibling
  data/                 the numeric CSVs the surrogates were trained on
  out/                  CSVs produced by the drivers (gitignored; regenerated)
  figs/                 generated figures (PDF)
  CMakeLists.txt
```

## Surrogates
All-numeric regression nets trained with the `baseline_trainer` (Growing-Neurons /
adaptive width search) over the corresponding `benchmark_datasets` entries.

| model | inputs -> outputs | arch | test R^2 |
|---|---|---|---|
| `cnc_multi` | (v_c, f, a_p, VB) -> (Ra, F) | 4->32->2 | mean ~0.84 (F~0.99, Ra~0.7) |
| `hea_10elt` | 10 atomic fractions -> HV | 10->37->1 | ~0.87 |
| `hea_6elt`  | 6 atomic fractions (AlCoCrCuFeNi) -> HV | 6->2->1 | ~0.90 |
| `steel_multi` | 13 composition + temperature -> (yield, UTS, elongation, RA) | 14->32->4 | 0.94 UTS, 0.93 yield, 0.81-0.83 ductility |

Data sources: Case A CNC turning dataset (AISI H13) — Canal & Borille, Kaggle,
DOI 10.34740/kaggle/ds/2205074. Case B HEA hardness — Yang et al., Acta Materialia 222
(2022) 117431, DOI 10.1016/j.actamat.2021.117431. Case C low-alloy steel tensile tests —
NIMS MatNavi materials database (https://mits.nims.go.jp/), public Kaggle mirror; a related
ML study on NIMS steel data is Xiong, Zhang & Shi, Sci. China Tech. Sci. 63 (2020) 1247-1255,
DOI 10.1007/s11431-020-1599-5 (NN+GA composition-design baseline: Zhu, Liang & Zou,
Materials 13 (2020) 5316, DOI 10.3390/ma13235316).

> Note: `*.json` holds the architecture; the weights live in the sibling `*.bin`
> (same basename). Keep them together or the net loads with uninitialized weights.

## OpenNN dependency (vendored, pinned)
The experiments are pinned to **OpenNN commit 5598487e3**. Because the upstream tree is
under active refactor, the build is made self-contained via `rerun/_vendor/`:
- `_vendor/opennn/` public headers at that commit (committed, small);
- `_vendor/eigen/`  Eigen headers (gitignored, regenerable);
- `_vendor/lib/`    `opennn.lib` + `zs.lib` + `jpeg-static.lib` (gitignored, ~190 MB).

To regenerate the gitignored binaries on another machine, build OpenNN at the pinned
commit and copy the outputs into `_vendor/`:
```
git -C <opennn-repo> worktree add <build-src> 5598487e3
cmake -S <build-src> -B <build-src>/build -G "Visual Studio 17 2022" -A x64 ^
      -DOpenNN_BUILD_EXAMPLES=OFF -DOpenNN_BUILD_TESTS=OFF -DOpenNN_ENABLE_LTO=OFF
cmake --build <build-src>/build --config Release --target opennn
# copy opennn.lib, zs.lib, jpeg-static.lib -> rerun/_vendor/lib/
# copy <build>/_deps/eigen-src/{Eigen,unsupported} -> rerun/_vendor/eigen/
```

## Build & run (Windows / MSVC)
```
cmake -S rerun -B rerun/build -G "Visual Studio 17 2022" -A x64
cmake --build rerun/build --config Release
cd rerun
./build/Release/case_a_cnc.exe        # writes out/cnc_*.csv
./build/Release/case_b_hea.exe        # writes out/hea_manifold.csv
./build/Release/train_steel.exe       # trains models/steel_multi.{json,bin} (needed by Case C)
./build/Release/case_c_steel.exe      # writes out/steel_{manifold,pareto,multitarget,...}.csv
./build/Release/baselines.exe cnc     # honest baselines (reject +/- repair, ours) + feasible dumps
                                      #   + exact-target sweep (out/baselines_cnc_exact.csv)
./build/Release/baselines.exe hea     # also exports models/<case>_surrogate.py for pymoo
./build/Release/baselines.exe steel   # steel exact-target sweep (out/baselines_steel_exact.csv)
python pymoo_baselines.py cnc         # scalar min-F panel: GA/DE/PSO/CMA-ES, same repair + budget
python pymoo_baselines.py hea
python pymoo_baselines.py steel       # steel: fix UTS + maximize elongation (same repair + budget)
python pymoo_baselines.py cnc exact   # exact-target panel: GA/DE/PSO/CMA-ES minimize |f-c|,
                                      #   count solutions AT target + diversity (out/pymoo_cnc_exact.csv)
python pymoo_baselines.py steel exact # steel exact-target panel (out/pymoo_steel_exact.csv)
python analyze_extrapolation.py cnc   # extrapolation analysis -> figs/fig_extrapolation_cnc.pdf
```

## Figures (plots)
```
cd rerun && python gen_figs.py         # reads out/cnc_manifold.csv + out/hea_manifold.csv -> figs/*.pdf
```
Produces `figs/fig_cnc_tradeoff.pdf`, `figs/fig_cnc_manifold3d.pdf`,
`figs/fig_hea_parallel.pdf`; `analyze_extrapolation.py` writes `figs/fig_extrapolation_cnc.pdf`.

## Reproducibility (seed)
The C++ drivers call `opennn::set_seed(1)` at startup, so `baselines`, `case_a_cnc` and
`case_b_hea` are bit-deterministic (only wall-clock timing varies) and every reported number
traces to that run. The `pymoo` panel is seeded independently (`seed=1`).

## Status
Case A, Case B, Case C, the honest baselines (rejection +/- repair, ours, pymoo scalar + exact panels)
and the extrapolation analysis are complete and reproducible at the 12k-eval budget under seed=1.
Case C (low-alloy steel): fixed-UTS composition manifold (5100 alloys at T=27C), ductility-vs-cost
Pareto (8 alloys), nonlinear weldability + affine cost repair, and a second box-domain confirmation
of the exact-target collapse (rejection 553->0 as the band shrinks 1e-2->1e-6; ours stays populated).

## License
The original code, scripts and data of this repository are released under the **MIT License**
(see `LICENSE`). The vendored OpenNN headers under `rerun/_vendor/opennn/` remain under their
own license (GNU LGPL v3), as stated in their file headers.

## Funding
Funded by the European Union (Marie Skłodowska-Curie Grant Agreement no. 101169541 — NEUTEN).
Views and opinions expressed are however those of the author(s) only and do not necessarily
reflect those of the European Union or the European Research Executive Agency (REA). Neither
the European Union nor the granting authority can be held responsible for them.
