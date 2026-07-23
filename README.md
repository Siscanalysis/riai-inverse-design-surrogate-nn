# Inverse design with mixed objectives and equality constraints on surrogate neural networks

Companion repository for the RIAI article
*"Problemas inversos con objetivos mixtos y restricciones de igualdad sobre redes
neuronales sustitutas"* (S. Scala, R. Lopez). Extension of the SCI 2026 symposium
paper *"Diseno industrial optimo con redes neuronales"*.

Inverse design is treated as recovering the **feasible solution manifold** defined by
an equality (fixed-target) constraint on a trained surrogate, with **mixed-sense**
objectives (minimize / maximize / fixed on the same problem) and a **robustness +
centrality** selection over the manifold.

## Layout
```
paper/            manuscript.tex (+ manuscript.pdf), figs/ (gen_figs.py -> PDFs)
rerun/            reproducible experiments
  case_a_cnc.cpp    Case A - CNC turning AISI 304 (fix Ra / min F / max MRR)
  case_b_hea.cpp    Case B - HEA hardness on the composition simplex (fix HV)
  models/           trained surrogates: each <name>.json REQUIRES its <name>.bin sibling
  data/             the numeric CSVs the surrogates were trained on
  out/              CSVs produced by the drivers (gitignored; regenerated)
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
./build/Release/baselines.exe cnc     # honest baselines (reject +/- repair, ours, NSGA-II)
./build/Release/baselines.exe hea     # also exports models/<case>_surrogate.py for pymoo
python pymoo_baselines.py cnc         # GA/DE/PSO/CMA-ES, same repair + budget (pymoo 0.6)
python pymoo_baselines.py hea
```

## Figures
```
python paper/figs/gen_figs.py          # reads rerun/out/*.csv -> paper/figs/*.pdf
```

## Manuscript
```
cd paper && pdflatex manuscript.tex && pdflatex manuscript.tex
```
The draft uses the stock `elsarticle` class so it compiles as-is; for submission,
swap in the RIAI `ecrc_RIAI.sty` front-matter package.

## Status
Case A and Case B are complete and reproducible. The baselines section
(rejection sampling + NSGA-II) is pending.
