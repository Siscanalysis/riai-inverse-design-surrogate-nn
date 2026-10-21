# Inverse design with mixed objectives and equality constraints on surrogate neural networks

Reproducibility repository for the RIAI (Revista Iberoamericana de Automática e Informática
Industrial) article *"Problemas inversos con objetivos mixtos y restricciones de igualdad sobre
redes neuronales sustitutas"* (S. Scala, R. López), an extension of the SCI 2026 symposium paper
*"Diseño industrial óptimo con redes neuronales"*.

**This repository contains only the reproducible experiments**: the code, the data, the trained
surrogates, the metrics and the scripts that regenerate every figure, table and number in the
article. The manuscript itself is not included here.

## What the study is about

An inverse-design specification is a network output that must take a given value, `f_k(x) = c`.
The paper is about **how that specification is enforced**, and about the fact that two families of
algorithms enforce it in a different order:

- a **gradient method** solves the continuous relaxation and rounds at the end, which is all it can
  do when a design variable is restricted to a finite catalogue, because no gradient exists over a
  discrete set;
- the **OpenNN response-optimization module** repairs each candidate *inside* the search loop: it
  rounds the discrete conditions first and then re-projects onto the specification through the
  network itself with a Levenberg-Marquardt step, so every point it returns is both admissible and
  compliant.

On a finite catalogue that order decides the outcome, because the optimum of an objective subject
to a specification lies on the boundary of the feasible region and rounding at the end leaves it.
On the CUDA kernel case the library reaches the enumerated optimum in all ten seeds while relaxed
and rounded SQP returns no valid point in any, and the one metaheuristic that matches it costs four
times more. On the two continuous cases there is no catalogue to round and a classical local solver
is cheaper than the repair; the paper reports that as well, and so do the metrics here.

The specification is stated three ways throughout: as an **equality** constraint, as a **band**
around the requested value, and as a **fixed-sense objective** scored `-|f_k(x) - c|`.

## Cases

| case | data | inputs -> outputs | hidden | test R<sup>2</sup> (spec / others) | surrogate error on the spec |
|---|---|---|---|---|---|
| concrete | ready-mix plant ledger, 1395 rows / 623 mixes | 9 -> strength, slump | 12 | 0.85 / 0.54 | 7.6 % |
| steel | NIMS MatNavi low-alloy steels, 915 tests / 95 alloys | 14 -> yield, UTS, elongation, RA | 16 | 0.87 / 0.85-0.74 | 4.6 % |
| kernel | KTT CUDA convolution auto-tuning, 3311 configurations on one GTX 1070 | 8 catalogues -> log10 duration, achieved occupancy | 24 | 0.99 / 0.88 | 5.8 % |

Splits are grouped a priori: a mix, an alloy or a configuration never appears in two roles. The
last column is the median relative error of the specified output on held-out rows within 25 % of
the requested value, recomputed by `check_spec_error.py`. It is the floor below which a tolerance
measures consistency with the model rather than accuracy of the part, and it sits four to six
orders of magnitude above the tolerances the optimizers are judged at.

The kernel catalogue has 4680 admissible configurations, so its solution set and its optimum are
known exactly by enumeration. That is why the case was chosen, and the paper says plainly that on
*this* catalogue enumerating is cheaper than searching: the claim is about catalogues too large to
enumerate, where the alternatives are relax-and-round and population search.

**Data licences differ, and one dataset is not redistributed here. See [DATA.md](DATA.md).**

## Layout

```
code/
  prep_data.py          source tables -> data/ (numeric tables + grouped a-priori splits)
  src/train_surrogates.cpp   trains and selects a surrogate      -> models/
  src/cases.h           every problem statement, in one place
  src/inverse_cases.cpp the OpenNN runs: set recovery (sample) and optimization (solve)
  run_library.ps1       all OpenNN runs of one case, seeds 1-10
  run_matched.py        the objective formulation at the budget of the constraint formulation
  baselines.py          rejection, SLSQP (relaxed / rounded / scalarized), GA/DE/PSO/CMA-ES,
                        NSGA-II and enumeration, at the same budgets
  surrogates.py         float64 evaluation of the networks from their OpenNN Python export
  problems.py           independent statement and re-checking of every returned point
  analyze.py            metrics -> out/metrics_*.csv, out/summary.json
  check_kernel.py       the enumerated catalogue, recall/precision, measured-duration contrast
  check_spec_error.py   each surrogate's error on its own specification -> out/spec_error.json
  check_fit.py          the case table's architecture and test R^2 columns -> out/fit.json
  check_isolation.py    the a-priori isolation test of the catalogue      -> out/isolation.json
  check_plausibility.py the concrete verisimilitude and data-support filter
  make_tables.py        the article's result tables
  audit_numbers.py      checks every number quoted in the article against the stored metrics
  figs.py               the article's figures -> figs/
  data/ models/ out/ figs/   tables, trained surrogates, metrics, figures
  nd/                   the Neural Designer projects, as provenance (see below)
  opennn_pin/           the exact OpenNN version, its patch and the measurement instrument
  probe/                a later OpenNN's analytical Jacobian, measured (README inside)
```

## Honest accounting

Every method in the comparison is held to the same budget and the same acceptance rule:

1. every method evaluates **the same network**, exported to Python and evaluated in float64,
   agreeing with the C++ network to better than 1e-6;
2. every method gets the same budget, counted in **network forward passes** rather than
   iterations, by a counter patched into the pinned OpenNN that only measures
   (`code/opennn_pin/`);
3. every method stops by the **same acceptance rule**, a relative residual of 1e-6 on the
   specification, so none pays for accuracy another does not;
4. the specification is given to every method with the **same band**, not a band for the library
   and an equality for the others;
5. every returned solution is **re-checked constraint by constraint** in double precision by
   `problems.py`, and only what passes is counted;
6. the catalogue **enumeration is ground truth, never a competing row**;
7. where the library loses, the numbers are reported.

The metaheuristics run with their default configuration, which is a disadvantage to them and is
stated as such in the article.

## Reproduce

Windows, Visual Studio 2022, Python 3 with numpy, scipy, pandas, pymoo 0.6.1 and matplotlib.

```
pwsh code/build.ps1                   # needs the pinned OpenNN: code/opennn_pin/README.md
cd code
bin\inverse_cases.exe problems
pwsh run_library.ps1 concrete         # likewise steel, kernel
python run_matched.py concrete        # likewise steel, kernel
python baselines.py concrete all      # likewise steel, kernel
python analyze.py
python check_kernel.py
python check_spec_error.py            # steel needs the MatNavi table, see DATA.md
python check_fit.py                   # likewise for the steel row
python check_isolation.py
python check_plausibility.py
python make_tables.py
python figs.py
python audit_numbers.py               # must print "0 mismatch(es), 0 without a metric"
```

Seeds are fixed (1 to 10); wall-clock times vary. The trained surrogates are committed, so
`train_surrogates.exe` is only needed to rebuild them from scratch. The metrics under `code/out/`
are committed too, and so are the seed-1 point files that `figs.py` opens, so **the tables, the
figures and the number audit all run from this checkout** without repeating any search. The rest of
the per-point output, several hundred megabytes of it, is not tracked.

`audit_numbers.py` checks 107 quantitative claims from the article against the stored metrics,
including the case table's architecture and R² columns and the isolation sweep. It exits non-zero
on any mismatch, and prints `MISS` rather than passing silently when a metric file is absent.

## Surrogate provenance

The surrogates were trained with [Neural Designer](https://www.neuraldesigner.com) and imported
into OpenNN model files by `import_nd_network.py`. `code/models/<case>.json` is byte-identical to
`code/models/<case>_nd.json`, so what every experiment loads is exactly what the import produced.

The `.ndm` projects under `code/nd/` are included as provenance, **not** as a reproduction path:
`run_nd.py` invokes a local engine build whose version this repository does not record, and Neural
Designer has changed since. Reproduce from the OpenNN model files in `code/models/`, which is what
every driver and every baseline reads. `code/nd/steel/` is absent for the licence reason in
[DATA.md](DATA.md).

Two sets of files are kept for the record and are **not** the networks used by the study:
`code/models/opennn_trained/` and `code/out/surrogate_*.txt` come from an earlier neuron search run
with OpenNN's own trainer, which the Neural Designer models superseded. They disagree with the
article on purpose: `surrogate_steel.txt` reports 8 hidden neurons where the article reports 16,
because the article describes the Neural Designer network in `code/models/steel.json`.

## Licence and funding

Code, scripts and derived metrics: MIT, see [LICENSE](LICENSE). Data: see [DATA.md](DATA.md), the
licences differ per dataset. OpenNN is LGPL.

Funded by the European Union (Marie Skłodowska-Curie Grant Agreement no. 101169541, NEUTEN). Views
and opinions expressed are those of the authors only and do not necessarily reflect those of the
European Union or the European Research Executive Agency (REA). Neither the European Union nor the
granting authority can be held responsible for them.
