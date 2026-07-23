# RIAI extension — plan

Extension of SCI 2026 paper **246 "Diseño industrial óptimo con redes neuronales"** (Scala, Lopez)
to a 12-page RIAI journal article. Must be substantially different in title and content
(<20% similarity), submitted via the RIAI platform before end of December 2026.
A prior email to jlcalvo@udc.es must (a) get the new title approved, and (b) precede the
platform upload announcing this is a Jornadas de Automática extension.

## Working title (pending approval)
"Problemas inversos con objetivos mixtos y restricciones de igualdad sobre redes neuronales sustitutas"

## What makes it NON-overlapping with SCI 2026
| SCI 2026 (short paper) | This RIAI extension |
|---|---|
| IDC (iterative domain contraction) | Not the focus |
| Affine repair of LINEAR constraints | Broader: equality-target + mixed senses |
| One optimum (SO) / Pareto front (MO) | The whole FEASIBLE SOLUTION MANIFOLD |
| photo_wf3, yacht | CNC turning AISI 304 + (Case B: HEA/simplex or geopolymer) |
| — | Jacobian ROBUSTNESS + centrality selection framework |
| — | Mixed-sense (Minimize / Maximize / Fixed on same problem) |
| — | Baselines: rejection sampling + NSGA-II (feasibility/diversity/robustness) |

## Method contribution (unified)
Mixed-sense response optimization on NN surrogates:
- per-output goal in {Minimize, Maximize, Fixed(=c)}
- Fixed(=c) equality targets → recover the feasible solution *manifold*, not a single point
- multi-target: pin ≥2 outputs → intersection of manifolds
- robustness: S(x) = Σ_i (∂f/∂x_i · span_i)^2 (analytic Jacobian, span-weighted)
- selection: centrality (margin to box walls) vs robustness dial, get_robust_point(front, balance)

## Case A — CNC turning AISI 304  (so_realdata/Q_materials_so/cnc_turning_aisi304)
- Decision levers (numeric): cutting speed vc, feed f, depth of cut ap  (+ tool wear VB as numeric input, 0/0.1/0.3)
- Retrain MULTI-OUTPUT surrogate: (vc,f,ap[,VB]) -> (Ra, F_resultant).  MRR = vc*f*ap analytic.
- Mixed-sense: FIX Ra = 1.6 µm (N7 class) ; MIN F ; MAX MRR
- Multi-target: FIX Ra AND FIX F -> curve
- 3 continuous levers -> fixed-Ra feasible set is a 2-D surface in 3-D (direct visualization, no Isomap)

## Case B — HEA hardness simplex manifold  (RESOLVED by scout, HEA data recovered)
- Dataset: hea_hardness_rao_wei (data now present: data/source/hardness_raw.csv, 370+ rows)
  Open mirror: github.com/dovahkiin0022/representations dataset/hardness.csv
- Use the 6-element AlCoCrCuFeNi subset (~236 rows): parse at% formulas -> fractions summing to 1 (TRUE simplex)
- Inputs: 6 composition fractions (Al,Co,Cr,Cu,Fe,Ni), all-numeric, sum=1 affine equality
- Output: Vickers hardness HV (109-920). Verified surrogate R2 ~= 0.94 (MLP 5-fold CV on subset)
- Case B story: FIX HV = target -> recover alloy-composition manifold on the simplex;
  robustness/centrality selection; parallel-coordinates exploration (high-dim -> Isomap ok)
- Fallback (unused): geopolymer R2=0.976 with imposed affine constraint
- TODO in build block: formula->fractions parser (6-elt subset CSV) + train multi/single-output surrogate

## Baselines
- Rejection sampling (uniform box + keep feasible): feasibility rate, diversity, robustness of retained set
- NSGA-II (pymoo) treating |f(x)-c| as objective / constraint: coverage of the manifold vs our method
- Metrics: feasibility %, spread/diversity, mean robustness, wall-clock, #NN evals

## Assets to reuse
- SCI rerun harness: Papers/DONE/Simposio_CEA_2026/rerun (C++ CMake drivers, pinned surrogates, python baselines)
- bho prototype: C:/Users/Artelnics/Desktop/bho (main.cpp Fixed-objective solver, get_robust_point, build_explorer.py, knn_stepper.py, explorer.html)
- OpenNN: ResponseOptimization::Sense::Fixed + get_robust_point (recent commits)

## Deliverables status
- [x] Title-proposal email drafted (email_propuesta_titulo.txt)
- [ ] Title approved by coordinator  <-- BLOCKS manuscript writing
- [ ] RIAI LaTeX template in place
      Template = Elsevier `elsarticle.cls [5p,times,authoryear]` + `ecrc_RIAI.sty` + `elsarticle-harv.bst`,
      main file `ejemplo_latex_RIAI.tex`. Overleaf: https://www.overleaf.com/latex/templates/plantilla-para-revista-iberoamericana-de-automatica-e-informatica-industrial-riai/hbxybyszgjzy
      MANUAL STEP: open on Overleaf -> Menu -> Download -> Source (zip), drop contents into RAI/.
      (elsarticle.cls + elsarticle-harv.bst ship with MiKTeX; only ecrc_RIAI.sty is RIAI-specific.)
      SCI paper used the sibling sciarticle.cls, so the front-matter port is small.
- [ ] CNC multi-output surrogate trained
- [ ] Case B dataset resolved (scout)
- [ ] Experiments (A, B, baselines)
- [ ] Manuscript
