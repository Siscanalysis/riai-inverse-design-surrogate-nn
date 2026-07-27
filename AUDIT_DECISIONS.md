# RAI paper — audit decisions ledger (ground truth)

Pre-submission internal-consistency audit (passes B–J; no reviewer round yet).
Created 2026-07-27. This file is authoritative: recorded decisions are settled;
do not re-litigate. Amending a decision is an explicit, dated action.

Decision vocab: **implement** (edit manuscript) · **verify** (confirm it survived) ·
**defer** (limitations/future-work + note) · **response-only** (n/a pre-review) ·
**reword** · for refs: keep / fix-field / replace / merge / add / remove.

| ID | Topic | Source | Finding | Decision | Status | Date |
|----|-------|--------|---------|----------|--------|------|
| B-1 | tab:mixed + mixed-sense para stale | Pass B | ours 78.0 (ms) vs 79.12 (CSV); Rechazo 86.6 vs 85.69. Unseeded C++ RNG => best-F drifts run to run | OPEN — needs user (seed+lock vs update+caveat) | open | 2026-07-27 |
| B-2 | Reproducibility: unseeded RNG | Pass B/C | manuscript numbers, figures, out/*.csv are 3 non-matching snapshots | OPEN — recommend seed the C++ drivers, rerun, lock, regenerate figs | open | 2026-07-27 |
| B-3 | Case-study budgets undisclosed | Pass B/I | 3400 recipes / 5100 intersection / 170 alloys from case_a,b at larger (undisclosed) budgets; Sec 5 uses 12000 | OPEN — disclose case-study budget | open | 2026-07-27 |
| F-1 | R undefined in selection blend | Pass F | Sec 2.4 defines S (eq:sens) and C, but the geometric mean uses R without defining R = normalized/inverted S | implement — add one sentence defining R | proposed | 2026-07-27 |
| J-1 | wen2019 missing DOI | Pass J | authors/venue/vol/pages VERIFIED correct; DOI absent | fix-field — add DOI 10.1016/j.actamat.2019.03.010 | proposed | 2026-07-27 |
| J-2 | canal2022 unverified | Pass J | Kaggle dataset exists (uploader "adorigueto"); could NOT verify authors "Canal, G.; Borille, A.V." nor that DOI resolves | verify — user confirm exact citation on Kaggle page | open | 2026-07-27 |
| J-3 | scala2026idc unpublished companion | Pass J | cited as "Manuscrito companion"; no venue/arXiv | defer — ensure arXiv/preprint id before submission | open | 2026-07-27 |
| G-1 | Spanish diacritics missing | Pass D/G | entire manuscript is unaccented Spanish (diseno, optimizacion...); RIAI is a Spanish journal expecting proper orthography | OPEN — needs user (add accents vs leave) | open | 2026-07-27 |
| G-2 | Length vs 12-page framing | Pass G | manuscript 6 pp (2-col RIAI); editor email frames it as "extension al limite de 12 paginas" | OPEN — needs user (expand vs accept) | open | 2026-07-27 |
| G-3 | Author emails placeholder | Pass G | \ead COMPLETAR@artelnics.com x2; S. Scala = Simone Scala (per email signature) | OPEN — user fills real addresses | open | 2026-07-27 |
| C-1 | Figures predate latest data | Pass C | all figs <=07-24; baselines rerun since (unseeded). Regenerate after numbers locked | tie to B-2 | open | 2026-07-27 |
| C-2 | Orphan figure files | Pass C | fig_baselines_feasibility/tradeoff, fig_hea_projection unreferenced | remove (harmless) — optional | proposed | 2026-07-27 |
| E-1 | MRR used before definition | Pass E | MRR acronym in Sec 2.3 (line 164) before expansion in Sec 3 (line 211) | reword — expand at first use | proposed | 2026-07-27 |
| E-2 | "caso 3" vs "Caso A" | Pass E | \ref{sec:cnc} yields "3" while section named "Caso A" | reword — "Caso A (Sec. 3)" | proposed | 2026-07-27 |
| H-1 | "reparacion es el habilitador" repeated | Pass H | stated ~4x (abstract, Finding 1, mixed-sense, concl) | note — mild; optional trim | open | 2026-07-27 |
| I-1 | HEA "170 rich" vs "pocas aleaciones" | Pass I | Sec B (170, large budget) vs Sec 5 exact (few, 12k vanishing band); budget diff undisclosed | tie to B-3 — add budget clause | open | 2026-07-27 |

## Decisions taken 2026-07-27 (seeding round)
- **B-2 RESOLVED (implement):** added `set_seed(1)` to baselines.cpp, case_a_cnc.cpp, case_b_hea.cpp
  (`opennn::set_seed`). baselines + case_a now bit-deterministic (only wall-clock `seconds` varies).
  All manuscript numbers re-locked to the seed=1 run. Reproducibility section to cite seed + commands.
- **B-1 RESOLVED (implement):** tab:mixed + tab:base + tab:exact + Case A + extrapolation all updated to
  seed=1 values (see LOCKED NUMBERS below).
- **B-4 NEW (implement):** HEA extrapolation claim REVERSES under seed=1 — ours d2train 0.253 vs reject
  0.220 (ours farther), because ours recovers only 3 HEA points => distribution metric is noise. The
  manuscript's "nuestro 0.16 frente a 0.22" is NOT reproducible. DECISION: drop the HEA extrapolation
  win; keep only the robust CNC result (ours edge 0.5%, d2train 0.156). Honest.
- **B-5 NEW (defer/bug):** case_b_hea.exe crashes at startup (0xC0000409) under the current vendored lib
  (case_a + baselines run fine with the same seed => case_b-specific, likely pre-existing after the lib
  refactor; case_b lacks `unitbuf` so buffered output is lost). Case B numbers kept from the committed
  out/hea_manifold.csv (170 alloys, 44-100% fill, HV~546, |sum-1|=0). TODO: debug case_b, then re-lock.
- **G-1 (implement):** full Spanish diacritics pass applied to manuscript.
- **G-2 (defer/plan):** expand content toward the ~12-page extension framing — planning item, see below.
- **F-1, J-1, E-1, E-2 (implement):** applied in the same rewrite.

## LOCKED NUMBERS (seed=1) — ground truth for the manuscript
- tab:base CNC: Rechazo-no 7.5% / 99.6% / minS 0.092 ; Rechazo-si 21.5% / 99.9% / **0.067** (now flattest) ;
  Nuestro 3.1% / 72.3% / 0.108. (cobertura text 79%->72%.)
- tab:mixed: Nuestro **78.1**, DE 78.0, PSO 78.0, GA 87.5 (+12%), Rechazo **93.5** (+20%), CMA-ES 114.8 (+47%).
- tab:exact (delta=1e-4): Nuestro 1700 / 18% / 2e-7 ; Rechazo 8 / 75% / 6e-6 ; DE 28/90%/4e-8 ; PSO 83/34%/2e-8 ;
  GA 100/0%/2e-13 ; CMA-ES 63/3%/1e-6. Collapse 941->100->8->2->0 ; ours 1700->1308 at 1e-6 ; reject floor 6e-6.
- Case A: Ra target 0.6213 ; manifold 3400 ; Ra band [0.605,0.638] ; F range **[95,402]** N ; MRR **[8.0,27.5]** ;
  alpha wall-margins **0.044 / 0.510 / 0.580** (a=1/0.5/0) ; min-F 77.9~78 ; intersection 5100.
- Case B (committed artifact): 170 alloys ; fill **44-100%** ; HV~546 ; |sum-1|=0.
- Finding-1 HEA simplex: reject-norepair 0.979(~0.98) ; ours 7.6e-8 ; pymoo ~1e-17.
- CNC extrapolation: ours edge<0.1 = **0.5%** vs reject **25-31%** ; ours d2train **0.156** vs reject 0.184-0.190.
- HEA extrapolation: DROPPED (B-4).

## G-2 expansion plan (proposed content to reach a fuller journal extension)
Candidates (pick with user): (a) Related work / positioning section (inverse design, surrogate optimization,
constraint handling) — currently absent; (b) formal method subsection (contraction pseudocode + repair-regime
table + complexity/eval-budget accounting); (c) a third case or a geopolymer/HEA-6elt variant; (d) ablation:
robustness alpha sweep + band-width sensitivity; (e) deeper baseline discussion (add a dedicated NLP/MO method);
(f) limitations + threats-to-validity section. Recommend (a)+(b)+(f) as the highest-value, lowest-risk additions.

## Verified-consistent (no action)
- tab:base (CNC) matches baselines_cnc.csv: 7.6/99.8/0.055, 23.1/99.96/0.094, 4.2/78.6/0.278. (Aggregate stats; stable.)
- tab:exact fully matches baselines_cnc_exact.csv + pymoo_cnc_exact.csv (generated this session).
- Text collapse 920->103->12->2->0 and 1700/877 match CSV.
- Finding-1 simplex residuals (ours 8.4e-8, pymoo ~1e-17, reject 0.98) match baselines_hea.csv + pymoo_hea.csv.
- wen2019 authors/year/venue/volume/pages verified against web.
- Title differs strongly from SCI 2026 "Diseno industrial optimo con redes neuronales" (>20% requirement met).
- All \ref/\label resolve; paper compiles 6 pp, no undefined refs, no overfull >20pt.
