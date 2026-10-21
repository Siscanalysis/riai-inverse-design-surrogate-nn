# Probe: the analytical network Jacobian of the newer OpenNN

The paper is pinned to OpenNN `89be7b7` (see `../opennn_pin/README.md`), whose feasibility repair
builds its constraint Jacobian by finite differences. A later version computes it analytically
(`26347f3e7`, corrected in `1e4eceb0d`). This probe answers whether that is a new result, which
would require rerunning the study, or an implementation saving, which would not.

Answer: an implementation saving. The repair returns the same number of feasible designs and the
same quality; it gets about twice cheaper in network sweeps. Full reading in
`../../notes/DECISIONS.md` (2026-10-06) and `../../AUDIT_DECISIONS.md` (Round 9, V-1 to V-6).

## Two things had to be fixed before any measurement meant anything

**The counter was blind to the new path.** `rai_forward_rows` is incremented inside
`Network::calculate_outputs`. `FeasibilityRepairSystem::calculate_network_jacobian` walks the
layers itself and never calls it, so an analytical Jacobian costs zero on the paper's meter. On
steel the identical work reads 375 912 units with finite differences and 50 588 with the
analytical Jacobian: a counter left unchanged would have credited a saving three and a half times
larger than the real one. `probe_instrument.patch` therefore adds two further counters and the
probe reports three raw quantities, never one pre-weighted total:

| column | meaning |
| --- | --- |
| `forwards` | inference rows through `Network::calculate_outputs`, the paper's unit |
| `jac_calls` | analytical Jacobians formed, none of which reach `calculate_outputs` |
| `jac_sweeps` | the layer sweeps those cost, charged as 1 value recomputation + 1 per network output |

Raw counts rather than a single total, so that a different weighting can be argued later without
running anything again. The charge of `1 + outputs` sweeps is conservative against our own method:
it over-charges the analytical path relative to measured wall clock.

**The default repair budget changed**, `feasibility_rounds` from 3 to 1. Left to the defaults the
two builds would have differed in budget, not in derivative. `probe_sample.cpp` sets rounds and
evaluations explicitly to the pinned values.

## Design

The two arms live in **one binary**, switched by `RAI_FORCE_NUMERICAL_JACOBIAN`, so they cannot
differ in compiler, flags, machine or library version:

| arm | Jacobian |
| --- | --- |
| `fd` | finite differences, the pinned version's method |
| `analytical` | the analytical network Jacobian |

The pinned build is a third arm, already in `../out/runs_<case>.csv`. It also carries the other
thirteen commits of the interval, so it answers a different question and is read as context only.

## Reproducing

```
git -C <opennn> archive 1e4eceb0d | tar -x -C C:\rai_onn_new\src
cd C:\rai_onn_new\src && git apply <this folder>\probe_instrument.patch   # or patch -p1
pwsh code\probe\build.ps1                     # separate build tree and bin; code\bin untouched
cd code
foreach ($case in "steel","concrete") { foreach ($seed in 1,2) {
    $env:RAI_FORCE_NUMERICAL_JACOBIAN="1"; .\probe\bin\probe_sample.exe $case $seed 2000
    Remove-Item Env:\RAI_FORCE_NUMERICAL_JACOBIAN; .\probe\bin\probe_sample.exe $case $seed 2000 } }
python probe\check_probe.py                   # independent double-precision re-check
```

Results land in `../out/probe/runs.csv` and `../out/probe/<case>_<arm>_s<seed>.csv`. Nothing here
writes to `../out/sample`, `../out/solve`, `../out/summary.json` or `code/bin`, so the paper's
numbers cannot be overwritten by running the probe.
