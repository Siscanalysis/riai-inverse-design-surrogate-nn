# OpenNN version used by the experiments

The experiments are pinned to the OpenNN `dev` branch at commit **e93d5479a**
("Consolidate response repair and avoid redundant inference", 2026-09-22) plus the
working-tree changes that were present on that day in the response-optimization module
(`uncommitted_on_e93d5479a.patch`: the `set_sampling_rounds` / `set_maximum_consecutive_failures`
setters, the two-cycle stop of the repair loop, and the box scales fixed at the first repair
round). Together they form the snapshot `89be7b7` (a `git stash create` object of that day).

## Rebuilding the snapshot

```
git -C <opennn> worktree add C:\rai_onn\src e93d5479a          # or: git archive e93d5479a | tar -x
git -C C:\rai_onn\src apply <this folder>\uncommitted_on_e93d5479a.patch
git -C C:\rai_onn\src apply <this folder>\forward_counter.patch   # see below
pwsh code\build.ps1                                                # builds the lib + drivers
```

`code/CMakeLists.txt` adds the snapshot as a subproject (`OPENNN_SOURCE_DIR`, default
`C:/rai_onn/src`), so the drivers compile with exactly the flags and the Eigen copy of the
library. Tests, examples and vision layers are off, LTO is off, CUDA is disabled. Eigen 5.0.1
is resolved by `find_package(Eigen3 5.0 CONFIG)`; on the build machine that found the
FetchContent copy of another local OpenNN build tree (same version, 5.0.1).

## Measurement instrument (does not change any computation)

`forward_counter.patch` adds one global counter, `opennn::rai_forward_rows`, incremented by
the batch size in `Network::calculate_outputs(const vector<TensorView>&)`, the function every
CPU inference call of the library ends in (the feasibility repair, its finite differences,
and the objective evaluations). The drivers reset it before a run and read it after, which
is how every OpenNN run reports the surrogate forward passes it spent and how the Python
baselines are given the same budget.
