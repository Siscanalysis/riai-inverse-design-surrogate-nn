# Data sources, licences and attribution

Three measured datasets. Two ship in `code/data/`; the third cannot be redistributed and has to
be downloaded by whoever wants to rebuild it.

## concrete — ships in this repository

`code/data/concrete.csv`, derived from:

> Lu, K., Qiu, M. (2024). *Dataset of compressive strength and slump of normal concrete.*
> Mendeley Data, V1. DOI [10.17632/zrsbhndz9f.1](https://doi.org/10.17632/zrsbhndz9f.1).

Licensed **CC BY 4.0**. Redistributed here under that licence, with changes: the columns used by
the study were selected and renamed, rows with missing entries were dropped, and a grouped
train/validation/test split was added in `code/data/concrete_split.csv`. `code/prep_data.py`
performs those transformations and is the record of them.

## kernel — ships in this repository

`code/data/kernel.csv` and `code/data/kernel_measured.csv`, derived from:

> Hozzová, J., Filipovič, J., Nezarat, A., Oľha, J., Petrovič, F. (2021). *Searching CUDA code
> autotuning spaces with hardware performance counters: data from benchmarks running on various
> GPU architectures.* Data in Brief 39, 107631.
> DOI [10.1016/j.dib.2021.107631](https://doi.org/10.1016/j.dib.2021.107631).
> Data: Mendeley Data V1, DOI [10.17632/nn53dskr7z.1](https://doi.org/10.17632/nn53dskr7z.1).

Licensed **CC BY 4.0**. Redistributed here under that licence, with changes: only the convolution
kernel measured on the GTX 1070 is kept (`1070-conv_output.csv`), the eight tuning parameters and
two responses used by the study were selected, and the duration was replaced by its base-10
logarithm because it spans a factor of 68. `code/prep_data.py` performs those transformations.

The related autotuning method paper is Filipovič et al. (2022), *Journal of Parallel and
Distributed Computing* 160, 16-35, DOI
[10.1016/j.jpdc.2021.10.003](https://doi.org/10.1016/j.jpdc.2021.10.003).

## steel — NOT redistributed

`code/data/steel.csv` is **deliberately absent** and is listed in `.gitignore`.

Its source is the NIMS Materials Database (MatNavi), National Institute for Materials Science,
Japan, <https://mits.nims.go.jp/>. The MatNavi Service Terms of Use reserve NIMS's copyright in
the database and do not permit reproduction, republication or distribution of its content to third
parties without written permission from NIMS, so this repository does not carry the table.

To rebuild it, register for MatNavi, export the low-alloy steel mechanical-property records
yourself, place the table where `prep_data.py` expects it, and run:

```
set RAI_BENCHMARK_DATA=<root that holds so_realdata/>
python code/prep_data.py
```

What you can still do without it:

- **every optimization experiment and every baseline**, because they all evaluate the trained
  surrogate in `code/models/steel.json` plus `steel.bin`, never the raw table;
- **every table and figure in the paper**, from the committed metrics in `code/out/`;
- **`audit_numbers.py`**, likewise.

What needs the table: retraining the steel surrogate, `check_spec_error.py` for steel, and the
data-support columns of `analyze.py` for steel.

The trained surrogate is a derived model, not a copy of the data, and is distributed here on that
basis. If NIMS considers any part of this repository to exceed their terms, open an issue and it
will be removed.

## Software

OpenNN (LGPL) is the library under test; the exact version, and the measurement patch applied to
it, are described in `code/opennn_pin/README.md`. The surrogates were trained with Neural
Designer; see the provenance section of the main README. Everything written for this study is MIT
licensed (`LICENSE`).
