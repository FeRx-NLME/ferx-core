# `mbma_naproxen.csv` — provenance

Longitudinal model-based meta-analysis of naproxen vs placebo in osteoarthritis:
18 trials, 36 treatment arms, WOMAC pain over time.

## Source

Derived — mechanically, by `mbma_naproxen_source.py` — from
`naproxen_data_transformed_se.csv`, published as supplementary material to:

> Bracis C, Taneja A, Lyauk YK, Barcomb H, de la Peña A, Cellière G.
> **Model-Based Meta-Analysis With MonolixSuite: A Tutorial for Longitudinal
> Categorical and Continuous Data.**
> *CPT: Pharmacometrics & Systems Pharmacology* 2026;15:e70158.

*CPT:PSP* is fully open access and publishes under CC BY-NC-ND (see the article
page for the exact terms attached to this paper). The rows are themselves
aggregate summary statistics — per-arm means and their standard errors — extracted
by those authors from the published trial reports; there is no individual-level
data here.

Re-used here as a validation fixture, with attribution. If you redistribute it,
carry this file with it. To refresh or re-derive it, fetch the supplementary CSV
from the article and run:

```
python3 data/mbma_naproxen_source.py <path-to>/naproxen_data_transformed_se.csv
```

## What the columns mean

| Column | Meaning |
|---|---|
| `ID` | study (`STUD` upstream) — the unit of between-study variability |
| `ARM` | treatment arm within the study — the occasion for a BTAV `kappa` |
| `TIME` | weeks since randomisation |
| `DV` | `WP / WPSE` — the SE-weighted arm mean (`transWP` upstream) |
| `TRT` | 1 = naproxen, 0 = placebo |
| `FLARE` | 1 = flare-design trial |
| `WPSE` | reported standard error of that arm's mean at that time |
| `NIND` | subjects behind the arm (`Nindiv` upstream) |
| `MDV` | 0 throughout — every row is a scored observation |

`DV` and the prediction are **both** divided by `WPSE`, with the residual variance
FIXED to 1, so each arm enters the likelihood with weight `1/WPSE²`. That is the
MBMA weighting scheme, not a unit conversion — see `examples/mbma_naproxen.ferx`.

## Used by

- `examples/mbma_naproxen.ferx` — the shipped model.
- `tests/mbma_naproxen_case_study.rs` — reproduces the published estimates and
  checks the compartment-free form against the `d/dt(clock) = 1` workaround.
