# `mbma_naproxen.csv` — provenance

Longitudinal model-based meta-analysis of naproxen vs placebo in osteoarthritis:
18 trials, 36 treatment arms, WOMAC pain over time.

> **Licence: CC BY-NC 4.0 — not MIT.** `mbma_naproxen.csv` is the one exception to
> the repository's MIT grant (this README is not — it is repo-authored MIT prose).
> See [`mbma_naproxen.LICENSE`](mbma_naproxen.LICENSE) beside it, and the note in
> the root [`LICENSE`](../LICENSE). Commercial use of the CSV is not granted. It is
> a test fixture, never linked into the binary, and `Cargo.toml` excludes it from
> the packaged crate.

## Source

Derived — mechanically, by `mbma_naproxen_source.py` — from
`PSP4-7-288-s007.csv`, published as supplementary material to:

> Boucher M, Bennetts M.
> **Many Flavors of Model-Based Meta-Analysis: Part II — Modeling Summary Level
> Longitudinal Responses.**
> *CPT: Pharmacometrics & Systems Pharmacology* 2018;7(5):288–297.
> [doi:10.1002/psp4.12299](https://doi.org/10.1002/psp4.12299) · `PMC5980518`

That article is open access under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — attribution
required, no commercial use, **no ND clause**. The rows are aggregate summary
statistics — per-arm means and their standard errors — extracted by those authors
from the published trial reports; there is no individual-level data here.

To refresh or re-derive it, fetch the supplementary CSV from the article and run:

```
python3 data/mbma_naproxen_source.py <path-to>/PSP4-7-288-s007.csv
```

The supplementary files are downloadable without a proof-of-work challenge from
Europe PMC:

```
curl -sSL -o supp.zip \
  https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5980518/supplementaryFiles
unzip -o supp.zip PSP4-7-288-s007.csv
```

The upstream file is 3,709 bytes, `sha256`
`c3753cea5dd664c19b32b11117d9baafdef44a81e2808df1582f49e3b69d9190`; the derived
CSV committed here is `sha256`
`4370af48bfda41872f4b7f09a44a4a31c72c80582099c58aa67c965c84641ab4`. If a
re-derivation does not reproduce the second hash, the difference is a finding.

### A note on Bracis et al.

The same dataset also appears, transformed, as supplementary material to Bracis
C, Taneja A, Lyauk YK, Barcomb H, de la Peña A, Cellière G, *Model-Based
Meta-Analysis With MonolixSuite: A Tutorial for Longitudinal Categorical and
Continuous Data*, **CPT:PSP 2026;15:e70158**. That tutorial replicates Boucher &
Bennetts' Case Study analysis in MonolixSuite, and its `populationParameters.txt`
is where the Monolix comparison values in
`tests/mbma_naproxen_case_study.rs` come from — but it is **not** the source of
the data, and it is licensed CC BY-NC-**ND**, whose no-derivatives clause forbids
distributing a transformed copy. This fixture is therefore derived from the
primary source only (see #1085).

The two derivations agree to the last committed digit on 121 of 122 rows. The one
difference is a double-rounding artifact in the tutorial's precomputed `transWP`
column, which stores `WP / WPSE` to 7 decimals: for study 3, naproxen, week 2,
`2.99 / 0.188083087 = 15.897229504…`, which correctly rounds to `15.897230`,
while the stored `15.8972295` is the float64 `15.89722949999…` and rounds down.
Computing the ratio from `WP` and `WPSE` avoids the intermediate.

## What the columns mean

| Column | Meaning |
|---|---|
| `ID` | study (`STUD` upstream) — the unit of between-study variability |
| `ARM` | treatment arm within the study (`NFLG` upstream) — the occasion for a BTAV `kappa` |
| `TIME` | weeks since randomisation |
| `DV` | `WP / WPSE` — the SE-weighted arm mean |
| `TRT` | 1 = naproxen, 0 = placebo (`NFLG` upstream, read as a covariate) |
| `FLARE` | 1 = flare-design trial |
| `WPSE` | reported standard error of that arm's mean at that time |
| `NIND` | subjects behind the arm (`NTRT` upstream) |
| `MDV` | 0 throughout — every row is a scored observation |

`DV` and the prediction are **both** divided by `WPSE`, with the residual variance
FIXED to 1, so each arm enters the likelihood with weight `1/WPSE²`. That is the
MBMA weighting scheme, not a unit conversion — see `examples/mbma_naproxen.ferx`.
Boucher & Bennetts state it directly: *"As the weights were based on the observed
SEs, σ was fixed to 1 in the modeling. However, in R, using the NLME function, it
was not possible to fix σ to 1."*

## Used by

- `examples/mbma_naproxen.ferx` — the shipped model.
- `tests/mbma_naproxen_case_study.rs` — reproduces the published estimates
  (Boucher & Bennetts Table 2, NONMEM column, plus the Monolix replication) and
  checks the compartment-free form against the `d/dt(clock) = 1` workaround.
