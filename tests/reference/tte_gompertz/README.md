# TTE Gompertz — reference / hand-off kit

Validate ferx's Phase 1 Gompertz TTE estimator. **ferx column is done; nlmixr2 and NONMEM
are the hand-off** (NONMEM needs a licence). There is **no base-R/`survreg` anchor** for
Gompertz, so recovery of the data-generating parameters is the license-free guard.

Dataset `tte_gompertz.csv`: 300-subject 2-arm RCT, **fixed-effects** Gompertz with a treatment
effect, censored at t=365 (231 events, 77%). Data-generating `log_alpha=−6.0`,
`log_gamma=−5.4`, `log_hr=−0.8`. `h=alpha·exp(gamma·t)·exp(log_hr·TRT)`.

## Files

| File | What it is | Status |
|---|---|---|
| `tte_gompertz.csv` | dataset (`ID,TIME,DV,TRT,EVID,CMT,MDV`) | — |
| `simulate.R` | regenerates the dataset (base R, seed 42) | done |
| `nlmixr2.R` | fixed-effects Gompertz fit (needs `nlmixr2`; uses FOCEI — `bobyqa` fails in 5.x) | ✅ done (column filled) |
| `nonmem.ctl` | fixed-effects `LAPLACIAN` fit with TRT covariate (needs NONMEM) | **hand-off** |

> **nlmixr2 done** — log_alpha/gamma/hr agree with ferx within ~3%; OFV not cross-tool
> comparable for Gompertz (constants, see `expected.md`). Only NONMEM remains. macOS link
> gotcha: see the FLIBS note in `../tte_exponential/README.md`.
| `expected.md` | filled comparison + nlmixr2 blog cross-reference | paste results |

## Run the hand-off pieces

```bash
nmfe75 nonmem.ctl run.lst     # report THETA(1)=log_alpha, THETA(2)=log_gamma, THETA(3)=log_hr, OFV
Rscript nlmixr2.R
```

Paste into the **Fixed-effects RCT recovery** table in `expected.md`.

## ferx target

Fixed-effects FOCEI on `tte_gompertz.csv` recovers the truth essentially exactly:
`alpha 0.002471` (log −6.003), `gamma 0.004525` (log −5.398), `log_hr −0.803`, OFV 3011.12.
nlmixr2 / NONMEM should agree closely (OFV may differ by an additive normalising constant).

> A separate *frailty*-Gompertz SSE (BSV on gamma) shows the same FOCEI over-estimation of a
> nonlinear-parameter `omega^2` documented for the Weibull shape (#440) — not relevant to this
> fixed-effects dataset, but see `expected.md`.

```bash
cargo test --features survival,slow-tests --test tte_convergence -- --nocapture
```
