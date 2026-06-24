# TTE Weibull — reference / hand-off kit

Validate ferx's Phase 1 Weibull TTE estimator. **ferx + base-R `survreg` columns are done;
the nlmixr2 and NONMEM columns are the hand-off** (NONMEM needs a licence).

Dataset `tte_weibull.csv`: 100 subjects, data-generating `scale=20`, `shape=2`, BSV on the
**shape** (`omega^2=0.20`), censored at t=30 (84 events, 16% censored). `H(t)=(t/scale)^shape`.

## Files

| File | What it is | Status |
|---|---|---|
| `tte_weibull.csv` | dataset everything fits (`ID,TIME,DV,EVID,CMT,MDV`) | — |
| `simulate.R` | regenerates the dataset (base R, seed 42) | done |
| `survreg.R` | fixed-effects Weibull MLE, mapped to ferx (scale, shape) | ✅ done |
| `nlmixr2.R` | mixed-effects FOCEI fit (needs `nlmixr2`) | ✅ done (column filled) |
| `nonmem.ctl` | mixed-effects `LAPLACIAN INTERACTION` fit (needs NONMEM) | **hand-off** |

> **nlmixr2 done** — agrees with ferx on scale/shape/−2LL; the shape-frailty ω² differs between
> the two FOCEI tools (0.204 vs 0.173, see #440 in `expected.md`). Only NONMEM remains. macOS
> link gotcha: see the FLIBS note in `../tte_exponential/README.md`.
| `expected.md` | filled comparison + the nonlinear-frailty omega^2 finding | paste results |

## Run the hand-off pieces

```bash
nmfe75 nonmem.ctl run.lst     # report exp(THETA(1))=scale, exp(THETA(2))=shape, OMEGA(1,1), OFV
Rscript nlmixr2.R             # mixed-effects FOCEI
```

Paste into the **Mixed-effects** table in `expected.md`.

## ferx targets

Fixed-effects (n_eta=0) — **exact** anchor vs `survreg`: scale 22.1766, shape 2.1192,
OFV 640.261. Mixed-effects FOCEI on `tte_weibull.csv`: scale 21.87, shape 2.21,
`omega^2` 0.204, OFV 639.99 — nlmixr2/NONMEM should land close.

> **Caveat for the omega^2 column (#440):** FOCEI-Laplace over-estimates the shape-frailty
> `omega^2` at large N (SSE: 0.34 vs truth 0.20; SAEM ≈ 0.13). On this particular n=100 file
> the FOCEI value (0.204) lands on the truth by sampling luck. Expect nlmixr2 FOCEI to agree
> with ferx FOCEI; a SAEM/IMP run will read lower. Compare like-for-like estimators.

```bash
cargo test --features survival,slow-tests --test tte_convergence -- --nocapture
```
