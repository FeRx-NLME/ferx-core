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

> **nlmixr2 done** — agrees with ferx on scale/shape/ω²/−2LL. The shape-frailty ω² now matches
> too (ferx 0.176 ≈ nlmixr2 0.173): ferx's earlier 0.204 was a BOBYQA outer-`ftol` convergence
> artifact, fixed in #469 (see `expected.md`). macOS link gotcha: see the FLIBS note in
> `../tte_exponential/README.md`.
| `expected.md` | filled comparison + the nonlinear-frailty omega^2 finding | paste results |

## Run the hand-off pieces

```bash
nmfe75 nonmem.ctl run.lst     # report exp(THETA(1))=scale, exp(THETA(2))=shape, OMEGA(1,1), OFV
Rscript nlmixr2.R             # mixed-effects FOCEI
```

Paste into the **Mixed-effects** table in `expected.md`.

## ferx targets

Fixed-effects (n_eta=0) — **exact** anchor vs `survreg`: scale 22.1766, shape 2.1192,
OFV 640.261. Mixed-effects FOCEI on `tte_weibull.csv`: scale 21.89, shape 2.21,
`omega^2` 0.176, OFV 639.98 — on the nlmixr2 0.173 / NONMEM 0.175 consensus (#469).

> **Caveat for the omega^2 column (#440):** FOCEI-Laplace over-estimates the shape-frailty
> `omega^2` at large N (SSE: 0.34 vs truth 0.20; SAEM ≈ 0.13). On this particular n=100 file
> the FOCEI value (0.176) sits near the truth. (ferx read 0.204 before #469 — a BOBYQA
> outer-`ftol` convergence artifact, now fixed; the +72% *method* bias is separate and still
> shows in the large-N SSE.) Expect nlmixr2 FOCEI to agree with ferx FOCEI; a SAEM/IMP run
> reads lower. Compare like-for-like estimators.

```bash
cargo test --features survival,slow-tests --test tte_convergence -- --nocapture
```
