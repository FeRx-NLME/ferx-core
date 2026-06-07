# Derived Columns and Output

This example demonstrates the `[derived]` and `[output]` blocks, which add post-fit computed columns and individual PK parameter columns to the sdtab. The complete model file is [`examples/warfarin_derived.ferx`](https://github.com/FeRx-NLME/ferx-core/blob/main/examples/warfarin_derived.ferx).

## When to use

Use `[derived]` when you need columns in the sdtab that are not observations or predictions — for example, elimination half-life, NCA metrics (AUC, Cmax), or dosing-day labels. Use `[output]` when you want individual parameter estimates (CL, V, …) or covariates echoed directly into the sdtab alongside diagnostics.

## Model file

This is the contents of [`examples/warfarin_derived.ferx`](https://github.com/FeRx-NLME/ferx-core/blob/main/examples/warfarin_derived.ferx):

```
# One-compartment oral PK model (warfarin) -- [derived] and [output] demo

[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)

  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30

  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[derived]
  # Per-row: elimination rate and half-life
  KE     = CL / V
  T_HALF = 0.6931472 / KE

  # Which dosing day (tau = 24 h)
  DAY      = floor(TAFD / 24) + 1
  TAU_TIME = TAFD mod 24

  # Subject-level aggregates (one scalar per subject, repeated across rows)
  CMAX       = max(IPRED)
  TMAX       = tmax(IPRED)
  CTROUGH    = min(IPRED, TAD < 1e-10)
  CMAX_D1    = max(IPRED, TAFD < 24)
  CMAX_D14   = max(IPRED, TAFD >= 312 && TAFD < 336)

  # AUC over first 72 h (fine internal grid, 500 steps)
  AUC_0_72   = integral(IPRED, from=0, to=72)

  # Periodic AUC: one value per 24-h dosing window
  AUC_TAU    = integral(IPRED, window=24, anchor=0, step=0.1)

  # DV-based AUC (observation times only -- no interpolation)
  AUC_DV_72  = integral(DV, from=0, to=72)

[output]
  CL V KA

[fit_options]
  method     = foce
  maxiter    = 300
  covariance = true
```

Key points:

- **`[derived]`** expressions are evaluated for each observation row after the fit. They can reference individual-parameter names (`CL`, `V`), theta names (`TVCL`), eta names (`ETA_CL`), covariate names, `IPRED`, `PRED`, `DV`, `TAFD`, `TAD`, and `TIME`.
- **Per-row** expressions like `KE = CL / V` produce one value per observation.
- **Aggregate** expressions (`max`, `min`, `tmax`) produce one value per subject, broadcast to all of that subject's observation rows.
- **Integral** expressions compute AUC. Use `from=/to=` for a fixed window or `window=` for periodic (per-dose) AUC. A `step=` argument causes evaluation on a uniform grid (useful for time-above-threshold or dense AUC).
- **`[output]`** echoes individual parameters (`CL V KA`) and covariates as additional sdtab columns. Derived columns (from `[derived]`) are already written automatically — no need to list them again in `[output]`.

## Running the fit

```bash
ferx examples/warfarin_derived.ferx --data data/warfarin.csv
```

Or via the Rust API:

```rust
let result = fit_from_files("examples/warfarin_derived.ferx", "data/warfarin.csv")?;
// Derived columns are in result.sdtab_extra or the written CSV
```

## Interpreting output

The sdtab gains extra columns after the mandatory minimum (`ID, TIME, DV, PRED, IPRED, …`):

```
ID,TIME,DV,PRED,IPRED,CWRES,IWRES,EBE_OFV,N_OBS,TAFD,TAD,KE,T_HALF,DAY,TAU_TIME,CMAX,TMAX,CTROUGH,CMAX_D1,CMAX_D14,AUC_0_72,AUC_TAU,AUC_DV_72,CL,V,KA
```

Subject-level aggregates (`CMAX`, `TMAX`, `AUC_TAU`, etc.) repeat the same value on every observation row for a given subject, making them easy to join to per-subject summaries.

## Tips

- **Sequential scoping**: a derived expression may reference any derived name defined earlier in the same `[derived]` block. `T_HALF = 0.6931472 / KE` is valid because `KE` is defined one line above.
- **AUC step size**: `step=0.1` (6-minute grid) is sufficient for most PK curves. Reduce to `step=0.01` only if the half-life is very short relative to the dosing interval.
- **Trough via TAD**: `min(IPRED, TAD < 1e-10)` catches the dose-time row after ADDL expansion. The `1e-10` tolerance absorbs floating-point residuals from modular time arithmetic.
- **Covariate shadowing**: listing a covariate name in `[derived]` replaces the raw covariate with your computed value. A `W_DERIVED_COVARIATE_SHADOW` warning is emitted; this is intentional if you want a transformed covariate in the sdtab.
- Names that clash with mandatory sdtab columns (`ID`, `TIME`, `DV`, `PRED`, `IPRED`, `CWRES`, `IWRES`, `EBE_OFV`, `N_OBS`, `TAFD`, `TAD`, `CENS`, `OCC`, `CMT`) are rejected at parse time.
