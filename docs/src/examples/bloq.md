# BLOQ (M3 method)

This example shows how to fit a model to data that contains observations below
the assay lower limit of quantification (BLOQ) using Beal's **M3 method**.
Instead of dropping BLOQ rows — which biases terminal-phase parameter estimates
— each censored observation contributes
`P(y < LLOQ | θ, η) = Φ((LLOQ − f)/√V)` to the likelihood.

## Dataset (`data/warfarin_bloq.csv`)

This is the warfarin dataset with an added `CENS` column. Ten late-time
observations that originally fell below an assay LLOQ of 2.0 µg/mL have been
marked `CENS=1` and their `DV` cell set to the LLOQ value:

```csv
ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,CENS
1,0,.,1,100,1,0,1,0
1,0.5,5.3653,0,.,1,0,0,0
...
1,96,2.5019,0,.,1,0,0,0
1,120,2,0,.,1,0,0,1
```

**NONMEM `CENS` convention**: when `CENS=1`, the row is censored and `DV`
carries the LLOQ value — not the true (unobserved) concentration. Rows with no
`CENS` column, or `CENS=0`, are treated as ordinary quantified observations.

## Model File (`examples/warfarin_bloq.ferx`)

```
# One-compartment oral PK model (warfarin) with M3 BLOQ likelihood.

[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)

  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30

  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method       = focei
  maxiter      = 300
  covariance   = true
  bloq_method  = m3
```

The only two lines that differ from a plain fit are `bloq_method = m3` in
`[fit_options]` and the added `CENS` column in the CSV. Nothing else in the
model changes.

## Running

```bash
ferx examples/warfarin_bloq.ferx --data data/warfarin_bloq.csv
```

## Expected Results

```
--- Objective Function ---
OFV:  -213.9104

--- THETA Estimates ---
TVCL                 0.126263
TVV                  7.629401
TVKA                 1.083302

--- OMEGA Estimates ---
  OMEGA(1,1) = 0.031381  (CV% = 17.7)
  OMEGA(2,2) = 0.009723  (CV% = 9.9)
  OMEGA(3,3) = 0.419327  (CV% = 64.8)

--- SIGMA Estimates ---
  SIGMA(1) = 0.010766
```

The diagnostic table (`warfarin_bloq-sdtab.csv`) gains a `CENS` column, and the
`IWRES` / `CWRES` cells for censored rows are written as empty (a weighted
Gaussian residual is undefined when the observed value is censored).

## Method Notes

- **Activation requires both pieces**: the `CENS` column in the data file *and*
  `bloq_method = m3` in `[fit_options]`. Without the option, `CENS=1` rows are
  treated as ordinary observations at the LLOQ value, which biases the fit.
- **FOCE is auto-promoted to FOCEI on affected subjects.** Mixing linearized
  Gaussian residuals with non-linearized `log Φ` terms produces inconsistent
  OFVs near the LLOQ boundary, so when `method = foce` and a subject has any
  `CENS=1` row, that subject is evaluated with η-interaction. A notice is
  written to `FitResult.warnings`; set `method = focei` explicitly to silence
  it.
- **Gauss-Newton caveat.** With `method = gn` or `gn_hybrid`, the BHHH
  information-matrix approximation degrades as the BLOQ fraction grows (each
  censored row carries less Fisher information than its Gaussian counterpart).
  A warning is emitted; for >20% censoring prefer `focei`.
- **SAEM** optimizes θ/σ with M3 in the M-step directly, so no special handling
  is required beyond setting `bloq_method = m3`.
