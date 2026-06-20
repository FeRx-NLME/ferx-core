# BLOQ (M3 method)

This example shows how to fit a model to data that contains observations below
the assay lower limit of quantification (BLOQ) using Beal's **M3 method**.
Instead of dropping BLOQ rows — which biases terminal-phase parameter estimates
— each below-LOQ observation contributes
`P(y < LLOQ | θ, η) = Φ((LLOQ − f)/√V)` to the likelihood. Above-LOQ rows use
the mirrored upper-tail contribution, `P(y > ULOQ | θ, η) = Φ((f − ULOQ)/√V)`.

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

**CENS convention**: when `CENS=1`, the row is left-censored and `DV`
carries the LLOQ value — not the true (unobserved) concentration. When
`CENS=-1`, the row is right-censored and `DV` carries the ULOQ value. This
matches the Monolix/nlmixr2 polarity. Rows with no `CENS` column, or `CENS=0`,
are treated as ordinary quantified observations.

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
OFV:  -217.1841

--- THETA Estimates ---
TVCL                 0.132811
TVV                  7.732354
TVKA                 0.811661

--- OMEGA Estimates ---
  OMEGA(1,1) = 0.029021  (CV% = 17.0)
  OMEGA(2,2) = 0.009557  (CV% = 9.8)
  OMEGA(3,3) = 0.338106  (CV% = 58.1)

--- SIGMA Estimates ---
  SIGMA(1) = 0.010761
```

These match a NONMEM 7.5.1 LAPLACE M3 reference (`tests/nonmem/warfarin_bloq.{ctl,lst}`)
to ~4 significant figures on the structural and variance parameters (TVCL 0.132801,
TVV 7.73139, TVKA 0.809824, PROP 0.010760, ω 0.028849 / 0.009544 / 0.335772). The
NONMEM objective uses the F_FLAG likelihood convention for censored rows, which
carries a different additive constant than ferx's M3 term, so the OFV is not directly
comparable (the MLE is).

The diagnostic table (`warfarin_bloq-sdtab.csv`) gains a `CENS` column, and the
`IWRES` / `CWRES` cells for censored rows are written as empty (a weighted
Gaussian residual is undefined when the observed value is censored).

## Method Notes

- **Activation requires both pieces**: the `CENS` column in the data file *and*
  `bloq_method = m3` in `[fit_options]`. Without the option, `CENS=1` (below
  LLOQ) and `CENS=-1` (above ULOQ) rows are treated as ordinary observations at
  the `DV` limit value, which biases the fit.
- **FOCE and FOCEI give different M3 optima.** `method = foce` keeps a consistent
  Sheiner–Beal objective: censored rows leave the linearized marginal and re-enter
  as `−log Φ((LLOQ − f̂)/√R⁰)` (and the `ULOQ` mirror for `CENS=-1`) with the
  population (η=0) variance, matching NONMEM `METHOD=1 LAPLACE` *without* INTER.
  `method = focei` evaluates the censored term at the conditional variance (NONMEM
  with INTER). Both have exact analytic gradients. On warfarin BLOQ the two land at
  meaningfully different `TVKA` (FOCE ≈ 0.71, FOCEI ≈ 0.81), exactly as the
  corresponding NONMEM runs do — pick the method to match your reference fit.
  (Earlier versions silently promoted censored subjects to FOCEI under
  `method = foce`.)
- **Gauss-Newton caveat.** With `method = gn` or `gn_hybrid`, the BHHH
  information-matrix approximation degrades as the BLOQ fraction grows (each
  censored row carries less Fisher information than its Gaussian counterpart).
  A warning is emitted; for >20% censoring prefer `focei`.
- **SAEM** optimizes θ/σ with M3 in the M-step directly, so no special handling
  is required beyond setting `bloq_method = m3`.
