# IOV with SAEM

This example extends the inter-occasion variability (IOV) model to use SAEM estimation instead of FOCE. The complete model file is [`examples/warfarin_iov_saem.ferx`](https://github.com/FeRx-NLME/ferx-core/blob/main/examples/warfarin_iov_saem.ferx).

## When to use

SAEM is a stochastic approximation EM algorithm that samples ETAs (and kappas) directly rather than optimising them. It is more robust to multi-modal individual likelihoods and does not require a covariance step to estimate the population parameters. Consider SAEM with IOV when:

- The FOCE IOV model converges slowly or fails on ill-conditioned problems
- You want Bayesian-flavoured shrinkage on kappa estimates without tuning a gradient-based inner loop
- The data are sparse per occasion (SAEM handles low per-occasion information better than FOCE)

## Dataset

The dataset (`data/warfarin_iov_saem.csv`) must include an `OCC` column:

```csv
ID,TIME,DV,EVID,AMT,CMT,MDV,OCC
1,0,.,1,100,1,1,1
1,1,9.49,0,.,.,0,1
...
1,24,.,1,100,1,1,2
1,25,10.1,0,.,.,0,2
```

Each subject has two occasions. The `OCC` column drives the kappa sampling: a fresh kappa draw is made for each (subject, occasion) pair.

## Model file

This is the contents of [`examples/warfarin_iov_saem.ferx`](https://github.com/FeRx-NLME/ferx-core/blob/main/examples/warfarin_iov_saem.ferx):

```
[parameters]
  theta TVCL(0.134, 0.001, 10.0)
  theta TVV(8.1, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)

  omega ETA_CL ~ 0.07
  omega ETA_V  ~ 0.02
  omega ETA_KA ~ 0.40

  kappa KAPPA_CL ~ 0.04

  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method          = saem
  iov_column      = OCC
  n_exploration   = 150
  n_convergence   = 250
  n_mh_steps      = 3
  omega_burnin    = 20
  adapt_interval  = 10
  seed            = 12345
  covariance      = false
```

Compared to the FOCE IOV model ([Inter-Occasion Variability](iov.md)):
- `method = saem` replaces `method = foce`
- `n_exploration` / `n_convergence` replace `maxiter` (SAEM has two phases)
- `omega_burnin` and `adapt_interval` are SAEM-specific tuning knobs
- `covariance = false` — the SAEM covariance step (SIR-based) is optional and slow; omit it for exploratory runs

## Running the fit

```bash
ferx examples/warfarin_iov_saem.ferx --data data/warfarin_iov_saem.csv
```

Or via the Rust API:

```rust
let result = fit_from_files("examples/warfarin_iov_saem.ferx", "data/warfarin_iov_saem.csv")?;
println!("Omega IOV: {:?}", result.omega_iov);
```

## Interpreting output

SAEM produces the same output fields as FOCE for IOV: a `KAPPA (IOV) Estimates` block in the console summary, `omega_iov` in the fit YAML, and per-subject per-occasion kappa EBEs in `FitResult.ebe_kappas`.

```
--- KAPPA (IOV) Estimates ---
  KAPPA_CL = 0.038500  (CV% = 19.6)
```

## Tips

- **SAEM is slower than FOCE** for small datasets. Use FOCE by default; switch to SAEM when FOCE fails to converge or produces large shrinkage warnings.
- **Seed reproducibility**: SAEM draws random samples. Fix `seed` for reproducible output across runs.
- **`omega_burnin`**: holds the Omega matrix fixed for the first N SAEM iterations while the sampler warms up. Increase to 50 if the parameter traces show early instability.
- **`n_mh_steps`**: number of Metropolis-Hastings steps per SAEM iteration. 3 is a reasonable default; increase to 5–10 for models with many random effects or strong correlations.
- **Polishing**: for a final production run, add `method = [saem, focei]` to polish the SAEM estimates with FOCEI, which typically improves the OFV by 1–5 units and gives more accurate SE values.
