# Inter-Occasion Variability (IOV)

This example extends the one-compartment oral warfarin model to account for occasion-to-occasion variability in clearance. The complete model file is `examples/warfarin_iov.ferx`.

## When to use IOV

Use IOV when:
- Subjects have data from multiple study periods (crossover designs, multiple visits)
- You expect within-subject PK variability across periods that is larger than pure residual error
- NONMEM models used `OMEGA BLOCK SAME` syntax for one or more parameters

## Dataset

The dataset must include an occasion-index column. Add an integer `OCC` column (or any name you choose — tell ferx-core via `iov_column`):

```csv
ID,TIME,DV,EVID,AMT,CMT,MDV,OCC
1,0,.,1,100,1,1,1
1,1,9.49,0,.,.,0,1
1,2,14.42,0,.,.,0,1
1,3,17.56,0,.,.,0,1
1,24,.,1,100,1,1,2
1,25,10.1,0,.,.,0,2
1,26,8.2,0,.,.,0,2
```

Occasion 1 = first period, occasion 2 = second period. Dose records and observation records in the same period share the same OCC value.

## Model File (Option A — diagonal IOV)

This is the contents of [`examples/warfarin_iov.ferx`](https://github.com/FeRx-NLME/ferx-core/blob/main/examples/warfarin_iov.ferx):

```
model warfarin_iov

[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)

  omega ETA_CL ~ 0.09      # between-subject variability in CL
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30

  kappa KAPPA_CL ~ 0.01    # inter-occasion variability in CL

  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = foce
  iov_column = OCC
  covariance = false
```

The key additions compared to a standard model:
1. `kappa KAPPA_CL ~ 0.01` — declares IOV on CL with starting variance 0.01
2. `+ KAPPA_CL` in the CL expression — kappa enters just like a BSV eta
3. `iov_column = OCC` — tells ferx-core which dataset column carries occasion labels

Set `covariance = true` if you want standard errors on the IOV variance (the shipped example leaves it off to keep the demo fast).

## Model File (Option B — correlated IOV)

If you have IOV on both CL and V and suspect they covary across occasions, use `block_kappa`:

```
[parameters]
  ...
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30

  block_kappa (KAPPA_CL, KAPPA_V) = [0.05, 0.01, 0.03]

  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V  + KAPPA_V)
  KA = TVKA * exp(ETA_KA)
```

The three values in `[0.05, 0.01, 0.03]` are the lower triangle of Ω_IOV:
- `0.05` = Var(KAPPA_CL)
- `0.01` = Cov(KAPPA_CL, KAPPA_V)
- `0.03` = Var(KAPPA_V)

## Running the Fit

The repository does not ship a warfarin dataset with occasion labels — point `--data` at your own NONMEM-format CSV that includes an `OCC` column (see [Dataset](#dataset) above):

```bash
ferx examples/warfarin_iov.ferx --data path/to/your_occ_data.csv
```

Or via the Rust API:

```rust
let result = fit_from_files("examples/warfarin_iov.ferx", "path/to/your_occ_data.csv")?;
println!("Omega IOV: {:?}", result.omega_iov);
```

## Interpreting Output

The console summary gains a `KAPPA (IOV) Estimates` block, and the fit YAML gains an `omega_iov:` section.

Console output (Option A, single kappa on CL):
```
--- KAPPA (IOV) Estimates ---
  KAPPA_CL = 0.041200  (CV% = 20.3)  SE = 0.008000
```

For Option B with off-diagonal correlations a `--- Correlations ---` block follows the diagonal entries.

Per-subject, per-occasion kappa EBEs are returned on `FitResult.ebe_kappas`. They are not yet emitted as columns in the sdtab CSV — access them via the Rust API if you need per-occasion plots:

```rust
for (i, subject_kappas) in result.ebe_kappas.iter().enumerate() {
    for (k, kappa_vec) in subject_kappas.iter().enumerate() {
        println!("subject {} occasion {}: {:?}", i + 1, k + 1, kappa_vec);
    }
}
```

> **Note**: per-kappa shrinkage is not yet computed (`shrinkage_kappa` is always returned empty). If you need a shrinkage diagnostic for IOV, derive it manually from `ebe_kappas` and `omega_iov`.

## Tips

- **Start with IOV on CL only** — it is the most commonly occasion-sensitive parameter. Add IOV on other parameters only when the OFV improvement justifies it.
- **Compare OFV** between BSV-only and IOV models — the difference is a likelihood-ratio test for the added IOV variance(s). Note that for a variance parameter at the boundary (H₀: σ² = 0) the asymptotic null is a 50:50 mixture of a point mass at 0 and χ²₁, not a plain χ²₁.
- **Use FOCEI for proportional errors** — the interaction term matters more when individual predictions vary across occasions.
- **SAEM is not supported** with IOV — use `foce` or `focei`.
