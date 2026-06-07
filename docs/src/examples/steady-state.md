# Steady-State Dosing (SS Column)

This example demonstrates the `SS` and `II` dataset columns, which tell the engine to initialise the compartment states analytically at steady state rather than from zero. The complete model file is [`examples/warfarin_ss.ferx`](https://github.com/FeRx-NLME/ferx-core/blob/main/examples/warfarin_ss.ferx).

## When to use

Use `SS = 1` when:
- The study design collects PK observations after the drug has reached steady state (e.g. after multiple days of regular dosing)
- You want to avoid simulating every prior dose — the engine derives the initial conditions analytically from dose amount, dosing interval (`II`), and current individual parameters
- You are converting a NONMEM dataset that already uses the `SS` column

## Dataset

The dataset (`data/warfarin_ss.csv`) has one dose row per subject with `SS=1`:

```csv
ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,SS,II
1,0,.,1,100.0,1,0,1,1,24
1,0.5,23.34,0,.,1,0,0,0,0
1,1.0,28.11,0,.,1,0,0,0,0
...
```

`SS=1` combined with `II=24` instructs the engine: "at TIME=0, the subject is already at steady state for once-daily 100-mg dosing." DV values are in the 15–35 mg/L range — roughly 3× the peak seen after a single dose of the same amount — reflecting accumulation at a 42-hour half-life.

## Model file

This is the contents of [`examples/warfarin_ss.ferx`](https://github.com/FeRx-NLME/ferx-core/blob/main/examples/warfarin_ss.ferx):

```
[parameters]
  theta TVCL(0.134, 0.001, 10.0)
  theta TVV(8.1, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)

  omega ETA_CL ~ 0.07
  omega ETA_V  ~ 0.02
  omega ETA_KA ~ 0.40

  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = foce
  maxiter    = 300
  covariance = true
```

The model itself is identical to the standard warfarin example — no special keywords are needed in the model file. The `SS` column in the dataset drives the initialisation automatically.

## Running the fit

```bash
ferx examples/warfarin_ss.ferx --data data/warfarin_ss.csv
```

Or via the Rust API:

```rust
let result = fit_from_files("examples/warfarin_ss.ferx", "data/warfarin_ss.csv")?;
println!("TVCL = {:.3}", result.theta["TVCL"].estimate);
```

## Interpreting output

Estimates should be close to the single-dose warfarin values (`TVCL ≈ 0.134`, `TVV ≈ 8.1`, `TVKA ≈ 1.0`), demonstrating that the SS initialisation correctly accounts for prior accumulation.

## Tips

- **`SS` requires `II`**: every row with `SS=1` must also have a non-zero `II` (dosing interval in the same time units as `TIME`).
- **Analytical solvers only**: the `SS=1` initialisation is currently implemented for the analytical 1-cpt and 2-cpt solvers. For ODE models at steady state, use a long ADDL pre-dosing sequence to approximate it.
- **TAD at steady state**: `TAD` is computed relative to the SS dose row, so the first observation at `TIME=0.5` has `TAD=0.5`. TAFD is measured from the same row.
- **Mixed designs**: a dataset may contain some subjects with `SS=1` and others without (e.g. a cross-over study). Each subject's dose rows are handled independently.
