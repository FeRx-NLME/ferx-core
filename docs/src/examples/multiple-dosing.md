# Multiple Dosing (ADDL Column)

This example shows how the `ADDL` and `II` dataset columns are used to represent multiple-dose designs without repeating every dose row. The complete model file is [`examples/warfarin_addl.ferx`](https://github.com/FeRx-NLME/ferx-core/blob/main/examples/warfarin_addl.ferx).

## When to use

Use `ADDL` / `II` when:
- Subjects receive multiple doses on a regular schedule and you do not want to list every dose row explicitly
- You are converting a NONMEM dataset that already uses `ADDL` / `II`
- TAD should reset at every dose event (ferx expands ADDL internally, so TAD is always relative to the most recent expanded dose)

## Dataset

The dataset (`data/warfarin_addl.csv`) has one explicit dose row per subject followed by observation rows:

```csv
ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,II,ADDL
1,0,.,1,100,1,0,1,24,6
1,0.5,4.90,0,.,1,0,0,0,0
1,1.0,8.21,0,.,1,0,0,0,0
...
```

`ADDL=6` with `II=24` means the dose at TIME=0 is followed by 6 additional doses at TIME=24, 48, 72, 96, 120, and 144 h — 7 daily doses total. The engine expands these internally; no model changes are needed.

## Model file

This is the contents of [`examples/warfarin_addl.ferx`](https://github.com/FeRx-NLME/ferx-core/blob/main/examples/warfarin_addl.ferx):

```
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
  KE     = CL / V
  T_HALF = 0.6931472 / KE

  DAY    = floor(TAFD / 24) + 1

  CMAX        = max(IPRED)
  CMIN_TAU    = min(IPRED, TAD < 1e-10)
  CMIN_LATE   = min(IPRED, TAD > 20)
  AUC_TAU     = integral(IPRED, window=24, anchor=0, step=0.1)

[output]
  CL V KA

[fit_options]
  method  = foce
  maxiter = 300
```

After ADDL expansion, `TAD` automatically resets to 0 at each expanded dose time, so `min(IPRED, TAD < 1e-10)` correctly captures the trough at every dosing interval.

## Running the fit

```bash
ferx examples/warfarin_addl.ferx --data data/warfarin_addl.csv
```

Or via the Rust API:

```rust
let result = fit_from_files("examples/warfarin_addl.ferx", "data/warfarin_addl.csv")?;
```

## Interpreting output

The sdtab includes the standard diagnostics plus the derived columns:

```
ID,TIME,DV,PRED,IPRED,CWRES,IWRES,EBE_OFV,N_OBS,TAFD,TAD,KE,T_HALF,DAY,CMAX,CMIN_TAU,CMIN_LATE,AUC_TAU,CL,V,KA
```

`DAY` cycles 1, 2, … across dosing intervals; `CMAX` and `AUC_TAU` are per-subject aggregates broadcast to all observation rows.

## Tips

- **`ADDL` with `II`**: `ADDL` is the number of *additional* doses after the one on the record. Total doses = 1 + ADDL.
- **Dose timing**: expanded dose times are `TIME + k * II` for `k = 1, ..., ADDL`. All must be before the first observation that needs them.
- **Analytical superposition**: the analytical 1-cpt and 2-cpt solvers handle ADDL via superposition; no `[odes]` block is needed for standard compartment models.
- **ODE models with ADDL**: the ODE solver handles ADDL by replaying the expanded dose sequence. Ensure that `II` is large enough that the ODE state between doses does not need mid-interval resets.
