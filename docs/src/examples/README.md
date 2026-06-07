# Examples

ferx-core includes several example models in the `examples/` directory with corresponding datasets in `data/`.

| Example | Model | Route | Features |
|---------|-------|-------|----------|
| [One-Compartment Oral](one-cpt-oral.md) | 1-cpt | Oral | Basic PopPK, FOCE and SAEM |
| [Two-Compartment IV](two-cpt-iv.md) | 2-cpt | IV bolus | Multi-compartment dynamics |
| [Covariates](covariates.md) | 2-cpt | Oral | Weight and renal function effects |
| [ODE Model](ode-model.md) | 1-cpt | Oral | Michaelis-Menten elimination, ODE solver |
| [BLOQ (M3 method)](bloq.md) | 1-cpt | Oral | Likelihood-based handling of censored observations |
| [Inter-Occasion Variability](iov.md) | 1-cpt | Oral | IOV with `kappa` parameters, FOCE |
| [IOV with SAEM](iov-saem.md) | 1-cpt | Oral | IOV estimated by SAEM |
| [Multiple Dosing (ADDL)](multiple-dosing.md) | 1-cpt | Oral | ADDL / II columns, TAD reset, derived NCA |
| [Steady-State Dosing (SS)](steady-state.md) | 1-cpt | Oral | SS=1 / II columns, analytical accumulation |
| [Unit Scaling](scaling.md) | 1-cpt | Oral | `obs_scale` for AMT/DV unit conversion |
| [Derived Columns and Output](derived-output.md) | 1-cpt | Oral | `[derived]` and `[output]` blocks, NCA metrics |
| [Data Selection](data-selection.md) | 1-cpt | Oral | `[data_selection]` IGNORE/ACCEPT filter |
| [ODE with Lag Time](ode-lagtime.md) | 1-cpt ODE | Oral | `LAGTIME` keyword, IIV on lag, `obs_scale = V` |
| [Transit Absorption](transit-absorption.md) | 2-cpt ODE | Oral | Transit chain, allometric scaling, FOCEI |
| [Stochastic Differential Equations](sde.md) | 1-cpt ODE | Oral | `[diffusion]` block, EKF process noise |
| [Multi-start](multistart.md) | 1-cpt ODE | Oral | Multiple starting values to avoid local minima |

## Running Examples

```bash
# One-compartment oral (warfarin)
ferx examples/warfarin.ferx --data data/warfarin.csv

# Two-compartment IV
ferx examples/two_cpt_iv.ferx --data data/two_cpt_iv.csv

# Covariates model
ferx examples/two_cpt_oral_cov.ferx --data data/two_cpt_oral_cov.csv

# ODE model (Michaelis-Menten)
ferx examples/mm_oral.ferx --data data/mm_oral.csv

# SAEM estimation
ferx examples/warfarin_saem.ferx --data data/warfarin.csv

# BLOQ (M3 method)
ferx examples/warfarin_bloq.ferx --data data/warfarin_bloq.csv

# IOV (inter-occasion variability)
ferx examples/warfarin_iov.ferx --data data/warfarin_iov.csv

# IOV with SAEM
ferx examples/warfarin_iov_saem.ferx --data data/warfarin_iov_saem.csv

# Multiple dosing (ADDL/II columns)
ferx examples/warfarin_addl.ferx --data data/warfarin_addl.csv

# Steady-state dosing (SS/II columns)
ferx examples/warfarin_ss.ferx --data data/warfarin_ss.csv

# Unit scaling (AMT in µg, DV in mg/L)
ferx examples/warfarin_scaled.ferx --data data/warfarin_scaled.csv

# Derived columns and output (NCA metrics, [output] block)
ferx examples/warfarin_derived.ferx --data data/warfarin.csv

# Data selection (IGNORE filter)
ferx examples/warfarin_data_selection.ferx --data data/warfarin.csv

# ODE model with absorption lag time
ferx examples/warfarin_ode_lagtime.ferx --data data/warfarin_ode_lagtime.csv

# Transit-compartment absorption (2-cpt ODE, allometric scaling)
ferx examples/transit_2cpt.ferx --data data/transit_2cpt.csv

# Stochastic differential equations (EKF process noise)
ferx examples/warfarin_sde.ferx --data data/warfarin_sde.csv

# Multi-start (Michaelis-Menten)
ferx examples/mm_multistart.ferx --data data/mm_oral.csv
```
