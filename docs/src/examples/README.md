# Examples

FeRx includes several example models in the `examples/` directory with corresponding datasets in `data/`.

| Example | Model | Route | Features |
|---------|-------|-------|----------|
| [One-Compartment Oral](one-cpt-oral.md) | 1-cpt | Oral | Basic PopPK, FOCE and SAEM |
| [Two-Compartment IV](two-cpt-iv.md) | 2-cpt | IV bolus | Multi-compartment dynamics |
| [Covariates](covariates.md) | 2-cpt | Oral | Weight and renal function effects |
| [ODE Model](ode-model.md) | 1-cpt | Oral | Michaelis-Menten elimination, ODE solver |
| [BLOQ (M3 method)](bloq.md) | 1-cpt | Oral | Likelihood-based handling of censored observations |

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
```
