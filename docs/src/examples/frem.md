# FREM (Covariate Analysis)

This example demonstrates FREM covariate analysis using the warfarin
one-compartment oral model with body weight (WT) and age (AGE) as covariates.

## Workflow

FREM is a two-step process: **transform** the base model and dataset, then
**fit** the extended model.

### R

```r
library(ferx)

# Step 1: FREM transformation
frem <- ferx_to_frem(
  model      = "warfarin.ferx",
  data       = "warfarin_cov.csv",
  covariates = c("WT", "AGE")
)

# Step 2: Fit the FREM model
fit <- ferx_fit(frem$model_path, frem$data_path, method = "saem",
                settings = list(n_exploration = 500L, n_convergence = 800L))

# Inspect results
fit$omega  # 5x5 block: rows/cols 1-3 are PK, 4-5 are covariates
```

### Rust API

```rust
use ferx_core::{prepare_frem, fit_from_files};

let frem = prepare_frem(
    &Path::new("warfarin.ferx"),
    &Path::new("warfarin_cov.csv"),
    &["WT".into(), "AGE".into()],
    None, None, None,
)?;

let result = fit_from_files(
    frem.model_path.to_str().unwrap(),
    frem.data_path.to_str().unwrap(),
    None, None,
)?;
```

## What `prepare_frem` does

1. Reads the base model and dataset
2. Computes sample means and variances for each covariate
3. Adds pseudo-observation rows per subject (one per covariate, `DV` = covariate value)
4. Generates a new `.ferx` model with:
   - Fixed covariate thetas (set to sample means)
   - Extended block omega (PK + covariate etas, initial covariate diag = sample variance)
   - `EPSCOV` sigma for covariate observations
   - `frem_predictions` and `frem_sigma` fit options

## Interpreting the omega matrix

After fitting, the 5×5 omega for this example has the structure:

```
         ETA_CL  ETA_V  ETA_KA  ETA_WT  ETA_AGE
ETA_CL   [  PK IIV  ]  [  PK-cov correlations  ]
ETA_V    [           ]  [                       ]
ETA_KA   [           ]  [                       ]
ETA_WT   [  PK-cov   ]  [  covariate variances  ]
ETA_AGE  [           ]  [                       ]
```

- **Diagonal (4,4)** and **(5,5)**: should approximate the sample variance
  of WT and AGE respectively.
- **Off-diagonal** elements reveal covariate-parameter associations without
  requiring explicit covariate models.

## Example model file

See [`examples/warfarin_frem.ferx`](https://github.com/FeRx-NLME/ferx-core/blob/main/examples/warfarin_frem.ferx)
for the generated FREM model structure.
