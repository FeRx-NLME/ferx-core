# FREM (Full Random Effects Model)

FREM (Karlsson 2012) is a covariate analysis method that treats covariates as
additional dependent variables, estimating covariate-parameter relationships
through an extended omega matrix rather than explicit covariate models.

## How it works

1. **Data augmentation**: For each subject, pseudo-observation rows are added
   with `DV` set to the subject's covariate value and a `FREMTYPE` column
   distinguishing covariate rows from PK observations.

2. **Model extension**: The base omega is expanded to a full block covering
   both PK random effects and covariate random effects. Covariate thetas are
   fixed at the population mean, and a small covariate sigma (`EPSCOV`) is
   added.

3. **Estimation**: The model is fit normally (FOCEI or SAEM). The off-diagonal
   blocks of the extended omega capture PK-covariate correlations.

## Usage (Rust API)

```rust
use ferx_core::prepare_frem;

let result = prepare_frem(
    &base_model_path,
    &data_path,
    &["WT".into(), "AGE".into()],
    None,  // no categoricals
    None,  // default output model path
    None,  // default output data path
)?;

// result.model_path  — generated FREM .ferx file
// result.data_path   — augmented CSV with FREMTYPE column
// result.n_total_etas — base etas + covariate etas
```

## Usage (R)

```r
library(ferx)

frem <- ferx_to_frem(
  model      = "warfarin.ferx",
  data       = "warfarin_cov.csv",
  covariates = c("WT", "AGE"),
  categorical = NULL
)

fit <- ferx_fit(frem$model_path, frem$data_path, method = "saem")
```

## Interpreting results

- **Covariate omega diagonals** should approximate the sample variance of each
  covariate (since the covariate thetas are fixed at the sample mean).
- **Off-diagonal elements** (or correlations) between PK etas and covariate
  etas reveal covariate-parameter associations.
- A positive correlation between ETA_CL and ETA_WT, for example, indicates
  that subjects with higher weight tend to have higher clearance.

## Estimation method choice

- **SAEM** is recommended for FREM models with many covariates, as the large
  block omega can cause convergence difficulties with gradient-based methods.
- **FOCEI** works well for smaller FREM models (2-3 covariates).

## Limitations

- Categorical covariates are binarized (one-hot encoded) before the FREM
  transformation.
- TTE + FREM combination is not yet supported.
- The `prepare_frem()` API handles the full transformation; manual FREM model
  construction is not recommended.
