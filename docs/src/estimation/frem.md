# FREM (Full Random Effects Model)

FREM (Karlsson 2012) is a covariate analysis method that treats covariates as
additional dependent variables, estimating covariate-parameter relationships
through an extended omega matrix rather than explicit covariate models.

## Specifying covariates

The covariates folded into the FREM model — and whether each is **continuous**
or **categorical** — are taken from the model's
[`[covariates]`](../model-file/covariates.md) block. That block is the single
source of truth and is **required**: the transformation uses every covariate it
declares.

```text
[covariates]
  WT  continuous
  AGE continuous
  SEX categorical
```

If you don't want *every* declared covariate in the FREM model, pass a
**subset filter** (the `covariates` argument): only the named covariates are
included, and each must be declared in the block. The filter never introduces
covariates the model hasn't declared, nor changes their kind.

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
    &[],   // empty filter → use every covariate from the model's [covariates] block
    None,  // categorical override (None → kinds come from the [covariates] block)
    None,  // default output model path
    None,  // default output data path
)?;

// Pass e.g. &["WT".into()] instead of &[] to FREM only a subset of the
// declared covariates.

// result.model_path  — generated FREM .ferx file
// result.data_path   — augmented CSV with FREMTYPE column
// result.n_total_etas — base etas + covariate etas
```

## Usage (R)

```r
library(ferx)

# Covariates (and their continuous/categorical kind) come from the model's
# [covariates] block; omit `covariates` to use all of them.
frem <- ferx_to_frem(
  model = "warfarin.ferx",
  data  = "warfarin_cov.csv"
)

# Or filter to a subset of the declared covariates:
# frem <- ferx_to_frem("warfarin.ferx", "warfarin_cov.csv", covariates = "WT")

fit <- ferx_fit(frem, method = "saem")   # frem is a ferx_model
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
