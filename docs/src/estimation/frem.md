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
    None,  // missing value indicator (default: -99)
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
- **IMPMAP** with `impmap_mceta` is a strong alternative to SAEM for FREM.
  The multi-start MAP (`impmap_mceta = 3`) helps the per-subject mode search
  escape local optima in the high-dimensional random-effect space. On larger
  FREM models (many covariates / ETAs) where the MAP surface has multiple
  modes, MCETA can dramatically improve importance-sampling efficiency and OFV.
- **FOCEI** works well for smaller FREM models (2-3 covariates) but can
  struggle with large block omegas.

### Warfarin FREM comparison (5 ETAs: 3 PK + WT + AGE, 10 subjects)

All methods use default tuning (SAEM: 500+800 iters; IMPMAP: 200 iters,
K=300). Covariate omega diagonals should approximate sample variances
(WT: 111.6, AGE: 99.4). OFV is the FOCE-Laplace objective.

| Parameter | FOCEI | SAEM | IMPMAP | IMPMAP+MCETA3 |
|-----------|------:|-----:|-------:|--------------:|
| OFV | -177.3 | -220.7 | -220.3 | -220.7 |
| TVCL | 0.133 | 0.133 | 0.133 | 0.133 |
| TVV | 7.73 | 7.74 | 7.74 | 7.74 |
| TVKA | 0.791 | 0.811 | 0.811 | 0.811 |
| ω²(CL) | 0.026 | 0.029 | 0.029 | 0.029 |
| ω²(V) | 0.012 | 0.010 | 0.010 | 0.010 |
| ω²(KA) | 0.351 | 0.336 | 0.336 | 0.336 |
| ω²(WT) | 135.1 | 106.8 | 106.8 | 106.8 |
| ω²(AGE) | 95.1 | 93.8 | 93.8 | 93.8 |
| σ (PROP) | 0.0106 | 0.0105 | 0.0111 | 0.0106 |

FOCEI is ~43 OFV units worse and overestimates the covariate variances,
while SAEM, IMPMAP, and IMPMAP+MCETA3 all agree closely. On this small model
(5 ETAs) the MCETA benefit is modest (~0.4 OFV); on larger FREM models with
many covariates the improvement can be dramatic.

## Limitations

- Categorical covariates are binarized (one-hot encoded) before the FREM
  transformation.
- TTE + FREM combination is not yet supported.
- The `prepare_frem()` API handles the full transformation; manual FREM model
  construction is not recommended.
