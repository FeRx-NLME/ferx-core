# BLOQ / Below Limit of Quantification

> **Maturity: beta** — see [Feature Maturity](../maturity.md) for what this means.

Observations below the lower limit of quantification (LLOQ) cannot be treated as ordinary data: they carry only the information that the true concentration is somewhere below the detection threshold. Ignoring them (dropping rows) biases parameter estimates; treating the reported value (e.g. LLOQ/2) as a real observation is also biased. The statistically correct approach is to integrate the likelihood over the censored region.

ferx-core supports two strategies, selected via `bloq_method` in `[fit_options]`.

---

## Data format

Mark a BLOQ observation by setting `CENS = 1` in the dataset. The `DV` column for a BLOQ row should contain the LLOQ value (used by the M3 method as the integration upper bound):

```
ID,TIME,DV,CENS,EVID,AMT
1,0.0,0,0,1,100
1,1.0,2.45,0,0,0
1,6.0,0.02,1,0,0   ← BLOQ: DV holds the LLOQ
1,12.0,0.52,0,0,0
```

Rows with `CENS = 1` and `MDV = 1` are skipped entirely (already excluded by `MDV`). Rows with `CENS = 0` are treated as ordinary quantified observations regardless of their value.

---

## Methods

### `bloq_method = drop` (default)

BLOQ rows (`CENS = 1`) are excluded from the likelihood. Fast and simple, but biases parameter estimates — particularly terminal half-life and residual error — when the BLOQ fraction is substantial (>10–15% of observations).

```
[fit_options]
  bloq_method = drop
```

### `bloq_method = m3`

Implements Beal's M3 method (2001): the likelihood contribution of a BLOQ observation is the probability that the true value falls below the LLOQ:

\\[
L_i^{\text{BLOQ}} = \Phi\!\left(\frac{\text{LLOQ} - f_{ij}}{\sqrt{V_{ij}}}\right)
\\]

where \\( f_{ij} \\) is the model prediction, \\( V_{ij} \\) is the residual variance, and \\( \Phi \\) is the standard normal CDF. This is the maximum-likelihood treatment of censored normal data.

```
[fit_options]
  bloq_method = m3
```

`bloq` is accepted as an alias for `bloq_method`.

---

## When to use M3

Use M3 whenever more than ~5–10% of observations are BLOQ. Common situations:

- Sparse PK sampling with a long terminal phase
- Studies where the LLOQ is relatively high compared to trough concentrations
- Pediatric or renal-impaired populations with reduced drug exposure

The cost is a modest increase in run time (the CDF evaluation adds a small overhead per BLOQ row). The OFV from M3 is not directly comparable to the drop-method OFV; always compare M3 models against other M3 models.

---

## Example

```
[parameters]
  theta TVCL(0.13, 0.01, 10.0)
  theta TVV(8.0, 1.0, 100.0)
  theta TVKA(1.0, 0.1, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method      = focei
  bloq_method = m3
```

A complete runnable example is in [`examples/warfarin_bloq.ferx`](https://github.com/FeRx-NLME/ferx-core/blob/main/examples/warfarin_bloq.ferx) with the corresponding dataset in `data/warfarin_bloq.csv`. See also the [BLOQ example walkthrough](../examples/bloq.md).

---

## Reference

Beal, S.L. (2001). *Ways to fit a PK model with some data below the quantification limit.* Journal of Pharmacokinetics and Pharmacodynamics, 28(5), 481–504.
