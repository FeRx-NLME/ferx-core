# LOQ-Censored Observations

> **Maturity: beta** — see [Feature Maturity](../maturity.md) for what this means.

Observations below the lower limit of quantification (LLOQ) or above the upper limit of quantification (ULOQ) cannot be treated as ordinary data: they carry only the information that the true concentration is outside the assay's quantified range. Ignoring them (dropping rows) biases parameter estimates; treating the reported value (e.g. LLOQ/2 or ULOQ) as a real observation is also biased. The statistically correct approach is to integrate the likelihood over the censored region.

ferx-core supports two strategies, selected via `bloq_method` in `[fit_options]`.

---

## Data format

Mark a below-LOQ observation by setting `CENS = 1` in the dataset. The `DV` column for that row should contain the LLOQ value (used by the M3 method as the integration upper bound). Mark an above-LOQ observation by setting `CENS = -1`; its `DV` column should contain the ULOQ value:

```
ID,TIME,DV,CENS,EVID,AMT
1,0.0,0,0,1,100
1,1.0,2.45,0,0,0
1,6.0,0.02,1,0,0   # below LOQ: DV holds the LLOQ
1,8.0,20.0,-1,0,0  # above LOQ: DV holds the ULOQ
1,12.0,0.52,0,0,0
```

Rows with `CENS != 0` and `MDV = 1` are skipped entirely (already excluded by `MDV`). Rows with `CENS = 0` are treated as ordinary quantified observations regardless of their value.

---

## Methods

### `bloq_method = drop` (default)

Censored rows (`CENS = 1` or `CENS = -1`) are excluded from special handling and treated as ordinary observations at the value in `DV`. Fast and simple, but biased when the censored fraction is substantial (>10–15% of observations).

```
[fit_options]
  bloq_method = drop
```

### `bloq_method = m3`

Implements Beal's M3 method (2001): the likelihood contribution of a below-LOQ observation is the probability that the true value falls below the LLOQ:

\\[
L_i^{\text{BLOQ}} = \Phi\!\left(\frac{\text{LLOQ} - f_{ij}}{\sqrt{V_{ij}}}\right)
\\]

where \\( f_{ij} \\) is the model prediction, \\( V_{ij} \\) is the residual variance, and \\( \Phi \\) is the standard normal CDF. This is the maximum-likelihood treatment of censored normal data.

For above-LOQ observations (`CENS = -1`), ferx uses the mirrored upper-tail probability:

\\[
L_i^{\text{ULOQ}} = \Phi\!\left(\frac{f_{ij} - \text{ULOQ}}{\sqrt{V_{ij}}}\right)
\\]

This `CENS` polarity matches Monolix and nlmixr2: `1` means left-censored below LLOQ, and `-1` means right-censored above ULOQ.

### NONMEM comparison for above-LOQ censoring

The upper-tail contribution is cross-checked against NONMEM 7.5.1 in
`tests/nonmem/right_censored_m3.ctl`. The control stream uses `F_FLAG=1` with
two identical `CENS=-1` rows, `F=12`, `DV=ULOQ=10`, and `W=2`, so each row has
\\( z = (F - \text{ULOQ}) / W = 1 \\). NONMEM reports OFV
`0.69101514210943182`, matching ferx's `-4 * log(Φ(1))` to within `1e-6`
(the small difference is the CDF approximation used by ferx).

```
[fit_options]
  bloq_method = m3
```

`bloq` is accepted as an alias for `bloq_method`.

---

## When to use M3

Use M3 whenever more than ~5–10% of observations are censored. Common situations:

- Sparse PK sampling with a long terminal phase
- Studies where the LLOQ is relatively high compared to trough concentrations, or the ULOQ is exceeded after high doses
- Pediatric or renal-impaired populations with reduced drug exposure

The cost is a modest increase in run time (the CDF evaluation adds a small overhead per censored row). The OFV from M3 is not directly comparable to the drop-method OFV; always compare M3 models against other M3 models.

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
