# Time-to-Event (TTE) Models

> **Phase 1 (current):** Parametric TTE with exponential, Weibull, and Gompertz hazard
> families; right-censored, exact, and interval-censored observations; left truncation
> via `TENTRY`; FOCEI and SAEM estimation; fixed-effects (`n_eta = 0`) and random-effects
> paths. `[structural_model]`, `[error_model]`, and `[individual_parameters]` are all
> optional for TTE-only models.

## Overview

TTE models describe the probability distribution of the time until a clinical event
(e.g. first adverse event, study dropout, disease progression). ferx uses the same
FOCEI Laplace objective as for continuous PK/PD data, extended to the per-subject
log-likelihood contribution

```
ℓᵢ = δᵢ · log h(Tᵢ) − H(Tᵢ)
```

where δᵢ is the event indicator, h(t) the hazard function, and H(t) = ∫₀ᵗ h(s) ds
the cumulative hazard.

## Model file syntax

Add one `[event_model]` block per TTE endpoint.

```
[event_model]
  cmt    = 2                        # data-file CMT value that carries TTE rows
  family = exponential              # exponential | weibull | gompertz
  scale  = TVLAMBDA * exp(ETA_LAMBDA)  # theta/eta/covariate expression
```

> **Note:** `[event_model]` parameter expressions are evaluated in the
> theta / eta / covariate namespace. Names defined in `[individual_parameters]`
> (e.g. `LAMBDA`) are **not** available here — write the full expression in
> terms of `theta` and `eta` names directly. This restriction will be lifted
> in a future release.

For Weibull, a `shape` parameter is also required:

```
[event_model]
  cmt    = 2
  family = weibull
  scale  = TVSCALE * exp(ETA_SCALE)
  shape  = TVSHAPE
```

Gompertz uses `alpha` (baseline hazard) and `gamma` (growth rate):

```
[event_model]
  cmt    = 2
  family = gompertz
  alpha  = TVALPHA
  gamma  = TVGAMMA
```

Multiple TTE endpoints are supported by repeating the block with unique names:

```
[event_model DROPOUT]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA_DROPOUT * exp(ETA_LAMBDA)

[event_model DEATH]
  cmt    = 3
  family = weibull
  scale  = TVSCALE_DEATH
  shape  = TVSHAPE_DEATH
```

### `[structural_model]` and `[error_model]` requirement

The parser currently requires both blocks even for TTE-only models. Include a
dummy 1-cpt block with fixed parameters:

```
[parameters]
  theta TVLAMBDA(0.05, 0.001, 10.0)
  theta DUMMY_CL(1.0, FIX)
  theta DUMMY_V(1.0, FIX)
  omega ETA_LAMBDA ~ 0.09
  sigma SIGMA_DV ~ 0.01 FIX

[individual_parameters]
  LAMBDA = TVLAMBDA * exp(ETA_LAMBDA)
  CL     = DUMMY_CL
  V      = DUMMY_V

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)     # inert: no CMT-1 observations in TTE-only data

[error_model]
  DV ~ additive(SIGMA_DV)       # inert: sigma carries zero gradient

[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA * exp(ETA_LAMBDA)  # must use theta/eta names directly
```

This restriction will be lifted in a future release.

## Data format

TTE rows follow the NONMEM observation convention with `EVID=0` and a CMT
that matches the `cmt =` declaration in `[event_model]`:

| DV  | Meaning                         |
|-----|---------------------------------|
| `0` | Right-censored (survived past T) |
| `1` | Exact event at T                |
| `2` | Interval-censored: right bound (pair with preceding DV=0 row for same CMT) |

For interval-censored observations, precede the DV=2 row with a DV=0 row
carrying the left bound (entry into the at-risk window):

```
ID,TIME,DV,EVID,CMT
1, 5.0, 0, 0, 2   # left bound of interval (entry)
1,10.0, 2, 0, 2   # right bound  →  event occurred in (5, 10]
```

### Left truncation (delayed entry)

If subjects enter the study at a time > 0 (e.g. they survived until enrolment),
add a `TENTRY` column. ferx computes

```
H_eff(T) = H(T) − H(TENTRY)
```

conditioning the likelihood on survival past the entry time:

```
ID,TIME,DV,EVID,CMT,TENTRY
1,18.3, 1, 0, 2, 5.0   # subject entered at t=5, event at t=18.3
2,30.0, 0, 0, 2, 5.0   # entered at t=5, censored at t=30
```

## Hazard families

| Family | Parameters | h(t) | H(t) |
|--------|-----------|------|------|
| `exponential` | `scale` = λ | λ · exp(loghr) | λ · exp(loghr) · t |
| `weibull` | `scale` = α, `shape` = γ | (γ/α)(t/α)^(γ−1) · exp(loghr) | (t/α)^γ · exp(loghr) |
| `gompertz` | `alpha` = α, `gamma` = γ | α · exp(γ·t) · exp(loghr) | (α/γ)(exp(γ·t) − 1) · exp(loghr) |

All parameters must be positive. The optional `loghr` key adds a proportional-hazards
covariate term: the entire hazard (and cumulative hazard) is multiplied by `exp(loghr)`.
When `loghr` is omitted it defaults to 0 (no effect).

```
# Exponential with PH covariate on sex (SEX=1 reference):
[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA * exp(ETA_LAMBDA)
  loghr  = BETA_SEX * SEX

# Weibull with PH covariate on weight:
[event_model]
  cmt    = 2
  family = weibull
  scale  = TVSCALE * exp(ETA_SCALE)
  shape  = TVSHAPE
  loghr  = BETA_WT * WT
```

## Estimation

The FOCEI Laplace objective includes a TTE-specific FD Hessian contribution:

```
OFV = −2 · Σᵢ [ ℓᵢ(η̂ᵢ) − ½ log |Ω⁻¹ + ∇²ℓᵢ(η̂ᵢ)| ]
```

where `∇²ℓᵢ` is computed via a 4-point central-difference finite-difference
stencil (Shi 2021 step-size selection).

Set `method = focei` in `[fit_options]`. The `gradient = fd` setting is
the default and the only supported path for TTE objectives.

## Comparison with nlmixr2 and NONMEM

Reference exponential fit (30 subjects, λ=0.05 h⁻¹, ω²=0.09, 30% censored):

| Parameter | True | ferx | nlmixr2 | NONMEM (LAPLACIAN) |
|-----------|------|------|---------|------------------|
| TVLAMBDA  | 0.050 | — | — | — |
| ω²(ETA_LAMBDA) | 0.09 | — | — | — |
| OFV | — | — | — | — |

*Reference values will be filled after running the nlmixr2 and NONMEM scripts
in `tests/reference/tte_exponential/`.*

### nlmixr2 equivalent

```r
m <- function() {
  ini({
    TVLAMBDA <- 0.05
    eta.lambda ~ 0.09
  })
  model({
    lambda <- TVLAMBDA * exp(eta.lambda)
    ll(time.to.event) ~ event * log(lambda) - lambda * time.to.event
  })
}
fit <- nlmixr(m, data, est = "focei")
```

### NONMEM equivalent

```
$ESTIMATION METHOD=LAPLACIAN INTERACTION MAXEVAL=500
$THETA (0.001, 0.05, 10)
$OMEGA 0.09
```

## See also

- `examples/tte_exponential.ferx` — minimal worked example
- `data/tte_exponential.csv` — simulated dataset (30 subjects)
- `tests/tte_smoke.rs` — Tier-2 parse and short-run smoke tests
- `plans/tte-survival-markov.md` — full multi-phase roadmap
