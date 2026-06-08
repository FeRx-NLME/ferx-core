# Time-to-Event Endpoints (`[event_model]`)

The `[event_model]` block registers a CMT column as a TTE (time-to-event)
endpoint.  Observations on that CMT are routed to a parametric survival
likelihood rather than the Gaussian residual-error model.

See [Time-to-Event Estimation](../estimation/tte.md) for the full reference
including data format, hazard families, and comparison with nlmixr2 / NONMEM.

## Syntax

```
[event_model]
  cmt    = <integer>    # CMT column value in the data file
  family = exponential  # exponential | weibull | gompertz
  scale  = <expression> # theta/eta/covariate expression — Exponential (rate λ) and Weibull
  rate   = <expression> # alias for scale (Exponential only)
  shape  = <expression> # Weibull only (required; error if present for exponential/gompertz)
  alpha  = <expression> # Gompertz only: baseline hazard at t=0
  gamma  = <expression> # Gompertz only: hazard growth rate
  loghr  = <expression> # optional (all families): proportional-hazards covariate term;
                        #   multiplies the full hazard by exp(loghr)
```

> **Expression namespace:** all expressions are evaluated in the theta / eta / covariate
> namespace. Names from `[individual_parameters]` are **not** in scope — write the full
> theta/eta expression directly (e.g. `TVLAMBDA * exp(ETA_LAMBDA)`, not `LAMBDA`).

Named blocks allow multiple TTE endpoints:

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

## DV coding

| DV  | Meaning |
|-----|---------|
| `0` | Right-censored |
| `1` | Exact event at this TIME |
| `2` | Interval-censored right bound (pair with a preceding DV=0 row on same CMT) |

## TENTRY column

Add `TENTRY` to the data file to apply left-truncation (delayed entry):
the likelihood conditions on survival past `TENTRY`.
