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
  scale  = <expression> # theta/eta/covariate expression for scale parameter
  shape  = <expression> # Weibull only (required for weibull; error if present for exponential)
  alpha  = <expression> # Gompertz only: baseline hazard
  gamma  = <expression> # Gompertz only: growth rate
```

Named blocks allow multiple TTE endpoints:

```
[event_model DROPOUT]
  cmt    = 2
  family = exponential
  scale  = LAMBDA

[event_model DEATH]
  cmt    = 3
  family = weibull
  scale  = SCALE_DEATH
  shape  = SHAPE_DEATH
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
