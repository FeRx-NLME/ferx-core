# Model File Reference

ferx-core models are defined in `.ferx` files using a declarative DSL. Each file is organized into blocks, each denoted by a `[block_name]` header.

## Block Overview

| Block | Required | Purpose |
|-------|----------|---------|
| `[parameters]` | Yes | Define theta, omega, and sigma parameters |
| `[individual_parameters]` | Yes | Map population parameters to individual PK parameters |
| `[structural_model]` | Yes | Specify the PK model (analytical or ODE) |
| `[error_model]` | Yes | Define the residual error model |
| `[odes]` | If ODE | ODE right-hand-side equations |
| `[fit_options]` | No | Configure estimation method and optimizer |
| `[simulation]` | No | Define a simulation trial design |

## Minimal Example

```
[parameters]
  theta TVCL(0.1, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma ADD_ERR ~ 1.0

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ additive(ADD_ERR)
```

Lines beginning with `#` are treated as comments.
