# Feature Maturity

Not every feature in ferx-core is equally battle-tested. Some — like FOCE/FOCEI
estimation and the analytical PK solutions — have been validated against gold
standard engines across many datasets. Others are newer and have only been
exercised on a handful of examples. To make this explicit, every major feature
carries a **maturity label**.

Where a feature has its own dedicated reference page, that page repeats its label
in a banner at the top, e.g.:

> **Maturity: beta** — see [Feature Maturity](maturity.md) for what this means.

The table below is the authoritative list; a few features documented inside a
shared reference page (e.g. the gradient method under [Fit Options](model-file/fit-options.md))
or only in an example carry their label here rather than in a per-page banner.

## Maturity levels

| Label | Meaning |
|-------|---------|
| **stable** | Well-tested functionality with proven stability across diverse datasets and estimation options. Performance is comparable to (or better than) gold standard NLME engines (NONMEM, Monolix). Safe for production use. |
| **beta** | Stable in limited testing and across a range of fit settings, but some caution is warranted before relying on it in production on unseen datasets. Validate against a reference where you can. |
| **experimental** | New functionality tested only on a single example or a small handful of (often toy) examples. Behaviour and syntax may change. Results should be treated as provisional and validated carefully. |

These labels describe ferx-core's *current* state and will move upward as
features accumulate testing and cross-validation. Most of ferx-core is currently
**beta**, with a mature core approaching **stable**; a few features remain
**experimental**.

## Runtime warnings

**Experimental** features emit a warning at fit time (surfaced in
`FitResult.warnings`, the CLI output, and `ferx check`) so their status is
visible at the point of use:

| Feature | Warning code |
|---------|--------------|
| Stochastic differential equations (`[diffusion]`) | `W_EXPERIMENTAL_SDE` |
| Neural networks (`[covariate_nn]`) | `W_EXPERIMENTAL_NN` |

**Beta** and **stable** features do not emit a maturity warning.

## Feature reference

### Model file features

| Feature | Maturity | Reference |
|---------|----------|-----------|
| Parameters (theta / omega / sigma, block omega) | **stable** | [Parameters](model-file/parameters.md) |
| Inter-occasion variability (IOV) | **stable** | [IOV](model-file/iov.md) |
| Covariates | **stable** | [Covariates](model-file/covariates.md) |
| Structural model — analytical PK (1/2/3-cpt) | **stable** | [Structural Model](model-file/structural-model.md) |
| Lag time | **stable** | [Lagtime](model-file/lagtime.md) |
| Steady-state doses (SS) | **stable** | [Steady-State Doses](model-file/steady-state.md) |
| Multiple dosing (ADDL / II) | **stable** | [Multiple Dosing](examples/multiple-dosing.md) |
| Error model (additive / proportional / combined) | **stable** | [Error Model](model-file/error-model.md) |
| Simulation | **stable** | [Simulation](model-file/simulation.md) |
| Individual parameters DSL | **beta** | [Individual Parameters](model-file/individual-parameters.md) |
| BLOQ / censored observations (M3) | **beta** | [BLOQ](model-file/bloq.md) |
| ODE models (Dormand-Prince RK45) | **beta** | [ODE Models](model-file/ode-models.md) |
| Scaling | **beta** | [Scaling](model-file/scaling.md) |
| Data selection | **beta** | [Data Selection](model-file/data-selection.md) |
| Derived columns | **beta** | [Derived Columns](model-file/derived.md) |
| Output columns | **beta** | [Output Columns](model-file/output.md) |
| Time-to-event endpoints (TTE) | **beta** | [Time-to-Event Endpoints](model-file/event-model.md) |
| Stochastic differential equations (SDE) | **experimental** | [SDE](model-file/diffusion.md) |
| Neural networks (DCM / NODE) | **experimental** | [Neural Networks](model-file/neural-networks.md) |

### Estimation features

| Feature | Maturity | Reference |
|---------|----------|-----------|
| FOCE / FOCEI | **stable** | [FOCE / FOCEI](estimation/foce.md) |
| Gauss-Newton (BHHH) — `gn`, `gn_hybrid` | **beta** | [Gauss-Newton](estimation/gauss-newton.md) |
| SAEM | **beta** | [SAEM](estimation/saem.md) |
| SIR | **beta** | [SIR](estimation/sir.md) |
| Importance sampling (IMP) | **beta** | [Importance Sampling](estimation/importance-sampling.md) |
| Outer optimizers (BOBYQA, SLSQP, L-BFGS, MMA, trust-region) | **beta** | [Outer Optimizers](estimation/optimizers.md) |
| Time-to-event estimation (TTE) | **beta** | [Time-to-Event](estimation/tte.md) |
| Automatic differentiation (`gradient_method = ad`) | **beta** | [Fit Options](model-file/fit-options.md) |

> The label reflects the engine's overall confidence in a feature, not the
> presence or absence of tests — a beta feature may still have extensive
> automated tests; "stable" additionally requires broad cross-validation against
> reference engines across diverse datasets.
