# Tools

`ferx-core` fits one model to one dataset. Everything that calls `fit()` more than once — resampling, screening, searching a model space — lives in `ferx-tools` and is exposed as a `ferx <tool>` subcommand. Each tool carries a maturity tag; see [Feature Maturity](../maturity.llms.md) for what the tags mean.

## Uncertainty

- **[Bootstrap](../tools/bootstrap.llms.md)** — non-parametric case bootstrap: resample subjects with replacement, refit, and read bias, standard errors and confidence intervals off the spread of the estimates. PsN `bootstrap` feature parity.

## Covariate screening

- **[GAM covariate screening](../tools/gam-screening.llms.md)** — rank covariate–ETA pairs by ΔAIC on the EBEs before paying for a stepwise search. The ferx equivalent of Xpose4’s `xpose.gam()`.

## Model-space searches

All searches read the same [`.ferxsearch` configuration file](../tools/search.llms.md), whose search space is written in Pharmpy’s Model Feature Language (MFL).

- **[Search configuration](../tools/search.llms.md)** — the `.ferxsearch` TOML file, the MFL grammar ferx accepts, and the coverage table of what is and is not supported.
- **[Covariate search](../tools/covsearch.llms.md)** — stepwise covariate modelling: PsN `scm` forward / forward-then-backward and Pharmpy `covsearch`, plus allometric scaling.
- **[Structural model search](../tools/modelsearch.llms.md)** — Pharmpy `modelsearch` over absorption route, peripheral compartments, transit compartments and lag time.
- **[Residual-error search](../tools/ruvsearch.llms.md)** — Pharmpy `ruvsearch` over the `[error_model]` block, tested by likelihood ratio.
- **[Variability-structure search](../tools/iivsearch.llms.md)** — Pharmpy `iivsearch`: which parameters carry an η and how the omegas are blocked.
- **[Inter-occasion variability search](../tools/iovsearch.llms.md)** — Pharmpy `iovsearch`: which parameters carry an occasion-level κ.

## End to end

- **[Automatic model development](../tools/amd.llms.md)** — Pharmpy `amd`: the searches above run in order from one `.ferxsearch` file, each seeded from the model the last one selected, reported as one object.

## Global search

- **[Global model search](../tools/global-search.llms.md)** — pyDarwin’s genetic algorithm or exhaustive enumeration over one grid of structural and covariate choices, ranked on pyDarwin’s penalized fitness; for the model spaces whose axes interact, where a stepwise search can lock in a local optimum.
