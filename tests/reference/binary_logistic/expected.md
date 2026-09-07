# Binary / logistic endpoint — cross-tool anchor (#760)

Fixed-effects logistic regression on `data/binary_logistic.csv` (60 subjects × 3
times; data-generating `TH0=-0.4, THX=0.9, THT=0.5`) with the model
`logit P(DV=1) = TH0 + THX*X + THT*TIME` (`examples/binary_logistic.ferx`).

A fixed-effects logistic fit *is* ordinary logistic regression, so ferx must
reproduce base-R `glm` and NONMEM `F_FLAG=1` exactly.

| Parameter | ferx | R `glm` | NONMEM F_FLAG=1 |
|-----------|-----:|--------:|----------------:|
| TH0 (intercept) | -0.774849 | -0.775172 | -0.775172 |
| THX (× X)       |  0.870901 |  0.870140 |  0.870140 |
| THT (× TIME)    |  0.826837 |  0.827029 |  0.827029 |
| OFV / deviance / -2logL | 213.5955 | 213.5955 | 213.59548 |

- **R glm**: `glm(DV ~ X + TIME, family = binomial)` — `coef()` + `$deviance`.
- **NONMEM** (7.6.0, `nonmemdocker:V0.1`): `binlog.ctl` in this directory. To
  reproduce: copy `data/binary_logistic.csv` next to `binlog.ctl`, then
  `nmfe76 binlog.ctl binlog.lst`. Final estimates in `binlog.ext` (`ITERATION
  -1000000000`); OFV `213.59547683`.
- **ferx**: `cargo run -p ferx-cli --features ferx-core/survival -- examples/binary_logistic.ferx
  --data data/binary_logistic.csv`; guarded by `tests/categorical_convergence.rs`.

NONMEM and glm agree to 6 sig figs (both exact ML / IRLS). ferx's OFV matches to 5
decimals; its θ estimates are within the derivative-free (BOBYQA) outer tolerance.
