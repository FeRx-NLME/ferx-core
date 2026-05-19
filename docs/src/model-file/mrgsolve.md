# Importing mrgsolve Models

ferx can read [mrgsolve](https://mrgsolve.org)-style `.cpp`/`.mod` model
files directly. Two workflows are supported:

```bash
# Direct fit — ferx auto-detects the .cpp/.mod extension.
ferx warfarin.mod --data warfarin.csv

# Inspect-and-edit workflow — emit an equivalent .ferx file first.
ferx translate warfarin.mod -o warfarin.ferx
ferx warfarin.ferx --data warfarin.csv
```

Under the hood, the mrgsolve front-end translates the model into ferx's
`.ferx` DSL and feeds that through the existing parser. Translation is
deterministic and side-effect-free — the same `.cpp` always produces the
same `.ferx`.

## Supported blocks

| Block | Behaviour |
|---|---|
| `$PROB` | Model name (used in log output). |
| `$PARAM` | Named thetas: `$PARAM TVCL=1, TVV=10`. |
| `$THETA` | Positional thetas; auto-named `THETA1..THETAn`. |
| `$INPUT` | Covariate names (defaults ignored — covariates come from data). |
| `$OMEGA` | Diagonal or `@block`, with optional `@labels` for ETA names. |
| `$SIGMA` | Same shape as `$OMEGA`, for residual error terms. |
| `$CMT` / `$INIT` | Compartment names; `$INIT` allows `NAME = init_value`. |
| `$MAIN` / `$PK` | Individual-parameter calculations (becomes `[individual_parameters]`). |
| `$ODE` | Differential equations; `dxdt_X = ...` becomes `d/dt(X) = ...`. |
| `$TABLE` / `$ERROR` | Residual-error definition (pattern-matched into one of ferx's three error models). |
| `$CAPTURE` | Recorded but not separately translated (every named variable is already exposed). |
| `$PKMODEL` | Analytical model dispatch — `ncmt=1..3`, `depot=TRUE`/`FALSE`. |
| `$SET` | Ignored (simulation options aren't carried into the fit). |

## Unsupported (hard-fail) blocks and constructs

The translator rejects these with a `file:line:` error pointing at the
offending source line:

- `$GLOBAL`, `$PREAMBLE`, `$INCLUDE` — arbitrary C++ scope can't be safely
  embedded in `.ferx`.
- `$PLUGIN` — pluggable extensions (`Rcpp`, `mrgx`, `nm-vars`, `autodec`) are
  not modelled.
- `$NMXML`, `$NMEXT` — NONMEM result imports.
- `$PRED`, `$EVENT` — alternative model entry points.
- `for` / `while` loops, `return`, `?:` ternaries, C-style casts (`(int)x`).
- `pow(a, b)` — rewrite as `a^b`.
- Preprocessor directives (`#include`, `#define`).

## Name resolution rules

Inside `$MAIN`/`$ODE`/`$TABLE` bodies:

- `THETA(n)` resolves to the *n*th entry of `$PARAM` (1-indexed). If
  `$THETA` is also declared, `$THETA` takes precedence.
- `ETA(n)` resolves to the *n*th label across all `$OMEGA` blocks. When a
  block has no `@labels`, etas are auto-named `ETA1..ETAn`.
- `EPS(n)` resolves to the *n*th label across all `$SIGMA` blocks. Same
  auto-naming rule applies.
- Out-of-range indices produce a hard error.

Named references (`TVCL`, `ETA_CL`, …) work identically to positional ones.
The emitted `.ferx` source uses names exclusively — no `THETA(1)` survives
translation.

## Supported error-model patterns

ferx represents residual error declaratively, so the `$TABLE` block must end
in one of these structural shapes:

| Pattern | Becomes |
|---|---|
| `Y = IPRED * (1 + EPS(p))` | `DV ~ proportional(SIGMA_p)` |
| `Y = IPRED + EPS(a)` | `DV ~ additive(SIGMA_a)` |
| `Y = IPRED * (1 + EPS(p)) + EPS(a)` | `DV ~ combined(SIGMA_p, SIGMA_a)` |

Any other shape (e.g. log-normal `Y = IPRED * exp(EPS(1))`, two
proportional terms) is rejected with a hint listing the three supported
forms.

## Default fit options

A translated model has no `[fit_options]` section in the source `.cpp`. The
emitter inserts these defaults:

```
[fit_options]
  method     = focei
  maxiter    = 300
  covariance = true
```

To customise them, run `ferx translate` to emit the `.ferx`, edit the
`[fit_options]` block, and run `ferx <model>.ferx --data ...` against the
edited file.

## Worked example

Source `examples/warfarin.mod`:

```cpp
$PROB warfarin one-cpt oral
$PARAM TVCL = 0.2, TVV = 10.0, TVKA = 1.5
$CMT GUT CENT
$OMEGA @labels ETA_CL ETA_V ETA_KA
0.09 0.04 0.30
$SIGMA @labels PROP_ERR
0.02
$MAIN
double CL = TVCL * exp(ETA_CL);
double V  = TVV  * exp(ETA_V);
double KA = TVKA * exp(ETA_KA);
$PKMODEL ncmt=1, depot=TRUE
$TABLE
capture IPRED = CENT/V;
capture Y     = IPRED * (1 + EPS(1));
```

Running `ferx translate examples/warfarin.mod` produces:

```
# warfarin one-cpt oral
# Translated from mrgsolve by `ferx translate`.

[parameters]
  theta TVCL(0.2)
  theta TVV(10)
  theta TVKA(1.5)

  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.04
  omega ETA_KA ~ 0.3

  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = focei
  maxiter    = 300
  covariance = true
```

This is functionally equivalent to the hand-written `examples/warfarin.ferx`.

## CLI reference

```
ferx translate <model.mod|model.cpp> [-o out.ferx] [--check] [--force]
```

- `-o, --output PATH` — write `.ferx` to `PATH` instead of stdout.
- `--check` — parse only; exit 0 on success, non-zero on error. No output
  is written.
- `--force` — overwrite an existing `--output` target.
