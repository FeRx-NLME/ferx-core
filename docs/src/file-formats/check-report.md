# Check Report (`ferx check --json`)

`ferx check <model.ferx> [--data <data.csv>] --json` emits a structured JSON
report describing every validation finding, instead of the human-readable
summary. The report is designed to be consumed programmatically — by editor
tooling, CI, or a coding agent authoring model files — so findings carry a
stable machine-readable `code` rather than only prose.

This is the same data the Rust API returns from
[`validate_model_file`](../api/parsing.md); see also the
[CLI reference](../cli.md) for the human-readable form and exit codes.

## Top-level shape

```json
{
  "valid": false,
  "model": "mymodel",
  "data": "data/mydata.csv",
  "diagnostics": [
    {
      "severity": "error",
      "code": "E_MISSING_COVARIATE",
      "message": "Model references covariate(s) not found in data (case-sensitive): WGT. Available covariate columns: (none).",
      "suggestion": "available covariate columns: (none)"
    }
  ]
}
```

| Field | Type | Notes |
|-------|------|-------|
| `valid` | bool | `true` when there are **no** `error`-severity diagnostics (warnings alone keep it `true`). |
| `model` | string | Model file stem. |
| `data` | string | Present only when `--data` was supplied. |
| `diagnostics` | array | Every finding; see below. May be empty. |

## Diagnostic object

| Field | Type | Notes |
|-------|------|-------|
| `severity` | `"error"` \| `"warning"` | Only `error` affects `valid` and the exit code. |
| `code` | string | Stable identifier (see table below). |
| `message` | string | Human-readable description. |
| `block` | string | *Optional.* Owning model block, e.g. `"error_model"`. Omitted when not attributable to one block. |
| `line` | integer | *Optional.* 1-based source line — see the caveat below. |
| `suggestion` | string | *Optional.* Actionable hint. |

Optional fields are omitted entirely when absent (not emitted as `null`).

## Error / warning codes

| Code | Severity | Meaning |
|------|----------|---------|
| `E_PARSE` | error | The model file failed to parse (catch-all for parser errors). |
| `E_MISSING_BLOCK` | error | A required `[block]` is absent. `block` names which one. |
| `E_NN_FEATURE_DISABLED` | error | A `[covariate_nn]` block requires building with `--features nn`. |
| `E_MISSING_COVARIATE` | error | The model references a covariate not present in the data (case-sensitive). |
| `E_PER_CMT_SCALING` | error | An observed compartment lacks a per-CMT scaling entry. |
| `E_PER_CMT_ERROR_MODEL` | error | An observed compartment lacks a per-CMT `[error_model]` entry. |
| `E_DATA` | error | The `--data` file could not be read or parsed. |
| `E_SDE_INCOMPATIBLE` | error | An SDE (`[diffusion]`) model used with an incompatible method (`saem`, `gn`, `gn_hybrid`) or `gradient_method = ad`. |
| `E_AD_UNAVAILABLE` | error | `gradient_method = ad` requested, but the binary was built without the `autodiff` feature. Use `auto`/`fd`, or rebuild with the Enzyme toolchain. Only emitted by non-autodiff builds. |
| `E_IMP_CHAIN` | error | `imp` is mis-placed in a method chain — repeated, or (with `is_eval_only = true`) not the terminal stage. The estimating `imp` may sit anywhere; an evaluation-only `imp` must be terminal. |
| `E_OPTIMIZER_IOV` | error | `optimizer = trust_region` used with an IOV model (`n_kappa > 0`). |
| `W_STEADY_STATE_II` | warning | SS=1 doses with missing / non-positive `II` (treated as non-SS). |
| `W_STEADY_STATE_INFUSION` | warning | SS=1 infusion with `T_inf > II` (overlapping pulses; SS skipped). |
| `W_SDE_RESET` | warning | EVID=3/4 resets under an SDE `[diffusion]` model are not honoured. |
| `W_EXPERIMENTAL_SDE` | warning | The model uses stochastic differential equations (`[diffusion]`), an [experimental](../maturity.md) feature. |
| `W_EXPERIMENTAL_NN` | warning | The model uses neural-network components (`[covariate_nn]`), an [experimental](../maturity.md) feature. |
| `W_NEGATIVE_LAGTIME` | warning | Lag time is negative at the initial typical-value point. |
| `E_DERIVED_NAME_CONFLICT` | error | A `[derived]` name clashes with a built-in sdtab column, theta, eta, or individual-parameter name. |
| `W_DERIVED_COVARIATE_SHADOW` | warning | A `[derived]` name shadows a covariate (allowed but may be confusing). |
| `W_DERIVED_STEP_IGNORED` | warning | `step=` given for a DV-based integral (ignored; DV integrals always use observation times). |
| `E_OUTPUT_UNKNOWN_COLUMN` | error | A name in `[output]` is not recognised as a covariate, individual parameter, or derived expression. |
| `W_OUTPUT_DUPLICATE` | warning | A name in `[output]` is already written to sdtab automatically (e.g. `TAFD`, `TAD`, an eta name). |
| `W_ADDL_MISSING_II` | warning | ADDL > 0 on a dose row but II is zero or missing; additional doses were not expanded. |
| `W_IOV_OCC_MISSING` | warning | Some rows in the IOV occasion column had missing or unparseable values; those rows were assigned occasion=0. |
| `W_MISSING_DV` | warning | One or more `EVID=0` rows had a missing `DV` (`.`/`NA`/blank) but were not marked `MDV=1`; they were skipped rather than scored as `DV=0`. |
| `E_IOV_MISSING_OCC` | error | Model declares kappa (IOV) parameters but no occasion labels were found in the dataset. Set `iov_column` in `[fit_options]`. |

Codes are stable; new ones may be added over time. Treat an unrecognised code
as a generic finding of its given `severity`.

## Line-number caveat

`line` is currently **block-level**: when present it points at the `[block]`
header that owns the finding, not the exact offending token or column. A finding
that is not attributable to a single block (for example a missing-covariate
error, where the reference may appear in several blocks) omits `line`. A missing
*required* block omits `line` too, since the block has no header in the source.
Token/column-level spans are a possible future enhancement.
