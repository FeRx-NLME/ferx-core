# CLI Reference

The `ferx` command-line tool runs population PK estimation from model files and data.

## Usage

```bash
ferx <model.ferx> --data <data.csv> [--output <run.fitrx>] [--include-data] [--inits-from-nca[=METHOD]]
ferx <model.ferx> --simulate          [--output <run.fitrx>]
ferx check <model.ferx> [--data <data.csv>] [--json]
```

## Commands

### Fit with Data

```bash
ferx model.ferx --data data.csv
```

Parses the model file, reads the data, and runs the estimation method specified in `[fit_options]` (defaults to FOCEI).

### Simulate and Fit

```bash
ferx model.ferx --simulate
```

Parses the model file, generates simulated data from the `[simulation]` block, and fits the model to the simulated data. Requires a `[simulation]` block in the model file.

### Validate without Fitting (`check`)

```bash
ferx check model.ferx                  # parse + structural validation
ferx check model.ferx --data data.csv  # also run data-dependent checks
ferx check model.ferx --data data.csv --json
```

Runs the parser and every validation step that normally happens at the start of
a fit — *without* fitting — then reports the findings. This is a fast
`author → diagnose → fix` loop, especially useful for tooling and coding agents
that author model files programmatically.

- Without `--data`, parse / structural and model–option compatibility checks
  run (no data is read) — e.g. an SDE model paired with SAEM, or
  `optimizer = trust_region` on an IOV model.
- With `--data`, the dataset is read and the data-dependent checks run too:
  referenced covariates present, per-CMT scaling / error-model coverage,
  steady-state dosing well-formed, and non-negative typical-value lag time.
- `--json` emits a structured [check report](file-formats/check-report.md) to
  stdout instead of the human-readable summary.

Human output lists one diagnostic per line as
`severity[CODE] block:line: message`, with an indented `help:` line for any
suggestion, then a one-line summary:

```text
error[E_MISSING_COVARIATE]: Model references covariate(s) not found in data (case-sensitive): WGT. Available covariate columns: (none).
    help: available covariate columns: (none)
invalid: mymodel — 1 error(s), 0 warning(s)
```

The exit code is `0` when no errors are found (warnings alone still exit `0`),
`1` when any error is found, and `2` on a usage error. See the
[check report reference](file-formats/check-report.md) for the JSON schema and
the full error-code table.

## Output Files

Three files are always generated, named after the model file:

| File | Contents |
|------|----------|
| `{model}-sdtab.csv` | Per-observation diagnostics |
| `{model}-fit.yaml` | Parameter estimates, standard errors, and (when the covariance step ran) a `covariance_matrix:` block containing the full optimizer-space parameter covariance matrix |
| `{model}-timing.txt` | Wall-clock estimation time |

See [Output Files](output.md) for detailed format descriptions.

## Portable Fit Bundle (`--output`)

Pass `--output run.fitrx` to additionally write a portable `.fitrx` bundle —
a zip of JSON and CSV designed to be read from Rust, R, Python, or Julia. Use
`--include-data` to embed the input NONMEM CSV inside the bundle (off by
default).

```bash
ferx model.ferx --data data.csv --output run1.fitrx --include-data
```

## NCA-based starting values (`--inits-from-nca`)

Pass `--inits-from-nca` to derive starting values from the data via
non-compartmental analysis before fitting, overriding the model file's defaults.
The flag overrides the `inits_from_nca` `[fit_options]` key for this run. A bare
`--inits-from-nca` uses the `nca_sweep` strategy; pass an explicit method to pick
another:

```bash
ferx model.ferx --data data.csv --inits-from-nca           # = nca_sweep
ferx model.ferx --data data.csv --inits-from-nca=nca       # NCA only (fastest)
ferx model.ferx --data data.csv --inits-from-nca=nca_ebe   # EBE-refined sweep
```

See [NCA-based starting values](model-file/fit-options.md#nca-based-starting-values)
for what each strategy does.

See [the `.fitrx` format reference](file-formats/fitrx.md) for the full
schema.

## Console Output

### Progress

The estimation progress is printed to stderr, including:
- Model and data summary (subjects, observations, parameters)
- Optimizer iterations with OFV values (FOCE) or condNLL values (SAEM)
- Covariance step status
- Final parameter table

### Result Summary

A brief summary is printed to stdout:
```
Fit completed!
OFV: -280.1838
Elapsed: 0.496s
  TVCL = 0.132735
  TVV = 7.694842
  TVKA = 0.757498
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success (for `check`: no errors found) |
| 1 | Error (parse failure, data error, convergence failure; for `check`: errors found) |
| 2 | `check` usage error (e.g. missing model path) |

## Examples

```bash
# One-compartment oral warfarin model
ferx examples/warfarin.ferx --data data/warfarin.csv

# Two-compartment IV with FOCE
ferx examples/two_cpt_iv.ferx --data data/two_cpt_iv.csv

# SAEM estimation
ferx examples/warfarin_saem.ferx --data data/warfarin.csv

# Simulation-estimation study
ferx examples/warfarin.ferx --simulate
```

## Building

```bash
# Build the ferx binary
cargo build --release --features autodiff

# Run directly via cargo
cargo run --release --features autodiff --bin ferx -- model.ferx --data data.csv
```
