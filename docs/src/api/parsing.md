# Parsing

## Model Parsing

### `parse_model_file()`

Parse a `.ferx` file into a `CompiledModel`. Only the core model blocks are processed (`[parameters]`, `[individual_parameters]`, `[structural_model]`, `[error_model]`).

```rust
pub fn parse_model_file(path: &Path) -> Result<CompiledModel, String>
```

### `parse_model_string()`

Parse a model from a string instead of a file.

```rust
pub fn parse_model_string(content: &str) -> Result<CompiledModel, String>
```

**Example:**
```rust
let model_str = r#"
[parameters]
  theta TVCL(0.1, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma ADD_ERR ~ 1.0

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ additive(ADD_ERR)
"#;

let model = parse_model_string(model_str)?;
```

### `parse_full_model_file()`

Parse a complete model file including `[fit_options]` and `[simulation]` blocks.

```rust
pub fn parse_full_model_file(path: &Path) -> Result<ParsedModel, String>
```

Returns a `ParsedModel` which contains:

```rust
pub struct ParsedModel {
    pub model: CompiledModel,
    pub simulation: Option<SimulationSpec>,
    pub fit_options: FitOptions,
    /// 1-based source line of each unnamed `[block]` header, keyed by the
    /// lowercased block type (e.g. `"individual_parameters" -> 7`).
    pub block_lines: std::collections::HashMap<String, usize>,
}
```

## Validation

### `validate_model_file()`

Validate a model file (and optionally a dataset) **without fitting**, returning
a structured [`CheckReport`](../file-formats/check-report.md). This is the engine
behind the [`ferx check`](../cli.md) CLI command: it runs the parser plus every
data-independent and data-dependent check, collecting *all* findings rather than
stopping at the first (as `fit()` does).

```rust
pub fn validate_model_file(model_path: &str, data_path: Option<&str>) -> CheckReport
```

**Example:**
```rust
let report = validate_model_file("model.ferx", Some("data.csv"));
if !report.valid {
    for d in &report.diagnostics {
        eprintln!("{:?} [{}]: {}", d.severity, d.code, d.message);
    }
}
```

When `data_path` is `None`, only parse / structural checks run. The same
validators feed `fit()` (via the internal `check_model_data` /
`check_model_data_warnings` functions), so a clean `check` and a `fit()` agree
on what is and isn't valid.

## Data Parsing

### `read_nonmem_csv()`

Read a NONMEM-format CSV file into a `Population`.

```rust
pub fn read_nonmem_csv(
    path: &Path,
    covariate_columns: Option<&[&str]>,
) -> Result<Population, String>
```

**Parameters:**
- `path`: Path to the CSV file
- `covariate_columns`: Optional list of covariate column names. If `None`, all non-standard columns are auto-detected as covariates.

**Example:**
```rust
// Auto-detect covariates
let pop = read_nonmem_csv(Path::new("data.csv"), None)?;

// Explicit covariate list
let pop = read_nonmem_csv(
    Path::new("data.csv"),
    Some(&["WT", "CRCL", "AGE"]),
)?;

println!("{} subjects, {} observations",
         pop.subjects.len(), pop.n_obs());
```

### Data Processing Details

- Column names are matched case-insensitively
- Standard NONMEM columns (`ID`, `TIME`, `DV`, `EVID`, `AMT`, `CMT`, `RATE`, `MDV`, `II`, `SS`) are recognized automatically
- Missing values (`.`, empty string) are handled appropriately
- Rows with `EVID=1` are treated as dose events
- Rows with `EVID=0` and `MDV=0` are treated as observations
- Time-constant covariates use the first non-missing value per subject
- Time-varying covariates use Last Observation Carried Forward (LOCF) per event — `[individual_parameters]` is re-evaluated at each dose and observation row using that row's covariate values (NONMEM-equivalent semantics). Currently supported on 1- and 2-compartment IV bolus / infusion models and all ODE-defined models; oral and 3-compartment models fall back to a single first-row snapshot
