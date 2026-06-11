# Covariates

> **Maturity: stable** — see [Feature Maturity](../maturity.md) for what this means.

The optional `[covariates]` block declares which dataset columns are
covariates and whether each is **continuous** or **categorical**. This is a
declaration of *availability* — it does **not** mean the covariate is used in
the structural model, only that it is potentially available. Once declared,
ferx-core echoes the covariate columns back on the fit result (the
[covariate table](#covariate-table)), which downstream tooling (e.g. the R
package) can use for summary statistics and covariate-search workflows.

## Syntax

Two line forms are accepted and may be mixed. The type is one of
`continuous`/`cont` or `categorical`/`cat` (case-insensitive):

```
[covariates]
  WT   continuous
  HT   continuous
  CRCL continuous
  SEX  categorical
  RACE categorical
```

The equivalent terser `TYPE: NAME, ...` form:

```
[covariates]
  continuous: WT, HT, CRCL
  categorical: SEX, RACE
```

Covariate names are **case-sensitive** and must match the CSV header exactly.

## Semantics

- **Optional & backward-compatible.** When the block is absent, behaviour is
  unchanged: every non-standard CSV column is auto-detected as a covariate.
- **Authoritative for the table and typing.** Only the listed columns appear in
  the covariate table and carry a declared type; other non-standard columns
  (e.g. `STUDY`, `DATE`) are not tabled.
- **Undeclared-but-used is a warning, not an error.** A covariate used in
  `[individual_parameters]` but missing from `[covariates]` is still usable —
  ferx reads it (leniently) and emits a warning recommending you declare it so
  its type is recorded and it appears in the table. Declaring a covariate the
  model does *not* use is also fine — that is the point.
- **Validation against the data.** A *declared* column that is absent from the
  dataset is an error (`E_MISSING_COVARIATE`).

## Categorical covariates must be numerically coded

Covariate values are carried as floating-point numbers, so categorical
covariates must be encoded as integer levels in the data (e.g. `SEX` as `0`/`1`,
not `"M"`/`"F"`). Under a `[covariates]` block this is enforced: a non-numeric
value in a declared covariate is a hard error rather than a silent coercion to
`0.0`. (In the legacy auto-detect path — no `[covariates]` block — a non-numeric
covariate value fails to parse, is dropped, and the covariate evaluates to `0.0`
in the model, preserving prior behaviour.)

Missing values (blank, `.`, `NA`) are permitted and recorded as missing.

`ferx check` reads through the same covariate-aware path the fit uses, so a
declared column that is absent (`E_MISSING_COVARIATE`) or non-numeric
(`E_COVARIATE_NOT_NUMERIC`), or a referenced covariate missing from the data, is
reported at check time rather than only failing once the fit starts.

## Covariate table

When a `[covariates]` block is present and the fit is launched from a data file,
the result carries a covariate table (`FitResult.covariate_table`) echoing the
declared columns:

- Columns: `ID`, `TIME`, `EVID`, then one column per declared covariate.
- **One row per input dataset record** — including dose and other-event rows.
  (This differs from the `sdtab` diagnostic table, which has observation rows
  only.)
- Missing values are written as empty cells (the in-memory representation uses
  `NaN`).

The CLI writes it to `{model}-covtab.csv` alongside `{model}-sdtab.csv` whenever
the model declares covariates.

## Example

See `examples/two_cpt_oral_cov.ferx`, which declares:

```
[covariates]
  WT   continuous
  CRCL continuous
```

Both `WT` and `CRCL` are used in `[individual_parameters]` to scale `CL` and
`V1`, so they are declared here. A covariate used in the model but left out of
the block still works, but the parser emits a warning recommending it be
declared (so its type is recorded and it appears in the covariate table).
