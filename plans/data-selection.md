# Plan: `[data_selection]` — IGNORE / ACCEPT support

**Branch:** `feat/data-selection` (to be cut)  
**Scope:** ferx-core (primary) + ferx-r (follow-up PR)

---

## Background

NONMEM's `$DATA IGNORE=` / `$DATA ACCEPT=` let users exclude or include records based on
per-row boolean conditions. ferx has no equivalent. The goal is to add:

1. A `[data_selection]` block in the `.ferx` model file (permanent, model-level exclusions).
2. A `ferx_selection()` R function for interactive/piped use.
3. Clear exclusion reporting in `ferx_runlog()`.

---

## Style decisions

### Expression syntax

Expressions use **bare (unquoted) comparisons** in the `.ferx` file, consistent with
how `[individual_parameters]` already writes `if (WT > 70) { ... }`. The `[data_selection]`
key=value parser splits on the first `=` and treats everything to the right as the raw
value string, so the `<`, `>` operators cause no ambiguity.

```toml
[data_selection]
ignore = DV < 0.001
ignore = EXCL == 1
accept = BW >= 30
```

Operators: `==`, `!=`, `>`, `>=`, `<`, `<=`.  
Column names are case-insensitive. Values are parsed as `f64`; string equality is
supported for non-numeric covariate columns.

**Why not NONMEM-style `.GT.`?** Less readable and no precedent in ferx syntax.

### R-side API (base-R, quoted strings + `c()` for multiples)

On the R side expressions are R character strings — not because of a design choice,
but because that is how R strings work. ferx-r has no rlang/dplyr dependency and uses
base-R throughout; there is no tidy-eval. Unquoted expressions would require
`substitute()`/`match.call()` for single conditions but break down for multiples
(`c(DV < 0.001, EXCL == 1)` would try to evaluate both in the calling environment).

Multiple conditions use a plain **character vector** with `c()`, which is the standard
base-R pattern for passing multiple string values to a function:

```r
# Single condition
ferx_selection(data, ignore = "DV < 0.001")

# Multiple ignore conditions (OR logic)
ferx_selection(data, ignore = c("DV < 0.001", "EXCL == 1"))

# Multiple accept conditions (AND logic)
ferx_selection(data, accept = c("BW >= 30", "AGE >= 18"))

# Combined
ferx_selection(data,
  ignore     = c("DV < 0.001", "EXCL == 1"),
  accept     = "BW >= 30",
  ignore_ids = c(3, 17))

# In a pipe into ferx_fit
data |>
  ferx_selection(ignore = c("DV < 0.001", "EXCL == 1"), accept = "BW >= 30") |>
  ferx_fit(model)

# Or directly on ferx_fit, skipping the explicit ferx_selection step
ferx_fit(model, data,
  ignore     = c("DV < 0.001", "EXCL == 1"),
  accept     = "BW >= 30",
  ignore_ids = c(3, 17))
```

The same expression — `DV < 0.001` — is bare in the `.ferx` file and wrapped in `""`
in R. The Rust parser strips surrounding quotes if present so both forms are accepted.

No tidy-eval. No new package dependencies.

---

## ignore vs accept semantics

### Inclusion rule

A record is **included** if and only if:

```
!any(ignore_i(row))  AND  all(accept_j(row))
```

Both independently contribute to exclusion. There is no logical conflict between them.
The implementation evaluates ignore first, then accept — this matters only for the
exclusion log (which condition is credited per record).

### User-facing description (docs / runlog — avoid OR/AND framing)

Do **not** describe this as "OR logic" and "AND logic" in documentation or error
messages — that framing is confusing in a pharmacometric context. Use behavioural
language, consistent with how NONMEM describes IGNORE/ACCEPT:

- **`ignore`**: "a record is excluded if it matches any ignore condition; each
  condition is an independent reason to exclude."
- **`accept`**: "a record is included only if it satisfies all accept conditions."

The key example to put front-and-centre in the docs and R function examples is the
**range filter**, because it is non-obvious and directly shows why multiple `accept`
conditions compose usefully:

```r
# Include only subjects within the protocol weight range
ferx_selection(data, accept = c("BW >= 30", "BW < 48"))
```

```toml
# .ferx equivalent
[data_selection]
accept = BW >= 30
accept = BW < 48
```

This is consistent with NONMEM, where multiple ACCEPT conditions are also all required
to hold (AND), and multiple IGNORE conditions each independently exclude (any match).

### Evaluation order (per record)

1. Evaluate each `ignore` expression in declaration order; stop at first match.
   - If any fires → record excluded; log which condition.
2. If no ignore matched, evaluate each `accept` expression in declaration order;
   stop at first failure.
   - If any fails → record excluded; log which condition.
3. If neither → record included.

### Can ignore and accept clash?

**Logically: no.** Both narrow the included set; combining them is always well-defined.

A user *can* write contradictory intent that empties the data:

| ignore | accept | Effective range included |
|--------|--------|--------------------------|
| `"BW > 80"` | `"BW >= 30"` | BW ∈ [30, 80] — fine |
| `"DV < 0.1"` | `"DV > 0"` | DV ∈ [0.1, ∞) — fine |
| `"BW > 60"` | `"BW > 60"` | Nothing — user error |
| `"EVID == 0"` | `"EVID == 0"` | No observations — user error |

User-error cases produce zero subjects or zero observations, which is caught as a fatal
error during `read_nonmem_csv` (same path as an empty CSV). No silent wrong results.

### Per-record vs per-subject

Both `ignore` and `accept` operate **per record** (per CSV row), identical to NONMEM.
For time-constant covariates the effect is naturally per-subject. For time-varying
covariates, partial-subject exclusion is possible and allowed (same as NONMEM).

**Warnings** are emitted for pathological partial exclusions:
- All dose records removed but observations remain for a subject.
- All observations removed but dose records remain for a subject.
- A subject's entire record set is removed.

### `ignore_subjects` shorthand

`ignore_subjects = [3, 17]` is syntactic sugar for `ignore = ID == 3` + `ignore = ID == 17` in the `.ferx` file (or `ignore = "ID == 3"` as an R string).
IDs are compared as strings (the `Subject.id` field is a `String`). Both numeric and
string IDs work.

---

## ferx-core changes

### Step 1 — `src/io/filter_expr.rs` (new file)

A two-level expression evaluator. No external parser crate.

**Level 1 — `FilterExpr`**: one `col op value` comparison.

**Level 2 — `FilterClause`**: one or more `FilterExpr` joined by `&&` within a
single string, consistent with how the existing ferx DSL uses `&&` in
`[individual_parameters]` and `[structural_model]` (e.g. `if (central / V > 0.5 && TAD < 24)`).
All sub-expressions must hold for the clause to fire.
`||` within a string is explicitly not supported — use multiple strings
(via `c()` in R or repeated lines in `.ferx`) for that.
`AND` / `OR` keyword forms are not used anywhere in the ferx DSL and are not supported here.

```rust
pub struct FilterExpr {
    col: String,   // case-folded to lowercase
    op:  CmpOp,
    rhs: FilterValue,
}

pub enum CmpOp { Eq, Ne, Lt, Le, Gt, Ge }
pub enum FilterValue { Num(f64), Str(String) }

/// One string from the user, potentially containing && / AND.
/// All sub-expressions must hold for the clause to evaluate true.
pub struct FilterClause {
    exprs: Vec<FilterExpr>,
}

impl FilterClause {
    /// Parse "DV < 0.001" or "BW >= 30 && BW < 48".
    /// Strips surrounding quotes so both bare (.ferx) and quoted (R) forms are accepted.
    /// `AND`/`OR` keyword forms and `||` are rejected with a clear parse error.
    pub fn parse(s: &str) -> Result<Self, String>

    pub fn eval(&self, ctx: &RowContext<'_>) -> bool {
        self.exprs.iter().all(|e| e.eval(ctx))
    }
}

pub struct RowContext<'a> {
    pub id:   &'a str,
    pub time: f64,
    pub dv:   f64,
    pub evid: u32,
    pub amt:  f64,
    pub cmt:  usize,
    pub rate: f64,
    pub mdv:  u32,
    pub cens: u8,
    pub ii:   f64,
    pub ss:   bool,
    pub covariates: &'a HashMap<String, f64>,
}
```

This gives two composable levels:

| Context | Composition | Effect |
|---------|-------------|--------|
| Multiple strings in `c()` / repeated lines | collection-level | independent: any match excludes (ignore) / all must pass (accept) |
| `&&` / `AND` within one string | clause-level | all must hold within that one clause |

Example showing where the distinction matters for `ignore`:

```r
# Excludes ALL observations AND any record with DV < 0.001 (including doses)
ferx_selection(data, ignore = c("EVID == 0", "DV < 0.001"))

# Excludes only observation records where DV < 0.001
ferx_selection(data, ignore = "EVID == 0 && DV < 0.001")
```

For `accept` the two forms produce the same result:
```r
ferx_selection(data, accept = c("BW >= 30", "BW < 48"))      # equivalent
ferx_selection(data, accept = "BW >= 30 && BW < 48")          # equivalent
```

Unit tests cover: all six operators, numeric and string RHS, case-insensitive column
names, `&&` separator, unknown column → `false` (never fire), `||` in string → parse
error with message "use multiple conditions instead of ||", `AND`/`OR` keywords →
parse error, malformed expression → parse error.

### Step 2 — `src/types.rs` — extend `FitOptions`

```rust
// In FitOptions:
pub ignore_exprs:    Vec<String>,   // raw strings; compiled to Vec<FilterClause> at read time
pub accept_exprs:    Vec<String>,
pub ignore_subjects: Vec<String>,   // compared as strings against Subject.id
```

Stored as raw strings so they serialise verbatim into `FitResult` / YAML output.

### Step 3 — `src/types.rs` — new `ExclusionSummary` + extend `Population`

```rust
pub struct ExclusionSummary {
    /// Subject IDs with zero remaining records after filtering.
    pub excluded_subject_ids: Vec<String>,
    /// Number of observation records (EVID==0) excluded.
    pub n_obs_excluded:       usize,
    /// Number of dose records (EVID!=0) excluded.
    pub n_dose_excluded:      usize,
    /// Total records read before filtering.
    pub n_records_total:      usize,
    /// Which ignore expressions fired at least once (in declaration order).
    pub fired_ignore:         Vec<String>,
    /// Which accept expressions rejected at least once (in declaration order).
    pub fired_accept:         Vec<String>,
}

// Added to Population:
pub exclusions: Option<ExclusionSummary>,
```

Also extend `FitResult` to carry `exclusions: Option<ExclusionSummary>`.

### Step 4 — `src/io/datareader.rs` — filter in `parse_subject`

`read_nonmem_csv` receives compiled `Vec<FilterExpr>` for ignore and accept (compiled
from the raw strings in `FitOptions`).

Inside the per-row loop in `parse_subject`, after parsing all standard fields and
covariates (but before the EVID branch that builds doses/obs), evaluate:

```rust
let ctx = RowContext { id, time, dv, evid, amt, cmt, rate, mdv, cens, ii, ss,
                       covariates: &cov_snapshot };

// ignore_subjects is pre-expanded into ignore_exprs as "ID == <id>" entries
let ignored = ignore_exprs.iter().any(|e| e.eval(&ctx));
if !ignored {
    let rejected = accept_exprs.iter().any(|e| !e.eval(&ctx));
    if rejected { /* log which accept expr */ } else { /* include row */ }
} else { /* log which ignore expr */ }
```

Track per-subject and aggregate into `ExclusionSummary`.

After building all subjects, emit warnings for pathological cases (doses-without-obs,
obs-without-doses, fully excluded subjects).

### Step 5 — `src/parser/model_parser.rs` — parse `[data_selection]`

New block type alongside `[fit_options]`. Parsed into the same `FitOptions` struct
(via the same `apply_fit_option` dispatcher for the new keys `ignore`, `accept`,
`ignore_subjects`).

`ignore_subjects` accepts a bracketed integer list: `[3, 17, 42]` → each ID is
appended to `ignore_subjects` as a string.

Repeated `ignore =` lines append to `ignore_exprs` (not override).
Repeated `accept =` lines append to `accept_exprs`.

### Step 6 — `src/io/output.rs` — CLI output

After the data line, print:

```
Data: warfarin.csv (32 subjects, 251 records total)
  Data selection applied:
    ignore: "DV < 0.001"    → 20 observations excluded
    ignore: subjects [3]    → subject 3 excluded (2 doses, 8 obs)
    accept: "BW >= 30"      → 0 records excluded
  Included: 31 subjects, 221 observations
```

If no selection is applied, the block is omitted.

### Step 7 — YAML output

The `fit.yaml` output gains an `exclusions:` key:

```yaml
exclusions:
  n_records_total: 251
  n_obs_excluded: 28
  n_dose_excluded: 2
  excluded_subject_ids: ["3"]
  fired_ignore: ["DV < 0.001", "ID == 3"]
  fired_accept: []
```

---

## ferx-r changes

> **Status note (2026-06-06):** the ferx-core side shipped and merged to `main`
> in PR #187 (merge `e572dde`). The public Rust surface the R side builds on is:
> `SelectionFilter::from_opts(ignore, accept, ignore_subjects)`,
> `read_nonmem_csv_filtered(path, cov_cols, iov, &filter)` and
> `read_nonmem_csv_with_covariates_filtered(...)`, and
> `ExclusionSummary { excluded_subject_ids, n_obs_excluded, n_dose_excluded,
> n_other_excluded, n_records_total, fired_ignore, fired_accept }` on both
> `Population` and `FitResult`. Parser keys are `ignore` / `accept` /
> `ignore_subjects`; dedup helper is `push_unique_expr`. The sections below are
> updated to match what actually shipped.

### `ferx_selection()` (new, `R/selection.R`)

```r
ferx_selection <- function(data, ignore = NULL, accept = NULL, ignore_ids = NULL)
```

- `data`: character path to CSV, or data.frame already in memory.
- `ignore`, `accept`: character vector of expression strings.
- `ignore_ids`: numeric or character vector of subject IDs.

**Behaviour:**
1. If `data` is a path, read it with `utils::read.csv()`.
2. Evaluate each expression using base-R on the data.frame columns (parse the same
   simple grammar: `col op value`). This is a thin R reimplementation of `FilterExpr`
   for the preview step — the canonical evaluation still happens in Rust at fit time.
3. Apply ignore-then-accept logic, producing a logical include-vector.
4. Return a `ferx_data` object: the filtered data.frame plus attributes:
   - `"source_path"` — original path (if given)
   - `"exclusions"` — named list: n_total, n_excluded_obs, n_excluded_dose,
     excluded_ids, fired_ignore, fired_accept
   - `"ignore"` / `"accept"` — the expression strings (passed to Rust)
   - `"ignore_ids"` — the ID vector

`ferx_fit()` accepts `ferx_data` objects as its `data` argument. When a `ferx_data`
is received, the exclusion expressions stored in its attributes are forwarded to
Rust as the `ignore`/`accept`/`ignore_subjects` selection rules and the **original
source CSV path** is passed through unchanged — **no temp CSV is written**. Rust
does the canonical filtering at read time (see the Rust-glue section), so the R
preview is purely informational (it lets users inspect what will be excluded
before committing to a fit). If `data` is an in-memory data.frame with no source
path, write it to a temp CSV once and pass that path.

### `ferx_selection_excluded(fit_or_data)` (new)

Returns a data.frame of the records that were excluded, reconstructed by re-reading
the source CSV and anti-joining against the included records.

```r
ferx_selection_excluded <- function(x) UseMethod("ferx_selection_excluded")
ferx_selection_excluded.ferx_data <- function(x) attr(x, "excluded_rows")
ferx_selection_excluded.ferx_fit  <- function(x) {
  # re-read source CSV, filter by exclusion summary stored in fit object
}
```

### `ferx_runlog()` update (`R/runlog.R`)

Add an exclusion block after the Data line, drawn from `fit$exclusions`:

```
── Data selection ─────────────────────────────────────────────────
  ignore: "DV < 0.001"    →  20 obs excluded
  ignore: IDs [3]         →  subject 3 excluded (8 obs, 2 doses)
  accept: "BW >= 30"      →  0 records excluded
  Excluded: 20 obs, 2 doses, 0 other (of 251 records)
  Included: 31 subjects, 221 observations / 32 subjects, 251 total
```

The counts come straight from `fit$exclusions` (mirroring `ExclusionSummary`):
`n_obs_excluded`, `n_dose_excluded`, `n_other_excluded` (EVID 2/3 and missing-DV
rows), `n_records_total`, `excluded_subject_ids`, `fired_ignore`, `fired_accept`.

Block is omitted when `fit$exclusions` is NULL (no selection applied).

### `ferx_fit()` update (`R/fit.R`)

- Add `ignore = NULL`, `accept = NULL`, `ignore_ids = NULL` parameters.
- Validate: must be character (ignore/accept), numeric or character (ignore_ids).
- Merge with any expressions from a `ferx_data` object passed as `data`.
- Inject as `settings` key/value pairs before the Rust call (same pathway as other
  settings). These are merged (OR'd / AND'd) with any `[data_selection]` block in
  the `.ferx` file — not overriding it.

### Rust glue (`src/rust/src/lib.rs`) — **read-ordering fix is the crux**

⚠️ **The glue currently bypasses the filter.** `ferx_rust_fit` reads the data with
`ferx_core::read_nonmem_csv(path, None, iov)` *before* the settings loop, then calls
`ferx_core::fit()` separately. It does **not** go through ferx-core's file-based
entry points, so populating `opts.ignore_exprs` via settings alone changes nothing —
`fit()` never re-filters. The fix:

1. Parse the model and clone `opts` as today.
2. Run the settings loop first, so `ignore`/`accept`/`ignore_subjects` land in
   `opts.ignore_exprs` / `accept_exprs` / `ignore_subjects` (they are **not** on the
   RESERVED list; they flow through `apply_fit_option` like other settings, and
   `push_unique_expr` dedups exact-string repeats across the `.ferx` file and the R
   call).
3. **Then** read the data, building a filter when any rule is set:
   `let filter = SelectionFilter::from_opts(&opts.ignore_exprs, &opts.accept_exprs, &opts.ignore_subjects)?;`
   and call `read_nonmem_csv_filtered(path, None, iov, &filter)` instead of
   `read_nonmem_csv`. (Use `read_nonmem_csv_with_covariates_filtered` if/when the
   glue grows a `[covariates]` path; today it passes `None`.)
4. After the fit, expose `result.exclusions` as `fit$exclusions` — a named list
   mirroring `ExclusionSummary` (include `n_other_excluded`).

The dedicated `ferx_fit(ignore=, accept=, ignore_ids=)` args are merged into the
same `settings` vectors R-side before the call (same pattern as `optimizer_trace` /
`inits_from_nca`), so a single Rust pathway handles `.ferx`-file and R-call rules.

---

## Priority: `.ferx` file vs R call

Conditions from both sources are **merged** (appended), not overriding:
- `ignore` conditions from `.ferx` `[data_selection]` and from `ferx_fit()` /
  `ferx_selection()` are OR'd together.
- `accept` conditions from both sources are AND'd together.

This is additive: the model file expresses permanent analysis-level exclusions;
the R call adds run-specific ones. Neither silently drops the other.

### Deduplication of identical expressions

If the same expression string appears in both the `.ferx` file and the R call (e.g.
both specify `ignore = "DV < 0.001"`), appending naively causes the condition to be
evaluated twice per record. The result is still correct (row excluded once), but the
exclusion log would list the condition twice and counts would appear doubled.

**Rule:** when collecting expressions from all sources into the final `Vec<String>`,
deduplicate on exact string match after trimming leading/trailing whitespace. The
second occurrence of an already-present string is silently dropped.

```rust
fn push_unique(vec: &mut Vec<String>, s: String) {
    let trimmed = s.trim().to_string();
    if !vec.iter().any(|e| e == &trimmed) {
        vec.push(trimmed);
    }
}
```

Textually different but semantically equivalent expressions (e.g. `"DV<0.001"` vs
`"DV < 0.001"`, or different operator spacing) are **not** deduplicated — normalising
expressions is not worth the added complexity. Both would evaluate; the result is
correct (identical rows excluded) and the log clearly shows both strings.

For scalar settings (e.g., `maxiter`), R call continues to override the `.ferx` file.

---

## Tests required

### ferx-core (unit, Tier 1)

- `FilterExpr::parse`: all six operators, numeric + string RHS, case-insensitive
  column, bad syntax → error.
- `FilterExpr::eval`: truth table across operators.
- `parse_subject` with ignore conditions: excluded obs count, dose count, partial
  subject warning.
- Pathological cases: all subjects excluded → error; contradictory ignore+accept
  → all excluded → error.

### ferx-core (integration, Tier 2)

- `fit()` with `ignore_exprs` set: returns Ok after a few iterations, `FitResult`
  carries correct `exclusions`.
- `ignore_subjects` shorthand: equivalent to equivalent `ignore` expression.

### ferx-r (unit)

- `ferx_selection()`: returns `ferx_data`, correct attributes, base-R evaluation
  matches Rust evaluation on a small synthetic CSV.
- `ferx_selection_excluded.ferx_data()`: returns correct rows.

---

## Docs

- `docs/model-file/` — new page `data-selection.qmd` describing `[data_selection]`
  block, both keywords, expression syntax, worked examples.
- `docs/_quarto.yml` — add entry.
- `docs/faq.qmd` — NONMEM equivalence table row: `$DATA IGNORE=` → `[data_selection] ignore =`.

### ferx-r package docs (`man/ferx_selection.Rd`)

The `ferx_selection()` roxygen examples must include:

1. **Single ignore** — excluding below-LLOQ observations:
   ```r
   ferx_selection(data, ignore = "DV < 0.001")
   ```

2. **Multiple ignore** — each condition independently excludes:
   ```r
   ferx_selection(data, ignore = c("DV < 0.001", "EXCL == 1"))
   ```

3. **Range filter with accept** — the primary example showing why multiple
   `accept` conditions are useful; do not describe as "AND logic", use:
   *"a record is included only if it satisfies all accept conditions"*:
   ```r
   # Include only subjects within the protocol weight range [30, 48)
   ferx_selection(data, accept = c("BW >= 30", "BW < 48"))
   ```

4. **Combined ignore + accept** — excludes flagged records and restricts range:
   ```r
   ferx_selection(data,
     ignore = "EXCL == 1",
     accept = c("BW >= 30", "BW < 48"))
   ```

5. **Pipe into ferx_fit**:
   ```r
   data |>
     ferx_selection(ignore = "DV < 0.001", accept = c("BW >= 30", "BW < 48")) |>
     ferx_fit(model)
   ```

The `@description` section must state: *"Behaviour is consistent with NONMEM's
`$DATA IGNORE=` / `$DATA ACCEPT=`: each ignore condition independently excludes
matching records; all accept conditions must hold for a record to be included."*
