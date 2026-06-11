# Data Selection

> **Maturity: beta** — see [Feature Maturity](../maturity.md) for what this means.

The optional `[data_selection]` block lets you exclude records from the dataset
at read time without modifying the CSV file.  It is the ferx equivalent of
NONMEM's `$DATA IGNORE=` / `$DATA ACCEPT=`.

## Syntax

```
[data_selection]
  ignore = <expression>
  accept = <expression>
  ignore_subjects = [<id>, ...]
```

All three keys are optional and may be repeated.  Missing the block entirely
means "use all records".

---

## `ignore`

A record is **excluded** when the expression is true.

```
[data_selection]
  ignore = DV < 0.001
  ignore = EVID != 0
```

Multiple `ignore` lines are independent: a record is excluded when **any one**
of them matches.  That means each line is a separate reason to drop the record;
the lines do not combine with OR into a single expression.

Within a single line you can join sub-conditions with `&&` (all must hold):

```
[data_selection]
  ignore = EVID == 0 && DV < 0.001
```

Use `ignore` when you want to flag specific outlier values or dose rows with
no matching observation.

---

## `accept`

A record is **kept** only when the expression is true; it is excluded otherwise.

```
[data_selection]
  accept = BW >= 30 && BW < 48
```

Multiple `accept` lines are independent; a record is excluded when **any one**
accept condition fails.

Use `accept` when it is easier to state what the valid range is rather than
listing each invalid condition.

---

## `ignore_subjects`

Exclude all records for one or more subjects, given by their ID values:

```
[data_selection]
  ignore_subjects = [3, 17]
```

Single-subject shorthand (no brackets):

```
[data_selection]
  ignore_subjects = 3
```

Subject IDs are matched as strings (the same way they appear in the ID column
of the CSV).  An entirely excluded subject does not appear in any output.

---

## Supported columns

| Column | Type | Notes |
|--------|------|-------|
| `ID`   | string equality / inequality | `ID == "3"` or `ID == 3` |
| `TIME` | numeric | |
| `DV`   | numeric | |
| `EVID` | numeric (0/1/2/3/4) | |
| `AMT`  | numeric | |
| `CMT`  | numeric | |
| `RATE` | numeric | |
| `MDV`  | numeric | |
| `CENS` | numeric | |
| `II`   | numeric | |
| `SS`   | numeric | |
| any covariate column | numeric | case-insensitive (`BW`, `bw`, `Bw` all match) |

Column names in expressions are case-insensitive.

A record whose value for a referenced column is **missing** (`.`, blank, `NA`)
never matches a comparison on that column. For example, `ignore = DV < 0.001`
does **not** exclude dose rows, whose `DV` is `.`; only observation rows with a
real `DV` below the threshold are dropped. (You can still write
`ignore = EVID == 0 && DV < 0.001` to be explicit.)

---

## Evaluation order

For each record, the checks run in this order:

1. `ignore_subjects` — if the record's ID is in the list, exclude immediately.
2. `ignore` clauses — if **any** clause matches, exclude.
3. `accept` clauses — if **any** clause does **not** match, exclude.

A record must pass all three stages to be included.

---

## Exclusion summary

After reading the data, ferx reports what was dropped:

```
--- Data Selection ---
  Records read: 420  Obs excluded: 12  Doses excluded: 0  Other excluded: 0
  Fired ignore conditions:
    * ignore: DV < 0.001
```

The same information appears in the `exclusions:` block of the YAML output file
(`*-fit.yaml`):

```yaml
exclusions:
  n_records_total: 420
  n_obs_excluded: 12
  n_dose_excluded: 0
  n_other_excluded: 0
  fired_ignore:
    - "ignore: DV < 0.001"
```

`Other excluded` (`n_other_excluded`) counts excluded records that are neither a
scored observation nor a dose — EVID 2 (other event), EVID 3 (reset), and
missing-DV observation rows (EVID 0 with MDV 1). The three counts together
account for every excluded record.

Each fired condition is listed once. Because checks short-circuit on the first
match (see *Evaluation order*), a record is attributed to the **first** rule
that excludes it — so a condition that only ever matches records already removed
by an earlier rule will not appear in the fired list.

---

## Limitations

- `||` (OR) within a single expression is not supported. Use multiple lines
  instead — each line is already an independent "any of these reasons"
  condition.
- `AND` / `OR` keywords are not supported; use `&&` within a line.
- String comparisons are limited to `==` and `!=`. Ordered comparisons
  (`<`, `<=`, `>`, `>=`) on `ID` are rejected at parse time, since `ID` is a
  string label — use `==`/`!=` or `ignore_subjects`.
- `ADDL` and the occasion / IOV column are **not** filter targets; a condition
  referencing them is an inert no-op. Filter on the columns in the table above
  (or expand `ADDL` rows beforehand if you must select on dose number).
- Covariate values seen by a condition reflect the subject's full record
  history (last-observation-carried-forward across all rows), independent of
  which records the filter removes. This matters only when filtering on a
  time-varying covariate whose value differs on the rows being excluded.
- A column name that is not present in the dataset always evaluates to false
  (never fires), so a typo silently has no effect — check the exclusion summary
  if a condition is not being applied. (Covariate columns referenced by a
  condition are read even when a `[covariates]` block does not declare them, so
  filtering on an undeclared covariate works as expected.)
- The covariate table echo (`*-covtab.csv`, written only when a `[covariates]`
  block is declared) is a faithful echo of the **raw input file** and therefore
  still lists records that `[data_selection]` excluded from the fit. The fit
  itself, the residual table (`*-sdtab.csv`), and all `[output]` columns are
  computed from the filtered data, so they only contain retained records.

---

## Merging with R call conditions

When you also supply `ignore` or `accept` conditions via the R function
`ferx_selection()`, the model-file conditions and the R-call conditions are
**merged** (not replaced).  Exact-duplicate expressions are deduplicated
automatically; a condition specified in both places is only evaluated once.

See the ferx-r documentation for `ferx_selection()` and `ferx_fit()`.

---

## NONMEM equivalent

| NONMEM | ferx |
|--------|------|
| `$DATA IGNORE=C` | `[data_selection]  ignore = C == 1` |
| `$DATA IGNORE=(BW.GT.80)` | `[data_selection]  ignore = BW > 80` |
| `$DATA ACCEPT=(DV.GE.0.001)` | `[data_selection]  accept = DV >= 0.001` |
| `$DATA IGNORE=(ID.EQ.3) IGNORE=(ID.EQ.17)` | `[data_selection]  ignore_subjects = [3, 17]` |

ferx uses standard inequality operators (`>`, `>=`, `<`, `<=`, `==`, `!=`)
instead of NONMEM's Fortran-style `.GT.`, `.GE.`, etc.
