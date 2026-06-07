# Output Columns (`[output]`)

The optional `[output]` block lists additional columns to include in the sdtab beyond the mandatory minimum. Each entry is a bare name, one or more per line (whitespace or newlines between names).

```
[output]
  WT AGE SEX
  CLi
```

## What can be listed

| Category | Example names | Notes |
|----------|--------------|-------|
| Covariates | `WT`, `AGE`, `SEX` | Written as-is from the data |
| Individual parameters | `CL`, `V` | EBE-derived per-subject value |
| Derived column names | `AUC`, `Cmax` | Defined in the `[derived]` block |
| Eta names | `ETA_CL`, `ETA_V` | ETAs are not sdtab columns; declaration is ignored with `W_OUTPUT_DUPLICATE` |

### Mandatory minimum (always present)

The following columns are always written to sdtab regardless of `[output]`:

`ID`, `TIME`, `DV`, `CENS`, `OCC`, `CMT`, `PRED`, `IPRED`, `CWRES`, `IWRES`, `EBE_OFV`, `N_OBS`, `TAFD`, `TAD`

Listing any of these in `[output]` is harmless but will emit `W_OUTPUT_DUPLICATE`.

## Diagnostics

| Code | Severity | Cause |
|------|----------|-------|
| `E_OUTPUT_UNKNOWN_COLUMN` | error | A name is not a covariate, individual parameter, or derived expression |
| `W_OUTPUT_DUPLICATE` | warning | A name is already in the mandatory minimum |

## Interaction with `[derived]`

A name defined in `[derived]` is automatically available in `[output]` without needing to be listed again — it is written to sdtab as soon as it appears in `[derived]`. Use `[output]` for covariates and individual parameters that are not computed by `[derived]`.
