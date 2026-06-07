# Data Selection (IGNORE / ACCEPT)

This example shows the `[data_selection]` block, which filters dataset rows at read time without modifying the CSV. The complete model file is [`examples/warfarin_data_selection.ferx`](https://github.com/FeRx-NLME/ferx-core/blob/main/examples/warfarin_data_selection.ferx).

## When to use

Use `[data_selection]` when you want to:
- Exclude observations below a lower limit of quantification (LLOQ surrogate)
- Restrict the analysis to a specific study part, occasion, or compartment
- Mirror a NONMEM `$DATA IGNORE=` or `ACCEPT=` statement in the model file instead of pre-processing the data

## Model file

This is the contents of [`examples/warfarin_data_selection.ferx`](https://github.com/FeRx-NLME/ferx-core/blob/main/examples/warfarin_data_selection.ferx):

```
# One-compartment oral PK model (warfarin) with data-selection filtering

[parameters]
  theta TVCL(0.134, 0.001, 10.0)
  theta TVV(8.1, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)

  omega ETA_CL ~ 0.07
  omega ETA_V  ~ 0.02
  omega ETA_KA ~ 0.40

  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[data_selection]
  # Drop observations below a surrogate LLOQ of 1.0 mg/L.
  ignore = DV < 1.0

[fit_options]
  method   = foce
  maxiter  = 300
  gradient = fd
```

The `[data_selection]` block accepts two optional directives:

| Key | Effect |
|-----|--------|
| `ignore = <expr>` | Drop any row for which `<expr>` evaluates to true |
| `accept = <expr>` | Keep only rows for which `<expr>` evaluates to true |

Only one of `ignore` / `accept` may be specified per model.

The filter expression can reference any column in the dataset (`DV`, `TIME`, `OCC`, `CMT`, covariates, …). Dose rows (`EVID = 1`) are never dropped regardless of the filter, preserving dose history for all subjects.

## Running the fit

```bash
ferx examples/warfarin_data_selection.ferx --data data/warfarin.csv
```

Or via the Rust API:

```rust
let result = fit_from_files("examples/warfarin_data_selection.ferx", "data/warfarin.csv")?;
println!("Excluded observations: {:?}", result.warnings);
```

## Interpreting output

The console summary reports how many observations were excluded:

```
Data: 32 subjects, 192 observations (12 excluded by data_selection)
```

Excluded rows do not contribute to the OFV, EBE optimisation, or sdtab output. Dose rows are always retained.

## Tips

- **Prefer `ignore` for LLOQ exclusion** — `ignore = DV < lloq` is the direct analogue of NONMEM `IGNORE=(DV.LT.lloq)`.
- **Column names are case-insensitive** — `ignore = dv < 1.0` and `ignore = DV < 1.0` are equivalent.
- **Combine with `[data_selection]` for study-design subsets**: `accept = OCC == 1` restricts the fit to the first occasion without touching the data file.
- **Dose rows are protected**: the engine never excludes EVID=1 rows, so PK history is always complete even when observations are filtered.
- **BLOQ observations**: if you use the M3 method (`[fit_options] bloq = m3`) alongside `[data_selection]`, set the filter so BLOQ-flagged rows (CENS=1) are *not* excluded — the M3 likelihood uses them. Use a tighter filter or remove the `ignore` entirely and rely on M3 alone.
