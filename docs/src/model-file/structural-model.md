# Structural Model

The `[structural_model]` block specifies the pharmacokinetic model used to generate predictions.

## Analytical PK Models

For standard compartmental models, use the `pk` keyword with a model function:

```
pk MODEL_NAME(param=VALUE, param=VALUE, ...)
```

### Available Models

| Model Function | Compartments | Route | Required Parameters |
|---------------|--------------|-------|---------------------|
| `one_cpt_iv` | 1 | IV (bolus and/or infusion) | `cl`, `v` |
| `one_cpt_oral` | 1 | Oral | `cl`, `v`, `ka` |
| `two_cpt_iv` | 2 | IV (bolus and/or infusion) | `cl`, `v1`, `q`, `v2` |
| `two_cpt_oral` | 2 | Oral | `cl`, `v1`, `q`, `v2`, `ka` |
| `three_cpt_iv` | 3 | IV (bolus and/or infusion) | `cl`, `v1`, `q2`, `v2`, `q3`, `v3` |
| `three_cpt_oral` | 3 | Oral | `cl`, `v1`, `q2`, `v2`, `q3`, `v3`, `ka` |

Each model has a `*_compartment_*` long-form alias (e.g. `three_compartment_iv`); the short and long names are interchangeable.

There is no separate bolus or infusion variant: every IV model selects the closed form per dose from the `RATE` column (`RATE=0` ⇒ bolus, `RATE>0` ⇒ infusion). A single subject can mix the two. This matches NONMEM, nlmixr2, and Monolix.

The earlier `*_iv_bolus` and `*_infusion` model names were retired in #176. The parser now rejects them with a migration message pointing at the unified `*_iv` name.

### Examples

One-compartment oral:
```
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
```

Two-compartment IV (bolus, infusion, or a mix — driven by RATE):
```
[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)
```

Two-compartment oral:
```
[structural_model]
  pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)
```

Three-compartment IV (note that `q2`/`q3` and `v2`/`v3` distinguish the two peripheral compartments):
```
[structural_model]
  pk three_cpt_iv(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3)
```

### Bioavailability

For oral models, bioavailability (F) defaults to 1.0. To estimate it, define an `F` parameter in `[individual_parameters]` -- it will be automatically used by the oral PK functions.

This applies to [ODE models](ode-models.md) too: an `F` parameter is applied when the dose enters the compartment (`F · AMT`), matching NONMEM and the analytical PK functions. Do not also multiply by `F` in the ODE right-hand side.

### Lagtime

All `pk` models accept an optional `lagtime=` parameter (or its NONMEM-style alias `alag=`) that delays the effective start of every dose record by the parameter's value. Defaults to `0.0` when omitted, so existing models behave identically. See [Lagtime](lagtime.md) for semantics, examples, and limitations.

### Dose Handling

Analytical IV models support:
- **Bolus doses**: Instantaneous input (when `RATE=0` in data)
- **Infusions**: Zero-order input (when `RATE>0` in data)
- **Mixed**: A single subject may receive both bolus and infusion doses; the route is read per event from `RATE`
- **Steady-state**: Pre-computed steady-state concentrations (when `SS=1` and `II>0` in data)
- **Dose superposition**: Multiple doses are handled by summing contributions from each dose event

### Numerical Stability

The analytical solutions include special handling for:
- **Near-equal rate constants**: When absorption and elimination rates are similar (KA ~ k), L'Hopital's rule is used to avoid division by zero
- **Two-compartment eigenvalues**: Vieta's formula is used for robust computation of alpha and beta

## ODE Models

For non-standard kinetics (e.g., saturable elimination), use the ODE specification:

```
[structural_model]
  ode(obs_cmt=COMPARTMENT_NAME, states=[state1, state2, ...])
```

When the observable is a derived quantity (e.g. amount / V), use the
amount-only ODE form and supply `[scaling] y = <expr>`:

```
[structural_model]
  ode(states=[depot, central])

[scaling]
  y = central / V
```

See [ODE Models](ode-models.md) for full ODE syntax and
[Scaling](scaling.md) for the `[scaling]` block.
