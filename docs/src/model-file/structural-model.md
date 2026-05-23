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
| `one_cpt_iv_bolus` | 1 | IV bolus | `cl`, `v` |
| `one_cpt_oral` | 1 | Oral | `cl`, `v`, `ka` |
| `one_cpt_infusion` | 1 | IV infusion | `cl`, `v` |
| `two_cpt_iv_bolus` | 2 | IV bolus | `cl`, `v1`, `q`, `v2` |
| `two_cpt_oral` | 2 | Oral | `cl`, `v1`, `q`, `v2`, `ka` |
| `two_cpt_infusion` | 2 | IV infusion | `cl`, `v1`, `q`, `v2` |

### Examples

One-compartment oral:
```
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
```

Two-compartment IV bolus:
```
[structural_model]
  pk two_cpt_iv_bolus(cl=CL, v1=V1, q=Q, v2=V2)
```

Two-compartment oral:
```
[structural_model]
  pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)
```

### Bioavailability

For oral models, bioavailability (F) defaults to 1.0. To estimate it, define an `F` parameter in `[individual_parameters]` -- it will be automatically used by the oral PK functions.

### Lagtime

All `pk` models accept an optional `lagtime=` parameter (or its NONMEM-style alias `alag=`) that delays the effective start of every dose record by the parameter's value. Defaults to `0.0` when omitted, so existing models behave identically. See [Lagtime](lagtime.md) for semantics, examples, and limitations.

### Dose Handling

Analytical models support:
- **Bolus doses**: Instantaneous input (default when `RATE=0` in data)
- **Infusions**: Zero-order input (when `RATE>0` in data)
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
