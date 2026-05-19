# Multi-Start: Michaelis-Menten Elimination

Michaelis-Menten (saturable) elimination is the canonical example of a local-minimum problem in NLME modelling. The Vmax/Km pair is only weakly identifiable from sparse PK data: many (Vmax, Km) combinations produce similar predicted concentrations, so the OFV surface contains a narrow ridge. A single-start FOCEI run is sensitive to the initial values — starting far from the true optimum, the optimizer often settles on an inflated Vmax/Km pair with a higher OFV than the true optimum.

## The problem

For the model below with true parameters Vmax ≈ 3.5 mg/h and Km ≈ 5.5 mg/L, starting with Vmax = 12 and Km = 20 (both roughly 3× too high) a single FOCEI run may converge to a local minimum rather than the global one. The two solutions predict similar concentration-time profiles on the observed time grid but differ in OFV by several units.

## The model file

```ferx
{{#include ../../../examples/mm_multistart.ferx}}
```

## What multi-start does

With `n_starts = 8` and `start_sigma = 0.5`:

- **Start 0** uses the exact initial values from `[parameters]` (Vmax=12, Km=20).
- **Starts 1–7** multiply each log-packed theta by `exp(N(0, 0.5))`, giving a spread of roughly ×0.6–×1.6 around the initial values. With `start_sigma = 0.5` this is wider than the default 0.3 — appropriate for a ridge-shaped surface.
- All 8 runs execute in parallel. On an 8-core machine wall-clock time equals a single run.
- The converged run with the lowest OFV is returned. If start 0 was already at the global optimum, nothing changes. If one of the perturbed starts found a better solution, a warning on `FitResult.warnings` reports which start won and its OFV.

## Running it

```bash
# Simulate data first
cargo run --release -- examples/mm_multistart.ferx --simulate

# Fit with multi-start
cargo run --release -- examples/mm_multistart.ferx --data mm_multistart-sim.csv
```

## When to use multi-start

| Model feature | Risk of local minima | Recommended n_starts |
|---------------|---------------------|----------------------|
| Linear 1-cpt / 2-cpt | Very low | 1 (default) |
| Michaelis-Menten elimination | High | 4–8 |
| Full-block omega (≥ 3 etas) | Moderate | 4 |
| Many correlated covariates | Moderate | 4–8 |

`start_sigma = 0.3` (the default, ≈ 30% CV) is appropriate for most models. For ridge-shaped surfaces like Vmax/Km, `start_sigma = 0.4–0.5` explores a wider neighbourhood.
