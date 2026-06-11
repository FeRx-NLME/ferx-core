# Steady-State Doses (SS=1)

> **Maturity: stable** — see [Feature Maturity](../maturity.md) for what this means.

A dose record with `SS=1` and `II > 0` tells ferx-core that, at the time
of the record, the compartmental state is at steady state under repeated
dosing of that amount/rate every `II` time units. The record itself is
treated as one of the pulses in the train.

This matches the NONMEM `SS=1` semantic: the compartments are initialised
to the value that would arise from an infinite-past pulse train at
interval `II`. Subsequent observations decay from that loaded state under
the model's normal dynamics; the SS train is **not** implicitly continued
past the SS dose record. To continue dosing forward in time, add explicit
dose records (the typical clinical pattern is one `SS=1` "loading" row
followed by any number of probe observations).

## Dataset columns

A steady-state row uses two NONMEM-format columns: `SS` and `II`.

```
ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,II,SS
1,0,.,1,100,1,0,1,24,1     # SS=1: at SS under q24h dosing of AMT=100
1,1,4.18,0,.,1,0,0,.,.
1,4,3.86,0,.,1,0,0,.,.
1,12,1.78,0,.,1,0,0,.,.
1,23,0.59,0,.,1,0,0,.,.
```

`II` is the dosing interval. For a bolus row (`RATE=0`) and an
infusion row (`RATE>0`), `SS=1` works the same way — the steady-state
state is computed from the corresponding single-dose response repeated
every `II`.

## Supported prediction paths

Every prediction path in ferx-core honours `SS=1`:

| Path                                              | Where                                                                     | How                                                |
| ------------------------------------------------- | ------------------------------------------------------------------------- | -------------------------------------------------- |
| Analytical (1-/2-/3-cpt, no TV covariates)        | `predict_concentration` in `src/pk/mod.rs`                                | Closed-form geometric-series                       |
| Analytical (1-/2-/3-cpt, time-varying covariates) | `event_driven_predictions_with_schedule` in `src/pk/event_driven.rs`      | Numerical pulse expansion via the propagator       |
| ODE (`[odes]`-block models)                       | `ode_predictions` / `ode_predictions_event_driven` in `src/ode/predictions.rs` | Numerical pulse expansion via the RK45 solver |

Both kinds of paths start from the same underlying identity — the
choice between them is a question of whether the geometric series has
a closed form for the model in question.

### Analytical path: closed-form geometric series

For linear analytical PK (1-/2-/3-cpt), the single-dose response is a
sum of exponentials with known eigenvalues. The steady-state series

```
C_ss(τ) = Σ_{n=0}^∞ C_single(τ + n·II)
```

collapses per eigenvalue: every exponential `A·exp(-λ·t)` in the
single-dose response contributes a steady-state amplitude
`A·exp(-λ·τ) / (1 - exp(-λ·II))`. ferx evaluates this exactly — no
iteration, single-pass cost equivalent to evaluating the single-dose
formula plus one extra division per eigenvalue. For oral models when
`KA ≈ λ` the closed form needs the L'Hopital limit; ferx handles that
case automatically.

### ODE path: brute-force pulse expansion

An `[odes]`-block model is an arbitrary user-defined RHS. There is no
general eigenstructure to exploit — `dy/dt = f(y, p, t)` can be
non-linear (Michaelis-Menten elimination), time-varying, or both, so
no closed form exists for the steady-state state in general.

Instead, ferx evaluates the geometric series *numerically* before the
SS dose is applied. The algorithm in `equilibrate_ss_state` is exactly
what its name says — brute-force pulse expansion:

1. **Reset** the compartment state to zero. NONMEM `SS=1` semantics
   say prior dynamics are discarded at the SS dose, so anything that
   happened earlier in the timeline is overwritten here.
2. **Loop `N = 50` cycles**:
   - **Apply the dose** for one cycle:
     - For a bolus, add `AMT` (with bioavailability `F1` applied) to
       the dose's compartment.
     - For an infusion, integrate for `T_inf = AMT/RATE` with a
       wrapped RHS that adds `+RATE` to the dose's compartment.
   - **Propagate** the ODE forward by the remainder of the cycle (`II`
     for boluses, `II - T_inf` for infusions). Same RK45 solver as the
     normal timeline, same tolerances, same wrapped-infusion mechanics.
3. After the loop, the compartment state equals the "just-before-the-
   next-pulse" steady-state amount.
4. Normal-timeline handling resumes — the SS dose's own pulse is
   applied through the standard bolus/infusion code path, taking the
   state from pre-pulse to at-pulse SS.

The result is mathematically equivalent to the analytical closed form
in the linear case (and is unit-tested against it — see
`ode_ss_iv_bolus_matches_analytical_ss` in `src/ode/predictions.rs`
for the 1-cpt cross-check). For non-linear PK, the same scheme still
produces a self-consistent steady-state state under the model's own
dynamics; just be aware that for systems that *don't* have a true
periodic steady state (e.g. zero-order elimination at saturation),
the iteration may not converge to anything meaningful and SS=1 isn't
really applicable.

### Why `N = 50`?

The truncation tail after `N` cycles is bounded by `exp(-N·λ·II)` for
each disposition rate `λ`. For typical PK
(`λ·II ≈ 2`, i.e. ~3 half-lives per dosing interval),
`exp(-100) ≈ 4e-44` — well below any meaningful precision. For very
slow PK (`λ·II = 0.1`, ~14 half-lives total over the 50 cycles),
`exp(-5) ≈ 7e-3` — the slowest realistic case. ferx uses fixed
`N = 50` rather than adaptive convergence checking because (a) the
bound is conservative and (b) skipping the convergence check keeps
the hot path branch-free for AD compatibility.

If you need tighter accuracy for an unusually slow PK, the constant
lives at `SS_EQUILIBRATION_CYCLES` in `src/ode/predictions.rs` and
`EVENT_DRIVEN_SS_EQUILIBRATION_CYCLES` in `src/pk/event_driven.rs`.

### Cost comparison

| Path                | Per SS-dose cost                          | Notes |
| ------------------- | ----------------------------------------- | ----- |
| Analytical closed   | ~1 single-dose evaluation                 | Effectively free |
| Event-driven pulse  | ~50× analytical propagator calls          | Still cheap; analytical propagator is fast |
| ODE pulse           | ~50× RK45 segment integrations            | Real cost — RK45 is the dominant per-prediction cost for ODE models, so SS doses cost ~50× more than non-SS doses |

SS doses are typically rare (one per subject at the start of a
maintenance regimen), so the absolute overhead per fit iteration is
modest even for ODE models. If a benchmark shows SS equilibration as
a hot spot, an adaptive `N` (loop until `||state - prev|| / ||state||
< tol`) is the obvious follow-up — it would early-exit on fast PK.

## DSL example

No new DSL syntax is required — the model file is identical to a
single-dose model. SS is a property of the **dataset**:

```
[parameters]
  theta TVCL(2.5, 0.01, 50.0)
  theta TVV(15.0, 0.5, 200.0)
  theta TVKA(1.0, 0.05, 20.0)

  omega ETA_CL ~ 0.05
  omega ETA_V  ~ 0.05
  omega ETA_KA ~ 0.1

  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method  = foce
  maxiter = 100
```

Combine with the SS=1 dataset above and run:

```
cargo run --release -- examples/ss_oral_q24.ferx --data data/ss_oral_q24.csv
```

A runnable example pair ships in the repository at
`examples/ss_oral_q24.ferx` + `data/ss_oral_q24.csv`.

## Combining with `LAGTIME`

`LAGTIME` (NONMEM `ALAG1`) shifts every pulse in the SS train, including
the SS pulse itself. Writing `τ = t − dose.time` for the time since the
dose *record* and `L` for the lagtime, the steady-state concentration is

```
C_ss(τ, L) = C_ss(τ − L,      0)    for τ ≥ L          (current interval)
C_ss(τ, L) = C_ss(τ − L + II, 0)    for 0 ≤ τ < L      (previous interval)
```

The second line matters: a sample taken **earlier than the lagtime**
(between the dose record time and the lagged arrival) is still at steady
state, so it shows the decaying tail of the *previous* dosing interval —
not zero. ferx-core implements both branches on all three prediction
paths (analytical superposition, the event-driven analytical walker, and
the ODE solver), and the result is validated against NONMEM `ALAG1` +
`SS=1` to 5 significant figures (see *Validation* below). Declaring
`LAGTIME` on a steady-state model is fully supported.

> Edge case for SS **infusions**: the previous-interval tail assumes the
> prior infusion has finished by the record time, i.e. `L ≤ II − T_inf`.
> This holds for any realistic lagtime; overlapping infusions
> (`T_inf > II`) are rejected as described under *Limitations*.

## Limitations

For the following malformed configurations, ferx-core skips the SS
pre-equilibration (treating the dose as a single, non-SS bolus or
infusion) and emits a warning in `FitResult.warnings`. The predictions
in these cases are *not* zero — the dose is still applied through the
normal flow — but the system isn't actually at steady state at the
dose time, so don't interpret the fit as a steady-state fit.

- **`SS=1` with `II ≤ 0`** — interval is required for SS predictions.
  The SS branch is gated on `dose.ii > 0`, so the dose falls through
  to the single-dose path. Set `II` in the dataset or remove the
  `SS=1` flag.
- **`SS=1` infusion with `T_inf > II`** (overlapping pulses) — no
  closed-form solution or pulse-expansion scheme covers this. The SS
  pre-equilibration returns a zero preload and the dose is then
  applied as a single infusion. Use a shorter infusion (`T_inf ≤ II`)
  or model overlapping infusions with an `[odes]` block and explicit
  periodic dose records (without `SS=1`).

### `SS=2`

NONMEM's `SS=2` semantic ("add to the existing train without
re-equilibrating") is **not implemented**. The CSV reader accepts any
positive integer in the `SS` column and treats it as a boolean: an
`SS=2` row is parsed identically to `SS=1` and goes through the
re-equilibration path described above. This is not bit-for-bit
NONMEM-compatible for datasets that distinguish `SS=1` and `SS=2`
records — convert `SS=2` rows to explicit dose records if the
no-re-equilibration semantic is required.

## NONMEM equivalence

- `SS=1`, `II=24`, `AMT=100`, `RATE=0` → matches NONMEM SS=1 bolus.
- `SS=1`, `II=24`, `AMT=100`, `RATE=25` → matches NONMEM SS=1 infusion
  of duration `4` (= `AMT/RATE`), repeated every 24 time units.
- The SS state is computed assuming the same `CL`, `V`, etc. as the
  current PK record — i.e. ferx uses the **current** subject's
  parameters, not a separate "SS parameters" set. This matches
  NONMEM's `$PK` evaluation convention.

## Validation

Closed-form SS expressions are unit-tested against 200- to 400-term
numerical pulse sums at 1e-9 relative tolerance (see
`src/pk/one_compartment.rs::tests::test_ss_*`, similarly for 2-cpt and
3-cpt). The ODE and event-driven paths are cross-checked against the
analytical closed forms in their own test modules. The end-to-end fit
path is covered by `tests/ss_fit_smoke.rs`.

SS + `LAGTIME` is additionally cross-checked against **NONMEM 7.5.1**
(`ADVAN1`/`ADVAN2`, `$ESTIMATION MAXEVAL=0`) to 1e-4 relative — including
the previous-interval tail for samples earlier than the lagtime. The
reference values and the NONMEM control files are documented in
`tests/ss_lagtime_nonmem.rs`, with per-path coverage in
`src/pk/mod.rs`, `src/pk/event_driven.rs`, and `src/ode/predictions.rs`
(`*_ss_*_with_lagtime_matches_nonmem`).
