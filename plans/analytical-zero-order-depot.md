# Analytical closed-form zero-order input into the oral depot (`D{depot}` on `pk *_oral`)

Tracking issue: **#400**

## Problem

A `RATE=-2` modeled-duration dose (`D{cmt}` → zero-order infusion of duration `D`)
into compartment **1** of an analytical oral model (`one_cpt_oral`,
`two_cpt_oral`, `three_cpt_oral`) is rejected at parse time:

```
[individual_parameters]: `D1` is a modeled infusion duration (RATE=-2) for
compartment 1, but the analytical `one_cpt_oral` model can only infuse into
compartment(s) [2] (`D2`). A zero-order input into another compartment
(e.g. an oral depot) needs an `ode(...)` model.
```

This is a real, common absorption model: **zero-order release into the depot,
then first-order `KA` absorption into central** (sustained-release / SC depot).
The dynamics are linear with piecewise-constant forcing, so they have a closed
form — there is no need to drop to an ODE. The rejection in
`PkModel::infusable_compartments()` simply hasn't been lifted because the oral
propagators implement only depot **bolus** + depot-**bypassing** central
infusion.

Relationship to existing issues:
- **#324** — coded-`RATE` plumbing (`RATE=-2` → `D{cmt}`, per-iteration resolve).
  Already shipped for infusion into **central**. This work *consumes* that
  `D1` forcing; it does not re-implement it.
- **#322** — absorption-models roadmap, deliberately **ODE-based** (input-rate
  functions forced into an explicit ODE; analytical `pk` is never silently
  rewritten to an ODE). Phase 2 lists a `zero_order` family but in the ODE
  framing. This work is the **analytical closed-form complement**: keep
  `pk *_oral` analytical, just allow `D{depot}`.

## Math

Zero-order input of rate `R` into the depot over a sub-interval of length `dt`.
By linearity, split the depot into its existing homogeneous evolution (already
handled by the propagators) plus a new **forced** part driven by `R`.

### 1-cpt oral — state `[A_depot, A_central]`, `ke = CL/V`

During the infusion sub-interval, the **new** forced contributions to add on top
of the existing homogeneous terms (`e_ka = exp(-ka·dt)`, `e_ke = exp(-ke·dt)`,
both already computed in `propagate_one_cpt_oral`):

```
state[0] += (R/ka) * (1 - e_ka)                                  # depot forced fill
state[1] += R/ke * (1 - e_ke)  -  R * (e_ka - e_ke) / (ke - ka)  # central forced
```

`ka ≈ ke` singularity (L'Hôpital): the second central term →  `R · dt · e_ke`.

Derivation: depot `A_d' = R − ka·A_d`; the forced depot part is
`(R/ka)(1 − e^{−ka t})`, so its drive into central is `ka·A_d_forced = R(1 − e^{−ka t})`.
Convolving with the central impulse response `e^{−ke t}` gives the central term.

### 2-cpt / 3-cpt oral

Same split. Forced response under constant depot input `R` = steady-state minus a
transient assembled from the system eigenmodes (`−ka`, `−α`, `−β`[, `−γ`]). The
steady state is analytic:
- `A_depot_ss   = R/ka`
- `A_central_ss = R/k10`  (mass conservation: all input clears through central)
- peripheral SS from the inter-compartmental balances (`k12·A_c = k21·A_p`, …).

The transient amplitudes are fixed by `A_forced(0) = 0`. Implement directly in
each propagator reusing the eigenvalues already computed there. **Validate
numerically against the RK45 ODE solver** (which integrates the depot-infusion
RHS correctly) rather than hand-checking the algebra — see Tests.

## Affected code & plan (one PR)

### 1. Lift the parser gate
- `src/types.rs` `PkModel::infusable_compartments()`: add `1` to `OneCptOral`,
  `TwoCptOral`, `ThreeCptOral` (depot is cmt 1). The parser routing in
  `model_parser.rs` (the `analytical_dur_slots` loop) then admits `D1` into a
  spare `PkParams` slot automatically — no change there. Confirm the rejection
  test that pins the old message is updated.

### 2. Event-driven propagator (the one math implementation)
- `src/pk/event_driven.rs`:
  - `propagate_with_bounds`: add a `rate_depot` channel alongside `rate_central`.
    New routing arms: `(OneCptOral, 1) | (TwoCptOral, 1) | (ThreeCptOral, 1)
    => rate_depot += r`. Keep the existing `(…Oral, 2) => rate_central` arms.
  - `propagate_one_cpt_oral` / `_two_cpt_oral` / `_three_cpt_oral`: add a
    `rate_depot: f64` parameter and the forced-response terms above.
  - The SS / single-dose infusion-pulse builders (`single_dose_*`,
    bound construction) must treat a cmt-1 oral infusion as a valid infusion
    window (they already split bounds on `rate>0 && duration>0`; verify the
    cmt dispatch doesn't panic for cmt 1 oral).
  - Update the module-doc "Infusion support … into the depot for oral models"
    note — it already *claims* depot infusion for oral; make it true.

### 3. Route depot-infusion subjects to the event-driven path
The no-TV superposition path (`compute_predictions` → `predict_concentration`,
and `one_compartment.rs` closed forms) handles oral **central** infusion by the
"infusion bypasses depot" IV formula but has **no** depot-infusion form. Rather
than duplicate the math, route any subject that has an oral **depot** infusion
(resolved `rate>0 && duration>0` into cmt 1 on an oral model) through
`event_driven::event_driven_predictions`, mirroring the existing `has_resets()`
guard in `compute_predictions` (mod.rs ~1041).
- Add a small predicate (e.g. `subject_has_oral_depot_infusion(pk_model, subject)`)
  used by `compute_predictions`.
- `compute_predictions_with_tv*` already prefer event-driven when supported, so
  the TV path is covered; just confirm the routing predicate.

### 4. Compartment states (`[derived]` / sdtab)
`compute_predictions_with_states` analytical branch builds states via
`predict_all_states` (superposition), which won't have a depot-infusion form.
Options, decide during impl:
- (preferred) add an event-driven states variant for depot-infusion subjects, or
- leave states empty → NaN with a `W_DERIVED_*` warning (consistent with the
  reset/TV analytical-states convention) and a follow-up issue.
Predictions (ipred) must be correct regardless; states can degrade gracefully.

### 5. AD path — no change required
`ad/ad_gradients.rs` routes any oral model with a `rate>0` dose to **FD**
gradients (`is_oral_model && rate>0 -> Fd`), so depot-infusion oral subjects get
correct FD gradients without touching the bolus-only AD oral propagators. Note
this explicitly in the PR; the autodiff oral-infusion arms stay a separate
follow-up (same status as oral central infusion under AD).

## Tests (all tiers per CLAUDE.md)

- **Tier 1 (unit, `src/pk/event_driven.rs`)**: `propagate_*_oral` with `rate_depot>0`
  vs a fine-grained RK45 integration of the same depot-infusion ODE — assert
  central (and periph) match to tolerance, including the `ka≈ke` branch.
- **Tier 1**: mass-balance — `∫` zero-order input over `[0,D]` delivers `F·AMT`
  into the depot (AUC / final-amount check vs `D2` central-infusion symmetry).
- **Tier 2 (`tests/`)**: `fit()`/`predict()` on a `one_cpt_oral` model with a
  `D1` (cmt-1) `RATE=-2` dose returns `Ok` and finite predictions (no panic,
  no parse error) — the regression for the lifted gate.
- **Tier 1**: parser no longer rejects `D1` on `one_cpt_oral`; the old
  rejection test now targets a still-invalid compartment (e.g. an oral
  peripheral) so the error path stays covered.
- Coverage: the diff carries its own tests (Codecov patch ≥90%).

## NONMEM validation (per CLAUDE.md)

Anchor a `one_cpt_oral` + `D1`-into-depot fit (or `predict`) against an
equivalent NONMEM `ADVAN2`/`$PK D1` zero-order-into-depot run. Record the OFV /
predictions comparison in `docs/src/faq.md` (or the relevant estimation page).

## Docs & changelog

- `docs/src/model-file/individual-parameters.md` and/or
  `docs/src/data-format.md`: document that `D{depot}` (cmt 1) is now a valid
  zero-order input on analytical oral models, with the absorption interpretation.
- `CHANGELOG.md` `## [Unreleased]` → `Added`: one line referencing the new issue #.
- If any new `pub` surface lands, follow up with the `ferx-r` lock bump
  (`tools/update-ferx-core-lock.sh`).

## Out of scope (follow-ups)
- Autodiff oral-infusion propagator arms (depot or central) — tracked separately.
- Oral **peripheral** infusion (cmt ≥ 3) — still rejected.
- `RATE=-1` modeled-rate (`R1`) into the depot — separate (#324 `R1` work).
