# Non-IOV ODE TV-cov × {EVID=2 pk-only, `ExpressionScale`} analytic sensitivities — #486

## Goal

On the **ODE path**, models with time-varying covariates (TV-cov) are analytic via the
`(θ,η)`-basis TV-cov walk. The **IOV** walk additionally made two TV-cov combinations
analytic in #590: **EVID=2 covariate-only breakpoints** (κ=0) and a per-occasion
**`ExpressionScale` `obs_scale`** scale-jet. The matching **non-IOV** TV-cov walk still
routes both to finite differences. This closes those two cells by porting #590's
machinery down to the non-IOV walk (there is exactly one "group" — no κ — so the
per-occasion logic collapses to a single subject-static jet).

Two independent sub-features, ship-able separately:
- **(a)** TV-cov + EVID=2 pk-only breakpoints → analytic.
- **(b)** TV-cov + `ExpressionScale` `obs_scale` → analytic.

## Current state (what already exists)

All anchors in `src/sens/ode_provider.rs` unless noted.

- **Outer Dual2 walk** `run_subject_tvcov<const M>` (`ode_provider.rs:1908`, `M = n_theta + n_eta`),
  dispatched from `ode_subject_sensitivities` (`712`, `dispatch_tv!` at `720-729`).
- **Inner Dual1 walk** `run_subject_tvcov_eta<const N>` (`2678`, `N = n_eta`), dispatched from
  `ode_subject_eta_grad` (`878`, macro `886-897`). Both share the gate `ode_tvcov_supported`,
  so opening the gate enables both — they MUST be extended in lockstep (see Risks).
- **Snapshot seeding** `seed_tvcov_snapshots<T>` (`1867`) already builds `pk_at_pk_only`
  (`1898-1900`) at the η-snapshot — the non-IOV pk-only seeding is essentially already present.
- **Event walk** `integrate_tvcov_g<T>` (`2992`) already pushes pk-only breakpoints
  (`K_PKONLY`, `3149-3151`) and consumes them in the segment loop (`3197-3205`, esp. `3202`).
  So the walk already carries EVID=2 breakpoints; only the gate declines them.
- **#534 static-walk obs_scale** is the non-IOV quotient template: applied in
  `ode_subject_sensitivities:820-848` via `apply_expression_scale_outer` (`provider.rs:2317`)
  → `apply_expression_scale<const M>` (`provider.rs:2201`) → the single-source row quotient
  `apply_scale_quotient_row<const M>` (`provider.rs:2252`). That row helper is already
  parameterized on `n_axes` (comment `provider.rs:2243-2251`: `n_eta` non-IOV vs `n_stacked`
  IOV) — it is the exact function the IOV per-group caller reuses, so it is reusable as-is.
- **#590 IOV analogues to mirror:**
  - pk-only κ=0 seeding: `seed_pk_only_cov` closures (`2246-2262` outer, `2522-2538` inner),
    routed by `seed_iov_events` (`2382`, pk-only at `2404-2406` / `2429-2434`); κ-drop via
    `seed_pk_dual2_iov(..., group=None, ...)` (`2100`). κ=0 source `iov_combined_pk_only`
    (`stats/likelihood.rs:2127-2132`). The per-subject IOV gate `ode_iov_subject_supported`
    (`1975`) carries NO pk-only decline (contrast non-IOV `572`).
  - per-occasion scale-jet: `group_scale` built in `run_subject_iov` (`2295-2315`) via
    `build_iov_scale_jets<T>` (`2173`), applied per-row by `apply_scale_quotient_row`
    (`2357-2371`); inner analogue `2567-2587` / `apply_scale_quotient_grad_iov` (`provider.rs:2157`).

## Gap

Two decline clauses in `ode_tvcov_supported` (declared `463`):

1. **(a) EVID=2 pk-only** — `572-574`:
   ```rust
   if !subject.pk_only_times.is_empty() { return false; }
   ```
   (rationale `568-571`: "not yet carried on the non-IOV TV-cov ODE path").
2. **(b) `ExpressionScale`** — `496-498`:
   ```rust
   if matches!(model.scaling, ScalingSpec::ExpressionScale { .. }) { return false; }
   ```
   (rationale `487-495`: the event-driven walk carries no post-walk scale quotient; IOV got
   it in #590, "the non-IOV walk just lacks that machinery").

## Design

### (a) TV-cov + EVID=2 pk-only — likely a gate deletion + validation

The walk machinery already exists (seeding `1898-1900`, consumption `3149-3151`/`3202`).
No κ exists on the non-IOV path, so there is **no `iov_combined_pk_only`/κ=0 analogue to
build** — pk-only events seed identically to obs/dose at the same η.

1. Remove the `572-574` decline clause in `ode_tvcov_supported`.
2. Verify the readout/`last_params` path (`3190-3204`) handles a segment that contains a
   pk-only-only breakpoint, and that pk-only events carrying their own covariate snapshot
   dedup correctly in `seed_tvcov_snapshots` (`1867`).
3. Update the gate-scope test `ode_tvcov_gate_scope` (`6740`, the `pk_only_times` assertion
   at `6757-6760`).

### (b) TV-cov + `ExpressionScale` — one subject-static jet onto the TV-cov walk

Because the scale divisor is **subject-static even under TV-cov** (`ode_provider.rs:490-494`,
mirrored by the IOV gate `1985-1989`), the IOV per-occasion logic collapses to a single jet:

1. Remove the `496-498` decline clause; instead admit `ExpressionScale { deriv: Some(p), .. }`
   when `expression_scale_axes_admissible(p, model)` (the gate already used by IOV at
   `672-673`, helper at `227`). Keep the `None`/LTBS arm declining (LTBS stays FD).
2. In `run_subject_tvcov` (outer), after the TV-cov walk produces `SubjectSens`: compute a
   **separate subject-static** `pk = (model.pk_param_fn)(...)` and `pd = param_derivatives(...)`
   at `subject.covariates` (the walk seeds per-event and has no whole-subject `pd`/`pk` on
   hand — this is the one new computation), then apply `apply_expression_scale_outer`
   exactly as the static walk does at `820-848` (`slots = indiv_param_program.pk_slots_ref()`).
   The per-row quotient is the already-reusable `apply_scale_quotient_row<const M>`
   (`provider.rs:2252`).
3. In `run_subject_tvcov_eta` (inner), add the matching inner quotient via
   `apply_expression_scale_inner_dispatch` (the static inner path `918-942`) / an inner row
   helper analogous to `apply_scale_quotient_grad_iov`. Inner and outer MUST move together.

## Tests

Mirror the five IOV combo tests at `provider.rs:6693-6805`; per the ≥90%-diff coverage gate
each combo needs outer + inner + a still-FD edge test.

- **(a)** outer: copy `ode_iov_tvcov_pkonly_breakpoint_matches_fd_of_predict_iov`
  (`provider.rs:6732`) into a non-IOV `ode_provider_tvcov_pkonly_matches_production` driving
  `check_vs_production` (`ode_provider.rs:4283`) + the 2nd-order FD-of-gradient pattern from
  `ode_provider_tvcov_matches_production` (`6446`, esp. `6497-6526`); inner mirrors
  `ode_iov_tvcov_pkonly_inner_eta_grad_matches_outer` (`6763`).
- **(b)** mirror `ode_iov_tvcov_expr_scale_provider_matches_fd_of_predict_iov`
  (`provider.rs:6693`) + its inner (`6721`). **Flip** the current must-FD assertion in
  `ode_provider_expression_scale_combos_fall_back_to_fd` (`ode_provider.rs:4724`, the TV-cov+
  ExpressionScale `is_none()` check at `4751-4755`) to now-analytic.
- Subject builders: `bolus_subject` (`3954`), `bolus_subject_wt` (`4009`), `tvcov_subject` (`6530`).

## NONMEM comparison (required by CLAUDE.md)

Fit a TV-cov ODE model with (a) an EVID=2 covariate-only row and (b) an `obs_scale = expr(θ,η)`
readout in ferx vs NONMEM `METHOD=1 INTER`; record OFV + parameter agreement (≤1e-5 OFV) in
the PR description. Reuse the #590 IOV anchor datasets with the κ column dropped.

## Docs + changelog

- Matrix issue **#486**: flip the **Time-varying covariates** row's ODE cells note (the
  "TV + EVID 2 on a *non-IOV* ODE model … → FD" clause) and the **`ExpressionScale` `obs_scale`**
  row ("*non-IOV* ODE TV-cov" residual-FD combo); drop both from the Audit ③ bucket (`168-176`).
- `docs/estimation/foce.qmd` / `docs/model-file/scaling.qmd`: flip the relevant FD-fallback note.
- `CHANGELOG.md` `[Unreleased]` → `Added`.
- ferx-r: no follow-up unless a `pub` signature changes (this is internal `sens/` work).

## Risk / watch-outs

- **Inner/outer parity is the guardrail.** Both walks share `ode_tvcov_supported` (`720`,
  `886`), so opening the gate auto-enables both; if the inner quotient/pk-only path isn't
  added in lockstep the inner EBE gradient silently diverges from the outer. The #575/#590
  inner-vs-outer parity tests are the safety net — add the analogous ones here.
- **Scale is subject-static** (`490-494`): build the `ExpressionScale` jet at
  `subject.covariates`, NOT per-observation-snapshot.
- **`pd`/`pk` availability**: the TV-cov walk doesn't carry a whole-subject `pd`/`pk`; compute
  a separate subject-static `pd` (`param_derivatives` at `subject.covariates`) for the jet.
- **Axis cap**: honor `prog.n_axes() == n_theta + n_eta ≤ MAX_SCALE_AXES`
  (`expression_scale_axes_admissible`, `227`) on the non-IOV path too.
- **LTBS stays FD**: the non-IOV `ScalingSpec::None if !model.log_transform` semantics — the
  `ExpressionScale` decline at `496` currently fires before any LTBS reasoning; keep LTBS FD.
- **Leave out of scope**: `init_fn` (`522`) and `input_rate` (`517`) declines are separate gaps.
