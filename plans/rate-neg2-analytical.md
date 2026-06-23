# RATE=-2 (modeled infusion duration) for analytical PK models — #324

## Goal

Extend the existing `RATE=-2` → `D{cmt}` modeled-duration support (today ODE-only)
to the analytical PK engine (1-cpt / 2-cpt closed forms, superposition + event-driven).
Match NONMEM: `F·AMT` delivered as a zero-order infusion over duration `D{cmt}`,
rate = `AMT / D{cmt}`. `RATE=-1` (`R{n}`) stays out of scope (Phase B, not done for ODE either).

## Current state (what already exists)

- `RateMode::{Fixed, ModeledDuration}` + `DoseEvent::modeled()` + `DoseEvent::resolve_rate()`
  (`src/types.rs`) — engine-agnostic, f64-only. Single source of truth for the resolve rule.
- `DoseAttrMap` + `DoseAttr::from_indexed_name()` recognises `D{n}` → `(Duration, n)`.
- Reader emits `ModeledDuration` doses for `RATE=-2` (`io/datareader.rs`).
- ODE path resolves via `resolve_subject_doses[_with]` before integrating; `dose_attr_map`
  lives on `OdeSpec`, populated in `build_ode_spec` from indexed indiv-param names.
- Analytical path REJECTS modeled doses:
  - `api.rs::check_modeled_dose_rates` → `E_MODELED_DURATION_ANALYTICAL` when `ode_spec` is None.
  - `pk::compute_predictions` / `event_driven_predictions` `assert!(all_doses_fixed())`.

## Gap

1. Analytical models build no `DoseAttrMap` and route only canonical PK names
   (cl/v/q/v2/ka/f/q3/v3/lagtime) into `PkParams`. A `D{cmt}` individual parameter
   is computed but never stored in a `PkParams` slot, and nothing records its slot.
2. Nowhere analytical-reachable holds a `DoseAttrMap` (`compute_predictions` takes only
   `pk_model, subject, pk_params`).
3. The two validation gates reject analytical outright.

## Design

### 1. Parser — route `D{cmt}` to a slot + build the map (analytical branch)

In `build_compiled_model` (model_parser.rs ~1132), for analytical models (`!is_ode`):
- Scan `indiv_var_names` for `DoseAttr::from_indexed_name` hits of `DoseAttr::Duration`.
- Allocate each a free `PkParams` slot from the spare region (slots 9..MAX_PK_PARAMS,
  which analytical models never use — canonical names occupy a subset of 0..8).
  Reject `cmt` beyond the analytical model's compartment count (1-cpt → cmt∈{1},
  2-cpt → cmt∈{1,2}) with a clear error, mirroring the ODE `n_states` check.
- Build a `DoseAttrMap` with `(Duration, cmt) -> slot`.
- Pass the `(slot, var_name)` list into `build_pk_param_fn` so the analytical
  closure writes `p.values[slot] = vars[var_slot]` alongside `pk_assignment_mapping`.

`build_pk_param_fn`: add an `extra_analytical_slots: &[(String /*var*/, usize /*slot*/)]`
param (empty for ODE), resolved to `(slot, var_slot)` like `pk_assignment_mapping` and
written in the `is_analytical_pk` arm.

### 2. `CompiledModel` — carry the map for analytical models

Add `pub dose_attr_map: DoseAttrMap` to `CompiledModel`. For ODE models it can mirror
`ode_spec.dose_attr_map` (or stay default and keep reading from `ode_spec` — ODE sites
are unchanged either way). For analytical models it holds the map built in step 1.
Update every `CompiledModel` constructor / test fixture / `generate_data.rs` to set it
(`Default::default()` where none).

### 3. Resolve at the analytical dispatch boundary

Mirror the ODE design: resolve modeled doses → `Fixed` ONCE before the closed-form
math, so `compute_predictions` / `predict_concentration` / `event_driven_*` keep their
`all_doses_fixed()` asserts (now satisfied) unchanged.

Add a small helper (reuse `resolve_rate`) and call it at the two analytical entrypoints
that hold a `&CompiledModel`:
- `api.rs::model_preds` (analytical arm, line ~62).
- `pk/mod.rs` estimation dispatcher (the no-TV `compute_predictions` arm ~1224 **and**
  the TV/event-driven arm ~1200) — resolve with per-dose `PkParams` on the TV path
  (like `resolve_subject_doses_with`'s per-dose closure), single `pk` on the no-TV path.

Produce a `Cow<Subject>`; the borrowed (all-Fixed) path stays allocation-free.

### 4. Validation gates

`api.rs::check_modeled_dose_rates`: replace the `None =>` analytical-rejection arm with
a check against `model.dose_attr_map` — error `E_MODELED_DURATION_NO_PARAM` (same code as
ODE) when no `D{cmt}` slot exists; otherwise accept. Keep the ODE arm. Drop / rewrite the
`E_MODELED_DURATION_ANALYTICAL` message.

Keep the defensive `assert!(all_doses_fixed())` in `compute_predictions` /
`event_driven_predictions` as tripwires (they now always hold because we resolve upstream).

## Tests

- **Unit (parser, Tier 1):** analytical model with `D1` routes the value into a spare
  slot and builds `dose_attr_map` with `(Duration,1)->slot`; `cmt` past compartment count errors.
- **Unit (types, Tier 1):** `resolve_rate` for analytical PkParams gives `rate = amt/D`,
  `duration = D`, `Fixed`.
- **Tier 2 / integration:** `RATE=-2` analytical 1-cpt IV and 2-cpt produce predictions
  numerically equal to the same model dosed with explicit `RATE = AMT/D` (equivalence test).
- Keep `test_compute_predictions_panics_on_modeled_dose` (direct-call tripwire).
- Coverage: the new parser slot-routing + gate branches need diff coverage ≥90%.

## NONMEM comparison (required by CLAUDE.md)

Add a `RATE=-2` analytical example (1-cpt IV, modeled `D1`) and compare IPRED/OFV against
a NONMEM `$PK D1=...` `ADVAN1`/`ADVAN3` run; record in `docs/faq.qmd` or the relevant
estimation/error doc + the PR description.

## Docs + changelog

- `docs/` (ode-models.qmd modeled-RATE note + wherever analytical dosing/RATE is
  documented; likely a data-format / dosing page): note `RATE=-2` now works for analytical
  models given a `D{cmt}` parameter; keep the `D{n}` reserved-name collision caveat.
- `CHANGELOG.md` `## [Unreleased]` → `Added`: "`RATE=-2` modeled infusion duration now
  supported for analytical PK models (#324)."
- ferx-r follow-up only if a `pub` API signature changed (likely not — internal).

## Risk / watch-outs

- Slot collision: `D{cmt}` must not land on a canonical PK slot or `PK_IDX_F`/`PK_IDX_LAGTIME`.
  Spare region 9.. is safe for analytical.
- `predict()/simulate()` bypass `check_model_data`; `assert_modeled_doses_supported`
  must accept the now-valid analytical case (it reuses `check_modeled_dose_rates`, so the
  gate fix covers it).
- SS infusions (`SS=1` + `RATE=-2`): duration resolves first, then existing analytical SS
  infusion handling applies — verify the equivalence test covers an SS row.
