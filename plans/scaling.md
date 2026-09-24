# ferx-core Scaling Plan — Phase 1 (`[scaling]` block)

Related issue: https://github.com/FeRx-NLME/ferx-core/issues/2

This plan covers Phase 1 only: an explicit `[scaling]` block in `.ferx` model files.
Phase 2 (automatic unit-based scaling via `dose_units` / `obs_units`) is out of scope and gets its own plan.

Read `AGENTS.md` first before starting any step.
Complete steps in order — later steps depend on earlier ones.
Each step specifies the files to touch and the expected outcome — but **the plan
below is a starting hypothesis, not a fixed recipe**. The actual repository state
may differ from what is described. Every step begins with deep evaluation of the
existing code, plan reconciliation, and explicit confirmation before any code is
written.

---

## Background

ferx-core currently assumes the structural model's raw output is in the same units as
the observed `DV`. For analytical PK this is usually fine. For ODE models, raw states
are amounts, so users must fold unit conversion into the ODE itself — this overloads
the ODE block and prevents writing ODEs in the natural amount-based form.

The `[scaling]` block adds a named section that declares how the structural model's
output maps to `DV`, without touching the structural model itself. Three forms are
supported:

**Form A — scalar multiplier (analytical or ODE):**
```
[scaling]
  obs_scale = 1000
```

**Form B — expression multiplier (references individual parameters / covariates):**
```
[scaling]
  obs_scale = 1000 / V
```

**Form C — explicit output expression (ODE only; replaces the `obs_cmt` readout):**
```
[structural_model]
  ode(states=[depot, central])

[scaling]
  y = central * 1000 / V
```

**Convention:** Forms A/B are divisive: `pred_scaled = pred_raw / obs_scale`.
Form C replaces the state readout entirely.

---

## Step 1 — Data model (`src/types.rs`)

**Add `ScalingSpec`:**

```rust
pub enum ScalingSpec {
    None,
    ScalarScale(f64),
    ExpressionScale {
        scale_fn: Box<dyn Fn(&[f64], &[f64], &HashMap<String, f64>) -> f64
                     + Send + Sync>,
    },
}
```

Add `pub scaling: ScalingSpec` to `CompiledModel` (default `None`).

**Extend `OdeSpec` (`src/ode/predictions.rs`):**

- Change `obs_cmt_idx: usize` → `obs_cmt_idx: Option<usize>`.
- Add `output_fn: Option<Box<dyn Fn(&[f64], &[f64], &HashMap<String, f64>) -> f64 + Send + Sync>>`.

**Why first:** every downstream step depends on these types. No behaviour change yet; all
existing construction sites compile after changing `obs_cmt_idx` to `Some(n)`.

**Multi-analyte note:** `ScalingSpec` is intentionally per-model today. When multiple
observed analytes land (CMT-keyed scaling), the enum gains a
`PerCmtScale(HashMap<usize, ScalingSpec>)` variant. Application sites (Steps 3/4)
already receive the per-observation CMT index via `subject.obs_cmts` — the dispatch
will live there. `output_fn` will extend to `output_fns: HashMap<usize, Box<...>>`,
keyed by CMT. No structural rework needed.

**Expected outcome:** `cargo check --tests` passes with no behaviour change.

---

## Step 2 — Migrate existing `OdeSpec` construction sites

Mechanical: all `OdeSpec { obs_cmt_idx: <n>, ... }` in:
- `src/ode/predictions.rs` (test helpers)
- `src/types.rs` (test factories)
- `tests/*.rs`

become `obs_cmt_idx: Some(n), output_fn: None`.

The three readout sites in `src/ode/predictions.rs` (currently lines ~135, ~179, ~342)
become:

```rust
let y = if let Some(ref f) = ode.output_fn {
    f(&u, pk_params_flat, &subject.covariates)
} else {
    u[ode.obs_cmt_idx.expect("either obs_cmt or output_fn must be set")]
};
predictions[obs_idx] = y;
```

**Expected outcome:** `cargo check --tests` passes; zero behaviour change.

---

## Step 3 — Apply scaling in the prediction dispatcher

**Files:** `src/pk/mod.rs`, `src/api.rs`

Add a free function `apply_scaling` in `src/pk/mod.rs`:

```rust
fn apply_scaling(
    scaling: &ScalingSpec,
    theta: &[f64],
    eta: &[f64],
    covariates: &HashMap<String, f64>,
    preds: &mut Vec<f64>,
) {
    match scaling {
        ScalingSpec::None => {}
        ScalingSpec::ScalarScale(k) => {
            let inv = 1.0 / k;
            preds.iter_mut().for_each(|p| *p *= inv);
        }
        ScalingSpec::ExpressionScale { scale_fn } => {
            let s = scale_fn(theta, eta, covariates);
            let inv = 1.0 / s;
            preds.iter_mut().for_each(|p| *p *= inv);
        }
    }
}
```

Call it at the end of `compute_predictions_with_tv_into_with_schedule` (covers all
analytical and ODE callers: FOCE, FOCEI, GN, trust-region, SAEM).

Mirror in `src/api.rs::model_preds` so `simulate()` and post-fit IPRED are scaled too.

**Likelihood / gradient correctness:**
- `stats/likelihood.rs` receives already-scaled predictions — no changes.
- Sigma estimates are in observation units (scaled space) — correct by construction.
- IIV/IOV: both flow through `pk_param_fn → compute_predictions_with_tv_into_with_schedule`; scaling is applied uniformly after prediction assembly regardless of ETA vector count.
- All outer optimizers (SLSQP, L-BFGS, MMA, BFGS, GN, trust-region) differentiate through `nll(theta)`, which calls this dispatcher — scaling is included automatically.

**Expected outcome:** `ScalingSpec::None` (default) leaves all existing tests passing.

---

## Step 4 — AD interaction (`src/ad/ad_gradients.rs`)

The AD path is a hand-duplicate of the analytical prediction logic and must stay consistent with Step 3.

- **`ScalarScale(k)`:** add `k: f64` as a `Const` input to `individual_nll_ad` and
  `predict_all_ad`. Divide `conc` by `k` just before the residual call (lines ~114
  and ~244 currently). Thread `k` through the `#[autodiff_*]` macro signatures.
- **`ExpressionScale`:** **hard error at parse time** (enforced in Step 5) when
  `gradient = ad` (explicit or auto-resolved) is combined with `ExpressionScale`.
  No AD code changes needed for this case.
- **Form C (ODE `output_fn`):** ODE never uses the analytical AD path. No changes.

**Expected outcome:** scalar scaling works with AD; expression scaling falls back to FD via a parse-time error.

---

## Step 5 — Parser (`src/parser/model_parser.rs`)

Most complex step. Do after Steps 1–4 so the target types exist.

**5a — Relax `parse_ode_structural`:** make `obs_cmt=NAME` optional.

```rust
let with_obs = Regex::new(r"ode\(\s*obs_cmt\s*=\s*(\w+)\s*,\s*states\s*=\s*\[([^\]]+)\]\s*\)").unwrap();
let without_obs = Regex::new(r"ode\(\s*states\s*=\s*\[([^\]]+)\]\s*\)").unwrap();
```

Return `Option<String>` for `obs_cmt`. Thread through to `build_ode_spec`.

**5b — `parse_scaling_block`:** new function modelled on `parse_fit_options`.
- Recognised keys: `obs_scale` (Forms A/B), `y` (Form C). Reject others.
- `obs_scale`: try `value.parse::<f64>()` → `ScalarScale`; otherwise run through
  the existing expression parser (`parse_expression` + `ParseCtx` seeded with
  `indiv_var_names`) → `ExpressionScale`.
- `y`: same expression parser with `ParseCtx` that additionally knows `state_names`
  as variables. Produces the `output_fn` closure.
- Returns `(ScalingSpec, Option<output_fn>)`.

**5c — Validation in `parse_full_model`:**
- `obs_cmt.is_some() XOR output_fn.is_some()` — error if neither or both for ODE models.
- `y = ...` on an analytical model → error.
- `ExpressionScale` + resolved `gradient_method == Ad` → error with message:
  _"Expression scaling is not yet supported with AD gradients — add `gradient = fd` to [fit_options]."_

**5d — Wiring:** call `parse_scaling_block` after `indiv_var_names` and `state_names` are
known (i.e., after `parse_individual_parameters` and `parse_ode_structural` have run).
Assign `model.scaling` and, for Form C, `ode_spec.output_fn`.

**Expected outcome:** all three forms parse correctly; invalid combinations error with clear messages.

---

## Step 6 — Tests

**Tier 1 — parser unit tests (`src/parser/model_parser.rs` `#[cfg(test)]`):**
- `test_parse_scaling_none` — no `[scaling]` → `ScalingSpec::None`.
- `test_parse_scaling_scalar` — `obs_scale = 1000` → `ScalarScale(1000.0)`.
- `test_parse_scaling_expression` — `obs_scale = 1000 / V` → `ExpressionScale`, evaluates correctly.
- `test_parse_scaling_y_ode` — ODE without `obs_cmt` + `y = central / V` parses successfully.
- `test_parse_scaling_both_errors` — `obs_cmt` + `y = ...` → error.
- `test_parse_scaling_neither_errors` — ODE without `obs_cmt` and no `[scaling]` → error.
- `test_parse_scaling_y_on_analytical_errors` — `y = ...` on analytical model → error.
- `test_parse_scaling_unknown_key_errors` — `[scaling] foo = 1` → error.
- `test_ad_errors_on_expression_scale` — `obs_scale = V` + `gradient = ad` → parse error.

**Tier 1 — prediction unit tests (`src/pk/mod.rs`, `src/ode/predictions.rs`):**
- `test_scalar_scale_divides_predictions` — predictions with `ScalarScale(1000)` are exactly `1/1000` of unscaled.
- `test_expression_scale_uses_indiv_param` — `obs_scale = V`, predictions equal `raw / V`.
- `test_ode_output_fn_replaces_obs_cmt` — ODE with `output_fn = states[1] / pk[1]` matches hand-computed value.

**Tier 2 — integration (`tests/*.rs`):**
- `test_ode_amount_vs_concentration_formulation` — same model written two ways (ODE
  with concentration baked in vs. ODE with amounts + `y = central / V`) gives identical
  predictions; returns `Ok` after a handful of outer iterations.

---

## Step 7 — Documentation & examples

- `docs/src/model-file/scaling.md` (new page): Forms A/B/C with worked examples,
  divisive convention explained, AD limitation noted.
- Add entry to `docs/src/SUMMARY.md`.
- Cross-reference from `docs/src/model-file/structural-model.md` (ODE section).
- `docs/src/faq.md`: entry comparing to NONMEM `S1`/`S2` and nlmixr2's `cmt(central)` convention.
- New example files:
  - `examples/scaling_scalar.ferx` (Form A on warfarin)
  - `examples/scaling_expression.ferx` (Form B on 1-cpt)
  - `examples/scaling_ode_amounts.ferx` (Form C: amount-based ODE)
- Render locally with `quarto render docs` to preview; commit only the `.qmd`
  sources (the rendered `docs/_site/` is git-ignored).

---

## Dependency order

```
Step 1 (types)
  └── Step 2 (ODE site migration — mechanical, no behaviour change)
        ├── Step 3 (dispatcher — first live behaviour)
        │     └── Step 4 (AD scalar support + ExpressionScale guard)
        └── Step 5 (parser — ties everything together)
              └── Step 6 (tests)
                    └── Step 7 (docs + examples)
```

Steps 3 and 4 can be done in parallel once Step 2 is complete.
Steps 6 and 7 can be split into sub-PRs if desired (unit tests land with Step 5; docs follow).

---

## Phase 2 outline (separate plan)

`[scaling]` gains `dose_units` and `obs_units` keys:

```
[scaling]
  dose_units = mg
  obs_units  = ng/mL
```

The parser derives an automatic `ScalarScale` or `ExpressionScale` from the unit
combination (e.g. `mg → ng/mL` on a model returning mg/L → `ScalarScale(1000)`).
Forms A/B/C still work as manual overrides.

---

## Multi-analyte forward-compatibility

The issue mentions CMT-keyed scaling (`if(CMT=2) IPRED = A2/V`). The current design
stays compatible:

| Concern | Phase 1 | Multi-analyte extension |
|---|---|---|
| Observed compartment | `obs_cmt_idx: Option<usize>` | `obs_cmt_indices: Vec<usize>` |
| Scaling dispatch | one `ScalingSpec` | `PerCmtScale(HashMap<usize, ScalingSpec>)` |
| Form C output | single `output_fn` | `output_fns: HashMap<usize, Box<...>>` |
| Parser | `y = <expr>` | `y[CMT=1] = <expr>`, `y[CMT=2] = <expr>` |

`subject.obs_cmts` already carries per-observation CMT — the infrastructure is in place.
Rule for Phase 1 implementation: **`apply_scaling` always receives the observation's CMT
index even though it ignores it today**, so the Phase 2 extension is purely additive.
