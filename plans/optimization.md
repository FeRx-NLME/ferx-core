# ferx-core Optimization Task Plan (Updated 2026-05-22)

This file is for use with Claude Code in the `ferx-core` repository.
Read `CLAUDE.md` first before starting any step.
Complete steps in order — later steps depend on earlier ones.
Each step specifies the files to touch and the expected outcome — but **the plan
below is a starting hypothesis, not a fixed recipe**. The actual repository state
may differ from what is described. Every step begins with deep evaluation of the
existing code, plan reconciliation, and explicit confirmation before any code is
written.

---

## Progress Summary (as of 2026-05-22)

All optimization steps completed on 2026-05-19. Steps 5b, 6b, 7, and 8 remain.
Since 2026-05-19, the following non-optimization PRs landed:
- #53 (NN-DCM feature), #58 (SLSQP overshoot fix), #57 (variance scale / `(sd)` annotation), #54 (PR-finding fix) — none affect the remaining optimization steps.
- #66 (Importance Sampling / IMP terminal chain stage — about to merge) — orthogonal to remaining optimization steps, **with one exception**: PR #66 hoists `obs_nll_single_into` from `saem.rs` to `stats/likelihood.rs` as `obs_nll_subject_into`. Step 8 must be started after PR #66 merges and must use `obs_nll_subject_into` from its new location.

| Step | PR | What | Why | What it adds |
|------|----|------|-----|--------------|
| **1** — rayon `par_iter` in GN | absorbed by #47 | Parallelize per-subject NLL evaluations in the GN finite-difference gradient loop | GN was computing subject NLL sequentially inside a per-parameter outer loop | Near-linear speedup with core count for the FD path of GN |
| **2** — Log-Cholesky for Ω | pre-existing | Store Cholesky diagonal in log space during packing | Keeps optimizer unconstrained (no positivity constraints on diagonal) | Was already done; no PR needed |
| **3** — AD gradients for GN/BHHH | [#47](https://github.com/FeRx-NLME/ferx-core/pull/47) | Replace central-FD score vectors with analytical Ω/σ gradients + forward-FD of predictions only for θ; restructure to rayon subject-parallel | Central-FD cost P·subjects per BHHH iteration; exact gradients are cheaper and improve Hessian quality | ~13k fewer model evaluations per fit on warfarin; exact BHHH matrix for GN; enables Step 5 |
| **4** — AD gradient for SAEM M-step | [#49](https://github.com/FeRx-NLME/ferx-core/pull/49) | Replace forward-FD of `obs_nll_sum` in the M-step NLopt closure with a single rayon subject-parallel pass using analytical σ gradients + FD-of-predictions for θ | Forward-FD launched n_dim sequential rayon jobs; analytical σ path eliminates extra predict calls entirely | Fewer NLopt OFV evaluations per M-step; better cache locality; single rayon launch vs n sequential ones |
| **5** — AD gradient in outer optimizer | [#48](https://github.com/FeRx-NLME/ferx-core/pull/48) | Replace central-FD `gradient_cd` in the NLopt SLSQP/LBFGS outer loop with `subject_nll_pop_grad` summed over subjects in parallel | Each outer FD gradient query cost P full inner-loop re-solves (re-running EBE for each parameter); AD provides the same gradient with fixed EBEs in one pass | Largest single wall-clock improvement for FOCE/FOCEI; eliminates the dominant cost of each outer iteration on ODE models |
| **6** — BHHH Hessian + AD gradient in trust-region | [#50](https://github.com/FeRx-NLME/ferx-core/pull/50) | Replace FD gradient and FD-of-gradient Hessian in `trust_region.rs` with `subject_nll_pop_grad` (gradient) and `4 Σ gᵢgᵢᵀ` (BHHH Hessian); cache per-subject gradients between the two calls; adaptive Steihaug CG budget | FD Hessian cost O(n²) OFV evaluations per outer iteration (98 for n=7); BHHH reuses gradients already computed, costs zero extra evaluations; BHHH is always PSD so CG is well-conditioned near constraints | Zero-cost Hessian per TR iteration; PSD guarantee eliminates CG conditioning failures; CG budget reduced from fixed 50 to adaptive ~5 for typical NLME models |
| **9** — Student-t SIR proposal | [#41](https://github.com/FeRx-NLME/ferx-core/pull/41) | Replace MVN proposal in SIR with multivariate Student-t (ν=5) | Normal proposal has thin tails; ESS collapses for parameters near boundaries (Ω variances, constrained θ) | Higher ESS without increasing `sir_samples`; more reliable 95% CIs for boundary-adjacent parameters |
| **10** — Parallel multi-start | [#42](https://github.com/FeRx-NLME/ferx-core/pull/42) | Run N independent full optimizations from perturbed initials in parallel via rayon; return lowest OFV | Local minima are the most common practical failure mode for nonlinear elimination, full-block Ω, and covariate models | On an 8-core machine, `n_starts = 8` gives ~8× lower probability of a local minimum at the same wall-clock cost |

**Remaining (1 step):**

| Step | Requires | Status |
|------|----------|--------|
| **5b** — IOV analytical gradient in outer optimizer | Step 5 ✅ | ✅ DONE — PR [#71](https://github.com/FeRx-NLME/ferx-core/pull/71) merged |
| **6b** — Eliminate double inner-solve in trust-region `cost()` | Step 6 ✅ | ✅ DONE — PR [#67](https://github.com/FeRx-NLME/ferx-core/pull/67) merged |
| **7** — GN → trust-region subproblem (replace LM damping + line search) | Steps 6 ✅ + 6b ✅ | ✅ DONE — PRs [#68](https://github.com/FeRx-NLME/ferx-core/pull/68) + [#69](https://github.com/FeRx-NLME/ferx-core/pull/69) merged |
| **8** — HMC proposals in SAEM E-step | Steps 3 ✅ + 4 ✅ + PR #66 merged | ❌ NOT STARTED |
| **11** — IOV support for SAEM | Steps 4 ✅ + 5b, Step 8 recommended | ❌ NOT STARTED |

**Recommended implementation order: 5b ✅ → 7 ✅ → 8 → 11 (IOV+SAEM, optional).**
- 6b: ✅ done (PR #67 merged).
- 7: ✅ done (PR #68 merged; missing `two_cpt_oral_cov` Tier 3 test added via PR #69).
- 5b: ✅ done (PR #71 merged). Key Phase B deviation: kappa H-matrices are not needed — the IOV NLL uses BSV-only FOCE linearisation + a direct kappa prior; `∂P_κ/∂L_iov` derived via forward/backward solve through `L_iov` only. No changes to `run_inner_loop_warm` or call sites.
- 8 last: largest scope, requires PR #66 merged; `autodiff` feature dependency means CI validation needs extra care.
- 11 (IOV+SAEM): optional, post-Step 8; see new step below.

---

## Status Legend

- ✅ **DONE** — fully implemented on `main`.
- 🔶 **PARTIAL** — partially done; see the per-step note.
- ❌ **NOT STARTED** — nothing relevant in `main` or open PRs.

---

## Project Goals — Non-Negotiable

ferx-core must be:

1. **Fast** — wall-clock time competitive with NONMEM FOCE and Monolix SAEM for
   equivalent models. Subject-level parallelism is fully exploited. Gradient
   computations use AD wherever the AD infrastructure supports it. Convergence
   is achieved in the minimum number of outer iterations consistent with accuracy.
2. **Efficient** — no wasted computation. No finite differences where AD is
   available. No Steihaug-CG iterations that the trust-region boundary makes
   irrelevant. No MH proposals at 40% acceptance when HMC at 65% is available.
3. **Stable** — convergence does not depend on user luck with initial estimates.
   Optimizer steps never violate constraints (positive-definiteness, parameter
   bounds). Failure modes are detected and reported as structured diagnostics,
   not silent wrong answers.
4. **Accurate** — final parameter estimates and OFV match NONMEM and Monolix
   to within numerical precision on standard test models. Standard errors and
   confidence intervals are honest (correct coverage, not just narrow numbers).

Every step in this plan exists to advance one or more of these properties.
A change that improves speed at the cost of accuracy is a regression.
A change that improves stability at the cost of speed is acceptable if the
stability gain is meaningful. When in doubt, accuracy wins.

---

## Workflow for Every Step — Mandatory

For each step below, follow this workflow exactly. **Do not skip the evaluation
phase. Do not start implementation before the plan is reconciled with reality.**

### Phase A — Deep Evaluation (before any code changes)

1. **Read every file the step touches**, in full. Do not skim. Open each file
   listed in "Files to touch" and read it end to end.

2. **Read the immediate callers** of any function the step modifies. Use
   `grep -r "function_name"` from the repo root to find callers. A change to
   a function signature or behaviour propagates — understand the propagation
   before making the change.

3. **Read the existing tests** for the affected modules. Look in
   `#[cfg(test)] mod tests` blocks at the bottom of each touched file.
   These tests encode the current contract. The plan must not break them
   silently — if a test will need to change, that's a flag that something
   in the plan is wrong or incomplete.

4. **Check `CLAUDE.md` for relevant constraints**. Specifically the AD-safe
   code rules (no `f64::max`/`f64::min` in AD-instrumented code), the
   `FitResult.warnings` convention (no stderr printing), and the test-first
   requirement (every new feature requires a test).

5. **Identify mismatches between this plan and the actual code.** Common
   mismatches:
   - The plan assumes a function exists or has a certain signature, but the
     actual code is different
   - The plan says "currently uses FD" but the code already uses AD
   - The plan describes a file structure that has since been refactored
   - The plan references an `Optimizer` enum variant or `FitOptions` field
     that doesn't exist yet, or is named differently

### Phase B — Plan Reconciliation

Based on Phase A findings, produce an **updated plan for this step** that
matches the actual code. The updated plan must:

1. List the exact function signatures, struct fields, and module paths as they
   exist today (not as the plan below assumes them to be).
2. Identify which sub-tasks are no-ops because the code already does what the
   plan describes (skip them, document why).
3. Identify any new sub-tasks the original plan missed but the actual code
   structure requires (e.g. an exported function signature must change in
   `src/ad/mod.rs` to expose a new AD function).
4. Restate the test plan in terms of the actual existing tests — which to
   extend, which to add, which to leave alone.
5. Explicitly state any deviation from the original plan and the reason for it.

Write this updated plan as a comment block at the top of the step's primary
PR description, or as a brief markdown summary before starting implementation.

### Phase C — Plan Verification

Before writing code:

1. **Read the updated plan back.** Does it cover the four project goals
   (fast, efficient, stable, accurate)? If a sub-task would introduce a
   regression in any of the four, fix the plan before proceeding.
2. **Check prerequisites.** Steps marked "Requires: Step N" require that
   Step N is fully merged and tested. Do not begin a dependent step until
   the prerequisite is on main and verified.
3. **Confirm AD-safe scope.** For steps that touch AD code, list every
   function reachable from the AD path and confirm none use `f64::max`,
   `f64::min`, or the LLVM `maximumnum`/`minimumnum` intrinsics.
4. **Confirm parallelism safety.** For steps that add or modify rayon usage,
   confirm there is no shared mutable state and that reductions use thread-safe
   primitives.

### Phase D — Implementation

Only now write code. Implement the updated plan from Phase B.

1. Make changes file by file.
2. Run `cargo check` after each file to catch type errors early.
3. Add tests as you go, not at the end. Follow the three-tier structure from
   `CLAUDE.md`:
   - **Tier 1 (unit, `src/`)** — Every new helper function (gradient correctness
     check, budget logic, cache hit/miss, leapfrog step) gets an inline
     `#[cfg(test)] mod tests` block in the same `.rs` file. These must not call
     `fit()`. Run with `cargo test --lib`.
   - **Tier 2 (integration, `tests/*.rs`)** — When the step introduces a new
     public-API behaviour (new `FitOptions` field, new optimizer variant, new
     estimation path), add an integration test that calls `fit()` with a low
     `outer_maxiter` (≤ 30) and asserts sane shape/non-panic/non-NaN — **not**
     convergence. These are compile-checked on every PR. See `tests/new_optimizers.rs`
     for the established pattern (`data_and_model()`, `base_options()`, low
     `outer_maxiter`).
   - **Tier 3 (slow convergence, `tests/*.rs`)** — Full population fits to
     convergence (OFV comparison, SE accuracy, iteration count comparison) must
     be gated:
     ```rust
     #[cfg_attr(not(feature = "slow-tests"), ignore = "slow: opt in with --features slow-tests")]
     ```
     These run nightly and on pushes to `main` that touch estimation code.
     Per-step "Test" sections below that say "OFV must match baseline" or
     "run N times" describe Tier 3 tests.
4. Run `cargo clippy` — fix any warnings before moving on.
5. Run `cargo test --lib` — all Tier 1 tests must pass before considering the
   step complete. Tier 2 tests are checked via `cargo check --tests` (they must
   compile; they run nightly).

### Phase E — Verification Against Goals

After implementation, verify the step has actually advanced the project goals:

1. **Fast** — Run a benchmark fit (warfarin or two_cpt_oral_cov, whichever
   is most affected by this step) and record wall-clock time. Compare to
   baseline (time on `main` before this step). If wall-clock got worse, the
   change is a regression — investigate before merging.
2. **Efficient** — For steps that change gradient computation, count function
   evaluations (NLopt provides this; argmin provides this; SAEM iterations are
   already counted). Confirm the new path uses fewer evaluations than the old.
3. **Stable** — Run the step's affected method on a deliberately ill-conditioned
   problem (e.g. warfarin with very bad initials, or `mm_oral.ferx` which has
   nonlinear elimination). Confirm convergence is at least as reliable as before.
4. **Accurate** — Run all examples. Every OFV must match the pre-step baseline
   to within 0.01 OFV units (the convergence tolerance). Standard errors must
   match to within 1% relative.

If any of the four checks fails, the step is not complete. Investigate, fix,
and re-verify before opening the PR.

### Phase F — Documentation and PR

1. Update `docs/src/` markdown sources for any user-visible change
   (new `[fit_options]` key, new optimizer variant, new estimation method
   behaviour). Rebuild the mdBook (`cd docs && mdbook build`) and commit
   both source and built output in the same commit.
2. Fill in every section of `.github/PULL_REQUEST_TEMPLATE.md`.
3. In the PR description, include the Phase B updated plan (the actual plan
   followed, not the one originally written below) and the Phase E benchmark
   numbers (before vs after).

---

## ✅ Step 2 — Log-Cholesky Parameterization for OMEGA

**DONE — no work needed.**

`src/estimation/parameterization.rs` already implements log-Cholesky for OMEGA.
In `pack_params`, the Cholesky diagonal is stored as `l[(i,i)].max(1e-10).ln()`
(lines 28 and 34 for diagonal and block cases respectively); off-diagonal
elements are stored as-is. In `unpack_params`, the diagonal is recovered via
`.exp()` (line 134). The round-trip is correct and `OmegaMatrix::from_matrix`
is called during unpack to keep the struct consistent.

The `.max(1e-10)` floor is still present at pack time but is harmless in
practice — it only clips variances below 1e-10, which are below any meaningful
estimation precision. This does not require a clamping workaround during
optimization.

A unit test exists at lines 487–512 of `parameterization.rs` that verifies the
log-theta round-trip. The Cholesky log-diagonal round-trip is implicitly covered
by the parameterization integration tests; no new test is required.

---

## Step 1 — rayon `par_iter` in FOCE Subject Loop

**Status: ✅ DONE — absorbed by PR #47 (Step 3).**

**No prerequisites. Start here.**

### What is already done

`src/stats/likelihood.rs` already uses `par_iter` for the subject loop:
- Line 582: `.par_iter()` in the FOCEI path
- Line 619: `.par_iter()` in the standard FOCE path

No changes needed to `likelihood.rs`.

### What remains

`src/estimation/gauss_newton.rs` — the `build_gn_system` function (starting
around line 483) computes per-subject score vectors via central finite
differences. The current loop structure is:

```
for j in 0..n_params {           // outer: parameter index
    for each subject {            // inner: subject
        nll_plus[subj] = ...
        nll_minus[subj] = ...
    }
    per_subj_grad[subj][j] = ...
}
```

This cannot be trivially parallelized over subjects at the outer level because
the outer loop is over parameters, not subjects. However, the per-subject NLL
evaluations within each FD step (`nll_plus` and `nll_minus` collections) can
each use `.par_iter()` in place of `.iter()`.

**Concretely:** The `.iter().enumerate().map(...)` chains that produce `nll_plus`
and `nll_minus` (search for `.iter()` near line 547 and 572 in `gauss_newton.rs`)
can be changed to `.par_iter().enumerate().map(...)` — both chains are independent
over subjects. The `per_subj_grad` accumulation loop (lines 595–633) can similarly
use a rayon `par_iter` with an `ndarray`-style atomic or a simple parallel `.map`
followed by a sequential accumulation.

**Note on architecture:** This is a useful but modest win for the FD path. The
larger structural improvement — where the outer loop becomes subject-indexed and
rayon parallelism is maximally exploited — arrives naturally with Step 3
(AD per-subject gradients). Step 1 here is a quick improvement to the FD path
that remains useful if GN is used without the AD feature enabled.

Add `use rayon::prelude::*;` inside the function scope, mirroring the pattern
in `saem.rs` (line 174) and `likelihood.rs` (line 6).

### Files to touch
- `src/estimation/gauss_newton.rs`

### Test
Run `cargo test --lib` — all tests must pass.
Run: `cargo run --release -- examples/warfarin.ferx --data data/warfarin.csv`
with `method = gn`. OFV must match the baseline exactly. Parallelism must not
change the result.

### Expected gain
Each FD perturbation evaluates all subjects' NLL. With 50 subjects, replacing
sequential `.iter()` with `.par_iter()` in those inner loops yields near-linear
speedup with core count for those collection passes.

---

## Step 3 — AD Gradients for GN/BHHH Per-Subject Score Vectors

**Status: ✅ DONE — merged via PR #47.**

**Phase B deviation (recorded):** The original plan called for propagating Dual
numbers through `tv_fn` to get population-parameter gradients. This is infeasible:
`tv_fn` is `Option<Box<dyn Fn(&[f64], &HashMap<String, f64>) -> Vec<f64>>>` — an
opaque closure that only accepts `f64` slices; Dual numbers cannot propagate
through it. The implementation instead uses:
- **Analytical matrix formulas** (exact) for omega and sigma packed parameters,
  derived from the Cholesky of R_tilde already computed during the FOCE NLL.
- **Forward-FD of `compute_predictions_with_tv` only** (not full NLL) for theta
  packed parameters — one predict call per theta parameter, reusing the baseline
  Cholesky. Equivalent accuracy: O(h) error same as central-FD for theta, exact
  for omega/sigma.
- **Central-FD fallback** retained for ODE models, IOV, and M3/BLOQ paths.

`build_gn_system` was restructured to Rayon subject-parallel accumulation
(previously per-parameter outer loop with inner subject `par_iter`).

**Requires: Step 1 complete. Requires PR #22 merged.**

### What exists in the AD infrastructure

`src/ad/ad_gradients.rs` provides:

- `individual_nll_ad(eta, tv, omega_inv_flat, log_det_omega, sigma_values, ...)`  
  — differentiates w.r.t. ETAs only. This is **Path A** from the original plan.
  The function is a flat-args Enzyme-differentiable function with a strict
  signature (all args as slices of f64, no generics). It cannot be used directly
  for population-parameter differentiation.

- `compute_nll_gradient_ad(eta, tv_adjusted, omega_inv_flat, ...) -> (f64, Vec<f64>)`  
  — convenience wrapper that calls `individual_nll_ad_grad` (Enzyme-generated
  derivative) and returns `(nll, d_eta)`. Same ETA-only scope.

- `predict_all_ad(...)` and `compute_jacobian_ad(...)` — predictions and
  Jacobian w.r.t. ETAs.

There is **no** population-parameter AD gradient function. A new function is
needed. This is the central deliverable of Step 3.

### What to do

**Sub-task 3a — Create `subject_nll_pop_grad`**

In `src/ad/ad_gradients.rs`, add a new function:

```rust
pub fn subject_nll_pop_grad(
    packed: &[f64],
    template: &ModelParameters,
    model: &CompiledModel,
    population: &Population,
    subj_idx: usize,
    eta_fixed: &[f64],
    h_matrix: &DMatrix<f64>,
    options: &FitOptions,
) -> (f64, Vec<f64>)
```

This function computes the per-subject FOCE NLL and its gradient with respect
to the packed population parameter vector, **with ETAs fixed** at their current
EBE values. Use forward-mode dual numbers (the `Dual` type already in
`src/ad/dual.rs`) — not Enzyme. The Enzyme path is for inner-loop ETA
differentiation; forward-mode dual numbers are the right tool for an
`n_packed`-dimensional gradient where `n_packed` is typically 5–20.

The function should:
1. Call `unpack_params(packed, template)` to get typed parameters.
2. Call the existing per-subject FOCE NLL function (same as `subject_nll_at` in
   `gauss_newton.rs`) but parameterised over `packed` as dual numbers.
3. Return the primal NLL and the gradient vector.

**Critical:** The dual-number path must not call `f64::max()` or `f64::min()`.
Review every function reachable from the dual-number path and replace with
explicit comparisons as required by CLAUDE.md.

**Sub-task 3b — Replace FD in `build_gn_system`**

In `src/estimation/gauss_newton.rs`, replace the central-FD score-vector loop
in `build_gn_system` with parallel calls to `subject_nll_pop_grad`:

```rust
let grads_and_nlls: Vec<(f64, Vec<f64>)> = population.subjects
    .par_iter()
    .enumerate()
    .map(|(i, _)| {
        subject_nll_pop_grad(&x, template, model, population, i,
                             &eta_hats[i], &h_matrices[i], options)
    })
    .collect();
```

Accumulate `H_BHHH = 4 Σ gᵢgᵢᵀ` and `grad = 2 Σ gᵢ` from `grads_and_nlls`.
The factor of 4 is correct: the BHHH Hessian approximation uses the outer
product of the OFV gradient (which is twice the NLL gradient).

After this change, the `_nll_base` computation at the top of `build_gn_system`
can be removed — the NLL values come out of the AD call for free.

**Sub-task 3c — Export from `src/ad/mod.rs`**

Add `pub use crate::ad::ad_gradients::subject_nll_pop_grad;` (or similar) to
`src/ad/mod.rs` so it can be used from `gauss_newton.rs` and later from
`outer_optimizer.rs` and `trust_region.rs`.

### Note on identity-packed thetas (PR #22)

After PR #22, some thetas use identity packing (`theta_packs_log` returns false
when `theta_lower < 0`). The dual-number gradient path must use the same
packing logic as `pack_params` / `unpack_params` — i.e., the chain rule is
trivial (derivative = 1) for identity-packed thetas, and `exp()` for
log-packed ones. The `theta_packs_log` helper in `parameterization.rs` must
be used consistently.

### Files to touch
- `src/ad/ad_gradients.rs` (add `subject_nll_pop_grad`)
- `src/ad/mod.rs` (export)
- `src/estimation/gauss_newton.rs` (replace FD loop with AD + rayon)

### Test

Add a unit test in `ad_gradients.rs` (or `gauss_newton.rs`) that:
1. Computes the per-subject NLL gradient for warfarin subject 1 via both
   central FD (h = 1e-5) and `subject_nll_pop_grad`.
2. Asserts agreement to 1e-5 relative tolerance for all packed parameters.
3. Exercises a model with an identity-packed theta (negative lower bound)
   to confirm the dual-number path handles mixed packing.

Run `method = gn` and `method = gn_hybrid` on warfarin. OFV must match the
FD baseline to within 0.01. Compare iteration counts.

### Expected gain

With P=7 packed parameters, FD costs 14 extra NLL evaluations per subject per
GN iteration. AD costs 1. With 50 subjects and 20 GN iterations: ~13,000 fewer
model evaluations per fit. BHHH Hessian quality improves because gradients are
exact.

---

## Step 4 — AD Gradient for SAEM M-Step (Theta and Sigma)

**Status: ✅ DONE — merged via PR #49.**

**Requires: Step 3 complete. Requires PR #22 merged.**

### Current state

After PR #22 merges, `src/estimation/saem.rs` will use forward-FD (not
central-FD) for the M-step NLopt objective gradient, with the base NLL value
reused. This is already a significant improvement (2.4× single-thread speedup
per the PR). Step 4 replaces the remaining FD gradient with AD.

### What to do

Locate the M-step NLopt objective closure in `saem.rs`. It has the signature:

```rust
|params: &[f64], gradient: Option<&mut [f64]>| -> f64
```

Replace the forward-FD gradient computation with a rayon-parallel sum of
`subject_nll_pop_grad` from Step 3:

```rust
if let Some(g) = gradient {
    let pop_grads: Vec<(f64, Vec<f64>)> = population.subjects
        .par_iter()
        .enumerate()
        .map(|(i, _)| {
            subject_nll_pop_grad(params, template, model, population, i,
                                 &current_eta_samples[i], &h_matrices[i], options)
        })
        .collect();
    let n = params.len();
    g.iter_mut().for_each(|x| *x = 0.0);
    for (_, grad_i) in &pop_grads {
        for k in 0..n { g[k] += grad_i[k]; }
    }
}
```

The M-step fixes ETAs at their sampled values from the E-step — these are
the `current_eta_samples` held in `SaemState`. Make sure you are passing the
right ETAs (sampled, not the mode/MAP estimates).

This change supersedes the PR #22 forward-FD improvement for the M-step.
Keep the PR #22 Ω⁻¹ caching improvement — it's orthogonal.

### Files to touch
- `src/estimation/saem.rs`
- Uses `subject_nll_pop_grad` from Step 3

### Test

Run warfarin SAEM (`method = saem`) 5 times. Compare:
- Final theta/omega estimates: must agree with FD baseline to within 0.1%
- Outer M-step NLopt evaluation count: must drop (NLopt reports this via its
  return value or a counter in the closure)
- Wall-clock time per SAEM iteration: must improve vs post-PR22 forward-FD

### Expected gain

M-step AD gradient eliminates all forward-FD evaluations. For P=7, this
removes 7 extra full-subject-loop evaluations per gradient query. Across
400 SAEM iterations with ~3 M-step gradient queries each: ~8,400 fewer
model evaluations.

---

## Step 5 — AD Gradient for FOCE/FOCEI Outer Optimizer

**Status: ✅ DONE — merged via PR #48.**

**Requires: Step 3 complete. Requires PR #22 merged.**

### What is already done

`TrustRegion` is already present in the `Optimizer` enum in `src/types.rs`
(line 1043) and is already wired in `outer_optimizer.rs` — it routes to
`src/estimation/trust_region.rs`. Users can set `optimizer = trust_region`
today. Documentation in `docs/src/model-file/fit-options.md` should be
verified/updated to include `trust_region`.

### What remains

`src/estimation/outer_optimizer.rs` calls `gradient_cd` (central FD,
defined at line 1293) to fill the NLopt gradient slice. After Step 3, replace
this with `subject_nll_pop_grad` summed over subjects via rayon:

```rust
let pop_grads: Vec<Vec<f64>> = population.subjects
    .par_iter()
    .enumerate()
    .map(|(i, _)| {
        subject_nll_pop_grad(&x, template, model, population, i,
                             &ehs[i], &hms[i], options).1
    })
    .collect();
for i in 0..g.len() {
    g[i] = pop_grads.iter().map(|gi| gi[i]).sum::<f64>() * scale[i];
}
```

This applies to both the primary SLSQP/LBFGS objective closure (`objective`,
around line 467) and the stagnation-guard path. Preserve the stagnation guard
from PR #22 — it operates on OFV values, not on whether FD or AD provides
the gradient.

Also apply to the `objective2` closure used for the post-convergence polish
pass (around line 693).

The built-in BFGS path (outside NLopt) in `outer_optimizer.rs` should also
receive the AD gradient if applicable — check that path and update it.

### Files to touch
- `src/estimation/outer_optimizer.rs` (replace `gradient_cd` calls with AD)
- `docs/src/model-file/fit-options.md` (verify `trust_region` is documented)

### Test

Run warfarin with `method = focei, optimizer = slsqp`. Compare outer iteration
count before and after — SLSQP should use fewer with an exact gradient.
Run with `optimizer = nlopt_lbfgs`. Run with `optimizer = trust_region`.
All OFVs must match baseline to within 0.01.

### Expected gain

For P=7 parameters, each NLopt FD gradient query in the outer loop costs 7
full inner-loop re-solves (each re-solving all subjects' EBEs). AD provides
the same gradient in 1 pass with fixed EBEs. This is the largest single
wall-clock improvement for FOCE/FOCEI on ODE models.

---

## Step 5b — Analytical Gradient for IOV Models in the Outer Optimizer

**Status: 🔶 IN REVIEW — PR [#71](https://github.com/FeRx-NLME/ferx-core/pull/71)**

**Requires: Step 5 ✅**

### Background

Step 5 replaced population-level central FD with `subject_nll_pop_grad` summed
in parallel over subjects. For non-IOV analytical PK models, `subject_nll_pop_grad`
takes the analytical path (`subject_nll_pop_grad_analytical`): exact for ω/σ
Cholesky elements, forward-FD of predictions only for θ. For IOV models,
`can_use_analytical = false` because `!kappas.is_empty()` (confirmed at
`src/estimation/gauss_newton.rs:847`), so it falls back to per-subject central FD
(cost = 2P subject NLL evals per outer gradient query).

The analytical gradient *can* be extended to IOV — the FOCE NLL structure is
identical, just with an expanded random-effects vector and a block-diagonal
variance matrix.

### Actual code state

`subject_nll_pop_grad` (`gauss_newton.rs:835`) has this gate:

```rust
let can_use_analytical = model.ode_spec.is_none()
    && kappas.is_empty()                     // ← IOV falls back here
    && !matches!(model.bloq_method, BloqMethod::M3);
```

`subject_nll_pop_grad_analytical` (`gauss_newton.rs:593`) accepts only
`eta_hat: &DVector<f64>` and `h_matrix: &DMatrix<f64>` — there is no
kappa/IOV parameter. IOV models always fall through to the central-FD path.

The existing FOCE IOV NLL path (`foce_subject_nll_iov` called at line 1014)
assembles a combined H-matrix and combined omega block internally — the
kappa H-matrices are computed there but not currently returned by
`run_inner_loop_warm`.

### Sub-task 5b-a — Prerequisite: expose kappa H-matrices from the inner loop

Read `src/estimation/inner_optimizer.rs` in full to find where kappa
H-matrices (`∂ipred/∂κ_occ`) are computed. They are used inside
`foce_subject_nll_iov` but not returned up the call stack.

Modify `run_inner_loop_warm` to return a `kappa_h_mats: Vec<Vec<DMatrix<f64>>>`
alongside the existing return tuple `(etas, h_mats, _, kappas)`. The outer type
is `Vec` over subjects; inner `Vec` over occasions. If the inner optimizer
already discards them after use, add the necessary storage.

This is a prerequisite — all subsequent sub-tasks depend on having kappa
H-matrices available at the outer optimizer call sites.

**Files:** `src/estimation/inner_optimizer.rs`, and all callers of
`run_inner_loop_warm` (search with `grep -rn "run_inner_loop_warm"` — expected
in `outer_optimizer.rs`, `trust_region.rs`, `gauss_newton.rs`). Each call site
must be updated to receive the new return value; unused sites may discard with `_`.

### Sub-task 5b-b — Add `subject_nll_pop_grad_analytical_iov`

In `src/estimation/gauss_newton.rs`, add a new function alongside
`subject_nll_pop_grad_analytical`:

```rust
fn subject_nll_pop_grad_analytical_iov(
    x: &[f64],
    template: &ModelParameters,
    model: &CompiledModel,
    population: &Population,
    subj_idx: usize,
    eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>,
    kappas: &[DVector<f64>],           // per-occasion κ EBEs
    kappa_h_mats: &[DMatrix<f64>],     // per-occasion ∂ipred/∂κ_occ Jacobians
    omega_iov: &OmegaMatrix,
    bounds: &PackedBounds,
    options: &FitOptions,
) -> Option<(f64, Vec<f64>)>
```

**Math:** For a subject with occasions `1…K`, the IOV random effects are
`[η; κ₁; …; κ_K]` and the combined variance block is:

```
Ω_combined = diag(Ω_bsv, Ω_iov, …, Ω_iov)   (K+1 blocks)
```

The FOCE linearisation gives:

```
r_tilde = R + H_combined · Ω_combined · H_combined^T
```

where `H_combined = [H_η | H_κ₁ | … | H_κ_K]` concatenates the Jacobians.

The gradient formulas follow the same pattern as `subject_nll_pop_grad_analytical`:
- `∂NLL/∂L_bsv[i,j]` — same formula applied to the η-block rows/columns of `C⁻¹ H`.
- `∂NLL/∂L_iov[i,j]` — **sum over occasions**: `Σ_occ [same formula applied to the κ_occ block]`.
- `∂NLL/∂σ_k` — unchanged.
- `∂NLL/∂θ` — forward-FD of ipred (unchanged).

### Sub-task 5b-c — Lift the `kappas.is_empty()` gate

In `subject_nll_pop_grad` (`gauss_newton.rs:847`), change `can_use_analytical` to:

```rust
let can_use_analytical = model.ode_spec.is_none()
    && !matches!(model.bloq_method, BloqMethod::M3);
```

Then dispatch on whether kappas are present:

```rust
if can_use_analytical {
    if kappas.is_empty() {
        if let Some(result) = subject_nll_pop_grad_analytical(...) {
            return result;
        }
    } else if let Some(ref omega_iov) = params.omega_iov {
        if let Some(result) = subject_nll_pop_grad_analytical_iov(
            ..., kappas, kappa_h_mats, omega_iov, ...
        ) {
            return result;
        }
    }
}
// Central-FD fallback (ODE / M3 / degenerate cases)
```

### Sub-task 5b-d — Thread `kappa_h_mats` through all outer call sites

Update `subject_nll_pop_grad`'s signature to accept
`kappa_h_mats: &[DMatrix<f64>]`. Update all call sites in:
- `src/estimation/outer_optimizer.rs` — pass per-subject kappa H-matrices from
  the `run_inner_loop_warm` return.
- `src/estimation/trust_region.rs` — same (currently passes `&[]` as a
  placeholder; replace with actual kappa H-matrices).
- `src/estimation/gauss_newton.rs` (the `build_gn_system` caller) — same.

### Files to touch
- `src/estimation/inner_optimizer.rs` — return kappa H-matrices
- `src/estimation/gauss_newton.rs` — add IOV analytical gradient function; update gate and dispatch; update `build_gn_system`
- `src/estimation/outer_optimizer.rs` — thread kappa H-matrices to gradient call
- `src/estimation/trust_region.rs` — thread kappa H-matrices to `compute_ad_grads`
- `src/types.rs` — add `kappa_h_mats` to `OuterResult` if needed for downstream consumers

### Tests

**Tier 1 (unit, `src/estimation/gauss_newton.rs`):**

Add a unit test analogous to `test_subject_nll_pop_grad_analytical_matches_fd`
but with a subject that has 2 occasions and non-empty `kappas`/`kappa_h_mats`.
Assert that `subject_nll_pop_grad_analytical_iov` output matches population-level
central FD to within `1e-4` for every packed parameter, including Ω_iov Cholesky
elements. Also update `test_outer_ad_gradient_fd_fallback_path` (or add a new
test) to confirm the analytical path is taken for non-ODE IOV models after
the gate change.

**Tier 2 (integration, `tests/iov_api.rs` or alongside existing IOV tests):**

Add a test that calls `fit()` on `examples/warfarin_iov.ferx` (or the IOV
fixture used in existing tests) with `outer_maxiter = 5`, asserts no panic and
non-NaN OFV — confirms the IOV analytical gradient path is wired through
to the outer optimizer without running to convergence.

**Tier 3 (slow, same file):**

Gate a full IOV convergence run:
```rust
#[cfg_attr(not(feature = "slow-tests"), ignore = "slow: opt in with --features slow-tests")]
```
Assert OFV matches the central-FD baseline to within 0.01. Assert iteration count
is equal or lower (analytical gradient should not regress convergence speed).

### Expected gain

Same ratio as Step 5 for non-IOV: instead of 2P full subject-NLL evals per
outer gradient query, cost drops to 1 forward-FD pass of predictions per θ
component. For IOV models with many occasions, the per-subject FD cost scales
with P, so the gain is proportional to P (number of packed population parameters).

---

## Step 6 — Trust Region: AD Gradient + BHHH Hessian + Adaptive Steihaug-CG

**Status: ✅ DONE — merged via PR #50.**

**Requires: Steps 3 and 5 complete.**

### What is already done

- `TrustRegion` enum variant exists and routes to `trust_region.rs`. ✅
- `steihaug_max_iters: usize` is in `FitOptions` (default `50` in `main`;
  changed to `50` in PR #22 as well). The field is present. ✅

### What remains

**Sub-task 6a — Replace FD gradient with AD gradient**

`src/estimation/trust_region.rs` implements `Gradient` for `FoceiProblem` via
the `grad_fixed` method (line 64). This is central FD with h = 1e-5. Replace
`grad_fixed` with a call to `subject_nll_pop_grad` from Step 3, summed over
subjects via rayon. The argmin `Gradient` trait expects `Vec<f64>` — match the
type.

**Sub-task 6b — Replace FD Hessian with BHHH approximate Hessian**

`src/estimation/trust_region.rs` implements `Hessian` for `FoceiProblem` via
forward-FD of `grad_fixed` (line 119). Replace with the BHHH approximation
from Step 3: `H = 4 Σ gᵢgᵢᵀ` where each `gᵢ` comes from `subject_nll_pop_grad`.
The AD call from sub-task 6a already computes the `gᵢ` vectors — cache them
in a `with_grads` helper to avoid a second population pass.

If argmin supports `HessianVectorProduct`, implement that instead of materializing
the full n×n BHHH matrix. Check the argmin docs for the current API version in
`Cargo.toml`.

**Sub-task 6c — Make Steihaug-CG iteration budget adaptive**

`steihaug_max_iters: usize` in `FitOptions` currently defaults to `50` (a fixed
value). The plan called for `Option<usize>` with adaptive default — but since
the field already exists as `usize`, the change is:

1. Change `steihaug_max_iters: usize` → `steihaug_max_iters: Option<usize>` in
   `FitOptions` (types.rs). Update the default to `None`.
2. In `trust_region.rs`, when `steihaug_max_iters` is `None`, use the adaptive
   budget function:

```rust
fn adaptive_steihaug_budget(
    n_params: usize,
    trust_radius: f64,
    min_radius_seen: f64,
) -> usize {
    let radius_ratio = (trust_radius / min_radius_seen.max(1e-10)).min(100.0);
    let base = (n_params as f64).sqrt().ceil() as usize;
    let budget = (base as f64 / radius_ratio.sqrt()).ceil() as usize;
    budget.clamp(5, n_params * 2)
}
```

Track `min_radius_seen` in the trust-region outer loop (it is accessible via
argmin's `TrustRegion` state — check whether argmin exposes the current trust
radius, or maintain it manually by comparing step norms).

The `steihaug_max_iters` field in `types.rs` is also referenced in the model
file parser (search for `"steihaug_max_iters"` — found at lines 1183 and 1193
in current `main`). Update those parse sites for the `Option<usize>` type.

When `Some(n)`, use the fixed value `n`. When `None`, use the adaptive budget.

### Files to touch
- `src/estimation/trust_region.rs` (sub-tasks 6a, 6b, 6c)
- `src/types.rs` (change `steihaug_max_iters: usize` → `Option<usize>`, update default and parse)
- `docs/src/estimation/foce.md` (document trust region, BHHH Hessian, adaptive CG)
- `docs/src/model-file/fit-options.md` (update `steihaug_max_iters` entry for `Option<usize>`)

### Test

Run warfarin with `optimizer = trust_region`. Confirm:
- Steihaug-CG iteration count is low (5–10) in early outer iterations and
  rises toward `sqrt(n_params)` near convergence.
- OFV matches SLSQP result to within 0.01.
- Setting `steihaug_max_iters = 50` overrides adaptive budget at every iteration.

Run `two_cpt_oral_cov.ferx`. Trust region should use fewer outer iterations
than SLSQP on the covariate model.

### Expected gain

BHHH (always PSD) avoids FD-Hessian conditioning failures near constraints.
Adaptive CG spends ~5 iterations early (boundary always hit) and ~10 late
(accurate subproblem). For n_params=8, fixed budget = 50 → adaptive ≈ 7 on
average: 7× reduction in total CG cost per outer iteration.

---

## Step 6b — Eliminate Double Inner-Solve in Trust-Region `cost()`

**Status: ❌ NOT STARTED**

**Requires: Step 6 ✅**

### Background

Flagged explicitly in the PR #50 review (confirmed by code inspection):

> "`cost()` calls `run_inner()` independently — it doesn't populate `grad_cache`.
> So within one TR iteration, the inner loop runs twice: once for `cost()` and
> once for `gradient()`. Not a regression, but a follow-up opportunity."

### Actual code state (confirmed in `trust_region.rs:126–133`)

```rust
impl CostFunction for FoceiProblem<'_> {
    fn cost(&self, p: &Vec<f64>) -> Result<f64, Error> {
        let (etas, h_mats) = self.run_inner(p);   // inner solve #1
        Ok(self.ofv_fixed(p, &etas, &h_mats))
        // grad_cache is NOT written here
    }
}
```

The argmin `TrustRegion` call order within each outer iteration (confirmed
from `trustregion_method.rs` in argmin 0.11.0):

1. Run Steihaug subproblem on current `(grad, hessian)` → step `pk` (no model calls).
2. `problem.cost(new_param)` → `run_inner(new_param)` (inner solve #1); nothing written to `grad_cache`.
3. If step accepted: `problem.gradient(new_param)` → `compute_ad_grads(new_param)` → cache miss → `run_inner(new_param)` again (inner solve #2) + AD pass.
4. `problem.hessian(new_param)` → `compute_ad_grads(new_param)` → full cache hit → free.

Inner solve #1 and #2 are redundant — they process the exact same `new_param`.

### What to do

Modify `cost()` to pre-warm `grad_cache` with the EBEs, using
`per_subj_grads: vec![]` as the sentinel meaning "EBEs ready, AD not yet done":

```rust
impl CostFunction for FoceiProblem<'_> {
    fn cost(&self, p: &Vec<f64>) -> Result<f64, Error> {
        let (etas, h_mats) = self.run_inner(p);
        let ofv = self.ofv_fixed(p, &etas, &h_mats);
        // Pre-warm cache so gradient() on the same x skips run_inner().
        *self.grad_cache.lock().unwrap() = Some(GradCache {
            x: p.clone(),
            etas: etas.clone(),
            h_mats: h_mats.clone(),
            per_subj_grads: vec![],  // empty = EBEs ready, AD not yet done
        });
        Ok(ofv)
    }
}
```

Modify `compute_ad_grads()` (`trust_region.rs:79–120`) to distinguish three cache states:

1. **Full hit** (`c.x == x` and `!c.per_subj_grads.is_empty()`): return cached EBEs + gradients directly.
2. **Partial hit** (`c.x == x` and `c.per_subj_grads.is_empty()`): EBEs are warm from `cost()` — skip `run_inner`, run the AD pass only, write full cache entry.
3. **Miss** (`c.x != x` or `cache.is_none()`): run `run_inner`, then AD pass, write full cache entry.

Concrete change in `compute_ad_grads` (starting at line 80):

```rust
fn compute_ad_grads(&self, x: &[f64]) -> (Vec<DVector<f64>>, Vec<DMatrix<f64>>, Vec<Vec<f64>>) {
    // Check for full hit or partial hit.
    let maybe_warm = {
        let cache = self.grad_cache.lock().unwrap();
        if let Some(ref c) = *cache {
            if c.x == x {
                if !c.per_subj_grads.is_empty() {
                    // Full hit: return everything from cache.
                    return (c.etas.clone(), c.h_mats.clone(), c.per_subj_grads.clone());
                } else {
                    // Partial hit: EBEs ready, AD still needed.
                    Some((c.etas.clone(), c.h_mats.clone()))
                }
            } else {
                None
            }
        } else {
            None
        }
    };

    // Either use warm EBEs (partial hit) or run inner solve (miss).
    let (etas, h_mats) = maybe_warm.unwrap_or_else(|| self.run_inner(x));

    // AD pass: compute per-subject gradients.
    let n_subj = self.population.subjects.len();
    let per_subj: Vec<Vec<f64>> = (0..n_subj)
        .into_par_iter()
        .map(|i| {
            subject_nll_pop_grad(
                x, self.init_params, self.model, self.population, i,
                &etas[i], &h_mats[i], &[], &self.bounds, self.options,
            ).1
        })
        .collect();

    // Write full cache entry.
    *self.grad_cache.lock().unwrap() = Some(GradCache {
        x: x.to_vec(),
        etas: etas.clone(),
        h_mats: h_mats.clone(),
        per_subj_grads: per_subj.clone(),
    });

    (etas, h_mats, per_subj)
}
```

### Files to touch
- `src/estimation/trust_region.rs` only

### Tests

**Tier 1 (unit, `src/estimation/trust_region.rs`):**

Add a test that verifies the pre-warm behaviour without needing a full model:
1. After calling `cost(x)`, assert `grad_cache` contains an entry with `x == x`
   and `per_subj_grads.is_empty() == true`.
2. After calling `gradient(x)` on the same `x`, assert `grad_cache` now has
   `!per_subj_grads.is_empty()`.
3. Assert that the OFV returned by `cost(x)` equals `ofv_fixed(x, &cached_etas, &cached_h_mats)`
   to `f64` precision — confirms the pre-warmed EBEs are consistent with the cost.
4. Calling `gradient(x)` without a preceding `cost(x)` must not panic — the
   miss path must still call `run_inner` as a fallback.

**Tier 3 (slow, `tests/new_optimizers.rs`):**

Gate a full warfarin `optimizer = trust_region` convergence run with
`#[cfg_attr(not(feature = "slow-tests"), ignore = "slow: opt in with --features slow-tests")]`.
Assert OFV matches the SLSQP baseline to within 0.01. This is a pure correctness
check — the cache change must be transparent to the result.

### Expected gain

Halves `run_inner_loop_warm` calls per accepted TR outer iteration. For ODE
models where the inner loop dominates (RK45 per subject), this is a direct
~2× reduction in the most expensive part of each outer step. For analytical
PK models the gain is smaller but still measurable (EBE BFGS solve per subject
is not free on large populations).

---

## Step 7 — GN: Replace LM Damping + Line Search with Trust-Region Subproblem

**Status: ❌ NOT STARTED**

**Requires: Steps 3 ✅, 6 ✅, and 6b**

### Background

Currently `method = gn` uses BHHH + LM (Levenberg-Marquardt) damping:
`δ = (H_bhhh + λI)⁻¹ (−g)` followed by a backtracking line search
(in `gauss_newton.rs`). This is a scalar approximation: λ scales all
directions equally. The trust-region subproblem introduced in Step 6 adapts
step length per eigendirection of `H_BHHH` — better suited to NLME parameter
spaces where Ω variances (well-identified) and covariate exponents
(weakly identified) live on very different curvature scales.

### Actual code state

`trust_region.rs` does NOT currently expose a standalone subproblem solver.
The argmin `Steihaug` CG implementation is embedded in the `optimize_trust_region`
function via the `Executor` framework and cannot be called for a single step in
isolation. A self-contained Steihaug-CG function must be added.

`gauss_newton.rs` — the LM + line search loop is in `run_gn_optimization`
(search for the `lambda` variable and the backtracking loop). These are the
code paths to replace.

### Sub-task 7a — Implement standalone `pub fn solve_trust_region_subproblem`

Add to `src/estimation/trust_region.rs`:

```rust
/// Steihaug truncated-CG trust-region subproblem.
/// Returns the step δ satisfying ‖δ‖ ≤ trust_radius that approximately
/// minimises the quadratic model ½ δᵀ H δ + gᵀ δ.
pub fn solve_trust_region_subproblem(
    g: &DVector<f64>,
    h: &DMatrix<f64>,
    trust_radius: f64,
    max_iters: usize,
) -> DVector<f64>
```

Implement standard Steihaug-CG (≈50 lines — Nocedal & Wright, Algorithm 7.2):

```
p = 0,  r = g,  d = −g
for j = 0..max_iters:
    if dᵀ H d ≤ 0:    // zero or negative curvature — go to boundary
        τ = solve ‖p + τd‖ = Δ  (take positive root)
        return p + τd
    α = rᵀr / dᵀHd
    p_new = p + α d
    if ‖p_new‖ ≥ Δ:   // step exits trust region
        τ = solve ‖p + τd‖ = Δ  (take positive root)
        return p + τd
    r_new = r + α H d
    if ‖r_new‖ < ε ‖r_0‖:  // converged
        return p_new
    β = r_newᵀ r_new / rᵀr
    d = −r_new + β d
    p = p_new,  r = r_new
return p
```

`ε = 1e-10`, initial `max_iters` via the `max_iters` parameter (use
`adaptive_steihaug_budget(g.len())` at the call site in GN — reuse the
existing helper from Step 6c). The "positive root" for the boundary step is:

```
τ = (−pᵀd + sqrt((pᵀd)² − ‖d‖²(‖p‖² − Δ²))) / ‖d‖²
```

This function must have no dependency on argmin — it is pure nalgebra.

### Sub-task 7b — Replace LM damping + line search in `gauss_newton.rs`

In `run_gn_optimization` (or wherever the LM step is computed), replace the
LM solve and backtracking loop with:

```rust
// Compute step via TR subproblem
let step = solve_trust_region_subproblem(&grad, &h_bhhh, trust_radius, cg_budget);

// Compute proposed OFV at x + step
let x_new = /* clamp to bounds */;
let (eta_hats_new, h_mats_new) = run_inner_loop_warm(..., x_new);
let ofv_new = 2.0 * pop_nll(..., x_new, ...);

// TR ratio update
let pred_reduction = -grad.dot(&step) - 0.5 * step.dot(&(h_bhhh * &step));
let rho = (ofv_current - ofv_new) / pred_reduction;

if rho > 0.75 {
    trust_radius = (trust_radius * 2.0).min(delta_max);  // expand
} else if rho < 0.25 {
    trust_radius /= 4.0;  // shrink, reject step
    continue;
}
// Accept step (rho > 0.25)
x = x_new;
ofv_current = ofv_new;
eta_hats = eta_hats_new;
h_mats = h_mats_new;
```

Initialise `trust_radius = 1.0`, `delta_max = 10.0` (same defaults as
`optimize_trust_region`). Remove `lambda` and the backtracking loop entirely.

### Sub-task 7c — Apply adaptive Steihaug budget

At the `solve_trust_region_subproblem` call site in GN, set:
```rust
let cg_budget = adaptive_steihaug_budget(x.len());
```
Reuse the `adaptive_steihaug_budget` helper already defined in `trust_region.rs`.
Make it `pub(crate)` if not already.

### Files to touch
- `src/estimation/trust_region.rs` — add `pub fn solve_trust_region_subproblem` and make `adaptive_steihaug_budget` accessible
- `src/estimation/gauss_newton.rs` — replace LM + backtracking with TR subproblem + ratio update

### Tests

**Tier 1 (unit, `src/estimation/trust_region.rs`):**

1. `test_solve_trust_region_subproblem_respects_radius`: given a 2×2 PSD
   Hessian and gradient, assert `‖result‖ ≤ trust_radius` for trust radii
   [0.1, 0.5, 1.0, 5.0].
2. `test_solve_trust_region_subproblem_improves_quadratic_model`: assert
   `½ δᵀ H δ + gᵀ δ < 0` (the quadratic model decreases).
3. `test_solve_trust_region_subproblem_negative_curvature`: pass a Hessian with
   a negative eigenvalue. Assert the returned step reaches the trust boundary
   (`‖result‖ ≈ trust_radius`) rather than panicking.

**Tier 1 (unit, `src/estimation/gauss_newton.rs`):**

4. `test_gn_tr_ratio_expands_radius`: mock `ofv_current`, `ofv_new`, `pred_reduction`
   such that `ρ = 0.9`. Assert `trust_radius` doubles after the update.
5. `test_gn_tr_ratio_rejects_step`: set values so `ρ = 0.1`. Assert step is
   rejected and `trust_radius` quarters.

**Tier 2 (integration, `tests/new_optimizers.rs`):**

Add a non-convergence test: `fit()` with `method = gn` and `outer_maxiter = 5`
on warfarin. Assert no panic and non-NaN OFV — confirms TR subproblem wiring
compiles and runs.

**Tier 3 (slow, `tests/gn_convergence.rs` — new file):**

Gate two convergence tests:
```rust
#[cfg_attr(not(feature = "slow-tests"), ignore = "slow: opt in with --features slow-tests")]
```
1. Warfarin `method = gn`: OFV must match the LM-damping baseline to within 0.01.
   Record outer iteration count — TR step should not regress.
2. `two_cpt_oral_cov.ferx` `method = gn`: compare outer iteration count vs
   LM-damping baseline. The TR step should use ≤ iterations on the covariate
   model (weakly identified covariate directions benefit from per-direction
   step scaling).

### Expected gain

The LM scalar `λ` treats all parameter directions equally. The TR subproblem
adapts step length per eigendirection of `H_BHHH`. For NLME covariate models
where Ω variances are well-identified and covariate exponents are weakly
identified, this avoids the pattern of shrinking the step globally when only
one direction is problematic — typically 10–30% fewer outer iterations.

---

## Step 8 — HMC Proposals in the SAEM E-Step

**Status: ❌ NOT STARTED**

**Requires: Steps 3 ✅, 4 ✅, and PR #66 merged**

### PR #66 interaction — read before starting

PR #66 (Importance Sampling) hoists `obs_nll_single_into` from `saem.rs` into
`stats/likelihood.rs` as `obs_nll_subject_into`, and updates all SAEM call sites.
Step 8 must be started **after PR #66 merges**. After the merge, re-read `saem.rs`
around line 744 to confirm the E-step loop structure before writing any code.

### Actual code state (current `main`, may differ post-PR #66 — verify)

`src/estimation/saem.rs` uses `mh_steps` (line 66) — a Metropolis-Hastings
random-walk sampler. Per-subject step sizes are tracked in `state.step_scales`
and adapted every `adapt_interval` steps. The adaptation targets acceptance
rates around 40% (line 951–955). No HMC code exists anywhere in the codebase.

`compute_nll_gradient_ad` in `src/ad/ad_gradients.rs` provides
`∇_η NLLᵢ(η | θ, Ω, σ)` — the ETA gradient needed for HMC leapfrog.
Before using it, confirm that `FlatDoseData::from_subject` is compatible with
how SAEM subjects are stored (read the relevant struct definitions to verify).

The `autodiff` feature flag gates Enzyme-generated gradient functions. The HMC
path is only available when `autodiff` is enabled. When it is not, the code
must fall back to the existing MH sampler with a warning.

### Sub-task 8a — New file `src/estimation/hmc.rs`

Create `src/estimation/hmc.rs` with two public functions.

#### `pub fn leapfrog`

Standard velocity-Störmer-Verlet integrator (half-step p, n full steps q + p,
half-step p):

```rust
pub fn leapfrog(
    eta: &[f64],
    momentum: &[f64],
    nll_grad_eta: &dyn Fn(&[f64]) -> Vec<f64>,
    step_size: f64,
    n_steps: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut q = eta.to_vec();
    let mut p = momentum.to_vec();
    let g = nll_grad_eta(&q);
    for k in 0..p.len() { p[k] -= 0.5 * step_size * g[k]; }
    for _ in 0..n_steps {
        for k in 0..q.len() { q[k] += step_size * p[k]; }
        let g = nll_grad_eta(&q);
        for k in 0..p.len() { p[k] -= step_size * g[k]; }
    }
    let g = nll_grad_eta(&q);
    for k in 0..p.len() { p[k] -= 0.5 * step_size * g[k]; }
    (q, p)
}
```

No AD restrictions apply to `leapfrog` itself — it operates on plain `f64`
vectors returned from the Enzyme-generated gradient. No `f64::max`/`f64::min`
concern here.

#### `pub fn hmc_step`

```rust
pub fn hmc_step(
    subject: &Subject,
    eta: &DVector<f64>,
    params: &ModelParameters,
    model: &CompiledModel,
    step_size: f64,
    n_leapfrog: usize,
    rng: &mut impl Rng,
) -> (DVector<f64>, bool)   // (new_eta, accepted)
```

Algorithm:
1. Sample momentum `p ~ N(0, Ω⁻¹)`. Mass matrix M⁻¹ = Ω means `p ~ N(0, Ω⁻¹)`;
   kinetic energy `K(p) = ½ pᵀ Ω p`. Rationale: ETAs are distributed as
   `N(0, Ω)`, so M = Ω⁻¹ pre-conditions the proposal to the prior curvature.
   Sample by drawing `z ~ N(0, I)` then computing `p = L_omega_inv * z` where
   `L_omega_inv` is the inverse of the Cholesky factor of Ω.
2. Compute current Hamiltonian: `H_curr = NLLᵢ(η) + ½ pᵀ Ω p`.
3. Call `leapfrog(eta, p, grad_fn, step_size, n_leapfrog)` → `(eta_prop, p_prop)`.
4. Compute proposed Hamiltonian: `H_prop = NLLᵢ(η_prop) + ½ p_propᵀ Ω p_prop`.
5. Accept with probability `min(1, exp(H_curr − H_prop))`.
6. Return `(eta_prop, true)` if accepted, `(eta.clone(), false)` if rejected.

The gradient closure for leapfrog wraps `compute_nll_gradient_ad`:
```rust
let grad_fn = |q: &[f64]| -> Vec<f64> {
    compute_nll_gradient_ad(q, tv_adjusted, omega_inv_flat, ...).1
};
```
Verify the exact `compute_nll_gradient_ad` signature at call time and match
the subject data layout to what `FlatDoseData::from_subject` produces.

### Sub-task 8b — Add `saem_n_leapfrog: usize` to `FitOptions`

In `src/types.rs`, add to `FitOptions`:
```rust
pub saem_n_leapfrog: usize,   // default: 3
```

Update `Default::default()` for `FitOptions`. Add parsing in
`src/parser/model_parser.rs` (follow the pattern of `saem_n_mh_steps`).

### Sub-task 8c — Replace `mh_steps` with `hmc_step` in the SAEM E-step loop

In `src/estimation/saem.rs`, inside the E-step loop (around line 744 — verify
post-PR #66 location), replace:

```rust
let (n_acc, nll_new) = mh_steps(..., n_mh_steps, ...);
```

With a feature-gated dispatch:

```rust
#[cfg(feature = "autodiff")]
let (new_eta, accepted) = hmc_step(subject, &eta, &params, model,
                                    state.step_scales[i], options.saem_n_leapfrog, &mut rng);
#[cfg(not(feature = "autodiff"))]
let (new_eta, accepted) = {
    // MH fallback — emit warning once per fit, not once per step
    (mh_result_eta, mh_accepted)
};
```

The MH fallback warning must be emitted into `FitResult.warnings` (not stderr).
Use a flag to emit it at most once per fit (not per E-step iteration or subject).

### Sub-task 8d — Update step-size adaptation target

The adaptation loop at lines 951–955 (verify post-PR #66) targets acceptance
rate implicitly around 40% (MH). For HMC, target 65%. Change the threshold
constant:

```rust
// Before (MH):
if rate > 0.4 { ... scale up ... } else { ... scale down ... }

// After (HMC):
let target_rate = if options.saem_n_leapfrog > 0 { 0.65 } else { 0.4 };
if rate > target_rate { ... scale up ... } else { ... scale down ... }
```

### Sub-task 8e — Register `hmc` module

In `src/estimation/mod.rs`, add:
```rust
pub(crate) mod hmc;
```

### Files to touch
- `src/estimation/hmc.rs` (new file)
- `src/estimation/mod.rs` (declare `hmc` module)
- `src/estimation/saem.rs` (replace `mh_steps` dispatch; update adaptation target)
- `src/types.rs` (add `saem_n_leapfrog: usize`)
- `src/parser/model_parser.rs` (parse `saem_n_leapfrog`)
- `docs/src/estimation/saem.md` (document HMC path, `saem_n_leapfrog`, fallback behaviour)
- `docs/src/model-file/fit-options.md` (add `saem_n_leapfrog` entry)

### Tests

**Tier 1 (unit, `src/estimation/hmc.rs`):**

1. `test_leapfrog_energy_conservation`: 1D harmonic oscillator where
   `NLL(q) = ½ q²` (Gaussian prior, no observations). Hamiltonian
   `H = ½ q² + ½ p²` is analytically conserved. Run leapfrog with
   `step_size = 0.1`, `n_steps = 10`. Assert `|H_before − H_after| < 0.01`
   (Verlet discretization error is O(ε²L)). This verifies the half-step
   implementation is correct.
2. `test_leapfrog_single_step_n0`: `n_steps = 0` degenerates to two half-steps
   only. The proposed q must equal the original eta (no full position step).
   Assert no panic.
3. `test_hmc_step_zero_step_size_always_accepts`: with `step_size = 0.0` and
   `n_leapfrog = 1`, the proposed position equals the current position, ΔH = 0,
   and the step must always be accepted. Assert `accepted == true` across 10
   random momentum draws.
4. `test_hmc_step_hamiltonian_computation`: hand-compute `H = NLL(η) + ½ pᵀ Ω p`
   for a known simple NLL and known Ω. Assert the internal computation in
   `hmc_step` matches to 1e-10.

**Tier 2 (integration, `tests/saem_hmc_api.rs` — new file):**

Gate on `#[cfg(feature = "autodiff")]` — the HMC path is not available without
the Enzyme toolchain. Call `fit()` with `method = saem` and `outer_maxiter = 5`
on warfarin. Assert no panic, non-NaN OFV, and `FitResult.warnings.is_empty()`
(confirms the HMC path was taken, not the MH fallback).

**Tier 3 (slow, same file):**

Gate two convergence tests:
```rust
#[cfg_attr(not(feature = "slow-tests"), ignore = "slow: opt in with --features slow-tests")]
```
1. Warfarin SAEM 5× with different seeds. Assert mean acceptance rate ≥ 55%
   (HMC). Final theta/omega must agree with the MH baseline to within 2%.
   Record acceptance rate vs seed for variance check.
2. `mm_oral.ferx` with full-block Ω. Confirm convergence and sane OFV — this
   model has nonlinear elimination that stresses the E-step; HMC should not
   regress convergence reliability.

### Expected gain

At MH acceptance ~40%, 60% of E-step ODE evaluations are wasted. HMC with
3 leapfrog steps targets 65–85% acceptance — each accepted proposal moves
further from the current point because the gradient guides the trajectory.
Effective sample size per E-step increases by 3–5×. The exploration phase
stabilizes faster, reducing total SAEM iterations needed.

---

## Step 9 — Student-t Proposal for SIR

**Status: ✅ DONE — merged via PR #41.**

**No prerequisites. Can be done at any time independently.**

### Current state

`src/estimation/sir.rs` uses a multivariate normal (MVN) proposal throughout:
- Samples drawn at line 135 via `StandardNormal` + Cholesky decomposition
- Log-proposal density computed at lines 214–218 as a standard MVN log-density

`rand_distr::ChiSquared` is not currently imported in `sir.rs`. Check whether
it is in `Cargo.toml` under `rand_distr` — if yes, no new dependency needed.

### What to do

Replace with a multivariate Student-t with ν = 5 degrees of freedom.

**Sampling:**
```rust
let z = (0..n_packed).map(|_| rng.sample(StandardNormal)).collect::<Vec<_>>();
let z_vec = DVector::from_vec(z);
let chi2: f64 = ChiSquared::new(nu as f64).unwrap().sample(&mut rng);
let delta = &proposal_chol * &z_vec * (nu as f64 / chi2).sqrt();
let sample = &center + delta;
```

**Importance weights** — use the multivariate Student-t log-density, not the
MVN density. The log-density (dimension d, df ν, scale Σ with log-det
`log_det_proposal`):
```
log p_t(x) = lgamma((ν+d)/2) - lgamma(ν/2) - d/2*log(νπ) - ½*log_det_proposal
             - (ν+d)/2 * log(1 + zᵀz / ν)
```
where `z = L⁻¹(x - center)` (the standardized residual already computed in
the existing code as `z_vec`).

The key change: replace `log_q_hat - 0.5 * z_vec.dot(&z_vec)` (current MVN
log-density, line 214 area) with the Student-t log-density above.

Add `sir_df: f64` to `FitOptions` (default: 5.0, following Dosne 2017).

### Files to touch
- `src/estimation/sir.rs`
- `src/types.rs` (add `sir_df: f64`)
- `docs/src/estimation/sir.md`
- `docs/src/model-file/fit-options.md`

### Test
Fit warfarin with `covariance = true`. Compare ESS with normal vs Student-t
proposal — Student-t ESS must be equal or higher.

Confirm `sir_df = 30.0` produces results essentially identical to the MVN
proposal (large ν → normal limit).

### Reference
Dosne A-G, Bergstrand M, Karlsson MO. Improving the estimation of parameter
uncertainty distributions in nonlinear mixed effects models using sampling
importance resampling. J Pharmacokinet Pharmacodyn. 2017;44(6):539–562.

### Expected gain
Higher ESS without increasing `sir_samples`. More reliable 95% CIs for
OMEGA variances (hard boundary at 0) and correlated parameters. Minimal
implementation cost.

---

## Step 10 — Parallel Multi-Start Outer Optimization

**Status: ✅ DONE — merged via PR #42.**

**No prerequisites. Can be done at any time. Uses rayon already in Cargo.toml.**

### What to do

Add `n_starts: usize` to `FitOptions` (default: 1 — single run, no behaviour
change). Add `start_sigma: f64` (default: 0.3).

When `n_starts > 1`, run N independent full optimizations in parallel via rayon.
Return the result with the lowest OFV among all converged runs.

**Perturbation scheme:**
```rust
fn perturb_init(
    params: &ModelParameters,
    start_idx: usize,
    sigma: f64,
    base_seed: u64,
) -> ModelParameters {
    if start_idx == 0 { return params.clone(); }  // start 0: exact user initials
    let mut rng = SmallRng::seed_from_u64(base_seed + start_idx as u64);
    let normal = Normal::new(0.0, sigma).unwrap();
    let mut p = params.clone();
    for (i, t) in p.theta.iter_mut().enumerate() {
        // Only perturb log-packed thetas in log space; perturb identity-packed
        // thetas additively (σ * sample).
        if theta_packs_log(p.theta_lower[i]) {
            *t *= normal.sample(&mut rng).exp();
        } else {
            *t += sigma * normal.sample(&mut rng);
        }
    }
    // Perturb omega diagonal variances in log-space; leave correlations unchanged
    p
}
```

Note: uses `theta_packs_log` from PR #22 to correctly handle negative-lower-bound
thetas in the perturbation scheme.

**Parallel execution:**
```rust
let results: Vec<FitResult> = (0..n_starts)
    .into_par_iter()
    .map(|k| {
        let init_k = perturb_init(&params, k, start_sigma, base_seed);
        fit_single(&model, &population, &init_k, &single_start_options)
    })
    .collect();

let best = results.into_iter()
    .filter(|r| r.converged)
    .min_by(|a, b| a.ofv.partial_cmp(&b.ofv).unwrap());
```

If no start converged, return the result with the lowest OFV regardless and
add a warning. Add `start_index` to the YAML output indicating which start won.

Note on nested rayon: each start uses rayon for subject-level parallelism (Step 1).
Rayon's work-stealing thread pool handles nested `par_iter` safely.

### Files to touch
- `src/api.rs` (wrap `fit()` call in multi-start parallel loop)
- `src/types.rs` (add `n_starts`, `start_sigma` to `FitOptions`)
- `docs/src/model-file/fit-options.md`
- `docs/src/estimation/foce.md` (add multi-start section)

### Test
Fit `mm_oral.ferx` with deliberately poor initial estimates (TVCL = 10× true).
With `n_starts = 1` it should sometimes land in a suboptimal OFV.
With `n_starts = 8` it should consistently find the global minimum.

Verify `n_starts = 1` gives identical results to the default.
Verify reproducibility: same `seed` and `n_starts` always gives the same best result.

### Expected gain
On an 8-core machine, `n_starts = 8` takes the same wall-clock time as a single
run but has ~8× lower probability of a local minimum. For models with nonlinear
elimination, full-block OMEGA, or many covariates, local minima are the most
common practical failure mode.

---

## Step 11 — IOV Support for SAEM

**Status: ❌ NOT STARTED**

**Requires: Steps 4 ✅, 5b ✅, and Step 8 (HMC E-step) recommended first**

### Background

SAEM currently rejects any model with `n_kappa > 0` at the API level (`api.rs:570-572`):

```rust
if model.n_kappa > 0 && chain.iter().any(|&m| m == EstimationMethod::Saem) {
    return Err("method = saem does not support IOV (n_kappa > 0). ...")
}
```

Supporting IOV in SAEM requires changes in three areas: the E-step (sampling kappas alongside etas), the M-step (including Ω_iov as an optimized parameter with its prior gradient), and removing the hard block.

The fundamental difficulty is that SAEM's E-step currently samples a per-subject eta vector from a single MH/HMC chain. For IOV subjects, the per-occasion kappas are additional random effects with their own prior `N(0, Ω_iov)` — they must also be sampled and tracked in SAEM state.

### Actual code state

**E-step** (`saem.rs`): `mh_steps` / `hmc_step` (Step 8) samples a single `DVector<f64>` per subject. SAEM state holds `Vec<DVector<f64>>` (one eta per subject). No kappa state exists.

**M-step** (`saem.rs`): The M-step NLopt closure calls `obs_nll_sum` (or `obs_nll_subject_into` post-PR #66), which evaluates the observation likelihood for fixed etas. Ω_iov is not included in the M-step optimizer's parameter vector; the kappa prior contributes nothing to the M-step objective.

**Block**: `api.rs:570-572` — explicit `Err` return before any estimation begins.

**Bobyqa + IOV**: already works. The FOCE outer optimizer routes `pop_nll` → `foce_population_nll_iov` correctly (`outer_optimizer.rs:136-149`). Only SAEM is blocked.

### Sub-task 11a — SAEM state: add kappa samples

In `src/estimation/saem.rs`, extend `SaemState` to hold per-subject per-occasion kappa samples:

```rust
pub(crate) struct SaemState {
    pub etas: Vec<DVector<f64>>,
    pub kappas: Vec<Vec<DVector<f64>>>,  // NEW: [subject][occasion]
    pub step_scales: Vec<f64>,
    // ... existing fields
}
```

Initialize `kappas` from zeros (or the inner loop warm-start) at the start of SAEM. At the end of each SAEM iteration, the kappa samples feed into the M-step alongside etas.

### Sub-task 11b — E-step: sample kappas per occasion

For each subject in the E-step, after sampling eta, sample each kappa_k independently using a second MH (or HMC if Step 8 is done) chain:

```
For each occasion k:
    κ_k_prop ~ q(κ_k_prop | κ_k_curr, step_scale_kappa)
    log_alpha = individual_nll(eta, κ_k_prop) - individual_nll(eta, κ_k_curr)
              + 0.5 * (κ_k_curr^T Ω_iov^{-1} κ_k_curr - κ_k_prop^T Ω_iov^{-1} κ_k_prop)
    accept with probability min(1, exp(-log_alpha))
```

The NLL function used here must accept a combined `[eta, kappa_k]` vector per occasion — reuse `individual_nll` with the combined eta that `foce_subject_nll_iov` already constructs.

Alternatively (more efficient): treat `[eta; κ_1; …; κ_K]` as a single augmented vector and sample jointly with one MH/HMC proposal. For HMC this is natural; for MH this requires a block proposal. The per-occasion Gibbs-style approach (option above) mixes better when K is large and eta/kappa are weakly correlated.

Add `saem_kappa_step_scales: Vec<Vec<f64>>` to `SaemState` for adaptation, or reuse a single scale per subject (simpler first cut).

### Sub-task 11c — M-step: include Ω_iov in parameter vector

Currently the M-step packs `[theta | omega_bsv chol | sigma]` (no IOV). Extend the M-step NLopt parameter vector to include the IOV Cholesky entries (same log-packed diagonal, identity off-diagonal as in `pack_params`).

The M-step objective for subject i with occasion k kappa samples is:

```
NLL_M(theta, Omega_bsv, Omega_iov, sigma) =
    obs_nll_i(theta, eta_i, sigma)                    (observation likelihood)
  + 0.5 * eta_i^T Omega_bsv^{-1} eta_i + 0.5 * log|Omega_bsv|   (BSV prior)
  + 0.5 * Σ_k [κ_{i,k}^T Ω_iov^{-1} κ_{i,k} + log|Ω_iov|]     (IOV prior)
```

The kappa prior contribution is identical to `foce_subject_nll_iov`'s kappa prior — use the same formula. The gradient w.r.t. Ω_iov Cholesky is the same as computed in Step 5b (`subject_nll_pop_grad_analytical_iov`'s IOV section), and can be reused directly.

**Files:** Add Ω_iov packing/unpacking to the M-step NLopt setup in `saem.rs` (search for where `pack_params` / `unpack_params` is called in the M-step). Thread the `omega_iov` gradient through `subject_nll_pop_grad` (already handles IOV after Step 5b).

### Sub-task 11d — Remove the API block

In `api.rs:570-572`, remove the IOV+SAEM error. Add a warning to `FitResult.warnings` instead if `n_kappa > 0 && chain contains Saem` — warn that convergence may be slower with many occasions (Gibbs sampling overhead).

### Sub-task 11e — Ω_iov M-step update (analytic, optional)

The SAEM M-step for Ω_bsv uses the standard analytic update `Ω = (1/N) Σ η_i η_i^T` when `mu_referencing` is active. An equivalent analytic update for Ω_iov is:

```
Ω_iov = (1 / Σ_i K_i) * Σ_i Σ_k κ_{i,k} κ_{i,k}^T
```

where K_i is the number of occasions for subject i. This replaces the NLopt sub-problem for Ω_iov when `mu_referencing = true`, eliminating one optimizer call per SAEM iteration. Implement alongside the existing Ω_bsv analytic update in the M-step.

### Files to touch
- `src/estimation/saem.rs` — add kappa state, E-step sampling, M-step parameter extension, analytic Ω_iov update
- `src/api.rs` — remove hard block at lines 570-572; add soft warning
- `src/types.rs` — add kappa step-scale fields to SAEM state if exposed

### Tests

**Tier 1 (unit, `src/estimation/saem.rs`):**
- Test kappa MH acceptance/rejection logic on a synthetic 1-kappa 2-occasion subject
- Test analytic Ω_iov update: known kappa samples, assert update matches manual formula

**Tier 2 (integration, `tests/saem_iov.rs` — new file):**
- `fit()` on `examples/warfarin_iov.ferx` with `method = saem` and `outer_maxiter = 5`; assert no panic, finite OFV, no error return

**Tier 3 (slow, same file):**
- Full warfarin_iov SAEM convergence; assert OFV within 0.5 of the FOCE/FOCEI result (SAEM and FOCE can differ slightly)
- Assert acceptance rate for kappa proposals > 20% (not stuck)

### Expected gain

Enables population PK/PD models with between-occasion variability to use SAEM — currently the only supported IOV estimation path is FOCE/FoceI. SAEM+IOV is relevant for models with many occasions per subject where FOCE linearisation error accumulates. No wall-clock improvement over FOCE is expected; the gain is access to a different estimation algorithm where FOCE assumptions break down.

---

## Completion Checklist

After all steps, verify the project goals are met.

### Correctness (Accurate)

```bash
# Tier 1 unit tests — must pass on every PR
cargo test --lib

# Tier 2 integration compile check — must pass on every PR
cargo check --tests

# Tier 3 slow convergence tests — run nightly / manually
cargo test --features slow-tests

# Clippy — no new warnings
cargo clippy
```

All Tier 3 OFVs must agree with pre-improvement baselines to within 0.01 OFV units.
All standard errors must agree within 1% relative.

### Performance (Fast)

Benchmark wall-clock time for the same four examples, before and after:
```bash
time cargo run --release -- examples/warfarin.ferx --data data/warfarin.csv
```
Record times in a benchmark table committed to `docs/src/benchmarks.md`.
Expected improvement: 3–10× on FOCE/FOCEI models with ODE structure,
2–5× on analytical models, 3–5× on SAEM models.

### Efficiency

Count function evaluations for one representative FOCE fit and one SAEM fit
before and after. Both must drop substantially:
- FOCE outer evaluations: expect 50–80% reduction (AD gradients in Steps 3, 5, 6)
- SAEM total ODE evaluations: expect 50–70% reduction (HMC acceptance + AD M-step)

### Stability

Run each method on a deliberately ill-conditioned scenario:
- FOCE/FOCEI: `mm_oral.ferx` with bad initial estimates (TVCL × 10)
- GN / GN_hybrid: `two_cpt_oral_cov.ferx` with correlated covariates
- SAEM: `mm_oral.ferx` with full-block OMEGA

For each, confirm:
- The optimizer converges (no NaN, no failure to terminate)
- With `n_starts = 8`, the global OFV is consistently found
- No silent wrong answers — failure modes appear in `FitResult.warnings`

### Documentation

Update `docs/src/estimation/index.md` with a summary table covering, per method:
- Gradient source (AD or FD)
- Optimizer (NLopt SLSQP, trust region, etc.)
- Steihaug-CG budget policy (fixed or adaptive)
- Multi-start support
- HMC support (for SAEM)
- SIR proposal type

### Final Sign-Off

The work is complete when all four project goals (fast, efficient, stable,
accurate) are demonstrably advanced and documented. Speed gains without
accuracy preservation are not acceptable. Stability gains without efficiency
checks are not acceptable. All four must move in the right direction.
