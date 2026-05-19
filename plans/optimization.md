# ferx-core Optimization Task Plan (Updated 2026-05-18)

This file is for use with Claude Code in the `ferx-core` repository.
Read `CLAUDE.md` first before starting any step.
Complete steps in order — later steps depend on earlier ones.
Each step specifies the files to touch and the expected outcome — but **the plan
below is a starting hypothesis, not a fixed recipe**. The actual repository state
may differ from what is described. Every step begins with deep evaluation of the
existing code, plan reconciliation, and explicit confirmation before any code is
written.

---

## Status Legend

- ✅ **DONE** — fully implemented on `main`.
- 🔶 **PARTIAL** — partially done; see the per-step note.
- 🔁 **IN PR** — addressed by open PR #22 (`perf/cross-engine-bench-fixes`); not yet on `main`.
- ❌ **NOT STARTED** — nothing relevant in `main` or open PRs.

---

## Important: Open PR #22 — Read Before Starting Steps 3–6

PR #22 (`perf/cross-engine-bench-fixes`) is open and **must be merged before
implementing Steps 3–6**. It touches `outer_optimizer.rs`, `parameterization.rs`,
`saem.rs`, and `types.rs` — the same files as Steps 3–6. Starting those steps on
the current `main` will produce merge conflicts and duplicate work.

PR #22 changes relevant to this plan:

- **`types.rs`**: default outer optimizer flipped from `Bobyqa` to `Slsqp`.
  Step 5 notes below are written for this post-PR22 default.
- **`parameterization.rs`**: adds identity packing for negative-lower-bound
  thetas (covariate exponents such as γ). Step 3 must handle the mixed-packing
  case — some thetas are log-packed, some are identity-packed — when assembling
  the population gradient vector. The function `theta_packs_log(theta_lower: f64)`
  exported from `parameterization.rs` is the gate.
- **`saem.rs`**: switches M-step gradient from central-FD to forward-FD and
  caches Ω⁻¹ — 2.4× single-thread speedup. Step 4 (AD M-step gradient) supersedes
  this change; verify the SAEM M-step call site after PR #22 merges before
  touching `saem.rs`.
- **`outer_optimizer.rs`**: adds a stagnation guard that short-circuits the
  NLopt loop when OFV improvement stalls. Step 5 must preserve this guard when
  wiring in the AD gradient.

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
3. Add tests as you go, not at the end. Every new function gets a unit test
   in the same file's `tests` module before the function is considered done.
4. Run `cargo clippy` — fix any warnings before moving on.
5. Run `cargo test --lib` — all tests must pass before considering the step
   complete.

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

**Status: ❌ NOT STARTED**

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

**Status: 🔶 PARTIAL**

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

**Status: 🔴 NOT STARTED**

**Requires: Step 5 complete.**

### Background

Step 5 replaced population-level central FD with `subject_nll_pop_grad` summed
in parallel over subjects. For non-IOV analytical PK models, `subject_nll_pop_grad`
takes the analytical path (`subject_nll_pop_grad_analytical`): exact for ω/σ
Cholesky elements, forward-FD of predictions only for θ. For IOV models,
`can_use_analytical = false` because `!kappas.is_empty()`, so it falls back to
per-subject central FD (cost = 2P subject NLL evals per outer gradient query).

The analytical gradient *can* be extended to IOV — the FOCE NLL structure is
identical, just with an expanded random-effects vector and a block-diagonal
variance matrix.

### What remains

#### Math

For a subject with occasions `1…K`, the IOV random effects are `[η; κ₁; …; κ_K]`
and the combined variance block is:

```
Ω_combined = diag(Ω_bsv, Ω_iov, …, Ω_iov)   (K+1 blocks)
```

The FOCE linearisation gives:

```
r_tilde = R + H_combined · Ω_combined · H_combined^T
```

where `H_combined = [H_η | H_κ₁ | … | H_κ_K]` concatenates the Jacobians
`∂ipred/∂η` and `∂ipred/∂κ_occ` for each occasion (most columns are zero
outside their occasion's observations).

The gradients follow the same formula as the non-IOV case:

```
∂NLL/∂L_bsv[i,j]  = ...  (same as non-IOV ∂NLL/∂L[i,j] for the bsv block)
∂NLL/∂L_iov[i,j]  = Σ_occ [same formula applied to the κ_occ block]
∂NLL/∂σ_k         = ...  (unchanged)
∂NLL/∂θ           = forward-FD of ipred (unchanged)
```

#### Implementation

1. Add `subject_nll_pop_grad_analytical_iov` in `src/estimation/gauss_newton.rs`
   (or extend `subject_nll_pop_grad_analytical` with an IOV branch).
2. In `subject_nll_pop_grad`, lift the `kappas.is_empty()` gate from
   `can_use_analytical` for non-ODE, non-M3 models.
3. The kappa H-matrices (`∂ipred/∂κ_occ`) must be available at the call site —
   verify they are returned by `run_inner_loop_warm` or add their computation.

### Files to touch
- `src/estimation/gauss_newton.rs` (new IOV analytical gradient variant)
- `src/estimation/outer_optimizer.rs` (lift `kappas.is_empty()` gate if needed)

### Test

Add a gradient test analogous to `test_outer_ad_gradient_block_omega` but with
`omega_iov` set and multiple occasions per subject. Verify that the IOV
analytical gradient matches population-level central FD to within `1e-4`.

Also verify that `test_outer_ad_gradient_fd_fallback_path` is superseded (or
update it to confirm the analytical path is now taken for non-ODE IOV).

### Expected gain

Same ratio as Step 5 for non-IOV: instead of 2P subject NLL evals per gradient
query, the cost drops to 1 forward-FD pass of predictions per θ component.
For IOV models with many occasions, the per-subject FD cost scales with P,
so the gain is proportional to P (number of packed population parameters).

---

## Step 6 — Trust Region: AD Gradient + BHHH Hessian + Adaptive Steihaug-CG

**Status: 🔶 PARTIAL**

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

## Step 7 — Natural Gradient: BHHH Hessian Inside the GN Trust-Region Step

**Status: ❌ NOT STARTED**

**Requires: Steps 3 and 6 complete.**

### What to do

Currently `method = gn` uses BHHH + LM damping + backtracking line search
(in `gauss_newton.rs`). `trust_region.rs` will now have an Steihaug-CG
trust-region subproblem solver with BHHH Hessian and AD gradient (from Steps 3
and 6).

Upgrade the GN method to use the trust-region subproblem solver instead of
LM damping and line search:

1. In `estimation/gauss_newton.rs`, after computing `H_BHHH` and `∇OFV`, call
   into the trust-region subproblem solver (expose it as a `pub fn` from
   `trust_region.rs`) to compute the step `δ`.

2. Replace the LM damping adaptation with the standard trust-region ratio update:
   ```
   ρ = (OFV_current - OFV_proposed) / (predicted_reduction_from_quadratic_model)
   ```
   - ρ > 0.75: expand trust radius (×2, up to Δ_max)
   - 0.25 < ρ ≤ 0.75: keep radius unchanged
   - ρ ≤ 0.25: shrink radius (÷4), reject step, retry

3. Remove the backtracking line search. The trust-region step is accepted or the
   radius shrinks — no line search is needed or appropriate.

The adaptive Steihaug-CG budget from Step 6c applies automatically here because
the same subproblem solver is reused.

### Files to touch
- `src/estimation/gauss_newton.rs` (replace LM + line search with TR subproblem call)
- `src/estimation/trust_region.rs` (expose subproblem solver as a reusable `pub fn`)

### Test
Run warfarin with `method = gn`. Compare outer iteration count against the LM version.
OFV must match LM baseline to within 0.01.

Run `two_cpt_oral_cov.ferx` with weight and creatinine clearance covariates.
Compare iteration counts against `method = focei` and the old `method = gn`.

### Expected gain
The LM damping factor is a scalar approximation to the trust radius — it treats
all parameter directions equally. The trust-region step adapts differently per
eigendirection of `H_BHHH`. For covariate models with weakly identified covariate
coefficients, this gives fewer wasted steps in the weakly-identified directions.

---

## Step 8 — HMC Proposals in the SAEM E-Step

**Status: ❌ NOT STARTED**

**Requires: Steps 3 and 4 complete.**

### Current state

`src/estimation/saem.rs` uses a Metropolis-Hastings (MH) random-walk E-step
(`mh_steps`, line 66). The per-subject adaptive step size (`step_scales`) is
already tracked and adapted per subject. No HMC code exists.

The ETA gradient needed for HMC leapfrog is `∇_η NLLᵢ(η | θ, Ω, σ)` —
provided by `compute_nll_gradient_ad` from `src/ad/ad_gradients.rs`. Confirm
this function works with the current subject data layout before proceeding
(check `FlatDoseData::from_subject` compatibility with the saem.rs subject
representation).

### What to do

Add a `leapfrog` function in `saem.rs` (or `src/estimation/hmc.rs`):

```rust
fn leapfrog(
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

Replace the MH random-walk proposal with HMC per subject:
1. Sample fresh momentum `p ~ N(0, Ω⁻¹)` — mass matrix = inverse OMEGA.
2. Run `leapfrog` for `saem_n_leapfrog` steps using `compute_nll_gradient_ad`.
3. Accept/reject: `H(q, p) = NLLᵢ(q) + ½ pᵀ Ω p`; accept with probability
   `min(1, exp(H_current - H_proposed))`.

The per-subject step size (`step_scales[i]`) becomes the HMC leapfrog step
size. Keep the adaptation logic but target 65% acceptance instead of ~40%.

Add `saem_n_leapfrog: usize` to `FitOptions` (default: 3). Document it.

Keep the existing MH sampler available as a fallback: if the AD ETA gradient
is unavailable for the current model configuration (e.g. unsupported PK model
in the Enzyme path), fall back to MH and emit a warning in `FitResult.warnings`.

### Files to touch
- `src/estimation/saem.rs`
- `src/types.rs` (add `saem_n_leapfrog: usize`)
- `docs/src/estimation/saem.md`
- `docs/src/model-file/fit-options.md`

### Test
Run warfarin SAEM 5 times with different seeds. Compare:
- Mean acceptance rate: should be ~65%, not ~40%
- Final parameter estimates: must agree with MH version to within 2%

Run `mm_oral.ferx` with full-block OMEGA (most correlated ETA posterior).

### Expected gain
At 40% acceptance, 60% of proposals are wasted ODE evaluations. HMC at
65–90% acceptance wastes far less. Effective sample size per E-step is 3–5×
larger. The exploration phase stabilizes faster.

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

## Completion Checklist

After all steps, verify the project goals are met.

### Correctness (Accurate)

Run the full example suite:
```bash
cargo run --release -- examples/warfarin.ferx --data data/warfarin.csv
cargo run --release -- examples/two_cpt_iv.ferx --data data/warfarin.csv
cargo run --release -- examples/two_cpt_oral_cov.ferx --data data/warfarin.csv
cargo run --release -- examples/mm_oral.ferx --data data/warfarin.csv
```
All OFVs must agree with pre-improvement baseline to within 0.01 OFV units.
All standard errors must agree within 1% relative.
Run `cargo test --lib` — all tests must pass.
Run `cargo clippy` — no new warnings.

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
