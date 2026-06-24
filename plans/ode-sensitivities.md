# Plan: Analytic PK-parameter sensitivities for user-specified ODE models (Option A)

**Tracking issue:** [#367](https://github.com/FeRx-NLME/ferx-core/issues/367) (NONMEM-class
analytic FOCE/FOCEI outer gradient; Almquist, Leander & Jirstrand 2015, PMC4432110)
**Scope:** ferx-core (primary) + ferx-r (follow-up once any `pub` API lands)
**Status:** design approved (Option A chosen 2026-06-17); Phase 1 in progress. Multi-PR / phased.
**Builds on:** the closed-form analytic provider (`src/sens/`, `provider.rs`) already shipping
exact `∂f/∂η, ∂²f/∂η², ∂f/∂θ, ∂²f/∂η∂θ` for 1-/2-/3-cpt analytical PK.

---

## Context

The analytic provider gives the FOCEI outer gradient exact PK-parameter sensitivities for the
**closed-form** PK models. User-specified `[odes]` models are excluded: their right-hand side is
a monomorphic `f64` closure — `OdeSpec.rhs: Box<dyn Fn(&[f64], &[f64], f64, &mut [f64])>`
(`src/ode/predictions.rs:511`) — which cannot carry derivatives. So ODE fits fall back to the
gradient-free path.

Underneath that closure, though, the `[odes]` block is already lowered to a **flat bytecode
stack VM**: the `Expression` AST → `compile_bytecode` → `Vec<Op>` → `eval_bytecode` over an
`f64` stack (`src/parser/model_parser.rs:6898`). The bytecode is identical in shape for every
user model. That is the single leverage point.

## The two options (recorded for the decision trail)

- **Option A — `Dual2` over the bytecode VM (CHOSEN).** Make `eval_bytecode` generic over
  `T: PkNum`, integrate the ODE state as `Dual2<N>` seeded on the PK parameters, and let the
  RK45 integrator (pure arithmetic) propagate `∂u/∂p` and `∂²u/∂p²` through every stage. The
  readout then yields `∂f/∂p, ∂²f/∂p²`, fed into the **same** η/θ chain assembly the analytic
  provider already uses. General for *any* user ODE with **zero per-model code**; exact;
  smallest surface. Cost: `O(N²)` per op (dense Hessian) in the VM.
- **Option B — symbolic-diff codegen of the augmented sensitivity RHS.** Symbolically
  differentiate the AST to emit new bytecode for the forward-sensitivity equations
  (`dS/dt = J·S + ∂rhs/∂p`, plus the second-order system), integrated in plain f64. Faster per
  step but a large, error-prone, per-model symbolic engine. **Deferred** — treated as an
  optional later optimization gated on profiling, exactly as the hand-written explicit kernels
  are to the generic `Dual2` analytic path. The empirical lesson from the analytic work
  (generic `Dual2` is the right baseline; right-sizing the dual width is the free win;
  hand/symbolic derivatives buy diminishing returns) carries over and is *stronger* for
  arbitrary user ODEs.

## Goals / non-goals

- **Goal:** exact first/second PK-parameter sensitivities for user `[odes]` models, reusing the
  existing provider η/θ chain and outer-gradient assembly — one VM generalization covers all.
- **Goal:** no behavioural change to prediction (`T = f64`) or to analytical-model fits.
- **Non-goal (initially):** non-smooth RHS niceties — `if`/`min`/`max`/`floor` in a user RHS
  are piecewise-constant in the derivative (measure-zero kinks ignored), the same contract the
  analytic `.max`/`.min`-on-value code already honours. Document, don't solve.
- **Non-goal (initially):** IOV, multiple endpoints, output transforms (scaling/LTBS),
  time-varying covariates — mirror the analytic provider's scope gate, widen later.
- **Non-goal:** Option B symbolic codegen (separate future plan if profiling demands it).

## Design

### 1. Numeric trait gap (Phase 1)
`eval_bytecode` uses these `Op`s. Split by smoothness:

- **Smooth (need `Dual2` rules):** `Add Sub Mul Div Pow Exp Ln Sqrt Abs InvLogit Logit`.
  `Dual2` already has `+ − × ÷ exp sqrt`. **Add:** `ln`, `pow` (binary `aᵇ`), `abs`,
  `inv_logit`, `logit`. `pow`/`logit`/`inv_logit` compose from `ln`/`exp`/`recip`; `abs` is a
  value-branch (`|x|` with `sign(x)` first deriv, zero second away from the kink); add a
  constant-integer-exponent fast path to `pow` so `x²` etc. work for any base sign.
- **Value-based (no trait change):** `Cmp* Logic* Jump* Mod Floor Ceil Round` operate on
  `.val()` and push `T::from_f64(0/1/…)` — piecewise-constant, derivative zero. Handled inside
  the generic VM, not in `PkNum`.

Extend `PkNum` (and its `f64` + `Dual2<N>` impls) with `ln, pow, abs, inv_logit, logit`. Each
validated against central finite differences of value (grad) and of grad (Hessian).

### 2. Generic VM (Phase 2)
Generalize `eval_bytecode` to `eval_bytecode<T: PkNum>(bc, stack: &mut Vec<T>, …)`. `f64`
monomorphization must be byte-identical to today (guard with the existing AST↔bytecode
equivalence test, extended to assert `Dual2<N>.val()` equals the `f64` path). Constants lift via
`T::from_f64`; theta/eta/cov/var inputs lift as constants *except* the seeded PK parameters.

### 3. `Dual2`-state integrator + augmented plumbing (Phase 3)
Make the RK45 stepper generic over the state scalar (or add a `Dual2`-state entry point).
**Step-size control reads `.val()` only** — adapt by the value; the derivative then flows
through a fixed step sequence (correct). Seed the PK params as `Dual2::var`; dose injection
carries `amt·F` (so bioavailability `F`, a PK param, seeds its derivative at the event);
`init_fn` expressions evaluate over `Dual2`; EVID 3/4 resets re-apply `init` over `Dual2`.

### 4. ODE sensitivity provider (Phase 4)
New path (sibling to `subject_sensitivities`): integrate the `Dual2<N>` state, apply the
readout to get `∂f/∂p, ∂²f/∂p²` per observation, then reuse the **existing** Term/η/θ chain
verbatim. An `ode_analytical_supported(model)` gate mirroring `analytical_supported`
(single endpoint, log-normal η, no IOV, no output transform, supported readout).

### 5. Wire + validate + right-size (Phase 5)
Dispatch ODE-supported models into the new provider from `population_gradient`. Validate the
provider against central FD of `compute_predictions_with_tv` on an ODE model (e.g. the
`transit_2cpt` example), and the end-to-end gradient against FD, with a NONMEM OFV/estimate
anchor. **Right-size `N`** to the count of PK parameters the model actually reads. Docs
(`docs/estimation/optimizers.qmd`, faq) + CHANGELOG.

## Phases (each lands with tests + anchor)

1. **Dual2/PkNum smooth-op gap** — `ln, pow, abs, inv_logit, logit` + FD unit tests. ← *here*
2. **Generic `eval_bytecode<T>`** — + extended AST↔bytecode↔`Dual2` equivalence test.
3. **`Dual2`-state RK45 + augmented state** — dose/init/reset seeding; value-based step control.
4. **ODE sensitivity provider** — readout → `∂f/∂p` → existing η/θ chain; ODE scope gate.
5. **Wire into outer gradient + validate vs FD/NONMEM + right-size N** — docs + CHANGELOG.

## Risks

- **Cost.** Dense `Dual2<N>` through every RK stage is the known price. Mitigation: right-size
  `N`; the gradient path replaces *reconverged-FD*, which is far more expensive, so the bar is
  low. Profile before reaching for Option B.
- **Non-smooth RHS.** `if`/`min`/`max` give correct values but kinked derivatives. Acceptable
  (documented); the gradient is exact wherever the RHS is locally smooth.
- **Step-control coupling.** Controlling steps on `.val()` is essential — adapting on a
  derivative norm would make the sensitivity inconsistent with the prediction.
