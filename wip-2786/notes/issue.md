## Summary

ferx's exact derivatives come from a family of hand-written dual types (`DualF`, `DualV`, `DualT` (nested), `DualH`, `DualMixed`, `Taylor`), each compiled at fixed widths. This issue measures what that costs and proposes one replacement design:

- **runtime-width forward lanes** in place of compile-time widths;
- **one tape** for reverse mode;
- **exact per-engine derivative rules**;
- **compiled code only for user-defined parts.**

It builds on #2481. That audit found the *modes* correct: forward in η, reverse in θ, the directional exact anchor. This issue is about the *representation*: fixed widths and the number of types.

## Measurements

Release binary of the #2747 head; perf, one core, one run per model. Scripts and data are in `/scratch/mu/ad/`.

### Binary size

- **142 distinct concrete number types**. About **42 MB of ferx-core's 56 MB of code (75%)** exists only because of them.
- **The engines dominate, not user expressions:**

  | Module | MB |
  |---|---|
  | `sens::ode_provider` | 9.8 |
  | `sens::focei_kernel` | 3.7 |
  | `sens::propagate` | 2.9 |
  | `sens::three_cpt` | 2.8 |
  | `ode::linear` | 2.6 |
  | `sens::two_cpt` | 2.3 |
  | `parser::model_parser` (user expressions) | 1.7 |

- **A few large types dominate:** `DualT<4,DualT<24,R>>` 3.2 MB, `DualV<6,21>` 2.4 MB, `DualT<4,DualH<4,10,R>>` 2.1 MB. The 98 smallest types together are 3.1 MB.

### Fit time

| Fit | Derivative code, total | Dual-typed code | Tape sweeps | User expressions | Compiled RHS | Cranelift compiling |
|---|---|---|---|---|---|---|
| pk2 (closed form) | 39% | 24% | 15% | 8% | – | 0.00% |
| wiov (IOV, closed form) | 25% | 24% | 1% | 6% | – | 0.03% |
| pk2_ode (linear ODE) | 50% | 30% | 20% | 6% | 3% | 0.21% |
| pk2_ode_cov (same, plus covariance step) | 53% | 33% | 20% | 5% | 3% | 0.14% |
| mm (nonlinear ODE) | 19% | 19% | 0% | 4% | 24% | 0.01% |

- **Compiling generated code is not a cost:** Cranelift takes at most 0.21% of a fit.
- **The derivative cost sits in the engines** (closed forms, linear propagator, ODE provider, tape sweeps), not in user expressions.
- **wiov spends ~9% of its time copying `DualV<6,21>`** (28 f64 each) by value inside `propagate_one_cpt_oral_g`. Passing by reference was tried in #2747 and did not pay; the cost belongs to the fixed-width representation.

## Proposed design

1. **One forward derivative type, generic in order only.**
   - A value plus `n` directional derivatives, in contiguous lanes in per-thread scratch.
   - `n` is a runtime value, processed in fixed SIMD chunks with a masked remainder.
   - There are no width buckets and no padding to the next bucket.
   - Higher orders come from univariate Taylor propagation along directions (order 1–4 as a const generic).
   - The goal is about 4–5 types in place of 142, keeping the `PkNum` trait so every formula stays written once.
2. **One tape for reverse mode.**
   - `vtape` becomes the only tape. θ-gradients come from a reverse sweep.
   - Laplace/FOCEI `½ log det H` gradients come from:
     - a frozen `W = H⁻¹`;
     - directional order-2 Taylor passes;
     - one reverse sweep;
     - the implicit-function adjoint for `η̂` (`Hλ = ∂ψ/∂η`, then `−λᵀ ∂²ℓ/∂η∂θ`).

     No third-derivative tensor is formed.
3. **Engine-level derivative rules.** Each of these is one taped primitive with an exact rule, written once per order instead of per type × width:
   - the linear ODE segment;
   - the closed-form propagators;
   - the ODE adjoint;
   - the η̂ solve.
4. **Compiled code only for user-defined parts.** The nonlinear ODE RHS keeps `rhs_ir` → Cranelift, with exact-width code and the IR-level reverse pass (`ir::reverse`).

Rejected alternatives:

| Alternative | Why rejected |
|---|---|
| A tape for everything | Interpretive overhead on small closed-form kernels |
| Derivative generation on the RHS IR only | Misses the engines, where the cost is |
| Whole-program source transformation | Compile cost for no gain over exact engine rules |

## Plan

1. **Prototype:** the runtime-width forward type on the closed-form propagators (`sens::propagate`). Measure against the current duals on pk2 and wiov:
   - retired instructions (interleaved A/B);
   - binary size;
   - Dual-vs-current parity at round-off.

   **Pass bar:** no meaningful instruction regression on pk2/wiov, and a clear size reduction for the ported path. Small regressions on tiny models are acceptable only if explained and the design is general.
2. If it holds, port engine by engine behind `PkNum`, measuring each step. Old types are deleted when unused.
3. Then the objective level: the Laplace/FOCEI recipe above on the single tape.

Every step keeps f64 predictions bit-identical and carries derivative parity tests per CLAUDE.md.
