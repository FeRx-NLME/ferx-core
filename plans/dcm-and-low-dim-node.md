# Deep Compartment Models in ferx-core

## Context

Janssen et al. 2022 (CPT:PSP, DOI 10.1002/psp4.12808) introduced **deep compartment models (DCM)**: a neural network maps subject covariates → PK parameters (CL, V, Q, …) which then feed a standard compartmental ODE/closed-form. The original paper is fixed-effects only, MSE loss, no etas — the reference implementation is `DeepCompartmentModels.jl` (Lux + DifferentialEquations.jl, example: 2 covariates → 16 hidden → 4 PK params, ~320 weights).

A pure fixed-effects port would mostly *bypass* ferx-core's FOCE/FOCEI machinery — the engine's reason for existing. The interesting variant for this codebase is **mixed-effects DCM**: NN provides the typical values, etas/omega/sigma stay, FOCEI fits the lot. The user's `[individual_parameters]` block becomes

```
[CL, V1, Q, V2, KA] = nn(WT, CRCL; layers=[16, 16]) .* exp([ETA_CL, ETA_V1, ETA_Q, ETA_V2, ETA_KA])
```

This is a near-drop-in extension because the existing extension point — `CompiledModel.pk_param_fn: Fn(&[f64], &[f64], &HashMap<String, f64>) -> PkParams` (src/types.rs:453) — already takes covariates as a name-indexed map and returns PK params. Replacing the closure body with an NN forward pass is sufficient; every downstream call site (likelihood, inner BFGS on etas, ODE setup, predictions, output) is unchanged.

User decisions: **mixed-effects DCM**, **in-package behind a feature flag**, **NN backend deferred** (design around a trait so a hand-rolled MLP or `candle`/`burn` can swap in later).

The main feasibility blocker isn't architectural — it's the **outer-loop gradient**. Outer gradients today are central finite differences (cost = 2·n_params NLL evals per gradient; see `outer_optimizer.rs:1287–1346` and `trust_region.rs:64–89`). Adding 320 NN weights → ~640 NLL evals per gradient step. Tractable for an MVP, painful at production scale. No outer-loop AD path exists; Enzyme AD is wired only through the *inner* (individual NLL w.r.t. eta) path (see `src/ad/ad_gradients.rs`, `inner_optimizer.rs:34–46`).

## Recommendation

Do it **in ferx-core behind a `nn` cargo feature**, not in a separate crate. The parser, NONMEM CSV I/O, analytical/ODE PK, parameterization, FOCEI inner loop, and result/output code are all reusable as-is — a sister crate would re-export or duplicate most of `lib.rs`. Feature-flagging keeps the dep surface (and any heavy ML crate, if adopted) opt-in. Ship in milestones; the first one is small enough that going down this path is reversible.

## Approach

### M1 — DSL + hand-rolled MLP, fixed-effects sanity check (≈ 1–2 weeks)

Goal: match the paper end-to-end on a published dataset; prove the plumbing.

1. **`nn` cargo feature** in [Cargo.toml](Cargo.toml). No new runtime deps for MVP — implement the MLP on `nalgebra` matmuls (the codebase already depends on it).
2. **DSL extension** in [src/parser/model_parser.rs](src/parser/model_parser.rs): new `[covariate_nn NAME]` block, e.g.
   ```
   [covariate_nn TYPICAL_PK]
     inputs  = [WT, CRCL]
     outputs = [CL, V1, Q, V2, KA]
     layers  = [16, 16]
     activation = softplus     # outputs are positive PK params; softplus or exp on the head
   ```
   Referenced from `[individual_parameters]` by name (e.g. `TYPICAL_PK.CL`) to consume the NN outputs and (later) compose with etas. The block name makes the role explicit at declaration site — paired with Phase B's `[dynamics_nn NAME]` so the parser can produce precise errors when the wrong block type is referenced in the wrong place. Parsing produces a `NeuralNetSpec` (shape + activations) stored on `CompiledModel` next to `pk_param_fn`.
3. **`CovariateMapper` trait** (new file, `src/nn/mod.rs`): the abstraction the rest of the engine talks to.
   ```rust
   pub trait CovariateMapper: Send + Sync {
       fn n_weights(&self) -> usize;
       fn forward(&self, weights: &[f64], covariates: &HashMap<String, f64>, out: &mut PkParams);
       fn jacobian(&self, weights: &[f64], covariates: &HashMap<String, f64>, jac: &mut [f64]); // optional, FD fallback otherwise
   }
   ```
   First impl: `MlpMapper` with manual forward + analytical backprop (Jacobian of outputs w.r.t. weights — standard MLP gradient, ~150 LOC).
4. **`build_pk_param_fn` branch** in [src/parser/model_parser.rs:2135](src/parser/model_parser.rs:2135): when a `[covariate_nn]` block is referenced from `[individual_parameters]`, the returned closure dispatches to the mapper instead of the expression evaluator. Same signature, same call sites — no changes downstream in `src/pk/mod.rs`, `src/api.rs`, etc.
5. **Parameterization** in [src/estimation/parameterization.rs](src/estimation/parameterization.rs): extend the pack/unpack layout to append NN weights after thetas. NN weights are unbounded (no log transform), unlike thetas. Add a `n_nn_weights` field on `ModelParameters`/`CompiledModel` and update `compute_scale` to scale-1.0 the weight block by default.
6. **MSE objective** (feature-flagged): in [src/api.rs](src/api.rs) `fit()`, when `method = nn_mse` is set in `[fit_options]`, the population NLL routine is replaced by Σ residual² (no omega, no sigma, etas forced to zero). This is the path that reproduces the paper.
7. **Outer optimizer**: use the existing built-in BFGS or NLopt LBFGS with FD gradients. No new optimizer in M1.
8. **Tests** (CLAUDE.md mandates one per feature) in `tests/` modules:
   - Unit: `MlpMapper::forward` matches a hand-set output for known weights.
   - Unit: `MlpMapper::jacobian` matches central FD to 1e-6.
   - Unit: parser rejects mismatched output count between `[covariate_nn]` outputs and `[structural_model]` PK params.
   - Unit: parser rejects a `[covariate_nn]` reference from inside `[odes]` with an error suggesting `[dynamics_nn]` (and the reverse).
   - Regression: a NN with zero hidden weights and a learned bias reproduces a flat typical-value fit on warfarin.
9. **Docs**: add `docs/src/model-file/covariate-nn.md` and a shared landing page `docs/src/model-file/neural-networks.md` (decision tree: "known PK + uncertain covariate model → covariate_nn; uncertain dynamics → dynamics_nn"). Link both from `docs/src/SUMMARY.md`; rebuild `docs/book/` per CLAUDE.md.

### M2 — Mixed-effects DCM (the production target, ≈ 1–2 weeks on top of M1)

Goal: NN provides typical values; FOCEI fits NN weights + omega + sigma + etas jointly.

1. **Eta composition**: extend the `[individual_parameters]` DSL so the NN output composes with etas exactly like today's mu-referenced analytical form:
   ```
   CL = TYPICAL_PK.CL * exp(ETA_CL)
   V1 = TYPICAL_PK.V1 * exp(ETA_V1)
   ```
   In code, the new `pk_param_fn` closure runs NN forward, then multiplies in `exp(eta[i])` per output. Implementation note: [`detect_pattern`](src/parser/model_parser.rs#L186) currently matches Pattern 1 (`THETA * exp(ETA)`) by walking the multiplicative chain for `Expression::Theta` nodes. NN-output access (`TYPICAL_PK.CL`) is a new AST node, so `detect_pattern` needs a small extension to treat that node as a theta-equivalent anchor — ~10 LOC, plus a unit test under `model_parser.rs::tests`. Once recognized, the existing inner-loop AD path (which differentiates `tv * exp(eta)` w.r.t. eta — see `src/ad/ad_gradients.rs:79`) handles the rest. The single most important property still holds: the inner FOCEI loop itself is unchanged because the NN is "upstream" of the eta application.
2. **`tv_fn` parity**: the parser auto-generates a `tv_fn` (typical-value function with eta=0) around [src/parser/model_parser.rs:844–849](src/parser/model_parser.rs:844). Add the NN-aware equivalent: call `MlpMapper::forward` and skip the eta multiplication.
3. **Outer gradient strategy** (the hard call):
   - **Default**: central FD on the full parameter vector, including NN weights. Cost is 2·(n_theta + n_weights) NLL evals per gradient. With the paper's ~320 weights and ferx-core's fast analytical PK, this is workable for proof-of-concept (≈ low-seconds per gradient step on warfarin-sized data).
   - **Upgrade path** (documented but not built in M2): hand-derive the Jacobian of the population NLL w.r.t. NN weights by composing (a) `MlpMapper::jacobian` (already in M1) with (b) ∂NLL/∂(typical values) via the chain rule. The latter is what the existing inner-loop AD path computes for etas; the same vector-Jacobian product applies to typical values. ~1 week of work, deferred to M3.
4. **Tests**:
   - Regression: warfarin two-cpt oral fit (`examples/two_cpt_oral_cov.ferx`) reproduced within 1% by a NN-DCM model whose layers collapse to the analytical covariate form (weights initialized so `nn(WT, CRCL) ≈ TVCL·(WT/70)^θ_WT·(CRCL/100)^θ_CRCL`).
   - End-to-end: simulate from a known NN, fit, recover NN weights within tolerance.
5. **Example**: `examples/warfarin_nn.ferx` — the workhorse for the docs page and regression test.

### M3 — Scale (optional, post-MVP)

Triggers only if users hit FD-gradient cost limits.

- Outer-loop AD via the chain-rule composition described in M2.3.
- Optional Adam/SGD outer optimizer for >1k NN weights (new file `src/estimation/adam.rs`). The existing `OuterOptimizer` enum gates this cleanly.
- Optional `candle` or `burn` backend behind a sub-feature (`nn-candle`, `nn-burn`) implementing the `CovariateMapper` trait. Hand-rolled MLP stays as the default no-extra-deps path.

## Critical files

| File | Change |
|------|--------|
| [Cargo.toml](Cargo.toml) | add `nn` feature |
| [src/lib.rs](src/lib.rs) | `#[cfg(feature = "nn")] mod nn;` |
| **NEW** `src/nn/mod.rs` | `CovariateMapper` trait + `MlpMapper` impl (forward + analytical Jacobian) |
| [src/types.rs:453](src/types.rs:453) | add `nn_spec: Option<NeuralNetSpec>` and `nn_weights_offset: usize` on `CompiledModel`; keep `PkParamFn` signature untouched |
| [src/parser/model_parser.rs:2135](src/parser/model_parser.rs:2135) | dispatch `build_pk_param_fn` to NN closure when `[covariate_nn]` is referenced from `[individual_parameters]` |
| [src/parser/model_parser.rs:844](src/parser/model_parser.rs:844) | NN-aware `tv_fn` |
| [src/estimation/parameterization.rs](src/estimation/parameterization.rs) | pack/unpack NN weights after thetas; unscaled bounds-free block |
| [src/api.rs](src/api.rs) | `method = nn_mse` shortcut for fixed-effects sanity check |
| `examples/warfarin_nn.ferx` | new example, M2 deliverable |
| `docs/src/model-file/covariate-nn.md` + `docs/src/model-file/neural-networks.md` (landing) + `docs/src/SUMMARY.md` | docs (CLAUDE.md requirement) |
| `docs/_site/` | rebuilt via `quarto render docs` (git-ignored, never committed) |

## Reusing what's already there

- `CompiledModel.pk_param_fn` (src/types.rs:453) — the entire integration point. No new abstraction needed at the call sites.
- `Subject.covariates: HashMap<String, f64>` and `subject.dose_cov(k)`/`obs_cov(j)` (src/types.rs:170, 216) — NN inputs come from here for free, including time-varying covariates.
- `src/ad/ad_gradients.rs` inner-loop AD over `tv * exp(eta)` — works unchanged for mixed-effects DCM because the NN is upstream of the eta multiplication.
- `src/estimation/inner_optimizer.rs` BFGS/Nelder-Mead on etas — does not care how typical values are produced; no edits needed.
- `src/estimation/parameterization.rs` `compute_scale` (lines 394–410) — already generic over parameter count; extending to append a weight block is a pure addition.

## Verification

End-to-end checks, in order:

1. `cargo build --features nn` and `cargo clippy --features nn` — no warnings.
2. `cargo test --features nn` — all new unit tests pass; existing tests still pass with default features.
3. **Sanity reproduction** (M1 deliverable): `cargo run --release -p ferx-cli --features ferx-core/nn -- examples/warfarin_nn_fixed.ferx --data data/warfarin.csv` with `method = nn_mse` produces NN weights whose forward pass on the training covariates matches the analytical typical values from `examples/two_cpt_oral_cov.ferx` within 5% RMSE.
4. **Mixed-effects reproduction** (M2 deliverable): the same `warfarin_nn.ferx` with `method = focei` reaches an OFV within 1 unit of the analytical model. The fit YAML's eta shrinkage and omega estimates should match the analytical fit to within numerical noise — strong evidence the NN layer is the only thing changed.
5. **Roundtrip**: simulate 100 subjects from a known NN+omega+sigma, fit, recover NN weights within 5% and omega within 10%.
6. **Docs build**: `quarto render docs` succeeds and the new page renders.

## Open questions worth flagging during M1 implementation

- **Bounds on NN outputs**: PK params must be positive. Either softplus/exp head on the NN, or unconstrained outputs + post-exp. Decide and document.
- **Covariate normalization**: NN training benefits from input scaling. Either auto-normalize at parse time (mean/std from data) or require users to pre-scale in the CSV. Auto-normalization is friendlier but adds state to `CompiledModel`.
- **Initialization**: zero weights → flat NN output → no gradient signal. Need a sensible default (Glorot/He) and a way to warm-start from an analytical-model fit (M3).

---

# Phase B addendum — Low-dimensional Neural ODEs (Bräm 2025)

## Context

Bräm et al. 2025 (CPT:PSP 14:5–16, DOI 10.1002/psp4.13265) demonstrates that "low-dimensional NODEs" — single-hidden-layer ReLU networks with ~5 hidden units, ≤62 total parameters per model — can be fit by standard FOCEI/SAEM through the existing ODE machinery of NONMEM and Monolix, with no backprop and no adjoint method. The NNs sit on the **right-hand side of the ODE**, replacing mechanistic terms like Michaelis–Menten or linear elimination when the dynamics are unknown. Etas attach to the NN weights and biases themselves — additively in Monolix, multiplicatively (with sign preservation) in NONMEM.

This is a different feature from DCM and a different integration point in ferx-core:

- **DCM (Phase A)** replaces `CompiledModel.pk_param_fn` — NN maps covariates → PK params, mechanistic ODE downstream unchanged.
- **Low-dim NODE (Phase B)** lives inside the `[odes]` block — NN expressions appear in `d/dt(...)` RHS terms, the rest of the engine (FOCEI inner loop, ODE solver, etas, sigma) is untouched.

The two are complementary, not alternatives. They share the `nn` cargo feature and the `MlpMapper` from Phase A; Phase B reuses that forward-pass code from inside the ODE expression evaluator instead of from `pk_param_fn`.

Bräm explicitly acknowledges that scaling beyond ~5 units would need "more tailored ML methods such as backpropagation" — so Phase B is genuinely scoped to *low-dim* NODE. Anything bigger lands in Phase A's M3 (outer-loop AD).

## Recommendation

Build Phase B **after Phase A M1 lands** (so the NN module + cargo feature already exist), but **before Phase A M3** (so the outer-loop AD upgrade benefits both features). Estimated effort: ~2 weeks on top of Phase A M1, mostly in the parser and a new initializer.

## What ferx-core already has for free

The Bräm method is unusually well-aligned with what's in ferx-core today:

- **`[odes]` block + Dormand-Prince RK45** in [src/ode/solver.rs](src/ode/solver.rs) and the parser path producing `ODE` structural models (see `examples/bioavailability_ode.ferx`). Bräm runs entirely on this.
- **Additive eta pattern** (`THETA + ETA`) is *already detected* by `detect_mu_refs` at [src/parser/model_parser.rs:213](src/parser/model_parser.rs:213) — `EtaParamType::Additive`. NN weights with signed normally-distributed etas drop in as ordinary additive-eta parameters; no eta-plumbing changes needed.
- **Per-parameter omega** — ferx etas attach to any named parameter, so `omega ETA_W_1_1 ~ 0.01` on a NN weight is already valid syntax.
- **FOCEI inner loop** in [src/estimation/inner_optimizer.rs](src/estimation/inner_optimizer.rs) does not care whether parameters are PK or NN-weight; it just optimizes etas against the individual NLL.
- **`ad/` Enzyme path** already differentiates through the ODE solver (that's what it does today for analytical etas in `tv * exp(eta)` form). For additive etas on NN weights, the same forward-mode AD applies — the chain rule terminates inside the ODE RHS evaluator regardless of how complicated the expression is.

The CLAUDE.md note about avoiding `f64::max()` / `f64::min()` in AD-instrumented code is directly relevant: Bräm hits the same NONMEM-side issue and implements ReLU as `if x > 0 { x } else { 0.0 }`. That's exactly the form ferx-core already mandates — so the AD-safety story for ReLU is solved by construction.

## What needs adding

### B1 — `nn(...)` function in the ODE expression evaluator (≈ 4–5 days)

Goal: let users write NN forward passes inside `d/dt(...)` expressions.

Proposed syntax — separate `[dynamics_nn NAME]` blocks defining shape, used by name from `[odes]`:

```
[dynamics_nn ka_nn]
  inputs     = [depot]
  hidden     = 5
  activation = relu
  iiv        = additive       # block-level: declares additive etas on every weight & bias

[dynamics_nn cl_nn]
  inputs     = [central]
  hidden     = 5
  activation = relu
  iiv        = additive

[odes]
  d/dt(depot)   = -ka_nn(depot)
  d/dt(central) =  ka_nn(depot) - cl_nn(central) * central
```

The `[dynamics_nn]` block name (as opposed to Phase A's `[covariate_nn]`) makes the role explicit at declaration site: the parser knows immediately that this NN belongs on an ODE RHS and rejects references from `[individual_parameters]` with a precise error. The block-level `iiv = additive` field auto-generates additive etas on every weight and bias — Bräm's IIV pattern — without forcing the user to hand-write dozens of `omega ETA_W_..._...` lines.

Implementation:
1. Parse `[dynamics_nn NAME]` blocks alongside the existing `[parameters]` block in [src/parser/model_parser.rs](src/parser/model_parser.rs). Each block declares `n_hid` and `inputs.len()`; the parser allocates `2*n_hid + n_hid + 1` named thetas (`W_NAME_1_1, …, B_NAME_OUT`) and registers them as if the user had written them by hand. When `iiv = additive` is set, the parser also auto-emits matching `omega ETA_W_..._... ~ <small>` declarations and the `THETA + ETA` mu-ref pattern (already handled by [model_parser.rs:213](src/parser/model_parser.rs:213)).
2. Add an `Nn { name: String, args: Vec<Expr> }` variant to the ODE expression AST. The evaluator looks up the network's weights from the current theta vector and runs `MlpMapper::forward_scalar` (a scalar-output variant of the Phase A `MlpMapper`).
3. Time-dependent NN trick from the paper: support an `inputs = [TIME]` form with `time_dependent = true` that reparameterizes input weights as `w' = -w_raw²` to enforce monotone decay. ~20 LOC.

The `MlpMapper` from Phase A is reused for the forward pass. The only new code is the AST integration and the time-dependent reparameterization.

### B2 — Activation-aware initializer (≈ 2 days)

Goal: fit doesn't fail because all ReLU units start deactivated.

Bräm's R-script samples `x_act` uniformly in `[x_min, x_max]` per unit and sets the bias so each unit activates at `x_act`. ferx-core port:

1. Add a `--init-nodes` CLI flag in [src/bin/](src/bin/) (or auto-trigger when any `[dynamics_nn]` block is present and the user did not provide explicit initial weights). DCM-style `[covariate_nn]` blocks have a different initialization story (Glorot/He) and are not affected by this flag.
2. Pre-fit pass: scan the dataset for the dynamic range of each NN's input column (e.g. `min/max` of `central` concentration across the dataset). Sample activation points; compute `b_1_i = -w_1_i * x_act_i`. Output weights and biases get random `[-0.3, 0.3]` initial values.
3. Persist sampled initial values to the fit YAML so runs are reproducible.

This lives in a new file `src/nn/init.rs` (under the `nn` feature gate).

### B3 — Two-step fit workflow (≈ 1 day)

Goal: support Bräm's "fit without IIV → fit with IIV" pattern as a single ferx invocation.

Add a `[fit_options]` key `two_step_nodes = true`. When set:
1. ferx runs the full fit with all NN-weight etas internally pinned at zero (`FIX 0` semantics).
2. Then re-fits using the step-1 estimates as warm start, with the eta variances unpinned.

This is purely orchestration in [src/api.rs](src/api.rs) `fit()` — both phases reuse the existing FOCEI loop unchanged.

### B4 — Tests + docs + example (≈ 2 days)

CLAUDE.md mandates one test per feature.

- Unit: parser accepts `[dynamics_nn]` block, allocates correct theta count, `iiv = additive` auto-emits matching omegas, mu-ref detection treats NN weights as additive-eta parameters.
- Unit: parser rejects `[dynamics_nn]` referenced from `[individual_parameters]` with an error pointing to `[covariate_nn]`, and the reverse.
- Unit: `nn(...)` inside `[odes]` evaluates to the same number as a hand-written `if`-then-else ReLU expression for the same weights.
- Unit: initializer produces activation points uniformly in the data range across multiple seeds.
- Regression: theophylline (`data/theophylline.csv` — add if missing) fit with a Bräm-style 2-cpt NODE recovers concentrations with MSE within 20% of the analytical two-cpt PO fit.
- New example: `examples/theophylline_node.ferx` mirroring the paper's first demo dataset.
- Docs: `docs/src/model-file/dynamics-nn.md` + add to the landing page `docs/src/model-file/neural-networks.md` introduced in Phase A and to `docs/src/SUMMARY.md`; rebuild `docs/book/`.

## Critical files

| File | Change |
|------|--------|
| [src/parser/model_parser.rs](src/parser/model_parser.rs) | parse `[dynamics_nn NAME]` blocks; add `Nn` variant to ODE AST; auto-declare NN weight thetas and (when `iiv = additive`) matching omegas |
| [src/ode/solver.rs](src/ode/solver.rs) (or expression evaluator if separate) | dispatch `Nn` AST node to `MlpMapper::forward_scalar` |
| [src/nn/mod.rs](src/nn/mod.rs) | add `forward_scalar` for single-output use from ODE block (Phase A's `MlpMapper` produces a `PkParams`; this needs a `&[f64] -> f64` variant) |
| **NEW** `src/nn/init.rs` | activation-aware initializer (Bräm-style) |
| [src/api.rs](src/api.rs) | implement `two_step_nodes` orchestration |
| [src/types.rs](src/types.rs) | extend `FitOptions` with `two_step_nodes: bool` |
| `examples/theophylline_node.ferx` | new example |
| `docs/src/model-file/dynamics-nn.md` (+ extend landing page from Phase A) + `docs/src/SUMMARY.md` | docs |
| `docs/book/` | rebuild |

## Reusing what's already there

- `MlpMapper` (Phase A M1) — forward pass code shared verbatim.
- `detect_mu_refs` Pattern 3 ([model_parser.rs:213](src/parser/model_parser.rs:213)) — additive etas on NN weights already supported.
- `EtaParamType::Additive` (used elsewhere in the parser) — same.
- `[odes]` block parser and Dormand-Prince solver — unchanged.
- FOCEI inner loop and Enzyme-AD path — unchanged. The AD path already differentiates through the ODE solver, so additive-eta NN weights inside an ODE RHS are differentiable end-to-end without any new AD wiring.
- The ReLU-via-`if` style is already mandated for AD-safe code in CLAUDE.md.

## Verification

End-to-end checks, in order:

1. `cargo build --features nn` and `cargo clippy --features nn` — no warnings.
2. `cargo test --features nn` — all new tests pass.
3. **Single-NN sanity**: a model with one `nn(...)` in the depot equation, zero hidden units, weights initialized so the NN reproduces `-KA * depot`, fits identically to the analytical one-cpt PO model on `examples/bioavailability_ode.ferx` data.
4. **Theophylline reproduction** (B4 deliverable): `examples/theophylline_node.ferx` reaches MSE within 20% of the classical two-cpt PO fit, matching Bräm's reported results.
5. **Two-step workflow**: `two_step_nodes = true` produces step-1 estimates with `omega = 0` and step-2 estimates strictly smaller in OFV (improvement from adding IIV). Logged to the fit YAML.
6. **Initializer reproducibility**: same seed → identical initial weights; different seeds → activation points uniformly distributed across observed input range (KS test in the unit test).

## Open questions

- **Identifiability and standard errors**: Bräm explicitly omits std errors for NN weights because units are interchangeable. ferx-core's covariance estimation in `outer_optimizer.rs` will likely produce ill-conditioned Hessians and may fail or emit warnings. Decision: auto-disable `covariance = true` when `[dynamics_nn]` blocks are present (with a `FitResult.warnings` entry per the CLAUDE.md convention). This is `[dynamics_nn]`-scoped only — `[covariate_nn]` weights are reasonably identifiable and standard covariance estimation should keep working there.
- **VPC behavior**: Bräm flags that conventional VPCs are unreliable for NODEs (ReLU-discontinuity + non-identifiable weights → unrealistic simulated trajectories). They propose conditional-distribution sampling instead. ferx-core does not currently produce VPCs in the engine (downstream tooling does), so this surfaces as a docs warning rather than code, but worth noting.
- **Time-dependent NN scope**: only needed for absorption-with-delay and similar patterns. Worth shipping in B1 but small; if it bloats, push to a B5.
- **Mixing NN and mechanistic in one RHS**: the natural composition (`d/dt(central) = nn(central) - CL/V * central`) Just Works under the proposed design. No special-casing needed — `nn(...)` is just one more expression node.

---

# Cross-phase guardrails (apply to Phase A + Phase B)

DCM and low-dim NODE touch disjoint code paths and can coexist in the same model file. The risks are not technical interference but UX confusion — same word "neural network" doing two different things with very different runtime, identifiability, and modeling implications. These guardrails are designed to make the difference visible at every touch point.

### Distinct block names at declaration site

- `[covariate_nn NAME]` for Phase A — NN maps covariates → PK params, consumed from `[individual_parameters]` via `NAME.OUTPUT`.
- `[dynamics_nn NAME]` for Phase B — NN appears on an ODE RHS, consumed from `[odes]` via `NAME(args)`.

The parser cross-checks usage: a `[covariate_nn]` referenced inside `[odes]` is a hard error pointing to `[dynamics_nn]`, and vice versa. Implemented in [src/parser/model_parser.rs](src/parser/model_parser.rs) during AST resolution. Tested under both Phase A M1 and Phase B B4.

### Block-level IIV declaration (`[dynamics_nn]` only)

`[dynamics_nn]` blocks accept `iiv = additive` to auto-emit additive etas on every weight and bias. This eliminates the dozen-or-more hand-written `omega ETA_W_..._...` lines a Bräm-style model otherwise requires. `[covariate_nn]` blocks intentionally do not expose `iiv` — etas in DCM live on the downstream PK params (CL, V, …), not on weights.

### Fit output reports active NN features (both formats)

ferx-core produces two parallel fit outputs: the CLI's human-readable `{model}-fit.yaml` (written from [src/io/output.rs](src/io/output.rs)) and the round-trippable `.fitrx` zip bundle (with `fit.json` inside, schema in [src/io/fitrx.rs](src/io/fitrx.rs)). Both should emit a top-level section listing every NN block active in the fit:

```
neural_networks:
  covariate_nn:
    - name: TYPICAL_PK
      inputs: [WT, CRCL]
      outputs: [CL, V1, Q, V2, KA]
      n_weights: 245
  dynamics_nn:
    - name: ka_nn
      inputs: [depot]
      hidden: 5
      iiv: additive
      n_weights: 11
```

Anyone debugging from a fit artifact alone can immediately tell what kind of NN model was fit. The `.fitrx` mirror lives in `fit.json` under the same key so `ferx-r`'s loader can round-trip it. This is also useful for `ferx`-internal tooling like the covariance auto-disable to find what's active.

### Runtime warning when `[dynamics_nn]` is present

On fit start, when any `[dynamics_nn]` block is present, push to `FitResult.warnings`:

```
Dynamics neural networks substantially increase fit time and reduce parameter identifiability.
Expect 10–100× slowdown vs classical PK and no standard errors for NN weights (Bräm et al. 2025).
```

Per CLAUDE.md the warning is also surfaced by the CLI layer. Sets expectations before the user thinks ferx-core is hanging. `[covariate_nn]` does not trigger this warning — DCM runs at roughly classical speed.

### Covariance auto-disable scoped to `[dynamics_nn]`

When `[dynamics_nn]` blocks are present and `covariance = true` in `[fit_options]`, the parser overrides `covariance = false` and emits a warning. `[covariate_nn]` weights are reasonably identifiable; covariance estimation should be left enabled there.

### Decision-tree docs landing page

`docs/src/model-file/neural-networks.md` (introduced in Phase A M1, extended in Phase B B4) is the canonical starting point. Required content:

- One-paragraph orientation contrasting DCM and low-dim NODE.
- A 6-row table answering "I have X, my dynamics are Y → use Z":

  | Known PK structure? | Known covariate model? | Use |
  |---|---|---|
  | Yes (e.g. 2-cpt oral) | Yes | Classical analytical — no NN needed |
  | Yes | No / partial | `[covariate_nn]` (Phase A) |
  | No / partial | Yes | `[dynamics_nn]` (Phase B) |
  | No | No | Both — they coexist |

- A two-line example of each block side-by-side.
- Links to the per-block reference pages (`covariate-nn.md`, `dynamics-nn.md`).

### Shared `nn` cargo feature, single MlpMapper

Both phases live behind `--features nn` and share `src/nn/mod.rs`. No `nn-dcm` / `nn-node` split — the distinction is a DSL-level decision, not a build-level one.
