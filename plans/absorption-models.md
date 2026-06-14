# Plan: Built-in absorption models (`[absorption]` block)

**Tracking issue:** [#322](https://github.com/FeRx-NLME/ferx-core/issues/322)
**Scope:** ferx-core (primary) + ferx-r (follow-up PR once `pub` API lands)
**Status:** approved roadmap, not yet implemented. Multi-PR / phased.

---

## Context

ferx-core today supports exactly one absorption model analytically: **first-order**
(`ka`), plus an optional lag time (`PK_IDX_LAGTIME`) and bioavailability (`PK_IDX_F`).
Anything richer — transit-compartment (Savic), Weibull, inverse-Gaussian (Freijer &
Post), zero-order, sequential/parallel/mixed — is only reachable by the user
hand-writing an `[odes]` chain (see `examples/transit_2cpt.ferx`). That hand-written
route is error-prone, **cannot estimate a continuous number of transit compartments**,
and forces the user to do math the engine should do.

The goal is to compute these **in Rust** as first-class, user-friendly built-ins, with
**robust handling of edge cases (no happy-path-only code)**. The named anchors are
**Savic 2007** (transit) and **Freijer & Post** (convection–dispersion ⇒ inverse-Gaussian
density input). The mechanism is a set of **built-in input-rate functions** (`transit`,
`igd`, `weibull`, `zero_order`, `first_order`) exposed two ways (see "DSL surface"):
**Layer 1** — callable directly in the `[odes]` RHS so the user writes the ODE explicitly
(Ron's proposal: `d/dt(depot) = transit(NTR, MTT) - KA*depot`), and **Layer 2** — an optional
`[absorption]` block that, on an analytical `pk` model, uses a closed form where one exists
and otherwise desugars to the Layer-1 ODE with a clear warning.

This is a **large, multi-PR feature**; the plan is phased so each model lands with its own
tests, NONMEM anchor, and docs.

## Goals / non-goals

- **Goal:** `[absorption]` block selecting transit / inverse-Gaussian / Weibull /
  zero-order / sequential / parallel / mixed, with parameters bound to
  `[individual_parameters]` (so they carry IIV, covariates, etc. for free).
- **Goal:** the `[absorption]` type works on **both** analytical (`pk ...`) and ODE
  (`ode(...)`) disposition models (Ron's review). On ODE models it injects the input as RHS
  forcing into the dosing compartment; on analytical models it dispatches a **closed form
  where one exists**, and falls back to numerical integration only where none does.
- **Goal:** **closed-form-first, not ODE-always.** Most absorption types reduce to
  superpositions of the analytical building blocks ferx already has (see "Which models stay
  analytical"), so they stay in the fast analytical engine. Weibull, inverse-Gaussian, and
  continuous-N transit have no elementary closed form; when one of those is applied to an
  analytically-specified model ferx integrates numerically **and emits a clear warning**
  (Ron's requirement) — never a silent ODE swap.
- **Goal:** **transparency.** Each model's math and its explicit ODE form are documented;
  `ferx check` surfaces the effective (synthesized) model so the numerical path is
  inspectable, not a black box. Where both forms exist they are cross-checked under the
  analytical↔ODE equivalence harness (ferx-r#127 / `tests/analytical_ode_equivalence.rs`).
- **Goal:** continuous (non-integer) N for the Savic transit model — the key thing the
  current hand-written ODE example cannot do.
- **Non-goal (this plan):** changing the existing analytical `pk` disposition solvers; they
  keep working unchanged for first-order/IV. Absorption models layer on top.
- **Non-goal (this plan):** the NONMEM coded-`RATE` data path itself (`RATE=-1`/`-2`,
  issue #324) — a separate data-reader feature this plan depends on for its zero-order
  family. See "Relationship to #324".

## Relationship to issue #324 (NONMEM coded RATE values)

#324 adds end-to-end support for NONMEM's coded `RATE` column (consolidating #95 and the
now-closed #282). In NONMEM, coded `RATE` is *parameter-driven*, not data-column-driven:
`RATE=-1` ⇒ the infusion **rate** is modeled (`R1` in `$PK`); `RATE=-2` ⇒ the infusion
**duration** is modeled (`D1` in `$PK`). Neither reads a data column; duration is the more
commonly estimated of the two. #324 is scoped to:

- **#324 Phase 0** — safety net: reject coded/malformed `RATE` (`-1`, `-2`, other negatives,
  non-finite) on a dose row instead of silently treating it as an IV bolus (today's bug).
  Ships first, standalone (PR #326).
- **#324 faithful support** — `RATE=-1` = rate modeled via an `R1`-style `.ferx` DSL
  parameter; `RATE=-2` = duration modeled via a `D1`-style DSL parameter. Both
  runtime/parameter-driven; **no `DURATION` data column**.

The piece this plan depends on is the **`RATE=-2` / `D1` modeled-duration** plumbing: a
*zero-order forcing term whose duration is an estimated model parameter* (`D1`-style
plumbing in both analytical and ODE paths). That same mechanism is what the zero-order
absorption family (`zero_order`, `sequential`, `mixed`) reuses.

- **Not a prerequisite** for Phase 0 (transit) or Phase 1 (inverse-Gaussian) of this plan —
  neither involves a zero-order input. The two headline models are unblocked by #324.
- **Is the foundation** for the zero-order absorption family in Phase 2 below.

Decision: ship #324's Phase 0 safety net first (independently valuable); its `D1`
modeled-duration path establishes the estimated-duration forcing this plan's Phase 2 then
reuses. Phase 0/1 of this plan can start in parallel, since they don't depend on it.

## DSL surface — two layers over one set of input-rate functions

The absorption math lives in **built-in input-rate functions** — `transit(n, mtt)`,
`igd(mat, cv2)` (inverse-Gaussian / Freijer & Post), `weibull(td, beta)`, `zero_order(dur)`,
and `first_order(ka)` (for composition). They are the single source of truth, exposed two
ways.

### Layer 1 — input-rate functions in `[odes]` (Ron's proposal; transparent foundation)

The user writes the ODE explicitly and calls the built-in for the input rate — keeping full
control of the compartment structure and *seeing* that it is an ODE — but without
hand-coding the Stirling gamma density:

```
[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = transit(NTR, MTT) - KA*depot
  d/dt(central) = KA*depot - (CL/V)*central
```

`transit(NTR, MTT)` returns the Savic transit-chain appearance rate into that compartment,
evaluated by the engine from time-after-dose and the dose amount (×F), superposed over doses
— the same dose context the infusion RHS-wrapper already carries. `igd(...)`, `weibull(...)`,
`zero_order(...)` behave identically. This is the natural home for the inherently-numerical
models (Weibull, IG, continuous-N transit) and satisfies Ron's transparency ask directly.

**Input-rate function for each model.** Each returns the dose-driven appearance rate into the
compartment it is added to (dose amount × F, superposed over doses). Fractions for parallel /
biphasic pathways are **plain scalar multipliers** — so no `pathway` grammar is needed in
Layer 1, and `frac` just splits the dose by linearity.

*Savic transit* — `transit(n, mtt)` into a depot, then first-order `ka` (shown above);
`ktr = (n+1)/mtt`, continuous `n`.

*Inverse-Gaussian (Freijer & Post)* — `igd(mat, cv2)` straight into central; the biphasic
form is two terms split by a fraction:
```
[odes]
  # single IG into 1-cpt
  d/dt(central) = igd(MAT, CV2) - (CL/V)*central
  # Freijer biphasic (sum of two IG), fraction FR through pathway 1
  d/dt(central) = FR*igd(MAT1, CV2_1) + (1-FR)*igd(MAT2, CV2_2) - (CL/V)*central
```

*Weibull* — `weibull(td, beta)` (td = scale, beta = shape):
```
[odes]
  d/dt(central) = weibull(TD, BETA) - (CL/V)*central
```

*Zero-order, estimated duration* — `zero_order(dur)` (constant rate over `dur`; this is the
modeled-duration / #324 `D1` case, reusable as an absorption input):
```
[odes]
  d/dt(central) = zero_order(DUR) - (CL/V)*central
```

*Parallel / dual first-order* — compose two `first_order(ka)` terms with a fraction (no need
for two depot compartments or per-compartment F):
```
[odes]
  d/dt(central) = FR*first_order(KA1) + (1-FR)*first_order(KA2) - (CL/V)*central
```

*Sequential (zero-order then first-order)* — `zero_order` fills the depot, `ka` to central:
```
[odes]
  d/dt(depot)   = zero_order(DUR) - KA*depot
  d/dt(central) = KA*depot - (CL/V)*central
```

*Mixed (zero-order + first-order, in parallel)*:
```
[odes]
  d/dt(central) = (1-FZO)*first_order(KA) + FZO*zero_order(DUR) - (CL/V)*central
```

(`first_order(ka)` is the existing first-order absorption exposed as an input-rate function
for composition; standalone first-order still uses the analytical `pk *_oral` path.)

Two implementation notes: **(i)** these are **engine intrinsics**, not pure expressions — the
`[odes]` evaluator must hand them the dose schedule, amount, F, and time-after-dose (extend
the expression evaluator plus the dose context the RHS-wrapper already holds). **(ii) Dose
routing:** when a compartment's RHS contains an input-rate function, the dose *feeds that
function* (it is the chain input) and must **not** also enter as a bolus into the same
compartment — define and test this rule explicitly (it is the classic Savic "dose into the
virtual transit, not the depot" subtlety).

### Layer 2 — `[absorption]` block (optional convenience; desugars to Layer 1 / closed forms)

A one-liner on an analytical disposition for users who don't want to write the ODE. It
selects the input model declaratively; the engine uses a **closed form where one exists**
(see "Which models stay analytical") and otherwise auto-builds the equivalent Layer-1 ODE,
emitting `W_ABSORPTION_NUMERICAL`. Parameters reference `[individual_parameters]` names
(like `cl=CL` on the pk line):

```
[structural_model]
  pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2)   # NB: no ka= when [absorption] present

[absorption]
  model = transit
  mtt   = MTT          # mean transit time
  n     = NTR          # number of transit compartments (continuous)
  ka    = KA           # optional; defaults to KTR=(n+1)/mtt
```

Multi-pathway models (Freijer sum-of-two-IG, parallel first-order) use repeated
`pathway` entries with a fraction:

```
[absorption]
  model    = inverse_gaussian
  pathway  = { mat = MAT1, cv2 = CV2_1, frac = FR1 }
  pathway  = { mat = MAT2, cv2 = CV2_2 }   # last frac defaults to 1 - sum(others)
```

> The `pathway = { ... }` inline-record form is a **new grammar construct** (no DSL
> precedent today). For the common ≤2-pathway case, repeated scalar keys (`mat1=`, `cv2_1=`,
> `frac1=`, `mat2=`, …) reach parity with zero new grammar; defer the brace form until a
> >2-pathway need is real.

Rules (all enforced at parse time, all with negative tests):
- `[absorption]` present ⇒ the `pk *_oral` line must **not** also map `ka=` (mutually
  exclusive: first-order is the no-`[absorption]` default). Mapping both is a parse error.
- `[absorption]` is only valid with an **oral** disposition (`one/two/three_cpt_oral`);
  combining with an IV-only model, or with a subject carrying `RATE>0` infusion dose rows,
  is a **parse error** (the dose route is ambiguous — see Robustness).
- Every referenced name must be a declared individual parameter (reuse the existing
  "undefined name" machinery — `parser/model_parser.rs` visit_*_nodes); unknown keys for a
  model are rejected; pathway fractions must be in (0,1] and sum to ≈1.

## The models (input rate `R_in(t)`, ∫₀^∞ R_in dt = F·Dose)

`tad = t − dose.time − lagtime`; `R_in = 0` for `tad ≤ 0`. Per-dose contributions are
superposed. `D` = `F·amt`.

| `model =` | `R_in(t)` | Params | Special fn | t→0 edge |
|---|---|---|---|---|
| `first_order` (default, no block) | `D·ka·e^{−ka·tad}` | ka | — | finite |
| `zero_order` | `D/Dur` on `[0,Dur]` else 0 | dur | — | step |
| `sequential` (0→1st) | zero-order fills depot over `dur`, then `ka` out | dur, ka | — | step |
| `parallel` (dual 1st-order) | `D·Σ fᵢ·kaᵢ·e^{−kaᵢ·tad}` | ka1,ka2,frac | — | finite |
| `mixed` (0 + 1st) | `f_zo·zero(dur) + (1−f_zo)·first(ka)` | dur, ka, frac | — | step |
| `transit` (**Savic**) | `D·KTR·(KTR·tad)^N·e^{−KTR·tad}/Γ(N+1)` into depot, then `ka`; `KTR=(N+1)/MTT` | mtt, n, [ka] | `ln_gamma` | `0^N`→0 (N>0) |
| `inverse_gaussian` (**Freijer&Post**) | `D·√(MAT/(2π·CV²·tad³))·exp(−(tad−MAT)²/(2·CV²·MAT·tad))` | mat, cv2 [, pathways] | exp/sqrt | →0, guard |
| `weibull` | `D·(β/Td)(tad/Td)^{β−1}·exp(−(tad/Td)^β)` | td (scale), beta (shape) | powf | β<1 ⇒ ∞ (integrable), guard |

Notes:
- **transit:** `n` counts the transit compartments **excluding** the final absorption (`ka`)
  compartment, so `KTR=(n+1)/MTT`. The gamma density is the chain output; it forces the
  **depot**, which empties to central via first-order `ka` (defaulting to `KTR` when omitted),
  matching rxode2/PKPDsim's `transit()`. Continuous N via `ln_gamma` (Lanczos; see Engine).
  Headline feature.
- **inverse_gaussian:** single or sum-of-two (Freijer biphasic). MAT = mean absorption time,
  CV² = relative dispersion of the absorption-time distribution — i.e. the standard
  inverse-Gaussian density with mean `μ=MAT` and shape `λ=MAT/CV²` (implementer mapping).

## Which models stay analytical vs need numerical integration

This is the crux of Ron's review: *does adding an absorption model silently turn an
analytical model into an ODE?* Mostly **no** — only where the math forces it.

| Absorption model | Closed form with linear disposition? | Engine |
|---|---|---|
| `first_order` | yes (Bateman) — already shipped | analytical |
| `zero_order` | yes (= infusion into depot/central) | analytical |
| `parallel` (dual first-order) | yes — superpose two `*_oral` solutions weighted by `frac` | analytical (reuses existing solvers) |
| `sequential` (0→1st) | yes (piecewise: zero-order fill, then first-order) | analytical |
| `mixed` (0 + 1st) | yes (superpose zero-order + first-order) | analytical |
| `transit` (Savic), **integer N** | yes (generalized Bateman / sum of N+1 terms) | analytical |
| `transit` (Savic), **continuous N** | yes **iff** the lower incomplete gamma `P(a,x)` is implemented; else numerical | analytical (Phase 3) or numerical |
| `weibull` | **no** elementary closed form | numerical |
| `inverse_gaussian` (Freijer & Post) | **no** elementary closed form (general multi-cpt) | numerical |

So, will analytical models be made into ODEs? **Not in general.** The
first-order/zero-order/parallel/sequential/mixed family and integer-N transit reduce to
**superpositions of the closed forms ferx already has** (e.g. `parallel` = two `two_cpt_oral`
evaluations weighted by `frac`) and stay analytical. Continuous-N transit stays analytical
once the incomplete-gamma special function lands (Phase 3, promoted from "optional"). Only
**Weibull** and **inverse-Gaussian** are inherently numerical — there is no closed-form
convolution of those input densities with a multi-compartment disposition, so they are
integrated (ODE / convolution quadrature) regardless of how the disposition was written.

**When the numerical path is used on an analytically-specified model, ferx warns**
(`W_ABSORPTION_NUMERICAL`, e.g. "absorption=weibull has no closed form with two_cpt_oral;
predictions use numerical integration (slower)"). A model written as `ode(...)` gets no
warning — the user already chose numerical integration.

## Engine architecture

Decouple **input function** from **disposition**, reusing existing machinery:

1. **Input function (new `src/pk/absorption.rs`):** `R_in(tad; θ)` per model. `src/ad/dual.rs`'s
   `Dual` already implements `exp`/`ln`/`sqrt`/`powf`, so the input functions can be written
   once over a small numeric trait that both `f64` and `Dual` satisfy — sharing one body across
   the plain-f64, dual-number, and Enzyme (concrete-f64) paths. This needs a **new** shared
   trait (today's AD path uses hand-written `_ad` duplicates, not generics) plus a `Dual` impl
   of `ln_gamma`; if either proves awkward, fall back to the existing duplicate-function
   pattern. Honor the `ad/` rule: **no `f64::max`/`min`** — use explicit comparisons (see
   CLAUDE.md). Each model also exposes `validate(θ) -> Result` and the analytic mass
   `∫R_in = F·Dose` (test invariant).
2. **Disposition — two dispatch modes** (chosen per absorption×disposition combo, per the
   table above):
   - **(a) Closed-form (preferred).** Stay in the analytical engine. The first-order /
     zero-order / parallel / sequential / mixed family and integer-N transit map `R_in` to a
     **superposition of the existing `pk/` closed forms** (e.g. `parallel` = two
     `two_cpt_oral` evaluations weighted by `frac`); continuous-N transit uses the
     incomplete-gamma closed form (Phase 3). No ODE solve, no warning.
   - **(b) Numerical fallback.** Weibull / inverse-Gaussian / continuous-N transit before the
     incomplete-gamma path lands: add `+R_in(tad)` forcing into the dosing compartment via the
     **same RHS-wrapper mechanism that already injects `+rate` for infusions**
     (`ode/predictions.rs` header doc: "adding `+rate` … via an RHS wrapper"). Emit
     `W_ABSORPTION_NUMERICAL` when the disposition was specified analytically.
   - **Works on both model kinds.** On an `ode(...)` disposition the absorption type injects
     the same `R_in(tad)` forcing into the user's dosing compartment (no warning — they chose
     ODE). On a `pk ...` disposition, mode (a)/(b) is selected by the table.
   - Shared by both modes: observed value via the existing obs-compartment plumbing; **SS=1**
     reuses `equilibrate_ss_state`; **lagtime/F** reuse `PK_IDX_LAGTIME` (shift `tad`) and
     `PK_IDX_F` (scale `D`); multiple doses / ADDL superpose `R_in` per dose.
3. **Special functions (`src/stats/special.rs`):** add `ln_gamma` via a **Lanczos** rational
   approximation (AD-safe, following the existing `erf` A&S precedent) — **not** bare Stirling:
   `N` is estimated continuously and transit `N` is commonly 1–10, where Stirling errs ~8% at
   N=1 / ~0.8% at N=10, enough to bias the absorption peak. IGD needs only `exp/sqrt`; Weibull
   needs `powf` — both already AD-safe.
4. **Incomplete-gamma closed form for transit (Phase 3, promoted from optional):** because
   keeping continuous-N transit analytical is now a goal (Ron's transparency concern),
   implement the regularized lower incomplete gamma `P(a,x)` (AD-safe) in `special.rs` so
   transit→1/2-cpt skips numerical integration. Sequence: ship transit on the numerical
   fallback first to prove the pipeline, then add the closed form and assert the two agree
   under the equivalence harness.

## Robustness ("no happy paths") — explicit requirements

Each item needs a negative/edge test so it registers Codecov patch coverage:

- **Parameter-domain validation** at parse + fit-init: `mtt>0`, `n≥0`, `dur>0`, `td>0`,
  `beta>0`, `mat>0`, `cv2>0`, `0<frac≤1`, `Σfrac≈1`. Parse errors for static violations;
  `FitResult.warnings` (`W_ABSORPTION_*`) for init-value violations (mirror the existing
  `W_NEGATIVE_LAGTIME` pattern in `diagnostics.rs`).
- **Singularity guards:** `tad ≤ ε ⇒ R_in = 0`; transit `0^N` and `log(tad)` guarded;
  Weibull `β<1` integrable spike capped/handled; IGD essential singularity at `tad→0`.
- **Mutual exclusivity & route checks:** `[absorption]` + `ka=` ⇒ error; `[absorption]` on
  IV-only disposition ⇒ error; `[absorption]` + a `RATE>0` infusion dose row ⇒ **parse error**
  (the dose route is ambiguous; decided over "documented precedence").
- **Mass-balance invariant** `∫R_in dt = F·Dose` as a unit test per model (catches a wrong
  normalization constant — the classic transit/IGD bug).
- **AD-safety:** no `f64::max`/`min` anywhere reachable from the AD path; re-enable a
  representative absorption test under the `autodiff` feature (per CLAUDE.md / issue #281
  CI work).

## Files (representative, not exhaustive)

- `src/types.rs` — new `AbsorptionSpec` + `AbsorptionModel` enum on `CompiledModel`; oral
  `PkModel` paths gain an optional spec.
- `src/parser/model_parser.rs` — parse/validate `[absorption]`; reuse undefined-name walker
  and the `consumes_pk_slot`/"declared-but-unused" census.
- `src/pk/absorption.rs` (new) — generic input functions + validation + mass.
- `src/stats/special.rs` — `ln_gamma` (+ later regularized incomplete gamma).
- `src/ode/predictions.rs` — synthesized-disposition + `R_in` forcing; SS reuse.
- prediction dispatcher / `src/estimation/inner_optimizer.rs` — route oral+absorption to the
  forced path; `src/diagnostics.rs` — new `W_ABSORPTION_*`.
- Docs: new `docs/src/model-file/absorption.md` + `SUMMARY.md`; cross-link
  `structural-model.md`; new `examples/*.ferx`; `CHANGELOG.md` (`[Unreleased] → Added`).
- `../ferx-r` follow-up PR for the new `pub` surface (+ `tools/update-ferx-core-lock.sh`).

## Phasing (one PR each)

- **Phase −1 — issue #324 (NONMEM coded `RATE`), standalone first.** Ship the safety net
  first (reject coded/malformed `RATE` instead of a silent IV bolus; PR #326). Faithful
  support follows as a parameter-driven DSL feature: `RATE=-1` = rate modeled (`R1`-style),
  `RATE=-2` = duration modeled (`D1`-style) — **no `DURATION` data column**. The
  `RATE=-2`/`D1` modeled-duration path establishes the estimated-duration forcing that this
  plan's Phase 2 zero-order family reuses. Independent of this plan's Phase 0/1, which can
  start in parallel.
- **Phase 0 — `transit()` input-rate function (Layer 1 first).** Implement the built-in
  `transit(n, mtt)` intrinsic callable in `[odes]` (Ron's proposal): the input-rate
  evaluator, dose-context wiring, the dose-routing rule (dose feeds the function, not a
  bolus), and `ln_gamma`. Anchor against the existing `transit_2cpt` dataset and a NONMEM
  Savic run — this proves the transparent path end-to-end. The optional `[absorption]` block
  (Layer 2), the closed-form-vs-numerical dispatch, and the `W_ABSORPTION_NUMERICAL` warning
  land here or as a Layer-2 follow-up (see Open questions).
- **Phase 1 — inverse-Gaussian (Freijer & Post).** Single + sum-of-two IG; **numerical**
  (no closed form). Anchor vs the Freijer & Post paper / a NONMEM `$DES` IG run.
- **Phase 2 — Weibull + zero-order + sequential + parallel + mixed.** Round out the
  catalogue; each with a NONMEM anchor. **Closed-form** for zero-order/sequential/parallel/
  mixed (superpose existing solvers; the zero-order family reuses #324's estimated-duration
  forcing); **numerical** for Weibull (warned on an analytical disposition).
- **Phase 3 — analytical incomplete-gamma closed form for transit** (1/2-cpt) so continuous-N
  transit stays in the analytical engine; assert it matches the Phase-0 numerical form under
  the equivalence harness.

## Tests & NONMEM anchoring (CLAUDE.md mandates)

- **Tier 1 (unit):** input-fn values vs hand-computed; mass-balance integral; `ln_gamma` vs
  reference; every param-validation error/warning.
- **Tier 2 (`tests/*.rs`):** parse `[absorption]` → `CompiledModel`; `fit()` returns
  immediately / errors on a bad spec (no convergence loop).
- **Tier 3 (slow, gated):** full fits per model to convergence (gate with
  `cfg_attr(not(feature="slow-tests"), ignore)`).
- **NONMEM comparison** (required for numeric features): transit & IG estimates/OFV vs
  equivalent NONMEM models, documented in the example pages or PR descriptions.
- **Gradient agreement (AD ≡ FD):** per model, a unit test asserting the AD/`Dual` gradient
  of `individual_nll` w.r.t. the absorption params matches the central-FD gradient to
  tolerance on a small fixture. It compiles/runs under **both** the default `--features ci`
  (FD) job and the `--features autodiff` (Enzyme) job — the FD job is the per-PR backstop;
  the `autodiff` job (#281 cadence) is where the AD path is actually exercised. This is the
  bridge that stops an AD-only regression (e.g. a wrong `ln_gamma` dual rule) slipping past
  FD-only PR CI — the #317 failure mode.

## Verification

- `cargo check --no-default-features --features ci`; push to CI for the test matrix.
  Coverage: `cargo +nightly llvm-cov --tests --no-default-features --features ci` to confirm
  patch ≥90% on each PR's diff.
- End-to-end smoke per phase: `ferx examples/transit_savic.ferx --data data/transit_2cpt.csv`
  → converges, `converged: true`, MTT/N estimates near the data-generating values.
- Per-PR `--features ci` (FD) verifies the FD gradient path; the `--features autodiff` job
  (#281 cadence) verifies the AD path. Mass-balance and the AD≡FD gradient-agreement test are
  the fast regression backstop and the bridge between the two.

## Open questions

- **Keep Layer 2 (`[absorption]` block) at all, or ship Layer-1 functions only?** Ron's
  proposal is just the `[odes]` input-rate functions (fully transparent, composable, no
  silent ODE). The block adds convenience but also the closed-form dispatch + warning
  machinery. Decision pending: ship Layer 1 first (Phase 0), then decide whether Layer 2
  earns its complexity once Layer 1 is in users' hands.
- **`transit()` argument form** — `transit(n, mtt)` (explicit, recommended) vs `transit(n)`
  reading a conventional `MTT`/`KTR` param. Explicit args avoid magic.

## Open risks

- **Speed:** ODE-forcing is slower than closed forms; acceptable baseline, Phase 3 mitigates
  transit. Quantify on warfarin-sized data.
- **AD through `ln_gamma` / `powf`:** must verify Enzyme handles them (the `autodiff` CI from
  issue #281 is the gate); fall back to FD for the absorption params if needed.
- **DSL ergonomics for multi-pathway** (`pathway = {...}`): a new inline-record sub-grammar
  with no DSL precedent — deferred in favour of repeated scalar keys for the ≤2-pathway case
  (see DSL surface).
