# Plan: Built-in absorption models (input-rate functions + `ode_template`)

**Tracking issue:** [#322](https://github.com/FeRx-NLME/ferx-core/issues/322)
**Scope:** ferx-core (primary) + ferx-r (follow-up PR once `pub` API lands)
**Status:** approved roadmap. Prerequisite #324 safety net (PR #326) **merged 2026-06-14**;
remaining models not yet implemented. Multi-PR / phased.

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
density input). The mechanism is a set of **built-in input-rate functions** (`transit`, `igd`, `weibull`,
`zero_order`, `first_order`) callable in the `[odes]` RHS, so the user writes the ODE
explicitly (Ron's proposal: `d/dt(depot) = transit(n=NTR, mtt=MTT) - KA*depot`). The disposition is
supplied explicitly — a hand-written `ode(...)` or a generated `ode_template ...` — never
invented behind an analytical `pk` request (which errors instead). See "DSL surface".

This is a **large, multi-PR feature**; the plan is phased so each model lands with its own
tests, NONMEM anchor, and docs.

## Goals / non-goals

- **Goal:** built-in absorption input-rate functions (`transit`, `igd`, `weibull`,
  `zero_order`, `first_order`) usable in `[odes]`, with arguments bound to
  `[individual_parameters]` (so they carry IIV, covariates, etc. for free).
- **Goal:** **explicit disposition, no surprises** (Ron). Absorption feeds an ODE supplied by
  the user — hand-written `ode(...)` or generated `ode_template ...`. Asking for an
  ODE-only absorption model (transit / IG / Weibull) on an analytical `pk ...` is a clear
  **error** pointing at `ode_template` — never a silent (or even warned) analytical→ODE swap.
- **Goal:** `ode_template NAME(...)` generates the standard disposition ODE from the codified
  analytical↔ODE transforms (ferx-r#127 / `tests/analytical_ode_equivalence.rs`); a general
  primitive, reusable beyond absorption (TMDD, …).
- **Goal:** continuous (non-integer) N for the Savic transit model — the key thing the
  current hand-written ODE example cannot do.
- **Non-goal (this plan):** the declarative `[absorption]` block — **dropped** (see DSL
  surface); absorption is the input-rate functions plus an explicit ODE disposition.
- **Non-goal (this plan):** changing the existing analytical `pk` disposition solvers; they
  keep working unchanged for the standard closed-form models.
- **Non-goal (this plan):** the NONMEM coded-`RATE` data path itself (`RATE=-1`/`-2`,
  issue #324) — a separate data-reader feature this plan depends on for its zero-order
  family. See "Relationship to #324".

## Relationship to issue #324 (NONMEM coded RATE values)

#324 adds end-to-end support for NONMEM's coded `RATE` column (consolidating #95 and the
now-closed #282). In NONMEM, coded `RATE` is *parameter-driven*, not data-column-driven:
`RATE=-1` ⇒ the infusion **rate** is modeled (`R1` in `$PK`); `RATE=-2` ⇒ the infusion
**duration** is modeled (`D1` in `$PK`). Neither reads a data column; duration is the more
commonly estimated of the two. #324 is scoped to:

- **#324 safety net (PR #326) — ✅ MERGED 2026-06-14.** Rejects coded/malformed `RATE` (`-1`,
  `-2`, other negatives, non-finite) on a dose row instead of silently treating it as a bolus
  (the original bug). Shipped standalone.
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

Decision: ship #324's safety net first (independently valuable); its `D1`
modeled-duration path establishes the estimated-duration forcing this plan's Phase 2 then
reuses. Phase 0/1 of this plan can start in parallel, since they don't depend on it.

## DSL surface

Absorption is a set of **built-in input-rate functions** — `transit(n, mtt)`,
`igd(mat, cv2)` (inverse-Gaussian / Freijer & Post), `weibull(td, beta)`, `zero_order(dur)`,
and `first_order(ka)` (for composition) — used **inside an ODE**. The disposition the
absorption feeds is supplied **explicitly**, never invented behind the user's back. Three
ways to supply it: a hand-written `ode(...)`, an analytical `pk ...` (closed-form models
only — see the error rule), or a generated `ode_template ...` (Ron's proposal — ferx writes
the standard disposition ODE for you). No model is ever *silently* turned into an ODE.

### Absorption: input-rate functions in `[odes]`

The user writes (or generates, below) the ODE and calls the built-in for the input rate —
keeping full control of the compartment structure and *seeing* that it is an ODE — without
hand-coding the Stirling gamma density:

```
[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = transit(n=NTR, mtt=MTT) - KA*depot
  d/dt(central) = KA*depot - (CL/V)*central
```

`transit(n=NTR, mtt=MTT)` returns the Savic transit-chain appearance rate into that compartment,
evaluated by the engine from time-after-dose and the dose amount (×F), superposed over doses
— the same dose context the infusion RHS-wrapper already carries. `igd(...)`, `weibull(...)`,
`zero_order(...)` behave identically. This is the natural home for the inherently-numerical
models (Weibull, IG, continuous-N transit) and satisfies Ron's transparency ask directly.

**Input-rate function for each model.** Each returns the dose-driven appearance rate into the
compartment it is added to (dose amount × F, superposed over doses). Arguments are **named**
(matching the `pk(...)` convention): `transit(n=NTR, mtt=MTT)` — order-proof and
parser-validated, so a swapped or typo'd argument errors instead of silently giving wrong
numbers. Fractions for parallel / biphasic pathways are **plain scalar multipliers** — so no
`pathway` grammar is needed, and `frac` just splits the dose by linearity.

*Savic transit* — `transit(n, mtt)` into a depot, then first-order `ka` (shown above);
`ktr = (n+1)/mtt`, continuous `n`.

*Inverse-Gaussian (Freijer & Post)* — `igd(mat, cv2)` straight into central; the biphasic
form is two terms split by a fraction:
```
[odes]
  # single IG into 1-cpt
  d/dt(central) = igd(mat=MAT, cv2=CV2) - (CL/V)*central
  # Freijer biphasic (sum of two IG), fraction FR through pathway 1
  d/dt(central) = FR*igd(mat=MAT1, cv2=CV2_1) + (1-FR)*igd(mat=MAT2, cv2=CV2_2) - (CL/V)*central
```

*Weibull* — `weibull(td, beta)` (td = scale, beta = shape):
```
[odes]
  d/dt(central) = weibull(td=TD, beta=BETA) - (CL/V)*central
```

*Zero-order, estimated duration* — `zero_order(dur)` (constant rate over `dur`; this is the
modeled-duration / #324 `D1` case, reusable as an absorption input):
```
[odes]
  d/dt(central) = zero_order(dur=DUR) - (CL/V)*central
```

*Parallel / dual first-order* — compose two `first_order(ka)` terms with a fraction (no need
for two depot compartments or per-compartment F):
```
[odes]
  d/dt(central) = FR*first_order(ka=KA1) + (1-FR)*first_order(ka=KA2) - (CL/V)*central
```

*Sequential (zero-order then first-order)* — `zero_order` fills the depot, `ka` to central:
```
[odes]
  d/dt(depot)   = zero_order(dur=DUR) - KA*depot
  d/dt(central) = KA*depot - (CL/V)*central
```

*Mixed (zero-order + first-order, in parallel)*:
```
[odes]
  d/dt(central) = (1-FZO)*first_order(ka=KA) + FZO*zero_order(dur=DUR) - (CL/V)*central
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

### Disposition: `ode_template` (Ron's proposal) — explicit, no surprises

Writing the full disposition ODE by hand every time is verbose. `ode_template NAME(...)`
tells ferx to **generate** the standard disposition ODE for a named model — the codified
analytical→ODE transforms already specified in ferx-r#127 / `ode-analytical-equivalence.md`
(e.g. `two_cpt_oral` → `depot`/`central`/`periph` states with the micro-constant RHS and
`obs_scale = V1`). The user customises it in `[odes]` using **override semantics** (Ron's
call): any `d/dt(X)` declared in `[odes]` **replaces** the template's equation for compartment
`X` (maximum flexibility); compartments left undeclared keep the generated RHS. To add the
Savic input, re-declare the depot with the transit term:

```
[structural_model]
  ode_template two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2)   # ferx writes the 2-cpt oral ODE

[odes]
  # re-declaring d/dt(depot) OVERRIDES the template's depot equation
  d/dt(depot) = transit(n=NTR, mtt=MTT) - KA*depot
  # d/dt(central) and d/dt(periph) keep the generated equations
```

Because the user typed `ode_template`, they **expect** ferx to write ODEs — there is no
hidden conversion (Ron's core requirement). The same primitive generalizes beyond absorption
(Ron: future uses such as **TMDD**, **TGI**, **neutropenia** — generate a standard
disposition, then add the extra terms), so it is worth building generically; this plan keeps
scope to **absorption first**.

### The error rule (analytical + ODE-only absorption ⇒ error)

If a user writes an **analytical** `pk one/two/three_cpt_oral(...)` and *also* asks for an
absorption model with no closed form (transit / IG / Weibull), ferx **errors** with a
message pointing at `ode_template` — it does **not** silently (or even with a warning) build
an ODE behind an analytical request. This is Ron's "avoid surprises":

> *"transit absorption requires an ODE; replace `pk two_cpt_oral(...)` with
> `ode_template two_cpt_oral(...)` and add `transit(...)` in `[odes]`."*

### Dropped: the declarative `[absorption]` block

An earlier draft proposed a declarative `[absorption]` block. **Dropped** — Ron confirmed it
is not needed (2026-06-14): the input-rate functions already give a one-line absorption term,
and keeping the surface generic/simple beats a dedicated block. On an analytical `pk` it would
be the silent analytical→ODE conversion Ron objected to (the error rule above handles that);
on `ode_template` it would be a second, redundant way to write what the functions already
express, plus a `pathway = {}` grammar and closed-form-dispatch machinery for no real gain.
Absorption = input-rate functions + an explicit ODE disposition (`ode` or `ode_template`).

Parser rules (enforced, with negative tests): input-rate-function arguments must be declared
individual parameters (reuse the undefined-name machinery in `parser/model_parser.rs`);
unknown function names / wrong arity are rejected; an input-rate function combined with an
analytical `pk` disposition triggers the error rule above; parallel/biphasic fractions must
be in (0,1].

## The models (input rate `R_in(t)`, ∫₀^∞ R_in dt = F·Dose)

`tad = t − dose.time − lagtime`; `R_in = 0` for `tad ≤ 0`. Per-dose contributions are
superposed. `D` = `F·amt`.

| `model =` | `R_in(t)` | Params | Special fn | t→0 edge |
|---|---|---|---|---|
| `first_order` (default) | `D·ka·e^{−ka·tad}` | ka | — | finite |
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

## Which models *could* be accelerated with a closed form (internal only)

With the error rule above, the user always specifies their disposition explicitly, so this
table is **no longer a user-facing dispatch** — it is an internal performance note. Where an
`ode_template`/`ode` absorption model happens to have a closed form, ferx *may* compute it
via the fast analytical path instead of integrating, **provided the two are proven identical
under the equivalence harness** (so there is no behavioural surprise — same numbers, faster).
Everything else integrates.

| Absorption model | Closed form with linear disposition? | Can ferx accelerate internally? |
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

The first-order / zero-order / parallel / sequential / mixed family and integer-N transit
are superpositions of closed forms ferx already has (e.g. `parallel` = two `two_cpt_oral`
evaluations weighted by `frac`); continuous-N transit gains a closed form once the
incomplete-gamma special function lands (Phase 3). **Weibull** and **inverse-Gaussian** have
no elementary closed-form convolution with a multi-compartment disposition and always
integrate. The user-facing model is the ODE they wrote; any closed-form acceleration is an
equivalence-tested optimization underneath it, not a different code path they can observe.

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
2. **Forcing into the user's ODE.** `R_in(tad)` is added into the dosing compartment of the
   disposition the user supplied (`ode(...)` or the `ode_template`-generated states) via the
   **same RHS-wrapper mechanism that already injects `+rate` for infusions**
   (`ode/predictions.rs` header doc: "adding `+rate` … via an RHS wrapper"). No silent
   conversion: an analytical `pk` disposition + an ODE-only absorption model is rejected by the
   error rule, not forced. Shared plumbing: observed value via the existing obs-compartment
   path; **SS=1** reuses `equilibrate_ss_state`; **lagtime/F** reuse `PK_IDX_LAGTIME` (shift
   `tad`) and `PK_IDX_F` (scale `D`); multiple doses / ADDL superpose `R_in` per dose.
   - **Internal closed-form acceleration (optional):** where the combo has a closed form
     (first/zero/parallel/sequential/mixed, integer-N transit, continuous-N transit via
     incomplete gamma), ferx *may* compute it via the `pk/` solvers instead of integrating —
     **only** when proven identical under the equivalence harness, so the user sees the same
     numbers, faster. This is an optimization beneath the ODE the user wrote, never a separate
     observable path.
   - **`ode_template` generation:** the named-model → ODE transform (states, micro-constant
     RHS, `obs_scale`) is codified once from `ode-analytical-equivalence.md`; the absorption
     term is appended to the generated depot/dosing compartment.
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
- **Route checks:** an ODE-only absorption input-rate function (`transit`/`igd`/`weibull`) on
  an analytical `pk` disposition ⇒ **error** pointing at `ode_template` (the error rule); an
  input-rate function plus a `RATE>0` infusion dose row into the same compartment ⇒ **parse
  error** (dose route ambiguous).
- **Mass-balance invariant** `∫R_in dt = F·Dose` as a unit test per model (catches a wrong
  normalization constant — the classic transit/IGD bug).
- **AD-safety:** no `f64::max`/`min` anywhere reachable from the AD path; re-enable a
  representative absorption test under the `autodiff` feature (per CLAUDE.md / issue #281
  CI work).

## Files (representative, not exhaustive)

- `src/types.rs` — new `AbsorptionSpec` + `AbsorptionModel` enum on `CompiledModel`; oral
  `PkModel` paths gain an optional spec.
- `src/parser/model_parser.rs` — parse `ode_template NAME(...)` + the input-rate function
  intrinsics; the error rule; reuse the undefined-name walker / "declared-but-unused" census.
- `src/pk/absorption.rs` (new) — generic input functions + validation + mass; `ode_template`
  generation reuses the `ode-analytical-equivalence.md` transforms.
- `src/stats/special.rs` — `ln_gamma` (+ later regularized incomplete gamma).
- `src/ode/predictions.rs` — synthesized-disposition + `R_in` forcing; SS reuse.
- prediction dispatcher / `src/estimation/inner_optimizer.rs` — route oral+absorption to the
  forced path; `src/diagnostics.rs` — new `W_ABSORPTION_*`.
- Docs: new `docs/src/model-file/absorption.md` + `SUMMARY.md`; cross-link
  `structural-model.md`; new `examples/*.ferx`; `CHANGELOG.md` (`[Unreleased] → Added`).
- `../ferx-r` follow-up PR for the new `pub` surface (+ `tools/update-ferx-core-lock.sh`).

## Phasing (one PR each)

- **Prerequisite — issue #324 (NONMEM coded `RATE`), standalone first.** Safety net **✅ MERGED**
  (PR #326, 2026-06-14: rejects coded/malformed `RATE` instead of a silent bolus). Faithful
  support follows as a parameter-driven DSL feature: `RATE=-1` = rate modeled (`R1`-style),
  `RATE=-2` = duration modeled (`D1`-style) — **no `DURATION` data column**. The
  `RATE=-2`/`D1` modeled-duration path establishes the estimated-duration forcing that this
  plan's Phase 2 zero-order family reuses. Independent of this plan's Phase 0/1, which can
  start in parallel.
- **Phase 0 — `transit()` input-rate function + `ode_template`.** Implement the built-in
  `transit(n, mtt)` intrinsic callable in `[odes]` (Ron's proposal): the input-rate evaluator,
  dose-context wiring, the dose-routing rule (dose feeds the function, not a bolus), `ln_gamma`,
  and `ode_template` generation for the standard PK models + the analytical-`pk`-plus-absorption
  **error rule**. Anchor against the existing `transit_2cpt` dataset and a NONMEM Savic run —
  proves the transparent path end-to-end.
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
- **Tier 2 (`tests/*.rs`):** parse `ode_template` + input-rate functions → `CompiledModel`;
  the error rule fires on `pk` + ODE-only absorption; `fit()` returns immediately / errors on a
  bad spec (no convergence loop).
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

## Resolved decisions

Ron, 2026-06-14:
- **No `[absorption]` block.** Input-rate functions in `[odes]` suffice; keep the surface
  generic/simple. Both **A** (hand-written `ode`) and **B** (`ode_template`) ship and coexist.
- **`ode_template` scope: absorption first.** TMDD / TGI / neutropenia are future uses of the
  same primitive, out of scope here.
- **Override semantics:** a `d/dt(X)` re-declared in `[odes]` **overrides** the template's
  equation for `X` (maximum flexibility) — no `+=` append form.

Argument convention:
- **Input-rate functions take named args** — `transit(n=NTR, mtt=MTT)`, not positional and
  not a conventional-param lookup. Matches the `pk(...)` convention; order-proof and
  parser-validated (swapped/typo'd args error rather than silently producing wrong results,
  per the repo's no-silent-wrong-results rule). Reuses the `pk(...)` kwarg-parsing logic,
  extended into the `[odes]` expression parser (today only `UnaryFn` exists there).

No open questions outstanding.

## Open risks

- **Speed:** ODE-forcing is slower than closed forms; acceptable baseline, Phase 3 mitigates
  transit. Quantify on warfarin-sized data.
- **AD through `ln_gamma` / `powf`:** must verify Enzyme handles them (the `autodiff` CI from
  issue #281 is the gate); fall back to FD for the absorption params if needed.
- **DSL ergonomics for multi-pathway** (`pathway = {...}`): a new inline-record sub-grammar
  with no DSL precedent — deferred in favour of repeated scalar keys for the ≤2-pathway case
  (see DSL surface).
