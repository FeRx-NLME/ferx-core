# Plan: Built-in absorption models (input-rate functions + `ode_template`)

**Tracking issue:** [#322](https://github.com/FeRx-NLME/ferx-core/issues/322)
**Scope:** ferx-core (primary) + ferx-r (follow-up PR once `pub` API lands)
**Status:** approved roadmap, in progress (updated 2026-06-27).
- **Prerequisite #324:** safety net (PR #326), **modeled infusion duration `Dn` /
  `RATE=-2`** (PR #384), and **modeled rate `Rn` / `RATE=-1` on both engines**
  (PR #418) **merged**; see #383 for any residual analytical-engine items.
- **Phase 0a — `transit()`** (PR #343) **and its NONMEM Savic anchor** (PR #385)
  **merged**; `ln_gamma` building block merged (#340).
- **Phase 0b — `ode_template` generation + the analytical-`pk`-plus-absorption
  error rule** (PR #363) **merged**.
- **Phase 1 — inverse-Gaussian `igd()`** (PR #389) **merged**.
- **Analytic `Dual2` ODE forcing (#430)** — `igd` (slice 1), `transit` (slice 2),
  and (with Phase 2) `weibull` are all lifted off the FD fallback; **#430 closed**.
- **Phase 2 — `weibull()`** (#497) **✅ MERGED — PR #498** (`6252946a`) — log-domain
  forcing, exact analytic `Dual2` gradients, permanent ODE-only error rule, AUC-mass
  invariant; **NONMEM `$DES` anchor ran** (slow-tests-gated `tests/weibull_nonmem_anchor.rs`:
  ferx FOCEI marginal OFV vs NONMEM `#OBJV` −943.833 at NONMEM's optimum, ~1.5 OFV apart).
  **Remaining in Phase 2** (the zero-order / dual-pathway family, now tracked):
  - **#504 — `zero_order(dur)` + `sequential`** — **✅ MERGED — PR #556 (`4fd737e9`,
    2026-06-27).** Single-input (no fraction); `InputRateKind::ZeroOrder` is the new intrinsic,
    `sequential` is just `zero_order(dur=DUR) - KA*depot` composition (no new mechanism).
    Delivered as a **per-segment constant** through the wrapper's spanning channel (like an
    infusion), not pointwise — the hard `tad ≤ dur` cutoff is placed on the ODE timeline so the
    post-cutoff segment can't over-count mass (`predictions.rs`). The merged `RATE=-2`/`Dn`
    forcing (#384) is reused for the **value path**; `dur` ships on **FD**
    (`InputRateKind::ZeroOrder.supported_over_dual()=false`), since its moving-boundary
    `∂/∂dur` (a dual-channel impulse at `tad=dur` that *also* lifts #384's `Dn` infusion off FD)
    is the cross-cutting **follow-up #530**, not a `zero_order` rider. Shipped
    `examples/{zero_order,sequential}_absorption.ferx` + a **direct NONMEM ADVAN1 anchor**
    (`tests/zero_order_absorption.rs`, max abs ≈ 4e-4) plus reset-mid-window & lag>0 adversarial
    tests. **ferx-r follow-up ✅ done — PR [#204](https://github.com/FeRx-NLME/ferx-r/pull/204)
    merged 2026-06-27 (`56e7cafc`): pin bump + zero_order/sequential examples + validate tests.**
  - **#505 — `parallel` / `mixed` + `first_order()` composition** ← **NEXT (after #388)**:
    dual-pathway, **blocked on the shared fraction multiplier in #388** (the parser still
    rejects scaled input-rate calls — `call_is_scaled_or_signed`, `model_parser.rs`). Needs a
    new `InputRateKind::FirstOrder` (`first_order(ka)`) so the existing first-order absorption
    can be composed in `[odes]`.
- **#388 — biphasic IG + the shared input-rate fraction mechanism.** Build the
  `FR*fn(...) + (1-FR)*fn(...)` multiplier once; consumed by both biphasic IG and
  #505's `parallel`/`mixed`.
- **Phase 3 — #386 — analytical incomplete-gamma / exponential tilting** for
  transit + IG. Now *speed-only* (#430 already gave transit/IG exact gradients on
  the ODE path), so this only removes the ODE solve. Lowest urgency. Weibull never
  reaches Phase 3 (no closed form) — its only exact-gradient route is the #430
  generic forcing.

Recommended sequence after Weibull (#498) and zero-order (#504, ✅ merged): **#388**
(shared input-rate **fraction multiplier** — biphasic IG; unblocks #505) → **#505**
(`parallel`/`mixed` + `first_order()`) → **#386** (Phase 3 speed pass). The
moving-boundary sensitivity **#530** (`zero_order` `dur` + `Dn` off FD) is a parallel,
cross-cutting track — independent of the catalogue work above, not gating it.

Multi-PR / phased.

> **Autodiff landscape (2026-06):** the Enzyme autodiff path (`src/ad/`, the `autodiff`
> Cargo feature) was **retired** (#367/#381) in favour of hand-rolled forward sensitivities in
> `src/sens/` over the `PkNum` trait — closed forms and ODE RHS are written **once** as
> `*_g<T: PkNum>` generics (`T = f64` → predictions, `T = Dual2<M>` → exact 1st/2nd-order
> `∂/∂η`, `∂/∂θ`). This plan's original AD notes — written against Enzyme — are updated
> throughout to the `sens/`/`Dual2` model. `gradient = auto` (analytic where in scope, else FD)
> is the default; `gradient = ad` now errors (`E_AD_RETIRED`); the `.max()/.min()`
> Enzyme-intrinsic restriction no longer applies (`Dual2` handles `max`/`min` by comparison).

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

> **#388 implementation constraint (decided on the issue):** the multiplier must be a **single
> declared individual parameter** (`FR*igd(...)`) so it binds to one slot (`frac_slot`) — *not*
> an arbitrary expression like `(1-FR)`. So a two-pathway split is written as **two declared
> parameters** that sum to 1: `FR1*igd(...) + FR2*igd(...)` with `FR2 = 1 - FR1` declared in
> `[individual_parameters]` (validated `Σfrac ≈ 1`). The `(1-FR)*...` / `(1-FZO)*...` forms in
> the #505 `parallel`/`mixed` examples below are shorthand — they lower to that declared-complement
> pattern, not inline arithmetic in the `[odes]` RHS.

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
absorption model with no closed form (**Weibull only**, after Phase 3), ferx **errors** with
a message pointing at `ode_template` — it does **not** silently (or even with a warning)
build an ODE behind an analytical request. This is Ron's "avoid surprises":

> *"Weibull absorption requires an ODE; replace `pk two_cpt_oral(...)` with
> `ode_template two_cpt_oral(...)` and add `weibull(...)` in `[odes]`."*

Transit and IG both have closed-form convolutions with linear 1/2-cpt disposition (see
Phase 3 and the closed-form table above), so after Phase 3 they are **not** subject to this
error rule — they route to the analytical path instead. Until Phase 3 ships, the error rule
applies to all three (transit / IG / Weibull), with the message pointing at `ode_template`
for the interim ODE path.

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
| `transit` (Savic), **continuous N** | yes — exponential tilting of the Gamma distribution: `∫₀ᵗ R_in·e^(k·u)du = M(k)·P(n+1,(KTR−k)·t)` where P is the regularized incomplete gamma; condition `k < KTR` | analytical (Phase 3) |
| `weibull` | **no** elementary closed form | numerical (analytic ∂ only via the `PkNum`-generic forcing — see Engine §) |
| `inverse_gaussian` (Freijer & Post) | yes — exponential tilting of the IG distribution: `∫₀ᵗ f_IG(u;μ,λ)·e^(k·u)du = M(k)·F_IG(t;μ*,λ)` where `μ*=μ/√(1−2μ²k/λ)` and `M(k)=exp(λ/μ·(1−√(1−2μ²k/λ)))`; condition `k < λ/(2μ²)` | analytical (Phase 3) |

The first-order / zero-order / parallel / sequential / mixed family and integer-N transit
are superpositions of closed forms ferx already has (e.g. `parallel` = two `two_cpt_oral`
evaluations weighted by `frac`). **Continuous-N transit** and **inverse-Gaussian** both have
closed-form convolutions with 1/2-cpt disposition via the **exponential tilting property**:
the Gamma and IG distributions are each closed under tilting by `e^(k·t)`, so the
convolution integral reduces to the respective distribution's CDF evaluated at a shifted
parameter — regularized incomplete gamma `P(a,x)` for transit, normal CDF `Φ` (via the
known IG CDF) for IG. The `TiltedAbsorption` trait (see Engine below) captures both.
**Weibull** alone has no elementary closed form and always integrates. The user-facing model
is the ODE they wrote; any closed-form acceleration is an equivalence-tested optimization
underneath it, not a different code path they can observe.

## Engine architecture

Decouple **input function** from **disposition**, reusing existing machinery:

1. **Input function (`src/pk/absorption.rs`):** `R_in(tad; θ)` per model. Today these are plain
   `f64` (`transit_input_rate`, `inverse_gaussian_input_rate`, and the hoisted
   `PreparedInputRate::rate`), each exposing `validate(θ) -> Result` and satisfying the analytic
   mass `∫R_in = F·Dose` (the mass-balance test invariant). The numeric building blocks are only
   `+ − * /`, `.ln()`, `.exp()`, `.sqrt()`, `ln_gamma` — nothing autodiff-hostile (the old "no
   `f64::max`/`min`" Enzyme rule is gone; `Dual2` handles `max`/`min` by comparison).
   - **Sensitivities (post-Enzyme):** the shared-numeric-trait idea is now realised by the
     `PkNum` trait in `src/sens/` — the closed-form PK solutions and ODE RHS are written once as
     `*_g<T: PkNum>` generics (`T = f64` → predictions, `T = Dual2<M>` → exact 1st/2nd-order
     sensitivities); there is no second `_ad` copy to keep in sync. The **smooth** absorption
     input functions are now `PkNum`-generic — `PreparedInputRate::rate`/`prepare_dual` (and the
     `ln_gamma` transit calls) evaluate over `Dual2`, so `transit()`/`igd()`/`weibull()` get
     **exact analytic** ODE forcing sensitivities (#468/#430/#498), gated by
     `InputRateKind::supported_over_dual()`.
   - **Phase 0a finding (resolved, with one designed exception):** transit originally shipped
     `f64`-only and differentiated via **finite differences**; #430 (`igd`) / #468 (`transit`) /
     #498 (`weibull`) lifted all three onto the analytic `Dual2` path (`src/sens/ode_provider.rs`:
     state integrated as `Dual2<N>` through the generic RK45), so they are no longer FD holdouts.
     The remaining FD case is **by design**: `zero_order` (#504) has a *moving boundary* (`R_in`
     cuts off at `tad=dur`), whose `∂/∂dur` is not a `PkNum`-generic expression but a dual-channel
     impulse at `tad=dur` (a Leibniz boundary term). It ships with `dur` on FD
     (`supported_over_dual()=false`) until that impulse is built — the same gap that keeps #384's
     modeled-duration `Dn` infusion on FD; see "Analytic sensitivities for the ODE forcing".
2. **Forcing into the user's ODE.** `R_in(tad)` is added into the dosing compartment of the
   disposition the user supplied (`ode(...)` or the `ode_template`-generated states) via the
   **same RHS-wrapper mechanism that already injects `+rate` for infusions**
   (`ode/predictions.rs` header doc: "adding `+rate` … via an RHS wrapper"). No silent
   conversion: an analytical `pk` disposition + an ODE-only absorption model is rejected by the
   error rule, not forced. Shared plumbing: observed value via the existing obs-compartment
   path; **lagtime/F** reuse `PK_IDX_LAGTIME` (shift `tad`) and `PK_IDX_F` (scale `D`); multiple
   doses / ADDL superpose `R_in` per dose. **SS=1** was intended to reuse `equilibrate_ss_state`,
   but Phase 0a **defers and rejects** it (`E_ABSORPTION_SS`): the periodic steady state of a
   forcing whose `R_in` tail spills across the dosing interval needs dedicated treatment (the
   per-cycle pulse equilibration is only exact when `II ≫` the absorption window). A later phase
   adds proper SS-with-forcing.
   - **Internal closed-form acceleration (optional):** where the combo has a closed form
     (first/zero/parallel/sequential/mixed, integer-N transit, continuous-N transit via
     incomplete gamma), ferx *may* compute it via the `pk/` solvers instead of integrating —
     **only** when proven identical under the equivalence harness, so the user sees the same
     numbers, faster. This is an optimization beneath the ODE the user wrote, never a separate
     observable path.
   - **`ode_template` generation:** the named-model → ODE transform (states, micro-constant
     RHS, `obs_scale`) is codified once from `ode-analytical-equivalence.md`; the absorption
     term is appended to the generated depot/dosing compartment.
3. **Special functions (`src/stats/special.rs`):** `ln_gamma` via a **Lanczos** rational
   approximation (merged in #340), **not** bare Stirling: `N` is estimated continuously and
   transit `N` is commonly 1–10, where Stirling errs ~8% at N=1 / ~0.8% at N=10, enough to bias
   the absorption peak. IGD needs only `exp`/`sqrt`; Weibull needs `powf`. For exact analytic
   sensitivities (rather than FD) these must be available over `PkNum`/`Dual2` (1st- and
   2nd-order chain rules), not just `f64` — that is the gate for the §4 closed forms below.
4. **Analytical closed forms for transit and IG (Phase 3):** both use the **exponential
   tilting property** of their respective distributions (Gamma for transit, IG for IG) — see
   the `TiltedAbsorption` trait and `convolve_1cpt`/`convolve_2cpt` in the Phase 3 section
   below. Special functions required: `regularized_gamma_p(a, x)` (transit; series + CF
   expansion) and `normal_cdf(x)` (IG; via `erfc`), both in `special.rs`, written over `PkNum`
   so the closed forms inherit exact `Dual2` sensitivities (else FD for those params). Sequence:
   ship each model on the numerical ODE path first (Phases 0/1) to prove the pipeline, then add
   the closed form in Phase 3 and assert they agree under the equivalence harness.

### Analytic sensitivities for the ODE forcing

**[#430](https://github.com/FeRx-NLME/ferx-core/issues/430) — ✅ DONE for the smooth functions.**
With the Enzyme retirement, ODE models get **analytic** `Dual2` sensitivities through
`src/sens/ode_provider.rs` (state integrated as `Dual2<N>` over the generic RK45). `transit()`,
`igd()` and `weibull()` were each lifted off the FD fallback by making `PreparedInputRate::rate`
(and `prepare_dual`) generic over `PkNum`, with the `Dual2` 1st/2nd-order rules for the special
functions each calls — exact analytic forcing sensitivities, gated by
`InputRateKind::supported_over_dual()` (#468 transit, #430 igd, #498 weibull). For Weibull this is
the **only** analytic-gradient route (no closed form, permanently on the numerical ODE path); for
transit/IG it gives exact gradients independent of the Phase 3 closed forms (#386), which replace
the ODE *disposition* rather than the *forcing*.

**Open ([#530](https://github.com/FeRx-NLME/ferx-core/issues/530)) — the moving-boundary forcing (`zero_order` #504 + modeled-duration `Dn` #384).** The
smooth-function recipe does **not** cover a forcing with a hard cutoff at a parameter-dependent
time. `zero_order(dur)` cuts `R_in` off at `tad=dur`, so the true `∂/∂dur` carries a Leibniz
boundary term — a dual-channel impulse of magnitude `R_in(boundary)·∂dur/∂p` injected at `tad=dur`
— that a pointwise `rate(tad)→Dual2` cannot express (the variational walk misses it and the
analytic gradient diverges from FD for any observation after the window closes). This is the *same*
gap that keeps #384's modeled-duration `Dn` infusion on FD (`ode_provider.rs` routes every
modeled-`RATE` dose to FD via `all_doses_fixed`). It is therefore a **cross-cutting follow-up**, not
a `zero_order` rider: #504 **shipped** `dur` on FD (`InputRateKind::ZeroOrder.supported_over_dual()
=false`, merged PR #556), and **#530** builds the boundary-impulse mechanism that lifts both
`zero_order` and `Dn` off FD. Scope/feasibility against the provider's `MAX_ODE_SENS_DIM`
monomorphisation budget.

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
- **Sensitivity parity:** for any absorption param whose model is on the analytic path (the
  Phase 3 closed forms), a unit test asserting the analytic `Dual2` gradient matches central FD
  on a small fixture — in the **default** `--features ci` build (no separate autodiff job; the
  old `.max()/.min()` Enzyme-intrinsic rule no longer applies).

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
- Docs: new `docs/model-file/absorption.qmd` + `_quarto.yml`; cross-link
  `structural-model.md`; new `examples/*.ferx`; `CHANGELOG.md` (`[Unreleased] → Added`).
- `../ferx-r` follow-up PR per user-facing phase — pin bump (`tools/update-ferx-core-lock.sh`)
  + an R example/test/`NEWS.md`; the input-rate functions need no new R glue. See the
  "ferx-r follow-up" section below.

## Phasing (one PR each)

- **Prerequisite — issue #324 (NONMEM coded `RATE`), standalone first.** Safety net **✅ MERGED**
  (PR #326, 2026-06-14: rejects coded/malformed `RATE` instead of a silent bolus). Faithful
  support is a parameter-driven DSL feature: `RATE=-1` = rate modeled (`Rn`-style), `RATE=-2` =
  duration modeled (`Dn`-style) — **no `DURATION` data column**. **Modeled duration `Dn` /
  `RATE=-2` ✅ MERGED (PR #384, 2026-06-17, ODE models)** — establishes the estimated-duration
  forcing that this plan's Phase 2 zero-order family reuses. Remaining on #324: modeled rate `Rn`
  / `RATE=-1`, plus analytical-engine support for both (tracked in **#383**). Independent of this
  plan's Phase 0/1, which proceeded in parallel.
- **Phase 0a — `transit()` input-rate function. ✅ MERGED — PR #343 (2026-06-15).**
  Built-in `transit(n, mtt)` intrinsic in `[odes]`: log-domain Savic evaluator (`ln_gamma`
  shipped in #340), dose-context wiring, and the dose-routing rule — the dose feeds `R_in`, its
  bolus is **suppressed**, `∫R_in dt = F·Dose`. `R_in(tad)` is injected via the infusion
  RHS-wrapper across **all four** ODE prediction paths (`ode_predictions`, event-driven, and the
  two compartment-state variants); F scales the mass, lagtime shifts `tad`, multiple doses
  superpose, and resets turn off pre-reset doses. Not-yet-supported combinations are **rejected
  loudly** (not silently mis-modeled): SS=1 into a transit compartment (`E_ABSORPTION_SS`) and
  `transit()` + a `[diffusion]` block (`E_ABSORPTION_DIFFUSION`). Validated by parameter recovery
  (`examples/transit_savic.ferx --simulate`: TVN 3.0→3.19, MTT 1.0→0.90) + the absorption-
  independent **AUC∞ = Dose/CL** invariant. **NONMEM Savic anchor ✅ MERGED (PR #385):**
  slow-tests-gated `tests/transit_nonmem_anchor.rs` asserts FOCEI OFV −1076.67 ± 2 vs NONMEM
  −1077.13 with ODE tolerances pinned to `1e-9` (the key finding: loose default ODE tols inflate
  the FOCEI ω² — tighten toward `1e-9` for variance-component accuracy on transit/stiff fits).
- **Phase 0b — `ode_template` generation + the analytical-`pk`-plus-absorption error rule.
  ✅ MERGED — PR #363 (2026-06-16).** `ode_template NAME(...)` in `[structural_model]` lowers
  (pre-pass desugar) to a generated `[odes]` disposition from the codified analytical↔ODE
  transforms, with user `d/dt(X)` overrides (top-level only) replacing a generated equation; an
  analytical `pk` disposition combined with an ODE-only absorption model (`transit`) is a hard
  error pointing at `ode_template`. Equivalence-tested (`ode_template` ≡ `pk` for all 6 models).
  ferx-r follow-up merged (PR #169).
- **Phase 1 — inverse-Gaussian (Freijer & Post). ✅ MERGED — PR #389 (`5b257208`, closes #347).**
  Single IG via the `igd(mat, cv2)` input-rate function (log-domain density, essential
  singularity `tad→0 ⇒ R→0` for free; shipped `f64`/FD-only like `transit`, since lifted to
  analytic `Dual2` by #430); ships as **numerical** (ODE
  forcing), even though a closed form exists (the `TiltedAbsorption` route, Phase 3 below) — the
  ODE path is the same pipeline as transit and validates the forcing end-to-end before the
  analytical fast path is added. Anchored vs a NONMEM `$DES` IG run
  (`tests/igd_nonmem_anchor.rs`) at the likelihood at the shared optimum — a path-independent
  check, because default derivative-free BOBYQA stalls on the flat mis-specified ridge while
  NONMEM's gradient FOCEI climbs `MAT`. The **biphasic sum-of-two IG** is deferred to **#388** (no
  biphasic NONMEM run yet → would be an unanchored happy path; its fraction-multiplier mechanism
  is shared with the planned parallel/mixed `first_order`, so design it once).
- **Phase 2 — Weibull + zero-order + sequential + parallel + mixed.** Round out the
  catalogue; each with a NONMEM anchor. Split into tracked slices:
  - **Weibull (#497) ✅ MERGED — PR #498 (`6252946a`)** — **numerical** (no closed
    form); errors on an analytical disposition (pointing at `ode_template`). Exact
    gradients come *only* via the `PkNum`-generic forcing route (see §"Analytic
    sensitivities for the ODE forcing"), never Phase 3. NONMEM `$DES` anchor ran
    (`tests/weibull_nonmem_anchor.rs`, slow-tests-gated, OFV −943.833).
  - **`zero_order` + `sequential` (#504) ✅ MERGED — PR #556 (`4fd737e9`)** — single-input;
    reuses #324's merged `RATE=-2`/`Dn` estimated-duration forcing **for the value path**.
    **Closed-form** (superpose existing solvers / piecewise). Shipped on the ODE path with
    `dur` on **FD** — its `∂/∂dur` is a *moving boundary* (the one hard part), and the analytic
    boundary-term impulse (shared with #384's `Dn` infusion) is the cross-cutting follow-up
    #530, not bundled here. Direct NONMEM ADVAN1 anchor (`tests/zero_order_absorption.rs`).
    Then optionally accelerated via the closed form. **ferx-r follow-up done — PR #204
    (`56e7cafc`, merged 2026-06-27).**
  - **#388 — shared input-rate fraction multiplier (biphasic IG)** ← **NEXT.** Add a
    fraction scale to `InputRateForcing` (`frac_slot: Option<usize>`, applied as
    `dy[cmt] += frac * R_in` in `add_prepared_input_rate_forcing`) and relax the two Phase-1
    parser guards (`call_is_scaled_or_signed` + the one-input-rate-call-per-`d/dt` limit) to
    accept a leading **named-parameter** multiplier (`FR*igd(...)`, declared individual
    parameter only — not an arbitrary `(1-FR)` expression) and ≥2 input-rate terms per
    equation. Validate `0 < frac ≤ 1` per term and `Σfrac ≈ 1`; per-model mass balance
    `∫ Σ fracᵢ·R_inᵢ = F·Dose`. Anchored vs a new biphasic-IG `$DES` run. The same multiplier
    then serves #505.
  - **`parallel` / `mixed` + `first_order()` composition (#505)** — dual-pathway,
    **blocked on the #388 shared fraction multiplier** (the parser rejects scaled
    input-rate calls today). Adds `InputRateKind::FirstOrder` (`first_order(ka)`) so existing
    first-order absorption can be composed. **Closed-form** (superpose `*_oral` solvers by
    `frac`).
- **Phase 3 — analytical closed forms for transit and IG** (1/2-cpt). Both are implemented
  via the **`TiltedAbsorption` trait** in a new `src/pk/analytical_absorption.rs`:

  ```rust
  pub trait TiltedAbsorption {
      fn mgf(&self, k: f64) -> f64;         // E[e^(k·X)]
      fn tilted_cdf(&self, t: f64, k: f64) -> f64;  // CDF of the e^(k·t)-tilted distribution
  }
  ```

  The generic convolution with 1-cpt or 2-cpt disposition is the same for both:

  ```rust
  pub fn convolve_1cpt<A: TiltedAbsorption>(abs: &A, t: f64, ke: f64, f_dose_over_v: f64) -> f64 {
      f_dose_over_v * abs.mgf(ke) * (-ke * t).exp() * abs.tilted_cdf(t, ke)
  }
  pub fn convolve_2cpt<A: TiltedAbsorption>(
      abs: &A, t: f64, alpha: f64, beta: f64, big_a: f64, big_b: f64, f_dose: f64,
  ) -> f64 {
      f_dose * (big_a * (-alpha*t).exp() * abs.mgf(alpha) * abs.tilted_cdf(t, alpha)
              + big_b * (-beta*t).exp()  * abs.mgf(beta)  * abs.tilted_cdf(t, beta))
  }
  ```

  > **`PkNum`-generic, not `f64`:** the signatures are shown over `f64` for readability, but
  > `mgf` / `tilted_cdf` / `convolve_*` are written over `T: PkNum` (the `sens/` `*_g`
  > convention) so the closed forms inherit exact `Dual2` 1st/2nd-order sensitivities — the
  > whole point of moving transit/IG off the FD-only ODE path.

  **Transit** (`TransitAbsorption { n, mtt }`): Gamma is closed under exponential tilting.
  `mgf(k) = (KTR/(KTR−k))^(n+1)`, `tilted_cdf(t,k) = P(n+1, (KTR−k)·t)` (regularized
  incomplete gamma). Condition: `k < KTR = (n+1)/mtt`. Sanity check: n=0 recovers Bateman
  exactly (since `P(1,x) = 1−e^(−x)`).

  **IG** (`IgAbsorption { mat, lambda }` with `lambda = mat/cv2`): IG is closed under
  exponential tilting. `mgf(k) = exp(λ/μ·(1−√(1−2μ²k/λ)))`,
  `tilted_cdf(t,k) = F_IG(t; μ*, λ)` with `μ* = μ/√(1−2μ²k/λ)`, and F_IG expressed via
  the normal CDF Φ (the known IG CDF formula). Condition: `k < λ/(2μ²) = 1/(2·MAT·CV²)` —
  satisfied for virtually all PK parameters. Reference: the tilting identity is a standard
  result; the IG closed form was identified by working through the issue #322 comment thread
  (2026-06-17). The Hof & Bridge (2021) paper (doi:10.1007/s10928-020-09719-8) confirms the
  analogous result for transit.

  **Special functions needed** (`src/stats/special.rs`):
  - `regularized_gamma_p(a, x)` — series for `x < a+1`, continued fraction for `x >= a+1`;
    `ln_gamma` (Lanczos) already present.
  - `normal_cdf(x)` — `0.5 * erfc(-x / sqrt(2))`.

  **Error rule update:** after Phase 3 merges, the "analytical `pk` + absorption" hard error
  no longer applies to transit or IG — they route to `convolve_1cpt`/`convolve_2cpt`
  directly. Weibull remains an error (no closed form).

  The speed win for transit is twofold: removes the adaptive ODE solve *and* moves transit
  onto the analytic `pk` path, whose sensitivities come from `sens/`'s `Dual2` (the `*_conc_g`
  generics) — dropping the FD-gradient multiplier without any autodiff feature/CI. The NONMEM
  anchor (PR #385) quantified the gap: ~89 s release at `ode_*tol=1e-9` vs NONMEM's ~16 s.
  Assert both transit and IG closed forms match their Phase-0/1 numerical ODE forms under the
  equivalence harness.

## ferx-r follow-up (per user-facing feature)

Every user-facing feature here must reach R users through `../ferx-r` (CLAUDE.md: a
newly-`pub` ferx-core change "expects a matching PR in `ferx-r`"). The follow-up for
this plan is **light**, because the absorption input-rate functions (`transit`, `igd`,
`weibull`, `zero_order`, `first_order`) and `ode_template` are **model-file DSL/parser
features**: ferx-r hands the `.ferx` file straight to ferx-core's parser
(`ferx_core::parser::model_parser::parse_full_model_file`, ferx-r `src/rust/src/lib.rs`)
and the R layer only carries file paths. So there is **no new exported R function** to
write — the feature works from R the moment ferx-r builds against a ferx-core commit
that has it.

Each user-facing phase's ferx-r PR therefore needs:

- **Pin bump (required for CI/release availability).** Bump `ferx-r/src/rust/Cargo.lock`
  to the ferx-core commit that landed the phase, via `ferx-r/tools/update-ferx-core-lock.sh`
  — never a bare `cargo update` (the `[patch]` would unpin it). Local ferx-r builds
  already see the feature through the `[patch]`; the bump is what makes it available in
  **CI and release** builds, which use the pinned lock. Note these are DSL features, so
  ferx-r still *compiles* against a stale pin — it simply can't *parse/run* the new model
  syntax until bumped (contrast a consumed `pub` API, where a stale pin fails CI with
  `error[E0603]: ... is private`).
- **R-facing surface (required for a user-facing model).** Register an example model +
  dataset in `ferx_example()` (e.g. `transit_savic`), add a fast R test that fits it, note
  it in the relevant vignette/reference, and add a `NEWS.md` entry.
- **No R glue** for the input-rate functions / `ode_template` themselves — they are model
  syntax, not R-callable APIs.

Per-phase mapping:

| Phase | ferx-core feature | User-facing? | ferx-r follow-up |
|---|---|---|---|
| 0a | `transit()` | yes | pin bump + `transit_savic` example/test + `NEWS.md` |
| 0b | `ode_template` | yes | pin bump + example/docs + `NEWS.md` |
| 1 | `igd()` | yes | pin bump + IG example/test + `NEWS.md` |
| 2 | `weibull` / `zero_order` / `sequential` / `mixed` | yes | pin bump + examples + `NEWS.md` |
| 3 | incomplete-gamma closed form (transit) + IG closed form | **no** (internal accel, numerically identical) | none — no API or behaviour change for R |

#324's faithful `R1`/`D1` is a separate data-format feature (coded `RATE`), so its ferx-r
follow-up — a pin bump plus any R-side dose-column docs — is tracked on #324, not here.

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
- **Gradient agreement (analytic ≡ FD):** for absorption models on the analytic path (Phase 3),
  a unit test asserting the analytic `Dual2` gradient of `individual_nll` w.r.t. the absorption
  params matches the central-FD gradient to tolerance on a small fixture. Because `Dual2` is the
  default `gradient = auto` path, this runs **per-PR in the default `--features ci` build** — no
  separate autodiff job or special cadence. It still catches an analytic-only regression (e.g. a
  wrong `ln_gamma` / `regularized_gamma_p` `Dual2` rule) that an FD-only check would miss — the
  #317 failure mode, now guarded in default CI.

## Verification

- `cargo check --no-default-features --features ci`; push to CI for the test matrix.
  Coverage: `cargo +nightly llvm-cov --tests --no-default-features --features ci` to confirm
  patch ≥90% on each PR's diff.
- End-to-end smoke per phase: `ferx examples/transit_savic.ferx --data data/transit_2cpt.csv`
  → converges, `converged: true`, MTT/N estimates near the data-generating values.
- The default per-PR `--features ci` build verifies **both** the FD path and the analytic
  `Dual2` path (both compile under `--features ci`; there is no separate autodiff job). The
  mass-balance and analytic≡FD gradient-agreement tests are the fast regression backstop.

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
- **`Dual2` rules for `ln_gamma` / `regularized_gamma_p` / `powf` / `erfc`:** the Phase 3
  analytic forms need exact 1st- **and** 2nd-order dual rules for these (the FOCEI/Bayes Hessian
  uses `Dual2`); where a rule is missing or awkward, the model falls back to FD for those params
  (correct, just slower). This replaces the retired "verify Enzyme handles them" risk.
- **DSL ergonomics for multi-pathway** (`pathway = {...}`): a new inline-record sub-grammar
  with no DSL precedent — deferred in favour of repeated scalar keys for the ≤2-pathway case
  (see DSL surface).
