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
density input). This plan adds a new `[absorption]` block that selects a built-in
absorption model whose input rate `R_in(t)` is computed analytically in Rust and routed
into the existing disposition machinery.

This is a **large, multi-PR feature**; the plan is phased so each model lands with its own
tests, NONMEM anchor, and docs.

## Goals / non-goals

- **Goal:** `[absorption]` block selecting transit / inverse-Gaussian / Weibull /
  zero-order / sequential / parallel / mixed, with parameters bound to
  `[individual_parameters]` (so they carry IIV, covariates, etc. for free).
- **Goal:** one engine path that works for **any** input function, reusing the ODE
  RHS-forcing mechanism already used for infusions.
- **Goal:** continuous (non-integer) N for the Savic transit model — the key thing the
  current ODE example cannot do.
- **Non-goal (this plan):** changing the analytical closed-form `pk` disposition solvers;
  they keep working unchanged for first-order/IV. Absorption models layer on top.
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

## DSL surface (decided: new `[absorption]` block)

Disposition stays on the `pk` line; absorption moves to its own block, mirroring
`[odes]`/`[error_model]`. Parameters reference names declared in
`[individual_parameters]` (exactly like `cl=CL` on the pk line):

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
2. **Disposition + forcing (`src/ode/predictions.rs`):** synthesize the linear 1/2/3-cpt
   disposition ODE internally and add `+R_in(tad)` as an appearance term into the depot (or
   central) compartment — the **same RHS-wrapper mechanism that already injects `+rate` for
   infusions** (`ode/predictions.rs` header doc: "adding `+rate` … via an RHS wrapper").
   - Observed value = `A_central / V` (reuse existing obs-compartment plumbing).
   - **SS=1:** reuse `equilibrate_ss_state` (already cycles apply-dose/integrate-II).
   - **lagtime/F:** reuse `PK_IDX_LAGTIME` (shift `tad`) and `PK_IDX_F` (scale `D`) — already
     applied to the dose, not the RHS.
   - Multiple doses / ADDL: superpose `R_in` per dose, same as discrete-dose superposition.
3. **Special functions (`src/stats/special.rs`):** add `ln_gamma` via a **Lanczos** rational
   approximation (AD-safe, following the existing `erf` A&S precedent) — **not** bare Stirling:
   `N` is estimated continuously and transit `N` is commonly 1–10, where Stirling errs ~8% at
   N=1 / ~0.8% at N=10, enough to bias the absorption peak. IGD needs only `exp/sqrt`; Weibull
   needs `powf` — both already AD-safe.
4. **Optional Phase 3 perf fast-path:** transit→1-cpt (and →2-cpt) has a closed-form
   convolution via the **lower incomplete gamma** `P(a,x)`; implementing that in
   `special.rs` lets transit skip ODE integration. Deferred — the ODE path is the robust
   baseline first.

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
- **Phase 0 — engine + Savic transit.** `[absorption]` grammar, `AbsorptionSpec`, generic
  input-fn trait, ODE-forcing plumbing, `ln_gamma`, transit end-to-end. Anchor against the
  existing `transit_2cpt` dataset and a NONMEM Savic run. Proves the architecture.
- **Phase 1 — inverse-Gaussian (Freijer & Post).** Single + sum-of-two IG; anchor vs the
  Freijer & Post paper / a NONMEM `$DES` IG run.
- **Phase 2 — Weibull + zero-order + sequential + parallel + mixed.** Round out the
  catalogue; each with a NONMEM anchor. The zero-order family reuses the estimated-duration
  forcing from #324 (Phase 2).
- **Phase 3 (optional) — analytical incomplete-gamma fast path** for transit→1/2-cpt.

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

## Open risks

- **Speed:** ODE-forcing is slower than closed forms; acceptable baseline, Phase 3 mitigates
  transit. Quantify on warfarin-sized data.
- **AD through `ln_gamma` / `powf`:** must verify Enzyme handles them (the `autodiff` CI from
  issue #281 is the gate); fall back to FD for the absorption params if needed.
- **DSL ergonomics for multi-pathway** (`pathway = {...}`): a new inline-record sub-grammar
  with no DSL precedent — deferred in favour of repeated scalar keys for the ≤2-pathway case
  (see DSL surface).
