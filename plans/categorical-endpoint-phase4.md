# Plan: Categorical / count endpoints (Track C · Phase 4 · #760) — Binary first slice

**Status:** scoping complete, not yet started. Prerequisites in (`survival` non-Gaussian
trunk + #773 discrete-state plumbing + trunk FD-Laplace). Parent: `plans/tte-survival-markov.md`
§3.5, §8.1–8.8, §13–§16. Tracking issue: #760 (Track C). This doc scopes the **Binary/logistic**
endpoint end-to-end; ordinal / Poisson / negative-binomial are later slices of the same seam.

**Decisions locked (Teun, 2026-07-11):** dedicated `[binary_model]` block; ship **Slice 1a (fit)**
then **Slice 1b (simulate/predict/diagnostics)**; this plan persisted here.

---

## 0. The one dangerous fact

`EndpointLikelihood` (`src/types.rs:2748`) has **no `#[derive]`s**. Its *only* exhaustive match in
the whole codebase is the hand-written `Debug` impl (`src/types.rs:2771`). Every other reference
(~30 sites) is `if let Tte` / `matches!(…, Tte)` / `let-else` — all of which compile clean with a
new `Binary` variant and **silently treat it as "not TTE," contributing ZERO to the likelihood**.
The compiler protects exactly one line. **§3's must-touch checklist is the real safety net**, and
every added arm must be covered by a test or it is invisibly wrong.

Second fact: `CompiledModel.endpoints: HashMap<usize, EndpointLikelihood>` (`types.rs:3053`) is a
**TTE-only registry in production** — `parse_event_model_block` only ever inserts `Tte`; `Gaussian(_)`
is constructed only in tests. So Binary needs a **new production construction path** (parser) plus new
consumers; there is no existing non-TTE example inside the enum to copy.

---

## 1. What Binary is

- **Likelihood (§3.5):** `p = inv_logit(lp)`, `NLL = −Σ[y·log p + (1−y)·log(1−p)]`. Implement the
  numerically stable canonical form **`NLL = −Σ[y·lp − softplus(lp)]`** (`softplus(x)=log1p(exp(−|x|))+max(x,0)`)
  to avoid probability underflow at extreme `lp`.
- **Data (§15.4):** `ObsRecord::DiscreteState { time, state, cmt }`, `state ∈ {0,1}`, DV column = the
  observed 0/1 outcome. Already produced by the datareader (`datareader.rs:1919`), currently unconsumed.
- **Math builtins already exist:** `inv_logit`/`expit` (`model_parser.rs:11545`) and `logit`
  (`:11554`) are stable builtins with autodiff derivatives at all four evaluator sites. **No new math.**
  (Contrast: Poisson/NB need `lgamma`, which does *not* exist, and unknown fn-names silently become the
  identity — `model_parser.rs:11558` — so those slices carry a real trap. Binary does not; hence it is
  the correct first slice.)
- **Simulation "fits the fixed grid" (§8.8.2):** sampler only — no ODE event-location, horizon, or
  Gillespie. Binary avoids every hard simulation piece; it is closer to the Gaussian grid path than TTE.

---

## 2. Architecture

```rust
// src/types.rs — add alongside Tte (comment at :2765 already reserves the name)
pub enum EndpointLikelihood {
    Gaussian(EndpointError),
    Tte { hazard: HazardSpec, recurrence: TteRecurrence, hazard_covariates: Vec<String> },
    Binary { link: LinkFn, lp_fn: LinearPredictorFn },   // NEW
}
pub enum LinkFn { Logit }   // probit/cloglog → parse error "not yet supported" (forward-compatible)
// mirror HazardParamFn (types.rs:2679):
pub type LinearPredictorFn = Box<dyn Fn(&[f64], &[f64], &HashMap<String,f64>) -> f64 + Send + Sync>;
```

- `Binary`’s `lp_fn` is the exact analogue of `HazardSpec::Analytic::param_fn` — a closure over
  `(theta, eta, covariates)` that evaluates the `logit` expression (with `[individual_parameters]`
  pulled in per-subject via `eval_indiv_param_vars`). The **`Debug` arm at `types.rs:2771` is
  compiler-forced** — update it.

**DSL block (`[binary_model]`):**
```
[binary_model]
cmt   = 3
logit = TH_INT + TH_SLOPE*COV + ETA_I      ; log-odds of P(Y=1); θ/η/cov/individual-params
; link = logit                              ; optional; default logit
```
Endpoint-only models may omit `[structural_model]`/`[error_model]`/`[individual_parameters]` (same as
TTE-only models). `DV` is **not** in the expression namespace (it is the scored outcome, consistent
with every existing block); `TIME` is available per-record via the existing thread-local.

---

## 3. Must-touch checklist (exhaustive; file:line from code map)

### Seam 1 — declare & parse
- `types.rs:2748` add `Binary` variant; `types.rs:2771` **Debug arm (compiler-forced)**.
- `model_parser.rs` new `parse_binary_model_block` (mirror `parse_event_model_block:3685`): parse
  `cmt`, `logit`, optional `link`; build `lp_fn` reusing `assigned_vars_in_order` + `needed_indiv_stmts`
  (`:4045`) + `eval_indiv_param_vars` (`:3654`) + `parse_scalar_expression` (`:5543`) exactly as the
  hazard `param_fn` does. **Reject IOV-κ references** in `logit` (fail-loud, matching event_model at
  `:4024/:4094`). Undefined bare names: keep event_model's **lenient covariate fallback + existing
  undeclared-covariate warning** (parity), not a hard reject.
- `model_parser.rs:2648–2661` extend the insert loop to call the new block parser and
  `model.endpoints.insert(cmt, Binary{..})` (dupe-checked already).
- `datareader.rs` — populate `ObsRouting.discrete` (struct `:357`, `integer_kind` `:381`, disjointness
  `validate` `:393`) for binary CMTs; nothing outside tests does this today.
- `api.rs:853–863` `read_population_for` — extract binary CMTs into `ObsRouting.discrete` alongside the
  `tte_cmts` extraction (currently TTE-only).

### Seam 2 — likelihood dispatch (4+1 silent-fallthrough sites)
Add a `Binary` arm → `binary_data_term(records, link, lp_fn, theta, eta, covariates)`:

| # | Site | Role | scaling to mirror |
|---|---|---|---|
| a | `stats/likelihood.rs:504` `individual_nll_into_with_schedule` | FOCEI individual | **2×** NLL |
| b | `stats/likelihood.rs:725` `obs_nll_subject_from_preds` | SAEM θ M-step grad | mirror tte factor |
| c | `stats/likelihood.rs:921` `foce_subject_nll` | NLL + FD Hessian | mirror tte factor |
| d | `stats/likelihood.rs:2416` `individual_nll_iov` | IOV path | **2×** NLL |
| e | `estimation/saem.rs:436` `obs_nll_subject_into_iov` | SAEM M-step | mirror tte factor |

**`binary_data_term` returns plain NLL; each call site applies the same factor it applies to
`tte_data_term`** (some ×2 for the −2LL/OFV scale, some ×1). Do not invent a convention — match the
neighbour line. New home: `src/categorical/mod.rs`.

### Seam 3 — the FD-Laplace / non-Gaussian predicate (correctness crux)
Add `CompiledModel::has_non_gaussian()` = `has_tte() || <any Binary endpoint>`. It **must equal
`has_tte()` bit-for-bit when no Binary is present** (preserves TTE exactly). Reroute:

| Site | Currently | Change to |
|---|---|---|
| `estimation/inner_optimizer.rs:80` | `has_tte()` gates FD inner gradient | `has_non_gaussian()` |
| `estimation/outer_optimizer.rs:1193` (and `:784`) | `resolve_outer_ftol(has_tte())` | `has_non_gaussian()` |
| `sens/provider.rs:611` | `… && !has_tte()` (analytic outer gradient) | `… && !has_non_gaussian()` |

**Leave** the genuinely TTE-specific `has_tte()` calls alone (RTTE checks `api.rs:1606`, horizon
`api.rs:393/417`, `check_survival_tv_covariates:1756`, `predict_survival:9662`, RTTE warning `:2177`).

### Seam 4 — simulate & predict producers (Slice 1b)
- `SimOutcome::Category { state }` (`types.rs:2796`) has **no producer** (only a `debug_assert!(false)`
  consumer at `types.rs:2817`). Wire the fixed-grid sim inner (`run_model_simulate`, `api.rs`, the
  `ipred + σ·ε` line) to dispatch on the CMT's endpoint: `p = inv_logit(lp@η)`, draw `u~U(0,1)`, emit
  `Category{(u<p) as usize}` on the existing RNG stream.
- **Sim CSV writer must emit the integer state** for `Category` rows (today `continuous_value()`→NAN).
- `predict()` → `Prediction::CatProbs { probs: [1−p, p] }` at EBE η (new predict path; `predict_survival`
  is TTE-only). sdtab: IPRED = p, Pearson residual `(y−p)/√(p(1−p))`.

### Does NOT branch on the enum (no arm needed — confirmed)
`stats/residual_error.rs` (Gaussian-only; Binary residual produced upstream), `io/output.rs`,
`ode/predictions.rs`, `estimation/{gauss_newton,importance_sampling,trust_region,parameterization}`,
`sens/*` (except the one `provider.rs:611` helper). **IMP gets Binary for free** — it calls
`individual_nll`, which Seam 2 updates.

---

## 4. Features that must be accounted for

- **n_eta=0 fixed-effects (§16 D7):** first-class — plain logistic regression, no inner loop. Test the
  empty-Ω FOCEI/outer path (mirror `EXP_TTE_FIXED` in `tests/tte_smoke.rs`). Also the exact anchor (§6).
- **Covariates:** baseline data-column covariates + `TIME`-in-expression work in slice 1. A
  **within-subject time-varying covariate *column*** on the lp is the #741 family — **explicitly
  deferred**, documented.
- **IOV:** reject κ in the `logit` expression (D-d); still wire seams (d)/(e) so a model with IOV on
  *other* parameters includes the binary term.
- **Estimators (§13):** FOCEI ✓ (FD Hessian + log-det), SAEM ✓ (preferred), IMP ✓ (free). Pure-GN must
  **skip** Binary (Gaussian J'R⁻¹J); GN-hybrid skips its GN warm-start for non-Gaussian (§9.4).
  `method=foce` (no interaction) must route through the FD-Laplace objective (log-det included) or warn
  — never the log-det-dropping path.
- **Covariance / SEs:** free — FD-of-objective, no endpoint branch. (Watch the macOS-LAPACK #751 parity
  caveat on local runs.)
- **SAEM sigma (§16 D3):** no residual variance for binary CMTs — skip the sigma update there.
- **Gating:** all new code + tests behind `#[cfg(feature="survival")]` (the Cargo comment scopes that
  flag to "TTE/survival/**categorical**"). No new flag, no gating rethink.

---

## 5. Simulation, prediction, diagnostics (Slice 1b detail)
Per §8.8.2/8.8.5. Binary is the easy row: fixed grid, sampler only; `Prediction::CatProbs`; standardized
`(y−p)` residual. The only non-trivial wiring is the missing `SimOutcome::Category` producer + the sim
CSV writer (§3 Seam 4).

---

## 6. Anchors (most-authoritative first)

1. **Exact, always-available — fixed-effects (n_eta=0) vs R `glm(family=binomial)`.** ferx θ ≈ glm
   coefficients to ~1e-4; ferx OFV ≈ glm deviance (−2·logLik) up to the additive constant. Deterministic,
   license-free, doubles as the D7 smoke test. **This is the gate.** (base-R `stats::glm` confirmed present.)
2. **NONMEM `F_FLAG=1` LAPLACE (CLAUDE.md-required, mixed case).** Write the `.ctl`, run via Docker.
   **⚠ verify the NONMEM Docker image is actually runnable first** — `docker images` did not list
   `nonmemdocker:V0.1` in this environment. Cross-check vs **saemix** on toenail (§14.5: de Backer 1998,
   294 subj; accept OFV ±1.0, fixed FX ±15%) — **⚠ `saemix`/`HSAUR2` not installed** (install or source).
3. **SSE round-trip (Tier 3, license-free):** simulate Binary from known (θ,Ω) → fit → recover θ within
   2×SE, Ω ~15%. **Only this catches a wrong sampler** (§8.8.8); fit tests cannot.
4. **Internal consistency:** fixed-effects Binary OFV identical across FOCEI and IMP; later, a custom
   Bernoulli `[ll_model]` (#789) must match built-in Binary to 1e-10.

---

## 7. Test plan & coverage

| Tier | File | Contents |
|---|---|---|
| 1 | `src/categorical/mod.rs` inline `#[cfg(test)]` | `binary_data_term` vs hand-computed; stability at `lp=±40`; `state=2`→fail-loud; stable-softplus vs naive form |
| 2 | `tests/categorical_smoke.rs` (`#[cfg(feature="survival")] mod {…}`, mirror `tte_smoke.rs`) | parse `[binary_model]`; `fit()` maxiter=3 finite OFV (mixed **and** n_eta=0); `simulate()` + `predict()` smoke; DV∉{0,1}→error |
| 3 | `tests/categorical_convergence.rs` (survival + slow-tests) | glm exact anchor; SSE recovery; NONMEM/toenail anchor |

**Coverage nuance:** the ≥90% patch gate is measured by the CI `cargo llvm-cov --features ci,survival`
job (`ci.yml:125`), which runs Tier 1/2 but **ignores Tier-3 slow tests**. So the parser block,
`binary_data_term`, the **sim producer, and the predict path must each be hit by a Tier-1/2 test**
(hence the fast `simulate()`/`predict()` smokes in Tier 2) — not left only to the Tier-3 SSE.
`ci.yml:45` also compile-checks `--features ci,survival,slow-tests` on every PR, so Tier-3 must compile.

---

## 8. Docs / CHANGELOG / ferx-r
- New `docs/model-file/categorical.qmd` (the `[binary_model]` DSL) + `docs/estimation/categorical.qmd`
  (estimation guidance + the glm/NONMEM comparison table) → add both to `docs/_quarto.yml`.
- `CHANGELOG.md` `[Unreleased] Added` entry referencing #760.
- ferx-r follow-up (after merge + pin bump): surface `EndpointLikelihood::Binary` + `Prediction::CatProbs`,
  a `predict_categorical` wrapper, a bundled example.

---

## 9. PR slicing & deferred scope

- **Slice 1a — fit:** §3 Seams 1–3, `binary_data_term`, `has_non_gaussian()`, n_eta=0, Tier-1 +
  fit-only Tier-2, **glm exact anchor + NONMEM anchor**, docs stub, CHANGELOG.
- **Slice 1b — simulate + predict + diagnostics:** §3 Seam 4, `SimOutcome::Category` producer + sim CSV,
  `Prediction::CatProbs`, sdtab residuals, **SSE round-trip**.

**Explicitly deferred (documented, never silent):** ordinal (monotone cut-points, §3.5/§17), Poisson/NB
(needs `lgamma` added to all four evaluator sites + derivative), IOV-on-lp, TV-covariate *columns* on lp
(#741 family), probit/cloglog links, VPC tooling (ferx-r).

---

## 10. Open items / risks
- **NONMEM runnability** — confirm the Docker image before promising the F_FLAG anchor (fallback: glm +
  SSE carry correctness; saemix/toenail as secondary).
- **`has_non_gaussian()` parity** — assert it equals `has_tte()` on the existing TTE test suite so TTE
  behaviour is provably unchanged.
- **Factor-of-2 scaling** at the five dispatch sites — the most likely silent bug; pin it with a
  fixed-effects OFV that matches glm deviance exactly (any factor error shows immediately).
- **Inner saddle for logistic** (§17) — reuse the TTE saddle-detection / FD-Hessian sentinel path.
