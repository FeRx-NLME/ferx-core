# Validating `method = vi`

**Status:** Anchor A is **implemented** (see §3). Anchors B–E remain plans.
**Scope:** how we establish that ferx's variational inference is *correct*, as opposed to
self-consistent. Companion to `VI_PLAN.md` §6, which lists the tests; this document is about
what those tests can and cannot establish, and which external anchor closes each gap.

---

## 1. The problem

`VI_PLAN.md` §6 tests 1, 3–8 are **internal consistency** checks: the analytic gradient against
finite differences of the same objective, the MC KL against the closed-form KL, the family's
`sample`/`moments` against each other. They are necessary and they are cheap, but as
`src/estimation/vi/elbo_oracle.rs` says in its own header, *"those all pass if the implementation
is self-consistently wrong."*

We have exactly one check today with an external truth: the linear-Gaussian oracle
(`elbo_oracle.rs`, `VI_PLAN.md` §6 test 2). Its construction makes `log f` exactly affine in `η`
under `log_additive` error, so the true posterior is Gaussian, a full-rank `q` represents it
exactly, and both `−2 log p(y)` and the optimal `(μ, S)` are available in arithmetic.

That is a strong anchor with one structural limit: **the bound gap is zero by construction.**
The oracle therefore cannot see any bug that lives in how we behave when `q` *cannot* match the
posterior — which is every real model, and is precisely where `elbo_tightness_ratio` and the
`Ω`-collapse behaviour live.

So the validation question is not "does VI agree with NONMEM" but: *which layer of the
implementation does each candidate anchor actually reach?*

## 2. Layers, and what can reach them

| # | Layer | Anchored today by | Gap |
|---|---|---|---|
| L1 | ELBO value and `∂/∂φ`, `∂/∂(θ,σ)` at fixed inputs | FD parity (§6.1), oracle (§6.2) | none material |
| L2 | The **bound property** — `−2·ELBO ≥ −2 log p(y)`, gap `= KL(q‖p(η|y))` | **`elbo_agq_bound.rs`** (Anchor A) | closed |
| L3 | Quality of `q` — is `(μᵢ, Sᵢ)` close to the true posterior? | nothing | **open** |
| L4 | Population estimates `θ, Ω, σ` | internal recovery tests + **`emvi` cross-check** (§4.11) | soft, but no longer single-tool |
| L5 | The method as published (Janssen Fig. 2/3) | out-of-tree fold-1 run | partial |

The ranking below follows the size of the gap, not the convenience of the tool.

---

## 3. Anchor A — the bound property, against AGQ  *(implemented)*

**Where:** `src/estimation/vi/elbo_agq_bound.rs`, plus one calibration test appended to
`elbo_oracle.rs`. Five fast tests (~3.9 s) and one `slow-tests`-gated end-to-end test.

**Establishes:** L2. `−2·ELBO ≥ −2 log p(y)` per subject, with AGQ standing in for the marginal.
A violation is an unambiguous bug: unlike every estimate-to-estimate comparison, there is no
"different estimator, different bias" defence.

**The anchoring chain, and how it was verified.** `method = laplace, n_agq = N` is adaptive
Gauss–Hermite quadrature and reproduces NONMEM `LAPLACIAN` to six significant figures, so the
chain is **NONMEM → AGQ → the ELBO bound**. The middle link is now pinned directly:
`agq_marginal_matches_the_exact_linear_gaussian_marginal` compares AGQ against the closed-form
marginal on the linear-Gaussian oracle fixture and gets **machine precision** — `2.6e-16`
relative at 3 nodes, `0.0` at 5 — which also proves the two sides share the additive-constant
convention (both omit `½·n_obs·log 2π`; a mismatch would have shown as an ≈18-absolute offset).

That test deliberately stops at five nodes. Beyond that the agreement *degrades* (`4.0e-5` at 7
nodes, `1.4e-5` at 11), and it is the **fixture** failing, not the quadrature: the oracle's `η`
is additive on `CL`, so a wide grid drives `CL` toward and past zero — half-width `√2·z_max·σ_post`
reaches `1.04` against `TVCL = 1.0` at 11 nodes. Wider is not tighter there.

**Why the bound is testable without a converged fit.** `ELBO(q) ≤ log p(y)` holds for *every* `q`,
so the strongest fast test is not one converged point but a sweep of deliberately-wrong `q`s —
means displaced ±1.5 posterior SDs, variances inflated 4× and collapsed 16×. Shifts are expressed
in posterior-SD units so the sweep means the same thing on every subject.

**The tests, and what each one is for.**

| Test | Claim |
|---|---|
| `agq_marginal_matches_the_exact_linear_gaussian_marginal` | AGQ *is* the marginal, in the ELBO's constant convention |
| `elbo_never_exceeds_the_agq_marginal` | the inequality holds for the ELBO **as a formula** (own quadrature, own KL) |
| `production_elbo_never_exceeds_the_agq_marginal` | the inequality holds for **`FitResult::vi::neg_two_elbo`** |
| `monte_carlo_elbo_matches_the_quadrature_elbo` | production's MC ELBO == a deterministic GH ELBO (two-sided) |
| `the_bound_gap_grows_as_q_leaves_the_posterior` | the gap is a divergence, not just non-negative |
| `production_kl_matches_the_textbook_kl` | production's closed-form KL == the textbook formula, exactly |
| `the_quadrature_transform_reproduces_gaussian_moments_and_kl` | this module's own GH transform and hand-written KL |
| `bound_holds_at_the_converged_vi_point` (slow) | the bound at the point Adam + Polyak + closed-form `Ω` actually reach |

**Verified by mutation.**

| Mutation | bound | bridge | KL |
|---|---|---|---|
| `neg_elbo = data − kl` (KL sign flipped) | **fails** | **fails** | passes |
| `neg_elbo = data + 0.5·kl` (KL halved) | **fails** | **fails** | passes |
| `neg_elbo = 0.98·data + kl` (too optimistic) | **fails** | **fails** | passes |
| `neg_elbo = 1.02·data + kl` (too pessimistic) | passes | **fails** | passes |
| `tr(Ω⁻¹S)` → diagonal only (a no-op in 1-D) | passes | passes | **fails** |

Three findings worth carrying forward.

**The bound is one-sided.** An objective that is too *large* satisfies it more comfortably, so
the bound alone is not a complete check and is paired with the two-sided bridge test.

**A bound test cannot substitute for comparing closed forms.** The last row is why
`production_kl_matches_the_textbook_kl` exists, and it was found by mutation rather than by
design. A diagonal-only `tr(Ω⁻¹S)` is exactly correct at `n_eta = 1` and silently discards
posterior correlation in 2-D, and *neither the bound nor the bridge could see it*: the dropped
term is only `0.003`–`0.012` on the `−2` scale against a `4·SEM` band of `0.047`–`0.099`. Both
sides of that comparison have a closed form, so routing it through a Monte-Carlo ELBO was
discarding two orders of magnitude of precision for nothing.

**`elbo_never_exceeds_the_agq_marginal` survives every row**, because it evaluates the ELBO
through the test module's own quadrature and its own KL — it tests the inequality as
mathematics. `production_elbo_…` is the one that makes the claim about the number
`FitResult::vi::neg_two_elbo` reports. Keeping both localizes a failure to either the
theory/AGQ or the implementation.

**The MC tolerances are measured, not chosen.** The ELBO omits `½·n_obs·log 2π` and so can land
arbitrarily near zero, which makes a *relative* tolerance meaningless (this bit once, at a value of
3.7). Both MC tests instead replicate over seeds and compare against a `4·SEM` band — one-sided for
the bound, two-sided for the bridge — so they fail on bias and not on noise.

**Side finding: the fixture must use additive error.** Under **proportional** error the residual
variance `(f·σ)²` collapses with the prediction, so with a lognormal `η` on `CL` the data term runs
`5.2` at `η = 0` to `9.5e11` at `η = +3`, flattening at `8.4e13` where the variance floor clamps it.
`E_q` is then dominated by tail nodes with `~1e−30` weights and `~1e13` integrands, and the
quadrature ELBO came out at `1e11`. Additive error stays bounded (`11.5` → `331`) and is what the
fixture uses. This is a property of the objective rather than of the test: the ELBO's data term
genuinely is tail-dominated for a proportional error model whose prediction can approach zero. VI's
MC estimator does not notice, because at `vi_mc_samples = 8` it never draws a `4σ` tail — worth
knowing as a variance hazard, and a separate question from the bound.

**A trap for anyone extending this.** `EbeResult::h_matrix` is FOCE's `∂f/∂η` Gauss-Newton artefact,
**not** the conditional Hessian. Using `1/h_matrix` as a posterior variance mis-scaled every `q` by
an order of magnitude and produced a 39-nat bound gap. The module finite-differences
`individual_nll` at the mode instead, which is the posterior precision with the prior included.

**Use AGQ, not `methods = vi, imp`.** An importance-sampling `−2 log L` is noisy, so a bound
violation becomes ambiguous — the estimate can dip below the truth by sampling error alone. AGQ is
deterministic, which is the entire point.

**Both follow-ups are now done.**

- **2-D `η`.** A second fixture puts lognormal `η` on `CL` *and* `V` under a `block_omega` with
  correlation `0.3`, which reaches the full-rank Cholesky `φ` layout and the multivariate KL. The
  `q` sweep displaces the mean along four directions in whitened posterior space — each axis and
  both diagonals — because an off-diagonal bug in `L` is invisible to an axis-aligned
  displacement. This is what surfaced the diagonal-trace finding above.
- **`agq_eval_only`.** A `laplace` stage can now evaluate the AGQ marginal at the parameters it
  is handed and report it as `ofv`, leaving `θ`/`Ω`/`σ` alone; it must be the terminal stage.
  So `method = vi, laplace` with `agq_eval_only = true, n_agq = 21` gives users the same
  deterministic marginal this anchor tests with, and `−2·ELBO − ofv` reads the bound gap
  directly. Tests in `tests/agq.rs`; docs in `docs/model-file/fit-options.qmd`.

**Cost.** Five fast tests plus the calibration run in ~2.1 s; the slow-gated end-to-end fit adds
~5.8 s. The Monte-Carlo tests dominate, and their budget (`N_SEEDS = 3`, `N_MC = 600`) was cut
until every mutation above was still caught — the deterministic tests are ~2.4 ms per quadrature
call and essentially free.

## 4. Anchor B — nlmixr2 `est = "emvi"`  *(best external check on VI-as-VI)*

**Status:** **run** (2026-08-19), against nlmixr2est 7.0.2 from CRAN. Design in §4.2–4.5,
results in [§4.11](#411-results-first-run-2026-08-19). Tiers 1 and 3 are done; Tier 2 is done on
the ferx side and partially blocked on pinning nlmixr2's Cholesky packing convention.

nlmixr2 7.0 ships variational inference. Per its `NEWS.md`, the method formerly called
`est = "advi"` was split into `emvi` (variational **EM**) and `fbvi` (**full Bayes**), because
*"the default mode was never the published algorithm but a variational-EM hybrid."* Controls are
`emviControl()`, with `fbviControl()` as a thin wrapper.

`pointEstimate` follows `est`, so `est = "emvi"` implies `TRUE`: the variational posterior covers
the per-subject `η` only, the population parameters are point estimates, and the docs state
"output semantics match FOCEi/SAEM". That is ferx's VI.

**Establishes:** L4, and uniquely **L1 across implementations** — the only external option that
gives ELBO-vs-ELBO on a real NLME model with a matching variational family. What it cannot reach
is L3; see §4.8.

### 4.1 Correction to an earlier claim in this document

An earlier revision asserted that `emvi`'s EM M-step for `Ω` **is** ferx's closed-form maximizer
`Ω* = (1/N)Σ(Sᵢ + μᵢμᵢᵀ)`, and read that as two implementations independently converging on the
same fix. **That is not supported.** `emviControl.Rd` says the population parameters are "point
estimates *maximized by the ELBO gradient*", and the optimizer state carries `logPopOmega` — so
`Ω` is stepped by gradient in log-variance space. That is ferx's `vi_omega_update = adam`, not its
default.

The comparison survives, and the rationale arguably improves. The closed form *is* the point where
the ELBO's `Ω` gradient vanishes, so both routes target the same stationary point and a converged
`Ω` should still agree. The comparison therefore tests that two different optimization routes reach
the same optimum — a real thing to test — rather than two implementations sharing an update rule.
It also means ferx should run `vi_omega_update = adam` for the closest like-for-like, and
`closed_form` as a second arm: agreement of *both* ferx arms with `emvi` is the stronger result.

**Confirmed on the first run.** The two ferx arms agree with each other to **under 0.5% on every
parameter** (§4.11), so the closed-form maximizer and the gradient M-step do land on the same
stationary point, as the argument above predicts.

### 4.2 Option mapping

Verified against `R/emviControl.R` and `R/vi.R`.

| ferx | nlmixr2 `emviControl()` | Note |
|---|---|---|
| `vi_family = full_rank` / `mean_field` | `viFamily = "fullRank"` / `"meanField"` | dense per-subject Cholesky vs diagonal |
| `vi_mc_samples` | `nMc` | their default is **1**, ours 8 — set explicitly |
| `vi_iters` | `iters` | their default is **300**, ours 25000 |
| `interaction = true` / `false` | `likelihood = "focei"` / `"foce"` | `likelihood` selects the interaction treatment, nothing more |
| Adam | `optim = "adam"` | their default is `"advi"` |
| `vi_seed` | `seed` | |
| `vi_final_ofv = laplace` | `likelihood = "laplace"` | |
| `vi_omega_update = adam` | (emvi's only mode) | see §4.1 |
| no convergence tolerance, by design | `tol` | **set `0`** |

`likelihood` resolves to `foceiControl(interaction = ..., foce = ...)` with
`maxInnerIterations = 0`, i.e. the FOCEi inner interface is used only to *evaluate* the per-subject
log-joint and its `∂/∂η` at a drawn `η` — not to optimize. Same construction as our data term.

### 4.3 Four defaults that would confound the comparison

- **`perNoCor = 0.75`** holds declared `Ω` off-diagonals at **zero** for the first 75% of the run
  (the `saemControl()` rule). On a `block_omega` model this alone would explain a correlation
  disagreement. Set `0`.
- **`optim = "advi"`** is the Kucukelbir adaptive step-size sequence, not Adam. Set `"adam"`.
- **`tol`** early-stops on the relative ELBO change, defaulting to `10^(-sigdig)`. Set `0` to run
  all `iters`, which is closer to our ceiling-plus-settling semantics.
- **`adaptEta = TRUE`** runs a step-size search over `etaCandidates` for the first
  `min(iters, 75)` iterations. Set `FALSE` — but see §4.7, the step size that then applies is not
  obvious from the source.

`klWarmup = 0` (no prior tempering) is already the default and matches ferx having no equivalent.
Set `covMethod = ""` to skip their covariance step; under `pointEstimate = TRUE`,
`covMethod = "vi"` silently falls back to FOCEi `"r,s"`, which is not what our FD-of-OFV Hessian
computes, so standard errors are excluded from the primary comparison.

### 4.4 Pinning the constant convention empirically

Absolute ELBO values are not comparable until the additive-constant conventions are known to
match — ferx omits `½·n_obs·log 2π`, and nlmixr2 has an `adjObf` flag whose default is `TRUE`
("adjusted to be closer to NONMEM's").

Rather than reason about `adjObf`, **measure it**: run FOCEI in both tools on the same model and
data first. The offset between the two FOCEI OFVs *is* the convention difference. Apply that same
offset to the ELBO comparison.

This costs almost nothing extra and is independently valuable — a nlmixr2-vs-ferx FOCEI comparison
is most of Anchor D (§6), the standing `vi.qmd` placeholder.

**Measured: the offset is zero.** ferx FOCEI `−286.004` against nlmixr2 FOCEI `−285.947`, a
`0.06` difference attributable to optimizer endpoints. A convention mismatch would have shown as
`n_obs · log 2π = 110 × 1.8379 ≈ 202`. So nlmixr2's default `adjObf = TRUE` lands on the same
convention ferx uses, and no offset is applied anywhere in §4.11.

### 4.5 The comparison, in three tiers

**Tier 1 — converged `θ`, `Ω`, `σ`.** The headline, on warfarin. Both are point estimators of the
same objective over the same family, so there is no "different estimator, different bias" escape
hatch of the kind that softens every FOCEI/SAEM/NONMEM comparison. Run ferx twice
(`vi_omega_update = adam` and `closed_form`) per §4.1.

**Tier 2 — the per-subject `q`.** The sharp tier: `(μᵢ, Sᵢ)` is what VI uniquely produces, and
where two implementations can differ while both look healthy at the `θ` level. `returnVi = TRUE`
returns a `nlmixr2vi` object carrying `mu`, `scale` (`Lpack` under full-rank), `elbo`, `theta` and
`popOmegaMat`, so `μᵢ` and `Sᵢ = LLᵀ` are both reachable.

One risk checked and cleared, twice, because the second check nearly reversed the first. `xform`
traces to `.iterPrintXParFromUi` and is **display-only**. But the returned object also carries
`etaScale = 0.1` alongside a stored covariance that unpacks to `≈ I`, which looks exactly like `q`
living in a whitened coordinate. It does not: `src/inner.cpp:14885` defines
`rho = etaScale · i^(−1/2+ε) / (τ + √s)`, so **`etaScale` is the ADVI step size** — one of
`etaCandidates`, and incidentally the answer to §4.7's first open item. The `≈ I` reading came from
unpacking the Cholesky with a *logged* diagonal; the `grad_L(i,j) = (−lp)ᵢ epsⱼ + [i==j]/L_ii`
comment at `inner.cpp:15054` shows the diagonal is stored **raw**. So `q` is over `η` directly and
no Jacobian is needed — but the two false leads are worth knowing about.

**Tier 3 — the ELBO.** Only after §4.4. Compare ELBO *differences* between two fits (mean-field vs
full-rank on the same data), which cancels any residual offset.

### 4.6 What it costs to run

Installable but not free. nlmixr2est 7.0.2 requires `rxode2 >= 5.1.5`; this machine has **4.1.1**.
So it is an rxode2 major-version upgrade plus nlmixr2est plus `L0Learn` / `lbfgsb3c` / `minqa`, all
compiled from source — roughly 20–40 minutes, and it mutates an R environment other work may
depend on. **That upgrade is the user's decision, not a step to take unprompted.**

**Harness location (as built).** `tools/vi-emvi-comparison/` — `run.sh` (driver),
`emvi-compare.R` (nlmixr2 side: FOCEI, `emvi`, `emvi` + `returnVi`), three `.ferx` models for the
ferx side, and a README. Committed, and deliberately **not** wired into CI: no CI image here
carries an R stack and the fits take minutes.

Persistent state lives **outside** the repo at `~/.local/share/ferx-vi-validation`
(override with `FERX_VI_STATE`): `Rlib/` is ~110 MB of R binaries, which has no business in a git
tree, and `results/` is regenerable. Deleting that directory undoes the whole install; the system
R library is never touched.

### 4.7 Open items — all resolved (the last two on 2026-08-24, from nlmixr2est's source)

- ~~The step size under `adaptEta = FALSE`.~~ **`etaScale`, which defaulted to `0.1`.** It is the
  ADVI step-size constant, not a reparameterization — see §4.5.
- ~~The per-subject layout of `mu` and `Lpack`.~~ **`mu` is `N × n_eta` (10 × 3), `Lpack` /
  `scale` is `N × n_eta(n_eta+1)/2` (10 × 6), raw lower-triangular, subject order matching the
  data.** `Lpack` and `scale` are the same object under `fullRank`.
- ~~Row- vs column-major packing of `Lpack`.~~ **Row-major — read from the source (2026-08-24),
  not inferred.** nlmixr2est 7.0.2 `src/inner.cpp` draws `η = μ + L·ε` indexing
  `Lpack(s, i*(i+1)/2 + j)` for `j ≤ i` (`:15107`) and unpacks the reported covariance the same
  way, `for r, for c ≤ r: L(r,c) = Lpack(i, r*(r+1)/2 + c)` (`:15293`): row `i` starts at offset
  `i(i+1)/2`, column `j` within it. The earlier plausibility argument was right, and
  `tier2-offdiag.R` now confirms it against a third quantity — the Anchor C NUTS posterior —
  where row-major reproduces the reference correlations to 0.20 worst-case and column-major
  misses by up to **1.39**, which is outside the range a correlation can be wrong by at all.
  This unblocks the off-diagonal half of Tier 2; see §4.14a.
- ~~Whether `emvi`'s ELBO uses the sampled-KL form or the analytic split, and what scale it is
  on.~~ **Sampled, on the log scale, with two per-subject constants dropped — also read from the
  source.** `adviElboGradCoreFR` (`:15059`) reports
  `Σᵢ[−likInner0(ηᵢ) + Σₖ log|L_kk,i|] − ½·N·log|Ω|`, averaged over `nMc` draws (`:15239`).
  `likInner0` (`:1738`) is the negative log joint on the 1× scale with no `2π` constants, so
  relative to a fully-normalized ELBO this drops the `η`-prior normalizer `−(d/2)·log 2π` and the
  Gaussian entropy constant `+(d/2)(1 + log 2π)`, netting `+d/2` per subject:

      ferx_ELBO = emvi_ELBO + N·d/2        (+15 on warfarin, N = 10, d = 3)

  and ferx reports `−2·ELBO`. Both sides drop the observation constant `½·n_obs·log 2π`, which
  §4.4 *measured* rather than assumed. This is the Tier-3 blocker; see §4.14a.

### 4.8 What this anchor cannot establish

**L3.** `emvi` shares VI's approximation. Two VI implementations agreeing on an understated `Sᵢ` is
exactly what we expect if both are correct, so this is blind to approximation quality by
construction. It complements Anchor C (§5); it does not replace it.

**It is co-validation, not anchoring.** The 7.0 announcement is explicit: *"These are research
methods. They are not validated to the standard of the established estimation methods... their
interface and defaults may change or be withdrawn in a future release without a deprecation
cycle."* Independent code, independent authors, same mathematics — agreement raises confidence
substantially, but neither side is truth. Anchor A remains the only item with ground truth.

### 4.9 Two claims only this anchor can reach

- **`block_omega` structural zeros.** `fbvi` errors outright on a correlated block; `emvi`
  estimates its off-diagonals. So `emvi` is the only external check on our claim that both
  `vi_omega_update` routes preserve the structural zeros of a mixed `Ω`. Requires `perNoCor = 0`
  (§4.3), or the schedule confounds it.
- **`mean_field` vs `full_rank` bound looseness**, since the families match by name.

### 4.10 Other traps

- **Endpoints only, never traces.** Different step-size schemes and different averaging (our
  Polyak window has no counterpart) mean the paths are not comparable even when the endpoints are.
  Match the averaging convention before reading a disagreement as a bug — `vi.qmd` already makes
  the point about last-iterate noise.
- **Exclude `fbvi`.** A full-rank variational posterior over the *population* parameters is a
  different target. Its nearest ferx analogue is `method = bayes`, and comparing them conflates
  VI-vs-MCMC with implementation correctness.
- **Shared blind spot, low weight.** Both implementations hand-roll forward sensitivities rather
  than using AD, so a common *class* of error is conceivable. CLAUDE.md's `Dual2`-vs-FD parity rule
  guards this independently.
- Adjacent but out of scope: 7.0 also ships `est = "vae"` (amortized inference). Not a VI anchor.

### 4.11a Correction — the first run's central conclusion was wrong (2026-08-20)

§4.11 below is kept as written, because what it got wrong is instructive. Two of its conclusions
do not survive, and both had the same shape: **two approximations agreeing was read as evidence
they were right, when they were wrong in the same direction.**

**Retracted 1 — "the OFV gap is a property of VI."** §4.11 concluded that both implementations
landing ~10 OFV units below their own FOCEI "says it is a property of VI on this problem rather
than a defect in either." Both halves were defects.

- *ferx:* the fit stopped short at the default `vi_mc_samples = 8`. VI returned `σ = 0.014150`
  where AGQ (`n_agq = 9`) and both FOCEI implementations give `0.010565` — 34% high — reporting
  `converged: true` at an ELBO 11.4 units short of what the same model reaches with `σ` pinned at
  the correct value. The cause is the convergence rule meeting its own noise floor: it stops once
  the ELBO's drift is indistinguishable from Monte-Carlo noise, and at 8 draws that happens while
  real drift remains, leaving `σ` — the slowest coordinate — furthest from its optimum. Raising
  the draw count resolves it (`128` → **0.55 OFV**, not 11.3); starting from a fitted FOCEI point
  does not.

  > **Attribution corrected the same day.** This was first written up as a `σ`-specific
  > step-size pathology, and a closed-form `σ` M-step was implemented as the fix
  > (`vi_sigma_update`, now the default, and worth having — it is exact and removes `σ`'s own
  > `vi_lr` sensitivity). But it is **not** what closes the gap: at `vi_mc_samples = 128` the two
  > `σ` routes agree to 0.3 OFV with `adam` marginally *ahead*, and at 8 draws both are wrong.
  > The evidence that looked σ-specific — the fit walking away from a fitted FOCEI start — is
  > equally consistent with stopping short, since `σ` legitimately rises early from a `q` that
  > starts at the prior. The lesson is the same one this section is about: a plausible mechanism
  > that explains the observation is not the same as the mechanism, and the discriminating test
  > here (vary the route *and* the draw count, not just the route) was cheap and skipped.
- *`emvi`:* not converged at `iters = 2000`. Its `σ` was still falling — `0.013128` at 2000,
  `0.011914` at 8000, `0.010369` at 20000, converging on the FOCEI/AGQ value — and this
  harness had set `tol = 0`, deliberately, to match ferx having no tolerance, which removed
  the early stop that would have flagged it. The harness default is now `iters = 20000`.

**Retracted 2 — the `S₁₁` spread and its explanation.** §4.11 reported `emvi`'s per-subject
variance swinging ~4× against ferx's flat band (23.8× on the full 10 subjects) and attributed it
to ferx averaging `φ` over a Polyak window against `emvi` reporting a last iterate. That
explanation is *directionally impossible*: ferx averages `φ` and then squares, so by Jensen
`E[L²] ≥ (E[L])²` and last-iterate noise would push `emvi`'s variances **up**, not down. The
spread was under-convergence — 23.8× at 2000 iterations, **2.9×** at 20000 against ferx's 2.4×.

**What made the difference: an arbiter.** No ferx-vs-`emvi` comparison could have caught either
defect, because §4.8 is right that neither side is ground truth — and here both were off. What
settled it was a third quantity neither codebase computes: the Laplace posterior covariance
`H⁻¹` at the AGQ estimate, differentiated with `numDeriv` from the closed form both tools
evaluate. Measured against it, ferx's variational `S` was **1.75×** too wide, uniformly across
all three `η` — the signature of a scalar `σ` error, since posterior width scales with `σ²` and
`1.339² = 1.79`. That is the diagnosis the cross-tool ratio could not deliver. The reference is
now `agq_ref.ferx` in the harness, and figures 3–4 are read against it rather than against
either tool.

**Standing lesson for this document.** Co-validation localizes a disagreement; it cannot orient
one. Every tier that compares two approximations to each other needs a third quantity with a
claim to truth, or a shared error reads as agreement. Anchor A had this property and was never
in doubt; Tier 1 and Tier 2 did not.

### 4.11 Results (first run, 2026-08-19) — *superseded, see §4.11a and §4.14*

Warfarin (`data/warfarin.csv`, 10 subjects, 110 observations), 1-cpt oral, lognormal `η` on
CL/V/KA, proportional error — the `tests/vi.rs` fixture, mirrored in nlmixr2 as a mu-referenced
`linCmt()` model (`exp(log(TVCL) + η)` is the same model as `TVCL·exp(η)`). `emvi` run with the
§4.3 confounders neutralised: `optim = "adam"`, `adaptEta = FALSE`, `perNoCor = 0`,
`klWarmup = 0`, `tol = 0`, `nMc = 8`, `iters = 2000`, `likelihood = "focei"`, `covMethod = ""`.

**Tier 1 — population parameters**

| | TVCL | TVV | TVKA | ω²CL | ω²V | ω²KA | σ | OFV |
|---|---|---|---|---|---|---|---|---|
| ferx FOCEI | 0.132695 | 7.7377 | 0.8108 | 0.02859 | 0.009592 | 0.33587 | 0.010565 | −286.004 |
| nlmixr2 FOCEI | 0.132738 | 7.7385 | 0.8285 | 0.03064 | 0.010205 | 0.34216 | 0.010574 | −285.947 |
| ferx VI, `Ω` adam | 0.132686 | 7.7389 | 0.8113 | 0.02859 | 0.009587 | 0.33582 | 0.014022 | −275.300 |
| ferx VI, `Ω` closed-form | 0.132688 | 7.7388 | 0.8113 | 0.02858 | 0.009549 | 0.33558 | 0.014150 | −274.666 |
| nlmixr2 `emvi` | 0.132687 | 7.7380 | 0.8109 | 0.02917 | 0.010080 | 0.32621 | 0.013128 | −276.096 |

ferx's VI `OFV` column is `vi_final_ofv = laplace`; `emvi`'s is its `likelihood = "focei"`
objective. Both are "the FOCE-family objective evaluated at the VI estimate", so they are the
comparable pair: `−274.67` against `−276.10`.

**ferx VI vs `emvi`:** `θ` to **0.00 / 0.01 / 0.05 %**, `Ω` to **2–5 %**, `σ` to **7.8 %**.

**Tier 3 — the ELBO.** ferx reports `−2·ELBO = −274.879` (closed-form) / `−275.558` (adam) with
`elbo_tightness_ratio` `0.960` / `0.969`. Not compared against `emvi`'s `viElbo` trace, whose last
value (`120.695`) is on a visibly different scale — a sign convention and/or a per-something
normalisation that §4.7's last open item would settle. The bound direction checks out on the ferx
side: `−2·ELBO = −274.88` sits above `−286.00`, as a bound on `−2 log L` must.

> **RETRACTED (§4.11a).** The paragraph below is wrong. Both gaps were defects — a `σ`
> step-size failure in ferx and an unconverged run in `emvi` — not a property of VI.

**Both VI implementations land ~10 OFV units worse than their own FOCEI**, in the same direction
and with similar magnitude: ferx `−274.7` vs `−286.0` (11.3), `emvi` `−276.1` vs `−285.9` (9.9).
Two independent implementations agreeing on the size of that gap says it is a property of VI on
this problem rather than a defect in either. Warfarin's `σ ≈ 1 %` makes the per-subject posteriors
very tight, which is a regime where a Gaussian `q` has little room to be wrong — so the gap is
more likely about the population step than about `q`.

**Tier 2 — per-subject `q`.** Variational means agree closely (`ETA_CL / ETA_V / ETA_KA`):

| subject | ferx | `emvi` |
|---|---|---|
| 1 | 0.029863, 0.064874, 0.369859 | 0.030954, 0.063984, 0.367429 |
| 2 | 0.017454, −0.063845, −0.411915 | 0.017901, −0.064552, −0.411994 |
| 3 | −0.022714, 0.089919, 0.152181 | −0.025274, 0.092381, 0.154228 |

— worst element 3.5 %, most under 1.5 %, and this is the tier nothing else in
`VI_VALIDATION.md` can reach.

The **covariances** are where it gets interesting. `S₁₁` is packing-independent (it is `L₁₁²`
under any lower-triangular convention starting at `(1,1)`), so it can be compared without
resolving §4.7's open item:

| `S₁₁` (var of `η_CL`) | subj 1 | subj 2 | subj 3 |
|---|---|---|---|
| ferx | 2.566e−5 | 2.546e−5 | 2.894e−5 |
| `emvi` | 2.847e−5 | 1.221e−5 | 4.774e−5 |

> **RETRACTED (§4.11a).** The spread was `emvi` under-convergence (23.8× on all 10 subjects at
> `iters = 2000`; 2.9× at 20000). The Polyak-vs-last-iterate explanation below is also
> directionally impossible — averaging then squaring makes ferx's the *smaller* of the two.

Same order of magnitude throughout, but **ferx's barely moves across subjects while `emvi`'s
swings roughly 4×** on a design where every subject has the same schedule and similar information
— so a large genuine spread is not expected. The likely explanation is reporting rather than
inference: ferx averages `φ` over a Polyak window, `emvi` reports a last iterate, and a last
iterate of a stochastic optimizer carries the full gradient noise of its final step (the argument
`vi.qmd` already makes under "Why the estimate is averaged"). If that is right it is a point in
ferx's favour, but it is currently an inference from one dataset, not a measurement — confirming
it means either averaging `emvi`'s `parHist` or re-running it across seeds.

**Not concluded:** the off-diagonals, which need the row- vs column-major question settled first.

### 4.12 What this run required, beyond the plan

- **`agq_eval_only` was not needed here**, but the per-subject `q` was not reachable from the CLI:
  `FitResult::vi::eta_means` / `eta_covs` existed only on the Rust and R APIs. Tier 2 was blocked
  on ferx, not on nlmixr2. The fit YAML now emits an `eta_posterior` block keyed by subject ID
  with the full covariance — see `CHANGELOG.md` and `docs/estimation/vi.qmd`.
- **The stale release binary.** `method = vi` is rejected by `target/release/ferx` if it predates
  the branch; build with `--profile ci-fast` (release-level optimisation, no LTO) rather than
  waiting on a fat-LTO link.

### 4.13 Mixed-`Ω` arms — the two §4.9 claims (2026-08-20)

> **Re-run under the corrected settings (2026-08-20).** The tables further down were measured
> before the `σ` fix of §4.11a and with `emvi` at `iters = 2000`; the converged numbers are here.
> **Claim (a) is unchanged** — every arm on both sides still holds both structural zeros, which
> is the one verdict that could not be moved by a convergence problem. **Claim (b) changes
> character: it is no longer directional-only on the ferx side.**
>
> | converged arm | σ | OFV | `−2·ELBO` | tightness | `cov(CL,V)` | zeros |
> |---|---|---|---|---|---|---|
> | ferx FOCEI (full `Ω`) | 0.010565 | −288.946 | — | — | 0.001898 | **not held** (§4.13's asymmetry) |
> | ferx VI, `adam` | 0.010913 | −285.972 | −285.723 | 1.027 | 0.001902 | held |
> | ferx VI, `closed_form` | 0.011118 | −285.733 | −285.576 | 1.000 | 0.001900 | held |
> | ferx VI, `mean_field` | 0.011019 | −285.861 | −277.873 | 1.009 | 0.001893 | held |
> | nlmixr2 FOCEI | 0.010562 | −286.083 | — | — | 0.003080 | held |
> | nlmixr2 `emvi` fullRank | — | −284.900 | (126.171) | — | 0.001958 | held |
> | nlmixr2 `emvi` meanField | — | −284.305 | (122.968) | — | 0.001944 | held |
>
> The block covariance now agrees across tools to **3 %** — `0.001900` (ferx) against `0.001958`
> (`emvi`) — where §4.11's under-converged run had them 29 % apart. It remains a parameter whose
> SE (`0.004939`) is 2.6× its estimate, so agreement here is weak evidence either way.
>
> **Claim (b) is now measured on the ferx side.** The two family arms converged to the *same*
> parameter vector — max drift **0.89 %** across `θ`, `Ω` and `σ` (θ ≤ 0.005 %, Ω ≤ 0.37 %, σ
> 0.89 %), inside the 2 % confound threshold `emvi-compare.R` enforces — and `mean_field`'s bound
> is **7.70 units looser** (`−2·ELBO` `−277.873` against `−285.576`). Against 0.14 units at two
> different parameter vectors before, that is a measurement rather than a direction, and it is the
> first evidence for a `vi.qmd` claim that had rested on theory alone. `emvi`'s arms still drift
> 5.73 % on the fixed effects so its side stays directional, but it agrees on the sign
> (`meanField` ELBO `122.968` against `126.171`).


Added after the first run: `warfarin_block_cmp.ferx` plus `vi_block_{adam,closed_form,mean_field}.ferx`
on the ferx side, and fits 4–6 in `emvi-compare.R` on the nlmixr2 side. Same warfarin data and
structural model as §4.11; `Ω` becomes `block_omega (ETA_CL, ETA_V)` + standalone `ETA_KA`, the
shape `src/estimation/vi/run_tests.rs::mixed_omega_fixture` uses, so `Ω(KA,CL)` and `Ω(KA,V)` are
the structural zeros under test. Run them with `tools/vi-emvi-comparison/run.sh`;
`FERX_VI_DIAG_ONLY=1` skips back to the §4.11 arms only.

**Claim (a) — structural zeros. ferx side confirmed.** All three VI arms held both slots while
estimating the block covariance:

| arm | `cov(CL,V)` | `cov(KA,CL)` | `cov(KA,V)` | `ω²CL` | `ω²V` | `ω²KA` | σ | `−2·ELBO` |
|---|---|---|---|---|---|---|---|---|
| FOCEI (full `Ω`, see below) | 0.001898 | −0.030329 | 0.019807 | 0.028588 | 0.009599 | 0.336014 | — | — |
| VI, `adam` | 0.001894 | *held* | *held* | 0.028650 | 0.009608 | 0.335890 | 0.013865 | −276.451 |
| VI, `closed_form` | 0.001899 | *held* | *held* | 0.028591 | 0.009570 | 0.335933 | 0.013878 | −276.409 |
| VI, `mean_field` | 0.001890 | *held* | *held* | 0.028592 | 0.009562 | 0.335762 | 0.011863 | −276.268 |

*held* is read by absence: `io/output.rs` emits an `ETA_x__ETA_y:` entry for every pair with
`|cov| > 1e-15`, so no entry proves `|cov| ≤ 1e-15`, and the `ETA_V__ETA_CL` entry being present
proves the emitter ran.

**Claim (a) — external half confirmed.** All three nlmixr2 mixed-`Ω` fits returned both slots as
exactly `0e+00`:

| nlmixr2 arm | OFV | `cov(CL,V)` | `cov(KA,CL)` | `cov(KA,V)` | `ω²CL` | `ω²V` | `ω²KA` | σ |
|---|---|---|---|---|---|---|---|---|
| FOCEI | −286.083 | 0.003080 | 0 | 0 | 0.029420 | 0.010136 | 0.332891 | 0.010562 |
| `emvi` fullRank | −277.004 | 0.002452 | 0 | 0 | 0.029193 | 0.010089 | 0.326059 | 0.012882 |
| `emvi` meanField | −270.646 | 0.002397 | 0 | 0 | 0.033195 | 0.010913 | 0.312499 | 0.014004 |

So **both implementations hold the structural zeros, under both `Ω` routes on ferx's side and
under both families on nlmixr2's** — the claim `vi.qmd` makes is co-validated, and this is the
one claim in the document that only `emvi` could have reached.

On the block covariance itself, ferx VI gives `0.00190` against `emvi`'s `0.00245` — 29% apart,
which sounds bad and is not. ferx's own FOCEI puts an SE of `0.004939` on that covariance, i.e.
2.6× the estimate: the parameter is statistically indistinguishable from zero on 10 subjects, so
a 29% gap sits well inside the noise. The correlation it implies is `0.11` on both sides.

**A casualty of the FOCEI asymmetry below:** the §4.4 constant-convention check cannot be re-run
on this model. ferx's FOCEI fits two extra parameters here, so its `−288.946` and nlmixr2's
`−286.083` are objectives of different models. The diagonal-model convention check (§4.4) stands
and still covers the VI arms.

**A found asymmetry — ferx's FOCEI does not honour the structural zeros.** On the same
declaration it returned `cov(KA,CL) = −0.030329`, `cov(KA,V) = 0.019807`, and `n_parameters = 10`
— 3 θ + a **full** 6-element lower triangle + 1 σ, where an honoured mixed `Ω` is 8. VI, SAEM
and GN all filter on `omega.free_mask`; the outer optimizer's `pack_params` / `unpack_params`
(`estimation/parameterization.rs`) walk the whole lower triangle, and `omega_structural_zero_mask`
is consulted only by the covariance step (#243). nlmixr2's FOCEI *does* honour the block, so a
cross-tool `Ω` comparison across the `ETA_KA` row compares two different models — the FOCEI arm
is a **full-`Ω` reference**, useful for `cov(CL,V)` and for the §4.4 convention check, not for the
`KA` row. Secondary: the VI arms also report `n_parameters = 10` despite holding the zeros, so
their AIC/BIC penalise two parameters that were never estimated. Both follow from the same root
cause — `packed_len` / `pack_params` do not consult `free_mask`. **Not fixed here**; this is an
estimator change with its own test, changelog and docs obligations, and it moves `n_parameters`
(hence AIC/BIC) for every existing mixed-`Ω` FOCEI fit.

**Claim (b) — `mean_field` vs `full_rank` looseness. Directional only, on both sides.** The
predicted ordering appears in both implementations:

| | `full_rank` | `mean_field` | direction |
|---|---|---|---|
| ferx `−2·ELBO` | −276.409 | −276.268 | `mean_field` looser ✓ |
| `emvi` ELBO (tail mean of 5) | 119.248 | 114.903 | `meanField` lower ✓ |

Neither is a measurement of the family effect, because each arm runs its own M-step and the arms
converged to different `(θ, Ω, σ)`. On ferx, σ differs by **14.5%** (`0.011863` vs `0.013878`),
which is what moves `mean_field`'s Laplace OFV to `−284.145` against `−276.137`. On nlmixr2 the
drift is **8.7%** on the fixed effects and **13.7%** on the `Ω` diagonal. An ELBO bounds
`−2 log L` at the parameter vector where it was evaluated, so a gap between two different
parameter vectors attributes nothing. `emvi-compare.R` now reports the drift beside the ELBO gap
and withholds a verdict above 2%.

The drift does not even go the same way in the two tools: `mean_field`'s σ is 14.5% *below*
`full_rank`'s on ferx and 8.7% *above* it on `emvi`. Two implementations disagreeing on the sign
of the parameter shift, while agreeing on the sign of the ELBO shift, is the cleanest available
evidence that the ELBO gap here is not reading the family.

Two traps. `elbo_tightness_ratio` is **not** the looseness measure — it is `excess / Σ(dᵢ/2)`, a
health diagnostic (`elbo.rs:1067`) whose denominator is `d = 3` in both families (`mean_field`
reports `1.014`, `full_rank` `0.976`). And the one point in the comparison's favour:
`meanField`-vs-`fullRank` is an ordering *within* one implementation, so the unreconciled ELBO
scale of §4.11 Tier 3 cancels out of the difference — this is the one Tier-3 quantity that is
comparable across the two families today.

**What would actually close claim (b):** both families' ELBO evaluated at **one** parameter
vector. No CLI option reaches that; it is a Tier-1 unit test over `elbo_agq_bound.rs` in the
shape `family_tests.rs` already uses (it loops the families over a fixed fixture). That test, not
this arm, is the route.

**Not attempted.** Per-subject `q` on the mixed model — `returnVi` is deliberately off for the
`meanField` arm, since under that family `scale` is diagonal-only with an unestablished layout
(raw sd? log sd?) and §4.7's row- vs column-major question is still open for the `fullRank`
packing. Both claims here are population-level and need none of it.

### 4.14 Corrected results — both sides converged (2026-08-20)

Same data and model as §4.11. What changed is on both sides: ferx carries the closed-form `σ`
M-step (`vi_sigma_update = closed_form`, now the default) and runs at `vi_mc_samples = 128`;
`emvi` runs at `iters = 20000`. Both harness settings are now the committed defaults, and
`agq_ref.ferx` supplies the arbiter.

**Tier 1 — population parameters.**

| | TVCL | TVV | TVKA | ω²CL | ω²V | ω²KA | σ | OFV |
|---|---|---|---|---|---|---|---|---|
| **ferx AGQ `n_agq = 9`** | 0.132687 | 7.73746 | 0.81090 | 0.028592 | 0.009592 | 0.336036 | **0.010565** | **−285.977** |
| ferx FOCEI | 0.132695 | 7.73771 | 0.81080 | 0.028590 | 0.009592 | 0.335871 | 0.010565 | −286.004 |
| nlmixr2 FOCEI | 0.132738 | 7.73851 | 0.82849 | 0.030642 | 0.010205 | 0.342155 | 0.010574 | −285.947 |
| ferx VI, `Ω` closed-form | 0.132692 | 7.73795 | 0.81097 | 0.028592 | 0.009587 | 0.335997 | 0.011234 | −285.425 |
| ferx VI, `Ω` adam | 0.132692 | 7.73792 | 0.81098 | 0.028594 | 0.009591 | 0.336019 | 0.011252 | −285.395 |
| nlmixr2 `emvi` | 0.132625 | 7.73894 | 0.81124 | 0.028650 | 0.009642 | 0.335019 | 0.010369 | −284.755 |

**Every VI arm is now within 1.3 OFV of AGQ**, against 9.9–11.3 before. ferx VI vs `emvi`: `θ` to
**0.01–0.05 %**, `Ω` to **0.2–0.6 %** (was 2–5.6 %), `σ` to **7.7 %**. The two ferx `Ω` routes
agree to 0.03 OFV.

`σ` is the one parameter still worth watching, and the residual now goes the other way: ferx sits
**+6.3 %** above the AGQ value and `emvi` **−1.9 %** below it. ferx's excess is the Monte-Carlo
floor rather than the `σ` step — the convergence rule stops when the ELBO's drift falls below its
own noise, and more draws walks it down (`vi_mc_samples` 8 → 32 → 128 gives σ `0.013763` →
`0.011646` → `0.011234`).

**Tier 2a — per-subject means.** Max absolute difference over all 10 subjects: `9.8e−4` /
`1.6e−3` / `1.6e−3` for `η` CL/V/KA, against `4.4e−3` / `3.3e−3` / `2.6e−3` before — 2–3× tighter.
Read these absolutely, not relatively: several subjects have a near-zero mean where a percentage
explodes on a `1e−3` gap (id 6's `η_CL` is `−0.008`, so `22 %` relative is `1.9e−3` absolute).

**Tier 2b — per-subject variances, against the AGQ reference.** This is the tier §4.11 got
backwards, and it now has an arbiter (`H⁻¹` at the AGQ estimate, §4.11a):

| median over subjects | `η` CL | `η` V | `η` KA |
|---|---|---|---|
| ferx VI / AGQ reference | 1.122 | 1.127 | 1.128 |
| `emvi` / AGQ reference | 1.050 | 0.966 | 1.183 |
| `emvi` / ferx (cross-tool) | 0.904 | 0.852 | 1.046 |

Both implementations now track the exact reference to within ~13 %, and ferx's residual excess is
*uniform* across all three `η` — `1.063² = 1.13`, i.e. exactly its `σ` being 6.3 % high, since
posterior width scales with `σ²`. The across-subject spread is comparable too: ferx 2.4× / 1.7× /
5.0× against `emvi` 2.9× / 2.3× / 5.9×, where §4.11 reported 2.0× against 23.8×.

**The off-diagonals** were left out here pending §4.7's row- vs column-major question. That is now
closed — row-major, from nlmixr2est's source — and the comparison is in §4.14a.

**What this leaves as the honest Anchor B result.** Tier 1 parity on population parameters, now
anchored on AGQ rather than on mutual agreement; Tier 2a parity on per-subject means; Tier 2b
agreement to ~13 % against ground truth, with each tool's residual explained by its own `σ`. The
claim §4.8 makes still stands and matters more than before: this is co-validation, and the value
of the exercise turned out to be less that the two tools agreed than that a third quantity showed
where they both did not.

### 4.14a Tier 2b off-diagonals and Tier 3, both unblocked (2026-08-24)

§4.7's last two open items were the two things §4.14 could not report. Both are now settled from
nlmixr2est's own source, which turns each from an inference into a reading.

**Tier 2b, the off-diagonals.** With the packing proven row-major, the whole per-subject
covariance is comparable — and it is read against Anchor C's NUTS posterior, not against the
other tool. `Rscript tools/vi-emvi-comparison/tier2-offdiag.R`, median and worst case over the
10 subjects:

| correlation | `(CL,V)` | `(CL,KA)` | `(V,KA)` |
|---|---|---|---|
| NUTS (reference) | 0.316 | 0.189 | 0.672 |
| ferx VI | 0.319 | 0.194 | 0.670 |
| `emvi` | 0.432 | 0.185 | 0.667 |
| worst \|ferx − NUTS\| | 0.004 | 0.006 | 0.008 |
| worst \|`emvi` − NUTS\| | 0.200 | 0.200 | 0.097 |

Both implementations recover the correlation structure — including the substantial `+0.67` on
`(V, KA)`, the regime a `mean_field` `q` could not represent at all. ferx tracks the exact
posterior to **≤0.008**; `emvi` to **≤0.20**, with the caveat that matters: Anchor C fixes every
population parameter at the AGQ estimate and `emvi`'s `q` comes from its own fit, whose `σ` is
~8% away, so part of that spread is the parameter difference rather than the `q`. The
column-major reading, for contrast, misses by up to **1.39** — impossible for a correlation, and
that is what makes the packing verdict a measurement rather than a plausibility argument.

**Tier 3, the ELBO.** The conversion of §4.7 makes the two bounds comparable for the first time.
In ferx's convention (`−2·ELBO`; lower is tighter), reading `emvi`'s bound off the mean of its
last 50 iterates (per-iterate `sd` ≈ 1.71–1.75 in ELBO units, so `SE ≈ 0.25`, i.e. `±0.50` after
the `−2` scaling — the harness itself prints the tail-5 mean, which is the same number to a
quarter-unit but with 3× the standard error):

| arm | ferx `−2·ELBO` | `emvi`, converted | difference |
|---|---|---|---|
| diagonal `Ω` | −285.242 | −282.786 | `emvi` looser by **2.46 ± 0.50** |
| mixed `Ω`, `full_rank` | −285.576 | −282.875 | `emvi` looser by **2.70 ± 0.50** |
| mixed `Ω`, `mean_field` | −277.873 | −276.823 | `emvi` looser by **1.05 ± 0.48** |

The diagonal row needed one harness re-run to be readable (2026-08-24). The first pass stored only
that arm's **last iterate** — `128.347`, which at `sd` 1.71 sits ~1σ above the local mean and
converts to `−286.693`, i.e. 0.72 *below* AGQ's `−2 log L = −285.977`, which a bound cannot do.
Read off the tail mean instead (`126.393`) it lands at `−282.786`, comfortably above the marginal
and in line with the mixed arms. That was the same last-iterate trap §4.11a fell into, in the
opposite direction, and nothing about `emvi` changed: the re-run reproduced every arm's estimates
to the digit — it is seeded — and only added the trace the harness now keeps (`elbo_summary` on
the diagonal arm too).

So the Tier-3 result, on all three arms: **the two ELBOs agree on scale once the `N·d/2` constant
is applied, and ferx's bound is the tighter of the two by 1–3 units.** What this does *not* settle
is why — the arms sit at different parameter vectors (§4.13), and a bound is a bound at the vector
where it was evaluated. Tier 3 is now comparable; it is not yet attributable.

## 5. Anchor C — per-subject NUTS  *(the only route to L3)*

**Establishes:** L3, which nothing else on this list reaches. Fix `(θ̂, Ω̂, σ̂)` as data, run NUTS
per subject on `p(η | yᵢ, θ̂)`, and compare the reference posterior mean and covariance against
`vi$eta_means` / `vi$eta_covs`.

This is what converts two claims in `docs/estimation/vi.qmd` from *asserted on Janssen's
authority* into *measured here*:

- "Variational posteriors understate posterior variance... on the order of 20–25%, and it averages
  away over subjects."
- "`Ω_iov` came back about 24% low" — with the caveat the docs already make, that confounding with
  a systematic trend can move it further and in either direction.

**Tooling.** Prefer **numpyro** over Stan. A numpyro `AutoMultivariateNormal` guide over a
per-subject plate with `Trace_ELBO` and Adam is structurally our algorithm, so one model file
yields three anchors: `Trace_ELBO` vs `TraceMeanField_ELBO` maps onto our `vi_kl = mc` /
`analytic` split; `log_density` gives **pointwise** checks of the data term and `∂/∂η` at chosen
`η` (sharper than any fitted-number comparison, and it lands exactly where CLAUDE.md warns a wrong
sensitivity is silent); and NUTS on the same model gives the reference posterior. Stan's ADVI
cannot express the nested per-subject structure cleanly and is semi-deprecated in favour of
Pathfinder, so use Stan for NUTS only if at all.

**Blocker for the gradient-level comparison.** numpyro's default reparameterized ELBO retains the
score term — that is our *standard* estimator, not the path derivative. `vi_estimator` appears in
`VI_PLAN.md` §4 but **has no implementation in `src/`**. Until that gate exists we can compare
converged `(μ, S)` but not gradients, and `VI_PLAN.md` §6 test 8 (the Fig. 2 variance claim)
cannot run either.

**Do not use ferx's own `method = bayes` for this.** It shares the likelihood code, making it an
internal consistency check rather than an anchor.

### 5.1 Results (2026-08-20) — **done**

numpyro 0.21.0 / jax 0.10.2 in an isolated venv under `$FERX_VI_STATE/pyenv` (the same
arrangement as the R library: system Python untouched, deleting the directory undoes it).
Harness in `tools/vi-nuts-anchor/`.

Both sides sit at the **same** parameter vector — the AGQ estimate, every population parameter
FIXed — so the comparison is about `q` alone. That configuration is only usable since the
convergence fix of §10 item 3; before it, the ferx side stopped after 500 iterations. The `q` it
now produces is genuinely converged: `−2·ELBO = −285.924` against AGQ's `−2 log L = −285.977`, a
gap of **0.053 units**, `elbo_tightness_ratio` 1.003. NUTS: 4 chains × 20 000 draws, **0
divergences**, `r̂ = 1.00`, `n_eff` 60 000+ on every coordinate.

| median over 10 subjects | `η` CL | `η` V | `η` KA |
|---|---|---|---|
| **warfarin (11 obs/subject)** | | | |
| \|VI mean − NUTS mean\| | 1.6e−5 | 2.1e−5 | 2.7e−5 |
| VI var / NUTS var | **1.002** | **0.999** | **1.002** |
| Laplace var / NUTS var | 1.001 | 1.000 | 0.999 |
| VI var / Laplace var | 1.000 | 1.000 | 1.002 |
| **sparse variant (2 obs/subject)** | | | |
| \|VI mean − NUTS mean\| | 1.3e−4 | 8.1e−5 | 1.5e−4 |
| VI var / NUTS var | 0.999 | **0.959** | **0.962** |
| Laplace var / NUTS var | 1.000 | 0.998 | 1.000 |
| VI var / Laplace var | 0.999 | 0.964 | 0.964 |

**L3 established, in ferx's favour.** The variational posterior matches the exact posterior to
**0.2 % in variance and ~2×10⁻⁵ in mean** on warfarin. This is the tier that had no anchor with
ground truth behind it; it now does, and ferx passes it.

**The 20–25 % understatement does not reproduce on this model family — and the reason is
measured, not guessed.** The `Laplace var / NUTS var` row is the diagnostic: at **1.000** it says
the true per-subject posterior *is* Gaussian, to 0.1 %. There is nothing for a Gaussian `q` to
understate. Thinning to 2 observations a subject moves the understatement to 4 % but the posterior
stays Gaussian (Laplace/NUTS still 1.000), which makes sense in both limits — data-rich is
Gaussian by asymptotics, data-poor is Gaussian because the `N(0, Ω)` prior dominates. The
non-Gaussian regime is neither, and this 1-cpt oral model with lognormal `η` does not reach it.

So the number `vi.qmd` publishes is **out of scope for the anchoring dataset**, not refuted: it
comes from Janssen et al.'s deep compartment models, where a neural-network structural model gives
the per-subject posterior a geometry this model has no way to produce. What is now measurable and
should be documented instead: on ferx's own anchoring dataset the understatement is **≤4 %**, and
the citation's figure applies to a regime we have not reproduced.

**What this does *not* close.** The claim as applied to a DCM. Testing it needs a non-Gaussian
per-subject posterior, and neither the rich nor the sparse limit of this model produces one — so
the follow-up is a fixture chosen for posterior geometry (a nonlinear/saturating structural model,
or the Janssen fold-1 setup of §7) rather than for sparsity. `Ω_iov` ~24 % low is untouched here
too: it needs the IOV fixture, not this one.

**The off-diagonals too, not just the diagonal.** Against NUTS there is no packing ambiguity —
both covariances come from the same convention — so §4.7's open question does not apply here and
the whole matrix is comparable:

| max \|VI − NUTS\| | corr(CL,V) | corr(CL,KA) | corr(V,KA) | whole matrix (relative Frobenius) |
|---|---|---|---|---|
| warfarin | 0.0042 | 0.0062 | 0.0077 | 0.93 % median, 5.3 % worst subject |
| sparse | 0.0107 | 0.0117 | 0.0005 | 1.6 % median, 2.6 % worst |

The correlations being reproduced are not small ones — `+0.67` on warfarin's `(V, KA)` and `+0.99`
in the sparse case — which is the regime where a full-rank `q` is doing work a `mean_field` one
could not, and it recovers them to under 0.01. So the per-subject posterior is validated as a
covariance, not as three variances.

**Figures.** `Rscript tools/vi-nuts-anchor/plots.R` — the overlay of each subject's NUTS marginal
with the variational Gaussian (`posterior-overlay.png`) is the direct form of the claim; the
variance ratios and the three-ratio decomposition are the numbers above. The decomposition figure
exists so the first cannot be over-read: a small understatement measured against a Gaussian truth
is a statement about the dataset.

**Not attempted:** the gradient-level comparison, which is still gated on `vi_estimator`
(§5 blocker) having no implementation in `src/`. The `(μ, S)` comparison above is the whole
variance claim and needed none of it.

## 6. Anchor D — FOCEI / SAEM / NONMEM / nlmixr2 `focei`  *(plumbing and placement)*

**Status:** **done** (2026-08-24). Results in [§6.1](#61-results-2026-08-24); the CLAUDE.md
"compare with NONMEM" obligation for `method = vi` is discharged, and
`docs/estimation/vi.qmd` carries the filled table instead of the placeholder.

**Establishes:** L4, plus the shared predictor / dose-bookkeeping / residual-model plumbing. This
is `VI_PLAN.md` §6 test 12.

Be clear about what it buys. These estimators target the same parameters by different
approximations, so every disagreement has an escape hatch, and the `elbo_oracle.rs` argument for
why NONMEM is not the anchor applies verbatim. Its genuine uses:

- the **`Ω`-collapse** second opinion the `#dcm-omega` section already prescribes ("refit the same
  structure with FOCEI, SAEM, or a parametric covariate model");
- an honest `−2 log L` comparison via `methods = vi, imp` with `imp_eval_only` — never the ELBO;
- nlmixr2 `focei` as a second NLME engine at near-zero marginal cost once Anchor B's harness
  exists.

**Not a substitute for simulation-based recovery.** The `Ω`-collapse and variance-understatement
results are *inference* claims about the method, not implementation claims. They need recovery
over many seeds with known truth — the shape of `tests/vi_dcm_omega_recovery.rs` — and no second
tool settles them. As `vi.qmd` says of the Janssen table: one fold, one replicate, one seed cannot
distinguish that from sampling noise.

### 6.1 Results (2026-08-24)

**No NONMEM run was needed.** `tests/nonmem/warfarin_imp.ctl` chains
`$EST METHOD=COND INTERACTION` → `$EST METHOD=IMP`, and its **first** step is FOCEI on exactly the
Anchor B model — `ADVAN2 TRANS2`, lognormal `η` on CL/V/KA, proportional error, and the same
initial estimates as `examples/warfarin.ferx`. So the licensed run already committed for the IMP
anchor carries the FOCEI column too: `TABLE NO. 1` of `warfarin_imp.ext`. NONMEM is not installed
on this machine and did not have to be. (`tests/warfarin_imp_nonmem.rs` asserts `TABLE NO. 2`;
the new `tests/warfarin_vi_nonmem.rs` asserts `TABLE NO. 1`.)

| | TVCL | TVV | TVKA | ω²CL | ω²V | ω²KA | σ (SD) | OFV |
|---|---|---|---|---|---|---|---|---|
| **NONMEM FOCEI** | 0.132695 | 7.73771 | 0.810796 | 0.0285884 | 0.00959179 | 0.335880 | 0.0105651 | **−286.004219** |
| ferx FOCEI | 0.132695 | 7.73771 | 0.810796 | 0.028590 | 0.009592 | 0.335871 | 0.010565 | **−286.004220** |
| ferx AGQ (`n_agq = 9`) | 0.132687 | 7.73746 | 0.810900 | 0.028592 | 0.009592 | 0.336036 | 0.010565 | −285.977 |
| ferx VI | 0.132693 | 7.73775 | 0.810916 | 0.028592 | 0.009587 | 0.335997 | 0.011175 | −285.519 |
| nlmixr2 FOCEI | 0.132738 | 7.73851 | 0.828490 | 0.030642 | 0.010205 | 0.342155 | 0.010574 | −285.947 |

**ferx FOCEI reproduces NONMEM's FOCEI to six decimal places on the OFV** (−286.004220 against
−286.004219) and to 5–6 significant figures on every parameter. That is the whole value of this
anchor for VI: it pins the shared plumbing — predictor, dose bookkeeping, residual model, and the
objective's additive constants — so a VI disagreement cannot be attributed to any of them. Note
what it does *not* say: FOCEI is the same approximation in both engines, so this is an
implementation check, not a statement about the estimator.

VI lands on the same basin from NONMEM's own starting values: `θ` within **0.015%**, `Ω` within
**0.05%**, and its Laplace objective 0.49 units off. `σ` is again the one parameter that misses,
by **+5.8%** — the Monte-Carlo floor of §4.11a, unchanged by the external comparison.

**One finding that only this anchor could produce: `vi_iters = 25000` is not enough from a
distant start.** From NONMEM's initial estimates (`0.2 / 10 / 1.5`, `σ` 0.02) VI needs
**~34 250 iterations** to settle. At the default it stops on the ceiling with `−2·ELBO ≈ −240`
and `TVCL` **4.9%** high — a fit that reports failure honestly, but a fit no assertion should be
read against. Given the room it converges by its own criterion and lands where the table says.
The `tests/vi.rs` fixture starts at `0.13 / 8 / 1.0` and needs ~17 500, which is why every
previous run of this model missed the ceiling. Documented, default unchanged; it belongs with the
draw-count finding of §4.11a as a "VI's stopping behaviour depends on where you start it" pair.

**Test.** `tests/warfarin_vi_nonmem.rs`, three `slow-tests`-gated cases (FOCEI, AGQ, VI). FOCEI
and AGQ carry **0.1%** and **0.5%** parameter bands; VI carries 1% on `θ`/`Ω`, 15% on `σ` and 3
OFV units, and additionally asserts `−2·ELBO ≥` NONMEM's OFV — the bound property against an
external reference.

## 7. Anchor E — Janssen et al.'s Julia implementation  *(narrow; lowest priority)*

**Establishes:** L5 only — the DCM+VI combination and the published Fig. 2/3 phenomena.

`VI_PLAN.md` §6 records "Not doing: the Janssen dataset replication" (no LICENSE in ME-DCM.jl, and
the covariate sampler depends on non-public checkpoints). That was superseded in practice: the
fold-1 reproduction now in `vi.qmd` was run out-of-tree, and it hit exactly the traps the plan
predicted (`amt` 2× the simulated dose, the `S1 = 1/1000` scaling, fold files sampled with
replacement).

Three things blunt it as a *correctness* oracle: their Adam-`Ω` and fully-sampled ELBO differ from
our defaults **by design**, so ferx must run in reproduce mode (`vi_omega_update = adam`,
`vi_kl = mc`, plus `vi_estimator = standard` once implemented) or the comparison is uninformative;
the dataset cannot be vendored; and their implementation carries the instabilities we deliberately
removed, so agreement is weak evidence. Keep it for the published-figure artifact, not for core
correctness.

---

## 8. Traceability — which anchor closes which documented claim

Every row is a claim `docs/estimation/vi.qmd` currently makes on internal evidence or on
citation.

| Claim in `vi.qmd` | Rests on today | Closed by |
|---|---|---|
| `−2·ELBO` is an upper bound on `−2 log L` | oracle (zero gap) + theory | **A** |
| `elbo_tightness_ratio ≈ 1` ideal; gap `≈ ½ tr(H·S)` | heuristic, calibrated on two runs | **A** |
| Closed-form `Ω*` is the exact ELBO maximizer | FD check (§6.4) | **A**, **B** (same M-step) |
| `block_omega` structural zeros preserved, both routes | internal + **ferx arms confirmed** (§4.13) + `emvi` both families | **B** — closed |
| `mean_field` bound looser than `full_rank` when correlated | **closed**: both families' ELBO at *one* parameter vector, against the closed form `−log(1 − r²)` on the posterior precision correlation — `elbo_agq_bound.rs::the_mean_field_bound_is_looser_than_full_rank_at_one_parameter_vector` | **A** (the unit test), corroborated by **B** |
| VI understates posterior variance by ~20–25% | ~~Janssen's authority~~ **measured ≤4% here, and 0.2% on the anchoring dataset** (§5.1) | closed for this model family; the DCM regime is untested |
| `Ω_iov` ~24% low | one simulated recovery | **C** + replicates |
| `Ω` collapse driven by capacity vs distinct covariate patterns | `vi_dcm_omega_recovery.rs` | **D** + replicates (D done; the replicates are not) |
| Path derivative lowers gradient variance vs standard | cited (Roeder 2017) | needs `vi_estimator` first |
| `θ/Ω/σ` land in the right place on warfarin | **AGQ + both FOCEI + `emvi` + NONMEM FOCEI** (§4.14, §6.1) | closed |
| the per-subject `q` recovers the posterior *correlations*, not just variances | **measured against NUTS: ferx ≤0.008, `emvi` ≤0.20** (§4.14a, §5.1) | **C**, cross-checked by **B** |
| VI's OFV sits ~10 units below FOCEI's | ~~two implementations agreeing~~ **retracted** (§4.11a) | n/a — it was two defects |
| Janssen fold-1 reproduction | one out-of-tree run | **E** |

## 9. Environment status

Checked 2026-08-19 on this machine.

| Tool | Status | Needed for |
|---|---|---|
| ferx AGQ (`method = laplace`, `n_agq`) | present | **A**, and the Tier 1/2 arbiter (§4.11a) |
| R 4.5 + `rxode2`, `numDeriv`, `reticulate` | present | B, D |
| `nlmixr2est` | **7.0.2 installed** (2026-08-19) at `~/.local/share/ferx-vi-validation/Rlib`, with rxode2 5.1.6 and RcppParallel 6.2.0 — prebuilt arm64 binaries, no compilation. The system library's 4.1.1 is untouched. Re-provision with `tools/vi-emvi-comparison/run.sh` | **B** (done) |
| numpyro / jax | **numpyro 0.21.0 + jax 0.10.2 installed** (2026-08-20) in an isolated venv at `$FERX_VI_STATE/pyenv`, built from `/opt/anaconda3/bin/python3.11` (the system `python3` is 3.14 with no `numpy` and no jax wheels). Re-provision with `tools/vi-nuts-anchor/run.sh` | **C** (done) |
| NONMEM | **absent, and not needed** — Anchor D reads the FOCEI step of the already-committed `tests/nonmem/warfarin_imp.ext` (§6.1) | D (done) |
| nlmixr2est **source** (CRAN tarball, read-only) | fetched 2026-08-24 to settle §4.7's `Lpack` and ELBO conventions; not installed, not committed | B (done) |
| cmdstan / cmdstanr / rstan | absent | C (fallback) |
| Julia | absent | E |

**One machine-level prerequisite, found the hard way.** R's `FLIBS` here points at
`/opt/gfortran/lib/...` for `-lgfortran -lquadmath`, and `/opt/gfortran` does not exist — the CRAN
gfortran that this R was configured against was never installed. rxode2 compiles each model to C
at runtime, so the *link* step fails and **no** rxode2/nlmixr2 model can be built. This is
pre-existing and version-independent: the system library's rxode2 4.1.1 fails identically, so it
is not a side effect of the upgrade above.

The fix needs no `sudo` and no global change — an empty `FLIBS`, scoped to one process:

```bash
printf 'FLIBS=\n' > /tmp/Makevars.nofortran
R_MAKEVARS_USER=/tmp/Makevars.nofortran Rscript your-script.R
```

It works because macOS links with `-undefined dynamic_lookup`, so the Fortran symbols reached
through `-lRblas`/`-lRlapack` resolve from R's own bundled libraries at load time.
`tools/vi-emvi-comparison/run.sh` sets this automatically; installing the CRAN gfortran is the
alternative and needs `sudo`.

So **A is available today and needs nothing installed.** B needs an nlmixr2est upgrade; C needs a
new Python stack (roughly an afternoon).

## 10. Sequencing

1. ~~**Anchor A**~~ — **done**, see §3, including both follow-ups (2-D `η` fixture,
   `agq_eval_only`).
2. ~~**Anchor D**, filling the `vi.qmd` NONMEM placeholder~~ — **done**, §6.1. No NONMEM run was
   needed: the FOCEI step of the committed `warfarin_imp` reference *is* the column. ferx FOCEI
   matches it to six decimals on the OFV; VI lands on the same basin from NONMEM's own initial
   estimates. Test: `tests/warfarin_vi_nonmem.rs`. One finding came out of it — `vi_iters = 25000`
   stops short from a distant start (~34 250 needed), documented, default unchanged.
3. ~~**Anchor B** on warfarin~~ — **done, corrected, and now complete**: §4.11 (first run), §4.11a
   (what it got wrong and why), §4.14 (both sides converged), §4.13 (mixed-`Ω` arms), §4.14a
   (off-diagonals and Tier 3). All four follow-ups are closed:
   - ~~settle nlmixr2's Cholesky packing to finish the Tier-2 off-diagonals~~ — **row-major**, read
     from nlmixr2est's source and confirmed against the NUTS arbiter (§4.7, §4.14a);
   - ~~reconcile the two ELBO scales for Tier 3~~ — **`ferx_ELBO = emvi_ELBO + N·d/2`**, derived
     from the source, and now comparable on **all three** arms: the diagonal one took one harness
     re-run (2026-08-24) to store a tail mean rather than a last iterate, which moved it from an
     apparent bound violation to `−282.786` (§4.7, §4.14a);
   - ~~re-run the mixed-`Ω` arms of §4.13 under the corrected settings~~ — **already done**: the
     saved `emvi-results.rds` has every block arm at `iters = 20000`, which is what §4.13's
     re-run box reports. This checklist line was stale, not outstanding;
   - ~~test the Polyak-vs-last-iterate explanation for the `S₁₁` spread~~ — **dropped**: the
     spread was under-convergence and the explanation was directionally impossible (§4.11a).

   ~~Claim (b) still needs a Tier-1 fixed-parameter test rather than more harness runs (§4.13).~~
   **Landed**: `elbo_agq_bound.rs::the_mean_field_bound_is_looser_than_full_rank_at_one_parameter_vector`
   evaluates both families at one parameter vector and one posterior mode by deterministic
   quadrature, and checks the gap against the closed form `−log(1 − r²)` on the posterior
   *precision* correlation — 4–8% agreement, where substituting the covariance diagonal for the
   precision diagonal (the natural wrong `q`) reads 17.1%.
   Four code findings came out of these two anchors and are **not** validation items: the FOCEI
   structural-zero asymmetry (§4.13, **unfixed**); the VI draw-count default, where
   `vi_mc_samples = 8` stops short on warfarin (§4.11a, **documented, default unchanged**); the
   `vi_iters = 25000` ceiling, which stops short from a distant start (§6.1, **documented,
   default unchanged**); and
   VI reporting `converged: true` after 500 iterations with `elbo_tightness_ratio: 78` when all
   population parameters are FIXed (**fixed**: the parameter-stability criterion was comparing a
   constant vector against itself and overriding the objective test; the same run now goes 6250
   iterations to `−2·ELBO = −282.6` with a ratio of 1.4). `vi_sigma_update = closed_form` landed and is
   pinned by `tests/vi.rs::vi_recovers_the_agq_solution_on_warfarin`, which asserts both `σ`
   routes reach the AGQ solution.
4. **`vi_estimator`**, which unblocks `VI_PLAN.md` §6 test 8 and the Anchor C gradient comparison.
5. ~~**Anchor C**~~ — **done**, §5.1. L3 established: the variational posterior matches NUTS to
   0.2% in variance. The understatement claim turned out to be out of scope for this model family
   (its posterior is Gaussian to 0.1%), so what remains is a fixture chosen for posterior
   *geometry* rather than sparsity, plus the `Ω_iov` recovery on the IOV fixture.
6. **Anchor E**, only if the published-figure artifact is wanted in-tree.

## 11. References

- Janssen A, Bennis FC, Cnossen MH, Mathôt RAA. *Mixed effect estimation in deep compartment
  models.* J Pharmacokinet Pharmacodyn (2024) 51:797–808.
- Kucukelbir A, Tran D, Ranganath R, Gelman A, Blei DM. *Automatic differentiation variational
  inference.* JMLR 18 (2017) — the style `emvi`/`fbvi` follow, per nlmixr2's own note that neither
  is the published algorithm.
- Roeder G, Wu Y, Duvenaud DK. *Sticking the landing.* NeurIPS 30 (2017).
- nlmixr2 7.0 announcement — <https://blog.nlmixr2.org/blog/2026-08-05-nlmixr2-7/>
- nlmixr2est `NEWS.md` — <https://github.com/nlmixr2/nlmixr2est/blob/main/NEWS.md>
