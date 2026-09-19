/// SAEM (Stochastic Approximation EM) for NLME population parameter estimation.
///
/// Reference: Delyon, Lavielle, Moulines (1999) Annals of Statistics 94–128.
///            Kuhn & Lavielle (2004) ESAIM: Probability and Statistics 8:115–131.
///
/// Two-phase step-size schedule (Monolix convention):
///   Phase 1 (exploration, k ≤ K1):  γₖ = 1          — rapid basin convergence
///   Phase 2 (convergence, k > K1):  γₖ = 1/(k−K1)   — almost-sure convergence to MLE
use crate::estimation::covariate_mu_ref::GroupStepInput;
use crate::estimation::fixed_eta_gradient::{
    obs_nll_subject_grad, obs_nll_subject_grad_iov, obs_nll_subject_into_iov,
};
use crate::estimation::inner_optimizer::run_inner_loop_warm;
use crate::estimation::outer_optimizer::{pop_nll, OuterResult};
use crate::estimation::parameterization::{compute_mu_k, *};
use crate::pk::EventPkParams;
use crate::stats::likelihood::{
    individual_nll, individual_nll_into, individual_nll_iov,
    individual_nll_iov_with_scratch_and_schedule, individual_nll_prepared, iov_occasion_groups,
    IndividualNllPrep, IndividualNllScratch,
};
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::StandardNormal;
use std::collections::VecDeque;

/// NLopt algorithm used for the SAEM M-step (non-mu-ref thetas + sigma).
///
/// BOBYQA was chosen over the prior SLSQP after the Emax PKPD benchmark
/// showed SLSQP locking onto one side of the Emax-Hill identifiability
/// ridge while BOBYQA's quadratic trust-region exploration landed much
/// closer to truth at ~40% lower wall (no FD-gradient eval per parameter).
/// On simpler PK-only models the two are numerically equivalent
/// (|ΔOFV| < 0.1) and within measurement noise on wall.
///
/// Exposed pub(crate) so the unit test can pin the choice across refactors.
pub(crate) const MSTEP_NLOPT_ALGORITHM: nlopt::Algorithm = nlopt::Algorithm::Bobyqa;

/// Initial BOBYQA trust radius of the numerical θ/σ M-step, in packed
/// (log / identity) units — a 10 % move (#1415).
///
/// The M-step is a *warm-started local* solve: it starts at the current
/// estimate and is expected to return the nearby maximiser of the η-frozen
/// conditional likelihood. Left to NLopt's default, the first design is a
/// quarter of the bound range — 2.5 log units for a theta declared on
/// `(0.01, 200)`, 3.25 for σ on its `[-8, 5]` box — so BOBYQA's first `2n+1`
/// interpolation points sit at ×12–×26 of the current value, its quadratic
/// model is built on garbage, and the few trust-region steps the budget allows
/// go nowhere. Measured on the thiotepa model of #1415 (11 free coordinates):
/// the returned point improved the conditional objective by 0.05–6 units per
/// M-step where a converged solve from the same point improved it by 13–224,
/// and several coordinates came back with a displacement of exactly zero.
/// With a 0.1 radius and [`MSTEP_FTOL_REL`], the same budget reproduces a
/// 4000-evaluation solve to 4 decimals on the melphalan model (3 free
/// coordinates) and captures 50–90 % of the achievable gain on thiotepa; a
/// 3× budget on top of that bought nothing (IS −2 log L 5835.0 against
/// 5827.3, at 2.3× the wall time), so the budget is left alone.
const MSTEP_INITIAL_STEP: f64 = 0.1;

/// Initial BOBYQA step for one packed coordinate: [`MSTEP_INITIAL_STEP`],
/// bounded by a quarter of the coordinate's own interval.
///
/// BOBYQA requires `upper - lower >= 2 * step` on every free coordinate and
/// otherwise rejects the *whole* problem with `NLOPT_INVALID_ARGS` before the
/// first evaluation (#1420 review). A quarter of the width is NLopt's own
/// default and leaves the factor-of-two margin; a pinned coordinate
/// (`upper == lower`) is eliminated by NLopt before BOBYQA runs and keeps the
/// nominal step, which only has to be positive. An unbounded coordinate
/// (`+∞` width) keeps the nominal step too.
fn mstep_initial_step(lower: f64, upper: f64) -> f64 {
    let width = upper - lower;
    if width > 0.0 && width.is_finite() {
        MSTEP_INITIAL_STEP.min(width / 4.0)
    } else {
        MSTEP_INITIAL_STEP
    }
}

/// Relative objective tolerance of the numerical θ/σ M-step (#1415).
///
/// Was `1e-4`: on a conditional objective of a few thousand units that stops
/// the solve as soon as an iteration gains less than ~0.1, which with the
/// default first design above was almost every iteration — 300 of 304 M-steps
/// on melphalan ended `FtolReached` after moving a fraction of the distance a
/// converged solve moves. `1e-7` is 1e-3 units on a 1e4 objective, below
/// anything the SA blend can resolve, and the `maxeval` budget still bounds
/// the work.
const MSTEP_FTOL_REL: f64 = 1e-7;

/// Relative objective tolerance of the numerical θ/σ M-step **under a mixture
/// model** — the pre-#1415 value, kept deliberately, together with NLopt's
/// default first design (no `set_initial_step`).
///
/// A `MIXNUM`-switched typical value is estimated by this solve from a hard
/// class draw (#996), and the NONMEM-anchored seed sweep in
/// `tests/mixture_nonmem.rs` was calibrated on the partial step that
/// configuration returns. Measured on that anchor (10-seed mean, NONMEM FOCEI
/// MLE `TVCL2` 2.842, `p(1)` 0.5, IMP marginal 300.87): the #1415
/// configuration lands `TVCL2` at 3.23; the local first design alone at 3.09;
/// the 1e-7 tolerance alone moves `p(1)` to 0.44 and the IMP marginal to
/// 306.9. An exact per-class M-step on a hard class draw is a different
/// estimator (classification EM, not EM), so the mixture channel keeps the
/// configuration its anchor measured until it gets its own schedule and its
/// own anchor — the same reason `damps_numerical_mstep` vetoes the damping for
/// mixtures.
const MSTEP_MIXTURE_FTOL_REL: f64 = 1e-4;

// ---------------------------------------------------------------------------
// SAEM state
// ---------------------------------------------------------------------------

/// Positive-definite floor for free BSV Ω diagonals in the M-step.
///
/// Larger than the IOV floor (1e-8) because the BSV MH proposal scale is
/// `step_scale · chol(Ω)`: if a diagonal is allowed near zero the proposal for
/// that η collapses and the chain can no longer move it, so Ω must stay large
/// enough to keep the random walk alive. 1e-6 keeps a free η explorable while
/// being far below any plausible estimated variance.
pub(crate) const SAEM_OMEGA_DIAG_FLOOR: f64 = 1e-6;

/// Target acceptance rate for the componentwise (1-D) eta kernel. The optimal
/// scaling result for single-coordinate random-walk Metropolis is ≈0.44
/// (Roberts & Rosenthal 2001), higher than the block kernel's 0.40 target.
const CW_TARGET_ACCEPT: f64 = 0.44;

/// Clamp on the per-subject **block** MH step scale `δ_i`, shared by both
/// adaptation rules so the two cannot drift apart (issue #1444).
const MH_BLOCK_SCALE_MIN: f64 = 0.01;
/// Upper end of [`MH_BLOCK_SCALE_MIN`]'s clamp.
const MH_BLOCK_SCALE_MAX: f64 = 5.0;
/// Clamp on the per-subject, per-η **componentwise** MH step scale.
///
/// The floor is 1e-6 rather than the block kernel's 0.01 to accommodate a
/// near-deterministic η (e.g. a FREM covariate η) whose posterior SD is orders
/// of magnitude below `√Ω_jj`.
const MH_CW_SCALE_MIN: f64 = 1e-6;
/// Upper end of [`MH_CW_SCALE_MIN`]'s clamp.
const MH_CW_SCALE_MAX: f64 = 5.0;

/// One step of the legacy [`ScaleAdaptation::Interval`] rule: a fixed ×1.1 if
/// the window's acceptance is above `target`, ×0.9 otherwise, clamped to
/// `[lo, hi]`.
///
/// Extracted so that the rule has **one** implementation: `run_saem` applies it
/// to the block and componentwise scales and the tests measure its reach
/// against the same code, rather than against a re-spelling of it that could
/// drift (issue #1444).
fn interval_scale_update(scale: f64, rate: f64, target: f64, lo: f64, hi: f64) -> f64 {
    if rate > target {
        (scale * 1.1).min(hi)
    } else {
        (scale * 0.9).max(lo)
    }
}

/// Robbins-Monro exponent for the step-scale adaptation: `c · k^-RM_SCALE_EXPONENT`.
///
/// Must lie in `(0.5, 1]`. That is the standard diminishing-adaptation window:
/// the steps have to vanish (so the adaptation dies out and the ergodicity
/// argument of Roberts & Rosenthal (2007) applies) without being so short that
/// the scale stops correcting before it arrives.
const RM_SCALE_EXPONENT: f64 = 0.6;

/// Gain on the Robbins-Monro step-scale update.
///
/// With `accept − target` in roughly `[-0.44, 0.56]` this moves `log δ` by at
/// most ~0.56 on iteration 1 and ~0.04 by iteration 100 — fast enough to find
/// the scale during exploration, small enough not to fight Ω afterwards.
const RM_SCALE_GAIN: f64 = 1.0;

/// One Robbins-Monro step on a log step-scale, clamped to `[lo, hi]`.
///
/// `k` is the 1-based SAEM iteration, so the first step is
/// `gain · (rate − target)` and the sequence decays as `k^-0.6`. `rate` is the
/// current iteration's own acceptance rate for the kernel being adapted.
///
/// This is the [`ScaleAdaptation::RobbinsMonro`] alternative to a fixed
/// ×1.1 / ×0.9 applied every `adapt_interval` iterations. The difference that
/// matters is not the schedule but the *reach*: over a 400-iteration run the
/// interval rule fires 8 times by a fixed factor, so `δ` can move by at most
/// ≈2.1× up or ≈0.43× down however far off the chain is. A model needing a 10×
/// smaller step never gets one (issue #1444).
fn rm_scale_update(scale: f64, rate: f64, target: f64, k: usize, lo: f64, hi: f64) -> f64 {
    let step = RM_SCALE_GAIN * (k as f64).powf(-RM_SCALE_EXPONENT) * (rate - target);
    (scale.ln() + step).exp().clamp(lo, hi)
}

/// Combined (block + componentwise) post-burn-in MH acceptance rate below which
/// SAEM appends a "sampler is not mixing" warning to `FitResult.warnings`
/// (issue #895). A genuinely stuck E-step never updates the ETAs, so the M-step
/// runs on degenerate sufficient statistics and the estimates are unreliable.
const SAEM_MH_STUCK_ACCEPT: f64 = 0.01;

/// How many trailing post-burn-in iterations the acceptance-rate diagnostic
/// looks at. The *tail* is the part a user can act on: the scales adapt over
/// the run, so a whole-run average of a chain that started badly and recovered
/// describes neither half.
const MH_RATE_WINDOW: usize = 100;

/// Fewest post-burn-in iterations the tail diagnostic will speak on.
///
/// This is the *statistical* floor: below it the tail is too short for its
/// mean to describe a settled rate at all. It is **not** what establishes that
/// the step scales have had a chance to adapt — that is
/// [`MhRateWindow::adapted_throughout`], and the two are separate gates
/// answering separate questions (#1451 review). A 45-iteration run at
/// `omega_burnin = 20` reaches this floor exactly while the interval rule,
/// which fires once per `adapt_interval = 50`, has not run once.
const MH_RATE_MIN_WINDOW: usize = 25;

/// Acceptance band, over the last [`MH_RATE_WINDOW`] iterations, outside which
/// SAEM reports that the E-step never reached its target rate (issue #1444).
///
/// Deliberately wide. The optimal-scaling targets are 0.40 (block) and 0.44
/// (componentwise), and a fit at 0.25 or 0.60 is unremarkable — this is not a
/// band of "good" rates, it is the range outside which the step scales have
/// demonstrably failed to find their target and the user should know. The lower
/// bound sits well above [`SAEM_MH_STUCK_ACCEPT`] because a chain at 2-4% is not
/// "stuck" by that 1% test yet still wastes most of the evaluations it pays for.
const MH_RATE_LOW: f64 = 0.10;
/// Upper end of the band described on [`MH_RATE_LOW`].
const MH_RATE_HIGH: f64 = 0.80;

/// Maximum growth of a free PK-residual σ during a SAEM run, in natural-log
/// units, when `iiv_on_ruv` is active (issue #895). The IIV-on-RUV
/// parameterization writes the residual variance as `σ²·exp(2·η_RUV)`, so the
/// *marginal* residual is a function of `σ²·exp(2·ω_RUV)`: σ and ω_RUV trade off
/// along a ridge. If the E-step mixes poorly the M-step can ride that ridge until
/// σ hits its e⁵ ceiling. Capping σ's growth to e³ ≈ 20× its initial SD keeps a
/// genuinely ill-posed run bounded near sensible values while leaving ample room
/// for a well-posed fit to correct a modest starting guess. The cap only ever
/// *tightens* the existing σ upper bound and only for the RUV-scaled residual
/// σ(s) — a FREM EPSCOV (always FIX) is untouched.
const SAEM_RUV_SIGMA_LN_GROWTH: f64 = 3.0;

/// Maximum growth of the `iiv_on_ruv` Ω *variance* during a SAEM run, in
/// natural-log units (issue #895). The other half of the σ × ω_RUV ridge: with
/// both free, the residual-error IIV variance can run away symmetrically to σ
/// (observed ω_RUV → ~49 in the original report). Capping ω_RUV's growth to
/// e³ ≈ 20× its starting variance bounds a genuinely ill-posed run near sensible
/// values while never binding on a well-posed fit. Applied as a
/// correlation-preserving rescale of the RUV row/column so a block Ω stays
/// positive-definite; a FIXed RUV Ω is untouched.
const SAEM_RUV_OMEGA_LN_GROWTH: f64 = 3.0;

/// Maximum per-iteration stochastic-approximation step for the Ω sufficient
/// statistic *during the exploration phase*. The θ/σ M-step uses the full γ
/// (1.0 in exploration), but Ω is averaged at no more than this rate so a single
/// un-equilibrated MCMC draw cannot overwrite a correlated Ω and trigger the
/// rank-1 collapse feedback. In the convergence phase the cap is lifted and Ω
/// uses the full decaying γ = 1/(k−k1), the same Robbins-Monro schedule as θ.
const OMEGA_SA_MAX_STEP: f64 = 0.1;

/// Maximum per-iteration stochastic-approximation step for the **residual σ**
/// half of the numerical θ/σ M-step, in *both* phases (#1445).
///
/// Ω has [`OMEGA_SA_MAX_STEP`], and a single free additive or proportional σ has
/// the averaged sufficient statistic of [`update_scalar_residual_sse`]. Every
/// other residual channel — `combined()`, per-endpoint, `block_sigma`,
/// magnitude-scaled, and any single σ in a model with a free numerical θ — rides
/// [`theta_sigma_mstep_light`], whose result was *assigned* outright in both
/// phases once [`MSTEP_SA_MAX_STEP`] defaulted to off (#1415). That left it as
/// the one SAEM statistic with no stochastic approximation at all, and the
/// reported σ was one draw of the M-step maximiser's sampling distribution
/// rather than its average.
///
/// For a well-identified σ that hardly matters — the maximiser barely moves. For
/// a **minority variance component** it decides the answer: on the #1445 repro
/// (300 subjects, median 1 observation each, block Ω(3), `combined(0.13, 1.8)`)
/// the per-M-step σ_add maximiser swings between 0.0087 and 2.34 with median
/// 0.035 against a truth of 1.8, because a weakly identified variance component
/// has a boundary-heavy sampling distribution. Whatever the last iteration drew
/// is what the fit reports: three seeds returned 0.011, 0.0025 and 0.011.
///
/// Capping γ at this value **in both phases** is what makes the estimate an
/// average rather than a draw:
///
/// * Exploration (`γ = 1`) would otherwise be pure assignment, and σ feeds
///   straight back into the next E-step's posterior — the same feedback the Ω
///   cap exists to break, here on the residual side.
/// * The first convergence iteration (`γ = 1/(k−k1) = 1` at `k = k1+1`) would
///   otherwise overwrite the whole exploration average with a single draw. Ω
///   accepts that hand-off deliberately, because `(1/N)Σηηᵀ` over N subjects is
///   a well-determined statistic; the σ maximiser of a minority component is
///   not, so σ does not get the same treatment. The cap only binds for the first
///   five convergence iterations — beyond `k − k1 = 5`, `1/(k−k1) < 0.2` and the
///   full decaying Robbins-Monro schedule takes over unchanged, which is what
///   the SA estimate needs to settle.
///
/// **The value is measured, not inherited from [`OMEGA_SA_MAX_STEP`].** A cap is
/// a trade: it is what makes the estimate an average, and it is also what makes
/// σ *lag*, since a σ that has to travel from its initial estimate now does so
/// at a bounded rate and is then averaged over a trajectory that was still
/// moving. Swept at 0.1 / 0.15 / 0.2 / 0.3 over 8 seeds each (release-equivalent
/// `ci-test`), against the three anchors this change has to satisfy at once:
///
/// | cap | #1445 fixture σ_add, truth 1.8 | cefepime σ_add / Laplace OFV, FOCEI 1.837 / 4311.7 | `covmuref_power` numeric `TH_WT`, NONMEM 0.9213 |
/// |---|---|---|---|
/// | off (pre-#1445) | 0.0025 – 0.011 (the floor) | ~1e-3 / 4337.0 | 0.9612 |
/// | 0.1  | 1.52 [1.03, 1.72] | 1.22 [1.14, 1.32] / 4348.1 | 0.8299 |
/// | 0.15 | 1.46 [1.17, 1.72] | 1.16 [0.94, 1.30] / 4337.0 | 0.9499 |
/// | 0.2  | 1.42 [1.11, 1.75] | 1.16 [0.97, 1.34] / **4331.7** | 0.9026 |
/// | 0.3  | 1.42 [1.23, 1.62] | 1.14 [**0.47**, 1.44] / 4332.0 | 0.8821 |
///
/// 0.3 is rejected on the left tail: one cefepime seed comes back at 0.47, a
/// partial collapse, which is the failure this change exists to remove. 0.1 is
/// rejected on the lag: it is the only cap whose cefepime objective is *worse*
/// than the un-averaged baseline (4348.1 against 4337.0, with σ_prop stuck at
/// 0.172 against FOCEI's 0.137), and it is the worst of the four on the #1415
/// NONMEM anchor. 0.2 is the one value that improves every anchor at once —
/// including both of the ones that were *not* broken: the cefepime objective
/// (4331.7, 5.3 better than the baseline) and `TH_WT` (0.9026, |Δ| 0.019 against
/// NONMEM where the undamped channel realises 0.040).
///
/// It is deliberately not a fit option, because it restores the Robbins-Monro
/// averaging every other SAEM statistic already has rather than adding a knob.
const SIGMA_SA_MAX_STEP: f64 = 0.2;

/// Default exploration-phase cap on the stochastic-approximation step for the
/// **numerical θ/σ M-step** — the `mstep_damping` fit option — and, since
/// #1415, **off** (`1.0`): the numerical maximiser is assigned outright in both
/// phases, exactly as it was before #1011.
///
/// #1011 introduced a 0.03 cap after a FREM `iiv_on_ruv` reprex whose
/// absorption-split fraction `TVFRD1` (no ETA) drifted from 0.383 to 0.039
/// under the undamped update, and read the cap's 0.290 as "landing on the
/// marginal optimum". #1415 measured what that cap actually does, with the
/// M-step solver itself repaired ([`MSTEP_INITIAL_STEP`], [`MSTEP_FTOL_REL`]):
///
/// * **It is a hold, not an estimate.** On the same reprex the capped fit ends
///   where it starts: 0.383 → 0.361, and from a deliberately wrong 0.2 start,
///   0.2 → 0.191. Its "good" answer was the model's initial estimate, which
///   that auto-generated FREM model had taken from a FOCEI fit.
/// * **It freezes every other no-ETA theta the same way**, which is the
///   #1415 report. The cap divides the number of EM steps the exploration
///   phase amounts to (50 M-steps × cap), and a covariate slope confounded
///   with an ETA needs all of them. `covmuref_power` with `TH_WT` routed onto
///   the numerical channel (NONMEM SAEM 0.921, start 0.3): cap 0.03 → 0.323,
///   0.1 → 0.340, 0.3 → 0.403, off → 0.961. Thiotepa (`ferx-testdata`,
///   eight no-ETA thetas, importance-sampled −2 log L, optimum 5824): cap
///   0.03 → 5928, 0.1 → 5897, 0.3 → 5860, off → 5827. Melphalan and
///   clofarabine order the same way.
/// * **The convergence-phase Robbins-Monro blend hurts too.** `γ = 1/(k−k1)`
///   is the right average for a statistic that is stationary around the
///   optimum, but the sequence of η-frozen maximisers is an EM trajectory,
///   not noise around a fixed point; averaging it weights the early, still
///   crawling iterates most (`TH_WT` 0.637 with the blend, 0.961 without).
///
/// The FREM drift is real and is *not* fixed by this: undamped, that reprex
/// settles near `TVFRD1` 0.18 from either start on the default schedule and
/// keeps drifting on a 3× longer one (0.032), and 200 MH steps per iteration
/// do not change it — so it is not plain E-step under-mixing. The
/// importance-sampled objective cannot rank the two points on that 12-eta
/// model (ESS/K = 0.001); the FOCEI objective can, and the drifted fit is 47
/// units worse than the held one (7492.7 against 7445.3). It is tracked as
/// #1421; `mstep_damping = 0.03`
/// remains available as the documented hold for that shape of model, and its
/// gate and no-effect warnings are unchanged.
///
/// `cap >= 1.0` is the "off" sentinel of [`mstep_sa_step`] — assignment in
/// both phases — which is now the default for every model except one shape,
/// see [`default_mstep_damping`].
const MSTEP_SA_MAX_STEP: f64 = 1.0;

/// The #1011 exploration cap, kept as the default for a model with
/// `iiv_on_ruv` — the one shape on which the undamped numerical channel was
/// measured to drift, and exactly the shape #1011 fixed.
///
/// The drift is the `iiv_on_ruv` coupling, not FREM and not the no-ETA channel
/// as such: the #1011 reprex with its `iiv_on_ruv` line and `ETA_RUV` omega
/// removed, otherwise identical (475 subjects, FREM block of 11), lands undamped
/// at `TVFRD1` 0.414, `TVMAT` 2.680, `TVV` 137.6 — on NONMEM IMP's 0.394 /
/// 2.680 / 133.8 — where the 0.03 cap leaves it at 0.342 / 2.228 / 147.5. With
/// `iiv_on_ruv` back in, undamped drifts to 0.18 from either start (47 FOCEI
/// units worse than the capped 0.36), 200 MH steps per iteration do not change
/// it, and #1011 had already ruled out sampler mixing and σ. Under `iiv_on_ruv`
/// the E-step is modified — `η_RUV` is re-centred into σ every iteration (#904)
/// — and a σ that shares the numerical M-step with the no-ETA thetas is what
/// that channel's σ-side of the blend acts on; that is where the pathology
/// lives, and it needs its own fix (#1421). Until then this cap holds those
/// thetas near their start, which on that reprex is the better answer.
const MSTEP_SA_MAX_STEP_IIV_ON_RUV: f64 = 0.03;

/// Default `mstep_damping` for a fit that did not set one (#1415).
///
/// Off ([`MSTEP_SA_MAX_STEP`]) unless the model has `iiv_on_ruv`, which keeps
/// the #1011 cap ([`MSTEP_SA_MAX_STEP_IIV_ON_RUV`]). A value the user sets wins
/// either way, and the gate and no-effect warnings in the caller are unchanged.
fn default_mstep_damping(has_iiv_on_ruv: bool) -> f64 {
    if has_iiv_on_ruv {
        MSTEP_SA_MAX_STEP_IIV_ON_RUV
    } else {
        MSTEP_SA_MAX_STEP
    }
}

/// Does this fit's numerical θ/σ M-step get the #1011 SA damping?
///
/// Since #1415 the damping is opt-in (`mstep_damping` below 1.0; the default
/// [`MSTEP_SA_MAX_STEP`] is off), so this gate only decides whether a cap the
/// user *set* applies, and whether to warn that it does not. When it does:
/// only when NLopt is left estimating a θ that **anchors no mu-reference** — the
/// #1011 shape, and exactly the condition the no-ETA advisory warns on.
/// `theta_is_mu_ref_anchor` is that predicate, and it is mu-reference *detection*
/// rather than a scan of ETA usage: an η attached in a form the parser cannot pair
/// with a θ (an additive `X = TVX + ETA_X`, a non-log-linear covariate model,
/// #619) counts as un-anchored here and is damped. That is deliberate — such a θ
/// has no closed-form shift either, so the numerical M-step really is its only
/// mover, which is the biased channel #1011 is about.
///
/// A θ that is `FIX`ed, or pinned out by the closed-form mu-ref shift (`pinned`),
/// is returned unchanged by NLopt and so has nothing to damp. A free θ that *does*
/// anchor a mu-reference is not the #1011 channel: it has the exact Robbins-Monro
/// `log θ += γ·mean(η)` update available, and where that shift is nevertheless
/// switched off for the fit (`mu_referencing = false`, an identity-packed θ, a
/// negative lower bound) the numerical M-step it falls back to is the one ferx
/// shipped before #1011. Damping those too would cap ~85 exploration M-steps at 3%
/// on a configuration the #1011 anchors never covered, so the gate is kept to the
/// shape that was measured; every fit whose free thetas all anchor a mu-reference
/// stays byte-identical to its pre-#1011 result, σ included.
///
/// `is_mixture` vetoes it outright. A `MIXNUM`-switched typical value is
/// estimated by this same numerical M-step (#996 routes it there deliberately,
/// because SAEM's hard class draw makes the per-class η mean a biased
/// statistic), but it is solving a different problem: the class typical values
/// must *separate* from a common start, and the class assignments only stabilise
/// once they have. Damping that excursion stalls the separation — on
/// `tests/nonmem/mixture_iv_saem` a 0.03 exploration cap leaves `TVCL1 = 1.145`
/// against NONMEM's 1.002, and drags the chained IMP marginal with it. Whether
/// the #1011 bias also affects mixture class θ (it plausibly does) needs its own
/// schedule and its own anchor, so it is left alone rather than half-fixed; the
/// no-ETA advisory still fires for those models.
fn damps_numerical_mstep(
    is_mixture: bool,
    n_theta: usize,
    theta_fixed: &[bool],
    pinned: &[usize],
    theta_is_mu_ref_anchor: &[bool],
) -> bool {
    if is_mixture {
        return false;
    }
    (0..n_theta).any(|t| {
        !theta_fixed.get(t).copied().unwrap_or(false)
            && !pinned.contains(&t)
            && !theta_is_mu_ref_anchor.get(t).copied().unwrap_or(false)
    })
}

/// SA step size for the **θ half** of the numerical θ/σ M-step (#1011).
///
/// [`theta_sigma_mstep_light`] is a *single* NLopt problem over the concatenated
/// `[θ; σ]` vector, so `theta_new` and `sigma_new` are one joint maximiser of the
/// same frozen-η conditional likelihood, and this γ used to blend both. Since
/// #1445 the σ half has its own, always-on step ([`sigma_mstep_sa_step`]), which
/// is never larger than this one — the case the original argument here warned
/// about (σ assigned at 1.0 while θ is held back, so σ absorbs the misfit the θ
/// damping refused to let θ fix) is exactly what that `min` rules out. The gate
/// below still reads as a θ property, and now gates only the θ half.
///
/// * No θ for the damping to act on (see [`damps_numerical_mstep`]) → `1.0`,
///   the undamped pre-#1011 assignment, so those fits are byte-identical.
/// * Exploration → capped at the `mstep_damping` value, the θ-side counterpart
///   of the [`OMEGA_SA_MAX_STEP`] cap on Ω. The default [`MSTEP_SA_MAX_STEP`]
///   is `1.0`, i.e. no cap (#1415).
/// * Convergence → the full decaying `γ = 1/(k−k1)`, same schedule as Ω.
///
/// **`cap >= 1.0` is a sentinel, not a cap value.** It disables the damping in
/// *both* phases, so the option is deliberately discontinuous at 1.0: `0.999`
/// still buys the full convergence schedule `γ = 1/(k−k1)`, and `1.0` buys none
/// of it. There is no way to spell "cap exploration a little, leave convergence
/// alone" — `cap` only ever governs exploration — and "off" has to restore the
/// whole pre-#1011 trajectory to be worth having (lifting only the exploration
/// cap left the reprex at TVFRD1 0.047 rather than its true undamped 0.039; the
/// unit test below pins that regression).
///
/// **The first convergence iteration is undamped.** At `k = k1 + 1`,
/// `γ = 1/(k−k1) = 1.0`, so the exploration-accumulated θ/σ is overwritten by
/// that iteration's maximiser — the same assignment the damping exists to avoid.
/// That is intentional and mirrors `gamma_omega`'s schedule exactly: by the end
/// of exploration the chain is equilibrated, which is the condition the
/// exploration cap was waiting for, and re-capping the hand-off would only delay
/// the same step.
fn mstep_sa_step(numerically_estimated_theta: bool, exploring: bool, gamma: f64, cap: f64) -> f64 {
    // `cap >= 1.0` is the documented "off" switch and must restore the pre-#1011
    // assignment in **both** phases. Returning `gamma` in convergence would still
    // damp at `1/(k−k1)`, so "off" has to short-circuit here rather than rely on
    // the `min` below.
    if !numerically_estimated_theta || cap >= 1.0 {
        return 1.0;
    }
    if exploring {
        gamma.min(cap)
    } else {
        gamma
    }
}

/// SA step for the **σ half** of the same numerical M-step result (#1445).
///
/// `min(γ_k, SIGMA_SA_MAX_STEP, γ_mstep)` — see [`SIGMA_SA_MAX_STEP`] for why the
/// cap applies in both phases. The third term is a one-sided guarantee: σ never
/// steps *faster* than θ. It does not make the two equal, and where they differ
/// is worth being precise about, because `mstep_damping` is the one knob a user
/// can point at this:
///
/// * `mstep_damping ≤ 0.2` (the `iiv_on_ruv` default of 0.03 is here):
///   **exploration** has `γ_σ = γ_θ = cap`, unchanged from before #1445.
/// * `mstep_damping > 0.2` (up to and including the "off" sentinel `1.0`):
///   exploration has `γ_θ = cap` but `γ_σ = 0.2`, so σ is the slower of the two.
/// * **Convergence**, either way: `γ_θ = 1/(k−k1)` uncapped, `γ_σ` capped at
///   0.2, so they differ for `k − k1 ∈ {1, 2, 3, 4}` (θ takes 1, ½, ⅓, ¼ while
///   σ takes 0.2) and are equal from `k − k1 = 5` on, where `1/(k−k1) ≤ 0.2`.
///
/// So even the `iiv_on_ruv` shape is not bit-identical: it changes over exactly
/// those four hand-off iterations.
///
/// **On [`mstep_sa_step`]'s "one γ for both components" argument.** That doc is
/// right that `theta_sigma_mstep_light` returns a *joint* maximiser, and that
/// blending θ at γ while assigning σ at 1.0 would let σ absorb the misfit the θ
/// damping just refused to let θ fix. The direction here is the opposite one:
/// with the default `mstep_damping` off, θ is *accepted in full*, so σ takes a
/// partial step toward the maximiser at the θ the fit actually adopted — a
/// Robbins-Monro average of the σ maximiser given the new θ, not a half-applied
/// joint step. When damping is on, `min` restores the locked pair. The asymmetry
/// is the same one Ω already has against θ, and for the same reason: which
/// statistic can be trusted from a single draw.
fn sigma_mstep_sa_step(gamma: f64, gamma_mstep: f64) -> f64 {
    gamma.min(SIGMA_SA_MAX_STEP).min(gamma_mstep)
}

/// Sanitise the `mstep_damping` fit option at its point of use.
///
/// The parser rejects anything outside `(0, 1]`, but `FitOptions` is public: a
/// Rust caller — the ferx-r glue, a test, any embedder — can build one directly
/// and never pass through that validator. The failure would be silent and
/// severe: a negative γ makes [`damp_mstep`] step θ and σ *away* from the M-step
/// maximiser on every iteration, and `0.0` freezes both for the whole
/// exploration phase, neither of which the fit would otherwise report. So clamp
/// here as well, and return the substituted value so the caller can warn.
///
/// `> 1.0` becomes `1.0`, which is what any γ above 1 already meant (the
/// documented "off" switch) — `+∞` included, since a caller writing
/// `f64::INFINITY` means "no damping", and routing it to the *maximum* damping
/// would be the exact opposite. `<= 0.0` (`-∞` included) and `NaN` fall back to
/// the default rather than to a hair above zero, since a near-zero cap is
/// itself a frozen fit (#1415). `None` means the value was already in range.
fn sanitize_mstep_damping(cap: f64) -> Option<f64> {
    if cap.is_nan() || cap <= 0.0 {
        Some(MSTEP_SA_MAX_STEP)
    } else if cap > 1.0 {
        Some(1.0)
    } else {
        None
    }
}

/// Robbins-Monro blend of a numerical M-step result into the running estimate:
/// `cur += γ·(new − cur)`. `γ >= 1.0` reproduces the undamped assignment
/// byte-for-byte, which is what a fit with no free numerical θ still does.
///
/// A pinned dimension (mu-referenced, or FIXed) is unaffected either way — NLopt
/// returns it unchanged, so `new == cur` and the blend is a no-op regardless of
/// γ. That is what keeps the blanket application honest on the θ side: the
/// damped set is exactly the free, un-anchored θ the gate is about. σ has no
/// such pinning — it is genuinely re-maximised every iteration — and is damped
/// deliberately, because it comes out of the *same* joint NLopt solve as θ (see
/// [`mstep_sa_step`]).
fn damp_mstep(cur: &mut [f64], new: &[f64], gamma: f64) {
    if gamma >= 1.0 {
        cur.copy_from_slice(new);
        return;
    }
    for (c, &n) in cur.iter_mut().zip(new.iter()) {
        *c += gamma * (n - *c);
    }
}

/// Robbins-Monro blend of the σ half of the numerical M-step result, taken on
/// the **variance** scale: `σ² += γ·(σ_new² − σ²)`, written back as `log σ`
/// (#1445).
///
/// `cur` and `new` are `log σ` (σ is the residual *SD*; the residual
/// correlations of a `block_sigma` are a separate vector and never enter this
/// one), so the blend is `log σ = ½·log((1−γ)·e^{2 log σ} + γ·e^{2 log σ_new})`.
///
/// **Why the variance scale and not the packed one.** The statistic SAEM is
/// approximating is a residual sum of squares — that is literally what
/// [`update_scalar_residual_sse`] averages on the one σ channel that already had
/// an SA step, and `σ = √(S_r/n)` is read off it. Averaging `log σ` instead is a
/// geometric mean, and on exactly the sequence this fix exists to tame it is not
/// a cosmetic difference: the #1445 repro's σ_add M-step maximisers over a
/// 250-iteration convergence phase (min 0.0087, max 2.34, median 0.035, truth
/// 1.8) Robbins-Monro to **0.097** in log space against **0.816** on the
/// variance scale. A boundary-heavy sampling distribution is what a weakly
/// identified minority variance component has, and a geometric mean of it is
/// dominated by the draws that came back near zero.
///
/// A FIXed σ is returned unchanged by NLopt (`lower == upper`), so `new == cur`
/// and the blend is a no-op for it at any γ — the same property that lets
/// [`damp_mstep`] be applied blanket-wise on the θ side. `γ >= 1.0` copies, so
/// the undamped assignment is reproduced bit-for-bit. A non-finite `new`
/// (NLopt returned garbage for a coordinate) leaves that coordinate alone
/// rather than poisoning the running average.
fn damp_mstep_sigma_variance(cur: &mut [f64], new: &[f64], gamma: f64) {
    if gamma >= 1.0 {
        cur.copy_from_slice(new);
        return;
    }
    for (c, &n) in cur.iter_mut().zip(new.iter()) {
        if !n.is_finite() {
            continue;
        }
        let blended = (1.0 - gamma) * (2.0 * *c).exp() + gamma * (2.0 * n).exp();
        if blended > 0.0 && blended.is_finite() {
            *c = 0.5 * blended.ln();
        }
    }
}

/// Raise every *free* diagonal entry of the BSV Ω that has fallen below `floor`
/// up to `floor`. FIX-ed diagonals (`omega_fixed[i] == true`) are left untouched
/// — they carry the user's declared variance and must not be perturbed.
///
/// Shared source of truth for the SAEM and IMPMAP estimators (`impmap.rs` calls
/// this instead of carrying its own byte-identical copy).
pub(crate) fn floor_omega_diagonal(omega_mat: &mut DMatrix<f64>, omega_fixed: &[bool], floor: f64) {
    for i in 0..omega_mat.nrows() {
        let fixed = omega_fixed.get(i).copied().unwrap_or(false);
        if !fixed && omega_mat[(i, i)] < floor {
            omega_mat[(i, i)] = floor;
        }
    }
}

struct SaemState {
    /// Per-subject current ETAs
    etas: Vec<Vec<f64>>,
    /// Per-subject per-occasion kappa samples. `kappas[i][k]` = kappas for
    /// subject i, occasion k.  Empty outer vecs when `n_kappa == 0`.
    kappas: Vec<Vec<Vec<f64>>>,
    /// Cached individual NLL at current ETAs (and kappas for IOV models)
    nll_cache: Vec<f64>,
    /// Per-subject MH step sizes (for the block eta kernel)
    step_scales: Vec<f64>,
    /// Per-subject, per-eta step sizes for the componentwise eta kernel
    /// (Kuhn-Lavielle kernel 2).  Adapted independently for each coordinate
    /// so that etas with vastly different posterior precision (e.g. FREM
    /// covariate etas vs PK etas) can converge to their individual optima.
    /// Indexed `[subject][eta]`.
    cw_step_scales: Vec<Vec<f64>>,
    /// Per-subject kappa MH step sizes.  Empty when `n_kappa == 0`.
    kappa_step_scales: Vec<f64>,
    /// Per-subject acceptance counts since last adaptation
    accept_counts: Vec<usize>,
    /// Per-subject proposal counts since last adaptation (1 for HMC, n_mh_steps for MH)
    proposal_counts: Vec<usize>,
    /// Per-subject, per-eta componentwise-kernel acceptance counts since last
    /// adaptation.  Indexed `[subject][eta]`.
    cw_accept_counts: Vec<Vec<usize>>,
    /// Per-subject, per-eta componentwise-kernel proposal counts since last
    /// adaptation.  Indexed `[subject][eta]`.
    cw_proposal_counts: Vec<Vec<usize>>,
    /// Per-subject kappa acceptance counts since last adaptation.
    kappa_accept_counts: Vec<usize>,
    /// Per-subject kappa proposal counts since last adaptation.
    kappa_proposal_counts: Vec<usize>,
    /// Steps since last adaptation
    steps_since_adapt: usize,
    /// SA sufficient statistic for Omega: running average of (1/N) Σ ηᵢηᵢᵀ
    s2: DMatrix<f64>,
    /// SA sufficient statistic for Omega_iov: running average of (1/N_occ) Σᵢ Σₖ κᵢₖκᵢₖᵀ.
    /// Zero-sized when `n_kappa == 0`.
    s2_iov: DMatrix<f64>,
    /// SA sufficient statistic for an eligible scalar residual variance: the
    /// running residual sum of squares, with proportional residuals divided by
    /// the squared individual prediction. `None` for the general numerical
    /// residual M-step.
    residual_sse: Option<f64>,
    /// Current theta
    theta: Vec<f64>,
    /// Current omega matrix
    omega_mat: DMatrix<f64>,
    /// Current Omega_iov matrix (zero-sized when `n_kappa == 0`).
    omega_iov_mat: DMatrix<f64>,
    /// Current sigma values
    sigma_vals: Vec<f64>,
}

/// Simple Gaussian residual channel whose complete-data σ M-step has a scalar
/// sufficient statistic. The eligibility gate below intentionally admits only
/// the legacy `ErrorSpec::Single` forms: per-endpoint, selected, correlated,
/// combined, transformed, and magnitude-scaled error models need their own
/// derivation rather than an approximation that silently changes their target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScalarResidualModel {
    Additive,
    Proportional,
}

/// Whether every free structural θ is moved by the log-mu-referenced M-step.
/// That update re-centres its paired η by the same realised shift, leaving the
/// latent individual parameters (and hence this residual statistic) unchanged.
/// A free numerical θ could change predictions between samples, so it stays on
/// the joint NLopt θ/σ path.
fn only_mu_referenced_free_thetas(
    n_theta: usize,
    theta_fixed: &[bool],
    mu_ref_pairs: &[(usize, usize)],
) -> bool {
    (0..n_theta).all(|i| {
        theta_fixed.get(i).copied().unwrap_or(false)
            || mu_ref_pairs.iter().any(|&(theta_idx, _)| theta_idx == i)
    })
}

/// Return the exact scalar residual-statistic M-step only for the deliberately
/// narrow models whose Gaussian complete-data likelihood is
/// `n_obs * log(σ) + RSS / (2σ²)`, up to constants. Unsupported shapes retain
/// the established numerical M-step.
fn scalar_residual_mstep_model(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    n_kappa: usize,
    is_mixture: bool,
    use_closed_form_mstep: bool,
    mu_ref_pairs: &[(usize, usize)],
) -> Option<ScalarResidualModel> {
    let scalar_model = match &model.error_spec {
        ErrorSpec::Single(ErrorModel::Additive) => ScalarResidualModel::Additive,
        ErrorSpec::Single(ErrorModel::Proportional) => ScalarResidualModel::Proportional,
        _ => return None,
    };

    let one_free_sigma = init_params.sigma.values.len() == 1
        && !init_params.sigma_fixed.first().copied().unwrap_or(false);
    let stable_latent_predictions = only_mu_referenced_free_thetas(
        init_params.theta.len(),
        &init_params.theta_fixed,
        mu_ref_pairs,
    ) && (use_closed_form_mstep
        || init_params.theta_fixed.iter().all(|&fixed| fixed));
    let plain_gaussian_rows = population.subjects.iter().all(|subject| {
        !subject.has_censored_observation()
            && subject.obs_records.is_empty()
            && !subject.observations.is_empty()
    });

    if one_free_sigma
        && stable_latent_predictions
        && plain_gaussian_rows
        && !is_mixture
        && n_kappa == 0
        && model.residual_error_eta.is_none()
        && model.residual_correlations.is_empty()
        && model.frem_config.is_none()
        && !model.has_custom_ruv_magnitude()
        && !model.log_transform
    {
        Some(scalar_model)
    } else {
        None
    }
}

/// Sum the scalar residual sufficient statistic at the retained individual
/// samples. A proportional prediction at (or too near) zero has no valid
/// `r² / f²` statistic, so callers fall back to the general M-step instead of
/// introducing an arbitrary denominator floor.
fn scalar_residual_sse(
    model: &CompiledModel,
    population: &Population,
    theta: &[f64],
    etas: &[Vec<f64>],
    residual_model: ScalarResidualModel,
) -> Option<(f64, usize)> {
    use rayon::prelude::*;

    let per_subject: Vec<Option<(f64, usize)>> = population
        .subjects
        .par_iter()
        .zip(etas.par_iter())
        .map_init(EventPkParams::default, |scratch, (subject, eta)| {
            let predictions =
                crate::pk::compute_predictions_with_tv_into(model, subject, theta, eta, scratch);
            if predictions.len() != subject.observations.len() {
                return None;
            }
            let mut sse = 0.0;
            for (&y, &f) in subject.observations.iter().zip(predictions.iter()) {
                if !(y.is_finite() && f.is_finite()) {
                    return None;
                }
                let residual = y - f;
                let term = match residual_model {
                    ScalarResidualModel::Additive => residual * residual,
                    ScalarResidualModel::Proportional => {
                        if f.abs() <= f64::MIN_POSITIVE {
                            return None;
                        }
                        residual * residual / (f * f)
                    }
                };
                if !term.is_finite() {
                    return None;
                }
                sse += term;
            }
            Some((sse, subject.observations.len()))
        })
        .collect();

    // Preserve thread-count reproducibility: Rayon may schedule subjects in a
    // different partition, but the collected vector remains input ordered.
    per_subject
        .into_iter()
        .try_fold((0.0, 0_usize), |(sum, n), item| {
            item.map(|(subject_sse, subject_n)| (sum + subject_sse, n + subject_n))
        })
}

/// Robbins-Monro update of a scalar residual sum of squares. The first retained
/// sample initializes the statistic; exploration's γ = 1 then has the usual
/// overwrite semantics without needing a synthetic RSS at the initial ETAs.
fn update_scalar_residual_sse(statistic: &mut Option<f64>, sample_sse: f64, gamma: f64) {
    match statistic {
        Some(current) => *current += gamma * (sample_sse - *current),
        None => *statistic = Some(sample_sse),
    }
}

// ---------------------------------------------------------------------------
// Metropolis-Hastings step for one subject
// ---------------------------------------------------------------------------

#[cfg(test)]
thread_local! {
    /// Test-only: when set, [`run_saem`] runs with **no** per-subject
    /// `EventSchedule` cache, i.e. the pre-#1447 behaviour of rebuilding the
    /// schedule inside every NLL evaluation.
    ///
    /// The cache is claimed to be bit-identical, not merely close, so the test
    /// for it is the same fit under both settings compared on the raw bits —
    /// and that needs a way to ask for the old behaviour. Thread-local, read on
    /// the thread that calls `run_saem` before any rayon fan-out, so concurrent
    /// tests cannot see each other's setting and no lock is involved.
    pub(crate) static SCHEDULE_CACHE_DISABLED: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

/// Test-only scope guard for [`SCHEDULE_CACHE_DISABLED`], restoring the previous
/// value on drop (including on an assertion unwind).
#[cfg(test)]
pub(crate) struct ScheduleCacheOff(bool);

#[cfg(test)]
impl ScheduleCacheOff {
    pub(crate) fn enter() -> Self {
        Self(SCHEDULE_CACHE_DISABLED.with(|c| c.replace(true)))
    }
}

#[cfg(test)]
impl Drop for ScheduleCacheOff {
    fn drop(&mut self) {
        SCHEDULE_CACHE_DISABLED.with(|c| c.set(self.0));
    }
}

/// The per-subject `EventSchedule` cache [`run_saem`] runs with.
///
/// Delegates to the shared [`build_schedule_cache`](crate::estimation::inner_optimizer::build_schedule_cache)
/// — the same builder the FOCE inner loop and the Bayes chain use, so the
/// staleness rules live in one place — except under the test-only kill switch
/// above, where it returns the all-`None` vector that reproduces the
/// rebuild-per-call path exactly.
fn saem_schedule_cache(
    model: &CompiledModel,
    population: &Population,
) -> Vec<Option<crate::pk::event_driven::EventSchedule>> {
    #[cfg(test)]
    if SCHEDULE_CACHE_DISABLED.with(|c| c.get()) {
        return population.subjects.iter().map(|_| None).collect();
    }
    crate::estimation::inner_optimizer::build_schedule_cache(model, population)
}

/// Every buffer one subject's MH sweep needs, owned by the rayon worker rather
/// than rebuilt per proposal.
///
/// The E-step evaluates the subject NLL ~39 times per subject per iteration
/// (`n_mh_steps` block proposals + `n_cw_sweeps · n_eta` componentwise ones),
/// and before this struct each of those allocated seven `Vec`/`DVector`s that
/// were dropped microseconds later: four here (`z`, the `DVector` copy of it,
/// the `chol(Ω)·z` product and `eta_prop`) and three inside
/// [`individual_nll_into_with_schedule`](crate::stats::likelihood). On a bench
/// like cefepime — 458 subjects, a **single** observation for the median
/// subject — that allocator traffic is a real share of the sweep, not a
/// rounding error, because there is so little arithmetic per evaluation to
/// hide it behind.
///
/// `prep` is the η-independent half of the NLL's inputs, refreshed once per
/// subject by [`MhScratch::begin_subject`]. Everything else is pure capacity:
/// each field is fully overwritten before it is read, so a reused scratch and a
/// fresh one score bit-identically.
pub(crate) struct MhScratch {
    nll: IndividualNllScratch,
    prep: IndividualNllPrep,
    z: DVector<f64>,
    perturbation: DVector<f64>,
    eta_prop: Vec<f64>,
}

impl Default for MhScratch {
    fn default() -> Self {
        Self {
            nll: IndividualNllScratch::default(),
            prep: IndividualNllPrep::default(),
            z: DVector::zeros(0),
            perturbation: DVector::zeros(0),
            eta_prop: Vec::new(),
        }
    }
}

impl MhScratch {
    /// Point the scratch at a new subject: rebuild the η-independent NLL inputs
    /// and size the proposal buffers. Called once per subject per SAEM
    /// iteration, never inside the proposal loop.
    pub(crate) fn begin_subject(
        &mut self,
        model: &CompiledModel,
        subject: &Subject,
        theta: &[f64],
        n_eta: usize,
    ) {
        self.prep.refresh(model, subject, theta);
        if self.z.len() != n_eta {
            self.z = DVector::zeros(n_eta);
            self.perturbation = DVector::zeros(n_eta);
        }
        self.eta_prop.clear();
        self.eta_prop.resize(n_eta, 0.0);
    }

    /// The per-event PK snapshot buffer, for the callers that still take a bare
    /// [`EventPkParams`] (the mixture class draw, the IOV NLL).
    pub(crate) fn pk(&mut self) -> &mut EventPkParams {
        &mut self.nll.pk
    }
}

/// `out ← chol(Ω) · z`, into a buffer the caller already owns.
///
/// This is the block MH proposal's correlated perturbation. It replaced
/// `l * DVector::from_column_slice(&z)`, which allocated a fresh `DVector` per
/// proposal; both reach the same `gemm_uninit` kernel, this one through
/// `Matrix::gemm` with **`beta = 0`**, which makes it *write* rather than
/// accumulate and so never reads `out`'s previous contents.
///
/// It is a named function rather than one line inlined in `mh_steps` so the
/// equivalence test can call the production code instead of a copy of it. It
/// was inlined at first, and the test that "pinned" it reimplemented the same
/// `gemm` call — so changing `beta` in the real one left the test green
/// (#1452 review round 2, found by running the mutation rather than reasoning
/// about it).
#[inline]
pub(crate) fn cholesky_perturbation_into(
    l: &DMatrix<f64>,
    z: &DVector<f64>,
    out: &mut DVector<f64>,
) {
    out.gemm(1.0, l, z, 0.0);
}

/// Run `n_steps` symmetric random-walk MH iterations for one subject in-place.
/// Returns (n_accepted, updated_nll).
///
/// `eta` is in deviation (eta_true) space — the same space the model's
/// `pk_param_fn` consumes — so proposals are random walks
/// `eta + step_scale · L · z` from the current position. The acceptance
/// log-ratio is `nll_current − nll_prop`, which is correct because the
/// symmetric proposal density cancels.
///
/// Note: an earlier version centred proposals on `mu_k` during exploration.
/// That was incorrect: `individual_nll` interprets `eta` as the deviation
/// `log(CL_i) − log(TVCL)`, while `mu_k = log(TVCL)`, so the model evaluated
/// `CL = TVCL · exp(log TVCL) = TVCL²` for every accepted exploration step.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mh_steps(
    eta: &mut [f64],
    nll_current: f64,
    subject: &Subject,
    model: &CompiledModel,
    theta: &[f64],
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    step_scale: f64,
    // Optional per-coordinate multiplier on the joint proposal (issue #895).
    // `None` (or a row of 1.0) reproduces the plain `chol(Ω)·z` block move. A
    // value < 1 shrinks the joint step for that coordinate — used to damp
    // near-deterministic FREM covariate ETAs (posterior SD ≈ √EPSCOV ≪ √Ω_jj),
    // whose full-scale block move would otherwise be rejected every time and
    // pin the whole joint acceptance at 0%. The multiplier is deterministic and
    // symmetric in η, so detailed balance is preserved. Indexed `[0, n_eta)`.
    eta_block_scale: Option<&[f64]>,
    rng: &mut impl Rng,
    n_steps: usize,
    // Caller-owned buffers, already pointed at this subject by
    // [`MhScratch::begin_subject`].
    scratch: &mut MhScratch,
    // This subject's cached `EventSchedule`, or `None` where reuse is unsound
    // (`inner_optimizer::cacheable_schedule`) and the predictor rebuilds it per
    // call as before.
    schedule: Option<&crate::pk::event_driven::EventSchedule>,
    // When Some, eta proposals are evaluated with IOV-aware NLL (kappas held fixed).
    // This is required for Gibbs correctness in IOV models: the acceptance ratio
    // must target p(η | κ, θ, data), which includes the per-occasion kappa terms.
    kappas_opt: Option<(&[Vec<f64>], &OmegaMatrix)>,
) -> (usize, f64) {
    let n_eta = eta.len();
    let l = &omega.chol;
    let mut nll = nll_current;
    let mut n_accepted = 0;
    // Split borrow: `perturbation.gemm(.., z, ..)` needs two fields of the
    // scratch at once, and `individual_nll_prepared` needs two more.
    let MhScratch {
        nll: nll_scratch,
        prep,
        z,
        perturbation,
        eta_prop,
    } = scratch;

    for _ in 0..n_steps {
        for slot in z.iter_mut() {
            *slot = rng.sample(StandardNormal);
        }
        cholesky_perturbation_into(l, z, perturbation);

        for j in 0..n_eta {
            let bs = eta_block_scale.map_or(1.0, |s| s[j]);
            eta_prop[j] = eta[j] + step_scale * bs * perturbation[j];
        }

        // Both arms reuse the caller's buffers. For IOV models correctness of
        // the Gibbs conditional p(η | κ, θ, data) requires the per-occasion
        // [eta_prop, kappa_k] predictions, which `individual_nll_prepared` does
        // not compute — hence the separate entry point, which since #1423-era
        // profiling takes the scratch rather than allocating an `EventPkParams`
        // per proposal.
        let nll_prop = if let Some((kappas, omega_iov)) = kappas_opt {
            individual_nll_iov_with_scratch_and_schedule(
                model,
                subject,
                theta,
                eta_prop,
                kappas,
                omega,
                Some(omega_iov),
                sigma_values,
                &mut nll_scratch.pk,
                schedule,
            )
        } else {
            individual_nll_prepared(
                model,
                subject,
                theta,
                eta_prop,
                omega,
                sigma_values,
                prep,
                schedule,
                nll_scratch,
            )
        };

        // Symmetric proposal q(η_prop|η) = q(η|η_prop) cancels in the ratio,
        // so the prior+likelihood difference encoded in `individual_nll` is
        // the full acceptance criterion.
        let log_u: f64 = rng.random::<f64>().ln();
        if log_u < nll - nll_prop {
            eta.copy_from_slice(eta_prop);
            nll = nll_prop;
            n_accepted += 1;
        }
    }

    (n_accepted, nll)
}

/// Componentwise (single-coordinate) Metropolis-within-Gibbs sweep for one
/// subject — the second kernel of the Kuhn & Lavielle (2004) mixture.
///
/// Each sweep proposes a perturbation to one η coordinate at a time,
/// `η'_j = η_j + step_scale · √Ω_jj · z`, holding the other coordinates fixed,
/// and accepts/rejects with the full conditional NLL (which carries the
/// correlated prior, so detailed balance for p(η | data) is preserved). Returns
/// `(n_accepted, n_proposed, updated_nll)` with `n_proposed = n_sweeps · n_eta`.
///
/// Why this kernel exists: the block kernel `mh_steps` proposes along
/// `chol(Ω)·z`, so once Ω drifts toward a high correlation the proposal can only
/// move η along that near-degenerate direction. The single-draw Ω M-step then
/// feeds the induced correlation back into Ω, and during the γ=1 exploration
/// phase (no SA averaging) this compounds into a runaway collapse toward a
/// rank-1 Ω (every off-diagonal correlation → ±1, one variance → 0). A
/// per-coordinate proposal can always move a single η independently of Ω's
/// off-diagonals, so the sampled draws are not forced collinear and the
/// sufficient statistic recovers the true correlation. See the
/// `saem-block-omega-rank1-collapse` investigation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mh_steps_componentwise(
    eta: &mut [f64],
    nll_current: f64,
    subject: &Subject,
    model: &CompiledModel,
    theta: &[f64],
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    // Per-eta step scales — each coordinate adapts its own scale independently
    // so that etas with vastly different posterior precision (e.g. near-
    // deterministic FREM covariate etas vs broad PK etas) can each reach
    // their optimal acceptance rate.
    step_scales: &[f64],
    // Per-coordinate proposal SD = √(marginal variance), precomputed once per
    // iteration from Ω's diagonal (it is identical across subjects) and floored
    // to match the Ω diagonal floor so a collapsing diagonal can't shrink the
    // decorrelating step to zero. Indexed `[0, n_eta)`.
    cw_sd: &[f64],
    rng: &mut impl Rng,
    n_sweeps: usize,
    // Caller-owned buffers, already pointed at this subject by
    // [`MhScratch::begin_subject`]. The sweep proposes in place on `eta`, so it
    // uses only the NLL half of the scratch.
    scratch: &mut MhScratch,
    schedule: Option<&crate::pk::event_driven::EventSchedule>,
    kappas_opt: Option<(&[Vec<f64>], &OmegaMatrix)>,
) -> (Vec<usize>, usize, f64) {
    let n_eta = eta.len();
    let mut nll = nll_current;
    let mut per_eta_accepted = vec![0usize; n_eta];
    let MhScratch {
        nll: nll_scratch,
        prep,
        ..
    } = scratch;

    for _ in 0..n_sweeps {
        for j in 0..n_eta {
            let z: f64 = rng.sample(StandardNormal);
            let old_j = eta[j];
            eta[j] = old_j + step_scales[j] * cw_sd[j] * z;

            let nll_prop = if let Some((kappas, omega_iov)) = kappas_opt {
                individual_nll_iov_with_scratch_and_schedule(
                    model,
                    subject,
                    theta,
                    eta,
                    kappas,
                    omega,
                    Some(omega_iov),
                    sigma_values,
                    &mut nll_scratch.pk,
                    schedule,
                )
            } else {
                individual_nll_prepared(
                    model,
                    subject,
                    theta,
                    eta,
                    omega,
                    sigma_values,
                    prep,
                    schedule,
                    nll_scratch,
                )
            };

            // Symmetric scalar proposal cancels, same as the block kernel.
            let log_u: f64 = rng.random::<f64>().ln();
            if log_u < nll - nll_prop {
                nll = nll_prop;
                per_eta_accepted[j] += 1;
            } else {
                eta[j] = old_j; // reject — restore
            }
        }
    }

    (per_eta_accepted, n_eta * n_sweeps, nll)
}

// ---------------------------------------------------------------------------
// Per-occasion kappa MH step for IOV models
// ---------------------------------------------------------------------------

/// Run one symmetric random-walk MH proposal for each occasion's kappa.
///
/// For each occasion k, proposes `κ_k_prop = κ_k + step_scale · L_iov · z` and
/// accepts/rejects using the full IOV individual NLL (includes both the kappa
/// prior and the observation likelihood).  The per-occasion Gibbs structure
/// means proposals are low-dimensional (n_kappa typically 1–3), so the MH
/// acceptance rate stays high even without HMC.
///
/// Returns `(n_accepted, n_proposed, updated_nll)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mh_kappa_steps(
    kappas: &mut [Vec<f64>],
    nll_current: f64,
    subject: &Subject,
    model: &CompiledModel,
    theta: &[f64],
    eta: &[f64],
    omega_bsv: &OmegaMatrix,
    omega_iov: &OmegaMatrix,
    sigma_values: &[f64],
    step_scale: f64,
    rng: &mut impl Rng,
    // This subject's cached `EventSchedule`, or `None` where reuse is unsound.
    // The κ sweep evaluates the full IOV NLL once per occasion per iteration and
    // rebuilt the schedule inside each one (#1452 review).
    schedule: Option<&crate::pk::event_driven::EventSchedule>,
    // Caller-owned per-event PK buffer, for the same reason the η kernels take
    // one: this is a per-proposal allocation otherwise.
    pk_scratch: &mut EventPkParams,
) -> (usize, usize, f64) {
    let n_kappa = omega_iov.matrix.nrows();
    let l = &omega_iov.chol;
    let mut nll = nll_current;
    let mut n_accepted = 0;
    let n_occ = kappas.len();

    for k in 0..n_occ {
        let z: Vec<f64> = (0..n_kappa).map(|_| rng.sample(StandardNormal)).collect();
        let z_vec = DVector::from_column_slice(&z);
        let perturbation = l * z_vec;

        let kap_prop: Vec<f64> = (0..n_kappa)
            .map(|j| kappas[k][j] + step_scale * perturbation[j])
            .collect();

        // Temporarily substitute kappa_k with the proposal.
        let old_kap = kappas[k].clone();
        kappas[k] = kap_prop;

        let nll_prop = individual_nll_iov_with_scratch_and_schedule(
            model,
            subject,
            theta,
            eta,
            kappas,
            omega_bsv,
            Some(omega_iov),
            sigma_values,
            pk_scratch,
            schedule,
        );

        let log_u: f64 = rng.random::<f64>().ln();
        if log_u < nll - nll_prop {
            // Accept
            nll = nll_prop;
            n_accepted += 1;
        } else {
            // Reject — restore old kappa
            kappas[k] = old_kap;
        }
    }

    (n_accepted, n_occ, nll)
}

// ---------------------------------------------------------------------------
// M-step objective assembly (per-subject gradients live in `fixed_eta_gradient`)
// ---------------------------------------------------------------------------

/// Serially fold per-subject `(nll, grad)` pairs, already collected in subject
/// order, into a single `(nll, grad)` total. Deterministic regardless of the
/// rayon worker count that produced `per_subj` (#703): a parallel `reduce`
/// would combine partials along thread-count-dependent boundaries, and f64
/// addition is non-associative.
fn fold_nll_grad(per_subj: Vec<(f64, Vec<f64>)>, n: usize) -> (f64, Vec<f64>) {
    per_subj
        .into_iter()
        .fold((0.0, vec![0.0f64; n]), |(nll_a, mut ga), (nll_b, gb)| {
            for (a, b) in ga.iter_mut().zip(gb.iter()) {
                *a += b;
            }
            (nll_a + nll_b, ga)
        })
}

/// Lightweight M-step: run NLopt SLSQP for a few iterations in packed
/// space, warm-started from the current packed theta / log-sigma.
///
/// `theta_packs_log_mask[i]` selects per-theta packing: log when true,
/// identity when false. Sigma is always log-packed (sigma > 0 by
/// construction). See the run_saem comment on `theta_packs_log_mask` for
/// motivation — without per-theta packing, any theta with `theta_lower < 0`
/// got pinned at 1e-10 and could never be estimated.
/// Mixture context for the θ/σ M-step (#985): each subject's drawn class plus
/// the per-class held σ overrides.
///
/// The class drives the `MIXNUM` guard (so a class-switched typical value is
/// estimated from its own members) *and* the σ vector each subject is scored
/// under: a `sigma(k)` override is held at its init, so a class-`k` subject's
/// residual variance does not depend on the free base σ at all. Scoring it under
/// the base σ anyway would drag the free base estimate toward the override — the
/// E-step samples η under `class_sigma(c)` while the M-step would optimise a
/// different objective (#987 review).
#[derive(Clone, Copy)]
pub(crate) struct MixMstep<'a> {
    /// Per-subject drawn class (0-based).
    pub classes: &'a [usize],
    /// Per-class held σ overrides, `[class][(sigma_index, held_value)]`.
    pub class_sigma_over: &'a [Vec<(usize, f64)>],
}

/// Substitute a class's held σ overrides into the base σ vector. Returns `None`
/// when the class has no override (the caller then uses the base vector as-is,
/// avoiding a per-subject allocation on the common path).
fn class_sigma_subst(sigma_values: &[f64], over: &[(usize, f64)]) -> Option<Vec<f64>> {
    if over.is_empty() {
        return None;
    }
    let mut v = sigma_values.to_vec();
    for &(s, val) in over {
        if s < v.len() {
            v[s] = val;
        }
    }
    Some(v)
}

#[allow(clippy::too_many_arguments)]
fn theta_sigma_mstep_light(
    model: &CompiledModel,
    population: &Population,
    etas: &[Vec<f64>],
    kappas_opt: Option<&[Vec<Vec<f64>>]>,
    log_theta_init: &[f64],
    log_sigma_init: &[f64],
    log_theta_lower: &[f64],
    log_theta_upper: &[f64],
    log_sigma_lower: &[f64],
    log_sigma_upper: &[f64],
    n_theta: usize,
    n_sigma: usize,
    maxiter: u32,
    scale_params: bool,
    theta_packs_log_mask: &[bool],
    // Mixture (#985): per-subject drawn class + held σ overrides. When `Some`,
    // every per-subject observation-likelihood evaluation runs under that
    // subject's `MIXNUM` guard and its class's σ, so a class-switched typical
    // value is estimated from its own class members and a held `sigma(k)`
    // override does not bias the free base σ.
    mix_mstep: Option<MixMstep<'_>>,
    // Per-subject cached `EventSchedule`s (`&[]` = no cache). The M-step's
    // derivative-free solve evaluates `obs_nll_sum` `maxiter * (n + 1)` times,
    // each one a full prediction per subject, so on the event-driven path this
    // is the same per-call schedule rebuild the E-step kernels stopped paying.
    schedules: &[Option<crate::pk::event_driven::EventSchedule>],
) -> (Vec<f64>, Vec<f64>) {
    let n = n_theta + n_sigma;

    let mut x: Vec<f64> = Vec::with_capacity(n);
    x.extend_from_slice(log_theta_init);
    x.extend_from_slice(log_sigma_init);

    let mut lower: Vec<f64> = Vec::with_capacity(n);
    lower.extend_from_slice(log_theta_lower);
    lower.extend_from_slice(log_sigma_lower);
    let mut upper: Vec<f64> = Vec::with_capacity(n);
    upper.extend_from_slice(log_theta_upper);
    upper.extend_from_slice(log_sigma_upper);

    for i in 0..n {
        x[i] = x[i].clamp(lower[i], upper[i]);
    }

    // Unpack a slice of packed theta values into natural-scale theta.
    // Closure (not local fn) so it captures `theta_packs_log_mask`.
    let unpack_thetas = |packed: &[f64]| -> Vec<f64> {
        (0..n_theta)
            .map(|i| {
                if theta_packs_log_mask[i] {
                    packed[i].exp()
                } else {
                    packed[i]
                }
            })
            .collect()
    };

    // Objective operating on the unscaled packed parameters.
    //
    // Gradient strategy: single rayon pass over subjects, each computing its
    // own partial gradient via `obs_nll_subject_grad` (analytical sigma,
    // FD-of-predictions for theta). This replaces the old per-parameter
    // forward-FD of `obs_nll_sum` which launched `n_dim` rayon jobs
    // sequentially. Key improvements:
    //  • Sigma gradient is analytical — no extra predict calls per sigma dim.
    //  • Single rayon launch instead of n_dim sequential launches.
    //  • Better cache locality: one subject's data stays in cache while
    //    iterating over all its theta perturbations.
    //  • Pinned dims (lower == upper) are skipped per-subject, saving the
    //    predict calls entirely (same as the old FD guard).
    let obj = |xv: &[f64], grad: Option<&mut [f64]>, _: &mut ()| -> f64 {
        let th: Vec<f64> = unpack_thetas(&xv[..n_theta]);
        let sg: Vec<f64> = xv[n_theta..].iter().map(|&v| v.exp()).collect();

        if let Some(g) = grad {
            use rayon::prelude::*;
            // Collect in subject order, then fold serially (#703): a parallel
            // `reduce` combines partial (nll, grad) pairs along thread-count-
            // dependent boundaries, and f64 addition is non-associative.
            let (val, grad_vec) = if let Some(kappas) = kappas_opt {
                let per_subj: Vec<(f64, Vec<f64>)> = population
                    .subjects
                    .par_iter()
                    .zip(etas.par_iter())
                    .zip(kappas.par_iter())
                    .enumerate()
                    .map_init(
                        EventPkParams::default,
                        |scratch, (i, ((subject, eta), kaps))| {
                            let cls = mix_mstep.map(|m| m.classes[i]);
                            let _g = cls.map(|c| {
                                crate::parser::model_parser::MixtureClassGuard::enter(c + 1)
                            });
                            let over: &[(usize, f64)] = match (mix_mstep, cls) {
                                (Some(m), Some(c)) => &m.class_sigma_over[c],
                                _ => &[],
                            };
                            let sub_sg = class_sigma_subst(&sg, over);
                            let sg_i: &[f64] = sub_sg.as_deref().unwrap_or(&sg);
                            let (nll, mut grad) = obs_nll_subject_grad_iov(
                                model,
                                subject,
                                &th,
                                sg_i,
                                eta,
                                kaps,
                                &theta_packs_log_mask,
                                &lower,
                                &upper,
                                n_theta,
                                n_sigma,
                                scratch,
                            );
                            // A held σ override carries no information about the
                            // free base σ — zero that subject's contribution.
                            for &(sidx, _) in over {
                                if n_theta + sidx < grad.len() {
                                    grad[n_theta + sidx] = 0.0;
                                }
                            }
                            (nll, grad)
                        },
                    )
                    .collect();
                fold_nll_grad(per_subj, n)
            } else {
                let per_subj: Vec<(f64, Vec<f64>)> = population
                    .subjects
                    .par_iter()
                    .zip(etas.par_iter())
                    .enumerate()
                    .map_init(EventPkParams::default, |scratch, (i, (subject, eta))| {
                        let cls = mix_mstep.map(|m| m.classes[i]);
                        let _g = cls
                            .map(|c| crate::parser::model_parser::MixtureClassGuard::enter(c + 1));
                        let over: &[(usize, f64)] = match (mix_mstep, cls) {
                            (Some(m), Some(c)) => &m.class_sigma_over[c],
                            _ => &[],
                        };
                        let sub_sg = class_sigma_subst(&sg, over);
                        let sg_i: &[f64] = sub_sg.as_deref().unwrap_or(&sg);
                        let (nll, mut grad) = obs_nll_subject_grad(
                            model,
                            subject,
                            &th,
                            sg_i,
                            eta,
                            &theta_packs_log_mask,
                            &lower,
                            &upper,
                            n_theta,
                            n_sigma,
                            scratch,
                        );
                        for &(sidx, _) in over {
                            if n_theta + sidx < grad.len() {
                                grad[n_theta + sidx] = 0.0;
                            }
                        }
                        (nll, grad)
                    })
                    .collect();
                fold_nll_grad(per_subj, n)
            };
            for (gi, &gv) in g.iter_mut().zip(grad_vec.iter()) {
                *gi = if gv.is_finite() { gv } else { 0.0 };
            }
            if val.is_finite() {
                val
            } else {
                1e20
            }
        } else {
            let val = match (mix_mstep, kappas_opt) {
                (Some(mx), Some(kappas)) => {
                    obs_nll_sum_iov_mix(model, population, &th, &sg, etas, kappas, mx)
                }
                (Some(mx), None) => {
                    obs_nll_sum_mix(model, population, &th, &sg, etas, mx, schedules)
                }
                (None, Some(kappas)) => {
                    obs_nll_sum_iov(model, population, &th, &sg, etas, kappas, schedules)
                }
                (None, None) => obs_nll_sum(model, population, &th, &sg, etas, schedules),
            };
            if val.is_finite() {
                val
            } else {
                1e20
            }
        }
    };

    // Compute per-element scale factors from the initial point.
    let scale: Vec<f64> = if scale_params {
        compute_scale(&x)
    } else {
        vec![1.0; n]
    };

    // Scaled starting point and bounds: xs[i] = x[i] / scale[i].
    let mut xs: Vec<f64> = (0..n).map(|i| x[i] / scale[i]).collect();
    let lower_s: Vec<f64> = (0..n).map(|i| lower[i] / scale[i]).collect();
    let upper_s: Vec<f64> = (0..n).map(|i| upper[i] / scale[i]).collect();

    // Wrapper objective: receives scaled xs, unscales before evaluating obj,
    // then scales the gradient back: d(OFV)/d(xs[i]) = d(OFV)/d(x[i]) * scale[i].
    let obj_s = |xv_s: &[f64], grad: Option<&mut [f64]>, data: &mut ()| -> f64 {
        let xv: Vec<f64> = (0..n).map(|i| xv_s[i] * scale[i]).collect();
        if let Some(g) = grad {
            let mut g_raw = vec![0.0_f64; n];
            let val = obj(&xv, Some(&mut g_raw), data);
            for i in 0..n {
                g[i] = g_raw[i] * scale[i];
            }
            val
        } else {
            obj(&xv, None, data)
        }
    };

    // See `MSTEP_NLOPT_ALGORITHM` for rationale (BOBYQA vs SLSQP).
    let mut opt = nlopt::Nlopt::new(MSTEP_NLOPT_ALGORITHM, n, obj_s, nlopt::Target::Minimize, ());
    opt.set_lower_bounds(&lower_s).unwrap();
    opt.set_upper_bounds(&upper_s).unwrap();
    opt.set_maxeval(maxiter * (n as u32 + 1)).unwrap();
    if mix_mstep.is_some() {
        // The mixture arm keeps the pre-#1415 configuration — NLopt's default
        // first design and the 1e-4 tolerance — on purpose. Its class typical
        // values are estimated from a *hard* class draw (#996), and the
        // NONMEM-anchored `tests/mixture_nonmem.rs` seed sweep was calibrated
        // on what this solve returned under that configuration, which is a
        // partial step. A converged solve moves the anchor off the NONMEM MLE
        // (seed-mean TVCL2 2.84 → 3.23 with both settings; 3.09 with the local
        // first design alone; p(1) 0.54 → 0.44 and IMP marginal 300.9 → 306.9
        // with the tolerance alone), so an exact per-class M-step on a hard
        // draw is not the same estimator, and needs its own schedule and its
        // own anchor before it changes. See `MSTEP_MIXTURE_FTOL_REL`.
        opt.set_ftol_rel(MSTEP_MIXTURE_FTOL_REL).unwrap();
    } else {
        opt.set_ftol_rel(MSTEP_FTOL_REL).unwrap();
        // A warm-started local solve needs a *local* first design (#1415).
        // Every coordinate gets the same trust radius in packed units, undone
        // through the optional magnitude scaling so the radius is the same
        // fraction of the parameter either way.
        //
        // Bounded by the coordinate's own interval: BOBYQA refuses the whole
        // problem (`NLOPT_INVALID_ARGS`, before a single evaluation) if any
        // free coordinate has `upper - lower < 2 * step`, and the error is
        // discarded below, so a theta declared on a narrow interval — CL on
        // (1.9, 2.1) is 0.10008 log units wide — would silently turn every
        // M-step into a no-op for θ *and* σ (#1420 review). A quarter of the
        // width is NLopt's own default and leaves the required factor-of-two
        // margin. A pinned coordinate (`upper == lower`) is eliminated by NLopt
        // before BOBYQA sees it; it keeps the nominal step, which must stay
        // positive.
        let initial_step: Vec<f64> = (0..n)
            .map(|i| mstep_initial_step(lower[i], upper[i]) / scale[i])
            .collect();
        opt.set_initial_step(&initial_step).unwrap();
    }

    let outcome = opt.optimize(&mut xs);
    // A configuration NLopt rejects outright is a bug in this function, not a
    // property of the fit; `xs` is then untouched and the M-step is a silent
    // no-op. Loud in debug (every unit and slow test), swallowed in release
    // like every other NLopt outcome here.
    debug_assert!(
        !matches!(outcome, Err((nlopt::FailState::InvalidArgs, _))),
        "SAEM numerical M-step: NLopt rejected the problem (InvalidArgs) — \
         check the initial step against the bounds"
    );
    let _ = outcome;

    // Unscale back to log-space.
    let x_final: Vec<f64> = (0..n).map(|i| xs[i] * scale[i]).collect();

    let log_theta_new = x_final[..n_theta].to_vec();
    let log_sigma_new = x_final[n_theta..].to_vec();
    (log_theta_new, log_sigma_new)
}

/// Sum of observation log-likelihoods with ETAs held fixed.
///
/// Under M3, censored rows contribute the matching normal-tail likelihood
/// instead of the Gaussian residual term. Without this branch, the SAEM M-step
/// would optimize θ/σ as if censored observations were exact Gaussians at the limit,
/// producing silently-biased population estimates.
///
/// Uses rayon's `map_init` so each worker thread allocates one
/// `EventPkParams` scratch on first use and reuses it across every
/// subject the worker handles. With NLopt's central-FD gradient
/// hitting `obs_nll_sum` `1 + 2·n_dim` times per M-step, this cuts
/// per-call `Vec<PkParams>` churn to near-zero on TV-cov data.
pub(crate) fn obs_nll_sum(
    model: &CompiledModel,
    population: &Population,
    theta: &[f64],
    sigma_values: &[f64],
    etas: &[Vec<f64>],
    // Per-subject cached `EventSchedule`s, parallel to `population.subjects`.
    // `&[]` means "no cache" and reproduces the per-call rebuild exactly; that is
    // what the tests and any caller without one pass.
    schedules: &[Option<crate::pk::event_driven::EventSchedule>],
) -> f64 {
    use rayon::prelude::*;
    // Collect in subject order and sum serially so the objective does not
    // depend on the rayon worker count (f64 addition is non-associative and a
    // parallel `.sum()` splits by thread count) — #703.
    let per_subj: Vec<f64> = population
        .subjects
        .par_iter()
        .enumerate()
        .map_init(EventPkParams::default, |scratch, (i, subject)| {
            crate::stats::likelihood::obs_nll_subject_into_with_schedule(
                model,
                subject,
                theta,
                sigma_values,
                &model.residual_correlations,
                &etas[i],
                scratch,
                schedules.get(i).and_then(|s| s.as_ref()),
            )
        })
        .collect();
    per_subj.iter().sum()
}

/// IOV variant of `obs_nll_sum`: per-occasion predictions using `[eta, kappa_k]`.
fn obs_nll_sum_iov(
    model: &CompiledModel,
    population: &Population,
    theta: &[f64],
    sigma_values: &[f64],
    etas: &[Vec<f64>],
    kappas: &[Vec<Vec<f64>>],
    // Per-subject cached `EventSchedule`s, parallel to `population.subjects`.
    // `&[]` means "no cache" and reproduces the per-call rebuild exactly.
    schedules: &[Option<crate::pk::event_driven::EventSchedule>],
) -> f64 {
    use rayon::prelude::*;
    // Deterministic reduction (collect in subject order, fold serially): a
    // parallel `.sum()` would make the objective depend on the rayon worker
    // count — #703.
    let per_subj: Vec<f64> = population
        .subjects
        .par_iter()
        .enumerate()
        .map_init(EventPkParams::default, |scratch, (i, subject)| {
            crate::estimation::fixed_eta_gradient::obs_nll_subject_into_iov_with_schedule(
                model,
                subject,
                theta,
                sigma_values,
                &etas[i],
                &kappas[i],
                scratch,
                schedules.get(i).and_then(|s| s.as_ref()),
            )
        })
        .collect();
    per_subj.iter().sum()
}

/// Mixture (#985) variant of [`obs_nll_sum`]: each subject's observation NLL is
/// evaluated under its drawn class's `MIXNUM` guard — so the class-switched
/// typical values (`if MIXNUM == k …`) are seen — and under that class's σ, so a
/// held `sigma(k)` override is honoured rather than replaced by the free base σ.
fn obs_nll_sum_mix(
    model: &CompiledModel,
    population: &Population,
    theta: &[f64],
    sigma_values: &[f64],
    etas: &[Vec<f64>],
    mix: MixMstep<'_>,
    schedules: &[Option<crate::pk::event_driven::EventSchedule>],
) -> f64 {
    use rayon::prelude::*;
    let per_subj: Vec<f64> = population
        .subjects
        .par_iter()
        .enumerate()
        .map_init(EventPkParams::default, |scratch, (i, subject)| {
            let c = mix.classes[i];
            let _g = crate::parser::model_parser::MixtureClassGuard::enter(c + 1);
            let sub_sg = class_sigma_subst(sigma_values, &mix.class_sigma_over[c]);
            let sg_i: &[f64] = sub_sg.as_deref().unwrap_or(sigma_values);
            crate::stats::likelihood::obs_nll_subject_into_with_schedule(
                model,
                subject,
                theta,
                sg_i,
                &model.residual_correlations,
                &etas[i],
                scratch,
                schedules.get(i).and_then(|s| s.as_ref()),
            )
        })
        .collect();
    per_subj.iter().sum()
}

/// Mixture (#985) + IOV variant of [`obs_nll_sum_iov`], class-guarded per subject.
fn obs_nll_sum_iov_mix(
    model: &CompiledModel,
    population: &Population,
    theta: &[f64],
    sigma_values: &[f64],
    etas: &[Vec<f64>],
    kappas: &[Vec<Vec<f64>>],
    mix: MixMstep<'_>,
) -> f64 {
    use rayon::prelude::*;
    let per_subj: Vec<f64> = population
        .subjects
        .par_iter()
        .enumerate()
        .map_init(EventPkParams::default, |scratch, (i, subject)| {
            let c = mix.classes[i];
            let _g = crate::parser::model_parser::MixtureClassGuard::enter(c + 1);
            let sub_sg = class_sigma_subst(sigma_values, &mix.class_sigma_over[c]);
            let sg_i: &[f64] = sub_sg.as_deref().unwrap_or(sigma_values);
            obs_nll_subject_into_iov(model, subject, theta, sg_i, &etas[i], &kappas[i], scratch)
        })
        .collect();
    per_subj.iter().sum()
}

/// True when a free (non-`FIX`) additive component of a `Combined` endpoint has
/// collapsed onto its optimizer lower bound.
///
/// Sigma is optimized in log space with a lower bound of `exp(-8) ≈ 3.35e-4`
/// (see `parameterization.rs`) and is carried here on the standard-deviation
/// scale. `SIGMA_FLOOR_NEAR = 1e-3` is the detection band just above that hard
/// bound: a value at or below it means the additive term pinned to the floor
/// rather than identifying a genuine non-zero additive error.
fn combined_additive_sigma_at_floor(model: &CompiledModel, params: &ModelParameters) -> bool {
    const SIGMA_FLOOR_NEAR: f64 = 1.0e-3;
    model
        .error_spec
        .combined_additive_sigma_indices()
        .into_iter()
        .any(|idx| {
            !params.sigma_fixed.get(idx).copied().unwrap_or(false)
                && params
                    .sigma
                    .values
                    .get(idx)
                    .copied()
                    .unwrap_or(f64::INFINITY)
                    <= SIGMA_FLOOR_NEAR
        })
}

/// Per-σ growth ceiling (log scale) for iiv_on_ruv SAEM runs (issue #895).
///
/// The IIV-on-RUV parameterization writes the residual variance as
/// `σ²·exp(2·η_RUV)`, so σ and ω_RUV trade off along a ridge; a poorly-mixing
/// E-step can let the M-step ride σ up to its e⁵ ceiling. This returns, per σ,
/// `Some(cap)` = `log σ₀ + SAEM_RUV_SIGMA_LN_GROWTH` for each *free* RUV-scaled
/// residual σ **whose growth cap is stricter than its own NLopt upper bound**,
/// and `None` for any σ that must not carry a growth cap: every σ when the model
/// has no `iiv_on_ruv` (`has_ruv_eta == false`), a FIXed σ, the FREM covariate σ
/// (EPSCOV — always FIX and independent of the RUV scaling), or a σ whose
/// user-set upper bound is already tighter than the growth cap (the NLopt bound
/// governs there, so a σ converging to its own bound must not be mis-flagged as a
/// bound RUV growth cap — #903 review). The cap is enforced as a post-M-step
/// clamp, never as an NLopt bound, so a well-posed fit whose σ stays below the cap
/// is unaffected.
fn compute_ruv_sigma_caps(
    has_ruv_eta: bool,
    frem_cov_sigma: Option<usize>,
    log_sigma_init: &[f64],
    log_sigma_upper: &[f64],
    sigma_fixed: &[bool],
) -> Vec<Option<f64>> {
    let n = log_sigma_init.len();
    if !has_ruv_eta {
        return vec![None; n];
    }
    (0..n)
        .map(|i| {
            if sigma_fixed.get(i).copied().unwrap_or(false) || Some(i) == frem_cov_sigma {
                return None;
            }
            let upper = log_sigma_upper.get(i).copied().unwrap_or(f64::INFINITY);
            let growth_cap = log_sigma_init[i] + SAEM_RUV_SIGMA_LN_GROWTH;
            // If the user's own upper bound is at least as tight, NLopt already
            // enforces it and our clamp/warning would just misattribute that bound
            // to the RUV safeguard — so carry no growth cap here.
            if growth_cap < upper {
                Some(growth_cap)
            } else {
                None
            }
        })
        .collect()
}

/// Whether the `iiv_on_ruv` η may be re-centered while exactly preserving each
/// subject's residual variance (issue #904 correctness gate).
///
/// Re-centering shifts η_RUV by `−mean`, multiplying the *whole* residual variance
/// `R = Σ σ_c²` by `exp(−2·mean)`, and compensates by scaling the free residual σ
/// by `exp(mean)`. The compensation is exact only if *every* non-FREM residual σ
/// component absorbs the shift, i.e. is free. If any RUV-scaled σ component is
/// FIXed (e.g. a fixed additive term of a combined error), scaling only the free
/// component leaves the fixed part uncompensated and silently perturbs `R`, so
/// re-centering must be skipped (the σ/ω_RUV growth caps then act as the
/// backstop; a fixed component also partially anchors the ridge). The FREM EPSCOV
/// (always FIX and not on a real-observation row) is exempt.
fn ruv_recenter_allowed(
    has_ruv_eta: bool,
    frem_cov_sigma: Option<usize>,
    sigma_fixed: &[bool],
) -> bool {
    has_ruv_eta && (0..sigma_fixed.len()).all(|i| Some(i) == frem_cov_sigma || !sigma_fixed[i])
}

/// Re-center the `iiv_on_ruv` η to zero mean, absorbing the shift into the
/// RUV-scaled residual σ (issue #904). Returns the mean that was removed.
///
/// `Y = f + EPS·exp(η_RUV)` has no typical-value θ, so — unlike a mu-referenced
/// structural η — η_RUV's mean is otherwise never absorbed and drifts along the
/// degenerate direction (`σ²·exp(2η)` is invariant to η→η−c, σ→σ·exp(c)),
/// polluting `ω_RUV = mean(η²)` with a spurious `mean²`. Shifting η_RUV by
/// `−mean` and scaling every absorbing σ by `exp(mean)` leaves each subject's
/// residual variance exactly unchanged while restoring `E[η_RUV] = 0`. `σ` is the
/// residual-scale typical value that plays the role of the structural TVP here.
///
/// `absorb_sigma[k]` marks the free, non-FREM, RUV-scaled σ that may take the
/// shift; when none absorb (every RUV σ FIXed) this is a no-op and the mean is
/// left in η_RUV, since a FIXed σ already pins the mean (no degeneracy).
fn recenter_ruv_eta(
    etas: &mut [Vec<f64>],
    kr: usize,
    log_sigma: &mut [f64],
    sigma_vals: &mut [f64],
    absorb_sigma: &[bool],
) -> f64 {
    let n = etas.len();
    if n == 0 || !absorb_sigma.iter().any(|&a| a) {
        return 0.0;
    }
    let mean = etas.iter().map(|e| e[kr]).sum::<f64>() / n as f64;
    for e in etas.iter_mut() {
        e[kr] -= mean;
    }
    for k_s in 0..log_sigma.len() {
        if absorb_sigma.get(k_s).copied().unwrap_or(false) {
            log_sigma[k_s] += mean;
            sigma_vals[k_s] = log_sigma[k_s].exp();
        }
    }
    mean
}

/// Log-variance ceiling for the `iiv_on_ruv` Ω diagonal in a SAEM run (#895).
///
/// Returns `Some(cap)` = `log(ω₀) + SAEM_RUV_OMEGA_LN_GROWTH` for the RUV eta's
/// variance when the model has an `iiv_on_ruv` eta whose Ω is free and in range,
/// else `None` (no `iiv_on_ruv`, out of range, or a FIXed RUV Ω). The cap is
/// enforced as a correlation-preserving rescale after the Ω M-step, so a
/// well-posed fit whose ω_RUV stays below it is unaffected.
fn compute_ruv_omega_cap(
    residual_error_eta: Option<usize>,
    n_eta: usize,
    omega_init: &DMatrix<f64>,
    omega_fixed: &[bool],
) -> Option<f64> {
    let k = residual_error_eta?;
    if k >= n_eta || omega_fixed.get(k).copied().unwrap_or(false) {
        return None;
    }
    let v0 = omega_init[(k, k)];
    if !(v0 > 0.0) {
        return None;
    }
    Some(v0.ln() + SAEM_RUV_OMEGA_LN_GROWTH)
}

/// Cap the `iiv_on_ruv` Ω diagonal at `exp(log_cap)` in place, rescaling the RUV
/// row/column covariances by `√(v_cap/v_old)` so every correlation with the RUV
/// eta is preserved and the matrix stays positive-definite (#895). A no-op when
/// the diagonal is already at or below the cap. Returns `true` when it clamped.
///
/// A FIXed off-diagonal partner `j` (`omega_fixed[j]`) is left untouched: its
/// covariance with the RUV eta is a user-declared constant that the Ω M-step
/// restores verbatim each iteration, so rescaling it would silently mutate a
/// FIXed entry (#903 review). Skipping it can only *lower* the correlation the
/// cap preserves, never break positive-definiteness (the diagonal still shrinks).
fn apply_ruv_omega_cap(
    omega_mat: &mut DMatrix<f64>,
    k: usize,
    log_cap: f64,
    omega_fixed: &[bool],
) -> bool {
    let v_old = omega_mat[(k, k)];
    let v_cap = log_cap.exp();
    if !(v_old > v_cap) {
        return false;
    }
    let s = (v_cap / v_old).sqrt();
    let n = omega_mat.nrows();
    for j in 0..n {
        if j != k && !omega_fixed.get(j).copied().unwrap_or(false) {
            omega_mat[(k, j)] *= s;
            omega_mat[(j, k)] *= s;
        }
    }
    omega_mat[(k, k)] = v_cap;
    true
}

/// Re-anchor the per-σ iiv_on_ruv growth caps to the data-informed σ reached by
/// the end of the exploration phase (#903 review). Each cap is loosened (never
/// tightened) to `max(existing, min(log σ_now + growth, upper))`, so a well-posed
/// fit started from a σ guess far below the truth is not spuriously clamped and
/// falsely flagged, while a genuine post-exploration runaway is still bounded.
fn reanchor_ruv_sigma_caps(caps: &mut [Option<f64>], log_sigma: &[f64], log_sigma_upper: &[f64]) {
    for (i, cap) in caps.iter_mut().enumerate() {
        if let Some(c) = cap {
            let upper = log_sigma_upper.get(i).copied().unwrap_or(f64::INFINITY);
            let settled = (log_sigma[i] + SAEM_RUV_SIGMA_LN_GROWTH).min(upper);
            *c = c.max(settled);
        }
    }
}

/// Re-anchor the iiv_on_ruv Ω growth cap to the ω_RUV variance reached by the end
/// of exploration (#903 review), loosening only. `None` (no cap) stays `None`;
/// a non-positive variance leaves the cap unchanged.
fn reanchor_ruv_omega_cap(cap: Option<f64>, omega_ruv_var: f64) -> Option<f64> {
    cap.map(|c| {
        if omega_ruv_var > 0.0 {
            c.max(omega_ruv_var.ln() + SAEM_RUV_OMEGA_LN_GROWTH)
        } else {
            c
        }
    })
}

/// One post-burn-in iteration's contribution to the acceptance diagnostic.
///
/// `target_weight` is `Σ_k n_proposals_k · target_k` over the kernels that ran
/// this iteration, so a window's **proposal-weighted** target is
/// `Σ target_weight / Σ proposed`. That matters because `accepted / proposed`
/// pools the primary block kernel with the componentwise sweep, and the two
/// have different optimal rates: naming the combined rate after the primary
/// kernel's target alone is wrong whenever the mix is not dominated by it.
/// With HMC it is wrong by a wide margin — one HMC proposal at 0.65 against
/// `n_cw_sweeps · n_eta` componentwise proposals at 0.44 gives a combined
/// target near 0.45, and a two-η fixture measured 0.4978 realised against the
/// 0.65 that used to be printed (#1451 review).
#[derive(Debug, Clone, Copy)]
struct MhRateSample {
    /// 1-based SAEM iteration this sample came from.
    iter: usize,
    accepted: u64,
    proposed: u64,
    target_weight: f64,
}

/// The trailing post-burn-in samples the diagnostic reads, plus whether the
/// step scales had already been adapted when each of them was drawn.
struct MhRateWindow<'a> {
    samples: &'a [MhRateSample],
    /// 1-based iteration at which the block/componentwise scales were first
    /// adapted, or `None` if they never were.
    ///
    /// A scale adapted at the end of iteration `k` first affects sampling at
    /// `k + 1`, so the tail tier speaks only when `first_adapt < samples[0].iter`
    /// — i.e. **every** iteration whose rate it is about to report was drawn
    /// with a scale the controller had already touched. Without this the
    /// diagnostic says a run "never reached" its target on a run where the rule
    /// never ran: measured on a 45-iteration fit at `omega_burnin = 20`,
    /// `adapt_interval = 50`, which reported "settled at 8.7%" after **zero**
    /// adaptations (#1451 review).
    first_adapt: Option<usize>,
}

impl MhRateWindow<'_> {
    /// Combined acceptance over the window, or `None` when it made no
    /// proposal. This is the single definition of "the tail rate": the
    /// diagnostic below and `FitResult::saem_mh_accept_tail` both read it, so
    /// the number a user sees in the message and the number they can assert on
    /// cannot drift apart.
    fn tail_rate(&self) -> Option<f64> {
        let prop: u64 = self.samples.iter().map(|s| s.proposed).sum();
        if prop == 0 {
            return None;
        }
        let acc: u64 = self.samples.iter().map(|s| s.accepted).sum();
        Some(acc as f64 / prop as f64)
    }

    /// Whether the controller had acted before every sample in the window.
    fn adapted_throughout(&self) -> bool {
        match (self.first_adapt, self.samples.first()) {
            (Some(k0), Some(first)) => k0 < first.iter,
            _ => false,
        }
    }
}

/// Decide whether SAEM should warn about its E-step acceptance rate.
///
/// `cum_acc` / `cum_prop` are the run-cumulative combined (block +
/// componentwise) MH accept / proposal counts over the post-burn-in iterations.
/// `window` holds the last [`MH_RATE_WINDOW`] post-burn-in samples — the
/// *converged tail*, which is what a user can act on; a run that mixes badly
/// early and well later averages to a number describing neither — together
/// with when the scales were first adapted. The target the message quotes is
/// **proposal-weighted** across the kernels that ran (see [`MhRateSample`]),
/// not the primary kernel's alone. `mode` is the active adaptation rule, which
/// decides what the message can usefully suggest.
///
/// Two tiers, most severe first:
///
/// 1. **Not mixing** — cumulative rate below [`SAEM_MH_STUCK_ACCEPT`]. The
///    sampled ETAs barely moved at all, so the M-step ran on degenerate
///    sufficient statistics and Ω/σ are unreliable. Keeps its historical
///    wording (issue #895), which `tests/frem_warfarin.rs` asserts the absence
///    of.
/// 2. **Persistently far from target** — the tail rate is outside
///    `[MH_RATE_LOW, MH_RATE_HIGH]`. Not necessarily wrong, but it means the
///    step scales never found their target and the chain is either crawling
///    (too-large steps, almost everything rejected) or barely moving per
///    accepted step (too-small steps, almost everything accepted). This tier
///    exists because tier 1's 1% threshold is low enough that a chain stuck at
///    2-4% for an entire run — which `examples/warfarin_saem.ferx` does, and
///    which costs real estimate quality — passed silently (issue #1444).
///
///    It is gated on the controller having *run*, not merely on the window
///    being long enough: "never reached its target" is a claim about an
///    adaptation that failed, and it must not be made about one that never
///    happened. See [`MhRateWindow::first_adapt`].
///
/// Both messages carry the numbers, so the reader does not have to re-run with
/// `optimizer_trace` to find out how far off it was.
fn saem_mixing_warning(
    cum_acc: u64,
    cum_prop: u64,
    window: &MhRateWindow<'_>,
    mode: crate::types::ScaleAdaptation,
) -> Option<String> {
    if cum_prop == 0 {
        return None;
    }
    let rate = cum_acc as f64 / cum_prop as f64;
    if rate < SAEM_MH_STUCK_ACCEPT {
        return Some(format!(
            "SAEM Metropolis-Hastings acceptance was {:.2}% over the post-burn-in \
             iterations — the E-step is not mixing, so Ω/σ estimates are unreliable. \
             Check for extreme Ω-diagonal scale differences (e.g. FREM covariate ETAs) \
             or a mis-scaled initial Ω.",
            rate * 100.0
        ));
    }

    let samples = window.samples;
    if samples.len() < MH_RATE_MIN_WINDOW || !window.adapted_throughout() {
        return None;
    }
    let w_prop: u64 = samples.iter().map(|s| s.proposed).sum();
    let w_rate = window.tail_rate()?;
    if !(MH_RATE_LOW..=MH_RATE_HIGH).contains(&w_rate) {
        // Proposal-weighted across the kernels that actually ran, so a run
        // whose proposals are mostly componentwise is judged against 0.44 and
        // not against the primary kernel's 0.40 (or HMC's 0.65).
        let w_target: f64 = samples.iter().map(|s| s.target_weight).sum::<f64>() / w_prop as f64;
        let direction = if w_rate < MH_RATE_LOW {
            "too large a step — most proposals are rejected"
        } else {
            "too small a step — almost every proposal is accepted, so each one moves little"
        };
        // The remedy has to depend on what is already running: telling a
        // Robbins-Monro run to switch to Robbins-Monro is a no-op (#1451
        // review). A chain that is still outside the band under the
        // per-iteration rule has hit a scale clamp or has an Ω so mis-scaled
        // that no step size helps, which is a different repair.
        let remedy = match mode {
            crate::types::ScaleAdaptation::Interval => {
                "Consider `[fit_options] scale_adaptation = robbins_monro`, which steps the \
                 scales every iteration rather than once per `adapt_interval`, or lower \
                 `adapt_interval` so the existing rule fires more often."
            }
            crate::types::ScaleAdaptation::RobbinsMonro => {
                "`scale_adaptation = robbins_monro` is already active, so the step scale has \
                 most likely hit a clamp or the chain is too short for it to arrive: check the \
                 scale of the initial Ω (a Ω-diagonal orders of magnitude from the posterior \
                 spread cannot be rescued by any step size) and consider more exploration \
                 iterations."
            }
        };
        return Some(format!(
            "SAEM Metropolis-Hastings acceptance settled at {:.1}% over the last {} \
             iterations, against a proposal-weighted target of {:.0}% ({:.1}% over all \
             post-burn-in iterations). That is {direction}, so the E-step explored less of \
             each subject's conditional distribution than it paid for and the Ω/σ estimates \
             carry more Monte-Carlo noise than they need to. {remedy}",
            w_rate * 100.0,
            samples.len(),
            w_target * 100.0,
            rate * 100.0
        ));
    }
    None
}

/// Build (theta_idx, eta_idx) pairs eligible for the closed-form EM M-step.
///
/// The complete-data maximiser of a mu-referenced parameter
/// `P_i = g⁻¹(g(θ) + η_i)` is `g(θ)_new = g(θ)_old + mean_i(η_i)` — the update
/// is *link-independent*: it holds for `g = log` (`P = θ·exp(η)`), for
/// `g = id` (`P = θ + η`) and for `g = logit` (`P = inv_logit(θ + η)`) alike.
/// What it requires is that the quantity SAEM actually steps — the **packed**
/// theta — *is* `g(θ)`. `run_saem` packs a theta as `log θ` when its lower
/// bound admits it (`theta_packs_log`) and as `θ` otherwise, so:
///
/// | mu transform       | mu scale   | eligible when            |
/// |--------------------|------------|--------------------------|
/// | `Log`              | `log θ`    | theta is **log**-packed  |
/// | `Logit`            | `θ`        | theta is identity-packed |
/// | `Identity`         | `θ`        | never (see below)        |
/// | `LogitProbability` | `logit θ`  | never — no packing matches |
///
/// A `Logit` mu-ref (`P = inv_logit(THETA + ETA)`) declares its theta on the
/// logit scale, so its lower bound is negative and it is identity-packed — the
/// packed value *is* the mu (#918).
///
/// Additive mu-refs are left out even when identity-packed: they are the
/// historical behaviour of this function, they are rare in practice, and the
/// change is not needed to fix #918. `LogitProbability`
/// (`inv_logit(logit(THETA) + ETA)`) has no packing whose scale is `logit θ`,
/// so the closed form does not apply. Both fall through to the regular NLopt
/// M-step, which is correct for any parameterisation, just slower.
///
/// `theta_lower` is the same lower-bound vector `run_saem` derives its packing
/// mask from (`init_params.theta_lower`), not `model.default_params`, so a
/// caller-overridden bound cannot desynchronise the two.
#[cfg(test)]
pub(crate) fn get_mu_ref_pairs(model: &CompiledModel, theta_lower: &[f64]) -> Vec<(usize, usize)> {
    classify_mu_ref_pairs(model, theta_lower).eligible
}

/// The full eligibility split behind the test-only `get_mu_ref_pairs` wrapper: the pairs the
/// closed-form M-step may take, plus the anchors it had to *drop* because the
/// theta's packing does not match its mu scale. Callers surface the latter as
/// advisories (#996 review, #918) — a user who declared `theta TVCL(1, -5, 100)`
/// or `theta LOGIT_F(0.5, 0, 5)` gets told why that theta sits on the numerical
/// M-step instead of the exact shift, and how to fix the declaration.
///
/// All dropped lists are sorted and de-duplicated. Additive and
/// probability-scale-logit anchors are neither eligible nor "dropped": there is
/// no bound the user could change to make the closed form apply.
pub(crate) struct MuRefPairs {
    /// `(theta_idx, eta_idx)` pairs whose packed scale is their mu scale, each
    /// theta owned by exactly one eta (see `shared_theta`).
    pub eligible: Vec<(usize, usize)>,
    /// Lognormal anchors (`THETA*exp(ETA)`) whose theta is identity-packed
    /// because its lower bound is negative.
    pub identity_packed_log: Vec<usize>,
    /// Logit-scale anchors (`inv_logit(THETA + ETA)`) whose theta is log-packed
    /// because its lower bound is non-negative.
    pub log_packed_logit: Vec<usize>,
    /// Thetas anchoring **more than one** eta (`F1 = inv_logit(LOGIT_F + ETA_F1)`
    /// and `F2 = inv_logit(LOGIT_F + ETA_F2)`, or the lognormal analogue
    /// `CL = TVP*exp(ETA_CL)` / `V = TVP*exp(ETA_V)`). Excluded from `eligible`
    /// entirely — see the note on `classify_mu_ref_pairs`.
    pub shared_theta: Vec<usize>,
    /// The pairs those shared thetas would have contributed. They take no
    /// closed-form shift, but they are still declared mu-references, so
    /// `resolve_covariate_mu_groups` must keep seeing them: a #619 covariate
    /// group may not claim a theta another eta anchors, whether or not the
    /// single-anchor channel ended up running (`mu_ref_pairs_for_cov_groups`).
    pub shared_pairs: Vec<(usize, usize)>,
}

/// The anchor pairs a #619 covariate-mu-ref group must not collide with: every
/// declared single-anchor pair, whether or not the closed form runs for it.
///
/// `eligible` alone is the wrong input — a theta dropped for anchoring two etas
/// is *more* contested, not less — and so is `eligible` plus a packing-dropped
/// theta, which the group step is free to move (it does not use the packed
/// scale). Only the shared-anchor drop is added back.
pub(crate) fn mu_ref_pairs_for_cov_groups(split: &MuRefPairs) -> Vec<(usize, usize)> {
    let mut v = split.eligible.clone();
    v.extend(split.shared_pairs.iter().copied());
    v
}

/// True when the **packed** theta — the quantity the optimiser steps — *is* the
/// mu scale `g(θ)`, which is the one condition the link-independent closed-form
/// shift `g(θ) += mean(η)` needs (#918).
///
/// `packs_log` is `theta_packs_log(lower)`: ferx packs a theta as `log θ` when
/// its lower bound admits it and as `θ` otherwise. So `Log` needs log-packing,
/// `Logit` (whose theta is *already* on the logit scale) needs identity-packing,
/// and `Identity` / `LogitProbability` never qualify — see the table on
/// [`classify_mu_ref_pairs`]'s wrapper `get_mu_ref_pairs`.
pub(crate) fn packed_scale_is_mu(transform: MuTransform, packs_log: bool) -> bool {
    matches!(
        (transform, packs_log),
        (MuTransform::Log, true) | (MuTransform::Logit, false)
    )
}

/// Split the model's mu-ref anchors into the closed-form-eligible pairs and the
/// ones that have to fall back to the numerical M-step.
///
/// **One eta per theta.** The closed form shifts a packed theta by that eta's
/// mean and then pins it for the numerical M-step, so a theta anchoring two etas
/// has no well-defined single shift: applying both moves it twice, and applying
/// only the first leaves the second eta un-recentred while pinning the theta
/// away from its joint optimum (ω for that second eta then inflates to absorb
/// the drift). The joint complete-data maximiser is a precision-weighted mean of
/// the two eta means, which this function does not compute — so a shared theta
/// and *both* its pairs are dropped, leaving the theta to the numerical M-step,
/// which is correct for any number of anchored etas.
///
/// Sharing is counted over **every declared anchor**, not over the eligible
/// ones: an ineligible anchor (`V = TVP + ETA_V`) still leaves its eta
/// un-recentred when the shift pins the theta on behalf of an eligible one
/// (#918 review).
///
/// This is deliberately stricter than [`get_mixture_mu_ref_pairs`], whose
/// class-aware rule keeps the first eta to claim a theta (#996). Both avoid the
/// double shift; only this one also avoids the pinned partial shift.
pub(crate) fn classify_mu_ref_pairs(model: &CompiledModel, theta_lower: &[f64]) -> MuRefPairs {
    let mut out = MuRefPairs {
        eligible: Vec::new(),
        identity_packed_log: Vec::new(),
        log_packed_logit: Vec::new(),
        shared_theta: Vec::new(),
        shared_pairs: Vec::new(),
    };
    // Every anchor the model declares, *before* the eligibility split — the
    // shared-theta rule below has to see the ineligible ones too. A theta
    // anchoring one eligible and one ineligible eta is exactly the pinned
    // partial shift this function exists to prevent: `CL = TVP*exp(ETA_CL)`
    // (log-packed Log, eligible) next to `V = TVP + ETA_V` (Identity, never
    // eligible) would shift and pin `TVP` by `mean(η_CL)` while `ETA_V` is
    // never re-centred for that delta. Scanning `eligible` alone misses it
    // (#918 review).
    let mut declared: Vec<(usize, usize, bool)> = Vec::new();
    for (eta_idx, eta_name) in model.eta_names.iter().enumerate() {
        let Some(mu_ref) = model.mu_refs.get(eta_name) else {
            continue;
        };
        let Some(theta_idx) = model
            .theta_names
            .iter()
            .position(|n| n == &mu_ref.theta_name)
        else {
            continue;
        };
        let Some(&lower) = theta_lower.get(theta_idx) else {
            continue;
        };
        let packs_log = crate::estimation::parameterization::theta_packs_log(lower);
        let eligible = packed_scale_is_mu(mu_ref.transform, packs_log);
        if !eligible {
            // Not eligible: name the ones a bound change could rescue.
            match mu_ref.transform {
                MuTransform::Log => out.identity_packed_log.push(theta_idx),
                MuTransform::Logit => out.log_packed_logit.push(theta_idx),
                MuTransform::Identity | MuTransform::LogitProbability => {}
            }
        }
        declared.push((theta_idx, eta_idx, eligible));
    }
    // Drop every pair on a theta claimed by more than one eta (see the doc
    // comment): both pairs go, the theta goes to the numerical M-step.
    let mut shared: Vec<usize> = Vec::new();
    for (i, &(t, _, _)) in declared.iter().enumerate() {
        if declared
            .iter()
            .enumerate()
            .any(|(j, &(t2, _, _))| j != i && t2 == t)
        {
            shared.push(t);
        }
    }
    for &(t, e, eligible) in &declared {
        if !eligible {
            continue;
        }
        if shared.contains(&t) {
            out.shared_pairs.push((t, e));
        } else {
            out.eligible.push((t, e));
        }
    }
    // Only name a theta whose *eligible* pair the rule actually dropped. Two
    // ineligible anchors on one theta share nothing the closed form would have
    // moved, and reporting them would read as a second, unrelated reason the
    // packing advisory above has already covered.
    out.shared_theta = shared
        .iter()
        .copied()
        .filter(|t| out.shared_pairs.iter().any(|&(t2, _)| t2 == *t))
        .collect();
    for v in [
        &mut out.identity_packed_log,
        &mut out.log_packed_logit,
        &mut out.shared_theta,
    ] {
        v.sort_unstable();
        v.dedup();
    }
    out
}

/// A class-aware (`MIXNUM`-switched) log-mu-ref pair: one eta paired with one
/// anchor theta **per class** (#996).
///
/// A class-shared typical value (`V = TVV * exp(ETA_V)` in a mixture model) is
/// represented as the same theta index repeated `n_classes` times, so both the
/// switched and the shared case run through one update rule — and the
/// all-classes-share-one-theta case reduces exactly to the classical pooled
/// `log θ += γ · mean(η)` shift.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MixtureMuRefPair {
    /// Index into `model.eta_names`.
    pub eta_idx: usize,
    /// Anchor theta index per class; `theta_idx[c]` serves class `c` (0-based).
    /// Length is always the mixture's `n_classes`.
    pub theta_idx: Vec<usize>,
    /// The link relating the anchor theta(s) to the individual parameter, which
    /// decides the packing the closed form needs (`packed_scale_is_mu`).
    /// `MIXNUM`-switched anchors are always [`MuTransform::Log`] — the parser
    /// detects no other switched pattern; a class-*shared* anchor may also be
    /// [`MuTransform::Logit`] (#918).
    pub transform: MuTransform,
}

/// Build the class-aware log-mu-ref pairs for a mixture model (#996).
///
/// Returns empty for a non-mixture model — use [`classify_mu_ref_pairs`] there.
/// Each eta contributes at most one pair: the class-aware anchor set detected
/// by the parser (`MixtureSpec::mu_refs`) when the typical value is
/// `MIXNUM`-switched, otherwise the classical single-theta mu-ref broadcast
/// across all classes.
///
/// The link is carried on the pair rather than filtered here: a class-shared
/// `F = inv_logit(LOGIT_F + ETA_F)` is as eligible as the lognormal form, since
/// the shift is link-independent and what it needs is that the packed theta *is*
/// the mu (#918). The caller applies that check with `packed_scale_is_mu`.
/// Additive (`THETA + ETA`) and probability-scale logit
/// (`inv_logit(logit(THETA) + ETA)`) mu-refs are excluded for the same reason as
/// in [`classify_mu_ref_pairs`]: no packing has their mu scale. A
/// `MIXNUM`-switched typical value is log-only — `detect_mixture_pattern` does
/// not recognise a switched logit chain, so such an eta has no mu-ref at all and
/// never reaches this function.
///
/// A theta is claimed by **at most one** pair. Two etas anchored to the same
/// typical value (`CL = TVP*exp(ETA_CL)` and `V = TVP*exp(ETA_V)`) have no
/// well-defined joint closed form — applying both shifts would move that θ twice
/// in one iteration — so the first eta to claim a θ keeps it and any later pair
/// that reuses it is dropped, leaving those θ to the numerical / weighted M-step
/// (#996 review).
pub(crate) fn get_mixture_mu_ref_pairs(model: &CompiledModel) -> Vec<MixtureMuRefPair> {
    let Some(spec) = model.mixture.as_ref() else {
        return Vec::new();
    };
    let k = spec.n_classes;
    let idx_of = |name: &str| model.theta_names.iter().position(|n| n == name);
    let mut out: Vec<MixtureMuRefPair> = Vec::new();
    let mut claimed: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut push_pair = |out: &mut Vec<MixtureMuRefPair>, pair: MixtureMuRefPair| {
        if pair.theta_idx.iter().any(|t| claimed.contains(t)) {
            return;
        }
        claimed.extend(pair.theta_idx.iter().copied());
        out.push(pair);
    };
    for (eta_idx, eta_name) in model.eta_names.iter().enumerate() {
        if let Some(m) = spec.mu_refs.iter().find(|m| &m.eta_name == eta_name) {
            if !m.log_transformed || m.theta_names.len() != k {
                continue;
            }
            let Some(theta_idx) = m
                .theta_names
                .iter()
                .map(|n| idx_of(n))
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            push_pair(
                &mut out,
                MixtureMuRefPair {
                    eta_idx,
                    theta_idx,
                    transform: MuTransform::Log,
                },
            );
        } else if let Some(mu_ref) = model.mu_refs.get(eta_name) {
            if !matches!(mu_ref.transform, MuTransform::Log | MuTransform::Logit) {
                continue;
            }
            let Some(t) = idx_of(&mu_ref.theta_name) else {
                continue;
            };
            push_pair(
                &mut out,
                MixtureMuRefPair {
                    eta_idx,
                    theta_idx: vec![t; k],
                    transform: mu_ref.transform,
                },
            );
        }
    }
    out
}

/// The packing split of [`get_mixture_mu_ref_pairs`], the mixture twin of
/// [`MuRefPairs`]: the pairs whose every class theta is packed on its mu scale,
/// and the thetas dropped because it is not.
pub(crate) struct MixtureMuRefPairs {
    /// Pairs the class-aware closed-form shift may take.
    pub eligible: Vec<MixtureMuRefPair>,
    /// Lognormal class anchors packed on the identity scale (#996).
    pub identity_packed_log: Vec<usize>,
    /// Logit-scale class anchors packed on the log scale (#918).
    pub log_packed_logit: Vec<usize>,
}

/// Apply the packing eligibility rule to a mixture model's mu-ref pairs.
///
/// Shared by `run_saem` and `run_mcem` so the two cannot disagree about which
/// class anchors the closed form owns — the same reason `classify_mu_ref_pairs`
/// is shared by their single-population paths. Each caller adds only what is
/// genuinely its own: SAEM additionally drops `MIXNUM`-switched anchors (its
/// hard class draw biases the per-class mean, #996), IMP/IMPMAP additionally
/// drops pairs whose eta has negligible IIV (#411).
///
/// `theta_packs_log_mask[t]` is `theta_packs_log(theta_lower[t])` — the same
/// per-theta packing the caller's `log_theta` vector was built with.
pub(crate) fn classify_mixture_mu_ref_pairs(
    model: &CompiledModel,
    theta_packs_log_mask: &[bool],
) -> MixtureMuRefPairs {
    let mut out = MixtureMuRefPairs {
        eligible: Vec::new(),
        identity_packed_log: Vec::new(),
        log_packed_logit: Vec::new(),
    };
    for p in get_mixture_mu_ref_pairs(model) {
        let mismatched: Vec<usize> = p
            .theta_idx
            .iter()
            .copied()
            .filter(|&t| !packed_scale_is_mu(p.transform, theta_packs_log_mask[t]))
            .collect();
        if mismatched.is_empty() {
            out.eligible.push(p);
            continue;
        }
        // Name only the thetas whose own bound is the problem: the advisory
        // tells the user which declaration to change.
        match p.transform {
            MuTransform::Logit => out.log_packed_logit.extend(mismatched),
            _ => out.identity_packed_log.extend(mismatched),
        }
    }
    for v in [&mut out.identity_packed_log, &mut out.log_packed_logit] {
        v.sort_unstable();
        v.dedup();
    }
    out
}

/// Responsibility-weighted per-class mu-ref mean shift (#996).
///
/// For each anchor theta `t`, returns `(Σ_i Σ_{c: θ_c = t} r_ic · η̄_ic) /
/// (Σ_i Σ_{c: θ_c = t} r_ic)` — the EM-optimal `Δ log θ_t` for the complete-data
/// log-likelihood of `log P_i = log θ_{c_i} + η_i` restricted to the members of
/// the classes that theta serves. `resp[i][c]` is subject `i`'s weight in class
/// `c`: a hard 0/1 indicator under SAEM (which draws one class per subject) and
/// the responsibility `PMIX_ic` under IMP/IMPMAP (which importance-samples
/// within every class). `eta_at(i, c)` supplies subject `i`'s η mean under
/// class `c` for the eta this pair carries.
///
/// The result is `None` for any theta no class weight reached — the label-switch
/// case where a class won zero subjects this iteration. The caller holds that
/// θ_k and lets the next iteration move it, rather than dividing by zero or
/// falling back to a pooled mean that would drag the class toward its neighbours.
///
/// When every class maps to the *same* theta the weights sum to one per subject
/// and this collapses to the pooled `mean_i(η_i)` of the classical mu-ref
/// update, in the same accumulation order — which is what makes the degenerate
/// single-theta mixture reproduce the non-mixture path exactly.
pub(crate) fn mixture_mu_ref_means(
    n_theta: usize,
    theta_idx_per_class: &[usize],
    resp: &[Vec<f64>],
    eta_at: impl Fn(usize, usize) -> f64,
) -> Vec<Option<f64>> {
    let mut numer = vec![0.0f64; n_theta];
    let mut denom = vec![0.0f64; n_theta];
    for (i, r_i) in resp.iter().enumerate() {
        for (c, &t) in theta_idx_per_class.iter().enumerate() {
            if t >= n_theta {
                continue;
            }
            let r = r_i.get(c).copied().unwrap_or(0.0);
            if r <= 0.0 {
                continue;
            }
            numer[t] += r * eta_at(i, c);
            denom[t] += r;
        }
    }
    (0..n_theta)
        .map(|t| (denom[t] > 0.0).then(|| numer[t] / denom[t]))
        .collect()
}

/// One-line description of the SAEM E-step sampler kernel, for the startup
/// banner. SAEM's estimation is sampling-based (not gradient-driven), so the
/// banner reports the kernel here instead of a gradient route. HMC is used
/// only when `saem_n_leapfrog > 0` on an analytical PK model (its η-gradient is
/// the analytic Dual2 gradient) — the same gate as [`run_saem`]; this mirrors
/// that condition so the banner reflects what will actually run.
pub(crate) fn saem_sampler_summary(model: &CompiledModel, options: &FitOptions) -> String {
    let n_leapfrog = options.saem_n_leapfrog;
    // HMC is BSV-only (`hmc_step` and the AD NLL/gradient are kappa-unaware), so
    // it is disabled for IOV models (`n_kappa > 0`); those subjects use the MH
    // kernels, whose acceptance targets the IOV conditional p(η | κ, θ, data).
    // IIV on residual error (#409) also disables HMC: the Dual2 gradient kernel
    // carries no `exp(2·η_ruv)` variance-scaling rule, so these models fall back
    // to MH (same gate as [`run_saem`]).
    let using_hmc = n_leapfrog > 0
        && model.ode_spec.is_none()
        && model.tv_fn.is_some()
        && model.n_kappa == 0
        && model.residual_error_eta.is_none();
    if using_hmc {
        format!("HMC ({n_leapfrog} leapfrog steps, Dual2 analytic gradients)")
    } else if n_leapfrog > 0 {
        "Metropolis-Hastings random walk \
         (HMC requested but unavailable — needs an analytical PK model, no IOV)"
            .to_string()
    } else {
        "Metropolis-Hastings random walk".to_string()
    }
}

/// Assemble a `ModelParameters` snapshot from the current SAEM `state`. Shared
/// by the final post-loop parameter build and by the periodic resume checkpoint
/// (#755) so the two never drift. `n_kappa > 0` preserves the IOV Ω structural
/// free-mask (see the inline note at the call site).
fn saem_state_to_params(
    state: &SaemState,
    init_params: &ModelParameters,
    n_kappa: usize,
) -> ModelParameters {
    let omega = OmegaMatrix::from_matrix(
        state.omega_mat.clone(),
        init_params.omega.eta_names.clone(),
        init_params.omega.diagonal,
    );
    ModelParameters {
        theta: state.theta.clone(),
        theta_names: init_params.theta_names.clone(),
        theta_lower: init_params.theta_lower.clone(),
        theta_upper: init_params.theta_upper.clone(),
        theta_fixed: init_params.theta_fixed.clone(),
        omega,
        omega_fixed: init_params.omega_fixed.clone(),
        sigma: SigmaVector {
            values: state.sigma_vals.clone(),
            names: init_params.sigma.names.clone(),
        },
        sigma_fixed: init_params.sigma_fixed.clone(),
        // SAEM does not update the `block_sigma` off-diagonals (#847); carry them
        // through so a chained estimator (e.g. `[saem, foce]`) and the result
        // snapshot both keep the declared residual covariance structure.
        residual_correlations: init_params.residual_correlations.clone(),
        residual_correlation_fixed: init_params.residual_correlation_fixed.clone(),
        omega_iov: if n_kappa > 0 {
            // Use from_matrix_with_mask so the structural free_mask is preserved
            // when this snapshot is handed to a chained estimator (e.g.
            // [saem, foce]); from_matrix would infer the mask from nonzeros and
            // could mark a legitimately-zero off-diagonal as structurally fixed.
            init_params.omega_iov.as_ref().map(|iov_ref| {
                OmegaMatrix::from_matrix_with_mask(
                    state.omega_iov_mat.clone(),
                    iov_ref.eta_names.clone(),
                    iov_ref.diagonal,
                    iov_ref.free_mask.clone(),
                )
            })
        } else {
            init_params.omega_iov.clone()
        },
        kappa_fixed: init_params.kappa_fixed.clone(),
        mixture: None,
    }
}

// ---------------------------------------------------------------------------
// E-step sizing
// ---------------------------------------------------------------------------

/// Value of [`FitOptions::saem_n_mh_steps`] that selects
/// [`auto_n_mh_steps`] — the shipped default, written `n_mh_steps = auto` in a
/// model file.
pub(crate) const SAEM_N_MH_STEPS_AUTO: usize = 0;

/// Block proposals the automatic rule asks for per observation per random
/// effect (issue #1459).
const AUTO_MH_STEPS_PER_OBS_PER_ETA: f64 = 2.5;

/// Floor of the automatic block-proposal count.
pub(crate) const AUTO_MH_STEPS_MIN: usize = 6;

/// Cap of the automatic block-proposal count — the pre-#1459 fixed default, so
/// a dataset dense enough to need it keeps exactly the historical E-step.
pub(crate) const AUTO_MH_STEPS_MAX: usize = 20;

/// Block-kernel proposals per subject per SAEM iteration, chosen from the shape
/// of the dataset (issue #1459).
///
/// ## Why this is not a constant
///
/// The step-scale controller holds the block kernel at its 40 % acceptance
/// target on every sparse benchmark measured (0.39–0.46 realised on cefepime,
/// vancomycin, busulfan and pembrolizumab, at every count from 2 to 20), so
/// each proposal there is worth the same fraction of a move, and 20 of them
/// buy the same E-step as 6 for roughly three times the proposal traffic. The one benchmark where extra
/// proposals *do* pay is the one where the controller cannot reach its target:
/// the Emax PKPD model of `docs/estimation/saem.qmd`, 16 observations per
/// subject against 2 η, realises **0.246**, and its final-estimate distance and
/// cross-seed stability are 2–3× better at 20 proposals than at 4–8 (measured,
/// 6 seeds — see the issue). A conditional much narrower than the prior is what
/// both facts have in common: it drives the adapted scale down, and it is what
/// makes a warm-started chain slow to follow a moving mode.
///
/// Observations per subject per η is the cheapest available proxy for that
/// narrowing — it is the information the conditional is built from — so the
/// rule is `2.5 · n_obs / (n_subjects · n_eta)`, clamped to
/// `[AUTO_MH_STEPS_MIN, AUTO_MH_STEPS_MAX]`. The cap is the historical fixed
/// default, so nothing changes on a dataset dense enough to earn it; the floor
/// is where the sparse benchmarks stop being distinguishable from 20.
///
/// It is a rule about the *dataset*, not about the run: identical inputs give
/// an identical count, so a fit stays reproducible. `verbose` prints the
/// resolved value, and an explicit `n_mh_steps = <n>` overrides it entirely.
pub(crate) fn auto_n_mh_steps(n_obs: usize, n_subjects: usize, n_eta: usize) -> usize {
    if n_subjects == 0 || n_eta == 0 {
        // Nothing to read the density off. Neither shape reaches a kernel —
        // SAEM rejects `n_eta == 0` up front and an empty population earlier
        // still — so this is only about not dividing by zero; answer with the
        // historical default rather than with the cheap end.
        return AUTO_MH_STEPS_MAX;
    }
    let obs_per_eta = n_obs as f64 / (n_subjects as f64 * n_eta as f64);
    let steps = (AUTO_MH_STEPS_PER_OBS_PER_ETA * obs_per_eta).round();
    // `steps` is finite and non-negative here (n_obs / positive), so the cast
    // after the clamp cannot be a saturating surprise.
    (steps as usize).clamp(AUTO_MH_STEPS_MIN, AUTO_MH_STEPS_MAX)
}

/// The η-block proposal count for the **Bayes** sampler: the requested value,
/// or — under [`SAEM_N_MH_STEPS_AUTO`] — the historical fixed count, *not*
/// [`auto_n_mh_steps`].
///
/// [`auto_n_mh_steps`] is calibrated on SAEM quantities (a controller-held
/// acceptance rate, final-estimate distance against a long-run reference), and
/// what makes a low count safe there is SAEM's componentwise kernel carrying
/// the dense case (#1466). The Bayes η block is the block kernel alone and is
/// judged on posterior mixing, which none of that measured — so `auto` means
/// "unchanged" here until there is a Bayes benchmark to move it (#1459).
pub(crate) fn resolve_n_mh_steps_bayes(requested: usize) -> usize {
    if requested == SAEM_N_MH_STEPS_AUTO {
        AUTO_MH_STEPS_MAX
    } else {
        requested
    }
}

/// The `verbose` line describing the resolved E-step kernel sizes. Kept as a
/// pure function so the wording is unit-testable, like
/// [`saem_final_ofv_report`].
fn mh_steps_report(requested: usize, resolved: usize, n_cw_sweeps: usize) -> String {
    let origin = if requested == SAEM_N_MH_STEPS_AUTO {
        "auto, from observations per subject per eta"
    } else {
        "set in fit options"
    };
    format!(
        "SAEM: {resolved} block MH proposals/subject/iteration ({origin}), \
         {n_cw_sweeps} componentwise sweeps"
    )
}

/// The block-proposal count a run actually uses: the requested value, or
/// [`auto_n_mh_steps`] when the caller left it at [`SAEM_N_MH_STEPS_AUTO`].
///
/// Every consumer of [`FitOptions::saem_n_mh_steps`] goes through here — the
/// main loop, the conditional-distribution pass and the Bayes η block — so the
/// sentinel cannot reach a kernel as a literal zero-proposal count.
pub(crate) fn resolve_n_mh_steps(
    requested: usize,
    n_obs: usize,
    n_subjects: usize,
    n_eta: usize,
) -> usize {
    if requested == SAEM_N_MH_STEPS_AUTO {
        auto_n_mh_steps(n_obs, n_subjects, n_eta)
    } else {
        requested
    }
}

/// Componentwise sweeps per E-step iteration (Kuhn–Lavielle kernel 2), given
/// the block-kernel proposal count.
///
/// Each sweep is `n_eta` single-coordinate proposals, so sizing it
/// `n_mh_steps / n_eta` keeps the kernel's NLL-eval cost roughly on par with
/// the block kernel, at a floor of 2 sweeps — the kernel is what stops a block
/// Ω collapsing to a near rank-1 correlation matrix (#191), so it must not
/// vanish when the block count is small. Skipped entirely for single-η models,
/// where there is no off-diagonal to decorrelate and the kernel would only
/// duplicate the block move.
///
/// Shared by the main loop and the conditional-distribution pass
/// ([`saem_conddist`](crate::estimation::saem_conddist)) so the two cannot
/// drift: both sample the same conditional with the same two kernels, and a
/// second copy of this arithmetic is a second definition of the E-step.
pub(crate) fn componentwise_sweeps(n_mh_steps: usize, n_eta: usize) -> usize {
    if n_eta >= 2 {
        (n_mh_steps / n_eta).max(2)
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// Main SAEM loop
// ---------------------------------------------------------------------------

/// Progress line printed once SAEM's final OFV is known, *before* the covariance
/// step runs (#893). SAEM learns its OFV only at the very end (the final FOCE
/// approximation), and the covariance step is often the most expensive part of
/// the run, so reporting the OFV first lets a CLI user interrupt (Ctrl-C) on a
/// bad OFV before paying for the covariance matrix. Kept as a pure function so
/// the message is unit-testable.
fn saem_final_ofv_report(ofv: f64) -> String {
    format!("SAEM completed. Final OFV = {:.4}", ofv)
}

pub fn run_saem(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> Result<OuterResult, String> {
    let n_subjects = population.subjects.len();
    let n_eta = model.n_eta;
    let n_kappa = model.n_kappa;
    let k1 = options.saem_n_exploration;
    let k2 = options.saem_n_convergence;
    let n_iter = k1 + k2;
    // Suppress the Ω M-step for the first `omega_burnin` iterations so the MH
    // chain warms up at the initial Ω before any variance component is
    // estimated. Clamped to the exploration length — burning in past K1 would
    // freeze Ω into the convergence phase. See `FitOptions::saem_omega_burnin`.
    let omega_burnin = options.saem_omega_burnin.min(k1);
    let n_mh_steps = resolve_n_mh_steps(
        options.saem_n_mh_steps,
        population.n_obs(),
        n_subjects,
        n_eta,
    );
    let n_cw_sweeps = componentwise_sweeps(n_mh_steps, n_eta);
    let adapt_interval = options.saem_adapt_interval;
    // `scale_adaptation`: the legacy ×1.1/×0.9 every `adapt_interval` (default),
    // or a Robbins-Monro step taken every iteration (#1444). Governs the primary
    // block and the componentwise scales only — κ stays on the interval rule
    // under both, see `FitOptions::saem_scale_adaptation` for why.
    let rm_scale_adaptation = matches!(
        options.saem_scale_adaptation,
        crate::types::ScaleAdaptation::RobbinsMonro
    );
    let verbose = options.verbose;
    let n_leapfrog = options.saem_n_leapfrog;
    // HMC is BSV-only (kappa-unaware); disable it for IOV models so eta sampling
    // uses the MH kernels that target the IOV conditional p(η | κ, θ, data).
    // Without this guard, an IOV model with an analytical PK path and
    // `n_leapfrog > 0` would propose eta against the kappa-free posterior and
    // hand a BSV-only NLL to the componentwise kernel as its (mismatched)
    // acceptance baseline.
    //
    // IIV on residual error (#409): the Dual2 NLL/gradient kernels build the
    // residual variance from σ alone and carry no `exp(2·η_ruv)` scaling rule,
    // so an HMC E-step would sample η against the unscaled conditional — η_ruv
    // sees no data curvature and collapses toward the prior. Disable HMC for
    // these models so the (correctly-scaled) MH kernels run instead.
    let using_hmc: bool = n_leapfrog > 0
        && model.ode_spec.is_none()
        && model.tv_fn.is_some()
        && n_kappa == 0
        && model.residual_error_eta.is_none()
        // Mixture models (#985): the E-step must sample the class indicator and
        // run η-MCMC within the drawn class (per-subject `MIXNUM` guard). The HMC
        // gradient kernel is class-unaware, so disable it and use the MH kernels,
        // which honour the class thread-local set in the E-step closure.
        && model.mixture.is_none();

    let n_theta = init_params.theta.len();
    let n_sigma = init_params.sigma.values.len();

    // Master RNG
    let master_seed = options.saem_seed.unwrap_or(12345);

    if verbose {
        eprintln!(
            "SAEM: {} subjects, {} ETAs, {} total iter ({} explore + {} converge)",
            n_subjects, n_eta, n_iter, k1, k2
        );
        // The block count is data-derived unless the caller pinned it, so say
        // which of the two a run is using — otherwise a reader cannot tell the
        // resolved count from a coincidence.
        eprintln!(
            "{}",
            mh_steps_report(options.saem_n_mh_steps, n_mh_steps, n_cw_sweeps)
        );
    }

    let mut warnings = Vec::new();
    if n_leapfrog > 0 && !using_hmc {
        // Keep the substring "HMC is unavailable" in both arms — `classify_warning`
        // keys on it to tag this as an Info/gradient_fallback warning.
        let reason = if n_kappa > 0 {
            "HMC is unavailable for IOV models (it is kappa-unaware)"
        } else if model.residual_error_eta.is_some() {
            "HMC is unavailable with IIV on residual error (iiv_on_ruv) — the Dual2 \
             gradient kernel has no exp(2·η_ruv) variance-scaling rule"
        } else {
            "HMC is unavailable (requires an analytical PK model the Dual2 gradient supports)"
        };
        warnings.push(format!(
            "saem_n_leapfrog > 0 but {reason}; falling back to Metropolis-Hastings"
        ));
    }
    let target_accept_rate = if using_hmc { 0.65_f64 } else { 0.40_f64 };

    // Initialize state
    let theta_cur = init_params.theta.clone();
    let omega_cur = init_params.omega.matrix.clone();
    let sigma_cur = init_params.sigma.values.clone();
    let s2 = omega_cur.clone();

    let etas: Vec<Vec<f64>> = (0..n_subjects)
        .map(|si| {
            let mut eta = get_eta_init(n_eta, None, None);
            // For FREM models, initialise covariate etas at their
            // conditional mode: eta_j = DV_cov - theta_k.  The posterior
            // for these etas is extremely peaked (EPSCOV ≈ 1e-6), so
            // starting at 0 leaves the chain far from the mode and
            // virtually every MH proposal gets rejected.
            if let Some(ref fc) = model.frem_config {
                let subj = &population.subjects[si];
                if !subj.fremtype.is_empty() {
                    for (&ft, &(theta_idx, eta_idx)) in &fc.fremtype_to_indices {
                        // Find the first observation with this FREMTYPE
                        if let Some(pos) = subj.fremtype.iter().position(|&f| f == ft) {
                            let dv = subj.observations[pos];
                            let tv = theta_cur[theta_idx];
                            eta[eta_idx] = dv - tv;
                        }
                    }
                }
            }
            eta
        })
        .collect();
    let step_scales = vec![0.3; n_subjects];
    // Componentwise kernel scales η'_j by √Ω_jj (a marginal SD), so a multiplier
    // near 1 is already a sensible 1-D step; start higher than the block kernel
    // and let adaptation climb toward the ~2.4 optimum.
    //
    // For FREM covariate etas the posterior is near-deterministic (EPSCOV
    // ≈ 1e-6) so the optimal CW step is orders of magnitude below the
    // prior SD.  Pre-compute: step_scale_j ≈ √(EPSCOV) / √(Ω_jj) so
    // that `step_scale_j · √Ω_jj ≈ √EPSCOV`.  This avoids thousands of
    // adaptation iterations to shrink from 1.0 down to ~1e-5.
    let cw_init = {
        let mut v = vec![1.0_f64; n_eta];
        if let Some(ref fc) = model.frem_config {
            let epscov = init_params.sigma.values[fc.covariate_sigma_index];
            for &(_theta_idx, eta_idx) in fc.fremtype_to_indices.values() {
                if eta_idx < n_eta {
                    let omega_jj = init_params.omega.matrix[(eta_idx, eta_idx)].max(1e-10);
                    // Target proposal SD = √EPSCOV; CW multiplies by √Ω_jj,
                    // so step_scale = √EPSCOV / √Ω_jj.  Floor at 1e-6.
                    v[eta_idx] = (epscov.sqrt() / omega_jj.sqrt()).max(1e-6);
                }
            }
        }
        v
    };
    let cw_step_scales = vec![cw_init; n_subjects];

    // Guard: the parser must guarantee omega_iov is present whenever kappas
    // are declared; if this fires, the caller wired up a broken ModelParameters.
    debug_assert!(
        n_kappa == 0 || init_params.omega_iov.is_some(),
        "n_kappa > 0 but init_params.omega_iov is None — model is misconfigured"
    );

    // Initialize IOV kappa state
    let (kappas_init, omega_iov_init, s2_iov_init): (
        Vec<Vec<Vec<f64>>>,
        DMatrix<f64>,
        DMatrix<f64>,
    ) = if n_kappa > 0 {
        let kaps: Vec<Vec<Vec<f64>>> = population
            .subjects
            .iter()
            .map(|s| {
                let n_occ = iov_occasion_groups(s).len();
                vec![vec![0.0f64; n_kappa]; n_occ]
            })
            .collect();
        let iov_mat = init_params
            .omega_iov
            .as_ref()
            .map(|iov| iov.matrix.clone())
            .unwrap_or_else(|| DMatrix::identity(n_kappa, n_kappa));
        (kaps, iov_mat.clone(), iov_mat)
    } else {
        (
            vec![vec![]; n_subjects],
            DMatrix::zeros(0, 0),
            DMatrix::zeros(0, 0),
        )
    };
    let kappa_step_scales = vec![0.3; n_subjects];

    // Initial NLL cache — use IOV-aware NLL when kappas are present
    let omega_iov_init_om = if n_kappa > 0 {
        init_params.omega_iov.clone()
    } else {
        None
    };
    let nll_cache: Vec<f64> = population
        .subjects
        .iter()
        .enumerate()
        .map(|(i, subject)| {
            if n_kappa > 0 {
                individual_nll_iov(
                    model,
                    subject,
                    &theta_cur,
                    &etas[i],
                    &kappas_init[i],
                    &init_params.omega,
                    omega_iov_init_om.as_ref(),
                    &sigma_cur,
                )
            } else {
                individual_nll(
                    model,
                    subject,
                    &theta_cur,
                    &etas[i],
                    &init_params.omega,
                    &sigma_cur,
                )
            }
        })
        .collect();

    // Per-theta packing flag: log for `theta_lower >= 0` (CL/V/KA…),
    // identity when `theta_lower < 0` (covariate exponents like
    // THETA_AGE_CL = -0.01 or THETA_CL_GAMMA = -0.8). Same convention
    // as `parameterization.rs::pack_params`. Without this, every theta
    // with a negative lower bound got clamped to 1e-10 by the old
    // `t.max(1e-10).ln()` packing and could never be estimated —
    // visible regression: SAD_SCEN4 SAEM left γ_CL stuck at 0 (truth
    // -0.8), letting the rest of the fit drift to compensate.
    let theta_packs_log_mask: Vec<bool> = init_params
        .theta_lower
        .iter()
        .map(|&lo| crate::estimation::parameterization::theta_packs_log(lo))
        .collect();
    let pack_theta = |i: usize, t: f64| -> f64 {
        if theta_packs_log_mask[i] {
            t.max(1e-10).ln()
        } else {
            t
        }
    };
    let unpack_theta = |i: usize, packed: f64| -> f64 {
        if theta_packs_log_mask[i] {
            packed.exp()
        } else {
            packed
        }
    };

    // Pack initial theta (per-mask) and sigma (always log).
    let mut log_theta: Vec<f64> = (0..n_theta).map(|i| pack_theta(i, theta_cur[i])).collect();
    let mut log_sigma: Vec<f64> = sigma_cur.iter().map(|&s| s.max(1e-10).ln()).collect();

    // Bounds in packed space — log when log-packed, identity otherwise.
    let mut log_theta_lower: Vec<f64> = (0..n_theta)
        .map(|i| {
            if theta_packs_log_mask[i] {
                init_params.theta_lower[i].max(1e-10).ln()
            } else {
                init_params.theta_lower[i]
            }
        })
        .collect();
    let mut log_theta_upper: Vec<f64> = (0..n_theta)
        .map(|i| {
            if theta_packs_log_mask[i] {
                init_params.theta_upper[i].min(1e9).ln()
            } else {
                init_params.theta_upper[i]
            }
        })
        .collect();
    let mut log_sigma_lower = vec![-8.0f64; n_sigma];
    let mut log_sigma_upper = vec![5.0f64; n_sigma];

    // Pin FIX parameters: set lower == upper == packed_value so the inner
    // NLopt M-step treats them as constants. Matches the FOCE/FOCEI treatment.
    for i in 0..n_theta {
        if init_params.theta_fixed.get(i).copied().unwrap_or(false) {
            log_theta_lower[i] = log_theta[i];
            log_theta_upper[i] = log_theta[i];
        }
    }
    for i in 0..n_sigma {
        if init_params.sigma_fixed.get(i).copied().unwrap_or(false) {
            log_sigma_lower[i] = log_sigma[i];
            log_sigma_upper[i] = log_sigma[i];
        }
    }

    // #895: iiv_on_ruv creates a σ × ω_RUV ridge (residual var = σ²·exp(2·η_RUV)),
    // so a poorly-mixing E-step can let the M-step ride σ up to its e⁵ ceiling.
    // Cap each *free* RUV-scaled residual σ's growth to a generous multiple of its
    // starting value (`SAEM_RUV_SIGMA_LN_GROWTH`) so a genuinely ill-posed run
    // stays bounded near sensible SDs. A FREM EPSCOV (always FIX, and independent
    // of the RUV scaling) is skipped, as is any FIXed σ.
    //
    // The cap is applied as a post-M-step clamp on `log_sigma`, NOT by tightening
    // the NLopt upper bound: a well-posed fit whose σ never approaches the cap
    // must reproduce the un-capped trajectory bit-for-bit (changing the bound
    // handed to NLopt perturbs its search path even when the optimum is interior).
    // `None` means "no cap for this σ"; a `Some(cap)` records the log-σ ceiling so
    // the update can be clamped and a run that ends pinned against it flagged.
    let mut ruv_sigma_caps: Vec<Option<f64>> = compute_ruv_sigma_caps(
        model.residual_error_eta.is_some(),
        model
            .frem_config
            .as_ref()
            .map(|fc| fc.covariate_sigma_index),
        &log_sigma,
        &log_sigma_upper,
        &init_params.sigma_fixed,
    );

    // #904: which residual σ absorb the iiv_on_ruv η-mean during re-centering —
    // exactly the free, non-FREM, RUV-scaled σ. Computed directly (not from
    // `ruv_sigma_caps`, which drops a free σ whose growth cap is looser than a
    // tight user upper bound — that σ must still absorb the shift). A FREM EPSCOV
    // or a FIXed σ is never scaled.
    let frem_cov_sigma_idx = model
        .frem_config
        .as_ref()
        .map(|fc| fc.covariate_sigma_index);
    let ruv_sigma_absorb: Vec<bool> = (0..n_sigma)
        .map(|i| {
            !init_params.sigma_fixed.get(i).copied().unwrap_or(false)
                && Some(i) != frem_cov_sigma_idx
        })
        .collect();

    // Re-centering scales η_RUV by −mean, which multiplies the *whole* residual
    // variance R = Σ σ_c² (all components, e.g. additive + proportional) by
    // exp(−2·mean). It preserves each subject's residual variance exactly only if
    // *every* non-FREM residual σ component absorbs the shift (each scaled by
    // exp(mean), so R → R·exp(2·mean)). If any RUV-scaled σ component is FIXed,
    // scaling only the free ones leaves the fixed part uncompensated and silently
    // perturbs R — so re-centering is disabled for that config and the σ/ω_RUV
    // growth caps carry the load instead (a FIXed component also partially anchors
    // the σ × η_RUV ridge). The FREM EPSCOV (always FIX, not on a real-obs row) is
    // exempt from this check.
    let ruv_recenter_ok = ruv_recenter_allowed(
        model.residual_error_eta.is_some(),
        model
            .frem_config
            .as_ref()
            .map(|fc| fc.covariate_sigma_index),
        &init_params.sigma_fixed,
    );

    // #895: log-variance ceiling for the iiv_on_ruv Ω diagonal — the ω_RUV half
    // of the σ × ω_RUV ridge backstop (see `compute_ruv_omega_cap`).
    let mut ruv_omega_cap: Option<f64> = compute_ruv_omega_cap(
        model.residual_error_eta,
        n_eta,
        &init_params.omega.matrix,
        &init_params.omega_fixed,
    );

    let mut state = SaemState {
        etas,
        kappas: kappas_init,
        nll_cache,
        step_scales,
        cw_step_scales,
        kappa_step_scales,
        accept_counts: vec![0; n_subjects],
        proposal_counts: vec![0; n_subjects],
        cw_accept_counts: vec![vec![0usize; n_eta]; n_subjects],
        cw_proposal_counts: vec![vec![0usize; n_eta]; n_subjects],
        kappa_accept_counts: vec![0; n_subjects],
        kappa_proposal_counts: vec![0; n_subjects],
        steps_since_adapt: 0,
        s2,
        s2_iov: s2_iov_init,
        residual_sse: None,
        theta: theta_cur,
        omega_mat: omega_cur,
        omega_iov_mat: omega_iov_init,
        sigma_vals: sigma_cur,
    };

    // Mu-referencing pairs for the closed-form M-step: (theta_idx, eta_idx).
    // Only pairs whose *packed* scale equals their mu scale are eligible
    // (`classify_mu_ref_pairs`), since the closed form steps the packed theta by
    // `γ · mean(η)` directly: log-packed for `THETA*exp(ETA)`, identity-packed
    // for `inv_logit(THETA + ETA)` (#918). The thetas it drops for a packing
    // mismatch are kept so the advisories below can name them.
    let mu_ref_split = classify_mu_ref_pairs(model, &init_params.theta_lower);
    // What a #619 covariate group may not collide with — every declared anchor
    // pair, including the ones the shared-anchor rule just dropped.
    let cov_group_conflicts = mu_ref_pairs_for_cov_groups(&mu_ref_split);
    let MuRefPairs {
        eligible: mu_ref_pairs,
        identity_packed_log: dropped_identity,
        log_packed_logit: dropped_log_packed_logit,
        shared_theta: dropped_shared_theta,
        shared_pairs: _,
    } = mu_ref_split;
    // Mixture: a MIXNUM-switched typical value pairs one η with several class
    // thetas, so the *pooled* `log_theta += γ·mean(η)` update above (one theta
    // per η) does not apply. It is well-posed per class though — SAEM draws a
    // hard class per subject, so `log θ_k += γ·mean_{i : c_i = k}(η_i)` — and
    // `get_mixture_mu_ref_pairs` supplies that class-resolved anchor set (#996).
    // Any class θ the parser could not resolve to a mu-ref pattern still routes
    // through the full NLopt θ/σ M-step, which — run under each subject's class
    // guard — estimates every class's thetas from its own members (#985).
    let mut saem_mix: Option<crate::estimation::saem_mixture::SaemMixture> =
        model.mixture.as_ref().map(|_| {
            crate::estimation::saem_mixture::SaemMixture::build(model, init_params, population)
        });
    if let Some(mix) = saem_mix.as_ref() {
        if mix.has_sigma_override() {
            warnings.push(
                "SAEM does not yet re-estimate per-class σ overrides (sigma(k)); they are held \
                 at their initial values. Class-switched θ, Ω overrides, and the mixing \
                 coefficients are estimated. Route to FOCEI if the σ overrides must be fit (#985)."
                    .to_string(),
            );
        }
        // Reject a theta that drives both the mixing expression and a structural
        // typical value: SAEM's separated M-step would estimate it from the
        // residual likelihood and then discard that for the mixing fit, silently
        // mis-fitting. FOCEI's joint marginal handles the shared parameter, so
        // route there instead of guessing (#987 review).
        let overlap = crate::estimation::saem_mixture::mixing_structural_overlap(
            model,
            init_params,
            population,
            &mix.mixing_theta_idx,
        );
        if !overlap.is_empty() {
            let names: Vec<String> = overlap
                .iter()
                .map(|&j| {
                    init_params
                        .theta_names
                        .get(j)
                        .cloned()
                        .unwrap_or_else(|| format!("theta[{j}]"))
                })
                .collect();
            return Err(format!(
                "SAEM cannot fit a mixture where a mixing-coefficient theta also drives the \
                 structural model: {} appear(s) in both the [mixture] mixing expression and an \
                 [individual_parameters] typical value. SAEM estimates the mixing coefficients \
                 and the structural thetas in separate M-steps, so a shared parameter would be \
                 double-owned. Split it into two thetas (one for structure, one for mixing), or \
                 fit with FOCE/FOCEI (whose joint marginal handles the shared parameter) (#985).",
                names.join(", ")
            ));
        }
    }
    // Per-class held σ overrides, hoisted out of the loop: they are constant
    // under SAEM (held at their inits) and the θ/σ M-step substitutes them per
    // subject so the free base σ is not dragged by override-class members.
    let mix_sigma_over: Vec<Vec<(usize, f64)>> = saem_mix
        .as_ref()
        .map(|m| m.class_sigma_overrides())
        .unwrap_or_default();

    // Class-aware mu-ref pairs for a mixture (#996). Empty for a non-mixture
    // model, which uses `mu_ref_pairs` above.
    //
    // SAEM takes only the **class-shared** anchors — an η whose typical value is
    // the same theta in every class (`V = TVV * exp(ETA_V)` inside a mixture).
    // For those the per-class update collapses to the classical pooled
    // `log θ += γ·mean(η)` (every class maps to one theta, so the weights sum to
    // N), so this is a strict extension of the single-population closed form:
    // before #996 a mixture disabled mu-referencing wholesale and even a
    // class-shared typical value went through the numerical M-step.
    //
    // A genuinely `MIXNUM`-switched typical value is deliberately *not* taken
    // here, even though `get_mixture_mu_ref_pairs` can express it. SAEM draws a
    // hard class per subject, and the per-class mean `mean_{i : c_i = k}(η_i)`
    // is then a classification-EM statistic: the class boundary is re-drawn from
    // the θ_k it just moved, and the two feed back. Measured on the two-class
    // anchor (`tests/nonmem/mixture_iv.csv`, seed 20250818) the class-aware
    // variant lands at `TVCL2 = 3.02` with OFV 305.0 against `2.75` / 302.2 for
    // the numerical M-step (NONMEM SAEM: 2.735; FOCEI optimum: 2.842) — worse on
    // every coordinate — and a Rao-Blackwellised soft-responsibility variant
    // converges to the same wrong point, so the bias is in the hard-class
    // sufficient statistic, not the weighting. IMP/IMPMAP, which importance-
    // samples within *every* class and weights by the responsibilities, does not
    // have this failure mode and does use the switched anchors.
    //
    // All of this is conditional on `mu_referencing`: with it off every θ goes
    // through the numerical M-step by construction, so telling the user to switch
    // estimator to get a closed-form shift they turned off is noise (#996 review).
    // Multi-theta (covariate) mu-references (#619). A group supersedes the
    // single-anchor pair on its own eta, and its thetas leave the packing
    // advisories (the group step moves them whatever their packing). Not run
    // under a mixture: the class draw would have to enter the group's prior
    // term and that has not been derived. Same `mu_referencing` gate as the
    // closed-form shift — with it off every theta is numerical by design.
    let (cov_mu_groups, cov_mu_notes) = if saem_mix.is_none() && options.mu_referencing {
        crate::estimation::covariate_mu_ref::resolve_covariate_mu_groups(
            model,
            population,
            &cov_group_conflicts,
            &init_params.theta_fixed,
            &init_params.omega.matrix,
        )
    } else {
        (Vec::new(), Vec::new())
    };
    for n in &cov_mu_notes {
        warnings.push(format!("SAEM: {n}"));
    }
    let cov_group_etas: Vec<usize> = cov_mu_groups.iter().map(|g| g.eta_idx).collect();
    let cov_group_thetas: Vec<usize> = cov_mu_groups
        .iter()
        .flat_map(|g| g.theta_idx.iter().copied())
        .collect();
    let mu_ref_pairs: Vec<(usize, usize)> = mu_ref_pairs
        .into_iter()
        .filter(|(_t, e)| !cov_group_etas.contains(e))
        .collect();
    let dropped_identity: Vec<usize> = dropped_identity
        .into_iter()
        .filter(|t| !cov_group_thetas.contains(t))
        .collect();
    let dropped_log_packed_logit: Vec<usize> = dropped_log_packed_logit
        .into_iter()
        .filter(|t| !cov_group_thetas.contains(t))
        .collect();

    let mut mix_switched_skipped: Vec<usize> = Vec::new();
    let mix_split = if saem_mix.is_some() && options.mu_referencing {
        classify_mixture_mu_ref_pairs(model, &theta_packs_log_mask)
    } else {
        MixtureMuRefPairs {
            eligible: Vec::new(),
            identity_packed_log: Vec::new(),
            log_packed_logit: Vec::new(),
        }
    };
    let mix_mu_ref_pairs: Vec<MixtureMuRefPair> = mix_split
        .eligible
        .into_iter()
        .filter(|p| {
            let shared = p.theta_idx.windows(2).all(|w| w[0] == w[1]);
            if !shared {
                mix_switched_skipped.extend(p.theta_idx.iter().copied());
            }
            shared
        })
        .collect();
    if !mix_switched_skipped.is_empty() {
        mix_switched_skipped.sort_unstable();
        mix_switched_skipped.dedup();
        let names: Vec<&str> = mix_switched_skipped
            .iter()
            .map(|&t| model.theta_names.get(t).map(String::as_str).unwrap_or("?"))
            .collect();
        warnings.push(format!(
            "SAEM: the MIXNUM-switched typical value(s) {} are estimated by the numerical \
             M-step, not the closed-form mu-referencing shift — SAEM's hard per-subject class \
             draw makes the per-class η mean a biased (classification-EM) statistic. Use \
             IMP/IMPMAP if the class-aware mu-ref shift is wanted (#996).",
            names.join(", ")
        ));
    }
    let mut dropped_identity = dropped_identity;
    let mut dropped_log_packed_logit = dropped_log_packed_logit;
    {
        // A class anchor whose packing does not match its mu scale joins whichever
        // advisory names the bound the user would have to change — identity-packed
        // lognormal (#996), or log-packed logit (#918).
        dropped_identity.extend(mix_split.identity_packed_log);
        dropped_log_packed_logit.extend(mix_split.log_packed_logit);
        for v in [&mut dropped_identity, &mut dropped_log_packed_logit] {
            v.sort_unstable();
            v.dedup();
        }
    }
    // Same gate: the identity-packing advisory is about a closed-form update that
    // does not run at all under `mu_referencing = false` (#996 review).
    if !dropped_identity.is_empty() && options.mu_referencing {
        let names: Vec<&str> = dropped_identity
            .iter()
            .map(|&t| model.theta_names.get(t).map(String::as_str).unwrap_or("?"))
            .collect();
        warnings.push(format!(
            "SAEM: typical value(s) {} are log-mu-referenced but declared with a negative \
             lower bound, so they are packed on the identity scale; the closed-form \
             `log θ += γ·mean(η)` update does not apply and they are estimated by the \
             numerical M-step instead. Give them a non-negative lower bound to use the \
             closed-form update (#996).",
            names.join(", ")
        ));
    }
    // Mirror image for a logit-scale theta (#918): `inv_logit(THETA + ETA)` has
    // mu scale θ, but a non-negative lower bound makes it log-packed.
    if !dropped_log_packed_logit.is_empty() && options.mu_referencing {
        let names: Vec<&str> = dropped_log_packed_logit
            .iter()
            .map(|&t| model.theta_names.get(t).map(String::as_str).unwrap_or("?"))
            .collect();
        warnings.push(format!(
            "SAEM: typical value(s) {} are logit-mu-referenced but declared with a \
             non-negative lower bound, so they are packed on the log scale; the closed-form \
             `θ += γ·mean(η)` update does not apply and they are estimated by the numerical \
             M-step instead. Declare the logit-scale theta with a negative lower bound to use \
             the closed-form update (#918).",
            names.join(", ")
        ));
    }
    // A theta anchoring two etas has no single closed-form shift (see
    // `classify_mu_ref_pairs`); it and both its pairs go to the numerical M-step.
    if !dropped_shared_theta.is_empty() && options.mu_referencing {
        let names: Vec<&str> = dropped_shared_theta
            .iter()
            .map(|&t| model.theta_names.get(t).map(String::as_str).unwrap_or("?"))
            .collect();
        warnings.push(format!(
            "SAEM: typical value(s) {} are the mu-reference anchor of more than one ETA, so the \
             closed-form mean shift has no single well-defined value for them; they are \
             estimated by the numerical M-step instead. Give each ETA its own typical value if \
             the closed-form update is wanted.",
            names.join(", ")
        ));
    }

    let use_closed_form_mstep = options.mu_referencing
        && if saem_mix.is_some() {
            // Mixture: the class-aware shift replaces the per-θ numerical
            // M-step for every log-mu-ref θ (#996).
            !mix_mu_ref_pairs.is_empty()
        } else {
            !mu_ref_pairs.is_empty() || !cov_mu_groups.is_empty()
        };

    // A scalar residual statistic is exact for this narrow subset of models.
    // Every excluded shape stays on the established joint numerical θ/σ M-step;
    // importantly, this is not a user-facing approximation switch.
    let scalar_residual_model = scalar_residual_mstep_model(
        model,
        population,
        init_params,
        n_kappa,
        saem_mix.is_some(),
        use_closed_form_mstep,
        &mu_ref_pairs,
    );

    // STRONG advisory: an estimated θ with **no associated ETA** is not
    // mu-referenced, so it never receives the γ-damped closed-form
    // `log θ += γ·mean(η)` shift. It is moved only by the η-frozen numerical
    // M-step, which re-maximises the conditional observation likelihood against
    // a *single* MCMC η draw with no stochastic-approximation damping — the
    // update is a random walk on a noisy surface, not an SA-averaged statistic,
    // and it can drift a long way from the marginal optimum.
    //
    // Measured on the FREM `iiv_on_ruv` reprex (475 subjects, 12 ETAs, the same
    // `FRD1` absorption fraction that motivated the IMP/IMPMAP advisory in
    // #406): SAEM drove `TVFRD1` 0.383 → 0.039 while IMP (0.311), IMPMAP
    // (0.318) and NONMEM IMP (0.394) agree, dragging `TVV` +6% and `TVMAT` +9%
    // with it. The drift is not a start artifact — restarting SAEM *at* the
    // IMPMAP solution still walked it down to 0.065 — and it is removed
    // entirely by attaching an ETA (`FRD1 = TVFRD1*exp(ETA_FRD1)`, ω² = 0.01 →
    // SAEM recovers 0.313) or by holding the parameter FIX (every other θ then
    // lands within 3% of NONMEM).
    let class_mu_ref_thetas: Vec<usize> = if use_closed_form_mstep && saem_mix.is_some() {
        mix_mu_ref_pairs
            .iter()
            .flat_map(|p| p.theta_idx.iter().copied())
            .collect()
    } else {
        Vec::new()
    };
    // Two kinds of θ are filtered out first, because "has NO associated ETA" is
    // factually wrong for them (#1012 review):
    //
    // * **Mixing-coefficient θ.** The numerical θ/σ M-step does not move them at
    //   all — they do not enter the residual/η likelihood, and `mstep_mixing`
    //   overwrites them from the responsibilities afterwards (step 4b below).
    //   They also *cannot* carry an η: the parser rejects a mixing expression
    //   that depends on one. Both the diagnosis and the remedy would be wrong.
    //   `run_mcem_mixture` drops the same set for the same reason.
    // * **Class typical values that already earned a dedicated message.** A
    //   MIXNUM-switched anchor, or a log-mu-ref θ dropped to the identity scale,
    //   *does* carry an η in the model text; it sits on the numerical M-step for
    //   a reason the two warnings above already state, and "add an ETA" on top
    //   of that is both false and contradictory. IMP/IMPMAP split these into a
    //   separate message (`shift_disabled` vs `thetas_without_eta`); SAEM
    //   already pushed the equivalent, so here it is a filter, not a partition.
    let mut already_explained: std::collections::HashSet<&str> = saem_mix
        .as_ref()
        .map(|m| m.mixing_theta_idx.as_slice())
        .unwrap_or(&[])
        .iter()
        .filter_map(|&j| model.theta_names.get(j).map(String::as_str))
        .collect();
    // `dropped_identity` only carries a message when `mu_referencing` is on; with
    // it off the θ is un-explained and stays in scope for this advisory.
    let identity_messaged: &[usize] = if options.mu_referencing {
        &dropped_identity
    } else {
        &[]
    };
    let logit_messaged: &[usize] = if options.mu_referencing {
        &dropped_log_packed_logit
    } else {
        &[]
    };
    already_explained.extend(
        mix_switched_skipped
            .iter()
            .chain(identity_messaged.iter())
            .chain(logit_messaged.iter())
            .filter_map(|&t| model.theta_names.get(t).map(String::as_str)),
    );
    let thetas_without_eta: Vec<String> = crate::estimation::impmap::non_fixed_thetas_without_eta(
        model,
        &init_params.theta_fixed,
        // Resolved groups, not parsed ones: a dropped group anchors nothing.
        &cov_group_thetas,
        &class_mu_ref_thetas,
    )
    .into_iter()
    .filter(|n| !already_explained.contains(n.as_str()))
    .collect();
    if !thetas_without_eta.is_empty() {
        warnings.push(format!(
            "SAEM: estimated parameter(s) [{}] have NO associated ETA, so they are not \
             mu-referenced and are moved only by the η-frozen numerical M-step, which \
             re-maximises the conditional likelihood against each iteration's MCMC η draw. \
             That is a valid (stochastic) EM update, but a noisier and slower one than the \
             closed-form mu-reference shift, and it can settle away from the marginal optimum \
             on a poorly mixing chain. For a typical value, put it in a mu-referenceable form \
             (`P = TVP * exp(ETA_P)` with a small, optionally FIX, omega — ferx applies \
             mu-referencing automatically). For a covariate coefficient, an allometric \
             exponent or a structural constant, cross-check the fit against FOCEI/IMPMAP, or \
             hold it FIX.",
            thetas_without_eta.join(", ")
        ));
    }

    // #1011: is the numerical M-step left estimating a fixed-effect-only θ this
    // fit? A θ that is FIXed, or pinned out by the closed-form mu-ref shift, is
    // returned unchanged by NLopt, so there is nothing for the SA damping below
    // to act on; a free θ that anchors a mu-reference has the exact closed-form
    // shift available to it, which is not the #1011 bias. Skipping the damping in
    // both cases keeps those fits byte-identical to the pre-#1011 behaviour (σ
    // included) — `thetas_without_eta` above lists exactly the θ that do trip it,
    // under the same mu-ref-detection predicate (see `theta_is_mu_ref_anchor_mask`
    // for what that does and does not see).
    //
    // **Mixtures are excluded.** A `MIXNUM`-switched typical value is estimated
    // by this same numerical M-step (#996 routes it there deliberately, because
    // SAEM's hard class draw makes the per-class η mean a biased statistic), but
    // it is solving a different problem: the class typical values must *separate*
    // from a common start, and the class assignments only stabilise once they
    // have. Damping that excursion stalls the separation — measured on
    // `tests/nonmem/mixture_iv_saem`, a 0.03 exploration cap leaves
    // `TVCL1 = 1.145` against NONMEM's 1.002, and drags the chained IMP marginal
    // with it. Whether the #1011 bias also affects mixture class θ (it plausibly
    // does) needs its own schedule and its own anchor, so it is left alone here
    // rather than half-fixed; the advisory above still fires for them.
    let numerically_estimated_theta = {
        let pinned: Vec<usize> = if !use_closed_form_mstep {
            Vec::new()
        } else if saem_mix.is_some() {
            class_mu_ref_thetas.clone()
        } else {
            mu_ref_pairs
                .iter()
                .map(|&(t, _e)| t)
                .chain(cov_group_thetas.iter().copied())
                .collect()
        };
        // `cov_group_thetas` is the *resolved* group list, so a group the
        // resolver dropped (weak IIV, a conflicting anchor, all-FIXed) leaves
        // its thetas counted as numerically estimated — which they are.
        let theta_is_mu_ref_anchor = crate::estimation::impmap::theta_is_mu_ref_anchor_mask(
            model,
            &cov_group_thetas,
            &class_mu_ref_thetas,
        );
        damps_numerical_mstep(
            saem_mix.is_some(),
            n_theta,
            &init_params.theta_fixed,
            &pinned,
            &theta_is_mu_ref_anchor,
        )
    };

    // #1011: exploration-phase cap on the numerical M-step's SA step. `None`
    // takes the calibrated default; `1.0` disables the damping entirely.
    let mstep_damping_cap = match options.saem_mstep_damping {
        None => default_mstep_damping(model.residual_error_eta.is_some()),
        Some(v) => match sanitize_mstep_damping(v) {
            None => v,
            Some(fixed) => {
                warnings.push(format!(
                    "SAEM: `mstep_damping` = {v} is outside (0, 1] — using {fixed} instead. A \
                     non-positive damping would step theta and sigma away from the M-step \
                     optimum on every iteration, and zero would freeze them (#1011)."
                ));
                fixed
            }
        },
    };
    // A setting the fit cannot act on is worse than no setting: say so rather
    // than accepting it silently.
    if options.saem_mstep_damping.is_some() && !numerically_estimated_theta {
        let reason = if saem_mix.is_some() {
            "this is a mixture model, whose class typical values must separate from a common \
             start before the class assignments settle — damping that excursion stalls it"
        } else if !options.mu_referencing {
            // The gate reads `theta_is_mu_ref_anchor_mask`, a *structural*
            // property of the model text, so it stays true under
            // `mu_referencing = false` even though no closed-form shift runs and
            // the numerical M-step really does move every theta. Skipping the
            // damping there is deliberate (see `damps_numerical_mstep`), but the
            // single-population wording below would be a lie about this fit.
            "`mu_referencing` is off, so every theta is estimated by the numerical M-step — but \
             each one is written in a mu-referenceable form, and #1011's damping was calibrated \
             only against thetas that have no such form at all. Damping this configuration is \
             untested, so it is left alone; drop `mu_referencing = false` to get the closed-form \
             shift instead"
        } else {
            "every estimated theta in this model is mu-referenced (so it has the exact \
             closed-form shift available, not the biased fixed-effect-only channel) or is FIX, \
             so the numerical M-step has no theta to damp"
        };
        warnings.push(format!(
            "SAEM: `mstep_damping` was set but has no effect — {reason} (#1011)."
        ));
    }

    // Accumulator for the `obs_nll_sum` (population OFV) evaluations skipped
    // by pinning mu-ref dims out of NLopt's central-FD gradient.  Each pinned
    // dim costs `2 * mstep_maxiter` `obs_nll_sum` calls inside NLopt — that's
    // the value we add per M-step that takes the closed-form branch.
    let mut mstep_grad_step_evals_saved: u64 = 0;

    // Per-subject flag: did this subject successfully use HMC at least once?
    // Only meaningful when `using_hmc = true`; stays all-false otherwise.
    let mut hmc_subjects = vec![false; n_subjects];

    // Run-cumulative combined (block + componentwise) MH accept / proposal counts
    // over the post-burn-in iterations, for the end-of-run "sampler not mixing"
    // warning (#895). Kept separate from `state.*_counts`, which the adaptation
    // step resets every `adapt_interval`.
    let mut cum_mh_acc: u64 = 0;
    let mut cum_mh_prop: u64 = 0;
    // Per-iteration samples for the last `MH_RATE_WINDOW` post-burn-in
    // iterations, backing the tail tier of the diagnostic (#1444).
    let mut mh_rate_window: VecDeque<MhRateSample> = VecDeque::with_capacity(MH_RATE_WINDOW);
    // 1-based iteration at which the block/componentwise step scales were
    // first adapted. The tail tier is gated on this: "never reached its
    // target" must not be said of a controller that never ran (#1451 review).
    let mut first_scale_adapt: Option<usize> = None;

    // Iterations on which each #619 group's solve returned `None` — its
    // typical value was not finite for some subject at the current θ. SAEM
    // leaves those thetas to the numerical M-step (it pins only after a
    // successful solve), so this is not the IMP freeze; it is still worth
    // reporting, because the estimate then did not come from the group the
    // user configured (#918 review).
    let mut cov_group_skipped = vec![0usize; cov_mu_groups.len()];

    // Per-subject `EventSchedule` cache, built once for the whole fit.
    //
    // A subject on the event-driven analytical path (time-varying covariates,
    // EVID-3/4 resets, or a `TIME`-reading `[individual_parameters]` program)
    // otherwise rebuilds its schedule inside *every* NLL evaluation: the
    // `schedule: None` arm of `compute_predictions_with_tv_recycle_with_schedule`
    // reaches `event_driven_predictions`, which calls `EventSchedule::for_subject`
    // — a merged event list over `2·n_doses + n_obs + n_pk_only + n_reset`, a
    // sort, and then a `Vec` of propagation bounds per interval, each an
    // O(n_doses) scan plus a sort and a dedup. That is ~39 rebuilds per subject
    // per iteration, 400 iterations deep, of something that does not depend on
    // η at all — and `event_driven_predictions_with_schedule`'s own docstring
    // already said where it belongs ("Hot loops should build the schedule once
    // per subject ... the merged event sort and per-interval infusion-bound
    // construction otherwise dominate per-call CPU on the TV-cov path"). FOCE's
    // inner loop and the Bayes chain both cache it; SAEM did not.
    //
    // [`build_schedule_cache`] is the shared builder those two use, so the
    // staleness rules (no η-dependent lagtime, no `F`-reshaped rate-defined
    // infusion) are stated in exactly one place — see `cacheable_schedule`. It
    // returns `None` per subject wherever reuse is unsound, which is then the
    // established rebuild-per-call behaviour.
    let schedules = saem_schedule_cache(model, population);

    // Main loop
    for k in 1..=n_iter {
        // Per-iteration combined (block + componentwise) accept / proposal
        // tallies — the honest E-step mixing rate reported to the trace (#895).
        let mut iter_acc: usize = 0;
        let mut iter_prop: usize = 0;
        // `Σ_k n_proposals_k · target_k` over the kernels that ran this
        // iteration, so the diagnostic can quote a proposal-weighted target
        // instead of the primary kernel's alone (#1451 review).
        let mut iter_target_weight: f64 = 0.0;
        if crate::cancel::is_cancelled(&options.cancel) {
            if verbose {
                eprintln!("SAEM: cancelled at iteration {}", k);
            }
            break;
        }
        let gamma = if k <= k1 { 1.0 } else { 1.0 / (k - k1) as f64 };
        // Damped SA step for the Ω sufficient statistic during exploration only.
        // With the full γ=1 used for θ, an undamped Ω would be overwritten each
        // exploration iteration by a single (warm-started, not-yet-equilibrated)
        // MCMC draw; for a correlated block that snapshot is biased toward the
        // chain's current correlation, and the bias feeds back through chol(Ω)
        // into the next proposal — a runaway toward a near rank-1 Ω. Capping the
        // Ω learning rate during exploration averages those draws (Robbins-Monro)
        // and breaks the feedback, while θ keeps moving at full γ. In the
        // convergence phase the cap is lifted: Ω uses the full decaying
        // γ = 1/(k−k1), the same schedule as θ, so the SA estimate settles
        // correctly (the chain is equilibrated by then, so the single-draw
        // overwrite risk that motivated the cap no longer applies).
        // #1011: SA step for the numerical θ/σ M-step result, mirroring
        // `gamma_omega`. Capped during exploration so a single un-equilibrated
        // MCMC draw cannot carry an un-mu-referenced θ away; full decaying γ in
        // the convergence phase. Left at 1.0 (undamped, pre-#1011) when NLopt has
        // no θ to estimate, so those fits are unchanged.
        //
        // Named for the M-step, but since #1445 it blends only the θ half of
        // the joint `[θ; σ]` maximiser `theta_sigma_mstep_light` returns; σ has
        // its own γ below, which is never larger than this one. See
        // `mstep_sa_step` and `sigma_mstep_sa_step`.
        let gamma_mstep = mstep_sa_step(
            numerically_estimated_theta,
            k <= k1,
            gamma,
            mstep_damping_cap,
        );
        // #1445: the σ half of that same joint maximiser takes its own,
        // always-on Robbins-Monro step. See `sigma_mstep_sa_step`.
        let gamma_sigma = sigma_mstep_sa_step(gamma, gamma_mstep);
        let gamma_omega = if k <= k1 {
            gamma.min(OMEGA_SA_MAX_STEP)
        } else {
            gamma
        };
        // Rebuild omega for this iteration
        let omega_k = OmegaMatrix::from_matrix(
            state.omega_mat.clone(),
            init_params.omega.eta_names.clone(),
            init_params.omega.diagonal,
        );

        // Rebuild omega_iov for this iteration.  Using from_matrix_with_mask
        // (not from_matrix) preserves the structural free_mask so that an
        // off-diagonal entry that converges to zero is not mistakenly treated
        // as a structural zero in the Cholesky proposal distribution.
        // Used in both the eta MH (Bug 2 fix) and the kappa MH (Step 1b).
        let omega_iov_cur_opt: Option<OmegaMatrix> = if n_kappa > 0 {
            init_params.omega_iov.as_ref().map(|iov_ref| {
                OmegaMatrix::from_matrix_with_mask(
                    state.omega_iov_mat.clone(),
                    iov_ref.eta_names.clone(),
                    iov_ref.diagonal,
                    iov_ref.free_mask.clone(),
                )
            })
        } else {
            None
        };

        // Mixture (#985): refresh every class's Ω/σ from the just-rebuilt base
        // plus the per-class overrides, then precompute the per-class proposal
        // Ω, σ, and componentwise SDs the E-step indexes by each subject's drawn
        // class. `class_omegas[0]` is the base (== omega_k) for a shared-Ω model.
        let (class_omegas, class_sigmas, class_cw_sd): (
            Vec<OmegaMatrix>,
            Vec<Vec<f64>>,
            Vec<Vec<f64>>,
        ) = if let Some(mix) = saem_mix.as_mut() {
            mix.sync_base(&omega_k, &state.sigma_vals);
            let n_c = mix.n_classes;
            let omg: Vec<OmegaMatrix> = (0..n_c).map(|c| mix.class_omega(c).clone()).collect();
            let sig: Vec<Vec<f64>> = (0..n_c).map(|c| mix.class_sigma(c).to_vec()).collect();
            let cw: Vec<Vec<f64>> = omg
                .iter()
                .map(|o| {
                    (0..n_eta)
                        .map(|j| o.matrix[(j, j)].max(SAEM_OMEGA_DIAG_FLOOR).sqrt())
                        .collect()
                })
                .collect();
            (omg, sig, cw)
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };

        // ---- Step 1: MH simulation (parallelized) ----
        // Symmetric random-walk MH in eta_true space, identical schedule
        // throughout exploration and convergence — the only thing that
        // changes between phases is the SA step size `gamma`.
        //
        // Two kernels run per subject per iteration (Kuhn & Lavielle 2004
        // mixture): (1) the primary block kernel — HMC when available, else a
        // `chol(Ω)`-preconditioned block RW; then (2) a componentwise sweep
        // (`mh_steps_componentwise`) that perturbs one η at a time. Kernel (2)
        // is what keeps a block Ω from collapsing to rank-1 — see that fn's
        // docstring.
        {
            use crate::parser::model_parser::MixtureClassGuard;
            use rayon::prelude::*;
            let theta_ref = &state.theta;
            let sigma_ref = &state.sigma_vals;
            let omega_ref = &omega_k;
            // Immutable view of the mixture state for the per-subject class draw
            // (the `sync_base` mutable borrow above has ended). The per-class
            // proposal Ω/σ/CW-SDs precomputed above are indexed by the draw.
            let mix_ref = saem_mix.as_ref();
            let class_omegas_ref = &class_omegas;
            let class_sigmas_ref = &class_sigmas;
            let class_cw_sd_ref = &class_cw_sd;
            // Per-coordinate componentwise proposal SDs — computed once here (Ω's
            // diagonal is shared across subjects) rather than per subject inside
            // the parallel kernel. Floored to match the Ω diagonal floor.
            let cw_sd: Vec<f64> = (0..n_eta)
                .map(|j| omega_k.matrix[(j, j)].max(SAEM_OMEGA_DIAG_FLOOR).sqrt())
                .collect();
            let cw_sd_ref = &cw_sd;
            // Per-coordinate multiplier for the block kernel (issue #895). 1.0 for
            // every ordinary ETA — so non-FREM models get the exact `chol(Ω)·z`
            // move, byte-for-byte (×1.0). FREM covariate ETAs are near-
            // deterministic (posterior SD ≈ √EPSCOV ≪ √Ω_jj), so a full-scale
            // joint proposal for them is rejected every time and pins the whole
            // block acceptance at 0%; damp their coordinate to ≈ √EPSCOV/√Ω_jj so
            // the joint move can still explore the correlated PK block. The
            // componentwise kernel handles the covariate ETAs themselves.
            let blk_eta_scale: Option<Vec<f64>> = model.frem_config.as_ref().map(|fc| {
                let epscov = state.sigma_vals[fc.covariate_sigma_index];
                let mut v = vec![1.0_f64; n_eta];
                for &(_theta_idx, eta_idx) in fc.fremtype_to_indices.values() {
                    if eta_idx < n_eta {
                        v[eta_idx] = (epscov.sqrt() / cw_sd[eta_idx]).clamp(1e-6, 1.0);
                    }
                }
                v
            });
            let blk_eta_scale_ref = blk_eta_scale.as_deref();
            // For IOV models, eta proposals must target p(η | κ, θ, data):
            // the per-occasion [eta_prop, kappa_k] predictions determine
            // which etas are accepted.  Pass omega_iov to mh_steps so it
            // can call individual_nll_iov with kappas held fixed.
            let omega_iov_for_eta_mh: Option<&OmegaMatrix> = omega_iov_cur_opt.as_ref();

            // Returns (eta_new, nll_after, n_acc_primary, n_prop_primary,
            //          per_eta_acc_cw, n_sweeps_cw, used_hmc, mix_class)
            #[allow(clippy::type_complexity)]
            let results: Vec<(
                Vec<f64>,
                f64,
                usize,
                usize,
                Vec<usize>,
                usize,
                bool,
                usize,
            )> = state
                .etas
                .par_iter()
                .zip(state.nll_cache.par_iter())
                .zip(state.step_scales.par_iter())
                .zip(state.cw_step_scales.par_iter())
                .zip(state.kappas.par_iter())
                .enumerate()
                // Per-rayon-worker `EventPkParams` scratch: allocated
                // once per worker per outer iteration, reused across
                // every subject the worker handles. Without `map_init`
                // the scratch was allocated per subject per outer
                // iter (5937 × N_iter on the cefepime SAEM bench);
                // with it, n_workers × N_iter ≈ 10 × N_iter.
                .map_init(
                    MhScratch::default,
                    |mh_scratch, (i, ((((eta, &nll), &scale), cw_sc_i), kappas_i))| {
                        let subject = &population.subjects[i];
                        let mut rng = StdRng::seed_from_u64(
                            master_seed
                                .wrapping_add(k as u64 * 100_000)
                                .wrapping_add(i as u64),
                        );
                        let kappas_mh_opt =
                            omega_iov_for_eta_mh.map(|iov| (kappas_i.as_slice(), iov));

                        // ---- Mixture E-step: draw the latent class ----
                        // Sample z_i ~ Categorical(PMIX_i) from the current
                        // posterior at this subject's η (and κ), then run the η
                        // moves *within* the drawn class: set the MIXNUM guard and
                        // point the proposal Ω/σ/CW-SDs at that class (#985).
                        // Draw the latent class; the per-subject posterior is
                        // recomputed at the converged params after the loop (via
                        // `mixture_ofv`), so only the sampled class is carried out.
                        let mix_class: usize = if let Some(mix) = mix_ref {
                            crate::estimation::saem_mixture::draw_class(
                                model,
                                subject,
                                theta_ref,
                                eta,
                                kappas_i,
                                mix,
                                omega_iov_for_eta_mh,
                                mh_scratch.pk(),
                                &mut rng,
                            )
                        } else {
                            0
                        };
                        let _class_guard = mix_ref.map(|_| MixtureClassGuard::enter(mix_class + 1));
                        // Point the worker's buffers at this subject and rebuild
                        // the η-independent half of the NLL's inputs (residual
                        // dispatch keys, `#484` magnitude multipliers) — once per
                        // subject, not once per proposal.
                        //
                        // **After the class guard, not before.** `IndividualNllPrep`
                        // hoists `model.ruv_obs_mult(..)` out of the proposal loop,
                        // and a residual-magnitude expression may legally read
                        // `MIXNUM` (`validate_ruv_expr` rejects η and NN outputs,
                        // not the class index), which resolves through the
                        // `MIXTURE_CLASS` thread-local this guard sets. The wrapper
                        // this replaced computed the multiplier *inside* the loop
                        // and therefore inside the guard; building it one line
                        // earlier evaluated it at the ambient class — class 1 — and
                        // served that to every class-2 subject's acceptance ratio.
                        // `draw_class` above only reads `mh_scratch.pk()`, which
                        // `begin_subject` does not touch, so the order is free to
                        // be the correct one.
                        mh_scratch.begin_subject(model, subject, theta_ref, n_eta);
                        let (omega_ref, sigma_ref, cw_sd_ref): (&OmegaMatrix, &[f64], &[f64]) =
                            if mix_ref.is_some() {
                                (
                                    &class_omegas_ref[mix_class],
                                    &class_sigmas_ref[mix_class],
                                    &class_cw_sd_ref[mix_class],
                                )
                            } else {
                                (omega_ref, sigma_ref.as_slice(), cw_sd_ref.as_slice())
                            };

                        let mut eta_work = eta.clone();

                        // ---- Kernel 1: primary block move ----
                        // Baseline for the first MH acceptance ratio is the cached
                        // NLL (as in the non-mixture path). For a mixture that value
                        // was computed under the previous iteration's drawn class, so
                        // on a class flip the very first proposal is scored against a
                        // slightly mismatched baseline — but the sweep recomputes the
                        // NLL under the current class from the next step on, so the
                        // effect is a single self-correcting proposal. Re-deriving the
                        // baseline at the drawn class each iteration was tried and
                        // measurably *worsened* the fit (it lets a wrong class draw
                        // drag η, inflating Monte-Carlo variance), so the cached
                        // baseline is deliberate (#987 review).
                        let mut nll_cur = nll;
                        let mut n_acc_primary = 0_usize;
                        let mut n_prop_primary = 0_usize;
                        // HMC path: one gradient-guided proposal per SAEM iteration.
                        // hmc_step returns None if HMC is unavailable for this subject
                        // (e.g. TV-cov subject with unsupported PK model); fall through
                        // to the block MH kernel. `did_hmc` doubles as the `used_hmc`
                        // flag reported back for diagnostics.
                        let did_hmc = if using_hmc {
                            if let Some((new_eta, new_nll, accepted, _divergent)) =
                                crate::estimation::hmc::hmc_step(
                                    subject, &eta_work, nll, model, theta_ref, omega_ref,
                                    sigma_ref, scale, n_leapfrog, &mut rng,
                                )
                            {
                                eta_work = new_eta;
                                nll_cur = new_nll;
                                n_acc_primary = accepted as usize;
                                n_prop_primary = 1;
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        };

                        if !did_hmc {
                            let (n_acc, nll_new) = mh_steps(
                                &mut eta_work,
                                nll_cur,
                                subject,
                                model,
                                theta_ref,
                                omega_ref,
                                sigma_ref,
                                scale,
                                blk_eta_scale_ref,
                                &mut rng,
                                n_mh_steps,
                                mh_scratch,
                                schedules[i].as_ref(),
                                kappas_mh_opt,
                            );
                            nll_cur = nll_new;
                            n_acc_primary = n_acc;
                            n_prop_primary = n_mh_steps;
                        }

                        // ---- Kernel 2: componentwise decorrelating sweep ----
                        let (per_eta_acc_cw, n_prop_cw, nll_cw) = mh_steps_componentwise(
                            &mut eta_work,
                            nll_cur,
                            subject,
                            model,
                            theta_ref,
                            omega_ref,
                            sigma_ref,
                            cw_sc_i,
                            cw_sd_ref,
                            &mut rng,
                            n_cw_sweeps,
                            mh_scratch,
                            schedules[i].as_ref(),
                            kappas_mh_opt,
                        );

                        (
                            eta_work,
                            nll_cw,
                            n_acc_primary,
                            n_prop_primary,
                            per_eta_acc_cw,
                            n_prop_cw,
                            did_hmc,
                            mix_class,
                        )
                    },
                )
                .collect();

            // Collect the drawn classes for the mixture M-step. The per-subject
            // posterior is recomputed at the converged parameters after the loop
            // (via `mixture_ofv`), matching what the FOCEI mixture path reports.
            let mut drawn_classes: Vec<usize> = vec![0; n_subjects];
            for (
                i,
                (eta_new, nll_new, n_acc, n_prop, per_eta_acc_cw, n_prop_cw, used_hmc, mix_class),
            ) in results.into_iter().enumerate()
            {
                drawn_classes[i] = mix_class;
                state.etas[i] = eta_new;
                state.nll_cache[i] = nll_new;
                state.accept_counts[i] += n_acc;
                state.proposal_counts[i] += n_prop;
                // Accumulate per-eta CW acceptance counts
                let cw_acc_i: usize = per_eta_acc_cw.iter().sum();
                for j in 0..n_eta {
                    state.cw_accept_counts[i][j] += per_eta_acc_cw[j];
                    state.cw_proposal_counts[i][j] += n_cw_sweeps;
                }
                hmc_subjects[i] |= used_hmc;
                // ---- Robbins-Monro step-scale adaptation (#1444, opt-in) ----
                // Stepped every iteration from *this* iteration's own rate, so
                // the correction is proportional to the discrepancy rather than
                // a fixed factor applied eight times a run. The counters above
                // are still maintained under this rule: the interval block
                // below resets them, and the trace reads them.
                if rm_scale_adaptation {
                    if n_prop > 0 || (n_cw_sweeps > 0 && !per_eta_acc_cw.is_empty()) {
                        first_scale_adapt.get_or_insert(k);
                    }
                    if n_prop > 0 {
                        let rate = n_acc as f64 / n_prop as f64;
                        state.step_scales[i] = rm_scale_update(
                            state.step_scales[i],
                            rate,
                            target_accept_rate,
                            k,
                            MH_BLOCK_SCALE_MIN,
                            MH_BLOCK_SCALE_MAX,
                        );
                    }
                    // Per-η, for the same reason the interval rule adapts each
                    // coordinate separately: a FREM covariate η and a PK η have
                    // posterior SDs orders of magnitude apart.
                    if n_cw_sweeps > 0 {
                        for j in 0..n_eta.min(per_eta_acc_cw.len()) {
                            let rate = per_eta_acc_cw[j] as f64 / n_cw_sweeps as f64;
                            state.cw_step_scales[i][j] = rm_scale_update(
                                state.cw_step_scales[i][j],
                                rate,
                                CW_TARGET_ACCEPT,
                                k,
                                MH_CW_SCALE_MIN,
                                MH_CW_SCALE_MAX,
                            );
                        }
                    }
                }
                // Combined (block + componentwise) acceptance for this iteration —
                // the honest E-step mixing metric (#895). The block kernel alone
                // reads 0% for FREM-scale Ω even when the componentwise sweep is
                // mixing fine, so a block-only rate is misleading.
                iter_acc += n_acc + cw_acc_i;
                iter_prop += n_prop + n_prop_cw;
                iter_target_weight +=
                    n_prop as f64 * target_accept_rate + n_prop_cw as f64 * CW_TARGET_ACCEPT;
            }

            // ---- Mixture bookkeeping (#985) ----
            // Record the drawn classes, SA-update the responsibility average and
            // the Ω-override statistics, and accumulate the posterior for the
            // final output (convergence phase only, once θ/Ω have settled).
            if let Some(mix) = saem_mix.as_mut() {
                mix.classes = drawn_classes;
                mix.update_rbar(gamma);
            }
        }

        // Fold this iteration's combined tallies into the post-burn-in totals
        // that back the end-of-run mixing warning (#895).
        if k > omega_burnin {
            cum_mh_acc += iter_acc as u64;
            cum_mh_prop += iter_prop as u64;
            // Keep the per-iteration pair too, so the diagnostic can look at the
            // *converged* tail rather than the whole post-burn-in average. A run
            // that mixes badly early and well later — or the reverse — averages
            // to something that describes neither (#1444).
            if mh_rate_window.len() == MH_RATE_WINDOW {
                mh_rate_window.pop_front();
            }
            mh_rate_window.push_back(MhRateSample {
                iter: k,
                accepted: iter_acc as u64,
                proposed: iter_prop as u64,
                target_weight: iter_target_weight,
            });
        }

        // ---- Step 1b: Per-occasion kappa MH (IOV models only) ----
        // For each subject, propose one new kappa per occasion and accept/reject
        // using the full IOV individual NLL (kappa prior + observation likelihood).
        // Parallel over subjects, and **bit-identical** to the serial loop it replaces
        // (#1344 item 5). Every access is `[i]`-disjoint — reads `etas[i]`, `kappas[i]`,
        // `kappa_step_scales[i]`; writes `kappas[i]`, `nll_cache[i]` and the two counters —
        // and each subject's RNG is seeded from `(master_seed, k, i)`, so no subject's draws
        // depend on the order subjects are visited in. That is what makes this a scheduling
        // change rather than a numerical one.
        //
        // Shaped as map-then-apply rather than a six-way `par_iter_mut().zip(..)`: the κ MH
        // both reads and writes `kappas[i]`, so the mutable shape needs the state
        // destructured into disjoint slices and threaded through nested tuples, which is
        // unreadable at six fields. The cost is one clone of `kappas[i]` (n_occ × n_kappa
        // f64s) per subject per iteration, against a full `individual_nll_iov` per proposal.
        //
        // The previous comment here said this loop was serial because the κ MH is "cheap
        // (low-dimensional, analytical PK) and share-free". Share-free is why this is safe.
        // Cheap is why the win is unmeasured: no one has profiled the phase's share of a
        // SAEM IOV fit, so treat this as removing a known serialisation, not as a measured
        // speedup.
        if n_kappa > 0 {
            if let Some(omega_iov_cur) = omega_iov_cur_opt.as_ref() {
                // Shared reborrow: the parallel phase only reads the state.
                use rayon::prelude::*;
                let st = &state;
                let updates: Vec<(Vec<Vec<f64>>, f64, usize, usize)> = (0..n_subjects)
                    .into_par_iter()
                    .map(|i| {
                        let mut kappas_i = st.kappas[i].clone();
                        let subject = &population.subjects[i];
                        // Mixture (#985): κ must be sampled inside the subject's drawn
                        // class — under that class's `MIXNUM` branch and its Ω/σ.
                        // Without the guard every subject's κ would be proposed against
                        // the class-1 typical values and class-1 Ω/σ, corrupting
                        // `st.kappas` (hence `s2_iov`, Ω_IOV and the θ/σ M-step)
                        // for every class-2+ subject (#987 review).
                        let cls = saem_mix.as_ref().map(|m| m.classes[i]);
                        let _class_guard = cls
                            .map(|c| crate::parser::model_parser::MixtureClassGuard::enter(c + 1));
                        let (omega_i, sigma_i): (&OmegaMatrix, &[f64]) = match cls {
                            Some(c) => (&class_omegas[c], class_sigmas[c].as_slice()),
                            None => (&omega_k, st.sigma_vals.as_slice()),
                        };
                        let mut rng = StdRng::seed_from_u64(
                            master_seed
                                .wrapping_add(k as u64 * 100_000)
                                .wrapping_add(i as u64)
                                .wrapping_add(999_999),
                        );
                        // Recompute NLL under the IOV-consistent function before
                        // proposing kappa.  After the eta MH block, nll_cache[i]
                        // may have been set by mh_steps via individual_nll_iov
                        // (with kappas fixed) — but to be safe we always recompute
                        // with the current kappas so detailed balance is guaranteed:
                        // both nll_kappa_ref and nll_prop are evaluated by the same
                        // individual_nll_iov, giving the correct acceptance ratio for
                        // p(κ | η, θ, data).
                        let mut kappa_pk = EventPkParams::default();
                        let sched_i = schedules[i].as_ref();
                        let nll_kappa_ref = individual_nll_iov_with_scratch_and_schedule(
                            model,
                            subject,
                            &st.theta,
                            &st.etas[i],
                            &kappas_i,
                            omega_i,
                            Some(omega_iov_cur),
                            sigma_i,
                            &mut kappa_pk,
                            sched_i,
                        );
                        let (n_acc, n_prop, nll_new) = mh_kappa_steps(
                            &mut kappas_i,
                            nll_kappa_ref,
                            subject,
                            model,
                            &st.theta,
                            &st.etas[i],
                            omega_i,
                            omega_iov_cur,
                            sigma_i,
                            st.kappa_step_scales[i],
                            &mut rng,
                            sched_i,
                            &mut kappa_pk,
                        );
                        (kappas_i, nll_new, n_acc, n_prop)
                    })
                    .collect();
                // Applied in subject order, so the counters accumulate identically to the
                // serial loop however the workers finished.
                for (i, (kappas_i, nll_new, n_acc, n_prop)) in updates.into_iter().enumerate() {
                    state.kappas[i] = kappas_i;
                    state.nll_cache[i] = nll_new;
                    state.kappa_accept_counts[i] += n_acc;
                    state.kappa_proposal_counts[i] += n_prop;
                }
            }
        }
        state.steps_since_adapt += 1;

        // ---- Step 2: SA update of sufficient statistic for Omega ----
        // #904: re-center the iiv_on_ruv η to zero mean, absorbing the shift into
        // the residual σ. `Y = f + EPS·exp(η_RUV)` has no typical-value θ, so —
        // unlike a mu-referenced structural η (CL/V…), whose mean is folded into
        // its TVP each iteration — η_RUV's mean is otherwise never absorbed. It
        // then drifts along the degenerate direction (`σ²·exp(2η)` is invariant to
        // η→η−c, σ→σ·exp(c)), and the drift pollutes ω_RUV = mean(η²) with a
        // spurious mean² term that pumps the σ × ω_RUV runaway. σ plays the role
        // of the residual-scale typical value here: shifting η_RUV by −mean and
        // scaling every RUV-scaled σ by exp(mean) leaves each subject's residual
        // variance exactly unchanged while restoring E[η_RUV] = 0, so the next Ω
        // M-step sees the true variance. Only done when every non-FREM residual σ
        // is free to absorb the shift (`ruv_recenter_ok`); a FIXed component would
        // leave R only partially rescaled and break the exact-invariance guarantee
        // (it also already partially pins the mean — no full degeneracy).
        if let (true, Some(kr)) = (ruv_recenter_ok, model.residual_error_eta) {
            recenter_ruv_eta(
                &mut state.etas,
                kr,
                &mut log_sigma,
                &mut state.sigma_vals,
                &ruv_sigma_absorb,
            );
        }

        // Mixture with `omega(k)` overrides (#987 review): the base Ω statistic
        // must be built from the subjects that actually *share* the base entry —
        // pooling an override class's members biases the base toward the
        // mixture-wide spread. Entries with no base members keep their previous
        // value. Without overrides this reduces to the pooled statistic below.
        let base_partitioned = saem_mix
            .as_ref()
            .filter(|m| !m.mp.omega_override_addr.is_empty())
            .map(|m| m.base_eta_outer(&state.etas, n_eta));

        if let Some((mean, counts)) = base_partitioned {
            for j in 0..n_eta {
                for l in 0..n_eta {
                    if counts[(j, l)] > 0 {
                        state.s2[(j, l)] =
                            (1.0 - gamma_omega) * state.s2[(j, l)] + gamma_omega * mean[(j, l)];
                    }
                }
            }
        } else {
            let mut eta_outer = DMatrix::zeros(n_eta, n_eta);
            for eta in &state.etas {
                let ev = DVector::from_column_slice(eta);
                eta_outer += &ev * ev.transpose();
            }
            eta_outer /= n_subjects as f64;

            state.s2 = (1.0 - gamma_omega) * &state.s2 + gamma_omega * &eta_outer;
        }

        // Per-class Ω-override statistic: SA-updated on the *same* schedule as
        // the base `s2` above — every iteration, including burn-in, so the first
        // post-burn-in override update reflects the warmed-up chain rather than a
        // single-iteration class mean. `omega_stat_active` is what gates whether
        // `sync_base` *applies* it, mirroring the base Ω's burn-in gate below.
        if let Some(mix) = saem_mix.as_mut() {
            mix.mstep_omega_overrides(&state.etas, gamma_omega);
            mix.omega_stat_active = k > omega_burnin;
        }

        // ---- Step 2b: SA update for Omega_iov (IOV only) ----
        // s2_iov = (1 - γ) s2_iov + γ · (1/N_occ) Σᵢ Σₖ κᵢₖ κᵢₖᵀ
        if n_kappa > 0 {
            let mut kappa_outer = DMatrix::zeros(n_kappa, n_kappa);
            let mut n_total_occ = 0_usize;
            for kappas_i in &state.kappas {
                for kap in kappas_i {
                    let kv = DVector::from_column_slice(kap);
                    kappa_outer += &kv * kv.transpose();
                    n_total_occ += 1;
                }
            }
            if n_total_occ > 0 {
                kappa_outer /= n_total_occ as f64;
            }
            state.s2_iov = (1.0 - gamma_omega) * &state.s2_iov + gamma_omega * &kappa_outer;
        }

        // ---- Step 3: M-step Omega (BSV + IOV) ----
        // Gated by the burn-in: while `k <= omega_burnin` Ω (and Ω_iov) are held
        // at their initial values so the MH chain can warm up before any
        // variance component is estimated. Step 2 still refreshes the SA
        // statistic `s2` each burn-in iteration (damped at `gamma_omega`, so it
        // is a running average of the warming chain rather than the latest
        // snapshot), so the first Ω update after burn-in reflects the warmed-up
        // chain, not the cold-start spread.
        if k > omega_burnin {
            // ---- Step 3a: Omega_bsv (closed form) ----
            // Restore FIX-ed rows / columns from the template. An eta flagged FIX
            // keeps its initial variance AND its initial off-diagonal couplings
            // (zero for a diagonal declaration, block cov for a FIX-ed block).
            // Letting the sufficient statistic bleed into row/col of a fixed eta
            // breaks positive-definiteness once the free-block diagonals shrink
            // during the exploration phase.
            state.omega_mat = state.s2.clone();
            // Zero structurally-absent off-diagonals. `s2 = (1/N) Σ ηη^T` always
            // produces a dense matrix; entries that aren't free parameters
            // (standalone etas, or etas from different `block_omega` declarations)
            // must be zeroed so they don't feed sampling correlations back into
            // the next iteration's Cholesky proposal. Without this the chain drives
            // Ω toward a rank-deficient state, log|Ω| → -∞, and the M-step pushes
            // thetas to bounds to compensate.
            for i in 0..n_eta {
                for j in 0..n_eta {
                    if !init_params.omega.free_mask[(i, j)] {
                        state.omega_mat[(i, j)] = 0.0;
                    }
                }
            }
            // Restore FIX-ed rows / columns from the template.
            for i in 0..n_eta {
                for j in 0..n_eta {
                    let fi = init_params.omega_fixed.get(i).copied().unwrap_or(false);
                    let fj = init_params.omega_fixed.get(j).copied().unwrap_or(false);
                    if fi || fj {
                        state.omega_mat[(i, j)] = init_params.omega.matrix[(i, j)];
                    }
                }
            }
            // Floor the free diagonal to keep Ω positive-definite, mirroring the
            // IOV Ω floor below. On sparse data (few obs/subject) a free η can
            // sample a near-zero spread early — once that feeds back into the
            // Cholesky MH proposal the scale collapses and the chain can never
            // re-inflate Ω, dumping between-subject variability into residual
            // error. FIX-ed entries were just restored from the template and are
            // left exactly as declared.
            floor_omega_diagonal(
                &mut state.omega_mat,
                &init_params.omega_fixed,
                SAEM_OMEGA_DIAG_FLOOR,
            );

            // #895: cap the iiv_on_ruv Ω variance so a runaway ridge can't inflate
            // ω_RUV without bound (the original report saw ~49). No-op for a
            // well-posed fit; a correlation-preserving rescale keeps Ω PD.
            if let (Some(k), Some(log_cap)) = (model.residual_error_eta, ruv_omega_cap) {
                apply_ruv_omega_cap(&mut state.omega_mat, k, log_cap, &init_params.omega_fixed);
            }

            // ---- Step 3b: Omega_iov (analytic, IOV only) ----
            // Apply the SA sufficient statistic, zeroing structural off-diagonals
            // and restoring FIX-ed kappa entries, mirroring the BSV omega treatment.
            if n_kappa > 0 {
                if let Some(omega_iov_ref) = init_params.omega_iov.as_ref() {
                    state.omega_iov_mat = state.s2_iov.clone();
                    // Zero structurally-absent off-diagonals.
                    for i in 0..n_kappa {
                        for j in 0..n_kappa {
                            if !omega_iov_ref.free_mask[(i, j)] {
                                state.omega_iov_mat[(i, j)] = 0.0;
                            }
                        }
                    }
                    // Restore FIX-ed kappa rows/columns from the template.
                    for i in 0..n_kappa {
                        for j in 0..n_kappa {
                            let fi = init_params.kappa_fixed.get(i).copied().unwrap_or(false);
                            let fj = init_params.kappa_fixed.get(j).copied().unwrap_or(false);
                            if fi || fj {
                                state.omega_iov_mat[(i, j)] = omega_iov_ref.matrix[(i, j)];
                            }
                        }
                    }
                    // Floor diagonal to stay positive-definite.
                    for i in 0..n_kappa {
                        if state.omega_iov_mat[(i, i)] < 1e-8 {
                            state.omega_iov_mat[(i, i)] = 1e-8;
                        }
                    }
                }
            }
        }

        // ---- Step 4: M-step theta, sigma (lightweight NLopt, warm-started) ----
        // Only run every few iterations during exploration to save time
        let run_mstep = k <= 5 || k % 3 == 0 || k > k1;
        let kappas_for_mstep = if n_kappa > 0 {
            Some(state.kappas.as_slice())
        } else {
            None
        };
        if run_mstep {
            // Use a few more iterations after exploration, when the M-step is
            // expected to settle rather than find its basin.
            let mstep_maxiter = if k <= k1 { 3 } else { 5 };
            if use_closed_form_mstep && saem_mix.is_none() {
                // Closed-form EM M-step for mu-referenced thetas.
                //
                // Model: g(P_i) = g(TVP) + η_i, η_i ~ N(0, ω²), where g is the
                // mu-scale link — `log` for `P = TVP·exp(η)`, `logit` for
                // `P = inv_logit(θ + η)`. Given the sampled individual
                // parameters, the data term does not involve TVP at all, so
                // the complete-data log-likelihood is maximised at
                //     g(TVP)_new = g(TVP)_old + mean_i(η_i)
                // and SAEM applies the stochastic-approximation step size γ:
                //     g(TVP)_new = g(TVP)_old + γ · mean_i(η_i)
                // `log_theta[j]` *is* g(TVP) for every pair in `mu_ref_pairs`
                // (that is what `classify_mu_ref_pairs` screens for), so the shift
                // below is applied to the packed value directly, whichever link
                // the parameter uses.
                // After the update, η_i is re-centred by `mean(η)` so the
                // sufficient statistic for ω is taken from zero-mean residuals
                // (ω is updated from `s2` *after* the next MH step, but
                // re-centring keeps `state.etas` consistent with the new TVP
                // for the rest of this iteration's NLL cache refresh).
                let n_subj = state.etas.len() as f64;
                let mut temp_theta_lower = log_theta_lower.clone();
                let mut temp_theta_upper = log_theta_upper.clone();
                let mut n_pinned: u64 = 0;
                for &(theta_idx, eta_idx) in &mu_ref_pairs {
                    if init_params
                        .theta_fixed
                        .get(theta_idx)
                        .copied()
                        .unwrap_or(false)
                    {
                        continue;
                    }
                    let mean_eta: f64 = state.etas.iter().map(|e| e[eta_idx]).sum::<f64>() / n_subj;
                    let log_theta_before = log_theta[theta_idx];
                    log_theta[theta_idx] = (log_theta_before + gamma * mean_eta)
                        .clamp(log_theta_lower[theta_idx], log_theta_upper[theta_idx]);
                    // Re-centre etas by the *actual* shift applied to log_theta,
                    // not by `gamma * mean_eta` directly: when the update is
                    // clamped at a bound the realised delta is smaller, and
                    // shifting etas by the unclamped quantity would break
                    // g(P_i) = g(TVP) + η_i until the next MH refresh.
                    let delta = log_theta[theta_idx] - log_theta_before;
                    for e in state.etas.iter_mut() {
                        e[eta_idx] -= delta;
                    }
                    // Pin so NLopt leaves the closed-form value unchanged.
                    temp_theta_lower[theta_idx] = log_theta[theta_idx];
                    temp_theta_upper[theta_idx] = log_theta[theta_idx];
                    n_pinned += 1;
                }
                // ---- Covariate mu-reference groups (#619): φ-frozen M-step ----
                // Exact (Gauss–Newton on the prior term) when the group's
                // covariates are time-constant, prior + data (BOBYQA) when one
                // varies within a subject. The result is blended with the same
                // SA step `gamma` as the shift above, then each subject's eta is
                // re-centred by the realised change in its own mu so that
                // `φ_i = g(A_i(θ)) + η_i` is unchanged — the per-subject twin of
                // the `e[eta_idx] -= delta` above.
                if !cov_mu_groups.is_empty() {
                    let unpack_all = |lt: &[f64]| -> Vec<f64> {
                        (0..n_theta)
                            .map(|i| {
                                if theta_packs_log_mask[i] {
                                    lt[i].exp()
                                } else {
                                    lt[i]
                                }
                            })
                            .collect()
                    };
                    let pack_one = |i: usize, v: f64| -> f64 {
                        if theta_packs_log_mask[i] {
                            v.max(1e-10).ln()
                        } else {
                            v
                        }
                    };
                    let sigma_now: Vec<f64> = log_sigma.iter().map(|s| s.exp()).collect();
                    for (gi, group) in cov_mu_groups.iter().enumerate() {
                        let theta_now = unpack_all(&log_theta);
                        let mu_old = group.mus(&theta_now, population);
                        let solved = {
                            let input = GroupStepInput {
                                theta: &theta_now,
                                theta_lower: &init_params.theta_lower,
                                theta_upper: &init_params.theta_upper,
                                theta_fixed: &init_params.theta_fixed,
                                theta_packs_log: &theta_packs_log_mask,
                                omega: &state.omega_mat,
                                etas: &state.etas,
                            };
                            if group.needs_data_term {
                                let k = group.eta_idx;
                                let etas_now = &state.etas;
                                let data = |th: &[f64], shift: &[f64]| -> f64 {
                                    let shifted: Vec<Vec<f64>> = etas_now
                                        .iter()
                                        .zip(shift.iter())
                                        .map(|(e, s)| {
                                            let mut e2 = e.clone();
                                            if k < e2.len() {
                                                e2[k] += s;
                                            }
                                            e2
                                        })
                                        .collect();
                                    match kappas_for_mstep {
                                        Some(kaps) => obs_nll_sum_iov(
                                            model, population, th, &sigma_now, &shifted, kaps,
                                            &schedules,
                                        ),
                                        None => obs_nll_sum(
                                            model, population, th, &sigma_now, &shifted, &schedules,
                                        ),
                                    }
                                };
                                group.solve_numerical(population, &input, mstep_maxiter, &data)
                            } else {
                                group.solve_exact(population, &input)
                            }
                        };
                        // Pinning happens below, inside this `Some` arm, so a
                        // skipped group leaves its thetas free for
                        // `theta_sigma_mstep_light` — no freeze, but the run is
                        // no longer the one the group describes.
                        let Some(theta_star) = solved else {
                            cov_group_skipped[gi] += 1;
                            continue;
                        };
                        for &t in &group.theta_idx {
                            if init_params.theta_fixed.get(t).copied().unwrap_or(false) {
                                continue;
                            }
                            let target = pack_one(t, theta_star[t]);
                            log_theta[t] = (log_theta[t] + gamma * (target - log_theta[t]))
                                .clamp(log_theta_lower[t], log_theta_upper[t]);
                            temp_theta_lower[t] = log_theta[t];
                            temp_theta_upper[t] = log_theta[t];
                            n_pinned += 1;
                        }
                        let theta_new = unpack_all(&log_theta);
                        let mu_new = group.mus(&theta_new, population);
                        for (i, e) in state.etas.iter_mut().enumerate() {
                            let d = mu_new[i] - mu_old[i];
                            if d.is_finite() && group.eta_idx < e.len() {
                                e[group.eta_idx] -= d;
                            }
                        }
                    }
                }
                // Each pinned mu-ref dim avoids 2 obs_nll_sum calls per NLopt
                // gradient request, capped at `mstep_maxiter` requests. FIXed
                // thetas are not pinned by the closed form (NLopt sees them as
                // FIXed via the regular bounds path) so they aren't counted.
                if scalar_residual_model.is_none() {
                    mstep_grad_step_evals_saved += 2 * mstep_maxiter as u64 * n_pinned;
                }

                // NLopt for any non-mu-ref theta and the general residual
                // channel. The scalar statistic gate has no free numerical θ
                // or σ dimension left for this solve.
                if scalar_residual_model.is_none() {
                    let (theta_new, sigma_new) = theta_sigma_mstep_light(
                        model,
                        population,
                        &state.etas,
                        kappas_for_mstep,
                        &log_theta,
                        &log_sigma,
                        &temp_theta_lower,
                        &temp_theta_upper,
                        &log_sigma_lower,
                        &log_sigma_upper,
                        n_theta,
                        n_sigma,
                        mstep_maxiter,
                        options.scale_params,
                        &theta_packs_log_mask,
                        // Closed-form branch is never taken for a mixture (disabled
                        // above), so no class guard is needed here.
                        None,
                        &schedules,
                    );
                    damp_mstep(&mut log_theta, &theta_new, gamma_mstep);
                    damp_mstep_sigma_variance(&mut log_sigma, &sigma_new, gamma_sigma);
                }
            } else if use_closed_form_mstep {
                // ---- Closed-form EM M-step under a mixture (#996) ----
                //
                // Runs the general class-resolved update
                //     log theta_k += gamma * mean_{i : c_i = k}(eta_i)
                // over `mix_mu_ref_pairs`, weighting each subject by a one-hot
                // indicator of the class SAEM drew for it this E-step. That pair
                // set is filtered to **class-shared** anchors upstream (see its
                // construction for the measurement that rules out the switched
                // case here), so in practice every class maps to one theta and
                // this reduces to the pooled `mean_i(eta_i)` of the
                // single-population branch above -- same accumulation order, same
                // result. The class-resolved form is kept because it is what makes
                // that reduction exact rather than a second copy of the formula.
                let mix = saem_mix.as_ref().expect("mixture arm without saem_mix");
                let resp: Vec<Vec<f64>> = mix
                    .classes
                    .iter()
                    .map(|&c| {
                        let mut r = vec![0.0f64; mix.n_classes];
                        if c < r.len() {
                            r[c] = 1.0;
                        }
                        r
                    })
                    .collect();
                let mut temp_theta_lower = log_theta_lower.clone();
                let mut temp_theta_upper = log_theta_upper.clone();
                let mut n_pinned: u64 = 0;
                for pair in &mix_mu_ref_pairs {
                    let eta_idx = pair.eta_idx;
                    let means = mixture_mu_ref_means(n_theta, &pair.theta_idx, &resp, |i, _c| {
                        state.etas[i][eta_idx]
                    });
                    // Realised per-theta shift, so the eta re-centering below uses
                    // the *clamped* delta (see the single-population branch).
                    let mut delta = vec![0.0f64; n_theta];
                    for (theta_idx, mean_eta) in means.iter().enumerate() {
                        // `None` means no subject was drawn into any class this
                        // theta serves (label switching, or a class that won nobody
                        // this iteration): hold theta_k, let the next E-step move it.
                        let Some(mean_eta) = mean_eta else { continue };
                        if init_params
                            .theta_fixed
                            .get(theta_idx)
                            .copied()
                            .unwrap_or(false)
                        {
                            continue;
                        }
                        let before = log_theta[theta_idx];
                        log_theta[theta_idx] = (before + gamma * mean_eta)
                            .clamp(log_theta_lower[theta_idx], log_theta_upper[theta_idx]);
                        delta[theta_idx] = log_theta[theta_idx] - before;
                        // Pin so NLopt leaves the closed-form value unchanged.
                        temp_theta_lower[theta_idx] = log_theta[theta_idx];
                        temp_theta_upper[theta_idx] = log_theta[theta_idx];
                        n_pinned += 1;
                    }
                    // Re-centre each subject by the shift applied to *its* class's
                    // theta, so `log(P_i) = log(TVP_{c_i}) + eta_i` still holds for
                    // the rest of this iteration's NLL cache refresh.
                    for (i, e) in state.etas.iter_mut().enumerate() {
                        let c = mix.classes[i];
                        if let Some(&t) = pair.theta_idx.get(c) {
                            e[eta_idx] -= delta[t];
                        }
                    }
                }
                mstep_grad_step_evals_saved += 2 * mstep_maxiter as u64 * n_pinned;

                // NLopt for the remaining (non-mu-ref) thetas and sigma, still
                // class-guarded so any class-switched theta outside the mu-ref set
                // is estimated from its own members (#985).
                let (theta_new, sigma_new) = theta_sigma_mstep_light(
                    model,
                    population,
                    &state.etas,
                    kappas_for_mstep,
                    &log_theta,
                    &log_sigma,
                    &temp_theta_lower,
                    &temp_theta_upper,
                    &log_sigma_lower,
                    &log_sigma_upper,
                    n_theta,
                    n_sigma,
                    mstep_maxiter,
                    options.scale_params,
                    &theta_packs_log_mask,
                    Some(MixMstep {
                        classes: mix.classes.as_slice(),
                        class_sigma_over: &mix_sigma_over,
                    }),
                    &schedules,
                );
                damp_mstep(&mut log_theta, &theta_new, gamma_mstep);
                damp_mstep_sigma_variance(&mut log_sigma, &sigma_new, gamma_sigma);
            } else {
                // mu_referencing = false (or a mixture whose class thetas could not
                // be class-aware mu-referenced): full NLopt M-step for all thetas
                // + sigma. For a mixture, `mstep_classes` guards each subject so
                // class-switched thetas are estimated per class (#985).
                if scalar_residual_model.is_none() {
                    let (theta_new, sigma_new) = theta_sigma_mstep_light(
                        model,
                        population,
                        &state.etas,
                        kappas_for_mstep,
                        &log_theta,
                        &log_sigma,
                        &log_theta_lower,
                        &log_theta_upper,
                        &log_sigma_lower,
                        &log_sigma_upper,
                        n_theta,
                        n_sigma,
                        mstep_maxiter,
                        options.scale_params,
                        &theta_packs_log_mask,
                        saem_mix.as_ref().map(|m| MixMstep {
                            classes: m.classes.as_slice(),
                            class_sigma_over: &mix_sigma_over,
                        }),
                        &schedules,
                    );
                    damp_mstep(&mut log_theta, &theta_new, gamma_mstep);
                    damp_mstep_sigma_variance(&mut log_sigma, &sigma_new, gamma_sigma);
                }
            }

            if let Some(residual_model) = scalar_residual_model {
                // The only θ updates this gate permits are FIXed or
                // log-mu-referenced shifts paired with eta re-centering, so the
                // current latent individual predictions remain the complete-data
                // samples to average. A runtime prediction failure keeps the
                // last valid σ rather than fabricating a statistic.
                let theta_for_stat: Vec<f64> = (0..n_theta)
                    .map(|i| unpack_theta(i, log_theta[i]))
                    .collect();
                if let Some((sample_sse, n_obs)) = scalar_residual_sse(
                    model,
                    population,
                    &theta_for_stat,
                    &state.etas,
                    residual_model,
                ) {
                    if n_obs > 0 {
                        update_scalar_residual_sse(&mut state.residual_sse, sample_sse, gamma);
                        if let Some(sse) = state.residual_sse {
                            let sigma = (sse / n_obs as f64).sqrt();
                            if sigma.is_finite() {
                                log_sigma[0] =
                                    sigma.ln().clamp(log_sigma_lower[0], log_sigma_upper[0]);
                            }
                        }
                    }
                }
            }

            // #895: clamp any RUV-scaled residual σ that the M-step pushed past its
            // growth cap. A no-op for a well-posed fit (σ stays well below the cap),
            // so the un-capped trajectory is reproduced bit-for-bit; only a runaway
            // riding the σ × ω_RUV ridge is pulled back.
            for (i, cap) in ruv_sigma_caps.iter().enumerate() {
                if let Some(cap) = cap {
                    if log_sigma[i] > *cap {
                        log_sigma[i] = *cap;
                    }
                }
            }

            state.theta = (0..n_theta)
                .map(|i| unpack_theta(i, log_theta[i]))
                .collect();
            state.sigma_vals = log_sigma.iter().map(|&v| v.exp()).collect();

            // ---- Step 4b: M-step for the mixing coefficients (#985) ----
            // The mixing thetas do not enter the residual/η likelihood, so the
            // θ/σ M-step above leaves them untouched. Update them from the SA
            // responsibility average `r̄_ik` (constant closed form or covariate
            // logistic fit), then re-sync `log_theta` for those indices.
            if let Some(mix) = saem_mix.as_ref() {
                crate::estimation::saem_mixture::mstep_mixing(
                    model,
                    population,
                    mix,
                    &mut state.theta,
                    &init_params.theta_lower,
                    &init_params.theta_upper,
                    mstep_maxiter,
                );
                for &j in &mix.mixing_theta_idx {
                    log_theta[j] = if theta_packs_log_mask[j] {
                        state.theta[j].max(1e-12).ln()
                    } else {
                        state.theta[j]
                    };
                }
            }
        }

        // ---- Update NLL cache (parallelized, needed for MH acceptance ratios) ----
        let omega_upd = OmegaMatrix::from_matrix(
            state.omega_mat.clone(),
            init_params.omega.eta_names.clone(),
            init_params.omega.diagonal,
        );
        // Mixture (#985): refresh each class's Ω/σ from the just-updated base and
        // evaluate every subject's cached NLL under its drawn class (guard + class
        // Ω/σ), so the next iteration's MH acceptance baseline is class-consistent.
        let mix_classes: Option<Vec<usize>> = saem_mix.as_ref().map(|m| m.classes.clone());
        let (mix_omegas, mix_sigmas): (Vec<OmegaMatrix>, Vec<Vec<f64>>) =
            if let Some(mix) = saem_mix.as_mut() {
                mix.sync_base(&omega_upd, &state.sigma_vals);
                (
                    (0..mix.n_classes)
                        .map(|c| mix.class_omega(c).clone())
                        .collect(),
                    (0..mix.n_classes)
                        .map(|c| mix.class_sigma(c).to_vec())
                        .collect(),
                )
            } else {
                (Vec::new(), Vec::new())
            };
        if n_kappa > 0 {
            // IOV NLL cache refresh — sequential rather than rayon-parallel.
            // individual_nll_iov is cheap (analytical PK, few occasions) and
            // the sequential loop avoids a second rayon scatter/gather.
            // Parallelise here if profiling shows a bottleneck.
            let omega_iov_upd = init_params.omega_iov.as_ref().map(|iov_ref| {
                OmegaMatrix::from_matrix_with_mask(
                    state.omega_iov_mat.clone(),
                    iov_ref.eta_names.clone(),
                    iov_ref.diagonal,
                    iov_ref.free_mask.clone(),
                )
            });
            let mut refresh_pk = EventPkParams::default();
            let new_nlls: Vec<f64> = (0..n_subjects)
                .map(|i| {
                    let cls = mix_classes.as_ref().map(|c| c[i]);
                    let _g =
                        cls.map(|c| crate::parser::model_parser::MixtureClassGuard::enter(c + 1));
                    let (omega_i, sigma_i): (&OmegaMatrix, &[f64]) = match cls {
                        Some(c) => (&mix_omegas[c], &mix_sigmas[c]),
                        None => (&omega_upd, state.sigma_vals.as_slice()),
                    };
                    individual_nll_iov_with_scratch_and_schedule(
                        model,
                        &population.subjects[i],
                        &state.theta,
                        &state.etas[i],
                        &state.kappas[i],
                        omega_i,
                        omega_iov_upd.as_ref(),
                        sigma_i,
                        &mut refresh_pk,
                        schedules[i].as_ref(),
                    )
                })
                .collect();
            state.nll_cache = new_nlls;
        } else {
            use rayon::prelude::*;
            // map_init lets each rayon worker keep one `EventPkParams`
            // scratch alive across every subject it handles, the same
            // pattern as the MH step above. Without it, the per-iter
            // refresh was allocating n_subj scratch buffers per outer
            // iter on TV-cov data.
            let mix_omegas_ref = &mix_omegas;
            let mix_sigmas_ref = &mix_sigmas;
            let mix_classes_ref = mix_classes.as_ref();
            let new_nlls: Vec<f64> = state
                .etas
                .par_iter()
                .enumerate()
                .map_init(EventPkParams::default, |scratch, (i, eta)| {
                    let cls = mix_classes_ref.map(|c| c[i]);
                    let _g =
                        cls.map(|c| crate::parser::model_parser::MixtureClassGuard::enter(c + 1));
                    let (omega_i, sigma_i): (&OmegaMatrix, &[f64]) = match cls {
                        Some(c) => (&mix_omegas_ref[c], &mix_sigmas_ref[c]),
                        None => (&omega_upd, state.sigma_vals.as_slice()),
                    };
                    crate::stats::likelihood::individual_nll_into_with_schedule(
                        model,
                        &population.subjects[i],
                        &state.theta,
                        eta,
                        omega_i,
                        sigma_i,
                        // Same ρ `individual_nll_into` passes — SAEM holds
                        // `block_sigma` at its declaration.
                        &model.residual_correlations,
                        scratch,
                        schedules.get(i).and_then(|s| s.as_ref()),
                    )
                })
                .collect();
            state.nll_cache = new_nlls;
        }

        // #903 review: re-anchor the iiv_on_ruv growth caps to the data-informed
        // σ/ω_RUV reached by the end of the exploration phase, taking the *looser*
        // of the init-based and settled-value-based ceilings. The caps are pure
        // runaway backstops (~20× a sensible scale); anchoring them to the raw
        // user *init* alone would spuriously clamp — and falsely warn about — a
        // well-posed fit started from a σ/ω guess many-fold below the truth. Only
        // ever loosens (`max`), so a genuine post-exploration runaway is still
        // bounded, and it preserves the bit-for-bit trajectory of a fit that never
        // approaches either ceiling.
        if k == k1 {
            reanchor_ruv_sigma_caps(&mut ruv_sigma_caps, &log_sigma, &log_sigma_upper);
            if let Some(kr) = model.residual_error_eta {
                ruv_omega_cap = reanchor_ruv_omega_cap(ruv_omega_cap, state.omega_mat[(kr, kr)]);
            }
        }

        // ---- Adapt MH step sizes ----
        if state.steps_since_adapt >= adapt_interval {
            for i in 0..n_subjects {
                // Use the actual per-subject proposal count as the denominator so
                // that MH-fallback subjects in HMC mode (which run n_mh_steps
                // proposals) are not scaled by the HMC denominator of 1.
                let total_proposals = state.proposal_counts[i].max(1);
                let rate = state.accept_counts[i] as f64 / total_proposals as f64;
                // Under the Robbins-Monro rule this scale is already stepped
                // every iteration; applying the multiplicative bump on top would
                // be a second, coarser controller fighting the first. The
                // counters are still *reset* here under either rule, because the
                // per-iteration trace reads them.
                if !rm_scale_adaptation {
                    first_scale_adapt.get_or_insert(k);
                    state.step_scales[i] = interval_scale_update(
                        state.step_scales[i],
                        rate,
                        target_accept_rate,
                        MH_BLOCK_SCALE_MIN,
                        MH_BLOCK_SCALE_MAX,
                    );
                }
                state.accept_counts[i] = 0;
                state.proposal_counts[i] = 0;
                // Adapt per-eta componentwise kernel scales toward the 1-D
                // optimum (~0.44 acceptance, Roberts & Rosenthal 2001).
                // Each eta adapts independently so that etas with very
                // different posterior precision (e.g. FREM covariate etas
                // with near-deterministic data vs broad PK etas) can each
                // reach their optimal step size.  The floor is 1e-6 (not
                // 0.01) to accommodate near-deterministic etas whose
                // posterior SD may be orders of magnitude below √Ω_jj.
                if n_cw_sweeps > 0 {
                    for j in 0..n_eta {
                        let cw_total = state.cw_proposal_counts[i][j].max(1);
                        let cw_rate = state.cw_accept_counts[i][j] as f64 / cw_total as f64;
                        // Same split as the block scale above.
                        if !rm_scale_adaptation {
                            state.cw_step_scales[i][j] = interval_scale_update(
                                state.cw_step_scales[i][j],
                                cw_rate,
                                CW_TARGET_ACCEPT,
                                MH_CW_SCALE_MIN,
                                MH_CW_SCALE_MAX,
                            );
                        }
                        state.cw_accept_counts[i][j] = 0;
                        state.cw_proposal_counts[i][j] = 0;
                    }
                }
                // Adapt kappa step sizes (target 40% for MH on kappas).
                if n_kappa > 0 {
                    let kappa_total = state.kappa_proposal_counts[i].max(1);
                    let kappa_rate = state.kappa_accept_counts[i] as f64 / kappa_total as f64;
                    if kappa_rate > 0.40 {
                        state.kappa_step_scales[i] = (state.kappa_step_scales[i] * 1.1).min(5.0);
                    } else {
                        state.kappa_step_scales[i] = (state.kappa_step_scales[i] * 0.9).max(0.01);
                    }
                    state.kappa_accept_counts[i] = 0;
                    state.kappa_proposal_counts[i] = 0;
                }
            }
            state.steps_since_adapt = 0;
        }

        // ---- Verbose output + optimizer trace ----
        {
            let phase = if k <= k1 { "explore" } else { "converge" };
            let cond_nll: f64 = state.nll_cache.iter().sum();
            // Combined (block + componentwise) acceptance for this iteration (#895).
            // Reporting the block kernel alone reads a misleading 0% for FREM-scale
            // Ω — the near-deterministic covariate ETAs reject every joint move —
            // even though the componentwise sweep is mixing the chain fine. The
            // combined rate is the honest E-step mixing diagnostic.
            let mh_accept_rate: f64 = iter_acc as f64 / iter_prop.max(1) as f64;

            if verbose && (k == 1 || k % 50 == 0 || k == n_iter) {
                eprintln!(
                    "  SAEM iter {:>4}/{} [{}] γ={:.3}  condNLL={:.3}",
                    k, n_iter, phase, gamma, cond_nll
                );
            }

            // Per-coordinate estimates (natural scale) for the trace's `val:*`
            // columns (#640). SAEM has no OFV gradient, so `grad:*` are NA.
            // Guard the per-iteration Vec build on `is_active` so it costs
            // nothing on the default (trace-off) path across the many SAEM
            // iterations (#640 review).
            if crate::estimation::trace::is_active() {
                let iov = init_params
                    .omega_iov
                    .as_ref()
                    .map(|m| (&state.omega_iov_mat, m.diagonal));
                let values = crate::estimation::parameterization::coordinate_values_raw(
                    &state.theta,
                    &state.omega_mat,
                    init_params.omega.diagonal,
                    &state.sigma_vals,
                    iov,
                );
                crate::estimation::trace::write_saem(
                    k,
                    phase,
                    cond_nll,
                    gamma,
                    mh_accept_rate,
                    &values,
                );
            }

            // Checkpoint (#755): snapshot the current SAEM estimates periodically
            // so an interrupted run resumes near here. `cond_nll` stands in for
            // the OFV (the FOCE OFV is not yet available mid-SAEM).
            if crate::io::checkpoint::is_due() {
                let snap = saem_state_to_params(&state, init_params, n_kappa);
                let packed = crate::estimation::parameterization::pack_params(&snap);
                crate::io::checkpoint::maybe_write(k, cond_nll, &packed);
            }
        }
    }

    // If the user cancelled mid-run the loop broke early; skip the final
    // EBE/OFV computation (which iterates over every subject) and abort.
    if crate::cancel::is_cancelled(&options.cancel) {
        return Err("cancelled by user".to_string());
    }

    if verbose {
        eprintln!("SAEM iterations complete. Computing final EBEs and OFV...");
    }

    // ---- Post-SAEM: build final parameters ----
    let mut final_params = saem_state_to_params(&state, init_params, n_kappa);
    // Mixture (#985): carry the estimated per-class Ω/σ overrides so the final
    // OFV, EBEs, and covariance step see the mixture structure. `mp`'s base was
    // synced from the final Ω/σ in the last NLL-cache refresh.
    if let Some(mix) = saem_mix.as_ref() {
        let mut mp = mix.mp.clone();
        // SAEM holds the per-class σ overrides at their inits (warned above), so
        // they are FIX as far as this fit is concerned. Marking them keeps the
        // covariance step from building an FD-Hessian row for a coordinate SAEM
        // never optimised — evaluated off its stationary point that row can push
        // an otherwise-PD Hessian into the eigen-floor/SIR fallback and degrade
        // every other SE (#987 review). Their SEs report as 0, matching FIX.
        mp.sigma_override_fixed = vec![true; mp.sigma_override_addr.len()];
        final_params.mixture = Some(mp);
    }

    if combined_additive_sigma_at_floor(model, &final_params) {
        warnings
            .push("SAEM combined-error additive sigma collapsed to its lower bound.".to_string());
    }

    // #895: warn when the E-step never mixed. A combined (block + componentwise)
    // post-burn-in acceptance rate near zero means the sampled ETAs barely moved
    // from their starting values, so the M-step ran on degenerate sufficient
    // statistics and the Ω/σ estimates are unreliable (the classic FREM-scale
    // "0% acceptance" failure).
    let mh_window = MhRateWindow {
        samples: mh_rate_window.make_contiguous(),
        first_adapt: first_scale_adapt,
    };
    // The number the diagnostic below reports on, exposed on `FitResult` so a
    // caller (and a test) can read it without parsing the message or the
    // optimizer trace (#1444).
    let saem_mh_accept_tail = mh_window.tail_rate();
    if let Some(w) = saem_mixing_warning(
        cum_mh_acc,
        cum_mh_prop,
        &mh_window,
        options.saem_scale_adaptation,
    ) {
        warnings.push(w);
    }

    // #918 review: a #619 group whose solve kept returning `None` never ran.
    for (gi, &skipped) in cov_group_skipped.iter().enumerate() {
        if skipped == 0 {
            continue;
        }
        let group = &cov_mu_groups[gi];
        let names: Vec<&str> = group
            .theta_idx
            .iter()
            .filter_map(|&t| model.theta_names.get(t).map(String::as_str))
            .collect();
        warnings.push(format!(
            "SAEM: the covariate mu-reference on {} had no admissible step on {} of {} \
             iteration(s) — its typical value was not finite for at least one subject at the \
             current θ (an additive typical value can go ≤ 0 for a low-covariate subject). {} \
             fell back to the numerical M-step on those iterations, which is the channel #619 \
             exists to avoid; consider bounding the covariate slope.",
            model
                .eta_names
                .get(group.eta_idx)
                .map(String::as_str)
                .unwrap_or("?"),
            skipped,
            n_iter,
            names.join(", ")
        ));
    }

    // #895: warn when a free RUV-scaled residual σ ended pinned against the
    // iiv_on_ruv growth cap. That signals the σ × ω_RUV ridge is poorly
    // identified from the data alone; the cap kept σ bounded but the split
    // between σ and ω_RUV should not be trusted — fix one of them (e.g. σ at a
    // known value) or drop the IIV on the residual error.
    for (i, cap) in ruv_sigma_caps.iter().enumerate() {
        if let Some(cap) = cap {
            if final_params.sigma.values[i].max(1e-300).ln() >= cap - 1e-6 {
                warnings.push(format!(
                    "SAEM residual sigma '{}' hit its iiv_on_ruv growth cap (≈{:.4}); the \
                     sigma × omega_RUV ridge is weakly identified and the estimates are not \
                     trustworthy. The most reliable fix is to FIX sigma at a known value (e.g. \
                     from an IMP/IMPMAP run) and re-fit — omega_RUV then recovers; alternatively \
                     FIX the RUV omega or remove iiv_on_ruv.",
                    final_params
                        .sigma
                        .names
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| format!("sigma[{i}]")),
                    cap.exp()
                ));
            }
        }
    }

    // #895: warn when the iiv_on_ruv Ω variance ended pinned against its growth
    // cap — the ω_RUV half of the same ridge instability. Same guidance: fix σ (or
    // the RUV Ω) and re-fit.
    if let (Some(k), Some(log_cap)) = (model.residual_error_eta, ruv_omega_cap) {
        if final_params.omega.matrix[(k, k)].max(1e-300).ln() >= log_cap - 1e-6 {
            warnings.push(format!(
                "SAEM iiv_on_ruv omega '{}' hit its growth cap (≈{:.4}); the sigma × omega_RUV \
                 ridge is weakly identified and the estimates are not trustworthy. FIX sigma at \
                 a known value (e.g. from an IMP/IMPMAP run) and re-fit — omega_RUV then recovers.",
                final_params
                    .omega
                    .eta_names
                    .get(k)
                    .cloned()
                    .unwrap_or_else(|| format!("eta[{k}]")),
                log_cap.exp()
            ));
        }
    }

    // ---- Final EBEs + OFV ----
    // For a mixture (#985), the population objective is the K-fold marginal
    // `−2 Σ_i log Σ_k p_ik e^{−nll_ik}`, and the reported EBEs / κ̂ are the
    // MIXEST class's — both produced by `mixture_ofv` (the same routine the
    // FOCEI mixture path uses), so SAEM and FOCEI report comparable OFVs and
    // per-subject posteriors. Otherwise the single-population FOCE approximation.
    let (eta_hats, h_matrices, final_kappas, ofv, mixture_posteriors) =
        if final_params.mixture.is_some() {
            let meval = crate::estimation::mixture::mixture_ofv(
                model,
                population,
                &final_params,
                options,
                None,
            );
            let posteriors = crate::estimation::outer_optimizer::MixturePosteriors {
                pmix: meval.pmix.clone(),
                mixest: meval.mixest.clone(),
            };
            (
                meval.mixest_etas,
                meval.mixest_h_mats,
                meval.mixest_kappas,
                meval.ofv,
                Some(posteriors),
            )
        } else {
            let warm_etas: Vec<DVector<f64>> = state
                .etas
                .iter()
                .map(|e| DVector::from_column_slice(e))
                .collect();
            let saem_final_mu_k = compute_mu_k(model, &final_params.theta, options.mu_referencing);
            let (eta_hats, h_matrices, _, final_kappas) = run_inner_loop_warm(
                model,
                population,
                &final_params,
                options.inner_maxiter,
                options.inner_tol,
                Some(&warm_etas),
                Some(&saem_final_mu_k),
                0, // SAEM: no EBE convergence tracking
                0, // SAEM final EBE is warm-started; no inner multi-start
            );
            let ofv = 2.0
                * pop_nll(
                    model,
                    population,
                    &final_params,
                    &eta_hats,
                    &h_matrices,
                    &final_kappas,
                    options.interaction,
                );
            (eta_hats, h_matrices, final_kappas, ofv, None)
        };

    // ---- Report OFV *before* the covariance step (#893) ----
    // SAEM only learns its OFV here (the final FOCE approximation); the covariance
    // step that follows can be the most expensive part of the run. Print the OFV
    // first so a CLI user can judge the fit and interrupt (Ctrl-C) before paying
    // for the covariance matrix when the OFV already rules the run out.
    if verbose {
        eprintln!("{}", saem_final_ofv_report(ofv));
    }

    // ---- Covariance step ----
    let packed = pack_params(&final_params);
    let out = crate::estimation::covariance::run_covariance_step(
        &packed,
        &final_params,
        model,
        population,
        &eta_hats,
        &h_matrices,
        &final_kappas,
        options,
        verbose.then_some("Running covariance step..."),
    );
    let crate::estimation::covariance::CovStepOutcome {
        matrix: covariance_matrix,
        wall_time_secs: covariance_wall_time_secs,
        warnings: cov_warnings,
        sir_fallback_proposal,
        method: covariance_method,
    } = out;
    warnings.extend(cov_warnings);

    let saem_mu_ref_m_step_evals_saved = if use_closed_form_mstep {
        Some(mstep_grad_step_evals_saved)
    } else {
        None
    };

    let saem_n_subjects_hmc = if using_hmc {
        Some(hmc_subjects.iter().filter(|&&b| b).count())
    } else {
        None
    };

    // ---- Post-fit conditional-distribution pass (opt-in, #257) ----
    // Characterise each subject's p(η_i | y_i; θ̂) by MCMC at the fixed
    // population parameters, warm-started from the EBE mode (`eta_hats`).
    // The conditional-distribution pass samples p(η_i | y_i) at fixed θ̂ with a
    // single-population target; it is not class-aware, so skip it for a mixture
    // (#985) rather than characterise the posterior under the wrong class.
    if options.saem_conddist && final_params.mixture.is_some() {
        warnings.push(
            "SAEM conditional-distribution pass (conddist) is not yet mixture-aware; skipping \
             it for this mixture fit (#985)."
                .to_string(),
        );
    }
    let cond_dist = if options.saem_conddist && final_params.mixture.is_none() {
        if verbose {
            eprintln!(
                "Running SAEM conditional-distribution pass ({} samples/subject, {} burn-in)...",
                options.saem_conddist_nsamp, options.saem_conddist_burnin
            );
        }
        Some(
            crate::estimation::saem_conddist::run_conditional_distribution(
                model,
                population,
                &final_params,
                &eta_hats,
                &final_kappas,
                options,
            ),
        )
    } else {
        None
    };

    // A finite-but-enormous OFV is the bounded blowup of a runaway, not a
    // converged fit — guard against it the same way IMP/IMPMAP does, since
    // SAEM is commonly the first phase of a SAEM→IMP chain (issue #528) — and a
    // `NaN` one is not a converged fit either (#1303). One shared gate, and it
    // reports which rule it applied.
    let mut converged = true;
    if let Some(w) =
        crate::estimation::impmap::gate_converged_on_mcem_objective(&mut converged, ofv)
    {
        warnings.push(w);
    }

    Ok(OuterResult {
        params: final_params,
        ofv,
        converged,
        n_iterations: n_iter,
        eta_hats,
        h_matrices,
        kappas: final_kappas,
        covariance_matrix,
        covariance_method,
        covariance_wall_time_secs,
        warnings,
        saem_mu_ref_m_step_evals_saved,
        saem_n_subjects_hmc,
        saem_mh_accept_tail,
        ebe_convergence_warnings: 0,
        max_unconverged_subjects: 0,
        total_ebe_fallbacks: 0,
        final_gradient: None,
        final_gradient_source: None,
        sir_fallback_proposal,
        impmap_trace: None,
        bayes: None,
        cond_dist,
        packed_estimate: None,
        left_init: None,
        mixture_posteriors,
        vi: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── #996 end-to-end helpers: a tiny 2-class mixture dataset ──────────
    //
    // Two weight groups with well-separated clearances, so the class draw is
    // stable and a handful of iterations is enough to exercise the M-step.

    /// Small 1-cpt IV dataset, `n_per` subjects at each of two weights.
    fn mix996_csv(n_per: usize) -> String {
        let mut s = String::from("ID,TIME,DV,AMT,EVID,CMT,WT\n");
        let mut sid = 0;
        for (g, &wt) in [60.0_f64, 90.0].iter().enumerate() {
            let cl = if g == 0 { 1.0 } else { 3.0 };
            for _ in 0..n_per {
                sid += 1;
                s.push_str(&format!("{sid},0,0,100,1,1,{wt}\n"));
                for (ti, t) in [0.5_f64, 1.0, 2.0, 4.0].iter().enumerate() {
                    let c = (100.0 / 10.0) * (-(cl / 10.0) * t).exp();
                    let dv = c * (1.0 + 0.02 * ((sid + ti) as f64).sin());
                    s.push_str(&format!("{sid},{t},{dv:.5},0,0,1,{wt}\n"));
                }
            }
        }
        s
    }

    fn mix996_pop(n_per: usize) -> Population {
        use std::io::Write;
        let csv = mix996_csv(n_per);
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(csv.as_bytes()).unwrap();
        crate::io::datareader::read_nonmem_csv(f.path(), Some(&["WT"]), None).unwrap()
    }

    /// Two-class mixture. `v_expr` picks whether V carries a class-shared
    /// mu-referenced ETA (`TVV * exp(ETA_V)`) or none (`TVV`).
    fn mix996_model(v_expr: &str) -> CompiledModel {
        let src = format!(
            r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(3.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09 FIX
  omega ETA_V ~ 0.04 FIX
  sigma EPS ~ 0.04 FIX

[mixture]
  nsub = 2
  logit(1) = MIXL

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = {v_expr}

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
"
        );
        crate::parser::model_parser::parse_model_string(&src).expect("mixture model parses")
    }

    /// Single-population 1-cpt IV model whose `V` either carries a
    /// mu-referenceable ETA or is a bare fixed-effect-only θ — the shape that
    /// makes SAEM's η-frozen numerical M-step the *only* channel that can move
    /// it (the FREM `FRD1` absorption fraction of #406).
    fn noeta_model(v_expr: &str) -> CompiledModel {
        let src = format!(
            r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.09 FIX
  omega ETA_V ~ 0.04 FIX
  sigma EPS ~ 0.04 FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = {v_expr}

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
"
        );
        crate::parser::model_parser::parse_model_string(&src).expect("model parses")
    }

    fn saem_noeta_warnings(v_expr: &str) -> Vec<String> {
        let model = noeta_model(v_expr);
        let pop = mix996_pop(3);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Saem;
        opts.saem_n_exploration = 4;
        opts.saem_n_convergence = 2;
        opts.saem_seed = Some(406);
        opts.run_covariance_step = false;
        crate::api::fit(&model, &pop, &model.default_params, &opts)
            .expect("SAEM Ok")
            .warnings
    }

    /// An estimated θ with no ETA is not mu-referenced, so it never gets the
    /// γ-damped closed-form shift and is left to the η-frozen numerical M-step.
    /// SAEM must name it, the way IMP/IMPMAP already do (#406) — on the FREM
    /// `iiv_on_ruv` reprex that channel drove `TVFRD1` 0.383 → 0.039 against
    /// IMP 0.311 / IMPMAP 0.318 / NONMEM 0.394, and attaching an ETA recovered
    /// 0.313.
    #[test]
    fn saem_warns_when_an_estimated_theta_has_no_eta() {
        let ws = saem_noeta_warnings("TVV");
        let hit = ws
            .iter()
            .find(|w| w.contains("NO associated ETA"))
            .unwrap_or_else(|| panic!("expected the no-ETA advisory, got {ws:?}"));
        assert!(
            hit.starts_with("SAEM:") && hit.contains("TVV"),
            "advisory must be SAEM-labelled and name TVV: {hit}"
        );
        assert!(
            !hit.contains("TVCL"),
            "TVCL is mu-referenced and must not be named: {hit}"
        );
    }

    /// #1011 keys off the fixed-effect-only channel, not off "NLopt has some
    /// free θ". With `mu_referencing = false` every θ goes through the numerical
    /// M-step, but a θ that anchors a mu-reference (detected here even though the
    /// shift is switched off) is not the biased channel — capping those at 3% for
    /// the whole exploration phase was never anchored, so such a fit keeps the
    /// pre-#1011 update and `mstep_damping` reports no effect.
    #[test]
    fn mstep_damping_is_inert_when_mu_referencing_is_off_but_every_theta_is_anchored() {
        let model = noeta_model("TVV * exp(ETA_V)");
        let pop = mix996_pop(3);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Saem;
        opts.saem_n_exploration = 4;
        opts.saem_n_convergence = 2;
        opts.saem_seed = Some(406);
        opts.run_covariance_step = false;
        opts.mu_referencing = false;
        opts.saem_mstep_damping = Some(0.01);
        let ws = crate::api::fit(&model, &pop, &model.default_params, &opts)
            .expect("SAEM Ok")
            .warnings;
        let hit = ws
            .iter()
            .find(|w| w.contains("`mstep_damping` was set but has no effect"))
            .unwrap_or_else(|| panic!("expected the no-effect warning, got {ws:?}"));
        // The reason must describe *this* fit. The single-population wording
        // ("every estimated theta ... is mu-referenced") would be false here:
        // `mu_referencing` is off, so nothing is mu-referenced and the numerical
        // M-step does move every theta (#1012 review).
        assert!(
            hit.contains("`mu_referencing` is off"),
            "the reason must name the mu_referencing = false case, not claim the \
             thetas are mu-referenced: {hit}"
        );
    }

    /// A `mstep_damping` that never passed the parser — a Rust caller building
    /// `FitOptions` directly — must be clamped and reported, not applied: a
    /// negative γ would walk θ/σ away from the M-step optimum every iteration.
    #[test]
    fn out_of_range_mstep_damping_from_a_programmatic_caller_is_clamped() {
        let model = noeta_model("TVV");
        let pop = mix996_pop(3);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Saem;
        opts.saem_n_exploration = 4;
        opts.saem_n_convergence = 2;
        opts.saem_seed = Some(406);
        opts.run_covariance_step = false;
        opts.saem_mstep_damping = Some(-0.5);
        let ws = crate::api::fit(&model, &pop, &model.default_params, &opts)
            .expect("SAEM Ok")
            .warnings;
        let hit = ws
            .iter()
            .find(|w| w.contains("is outside (0, 1]"))
            .unwrap_or_else(|| panic!("expected the clamp warning, got {ws:?}"));
        assert!(
            hit.contains("-0.5") && hit.contains(&format!("{MSTEP_SA_MAX_STEP}")),
            "warning must name the rejected value and the substitute: {hit}"
        );
    }

    /// The undamped assignment must be reproduced byte-for-byte at `γ >= 1.0`,
    /// and a pinned dimension (where NLopt returns `new == cur`) must be a no-op
    /// at every γ — those two properties are what keep a fit with no free
    /// numerical θ identical to its pre-#1011 result.
    #[test]
    fn damp_mstep_is_a_robbins_monro_blend() {
        let new = [1.0_f64, -2.0, 0.5];

        // γ = 1 (and above) → straight assignment, bit-for-bit.
        for g in [1.0_f64, 1.5] {
            let mut cur = [10.0_f64, 10.0, 10.0];
            damp_mstep(&mut cur, &new, g);
            assert_eq!(cur, new, "γ = {g} must assign outright");
        }

        // 0 < γ < 1 → cur += γ·(new − cur).
        let mut cur = [0.0_f64, 0.0, 0.0];
        damp_mstep(&mut cur, &new, 0.25);
        for (c, n) in cur.iter().zip(new.iter()) {
            assert!(
                (c - 0.25 * n).abs() < 1e-15,
                "expected {}, got {c}",
                0.25 * n
            );
        }

        // γ = 0 → frozen.
        let mut cur = [3.0_f64, 4.0, 5.0];
        let before = cur;
        damp_mstep(&mut cur, &new, 0.0);
        assert_eq!(cur, before);

        // A pinned dim (new == cur) is untouched at any γ — no drift from the
        // closed-form mu-ref value the M-step was told to leave alone.
        for g in [0.0_f64, 0.03, 0.5, 1.0] {
            let mut cur = [7.0_f64; 3];
            damp_mstep(&mut cur, &[7.0; 3], g);
            assert_eq!(cur, [7.0; 3], "pinned dim moved at γ = {g}");
        }
    }

    /// The SA schedule: off entirely when NLopt has no θ to estimate (so those
    /// fits keep their pre-#1011 numbers); when the user sets a cap, capped in
    /// exploration and full decaying γ in convergence; and off by default
    /// (#1415), which is the sentinel value in both phases.
    #[test]
    fn mstep_sa_step_caps_exploration_and_frees_convergence() {
        let d = 0.03; // the pre-#1415 default, now an opt-in value

        // No numerically-estimated θ → undamped, in both phases.
        assert_eq!(mstep_sa_step(false, true, 1.0, d), 1.0);
        assert_eq!(mstep_sa_step(false, false, 0.002, d), 1.0);

        // Exploration (γ = 1.0) is capped.
        assert_eq!(mstep_sa_step(true, true, 1.0, d), d);
        // ...but a γ already below the cap is not raised to it.
        assert_eq!(mstep_sa_step(true, true, 0.001, d), 0.001);
        // Convergence uses the full decaying γ, uncapped.
        assert_eq!(mstep_sa_step(true, false, 0.5, d), 0.5);
        assert_eq!(mstep_sa_step(true, false, 0.002, d), 0.002);

        // `mstep_damping` overrides the cap.
        assert_eq!(mstep_sa_step(true, true, 1.0, 0.005), 0.005);

        // The "off" value of 1.0 is assignment in BOTH phases — convergence
        // included, where the schedule would otherwise still damp at
        // γ = 1/(k−k1). Regression: `cap = 1.0` first only lifted the
        // exploration cap, which left the #1011 reprex at TVFRD1 0.047 instead
        // of its true undamped 0.039.
        assert_eq!(mstep_sa_step(true, true, 1.0, 1.0), 1.0);
        assert_eq!(mstep_sa_step(true, false, 0.002, 1.0), 1.0);
        assert_eq!(mstep_sa_step(true, false, 0.5, 1.0), 1.0);

        // #1415: the default IS that "off" value. A capped default divides the
        // number of EM steps the exploration phase amounts to and froze every
        // no-ETA theta near its start (`covmuref_power` TH_WT 0.323 at 0.03
        // against NONMEM's 0.921; 0.961 off) — see `MSTEP_SA_MAX_STEP`.
        assert_eq!(MSTEP_SA_MAX_STEP, 1.0);
        assert_eq!(mstep_sa_step(true, true, 1.0, MSTEP_SA_MAX_STEP), 1.0);
        assert_eq!(mstep_sa_step(true, false, 0.004, MSTEP_SA_MAX_STEP), 1.0);
    }

    /// The σ schedule (#1445): `min(γ, 0.2, γ_mstep)`, capped in **both**
    /// phases. The two properties that matter are the ones the collapse turned
    /// on — exploration must not assign, and the first convergence iteration
    /// (where `γ = 1/(k−k1) = 1`) must not assign either — plus the decay
    /// surviving intact once `1/(k−k1) ≤ 0.2` (`k − k1 ≥ 5`), without which the
    /// SA estimate never settles.
    ///
    /// Regression this exists to catch: dropping either cap puts σ back on a
    /// single draw of the M-step maximiser, which is #1445. Mutation check —
    /// returning `gamma_mstep` (the pre-#1445 expression) fails the first two
    /// assertions; returning a bare `SIGMA_SA_MAX_STEP` fails the decay ones.
    #[test]
    fn sigma_mstep_sa_step_never_assigns_and_keeps_the_decay() {
        // Exploration: γ = 1 with the damping off must NOT come back as an
        // assignment. This is the pre-#1445 behaviour, and the whole bug.
        assert_eq!(sigma_mstep_sa_step(1.0, 1.0), SIGMA_SA_MAX_STEP);
        // First convergence iteration: γ = 1/(k−k1) = 1 at k = k1+1. Ω takes
        // that hand-off assignment deliberately; σ must not.
        assert_eq!(sigma_mstep_sa_step(1.0, 1.0), SIGMA_SA_MAX_STEP);
        assert!(sigma_mstep_sa_step(1.0, 1.0) < 1.0);

        // The cap binds for the first five convergence iterations only...
        assert_eq!(sigma_mstep_sa_step(1.0 / 3.0, 1.0), SIGMA_SA_MAX_STEP);
        assert_eq!(sigma_mstep_sa_step(1.0 / 5.0, 1.0), SIGMA_SA_MAX_STEP);
        // ...and past that the full decaying Robbins-Monro schedule is intact.
        assert_eq!(sigma_mstep_sa_step(1.0 / 6.0, 1.0), 1.0 / 6.0);
        assert_eq!(sigma_mstep_sa_step(1.0 / 250.0, 1.0), 1.0 / 250.0);

        // σ never takes a larger step than θ: with `mstep_damping` set (the
        // `iiv_on_ruv` default) the pair stays locked, so that shape is
        // unchanged wherever γ_mstep is the binding term.
        assert_eq!(
            sigma_mstep_sa_step(1.0, MSTEP_SA_MAX_STEP_IIV_ON_RUV),
            MSTEP_SA_MAX_STEP_IIV_ON_RUV
        );
        assert_eq!(sigma_mstep_sa_step(0.5, 0.004), 0.004);
        // The cap is a measured value, not Ω's (see `SIGMA_SA_MAX_STEP` for the
        // sweep). Pinned so a future edit has to revisit that table: a *smaller*
        // cap makes σ lag (0.1 is the only swept value whose cefepime objective
        // is worse than the un-averaged baseline), a larger one lets a seed
        // partially collapse again (0.3 realises σ_add = 0.47 on cefepime).
        assert_eq!(SIGMA_SA_MAX_STEP, 0.2);
    }

    /// The σ half of every `theta_sigma_mstep_light` result goes through
    /// [`damp_mstep_sigma_variance`] at [`sigma_mstep_sa_step`]'s γ — a source
    /// check, because the behaviour it pins has no cheaper test.
    ///
    /// The two tests above cover the schedule and the blend as *functions*;
    /// neither can see the smallest edit that removes #1445, which is to leave
    /// both functions alone and restore the pre-#1445 call —
    /// `damp_mstep(&mut log_sigma, &sigma_new, gamma_mstep)` — at the three
    /// sites. Mutation-checked: that revert leaves every unit test in this file
    /// green and is caught only by
    /// `tests/saem_combined_error.rs::saem_sparse_combined_additive_sigma_is_not_a_single_draw`,
    /// which is `slow-tests`-gated and therefore never runs on a PR. This test
    /// closes that window on the PR job.
    ///
    /// Whitespace is stripped before matching, so rustfmt may re-wrap the calls
    /// freely; renaming a local (`log_sigma`, `sigma_new`) is what would need
    /// this test updated, and that is the point — the rename should have to look
    /// here.
    #[test]
    fn the_sigma_mstep_result_is_blended_not_assigned() {
        // Comment lines are dropped first (this test's doc comment quotes the
        // forbidden call, and so does `sigma_mstep_sa_step`'s), and every needle
        // is assembled from fragments at runtime so the assertions below do not
        // match themselves.
        let src: String = include_str!("saem.rs")
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .flat_map(|l| l.chars())
            .filter(|c| !c.is_whitespace())
            .collect();
        let sigma_arg = format!("(&mut{}_sigma", "log");

        let forbidden = format!("{}{sigma_arg}", "damp_mstep");
        assert!(
            !src.contains(&forbidden),
            "σ must not ride the θ blend (`{forbidden}`, ..) — that is the pre-#1445 assignment"
        );

        let wanted = format!(
            "{}{sigma_arg},&sigma_new,gamma_sigma)",
            "damp_mstep_sigma_variance"
        );
        let calls = src.matches(&wanted).count();
        assert_eq!(
            calls, 3,
            "expected the three θ/σ M-step arms (mu-ref, mixture, mu_referencing = false) to \
             blend σ at γ_σ; found {calls} of `{wanted}`"
        );

        // ...and that the γ they are handed is the σ schedule, not θ's.
        let schedule = format!(
            "letgamma_sigma={}(gamma,gamma_mstep)",
            "sigma_mstep_sa_step"
        );
        assert!(
            src.contains(&schedule),
            "γ_σ must come from the σ schedule: `{schedule}` not found"
        );
    }

    /// The σ blend is on the **variance** scale, not the packed one (#1445).
    ///
    /// Regression this exists to catch: re-spelling the blend as
    /// `damp_mstep(&mut log_sigma, …)` — the pre-#1445 call, and the obvious
    /// "simplification" — which is a geometric mean of the maximiser sequence.
    /// The last assertion is the discriminator: on a two-point sequence with the
    /// boundary-heavy shape a minority variance component actually produces, the
    /// two spellings differ by a **measured 10.000125×**, and log space is the
    /// collapsed one. A test that only checked the γ ≥ 1 and pinned cases would
    /// pass under either spelling. Every bound below is a closed form plus a
    /// floating-point tolerance, not a fitted number.
    #[test]
    fn damp_mstep_sigma_variance_blends_on_the_variance_scale() {
        // γ ≥ 1 assigns, bit-for-bit — the undamped path stays reproducible.
        for g in [1.0_f64, 2.0] {
            let mut cur = [0.5_f64, -1.0];
            let new = [(0.2_f64).ln(), (3.0_f64).ln()];
            damp_mstep_sigma_variance(&mut cur, &new, g);
            assert_eq!(cur, new, "γ = {g} must assign outright");
        }

        // A FIXed / pinned σ (NLopt returns it unchanged) is a no-op at any γ.
        for g in [0.0_f64, 0.1, 0.5, 1.0] {
            let mut cur = [(1.7_f64).ln(); 2];
            damp_mstep_sigma_variance(&mut cur, &[(1.7_f64).ln(); 2], g);
            for c in cur {
                assert!(
                    (c - (1.7_f64).ln()).abs() < 1e-15,
                    "pinned σ moved at γ = {g}"
                );
            }
        }

        // The blend itself: σ² += γ·(σ_new² − σ²). A closed form, so the bound
        // is a floating-point tolerance and nothing else:
        // σ = √(0.75·2² + 0.25·4²) = √7 = 2.6457513110645906.
        let mut cur = [(2.0_f64).ln()];
        damp_mstep_sigma_variance(&mut cur, &[(4.0_f64).ln()], 0.25);
        let want = 7.0_f64.sqrt();
        assert!(
            (cur[0].exp() - want).abs() < 1e-12,
            "variance blend: got {}, want {want}",
            cur[0].exp()
        );
        // ...which is NOT the packed-scale blend, whose closed form here is
        // exp(0.75·ln2 + 0.25·ln4) = 2.3784142300054421. The gap is a measured
        // 0.267337, so the 1e-12 tolerance above is ~11 orders below the
        // difference it has to resolve and cannot pass on the wrong formula.
        let log_space = (0.75 * (2.0_f64).ln() + 0.25 * (4.0_f64).ln()).exp();
        let gap = cur[0].exp() - log_space;
        assert!(
            (gap - 0.267_337).abs() < 1e-5,
            "the two spellings must differ by the measured 0.267337, got {gap}"
        );

        // A non-finite maximiser leaves that coordinate alone rather than
        // poisoning the running average with NaN.
        let mut cur = [(1.5_f64).ln(), (2.5_f64).ln()];
        let before = cur;
        damp_mstep_sigma_variance(&mut cur, &[f64::NAN, f64::INFINITY], 0.1);
        assert_eq!(cur, before);

        // The discriminator. A boundary-heavy sequence — half the M-step
        // maximisers near the σ floor, half at a healthy value — averages to
        // something usable on the variance scale and collapses in log space.
        // The numbers are the shape measured on the #1445 repro (floor-adjacent
        // draws of 0.01 against healthy draws of 2.0).
        //
        // Both limits are closed forms at γ_i = 1/i. The log-space one is the
        // plain geometric mean √(0.01·2) = √0.02 = 0.1414213562373095; the
        // variance-scale one is 1.4142312399321406 — close to √2 but not equal
        // to it, because the last step's γ = 1/8 leaves the 0.01 draw a residue.
        // Realised separation: **10.000125×**.
        let seq = [0.01_f64, 2.0, 0.01, 2.0, 0.01, 2.0, 0.01, 2.0];
        let mut var_scale = [(1.0_f64).ln()];
        let mut log_scale = [(1.0_f64).ln()];
        for (i, &s) in seq.iter().enumerate() {
            let g = 1.0 / (i + 1) as f64;
            damp_mstep_sigma_variance(&mut var_scale, &[s.ln()], g);
            damp_mstep(&mut log_scale, &[s.ln()], g);
        }
        let want_var = 1.414_231_239_932_140_6_f64;
        let want_log = 0.02_f64.sqrt(); // the geometric mean of 0.01 and 2
        assert!(
            (var_scale[0].exp() - want_var).abs() < 1e-12,
            "variance-scale average: got {}, want {want_var}",
            var_scale[0].exp()
        );
        assert!(
            (log_scale[0].exp() - want_log).abs() < 1e-12,
            "log-scale average: got {}, want {want_log}",
            log_scale[0].exp()
        );
        let ratio = var_scale[0].exp() / log_scale[0].exp();
        assert!(
            (ratio - 10.000_125).abs() < 1e-5,
            "the measured separation is 10.000125×, got {ratio}"
        );
    }

    /// The default cap is keyed to the one shape it was measured to help
    /// (#1415): `iiv_on_ruv` keeps #1011's 0.03 (its reprex drifts undamped —
    /// TVFRD1 0.18 against the held 0.36, 47 FOCEI units worse — and the same
    /// model without `iiv_on_ruv` lands on NONMEM IMP undamped), everything
    /// else is off. Mutation check: returning 0.03 for both freezes the Tier-3
    /// `saem_recovers_the_allometric_exponent_on_the_numerical_mstep`; returning
    /// 1.0 for both is the #1011 regression by default.
    #[test]
    fn default_mstep_damping_caps_only_iiv_on_ruv() {
        assert_eq!(default_mstep_damping(true), MSTEP_SA_MAX_STEP_IIV_ON_RUV);
        assert_eq!(default_mstep_damping(true), 0.03);
        assert_eq!(default_mstep_damping(false), MSTEP_SA_MAX_STEP);
        assert_eq!(default_mstep_damping(false), 1.0);
        // The iiv_on_ruv default is a real cap, so the schedule engages: capped
        // exploration, decaying γ in convergence.
        let d = default_mstep_damping(true);
        assert_eq!(mstep_sa_step(true, true, 1.0, d), d);
        assert_eq!(mstep_sa_step(true, false, 0.5, d), 0.5);
        // And a fit that sets `mstep_damping` overrides both defaults (the
        // caller's `match options.saem_mstep_damping`), so the constants are
        // only ever the `None` arm.
    }

    /// The damping gate (#1011). A mixture is vetoed outright; otherwise the
    /// damping runs exactly when NLopt is left estimating a θ that anchors no
    /// mu-reference, so an all-mu-referenced or all-`FIX` fit keeps its pre-#1011
    /// numbers.
    #[test]
    fn damps_numerical_mstep_only_for_a_free_unanchored_theta() {
        let free = [false, false, false];
        let no_eta = [false, false, false];

        // One free, unpinned θ anchoring no mu-reference (the #1011 shape) → damp.
        assert!(damps_numerical_mstep(
            false,
            3,
            &free,
            &[0, 1],
            &[true, true, false]
        ));
        // Every θ pinned by the closed-form mu-ref shift → nothing to damp.
        assert!(!damps_numerical_mstep(false, 3, &free, &[0, 1, 2], &no_eta));
        // Every θ FIXed → nothing to damp.
        assert!(!damps_numerical_mstep(
            false,
            3,
            &[true, true, true],
            &[],
            &no_eta
        ));
        // Mixed: the only unpinned θ is also FIXed → nothing to damp.
        assert!(!damps_numerical_mstep(
            false,
            3,
            &[false, false, true],
            &[0, 1],
            &no_eta
        ));
        // Nothing pinned, but every free θ anchors a mu-reference — the
        // `mu_referencing = false` / identity-packed shape. Those θ go through
        // the numerical M-step too, but they have the exact closed-form shift
        // available, and damping ~85 exploration M-steps to 3% on a configuration
        // none of the #1011 anchors covered would be a blind change: leave them
        // undamped.
        assert!(!damps_numerical_mstep(
            false,
            3,
            &free,
            &[],
            &[true, true, true]
        ));
        // ...but an un-anchored θ in that same unpinned fit does trip it.
        assert!(damps_numerical_mstep(
            false,
            3,
            &free,
            &[],
            &[true, false, true]
        ));
        // No thetas at all (σ-only M-step) → undamped, as before #1011.
        assert!(!damps_numerical_mstep(false, 0, &[], &[], &[]));

        // A mixture is vetoed even with a free unpinned, un-anchored θ — its class
        // typical values must separate from a common start before the class
        // assignments settle, and damping that stalls it.
        assert!(damps_numerical_mstep(false, 3, &free, &[], &no_eta));
        assert!(!damps_numerical_mstep(true, 3, &free, &[], &no_eta));
        assert!(!damps_numerical_mstep(true, 3, &free, &[0, 1], &no_eta));
    }

    /// `mstep_damping` is validated by the parser, but `FitOptions` is public:
    /// a programmatic caller can hand `run_saem` a γ the parser would have
    /// rejected. Those must be clamped, not applied — a negative γ steps θ/σ
    /// away from the M-step optimum every iteration, and 0.0 freezes them.
    #[test]
    fn sanitize_mstep_damping_clamps_out_of_range_values() {
        // In range → untouched.
        for v in [f64::MIN_POSITIVE, 0.003, MSTEP_SA_MAX_STEP, 0.5, 1.0] {
            assert_eq!(sanitize_mstep_damping(v), None, "{v} is in (0, 1]");
        }
        // Non-positive, or NaN → back to the calibrated default.
        for v in [0.0, -0.0, -0.5, f64::NAN, f64::NEG_INFINITY] {
            assert_eq!(
                sanitize_mstep_damping(v),
                Some(MSTEP_SA_MAX_STEP),
                "{v} must fall back to the default"
            );
        }
        // Above 1 already meant "off" → the documented off value. `+∞` belongs
        // here, not with the fallbacks: a caller writing `f64::INFINITY` means
        // "no damping", and the old `!is_finite()` test sent it to the *maximum*
        // damping instead — the exact opposite of the intent (#1012 review).
        for v in [1.0000001, 2.0, 1e9, f64::INFINITY] {
            assert_eq!(
                sanitize_mstep_damping(v),
                Some(1.0),
                "{v} must clamp to 1.0"
            );
        }
        // ...and the clamped value must actually disable the damping.
        assert_eq!(mstep_sa_step(true, true, 1.0, 1.0), 1.0);
    }

    /// ...and stays quiet once every estimated θ carries one, so the advisory
    /// keeps its signal.
    #[test]
    fn saem_no_eta_advisory_silent_when_every_theta_is_mu_referenced() {
        let ws = saem_noeta_warnings("TVV * exp(ETA_V)");
        assert!(
            !ws.iter().any(|w| w.contains("NO associated ETA")),
            "no advisory when every θ is mu-referenced, got {ws:?}"
        );
    }

    /// Warnings from a two-class mixture fit, whose `V` is either
    /// mu-referenceable or a bare fixed-effect-only θ.
    fn saem_mixture_warnings(v_expr: &str) -> Vec<String> {
        let model = mix996_model(v_expr);
        let pop = mix996_pop(3);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Saem;
        opts.saem_n_exploration = 4;
        opts.saem_n_convergence = 2;
        opts.saem_seed = Some(406);
        opts.run_covariance_step = false;
        crate::api::fit(&model, &pop, &model.default_params, &opts)
            .expect("SAEM Ok")
            .warnings
    }

    /// The no-ETA advisory must not name a θ for which "add an ETA" is wrong
    /// advice (#1012 review). Two such classes exist in `mix996_model`:
    ///
    /// * `MIXL` is a mixing coefficient. The numerical θ/σ M-step never moves it
    ///   (it does not enter the residual/η likelihood, and `mstep_mixing`
    ///   overwrites it from the responsibilities), and the parser *forbids* an η
    ///   in a mixing expression — so both the diagnosis and the remedy are wrong.
    /// * `TVCL1` / `TVCL2` are MIXNUM-switched class typical values. They carry
    ///   `ETA_CL` in every arm; they are on the numerical M-step because #996
    ///   routes them there, which the preceding advisory already says.
    ///
    /// With `V` mu-referenced, that leaves nothing for this advisory to say.
    #[test]
    fn saem_no_eta_advisory_skips_mixing_and_already_explained_class_thetas() {
        let ws = saem_mixture_warnings("TVV * exp(ETA_V)");
        assert!(
            !ws.iter().any(|w| w.contains("NO associated ETA")),
            "MIXL / TVCL1 / TVCL2 must not be flagged as ETA-less, got {ws:?}"
        );
        // The #996 message that explains TVCL1/TVCL2 must still be there — the
        // filter suppresses the wrong message, not the right one.
        assert!(
            ws.iter()
                .any(|w| w.contains("MIXNUM-switched typical value")
                    && w.contains("TVCL1")
                    && w.contains("TVCL2")),
            "the #996 advisory must still name the switched class thetas, got {ws:?}"
        );
    }

    /// ...and the filter keeps its signal: a genuinely fixed-effect-only θ in the
    /// same mixture is still named, and named *alone*.
    #[test]
    fn saem_no_eta_advisory_still_fires_for_a_real_no_eta_theta_in_a_mixture() {
        let ws = saem_mixture_warnings("TVV");
        let hit = ws
            .iter()
            .find(|w| w.contains("NO associated ETA"))
            .unwrap_or_else(|| panic!("expected the no-ETA advisory, got {ws:?}"));
        assert!(hit.contains("TVV"), "must name TVV: {hit}");
        for skipped in ["MIXL", "TVCL1", "TVCL2"] {
            assert!(
                !hit.contains(skipped),
                "{skipped} must stay filtered out: {hit}"
            );
        }
    }

    #[test]
    fn saem_mixture_uses_closed_form_for_class_shared_mu_ref() {
        // Before #996 a mixture disabled mu-referencing wholesale, so even a
        // class-*shared* typical value (V = TVV * exp(ETA_V)) went through the
        // numerical M-step. It now takes the pooled closed-form shift, which the
        // saved-evaluation counter reports.
        let model = mix996_model("TVV * exp(ETA_V)");
        let pop = mix996_pop(3);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Saem;
        opts.saem_n_exploration = 4;
        opts.saem_n_convergence = 2;
        opts.saem_seed = Some(996);
        opts.run_covariance_step = false;
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("SAEM Ok");
        assert!(
            res.saem_mu_ref_m_step_evals_saved.unwrap_or(0) > 0,
            "class-shared mu-ref must take the closed-form M-step: {:?}",
            res.saem_mu_ref_m_step_evals_saved
        );
        // ...and the MIXNUM-switched clearances must be named as staying numerical.
        assert!(
            res.warnings.iter().any(|w| {
                w.contains("MIXNUM-switched typical value")
                    && w.contains("TVCL1")
                    && w.contains("TVCL2")
            }),
            "expected the #996 switched-theta warning, got {:?}",
            res.warnings
        );
    }

    #[test]
    fn saem_mixture_mu_referencing_off_suppresses_the_muref_advisories() {
        // With mu_referencing off every θ goes through the numerical M-step by
        // construction, so advising the user to switch estimator to get a
        // closed-form shift they turned off is noise (#996 review).
        let model = mix996_model("TVV * exp(ETA_V)");
        let pop = mix996_pop(3);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Saem;
        opts.saem_n_exploration = 4;
        opts.saem_n_convergence = 2;
        opts.saem_seed = Some(996);
        opts.run_covariance_step = false;
        opts.mu_referencing = false;
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("SAEM Ok");
        assert!(
            !res.warnings
                .iter()
                .any(|w| w.contains("MIXNUM-switched typical value")),
            "no switched-theta advisory with mu_referencing off, got {:?}",
            res.warnings
        );
        assert!(
            !res.warnings
                .iter()
                .any(|w| w.contains("packed on the identity scale")),
            "no identity-pack advisory with mu_referencing off, got {:?}",
            res.warnings
        );
    }

    /// SAEM twin of the IMP test of the same name (#918): a logit mu-ref whose
    /// theta is log-packed takes no closed-form pair and the advisory names it.
    /// `TVCL` (log-packed lognormal) still takes the closed form, which is why
    /// the saved-eval count is non-zero here.
    #[test]
    fn saem_log_packed_logit_mu_ref_routes_to_numerical_mstep() {
        let src = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta LOGIT_F(0.5, 0.0, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_F ~ 0.04
  sigma EPS ~ 0.04 FIX

[individual_parameters]
  F  = inv_logit(LOGIT_F + ETA_F)
  CL = TVCL * exp(ETA_CL) * F
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";
        let model = crate::parser::model_parser::parse_model_string(src).expect("parses");
        let pop = mix996_pop(3);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Saem;
        opts.saem_n_exploration = 4;
        opts.saem_n_convergence = 2;
        opts.saem_seed = Some(918);
        opts.run_covariance_step = false;
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("SAEM Ok");
        let hit = res
            .warnings
            .iter()
            .find(|w| w.contains("packed on the log scale"))
            .unwrap_or_else(|| {
                panic!("expected the #918 packing advisory, got {:?}", res.warnings)
            });
        assert!(
            hit.contains("LOGIT_F"),
            "LOGIT_F named in the advisory: {hit}"
        );
        assert!(
            !hit.contains("TVCL"),
            "TVCL is a log-packed lognormal, not listed: {hit}"
        );
        assert!(
            res.saem_mu_ref_m_step_evals_saved.unwrap_or(0) > 0,
            "TVCL still takes the closed form: {:?}",
            res.saem_mu_ref_m_step_evals_saved
        );
        // The θ is un-explained by the "NO associated ETA" advisory: it carries
        // one, and the packing advisory above already says why it is numerical.
        assert!(
            !res.warnings
                .iter()
                .any(|w| w.contains("NO associated ETA") && w.contains("LOGIT_F")),
            "LOGIT_F must not also be reported as having no ETA, got {:?}",
            res.warnings
        );
    }

    /// With `mu_referencing = false` no closed-form shift runs at all, so the
    /// packing advisory is noise and must stay silent (same gate as #996).
    #[test]
    fn saem_mu_referencing_off_suppresses_the_log_packed_logit_advisory() {
        let src = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta LOGIT_F(0.5, 0.0, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_F ~ 0.04
  sigma EPS ~ 0.04 FIX

[individual_parameters]
  F  = inv_logit(LOGIT_F + ETA_F)
  CL = TVCL * exp(ETA_CL) * F
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";
        let model = crate::parser::model_parser::parse_model_string(src).expect("parses");
        let pop = mix996_pop(3);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Saem;
        opts.saem_n_exploration = 4;
        opts.saem_n_convergence = 2;
        opts.saem_seed = Some(918);
        opts.run_covariance_step = false;
        opts.mu_referencing = false;
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("SAEM Ok");
        assert!(
            !res.warnings
                .iter()
                .any(|w| w.contains("packed on the log scale")),
            "no packing advisory with mu_referencing off, got {:?}",
            res.warnings
        );
    }

    /// Eight subjects with CRCL 40..140, CL generated from the additive renal
    /// model (`CL = 5 + (CRCL-90)*0.05`, no IIV in the data) so a covariate
    /// group has real between-subject structure to fit in a handful of
    /// iterations. `V = 50`.
    fn covmuref_csv(with_v_eta: bool) -> String {
        let mut s = String::from("ID,TIME,DV,AMT,EVID,CMT,CRCL\n");
        for (i, crcl) in [40.0_f64, 55.0, 70.0, 85.0, 95.0, 110.0, 125.0, 140.0]
            .iter()
            .enumerate()
        {
            let id = i + 1;
            let cl = 5.0 + (crcl - 90.0) * 0.05;
            let v = if with_v_eta {
                50.0 * (0.1 * ((id as f64) - 4.5) / 4.5).exp()
            } else {
                50.0
            };
            s.push_str(&format!("{id},0,0,100,1,1,{crcl}\n"));
            for (ti, t) in [1.0_f64, 4.0, 8.0, 16.0, 24.0].iter().enumerate() {
                let c = 100.0 / v * (-(cl / v) * t).exp();
                let dv = c * (1.0 + 0.03 * ((id + ti) as f64).sin());
                s.push_str(&format!("{id},{t},{dv:.6},0,0,1,{crcl}\n"));
            }
        }
        s
    }

    fn covmuref_pop(with_v_eta: bool) -> Population {
        use std::io::Write;
        let csv = covmuref_csv(with_v_eta);
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(csv.as_bytes()).unwrap();
        crate::io::datareader::read_nonmem_csv(f.path(), Some(&["CRCL"]), None).unwrap()
    }

    /// The #619 additive form with the covariate group as the **only**
    /// eta-bearing parameter (`V` carries no eta), so a closed-form channel
    /// exists if and only if the group is active.
    const COVMUREF_ONLY_MODEL: &str = r"
[parameters]
  theta TVCL(4.0, 0.0, 100.0)
  theta TH_CRCL(0.02, 0.0, 5.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma EPS ~ 0.01 FIX

[individual_parameters]
  CL = (TVCL + (CRCL - 90.0) * TH_CRCL) * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

    /// #619: with the covariate group active, SAEM has a closed-form channel
    /// (the saved-eval count is `Some(>0)`), the parameter is not reported as
    /// un-mu-referenced, and no group note fires. Without the group this model
    /// has no mu-referenced parameter at all, so the count would be `None` —
    /// that is the discriminating signature, as in the #918 test.
    #[test]
    fn saem_covariate_mu_ref_group_drives_the_closed_form_channel() {
        let model =
            crate::parser::model_parser::parse_model_string(COVMUREF_ONLY_MODEL).expect("parses");
        assert_eq!(model.covariate_mu_refs.len(), 1);
        assert!(
            model.mu_refs.is_empty(),
            "no single-anchor pair in this model"
        );
        let pop = covmuref_pop(false);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Saem;
        opts.saem_n_exploration = 6;
        opts.saem_n_convergence = 3;
        opts.saem_seed = Some(619);
        opts.run_covariance_step = false;
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("SAEM Ok");
        assert!(
            res.saem_mu_ref_m_step_evals_saved.unwrap_or(0) > 0,
            "the group step must pin its thetas out of the numerical M-step: {:?}",
            res.saem_mu_ref_m_step_evals_saved
        );
        assert!(
            !res.warnings
                .iter()
                .any(|w| w.contains("not mu-referenced") && w.contains("CL")),
            "CL is mu-referenced through the group, got {:?}",
            res.warnings
        );
        assert!(
            !res.warnings
                .iter()
                .any(|w| w.contains("covariate mu-reference on")),
            "no group should be dropped, got {:?}",
            res.warnings
        );
        // The renal slope moved off its start toward the data-generating 0.05.
        let slope = res.theta[1];
        assert!(
            slope > 0.02,
            "TH_CRCL must move from its 0.02 start: {slope}"
        );
    }

    /// [`COVMUREF_ONLY_MODEL`] started where the additive typical value is
    /// negative for the lowest-CRCL subject (`4 − 50·0.2 = −6`), so both group
    /// engines return `None`.
    const COVMUREF_INADMISSIBLE_START: &str = r"
[parameters]
  theta TVCL(4.0, 0.0, 100.0)
  theta TH_CRCL(0.2, 0.0, 5.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma EPS ~ 0.01 FIX

[individual_parameters]
  CL = (TVCL + (CRCL - 90.0) * TH_CRCL) * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

    /// SAEM pins a group's thetas only *after* a successful solve, so an
    /// inadmissible start costs no freeze — but it silently reverts the group
    /// to the numerical M-step, which is the channel #619 exists to avoid. The
    /// fit must say so (#918 review).
    ///
    /// The straddle is [`saem_covariate_mu_ref_group_drives_the_closed_form_channel`],
    /// the same model from an admissible start, which asserts this warning does
    /// *not* fire — so a fix that always warned would fail there.
    #[test]
    fn saem_reports_a_covariate_group_that_never_stepped() {
        let model = crate::parser::model_parser::parse_model_string(COVMUREF_INADMISSIBLE_START)
            .expect("parses");
        let pop = covmuref_pop(false);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Saem;
        opts.saem_n_exploration = 4;
        opts.saem_n_convergence = 2;
        opts.saem_seed = Some(918);
        opts.run_covariance_step = false;
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("SAEM Ok");
        let hit = res
            .warnings
            .iter()
            .find(|w| w.contains("had no admissible step"))
            .unwrap_or_else(|| panic!("expected the skipped-step report, got {:?}", res.warnings));
        assert!(
            hit.contains("ETA_CL") && hit.contains("TH_CRCL"),
            "the report names the eta and the thetas: {hit}"
        );
    }

    /// `mu_referencing = false` turns the group off with the rest of the
    /// closed-form channel; the count is then `None` and no group note fires.
    #[test]
    fn saem_mu_referencing_off_disables_the_covariate_group() {
        let model =
            crate::parser::model_parser::parse_model_string(COVMUREF_ONLY_MODEL).expect("parses");
        let pop = covmuref_pop(false);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Saem;
        opts.saem_n_exploration = 4;
        opts.saem_n_convergence = 2;
        opts.saem_seed = Some(619);
        opts.run_covariance_step = false;
        opts.mu_referencing = false;
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("SAEM Ok");
        assert_eq!(res.saem_mu_ref_m_step_evals_saved, None);
    }

    /// A group whose theta is also another eta's single anchor is declined
    /// with a note naming both, and the fit proceeds on the numerical M-step.
    #[test]
    fn saem_covariate_group_sharing_an_anchor_is_declined_with_a_note() {
        let src = r"
[parameters]
  theta TVCL(4.0, 0.0, 100.0)
  theta TH_CRCL(0.02, 0.0, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.04
  sigma EPS ~ 0.01 FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = (TVCL * 10.0 + TH_CRCL * CRCL) * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";
        let model = crate::parser::model_parser::parse_model_string(src).expect("parses");
        let pop = covmuref_pop(true);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Saem;
        opts.saem_n_exploration = 3;
        opts.saem_n_convergence = 2;
        opts.saem_seed = Some(619);
        opts.run_covariance_step = false;
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("SAEM Ok");
        let note = res
            .warnings
            .iter()
            .find(|w| w.contains("covariate mu-reference on ETA_V"))
            .unwrap_or_else(|| panic!("expected the #619 note, got {:?}", res.warnings));
        assert!(note.contains("TVCL") && note.contains("ETA_CL"), "{note}");
    }

    #[test]
    fn saem_mixture_takes_a_class_shared_logit_anchor() {
        // SAEM's mixture path takes the class-**shared** anchors, and a logit one
        // qualifies exactly like a lognormal one once its theta is identity-packed
        // (#918). `saem_mu_ref_m_step_evals_saved` counts the θ the closed form
        // pinned, so it is `Some(> 0)` only if LOGIT_F is in the pair set — the
        // control being `saem_mixture_without_shared_mu_ref_stays_on_numerical_mstep`
        // below, whose identical fixture has no shared anchor and reports `None`.
        let src = r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(3.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta LOGIT_F(-0.405465, -10.0, 10.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_F ~ 0.09 FIX
  sigma EPS ~ 0.04 FIX

[mixture]
  nsub = 2
  logit(1) = MIXL

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 else TVCL2
  V  = TVV
  F  = inv_logit(LOGIT_F + ETA_F)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V, f=F)

[error_model]
  DV ~ proportional(EPS)
";
        let model = crate::parser::model_parser::parse_model_string(src).expect("parses");
        let pop = mix996_pop(3);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Saem;
        opts.saem_n_exploration = 4;
        opts.saem_n_convergence = 2;
        opts.saem_seed = Some(918);
        opts.run_covariance_step = false;
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("SAEM Ok");
        let saved = res
            .saem_mu_ref_m_step_evals_saved
            .expect("the class-shared logit anchor must drive the closed-form M-step");
        assert!(saved > 0, "expected pinned-θ eval savings, got {saved}");
        assert!(
            !res.warnings
                .iter()
                .any(|w| w.contains("packed on the log scale")),
            "LOGIT_F is identity-packed; no packing advisory expected, got {:?}",
            res.warnings
        );
    }

    #[test]
    fn saem_mixture_without_shared_mu_ref_stays_on_numerical_mstep() {
        // Only the class-switched CL is mu-ref-shaped; V carries no ETA. SAEM
        // takes no closed-form pair, so the whole θ/σ M-step stays numerical.
        let model = mix996_model("TVV");
        let pop = mix996_pop(3);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Saem;
        opts.saem_n_exploration = 4;
        opts.saem_n_convergence = 2;
        opts.saem_seed = Some(996);
        opts.run_covariance_step = false;
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("SAEM Ok");
        assert_eq!(res.saem_mu_ref_m_step_evals_saved, None);
    }
    use crate::types::test_helpers::analytical_model;
    use crate::types::{GradientMethod, MuRef};

    #[test]
    fn saem_final_ofv_report_formats_ofv_to_four_decimals() {
        // #893: the pre-covariance progress line reports the OFV to 4 dp so a
        // CLI user can judge the fit before the covariance step runs.
        assert_eq!(
            saem_final_ofv_report(1234.56789),
            "SAEM completed. Final OFV = 1234.5679"
        );
        assert_eq!(
            saem_final_ofv_report(-42.0),
            "SAEM completed. Final OFV = -42.0000"
        );
    }

    #[test]
    fn class_sigma_subst_is_none_without_overrides() {
        assert!(class_sigma_subst(&[0.2, 0.3], &[]).is_none());
    }

    #[test]
    fn class_sigma_subst_replaces_only_overridden_indices() {
        let out = class_sigma_subst(&[0.2, 0.3, 0.4], &[(2, 0.9)]).unwrap();
        assert_eq!(out, vec![0.2, 0.3, 0.9]);
        // Out-of-range indices are ignored rather than panicking.
        let out = class_sigma_subst(&[0.2], &[(7, 0.9)]).unwrap();
        assert_eq!(out, vec![0.2]);
    }

    /// The mixture M-step objective must actually reach the `MIXNUM` branch: the
    /// same subjects scored as class 1 vs class 2 must give different objective
    /// values, since the model's typical clearance switches on `MIXNUM` (#987
    /// review — this path was previously only exercised by slow-tests).
    #[test]
    fn obs_nll_sum_mix_class_guard_reaches_mixnum() {
        const MIX: &str = r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(5.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma EPS ~ 0.04

[mixture]
  nsub = 2
  logit(1) = MIXL
  sigma(2) EPS ~ 0.25

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";
        let model = crate::parser::model_parser::parse_model_string(MIX).unwrap();
        let mut csv = String::from(
            "ID,TIME,DV,AMT,EVID,CMT
",
        );
        for sid in 1..=2 {
            csv.push_str(&format!(
                "{sid},0,0,100,1,1
"
            ));
            for t in [0.5_f64, 1.0, 2.0, 4.0] {
                csv.push_str(&format!(
                    "{sid},{t},{:.5},0,0,1
",
                    10.0 * (-0.1 * t).exp()
                ));
            }
        }
        let mut f = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut f, csv.as_bytes()).unwrap();
        let pop = crate::io::datareader::read_nonmem_csv(f.path(), None, None).unwrap();

        let params = &model.default_params;
        let etas = vec![vec![0.0], vec![0.0]];
        let sigma = &params.sigma.values;
        let no_over: Vec<Vec<(usize, f64)>> = vec![Vec::new(), Vec::new()];

        let as_class1 = obs_nll_sum_mix(
            &model,
            &pop,
            &params.theta,
            sigma,
            &etas,
            MixMstep {
                classes: &[0, 0],
                class_sigma_over: &no_over,
            },
            &[],
        );
        let as_class2 = obs_nll_sum_mix(
            &model,
            &pop,
            &params.theta,
            sigma,
            &etas,
            MixMstep {
                classes: &[1, 1],
                class_sigma_over: &no_over,
            },
            &[],
        );
        assert!(as_class1.is_finite() && as_class2.is_finite());
        assert!(
            (as_class1 - as_class2).abs() > 1e-6,
            "class guard must switch TVCL1/TVCL2: {as_class1} vs {as_class2}"
        );

        // A held `sigma(2)` override must be what a class-2 subject is scored
        // under — not the free base σ the optimizer is moving.
        let mix = crate::estimation::saem_mixture::SaemMixture::build(&model, params, &pop);
        let over = mix.class_sigma_overrides();
        assert_eq!(over[1].len(), 1, "sigma(2) override present");
        let with_override = obs_nll_sum_mix(
            &model,
            &pop,
            &params.theta,
            sigma,
            &etas,
            MixMstep {
                classes: &[1, 1],
                class_sigma_over: &over,
            },
            &[],
        );
        assert!(
            (with_override - as_class2).abs() > 1e-6,
            "class-2 σ override must change the M-step objective"
        );
    }

    #[test]
    fn fold_nll_grad_sums_nll_and_grad_elementwise_in_input_order() {
        let per_subj = vec![
            (1.0, vec![1.0, 10.0]),
            (2.0, vec![2.0, 20.0]),
            (3.0, vec![3.0, 30.0]),
        ];
        let (nll, grad) = fold_nll_grad(per_subj, 2);
        assert_eq!(nll, 6.0);
        assert_eq!(grad, vec![6.0, 60.0]);
    }

    #[test]
    fn fold_nll_grad_of_empty_input_is_zero() {
        let (nll, grad) = fold_nll_grad(vec![], 3);
        assert_eq!(nll, 0.0);
        assert_eq!(grad, vec![0.0, 0.0, 0.0]);
    }

    /// Pin the SAEM M-step optimizer choice.
    ///
    /// BOBYQA (derivative-free trust-region) was chosen over the prior SLSQP
    /// after the Emax PKPD benchmark surfaced an Emax-Hill identifiability
    /// failure mode where SLSQP locks population thetas onto one side of the
    /// ridge (EMAX under-estimated by ~40%, OFV virtually identical to the
    /// nlmixr2-matching basin). BOBYQA's quadratic trust-region exploration
    /// lands much closer to truth at ~40% lower wall on that benchmark.
    /// Simpler PK-only models are numerically equivalent across the two
    /// algorithms (|ΔOFV| < 0.1).
    ///
    /// If a future change switches to a different algorithm — particularly
    /// any gradient-based one (LBFGS, SLSQP, MMA) — re-run the Emax PKPD
    /// regression in the experiment repo and confirm EMAX/EC50 recovery
    /// before merging. The OFV alone is NOT a sufficient regression signal
    /// here because the Hill ridge produces near-identical OFV at very
    /// different parameter values.
    #[test]
    fn mstep_uses_bobyqa_optimizer() {
        assert!(
            matches!(MSTEP_NLOPT_ALGORITHM, nlopt::Algorithm::Bobyqa),
            "MSTEP_NLOPT_ALGORITHM changed — see comment above this test \
             for the Emax-Hill identifiability rationale before adjusting."
        );
    }

    /// #1415 fixture: a 1-cpt IV model whose two thetas carry no ETA, three
    /// subjects, data generated at `CL = 2, V = 20` for a 100 mg bolus with a
    /// fixed ±10 % pattern (so σ has a non-degenerate maximiser).
    fn noeta_mstep_fixture() -> (CompiledModel, crate::types::Population) {
        use std::io::Write as _;

        const MODEL: &str = r#"
[parameters]
theta TVCL(1.0, 0.01, 200.0)
theta TVV(10.0, 0.1, 500.0)
sigma EPS ~ 0.05

[individual_parameters]
CL = TVCL
V = TVV

[structural_model]
pk one_cpt_iv(cl=CL, v=V)

[error_model]
DV ~ proportional(EPS)
"#;
        let model = crate::parser::model_parser::parse_model_string(MODEL).unwrap();

        let (cl, v, dose) = (2.0_f64, 20.0_f64, 100.0_f64);
        let mut csv = String::from("ID,TIME,DV,AMT,EVID,CMT\n");
        for id in 1..=3 {
            csv.push_str(&format!("{id},0,0,{dose},1,1\n"));
            for (j, t) in [1.0_f64, 2.0, 4.0, 8.0, 12.0].iter().enumerate() {
                let c = dose / v * (-cl / v * t).exp();
                let bump = if (id + j) % 2 == 0 { 1.1 } else { 0.9 };
                csv.push_str(&format!("{id},{t},{:.6},0,0,1\n", c * bump));
            }
        }
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(csv.as_bytes()).unwrap();
        let pop = crate::io::datareader::read_nonmem_csv(f.path(), None, None).unwrap();
        (model, pop)
    }

    /// #1415: the budgeted, warm-started numerical M-step must return the
    /// *nearby maximiser* of the η-frozen conditional likelihood, not a point
    /// somewhere along the way to it — or, as the old configuration did on this
    /// very fixture, a point further away.
    ///
    /// Fixture: a 1-cpt IV model whose two thetas carry no ETA (the #1415
    /// shape), three subjects, data generated at `CL = 2, V = 20` with a fixed
    /// ±10 % pattern (so σ has a non-degenerate maximiser), and the solve
    /// started 0.05 log units away on every free coordinate — the distance one
    /// SAEM iteration's blend typically leaves it from the next draw's
    /// maximiser. The reference is the same solver with a 300× budget from the
    /// same start.
    ///
    /// Measured on this fixture (packed-space |Δ| against the reference, then
    /// the objective gap), production budgets `mstep_maxiter = 3` (exploration)
    /// and `5` (convergence):
    ///
    /// | config | budget | CL | V | σ | gap |
    /// |---|---|---|---|---|---|
    /// | before #1415 (NLopt default first design, `ftol_rel = 1e-4`) | 3 | 0.060 | 0.046 | 0.044 | 1.19 |
    /// | before #1415 | 5 | 0.024 | 0.034 | 0.051 | 0.40 |
    /// | **now** | 3 | 0.005 | 0.004 | 0.048 | 0.045 |
    /// | **now** | 5 | 3e-4 | 2e-4 | 9e-4 | 1e-4 |
    ///
    /// The old solve started 0.05 off and came back 0.06 off on CL: its first
    /// `2n+1` design points sat at ×12–×26 of the start, and the trust-region
    /// steps the budget then allowed did not recover. Mutation check: drop
    /// `set_initial_step` and the CL bound fails at both budgets (realised
    /// 0.060 / 0.024). What this fixture pins is the first design; the
    /// `1e-4 → 1e-7` tolerance change is *not* observable here — on an
    /// objective of −11 a relative 1e-4 is already 1e-3 absolute — and is
    /// pinned by the production measurement in [`MSTEP_FTOL_REL`]'s doc
    /// instead (a 100-subject variant of this fixture does trip on it, but
    /// the tolerance then truncates the reference solve too, so the
    /// comparison stops meaning anything).
    #[test]
    fn budgeted_mstep_returns_the_converged_conditional_maximiser() {
        let (model, pop) = noeta_mstep_fixture();
        let (cl, v) = (2.0_f64, 20.0_f64);
        let etas: Vec<Vec<f64>> = vec![vec![]; pop.subjects.len()];

        // Start 0.05 log units from the generating values, on every coordinate.
        let d = 0.05_f64;
        let start_theta = vec![(cl.ln() - d), (v.ln() + d)];
        let start_sigma = vec![(0.1_f64).ln() + d];
        let theta_lower = vec![(0.01_f64).ln(), (0.1_f64).ln()];
        let theta_upper = vec![(200.0_f64).ln(), (500.0_f64).ln()];
        let sigma_lower = vec![-8.0];
        let sigma_upper = vec![5.0];
        let packs_log = vec![true, true];

        let solve = |maxiter: u32| {
            theta_sigma_mstep_light(
                &model,
                &pop,
                &etas,
                None,
                &start_theta,
                &start_sigma,
                &theta_lower,
                &theta_upper,
                &sigma_lower,
                &sigma_upper,
                2,
                1,
                maxiter,
                false,
                &packs_log,
                None,
                &[],
            )
        };
        let objective = |lt: &[f64], ls: &[f64]| {
            let th: Vec<f64> = lt.iter().map(|x| x.exp()).collect();
            let sg: Vec<f64> = ls.iter().map(|x| x.exp()).collect();
            obs_nll_sum(&model, &pop, &th, &sg, &etas, &[])
        };

        let (theta_ref, sigma_ref) = solve(900);
        let f_start = objective(&start_theta, &start_sigma);
        let f_ref = objective(&theta_ref, &sigma_ref);
        assert!(f_ref.is_finite() && f_start.is_finite());
        // The reference really is a maximiser: within 2 % of the generating
        // values (the ±10 % pattern is not exactly symmetric under a
        // proportional error, so the MLE is ~1 % off: realised CL 2.0195,
        // V 20.06), and the fixture is not degenerate.
        assert!(
            (theta_ref[0] - cl.ln()).abs() < 0.02 && (theta_ref[1] - v.ln()).abs() < 0.02,
            "reference solve must land near the generating CL / V: {:?}",
            theta_ref.iter().map(|x| x.exp()).collect::<Vec<_>>()
        );
        assert!(
            f_ref < f_start - 1.0,
            "fixture must be non-degenerate: reference {f_ref} vs start {f_start}"
        );

        // (budget, |Δθ| bound, |Δσ| bound, objective-gap bound): each bound is
        // ~2× the realised value in the table above, and every one sits below
        // what the old configuration realised.
        for (maxiter, tol_theta, tol_sigma, tol_gap) in
            [(3_u32, 0.01, 0.1, 0.1), (5_u32, 2e-3, 5e-3, 1e-2)]
        {
            let (theta_b, sigma_b) = solve(maxiter);
            let f_b = objective(&theta_b, &sigma_b);
            assert!(f_b.is_finite());
            for (i, (b, r)) in theta_b.iter().zip(theta_ref.iter()).enumerate() {
                assert!(
                    (b - r).abs() < tol_theta,
                    "maxiter {maxiter}, theta[{i}]: budgeted {b:.5} vs converged {r:.5} — the \
                     warm-started M-step stopped short (#1415)"
                );
            }
            assert!(
                (sigma_b[0] - sigma_ref[0]).abs() < tol_sigma,
                "maxiter {maxiter}, sigma: budgeted {:.5} vs converged {:.5}",
                sigma_b[0],
                sigma_ref[0]
            );
            assert!(
                f_b - f_ref < tol_gap,
                "maxiter {maxiter}: budgeted objective {f_b} must match the converged {f_ref}"
            );
        }

        // The mixture arm deliberately keeps the pre-#1415 configuration (see
        // `MSTEP_MIXTURE_FTOL_REL`): the same start under a (single-class)
        // `MixMstep` must reproduce the old partial step, not the converged
        // maximiser. Realised: CL 0.060 off at `mstep_maxiter = 3`, against
        // 0.005 on the non-mixture arm above. This pins the asymmetry so that
        // re-unifying the two arms is a deliberate change with its own anchor.
        let classes = vec![0usize; pop.subjects.len()];
        let class_sigma_over: Vec<Vec<(usize, f64)>> = vec![Vec::new()];
        let (theta_mix, _) = theta_sigma_mstep_light(
            &model,
            &pop,
            &etas,
            None,
            &start_theta,
            &start_sigma,
            &theta_lower,
            &theta_upper,
            &sigma_lower,
            &sigma_upper,
            2,
            1,
            3,
            false,
            &packs_log,
            Some(MixMstep {
                classes: &classes,
                class_sigma_over: &class_sigma_over,
            }),
            &[],
        );
        assert!(
            (theta_mix[0] - theta_ref[0]).abs() > 0.03,
            "mixture arm: CL {:.5} vs converged {:.5} — the mixture M-step is expected to keep \
             the pre-#1415 (partial-step) configuration; if this was changed on purpose, \
             re-anchor tests/mixture_nonmem.rs and update MSTEP_MIXTURE_FTOL_REL",
            theta_mix[0],
            theta_ref[0]
        );
    }

    /// The per-coordinate initial step is bounded by the coordinate's own
    /// interval (#1420 review): BOBYQA requires `upper − lower ≥ 2·step`.
    #[test]
    fn mstep_initial_step_is_bounded_by_the_interval() {
        // Wide interval: the nominal step.
        assert_eq!(
            mstep_initial_step((0.01_f64).ln(), (200.0_f64).ln()),
            MSTEP_INITIAL_STEP
        );
        assert_eq!(mstep_initial_step(-8.0, 5.0), MSTEP_INITIAL_STEP);
        // Narrow interval: a quarter of the width, leaving the factor-of-two
        // margin BOBYQA needs. CL on (1.9, 2.1) is 0.10008 log units wide.
        let (lo, hi) = ((1.9_f64).ln(), (2.1_f64).ln());
        let step = mstep_initial_step(lo, hi);
        assert!(step < MSTEP_INITIAL_STEP);
        assert!((step - (hi - lo) / 4.0).abs() < 1e-15);
        assert!(hi - lo >= 2.0 * step);
        // Exactly at the threshold and just above: still bounded by the width.
        assert!(mstep_initial_step(0.0, 0.4) <= 0.1 && mstep_initial_step(0.0, 0.4) > 0.0);
        assert!((mstep_initial_step(0.0, 0.2) - 0.05).abs() < 1e-15);
        // Pinned and unbounded coordinates keep a positive nominal step.
        assert_eq!(mstep_initial_step(0.7, 0.7), MSTEP_INITIAL_STEP);
        assert_eq!(
            mstep_initial_step(f64::NEG_INFINITY, f64::INFINITY),
            MSTEP_INITIAL_STEP
        );
        assert_eq!(mstep_initial_step(0.0, f64::INFINITY), MSTEP_INITIAL_STEP);
    }

    /// #1420 review (P1): a free theta declared on an interval narrower than
    /// twice the nominal step made BOBYQA reject the whole problem before its
    /// first evaluation, and because the outcome is discarded the joint θ/σ
    /// M-step silently returned its start on every iteration — freezing θ *and*
    /// σ. Same fixture as above with CL bounded to (1.9, 2.1), 0.10008 log
    /// units wide, started at 1.95. Mutation check: use the unbounded
    /// `MSTEP_INITIAL_STEP` for every coordinate and this fails on the
    /// `f_b < f_start` assertion with the start returned unchanged (in a debug
    /// build the `debug_assert!` on `InvalidArgs` fires first).
    #[test]
    fn mstep_moves_a_theta_declared_on_a_narrow_interval() {
        let (model, pop) = noeta_mstep_fixture();
        let etas: Vec<Vec<f64>> = vec![vec![]; pop.subjects.len()];

        let start_theta = vec![(1.95_f64).ln(), (20.0_f64).ln() + 0.05];
        let start_sigma = vec![(0.1_f64).ln() + 0.05];
        let theta_lower = vec![(1.9_f64).ln(), (0.1_f64).ln()];
        let theta_upper = vec![(2.1_f64).ln(), (500.0_f64).ln()];
        let sigma_lower = vec![-8.0];
        let sigma_upper = vec![5.0];
        let packs_log = vec![true, true];
        assert!(
            theta_upper[0] - theta_lower[0] < 2.0 * MSTEP_INITIAL_STEP,
            "fixture must be narrower than twice the nominal step to exercise the bound"
        );

        let objective = |lt: &[f64], ls: &[f64]| {
            let th: Vec<f64> = lt.iter().map(|x| x.exp()).collect();
            let sg: Vec<f64> = ls.iter().map(|x| x.exp()).collect();
            obs_nll_sum(&model, &pop, &th, &sg, &etas, &[])
        };
        let f_start = objective(&start_theta, &start_sigma);
        let (theta_b, sigma_b) = theta_sigma_mstep_light(
            &model,
            &pop,
            &etas,
            None,
            &start_theta,
            &start_sigma,
            &theta_lower,
            &theta_upper,
            &sigma_lower,
            &sigma_upper,
            2,
            1,
            5,
            false,
            &packs_log,
            None,
            &[],
        );
        let f_b = objective(&theta_b, &sigma_b);
        assert!(f_b.is_finite() && f_start.is_finite());
        assert!(
            f_b < f_start - 0.5,
            "the M-step must improve the conditional objective from a narrow-interval start: \
             {f_b} vs {f_start} (an unchanged start means NLopt rejected the problem)"
        );
        // Every free coordinate moved — the failure mode was all three frozen.
        assert!(
            (theta_b[0] - start_theta[0]).abs() > 1e-4,
            "CL did not move"
        );
        assert!((theta_b[1] - start_theta[1]).abs() > 1e-3, "V did not move");
        assert!(
            (sigma_b[0] - start_sigma[0]).abs() > 1e-3,
            "sigma did not move"
        );
        // And CL stayed inside its declared interval.
        assert!(theta_b[0] >= theta_lower[0] - 1e-12 && theta_b[0] <= theta_upper[0] + 1e-12);
    }

    #[test]
    fn scalar_residual_statistic_uses_the_averaged_rss_not_the_latest_draw() {
        // Fixture from the saemix 3.5 investigation: prior RSS = 2, retained
        // RSS = 18, gamma = 1/2, and two observations. The statistic is 10,
        // so the M-step SD is sqrt(5); a latest-draw update would be 3 instead.
        let mut statistic = None;
        update_scalar_residual_sse(&mut statistic, 2.0, 1.0);
        update_scalar_residual_sse(&mut statistic, 18.0, 0.5);
        let sse = statistic.expect("first retained draw initializes the statistic");
        assert!((sse - 10.0).abs() < 1e-12);
        assert!(((sse / 2.0).sqrt() - 5.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn scalar_residual_mstep_gate_accepts_only_stable_simple_models() {
        let model = noeta_model("TVV * exp(ETA_V)");
        let pop = mix996_pop(2);
        let mut params = model.default_params.clone();
        params.sigma_fixed[0] = false;
        let pairs = get_mu_ref_pairs(&model, &model.default_params.theta_lower);

        assert_eq!(
            scalar_residual_mstep_model(&model, &pop, &params, 0, false, true, &pairs),
            Some(ScalarResidualModel::Proportional),
        );

        // A free fixed-effect-only theta changes predictions in the numerical
        // M-step, so an RSS accumulated in its old coordinate system is not a
        // valid sufficient statistic.
        let unanchored = noeta_model("TVV");
        let mut unanchored_params = unanchored.default_params.clone();
        unanchored_params.sigma_fixed[0] = false;
        let unanchored_pairs =
            get_mu_ref_pairs(&unanchored, &unanchored.default_params.theta_lower);
        assert_eq!(
            scalar_residual_mstep_model(
                &unanchored,
                &pop,
                &unanchored_params,
                0,
                false,
                true,
                &unanchored_pairs,
            ),
            None,
        );

        // FIXed residual SDs are preserved exactly rather than re-estimated.
        params.sigma_fixed[0] = true;
        assert_eq!(
            scalar_residual_mstep_model(&model, &pop, &params, 0, false, true, &pairs),
            None,
        );
    }

    #[test]
    fn scalar_residual_sse_is_finite_for_additive_and_proportional_samples() {
        let model = noeta_model("TVV * exp(ETA_V)");
        let pop = mix996_pop(2);
        let etas = vec![vec![0.0; model.n_eta]; pop.subjects.len()];
        let additive = scalar_residual_sse(
            &model,
            &pop,
            &model.default_params.theta,
            &etas,
            ScalarResidualModel::Additive,
        )
        .expect("finite additive statistic");
        let proportional = scalar_residual_sse(
            &model,
            &pop,
            &model.default_params.theta,
            &etas,
            ScalarResidualModel::Proportional,
        )
        .expect("finite proportional statistic");
        assert_eq!(additive.1, 16);
        assert_eq!(proportional.1, additive.1);
        assert!(additive.0.is_finite() && additive.0 > 0.0);
        assert!(proportional.0.is_finite() && proportional.0 > 0.0);
        assert_ne!(additive.0, proportional.0);
    }

    #[test]
    fn scalar_residual_exact_fit_updates_sigma_to_lower_bound() {
        use std::io::Write as _;

        const ZERO_FIT_MODEL: &str = r#"
[parameters]
theta TVCL(1.0, 0.1, 100.0) FIX
theta TVV(10.0, 0.1, 1000.0) FIX
sigma EPS ~ 0.04

[individual_parameters]
CL = TVCL
V = TVV

[structural_model]
pk one_cpt_iv(cl=CL, v=V)

[error_model]
DV ~ additive(EPS)
"#;

        let model = crate::parser::model_parser::parse_model_string(ZERO_FIT_MODEL).unwrap();
        let mut csv = String::from("ID,TIME,DV,AMT,EVID,CMT\n");
        csv.push_str("1,1,0,0,0,1\n");

        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(csv.as_bytes()).unwrap();
        let pop = crate::io::datareader::read_nonmem_csv(f.path(), None, None).unwrap();

        let etas = vec![vec![]];
        let (sample_sse, n_obs) = scalar_residual_sse(
            &model,
            &pop,
            &model.default_params.theta,
            &etas,
            ScalarResidualModel::Additive,
        )
        .expect("exact-fit fixture should produce scalar SSE");
        assert_eq!(sample_sse, 0.0);
        assert_eq!(n_obs, 1);

        let mut statistic = None;
        update_scalar_residual_sse(&mut statistic, sample_sse, 1.0);
        let mut log_sigma = vec![0.0];
        let log_sigma_lower = vec![(-8.0_f64)];
        let log_sigma_upper = vec![1.0];

        if let Some(sse) = statistic {
            let sigma = (sse / n_obs as f64).sqrt();
            if sigma.is_finite() {
                log_sigma[0] = sigma.ln().clamp(log_sigma_lower[0], log_sigma_upper[0]);
            }
        }
        assert_eq!(log_sigma[0], log_sigma_lower[0]);
    }

    /// `combined_additive_sigma_at_floor` flags only a free additive component
    /// (sigma index 1) sitting at/below the near-floor band, and ignores
    /// non-combined specs and FIXed sigmas.
    #[test]
    fn combined_additive_sigma_at_floor_detects_collapsed_free_additive() {
        let mut model = analytical_model(GradientMethod::Fd);
        model.error_spec = ErrorSpec::Single(ErrorModel::Combined);

        let mut params = model.default_params.clone();
        params.sigma = SigmaVector {
            values: vec![0.1, 0.5],
            names: vec!["PROP".into(), "ADD".into()],
        };
        params.sigma_fixed = vec![false, false];

        // Healthy additive term well above the floor band.
        assert!(!combined_additive_sigma_at_floor(&model, &params));

        // Additive term collapsed onto the floor → flagged.
        params.sigma.values[1] = 5.0e-4;
        assert!(combined_additive_sigma_at_floor(&model, &params));

        // A FIXed additive at the floor is intentional, not a collapse.
        params.sigma_fixed[1] = true;
        assert!(!combined_additive_sigma_at_floor(&model, &params));

        // Non-combined specs never flag, even with a tiny second sigma.
        params.sigma_fixed[1] = false;
        model.error_spec = ErrorSpec::Single(ErrorModel::Proportional);
        assert!(!combined_additive_sigma_at_floor(&model, &params));
    }

    #[test]
    fn saem_sampler_summary_defaults_to_metropolis_hastings() {
        // Default options (saem_n_leapfrog = 0) → MH random walk in every build.
        let model = analytical_model(GradientMethod::Auto);
        let opts = crate::types::FitOptions::default();
        let s = saem_sampler_summary(&model, &opts);
        assert!(
            s.starts_with("Metropolis-Hastings"),
            "default SAEM kernel should be MH, got: {s}"
        );
        // Requesting leapfrog steps without HMC support must say so, not claim HMC.
        let mut hmc_opts = crate::types::FitOptions::default();
        hmc_opts.saem_n_leapfrog = 10;
        let s2 = saem_sampler_summary(&model, &hmc_opts);
        assert!(
            s2.starts_with("HMC"),
            "analytical model + leapfrog steps should use HMC (Dual2 gradient), got: {s2}"
        );
    }

    fn model_with_mu_refs(
        theta_names: &[&str],
        eta_names: &[&str],
        mu_refs: &[(&str, &str, MuTransform)],
    ) -> CompiledModel {
        let mut m = analytical_model(GradientMethod::Auto);
        m.theta_names = theta_names.iter().map(|s| (*s).to_string()).collect();
        m.eta_names = eta_names.iter().map(|s| (*s).to_string()).collect();
        m.n_theta = theta_names.len();
        m.n_eta = eta_names.len();
        m.mu_refs = mu_refs
            .iter()
            .map(|(eta, theta, transform)| {
                (
                    (*eta).to_string(),
                    MuRef {
                        theta_name: (*theta).to_string(),
                        transform: *transform,
                    },
                )
            })
            .collect();
        m
    }

    // ── Class-aware mu-referencing for mixtures (#996) ──────────────────

    /// `model_with_mu_refs` plus a `MixtureSpec` carrying `n_classes` and the
    /// given class-aware anchors (`(eta, [theta per class])`).
    fn model_with_mixture_mu_refs(
        theta_names: &[&str],
        eta_names: &[&str],
        mu_refs: &[(&str, &str, MuTransform)],
        n_classes: usize,
        class_refs: &[(&str, Vec<&str>)],
    ) -> CompiledModel {
        let mut m = model_with_mu_refs(theta_names, eta_names, mu_refs);
        m.mixture = Some(crate::types::MixtureSpec {
            n_classes,
            mixing: Vec::new(),
            logit_covariates: Vec::new(),
            omega_overrides: Vec::new(),
            sigma_overrides: Vec::new(),
            mu_refs: class_refs
                .iter()
                .map(|(eta, thetas)| crate::types::MixtureMuRef {
                    eta_name: (*eta).to_string(),
                    theta_names: thetas.iter().map(|t| (*t).to_string()).collect(),
                    log_transformed: true,
                })
                .collect(),
        });
        m
    }

    #[test]
    fn mixture_mu_ref_pairs_empty_for_non_mixture() {
        let m = model_with_mu_refs(
            &["TVCL"],
            &["ETA_CL"],
            &[("ETA_CL", "TVCL", MuTransform::Log)],
        );
        assert!(get_mixture_mu_ref_pairs(&m).is_empty());
    }

    #[test]
    fn mixture_mu_ref_pairs_map_class_switched_anchors() {
        let m = model_with_mixture_mu_refs(
            &["TVCL1", "TVCL2", "TVV"],
            &["ETA_CL"],
            &[],
            2,
            &[("ETA_CL", vec!["TVCL1", "TVCL2"])],
        );
        assert_eq!(
            get_mixture_mu_ref_pairs(&m),
            vec![MixtureMuRefPair {
                eta_idx: 0,
                theta_idx: vec![0, 1],
                transform: MuTransform::Log,
            }]
        );
    }

    #[test]
    fn mixture_mu_ref_pairs_broadcast_class_shared_theta() {
        // A plain (non-switched) mu-ref in a mixture model becomes the same
        // theta in every class slot, so one update rule covers both shapes.
        let m = model_with_mixture_mu_refs(
            &["TVCL1", "TVCL2", "TVV"],
            &["ETA_CL", "ETA_V"],
            &[("ETA_V", "TVV", MuTransform::Log)],
            3,
            &[("ETA_CL", vec!["TVCL1", "TVCL2", "TVCL2"])],
        );
        assert_eq!(
            get_mixture_mu_ref_pairs(&m),
            vec![
                MixtureMuRefPair {
                    eta_idx: 0,
                    theta_idx: vec![0, 1, 1],
                    transform: MuTransform::Log,
                },
                MixtureMuRefPair {
                    eta_idx: 1,
                    theta_idx: vec![2, 2, 2],
                    transform: MuTransform::Log,
                },
            ]
        );
    }

    /// A class-**shared** logit mu-ref inside a mixture model is a pair like any
    /// other: the shift is link-independent, and the packing check that decides
    /// whether it may run lives in the caller (`packed_scale_is_mu`), not here.
    /// Before the codex review of #1375 this builder filtered on
    /// `log_transformed()`, so `F = inv_logit(LOGIT_F + ETA_F)` silently dropped
    /// out of the closed form in every mixture fit (#918).
    #[test]
    fn mixture_mu_ref_pairs_include_a_class_shared_logit_anchor() {
        let m = model_with_mixture_mu_refs(
            &["TVCL1", "TVCL2", "LOGIT_F"],
            &["ETA_CL", "ETA_F"],
            &[("ETA_F", "LOGIT_F", MuTransform::Logit)],
            2,
            &[("ETA_CL", vec!["TVCL1", "TVCL2"])],
        );
        assert_eq!(
            get_mixture_mu_ref_pairs(&m),
            vec![
                MixtureMuRefPair {
                    eta_idx: 0,
                    theta_idx: vec![0, 1],
                    transform: MuTransform::Log,
                },
                MixtureMuRefPair {
                    eta_idx: 1,
                    theta_idx: vec![2, 2],
                    transform: MuTransform::Logit,
                },
            ]
        );
    }

    /// The packing gate both mixture estimators filter on. Same model, three
    /// packings: a logit anchor is eligible **only** identity-packed and a log
    /// anchor **only** log-packed, and each mismatch lands in the list whose
    /// advisory names the bound to change.
    ///
    /// This is the mutation target for the filter itself: written against
    /// `theta_packs_log_mask[t]` alone (the pre-review code), the logit row is
    /// classified backwards — the eligible case reads as a packing mismatch.
    #[test]
    fn classify_mixture_mu_ref_pairs_gates_each_transform_on_its_own_packing() {
        // theta 0 = LOGIT_F (logit anchor), theta 1 = TVV (log anchor).
        let m = model_with_mixture_mu_refs(
            &["LOGIT_F", "TVV"],
            &["ETA_F", "ETA_V"],
            &[
                ("ETA_F", "LOGIT_F", MuTransform::Logit),
                ("ETA_V", "TVV", MuTransform::Log),
            ],
            2,
            &[],
        );
        // Both declared the way the closed form wants: logit identity-packed,
        // lognormal log-packed.
        let split = classify_mixture_mu_ref_pairs(&m, &[false, true]);
        assert_eq!(
            split.eligible.iter().map(|p| p.eta_idx).collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert!(split.identity_packed_log.is_empty());
        assert!(split.log_packed_logit.is_empty());

        // Both declared the wrong way round: each drops, into its own list.
        let split = classify_mixture_mu_ref_pairs(&m, &[true, false]);
        assert!(split.eligible.is_empty());
        assert_eq!(split.log_packed_logit, vec![0]);
        assert_eq!(split.identity_packed_log, vec![1]);
    }

    /// A `MIXNUM`-switched pair is dropped whole when any one of its class
    /// thetas is mis-packed — the shift moves them together — but only the
    /// offending theta is named, since the others' bounds are already right.
    #[test]
    fn classify_mixture_mu_ref_pairs_drops_a_pair_on_one_mispacked_class_theta() {
        let m = model_with_mixture_mu_refs(
            &["TVCL1", "TVCL2"],
            &["ETA_CL"],
            &[],
            2,
            &[("ETA_CL", vec!["TVCL1", "TVCL2"])],
        );
        let split = classify_mixture_mu_ref_pairs(&m, &[true, false]);
        assert!(split.eligible.is_empty());
        assert_eq!(split.identity_packed_log, vec![1]);
    }

    /// The two forms a mixture anchor can never take, for the same reason as in
    /// the single-population classifier: no packing has their mu scale.
    #[test]
    fn mixture_mu_ref_pairs_exclude_probability_scale_logit() {
        let m = model_with_mixture_mu_refs(
            &["TVF"],
            &["ETA_F"],
            &[("ETA_F", "TVF", MuTransform::LogitProbability)],
            2,
            &[],
        );
        assert!(get_mixture_mu_ref_pairs(&m).is_empty());
    }

    #[test]
    fn mixture_mu_ref_pairs_drop_a_theta_claimed_by_a_second_eta() {
        // Two etas anchored to the same typical value: the shift loops apply one
        // update per pair, so keeping both would move TVCL twice in a single
        // iteration. The first eta keeps it; the second pair is dropped (#996
        // review).
        let m = model_with_mixture_mu_refs(
            &["TVCL", "TVV"],
            &["ETA_CL", "ETA_V"],
            &[
                ("ETA_CL", "TVCL", MuTransform::Log),
                ("ETA_V", "TVCL", MuTransform::Log),
            ],
            2,
            &[],
        );
        let pairs = get_mixture_mu_ref_pairs(&m);
        assert_eq!(
            pairs,
            vec![MixtureMuRefPair {
                eta_idx: 0,
                theta_idx: vec![0, 0],
                transform: MuTransform::Log,
            }]
        );
        // Every theta is claimed at most once across the whole pair set.
        let mut all: Vec<usize> = pairs
            .iter()
            .flat_map(|p| p.theta_idx.iter().copied())
            .collect();
        all.sort_unstable();
        let n = all.len();
        all.dedup();
        assert_eq!(all.len(), 1, "theta claimed once, got {n} slots");
    }

    #[test]
    fn mixture_mu_ref_pairs_drop_a_partially_overlapping_class_anchor_set() {
        // ETA_CL claims TVCL1/TVCL2; ETA_V's class anchors reuse TVCL2, so its
        // whole pair is dropped rather than double-shifting that theta.
        let m = model_with_mixture_mu_refs(
            &["TVCL1", "TVCL2", "TVV1"],
            &["ETA_CL", "ETA_V"],
            &[],
            2,
            &[
                ("ETA_CL", vec!["TVCL1", "TVCL2"]),
                ("ETA_V", vec!["TVV1", "TVCL2"]),
            ],
        );
        assert_eq!(
            get_mixture_mu_ref_pairs(&m),
            vec![MixtureMuRefPair {
                eta_idx: 0,
                theta_idx: vec![0, 1],
                transform: MuTransform::Log,
            }]
        );
    }

    #[test]
    fn mixture_mu_ref_pairs_exclude_additive_and_unknown_thetas() {
        let m = model_with_mixture_mu_refs(
            &["TVCL1", "TVCL2", "TVV"],
            &["ETA_CL", "ETA_V"],
            // additive: excluded, like the single-population `classify_mu_ref_pairs`
            &[("ETA_V", "TVV", MuTransform::Identity)],
            2,
            &[("ETA_CL", vec!["TVCL1", "MISSING"])],
        );
        assert!(get_mixture_mu_ref_pairs(&m).is_empty());
    }

    #[test]
    fn mixture_mu_ref_means_reduce_to_pooled_mean_when_theta_shared() {
        // Degenerate oracle: every class anchors on theta 0, so the weighted
        // per-class update must equal the classical pooled mean(η) exactly —
        // including bit-for-bit, since the accumulation order is the same.
        let etas = [0.3f64, -0.1, 0.7, -0.4];
        let resp = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let means = mixture_mu_ref_means(2, &[0, 0], &resp, |i, _c| etas[i]);
        let pooled = etas.iter().sum::<f64>() / etas.len() as f64;
        assert_eq!(means[0], Some(pooled));
        assert_eq!(means[1], None, "theta 1 is served by no class");
    }

    #[test]
    fn mixture_mu_ref_means_reduce_to_pooled_mean_under_soft_responsibilities() {
        // The same degeneracy under IMP-style fractional responsibilities:
        // every subject's weights sum to 1, so the shared-theta update is the
        // responsibility-weighted average of the per-class η means.
        let eta = |i: usize, c: usize| [[0.2f64, 0.4], [-0.6, -0.2]][i][c];
        let resp = vec![vec![0.75, 0.25], vec![0.4, 0.6]];
        let means = mixture_mu_ref_means(1, &[0, 0], &resp, eta);
        let expect = (0.75 * 0.2 + 0.25 * 0.4 + 0.4 * -0.6 + 0.6 * -0.2) / 2.0;
        assert!((means[0].unwrap() - expect).abs() < 1e-15);
    }

    #[test]
    fn mixture_mu_ref_means_split_by_class_membership() {
        // Hard classes: subjects 0,2 in class 1 (theta 0); 1,3 in class 2
        // (theta 1). Each theta moves by its own members' mean only.
        let etas = [0.3f64, -0.1, 0.7, -0.5];
        let resp = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let means = mixture_mu_ref_means(2, &[0, 1], &resp, |i, _c| etas[i]);
        assert!((means[0].unwrap() - 0.5).abs() < 1e-15);
        assert!((means[1].unwrap() - (-0.3)).abs() < 1e-15);
    }

    #[test]
    fn mixture_mu_ref_means_hold_theta_for_an_empty_class() {
        // Label switching: class 2 won zero subjects this iteration, so its
        // theta has no η mean and must be held rather than dragged.
        let etas = [0.2f64, 0.4];
        let resp = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
        let means = mixture_mu_ref_means(2, &[0, 1], &resp, |i, _c| etas[i]);
        assert!((means[0].unwrap() - 0.3).abs() < 1e-15);
        assert_eq!(means[1], None);
    }

    #[test]
    fn mixture_mu_ref_means_weight_classes_by_responsibility() {
        // Soft responsibilities with distinct class thetas: each theta gets the
        // responsibility-weighted mean of its own class's per-class η means.
        let eta = |i: usize, c: usize| [[0.1f64, 0.9], [0.3, 0.5]][i][c];
        let resp = vec![vec![0.8, 0.2], vec![0.25, 0.75]];
        let means = mixture_mu_ref_means(2, &[0, 1], &resp, eta);
        let e0 = (0.8 * 0.1 + 0.25 * 0.3) / (0.8 + 0.25);
        let e1 = (0.2 * 0.9 + 0.75 * 0.5) / (0.2 + 0.75);
        assert!((means[0].unwrap() - e0).abs() < 1e-15);
        assert!((means[1].unwrap() - e1).abs() < 1e-15);
    }

    #[test]
    fn floor_omega_diagonal_floors_free_entries_only() {
        // Three etas: a free near-zero diagonal (should be floored), a free
        // healthy diagonal (untouched), and a FIX-ed near-zero diagonal (kept).
        let mut omega = DMatrix::<f64>::zeros(3, 3);
        omega[(0, 0)] = 1e-9; // free, below floor → raised
        omega[(1, 1)] = 0.2; // free, above floor → unchanged
        omega[(2, 2)] = 1e-9; // FIX-ed, below floor → preserved
                              // an off-diagonal that must not be touched by the diagonal floor
        omega[(0, 1)] = 0.01;
        omega[(1, 0)] = 0.01;

        let omega_fixed = vec![false, false, true];
        floor_omega_diagonal(&mut omega, &omega_fixed, 1e-6);

        assert_eq!(
            omega[(0, 0)],
            1e-6,
            "free near-zero diagonal must be floored"
        );
        assert_eq!(
            omega[(1, 1)],
            0.2,
            "healthy free diagonal must be unchanged"
        );
        assert_eq!(
            omega[(2, 2)],
            1e-9,
            "FIX-ed diagonal must be left exactly as declared"
        );
        assert_eq!(omega[(0, 1)], 0.01, "off-diagonals must not be touched");
    }

    #[test]
    fn floor_omega_diagonal_treats_missing_fixed_flags_as_free() {
        // `omega_fixed` shorter than the matrix: missing entries default to free.
        let mut omega = DMatrix::<f64>::zeros(2, 2);
        omega[(0, 0)] = 1e-9;
        omega[(1, 1)] = 1e-9;
        floor_omega_diagonal(&mut omega, &[], 1e-6);
        assert_eq!(omega[(0, 0)], 1e-6);
        assert_eq!(omega[(1, 1)], 1e-6);
    }

    /// Lower bounds that make every theta log-packed (the usual PK case).
    fn positive_lowers(n: usize) -> Vec<f64> {
        vec![0.001; n]
    }

    #[test]
    fn get_mu_ref_pairs_empty_when_no_mu_refs() {
        let m = analytical_model(GradientMethod::Auto);
        assert!(get_mu_ref_pairs(&m, &positive_lowers(m.theta_names.len())).is_empty());
    }

    #[test]
    fn get_mu_ref_pairs_returns_log_transformed_pair() {
        let m = model_with_mu_refs(
            &["CL", "V"],
            &["ETA_CL", "ETA_V"],
            &[
                ("ETA_CL", "CL", MuTransform::Log),
                ("ETA_V", "V", MuTransform::Log),
            ],
        );
        let mut pairs = get_mu_ref_pairs(&m, &positive_lowers(2));
        pairs.sort();
        assert_eq!(pairs, vec![(0, 0), (1, 1)]);
    }

    #[test]
    fn get_mu_ref_pairs_excludes_additive_mu_refs() {
        // ETA_CL is lognormal (THETA*exp(ETA)) — included.
        // ETA_V is additive (THETA+ETA) — excluded because the gradient-step
        // chain rule used in run_saem assumes log-transformed parameters.
        let m = model_with_mu_refs(
            &["CL", "V"],
            &["ETA_CL", "ETA_V"],
            &[
                ("ETA_CL", "CL", MuTransform::Log),
                ("ETA_V", "V", MuTransform::Identity),
            ],
        );
        assert_eq!(get_mu_ref_pairs(&m, &positive_lowers(2)), vec![(0, 0)]);
    }

    #[test]
    fn get_mu_ref_pairs_skips_orphaned_theta() {
        // mu_ref points at a theta name that doesn't exist — silently skipped.
        let m = model_with_mu_refs(
            &["CL"],
            &["ETA_CL"],
            &[("ETA_CL", "MISSING", MuTransform::Log)],
        );
        assert!(get_mu_ref_pairs(&m, &positive_lowers(1)).is_empty());
    }

    /// #918: `F = inv_logit(LOGIT_F + ETA_F)` declares its theta on the logit
    /// scale (lower bound < 0 → identity-packed), so the packed value *is* the
    /// mu and the closed-form `packed += γ·mean(η)` update applies.
    #[test]
    fn get_mu_ref_pairs_includes_identity_packed_logit() {
        let m = model_with_mu_refs(
            &["LOGIT_F"],
            &["ETA_F"],
            &[("ETA_F", "LOGIT_F", MuTransform::Logit)],
        );
        assert_eq!(get_mu_ref_pairs(&m, &[-10.0]), vec![(0, 0)]);
    }

    /// A logit mu-ref whose theta happens to be log-packed (lower bound ≥ 0)
    /// has packed scale `log θ` but mu scale `θ` — the closed form does not
    /// apply, so it must fall through to the NLopt M-step.
    #[test]
    fn get_mu_ref_pairs_excludes_log_packed_logit() {
        let m = model_with_mu_refs(
            &["LOGIT_F"],
            &["ETA_F"],
            &[("ETA_F", "LOGIT_F", MuTransform::Logit)],
        );
        assert!(get_mu_ref_pairs(&m, &[0.0]).is_empty());
    }

    /// Mirror image: a lognormal mu-ref on an identity-packed theta (negative
    /// lower bound) has mu scale `log θ` but packed scale `θ` — also excluded.
    #[test]
    fn get_mu_ref_pairs_excludes_identity_packed_lognormal() {
        let m = model_with_mu_refs(&["CL"], &["ETA_CL"], &[("ETA_CL", "CL", MuTransform::Log)]);
        assert!(get_mu_ref_pairs(&m, &[-1.0]).is_empty());
    }

    /// `inv_logit(logit(THETA) + ETA)` puts THETA on the probability scale, so
    /// the mu is `logit(θ)` — a scale no packing produces. Never eligible.
    #[test]
    fn get_mu_ref_pairs_excludes_probability_scale_logit() {
        let m = model_with_mu_refs(
            &["F"],
            &["ETA_F"],
            &[("ETA_F", "F", MuTransform::LogitProbability)],
        );
        assert!(get_mu_ref_pairs(&m, &[0.001]).is_empty());
        assert!(get_mu_ref_pairs(&m, &[-1.0]).is_empty());
    }

    /// The dropped-for-packing lists behind the advisories: a lognormal anchor
    /// on an identity-packed theta lands in `identity_packed_log` (#996), a
    /// logit anchor on a log-packed theta in `log_packed_logit` (#918), and the
    /// eligible pairs are exactly the complement. Additive / probability-scale
    /// anchors appear in neither list — no bound change makes them eligible.
    #[test]
    fn classify_mu_ref_pairs_splits_eligible_from_packing_mismatches() {
        let m = model_with_mu_refs(
            &["CL", "V", "LOGIT_F", "LOGIT_G", "ADD", "PF"],
            &["ETA_CL", "ETA_V", "ETA_F", "ETA_G", "ETA_ADD", "ETA_PF"],
            &[
                ("ETA_CL", "CL", MuTransform::Log),
                ("ETA_V", "V", MuTransform::Log),
                ("ETA_F", "LOGIT_F", MuTransform::Logit),
                ("ETA_G", "LOGIT_G", MuTransform::Logit),
                ("ETA_ADD", "ADD", MuTransform::Identity),
                ("ETA_PF", "PF", MuTransform::LogitProbability),
            ],
        );
        // CL log-packed (ok), V identity-packed (dropped), LOGIT_F identity-packed
        // (ok), LOGIT_G log-packed (dropped), ADD / PF never eligible.
        let lowers = [0.001, -5.0, -10.0, 0.0, -1.0, 0.001];
        let split = classify_mu_ref_pairs(&m, &lowers);
        let mut eligible = split.eligible.clone();
        eligible.sort();
        assert_eq!(eligible, vec![(0, 0), (2, 2)]);
        assert_eq!(split.identity_packed_log, vec![1]);
        assert_eq!(split.log_packed_logit, vec![3]);
        assert_eq!(get_mu_ref_pairs(&m, &lowers), split.eligible);
    }

    /// Two etas anchored to one theta must not list that theta twice.
    #[test]
    fn classify_mu_ref_pairs_dedups_dropped_thetas() {
        let m = model_with_mu_refs(
            &["TVP"],
            &["ETA_A", "ETA_B"],
            &[
                ("ETA_A", "TVP", MuTransform::Log),
                ("ETA_B", "TVP", MuTransform::Log),
            ],
        );
        let split = classify_mu_ref_pairs(&m, &[-1.0]);
        assert!(split.eligible.is_empty());
        assert_eq!(split.identity_packed_log, vec![0]);
    }

    /// Two logit etas anchored to one theta: the closed form shifts a packed
    /// theta by *one* eta mean and pins it, so keeping either pair would move
    /// `LOGIT_F` by a quantity that is not the joint maximiser and leave the
    /// other eta un-recentred. Both pairs are dropped and the theta is named.
    ///
    /// Without the `shared_theta` filter this returns *two* eligible pairs on
    /// theta 0, and `run_saem`/`run_mcem` apply `θ += mean(η_1)` followed by
    /// `θ += mean(η_2)` in the same iteration (codex review of #1375).
    #[test]
    fn classify_mu_ref_pairs_drops_a_theta_anchoring_two_logit_etas() {
        let m = model_with_mu_refs(
            &["LOGIT_F"],
            &["ETA_F1", "ETA_F2"],
            &[
                ("ETA_F1", "LOGIT_F", MuTransform::Logit),
                ("ETA_F2", "LOGIT_F", MuTransform::Logit),
            ],
        );
        // Identity-packed, so packing is *not* what makes these ineligible.
        let split = classify_mu_ref_pairs(&m, &[-10.0]);
        assert!(
            split.eligible.is_empty(),
            "a shared anchor has no single closed-form shift, got {:?}",
            split.eligible
        );
        assert_eq!(split.shared_theta, vec![0]);
        assert!(split.log_packed_logit.is_empty());
    }

    /// The shared-anchor drop must not open a door for a #619 covariate group.
    /// `resolve_covariate_mu_groups` declines a group whose theta another eta
    /// anchors; feeding it `eligible` alone would hide exactly the thetas that
    /// are *most* contested, so the dropped pairs are handed back through
    /// `mu_ref_pairs_for_cov_groups`.
    #[test]
    fn mu_ref_pairs_for_cov_groups_keeps_the_shared_anchor_pairs() {
        let m = model_with_mu_refs(
            &["TVP"],
            &["ETA_A", "ETA_B"],
            &[
                ("ETA_A", "TVP", MuTransform::Log),
                ("ETA_B", "TVP", MuTransform::Log),
            ],
        );
        let split = classify_mu_ref_pairs(&m, &positive_lowers(1));
        assert!(split.eligible.is_empty(), "shared anchor takes no shift");
        assert_eq!(mu_ref_pairs_for_cov_groups(&split), vec![(0, 0), (0, 1)]);
    }

    /// …while a theta dropped for a *packing* mismatch is not a conflict: the
    /// group step does not work in the packed scale, so it may move it.
    #[test]
    fn mu_ref_pairs_for_cov_groups_omits_a_packing_dropped_theta() {
        let m = model_with_mu_refs(
            &["TVCL"],
            &["ETA_CL"],
            &[("ETA_CL", "TVCL", MuTransform::Log)],
        );
        let split = classify_mu_ref_pairs(&m, &[-1.0]);
        assert_eq!(split.identity_packed_log, vec![0]);
        assert!(mu_ref_pairs_for_cov_groups(&split).is_empty());
    }

    /// The lognormal spelling of the same hazard (`CL = TVP*exp(ETA_CL)`,
    /// `V = TVP*exp(ETA_V)`), and the control that an *unshared* theta in the
    /// same model keeps its pair — so the filter drops the shared anchor, not
    /// the model.
    #[test]
    fn classify_mu_ref_pairs_drops_a_shared_log_anchor_but_keeps_the_others() {
        let m = model_with_mu_refs(
            &["TVP", "TVV"],
            &["ETA_CL", "ETA_V", "ETA_OTHER"],
            &[
                ("ETA_CL", "TVP", MuTransform::Log),
                ("ETA_V", "TVP", MuTransform::Log),
                ("ETA_OTHER", "TVV", MuTransform::Log),
            ],
        );
        let split = classify_mu_ref_pairs(&m, &positive_lowers(2));
        assert_eq!(split.eligible, vec![(1, 2)]);
        assert_eq!(split.shared_theta, vec![0]);
        assert!(split.identity_packed_log.is_empty());
    }

    /// The shared-anchor rule counts sharing over **every declared anchor**,
    /// not just the eligible ones (#918 review).
    ///
    /// `CL = TVP*exp(ETA_CL)` is log-packed `Log`, so eligible; `V = TVP +
    /// ETA_V` is `Identity`, which is never eligible whatever the packing.
    /// Scanning `eligible` alone sees `TVP` claimed once, shifts it by
    /// `mean(η_CL)` and pins it, while `ETA_V` is never re-centred for that
    /// delta — the pinned partial shift the rule exists to prevent.
    /// `ETA_OTHER` on `TVV` is the straddle: an unshared anchor in the same
    /// model must keep its pair, so a fix that widened the drop instead of the
    /// detection fails here.
    #[test]
    fn classify_mu_ref_pairs_drops_an_anchor_shared_with_an_ineligible_eta() {
        let m = model_with_mu_refs(
            &["TVP", "TVV"],
            &["ETA_CL", "ETA_V", "ETA_OTHER"],
            &[
                ("ETA_CL", "TVP", MuTransform::Log),
                ("ETA_V", "TVP", MuTransform::Identity),
                ("ETA_OTHER", "TVV", MuTransform::Log),
            ],
        );
        let split = classify_mu_ref_pairs(&m, &positive_lowers(2));
        assert_eq!(
            split.eligible,
            vec![(1, 2)],
            "only the unshared anchor keeps its closed-form shift"
        );
        assert_eq!(split.shared_pairs, vec![(0, 0)]);
        assert_eq!(split.shared_theta, vec![0]);
        // `Identity` is not a packing problem, so it earns no packing advisory.
        assert!(split.identity_packed_log.is_empty());
        assert!(split.log_packed_logit.is_empty());
        // The dropped pair still counts as a conflict for a #619 group.
        assert_eq!(mu_ref_pairs_for_cov_groups(&split), vec![(1, 2), (0, 0)]);
    }

    /// …but two *ineligible* anchors on one theta share nothing the closed
    /// form would have moved, so they are not named as a shared-anchor drop —
    /// the packing advisory already explains them, and a second, unrelated
    /// message would send the user after the wrong fix.
    #[test]
    fn classify_mu_ref_pairs_does_not_name_a_theta_only_ineligible_etas_share() {
        let m = model_with_mu_refs(
            &["TVP"],
            &["ETA_A", "ETA_B"],
            &[
                ("ETA_A", "TVP", MuTransform::Log),
                ("ETA_B", "TVP", MuTransform::Log),
            ],
        );
        // Identity-packed, so *both* anchors are ineligible for packing
        // reasons before sharing is even considered.
        let split = classify_mu_ref_pairs(&m, &[-1.0]);
        assert!(split.eligible.is_empty());
        assert!(split.shared_pairs.is_empty());
        assert!(split.shared_theta.is_empty());
        assert_eq!(split.identity_packed_log, vec![0]);
    }

    /// `packed_scale_is_mu` is the one condition the link-independent shift
    /// needs; the table is the contract every caller (single-population and
    /// mixture) filters on.
    #[test]
    fn packed_scale_is_mu_matches_the_packing_table() {
        assert!(packed_scale_is_mu(MuTransform::Log, true));
        assert!(!packed_scale_is_mu(MuTransform::Log, false));
        assert!(packed_scale_is_mu(MuTransform::Logit, false));
        assert!(!packed_scale_is_mu(MuTransform::Logit, true));
        for packs_log in [true, false] {
            assert!(!packed_scale_is_mu(MuTransform::Identity, packs_log));
            assert!(!packed_scale_is_mu(
                MuTransform::LogitProbability,
                packs_log
            ));
        }
    }

    // ---- Regression tests for the three SAEM correctness bugs ----

    /// Bug 1 (diagonal): `from_diagonal` produces a free_mask that marks only
    /// diagonal entries free. The SAEM M-step uses this mask to zero
    /// SA-accumulated off-diagonals, preventing the rank-deficient Ω failure.
    #[test]
    fn diagonal_omega_free_mask_has_no_off_diagonals() {
        let omega = OmegaMatrix::from_diagonal(&[0.1, 0.2], vec!["ETA_CL".into(), "ETA_V".into()]);
        assert!(omega.free_mask[(0, 0)]);
        assert!(omega.free_mask[(1, 1)]);
        assert!(!omega.free_mask[(0, 1)]);
        assert!(!omega.free_mask[(1, 0)]);
    }

    /// Bug 1 (mixed structure): `from_matrix_with_mask` preserves an explicit
    /// mask that marks cross-block entries as structural zeros. This is the
    /// case that the `diagonal` flag alone cannot express (one standalone eta
    /// + one block_omega pair → diagonal=false, but cross entries are zero).
    #[test]
    fn mixed_omega_free_mask_zeros_cross_block_entries() {
        // Three etas: ETA_CL(0) and ETA_V(1) in a block; ETA_KA(2) standalone.
        let mut matrix = nalgebra::DMatrix::zeros(3, 3);
        matrix[(0, 0)] = 0.1;
        matrix[(1, 1)] = 0.2;
        matrix[(2, 2)] = 0.1;
        matrix[(0, 1)] = 0.01;
        matrix[(1, 0)] = 0.01;

        let mut free_mask = nalgebra::DMatrix::from_element(3, 3, false);
        free_mask[(0, 0)] = true;
        free_mask[(1, 1)] = true;
        free_mask[(2, 2)] = true;
        free_mask[(0, 1)] = true; // within CL-V block
        free_mask[(1, 0)] = true;

        let names = vec!["ETA_CL".into(), "ETA_V".into(), "ETA_KA".into()];
        let omega = OmegaMatrix::from_matrix_with_mask(matrix, names, false, free_mask);

        assert!(omega.free_mask[(0, 1)]);
        assert!(omega.free_mask[(1, 0)]);
        assert!(!omega.free_mask[(2, 0)]);
        assert!(!omega.free_mask[(0, 2)]);
        assert!(!omega.free_mask[(2, 1)]);
        assert!(!omega.free_mask[(1, 2)]);
    }

    /// Bug 2: `mh_steps` is a symmetric random walk — proposals are
    /// `eta_prop = eta + step·perturbation`, not `mu_k + step·perturbation`.
    ///
    /// Discriminator: with `step_scale = 0` the new kernel proposes exactly
    /// the current eta, so the chain cannot move regardless of the data.
    /// The pre-fix `mu_k`-centred kernel proposed exactly `mu_k` (= log TVCL),
    /// so a starting eta far from `mu_k` would either jump to `mu_k`
    /// whenever the proposal looked better, or oscillate. We pick a starting
    /// eta of 5.0 with TVCL=1 (mu_k=0): the simulated observation lives near
    /// the data-generating eta=0 region, so individual_nll(eta=0) is much
    /// lower than individual_nll(eta=5), meaning the broken kernel would
    /// accept the eta=0 proposal with probability ≈1 on the first step.
    /// The new kernel must leave eta at exactly 5.0.
    #[test]
    fn mh_steps_random_walk_uses_current_eta_not_mu_k() {
        use crate::stats::likelihood::individual_nll;
        use crate::types::{DoseEvent, SigmaVector};
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        use std::collections::HashMap;

        let model = analytical_model(GradientMethod::Auto);
        let subj = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0],
            obs_raw_times: Vec::new(),
            observations: vec![1.0],
            obs_cmts: vec![1],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            reset_covariates: Vec::new(),
            cens: vec![0],
            occasions: vec![],
            obs_l2: Vec::new(),
            dose_occasions: vec![],
            reset_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
        };
        let omega = OmegaMatrix::from_diagonal(&[1.0], vec!["ETA_CL".into()]);
        let sigma = SigmaVector {
            values: vec![1.0],
            names: vec!["PROP".into()],
        };
        let theta = vec![1.0]; // mu_k = log(1) = 0
        let mut eta = vec![5.0_f64]; // far from mu_k
        let nll_start = individual_nll(&model, &subj, &theta, &eta, &omega, &sigma.values);
        let mut rng = StdRng::seed_from_u64(42);

        let mut pk_scratch = MhScratch::default();
        pk_scratch.begin_subject(&model, &subj, &theta, eta.len());
        mh_steps(
            &mut eta,
            nll_start,
            &subj,
            &model,
            &theta,
            &omega,
            &sigma.values,
            0.0, // zero perturbation: random walk MUST stay put exactly
            None,
            &mut rng,
            100,
            &mut pk_scratch,
            None,
            None,
        );

        // Random walk with step=0: every proposal == current eta, accepted as
        // identity. The pre-fix kernel would have proposed mu_k=0 every step
        // and accepted it (lower nll than eta=5), driving eta to 0.
        assert_eq!(
            eta[0], 5.0,
            "eta moved despite step_scale=0 — proposals were re-centred on mu_k"
        );
    }

    /// Bug 3 / closed-form M-step: a synthetic SAEM run with mu_referencing=true
    /// and mean(eta) ≠ 0 must move log_theta in the right direction *without*
    /// pinning at the bound. We exercise the closed-form formula directly:
    /// `log_theta_new = log_theta_old + γ · mean(eta)`.
    #[test]
    fn closed_form_mu_ref_mstep_is_bounded_and_signed_correctly() {
        // Simulate post-MH state: 5 subjects, eta_mean = +0.4 (population CL
        // is higher than current TVCL), gamma = 1.0 (exploration step).
        let etas: Vec<Vec<f64>> = vec![vec![0.5], vec![0.3], vec![0.4], vec![0.6], vec![0.2]];
        let n = etas.len() as f64;
        let mean_eta: f64 = etas.iter().map(|e| e[0]).sum::<f64>() / n;
        assert!((mean_eta - 0.4).abs() < 1e-12);

        let gamma = 1.0;
        let log_theta_old = 0.0_f64; // TVCL = 1.0
        let log_theta_new = log_theta_old + gamma * mean_eta;
        // log_theta moved by exactly mean(eta), independent of N.  This is the
        // property that the broken gradient step (γ · Σ ∂obs_nll/∂eta) lacked:
        // its update scaled with N and pinned thetas at bounds for moderate N.
        assert!((log_theta_new - 0.4).abs() < 1e-12);

        // After re-centring etas by gamma*mean, mean(eta) = 0.
        let mut etas_recentered = etas.clone();
        for e in etas_recentered.iter_mut() {
            e[0] -= gamma * mean_eta;
        }
        let new_mean: f64 = etas_recentered.iter().map(|e| e[0]).sum::<f64>() / n;
        assert!(new_mean.abs() < 1e-12);
    }

    /// Bug 3 follow-up: the broken gradient step (γ · Σᵢ ∂obs_nll/∂eta) is no
    /// longer in the code path. The closed-form `log_theta += γ · mean(η)` is
    /// what runs when mu_referencing=true. Pair detection is unchanged.
    #[test]
    fn mu_ref_pair_detection_drives_closed_form_branch() {
        let m = model_with_mu_refs(
            &["CL", "V"],
            &["ETA_CL", "ETA_V"],
            &[
                ("ETA_CL", "CL", MuTransform::Log),
                ("ETA_V", "V", MuTransform::Log),
            ],
        );
        let pairs = get_mu_ref_pairs(&m, &positive_lowers(2));
        assert_eq!(pairs.len(), 2);
        // The closed-form branch is taken iff `options.mu_referencing` AND
        // `!pairs.is_empty()`.  Both conditions are tested via the public API
        // in api::iov_integration::test_iov_foce_mu_referencing_on; this unit
        // test pins the precondition (pair detection still produces work).
    }

    /// A pre-cancelled `CancelFlag` makes the SAEM main loop break at the
    /// first iteration and `run_saem` must return `Err("cancelled by user")`
    /// without entering the post-loop "Computing final EBEs and OFV..." block
    /// (which iterates over every subject and is what makes a cancelled run
    /// feel like it isn't aborting).
    #[test]
    fn cancelled_run_returns_err_and_skips_final_ebe() {
        use crate::cancel::CancelFlag;
        use crate::types::{DoseEvent, FitOptions, Population};
        use std::collections::HashMap;

        let model = analytical_model(GradientMethod::Auto);
        let subj = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 2.0],
            obs_raw_times: Vec::new(),
            observations: vec![1.0, 0.5],
            obs_cmts: vec![1, 1],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            reset_covariates: Vec::new(),
            cens: vec![0, 0],
            occasions: vec![],
            obs_l2: Vec::new(),
            dose_occasions: vec![],
            reset_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
        };
        let population = Population {
            subjects: vec![subj],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let flag = CancelFlag::new();
        flag.cancel(); // pre-cancel: loop breaks at iteration 1

        let mut opts = FitOptions::default();
        opts.verbose = false;
        opts.run_covariance_step = false;
        opts.cancel = Some(flag);

        match run_saem(&model, &population, &model.default_params, &opts) {
            Err(msg) => assert!(
                msg.contains("cancelled by user"),
                "unexpected error message: {msg}"
            ),
            Ok(_) => panic!("pre-cancelled SAEM must return Err, not Ok"),
        }
    }

    /// Per-theta packing must round-trip values identically for both log-packed
    /// (`theta_lower >= 0`) and identity-packed (`theta_lower < 0`) thetas. SAEM
    /// uses its own pack/unpack closures inside the M-step, so this exercises
    /// the same math the closures rely on (`theta_packs_log` from
    /// parameterization plus the `if mask[i] { ln/exp } else { identity }`
    /// branches in `theta_sigma_mstep_light`).
    #[test]
    fn saem_pack_unpack_handles_negative_lower_bound() {
        use crate::estimation::parameterization::theta_packs_log;

        // Mix: CL (lower=0), V (lower=0.001), THETA_AGE_CL (lower=-1).
        let lowers: [f64; 3] = [0.0, 0.001, -1.0];
        let values: [f64; 3] = [5.0, 20.0, -0.01];
        let mask: Vec<bool> = lowers.iter().map(|&lo| theta_packs_log(lo)).collect();
        assert_eq!(mask, vec![true, true, false]);

        // Forward: simulate the SAEM init-pack construction (lines ~444–451 of
        // run_saem: log when log-packed, identity when identity-packed).
        let packed: Vec<f64> = values
            .iter()
            .zip(mask.iter())
            .map(|(&v, &log_pack)| if log_pack { v.max(1e-10).ln() } else { v })
            .collect();

        // Reverse: the M-step `unpack_thetas` closure.
        let unpacked: Vec<f64> = packed
            .iter()
            .zip(mask.iter())
            .map(|(&p, &log_pack)| if log_pack { p.exp() } else { p })
            .collect();

        for (orig, round) in values.iter().zip(unpacked.iter()) {
            assert!(
                (orig - round).abs() < 1e-12,
                "saem pack/unpack should round-trip: {orig} != {round}"
            );
        }
        // The identity-packed theta carries a negative value through —
        // pre-fix, this was clamped to 1e-10 by the log path.
        assert!(unpacked[2] < 0.0);
    }

    // ── IOV kappa MH: rejection restores kappa ─────────────────────────────

    /// With `step_scale = 0` the proposal is always identical to the current
    /// kappa, so ΔH = 0 and every step is accepted.  The kappa values must
    /// not change (proposal == current).
    #[test]
    fn mh_kappa_zero_step_always_accepts_and_preserves_kappa() {
        use crate::types::test_helpers::analytical_model;
        use std::collections::HashMap;

        let model = analytical_model(GradientMethod::Auto);

        // One subject with 2 occasions (occasions = [1,1,2,2]).
        let subject = Subject {
            id: "S1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 2.0, 3.0, 4.0],
            obs_raw_times: Vec::new(),
            observations: vec![50.0, 40.0, 35.0, 28.0],
            obs_cmts: vec![1; 4],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            reset_covariates: Vec::new(),
            cens: vec![0; 4],
            occasions: vec![1u32, 1, 2, 2],
            obs_l2: Vec::new(),
            dose_occasions: vec![1u32],
            reset_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
        };

        let omega_bsv = OmegaMatrix::from_diagonal(&[0.09], vec!["ETA_CL".into()]);
        let omega_iov = OmegaMatrix::from_diagonal(&[0.04], vec!["KAPPA_CL".into()]);
        let theta = vec![5.0, 50.0];
        let eta = vec![0.0];
        let sigma = vec![0.1];
        // Two occasions, each with one kappa.
        let mut kappas = vec![vec![0.2_f64], vec![-0.1_f64]];
        let kappas_before = kappas.clone();

        let nll0 = individual_nll_iov(
            &model,
            &subject,
            &theta,
            &eta,
            &kappas,
            &omega_bsv,
            Some(&omega_iov),
            &sigma,
        );

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let (n_acc, n_prop, nll_after) = mh_kappa_steps(
            &mut kappas,
            nll0,
            &subject,
            &model,
            &theta,
            &eta,
            &omega_bsv,
            &omega_iov,
            &sigma,
            0.0, // step_scale = 0 → proposal == current → always accepted
            &mut rng,
            None,
            &mut EventPkParams::default(),
        );

        // With step_scale=0 every occasion proposal is accepted (2 occasions).
        assert_eq!(n_prop, 2, "expected 2 proposals (one per occasion)");
        assert_eq!(n_acc, 2, "step_scale=0: all proposals must be accepted");
        // Kappa values must be unchanged (proposal == current point).
        assert_eq!(
            kappas, kappas_before,
            "kappas must not change with step_scale=0"
        );
        // NLL must not change either.
        assert!(
            (nll_after - nll0).abs() < 1e-10,
            "NLL must not change with step_scale=0"
        );
    }

    // ── κ-phase parallel schedule is bit-identical (#1346 review, 2026-09-11) ──

    /// Mixture **and** IOV together — the two per-subject contexts the κ phase's
    /// map-then-apply parallelisation (#1344 item 5) has to reconstruct inside
    /// each `rayon` worker: the per-occasion kappa MH itself, and the mixture
    /// class guard that must route it into the subject's drawn class before
    /// proposing κ. Neither the existing serial-IOV nor the existing
    /// mixture-only SAEM fixtures exercise both at once.
    fn mix_iov_model() -> CompiledModel {
        let src = r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(3.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09 FIX
  kappa KAPPA_CL ~ 0.03 FIX
  sigma EPS ~ 0.04 FIX

[mixture]
  nsub = 2
  logit(1) = MIXL

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL + KAPPA_CL) else TVCL2 * exp(ETA_CL + KAPPA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";
        crate::parser::model_parser::parse_model_string(src).expect("mixture+IOV model parses")
    }

    /// Four subjects, two per class, two occasions each with a dose at the
    /// occasion boundary — one `kappas[i]` entry per occasion, and a class per
    /// subject, exactly what the κ phase's per-`i` guard branches on.
    fn mix_iov_pop() -> Population {
        use std::collections::HashMap;
        let subjects = (0..4)
            .map(|i| {
                let cl: f64 = if i < 2 { 1.0 } else { 3.0 };
                let obs_times = vec![0.5, 1.0, 2.0, 4.0];
                let observations = obs_times
                    .iter()
                    .enumerate()
                    .map(|(j, &t)| {
                        (10.0 * (-(cl / 10.0) * t).exp()) * (1.0 + 0.02 * ((i + j) as f64).sin())
                    })
                    .collect();
                Subject {
                    id: (i + 1).to_string(),
                    doses: vec![
                        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                        DoseEvent::new(2.0, 100.0, 1, 0.0, false, 0.0),
                    ],
                    obs_times,
                    obs_raw_times: Vec::new(),
                    observations,
                    obs_cmts: vec![1, 1, 1, 1],
                    covariates: HashMap::new(),
                    dose_covariates: Vec::new(),
                    obs_covariates: Vec::new(),
                    pk_only_times: Vec::new(),
                    pk_only_covariates: Vec::new(),
                    reset_times: Vec::new(),
                    reset_covariates: Vec::new(),
                    cens: vec![0, 0, 0, 0],
                    occasions: vec![1, 1, 2, 2],
                    obs_l2: Vec::new(),
                    dose_occasions: vec![1, 2],
                    reset_occasions: Vec::new(),
                    fremtype: Vec::new(),
                    obs_records: vec![],
                }
            })
            .collect();
        Population {
            subjects,
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        }
    }

    /// The regression coverage the PR #1346 review asked for: the κ phase's
    /// parallelisation is claimed to be **bit-identical** to the serial loop it
    /// replaced (commit `055e6e2f`) because every access is `[i]`-disjoint and
    /// every subject's RNG is seeded from `(master_seed, k, i)` alone, never from
    /// which worker or in what order subjects finish. That claim is a scheduling
    /// property, not a numerical one, so the oracle for it is the *same* SAEM
    /// run under a different `rayon` worker count — exactly what every other
    /// parallel reduction in this file already pins against thread-count
    /// dependence (`obs_nll_sum_iov` and friends, #703). A defect that let a
    /// worker read or write another subject's `kappas[i]`, or reseeded from
    /// anything thread-local, would move `state.kappas` without necessarily
    /// moving `ofv` — so `kappas` is the field that actually exercises the claim,
    /// not just a plausible one to also check.
    #[test]
    fn kappa_phase_parallel_schedule_is_bit_identical() {
        let model = mix_iov_model();
        let population = mix_iov_pop();
        let opts = FitOptions {
            saem_n_exploration: 3,
            saem_n_convergence: 2,
            saem_n_mh_steps: 4,
            saem_omega_burnin: 0,
            saem_seed: Some(20260911),
            run_covariance_step: false,
            verbose: false,
            ..FitOptions::default()
        };

        let run = |width: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(width)
                .stack_size(crate::FIT_RAYON_STACK_SIZE)
                .build()
                .expect("thread pool build");
            pool.install(|| {
                run_saem(&model, &population, &model.default_params, &opts)
                    .expect("SAEM run must succeed")
            })
        };

        let one_worker = run(1);
        let four_workers = run(4);

        assert_eq!(
            one_worker.ofv.to_bits(),
            four_workers.ofv.to_bits(),
            "final OFV must not depend on the rayon worker count"
        );
        assert_eq!(
            one_worker.params.theta.len(),
            four_workers.params.theta.len()
        );
        for (m, (a, b)) in one_worker
            .params
            .theta
            .iter()
            .zip(four_workers.params.theta.iter())
            .enumerate()
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "theta[{m}] must not depend on the rayon worker count"
            );
        }

        assert_eq!(one_worker.kappas.len(), four_workers.kappas.len());
        for (i, (occs_a, occs_b)) in one_worker
            .kappas
            .iter()
            .zip(four_workers.kappas.iter())
            .enumerate()
        {
            assert_eq!(
                occs_a.len(),
                occs_b.len(),
                "subject {i}: occasion count must not depend on the rayon worker count"
            );
            for (occ, (ka, kb)) in occs_a.iter().zip(occs_b.iter()).enumerate() {
                assert_eq!(
                    ka.as_slice()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    kb.as_slice()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    "subject {i} occasion {occ}: kappa must not depend on the rayon worker count"
                );
            }
        }
    }

    // ── IOV omega analytic update formula ──────────────────────────────────

    /// The analytic update `(1/N_occ) Σᵢ Σₖ κᵢₖ κᵢₖᵀ` for a 1-dimensional
    /// omega_iov with two subjects, two occasions each, and known kappas must
    /// match the hand-computed value exactly.
    #[test]
    fn iov_omega_analytic_update_matches_hand_computation() {
        // Subject 1: occ1 = [0.2], occ2 = [-0.1]
        // Subject 2: occ1 = [0.3], occ2 = [-0.2]
        // Hand sum = 0.2² + 0.1² + 0.3² + 0.2² = 0.04 + 0.01 + 0.09 + 0.04 = 0.18
        // Divided by 4 occasions → 0.045
        let kappas: Vec<Vec<Vec<f64>>> =
            vec![vec![vec![0.2], vec![-0.1]], vec![vec![0.3], vec![-0.2]]];
        let n_kappa = 1_usize;
        let mut kappa_outer = DMatrix::zeros(n_kappa, n_kappa);
        let mut n_total_occ = 0_usize;
        for kappas_i in &kappas {
            for kap in kappas_i {
                let kv = DVector::from_column_slice(kap);
                kappa_outer += &kv * kv.transpose();
                n_total_occ += 1;
            }
        }
        kappa_outer /= n_total_occ as f64;
        let expected = (0.04 + 0.01 + 0.09 + 0.04) / 4.0;
        assert!(
            (kappa_outer[(0, 0)] - expected).abs() < 1e-12,
            "IOV omega analytic update: got {:.6e}, expected {:.6e}",
            kappa_outer[(0, 0)],
            expected
        );
    }

    /// SAEM with the optimizer trace active must emit the per-parameter `val:*`
    /// columns and write `NA` for every `grad:*` column (SAEM has no OFV
    /// gradient). This is the fast test that registers PR coverage for the SAEM
    /// trace call site (#640), which is otherwise reached only by slow fits.
    #[test]
    fn saem_trace_emits_value_columns_with_na_grads() {
        use crate::types::{DoseEvent, FitOptions, Population};
        use std::collections::HashMap;

        let model = analytical_model(GradientMethod::Auto);
        let subj = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 2.0],
            obs_raw_times: Vec::new(),
            observations: vec![1.0, 0.5],
            obs_cmts: vec![1, 1],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            reset_covariates: Vec::new(),
            cens: vec![0, 0],
            occasions: vec![],
            obs_l2: Vec::new(),
            dose_occasions: vec![],
            reset_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
        };
        let population = Population {
            subjects: vec![subj],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let coord_names =
            crate::estimation::parameterization::coordinate_names(&model.default_params);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = format!(
            "/tmp/ferx_trace_saem_param_{}_{}.csv",
            std::process::id(),
            nanos
        );
        crate::estimation::trace::init(path.clone(), &coord_names).unwrap();

        let opts = FitOptions {
            saem_n_exploration: 2,
            saem_n_convergence: 0,
            run_covariance_step: false,
            verbose: false,
            ..FitOptions::default()
        };
        let _ = run_saem(&model, &population, &model.default_params, &opts);
        crate::estimation::trace::finish();

        let contents = std::fs::read_to_string(&path).unwrap();
        let mut lines = contents.lines();
        let header: Vec<String> = lines.next().unwrap().split(',').map(String::from).collect();
        assert!(header.iter().any(|c| c.starts_with("val:")));
        let n_coords = coord_names.len();
        let fixed = 17;
        let mut rows = 0;
        for row in lines {
            let c: Vec<String> = row.split(',').map(String::from).collect();
            assert_eq!(c.len(), fixed + 2 * n_coords, "SAEM row column count");
            // Every value column finite; every gradient column NA.
            for i in fixed..(fixed + n_coords) {
                assert!(c[i].parse::<f64>().is_ok(), "val column must be finite");
            }
            for i in (fixed + n_coords)..(fixed + 2 * n_coords) {
                assert_eq!(c[i], "NA", "SAEM grad column must be NA");
            }
            rows += 1;
        }
        assert!(rows >= 1, "expected at least one SAEM trace row");
        std::fs::remove_file(&path).ok();
    }

    // ---- #895: iiv_on_ruv sigma-ridge cap ----

    #[test]
    fn ruv_sigma_caps_none_without_iiv_on_ruv() {
        // A model with no iiv_on_ruv carries no cap on any sigma, so the M-step
        // sigma trajectory is left exactly as the un-capped code produced it.
        let caps = compute_ruv_sigma_caps(
            false,
            None,
            &[(-1.75_f64), 0.0],
            &[5.0, 5.0],
            &[false, false],
        );
        assert_eq!(caps, vec![None, None]);
    }

    #[test]
    fn ruv_sigma_caps_bounds_free_residual_sigma_growth() {
        // iiv_on_ruv active: the free residual sigma gets a cap of
        // log(σ₀) + SAEM_RUV_SIGMA_LN_GROWTH (well below the e⁵ ceiling).
        let ln_s0 = 0.1738_f64.ln(); // ≈ -1.75
        let caps = compute_ruv_sigma_caps(true, None, &[ln_s0], &[5.0], &[false]);
        assert_eq!(caps.len(), 1);
        let cap = caps[0].expect("free residual sigma must carry a cap");
        assert!((cap - (ln_s0 + SAEM_RUV_SIGMA_LN_GROWTH)).abs() < 1e-12);
        // e³ ≈ 20× growth in SD, and comfortably under the e⁵ ceiling.
        assert!(cap < 5.0);
        assert!((cap.exp() / 0.1738 - SAEM_RUV_SIGMA_LN_GROWTH.exp()).abs() < 1e-9);
    }

    #[test]
    fn ruv_sigma_caps_skip_fixed_and_frem_covariate_sigma() {
        // sigma[0] = free PK residual (capped), sigma[1] = FREM EPSCOV (skipped),
        // sigma[2] = FIXed (skipped).
        let caps = compute_ruv_sigma_caps(
            true,
            Some(1),
            &[0.0, -6.0, -1.0],
            &[5.0, 5.0, 5.0],
            &[false, true, true],
        );
        assert!(caps[0].is_some());
        assert_eq!(caps[1], None, "FREM EPSCOV must not be capped");
        assert_eq!(caps[2], None, "FIXed sigma must not be capped");
    }

    #[test]
    fn ruv_sigma_caps_defer_to_tighter_user_upper_bound() {
        // If the model's own upper bound is tighter than log σ₀ + growth, NLopt
        // already enforces it, so no RUV growth cap is carried — a σ converging to
        // its own bound must not be mis-flagged as hitting the iiv_on_ruv cap (#903).
        let caps = compute_ruv_sigma_caps(true, None, &[0.0], &[1.0], &[false]);
        assert_eq!(caps[0], None);
        // A growth cap strictly below the upper bound is still carried.
        let caps = compute_ruv_sigma_caps(true, None, &[0.0], &[5.0], &[false]);
        assert_eq!(caps[0], Some(SAEM_RUV_SIGMA_LN_GROWTH));
    }

    // ---- #904: iiv_on_ruv eta re-centering ----

    #[test]
    fn recenter_ruv_eta_zeroes_mean_and_preserves_residual_variance() {
        // η_RUV at index 1, with a non-zero mean; two etas per subject.
        let mut etas = vec![vec![0.3, 0.8], vec![-0.1, -0.4], vec![0.2, 0.2]];
        let kr = 1;
        // Per-subject residual variance before: σ²·exp(2·η_RUV) (σ = sd = 0.2).
        let sd0 = 0.2_f64;
        let var_before: Vec<f64> = etas
            .iter()
            .map(|e| sd0 * sd0 * (2.0_f64 * e[kr]).exp())
            .collect();
        let mean_in = (0.8 - 0.4 + 0.2) / 3.0;

        let mut log_sigma = vec![sd0.ln()];
        let mut sigma_vals = vec![sd0];
        let removed = recenter_ruv_eta(&mut etas, kr, &mut log_sigma, &mut sigma_vals, &[true]);

        // Removed exactly the mean; η_RUV now zero-mean.
        assert!((removed - mean_in).abs() < 1e-12);
        let mean_after = etas.iter().map(|e| e[kr]).sum::<f64>() / 3.0;
        assert!(mean_after.abs() < 1e-12);
        // σ scaled by exp(mean).
        assert!((sigma_vals[0] - sd0 * mean_in.exp()).abs() < 1e-12);
        assert!((log_sigma[0] - (sd0.ln() + mean_in)).abs() < 1e-12);
        // Each subject's residual variance is exactly unchanged (the invariance).
        for (e, &v0) in etas.iter().zip(var_before.iter()) {
            let v_after = sigma_vals[0] * sigma_vals[0] * (2.0_f64 * e[kr]).exp();
            assert!(
                (v_after - v0).abs() < 1e-12,
                "residual variance moved: {v_after} vs {v0}"
            );
        }
        // Non-RUV coordinate untouched.
        assert_eq!(etas[0][0], 0.3);
    }

    #[test]
    fn ruv_recenter_allowed_only_when_all_residual_sigma_free() {
        // No iiv_on_ruv → never re-center.
        assert!(!ruv_recenter_allowed(false, None, &[false]));
        // Single free residual σ → OK.
        assert!(ruv_recenter_allowed(true, None, &[false]));
        // Combined error, additive FIXed (index 1) → NOT OK: scaling only the free
        // proportional σ would leave the fixed additive part uncompensated and
        // break the residual-variance invariance (the #903 review finding).
        assert!(!ruv_recenter_allowed(true, None, &[false, true]));
        // FREM EPSCOV (index 1, always FIX) is exempt — a free PK residual σ at 0
        // still re-centers.
        assert!(ruv_recenter_allowed(true, Some(1), &[false, true]));
        // FREM EPSCOV exempt, but a second *real* residual σ FIXed (index 2) blocks.
        assert!(!ruv_recenter_allowed(true, Some(1), &[false, true, true]));
    }

    #[test]
    fn recenter_ruv_eta_noop_when_no_sigma_absorbs() {
        // Every RUV σ FIXed (none absorbs) → η mean is left in place, σ untouched.
        let mut etas = vec![vec![0.5], vec![-0.1]];
        let mut log_sigma = vec![0.2_f64.ln()];
        let mut sigma_vals = vec![0.2];
        let removed = recenter_ruv_eta(&mut etas, 0, &mut log_sigma, &mut sigma_vals, &[false]);
        assert_eq!(removed, 0.0);
        assert_eq!(etas, vec![vec![0.5], vec![-0.1]]);
        assert_eq!(sigma_vals, vec![0.2]);
    }

    // ---- #895: iiv_on_ruv omega-ridge cap ----

    #[test]
    fn ruv_omega_cap_none_without_ruv_or_when_fixed() {
        let om = DMatrix::from_diagonal(&DVector::from_vec(vec![0.3, 0.2]));
        // No iiv_on_ruv eta.
        assert_eq!(compute_ruv_omega_cap(None, 2, &om, &[false, false]), None);
        // RUV eta index out of range.
        assert_eq!(
            compute_ruv_omega_cap(Some(5), 2, &om, &[false, false]),
            None
        );
        // RUV omega FIXed.
        assert_eq!(compute_ruv_omega_cap(Some(0), 2, &om, &[true, false]), None);
    }

    #[test]
    fn ruv_omega_cap_is_log_variance_plus_growth() {
        let om = DMatrix::from_diagonal(&DVector::from_vec(vec![0.2977886, 0.2]));
        let cap = compute_ruv_omega_cap(Some(0), 2, &om, &[false, false]).unwrap();
        assert!((cap - (0.2977886_f64.ln() + SAEM_RUV_OMEGA_LN_GROWTH)).abs() < 1e-12);
        // ≈ 20× the starting variance.
        assert!((cap.exp() / 0.2977886 - SAEM_RUV_OMEGA_LN_GROWTH.exp()).abs() < 1e-9);
    }

    #[test]
    fn apply_ruv_omega_cap_noop_below_cap() {
        let mut om = DMatrix::from_row_slice(2, 2, &[0.3, 0.05, 0.05, 0.2]);
        let before = om.clone();
        // cap = exp(log(0.3)+3) ≈ 6.0, well above 0.3 → no change.
        let clamped = apply_ruv_omega_cap(
            &mut om,
            0,
            0.3_f64.ln() + SAEM_RUV_OMEGA_LN_GROWTH,
            &[false, false],
        );
        assert!(!clamped);
        assert_eq!(om, before);
    }

    #[test]
    fn apply_ruv_omega_cap_leaves_fixed_offdiagonal_untouched() {
        // ω_RUV (index 0) runs away to 9.0 with a covariance to a FIXed eta (index
        // 1). The cap must pull the diagonal down but leave the user-declared FIXed
        // covariance Ω[0,1] exactly as-is (#903 review), never silently rescale it.
        let cov = 0.5_f64;
        let mut om = DMatrix::from_row_slice(2, 2, &[9.0, cov, cov, 0.2]);
        let log_cap = 0.2977886_f64.ln() + SAEM_RUV_OMEGA_LN_GROWTH;
        let clamped = apply_ruv_omega_cap(&mut om, 0, log_cap, &[false, true]);
        assert!(clamped);
        assert!((om[(0, 0)] - log_cap.exp()).abs() < 1e-9);
        // FIXed off-diagonal unchanged (both symmetric entries).
        assert_eq!(om[(0, 1)], cov);
        assert_eq!(om[(1, 0)], cov);
        // Diagonal shrank while the off-diagonal held, so it stays PD here
        // (det = 5.98·0.2 − 0.25 > 0).
        let det = om[(0, 0)] * om[(1, 1)] - om[(0, 1)] * om[(1, 0)];
        assert!(det > 0.0);
    }

    #[test]
    fn apply_ruv_omega_cap_preserves_correlation_and_pd() {
        // ω_RUV runaway to 9.0, correlated with a second eta (var 0.2).
        let cov = 0.9_f64; // corr = 0.9/sqrt(9*0.2) ≈ 0.671
        let mut om = DMatrix::from_row_slice(2, 2, &[9.0, cov, cov, 0.2]);
        let corr_before = om[(0, 1)] / (om[(0, 0)] * om[(1, 1)]).sqrt();
        let log_cap = 0.2977886_f64.ln() + SAEM_RUV_OMEGA_LN_GROWTH; // ≈ log(5.98)
        let clamped = apply_ruv_omega_cap(&mut om, 0, log_cap, &[false, false]);
        assert!(clamped);
        // Diagonal pulled down to the cap.
        assert!((om[(0, 0)] - log_cap.exp()).abs() < 1e-9);
        // Correlation with the other eta preserved.
        let corr_after = om[(0, 1)] / (om[(0, 0)] * om[(1, 1)]).sqrt();
        assert!((corr_after - corr_before).abs() < 1e-9);
        // Still symmetric and positive-definite (2×2: det > 0, diag > 0).
        assert_eq!(om[(0, 1)], om[(1, 0)]);
        let det = om[(0, 0)] * om[(1, 1)] - om[(0, 1)] * om[(1, 0)];
        assert!(det > 0.0 && om[(0, 0)] > 0.0);
    }

    // ---- #903 review: end-of-exploration cap re-anchoring ----

    #[test]
    fn reanchor_ruv_sigma_caps_loosens_for_low_init_only() {
        // σ init 0.01 (log ≈ -4.6): init cap = -4.6 + 3 = -1.6 (≈ 0.2). If the fit
        // legitimately settled near σ = 0.3 (log ≈ -1.2) by end of exploration, the
        // cap must loosen to -1.2 + 3 = 1.8, not clamp the well-posed fit at 0.2.
        let ln_init = 0.01_f64.ln();
        let mut caps = vec![Some(ln_init + SAEM_RUV_SIGMA_LN_GROWTH)];
        let log_sigma_settled = vec![0.3_f64.ln()];
        reanchor_ruv_sigma_caps(&mut caps, &log_sigma_settled, &[5.0]);
        assert!((caps[0].unwrap() - (0.3_f64.ln() + SAEM_RUV_SIGMA_LN_GROWTH)).abs() < 1e-12);

        // If σ never moved above its init, the cap is unchanged (only loosens).
        let mut caps = vec![Some(ln_init + SAEM_RUV_SIGMA_LN_GROWTH)];
        reanchor_ruv_sigma_caps(&mut caps, &[ln_init], &[5.0]);
        assert!((caps[0].unwrap() - (ln_init + SAEM_RUV_SIGMA_LN_GROWTH)).abs() < 1e-12);

        // Never loosens past the user's own upper bound.
        let mut caps = vec![Some(ln_init + SAEM_RUV_SIGMA_LN_GROWTH)];
        reanchor_ruv_sigma_caps(&mut caps, &[10.0], &[2.0]);
        assert_eq!(caps[0], Some(2.0));

        // `None` (uncapped σ) stays `None`.
        let mut caps = vec![None];
        reanchor_ruv_sigma_caps(&mut caps, &[0.0], &[5.0]);
        assert_eq!(caps[0], None);
    }

    #[test]
    fn reanchor_ruv_omega_cap_loosens_only() {
        let ln_init = 0.01_f64.ln();
        let cap = Some(ln_init + SAEM_RUV_OMEGA_LN_GROWTH);
        // Settled ω_RUV = 0.3 → loosen to log(0.3) + growth.
        let out = reanchor_ruv_omega_cap(cap, 0.3);
        assert!((out.unwrap() - (0.3_f64.ln() + SAEM_RUV_OMEGA_LN_GROWTH)).abs() < 1e-12);
        // Settled below init → unchanged.
        assert_eq!(reanchor_ruv_omega_cap(cap, 0.01), cap);
        // No cap stays none; non-positive variance is a no-op.
        assert_eq!(reanchor_ruv_omega_cap(None, 0.3), None);
        assert_eq!(reanchor_ruv_omega_cap(cap, 0.0), cap);
    }

    // ---- #895 / #1444 / #1451: MH acceptance diagnostic ----

    /// First iteration number the synthetic windows below use. Any value works;
    /// a non-1 start makes the "adapted before the window" gate visible.
    const WSTART: usize = 301;

    /// `n` samples at a constant acceptance `rate`, 1000 proposals an
    /// iteration, every proposal carrying `target` as its kernel's target.
    fn window_at_n_t(rate: f64, n: usize, target: f64) -> Vec<MhRateSample> {
        let prop = 1_000_u64;
        (0..n)
            .map(|j| MhRateSample {
                iter: WSTART + j,
                accepted: (rate * prop as f64).round() as u64,
                proposed: prop,
                target_weight: prop as f64 * target,
            })
            .collect()
    }

    fn window_at_n(rate: f64, n: usize) -> Vec<MhRateSample> {
        window_at_n_t(rate, n, 0.40)
    }

    /// A full-length (`MH_RATE_WINDOW`) tail at a constant acceptance `rate`.
    fn window_at(rate: f64) -> Vec<MhRateSample> {
        window_at_n(rate, MH_RATE_WINDOW)
    }

    /// Wrap samples as a window whose scales were adapted one iteration before
    /// the first sample — the ordinary case.
    fn adapted(samples: &[MhRateSample]) -> MhRateWindow<'_> {
        MhRateWindow {
            samples,
            first_adapt: samples.first().map(|s| s.iter - 1),
        }
    }

    /// `saem_mixing_warning` on an adapted window, at the default rule.
    fn warn(cum_acc: u64, cum_prop: u64, samples: &[MhRateSample]) -> Option<String> {
        saem_mixing_warning(
            cum_acc,
            cum_prop,
            &adapted(samples),
            crate::types::ScaleAdaptation::Interval,
        )
    }

    #[test]
    fn saem_mixing_warning_fires_only_when_stuck() {
        let healthy = window_at(0.40);
        // 0% acceptance over the post-burn-in window → warn.
        assert!(warn(0, 100_000, &window_at(0.0)).is_some());
        // Just below the 1% threshold → warn.
        assert!(warn(50, 100_000, &window_at(0.0005)).is_some());
        // Healthy acceptance → silent.
        assert!(warn(8_000, 100_000, &healthy).is_none());
        // No proposals accumulated (e.g. burn-in ≥ n_iter) → silent, no div-by-0.
        assert!(warn(0, 0, &[]).is_none());
        // Cumulative fine but no window at all → silent, no div-by-0 either.
        assert!(warn(8_000, 100_000, &[]).is_none());
        // A window of empty iterations (every proposal count zero) → silent.
        let empty: Vec<MhRateSample> = (0..MH_RATE_WINDOW)
            .map(|j| MhRateSample {
                iter: WSTART + j,
                accepted: 0,
                proposed: 0,
                target_weight: 0.0,
            })
            .collect();
        assert!(warn(8_000, 100_000, &empty).is_none());
        // The stuck tier keeps its historical wording: `tests/frem_warfarin.rs`
        // asserts the *absence* of "not mixing", so the tail tier below must
        // not reuse that phrase.
        let stuck = warn(0, 100_000, &window_at(0.0)).unwrap();
        assert!(stuck.contains("not mixing"), "{stuck}");
    }

    #[test]
    fn saem_mixing_warning_reports_a_tail_that_never_reached_target() {
        // The regression this exists to catch: a chain parked at 2-4% for a
        // whole run. That is 2-4× above the 1% "stuck" threshold, so the
        // pre-existing tier stayed silent on it — measured on
        // `examples/warfarin_saem.ferx`, which sat at 1.6-4.5% for 375 of 400
        // iterations against a 40% target (issue #1444).
        let w = warn(3_000, 100_000, &window_at(0.03)).expect("a 3% tail must be reported");
        assert!(!w.contains("not mixing"), "wrong tier: {w}");
        // The numbers have to be in the message — the point is that the reader
        // does not have to re-run with a trace to learn how far off it was.
        assert!(w.contains("3.0%"), "tail rate missing: {w}");
        assert!(w.contains("40%"), "target missing: {w}");
        assert!(w.contains("100"), "window length missing: {w}");
        assert!(w.contains("rejected"), "direction missing: {w}");
        assert!(w.contains("scale_adaptation"), "remedy missing: {w}");

        // The other end of the band: near-total acceptance means a step so small
        // each accepted move goes nowhere.
        let hi = warn(90_000, 100_000, &window_at(0.9)).expect("a 90% tail must be reported");
        assert!(hi.contains("90.0%") && hi.contains("accepted"), "{hi}");

        // The band must be a band, not a one-sided test: rates inside it are
        // silent even when they are nowhere near the nominal target.
        for r in [0.11, 0.25, 0.40, 0.60, 0.79] {
            assert!(
                warn(40_000, 100_000, &window_at(r)).is_none(),
                "rate {r} is inside [{MH_RATE_LOW}, {MH_RATE_HIGH}] and must stay silent"
            );
        }

        // The *tail* decides, not the cumulative average. A run that mixed
        // badly early and recovered must NOT warn, and the reverse MUST — which
        // is the whole reason the window exists. Both pass the same cumulative
        // counts, so only the window can distinguish them.
        assert!(
            warn(20_000, 100_000, &window_at(0.42)).is_none(),
            "recovered tail must be silent despite a poor cumulative average"
        );
        assert!(
            warn(20_000, 100_000, &window_at(0.02)).is_some(),
            "degraded tail must warn despite an acceptable cumulative average"
        );
    }

    #[test]
    fn saem_mixing_warning_says_nothing_before_the_scales_had_a_chance() {
        // The *statistical* floor, on its own. A tail this short cannot
        // describe a settled rate whatever the controller did, so the tier
        // holds its tongue. Regression this catches: dropping the minimum and
        // firing the diagnostic on every 5-iteration smoke-test fit.
        assert!(
            warn(3_000, 100_000, &window_at_n(0.03, MH_RATE_MIN_WINDOW - 1)).is_none(),
            "a {}-iteration tail is too short to speak on",
            MH_RATE_MIN_WINDOW - 1
        );
        // One more iteration and it does — so the gate is the length, and this
        // pair cannot pass by the tier being dead.
        assert!(
            warn(3_000, 100_000, &window_at_n(0.03, MH_RATE_MIN_WINDOW)).is_some(),
            "a {MH_RATE_MIN_WINDOW}-iteration tail must be reported"
        );
        // The short tail does not mask the more severe tier: a genuinely stuck
        // run is still reported however short its window.
        assert!(
            warn(0, 100_000, &window_at_n(0.0, 2)).is_some(),
            "tier 1 must not be gated on the window length"
        );
    }

    #[test]
    fn saem_mixing_warning_is_silent_until_the_controller_has_actually_run() {
        // #1451 review. The length floor above does **not** establish that any
        // adaptation happened, and the two must not be conflated: a 45-iteration
        // fit at `omega_burnin = 20`, `adapt_interval = 50` reaches a 25-sample
        // window while the interval rule has fired **zero** times. Measured on
        // that fixture before this gate existed, the run reported "acceptance
        // settled at 8.7% ... against a target of 40%" — a claim about a
        // controller that never ran.
        let samples = window_at(0.03);
        let mode = crate::types::ScaleAdaptation::Interval;

        // Never adapted → silent, however long and however far off the tail is.
        let never = MhRateWindow {
            samples: &samples,
            first_adapt: None,
        };
        assert!(
            saem_mixing_warning(3_000, 100_000, &never, mode).is_none(),
            "a controller that never ran cannot have failed to reach target"
        );

        // Adapted for the first time *during* the window → still silent, because
        // the earlier samples were drawn at the untouched initial scale and the
        // reported mean is partly about the initial condition.
        let mid = MhRateWindow {
            samples: &samples,
            first_adapt: Some(samples[samples.len() / 2].iter),
        };
        assert!(
            saem_mixing_warning(3_000, 100_000, &mid, mode).is_none(),
            "a window straddling the first adaptation must not be reported"
        );
        // The exact boundary, both sides: adapting *at* the first sample's
        // iteration is too late (that sample was drawn before the update),
        // one iteration earlier is in time.
        let at = MhRateWindow {
            samples: &samples,
            first_adapt: Some(samples[0].iter),
        };
        assert!(
            saem_mixing_warning(3_000, 100_000, &at, mode).is_none(),
            "off-by-one: too late"
        );
        let before = MhRateWindow {
            samples: &samples,
            first_adapt: Some(samples[0].iter - 1),
        };
        assert!(
            saem_mixing_warning(3_000, 100_000, &before, mode).is_some(),
            "adapted before the window: the tier must speak"
        );

        // And the gate is specific to the tail tier: a genuinely stuck chain is
        // still reported even if nothing was ever adapted.
        assert!(
            saem_mixing_warning(0, 100_000, &never, mode).is_some(),
            "tier 1 must not be gated on adaptation"
        );
    }

    #[test]
    fn saem_mixing_warning_quotes_a_proposal_weighted_target() {
        // #1451 review. `accepted / proposed` pools the primary kernel with the
        // componentwise sweep, so naming the combined rate after the primary
        // kernel's target alone is wrong whenever the mix is not dominated by
        // it. The HMC case is the extreme: one HMC proposal at 0.65 against 20
        // componentwise proposals at 0.44 is a combined target of
        // (1·0.65 + 20·0.44) / 21 = 0.4500, and a two-η HMC fixture measured
        // 0.4978 realised — which the old message would have called 15 points
        // *below* a 65% target when it is 5 points above the real one.
        let hmc: Vec<MhRateSample> = (0..MH_RATE_WINDOW)
            .map(|j| MhRateSample {
                iter: WSTART + j,
                // Drive it below the band so the message is emitted at all.
                accepted: 1,
                proposed: 21,
                target_weight: 1.0 * 0.65 + 20.0 * CW_TARGET_ACCEPT,
            })
            .collect();
        // Cumulative kept above the 1% "stuck" tier so this exercises tier 2.
        let w = saem_mixing_warning(
            5_000,
            100_000,
            &adapted(&hmc),
            crate::types::ScaleAdaptation::Interval,
        )
        .expect("a 4.8% tail must be reported");
        let want = (0.65 + 20.0 * CW_TARGET_ACCEPT) / 21.0;
        assert!(
            (want - 0.45).abs() < 5e-3,
            "hand-computed weighted target moved: {want}"
        );
        assert!(
            w.contains("target of 45%"),
            "must quote the weighted 45%, not the primary kernel's 65%: {w}"
        );
        assert!(
            !w.contains("65%"),
            "the primary-only target must not appear: {w}"
        );

        // Control: when every proposal shares one target the weighting is a
        // no-op, so the single-kernel case still reads exactly as before.
        let single = warn(3_000, 100_000, &window_at_n_t(0.03, MH_RATE_WINDOW, 0.40))
            .expect("must be reported");
        assert!(single.contains("target of 40%"), "{single}");
    }

    #[test]
    fn saem_mixing_warning_does_not_prescribe_the_rule_already_running() {
        // #1451 review: recommending `scale_adaptation = robbins_monro` to a run
        // that is already using it is a no-op repair.
        let samples = window_at(0.03);
        let under_interval = saem_mixing_warning(
            3_000,
            100_000,
            &adapted(&samples),
            crate::types::ScaleAdaptation::Interval,
        )
        .expect("reported");
        assert!(
            under_interval.contains("scale_adaptation = robbins_monro")
                && under_interval.contains("adapt_interval"),
            "the interval arm must offer the switch: {under_interval}"
        );

        let under_rm = saem_mixing_warning(
            3_000,
            100_000,
            &adapted(&samples),
            crate::types::ScaleAdaptation::RobbinsMonro,
        )
        .expect("reported");
        assert!(
            under_rm.contains("already active"),
            "the RM arm must say so: {under_rm}"
        );
        assert!(
            !under_rm.contains("Consider `[fit_options] scale_adaptation = robbins_monro`"),
            "must not prescribe the rule that is already running: {under_rm}"
        );
        // Both still carry the numbers; only the remedy differs.
        for m in [&under_interval, &under_rm] {
            assert!(m.contains("3.0%") && m.contains("target of 40%"), "{m}");
        }
    }

    // ---- #1444: MH step-scale adaptation ----

    #[test]
    fn interval_scale_update_is_the_legacy_times_1_1_or_0_9() {
        // The legacy rule, pinned as arithmetic so the extraction into a
        // function cannot silently have changed it. Regression this catches:
        // a factor, a comparison direction or a clamp drifting while the
        // Robbins-Monro arm is edited next door.
        assert_eq!(interval_scale_update(0.3, 0.9, 0.4, 0.01, 5.0), 0.3 * 1.1);
        assert_eq!(interval_scale_update(0.3, 0.1, 0.4, 0.01, 5.0), 0.3 * 0.9);
        // Strictly-above, so an exactly-on-target window shrinks (the historical
        // `if rate > target` branch, not `>=`).
        assert_eq!(interval_scale_update(0.3, 0.4, 0.4, 0.01, 5.0), 0.3 * 0.9);
        // Both clamps bind.
        assert_eq!(interval_scale_update(4.9, 1.0, 0.4, 0.01, 5.0), 5.0);
        assert_eq!(interval_scale_update(0.0105, 0.0, 0.4, 0.01, 5.0), 0.01);
    }

    #[test]
    fn rm_scale_update_matches_the_hand_computed_formula() {
        // Exact formula, not just the sign: log δ += c·k^-0.6·(rate − target).
        let up = rm_scale_update(0.3, 0.9, 0.4, 1, 0.01, 5.0);
        let down = rm_scale_update(0.3, 0.1, 0.4, 1, 0.01, 5.0);
        assert!(up > 0.3, "rate above target must raise the scale: {up}");
        assert!(down < 0.3, "rate below target must lower it: {down}");
        // Hand-computed: k = 1 → k^-0.6 = 1, so log δ moves by exactly 0.5.
        let want = (0.3_f64.ln() + 0.5).exp();
        assert!((up - want).abs() < 1e-12, "got {up}, want {want}");
        // And at k = 32: 32^-0.6 = 2^(-3) = 0.125 exactly, so log δ moves by
        // 0.125·(0.9 − 0.4) = 0.0625. A power of two makes this hand-checkable
        // rather than a number copied out of the implementation.
        let at32 = rm_scale_update(0.3, 0.9, 0.4, 32, 0.01, 5.0);
        let want32 = (0.3_f64.ln() + 0.0625).exp();
        assert!(
            (at32 - want32).abs() < 1e-12,
            "k=32: got {at32}, want {want32}"
        );

        // Diminishing adaptation: the same discrepancy must move log δ strictly
        // less at a later iteration while still pointing the right way. That is
        // the condition the ergodicity argument needs, so it is worth pinning
        // rather than assuming.
        let early = rm_scale_update(0.3, 0.9, 0.4, 1, 0.01, 5.0).ln() - 0.3_f64.ln();
        let late = rm_scale_update(0.3, 0.9, 0.4, 400, 0.01, 5.0).ln() - 0.3_f64.ln();
        assert!(late > 0.0 && late < early * 0.1, "{early} then {late}");

        // Zero discrepancy is a fixed point; both clamps bind.
        assert!((rm_scale_update(0.3, 0.4, 0.4, 7, 0.01, 5.0) - 0.3).abs() < 1e-12);
        assert_eq!(rm_scale_update(4.9, 1.0, 0.0, 1, 0.01, 5.0), 5.0);
        assert_eq!(rm_scale_update(0.011, 0.0, 1.0, 1, 0.01, 5.0), 0.01);
    }

    /// Synthetic acceptance response for the closed-loop straddle below: a
    /// random-walk MH kernel accepts less the larger its step, so `A/δ` capped
    /// at 1 is a monotone, exactly-computable stand-in.
    ///
    /// `A = 0.008` puts the 40% target at `δ* = 0.02`, 15× below the 0.3 the
    /// SAEM state starts at — and it reproduces the reported symptom: at the
    /// 0.3 start the rate is `0.008/0.3 = 2.7%`, inside the 2-4% measured on
    /// `examples/warfarin_saem.ferx` (issue #1444).
    fn synth_accept(delta: f64) -> f64 {
        (0.008_f64 / delta).min(1.0)
    }

    #[test]
    fn rm_reaches_a_target_the_interval_rule_provably_cannot() {
        // A differential pair, run as two closed loops over the *production*
        // update functions, that has to straddle the diagnostic's gate: from
        // the same 0.3 start and the same acceptance response, the interval
        // rule must still be far below target after 400 iterations and
        // Robbins-Monro must have arrived.
        const TARGET: f64 = 0.40;
        const N_ITER: usize = 400;
        const ADAPT_INTERVAL: usize = 50;

        // --- interval arm: 8 corrections in 400 iterations ---
        let mut d_int = 0.3_f64;
        let mut n_corrections = 0;
        for k in 1..=N_ITER {
            if k % ADAPT_INTERVAL == 0 {
                d_int = interval_scale_update(
                    d_int,
                    synth_accept(d_int),
                    TARGET,
                    MH_BLOCK_SCALE_MIN,
                    MH_BLOCK_SCALE_MAX,
                );
                n_corrections += 1;
            }
        }
        assert_eq!(n_corrections, 8, "the reach argument assumes 8 corrections");
        // The reach, as arithmetic rather than as prose: 8 × 0.9, nothing more.
        let reach_floor = 0.3 * 0.9_f64.powi(8);
        assert!(
            (d_int - reach_floor).abs() < 1e-15,
            "interval arm must land exactly on 0.3·0.9^8 = {reach_floor}, got {d_int}"
        );

        // --- Robbins-Monro arm: a step every iteration ---
        let mut d_rm = 0.3_f64;
        for k in 1..=N_ITER {
            d_rm = rm_scale_update(
                d_rm,
                synth_accept(d_rm),
                TARGET,
                k,
                MH_BLOCK_SCALE_MIN,
                MH_BLOCK_SCALE_MAX,
            );
        }

        let rate_int = synth_accept(d_int);
        let rate_rm = synth_accept(d_rm);
        // `f64::max`/`min` swallow NaN and `clamp` would propagate one, so check
        // before comparing (CLAUDE.md's fold trap).
        assert!(
            rate_int.is_finite() && rate_rm.is_finite(),
            "non-finite realised rates: interval {rate_int}, rm {rate_rm}"
        );
        // Realised on this stream: interval 0.0619, RM 0.3998.
        assert!(
            (rate_rm - TARGET).abs() < 0.02,
            "RM must arrive at the target: {rate_rm}"
        );
        // Assert the straddle itself, so the pair cannot quietly become a
        // tautology if the acceptance response or the start is ever retuned:
        // the two arms must sit on opposite sides of the diagnostic's gate.
        assert!(
            rate_int < MH_RATE_LOW && MH_RATE_LOW < rate_rm,
            "the pair must straddle MH_RATE_LOW = {MH_RATE_LOW}: interval {rate_int}, rm {rate_rm}"
        );

        // It must be a controller, not a ratchet: fed an acceptance *above*
        // target from the same start it moves the other way. Without this, a
        // rule that only ever shrinks δ would pass everything above.
        let mut up = 0.3_f64;
        for k in 1..=N_ITER {
            up = rm_scale_update(up, 0.90, TARGET, k, MH_BLOCK_SCALE_MIN, MH_BLOCK_SCALE_MAX);
        }
        assert!(up > 0.3, "RM must also grow the step: {up}");
    }

    #[test]
    fn scale_adaptation_parses_every_documented_spelling() {
        use crate::types::ScaleAdaptation;
        let parse = |key: &str, v: &str| -> Result<ScaleAdaptation, String> {
            let mut opts = FitOptions::default();
            crate::parser::model_parser::apply_fit_option(&mut opts, key, v)
                .map(|_| opts.saem_scale_adaptation)
        };
        assert_eq!(
            parse("scale_adaptation", "robbins_monro").unwrap(),
            ScaleAdaptation::RobbinsMonro
        );
        assert_eq!(
            parse("scale_adaptation", "robbins-monro").unwrap(),
            ScaleAdaptation::RobbinsMonro
        );
        assert_eq!(
            parse("scale_adaptation", "RM").unwrap(),
            ScaleAdaptation::RobbinsMonro
        );
        assert_eq!(
            parse("scale_adaptation", "interval").unwrap(),
            ScaleAdaptation::Interval
        );
        assert_eq!(
            parse("saem_scale_adaptation", "legacy").unwrap(),
            ScaleAdaptation::Interval
        );
        let err = parse("scale_adaptation", "adam").unwrap_err();
        assert!(
            err.contains("scale_adaptation") && err.contains("adam"),
            "{err}"
        );

        // The default stays the legacy rule: shipping Robbins-Monro as the
        // default would silently move every existing SAEM user's estimates, and
        // it is measured to regress one benchmark (issue #1444).
        assert_eq!(
            FitOptions::default().saem_scale_adaptation,
            ScaleAdaptation::Interval
        );
    }

    #[test]
    fn scale_adaptation_keys_are_registered_as_saem_options() {
        // A new SAEM key must be added to `method_specific_keys` as well as to
        // the parser, and nothing forces that — an option that works while
        // warning "is not used by method `SAEM` and will be ignored" is the
        // failure mode.
        for key in ["scale_adaptation", "saem_scale_adaptation"] {
            let opts = FitOptions {
                method: EstimationMethod::Saem,
                user_set_keys: vec![key.to_string()],
                ..FitOptions::default()
            };
            let w = opts.unsupported_keys_warnings();
            assert!(w.is_empty(), "`{key}` must not warn under saem: {w:?}");
        }
        // Control: a genuinely non-SAEM key still warns, so the emptiness above
        // is not "this function never warns".
        let opts = FitOptions {
            method: EstimationMethod::Saem,
            user_set_keys: vec!["imp_samples".to_string()],
            ..FitOptions::default()
        };
        assert!(opts
            .unsupported_keys_warnings()
            .iter()
            .any(|m| m.contains("imp_samples")));
    }

    // ---- #1444: the rules, wired into `run_saem` ----
    //
    // These four tests call `fit()`, which CLAUDE.md's Tier 1 says to avoid.
    // They stay in `--lib` deliberately (#1451 review), on three grounds.
    // First, cost: all four together run in **0.79 s** single-threaded
    // (`cargo test --lib -- --test-threads=1`, 11 fits of 60 iterations on 8
    // subjects), against the tier's "seconds total" budget. Second, precedent:
    // `origin/main`'s own test module in this file already makes 14 `fit()`
    // calls with `saem_n_exploration = 4`-style budgets, so a short fixed-budget
    // fit is this file's established unit-test shape, not a new one. Third, and
    // decisively, what they cover cannot move: mutation M2 — deleting the
    // Robbins-Monro block from `run_saem` entirely — is invisible to every pure
    // update-rule test, and `slow-tests` never runs on a PR, so moving these to
    // Tier 3 would restore exactly the hole mutation testing found. The genuine
    // convergence run (400 iterations on warfarin) *is* gated, in
    // `tests/saem_scale_adaptation.rs`.

    /// 1-cpt IV, two estimated ETAs, `n_per` subjects. `omega_cl`/`omega_v` are
    /// the (FIXed) ETA variances, which is the knob that sets how far the MH
    /// proposals overshoot the posterior and hence the acceptance rate.
    fn scale1444_model(omega_cl: f64, omega_v: f64) -> CompiledModel {
        let src = format!(
            r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ {omega_cl} FIX
  omega ETA_V ~ {omega_v} FIX
  sigma EPS ~ 0.04 FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
"
        );
        crate::parser::model_parser::parse_model_string(&src).expect("model parses")
    }

    /// `omega_burnin` for the fixture below: the iterations excluded from the
    /// diagnostic's window, and hence from the tail rate it reports.
    const SCALE1444_BURNIN: usize = 30;

    /// 60 SAEM iterations — 30 post-burn-in, past `MH_RATE_MIN_WINDOW` — with
    /// two firings of the interval rule before the window opens, so both arms
    /// are exercised and the tail tier's "the controller has run" gate is met.
    ///
    /// Returns the fit **and** its realised combined MH acceptance over those
    /// post-burn-in iterations, read from `FitResult::saem_mh_accept_tail` —
    /// the same number the diagnostic reports, so the tests measure what the
    /// warning says rather than a proxy.
    ///
    /// This used to read the optimizer trace instead, which was wrong for a
    /// reason worth recording: the trace writer is a thread-local and
    /// `fit_inner` runs inside a shared Rayon pool, so any *other* concurrent
    /// fit whose job is stolen onto this one's thread perturbs it. Under CI's
    /// parallelism that made these tests fail on `trace_path` being `None`.
    /// A field on the result has no such coupling.
    fn scale1444_fit(
        omega_cl: f64,
        omega_v: f64,
        rule: crate::types::ScaleAdaptation,
    ) -> (crate::types::FitResult, f64) {
        let model = scale1444_model(omega_cl, omega_v);
        let pop = mix996_pop(4);
        let opts = FitOptions {
            method: EstimationMethod::Saem,
            saem_n_exploration: 40,
            saem_n_convergence: 20,
            saem_adapt_interval: 25,
            // `omega_burnin = 30` puts the whole reported window (iterations
            // 31-60) after the interval rule's first firing at k = 25, which
            // the tail tier now requires (#1451 review). At the default 20 the
            // window would open at 21 and straddle that first adaptation.
            saem_omega_burnin: SCALE1444_BURNIN,
            saem_scale_adaptation: rule,
            saem_seed: Some(1444),
            run_covariance_step: false,
            ..FitOptions::default()
        };
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("SAEM Ok");
        let tail = res
            .saem_mh_accept_tail
            .expect("a SAEM fit with post-burn-in iterations must report a tail rate");
        assert!(tail.is_finite(), "non-finite tail acceptance: {tail}");
        (res, tail)
    }

    #[test]
    fn interval_is_the_default_and_bit_identical_to_asking_for_it() {
        use crate::types::ScaleAdaptation;
        let model = scale1444_model(0.09, 0.04);
        let pop = mix996_pop(4);
        let base = FitOptions {
            method: EstimationMethod::Saem,
            saem_n_exploration: 40,
            saem_n_convergence: 20,
            saem_adapt_interval: 25,
            saem_seed: Some(1444),
            run_covariance_step: false,
            ..FitOptions::default()
        };
        // Arm 1: the option left alone. Arm 2: `interval` asked for explicitly.
        let untouched = crate::api::fit(&model, &pop, &model.default_params, &base).expect("Ok");
        let explicit = crate::api::fit(
            &model,
            &pop,
            &model.default_params,
            &FitOptions {
                saem_scale_adaptation: ScaleAdaptation::Interval,
                ..base
            },
        )
        .expect("Ok");
        // Bit-for-bit, not to a tolerance: the legacy path must still be the
        // code that ran before #1444, and the default must select it. A single
        // `f64` differing in its last bit fails this. The regression it catches
        // is the default flipping to `RobbinsMonro` — which is what the
        // research branch this came from shipped.
        assert!(untouched.ofv.is_finite(), "OFV {}", untouched.ofv);
        assert_eq!(
            untouched.ofv.to_bits(),
            explicit.ofv.to_bits(),
            "default arm OFV {} vs explicit-interval {}",
            untouched.ofv,
            explicit.ofv
        );
        assert_eq!(untouched.theta.len(), explicit.theta.len());
        for (a, b) in untouched.theta.iter().zip(&explicit.theta) {
            assert_eq!(a.to_bits(), b.to_bits(), "theta {a} vs {b}");
        }
        for (a, b) in untouched.sigma.iter().zip(&explicit.sigma) {
            assert_eq!(a.to_bits(), b.to_bits(), "sigma {a} vs {b}");
        }
    }

    #[test]
    fn robbins_monro_moves_the_realised_acceptance_toward_target_both_ways() {
        use crate::types::ScaleAdaptation;
        // The wiring test, and a differential pair in *both* directions on a
        // real fit. Asserting only "the two rules give different estimates" is
        // not enough: switching to Robbins-Monro also switches the interval
        // bump off, so the runs differ even when the RM update itself is a
        // no-op — a mutation that deletes the RM block in `run_saem` passes
        // that weaker test. Measuring the realised acceptance rate is what
        // distinguishes "adapting differently" from "not adapting".
        const TARGET: f64 = 0.40;

        // (a) Starting far BELOW target: a grossly over-dispersed Ω (SD 10 on a
        //     log-scale η) makes the block proposal ≈ 3 log-units, so nearly
        //     everything is rejected. Realised: interval 0.0844, RM 0.3271.
        let (_, lo_int) = scale1444_fit(100.0, 100.0, ScaleAdaptation::Interval);
        let (_, lo_rm) = scale1444_fit(100.0, 100.0, ScaleAdaptation::RobbinsMonro);
        // (b) Starting ABOVE target, so the rule is shown to be a controller
        //     and not a ratchet on a real chain too. Realised: interval 0.6114,
        //     RM 0.4324.
        let (_, hi_int) = scale1444_fit(0.09, 0.04, ScaleAdaptation::Interval);
        let (_, hi_rm) = scale1444_fit(0.09, 0.04, ScaleAdaptation::RobbinsMonro);
        for (name, r) in [
            ("lo_int", lo_int),
            ("lo_rm", lo_rm),
            ("hi_int", hi_int),
            ("hi_rm", hi_rm),
        ] {
            assert!(r.is_finite(), "{name} is not finite: {r}");
        }

        // Each RM arm must be closer to target than its interval twin. Under a
        // mutation that unwires the RM block the two arms become *equal*, so
        // both of these die.
        assert!(
            (lo_rm - TARGET).abs() < (lo_int - TARGET).abs() - 0.1,
            "below target: RM {lo_rm:.4} must be closer to {TARGET} than interval {lo_int:.4}"
        );
        assert!(
            (hi_rm - TARGET).abs() < (hi_int - TARGET).abs() - 0.1,
            "above target: RM {hi_rm:.4} must be closer to {TARGET} than interval {hi_int:.4}"
        );
        // Each RM arm must also land *near* target in absolute terms, not merely
        // nearer than its twin. This is what separates "the RM block ran" from
        // "half of it ran": deleting the per-η componentwise update alone leaves
        // the block update to carry the combined rate, and it gets only part of
        // the way. Both bounds are measured, not picked (#1451 review) — the
        // realised |rate − target| is:
        //
        //             both halves   componentwise half deleted   bound
        //   below       0.0729              0.1624               0.12
        //   above       0.0324              0.0807               0.06
        //
        // i.e. each bound sits between the two, with 1.6x / 1.9x headroom over
        // the passing case, so it is exactly this mutation that both die for.
        // The block half alone is caught by the `closer than its twin` pair
        // above.
        assert!(
            (lo_rm - TARGET).abs() < 0.12,
            "below target: RM must land near {TARGET}, got {lo_rm:.4}"
        );
        assert!(
            (hi_rm - TARGET).abs() < 0.06,
            "above target: RM must land near {TARGET}, got {hi_rm:.4}"
        );
        // And the low pair must straddle the diagnostic's own gate, asserted so
        // the pair cannot quietly become two runs on the same side of it.
        assert!(
            lo_int < MH_RATE_LOW && MH_RATE_LOW < lo_rm,
            "the low pair must straddle MH_RATE_LOW = {MH_RATE_LOW}: \
             interval {lo_int:.4}, rm {lo_rm:.4}"
        );
        // The high pair moves the other way: RM must *reduce* the rate, which a
        // rule that only ever shrinks the step could not do.
        assert!(
            hi_rm < hi_int,
            "above target RM must lower the rate: {hi_rm:.4} vs {hi_int:.4}"
        );
    }

    #[test]
    fn saem_says_nothing_about_a_target_the_interval_rule_never_tried_to_reach() {
        use crate::types::ScaleAdaptation;
        // #1451 review, end to end on the exact fixture it constructed:
        // `omega_burnin = 20`, `adapt_interval = 50`, 45 iterations. The window
        // reaches its 25-sample floor at iteration 45, but the interval rule
        // fires at multiples of 50, so it has run **zero** times. Before the
        // gate this reported "acceptance settled at 8.7% ... against a target
        // of 40%" — a verdict on a controller that never ran.
        //
        // The Ω is the same grossly over-dispersed one the positive fixture
        // uses, so the rate really is far outside the band: what changes is
        // only whether ferx is entitled to call that a failure to adapt.
        let model = scale1444_model(100.0, 100.0);
        let pop = mix996_pop(4);
        let opts = FitOptions {
            method: EstimationMethod::Saem,
            saem_n_exploration: 40,
            saem_n_convergence: 5,
            saem_adapt_interval: 50,
            saem_omega_burnin: 20,
            saem_scale_adaptation: ScaleAdaptation::Interval,
            saem_seed: Some(1444),
            run_covariance_step: false,
            ..FitOptions::default()
        };
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("SAEM Ok");
        assert!(
            !res.warnings
                .iter()
                .any(|w| w.contains("acceptance settled at")),
            "no adaptation has happened yet, so the tail tier must stay quiet: {:?}",
            res.warnings
        );

        // The control that stops this from passing for the wrong reason: the
        // *same* model and Ω, run long enough that the rule does fire before
        // the window opens, must warn. Without this the assertion above is
        // satisfied by a diagnostic that never speaks at all.
        let (loud, _) = scale1444_fit(100.0, 100.0, ScaleAdaptation::Interval);
        assert!(
            loud.warnings
                .iter()
                .any(|w| w.contains("acceptance settled at")),
            "the adapted control must still warn: {:?}",
            loud.warnings
        );
    }

    #[test]
    fn saem_reports_a_tail_acceptance_far_from_target_end_to_end() {
        use crate::types::ScaleAdaptation;
        // A grossly over-dispersed Ω (SD 10 on a log-scale η): the block
        // proposals are `δ·chol(Ω)·z` ≈ 3 log-units, so almost everything is
        // rejected. Realised tail rate on this fixture: 8.4% — below the 10%
        // band and *above* the 1% "not mixing" tier, i.e. exactly the regime
        // the pre-#1444 diagnostic could not see, reached by a real fit rather
        // than by calling the predicate directly.
        let (hot, _) = scale1444_fit(100.0, 100.0, ScaleAdaptation::Interval);
        let hit = hot
            .warnings
            .iter()
            .find(|w| w.contains("acceptance settled at"))
            .unwrap_or_else(|| {
                panic!(
                    "expected the tail-acceptance warning, got {:?}",
                    hot.warnings
                )
            });
        assert!(
            !hot.warnings.iter().any(|w| w.contains("not mixing")),
            "must be the tail tier, not the stuck tier: {:?}",
            hot.warnings
        );
        assert!(
            hit.contains("rejected") && hit.contains("over the last"),
            "message must name the direction and the window: {hit}"
        );
        // The proposal-weighted target has to survive the *wiring*, not just
        // the predicate: this fixture runs 20 block proposals at 0.40 and 20
        // componentwise at 0.44 per subject-iteration, so the weighted target
        // is exactly 0.42. A mutation that stops accumulating `target_weight`
        // in `run_saem` still satisfies every unit test of the predicate (they
        // build their own samples) and prints "target of 0%" here (#1451 F2).
        assert!(
            hit.contains("proposal-weighted target of 42%"),
            "must quote the weighted 42% for this 0.40/0.44 mix: {hit}"
        );
        // Parse the realised rate back out and check it really is outside the
        // band, so a message carrying a wrong number could not pass.
        let pct: f64 = hit
            .split("settled at ")
            .nth(1)
            .and_then(|s| s.split('%').next())
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| panic!("no parseable rate in: {hit}"));
        assert!(pct.is_finite(), "non-finite rate in: {hit}");
        assert!(
            pct / 100.0 < MH_RATE_LOW && pct / 100.0 > SAEM_MH_STUCK_ACCEPT,
            "{pct}% must be inside the gap the old threshold missed: {hit}"
        );

        // The negative control, same code path and iteration counts: a
        // sensibly-scaled Ω (realised tail 61.14%) must not trip it. Without
        // this the assertion above is satisfied by a diagnostic that fires on
        // everything.
        let (ok, ok_tail) = scale1444_fit(0.09, 0.04, ScaleAdaptation::Interval);
        // Pin that the control is silent *because* its rate is inside the band,
        // not because the fixture happens to miss the tier for some other
        // reason. Realised: 0.6114.
        assert!(
            ok_tail.is_finite() && (MH_RATE_LOW..=MH_RATE_HIGH).contains(&ok_tail),
            "the control's realised tail {ok_tail:.4} must be inside the band, \
             or its silence proves nothing"
        );
        assert!(
            !ok.warnings
                .iter()
                .any(|w| w.contains("acceptance settled at")),
            "well-scaled fit must stay silent, got {:?}",
            ok.warnings
        );
    }

    // ---- #895: block-kernel per-coordinate scaling ----

    fn one_obs_subject() -> Subject {
        use crate::types::DoseEvent;
        use std::collections::HashMap;
        Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0],
            obs_raw_times: Vec::new(),
            observations: vec![1.0],
            obs_cmts: vec![1],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            reset_covariates: Vec::new(),
            cens: vec![0],
            occasions: vec![],
            obs_l2: Vec::new(),
            dose_occasions: vec![],
            reset_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
        }
    }

    #[test]
    fn mh_steps_block_scale_none_equals_all_ones() {
        // `None` must reproduce the plain chol(Ω)·z block move bit-for-bit, and an
        // explicit all-ones scale must be identical to it — the multiplier is a
        // no-op at 1.0, so non-FREM models are unaffected.
        use crate::stats::likelihood::individual_nll;
        use crate::types::SigmaVector;
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let model = analytical_model(GradientMethod::Auto);
        let subj = one_obs_subject();
        let omega =
            OmegaMatrix::from_diagonal(&[0.09, 0.04], vec!["ETA_CL".into(), "ETA_V".into()]);
        let sigma = SigmaVector {
            values: vec![0.1],
            names: vec!["PROP".into()],
        };
        let theta = vec![1.0, 8.0];

        let run = |scale: Option<&[f64]>| {
            let mut eta = vec![0.2_f64, -0.1];
            let nll0 = individual_nll(&model, &subj, &theta, &eta, &omega, &sigma.values);
            let mut rng = StdRng::seed_from_u64(7);
            let mut scratch = MhScratch::default();
            scratch.begin_subject(&model, &subj, &theta, eta.len());
            let (acc, nll) = mh_steps(
                &mut eta,
                nll0,
                &subj,
                &model,
                &theta,
                &omega,
                &sigma.values,
                0.5,
                scale,
                &mut rng,
                50,
                &mut scratch,
                None,
                None,
            );
            (eta, acc, nll)
        };

        let (eta_none, acc_none, nll_none) = run(None);
        let (eta_ones, acc_ones, nll_ones) = run(Some(&[1.0, 1.0]));
        assert_eq!(eta_none, eta_ones);
        assert_eq!(acc_none, acc_ones);
        assert_eq!(nll_none, nll_ones);
    }

    #[test]
    fn mh_steps_block_scale_freezes_damped_coordinate() {
        // A block scale of 0 on coordinate 1 means the joint proposal never moves
        // that coordinate (its perturbation is multiplied by 0), while coordinate
        // 0 is free to move — the mechanism that damps near-deterministic FREM
        // covariate ETAs so the block kernel can still explore the PK block.
        use crate::stats::likelihood::individual_nll;
        use crate::types::SigmaVector;
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let model = analytical_model(GradientMethod::Auto);
        let subj = one_obs_subject();
        let omega =
            OmegaMatrix::from_diagonal(&[0.09, 0.04], vec!["ETA_CL".into(), "ETA_V".into()]);
        let sigma = SigmaVector {
            values: vec![0.1],
            names: vec!["PROP".into()],
        };
        let theta = vec![1.0, 8.0];
        let eta0 = vec![0.2_f64, -0.1];

        let mut eta = eta0.clone();
        let nll0 = individual_nll(&model, &subj, &theta, &eta, &omega, &sigma.values);
        let mut rng = StdRng::seed_from_u64(11);
        let mut scratch = MhScratch::default();
        scratch.begin_subject(&model, &subj, &theta, eta.len());
        mh_steps(
            &mut eta,
            nll0,
            &subj,
            &model,
            &theta,
            &omega,
            &sigma.values,
            0.8,
            Some(&[1.0, 0.0]),
            &mut rng,
            100,
            &mut scratch,
            None,
            None,
        );
        // Coordinate 1 is frozen at its start; coordinate 0 has moved.
        assert_eq!(
            eta[1], eta0[1],
            "coordinate with block scale 0 must not move"
        );
        assert_ne!(eta[0], eta0[0], "unclamped coordinate should have moved");
    }

    // ── E-step sizing (#1459) ────────────────────────────────────────────────

    /// The automatic count on the shapes it was measured on. Each row is a real
    /// benchmark from issue #1459, with its observations-per-η and the count
    /// the rule gives it; a mutation that drops the density term (always the
    /// floor, or always the cap) reddens a different subset of these rows, so
    /// no single constant satisfies the table.
    #[test]
    fn auto_n_mh_steps_tracks_observations_per_eta() {
        // The expected counts below are written out rather than spelled with
        // the clamp constants, so that moving a clamp reddens this test instead
        // of moving the expectations with it.
        assert_eq!((AUTO_MH_STEPS_MIN, AUTO_MH_STEPS_MAX), (6, 20));

        // (n_obs, n_subjects, n_eta, expected, what it is)
        let cases = [
            // cefepime: 702 obs / 458 subjects / 3 η = 0.51 per η → 1.3 → floor.
            (702, 458, 3, 6, "cefepime, 0.51 obs/eta"),
            // vancomycin: 274 / 100 / 3 = 0.91 per η → 2.3 → floor.
            (274, 100, 3, 6, "vancomycin, 0.91 obs/eta"),
            // pembrolizumab: 1231 / 303 / 4 = 1.02 per η → 2.5 → floor.
            (1231, 303, 4, 6, "pembrolizumab, 1.02 obs/eta"),
            // busulfan: 5259 / 600 / 3 = 2.92 per η → 7.3 → 7, the arm where
            // the density term actually decides the answer.
            (5259, 600, 3, 7, "busulfan, 2.92 obs/eta"),
            // Emax PKPD: 1600 / 100 / 2 = 8.0 per η → 20 → the cap, i.e. the
            // pre-#1459 default is kept exactly where it was calibrated.
            (1600, 100, 2, 20, "emax PKPD, 8.0 obs/eta"),
            // Denser still must not exceed the cap.
            (100_000, 100, 2, 20, "500 obs/eta"),
            // A single observation per subject against 6 η cannot go below it.
            (300, 300, 6, 6, "0.17 obs/eta"),
            // Just under and just over the knee where the density term takes
            // over from the floor (2.4 obs/η ⇒ exactly 6).
            (2400, 1000, 1, 6, "2.4 obs/eta, 1 eta"),
            (2500, 1000, 1, 6, "2.5 obs/eta, 1 eta — rounds to 6"),
            (
                2600,
                1000,
                1,
                7,
                "2.6 obs/eta, 1 eta — first count above the floor",
            ),
        ];
        for (n_obs, n_subjects, n_eta, want, what) in cases {
            assert_eq!(
                auto_n_mh_steps(n_obs, n_subjects, n_eta),
                want,
                "{what}: {n_obs} obs / {n_subjects} subjects / {n_eta} eta"
            );
        }
    }

    /// Degenerate inputs must not divide by zero or return a zero count — a
    /// zero would reach `mh_steps` as "propose nothing".
    #[test]
    fn auto_n_mh_steps_is_safe_on_degenerate_shapes() {
        for (n_obs, n_subjects, n_eta) in [(0, 0, 0), (0, 10, 2), (10, 0, 2), (10, 10, 0)] {
            let n = auto_n_mh_steps(n_obs, n_subjects, n_eta);
            assert!(
                (AUTO_MH_STEPS_MIN..=AUTO_MH_STEPS_MAX).contains(&n),
                "auto count out of range on ({n_obs}, {n_subjects}, {n_eta}): {n}"
            );
        }
    }

    /// An explicit count is passed through untouched — including one that
    /// equals neither clamp, and one the rule would never produce.
    #[test]
    fn resolve_n_mh_steps_passes_an_explicit_count_through() {
        // The sparse shape whose auto count is the floor.
        let (n_obs, n_subj, n_eta) = (274, 100, 3);
        assert_eq!(
            resolve_n_mh_steps(SAEM_N_MH_STEPS_AUTO, n_obs, n_subj, n_eta),
            AUTO_MH_STEPS_MIN,
            "the sentinel must resolve"
        );
        for requested in [1, 3, 40, 200] {
            assert_eq!(
                resolve_n_mh_steps(requested, n_obs, n_subj, n_eta),
                requested,
                "an explicit count must survive the resolver"
            );
        }
    }

    /// The Bayes η block resolves `auto` to the historical fixed count, *not*
    /// to the rule — on a shape where the two visibly differ, so the test reads
    /// "does not consult the rule" rather than "agrees with it here".
    #[test]
    fn resolve_n_mh_steps_bayes_keeps_the_historical_count() {
        // A sparse shape the SAEM rule answers with the floor.
        let (n_obs, n_subj, n_eta) = (274, 100, 3);
        assert_eq!(
            resolve_n_mh_steps(SAEM_N_MH_STEPS_AUTO, n_obs, n_subj, n_eta),
            AUTO_MH_STEPS_MIN,
            "precondition: the SAEM rule gives this shape the floor, not the cap"
        );
        assert_ne!(
            AUTO_MH_STEPS_MIN, AUTO_MH_STEPS_MAX,
            "precondition: the two answers are distinguishable"
        );
        assert_eq!(
            resolve_n_mh_steps_bayes(SAEM_N_MH_STEPS_AUTO),
            AUTO_MH_STEPS_MAX,
            "Bayes keeps the historical count under `auto` (see the fn docs)"
        );
        for requested in [1, 6, 40] {
            assert_eq!(
                resolve_n_mh_steps_bayes(requested),
                requested,
                "an explicit count applies to Bayes too"
            );
        }
    }

    /// The componentwise sweep count, including the floor that keeps the
    /// anti-collapse kernel alive at a small block count (#191) and the
    /// single-η case where the kernel is skipped entirely.
    #[test]
    fn componentwise_sweeps_floor_and_single_eta() {
        assert_eq!(componentwise_sweeps(20, 3), 6, "20/3 = 6 sweeps");
        assert_eq!(componentwise_sweeps(20, 2), 10);
        assert_eq!(
            componentwise_sweeps(6, 3),
            2,
            "6/3 = 2, which is also the floor"
        );
        assert_eq!(
            componentwise_sweeps(6, 4),
            2,
            "6/4 = 1 must be lifted to the floor, not left to vanish"
        );
        assert_eq!(
            componentwise_sweeps(0, 4),
            2,
            "even a zero block count keeps the anti-collapse kernel"
        );
        assert_eq!(
            componentwise_sweeps(20, 1),
            0,
            "single-η models skip the componentwise kernel"
        );
    }

    /// The verbose line says which of the two sources the count came from.
    #[test]
    fn mh_steps_report_names_its_source() {
        let auto = mh_steps_report(SAEM_N_MH_STEPS_AUTO, 6, 2);
        assert!(auto.contains("6 block MH proposals"), "got: {auto}");
        assert!(auto.contains("auto"), "got: {auto}");
        assert!(auto.contains("2 componentwise sweeps"), "got: {auto}");

        let explicit = mh_steps_report(20, 20, 6);
        assert!(explicit.contains("set in fit options"), "got: {explicit}");
        assert!(!explicit.contains("auto"), "got: {explicit}");
    }
}

#[cfg(test)]
#[path = "saem_hotpath_tests.rs"]
mod hotpath_tests;
