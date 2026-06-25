//! Tier-3 convergence + simulation-estimation (SSE) tests for Phase 1 TTE.
//!
//! Gated behind BOTH `survival` (the TTE feature) and `slow-tests` (these run a
//! full fit to convergence, so they are skipped on the per-PR `Test` job and run
//! nightly via `slow-tests.yml` — see the test-tier rules in CLAUDE.md).
//!
//! Two kinds of guard:
//!
//!   * **SSE** (`tte_sse_*`): generate a dataset from known `(theta, Omega)` with
//!     ferx's *own* `simulate()`, refit with ferx, and assert parameter recovery.
//!     This is the primary guard on the generative path (plan §14.11): a wrong
//!     event-time sampler or a wrong likelihood constant shows up here as
//!     non-recovery, which a fit-only test cannot detect.
//!
//!   * **Cross-tool** (`tte_convergence_*`): fit the committed reference dataset
//!     `tests/reference/tte_exponential/tte_exp.csv` — the *same* file NONMEM and
//!     nlmixr2 fit — through the real datareader (`read_population_for`, so the
//!     DV→`obs_records` routing is exercised too) and assert the estimates land in
//!     the documented bands. The fixed-effects (`n_eta = 0`) fit is checked
//!     *exactly* against the base-R `survival::survreg` closed-form MLE
//!     (`lambda = events / sum(time)`); the mixed-effects fit must instead recover
//!     the data-generating `lambda_pop = 0.1`, `omega^2 = 0.25` (the pooled
//!     fixed-effects rate is biased low by the ignored between-subject variance —
//!     which is the whole point of fitting the mixed model).

#![cfg(all(feature = "survival", feature = "slow-tests"))]

mod common;

use ferx_core::api::read_population_for;
use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::Population;
use ferx_core::{fit, simulate_with_seed, FitOptions, SimOutcome, SimulationResult};

// ── Model strings ────────────────────────────────────────────────────────────

/// Data-generating ("truth") model used by the SSE test: lambda_pop = 0.1,
/// omega^2 = 0.25 on log(lambda). `scale` is the Exponential rate.
const EXP_TRUTH: &str = r"
[parameters]
  theta TVLAMBDA(0.1, 0.001, 10.0)
  omega ETA_LAMBDA ~ 0.25

[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA * exp(ETA_LAMBDA)
";

/// Mixed-effects fit model, initialised *away* from the truth so recovery is a
/// real test (rate 2x low, variance ~3x low).
const EXP_FIT: &str = r"
[parameters]
  theta TVLAMBDA(0.05, 0.001, 10.0)
  omega ETA_LAMBDA ~ 0.09

[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA * exp(ETA_LAMBDA)
";

/// Fixed-effects (n_eta = 0) fit model — ordinary parametric Exponential PH, the
/// plain-likelihood special case (plan D7). Anchored against `survreg`.
const EXP_FIT_FIXED: &str = r"
[parameters]
  theta TVLAMBDA(0.05, 0.001, 10.0)

[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA
";

/// Competing-risks "truth" model: two exponential causes (CMT 2, CMT 3) linked
/// by a shared log-frailty `ETA_F` (ω²=0.25). The cause-specific *rates* are
/// well-identified and recover tightly; the shared frailty ω² is weakly
/// identified and reads high under FOCEI-Laplace (#440/#469) — see the test's
/// band comment.
const COMPETING_TRUTH: &str = r"
[parameters]
  theta TVLAMBDA_A(0.10, 0.001, 10.0)
  theta TVLAMBDA_B(0.06, 0.001, 10.0)
  omega ETA_F ~ 0.25

[event_model cause_a]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA_A * exp(ETA_F)

[event_model cause_b]
  cmt    = 3
  family = exponential
  scale  = TVLAMBDA_B * exp(ETA_F)
";

/// Competing-risks fit model, initialised away from the truth.
const COMPETING_FIT: &str = r"
[parameters]
  theta TVLAMBDA_A(0.05, 0.001, 10.0)
  theta TVLAMBDA_B(0.03, 0.001, 10.0)
  omega ETA_F ~ 0.09

[event_model cause_a]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA_A * exp(ETA_F)

[event_model cause_b]
  cmt    = 3
  family = exponential
  scale  = TVLAMBDA_B * exp(ETA_F)
";

/// Fixed-effects (no frailty) competing-risks model — anchors the cause-specific
/// likelihood against the per-cause closed-form / `survreg` MLE.
const COMPETING_FIXED: &str = r"
[parameters]
  theta TVLAMBDA_A(0.05, 0.001, 10.0)
  theta TVLAMBDA_B(0.03, 0.001, 10.0)

[event_model cause_a]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA_A

[event_model cause_b]
  cmt    = 3
  family = exponential
  scale  = TVLAMBDA_B
";

// ── Helpers ──────────────────────────────────────────────────────────────────

fn fit_opts() -> FitOptions {
    // Default is already FOCEI / outer_maxiter = 500 / interaction = true — i.e.
    // run-to-convergence. We only quiet it and request the covariance step so the
    // comparison table can report SEs.
    FitOptions {
        verbose: false,
        run_covariance_step: true,
        ..FitOptions::default()
    }
}

/// `n` bare TTE subjects (one placeholder Event each on CMT 2) used only as a
/// `simulate()` template. Each placeholder carries the administrative censoring
/// window `censor` in its record `time`; `simulate_tte` right-censors any draw
/// that reaches it (the drawn event time otherwise replaces the placeholder).
/// This is exactly [`common::tte_pop_from_pairs`] over `n` right-censored rows at
/// `t = censor`.
fn tte_sim_template(n: usize, censor: f64) -> Population {
    common::tte_pop_from_pairs(&vec![(censor, 0); n])
}

/// Map ferx `simulate()` outcomes to `(time, dv)` pairs. `simulate_tte` already
/// applies administrative right-censoring at each subject's observation window
/// (set by [`tte_sim_template`]), so this just reads the `observed` flag: an
/// observed draw is an event row `(time, 1)`, a censored draw a `(window, 0)` row.
/// Panics on any non-`Event` outcome — that would mean a Gaussian model/template
/// was simulated by mistake.
fn sims_to_pairs(sims: &[SimulationResult]) -> Vec<(f64, u8)> {
    sims.iter()
        .map(|r| match r.outcome {
            SimOutcome::Event { time, observed } => (time, observed as u8),
            _ => panic!("expected an Event outcome for a TTE simulation"),
        })
        .collect()
}

/// Rebuild a competing-risks fit population from `simulate()` outcomes: group the
/// per-cause rows by subject and turn each into an `ObsRecord::Event` (exact when
/// observed, right-censored otherwise) on its CMT. `BTreeMap` keeps subject order
/// deterministic.
fn competing_pop_from_sims(sims: &[SimulationResult]) -> Population {
    use ferx_core::types::{EventType, ObsRecord};
    use std::collections::BTreeMap;

    let mut by_id: BTreeMap<String, Vec<(usize, f64, bool)>> = BTreeMap::new();
    for r in sims {
        match &r.outcome {
            SimOutcome::Event { time, observed } => {
                by_id
                    .entry(r.id.clone())
                    .or_default()
                    .push((r.cmt, *time, *observed));
            }
            _ => panic!("expected an Event outcome for a competing-risks simulation"),
        }
    }

    let subjects = by_id
        .into_iter()
        .map(|(id, recs)| {
            let mut s = common::subject(&id, vec![], vec![], vec![], vec![]);
            s.obs_records = recs
                .into_iter()
                .map(|(cmt, time, observed)| ObsRecord::Event {
                    time,
                    event_type: if observed {
                        EventType::Exact
                    } else {
                        EventType::RightCensored
                    },
                    entry_time: 0.0,
                    cmt,
                })
                .collect();
            s
        })
        .collect();

    Population {
        subjects,
        covariate_names: vec![],
        dv_column: "DV".to_string(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

const REF_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/reference/tte_exponential/tte_exp.csv"
);

/// `survival::survreg(Surv(TIME,DV) ~ 1, dist="exponential")` on `tte_exp.csv`.
/// Closed form `lambda = events / sum(time) = 82 / 1100.6`. Regenerate via
/// `tests/reference/tte_exponential/survreg.R`.
const SURVREG_LAMBDA: f64 = 0.074506;

/// `-2 logLik` reported by that same `survreg` exponential fit (printed by
/// `survreg.R`, tabulated in `expected.md`). The MLE argmax is invariant to a
/// dropped or doubled censored-likelihood normalizing constant, so matching
/// `lambda` alone cannot catch such a regression — the OFV must be anchored too
/// (the headline "likelihood constants are correct" rests on this number).
const SURVREG_LAMBDA_M2LL: f64 = 589.888;

// ── Tests ────────────────────────────────────────────────────────────────────

/// SSE: simulate from (lambda_pop=0.1, omega^2=0.25) with ferx's own `simulate()`,
/// apply administrative censoring at t=24 (matching the reference design), refit,
/// and assert both parameters are recovered. Guards the event-time sampler and the
/// censored-likelihood together.
#[test]
fn tte_sse_exponential_recovers_truth() {
    const N: usize = 2000;
    const T_CENSOR: f64 = 24.0;
    const SEED: u64 = 20260621;

    let truth = parse_model_string(EXP_TRUTH).expect("truth model must parse");
    let template = tte_sim_template(N, T_CENSOR);

    let sims = simulate_with_seed(&truth, &template, &truth.default_params, 1, SEED);
    assert_eq!(sims.len(), N, "one simulated event per template subject");

    let pairs = sims_to_pairs(&sims);

    let event_frac = pairs.iter().filter(|(_, dv)| *dv == 1).count() as f64 / N as f64;
    eprintln!("[SSE] event fraction = {event_frac:.4} (expected ~0.88 at lambda_pop=0.1, omega^2=0.25, censor t=24)");
    // At the truth (lambda_pop=0.1, omega^2=0.25, censor t=24) this N=2000 draw yields
    // ~0.88 events (the noisier 100-subject reference file lands at 0.82). The band
    // brackets that yet *excludes* a factor-of-2 sampler error, which the old
    // (0.5..0.98) did not: halving the rate (lambda~0.05) drops the fraction to ~0.66
    // (below 0.82) and doubling it pushes ~0.97 (above 0.93). The simulated data is
    // RNG-seed-deterministic, so this stays tight without being flaky.
    assert!(
        (0.82..0.93).contains(&event_frac),
        "simulated event fraction {event_frac:.4} off the expected ~0.88 — a 2x sampler error (rate halved → ~0.66, doubled → ~0.97) lands outside this band"
    );

    let model = parse_model_string(EXP_FIT).expect("fit model must parse");
    let pop = common::tte_pop_from_pairs(&pairs);
    let r = fit(&model, &pop, &model.default_params, &fit_opts()).expect("SSE fit must succeed");

    let lambda = r.theta[0];
    let omega2 = r.omega[(0, 0)];
    eprintln!("[SSE] lambda_pop = {lambda:.5} (truth 0.1), omega^2 = {omega2:.5} (truth 0.25), OFV = {:.4}", r.ofv);

    // N is large enough that single-replicate Monte-Carlo noise is small, so these
    // bands test the *estimator* (near-unbiased recovery), not luck. Observed
    // (seed-fixed, deterministic): lambda ~= 0.099, omega^2 ~= 0.232 — the ~7%
    // omega^2 shortfall is the expected mild FOCEI-Laplace bias for TTE at this
    // event rate (plan §3.3), and grows at lower event rates. Bands are wide
    // enough to absorb that bias yet catch a gross sampler/likelihood error
    // (wrong constant, sign flip, factor-of-2).
    assert!(
        (0.090..0.110).contains(&lambda),
        "lambda_pop not recovered: got {lambda:.5}, expected ~0.1"
    );
    assert!(
        (0.200..0.275).contains(&omega2),
        "omega^2 not recovered: got {omega2:.5}, expected ~0.25 (mild Laplace bias allowed)"
    );
    assert!(r.ofv.is_finite(), "OFV must be finite");
}

/// SSE for **competing risks**: simulate two cause-specific exponential hazards
/// (CMT 2, CMT 3) with a shared frailty via ferx's own `simulate()` (earliest
/// cause wins, the other censored), refit, and assert both cause-specific rates
/// and the frailty are recovered. Guards the competing-risks generative path
/// (earliest-of-causes + per-cause censoring) and the cause-specific likelihood
/// together. Linear (rate) frailty ⇒ FOCEI is near-unbiased (cf. the nonlinear
/// shape-frailty bias tracked in #469/#440).
#[test]
fn tte_sse_competing_risks_recovers_truth() {
    const N: usize = 2000;
    const T_CENSOR: f64 = 14.0;
    const SEED: u64 = 20260624;

    let truth = parse_model_string(COMPETING_TRUTH).expect("competing truth model must parse");
    let template = common::tte_competing_pop(&vec![(T_CENSOR, 0u8); N]);

    let sims = simulate_with_seed(&truth, &template, &truth.default_params, 1, SEED);
    assert_eq!(
        sims.len(),
        2 * N,
        "two rows per subject (one per cause CMT)"
    );

    // Cause-specific observed-event fractions (a subject contributes to at most one).
    let (mut f2, mut f3) = (0usize, 0usize);
    for r in &sims {
        if let SimOutcome::Event { observed: true, .. } = r.outcome {
            if r.cmt == 2 {
                f2 += 1;
            } else {
                f3 += 1;
            }
        }
    }
    let (frac2, frac3) = (f2 as f64 / N as f64, f3 as f64 / N as f64);
    eprintln!("[SSE-CR] cause-2 event frac = {frac2:.4}, cause-3 event frac = {frac3:.4}");
    // λ_A:λ_B = 0.10:0.06 ⇒ cause 2 (higher rate) produces more events; with τ=14 the
    // all-cause event rate is high, so both causes are well-populated and some subjects
    // are administratively censored.
    assert!(
        frac2 > frac3,
        "cause 2 (higher rate) must produce more events"
    );
    assert!(frac2 > 0.05 && frac3 > 0.05, "both causes must fire");

    let model = parse_model_string(COMPETING_FIT).expect("competing fit model must parse");
    let pop = competing_pop_from_sims(&sims);
    assert_eq!(
        pop.subjects.len(),
        N,
        "one fit subject per simulated subject"
    );

    let r = fit(&model, &pop, &model.default_params, &fit_opts()).expect("SSE fit must succeed");
    let (la, lb, om) = (r.theta[0], r.theta[1], r.omega[(0, 0)]);
    eprintln!(
        "[SSE-CR] lambda_A = {la:.5} (0.10), lambda_B = {lb:.5} (0.06), omega^2 = {om:.5} (0.25), OFV = {:.4}",
        r.ofv
    );

    // The cause-specific RATES are well-identified — assert tight recovery. These
    // are the headline competing-risks guard: a wrong cause→CMT routing, a dropped
    // censoring row, or a factor-of-2 in either hazard breaks them. (Deterministic
    // under RAYON_NUM_THREADS=1, as CI pins; seed-fixed values λ_A≈0.098, λ_B≈0.061.)
    assert!(
        (0.090..0.110).contains(&la),
        "lambda_A not recovered: {la:.5}"
    );
    assert!(
        (0.052..0.070).contains(&lb),
        "lambda_B not recovered: {lb:.5}"
    );

    // The SHARED frailty ω² is weakly identified (the −2LL is flat over a wide ω²
    // range — #469) AND over-estimated by FOCEI-Laplace at this all-cause event
    // rate (#440): a seed sweep gives 0.28–0.41 (truth 0.25), seed-20260624 ≈ 0.41.
    // This mirrors the single-cause exp(λ_tot) frailty problem to which competing
    // risks reduces, so it is NOT a competing-risks bug — the clean λ recovery
    // above is the proof. The band brackets the observed value with margin and
    // EXCLUDES the truth, so it characterises the bias and would fail (prompting a
    // re-baseline) if the Laplace frailty estimate is ever fixed to ~0.25.
    assert!(
        (0.32..0.50).contains(&om),
        "omega^2 {om:.5}: expected the FOCEI over-estimate ~0.41 (truth 0.25, weakly identified). \
         Below 0.32 may mean #440/#469 improved (FOCEI now nearer 0.25) — re-baseline this band."
    );
    assert!(r.ofv.is_finite(), "OFV must be finite");
}

/// Cross-tool: mixed-effects FOCEI fit of the committed reference dataset. Must
/// recover the data-generating parameters (this is the row NONMEM/nlmixr2 fill).
#[test]
fn tte_convergence_exponential_mixed() {
    let model = parse_model_string(EXP_FIT).expect("fit model must parse");
    let (pop, _cov) =
        read_population_for(&model, &None, REF_CSV, None, None, None).expect("reference CSV reads");
    assert_eq!(pop.subjects.len(), 100, "reference dataset is 100 subjects");

    let r = fit(&model, &pop, &model.default_params, &fit_opts()).expect("fit must succeed");

    let lambda = r.theta[0];
    let omega2 = r.omega[(0, 0)];
    eprintln!(
        "[mixed] lambda_pop = {lambda:.5} (truth 0.1), omega^2 = {omega2:.5} (truth 0.25), OFV = {:.4}, converged = {}",
        r.ofv, r.converged
    );

    // Single 100-subject realisation (seed 42) is noisy, so the exact estimates
    // (deterministic: lambda ~= 0.0768, omega^2 ~= 0.290, OFV ~= 588.93) are what
    // gets tabulated next to NONMEM/nlmixr2 in
    // `tests/reference/tte_exponential/expected.md` — every tool fits *this* file,
    // so they should agree with each other regardless of the 0.1/0.25 truth. The
    // bands here are only a gross-failure guard.
    assert!(
        (0.06..0.14).contains(&lambda),
        "lambda_pop off: got {lambda:.5}, expected ~0.1"
    );
    assert!(
        (0.12..0.50).contains(&omega2),
        "omega^2 off: got {omega2:.5}, expected ~0.25"
    );
    assert!(r.ofv.is_finite(), "OFV must be finite");
}

/// Cross-tool, exact: fixed-effects (n_eta=0) Exponential MLE must match the
/// base-R `survreg` closed-form rate on the same dataset to a tight tolerance.
#[test]
fn tte_convergence_exponential_fixed_matches_survreg() {
    let model = parse_model_string(EXP_FIT_FIXED).expect("fixed-effects model must parse");
    assert_eq!(model.n_eta, 0, "fixed-effects model must have no etas");

    let (pop, _cov) =
        read_population_for(&model, &None, REF_CSV, None, None, None).expect("reference CSV reads");

    let r = fit(&model, &pop, &model.default_params, &fit_opts()).expect("fit must succeed");
    let lambda = r.theta[0];
    eprintln!(
        "[fixed] lambda = {lambda:.6} (survreg {SURVREG_LAMBDA:.6}), OFV = {:.4}",
        r.ofv
    );

    let rel_err = (lambda - SURVREG_LAMBDA).abs() / SURVREG_LAMBDA;
    assert!(
        rel_err < 0.01,
        "fixed-effects rate {lambda:.6} must match survreg {SURVREG_LAMBDA:.6} within 1% (rel_err {rel_err:.4})"
    );
    // Anchor the OFV to survreg's -2logLik too (not just `is_finite`): this is the
    // number that pins the likelihood *constants*, which `lambda` cannot (#441 #2).
    assert!(
        (r.ofv - SURVREG_LAMBDA_M2LL).abs() < 1e-3,
        "fixed-effects OFV {:.4} must match survreg -2logLik {SURVREG_LAMBDA_M2LL} within 1e-3",
        r.ofv
    );
}

/// Cross-tool, exact: a FIXED-EFFECTS (n_eta=0) competing-risks exponential fit
/// must recover each cause's closed-form MLE `λ̂_k = d_k / Σ_i t_i` — identical to
/// base-R `survreg(dist="exponential")` fitted per cause (other-cause events as
/// censoring). Without a shared frailty the cause-specific likelihood factorises
/// into independent per-cause exponential regressions, so this anchors the
/// cause-specific hazard likelihood against an external reference (the
/// competing-risks analogue of `tte_convergence_exponential_fixed_matches_survreg`).
#[test]
fn tte_competing_fixed_matches_per_cause_mle() {
    // data/tte_competing_risks.csv: N=40, d_A(CMT2)=23, d_B(CMT3)=15, Σt=201.6552
    //   ⇒ λ̂_A = 23/201.6552 = 0.114056,  λ̂_B = 15/201.6552 = 0.074384
    //   (verified equal to survreg(Surv(time, cause==k) ~ 1, dist="exponential")).
    const MLE_A: f64 = 0.114056;
    const MLE_B: f64 = 0.074384;
    const COMPETING_CSV: &str =
        concat!(env!("CARGO_MANIFEST_DIR"), "/data/tte_competing_risks.csv");

    let model = parse_model_string(COMPETING_FIXED).expect("fixed competing model must parse");
    assert_eq!(model.n_eta, 0, "fixed-effects model must have no etas");

    let (pop, _cov) = read_population_for(&model, &None, COMPETING_CSV, None, None, None)
        .expect("competing CSV reads");

    let r = fit(&model, &pop, &model.default_params, &fit_opts()).expect("fixed competing fit");
    let (la, lb) = (r.theta[0], r.theta[1]);
    eprintln!(
        "[CR-fixed] lambda_A = {la:.6} (MLE {MLE_A:.6}), lambda_B = {lb:.6} (MLE {MLE_B:.6}), OFV = {:.4}",
        r.ofv
    );

    let rel_a = (la - MLE_A).abs() / MLE_A;
    let rel_b = (lb - MLE_B).abs() / MLE_B;
    assert!(
        rel_a < 0.01,
        "lambda_A {la:.6} must match per-cause MLE {MLE_A:.6} within 1% (rel {rel_a:.4})"
    );
    assert!(
        rel_b < 0.01,
        "lambda_B {lb:.6} must match per-cause MLE {MLE_B:.6} within 1% (rel {rel_b:.4})"
    );
    assert!(r.ofv.is_finite(), "OFV must be finite");
}

// ═══════════════════════════ Weibull ═══════════════════════════════════════════
//
// Validation dataset puts between-subject variability on the *shape* (scale fixed):
// scale_pop = 20, shape_pop = 2, omega^2(log shape) = 0.20, censored at t = 30.

const WEIBULL_TRUTH: &str = r"
[parameters]
  theta TVSCALE(20.0, 0.1, 500.0)
  theta TVSHAPE(2.0,  0.1, 10.0)
  omega ETA_SHAPE ~ 0.20

[event_model]
  cmt    = 2
  family = weibull
  scale  = TVSCALE
  shape  = TVSHAPE * exp(ETA_SHAPE)
";

const WEIBULL_FIT: &str = r"
[parameters]
  theta TVSCALE(15.0, 0.1, 500.0)
  theta TVSHAPE(1.2,  0.1, 10.0)
  omega ETA_SHAPE ~ 0.05

[event_model]
  cmt    = 2
  family = weibull
  scale  = TVSCALE
  shape  = TVSHAPE * exp(ETA_SHAPE)
";

const WEIBULL_FIT_FIXED: &str = r"
[parameters]
  theta TVSCALE(15.0, 0.1, 500.0)
  theta TVSHAPE(1.2,  0.1, 10.0)

[event_model]
  cmt    = 2
  family = weibull
  scale  = TVSCALE
  shape  = TVSHAPE
";

const WEIBULL_REF_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/reference/tte_weibull/tte_weibull.csv"
);

// survreg(Surv(TIME,DV)~1, dist="weibull") on tte_weibull.csv, mapped to ferx's
// (scale, shape) via shape = 1/scale_sr, scale = exp(intercept). See survreg.R.
const SURVREG_WEIBULL_SHAPE: f64 = 2.119250;
const SURVREG_WEIBULL_SCALE: f64 = 22.176599;
/// `-2 logLik` of the same `survreg` weibull fit (see `survreg.R` / `expected.md`).
/// Anchors the fixed-effects OFV so the likelihood constants are pinned, not just
/// the (scale, shape) argmax.
const SURVREG_WEIBULL_M2LL: f64 = 640.261;

/// SSE: recover (scale=20, shape=2, omega^2=0.20) from a large simulated dataset.
#[test]
fn tte_sse_weibull_recovers_truth() {
    const N: usize = 2000;
    const T_CENSOR: f64 = 30.0;
    const SEED: u64 = 20260622;

    let truth = parse_model_string(WEIBULL_TRUTH).expect("truth model must parse");
    let sims = simulate_with_seed(
        &truth,
        &tte_sim_template(N, T_CENSOR),
        &truth.default_params,
        1,
        SEED,
    );

    let pairs = sims_to_pairs(&sims);

    let model = parse_model_string(WEIBULL_FIT).expect("fit model must parse");
    let r = fit(
        &model,
        &common::tte_pop_from_pairs(&pairs),
        &model.default_params,
        &fit_opts(),
    )
    .expect("SSE fit must succeed");

    let scale = r.theta[0];
    let shape = r.theta[1];
    let omega2 = r.omega[(0, 0)];
    eprintln!(
        "[SSE weibull] scale = {scale:.4} (truth 20), shape = {shape:.4} (truth 2), omega^2 = {omega2:.4} (truth 0.20)"
    );

    // Structural parameters recover tightly under FOCEI.
    assert!(
        (18.5..21.5).contains(&scale),
        "scale not recovered: got {scale:.4}, expected ~20"
    );
    assert!(
        (1.85..2.15).contains(&shape),
        "shape not recovered: got {shape:.4}, expected ~2"
    );
    // omega^2 (frailty on the *shape* — a nonlinear hazard parameter) is materially
    // OVER-estimated by FOCEI-Laplace: this seed deterministically gives ~0.344 vs a
    // true 0.20 (+72%), and the bias does NOT vanish as the true omega^2 shrinks
    // (truth 0.05 → ~0.16). A SAEM fit of the same data reads ~0.13 — the estimators
    // straddle the truth, confirming a FOCEI *approximation* limitation for
    // nonlinear-parameter frailty (plan §3.3/§13: SAEM/IMP preferred for TTE), NOT a
    // likelihood bug (the fixed-effects fit matches survreg exactly). Tracked in #440.
    //
    // This band CHARACTERIZES that bias on purpose: it brackets the current biased
    // value (~0.344) with platform margin but EXCLUDES both the truth (0.20) and the
    // SAEM value (0.13). So when #440 is resolved and FOCEI recovers omega^2 ~ 0.20,
    // this assertion will FAIL — the intended signal to revisit it (assert recovery to
    // the truth and drop the over-estimation note). A wide "sane-range" band that
    // admitted both the broken and the fixed value could never flip, letting #440 close
    // by attrition with no test ever signalling (#441 review #4).
    assert!(
        (0.29..0.40).contains(&omega2),
        "omega^2 {omega2:.4}: expected the documented FOCEI over-estimate ~0.34 (truth 0.20). \
         Below 0.29 likely means #440 is fixed (FOCEI now recovers ~0.20) — update this test; \
         above 0.40 means the shape frailty exploded."
    );
    assert!(r.ofv.is_finite(), "OFV must be finite");
}

/// Cross-tool, exact: fixed-effects (n_eta=0) Weibull MLE must match `survreg`
/// (both scale and shape) on the same dataset.
#[test]
fn tte_convergence_weibull_fixed_matches_survreg() {
    let model = parse_model_string(WEIBULL_FIT_FIXED).expect("fixed-effects model must parse");
    assert_eq!(model.n_eta, 0, "fixed-effects model must have no etas");

    let (pop, _cov) = read_population_for(&model, &None, WEIBULL_REF_CSV, None, None, None)
        .expect("reference CSV reads");
    let r = fit(&model, &pop, &model.default_params, &fit_opts()).expect("fit must succeed");

    let scale = r.theta[0];
    let shape = r.theta[1];
    eprintln!(
        "[fixed weibull] scale = {scale:.4} (survreg {SURVREG_WEIBULL_SCALE:.4}), shape = {shape:.4} (survreg {SURVREG_WEIBULL_SHAPE:.4}), OFV = {:.4}",
        r.ofv
    );

    let scale_err = (scale - SURVREG_WEIBULL_SCALE).abs() / SURVREG_WEIBULL_SCALE;
    let shape_err = (shape - SURVREG_WEIBULL_SHAPE).abs() / SURVREG_WEIBULL_SHAPE;
    assert!(
        scale_err < 0.01,
        "scale {scale:.4} must match survreg {SURVREG_WEIBULL_SCALE:.4} within 1% (rel_err {scale_err:.4})"
    );
    assert!(
        shape_err < 0.01,
        "shape {shape:.4} must match survreg {SURVREG_WEIBULL_SHAPE:.4} within 1% (rel_err {shape_err:.4})"
    );
    // Pin the OFV to survreg's -2logLik (the likelihood-constants anchor, #441 #2).
    assert!(
        (r.ofv - SURVREG_WEIBULL_M2LL).abs() < 1e-3,
        "fixed-effects OFV {:.4} must match survreg -2logLik {SURVREG_WEIBULL_M2LL} within 1e-3",
        r.ofv
    );
}

/// Cross-tool: mixed-effects (frailty-on-shape) FOCEI fit of the committed Weibull
/// dataset — the row NONMEM/nlmixr2 fill. Records ferx's estimates; the FOCEI
/// over-estimation of the shape-frailty omega^2 (#440) shows up here on real data too.
#[test]
fn tte_convergence_weibull_mixed() {
    let model = parse_model_string(WEIBULL_FIT).expect("fit model must parse");
    let (pop, _cov) = read_population_for(&model, &None, WEIBULL_REF_CSV, None, None, None)
        .expect("reference CSV reads");
    assert_eq!(pop.subjects.len(), 100, "reference dataset is 100 subjects");

    let r = fit(&model, &pop, &model.default_params, &fit_opts()).expect("fit must succeed");
    let scale = r.theta[0];
    let shape = r.theta[1];
    let omega2 = r.omega[(0, 0)];
    eprintln!(
        "[weibull mixed] scale = {scale:.4} (truth 20), shape = {shape:.4} (truth 2), omega^2 = {omega2:.4} (truth 0.20), OFV = {:.4}",
        r.ofv
    );

    // Gross-failure guard; exact estimates are tabulated in expected.md.
    assert!(
        (15.0..26.0).contains(&scale),
        "scale off: got {scale:.4}, expected ~20"
    );
    assert!(
        (1.5..2.8).contains(&shape),
        "shape off: got {shape:.4}, expected ~2"
    );
    // On this single n=100 realisation ferx's shape-frailty omega^2 is ~0.204 — at the
    // truth, because the FOCEI over-estimation that the large-N SSE isolates (~0.34) is
    // masked by single-realisation noise here; nlmixr2 reads 0.173 on the same file
    // (the cross-tool spread is itself #440 evidence). Narrow characterization band
    // around ferx's deterministic value — the previous (0.05..0.70) was far too wide to
    // catch a regression (#441 review #4). The bias itself is tracked by the SSE test
    // above, not here.
    assert!(
        (0.17..0.24).contains(&omega2),
        "omega^2 {omega2:.4} off ferx's documented ~0.204 on this file (truth 0.20, nlmixr2 0.173); see expected.md / #440"
    );
    assert!(r.ofv.is_finite(), "OFV must be finite");
}

// ═══════════════════════════ Gompertz ══════════════════════════════════════════
//
// Validation dataset is a fixed-effects 2-arm RCT: h = alpha*exp(gamma*t)*exp(loghr*TRT),
// alpha = exp(-6) ≈ 0.00248, gamma = exp(-5.4) ≈ 0.00450, loghr = -0.8, censored at 365.
// No survreg/base-R anchor exists for Gompertz, so recovery is the guard (NONMEM/nlmixr2
// are the cross-tool hand-off). The fit exercises the [event_model] covariate + loghr path.

const GOMPERTZ_FIT: &str = r"
[parameters]
  theta TVALPHA(0.001, 1e-6, 1.0)
  theta TVGAMMA(0.003, 1e-5, 5.0)
  theta LHR(-0.3, -5.0, 5.0)

[event_model]
  cmt    = 2
  family = gompertz
  alpha  = TVALPHA
  gamma  = TVGAMMA
  loghr  = LHR * TRT
";

const GOMPERTZ_REF_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/reference/tte_gompertz/tte_gompertz.csv"
);

/// Frailty-Gompertz SSE truth (BSV on gamma) — recover alpha, gamma, omega^2.
const GOMPERTZ_TRUTH_FRAILTY: &str = r"
[parameters]
  theta TVALPHA(0.002, 1e-6, 1.0)
  theta TVGAMMA(0.05,  1e-5, 5.0)
  omega ETA_GAMMA ~ 0.05

[event_model]
  cmt    = 2
  family = gompertz
  alpha  = TVALPHA
  gamma  = TVGAMMA * exp(ETA_GAMMA)
";

const GOMPERTZ_FIT_FRAILTY: &str = r"
[parameters]
  theta TVALPHA(0.001, 1e-6, 1.0)
  theta TVGAMMA(0.03,  1e-5, 5.0)
  omega ETA_GAMMA ~ 0.02

[event_model]
  cmt    = 2
  family = gompertz
  alpha  = TVALPHA
  gamma  = TVGAMMA * exp(ETA_GAMMA)
";

/// Cross-tool: fixed-effects Gompertz RCT fit on the committed dataset. Recovers
/// the data-generating alpha/gamma/loghr and exercises the TRT covariate path.
/// (NONMEM/nlmixr2 columns are the hand-off.)
#[test]
fn tte_convergence_gompertz_rct_recovers() {
    let model = parse_model_string(GOMPERTZ_FIT).expect("gompertz model must parse");
    assert_eq!(model.n_eta, 0, "RCT model is fixed-effects");
    assert!(
        model.referenced_covariates.contains(&"TRT".to_string()),
        "TRT must be picked up from the loghr expression"
    );

    let (pop, _cov) = read_population_for(&model, &None, GOMPERTZ_REF_CSV, None, None, None)
        .expect("reference CSV reads");
    assert_eq!(pop.subjects.len(), 300, "reference dataset is 300 subjects");

    let r = fit(&model, &pop, &model.default_params, &fit_opts()).expect("fit must succeed");
    let alpha = r.theta[0];
    let gamma = r.theta[1];
    let loghr = r.theta[2];
    eprintln!(
        "[gompertz RCT] alpha = {alpha:.6} (truth 0.00248), gamma = {gamma:.6} (truth 0.00450), loghr = {loghr:.4} (truth -0.8), OFV = {:.4}",
        r.ofv
    );

    assert!(
        (0.0012..0.0045).contains(&alpha),
        "alpha off: got {alpha:.6}, expected ~0.00248"
    );
    assert!(
        (0.0025..0.0070).contains(&gamma),
        "gamma off: got {gamma:.6}, expected ~0.00450"
    );
    assert!(
        (-1.3..-0.3).contains(&loghr),
        "loghr off: got {loghr:.4}, expected ~-0.8"
    );
    assert!(r.ofv.is_finite(), "OFV must be finite");
}

/// SSE: frailty-Gompertz (BSV on gamma) recovery of alpha/gamma/omega^2.
#[test]
fn tte_sse_gompertz_recovers_truth() {
    const N: usize = 2000;
    const T_CENSOR: f64 = 80.0;
    const SEED: u64 = 20260623;

    let truth = parse_model_string(GOMPERTZ_TRUTH_FRAILTY).expect("truth model must parse");
    let sims = simulate_with_seed(
        &truth,
        &tte_sim_template(N, T_CENSOR),
        &truth.default_params,
        1,
        SEED,
    );

    let pairs = sims_to_pairs(&sims);

    let model = parse_model_string(GOMPERTZ_FIT_FRAILTY).expect("fit model must parse");
    let r = fit(
        &model,
        &common::tte_pop_from_pairs(&pairs),
        &model.default_params,
        &fit_opts(),
    )
    .expect("SSE fit must succeed");

    let alpha = r.theta[0];
    let gamma = r.theta[1];
    let omega2 = r.omega[(0, 0)];
    eprintln!(
        "[SSE gompertz] alpha = {alpha:.5} (truth 0.002), gamma = {gamma:.5} (truth 0.05), omega^2 = {omega2:.4} (truth 0.05)"
    );

    // alpha and gamma trade off (the Gompertz baseline/growth collinearity), so
    // allow a wider alpha band; gamma is the better-determined of the two.
    assert!(
        (0.0012..0.0028).contains(&alpha),
        "alpha not recovered: got {alpha:.5}, expected ~0.002"
    );
    assert!(
        (0.043..0.060).contains(&gamma),
        "gamma not recovered: got {gamma:.5}, expected ~0.05"
    );
    // omega^2 (frailty on *gamma*, a nonlinear hazard parameter) is over-estimated by
    // FOCEI-Laplace — this seed deterministically gives ~0.081 vs truth 0.05 (+62%),
    // the same nonlinear-frailty limitation seen for the Weibull shape (#440). As for
    // the Weibull SSE, this is a CHARACTERIZATION band: it brackets the biased value
    // but EXCLUDES the truth (0.05), so it flips (fails) once #440 lets FOCEI recover
    // ~0.05 — the signal to revisit. Documented in expected.md.
    assert!(
        (0.065..0.105).contains(&omega2),
        "omega^2 {omega2:.4}: expected the documented FOCEI over-estimate ~0.08 (truth 0.05). \
         Below 0.065 likely means #440 is fixed (recovers ~0.05) — update this test; \
         above 0.105 means the gamma frailty exploded."
    );
    assert!(r.ofv.is_finite(), "OFV must be finite");
}
