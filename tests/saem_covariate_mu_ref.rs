//! Covariate (multi-theta) mu-referencing under SAEM and IMP (issue #619),
//! anchored to NONMEM.
//!
//! A typical value that reads several thetas — `CL = (TVCL + (CRCL-90)*TH_CRCL)
//! * exp(ETA_CL)`, `CL = TVCL * (WT/70)^TH_WT * exp(ETA_CL)` — has no single
//! mu-reference anchor, so before #619 its thetas sat on the eta-frozen numerical
//! M-step and a covariate slope stalled at its start (measured on these very
//! fixtures: `TH_CRCL` 0.0198 from a 0.02 start, `TH_WT` 0.332 from 0.3). ferx
//! now records a *covariate mu-reference* and re-fits the group's thetas to the
//! population of individual values each iteration — NONMEM's
//! `MU_1 = LOG(THETA(1) + (CRCL-90)*THETA(2))`.
//!
//! ## Fixtures
//!
//! `data/covmuref_additive.csv` and `data/covmuref_power.csv`: 1-cpt IV bolus,
//! 60 subjects × 7 samples, simulated from the respective model by
//! `nonmem_anchor/simulate_covmuref_data.py` (matched, well-specified fits). The
//! covariates are constant within each subject, so the exact Gauss–Newton engine
//! runs; the time-varying (prior + data) engine is exercised on the fluconazole
//! model in the PR write-up, which has no NONMEM SAEM comparator.
//!
//! ## NONMEM anchors
//!
//! `nonmem_anchor/covmuref_{additive,power}_saem.ctl`, NONMEM 7.5.1
//! `METHOD=SAEM` (NBURN=2000 NITER=1000 ISAMPLE=10 SEED=619) followed by an
//! `EONLY` IMP objective; final rows of `nonmem_anchor/results/*.ext`.
//!
//! Run the slow tests with:
//!
//!   cargo test --release --features slow-tests --test saem_covariate_mu_ref

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::MuTransform;
use ferx_core::types::SaemMstepSolver;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions, FitResult};
use std::path::Path;

/// NONMEM 7.5.1 SAEM finals, `nonmem_anchor/results/covmuref_additive_saem.ext`.
/// Columns: THETA1 (TVCL) THETA2 (TH_CRCL) THETA3 (TVV) SIGMA(1,1) OMEGA(1,1) OMEGA(2,2).
const NM_ADD_TVCL: f64 = 4.916_73;
const NM_ADD_TH_CRCL: f64 = 0.041_777_0;
const NM_ADD_TVV: f64 = 48.085_9;
const NM_ADD_OMEGA_CL: f64 = 0.040_098_0;
const NM_ADD_OMEGA_V: f64 = 0.036_046_6;

/// NONMEM 7.5.1 SAEM finals, `nonmem_anchor/results/covmuref_power_saem.ext`.
const NM_POW_TVCL: f64 = 4.787_63;
const NM_POW_TH_WT: f64 = 0.921_283;
const NM_POW_TVV: f64 = 49.516_1;
const NM_POW_OMEGA_CL: f64 = 0.069_657_2;
const NM_POW_OMEGA_V: f64 = 0.045_890_9;

/// Where the pre-#619 binary landed on the same fixtures and seed (release,
/// `worktree-918-logit-mu-ref`): the covariate theta never left its start.
const BEFORE_TH_CRCL: f64 = 0.019_768;
const BEFORE_TH_WT: f64 = 0.332_157;

fn load(model_file: &str, data_file: &str) -> (ferx_core::CompiledModel, ferx_core::Population) {
    let src = std::fs::read_to_string(Path::new("nonmem_anchor").join(model_file))
        .expect("anchor model file");
    let model = parse_full_model(&src).expect("anchor model parses").model;
    let pop =
        read_nonmem_csv(&Path::new("data").join(data_file), None, None).expect("anchor data loads");
    (model, pop)
}

fn theta(result: &FitResult, name: &str) -> f64 {
    let i = result
        .theta_names
        .iter()
        .position(|n| n == name)
        .unwrap_or_else(|| panic!("theta {name} missing from {:?}", result.theta_names));
    result.theta[i]
}

fn omega(result: &FitResult, eta: &str) -> f64 {
    let i = result
        .eta_names
        .iter()
        .position(|n| n == eta)
        .unwrap_or_else(|| panic!("eta {eta} missing from {:?}", result.eta_names));
    result.omega[(i, i)]
}

fn saem_opts() -> FitOptions {
    FitOptions {
        method: EstimationMethod::Saem,
        run_covariance_step: false,
        verbose: false,
        saem_seed: Some(619),
        ..FitOptions::default()
    }
}

fn assert_finite_close(what: &str, got: f64, want: f64, tol: f64) {
    assert!(got.is_finite(), "{what} is not finite: {got}");
    assert!(
        (got - want).abs() < tol,
        "{what}: ferx {got:.6} vs reference {want:.6} (|Δ| = {:.2e} ≥ {tol:.1e})",
        (got - want).abs()
    );
}

#[test]
fn covariate_mu_refs_are_detected_in_both_anchor_models() {
    let (add, _) = load("covmuref_additive_saem_fit.ferx", "covmuref_additive.csv");
    assert_eq!(add.covariate_mu_refs.len(), 1);
    let g = &add.covariate_mu_refs[0];
    assert_eq!(g.eta_name, "ETA_CL");
    assert_eq!(g.theta_names, vec!["TVCL", "TH_CRCL"]);
    assert_eq!(g.transform, MuTransform::Log);
    assert_eq!(g.covariate_names, vec!["CRCL"]);
    // The additive form never matched a single-anchor pattern: `mu_refs` is V only.
    assert!(!add.mu_refs.contains_key("ETA_CL"));
    assert!(add.mu_refs.contains_key("ETA_V"));

    let (pow, _) = load("covmuref_power_saem_fit.ferx", "covmuref_power.csv");
    assert_eq!(pow.covariate_mu_refs.len(), 1);
    assert_eq!(pow.covariate_mu_refs[0].theta_names, vec!["TVCL", "TH_WT"]);
    // The power form keeps its historical single anchor on TVCL as well.
    assert_eq!(
        pow.mu_refs.get("ETA_CL").map(|m| m.theta_name.as_str()),
        Some("TVCL")
    );
}

/// Tier-2: the group is a closed-form channel. `V` keeps its own single-anchor
/// pair, so the discriminating signal is the `not mu-referenced: CL` warning,
/// which the pre-#619 binary emitted on this exact model.
#[test]
fn saem_covariate_group_removes_the_not_mu_referenced_warning() {
    let (model, pop) = load("covmuref_additive_saem_fit.ferx", "covmuref_additive.csv");
    let opts = FitOptions {
        saem_n_exploration: 2,
        saem_n_convergence: 1,
        ..saem_opts()
    };
    let result = fit(&model, &pop, &model.default_params, &opts).expect("short SAEM must run");
    assert!(
        result.saem_mu_ref_m_step_evals_saved.unwrap_or(0) > 0,
        "closed-form channel must be active: {:?}",
        result.saem_mu_ref_m_step_evals_saved
    );
    assert!(
        !result
            .warnings
            .iter()
            .any(|w| w.contains("individual parameter(s) not mu-referenced")),
        "CL is mu-referenced through its covariate group, got {:?}",
        result.warnings
    );
    assert!(
        !result
            .warnings
            .iter()
            .any(|w| w.contains("covariate mu-reference on")),
        "no group may be declined on this model, got {:?}",
        result.warnings
    );
}

/// Tier-3: the additive renal gradient — NONMEM's *nonlinear* MU case — lands
/// where NONMEM SAEM lands. Bounds are measured: realised |Δ| on the anchor run
/// was 4e-3 (TVCL), 8e-6 (TH_CRCL), 3e-4 (TVV), 2e-4 (both ω²); each bound
/// below keeps ≥ 25× headroom on that and still excludes the pre-#619 value.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored covariate mu-referencing (#619): opt in with --features slow-tests"
)]
fn saem_recovers_the_additive_renal_gradient() {
    let (model, pop) = load("covmuref_additive_saem_fit.ferx", "covmuref_additive.csv");
    let result = fit(&model, &pop, &model.default_params, &saem_opts()).expect("SAEM must run");
    let th_crcl = theta(&result, "TH_CRCL");
    assert_finite_close("TH_CRCL", th_crcl, NM_ADD_TH_CRCL, 0.003);
    assert!(
        (th_crcl - BEFORE_TH_CRCL).abs() > 0.015,
        "TH_CRCL = {th_crcl:.5} must be clear of the pre-#619 stall at {BEFORE_TH_CRCL}"
    );
    assert_finite_close("TVCL", theta(&result, "TVCL"), NM_ADD_TVCL, 0.15);
    assert_finite_close("TVV", theta(&result, "TVV"), NM_ADD_TVV, 1.0);
    assert_finite_close(
        "omega^2(ETA_CL)",
        omega(&result, "ETA_CL"),
        NM_ADD_OMEGA_CL,
        0.01,
    );
    assert_finite_close(
        "omega^2(ETA_V)",
        omega(&result, "ETA_V"),
        NM_ADD_OMEGA_V,
        0.01,
    );
}

/// Tier-3: the allometric exponent — NONMEM's linear-in-theta MU, its
/// efficient case — so the group step must not lose ground there either.
/// Realised |Δ| on the anchor run: 7e-4 (TVCL), 8e-4 (TH_WT), 2e-2 (TVV),
/// 4e-4 / 1e-4 (ω²).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored covariate mu-referencing (#619): opt in with --features slow-tests"
)]
fn saem_recovers_the_allometric_exponent() {
    let (model, pop) = load("covmuref_power_saem_fit.ferx", "covmuref_power.csv");
    let result = fit(&model, &pop, &model.default_params, &saem_opts()).expect("SAEM must run");
    let th_wt = theta(&result, "TH_WT");
    assert_finite_close("TH_WT", th_wt, NM_POW_TH_WT, 0.05);
    assert!(
        (th_wt - BEFORE_TH_WT).abs() > 0.3,
        "TH_WT = {th_wt:.4} must be clear of the pre-#619 stall at {BEFORE_TH_WT}"
    );
    assert_finite_close("TVCL", theta(&result, "TVCL"), NM_POW_TVCL, 0.15);
    assert_finite_close("TVV", theta(&result, "TVV"), NM_POW_TVV, 1.0);
    assert_finite_close(
        "omega^2(ETA_CL)",
        omega(&result, "ETA_CL"),
        NM_POW_OMEGA_CL,
        0.01,
    );
    assert_finite_close(
        "omega^2(ETA_V)",
        omega(&result, "ETA_V"),
        NM_POW_OMEGA_V,
        0.01,
    );
}

/// Tier-3, #1415: the same allometric exponent with **no** mu-reference group —
/// the weight covariate goes through a conditionally assigned local, which the
/// #619 classifier rejects, so `TH_WT` carries no ETA and is moved only by the
/// eta-frozen numerical M-step. Same data, same MLE, same NONMEM SAEM anchor as
/// [`saem_recovers_the_allometric_exponent`].
///
/// Before #1415 this fixture reproduced the issue exactly: the M-step's NLopt
/// solve started from a quarter-of-the-bound-range first design and stopped on
/// a 1e-4 relative tolerance, so its "maximiser" was a few percent of a step,
/// and the 0.03 exploration cap on top of that left `TH_WT` at 0.332 from its
/// 0.3 start (release binary, seed 619) — the #619 stall, on the channel #619
/// did not touch. With the local first design and no exploration cap it lands
/// at 0.961 (realised |Δ| 0.040 against NONMEM's 0.921; the grouped model
/// realises 8e-4, this channel is noisier by construction). A converged solve
/// under the old 0.03 cap reached only 0.323, and caps of 0.1 / 0.3 reached
/// 0.340 / 0.403: the cap divides the number of EM steps the exploration phase
/// amounts to, and a covariate slope confounded with `ETA_CL` needs all of
/// them. Mutation check: restore either the default first design or the 0.03
/// cap and the `TH_WT` bound fails.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored numerical M-step (#1415): opt in with --features slow-tests"
)]
fn saem_recovers_the_allometric_exponent_on_the_numerical_mstep() {
    let (model, pop) = load("covmuref_power_numeric_saem_fit.ferx", "covmuref_power.csv");
    // The fixture must really be on the numerical channel, or it tests #619
    // again: no covariate group, and the advisory names TH_WT.
    assert!(
        model.covariate_mu_refs.is_empty(),
        "fixture must defeat covariate mu-ref detection: {:?}",
        model.covariate_mu_refs
    );
    let result = fit(&model, &pop, &model.default_params, &saem_opts()).expect("SAEM must run");
    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.contains("NO associated ETA") && w.contains("TH_WT")),
        "TH_WT must be reported as the no-ETA numerical-M-step theta: {:?}",
        result.warnings
    );
    let th_wt = theta(&result, "TH_WT");
    assert_finite_close("TH_WT", th_wt, NM_POW_TH_WT, 0.12);
    assert!(
        (th_wt - BEFORE_TH_WT).abs() > 0.3,
        "TH_WT = {th_wt:.4} must be clear of the pre-#1415 stall at {BEFORE_TH_WT}"
    );
    assert_finite_close("TVCL", theta(&result, "TVCL"), NM_POW_TVCL, 0.15);
    assert_finite_close("TVV", theta(&result, "TVV"), NM_POW_TVV, 1.0);
    assert_finite_close(
        "omega^2(ETA_CL)",
        omega(&result, "ETA_CL"),
        NM_POW_OMEGA_CL,
        0.02,
    );
    assert_finite_close(
        "omega^2(ETA_V)",
        omega(&result, "ETA_V"),
        NM_POW_OMEGA_V,
        0.01,
    );
}

/// Tier-3: IMP shares the group step (exact engine on the importance-weighted
/// posterior means). Same MLE as SAEM, so the same NONMEM finals are the
/// reference; IMP's own Monte-Carlo noise is why the bounds are wider (realised
/// |Δ| on the anchor run: 1e-4 for TH_CRCL, 3e-3 for TVCL).
///
/// This is also the regression pin for the φ-preserving re-centre of the IMP
/// proposal after a group step: without it the first step lands right here
/// (0.041) and the next five run the slope up to 0.10 on a stale proposal, after
/// which it collapses to the bound (measured 1e-5) — the `> 0.015` clearance
/// below is what fails in that case.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored covariate mu-referencing (#619): opt in with --features slow-tests"
)]
fn imp_recovers_the_additive_renal_gradient() {
    let (model, pop) = load("covmuref_additive_saem_fit.ferx", "covmuref_additive.csv");
    let opts = FitOptions {
        method: EstimationMethod::Imp,
        imp_iterations: 150,
        imp_samples: 500,
        imp_auto: false,
        imp_seed: Some(619),
        run_covariance_step: false,
        verbose: false,
        ..FitOptions::default()
    };
    let result = fit(&model, &pop, &model.default_params, &opts).expect("IMP must run");
    let th_crcl = theta(&result, "TH_CRCL");
    assert_finite_close("TH_CRCL", th_crcl, NM_ADD_TH_CRCL, 0.006);
    assert!(
        (th_crcl - BEFORE_TH_CRCL).abs() > 0.015,
        "TH_CRCL = {th_crcl:.5} must be clear of the pre-#619 stall at {BEFORE_TH_CRCL}"
    );
    assert_finite_close("TVCL", theta(&result, "TVCL"), NM_ADD_TVCL, 0.3);
    assert_finite_close("TVV", theta(&result, "TVV"), NM_ADD_TVV, 2.0);
}

/// Tier-3, #1458 / #1480. The same NONMEM-anchored no-ETA θ, estimated by the
/// alternatives to the single-draw maximiser. One test per arm, because after
/// #1480 the two arms are not in the same state: a loop panics on the first
/// failing arm, which is how `mstep_draws = 3` went unreported for three
/// nightly runs behind `score_sa` (#1475).
///
/// **Why this fixture is the right regression.** `TH_WT` here is the whole
/// point of #1458: it carries no ETA, defeats covariate mu-reference detection
/// (asserted below, as in the test above), and therefore moves *only* through
/// the numerical M-step. NONMEM 7.5.1 `METHOD=SAEM` on the same data and the
/// same start puts it at 0.921283, which is an external reference rather than
/// one of ferx's own readouts.
///
/// **Measured**, release-equivalent `ci-test`, seed 619, aarch64, `origin/main`
/// `57163fbf` + #1480 (`|Δ|` against the NONMEM anchor):
///
/// | arm | schedule | `TH_WT` | \|Δ\| |
/// |---|---|---|---|
/// | pre-#1415 stall | 150/250 | 0.3322 | 0.589 |
/// | `bobyqa` (the default) | 150/250 | 0.9997 | 0.078 |
/// | `mstep_draws = 3` | 150/250 | 0.8158 | 0.106 |
/// | `score_sa` | 150/250 | **0.7460** | **0.175** |
/// | `score_sa` | 300/700 | 0.9845 | 0.063 |
/// | `bobyqa` | 300/700 | **1.1485** | **0.227** |
///
/// The last two rows are why `score_sa`'s miss at the default schedule is
/// gated rather than widened away, and why the window is *not* a property of
/// the estimator alone: run long enough, `score_sa` walks onto the anchor and
/// `bobyqa` walks past it. See `score_sa_reaches_the_allometric_exponent_on_a_longer_schedule`.
///
/// The assertion is **not** "the new arm is closer" — one seed cannot carry
/// that claim, and the busulfan benchmark in the PR description is where the
/// six-seed comparison lives. What is pinned here is that the arm lands inside
/// the same anchored window the default arm has to satisfy, so a change that
/// breaks it (a sign slip in the accumulators, a stored draw that is never
/// re-centred) cannot land green.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored numerical M-step (#1458): opt in with --features slow-tests"
)]
fn the_k_draw_mstep_estimator_also_recovers_the_allometric_exponent() {
    assert_mstep_arm_recovers_th_wt(
        "mstep_draws=3",
        FitOptions {
            saem_mstep_draws: 3,
            ..saem_opts()
        },
    );
}

/// Tier-3, #1480. The `mstep_solver = score_sa` arm of the test above, at the
/// **default** 150/250 schedule, where it misses the anchored window by 0.055.
///
/// `#[ignore]`d rather than widened: 0.12 is the window the default solver has
/// to satisfy on this fixture and sharing it is the point (see the table on the
/// test above). #1480 fixed `score_sa`'s σ channel — the #1445 collapse it
/// re-opened — and that moved `TH_WT` only from 0.7247 to 0.7460, so the θ
/// channel's under-recovery at this schedule is a separate, open defect and is
/// tracked as the remaining half of #1480. `score_sa` stays opt-in until it
/// closes.
///
/// **Un-ignore this test, do not edit its bound**, when the θ channel is fixed;
/// `score_sa_reaches_the_allometric_exponent_on_a_longer_schedule` is the live
/// assertion in the meantime and would go red if the channel became *wrong*
/// rather than slow.
#[test]
#[ignore = "#1480: score_sa under-recovers TH_WT at the default 150/250 schedule \
            (0.7460 against NONMEM SAEM 0.9213, |Δ| 0.175, window 0.12). Its σ \
            channel is fixed; the θ channel's convergence rate is not. \
            score_sa_reaches_the_allometric_exponent_on_a_longer_schedule covers \
            the same θ at 300/700, where it lands at 0.9845."]
fn score_sa_also_recovers_the_allometric_exponent() {
    assert_mstep_arm_recovers_th_wt(
        "score_sa",
        FitOptions {
            saem_mstep_solver: SaemMstepSolver::ScoreSa,
            ..saem_opts()
        },
    );
}

/// Tier-3, #1480. `score_sa`'s θ channel is **slow on this fixture, not wrong**:
/// given 300/700 in place of the default 150/250 it lands on the NONMEM anchor,
/// inside the very window it misses at the default schedule.
///
/// This is the live half of the pair. `score_sa_also_recovers_the_allometric_exponent`
/// is `#[ignore]`d on the open defect, so without this test nothing in CI would
/// exercise `score_sa`'s no-ETA θ channel against an external reference at all,
/// and a change that made it *wrong* — a sign slip in the score accumulator, a
/// mis-indexed information row — would land green behind the ignore.
///
/// Realised when written (`ci-test`, aarch64, seed 619): `TH_WT` **0.9845**,
/// |Δ| 0.0632 against the anchor's 0.921283, so the 0.12 bound — the same one
/// the default arm is held to, deliberately not widened — carries 1.9× headroom.
/// The `bobyqa` arm at this same schedule realises 1.1485 (|Δ| 0.227) and would
/// *fail* it, which is why this test is about `score_sa` and is not a second
/// copy of the default-solver anchors above.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored numerical M-step (#1480): opt in with --features slow-tests"
)]
fn score_sa_reaches_the_allometric_exponent_on_a_longer_schedule() {
    assert_mstep_arm_recovers_th_wt(
        "score_sa @ 300/700",
        FitOptions {
            saem_mstep_solver: SaemMstepSolver::ScoreSa,
            saem_n_exploration: 300,
            saem_n_convergence: 700,
            outer_maxiter: 1000,
            ..saem_opts()
        },
    );
}

/// The body shared by the three tests above — one implementation of the anchor,
/// so an arm cannot drift onto a different set of assertions than its siblings.
///
/// `name` is only for the failure messages; every bound here is the one the
/// default-solver tests in this file are held to.
fn assert_mstep_arm_recovers_th_wt(name: &str, opts: FitOptions) {
    let (model, pop) = load("covmuref_power_numeric_saem_fit.ferx", "covmuref_power.csv");
    assert!(
        model.covariate_mu_refs.is_empty(),
        "fixture must defeat covariate mu-ref detection: {:?}",
        model.covariate_mu_refs
    );

    let result = fit(&model, &pop, &model.default_params, &opts)
        .unwrap_or_else(|e| panic!("{name}: SAEM must run: {e}"));
    // The arm must actually be in force: a model the scope gate refused
    // would fall back to the default solver and this test would then be a
    // second copy of the one above.
    assert!(
        !result
            .warnings
            .iter()
            .any(|w| w.contains("#1458") && w.contains("not available")),
        "{name}: the scope gate refused this fixture, so the arm never ran: {:?}",
        result.warnings
    );
    let th_wt = theta(&result, "TH_WT");
    assert!(th_wt.is_finite(), "{name}: TH_WT is not finite: {th_wt}");
    assert!(
        (th_wt - NM_POW_TH_WT).abs() < 0.12,
        "{name}: TH_WT {th_wt:.4} against NONMEM SAEM {NM_POW_TH_WT:.4} \
         (|Δ| {:.4}); the default `bobyqa` arm realises 0.9997 on this fixture",
        (th_wt - NM_POW_TH_WT).abs()
    );
    assert!(
        (th_wt - BEFORE_TH_WT).abs() > 0.3,
        "{name}: TH_WT = {th_wt:.4} must be clear of the pre-#1415 stall at \
         {BEFORE_TH_WT} — an arm that never moves the theta passes the anchor \
         bound above only because the start is not far from it"
    );
    assert_finite_close(
        &format!("{name}: TVCL"),
        theta(&result, "TVCL"),
        NM_POW_TVCL,
        0.25,
    );
    assert_finite_close(
        &format!("{name}: TVV"),
        theta(&result, "TVV"),
        NM_POW_TVV,
        1.5,
    );
    assert_finite_close(
        &format!("{name}: omega^2(ETA_CL)"),
        omega(&result, "ETA_CL"),
        NM_POW_OMEGA_CL,
        0.03,
    );
}

/// Tier-2, #1458. A model outside the plain-Gaussian scope must say so and keep
/// the historical solver, rather than stepping on an information matrix whose
/// closed form does not hold there. Fast: the fit is cut to a handful of
/// iterations, and what is asserted is the warning, not the estimate.
///
/// The out-of-scope shape used here is a **residual magnitude** (`weight = …`),
/// which is the case closest to in-scope — θ reaches the residual variance
/// through a second channel the expected-information closed form does not carry
/// — and which runs on the same data as the in-scope control.
///
/// It takes **three** edits, not one: the parser refuses `weight =` on a purely
/// proportional error ("a common scale factor cancels out of a proportional
/// (constant-CV) error"), so the variant also needs an additive component for
/// the weight to act on, and a magnitude expression must declare its covariate
/// ("an undeclared name silently evaluates to 0 and would make the magnitude a
/// constant"). `has_custom_ruv_magnitude` is asserted below because
/// that is the property the scope gate actually reads — without it a fixture
/// that parsed but carried no magnitude would exercise the in-scope path and
/// the test would be asserting nothing.
#[test]
fn an_out_of_scope_model_falls_back_from_the_new_mstep_estimators() {
    let (model, pop) = load("covmuref_power_numeric_saem_fit.ferx", "covmuref_power.csv");
    let short = |mut o: FitOptions| {
        o.saem_n_exploration = 2;
        o.saem_n_convergence = 2;
        o.saem_n_mh_steps = 2;
        o
    };
    let score_sa = || FitOptions {
        saem_mstep_solver: SaemMstepSolver::ScoreSa,
        ..saem_opts()
    };
    let k_draws = || FitOptions {
        saem_mstep_draws: 3,
        ..saem_opts()
    };

    // In scope: no fallback warning, so the assertions below are about the gate
    // and not about every fit emitting the message.
    let ok = fit(&model, &pop, &model.default_params, &short(score_sa()))
        .expect("in-scope SAEM must run");
    assert!(
        !ok.warnings.iter().any(|w| w.contains("not available")),
        "an in-scope model must not warn: {:?}",
        ok.warnings
    );

    // Out of scope by one line: the same model with a residual magnitude.
    let src = std::fs::read_to_string(
        Path::new("nonmem_anchor").join("covmuref_power_numeric_saem_fit.ferx"),
    )
    .expect("anchor model file");
    let magnitude_src = src
        .replace(
            "  sigma PROP_ERR ~ 0.02",
            "  sigma PROP_ERR ~ 0.02\n  sigma ADD_ERR  ~ 0.50 (sd)",
        )
        .replace(
            "DV ~ proportional(PROP_ERR)",
            "DV ~ combined(PROP_ERR, ADD_ERR) weight = WT

[covariates]
  WT continuous",
        );
    assert_ne!(magnitude_src, src, "the magnitude edit did not apply");
    assert!(
        magnitude_src.contains("ADD_ERR") && magnitude_src.contains("weight = WT"),
        "both halves of the magnitude edit must land: {magnitude_src}"
    );
    let mag = parse_full_model(&magnitude_src)
        .expect("magnitude variant parses")
        .model;
    assert!(
        mag.has_custom_ruv_magnitude(),
        "the variant must really carry a magnitude, or this tests nothing"
    );

    let refused = fit(&mag, &pop, &mag.default_params, &short(score_sa()))
        .expect("out-of-scope SAEM must still run");
    assert!(
        refused
            .warnings
            .iter()
            .any(|w| w.contains("score_sa") && w.contains("magnitude")),
        "a residual magnitude must be refused by name: {:?}",
        refused.warnings
    );

    let refused_k = fit(&mag, &pop, &mag.default_params, &short(k_draws()))
        .expect("out-of-scope SAEM must still run");
    assert!(
        refused_k
            .warnings
            .iter()
            .any(|w| w.contains("mstep_draws") && w.contains("magnitude")),
        "`mstep_draws` must be refused by name too: {:?}",
        refused_k.warnings
    );
}

/// Tier-2, #1458. **Each option must be distinguishable from the default
/// through `fit()`.**
///
/// This is the gate the rest of the #1458 tests were missing. The Tier-1 tests
/// drive `MstepScoreSa::step` and `theta_sigma_mstep_light` directly, and the
/// Tier-3 anchor below asserts each arm lands in a window the **default** arm
/// already satisfies — so deleting the wiring in `run_saem` (returning
/// `score_sa = None` after the option is read, or forcing `mstep_draws = 1`
/// before the history push) left every one of them green. Codex review of
/// PR #1462 found that; this test is the answer.
///
/// It runs the production path three times on one model and seed, changing
/// nothing but the option, and asserts the estimates **differ**. A fit is a
/// deterministic function of (model, data, options, seed), so two arms of the
/// same estimator would be bit-identical: any non-zero difference is the option
/// taking effect, and a mutation that stops it taking effect makes the
/// difference exactly zero.
///
/// Realised worst |Δ log θ| against the `bobyqa` arm on this fixture at
/// 8/8/3 (printed by the assertion when it fires): `score_sa` **1.03e-1**,
/// `mstep_draws = 3` **1.43e-2**. The `1e-4` gate is two orders below the
/// smaller of those — loose enough that the schedule can be retuned without
/// re-measuring, tight enough that only an exactly-zero difference (the
/// mutation) fails it.
///
/// Deliberately short (8 exploration + 8 convergence, 3 MH steps): what is
/// asserted is that the option reaches the estimator, not where it converges.
#[test]
fn each_mstep_option_changes_the_fit_through_the_public_api() {
    let (model, pop) = load("covmuref_power_numeric_saem_fit.ferx", "covmuref_power.csv");

    let short = |mut o: FitOptions| {
        o.saem_n_exploration = 8;
        o.saem_n_convergence = 8;
        o.saem_n_mh_steps = 3;
        o
    };
    let run = |o: FitOptions| -> Vec<f64> {
        let r = fit(&model, &pop, &model.default_params, &short(o)).expect("SAEM must run");
        assert!(
            !r.warnings
                .iter()
                .any(|w| w.contains("#1458") && w.contains("not available")),
            "the scope gate refused this fixture, so no arm ran: {:?}",
            r.warnings
        );
        r.theta.clone()
    };

    let base = run(saem_opts());

    // The control: the default arm run twice is bit-identical, so any
    // difference below is the option and not fit-to-fit noise.
    assert_eq!(
        base,
        run(saem_opts()),
        "two default fits at the same seed are not bit-identical — the \
         differences below cannot be attributed to the option"
    );

    let worst = |a: &[f64], b: &[f64]| -> f64 {
        a.iter().zip(b.iter()).fold(0.0f64, |m, (x, y)| {
            assert!(
                x.is_finite() && y.is_finite(),
                "non-finite theta: {x} vs {y}"
            );
            m.max((x.ln() - y.ln()).abs())
        })
    };

    for (name, opts) in [
        (
            "mstep_solver = score_sa",
            FitOptions {
                saem_mstep_solver: SaemMstepSolver::ScoreSa,
                ..saem_opts()
            },
        ),
        (
            "mstep_draws = 3",
            FitOptions {
                saem_mstep_draws: 3,
                ..saem_opts()
            },
        ),
    ] {
        let arm = run(opts);
        let d = worst(&arm, &base);
        assert!(
            d > 1e-4,
            "{name} did not change the fit (worst |Δ log θ| = {d:.4e}) — the option is not \
             reaching the estimator. Realised when written: score_sa 1.03e-1, \
             mstep_draws = 3 1.43e-2. base = {base:?}, arm = {arm:?}"
        );
    }
}

/// Tier-2, #1480. `mstep_damping` must reach σ under `mstep_solver = score_sa`.
///
/// **The regression this exists to catch** is the one the Tier-1 tests in
/// `src/estimation/saem.rs` structurally cannot: `run_saem` handing
/// `MstepScoreSa::step` the shared `gamma` where it should hand `gamma_mstep`.
/// Under the shipped default (`mstep_damping` off, i.e. `gamma_mstep = 1`) that
/// substitution is a **no-op** — `min(γ, 0.2, γ)` and `min(γ, 0.2, 1)` are the
/// same number — so every unit test of `step` and every fit in this repo stays
/// green under it. It only bites when the option is set, which is the one case
/// nothing else covers, and what it breaks is #1445's one-sided guarantee that
/// σ never steps faster than θ.
///
/// σ is the *only* channel `mstep_damping` has into a `score_sa` fit: θ is
/// deliberately left on the shared γ there (see `MstepScoreSa`'s docs), so a
/// changed σ is a sufficient as well as a necessary signal. θ is not asserted
/// either way — it is not held fixed, because a different σ changes the next
/// iteration's score and information.
///
/// **Both call sites, asserted separately.** `run_saem` calls
/// `MstepScoreSa::step` twice — once from the mu-referenced branch and once from
/// the `mu_referencing = false` one — and a single fixture only reaches the
/// first. The two are run here as two arms with the failure message naming
/// which, so deleting either argument reddens the arm that owns it rather than
/// being covered for by its twin.
///
/// Realised when written (25 + 10 iterations): worst |Δ log σ| = **1.229e-1** on
/// the mu-referenced branch and **1.212e-1** on the other, between
/// `mstep_damping` off and 0.005, against a 1e-3 bound — 120× headroom on both.
/// The schedule is 25 exploration iterations rather than the 8 the test above
/// uses because γ_σ and γ_mstep only differ *during* exploration, and at 8 + 8
/// with `mstep_damping = 0.02` the `mu_referencing = false` branch realised
/// 1.617e-3, only 1.6× the bound — measured, not assumed, and too thin to ship.
/// Mutation-checked: handing `step` the shared `gamma` makes both arms
/// bit-identical (0.000e0), and no other test in the repo moves.
#[test]
fn mstep_damping_reaches_sigma_under_score_sa() {
    let (model, pop) = load("covmuref_power_numeric_saem_fit.ferx", "covmuref_power.csv");

    let run = |mu_referencing: bool, damping: Option<f64>| -> Vec<f64> {
        let opts = FitOptions {
            saem_mstep_solver: SaemMstepSolver::ScoreSa,
            saem_mstep_damping: damping,
            mu_referencing,
            saem_n_exploration: 25,
            saem_n_convergence: 10,
            saem_n_mh_steps: 3,
            ..saem_opts()
        };
        let r = fit(&model, &pop, &model.default_params, &opts).expect("SAEM must run");
        assert!(
            !r.warnings
                .iter()
                .any(|w| w.contains("#1458") && w.contains("not available")),
            "the scope gate refused this fixture, so score_sa never ran: {:?}",
            r.warnings
        );
        r.sigma.clone()
    };

    for (branch, mu_referencing) in [
        ("mu-referenced branch", true),
        ("mu_referencing = false branch", false),
    ] {
        let off = run(mu_referencing, None);
        // The control: the same arm twice is bit-identical, so the difference
        // below is the option and not fit-to-fit noise.
        assert_eq!(
            off,
            run(mu_referencing, None),
            "{branch}: two score_sa fits at the same seed are not bit-identical — the \
             difference below cannot be attributed to `mstep_damping`"
        );

        let damped = run(mu_referencing, Some(0.005));
        assert_eq!(
            off.len(),
            damped.len(),
            "{branch}: the two arms returned different σ shapes"
        );
        let worst = off.iter().zip(damped.iter()).fold(0.0f64, |m, (a, b)| {
            assert!(
                a.is_finite() && b.is_finite() && *a > 0.0 && *b > 0.0,
                "{branch}: σ must be finite and positive: {a} vs {b}"
            );
            m.max((a.ln() - b.ln()).abs())
        });
        assert!(
            worst > 1e-3,
            "{branch}: `mstep_damping` did not reach σ under score_sa (worst \
             |Δ log σ| = {worst:.3e}) — γ_σ is being built from the shared γ instead of \
             γ_mstep, so a damped fit steps σ at the undamped rate. off = {off:?}, \
             damped = {damped:?}"
        );
    }
}
