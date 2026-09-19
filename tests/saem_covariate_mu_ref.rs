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

/// Tier-3, #1458. The same NONMEM-anchored no-ETA θ, estimated by the two
/// alternatives to the single-draw maximiser: `mstep_solver = score_sa`
/// (stochastic approximation on the score and expected information) and
/// `mstep_draws = 3` (the M-step objective averaged over three E-step draws).
///
/// **Why this fixture is the right regression.** `TH_WT` here is the whole
/// point of #1458: it carries no ETA, defeats covariate mu-reference detection
/// (asserted below, as in the test above), and therefore moves *only* through
/// the numerical M-step. NONMEM 7.5.1 `METHOD=SAEM` on the same data and the
/// same start puts it at 0.921283, which is an external reference rather than
/// one of ferx's own readouts.
///
/// **Measured**, release-equivalent `ci-test`, seed 619, the default 150/250
/// schedule (`|Δ|` against the NONMEM anchor):
///
/// | arm | `TH_WT` | \|Δ\| |
/// |---|---|---|
/// | pre-#1415 stall | 0.3322 | 0.589 |
/// | `bobyqa` (the default, post-#1445) | 0.9026 | 0.019 |
/// | `score_sa` | see the assertion message | |
/// | `mstep_draws = 3` | see the assertion message | |
///
/// The assertion is **not** "the new arm is closer" — one seed cannot carry
/// that claim, and the busulfan benchmark in the PR description is where the
/// six-seed comparison lives. What is pinned here is that each arm lands inside
/// the same anchored window the default arm has to satisfy, so a change that
/// breaks one of them (a sign slip in the accumulators, a stored draw that is
/// never re-centred) cannot land green.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored numerical M-step (#1458): opt in with --features slow-tests"
)]
fn the_alternative_mstep_estimators_also_recover_the_allometric_exponent() {
    let (model, pop) = load("covmuref_power_numeric_saem_fit.ferx", "covmuref_power.csv");
    assert!(
        model.covariate_mu_refs.is_empty(),
        "fixture must defeat covariate mu-ref detection: {:?}",
        model.covariate_mu_refs
    );

    let arms: [(&str, FitOptions); 2] = [
        (
            "score_sa",
            FitOptions {
                saem_mstep_solver: SaemMstepSolver::ScoreSa,
                ..saem_opts()
            },
        ),
        (
            "mstep_draws=3",
            FitOptions {
                saem_mstep_draws: 3,
                ..saem_opts()
            },
        ),
    ];

    for (name, opts) in arms {
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
             (|Δ| {:.4}); the default `bobyqa` arm realises 0.9026 on this fixture",
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
}

/// Tier-2, #1458. A model outside the plain-Gaussian scope must say so and keep
/// the historical solver, rather than stepping on an information matrix whose
/// closed form does not hold there. Fast: the fit is cut to a handful of
/// iterations, and what is asserted is the warning, not the estimate.
///
/// The out-of-scope shape used here is a **residual magnitude** (`weight = …`),
/// which is the case closest to in-scope — θ reaches the residual variance
/// through a second channel the expected-information closed form does not carry
/// — and which runs on the same data as the in-scope control, so the two arms
/// differ by one line of model text and nothing else.
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
    let magnitude_src = src.replace(
        "DV ~ proportional(PROP_ERR)",
        "DV ~ proportional(PROP_ERR) weight = WT",
    );
    assert_ne!(magnitude_src, src, "the magnitude edit did not apply");
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
