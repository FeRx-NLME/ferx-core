//! Time-varying clearance combined with inter-occasion variability, fitted by VI.
//!
//! Gate the slow ones so they are skipped in the default PR job:
//!
//!   cargo test --features slow-tests --test vi_time_varying_iov
//!
//! ## Why this pairing gets its own file
//!
//! A declining clearance and a per-occasion random effect are *confusable by
//! construction*. Both make a subject's clearance differ between early and late records.
//! Given only the data, the split between "CL is falling over the course" and "CL is
//! redrawn each occasion" is identified solely by the shape of the difference —
//! systematic and monotone versus exchangeable across occasions. A fit that gets the
//! split wrong still converges, still reports plausible parameters, and is wrong in the
//! way that matters clinically: extrapolating a decline that isn't there, or attributing
//! a real decline to noise.
//!
//! This is the busulfan shape — a multi-day course whose clearance falls day over day
//! while each day carries its own variability — and it is the case where "IOV works" and
//! "time-varying covariates work" being separately true does not imply the pair works.
//!
//! ## Validation strategy
//!
//! A round-trip against ferx's own `simulate()`, not against NONMEM. That is deliberate
//! and it is the *stronger* anchor for this particular claim: the question is whether the
//! estimator recovers a known decomposition into `KDEC` and `Ω_iov`, and only a
//! simulation knows the true split. A NONMEM comparison would tell us the two tools agree
//! on an estimate, not that the estimate is right. (The NONMEM anchor for VI's parameters
//! generally lives in `docs/estimation/vi.qmd`; the IOV objective itself is anchored in
//! `tests/warfarin_iov_nonmem.rs`.)
//!
//! Three claims, in increasing order of what they'd catch:
//!
//! 1. VI recovers `KDEC` and `Ω_iov` simultaneously from data containing both.
//! 2. Dropping IOV from the model biases the recovered decline — the confusion is real,
//!    so a test that only ever fits the correct model would not notice the estimator
//!    leaning on the wrong term.
//! 3. Dropping the decline costs objective function on 1 df — the decline is genuinely
//!    identified, not absorbed by the per-occasion effect.

use ferx_core::{
    fit, simulate_with_seed, CompiledModel, DoseEvent, EstimationMethod, FitOptions, FitResult,
    ModelParameters, Population, SimOutcome, Subject, ViFinalOfv,
};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Truth
// ---------------------------------------------------------------------------

const TRUE_TVCL: f64 = 1.0;
const TRUE_TVV: f64 = 10.0;
/// Per-hour decline. Over a 96 h course clearance falls to `exp(-0.96) ≈ 0.38` of its
/// day-0 value — large enough to be identified against `Ω_iov` at this design, which is
/// the point of the test rather than a claim about real busulfan.
const TRUE_KDEC: f64 = 0.01;
const TRUE_OMEGA_CL: f64 = 0.09;
const TRUE_OMEGA_V: f64 = 0.04;
/// Between-occasion variance, SD ≈ 0.30 on log-CL.
///
/// Sized deliberately rather than picked: at `Ω_iov = 0.04` (SD 0.20) the per-occasion
/// effect is small enough that omitting it from the model barely moves the decline
/// estimate (measured: `KDEC` shifts 4.3%), and the misspecification test below reduces to
/// a coin flip against Monte-Carlo noise. At SD 0.30 the two effects genuinely compete and
/// the consequences of confusing them are unambiguous.
const TRUE_OMEGA_IOV: f64 = 0.09;
const TRUE_SIGMA: f64 = 0.1;

const N_SUBJECTS: usize = 60;
const N_DAYS: usize = 4;
const SEED: u64 = 20_260_812;

/// `with_decline` toggles the `KDEC` term; `with_iov` toggles the per-occasion kappa.
/// Everything else is held identical so a difference between fits is attributable to the
/// toggled term alone.
fn model_src(with_decline: bool, with_iov: bool) -> String {
    let kdec_par = if with_decline {
        "  theta KDEC(0.005, 0.00001, 1.0)\n"
    } else {
        ""
    };
    let kappa_par = if with_iov {
        "  kappa KAPPA_CL ~ 0.02\n"
    } else {
        ""
    };
    let decline = if with_decline {
        " * exp(-KDEC * TIME)"
    } else {
        ""
    };
    let kappa = if with_iov { " + KAPPA_CL" } else { "" };
    format!(
        r"
[parameters]
  theta TVCL(0.6, 0.01, 20.0)
  theta TVV(8.0, 0.5, 200.0)
{kdec_par}
  omega ETA_CL ~ 0.05
  omega ETA_V  ~ 0.05
{kappa_par}
  sigma PROP_ERR ~ 0.09 (sd)

[individual_parameters]
  CL = TVCL{decline} * exp(ETA_CL{kappa})
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  iov_column = OCC
"
    )
}

fn parse(with_decline: bool, with_iov: bool) -> CompiledModel {
    ferx_core::parser::model_parser::parse_model_string(&model_src(with_decline, with_iov))
        .expect("fixture parses")
}

/// The truth, expressed in the full (decline + IOV) model's parameter layout.
fn truth(model: &CompiledModel) -> ModelParameters {
    let mut p = model.default_params.clone();
    let set = |p: &mut ModelParameters, name: &str, v: f64| {
        let i = p
            .theta_names
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("theta {name} not found"));
        p.theta[i] = v;
    };
    set(&mut p, "TVCL", TRUE_TVCL);
    set(&mut p, "TVV", TRUE_TVV);
    set(&mut p, "KDEC", TRUE_KDEC);
    p.omega = ferx_core::OmegaMatrix::from_diagonal(
        &[TRUE_OMEGA_CL, TRUE_OMEGA_V],
        vec!["ETA_CL".into(), "ETA_V".into()],
    );
    p.omega_iov = Some(ferx_core::OmegaMatrix::from_diagonal(
        &[TRUE_OMEGA_IOV],
        vec!["KAPPA_CL".into()],
    ));
    p.sigma.values = vec![TRUE_SIGMA];
    p
}

// ---------------------------------------------------------------------------
// Design + simulation
// ---------------------------------------------------------------------------

/// One 100 mg IV bolus per day for `N_DAYS`, sampled at 1 h and 6 h post-dose.
///
/// Two samples per occasion is the minimum that separates the two effects: with a single
/// trough per day, a systematic decline and an exchangeable per-day offset produce the
/// same data and no estimator could tell them apart.
fn design() -> Population {
    let subjects = (0..N_SUBJECTS)
        .map(|s| {
            let mut doses = Vec::new();
            let mut dose_occasions = Vec::new();
            let mut obs_times = Vec::new();
            let mut occasions = Vec::new();
            for d in 0..N_DAYS {
                let t0 = 24.0 * d as f64;
                doses.push(DoseEvent::new(t0, 100.0, 1, 0.0, false, 0.0));
                dose_occasions.push(d as u32 + 1);
                for dt in [1.0f64, 6.0] {
                    obs_times.push(t0 + dt);
                    occasions.push(d as u32 + 1);
                }
            }
            let n_obs = obs_times.len();
            Subject {
                id: format!("{}", s + 1),
                doses,
                obs_times,
                obs_raw_times: Vec::new(),
                observations: vec![0.0; n_obs],
                obs_cmts: vec![1; n_obs],
                covariates: HashMap::new(),
                dose_covariates: Vec::new(),
                obs_covariates: Vec::new(),
                pk_only_times: Vec::new(),
                pk_only_covariates: Vec::new(),
                reset_times: Vec::new(),
                reset_covariates: Vec::new(),
                cens: vec![0; n_obs],
                occasions,
                obs_l2: Vec::new(),
                dose_occasions,
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

/// Simulate under `truth` and write the drawn DVs back onto the design, giving a dataset
/// whose true decomposition into `KDEC` and `Ω_iov` is known by construction.
fn simulated_population(model: &CompiledModel, params: &ModelParameters) -> Population {
    let mut pop = design();
    let rows = simulate_with_seed(model, &pop, params, 1, SEED);

    let mut per_subject: HashMap<String, Vec<f64>> = HashMap::new();
    for r in &rows {
        if let SimOutcome::Continuous { value } = r.outcome {
            per_subject.entry(r.id.clone()).or_default().push(value);
        }
    }
    for subj in &mut pop.subjects {
        let dv = per_subject
            .remove(&subj.id)
            .unwrap_or_else(|| panic!("no simulated rows for subject {}", subj.id));
        assert_eq!(
            dv.len(),
            subj.observations.len(),
            "simulated row count must match the design for subject {}",
            subj.id
        );
        assert!(
            dv.iter().all(|v| v.is_finite() && *v > 0.0),
            "simulation produced a non-positive DV for subject {}",
            subj.id
        );
        subj.observations = dv;
    }
    pop
}

fn vi_opts() -> FitOptions {
    FitOptions {
        method: EstimationMethod::Vi,
        vi_seed: Some(SEED),
        vi_final_ofv: ViFinalOfv::Laplace,
        run_covariance_step: false,
        ..Default::default()
    }
}

fn fit_with(model: &CompiledModel, pop: &Population) -> FitResult {
    fit(model, pop, &model.default_params, &vi_opts()).expect("VI fit returns")
}

fn theta_of(fit: &FitResult, model: &CompiledModel, name: &str) -> f64 {
    let i = model
        .theta_names
        .iter()
        .position(|n| n == name)
        .unwrap_or_else(|| panic!("theta {name} not found"));
    fit.theta[i]
}

fn rel_err(got: f64, want: f64) -> f64 {
    (got - want).abs() / want.abs()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Claim 1: with both effects present in the data and in the model, VI recovers both.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn vi_recovers_time_varying_clearance_and_iov() {
    let full = parse(true, true);
    let pop = simulated_population(&full, &truth(&full));
    let f = fit_with(&full, &pop);

    let tvcl = theta_of(&f, &full, "TVCL");
    let tvv = theta_of(&f, &full, "TVV");
    let kdec = theta_of(&f, &full, "KDEC");
    let omega_cl = f.omega[(0, 0)];
    let omega_v = f.omega[(1, 1)];
    let omega_iov = f
        .omega_iov
        .as_ref()
        .expect("an IOV fit must report Omega_iov")[(0, 0)];

    // Measured at this design and seed:
    //
    //   TVCL     0.980    (true 1.0,    −2.0%)
    //   TVV      9.871    (true 10.0,   −1.3%)
    //   KDEC     0.008715 (true 0.01,  −12.9%)
    //   Ω_iov    0.0680   (true 0.09,  −24.4%)
    //   σ        0.1058   (true 0.1,    +5.8%)
    //
    // Bounds are set around those with headroom, not tightened onto them — this is a
    // stochastic estimator on 60 simulated subjects and pinning it to 3 s.f. would make
    // the test a tripwire for irrelevant changes.
    assert!(
        rel_err(tvcl, TRUE_TVCL) < 0.15,
        "TVCL: got {tvcl:.4}, true {TRUE_TVCL}"
    );
    assert!(
        rel_err(tvv, TRUE_TVV) < 0.15,
        "TVV: got {tvv:.4}, true {TRUE_TVV}"
    );
    // KDEC comes back ~13% low. Recorded rather than tightened away: the decline is the
    // parameter this whole file is about, and a bound that hid a real downward bias would
    // defeat the purpose.
    assert!(
        rel_err(kdec, TRUE_KDEC) < 0.30,
        "KDEC: got {kdec:.5}, true {TRUE_KDEC}"
    );
    // Ω_iov comes back ~24% low, and the sign is the one VI_PLAN §10.6 predicted:
    // variational posteriors understate posterior variance, Ω_iov* is a mean of
    // `S + μμᵀ`, and unlike Ω_bsv it has only `Σᵢ Kᵢ` occasions rather than N subjects to
    // average the bias away. An earlier scratch run reported that this "did not
    // reproduce"; at this design it does. It is a known characteristic of the estimator,
    // not a regression, so the band accommodates it — but the band is deliberately not
    // widened past what would still catch the real failure mode, Ω_iov collapsing toward
    // zero while the decline swallows the per-occasion variation.
    assert!(
        omega_iov > 0.4 * TRUE_OMEGA_IOV && omega_iov < 2.5 * TRUE_OMEGA_IOV,
        "Omega_iov: got {omega_iov:.5}, true {TRUE_OMEGA_IOV}"
    );

    // Ω_bsv, which this file simulated from `TRUE_OMEGA_CL` / `TRUE_OMEGA_V` and then did
    // not look at. Every other `Ω` assertion VI carries is structurally blind to a
    // downward bias in it — the exact-case oracle fits a linear-Gaussian model where
    // full-rank VI is exact, and the `closed_form_omega` tests check that `Ω*` is the right
    // maximizer *given* the `φᵢ`, which a biased `φ` satisfies. That left the estimator's
    // headline output unguarded here, on the one test in position to check it.
    //
    // Wider on `ETA_V` than on `ETA_CL` for a design reason, not a tolerance-fudging one:
    // with `KAPPA_CL` and the `KDEC` decline both loading on clearance, `ETA_V` is the eta
    // this design informs least, and at `Ω_V = 0.04` (SD 0.20) it has the least room above
    // the noise floor.
    //
    // The `Ω_iov` band above is the tighter statement about the per-occasion term; this is
    // the statement that the between-subject term did not pay for it. Those two failing
    // together means VI understated variance overall; `Ω_iov` low while `Ω_bsv` holds is
    // the documented §10.6 characteristic; `Ω_bsv` low while `Ω_iov` holds would be new and
    // is exactly what no test could previously see.
    assert!(
        omega_cl > 0.5 * TRUE_OMEGA_CL && omega_cl < 2.0 * TRUE_OMEGA_CL,
        "Omega_cl: got {omega_cl:.5}, true {TRUE_OMEGA_CL} — between-subject IIV on CL \
         not recovered"
    );
    assert!(
        omega_v > 0.4 * TRUE_OMEGA_V && omega_v < 2.5 * TRUE_OMEGA_V,
        "Omega_v: got {omega_v:.5}, true {TRUE_OMEGA_V} — between-subject IIV on V \
         not recovered"
    );

    eprintln!(
        "\nVI recovery: TVCL {tvcl:.4} TVV {tvv:.4} KDEC {kdec:.5} | \
         w2(CL) {omega_cl:.5} w2(V) {omega_v:.5} w2_iov {omega_iov:.5} | sigma {:.4}",
        f.sigma[0]
    );
}

/// Claim 2: omitting IOV biases the recovered decline and inflates residual error.
///
/// This is the test that would catch an estimator quietly leaning on whichever term is
/// available. The direction is asserted, not just the magnitude: with no per-occasion
/// term to absorb it, between-occasion variation has nowhere to go but the residual.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn omitting_iov_biases_the_decline_and_inflates_residual_error() {
    let full = parse(true, true);
    let pop = simulated_population(&full, &truth(&full));

    let f_full = fit_with(&full, &pop);
    let no_iov = parse(true, false);
    let f_no_iov = fit_with(&no_iov, &pop);

    // Measured: σ 0.1058 → 0.1529 (×1.45) and KDEC 0.008715 → 0.007727 (−11.3%).
    // Both bounds sit well inside those margins.
    let sigma_full = f_full.sigma[0];
    let sigma_no_iov = f_no_iov.sigma[0];
    assert!(
        sigma_no_iov > 1.15 * sigma_full,
        "dropping IOV must push between-occasion variation into the residual: \
         sigma {sigma_no_iov:.4} without IOV vs {sigma_full:.4} with it"
    );

    // The decline is *understated* when the per-occasion term is missing, not merely
    // moved. Asserting the direction is what makes this a statement about the confusion
    // between the two effects: the misspecified model has already spent some of the
    // between-occasion variation on the residual, leaving less systematic signal for the
    // trend to explain.
    let kdec_full = theta_of(&f_full, &full, "KDEC");
    let kdec_no_iov = theta_of(&f_no_iov, &no_iov, "KDEC");
    assert!(
        kdec_no_iov < kdec_full,
        "dropping IOV should understate the decline, not overstate it: \
         KDEC {kdec_no_iov:.5} without IOV vs {kdec_full:.5} with it"
    );
    assert!(
        rel_err(kdec_no_iov, kdec_full) > 0.05,
        "dropping IOV must visibly move the decline estimate: \
         KDEC {kdec_no_iov:.5} without IOV vs {kdec_full:.5} with it"
    );
    // And the misspecified fit must be the worse one, on a genuine marginal likelihood
    // rather than on the ELBO (`vi_final_ofv = laplace`).
    assert!(
        f_no_iov.ofv > f_full.ofv,
        "the model missing a real effect must fit worse: \
         OFV {:.2} without IOV vs {:.2} with it",
        f_no_iov.ofv,
        f_full.ofv
    );
}

/// Claim 3: the decline is genuinely identified on top of IOV — dropping it costs
/// objective function on 1 df, rather than being absorbed by the per-occasion effect.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn the_decline_is_worth_its_degree_of_freedom_on_top_of_iov() {
    let full = parse(true, true);
    let pop = simulated_population(&full, &truth(&full));

    let f_full = fit_with(&full, &pop);
    let no_decline = parse(false, true);
    let f_no_decline = fit_with(&no_decline, &pop);

    let d_ofv = f_no_decline.ofv - f_full.ofv;
    assert!(
        f_full.ofv.is_finite() && f_no_decline.ofv.is_finite(),
        "both fits need a real -2 log L (vi_final_ofv = laplace): {:.3} / {:.3}",
        f_full.ofv,
        f_no_decline.ofv
    );
    // Measured: 920.0 with the decline, 1015.1 without — ΔOFV 95.1 on 1 df. That number
    // is a property of *this* design (the simulated decline is strong: CL falls to ~38%
    // of its day-0 value over the course), not a portable effect size, so the bound below
    // is the significance threshold rather than a pin near 95.
    //
    // 1 df at α = 0.001 is 10.83. Requiring well past that keeps the assertion about the
    // decline being identified rather than about a marginal p-value.
    assert!(
        d_ofv > 10.83,
        "dropping the clearance decline should cost real objective on 1 df, got \
         dOFV {d_ofv:.2} (full {:.2}, no-decline {:.2})",
        f_full.ofv,
        f_no_decline.ofv
    );
}
