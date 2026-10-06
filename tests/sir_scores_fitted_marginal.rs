//! Post-fit computations score the marginal the fit was estimated under (#1710).
//!
//! A `method = foce` fit used to leave `FitOptions::interaction` at its `true`
//! default. The estimation stage set its own flag (FOCE off), but everything after the
//! stage loop read the top-level one: SIR weighted its draws with the **FOCEI**
//! objective while centred on the FOCE optimum, the M3 warning's gate never opened, and
//! `FitResult::interaction` said `true`. The standalone `run_sir` / `run_covariance`
//! read the caller's flag the same way. On warfarin_iov the two objectives are 151 units
//! apart at the FOCE estimates (SIR's recompute 354.985 vs `ofv` 203.557), so the SIR
//! effective sample size collapsed from 396 to 4.9 per 1000 and ω²(KA) = 0.919 fell
//! outside its own SIR interval.
//!
//! Every fixture here builds `FitOptions { method: Foce, ..default() }` through the API,
//! so `interaction` arrives `true`. The model-file parser now clears it for `foce`; an
//! API caller (and ferx-r, when R gives `method` but the flag rides along) does not go
//! through the parser, so these fixtures reach the post-fit resolution in `fit_inner`
//! directly and the parser fix cannot mask a regression there.
//!
//! Engine: warfarin_iov (one-compartment oral, IOV on CL) and warfarin_bloq (M3) are both
//! inside the analytic `Dual2` scope; each test asserts the fit reports no
//! finite-difference fallback, so the fixtures run on the analytic path.

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::{
    fit, prepare_run, read_nonmem_csv, run_covariance, run_sir, EstimationMethod, FitOptions,
    FitResult, OmegaMatrix, PreparedRun,
};
use std::path::Path;

/// ferx's FOCE optimum on `examples/warfarin_iov.ferx` (OFV 203.557473), measured on
/// macOS arm64 at this PR's tree by a full fit from the file's initial values. The
/// per-PR test starts here with `outer_maxiter = 0`, so it scores the optimum without a
/// convergence loop.
const IOV_THETA: [f64; 3] = [0.3226552487466094, 8.428320965471485, 2.4764289059572344];
const IOV_OMEGA: [f64; 3] = [0.4720598638218765, 0.01286653298899196, 0.9190389062206252];
const IOV_KAPPA: f64 = 0.039038011478998845;
const IOV_SIGMA: f64 = 0.20066864578692198;

fn warfarin_iov() -> PreparedRun {
    prepare_run("examples/warfarin_iov.ferx", Some("data/warfarin_iov.csv"))
        .expect("warfarin_iov must prepare")
}

/// No subject may fall back to finite-difference inner gradients: the reviewer hint
/// names the analytic path as the engine, so the fixture has to stay on it.
fn assert_analytic(r: &FitResult) {
    assert!(
        !r.warnings
            .iter()
            .any(|w| w.contains("finite-difference inner gradients")),
        "fixture left the analytic path: {:?}",
        r.warnings
    );
}

/// API-built FOCE options. `interaction` is the default `true`, which is the point.
fn api_foce_sir(samples: usize, resamples: usize, seed: u64) -> FitOptions {
    let o = FitOptions {
        method: EstimationMethod::Foce,
        run_covariance_step: true,
        sir: true,
        sir_samples: samples,
        sir_resamples: resamples,
        sir_seed: Some(seed),
        ..FitOptions::default()
    };
    assert!(
        o.interaction,
        "fixture premise: API default interaction = true"
    );
    o
}

/// T4. warfarin_iov at its FOCE optimum, `outer_maxiter = 0`, covariance + SIR.
///
/// (i) The in-fit SIR scores the FOCE marginal: ESS above a floor the leaked FOCEI
/// weights cannot reach, and the recorded `interaction` is `false`.
/// (ii) Standalone `run_sir`, handed options that still say `interaction = true` and a
/// fit stamped with the leaked flag, reproduces the in-fit SIR bit for bit.
/// (iii) Standalone `run_covariance` with the same options reproduces the in-fit SEs bit
/// for bit.
#[test]
fn foce_sir_and_standalone_entries_score_the_foce_marginal() {
    let prep = warfarin_iov();
    let mut init = prep.init_params.clone();
    init.theta = IOV_THETA.to_vec();
    init.omega = OmegaMatrix::from_diagonal(&IOV_OMEGA, init.omega.eta_names.clone());
    let iov = init
        .omega_iov
        .as_ref()
        .expect("warfarin_iov declares a kappa");
    init.omega_iov = Some(OmegaMatrix::from_diagonal(
        &[IOV_KAPPA],
        iov.eta_names.clone(),
    ));
    init.sigma.values = vec![IOV_SIGMA];

    let mut opts = api_foce_sir(300, 100, 1710);
    opts.outer_maxiter = 0;
    let r = fit(&prep.parsed.model, &prep.population, &init, &opts).expect("fit must run");
    assert_analytic(&r);
    assert!(r.ofv.is_finite());
    let ess = r.sir_ess.expect("in-fit SIR must run");
    eprintln!("MEASURE T4 ofv={:.6} ess={ess:.4}", r.ofv);
    // (i) Realised at 300 draws, seed 1710 (macOS arm64 and Linux aarch64 agree in every printed digit): ESS 150.575 under
    // the fix, 3.184 with the post-fit resolution in `fit_inner` reverted (the FOCEI
    // weights). The floor of 50 is 3.0× below the first and 15.7× above the second.
    assert!(
        ess >= 50.0,
        "in-fit SIR ESS {ess} — weights are not the FOCE marginal's"
    );
    assert!(!r.interaction, "a FOCE fit must record interaction = false");

    // The standalone entries get the fit as a pre-#1710 `.fitrx` / R object carries it:
    // a FOCE fit stamped `interaction: true`. They must key on `fit.method` first.
    // Its SIR fields are cleared, so a standalone run that stopped filling one cannot
    // pass on the in-fit value it inherited.
    let mut stale = r.clone();
    stale.interaction = true;
    stale.sir_ess = None;
    stale.sir_ci_theta = None;
    stale.sir_ci_omega = None;
    stale.sir_ci_sigma = None;
    stale.sir_ci_kappa = None;

    // (ii) Standalone SIR, caller options unchanged (interaction = true).
    let sir = run_sir(
        &stale,
        Some(&prep.parsed.model),
        Some(&prep.population),
        &opts,
    )
    .expect("standalone run_sir must run");
    let sess = sir.sir_ess.expect("standalone SIR ESS");
    eprintln!("MEASURE T4 standalone ess={sess:.4}");
    assert_eq!(
        sess.to_bits(),
        ess.to_bits(),
        "standalone run_sir ESS {sess} differs from the in-fit {ess}"
    );
    assert_eq!(sir.sir_ci_theta, r.sir_ci_theta);
    assert_eq!(sir.sir_ci_omega, r.sir_ci_omega);
    assert_eq!(sir.sir_ci_sigma, r.sir_ci_sigma);
    assert!(
        r.sir_ci_kappa.is_some(),
        "in-fit SIR filled no kappa interval"
    );
    assert_eq!(sir.sir_ci_kappa, r.sir_ci_kappa);

    // (iii) Standalone covariance, same options.
    let cov = run_covariance(
        &stale,
        Some(&prep.parsed.model),
        Some(&prep.population),
        &opts,
    )
    .expect("standalone run_covariance must run");
    let (a, b) = (
        cov.se_theta.as_ref().expect("standalone SE"),
        r.se_theta.as_ref().expect("in-fit SE"),
    );
    eprintln!("MEASURE T4 se_theta standalone={a:?} in-fit={b:?}");
    for (k, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(x.is_finite() && y.is_finite(), "SE(theta[{k}]) not finite");
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "SE(theta[{k}]): standalone {x} vs in-fit {y}"
        );
    }
    assert_eq!(cov.se_omega, r.se_omega);
    assert_eq!(cov.se_sigma, r.se_sigma);
    assert_eq!(cov.se_kappa, r.se_kappa);
}

/// T5. The M3 non-interaction warning fires for an API-built FOCE fit with censored
/// observations (its gate read the leaked `true` and never opened), and stays silent
/// for FOCEI on the same data. Both sides in one test, so a gate stuck either way fails.
#[test]
fn m3_warning_follows_the_fitted_marginal() {
    let model = parse_model_file(Path::new("examples/warfarin_bloq.ferx"))
        .expect("warfarin BLOQ model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin_bloq.csv"), None, None)
        .expect("warfarin BLOQ data must load");
    let run = |method: EstimationMethod| {
        let o = FitOptions {
            method,
            outer_maxiter: 0,
            run_covariance_step: false,
            ..FitOptions::default()
        };
        assert!(
            o.interaction,
            "fixture premise: API default interaction = true"
        );
        fit(&model, &population, &model.default_params, &o).expect("M3 fit must run")
    };
    let has_m3 = |r: &FitResult| {
        r.warnings
            .iter()
            .any(|w| w.contains("M3 censoring under FOCE uses non-interaction"))
    };
    let foce = run(EstimationMethod::Foce);
    let focei = run(EstimationMethod::FoceI);
    assert_analytic(&foce);
    assert!(
        has_m3(&foce),
        "FOCE + M3 + censored rows must warn: {:?}",
        foce.warnings
    );
    assert!(!foce.interaction);
    assert!(
        !has_m3(&focei),
        "FOCEI must not carry the FOCE-M3 warning: {:?}",
        focei.warnings
    );
    assert!(focei.interaction);
}

/// T6. The issue's exit condition, end to end on the shipped model file: a converged
/// warfarin_iov FOCE fit, SIR 1000 / 250, seed 1705. ESS ≥ 100 (396.2 measured on
/// Linux aarch64 and macOS arm64; 4.87 under the leak) and every θ / ω / σ / κ estimate
/// inside its own SIR 95% interval (under the leak ω²(KA) = 0.919 sat outside
/// [0.078, 0.342]).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn warfarin_iov_foce_sir_contains_its_estimates() {
    let prep = warfarin_iov();
    let mut o = prep.parsed.fit_options.clone();
    o.run_covariance_step = true;
    o.sir = true;
    o.sir_samples = 1000;
    o.sir_resamples = 250;
    o.sir_seed = Some(1705);
    let r = fit(&prep.parsed.model, &prep.population, &prep.init_params, &o).expect("fit");
    assert_analytic(&r);
    assert_eq!(r.method, EstimationMethod::Foce);
    let ess = r.sir_ess.expect("SIR must run");
    eprintln!("MEASURE T6 ofv={:.6} ess={ess:.4}", r.ofv);
    assert!(ess.is_finite() && ess >= 100.0, "SIR ESS {ess} < 100");
    let check = |what: &str, est: &[f64], ci: &Option<Vec<(f64, f64)>>| {
        let ci = ci
            .as_ref()
            .unwrap_or_else(|| panic!("no SIR CI for {what}"));
        assert_eq!(ci.len(), est.len(), "{what}: CI length");
        for (k, (&e, &(lo, hi))) in est.iter().zip(ci).enumerate() {
            eprintln!("MEASURE T6 {what}[{k}] {e:.6} in [{lo:.6}, {hi:.6}]");
            assert!(
                lo.is_finite() && hi.is_finite() && lo <= e && e <= hi,
                "{what}[{k}] = {e} outside its SIR CI [{lo}, {hi}]"
            );
        }
    };
    check("theta", &r.theta, &r.sir_ci_theta);
    check("omega", r.omega.diagonal().as_slice(), &r.sir_ci_omega);
    check("sigma", &r.sigma, &r.sir_ci_sigma);
    let kappa = r
        .omega_iov
        .as_ref()
        .expect("warfarin_iov reports a kappa variance");
    check("kappa", kappa.diagonal().as_slice(), &r.sir_ci_kappa);
    assert!(!r.interaction);
}
