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

/// NONMEM's FOCE optimum on `examples/warfarin.ferx` (the `warfarin_foce_cwres` anchor's
/// `.ext`), so the GN fixture below scores a converged point without a loop.
const WARF_THETA: [f64; 3] = [
    1.3296190773543662E-01,
    7.7305172240926252E+00,
    7.2537506677117802E-01,
];
const WARF_OMEGA: [f64; 3] = [
    2.8596520575084699E-02,
    9.5781911176013226E-03,
    3.4881225592887283E-01,
];
const WARF_SIGMA_VAR: f64 = 1.1548425869877831E-04;

/// The pass-through branch of `fitted_marginal_options`: for a method the rule does not
/// fix (here Gauss-Newton), the standalone entries take the flag the **fit** recorded,
/// not the caller's. A `FoceGn` fit with `interaction = false`, re-run through
/// `run_covariance` with `interaction = true` options, must reproduce the in-fit SEs bit
/// for bit — reading the caller's flag instead differentiates the interaction marginal
/// (#1725 review round 1, finding 1).
#[test]
fn standalone_covariance_keeps_a_gn_fits_own_interaction() {
    let prep = prepare_run("examples/warfarin.ferx", Some("data/warfarin.csv"))
        .expect("warfarin must prepare");
    let mut init = prep.init_params.clone();
    init.theta = WARF_THETA.to_vec();
    init.omega = OmegaMatrix::from_diagonal(&WARF_OMEGA, init.omega.eta_names.clone());
    init.sigma.values = vec![WARF_SIGMA_VAR.sqrt()];
    let opts = FitOptions {
        method: EstimationMethod::FoceGn,
        interaction: false,
        outer_maxiter: 0,
        run_covariance_step: true,
        ..FitOptions::default()
    };
    let r = fit(&prep.parsed.model, &prep.population, &init, &opts).expect("GN fit must run");
    assert_analytic(&r);
    assert_eq!(r.method, EstimationMethod::FoceGn);
    assert!(
        !r.interaction,
        "a GN fit keeps the caller's interaction = false"
    );
    let caller = FitOptions {
        interaction: true,
        ..opts.clone()
    };
    let cov = run_covariance(
        &r,
        Some(&prep.parsed.model),
        Some(&prep.population),
        &caller,
    )
    .expect("standalone run_covariance must run");
    let (a, b) = (
        cov.se_theta.as_ref().expect("standalone SE"),
        r.se_theta.as_ref().expect("in-fit SE"),
    );
    eprintln!("MEASURE GN se_theta standalone={a:?} in-fit={b:?}");
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

// ---------------------------------------------------------------------------------------
// #1755 / #1704: SIR scores with the fit's own objective — its final estimating
// method's marginal, and the K-class marginal for a `[mixture]` model — at parameters
// rebuilt from the fit, per-class overrides included. `run_sir(fit, default options)`
// therefore reproduces the in-fit SIR bit for bit.
//
// Engine: warfarin (one-compartment oral) and the `[mixture]` one-compartment IV model
// are inside the analytic `Dual2` scope; every fit asserts no FD fallback.
// ---------------------------------------------------------------------------------------

/// SIR settings shared by the #1755 fixtures: `FitOptions::default()` plus three SIR
/// fields, which is what a caller hands `run_sir` ("default options").
fn sir_defaults() -> FitOptions {
    FitOptions {
        verbose: false,
        sir_samples: 400,
        sir_resamples: 200,
        sir_seed: Some(7),
        ..FitOptions::default()
    }
}

/// ferx's Laplace optimum on `examples/warfarin.ferx` (OFV −285.9702267410695), packed
/// (`[ln θ₁..₃, ln L_Ω₁..₃, ln σ]`) — a full Laplace fit from the file's values, macOS
/// arm64. Unpacked onto the file's template, so the start is that exact point.
const WARF_LAPLACE_PACKED: [f64; 7] = [
    -2.019760138501293,
    2.0460733046385786,
    -0.20960659528716724,
    -1.7773234722669555,
    -2.3234007261778418,
    -0.5452958750862014,
    -4.550252146959798,
];

/// warfarin at its Laplace optimum, `outer_maxiter = 0`, covariance + SIR, under
/// `method` / `methods`.
fn warfarin_sir_fit(
    method: EstimationMethod,
    methods: Vec<EstimationMethod>,
) -> (PreparedRun, FitResult) {
    let prep = prepare_run("examples/warfarin.ferx", Some("data/warfarin.csv"))
        .expect("warfarin must prepare");
    let init = ferx_core::estimation::parameterization::unpack_params(
        &WARF_LAPLACE_PACKED,
        &prep.init_params,
    );
    let opts = FitOptions {
        method,
        methods,
        outer_maxiter: 0,
        run_covariance_step: true,
        sir: true,
        ..sir_defaults()
    };
    let r = fit(&prep.parsed.model, &prep.population, &init, &opts).expect("fit must run");
    assert_analytic(&r);
    (prep, r)
}

/// A copy of `r` with every SIR output cleared, so a standalone run that stopped
/// filling one cannot pass on the in-fit value it inherited.
fn cleared(r: &FitResult) -> FitResult {
    let mut c = r.clone();
    c.sir_ess = None;
    c.sir_ci_theta = None;
    c.sir_ci_omega = None;
    c.sir_ci_sigma = None;
    c.sir_ci_kappa = None;
    c
}

/// ESS and every SIR interval of `a` equal `b`'s, bit for bit. `side` names the
/// comparison in the failure message.
fn assert_sir_identical(a: &FitResult, b: &FitResult, side: &str) {
    let (ea, eb) = (
        a.sir_ess
            .unwrap_or_else(|| panic!("{side}: no ESS on the first")),
        b.sir_ess
            .unwrap_or_else(|| panic!("{side}: no ESS on the second")),
    );
    assert!(ea.is_finite() && eb.is_finite(), "{side}: ESS not finite");
    assert_eq!(ea.to_bits(), eb.to_bits(), "{side}: ESS {ea} vs {eb}");
    assert!(a.sir_ci_theta.is_some(), "{side}: no theta interval");
    assert_eq!(a.sir_ci_theta, b.sir_ci_theta, "{side}: theta CIs");
    assert_eq!(a.sir_ci_omega, b.sir_ci_omega, "{side}: omega CIs");
    assert_eq!(a.sir_ci_sigma, b.sir_ci_sigma, "{side}: sigma CIs");
    assert_eq!(a.sir_ci_kappa, b.sir_ci_kappa, "{side}: kappa CIs");
}

/// T1. A Laplace fit's `run_sir` with default options (`method = FoceI`) scores the
/// Laplace marginal, as the in-fit SIR does. Measured on Linux aarch64: both ESS
/// 173.12789709232797; with the standalone builder not taking the fit's method the
/// standalone run weights with FOCEI, ESS 173.21083683015937 (macOS arm64 reaches a
/// different ESS on this fixture, 141.06, and the identity holds there too).
#[test]
fn laplace_run_sir_default_options_is_identical_to_in_fit() {
    let (prep, r) = warfarin_sir_fit(EstimationMethod::Laplace, Vec::new());
    assert_eq!(r.method, EstimationMethod::Laplace);
    let opts = sir_defaults();
    assert_eq!(
        opts.method,
        EstimationMethod::FoceI,
        "fixture premise: the caller's method is not the fit's"
    );
    let s = run_sir(
        &cleared(&r),
        Some(&prep.parsed.model),
        Some(&prep.population),
        &opts,
    )
    .expect("standalone run_sir must run");
    eprintln!(
        "MEASURE T1 in-fit ess={:?} standalone ess={:?}",
        r.sir_ess, s.sir_ess
    );
    assert_sir_identical(&s, &r, "run_sir vs in-fit (Laplace)");
}

/// T2. A chain `[focei, laplace]` with the top-level `method` left at FoceI: the
/// estimates are Laplace's, so the in-fit SIR **and** `run_sir` score the Laplace
/// marginal. The identity alone cannot see a defect both sides share, so each side is
/// pinned separately to a run that scores the same estimates as a single-stage Laplace
/// fit — the message names the side that moved. Before the fix the in-fit SIR scored
/// FOCEI. Measured on Linux aarch64: Laplace 173.12789709232797, FOCEI
/// 173.21083683015937. Mutating each side alone reddens this test with that side's name.
#[test]
fn chain_sir_scores_the_final_estimating_method() {
    let (prep, r) = warfarin_sir_fit(
        EstimationMethod::FoceI,
        vec![EstimationMethod::FoceI, EstimationMethod::Laplace],
    );
    assert_eq!(
        r.method,
        EstimationMethod::Laplace,
        "the chain's final stage"
    );

    // Reference: `run_sir` with caller options that already say Laplace, so no
    // resolution from the fit is needed to get it right.
    let laplace_opts = FitOptions {
        method: EstimationMethod::Laplace,
        ..sir_defaults()
    };
    let reference = run_sir(
        &cleared(&r),
        Some(&prep.parsed.model),
        Some(&prep.population),
        &laplace_opts,
    )
    .expect("reference run_sir must run");
    // The straddle: the same estimates scored with FOCEI must differ, or the pair below
    // could not tell the two marginals apart.
    let focei = {
        let mut f = cleared(&r);
        f.method = EstimationMethod::FoceI;
        run_sir(
            &f,
            Some(&prep.parsed.model),
            Some(&prep.population),
            &sir_defaults(),
        )
        .expect("FOCEI-scored run_sir must run")
    };
    eprintln!(
        "MEASURE T2 in-fit={:?} laplace-ref={:?} focei={:?}",
        r.sir_ess, reference.sir_ess, focei.sir_ess
    );
    assert_ne!(
        focei.sir_ess.map(f64::to_bits),
        reference.sir_ess.map(f64::to_bits),
        "fixture premise: FOCEI and Laplace weights must differ on this fit"
    );

    // Side 1: the in-fit SIR (`fit()`'s `scoring_options(final_method, …)`).
    assert_sir_identical(&r, &reference, "in-fit SIR vs Laplace scoring");
    // Side 2: `run_sir` with default options (`fitted_marginal_options` over `fit.method`).
    let s = run_sir(
        &cleared(&r),
        Some(&prep.parsed.model),
        Some(&prep.population),
        &sir_defaults(),
    )
    .expect("standalone run_sir must run");
    assert_sir_identical(&s, &reference, "run_sir vs Laplace scoring");
}

/// T7. `run_covariance` on a Laplace fit takes the Hessian of the Laplace objective,
/// whatever `method` the caller passes. Not bitwise: the in-fit Laplace covariance
/// re-solves EBEs at the AGQ stage's `inner_tol.min(1e-8)`, which a `FitResult` does
/// not record (#1758). Measured max difference on the correlation scale: 1.012e-6 on
/// Linux aarch64 (1.43e-6 on macOS arm64); 1.131e-3 when `run_covariance` takes the
/// caller's method (FOCEI). The bound of 1e-4 is 99× the Linux value and 11× below the
/// mutation.
#[test]
fn laplace_run_covariance_takes_the_fits_method() {
    let (prep, r) = warfarin_sir_fit(EstimationMethod::Laplace, Vec::new());
    let a = r.covariance_matrix.as_ref().expect("in-fit covariance");
    let c = run_covariance(
        &r,
        Some(&prep.parsed.model),
        Some(&prep.population),
        &sir_defaults(),
    )
    .expect("standalone run_covariance must run");
    let b = c.covariance_matrix.as_ref().expect("standalone covariance");
    assert_eq!(a.shape(), b.shape());
    // On the correlation scale, `|Δ C_ij| / √(C_ii C_jj)`: an element-relative
    // difference is dominated by near-zero covariances.
    let mut worst = 0.0f64;
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            assert!(
                a[(i, j)].is_finite() && b[(i, j)].is_finite(),
                "non-finite entry"
            );
            let scale = (a[(i, i)] * a[(j, j)]).sqrt();
            assert!(
                scale > 0.0,
                "covariance diagonal not positive at ({i}, {j})"
            );
            worst = worst.max((a[(i, j)] - b[(i, j)]).abs() / scale);
        }
    }
    eprintln!("MEASURE T7 max scaled diff {worst:.3e}");
    assert!(
        worst <= T7_BOUND,
        "run_covariance differs from the in-fit Laplace covariance by {worst:.3e}"
    );
}
const T7_BOUND: f64 = 1e-4;

/// The `tests/mixture_nonmem.rs` model with Ω and Σ **free** and a class-2 Ω override.
const MIX_OVERRIDE: &str = r"
[parameters]
  theta TVCL1(1.2, 0.01, 100.0)
  theta TVCL2(2.5, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma EPS ~ 0.04

[mixture]
  nsub = 2
  logit(1) = MIXL
  omega(2) ETA_CL ~ 0.30

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

/// ferx's FOCEI optimum of `MIX_OVERRIDE` on `tests/nonmem/mixture_iv.csv`, packed
/// (`[ln θ₁..₃, MIXL, ln L_Ω, ln σ, ln L_Ω(2)]`): OFV 298.3278664427295, class-1 ω²
/// 0.02197, class-2 override 0.0972 (from 0.30). A full fit from the file's values.
const MIX_PACKED: [f64; 7] = [
    -0.07780620268043612,
    0.993379148024101,
    2.3028646917499356,
    -0.30316582094104605,
    -1.9090553430428145,
    -1.5518494729012504,
    -1.1652410428760827,
];

/// The `[mixture]` fit at `MIX_PACKED` (`outer_maxiter = 0`), covariance + SIR. With
/// `with_override = false` the class-2 line is dropped and the start is the first six
/// coordinates.
fn mixture_sir_fit(
    with_override: bool,
) -> (ferx_core::CompiledModel, ferx_core::Population, FitResult) {
    let src = if with_override {
        MIX_OVERRIDE.to_string()
    } else {
        MIX_OVERRIDE.replace("  omega(2) ETA_CL ~ 0.30\n", "")
    };
    let model = ferx_core::parse_model_string(&src).expect("mixture model must parse");
    let pop = read_nonmem_csv(
        Path::new("tests/nonmem/mixture_iv.csv"),
        Some(&["WT"]),
        None,
    )
    .expect("mixture data must load");
    let n = if with_override { 7 } else { 6 };
    let init = ferx_core::estimation::parameterization::unpack_params(
        &MIX_PACKED[..n],
        &model.default_params,
    );
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        outer_maxiter: 0,
        run_covariance_step: true,
        sir: true,
        ..sir_defaults()
    };
    let r = fit(&model, &pop, &init, &opts).expect("mixture fit must run");
    assert_analytic(&r);
    (model, pop, r)
}

/// T4. A `[mixture]` fit with a class-2 Ω override: the in-fit SIR scores the K-class
/// marginal (ESS above a floor, every estimate inside its own interval), and `run_sir`
/// with default options rebuilds the override and reproduces it bit for bit.
///
/// Measured ESS 22.529646647511825 of 400 (Linux aarch64; macOS arm64 the same to 11
/// digits). Under the one-class scorer every draw's objective sat 97–724 units off the
/// mixture one and the ESS fell to 1.0947; the identity stays green under that
/// mutation (both sides share the scorer), so the floor of 8 — 2.8× below the fix,
/// 7.3× above the mutation — and the brackets are what kill it. Before #1704 `run_sir` was refused
/// (covariance 7×7 against 6 packed coordinates).
#[test]
fn mixture_override_run_sir_is_identical_to_in_fit() {
    let (model, pop, r) = mixture_sir_fit(true);
    let ess = r.sir_ess.expect("in-fit SIR must run");
    eprintln!("MEASURE T4 ofv={:?} ess={ess:?}", r.ofv);
    assert!(
        ess.is_finite() && ess >= T4_ESS_FLOOR,
        "in-fit mixture SIR ESS {ess}: the draws are not scored with the mixture marginal"
    );
    let bracket = |what: &str, est: &[f64], ci: &Option<Vec<(f64, f64)>>| {
        let ci = ci
            .as_ref()
            .unwrap_or_else(|| panic!("no SIR CI for {what}"));
        assert_eq!(ci.len(), est.len(), "{what}: CI length");
        for (k, (&e, &(lo, hi))) in est.iter().zip(ci).enumerate() {
            assert!(
                lo.is_finite() && hi.is_finite() && lo <= e && e <= hi,
                "{what}[{k}] = {e} outside its SIR CI [{lo}, {hi}]"
            );
        }
    };
    bracket("theta", &r.theta, &r.sir_ci_theta);
    bracket("omega", r.omega.diagonal().as_slice(), &r.sir_ci_omega);
    bracket("sigma", &r.sigma, &r.sir_ci_sigma);

    let s = run_sir(&cleared(&r), Some(&model), Some(&pop), &sir_defaults())
        .expect("standalone run_sir on a mixture-override fit must run");
    assert_sir_identical(&s, &r, "run_sir vs in-fit (mixture override)");
}
const T4_ESS_FLOOR: f64 = 8.0;

/// T5. `run_covariance` on the mixture-override fit reproduces the in-fit 7×7 matrix bit
/// for bit (before #1704: `Ok` with `covariance_matrix: None`).
#[test]
fn mixture_override_run_covariance_is_identical_to_in_fit() {
    let (model, pop, r) = mixture_sir_fit(true);
    let a = r.covariance_matrix.as_ref().expect("in-fit covariance");
    assert_eq!(a.nrows(), 7, "the override is a packed coordinate");
    let c = run_covariance(&r, Some(&model), Some(&pop), &sir_defaults())
        .expect("standalone run_covariance must run");
    let b = c
        .covariance_matrix
        .as_ref()
        .expect("standalone run_covariance returned no matrix");
    assert_eq!(a.shape(), b.shape());
    for (k, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(x.is_finite(), "covariance[{k}] not finite");
        assert_eq!(x.to_bits(), y.to_bits(), "covariance[{k}]: {x} vs {y}");
    }
}

/// T6. Both sides of the "override values are not stored" gate, and both `Err` rows of
/// the message's input space:
///
/// | fit         | `packed_estimate` | outcome                                     |
/// |-------------|-------------------|---------------------------------------------|
/// | override    | `None`            | `Err`: names the override and why it is gone |
/// | override    | shorter / longer  | `Err`: layout mismatch, no `.fitrx` story   |
/// | override    | right length, θ edited or vector damaged | `Err`: stale, no layout / `.fitrx` story (#1815) |
/// | no override | `None`            | `Ok`, identical to the in-fit SIR            |
///
/// Each sentence of each message is asserted by a substring, so deleting one reddens
/// this test; the forbidden claims (refitting, declared values) are asserted absent.
#[test]
fn mixture_without_packed_estimate() {
    let (model, pop, r) = mixture_sir_fit(true);

    // Row 1: the override fit as a `.fitrx` / R object / SAEM fit carries it.
    let mut stored = cleared(&r);
    stored.packed_estimate = None;
    for (entry, res) in [
        (
            "run_sir",
            run_sir(&stored, Some(&model), Some(&pop), &sir_defaults()).map(|_| ()),
        ),
        (
            "run_covariance",
            run_covariance(&stored, Some(&model), Some(&pop), &sir_defaults()).map(|_| ()),
        ),
    ] {
        let e = res.expect_err("an override fit without its packed estimate must be refused");
        for must in [
            format!("{entry}: this [mixture] fit"),
            "per-class override(s) omega(2) ETA_CL".to_string(),
            "a FitResult does not store their fitted values".to_string(),
            "carried by the result of a FOCE, FOCEI, Laplace or Gauss-Newton fit()".to_string(),
            "and by a .fitrx bundle save_fit wrote from one".to_string(),
            "a fit built in R, a .fitrx bundle saved before #1815 or written by ferx-r, or a \
             fit estimated by SAEM, IMP or Bayes lacks them (#1765)"
                .to_string(),
        ] {
            assert!(
                e.to_string().contains(&must),
                "{entry} message lacks {must:?}: {e}"
            );
        }
        for must_not in ["refit", "declared", "layout", "a fit read from .fitrx"] {
            assert!(
                !e.to_string().contains(must_not),
                "{entry} message says {must_not:?}: {e}"
            );
        }
    }

    // Row 2: a packed estimate of another layout, shorter and longer — a `>=` length
    // guard passes the longer one and reads the overrides from the wrong slots (#1768
    // review finding 2).
    for other_len in [6, 8] {
        let mut v = MIX_PACKED.to_vec();
        v.resize(other_len, 0.0);
        let mut other = cleared(&r);
        other.packed_estimate = Some(v);
        let e = run_sir(&other, Some(&model), Some(&pop), &sir_defaults())
            .expect_err("a packed estimate of the wrong length must be refused");
        for must in [
            format!(
                "run_sir: the fit's packed estimate has {other_len} coordinates but this \
                 model's parameter layout has 7"
            ),
            "Supply the model the fit was estimated with.".to_string(),
        ] {
            assert!(
                e.to_string().contains(&must),
                "layout message lacks {must:?}: {e}"
            );
        }
        assert!(
            !e.to_string().contains(".fitrx"),
            "layout message tells the .fitrx story: {e}"
        );
    }

    // Row 2b (#1815): a packed estimate of the right length that does not unpack to the
    // fit's reported estimates. Two causes, one message: θ edited after the fit, and a
    // damaged vector (the override slot NaN) beside unchanged estimates (review r1 #1 —
    // before the pre-check that one panicked). The override cannot be trusted, and the
    // message says so without the length or `.fitrx` stories.
    let mut edited = cleared(&r);
    edited.theta[0] *= 1.01;
    let mut damaged = cleared(&r);
    damaged.packed_estimate.as_mut().unwrap()[6] = f64::NAN;
    for (cause, stale) in [("theta edited", &edited), ("vector damaged", &damaged)] {
        assert_eq!(stale.packed_estimate.as_ref().map(Vec::len), Some(7));
        for (entry, res) in [
            (
                "run_sir",
                run_sir(stale, Some(&model), Some(&pop), &sir_defaults()).map(|_| ()),
            ),
            (
                "run_covariance",
                run_covariance(stale, Some(&model), Some(&pop), &sir_defaults()).map(|_| ()),
            ),
        ] {
            let e = res.expect_err("a stale override fit must be refused");
            for must in [
                format!("{entry}: the fit's packed estimate does not reproduce its reported"),
                "estimates (theta / Omega / Sigma / Omega_IOV / residual correlations)".to_string(),
                "so the [mixture] override values it carries cannot be trusted".to_string(),
                "the estimates or the packed estimate were changed after the fit, or this is \
                 not the model the fit was estimated with"
                    .to_string(),
            ] {
                assert!(
                    e.to_string().contains(&must),
                    "{cause}: {entry} stale message lacks {must:?}: {e}"
                );
            }
            for must_not in ["layout", "coordinates", ".fitrx", "lacks them"] {
                assert!(
                    !e.to_string().contains(must_not),
                    "{cause}: {entry} stale message says {must_not:?}: {e}"
                );
            }
        }
    }

    // Row 3: no override — the base is exact, so nothing is missing.
    let (model0, pop0, r0) = mixture_sir_fit(false);
    let mut stored0 = cleared(&r0);
    stored0.packed_estimate = None;
    let s0 = run_sir(&stored0, Some(&model0), Some(&pop0), &sir_defaults())
        .expect("a no-override mixture fit needs no packed estimate");
    assert_sir_identical(&s0, &r0, "run_sir vs in-fit (no-override mixture)");
}

/// #1815 T4. The mixture-override fit saved to `.fitrx` and loaded back: `run_covariance`
/// and `run_sir` reproduce the in-fit covariance and SIR bit for bit, because the bundle
/// carries the packed vector that holds the override's fitted value. Premise: the same
/// reloaded fit without the vector — every bundle before #1815 — is refused (#1765).
///
/// Mutation — `load_fit` reads `packed_estimate: None`: both claims are refused.
#[test]
fn mixture_override_reloaded_from_fitrx_is_identical_to_in_fit() {
    use ferx_core::io::fitrx::{load_fit, save_fit, SaveFitOptions};
    let (model, pop, r) = mixture_sir_fit(true);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixture.fitrx");
    save_fit(&r, &pop, MIX_OVERRIDE, &path, SaveFitOptions::default()).expect("save_fit");
    let loaded = load_fit(&path).expect("load_fit").fit;

    let mut legacy = cleared(&loaded);
    legacy.packed_estimate = None;
    let e = run_sir(&legacy, Some(&model), Some(&pop), &sir_defaults())
        .expect_err("premise: a reloaded override fit without its packed vector is refused");
    assert!(
        e.to_string()
            .contains("a FitResult does not store their fitted values"),
        "premise: the refusal must be the #1765 one, not another error: {e}"
    );

    let c = run_covariance(&loaded, Some(&model), Some(&pop), &sir_defaults())
        .expect("run_covariance on the reloaded mixture fit must run");
    let (a, b) = (
        r.covariance_matrix.as_ref().expect("in-fit covariance"),
        c.covariance_matrix
            .as_ref()
            .expect("run_covariance on the reloaded fit returned no matrix"),
    );
    assert_eq!(a.shape(), b.shape());
    for (k, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(x.is_finite(), "covariance[{k}] not finite");
        assert_eq!(x.to_bits(), y.to_bits(), "covariance[{k}]: {x} vs {y}");
    }

    let s = run_sir(&cleared(&loaded), Some(&model), Some(&pop), &sir_defaults())
        .expect("run_sir on the reloaded mixture fit must run");
    assert_sir_identical(&s, &r, "run_sir on reloaded vs in-fit (mixture override)");
}

/// #1815 review r1 #2: VI's packed vector is always `Stale` (its stored Ω is 1 ULP off
/// the unpack, #1847), so an in-memory VI fit of an override model would be refused with
/// the Stale message. That cannot happen: `fit()` refuses VI on any `[mixture]` model
/// before a stage runs. Pinned so the disposition stays true if VI is ever wired for
/// mixtures — this row then fails and says to re-check the Stale message for VI.
#[test]
fn vi_never_reaches_a_mixture_override_fit() {
    let model = ferx_core::parse_model_string(MIX_OVERRIDE).expect("mixture model must parse");
    let pop = read_nonmem_csv(
        Path::new("tests/nonmem/mixture_iv.csv"),
        Some(&["WT"]),
        None,
    )
    .expect("mixture data must load");
    let opts = FitOptions {
        method: EstimationMethod::Vi,
        ..sir_defaults()
    };
    let e = fit(&model, &pop, &model.default_params, &opts)
        .map(|_| ())
        .expect_err("VI on a [mixture] model must be refused up front");
    assert!(
        e.to_string().contains("is not yet wired for mixtures"),
        "VI x [mixture] must be the up-front method refusal: {e}"
    );
}
