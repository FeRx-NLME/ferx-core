//! `profile`d θ level blocks (#1065): fits to convergence, gated `slow-tests`.
//!
//! - **D1 / D2**, the degenerate oracle: the profiled fit and the unprofiled fit of one
//!   model reach the same OFV, structural estimates and standard errors. The unprofiled
//!   side is **started at the profiled optimum** and must not improve on it by more than
//!   the bound — its own stopping point is loose at this size (the plan's M6 measured
//!   `Converged: YES` at `|g|∞` 2.6–14.9), so two independent fits would compare two
//!   stopping rules, not two objectives. D1 is F1 (linear in η and κ), D2 is F2
//!   (nonlinear in η).
//! - **A2 (fit)**: the profiled F2 fit against NONMEM's OFV with every level a `THETA`.
//! - **S1**: the plan's 100-study × 8-visit measurement — the profiled fit converges and
//!   its covariance is `r` and succeeds; unprofiled, the same model is 705 packed
//!   coordinates, and its covariance routes to `s` and has no standard error at all (M6).

use ferx_core::types::{CovarianceMethod, FitOptions, FitResult, ModelParameters, OmegaMatrix};
use std::path::PathBuf;
use std::time::Instant;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

const PROFILED: &str = "](0.0) profile";

fn prep(model: &str, data: &str, unprofiled: Option<&str>) -> ferx_core::PreparedRun {
    let text = std::fs::read_to_string(root().join(model)).unwrap();
    assert!(text.contains(PROFILED), "fixture moved");
    let text = match unprofiled {
        Some(decl) => text.replace(PROFILED, decl),
        None => text,
    };
    let path = std::env::temp_dir().join(format!(
        "ferx_profiled_fits_{}_{}_{}.ferx",
        std::process::id(),
        unprofiled.is_some(),
        model.replace('/', "_")
    ));
    std::fs::write(&path, text).unwrap();
    ferx_core::prepare_run(
        path.to_str().unwrap(),
        Some(root().join(data).to_str().unwrap()),
    )
    .unwrap_or_else(|e| panic!("{e}"))
}

fn fit(p: &ferx_core::PreparedRun, params: &ModelParameters, cov: bool) -> (FitResult, f64) {
    let opts = FitOptions {
        verbose: false,
        run_covariance_step: cov,
        ..p.parsed.fit_options.clone()
    };
    let t = Instant::now();
    let f = ferx_core::fit(&p.parsed.model, &p.population, params, &opts)
        .unwrap_or_else(|e| panic!("{e}"));
    (f, t.elapsed().as_secs_f64())
}

/// The fit's estimates as a start point for `template`'s model.
fn start_at(template: &ModelParameters, f: &FitResult) -> ModelParameters {
    let mut p = template.clone();
    p.theta = f.theta.clone();
    p.omega =
        OmegaMatrix::from_matrix(f.omega.clone(), p.omega.eta_names.clone(), p.omega.diagonal);
    if let (Some(iov), Some(m)) = (p.omega_iov.as_mut(), f.omega_iov.as_ref()) {
        *iov = OmegaMatrix::from_matrix(m.clone(), iov.eta_names.clone(), iov.diagonal);
    }
    p.sigma.values = f.sigma.clone();
    p
}

/// D1 / D2: profiled, then unprofiled from the profiled optimum.
fn degenerate_oracle(model: &str, data: &str, decl: &str, n_struct: usize) {
    let prof_prep = prep(model, data, None);
    let (prof, t_prof) = fit(&prof_prep, &prof_prep.init_params, true);
    assert!(prof.converged, "{:?}", prof.warnings);
    let plain_prep = prep(model, data, Some(decl));
    let (plain, t_plain) = fit(&plain_prep, &start_at(&plain_prep.init_params, &prof), true);
    assert!(prof.ofv.is_finite() && plain.ofv.is_finite());
    let improvement = prof.ofv - plain.ofv;
    let worst_theta = (0..n_struct)
        .map(|k| ((prof.theta[k] - plain.theta[k]) / plain.theta[k]).abs())
        .fold(0.0_f64, |a, v| {
            assert!(v.is_finite());
            a.max(v)
        });
    let (sp, su) = (
        prof.se_theta.as_ref().expect("profiled SE"),
        plain.se_theta.as_ref().expect("unprofiled SE"),
    );
    let worst_se = (0..sp.len())
        .map(|k| ((sp[k] - su[k]) / su[k]).abs())
        .fold(0.0_f64, |a, v| {
            assert!(v.is_finite());
            a.max(v)
        });
    eprintln!(
        "{model}: profiled OFV {:.9} ({t_prof:.1}s, {} evals), unprofiled from there {:.9} \
         ({t_plain:.1}s); improvement {improvement:.3e}, structural θ {worst_theta:.3e}, \
         SE {worst_se:.3e}",
        prof.ofv, prof.n_iterations, plain.ofv
    );
    assert!(
        improvement < D_OFV,
        "the unprofiled fit improved by {improvement}"
    );
    assert!(
        improvement > -D_OFV,
        "the unprofiled fit lost {improvement}"
    );
    assert!(worst_theta < D_THETA, "{worst_theta}");
    assert!(worst_se < D_SE, "{worst_se}");
}

/// Realised on Linux aarch64: D1 the unprofiled restart improves by 2.2e-6, D2 by
/// −2.4e-7; the structural θ move 2.5e-4 (D1) and 0 (D2) relative; SEs differ by 7.0e-3
/// (D1) and 4.7e-3 (D2). macOS arm64 within 5e-7 of each. Bands at ≥ 2.9× the worst.
const D_OFV: f64 = 1e-4;
const D_THETA: f64 = 1e-3;
const D_SE: f64 = 2e-2;

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn d1_a_profiled_fit_is_the_unprofiled_fit_linear_in_eta() {
    degenerate_oracle(
        "tests/data/profile_mbma/f1_profile.ferx",
        "tests/data/mbma_placebo/mbma_placebo.csv",
        "](0.0, -50.0, 50.0)",
        4,
    );
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn d2_a_profiled_fit_is_the_unprofiled_fit_nonlinear_in_eta() {
    degenerate_oracle(
        "nonmem_anchor/profile_mbma_fit.ferx",
        "nonmem_anchor/profile_mbma.csv",
        "](0.0, -50.0, 50.0)",
        3,
    );
}

/// The profiled F2 fit against NONMEM's own optimum, every level a `THETA` there.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn a2_a_profiled_fit_reaches_nonmems_optimum() {
    let text =
        std::fs::read_to_string(root().join("nonmem_anchor/results/profile_mbma.ext")).unwrap();
    let obj: f64 = text
        .lines()
        .find(|l| l.trim_start().starts_with("-1000000000"))
        .and_then(|l| l.split_whitespace().last())
        .unwrap()
        .parse()
        .unwrap();
    let p = prep(
        "nonmem_anchor/profile_mbma_fit.ferx",
        "nonmem_anchor/profile_mbma.csv",
        None,
    );
    let (f, t) = fit(&p, &p.init_params, false);
    eprintln!(
        "A2 fit: ferx {:.9} NONMEM {obj:.9} (Δ {:.3e}), {t:.1}s",
        f.ofv,
        f.ofv - obj
    );
    assert!(f.converged && f.ofv.is_finite());
    assert!((f.ofv - obj).abs() < A2_FIT_OFV, "{} vs {obj}", f.ofv);
}

/// Realised 2.4e-8 on Linux aarch64 (24 evaluations; 1.5e-8 macOS arm64), from the
/// model file's inits. A
/// stopping rule, not an engine difference, sets this; 1e-5 leaves room for another
/// platform's path to the optimum while still failing the 0.0185 BOBYQA stopped at.
const A2_FIT_OFV: f64 = 1e-5;

/// S1: 100 studies × 8 visits. Profiled: a 5-coordinate search and the `r` covariance
/// with a standard error for every structural parameter and every level. Unprofiled
/// (705 packed coordinates) it is fitted without its covariance step, which the plan's
/// M6 measured routing to `s` and failing (S rank-deficient: 100 subjects, 705
/// coordinates). The profiled OFV must not be above the unprofiled one; measured
/// (Linux aarch64, macOS the same to the printed digit) it is 0.42 below, the
/// unprofiled fit stopping early.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn s1_a_hundred_study_placebo_model_profiles_to_a_small_search() {
    let data = "tests/data/profile_mbma/s100t8.csv";
    let model = "tests/data/profile_mbma/s1_profile.ferx";
    let p = prep(model, data, None);
    let (prof, t_prof) = fit(&p, &p.init_params, true);
    assert!(prof.converged, "{:?}", prof.warnings);
    assert_eq!(prof.covariance_method, Some(CovarianceMethod::Hessian));
    let se = prof.se_theta.as_ref().expect("profiled SE");
    assert!(se.iter().all(|s| s.is_finite() && *s > 0.0));
    let cov = prof.covariance_matrix.as_ref().unwrap();
    let levels = &ferx_core::theta_level_values(&p.parsed.model, &prof.theta, Some(cov), None)
        .unwrap()["PLACEBO"];
    assert_eq!(levels.len(), 800);
    assert!(levels
        .iter()
        .all(|l| l.se.is_some_and(|s| s.is_finite() && s > 0.0)));

    let u = prep(model, data, Some("](0.0, -10.0, 10.0)"));
    let (plain, t_plain) = fit(&u, &u.init_params, false);
    eprintln!(
        "S1: profiled OFV {:.6} ({t_prof:.1}s, {} evals, n_parameters {}); unprofiled OFV \
         {:.6} ({t_plain:.1}s, {} evals, no covariance)",
        prof.ofv, prof.n_iterations, prof.n_parameters, plain.ofv, plain.n_iterations
    );
    assert_eq!(prof.n_parameters, plain.n_parameters);
    assert!(
        prof.ofv <= plain.ofv + 1e-3,
        "{} vs {}",
        prof.ofv,
        plain.ofv
    );
}
