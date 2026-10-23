//! `profile`d θ level blocks (#1065): fits to convergence, gated `slow-tests`.
//!
//! - **D1 / D2**, the degenerate oracle: the profiled fit and the unprofiled fit of one
//!   model reach the same OFV and structural estimates. The unprofiled side is **started
//!   at the profiled optimum** and must not improve on it by more than the bound — its own
//!   stopping point is loose at this size (the plan's M6 measured `Converged: YES` at
//!   `|g|∞` 2.6–14.9), so two independent fits would compare two stopping rules, not two
//!   objectives. D1 is F1 (linear in η and κ), D2 is F2 (nonlinear in η). The standard
//!   errors are compared at one fixed point instead, by C1 in `tests/profiled_levels.rs`
//!   (Tier 2, every PR).
//! - **A2 (fit)**: the profiled F2 fit against NONMEM's OFV with every level a `THETA`.
//! - **S1**: the plan's 100-study × 8-visit measurement — the profiled fit converges and
//!   its covariance is `r` and succeeds; unprofiled, the same model is 705 packed
//!   coordinates, and its covariance routes to `s` and has no standard error at all (M6).
//!
//! **Run time (#2130 follow-up).** The core slow leg has a 100-minute budget (#1239) and
//! was at 99m59s on `main` before this file landed. On the CI runner (one Rayon thread,
//! tests in parallel) the file took 559 s at `e44a7e2e`: S1 559 s, D2 491 s, D1 222 s,
//! A2 70 s. Most of D1/D2 was the unprofiled restart's FD `R` (41 coordinates for D2:
//! 2·41² reconverged objectives) and most of S1 the unprofiled 705-coordinate fit. Both
//! are gone. Each test is also ignored under `nn`: they read no nn code, and the `nn`
//! leg re-runs the workspace under the same budget (#2138 did the same).

use ferx_core::types::{CovarianceMethod, FitOptions, FitResult, ModelParameters, OmegaMatrix};
use std::path::PathBuf;
use std::time::Instant;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

const PROFILED: &str = "](0.0) profile";

/// A temp path no other call in this process gets: tests run on parallel threads, and two
/// that prep one fixture under one name raced — one read the other's half-written file
/// ("Missing [parameters] block", slow-tests run 39170264924; #2090 is the same bug).
fn unique_temp(stem: &str) -> std::path::PathBuf {
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!("{stem}_{}_{n}.ferx", std::process::id()))
}

fn prep(model: &str, data: &str, unprofiled: Option<&str>) -> ferx_core::PreparedRun {
    let text = std::fs::read_to_string(root().join(model)).unwrap();
    assert!(text.contains(PROFILED), "fixture moved");
    let text = match unprofiled {
        Some(decl) => text.replace(PROFILED, decl),
        None => text,
    };
    let path = unique_temp("ferx_profiled_fits");
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
    let (prof, t_prof) = fit(&prof_prep, &prof_prep.init_params, false);
    assert!(prof.converged, "{:?}", prof.warnings);
    let plain_prep = prep(model, data, Some(decl));
    let (plain, t_plain) = fit(
        &plain_prep,
        &start_at(&plain_prep.init_params, &prof),
        false,
    );
    assert!(prof.ofv.is_finite() && plain.ofv.is_finite());
    let improvement = prof.ofv - plain.ofv;
    let worst_theta = (0..n_struct)
        .map(|k| ((prof.theta[k] - plain.theta[k]) / plain.theta[k]).abs())
        .fold(0.0_f64, |a, v| {
            assert!(v.is_finite());
            a.max(v)
        });
    eprintln!(
        "{model}: profiled OFV {:.9} ({t_prof:.1}s, {} evals), unprofiled from there {:.9} \
         ({t_plain:.1}s); improvement {improvement:.3e}, structural θ {worst_theta:.3e}",
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
}

/// Realised on Linux aarch64: D1 the unprofiled restart improves by 1.6e-8, D2 by
/// −2.4e-7; the structural θ move 7.1e-8 (D1) and 0 (D2) relative. Bands at ≥ 4× the
/// worst.
const D_OFV: f64 = 1e-4;
const D_THETA: f64 = 1e-3;

#[test]
#[cfg_attr(
    any(not(feature = "slow-tests"), feature = "nn"),
    ignore = "slow: opt in with --features slow-tests; not run under nn, see the module doc"
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
    any(not(feature = "slow-tests"), feature = "nn"),
    ignore = "slow: opt in with --features slow-tests; not run under nn, see the module doc"
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
    any(not(feature = "slow-tests"), feature = "nn"),
    ignore = "slow: opt in with --features slow-tests; not run under nn, see the module doc"
)]
fn a2_a_profiled_fit_reaches_nonmems_optimum() {
    let obj = a2_nonmem_ofv();
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

/// NONMEM's minimum on the F2 anchor, from the `.ext` final row.
fn a2_nonmem_ofv() -> f64 {
    let text =
        std::fs::read_to_string(root().join("nonmem_anchor/results/profile_mbma.ext")).unwrap();
    text.lines()
        .find(|l| l.trim_start().starts_with("-1000000000"))
        .and_then(|l| l.split_whitespace().last())
        .unwrap()
        .parse()
        .unwrap()
}

/// #2285: A2 from starts near the file's own, where a fit that says `converged: true`
/// must be on NONMEM's optimum. Before #2285 the profiled L-BFGS stopped on the
/// unprofiled `ftol_rel = 1e-6` (≈ 1e-4 absolute at this OFV) while still descending:
/// (45, 7.5, 2) reported `converged: true` 3.88e-4 above the optimum on `main`, and
/// (45, 8, 2.2) 1.49e-4 above it on #2280's inner solver. Those are the two starts here;
/// the other five of the issue's seven landed either way, and each fit costs as much
/// as A2. The issue would accept `converged: false` here; this asks for the optimum,
/// which every one of the seven now reaches. Realised on macOS arm64 at #2285's head:
/// Δ 3.5e-8 (29 evaluations) and 3.8e-8 (31), against 3.7e-8 from the file's own start.
#[test]
#[cfg_attr(
    any(not(feature = "slow-tests"), feature = "nn"),
    ignore = "slow: opt in with --features slow-tests; not run under nn, see the module doc"
)]
fn a2_a_nearby_start_reaches_nonmems_optimum() {
    let obj = a2_nonmem_ofv();
    let p = prep(
        "nonmem_anchor/profile_mbma_fit.ferx",
        "nonmem_anchor/profile_mbma.csv",
        None,
    );
    assert_eq!(p.init_params.theta[..3], [45.0, 8.0, 2.0], "fixture moved");
    for start in [[45.0, 7.5, 2.0], [45.0, 8.0, 2.2]] {
        let mut params = p.init_params.clone();
        params.theta[..3].copy_from_slice(&start);
        let (f, t) = fit(&p, &params, false);
        eprintln!(
            "A2 from {start:?}: ferx {:.9} (Δ {:.3e}), converged {}, {} evals, {t:.1}s",
            f.ofv,
            f.ofv - obj,
            f.converged,
            f.n_iterations
        );
        assert!(
            f.converged && f.ofv.is_finite(),
            "{start:?}: {:?}",
            f.warnings
        );
        assert!(
            (f.ofv - obj).abs() < A2_FIT_OFV,
            "{start:?}: converged at {} vs NONMEM {obj}",
            f.ofv
        );
    }
}

/// Realised 2.4e-8 on Linux aarch64 (24 evaluations; 1.5e-8 macOS arm64), from the
/// model file's inits. A
/// stopping rule, not an engine difference, sets this; 1e-5 leaves room for another
/// platform's path to the optimum while still failing the 0.0185 BOBYQA stopped at.
const A2_FIT_OFV: f64 = 1e-5;

/// S1: 100 studies × 8 visits. Profiled: a 5-coordinate search and the `r` covariance
/// with a standard error for every structural parameter and every level. The profiled
/// OFV must not be above the unprofiled fit's, which is the measured constant
/// [`S1_UNPROFILED_OFV`] rather than a second fit: that fit is a 705-coordinate FD search
/// and was most of this test's CI time. Measured, the profiled fit is 0.42 below it.
#[test]
#[cfg_attr(
    any(not(feature = "slow-tests"), feature = "nn"),
    ignore = "slow: opt in with --features slow-tests; not run under nn, see the module doc"
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

    eprintln!(
        "S1: profiled OFV {:.6} ({t_prof:.1}s, {} evals, n_parameters {}); unprofiled \
         OFV {S1_UNPROFILED_OFV} (measured)",
        prof.ofv, prof.n_iterations, prof.n_parameters
    );
    assert_eq!(
        prof.n_parameters, 705,
        "the unprofiled count: 3 θ, 700 levels, ω, σ"
    );
    assert!(
        prof.ofv <= S1_UNPROFILED_OFV + 1e-3,
        "{} vs {S1_UNPROFILED_OFV}",
        prof.ofv
    );
}

/// The unprofiled S1 fit's OFV (705 packed coordinates, from the model file's inits, no
/// covariance step): −3579.167062, the same six digits on Linux aarch64
/// (`tools/linux-test.sh`) and macOS arm64 at #2130's head, and in the plan's M6. It
/// stops early (`|g|∞` 2.6); the profiled fit reaches −3579.589899. Re-measure it if the
/// unprofiled engine changes: the assertion only needs the profiled fit not to be above
/// it.
const S1_UNPROFILED_OFV: f64 = -3579.167062;
