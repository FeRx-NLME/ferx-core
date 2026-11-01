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
//!   its covariance is `r` and succeeds; unprofiled, the same model is 706 packed
//!   coordinates and reaches the same OFV. Since #2326 it carries a study baseline η both
//!   arms read, so `auto` splits the block per study; without one the 800 levels are one
//!   group (#2135) and the fit does not finish in the slow tier.
//! - **R1**: `run_covariance` on the converged F1 fit repeats its covariance step bit for
//!   bit, in memory and after a `.fitrx` reload (#2213).
//!
//! **Run time (#2130 follow-up).** The core slow leg has a 100-minute budget (#1239) and
//! was at 99m59s on `main` before this file landed. On the CI runner (one Rayon thread,
//! tests in parallel) the file took 559 s at `e44a7e2e`: S1 559 s, D2 491 s, D1 222 s,
//! A2 70 s. Most of D1/D2 was the unprofiled restart's FD `R` (41 coordinates for D2:
//! 2·41² reconverged objectives) and most of S1 the unprofiled 705-coordinate fit. Both
//! were dropped; S1's unprofiled fit is back since #2326, at 0.7–9 s. Each test is also
//! ignored under `nn`: they read no nn code, and the `nn` leg re-runs the workspace under
//! the same budget (#2138 did the same).

use ferx_core::types::{
    CovarianceMethod, FitOptions, FitResult, ModelParameters, OmegaMatrix, Optimizer,
};
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
    degenerate_oracle_with(model, data, None, decl, n_struct);
}

/// The degenerate oracle with the profiled block line rewritten to `prof_decl` when given
/// (D2b, a bounded profiled block), returning both fits.
fn degenerate_oracle_with(
    model: &str,
    data: &str,
    prof_decl: Option<&str>,
    decl: &str,
    n_struct: usize,
) -> (ferx_core::PreparedRun, FitResult, FitResult) {
    let prof_prep = prep(model, data, prof_decl);
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
    (plain_prep, prof, plain)
}

/// D2b (#2112): the degenerate oracle on a **bounded** block — profiled with the box, then
/// unprofiled with the same bounds on every level, from the profiled optimum. The level the
/// box binds is on its bound in both fits (exactly, in the profiled one) and the control
/// level, near the same bound, is strictly inside it in both: the straddle, asserted, so
/// the equality is about a constrained optimum and not two unconstrained ones. Bounds
/// below 0 on both sides, so neither fit log-packs a level.
fn bounded_degenerate_oracle(
    model: &str,
    data: &str,
    bx: &str,
    n_struct: usize,
    upper: f64,
    pinned: &str,
    control: &str,
) {
    let (p, prof, plain) = degenerate_oracle_with(
        model,
        data,
        Some(&format!("]{bx} profile")),
        &format!("]{bx}"),
        n_struct,
    );
    let idx = |name: &str| {
        p.init_params
            .theta_names
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("{name}"))
    };
    let (k, c) = (idx(pinned), idx(control));
    eprintln!(
        "D2b {model}: {pinned} profiled {} unprofiled {}; {control} profiled {} unprofiled {}",
        prof.theta[k], plain.theta[k], prof.theta[c], plain.theta[c]
    );
    assert_eq!(
        prof.theta[k], upper,
        "the profiled fit holds it on the bound"
    );
    assert!((plain.theta[k] - upper).abs() < 1e-6, "{}", plain.theta[k]);
    for v in [prof.theta[c], plain.theta[c]] {
        assert!(v.is_finite() && v < upper - 1e-3, "{control}: {v}");
    }
    assert!(ferx_core::estimate_near_boundary(&prof));
}

/// Realised (Linux aarch64): the unprofiled restart improves by −2.3e-9 (F1) and −7.0e-7
/// (F2), structural θ 4.8e-9 and 3.3e-6 relative; inside D1/D2's bands.
#[test]
#[cfg_attr(
    any(not(feature = "slow-tests"), feature = "nn"),
    ignore = "slow: opt in with --features slow-tests; not run under nn, see the module doc"
)]
fn d2b_a_bounded_profiled_fit_is_the_bounded_unprofiled_fit() {
    bounded_degenerate_oracle(
        "tests/data/profile_mbma/f1_profile.ferx",
        "tests/data/mbma_placebo/mbma_placebo.csv",
        "(0.0, -10.0, 3.0)",
        4,
        3.0,
        "PLACEBO[STUDY=3,TIME=0]",
        "PLACEBO[STUDY=4,TIME=0]",
    );
    bounded_degenerate_oracle(
        "nonmem_anchor/profile_mbma_fit.ferx",
        "nonmem_anchor/profile_mbma.csv",
        "(0.0, -10.0, 2.5)",
        3,
        2.5,
        "PLACEBO[STUDY=7,TIME=4]",
        "PLACEBO[STUDY=7,TIME=1]",
    );
}

/// A3 (#2112), the fit: the bounded profiled F2 fit from the model file's inits lands on
/// NONMEM's optimum with THETA24 FIXed at the bound, its structural θ on NONMEM's.
/// Realised (Linux aarch64): ΔOFV 2.5e-8, structural θ 9.5e-6 relative (macOS arm64:
/// 1.4e-8, 2.0e-6). A2's OFV band and D's θ band.
#[test]
#[cfg_attr(
    any(not(feature = "slow-tests"), feature = "nn"),
    ignore = "slow: opt in with --features slow-tests; not run under nn, see the module doc"
)]
fn a3_a_bounded_profiled_fit_reaches_nonmems_optimum_with_the_level_fixed() {
    let text = std::fs::read_to_string(root().join("nonmem_anchor/results/profile_mbma_bound.ext"))
        .unwrap();
    let last = text
        .lines()
        .find(|l| l.trim_start().starts_with("-1000000000"))
        .unwrap();
    let row: Vec<f64> = last
        .split_whitespace()
        .skip(1)
        .map(|v| v.parse().unwrap())
        .collect();
    let obj = *row.last().unwrap();
    let p = prep(
        "nonmem_anchor/profile_mbma_fit.ferx",
        "nonmem_anchor/profile_mbma.csv",
        Some("](0.0, -10.0, 2.5) profile"),
    );
    let (f, t) = fit(&p, &p.init_params, false);
    let worst_theta =
        (0..3)
            .map(|k| ((f.theta[k] - row[k]) / row[k]).abs())
            .fold(0.0_f64, |a, v| {
                assert!(v.is_finite());
                a.max(v)
            });
    eprintln!(
        "A3 fit: ferx {:.9} NONMEM {obj:.9} (Δ {:.3e}), structural θ {worst_theta:.3e}, {t:.1}s",
        f.ofv,
        f.ofv - obj
    );
    assert!(f.converged && f.ofv.is_finite(), "{:?}", f.warnings);
    assert_eq!(f.theta[23], 2.5);
    assert!((f.ofv - obj).abs() < A2_FIT_OFV, "{} vs {obj}", f.ofv);
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

/// `model` with `method = laplace` (one node), written to a temp file; its absolute path,
/// which `prep`'s `root().join` keeps as is.
fn laplace_copy(model: &str) -> String {
    let text = std::fs::read_to_string(root().join(model)).unwrap();
    assert!(text.contains("  method     = focei\n"), "fixture moved");
    let path = unique_temp("ferx_profiled_fits_laplace");
    std::fs::write(
        &path,
        text.replace("  method     = focei\n", "  method     = laplace\n"),
    )
    .unwrap();
    path.to_str().unwrap().to_string()
}

/// D1-L (#2107): D1 under `laplace`. F1 is linear in (η, κ), so the profile solve takes
/// one step there under Laplace too.
#[test]
#[cfg_attr(
    any(not(feature = "slow-tests"), feature = "nn"),
    ignore = "slow: opt in with --features slow-tests; not run under nn, see the module doc"
)]
fn d1_l_a_profiled_laplace_fit_is_the_unprofiled_fit_linear_in_eta() {
    degenerate_oracle(
        &laplace_copy("tests/data/profile_mbma/f1_profile.ferx"),
        "tests/data/mbma_placebo/mbma_placebo.csv",
        "](0.0, -50.0, 50.0)",
        4,
    );
}

/// D2-L (#2107): D2 under `laplace`, on F2, where the Laplace objective is not FOCEI's.
/// Measured on the probe (macOS arm64): the unprofiled restart improved by 4.1e-10.
#[test]
#[cfg_attr(
    any(not(feature = "slow-tests"), feature = "nn"),
    ignore = "slow: opt in with --features slow-tests; not run under nn, see the module doc"
)]
fn d2_l_a_profiled_laplace_fit_is_the_unprofiled_fit_nonlinear_in_eta() {
    degenerate_oracle(
        &laplace_copy("nonmem_anchor/profile_mbma_fit.ferx"),
        "nonmem_anchor/profile_mbma.csv",
        "](0.0, -50.0, 50.0)",
        3,
    );
}

/// The profiled F2 fit against NONMEM's own optimum, every level a `THETA` there.
/// On #2280's inner BFGS without #2285's `xtol_rel` ceiling and verification leg, this
/// start stopped `converged: true` (`XtolReached`) 1.584e-3 above it (Linux aarch64).
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

/// A2-L fit (#2107): profiled `laplace` fits of F2 from the model file's start and from
/// A2's two #2285 starts land on NONMEM's LAPLACE optimum (`profile_mbma_laplace.ext`),
/// converged. The band is A2's: a stopping rule sets it, and it still fails FOCEI's
/// optimum, 0.0626 away.
#[test]
#[cfg_attr(
    any(not(feature = "slow-tests"), feature = "nn"),
    ignore = "slow: opt in with --features slow-tests; not run under nn, see the module doc"
)]
fn a2_l_a_profiled_laplace_fit_reaches_nonmems_optimum() {
    let obj = ext_final_ofv("profile_mbma_laplace.ext");
    assert!(
        (obj - a2_nonmem_ofv()).abs() > 1e3 * A2_FIT_OFV,
        "the straddle"
    );
    let p = prep(
        &laplace_copy("nonmem_anchor/profile_mbma_fit.ferx"),
        "nonmem_anchor/profile_mbma.csv",
        None,
    );
    assert_eq!(p.init_params.theta[..3], [45.0, 8.0, 2.0], "fixture moved");
    for start in [[45.0, 8.0, 2.0], [45.0, 6.0, 2.0], [40.0, 6.0, 2.0]] {
        let mut params = p.init_params.clone();
        params.theta[..3].copy_from_slice(&start);
        let (f, t) = fit(&p, &params, false);
        eprintln!(
            "A2-L from {start:?}: ferx {:.9} NONMEM {obj:.9} (Δ {:.3e}), converged {}, {} \
             evals, {t:.1}s",
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

/// NONMEM's minimum on the F2 anchor, from the `.ext` final row.
fn a2_nonmem_ofv() -> f64 {
    ext_final_ofv("profile_mbma.ext")
}

/// The OBJ of a NONMEM run's `.ext` final row, `name` under `nonmem_anchor/results`.
fn ext_final_ofv(name: &str) -> f64 {
    let text = std::fs::read_to_string(root().join("nonmem_anchor/results").join(name)).unwrap();
    text.lines()
        .find(|l| l.trim_start().starts_with("-1000000000"))
        .and_then(|l| l.split_whitespace().last())
        .unwrap()
        .parse()
        .unwrap()
}

/// #2285: A2 from starts near the file's own, where a fit that says `converged: true`
/// must be on NONMEM's optimum. A profiled L-BFGS stop is only a test on one iteration,
/// so the fit is converged when a fresh leg from its best point gains no more than
/// `PROFILED_VERIFY_GAIN`, every leg at `xtol_rel ≤ 1e-6`. The two starts are the ones
/// that kill each half of that rule on Linux aarch64 (`tools/linux-test.sh`, a 34-start
/// grid around (45, 8, 2), on `main` at `5bd862b8` and on #2280's inner BFGS merged in):
///
/// - (45, 6, 2): with no verification leg it stops `converged: true` 2.16e-5 above the
///   optimum on `main` and 2.27e-5 on #2280.
/// - (40, 6, 2): with no `xtol_rel` ceiling, on #2280, every leg ends on a short step and
///   gains under the threshold while 1.79e-5 short. On `main` it lands either way, and
///   no start of the grid kills the ceiling there; `a_profiled_gradient_fit_caps_its_xtol`
///   (Tier 1) is its guard on `main`'s engine.
///
/// With both halves, every start of that grid landed within 7.2e-7 of the optimum,
/// converged, on both engines.
///
/// - (50, 6, 2) (#2340): its verification leg stops `XtolReached` after a 3.67e-7
///   displacement and gains 7.4e-8, under the threshold, while 1.24e-5 short. Before
///   `PROFILED_LEG_CREEP` that settled the fit `converged: true` outside the band. Now the
///   loop runs on and settles at +8.7e-8 (Linux aarch64, `main` at `8f25c217`; of
///   #2340's 98 starts, the only one the rule brings into the band. The other start it
///   changes, (55, 7, 1.2), only takes 15 more evaluations).
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
    for start in [[45.0, 6.0, 2.0], [40.0, 6.0, 2.0], [50.0, 6.0, 2.0]] {
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

/// #2336: a profiled `optimizer = bobyqa` fit of A2 that says `converged: true` is on
/// NONMEM's optimum. Before, its first `FtolReached` was reported converged: from the
/// model file's start 0.0322 above it (macOS arm64; 3.46 on `main` at `5bd862b8`, the
/// issue), and up to 32 above over an 18-start grid (40/45/50 × 6/8/12 × 2/3.2). The
/// polish legs run at the `POLISH_*` tolerances and settle at `PROFILED_VERIFY_GAIN`.
/// (45, 12, 2) is the start that sets that threshold: its first leg gains 5.5e-4, which
/// the unprofiled #1782 `POLISH_MIN_GAIN` (1e-3) would settle on, converged 9.1e-5
/// above; at 1e-6 three more legs follow and it lands 1.6e-6 above (macOS arm64). A
/// fit that needs the leg cap there may instead report not converged, which the issue
/// accepts, so that start asserts only that `converged` implies the optimum.
#[test]
#[cfg_attr(
    any(not(feature = "slow-tests"), feature = "nn"),
    ignore = "slow: opt in with --features slow-tests; not run under nn, see the module doc"
)]
fn a2_a_profiled_bobyqa_fit_is_converged_only_on_nonmems_optimum() {
    let obj = a2_nonmem_ofv();
    let mut p = prep(
        "nonmem_anchor/profile_mbma_fit.ferx",
        "nonmem_anchor/profile_mbma.csv",
        None,
    );
    p.parsed.fit_options.optimizer = Optimizer::Bobyqa;
    assert_eq!(p.init_params.theta[..3], [45.0, 8.0, 2.0], "fixture moved");
    for (start, must_converge) in [([45.0, 8.0, 2.0], true), ([45.0, 12.0, 2.0], false)] {
        let mut params = p.init_params.clone();
        params.theta[..3].copy_from_slice(&start);
        let (f, t) = fit(&p, &params, false);
        eprintln!(
            "A2 BOBYQA from {start:?}: ferx {:.9} (Δ {:.3e}), converged {}, {} evals, {t:.1}s",
            f.ofv,
            f.ofv - obj,
            f.converged,
            f.n_iterations
        );
        assert!(f.ofv.is_finite(), "{start:?}: {:?}", f.warnings);
        assert!(f.converged || !must_converge, "{start:?}: {:?}", f.warnings);
        assert!(
            !f.converged || (f.ofv - obj).abs() < A2_FIT_OFV,
            "{start:?}: converged at {} vs NONMEM {obj}",
            f.ofv
        );
    }
}

/// #2339: from TVE0 ≥ 52 the first L-BFGS step puts TVE0 on ferx's internal `1e-10`
/// log-packing floor, κ grows to 2071 to absorb the baseline, and the fit used to stop
/// `converged: true` at OFV 273.848987, 171.9 above NONMEM's optimum, profiled and
/// unprofiled alike (Linux aarch64, `main` at `35a417ff`; 51 still landed). NONMEM reaches
/// its optimum from TVE0 = 55 (`nonmem_anchor/profile_mbma_tve0_55.ctl`, pinned here too). The
/// floor escape probes TVE0 just above the floor, finds the objective lower, and restarts
/// with TVE0 at its start. Each fit must name that restart in its warnings, so a fixture
/// that stopped reaching the floor fails here rather than passing without testing it.
///
/// Realised (Linux aarch64): profiled from (55, 8, 2) Δ 2.7e-8 and from (52, 8, 2) 4.1e-8,
/// inside [`A2_FIT_OFV`]; all 16 TVE0 = 55 starts of #2339's grid land within 4.2e-8. The
/// unprofiled restart is held to [`A2_UNPROFILED_RESTART_OFV`] instead: see there.
#[test]
#[cfg_attr(
    any(not(feature = "slow-tests"), feature = "nn"),
    ignore = "slow: opt in with --features slow-tests; not run under nn, see the module doc"
)]
fn a2_a_start_that_floors_tve0_reaches_nonmems_optimum() {
    let obj = a2_nonmem_ofv();
    let nm55 = ext_ofv("nonmem_anchor/results/profile_mbma_tve0_55.ext");
    assert!(
        (nm55 - obj).abs() < A2_FIT_OFV,
        "NONMEM from TVE0 = 55 ends at {nm55}, not its own optimum {obj}"
    );
    let prof = prep(
        "nonmem_anchor/profile_mbma_fit.ferx",
        "nonmem_anchor/profile_mbma.csv",
        None,
    );
    let plain = prep(
        "nonmem_anchor/profile_mbma_fit.ferx",
        "nonmem_anchor/profile_mbma.csv",
        Some("](0.0, -50.0, 50.0)"),
    );
    for (label, p, start, band) in [
        ("profiled", &prof, [55.0, 8.0, 2.0], A2_FIT_OFV),
        ("profiled", &prof, [52.0, 8.0, 2.0], A2_FIT_OFV),
        (
            "unprofiled",
            &plain,
            [55.0, 8.0, 2.0],
            A2_UNPROFILED_RESTART_OFV,
        ),
    ] {
        let mut params = p.init_params.clone();
        params.theta[..3].copy_from_slice(&start);
        let (f, t) = fit(p, &params, false);
        eprintln!(
            "A2 {label} from {start:?}: ferx {:.9} (Δ {:.3e}), converged {}, TVE0 {:.6}, {} evals, {t:.1}s",
            f.ofv,
            f.ofv - obj,
            f.converged,
            f.theta[0],
            f.n_iterations
        );
        assert!(
            f.warnings.iter().any(|w| w.contains("log-packing floor")
                && w.contains("resumed from the best point seen")),
            "{label} {start:?}: the floor escape did not run: {:?}",
            f.warnings
        );
        assert!(
            f.converged && f.ofv.is_finite(),
            "{label} {start:?}: {:?}",
            f.warnings
        );
        assert!(
            (f.ofv - obj).abs() < band,
            "{label} {start:?}: converged at {} vs NONMEM {obj}",
            f.ofv
        );
    }
}

/// The unprofiled restart's band. Before #2462, over #2339's 16 TVE0 = 55 starts (Linux
/// aarch64) the restart reached 8.8e-7 to 5.3e-4 above NONMEM, 15 of them near 4.1e-4,
/// and this band was 1e-3. Its L-BFGS stopped there on the default `inner_tol`'s noise
/// and the plateau rule accepted the stop. Since #2462 every such stop is verified by a
/// leg at a tight inner tolerance, and the 80-start grid around these starts lands within
/// 5.1e-7 (Linux aarch64), 20× inside this band, which is the profiled fit's own
/// [`A2_FIT_OFV`].
const A2_UNPROFILED_RESTART_OFV: f64 = A2_FIT_OFV;

/// The `#OBJV` of an `.ext` file's final (`-1000000000`) row.
fn ext_ofv(rel: &str) -> f64 {
    let text = std::fs::read_to_string(root().join(rel)).unwrap();
    text.lines()
        .find(|l| l.trim_start().starts_with("-1000000000"))
        .and_then(|l| l.split_whitespace().last())
        .unwrap()
        .parse()
        .unwrap()
}

/// Realised on Linux aarch64 from the model file's inits, #2285's verification leg
/// included: 3.4e-8 (38 evaluations) on `main` at `5bd862b8`, 3.9e-8 (45) with #2280
/// merged. A
/// stopping rule, not an engine difference, sets this; 1e-5 leaves room for another
/// platform's path to the optimum while still failing the 0.0185 BOBYQA stopped at.
const A2_FIT_OFV: f64 = 1e-5;

/// S1: 100 studies × 8 visits. Profiled: a 6-coordinate search and the `r` covariance
/// with a standard error for every structural parameter and every level. The profiled
/// OFV must match the unprofiled fit's (706 coordinates, from the model file's inits, no
/// covariance step), run here: since S1 has the study baseline `ETA_E0` (#2326) that fit
/// converges in 274 evaluations, 0.7 s alone and 8.8 s beside the file's other fits.
/// Measured on Linux aarch64: profiled −3792.330527 (31–39 s), unprofiled −3792.330530,
/// 3.0e-6 apart. [`S1_OFV`] leaves 30× headroom on that gap and still fails a profiled fit
/// stopped short of the optimum.
#[test]
#[cfg_attr(
    any(not(feature = "slow-tests"), feature = "nn"),
    ignore = "slow: opt in with --features slow-tests; not run under nn, see the module doc"
)]
fn s1_a_hundred_study_placebo_model_profiles_to_a_small_search() {
    let s1 = (
        "tests/data/profile_mbma/s1_profile.ferx",
        "tests/data/profile_mbma/s100t8.csv",
    );
    let p = prep(s1.0, s1.1, None);
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
        "S1: profiled OFV {:.6} ({t_prof:.1}s, covariance {:.2}s, {} evals, n_parameters {})",
        prof.ofv, prof.covariance_wall_time_secs, prof.n_iterations, prof.n_parameters
    );
    assert_eq!(
        prof.n_parameters, 706,
        "the unprofiled count: 3 θ, 700 levels (sum_to_zero_within), 2 ω, σ"
    );
    let plain_prep = prep(s1.0, s1.1, Some("](0.0, -50.0, 50.0)"));
    let (plain, t_plain) = fit(&plain_prep, &plain_prep.init_params, false);
    eprintln!(
        "S1: unprofiled OFV {:.6} ({t_plain:.1}s, {} evals, converged {})",
        plain.ofv, plain.n_iterations, plain.converged
    );
    assert!(
        plain.converged && plain.ofv.is_finite(),
        "{:?}",
        plain.warnings
    );
    assert!(
        (prof.ofv - plain.ofv).abs() < S1_OFV,
        "profiled {} vs unprofiled {}",
        prof.ofv,
        plain.ofv
    );
}

/// The profiled and unprofiled S1 fits reach one optimum: 3.0e-6 apart on Linux aarch64
/// (module doc of the test). 1e-4 is 30× that.
const S1_OFV: f64 = 1e-4;

/// S1's old shape (#2326: no `ETA_E0`, so nothing is absorbed) under an explicit global
/// `sum_to_zero` (#2135): one group of 799 θ tied by the dependent level, which study
/// 100's subjects read. Before the bordered solve each of them was differenced over the
/// whole block — one evaluation at the inits took 201 s and the fit did not finish in
/// 5 400 s (Linux x86_64 CI, #2326), 85 s for that one evaluation on macOS arm64.
/// Differenced over its own 8 levels it is 1.5 s, and the whole fit with its covariance
/// step 108 s (macOS arm64, 8 threads; both on the data before #2326 added a study
/// baseline). With `ETA_E0` global `sum_to_zero` is refused, since that η is each study's
/// mean (`profiled_levels.rs` P2). The OFV must not be above the same model's
/// `sum_to_zero_within` fit, run here: within nests in global, which leaves each study's
/// mean free.
#[test]
#[cfg_attr(
    any(not(feature = "slow-tests"), feature = "nn"),
    ignore = "slow: opt in with --features slow-tests; not run under nn, see the module doc"
)]
fn s1_a_global_sum_to_zero_profiles_over_each_subjects_own_levels() {
    let data = "tests/data/profile_mbma/s100t8.csv";
    let model = "tests/data/profile_mbma/s1_profile.ferx";
    let text = std::fs::read_to_string(root().join(model)).unwrap();
    let no_e0 = text
        .replace("  omega ETA_E0 ~ 0.1\n", "")
        .replace("E0 = TVE0 + ETA_E0", "E0 = TVE0");
    let block = "PLACEBO[STUDY, TIME]";
    assert!(
        no_e0.contains(block) && !no_e0.contains("omega ETA_E0") && !no_e0.contains("+ ETA_E0"),
        "fixture moved"
    );
    let with_contrast = |contrast: &str| {
        let path = unique_temp("ferx_profiled_s1_global");
        std::fs::write(
            &path,
            no_e0.replace(
                block,
                &format!("PLACEBO[STUDY, TIME, contrast = {contrast}]"),
            ),
        )
        .unwrap();
        ferx_core::prepare_run(
            path.to_str().unwrap(),
            Some(root().join(data).to_str().unwrap()),
        )
        .unwrap_or_else(|e| panic!("{e}"))
    };
    let p = with_contrast("sum_to_zero");
    let (prof, t_prof) = fit(&p, &p.init_params, true);
    eprintln!(
        "S1 global sum_to_zero: profiled OFV {:.6} ({t_prof:.1}s, covariance {:.2}s, {} evals, \
         n_parameters {})",
        prof.ofv, prof.covariance_wall_time_secs, prof.n_iterations, prof.n_parameters
    );
    assert!(prof.converged, "{:?}", prof.warnings);
    assert_eq!(prof.covariance_method, Some(CovarianceMethod::Hessian));
    assert_eq!(prof.n_parameters, 804, "3 θ, 799 levels, ω, σ");
    let se = prof.se_theta.as_ref().expect("profiled SE");
    assert!(se.iter().all(|s| s.is_finite() && *s > 0.0));
    let cov = prof.covariance_matrix.as_ref().unwrap();
    let levels = &ferx_core::theta_level_values(&p.parsed.model, &prof.theta, Some(cov), None)
        .unwrap()["PLACEBO"];
    assert_eq!(levels.len(), 800);
    assert!(levels
        .iter()
        .all(|l| l.se.is_some_and(|s| s.is_finite() && s > 0.0)));
    let w = with_contrast("sum_to_zero_within");
    let (within, t_within) = fit(&w, &w.init_params, false);
    eprintln!(
        "S1 sum_to_zero_within: profiled OFV {:.6} ({t_within:.1}s, n_parameters {})",
        within.ofv, within.n_parameters
    );
    assert!(
        within.converged && within.ofv.is_finite(),
        "{:?}",
        within.warnings
    );
    assert_eq!(within.n_parameters, 705, "3 θ, 700 levels, ω, σ");
    assert!(
        prof.ofv <= within.ofv + 1e-3,
        "global {} vs within {}",
        prof.ofv,
        within.ofv
    );
}

/// `run_covariance` on a **converged** profiled fit repeats its covariance step bit for bit
/// (#2173, #2213): F1 on `mbma_placebo` fitted to convergence, then replayed in memory and
/// after a `.fitrx` reload. The Tier-2 parity test (`profiled_levels.rs`) is an evaluation,
/// so the optimizer's incumbent warm start and its `x_best` never reach a replay there.
///
/// It is also the fixture where the replay's packed point matters: the converged `x_best`
/// is not a repack of its own estimates, so `pack(unpack(x̂)) ≠ x̂` — on one of the 25
/// coordinates, the Cholesky diagonal of `ETA_E0` by one ulp — asserted, so the fixture
/// cannot quietly become one where it is exact. A replay that repacks the estimates
/// instead of running at `x̂` differs on 576 of the 625 entries (macOS arm64); with #2211,
/// 0 of 625, in memory and reloaded. Measured from `TVE0 = 47` (#2109): OFV 97.070036143
/// after 29 evaluations, one lossy coordinate and 0 of 625 on Linux aarch64; the lossy
/// coordinate the same on macOS arm64.
/// (Short Tier-2 runs, `outer_maxiter` 1–4 from two starts, left `x̂` lossy once and never
/// on a coordinate the stencil could see, so this is the test that pins it.)
#[test]
#[cfg_attr(
    any(not(feature = "slow-tests"), feature = "nn"),
    ignore = "slow: opt in with --features slow-tests; not run under nn, see the module doc"
)]
fn r1_run_covariance_replays_a_converged_profiled_fit() {
    use ferx_core::estimation::parameterization::{pack_params, unpack_params};
    use ferx_core::io::fitrx::{load_fit, save_fit, SaveFitOptions};
    let model = "tests/data/profile_mbma/f1_profile.ferx";
    let p = prep(model, "tests/data/mbma_placebo/mbma_placebo.csv", None);
    // Which coordinates of the converged `x̂` survive a repack is a property of its last
    // bits, so no one start is relied on: the fit runs from each `TVE0` below in turn until
    // one leaves `x̂` lossy (#2401 review row 4). The file's own 45 stopped doing so once the
    // profile solve's level gradient moved to the adjoint (#2109). Measured: 47 leaves the
    // `ETA_E0` coordinate (22) lossy on Linux aarch64 and macOS arm64, 44.5 does on both,
    // 45, 46 and 46.5 on neither; so a later engine change has to move every start in the
    // list at once to make this fail.
    assert_eq!(p.init_params.theta_names[0], "TVE0");
    let lossy_of = |f: &FitResult| -> usize {
        let x_hat = f
            .packed_estimate
            .as_ref()
            .expect("a profiled fit carries its packed point");
        let repack = pack_params(&unpack_params(x_hat, &p.init_params));
        x_hat
            .iter()
            .zip(&repack)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count()
    };
    let (f, lossy) = [47.0, 44.5, 45.5, 43.0, 44.0, 45.0]
        .into_iter()
        .find_map(|tve0| {
            let mut start = p.init_params.clone();
            start.theta[0] = tve0;
            let (f, t) = fit(&p, &start, true);
            let lossy = lossy_of(&f);
            eprintln!(
                "R1: from TVE0 {tve0}: OFV {:.9} ({t:.1}s, {} evals), {lossy} of {} \
                 coordinates lossy",
                f.ofv,
                f.n_iterations,
                f.packed_estimate.as_ref().map_or(0, Vec::len)
            );
            (lossy > 0).then_some((f, lossy))
        })
        .expect("no start left x̂ lossy: the packed point is untested");
    assert!(f.converged, "{:?}", f.warnings);
    assert!(lossy > 0);
    let want = f.covariance_matrix.as_ref().expect("inline covariance");
    let opts = FitOptions {
        verbose: false,
        ..p.parsed.fit_options.clone()
    };
    let replay = |fit: &FitResult, how: &str| {
        let r = ferx_core::run_covariance(fit, Some(&p.parsed.model), Some(&p.population), &opts)
            .unwrap_or_else(|e| panic!("{how}: {e}"));
        let got = r.covariance_matrix.as_ref().expect("replayed covariance");
        assert_eq!(got.shape(), want.shape(), "{how}");
        let differing = got
            .iter()
            .zip(want.iter())
            .filter(|(g, w)| g.to_bits() != w.to_bits())
            .count();
        eprintln!(
            "R1 {how}: {differing} of {} covariance entries differ",
            want.len()
        );
        assert_eq!(differing, 0, "{how}: {differing} covariance entries differ");
        assert_eq!(r.se_theta, f.se_theta, "{how}");
    };
    replay(&f, "in memory");
    let path = std::env::temp_dir().join(format!("ferx_profiled_r1_{}.fitrx", std::process::id()));
    let text = std::fs::read_to_string(root().join(model)).unwrap();
    save_fit(&f, &p.population, &text, &path, SaveFitOptions::default()).expect("saves");
    let loaded = load_fit(&path).expect("loads");
    let _ = std::fs::remove_file(&path);
    replay(&loaded.fit, "after .fitrx reload");
}
