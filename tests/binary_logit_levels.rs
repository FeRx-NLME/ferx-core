//! A θ level block read by a `[binary_model]` predictor (#2562), through the public `fit()`:
//! the NONMEM anchor A-B1. Every fit is an evaluation (`outer_maxiter = 0`), so the file is
//! Tier 2.
//!
//! **A-B1.** `nonmem_anchor/profile_logit_b1_laplace.ctl` codes B1 (2400 Bernoulli rows,
//! `logit p = TVL + PLACEBO[study, visit] + η + EMAX·TRT`, ID = study) with every level a
//! `THETA`, each study's last visit dependent as ferx lays it out, under
//! `METHOD=COND LAPLACE LIKE`. ferx is handed NONMEM's whole optimum — the 32 θ and Ω — and
//! evaluates the objective once. Each record is scored at its own level (the visit) and its
//! own arm (`TRT` changes within each study), so a record read at its subject's baseline
//! moves the objective by hundreds.
//!
//! ferx's FOCE and FOCEI on a binary row are the Laplace form (the exact discrete
//! η-Hessian, #2437; the `reset_clock_cat` anchor), so all three methods are held to the
//! one NONMEM objective.
//!
//! The same anchor holds the block `profile`d (#2111): ferx is handed NONMEM's structural
//! point only and solves the levels itself, then the profiled covariance is checked against
//! the unprofiled one (C1-bin), and a `slow-tests` fit from the inits lands on NONMEM's
//! objective (D1-bin).

#![cfg(feature = "survival")]

use ferx_core::types::{EstimationMethod, FitOptions, OmegaMatrix};
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The final row of `nonmem_anchor/results/profile_logit_b1_laplace.ext`: θ in order, Ω,
/// and the objective.
fn nonmem_optimum() -> (Vec<f64>, f64, f64) {
    let text =
        std::fs::read_to_string(root().join("nonmem_anchor/results/profile_logit_b1_laplace.ext"))
            .unwrap();
    let mut names: Vec<String> = Vec::new();
    for line in text.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        match fields.first() {
            Some(&"ITERATION") => names = fields.iter().map(|s| s.to_string()).collect(),
            Some(&"-1000000000") => {
                let get = |name: &str| -> f64 {
                    let i = names.iter().position(|n| n == name).unwrap();
                    fields[i].parse().unwrap()
                };
                let theta = (1..=32).map(|k| get(&format!("THETA{k}"))).collect();
                return (theta, get("OMEGA(1,1)"), get("OBJ"));
            }
            _ => {}
        }
    }
    panic!("profile_logit_b1_laplace.ext: no final row");
}

/// |ΔOFV| bound. Measured 1.42e-9 under all three methods, macOS arm64 and Linux aarch64
/// alike — the gap of an independent Python Laplace at the same point (1.46e-9), so it is
/// NONMEM's printed precision, not ferx's. The bound sits 35× above it. A record read at
/// its subject's baseline level or arm moves the objective by far more.
const A_B1_OFV: f64 = 5e-8;

fn evaluate(method: EstimationMethod, interaction: bool) -> (f64, f64, Vec<String>) {
    let p = ferx_core::prepare_run(
        root()
            .join("nonmem_anchor/profile_logit_b1.ferx")
            .to_str()
            .unwrap(),
        Some(
            root()
                .join("nonmem_anchor/profile_logit_b1.csv")
                .to_str()
                .unwrap(),
        ),
    )
    .unwrap_or_else(|e| panic!("B1 loads: {e}"));
    assert_eq!(
        p.parsed.model.theta_names.len(),
        32,
        "TVL, EMAX and 30 free levels, NONMEM's THETA order"
    );
    assert_eq!(p.parsed.model.theta_names[2], "PLACEBO[STUDY=1,TIME=0]");
    assert_eq!(p.parsed.model.theta_names[31], "PLACEBO[STUDY=10,TIME=8]");
    let (theta, omega, obj) = nonmem_optimum();
    let mut params = p.init_params.clone();
    params.theta = theta;
    params.omega = OmegaMatrix::from_diagonal(&[omega], vec!["ETA_S".into()]);
    let options = FitOptions {
        verbose: false,
        outer_maxiter: 0,
        run_covariance_step: false,
        inner_tol: 1e-8,
        method,
        interaction,
        ..p.parsed.fit_options.clone()
    };
    let f = ferx_core::fit(&p.parsed.model, &p.population, &params, &options)
        .unwrap_or_else(|e| panic!("{method:?}: {e}"));
    (f.ofv, obj, f.warnings)
}

/// A-B1 under `method`: the objective at NONMEM's optimum, on the exact discrete
/// η-Hessian and gradient (a finite-difference warning would mean the predictor's
/// per-record read left the dual path).
fn assert_reproduces(method: EstimationMethod, interaction: bool) {
    let (ofv, obj, warnings) = evaluate(method, interaction);
    let d = ofv - obj;
    eprintln!("A-B1 {method:?}: ferx {ofv:.10} NONMEM {obj:.10} ΔOFV {d:.3e}");
    assert!(ofv.is_finite(), "{method:?}: OFV {ofv}");
    assert!(d.abs() <= A_B1_OFV, "{method:?}: |ΔOFV| {:.3e}", d.abs());
    assert!(
        !warnings
            .iter()
            .any(|w| w.contains("finite difference") || w.contains("finite-difference")),
        "{method:?}: {warnings:?}"
    );
}

#[test]
fn binary_levels_reproduce_nonmem_laplace() {
    assert_reproduces(EstimationMethod::Laplace, true);
}

#[test]
fn binary_levels_reproduce_nonmem_under_focei() {
    assert_reproduces(EstimationMethod::FoceI, true);
}

#[test]
fn binary_levels_reproduce_nonmem_under_foce() {
    assert_reproduces(EstimationMethod::Foce, false);
}

// ── #2111: the same block `profile`d ────────────────────────────────────────────

const UNPROFILED: &str = "](0.0, -10.0, 10.0)\n";
const PROFILED: &str = "](0.0) profile\n";

/// B1 with its block `profile`d, or as committed.
fn prep(profiled: bool) -> ferx_core::PreparedRun {
    let text = std::fs::read_to_string(root().join("nonmem_anchor/profile_logit_b1.ferx")).unwrap();
    assert!(text.contains(UNPROFILED), "the fixture moved");
    let text = if profiled {
        text.replace(UNPROFILED, PROFILED)
    } else {
        text
    };
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "ferx_binary_levels_{}_{n}.ferx",
        std::process::id()
    ));
    std::fs::write(&path, text).unwrap();
    ferx_core::prepare_run(
        path.to_str().unwrap(),
        Some(
            root()
                .join("nonmem_anchor/profile_logit_b1.csv")
                .to_str()
                .unwrap(),
        ),
    )
    .unwrap_or_else(|e| panic!("B1 loads: {e}"))
}

/// One evaluation (`outer_maxiter = 0`) of `p` at `params` under `method`.
fn evaluate_at(
    p: &ferx_core::PreparedRun,
    params: &ferx_core::types::ModelParameters,
    method: EstimationMethod,
    interaction: bool,
    covariance: bool,
) -> ferx_core::types::FitResult {
    let options = FitOptions {
        verbose: false,
        outer_maxiter: 0,
        run_covariance_step: covariance,
        inner_tol: 1e-8,
        method,
        interaction,
        ..p.parsed.fit_options.clone()
    };
    ferx_core::fit(&p.parsed.model, &p.population, params, &options)
        .unwrap_or_else(|e| panic!("{method:?}: {e}"))
}

/// The profiled B1 at NONMEM's structural point (TVL, EMAX, ω), its levels at their
/// initial 0 for the profile solve to find.
fn profiled_at_nonmem(
    method: EstimationMethod,
    interaction: bool,
    covariance: bool,
) -> (ferx_core::types::FitResult, Vec<f64>, f64) {
    let (theta, omega, obj) = nonmem_optimum();
    let p = prep(true);
    let mut params = p.init_params.clone();
    params.theta[0] = theta[0];
    params.theta[1] = theta[1];
    params.omega = OmegaMatrix::from_diagonal(&[omega], vec!["ETA_S".into()]);
    let f = evaluate_at(&p, &params, method, interaction, covariance);
    (f, theta, obj)
}

/// A-B1, profiled (#2111): ferx is handed NONMEM's structural optimum and **solves the 30
/// free levels itself**, through the predictor's implicit logit link, under each method.
/// The objective must be NONMEM's `METHOD=COND LAPLACE LIKE` OBJ (the unprofiled half above
/// is PR-0's, #2562) and the levels NONMEM's. Measured: ΔOFV −4.54e-9 under all three
/// methods (the unprofiled evaluation is −1.42e-9: ferx's solve ends a hair below NONMEM's
/// 4-significant-digit levels), levels within 7.0e-6 of NONMEM's (absolute; NONMEM's
/// `$COV` is singular here, so there is no NONMEM SE to scale by). macOS arm64. The bounds
/// are [`A_B1_OFV`] (11× headroom) and 1e-4. A record scored at its subject's baseline
/// level moves the objective by hundreds. macOS arm64 and Linux aarch64 agree in every
/// printed digit.
///
/// The warning check sees a declined adjoint gradient, not an off-mode one: every B1
/// subject has no Gaussian observation, so `min_obs_for_convergence_check` excuses an
/// off-mode subject from the FD-fallback warning. The Tier-1 tests
/// (`profile_binary_tests.rs`) assert `fd_gradients == 0` from the solve's own counts. No level gradient or Newton matrix may come from finite
/// differences, so no warning may name them.
fn assert_profiled_reproduces(method: EstimationMethod, interaction: bool) {
    let (f, theta, obj) = profiled_at_nonmem(method, interaction, false);
    assert_eq!(f.profiled_blocks, ["PLACEBO"], "{method:?}");
    for w in &f.warnings {
        assert!(
            !w.contains("finite difference") && !w.contains("finite-difference"),
            "{method:?}: {w}"
        );
    }
    let d = f.ofv - obj;
    let level = (2..32)
        .map(|k| {
            let g = (f.theta[k] - theta[k]).abs();
            assert!(g.is_finite(), "{method:?} θ{k}");
            g
        })
        .fold(0.0, f64::max);
    eprintln!(
        "A-B1 profiled {method:?}: ferx {:.10} NONMEM {obj:.10} ΔOFV {d:.3e}, levels {level:.3e}",
        f.ofv
    );
    assert!(d.abs() <= A_B1_OFV, "{method:?}: |ΔOFV| {:.3e}", d.abs());
    assert!(level < 1e-4, "{method:?}: levels {level:.3e}");
}

#[test]
fn profiled_binary_levels_reproduce_nonmem_laplace() {
    assert_profiled_reproduces(EstimationMethod::Laplace, true);
}

#[test]
fn profiled_binary_levels_reproduce_nonmem_under_focei() {
    assert_profiled_reproduces(EstimationMethod::FoceI, true);
}

#[test]
fn profiled_binary_levels_reproduce_nonmem_under_foce() {
    assert_profiled_reproduces(EstimationMethod::Foce, false);
}

/// C1-bin. The profiled `R` (the Schur complement over the structural parameters, the
/// level blocks from the adjoint) against the unprofiled step at the profiled point, with
/// every level a free θ, under `laplace`. The unprofiled side is the analytic R-matrix,
/// which reads no step, so a gap is the profile's. Measured: worst SE 1.6e-13 relative,
/// ΔOFV −2.5e-11, macOS arm64.
#[test]
fn the_profiled_binary_covariance_is_the_unprofiled_one_at_the_same_point() {
    let (fp, _, _) = profiled_at_nonmem(EstimationMethod::Laplace, true, true);
    let (_, omega, _) = nonmem_optimum();
    let pu = prep(false);
    let mut params = pu.init_params.clone();
    params.theta.clone_from(&fp.theta);
    params.omega = OmegaMatrix::from_diagonal(&[omega], vec!["ETA_S".into()]);
    let fu = evaluate_at(&pu, &params, EstimationMethod::Laplace, true, true);
    let sp = fp
        .se_theta
        .as_ref()
        .unwrap_or_else(|| panic!("profiled covariance: {:?}", fp.warnings));
    let su = fu
        .se_theta
        .as_ref()
        .unwrap_or_else(|| panic!("unprofiled covariance: {:?}", fu.warnings));
    assert_eq!(sp.len(), su.len(), "one SE per θ on both sides");
    let mut worst = 0.0_f64;
    for (a, b) in sp.iter().zip(su) {
        let rel = ((a - b) / b).abs();
        assert!(rel.is_finite(), "{a} vs {b}");
        worst = worst.max(rel);
    }
    eprintln!("C1-bin: ΔOFV {:.3e}, worst SE {worst:.3e}", fu.ofv - fp.ofv);
    assert!((fu.ofv - fp.ofv).abs() < 1e-7);
    assert!(worst < 1e-9, "{worst:.3e}");
}

/// D1-bin: the profiled fit from the model file's inits lands on NONMEM's OBJ, and the
/// unprofiled fit started at the profiled optimum does not improve on it: the profile is
/// the minimum over the levels at every structural point, so the two objectives share one
/// optimum. `laplace` only: FOCE and FOCEI are the same objective on binary rows (the
/// unprofiled tests above). Measured: profiled −9.2e-9 from NONMEM, `converged`; the
/// unprofiled fit from there −8.8e-11 (macOS arm64).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn a_profiled_binary_fit_lands_on_nonmems_optimum() {
    let (_, _, obj) = nonmem_optimum();
    let p = prep(true);
    let o = FitOptions {
        verbose: false,
        ..p.parsed.fit_options.clone()
    };
    let fp = ferx_core::fit(&p.parsed.model, &p.population, &p.init_params, &o)
        .unwrap_or_else(|e| panic!("{e}"));
    let pu = prep(false);
    let mut start = pu.init_params.clone();
    start.theta.clone_from(&fp.theta);
    start.omega = OmegaMatrix::from_diagonal(&[fp.omega[(0, 0)]], vec!["ETA_S".into()]);
    let ou = FitOptions {
        verbose: false,
        ..pu.parsed.fit_options.clone()
    };
    let fu = ferx_core::fit(&pu.parsed.model, &pu.population, &start, &ou)
        .unwrap_or_else(|e| panic!("{e}"));
    eprintln!(
        "D1-bin: profiled {:.9} ({}), vs NONMEM {:.3e}, unprofiled from there {:.3e}",
        fp.ofv,
        fp.converged,
        fp.ofv - obj,
        fu.ofv - fp.ofv
    );
    assert!(fp.converged, "{:?}", fp.warnings);
    assert!((fp.ofv - obj).abs() < 1e-6, "{}", fp.ofv);
    assert!(fu.ofv - fp.ofv > -1e-6, "{}", fu.ofv);
}
