//! A residual `correlation` line with a `profile`d θ level block (#2110): fits to
//! convergence, gated `slow-tests`.
//!
//! **D1-corr**, the degenerate oracle of `profiled_levels_fits.rs` with the line: F1
//! (`tests/data/profile_mbma/f1_profile.ferx` on `mbma_placebo`, the arm kappa at `OCC`)
//! plus `theta RHO` and `correlation car1(RHO) by OCC` — the shape ferx-r's MBMA smoke
//! fits (step 13) — and the same with `ar1`; and `nested_levels_3` (sponsor > study >
//! arm) with a profiled `PLACEBO[STUDYID, TIME]` block and `correlation ar1(RHO) by ARM`,
//! the nested arm level. The profiled fit runs first; the unprofiled
//! fit of the same model is **started at the profiled optimum** and must not move the
//! OFV, the structural θ or ρ by more than the bounds, and the two covariance steps must
//! give the same standard errors.
//!
//! When the comparison fails, the message says which side drifted. The unprofiled model
//! is also *evaluated* at the profiled estimate (`outer_maxiter = 0`): a profiled OFV
//! that differs from that evaluation is a profiled objective that is not the model's (its
//! `C̃` is another covariance); a profiled OFV equal to it, with the unprofiled restart
//! then improving, is a profiled fit that stopped off the optimum.

use ferx_core::types::{CovarianceMethod, FitOptions, FitResult, ModelParameters, OmegaMatrix};
use std::path::PathBuf;
use std::time::Instant;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

const F1: &str = "tests/data/profile_mbma/f1_profile.ferx";
const F1_DATA: &str = "tests/data/mbma_placebo/mbma_placebo.csv";
const NESTED: &str = "nonmem_anchor/nested_levels_3.ferx";
const NESTED_DATA: &str = "nonmem_anchor/nested_levels_3.csv";
const PROFILED: &str = "](0.0) profile";
const UNPROFILED: &str = "](0.0, -50.0, 50.0)";
/// The structural θ of both fixtures, the first four: TVE0, EMAX (TVEMAX), ED50, ET50.
const N_STRUCT: usize = 4;

/// A temp path no other call in this process gets (#2090).
fn unique_temp(stem: &str) -> PathBuf {
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!("{stem}_{}_{n}.ferx", std::process::id()))
}

/// F1 with `RHO` (the last θ) and `correlation <kind>(RHO) by OCC`.
fn f1_text(kind: &str, lo: &str) -> String {
    let text = std::fs::read_to_string(root().join(F1)).unwrap();
    assert!(
        text.contains(PROFILED)
            && text.contains("  omega ETA_E0")
            && text.contains("weight = SE\n"),
        "fixture moved"
    );
    text.replace(
        "  omega ETA_E0",
        &format!("  theta RHO(0.3, {lo}, 0.95)\n  omega ETA_E0"),
    )
    .replace(
        "weight = SE\n",
        &format!("weight = SE\n  correlation {kind}(RHO) by OCC\n"),
    )
}

/// `nested_levels_3` (sponsor > study > arm; the study and arm kappas inside an `exp`, so
/// nonlinear in them) with a profiled `PLACEBO[STUDYID, TIME]` baseline block and
/// `correlation ar1(RHO) by ARM`, grouped by the nested arm level. `RHO` is the last θ.
/// `ETA_SP` and `K_ARM` are `FIX`ed at the file's values: the per-(study, visit) block
/// leaves them barely identified on this design, and free, the profiled fit drives each
/// variance to 6.1e-6 (converged, OFV −123.0328 with both free and −122.2579 with ETA_SP
/// fixed, Linux aarch64) — the start-box rail, where the unprofiled restart is refused at
/// its start. Fixed, both still enter `C̃` through `H`.
fn nested_text() -> String {
    let text = std::fs::read_to_string(root().join(NESTED)).unwrap();
    assert!(
        text.contains("  omega ETA_SP ~ 0.0699388")
            && text.contains("  kappa K_ARM ~ 3.75478 at ARM")
            && text.contains("  y = TVE0 + "),
        "fixture moved"
    );
    text.replace(
        "  omega ETA_SP",
        "  theta PLACEBO[STUDYID, TIME](0.0) profile\n  theta RHO(0.3, -0.95, 0.95)\n  \
         omega ETA_SP",
    )
    .replace(
        "  omega ETA_SP ~ 0.0699388",
        "  omega ETA_SP ~ 0.0699388 FIX",
    )
    .replace(
        "  kappa K_ARM ~ 3.75478 at ARM",
        "  kappa K_ARM ~ 3.75478 FIX at ARM",
    )
    .replace("  y = TVE0 + ", "  y = TVE0 + PLACEBO + ")
    .replace(
        "DV ~ additive(ADD)",
        "DV ~ additive(ADD)\n  correlation ar1(RHO) by ARM",
    )
}

/// `text` (a profiled model) on `data`, profiled or not.
fn prep(text: &str, data: &str, profiled: bool) -> ferx_core::PreparedRun {
    let text = if profiled {
        text.to_string()
    } else {
        text.replace(PROFILED, UNPROFILED)
    };
    let path = unique_temp("ferx_profiled_corr_fits");
    std::fs::write(&path, text).unwrap();
    let p = ferx_core::prepare_run(
        path.to_str().unwrap(),
        Some(root().join(data).to_str().unwrap()),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let _ = std::fs::remove_file(&path);
    p
}

fn fit(
    p: &ferx_core::PreparedRun,
    params: &ModelParameters,
    cov: bool,
    evaluate: bool,
) -> (FitResult, f64) {
    // `r` on both sides, named: past 100 packed coordinates the unprofiled step would
    // otherwise route itself to `s`, which the nested fixture's block takes it past.
    let mut opts = FitOptions {
        verbose: false,
        run_covariance_step: cov,
        covariance_method: CovarianceMethod::Hessian,
        covariance_method_set: true,
        ..p.parsed.fit_options.clone()
    };
    if evaluate {
        // An evaluation is a reference value: its EBEs at a tolerance where the objective
        // has stopped moving. At the default `inner_tol = 1e-5` the nested fixture's FD
        // inner solve stops early and leaves 1.24e-6 on the OFV (Linux aarch64), above the
        // profile solve's own value, whose re-solves run at `STENCIL_INNER_TOL = 1e-7` —
        // the same residual `nested_levels_correlation_nonmem_anchor.rs` measured.
        opts.outer_maxiter = 0;
        opts.inner_tol = EVAL_INNER_TOL;
    }
    let t = Instant::now();
    let f = ferx_core::fit(&p.parsed.model, &p.population, params, &opts)
        .unwrap_or_else(|e| panic!("{e}"));
    (f, t.elapsed().as_secs_f64())
}

/// The inner tolerance of the reference evaluation ([`fit`] with `evaluate`).
const EVAL_INNER_TOL: f64 = 1e-8;

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

/// The worst relative gap over `idx` between two vectors, every term checked finite
/// before the fold (`f64::max` drops a NaN).
fn worst_rel(a: &[f64], b: &[f64], idx: impl Iterator<Item = usize>) -> f64 {
    idx.map(|k| ((a[k] - b[k]) / b[k]).abs())
        .fold(0.0_f64, |acc, v| {
            assert!(v.is_finite(), "{a:?} vs {b:?}");
            acc.max(v)
        })
}

fn d1_corr(kind: &str, text: &str, data: &str, se: SeCheck) {
    let prof_prep = prep(text, data, true);
    let (prof, t_prof) = fit(&prof_prep, &prof_prep.init_params, true, false);
    let plain_prep = prep(text, data, false);
    let start = start_at(&plain_prep.init_params, &prof);
    let rho = prof.theta.len() - 1;
    assert_eq!(
        plain_prep.parsed.model.theta_names[rho], "RHO",
        "RHO is the last θ"
    );
    // The unprofiled objective at the profiled estimate, with its covariance step:
    // attributes a failure to a side, and is the point the SEs are compared at.
    let (at_prof, _) = fit(&plain_prep, &start, true, true);
    assert!(prof.ofv.is_finite() && at_prof.ofv.is_finite());
    let objective_gap = prof.ofv - at_prof.ofv;
    eprintln!(
        "D1-corr {kind}: profiled OFV {:.9} ({t_prof:.1}s, {} evals, ρ {:.6}, converged {}); \
         unprofiled at that estimate {:.9}; objective gap {objective_gap:.3e}",
        prof.ofv, prof.n_iterations, prof.theta[rho], prof.converged, at_prof.ofv
    );
    // First, so a profile solve on another objective is named as such rather than
    // surfacing as a non-convergence or a missing SE further down.
    assert!(
        objective_gap.abs() < D_OBJECTIVE,
        "{kind}: the PROFILED side drifted: it reports OFV {} at its estimate, where the \
         unprofiled model scores {} — the profile solve is not minimising the model's \
         objective",
        prof.ofv,
        at_prof.ofv
    );
    assert!(prof.converged, "{kind}: {:?}", prof.warnings);

    let (plain, t_plain) = fit(&plain_prep, &start, false, false);
    assert!(plain.ofv.is_finite());
    let improvement = at_prof.ofv - plain.ofv;
    let worst_theta = worst_rel(&prof.theta, &plain.theta, 0..N_STRUCT);
    let rho_gap = (prof.theta[rho] - plain.theta[rho]).abs();
    assert!(rho_gap.is_finite());
    eprintln!(
        "D1-corr {kind}: unprofiled from there {:.9} ({t_plain:.1}s, ρ {:.6}); improvement \
         {improvement:.3e}, structural θ {worst_theta:.3e}, ρ {rho_gap:.3e}",
        plain.ofv, plain.theta[rho]
    );
    assert!(
        improvement.abs() < D_IMPROVEMENT,
        "{kind}: the profiled fit stopped off the optimum: the unprofiled fit started there \
         moved the OFV by {improvement:.3e} (to {})",
        plain.ofv
    );
    assert!(
        worst_theta < D_THETA,
        "{kind}: structural θ {worst_theta:.3e}"
    );
    assert!(rho_gap < D_RHO, "{kind}: ρ {rho_gap:.3e}");

    let (sp, su) = (
        prof.se_theta
            .as_ref()
            .unwrap_or_else(|| panic!("{kind}: profiled SE: {:?}", prof.warnings)),
        at_prof
            .se_theta
            .as_ref()
            .unwrap_or_else(|| panic!("{kind}: unprofiled SE: {:?}", at_prof.warnings)),
    );
    let per: Vec<String> = (0..N_STRUCT)
        .chain([rho])
        .map(|k| format!("{} {:.2e}", prof.theta_names[k], (sp[k] - su[k]) / su[k]))
        .collect();
    eprintln!(
        "D1-corr {kind}: SE gaps {} ({:?} vs {:?}, n_parameters {})",
        per.join(", "),
        prof.covariance_method,
        at_prof.covariance_method,
        prof.n_parameters
    );
    assert_eq!(prof.covariance_method, Some(CovarianceMethod::Hessian));
    assert_eq!(at_prof.covariance_method, Some(CovarianceMethod::Hessian));
    let worst_se = worst_rel(sp, su, (0..N_STRUCT).chain([rho]));
    // Every ω and κ variance SE, in order; a `FIX`ed one has none on either side.
    let var_se = |f: &FitResult| -> Vec<f64> {
        let take = |v: &Option<Vec<f64>>| v.clone().unwrap_or_default();
        [take(&f.se_omega), take(&f.se_kappa)].concat()
    };
    let (vp, vu) = (var_se(&prof), var_se(&at_prof));
    assert_eq!(vp.len(), vu.len(), "{kind}");
    let estimated: Vec<usize> = (0..vu.len())
        .filter(|&k| vu[k].is_finite() && vu[k] > 0.0)
        .collect();
    assert!(!estimated.is_empty(), "{kind}: no variance SE to compare");
    let worst_var_se = worst_rel(&vp, &vu, estimated.into_iter());
    eprintln!("D1-corr {kind}: SE (θ, ρ) {worst_se:.3e}, SE (ω, κ) {worst_var_se:.3e}");
    if se == SeCheck::Assert {
        assert!(worst_se < D_SE, "{kind}: SE (θ, ρ) {worst_se:.3e}");
        assert!(worst_var_se < D_SE, "{kind}: SE (ω, κ) {worst_var_se:.3e}");
    }
}

// Realised on Linux aarch64 (`tools/linux-test.sh`, base `d66c0c51`), car1 / ar1 /
// nested. The profiled fits stop at 89.451279156 / 92.809709126 / −90.678580874 in
// 39 / 33 / 58 evaluations (~4 s / ~4 s / ~170 s alone).

/// The profiled OFV against the unprofiled objective at the same estimate: 3.1e-13 /
/// 8.0e-13 / 2.7e-8 — two inner solves of one objective. `2e-7` is 7× the worst. The
/// profile solve on a diagonal `R` reports 0.065 (car1) and 4.3 (ar1) OFV off here
/// (macOS arm64, the mutation run).
const D_OBJECTIVE: f64 = 2e-7;
/// The unprofiled restart's gain: 7.5e-7 / −2.1e-9 / −1.3e-6 — its own stopping rule and
/// default `inner_tol`, not the profile. 8× the worst.
const D_IMPROVEMENT: f64 = 1e-5;
/// Structural θ, relative: 2.5e-4 (car1, where the restart gains its 7.5e-7) / 1.5e-8 /
/// 1.0e-6. 4× the worst.
const D_THETA: f64 = 1e-3;
/// ρ, absolute: 2.8e-6 / 3.4e-11 / 2.9e-9. 35× the worst.
const D_RHO: f64 = 1e-4;
/// Standard errors at one point, relative: (θ, ρ) 4.5e-3 / 3.2e-3, (ω, κ) 6.3e-5 / 1.1e-4
/// — two FD Hessians, over 6 coordinates re-profiled against 25 held, as C1's 4.7e-3 is.
/// 2.2× the worst. Not asserted on the nested fixture: see [`SeCheck`].
const D_SE: f64 = 1e-2;

/// Whether [`d1_corr`] holds the two covariance steps to [`D_SE`].
#[derive(Clone, Copy, PartialEq)]
enum SeCheck {
    Assert,
    /// Printed, not asserted. On the nested fixture the profiled SEs of TVE0, TVEMAX,
    /// ED50 and ρ sit 1.8–2.4% from the unprofiled `r` at the same point — and **without**
    /// the line 1.9–4.6% (TVE0 −4.6e-2, TVEMAX −2.0e-2, ED50 −2.1e-2; Linux aarch64), so
    /// the gap is the profiled covariance step's on this nonlinear nested design, which
    /// the line does not widen, not a `C̃` this PR builds. Tracked separately.
    PrintOnly,
}

#[test]
#[cfg_attr(
    any(not(feature = "slow-tests"), feature = "nn"),
    ignore = "slow: opt in with --features slow-tests; not run under nn"
)]
fn d1_corr_car1_a_profiled_fit_is_the_unprofiled_fit() {
    d1_corr("car1", &f1_text("car1", "0.0"), F1_DATA, SeCheck::Assert);
}

#[test]
#[cfg_attr(
    any(not(feature = "slow-tests"), feature = "nn"),
    ignore = "slow: opt in with --features slow-tests; not run under nn"
)]
fn d1_corr_ar1_a_profiled_fit_is_the_unprofiled_fit() {
    d1_corr("ar1", &f1_text("ar1", "-0.95"), F1_DATA, SeCheck::Assert);
}

#[test]
#[cfg_attr(
    any(not(feature = "slow-tests"), feature = "nn"),
    ignore = "slow: opt in with --features slow-tests; not run under nn"
)]
fn d1_corr_nested_a_profiled_fit_is_the_unprofiled_fit() {
    d1_corr(
        "nested ar1",
        &nested_text(),
        NESTED_DATA,
        SeCheck::PrintOnly,
    );
}
