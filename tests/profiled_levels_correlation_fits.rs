//! A residual `correlation` line with a `profile`d θ level block (#2110): fits to
//! convergence, gated `slow-tests`.
//!
//! **D1-corr**, the degenerate oracle of `profiled_levels_fits.rs` with the line: F1
//! (`tests/data/profile_mbma/f1_profile.ferx` on `mbma_placebo`, the arm kappa at `OCC`)
//! plus `theta RHO` and `correlation car1(RHO) by OCC` — the shape ferx-r's MBMA smoke
//! fits (step 13) — and the same with `ar1`; and `nested_levels_3` (sponsor > study >
//! arm) with a profiled `PLACEBO[STUDYID]` block and `correlation ar1(RHO) by ARM`,
//! the nested arm level, on that file and on a re-simulation of it whose arms carry a live
//! serial correlation (#2393). The profiled fit runs first; the unprofiled
//! fit of the same model is **started at the profiled optimum** and must not move the
//! OFV, the structural θ or ρ by more than the bounds; and at the profiled estimate the
//! two covariance steps (`r` on both sides) must give the same standard errors.
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
/// `nested_levels_3`'s design with DV re-simulated under AR(1) residuals within each arm,
/// ρ = 0.9, and a per-study placebo offset (#2393; `make_data.py` next to it).
const NESTED_LIVE_DATA: &str = "tests/data/nested_live_rho/nested_live_rho.csv";
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
/// nonlinear in them) with a profiled per-study baseline block `PLACEBO[STUDYID]` and
/// `correlation ar1(RHO) by ARM`, grouped by the nested arm level. `RHO` is the last θ.
///
/// Per study, not per (study, visit): this design has two doses per study and no placebo
/// arm, so a `[STUDYID, TIME]` block absorbs every study-visit mean and leaves only the
/// within-study dose contrast, which fixes `TVEMAX / ED50` and not each. That fit ran
/// TVEMAX to its bound 1000 (ED50 6342, the FD Hessian's smallest eigenvalue 1e-6) and
/// collapsed `ETA_SP` and `K_ARM` to the variance rail — no fixture for an SE comparison.
/// The Tier-1 solve tests keep the `[STUDYID, TIME]` shape at the file's structural point,
/// where nothing is estimated.
fn nested_text() -> String {
    let text = std::fs::read_to_string(root().join(NESTED)).unwrap();
    assert!(
        text.contains("  omega ETA_SP") && text.contains("  y = TVE0 + "),
        "fixture moved"
    );
    text.replace(
        "  omega ETA_SP",
        "  theta PLACEBO[STUDYID](0.0) profile\n  theta RHO(0.3, -0.95, 0.95)\n  omega ETA_SP",
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
        // has stopped moving. At the default `inner_tol = 1e-5` a nested `[STUDYID, TIME]`
        // shape of this fixture left 1.24e-6 on the OFV (Linux aarch64), above the profile
        // solve's own value, whose re-solves run at `STENCIL_INNER_TOL = 1e-7` — the
        // residual `nested_levels_correlation_nonmem_anchor.rs` measured on the FD inner
        // gradient — and the attribution misread the solve as drifting.
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

/// Returns the profiled fit, for a case that asserts on where it landed. `d_se` bounds
/// the relative SE gaps ([`D_SE`], [`D_SE_LIVE`]).
fn d1_corr(kind: &str, text: &str, data: &str, d_se: f64) -> FitResult {
    let prof_prep = prep(text, data, true);
    let (prof, t_prof) = fit(&prof_prep, &prof_prep.init_params, true, false);
    // First of all, a degenerate optimum: a structural θ on its bound, or a variance
    // collapsed toward 0. Neither covariance step means anything there, and a collapsed
    // variance sits on the start-box rail the unprofiled restart refuses. The nested
    // fixture's first shape (a `[STUDYID, TIME]` block) did both: without the line it ran
    // TVEMAX to 1000, with it ETA_SP and K_ARM to 6.1e-6.
    let t = &prof_prep.init_params;
    for k in 0..N_STRUCT {
        let (lo, hi, v) = (t.theta_lower[k], t.theta_upper[k], prof.theta[k]);
        assert!(
            (v - lo) > 1e-3 * (hi - lo).min(v.abs().max(1.0))
                && (hi - v) > 1e-3 * (hi - lo).min(v.abs().max(1.0)),
            "{kind}: {} = {v} is on its bounds [{lo}, {hi}]",
            prof.theta_names[k]
        );
    }
    let diag = |m: &nalgebra::DMatrix<f64>| (0..m.nrows()).map(|i| m[(i, i)]).collect::<Vec<_>>();
    let start_iov = t.omega_iov.as_ref();
    let names = t
        .omega
        .eta_names
        .iter()
        .chain(start_iov.map(|o| o.eta_names.iter()).into_iter().flatten());
    let at_start = diag(&t.omega.matrix)
        .into_iter()
        .chain(start_iov.map(|o| diag(&o.matrix)).unwrap_or_default());
    let fitted = diag(&prof.omega)
        .into_iter()
        .chain(prof.omega_iov.as_ref().map(diag).unwrap_or_default());
    for ((name, v0), v) in names.zip(at_start).zip(fitted) {
        assert!(
            v.is_finite() && v > 1e-3 * v0,
            "{kind}: the variance of {name} collapsed to {v:.3e} (start {v0})"
        );
    }
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
         unprofiled at that estimate {:.9}; objective gap {objective_gap:.3e}; structural θ \
         {:?}",
        prof.ofv,
        prof.n_iterations,
        prof.theta[rho],
        prof.converged,
        at_prof.ofv,
        &prof.theta[..N_STRUCT]
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
    assert!(worst_se < d_se, "{kind}: SE (θ, ρ) {worst_se:.3e}");
    assert!(worst_var_se < d_se, "{kind}: SE (ω, κ) {worst_var_se:.3e}");
    prof
}

// Realised on Linux aarch64 (`tools/linux-test.sh`, base `d66c0c51`), car1 / ar1 /
// nested. The profiled fits stop at 89.451279156 / 92.809709126 / 80.446292095 in
// 39 / 33 / 48 evaluations, the binary in 26 s. The nested optimum has ρ = 0.020, so
// there the line is barely live at convergence; the nested solve at a live ρ = 0.3 is the
// Tier-1 D0 check (`profile_tests.rs`). The fourth number in each list below is the live
// nested case (#2393, base `a94b5c66`): −273.070823846 in 43 evaluations, ρ = 0.648, the
// binary then 49 s; the other three cases reproduced their numbers on that base.

/// The profiled OFV against the unprofiled objective at the same estimate: 3.1e-13 /
/// 8.0e-13 / 1.9e-8 / 1.7e-8 — two inner solves of one objective. `2e-7` is 10× the
/// worst. The profile solve on a diagonal `R` reports 0.065 (car1) and 4.3 (ar1) OFV off
/// here, and 9.8 (nested) and 45.4 (live nested); the profile solve reading ρ/2 reports
/// 7.8e-3 on the nested case and 25.9 on the live one (macOS arm64, the mutation runs).
const D_OBJECTIVE: f64 = 2e-7;
/// The unprofiled restart's gain: 7.5e-7 / −2.1e-9 / −2.3e-7 / −5.3e-8 — its own
/// stopping rule and default `inner_tol`, not the profile. 13× the worst. With ρ's entry
/// of the envelope gradient zeroed, ρ stays at its start 0.3 and the restart gains 8.5
/// (nested) and 13.2 (live nested).
const D_IMPROVEMENT: f64 = 1e-5;
/// Structural θ, relative: 2.5e-4 (car1, where the restart gains its 7.5e-7) / 1.5e-8 /
/// 0 / 1.9e-12. 4× the worst.
const D_THETA: f64 = 1e-3;
/// ρ, absolute: 2.8e-6 / 3.4e-11 / 0 / 3.8e-13. 35× the worst.
const D_RHO: f64 = 1e-4;
/// Standard errors at one point, relative, the first three cases (car1 / ar1 / nested):
/// (θ, ρ) 4.0e-4 / 2.8e-4 / 5.1e-4, (ω, κ) 5.9e-5 / 5.6e-5 / 6.6e-6 (Linux aarch64). The
/// unprofiled side is the analytic R-matrix, exact, so these are the profiled `A_p`
/// stencil's own error at its default step `3e-3`. At the global `1e-2` they were
/// 4.5e-3 / 3.2e-3 / 5.2e-3, past this bound. 3.9× the worst.
const D_SE: f64 = 2e-3;
/// [`D_SE`] for the live nested case: (θ, ρ) 6.2e-4 (ρ's own), (ω, κ) 2.2e-5 (Linux
/// aarch64); 7.8e-3 at `1e-2`. 3.2× the worst.
const D_SE_LIVE: f64 = 2e-3;
/// Where the live nested fit's ρ must land: realised 0.648, so 0.25 inside each end. The
/// data are simulated at ρ = 0.9 and the fit lands lower by design, not by defect: the
/// realised residuals carry a pooled lag-1 correlation of 0.891, and the AR(1) maximum
/// likelihood on them alone (outside ferx, `tests/data/nested_live_rho/rho_check.py`)
/// gives 0.747 with a free per-study intercept (PLACEBO) and 0.205 with a free per-arm
/// amplitude on the time curve added; ferx's `K_ARM`, shrunk by its prior but not fixed,
/// falls between.
const RHO_LIVE: (f64, f64) = (0.4, 0.9);

#[test]
#[cfg_attr(
    any(not(feature = "slow-tests"), feature = "nn"),
    ignore = "slow: opt in with --features slow-tests; not run under nn"
)]
fn d1_corr_car1_a_profiled_fit_is_the_unprofiled_fit() {
    d1_corr("car1", &f1_text("car1", "0.0"), F1_DATA, D_SE);
}

#[test]
#[cfg_attr(
    any(not(feature = "slow-tests"), feature = "nn"),
    ignore = "slow: opt in with --features slow-tests; not run under nn"
)]
fn d1_corr_ar1_a_profiled_fit_is_the_unprofiled_fit() {
    d1_corr("ar1", &f1_text("ar1", "-0.95"), F1_DATA, D_SE);
}

#[test]
#[cfg_attr(
    any(not(feature = "slow-tests"), feature = "nn"),
    ignore = "slow: opt in with --features slow-tests; not run under nn"
)]
fn d1_corr_nested_a_profiled_fit_is_the_unprofiled_fit() {
    d1_corr("nested ar1", &nested_text(), NESTED_DATA, D_SE);
}

/// The nested case at a live ρ: on `nested_levels_3.csv` the optimum has ρ = 0.020, so
/// the comparison above holds with the line all but switched off. This data is simulated
/// at ρ = 0.9 within each arm and fits to ρ = 0.648 ([`RHO_LIVE`]), so the profile
/// solve's `R` is far from diagonal along the optimizer's path and at the optimum the
/// oracle compares. A profile solve reading ρ/2 is 7.8e-3 OFV off on the nested case and
/// 25.9 here (macOS arm64, the mutation run).
#[test]
#[cfg_attr(
    any(not(feature = "slow-tests"), feature = "nn"),
    ignore = "slow: opt in with --features slow-tests; not run under nn"
)]
fn d1_corr_nested_live_rho_a_profiled_fit_is_the_unprofiled_fit() {
    let prof = d1_corr(
        "nested live ar1",
        &nested_text(),
        NESTED_LIVE_DATA,
        D_SE_LIVE,
    );
    let rho = prof.theta[prof.theta.len() - 1];
    assert!(
        (RHO_LIVE.0..RHO_LIVE.1).contains(&rho),
        "nested live ar1: ρ = {rho}, outside the live range {RHO_LIVE:?}"
    );
}
