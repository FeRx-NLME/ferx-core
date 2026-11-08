//! #2797: a coordinate collapsed on ferx's internal lower guard is reported in
//! `FitResult::held_at_bound`, and the covariance step holds it — the other SEs conditional
//! on it at its estimate — only after a clean verdict on the unconditional `R`.
//!
//! Tier 2: every fit is an evaluation (`maxiter = 0`) at a fixed point, which runs the
//! covariance step once, so nothing here converges. A `maxiter = 0` fit reports
//! `converged = false`, so no test asserts it.
//!
//! The fixture (`tests/data/guard_hold/`, with its generator) is a one-compartment oral
//! population simulated with **no** KA variability, fitted with an ETA on KA: the ETA
//! collapses onto the rail ω = e⁻¹² = 6.1442123533282098e-6. The two points straddle the
//! verdict:
//! - **stationary** — the converged estimate (OFV 353.754540470156, macOS arm64): a clean
//!   `R`, so ETA_KA is held;
//! - **poor σ** — the same with σ = 0.02 (OFV 7887.7): the analytic `R` is indefinite
//!   along ETA_KA itself (min eig −21.96), floored and reported "severe". Holding there
//!   first would hide that floor and move σ's SE by −98.6% (the audit's probe G), so the
//!   step must not hold.

use std::path::{Path, PathBuf};

use ferx_core::io::output::parameter_table;
use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::types::{FitOptions, FitResult, HeldReason, ParameterKind};
use ferx_core::{run_covariance, run_model_with_data};

const DATA: &str = "tests/data/guard_hold/sim_oral_no_ka_iiv.csv";
/// The rail: ω = exp(−12), the log-Cholesky diagonal's lower guard −6, written with 17
/// significant digits so it packs back onto −6 exactly.
const RAIL: &str = "6.1442123533282098e-6";
/// The converged σ of the fixture (SD scale).
const SIGMA_HAT: &str = "0.10069194732313741";
/// ETA_KA's packed row: `[TVCL, TVV, TVKA, ETA_CL, ETA_V, ETA_KA, PROP_ERR, …]`.
const ETA_KA_ROW: usize = 5;

fn model_source(sigma: &str, kappa: bool, analytic_cov: bool) -> String {
    let (kappa_decl, ka, iov) = if kappa {
        (
            format!("  kappa K_KA ~ {RAIL}\n"),
            "KA = TVKA * exp(ETA_KA + K_KA)",
            "  iov_column = OCC\n",
        )
    } else {
        (String::new(), "KA = TVKA * exp(ETA_KA)", "")
    };
    format!(
        "[parameters]
  theta TVCL(0.12250099819205257, 0.001, 10.0)
  theta TVV(7.753670975867449, 0.1, 500.0)
  theta TVKA(0.9982755334099482, 0.01, 50.0)
  omega ETA_CL ~ 0.0792688562315328
  omega ETA_V  ~ 0.05944155243978235
  omega ETA_KA ~ {RAIL}
{kappa_decl}  sigma PROP_ERR ~ {sigma} (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  {ka}

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = focei
  maxiter    = 0
  covariance = true
  analytic_cov_hessian = {analytic_cov}
{iov}"
    )
}

/// Fit `src` (an evaluation) and return the fit, the model path and the population, for
/// the FIX-clone `run_covariance` comparisons.
fn evaluate(
    name: &str,
    src: &str,
) -> (
    FitResult,
    ferx_core::types::CompiledModel,
    ferx_core::types::Population,
    tempfile::TempDir,
) {
    let dir = tempfile::tempdir().unwrap();
    let path: PathBuf = dir.path().join(format!("{name}.ferx"));
    std::fs::write(&path, src).unwrap();
    let (fit, pop) = run_model_with_data(path.to_str().unwrap(), Some(DATA)).expect("evaluate");
    let model = parse_model_file(Path::new(&path)).expect("parse");
    (fit, model, pop, dir)
}

fn all_ses(r: &FitResult) -> Vec<f64> {
    let mut v = r.se_theta.clone().unwrap_or_default();
    v.extend(r.se_omega.clone().unwrap_or_default());
    v.extend(r.se_sigma.clone().unwrap_or_default());
    v.extend(r.se_kappa.clone().unwrap_or_default());
    v
}

/// The same fit with every collapse FIX, through `run_covariance`: the matrix a NONMEM
/// `$OMEGA … FIX` run computes, from the same engine. It pins *conditional versus
/// marginal*, not `R` itself — the analytic route builds both from the same `R` entries;
/// the NONMEM anchor (`nonmem_anchor/guard_hold_*`) is the external reference.
fn fix_clone(
    fit: &FitResult,
    model: &ferx_core::types::CompiledModel,
    pop: &ferx_core::types::Population,
    analytic: bool,
) -> FitResult {
    let mut held = fit.clone();
    for h in &fit.held_at_bound {
        match h.kind {
            ParameterKind::Omega => held.omega_fixed[h.index] = true,
            ParameterKind::Kappa => held.kappa_fixed[h.index] = true,
            ParameterKind::Sigma => held.sigma_fixed[h.index] = true,
            ParameterKind::Theta => held.theta_fixed[h.index] = true,
        }
    }
    let options = FitOptions {
        analytic_cov_hessian: analytic,
        ..FitOptions::default()
    };
    run_covariance(&held, Some(model), Some(pop), &options).expect("FIX clone")
}

fn regularized(r: &FitResult) -> bool {
    r.warnings
        .iter()
        .any(|w| w.starts_with("Covariance step regularized"))
}

fn guard_warning(r: &FitResult) -> &str {
    r.warnings
        .iter()
        .find(|w| w.starts_with("Internal optimizer parameter guard reached"))
        .expect("the guard warning")
}

/// Tests 5 and T-6b in one, so the pair cannot silently land on one side: at the
/// stationary point ETA_KA is held, at the poor-σ point it is listed but not held, and the
/// straddle itself (no floor / a floor) is asserted.
///
/// Mutations, each named by the assertion that dies: never hold → `cov_fixed` on the
/// stationary point; hold before the verdict → the poor-σ floor and σ-SE ratio; drop the
/// list fill → `held_at_bound`.
#[test]
fn a_collapse_is_held_at_a_stationary_point_and_only_listed_at_a_poor_one() {
    // ── Stationary: held ──────────────────────────────────────────────────────────────
    let (f, model, pop, _d) = evaluate("stationary", &model_source(SIGMA_HAT, false, true));
    assert!(f.covariance_matrix.is_some(), "{:?}", f.warnings);
    assert!(
        !regularized(&f),
        "the stationary point has a clean R: {:?}",
        f.warnings
    );
    let held: Vec<(&str, ParameterKind, usize, usize, HeldReason)> = f
        .held_at_bound
        .iter()
        .map(|h| (h.name.as_str(), h.kind, h.index, h.packed_index, h.reason))
        .collect();
    assert_eq!(
        held,
        [(
            "ETA_KA",
            ParameterKind::Omega,
            2,
            ETA_KA_ROW,
            HeldReason::CollapsedAtGuard
        )]
    );
    assert!(
        f.cov_fixed[ETA_KA_ROW],
        "the step held ETA_KA: {:?}",
        f.cov_fixed
    );
    assert_eq!(f.cov_fixed.iter().filter(|&&b| b).count(), 1);
    let cov = f.covariance_matrix.as_ref().unwrap();
    for k in 0..cov.nrows() {
        assert_eq!(cov[(ETA_KA_ROW, k)], 0.0);
    }
    assert_eq!(f.se_omega.as_ref().unwrap()[2], 0.0);
    for (i, se) in all_ses(&f).iter().enumerate() {
        assert!(se.is_finite(), "SE {i} = {se}");
        if i != ETA_KA_ROW {
            assert!(*se > 0.0, "SE {i} = {se}");
        }
    }
    assert!(
        guard_warning(&f).ends_with(
            "Their standard errors are not reported; the other standard errors are computed \
             with them held at the floor."
        ),
        "{}",
        guard_warning(&f)
    );
    let row = parameter_table(&f)
        .lines()
        .find(|l| l.starts_with("ETA_KA "))
        .unwrap()
        .to_string();
    assert_eq!(
        row.split_whitespace().collect::<Vec<_>>()[2..4],
        ["N/A", "N/A"],
        "{row}"
    );

    // The held SEs are exactly the FIX clone's on the analytic R: a block of the analytic
    // R along the free axes is the full R's sub-block to the bit (`analytic_cov_axes`), and
    // both invert it the same way. Measured: 0 ULP on every SE (macOS arm64).
    let clone = fix_clone(&f, &model, &pop, true);
    assert_eq!(
        all_ses(&clone),
        all_ses(&f),
        "analytic: held SEs = FIX clone SEs"
    );
    assert_eq!(clone.covariance_matrix, f.covariance_matrix);

    // ── Poor σ: listed, not held ──────────────────────────────────────────────────────
    let (g, model, pop, _d) = evaluate("poor_sigma", &model_source("0.02", false, true));
    assert!(g.covariance_matrix.is_some(), "{:?}", g.warnings);
    assert!(
        regularized(&g),
        "the straddle: poor σ floors R: {:?}",
        g.warnings
    );
    let floor = g
        .warnings
        .iter()
        .find(|w| w.starts_with("Covariance step regularized"))
        .unwrap();
    assert!(floor.contains("severity: severe"), "{floor}");
    assert_eq!(g.held_at_bound.len(), 1);
    assert_eq!(g.held_at_bound[0].name, "ETA_KA");
    assert!(
        !g.cov_fixed[ETA_KA_ROW],
        "a floored R is not held: {:?}",
        g.cov_fixed
    );
    // The unconditional SE is kept in the field; only the display says N/A.
    assert!(g.se_omega.as_ref().unwrap()[2] > 0.0);
    let row = parameter_table(&g)
        .lines()
        .find(|l| l.starts_with("ETA_KA "))
        .unwrap()
        .to_string();
    assert_eq!(
        row.split_whitespace().collect::<Vec<_>>()[2..4],
        ["N/A", "N/A"],
        "{row}"
    );
    assert!(
        guard_warning(&g).ends_with(" Their standard errors are not reported."),
        "{}",
        guard_warning(&g)
    );
    // Holding before the verdict is what this pins against: the FIX clone at the poor
    // point has no floor and a σ SE about 70× smaller (0.00015 against 0.0105, measured);
    // the reported SE must stay the unconditional one.
    let clone = fix_clone(&g, &model, &pop, true);
    assert!(!regularized(&clone), "{:?}", clone.warnings);
    let (sg, sc) = (
        g.se_sigma.as_ref().unwrap()[0],
        clone.se_sigma.as_ref().unwrap()[0],
    );
    assert!(
        sg.is_finite() && sc.is_finite() && sg > 10.0 * sc,
        "σ SE {sg} vs held {sc}"
    );
}

/// Test 6, FD side: on the FD stencil the step holds too, with the other SEs equal to the
/// FD FIX clone's. Measured: 0 ULP on every SE (macOS arm64) — each stencil entry is the
/// same objective evaluations whichever free set the step runs over — so the bound is
/// equality. The route itself is asserted, so a fixture that silently ran the same engine
/// twice fails by name: the stencil's SEs differ from the analytic R's (by up to 1.8e-4
/// relative, measured).
#[test]
fn a_collapse_is_held_on_the_fd_stencil_route_too() {
    let (f, model, pop, _d) = evaluate("stationary_fd", &model_source(SIGMA_HAT, false, false));
    assert!(!regularized(&f), "{:?}", f.warnings);
    assert!(f.cov_fixed[ETA_KA_ROW], "{:?}", f.cov_fixed);
    assert_eq!(f.se_omega.as_ref().unwrap()[2], 0.0);
    let clone = fix_clone(&f, &model, &pop, false);
    assert_eq!(all_ses(&clone), all_ses(&f), "FD: held SEs = FIX clone SEs");
    let (analytic, _, _, _d) = evaluate("stationary_an", &model_source(SIGMA_HAT, false, true));
    assert_ne!(
        all_ses(&analytic),
        all_ses(&f),
        "the two fits must run different routes"
    );
}

/// Test 7: an IOV κ collapsed with the ETA — both held, two entries in packed order.
#[test]
fn a_collapsed_kappa_is_held_with_the_eta() {
    let (f, model, pop, _d) = evaluate("kappa", &model_source(SIGMA_HAT, true, true));
    assert!(!regularized(&f), "{:?}", f.warnings);
    let held: Vec<(&str, ParameterKind, usize, usize)> = f
        .held_at_bound
        .iter()
        .map(|h| (h.name.as_str(), h.kind, h.index, h.packed_index))
        .collect();
    // Packed: [θ×3, Ω×3, σ, κ].
    assert_eq!(
        held,
        [
            ("ETA_KA", ParameterKind::Omega, 2, ETA_KA_ROW),
            ("K_KA", ParameterKind::Kappa, 0, 7),
        ]
    );
    assert!(
        f.cov_fixed[ETA_KA_ROW] && f.cov_fixed[7],
        "{:?}",
        f.cov_fixed
    );
    assert_eq!(f.se_kappa.as_ref().unwrap()[0], 0.0);
    let clone = fix_clone(&f, &model, &pop, true);
    assert_eq!(all_ses(&clone), all_ses(&f));
}

/// Test 10 and D7: with nothing on the rail the list is empty and nothing is marked; and a
/// `run_covariance` re-run on the held fit rebuilds the list, the mask and the warning from
/// its own matrix — here the same ones, bit for bit.
#[test]
fn no_collapse_lists_nothing_and_a_re_run_rebuilds_the_report() {
    let src = model_source(SIGMA_HAT, false, true)
        .replace(&format!("omega ETA_KA ~ {RAIL}"), "omega ETA_KA ~ 0.1");
    let (f, _, _, _d) = evaluate("interior", &src);
    assert!(f.held_at_bound.is_empty());
    assert!(!f.cov_fixed.iter().any(|&b| b), "{:?}", f.cov_fixed);
    assert!(!f.warnings.iter().any(|w| w.contains("parameter guard")));

    let (f, model, pop, _d) = evaluate("rerun", &model_source(SIGMA_HAT, false, true));
    let options = FitOptions::default();
    let again = run_covariance(&f, Some(&model), Some(&pop), &options).expect("re-run");
    assert_eq!(again.held_at_bound, f.held_at_bound);
    assert_eq!(again.cov_fixed, f.cov_fixed);
    assert_eq!(all_ses(&again), all_ses(&f));
    assert_eq!(guard_warning(&again), guard_warning(&f));
    assert_eq!(
        again
            .warnings
            .iter()
            .filter(|w| w.starts_with("Internal optimizer parameter guard"))
            .count(),
        1,
        "the stale copy is dropped"
    );
}
