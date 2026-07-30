//! NONMEM cross-check for the built-in Savic transit-compartment absorption
//! function `transit(n, mtt)` (PR FeRx-NLME/ferx-core#343, issue #322).
//!
//! The model, data, and NONMEM control stream are the anchor kit in the
//! repository's `nonmem_anchor/` directory:
//!
//!   - `nonmem_anchor/savic_transit.ctl` is the NONMEM 7.x `ADVAN13 TOL=9`
//!     FOCEI control; its final estimates are in `nonmem_anchor/results/`.
//!   - `data/transit_oral.csv` is the 20-subject / 240-observation single-dose
//!     oral dataset, simulated from the Savic model (truths `TVCL=5`, `TVV=50`,
//!     `TVKA=1`, `TVMTT=1`, `TVN=3`; IIV on CL & V ω²=0.09; prop SD 0.15).
//!
//! ## NONMEM cross-check
//!
//! From `nonmem_anchor/results/savic_transit.ext` (NONMEM, FOCEI):
//!
//! | Quantity | NONMEM | ferx (tight ODE tols) |
//! |----------|--------|-----------------------|
//! | OFV      | −1077.13 | −1076.67 |
//! | TVCL     | 5.386  | 5.453 |
//! | TVV      | 56.169 | 55.759 |
//! | TVKA     | 0.952  | 0.950 |
//! | TVMTT    | 0.965  | 0.966 |
//! | TVN      | 3.133  | 3.128 |
//!
//! ferx and NONMEM agree only when the ODE solver runs at NONMEM-equivalent
//! accuracy (`ode_reltol = ode_abstol = 1e-9`, matching `TOL=9`). The loose
//! defaults (`1e-4`/`1e-6`) recover the fixed effects but inject integration
//! noise into the FOCEI Hessian that inflates the ω² estimates (see
//! `docs/model-file/absorption.qmd` "Verification against NONMEM"). This
//! test therefore sets the tight tolerances explicitly.

use ferx_core::ode::{ode_predictions_with_solver_stats, OdeSolverStats};
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

/// `nonmem_anchor/transit_savic_fit.ferx` minus the `[fit_options]` block, which
/// is set programmatically below so the test pins the ODE tolerances.
const MODEL_SRC: &str = r"
[parameters]
  theta TVCL(5.0,   0.1, 100.0)
  theta TVV(50.0,   5.0, 500.0)
  theta TVKA(1.0,  0.05,  24.0)
  theta TVMTT(1.0, 0.05,  24.0)
  theta TVN(3.0,    0.1,  30.0)

  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09

  sigma PROP_ERR ~ 0.15 (sd)

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV  * exp(ETA_V)
  KA  = TVKA
  MTT = TVMTT
  NTR = TVN

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = transit(n=NTR, mtt=MTT) - KA*depot
  d/dt(central) = KA*depot/V - CL/V*central

[error_model]
  DV ~ proportional(PROP_ERR)
";

/// FOCEI fit of the Savic transit model, at NONMEM-equivalent ODE accuracy, must
/// land on the NONMEM objective.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored Savic transit (#343/#322) acceptance: opt in with --features slow-tests"
)]
fn savic_transit_matches_nonmem_ofv() {
    // NONMEM #OBJV is −1077.13 (nonmem_anchor/results/savic_transit.ext). ferx
    // lands at −1076.67 on the FD path CI exercises, with the tight tolerances
    // set below; the ±2 band excludes the loose-tolerance regime (−1069.07,
    // ω² inflated ~60–120%) this test guards against.
    const EXPECTED_OFV: f64 = -1076.67;
    const TOLERANCE: f64 = 2.0;

    let model = parse_full_model(MODEL_SRC)
        .expect("Savic transit model must parse")
        .model;
    let pop = read_nonmem_csv(Path::new("data/transit_oral.csv"), None, None)
        .expect("transit_oral data must load");

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.outer_maxiter = 500;
    opts.run_covariance_step = false;
    opts.verbose = false;
    // NONMEM-equivalent ODE accuracy (TOL=9). Without this the ω² inflate and
    // the OFV drifts ~8 units high — the regression this test pins.
    opts.ode_reltol = 1e-9;
    opts.ode_abstol = 1e-9;
    opts.inner_tol = 1e-6;

    let result =
        fit(&model, &pop, &model.default_params, &opts).expect("Savic transit fit must run");

    assert!(
        result.ofv.is_finite(),
        "OFV must be finite, got {}",
        result.ofv
    );
    assert!(
        (result.ofv - EXPECTED_OFV).abs() < TOLERANCE,
        "OFV {:.3} is outside the NONMEM-anchored band {:.2} ± {:.1} (NONMEM #OBJV \
         −1077.13, nonmem_anchor/results/savic_transit.ext); a value near −1069 \
         signals the loose-ODE-tolerance regression (inflated ω²) this guards",
        result.ofv,
        EXPECTED_OFV,
        TOLERANCE,
    );
}

/// The #387 solver-decision gate: at NONMEM-equivalent tolerances the Savic
/// transit anchor should be accepted-step dominated, not rejection/min-step
/// dominated. That means a stabilized low-order method such as ROCK2 should not
/// be treated as the expected #385 fix without a separate loose-tolerance stiff
/// benchmark.
#[test]
fn savic_transit_1e9_rk45_stats_are_accepted_step_dominated() {
    let mut model = parse_full_model(MODEL_SRC)
        .expect("Savic transit model must parse")
        .model;
    let pop = read_nonmem_csv(Path::new("data/transit_oral.csv"), None, None)
        .expect("transit_oral data must load");
    {
        let ode = model
            .ode_spec
            .as_mut()
            .expect("Savic transit model must be ODE-based");
        ode.solver_opts.reltol = 1e-9;
        ode.solver_opts.abstol = 1e-9;
    }

    // Tight-tolerance ferx estimates from the NONMEM anchor documentation:
    // docs/model-file/absorption.qmd, "Savic transit (`transit`)".
    let theta = [5.453, 55.759, 0.950, 0.966, 3.128];
    let eta = vec![0.0; model.n_eta];
    let mut stats = OdeSolverStats::default();
    let mut n_pred = 0usize;

    for subject in &pop.subjects {
        let pk = (model.pk_param_fn)(&theta, &eta, &subject.covariates, 0.0);
        let (pred, subject_stats) = ode_predictions_with_solver_stats(
            model.ode_spec.as_ref().unwrap(),
            &pk.values,
            &theta,
            &eta,
            subject,
        );
        n_pred += pred.len();
        stats.attempted_steps += subject_stats.attempted_steps;
        stats.accepted_steps += subject_stats.accepted_steps;
        stats.rejected_steps += subject_stats.rejected_steps;
        stats.min_step_clamped_steps += subject_stats.min_step_clamped_steps;
    }

    assert_eq!(n_pred, 240);
    assert_eq!(
        stats.attempted_steps,
        stats.accepted_steps + stats.rejected_steps
    );
    assert!(
        stats.accepted_steps > 10 * stats.rejected_steps.max(1),
        "Savic transit at 1e-9 should be accepted-step dominated; stats = {stats:?}"
    );
    assert_eq!(
        stats.min_step_clamped_steps, 0,
        "Savic transit at 1e-9 should not be min-step clamped; stats = {stats:?}"
    );
}

/// The #387 solver-decision gate, part 2: the *same* anchor under every available
/// `ode_method`, at NONMEM-equivalent accuracy and at the loose defaults.
///
/// #387 asks that a method be chosen from a measured regime rather than from the assumption
/// that a slow ODE fit must be stability-limited. Part 1
/// ([`savic_transit_1e9_rk45_stats_are_accepted_step_dominated`]) establishes that this fit is
/// **accuracy-limited**: RK45 accepts nearly every step and is never min-step clamped. This
/// test carries that conclusion over to the stiff methods — they integrate the same anchor to
/// the same predictions without clamping, but they do *not* rescue a fit that was never
/// stability-limited in the first place. The Rosenbrock steppers are for the loose-tolerance
/// stiff regime (fast binding / TMDD, see `tests/stiff_ode_method.rs`), not for this one.
///
/// Run with `--nocapture` to see the per-method step counts and wall-clock — that printout is
/// the instrumentation output #387 asks to have on record.
#[test]
fn savic_transit_solver_regime_across_ode_methods() {
    use ferx_core::ode::OdeMethod;
    use std::time::Instant;

    let theta = [5.453, 55.759, 0.950, 0.966, 3.128];

    for (label, reltol, abstol) in [
        ("1e-9 (NONMEM TOL=9)", 1e-9, 1e-9),
        ("defaults", 1e-4, 1e-6),
    ] {
        let mut baseline: Option<Vec<f64>> = None;
        println!("\n=== Savic transit anchor @ {label} ===");
        println!(
            "{:<14} {:>9} {:>9} {:>9} {:>9} {:>10}",
            "method", "attempt", "accept", "reject", "clamped", "ms"
        );

        for method in [
            OdeMethod::Rk45,
            OdeMethod::Vern7,
            OdeMethod::Rosenbrock23,
            OdeMethod::Rodas4,
            OdeMethod::Rodas5P,
        ] {
            let mut model = parse_full_model(MODEL_SRC)
                .expect("Savic transit model must parse")
                .model;
            let pop = read_nonmem_csv(Path::new("data/transit_oral.csv"), None, None)
                .expect("transit_oral data must load");
            {
                let ode = model.ode_spec.as_mut().expect("must be ODE-based");
                ode.solver_opts.reltol = reltol;
                ode.solver_opts.abstol = abstol;
                ode.solver_opts.method = method;
            }
            let eta = vec![0.0; model.n_eta];

            let mut stats = OdeSolverStats::default();
            let mut preds: Vec<f64> = Vec::new();
            let t0 = Instant::now();
            for subject in &pop.subjects {
                let pk = (model.pk_param_fn)(&theta, &eta, &subject.covariates, 0.0);
                let (pred, s) = ode_predictions_with_solver_stats(
                    model.ode_spec.as_ref().unwrap(),
                    &pk.values,
                    &theta,
                    &eta,
                    subject,
                );
                preds.extend_from_slice(&pred);
                stats.attempted_steps += s.attempted_steps;
                stats.accepted_steps += s.accepted_steps;
                stats.rejected_steps += s.rejected_steps;
                stats.min_step_clamped_steps += s.min_step_clamped_steps;
            }
            let ms = t0.elapsed().as_secs_f64() * 1e3;

            println!(
                "{:<14} {:>9} {:>9} {:>9} {:>9} {:>10.1}",
                format!("{method:?}"),
                stats.attempted_steps,
                stats.accepted_steps,
                stats.rejected_steps,
                stats.min_step_clamped_steps,
                ms
            );

            // No method is stability-throttled on this anchor — that is the #387 finding.
            assert_eq!(
                stats.min_step_clamped_steps, 0,
                "{method:?} @ {label} hit min_dt on an accuracy-limited fit; stats = {stats:?}"
            );
            // Every method must integrate the anchor to the same predictions.
            match &baseline {
                None => baseline = Some(preds),
                Some(b) => {
                    assert_eq!(preds.len(), b.len(), "{method:?} @ {label}");
                    for (p, r) in preds.iter().zip(b.iter()) {
                        let tol = 1e-6 + 1e-3 * r.abs();
                        assert!(
                            (p - r).abs() <= tol,
                            "{method:?} @ {label}: PRED {p:.9} vs RK45 {r:.9}"
                        );
                    }
                }
            }
        }
    }
}
