//! NONMEM anchor for an infusion end that moves with the parameters across observations
//! (#3010): a duration-modeled infusion (`RATE=-2`, `D1 = TVD1·exp(η)`) on a linear `[odes]`
//! model, whose ends ferx walks inside their dosing interval rather than sorting them against
//! the observations (`integrate_tvcov_g`'s `soft`, `ode::linear::AffinePieces`) — and the same
//! model on the closed form, `pk one_cpt_iv` (#3033), whose analytic event walk walks them the
//! same way (`sens::propagate`'s `walked_ends` / `walk_cut`). The control stream is an
//! `ADVAN1 TRANS2` one, so the closed-form test is its literal twin.
//!
//! The kit is `nonmem_anchor/moving_infusion_end*`: the dataset (60 subjects, four doses q12h,
//! sampled 1, 2, 2.5, 3, 6 and 11.5 h after each, simulated from the closed form by
//! `moving_infusion_end_gen.py`), the estimation control stream (NONMEM 7.6.0 `ADVAN1 TRANS2`,
//! `METHOD=1 INTERACTION`; results in `nonmem_anchor/results/moving_infusion_end.{ext,lst}`), and
//! the evaluation control stream this test is anchored on.
//!
//! ## What this anchors — the objective at fixed parameters, subject by subject
//!
//! ferx's FOCEI objective at NONMEM's final estimates (`maxiter = 0`, ferx's `MAXEVAL=0`)
//! against NONMEM's at the same parameters, per subject. The objective is the inner EBE
//! search (whose gradient comes from the compiled plans) and the value at its optimum (from
//! the plan's value pass where it serves), so both run through the walked ends.
//!
//! This model's individual objectives have more than one mode in η_D1, and NONMEM's own
//! search from η = 0 stops in a higher one on five subjects: its cold `MAXEVAL=0` objective is
//! −2192.6846, ferx's −2204.3061, every other subject equal to within 3.5e-8. Started from
//! ferx's EBEs (`moving_infusion_end_eval.ctl`: `$ETAS` from `moving_infusion_end_etas.txt`,
//! `MCETA=1`), NONMEM reaches −2204.3060691562 (`results/moving_infusion_end_eval.phi`), and
//! the two agree on every subject — measured worst 3.6e-8, total 4.6e-9. So the engines
//! compute the same objective; they differed only in where an inner search started. (NONMEM's
//! estimation itself ends with `ERROR=134`, rounding errors, at OFV −2192.6846 — the same
//! higher modes.)
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

/// `nonmem_anchor/moving_infusion_end_fit.ferx` at NONMEM's final estimates
/// (`results/moving_infusion_end.ext`): `sigma` is the proportional variance, as `$SIGMA`.
const MODEL_AT_NONMEM_ESTIMATES: &str = r"
[parameters]
  theta TVCL(3.98653, 0.1, 100.0)
  theta TVV(28.3574, 1.0, 500.0)
  theta TVD1(1.88514, 0.1, 24.0)
  omega ETA_CL ~ 0.0979076
  omega ETA_V ~ 0.0873419
  omega ETA_D1 ~ 0.0997316
  sigma PROP_ERR ~ 0.0101129
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV * exp(ETA_V)
  D1 = TVD1 * exp(ETA_D1)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
";

/// NONMEM's per-subject `OBJ`, by `ID`, from the evaluation's `.phi` table.
fn nonmem_obj() -> Vec<(String, f64)> {
    let phi = std::fs::read_to_string("nonmem_anchor/results/moving_infusion_end_eval.phi")
        .expect("the NONMEM evaluation table");
    phi.lines()
        .skip(2)
        .map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            (f[1].to_string(), f.last().unwrap().parse().expect("an OBJ"))
        })
        .collect()
}

/// ferx's FOCEI objective at NONMEM's estimates equals NONMEM's on every subject. Measured
/// (Linux x86_64): worst subject 3.6e-8, total 4.6e-9 — the bound, 1e-6 per subject, is ~30×
/// that, and three orders below the smallest mode gap NONMEM's cold search showed (2e-4 on ID
/// 32). Non-degenerate: at the EBEs, the infusion ends of at least ten subjects lie before the
/// 2 h sample and of ten after the 2.5 h one, so the objective's inner searches move ends across
/// observations; and no subject's gradient falls back to finite differences, so the walked ends
/// are what ran.
#[test]
fn objective_matches_nonmem_with_infusion_ends_crossing_observations() {
    objective_matches_nonmem(MODEL_AT_NONMEM_ESTIMATES, &[]);
}

/// The same anchor on the closed form (#3033): [`MODEL_AT_NONMEM_ESTIMATES`] with its `[odes]`
/// block swapped for `pk one_cpt_iv`, the model `ADVAN1 TRANS2` itself computes. Its inner
/// searches move the ends across the samples through the analytic event walk's walked ends
/// (`sens::propagate`'s `walked_ends`).
///
/// Measured: 58 subjects equal to NONMEM's objective within 9.8e-8 (bound 1e-6). On two the
/// closed form's inner search stops **lower** than the point NONMEM evaluated, which started
/// from the `[odes]` fit's EBEs (`MCETA=1`), so those two are held to "not above NONMEM":
/// ID 32 in another mode, 0.164 lower (−51.7303 against −51.5658; the subject whose modes
/// NONMEM's cold search also split); and ID 27 by 1.7e-6, whose mode is a corner — its first
/// infusion end sits on the 3 h sample (2.99999999999999 h), where the objective has a kink,
/// so the 1e-6 differences in η between the two points show to first order rather than second.
///
/// Before #3033 this test failed on 41 subjects: a plan recorded where a runaway EBE step had
/// overflowed `D` to `∞` served every finite `D` as no infusion
/// (`a_plan_recorded_at_an_overflowed_window_does_not_serve_a_finite_one`).
#[test]
fn closed_form_objective_matches_nonmem_with_infusion_ends_crossing_observations() {
    let ode = "  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n";
    assert!(MODEL_AT_NONMEM_ESTIMATES.contains(ode));
    let closed_form = MODEL_AT_NONMEM_ESTIMATES.replace(ode, "  pk one_cpt_iv(cl=CL, v=V)\n");
    objective_matches_nonmem(&closed_form, &["27", "32"]);
}

/// The anchor on the model `src`: every subject's objective equal to NONMEM's, except the
/// subjects in `lower`, whose objective may only be at or below it (a better point of the
/// subject's own objective than the one NONMEM evaluated) — and must be: one that matches is a
/// stale exemption.
fn objective_matches_nonmem(src: &str, lower: &[&str]) {
    let model = parse_full_model(src).expect("the anchor model parses").model;
    // The engine the test names: the closed form has no ODE system.
    assert_eq!(model.ode_spec.is_some(), src.contains("[odes]"));
    let pop = read_nonmem_csv(Path::new("nonmem_anchor/moving_infusion_end.csv"), None, None)
        .expect("the anchor data loads");
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.outer_maxiter = 0;
    opts.run_covariance_step = false;
    opts.verbose = false;
    let result = fit(&model, &pop, &model.default_params, &opts).expect("the evaluation runs");

    assert!(
        !result.warnings.iter().any(|w| w.contains("finite-difference inner gradients")),
        "every subject's inner gradient is the plans': {:?}",
        result.warnings
    );
    let tvd1 = model.default_params.theta[2];
    let ends: Vec<f64> = result.subjects.iter().map(|s| tvd1 * s.eta[2].exp()).collect();
    let (early, late) =
        (ends.iter().filter(|&&e| e < 2.0).count(), ends.iter().filter(|&&e| e > 2.5).count());
    assert!(early >= 10 && late >= 10, "ends before 2 h: {early}, after 2.5 h: {late}");

    let want = nonmem_obj();
    assert_eq!(want.len(), result.subjects.len());
    let mut worst = 0.0f64;
    for (id, obj) in &want {
        let s = result
            .subjects
            .iter()
            .find(|s| &s.id == id)
            .unwrap_or_else(|| panic!("subject {id} in the fit"));
        let gap = (s.ofv_contribution - obj).abs();
        assert!(gap.is_finite(), "subject {id}: {}", s.ofv_contribution);
        if lower.contains(&id.as_str()) {
            eprintln!("subject {id}: ferx {:.10} vs NONMEM {obj:.10}", s.ofv_contribution);
            assert!(
                s.ofv_contribution < obj - 1e-6,
                "subject {id}: below NONMEM, ferx {:.10} vs {obj:.10}",
                s.ofv_contribution
            );
            continue;
        }
        assert!(gap < 1e-6, "subject {id}: ferx {:.10} vs NONMEM {obj:.10}", s.ofv_contribution);
        worst = worst.max(gap);
    }
    // The total over the subjects held to equality.
    let total: f64 = want.iter().filter(|w| !lower.contains(&w.0.as_str())).map(|w| w.1).sum();
    let ours: f64 = result
        .subjects
        .iter()
        .filter(|s| !lower.contains(&s.id.as_str()))
        .map(|s| s.ofv_contribution)
        .sum();
    eprintln!(
        "worst subject {worst:e}; ferx {ours:.10} vs NONMEM {total:.10} (OFV {:.10}); ends early {early} late {late}",
        result.ofv
    );
    assert!((ours - total).abs() < 1e-5, "ferx {ours:.10} vs NONMEM {total:.10}");
}
