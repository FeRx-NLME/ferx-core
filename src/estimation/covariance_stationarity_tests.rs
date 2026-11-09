//! #2541: a sampler's estimate is tested for stationarity from the gradient of the objective
//! `R` is the Hessian of, not from `R`'s positive-definiteness.
//!
//! The fixture is `examples/warfarin.ferx` on `data/warfarin.csv` at its FOCEI minimum (the
//! release CLI's estimates, 6 significant figures) and at points displaced along `TVCL` only,
//! where `R` stays positive definite. That is the #2536 seed-2 cell: before #2541 a sampler's
//! fit there reported standard errors with no warning however far it sat from the minimum.
use super::{compute_covariance, CovarianceOutput, CovarianceStepResult};
use crate::estimation::cov_diagnostics::{
    NON_STATIONARY_DECREMENT_OFV, NON_STATIONARY_OPTIMUM_TOKEN, NON_STATIONARY_TOKEN,
};
use crate::estimation::parameterization::pack_params;
use crate::types::{
    CompiledModel, CovarianceMethod, EstimationMethod as M, FitOptions, Population,
};
use nalgebra::{DMatrix, DVector};

/// The FOCEI minimum's `TVCL`, as the release CLI prints it.
const TVCL_HAT: f64 = 0.132695;

/// `examples/warfarin.ferx` with its initial estimates moved to the FOCEI minimum, `TVCL`
/// replaced by `tvcl`.
fn warfarin_at(tvcl: f64) -> (CompiledModel, Population) {
    warfarin_at_bounded(tvcl, 0.001)
}

/// [`warfarin_at`] with `TVCL`'s lower bound at `lower` (#2785).
fn warfarin_at_bounded(tvcl: f64, lower: f64) -> (CompiledModel, Population) {
    let src = std::fs::read_to_string("examples/warfarin.ferx").expect("model file");
    let src = src
        .replace(
            "theta TVCL(0.2, 0.001, 10.0)",
            &format!("theta TVCL({tvcl:?}, {lower:?}, 10.0)"),
        )
        .replace("theta TVV(10.0,", "theta TVV(7.737706,")
        .replace("theta TVKA(1.5,", "theta TVKA(0.810796,")
        .replace("omega ETA_CL ~ 0.09", "omega ETA_CL ~ 0.028590")
        .replace("omega ETA_V  ~ 0.04", "omega ETA_V  ~ 0.009592")
        .replace("omega ETA_KA ~ 0.30", "omega ETA_KA ~ 0.335871")
        .replace(
            "sigma PROP_ERR ~ 0.02 (sd)",
            "sigma PROP_ERR ~ 0.010565 (sd)",
        );
    let model = crate::parser::model_parser::parse_model_string(&src).expect("parse");
    assert_eq!(
        model.default_params.theta[0], tvcl,
        "the TVCL replacement must have applied"
    );
    let (pop, _) =
        crate::api::read_population_for(&model, &None, "data/warfarin.csv", None, None, None, &[])
            .expect("data");
    (model, pop)
}

fn opts(method: M) -> FitOptions {
    FitOptions {
        method,
        interaction: true,
        covariance_method: CovarianceMethod::Hessian,
        ..FitOptions::default()
    }
}

/// `compute_covariance` at the model's initial estimates, as `method`'s fit would call it.
fn cov(model: &CompiledModel, pop: &Population, method: M) -> (Vec<f64>, CovarianceOutput) {
    let o = opts(method);
    let p = model.default_params.clone();
    let x = pack_params(&p);
    let (etas, hs): (Vec<DVector<f64>>, Vec<DMatrix<f64>>) = pop
        .subjects
        .iter()
        .map(|s| {
            let e = crate::estimation::inner_optimizer::find_ebe_for_stage(
                model, s, &p, 200, 1e-8, None, None, 0, &o,
            );
            (e.eta, e.h_matrix)
        })
        .unzip();
    let kappas = vec![Vec::new(); pop.subjects.len()];
    match compute_covariance(&x, &p, model, pop, &etas, &hs, &kappas, &o) {
        CovarianceStepResult::Success(out) => (x, out),
        CovarianceStepResult::Unusable(m) => panic!("{method:?}: unusable: {m}"),
        CovarianceStepResult::FailedNonPd { reason, .. } => panic!("{method:?}: non-PD: {reason}"),
    }
}

fn stationarity_warnings(out: &CovarianceOutput) -> Vec<&String> {
    out.warnings
        .iter()
        .filter(|w| w.starts_with(NON_STATIONARY_TOKEN))
        .collect()
}

/// The decrement the warning prints, read back from its text.
fn printed_decrement(w: &str) -> f64 {
    let tail = w
        .split("by an estimated ")
        .nth(1)
        .expect("the decrement clause");
    tail.split(" OFV units")
        .next()
        .unwrap()
        .parse()
        .expect("a number")
}

/// The squared Mahalanobis distance `δᵀ C⁻¹ δ` from the minimum to `x`, in the minimum's own
/// `R⁻¹` — what the Newton decrement estimates under a quadratic model, computed without it.
fn mahalanobis_sq(c_hat: &DMatrix<f64>, x_hat: &[f64], x: &[f64]) -> f64 {
    let d = DVector::from_iterator(x.len(), x.iter().zip(x_hat).map(|(a, b)| a - b));
    let c_inv = c_hat
        .clone()
        .try_inverse()
        .expect("R⁻¹ at the minimum inverts");
    d.dot(&(&c_inv * &d))
}

/// Three straddles in one test, so a gate stuck on either side of any of them reddens it:
///
/// - **distance**: a SAEM estimate `far` from the minimum warns, one `near` it does not, and the
///   two decrements sit on either side of [`NON_STATIONARY_DECREMENT_OFV`];
/// - **method**: FOCEI at the `far` point says nothing, and its matrix is SAEM's to the bit —
///   the warning adds a sentence and changes no number;
/// - **scale**: the printed decrement agrees with the squared Mahalanobis distance to the
///   minimum computed from the minimum's own `R⁻¹`, so a decrement off by the `R = ½·H_ofv`
///   factor of 2 (either way) fails here rather than moving the threshold silently.
///
/// Measured (debug, macOS arm64): `far` (TVCL × 1.06) prints 1.267 against a Mahalanobis² of
/// 1.187 (6.7% apart; the objective is not quadratic, and at × 1.08 / 1.10 the two read
/// 2.341 / 2.071 and 3.915 / 3.176), and `near` (TVCL × 1.04) has a Mahalanobis² of 0.538. At
/// × 1.2 `R` is indefinite and the #1781 fallback takes over, so `far` is kept close in.
#[test]
fn a_positive_definite_r_at_a_non_stationary_sampler_estimate_warns() {
    let (model, pop) = warfarin_at(TVCL_HAT);
    let (x_hat, at_min) = cov(&model, &pop, M::FoceI);
    assert!(
        stationarity_warnings(&at_min).is_empty(),
        "{:?}",
        at_min.warnings
    );
    let (_, saem_at_min) = cov(&model, &pop, M::Saem);
    assert!(
        stationarity_warnings(&saem_at_min).is_empty(),
        "SAEM at the minimum: {:?}",
        saem_at_min.warnings
    );

    // Far: the gate fires, R⁻¹ is still what comes back.
    let (model, pop) = warfarin_at(TVCL_HAT * 1.06);
    let (x_far, saem) = cov(&model, &pop, M::Saem);
    assert_eq!(saem.method, CovarianceMethod::Hessian, "R⁻¹ is returned");
    let w = stationarity_warnings(&saem);
    assert_eq!(w.len(), 1, "{:?}", saem.warnings);
    assert!(
        w[0].starts_with(
            "W_COV_NON_STATIONARY_ESTIMATE: Covariance step: the SAEM estimate is not a \
             stationary point of the FOCEI objective"
        ),
        "{}",
        w[0]
    );
    assert!(
        !saem.warnings.iter().any(|w| w.contains("eigenvalue floor")),
        "R is positive definite here, so nothing was floored: {:?}",
        saem.warnings
    );
    let printed = printed_decrement(w[0]);
    let far_m2 = mahalanobis_sq(&at_min.matrix, &x_hat, &x_far);
    eprintln!("far: printed decrement {printed}, Mahalanobis² {far_m2}");
    assert!(
        (printed / far_m2 - 1.0).abs() < 0.15 && printed > NON_STATIONARY_DECREMENT_OFV,
        "decrement {printed} vs Mahalanobis² {far_m2}"
    );

    // VI is on the same side of the gate, and is the one sampler whose objective carries the
    // covariate-NN penalty (the step hands `cov_objective_gradient` its regularizer; inert here).
    let (_, vi) = cov(&model, &pop, M::Vi);
    let w_vi = stationarity_warnings(&vi);
    assert_eq!(w_vi.len(), 1, "VI: {:?}", vi.warnings);
    assert!(
        w_vi[0].contains("the VI estimate is not a stationary point"),
        "{}",
        w_vi[0]
    );
    assert_eq!(
        printed_decrement(w_vi[0]),
        printed,
        "same point, same objective"
    );

    let (_, focei) = cov(&model, &pop, M::FoceI);
    assert!(
        stationarity_warnings(&focei).is_empty(),
        "FOCEI minimised this objective: {:?}",
        focei.warnings
    );
    assert_eq!(focei.matrix, saem.matrix, "the warning changes no number");

    // The FD leg: under `gradient = fd` the exact reporting gradient declines and
    // `reporting_fd_gradient` supplies `g` (and the stencil supplies `R`). It must read the same
    // decrement, so neither leg can carry a scale the other does not.
    let mut model = model;
    model.gradient_method = crate::types::GradientMethod::Fd;
    let (_, saem_fd) = cov(&model, &pop, M::Saem);
    let w_fd = stationarity_warnings(&saem_fd);
    assert_eq!(w_fd.len(), 1, "FD leg: {:?}", saem_fd.warnings);
    let printed_fd = printed_decrement(w_fd[0]);
    eprintln!("far, FD: printed decrement {printed_fd}");
    assert!(
        (printed_fd / printed - 1.0).abs() < 0.02,
        "FD {printed_fd} vs exact {printed}"
    );

    // Near: under the threshold, no warning — and close enough to it that a decrement doubled
    // by the R / H_ofv factor would cross it.
    let (model, pop) = warfarin_at(TVCL_HAT * 1.04);
    let (x_near, saem_near) = cov(&model, &pop, M::Saem);
    let near_m2 = mahalanobis_sq(&at_min.matrix, &x_hat, &x_near);
    eprintln!("near: Mahalanobis² {near_m2}");
    assert!(
        near_m2 < NON_STATIONARY_DECREMENT_OFV && 2.0 * near_m2 > NON_STATIONARY_DECREMENT_OFV,
        "fixture: the near point must straddle the threshold under a factor of 2: {near_m2}"
    );
    assert!(
        stationarity_warnings(&saem_near).is_empty(),
        "{:?}",
        saem_near.warnings
    );
}

fn optimum_warnings(out: &CovarianceOutput) -> Vec<&String> {
    out.warnings
        .iter()
        .filter(|w| w.starts_with(NON_STATIONARY_OPTIMUM_TOKEN))
        .collect()
}

/// #2785: the same test on an estimate whose method *did* minimise the objective — FOCEI —
/// which demotes the fit instead of only warning. Three straddles in one test:
///
/// - **distance**: FOCEI at the minimum reads a decrement far under the threshold and is not
///   flagged; at `TVCL × 1.06` (the #2541 `far` point, `R` still positive definite) it is,
///   with one `W_NON_STATIONARY_OPTIMUM` naming `TVCL` first;
/// - **the same number as the sampler's test**: at that point, with nothing on a bound, the
///   decrement is SAEM's `W_COV_NON_STATIONARY_ESTIMATE` decrement to the bit, so the two
///   gates cannot drift apart in scale;
/// - **bound**: the same point with `TVCL`'s lower bound moved onto it — a constrained optimum
///   whose slope points out of the box — holds `TVCL`, and the decrement of what is left falls
///   under the threshold. Without the hold it would be the unconstrained one again.
///
/// Measured (debug, Linux x86_64): 1.9e-8 at the minimum, 1.2667 at `far` (SAEM's own
/// printed 1.267), 0.0629 with `TVCL` held.
#[test]
fn a_non_stationary_focei_estimate_is_demoted_unless_a_bound_holds_it() {
    let (model, pop) = warfarin_at(TVCL_HAT);
    let (_, at_min) = cov(&model, &pop, M::FoceI);
    let d_min = at_min
        .newton_decrement
        .expect("FOCEI on an unclipped R is tested");
    eprintln!("at the minimum: decrement {d_min:e}");
    assert!(d_min.is_finite() && d_min < 1e-3, "{d_min}");
    assert!(!at_min.non_stationary_optimum);
    assert!(
        optimum_warnings(&at_min).is_empty(),
        "{:?}",
        at_min.warnings
    );

    let (model, pop) = warfarin_at(TVCL_HAT * 1.06);
    let (_, focei) = cov(&model, &pop, M::FoceI);
    let d_far = focei.newton_decrement.expect("tested");
    eprintln!("far: decrement {d_far}");
    assert!(focei.non_stationary_optimum, "decrement {d_far}");
    let w = optimum_warnings(&focei);
    assert_eq!(w.len(), 1, "{:?}", focei.warnings);
    assert!(
        w[0].contains("the FOCEI estimate is not a stationary point")
            && w[0].contains("in standard errors: theta[TVCL] -"),
        "TVCL is the displaced coordinate, and a minimiser lowers it: {}",
        w[0]
    );
    // The sampler's gate, not this one, owns `W_COV_NON_STATIONARY_ESTIMATE`.
    assert!(
        stationarity_warnings(&focei).is_empty(),
        "{:?}",
        focei.warnings
    );
    let (_, saem) = cov(&model, &pop, M::Saem);
    assert!(
        !saem.non_stationary_optimum,
        "a sampler is warned, never demoted"
    );
    assert_eq!(
        saem.newton_decrement,
        Some(d_far),
        "same point, same objective"
    );

    let (model, pop) = warfarin_at_bounded(TVCL_HAT * 1.06, TVCL_HAT * 1.06);
    let (_, held) = cov(&model, &pop, M::FoceI);
    let d_held = held.newton_decrement.expect("tested");
    eprintln!("far, TVCL on its lower bound: decrement {d_held}");
    assert!(d_held < NON_STATIONARY_DECREMENT_OFV, "{d_held}");
    assert!(!held.non_stationary_optimum);
    assert!(optimum_warnings(&held).is_empty(), "{:?}", held.warnings);
}

#[test]
fn a_non_stationary_optimum_demotes_converged_once() {
    use super::demote_non_stationary_optimum as demote;
    let (mut converged, mut warnings) = (true, Vec::new());
    demote(false, &mut converged, &mut warnings);
    assert!(converged && warnings.is_empty(), "nothing to demote");
    demote(true, &mut converged, &mut warnings);
    assert!(!converged);
    assert_eq!(
        warnings,
        vec!["Outer optimization did not converge".to_string()]
    );
    // Already unconverged: the generic line is not pushed a second time.
    demote(true, &mut converged, &mut warnings);
    assert_eq!(warnings.len(), 1);
}

/// `examples/warfarin.ferx` fitted end to end by the built-in BFGS (`Optimizer::Bfgs`), which
/// stops on a raw gradient norm `|g| < outer_gtol` (#2785).
fn bfgs_fit(outer_gtol: f64) -> crate::types::FitResult {
    let (model, pop) = warfarin_at(0.2);
    let o = FitOptions {
        optimizer: crate::types::Optimizer::Bfgs,
        outer_gtol,
        ..opts(M::FoceI)
    };
    crate::api::fit(&model, &pop, &model.default_params, &o).expect("fit")
}

/// #2785's shareable fixture: a stop rule that is met at a point that is not a minimum, and
/// was reported `converged: true` before the stationarity gate. The built-in BFGS stops when
/// the **raw** scaled-space gradient norm drops under `outer_gtol` — the scale-blind measure
/// the issue names — and at `outer_gtol = 100` it does so after 4 iterations, 12.9 OFV above
/// the minimum (−273.08 against −286.00). The gate is the only thing that
/// demotes it: the warning set below carries no other non-convergence reason, so removing the
/// demotion reports `converged: true` and reddens this test.
///
/// The control is the same fit at the default `outer_gtol`, which runs to the minimum and is
/// left converged with a decrement far under the threshold — the straddle that keeps a gate
/// stuck on "demote" from passing.
///
/// Measured (debug, Linux x86_64), sweeping `outer_gtol`: 30 and 50 stop 0.36 / 0.49 OFV
/// short with decrements of 0.36 / 0.47 — under one standard error, so left converged, as the
/// threshold intends; 80 and 100 stop at the same point with a decrement of 337.4 (√ ≈ 18
/// SEs); 200 stops after 2 iterations where `R` needs an eigenvalue floor, so the step makes
/// no claim (`newton_decrement = None`). The default `outer_gtol` reads 1.9e-8 at the minimum.
#[test]
fn a_bfgs_stop_on_a_raw_gradient_norm_short_of_the_minimum_is_not_converged() {
    let loose = bfgs_fit(100.0);
    let d = loose
        .newton_decrement
        .expect("FOCEI, covariance step on an unclipped R");
    eprintln!(
        "outer_gtol = 100: decrement {d:e}, {} iterations",
        loose.n_iterations
    );
    assert!(
        d > NON_STATIONARY_DECREMENT_OFV,
        "fixture: stops short ({d})"
    );
    assert!(!loose.converged, "{:?}", loose.warnings);
    let tokens: Vec<_> = loose
        .warnings
        .iter()
        .filter(|w| w.contains(NON_STATIONARY_OPTIMUM_TOKEN))
        .collect();
    assert_eq!(tokens.len(), 1, "{:?}", loose.warnings);
    assert!(tokens[0].contains("the FOCEI estimate"), "{}", tokens[0]);
    assert_eq!(
        loose
            .warnings
            .iter()
            .filter(|w| w.as_str() == "Outer optimization did not converge")
            .count(),
        1,
        "the generic line, once: {:?}",
        loose.warnings
    );
    assert!(
        loose.warnings_structured.iter().any(|e| {
            e.message.contains(NON_STATIONARY_OPTIMUM_TOKEN)
                && e.category == crate::types::WarningCode::Convergence
        }),
        "{:?}",
        loose.warnings_structured
    );
    // The built-in BFGS is not NLopt: no NLopt state to publish.
    assert_eq!(loose.optimizer_stop, None);

    let tight = bfgs_fit(FitOptions::default().outer_gtol);
    let d = tight.newton_decrement.expect("tested");
    eprintln!("default outer_gtol: decrement {d:e}");
    assert!(d < 1e-3, "{d}");
    assert!(tight.converged, "{:?}", tight.warnings);
    assert!(
        !tight
            .warnings
            .iter()
            .any(|w| w.contains(NON_STATIONARY_OPTIMUM_TOKEN)),
        "{:?}",
        tight.warnings
    );
}
