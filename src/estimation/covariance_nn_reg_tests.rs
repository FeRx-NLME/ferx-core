//! The covariate-NN regularizer's curvature in the covariance step (#1352).
//!
//! The optimizer minimises `OFV_data + penalty(θ)`; the reported SEs must be that objective's
//! curvature. The penalty is added to `R` post-assembly, so on a fixed estimate two steps at
//! different λ share every objective evaluation and differ by the penalty Hessians alone:
//! `H_b − H_a = ∇²penalty_b − ∇²penalty_a`, where `H = 2·R⁻¹` is the OFV-scale Hessian the
//! step inverts. For the L2 term that is exactly `2(λ_b − λ_a)` on each weight-matrix
//! coordinate and zero elsewhere — the closed form, read off the returned covariance, on each
//! route the R matrix is built by.
//!
//! The fixture's point is the data-generating one, not an optimum, and at λ = 0 its data
//! Hessian is indefinite along a weight direction (measured: min eig −2.35 of max 735). That
//! is the regime the issue is about — the step floors it — so the identities are taken between
//! two λ > 0 steps, both of which the penalty makes positive definite, and the λ = 0 floor is
//! asserted as the straddle.
use super::{compute_covariance_for, reconverge_population, CovarianceStepResult};
use crate::estimation::parameterization::{pack_params, pack_with_bounds};
use crate::parser::model_parser::parse_model_string;
use crate::types::{
    CompiledModel, CovarianceMethod, DoseEvent, EstimationMethod, FitOptions, ModelParameters,
    Population, Subject,
};
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

/// One-cpt oral model whose typical CL is a 1→1→1 covariate network on `WT`: four NN thetas
/// (`W1, b1, W2, b2`), of which `W1` and `W2` carry the L2 penalty.
const DCM: &str = r#"
[parameters]
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP_ERR ~ 0.04

[covariate_nn TVCL_NN]
  inputs     = [WT]
  outputs    = [CL]
  layers     = [1]
  activation = tanh
  output     = softplus
  center     = [70]
  scale      = [12]

[individual_parameters]
  CL = TVCL_NN.CL * exp(ETA_CL)
  V  = TVV * exp(ETA_V)
  KA = TVKA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// The model, its parameters at the data-generating point, and a population simulated there
/// (deterministic per-subject η and a residual pattern, so no subject's term copies another's).
fn fixture() -> (CompiledModel, ModelParameters, Population) {
    let model = parse_model_string(DCM).expect("DCM parses with --features nn");
    let nn = &model.covariate_nns[0];
    assert_eq!(nn.mapper.mlp().n_weights(), 4, "1→1→1 network");
    let mut params = model.default_params.clone();
    let off = nn.weights_offset;
    params.theta[off..off + 4].copy_from_slice(&[0.8, 0.1, 0.6, 1.5]);
    let times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let subjects = (0..16)
        .map(|k| {
            let wt = 45.0 + 50.0 * k as f64 / 15.0;
            let mut s = Subject {
                id: format!("nnreg-{k}"),
                doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                obs_times: times.to_vec(),
                observations: vec![0.0; times.len()],
                obs_cmts: vec![1; times.len()],
                covariates: HashMap::from([("WT".to_string(), wt)]),
                cens: vec![0; times.len()],
                occasions: vec![1; times.len()],
                ..Subject::default()
            };
            let eta = [0.25 * (k as f64 * 1.7).sin(), 0.15 * (k as f64 * 0.9).cos()];
            let preds = crate::pk::compute_predictions_with_tv(&model, &s, &params.theta, &eta);
            s.observations = preds
                .iter()
                .enumerate()
                .map(|(j, p)| p * (1.0 + 0.12 * ((k * 7 + j * 3) as f64).sin()))
                .collect();
            s
        })
        .collect();
    let population = Population {
        subjects,
        covariate_names: vec!["WT".to_string()],
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    (model, params, population)
}

fn options(l2: f64, smooth: f64, analytic: bool, method: CovarianceMethod) -> FitOptions {
    FitOptions {
        method: EstimationMethod::FoceI,
        interaction: true,
        analytic_cov_hessian: analytic,
        covariance_method: method,
        nn_l2_lambda: l2,
        nn_smooth_lambda: smooth,
        verbose: false,
        ..FitOptions::default()
    }
}

/// `(covariance, warnings)` of the step at the fixture's point, for the estimate's own
/// `fit_method`.
fn step(
    fit_method: EstimationMethod,
    (model, params, population): &(CompiledModel, ModelParameters, Population),
    o: &FitOptions,
) -> (DMatrix<f64>, Vec<String>) {
    let x_hat = pack_params(params);
    // The fit's modes and `H` matrices at this point, as a finalizer would hand them over —
    // `S` reads `H` directly.
    let zeros = vec![DVector::zeros(model.n_eta); population.subjects.len()];
    let (_, etas, h, kappas) =
        reconverge_population(&x_hat, model, population, params, &zeros, o, 1e-10, true);
    match compute_covariance_for(
        fit_method, &x_hat, params, model, population, &etas, &h, &kappas, o,
    ) {
        CovarianceStepResult::Success(out) => (out.matrix, out.warnings),
        CovarianceStepResult::Unusable(m) => panic!("{m}"),
        CovarianceStepResult::FailedNonPd { reason, .. } => panic!("{reason}"),
    }
}

fn free_idx(params: &ModelParameters) -> Vec<usize> {
    let packed = pack_with_bounds(params);
    (0..packed.fixed.len())
        .filter(|&i| !packed.fixed[i])
        .collect()
}

/// The OFV-scale Hessian `H = 2·R⁻¹` over the free block.
fn ofv_hessian(cov: &DMatrix<f64>, free: &[usize]) -> DMatrix<f64> {
    let m = DMatrix::from_fn(free.len(), free.len(), |a, b| cov[(free[a], free[b])]);
    m.try_inverse().expect("covariance invertible") * 2.0
}

/// The L2 closed form, per route: between `λ_a` and `λ_b`, `H_b − H_a` is `2(λ_b − λ_a)` on
/// the two weight coordinates and zero on every other entry, so each SE at `λ_b` is the
/// closed-form inverse of that diagonal update of `H_a` — for a lone penalized coordinate the
/// Sherman–Morrison `SE_b/SE_a = 1/√(1 + 2Δλ/h_a)`, `h_a = 1/[H_a⁻¹]_ww`. Deleting the
/// curvature hook in `compute_covariance_for` reddens both routes, each naming itself: the
/// λ > 0 steps then floor exactly as λ = 0 does. Neither λ > 0 step may floor, or the inverse
/// would not be the matrix the identity is about; the λ = 0 step must, or the fixture would
/// not be the shape of fit the penalty is for.
#[test]
fn the_l2_penalty_adds_exactly_two_lambda_to_the_weight_curvature_on_both_routes() {
    let fx = fixture();
    let (model, params, _) = &fx;
    let free = free_idx(params);
    let nn = &model.covariate_nns[0];
    let weight_coords: Vec<usize> = nn
        .mapper
        .mlp()
        .weight_param_indices()
        .into_iter()
        .map(|i| nn.weights_offset + i)
        .collect();
    assert_eq!(weight_coords.len(), 2, "W1 and W2");
    let (lambda_a, lambda_b) = (5.0, 12.0);
    for analytic in [false, true] {
        let route = if analytic { "analytic R" } else { "FD stencil" };
        let run = |l2| {
            step(
                EstimationMethod::FoceI,
                &fx,
                &options(l2, 0.0, analytic, CovarianceMethod::Hessian),
            )
        };
        let (_, w0) = run(0.0);
        assert!(
            w0.iter().any(|w| w.contains("eigenvalue floor")),
            "{route}: the unpenalized Hessian must be indefinite here: {w0:?}"
        );
        assert!(
            !w0.iter().any(|w| w.contains("W_COV_NN_PENALTY")),
            "{route}: λ = 0 gets no penalty note: {w0:?}"
        );
        let (cov_a, wa) = run(lambda_a);
        let (cov_b, wb) = run(lambda_b);
        for w in wa.iter().chain(&wb) {
            assert!(
                !w.contains("eigenvalue floor"),
                "{route}: the penalized step floored — is the penalty in R? {w}"
            );
        }
        assert!(
            wb.iter()
                .any(|w| w.starts_with("W_COV_NN_PENALTY")
                    && w.contains("penalized-ML standard errors")),
            "{route}: the SEs must be named as penalized: {wb:?}"
        );
        let (ha, hb) = (ofv_hessian(&cov_a, &free), ofv_hessian(&cov_b, &free));
        let scale = ha.amax();
        let mut worst = 0.0f64;
        for a in 0..free.len() {
            for b in 0..free.len() {
                let want = if a == b && weight_coords.contains(&free[a]) {
                    2.0 * (lambda_b - lambda_a)
                } else {
                    0.0
                };
                let err = (hb[(a, b)] - ha[(a, b)] - want).abs();
                assert!(err.is_finite(), "{route}: non-finite at ({a}, {b})");
                worst = worst.max(err);
            }
        }
        eprintln!("{route}: worst |ΔH − 2ΔλD| = {worst:e} (max |H_a| = {scale:e})");
        // Measured: 2.3e-13 (FD stencil) and 6.8e-13 (analytic R) against max |H_a| ≈ 653 —
        // inversion round-off, ~1e-15 relative. The bound is ~1,000× above that and ~1e10×
        // below the `2Δλ = 14` a missing hook would leave.
        assert!(
            worst < 1e-12 * scale,
            "{route}: H_b − H_a is not the L2 penalty Hessian (worst {worst:e})"
        );
        // The SE consequence, against the closed form built from the λ_a matrix alone.
        let mut h_want = ha.clone();
        for (a, &i) in free.iter().enumerate() {
            if weight_coords.contains(&i) {
                h_want[(a, a)] += 2.0 * (lambda_b - lambda_a);
            }
        }
        let cov_want = h_want.try_inverse().expect("PD") * 2.0;
        for (a, &i) in free.iter().enumerate() {
            let (se_a, se_b, se_want) = (
                cov_a[(i, i)].sqrt(),
                cov_b[(i, i)].sqrt(),
                cov_want[(a, a)].sqrt(),
            );
            assert!(
                (se_b / se_want - 1.0).abs() < 1e-8,
                "{route}: SE of coordinate {i}: {se_b} vs closed form {se_want}"
            );
            if weight_coords.contains(&i) {
                assert!(
                    se_b < se_a,
                    "{route}: a more penalized weight's SE must shrink"
                );
            }
        }
    }
}

/// With `nn_smooth` on, the step adds the penalty's **exact** Hessian (not the optimizer's
/// Gauss–Newton one): against an L2-only twin at the same `nn_l2`, `H_b − H_a` equals the
/// smoothness term's `add_exact_packed_hessian` entry for entry. That the exact form is the
/// true second derivative is pinned in `nn::regularizer_fit_tests`; this pins that the
/// covariance step consumes it, and the asserted Gauss–Newton gap is what lets a revert to
/// `add_packed_hessian` redden it.
#[test]
fn the_smoothness_penalty_adds_its_exact_hessian() {
    let fx = fixture();
    let (model, params, population) = &fx;
    let free = free_idx(params);
    let l2 = 5.0;
    let smooth = 3.0;
    let o_a = options(l2, 0.0, false, CovarianceMethod::Hessian);
    let o_b = options(l2, smooth, false, CovarianceMethod::Hessian);
    let (cov_a, wa) = step(EstimationMethod::FoceI, &fx, &o_a);
    let (cov_b, wb) = step(EstimationMethod::FoceI, &fx, &o_b);
    for w in wa.iter().chain(&wb) {
        assert!(!w.contains("eigenvalue floor"), "the step floored: {w}");
    }
    let n = pack_params(params).len();
    let smooth_only = options(0.0, smooth, false, CovarianceMethod::Hessian);
    let reg = crate::estimation::nn_reg::NnRegularizer::build(model, population, &smooth_only);
    let mut pen = DMatrix::<f64>::zeros(n, n);
    reg.add_exact_packed_hessian(&params.theta, &mut |i, j, v| pen[(i, j)] += v);
    let mut gn = DMatrix::<f64>::zeros(n, n);
    reg.add_packed_hessian(&params.theta, &mut |i, j, v| gn[(i, j)] += v);
    let (ha, hb) = (ofv_hessian(&cov_a, &free), ofv_hessian(&cov_b, &free));
    let scale = ha.amax();
    let (mut worst, mut gn_gap) = (0.0f64, 0.0f64);
    for (a, &i) in free.iter().enumerate() {
        for (b, &j) in free.iter().enumerate() {
            let d = hb[(a, b)] - ha[(a, b)];
            assert!(d.is_finite(), "non-finite at ({i}, {j})");
            worst = worst.max((d - pen[(i, j)]).abs());
            gn_gap = gn_gap.max((d - gn[(i, j)]).abs());
        }
    }
    eprintln!(
        "smooth: worst |ΔH − exact| = {worst:e}, |ΔH − GN| = {gn_gap:e} (max |H_a| = {scale:e})"
    );
    // Measured: 6.3e-13 against max |H_a| ≈ 653, and a Gauss–Newton gap of 8.0e-2 — the
    // bound sits ~1,000× above the first and ~1e8× below the second.
    assert!(
        worst < 1e-12 * scale,
        "H_b − H_a is not the exact smoothness Hessian (worst {worst:e})"
    );
    assert!(
        gn_gap > 1e3 * worst.max(1e-12),
        "the fixture must tell the exact Hessian from Gauss–Newton (gap {gn_gap:e})"
    );
}

/// The penalty is added only when the estimator that produced the point minimised it: the
/// same inputs scored as a SAEM estimate (which never applies `nn_l2`) are bit-identical to
/// λ = 0 and carry no note, while the FOCEI-scored step differs. Both sides in one test, so a
/// gate stuck either way reddens it.
#[test]
fn the_penalty_curvature_is_gated_on_the_estimator_that_applied_it() {
    let fx = fixture();
    let o0 = options(0.0, 0.0, false, CovarianceMethod::Hessian);
    let o1 = options(0.5, 0.0, false, CovarianceMethod::Hessian);
    let (cov0, _) = step(EstimationMethod::Saem, &fx, &o0);
    let (saem, w_saem) = step(EstimationMethod::Saem, &fx, &o1);
    let (focei, _) = step(EstimationMethod::FoceI, &fx, &o1);
    assert_eq!(
        saem, cov0,
        "SAEM never applied the penalty, so R must not carry it"
    );
    assert!(
        !w_saem.iter().any(|w| w.contains("W_COV_NN_PENALTY")),
        "no penalty note on an unregularized estimate: {w_saem:?}"
    );
    assert_ne!(focei, cov0, "FOCEI applied the penalty, so R must carry it");
}

/// `S` has no per-subject part for the penalty to live in, so `covariance_method = s` says its
/// SEs ignore it (and the sandwich names its `R` as the penalized half). The `s` matrix itself
/// is unchanged by λ — it never reads `R`.
#[test]
fn the_score_estimators_say_what_they_do_with_the_penalty() {
    let fx = fixture();
    let (s0, w0) = step(
        EstimationMethod::FoceI,
        &fx,
        &options(0.0, 0.0, false, CovarianceMethod::CrossProduct),
    );
    let (s1, w1) = step(
        EstimationMethod::FoceI,
        &fx,
        &options(0.5, 0.0, false, CovarianceMethod::CrossProduct),
    );
    assert_eq!(s0, s1, "S⁻¹ cannot depend on the penalty");
    assert!(w0.is_empty(), "{w0:?}");
    assert!(
        w1.iter()
            .any(|w| w.contains("not represented") && w.contains("covariance_method = r")),
        "{w1:?}"
    );
    let (_, ws) = step(
        EstimationMethod::FoceI,
        &fx,
        &options(0.5, 0.0, false, CovarianceMethod::Sandwich),
    );
    assert!(
        ws.iter().any(|w| w.contains("penalized-ML sandwich")),
        "{ws:?}"
    );
}

/// A regularized fit, run briefly with its inline covariance step.
fn regularized_fit(
    fx: &(CompiledModel, ModelParameters, Population),
    l2: f64,
) -> crate::types::FitResult {
    let (model, params, population) = fx;
    let o = FitOptions {
        outer_maxiter: 3,
        run_covariance_step: true,
        optimizer: crate::types::Optimizer::Bobyqa,
        ..options(l2, 0.0, false, CovarianceMethod::Hessian)
    };
    crate::fit(model, population, params, &o).expect("fit runs")
}

/// `run_covariance` on a regularized fit, with the caller's default options, differentiates
/// the objective the fit minimised (#1352 review): the stage record carries `nn_l2`, so the
/// post-hoc matrix is the inline one. The straddle is a record written before the strengths
/// were recorded (`None`): the same call then takes the caller's λ = 0 and the matrix moves —
/// so the parity is the record's doing, not a coincidence of the fixture. And a caller's
/// non-zero λ on that regularized fit does not change it: a recorded strength wins.
#[test]
fn run_covariance_restores_the_fitted_penalty_strengths() {
    let fx = fixture();
    let (model, _, population) = &fx;
    let fit = regularized_fit(&fx, 5.0);
    let rec = fit
        .scoring_settings
        .as_ref()
        .expect("a fit records its scoring settings");
    assert_eq!(rec.nn_l2_lambda, Some(5.0));
    assert_eq!(rec.nn_smooth_lambda, Some(0.0));
    let inline = fit.covariance_matrix.clone().expect("inline covariance");
    let post = |fit: &crate::types::FitResult, o: &FitOptions| {
        crate::run_covariance(fit, Some(model), Some(population), o)
            .expect("run_covariance succeeds")
    };

    // `analytic_cov_hessian` says how the caller asks the step to be computed — not recorded,
    // so passed as the inline step ran it; everything else is the default.
    let step_opts = FitOptions {
        analytic_cov_hessian: false,
        ..FitOptions::default()
    };
    let out = post(&fit, &step_opts);
    assert_eq!(
        out.covariance_matrix.as_ref(),
        Some(&inline),
        "default options must reproduce the inline penalized covariance"
    );
    let caller_l2 = FitOptions {
        nn_l2_lambda: 0.5,
        ..step_opts.clone()
    };
    assert_eq!(
        post(&fit, &caller_l2).covariance_matrix.as_ref(),
        Some(&inline),
        "a recorded strength wins over the caller's"
    );

    let mut legacy = fit.clone();
    let rec = legacy.scoring_settings.as_mut().unwrap();
    rec.nn_l2_lambda = None;
    rec.nn_smooth_lambda = None;
    let unrecorded = post(&legacy, &step_opts);
    assert_ne!(
        unrecorded.covariance_matrix.as_ref(),
        Some(&inline),
        "with no recorded strength the caller's λ = 0 applies"
    );
}

/// The penalty notes classify as covariance-step notes, so `run_covariance` replaces them
/// rather than keeping an `r` note beside the `s` one of the step it reran (#1352 review).
/// Severity follows the sentence: Info where the SEs carry the penalty, Warning where not.
#[test]
fn the_penalty_notes_are_covariance_step_warnings_and_are_replaced() {
    use crate::types::{classify_warning, WarningCode, WarningSeverity};
    for (method, severity) in [
        (CovarianceMethod::Hessian, WarningSeverity::Info),
        (CovarianceMethod::Sandwich, WarningSeverity::Info),
        (CovarianceMethod::CrossProduct, WarningSeverity::Warning),
    ] {
        let w = classify_warning(&super::format_nn_penalty_cov_note(method));
        assert_eq!(w.category, WarningCode::CovarianceStep, "{method:?}");
        assert_eq!(w.severity, severity, "{method:?}");
    }

    let fx = fixture();
    let (model, _, population) = &fx;
    let fit = regularized_fit(&fx, 5.0);
    let count = |ws: &[String], needle: &str| {
        ws.iter()
            .filter(|w| w.starts_with("W_COV_NN_PENALTY") && w.contains(needle))
            .count()
    };
    assert_eq!(count(&fit.warnings, "includes the penalty"), 1);
    let s = FitOptions {
        covariance_method: CovarianceMethod::CrossProduct,
        ..FitOptions::default()
    };
    let out = crate::run_covariance(&fit, Some(model), Some(population), &s).unwrap();
    assert_eq!(
        count(&out.warnings, "includes the penalty"),
        0,
        "the inline r note must not survive a rerun under s: {:?}",
        out.warnings
    );
    assert_eq!(
        count(&out.warnings, "not represented"),
        1,
        "{:?}",
        out.warnings
    );
    let again =
        crate::run_covariance(&fit, Some(model), Some(population), &FitOptions::default()).unwrap();
    assert_eq!(
        count(&again.warnings, "includes the penalty"),
        1,
        "a rerun under r must not duplicate the note: {:?}",
        again.warnings
    );
}

/// #2541 review r1 §4: the stationarity gradient carries the NN penalty exactly when the
/// estimator minimised it, so `g` and `H` are always the same objective's.
///
/// Two halves. Directly on `cov_objective_gradient`: the gradient with the regularizer minus
/// the one without is the penalty's own packed gradient, and the one without at λ = 5 is the
/// λ = 0 gradient to the bit. Through the step: VI (which applies `nn_l2`) at λ = 5 (the
/// L2 test's λ_a above, where `R` is positive definite; at 0.5 it is still indefinite)
/// prints the decrement recomputed here from the *penalized* gradient and the returned `R⁻¹`,
/// and not the one from the unpenalized gradient — so dropping the step's regularizer at the
/// call site reddens it. SAEM never reaches the gradient on this fixture: without
/// the penalty its `R` is indefinite here (the module doc) and the #1781 fallback takes it.
#[test]
fn the_stationarity_gradient_carries_the_penalty_exactly_when_the_estimator_applied_it() {
    use crate::estimation::cov_diagnostics::NON_STATIONARY_TOKEN;
    use crate::estimation::outer_optimizer::cov_objective_gradient;
    let fx = fixture();
    let (model, params, population) = &fx;
    let x = pack_params(params);
    let bounds = pack_with_bounds(params).bounds;
    let o0 = options(0.0, 0.0, true, CovarianceMethod::Hessian);
    let o1 = options(5.0, 0.0, true, CovarianceMethod::Hessian);
    let zeros = vec![DVector::zeros(model.n_eta); population.subjects.len()];
    let (_, etas, h, kappas) =
        reconverge_population(&x, model, population, params, &zeros, &o1, 1e-10, true);
    let reg = crate::estimation::nn_reg::NnRegularizer::build(model, population, &o1);
    assert!(reg.is_active());
    let grad = |o: &FitOptions, reg| {
        cov_objective_gradient(
            &x, params, model, population, &etas, &h, &kappas, &bounds, o, reg,
        )
    };
    let unpenalized = grad(&o1, None);
    assert_eq!(
        unpenalized,
        grad(&o0, None),
        "λ must not reach an unpenalized g"
    );
    let penalized = grad(&o1, Some(&reg));
    let mut penalty = vec![0.0; x.len()];
    reg.add_packed_gradient(&params.theta, &mut penalty);
    assert!(
        penalty.iter().map(|v| v.abs()).fold(0.0, f64::max) > 1e-3,
        "fixture: the penalty gradient must be live: {penalty:?}"
    );
    for k in 0..x.len() {
        let gap = penalized[k] - unpenalized[k] - penalty[k];
        assert!(
            gap.abs() <= 1e-9 * (1.0 + penalty[k].abs()),
            "coordinate {k}: penalized − unpenalized − penalty = {gap:e}"
        );
    }

    // Through the step, under VI.
    let (cov, warnings) = step(EstimationMethod::Vi, &fx, &o1);
    let w: Vec<&String> = warnings
        .iter()
        .filter(|w| w.starts_with(NON_STATIONARY_TOKEN))
        .collect();
    assert_eq!(w.len(), 1, "the fixture is not an optimum: {warnings:?}");
    assert!(!w[0].contains("eigenvalue floor"), "R kept: {}", w[0]);
    let printed: f64 = w[0]
        .split("by an estimated ")
        .nth(1)
        .and_then(|t| t.split(" OFV units").next())
        .and_then(|t| t.parse().ok())
        .expect("the decrement clause");
    let free = free_idx(params);
    let h_inv = DMatrix::from_fn(free.len(), free.len(), |a, b| cov[(free[a], free[b])] / 2.0);
    let decrement = |g: &[f64]| {
        let gf = DVector::from_iterator(free.len(), free.iter().map(|&i| g[i]));
        0.5 * gf.dot(&(&h_inv * &gf))
    };
    let (right, wrong) = (decrement(&penalized), decrement(&unpenalized));
    eprintln!("VI decrement: printed {printed}, penalized g {right}, unpenalized g {wrong}");
    assert!(
        (printed - right).abs() <= 1e-3 * right.max(1.0),
        "printed {printed} vs the penalized gradient's {right}"
    );
    assert!(
        (printed - wrong).abs() > 1e-2 * wrong.max(1.0),
        "fixture: the two flags must print different decrements ({right} vs {wrong})"
    );
}
