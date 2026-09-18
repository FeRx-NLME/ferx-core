//! Tests for the analytic `[covariate_nn]` weight-θ gradient.
//!
//! The anchor throughout is **central** finite differences of
//! `obs_nll_subject_into` — an evaluator that knows nothing about the
//! decomposition under test. That matters: the hybrid path and the per-θ FD
//! path share `d_nll_d_f` and the prediction solver, so checking one against
//! the other would only confirm they agree about the chain rule, not that
//! either is right. Central FD of the raw objective is external to both.
//!
//! Central FD is also *more* accurate than the forward FD this work replaces,
//! so `hybrid_is_closer_to_central_fd_than_forward_fd` can assert the direction
//! of the change rather than merely bounding the disagreement.

use super::*;
use crate::estimation::fixed_eta_gradient::{
    obs_nll_subject_grad, obs_nll_subject_grad_iov, obs_nll_subject_into_iov,
};
use crate::parser::model_parser::parse_model_string;
use crate::pk::EventPkParams;
use crate::stats::likelihood::obs_nll_subject_into;
use crate::types::{CompiledModel, DoseEvent, Subject};
use std::collections::HashMap;

/// A DCM small enough for a unit test but structurally identical to
/// `examples/warfarin_dcm.ferx`: `tanh` hidden layer, `softplus` output head,
/// etas composed on top of the NN outputs.
///
/// 2 inputs → 3 hidden → 2 outputs is `2·3+3 + 3·2+2 = 17` weights, so the
/// hybrid does 2 solves where the per-θ loop does 17 — enough separation that
/// a bug in the routing shows up as a numeric disagreement rather than as
/// coincidentally-equal answers.
fn dcm_model_src() -> String {
    r#"
[parameters]
  theta TVKA(1.0, 0.001, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP ~ 0.04 (sd)

[covariate_nn TYPICAL_PK]
  inputs = [WT, CRCL]
  outputs = [CL, V]
  layers = [3]
  activation = tanh
  output = softplus
  # Without normalisation the raw covariates (WT ≈ 72, CRCL ≈ 95) saturate the
  # tanh layer outright: every hidden unit pins at ±1, tanh' underflows to 0,
  # and the whole first layer's gradient is *exactly* zero. A parity test on
  # that model would confirm nothing about the first layer. See the
  # `NamedMlpMapper::input_scale` docs for the same pathology in a real fit.
  center = [70, 90]
  scale  = [15, 30]

[individual_parameters]
  CL = TYPICAL_PK.CL * exp(ETA_CL)
  V  = TYPICAL_PK.V  * exp(ETA_V)
  KA = TVKA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP)
"#
    .to_string()
}

/// Subject with static covariates — the case the hybrid serves.
fn static_subject() -> Subject {
    let mut cov = HashMap::new();
    cov.insert("WT".to_string(), 72.0);
    cov.insert("CRCL".to_string(), 95.0);
    Subject {
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![1.0, 4.0, 8.0, 12.0],
        obs_raw_times: Vec::new(),
        observations: vec![7.5, 6.0, 3.8, 2.1],
        obs_cmts: vec![1, 1, 1, 1],
        covariates: cov,
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        reset_occasions: Vec::new(),
        cens: vec![0, 0, 0, 0],
        occasions: vec![],
        obs_l2: Vec::new(),
        dose_occasions: vec![],
        fremtype: Vec::new(),
        obs_records: vec![],
    }
}

/// A probe point that is both non-degenerate and *physiologically sane*.
///
/// The parsed defaults leave every bias at 0, so `softplus` emits `CL ≈ V ≈
/// 0.7` and a 100 mg dose predicts concentrations two orders of magnitude above
/// the observations. Gradients there run to ~1e7 and are dominated by
/// finite-difference truncation, which makes a parity test measure the
/// reference's error rather than the estimator's. Shifting the output biases to
/// `CL ≈ 1`, `V ≈ 20` puts predictions alongside the data; the small
/// per-weight jitter keeps every hidden unit off a symmetric point.
fn probe_theta(model: &CompiledModel) -> Vec<f64> {
    let mut theta: Vec<f64> = model
        .default_params
        .theta
        .iter()
        .enumerate()
        .map(|(i, &t)| t + 0.13 * ((i as f64) * 0.7).sin())
        .collect();
    for nn in &model.covariate_nns {
        // softplus(z) ≈ z in the linear regime, so the bias is roughly the
        // output value once the (small) hidden contribution is added.
        for (k, target) in [0.55f64, 20.0].iter().enumerate() {
            theta[nn.weights_offset + nn.mapper.mlp().output_bias_index(k)] = *target;
        }
    }
    theta
}

/// The θ indices belonging to the model's NN weight blocks.
fn nn_theta_indices(model: &CompiledModel) -> Vec<usize> {
    model
        .covariate_nns
        .iter()
        .flat_map(|nn| nn.weights_offset..nn.weights_offset + nn.mapper.mlp().n_weights())
        .collect()
}

/// Central FD of `obs_nll_subject_into` in the packed `[log_theta | log_sigma]`
/// space, restricted to the θ block. The reference every parity test compares
/// against.
fn central_fd_theta_grad(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    sigma_values: &[f64],
    eta: &[f64],
    mask: &[bool],
) -> Vec<f64> {
    let mut scratch = EventPkParams::default();
    let mut out = vec![0.0f64; theta.len()];
    for i in 0..theta.len() {
        let h = 1e-6 * (1.0 + theta[i].abs());
        let mut tp = theta.to_vec();
        tp[i] += h;
        let f_plus = obs_nll_subject_into(
            model,
            subject,
            &tp,
            sigma_values,
            &model.residual_correlations,
            eta,
            &mut scratch,
        );
        tp[i] = theta[i] - h;
        let f_minus = obs_nll_subject_into(
            model,
            subject,
            &tp,
            sigma_values,
            &model.residual_correlations,
            eta,
            &mut scratch,
        );
        let raw = (f_plus - f_minus) / (2.0 * h);
        out[i] = if mask[i] { theta[i] * raw } else { raw };
    }
    out
}

/// Bounds wide enough that nothing is pinned, plus the model's own log-packing
/// mask.
fn unpinned(model: &CompiledModel, n: usize) -> (Vec<bool>, Vec<f64>, Vec<f64>) {
    let mask: Vec<bool> = model
        .default_params
        .theta_lower
        .iter()
        .map(|&lo| crate::estimation::parameterization::theta_packs_log(lo))
        .collect();
    (mask, vec![-1e30f64; n], vec![1e30f64; n])
}

fn relative(a: f64, b: f64) -> f64 {
    let scale = a.abs().max(b.abs()).max(1e-8);
    (a - b).abs() / scale
}

/// The headline correctness claim: every NN weight's gradient entry, assembled
/// from 2 solves instead of 17, matches central FD of the objective.
#[test]
fn hybrid_nn_weight_gradient_matches_central_fd() {
    let model = parse_model_string(&dcm_model_src()).expect("DCM model parses");
    let subject = static_subject();
    let theta = probe_theta(&model);
    let sigma_values = vec![0.2f64];
    let eta = vec![0.15f64, -0.1f64];
    let n_theta = theta.len();
    let n_sigma = 1;
    let (mask, lo, hi) = unpinned(&model, n_theta + n_sigma);

    // Guard the premise: without a plan this test would be checking the old
    // FD loop against FD and would pass for the wrong reason.
    assert!(
        NnGradPlan::build(&model, &subject, &theta, n_theta).is_some(),
        "static-covariate DCM subject must be served by the hybrid path"
    );

    let mut scratch = EventPkParams::default();
    let (_nll, grad) = obs_nll_subject_grad(
        &model,
        &subject,
        &theta,
        &sigma_values,
        &eta,
        &mask,
        &lo,
        &hi,
        n_theta,
        n_sigma,
        &mut scratch,
    );
    let reference = central_fd_theta_grad(&model, &subject, &theta, &sigma_values, &eta, &mask);

    let nn_idx = nn_theta_indices(&model);
    assert_eq!(nn_idx.len(), 17, "unexpected NN weight count");
    let peak = nn_idx
        .iter()
        .map(|&i| reference[i].abs())
        .fold(0.0f64, f64::max);
    let mut n_informative = 0;
    for &i in &nn_idx {
        if reference[i].abs() > 1e-3 * peak {
            n_informative += 1;
        }
        // Observed 1e-11…7e-7, limited by the *reference's* own truncation on
        // the smallest entries. 1e-5 keeps headroom while still discriminating:
        // the superseded forward-FD-per-weight estimator lands at ~2e-5 here
        // and fails this bound.
        assert!(
            relative(grad[i], reference[i]) < 1e-5,
            "NN weight theta[{i}]: hybrid={:.8e}, central FD={:.8e}, rel={:.2e}",
            grad[i],
            reference[i],
            relative(grad[i], reference[i])
        );
    }
    // A network whose weights all had ~zero gradient would satisfy the loop
    // above trivially.
    assert!(
        n_informative >= 12,
        "expected most NN weights to carry signal, got {n_informative}/17"
    );
}

/// The non-NN θ (`TVKA`) and the σ block must come out of the untouched FD
/// path, i.e. the hybrid must not leak into coordinates it does not own.
#[test]
fn non_nn_theta_is_unaffected_by_the_hybrid_path() {
    let model = parse_model_string(&dcm_model_src()).expect("DCM model parses");
    let subject = static_subject();
    let theta = probe_theta(&model);
    let sigma_values = vec![0.2f64];
    let eta = vec![0.15f64, -0.1f64];
    let n_theta = theta.len();
    let (mask, lo, hi) = unpinned(&model, n_theta + 1);

    let plan = NnGradPlan::build(&model, &subject, &theta, n_theta).expect("plan builds");
    let tvka_idx = model
        .default_params
        .theta_names
        .iter()
        .position(|n| n == "TVKA")
        .expect("TVKA present");
    assert!(!plan.covers(tvka_idx), "TVKA must stay on the FD path");

    let mut scratch = EventPkParams::default();
    let (_nll, grad) = obs_nll_subject_grad(
        &model,
        &subject,
        &theta,
        &sigma_values,
        &eta,
        &mask,
        &lo,
        &hi,
        n_theta,
        1,
        &mut scratch,
    );
    let reference = central_fd_theta_grad(&model, &subject, &theta, &sigma_values, &eta, &mask);
    assert!(
        relative(grad[tvka_idx], reference[tvka_idx]) < 1e-4,
        "TVKA: got {:.8e}, central FD {:.8e}",
        grad[tvka_idx],
        reference[tvka_idx]
    );
}

/// The accuracy claim, not just the agreement claim. The hybrid's only
/// remaining FD error is `n_outputs` directional derivatives; every
/// weight-specific factor is exact. So on the weights it must sit closer to
/// central FD than the forward-FD-per-θ loop it replaces.
#[test]
fn hybrid_is_closer_to_central_fd_than_forward_fd() {
    let model = parse_model_string(&dcm_model_src()).expect("DCM model parses");
    let subject = static_subject();
    let theta = probe_theta(&model);
    let sigma_values = vec![0.2f64];
    let eta = vec![0.15f64, -0.1f64];
    let n_theta = theta.len();
    let (mask, lo, hi) = unpinned(&model, n_theta + 1);

    let mut scratch = EventPkParams::default();
    let (_nll, hybrid) = obs_nll_subject_grad(
        &model,
        &subject,
        &theta,
        &sigma_values,
        &eta,
        &mask,
        &lo,
        &hi,
        n_theta,
        1,
        &mut scratch,
    );
    let reference = central_fd_theta_grad(&model, &subject, &theta, &sigma_values, &eta, &mask);

    // The superseded estimator, reproduced here so the comparison is explicit:
    // forward FD at the same 1e-5 relative step the production loop used.
    let mut forward = vec![0.0f64; n_theta];
    let f0 = obs_nll_subject_into(
        &model,
        &subject,
        &theta,
        &sigma_values,
        &model.residual_correlations,
        &eta,
        &mut scratch,
    );
    for i in 0..n_theta {
        let delta = 1e-5 * (1.0 + theta[i].abs());
        let mut tp = theta.clone();
        tp[i] += delta;
        let fp = obs_nll_subject_into(
            &model,
            &subject,
            &tp,
            &sigma_values,
            &model.residual_correlations,
            &eta,
            &mut scratch,
        );
        let raw = (fp - f0) / delta;
        forward[i] = if mask[i] { theta[i] * raw } else { raw };
    }

    let mut hybrid_err = 0.0f64;
    let mut forward_err = 0.0f64;
    for &i in &nn_theta_indices(&model) {
        hybrid_err += (hybrid[i] - reference[i]).powi(2);
        forward_err += (forward[i] - reference[i]).powi(2);
    }
    assert!(
        hybrid_err < forward_err,
        "hybrid should be nearer central FD than forward FD: \
         hybrid SSE {hybrid_err:.3e} vs forward SSE {forward_err:.3e}"
    );
}

/// CLAUDE.md's routing rule: a model outside the analytic scope must fail
/// loudly to FD, not silently return a wrong gradient. A time-varying NN input
/// gives the network a distinct output vector per event, which the
/// single-`z` factorization cannot represent.
#[test]
fn time_varying_nn_input_declines_the_plan_and_still_matches_fd() {
    let model = parse_model_string(&dcm_model_src()).expect("DCM model parses");
    let mut subject = static_subject();
    // CRCL drifts across the observation records; WT stays put.
    subject.obs_covariates = vec![
        HashMap::from([("WT".into(), 72.0), ("CRCL".into(), 95.0)]),
        HashMap::from([("WT".into(), 72.0), ("CRCL".into(), 88.0)]),
        HashMap::from([("WT".into(), 72.0), ("CRCL".into(), 81.0)]),
        HashMap::from([("WT".into(), 72.0), ("CRCL".into(), 77.0)]),
    ];
    subject.dose_covariates = vec![HashMap::from([("WT".into(), 72.0), ("CRCL".into(), 95.0)])];

    let theta = probe_theta(&model);
    let n_theta = theta.len();
    assert!(
        NnGradPlan::build(&model, &subject, &theta, n_theta).is_none(),
        "a time-varying NN input must route to the per-theta FD loop"
    );

    // And the fallback must still be right — declining is only acceptable
    // because the FD path remains correct.
    let sigma_values = vec![0.2f64];
    let eta = vec![0.15f64, -0.1f64];
    let (mask, lo, hi) = unpinned(&model, n_theta + 1);
    let mut scratch = EventPkParams::default();
    let (_nll, grad) = obs_nll_subject_grad(
        &model,
        &subject,
        &theta,
        &sigma_values,
        &eta,
        &mask,
        &lo,
        &hi,
        n_theta,
        1,
        &mut scratch,
    );
    let reference = central_fd_theta_grad(&model, &subject, &theta, &sigma_values, &eta, &mask);
    for &i in &nn_theta_indices(&model) {
        assert!(
            relative(grad[i], reference[i]) < 1e-3,
            "FD fallback theta[{i}]: got {:.8e}, central FD {:.8e}",
            grad[i],
            reference[i]
        );
    }
}

/// A *non*-NN covariate varying in time is irrelevant to the factorization —
/// only the network's own inputs matter. Without this the predicate would be
/// needlessly conservative on any TV-covariate dataset.
#[test]
fn time_varying_non_nn_covariate_keeps_the_plan() {
    let model = parse_model_string(&dcm_model_src()).expect("DCM model parses");
    let mut subject = static_subject();
    let snap = |dose: f64| {
        HashMap::from([
            ("WT".into(), 72.0),
            ("CRCL".into(), 95.0),
            ("CONMED".into(), dose),
        ])
    };
    subject.obs_covariates = vec![snap(0.0), snap(1.0), snap(1.0), snap(0.0)];
    subject.dose_covariates = vec![snap(0.0)];
    subject.covariates = snap(0.0);

    let theta = probe_theta(&model);
    assert!(
        NnGradPlan::build(&model, &subject, &theta, theta.len()).is_some(),
        "CONMED is not an NN input; its variation must not disable the plan"
    );
}

/// A model with no `[covariate_nn]` block must produce no plan at all, so its
/// gradient is computed by byte-identical code to before this change.
#[test]
fn plain_model_has_no_plan() {
    let src = r#"
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(20.0, 0.001, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)
"#;
    let model = parse_model_string(src).expect("plain model parses");
    let subject = static_subject();
    let theta = model.default_params.theta.clone();
    assert!(NnGradPlan::build(&model, &subject, &theta, theta.len()).is_none());
}

/// Pinned weights report zero, and pinning does not corrupt the weights that
/// remain free — including when the pinned coordinate is an output bias, whose
/// finite difference every other weight in the block depends on.
#[test]
fn pinned_output_bias_reports_zero_without_disturbing_free_weights() {
    let model = parse_model_string(&dcm_model_src()).expect("DCM model parses");
    let subject = static_subject();
    let theta = probe_theta(&model);
    let sigma_values = vec![0.2f64];
    let eta = vec![0.15f64, -0.1f64];
    let n_theta = theta.len();
    let n = n_theta + 1;
    let (mask, lo, hi) = unpinned(&model, n);

    let mut scratch = EventPkParams::default();
    let (_nll, free) = obs_nll_subject_grad(
        &model,
        &subject,
        &theta,
        &sigma_values,
        &eta,
        &mask,
        &lo,
        &hi,
        n_theta,
        1,
        &mut scratch,
    );

    let nn = &model.covariate_nns[0];
    let bias0 = nn.weights_offset + nn.mapper.mlp().output_bias_index(0);
    let mut lo_p = lo.clone();
    let mut hi_p = hi.clone();
    lo_p[bias0] = theta[bias0];
    hi_p[bias0] = theta[bias0];

    let (_nll, pinned) = obs_nll_subject_grad(
        &model,
        &subject,
        &theta,
        &sigma_values,
        &eta,
        &mask,
        &lo_p,
        &hi_p,
        n_theta,
        1,
        &mut scratch,
    );

    assert_eq!(pinned[bias0], 0.0, "a pinned theta must report zero");
    for &i in &nn_theta_indices(&model) {
        if i == bias0 {
            continue;
        }
        assert!(
            relative(pinned[i], free[i]) < 1e-12,
            "pinning the output bias changed free weight theta[{i}]: \
             {:.8e} vs {:.8e}",
            pinned[i],
            free[i]
        );
    }
}

/// The IOV entry point shares the decomposition, so it needs its own parity
/// check — the two functions have separate FD loops and could drift apart.
#[test]
fn hybrid_nn_weight_gradient_matches_central_fd_under_iov() {
    let src = dcm_model_src().replace(
        "  CL = TYPICAL_PK.CL * exp(ETA_CL)",
        "  CL = TYPICAL_PK.CL * exp(ETA_CL + KAPPA_CL)",
    );
    let src = src.replace(
        "  omega ETA_V  ~ 0.09",
        "  omega ETA_V  ~ 0.09\n  kappa KAPPA_CL ~ 0.05",
    );
    let model = parse_model_string(&src).expect("IOV DCM model parses");
    assert!(model.n_kappa > 0, "model must carry IOV");

    let mut subject = static_subject();
    subject.occasions = vec![1, 1, 2, 2];
    subject.dose_occasions = vec![1];

    let theta = probe_theta(&model);
    let sigma_values = vec![0.2f64];
    let eta = vec![0.15f64, -0.1f64];
    let kappas = vec![vec![0.08f64], vec![-0.06f64]];
    let n_theta = theta.len();
    let n_sigma = 1;
    let (mask, lo, hi) = unpinned(&model, n_theta + n_sigma);

    assert!(
        NnGradPlan::build(&model, &subject, &theta, n_theta).is_some(),
        "IOV DCM subject must be served by the hybrid path"
    );

    let mut scratch = EventPkParams::default();
    let (_nll, grad) = obs_nll_subject_grad_iov(
        &model,
        &subject,
        &theta,
        &sigma_values,
        &eta,
        &kappas,
        &mask,
        &lo,
        &hi,
        n_theta,
        n_sigma,
        &mut scratch,
    );

    // Central FD of the IOV evaluator, in the same packed space.
    let mut reference = vec![0.0f64; n_theta];
    for i in 0..n_theta {
        let h = 1e-6 * (1.0 + theta[i].abs());
        let mut tp = theta.clone();
        tp[i] += h;
        let f_plus = obs_nll_subject_into_iov(
            &model,
            &subject,
            &tp,
            &sigma_values,
            &eta,
            &kappas,
            &mut scratch,
        );
        tp[i] = theta[i] - h;
        let f_minus = obs_nll_subject_into_iov(
            &model,
            &subject,
            &tp,
            &sigma_values,
            &eta,
            &kappas,
            &mut scratch,
        );
        let raw = (f_plus - f_minus) / (2.0 * h);
        reference[i] = if mask[i] { theta[i] * raw } else { raw };
    }

    for &i in &nn_theta_indices(&model) {
        assert!(
            relative(grad[i], reference[i]) < 1e-5,
            "IOV NN weight theta[{i}]: hybrid={:.8e}, central FD={:.8e}",
            grad[i],
            reference[i]
        );
    }
}

// ---------------------------------------------------------------------------
// DCM + IOV: the eta-only analytic route
// ---------------------------------------------------------------------------

/// A DCM fixture with a per-occasion `κ` on clearance.
fn dcm_iov_model() -> CompiledModel {
    let src = dcm_model_src()
        .replace(
            "  CL = TYPICAL_PK.CL * exp(ETA_CL)",
            "  CL = TYPICAL_PK.CL * exp(ETA_CL + KAPPA_CL)",
        )
        .replace(
            "  omega ETA_V  ~ 0.09",
            "  omega ETA_V  ~ 0.09\n  kappa KAPPA_CL ~ 0.05",
        );
    parse_model_string(&src).expect("DCM+IOV parses")
}

/// A DCM **with IOV** must get its `∂/∂η` from the analytic provider, and that gradient
/// must match central finite differences of the individual NLL across the whole stacked
/// `[η_bsv, κ₁ … κ_K]` vector.
///
/// CLAUDE.md's rule applied to a path that has just become analytic. Before this,
/// `iov_analytical_supported`'s `n_theta_axis() == model.n_theta` clause could never hold
/// for a `[covariate_nn]` model — the program's θ axes cover only the *declared* thetas,
/// never the auto-generated weights — so every DCM+IOV subject fell to finite
/// differences (measured: 60 of 60 on a busulfan-shaped fit).
///
/// The κ columns are the point. A gradient correct on the BSV block and wrong on the
/// per-occasion block would still have the right length and still descend — to the wrong
/// split between between-subject and between-occasion variability.
#[test]
fn dcm_iov_eta_gradient_is_analytic_and_matches_central_fd() {
    use crate::estimation::inner_optimizer::analytic_eta_nll_gradient_iov;
    use crate::stats::likelihood::individual_nll_iov;

    let model = dcm_iov_model();
    assert!(model.n_kappa > 0, "fixture must carry IOV");
    assert!(
        crate::sens::provider::iov_analytical_eta_supported(&model),
        "a DCM+IOV model must be inside the eta-only analytic scope"
    );
    // Since #1339 the full (outer) scope admits a DCM too — the θ-axis clause is on the
    // declared thetas and the weight columns are chained through the network outputs. The
    // inner walk keeps the η-only `Dual1` builder for a DCM regardless
    // (`subject_eta_grad_iov_analytical` pins `eta_only` on `nn_weight_theta_count`), so
    // this test still exercises exactly the route it did before.
    assert!(
        crate::sens::provider::iov_analytical_supported(&model),
        "a DCM+IOV model is inside the full analytic IOV scope as of #1339"
    );

    let mut subject = static_subject();
    subject.occasions = vec![1, 1, 2, 2];
    subject.dose_occasions = vec![1];

    let theta = probe_theta(&model);
    let sigma = vec![0.2f64];
    let omega = model.default_params.omega.clone();
    let omega_iov = model
        .default_params
        .omega_iov
        .clone()
        .expect("IOV model carries omega_iov");
    let n_eta = model.n_eta;
    let k = 2usize;
    let z = vec![0.12f64, -0.08, 0.06, -0.05];
    assert_eq!(z.len(), n_eta + k * model.n_kappa);

    let g = analytic_eta_nll_gradient_iov(
        &model,
        &subject,
        &theta,
        &z,
        &omega,
        &omega_iov,
        &sigma,
        n_eta,
        model.n_kappa,
        k,
        None,
    )
    .expect("the eta-only analytic route must serve a DCM+IOV subject");

    let nll_at = |zz: &[f64]| {
        let (eta, kaps) = zz.split_at(n_eta);
        let kappas: Vec<Vec<f64>> = (0..k)
            .map(|g| kaps[g * model.n_kappa..(g + 1) * model.n_kappa].to_vec())
            .collect();
        individual_nll_iov(
            &model,
            &subject,
            &theta,
            eta,
            &kappas,
            &omega,
            Some(&omega_iov),
            &sigma,
        )
    };

    for i in 0..z.len() {
        let h = 1e-6 * (1.0 + z[i].abs());
        let mut zp = z.clone();
        zp[i] = z[i] + h;
        let fp = nll_at(&zp);
        zp[i] = z[i] - h;
        let fm = nll_at(&zp);
        let fd = (fp - fm) / (2.0 * h);
        assert!(
            fd.abs() > 1e-8,
            "z[{i}] must carry signal or the comparison is vacuous"
        );
        let rel = (g[i] - fd).abs() / fd.abs().max(1e-6);
        assert!(
            rel < 1e-5,
            "d/dz[{i}] ({}): analytic={:.8e} central FD={:.8e} rel={:.2e}",
            if i < n_eta { "eta" } else { "kappa" },
            g[i],
            fd,
            rel
        );
    }
}

/// The η-only scope must reach the **inner loop's own gate**, not just the walk it guards.
///
/// `subject_eta_grad_iov_analytical` consults `iov_analytical_eta_supported` internally, but
/// every inner-loop entry point screens on a model-level predicate *first*
/// (`iov_inner_subject_route`, `analytic_iov_inner`, `inner_reports_analytic_model`). While
/// those read the strict `iov_sens_supported`, relaxing the walk changed nothing for FOCE /
/// FOCEI: every DCM+IOV subject still took the FD η-gradient and still counted toward
/// `n_fd_subjects`. Worse, AGQ reaches `analytic_eta_nll_gradient_iov` with no such screen, so
/// the route taken and the route reported disagreed — the drift the #637 guards exist to stop.
///
/// Pinned on the predicates rather than on a fit, so a regression names the gate that moved.
#[test]
fn the_dcm_iov_eta_scope_reaches_the_inner_loop_gate() {
    use crate::estimation::inner_optimizer::inner_reports_analytic_model;
    use crate::sens::provider::{iov_sens_eta_supported, iov_sens_supported};

    let model = dcm_iov_model();
    assert!(model.n_kappa > 0, "fixture must carry IOV");

    // Both predicates admit a DCM now: the η-only one since #1015, the strict outer one
    // since #1339 (declared-θ axis clause + the chained weight columns). What this test
    // pins is that the *inner* gate reads the η-only predicate — the two agreeing on this
    // model no longer distinguishes them, so the check below is on the reported route.
    assert!(
        iov_sens_eta_supported(&model),
        "a DCM+IOV model is inside the η-only IOV scope"
    );
    assert!(
        iov_sens_supported(&model),
        "a DCM+IOV model is inside the strict (outer) IOV scope as of #1339"
    );

    // The reported inner method — which `build_info::gradient_method_inner` and the
    // FD-fallback warning both read — must follow the route the subject actually takes.
    assert!(
        inner_reports_analytic_model(&model),
        "the inner loop must report (and take) the analytic η-gradient for a DCM+IOV model"
    );

    // A closed-form IOV model with **no** weight block is unaffected: its program seeds every
    // declared θ, so the two predicates agree and the relaxation cannot have widened anything
    // for a model that was already served.
    let plain = parse_model_string(
        r#"
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.001, 500.0)
  omega ETA_CL ~ 0.09
  kappa KAPPA_CL ~ 0.05
  sigma PROP ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)
"#,
    )
    .expect("plain IOV model parses");
    assert!(plain.covariate_nns.is_empty(), "control must carry no NN");
    assert!(
        iov_sens_supported(&plain),
        "a plain closed-form IOV model is inside the strict scope"
    );
    assert_eq!(
        iov_sens_eta_supported(&plain),
        iov_sens_supported(&plain),
        "the two predicates must agree on every model without an NN weight block"
    );
}

/// The generic (`PkNum`) statement evaluator must see the network's **real** output.
///
/// `IndivParamProgram` carries statements and layout but no `[covariate_nn]` handles, so
/// before `ModelNnGuard` existed `eval_statements_g` evaluated `Op::PushNnOutput` against
/// a hardcoded empty slice. In a debug build that trips the arm's `debug_assert`; in a
/// **release** build it silently pushed `0.0`. Nothing detected it, because the θ-axis
/// accounting happened to exclude every DCM from the program-driven analytic paths and
/// the arm was unreachable. Relaxing that gate for the IOV η-gradient made it reachable,
/// and a network output silently read as zero is a wrong gradient with the right shape.
///
/// Pinned directly rather than through a gradient, so removing the plumbing fails with a
/// message that says what broke. The unguarded case is deliberately *not* exercised — it
/// panics under `debug_assert`, which is the behaviour we want and not something to
/// assert around.
#[test]
fn the_generic_evaluator_sees_real_nn_outputs_under_the_guard() {
    use crate::nn::CovariateMapper;
    use crate::parser::model_parser::ModelNnGuard;

    let model = parse_model_string(&dcm_model_src()).expect("DCM parses");
    let theta = probe_theta(&model);
    let prog = model
        .indiv_param_partials
        .indiv_param_program
        .as_ref()
        .expect("DCM carries an individual-parameter program");
    let cov = HashMap::from([("WT".to_string(), 72.0), ("CRCL".to_string(), 95.0)]);

    let nn = &model.covariate_nns[0];
    let w = &theta[nn.weights_offset..nn.weights_offset + nn.mapper.n_weights()];
    let truth = nn.mapper.forward_raw(w, &cov).expect("forward");
    assert!(truth[0] > 0.0, "fixture must emit a non-zero CL");

    let eta = vec![0.0f64, 0.0];
    // Slot order follows `pk_slots`; row 0 is the NN-fed CL, and with eta = 0 the
    // mu-ref composition `TYPICAL_PK.CL * exp(ETA_CL)` is exactly the NN output.
    let outer = ModelNnGuard::enter(vec![truth.clone()]);
    let guarded = prog.eval_param_eta_grad::<2>(&theta, &eta, &cov);
    assert!(
        (guarded[0].value - truth[0]).abs() < 1e-12,
        "guarded evaluator must see the real NN output: got {}, want {}",
        guarded[0].value,
        truth[0]
    );

    // A nested guard must shadow, and restore the outer values on drop — the property
    // that keeps a per-event guard from leaking across events.
    let shadow = truth[0] * 3.0 + 1.0;
    {
        let _inner = ModelNnGuard::enter(vec![vec![shadow, truth[1]]]);
        let inner_vals = prog.eval_param_eta_grad::<2>(&theta, &eta, &cov);
        assert!(
            (inner_vals[0].value - shadow).abs() < 1e-12,
            "the inner guard must shadow the outer one"
        );
    }
    let restored = prog.eval_param_eta_grad::<2>(&theta, &eta, &cov);
    assert!(
        (restored[0].value - truth[0]).abs() < 1e-12,
        "ModelNnGuard must restore the previous ambient outputs on drop: got {}, want {}",
        restored[0].value,
        truth[0]
    );
    drop(outer);
}

// ---------------------------------------------------------------------------
// The `fd_all` branch: M3 / dense-residual-covariance / TTE
// ---------------------------------------------------------------------------
//
// `obs_nll_subject_grad` and `obs_nll_subject_grad_iov` each fork on `fd_all`
// (`BloqMethod::M3`, a non-empty `residual_correlations`, or — under `survival` —
// any endpoint) into a *second* FD loop that reuses none of the non-M3 code. The
// hybrid is wired into both forks separately: each has its own
// `NnGradPlan::build`, its own `covers(i)` skip, and its own `plan.accumulate`.
//
// So the parity checks above, which run on a plain `proportional` error model,
// say nothing about the M3 fork — a plan dropped there, or a `covers` skip left
// without the matching `accumulate`, would leave every NN weight's gradient at
// exactly 0.0 and no existing test would notice. These two mirror the non-M3
// parity tests across that fork, against central FD of the same evaluators.

/// `dcm_model_src()` plus `bloq_method = m3`, which is what selects the `fd_all`
/// fork. The censoring itself is incidental — the fork is chosen by the *option*,
/// not by the data — but one genuinely censored row is included so the M3
/// normal-tail term contributes to the objective the FD reference differentiates.
fn dcm_m3_model_src() -> String {
    format!("{}\n[fit_options]\n  bloq_method = m3\n", dcm_model_src())
}

/// A subject whose last observation is below the quantitation limit.
fn m3_subject() -> Subject {
    let mut subject = static_subject();
    // `cens = 1` marks the row censored; the recorded DV is the LOQ.
    subject.cens = vec![0, 0, 0, 1];
    subject.observations[3] = 2.0;
    subject
}

/// The M3 fork of `obs_nll_subject_grad` must serve NN weights through the hybrid,
/// to the same tolerance as the non-M3 fork.
#[test]
fn hybrid_nn_weight_gradient_matches_central_fd_under_m3() {
    let model = parse_model_string(&dcm_m3_model_src()).expect("M3 DCM model parses");
    assert!(
        matches!(model.bloq_method, crate::types::BloqMethod::M3),
        "fixture must select the fd_all fork"
    );

    let subject = m3_subject();
    let theta = probe_theta(&model);
    let sigma_values = vec![0.2f64];
    let eta = vec![0.15f64, -0.1f64];
    let n_theta = theta.len();
    let n_sigma = 1;
    let (mask, lo, hi) = unpinned(&model, n_theta + n_sigma);

    // Same premise guard as the non-M3 test: without a plan this would be
    // comparing the plain FD loop against FD and would pass for the wrong reason.
    assert!(
        NnGradPlan::build(&model, &subject, &theta, n_theta).is_some(),
        "static-covariate DCM subject must be served by the hybrid path under M3"
    );

    let mut scratch = EventPkParams::default();
    let (_nll, grad) = obs_nll_subject_grad(
        &model,
        &subject,
        &theta,
        &sigma_values,
        &eta,
        &mask,
        &lo,
        &hi,
        n_theta,
        n_sigma,
        &mut scratch,
    );
    let reference = central_fd_theta_grad(&model, &subject, &theta, &sigma_values, &eta, &mask);

    let nn_idx = nn_theta_indices(&model);
    let peak = nn_idx
        .iter()
        .map(|&i| reference[i].abs())
        .fold(0.0f64, f64::max);
    assert!(
        peak > 0.0,
        "the M3 objective must actually depend on the network weights"
    );
    let mut n_informative = 0;
    for &i in &nn_idx {
        if reference[i].abs() > 1e-3 * peak {
            n_informative += 1;
        }
        assert!(
            relative(grad[i], reference[i]) < 1e-5,
            "M3 NN weight theta[{i}]: hybrid={:.8e}, central FD={:.8e}, rel={:.2e}",
            grad[i],
            reference[i],
            relative(grad[i], reference[i])
        );
    }
    assert!(
        n_informative >= 12,
        "expected most NN weights to carry signal under M3, got {n_informative}/{}",
        nn_idx.len()
    );
}

/// The IOV × M3 corner: `obs_nll_subject_grad_iov`'s own `fd_all` fork. Reached by
/// no other test — the IOV parity test above is non-M3, and the M3 test above is
/// non-IOV — yet it is where a heavily-censored DCM+IOV fit actually runs (#654).
#[test]
fn hybrid_nn_weight_gradient_matches_central_fd_under_m3_iov() {
    let src = dcm_m3_model_src()
        .replace(
            "  CL = TYPICAL_PK.CL * exp(ETA_CL)",
            "  CL = TYPICAL_PK.CL * exp(ETA_CL + KAPPA_CL)",
        )
        .replace(
            "  omega ETA_V  ~ 0.09",
            "  omega ETA_V  ~ 0.09\n  kappa KAPPA_CL ~ 0.05",
        );
    let model = parse_model_string(&src).expect("M3 IOV DCM model parses");
    assert!(model.n_kappa > 0, "model must carry IOV");
    assert!(
        matches!(model.bloq_method, crate::types::BloqMethod::M3),
        "fixture must select the fd_all fork"
    );

    let mut subject = m3_subject();
    subject.occasions = vec![1, 1, 2, 2];
    subject.dose_occasions = vec![1];

    let theta = probe_theta(&model);
    let sigma_values = vec![0.2f64];
    let eta = vec![0.15f64, -0.1f64];
    let kappas = vec![vec![0.08f64], vec![-0.06f64]];
    let n_theta = theta.len();
    let n_sigma = 1;
    let (mask, lo, hi) = unpinned(&model, n_theta + n_sigma);

    assert!(
        NnGradPlan::build(&model, &subject, &theta, n_theta).is_some(),
        "IOV DCM subject must be served by the hybrid path under M3"
    );

    let mut scratch = EventPkParams::default();
    let (_nll, grad) = obs_nll_subject_grad_iov(
        &model,
        &subject,
        &theta,
        &sigma_values,
        &eta,
        &kappas,
        &mask,
        &lo,
        &hi,
        n_theta,
        n_sigma,
        &mut scratch,
    );

    let mut reference = vec![0.0f64; n_theta];
    for i in 0..n_theta {
        let h = 1e-6 * (1.0 + theta[i].abs());
        let mut tp = theta.clone();
        tp[i] += h;
        let f_plus = obs_nll_subject_into_iov(
            &model,
            &subject,
            &tp,
            &sigma_values,
            &eta,
            &kappas,
            &mut scratch,
        );
        tp[i] = theta[i] - h;
        let f_minus = obs_nll_subject_into_iov(
            &model,
            &subject,
            &tp,
            &sigma_values,
            &eta,
            &kappas,
            &mut scratch,
        );
        let raw = (f_plus - f_minus) / (2.0 * h);
        reference[i] = if mask[i] { theta[i] * raw } else { raw };
    }

    let nn_idx = nn_theta_indices(&model);
    let peak = nn_idx
        .iter()
        .map(|&i| reference[i].abs())
        .fold(0.0f64, f64::max);
    assert!(
        peak > 0.0,
        "the M3 IOV objective must actually depend on the network weights"
    );
    for &i in &nn_idx {
        assert!(
            relative(grad[i], reference[i]) < 1e-5,
            "M3 IOV NN weight theta[{i}]: hybrid={:.8e}, central FD={:.8e}, rel={:.2e}",
            grad[i],
            reference[i],
            relative(grad[i], reference[i])
        );
    }
}

// ---------------------------------------------------------------------------
// `NnGradPlan` guard rails
// ---------------------------------------------------------------------------

/// `NnGradPlan::build` declines rather than panicking when a block's weight run
/// would fall outside the caller's θ accounting.
///
/// This is the wiring-bug guard, not a user-facing condition: a `weights_offset`
/// past `n_theta` means the parser and the optimizer disagree about the θ layout.
/// Declining routes the caller to its plain per-θ FD loop, which is slower but
/// correct; indexing would be an out-of-bounds panic mid-fit.
#[test]
fn nn_grad_plan_declines_when_a_block_runs_past_the_theta_count() {
    let model = parse_model_string(&dcm_model_src()).expect("DCM parses");
    let subject = static_subject();
    let theta = probe_theta(&model);
    let n_theta = theta.len();

    // The premise: the honest accounting is served.
    assert!(
        NnGradPlan::build(&model, &subject, &theta, n_theta).is_some(),
        "the plan must be available at the true theta count"
    );

    // `n_theta` understated — the block's run ends past it.
    let nn = &model.covariate_nns[0];
    let short_n_theta = nn.weights_offset + nn.mapper.mlp().n_weights() - 1;
    assert!(
        NnGradPlan::build(&model, &subject, &theta, short_n_theta).is_none(),
        "a block ending past `n_theta` must decline the plan"
    );

    // The θ slice itself truncated, with `n_theta` still honest — the second half
    // of the same guard, and the one an out-of-bounds slice would trip.
    let truncated = &theta[..theta.len() - 1];
    assert!(
        NnGradPlan::build(&model, &subject, truncated, n_theta).is_none(),
        "a block ending past the theta slice must decline the plan"
    );
}

/// A block whose every weight is pinned is skipped outright — including the two
/// output-bias solves, which are the whole cost of the hybrid.
///
/// `pinned_output_bias_reports_zero_without_disturbing_free_weights` covers a
/// *partly* pinned block; this is the all-pinned short circuit, where the right
/// behaviour is to do no work at all rather than to solve and multiply by zero.
#[test]
fn a_fully_pinned_nn_block_does_no_solves() {
    let model = parse_model_string(&dcm_model_src()).expect("DCM parses");
    let subject = static_subject();
    let theta = probe_theta(&model);
    let n_theta = theta.len();
    let n_sigma = 1;
    let (mask, mut lo, mut hi) = unpinned(&model, n_theta + n_sigma);

    let plan = NnGradPlan::build(&model, &subject, &theta, n_theta).expect("plan available");

    // Pin every NN weight by collapsing its bounds onto its value.
    for &i in &nn_theta_indices(&model) {
        lo[i] = theta[i];
        hi[i] = theta[i];
    }

    let mut solves = 0usize;
    let mut grad = vec![0.0f64; n_theta + n_sigma];
    plan.accumulate(
        |_i, _sign| {
            solves += 1;
            1.0
        },
        &theta,
        &mask,
        &lo,
        &hi,
        &mut grad,
    );

    assert_eq!(
        solves, 0,
        "an all-pinned block must skip its output-bias solves, not just zero the result"
    );
    for &i in &nn_theta_indices(&model) {
        assert_eq!(
            grad[i], 0.0,
            "pinned NN weight theta[{i}] must be left at zero"
        );
    }
}

/// A log-packed NN weight's entry is scaled by `theta[g]`, matching the chain rule
/// the caller's own FD loop applies to log-packed θ.
///
/// The DSL gives NN weights a lower bound of `-inf`, so `theta_packs_log` is false
/// for all of them and the production mask never selects this arm. It is still the
/// arm that would run if a future parameterization change made a weight positive-
/// bounded, so it is pinned here against the same `theta[g] * raw` rule the rest of
/// the θ vector obeys — driven through `accumulate` with a hand-built mask.
#[test]
fn a_log_packed_nn_weight_is_scaled_by_its_value() {
    let model = parse_model_string(&dcm_model_src()).expect("DCM parses");
    let subject = static_subject();
    let theta = probe_theta(&model);
    let n_theta = theta.len();
    let n_sigma = 1;
    let (_mask, lo, hi) = unpinned(&model, n_theta + n_sigma);
    let nn_idx = nn_theta_indices(&model);

    let plan = NnGradPlan::build(&model, &subject, &theta, n_theta).expect("plan available");

    // A constant derivative makes the two runs differ *only* by the log-packing
    // factor, so the assertion below isolates that factor.
    let deriv = |_i: usize, _sign: f64| 0.25f64;

    let mut raw_grad = vec![0.0f64; n_theta + n_sigma];
    plan.accumulate(
        deriv,
        &theta,
        &vec![false; n_theta + n_sigma],
        &lo,
        &hi,
        &mut raw_grad,
    );

    let mut logged_grad = vec![0.0f64; n_theta + n_sigma];
    plan.accumulate(
        deriv,
        &theta,
        &vec![true; n_theta + n_sigma],
        &lo,
        &hi,
        &mut logged_grad,
    );

    let mut n_checked = 0;
    for &g in &nn_idx {
        if raw_grad[g] == 0.0 {
            continue;
        }
        n_checked += 1;
        let expected = theta[g] * raw_grad[g];
        assert!(
            (logged_grad[g] - expected).abs() <= 1e-12 * expected.abs().max(1e-12),
            "log-packed NN weight theta[{g}]: got {:.8e}, want theta[g]*raw = {:.8e}",
            logged_grad[g],
            expected
        );
    }
    assert!(
        n_checked > 0,
        "the fixture must produce at least one non-zero weight entry to scale"
    );
}

/// The snapshots can be perfectly self-consistent and *still* disagree with the
/// subject-static covariate map — that is the second half of
/// `static_nn_input_map`'s check, and a distinct decline from the time-varying
/// case above (where the snapshots disagree with each other).
///
/// It matters because the static map is what a subject's records fall back to for
/// a *kind* of record it has none of: a subject with observation snapshots but no
/// dose snapshots reads `subject.covariates` at its dose events. If the two
/// disagree, the network sees one input vector at the doses and another at the
/// observations — time-varying in effect, however uniform each snapshot set looks
/// — so the single-`z` factorization cannot represent it and the plan must
/// decline rather than pick one of the two maps.
#[test]
fn snapshots_disagreeing_with_the_static_map_decline_the_plan() {
    let model = parse_model_string(&dcm_model_src()).expect("DCM model parses");
    let mut subject = static_subject();
    // Every snapshot agrees with every other snapshot...
    let snap = HashMap::from([("WT".into(), 80.0), ("CRCL".into(), 95.0)]);
    subject.obs_covariates = vec![snap.clone(), snap.clone(), snap.clone(), snap.clone()];
    // ...but `subject.covariates` still carries WT = 72 (see `static_subject`),
    // and this subject has no dose snapshots, so its dose event reads that.
    assert!(subject.dose_covariates.is_empty());
    assert_ne!(
        subject.covariates.get("WT"),
        snap.get("WT"),
        "the premise is a base map that disagrees with the snapshots"
    );

    let theta = probe_theta(&model);
    let n_theta = theta.len();
    assert!(
        NnGradPlan::build(&model, &subject, &theta, n_theta).is_none(),
        "a static map disagreeing with the snapshots must route to the per-theta FD loop"
    );

    // Agreeing on the NN's inputs is enough — the base map may still differ on a
    // covariate the network does not read, which pins the check to `input_names`
    // rather than to the whole map.
    let mut agreeing = subject.clone();
    agreeing.covariates.insert("WT".to_string(), 80.0);
    agreeing.covariates.insert("UNREAD".to_string(), 1234.0);
    assert!(
        NnGradPlan::build(&model, &agreeing, &theta, n_theta).is_some(),
        "agreement on the network's own inputs must be sufficient"
    );
}

// ---------------------------------------------------------------------------
// #1016 follow-up: a model that reads a generated weight theta directly
// ---------------------------------------------------------------------------
//
// `theta_names` is extended with the generated `W_…` / `B_…` names *before*
// `[individual_parameters]` parses, so a model can name one of them. The hybrid
// factorization is exact only while the weights reach the likelihood through
// the network output alone, so this must route to the plain per-theta FD loop
// rather than return a gradient that smears the direct term across the block.

/// The DCM fixture with an output bias added directly to `CL`.
fn dcm_direct_bias_src() -> String {
    dcm_model_src().replace(
        "  CL = TYPICAL_PK.CL * exp(ETA_CL)",
        "  CL = TYPICAL_PK.CL * exp(ETA_CL) + B_TYPICAL_PK_2_1",
    )
}

/// The same, but reading a non-bias weight — the variant where the direct term
/// is dropped entirely rather than misattributed, because only output biases
/// are perturbed.
fn dcm_direct_weight_src() -> String {
    dcm_model_src().replace(
        "  CL = TYPICAL_PK.CL * exp(ETA_CL)",
        "  CL = TYPICAL_PK.CL * exp(ETA_CL) + W_TYPICAL_PK_1_1_1",
    )
}

#[test]
fn a_direct_weight_reference_parses_and_is_recorded() {
    // The premise of the finding: this really is accepted syntax. If a future
    // change rejects it at parse time instead, this test is the one that says
    // so out loud rather than letting the decline path quietly go dead.
    for src in [dcm_direct_bias_src(), dcm_direct_weight_src()] {
        let model = parse_model_string(&src).expect("a direct weight reference parses");
        assert!(
            model.parse_warnings.iter().any(|w| w
                .contains(crate::parser::model_parser::NN_WEIGHT_DIRECT_REFERENCE_MARKER)),
            "the direct reference must be recorded: {:?}",
            model.parse_warnings
        );
    }
}

#[test]
fn the_plain_dcm_model_records_no_direct_reference() {
    // The marker has to discriminate, or it would disable the hybrid for every
    // NN model and the shortcut would never run at all.
    let model = parse_model_string(&dcm_model_src()).expect("DCM parses");
    assert!(
        !model
            .parse_warnings
            .iter()
            .any(|w| w.contains(crate::parser::model_parser::NN_WEIGHT_DIRECT_REFERENCE_MARKER)),
        "a network read only through its outputs must keep the hybrid: {:?}",
        model.parse_warnings
    );
}

#[test]
fn nn_grad_plan_declines_a_direct_weight_reference() {
    for src in [dcm_direct_bias_src(), dcm_direct_weight_src()] {
        let model = parse_model_string(&src).expect("parses");
        let subject = static_subject();
        let theta = probe_theta(&model);
        let n_theta = theta.len();
        assert!(
            NnGradPlan::build(&model, &subject, &theta, n_theta).is_none(),
            "a direct weight reference must route to the per-theta FD path"
        );
    }
}

/// The regression the finding asked for: with the shortcut declined, the theta
/// gradient of a model that reads a weight directly matches central finite
/// differences.
///
/// Before the fix the reported probe was `hybrid = -1.597e-1` against
/// `central = -6.379e-2` — a relative error of 0.6 on `theta[1]`, silently fed
/// to SAEM/IMP/VI.
#[test]
fn a_direct_weight_reference_still_matches_central_fd() {
    for src in [dcm_direct_bias_src(), dcm_direct_weight_src()] {
        let model = parse_model_string(&src).expect("parses");
        let subject = static_subject();
        let theta = probe_theta(&model);
        let sigma_values = vec![0.2f64];
        let eta = vec![0.15f64, -0.1f64];
        let n_theta = theta.len();
        let n_sigma = sigma_values.len();
        let (mask, lo, hi) = unpinned(&model, n_theta + n_sigma);

        let mut scratch = EventPkParams::default();
        let (_nll, grad) = obs_nll_subject_grad(
            &model,
            &subject,
            &theta,
            &sigma_values,
            &eta,
            &mask,
            &lo,
            &hi,
            n_theta,
            n_sigma,
            &mut scratch,
        );
        let reference = central_fd_theta_grad(&model, &subject, &theta, &sigma_values, &eta, &mask);

        // Every theta, not just the NN block: the directly-referenced weight is
        // exactly the coordinate the shortcut got wrong.
        let peak = reference.iter().map(|g| g.abs()).fold(0.0f64, f64::max);
        assert!(peak > 0.0, "the probe must carry signal");
        for i in 0..n_theta {
            let tol = 1e-4 * peak.max(1.0);
            assert!(
                (grad[i] - reference[i]).abs() < tol,
                "theta[{i}]: got {:.8e}, central FD {:.8e} (abs diff {:.2e} > {tol:.2e})",
                grad[i],
                reference[i],
                (grad[i] - reference[i]).abs()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// DCM + IOV: the analytic OUTER route (#1339)
// ---------------------------------------------------------------------------

/// The busulfan shape of #1327/#1339: 2-cpt IV infusion, a 4 → 2 → 4 network on every
/// typical value (22 weights), κ on CL and V1, combined error. `n_theta + n_stacked`
/// (22 + 3 + 2K) exceeds the walk's 24-axis cap for any K ≥ 1, so the outer walk must
/// chunk its θ columns — which is the point of the fixture.
fn dcm_two_kappa_src() -> &'static str {
    r#"
[parameters]
  omega ETA_CL ~ 0.068
  omega ETA_V1 ~ 0.038
  omega ETA_V2 ~ 0.077
  kappa KAPPA_CL ~ 0.0111
  kappa KAPPA_V1 ~ 0.0153
  sigma PROP_ERR ~ 0.05 (sd)
  sigma ADD_ERR  ~ 0.1 (sd)

[covariate_nn TYPICAL_PK]
  inputs     = [LAGE, LWT, LHT, SEX]
  center     = [2.639, 3.727, 4.91, 0]
  scale      = [1.548, 0.907, 0.342, 1]
  outputs    = [CL, V1, Q, V2]
  layers     = [2]
  activation = tanh
  output     = softplus
  init       = [11.6, 46.5, 14.3, 10.8]

[individual_parameters]
  CL = TYPICAL_PK.CL * exp(ETA_CL + KAPPA_CL)
  V1 = TYPICAL_PK.V1 * exp(ETA_V1 + KAPPA_V1)
  Q  = TYPICAL_PK.Q
  V2 = TYPICAL_PK.V2 * exp(ETA_V2)

[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)

[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR)

[fit_options]
  method = focei
  iov_column = OCC
"#
}

fn dcm_two_kappa_model() -> CompiledModel {
    parse_model_string(dcm_two_kappa_src()).expect("2-κ DCM parses")
}

/// Three occasions, one 3-hour infusion each, two observations per occasion — the later
/// doses land with residual drug present (a multi-dose fixture, per CLAUDE.md's
/// non-degeneracy rule), so κ on a later occasion moves both its own rows and the
/// carry-over into the next.
fn dcm_two_kappa_subject() -> Subject {
    let mut cov = HashMap::new();
    cov.insert("LAGE".to_string(), 1.8);
    cov.insert("LWT".to_string(), 3.1);
    cov.insert("LHT".to_string(), 4.7);
    cov.insert("SEX".to_string(), 1.0);
    let dose = |t: f64| DoseEvent::new(t, 100.0, 1, 100.0 / 3.0, false, 0.0);
    Subject {
        id: "1".into(),
        doses: vec![dose(0.0), dose(24.0), dose(48.0)],
        obs_times: vec![3.5, 6.0, 27.5, 30.0, 51.5, 54.0],
        obs_raw_times: Vec::new(),
        observations: vec![1.9, 1.5, 2.3, 1.8, 2.5, 2.0],
        obs_cmts: vec![1; 6],
        covariates: cov,
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        reset_occasions: Vec::new(),
        cens: vec![0; 6],
        occasions: vec![1, 1, 2, 2, 3, 3],
        obs_l2: Vec::new(),
        dose_occasions: vec![1, 2, 3],
        fremtype: Vec::new(),
        obs_records: vec![],
    }
}

/// Every weight off zero (a zero hidden→output weight zeroes the whole first layer's
/// gradient, and a zero hidden activation zeroes the output weights' — the parsed
/// defaults have both), output biases restored to the declared `init` values so the
/// typical values stay physiological.
fn dcm_two_kappa_theta(model: &CompiledModel) -> Vec<f64> {
    let mut theta: Vec<f64> = model
        .default_params
        .theta
        .iter()
        .enumerate()
        .map(|(i, &t)| t + 0.3 * ((i as f64) * 0.7 + 0.3).sin())
        .collect();
    for nn in &model.covariate_nns {
        for (k, v) in [11.6f64, 46.5, 14.3, 10.8].iter().enumerate() {
            // softplus⁻¹(v) = ln(eᵛ − 1)
            theta[nn.weights_offset + nn.mapper.mlp().output_bias_index(k)] = (v.exp() - 1.0).ln();
        }
    }
    theta
}

/// The outer IOV sensitivities of a DCM must be **analytic** and match central finite
/// differences of `predict_iov` on all four blocks — `∂f/∂θ` over the whole weight block,
/// `∂f/∂[η, κ]`, `∂²f/∂[η,κ]²` and the mixed `∂²f/∂[η,κ]∂θ` — on a subject whose stacked
/// width forces the θ columns into two chunks.
///
/// What it catches (each verified by mutation while writing it): a weight column left
/// unchained (`iov_nn_combined_derivs_dyn` without the `∂z/∂w` product reads as zero,
/// and the FD reference is not); a chunk scattered onto the wrong columns (the second
/// chunk's `θ_15..θ_21` written at `0..7`); a merge that drops the earlier chunk. The
/// stacked-η blocks are the same program gradient the inner walk already pins
/// (`dcm_iov_eta_gradient_is_analytic_and_matches_central_fd`) — checked here too so
/// the chunked scatter cannot corrupt them.
#[test]
fn dcm_iov_outer_sensitivities_are_analytic_and_match_fd_of_predict_iov() {
    let model = dcm_two_kappa_model();
    let subject = dcm_two_kappa_subject();
    let theta = dcm_two_kappa_theta(&model);
    assert_eq!(model.n_theta, 22, "4→2→4 network is 22 weights");
    check_dcm_iov_outer_vs_fd(&model, &subject, &theta);
}

/// FD-parity harness for the analytic IOV **outer** walk on a θ block wide enough to be
/// chunked: compares all four `SubjectSens` blocks against central differences of
/// `predict_iov`, and asserts that every θ column is *live* on at least one observation
/// so no column's parity is the vacuous `0 == 0`. Shared by the plain DCM fixture and by
/// the absolute-axis-step fixtures (`[initial_conditions]`, `obs_scale`), which run the
/// same walk with one extra post-walk step in the chunk's own basis (#1339).
fn check_dcm_iov_outer_vs_fd(model: &CompiledModel, subject: &Subject, theta: &[f64]) {
    use crate::sens::provider::{iov_analytical_supported, subject_sensitivities_iov};

    let n_eta = model.n_eta;
    let n_kappa = model.n_kappa;
    let n_theta = model.n_theta;
    let k = crate::stats::likelihood::iov_occasion_groups(subject).len();
    assert_eq!(k, 3);
    let n_st = n_eta + k * n_kappa;
    assert!(
        n_theta + n_st > 24,
        "fixture must exceed the single-chunk walk width or the chunking is untested"
    );
    assert!(
        iov_analytical_supported(model),
        "a DCM+IOV model must be inside the full analytic IOV scope (#1339)"
    );

    let stacked: Vec<f64> = (0..n_st)
        .map(|i| 0.2 * ((i as f64) * 1.3 + 0.5).sin())
        .collect();
    let sens = subject_sensitivities_iov(model, subject, theta, &stacked)
        .expect("the analytic IOV outer walk must serve a DCM+IOV subject");
    assert_eq!(sens.obs.len(), subject.obs_times.len());

    let pred = |st: &[f64], th: &[f64], j: usize| -> f64 {
        let eta_bsv = st[..n_eta].to_vec();
        let kappas: Vec<Vec<f64>> = (0..k)
            .map(|g| st[n_eta + g * n_kappa..n_eta + (g + 1) * n_kappa].to_vec())
            .collect();
        crate::pk::predict_iov(model, subject, th, &eta_bsv, &kappas)[j]
    };
    let he = 1e-6;
    let heh = 1e-4;
    // Per *column* liveness, not a count of live `(θ, obs)` pairs: a pair count can be
    // satisfied by a subset of the columns firing on many observations while a whole
    // chunk stays silent, and the parity on a silent column is vacuous (0 vs 0).
    let mut live_weight_col = vec![false; n_theta];
    // Peak |∂f/∂θ_m| over the observations, reported when a column comes out silent so the
    // threshold below can be read off a measurement rather than argued about.
    let mut peak_col = vec![0.0f64; n_theta];
    // A column counts as live when its FD gradient is far enough above the parity
    // assertion's own `epsilon = 1e-7` that a wrong analytic value of 0 would actually
    // fail it: `|0 − g| = g > max(1e-7, 2e-4·g)` for any `g` past ~1e-7, so 100× that is
    // ample headroom. Measured quietest columns across the three fixtures this harness
    // serves: 1.9e-3 (plain DCM), 8.1e-4 (`[initial_conditions]`) and 2.7e-5 (`obs_scale`,
    // whose `f/(TVBASE·e^η·V1)` divides every derivative by ~20). An absolute `1e-3` bar —
    // the first cut — is really a bar on the *prediction's scale*, and silently declared
    // 18 of the 23 obs_scale columns dark.
    const LIVE_COL_MIN: f64 = 1e-5;
    for (j, obs) in sens.obs.iter().enumerate() {
        approx::assert_relative_eq!(obs.f, pred(&stacked, &theta, j), max_relative = 1e-9);
        for p in 0..n_st {
            let mut sp = stacked.clone();
            sp[p] += he;
            let mut sm = stacked.clone();
            sm[p] -= he;
            let g = (pred(&sp, &theta, j) - pred(&sm, &theta, j)) / (2.0 * he);
            approx::assert_relative_eq!(obs.df_deta[p], g, max_relative = 2e-4, epsilon = 1e-7);
            for q in 0..n_st {
                let mut pp = stacked.clone();
                pp[p] += heh;
                pp[q] += heh;
                let mut pm = stacked.clone();
                pm[p] += heh;
                pm[q] -= heh;
                let mut mp = stacked.clone();
                mp[p] -= heh;
                mp[q] += heh;
                let mut mm = stacked.clone();
                mm[p] -= heh;
                mm[q] -= heh;
                let hh = (pred(&pp, &theta, j) - pred(&pm, &theta, j) - pred(&mp, &theta, j)
                    + pred(&mm, &theta, j))
                    / (4.0 * heh * heh);
                approx::assert_relative_eq!(
                    obs.d2f_deta2[p * n_st + q],
                    hh,
                    max_relative = 3e-3,
                    epsilon = 1e-5
                );
            }
        }
        for m in 0..n_theta {
            let s = he * (1.0 + theta[m].abs());
            let mut tp = theta.to_vec();
            tp[m] += s;
            let mut tm = theta.to_vec();
            tm[m] -= s;
            let g = (pred(&stacked, &tp, j) - pred(&stacked, &tm, j)) / (2.0 * s);
            assert!(g.is_finite(), "FD probe of θ_{m} at obs {j} is not finite");
            if g.abs() > peak_col[m] {
                peak_col[m] = g.abs();
            }
            if g.abs() > LIVE_COL_MIN {
                live_weight_col[m] = true;
            }
            approx::assert_relative_eq!(obs.df_dtheta[m], g, max_relative = 2e-4, epsilon = 1e-7);
            for p in 0..n_st {
                let sh = heh * (1.0 + theta[m].abs());
                let mut ep = stacked.clone();
                ep[p] += heh;
                let mut em = stacked.clone();
                em[p] -= heh;
                let mut tp2 = theta.to_vec();
                tp2[m] += sh;
                let mut tm2 = theta.to_vec();
                tm2[m] -= sh;
                let hh = (pred(&ep, &tp2, j) - pred(&ep, &tm2, j) - pred(&em, &tp2, j)
                    + pred(&em, &tm2, j))
                    / (4.0 * heh * sh);
                approx::assert_relative_eq!(
                    obs.d2f_deta_dtheta[p * n_theta + m],
                    hh,
                    max_relative = 3e-3,
                    epsilon = 1e-5
                );
            }
        }
    }
    // Every weight column must carry signal on at least one observation, or a zeroed
    // chain would pass the comparison above on that column (both sides zero). Both
    // chunks (`0..15`, `15..22`) are covered by the whole-block sweep.
    let dead: Vec<(usize, f64)> = (0..n_theta)
        .filter(|&m| !live_weight_col[m])
        .map(|m| (m, peak_col[m]))
        .collect();
    assert!(
        dead.is_empty(),
        "weight columns {dead:?} (index, peak |∂f/∂θ|) of {n_theta} are silent on every \
         observation — the probe leaves part of the network dark and the parity there is \
         vacuous"
    );
}

/// The relaxation must reach the **outer** dispatch, not just the provider: the strict
/// IOV predicate, the shared analytic-outer-gradient predicate, and the `auto` optimizer
/// resolution — which picks the gradient optimizer only when the loop will actually
/// compute an analytic gradient — all have to agree on a DCM+IOV model. Before #1339 the
/// reference fit ran derivative-free BOBYQA over 32 coordinates (546 evals to 59 470)
/// while the analytic base took L-BFGS over 14 (41 evals to 56 916).
#[test]
fn dcm_iov_model_gets_the_analytic_outer_gradient_and_a_gradient_optimizer() {
    use crate::sens::provider::{
        analytic_outer_gradient_available, iov_analytical_supported, iov_sens_supported,
    };
    use crate::types::Optimizer;

    let model = dcm_two_kappa_model();
    assert!(iov_analytical_supported(&model));
    assert!(iov_sens_supported(&model));
    assert!(analytic_outer_gradient_available(&model));
    assert_eq!(
        Optimizer::Auto.resolve_auto(&model, true),
        Optimizer::NloptLbfgs,
        "auto must resolve to the gradient optimizer now that the outer gradient is analytic"
    );

    // A DCM that reads a generated weight θ directly breaks the output-channel
    // factorization the chain relies on; it must decline the outer route (the same
    // predicate `tvcov_analytical_supported` / `nn_theta_gradient` apply) while the
    // η-only inner route — which needs no weight columns — still serves it.
    let direct = parse_model_string(
        &dcm_model_src()
            .replace(
                "  CL = TYPICAL_PK.CL * exp(ETA_CL)",
                "  CL = TYPICAL_PK.CL * exp(ETA_CL + KAPPA_CL) + 0.0 * B_TYPICAL_PK_2_1",
            )
            .replace(
                "  omega ETA_V  ~ 0.09",
                "  omega ETA_V  ~ 0.09\n  kappa KAPPA_CL ~ 0.05",
            ),
    )
    .expect("direct-weight DCM+IOV parses");
    assert!(
        !crate::sens::provider::nn_output_chain_supported(&direct),
        "fixture must actually trip the direct-reference marker"
    );
    assert!(!iov_analytical_supported(&direct));
    assert!(crate::sens::provider::iov_analytical_eta_supported(&direct));
}

/// PR #1340 review (P2): the `[initial_conditions]` impulse must run **off** the identity
/// θ chunk, in the chunk's own basis, and match FD there.
///
/// The impulse is one of three post-walk steps whose program seeds *direct* θ / η
/// references; the first cut of #1339 could only place those on absolute axes, so a
/// chunked walk declined them and the model-level gate had to guess whether any subject
/// would ever land on one chunk. It cannot: that depends on the subject's occasion count
/// `K` (`n_theta + n_eta + K·n_kappa ≤ 24`), which a model-level predicate never sees, so
/// every bound written there reported analytic for a band of models whose real (`K ≥ 2`)
/// subjects all reconverged on FD — the #637 route/report drift. `eval_scale_dual_cols`
/// removes the question by seeding on the chunk's columns instead.
///
/// The fixture is built so the mutation is reachable on **both** halves of the mapping:
/// `init(central) = TVBASE * exp(ETA_CL)` reads a declared θ *and* an η directly, the
/// subject needs two chunks (23 θ, 9 stacked → `[0..15]`, `[15..23]`), and the θ it reads
/// sits in the first chunk only — so a walk that kept seeding `θ_m → m` would write the
/// second chunk's `θ_15` axis, and one that kept the η block at `n_theta + k` would land
/// on a κ axis. Both show up as an FD mismatch here.
#[test]
fn dcm_iov_init_impulse_matches_fd_off_the_identity_chunk() {
    use crate::sens::provider::{iov_analytical_supported, subject_sensitivities_iov};

    let src = dcm_two_kappa_src()
        .replace(
            "[covariate_nn TYPICAL_PK]",
            "  theta TVBASE(0.4, 0.01, 10.0)\n\n[covariate_nn TYPICAL_PK]",
        )
        .replace(
            "[error_model]",
            "[initial_conditions]\n  init(central) = TVBASE * exp(ETA_CL) * V1\n\n[error_model]",
        );
    let model = parse_model_string(&src).expect("DCM+IOV+init parses");
    let subject = dcm_two_kappa_subject();
    let theta = dcm_two_kappa_theta(&model);

    assert!(
        !model.analytical_init.is_empty(),
        "fixture must carry the init impulse"
    );
    assert_eq!(
        model.n_theta, 23,
        "one declared θ ahead of the 22 network weights"
    );
    let n_stacked =
        model.n_eta + crate::stats::likelihood::iov_occasion_groups(&subject).len() * model.n_kappa;
    assert!(
        model.n_theta + n_stacked > 24,
        "fixture must force a second θ chunk, or the absolute-axis mutation is unreachable"
    );
    // The declared θ the init reads is column 0 — first chunk — while the second chunk
    // starts at 15, so an absolute `θ_m → m` seeding corrupts the second chunk's columns.
    assert_eq!(
        model.covariate_nns[0].weights_offset, 1,
        "the declared θ must precede the weight block"
    );

    assert!(
        iov_analytical_supported(&model),
        "an absolute-axis step no longer narrows the outer IOV gate (#1339)"
    );
    assert!(
        subject_sensitivities_iov(&model, &subject, &theta, &vec![0.05; n_stacked]).is_some(),
        "gate says analytic, so the chunked walk must serve the subject"
    );
    check_dcm_iov_outer_vs_fd(&model, &subject, &theta);
}

/// The `ExpressionScale` `obs_scale` quotient, the third absolute-axis step, on the same
/// two-chunk geometry — see `dcm_iov_init_impulse_matches_fd_off_the_identity_chunk`.
///
/// The quotient is applied per occasion group and rewrites the whole `ObsSens` row, so it
/// also pins the half the init impulse cannot: under a chunk it must rewrite **only** the
/// θ columns the chunk carries. A version that looped `0..n_theta` would divide the other
/// chunk's columns a second time when they are merged, which the θ block of the FD
/// comparison catches.
#[test]
fn dcm_iov_expression_scale_matches_fd_off_the_identity_chunk() {
    use crate::sens::provider::{iov_analytical_supported, subject_sensitivities_iov};
    use crate::types::ScalingSpec;

    let src = dcm_two_kappa_src()
        .replace(
            "[covariate_nn TYPICAL_PK]",
            "  theta TVBASE(0.4, 0.01, 10.0)\n\n[covariate_nn TYPICAL_PK]",
        )
        .replace(
            "[error_model]",
            "[scaling]\n  obs_scale = TVBASE * exp(ETA_V1) * V1\n\n[error_model]",
        );
    let model = parse_model_string(&src).expect("DCM+IOV+obs_scale parses");
    let subject = dcm_two_kappa_subject();
    let theta = dcm_two_kappa_theta(&model);

    assert!(
        matches!(model.scaling, ScalingSpec::ExpressionScale { .. }),
        "fixture must carry an ExpressionScale obs_scale"
    );
    let n_stacked =
        model.n_eta + crate::stats::likelihood::iov_occasion_groups(&subject).len() * model.n_kappa;
    assert!(
        model.n_theta + n_stacked > 24,
        "fixture must force a second θ chunk, or the absolute-axis mutation is unreachable"
    );
    assert!(
        iov_analytical_supported(&model),
        "an absolute-axis step no longer narrows the outer IOV gate (#1339)"
    );
    assert!(
        subject_sensitivities_iov(&model, &subject, &theta, &vec![0.05; n_stacked]).is_some(),
        "gate says analytic, so the chunked walk must serve the subject"
    );
    check_dcm_iov_outer_vs_fd(&model, &subject, &theta);
}

/// The analytic Form C readout, the last of the three post-walk steps, on the same
/// two-chunk geometry — see `dcm_iov_init_impulse_matches_fd_off_the_identity_chunk`.
///
/// Unlike the other two this one was always chunk-agnostic and needed no fix: a
/// dual-evaluable readout carries no `PushTheta`/`PushEta` op (the parser clears
/// `dual_evaluable` when it cannot desugar one), so it composes purely from the walk's own
/// per-observation PK duals, which are already in the chunk's basis. The first cut of
/// #1339 declined it anyway, alongside the two that did need remapping. Pinned here so a
/// future reader does not re-add that decline "for symmetry": a chunked DCM with a
/// nonlinear readout is served, and its four blocks match FD.
///
/// Run on a wide stack: the readout adds a `[Dual2<M>; N_PK]` PK-slot vector and the
/// bytecode jet on top of the walk's own frames, and at the chunked width `M = 24` that
/// clears the 2 MiB default test-thread stack in a debug build (measured: passes at 4 MiB,
/// aborts at 2). Production fits already run on the 32 MiB Rayon stack
/// (`api::FIT_RAYON_STACK_SIZE`) for exactly this reason, so mirror it rather than shrink
/// a fixture that has to be wide to chunk at all.
#[test]
fn dcm_iov_form_c_readout_matches_fd_off_the_identity_chunk() {
    std::thread::Builder::new()
        .stack_size(crate::api::FIT_RAYON_STACK_SIZE)
        .spawn(dcm_iov_form_c_readout_body)
        .expect("spawn wide-stack test thread")
        .join()
        .expect("chunked DCM+IOV Form C readout test panicked");
}

fn dcm_iov_form_c_readout_body() {
    use crate::sens::provider::{iov_analytical_supported, subject_sensitivities_iov};

    let src = dcm_two_kappa_src()
        .replace(
            "[covariate_nn TYPICAL_PK]",
            "  theta TVBMAX(3.0, 0.01, 100.0)\n\n[covariate_nn TYPICAL_PK]",
        )
        .replace(
            "[error_model]",
            "[scaling]\n  y = central / V1 + TVBMAX * (central / V1) / (2.0 + central / V1)\n\n\
             [error_model]",
        );
    let model = parse_model_string(&src).expect("DCM+IOV+Form C readout parses");
    let subject = dcm_two_kappa_subject();
    let theta = dcm_two_kappa_theta(&model);

    assert!(
        model
            .analytic_readout
            .as_ref()
            .and_then(|ar| ar.program.as_ref())
            .is_some(),
        "fixture must carry a compiled Form C readout program"
    );
    let n_stacked =
        model.n_eta + crate::stats::likelihood::iov_occasion_groups(&subject).len() * model.n_kappa;
    assert!(
        model.n_theta + n_stacked > 24,
        "fixture must force a second θ chunk, or the chunked readout is untested"
    );
    assert!(
        iov_analytical_supported(&model),
        "a Form C readout no longer narrows the outer IOV gate (#1339)"
    );
    assert!(
        subject_sensitivities_iov(&model, &subject, &theta, &vec![0.05; n_stacked]).is_some(),
        "gate says analytic, so the chunked walk must serve the subject"
    );
    check_dcm_iov_outer_vs_fd(&model, &subject, &theta);
}

/// The exact band PR #1340's review named: a DCM whose weight block fits **one** occasion
/// (12 weights + 2 η + 1 κ = 15 ≤ 24, so any `K = 1` model-level bound reads analytic) but
/// whose real, many-occasion subjects stack past the cap (12 + 2 + 11 = 25) and need a
/// second θ chunk. With an `[initial_conditions]` impulse the first cut of #1339 declined
/// every such subject while the gate still advertised an analytic outer gradient — the
/// #637 route/report drift, and `K ≥ 2` is the normal case for IOV data, so it was not a
/// minority of subjects but all of them.
///
/// Both subjects are served now, and the 11-occasion one's `∂f/∂θ` is pinned against
/// central differences of `predict_iov`: a chunked impulse that kept seeding its direct
/// θ / η references on absolute axes writes another chunk's columns, which shows up here
/// as a θ-gradient mismatch rather than as a silent FD fallback.
#[test]
fn dcm_iov_chunked_init_impulse_serves_a_many_occasion_subject() {
    use crate::sens::provider::{iov_analytical_supported, subject_sensitivities_iov};

    const SMALL_DCM_IOV: &str = r#"
[parameters]
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.02
  sigma PROP ~ 0.04 (sd)

[covariate_nn TYPICAL_PK]
  inputs = [WT, CRCL]
  center = [70, 90]
  scale  = [15, 30]
  outputs = [CL, V]
  layers = [2]
  activation = tanh
  output = softplus
  init = [0.5, 20]

[individual_parameters]
  CL = TYPICAL_PK.CL * exp(ETA_CL + KAPPA_CL)
  V  = TYPICAL_PK.V  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)

[fit_options]
  method = focei
  iov_column = OCC
"#;
    let plain = parse_model_string(SMALL_DCM_IOV).expect("small DCM+IOV parses");
    let with_init = parse_model_string(&SMALL_DCM_IOV.replace(
        "[error_model]",
        "[initial_conditions]\n  init(central) = 0.1 * V\n\n[error_model]",
    ))
    .expect("small DCM+IOV+init parses");
    assert_eq!(plain.n_theta, 12, "2→2→2 network is 12 weights");
    for m in [&plain, &with_init] {
        assert!(
            m.n_theta + m.n_eta + m.n_kappa <= 24,
            "one occasion must fit the walk, or the model-level gate declines first"
        );
        assert!(
            iov_analytical_supported(m),
            "model-level gate is analytic for both"
        );
    }

    // Eleven occasions: 2 + 11 stacked axes + 12 θ = 25 > 24 → two chunks.
    let k = 11usize;
    let mut cov = HashMap::new();
    cov.insert("WT".to_string(), 72.0);
    cov.insert("CRCL".to_string(), 95.0);
    let mut doses = Vec::new();
    let mut obs_times = Vec::new();
    let mut occasions = Vec::new();
    let mut dose_occasions = Vec::new();
    for g in 0..k {
        let t0 = 24.0 * g as f64;
        doses.push(DoseEvent::new(t0, 100.0, 1, 0.0, false, 0.0));
        obs_times.push(t0 + 2.0);
        obs_times.push(t0 + 8.0);
        occasions.push((g + 1) as u32);
        occasions.push((g + 1) as u32);
        dose_occasions.push((g + 1) as u32);
    }
    let n = obs_times.len();
    let subject = Subject {
        id: "1".into(),
        doses,
        obs_times,
        obs_raw_times: Vec::new(),
        observations: vec![2.0; n],
        obs_cmts: vec![1; n],
        covariates: cov,
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        reset_occasions: Vec::new(),
        cens: vec![0; n],
        occasions,
        obs_l2: Vec::new(),
        dose_occasions,
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    let n_st = plain.n_eta + k * plain.n_kappa;
    assert!(plain.n_theta + n_st > 24, "subject must force chunking");
    let theta: Vec<f64> = plain
        .default_params
        .theta
        .iter()
        .enumerate()
        .map(|(i, &t)| t + 0.3 * ((i as f64) * 0.7 + 0.3).sin())
        .collect();
    let stacked = vec![0.05; n_st];

    assert!(
        subject_sensitivities_iov(&plain, &subject, &theta, &stacked).is_some(),
        "the plain network is served through two chunks"
    );
    let sens = subject_sensitivities_iov(&with_init, &subject, &theta, &stacked)
        .expect("the chunked walk must serve the init impulse on a many-occasion subject");

    // θ-gradient parity against central differences of `predict_iov`. First order only:
    // the mixed and η blocks are covered on the wider fixture by
    // `dcm_iov_init_impulse_matches_fd_off_the_identity_chunk`, and this test exists for
    // the `K = 11` geometry, not for a second copy of that sweep.
    let n_theta = with_init.n_theta;
    let pred = |th: &[f64], j: usize| -> f64 {
        let eta_bsv = stacked[..with_init.n_eta].to_vec();
        let kappas: Vec<Vec<f64>> = (0..k)
            .map(|g| {
                stacked[with_init.n_eta + g * with_init.n_kappa
                    ..with_init.n_eta + (g + 1) * with_init.n_kappa]
                    .to_vec()
            })
            .collect();
        crate::pk::predict_iov(&with_init, &subject, th, &eta_bsv, &kappas)[j]
    };
    let mut live = vec![false; n_theta];
    for (j, obs) in sens.obs.iter().enumerate() {
        approx::assert_relative_eq!(obs.f, pred(&theta, j), max_relative = 1e-9);
        for m in 0..n_theta {
            let s = 1e-6 * (1.0 + theta[m].abs());
            let mut tp = theta.clone();
            tp[m] += s;
            let mut tm = theta.clone();
            tm[m] -= s;
            let g = (pred(&tp, j) - pred(&tm, j)) / (2.0 * s);
            if g.abs() > 1e-3 {
                live[m] = true;
            }
            approx::assert_relative_eq!(obs.df_dtheta[m], g, max_relative = 2e-4, epsilon = 1e-7);
        }
    }
    // Columns 12.. do not exist here (12 weights), but a silent column inside the block
    // would make its own parity vacuous — the second chunk is `[11..12]`, one column, so
    // this is the assertion that keeps the chunked half of the sweep non-degenerate.
    let dead: Vec<usize> = (0..n_theta).filter(|&m| !live[m]).collect();
    assert!(
        dead.is_empty(),
        "weight columns {dead:?} of {n_theta} are silent on every observation"
    );
}
