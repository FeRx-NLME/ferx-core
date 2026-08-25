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
        let f_plus = obs_nll_subject_into(model, subject, &tp, sigma_values, eta, &mut scratch);
        tp[i] = theta[i] - h;
        let f_minus = obs_nll_subject_into(model, subject, &tp, sigma_values, eta, &mut scratch);
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
    let f0 = obs_nll_subject_into(&model, &subject, &theta, &sigma_values, &eta, &mut scratch);
    for i in 0..n_theta {
        let delta = 1e-5 * (1.0 + theta[i].abs());
        let mut tp = theta.clone();
        tp[i] += delta;
        let fp = obs_nll_subject_into(&model, &subject, &tp, &sigma_values, &eta, &mut scratch);
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
    assert!(
        !crate::sens::provider::iov_analytical_supported(&model),
        "and outside the full one — if this flips, the theta-axis accounting changed and \
         this test no longer exercises the eta-only route"
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

    // The two predicates must genuinely differ on this model, or the rest is vacuous.
    assert!(
        iov_sens_eta_supported(&model),
        "a DCM+IOV model is inside the η-only IOV scope"
    );
    assert!(
        !iov_sens_supported(&model),
        "and outside the strict one — if this flips, the θ-axis accounting changed and this \
         test no longer exercises the relaxation"
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
