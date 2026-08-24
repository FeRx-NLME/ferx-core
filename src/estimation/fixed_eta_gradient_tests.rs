//! FD-parity tests for the fixed-η observation-NLL gradients.
//!
//! Each test pins one branch of [`super::obs_nll_subject_grad`] /
//! [`super::obs_nll_subject_grad_iov`] against a forward finite difference of the
//! corresponding NLL evaluator in the same packed `[log_theta | log_sigma]` space.
//! These functions are shared by SAEM's M-step and variational inference, so a
//! wrong entry here is silently wrong in two estimators at once.

use super::*;
// The FD reference for the population-level checks: SAEM's M-step objective,
// which sums `obs_nll_subject_into` over subjects at fixed η.
use crate::estimation::saem::obs_nll_sum;
use crate::types::test_helpers::analytical_model;
use crate::types::GradientMethod;
/// `obs_nll_subject_grad` summed over subjects must match the reference
/// forward-FD of `obs_nll_sum` to within 1e-4 relative tolerance for all
/// non-pinned packed parameters (theta + sigma).
#[test]
fn obs_nll_subject_grad_matches_obs_nll_sum_fd() {
    use crate::types::{DoseEvent, Population};
    use std::collections::HashMap;

    let model = analytical_model(GradientMethod::Auto);

    let make_subj = |id: &str, obs: f64| Subject {
        id: id.into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![1.0, 4.0, 8.0],
        obs_raw_times: Vec::new(),
        observations: vec![obs, obs * 0.6, obs * 0.3],
        obs_cmts: vec![1, 1, 1],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0, 0, 0],
        occasions: vec![],
        obs_l2: Vec::new(),
        dose_occasions: vec![],
        fremtype: Vec::new(),
        obs_records: vec![],
    };

    let population = Population {
        subjects: vec![
            make_subj("1", 8.0),
            make_subj("2", 5.0),
            make_subj("3", 11.0),
        ],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };

    let theta = vec![1.5f64, 20.0]; // CL, V
    let sigma_values = vec![0.2f64]; // proportional
    let etas: Vec<Vec<f64>> = vec![vec![0.0], vec![0.1], vec![-0.1]];
    let n_theta = 2;
    let n_sigma = 1;
    let n = n_theta + n_sigma;

    // Compute reference gradient via forward-FD of obs_nll_sum.
    let f0 = obs_nll_sum(&model, &population, &theta, &sigma_values, &etas);
    let h = 1e-5;
    let mut ref_grad = vec![0.0f64; n];
    // Theta perturbations (in natural scale).
    for i in 0..n_theta {
        let mut theta_p = theta.clone();
        theta_p[i] += h;
        let fp = obs_nll_sum(&model, &population, &theta_p, &sigma_values, &etas);
        // FD in natural scale; convert to log-packed space (d/d_log = theta * d/d_theta)
        ref_grad[i] = theta[i] * (fp - f0) / h;
    }
    // Sigma perturbation (in natural scale; convert to log-packed).
    {
        let mut sigma_p = sigma_values.clone();
        sigma_p[0] += h;
        let fp = obs_nll_sum(&model, &population, &theta, &sigma_p, &etas);
        ref_grad[n_theta] = sigma_values[0] * (fp - f0) / h;
    }

    // Compute gradient via obs_nll_subject_grad summed over subjects.
    let mask: Vec<bool> = theta.iter().map(|_| true).collect(); // all log-packed
    let lo = vec![-1e30f64; n];
    let hi = vec![1e30f64; n];
    let mut total_nll = 0.0f64;
    let mut total_grad = vec![0.0f64; n];
    let mut scratch = EventPkParams::default();
    for (i, subject) in population.subjects.iter().enumerate() {
        let (nll_i, grad_i) = obs_nll_subject_grad(
            &model,
            subject,
            &theta,
            &sigma_values,
            &etas[i],
            &mask,
            &lo,
            &hi,
            n_theta,
            n_sigma,
            &mut scratch,
        );
        total_nll += nll_i;
        for (g, gi) in total_grad.iter_mut().zip(grad_i.iter()) {
            *g += gi;
        }
    }

    assert!(
        (total_nll - f0).abs() < 1e-10,
        "nll mismatch: {} vs {}",
        total_nll,
        f0
    );

    for j in 0..n {
        let rel = if ref_grad[j].abs() > 1e-10 {
            (total_grad[j] - ref_grad[j]).abs() / ref_grad[j].abs()
        } else {
            (total_grad[j] - ref_grad[j]).abs()
        };
        assert!(
            rel < 1e-4,
            "grad[{j}]: obs_nll_subject_grad={:.6e}, ref={:.6e}, rel={:.2e}",
            total_grad[j],
            ref_grad[j],
            rel
        );
    }
}

/// IOV M-step gradient (`obs_nll_subject_grad_iov`) must match the forward-FD
/// of `obs_nll_subject_into_iov` in log-packed space. This guards the
/// analytical gradient that the gradient-based M-step would use — it is not
/// exercised by the default BOBYQA M-step (derivative-free), so without this
/// direct test the function is untested. Single subject, 2 occasions, κ on CL.
#[test]
fn obs_nll_subject_grad_iov_matches_fd() {
    use crate::types::{
        BloqMethod, CompiledModel, DoseEvent, ErrorModel, ErrorSpec, GradientMethod,
        ModelParameters, OmegaMatrix, PkModel, PkParams, ScalingSpec, SigmaVector, Subject,
    };
    use std::collections::HashMap;

    // Minimal IOV model: CL = TVCL·exp(ETA_CL + KAPPA_CL), V = TVV.
    let model = CompiledModel {
        name: "iov_grad_test".into(),
        pk_model: PkModel::OneCptIv,
        error_model: ErrorModel::Proportional,
        error_spec: ErrorSpec::Single(ErrorModel::Proportional),
        residual_correlations: Vec::new(),
        pk_param_fn: Box::new(
            |theta: &[f64], eta: &[f64], _: &HashMap<String, f64>, _t: f64| {
                let mut p = PkParams::default();
                let kappa = if eta.len() > 1 { eta[1] } else { 0.0 };
                p.values[0] = theta[0] * (eta[0] + kappa).exp();
                p.values[1] = theta[1];
                p
            },
        ),
        n_theta: 2,
        n_eta: 1,
        n_epsilon: 1,
        n_kappa: 1,
        kappa_names: vec!["KAPPA_CL".into()],
        theta_names: vec!["TVCL".into(), "TVV".into()],
        eta_names: vec!["ETA_CL".into()],
        indiv_param_names: vec!["CL".into(), "V".into()],
        indiv_param_partials: crate::types::IndivParamPartials::empty(),
        default_params: ModelParameters {
            theta: vec![5.0, 50.0],
            theta_names: vec!["TVCL".into(), "TVV".into()],
            theta_lower: vec![0.1, 5.0],
            theta_upper: vec![50.0, 500.0],
            theta_fixed: vec![false; 2],
            omega: OmegaMatrix::from_diagonal(&[0.09], vec!["ETA_CL".into()]),
            omega_fixed: vec![false],
            sigma: SigmaVector {
                values: vec![0.05],
                names: vec!["PROP_ERR".into()],
            },
            sigma_fixed: vec![false],
            omega_iov: Some(OmegaMatrix::from_diagonal(&[0.04], vec!["KAPPA_CL".into()])),
            kappa_fixed: vec![false],
            mixture: None,
        },
        omega_init_as_sd: vec![false],
        sigma_init_as_sd: vec![false],
        kappa_init_as_sd: vec![false],
        kappa_weights: Vec::new(),
        mu_refs: HashMap::new(),
        kappa_mu_refs: HashMap::new(),
        tv_fn: None,
        pk_indices: vec![0, 1],
        eta_map: vec![0],
        pk_idx_f64: vec![0.0, 1.0],
        sel_flat: vec![1.0, 0.0],
        ode_spec: None,
        dose_attr_map: Default::default(),
        diffusion_theta_start: None,
        diffusion_state_indices: Vec::new(),
        bloq_method: BloqMethod::Drop,
        referenced_covariates: Vec::new(),
        gradient_method: GradientMethod::Fd,
        parse_warnings: Vec::new(),
        has_conditional_eta_params: false,
        eta_param_info: Vec::new(),
        theta_transform: Vec::new(),
        #[cfg(feature = "nn")]
        covariate_nns: Vec::new(),
        scaling: ScalingSpec::None,
        log_transform: false,
        dv_pre_logged: false,
        derived_exprs: Vec::new(),
        output_columns: Vec::new(),
        #[cfg(feature = "survival")]
        endpoints: HashMap::new(),
        frem_config: None,
        residual_error_eta: None,
        analytical_init: Vec::new(),
        analytic_readout: None,
        ruv_magnitude: None,
        absorption_ode_equivalent: None,
        mixture: None,
    };

    // One subject, 2 occasions (times 1–3 occ 1, 4–6 occ 2), one dose each.
    let subject = Subject {
        id: "S1".into(),
        doses: vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(3.5, 100.0, 1, 0.0, false, 0.0),
        ],
        obs_times: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        obs_raw_times: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        observations: vec![36.0, 28.0, 21.0, 34.0, 26.0, 19.0],
        obs_cmts: vec![1; 6],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0; 6],
        occasions: vec![1, 1, 1, 2, 2, 2],
        obs_l2: Vec::new(),
        dose_occasions: vec![1, 2],
        fremtype: Vec::new(),
        obs_records: Vec::new(),
    };

    let theta = vec![5.0f64, 50.0];
    let sigma = vec![0.05f64];
    let eta = vec![0.1f64];
    let kappas: Vec<Vec<f64>> = vec![vec![0.05], vec![-0.05]]; // one per occasion
    let n_theta = 2;
    let n_sigma = 1;
    let n = n_theta + n_sigma;

    let mut scratch = EventPkParams::default();
    let (nll, grad) = obs_nll_subject_grad_iov(
        &model,
        &subject,
        &theta,
        &sigma,
        &eta,
        &kappas,
        &[true, true, true],
        &[-1e30; 3],
        &[1e30; 3],
        n_theta,
        n_sigma,
        &mut scratch,
    );

    // Reference: forward-FD of obs_nll_subject_into_iov in log-packed space.
    let f0 = obs_nll_subject_into_iov(
        &model,
        &subject,
        &theta,
        &sigma,
        &eta,
        &kappas,
        &mut scratch,
    );
    assert!((nll - f0).abs() < 1e-10, "nll mismatch: {nll} vs {f0}");

    let h = 1e-6;
    let mut ref_grad = vec![0.0f64; n];
    for i in 0..n_theta {
        let mut tp = theta.clone();
        tp[i] += h;
        let fp =
            obs_nll_subject_into_iov(&model, &subject, &tp, &sigma, &eta, &kappas, &mut scratch);
        ref_grad[i] = theta[i] * (fp - f0) / h; // d/d_log = theta · d/d_theta
    }
    {
        let mut sp = sigma.clone();
        sp[0] += h;
        let fp =
            obs_nll_subject_into_iov(&model, &subject, &theta, &sp, &eta, &kappas, &mut scratch);
        ref_grad[n_theta] = sigma[0] * (fp - f0) / h;
    }

    for j in 0..n {
        let rel = if ref_grad[j].abs() > 1e-8 {
            (grad[j] - ref_grad[j]).abs() / ref_grad[j].abs()
        } else {
            (grad[j] - ref_grad[j]).abs()
        };
        assert!(
            rel < 1e-4,
            "grad[{j}]: analytical={:.6e}, fd={:.6e}, rel={:.2e}",
            grad[j],
            ref_grad[j],
            rel
        );
    }
}

/// Per-CMT (multi-endpoint) M-step gradient must match the forward-FD of
/// `obs_nll_sum` — the correctness gate for the per-CMT `dvar_df` /
/// `dvar_dlogsigma` score terms. Two endpoints with *different* error
/// models (proportional PK on CMT=1, additive PD on CMT=2) so a single
/// error model would give the wrong Jacobian for one endpoint.
#[test]
fn obs_nll_subject_grad_per_cmt_matches_fd() {
    use crate::parser::model_parser::parse_model_string;
    use crate::types::{DoseEvent, Population};
    use std::collections::HashMap;

    let model = parse_model_string(
        r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  theta TVKE0(0.5, 0.05, 5.0)
  omega ETA_CL ~ 0.04
  sigma PROP_ERR_PK ~ 0.10 (sd)
  sigma ADD_ERR_PD  ~ 0.50 (sd)

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  KE0 = TVKE0

[structural_model]
  ode(states=[central, effect])

[odes]
  d/dt(central) = -CL/V * central
  d/dt(effect)  =  KE0 * (central/V - effect)

[scaling]
  y[CMT=1] = central / V
  y[CMT=2] = effect

[error_model]
  CMT=1: DV ~ proportional(PROP_ERR_PK)
  CMT=2: DV ~ additive(ADD_ERR_PD)
",
    )
    .expect("per-CMT ODE model parses");

    // obs at CMT=1 (PK) and CMT=2 (PD), interleaved.
    let make_subj = |id: &str, scale: f64| Subject {
        id: id.into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![1.0, 1.0, 2.0, 2.0, 4.0, 4.0],
        obs_raw_times: Vec::new(),
        observations: vec![
            8.0 * scale,
            2.0 * scale,
            6.0 * scale,
            3.0 * scale,
            4.0 * scale,
            3.5 * scale,
        ],
        obs_cmts: vec![1, 2, 1, 2, 1, 2],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0; 6],
        occasions: vec![],
        obs_l2: Vec::new(),
        dose_occasions: vec![],
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    let population = Population {
        subjects: vec![make_subj("1", 1.0), make_subj("2", 1.1)],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };

    let theta = vec![1.0f64, 10.0, 0.5];
    let sigma_values = vec![0.10f64, 0.50];
    let etas: Vec<Vec<f64>> = vec![vec![0.0], vec![0.05]];
    let n_theta = 3;
    let n_sigma = 2;
    let n = n_theta + n_sigma;

    // Reference gradient: forward-FD of obs_nll_sum, in log-packed space.
    let f0 = obs_nll_sum(&model, &population, &theta, &sigma_values, &etas);
    let h = 1e-6;
    let mut ref_grad = vec![0.0f64; n];
    for i in 0..n_theta {
        let mut tp = theta.clone();
        tp[i] += h;
        let fp = obs_nll_sum(&model, &population, &tp, &sigma_values, &etas);
        ref_grad[i] = theta[i] * (fp - f0) / h;
    }
    for k in 0..n_sigma {
        let mut sp = sigma_values.clone();
        sp[k] += h;
        let fp = obs_nll_sum(&model, &population, &theta, &sp, &etas);
        ref_grad[n_theta + k] = sigma_values[k] * (fp - f0) / h;
    }

    // Analytical gradient: sum of per-subject obs_nll_subject_grad.
    let mask = vec![true; n_theta];
    let lo = vec![-1e30f64; n];
    let hi = vec![1e30f64; n];
    let mut total_nll = 0.0f64;
    let mut total_grad = vec![0.0f64; n];
    let mut scratch = EventPkParams::default();
    for (i, subject) in population.subjects.iter().enumerate() {
        let (nll_i, grad_i) = obs_nll_subject_grad(
            &model,
            subject,
            &theta,
            &sigma_values,
            &etas[i],
            &mask,
            &lo,
            &hi,
            n_theta,
            n_sigma,
            &mut scratch,
        );
        total_nll += nll_i;
        for (g, gi) in total_grad.iter_mut().zip(grad_i.iter()) {
            *g += gi;
        }
    }

    assert!(
        (total_nll - f0).abs() < 1e-8,
        "nll mismatch: {total_nll} vs {f0}"
    );
    for j in 0..n {
        let rel = if ref_grad[j].abs() > 1e-8 {
            (total_grad[j] - ref_grad[j]).abs() / ref_grad[j].abs()
        } else {
            (total_grad[j] - ref_grad[j]).abs()
        };
        assert!(
            rel < 1e-3,
            "per-CMT grad[{j}]: analytical={:.6e}, fd={:.6e}, rel={:.2e}",
            total_grad[j],
            ref_grad[j],
            rel
        );
    }
}

/// Dense residual-covariance M-step gradient must match FD of the same
/// dense observation NLL. This exercises the `block_sigma` SAEM path, which
/// deliberately routes through full FD because the analytic scalar-RUV score
/// terms do not apply to off-diagonal R blocks.
#[test]
fn obs_nll_subject_grad_block_sigma_cross_endpoint_matches_fd() {
    use crate::parser::model_parser::parse_model_string;
    use crate::types::{DoseEvent, Population};
    use std::collections::HashMap;

    let model = parse_model_string(
        r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.04
  block_sigma (PROP_ERR_UNBOUND, PROP_ERR_TOTAL) = [
0.04,
0.01, 0.09
  ]

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = -CL/V * central

[scaling]
  y[CMT=1] = 2.0 * central / V
  y[CMT=2] = central / V

[error_model]
  CMT=1: DV ~ proportional(PROP_ERR_TOTAL)
  CMT=2: DV ~ proportional(PROP_ERR_UNBOUND)
",
    )
    .expect("cross-endpoint block_sigma ODE model parses");

    let subject = Subject {
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![1.0, 1.0, 2.0, 2.0],
        obs_raw_times: Vec::new(),
        observations: vec![17.0, 8.0, 15.0, 7.0],
        obs_cmts: vec![1, 2, 1, 2],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0; 4],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    let population = Population {
        subjects: vec![subject.clone()],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };

    let theta = vec![1.0f64, 10.0];
    let sigma_values = vec![0.20f64, 0.30];
    let etas: Vec<Vec<f64>> = vec![vec![0.05]];
    let n_theta = 2;
    let n_sigma = 2;
    let n = n_theta + n_sigma;

    let f0 = obs_nll_sum(&model, &population, &theta, &sigma_values, &etas);
    let h = 1e-6;
    let mut ref_grad = vec![0.0f64; n];
    for i in 0..n_theta {
        let mut tp = theta.clone();
        tp[i] += h;
        let fp = obs_nll_sum(&model, &population, &tp, &sigma_values, &etas);
        ref_grad[i] = theta[i] * (fp - f0) / h;
    }
    for k in 0..n_sigma {
        let mut sp = sigma_values.clone();
        sp[k] += h;
        let fp = obs_nll_sum(&model, &population, &theta, &sp, &etas);
        ref_grad[n_theta + k] = sigma_values[k] * (fp - f0) / h;
    }

    let mask = vec![true; n_theta];
    let lo = vec![-1e30f64; n];
    let hi = vec![1e30f64; n];
    let mut scratch = EventPkParams::default();
    let (nll, grad) = obs_nll_subject_grad(
        &model,
        &subject,
        &theta,
        &sigma_values,
        &etas[0],
        &mask,
        &lo,
        &hi,
        n_theta,
        n_sigma,
        &mut scratch,
    );

    assert!((nll - f0).abs() < 1e-8, "nll mismatch: {nll} vs {f0}");
    for j in 0..n {
        let rel = if ref_grad[j].abs() > 1e-8 {
            (grad[j] - ref_grad[j]).abs() / ref_grad[j].abs()
        } else {
            (grad[j] - ref_grad[j]).abs()
        };
        assert!(
            rel < 1e-4,
            "block_sigma grad[{j}]: fd-path={:.6e}, ref={:.6e}, rel={:.2e}",
            grad[j],
            ref_grad[j],
            rel
        );
    }
}

// ── #484/#1029: the residual magnitude reaches the SAEM M-step ──────────
//
// `obs_nll_sum` routes through `likelihood::obs_nll_subject_into`, the same
// magnitude-aware data term FOCE/FOCEI score. So asserting the M-step's own
// `nll` against it *is* the cross-estimator likelihood-agreement check, and
// FD of it pins the θ/σ score terms — including the magnitude's direct-θ
// channel, which the prediction chain rule alone would drop.

/// Shared body: analytic M-step `(nll, grad)` for `model` vs `obs_nll_sum`
/// and its forward difference, in the same log-packed space SAEM optimises.
fn check_saem_mstep_matches_fd(model: &CompiledModel, theta: &[f64], sigma_values: &[f64]) {
    use crate::types::{DoseEvent, Population};

    let make_subj = |id: &str, wpse: f64, scale: f64| Subject {
        id: id.into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![1.0, 4.0, 8.0],
        observations: vec![8.0 * scale, 6.0 * scale, 4.0 * scale],
        obs_cmts: vec![1; 3],
        cens: vec![0; 3],
        // WPSE varies within the subject: a per-record snapshot, so a
        // magnitude frozen at the subject's first value would be caught.
        covariates: [("WPSE".to_string(), wpse)].into_iter().collect(),
        obs_covariates: vec![
            [("WPSE".to_string(), wpse)].into_iter().collect(),
            [("WPSE".to_string(), wpse * 1.5)].into_iter().collect(),
            [("WPSE".to_string(), wpse * 2.0)].into_iter().collect(),
        ],
        ..Default::default()
    };
    let population = Population {
        subjects: vec![make_subj("1", 0.5, 1.0), make_subj("2", 0.8, 1.1)],
        covariate_names: vec!["WPSE".to_string()],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };

    let etas: Vec<Vec<f64>> = vec![vec![0.0], vec![0.05]];
    let n_theta = theta.len();
    let n_sigma = sigma_values.len();
    let n = n_theta + n_sigma;

    let f0 = obs_nll_sum(model, &population, theta, sigma_values, &etas);
    let h = 1e-6;
    let mut ref_grad = vec![0.0f64; n];
    for i in 0..n_theta {
        let mut tp = theta.to_vec();
        tp[i] += h;
        ref_grad[i] =
            theta[i] * (obs_nll_sum(model, &population, &tp, sigma_values, &etas) - f0) / h;
    }
    for k in 0..n_sigma {
        let mut sp = sigma_values.to_vec();
        sp[k] += h;
        ref_grad[n_theta + k] =
            sigma_values[k] * (obs_nll_sum(model, &population, theta, &sp, &etas) - f0) / h;
    }

    let mask = vec![true; n_theta];
    let lo = vec![-1e30f64; n];
    let hi = vec![1e30f64; n];
    let mut total_nll = 0.0f64;
    let mut total_grad = vec![0.0f64; n];
    let mut scratch = EventPkParams::default();
    for (i, subject) in population.subjects.iter().enumerate() {
        let (nll_i, grad_i) = obs_nll_subject_grad(
            model,
            subject,
            theta,
            sigma_values,
            &etas[i],
            &mask,
            &lo,
            &hi,
            n_theta,
            n_sigma,
            &mut scratch,
        );
        total_nll += nll_i;
        for (g, gi) in total_grad.iter_mut().zip(grad_i.iter()) {
            *g += gi;
        }
    }

    assert!(
        (total_nll - f0).abs() < 1e-8,
        "M-step NLL disagrees with the shared magnitude-aware data term: \
         {total_nll} vs {f0}"
    );
    for j in 0..n {
        let rel = if ref_grad[j].abs() > 1e-8 {
            (total_grad[j] - ref_grad[j]).abs() / ref_grad[j].abs()
        } else {
            (total_grad[j] - ref_grad[j]).abs()
        };
        assert!(
            rel < 1e-3,
            "weighted M-step grad[{j}]: analytic={:.6e}, fd={:.6e}, rel={:.2e}",
            total_grad[j],
            ref_grad[j],
            rel
        );
    }
}

/// `weight = <covariate>` (#1029): θ-free, so the analytic prediction chain
/// rule stays exact once V / ∂V∂f / ∂V∂logσ take their `_scaled` forms.
#[test]
fn obs_nll_subject_grad_weighted_error_matches_fd() {
    let model = crate::parser::model_parser::parse_model_string(
        r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.04
  sigma PROP_ERR ~ 0.10 (sd)
  sigma ADD_ERR  ~ 0.50 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ combined(PROP_ERR * (1.0 + 0.5 * WPSE), ADD_ERR) weight = WPSE
[covariates]
  WPSE continuous
",
    )
    .expect("weighted model parses");
    assert!(model.has_custom_ruv_magnitude());
    assert!(
        !model.has_theta_dependent_ruv_magnitude(),
        "a covariate weight must not be flagged θ-dependent"
    );
    check_saem_mstep_matches_fd(&model, &[1.0, 10.0], &[0.10, 0.50]);
}

/// θ-*dependent* magnitude (#484): θ now moves the residual variance
/// directly as well as through the prediction. The M-step θ gradient must
/// carry both channels — the analytic `∂nll/∂f · ∂f/∂θ` chain alone fails
/// this test.
#[test]
fn obs_nll_subject_grad_theta_dependent_magnitude_matches_fd() {
    let model = crate::parser::model_parser::parse_model_string(
        r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  theta RUV_W(0.30, 0.01, 5.0)
  omega ETA_CL ~ 0.04
  sigma PROP_ERR ~ 0.10 (sd)
  sigma ADD_ERR  ~ 0.50 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR * (1.0 + RUV_W * WPSE))
[covariates]
  WPSE continuous
",
    )
    .expect("theta-dependent magnitude model parses");
    assert!(model.has_theta_dependent_ruv_magnitude());
    check_saem_mstep_matches_fd(&model, &[1.0, 10.0, 0.30], &[0.10, 0.50]);
}
