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
        reset_covariates: Vec::new(),
        cens: vec![0, 0, 0],
        occasions: vec![],
        obs_l2: Vec::new(),
        dose_occasions: vec![],
        reset_occasions: Vec::new(),
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
    let f0 = obs_nll_sum(&model, &population, &theta, &sigma_values, &etas, &[]);
    let h = 1e-5;
    let mut ref_grad = vec![0.0f64; n];
    // Theta perturbations (in natural scale).
    for i in 0..n_theta {
        let mut theta_p = theta.clone();
        theta_p[i] += h;
        let fp = obs_nll_sum(&model, &population, &theta_p, &sigma_values, &etas, &[]);
        // FD in natural scale; convert to log-packed space (d/d_log = theta * d/d_theta)
        ref_grad[i] = theta[i] * (fp - f0) / h;
    }
    // Sigma perturbation (in natural scale; convert to log-packed).
    {
        let mut sigma_p = sigma_values.clone();
        sigma_p[0] += h;
        let fp = obs_nll_sum(&model, &population, &theta, &sigma_p, &etas, &[]);
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
        priors: Vec::new(),
        prior_from_fit: None,
        covariate_model: None,
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
            residual_correlations: Vec::new(),
            residual_correlation_fixed: Vec::new(),
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
        covariate_mu_refs: Vec::new(),
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
        theta_eta_linked: Vec::new(),
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
        reset_covariates: Vec::new(),
        cens: vec![0; 6],
        occasions: vec![1, 1, 1, 2, 2, 2],
        obs_l2: Vec::new(),
        dose_occasions: vec![1, 2],
        reset_occasions: Vec::new(),
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
        reset_covariates: Vec::new(),
        cens: vec![0; 6],
        occasions: vec![],
        obs_l2: Vec::new(),
        dose_occasions: vec![],
        reset_occasions: Vec::new(),
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
    let f0 = obs_nll_sum(&model, &population, &theta, &sigma_values, &etas, &[]);
    let h = 1e-6;
    let mut ref_grad = vec![0.0f64; n];
    for i in 0..n_theta {
        let mut tp = theta.clone();
        tp[i] += h;
        let fp = obs_nll_sum(&model, &population, &tp, &sigma_values, &etas, &[]);
        ref_grad[i] = theta[i] * (fp - f0) / h;
    }
    for k in 0..n_sigma {
        let mut sp = sigma_values.clone();
        sp[k] += h;
        let fp = obs_nll_sum(&model, &population, &theta, &sp, &etas, &[]);
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
        reset_covariates: Vec::new(),
        cens: vec![0; 4],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
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

    let f0 = obs_nll_sum(&model, &population, &theta, &sigma_values, &etas, &[]);
    let h = 1e-6;
    let mut ref_grad = vec![0.0f64; n];
    for i in 0..n_theta {
        let mut tp = theta.clone();
        tp[i] += h;
        let fp = obs_nll_sum(&model, &population, &tp, &sigma_values, &etas, &[]);
        ref_grad[i] = theta[i] * (fp - f0) / h;
    }
    for k in 0..n_sigma {
        let mut sp = sigma_values.clone();
        sp[k] += h;
        let fp = obs_nll_sum(&model, &population, &theta, &sp, &etas, &[]);
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

    let f0 = obs_nll_sum(model, &population, theta, sigma_values, &etas, &[]);
    let h = 1e-6;
    let mut ref_grad = vec![0.0f64; n];
    for i in 0..n_theta {
        let mut tp = theta.to_vec();
        tp[i] += h;
        ref_grad[i] =
            theta[i] * (obs_nll_sum(model, &population, &tp, sigma_values, &etas, &[]) - f0) / h;
    }
    for k in 0..n_sigma {
        let mut sp = sigma_values.to_vec();
        sp[k] += h;
        ref_grad[n_theta + k] =
            sigma_values[k] * (obs_nll_sum(model, &population, theta, &sp, &etas, &[]) - f0) / h;
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

/// #1182: the `power(σ, P)` exponent is a θ that moves the residual variance
/// directly; the M-step θ gradient differences the whole magnitude-aware
/// data term, and the σ gradient reads `dvar_dlogsigma_scaled`'s exponent arm.
#[test]
fn obs_nll_subject_grad_power_exponent_matches_fd() {
    let model = crate::parser::model_parser::parse_model_string(
        r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  theta RUV_POW(1.3, 0.01, 10.0)
  omega ETA_CL ~ 0.04
  sigma PROP_ERR ~ 0.10 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ power(PROP_ERR, RUV_POW)
[covariates]
  WPSE continuous
",
    )
    .expect("power model parses");
    assert!(model.has_ruv_exponent());
    check_saem_mstep_matches_fd(&model, &[1.0, 10.0, 1.3], &[0.10]);
}

// ── #1458: the expected (Fisher) information of the frozen-η M-step ────────
//
// `obs_nll_subject_grad_fisher` returns `E[∂²(−log L)/∂x_a∂x_b]` for the
// Gaussian observation model at fixed η. That expectation is over the data, so
// it is *not* the observed Hessian of any one dataset — which is exactly what
// makes a naive "compare it to a finite-difference Hessian" check wrong, and
// what the fixture below is built to fix.
//
// Writing the observed Hessian out (r = y − f, V the residual variance):
//
//   ∂²nll/∂a∂b = ½[ V_ab/V − V_aV_b/V² + 2 f_a f_b/V − r²(V_ab/V² − 2V_aV_b/V³)
//                   − 2r f_ab/V + 2r f_a V_b/V² + 2r f_b V_a/V² ]
//
// and the terms that separate it from the information are the ones carrying a
// bare `r` (they vanish under `E[r] = 0`) and the ones carrying `r²` (they need
// `E[r²] = V`). So a dataset in which **every observation time appears twice,
// with residuals `+√V` and `−√V`**, makes the two coincide *exactly*, not in
// expectation: the `r`-linear terms cancel between the twins and each twin
// contributes `r² = V`. A central finite difference of the real objective is
// then an oracle for the information, computed outside the routine under test.

/// Build a subject whose observations are paired `f ± √V` at every time, so the
/// observed Hessian of `obs_nll_sum` equals the expected information exactly.
/// Returns the subject.
fn paired_residual_subject(
    model: &CompiledModel,
    id: &str,
    doses: Vec<DoseEvent>,
    times: &[f64],
    theta: &[f64],
    sigma_values: &[f64],
    eta: &[f64],
) -> Subject {
    use crate::types::Subject;
    // First pass: predictions at the *paired* times, with placeholder DVs.
    let mut obs_times = Vec::new();
    for &t in times {
        obs_times.push(t);
        obs_times.push(t);
    }
    let n = obs_times.len();
    let mut subject = Subject {
        id: id.into(),
        doses,
        obs_times: obs_times.clone(),
        observations: vec![0.0; n],
        obs_cmts: vec![1; n],
        cens: vec![0; n],
        ..Default::default()
    };
    let mut scratch = EventPkParams::default();
    let preds =
        crate::pk::compute_predictions_with_tv_into(model, &subject, theta, eta, &mut scratch);
    let keys: Vec<usize> = model.error_spec.obs_keys(&subject).to_vec();
    for j in 0..n {
        let f = model.floor_prediction(preds[j]);
        let v = model.residual_variance_at_scaled(keys[j], f, sigma_values, None);
        let s = if j % 2 == 0 { 1.0 } else { -1.0 };
        subject.observations[j] = f + s * v.sqrt();
    }
    subject
}

/// Central-difference Hessian of `obs_nll_sum` in the packed
/// `[log θ | log σ]` space, row-major `n × n`.
fn fd_packed_hessian(
    model: &CompiledModel,
    population: &Population,
    theta: &[f64],
    sigma_values: &[f64],
    etas: &[Vec<f64>],
    h: f64,
) -> Vec<f64> {
    let n_theta = theta.len();
    let n_sigma = sigma_values.len();
    let n = n_theta + n_sigma;
    let mut x: Vec<f64> = theta.iter().map(|v| v.ln()).collect();
    x.extend(sigma_values.iter().map(|v| v.ln()));
    let at = |x: &[f64]| -> f64 {
        let th: Vec<f64> = x[..n_theta].iter().map(|v| v.exp()).collect();
        let sg: Vec<f64> = x[n_theta..].iter().map(|v| v.exp()).collect();
        obs_nll_sum(model, population, &th, &sg, etas, &[])
    };
    let mut out = vec![0.0f64; n * n];
    for a in 0..n {
        for b in a..n {
            let mut xpp = x.clone();
            xpp[a] += h;
            xpp[b] += h;
            let mut xpm = x.clone();
            xpm[a] += h;
            xpm[b] -= h;
            let mut xmp = x.clone();
            xmp[a] -= h;
            xmp[b] += h;
            let mut xmm = x.clone();
            xmm[a] -= h;
            xmm[b] -= h;
            let v = (at(&xpp) - at(&xpm) - at(&xmp) + at(&xmm)) / (4.0 * h * h);
            out[a * n + b] = v;
            out[b * n + a] = v;
        }
    }
    out
}

fn combined_error_two_cpt_model() -> CompiledModel {
    crate::parser::model_parser::parse_model_string(
        r"
[parameters]
  theta TVCL(1.4, 0.1, 10.0)
  theta TVV(11.0, 1.0, 100.0)
  theta TVQ(2.1, 0.1, 50.0)
  theta TVV2(23.0, 1.0, 300.0)
  omega ETA_CL ~ 0.06
  sigma PROP_ERR ~ 0.15 (sd)
  sigma ADD_ERR  ~ 0.40 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  Q  = TVQ
  V2 = TVV2
[structural_model]
  pk two_cpt_iv(cl=CL, v1=V, q=Q, v2=V2)
[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR)
",
    )
    .expect("fixture model parses")
}

/// The information must equal the observed Hessian on the paired-residual
/// fixture, block for block — including the θσ cross block, which a `∂f∂f/V`
/// implementation that forgot the variance channel would get wrong.
///
/// The realised worst relative disagreement on this fixture is **2.1e-5**,
/// dominated by the forward finite difference the θ derivatives are built from
/// (`h = 1e-5·(1+|θ|)`); the reference's own central second difference is two
/// orders tighter. The bound below is 10× that, measured rather than argued.
#[test]
fn fisher_information_matches_the_observed_hessian_on_paired_residuals() {
    use crate::types::{DoseEvent, Population};
    let model = combined_error_two_cpt_model();
    let theta = vec![1.4f64, 11.0, 2.1, 23.0];
    let sigma_values = vec![0.15f64, 0.40];
    let n_theta = theta.len();
    let n_sigma = sigma_values.len();
    let n = n_theta + n_sigma;

    let times = [0.5f64, 2.0, 6.0, 12.0, 24.0];
    let etas = vec![vec![0.0f64], vec![0.12], vec![-0.08]];
    let subjects: Vec<Subject> = etas
        .iter()
        .enumerate()
        .map(|(i, eta)| {
            paired_residual_subject(
                &model,
                &format!("{}", i + 1),
                vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                &times,
                &theta,
                &sigma_values,
                eta,
            )
        })
        .collect();
    let population = Population {
        subjects,
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };

    let mask = vec![true; n_theta];
    let lo = vec![-1e30f64; n];
    let hi = vec![1e30f64; n];
    let mut scratch = EventPkParams::default();
    let mut info = vec![0.0f64; n * n];
    for (i, subject) in population.subjects.iter().enumerate() {
        let (_, _, fi) = obs_nll_subject_grad_fisher(
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
        let fi = fi.expect("this fixture is inside the Gaussian scope");
        for (a, v) in fi.iter().enumerate() {
            info[a] += v;
        }
    }

    let href = fd_packed_hessian(&model, &population, &theta, &sigma_values, &etas, 1e-4);

    // Non-degeneracy first: an all-zero information would pass any tolerance
    // against an all-zero Hessian, and a fixture whose cross block is dead
    // cannot see a missing variance channel.
    let scale = href.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(
        scale > 1.0,
        "reference Hessian is degenerate: max {scale:.3e}"
    );
    for a in 0..n {
        assert!(
            info[a * n + a].abs() > 1e-6 * scale,
            "coordinate {a} carries no information on this fixture — the check cannot see it"
        );
    }
    // The θσ cross block must be live, or the `½ V_a V_b / V²` term is untested.
    let cross = (0..n_theta)
        .flat_map(|a| (n_theta..n).map(move |b| (a, b)))
        .fold(0.0f64, |m, (a, b)| m.max(info[a * n + b].abs()));
    assert!(
        cross > 1e-3 * scale,
        "theta-sigma cross block is dead ({cross:.3e}) — use a combined error model"
    );

    let mut worst = 0.0f64;
    let mut worst_at = (0usize, 0usize);
    for a in 0..n {
        for b in 0..n {
            let got = info[a * n + b];
            let want = href[a * n + b];
            // `f64::max` would swallow a NaN and leave `worst` at whatever the
            // finite entries produced, so reject non-finite entries outright.
            assert!(
                got.is_finite() && want.is_finite(),
                "non-finite entry at ({a},{b}): info={got}, ref={want}"
            );
            let rel = (got - want).abs() / want.abs().max(1e-3 * scale);
            if rel > worst {
                worst = rel;
                worst_at = (a, b);
            }
        }
    }
    assert!(
        worst < 2.1e-4,
        "expected information disagrees with the paired-residual observed Hessian: \
         worst relative {worst:.3e} at {worst_at:?} (info {:.6e} vs ref {:.6e})",
        info[worst_at.0 * n + worst_at.1],
        href[worst_at.0 * n + worst_at.1]
    );
}

/// The information is symmetric and positive semi-definite by construction —
/// which is the property that makes it usable as Newton curvature where the
/// observed Hessian is not.
#[test]
fn fisher_information_is_symmetric_and_psd() {
    use crate::types::{DoseEvent, Population};
    let model = combined_error_two_cpt_model();
    let theta = vec![1.4f64, 11.0, 2.1, 23.0];
    let sigma_values = vec![0.15f64, 0.40];
    let n_theta = theta.len();
    let n_sigma = sigma_values.len();
    let n = n_theta + n_sigma;
    let eta = vec![0.2f64];
    let subject = paired_residual_subject(
        &model,
        "1",
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &[0.5, 2.0, 6.0, 12.0, 24.0],
        &theta,
        &sigma_values,
        &eta,
    );
    let _ = Population {
        subjects: vec![subject.clone()],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };
    let mask = vec![true; n_theta];
    let lo = vec![-1e30f64; n];
    let hi = vec![1e30f64; n];
    let mut scratch = EventPkParams::default();
    let (_, _, fi) = obs_nll_subject_grad_fisher(
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
    let fi = fi.expect("in scope");
    for a in 0..n {
        for b in 0..n {
            assert_eq!(
                fi[a * n + b],
                fi[b * n + a],
                "information is not symmetric at ({a},{b})"
            );
        }
    }
    let m = nalgebra::DMatrix::from_fn(n, n, |a, b| fi[a * n + b]);
    let eig = m.symmetric_eigenvalues();
    let min = eig.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = eig.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    assert!(
        max > 1.0,
        "degenerate fixture: largest eigenvalue {max:.3e}"
    );
    assert!(
        min > -1e-8 * max,
        "information is not PSD: smallest eigenvalue {min:.3e} against largest {max:.3e}"
    );
}

/// A pinned coordinate keeps an all-zero row and column — the property the
/// caller's linear solve relies on to leave a FIXed or mu-referenced θ alone.
#[test]
fn fisher_information_zeroes_a_pinned_coordinate() {
    use crate::types::DoseEvent;
    let model = combined_error_two_cpt_model();
    let theta = vec![1.4f64, 11.0, 2.1, 23.0];
    let sigma_values = vec![0.15f64, 0.40];
    let n_theta = theta.len();
    let n_sigma = sigma_values.len();
    let n = n_theta + n_sigma;
    let eta = vec![0.0f64];
    let subject = paired_residual_subject(
        &model,
        "1",
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &[0.5, 2.0, 6.0, 12.0],
        &theta,
        &sigma_values,
        &eta,
    );
    let mask = vec![true; n_theta];
    let mut lo = vec![-1e30f64; n];
    let mut hi = vec![1e30f64; n];
    // Pin TVQ (index 2) and ADD_ERR (index n_theta + 1).
    lo[2] = 0.0;
    hi[2] = 0.0;
    lo[n_theta + 1] = 0.0;
    hi[n_theta + 1] = 0.0;
    let mut scratch = EventPkParams::default();
    let (_, grad, fi) = obs_nll_subject_grad_fisher(
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
    let fi = fi.expect("in scope");
    for &p in &[2usize, n_theta + 1] {
        assert_eq!(grad[p], 0.0, "pinned coordinate {p} has a score");
        for b in 0..n {
            assert_eq!(fi[p * n + b], 0.0, "pinned row {p} column {b} is non-zero");
            assert_eq!(fi[b * n + p], 0.0, "pinned column {p} row {b} is non-zero");
        }
    }
    // The free coordinates still carry information — otherwise this test would
    // pass on an implementation that returned an all-zero matrix.
    assert!(
        fi[0] > 0.0 && fi[n_theta * n + n_theta] > 0.0,
        "free coordinates lost their information: {:.3e}, {:.3e}",
        fi[0],
        fi[n_theta * n + n_theta]
    );
}

/// Out of the Gaussian scope the *gradient* is still returned and still exact;
/// only the information is withheld, so the caller falls back loudly instead of
/// stepping on a formula that does not hold. A residual magnitude is the case
/// closest to in-scope — θ reaches `V` through a second channel — so it is the
/// one worth pinning.
#[test]
fn fisher_information_is_withheld_outside_the_gaussian_scope() {
    use crate::types::DoseEvent;
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
    .expect("magnitude fixture parses");
    assert!(model.has_custom_ruv_magnitude());
    let theta = vec![1.0f64, 10.0];
    let sigma_values = vec![0.10f64, 0.50];
    let n_theta = theta.len();
    let n_sigma = sigma_values.len();
    let n = n_theta + n_sigma;
    let subject = Subject {
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![1.0, 4.0, 8.0],
        observations: vec![8.0, 5.0, 3.0],
        obs_cmts: vec![1; 3],
        cens: vec![0; 3],
        covariates: [("WPSE".to_string(), 0.7)].into_iter().collect(),
        obs_covariates: vec![
            [("WPSE".to_string(), 0.7)].into_iter().collect(),
            [("WPSE".to_string(), 0.9)].into_iter().collect(),
            [("WPSE".to_string(), 1.3)].into_iter().collect(),
        ],
        ..Default::default()
    };
    let mask = vec![true; n_theta];
    let lo = vec![-1e30f64; n];
    let hi = vec![1e30f64; n];
    let mut scratch = EventPkParams::default();
    let (nll, grad, fi) = obs_nll_subject_grad_fisher(
        &model,
        &subject,
        &theta,
        &sigma_values,
        &subject_eta(),
        &mask,
        &lo,
        &hi,
        n_theta,
        n_sigma,
        &mut scratch,
    );
    assert!(
        fi.is_none(),
        "a residual magnitude must withhold the information"
    );
    assert!(nll.is_finite(), "the objective is still returned");
    let (nll_plain, grad_plain) = obs_nll_subject_grad(
        &model,
        &subject,
        &theta,
        &sigma_values,
        &subject_eta(),
        &mask,
        &lo,
        &hi,
        n_theta,
        n_sigma,
        &mut scratch,
    );
    assert_eq!(nll, nll_plain);
    assert_eq!(
        grad, grad_plain,
        "the gradient must be the same one entry point"
    );
}

fn subject_eta() -> Vec<f64> {
    vec![0.05]
}

/// Asking for the information must not change the gradient. `want_fisher` adds
/// a per-observation vector to the θ loop and re-associates its contraction, so
/// this is the check that the re-association stayed inside tolerance and that
/// the two entry points are one implementation.
#[test]
fn fisher_entry_point_returns_the_same_gradient() {
    use crate::types::DoseEvent;
    let model = combined_error_two_cpt_model();
    let theta = vec![1.4f64, 11.0, 2.1, 23.0];
    let sigma_values = vec![0.15f64, 0.40];
    let n_theta = theta.len();
    let n_sigma = sigma_values.len();
    let n = n_theta + n_sigma;
    let eta = vec![0.1f64];
    let subject = paired_residual_subject(
        &model,
        "1",
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &[0.5, 2.0, 6.0, 12.0],
        &theta,
        &sigma_values,
        &eta,
    );
    let mask = vec![true; n_theta];
    let lo = vec![-1e30f64; n];
    let hi = vec![1e30f64; n];
    let mut scratch = EventPkParams::default();
    let (nll_a, grad_a, fi) = obs_nll_subject_grad_fisher(
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
    assert!(fi.is_some());
    let (nll_b, grad_b) = obs_nll_subject_grad(
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
    assert_eq!(nll_a, nll_b, "the objective must be bit-identical");
    let worst = grad_a.iter().zip(grad_b.iter()).fold(0.0f64, |m, (a, b)| {
        assert!(a.is_finite() && b.is_finite());
        m.max((a - b).abs() / b.abs().max(1e-8))
    });
    // Realised **3.553e-8** on this fixture. The σ half is bit-identical; the θ
    // half sums the same products in a different order (`Σ dl·(Δf/h)` against
    // `Σ (dl·Δf)/h`), and the forward difference `Δf` is already a cancelling
    // subtraction, so the two orders separate at that scale. The bound is 30×
    // the realised value — enough headroom for another fixture, far below any
    // gradient error that would matter.
    assert!(
        worst < 1e-6,
        "the two entry points disagree on the gradient: worst relative {worst:.3e}"
    );
}
