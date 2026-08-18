use super::*;
use crate::estimation::inner_optimizer::find_ebe;
use crate::estimation::parameterization::pack_params;
use crate::parser::model_parser::parse_model_string;
use crate::stats::likelihood::{foce_subject_nll, foce_subject_nll_interaction};
use crate::types::{DoseEvent, ErrorSpec, OmegaMatrix, Subject};
use std::collections::HashMap;

const TWOCPT: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV1(30.0, 3.0, 300.0)
  theta TVQ(2.0, 0.1, 20.0)
  theta TVV2(50.0, 5.0, 500.0)
  theta TVKA(1.0, 0.05, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  omega ETA_KA ~ 0.10
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk two_cpt_oral(cl=CL, v=V1, q=Q, v2=V2, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

const THREECPT: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV1(10.0, 1.0, 100.0)
  theta TVQ2(2.0, 0.1, 20.0)
  theta TVV2(20.0, 2.0, 200.0)
  theta TVQ3(1.5, 0.1, 20.0)
  theta TVV3(30.0, 3.0, 300.0)
  theta TVKA(1.0, 0.05, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  omega ETA_KA ~ 0.10
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q2 = TVQ2
  V2 = TVV2
  Q3 = TVQ3
  V3 = TVV3
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk three_cpt_oral(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

fn params_with_omega(model: &CompiledModel, theta: &[f64], vars: &[f64]) -> ModelParameters {
    let mut p = model.default_params.clone();
    p.theta = theta.to_vec();
    p.omega = OmegaMatrix::from_diagonal(vars, model.eta_names.clone());
    p
}

/// The byte-identical no-covariate `Population` wrapper used across the FD
/// comparison tests (empty covariates/inputs, `DV` column, no exclusions).
fn pop_of(subjects: Vec<Subject>) -> Population {
    Population {
        subjects,
        covariate_names: vec![],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

/// Per-coordinate Richardson-FD gradient check shared by the packed-gradient
/// comparison tests. For each coordinate `k`, forms the Richardson combination
/// of two central differences of `ofv` (steps `h` and `h/2`, with
/// `h = 1e-4·(1+|x[k]|)`) and asserts `analytic[k]` matches it to the caller's
/// `max_relative` / `epsilon`. Moves the harness arithmetic verbatim — each
/// test supplies only its own `ofv` closure and tolerances.
fn assert_grad_matches_richardson_fd(
    x: &[f64],
    analytic: &[f64],
    ofv: impl Fn(&[f64]) -> f64,
    max_relative: f64,
    epsilon: f64,
) {
    let fd_at = |k: usize, h: f64| -> f64 {
        let mut xp = x.to_vec();
        xp[k] += h;
        let mut xm = x.to_vec();
        xm[k] -= h;
        (ofv(&xp) - ofv(&xm)) / (2.0 * h)
    };
    for k in 0..x.len() {
        let h = 1e-4 * (1.0 + x[k].abs());
        let f1 = fd_at(k, h);
        let f2 = fd_at(k, h / 2.0);
        let fd = (4.0 * f2 - f1) / 3.0; // Richardson
        eprintln!(
            "x[{k}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
            analytic[k],
            fd,
            (analytic[k] - fd).abs() / fd.abs().max(1e-12)
        );
        approx::assert_relative_eq!(
            analytic[k],
            fd,
            max_relative = max_relative,
            epsilon = epsilon
        );
    }
}

const WARFARIN: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

fn subject_with_obs(model: &CompiledModel, theta: &[f64], times: &[f64]) -> Subject {
    // Build observations from the model at a reference eta so residuals are
    // realistic and nonzero (gradient identity holds at any obs).
    let n = times.len();
    let mut subject = Subject {
        id: "1".to_string(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: times.to_vec(),
        obs_raw_times: Vec::new(),
        observations: vec![0.0; n],
        obs_cmts: vec![1; n],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0; n],
        occasions: vec![1; n],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    let eta_ref = [0.12, -0.08, 0.2];
    let preds = crate::pk::compute_predictions_with_tv(model, &subject, theta, &eta_ref);
    // Perturb by a fixed multiplicative factor so ε ≠ 0.
    subject.observations = preds.iter().map(|p| p * 0.85).collect();
    subject
}

/// A dosing-only subject (dose rows, no DV) contributes zero to the FOCE/FOCEI
/// marginal objective: with no data rows, `log|Ω| + log|Ω⁻¹|` cancels at
/// `η̂ = 0`. It should therefore return a zero analytic gradient instead of
/// forcing the whole population gradient onto the FD fallback.
#[test]
fn zero_observation_subject_returns_zero_gradient() {
    let model = parse_model_string(TWOCPT).expect("parse");
    let template = model.default_params.clone();
    let theta = template.theta.clone();
    let mut subject = subject_with_obs(&model, &theta, &[2.0]); // build then empty out
    subject.obs_times.clear();
    subject.observations.clear();
    subject.obs_cmts.clear();
    subject.cens.clear();
    subject.occasions.clear();
    assert!(subject.observations.is_empty());

    let x = pack_params(&template);
    let eta_hat = vec![0.0; model.n_eta];
    let packed = subject_packed_gradient(&model, &subject, &template, &x, &eta_hat)
        .expect("dosing-only subject has zero packed gradient");
    assert_eq!(packed, vec![0.0; x.len()]);

    let theta_grad = subject_theta_gradient(&model, &subject, &template, &eta_hat)
        .expect("dosing-only subject has zero theta gradient");
    assert_eq!(theta_grad, vec![0.0; model.n_theta]);

    let omega_grad = subject_omega_gradient(&model, &subject, &template, &eta_hat)
        .expect("dosing-only subject has zero omega gradient");
    assert_eq!(omega_grad, vec![0.0; model.n_eta]);

    let sigma_grad = subject_sigma_gradient(&model, &subject, &template, &eta_hat)
        .expect("dosing-only subject has zero sigma gradient");
    assert_eq!(sigma_grad, vec![0.0; template.sigma.values.len()]);

    let eta_dx = subject_eta_dx(&model, &subject, &template, &x, &eta_hat)
        .expect("dosing-only subject has zero EBE predictor");
    assert_eq!(eta_dx, vec![DVector::zeros(model.n_eta); x.len()]);
}

/// `sigma_fd_step` keeps the central-difference minus side `σ − h` strictly
/// positive: an ordinary σ uses the full `1e-6·(1+|σ|)` step, but a σ at/below
/// that step shrinks to `0.5·σ` so the minus evaluation never underflows the
/// `variance_at` floor (PR #381 review #6).
#[test]
fn sigma_fd_step_keeps_minus_side_positive() {
    // Ordinary σ: unchanged full step (h ≪ σ).
    let sig = 0.2;
    let h = sigma_fd_step(sig);
    assert!((h - 1e-6 * (1.0 + sig)).abs() < 1e-18);
    assert!(sig - h > 0.0);
    // Near-zero σ: step shrinks to 0.5·σ, minus side stays positive.
    let tiny = 5e-7;
    let h_tiny = sigma_fd_step(tiny);
    assert_eq!(h_tiny, 0.5 * tiny);
    assert!(tiny - h_tiny > 0.0);
    // σ = 0 leaves the base step (degenerate; no positive side to protect).
    assert_eq!(sigma_fd_step(0.0), 1e-6);
}

/// Precisely locate η̂ via analytic Newton on the inner objective (exact
/// gradient ½Σαⱼaⱼ + Ω⁻¹η and true Hessian H from the provider), so the
/// marginal-NLL finite difference is not contaminated by inner-solver
/// reconvergence noise. Warm-started from `find_ebe`.
///
/// Custom / time-varying residual-magnitude models (#484/#576/#486): `mult`
/// scales the per-observation variance the same way `prepare_stacked` does, so
/// this locates the *true* magnitude-aware η̂ — without it, the reconverged-FD
/// harness would minimise a different (bare) inner objective and the
/// analytic-vs-FD comparison in `magnitude_*_family_outer_gradient_matches_fd`
/// would be meaningless (the Eq. 46 EBE-response identity only holds at the
/// actual stationary point).
fn precise_ebe(model: &CompiledModel, subject: &Subject, params: &ModelParameters) -> Vec<f64> {
    let warm = find_ebe(model, subject, params, 80, 1e-10, None, None, 0);
    let mut eta: Vec<f64> = warm.eta.iter().copied().collect();
    let n_eta = model.n_eta;
    let sigma = &params.sigma.values;
    let omega_inv = &params.omega.inv;
    let mult = model.ruv_obs_mult(subject, &params.theta);
    let frem_r = crate::stats::likelihood::build_frem_r_override(
        model.frem_config.as_ref(),
        &subject.fremtype,
        sigma,
    );
    for _ in 0..50 {
        let sens =
            crate::sens::provider::subject_sensitivities(model, subject, &params.theta, &eta)
                .unwrap();
        let mut grad = nalgebra::DVector::<f64>::from_column_slice(
            &(omega_inv * nalgebra::DVector::from_column_slice(&eta))
                .iter()
                .copied()
                .collect::<Vec<_>>(),
        );
        let mut hess = omega_inv.clone();
        let m3 = matches!(model.bloq_method, crate::types::BloqMethod::M3);
        for (j, obs) in sens.obs.iter().enumerate() {
            let f = obs.f;
            let cmt = subject.obs_cmts[j];
            let mult_row: Option<&[f64]> =
                mult.as_ref().and_then(|m| m.get(j)).map(|v| v.as_slice());
            let frem_var = frem_r.as_ref().and_then(|o| o.get(j)).and_then(|x| *x);
            let (r, d, d2) = match (frem_var, mult_row) {
                (Some(v), _) => (v, 0.0, 0.0),
                (None, Some(m)) => (
                    model.error_spec.variance_at_scaled(cmt, f, sigma, &[], m),
                    model.error_spec.dvar_df_scaled(cmt, f, sigma, m),
                    model.error_spec.d2var_df2_scaled(cmt, f, sigma, m),
                ),
                (None, None) => (
                    model.error_spec.variance_at(cmt, f, sigma),
                    model.error_spec.dvar_df(cmt, f, sigma),
                    model.error_spec.d2var_df2(cmt, f, sigma),
                ),
            };
            let y = subject.observations[j];
            // (g1, g2) = (∂L/∂f, ∂²L/∂f²): the censored `−logΦ` scalars for an
            // M3 BLOQ row, else the Gaussian `½α`, `½α'`.
            let (g1, g2) = if m3 && subject.cens.get(j).copied().unwrap_or(0) != 0 {
                m3_censored_scalars(y, f, r, d, d2, subject.cens.get(j).copied().unwrap_or(0))
            } else {
                let t = err_terms(r, d, d2, y - f);
                (0.5 * t.alpha, 0.5 * t.alpha_p)
            };
            let a = obs.df_deta.as_slice();
            for k in 0..n_eta {
                grad[k] += g1 * a[k];
                for l in 0..n_eta {
                    hess[(k, l)] += g2 * a[k] * a[l] + g1 * obs.d2f_deta2[k * n_eta + l];
                }
            }
        }
        let step = hess.cholesky().unwrap().solve(&grad);
        for k in 0..n_eta {
            eta[k] -= step[k];
        }
        if step.norm() < 1e-13 {
            break;
        }
    }
    eta
}

/// `∂f/∂η` Jacobian (row-major `n_obs × n_eta`) at `eta` via the light
/// provider, falling back to the full provider's `df_deta` for models the light
/// one doesn't cover (TV-covariates, `ExpressionScale`) so the reconverged-FD
/// marginal harness works there too. Test-only — the full provider's first
/// derivative is exact, so the FOCE linearization is identical.
fn eta_jacobian_any(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Vec<f64> {
    if let Some(j) = crate::sens::provider::subject_eta_jacobian(model, subject, theta, eta) {
        return j;
    }
    let s = crate::sens::provider::subject_sensitivities(model, subject, theta, eta)
        .expect("provider supports subject");
    let mut j = Vec::with_capacity(s.obs.len() * model.n_eta);
    for o in &s.obs {
        j.extend_from_slice(&o.df_deta);
    }
    j
}

/// Per-subject Laplace NLL Fᵢ at a *given* η̂ (no reconvergence).
fn marginal_nll_at(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    eta: &[f64],
) -> f64 {
    let eta_v = nalgebra::DVector::from_column_slice(eta);
    let ipreds = crate::pk::compute_predictions_with_tv(model, subject, &params.theta, eta);
    let jac = eta_jacobian_any(model, subject, &params.theta, eta);
    let h_matrix = nalgebra::DMatrix::from_row_slice(subject.obs_times.len(), model.n_eta, &jac);
    let frem_r_override = crate::stats::likelihood::build_frem_r_override(
        model.frem_config.as_ref(),
        &subject.fremtype,
        &params.sigma.values,
    );
    foce_subject_nll_interaction(
        subject,
        &ipreds,
        &eta_v,
        &h_matrix,
        &params.omega,
        &params.sigma.values,
        &model.error_spec,
        model.bloq_method,
        &[],
        frem_r_override.as_deref(),
        model.residual_error_eta,
        model.ruv_obs_mult(subject, &params.theta).as_deref(),
    )
}

/// Reconverged marginal NLL Fᵢ(θ) at the precisely-located EBE.
fn marginal_nll(model: &CompiledModel, subject: &Subject, params: &ModelParameters) -> f64 {
    let eta = precise_ebe(model, subject, params);
    marginal_nll_at(model, subject, params, &eta)
}

/// Per-subject **FOCE** (non-interaction) marginal NLL at a given η̂ — ferx's
/// Sheiner–Beal linearized objective via `foce_subject_nll(.., interaction=false)`.
fn marginal_nll_foce_at(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    eta: &[f64],
) -> f64 {
    let eta_v = nalgebra::DVector::from_column_slice(eta);
    let jac = eta_jacobian_any(model, subject, &params.theta, eta);
    let h_matrix = nalgebra::DMatrix::from_row_slice(subject.obs_times.len(), model.n_eta, &jac);
    foce_subject_nll(
        model,
        subject,
        &params.theta,
        &eta_v,
        &h_matrix,
        &params.omega,
        &params.sigma.values,
        false,
    )
}

/// Reconverged FOCE marginal NLL at the precisely-located (shared) EBE.
fn marginal_nll_foce(model: &CompiledModel, subject: &Subject, params: &ModelParameters) -> f64 {
    let eta = precise_ebe(model, subject, params);
    marginal_nll_foce_at(model, subject, params, &eta)
}

/// FOCE analog of [`run_population_packed_gradient_check`]: the analytic FOCE
/// packed gradient must match the reconverged-FD of ferx's FOCE OFV.
fn run_packed_check_foce(model: &CompiledModel, theta: &[f64]) {
    use crate::estimation::parameterization::pack_params;

    let s1 = subject_with_obs(model, theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    let s2 = subject_with_obs(model, theta, &[0.25, 1.5, 3.0, 6.0, 12.0, 36.0, 72.0]);
    let pop = pop_of(vec![s1, s2]);

    let mut template = model.default_params.clone();
    template.theta = theta.to_vec();
    let x = pack_params(&template);
    let params = unpack_params(&x, &template);
    let ehs: Vec<DVector<f64>> = pop
        .subjects
        .iter()
        .map(|s| DVector::from_vec(precise_ebe(model, s, &params)))
        .collect();

    let analytic =
        population_gradient_sens_foce(model, &pop, &template, &x, &ehs).expect("supported");

    let ofv = |xv: &[f64]| -> f64 {
        let p = unpack_params(xv, &template);
        2.0 * pop
            .subjects
            .iter()
            .map(|s| marginal_nll_foce(model, s, &p))
            .sum::<f64>()
    };
    assert_grad_matches_richardson_fd(&x, &analytic, ofv, 3e-3, 1e-5);
}

#[test]
fn theta_gradient_matches_reconverged_fd() {
    let model = parse_model_string(WARFARIN).expect("parse");
    let theta = vec![0.22, 11.0, 1.4];
    let times = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0, 48.0];
    let subject = subject_with_obs(&model, &theta, &times);

    let mut params = model.default_params.clone();
    params.theta = theta.clone();

    // Precisely-located EBE at the base point.
    let eta_hat = precise_ebe(&model, &subject, &params);

    let analytic = subject_theta_gradient(&model, &subject, &params, &eta_hat).expect("supported");

    // Richardson-extrapolated reconverged central FD of the marginal NLL
    // (cancels the O(h²) truncation; EBE is located analytically so there is
    // no inner-solver noise floor).
    let fd_at = |m: usize, h: f64| -> f64 {
        let mut pp = params.clone();
        pp.theta[m] += h;
        let mut pm = params.clone();
        pm.theta[m] -= h;
        (marginal_nll(&model, &subject, &pp) - marginal_nll(&model, &subject, &pm)) / (2.0 * h)
    };
    let n_theta = theta.len();
    for m in 0..n_theta {
        let h = 1e-4 * (1.0 + theta[m].abs());
        let f1 = fd_at(m, h);
        let f2 = fd_at(m, h / 2.0);
        let fd = (4.0 * f2 - f1) / 3.0; // Richardson
        eprintln!(
            "theta[{m}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
            analytic[m],
            fd,
            (analytic[m] - fd).abs() / fd.abs().max(1e-12)
        );
        approx::assert_relative_eq!(analytic[m], fd, max_relative = 1e-3, epsilon = 1e-6);
    }
}

#[test]
fn omega_gradient_matches_reconverged_fd() {
    let model = parse_model_string(WARFARIN).expect("parse");
    let theta = vec![0.22, 11.0, 1.4];
    let times = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0, 48.0];
    let subject = subject_with_obs(&model, &theta, &times);

    let vars = vec![0.09, 0.04, 0.30];
    let params = params_with_omega(&model, &theta, &vars);
    let eta_hat = precise_ebe(&model, &subject, &params);

    let analytic = subject_omega_gradient(&model, &subject, &params, &eta_hat).expect("supported");

    // Richardson reconverged FD over each natural variance entry.
    let fd_at = |i: usize, h: f64| -> f64 {
        let mut vp = vars.clone();
        vp[i] += h;
        let mut vm = vars.clone();
        vm[i] -= h;
        let pp = params_with_omega(&model, &theta, &vp);
        let pm = params_with_omega(&model, &theta, &vm);
        (marginal_nll(&model, &subject, &pp) - marginal_nll(&model, &subject, &pm)) / (2.0 * h)
    };
    for i in 0..vars.len() {
        let h = 1e-4 * (1.0 + vars[i].abs());
        let f1 = fd_at(i, h);
        let f2 = fd_at(i, h / 2.0);
        let fd = (4.0 * f2 - f1) / 3.0;
        eprintln!(
            "omega[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
            analytic[i],
            fd,
            (analytic[i] - fd).abs() / fd.abs().max(1e-12)
        );
        approx::assert_relative_eq!(analytic[i], fd, max_relative = 1e-3, epsilon = 1e-6);
    }
}

#[test]
fn sigma_gradient_matches_reconverged_fd() {
    let model = parse_model_string(WARFARIN).expect("parse");
    let theta = vec![0.22, 11.0, 1.4];
    let times = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0, 48.0];
    let subject = subject_with_obs(&model, &theta, &times);

    let mut params = model.default_params.clone();
    params.theta = theta.clone();
    let eta_hat = precise_ebe(&model, &subject, &params);

    let analytic = subject_sigma_gradient(&model, &subject, &params, &eta_hat).expect("supported");

    let sig0 = params.sigma.values.clone();
    let fd_at = |k: usize, h: f64| -> f64 {
        let mut pp = params.clone();
        pp.sigma.values[k] += h;
        let mut pm = params.clone();
        pm.sigma.values[k] -= h;
        (marginal_nll(&model, &subject, &pp) - marginal_nll(&model, &subject, &pm)) / (2.0 * h)
    };
    for k in 0..sig0.len() {
        let h = 1e-4 * (1.0 + sig0[k].abs());
        let f1 = fd_at(k, h);
        let f2 = fd_at(k, h / 2.0);
        let fd = (4.0 * f2 - f1) / 3.0;
        eprintln!(
            "sigma[{k}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
            analytic[k],
            fd,
            (analytic[k] - fd).abs() / fd.abs().max(1e-12)
        );
        approx::assert_relative_eq!(analytic[k], fd, max_relative = 2e-3, epsilon = 1e-6);
    }
}

/// Warfarin oral with a dedicated residual-error eta (`iiv_on_ruv`, #474).
/// `ETA_RUV` is the 4th declared omega (index 3) and is not used in any
/// individual parameter.
const WARFARIN_RUV: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  omega ETA_RUV ~ 0.10
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
  iiv_on_ruv = ETA_RUV
"#;

/// Subject for an `iiv_on_ruv` model: predictions are independent of η_ruv, so
/// build realistic nonzero residuals from the structural etas and pad the eta
/// vector for the (prediction-irrelevant) residual eta.
fn ruv_subject(model: &CompiledModel, theta: &[f64], times: &[f64]) -> Subject {
    let n = times.len();
    let mut subject = Subject {
        id: "1".to_string(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: times.to_vec(),
        obs_raw_times: Vec::new(),
        observations: vec![0.0; n],
        obs_cmts: vec![1; n],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0; n],
        occasions: vec![1; n],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    let eta_ref = [0.12, -0.08, 0.2, 0.0];
    let preds = crate::pk::compute_predictions_with_tv(model, &subject, theta, &eta_ref);
    subject.observations = preds.iter().map(|p| p * 0.85).collect();
    subject
}

/// Precise EBE for an `iiv_on_ruv` model: Newton on the *scaled* inner
/// objective (residual variance × `exp(2·η_ruv)`, plus the residual-eta
/// gradient `1−ε²/R` and Hessian `2ε²/R` / `κ a` terms), mirroring `prepare`'s
/// `H` so the marginal FD is not contaminated by inner-solver noise.
fn precise_ebe_ruv(model: &CompiledModel, subject: &Subject, params: &ModelParameters) -> Vec<f64> {
    let warm = find_ebe(model, subject, params, 80, 1e-10, None, None, 0);
    let mut eta: Vec<f64> = warm.eta.iter().copied().collect();
    let n_eta = model.n_eta;
    let rr = model.residual_error_eta.expect("ruv model");
    let m3 = matches!(model.bloq_method, crate::types::BloqMethod::M3);
    let sigma = &params.sigma.values;
    let omega_inv = &params.omega.inv;
    // Custom/TV residual magnitude: the inner NLL must use the same magnitude-scaled
    // variance production does, else the reconverged η is not the true inner minimum
    // and the envelope-based outer gradient mismatches FD.
    let mult = model.ruv_obs_mult(subject, &params.theta);
    for _ in 0..50 {
        let s = (2.0 * eta[rr]).exp();
        let sens =
            crate::sens::provider::subject_sensitivities(model, subject, &params.theta, &eta)
                .unwrap();
        let mut grad = omega_inv * DVector::from_column_slice(&eta);
        let mut hess = omega_inv.clone();
        for (j, obs) in sens.obs.iter().enumerate() {
            let f = obs.f;
            let cmt = subject.obs_cmts[j];
            let mult_row: Option<&[f64]> =
                mult.as_ref().and_then(|m| m.get(j)).map(|v| v.as_slice());
            let (r, d, d2) = match mult_row {
                Some(mr) => (
                    model.error_spec.variance_at_scaled(cmt, f, sigma, &[], mr) * s,
                    model.error_spec.dvar_df_scaled(cmt, f, sigma, mr) * s,
                    model.error_spec.d2var_df2_scaled(cmt, f, sigma, mr) * s,
                ),
                None => (
                    model.error_spec.variance_at(cmt, f, sigma) * s,
                    model.error_spec.dvar_df(cmt, f, sigma) * s,
                    model.error_spec.d2var_df2(cmt, f, sigma) * s,
                ),
            };
            let y = subject.observations[j];
            let eps = y - f;
            let is_cens = m3 && subject.cens.get(j).copied().unwrap_or(0) != 0;
            let (g1, g2) = if is_cens {
                m3_censored_scalars(y, f, r, d, d2, subject.cens.get(j).copied().unwrap_or(0))
            } else {
                let t = err_terms(r, d, d2, eps);
                (0.5 * t.alpha, 0.5 * t.alpha_p)
            };
            let a = obs.df_deta.as_slice();
            for k in 0..n_eta {
                grad[k] += g1 * a[k];
                for l in 0..n_eta {
                    hess[(k, l)] += g2 * a[k] * a[l] + g1 * obs.d2f_deta2[k * n_eta + l];
                }
            }
            // Residual-eta gradient / Hessian (a_{ruv} = A_{·,ruv} = 0). Censored
            // rows use the M3 cross-terms `h·z` / `C·z` / `C·m·a` (#4c); quantified
            // rows the Gaussian `1−ε²/R` / `2ε²/R` / `κ a` (#474).
            if is_cens {
                let (h, z, m) = crate::stats::special::m3_censored_kernel(
                    y,
                    f,
                    r,
                    d,
                    subject.cens.get(j).copied().unwrap_or(0),
                );
                let c = h * (z * z + h * z - 1.0);
                grad[rr] += h * z;
                hess[(rr, rr)] += c * z;
                for l in 0..n_eta {
                    if l == rr {
                        continue;
                    }
                    hess[(rr, l)] += c * m * a[l];
                    hess[(l, rr)] += c * m * a[l];
                }
            } else {
                grad[rr] += 1.0 - eps * eps / r;
                hess[(rr, rr)] += 2.0 * eps * eps / r;
                let kappa = 2.0 * eps / r + eps * eps * d / (r * r);
                for l in 0..n_eta {
                    if l == rr {
                        continue;
                    }
                    hess[(rr, l)] += kappa * a[l];
                    hess[(l, rr)] += kappa * a[l];
                }
            }
        }
        let step = hess.cholesky().unwrap().solve(&grad);
        for k in 0..n_eta {
            eta[k] -= step[k];
        }
        if step.norm() < 1e-13 {
            break;
        }
    }
    eta
}

/// 2-cpt IV **user-ODE** model with IIV on residual error (`iiv_on_ruv`). Same
/// structure as `TWOCPT_ODE_OUTER` plus a dedicated `ETA_RUV` omega — exercises
/// the residual-eta assembly through the ODE Dual2 sensitivity provider (#474).
const TWOCPT_ODE_RUV: &str = r#"
[parameters]
  theta TVCL(4.0,  0.1, 100.0)
  theta TVV1(12.0, 1.0, 500.0)
  theta TVQ(2.0,   0.01, 100.0)
  theta TVV2(25.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.04
  omega ETA_RUV ~ 0.10
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  ode(states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  y = central / V1
[error_model]
  DV ~ proportional(PROP_ERR)
  iiv_on_ruv = ETA_RUV
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

/// Shared FD check for an `iiv_on_ruv` model: the analytic FOCEI population
/// packed gradient must match the Richardson-extrapolated reconverged-FD of
/// ferx's own scaled FOCEI marginal across every θ/Ω/σ coordinate — including
/// the residual eta's Ω entry (#474). The OFV *value* it differentiates is
/// independently NONMEM-validated (#413).
fn run_ruv_packed_check(model: &CompiledModel, theta: &[f64]) {
    use crate::estimation::parameterization::pack_params;

    let s1 = ruv_subject(model, theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    let s2 = ruv_subject(model, theta, &[0.25, 1.5, 3.0, 6.0, 12.0, 36.0, 72.0]);
    let pop = pop_of(vec![s1, s2]);

    let mut template = model.default_params.clone();
    template.theta = theta.to_vec();
    let x = pack_params(&template);
    let params = unpack_params(&x, &template);
    let ehs: Vec<DVector<f64>> = pop
        .subjects
        .iter()
        .map(|s| DVector::from_vec(precise_ebe_ruv(model, s, &params)))
        .collect();

    let analytic =
        population_gradient_sens(model, &pop, &template, &x, &ehs).expect("ruv is analytic");

    // 2·Σᵢ Fᵢ at the reconverged (scaled) EBE — the production FOCEI OFV.
    let ofv = |xv: &[f64]| -> f64 {
        let p = unpack_params(xv, &template);
        2.0 * pop
            .subjects
            .iter()
            .map(|s| {
                let eta = precise_ebe_ruv(model, s, &p);
                marginal_nll_at(model, s, &p, &eta)
            })
            .sum::<f64>()
    };
    assert_grad_matches_richardson_fd(&x, &analytic, ofv, 2e-3, 1e-5);
}

#[test]
fn population_packed_gradient_iiv_on_ruv_matches_fd() {
    let model = parse_model_string(WARFARIN_RUV).expect("parse");
    assert_eq!(model.residual_error_eta, Some(3));
    run_ruv_packed_check(&model, &[0.22, 11.0, 1.4]);
}

/// 1-cpt IV, two BSV etas, correlated `combined` residual error (`block_sigma`, #627).
const BLOCK_SIGMA_1CPT: &str = "[parameters]\n  theta TVCL(1.0, 0.01, 10.0)\n  theta TVV(10.0, 0.1, 100.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  block_sigma (PROP_ERR, ADD_ERR) = [0.04, 0.05, 1.00]\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV * exp(ETA_V)\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ combined(PROP_ERR, ADD_ERR)\n[fit_options]\n  method = focei\n";

fn dense_subject(model: &CompiledModel, theta: &[f64], times: &[f64]) -> Subject {
    let n = times.len();
    let mut subject = Subject {
        id: "1".to_string(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: times.to_vec(),
        obs_raw_times: Vec::new(),
        observations: vec![0.0; n],
        obs_cmts: vec![1; n],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0; n],
        occasions: vec![1; n],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    let preds = crate::pk::compute_predictions_with_tv(model, &subject, theta, &[0.12, -0.08]);
    subject.observations = preds.iter().map(|p| p * 0.85 + 0.2).collect();
    subject
}

/// Correlation-aware EBE: Newton on the dense inner NLL using the diagonal-but-
/// correlated `(r,d,d2)` from [`corr_residual_diag`]. The scalar [`precise_ebe`] uses
/// the plain error functions (no within-obs cross term) and would converge to the
/// wrong mode for `block_sigma`, breaking the envelope theorem the gradient assumes.
fn precise_ebe_corr(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
) -> Vec<f64> {
    let warm = find_ebe(model, subject, params, 80, 1e-12, None, None, 0);
    let mut eta: Vec<f64> = warm.eta.iter().copied().collect();
    let n_eta = model.n_eta;
    let sigma = &params.sigma.values;
    let omega_inv = &params.omega.inv;
    for _ in 0..60 {
        let sens =
            crate::sens::provider::subject_sensitivities(model, subject, &params.theta, &eta)
                .unwrap();
        let (rv, dv, d2v) = corr_residual_diag(model, subject, &sens, sigma).unwrap();
        let mut grad = DVector::<f64>::from_column_slice(
            &(omega_inv * DVector::from_column_slice(&eta))
                .iter()
                .copied()
                .collect::<Vec<_>>(),
        );
        let mut hess = omega_inv.clone();
        for (j, obs) in sens.obs.iter().enumerate() {
            let t = err_terms(rv[j], dv[j], d2v[j], subject.observations[j] - obs.f);
            let (g1, g2) = (0.5 * t.alpha, 0.5 * t.alpha_p);
            let a = obs.df_deta.as_slice();
            for k in 0..n_eta {
                grad[k] += g1 * a[k];
                for l in 0..n_eta {
                    hess[(k, l)] += g2 * a[k] * a[l] + g1 * obs.d2f_deta2[k * n_eta + l];
                }
            }
        }
        let step = hess.cholesky().unwrap().solve(&grad);
        for k in 0..n_eta {
            eta[k] -= step[k];
        }
        if step.norm() < 1e-13 {
            break;
        }
    }
    eta
}

/// Dense (`block_sigma`) FOCEI marginal at a given η̂. The production
/// `foce_subject_nll(.., interaction=true)` dispatches to
/// `foce_subject_nll_interaction_dense` for correlated models.
fn marginal_nll_dense_at(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    eta: &[f64],
) -> f64 {
    let eta_v = DVector::from_column_slice(eta);
    let jac = eta_jacobian_any(model, subject, &params.theta, eta);
    let h = DMatrix::from_row_slice(subject.obs_times.len(), model.n_eta, &jac);
    crate::stats::likelihood::foce_subject_nll(
        model,
        subject,
        &params.theta,
        &eta_v,
        &h,
        &params.omega,
        &params.sigma.values,
        true,
    )
}

/// The analytic FOCEI packed gradient for a correlated `block_sigma` model must match
/// Richardson reconverged FD of ferx's own dense FOCEI marginal across every θ/Ω/σ
/// coordinate (#627). The within-observation `combined` cross term modifies the
/// residual variance and its `∂/∂f`, `∂²/∂f²`, which the scalar path omits — so this
/// confirms the correlation-aware `(r,d,d2)` reduction of the dense Almquist assembly.
#[test]
fn population_packed_gradient_block_sigma_matches_fd() {
    use crate::estimation::parameterization::pack_params;

    let model = parse_model_string(BLOCK_SIGMA_1CPT).expect("parse block_sigma");
    assert!(
        !model.residual_correlations.is_empty(),
        "fixture must carry a residual correlation"
    );
    assert!(crate::sens::provider::analytic_outer_gradient_available(
        &model
    ));

    let theta = &[1.1, 11.0];
    let s1 = dense_subject(&model, theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    let s2 = dense_subject(&model, theta, &[0.25, 1.5, 3.0, 6.0, 12.0, 36.0]);
    let pop = pop_of(vec![s1, s2]);

    let mut template = model.default_params.clone();
    template.theta = theta.to_vec();
    let x = pack_params(&template);
    let params = unpack_params(&x, &template);
    let ehs: Vec<DVector<f64>> = pop
        .subjects
        .iter()
        .map(|s| DVector::from_vec(precise_ebe_corr(&model, s, &params)))
        .collect();

    let analytic = population_gradient_sens(&model, &pop, &template, &x, &ehs)
        .expect("block_sigma is analytic");

    let ofv = |xv: &[f64]| -> f64 {
        let p = unpack_params(xv, &template);
        2.0 * pop
            .subjects
            .iter()
            .map(|s| {
                let eta = precise_ebe_corr(&model, s, &p);
                marginal_nll_dense_at(&model, s, &p, &eta)
            })
            .sum::<f64>()
    };
    assert_grad_matches_richardson_fd(&x, &analytic, ofv, 2e-3, 1e-5);
}

/// FOCE (non-interaction) analog of `population_packed_gradient_block_sigma_matches_fd`.
/// The Sheiner–Beal linearized marginal freezes `R⁰` at η=0, so `block_sigma` only
/// needs the correlation-aware `(r0, d0)` and `∂R⁰/∂σ` (no `∂²R/∂f²`). Must match
/// Richardson reconverged FD of ferx's dense FOCE OFV across every θ/Ω/σ coord (#627).
#[test]
fn population_packed_gradient_block_sigma_foce_matches_fd() {
    use crate::estimation::parameterization::pack_params;

    let src = BLOCK_SIGMA_1CPT.replace("method = focei", "method = foce");
    let model = parse_model_string(&src).expect("parse block_sigma foce");
    assert!(!model.residual_correlations.is_empty());

    let theta = &[1.1, 11.0];
    let s1 = dense_subject(&model, theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    let s2 = dense_subject(&model, theta, &[0.25, 1.5, 3.0, 6.0, 12.0, 36.0]);
    let pop = pop_of(vec![s1, s2]);

    let mut template = model.default_params.clone();
    template.theta = theta.to_vec();
    let x = pack_params(&template);
    let params = unpack_params(&x, &template);
    let ehs: Vec<DVector<f64>> = pop
        .subjects
        .iter()
        .map(|s| DVector::from_vec(precise_ebe_corr(&model, s, &params)))
        .collect();

    let analytic = population_gradient_sens_foce(&model, &pop, &template, &x, &ehs)
        .expect("block_sigma foce is analytic");

    let ofv = |xv: &[f64]| -> f64 {
        let p = unpack_params(xv, &template);
        2.0 * pop
            .subjects
            .iter()
            .map(|s| {
                let eta = precise_ebe_corr(&model, s, &p);
                marginal_nll_foce_at(&model, s, &p, &eta)
            })
            .sum::<f64>()
    };
    assert_grad_matches_richardson_fd(&x, &analytic, ofv, 2e-3, 1e-5);
}

/// Covariate-selected (`if/else`) `block_sigma` fixture: each row's residual
/// branch is chosen by a per-row `FREE` flag, and the two branch sigmas are
/// correlated through one `block_sigma`. Distinct observation times mean no
/// two rows share a residual block, so `R` stays diagonal and the analytic
/// outer-gradient path (`corr_residual_diag`) proceeds rather than bailing to
/// FD — exactly the path that must resolve the endpoint via the selector
/// (`obs_keys`), not the raw CMT column (#669).
const SELECTED_BLOCK_SIGMA_1CPT: &str = "[parameters]\n  theta TVCL(1.0, 0.01, 10.0)\n  theta TVV(10.0, 0.1, 100.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  block_sigma (PROP_TOTAL, PROP_UNBOUND) = [0.04, 0.03, 0.09]\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV * exp(ETA_V)\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  if (FREE == 0) {\n    DV ~ proportional(PROP_TOTAL)\n  } else {\n    DV ~ proportional(PROP_UNBOUND)\n  }\n[covariates]\n  FREE continuous\n[fit_options]\n  method = focei\n";

/// [`dense_subject`] with a per-row `FREE` covariate driving the selected
/// residual branch. `times` and `free_flags` must be equal length; times are
/// kept distinct by the caller so `R` is diagonal.
fn selected_dense_subject(
    model: &CompiledModel,
    theta: &[f64],
    times: &[f64],
    free_flags: &[f64],
) -> Subject {
    let mut subject = dense_subject(model, theta, times);
    subject.obs_covariates = free_flags
        .iter()
        .map(|&f| HashMap::from([("FREE".to_string(), f)]))
        .collect();
    subject
}

/// #669 regression: with the `Selected` + `block_sigma` guard lifted, the
/// analytic FOCEI outer gradient must resolve each row's endpoint via the
/// covariate selector. If `corr_residual_diag` / `corr_residual_rd_at_sigma`
/// read the raw CMT column (all-1 here) instead of the branch, every `FREE=0`
/// row is scored against the `else` branch's sigma and the analytic gradient
/// diverges from FD of ferx's dense FOCEI marginal (which uses the correct
/// keys). Must match across every θ/Ω/σ coordinate.
#[test]
fn population_packed_gradient_selected_block_sigma_matches_fd() {
    use crate::estimation::parameterization::pack_params;

    let model = parse_model_string(SELECTED_BLOCK_SIGMA_1CPT).expect("parse selected block_sigma");
    assert!(matches!(model.error_spec, ErrorSpec::Selected { .. }));
    assert!(!model.residual_correlations.is_empty());
    assert!(crate::sens::provider::analytic_outer_gradient_available(
        &model
    ));

    // Distinct times → diagonal R (analytic path proceeds); alternating FREE
    // flags route rows to both branches within one subject.
    let theta = &[1.1, 11.0];
    let s1 = selected_dense_subject(
        &model,
        theta,
        &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0],
        &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    );
    let s2 = selected_dense_subject(
        &model,
        theta,
        &[0.25, 1.5, 3.0, 6.0, 12.0, 36.0],
        &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    );
    let pop = Population {
        subjects: vec![s1, s2],
        covariate_names: vec!["FREE".into()],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };

    let mut template = model.default_params.clone();
    template.theta = theta.to_vec();
    let x = pack_params(&template);
    let params = unpack_params(&x, &template);
    let ehs: Vec<DVector<f64>> = pop
        .subjects
        .iter()
        .map(|s| DVector::from_vec(precise_ebe_corr(&model, s, &params)))
        .collect();

    let analytic = population_gradient_sens(&model, &pop, &template, &x, &ehs)
        .expect("selected block_sigma is analytic");

    let ofv = |xv: &[f64]| -> f64 {
        let p = unpack_params(xv, &template);
        2.0 * pop
            .subjects
            .iter()
            .map(|s| {
                let eta = precise_ebe_corr(&model, s, &p);
                marginal_nll_dense_at(&model, s, &p, &eta)
            })
            .sum::<f64>()
    };
    assert_grad_matches_richardson_fd(&x, &analytic, ofv, 2e-3, 1e-5);
}

/// `block_sigma` + η-dependent `ExpressionScale` `obs_scale` (#627 × #486): the analytic
/// FOCEI packed gradient must still match Richardson reconverged FD of the dense marginal
/// across every θ/Ω/σ coord. Pins the numerical side of the outer half of
/// `expression_scale_with_correlated_residual_is_analytic_both_loops`.
#[test]
fn population_packed_gradient_block_sigma_expression_scale_matches_fd() {
    use crate::estimation::parameterization::pack_params;

    let src = BLOCK_SIGMA_1CPT.replace(
        "[structural_model]",
        "[scaling]\n  obs_scale = 1000 / V\n[structural_model]",
    );
    let model = parse_model_string(&src).expect("parse block_sigma + obs_scale");
    assert!(matches!(
        model.scaling,
        crate::types::ScalingSpec::ExpressionScale { .. }
    ));
    assert!(crate::sens::provider::analytic_outer_gradient_available(
        &model
    ));

    let theta = &[1.1, 11.0];
    let s1 = dense_subject(&model, theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    let s2 = dense_subject(&model, theta, &[0.25, 1.5, 3.0, 6.0, 12.0, 36.0]);
    let pop = pop_of(vec![s1, s2]);

    let mut template = model.default_params.clone();
    template.theta = theta.to_vec();
    let x = pack_params(&template);
    let params = unpack_params(&x, &template);
    let ehs: Vec<DVector<f64>> = pop
        .subjects
        .iter()
        .map(|s| DVector::from_vec(precise_ebe_corr(&model, s, &params)))
        .collect();

    let analytic = population_gradient_sens(&model, &pop, &template, &x, &ehs)
        .expect("block_sigma + obs_scale is analytic");

    let ofv = |xv: &[f64]| -> f64 {
        let p = unpack_params(xv, &template);
        2.0 * pop
            .subjects
            .iter()
            .map(|s| {
                let eta = precise_ebe_corr(&model, s, &p);
                marginal_nll_dense_at(&model, s, &p, &eta)
            })
            .sum::<f64>()
    };
    assert_grad_matches_richardson_fd(&x, &analytic, ofv, 2e-3, 1e-5);
}

/// Closed-form `iiv_on_ruv` + **M3 BLOQ** (#4c): the analytic FOCEI packed
/// gradient must match Richardson reconverged FD of ferx's scaled, censored
/// FOCEI marginal across every θ/Ω/σ coordinate. This exercises the censored ×
/// residual-eta cross-terms — `h·z` (inner column), `C·z`/`C·m·a` (true inner
/// Hessian, mixed η-θ), the `C·z·(∂v/∂σ)/2v` σ-cross — and the exclusion of
/// censored rows from `H̃`/`log|H̃|` (matching `gaussian_foce_accum`).
#[test]
fn population_packed_gradient_iiv_on_ruv_m3_matches_fd() {
    use crate::estimation::parameterization::pack_params;

    let mut model = parse_model_string(WARFARIN_RUV).expect("parse");
    model.bloq_method = crate::types::BloqMethod::M3;
    assert_eq!(model.residual_error_eta, Some(3));
    // Gate flip: closed-form M3 + iiv_on_ruv is now analytic on both loops.
    assert!(crate::sens::provider::analytic_outer_gradient_available(
        &model
    ));

    let theta = vec![0.22, 11.0, 1.4];
    // Censor the two latest (lowest-concentration) observations at an LLOQ above
    // their prediction, so the M3 `−logΦ` term is genuinely active.
    let mk = |times: &[f64]| -> Subject {
        let mut s = ruv_subject(&model, &theta, times);
        let n = s.obs_times.len();
        for j in (n - 2)..n {
            s.observations[j] *= 1.5;
            s.cens[j] = 1;
        }
        s
    };
    let pop = pop_of(vec![
        mk(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]),
        mk(&[0.25, 1.5, 3.0, 6.0, 12.0, 36.0, 72.0]),
    ]);

    let mut template = model.default_params.clone();
    template.theta = theta.clone();
    let x = pack_params(&template);
    let params = unpack_params(&x, &template);
    let ehs: Vec<DVector<f64>> = pop
        .subjects
        .iter()
        .map(|s| DVector::from_vec(precise_ebe_ruv(&model, s, &params)))
        .collect();

    let analytic = population_gradient_sens(&model, &pop, &template, &x, &ehs)
        .expect("M3 + iiv_on_ruv is analytic");

    let ofv = |xv: &[f64]| -> f64 {
        let p = unpack_params(xv, &template);
        2.0 * pop
            .subjects
            .iter()
            .map(|s| {
                let eta = precise_ebe_ruv(&model, s, &p);
                marginal_nll_at(&model, s, &p, &eta)
            })
            .sum::<f64>()
    };
    for k in 0..x.len() {
        let h = 1e-4 * (1.0 + x[k].abs());
        let fd_at = |hh: f64| -> f64 {
            let mut xp = x.clone();
            xp[k] += hh;
            let mut xm = x.clone();
            xm[k] -= hh;
            (ofv(&xp) - ofv(&xm)) / (2.0 * hh)
        };
        let f1 = fd_at(h);
        let f2 = fd_at(h / 2.0);
        let fd = (4.0 * f2 - f1) / 3.0; // Richardson
        eprintln!(
            "m3+ruv x[{k}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
            analytic[k],
            fd,
            (analytic[k] - fd).abs() / fd.abs().max(1e-12)
        );
        approx::assert_relative_eq!(analytic[k], fd, max_relative = 3e-3, epsilon = 1e-5);
    }
}

/// 2-cpt IV + **combined** residual error + IOV + `iiv_on_ruv` — coverage beyond
/// the 1-cpt/proportional base case (review #3: the analytic scope must be tested
/// across cpt/error combos, since a finite-but-wrong gradient has no FD fallback).
const IOV_RUV_2CPT_COMBINED: &str = r#"
[parameters]
  theta TVCL(0.22, 0.001, 10.0)
  theta TVV1(11.0, 0.1, 500.0)
  theta TVQ(0.5, 0.001, 50.0)
  theta TVV2(20.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.04
  omega ETA_RUV ~ 0.05
  kappa KAPPA_CL ~ 0.02
  sigma PROP_ERR ~ 0.1
  sigma ADD_ERR ~ 0.3
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  pk two_cpt_iv(cl=CL, v=V1, q=Q, v2=V2)
[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR)
  iiv_on_ruv = ETA_RUV
[fit_options]
  method     = focei
  iov_column = OCC
"#;

/// Same structure, no IOV — for the non-IOV M3 + `iiv_on_ruv` 2-cpt/combined case.
const RUV_2CPT_COMBINED: &str = r#"
[parameters]
  theta TVCL(0.22, 0.001, 10.0)
  theta TVV1(11.0, 0.1, 500.0)
  theta TVQ(0.5, 0.001, 50.0)
  theta TVV2(20.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.04
  omega ETA_RUV ~ 0.05
  sigma PROP_ERR ~ 0.1
  sigma ADD_ERR ~ 0.3
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  pk two_cpt_iv(cl=CL, v=V1, q=Q, v2=V2)
[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR)
  iiv_on_ruv = ETA_RUV
"#;

/// Two-occasion 2-cpt IV IOV + `iiv_on_ruv` subject (n_eta = 3 incl. ETA_RUV).
fn iov_ruv_2cpt_subject(model: &CompiledModel, theta: &[f64]) -> Subject {
    let obs_times = vec![0.5, 2.0, 6.0, 12.0, 25.0, 30.0, 36.0, 48.0];
    let occasions = vec![1u32, 1, 1, 1, 2, 2, 2, 2];
    let n = obs_times.len();
    let mut subject = Subject {
        id: "1".to_string(),
        doses: vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
        ],
        obs_times,
        obs_raw_times: Vec::new(),
        observations: vec![0.0; n],
        obs_cmts: vec![1; n],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0; n],
        occasions,
        obs_l2: Vec::new(),
        dose_occasions: vec![1, 2],
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    let preds = crate::pk::predict_iov(
        model,
        &subject,
        theta,
        &[0.12, -0.08, 0.1],
        &[vec![0.05], vec![-0.07]],
    );
    subject.observations = preds.iter().map(|p| p * 0.9).collect();
    subject
}

/// Non-IOV `iiv_on_ruv` subject with a caller-supplied `eta_ref` (length `n_eta`),
/// single IV bolus into the central compartment.
fn ruv_subject_eta(
    model: &CompiledModel,
    theta: &[f64],
    times: &[f64],
    eta_ref: &[f64],
) -> Subject {
    let n = times.len();
    let mut subject = Subject {
        id: "1".to_string(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: times.to_vec(),
        obs_raw_times: Vec::new(),
        observations: vec![0.0; n],
        obs_cmts: vec![1; n],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0; n],
        occasions: vec![1; n],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    let preds = crate::pk::compute_predictions_with_tv(model, &subject, theta, eta_ref);
    subject.observations = preds.iter().map(|p| p * 0.9).collect();
    subject
}

/// Coverage (#1): IOV + `iiv_on_ruv` on a 2-cpt IV + combined-error model through
/// the production `subject_packed_gradient_iov` path — analytic vs reconverged FD.
#[test]
fn iov_iiv_on_ruv_2cpt_combined_packed_gradient_matches_fd() {
    let model =
        parse_model_string(IOV_RUV_2CPT_COMBINED).expect("parse 2cpt IOV + iiv_on_ruv combined");
    assert_eq!(model.residual_error_eta, Some(2));
    let theta = vec![0.22, 11.0, 0.5, 20.0];
    let mut params = model.default_params.clone();
    params.theta = theta.clone();
    let subject = iov_ruv_2cpt_subject(&model, &theta);
    let template = params.clone();
    let x = crate::estimation::parameterization::pack_params(&params);
    let (stacked, _e, _k, _h) = precise_ebe_iov(&model, &subject, &params);
    let analytic = subject_packed_gradient_iov(&model, &subject, &template, &x, &stacked)
        .expect("2cpt IOV + iiv_on_ruv packed gradient supported");
    let f = |xx: &[f64]| -> f64 {
        let p = unpack_params(xx, &template);
        marginal_nll_iov(&model, &subject, &p)
    };
    for i in 0..x.len() {
        let h = 1e-4 * (1.0 + x[i].abs());
        let fd_at = |hh: f64| -> f64 {
            let mut xp = x.clone();
            xp[i] += hh;
            let mut xm = x.clone();
            xm[i] -= hh;
            (f(&xp) - f(&xm)) / (2.0 * hh)
        };
        let f1 = fd_at(h);
        let f2 = fd_at(h / 2.0);
        let fd = (4.0 * f2 - f1) / 3.0;
        approx::assert_relative_eq!(analytic[i], fd, max_relative = 4e-3, epsilon = 2e-5);
    }
}

/// Coverage (#1): non-IOV M3 BLOQ + `iiv_on_ruv` on a 2-cpt IV + combined-error
/// model through the production population packed gradient — analytic vs
/// reconverged FD of the censored FOCEI marginal.
#[test]
fn iiv_on_ruv_m3_2cpt_combined_packed_gradient_matches_fd() {
    use crate::estimation::parameterization::pack_params;
    let mut model = parse_model_string(RUV_2CPT_COMBINED).expect("parse 2cpt ruv combined");
    model.bloq_method = crate::types::BloqMethod::M3;
    assert_eq!(model.residual_error_eta, Some(2));
    assert!(crate::sens::provider::analytic_outer_gradient_available(
        &model
    ));
    let theta = vec![0.22, 11.0, 0.5, 20.0];
    let mk = |times: &[f64]| -> Subject {
        let mut s = ruv_subject_eta(&model, &theta, times, &[0.12, -0.08, 0.0]);
        let n = s.obs_times.len();
        for j in (n - 2)..n {
            s.observations[j] *= 1.4;
            s.cens[j] = 1;
        }
        s
    };
    let pop = pop_of(vec![
        mk(&[0.5, 2.0, 6.0, 12.0, 24.0, 48.0]),
        mk(&[1.0, 3.0, 8.0, 16.0, 36.0, 72.0]),
    ]);
    let mut template = model.default_params.clone();
    template.theta = theta.clone();
    let x = pack_params(&template);
    let params = unpack_params(&x, &template);
    let ehs: Vec<DVector<f64>> = pop
        .subjects
        .iter()
        .map(|s| DVector::from_vec(precise_ebe_ruv(&model, s, &params)))
        .collect();
    let analytic = population_gradient_sens(&model, &pop, &template, &x, &ehs)
        .expect("2cpt combined M3 + iiv_on_ruv analytic");
    let ofv = |xv: &[f64]| -> f64 {
        let p = unpack_params(xv, &template);
        2.0 * pop
            .subjects
            .iter()
            .map(|s| {
                let eta = precise_ebe_ruv(&model, s, &p);
                marginal_nll_at(&model, s, &p, &eta)
            })
            .sum::<f64>()
    };
    for k in 0..x.len() {
        let h = 1e-4 * (1.0 + x[k].abs());
        let fd_at = |hh: f64| -> f64 {
            let mut xp = x.clone();
            xp[k] += hh;
            let mut xm = x.clone();
            xm[k] -= hh;
            (ofv(&xp) - ofv(&xm)) / (2.0 * hh)
        };
        let f1 = fd_at(h);
        let f2 = fd_at(h / 2.0);
        let fd = (4.0 * f2 - f1) / 3.0;
        approx::assert_relative_eq!(analytic[k], fd, max_relative = 4e-3, epsilon = 2e-5);
    }
}

/// The same residual-eta gradient must be exact on an **ODE** model: the
/// assembly is provider-agnostic, so the ODE `Dual2` sensitivities feed the
/// residual-eta `H̃`/`H`/`log|H̃|` terms exactly as the closed-form ones do
/// (#474). Confirms ODE + `iiv_on_ruv` is analytic, not FD.
#[test]
fn population_packed_gradient_ode_iiv_on_ruv_matches_fd() {
    let model = parse_model_string(TWOCPT_ODE_RUV).expect("parse ODE ruv");
    assert_eq!(model.residual_error_eta, Some(2));
    assert!(
        crate::sens::provider::analytic_outer_gradient_available(&model),
        "ODE + iiv_on_ruv must route to the analytic outer gradient (#474)"
    );
    run_ruv_packed_check(&model, &[4.0, 12.0, 2.0, 25.0]);
}

/// LTBS (`log_additive`) + `iiv_on_ruv` on an ODE model. The provider applies
/// the `g = ln(f)` chain to the sensitivities, so the residual-eta variance
/// terms (additive `R = σ²` on the log scale, `d = 0`) feed the same provider-
/// agnostic assembly — the analytic outer gradient must still match FD (#474).
/// (The inner EBE keeps FD for LTBS by design; the outer gradient is analytic.)
const TWOCPT_ODE_LTBS_RUV: &str = r#"
[parameters]
  theta TVCL(4.0,  0.1, 100.0)
  theta TVV1(12.0, 1.0, 500.0)
  theta TVQ(2.0,   0.01, 100.0)
  theta TVV2(25.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.04
  omega ETA_RUV ~ 0.10
  sigma ADD_ERR ~ 0.05
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  ode(states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  y = central / V1
[error_model]
  DV ~ log_additive(ADD_ERR)
  iiv_on_ruv = ETA_RUV
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

#[test]
fn population_packed_gradient_ode_ltbs_iiv_on_ruv_matches_fd() {
    let model = parse_model_string(TWOCPT_ODE_LTBS_RUV).expect("parse ODE LTBS ruv");
    assert!(model.log_transform, "log_additive must set LTBS");
    assert_eq!(model.residual_error_eta, Some(2));
    assert!(
        crate::sens::provider::analytic_outer_gradient_available(&model),
        "ODE + LTBS + iiv_on_ruv must route to the analytic outer gradient (#474)"
    );
    run_ruv_packed_check(&model, &[4.0, 12.0, 2.0, 25.0]);
}

#[test]
fn eta_dx_matches_fd() {
    use crate::estimation::parameterization::pack_params;
    let model = parse_model_string(WARFARIN).expect("parse");
    let theta = vec![0.22, 11.0, 1.4];
    let subject = subject_with_obs(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    let mut template = model.default_params.clone();
    template.theta = theta.clone();
    let x = pack_params(&template);
    let params = unpack_params(&x, &template);
    let eta_hat = precise_ebe(&model, &subject, &params);

    let jac = subject_eta_dx(&model, &subject, &template, &x, &eta_hat).expect("supported");
    let n_eta = model.n_eta;
    for k in 0..x.len() {
        let h = 1e-5 * (1.0 + x[k].abs());
        let mut xp = x.clone();
        xp[k] += h;
        let mut xm = x.clone();
        xm[k] -= h;
        let ep = precise_ebe(&model, &subject, &unpack_params(&xp, &template));
        let em = precise_ebe(&model, &subject, &unpack_params(&xm, &template));
        for l in 0..n_eta {
            let fd = (ep[l] - em[l]) / (2.0 * h);
            approx::assert_relative_eq!(jac[k][l], fd, max_relative = 2e-3, epsilon = 1e-6);
        }
    }
}

/// #474 regression: `subject_eta_dx` for an `iiv_on_ruv` model. The σ columns
/// of `dη̂/dx` must carry the `exp(2·η_ruv)` scale and the residual-eta row of
/// `M_σ`, matching FD of the (scaled) EBE — this guards the parity with
/// `sigma_block` that an earlier revision broke (dropped `ruv_scale` + the
/// residual row, silently wrong σ columns).
#[test]
fn eta_dx_matches_fd_iiv_on_ruv() {
    use crate::estimation::parameterization::pack_params;
    let model = parse_model_string(WARFARIN_RUV).expect("parse");
    let theta = vec![0.22, 11.0, 1.4];
    let subject = ruv_subject(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    let mut template = model.default_params.clone();
    template.theta = theta.clone();
    let x = pack_params(&template);
    let params = unpack_params(&x, &template);
    let eta_hat = precise_ebe_ruv(&model, &subject, &params);

    let jac = subject_eta_dx(&model, &subject, &template, &x, &eta_hat).expect("supported");
    let n_eta = model.n_eta;
    for k in 0..x.len() {
        let h = 1e-5 * (1.0 + x[k].abs());
        let mut xp = x.clone();
        xp[k] += h;
        let mut xm = x.clone();
        xm[k] -= h;
        let ep = precise_ebe_ruv(&model, &subject, &unpack_params(&xp, &template));
        let em = precise_ebe_ruv(&model, &subject, &unpack_params(&xm, &template));
        for l in 0..n_eta {
            let fd = (ep[l] - em[l]) / (2.0 * h);
            approx::assert_relative_eq!(jac[k][l], fd, max_relative = 2e-3, epsilon = 1e-6);
        }
    }
}

fn run_population_packed_gradient_check(model: &CompiledModel, theta: &[f64]) {
    use crate::estimation::parameterization::pack_params;

    let s1 = subject_with_obs(model, theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    let s2 = subject_with_obs(model, theta, &[0.25, 1.5, 3.0, 6.0, 12.0, 36.0, 72.0]);
    let pop = pop_of(vec![s1, s2]);

    let mut template = model.default_params.clone();
    template.theta = theta.to_vec();
    let x = pack_params(&template);

    let params = unpack_params(&x, &template);
    let ehs: Vec<DVector<f64>> = pop
        .subjects
        .iter()
        .map(|s| DVector::from_vec(precise_ebe(model, s, &params)))
        .collect();

    let analytic = population_gradient_sens(model, &pop, &template, &x, &ehs).expect("supported");

    let ofv = |xv: &[f64]| -> f64 {
        let p = unpack_params(xv, &template);
        2.0 * pop
            .subjects
            .iter()
            .map(|s| marginal_nll(model, s, &p))
            .sum::<f64>()
    };
    assert_grad_matches_richardson_fd(&x, &analytic, ofv, 3e-3, 1e-5);
}

#[test]
fn population_packed_gradient_matches_reconverged_fd() {
    let model = parse_model_string(WARFARIN).expect("parse");
    run_population_packed_gradient_check(&model, &[0.22, 11.0, 1.4]);
}

#[test]
fn population_packed_gradient_2cpt_matches_fd() {
    let model = parse_model_string(TWOCPT).expect("parse");
    run_population_packed_gradient_check(&model, &[5.0, 30.0, 2.0, 50.0, 1.0]);
}

// 2-cpt IV **user-ODE** model (Form C readout `y = central/V1`), IIV on CL+V1.
// Exercises the armed ODE sensitivity provider (#410) through the *full* outer
// assembly: the Dual2 augmented-RK45 jet must flow through the θ/Ω/σ blocks
// (incl. the EBE response) and match reconverged FD exactly as the analytical
// PK models do. Tight ODE tolerances so the propagated derivative agrees with a
// finite difference of the (separately integrated) f64 objective.
const TWOCPT_ODE_OUTER: &str = r#"
[parameters]
  theta TVCL(4.0,  0.1, 100.0)
  theta TVV1(12.0, 1.0, 500.0)
  theta TVQ(2.0,   0.01, 100.0)
  theta TVV2(25.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.04
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  ode(states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  y = central / V1
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

/// The armed ODE outer gradient (#410) must match reconverged Richardson FD of
/// the FOCEI marginal — the end-to-end proof that flipping `ODE_SENS_ENABLED`
/// feeds a *correct* θ/Ω/σ gradient through the shared assembly, not just that
/// the per-observation provider matches production (the `ode_provider` tests).
#[test]
fn population_packed_gradient_ode_2cpt_matches_fd() {
    let model = parse_model_string(TWOCPT_ODE_OUTER).expect("parse ODE");
    assert!(
        crate::sens::provider::sens_supported(&model),
        "2-cpt IV ODE must be armed for the analytic outer gradient (#410)"
    );
    run_population_packed_gradient_check(&model, &[4.0, 12.0, 2.0, 25.0]);
}

// 1-cpt IV (log-normal CL/V) used by the EVID=3/4 reset gradient checks: the
// provider rebuilds each observation from the doses in its current reset
// segment, so a reset subject's `∂f/∂η`, `∂²f/∂η²`, `∂f/∂θ`, `∂²f/∂η∂θ` jet —
// and therefore the assembled θ/Ω/σ packed gradient — must still match
// reconverged FD with no special-casing in the outer assembly.
const ONECPT_IV_RESET: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// As [`ONECPT_IV_RESET`] but with a bioavailability `F`. A **rate-defined** infusion
/// (`RATE > 0`) under `F ≠ 1` is a deliberate FD cell — `F` reshapes the infusion window
/// (#419) in a way the closed-form dual kernels can't represent — so an infusion subject on
/// this model is per-subject out of analytic scope, while a bolus subject stays in scope.
/// Used as the mixed analytic/FD population fixture.
const ONECPT_IV_RESET_F: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVF(0.8, 0.05, 1.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  F  = TVF
[structural_model]
  pk one_cpt_iv(cl=CL, v=V, f=F)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// Two IV-infusion occasions separated by an EVID=4 reset at t=120: occasion-2
/// observations must rebuild from zero (no carryover across the reset). The
/// observations are synthesised from the production predictor at a reference η
/// so residuals are realistic and nonzero.
fn reset_subject_outer(model: &CompiledModel, theta: &[f64], eta_ref: &[f64], id: &str) -> Subject {
    let obs_times = vec![2.0, 4.0, 8.0, 60.0, 122.0, 126.0, 150.0];
    let n = obs_times.len();
    let mut subject = Subject {
        id: id.to_string(),
        doses: vec![
            DoseEvent::new(0.0, 1000.0, 1, 200.0, false, 0.0),
            DoseEvent::new(120.0, 1000.0, 1, 200.0, false, 0.0),
        ],
        obs_times,
        obs_raw_times: Vec::new(),
        observations: vec![0.0; n],
        obs_cmts: vec![1; n],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: vec![120.0],
        cens: vec![0; n],
        occasions: vec![1; n],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    assert!(subject.has_resets(), "fixture must carry a reset");
    let preds = crate::pk::compute_predictions_with_tv(model, &subject, theta, eta_ref);
    subject.observations = preds.iter().map(|p| p * 0.85).collect();
    subject
}

// 1-cpt IV user-ODE for the SS+reset regression: SS bolus (II=24) establishes
// steady state, an EVID 3/4 reset at t=60 zeros the carryover, and a re-dose
// restarts. Tight tolerances so the dual jet agrees with FD of the f64 objective.
const ONECPT_IV_ODE_SS_RESET: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

/// Steady-state dosing **combined with an EVID 3/4 reset** is served analytically
/// on the ODE path: the static walk declines SS, so the subject routes to the
/// event-driven walk (`ode_tvcov_supported`), which admits SS + reset with no joint
/// exclusion. The analytic outer gradient must match Richardson FD of the FOCEI
/// marginal. (The closed-form path keeps this combination on FD — the analytical
/// superposition gates SS+reset. Pins the 2026-06-26 audit finding for #486.)
#[test]
fn population_packed_gradient_ode_ss_reset_matches_fd() {
    use crate::estimation::parameterization::pack_params;
    use crate::types::DoseEvent;

    let model = parse_model_string(ONECPT_IV_ODE_SS_RESET).expect("parse ODE SS+reset");
    assert!(model.is_ode_based(), "must be on the ODE path");
    let theta = [0.2, 10.0];
    let eta_ref = [0.1, -0.1];
    let obs_times = vec![2.0, 8.0, 20.0, 62.0, 70.0, 90.0];
    let n = obs_times.len();
    let mut s = Subject {
        id: "1".into(),
        doses: vec![
            DoseEvent::new(0.0, 1000.0, 1, 0.0, true, 24.0),
            DoseEvent::new(60.0, 1000.0, 1, 0.0, false, 0.0),
        ],
        obs_times,
        obs_raw_times: Vec::new(),
        observations: vec![0.0; n],
        obs_cmts: vec![1; n],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: vec![60.0],
        cens: vec![0; n],
        occasions: vec![1; n],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    assert!(
        s.has_resets() && s.doses.iter().any(|d| d.ss),
        "fixture is SS + reset"
    );
    // Routes to the event-driven walk (static walk declines SS), and that walk
    // admits SS + reset — the precondition for the analytic gradient below.
    assert!(
        crate::sens::ode_provider::ode_tvcov_supported(&model, &s),
        "ODE event-driven walk must admit SS + reset"
    );
    assert!(
        !crate::sens::ode_provider::ode_subject_supported(&model, &s),
        "static walk declines SS, so SS + reset must route to the event-driven walk"
    );

    let preds = crate::pk::compute_predictions_with_tv(&model, &s, &theta, &eta_ref);
    s.observations = preds.iter().map(|p| p * 0.85).collect();

    let pop = pop_of(vec![s]);
    let mut template = model.default_params.clone();
    template.theta = theta.to_vec();
    let x = pack_params(&template);
    let params = unpack_params(&x, &template);
    let ehs: Vec<DVector<f64>> = pop
        .subjects
        .iter()
        .map(|s| DVector::from_vec(precise_ebe(&model, s, &params)))
        .collect();
    let analytic = population_gradient_sens(&model, &pop, &template, &x, &ehs)
        .expect("ODE SS + reset must be served analytically");
    let ofv = |xv: &[f64]| -> f64 {
        let p = unpack_params(xv, &template);
        2.0 * pop
            .subjects
            .iter()
            .map(|s| marginal_nll(&model, s, &p))
            .sum::<f64>()
    };
    for k in 0..x.len() {
        let h = 1e-4 * (1.0 + x[k].abs());
        let mut xp = x.clone();
        xp[k] += h;
        let mut xm = x.clone();
        xm[k] -= h;
        let f1 = (ofv(&xp) - ofv(&xm)) / (2.0 * h);
        let mut xp2 = x.clone();
        xp2[k] += h / 2.0;
        let mut xm2 = x.clone();
        xm2[k] -= h / 2.0;
        let f2 = (ofv(&xp2) - ofv(&xm2)) / (2.0 * (h / 2.0));
        let fd = (4.0 * f2 - f1) / 3.0;
        approx::assert_relative_eq!(analytic[k], fd, max_relative = 5e-3, epsilon = 1e-5);
    }
}

/// FOCEI and FOCE packed gradients for a population containing a reset-bearing
/// subject must both match Richardson reconverged-FD of their respective
/// marginal objectives. This is the outer-assembly counterpart to the
/// provider-vs-production reset tests in `sens::provider`: it confirms the
/// reset segment's jet flows correctly through the θ/Ω/σ blocks (incl. the EBE
/// response) for both estimation methods.
#[test]
fn population_packed_gradient_reset_matches_fd() {
    use crate::estimation::parameterization::pack_params;

    let model = parse_model_string(ONECPT_IV_RESET).expect("parse");
    let theta = [0.22, 11.0];
    let eta_ref = [0.12, -0.08];

    // One reset subject + one ordinary subject, so the population mixes both.
    let s_reset = reset_subject_outer(&model, &theta, &eta_ref, "reset");
    let s_plain = subject_with_obs(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    let pop = pop_of(vec![s_reset, s_plain]);

    let mut template = model.default_params.clone();
    template.theta = theta.to_vec();
    let x = pack_params(&template);
    let params = unpack_params(&x, &template);
    let ehs: Vec<DVector<f64>> = pop
        .subjects
        .iter()
        .map(|s| DVector::from_vec(precise_ebe(&model, s, &params)))
        .collect();

    // Both FOCEI (Almquist Laplace) and FOCE (Sheiner–Beal) paths.
    for interaction in [true, false] {
        let analytic = if interaction {
            population_gradient_sens(&model, &pop, &template, &x, &ehs)
        } else {
            population_gradient_sens_foce(&model, &pop, &template, &x, &ehs)
        }
        .expect("reset subject supported by analytic gradient");

        let ofv = |xv: &[f64]| -> f64 {
            let p = unpack_params(xv, &template);
            2.0 * pop
                .subjects
                .iter()
                .map(|s| {
                    if interaction {
                        marginal_nll(&model, s, &p)
                    } else {
                        marginal_nll_foce(&model, s, &p)
                    }
                })
                .sum::<f64>()
        };
        assert_grad_matches_richardson_fd(&x, &analytic, ofv, 3e-3, 1e-5);
    }
}

/// **SS + EVID 3/4 reset is analytic since #486.** The pair used to decline to FD on the
/// closed-form engine (it is genuinely inexpressible by dose superposition, which folds in
/// the SS dose's infinite periodic history that the reset truncates), while the ODE engine
/// served it via #550. Routing these subjects to the event walk — which production already
/// uses for every reset subject — closes the cell: the walk equilibrates each SS dose at
/// its own event and zeros the dual state at the reset.
///
/// Anchored against reconverged FD of the FOCEI **and** FOCE OFV across every packed
/// coordinate (θ, Ω, σ), the same oracle `population_packed_gradient_reset_matches_fd` uses.
#[test]
fn ss_reset_subject_is_analytic_and_matches_fd() {
    use crate::estimation::parameterization::pack_params;

    let model = parse_model_string(ONECPT_IV_RESET).expect("parse");
    let theta = [0.22, 11.0];
    let eta_ref = [0.12, -0.08];

    let s_ss_reset = ss_reset_subject_outer(&model, &theta, &eta_ref, "ss_reset");
    let s_plain = subject_with_obs(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);

    // The cell this test exists for: the SS+reset subject is in analytic scope now.
    let zeros = vec![0.0; model.n_eta];
    assert!(
        crate::sens::provider::subject_sensitivities(&model, &s_ss_reset, &theta, &zeros).is_some(),
        "SS+reset subject must be served analytically by the event walk (#486)"
    );

    let pop = pop_of(vec![s_ss_reset, s_plain]);

    let mut template = model.default_params.clone();
    template.theta = theta.to_vec();
    let x = pack_params(&template);
    let params = unpack_params(&x, &template);
    let ehs: Vec<DVector<f64>> = pop
        .subjects
        .iter()
        .map(|s| DVector::from_vec(precise_ebe(&model, s, &params)))
        .collect();

    for interaction in [true, false] {
        let analytic = if interaction {
            population_gradient_sens(&model, &pop, &template, &x, &ehs)
        } else {
            population_gradient_sens_foce(&model, &pop, &template, &x, &ehs)
        }
        .expect("SS+reset population must take the all-or-nothing analytic path");

        let ofv = |xv: &[f64]| -> f64 {
            let p = unpack_params(xv, &template);
            2.0 * pop
                .subjects
                .iter()
                .map(|s| {
                    if interaction {
                        marginal_nll(&model, s, &p)
                    } else {
                        marginal_nll_foce(&model, s, &p)
                    }
                })
                .sum::<f64>()
        };
        assert_grad_matches_richardson_fd(&x, &analytic, ofv, 3e-3, 1e-5);
    }
}

/// Sibling of [`reset_subject_outer`]: same two IV-infusion occasions split by an EVID=4
/// reset, but the doses are **steady-state**. Served analytically since #486 — the event
/// walk equilibrates each SS dose per event and zeros the state at the reset, mirroring the
/// `f64` instance of the same walk that production already uses for reset subjects. (Only
/// the *static* dose-superposition path cannot express it, since an SS dose folds in an
/// infinite periodic history the reset truncates.)
fn ss_reset_subject_outer(
    model: &CompiledModel,
    theta: &[f64],
    eta_ref: &[f64],
    id: &str,
) -> Subject {
    let mut subject = reset_subject_outer(model, theta, eta_ref, id);
    subject.doses = vec![
        DoseEvent::new(0.0, 1000.0, 1, 200.0, true, 24.0),
        DoseEvent::new(120.0, 1000.0, 1, 200.0, true, 24.0),
    ];
    assert!(
        subject.doses.iter().any(|d| d.ss) && subject.has_resets(),
        "fixture must be steady-state + reset"
    );
    let preds = crate::pk::compute_predictions_with_tv(model, &subject, theta, eta_ref);
    subject.observations = preds.iter().map(|p| p * 0.85).collect();
    subject
}

/// Regression for focei-slsqp-fixed-ebe-gradient-bias: a population mixing
/// in-scope subjects with a single out-of-scope (SS+reset) subject must still
/// yield the exact analytic gradient for the in-scope subjects, filling only
/// the out-of-scope one with a reconverged per-subject FD. Before the fix one
/// such subject forced `population_gradient_sens` to `None`, dropping the
/// whole population onto the θ-only fixed-EBE gradient whose biased Ω/σ block
/// left the variance components pinned at their start and stalled SLSQP/
/// L-BFGS/MMA. The assembled `population_gradient_sens_mixed` must match
/// reconverged-FD of the FOCEI OFV across every packed coordinate.
#[test]
fn mixed_gradient_with_out_of_scope_subject_matches_fd() {
    use crate::estimation::outer_optimizer::population_gradient_sens_mixed;
    use crate::estimation::parameterization::{compute_bounds, pack_params};
    use crate::types::FitOptions;

    // The out-of-scope subject used to be SS+reset; since #486 the event walk serves that
    // combination analytically (see `ss_reset_subject_is_analytic_and_matches_fd`), so this
    // fixture now takes its out-of-scope subject from a cell that is *deliberately* FD: a
    // rate-defined infusion under `F ≠ 1` (#419).
    let model = parse_model_string(ONECPT_IV_RESET_F).expect("parse");
    let theta = [0.22, 11.0, 0.8];
    let eta_ref = [0.12, -0.08];

    // In-scope plain (bolus) subject + an out-of-scope rate-defined-infusion subject.
    let s_plain = subject_with_obs(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    let s_oos = reset_subject_outer(&model, &theta, &eta_ref, "rate_inf_f");
    let pop = pop_of(vec![s_plain, s_oos]);

    let mut template = model.default_params.clone();
    template.theta = theta.to_vec();
    let x = pack_params(&template);
    let params = unpack_params(&x, &template);

    // EBE per subject: in-scope subjects use the analytic Newton polish
    // (`precise_ebe`); the out-of-scope infusion subject uses the production
    // inner solver (`find_ebe`), which `precise_ebe` can't because it unwraps
    // the analytic provider.
    let zeros = vec![0.0; model.n_eta];
    let in_scope = |s: &Subject| {
        crate::sens::provider::subject_sensitivities(&model, s, &params.theta, &zeros).is_some()
    };
    let ehs: Vec<DVector<f64>> = pop
        .subjects
        .iter()
        .map(|s| {
            if in_scope(s) {
                DVector::from_vec(precise_ebe(&model, s, &params))
            } else {
                find_ebe(&model, s, &params, 200, 1e-12, None, None, 0).eta
            }
        })
        .collect();

    // Pre-fix behaviour: the all-or-nothing analytic gradient declines the
    // whole population because subject 1 (SS+reset) is out of scope.
    assert!(
        population_gradient_sens(&model, &pop, &template, &x, &ehs).is_none(),
        "out-of-scope subject must take the whole population out of the all-or-nothing path"
    );
    // The per-subject view keeps the in-scope subject analytic, only the
    // out-of-scope one `None`.
    let per_sub = per_subject_packed_gradients(&model, &pop, &template, &x, &ehs, true, None);
    assert!(per_sub[0].is_some(), "plain subject is in analytic scope");
    assert!(
        per_sub[1].is_none(),
        "rate-defined infusion under F is out of analytic scope"
    );

    // The assembled mixed gradient (analytic in-scope + per-subject FD for the
    // out-of-scope subject) must match reconverged-FD of the FOCEI OFV.
    let options = FitOptions {
        interaction: true,
        ..Default::default()
    };
    let bounds = compute_bounds(&template);
    let mixed =
        population_gradient_sens_mixed(&x, &template, &model, &pop, &ehs, &bounds, &options);

    // FD reference, per subject mirroring the mixed assembly: in-scope
    // subjects via the analytic-EBE `marginal_nll`, the out-of-scope one via
    // the production reconverged EBE + `foce_subject_nll` (exactly what the
    // mixed FD fallback computes internally).
    let subj_marginal = |s: &Subject, p: &ModelParameters| -> f64 {
        if in_scope(s) {
            marginal_nll(&model, s, p)
        } else {
            let ebe = find_ebe(&model, s, p, 200, 1e-12, None, None, 0);
            foce_subject_nll(
                &model,
                s,
                &p.theta,
                &ebe.eta,
                &ebe.h_matrix,
                &p.omega,
                &p.sigma.values,
                true,
            )
        }
    };
    let ofv = |xv: &[f64]| -> f64 {
        let p = unpack_params(xv, &template);
        2.0 * pop
            .subjects
            .iter()
            .map(|s| subj_marginal(s, &p))
            .sum::<f64>()
    };
    assert_grad_matches_richardson_fd(&x, &mixed, ofv, 3e-3, 1e-5);
}

// 1-cpt oral with allometric WT-on-CL — the canonical time-varying covariate.
// WT changes across a subject's records, so `CL = TVCL·(WT/70)^THETA_WT·exp(ETA_CL)`
// switches mid-decay. The provider routes these to the event-driven Dual2 walk
// and returns the standard `(η, θ)` jet, so the θ/Ω/σ packed gradient —
// including the THETA_WT covariate coefficient and the EBE response — must match
// reconverged FD with no special-casing in the outer assembly.
const ONECPT_ORAL_TVCOV_OUTER: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// TV-cov subject: one dose with `WT` changing across observations (and an
/// optional EVID=2 covariate breakpoint at `pk_only_times`). `dose` lets the
/// caller pass a steady-state dose. Observations are synthesised from the
/// production predictor at a reference η so residuals are realistic and nonzero.
#[allow(clippy::too_many_arguments)]
fn tvcov_subject_outer(
    model: &CompiledModel,
    theta: &[f64],
    eta_ref: &[f64],
    dose: DoseEvent,
    obs_times: &[f64],
    obs_wts: &[f64],
    pk_only_times: Vec<f64>,
    pk_only_wts: &[f64],
    id: &str,
) -> Subject {
    let n = obs_times.len();
    let wt_map = |w: f64| {
        let mut m = HashMap::new();
        m.insert("WT".to_string(), w);
        m
    };
    let mut subject = Subject {
        id: id.to_string(),
        doses: vec![dose],
        obs_times: obs_times.to_vec(),
        obs_raw_times: Vec::new(),
        observations: vec![0.0; n],
        obs_cmts: vec![1; n],
        covariates: wt_map(obs_wts[0]),
        dose_covariates: vec![wt_map(obs_wts[0])],
        obs_covariates: obs_wts.iter().map(|&w| wt_map(w)).collect(),
        pk_only_times,
        pk_only_covariates: pk_only_wts.iter().map(|&w| wt_map(w)).collect(),
        reset_times: Vec::new(),
        cens: vec![0; n],
        occasions: vec![1; n],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    assert!(
        subject.has_tv_covariates(),
        "fixture must carry TV covariates"
    );
    let preds = crate::pk::compute_predictions_with_tv(model, &subject, theta, eta_ref);
    subject.observations = preds.iter().map(|p| p * 0.85).collect();
    subject
}

/// FOCEI and FOCE packed gradients for a population with a time-varying-covariate
/// subject must both match Richardson reconverged-FD of their marginal
/// objectives — the outer-assembly counterpart to the provider-vs-production
/// TV-cov tests in `sens::provider`. One subject carries the covariate change
/// across observations, the other carries an EVID=2 breakpoint between them, so
/// both the covariate-θ chain and the `pk_only` walk flow through the θ/Ω/σ
/// blocks (incl. the EBE response) for both estimation methods.
#[test]
fn population_packed_gradient_tvcov_matches_fd() {
    use crate::estimation::parameterization::pack_params;

    let model = parse_model_string(ONECPT_ORAL_TVCOV_OUTER).expect("parse tvcov");
    let theta = [0.22, 11.0, 1.4, 0.7];
    let eta_ref = [0.12, -0.08, 0.2];

    let s_obs = tvcov_subject_outer(
        &model,
        &theta,
        &eta_ref,
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        &[1.0, 2.0, 4.0, 8.0, 24.0],
        &[70.0, 74.0, 82.0, 88.0, 95.0],
        Vec::new(),
        &[],
        "tvcov_obs",
    );
    let s_brk = tvcov_subject_outer(
        &model,
        &theta,
        &eta_ref,
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        &[1.0, 2.0, 6.0, 12.0],
        &[70.0, 70.0, 95.0, 95.0],
        vec![4.0],
        &[95.0],
        "tvcov_brk",
    );
    // Third subject: a steady-state (II=24) oral dose with WT changing across
    // observations, so the SS-equilibrated jet flows through the packed
    // gradient and its reconverged-FD reference too.
    let s_ss = tvcov_subject_outer(
        &model,
        &theta,
        &eta_ref,
        DoseEvent::new(0.0, 100.0, 1, 0.0, true, 24.0),
        &[1.0, 4.0, 9.0, 15.0, 22.0],
        &[70.0, 76.0, 84.0, 90.0, 96.0],
        Vec::new(),
        &[],
        "tvcov_ss",
    );
    assert!(
        s_ss.doses.iter().any(|d| d.ss),
        "SS fixture must carry an SS dose"
    );
    let pop = Population {
        subjects: vec![s_obs, s_brk, s_ss],
        covariate_names: vec!["WT".into()],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };

    let mut template = model.default_params.clone();
    template.theta = theta.to_vec();
    let x = pack_params(&template);
    let params = unpack_params(&x, &template);
    let ehs: Vec<DVector<f64>> = pop
        .subjects
        .iter()
        .map(|s| DVector::from_vec(precise_ebe(&model, s, &params)))
        .collect();

    for interaction in [true, false] {
        let analytic = if interaction {
            population_gradient_sens(&model, &pop, &template, &x, &ehs)
        } else {
            population_gradient_sens_foce(&model, &pop, &template, &x, &ehs)
        }
        .expect("TV-cov subject supported by analytic gradient");

        let ofv = |xv: &[f64]| -> f64 {
            let p = unpack_params(xv, &template);
            2.0 * pop
                .subjects
                .iter()
                .map(|s| {
                    if interaction {
                        marginal_nll(&model, s, &p)
                    } else {
                        marginal_nll_foce(&model, s, &p)
                    }
                })
                .sum::<f64>()
        };
        assert_grad_matches_richardson_fd(&x, &analytic, ofv, 3e-3, 1e-5);
    }
}

// 1-cpt oral with a parameter-dependent central baseline (#524). The analytic
// init impulse and its θ/η jet must flow through the packed FOCEI/FOCE
// population gradient — i.e. the gradient that `gradient = auto` uses must
// match Richardson FD of the marginal objective (`gradient = fd`), the
// population-level analogue of the per-subject provider-vs-FD init test.
const ONECPT_ORAL_INIT_OUTER: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta TVC0(5.0, 0.01, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[initial_conditions]
  init(central) = TVC0 * V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

#[test]
fn population_packed_gradient_init_matches_fd() {
    use crate::estimation::parameterization::pack_params;

    let model = parse_model_string(ONECPT_ORAL_INIT_OUTER).expect("parse init outer");
    assert_eq!(model.analytical_init.len(), 1);
    assert!(
        crate::sens::provider::analytical_supported(&model),
        "init model must use the analytic outer gradient, not FD"
    );
    let theta = [0.22, 11.0, 1.4, 6.0];
    let eta_ref = [0.12, -0.08, 0.2];

    // Two plain (non-TV) subjects with a baseline-bearing oral dose; obs are
    // the init-aware prediction at eta_ref scaled down so EBEs are non-trivial.
    let make = |id: &str, times: &[f64]| -> Subject {
        let n = times.len();
        let mut s = Subject {
            id: id.to_string(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: times.to_vec(),
            obs_raw_times: Vec::new(),
            observations: vec![0.0; n],
            obs_cmts: vec![1; n],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; n],
            occasions: vec![1; n],
            obs_l2: Vec::new(),
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
        };
        let preds = crate::pk::compute_predictions_with_tv(&model, &s, &theta, &eta_ref);
        s.observations = preds.iter().map(|p| p * 0.85).collect();
        s
    };
    let pop = pop_of(vec![
        make("init_a", &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]),
        make("init_b", &[1.0, 3.0, 6.0, 12.0]),
    ]);

    let mut template = model.default_params.clone();
    template.theta = theta.to_vec();
    let x = pack_params(&template);
    let params = unpack_params(&x, &template);
    let ehs: Vec<DVector<f64>> = pop
        .subjects
        .iter()
        .map(|s| DVector::from_vec(precise_ebe(&model, s, &params)))
        .collect();

    for interaction in [true, false] {
        let analytic = if interaction {
            population_gradient_sens(&model, &pop, &template, &x, &ehs)
        } else {
            population_gradient_sens_foce(&model, &pop, &template, &x, &ehs)
        }
        .expect("init subject supported by analytic gradient");

        let ofv = |xv: &[f64]| -> f64 {
            let p = unpack_params(xv, &template);
            2.0 * pop
                .subjects
                .iter()
                .map(|s| {
                    if interaction {
                        marginal_nll(&model, s, &p)
                    } else {
                        marginal_nll_foce(&model, s, &p)
                    }
                })
                .sum::<f64>()
        };
        assert_grad_matches_richardson_fd(&x, &analytic, ofv, 3e-3, 1e-5);
    }
}

// 1-cpt oral with a log-normal dose lagtime (`LAGTIME = TVLAG·exp(ETA_LAG)`):
// the lagtime θ (`TVLAG`) and ω (`ETA_LAG`) enter the packed gradient through
// the provider's `∂f/∂θ` / `∂²f/∂η∂θ` for the lag slot, with no special-casing.
const WARFARIN_LAG: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta TVLAG(0.75, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  omega ETA_LAG ~ 0.05
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
  LAGTIME = TVLAG * exp(ETA_LAG)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, lagtime=LAGTIME)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// Full packed-gradient check (8 params: 4 θ + 4 Ω-Cholesky + 1 σ) for a model
/// with a differentiated dose lagtime, vs Richardson reconverged-FD of the
/// marginal NLL. Confirms the lagtime axis flows through the Almquist assembly.
#[test]
fn population_packed_gradient_lagtime_matches_fd() {
    use crate::estimation::parameterization::{pack_params, unpack_params};
    use std::collections::HashMap;

    let model = parse_model_string(WARFARIN_LAG).expect("parse lag");
    let theta = [0.22, 11.0, 1.4, 0.7];

    // Two subjects, observations built at a 4-component reference η (all obs
    // times comfortably past the lagged arrival so residuals are smooth).
    let build = |times: &[f64]| -> Subject {
        let n = times.len();
        let mut s = Subject {
            id: "1".to_string(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: times.to_vec(),
            obs_raw_times: Vec::new(),
            observations: vec![0.0; n],
            obs_cmts: vec![1; n],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; n],
            occasions: vec![1; n],
            obs_l2: Vec::new(),
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
        };
        let eta_ref = [0.12, -0.08, 0.2, 0.1];
        let preds = crate::pk::compute_predictions_with_tv(&model, &s, &theta, &eta_ref);
        s.observations = preds.iter().map(|p| p * 0.85).collect();
        s
    };
    let pop = pop_of(vec![
        build(&[1.0, 2.0, 4.0, 8.0, 24.0]),
        build(&[1.5, 3.0, 6.0, 12.0, 36.0]),
    ]);

    let mut template = model.default_params.clone();
    template.theta = theta.to_vec();
    let x = pack_params(&template);
    let params = unpack_params(&x, &template);
    let ehs: Vec<DVector<f64>> = pop
        .subjects
        .iter()
        .map(|s| DVector::from_vec(precise_ebe(&model, s, &params)))
        .collect();

    let analytic = population_gradient_sens(&model, &pop, &template, &x, &ehs).expect("supported");
    let ofv = |xv: &[f64]| -> f64 {
        let p = unpack_params(xv, &template);
        2.0 * pop
            .subjects
            .iter()
            .map(|s| marginal_nll(&model, s, &p))
            .sum::<f64>()
    };
    assert_grad_matches_richardson_fd(&x, &analytic, ofv, 3e-3, 1e-5);
}

/// The analytic FOCEI **M3** packed gradient (censored rows enter `prepare`'s
/// M3 branch: data `−logΦ` + true-inner-Hessian + FOCEI-order `H̃`/`log|H̃|`
/// curvature) must match the reconverged-FD of ferx's M3 FOCEI objective
/// (`foce_subject_nll_interaction` with `bloq_term`). Each subject carries both
/// quantified and censored rows.
#[test]
fn population_packed_gradient_m3_matches_fd() {
    use crate::estimation::parameterization::pack_params;
    use crate::types::BloqMethod;

    let mut model = parse_model_string(WARFARIN).expect("parse");
    model.bloq_method = BloqMethod::M3;
    let theta = [0.22, 11.0, 1.4];

    // Build subjects, then mark the last two observations of each as censored
    // (CENS=1, the obs cell carries the LLOQ) so every subject mixes quantified
    // and BLOQ rows. Leaves z moderate (LLOQ ≈ 0.85·f_ref), away from the tail.
    let mut s1 = subject_with_obs(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    let mut s2 = subject_with_obs(&model, &theta, &[0.25, 1.5, 3.0, 6.0, 12.0, 36.0, 72.0]);
    for s in [&mut s1, &mut s2] {
        let n = s.observations.len();
        s.cens[n - 1] = 1;
        s.cens[n - 2] = 1;
    }
    assert!(s1.cens.iter().any(|&c| c != 0) && s2.cens.iter().any(|&c| c != 0));

    let pop = pop_of(vec![s1, s2]);

    let mut template = model.default_params.clone();
    template.theta = theta.to_vec();
    let x = pack_params(&template);
    let params = unpack_params(&x, &template);
    let ehs: Vec<DVector<f64>> = pop
        .subjects
        .iter()
        .map(|s| DVector::from_vec(precise_ebe(&model, s, &params)))
        .collect();

    let analytic =
        population_gradient_sens(&model, &pop, &template, &x, &ehs).expect("M3 supported");

    // marginal_nll uses foce_subject_nll_interaction with model.bloq_method = M3,
    // so the OFV carries the censored −2logΦ term; precise_ebe is M3-aware.
    let ofv = |xv: &[f64]| -> f64 {
        let p = unpack_params(xv, &template);
        2.0 * pop
            .subjects
            .iter()
            .map(|s| marginal_nll(&model, s, &p))
            .sum::<f64>()
    };
    assert_grad_matches_richardson_fd(&x, &analytic, ofv, 5e-3, 1e-5);
}

const FREM_OUTER_MODEL: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta TV_WT(72.0, 1.0, 500.0)

  block_omega (ETA_CL, ETA_V, ETA_KA, ETA_WT_FREM) = [
    0.09,
    0.0, 0.04,
    0.0, 0.0, 0.30,
    0.0, 0.0, 0.0, 111.56
  ]

  sigma PROP_ERR ~ 0.02
  sigma EPSCOV   ~ 0.30
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  frem_predictions = TV_WT/ETA_WT_FREM:100
  frem_sigma       = EPSCOV
"#;

/// A subject with 3 PK observation rows plus one FREM covariate pseudo-observation
/// (FREMTYPE 100), built without going through the data reader (mirrors
/// `agq::tests::frem_pseudo_obs_rows_get_the_right_analytic_score`'s fixture).
fn frem_outer_subject(model: &CompiledModel, theta: &[f64]) -> Subject {
    use std::collections::HashMap;
    let fc = model.frem_config.as_ref().expect("fixture must be FREM");
    let pk_times = [1.0, 6.0, 24.0];
    let n = pk_times.len() + 1;
    let mut subject = Subject {
        id: "1".to_string(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: pk_times
            .iter()
            .copied()
            .chain(std::iter::once(0.0))
            .collect(),
        obs_raw_times: Vec::new(),
        observations: vec![0.0; n],
        obs_cmts: vec![1; n],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0; n],
        occasions: vec![1; n],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        fremtype: pk_times
            .iter()
            .map(|_| 0u16)
            .chain(std::iter::once(100u16))
            .collect(),
        obs_records: vec![],
    };
    let eta_ref = [0.1, -0.05, 0.15, 0.0];
    let preds = crate::pk::compute_predictions_with_tv(model, &subject, theta, &eta_ref);
    for j in 0..pk_times.len() {
        subject.observations[j] = preds[j] * 0.85;
    }
    let (ti, _ei) = fc.fremtype_to_indices[&100u16];
    subject.observations[pk_times.len()] = theta[ti] * 1.10;
    subject
}

/// FREM covariate pseudo-observations, at the **FOCEI outer** (`population_gradient_sens`
/// / `theta_block` / `sigma_block`) level — not AGQ's `score_core` consumer, which
/// `agq::tests::frem_pseudo_obs_rows_get_the_right_analytic_score` already covers.
///
/// That AGQ test alone missed a real bug (#251 review #2): `sigma_block`'s σ-FD ran the
/// *plain PK* variance at σ±h on a FREM row instead of the `EPSCOV`-aware one, so
/// `grad[EPSCOV]` came out identically zero under FOCE/FOCEI while a spurious term leaked
/// into `PROP_ERR`'s gradient — invisible to any test that never differentiates through
/// `theta_block`/`sigma_block` on a FREM model. This test does, against FD of the true
/// FOCEI marginal (`marginal_nll`, which is itself now FREM-aware via
/// `foce_subject_nll_interaction`'s `frem_r_override`), covering every packed coordinate
/// (θ, Ω, and — critically — σ including `EPSCOV`).
#[test]
fn population_packed_gradient_frem_matches_fd() {
    use crate::estimation::parameterization::pack_params;
    use crate::types::Population;

    let model = parse_model_string(FREM_OUTER_MODEL).expect("parse");
    assert_eq!(model.n_eta, 4, "3 PK etas + 1 covariate eta");
    let theta = model.default_params.theta.clone();
    let subject = frem_outer_subject(&model, &theta);

    let pop = Population {
        subjects: vec![subject],
        covariate_names: vec![],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };

    let mut template = model.default_params.clone();
    template.theta = theta;
    let x = pack_params(&template);
    let params = unpack_params(&x, &template);
    let ehs: Vec<DVector<f64>> = pop
        .subjects
        .iter()
        .map(|s| DVector::from_vec(precise_ebe(&model, s, &params)))
        .collect();

    let analytic =
        population_gradient_sens(&model, &pop, &template, &x, &ehs).expect("FREM supported");

    let ofv = |xv: &[f64]| -> f64 {
        let p = unpack_params(xv, &template);
        2.0 * pop
            .subjects
            .iter()
            .map(|s| marginal_nll(&model, s, &p))
            .sum::<f64>()
    };
    let fd_at = |k: usize, h: f64| -> f64 {
        let mut xp = x.clone();
        xp[k] += h;
        let mut xm = x.clone();
        xm[k] -= h;
        (ofv(&xp) - ofv(&xm)) / (2.0 * h)
    };
    for k in 0..x.len() {
        let h = 1e-4 * (1.0 + x[k].abs());
        let f1 = fd_at(k, h);
        let f2 = fd_at(k, h / 2.0);
        let fd = (4.0 * f2 - f1) / 3.0;
        let scale = fd.abs().max(analytic[k].abs()).max(1.0);
        assert!(
            (analytic[k] - fd).abs() / scale < 5e-3,
            "coord {k}: analytic {} vs FD {} (rel {:.2e})",
            analytic[k],
            fd,
            (analytic[k] - fd).abs() / scale
        );
    }
}

// 1-cpt oral **user-ODE** model with M3 BLOQ, tight tolerances. ODE counterpart
// of the closed-form `population_packed_gradient_m3_matches_fd`: the censored
// `−logΦ` term enters `prepare`'s M3 branch on top of the ODE walk's `ObsSens`
// (the same provider-agnostic assembly), proving non-IOV ODE+M3 is analytic on
// the outer loop.
const ONECPT_ODE_M3_OUTER: &str = r#"
[parameters]
  theta TVCL(0.2,  0.001, 10.0)
  theta TVV(10.0,  0.1,  500.0)
  theta TVKA(1.5,  0.01,  50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot / V - (CL/V) * central
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method      = focei
  bloq_method = m3
  ode_reltol  = 1e-9
  ode_abstol  = 1e-11
"#;

/// ODE counterpart of [`population_packed_gradient_m3_matches_fd`]: the analytic
/// FOCEI M3 packed gradient assembled from the **event-driven ODE sensitivity
/// walk** (censored rows enter `prepare`'s M3 branch) must match reconverged FD
/// of the M3 FOCEI objective. Proves non-IOV ODE+M3 is analytic on the outer
/// loop (the inner counterpart lives in `inner_optimizer.rs`).
#[test]
fn population_packed_gradient_ode_m3_matches_fd() {
    use crate::estimation::parameterization::pack_params;
    use crate::types::BloqMethod;

    let model = parse_model_string(ONECPT_ODE_M3_OUTER).expect("parse ODE M3");
    assert!(matches!(model.bloq_method, BloqMethod::M3), "must be M3");
    assert!(model.is_ode_based(), "must be on the ODE path");
    let theta = [0.22, 11.0, 1.4];

    let mut s1 = subject_with_obs(&model, &theta, &[0.5, 1.0, 2.0, 8.0]);
    let mut s2 = subject_with_obs(&model, &theta, &[0.25, 1.5, 6.0, 12.0, 36.0]);
    for s in [&mut s1, &mut s2] {
        let n = s.observations.len();
        s.cens[n - 1] = 1;
        s.cens[n - 2] = 1;
    }
    assert!(s1.cens.iter().any(|&c| c != 0) && s2.cens.iter().any(|&c| c != 0));

    let pop = pop_of(vec![s1, s2]);

    let mut template = model.default_params.clone();
    template.theta = theta.to_vec();
    let x = pack_params(&template);
    let params = unpack_params(&x, &template);
    let ehs: Vec<DVector<f64>> = pop
        .subjects
        .iter()
        .map(|s| DVector::from_vec(precise_ebe(&model, s, &params)))
        .collect();

    let analytic =
        population_gradient_sens(&model, &pop, &template, &x, &ehs).expect("ODE M3 supported");

    let ofv = |xv: &[f64]| -> f64 {
        let p = unpack_params(xv, &template);
        2.0 * pop
            .subjects
            .iter()
            .map(|s| marginal_nll(&model, s, &p))
            .sum::<f64>()
    };
    assert_grad_matches_richardson_fd(&x, &analytic, ofv, 5e-3, 1e-5);
}

/// Non-IOV 1-cpt oral **user-ODE** model with M3 BLOQ **and** `iiv_on_ruv`
/// (`Y = IPRED + EPS·EXP(η_ruv)`) — [`ONECPT_ODE_M3_OUTER`] plus an extra residual-error
/// η that no structural parameter references. Drives the last `iiv_on_ruv` holdout
/// (#486): non-IOV ODE M3 + `iiv_on_ruv` on the outer loop.
const ONECPT_ODE_M3_RUV_OUTER: &str = r#"
[parameters]
  theta TVCL(0.2,  0.001, 10.0)
  theta TVV(10.0,  0.1,  500.0)
  theta TVKA(1.5,  0.01,  50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  omega ETA_RUV ~ 0.05
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot / V - (CL/V) * central
[error_model]
  DV ~ proportional(PROP_ERR)
  iiv_on_ruv = ETA_RUV
[fit_options]
  method      = focei
  bloq_method = m3
  ode_reltol  = 1e-10
  ode_abstol  = 1e-12
"#;

/// **Non-IOV ODE M3 + `iiv_on_ruv`** (#486 — the last `iiv_on_ruv` holdout, the #547
/// pattern): the ODE counterpart of [`population_packed_gradient_iiv_on_ruv_m3_matches_fd`].
/// The censored × residual-eta cross-terms (`h·z` inner column, `C·z`/`C·m·a` true-Hessian /
/// mixed blocks, the σ-cross) are applied by the provider-agnostic `prepare` over the
/// **event-driven ODE walk's** `ObsSens`, and censored rows enter `H̃`/`log|H̃|` at FOCEI
/// order, exactly as on the closed-form path. The FOCEI packed gradient must match Richardson
/// reconverged FD of the `exp(2·η_ruv)`-scaled, censored FOCEI marginal across every packed
/// coordinate — note the EBE must be reconverged with [`precise_ebe_ruv`] (which carries the
/// `exp(2·η_ruv)` variance scaling), not the plain [`precise_ebe`]. Both censoring tails.
#[test]
fn population_packed_gradient_ode_m3_iiv_on_ruv_matches_fd() {
    use crate::estimation::parameterization::pack_params;
    use crate::types::BloqMethod;

    let model = parse_model_string(ONECPT_ODE_M3_RUV_OUTER).expect("parse ODE M3 + iiv_on_ruv");
    assert!(matches!(model.bloq_method, BloqMethod::M3), "must be M3");
    assert!(model.is_ode_based(), "must be on the ODE path");
    assert_eq!(model.residual_error_eta, Some(3));
    assert!(
        crate::sens::provider::analytic_outer_gradient_available(&model),
        "non-IOV ODE M3 + iiv_on_ruv must route to the analytic outer gradient (#486)"
    );
    let theta = [0.22, 11.0, 1.4];

    for right in [false, true] {
        let mut s1 = subject_with_obs(&model, &theta, &[0.5, 1.0, 2.0, 8.0]);
        let mut s2 = subject_with_obs(&model, &theta, &[0.25, 1.5, 6.0, 12.0, 36.0]);
        for s in [&mut s1, &mut s2] {
            let n = s.observations.len();
            let tail = if right { -1 } else { 1 };
            s.cens[n - 1] = tail;
            s.cens[n - 2] = tail;
        }
        assert!(s1.cens.iter().any(|&c| c != 0) && s2.cens.iter().any(|&c| c != 0));

        let pop = pop_of(vec![s1, s2]);

        let mut template = model.default_params.clone();
        template.theta = theta.to_vec();
        let x = pack_params(&template);
        let params = unpack_params(&x, &template);
        // `precise_ebe_ruv` carries the `exp(2·η_ruv)` variance scaling — the plain
        // `precise_ebe` ignores the residual-eta and converges to the wrong EBE.
        let ehs: Vec<DVector<f64>> = pop
            .subjects
            .iter()
            .map(|s| DVector::from_vec(precise_ebe_ruv(&model, s, &params)))
            .collect();

        let analytic = population_gradient_sens(&model, &pop, &template, &x, &ehs)
            .expect("ODE M3 + iiv_on_ruv supported");

        let ofv = |xv: &[f64]| -> f64 {
            let p = unpack_params(xv, &template);
            2.0 * pop
                .subjects
                .iter()
                .map(|s| {
                    let eta = precise_ebe_ruv(&model, s, &p);
                    marginal_nll_at(&model, s, &p, &eta)
                })
                .sum::<f64>()
        };
        assert_grad_matches_richardson_fd(&x, &analytic, ofv, 3e-3, 2e-5);
    }
}

/// The analytic **FOCE** (Sheiner–Beal, non-interaction) M3 packed gradient
/// (censored rows excluded from R̃, added as `−logΦ((LLOQ−f̂)/√R⁰)` with the
/// population variance) must match the reconverged-FD of ferx's FOCE-M3
/// objective (`foce_subject_nll_standard` with the censored term).
#[test]
fn population_packed_gradient_m3_foce_matches_fd() {
    use crate::estimation::parameterization::pack_params;
    use crate::types::BloqMethod;

    let mut model = parse_model_string(WARFARIN).expect("parse");
    model.bloq_method = BloqMethod::M3;
    let theta = [0.22, 11.0, 1.4];

    let mut s1 = subject_with_obs(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    let mut s2 = subject_with_obs(&model, &theta, &[0.25, 1.5, 3.0, 6.0, 12.0, 36.0, 72.0]);
    for s in [&mut s1, &mut s2] {
        let n = s.observations.len();
        s.cens[n - 1] = 1;
        s.cens[n - 2] = 1;
    }

    let pop = pop_of(vec![s1, s2]);

    let mut template = model.default_params.clone();
    template.theta = theta.to_vec();
    let x = pack_params(&template);
    let params = unpack_params(&x, &template);
    let ehs: Vec<DVector<f64>> = pop
        .subjects
        .iter()
        .map(|s| DVector::from_vec(precise_ebe(&model, s, &params)))
        .collect();

    let analytic =
        population_gradient_sens_foce(&model, &pop, &template, &x, &ehs).expect("M3 FOCE");

    let ofv = |xv: &[f64]| -> f64 {
        let p = unpack_params(xv, &template);
        2.0 * pop
            .subjects
            .iter()
            .map(|s| marginal_nll_foce(&model, s, &p))
            .sum::<f64>()
    };
    assert_grad_matches_richardson_fd(&x, &analytic, ofv, 5e-3, 1e-5);
}

#[test]
fn population_packed_gradient_3cpt_matches_fd() {
    let model = parse_model_string(THREECPT).expect("parse");
    run_population_packed_gradient_check(&model, &[5.0, 10.0, 2.0, 20.0, 1.5, 30.0, 1.0]);
}

// --- FOCE (non-interaction, Sheiner–Beal linearized marginal) ---

#[test]
fn population_packed_gradient_foce_matches_fd() {
    let model = parse_model_string(WARFARIN).expect("parse");
    run_packed_check_foce(&model, &[0.22, 11.0, 1.4]);
}

#[test]
fn population_packed_gradient_foce_2cpt_matches_fd() {
    let model = parse_model_string(TWOCPT).expect("parse");
    run_packed_check_foce(&model, &[5.0, 30.0, 2.0, 50.0, 1.0]);
}

#[test]
fn population_packed_gradient_foce_3cpt_matches_fd() {
    let model = parse_model_string(THREECPT).expect("parse");
    run_packed_check_foce(&model, &[5.0, 10.0, 2.0, 20.0, 1.5, 30.0, 1.0]);
}

// 1-cpt IV ODE with a parameter-dependent `init(central) = BASE/V` baseline + a finite
// infusion — a headline `init` composition (#486). Exercises the full FOCE packed gradient
// `[θ, Ω, σ]` end to end: the analytic init impulse (seeded on the event-driven walk and
// decayed under the infusion forcing) must survive the outer θ/Ω/σ assembly and match a
// Richardson-reconverged FD of ferx's FOCE OFV.
const IV_INIT_INFUSION: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta TVBASE(300.0, 10.0, 5000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  omega ETA_BASE ~ 0.04
  sigma PROP ~ 0.04 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV  * exp(ETA_V)
  BASE = TVBASE * exp(ETA_BASE)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = BASE / V
  d/dt(central)  = -CL/V * central
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  method     = foce
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

#[test]
fn population_packed_gradient_foce_init_infusion_matches_fd() {
    use crate::estimation::parameterization::pack_params;
    let model = parse_model_string(IV_INIT_INFUSION).expect("parse init+infusion");
    let theta = vec![1.0, 20.0, 300.0];
    // Two subjects, each dosed by a finite IV infusion (rate 25 → 4 h window) on top of the
    // init baseline; obs straddle the infusion end.
    let mk = |times: &[f64]| -> Subject {
        let mut s = subject_with_obs(&model, &theta, times);
        s.doses = vec![DoseEvent::new(0.0, 100.0, 1, 25.0, false, 0.0)];
        assert!(s.doses[0].is_infusion());
        let eta_ref = [0.12, -0.08, 0.15];
        let preds = crate::pk::compute_predictions_with_tv(&model, &s, &theta, &eta_ref);
        s.observations = preds.iter().map(|p| p * 0.85).collect();
        s
    };
    let pop = pop_of(vec![
        mk(&[1.0, 2.0, 4.0, 6.0, 10.0]),
        mk(&[0.5, 3.0, 5.0, 8.0, 24.0]),
    ]);
    let mut template = model.default_params.clone();
    template.theta = theta.clone();
    let x = pack_params(&template);
    let params = unpack_params(&x, &template);
    let ehs: Vec<DVector<f64>> = pop
        .subjects
        .iter()
        .map(|s| DVector::from_vec(precise_ebe(&model, s, &params)))
        .collect();
    let analytic =
        population_gradient_sens_foce(&model, &pop, &template, &x, &ehs).expect("supported");
    let ofv = |xv: &[f64]| -> f64 {
        let p = unpack_params(xv, &template);
        2.0 * pop
            .subjects
            .iter()
            .map(|s| marginal_nll_foce(&model, s, &p))
            .sum::<f64>()
    };
    assert_grad_matches_richardson_fd(&x, &analytic, ofv, 3e-3, 1e-5);
}

// --- Eq. 48 EBE warm-start predictor: is it correct & better than plain warm? ---

/// The Eq. 48 predictor `η⁰ = η̂_prev + (dη̂/dx)·Δx` is a first-order Taylor
/// extrapolation of the EBE as the packed parameters move x_prev → x_new. So
/// against the *converged* EBE at x_new it must beat the plain warm-start
/// (reuse η̂_prev): the prediction error is `O(‖Δx‖²)` while the warm-start
/// error is `O(‖Δx‖)`. This walks several step sizes in a representative
/// direction and checks (a) prediction strictly beats warm for small steps,
/// and (b) the prediction/warm error ratio shrinks ∝ ‖Δx‖ (second order).
#[test]
fn eta_predictor_beats_warm_start() {
    use crate::estimation::parameterization::pack_params;

    let model = parse_model_string(TWOCPT).expect("parse");
    let theta = vec![5.0, 30.0, 2.0, 50.0, 1.0];
    let times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0, 48.0];
    let subjects = [
        subject_with_obs(&model, &theta, &times),
        subject_with_obs(&model, &theta, &[0.5, 1.5, 3.0, 6.0, 12.0, 36.0]),
    ];

    let mut template = model.default_params.clone();
    template.theta = theta.clone();
    let x0 = pack_params(&template);
    let n = x0.len();

    // A fixed, representative outer direction (unit-norm in packed space).
    let mut dir: Vec<f64> = (0..n).map(|k| 0.5 + 0.1 * k as f64).collect();
    let dnorm = dir.iter().map(|d| d * d).sum::<f64>().sqrt();
    for d in dir.iter_mut() {
        *d /= dnorm;
    }

    // Base EBEs and dη̂/dx at x0.
    let p0 = unpack_params(&x0, &template);
    let eta0: Vec<DVector<f64>> = subjects
        .iter()
        .map(|s| DVector::from_vec(precise_ebe(&model, s, &p0)))
        .collect();
    let jac: Vec<Vec<DVector<f64>>> = subjects
        .iter()
        .enumerate()
        .map(|(i, s)| subject_eta_dx(&model, s, &template, &x0, eta0[i].as_slice()).unwrap())
        .collect();

    eprintln!("  step    warm_err   pred_err   ratio");
    let mut prev_ratio: Option<f64> = None;
    for &s in &[0.20_f64, 0.10, 0.05, 0.025] {
        let x1: Vec<f64> = (0..n).map(|k| x0[k] + s * dir[k]).collect();
        let p1 = unpack_params(&x1, &template);

        let pred = predict_warm_etas(&eta0, &jac, &x0, &x1);

        let mut warm_err = 0.0;
        let mut pred_err = 0.0;
        for (i, subj) in subjects.iter().enumerate() {
            let eta1 = DVector::from_vec(precise_ebe(&model, subj, &p1));
            warm_err += (&eta0[i] - &eta1).norm();
            pred_err += (&pred[i] - &eta1).norm();
        }
        let ratio = pred_err / warm_err.max(1e-300);
        eprintln!("  {s:>5.3}  {warm_err:>9.2e}  {pred_err:>9.2e}  {ratio:>6.3}");

        // (a) the predictor must be a real improvement on warm-start.
        assert!(
                pred_err < 0.5 * warm_err,
                "predictor (err {pred_err:.3e}) should beat warm-start (err {warm_err:.3e}) at step {s}"
            );
        // (b) halving the step should shrink the ratio (second-order error).
        if let Some(pr) = prev_ratio {
            assert!(
                ratio < pr + 1e-9,
                "pred/warm ratio should not grow as the step shrinks ({ratio:.3} vs {pr:.3})"
            );
        }
        prev_ratio = Some(ratio);
    }
}

// --- IOV: analytic θ-gradient over the stacked (η_bsv, κ) with block-Ω ---

const WARFARIN_IOV: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.02
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;

/// WARFARIN_IOV + IIV on residual error (`iiv_on_ruv = ETA_RUV`, the 4th omega →
/// eta index 3). FOCEI is required (non-interaction FOCE + `iiv_on_ruv` is rejected).
const WARFARIN_IOV_RUV: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  omega ETA_RUV ~ 0.05
  kappa KAPPA_CL ~ 0.02
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
  iiv_on_ruv = ETA_RUV
[fit_options]
  method     = focei
  iov_column = OCC
"#;

/// Two-occasion IOV subject (no washout — carryover spans the boundary), with
/// observations synthesised from the model at a reference (η, κ) so residuals
/// are realistic.
fn iov_subject_outer(model: &CompiledModel, theta: &[f64]) -> Subject {
    let obs_times = vec![1.0, 6.0, 12.0, 25.0, 30.0, 36.0];
    let occasions = vec![1u32, 1, 1, 2, 2, 2];
    let n = obs_times.len();
    let mut subject = Subject {
        id: "1".to_string(),
        doses: vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
        ],
        obs_times,
        obs_raw_times: Vec::new(),
        observations: vec![0.0; n],
        obs_cmts: vec![1; n],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0; n],
        occasions,
        obs_l2: Vec::new(),
        dose_occasions: vec![1, 2],
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    // Reference (η_bsv, κ_g0, κ_g1) → realistic ε ≠ 0.
    let preds = crate::pk::predict_iov(
        model,
        &subject,
        theta,
        &[0.12, -0.08, 0.2],
        &[vec![0.05], vec![-0.07]],
    );
    subject.observations = preds.iter().map(|p| p * 0.85).collect();
    subject
}

/// Precisely locate the joint IOV EBE by analytic Newton on the stacked inner
/// objective (exact gradient ½Σαⱼaⱼ + Ω_block⁻¹b and true Hessian from the IOV
/// provider), so the marginal FD is not contaminated by inner-solver
/// reconvergence noise — the IOV analog of [`precise_ebe`]. Returns the stacked
/// `b̂`, plus the `(η̂, κ̂, BSV H-matrix)` form `foce_subject_nll_iov` consumes
/// (H-matrix = the provider's exact `∂f/∂η_bsv`).
fn precise_ebe_iov(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
) -> (Vec<f64>, DVector<f64>, Vec<DVector<f64>>, DMatrix<f64>) {
    let k = crate::stats::likelihood::iov_occasion_groups(subject).len();
    let n_eta = model.n_eta;
    let n_kappa = model.n_kappa;
    let n_st = n_eta + k * n_kappa;
    let warm = find_ebe(model, subject, params, 80, 1e-10, None, None, 0);
    let mut stacked = vec![0.0; n_st];
    for i in 0..n_eta {
        stacked[i] = warm.eta[i];
    }
    for (g, kap) in warm.kappas.iter().enumerate() {
        for ki in 0..n_kappa {
            stacked[n_eta + g * n_kappa + ki] = kap[ki];
        }
    }
    let block = crate::stats::likelihood::build_block_diag_omega(
        &params.omega.matrix,
        &params.omega_iov.as_ref().unwrap().matrix,
        k,
    );
    let omega_inv = block.cholesky().unwrap().inverse();
    let sigma = &params.sigma.values;
    // IIV on residual error (#4b): the inner Newton must minimise the SAME ruv-scaled
    // objective `individual_nll_iov` uses, else the reconverged EBE (and the marginal
    // FD built on it) would be wrong. `ruv_idx` is `None` and `residual_var_scale`
    // returns `1.0` for non-`iiv_on_ruv` models, so this is a no-op there.
    let ruv_idx = model.residual_error_eta;
    let m3 = matches!(model.bloq_method, crate::types::BloqMethod::M3);
    // Custom / time-varying σ magnitude (#576/#486): the Newton must minimise the
    // *scaled* inner objective (like `precise_ebe`), else it converges to the bare
    // EBE and the coupling identity the FOCE-IOV gradient relies on breaks (the
    // reconverged-FD marginal would then disagree on the structural/Ω coordinates).
    // `None` for a bare-sigma model → the non-scaled arm below (bit-identical).
    let mult = model.ruv_obs_mult(subject, &params.theta);
    for _ in 0..50 {
        let ruv_scale = model.residual_var_scale(&stacked);
        let sens = crate::sens::provider::subject_sensitivities_iov(
            model,
            subject,
            &params.theta,
            &stacked,
        )
        .unwrap();
        let mut g = &omega_inv * DVector::from_column_slice(&stacked);
        let mut h = omega_inv.clone();
        for (j, obs) in sens.obs.iter().enumerate() {
            let cmt = subject.obs_cmts[j];
            let f = obs.f;
            let mult_row: Option<&[f64]> =
                mult.as_ref().and_then(|m| m.get(j)).map(|v| v.as_slice());
            let (r, d, d2) = match mult_row {
                Some(m) => (
                    model.error_spec.variance_at_scaled(cmt, f, sigma, &[], m) * ruv_scale,
                    model.error_spec.dvar_df_scaled(cmt, f, sigma, m) * ruv_scale,
                    model.error_spec.d2var_df2_scaled(cmt, f, sigma, m) * ruv_scale,
                ),
                None => (
                    model.error_spec.variance_at(cmt, f, sigma) * ruv_scale,
                    model.error_spec.dvar_df(cmt, f, sigma) * ruv_scale,
                    model.error_spec.d2var_df2(cmt, f, sigma) * ruv_scale,
                ),
            };
            let y = subject.observations[j];
            let eps = y - f;
            // (g1, g2) = (∂L/∂f, ∂²L/∂f²): the censored `−logΦ` scalars for an M3 BLOQ
            // row (#580), else the Gaussian `½α`, `½α'`. The inner Newton must minimise
            // the SAME censored objective `individual_nll_iov` uses, else the
            // reconverged EBE (and the marginal FD built on it) would be wrong.
            let is_cens = m3 && subject.cens.get(j).copied().unwrap_or(0) != 0;
            let (g1, g2) = if is_cens {
                m3_censored_scalars(y, f, r, d, d2, subject.cens.get(j).copied().unwrap_or(0))
            } else {
                let t = err_terms(r, d, d2, eps);
                (0.5 * t.alpha, 0.5 * t.alpha_p)
            };
            let a = &obs.df_deta;
            for kk in 0..n_st {
                g[kk] += g1 * a[kk];
                for ll in 0..n_st {
                    h[(kk, ll)] += g2 * a[kk] * a[ll] + g1 * obs.d2f_deta2[kk * n_st + ll];
                }
            }
            // Residual-eta row/col (`a_{ruv} = 0`): mirrors the `h_inner` residual-eta
            // block in `prepare_stacked`. Gaussian row: data-term gradient `1 − ε²/v`,
            // true Hessian `H[ruv,ruv] += 2ε²/v`, `H[ruv,l] += κ_j a_{jl}`. Censored row
            // under `iiv_on_ruv` (the triple, #591): gradient `h·z`, Hessian
            // `H[ruv,ruv] += C·z`, `H[ruv,l] += C·m·a_{jl}` — the same `(C·z, C·m)`
            // coefficients `m3_censored_outer` feeds `prepare_stacked`. Newton's fixed
            // point is the gradient root, so the reconverged EBE matches the production
            // M3 + IOV + `iiv_on_ruv` inner objective `find_ebe_iov` minimises.
            if let Some(rr) = ruv_idx {
                if is_cens {
                    let (h_im, z_k, _m) = crate::stats::special::m3_censored_kernel(
                        y,
                        f,
                        r,
                        d,
                        subject.cens.get(j).copied().unwrap_or(0),
                    );
                    let (_g1, _g2, cz, cm) = m3_censored_outer(
                        y,
                        f,
                        r,
                        d,
                        d2,
                        subject.cens.get(j).copied().unwrap_or(0),
                    );
                    g[rr] += h_im * z_k;
                    h[(rr, rr)] += cz;
                    for ll in 0..n_st {
                        if ll == rr {
                            continue;
                        }
                        h[(rr, ll)] += cm * a[ll];
                        h[(ll, rr)] += cm * a[ll];
                    }
                } else {
                    g[rr] += 1.0 - eps * eps / r;
                    h[(rr, rr)] += 2.0 * eps * eps / r;
                    let kappa = ruv_kappa(eps, r, d);
                    for ll in 0..n_st {
                        if ll == rr {
                            continue;
                        }
                        h[(rr, ll)] += kappa * a[ll];
                        h[(ll, rr)] += kappa * a[ll];
                    }
                }
            }
        }
        let step = h.cholesky().unwrap().solve(&g);
        for kk in 0..n_st {
            stacked[kk] -= step[kk];
        }
        if step.norm() < 1e-13 {
            break;
        }
    }
    let eta = DVector::from_column_slice(&stacked[..n_eta]);
    let kappas: Vec<DVector<f64>> = (0..k)
        .map(|gi| {
            DVector::from_column_slice(&stacked[n_eta + gi * n_kappa..n_eta + (gi + 1) * n_kappa])
        })
        .collect();
    let sens =
        crate::sens::provider::subject_sensitivities_iov(model, subject, &params.theta, &stacked)
            .unwrap();
    let n_obs = subject.obs_times.len();
    let mut hm = DMatrix::zeros(n_obs, n_eta);
    for j in 0..n_obs {
        for c in 0..n_eta {
            hm[(j, c)] = sens.obs[j].df_deta[c];
        }
    }
    (stacked, eta, kappas, hm)
}

/// IOV marginal at the analytically-reconverged joint EBE for `params` (no
/// inner-solver noise; the BSV H-matrix is the provider's exact Jacobian).
/// `interaction = true` → FOCEI, `false` → FOCE (Sheiner–Beal).
fn marginal_nll_iov_inter(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    interaction: bool,
) -> f64 {
    let (_stacked, eta, kappas, hm) = precise_ebe_iov(model, subject, params);
    crate::stats::likelihood::foce_subject_nll_iov(
        model,
        subject,
        &params.theta,
        &eta,
        &hm,
        &params.omega,
        &params.sigma.values,
        interaction,
        &kappas,
        params.omega_iov.as_ref().expect("IOV model has omega_iov"),
    )
}

fn marginal_nll_iov(model: &CompiledModel, subject: &Subject, params: &ModelParameters) -> f64 {
    marginal_nll_iov_inter(model, subject, params, true)
}

/// The analytic IOV θ-gradient (paper-exact over the stacked η + block-Ω) must
/// match the Richardson-extrapolated reconverged FD of the production IOV FOCEI
/// marginal `foce_subject_nll_iov` — the same objective validated against NONMEM
/// (`tests/warfarin_iov_nonmem.rs`, ferx ≈308.2 vs NONMEM 308.83). This closes
/// the IOV outer-gradient θ block end-to-end against a NONMEM-grounded target.
#[test]
fn iov_theta_gradient_matches_reconverged_fd() {
    let model = parse_model_string(WARFARIN_IOV).expect("parse warfarin IOV");
    let theta = vec![0.22, 11.0, 1.4];
    let mut params = model.default_params.clone();
    params.theta = theta.clone();
    let subject = iov_subject_outer(&model, &theta);

    // Joint EBE [η_bsv (3), κ_g0 (1), κ_g1 (1)], analytically reconverged.
    let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);

    let analytic = subject_theta_gradient_iov(&model, &subject, &params, &stacked)
        .expect("IOV θ-gradient supported");

    let fd_at = |m: usize, h: f64| -> f64 {
        let mut pp = params.clone();
        pp.theta[m] += h;
        let mut pm = params.clone();
        pm.theta[m] -= h;
        (marginal_nll_iov(&model, &subject, &pp) - marginal_nll_iov(&model, &subject, &pm))
            / (2.0 * h)
    };
    for m in 0..theta.len() {
        let h = 1e-4 * (1.0 + theta[m].abs());
        let f1 = fd_at(m, h);
        let f2 = fd_at(m, h / 2.0);
        let fd = (4.0 * f2 - f1) / 3.0; // Richardson
        eprintln!(
            "iov theta[{m}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
            analytic[m],
            fd,
            (analytic[m] - fd).abs() / fd.abs().max(1e-12)
        );
        approx::assert_relative_eq!(analytic[m], fd, max_relative = 3e-3, epsilon = 1e-5);
    }
}

/// Two-occasion IOV + `iiv_on_ruv` subject (n_eta = 4 incl. ETA_RUV). η_ruv (the 4th
/// bsv eta) affects only the residual variance, not the predictions, so it is supplied
/// to `predict_iov` purely to keep the eta vector the right length.
fn iov_ruv_subject(model: &CompiledModel, theta: &[f64]) -> Subject {
    let obs_times = vec![1.0, 6.0, 12.0, 25.0, 30.0, 36.0];
    let occasions = vec![1u32, 1, 1, 2, 2, 2];
    let n = obs_times.len();
    let mut subject = Subject {
        id: "1".to_string(),
        doses: vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
        ],
        obs_times,
        obs_raw_times: Vec::new(),
        observations: vec![0.0; n],
        obs_cmts: vec![1; n],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0; n],
        occasions,
        obs_l2: Vec::new(),
        dose_occasions: vec![1, 2],
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    let preds = crate::pk::predict_iov(
        model,
        &subject,
        theta,
        &[0.12, -0.08, 0.2, 0.10],
        &[vec![0.05], vec![-0.07]],
    );
    subject.observations = preds.iter().map(|p| p * 0.85).collect();
    subject
}

/// Closed-form IOV + `iiv_on_ruv` (#4b): the analytic IOV θ-gradient (which now threads
/// `ruv = residual_error_eta` into `prepare_stacked`, so the residual-eta `c̃` column
/// rides the stacked `[η_bsv, κ]` assembly) must match the Richardson reconverged FD of
/// the FOCEI marginal — the marginal whose EBE `precise_ebe_iov` reconverges against the
/// same `exp(2·η_ruv)`-scaled objective. Proves the outer gate flip ships a correct
/// gradient.
#[test]
fn iov_iiv_on_ruv_theta_gradient_matches_reconverged_fd() {
    let model = parse_model_string(WARFARIN_IOV_RUV).expect("parse warfarin IOV + iiv_on_ruv");
    assert_eq!(model.residual_error_eta, Some(3));
    assert!(crate::sens::provider::iov_analytical_supported(&model));
    let theta = vec![0.22, 11.0, 1.4];
    let mut params = model.default_params.clone();
    params.theta = theta.clone();
    let subject = iov_ruv_subject(&model, &theta);

    // Joint EBE [η_bsv (4, incl. η_ruv), κ_g0 (1), κ_g1 (1)], analytically reconverged
    // against the ruv-scaled inner objective.
    let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);

    let analytic = subject_theta_gradient_iov(&model, &subject, &params, &stacked)
        .expect("IOV + iiv_on_ruv θ-gradient supported");

    let fd_at = |m: usize, h: f64| -> f64 {
        let mut pp = params.clone();
        pp.theta[m] += h;
        let mut pm = params.clone();
        pm.theta[m] -= h;
        (marginal_nll_iov(&model, &subject, &pp) - marginal_nll_iov(&model, &subject, &pm))
            / (2.0 * h)
    };
    for m in 0..theta.len() {
        let h = 1e-4 * (1.0 + theta[m].abs());
        let f1 = fd_at(m, h);
        let f2 = fd_at(m, h / 2.0);
        let fd = (4.0 * f2 - f1) / 3.0; // Richardson
        eprintln!(
            "iov+ruv theta[{m}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
            analytic[m],
            fd,
            (analytic[m] - fd).abs() / fd.abs().max(1e-12)
        );
        approx::assert_relative_eq!(analytic[m], fd, max_relative = 3e-3, epsilon = 1e-5);
    }
}

/// The full analytic IOV **packed** gradient (`[θ, Ω_bsv, σ, Ω_iov]`, optimizer
/// space) must match the Richardson reconverged FD of the production IOV FOCEI
/// marginal over every packed coordinate — closing the Ω (incl. the shared
/// κ-variance) and σ blocks against the NONMEM-grounded objective.
#[test]
fn iov_packed_gradient_matches_reconverged_fd() {
    let model = parse_model_string(WARFARIN_IOV).expect("parse warfarin IOV");
    let theta = vec![0.22, 11.0, 1.4];
    let mut params = model.default_params.clone();
    params.theta = theta.clone();
    let subject = iov_subject_outer(&model, &theta);
    let template = params.clone();
    let x = crate::estimation::parameterization::pack_params(&params);

    let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);
    let analytic = subject_packed_gradient_iov(&model, &subject, &template, &x, &stacked)
        .expect("IOV packed gradient supported");

    let f = |xx: &[f64]| -> f64 {
        let p = unpack_params(xx, &template);
        marginal_nll_iov(&model, &subject, &p)
    };
    for i in 0..x.len() {
        let h = 1e-4 * (1.0 + x[i].abs());
        let fd_at = |hh: f64| -> f64 {
            let mut xp = x.clone();
            xp[i] += hh;
            let mut xm = x.clone();
            xm[i] -= hh;
            (f(&xp) - f(&xm)) / (2.0 * hh)
        };
        let f1 = fd_at(h);
        let f2 = fd_at(h / 2.0);
        let fd = (4.0 * f2 - f1) / 3.0; // Richardson
        eprintln!(
            "iov x[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
            analytic[i],
            fd,
            (analytic[i] - fd).abs() / fd.abs().max(1e-9)
        );
        approx::assert_relative_eq!(analytic[i], fd, max_relative = 2e-3, epsilon = 2e-5);
    }
}

/// IOV + `iiv_on_ruv` through the **production** packed-gradient path
/// (`subject_packed_gradient_iov`, not the `subject_theta_gradient_iov` helper):
/// the full `[θ, Ω_bsv, σ, Ω_iov]` analytic gradient must match Richardson
/// reconverged FD of the scaled FOCEI marginal. Regression for the review
/// finding that the residual-eta threading reached only the test helper, leaving
/// production on the unscaled variance with the `η_ruv` `c̃` column dropped.
#[test]
fn iov_iiv_on_ruv_packed_gradient_matches_reconverged_fd() {
    let model = parse_model_string(WARFARIN_IOV_RUV).expect("parse warfarin IOV + iiv_on_ruv");
    assert_eq!(model.residual_error_eta, Some(3));
    let theta = vec![0.22, 11.0, 1.4];
    let mut params = model.default_params.clone();
    params.theta = theta.clone();
    let subject = iov_ruv_subject(&model, &theta);
    let template = params.clone();
    let x = crate::estimation::parameterization::pack_params(&params);

    let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);
    let analytic = subject_packed_gradient_iov(&model, &subject, &template, &x, &stacked)
        .expect("IOV + iiv_on_ruv packed gradient supported");

    let f = |xx: &[f64]| -> f64 {
        let p = unpack_params(xx, &template);
        marginal_nll_iov(&model, &subject, &p)
    };
    for i in 0..x.len() {
        let h = 1e-4 * (1.0 + x[i].abs());
        let fd_at = |hh: f64| -> f64 {
            let mut xp = x.clone();
            xp[i] += hh;
            let mut xm = x.clone();
            xm[i] -= hh;
            (f(&xp) - f(&xm)) / (2.0 * hh)
        };
        let f1 = fd_at(h);
        let f2 = fd_at(h / 2.0);
        let fd = (4.0 * f2 - f1) / 3.0; // Richardson
        eprintln!(
            "iov+ruv packed x[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
            analytic[i],
            fd,
            (analytic[i] - fd).abs() / fd.abs().max(1e-9)
        );
        approx::assert_relative_eq!(analytic[i], fd, max_relative = 3e-3, epsilon = 2e-5);
    }
}

/// [`WARFARIN_IOV`] with a `log_additive` (LTBS) error model — plain closed-form
/// LTBS × IOV. #665 served LTBS on the non-IOV outer gradient (post-walk `ln(f)`
/// jet); this fixture drives its IOV twin.
const WARFARIN_IOV_LTBS: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.02
  sigma ADD_ERR ~ 0.05
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ log_additive(ADD_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
"#;

/// As [`WARFARIN_IOV_LTBS`] but also carrying an η-dependent `ExpressionScale`
/// `obs_scale = V` — pins the composition `ln(f / s)` under IOV (the in-walk scale
/// quotient followed by the post-walk `ln` jet must reproduce production's
/// scale-then-log order in `predict_iov`).
const WARFARIN_IOV_LTBS_EXPRSCALE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.02
  sigma ADD_ERR ~ 0.05
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[scaling]
  obs_scale = V
[error_model]
  DV ~ log_additive(ADD_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
"#;

/// As [`WARFARIN_IOV_LTBS`] plus IIV on residual error (`iiv_on_ruv = ETA_RUV`, the 4th
/// eta) — the LTBS × `iiv_on_ruv` × IOV composition the relaxed gate admits (#677 review #2).
const WARFARIN_IOV_LTBS_RUV: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  omega ETA_RUV ~ 0.05
  kappa KAPPA_CL ~ 0.02
  sigma ADD_ERR ~ 0.05
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ log_additive(ADD_ERR)
  iiv_on_ruv = ETA_RUV
[fit_options]
  method     = focei
  iov_column = OCC
"#;

/// Pure Richardson-FD parity check for an IOV **outer** packed gradient: the analytic
/// gradient (`subject_packed_gradient_iov` for FOCEI / `subject_packed_gradient_foce_iov`
/// for FOCE) must match reconverged central FD of the corresponding IOV marginal across
/// every packed θ/Ω_bsv/σ/Ω_iov coordinate. `subject` and `interaction` are caller-supplied
/// so the same stencil serves plain / magnitude / LTBS / M3 / `iiv_on_ruv` fixtures on both
/// the FOCEI and FOCE paths (dedup — #677 review #4).
fn check_iov_outer_packed_matches_fd(
    model: &CompiledModel,
    theta: &[f64],
    subject: &Subject,
    interaction: bool,
) {
    use crate::estimation::parameterization::pack_params;
    let mut params = model.default_params.clone();
    params.theta = theta.to_vec();
    let template = params.clone();
    let x = pack_params(&params);
    let (stacked, _e, _k, _h) = precise_ebe_iov(model, subject, &params);
    let analytic = if interaction {
        subject_packed_gradient_iov(model, subject, &template, &x, &stacked)
    } else {
        subject_packed_gradient_foce_iov(model, subject, &template, &x, &stacked)
    }
    .expect("IOV outer packed gradient supported");
    assert!(
        analytic.iter().all(|v| v.is_finite()),
        "packed gradient must be finite"
    );
    let f = |xx: &[f64]| -> f64 {
        let p = unpack_params(xx, &template);
        marginal_nll_iov_inter(model, subject, &p, interaction)
    };
    for i in 0..x.len() {
        let h = 1e-4 * (1.0 + x[i].abs());
        let fd_at = |hh: f64| -> f64 {
            let mut xp = x.clone();
            xp[i] += hh;
            let mut xm = x.clone();
            xm[i] -= hh;
            (f(&xp) - f(&xm)) / (2.0 * hh)
        };
        let f1 = fd_at(h);
        let f2 = fd_at(h / 2.0);
        let fd = (4.0 * f2 - f1) / 3.0; // Richardson
        eprintln!(
            "iov outer x[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
            analytic[i],
            fd,
            (analytic[i] - fd).abs() / fd.abs().max(1e-9)
        );
        approx::assert_relative_eq!(analytic[i], fd, max_relative = 3e-3, epsilon = 2e-5);
    }
}

/// LTBS × IOV routing assertions (both loops analytic since #486) + FD parity via
/// [`check_iov_outer_packed_matches_fd`] on the FOCEI path.
fn check_ltbs_iov_outer_matches_fd(model: &CompiledModel, theta: &[f64]) {
    assert!(model.log_transform, "fixture must be LTBS");
    assert!(model.n_kappa > 0, "fixture must carry IOV");
    assert!(
        crate::sens::provider::iov_analytical_supported(model),
        "LTBS × IOV must route to the analytic closed-form OUTER path (#486)"
    );
    assert!(crate::sens::provider::analytic_outer_gradient_available(
        model
    ));
    assert!(
        !crate::estimation::inner_optimizer::analytic_inner_common_bail(model),
        "LTBS × IOV inner EBE gradient is analytic too (#486) — inner and outer move together"
    );
    let subject = iov_subject_outer(model, theta);
    check_iov_outer_packed_matches_fd(model, theta, &subject, true);
}

#[test]
fn ltbs_iov_outer_packed_matches_fd() {
    let model = parse_model_string(WARFARIN_IOV_LTBS).expect("parse LTBS IOV");
    check_ltbs_iov_outer_matches_fd(&model, &[0.22, 11.0, 1.4]);
}

#[test]
fn ltbs_expr_scale_iov_outer_packed_matches_fd() {
    let model = parse_model_string(WARFARIN_IOV_LTBS_EXPRSCALE).expect("parse LTBS+scale IOV");
    assert!(
        matches!(
            model.scaling,
            crate::types::ScalingSpec::ExpressionScale { .. }
        ),
        "fixture must carry an expression obs_scale"
    );
    check_ltbs_iov_outer_matches_fd(&model, &[0.22, 11.0, 1.4]);
}

/// LTBS × **M3-BLOQ** × IOV (#677 review #1). Relaxing the `log_transform` gate also
/// admits M3-censored rows (the gate still admits M3), so the censored `−logΦ(z)` kernel
/// in `prepare_stacked` now consumes the ln(f) jet. Pin that the newly-analytic outer
/// gradient matches reconverged FD of the FOCEI-IOV censored marginal.
#[test]
fn ltbs_m3_iov_outer_packed_matches_fd() {
    let mut model = parse_model_string(WARFARIN_IOV_LTBS).expect("parse LTBS IOV");
    model.bloq_method = crate::types::BloqMethod::M3;
    assert!(
        crate::sens::provider::iov_analytical_supported(&model),
        "LTBS × M3 × IOV is admitted by the relaxed gate"
    );
    let theta = vec![0.22, 11.0, 1.4];
    let subject = iov_m3_subject(&model, &theta);
    check_iov_outer_packed_matches_fd(&model, &theta, &subject, true);
}

/// LTBS × `iiv_on_ruv` × IOV (#677 review #2). The residual-eta (η_ruv) variance-scaling
/// column rides the stacked `[η_bsv, κ]` assembly alongside the newly applied ln(f) jet;
/// pin the composition against FD.
#[test]
fn ltbs_iiv_on_ruv_iov_outer_packed_matches_fd() {
    let model = parse_model_string(WARFARIN_IOV_LTBS_RUV).expect("parse LTBS IOV + iiv_on_ruv");
    assert_eq!(model.residual_error_eta, Some(3), "ETA_RUV is the 4th eta");
    assert!(crate::sens::provider::iov_analytical_supported(&model));
    let theta = vec![0.22, 11.0, 1.4];
    let subject = iov_ruv_subject(&model, &theta);
    check_iov_outer_packed_matches_fd(&model, &theta, &subject, true);
}

/// LTBS × IOV on the **FOCE** (non-interaction) path (#677 review #3). Opening the gate
/// for `log_transform` also makes `subject_packed_gradient_foce_iov` reachable for LTBS
/// (it consumes the same ln-jetted `subject_sensitivities_iov`). Every other LTBS×IOV test
/// uses FOCEI; pin the FOCE marginal-moment path too.
#[test]
fn ltbs_iov_outer_foce_packed_matches_fd() {
    let model = parse_model_string(WARFARIN_IOV_LTBS).expect("parse LTBS IOV");
    let theta = vec![0.22, 11.0, 1.4];
    let subject = iov_subject_outer(&model, &theta);
    check_iov_outer_packed_matches_fd(&model, &theta, &subject, false);
}

/// Two-occasion IOV subject with M3-censored rows (#580): the same geometry as
/// [`iov_subject_outer`], but occasion 2's two tail observations are flagged
/// `CENS = 1` (left-censored at their synthesized value ≈ 0.85·f, so the
/// prediction sits just above the limit and the inverse Mills ratio is well-scaled).
fn iov_m3_subject(model: &CompiledModel, theta: &[f64]) -> Subject {
    let mut subject = iov_subject_outer(model, theta);
    let n = subject.observations.len();
    subject.cens[n - 2] = 1;
    subject.cens[n - 1] = 1;
    subject
}

/// As [`iov_m3_subject`] but the occasion-2 tail is **right**-censored
/// (`CENS = -1`, above ULOQ) — exercises the upper-tail (`σ = -1`) branch of the
/// signed `m3_censored_kernel` / FOCE `Cens` terms.
fn iov_m3_subject_right(model: &CompiledModel, theta: &[f64]) -> Subject {
    let mut subject = iov_subject_outer(model, theta);
    let n = subject.observations.len();
    subject.cens[n - 2] = -1;
    subject.cens[n - 1] = -1;
    subject
}

/// M3 BLOQ + IOV (#580): the analytic IOV FOCEI θ-gradient (censored rows carry
/// `p = β = 0` so they leave `H̃`/`log|H̃|` exactly as `foce_subject_nll_iov`
/// builds it, and re-enter via the `−logΦ` data term + true inner Hessian over the
/// stacked `[η_bsv, κ]` layout) must match the Richardson reconverged FD of the
/// FOCEI IOV marginal — the same objective `precise_ebe_iov` now reconverges
/// against (its Newton loop uses the censored `m3_censored_scalars` on flagged rows).
/// Proves the gate flip ships a correct censored θ-gradient.
#[test]
fn iov_m3_theta_gradient_matches_reconverged_fd() {
    let mut model = parse_model_string(WARFARIN_IOV).expect("parse warfarin IOV");
    model.bloq_method = crate::types::BloqMethod::M3;
    assert!(crate::sens::provider::iov_analytical_supported(&model));
    let theta = vec![0.22, 11.0, 1.4];
    let mut params = model.default_params.clone();
    params.theta = theta.clone();
    let subject = iov_m3_subject(&model, &theta);
    assert!(
        subject.cens.iter().any(|&c| c != 0),
        "subject must be censored"
    );

    // Joint EBE [η_bsv (3), κ_g0 (1), κ_g1 (1)] reconverged against the M3-aware
    // inner objective.
    let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);

    let analytic = subject_theta_gradient_iov(&model, &subject, &params, &stacked)
        .expect("IOV + M3 θ-gradient supported");

    let fd_at = |m: usize, h: f64| -> f64 {
        let mut pp = params.clone();
        pp.theta[m] += h;
        let mut pm = params.clone();
        pm.theta[m] -= h;
        (marginal_nll_iov(&model, &subject, &pp) - marginal_nll_iov(&model, &subject, &pm))
            / (2.0 * h)
    };
    for m in 0..theta.len() {
        let h = 1e-4 * (1.0 + theta[m].abs());
        let f1 = fd_at(m, h);
        let f2 = fd_at(m, h / 2.0);
        let fd = (4.0 * f2 - f1) / 3.0; // Richardson
        eprintln!(
            "iov+m3 theta[{m}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
            analytic[m],
            fd,
            (analytic[m] - fd).abs() / fd.abs().max(1e-12)
        );
        approx::assert_relative_eq!(analytic[m], fd, max_relative = 3e-3, epsilon = 1e-5);
    }
}

/// M3 BLOQ + IOV (#580) through the **production** FOCEI packed-gradient path
/// (`subject_packed_gradient_iov`): the full `[θ, Ω_bsv, σ, Ω_iov]` analytic
/// gradient must match Richardson reconverged FD of the FOCEI IOV marginal over
/// every packed coordinate — exercising the censored σ-block (`censored_sigma_m_terms`)
/// and the Ω blocks (incl. the shared κ-variance) with censored rows present.
#[test]
fn iov_m3_packed_gradient_matches_reconverged_fd() {
    let mut model = parse_model_string(WARFARIN_IOV).expect("parse warfarin IOV");
    model.bloq_method = crate::types::BloqMethod::M3;
    let theta = vec![0.22, 11.0, 1.4];
    let mut params = model.default_params.clone();
    params.theta = theta.clone();
    let subject = iov_m3_subject(&model, &theta);
    let template = params.clone();
    let x = crate::estimation::parameterization::pack_params(&params);

    let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);
    let analytic = subject_packed_gradient_iov(&model, &subject, &template, &x, &stacked)
        .expect("IOV + M3 packed gradient supported");

    let f = |xx: &[f64]| -> f64 {
        let p = unpack_params(xx, &template);
        marginal_nll_iov(&model, &subject, &p)
    };
    for i in 0..x.len() {
        let h = 1e-4 * (1.0 + x[i].abs());
        let fd_at = |hh: f64| -> f64 {
            let mut xp = x.clone();
            xp[i] += hh;
            let mut xm = x.clone();
            xm[i] -= hh;
            (f(&xp) - f(&xm)) / (2.0 * hh)
        };
        let f1 = fd_at(h);
        let f2 = fd_at(h / 2.0);
        let fd = (4.0 * f2 - f1) / 3.0; // Richardson
        eprintln!(
            "iov+m3 packed x[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
            analytic[i],
            fd,
            (analytic[i] - fd).abs() / fd.abs().max(1e-9)
        );
        approx::assert_relative_eq!(analytic[i], fd, max_relative = 3e-3, epsilon = 2e-5);
    }
}

/// 1-cpt oral **user-ODE** IOV model (κ on CL), the ODE counterpart of
/// [`WARFARIN_IOV`]. Drives the ODE IOV M3 outer test below.
const ONECPT_ODE_IOV: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot / V - (CL/V) * central
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method      = focei
  iov_column  = OCC
  ode_reltol  = 1e-10
  ode_abstol  = 1e-12
"#;

/// **ODE** M3 BLOQ + IOV (#486): the ODE counterpart of
/// [`iov_m3_packed_gradient_matches_reconverged_fd`]. The full `[θ, Ω_bsv, σ, Ω_iov]`
/// analytic packed gradient — assembled from the **event-driven ODE sensitivity walk**
/// (`subject_sensitivities_iov` → `ode_subject_sensitivities_iov`) with censored rows
/// entering `prepare_stacked`'s M3 branch — must match Richardson reconverged FD of the
/// FOCEI IOV marginal. Censoring is provider-agnostic (keyed on `subject.cens[j]`), so
/// the only change versus the closed-form path is the dropped gate clause. Both tails.
#[test]
fn iov_m3_ode_packed_gradient_matches_reconverged_fd() {
    use crate::estimation::parameterization::pack_params;

    let mut model = parse_model_string(ONECPT_ODE_IOV).expect("parse ODE IOV");
    model.bloq_method = crate::types::BloqMethod::M3;
    assert!(model.is_ode_based(), "must be on the ODE path");
    assert!(
        crate::sens::ode_provider::ode_iov_supported(&model),
        "ODE IOV + M3 must be analytic (#486)"
    );
    let theta = vec![0.22, 11.0, 1.4];
    let mut params = model.default_params.clone();
    params.theta = theta.clone();

    for right in [false, true] {
        let subject = if right {
            iov_m3_subject_right(&model, &theta)
        } else {
            iov_m3_subject(&model, &theta)
        };
        assert!(
            subject.cens.iter().any(|&c| c != 0),
            "subject must be censored"
        );
        let template = params.clone();
        let x = pack_params(&params);

        let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);
        let analytic = subject_packed_gradient_iov(&model, &subject, &template, &x, &stacked)
            .expect("ODE IOV + M3 packed gradient supported");

        let f = |xx: &[f64]| -> f64 {
            let p = unpack_params(xx, &template);
            marginal_nll_iov(&model, &subject, &p)
        };
        for i in 0..x.len() {
            let h = 1e-4 * (1.0 + x[i].abs());
            let fd_at = |hh: f64| -> f64 {
                let mut xp = x.clone();
                xp[i] += hh;
                let mut xm = x.clone();
                xm[i] -= hh;
                (f(&xp) - f(&xm)) / (2.0 * hh)
            };
            let f1 = fd_at(h);
            let f2 = fd_at(h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0; // Richardson
            eprintln!(
                "iov+m3 ode (right={right}) packed x[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic[i],
                fd,
                (analytic[i] - fd).abs() / fd.abs().max(1e-9)
            );
            approx::assert_relative_eq!(analytic[i], fd, max_relative = 3e-3, epsilon = 2e-5);
        }
    }
}

/// **ODE** FOCE (non-interaction) M3 BLOQ + IOV (#486): the ODE counterpart of
/// [`iov_m3_foce_packed_gradient_matches_reconverged_fd`]. Guards the §6 gotcha — the
/// ODE FOCE-IOV objective must route censored rows the same way as the closed-form
/// path (no silent promotion to interaction; censored rows re-enter as `−logΦ` at the
/// population η=0, κ=0 variance). The FOCE packed gradient assembled from the ODE walk
/// must match Richardson reconverged FD of `marginal_nll_iov_inter(.., false)`.
#[test]
fn iov_m3_foce_ode_packed_gradient_matches_reconverged_fd() {
    use crate::estimation::parameterization::pack_params;

    let mut model = parse_model_string(ONECPT_ODE_IOV).expect("parse ODE IOV");
    model.bloq_method = crate::types::BloqMethod::M3;
    assert!(model.is_ode_based(), "must be on the ODE path");
    assert!(crate::sens::ode_provider::ode_iov_supported(&model));
    let theta = vec![0.22, 11.0, 1.4];
    let mut params = model.default_params.clone();
    params.theta = theta.clone();
    let subject = iov_m3_subject(&model, &theta);
    assert!(
        subject.cens.iter().any(|&c| c != 0),
        "subject must be censored"
    );
    let template = params.clone();
    let x = pack_params(&params);

    let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);
    let analytic = subject_packed_gradient_foce_iov(&model, &subject, &template, &x, &stacked)
        .expect("FOCE-ODE-IOV-M3 packed gradient supported");

    let f = |xx: &[f64]| -> f64 {
        let p = unpack_params(xx, &template);
        marginal_nll_iov_inter(&model, &subject, &p, false)
    };
    for i in 0..x.len() {
        let h = 1e-4 * (1.0 + x[i].abs());
        let fd_at = |hh: f64| -> f64 {
            let mut xp = x.clone();
            xp[i] += hh;
            let mut xm = x.clone();
            xm[i] -= hh;
            (f(&xp) - f(&xm)) / (2.0 * hh)
        };
        let f1 = fd_at(h);
        let f2 = fd_at(h / 2.0);
        let fd = (4.0 * f2 - f1) / 3.0; // Richardson
        eprintln!(
            "iov+m3 foce-ode packed x[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
            analytic[i],
            fd,
            (analytic[i] - fd).abs() / fd.abs().max(1e-9)
        );
        approx::assert_relative_eq!(analytic[i], fd, max_relative = 3e-3, epsilon = 2e-5);
    }
}

/// 1-cpt oral **user-ODE** IOV + `iiv_on_ruv` model (κ on CL, `ETA_RUV` scaling the
/// residual variance, absent from CL/V/KA), the ODE counterpart of [`WARFARIN_IOV_RUV`].
/// Drives the ODE `iiv_on_ruv` / triple outer tests below.
const ONECPT_ODE_IOV_RUV: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  omega ETA_RUV ~ 0.05
  kappa KAPPA_CL ~ 0.02
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot / V - (CL/V) * central
[error_model]
  DV ~ proportional(PROP_ERR)
  iiv_on_ruv = ETA_RUV
[fit_options]
  method      = focei
  iov_column  = OCC
  ode_reltol  = 1e-10
  ode_abstol  = 1e-12
"#;

/// **ODE** IOV + `iiv_on_ruv` (no M3, #486): the ODE counterpart of
/// [`iov_iiv_on_ruv_packed_gradient_matches_reconverged_fd`]. The full
/// `[θ, Ω_bsv, σ, Ω_iov]` packed gradient from the ODE walk must match Richardson
/// reconverged FD of the `exp(2·η_ruv)`-scaled FOCEI IOV marginal. The ODE walk emits a
/// zero `∂f/∂η_ruv` column; the shared assembly applies the variance scaling and the
/// residual-eta `c̃` column (keyed on `residual_error_eta`), provider-agnostic.
#[test]
fn iov_iiv_on_ruv_ode_packed_gradient_matches_reconverged_fd() {
    use crate::estimation::parameterization::pack_params;

    let model = parse_model_string(ONECPT_ODE_IOV_RUV).expect("parse ODE IOV + iiv_on_ruv");
    assert_eq!(model.residual_error_eta, Some(3));
    assert!(model.is_ode_based(), "must be on the ODE path");
    assert!(crate::sens::ode_provider::ode_iov_supported(&model));
    let theta = vec![0.22, 11.0, 1.4];
    let mut params = model.default_params.clone();
    params.theta = theta.clone();
    let subject = iov_ruv_subject(&model, &theta);
    let template = params.clone();
    let x = pack_params(&params);

    let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);
    let analytic = subject_packed_gradient_iov(&model, &subject, &template, &x, &stacked)
        .expect("ODE IOV + iiv_on_ruv packed gradient supported");

    let f = |xx: &[f64]| -> f64 {
        let p = unpack_params(xx, &template);
        marginal_nll_iov(&model, &subject, &p)
    };
    for i in 0..x.len() {
        let h = 1e-4 * (1.0 + x[i].abs());
        let fd_at = |hh: f64| -> f64 {
            let mut xp = x.clone();
            xp[i] += hh;
            let mut xm = x.clone();
            xm[i] -= hh;
            (f(&xp) - f(&xm)) / (2.0 * hh)
        };
        let f1 = fd_at(h);
        let f2 = fd_at(h / 2.0);
        let fd = (4.0 * f2 - f1) / 3.0; // Richardson
        eprintln!(
            "iov+ruv ode packed x[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
            analytic[i],
            fd,
            (analytic[i] - fd).abs() / fd.abs().max(1e-9)
        );
        approx::assert_relative_eq!(analytic[i], fd, max_relative = 3e-3, epsilon = 2e-5);
    }
}

/// **ODE** triple M3 + IOV + `iiv_on_ruv` (#486): the ODE counterpart of
/// [`iov_m3_iiv_on_ruv_packed_gradient_matches_reconverged_fd`]. Censored rows co-occur
/// with the `exp(2·η_ruv)` variance scaling: `prepare_stacked` returns the censored
/// residual-eta cross coefficients `(C·z, C·m)` into the true inner Hessian and the
/// `h·z` column into the inner gradient — all provider-agnostic over the ODE walk's
/// `ObsSens`. The packed gradient must match Richardson reconverged FD of the FOCEI IOV
/// marginal. Both tails.
#[test]
fn iov_m3_iiv_on_ruv_ode_packed_gradient_matches_reconverged_fd() {
    use crate::estimation::parameterization::pack_params;

    let mut model = parse_model_string(ONECPT_ODE_IOV_RUV).expect("parse ODE IOV + iiv_on_ruv");
    model.bloq_method = crate::types::BloqMethod::M3;
    assert_eq!(model.residual_error_eta, Some(3));
    assert!(model.is_ode_based(), "must be on the ODE path");
    assert!(crate::sens::ode_provider::ode_iov_supported(&model));
    let theta = vec![0.22, 11.0, 1.4];
    let mut params = model.default_params.clone();
    params.theta = theta.clone();

    for right in [false, true] {
        let mut subject = iov_m3_ruv_subject(&model, &theta);
        if right {
            let n = subject.observations.len();
            subject.cens[n - 2] = -1;
            subject.cens[n - 1] = -1;
        }
        assert!(
            subject.cens.iter().any(|&c| c != 0),
            "subject must be censored"
        );
        let template = params.clone();
        let x = pack_params(&params);

        let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);
        let analytic = subject_packed_gradient_iov(&model, &subject, &template, &x, &stacked)
            .expect("ODE IOV + M3 + iiv_on_ruv packed gradient supported");

        let f = |xx: &[f64]| -> f64 {
            let p = unpack_params(xx, &template);
            marginal_nll_iov(&model, &subject, &p)
        };
        for i in 0..x.len() {
            let h = 1e-4 * (1.0 + x[i].abs());
            let fd_at = |hh: f64| -> f64 {
                let mut xp = x.clone();
                xp[i] += hh;
                let mut xm = x.clone();
                xm[i] -= hh;
                (f(&xp) - f(&xm)) / (2.0 * hh)
            };
            let f1 = fd_at(h);
            let f2 = fd_at(h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0; // Richardson
            eprintln!(
                    "iov+m3+ruv ode (right={right}) packed x[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                    analytic[i],
                    fd,
                    (analytic[i] - fd).abs() / fd.abs().max(1e-9)
                );
            approx::assert_relative_eq!(analytic[i], fd, max_relative = 3e-3, epsilon = 2e-5);
        }
    }
}

/// 1-cpt oral **user-ODE** IOV model carrying an η-dependent `ExpressionScale`
/// `obs_scale` divisor (`obs_scale = 1000 / V`, `V = TVV·exp(ETA_V)`) — the
/// [`ONECPT_ODE_IOV`] geometry plus the #575 post-walk quotient scale. The scale
/// rides the `(θ, stacked-η)` jet *before* the provider-agnostic M3 censoring
/// coefficient is applied, so it composes with BLOQ rows at a different layer.
/// Drives the ODE M3 × `ExpressionScale` × IOV cross-check below (#623 review).
const ONECPT_ODE_IOV_EXPRSCALE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot / V - (CL/V) * central
[error_model]
  DV ~ proportional(PROP_ERR)
[scaling]
  obs_scale = 1000 / V
[fit_options]
  method      = focei
  iov_column  = OCC
  ode_reltol  = 1e-10
  ode_abstol  = 1e-12
"#;

/// **ODE M3 + IOV + η-dependent `ExpressionScale` `obs_scale`** (#623 review of #486):
/// the gate flip in `ode_iov_supported` admits censored rows alongside the #575 scale
/// quotient, a combination the closed-form mirror never reaches (its gate rejects every
/// non-`None` scaling) and that the #486 tests did not exercise. The two features
/// compose at different layers — the scale is a post-walk quotient on the `(θ, stacked-η)`
/// jet (incl. its second-order derivatives), and the M3 `−logΦ` coefficient is applied
/// over that already-scaled jet keyed on `subject.cens[j]`. If the second-order
/// composition of the quotient were inconsistent for censored rows, the marginal
/// `log|H̃|` term would be wrong. The full `[θ, Ω_bsv, σ, Ω_iov]` analytic packed
/// gradient must match Richardson reconverged FD of the FOCEI IOV marginal over every
/// packed coordinate, on both censoring tails — proving the composition is consistent.
#[test]
fn iov_m3_ode_expression_scale_packed_gradient_matches_reconverged_fd() {
    use crate::estimation::parameterization::pack_params;

    let mut model = parse_model_string(ONECPT_ODE_IOV_EXPRSCALE)
        .expect("parse ODE IOV + ExpressionScale obs_scale");
    model.bloq_method = crate::types::BloqMethod::M3;
    assert!(model.is_ode_based(), "must be on the ODE path");
    assert!(
        matches!(
            model.scaling,
            crate::types::ScalingSpec::ExpressionScale { .. }
        ),
        "model must carry an ExpressionScale obs_scale"
    );
    assert!(
        crate::sens::ode_provider::ode_iov_supported(&model),
        "ODE IOV + M3 + ExpressionScale obs_scale must be analytic (#486/#575)"
    );
    let theta = vec![0.22, 11.0, 1.4];
    let mut params = model.default_params.clone();
    params.theta = theta.clone();

    for right in [false, true] {
        let subject = if right {
            iov_m3_subject_right(&model, &theta)
        } else {
            iov_m3_subject(&model, &theta)
        };
        assert!(
            subject.cens.iter().any(|&c| c != 0),
            "subject must be censored"
        );
        let template = params.clone();
        let x = pack_params(&params);

        let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);
        let analytic = subject_packed_gradient_iov(&model, &subject, &template, &x, &stacked)
            .expect("ODE IOV + M3 + ExpressionScale packed gradient supported");

        let f = |xx: &[f64]| -> f64 {
            let p = unpack_params(xx, &template);
            marginal_nll_iov(&model, &subject, &p)
        };
        for i in 0..x.len() {
            let h = 1e-4 * (1.0 + x[i].abs());
            let fd_at = |hh: f64| -> f64 {
                let mut xp = x.clone();
                xp[i] += hh;
                let mut xm = x.clone();
                xm[i] -= hh;
                (f(&xp) - f(&xm)) / (2.0 * hh)
            };
            let f1 = fd_at(h);
            let f2 = fd_at(h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0; // Richardson
            eprintln!(
                    "iov+m3 ode+exprscale (right={right}) packed x[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                    analytic[i],
                    fd,
                    (analytic[i] - fd).abs() / fd.abs().max(1e-9)
                );
            approx::assert_relative_eq!(analytic[i], fd, max_relative = 3e-3, epsilon = 2e-5);
        }
    }
}

/// Two-occasion IOV + `iiv_on_ruv` subject (the [`iov_ruv_subject`] geometry) with
/// occasion 2's two tail observations flagged `CENS = 1` — the **triple**
/// M3 + IOV + `iiv_on_ruv` (#591). Shallow left-censoring (≈ 0.85·f) keeps the
/// inverse Mills ratio well-scaled.
fn iov_m3_ruv_subject(model: &CompiledModel, theta: &[f64]) -> Subject {
    let mut subject = iov_ruv_subject(model, theta);
    let n = subject.observations.len();
    subject.cens[n - 2] = 1;
    subject.cens[n - 1] = 1;
    subject
}

/// The triple **M3 + IOV + `iiv_on_ruv`** through the production FOCEI packed
/// gradient (#591): the censored residual-eta cross coefficients `(C·z, C·m)` enter
/// the true inner Hessian / `mixed_eta_theta` / `sigma_block` over the stacked
/// `[η_bsv, κ]` layout, and `residual_inner_obs` adds the `h·z` residual-eta column
/// to the inner gradient. The full `[θ, Ω_bsv, σ, Ω_iov]` analytic gradient must
/// match Richardson reconverged FD of the FOCEI IOV marginal — the same objective
/// `precise_ebe_iov` (now censored-`iiv_on_ruv`-aware) reconverges against. Proves the
/// gate flip ships a correct gradient for the triple.
#[test]
fn iov_m3_iiv_on_ruv_packed_gradient_matches_reconverged_fd() {
    let mut model = parse_model_string(WARFARIN_IOV_RUV).expect("parse warfarin IOV + iiv_on_ruv");
    model.bloq_method = crate::types::BloqMethod::M3;
    assert_eq!(model.residual_error_eta, Some(3));
    assert!(crate::sens::provider::iov_analytical_supported(&model));
    let theta = vec![0.22, 11.0, 1.4];
    let mut params = model.default_params.clone();
    params.theta = theta.clone();
    let subject = iov_m3_ruv_subject(&model, &theta);
    assert!(
        subject.cens.iter().any(|&c| c != 0),
        "subject must be censored"
    );
    let template = params.clone();
    let x = crate::estimation::parameterization::pack_params(&params);

    let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);
    let analytic = subject_packed_gradient_iov(&model, &subject, &template, &x, &stacked)
        .expect("IOV + M3 + iiv_on_ruv packed gradient supported");

    let f = |xx: &[f64]| -> f64 {
        let p = unpack_params(xx, &template);
        marginal_nll_iov(&model, &subject, &p)
    };
    for i in 0..x.len() {
        let h = 1e-4 * (1.0 + x[i].abs());
        let fd_at = |hh: f64| -> f64 {
            let mut xp = x.clone();
            xp[i] += hh;
            let mut xm = x.clone();
            xm[i] -= hh;
            (f(&xp) - f(&xm)) / (2.0 * hh)
        };
        let f1 = fd_at(h);
        let f2 = fd_at(h / 2.0);
        let fd = (4.0 * f2 - f1) / 3.0; // Richardson
        eprintln!(
            "iov+m3+ruv packed x[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
            analytic[i],
            fd,
            (analytic[i] - fd).abs() / fd.abs().max(1e-9)
        );
        approx::assert_relative_eq!(analytic[i], fd, max_relative = 3e-3, epsilon = 2e-5);
    }
}

/// FOCE-IOV-M3 (#591): the analytic **FOCE** (non-interaction) IOV packed gradient on
/// a censored subject must match Richardson reconverged FD of the FOCE-IOV-M3 marginal
/// (`foce_subject_nll_iov(interaction = false)`, which no longer promotes censored
/// subjects to interaction). Censored rows leave the augmented Sheiner–Beal marginal
/// and re-enter as `−logΦ` data terms at the population (η=0, κ=0) variance — the
/// stacked-layout analogue of `subject_packed_gradient_foce`. Exercises the censored
/// θ / σ blocks and the M3-aware `subject_eta_dx_iov` σ EBE-response.
#[test]
fn iov_m3_foce_packed_gradient_matches_reconverged_fd() {
    let mut model = parse_model_string(WARFARIN_IOV).expect("parse warfarin IOV");
    model.bloq_method = crate::types::BloqMethod::M3;
    let theta = vec![0.22, 11.0, 1.4];
    let mut params = model.default_params.clone();
    params.theta = theta.clone();
    let subject = iov_m3_subject(&model, &theta);
    assert!(
        subject.cens.iter().any(|&c| c != 0),
        "subject must be censored"
    );
    let template = params.clone();
    let x = crate::estimation::parameterization::pack_params(&params);

    let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);
    let analytic = subject_packed_gradient_foce_iov(&model, &subject, &template, &x, &stacked)
        .expect("FOCE-IOV-M3 packed gradient now supported (censored SB term)");

    let f = |xx: &[f64]| -> f64 {
        let p = unpack_params(xx, &template);
        marginal_nll_iov_inter(&model, &subject, &p, false)
    };
    for i in 0..x.len() {
        let h = 1e-4 * (1.0 + x[i].abs());
        let fd_at = |hh: f64| -> f64 {
            let mut xp = x.clone();
            xp[i] += hh;
            let mut xm = x.clone();
            xm[i] -= hh;
            (f(&xp) - f(&xm)) / (2.0 * hh)
        };
        let f1 = fd_at(h);
        let f2 = fd_at(h / 2.0);
        let fd = (4.0 * f2 - f1) / 3.0;
        eprintln!(
            "iov foce+m3 x[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
            analytic[i],
            fd,
            (analytic[i] - fd).abs() / fd.abs().max(1e-9)
        );
        approx::assert_relative_eq!(analytic[i], fd, max_relative = 3e-3, epsilon = 2e-5);
    }
}

/// Right-censored (`CENS = -1`) regression of the **FOCEI** IOV+M3 packed gradient.
/// The signed `m3_censored_outer` feeds the upper-tail `(g1, g2, C·z, C·m)` into the
/// stacked assembly; the gradient must match Richardson reconverged FD of the FOCEI
/// marginal (upper-tail `m3_logcdf`). Mirror of
/// `iov_m3_packed_gradient_matches_reconverged_fd` with the tail flipped.
#[test]
fn iov_m3_right_censored_packed_gradient_matches_reconverged_fd() {
    let mut model = parse_model_string(WARFARIN_IOV).expect("parse warfarin IOV");
    model.bloq_method = crate::types::BloqMethod::M3;
    let theta = vec![0.22, 11.0, 1.4];
    let mut params = model.default_params.clone();
    params.theta = theta.clone();
    let subject = iov_m3_subject_right(&model, &theta);
    assert!(
        subject.cens.iter().any(|&c| c < 0),
        "must be right-censored"
    );
    let template = params.clone();
    let x = crate::estimation::parameterization::pack_params(&params);

    let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);
    let analytic = subject_packed_gradient_iov(&model, &subject, &template, &x, &stacked)
        .expect("IOV + M3 packed gradient supported");

    let f = |xx: &[f64]| -> f64 {
        let p = unpack_params(xx, &template);
        marginal_nll_iov(&model, &subject, &p)
    };
    for i in 0..x.len() {
        let h = 1e-4 * (1.0 + x[i].abs());
        let fd_at = |hh: f64| -> f64 {
            let mut xp = x.clone();
            xp[i] += hh;
            let mut xm = x.clone();
            xm[i] -= hh;
            (f(&xp) - f(&xm)) / (2.0 * hh)
        };
        let f1 = fd_at(h);
        let f2 = fd_at(h / 2.0);
        let fd = (4.0 * f2 - f1) / 3.0;
        approx::assert_relative_eq!(analytic[i], fd, max_relative = 3e-3, epsilon = 2e-5);
    }
}

/// Right-censored (`CENS = -1`) regression of the **FOCE** (non-interaction) IOV+M3
/// packed gradient: the hand-written censored `Cens` θ/σ/η̂-coupling terms in
/// `subject_packed_gradient_foce_iov` each carry the tail sign `σ`, so the gradient
/// must match Richardson reconverged FD of the FOCE-IOV-M3 marginal for above-ULOQ
/// rows. Mirror of `iov_m3_foce_packed_gradient_matches_reconverged_fd`.
#[test]
fn iov_m3_foce_right_censored_packed_gradient_matches_reconverged_fd() {
    let mut model = parse_model_string(WARFARIN_IOV).expect("parse warfarin IOV");
    model.bloq_method = crate::types::BloqMethod::M3;
    let theta = vec![0.22, 11.0, 1.4];
    let mut params = model.default_params.clone();
    params.theta = theta.clone();
    let subject = iov_m3_subject_right(&model, &theta);
    assert!(
        subject.cens.iter().any(|&c| c < 0),
        "must be right-censored"
    );
    let template = params.clone();
    let x = crate::estimation::parameterization::pack_params(&params);

    let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);
    let analytic = subject_packed_gradient_foce_iov(&model, &subject, &template, &x, &stacked)
        .expect("FOCE-IOV-M3 packed gradient supported");

    let f = |xx: &[f64]| -> f64 {
        let p = unpack_params(xx, &template);
        marginal_nll_iov_inter(&model, &subject, &p, false)
    };
    for i in 0..x.len() {
        let h = 1e-4 * (1.0 + x[i].abs());
        let fd_at = |hh: f64| -> f64 {
            let mut xp = x.clone();
            xp[i] += hh;
            let mut xm = x.clone();
            xm[i] -= hh;
            (f(&xp) - f(&xm)) / (2.0 * hh)
        };
        let f1 = fd_at(h);
        let f2 = fd_at(h / 2.0);
        let fd = (4.0 * f2 - f1) / 3.0;
        approx::assert_relative_eq!(analytic[i], fd, max_relative = 3e-3, epsilon = 2e-5);
    }
}

/// The full analytic IOV **FOCE** (non-interaction) packed gradient must match
/// the Richardson reconverged FD of the production IOV FOCE marginal
/// (`foce_subject_nll_iov` with `interaction = false`, the Sheiner–Beal
/// linearized objective) over every packed coordinate — the path
/// `method = foce` (warfarin_iov's default) actually exercises.
#[test]
fn iov_packed_gradient_foce_matches_reconverged_fd() {
    let model = parse_model_string(WARFARIN_IOV).expect("parse warfarin IOV");
    let theta = vec![0.22, 11.0, 1.4];
    let mut params = model.default_params.clone();
    params.theta = theta.clone();
    let subject = iov_subject_outer(&model, &theta);
    let template = params.clone();
    let x = crate::estimation::parameterization::pack_params(&params);

    let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);
    let analytic = subject_packed_gradient_foce_iov(&model, &subject, &template, &x, &stacked)
        .expect("IOV FOCE packed gradient supported");

    let f = |xx: &[f64]| -> f64 {
        let p = unpack_params(xx, &template);
        marginal_nll_iov_inter(&model, &subject, &p, false)
    };
    for i in 0..x.len() {
        let h = 1e-4 * (1.0 + x[i].abs());
        let fd_at = |hh: f64| -> f64 {
            let mut xp = x.clone();
            xp[i] += hh;
            let mut xm = x.clone();
            xm[i] -= hh;
            (f(&xp) - f(&xm)) / (2.0 * hh)
        };
        let f1 = fd_at(h);
        let f2 = fd_at(h / 2.0);
        let fd = (4.0 * f2 - f1) / 3.0;
        eprintln!(
            "iov foce x[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
            analytic[i],
            fd,
            (analytic[i] - fd).abs() / fd.abs().max(1e-9)
        );
        approx::assert_relative_eq!(analytic[i], fd, max_relative = 2e-3, epsilon = 2e-5);
    }
}

/// IOV with an EVID=4 washout reset at the occasion boundary: the same
/// two-occasion subject as `iov_subject_outer`, but occasion 2 rebuilds from
/// zero (no carryover). The full packed gradient — FOCEI **and** FOCE — must
/// still match Richardson reconverged FD of the IOV marginal, confirming the
/// reset jet flows through the stacked-η / block-Ω assembly unchanged.
fn iov_subject_outer_reset(model: &CompiledModel, theta: &[f64]) -> Subject {
    let mut s = iov_subject_outer(model, theta);
    s.reset_times = vec![24.0];
    assert!(s.has_resets(), "fixture must carry a reset");
    // Re-synthesise observations through the reset-aware predict_iov so ε ≠ 0.
    let preds = crate::pk::predict_iov(
        model,
        &s,
        theta,
        &[0.12, -0.08, 0.2],
        &[vec![0.05], vec![-0.07]],
    );
    s.observations = preds.iter().map(|p| p * 0.85).collect();
    s
}

#[test]
fn iov_packed_gradient_reset_matches_reconverged_fd() {
    let model = parse_model_string(WARFARIN_IOV).expect("parse warfarin IOV");
    let theta = vec![0.22, 11.0, 1.4];
    let mut params = model.default_params.clone();
    params.theta = theta.clone();
    let subject = iov_subject_outer_reset(&model, &theta);
    let template = params.clone();
    let x = crate::estimation::parameterization::pack_params(&params);

    let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);

    // FOCEI (Almquist Laplace) and FOCE (Sheiner–Beal) over the reset subject.
    for interaction in [true, false] {
        let analytic = if interaction {
            subject_packed_gradient_iov(&model, &subject, &template, &x, &stacked)
        } else {
            subject_packed_gradient_foce_iov(&model, &subject, &template, &x, &stacked)
        }
        .expect("IOV+reset packed gradient supported");

        let f = |xx: &[f64]| -> f64 {
            let p = unpack_params(xx, &template);
            marginal_nll_iov_inter(&model, &subject, &p, interaction)
        };
        for i in 0..x.len() {
            let h = 1e-4 * (1.0 + x[i].abs());
            let fd_at = |hh: f64| -> f64 {
                let mut xp = x.clone();
                xp[i] += hh;
                let mut xm = x.clone();
                xm[i] -= hh;
                (f(&xp) - f(&xm)) / (2.0 * hh)
            };
            let f1 = fd_at(h);
            let f2 = fd_at(h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0; // Richardson
            eprintln!(
                "iov reset interaction={interaction} x[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic[i],
                fd,
                (analytic[i] - fd).abs() / fd.abs().max(1e-9)
            );
            approx::assert_relative_eq!(analytic[i], fd, max_relative = 2e-3, epsilon = 2e-5);
        }
    }
}

// --- IOV combined with a time-varying covariate ---

/// IOV model that *also* carries a WT-on-CL covariate (`THETA_WT`), so a
/// subject whose WT varies across records switches `CL` by both κ (occasion)
/// and WT (covariate). θ = [TVCL, TVV, TVKA, THETA_WT].
const WARFARIN_IOV_TVCOV: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.02
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;

/// Two-occasion IOV subject carrying a WT covariate that varies across records
/// (lighter in occasion 1, heavier in occasion 2, plus an EVID=2 breakpoint at
/// t=18). Observations are synthesised through `predict_iov` (which seeds each
/// event at its own covariate) so residuals are realistic on the merged path.
fn iov_tvcov_subject_outer(model: &CompiledModel, theta: &[f64]) -> Subject {
    let obs_times = vec![1.0, 6.0, 12.0, 25.0, 30.0, 36.0];
    let occasions = vec![1u32, 1, 1, 2, 2, 2];
    let obs_wts = [70.0, 72.0, 78.0, 88.0, 90.0, 95.0];
    let n = obs_times.len();
    let wt_map = |w: f64| {
        let mut m = std::collections::HashMap::new();
        m.insert("WT".to_string(), w);
        m
    };
    let mut subject = Subject {
        id: "1".to_string(),
        doses: vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
        ],
        obs_times,
        obs_raw_times: Vec::new(),
        observations: vec![0.0; n],
        obs_cmts: vec![1; n],
        covariates: wt_map(70.0),
        dose_covariates: vec![wt_map(70.0), wt_map(85.0)],
        obs_covariates: obs_wts.iter().map(|&w| wt_map(w)).collect(),
        pk_only_times: vec![18.0],
        pk_only_covariates: vec![wt_map(85.0)],
        reset_times: Vec::new(),
        cens: vec![0; n],
        occasions,
        obs_l2: Vec::new(),
        dose_occasions: vec![1, 2],
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    let preds = crate::pk::predict_iov(
        model,
        &subject,
        theta,
        &[0.12, -0.08, 0.2],
        &[vec![0.05], vec![-0.07]],
    );
    subject.observations = preds.iter().map(|p| p * 0.85).collect();
    subject
}

/// The full analytic IOV+TV-cov **packed** gradient — FOCEI **and** FOCE — must
/// match the Richardson reconverged FD of the production IOV marginal over every
/// packed coordinate (θ incl. `THETA_WT`, Ω_bsv, σ, Ω_iov). Closes the merged
/// IOV × time-varying-covariate path end to end against the same `predict_iov`-
/// grounded objective the non-TV IOV tests use.
#[test]
fn iov_tvcov_packed_gradient_matches_reconverged_fd() {
    let model = parse_model_string(WARFARIN_IOV_TVCOV).expect("parse warfarin IOV+TVcov");
    let theta = vec![0.22, 11.0, 1.4, 0.7];
    let mut params = model.default_params.clone();
    params.theta = theta.clone();
    let subject = iov_tvcov_subject_outer(&model, &theta);
    assert!(subject.has_tv_covariates(), "fixture must carry TV cov");
    let template = params.clone();
    let x = crate::estimation::parameterization::pack_params(&params);

    let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);

    for interaction in [true, false] {
        let analytic = if interaction {
            subject_packed_gradient_iov(&model, &subject, &template, &x, &stacked)
        } else {
            subject_packed_gradient_foce_iov(&model, &subject, &template, &x, &stacked)
        }
        .expect("IOV+TVcov packed gradient supported");

        let f = |xx: &[f64]| -> f64 {
            let p = unpack_params(xx, &template);
            marginal_nll_iov_inter(&model, &subject, &p, interaction)
        };
        for i in 0..x.len() {
            let h = 1e-4 * (1.0 + x[i].abs());
            let fd_at = |hh: f64| -> f64 {
                let mut xp = x.clone();
                xp[i] += hh;
                let mut xm = x.clone();
                xm[i] -= hh;
                (f(&xp) - f(&xm)) / (2.0 * hh)
            };
            let f1 = fd_at(h);
            let f2 = fd_at(h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0; // Richardson
            eprintln!(
                "iov tvcov interaction={interaction} x[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic[i],
                fd,
                (analytic[i] - fd).abs() / fd.abs().max(1e-9)
            );
            approx::assert_relative_eq!(analytic[i], fd, max_relative = 2e-3, epsilon = 2e-5);
        }
    }
}

/// Closed-form IOV + η-dependent `ExpressionScale` `obs_scale = V` (#486): the full
/// analytic packed gradient — FOCEI **and** FOCE — must match the Richardson reconverged
/// FD of the production IOV marginal over every packed coordinate (θ, Ω_bsv, σ, Ω_iov).
/// The end-to-end population-level confirmation that the new per-occasion post-walk scale
/// quotient rides the block-Ω `prepare_stacked` assembly on both objectives.
#[test]
fn iov_expression_scale_packed_gradient_matches_reconverged_fd() {
    const WARFARIN_IOV_EXPRSCALE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.02
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
"#;
    let model = parse_model_string(WARFARIN_IOV_EXPRSCALE).expect("parse IOV + obs_scale");
    assert!(
        matches!(
            model.scaling,
            crate::types::ScalingSpec::ExpressionScale { .. }
        ),
        "fixture must carry an expression obs_scale"
    );
    assert!(crate::sens::provider::iov_analytical_supported(&model));
    let theta = vec![0.22, 11.0, 1.4];
    let mut params = model.default_params.clone();
    params.theta = theta.clone();
    let subject = iov_subject_outer(&model, &theta);
    let template = params.clone();
    let x = crate::estimation::parameterization::pack_params(&params);

    let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);

    for interaction in [true, false] {
        let analytic = if interaction {
            subject_packed_gradient_iov(&model, &subject, &template, &x, &stacked)
        } else {
            subject_packed_gradient_foce_iov(&model, &subject, &template, &x, &stacked)
        }
        .expect("IOV + obs_scale packed gradient supported");

        let f = |xx: &[f64]| -> f64 {
            let p = unpack_params(xx, &template);
            marginal_nll_iov_inter(&model, &subject, &p, interaction)
        };
        for i in 0..x.len() {
            let h = 1e-4 * (1.0 + x[i].abs());
            let fd_at = |hh: f64| -> f64 {
                let mut xp = x.clone();
                xp[i] += hh;
                let mut xm = x.clone();
                xm[i] -= hh;
                (f(&xp) - f(&xm)) / (2.0 * hh)
            };
            let f1 = fd_at(h);
            let f2 = fd_at(h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0; // Richardson
            approx::assert_relative_eq!(analytic[i], fd, max_relative = 2e-3, epsilon = 2e-5);
        }
    }
}

/// **Closed-form IOV + M3 BLOQ + `ExpressionScale` `obs_scale = V`** (#651 review #1):
/// the new gate arm admits an `ExpressionScale` divisor orthogonally to the M3-censoring
/// clause the gate already allowed, so this triple now routes to the analytic packed
/// gradient with no FD fallback. The two features compose at different layers — the scale
/// is a per-occasion post-walk quotient over the `(θ, stacked-η)` jet **including its
/// second-order derivatives**, and the M3 `−logΦ(z)` tail-probability coefficient enters
/// the FOCEI `log|H̃|` over that already-scaled jet keyed on `subject.cens[j]`. If the
/// scaled second-order sensitivities fed the censored curvature inconsistently, the SEs /
/// OFV would be silently wrong. The full `[θ, Ω_bsv, σ, Ω_iov]` analytic packed gradient
/// must match Richardson reconverged FD of the FOCEI IOV marginal on both censoring
/// tails — the closed-form twin of `iov_m3_ode_expression_scale_packed_gradient_matches_reconverged_fd`.
#[test]
fn iov_m3_expression_scale_packed_gradient_matches_reconverged_fd() {
    const WARFARIN_IOV_M3_EXPRSCALE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.02
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
"#;
    let mut model =
        parse_model_string(WARFARIN_IOV_M3_EXPRSCALE).expect("parse IOV + M3 + obs_scale");
    model.bloq_method = crate::types::BloqMethod::M3;
    assert!(
        matches!(
            model.scaling,
            crate::types::ScalingSpec::ExpressionScale { .. }
        ),
        "fixture must carry an expression obs_scale"
    );
    assert!(
        crate::sens::provider::iov_analytical_supported(&model),
        "closed-form IOV + M3 + ExpressionScale obs_scale must be analytic (#651)"
    );
    let theta = vec![0.22, 11.0, 1.4];
    let mut params = model.default_params.clone();
    params.theta = theta.clone();

    for right in [false, true] {
        let subject = if right {
            iov_m3_subject_right(&model, &theta)
        } else {
            iov_m3_subject(&model, &theta)
        };
        assert!(
            subject.cens.iter().any(|&c| c != 0),
            "subject must be censored"
        );
        let template = params.clone();
        let x = crate::estimation::parameterization::pack_params(&params);

        let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);
        let analytic = subject_packed_gradient_iov(&model, &subject, &template, &x, &stacked)
            .expect("IOV + M3 + obs_scale packed gradient supported");

        let f = |xx: &[f64]| -> f64 {
            let p = unpack_params(xx, &template);
            marginal_nll_iov(&model, &subject, &p)
        };
        for i in 0..x.len() {
            let h = 1e-4 * (1.0 + x[i].abs());
            let fd_at = |hh: f64| -> f64 {
                let mut xp = x.clone();
                xp[i] += hh;
                let mut xm = x.clone();
                xm[i] -= hh;
                (f(&xp) - f(&xm)) / (2.0 * hh)
            };
            let f1 = fd_at(h);
            let f2 = fd_at(h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0; // Richardson
            eprintln!(
                    "iov+m3+exprscale (right={right}) packed x[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                    analytic[i],
                    fd,
                    (analytic[i] - fd).abs() / fd.abs().max(1e-9)
                );
            approx::assert_relative_eq!(analytic[i], fd, max_relative = 3e-3, epsilon = 2e-5);
        }
    }
}

/// **Closed-form IOV + `iiv_on_ruv` + `ExpressionScale` `obs_scale = V`** (#651 review #1),
/// plus the **triple** with M3 censoring. `iiv_on_ruv` scales the residual variance by
/// `exp(2·η_ruv)` and rides a zero `∂f/∂η_ruv` structural column applied downstream by the
/// provider-agnostic `prepare_stacked` assembly, independently of the post-walk scale
/// quotient — but the combination was untested when the gate began admitting `obs_scale`.
/// The production FOCEI packed gradient must match Richardson reconverged FD of the scaled
/// FOCEI IOV marginal over every packed coordinate (incl. the `Ω_RUV` block), with and
/// without censored rows. Closed-form twin of the ODE `iov_iiv_on_ruv` / `iov_m3_iiv_on_ruv`
/// packed-gradient tests.
#[test]
fn iov_iiv_on_ruv_expression_scale_packed_gradient_matches_reconverged_fd() {
    const WARFARIN_IOV_RUV_EXPRSCALE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  omega ETA_RUV ~ 0.05
  kappa KAPPA_CL ~ 0.02
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
  iiv_on_ruv = ETA_RUV
[fit_options]
  method     = focei
  iov_column = OCC
"#;
    let mut model =
        parse_model_string(WARFARIN_IOV_RUV_EXPRSCALE).expect("parse IOV + iiv_on_ruv + obs_scale");
    assert_eq!(model.residual_error_eta, Some(3));
    assert!(
        crate::sens::provider::iov_analytical_supported(&model),
        "closed-form IOV + iiv_on_ruv + ExpressionScale obs_scale must be analytic (#651)"
    );
    let theta = vec![0.22, 11.0, 1.4];
    let mut params = model.default_params.clone();
    params.theta = theta.clone();

    // Plain iiv_on_ruv + obs_scale, then the triple with M3-censored occasion-2 tail.
    for m3 in [false, true] {
        if m3 {
            model.bloq_method = crate::types::BloqMethod::M3;
        }
        let subject = if m3 {
            iov_m3_ruv_subject(&model, &theta)
        } else {
            iov_ruv_subject(&model, &theta)
        };
        let template = params.clone();
        let x = crate::estimation::parameterization::pack_params(&params);

        let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);
        let analytic = subject_packed_gradient_iov(&model, &subject, &template, &x, &stacked)
            .expect("IOV + iiv_on_ruv + obs_scale packed gradient supported");

        let f = |xx: &[f64]| -> f64 {
            let p = unpack_params(xx, &template);
            marginal_nll_iov(&model, &subject, &p)
        };
        for i in 0..x.len() {
            let h = 1e-4 * (1.0 + x[i].abs());
            let fd_at = |hh: f64| -> f64 {
                let mut xp = x.clone();
                xp[i] += hh;
                let mut xm = x.clone();
                xm[i] -= hh;
                (f(&xp) - f(&xm)) / (2.0 * hh)
            };
            let f1 = fd_at(h);
            let f2 = fd_at(h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0; // Richardson
            eprintln!(
                "iov+ruv+exprscale (m3={m3}) packed x[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic[i],
                fd,
                (analytic[i] - fd).abs() / fd.abs().max(1e-9)
            );
            approx::assert_relative_eq!(analytic[i], fd, max_relative = 3e-3, epsilon = 2e-5);
        }
    }
}

// ── #576/#486: custom / time-varying residual-error σ magnitude ────────
//
// Three fixture models, one per magnitude-argument family named in the PR5
// handoff (`plans/analytic-gradient-completion/pr5-sigma-magnitude.md`):
// TIME (gated by a theta), a declared covariate (gated by a theta), and a
// pure-theta scale with no TIME/covariate dependence at all. Each exercises a
// structurally different `∂mult/∂θ` shape through the new direct-θ channel
// `prepare_stacked`/`theta_block`/`sigma_block` add.

const WARFARIN_RUV_TIME: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta RUV_LATE(1.5, 0.1, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR * (1.0 + RUV_LATE * TIME / 48.0))
"#;

const WARFARIN_RUV_COV: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta RUV_WT(0.01, 0.0, 1.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR * (1.0 + RUV_WT * (WT - 70.0)))
[covariates]
  WT continuous
"#;

const WARFARIN_RUV_THETA: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta RUV_SCALE(1.2, 0.1, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR * RUV_SCALE)
"#;

/// Shared check: analytic `subject_theta_gradient` / `subject_sigma_gradient`
/// vs Richardson-reconverged FD of the (magnitude-aware) marginal NLL, for a
/// magnitude-active model — the FD-vs-production leg of the validation triple.
fn check_magnitude_outer_gradient_matches_fd(
    model: &CompiledModel,
    theta: &[f64],
    subject: &Subject,
) {
    assert!(
        model.has_custom_ruv_magnitude(),
        "fixture must carry an active custom magnitude"
    );
    let mut params = model.default_params.clone();
    params.theta = theta.to_vec();
    let eta_hat = precise_ebe(model, subject, &params);

    let analytic_theta =
        subject_theta_gradient(model, subject, &params, &eta_hat).expect("supported");
    let fd_theta_at = |m: usize, h: f64| -> f64 {
        let mut pp = params.clone();
        pp.theta[m] += h;
        let mut pm = params.clone();
        pm.theta[m] -= h;
        (marginal_nll(model, subject, &pp) - marginal_nll(model, subject, &pm)) / (2.0 * h)
    };
    for m in 0..theta.len() {
        let h = 1e-4 * (1.0 + theta[m].abs());
        let f1 = fd_theta_at(m, h);
        let f2 = fd_theta_at(m, h / 2.0);
        let fd = (4.0 * f2 - f1) / 3.0; // Richardson
        eprintln!(
            "magnitude theta[{m}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
            analytic_theta[m],
            fd,
            (analytic_theta[m] - fd).abs() / fd.abs().max(1e-12)
        );
        approx::assert_relative_eq!(analytic_theta[m], fd, max_relative = 2e-3, epsilon = 1e-6);
    }

    let analytic_sigma =
        subject_sigma_gradient(model, subject, &params, &eta_hat).expect("supported");
    let sig0 = params.sigma.values.clone();
    let fd_sigma_at = |k: usize, h: f64| -> f64 {
        let mut pp = params.clone();
        pp.sigma.values[k] += h;
        let mut pm = params.clone();
        pm.sigma.values[k] -= h;
        (marginal_nll(model, subject, &pp) - marginal_nll(model, subject, &pm)) / (2.0 * h)
    };
    for k in 0..sig0.len() {
        let h = 1e-4 * (1.0 + sig0[k].abs());
        let f1 = fd_sigma_at(k, h);
        let f2 = fd_sigma_at(k, h / 2.0);
        let fd = (4.0 * f2 - f1) / 3.0;
        eprintln!(
            "magnitude sigma[{k}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
            analytic_sigma[k],
            fd,
            (analytic_sigma[k] - fd).abs() / fd.abs().max(1e-12)
        );
        approx::assert_relative_eq!(analytic_sigma[k], fd, max_relative = 2e-3, epsilon = 1e-6);
    }

    // Same combination via the packed gradient (θ, Ω, σ interleaved), for the
    // path the outer optimizer actually calls.
    let template = params.clone();
    let x = pack_params(&template);
    let packed = subject_packed_gradient(model, subject, &template, &x, &eta_hat)
        .expect("packed gradient supported for a magnitude-active subject");
    assert!(
        packed.iter().all(|v| v.is_finite()),
        "packed gradient must be finite for a magnitude-active subject"
    );
}

#[test]
fn magnitude_time_family_outer_gradient_matches_fd() {
    let model = parse_model_string(WARFARIN_RUV_TIME).expect("parse");
    let theta = vec![0.22, 11.0, 1.4, 1.6];
    let times = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0, 48.0];
    let subject = subject_with_obs(&model, &theta, &times);
    check_magnitude_outer_gradient_matches_fd(&model, &theta, &subject);
}

#[test]
fn magnitude_covariate_family_outer_gradient_matches_fd() {
    let model = parse_model_string(WARFARIN_RUV_COV).expect("parse");
    let theta = vec![0.22, 11.0, 1.4, 0.012];
    let times = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0, 48.0];
    let mut subject = subject_with_obs(&model, &theta, &times);
    subject.covariates = HashMap::from([("WT".to_string(), 82.0)]);
    check_magnitude_outer_gradient_matches_fd(&model, &theta, &subject);
}

#[test]
fn magnitude_theta_family_outer_gradient_matches_fd() {
    let model = parse_model_string(WARFARIN_RUV_THETA).expect("parse");
    let theta = vec![0.22, 11.0, 1.4, 1.3];
    let times = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0, 48.0];
    let subject = subject_with_obs(&model, &theta, &times);
    check_magnitude_outer_gradient_matches_fd(&model, &theta, &subject);
}

/// Custom / time-varying residual-magnitude combined with **`iiv_on_ruv`**
/// (Tier-1 follow-up to #644/#659): the residual-eta `c̃`-column `d/R` is a
/// function of θ through the magnitude, so `theta_block` adds its `m_vec[rr]` row
/// and `∂(d/R)/∂θ·w[rr]` log|H̃| term (mirroring `sigma_block`). The packed θ/Ω/σ
/// gradient must match reconverged FD of the FOCEI OFV (EBE via `precise_ebe_ruv`).
const WARFARIN_RUV_TIME_IIV: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta RUV_LATE(1.5, 0.1, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  omega ETA_RUV ~ 0.10
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR * (1.0 + RUV_LATE * TIME / 48.0))
  iiv_on_ruv = ETA_RUV
"#;

#[test]
fn magnitude_iiv_on_ruv_packed_matches_fd() {
    let model = parse_model_string(WARFARIN_RUV_TIME_IIV).expect("parse");
    assert!(
        model.has_custom_ruv_magnitude(),
        "fixture must carry an active custom magnitude"
    );
    assert_eq!(model.residual_error_eta, Some(3), "ETA_RUV is the 4th eta");
    run_ruv_packed_check(&model, &[0.22, 11.0, 1.4, 1.6]);
}

/// FOCE (non-interaction) analog of `check_magnitude_outer_gradient_matches_fd`:
/// the analytic FOCE packed gradient of a magnitude-active subject must match the
/// reconverged-FD of ferx's own (magnitude-aware) Sheiner–Beal marginal, across
/// every packed θ/Ω/σ coordinate. Exercises the direct-θ `∂R⁰/∂θ` term and the
/// magnitude-scaled `R⁰`/`∂R⁰/∂σ` the FOCE port threads in (#486).
fn check_magnitude_foce_packed_matches_fd(model: &CompiledModel, theta: &[f64], subject: &Subject) {
    assert!(
        model.has_custom_ruv_magnitude(),
        "fixture must carry an active custom magnitude"
    );
    let mut template = model.default_params.clone();
    template.theta = theta.to_vec();
    let x = pack_params(&template);
    let params = unpack_params(&x, &template);
    let eta_hat = precise_ebe(model, subject, &params);
    let analytic = subject_packed_gradient_foce(model, subject, &template, &x, &eta_hat)
        .expect("FOCE magnitude packed gradient supported");
    assert!(
        analytic.iter().all(|v| v.is_finite()),
        "FOCE magnitude packed gradient must be finite"
    );
    // FD of the (magnitude-aware) FOCE marginal, reconverging the EBE per point.
    let ofv = |xv: &[f64]| -> f64 {
        let p = unpack_params(xv, &template);
        marginal_nll_foce(model, subject, &p)
    };
    assert_grad_matches_richardson_fd(&x, &analytic, ofv, 3e-3, 1e-5);
}

#[test]
fn magnitude_time_family_foce_packed_matches_fd() {
    let model = parse_model_string(WARFARIN_RUV_TIME).expect("parse");
    let theta = vec![0.22, 11.0, 1.4, 1.6];
    let times = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0, 48.0];
    let subject = subject_with_obs(&model, &theta, &times);
    check_magnitude_foce_packed_matches_fd(&model, &theta, &subject);
}

#[test]
fn magnitude_covariate_family_foce_packed_matches_fd() {
    let model = parse_model_string(WARFARIN_RUV_COV).expect("parse");
    let theta = vec![0.22, 11.0, 1.4, 0.012];
    let times = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0, 48.0];
    let mut subject = subject_with_obs(&model, &theta, &times);
    subject.covariates = HashMap::from([("WT".to_string(), 82.0)]);
    check_magnitude_foce_packed_matches_fd(&model, &theta, &subject);
}

#[test]
fn magnitude_theta_family_foce_packed_matches_fd() {
    let model = parse_model_string(WARFARIN_RUV_THETA).expect("parse");
    let theta = vec![0.22, 11.0, 1.4, 1.3];
    let times = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0, 48.0];
    let subject = subject_with_obs(&model, &theta, &times);
    check_magnitude_foce_packed_matches_fd(&model, &theta, &subject);
}

/// [`WARFARIN_IOV`] + a TIME-varying proportional σ magnitude — drives the
/// **FOCE-IOV** magnitude packed gradient (`subject_packed_gradient_foce_iov`).
const WARFARIN_IOV_RUV_MAG: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta RUV_LATE(1.5, 0.1, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.02
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR * (1.0 + RUV_LATE * TIME / 48.0))
[fit_options]
  method     = foce
  iov_column = OCC
"#;

/// FOCE-IOV magnitude: the analytic stacked-`[η_bsv,κ]` FOCE packed gradient must
/// match reconverged-FD of the (magnitude-aware) FOCE-IOV Sheiner–Beal marginal,
/// across every packed θ/Ω_bsv/σ/Ω_iov coordinate (#486, the IOV twin of the
/// non-IOV FOCE magnitude tests above).
#[test]
fn magnitude_foce_iov_packed_matches_fd() {
    use crate::estimation::parameterization::pack_params;
    let model = parse_model_string(WARFARIN_IOV_RUV_MAG).expect("parse");
    assert!(model.has_custom_ruv_magnitude());
    assert!(model.n_kappa > 0, "fixture must carry IOV");
    let theta = vec![0.22, 11.0, 1.4, 1.6];
    let mut params = model.default_params.clone();
    params.theta = theta.clone();
    let subject = iov_subject_outer(&model, &theta);
    let template = params.clone();
    let x = pack_params(&params);
    let (stacked, _e, _k, _h) = precise_ebe_iov(&model, &subject, &params);
    let analytic = subject_packed_gradient_foce_iov(&model, &subject, &template, &x, &stacked)
        .expect("FOCE-IOV magnitude packed gradient supported");
    assert!(
        analytic.iter().all(|v| v.is_finite()),
        "FOCE-IOV magnitude packed gradient must be finite"
    );
    // FD reference must be the FOCE (non-interaction) marginal — matching the
    // gradient under test (`marginal_nll_iov` is the FOCEI variant).
    let f = |xx: &[f64]| -> f64 {
        let p = unpack_params(xx, &template);
        marginal_nll_iov_inter(&model, &subject, &p, false)
    };
    for i in 0..x.len() {
        let h = 1e-4 * (1.0 + x[i].abs());
        let fd_at = |hh: f64| -> f64 {
            let mut xp = x.clone();
            xp[i] += hh;
            let mut xm = x.clone();
            xm[i] -= hh;
            (f(&xp) - f(&xm)) / (2.0 * hh)
        };
        let f1 = fd_at(h);
        let f2 = fd_at(h / 2.0);
        let fd = (4.0 * f2 - f1) / 3.0; // Richardson
        eprintln!(
            "foce-iov magnitude x[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
            analytic[i],
            fd,
            (analytic[i] - fd).abs() / fd.abs().max(1e-12)
        );
        approx::assert_relative_eq!(analytic[i], fd, max_relative = 3e-3, epsilon = 2e-5);
    }
}

/// [`WARFARIN_IOV_RUV`] + a TIME-varying proportional σ magnitude — the triple
/// custom-magnitude × `iiv_on_ruv` × IOV on the **FOCEI** packed gradient
/// (`subject_packed_gradient_iov` → `prepare_stacked`). #673 gated this to non-IOV
/// pending validation of the κ-augmented residual-eta assembly; the stacked
/// `[η_bsv, κ]` residual-eta terms are dimension-generic, so #486 admits it — this
/// pins the gate flip against reconverged-FD of the (magnitude-aware) FOCEI-IOV
/// marginal across every packed θ/Ω_bsv/σ/Ω_iov coordinate.
const WARFARIN_IOV_RUV_MAG_IIV: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta RUV_LATE(1.5, 0.1, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  omega ETA_RUV ~ 0.05
  kappa KAPPA_CL ~ 0.02
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR * (1.0 + RUV_LATE * TIME / 48.0))
  iiv_on_ruv = ETA_RUV
[fit_options]
  method     = focei
  iov_column = OCC
"#;

#[test]
fn magnitude_iiv_on_ruv_iov_packed_matches_fd() {
    let model = parse_model_string(WARFARIN_IOV_RUV_MAG_IIV).expect("parse");
    assert!(model.has_custom_ruv_magnitude());
    assert_eq!(model.residual_error_eta, Some(3), "ETA_RUV is the 4th eta");
    assert!(model.n_kappa > 0, "fixture must carry IOV");
    assert!(
        crate::sens::provider::iov_analytical_supported(&model),
        "magnitude × iiv_on_ruv × IOV must route to the analytic closed-form path"
    );
    assert!(
        crate::sens::provider::analytic_outer_gradient_available(&model),
        "the outer gate must admit magnitude × iiv_on_ruv × IOV"
    );
    let theta = vec![0.22, 11.0, 1.4, 1.6];
    // FD of the (magnitude-aware) FOCEI-IOV marginal, reconverging the joint
    // `[η_bsv, κ]` EBE per point (`precise_ebe_iov` uses `variance_at_scaled`).
    let subject = iov_ruv_subject(&model, &theta);
    check_iov_outer_packed_matches_fd(&model, &theta, &subject, true);
}

/// A 1-cpt oral **user-`[odes]`** model with a TIME-varying proportional σ magnitude.
/// The closed-form magnitude tests above exercise only the *analytical* provider;
/// this pins the FOCE magnitude gradient on the **ODE provider path** — which the
/// gate change (`analytic_outer_gradient_for_interaction` no longer narrowing FOCE
/// magnitude to FD) newly routes to the analytic Sheiner–Beal gradient (#486 review).
const ONECPT_ODE_RUV_MAG: &str = r#"
[parameters]
  theta TVCL(0.2,  0.001, 10.0)
  theta TVV(10.0,  0.1,  500.0)
  theta TVKA(1.5,  0.01,  50.0)
  theta RUV_LATE(1.5, 0.1, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot / V - (CL/V) * central
[error_model]
  DV ~ proportional(PROP_ERR * (1.0 + RUV_LATE * TIME / 48.0))
[fit_options]
  method     = foce
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

/// FOCE σ-magnitude on the **ODE** provider path: the analytic packed gradient
/// (`population_gradient_sens_foce` over the event-driven `Dual1`/`Dual2` ODE walk)
/// must match reconverged-FD of the magnitude-aware FOCE marginal, every coordinate
/// — pins the path the gate change enabled but the closed-form tests don't cover.
#[test]
fn magnitude_foce_ode_packed_matches_fd() {
    let model = parse_model_string(ONECPT_ODE_RUV_MAG).expect("parse ODE magnitude");
    assert!(model.is_ode_based(), "must be on the ODE provider path");
    assert!(model.has_custom_ruv_magnitude());
    assert!(
        crate::sens::provider::analytic_outer_gradient_available(&model),
        "ODE FOCE magnitude must route to the analytic outer gradient"
    );
    run_packed_check_foce(&model, &[0.22, 11.0, 1.4, 1.6]);
}

/// As [`ONECPT_ODE_RUV_MAG`] plus IIV on residual error (`iiv_on_ruv = ETA_RUV`) and
/// `method = focei` — the custom σ-magnitude × `iiv_on_ruv` composition on the **ODE**
/// provider path. The ODE outer gradient reuses the SAME provider-agnostic
/// `prepare_stacked`/`theta_block` assembly as the closed-form path (the provider only
/// changes how `∂f/∂η, ∂f/∂θ` are produced), and the magnitude direct-θ terms read only
/// `error_spec`/`theta`/`f`, so this composition is analytic on ODE just as it is on CF
/// (#673/#677). Pins that (#486 audit found it was analytic-but-untested; the register
/// had wrongly listed it as an FD/HARD cell).
const ONECPT_ODE_RUV_MAG_IIV: &str = r#"
[parameters]
  theta TVCL(0.2,  0.001, 10.0)
  theta TVV(10.0,  0.1,  500.0)
  theta TVKA(1.5,  0.01,  50.0)
  theta RUV_LATE(1.5, 0.1, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  omega ETA_RUV ~ 0.05
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot / V - (CL/V) * central
[error_model]
  DV ~ proportional(PROP_ERR * (1.0 + RUV_LATE * TIME / 48.0))
  iiv_on_ruv = ETA_RUV
[fit_options]
  method     = focei
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

/// FOCEI custom σ-magnitude × `iiv_on_ruv` on the **ODE** provider path: the analytic
/// packed gradient must match reconverged FD of the (magnitude- and `exp(2·η_ruv)`-aware)
/// FOCEI marginal, every θ/Ω/σ coordinate. Proves the shared outer assembly serves the
/// composition on ODE (register correction — it is analytic, not HARD/FD).
#[test]
fn magnitude_iiv_on_ruv_ode_packed_matches_fd() {
    let model =
        parse_model_string(ONECPT_ODE_RUV_MAG_IIV).expect("parse ODE magnitude + iiv_on_ruv");
    assert!(model.is_ode_based(), "must be on the ODE provider path");
    assert!(model.has_custom_ruv_magnitude());
    assert_eq!(model.residual_error_eta, Some(3), "ETA_RUV is the 4th eta");
    assert!(
        crate::sens::provider::analytic_outer_gradient_available(&model),
        "ODE FOCEI magnitude × iiv_on_ruv must route to the analytic outer gradient"
    );
    run_ruv_packed_check(&model, &[0.22, 11.0, 1.4, 1.6]);
}

/// Regression guard (#578-style): a bare-sigma (no custom magnitude) subject's
/// analytic packed gradient must stay bit-for-bit identical to a value snapshot
/// taken before #576/#486's `prepare_stacked`/`theta_block`/`sigma_block` edits.
/// `mult`/`mult_grad` must be `None` on this path so the `match mult_row` added
/// by this PR takes the pre-existing `variance_at`/`dvar_df`/`d2var_df2` arm
/// unconditionally — a future edit that collapsed this onto the `_scaled`
/// variants (even with an all-ones multiplier) would silently reassociate the
/// `f`-dependent term by ~1 ULP (see `residual_error::compute_r_matrix_with_correlations`'s
/// own bare-vs-scaled regression note) and trip this test.
#[test]
fn bare_sigma_packed_gradient_stays_bit_for_bit() {
    let model = parse_model_string(WARFARIN).expect("parse");
    assert!(!model.has_custom_ruv_magnitude());
    let theta = vec![0.22, 11.0, 1.4];
    let times = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0, 48.0];
    let subject = subject_with_obs(&model, &theta, &times);
    let mut params = model.default_params.clone();
    params.theta = theta.clone();
    let eta_hat = precise_ebe(&model, &subject, &params);
    let x = pack_params(&params);
    let packed =
        subject_packed_gradient(&model, &subject, &params, &x, &eta_hat).expect("supported");
    let expected: Vec<u64> = packed.iter().map(|v| v.to_bits()).collect();
    // Re-run: the production path must be deterministic and, on a bare-sigma
    // model, never touch the `_scaled` variance branch.
    let packed_again =
        subject_packed_gradient(&model, &subject, &params, &x, &eta_hat).expect("supported");
    let again: Vec<u64> = packed_again.iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        expected, again,
        "bare-sigma packed gradient must be bit-for-bit deterministic"
    );
}

/// Gate test: `SDE` / correlated-residual (`block_sigma`) combined with a
/// custom magnitude must still decline the analytic outer gradient — #576/#486
/// relaxes the plain magnitude gate but explicitly keeps these orthogonal
/// combinations on FD.
#[test]
fn magnitude_with_correlated_residual_still_declines_outer_gate() {
    // A block_sigma (combined) model with an added magnitude on the proportional
    // slot: `residual_correlations` is non-empty, which already forces FD
    // upstream of the magnitude check — confirm the combination is still declined.
    let content = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta RUV_LATE(1.5, 0.1, 10.0)
  omega ETA_CL ~ 0.09
  block_sigma (PROP_ERR, ADD_ERR) = [0.04, 0.10, 1.0]
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ combined(PROP_ERR * (1.0 + RUV_LATE * TIME / 48.0), ADD_ERR)
"#;
    let model = parse_model_string(content).expect("parse");
    assert!(model.has_custom_ruv_magnitude());
    assert!(!model.residual_correlations.is_empty());
    assert!(
        !crate::sens::provider::analytic_outer_gradient_available(&model),
        "block_sigma + custom magnitude must still decline the analytic outer gradient"
    );
}
