//! The adjoint gradient, covariance Hessian and exact η-quantities against external
//! oracles: Richardson differences of the production objective with the EBE re-solved at
//! every point (which shares no derivative code with the adjoint), and of the production
//! predictor.

use super::*;
use crate::estimation::inner_optimizer::find_ebe;
use crate::estimation::parameterization::pack_params;
use crate::parser::model_parser::parse_model_string;
use crate::types::{DoseEvent, Subject};
use std::collections::HashMap;

pub(super) fn subject(
    model: &CompiledModel,
    doses: Vec<DoseEvent>,
    times: &[f64],
    eta_ref: &[f64],
) -> Subject {
    let n = times.len();
    let mut s = Subject {
        obs_rgroup: Vec::new(),
        id: "1".to_string(),
        doses,
        obs_times: times.to_vec(),
        obs_raw_times: Vec::new(),
        observations: vec![0.0; n],
        obs_cmts: vec![1; n],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; n],
        occasions: vec![1; n],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
        endpoint_covariates: Vec::new(),
        filled_cells: Vec::new(),
        simulated_mixnum: None,
    };
    let theta = &model.default_params.theta;
    let preds = crate::pk::compute_predictions_with_tv(model, &s, theta, eta_ref);
    // Non-trivial residuals of both signs, so every term of Ψ is live.
    s.observations = preds
        .iter()
        .enumerate()
        .map(|(j, p)| p * (1.0 + 0.15 * (1.3 * j as f64).sin()))
        .collect();
    if model.log_transform {
        // LTBS compares log predictions with logged DV (the data reader's convention).
        s.observations = s
            .observations
            .iter()
            .map(|v| crate::pk::ltbs_log_g(*v))
            .collect();
    }
    s
}

/// Parameters moved off the defaults, so the gradient is far from zero in every block.
pub(super) fn moved_params(model: &CompiledModel) -> ModelParameters {
    let mut p = model.default_params.clone();
    for (m, t) in p.theta.iter_mut().enumerate() {
        *t *= if m % 2 == 0 { 1.12 } else { 0.91 };
    }
    p
}

/// The worst relative gap `|adj − fd| / (1 + |fd|)` over all packed coordinates between the
/// FOCEI adjoint gradient and the reconverged-FD oracle ([`fd_oracle`]).
pub(super) fn parity(model: &CompiledModel, s: &Subject) -> f64 {
    parity_for(model, s, true)
}

/// [`parity`] for FOCE (no interaction).
fn parity_foce(model: &CompiledModel, s: &Subject) -> f64 {
    parity_for(model, s, false)
}

/// The EBE the fit would use: FOCE's under its frozen-variance convention
/// (`EbeVariance::Population`), FOCEI's conditional one.
pub(super) fn ebe_for(
    model: &CompiledModel,
    s: &Subject,
    params: &ModelParameters,
    tol: f64,
    interaction: bool,
) -> nalgebra::DVector<f64> {
    use crate::estimation::inner_optimizer::{find_ebe_with_variance, EbeVariance};
    let v = EbeVariance::for_stage(crate::types::EstimationMethod::Foce, interaction, 0, model);
    let r = find_ebe_with_variance(model, s, params, 500, tol, None, None, 0, v);
    // Under IOV the gradient takes the stacked `[η, κ_0, …, κ_{K−1}]`.
    let mut stacked: Vec<f64> = r.eta.iter().copied().collect();
    for k in &r.kappas {
        stacked.extend(k.iter());
    }
    nalgebra::DVector::from_vec(stacked)
}

fn parity_for(model: &CompiledModel, s: &Subject, interaction: bool) -> f64 {
    fd_oracle(model, s, interaction)
}

const ONE_CPT_ORAL_ODE: &str = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(30.0, 1.0, 500.0)
  theta TVKA(1.2, 0.05, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.06
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV * exp(ETA_V)
  KA = TVKA
[structural_model]
  ode(states=[depot, central])
[odes]
  d/dt(depot) = -KA * depot
  d/dt(central) = KA * depot - (CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

const TWO_CPT_INF_COMBINED_BLOCK: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV1(20.0, 1.0, 500.0)
  theta TVQ(3.0, 0.1, 50.0)
  theta TVV2(40.0, 1.0, 500.0)
  block_omega (ETA_CL, ETA_V1) = [0.09, 0.03, 0.06]
  omega ETA_V2 ~ 0.05
  sigma PROP_ERR ~ 0.1 (sd)
  sigma ADD_ERR ~ 0.05 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q = TVQ
  V2 = TVV2 * exp(ETA_V2)
[structural_model]
  ode(states=[central, peripheral])
[odes]
  d/dt(central) = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) = (Q/V1) * central - (Q/V2) * peripheral
[scaling]
  y = central / V1
[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR)
"#;

const ADDITIVE_FOUR_ETA: &str = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(30.0, 1.0, 500.0)
  theta TVKA(1.2, 0.05, 20.0)
  theta TVKE0(0.3, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.06
  omega ETA_KA ~ 0.2
  omega ETA_KE0 ~ 0.1
  sigma ADD_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
  KE0 = TVKE0 * exp(ETA_KE0)
[structural_model]
  ode(states=[depot, central, effect])
[odes]
  d/dt(depot) = -KA * depot
  d/dt(central) = KA * depot - (CL/V) * central
  d/dt(effect) = KE0 * (central / V - effect)
[scaling]
  y = effect
[error_model]
  DV ~ additive(ADD_ERR)
"#;

/// A linear chain with `k` transit rates, each its own individual parameter:
/// `k + 2` individual parameters, IIV on three.
fn chain_model(k: usize) -> String {
    let mut s = String::from(
        "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  theta TVV(30.0, 1.0, 500.0)\n",
    );
    for i in 1..=k {
        s += &format!("  theta TVK{i}({:.2}, 0.01, 50.0)\n", 1.0 + 0.4 * i as f64);
    }
    s += "  omega ETA_CL ~ 0.1\n  omega ETA_V ~ 0.1\n  omega ETA_K1 ~ 0.1\n  sigma PROP ~ 0.1 (sd)\n";
    s += "[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n  K1 = TVK1 * exp(ETA_K1)\n";
    for i in 2..=k {
        s += &format!("  K{i} = TVK{i}\n");
    }
    let states: Vec<String> = (1..=k)
        .map(|i| format!("a{i}"))
        .chain(std::iter::once("central".into()))
        .collect();
    s += &format!(
        "[structural_model]\n  ode(states=[{}])\n[odes]\n  d/dt(a1) = -K1 * a1\n",
        states.join(", ")
    );
    for i in 2..=k {
        s += &format!("  d/dt(a{i}) = K{} * a{} - K{i} * a{i}\n", i - 1, i - 1);
    }
    s += &format!("  d/dt(central) = K{k} * a{k} - (CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP)\n");
    s
}

pub(super) fn with_tol(mut model: CompiledModel, reltol: f64, abstol: f64) -> CompiledModel {
    if let Some(ode) = model.ode_spec.as_mut() {
        ode.solver_opts.reltol = reltol;
        ode.solver_opts.abstol = abstol;
    }
    model
}

/// Both gradients differentiate the same discrete trajectory, so at a tolerance where
/// that trajectory is converged they must agree to round-off.
pub(super) fn tight(model: CompiledModel) -> CompiledModel {
    with_tol(model, 1e-10, 1e-12)
}

const TIMES: [f64; 9] = [0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0];

#[test]
fn adjoint_matches_reconverged_fd_one_cpt_oral_proportional() {
    let model = tight(parse_model_string(ONE_CPT_ORAL_ODE).unwrap());
    let s = subject(
        &model,
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &TIMES,
        &[0.1, -0.05],
    );
    let worst = parity(&model, &s);
    // Measured 2.0e-7 vs reconverged FD (Richardson, EBE re-solved per point); 5× headroom.
    assert!(worst < 1e-6, "worst relative difference {worst:.2e}");
}

#[test]
fn adjoint_matches_reconverged_fd_two_cpt_infusion_combined_block_omega_multidose() {
    // At reltol 1e-12, so the integrator's error sits well below the oracle's.
    let model = with_tol(
        parse_model_string(TWO_CPT_INF_COMBINED_BLOCK).unwrap(),
        1e-12,
        1e-14,
    );
    // Three infusions, the later two landing on residual drug.
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0),
        DoseEvent::new(8.0, 100.0, 1, 50.0, false, 0.0),
        DoseEvent::new(16.0, 100.0, 1, 50.0, false, 0.0),
    ];
    let times = [1.0, 2.0, 4.0, 7.9, 9.0, 12.0, 17.0, 20.0, 24.0, 30.0];
    let s = subject(&model, doses, &times, &[0.1, -0.05, 0.08]);
    let worst = parity(&model, &s);
    // Measured 2.5e-7 vs reconverged FD (Richardson, EBE re-solved per point); 5× headroom.
    assert!(worst < 1.3e-6, "worst relative difference {worst:.2e}");
}

#[test]
fn adjoint_matches_reconverged_fd_additive_four_eta() {
    let model = tight(parse_model_string(ADDITIVE_FOUR_ETA).unwrap());
    let s = subject(
        &model,
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &TIMES,
        &[0.1, -0.05, 0.2, -0.1],
    );
    let worst = parity(&model, &s);
    // Measured 2.9e-7 vs reconverged FD (Richardson, EBE re-solved per point); 5× headroom.
    assert!(worst < 1.5e-6, "worst relative difference {worst:.2e}");
}

#[test]
fn adjoint_matches_reconverged_fd_past_twelve_individual_parameters() {
    // 14 individual parameters: wider than the retired forward provider's static-walk
    // tables, which the adjoint (seeding only η) never had.
    let model = tight(parse_model_string(&chain_model(12)).unwrap());
    assert!(
        model.pk_indices.len() > 12,
        "wider than any per-parameter dual table"
    );
    let s = subject(
        &model,
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &TIMES,
        &[0.1, -0.05, 0.1],
    );
    let worst = parity(&model, &s);
    assert!(worst < 1e-6, "worst relative difference {worst:.2e}");
}

#[test]
fn adjoint_matches_reconverged_fd_on_the_rosenbrock_lu_path() {
    let mut model = tight(parse_model_string(TWO_CPT_INF_COMBINED_BLOCK).unwrap());
    model.ode_spec.as_mut().unwrap().solver_opts.method = crate::ode::solver::OdeMethod::Rodas5P;
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let s = subject(&model, doses, &TIMES, &[0.1, -0.05, 0.08]);
    let worst = parity(&model, &s);
    // Measured 2.0e-6 vs reconverged FD (Richardson, EBE re-solved per point); 5× headroom.
    assert!(worst < 1e-5, "worst relative difference {worst:.2e}");
}

#[test]
fn production_gradient_routes_in_scope_subjects_to_the_adjoint() {
    let model = parse_model_string(ONE_CPT_ORAL_ODE).unwrap();
    let s = subject(
        &model,
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &TIMES,
        &[0.1, -0.05],
    );
    let params = moved_params(&model);
    let x = pack_params(&params);
    let eta = find_ebe(&model, &s, &params, 500, 1e-11, None, None, 0).eta;
    let via_public = crate::estimation::sens_outer_gradient::subject_packed_gradient(
        &model,
        &s,
        &params,
        &x,
        eta.as_slice(),
    )
    .unwrap();
    let adj =
        subject_adjoint_packed_gradient(&model, &s, &params, &x, eta.as_slice(), true).unwrap();
    for k in 0..x.len() {
        assert_eq!(
            via_public[k].to_bits(),
            adj[k].to_bits(),
            "x[{k}] did not come from the adjoint"
        );
    }
}

/// Whether the adjoint serves `s` at the model defaults, evaluated at the EBE (where `H`
/// is positive definite; the adjoint declines an indefinite `H` rather than return an
/// unreliable gradient).
fn in_scope(model: &CompiledModel, s: &Subject) -> bool {
    let p = model.default_params.clone();
    let x = pack_params(&p);
    let eta = find_ebe(model, s, &p, 500, 1e-11, None, None, 0).eta;
    subject_adjoint_packed_gradient(model, s, &p, &x, eta.as_slice(), true).is_some()
}

#[test]
fn the_in_scope_fixture_is_served() {
    let base = parse_model_string(ONE_CPT_ORAL_ODE).unwrap();
    let s = subject(
        &base,
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &TIMES,
        &[0.0, 0.0],
    );
    assert!(in_scope(&base, &s));
}

/// Reconverged FOCEI subject objective `Fᵢ(x)`, with the EBE re-solved at every `x`. This
/// external oracle shares no derivative code with either gradient.
fn reconverged_f(model: &CompiledModel, s: &Subject, template: &ModelParameters, x: &[f64]) -> f64 {
    reconverged_f_for(model, s, template, x, true)
}

/// `∂f/∂η` for the oracle objective: a Richardson difference of the production predictor,
/// so the oracle never depends on the engine under test.
pub(super) fn oracle_jacobian(
    model: &CompiledModel,
    s: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> nalgebra::DMatrix<f64> {
    let n_obs = s.obs_times.len();
    let mut h = nalgebra::DMatrix::zeros(n_obs, model.n_eta);
    for k in 0..model.n_eta {
        let d = |st: f64| {
            let (mut ep, mut em) = (eta.to_vec(), eta.to_vec());
            ep[k] += st;
            em[k] -= st;
            let (yp, ym) = (
                crate::pk::compute_predictions_with_tv(model, s, theta, &ep),
                crate::pk::compute_predictions_with_tv(model, s, theta, &em),
            );
            yp.iter()
                .zip(&ym)
                .map(|(a, b)| (a - b) / (2.0 * st))
                .collect::<Vec<f64>>()
        };
        let (d1, d2) = (d(1e-4), d(2e-4));
        for j in 0..n_obs {
            h[(j, k)] = (4.0 * d1[j] - d2[j]) / 3.0;
        }
    }
    // FREM pseudo-observation rows carry their exact `{0, 1}` Jacobian, as production's
    // H-matrix does on every route.
    crate::estimation::inner_optimizer::overwrite_frem_pseudo_obs_rows(
        &mut h,
        model,
        s,
        model.n_eta,
    );
    h
}

/// The subject objective with the EBE re-found at `x`: the quantity both gradients
/// differentiate, as an independent oracle (`interaction = false` is FOCE).
fn reconverged_f_for(
    model: &CompiledModel,
    s: &Subject,
    template: &ModelParameters,
    x: &[f64],
    interaction: bool,
) -> f64 {
    let params = crate::estimation::parameterization::unpack_params(x, template);
    let eta = ebe_for(model, s, &params, ORACLE_EBE_TOL.get(), interaction);
    if model.n_kappa > 0 {
        // IOV: the BSV H-matrix at the stacked mode, κ̂ per occasion group.
        let n = model.n_eta;
        let jac = exact_eta_jacobian(model, s, &params.theta, eta.as_slice())
            .expect("stacked η-Jacobian");
        let hm = jac.columns(0, n).into_owned();
        let kap: Vec<nalgebra::DVector<f64>> = eta.as_slice()[n..]
            .chunks(model.n_kappa)
            .map(nalgebra::DVector::from_column_slice)
            .collect();
        // The fit's objective at the live `block_sigma` ρ (#979), as below.
        return crate::stats::likelihood::foce_subject_nll_iov_cov(
            model,
            s,
            &params.theta,
            &nalgebra::DVector::from_column_slice(&eta.as_slice()[..n]),
            &hm,
            &params.omega,
            &params.sigma.values,
            model.residual_cov_at(&params),
            interaction,
            &kap,
            params.omega_iov.as_ref().unwrap(),
        );
    }
    let hm = oracle_jacobian(model, s, &params.theta, eta.as_slice());
    // The fit's objective over the full residual structure (the public `foce_subject_nll`
    // wraps `block_sigma` pairs only and would drop a serial correlation).
    crate::stats::likelihood::foce_subject_nll_cov(
        model,
        s,
        &params.theta,
        &eta,
        &hm,
        &params.omega,
        &params.sigma.values,
        model.residual_cov_with(&params.residual_correlations, &params.theta),
        interaction,
    )
}

#[test]
fn adjoint_matches_reconverged_finite_differences_of_the_objective() {
    let model = with_tol(parse_model_string(ONE_CPT_ORAL_ODE).unwrap(), 1e-12, 1e-14);
    let s = subject(
        &model,
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &TIMES,
        &[0.1, -0.05],
    );
    let worst = fd_oracle(&model, &s, true);
    // Measured worst 5.3e-7 (the Richardson FD noise floor at tol 1e-12); 10× headroom.
    assert!(worst < 5e-6, "worst relative difference {worst:.2e}");
}

thread_local! {
    /// The oracle's EBE tolerance (the base point's and every re-solve's).
    static ORACLE_EBE_TOL: std::cell::Cell<f64> = const { std::cell::Cell::new(1e-12) };
}

/// Worst relative gap `|adj − FD|/(1 + |FD|)` between the adjoint gradient and a
/// Richardson-extrapolated central difference of the objective with the EBE re-found at
/// every point: an oracle that shares no derivative code with either provider.
pub(super) fn fd_oracle(model: &CompiledModel, s: &Subject, interaction: bool) -> f64 {
    let params = moved_params(model);
    let x = pack_params(&params);
    let eta = ebe_for(model, s, &params, ORACLE_EBE_TOL.get(), interaction);
    let adj = subject_adjoint_packed_gradient(model, s, &params, &x, eta.as_slice(), interaction)
        .expect("subject must be in adjoint scope");
    let mut worst = 0.0f64;
    for k in 0..x.len() {
        let fd = |h: f64| {
            let mut xp = x.clone();
            xp[k] += h;
            let mut xm = x.clone();
            xm[k] -= h;
            (reconverged_f_for(model, s, &params, &xp, interaction)
                - reconverged_f_for(model, s, &params, &xm, interaction))
                / (2.0 * h)
        };
        let rich = (4.0 * fd(1e-3) - fd(2e-3)) / 3.0;
        assert!(
            rich.is_finite() && adj[k].is_finite(),
            "x[{k}] adj={} fd={rich}",
            adj[k]
        );
        let rel = (adj[k] - rich).abs() / (1.0 + rich.abs());
        eprintln!(
            "x[{k}] adjoint={:.10e} reconverged-FD={rich:.10e} rel={rel:.2e}",
            adj[k]
        );
        worst = worst.max(rel);
    }
    assert!(
        adj.iter().map(|v| v.abs()).fold(0.0, f64::max) > 1e-2,
        "gradient too small to test"
    );
    worst
}

// ---- closed-form (analytical) engine --------------------------------------------------

const ONE_CPT_ORAL: &str = r#"
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

const TWO_CPT_IV_COMBINED: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV1(20.0, 1.0, 500.0)
  theta TVQ(3.0, 0.1, 50.0)
  theta TVV2(40.0, 1.0, 500.0)
  block_omega (ETA_CL, ETA_V1) = [0.09, 0.03, 0.06]
  sigma PROP_ERR ~ 0.1 (sd)
  sigma ADD_ERR ~ 0.05 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q = TVQ
  V2 = TVV2
[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)
[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR)
"#;

const ONE_CPT_ORAL_LAG_LTBS: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta TVLAG(0.4, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_LAG ~ 0.1
  sigma ADD_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA
  ALAG = TVLAG * exp(ETA_LAG)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, lagtime=ALAG)
[error_model]
  log(DV) ~ additive(ADD_ERR)
"#;

#[test]
fn analytic_adjoint_matches_reconverged_fd_one_cpt_oral_multidose() {
    let model = parse_model_string(ONE_CPT_ORAL).unwrap();
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.5, 14.0, 18.0, 24.0];
    let s = subject(&model, doses, &times, &[0.1, -0.05, 0.2]);
    let worst = parity(&model, &s);
    // Measured 1.2e-7 vs reconverged FD (Richardson, EBE re-solved per point); 5× headroom.
    assert!(worst < 6e-7, "worst relative difference {worst:.2e}");
}

#[test]
fn analytic_adjoint_matches_reconverged_fd_two_cpt_iv_infusion_combined_block() {
    let model = parse_model_string(TWO_CPT_IV_COMBINED).unwrap();
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0),
        DoseEvent::new(8.0, 100.0, 1, 50.0, false, 0.0),
    ];
    let times = [1.0, 2.0, 4.0, 7.9, 9.0, 12.0, 18.0, 24.0];
    let s = subject(&model, doses, &times, &[0.1, -0.05]);
    let worst = parity(&model, &s);
    // Measured 2.2e-8 vs reconverged FD (Richardson, EBE re-solved per point); 5× headroom.
    assert!(worst < 1.2e-7, "worst relative difference {worst:.2e}");
}

#[test]
fn analytic_adjoint_matches_reconverged_fd_lagtime_ltbs() {
    let model = parse_model_string(ONE_CPT_ORAL_LAG_LTBS).unwrap();
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
    ];
    // Every observation well after the lagged arrival: a pre-arrival row sits on the LTBS
    // floor, where the objective has a kink and no gradient is meaningful (measured: the
    // reconverged FD there is O(10³) off both analytic gradients).
    let times = [1.5, 2.0, 3.0, 4.0, 8.0, 13.5, 14.0, 18.0, 24.0];
    let s = subject(&model, doses, &times, &[0.1, -0.05, 0.2]);
    assert!(
        in_scope(&model, &s),
        "lagtime + LTBS must be served by the closed-form engine"
    );
    let worst = parity(&model, &s);
    // Measured 1.5e-6 vs reconverged FD (Richardson, EBE re-solved per point); 5× headroom.
    assert!(worst < 8e-6, "worst relative difference {worst:.2e}");
}

#[test]
fn analytic_adjoint_matches_reconverged_fd_steady_state() {
    let model = parse_model_string(ONE_CPT_ORAL).unwrap();
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.5, 14.0, 18.0, 24.0];
    let s = subject(&model, doses, &times, &[0.1, -0.05, 0.2]);
    assert!(
        in_scope(&model, &s),
        "steady state must be served by the closed-form engine"
    );
    let worst = parity(&model, &s);
    // Measured 1.8e-8 vs reconverged FD (Richardson, EBE re-solved per point); 5× headroom.
    assert!(worst < 1e-7, "worst relative difference {worst:.2e}");
}

// ---- likelihood kernel: M3 censoring and IIV-on-RUV ------------------------------------

/// Censor the rows in `rows` (left, `CENS=1`) at a limit above the prediction, so the M3
/// term is live and non-trivial.
pub(super) fn censor(mut s: Subject, rows: &[usize]) -> Subject {
    for &j in rows {
        s.cens[j] = 1;
        s.observations[j] *= 1.6;
    }
    s
}

#[test]
fn kernel_m3_censored_rows_match_reconverged_fd() {
    let mut model = parse_model_string(ONE_CPT_ORAL).unwrap();
    model.bloq_method = crate::types::BloqMethod::M3;
    let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    let s = censor(
        subject(&model, doses, &TIMES, &[0.1, -0.05, 0.2]),
        &[0, 7, 8],
    );
    assert!(in_scope(&model, &s), "M3 rows must be served by the kernel");

    let params = moved_params(&model);
    let x = pack_params(&params);
    let eta = find_ebe(&model, &s, &params, 500, 1e-12, None, None, 0).eta;
    let adj =
        subject_adjoint_packed_gradient(&model, &s, &params, &x, eta.as_slice(), true).unwrap();
    let mut worst_fd = 0.0f64;
    for k in 0..x.len() {
        let fd = |h: f64| {
            let mut xp = x.clone();
            xp[k] += h;
            let mut xm = x.clone();
            xm[k] -= h;
            (reconverged_f(&model, &s, &params, &xp) - reconverged_f(&model, &s, &params, &xm))
                / (2.0 * h)
        };
        let rich = (4.0 * fd(1e-3) - fd(2e-3)) / 3.0;
        let rel = (adj[k] - rich).abs() / (1.0 + rich.abs());
        eprintln!(
            "M3 x[{k}] adjoint={:.10e} reconverged-FD={rich:.10e} rel={rel:.2e}",
            adj[k]
        );
        worst_fd = worst_fd.max(rel);
    }
    assert!(
        worst_fd < 1e-5,
        "worst relative difference vs reconverged FD {worst_fd:.2e}"
    );
    let worst = parity(&model, &s);
    assert!(worst < 3e-7, "worst relative difference {worst:.2e}");
}

const TWO_CPT_ODE_RUV: &str = r#"
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
"#;

#[test]
fn kernel_iiv_on_ruv_matches_reconverged_fd() {
    let model = tight(parse_model_string(TWO_CPT_ODE_RUV).unwrap());
    assert!(model.residual_error_eta.is_some());
    let s = subject(
        &model,
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &TIMES,
        &[0.1, -0.05, 0.0],
    );
    assert!(in_scope(&model, &s));
    let worst = parity(&model, &s);
    // Measured 1.4e-7 vs reconverged FD (Richardson, EBE re-solved per point); 5× headroom.
    assert!(worst < 7e-7, "worst relative difference {worst:.2e}");
}

#[test]
fn kernel_iiv_on_ruv_with_m3_rows_matches_reconverged_fd() {
    let mut model = tight(parse_model_string(TWO_CPT_ODE_RUV).unwrap());
    model.bloq_method = crate::types::BloqMethod::M3;
    let s = censor(
        subject(
            &model,
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            &TIMES,
            &[0.1, -0.05, 0.0],
        ),
        &[0, 8],
    );
    assert!(in_scope(&model, &s));
    let worst = parity(&model, &s);
    // Measured 1.4e-6 vs reconverged FD (Richardson, EBE re-solved per point); 5× headroom.
    assert!(worst < 7e-6, "worst relative difference {worst:.2e}");
}

// ---- ODE engine: init(...), lag (event walk), time-varying covariates, steady state ----

const ODE_INIT: &str = r#"
[parameters]
  theta TVCL(0.4, 0.01, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVBASE(2.0, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_BASE ~ 0.04
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  BASE = TVBASE * exp(ETA_BASE)
[structural_model]
  ode(states=[central])
[odes]
  init(central) = BASE * V
  d/dt(central) = -(CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

const ODE_LAG: &str = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(30.0, 1.0, 500.0)
  theta TVKA(1.2, 0.05, 20.0)
  theta TVLAG(0.4, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_LAG ~ 0.1
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  KA = TVKA
  LAGTIME = TVLAG * exp(ETA_LAG)
[structural_model]
  ode(states=[depot, central])
[odes]
  d/dt(depot) = -KA * depot
  d/dt(central) = KA * depot - (CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

const ODE_TVCOV: &str = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(30.0, 1.0, 500.0)
  theta TVKA(1.2, 0.05, 20.0)
  theta CL_WT(0.75, -2.0, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.06
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * (WT / 70.0)^CL_WT * exp(ETA_CL)
  V = TVV * exp(ETA_V)
  KA = TVKA
[structural_model]
  ode(states=[depot, central])
[odes]
  d/dt(depot) = -KA * depot
  d/dt(central) = KA * depot - (CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

#[test]
fn ode_init_on_the_static_walk_matches_reconverged_fd() {
    let model = tight(parse_model_string(ODE_INIT).unwrap());
    assert!(model.ode_spec.as_ref().unwrap().init_fn.is_some());
    let s = subject(
        &model,
        vec![DoseEvent::new(6.0, 100.0, 1, 0.0, false, 0.0)],
        &TIMES,
        &[0.1, -0.05],
    );
    assert!(in_scope(&model, &s), "init(...) must be served");
    let worst = parity(&model, &s);
    assert!(worst < 1e-7, "worst relative difference {worst:.2e}");
}

#[test]
fn ode_lagtime_on_the_event_walk_matches_reconverged_fd() {
    let model = tight(parse_model_string(ODE_LAG).unwrap());
    assert!(model.has_lagtime());
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let times = [1.0, 2.0, 4.0, 8.0, 13.0, 14.0, 18.0, 24.0];
    let s = subject(&model, doses, &times, &[0.1, -0.05]);
    assert!(
        in_scope(&model, &s),
        "an estimated lag must be served by the event walk"
    );
    let worst = parity(&model, &s);
    // Measured 6.0e-7 vs reconverged FD (Richardson, EBE re-solved per point); 5× headroom.
    assert!(worst < 3e-6, "worst relative difference {worst:.2e}");
}

#[test]
fn ode_time_varying_covariate_on_the_event_walk_matches_reconverged_fd() {
    // At reltol 1e-8 (the adjoint converges smoothly in the tolerance: 1e-8 → 1e-12 moves
    // it 1.4e-10).
    let model = with_tol(parse_model_string(ODE_TVCOV).unwrap(), 1e-8, 1e-10);
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let times = [1.0, 2.0, 4.0, 8.0, 13.0, 14.0, 18.0, 24.0];
    let wt_at = |t: f64| 60.0 + 1.5 * t;
    let base = {
        let mut b = subject_shell(doses.clone(), &times);
        b.covariates.insert("WT".into(), 70.0);
        b
    };
    let mut s = base.clone();
    s.obs_covariates = times
        .iter()
        .map(|&t| HashMap::from([("WT".to_string(), wt_at(t))]))
        .collect();
    s.dose_covariates = doses
        .iter()
        .map(|d| HashMap::from([("WT".to_string(), wt_at(d.time))]))
        .collect();
    let preds = crate::pk::compute_predictions_with_tv(
        &model,
        &s,
        &model.default_params.theta,
        &[0.1, -0.05],
    );
    s.observations = preds
        .iter()
        .enumerate()
        .map(|(j, p)| p * (1.0 + 0.15 * (1.3 * j as f64).sin()))
        .collect();
    assert!(s.has_tv_covariates());
    assert!(
        in_scope(&model, &s),
        "time-varying covariates must be served by the event walk"
    );
    let worst = parity(&model, &s);
    // Measured 7.4e-7 vs reconverged FD (Richardson, EBE re-solved per point); 5× headroom.
    assert!(worst < 4e-6, "worst relative difference {worst:.2e}");
}

/// An estimated lag and a clearance that both read `X`, on the ODE and closed-form walks;
/// under `iov` the lag also carries a κ.
fn addl_lag_src(ode: bool, iov: bool) -> String {
    let structural = if ode {
        "  ode(states=[depot, central])\n[odes]\n  d/dt(depot) = -KA * depot\n  \
         d/dt(central) = KA * depot - (CL/V) * central\n[scaling]\n  y = central / V\n"
    } else {
        "  pk one_cpt_oral(cl=CL, v=V, ka=KA, lagtime=LAGTIME)\n"
    };
    let (kappa, kappa_term, fit) = if iov {
        (
            "  kappa KAPPA_LAG ~ 0.04\n",
            " + KAPPA_LAG",
            "[fit_options]\n  iov_column = OCC\n",
        )
    } else {
        ("", "", "")
    };
    format!(
        "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  theta TVV(30.0, 1.0, 500.0)\n  \
         theta TVKA(1.2, 0.05, 20.0)\n  theta TVLAG(0.4, 0.01, 5.0)\n  omega ETA_CL ~ 0.09\n  \
         omega ETA_LAG ~ 0.1\n{kappa}  sigma PROP_ERR ~ 0.1 (sd)\n[individual_parameters]\n  \
         CL = TVCL * sqrt(X) * exp(ETA_CL)\n  V = TVV\n  KA = TVKA\n  \
         LAGTIME = TVLAG * X * exp(ETA_LAG{kappa_term})\n[structural_model]\n{structural}\
         [error_model]\n  DV ~ proportional(PROP_ERR)\n{fit}"
    )
}

/// #1909: the dose at 12 is implied by the row at 0 (X = 1) and governed by the
/// observation at 13 (X = 2), so its clearance is the governing record's and its lag the
/// parent's — 0.4·e^η, not 0.8·e^η. The walk copies the lag's dual from the parent with its
/// value, so a gradient of the governing-record lag, or of a value the predictor never
/// used, both show as a gap to FD of the production objective. `tv = false` holds X at 1,
/// so that only an occasion separates the two records (the IOV walk's static branch);
/// under IOV the dose rows sit on occasions 1 and 2, the implied one on its governing
/// record's.
fn addl_implied_lag_subject(model: &CompiledModel, tv: bool) -> Subject {
    let mut implied = DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0);
    implied.implied = Some(crate::types::ImpliedDose {
        parent: 0,
        param_time: 13.0,
        gov_time: 13.0,
    });
    let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0), implied];
    let times = [1.0, 2.0, 4.0, 8.0, 13.0, 14.0, 18.0, 24.0];
    let x = |v: f64| HashMap::from([("X".to_string(), v)]);
    let mut s = subject_shell(doses, &times);
    s.covariates.insert("X".into(), 1.0);
    if tv {
        s.dose_covariates = vec![x(1.0), x(2.0)];
        s.obs_covariates = times
            .iter()
            .map(|&t| x(if t < 13.0 { 1.0 } else { 2.0 }))
            .collect();
    }
    if model.n_kappa > 0 {
        s.occasions = times
            .iter()
            .map(|&t| if t < 12.0 { 1 } else { 2 })
            .collect();
        s.dose_occasions = vec![1, 2];
    }
    let preds = crate::pk::compute_predictions_with_tv(
        model,
        &s,
        &model.default_params.theta,
        &[0.1, -0.05],
    );
    s.observations = preds
        .iter()
        .enumerate()
        .map(|(j, p)| p * (1.0 + 0.15 * (1.3 * j as f64).sin()))
        .collect();
    assert_eq!(s.implied_dose_parent(1), Some(0));
    assert_eq!(s.has_tv_covariates(), tv);
    // Served by the adjoint, at the stacked `[η, κ…]` under IOV.
    let p = model.default_params.clone();
    let stacked = ebe_for(model, &s, &p, 1e-11, true);
    assert!(
        subject_adjoint_packed_gradient(model, &s, &p, &pack_params(&p), stacked.as_slice(), true)
            .is_some(),
        "the adjoint must serve the fixture"
    );
    s
}

#[test]
fn an_addl_implied_dose_lag_from_its_parent_matches_reconverged_fd() {
    // (ODE, IOV, time-varying X, bound): one row per seeder that copies the lag — the
    // closed-form walk, `seed_tvcov_snapshots`, and both branches of `seed_iov_events`.
    // Measured vs reconverged FD (Richardson, EBE re-solved per point) at 5998edf2 on
    // Linux x86_64: 9.0e-8, 3.3e-9, 4.3e-8 and 5.2e-8. macOS arm64 agrees on the three ODE
    // rows and gives 6.5e-7 on the closed-form walk. Each bound is 5×
    // the larger of the two, so 37× the Linux value on the first row. Without the copy the
    // closed-form walk's gap is 3.0.
    let cases = [
        (false, false, true, 3.3e-6),
        (true, false, true, 1.7e-8),
        (true, true, true, 2.2e-7),
        (true, true, false, 2.6e-7),
    ];
    for (ode, iov, tv, bound) in cases {
        let label = format!("ode {ode}, iov {iov}, tv {tv}");
        let src = addl_lag_src(ode, iov);
        let model = if ode {
            with_tol(parse_model_string(&src).unwrap(), 1e-8, 1e-10)
        } else {
            parse_model_string(&src).unwrap()
        };
        let s = addl_implied_lag_subject(&model, tv);
        if !iov {
            // The copy is live: the parent's lag is half the governing record's.
            let lag = model.active_dose_attr_map().lag_slot(1);
            let mut dose = Vec::new();
            let theta = &model.default_params.theta;
            crate::pk::compute_dose_pk_params_into(&model, &s, theta, &[0.0, 0.0], &mut dose);
            assert_eq!(dose[1].values[lag], dose[0].values[lag], "{label}");
            let params = moved_params(&model);
            let (gf, gl) = objective_gaps::<2>(&model, &s, &params, &[0.1, -0.05]);
            assert!(
                gf < 1e-10 && gl < 1e-10,
                "{label}: kernel vs production {gf:.2e}, {gl:.2e}"
            );
        }
        let worst = parity(&model, &s);
        eprintln!("addl implied lag ({label}) vs reconverged FD {worst:.2e}");
        assert!(
            worst < bound,
            "{label}: worst relative difference {worst:.2e}"
        );
    }
}

#[test]
fn ode_steady_state_dose_matches_reconverged_fd() {
    let model = tight(parse_model_string(ONE_CPT_ORAL_ODE).unwrap());
    let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)];
    let times = [0.5, 1.0, 2.0, 4.0, 8.0, 11.0];
    let s = subject(&model, doses, &times, &[0.1, -0.05]);
    assert!(
        in_scope(&model, &s),
        "the dual walk equilibrates steady state"
    );
    let worst = parity(&model, &s);
    // Measured 1.9e-6 vs reconverged FD (Richardson, EBE re-solved per point); 5× headroom.
    assert!(worst < 1e-5, "worst relative difference {worst:.2e}");
}

/// A steady-state dose under an estimated lag, on both sides of `lag ≥ II` (#1121, #1311): past
/// the interval the walk clamps the seed phase, and the derivative has to follow it. The
/// forward provider's per-PR pin (`ode_provider_ss_lagtime_at_or_past_the_interval_matches_
/// production`) went with it in #2093; this is its adjoint successor, and what lets the two
/// `ss_lagtime_edge` NONMEM objective anchors stay nightly. Same model, same subject times,
/// only `II` moves: 4 h (lag 0.36 h < II) and 0.3 h (lag > II).
#[test]
fn ode_steady_state_under_a_lag_past_the_interval_matches_reconverged_fd() {
    let model = tight(parse_model_string(ODE_LAG).unwrap());
    let lag = moved_params(&model).theta[3];
    let times = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0];
    for (ii, label) in [(4.0, "lag < II"), (0.3, "lag ≥ II")] {
        assert_eq!(
            lag >= ii,
            label == "lag ≥ II",
            "{label}: the straddle (lag {lag})"
        );
        let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, ii)];
        let s = subject(&model, doses, &times, &[0.1, -0.05]);
        assert!(
            in_scope(&model, &s),
            "{label}: SS under a lag must be served"
        );
        let worst = parity(&model, &s);
        eprintln!("SS under a lag, {label}: worst {worst:.2e}");
        // Measured 1.9e-7 (lag < II) and 1.4e-8 (lag ≥ II) on Linux; ~10× headroom.
        assert!(
            worst < 2e-6,
            "{label}: worst relative difference {worst:.2e}"
        );
    }
}

/// A certified-linear `[odes]` model (#852): `ode_method = auto` resolves to the exact
/// matrix-exponential path, with an estimated lagtime and a peripheral compartment.
const TWOCPT_DEPOT_LINEAR_ODE: &str = r#"
[parameters]
  theta TVCL(4.0,  0.1, 100.0)
  theta TVV1(12.0, 1.0, 500.0)
  theta TVQ(2.0,   0.01, 100.0)
  theta TVV2(25.0, 1.0, 500.0)
  theta TVKA(1.3,  0.01, 50.0)
  theta TVLAG(0.4, 0.01, 5.0)
  omega ETA_CL ~ 0.15
  omega ETA_KA ~ 0.10
  omega ETA_LAG ~ 0.05
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1
  Q  = TVQ
  V2 = TVV2
  KA = TVKA * exp(ETA_KA)
  LAGTIME = TVLAG * exp(ETA_LAG)
[structural_model]
  ode(states=[depot, central, peripheral])
[odes]
  d/dt(depot)      = -KA * depot
  d/dt(central)    =  KA * depot - (CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  y = central / V1
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// The adjoint on the exact linear path (#852, Teun on #2093): its recordings run the
/// matrix exponential over the tape types, whose derivative self-check reads
/// `jet_max_abs`. Each subject's gradient must match reconverged FD, and the recording
/// must have taken the exact path: no integrator step and no declined segment. Its rk45
/// twin is the straddle — the same recording, stepped — so a fixture that silently
/// integrated would be seen.
#[test]
fn adjoint_on_the_exact_linear_path_matches_reconverged_fd() {
    use crate::ode::solver::SolverStatsScope;
    let model = parse_model_string(TWOCPT_DEPOT_LINEAR_ODE).unwrap();
    assert_eq!(
        model
            .ode_spec
            .as_ref()
            .unwrap()
            .effective_solver_opts()
            .method,
        crate::ode::OdeMethod::Linear,
        "the fixture must take the exact path"
    );
    let mut rk45 = parse_model_string(TWOCPT_DEPOT_LINEAR_ODE).unwrap();
    rk45.ode_spec.as_mut().unwrap().solver_opts.method = crate::ode::OdeMethod::Rk45;
    let eta_ref = [0.1, -0.05, 0.08];
    let multidose = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let infusion = vec![
        DoseEvent::new(0.0, 100.0, 2, 25.0, false, 0.0),
        DoseEvent::new(8.0, 100.0, 2, 25.0, false, 0.0),
    ];
    let steady = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)];
    for (label, doses, times) in [
        (
            "multidose",
            multidose,
            &[0.5, 1.0, 3.0, 12.6, 13.0, 18.0, 30.0][..],
        ),
        ("infusion", infusion, &[1.0, 3.0, 5.0, 9.0, 16.0][..]),
        ("steady state", steady, &[0.6, 2.0, 6.0, 11.5][..]),
    ] {
        let s = subject(&model, doses, times, &eta_ref);
        let steps = |m: &CompiledModel| {
            let params = moved_params(m);
            let x = pack_params(&params);
            let eta = ebe_for(m, &s, &params, 1e-10, true);
            let scope = SolverStatsScope::enter();
            subject_adjoint_packed_gradient(m, &s, &params, &x, eta.as_slice(), true)
                .expect("in scope");
            let st = scope.collected();
            (st.attempted_steps, st.linear_declined_segments)
        };
        assert_eq!(
            steps(&model),
            (0, 0),
            "{label}: the exact path stepped or declined"
        );
        assert!(steps(&rk45).0 > 0, "{label}: the rk45 twin must step");
        let worst = parity(&model, &s);
        eprintln!("linear exact path, {label}: worst {worst:.2e}");
        // Measured 2.3e-6 / 1.9e-6 / 5.3e-7 (multidose / infusion / SS) vs reconverged FD;
        // 4× headroom over the worst.
        assert!(
            worst < 1e-5,
            "{label}: worst relative difference {worst:.2e}"
        );
    }
}

/// A subject skeleton with the given doses and observation times (no DV yet).
pub(super) fn subject_shell(doses: Vec<DoseEvent>, times: &[f64]) -> Subject {
    let n = times.len();
    Subject {
        obs_rgroup: Vec::new(),
        id: "1".to_string(),
        doses,
        obs_times: times.to_vec(),
        obs_raw_times: Vec::new(),
        observations: vec![0.0; n],
        obs_cmts: vec![1; n],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; n],
        occasions: vec![1; n],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
        endpoint_covariates: Vec::new(),
        filled_cells: Vec::new(),
        simulated_mixnum: None,
    }
}

/// Dense residual covariance through the generic kernel: a serial `correlation` (ρ a θ,
/// entering `R` directly) and a `block_sigma` pair.
/// Same fixture as the pre-#2093 `dense_r_serial_correlation_gradient_matches_fd` (retired in
/// #2093).
fn dense_src(corr: &str, block: bool) -> String {
    let sig = if block {
        "  block_sigma (PROP_ERR, ADD_ERR) = [0.0225, 0.01, 0.09]\n"
    } else {
        "  sigma PROP_ERR ~ 0.15 (sd)\n  sigma ADD_ERR ~ 0.3 (sd)\n"
    };
    format!(
        "[parameters]\n  theta TVCL(1.0, 0.01, 10.0)\n  theta TVV(10.0, 0.1, 100.0)\n  \
         theta RHO(0.45, -0.4, 0.95)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n{sig}[individual_parameters]\n  \
         CL = TVCL * exp(ETA_CL)\n  V  = TVV * exp(ETA_V)\n[structural_model]\n  \
         pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ combined(PROP_ERR, ADD_ERR)\n{corr}"
    )
}

fn dense_subject(model: &CompiledModel) -> Subject {
    let times = [0.5, 1.5, 3.0, 6.5, 13.0, 14.5, 18.0, 23.5];
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 80.0, 1, 0.0, false, 0.0),
    ];
    let mut s = subject(model, doses, &times, &[0.1, -0.15]);
    s.obs_rgroup = vec![1, 1, 1, 1, 2, 2, 2, 2];
    s
}

#[test]
fn kernel_serial_residual_correlation_matches_reconverged_fd() {
    for kind in ["cs", "ar1", "car1"] {
        let model = parse_model_string(&dense_src(
            &format!("  correlation {kind}(RHO) by ARM\n"),
            false,
        ))
        .unwrap();
        assert!(model.residual_serial.is_some());
        let s = dense_subject(&model);
        assert!(
            in_scope(&model, &s),
            "{kind}: a serial correlation must be served"
        );
        let worst = parity(&model, &s);
        eprintln!("{kind}: worst {worst:.2e}");
        assert!(
            worst < 1e-6,
            "{kind}: worst relative difference {worst:.2e}"
        );
    }
}

#[test]
fn kernel_block_sigma_matches_reconverged_fd() {
    let model = parse_model_string(&dense_src("", true)).unwrap();
    assert!(!model.default_params.residual_correlations.is_empty());
    let s = dense_subject(&model);
    assert!(in_scope(&model, &s), "block_sigma must be served");
    let worst = parity(&model, &s);
    assert!(worst < 1e-6, "worst relative difference {worst:.2e}");
}

// ---- FREM covariate pseudo-observations through the kernel ------------------------------

const FREM_MODEL: &str = r#"
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

#[test]
fn kernel_frem_pseudo_observations_match_reconverged_fd() {
    let model = parse_model_string(FREM_MODEL).unwrap();
    let fc = model.frem_config.as_ref().expect("fixture must be FREM");
    let pk_times = [1.0, 6.0, 24.0];
    let mut s = subject_shell(
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &[1.0, 6.0, 24.0, 0.0],
    );
    s.fremtype = vec![0, 0, 0, 100];
    let theta = model.default_params.theta.clone();
    let preds =
        crate::pk::compute_predictions_with_tv(&model, &s, &theta, &[0.1, -0.05, 0.15, 0.0]);
    for j in 0..pk_times.len() {
        s.observations[j] = preds[j] * 0.85;
    }
    let (ti, _) = fc.fremtype_to_indices[&100u16];
    s.observations[3] = theta[ti] * 1.10;
    assert!(
        in_scope(&model, &s),
        "FREM rows must be served by the kernel"
    );
    // The FREM η variance (111.6) makes `H⁻¹` large, so the oracle's re-solved modes carry
    // the inner tolerance into its differences: measured 4.0e-4 at the default EBE tol
    // 1e-12 and 7.4e-6 at 1e-15 — the gap is the oracle's, and shrinks with it. (Its
    // H-matrix carries the FREM rows' exact `{0, 1}` Jacobian, as production's does.)
    ORACLE_EBE_TOL.set(1e-15);
    let worst = parity(&model, &s);
    ORACLE_EBE_TOL.set(1e-12);
    eprintln!("FREM vs reconverged FD at EBE tol 1e-15: {worst:.2e}");
    assert!(worst < 4e-5, "worst relative difference {worst:.2e}");
}

// ---- the f64 kernel IS the production objective ------------------------------------------

/// Kernel `F` and `l` at `(params, η)` against production's `foce_subject_nll` / inner
/// objective, on the production predictions and η-Jacobian. Returns `(|ΔF|, |Δl|)`, each
/// relative to `1 + |F|`.
fn objective_gaps<const N: usize>(
    model: &CompiledModel,
    s: &Subject,
    params: &ModelParameters,
    eta: &[f64],
) -> (f64, f64) {
    objective_gaps_for::<N>(model, s, params, eta, true)
}

fn objective_gaps_for<const N: usize>(
    model: &CompiledModel,
    s: &Subject,
    params: &ModelParameters,
    eta: &[f64],
    interaction: bool,
) -> (f64, f64) {
    let preds = crate::pk::compute_predictions_with_tv(model, s, &params.theta, eta);
    let hm = exact_eta_jacobian(model, s, &params.theta, eta).expect("η-Jacobian");
    let ev = nalgebra::DVector::from_column_slice(eta);
    // The fit's objective: `foce_subject_nll_cov` over the full residual structure (the
    // public `foce_subject_nll` wraps `block_sigma` pairs only).
    let f_prod = crate::stats::likelihood::foce_subject_nll_cov(
        model,
        s,
        &params.theta,
        &ev,
        &hm,
        &params.omega,
        &params.sigma.values,
        model.residual_cov_with(&params.residual_correlations, &params.theta),
        interaction,
    );
    // The inner objective the fit's EBE minimises: FOCE freezes the variance at f(η = 0)
    // (`EbeVariance::Population`); `None` is the conditional one.
    let frozen = if interaction {
        None
    } else {
        crate::stats::likelihood::population_variance_preds(model, s, &params.theta, N)
    };
    let mut scratch = crate::pk::EventPkParams::with_capacity_for(s);
    let keys = model.error_spec.obs_keys(s);
    let l_prod = crate::stats::likelihood::individual_nll_into_prepared_with_schedule_frozen(
        model,
        s,
        &params.theta,
        eta,
        &params.omega,
        &params.sigma.values,
        model.residual_cov_with(&params.residual_correlations, &params.theta),
        &mut scratch,
        None,
        keys.as_ref(),
        model.ruv_obs_mult(s, &params.theta).as_deref(),
        &mut Vec::new(),
        &mut nalgebra::DVector::zeros(eta.len()),
        &mut nalgebra::DVector::zeros(eta.len()),
        frozen.as_deref(),
    );
    let ctx = crate::sens::focei_kernel::LikCtx::new(model, s, params);
    let p = crate::sens::focei_kernel::ParT::of(model, s, params);
    let g: Vec<[f64; N]> = (0..hm.nrows())
        .map(|j| std::array::from_fn(|k| hm[(j, k)]))
        .collect();
    let e: [f64; N] = std::array::from_fn(|k| eta[k]);
    let fk = if interaction {
        crate::sens::focei_kernel::focei_f_grad::<f64, N>(&ctx, &preds, &g, &e, &p).map(|r| r.0)
    } else {
        crate::sens::focei_kernel::foce_f::<f64, N>(&ctx, &preds, &g, &e, frozen.as_deref(), &p)
    }
    .expect("kernel F");
    let lk = crate::sens::focei_kernel::inner_l::<f64, N>(&ctx, &preds, &e, &p, frozen.as_deref())
        .expect("kernel l");
    // (The kernel's F is `l + ½log|H̃|` by construction, so these two gaps also pin
    // production's `F − l` to the kernel's `½log|H̃|`.)
    let sc = 1.0 + f_prod.abs();
    ((fk - f_prod).abs() / sc, (lk - l_prod).abs() / sc)
}

#[test]
fn focei_objective_is_inner_plus_log_det_across_residual_structures() {
    let eta = [0.08, -0.06];
    // 1. plain diagonal, proportional + M3 rows
    let mut m3 = parse_model_string(ONE_CPT_ORAL).unwrap();
    m3.bloq_method = crate::types::BloqMethod::M3;
    let s3 = censor(
        subject(
            &m3,
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            &TIMES,
            &[0.1, -0.05, 0.2],
        ),
        &[0, 8],
    );
    let g = objective_gaps::<3>(&m3, &s3, &m3.default_params, &[0.08, -0.06, 0.1]);
    assert!(g.0 < 1e-12 && g.1 < 1e-12, "diagonal + M3: {g:?}");

    // 2. block_sigma + M3 rows (row-wise for both F and l, live ρ)
    let mut bs = parse_model_string(&dense_src("", true)).unwrap();
    bs.bloq_method = crate::types::BloqMethod::M3;
    let sb = censor(dense_subject(&bs), &[0, 7]);
    let mut pb = bs.default_params.clone();
    // A live ρ different from the declaration: the inner loop must read the live one.
    pb.residual_correlations[0].rho += 0.07;
    let g = objective_gaps::<2>(&bs, &sb, &pb, &eta);
    assert!(g.0 < 1e-10 && g.1 < 1e-10, "block_sigma + M3: {g:?}");

    // 3. serial correlation, dense both ways
    let sm = parse_model_string(&dense_src("  correlation ar1(RHO) by ARM\n", false)).unwrap();
    let ss = dense_subject(&sm);
    let g = objective_gaps::<2>(&sm, &ss, &sm.default_params, &eta);
    assert!(g.0 < 1e-10 && g.1 < 1e-10, "serial: {g:?}");

    // 4. block_sigma + IIV on the residual: F scales R by exp(2η_r) like l does. (A serial
    // `correlation` with `iiv_on_ruv` is rejected by the parser.)
    let src = dense_src("  iiv_on_ruv = ETA_RUV\n", true).replace(
        "  omega ETA_V ~ 0.04\n",
        "  omega ETA_V ~ 0.04\n  omega ETA_RUV ~ 0.1\n",
    );
    let sr = match parse_model_string(&src) {
        Ok(m) => m,
        Err(e) => panic!("block_sigma + iiv_on_ruv fixture: {e}"),
    };
    assert!(sr.residual_error_eta.is_some() && !sr.default_params.residual_correlations.is_empty());
    let srs = dense_subject(&sr);
    let g = objective_gaps::<3>(&sr, &srs, &sr.default_params, &[0.08, -0.06, 0.12]);
    assert!(g.0 < 1e-10 && g.1 < 1e-10, "block_sigma + RUV IIV: {g:?}");

    // 5. a serial correlation with a `FREMTYPE > 0` row that no `[frem]` block maps:
    //    production scores the subject row by row on any non-zero FREMTYPE (#2093 review
    //    round 2, finding 10); the kernel used to require a mapping and scored it densely.
    //    Cross-row correlation is what makes the two differ (a `block_sigma` pair correlates
    //    within a row, where dense and row-wise agree), so the straddle uses `ar1`.
    let mut sf = dense_subject(&sm);
    sf.fremtype = vec![0; sf.observations.len()];
    sf.fremtype[3] = 3;
    assert!(sm.frem_config.is_none(), "the FREMTYPE must be unmapped");
    let g = objective_gaps::<2>(&sm, &sf, &sm.default_params, &eta);
    assert!(
        g.0 < 1e-10 && g.1 < 1e-10,
        "serial + unmapped FREMTYPE: {g:?}"
    );
}

// ---- expression `obs_scale` divisor as its own tape leaf ---------------------------------

fn scaled_ode(src: &str) -> CompiledModel {
    let src = src
        .replace(
            "[scaling]\n  y = central / V\n",
            "[scaling]\n  obs_scale = V\n",
        )
        .replace(
            "ode(states=[depot, central])",
            "ode(obs_cmt=central, states=[depot, central])",
        );
    let m = parse_model_string(&src).unwrap();
    assert!(matches!(
        m.scaling,
        crate::types::ScalingSpec::ExpressionScale { .. }
    ));
    m
}

#[test]
fn expression_scale_on_the_ode_static_walk_matches_reconverged_fd() {
    let model = tight(scaled_ode(ONE_CPT_ORAL_ODE));
    let s = subject(
        &model,
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &TIMES,
        &[0.1, -0.05],
    );
    assert!(
        in_scope(&model, &s),
        "an expression obs_scale must be served"
    );
    let worst = parity(&model, &s);
    // Measured 2.0e-7 vs reconverged FD (Richardson, EBE re-solved per point); 5× headroom.
    assert!(worst < 1e-6, "worst relative difference {worst:.2e}");
}

#[test]
fn expression_scale_on_the_ode_event_walk_matches_reconverged_fd() {
    let model = tight(scaled_ode(ODE_LAG));
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let times = [1.0, 2.0, 4.0, 8.0, 13.0, 14.0, 18.0, 24.0];
    let s = subject(&model, doses, &times, &[0.1, -0.05]);
    assert!(in_scope(&model, &s));
    let worst = parity(&model, &s);
    // Measured 6.0e-7 vs reconverged FD (Richardson, EBE re-solved per point); 5× headroom.
    assert!(worst < 3e-6, "worst relative difference {worst:.2e}");
}

#[test]
fn expression_scale_on_the_closed_form_engine_matches_reconverged_fd() {
    let src = ONE_CPT_ORAL.replace(
        "[error_model]",
        "[scaling]\n  obs_scale = V / 10.0\n[error_model]",
    );
    let model = parse_model_string(&src).unwrap();
    assert!(matches!(
        model.scaling,
        crate::types::ScalingSpec::ExpressionScale { .. }
    ));
    let s = subject(
        &model,
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &TIMES,
        &[0.1, -0.05, 0.2],
    );
    assert!(in_scope(&model, &s));
    let worst = parity(&model, &s);
    // Measured 2.0e-7 vs reconverged FD (Richardson, EBE re-solved per point); 5× headroom.
    assert!(worst < 1e-6, "worst relative difference {worst:.2e}");
}

/// Regression (`warfarin_bloq` fit diverged): a fit arms M3 through the call-scoped
/// `bloq_in_force` override, with `model.bloq_method` left at its default. The kernel must
/// score the censored rows as the objective does.
#[test]
fn m3_under_the_fit_scoped_bloq_override_matches_reconverged_fd() {
    let model = parse_model_string(ONE_CPT_ORAL).unwrap();
    assert_ne!(model.bloq_method, crate::types::BloqMethod::M3);
    let _guard = crate::types::arm_bloq_override(Some(crate::types::BloqMethod::M3));
    let s = censor(
        subject(
            &model,
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            &TIMES,
            &[0.1, -0.05, 0.2],
        ),
        &[0, 7, 8],
    );
    let worst = parity(&model, &s);
    assert!(worst < 3e-7, "worst relative difference {worst:.2e}");
}

// ---- FOCE (no interaction) ------------------------------------------------------------

#[test]
fn foce_objective_matches_production_across_residual_structures() {
    // The kernel's Sheiner–Beal marginal (Woodbury on a diagonal R⁰, direct on a dense
    // one) against `foce_subject_nll_cov(interaction = false)`, and the shared inner `l`.
    let eta = [0.08, -0.06];
    let one = parse_model_string(ONE_CPT_ORAL).unwrap();
    let s1 = subject(
        &one,
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &TIMES,
        &[0.1, -0.05, 0.2],
    );
    let g = objective_gaps_for::<3>(&one, &s1, &one.default_params, &[0.08, -0.06, 0.1], false);
    assert!(g.0 < 1e-12 && g.1 < 1e-12, "diagonal: {g:?}");

    let mut m3 = parse_model_string(ONE_CPT_ORAL).unwrap();
    m3.bloq_method = crate::types::BloqMethod::M3;
    let s3 = censor(s1.clone(), &[0, 8]);
    let g = objective_gaps_for::<3>(&m3, &s3, &m3.default_params, &[0.08, -0.06, 0.1], false);
    assert!(g.0 < 1e-12 && g.1 < 1e-12, "diagonal + M3: {g:?}");

    let bs = parse_model_string(&dense_src("", true)).unwrap();
    let mut pb = bs.default_params.clone();
    pb.residual_correlations[0].rho += 0.07;
    let g = objective_gaps_for::<2>(&bs, &dense_subject(&bs), &pb, &eta, false);
    assert!(g.0 < 1e-10 && g.1 < 1e-10, "block_sigma: {g:?}");

    let sm = parse_model_string(&dense_src("  correlation ar1(RHO) by ARM\n", false)).unwrap();
    let g = objective_gaps_for::<2>(&sm, &dense_subject(&sm), &sm.default_params, &eta, false);
    assert!(g.0 < 1e-10 && g.1 < 1e-10, "serial: {g:?}");
}

// FOCE's gradient is checked against the reconverged objective under FOCE's own EBE
// convention (`ebe_for`: the frozen-variance mode).

#[test]
fn foce_adjoint_matches_reconverged_fd() {
    let one = parse_model_string(ONE_CPT_ORAL).unwrap();
    let s = subject(
        &one,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &TIMES,
        &[0.1, -0.05, 0.2],
    );
    let worst = parity_foce(&one, &s);
    // Measured 1.9e-7 vs reconverged FD (Richardson, EBE re-solved per point); 5× headroom.
    assert!(worst < 1e-6, "closed form: {worst:.2e}");
    let ode = tight(parse_model_string(TWO_CPT_INF_COMBINED_BLOCK).unwrap());
    let s = subject(
        &ode,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 50.0, false, 0.0),
        ],
        &TIMES,
        &[0.1, -0.05, 0.08],
    );
    let worst = parity_foce(&ode, &s);
    eprintln!("FOCE ODE vs reconverged FD {worst:.2e}");
    // Measured 7.9e-7 vs reconverged FD (Richardson, EBE re-solved per point); 5× headroom.
    assert!(worst < 4e-6, "ODE: {worst:.2e}");
}

#[test]
fn foce_adjoint_matches_reconverged_fd_closed_form_and_ode() {
    // Proportional error, so R⁰ is at the population prediction (the η = 0 pass).
    let one = parse_model_string(ONE_CPT_ORAL).unwrap();
    let s = subject(
        &one,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &TIMES,
        &[0.1, -0.05, 0.2],
    );
    let worst = fd_oracle(&one, &s, false);
    assert!(worst < FOCE_FD_TOL, "closed form: {worst:.2e}");

    let ode = with_tol(
        parse_model_string(TWO_CPT_INF_COMBINED_BLOCK).unwrap(),
        1e-12,
        1e-14,
    );
    let s = subject(
        &ode,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 50.0, false, 0.0),
        ],
        &TIMES,
        &[0.1, -0.05, 0.08],
    );
    let worst = fd_oracle(&ode, &s, false);
    assert!(
        worst < FOCE_FD_TOL,
        "ODE, combined error, block Ω: {worst:.2e}"
    );
}

#[test]
fn foce_adjoint_matches_reconverged_fd_m3_and_dense_residuals() {
    let mut m3 = parse_model_string(ONE_CPT_ORAL).unwrap();
    m3.bloq_method = crate::types::BloqMethod::M3;
    let s = censor(
        subject(
            &m3,
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            &TIMES,
            &[0.1, -0.05, 0.2],
        ),
        &[0, 8],
    );
    let worst = fd_oracle(&m3, &s, false);
    assert!(worst < FOCE_FD_TOL, "M3: {worst:.2e}");

    let bs = parse_model_string(&dense_src("", true)).unwrap();
    let worst = fd_oracle(&bs, &dense_subject(&bs), false);
    assert!(worst < FOCE_FD_TOL, "block_sigma: {worst:.2e}");

    let sm = parse_model_string(&dense_src("  correlation ar1(RHO) by ARM\n", false)).unwrap();
    let worst = fd_oracle(&sm, &dense_subject(&sm), false);
    assert!(worst < FOCE_FD_TOL, "serial: {worst:.2e}");
}

/// Measured worst 4.8e-7 over the five FOCE fixtures (the Richardson FD floor at EBE tol
/// 1e-12); 8× headroom.
const FOCE_FD_TOL: f64 = 4e-6;

// ---- IOV: the stacked [η, κ_0, …, κ_{K−1}] -------------------------------------------

const ODE_IOV: &str = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(30.0, 1.0, 500.0)
  theta TVKA(1.2, 0.05, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.06
  kappa KAPPA_CL ~ 0.03
  kappa KAPPA_V ~ 0.02
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V = TVV * exp(ETA_V + KAPPA_V)
  KA = TVKA
[structural_model]
  ode(states=[depot, central])
[odes]
  d/dt(depot) = -KA * depot
  d/dt(central) = KA * depot - (CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  iov_column = OCC
"#;

/// Three occasions, a dose at the start of each, so every later occasion starts with
/// residual drug from the previous one (κ_g reaches the following occasions too).
pub(super) fn iov_subject(model: &CompiledModel) -> Subject {
    let times = [1.0, 4.0, 10.0, 13.0, 16.0, 22.0, 25.0, 28.0, 34.0];
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let mut s = subject_shell(doses, &times);
    s.occasions = vec![1, 1, 1, 2, 2, 2, 3, 3, 3];
    s.dose_occasions = vec![1, 2, 3];
    let kappas = vec![vec![0.1, -0.05], vec![-0.08, 0.04], vec![0.05, 0.06]];
    let preds = crate::pk::predict_iov(
        model,
        &s,
        &model.default_params.theta,
        &[0.1, -0.05],
        &kappas,
    );
    s.observations = preds
        .iter()
        .enumerate()
        .map(|(j, p)| p * (1.0 + 0.12 * (1.7 * j as f64).sin()))
        .collect();
    s
}

pub(super) fn iov_parity(model: &CompiledModel, s: &Subject, interaction: bool) -> f64 {
    assert!(
        model.n_kappa > 0 && s.occasions.iter().any(|&o| o > 0),
        "an IOV fixture"
    );
    fd_oracle(model, s, interaction)
}

#[test]
fn iov_focei_adjoint_matches_reconverged_fd() {
    let model = tight(parse_model_string(ODE_IOV).unwrap());
    let s = iov_subject(&model);
    let worst = iov_parity(&model, &s, true);
    // Measured 3.1e-6 vs reconverged FD (Richardson, EBE re-solved per point); 5× headroom.
    assert!(worst < 1.6e-5, "IOV FOCEI: {worst:.2e}");
}

#[test]
fn iov_foce_adjoint_matches_reconverged_fd() {
    let model = tight(parse_model_string(ODE_IOV).unwrap());
    let s = iov_subject(&model);
    let worst = iov_parity(&model, &s, false);
    // Measured 7.6e-6 vs reconverged FD (Richardson, EBE re-solved per point); 5× headroom.
    assert!(worst < 4e-5, "IOV FOCE: {worst:.2e}");
}

/// The exact outer gradient of an IOV subject carrying a TTE record: the term reads BSV η
/// only, so it rides the stacked system's leading block, in the objective (`2q`, `∇²q` in
/// `H̃`) and in the mode system. Against the reconverged production IOV marginal (which
/// scores the term since the marginal fix), under both interaction settings.
#[cfg(feature = "survival")]
#[test]
fn the_iov_outer_gradient_is_exact_with_a_non_gaussian_term() {
    use crate::types::{EventType, ObsRecord};
    let src = ODE_IOV
        .replace(
            "  theta TVKA(1.2, 0.05, 20.0)\n",
            "  theta TVKA(1.2, 0.05, 20.0)\n  theta TVLAM(0.05, 0.001, 10.0)\n",
        )
        .replace(
            "[fit_options]",
            "[event_model]\n  cmt    = 3\n  family = exponential\n  \
             scale  = TVLAM * exp(0.5 * ETA_CL)\n[fit_options]",
        );
    let model = tight(parse_model_string(&src).unwrap());
    assert!(model.n_kappa > 0 && model.has_tte());
    assert!(outer_model_in_scope(&model), "the outer route is analytic");
    let mut s = iov_subject(&model);
    s.obs_records = vec![ObsRecord::Event {
        time: 15.0,
        event_type: EventType::Exact,
        entry_time: 0.0,
        cmt: 3,
    }];
    let mut worst = 0.0f64;
    for interaction in [true, false] {
        let w = fd_oracle(&model, &s, interaction);
        eprintln!("IOV + TTE interaction={interaction}: worst {w:.2e}");
        worst = worst.max(w);
    }
    // The quadrature over the stacked `[η, κ]`, the term on its BSV block: Laplace, and the
    // Gauss–Newton anchor on a 2-node grid.
    for (n_agq, exact) in [(1, true), (2, false)] {
        let w = super::phenomena_tests::fd_oracle_agq_iov(&model, &s, n_agq, exact);
        eprintln!("IOV + TTE n_agq={n_agq} exact={exact}: worst {w:.2e}");
        worst = worst.max(w);
    }
    // The IOV fixtures' reconverged oracle floor is 3e-6 … 8e-6 (above).
    assert!(worst < 5e-5, "{worst:.2e}");
}

/// #979: a two-endpoint model — total and unbound concentration, a per-CMT proportional
/// error each — whose two sigmas form a `block_sigma`, with a κ on CL. The shape of the
/// fluconazole total/unbound model that tripped `E_BLOCK_SIGMA_IOV_UNSUPPORTED`.
pub(super) const ODE_IOV_BLOCK_SIGMA: &str = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(30.0, 1.0, 500.0)
  theta FU(0.3, 0.01, 1.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.06
  kappa KAPPA_CL ~ 0.04
  block_sigma (PROP_TOT, PROP_UNB) = [0.01, 0.006, 0.0225]
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V = TVV * exp(ETA_V)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y[CMT=1] = central / V
  y[CMT=2] = FU * central / V
[error_model]
  CMT=1: DV ~ proportional(PROP_TOT)
  CMT=2: DV ~ proportional(PROP_UNB)
[fit_options]
  iov_column = OCC
"#;

/// Three occasions, a dose at the start of each so every later occasion starts with
/// residual drug (κ_g reaches the following occasions), and a total and an unbound
/// sample at every sampling time — the rows `block_sigma` pairs.
pub(super) fn iov_two_endpoint_subject(model: &CompiledModel) -> Subject {
    let base = [1.0, 4.0, 10.0, 13.0, 16.0, 22.0, 25.0, 28.0, 34.0];
    let times: Vec<f64> = base.iter().flat_map(|&t| [t, t]).collect();
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let mut s = subject_shell(doses, &times);
    s.obs_cmts = (0..times.len()).map(|j| 1 + j % 2).collect();
    s.occasions = times.iter().map(|&t| 1 + (t / 12.0) as u32).collect();
    s.dose_occasions = vec![1, 2, 3];
    let kappas = vec![vec![0.12], vec![-0.1], vec![0.05]];
    let preds = crate::pk::predict_iov(
        model,
        &s,
        &model.default_params.theta,
        &[0.1, -0.05],
        &kappas,
    );
    // Correlated noise: the unbound row of a pair shares most of its total row's error.
    s.observations = preds
        .iter()
        .enumerate()
        .map(|(j, p)| {
            let e = 0.1 * (1.7 * (j / 2) as f64).sin();
            p * (1.0 + if j % 2 == 0 { e } else { 0.8 * e + 0.04 })
        })
        .collect();
    s
}

/// #979: the adjoint's exact outer gradient of the IOV marginal under a two-endpoint
/// `block_sigma` — every θ, Ω, Ω_IOV, σ and the ρ coordinate — against the reconverged
/// FD oracle, which scores `foce_subject_nll_iov_cov` at the live ρ with the EBE re-solved
/// (`find_ebe_iov`, also at the live ρ) at every point. FOCEI and FOCE.
///
/// Kills: the IOV marginal or the IOV EBE objective reading the *declared* ρ
/// (`residual_cov_theta`) instead of the live one — the FD column on ρ then sees no ρ,
/// or a mode of a different objective, while the kernel differentiates the live one.
#[test]
fn iov_block_sigma_adjoint_matches_reconverged_fd() {
    let model = tight(parse_model_string(ODE_IOV_BLOCK_SIGMA).unwrap());
    assert!(model.n_kappa == 1 && !model.residual_correlations.is_empty());
    let s = iov_two_endpoint_subject(&model);
    // The pairs are live: every total row has an unbound partner, so `R` is dense.
    assert!(
        crate::stats::residual_error::has_cross_observation_residual(
            &model.error_spec,
            &s,
            &model.default_params.sigma.values,
            model.residual_cov_at(&model.default_params),
        )
    );
    for (interaction, tol) in [(true, 2e-5), (false, 4e-5)] {
        let worst = iov_parity(&model, &s, interaction);
        eprintln!("interaction {interaction}: worst {worst:.2e}");
        assert!(worst < tol, "interaction {interaction}: {worst:.2e}");
    }
}

#[test]
fn iov_linearization_uses_the_exact_kappa_jacobian() {
    // The IOV objective's κ columns come from the analytic stacked η-Jacobian, not finite
    // differences: pinned against an independent Richardson difference of `predict_iov`,
    // on a subject whose later occasions carry the earlier ones' drug (κ_g reaches them).
    let model = tight(parse_model_string(ODE_IOV).unwrap());
    let s = iov_subject(&model);
    let p = moved_params(&model);
    let eta = [0.07, -0.04];
    let kappas = [vec![0.06, -0.03], vec![-0.05, 0.02], vec![0.04, 0.05]];
    let stacked: Vec<f64> = eta
        .iter()
        .copied()
        .chain(kappas.iter().flatten().copied())
        .collect();
    let exact = exact_obs_grads(&model, &s, &p.theta, &stacked)
        .expect("the exact branch must serve this subject");
    let n_obs = s.obs_times.len();
    let lin = crate::stats::likelihood::iov_linearization(
        &model,
        &s,
        &p.theta,
        &nalgebra::DVector::from_column_slice(&eta),
        &nalgebra::DMatrix::zeros(n_obs, 2),
        &p.omega,
        &kappas
            .iter()
            .map(|k| nalgebra::DVector::from_column_slice(k))
            .collect::<Vec<_>>(),
        p.omega_iov.as_ref().unwrap(),
    )
    .unwrap();
    let mut worst = 0.0f64;
    for g in 0..3 {
        for ki in 0..2 {
            let fd = |h: f64| {
                let (mut kp, mut km) = (kappas.to_vec(), kappas.to_vec());
                kp[g][ki] += h;
                km[g][ki] -= h;
                let (yp, ym) = (
                    crate::pk::predict_iov(&model, &s, &p.theta, &eta, &kp),
                    crate::pk::predict_iov(&model, &s, &p.theta, &eta, &km),
                );
                yp.iter()
                    .zip(&ym)
                    .map(|(a, b)| (a - b) / (2.0 * h))
                    .collect::<Vec<f64>>()
            };
            let (f1, f2) = (fd(1e-3), fd(2e-3));
            let col = 2 + 2 * g + ki;
            for j in 0..n_obs {
                let rich = (4.0 * f1[j] - f2[j]) / 3.0;
                let exact = lin.h_full[(j, col)];
                assert!(exact.is_finite() && rich.is_finite());
                worst = worst.max((exact - rich).abs() / (1.0 + rich.abs()));
            }
        }
        // κ_g must reach a later occasion's rows (the carryover), or the fixture is blind
        // to a column that only fills its own occasion.
        assert!(
            g == 2 || lin.h_full[(3 * g + 4, 2 + 2 * g)].abs() > 1e-6,
            "κ_{g} must reach the next occasion"
        );
    }
    // The columns ARE the provider's (a central difference would also sit near the
    // Richardson value, so only bit-identity pins which branch produced them)…
    for j in 0..n_obs {
        for c in 2..8 {
            assert_eq!(
                lin.h_full[(j, c)].to_bits(),
                exact[j].df_deta[c].to_bits(),
                "row {j} column {c}"
            );
        }
    }
    // …and they are right: measured worst 6.8e-12 against the Richardson difference.
    assert!(worst < 1e-9, "worst κ-column gap {worst:.2e}");
}

// ---- IOV past the retired 24-axis program cap (#2031) ---------------------------------

/// `src` with `n` live θ `PAD0…PAD{n-1}` added after the last θ line, each scaling the
/// `[individual_parameters]` line `target` by `(1 + 0.02·PADi)`. Live, not inert: a pad at
/// `+ 0.0 * PADi` has an exactly-zero θ column, so the parity below would not see a
/// θ column the recording dropped or misplaced.
fn with_live_pads(src: &str, last_theta: &str, target: &str, n: usize) -> String {
    let thetas: String = (0..n)
        .map(|i| format!("  theta PAD{i}(0.5, 0.01, 10.0)\n"))
        .collect();
    let terms: String = (0..n).map(|i| format!(" + 0.02 * PAD{i}")).collect();
    assert!(
        src.contains(last_theta) && src.contains(target),
        "pad anchors"
    );
    src.replacen(last_theta, &format!("{last_theta}{thetas}"), 1)
        .replacen(target, &format!("{target} * (1.0{terms})"), 1)
}

/// Compartment-free IOV twin of [`ODE_IOV`]: 3 θ, 2 η, 2 κ (7 axes), the same η/κ layout.
const ALGEBRAIC_IOV: &str = r#"
[parameters]
  theta TVE0(10.0, 0.1, 100.0)
  theta TVEMAX(6.0, 0.1, 100.0)
  theta TVET50(2.0, 0.01, 100.0)
  omega ETA_E0 ~ 0.09
  omega ETA_EMAX ~ 0.06
  kappa KAPPA_E0 ~ 0.03
  kappa KAPPA_EMAX ~ 0.02
  sigma ADD ~ 1.0 (sd)
[individual_parameters]
  E0   = TVE0 * exp(ETA_E0 + KAPPA_E0)
  EMAX = TVEMAX * exp(ETA_EMAX + KAPPA_EMAX)
  ET50 = TVET50
[structural_model]
  y = E0 - EMAX * TIME / (ET50 + TIME)
[error_model]
  DV ~ additive(ADD)
[fit_options]
  iov_column = OCC
"#;

/// [`iov_subject`] for a compartment-free model: the same three occasions, no doses.
fn algebraic_iov_subject(model: &CompiledModel) -> Subject {
    let times = [1.0, 4.0, 10.0, 13.0, 16.0, 22.0, 25.0, 28.0, 34.0];
    let mut s = subject_shell(vec![], &times);
    s.occasions = vec![1, 1, 1, 2, 2, 2, 3, 3, 3];
    let kappas = vec![vec![0.1, -0.05], vec![-0.08, 0.04], vec![0.05, 0.06]];
    let preds = crate::pk::predict_iov(
        model,
        &s,
        &model.default_params.theta,
        &[0.1, -0.05],
        &kappas,
    );
    s.observations = preds
        .iter()
        .enumerate()
        .map(|(j, p)| p * (1.0 + 0.12 * (1.7 * j as f64).sin()))
        .collect();
    s
}

/// The inner loop's stacked `[η, κ_0, κ_1, κ_2]` Jacobian (`exact_obs_grads`, all the inner
/// gradient reads) against a Richardson difference of `predict_iov`: the worst relative gap
/// over every row and stacked column.
fn iov_jacobian_gap(model: &CompiledModel, s: &Subject, theta: &[f64]) -> f64 {
    let eta = [0.07, -0.04];
    let kappas = [vec![0.06, -0.03], vec![-0.05, 0.02], vec![0.04, 0.05]];
    let stacked: Vec<f64> = eta
        .iter()
        .copied()
        .chain(kappas.iter().flatten().copied())
        .collect();
    let exact = exact_obs_grads(model, s, theta, &stacked).expect("inner recording serves it");
    let predict = |st: &[f64]| {
        let k: Vec<Vec<f64>> = st[2..].chunks(2).map(<[f64]>::to_vec).collect();
        crate::pk::predict_iov(model, s, theta, &st[..2], &k)
    };
    let mut worst = 0.0f64;
    for c in 0..stacked.len() {
        let fd = |h: f64| {
            let (mut p, mut m) = (stacked.clone(), stacked.clone());
            p[c] += h;
            m[c] -= h;
            let (yp, ym) = (predict(&p), predict(&m));
            yp.iter()
                .zip(&ym)
                .map(|(a, b)| (a - b) / (2.0 * h))
                .collect::<Vec<f64>>()
        };
        let (f1, f2) = (fd(1e-3), fd(2e-3));
        for (j, row) in exact.iter().enumerate() {
            let rich = (4.0 * f1[j] - f2[j]) / 3.0;
            assert!(
                row.df_deta[c].is_finite() && rich.is_finite(),
                "row {j} column {c}: exact {} fd {rich}",
                row.df_deta[c]
            );
            worst = worst.max((row.df_deta[c] - rich).abs() / (1.0 + rich.abs()));
        }
    }
    worst
}

/// #2031: before #2093 the forward provider declined an IOV model on the ODE engine
/// (`ode_iov_supported`) or the compartment-free readout (`algebraic::supported_iov`) once
/// `n_theta + n_eta + n_kappa > 24`, FIX θ included, so both loops ran on finite
/// differences. The adjoint seeds θ on its reverse tape and only `[η, κ]` forward, so a
/// 25-axis model (18 live pads on a 7-axis fixture) records like a 7-axis one: the outer
/// FOCEI gradient matches the reconverged FD of the production objective along every packed
/// coordinate, the pads' included, and the inner stacked Jacobian matches FD of
/// `predict_iov`. A width clause reinstated anywhere on the subject path panics at the
/// `expect`s instead.
fn assert_wide_iov_parity(engine: &str, model: &CompiledModel, s: &Subject, tol: (f64, f64)) {
    assert!(
        model.n_theta + model.n_eta + model.n_kappa > 24,
        "{engine}: the fixture must sit past the old cap"
    );
    assert!(recorder_in_scope(model), "{engine}: model-level scope");
    let jac = iov_jacobian_gap(model, s, &moved_params(model).theta);
    let outer = iov_parity(model, s, true);
    eprintln!("#2031 {engine} at 25 axes: inner Jacobian vs FD {jac:.2e}, FOCEI vs reconverged FD {outer:.2e}");
    assert!(jac < tol.0, "{engine}: inner Jacobian vs FD {jac:.2e}");
    assert!(
        outer < tol.1,
        "{engine}: FOCEI vs reconverged FD {outer:.2e}"
    );
}

#[test]
fn ode_iov_past_24_axes_matches_fd() {
    let src = with_live_pads(
        ODE_IOV,
        "  theta TVKA(1.2, 0.05, 20.0)\n",
        "  CL = TVCL * exp(ETA_CL + KAPPA_CL)",
        18,
    );
    let model = tight(parse_model_string(&src).unwrap());
    let s = iov_subject(&model);
    // Measured (macOS arm64): Jacobian 5.0e-13, FOCEI 3.1e-9 (the Richardson floor at EBE
    // tol 1e-12, as on the 7-axis fixture); 10× headroom each.
    assert_wide_iov_parity("ODE", &model, &s, (5e-12, 3e-8));
}

#[test]
fn compartment_free_iov_past_24_axes_matches_fd() {
    let src = with_live_pads(
        ALGEBRAIC_IOV,
        "  theta TVET50(2.0, 0.01, 100.0)\n",
        "  E0   = TVE0 * exp(ETA_E0 + KAPPA_E0)",
        18,
    );
    let model = parse_model_string(&src).unwrap();
    assert!(model.is_algebraic());
    let s = algebraic_iov_subject(&model);
    // The oracle's EBE at 1e-7, not 1e-12: this exact objective has no stall stop, and its
    // BFGS cannot certify `|g| < 1e-8` at the moved θ (the inner gradient agrees with a
    // central difference to 9 figures; it is the L2 floor), so every stencil point fell to
    // Nelder–Mead and the test took 36 s in debug.
    ORACLE_EBE_TOL.set(1e-7);
    // Measured (macOS arm64): Jacobian 1.7e-13, FOCEI 6.4e-7 (6.3e-12 with the oracle at
    // 1e-12); 10× headroom each. A dropped or misplaced θ column is an O(1) relative gap.
    assert_wide_iov_parity("compartment-free", &model, &s, (2e-12, 6.4e-6));
    ORACLE_EBE_TOL.set(1e-12);
}

// ---- closed-form event walk: time-varying covariates and IOV --------------------------

const CF_TVCOV: &str = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(30.0, 1.0, 500.0)
  theta TVKA(1.2, 0.05, 20.0)
  theta CL_WT(0.75, -2.0, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.06
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * (WT / 70.0)^CL_WT * exp(ETA_CL)
  V = TVV * exp(ETA_V)
  KA = TVKA
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

fn cf_tvcov_subject(model: &CompiledModel) -> Subject {
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let times = [1.0, 2.0, 4.0, 8.0, 13.0, 14.0, 18.0, 24.0];
    // WT moves between every pair of records, so each dose and observation reads its own
    // snapshot and the walk switches parameters with drug on board.
    let wt_at = |t: f64| 60.0 + 1.5 * t;
    let mut s = subject_shell(doses.clone(), &times);
    s.covariates.insert("WT".into(), 70.0);
    s.obs_covariates = times
        .iter()
        .map(|&t| HashMap::from([("WT".to_string(), wt_at(t))]))
        .collect();
    s.dose_covariates = doses
        .iter()
        .map(|d| HashMap::from([("WT".to_string(), wt_at(d.time))]))
        .collect();
    let preds = crate::pk::compute_predictions_with_tv(
        model,
        &s,
        &model.default_params.theta,
        &[0.1, -0.05],
    );
    s.observations = preds
        .iter()
        .enumerate()
        .map(|(j, p)| p * (1.0 + 0.15 * (1.3 * j as f64).sin()))
        .collect();
    assert!(
        s.has_tv_covariates() && crate::sens::provider::subject_routes_to_event_walk(model, &s)
    );
    s
}

#[test]
fn closed_form_time_varying_covariate_matches_reconverged_fd() {
    let model = parse_model_string(CF_TVCOV).unwrap();
    let s = cf_tvcov_subject(&model);
    assert!(
        in_scope(&model, &s),
        "the closed-form event walk must serve TV covariates"
    );
    let worst = parity(&model, &s);
    // Measured 7.2e-9 (FOCEI) vs reconverged FD (Richardson, EBE re-solved per point); 5× headroom.
    assert!(worst < 4e-8, "FOCEI: {worst:.2e}");
    let worst = parity_foce(&model, &s);
    eprintln!("closed-form TV-cov FOCE vs reconverged FD {worst:.2e}");
    // Measured 5.2e-7 vs reconverged FD (Richardson, EBE re-solved per point); 5× headroom.
    assert!(worst < 2.6e-6, "FOCE: {worst:.2e}");
}

const CF_IOV: &str = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(30.0, 1.0, 500.0)
  theta TVKA(1.2, 0.05, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.06
  kappa KAPPA_CL ~ 0.03
  kappa KAPPA_V ~ 0.02
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V = TVV * exp(ETA_V + KAPPA_V)
  KA = TVKA
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  iov_column = OCC
"#;

#[test]
fn closed_form_iov_matches_reconverged_fd() {
    let model = parse_model_string(CF_IOV).unwrap();
    assert!(model.ode_spec.is_none());
    let s = iov_subject(&model);
    let worst = iov_parity(&model, &s, true);
    // Measured 9.1e-9 vs reconverged FD (Richardson, EBE re-solved per point); 5× headroom.
    assert!(worst < 5e-8, "IOV FOCEI: {worst:.2e}");
    let worst = iov_parity(&model, &s, false);
    eprintln!("closed-form IOV FOCE vs reconverged FD {worst:.2e}");
    // Measured 3.0e-7 vs reconverged FD (Richardson, EBE re-solved per point); 5× headroom.
    assert!(worst < 1.5e-6, "IOV FOCE: {worst:.2e}");
}

#[test]
fn ode_iov_with_ltbs_matches_reconverged_fd() {
    // LTBS under ODE IOV (outside the retired forward provider's scope, inside the shared
    // walk's `ode_iov_model_scope`): against the reconverged objective.
    let src = ODE_IOV
        .replace("  sigma PROP_ERR ~ 0.1 (sd)", "  sigma ADD_ERR ~ 0.1 (sd)")
        .replace(
            "  DV ~ proportional(PROP_ERR)",
            "  log(DV) ~ additive(ADD_ERR)",
        );
    let model = tight(parse_model_string(&src).unwrap());
    assert!(model.log_transform);
    let s = iov_subject(&model);
    let params = moved_params(&model);
    let x = pack_params(&params);
    let eta = ebe_for(&model, &s, &params, 1e-12, true);
    let adj = subject_adjoint_packed_gradient(&model, &s, &params, &x, eta.as_slice(), true)
        .expect("LTBS + ODE IOV must be in adjoint scope");
    let kap = |e: &[f64]| -> Vec<nalgebra::DVector<f64>> {
        (0..3)
            .map(|g| nalgebra::DVector::from_column_slice(&e[2 + 2 * g..4 + 2 * g]))
            .collect()
    };
    let f_at = |xv: &[f64]| -> f64 {
        let p = crate::estimation::parameterization::unpack_params(xv, &params);
        let e = ebe_for(&model, &s, &p, 1e-12, true);
        let jac = exact_obs_grads(&model, &s, &p.theta, e.as_slice()).unwrap();
        let hm = nalgebra::DMatrix::from_fn(s.obs_times.len(), 2, |j, k| jac[j].df_deta[k]);
        crate::stats::likelihood::foce_subject_nll_iov(
            &model,
            &s,
            &p.theta,
            &nalgebra::DVector::from_column_slice(&e.as_slice()[..2]),
            &hm,
            &p.omega,
            &p.sigma.values,
            true,
            &kap(e.as_slice()),
            p.omega_iov.as_ref().unwrap(),
        )
    };
    let mut worst = 0.0f64;
    for k in 0..x.len() {
        let fd = |h: f64| {
            let (mut xp, mut xm) = (x.clone(), x.clone());
            xp[k] += h;
            xm[k] -= h;
            (f_at(&xp) - f_at(&xm)) / (2.0 * h)
        };
        let rich = (4.0 * fd(1e-3) - fd(2e-3)) / 3.0;
        assert!(rich.is_finite() && adj[k].is_finite());
        let rel = (adj[k] - rich).abs() / (1.0 + rich.abs());
        eprintln!(
            "x[{k}] adjoint={:.10e} reconverged-FD={rich:.10e} rel={rel:.2e}",
            adj[k]
        );
        worst = worst.max(rel);
    }
    // Measured worst 3.7e-6 (the Ω coordinates; θ and σ ≤ 8e-7): the Richardson floor of
    // an objective re-solved over an 8-wide stacked EBE at tol 1e-12. 10× headroom.
    assert!(worst < 4e-5, "{worst:.2e}");
}

const CF_ORAL_F: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta TVF(0.7, 0.05, 1.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
  F  = TVF
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, f=F)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

#[test]
fn closed_form_rate_infusion_under_bioavailability_matches_reconverged_fd() {
    // A rate-defined (zero-order) dose under an estimated F: production holds the rate and
    // stretches the window to `F·dur` (#419), applying the dose-time `F` once. The shared
    // closed-form walk used to scale the rate by the interval's `F` over the unstretched
    // window instead — measured, the gradient was off the reconverged objective by up to
    // 5× (∂/∂ln σ). Two doses so the second lands on drug on board; the oracle is the
    // reconverged objective.
    let model = parse_model_string(CF_ORAL_F).unwrap();
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 50.0, false, 0.0),
    ];
    let s = subject(&model, doses, &TIMES, &[0.1, -0.05, 0.2]);
    let worst = fd_oracle(&model, &s, true);
    // Measured worst 1.1e-5: the oracle's own floor here (it differences its η-Jacobian
    // too, across the moving window end). ~9× headroom.
    assert!(worst < 1e-4, "oral infusion under F: {worst:.2e}");
}

#[test]
fn closed_form_iv_infusion_under_bioavailability_matches_reconverged_fd() {
    // A static IV infusion under F: production predicts it by superposition with the
    // stretched window; the adjoint routes it to the event walk (the superposition kernels
    // take a fixed window), the same function.
    let src = CF_ORAL_F
        .replace(
            "pk one_cpt_oral(cl=CL, v=V, ka=KA, f=F)",
            "pk one_cpt_iv(cl=CL, v=V, f=F)",
        )
        .replace("  KA = TVKA * exp(ETA_KA)\n", "")
        .replace("  omega ETA_KA ~ 0.30\n", "")
        .replace("  theta TVKA(1.5, 0.01, 50.0)\n", "");
    let model = parse_model_string(&src).unwrap();
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 50.0, false, 0.0),
    ];
    let s = subject(&model, doses, &TIMES, &[0.1, -0.05]);
    assert!(
        !crate::sens::provider::subject_routes_to_event_walk(&model, &s),
        "production predicts this subject by superposition"
    );
    let worst = fd_oracle(&model, &s, true);
    // Measured worst 8.7e-6 (the same oracle floor).
    assert!(worst < 1e-4, "IV infusion under F: {worst:.2e}");
}

// ---- Laplace: `l + ½log|∇²_η l|`, exact Hessian ---------------------------------------

/// Production's Laplace subject objective (`n_agq = 1`, exact anchor) at `η`.
pub(super) fn laplace_prod(
    model: &CompiledModel,
    s: &Subject,
    params: &ModelParameters,
    eta: &[f64],
) -> f64 {
    agq_prod(model, s, params, eta, 1, true)
}

/// Production's AGQ subject objective at `η`.
fn agq_prod(
    model: &CompiledModel,
    s: &Subject,
    params: &ModelParameters,
    eta: &[f64],
    n_agq: usize,
    exact: bool,
) -> f64 {
    use crate::types::HessianAnchor;
    crate::estimation::agq::agq_subject_objective_anchored(
        model,
        s,
        params,
        eta,
        n_agq,
        if exact {
            HessianAnchor::Exact
        } else {
            HessianAnchor::GaussNewton
        },
    )
}

/// The kernel's Laplace objective at `η`, from the recorder's exact `(f, G, Q)`.
fn laplace_kernel<const N: usize, const M: usize>(
    model: &CompiledModel,
    s: &Subject,
    params: &ModelParameters,
    eta: &[f64],
) -> f64 {
    use crate::sens::rtape::DualH;
    let jets = canonical_jets::<N, M>(model, s, None, &params.theta, eta).expect("recorded η-jets");
    let fh: Vec<DualH<N, M, f64>> = jets
        .iter()
        .map(|p| DualH {
            v: p.v,
            g: p.g,
            h: p.h,
        })
        .collect();
    let eh: [DualH<N, M, f64>; N] = std::array::from_fn(|k| {
        let mut g = [0.0; N];
        g[k] = 1.0;
        DualH {
            v: eta[k],
            g,
            h: [0.0; M],
        }
    });
    let ctx = crate::sens::focei_kernel::LikCtx::new(model, s, params);
    let p = crate::sens::focei_kernel::ParT::of(model, s, params);
    let l = crate::sens::focei_kernel::inner_l(
        &ctx,
        &fh,
        &eh,
        &p.map(|x| DualH {
            v: x,
            g: [0.0; N],
            h: [0.0; M],
        }),
        None,
    )
    .unwrap();
    // The Laplace approximation's own `½log|H|`, no jitter (#2538): this fixture's
    // Hessian is positive-definite, where production factors it as is.
    let h = nalgebra::DMatrix::from_fn(N, N, |a, b| l.hess(a, b));
    l.v + 0.5
        * h.cholesky()
            .unwrap()
            .l()
            .diagonal()
            .iter()
            .map(|d| 2.0 * d.ln())
            .sum::<f64>()
}

/// Where the regularised exact anchor is not positive-definite, production's Laplace
/// objective falls back to the prior precision `Ω⁻¹` (`build_proposal`) and stays finite.
/// The fixture point is found on a fixed grid of η where `H` is indefinite, which is
/// asserted — the straddle — so the check cannot pass on a point that never reaches the
/// fallback. A grid point is not a mode: there the adjoint declines (its identity needs
/// `∇_η l = 0`; #2093 review round 2, finding 14), and the decline is asserted. At a
/// stationary point it serves an indefinite `H` through `mode_solve`'s LU branch
/// (`mode_solve_takes_an_indefinite_nonsingular_hessian`,
/// `mode_solve_declines_an_indefinite_hessian_off_the_mode`).
#[test]
fn laplace_serves_the_prior_precision_anchor_fallback() {
    let one = parse_model_string(ONE_CPT_ORAL).unwrap();
    let s = subject(
        &one,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &TIMES,
        &[0.1, -0.05, 0.2],
    );
    let params = &one.default_params;
    let x = pack_params(params);
    let indefinite = |eta: &[f64]| {
        let h = exact_inner_hessian(&one, &s, params, eta).expect("recorded");
        let reg = DMatrix::from_fn(3, 3, |a, b| {
            h[(a, b)]
                + if a == b {
                    (1e-6 * h[(a, a)].abs()).max(1e-10)
                } else {
                    0.0
                }
        });
        reg.cholesky().is_none()
    };
    let grid = [-2.5, -1.5, -0.75, 0.75, 1.5, 2.5];
    let mut points = Vec::new();
    for &a in &grid {
        for &b in &grid {
            for &c in &grid {
                points.push([a, b, c]);
            }
        }
    }
    let eta = points
        .into_iter()
        .find(|e| indefinite(e))
        .expect("a point where the exact anchor is indefinite");
    let prod = laplace_prod(&one, &s, params, &eta);
    assert!(
        prod.is_finite() && prod < 1e19,
        "production falls back: {prod}"
    );
    let laplace = Objective::Agq {
        n_agq: 1,
        exact: true,
    };
    assert!(
        subject_adjoint_gradient(&one, &s, params, &x, &eta, laplace).is_none(),
        "off the mode, an indefinite H must decline rather than drop wᵀ∇_η l"
    );
    // The optimizer's salvage for that subject: the held-EBE score, `∂A/∂x` at this fixed
    // `η` — FD of production's objective with `η` held, not reconverged.
    let held = subject_adjoint_held_ebe_score(&one, &s, params, &x, &eta, laplace)
        .expect("the held-EBE score serves the declined subject");
    let mut worst = 0.0f64;
    for k in 0..x.len() {
        let h = 1e-5 * (1.0 + x[k].abs());
        let at = |d: f64| {
            let mut xv = x.clone();
            xv[k] += d;
            let p = crate::estimation::parameterization::unpack_params(&xv, params);
            laplace_prod(&one, &s, &p, &eta)
        };
        let fd = (at(h) - at(-h)) / (2.0 * h);
        assert!(
            held[k].is_finite() && fd.is_finite(),
            "x[{k}]: {} vs {fd}",
            held[k]
        );
        worst = worst.max((held[k] - fd).abs() / (1.0 + fd.abs()));
    }
    eprintln!("held-EBE score vs fixed-η FD: worst rel {worst:.3e}");
    // Measured 4.794e-10 (central FD, h = 1e-5); ~10× headroom.
    assert!(
        worst < 5e-9,
        "held-EBE score off the fixed-η FD by {worst:.3e}"
    );
    // FOCE/FOCEI have their own salvage in the outer optimizer: no held route here.
    assert!(subject_adjoint_held_ebe_score(&one, &s, params, &x, &eta, Objective::Focei).is_none());

    // The population gradient takes that route for the optimizer — the held score, not
    // `2·n_x` reconverged ones — and says so in the outer FD-fallback warning.
    use crate::estimation::outer_optimizer::{outer_fd_fallback_warning, OuterFdDeclineLog};
    use crate::types::{EstimationMethod, FitOptions, Population};
    let population = Population {
        subjects: vec![s.clone()],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let options = FitOptions {
        method: EstimationMethod::Laplace,
        ..FitOptions::default()
    };
    let log = OuterFdDeclineLog::new(1);
    let g = crate::estimation::agq::population_gradient_mixed(
        &one,
        &population,
        params,
        &x,
        &[DVector::from_column_slice(&eta)],
        &[Vec::new()],
        &crate::estimation::parameterization::compute_bounds(params),
        &options,
        false,
        None,
        Some(&log),
    )
    .expect("population gradient");
    for (k, (&gk, &hk)) in g.iter().zip(&held).enumerate() {
        assert_eq!(
            gk.to_bits(),
            (2.0 * hk).to_bits(),
            "x[{k}]: OFV = 2·F of the held score"
        );
    }
    let w = outer_fd_fallback_warning(&one, &population, &log).expect("the salvage is reported");
    assert!(w.contains("used fixed-EBE outer gradients"), "{w}");
}

#[test]
fn laplace_kernel_is_the_production_objective() {
    let one = parse_model_string(ONE_CPT_ORAL).unwrap();
    let s = subject(
        &one,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &TIMES,
        &[0.1, -0.05, 0.2],
    );
    let eta = [0.08, -0.06, 0.1];
    let (k, p) = (
        laplace_kernel::<3, 6>(&one, &s, &one.default_params, &eta),
        laplace_prod(&one, &s, &one.default_params, &eta),
    );
    eprintln!("kernel {k:.15e} production {p:.15e}");
    // Measured worst 0 (bit-identical) on Linux (#2093 review round 2); ~10× headroom.
    // Under the old jittered `½log|H_reg|` production sat 2.6e-6 from this oracle
    // (8.409368521 vs 8.409365935), so the bound also catches the jitter returning.
    assert!((k - p).abs() / (1.0 + p.abs()) < 1e-12, "{k} vs {p}");
}

/// [`fd_oracle`] for an AGQ objective: Richardson FD of production's AGQ objective with
/// the (conditional) EBE re-found at every point.
pub(super) fn fd_oracle_agq(model: &CompiledModel, s: &Subject, n_agq: usize, exact: bool) -> f64 {
    let params = moved_params(model);
    let x = pack_params(&params);
    let eta = ebe_for(model, s, &params, 1e-12, true);
    let adj = subject_adjoint_gradient(
        model,
        s,
        &params,
        &x,
        eta.as_slice(),
        Objective::Agq { n_agq, exact },
    )
    .expect("subject must be in adjoint scope");
    let f_at = |xv: &[f64]| {
        let p = crate::estimation::parameterization::unpack_params(xv, &params);
        let e = ebe_for(model, s, &p, 1e-12, true);
        agq_prod(model, s, &p, e.as_slice(), n_agq, exact)
    };
    let mut worst = 0.0f64;
    for k in 0..x.len() {
        let fd = |h: f64| {
            let (mut xp, mut xm) = (x.clone(), x.clone());
            xp[k] += h;
            xm[k] -= h;
            (f_at(&xp) - f_at(&xm)) / (2.0 * h)
        };
        let rich = (4.0 * fd(1e-3) - fd(2e-3)) / 3.0;
        assert!(
            rich.is_finite() && adj[k].is_finite(),
            "x[{k}] adj={} fd={rich}",
            adj[k]
        );
        let rel = (adj[k] - rich).abs() / (1.0 + rich.abs());
        eprintln!(
            "x[{k}] adjoint={:.10e} reconverged-FD={rich:.10e} rel={rel:.2e}",
            adj[k]
        );
        worst = worst.max(rel);
    }
    worst
}

fn fd_oracle_laplace(model: &CompiledModel, s: &Subject) -> f64 {
    fd_oracle_agq(model, s, 1, true)
}

#[test]
fn agq_adjoint_matches_reconverged_fd_both_anchors() {
    let one = parse_model_string(ONE_CPT_ORAL).unwrap();
    let s = subject(
        &one,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &TIMES,
        &[0.1, -0.05, 0.2],
    );
    for (n_agq, exact) in [(3, true), (1, false), (3, false)] {
        let worst = fd_oracle_agq(&one, &s, n_agq, exact);
        eprintln!("n_agq = {n_agq}, exact = {exact}: worst {worst:.2e}");
        // Measured worst 6.0e-8 on Linux (#2093 review round 2); ~10× headroom.
        assert!(
            worst < 6e-7,
            "n_agq = {n_agq}, exact = {exact}: {worst:.2e}"
        );
    }
}

/// A packed vector of the wrong length is declined (`None`), not read past its end:
/// `unpack_params` indexes it unchecked, and `scope` used to unpack before its own length
/// check, so a short `x` panicked with an index out of bounds (#2294 review). Every entry
/// point goes through `PackedPoint::new`, which checks first.
#[test]
fn a_packed_vector_of_the_wrong_length_declines() {
    let one = parse_model_string(ONE_CPT_ORAL).unwrap();
    let s = subject(
        &one,
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &TIMES,
        &[0.1, -0.05, 0.2],
    );
    let params = moved_params(&one);
    let x = pack_params(&params);
    let eta = ebe_for(&one, &s, &params, 1e-10, true);
    assert!(
        subject_adjoint_gradient(&one, &s, &params, &x, eta.as_slice(), Objective::Focei).is_some()
    );
    let longer: Vec<f64> = x.iter().copied().chain([0.0]).collect();
    for (label, bad) in [("short", &x[..x.len() - 1]), ("long", &longer[..])] {
        for objective in [
            Objective::Focei,
            Objective::Agq {
                n_agq: 1,
                exact: true,
            },
        ] {
            assert!(
                subject_adjoint_gradient(&one, &s, &params, bad, eta.as_slice(), objective)
                    .is_none(),
                "{label} x, {objective:?}"
            );
        }
        assert!(
            subject_adjoint_cov_hessian(&one, &s, &params, bad, eta.as_slice(), Objective::Focei)
                .is_none(),
            "{label} x, covariance Hessian"
        );
        assert!(
            subject_adjoint_eta_dx(&one, &s, &params, bad, eta.as_slice(), true).is_none(),
            "{label} x, mode response"
        );
    }
}

/// A population sweep unpacks the packed point once (`PackedPoint`), not once per subject:
/// the FOCEI population gradient (each subject checked bit for bit against the per-subject
/// entry point), the mode responses, the AGQ population gradient, and an AGQ score context
/// scoring every subject, whose scores must take the exact route. On warfarin's ten
/// subjects. The count is the regression guard: unpacking per subject cloned every θ name
/// per subject, a third of the gradient at 1,317 θ.
#[test]
fn a_population_sweep_unpacks_the_packed_point_once() {
    use crate::estimation::agq::{ScoreRoute, SubjectScoreContext};
    use crate::estimation::sens_outer_gradient::{
        per_subject_packed_gradients, population_eta_dx, subject_packed_gradient,
    };
    let model = crate::parser::model_parser::parse_model_file(std::path::Path::new(
        "examples/warfarin.ferx",
    ))
    .unwrap();
    let pop = crate::read_nonmem_csv(std::path::Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data loads");
    let params = moved_params(&model);
    let x = pack_params(&params);
    let etas: Vec<nalgebra::DVector<f64>> = pop
        .subjects
        .iter()
        .map(|s| ebe_for(&model, s, &params, 1e-10, true))
        .collect();
    let count = |f: &dyn Fn()| super::watch_packed_points(&params, f);
    let grads = std::cell::RefCell::new(Vec::new());
    let n = count(&|| {
        *grads.borrow_mut() =
            per_subject_packed_gradients(&model, &pop, &params, &x, &etas, true, None)
    });
    assert_eq!(n, 1, "FOCEI population gradient");
    for (i, (g, s)) in grads.borrow().iter().zip(&pop.subjects).enumerate() {
        let want = subject_packed_gradient(&model, s, &params, &x, etas[i].as_slice());
        assert_eq!(
            g.as_ref().expect("served"),
            &want.expect("served"),
            "subject {i}"
        );
    }
    let n = count(&|| {
        population_eta_dx(&model, &pop, &params, &x, &etas, true).expect("served");
    });
    assert_eq!(n, 1, "mode responses");
    let n = count(&|| {
        crate::estimation::agq::agq_population_gradient(
            &model,
            &pop,
            &params,
            &params,
            &x,
            &etas,
            &[],
            1,
            crate::types::HessianAnchor::Exact,
        )
        .expect("served");
    });
    assert_eq!(n, 1, "AGQ population gradient");
    let options = crate::types::FitOptions {
        method: crate::types::EstimationMethod::Laplace,
        ..Default::default()
    };
    let bounds = crate::estimation::parameterization::pack_with_bounds(&params).bounds;
    let n = count(&|| {
        let ctx = SubjectScoreContext::new(&model, &params, &x, &options, &bounds, false);
        for (s, eta) in pop.subjects.iter().zip(&etas) {
            let (_, route) = ctx
                .score_routed(s, eta.as_slice(), &[], true)
                .expect("scored");
            assert_eq!(route, ScoreRoute::Exact, "subject {}", s.id);
        }
    });
    assert_eq!(n, 1, "AGQ score context");
}

#[test]
fn laplace_adjoint_matches_reconverged_fd() {
    let one = parse_model_string(ONE_CPT_ORAL).unwrap();
    let s = subject(
        &one,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &TIMES,
        &[0.1, -0.05, 0.2],
    );
    let worst = fd_oracle_laplace(&one, &s);
    // Measured worst 3.7e-7 on Linux (#2093 review round 2); ~10× headroom.
    assert!(worst < 4e-6, "closed form: {worst:.2e}");
    let ode = with_tol(
        parse_model_string(TWO_CPT_INF_COMBINED_BLOCK).unwrap(),
        1e-12,
        1e-14,
    );
    let s = subject(
        &ode,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 50.0, false, 0.0),
        ],
        &TIMES,
        &[0.1, -0.05, 0.08],
    );
    let worst = fd_oracle_laplace(&ode, &s);
    // Measured worst 5.0e-8 on Linux (#2093 review round 2); ~10× headroom.
    assert!(worst < 5e-7, "ODE: {worst:.2e}");
}

// ---- the exact η-Jacobian and inner Hessian that replace production FD ---------------

#[test]
fn exact_eta_jacobian_matches_richardson_fd_of_the_predictor() {
    let check = |m: &CompiledModel, s: &Subject, eta: &[f64], tag: &str| {
        let jac =
            exact_eta_jacobian(m, s, &m.default_params.theta, eta).expect("recorded Jacobian");
        let fd = oracle_jacobian(m, s, &m.default_params.theta, eta);
        let mut worst = 0.0f64;
        for (a, b) in jac.iter().zip(fd.iter()) {
            assert!(
                a.is_finite() && b.is_finite(),
                "{tag}: non-finite {a} vs {b}"
            );
            worst = worst.max((a - b).abs() / (1.0 + b.abs()));
        }
        eprintln!("{tag}: exact vs Richardson FD {worst:.2e}");
        worst
    };
    // An ODE infusion fixture.
    let ode = tight(parse_model_string(TWO_CPT_INF_COMBINED_BLOCK).unwrap());
    let s = subject(
        &ode,
        vec![DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0)],
        &TIMES,
        &[0.1, -0.05, 0.08],
    );
    let w = check(&ode, &s, &[0.05, -0.03, 0.07], "ODE");
    assert!(w < 1e-5, "ODE vs FD: {w:.2e}");
    // A rate-defined infusion under F (the stretched window, a moving boundary), two doses.
    let m = parse_model_string(CF_ORAL_F).unwrap();
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 50.0, false, 0.0),
    ];
    let s = subject(&m, doses, &TIMES, &[0.1, -0.05, 0.2]);
    let w = check(&m, &s, &[0.08, -0.06, 0.1], "infusion under F");
    assert!(w < 1e-5, "infusion under F vs FD: {w:.2e}");
}

#[test]
fn exact_inner_hessian_matches_fd_of_the_production_objective() {
    let one = parse_model_string(ONE_CPT_ORAL).unwrap();
    let s = subject(
        &one,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &TIMES,
        &[0.1, -0.05, 0.2],
    );
    let p = &one.default_params;
    let eta = [0.08, -0.06, 0.1];
    let h = exact_inner_hessian(&one, &s, p, &eta).expect("exact Hessian");
    let l = |e: &[f64]| {
        let mut scratch = crate::pk::EventPkParams::with_capacity_for(&s);
        crate::stats::likelihood::individual_nll_into_with_schedule_cov(
            &one,
            &s,
            &p.theta,
            e,
            &p.omega,
            &p.sigma.values,
            one.residual_cov_with(&p.residual_correlations, &p.theta),
            &mut scratch,
            None,
        )
    };
    let mut worst = 0.0f64;
    for a in 0..3 {
        for b in 0..3 {
            let d = |st: f64| {
                let at = |sa: f64, sb: f64| {
                    let mut e = eta;
                    e[a] += sa;
                    e[b] += sb;
                    l(&e)
                };
                (at(st, st) - at(st, -st) - at(-st, st) + at(-st, -st)) / (4.0 * st * st)
            };
            let rich = (4.0 * d(1e-3) - d(2e-3)) / 3.0;
            worst = worst.max((h[(a, b)] - rich).abs() / (1.0 + rich.abs()));
        }
    }
    eprintln!("exact Hessian vs Richardson FD: {worst:.2e}");
    // Measured worst 3.5e-10 on Linux (#2093 review round 2); ~10× headroom.
    assert!(worst < 4e-9, "{worst:.2e}");
}

/// An `init(...)` that reads the mixture index compiles too (`Op::PushMixNum` reads the class
/// thread-local, as the closure does), so a mixture's dual walk seeds its initial state
/// exactly rather than through the closure's difference quotient: bit-identical to the
/// closure under each class, and the class really changes it.
#[test]
fn a_compiled_init_program_reads_the_mixture_class() {
    use crate::parser::model_parser::MixtureClassGuard;
    let src = ODE_INIT
        .replace(
            "  sigma PROP_ERR ~ 0.1 (sd)\n",
            "  theta P1(0.4, 0.01, 0.99)\n  sigma PROP_ERR ~ 0.1 (sd)\n[mixture]\n  nsub = 2\n  p(1) = P1\n",
        )
        .replace(
            "init(central) = BASE * V",
            "init(central) = if (MIXNUM == 1) BASE * V else 3 * BASE * V",
        );
    let model = parse_model_string(&src).unwrap();
    let ode = model.ode_spec.as_ref().unwrap();
    let prog = ode.rhs_program.as_ref().unwrap();
    let theta = &model.default_params.theta;
    let pk = (model.pk_param_fn)(theta, &[0.1, -0.05], &HashMap::new(), 0.0);
    let mut seen = Vec::new();
    for cls in [1, 2] {
        let _g = MixtureClassGuard::enter(cls);
        let u = prog
            .init_state_g::<f64>(&pk.values, ode.n_states)
            .expect("a MIXNUM init compiles");
        let closure = (ode.init_fn.as_ref().unwrap())(&pk.values);
        assert_eq!(u[0].to_bits(), closure[0].to_bits(), "class {cls}");
        seen.push(u[0]);
    }
    assert!((seen[1] / seen[0] - 3.0).abs() < 1e-12, "{seen:?}");
}

/// An `init(...)` with no compiled program declines the walks at the model scope (the FD
/// fallback serves it), rather than reaching `init_seed` without a seed.
#[test]
fn an_init_without_a_compiled_program_declines_the_walks() {
    let mut model = parse_model_string(ODE_INIT).unwrap();
    assert!(
        crate::sens::ode_provider::ode_walk_model_scope(&model),
        "control"
    );
    model
        .ode_spec
        .as_mut()
        .unwrap()
        .rhs_program
        .as_mut()
        .unwrap()
        .drop_init_program_for_tests();
    assert!(!crate::sens::ode_provider::ode_walk_model_scope(&model));
    assert!(!recorder_in_scope(&model));
}

#[test]
fn compiled_init_program_is_the_closure_and_differentiates_exactly() {
    use crate::sens::rtape::DualT;
    let model = parse_model_string(ODE_INIT).unwrap();
    let ode = model.ode_spec.as_ref().unwrap();
    let prog = ode.rhs_program.as_ref().unwrap();
    let theta = &model.default_params.theta;
    let pk = (model.pk_param_fn)(theta, &[0.1, -0.05], &HashMap::new(), 0.0);
    // The exact path the walks seed from.
    let u = prog
        .init_state_g::<f64>(&pk.values, ode.n_states)
        .expect("compiled init program");
    let closure = (ode.init_fn.as_ref().unwrap())(&pk.values);
    for s in 0..ode.n_states {
        assert_eq!(
            u[s].to_bits(),
            closure[s].to_bits(),
            "state {s}: compiled {} vs closure {}",
            u[s],
            closure[s]
        );
    }
    // ∂init/∂p through a dual, against a Richardson difference of the closure.
    for &slot in &model.pk_indices {
        let duals: Vec<DualT<1, f64>> = pk
            .values
            .iter()
            .enumerate()
            .map(|(i, &v)| DualT {
                v,
                g: [if i == slot { 1.0 } else { 0.0 }],
            })
            .collect();
        let ud = prog
            .init_state_g::<DualT<1, f64>>(&duals, ode.n_states)
            .unwrap();
        let fd = |h: f64| {
            let (mut up, mut dn) = (pk.values.clone(), pk.values.clone());
            up[slot] += h;
            dn[slot] -= h;
            let (a, b) = (
                (ode.init_fn.as_ref().unwrap())(&up),
                (ode.init_fn.as_ref().unwrap())(&dn),
            );
            a.iter()
                .zip(&b)
                .map(|(x, y)| (x - y) / (2.0 * h))
                .collect::<Vec<f64>>()
        };
        let (f1, f2) = (fd(1e-4), fd(2e-4));
        for s in 0..ode.n_states {
            let rich = (4.0 * f1[s] - f2[s]) / 3.0;
            assert!(
                (ud[s].g[0] - rich).abs() <= 1e-8 * (1.0 + rich.abs()),
                "slot {slot}, state {s}: dual {} vs FD {rich}",
                ud[s].g[0]
            );
        }
    }
}

// ---- custom residual magnitude: `power(σ, θ)` and a θ-dependent multiplier -----------

const ORAL_POWER: &str = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(30.0, 1.0, 500.0)
  theta TVKA(1.2, 0.05, 20.0)
  theta RUV_POW(0.8, 0.01, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.06
  omega ETA_KA ~ 0.1
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ power(PROP_ERR, RUV_POW)
"#;

const ORAL_THETA_MULT: &str = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(30.0, 1.0, 500.0)
  theta TVKA(1.2, 0.05, 20.0)
  theta TH_M(1.4, 0.1, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.06
  omega ETA_KA ~ 0.1
  sigma PROP_ERR ~ 0.1 (sd)
  sigma ADD_ERR ~ 0.05 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ combined(PROP_ERR * TH_M, ADD_ERR)
"#;

#[test]
fn custom_residual_magnitude_objective_and_gradients() {
    for src in [ORAL_POWER, ORAL_THETA_MULT] {
        let model = parse_model_string(src).unwrap();
        assert!(
            model
                .ruv_magnitude
                .as_ref()
                .is_some_and(|rm| rm.is_active() && rm.theta_dependent),
            "fixture must carry a θ-dependent magnitude"
        );
        let s = subject(
            &model,
            vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
            ],
            &TIMES,
            &[0.1, -0.05, 0.2],
        );
        let g = objective_gaps::<3>(&model, &s, &model.default_params, &[0.08, -0.06, 0.1]);
        eprintln!("objective gaps {g:?}");
        assert!(g.0 < 1e-12 && g.1 < 1e-12, "objective {g:?}");
        for (label, worst) in [
            ("FOCEI", fd_oracle(&model, &s, true)),
            ("FOCE", fd_oracle(&model, &s, false)),
            ("Laplace", fd_oracle_laplace(&model, &s)),
        ] {
            eprintln!("{label}: worst {worst:.2e}");
            // Measured worst 8.9e-7 on Linux (#2093 review round 2); ~10× headroom.
            assert!(worst < 9e-6, "{label}: {worst:.2e}");
        }
    }
}

// ---- closed-form analytic `init(...)` impulses -----------------------------------------

const CF_INIT: &str = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(30.0, 1.0, 500.0)
  theta TVKA(1.2, 0.05, 20.0)
  theta TVC0(0.4, 0.01, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.06
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV * exp(ETA_V)
  KA = TVKA
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[initial_conditions]
  init(central) = TVC0 * V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

#[test]
fn closed_form_init_impulse_static_and_event_walk() {
    let model = parse_model_string(CF_INIT).unwrap();
    assert!(!model.analytical_init.is_empty());
    // Static superposition.
    let s = subject(
        &model,
        vec![DoseEvent::new(2.0, 100.0, 1, 0.0, false, 0.0)],
        &TIMES,
        &[0.1, -0.05],
    );
    let worst = parity(&model, &s);
    // Measured worst 1.2e-7 on Linux (#2093 review round 2); ~10× headroom.
    assert!(worst < 1.2e-6, "static: {worst:.2e}");
    // Time-varying covariate (the event walk): against the reconverged objective.
    let src = CF_INIT.replace(
        "CL = TVCL * exp(ETA_CL)",
        "CL = TVCL * (WT / 70.0)^0.75 * exp(ETA_CL)",
    );
    let m2 = parse_model_string(&src).unwrap();
    let mut s2 = cf_tvcov_subject(&m2);
    s2.observations =
        crate::pk::compute_predictions_with_tv(&m2, &s2, &m2.default_params.theta, &[0.1, -0.05])
            .iter()
            .enumerate()
            .map(|(j, p)| p * (1.0 + 0.15 * (1.3 * j as f64).sin()))
            .collect();
    let worst = fd_oracle(&m2, &s2, true);
    // Measured worst 1.7e-7 on Linux (#2093 review round 2); ~10× headroom.
    assert!(worst < 1.7e-6, "event walk: {worst:.2e}");
}

/// #527, restored on the adjoint (#2093 review round 2, finding 3): an `init()` amount that
/// evaluates to exactly 0 still carries a gradient. At `TVC0 = 0` (bounds admit it, and the
/// oracle's `moved_params` keeps it at 0) `A₀ = TVC0·V = 0` but `∂A₀/∂TVC0 = V`, so the
/// impulse adds nothing to the value and `V·kernel` to `∂f/∂TVC0`. Skipping the impulse on
/// `A₀ == 0` zeroed that axis while the reconverged FD sees the slope. The mutation this
/// catches is restoring the `a0.val() == 0.0` skip in `provider.rs`.
#[test]
fn closed_form_init_with_a_zero_amount_keeps_its_gradient() {
    let src = CF_INIT.replace(
        "theta TVC0(0.4, 0.01, 10.0)",
        "theta TVC0(0.0, -50.0, 50.0)",
    );
    let model = parse_model_string(&src).unwrap();
    assert_eq!(model.analytical_init.len(), 1);
    assert_eq!(
        moved_params(&model).theta[3],
        0.0,
        "the oracle must sit on A₀ = 0"
    );
    // Dosed at t = 0 and sampled after it, as the #527 fixture was: every prediction then
    // carries the dose, so the init's ±h·V·kernel stays a small perturbation and no row
    // sits on the proportional-error variance floor (a row seeing only `A₀ = 0` would,
    // and the FD would measure the floor, not the slope).
    let s = subject(
        &model,
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &[1.0, 2.0, 4.0, 8.0, 24.0],
        &[0.1, -0.05],
    );
    let worst = fd_oracle(&model, &s, true);
    eprintln!("zero-amount init vs reconverged FD {worst:.2e}");
    // Measured worst 3.6e-8 on Linux; ~10× headroom.
    assert!(worst < 4e-7, "zero-amount init vs FD {worst:.2e}");
}

#[test]
fn padded_width_bucket_matches_reconverged_fd() {
    // Six occasions: stacked width 2 + 6·2 = 14 runs in the 16 bucket, two directions inert.
    let model = tight(parse_model_string(ODE_IOV).unwrap());
    let k = 6;
    let times: Vec<f64> = (0..k)
        .flat_map(|g| [1.0, 4.0, 10.0].map(|t| 12.0 * g as f64 + t))
        .collect();
    let doses: Vec<DoseEvent> = (0..k)
        .map(|g| DoseEvent::new(12.0 * g as f64, 100.0, 1, 0.0, false, 0.0))
        .collect();
    let mut s = subject_shell(doses, &times);
    s.occasions = (0..k).flat_map(|g| [g as u32 + 1; 3]).collect();
    s.dose_occasions = (1..=k as u32).collect();
    let kap: Vec<Vec<f64>> = (0..k)
        .map(|g| vec![0.05 * (g as f64 - 2.5), -0.03 * (g as f64 - 2.0)])
        .collect();
    let preds =
        crate::pk::predict_iov(&model, &s, &model.default_params.theta, &[0.1, -0.05], &kap);
    s.observations = preds
        .iter()
        .enumerate()
        .map(|(j, p)| p * (1.0 + 0.12 * (1.7 * j as f64).sin()))
        .collect();
    assert_eq!(
        crate::stats::stacked_layout::StackedLayout::of(&model, &s).n_stacked(),
        14
    );
    let params = moved_params(&model);
    let x = pack_params(&params);
    let eta = ebe_for(&model, &s, &params, 1e-11, false);
    assert!(
        subject_adjoint_packed_gradient(&model, &s, &params, &x, eta.as_slice(), false).is_some(),
        "the 16 bucket must serve width 14"
    );
    let worst = fd_oracle(&model, &s, false);
    eprintln!("padded bucket vs reconverged FD {worst:.2e}");
    assert!(worst < 1e-5, "{worst:.2e}");
}

#[test]
fn closed_form_iov_with_expression_scale_and_readout_matches_reconverged_fd() {
    // The expression scale reads a κ-carrying V, so each occasion's scale differs.
    let scaled = CF_IOV.replace(
        "[error_model]",
        "[scaling]\n  obs_scale = V / 10.0\n[error_model]",
    );
    let model = parse_model_string(&scaled).unwrap();
    assert!(matches!(
        model.scaling,
        crate::types::ScalingSpec::ExpressionScale { .. }
    ));
    let s = iov_subject(&model);
    for interaction in [true, false] {
        let worst = iov_parity(&model, &s, interaction);
        // Measured worst 1.0e-6 on Linux (#2093 review round 2); ~10× headroom.
        assert!(
            worst < 1e-5,
            "expression scale, interaction = {interaction}: {worst:.2e}"
        );
    }
    // A Form C readout of the closed-form central amount, reading the κ-carrying V.
    let ro = CF_IOV.replace(
        "[error_model]",
        "[scaling]\n  y = central / V + 0.1 * central / (V * V)\n[error_model]",
    );
    let model = parse_model_string(&ro).unwrap();
    assert!(model.analytic_readout.is_some());
    let s = iov_subject(&model);
    for interaction in [true, false] {
        let worst = iov_parity(&model, &s, interaction);
        // Measured worst 2.8e-7 on Linux (#2093 review round 2); ~10× headroom.
        assert!(
            worst < 3e-6,
            "readout, interaction = {interaction}: {worst:.2e}"
        );
    }
}

#[test]
fn closed_form_per_compartment_scale_matches_reconverged_fd() {
    let src = ONE_CPT_ORAL.replace(
        "[error_model]",
        "[scaling]\n  obs_scale[CMT=1] = V / 10.0\n[error_model]",
    );
    let model = parse_model_string(&src).unwrap();
    assert!(
        matches!(model.scaling, crate::types::ScalingSpec::PerCmt(_)),
        "fixture must carry a per-compartment scale"
    );
    let s = subject(
        &model,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &TIMES,
        &[0.1, -0.05, 0.2],
    );
    assert!(in_scope(&model, &s));
    for interaction in [true, false] {
        let worst = fd_oracle(&model, &s, interaction);
        // Measured worst 5.3e-7 on Linux (#2093 review round 2); ~10× headroom.
        assert!(worst < 5e-6, "interaction = {interaction}: {worst:.2e}");
    }
}

// ---- covariance Hessian (`hessian::subject_adjoint_cov_hessian`) -------------------------

/// The adjoint profile Hessian against Richardson differences of the exact gradient with
/// the EBE re-solved at every point — the profile objective's own definition of `d²F̂/dx²`,
/// independent of `Uᵀ∇²ΨU`. Returns the worst relative gap `|H − ref| / (1 + |ref|)` (and
/// `NaN`, the slot a second reference used to fill). Measured 8.8e-8 … 6.6e-6 across the
/// fixtures below (the IOV ones are the loosest: the oracle re-solves a stacked mode), so the
/// shared `1e-5` bound has ≥ 1.5× headroom on the worst and two orders on most. Against the
/// forward assembly, before it was removed: 6.6e-9 (closed form), 7.9e-8 (ODE), 2.0e-6 (IOV,
/// whose third order the forward side central-differenced).
pub(super) fn hessian_gaps(model: &CompiledModel, s: &Subject, interaction: bool) -> (f64, f64) {
    hessian_gaps_for(
        model,
        s,
        if interaction {
            Objective::Focei
        } else {
            Objective::Foce
        },
    )
}

pub(super) fn hessian_gaps_for(
    model: &CompiledModel,
    s: &Subject,
    objective: Objective,
) -> (f64, f64) {
    let interaction = objective != Objective::Foce;
    use crate::estimation::parameterization::unpack_params;
    let params = moved_params(model);
    let x = pack_params(&params);
    let eta = ebe_for(model, s, &params, 1e-12, interaction);
    // On a thread with the fit pool's worker stack (`FIT_RAYON_STACK_SIZE`), where
    // production runs the nested-dual passes; a test thread's default 2 MB is too small.
    let h = std::thread::scope(|scope| {
        std::thread::Builder::new()
            .stack_size(crate::FIT_RAYON_STACK_SIZE)
            .spawn_scoped(scope, || {
                subject_adjoint_cov_hessian(model, s, &params, &x, eta.as_slice(), objective)
            })
            .expect("spawn")
            .join()
            .expect("join")
    })
    .expect("subject must be in adjoint scope");
    let n = x.len();
    assert_eq!((h.nrows(), h.ncols()), (n, n));
    let mut fd = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        let grad_at = |d: f64| {
            let mut xp = x.clone();
            xp[i] += d;
            let pp = unpack_params(&xp, &params);
            let e = ebe_for(model, s, &pp, 1e-12, interaction);
            subject_adjoint_gradient(model, s, &params, &xp, e.as_slice(), objective)
                .expect("gradient in scope")
        };
        // Richardson over central differences at h and h/2: O(h⁴) truncation, and a step
        // large enough that the EBE's convergence noise stays below it.
        let h = 1e-3 * (1.0 + x[i].abs());
        let central = |h: f64| {
            let (gp, gm) = (grad_at(h), grad_at(-h));
            (0..n)
                .map(|j| (gp[j] - gm[j]) / (2.0 * h))
                .collect::<Vec<_>>()
        };
        let (d1, d2) = (central(h), central(0.5 * h));
        for j in 0..n {
            fd[(j, i)] = (4.0 * d2[j] - d1[j]) / 3.0;
        }
    }
    let gap = |a: &DMatrix<f64>, b: &DMatrix<f64>| {
        let mut worst = 0.0f64;
        for (u, v) in a.iter().zip(b.iter()) {
            assert!(
                u.is_finite() && v.is_finite(),
                "non-finite Hessian entry {u} vs {v}"
            );
            worst = worst.max((u - v).abs() / (1.0 + v.abs()));
        }
        worst
    };
    // Exactly symmetric up to round-off (each entry is one mixed second derivative).
    assert!(
        gap(&h, &h.transpose()) < 1e-10,
        "asymmetric: {:.2e}",
        gap(&h, &h.transpose())
    );
    assert!(
        h.iter().map(|v| v.abs()).fold(0.0, f64::max) > 1e-1,
        "a vanishing Hessian tests nothing"
    );
    let vs_fd = gap(&h, &fd);
    eprintln!("hessian: vs reconverged FD of the gradient {vs_fd:.2e}");
    (vs_fd, f64::NAN)
}

/// [`ONE_CPT_ORAL`] with `n` random effects (`n ≥ 3`): the extra ones add to CL, V and KA's
/// exponents in turn, each with its own variance, so every effect is identifiable.
fn one_cpt_oral_n_eta(n: usize) -> String {
    let mut omegas = String::new();
    let mut sums = [String::new(), String::new(), String::new()];
    for k in 0..n {
        omegas.push_str(&format!("  omega E{k} ~ {:.3}\n", 0.02 + 0.01 * k as f64));
        sums[k % 3].push_str(&format!(" + E{k}"));
    }
    format!(
        "[parameters]\n  theta TVCL(0.2, 0.001, 10.0)\n  theta TVV(10.0, 0.1, 500.0)\n  \
         theta TVKA(1.5, 0.01, 50.0)\n{omegas}  sigma PROP_ERR ~ 0.04\n\
         [individual_parameters]\n  CL = TVCL * exp(0{})\n  V = TVV * exp(0{})\n  \
         KA = TVKA * exp(0{})\n[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n\
         [error_model]\n  DV ~ proportional(PROP_ERR)\n",
        sums[0], sums[1], sums[2]
    )
    .replace("exp(0 + ", "exp(")
}

/// The width rule the covariance diagnostics report (`cov_width_declines`) is the one the
/// Hessian dispatch runs on: a nine-effect subject is declined under the exact anchor and
/// served under FOCEI, and each answer is checked against the dispatch itself, so the
/// regularization warning cannot promise a route the dispatch refuses (Copilot on #2093).
/// The PK-space closed form (`pk_space_predictions`) against the generic one it replaces on
/// static closed-form subjects: the FOCEI and FOCE gradients, the η-Jacobian and the exact
/// inner Hessian agree to rounding, on 1-/2-compartment fixtures with 2 to 5 live PK
/// parameters (block Ω, combined error, lag + LTBS, infusions). A Taylor term dropped or
/// misweighted is a gap of order the term itself, far above this bound; the hit counter
/// shows the fast path ran where it applies (second order, or taped), and the Laplace
/// exact anchor (order 3) shows it stood aside.
#[test]
fn the_pk_space_closed_form_is_the_generic_one() {
    use super::{PK_SPACE_FORCE, PK_SPACE_HITS, PK_SPACE_OFF};
    let two_doses = |rate: f64| {
        vec![
            DoseEvent::new(0.0, 100.0, 1, rate, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, rate, false, 0.0),
        ]
    };
    let cases: Vec<(&str, CompiledModel, Vec<DoseEvent>, Vec<f64>)> = vec![
        (
            "1-cpt oral",
            parse_model_string(ONE_CPT_ORAL).unwrap(),
            two_doses(0.0),
            vec![0.1, -0.05, 0.2],
        ),
        (
            "2-cpt iv infusion, block Ω, combined",
            parse_model_string(TWO_CPT_IV_COMBINED).unwrap(),
            two_doses(50.0),
            vec![0.1, -0.05],
        ),
        (
            "1-cpt oral, lag + LTBS",
            parse_model_string(ONE_CPT_ORAL_LAG_LTBS).unwrap(),
            two_doses(0.0),
            vec![0.1, -0.05, 0.1],
        ),
        (
            "1-cpt oral, 5 η",
            parse_model_string(&one_cpt_oral_n_eta(5)).unwrap(),
            two_doses(0.0),
            vec![0.1, -0.05, 0.2, 0.03, -0.02],
        ),
    ];
    let rel = |a: f64, b: f64| (a - b).abs() / (1.0 + b.abs());
    let mut worst = 0.0f64;
    for (label, model, doses, eta_ref) in &cases {
        let s = subject(model, doses.clone(), &TIMES, eta_ref);
        let params = moved_params(model);
        let x = pack_params(&params);
        for interaction in [true, false] {
            let eta = ebe_for(model, &s, &params, 1e-10, interaction);
            let both = |off: bool| {
                PK_SPACE_OFF.with(|c| c.set(off));
                PK_SPACE_FORCE.with(|c| c.set(!off));
                PK_SPACE_HITS.with(|c| c.set(0));
                let g = subject_adjoint_packed_gradient(
                    model,
                    &s,
                    &params,
                    &x,
                    eta.as_slice(),
                    interaction,
                )
                .expect("in scope");
                let j =
                    exact_eta_jacobian(model, &s, &params.theta, eta.as_slice()).expect("jacobian");
                let h = exact_inner_hessian(model, &s, &params, eta.as_slice()).expect("hessian");
                let hits = PK_SPACE_HITS.with(|c| c.get());
                PK_SPACE_OFF.with(|c| c.set(false));
                PK_SPACE_FORCE.with(|c| c.set(false));
                (g, j, h, hits)
            };
            let (g0, j0, h0, off_hits) = both(true);
            let (g1, j1, h1, hits) = both(false);
            assert_eq!(off_hits, 0, "{label}: the switch must turn the path off");
            // The taped gradient recording and the inner Hessian's second-order one take
            // the PK-space path; the η-Jacobian (first order, tape-free) keeps the generic
            // path by design, so it is compared here but not counted.
            assert!(
                hits >= 2,
                "{label}: the PK-space path did not run ({hits} hits)"
            );
            for (a, b) in g1
                .iter()
                .zip(&g0)
                .chain(j1.iter().zip(j0.iter()))
                .chain(h1.iter().zip(h0.iter()))
            {
                assert!(a.is_finite() && b.is_finite(), "{label}");
                worst = worst.max(rel(*a, *b));
            }
            assert!(
                g0.iter().map(|v| v.abs()).fold(0.0, f64::max) > 1e-2,
                "{label}: a vanishing gradient tests nothing"
            );
        }
    }
    eprintln!("PK-space vs generic closed form: worst relative gap {worst:.2e}");
    // Measured 9.6e-15: rounding (the polynomial and the closed form associate
    // differently); 100× headroom, still ten orders below a misweighted Taylor term.
    assert!(worst < 1e-12, "PK-space vs generic: {worst:.2e}");
    // The exact anchor's recording tapes the η-Hessian (order 3), and the covariance
    // step's nested duals observe three orders (four under the exact anchor): each takes
    // the PK-space path at its own order now, and must agree with the generic one.
    let (_, model, doses, eta_ref) = &cases[0];
    let s = subject(model, doses.clone(), &TIMES, eta_ref);
    let worst_hi = pk_space_twin_high_order(model, &s, "1-cpt oral");
    eprintln!("PK-space vs generic closed form, orders 3–4: worst relative gap {worst_hi:.2e}");
    assert!(worst_hi < 1e-10, "orders 3–4: {worst_hi:.2e}");
}

/// The Laplace gradient (its taped exact-anchor recording, order 3) and the FOCEI and
/// Laplace covariance Hessians (the nested duals, orders 3 and 4) with the PK-space path
/// on and off: the worst relative gap, after asserting the path ran each time.
fn pk_space_twin_high_order(model: &CompiledModel, s: &Subject, label: &str) -> f64 {
    use super::{PK_SPACE_FORCE, PK_SPACE_HITS, PK_SPACE_OFF};
    let params = moved_params(model);
    let x = pack_params(&params);
    let laplace = Objective::Agq {
        n_agq: 1,
        exact: true,
    };
    let rel = |a: f64, b: f64| (a - b).abs() / (1.0 + b.abs());
    let mut worst = 0.0f64;
    let eta = ebe_for(model, s, &params, 1e-10, true);
    let run = |off: bool| {
        PK_SPACE_OFF.with(|c| c.set(off));
        PK_SPACE_FORCE.with(|c| c.set(!off));
        PK_SPACE_HITS.with(|c| c.set(0));
        let g = subject_adjoint_gradient(model, s, &params, &x, eta.as_slice(), laplace)
            .expect("in scope");
        let g_hits = PK_SPACE_HITS.with(|c| c.get());
        let hf =
            subject_adjoint_cov_hessian(model, s, &params, &x, eta.as_slice(), Objective::Focei)
                .expect("FOCEI covariance Hessian");
        let hl = subject_adjoint_cov_hessian(model, s, &params, &x, eta.as_slice(), laplace)
            .expect("Laplace covariance Hessian");
        let hits = PK_SPACE_HITS.with(|c| c.get());
        PK_SPACE_OFF.with(|c| c.set(false));
        PK_SPACE_FORCE.with(|c| c.set(false));
        (g, hf, hl, g_hits, hits)
    };
    let (g0, hf0, hl0, _, off_hits) = run(true);
    let (g1, hf1, hl1, g_hits, hits) = run(false);
    assert_eq!(off_hits, 0, "{label}: the switch must turn the path off");
    // The canonical pass (order 2) and the taped exact-anchor recording (order 3).
    assert!(
        g_hits >= 2,
        "{label}: the exact anchor's recording did not lift ({g_hits})"
    );
    assert!(
        hits > g_hits,
        "{label}: the covariance passes did not lift ({hits})"
    );
    for (a, b) in g1
        .iter()
        .zip(&g0)
        .chain(hf1.iter().zip(hf0.iter()))
        .chain(hl1.iter().zip(hl0.iter()))
    {
        assert!(a.is_finite() && b.is_finite(), "{label}");
        worst = worst.max(rel(*a, *b));
    }
    assert!(
        hf0.iter().map(|v| v.abs()).fold(0.0, f64::max) > 1e-2,
        "{label}: a vanishing Hessian tests nothing"
    );
    worst
}

/// The ODE engine's PK-space path against its generic recording: the FOCEI and FOCE
/// gradients, the η-Jacobian and inner Hessian (order ≤ 2), then the Laplace gradient and
/// both covariance Hessians (orders 3 and 4), on static ODE subjects — RK45 and Rosenbrock,
/// multi-dose infusion, four η, a transit chain.
#[test]
fn the_pk_space_ode_walk_is_the_generic_one() {
    use super::{PK_SPACE_FORCE, PK_SPACE_HITS, PK_SPACE_OFF};
    let mut rodas = tight(parse_model_string(TWO_CPT_INF_COMBINED_BLOCK).unwrap());
    rodas.ode_spec.as_mut().unwrap().solver_opts.method = crate::ode::solver::OdeMethod::Rodas5P;
    let infusions = vec![
        DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0),
        DoseEvent::new(8.0, 100.0, 1, 50.0, false, 0.0),
    ];
    let bolus = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let cases: Vec<(&str, CompiledModel, Vec<DoseEvent>, Vec<f64>)> = vec![
        (
            "one_cpt_oral_ode",
            tight(parse_model_string(ONE_CPT_ORAL_ODE).unwrap()),
            bolus.clone(),
            vec![0.1, -0.05],
        ),
        (
            "two_cpt_infusion",
            tight(parse_model_string(TWO_CPT_INF_COMBINED_BLOCK).unwrap()),
            infusions.clone(),
            vec![0.1, -0.05, 0.08],
        ),
        ("two_cpt_rodas5p", rodas, infusions, vec![0.1, -0.05, 0.08]),
        (
            "additive_four_eta",
            tight(parse_model_string(ADDITIVE_FOUR_ETA).unwrap()),
            bolus.clone(),
            vec![0.1, -0.05, 0.2, -0.1],
        ),
        (
            "chain_4",
            tight(parse_model_string(&chain_model(4)).unwrap()),
            bolus,
            vec![0.1, -0.05, 0.1],
        ),
    ];
    let rel = |a: f64, b: f64| (a - b).abs() / (1.0 + b.abs());
    let (mut worst2, mut worst_hi) = (0.0f64, 0.0f64);
    for (label, model, doses, eta_ref) in &cases {
        let s = subject(model, doses.clone(), &TIMES, eta_ref);
        let params = moved_params(model);
        let x = pack_params(&params);
        for interaction in [true, false] {
            let eta = ebe_for(model, &s, &params, 1e-10, interaction);
            let both = |off: bool| {
                PK_SPACE_OFF.with(|c| c.set(off));
                PK_SPACE_FORCE.with(|c| c.set(!off));
                PK_SPACE_HITS.with(|c| c.set(0));
                let g = subject_adjoint_packed_gradient(
                    model,
                    &s,
                    &params,
                    &x,
                    eta.as_slice(),
                    interaction,
                )
                .expect("in scope");
                let j =
                    exact_eta_jacobian(model, &s, &params.theta, eta.as_slice()).expect("jacobian");
                let h = exact_inner_hessian(model, &s, &params, eta.as_slice()).expect("hessian");
                let hits = PK_SPACE_HITS.with(|c| c.get());
                PK_SPACE_OFF.with(|c| c.set(false));
                PK_SPACE_FORCE.with(|c| c.set(false));
                (g, j, h, hits)
            };
            let (g0, j0, h0, _) = both(true);
            let (g1, j1, h1, hits) = both(false);
            assert!(
                hits >= 2,
                "{label}: the PK-space path did not run ({hits} hits)"
            );
            for (a, b) in g1
                .iter()
                .zip(&g0)
                .chain(j1.iter().zip(j0.iter()))
                .chain(h1.iter().zip(h0.iter()))
            {
                assert!(a.is_finite() && b.is_finite(), "{label}");
                worst2 = worst2.max(rel(*a, *b));
            }
        }
        worst_hi = worst_hi.max(pk_space_twin_high_order(model, &s, label));
        eprintln!("{label}: worst so far {worst2:.2e} (order 2), {worst_hi:.2e} (orders 3–4)");
    }
    assert!(
        worst2 < 1e-10,
        "PK-space vs generic ODE, order 2: {worst2:.2e}"
    );
    assert!(
        worst_hi < 1e-9,
        "PK-space vs generic ODE, orders 3–4: {worst_hi:.2e}"
    );
}

#[test]
fn the_covariance_width_rule_is_the_dispatchs() {
    let laplace = Objective::Agq {
        n_agq: 1,
        exact: true,
    };
    let doses = || {
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ]
    };
    let three = parse_model_string(ONE_CPT_ORAL).unwrap();
    let nine = parse_model_string(&one_cpt_oral_n_eta(9)).unwrap();
    assert_eq!(nine.n_eta, 9);
    let s3 = subject(&three, doses(), &TIMES, &[0.1, -0.05, 0.2]);
    let s9 = subject(&nine, doses(), &TIMES, &[0.05; 9]);
    let served = |m: &CompiledModel, s: &Subject, objective: Objective| {
        let params = moved_params(m);
        let x = pack_params(&params);
        let eta = ebe_for(m, s, &params, 1e-10, true);
        std::thread::scope(|scope| {
            std::thread::Builder::new()
                .stack_size(crate::FIT_RAYON_STACK_SIZE)
                .spawn_scoped(scope, || {
                    subject_adjoint_cov_hessian(m, s, &params, &x, eta.as_slice(), objective)
                        .is_some()
                })
                .expect("spawn")
                .join()
                .expect("join")
        })
    };
    for (label, m, s, objective, declines) in [
        ("3 η, Laplace", &three, &s3, laplace, false),
        ("9 η, Laplace", &nine, &s9, laplace, true),
        ("9 η, FOCEI", &nine, &s9, Objective::Focei, false),
    ] {
        assert_eq!(cov_width_declines(m, s, objective), declines, "{label}");
        assert_eq!(
            served(m, s, objective),
            !declines,
            "{label}: the dispatch disagrees"
        );
    }
    assert_eq!(super::hessian::EXACT_ANCHOR_HESSIAN_WIDTH, 8);
}

/// `f` on a fit pool worker's stack (`FIT_RAYON_STACK_SIZE`), as production runs the
/// covariance recorders.
fn with_fit_stack<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .stack_size(crate::FIT_RAYON_STACK_SIZE)
            .spawn_scoped(scope, f)
            .expect("spawn")
            .join()
            .expect("join")
    })
}

/// `subject_adjoint_cov_diag_at` is the full Hessian's diagonal along `axes`, in `axes`
/// order: on a five-effect subject (bucket 6) under FOCEI, FOCE and Laplace's exact anchor
/// (bucket 8), with the axes out of order and spanning θ, Ω and σ. Each entry is the same
/// lane of the same sweep on either route, read along its own direction only, so the
/// diagonal is the full Hessian's to the bit: measured worst 0 on Linux. An axis past `x`
/// is refused.
#[test]
fn cov_diag_on_axes_is_the_full_hessians_diagonal() {
    let five = parse_model_string(&one_cpt_oral_n_eta(5)).unwrap();
    let s = subject(
        &five,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &TIMES,
        &[0.1, -0.05, 0.2, 0.03, -0.02],
    );
    let params = moved_params(&five);
    let x = pack_params(&params);
    let pt = PackedPoint::new(&params, &x).expect("packed point");
    let axes = [x.len() - 1, 0, 4, 2];
    let mut worst = 0.0f64;
    for objective in [
        Objective::Focei,
        Objective::Foce,
        Objective::Agq {
            n_agq: 1,
            exact: true,
        },
    ] {
        let interaction = !matches!(objective, Objective::Foce);
        let eta = ebe_for(&five, &s, &params, 1e-10, interaction);
        let (full, diag) = with_fit_stack(|| {
            (
                subject_adjoint_cov_hessian(&five, &s, &params, &x, eta.as_slice(), objective)
                    .expect("full Hessian served"),
                subject_adjoint_cov_diag_at(&five, &s, &pt, eta.as_slice(), objective, &axes)
                    .expect("diagonal served"),
            )
        });
        assert_eq!(diag.len(), axes.len());
        for (&a, &got) in axes.iter().zip(&diag) {
            let want = full[(a, a)];
            assert!(got.is_finite(), "{objective:?} [{a}] {got}");
            worst = worst.max((got - want).abs() / (1.0 + want.abs()));
        }
    }
    eprintln!("axes diagonal vs full Hessian: worst rel {worst:.2e}");
    assert!(worst == 0.0, "{worst:.2e}");
    let eta = ebe_for(&five, &s, &params, 1e-10, true);
    let past = [x.len()];
    assert!(
        subject_adjoint_cov_diag_at(&five, &s, &pt, eta.as_slice(), Objective::Focei, &past)
            .is_none(),
        "an axis past x is refused"
    );
}

/// `subject_adjoint_cov_block_at` is the full Hessian's dense sub-block over `axes`, rows and
/// columns in `axes` order (#2109): same subject, objectives and out-of-order axes as the
/// diagonal's test, plus the off-diagonal pairs the diagonal never reads. Each entry is the
/// same lane of a sweep along the same direction on either route, so the block is the full
/// Hessian's to the bit: measured worst 0 (macOS arm64). An axis past `x` is refused. Dies
/// when the block is read in sorted rather than `axes` order. Not when the η tangent is
/// dropped: the full Hessian shares that code, so the profile's C2 and C4 are its oracle.
#[test]
fn cov_block_on_axes_is_the_full_hessians_block() {
    let five = parse_model_string(&one_cpt_oral_n_eta(5)).unwrap();
    let s = subject(
        &five,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &TIMES,
        &[0.1, -0.05, 0.2, 0.03, -0.02],
    );
    let params = moved_params(&five);
    let x = pack_params(&params);
    let pt = PackedPoint::new(&params, &x).expect("packed point");
    let axes = [x.len() - 1, 0, 4, 2, 1];
    let mut worst = 0.0f64;
    for objective in [
        Objective::Focei,
        Objective::Foce,
        Objective::Agq {
            n_agq: 1,
            exact: true,
        },
    ] {
        let interaction = !matches!(objective, Objective::Foce);
        let eta = ebe_for(&five, &s, &params, 1e-10, interaction);
        let (full, block) = with_fit_stack(|| {
            (
                subject_adjoint_cov_hessian(&five, &s, &params, &x, eta.as_slice(), objective)
                    .expect("full Hessian served"),
                subject_adjoint_cov_block_at(&five, &s, &pt, eta.as_slice(), objective, &axes)
                    .expect("block served"),
            )
        });
        assert_eq!(block.shape(), (axes.len(), axes.len()));
        for (r, &a) in axes.iter().enumerate() {
            for (c, &b) in axes.iter().enumerate() {
                let (got, want) = (block[(r, c)], full[(a, b)]);
                assert!(got.is_finite(), "{objective:?} [{a}, {b}] {got}");
                worst = worst.max((got - want).abs() / (1.0 + want.abs()));
            }
        }
    }
    eprintln!("axes block vs full Hessian: worst rel {worst:.2e}");
    assert!(worst == 0.0, "{worst:.2e}");
    let eta = ebe_for(&five, &s, &params, 1e-10, true);
    assert!(
        subject_adjoint_cov_block_at(
            &five,
            &s,
            &pt,
            eta.as_slice(),
            Objective::Focei,
            &[0, x.len()]
        )
        .is_none(),
        "an axis past x is refused"
    );
}

/// A block on a `[covariate_nn]` subject goes along its own packed axes, not through the
/// networks' reduced basis (`NnReduction`), and is the full Hessian's sub-block there: axes
/// with a weight from each layer, a θ and a σ, under FOCEI and FOCE. A profiled fit can carry
/// a network (`check_profiled_levels` does not refuse one), so this is the route its
/// covariance step reads (#2417 review row 5). The counter shows the reduced basis did not
/// run for the block. Dies when a block reaches `NnReduction::hessian` (its `unreachable!`).
#[cfg(feature = "nn")]
#[test]
fn cov_block_on_a_covariate_nn_subject_is_the_full_hessians_block() {
    use super::hessian::{NN_REDUCED_HITS, NN_REDUCE_OFF};
    let (model, s) = covariate_nn_fixture();
    let params = moved_params(&model);
    let x = pack_params(&params);
    let pt = PackedPoint::new(&params, &x).expect("packed point");
    let w = model.covariate_nns[0].weights_offset;
    let n_w = model.covariate_nns[0].mapper.mlp().n_weights();
    let axes = [w, w + n_w - 1, 0, x.len() - 1];
    let mut worst = 0.0f64;
    for (objective, interaction) in [(Objective::Focei, true), (Objective::Foce, false)] {
        let eta = ebe_for(&model, &s, &params, 1e-12, interaction);
        let (full, block, hits) = with_fit_stack(|| {
            NN_REDUCE_OFF.with(|c| c.set(true));
            let full =
                subject_adjoint_cov_hessian(&model, &s, &params, &x, eta.as_slice(), objective);
            NN_REDUCE_OFF.with(|c| c.set(false));
            NN_REDUCED_HITS.with(|c| c.set(0));
            let block =
                subject_adjoint_cov_block_at(&model, &s, &pt, eta.as_slice(), objective, &axes);
            (full, block, NN_REDUCED_HITS.with(|c| c.get()))
        });
        let (full, block) = (full.expect("full Hessian"), block.expect("block"));
        assert_eq!(hits, 0, "{objective:?}: the block ran in the reduced basis");
        for (r, &a) in axes.iter().enumerate() {
            for (c, &b) in axes.iter().enumerate() {
                let (got, want) = (block[(r, c)], full[(a, b)]);
                assert!(got.is_finite(), "{objective:?} [{a}, {b}] {got}");
                worst = worst.max((got - want).abs() / (1.0 + want.abs()));
            }
        }
    }
    eprintln!("covariate NN: block vs full Hessian {worst:.2e}");
    assert!(worst == 0.0, "{worst:.2e}");
}

/// The diagonal on a `[covariate_nn]` subject goes through the networks' reduced basis
/// (`NnReduction`, as the full covariance Hessian does) and equals the diagonal of the
/// Hessian assembled along every packed axis, under FOCEI, FOCE and Laplace, on axes
/// covering a network weight from each layer, a θ and a σ. The counter shows the reduced
/// basis ran. Without it a candidate set of network weights (the #2159 pre-flight's
/// motivating case) would sweep one direction per weight.
#[cfg(feature = "nn")]
#[test]
fn cov_diag_on_a_covariate_nn_subject_runs_in_the_reduced_basis() {
    use super::hessian::{NN_REDUCED_HITS, NN_REDUCE_OFF};
    let (model, s) = covariate_nn_fixture();
    let params = moved_params(&model);
    let x = pack_params(&params);
    let pt = PackedPoint::new(&params, &x).expect("packed point");
    let w = model.covariate_nns[0].weights_offset;
    let n_w = model.covariate_nns[0].mapper.mlp().n_weights();
    let axes = [w, w + n_w - 1, w + n_w / 2, 0, x.len() - 1];
    let mut worst = 0.0f64;
    for (objective, interaction) in [
        (Objective::Focei, true),
        (Objective::Foce, false),
        (
            Objective::Agq {
                n_agq: 1,
                exact: true,
            },
            true,
        ),
    ] {
        let eta = ebe_for(&model, &s, &params, 1e-12, interaction);
        let (full, diag, hits) = with_fit_stack(|| {
            NN_REDUCE_OFF.with(|c| c.set(true));
            let full =
                subject_adjoint_cov_hessian(&model, &s, &params, &x, eta.as_slice(), objective);
            NN_REDUCE_OFF.with(|c| c.set(false));
            NN_REDUCED_HITS.with(|c| c.set(0));
            let diag =
                subject_adjoint_cov_diag_at(&model, &s, &pt, eta.as_slice(), objective, &axes);
            (full, diag, NN_REDUCED_HITS.with(|c| c.get()))
        });
        let (full, diag) = (full.expect("full Hessian"), diag.expect("diagonal"));
        assert_eq!(hits, 1, "{objective:?}: the reduced basis did not run");
        assert!(
            diag[..3].iter().any(|v| v.abs() > 1e-3),
            "{objective:?}: a vanishing weight curvature tests nothing"
        );
        for (&a, &got) in axes.iter().zip(&diag) {
            let want = full[(a, a)];
            assert!(got.is_finite(), "{objective:?} [{a}] {got}");
            worst = worst.max((got - want).abs() / (1.0 + want.abs()));
        }
    }
    eprintln!("covariate NN: reduced diagonal vs full Hessian {worst:.2e}");
    // Measured worst 4.4e-15 on Linux (the reduced basis sums in a different order); the
    // bound leaves ~20× for a platform's summation.
    assert!(
        worst < 1e-13,
        "reduced diagonal vs full Hessian {worst:.2e}"
    );
}

/// The Hessian buckets past the first: a five-effect subject runs FOCEI / FOCE on the
/// 6-wide bucket and Laplace's exact anchor on its 8-wide one (padded), each against the
/// reconverged FD of the adjoint gradient.
#[test]
fn cov_hessian_matches_profile_fd_on_the_wider_buckets() {
    let five = parse_model_string(&one_cpt_oral_n_eta(5)).unwrap();
    assert_eq!(five.n_eta, 5);
    let s = subject(
        &five,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &TIMES,
        &[0.1, -0.05, 0.2, 0.03, -0.02],
    );
    for (objective, name) in [
        (Objective::Focei, "FOCEI (bucket 6)"),
        (Objective::Foce, "FOCE (bucket 6)"),
        (
            Objective::Agq {
                n_agq: 1,
                exact: true,
            },
            "Laplace (bucket 8)",
        ),
    ] {
        let (fd, _) = hessian_gaps_for(&five, &s, objective);
        // Measured 2.0e-8 / 9.6e-8 / 6.6e-8 (FOCEI / FOCE / Laplace); 10× headroom.
        assert!(fd < 1e-6, "{name} vs FD {fd:.2e}");
    }
}

#[test]
fn cov_hessian_matches_profile_fd_closed_form_focei_and_foce() {
    let one = parse_model_string(ONE_CPT_ORAL).unwrap();
    let s = subject(
        &one,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &TIMES,
        &[0.1, -0.05, 0.2],
    );
    let (fd, _) = hessian_gaps(&one, &s, true);
    assert!(fd < 1e-5, "FOCEI vs FD {fd:.2e}");
    // Proportional error under FOCE: the frozen-variance mode moves with θ through f(η = 0).
    let (fd, _) = hessian_gaps(&one, &s, false);
    assert!(fd < 1e-5, "FOCE vs FD {fd:.2e}");
}

/// `psi_along` under a tape budget spends its directions over narrower passes
/// (`PASS_WIDTH_CAP` stands in for a large tape). Each column must still land on its own
/// direction: width 3 over 7 packed axes leaves a ragged last pass, so an offset taken from
/// the pass's own width rather than the planned one misplaces columns. Measured: bit-identical
/// at widths 1 and 3 (a column's arithmetic does not see the other tangents); a misplaced
/// column moves the matrix by O(1) of its largest entry.
#[test]
fn cov_hessian_narrow_passes_equal_the_full_width_ones() {
    use super::hessian::PASS_WIDTH_CAP;
    let one = parse_model_string(ONE_CPT_ORAL).unwrap();
    let s = subject(
        &one,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &TIMES,
        &[0.1, -0.05, 0.2],
    );
    let params = moved_params(&one);
    let x = pack_params(&params);
    assert_eq!(
        x.len(),
        7,
        "the fixture must leave a ragged last pass at width 3"
    );
    let eta = ebe_for(&one, &s, &params, 1e-12, true);
    let run = |cap: Option<usize>| {
        std::thread::scope(|scope| {
            std::thread::Builder::new()
                .stack_size(crate::FIT_RAYON_STACK_SIZE)
                .spawn_scoped(scope, || {
                    PASS_WIDTH_CAP.with(|c| c.set(cap));
                    subject_adjoint_cov_hessian(
                        &one,
                        &s,
                        &params,
                        &x,
                        eta.as_slice(),
                        Objective::Focei,
                    )
                })
                .expect("spawn")
                .join()
                .expect("join")
        })
        .expect("in scope")
    };
    let full = run(None);
    for cap in [1, 3] {
        let narrow = run(Some(cap));
        let scale = full.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let mut worst = 0.0f64;
        for (a, b) in narrow.iter().zip(full.iter()) {
            assert!(
                a.is_finite() && b.is_finite(),
                "width {cap}: non-finite entry"
            );
            worst = worst.max((a - b).abs() / scale);
        }
        eprintln!("cov Hessian at width {cap} vs 8: {worst:.2e}");
        assert!(worst < 1e-12, "width {cap} vs 8: {worst:.2e}");
    }
}

/// The tape budget's width rule: the full `K` on a small tape, fewer as one pass along none
/// approaches the budget, never zero.
#[test]
fn cov_hessian_pass_width_follows_the_tape_budget() {
    use super::hessian::{pass_width, TAPE_NODE_BUDGET};
    // A small tape takes the full `K = 4` directions per pass (#1564: 4 measured faster
    // than 8 on every bench fixture).
    assert_eq!(pass_width(0), 4);
    assert_eq!(pass_width(1000), 4);
    // 5 → (5 − 1)·4/5 = 3 directions: ~1 + 1.25·3 ≈ 4.75 × nodes, inside the budget.
    assert_eq!(pass_width(TAPE_NODE_BUDGET / 5), 3);
    assert_eq!(pass_width(TAPE_NODE_BUDGET), 1);
    assert_eq!(pass_width(usize::MAX), 1);
}

#[test]
fn cov_hessian_matches_profile_fd_ode_block_omega_combined() {
    let ode = with_tol(
        parse_model_string(TWO_CPT_INF_COMBINED_BLOCK).unwrap(),
        1e-12,
        1e-14,
    );
    let s = subject(
        &ode,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 50.0, false, 0.0),
        ],
        &TIMES,
        &[0.1, -0.05, 0.08],
    );
    let (fd, _) = hessian_gaps(&ode, &s, true);
    assert!(fd < 1e-5, "FOCEI vs FD {fd:.2e}");
    let (fd, _) = hessian_gaps(&ode, &s, false);
    assert!(fd < 1e-5, "FOCE vs FD {fd:.2e}");
}

/// #2101: the exact R-matrix of a nested-level subject whose residuals a `correlation
/// ar1(RHO) by ARM` line couples across rows (#1082 PR5), FOCEI and FOCE, against the
/// reconverged FD of the adjoint gradient. ρ is a θ, so its row and column are in the
/// matrix; the subject's blocks are cross-row (asserted), so the dense `R` is live.
///
/// FOCE and FOCEI coincide on this additive-error fixture (Woodbury), so one stage runs.
/// Slow (≈4 min: the Richardson oracles re-solve the nested EBE at every stencil point).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn cov_hessian_matches_profile_fd_nested_residual_correlation() {
    use crate::stats::nested_levels::test_helpers as fx;
    let src = fx::algebraic()
        .replace(
            "  sigma ADD ~ 0.5\n",
            "  sigma ADD ~ 0.5\n  theta RHO(0.6, -0.9, 0.9)\n",
        )
        .replace(
            "  DV ~ additive(ADD)\n",
            "  DV ~ additive(ADD)\n  correlation ar1(RHO) by ARM\n",
        );
    let (m, mut pop) = fx::prepared(&src, 1);
    crate::stats::residual_error::assign_residual_groups(&m, &mut pop);
    let s = &pop.subjects[0];
    assert!(crate::estimation::inner_optimizer::iov_inner_residual_correlation(&m, s));
    for interaction in [true] {
        // The gradient first, against the Richardson oracle (EBE re-solved to 1e-12).
        let g = fd_oracle(&m, s, interaction);
        eprintln!("nested + ar1, interaction = {interaction}: gradient vs FD {g:.2e}");
        let (fd, _) = hessian_gaps(&m, s, interaction);
        eprintln!("nested + ar1, interaction = {interaction}: Hessian vs FD {fd:.2e}");
        // Measured (FOCEI): gradient 3.5e-6, Hessian 1.9e-5. That is the oracles' floor on
        // this nested fixture, not the correlation's: the same subject without the line —
        // a path NONMEM anchors — measures 1.6e-6 and 1.65e-5. ~3× headroom on each.
        assert!(
            g < 1e-5,
            "interaction = {interaction}: gradient vs FD {g:.2e}"
        );
        assert!(
            fd < 5e-5,
            "interaction = {interaction}: Hessian vs FD {fd:.2e}"
        );
    }
}

#[test]
fn cov_hessian_matches_profile_fd_iov() {
    let model = tight(parse_model_string(ODE_IOV).unwrap());
    let s = iov_subject(&model);
    let (fd, _) = hessian_gaps(&model, &s, true);
    assert!(fd < 1e-5, "IOV FOCEI vs FD {fd:.2e}");
    let (fd, _) = hessian_gaps(&model, &s, false);
    assert!(fd < 1e-5, "IOV FOCE vs FD {fd:.2e}");
}

#[test]
fn cov_hessian_matches_profile_fd_laplace_and_agq() {
    let one = parse_model_string(ONE_CPT_ORAL).unwrap();
    let s = subject(
        &one,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &TIMES,
        &[0.1, -0.05, 0.2],
    );
    for (objective, name) in [
        (
            Objective::Agq {
                n_agq: 1,
                exact: true,
            },
            "Laplace",
        ),
        (
            Objective::Agq {
                n_agq: 3,
                exact: false,
            },
            "AGQ-3 H̃",
        ),
        (
            Objective::Agq {
                n_agq: 3,
                exact: true,
            },
            "AGQ-3 exact",
        ),
    ] {
        let (fd, _) = hessian_gaps_for(&one, &s, objective);
        assert!(fd < 1e-5, "{name} vs FD {fd:.2e}");
    }
}

// ---- compartment-free, covariate-NN and mixture-class recordings -----------------------

const ALGEBRAIC: &str = r#"
[parameters]
  theta TVE0(10.0, 0.1, 100.0)
  theta TVEMAX(6.0, 0.1, 100.0)
  theta TVET50(2.0, 0.01, 100.0)
  omega ETA_E0 ~ 0.09
  omega ETA_EMAX ~ 0.16
  sigma PROP ~ 0.04 (sd)
[individual_parameters]
  E0   = TVE0 * exp(ETA_E0)
  EMAX = TVEMAX * exp(ETA_EMAX)
  ET50 = TVET50
[structural_model]
  y = E0 - EMAX * TIME / (ET50 + TIME)
[error_model]
  DV ~ proportional(PROP)
"#;

#[test]
fn algebraic_model_matches_reconverged_fd() {
    let model = crate::parser::model_parser::parse_full_model(ALGEBRAIC)
        .unwrap()
        .model;
    assert!(model.is_algebraic());
    let s = subject(&model, vec![], &TIMES, &[0.1, -0.2]);
    assert!(
        in_scope(&model, &s),
        "the compartment-free recorder must serve the subject"
    );
    let fd = fd_oracle(&model, &s, true);
    eprintln!("algebraic FOCEI: vs reconverged FD {fd:.2e}");
    assert!(fd < 1e-5, "FOCEI: FD {fd:.2e}");
    let fd = fd_oracle(&model, &s, false);
    assert!(fd < FOCE_FD_TOL, "FOCE vs FD {fd:.2e}");
}

#[test]
fn mixture_class_gradient_matches_reconverged_fd_under_the_class_guard() {
    let src = r#"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(4.0, 0.01, 100.0)
  theta TVV(20.0, 0.1, 500.0)
  theta P1(0.6, 0.01, 0.99)
  omega ETA_CL ~ 0.09
  sigma EPS ~ 0.04
[mixture]
  nsub = 2
  p(1) = P1
[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(EPS)
"#;
    let model = parse_model_string(src).unwrap();
    let s = subject(
        &model,
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &TIMES,
        &[0.1],
    );
    let params = crate::estimation::mixture::class_params(&moved_params(&model), 1);
    let x = pack_params(&params);
    // Class 2 reads TVCL2: the gradient on TVCL1 must vanish and TVCL2's must not.
    let _class = crate::parser::model_parser::MixtureClassGuard::enter(2);
    let eta = ebe_for(&model, &s, &params, 1e-11, true);
    let adj = subject_adjoint_packed_gradient(&model, &s, &params, &x, eta.as_slice(), true)
        .expect("mixture class in adjoint scope");
    assert!(
        adj[0] == 0.0 && adj[1].abs() > 1e-3,
        "class 2 reads TVCL2 only: {adj:?}"
    );
    // Against the reconverged class-2 objective (the guard is live on this thread).
    let mut worst = 0.0f64;
    for k in 0..x.len() {
        let fd = |h: f64| {
            let (mut xp, mut xm) = (x.clone(), x.clone());
            xp[k] += h;
            xm[k] -= h;
            (reconverged_f(&model, &s, &params, &xp) - reconverged_f(&model, &s, &params, &xm))
                / (2.0 * h)
        };
        let rich = (4.0 * fd(1e-3) - fd(2e-3)) / 3.0;
        assert!(
            rich.is_finite() && adj[k].is_finite(),
            "x[{k}] adj={} fd={rich}",
            adj[k]
        );
        worst = worst.max((adj[k] - rich).abs() / (1.0 + rich.abs()));
    }
    eprintln!("mixture class 2 vs reconverged FD {worst:.2e}");
    assert!(
        worst < 1e-5,
        "mixture class 2 vs reconverged FD {worst:.2e}"
    );
}

/// A two-η covariate-NN model (`tanh` hidden layer, `softplus` output) off its `init`
/// symmetric point, and a two-dose subject on its predictions with residuals of both signs.
#[cfg(feature = "nn")]
fn covariate_nn_fixture() -> (CompiledModel, Subject) {
    covariate_nn_fixture_from(COVARIATE_NN_IV)
}

#[cfg(feature = "nn")]
const COVARIATE_NN_IV: &str = r#"
[parameters]
  theta TVQ(6.0, 0.1, 50.0)
  omega ETA_CL ~ 0.1
  omega ETA_V1 ~ 0.1
  sigma PROP ~ 0.04 (sd)

[covariate_nn TYPICAL_PK]
  inputs     = [CRCL, WT]
  center     = [90, 70]
  scale      = [30, 15]
  outputs    = [CL, V1, V2]
  layers     = [3]
  activation = tanh
  output     = softplus
  init       = [4.5, 60.0, 50.0]

[individual_parameters]
  CL = TYPICAL_PK.CL * exp(ETA_CL)
  V1 = TYPICAL_PK.V1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TYPICAL_PK.V2

[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)

[error_model]
  DV ~ proportional(PROP)
"#;

/// [`COVARIATE_NN_IV`] with **two** networks: `CL` from one over both inputs, `V1`/`V2`
/// from another over `WT` alone, each with its own weight block and inputs.
#[cfg(feature = "nn")]
const COVARIATE_NN_TWO_NETS: &str = r#"
[parameters]
  theta TVQ(6.0, 0.1, 50.0)
  omega ETA_CL ~ 0.1
  omega ETA_V1 ~ 0.1
  sigma PROP ~ 0.04 (sd)

[covariate_nn CLNET]
  inputs     = [CRCL, WT]
  center     = [90, 70]
  scale      = [30, 15]
  outputs    = [CL]
  layers     = [3]
  activation = tanh
  output     = softplus
  init       = [4.5]

[covariate_nn VNET]
  inputs     = [WT]
  center     = [70]
  scale      = [15]
  outputs    = [V1, V2]
  layers     = [2]
  activation = tanh
  output     = softplus
  init       = [60.0, 50.0]

[individual_parameters]
  CL = CLNET.CL * exp(ETA_CL)
  V1 = VNET.V1 * exp(ETA_V1)
  Q  = TVQ
  V2 = VNET.V2

[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)

[error_model]
  DV ~ proportional(PROP)
"#;

/// [`COVARIATE_NN_IV`] on the user-ODE engine: the same kinetics as `[odes]`.
#[cfg(feature = "nn")]
fn covariate_nn_ode_src() -> String {
    COVARIATE_NN_IV.replace(
        "[structural_model]\n  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)\n",
        "[structural_model]\n  ode(states=[central, peripheral])\n\n[odes]\n  \
         d/dt(central) = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral\n  \
         d/dt(peripheral) = (Q/V1) * central - (Q/V2) * peripheral\n\n[scaling]\n  \
         y = central / V1\n",
    )
}

/// The fixture's subject (two doses, the shared times, covariates `CRCL = 72`, `WT = 81`,
/// observations on its predictions with residuals of both signs) for any covariate-NN
/// model `src`, with θ moved off the `init` symmetric point.
#[cfg(feature = "nn")]
fn covariate_nn_fixture_from(src: &str) -> (CompiledModel, Subject) {
    let mut model = parse_model_string(src).unwrap();
    // Off the `init` symmetric point (zero output-layer weights would zero every hidden
    // weight's gradient).
    for (i, t) in model.default_params.theta.iter_mut().enumerate() {
        *t += 0.13 * ((i as f64) * 0.7).sin();
    }
    let mut s = subject(
        &model,
        vec![
            DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 1000.0, 1, 0.0, false, 0.0),
        ],
        &TIMES,
        &[0.1, -0.05],
    );
    s.covariates = HashMap::from([("CRCL".to_string(), 72.0), ("WT".to_string(), 81.0)]);
    let preds = crate::pk::compute_predictions_with_tv(
        &model,
        &s,
        &model.default_params.theta,
        &[0.1, -0.05],
    );
    s.observations = preds
        .iter()
        .enumerate()
        .map(|(j, p)| p * (1.0 + 0.15 * (1.3 * j as f64).sin()))
        .collect();
    assert!(
        in_scope(&model, &s),
        "the recorders must serve a covariate-NN subject"
    );
    (model, s)
}

#[cfg(feature = "nn")]
#[test]
fn covariate_nn_matches_reconverged_fd() {
    let (model, s) = covariate_nn_fixture();
    let fd = fd_oracle(&model, &s, true);
    eprintln!("covariate NN FOCEI vs reconverged FD {fd:.2e}");
    assert!(fd < 1e-5, "covariate NN FOCEI vs FD {fd:.2e}");
    // The FOCEI R-matrix through the network: its nested dual differentiates `tanh`'s
    // `φ‴` (`DualT::lift3`), which the gradient above never reaches.
    let (fd, _) = hessian_gaps_for(&model, &s, Objective::Focei);
    eprintln!("covariate NN FOCEI Hessian vs reconverged FD {fd:.2e}");
    assert!(fd < 1e-5, "covariate NN FOCEI Hessian vs FD {fd:.2e}");
}

/// An `if` whose condition reads a network output takes the branch production takes. The
/// generic evaluator's conditions run on an `f64` view and were handed no network outputs
/// (the ambient thread-local they read had no writer since #2093), so `TYPICAL_PK.CL > 1`
/// read `0 > 1` on every recording and took the `else` arm while the production predictor
/// took the `if` one. The outputs now reach the conditions with the program's arguments.
/// Against the reconverged FD of production's objective, with the leaves and with every
/// network taped.
#[cfg(feature = "nn")]
#[test]
fn a_condition_on_a_network_output_takes_the_production_branch() {
    use super::NN_OUTPUT_LEAVES_OFF;
    let src = COVARIATE_NN_IV.replace(
        "  CL = TYPICAL_PK.CL * exp(ETA_CL)\n",
        "  if (TYPICAL_PK.CL > 1.0) {\n    CL = TYPICAL_PK.CL * exp(ETA_CL)\n  } else {\n    \
         CL = 2.0 * exp(ETA_CL)\n  }\n",
    );
    assert_ne!(src, COVARIATE_NN_IV, "the fixture must carry the branch");
    let (model, s) = covariate_nn_fixture_from(&src);
    let mut worst = 0.0f64;
    for off in [false, true] {
        NN_OUTPUT_LEAVES_OFF.with(|c| c.set(off));
        let fd = fd_oracle(&model, &s, true);
        NN_OUTPUT_LEAVES_OFF.with(|c| c.set(false));
        eprintln!("NN-output condition, leaves off = {off}: vs reconverged FD {fd:.2e}");
        worst = worst.max(fd);
    }
    // Measured worst 3.1e-7 on Linux (the FD's own noise; the same as the branch-free
    // fixture's); ~10× headroom. The `else` arm misses by O(1).
    assert!(worst < 3e-6, "NN-output condition vs FD {worst:.2e}");
}

/// The outer gradient of a covariate-NN subject reads the networks' outputs as tape leaves
/// and pulls their adjoints back through each network (`nn_output_leaves`), and must equal
/// the gradient with every network taped over its weights, under FOCEI, FOCE, Laplace's
/// exact anchor (the taped second-order recording) and AGQ's Gauss–Newton one. On four
/// engines: the closed-form IV walk, the same model with **two** networks (each network's
/// mapper, offset and inputs paired with its own leaves), the closed-form oral model
/// `warfarin_dcm` runs on, and the user-ODE recorder. Only the summation order of the
/// weights' chain rule differs. The counter shows the leaves were read, the switch that
/// they stood aside; every network's weight block is asserted live.
#[cfg(feature = "nn")]
#[test]
fn covariate_nn_output_leaf_gradient_equals_the_taped_networks() {
    use super::{NN_OUTPUT_LEAF_HITS, NN_OUTPUT_LEAVES_OFF};
    let oral = COVARIATE_NN_IV
        .replace(
            "  theta TVQ(6.0, 0.1, 50.0)\n",
            "  theta TVQ(6.0, 0.1, 50.0)\n  theta TVKA(1.2, 0.05, 10.0)\n",
        )
        .replace(
            "pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)",
            "pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)",
        )
        .replace(
            "  V2 = TYPICAL_PK.V2\n",
            "  V2 = TYPICAL_PK.V2\n  KA = TVKA\n",
        );
    let ode = covariate_nn_ode_src();
    let mut worst = 0.0f64;
    for (fixture, src) in [
        ("closed-form IV", COVARIATE_NN_IV),
        ("two networks", COVARIATE_NN_TWO_NETS),
        ("closed-form oral", oral.as_str()),
        ("ODE", ode.as_str()),
    ] {
        let (model, s) = covariate_nn_fixture_from(src);
        let params = moved_params(&model);
        let x = pack_params(&params);
        for (objective, interaction, name) in [
            (Objective::Focei, true, "FOCEI"),
            (Objective::Foce, false, "FOCE"),
            (
                Objective::Agq {
                    n_agq: 1,
                    exact: true,
                },
                true,
                "Laplace",
            ),
            (
                Objective::Agq {
                    n_agq: 3,
                    exact: false,
                },
                true,
                "AGQ-3",
            ),
        ] {
            let label = format!("{fixture}, {name}");
            let eta = ebe_for(&model, &s, &params, 1e-12, interaction);
            let run = |off: bool| {
                NN_OUTPUT_LEAVES_OFF.with(|c| c.set(off));
                NN_OUTPUT_LEAF_HITS.with(|c| c.set(0));
                let g =
                    subject_adjoint_gradient(&model, &s, &params, &x, eta.as_slice(), objective);
                NN_OUTPUT_LEAVES_OFF.with(|c| c.set(false));
                (g, NN_OUTPUT_LEAF_HITS.with(|c| c.get()))
            };
            let (taped, off_hits) = run(true);
            let (leaves, hits) = run(false);
            let (taped, leaves) = (
                taped.unwrap_or_else(|| panic!("{label}: taped gradient in scope")),
                leaves.unwrap_or_else(|| panic!("{label}: leaf gradient in scope")),
            );
            assert_eq!(off_hits, 0, "{label}: the switch must tape the networks");
            assert_eq!(hits, 1, "{label}: the output leaves were not read");
            for nn in &model.covariate_nns {
                let (w, n_w) = (nn.weights_offset, nn.mapper.mlp().n_weights());
                assert!(
                    (w..w + n_w).filter(|&i| taped[i].abs() > 1e-3).count() > n_w / 2,
                    "{label}: a vanishing weight gradient tests nothing (network at {w})"
                );
            }
            for (a, b) in leaves.iter().zip(&taped) {
                assert!(a.is_finite() && b.is_finite(), "{label}: {a} vs {b}");
                worst = worst.max((a - b).abs() / (1.0 + b.abs()));
            }
        }
    }
    eprintln!("covariate NN: output-leaf vs taped-network gradient {worst:.2e}");
    // Measured worst 9.8e-15 on Linux over the four fixtures (1.4e-16 on the IV one alone);
    // ~10× headroom.
    assert!(
        worst < 1e-13,
        "output-leaf vs taped-network gradient {worst:.2e}"
    );
}

/// The covariance Hessian of a covariate-NN subject is assembled in the networks' output
/// basis (`NnReduction`: 1 + 3 + 2 directions here against 21 weights), and must equal the
/// one assembled along every packed axis, under each objective. Dropping the weights' own
/// curvature (`Σ ∂Ψ/∂z_k ∇²z_k`) or the η part of a weight's direction moves the weight
/// block; the counter shows the reduction ran, the switch that it stood aside.
#[cfg(feature = "nn")]
#[test]
fn covariate_nn_reduced_hessian_equals_the_full_one() {
    use super::hessian::{NN_REDUCED_HITS, NN_REDUCE_OFF};
    let (model, s) = covariate_nn_fixture();
    let params = moved_params(&model);
    let x = pack_params(&params);
    for (objective, interaction, name) in [
        (Objective::Focei, true, "FOCEI"),
        (Objective::Foce, false, "FOCE"),
        (
            Objective::Agq {
                n_agq: 1,
                exact: true,
            },
            true,
            "Laplace",
        ),
        (
            Objective::Agq {
                n_agq: 3,
                exact: false,
            },
            true,
            "AGQ-3",
        ),
    ] {
        let eta = ebe_for(&model, &s, &params, 1e-12, interaction);
        let run = |off: bool| {
            std::thread::scope(|scope| {
                std::thread::Builder::new()
                    .stack_size(crate::FIT_RAYON_STACK_SIZE)
                    .spawn_scoped(scope, || {
                        NN_REDUCE_OFF.with(|c| c.set(off));
                        let h = subject_adjoint_cov_hessian(
                            &model,
                            &s,
                            &params,
                            &x,
                            eta.as_slice(),
                            objective,
                        );
                        (h, NN_REDUCED_HITS.with(|c| c.get()))
                    })
                    .expect("spawn")
                    .join()
                    .expect("join")
            })
        };
        let (full, off_hits) = run(true);
        let (reduced, hits) = run(false);
        let (full, reduced) = (
            full.expect("full Hessian in scope"),
            reduced.expect("reduced Hessian in scope"),
        );
        assert_eq!(
            off_hits, 0,
            "{name}: the switch must turn the reduction off"
        );
        assert_eq!(hits, 1, "{name}: the reduction did not run");
        let w = model.covariate_nns[0].weights_offset;
        assert!(
            (w..w + 21).any(|i| full[(i, i)].abs() > 1e-3),
            "{name}: a vanishing weight block tests nothing"
        );
        let mut worst = 0.0f64;
        for (a, b) in reduced.iter().zip(full.iter()) {
            assert!(a.is_finite() && b.is_finite(), "{name}: {a} vs {b}");
            worst = worst.max((a - b).abs() / (1.0 + b.abs()));
        }
        eprintln!("covariate NN {name}: reduced vs full Hessian {worst:.2e}");
        assert!(worst < 1e-9, "{name}: reduced vs full Hessian {worst:.2e}");
    }
}

/// The exact anchor through a `tanh` network: its taped-Hessian sweep differentiates the
/// activation's `φ″` (`DualV::lift3`), and the covariance step's nested dual its `φ‴`
/// (`DualH::lift4`). With either order missing the sweep records `NaN` and the subject
/// declines to finite differences, which `fd_oracle_agq` / `hessian_gaps_for` refuse
/// (they `expect` the adjoint), so each order has its own failing assertion here.
#[cfg(feature = "nn")]
#[test]
fn covariate_nn_exact_anchor_gradient_and_hessian_match_fd() {
    let (model, s) = covariate_nn_fixture();
    for (n_agq, name) in [(1, "Laplace"), (3, "AGQ-3 exact")] {
        let worst = fd_oracle_agq(&model, &s, n_agq, true);
        eprintln!("covariate NN {name} gradient vs reconverged FD {worst:.2e}");
        assert!(
            worst < 1e-5,
            "covariate NN {name} gradient vs FD {worst:.2e}"
        );
    }
    let (fd, _) = hessian_gaps_for(
        &model,
        &s,
        Objective::Agq {
            n_agq: 1,
            exact: true,
        },
    );
    eprintln!("covariate NN Laplace Hessian vs reconverged FD {fd:.2e}");
    assert!(fd < 1e-5, "covariate NN Laplace Hessian vs FD {fd:.2e}");
}

#[test]
fn pooled_model_without_random_effects_matches_reconverged_fd() {
    // No random effects: `Ψ = F`, an empty `H` — the width-0 bucket.
    let src = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(40.0, 1.0, 500.0)
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let model = parse_model_string(src).unwrap();
    assert_eq!(model.n_eta, 0);
    let s = subject(
        &model,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &TIMES,
        &[],
    );
    assert!(in_scope(&model, &s), "a pooled subject is in scope");
    for interaction in [true, false] {
        let worst = fd_oracle(&model, &s, interaction);
        eprintln!("pooled, interaction = {interaction}: vs reconverged FD {worst:.2e}");
        // Measured 1.1e-10 (both) vs reconverged FD (Richardson, EBE re-solved per point); 5× headroom.
        assert!(
            worst < 1e-9,
            "pooled, interaction = {interaction}: {worst:.2e}"
        );
    }
}

/// A 2-/3-cpt closed form with η on every disposition parameter, so each macro rate
/// moves with η (`pk` line and the `ka` arguments vary by model).
fn multi_cpt_model(pk: &str, oral: bool) -> CompiledModel {
    let three = pk.starts_with("three");
    let mut src = String::from(
        "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  theta TVV1(20.0, 1.0, 500.0)\n  \
         theta TVQ2(6.0, 0.1, 100.0)\n  theta TVV2(40.0, 1.0, 500.0)\n",
    );
    if three {
        src += "  theta TVQ3(1.5, 0.01, 50.0)\n  theta TVV3(90.0, 1.0, 900.0)\n";
    }
    if oral {
        src += "  theta TVKA(1.3, 0.01, 20.0)\n";
    }
    src += "  omega ETA_CL ~ 0.09\n  omega ETA_V1 ~ 0.06\n  omega ETA_Q2 ~ 0.05\n  omega ETA_V2 ~ 0.04\n";
    if three {
        src += "  omega ETA_Q3 ~ 0.05\n  omega ETA_V3 ~ 0.04\n";
    }
    src += "  sigma PROP_ERR ~ 0.1 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  \
            V1 = TVV1 * exp(ETA_V1)\n  Q2 = TVQ2 * exp(ETA_Q2)\n  V2 = TVV2 * exp(ETA_V2)\n";
    if three {
        src += "  Q3 = TVQ3 * exp(ETA_Q3)\n  V3 = TVV3 * exp(ETA_V3)\n";
    }
    if oral {
        src += "  KA = TVKA\n";
    }
    let ka = if oral { ", ka=KA" } else { "" };
    let disp = if three {
        "cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3"
    } else {
        "cl=CL, v1=V1, q=Q2, v2=V2"
    };
    src += &format!(
        "[structural_model]\n  pk {pk}({disp}{ka})\n[error_model]\n  DV ~ proportional(PROP_ERR)\n"
    );
    parse_model_string(&src).unwrap()
}

/// The 2-/3-cpt static superposition evaluates each subject's macro rates once
/// (`SeededPkDuals::with_disposition`) and every dose kind through the `_d` kernels. Its
/// predictions must equal the production `f64` predictor and its `∂f/∂η` a central
/// difference of it, for a multi-dose subject whose later doses land on residual drug,
/// with an infusion (IV) and an SS dose — every branch of the `_d` dispatchers.
#[test]
fn multi_cpt_static_superposition_matches_production_and_fd() {
    for (pk, oral) in [
        ("two_cpt_oral", true),
        ("two_cpt_iv", false),
        ("three_cpt_iv", false),
        ("three_cpt_oral", true),
    ] {
        let model = multi_cpt_model(pk, oral);
        let mut doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 80.0, 1, if oral { 0.0 } else { 40.0 }, false, 0.0),
        ];
        let mut times = vec![0.5, 1.0, 2.0, 6.0, 12.5, 13.5, 16.0, 24.0];
        // A separate SS subject: an SS record resets the train, so it gets its own walk.
        let ss = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)];
        for (label, doses, times) in [
            (
                "multidose",
                std::mem::take(&mut doses),
                std::mem::take(&mut times),
            ),
            ("ss", ss, vec![0.5, 2.0, 6.0, 11.0]),
        ] {
            let s = subject(&model, doses, &times, &[0.1, -0.05, 0.15, -0.1, 0.05, 0.08]);
            // The fixture must reach the hoisted path, not the event walk.
            assert!(
                !crate::sens::provider::subject_routes_to_event_walk(&model, &s),
                "{pk}/{label}: fixture routed to the event walk"
            );
            let theta = &moved_params(&model).theta;
            let eta: Vec<f64> = (0..model.n_eta).map(|k| 0.12 - 0.05 * k as f64).collect();
            let rows = exact_obs_grads(&model, &s, theta, &eta).expect("in recorder scope");
            let prod = |e: &[f64]| crate::pk::compute_predictions_with_tv(&model, &s, theta, e);
            let p0 = prod(&eta);
            let mut worst_v = 0.0f64;
            let mut worst_g = 0.0f64;
            for (j, row) in rows.iter().enumerate() {
                assert!(
                    row.f.is_finite() && p0[j].is_finite(),
                    "{pk}/{label} row {j}"
                );
                assert!(p0[j] > 1e-3, "{pk}/{label} row {j}: drug must be present");
                worst_v = worst_v.max((row.f - p0[j]).abs() / p0[j]);
                for k in 0..model.n_eta {
                    let fd = |h: f64| {
                        let (mut ep, mut em) = (eta.clone(), eta.clone());
                        ep[k] += h;
                        em[k] -= h;
                        (prod(&ep)[j] - prod(&em)[j]) / (2.0 * h)
                    };
                    let rich = (4.0 * fd(1e-4) - fd(2e-4)) / 3.0;
                    assert!(rich.is_finite() && row.df_deta[k].is_finite());
                    worst_g = worst_g.max((row.df_deta[k] - rich).abs() / (1e-3 + rich.abs()));
                }
            }
            eprintln!("{pk}/{label}: value {worst_v:.2e}, dη {worst_g:.2e}");
            // Measured: values bit-identical (0.0) on every model and dose kind; ∂f/∂η worst
            // 4.4e-8 (three_cpt_oral/ss), the Richardson floor at h = 1e-4. 4.5× headroom.
            assert!(worst_v < 1e-14, "{pk}/{label}: value gap {worst_v:.2e}");
            assert!(worst_g < 2e-7, "{pk}/{label}: ∂f/∂η gap {worst_g:.2e}");
        }
    }
}

/// The walk's schedule is built once per subject while its inputs are unchanged
/// (`provider::cached_event_schedule`): a second identical recording builds none.
#[test]
fn walk_schedule_is_reused_across_recordings() {
    let model = parse_model_string(CF_TVCOV).unwrap();
    let s = cf_tvcov_subject(&model);
    let theta = &model.default_params.theta;
    let first = exact_obs_grads(&model, &s, theta, &[0.1, -0.05]).unwrap();
    crate::pk::event_driven::reset_schedule_build_count();
    let second = exact_obs_grads(&model, &s, theta, &[0.2, 0.05]).unwrap();
    assert_eq!(
        crate::pk::event_driven::schedule_build_count(),
        0,
        "an unchanged subject must reuse its schedule"
    );
    assert!(
        first.iter().zip(&second).any(|(a, b)| a.f != b.f),
        "η must move the walk"
    );
}

/// The cache keys on every schedule input, so a subject mutated in place (same address,
/// as a test or an editing caller does) misses and rebuilds — each time vector moved
/// separately, and each result checked against a fresh-thread evaluation.
#[test]
fn walk_schedule_cache_never_serves_a_mutated_subject() {
    let model = parse_model_string(CF_TVCOV).unwrap();
    let theta = model.default_params.theta.clone();
    let eta = [0.1, -0.05];
    let fresh = |s: &Subject| {
        let (m, s, th) = (
            parse_model_string(CF_TVCOV).unwrap(),
            s.clone(),
            theta.clone(),
        );
        std::thread::spawn(move || exact_obs_grads(&m, &s, &th, &eta).unwrap())
            .join()
            .unwrap()
            .iter()
            .map(|r| r.f.to_bits())
            .collect::<Vec<_>>()
    };
    let mut s = cf_tvcov_subject(&model);
    let _ = exact_obs_grads(&model, &s, &theta, &eta).unwrap(); // populate the cache
    let edits: [(&str, fn(&mut Subject)); 4] = [
        ("an observation time", |s| s.obs_times[2] += 0.5),
        ("a dose time", |s| s.doses[1].time += 1.0),
        ("a dose rate (bolus to infusion)", |s| {
            s.doses[0].rate = 50.0;
            s.doses[0].duration = 2.0;
        }),
        ("a reset time", |s| {
            s.reset_times.push(10.0);
            s.reset_covariates.push(s.covariates.clone());
            s.reset_occasions.push(1);
        }),
    ];
    for (what, edit) in edits {
        edit(&mut s);
        let got: Vec<u64> = exact_obs_grads(&model, &s, &theta, &eta)
            .unwrap()
            .iter()
            .map(|r| r.f.to_bits())
            .collect();
        assert_eq!(got, fresh(&s), "a stale schedule after moving {what}");
    }
}

/// A model with an `[individual_parameters]` entry no PK slot reads (`S`, with its own η)
/// and a literal PK argument (`ka = 1.0`) stays in the recorders' scope: the unused
/// parameter is inert and the literal is a constant slot (tests/unbound_param_slot_gradient).
#[test]
fn unbound_parameter_and_literal_pk_argument_stay_in_recorder_scope() {
    let model = parse_model_string(
        "[parameters]\n  theta THCL(0.134, 0.001, 10.0)\n  theta THV(8.11, 0.1, 500.0)\n  \
         theta THS(3.0, 0.01, 100.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  \
         omega ETA_S ~ 0.25\n  sigma PROP_ERR ~ 0.02 (sd)\n[individual_parameters]\n  \
         CL = THCL * exp(ETA_CL)\n  V = THV * exp(ETA_V)\n  S = THS * exp(ETA_S)\n\
         [structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=1.0)\n[error_model]\n  \
         DV ~ proportional(PROP_ERR)\n",
    )
    .unwrap();
    let prog = model.indiv_param_partials.indiv_param_program.as_ref();
    eprintln!(
        "closed_form_model_scope={} program={} analytic_param_program={} n_eta_axis={:?} n_eta={} required={:?} prog_slots={:?}",
        crate::sens::provider::closed_form_model_scope(&model),
        prog.is_some(),
        crate::sens::provider::analytic_param_program(&model).is_some(),
        prog.map(|p| p.n_eta_axis()),
        model.n_eta,
        model.pk_model.required_pk_params(),
        prog.map(|p| p.pk_slots_ref().to_vec()),
    );
    assert!(
        recorder_in_scope(&model),
        "the recorders must serve this model"
    );
    // The literal slot is seeded as a constant: the adjoint must still be the exact gradient.
    let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    let s = subject(&model, doses, &TIMES, &[0.1, -0.05, 0.2]);
    // The oracle re-solves the EBE at 1e-14: at its default 1e-12 the worst coordinate
    // (ln L_11, ω_CL) differs by 5.8e-6, and tightening moves the *FD* value towards the
    // adjoint's (0.638742 → 0.638737 against 0.638733) while the adjoint is unchanged —
    // the oracle's EBE noise, not the gradient. Measured worst at 1e-14: 3.7e-6; 5× headroom.
    ORACLE_EBE_TOL.set(1e-14);
    let worst = fd_oracle(&model, &s, true);
    ORACLE_EBE_TOL.set(1e-12);
    eprintln!("literal ka + unbound S: gradient vs reconverged FD {worst:.2e}");
    assert!(worst < 2e-5, "{worst:.2e}");
}

/// Dense residual covariance beyond FOCEI's gradient: AGQ/Laplace with a free
/// `block_sigma` (the ρ coordinate written by the adjoint, which the pre-#486 fixed-η
/// score did not), and the R-matrix under `block_sigma` and each serial correlation kind.
/// The serial `ar1`/`car1` Hessians read `∂²C/∂ρ²` (`ρⁿ`'s curvature), which the dense
/// R once lifted as zero: their gap to profile FD was 0.47 and 0.30, against ~5e-8 now.
/// At ρ = 0 a pair's coefficient vanishes while its derivative does not, which a
/// value/derivative pairing filtered separately dropped.
#[test]
fn dense_residual_agq_gradient_and_hessians_match_fd() {
    let block = parse_model_string(&dense_src("", true)).unwrap();
    let sb = dense_subject(&block);
    // Measured 6.0e-7 (1 node, exact anchor) and 1.3e-8 (3 nodes, Gauss–Newton); 5×.
    for (n_agq, exact, bound) in [(1, true, 3e-6), (3, false, 7e-8)] {
        let worst = fd_oracle_agq(&block, &sb, n_agq, exact);
        eprintln!("block_sigma AGQ n={n_agq} exact={exact}: gradient {worst:.2e}");
        assert!(worst < bound, "block_sigma AGQ n={n_agq}: {worst:.2e}");
    }
    // Measured 1.4e-7; 5×.
    let (h, _) = hessian_gaps(&block, &sb, true);
    assert!(h < 7e-7, "block_sigma FOCEI hessian {h:.2e}");
    // Measured cs 6.1e-9 before #2093's inner-solver changes and 9.7e-8 after, with the
    // analytic Hessian unchanged to 1e-10 and the reference EBEs agreeing to 1e-15 in the
    // objective: the spread is the reconverged-FD reference's own floor on this fixture
    // (block_sigma 1.4e-7 / 8.4e-8 on the same two builds; ar1 4.9e-8, car1 6.2e-8). So `cs`
    // takes its siblings' bound, ~3× over the larger reading. The defect it guards — the
    // dense R lifting `∂²C/∂ρ²` as zero — gave gaps of 0.47 and 0.30.
    for (kind, bound) in [("cs", 3e-7), ("ar1", 2.5e-7), ("car1", 3e-7)] {
        let m = parse_model_string(&dense_src(
            &format!("  correlation {kind}(RHO) by ARM\n"),
            false,
        ))
        .unwrap();
        let (h, _) = hessian_gaps(&m, &dense_subject(&m), true);
        eprintln!("{kind} FOCEI hessian {h:.2e}");
        assert!(h < bound, "{kind} FOCEI hessian {h:.2e}");
    }
    // ρ = 0 exactly (`moved_params` keeps a zero θ at zero): the derivative pairs survive.
    for kind in ["cs", "ar1"] {
        let src = dense_src(&format!("  correlation {kind}(RHO) by ARM\n"), false)
            .replace("theta RHO(0.45,", "theta RHO(0.0,");
        let m = parse_model_string(&src).unwrap();
        assert_eq!(
            moved_params(&m).theta[m.residual_serial.as_ref().unwrap().rho_theta],
            0.0
        );
        let s = dense_subject(&m);
        let g = fd_oracle(&m, &s, true);
        let (h, _) = hessian_gaps(&m, &s, true);
        eprintln!("{kind} at rho = 0: gradient {g:.2e}, hessian {h:.2e}");
        // Measured 1.5e-9 (gradient) and 3.5e-10 (Hessian) for both kinds; 5×.
        assert!(
            g < 8e-9 && h < 2e-9,
            "{kind} at rho = 0: gradient {g:.2e}, hessian {h:.2e}"
        );
    }
}

/// `ode_template` models take their depot ODE and their observation scale from the template;
/// only the overridden equation is user-written. The adjoint gradient of such a model must be
/// exact like its hand-written twin (`tests/model_edit_equivalence.rs`'s Michaelis–Menten case,
/// whose two fits must agree).
#[test]
fn ode_template_model_gradient_matches_fd_like_its_hand_written_twin() {
    const PARAMS: &str =
        "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  theta TVV(40.0, 1.0, 500.0)\n  \
        theta TVKA(1.0, 0.01, 10.0)\n  theta TVKM(5.0, 0.0, 100.0)\n  omega ETA_CL ~ 0.15\n  \
        omega ETA_V ~ 0.15\n  omega ETA_KA ~ 0.20\n  sigma PROP_ERR ~ 0.04 (sd)\n\
        [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n  \
        KA = TVKA * exp(ETA_KA)\n  KM = TVKM\n";
    let mm = "d/dt(central) = KA * depot - ((CL * KM / (KM + central / V)) / V) * central\n";
    let templ = format!(
        "{PARAMS}[structural_model]\n  ode_template one_cpt_oral(cl=CL, v=V, ka=KA)\n\
         [error_model]\n  DV ~ proportional(PROP_ERR)\n[odes]\n  {mm}"
    );
    let hand = format!(
        "{PARAMS}[structural_model]\n  ode(obs_cmt=central, states=[depot, central])\n\
         [odes]\n  d/dt(depot) = -KA * depot\n  {mm}[scaling]\n  obs_scale = V\n\
         [error_model]\n  DV ~ proportional(PROP_ERR)\n"
    );
    for (name, src) in [("template", templ), ("hand-written", hand)] {
        let model = tight(parse_model_string(&src).unwrap());
        let doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ];
        let s = subject(&model, doses, &TIMES, &[0.1, -0.05, 0.2]);
        assert!(in_scope(&model, &s), "{name}: must be served");
        let worst = fd_oracle(&model, &s, true);
        eprintln!("{name}: gradient vs reconverged FD {worst:.2e}");
        // Measured 3.7e-7 on both forms (the Richardson floor of an RK45 objective); 5×.
        assert!(worst < 2e-6, "{name}: {worst:.2e}");
    }
}

/// The `ode_template` and hand-written Michaelis–Menten models are one objective: on the
/// equivalence test's data, at default solver settings, their EBEs are bit-identical and
/// their per-subject adjoint gradients agree to rounding — not bit for bit, because each
/// model's own RHS bytecode is taped and the template spells the depot term differently.
#[test]
fn mm_template_and_hand_written_gradients_agree_per_subject() {
    const PARAMS: &str =
        "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  theta TVV(40.0, 1.0, 500.0)\n  \
        theta TVKA(1.0, 0.01, 10.0)\n  theta TVKM(5.0, 0.0, 100.0)\n  omega ETA_CL ~ 0.15\n  \
        omega ETA_V ~ 0.15\n  omega ETA_KA ~ 0.20\n  sigma PROP_ERR ~ 0.04 (sd)\n\
        [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n  \
        KA = TVKA * exp(ETA_KA)\n  KM = TVKM\n";
    let mm = "d/dt(central) = KA * depot - ((CL * KM / (KM + central / V)) / V) * central\n";
    let templ = parse_model_string(&format!(
        "{PARAMS}[structural_model]\n  ode_template one_cpt_oral(cl=CL, v=V, ka=KA)\n\
         [error_model]\n  DV ~ proportional(PROP_ERR)\n[odes]\n  {mm}"
    ))
    .unwrap();
    let hand = parse_model_string(&format!(
        "{PARAMS}[structural_model]\n  ode(obs_cmt=central, states=[depot, central])\n\
         [odes]\n  d/dt(depot) = -KA * depot\n  {mm}[scaling]\n  obs_scale = V\n\
         [error_model]\n  DV ~ proportional(PROP_ERR)\n"
    ))
    .unwrap();
    let pop = crate::read_nonmem_csv(
        std::path::Path::new("data/two_cpt_oral_cov.csv"),
        None,
        None,
    )
    .unwrap();
    let p = templ.default_params.clone();
    let x = pack_params(&p);
    let (mut n_diff, mut worst) = (0usize, 0.0f64);
    for s in &pop.subjects {
        let e = find_ebe(&templ, s, &p, 500, 1e-8, None, None, 0).eta;
        let eh = find_ebe(&hand, s, &p, 500, 1e-8, None, None, 0).eta;
        let same_eta = e
            .iter()
            .zip(eh.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits());
        let gt = subject_adjoint_packed_gradient(&templ, s, &p, &x, e.as_slice(), true).unwrap();
        let gh = subject_adjoint_packed_gradient(&hand, s, &p, &x, e.as_slice(), true).unwrap();
        assert!(same_eta, "subject {}: the EBEs must be bit-identical", s.id);
        // Finite first: `f64::max` would discard a NaN and pass on the finite rows.
        assert!(
            gt.iter().chain(&gh).all(|v| v.is_finite()),
            "subject {}",
            s.id
        );
        let d = gt
            .iter()
            .zip(&gh)
            .map(|(a, b)| (a - b).abs() / (1.0 + b.abs()))
            .fold(0.0, f64::max);
        if d > 0.0 {
            n_diff += 1;
            worst = worst.max(d);
        }
    }
    eprintln!(
        "{} subjects, {n_diff} differ, worst {worst:.2e}",
        pop.subjects.len()
    );
    // Measured worst 1.4e-13 (subject 19); 7×. A wrong gradient on either form sits
    // ≥ 1e-7 from the other (each is exact against reconverged FD to ~4e-7).
    assert!(worst < 1e-12, "{worst:.2e}");
}

/// `[individual_parameters]` entries no PK slot reads, each a function of θ
/// (`FILLER_i = TVCL·c_i`, tests/large_ode_slot_headroom.rs), are bit-inert: they put nodes on
/// the tape that receive no adjoint, so every per-subject quantity a fit consumes — the
/// inner η-Jacobian, gradient, Hessian and H̃, the outer gradient and the EBE response —
/// is bit-identical to the model without them.
#[test]
fn unread_theta_dependent_parameters_are_bit_inert() {
    let base = "[parameters]\n  theta TVCL(0.15, 0.001, 10.0)\n  theta TVV(8.0, 0.1, 500.0)\n  \
        theta TVKA(1.2, 0.01, 50.0)\n  omega ETA_CL ~ 0.07\n  omega ETA_V  ~ 0.02\n  \
        omega ETA_KA ~ 0.10\n  sigma PROP_ERR ~ 0.01 (sd)\n[individual_parameters]\n  \
        CL = TVCL * exp(ETA_CL)\n  V  = TVV  * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n";
    let tail = "[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n[error_model]\n  \
        DV ~ proportional(PROP_ERR)\n";
    let fillers: String = (1..=15)
        .map(|i| format!("  FILLER_{i} = TVCL * {:.3}\n", 1.0 + i as f64 * 0.01))
        .collect();
    let plain = parse_model_string(&format!("{base}{tail}")).unwrap();
    let bloat = parse_model_string(&format!("{base}{fillers}{tail}")).unwrap();
    let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    let s = subject(&plain, doses, &[1.0, 4.0, 12.0], &[0.1, -0.05, 0.2]);
    let p = moved_params(&plain);
    let x = pack_params(&p);
    let eta = [0.1, -0.05, 0.2];
    let og = |m: &CompiledModel| exact_obs_grads(m, &s, &p.theta, &eta).unwrap();
    let (a, b) = (og(&plain), og(&bloat));
    let rows_same = a.iter().zip(&b).all(|(u, v)| {
        u.f.to_bits() == v.f.to_bits()
            && u.df_deta
                .iter()
                .zip(&v.df_deta)
                .all(|(x, y)| x.to_bits() == y.to_bits())
    });
    let ga = subject_adjoint_packed_gradient(&plain, &s, &p, &x, &eta, true).unwrap();
    let gb = subject_adjoint_packed_gradient(&bloat, &s, &p, &x, &eta, true).unwrap();
    let diffs: Vec<(usize, f64, f64)> = ga
        .iter()
        .zip(&gb)
        .enumerate()
        .filter(|(_, (u, v))| u.to_bits() != v.to_bits())
        .map(|(k, (u, v))| (k, *u, *v))
        .collect();
    let bits = |m: &nalgebra::DMatrix<f64>| m.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    let ha = exact_inner_hessian(&plain, &s, &p, &eta).unwrap();
    let hb = exact_inner_hessian(&bloat, &s, &p, &eta).unwrap();
    let ta = exact_htilde(&plain, &s, &p, &eta).unwrap();
    let tb = exact_htilde(&bloat, &s, &p, &eta).unwrap();
    let ia = exact_inner_gradient(&plain, &s, &p, &eta, None).unwrap();
    let ib = exact_inner_gradient(&bloat, &s, &p, &eta, None).unwrap();
    assert!(rows_same, "inner η-Jacobian");
    assert!(
        diffs.is_empty(),
        "outer gradient coordinates differ: {diffs:?}"
    );
    assert_eq!(bits(&ha), bits(&hb), "inner Hessian");
    assert_eq!(bits(&ta), bits(&tb), "H̃");
    assert!(
        ia.iter().zip(&ib).all(|(u, v)| u.to_bits() == v.to_bits()),
        "inner gradient"
    );
    let run_dx = |m: &CompiledModel| {
        std::thread::scope(|sc| {
            std::thread::Builder::new()
                .stack_size(crate::FIT_RAYON_STACK_SIZE)
                .spawn_scoped(sc, || subject_adjoint_eta_dx(m, &s, &p, &x, &eta, true))
                .unwrap()
                .join()
                .unwrap()
        })
        .unwrap()
    };
    let (da, db) = (run_dx(&plain), run_dx(&bloat));
    let dx_same = da.iter().zip(&db).all(|(u, v)| {
        u.iter()
            .zip(v.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
    });
    let dx_gap = da
        .iter()
        .zip(&db)
        .flat_map(|(u, v)| {
            u.iter()
                .zip(v.iter())
                .map(|(a, b)| (a - b).abs())
                .collect::<Vec<_>>()
        })
        .fold(0.0, f64::max);
    assert!(dx_same, "EBE response differs by {dx_gap:.2e}");
}

/// `mode_solve` serves an indefinite but nonsingular exact Hessian (an EBE the inner
/// search's fallback left at a stationary point that is not a strict minimum) exactly, and
/// declines a singular one. Before it, a Cholesky failure here sent the subject to the
/// reconverged finite-difference score: on a 141-weight covariate network under Laplace,
/// 43 of 3030 subject scores, and most of the fit's time.
#[test]
fn mode_solve_takes_an_indefinite_nonsingular_hessian() {
    // Eigenvalues 3 and −1: nonsingular, not definite.
    let h = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
    assert!(
        h.clone().cholesky().is_none(),
        "the fixture must be indefinite"
    );
    let b = DVector::from_column_slice(&[1.0, -3.0]);
    let stationary = [0.0, 3e-6];
    let w = super::mode_solve::<0>(&h, &b, &stationary, || None).expect("nonsingular");
    let r = &h * &w - &b;
    assert!(r.norm() < 1e-14, "residual {:.2e}", r.norm());
    // A strict mode keeps the Cholesky solve.
    let spd = DMatrix::from_row_slice(2, 2, &[2.0, 0.5, 0.5, 1.0]);
    let w = super::mode_solve::<0>(&spd, &b, &stationary, || None).expect("definite");
    assert!((&spd * &w - &b).norm() < 1e-14);
    // Singular: declined, not a non-finite `w`.
    let sing = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]);
    assert!(super::mode_solve::<0>(&sing, &b, &stationary, || None).is_none());
}

/// #2093 review (round 2, finding 14): the indefinite branch serves only a stationary `η̂`.
/// A fallback end point the inner search left short of the mode (gradient 1e-2 here) has a
/// non-zero `wᵀ∇_η l`, which the identity drops, so it declines to the reconverged score.
/// The straddle is `∇_η l` alone, on the same indefinite `H`; a strict mode keeps its
/// Cholesky solve whatever the gradient (the inner search's own convergence gate covers
/// it). A `NaN` gradient declines too. The mutation this catches is removing the gate.
#[test]
fn mode_solve_declines_an_indefinite_hessian_off_the_mode() {
    let h = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
    let b = DVector::from_column_slice(&[1.0, -3.0]);
    assert!(super::mode_solve::<0>(&h, &b, &[0.0, 5e-5], || None).is_some());
    assert!(super::mode_solve::<0>(&h, &b, &[1e-2, 0.0], || None).is_none());
    assert!(super::mode_solve::<0>(&h, &b, &[f64::NAN, 0.0], || None).is_none());
    let spd = DMatrix::from_row_slice(2, 2, &[2.0, 0.5, 0.5, 1.0]);
    assert!(super::mode_solve::<0>(&spd, &b, &[1e-2, 0.0], || None).is_some());
}

/// A declined stationarity gate is judged again by the inner search's own gradient, which
/// the definite branch never computes. The recording's `∇_η l` and the search's are exact
/// derivatives of two floating-point solves that part at the solver's tolerance where the
/// tangents are not error-controlled (`inner_search_gradient`), so an EBE the search drove
/// to zero can read non-stationary on the recording.
#[test]
fn mode_solve_judges_a_declined_gate_by_the_inner_search_gradient() {
    use std::cell::Cell;
    let indefinite = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
    let spd = DMatrix::from_row_slice(2, 2, &[2.0, 0.5, 0.5, 1.0]);
    let b = DVector::from_column_slice(&[1.0, -3.0]);
    let calls = Cell::new(0);
    let inner = |g: [f64; 2]| {
        let calls = &calls;
        move || {
            calls.set(calls.get() + 1);
            Some(g)
        }
    };
    // The recording reads 1e-2, the search 5e-5: taken, by the search's gradient.
    let w = super::mode_solve(&indefinite, &b, &[1e-2, 0.0], inner([5e-5, 0.0]));
    assert!(w.is_some());
    assert_eq!(calls.get(), 1);
    // Both non-stationary: declined.
    assert!(super::mode_solve(&indefinite, &b, &[1e-2, 0.0], inner([1e-2, 0.0])).is_none());
    // No inner gradient: declined.
    assert!(super::mode_solve::<2>(&indefinite, &b, &[1e-2, 0.0], || None).is_none());
    // A definite Hessian and a stationary recording never ask.
    calls.set(0);
    assert!(super::mode_solve(&spd, &b, &[1e-2, 0.0], inner([1.0, 1.0])).is_some());
    assert!(super::mode_solve(&indefinite, &b, &[0.0, 5e-5], inner([1.0, 1.0])).is_some());
    // #2468 review: a singular `H` declines whatever the gradient, so a declined gate there
    // never pays the inner search's recording.
    let singular = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]);
    let never = || -> Option<[f64; 2]> { panic!("a singular H must not re-judge the gate") };
    assert!(super::mode_solve(&singular, &b, &[1e-2, 0.0], never).is_none());
    assert_eq!(calls.get(), 0);
}

/// `inner_search_gradient` is the inner search's gradient: `exact_inner_gradient`, under
/// FOCEI's conditional and FOCE's frozen variance alike.
#[test]
fn the_gate_gradient_is_the_inner_search_gradient() {
    let model = parse_model_string(ONE_CPT_ORAL).unwrap();
    let s = subject(
        &model,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &TIMES,
        &[0.1, -0.05, 0.2],
    );
    let params = moved_params(&model);
    let eta = [0.08, -0.03, 0.15];
    let ctx = LikCtx::new(&model, &s, &params);
    let pv = ParT::of(&model, &s, &params);
    let mut pv = pv;
    pv.omega = crate::sens::focei_kernel::OmegaT::from_chol(
        super::stacked_omega::<3>(&model, &s, &params, &params, 3).chol,
    )
    .unwrap();
    let fpop: Vec<f64> = canonical_jets1::<3>(&model, &s, None, &params.theta, &[0.0; 3])
        .unwrap()
        .iter()
        .map(|q| q.v)
        .collect();
    for frozen in [None, Some(fpop.as_slice())] {
        let got = super::inner_search_gradient::<3>(
            &model,
            &s,
            None,
            &params.theta,
            &eta,
            &ctx,
            &pv,
            frozen,
        )
        .unwrap();
        let want = exact_inner_gradient(&model, &s, &params, &eta, frozen).unwrap();
        assert!(
            got.iter().any(|g| g.abs() > 1e-3),
            "a vanishing gradient tests nothing"
        );
        for (a, b) in got.iter().zip(&want) {
            assert_eq!(a.to_bits(), b.to_bits(), "frozen = {}", frozen.is_some());
        }
    }
}

/// #2093 review (round 2, finding 13): the EBE warm-start predictor `dη̂/dx` serves 5- and
/// 6-effect subjects, which `hessian_bucket` sends to width 6. That arm was missing, so every
/// such model lost the predictor. Checked against a central difference of the re-solved EBE
/// (the EBE is re-found at `x ± h·e_k`): an oracle sharing no derivative code with it.
#[test]
fn eta_dx_serves_a_five_effect_subject_and_matches_the_re_solved_ebe() {
    use crate::estimation::parameterization::unpack_params;
    let model = parse_model_string(&one_cpt_oral_n_eta(5)).unwrap();
    assert_eq!(model.n_eta, 5);
    let s = subject(
        &model,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &TIMES,
        &[0.1, -0.05, 0.2, 0.03, -0.02],
    );
    let params = moved_params(&model);
    let x = pack_params(&params);
    // FOCEI's conditional mode and FOCE's frozen-variance one each respond on their own
    // inner problem (`population_eta_dx` passes the fit's `interaction`).
    for interaction in [true, false] {
        let eta = ebe_for(&model, &s, &params, 1e-12, interaction);
        let dx = std::thread::scope(|sc| {
            std::thread::Builder::new()
                .stack_size(crate::FIT_RAYON_STACK_SIZE)
                .spawn_scoped(sc, || {
                    subject_adjoint_eta_dx(&model, &s, &params, &x, eta.as_slice(), interaction)
                })
                .unwrap()
                .join()
                .unwrap()
        })
        .expect("a five-effect subject must be served (width-6 bucket)");
        let mut worst = 0.0f64;
        for k in 0..x.len() {
            let h = 1e-4 * (1.0 + x[k].abs());
            let at = |d: f64| {
                let mut xp = x.clone();
                xp[k] += d;
                ebe_for(&model, &s, &unpack_params(&xp, &params), 1e-12, interaction)
            };
            let (ep, em) = (at(h), at(-h));
            for j in 0..5 {
                let fd = (ep[j] - em[j]) / (2.0 * h);
                assert!(fd.is_finite() && dx[k][j].is_finite(), "x[{k}] η[{j}]");
                worst = worst.max((dx[k][j] - fd).abs() / (1.0 + fd.abs()));
            }
        }
        eprintln!("eta_dx (5 η, interaction = {interaction}) vs re-solved EBE {worst:.2e}");
        // Measured on Linux: 1.8e-7 (FOCEI), 1.6e-5 (FOCE); ~10× headroom each. FOCE's
        // response taken on FOCEI's conditional mode (the old hard-coded `interaction = true`
        // in `population_eta_dx`) misses it by 6.1e-2, so the FOCE bound straddles that defect.
        let bound = if interaction { 2e-6 } else { 2e-4 };
        assert!(
            worst < bound,
            "eta_dx (interaction = {interaction}) vs re-solved EBE {worst:.2e}"
        );
    }
}

/// The EBE warm-start predictor `dη̂/dx` on subjects carrying non-Gaussian records (the
/// outer optimizer now engages it on every model the outer gradient serves): the term's
/// `∇q`/`∇²q` enter the pass's `g`, so the mode response is the full inner problem's.
/// Against the central difference of the re-solved EBE, under both interaction settings.
#[cfg(feature = "survival")]
#[test]
fn eta_dx_matches_the_re_solved_ebe_with_a_non_gaussian_term() {
    use crate::estimation::parameterization::unpack_params;
    let root = env!("CARGO_MANIFEST_DIR");
    let mut worst_all = 0.0f64;
    for (m, d) in [
        ("examples/pktte_joint.ferx", "data/pktte_joint.csv"),
        ("examples/tte_exponential.ferx", "data/tte_exponential.csv"),
    ] {
        let mut prep =
            crate::api::prepare_run(&format!("{root}/{m}"), Some(&format!("{root}/{d}"))).unwrap();
        if let Some(ode) = prep.parsed.model.ode_spec.as_mut() {
            ode.solver_opts.reltol = 1e-12;
            ode.solver_opts.abstol = 1e-14;
        }
        let model = &prep.parsed.model;
        let s = prep
            .population
            .subjects
            .iter()
            .find(|s| !s.obs_records.is_empty())
            .unwrap();
        let params = moved_params(model);
        let x = pack_params(&params);
        for interaction in [true, false] {
            let eta = ebe_for(model, s, &params, 1e-12, interaction);
            let dx = std::thread::scope(|sc| {
                std::thread::Builder::new()
                    .stack_size(crate::FIT_RAYON_STACK_SIZE)
                    .spawn_scoped(sc, || {
                        subject_adjoint_eta_dx(model, s, &params, &x, eta.as_slice(), interaction)
                    })
                    .unwrap()
                    .join()
                    .unwrap()
            })
            .unwrap_or_else(|| panic!("{m}: a non-Gaussian subject must be served"));
            let mut worst = 0.0f64;
            let mut live = 0.0f64;
            for k in 0..x.len() {
                let h = 1e-4 * (1.0 + x[k].abs());
                let at = |d: f64| {
                    let mut xp = x.clone();
                    xp[k] += d;
                    ebe_for(model, s, &unpack_params(&xp, &params), 1e-12, interaction)
                };
                let (ep, em) = (at(h), at(-h));
                for j in 0..model.n_eta {
                    let fd = (ep[j] - em[j]) / (2.0 * h);
                    assert!(fd.is_finite() && dx[k][j].is_finite(), "x[{k}] η[{j}]");
                    worst = worst.max((dx[k][j] - fd).abs() / (1.0 + fd.abs()));
                    live = live.max(fd.abs());
                }
            }
            assert!(
                live > 1e-2,
                "{m}: the mode does not move — the test sees nothing"
            );
            eprintln!("{m} interaction={interaction}: eta_dx vs re-solved EBE {worst:.2e}");
            worst_all = worst_all.max(worst);
        }
    }
    assert!(worst_all < 2e-4, "eta_dx vs re-solved EBE {worst_all:.2e}");
}

/// The `tests/ss_fit_smoke.rs` model, subject 3 of `data/ss_oral_q24.csv` (#1498).
const SS_ORAL_Q24: &str = r#"
[parameters]
  theta TVCL(5.0, 0.01, 50.0)
  theta TVV(5.0, 0.5, 200.0)
  theta TVKA(4.5, 0.05, 20.0)
  omega ETA_CL ~ 0.05
  omega ETA_V  ~ 0.05
  omega ETA_KA ~ 0.1
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// #2093 review (round 2, finding 9): base's `foce_rtilde_inverse` declined the Woodbury
/// identity above `1 + tr(S) = 1e6` (#1498), and the adjoint's FOCE kernel had no such gate.
/// On this subject the typical prediction at t = 20 h is ~5.3e-8, so its `R⁰` is ~1.1e-18
/// while `G(η̂)` is O(1), and `1 + tr(S)` is far past the gate. The oracle is production's
/// FOCE objective at the same `η̂` (its dense route), compared with the kernel's `F` with no
/// re-solve: the reconverged FD of the gradient measures the EBE's own noise here, since the
/// near-singular row weights the inner objective by ~1e18. The straddle: with the gate
/// forced open (`FORCE_WOODBURY`) the kernel's value must be visibly wrong, or the gated
/// assertion would pass on a fixture that cannot see the loss.
///
/// #2441: this fixture used t = 23 h, where `R⁰` sat on the old absolute `1e-12` variance
/// floor. With the floor gone that row's `R⁰` is ~1e-25 and the kernel declines it
/// outright, so the fixture moved to a row that is near-singular but scored as it is. It
/// also pins that the kernel's `R` (`loadings_variance_g`) guards exactly like production's:
/// restoring the floor in `loadings_variance_g` alone puts the gated gap at 4.5e-7.
#[test]
fn foce_kernel_declines_woodbury_on_a_near_singular_row() {
    use crate::sens::focei_kernel::FORCE_WOODBURY;
    const T_LATE: f64 = 20.0;
    let model = parse_model_string(SS_ORAL_Q24).unwrap();
    let mut s = subject(
        &model,
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 24.0)],
        &[1.0, 4.0, 12.0, T_LATE],
        &[0.0, 0.0, 0.0],
    );
    s.observations = vec![4.048116, 4.014245, 2.184979, 0.872772];
    let p = model.default_params.clone();
    let typ = crate::pk::compute_predictions_with_tv(&model, &s, &p.theta, &[0.0; 3]);
    let r0 = model.residual_variance_at(1, typ[3], &p.sigma.values);
    eprintln!(
        "t = {T_LATE} h: typical prediction {:.3e}, R⁰ {r0:.3e}",
        typ[3]
    );
    let eta = ebe_for(&model, &s, &p, 1e-10, false);
    let gated = objective_gaps_for::<3>(&model, &s, &p, eta.as_slice(), false);
    FORCE_WOODBURY.with(|c| c.set(true));
    let ungated = objective_gaps_for::<3>(&model, &s, &p, eta.as_slice(), false);
    FORCE_WOODBURY.with(|c| c.set(false));
    eprintln!(
        "near-singular row: gated F gap {:.3e}, ungated {:.3e}",
        gated.0, ungated.0
    );
    // Measured on macOS arm64 (#2441, t = 20 h): ungated 7.4e-2 relative, gated 7.7e-12. The
    // ungated bound sits 10× inside its side; the gated one (kept from the Linux-measured
    // t = 23 h fixture, 2.3e-11) has ~30× headroom here. Either reddens if the gate stops
    // separating them.
    assert!(
        ungated.0 > 7e-3,
        "the fixture no longer exposes the Woodbury loss (ungated gap {:.2e})",
        ungated.0
    );
    assert!(
        gated.0 < 2.5e-10,
        "gated FOCE F vs production: {:.2e}",
        gated.0
    );
}

/// #2093 review (round 2, finding 17): a subject with no observation rows (a dose-only record,
/// contributing its prior alone) is served under Laplace's exact anchor as under every other
/// objective. The taped-Hessian sweep chose its directions from the largest row's `Q`-adjoint
/// and declined when there was no row; with none, any basis is exact.
#[test]
fn a_subject_without_observations_is_served_under_the_exact_anchor() {
    let model = parse_model_string(ONE_CPT_ORAL).unwrap();
    let mut s = subject(
        &model,
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &TIMES,
        &[0.1, -0.05, 0.2],
    );
    s.obs_times.clear();
    s.observations.clear();
    s.obs_cmts.clear();
    s.cens.clear();
    s.occasions.clear();
    let params = moved_params(&model);
    let x = pack_params(&params);
    let laplace = Objective::Agq {
        n_agq: 1,
        exact: true,
    };
    let g = subject_adjoint_gradient(&model, &s, &params, &x, &[0.0; 3], laplace)
        .expect("a dose-only subject must be served under the exact anchor");
    let focei = subject_adjoint_gradient(&model, &s, &params, &x, &[0.0; 3], Objective::Focei)
        .expect("and under FOCEI");
    // With no rows the objective is the prior's alone, `½ηᵀΩ⁻¹η + ½log|Ω| + ½log|H|` with
    // `H = Ω⁻¹` exactly, which is identically 0 at the mode `η̂ = 0` whatever the parameters:
    // the exact gradient is zero on every coordinate, under either objective.
    for (k, (a, b)) in g.iter().zip(&focei).enumerate() {
        assert!(
            a.is_finite() && a.abs() < 1e-12 && b.abs() < 1e-12,
            "x[{k}]: Laplace {a}, FOCEI {b}"
        );
    }
}

/// #2093 review (round 2, finding 17): the exact anchor's covariance nested dual differentiates
/// the inverse Mills ratio (M3 rows) and a serial residual correlation instead of poisoning to
/// `NaN` and declining the subject to the stencil. Both against the reconverged FD of the
/// production gradient.
#[test]
fn exact_anchor_r_matrix_serves_m3_rows_and_a_serial_correlation() {
    let laplace = Objective::Agq {
        n_agq: 1,
        exact: true,
    };
    let mut m3 = parse_model_string(ONE_CPT_ORAL).unwrap();
    m3.bloq_method = crate::types::BloqMethod::M3;
    let s3 = censor(
        subject(
            &m3,
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            &TIMES,
            &[0.1, -0.05, 0.2],
        ),
        &[0, 8],
    );
    let (fd, _) = hessian_gaps_for(&m3, &s3, laplace);
    eprintln!("exact-anchor R-matrix, M3 rows: {fd:.2e}");
    // Measured 7.8e-7 (M3) and 7.7e-8 (ar1) on Linux; ~10× headroom.
    assert!(fd < 8e-6, "M3 rows: {fd:.2e}");
    let serial = parse_model_string(&dense_src("  correlation ar1(RHO) by ARM\n", false)).unwrap();
    let ss = dense_subject(&serial);
    let (fd, _) = hessian_gaps_for(&serial, &ss, laplace);
    eprintln!("exact-anchor R-matrix, serial ar1: {fd:.2e}");
    assert!(fd < 8e-7, "serial ar1: {fd:.2e}");
}

// ---- the exact inner Hessian through the predictions (`exact_inner_hessian_kernel`) ------

/// Arbitrary but live prediction jets for `s`'s rows: `f` stays positive (a proportional
/// variance is defined), `G` and `Q` carry no zero entries.
fn live_jets<const N: usize, const M: usize>(n: usize) -> (Vec<f64>, Vec<[f64; N]>, Vec<[f64; M]>) {
    let f = (0..n)
        .map(|j| 0.6 + 0.4 * (1.7 * j as f64).sin().abs())
        .collect();
    let g = (0..n)
        .map(|j| std::array::from_fn(|k| 0.3 + 0.4 * (1.3 * j as f64 + 0.7 * k as f64).sin()))
        .collect();
    let q = (0..n)
        .map(|j| std::array::from_fn(|t| 0.25 + 0.2 * (0.9 * j as f64 + 1.1 * t as f64).cos()))
        .collect();
    (f, g, q)
}

/// The worst relative entry difference between the closed form (rows over `active`) and the
/// generic definition (`inner_l` over a `DualH`) on the same jets.
fn closed_vs_generic<const N: usize, const M: usize>(
    model: &CompiledModel,
    s: &Subject,
    eta: [f64; N],
    f: &[f64],
    g: &[[f64; N]],
    q: &[[f64; M]],
    active: usize,
) -> f64 {
    use crate::sens::focei_kernel::{inner_hessian_generic, inner_hessian_through_predictions};
    let params = &model.default_params;
    let ctx = crate::sens::focei_kernel::LikCtx::new(model, s, params);
    let p = crate::sens::focei_kernel::ParT::of(model, s, params);
    let closed = inner_hessian_through_predictions(
        &ctx,
        f,
        g.as_flattened(),
        q.as_flattened(),
        N,
        active,
        &p,
    )
    .expect("closed form");
    let generic = inner_hessian_generic(&ctx, f, g, q, &eta, &p).expect("generic");
    let mut worst = 0.0f64;
    for a in 0..N {
        for b in 0..N {
            let (c, d) = (closed[a][b], generic[a][b]);
            assert!(c.is_finite() && d.is_finite(), "({a}, {b}): {c} vs {d}");
            worst = worst.max((c - d).abs() / (1.0 + d.abs()));
        }
    }
    worst
}

/// The closed form is the generic exact inner Hessian, on arbitrary jets — the identity is
/// the chain rule through each row's prediction, so it must hold for any `(f, G, Q)`, not only
/// a PK model's — on every row kind it serves: proportional, combined (with a block Ω),
/// power, and M3-censored rows.
/// Mutations, each run: dropping the `ℓ′·Q` term, the `½`, or the prior reddens it (and the
/// FD anchor below, at 6.4e-2, 9.6e-1 and 5.8e-2 on its combined case).
#[test]
fn the_inner_hessian_through_predictions_is_the_generic_one() {
    let dose = || vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    let one = parse_model_string(ONE_CPT_ORAL).unwrap();
    let combined = parse_model_string(TWO_CPT_IV_COMBINED).unwrap();
    let power = parse_model_string(ORAL_POWER).unwrap();
    let mut m3 = parse_model_string(ONE_CPT_ORAL).unwrap();
    m3.bloq_method = crate::types::BloqMethod::M3;
    let (f3, g3, q3) = live_jets::<3, 6>(TIMES.len());
    let (f2, g2, q2) = live_jets::<2, 3>(TIMES.len());
    let cases = [
        (
            "proportional",
            closed_vs_generic::<3, 6>(
                &one,
                &subject(&one, dose(), &TIMES, &[0.1, -0.05, 0.2]),
                [0.1, -0.05, 0.2],
                &f3,
                &g3,
                &q3,
                3,
            ),
        ),
        (
            "combined, block Ω",
            closed_vs_generic::<2, 3>(
                &combined,
                &subject(&combined, dose(), &TIMES, &[0.1, -0.05]),
                [0.1, -0.05],
                &f2,
                &g2,
                &q2,
                2,
            ),
        ),
        (
            "power",
            closed_vs_generic::<3, 6>(
                &power,
                &subject(&power, dose(), &TIMES, &[0.1, -0.05, 0.2]),
                [0.1, -0.05, 0.2],
                &f3,
                &g3,
                &q3,
                3,
            ),
        ),
        (
            "M3-censored rows",
            closed_vs_generic::<3, 6>(
                &m3,
                &censor(subject(&m3, dose(), &TIMES, &[0.1, -0.05, 0.2]), &[0, 7, 8]),
                [0.1, -0.05, 0.2],
                &f3,
                &g3,
                &q3,
                3,
            ),
        ),
    ];
    for (label, worst) in cases {
        // Measured worst 1.2e-15 (proportional), 2.5e-16 (combined), 1.2e-15 (power),
        // 1.8e-15 (M3): the two evaluate the same chain rule in a different order, so they
        // agree to rounding. ~10× headroom.
        assert!(
            worst < 2e-14,
            "{label}: worst relative difference {worst:.2e}"
        );
    }
}

/// Assembling the rows over the active width only is the full Hessian when the directions
/// past it are a bucket's inert padding — exactly zero `G` and `Q` — so the closed form may
/// skip them. A 2-effect model's jets in a 3-wide bucket, the third direction zeroed.
/// Mutations, each run: indexing the packed `Q` with the active width as its stride
/// (`m = active(active+1)/2`) reddens it; so does assembling over `active − 1`.
#[test]
fn the_closed_form_assembles_over_the_active_width() {
    let combined = parse_model_string(TWO_CPT_IV_COMBINED).unwrap();
    let s = subject(
        &combined,
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &TIMES,
        &[0.1, -0.05],
    );
    let (f, mut g, mut q) = live_jets::<3, 6>(TIMES.len());
    for j in 0..TIMES.len() {
        g[j][2] = 0.0;
        for k in 0..3 {
            q[j][crate::sens::vtape::tri(2, k)] = 0.0;
        }
    }
    // The 3-wide ParT: Ω padded with the inert unit diagonal, as `stacked_omega` pads.
    let params = &combined.default_params;
    let ctx = crate::sens::focei_kernel::LikCtx::new(&combined, &s, params);
    let mut p = crate::sens::focei_kernel::ParT::of(&combined, &s, params);
    let l = &params.omega.chol;
    p.omega = crate::sens::focei_kernel::OmegaT::from_chol(
        (0..3)
            .map(|r| {
                (0..3)
                    .map(|c| {
                        if r < 2 && c < 2 {
                            l[(r, c)]
                        } else if r == c {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect(),
    )
    .unwrap();
    let closed = crate::sens::focei_kernel::inner_hessian_through_predictions(
        &ctx,
        &f,
        g.as_flattened(),
        q.as_flattened(),
        3,
        2,
        &p,
    )
    .unwrap();
    let generic =
        crate::sens::focei_kernel::inner_hessian_generic(&ctx, &f, &g, &q, &[0.1, -0.05, 0.0], &p)
            .unwrap();
    for a in 0..3 {
        for b in 0..3 {
            let (c, d) = (closed[a][b], generic[a][b]);
            assert!(
                (c - d).abs() <= 2e-14 * (1.0 + d.abs()),
                "({a}, {b}): closed {c} vs generic {d}"
            );
        }
    }
    assert_eq!(
        closed[2][2], 1.0,
        "the padding keeps the prior's unit diagonal"
    );
}

/// Which way `exact_inner_hessian_kernel` formed the Hessian for `s` while `f` ran:
/// `(closed form, generic)`.
fn hessian_route(s: &Subject, f: &dyn Fn()) -> (usize, usize) {
    use crate::sens::focei_kernel::INNER_HESSIAN_ROUTE;
    let addr = s as *const Subject as usize;
    INNER_HESSIAN_ROUTE.lock().unwrap().push((addr, 0, 0));
    f();
    let mut watch = INNER_HESSIAN_ROUTE.lock().unwrap();
    let i = watch.iter().position(|e| e.0 == addr).expect("watched");
    let (_, c, g) = watch.swap_remove(i);
    (c, g)
}

/// The production sites take the closed form wherever it applies, and the generic path only
/// where a row reads η directly. Both routes give the same Hessian, so without this the
/// route is invisible: a site that silently fell back (a wrong width, a widened decline)
/// would lose the speedup with every value test green. Sites: `exact_inner_hessian` (the inner
/// loop's terminal Hessian and the quadrature objective), the Laplace gradient (its untaped
/// and taped anchors), and the Laplace covariance step.
/// Mutation, run: declining every subject (`reads_eta_directly` → `true`) reddens each site.
#[test]
fn production_sites_take_the_closed_form_inner_hessian() {
    let one = parse_model_string(ONE_CPT_ORAL).unwrap();
    let s = subject(
        &one,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &TIMES,
        &[0.1, -0.05, 0.2],
    );
    let params = moved_params(&one);
    let x = pack_params(&params);
    let eta = ebe_for(&one, &s, &params, 1e-10, true);
    let laplace = Objective::Agq {
        n_agq: 1,
        exact: true,
    };
    let r = hessian_route(&s, &|| {
        exact_inner_hessian(&one, &s, &params, eta.as_slice()).expect("served");
    });
    assert_eq!(r, (1, 0), "exact_inner_hessian");
    let r = hessian_route(&s, &|| {
        subject_adjoint_gradient(&one, &s, &params, &x, eta.as_slice(), laplace).expect("served");
    });
    assert!(
        r.0 >= 2 && r.1 == 0,
        "Laplace gradient (untaped + taped anchor): {r:?}"
    );
    let r = hessian_route(&s, &|| {
        subject_adjoint_cov_hessian(&one, &s, &params, &x, eta.as_slice(), laplace)
            .expect("served");
    });
    assert!(r.0 >= 1 && r.1 == 0, "Laplace covariance step: {r:?}");
    // A FREM row reads η directly: the generic path, and only it.
    let frem = parse_model_string(FREM_MODEL).unwrap();
    let mut fs = subject_shell(
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &[1.0, 6.0, 24.0, 0.0],
    );
    fs.fremtype = vec![0, 0, 0, 100];
    fs.observations = vec![0.8, 0.6, 0.3, 75.0];
    let r = hessian_route(&fs, &|| {
        let _ = exact_inner_hessian(&frem, &fs, &frem.default_params, &[0.1, -0.05, 0.15, 0.2]);
    });
    assert!(
        r.0 == 0 && r.1 >= 1,
        "a FREM row takes the generic path: {r:?}"
    );
}

/// Rows that read η other than through their prediction decline the closed form — a FREM
/// row (`θ_i + η_e`), the RUV-IIV scale `exp(2η_r)`, a dense residual covariance (a
/// `block_sigma` pair) — and the kernel then returns the generic Hessian. Each fixture is
/// otherwise one the closed form serves. Mutations, each run: dropping the dense, RUV-IIV or
/// FREM condition from `reads_eta_directly` reddens exactly its case.
#[test]
fn rows_that_read_eta_directly_decline_the_closed_form_hessian() {
    use crate::sens::focei_kernel::{exact_inner_hessian_kernel, inner_hessian_generic};
    let dose = || vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    fn check<const N: usize, const M: usize>(
        label: &str,
        model: &CompiledModel,
        s: &Subject,
        eta: [f64; N],
    ) {
        let params = &model.default_params;
        let ctx = crate::sens::focei_kernel::LikCtx::new(model, s, params);
        let p = crate::sens::focei_kernel::ParT::of(model, s, params);
        let (f, g, q) = live_jets::<N, M>(s.observations.len());
        let got = std::cell::RefCell::new(None);
        let r = hessian_route(s, &|| {
            *got.borrow_mut() = exact_inner_hessian_kernel(&ctx, &f, &g, &q, &eta, &p, N)
        });
        let got = got.into_inner();
        assert_eq!(r, (0, 1), "{label}: route {r:?}");
        assert_eq!(
            got,
            inner_hessian_generic(&ctx, &f, &g, &q, &eta, &p),
            "{label}"
        );
    }
    let frem = parse_model_string(FREM_MODEL).unwrap();
    let mut fs = subject_shell(dose(), &[1.0, 6.0, 24.0, 0.0]);
    fs.fremtype = vec![0, 0, 0, 100];
    fs.observations = vec![0.8, 0.6, 0.3, 75.0];
    check::<4, 10>("a FREM row", &frem, &fs, [0.1, -0.05, 0.15, 0.2]);
    let ruv = parse_model_string(TWO_CPT_ODE_RUV).unwrap();
    check::<3, 6>(
        "the RUV-IIV scale",
        &ruv,
        &subject(&ruv, dose(), &TIMES, &[0.1, -0.05, 0.1]),
        [0.1, -0.05, 0.1],
    );
    let dense = parse_model_string(&dense_src("", true)).unwrap();
    check::<2, 3>(
        "a dense residual covariance",
        &dense,
        &dense_subject(&dense),
        [0.1, -0.15],
    );
}

/// A row whose objective is undefined at these jets (an infinite prediction, so no finite variance) makes
/// the kernel return `None` at once: the generic path would fail at the same row, so it is
/// not re-run — on the scalar tape that would leave the closed form's partial recording as
/// dead nodes and record the M-lane evaluation a second time, only to reach `None` again.
/// Mutation, run: falling back to the generic path on a closed-form `None` reddens the route.
#[test]
fn an_undefined_row_returns_none_without_the_generic_rerun() {
    use crate::sens::focei_kernel::exact_inner_hessian_kernel;
    let one = parse_model_string(ONE_CPT_ORAL).unwrap();
    let s = subject(
        &one,
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &TIMES,
        &[0.1, -0.05, 0.2],
    );
    let params = &one.default_params;
    let ctx = crate::sens::focei_kernel::LikCtx::new(&one, &s, params);
    let p = crate::sens::focei_kernel::ParT::of(&one, &s, params);
    let (mut f, g, q) = live_jets::<3, 6>(TIMES.len());
    f[4] = f64::INFINITY;
    let got = std::cell::RefCell::new(Some(Vec::new()));
    let r = hessian_route(&s, &|| {
        *got.borrow_mut() = exact_inner_hessian_kernel(&ctx, &f, &g, &q, &[0.1, -0.05, 0.2], &p, 3)
    });
    let got = got.into_inner();
    assert!(got.is_none(), "an undefined row is None");
    assert_eq!(
        r,
        (1, 0),
        "one closed-form attempt, no generic re-run: {r:?}"
    );
}

/// `exact_inner_hessian` on the closed form against Richardson differences of the production
/// objective — the external oracle, sharing no derivative code — on the row kinds the
/// proportional `exact_inner_hessian_matches_fd_of_the_production_objective` does not reach:
/// M3-censored rows (the `ℓ″` of `log Φ` through the 1-D second-order dual) and the combined
/// error model (a variance with both an `f`-proportional and a constant loading).
/// Mutation, run: halving `log Φ`'s second derivative — a wrong `ℓ″` on the M3 rows, which the
/// parity test cannot see because both of its sides share `log_phi` — gives 6.1e-2 against
/// the M3 bound of 5e-6, and nothing else reddens.
#[test]
fn exact_inner_hessian_matches_fd_on_m3_and_combined_rows() {
    let doses = || {
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ]
    };
    let mut m3 = parse_model_string(ONE_CPT_ORAL).unwrap();
    m3.bloq_method = crate::types::BloqMethod::M3;
    let m3_s = censor(
        subject(&m3, doses(), &TIMES, &[0.1, -0.05, 0.2]),
        &[0, 7, 8],
    );
    let combined = parse_model_string(TWO_CPT_IV_COMBINED).unwrap();
    let comb_s = subject(&combined, doses(), &TIMES, &[0.1, -0.05]);
    let mut results = Vec::new();
    for (label, model, s, eta) in [
        ("M3-censored rows", &m3, &m3_s, vec![0.08, -0.06, 0.1]),
        ("combined error", &combined, &comb_s, vec![0.08, -0.06]),
    ] {
        let p = &model.default_params;
        let n = eta.len();
        let h = exact_inner_hessian(model, s, p, &eta).expect("exact Hessian");
        let route = hessian_route(s, &|| {
            exact_inner_hessian(model, s, p, &eta).expect("exact Hessian");
        });
        assert_eq!(route, (1, 0), "{label}: on the closed form");
        let l = |e: &[f64]| {
            let mut scratch = crate::pk::EventPkParams::with_capacity_for(s);
            crate::stats::likelihood::individual_nll_into_with_schedule_cov(
                model,
                s,
                &p.theta,
                e,
                &p.omega,
                &p.sigma.values,
                model.residual_cov_with(&p.residual_correlations, &p.theta),
                &mut scratch,
                None,
            )
        };
        let mut worst = 0.0f64;
        for a in 0..n {
            for b in 0..n {
                let d = |st: f64| {
                    let at = |sa: f64, sb: f64| {
                        let mut e = eta.clone();
                        e[a] += sa;
                        e[b] += sb;
                        l(&e)
                    };
                    (at(st, st) - at(st, -st) - at(-st, st) + at(-st, -st)) / (4.0 * st * st)
                };
                let rich = (4.0 * d(1e-3) - d(2e-3)) / 3.0;
                assert!(
                    rich.is_finite() && h[(a, b)].is_finite(),
                    "{label} ({a}, {b})"
                );
                worst = worst.max((h[(a, b)] - rich).abs() / (1.0 + rich.abs()));
            }
        }
        eprintln!("MEASURE {label}: exact Hessian vs Richardson FD worst {worst:.2e}");
        results.push((label, worst));
    }
    // Bounds from the measured worst, per row kind, because the oracle's accuracy differs:
    // * combined: measured 2.27e-11; 3e-10 (13× headroom).
    // * M3: measured 8.26e-7. That gap is the oracle's, not the kernel's: production's
    //   `erf` (`stats::special`, Abramowitz–Stegun 7.1.26) carries up to 1.5e-7 absolute
    //   error, so differences of its `log Φ` cannot resolve an exact second derivative
    //   below that level (`kernel_m3_censored_rows_match_reconverged_fd` bounds its M3
    //   gradient at 1e-5 for the same reason). 5e-6 (6× headroom); a wrong `ℓ″` lands far
    //   above it (see the mutation note on this test).
    for (label, worst) in results {
        let bound = if label.starts_with("M3") { 5e-6 } else { 3e-10 };
        assert!(worst < bound, "{label}: {worst:.2e} vs bound {bound:.0e}");
    }
}

/// `∂f_j/∂θ_m` at a fixed `eta` by forward mode: one recording per θ with that θ's unit
/// tangent (`DualF<1, 0>`) — the same recorder, differentiated forwards, sharing no
/// reverse-pass code with [`fixed_eta_theta_adjoint`].
fn forward_theta_jacobian(
    model: &CompiledModel,
    s: &Subject,
    theta: &[f64],
    eta: &[f64],
    n_axis: usize,
) -> Vec<Vec<f64>> {
    (0..n_axis)
        .map(|m| {
            let th: Vec<DualF<1, 0>> = theta
                .iter()
                .enumerate()
                .map(|(k, &v)| DualF::free(v, [if k == m { 1.0 } else { 0.0 }], []))
                .collect();
            let et: Vec<DualF<1, 0>> = eta.iter().map(|&e| DualF::constant(e)).collect();
            record(model, s, None, theta, eta, &ThetaLeaves::theta(&th), &et)
                .expect("in scope")
                .iter()
                .map(|q| q.g[0])
                .collect()
        })
        .collect()
}

/// The fixed-η θ adjoint (SAEM's M-step and VI's score) against forward-mode tangents of
/// the same recorder: the contracted gradient and every `∂f_j/∂θ_m` of the Jacobian, on
/// ODE (RK45 and Rosenbrock), multi-dose infusion, a 14-parameter chain and closed-form
/// models. Both are exact derivatives of the same discrete trajectory, so they agree to
/// rounding; no finite difference anywhere.
#[test]
fn the_fixed_eta_theta_adjoint_is_the_forward_theta_tangents() {
    let mut rodas = parse_model_string(TWO_CPT_INF_COMBINED_BLOCK).unwrap();
    rodas.ode_spec.as_mut().unwrap().solver_opts.method = crate::ode::solver::OdeMethod::Rodas5P;
    let infusions = vec![
        DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0),
        DoseEvent::new(8.0, 100.0, 1, 50.0, false, 0.0),
    ];
    let bolus = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    let cases: Vec<(&str, CompiledModel, Vec<DoseEvent>, Vec<f64>)> = vec![
        (
            "one_cpt_oral_ode",
            parse_model_string(ONE_CPT_ORAL_ODE).unwrap(),
            bolus.clone(),
            vec![0.1, -0.05],
        ),
        (
            "two_cpt_infusion",
            parse_model_string(TWO_CPT_INF_COMBINED_BLOCK).unwrap(),
            infusions.clone(),
            vec![0.1, -0.05, 0.08],
        ),
        ("two_cpt_rodas5p", rodas, infusions, vec![0.1, -0.05, 0.08]),
        (
            "additive_four_eta",
            parse_model_string(ADDITIVE_FOUR_ETA).unwrap(),
            bolus.clone(),
            vec![0.1, -0.05, 0.2, -0.1],
        ),
        (
            "chain_12",
            parse_model_string(&chain_model(12)).unwrap(),
            bolus.clone(),
            vec![0.1, -0.05, 0.1],
        ),
        (
            "one_cpt_oral_closed_form",
            parse_model_string(ONE_CPT_ORAL).unwrap(),
            bolus.clone(),
            vec![0.1, -0.05],
        ),
        (
            "two_cpt_iv_closed_form",
            parse_model_string(TWO_CPT_IV_COMBINED).unwrap(),
            bolus.clone(),
            vec![0.1, -0.05, 0.08],
        ),
        // Between-occasion variability: η then one occasion's κ, stacked.
        (
            "ode_iov",
            parse_model_string(ODE_IOV).unwrap(),
            bolus,
            vec![0.1, -0.05, 0.08, -0.03],
        ),
    ];
    let mut worst_all = 0.0f64;
    for (name, model, doses, eta) in cases {
        let mut worst = 0.0f64;
        let width = model.n_eta + model.n_kappa;
        let mut eta = eta[..width.min(eta.len())].to_vec();
        eta.resize(width, 0.05);
        let s = subject(&model, doses, &TIMES, &eta[..model.n_eta]);
        let theta = model.default_params.theta.clone();
        let weights: Vec<f64> = (0..s.observations.len())
            .map(|j| (0.7 * j as f64 + 0.3).sin())
            .collect();
        let x = fixed_eta_theta_adjoint(&model, &s, &theta, &eta, true, &weights)
            .unwrap_or_else(|| panic!("{name}: out of the recorders' scope"));
        let n_axis = theta_axis(&model).unwrap().min(theta.len());
        let fwd = forward_theta_jacobian(&model, &s, &theta, &eta, n_axis);
        let jac = x.jac.as_ref().expect("asked for");
        let n_theta = theta.len();
        let mut live = 0;
        for m in 0..n_axis {
            let g_fwd: f64 = fwd[m].iter().zip(&weights).map(|(d, w)| d * w).sum();
            assert!(g_fwd.is_finite() && x.grad[m].is_finite(), "{name}: θ{m}");
            worst = worst.max((x.grad[m] - g_fwd).abs() / (1.0 + g_fwd.abs()));
            for j in 0..s.observations.len() {
                let (a, f) = (jac[j * n_theta + m], fwd[m][j]);
                assert!(a.is_finite() && f.is_finite(), "{name}: ∂f{j}/∂θ{m}");
                worst = worst.max((a - f).abs() / (1.0 + f.abs()));
                live += (f != 0.0) as usize;
            }
        }
        // Non-degenerate: the predictions really depend on the θ under test.
        assert!(
            live > s.observations.len(),
            "{name}: only {live} live entries"
        );
        // Measured: 3.8e-16 on the explicit and closed-form engines; 1.2e-10 on Rodas5P,
        // where the taped side solves `W` through the vector tape's block LU and the
        // forward side element by element over the dual, and the stiff `W`'s conditioning
        // amplifies that rounding difference. ~25× and ~9× headroom.
        let bound = if name.contains("rodas") { 1e-9 } else { 1e-14 };
        assert!(worst < bound, "{name}: worst relative difference {worst:e}");
        worst_all = worst_all.max(worst);
    }
    eprintln!("fixed-η θ adjoint vs forward tangents: worst relative difference {worst_all:e}");
}
/// The lift is taken where its jet type is no more work per operation than the recording's
/// (`pk_space_predictions`). Each case names the fit it decides: a taped FOCEI recording over
/// as many PK slots lifts (the tape is what it drops); a two-η Michaelis–Menten fit over four
/// live slots does not, at order 2 or under Laplace's directional order 3, where
/// `Taylor<4, 3>`'s 165-term product was measured slower than the recording it replaced.
#[test]
fn the_lift_is_taken_where_it_is_less_work() {
    use crate::sens::num::PkNum;
    let lifts = |order: usize, n: usize, work: usize| {
        super::pk_space_work(order, n).is_some_and(|w| w <= work)
    };
    // Five η over five slots, FOCEI's taped recording.
    assert!(lifts(2, 5, <DualV<6, 21> as PkNum>::WORK));
    // Two η over four slots: FOCEI, and Laplace's exact anchor (`DualV<N, N, true>`).
    assert!(!lifts(2, 4, <DualV<2, 3> as PkNum>::WORK));
    assert!(!lifts(3, 4, <DualV<2, 2, true> as PkNum>::WORK));
    // The exact anchor's widths, counted at their stored lanes (`1 + 2N`, all taped): eight
    // η over two slots lifts; four η over two, and 24 over four, do not. Four η over two
    // slots was measured (a Laplace gradient, three η, two on PK slots): lifted 1.75× the
    // generic recording's time, at a 4.5e-15 gap — so not lifting it is the faster choice.
    assert_eq!(<DualV<4, 4, true> as PkNum>::WORK, 27);
    assert!(lifts(3, 2, <DualV<8, 8, true> as PkNum>::WORK));
    assert!(!lifts(3, 2, <DualV<4, 4, true> as PkNum>::WORK));
    assert!(!lifts(3, 4, <DualV<24, 24, true> as PkNum>::WORK));
    // Many η over few slots lifts even untaped.
    assert!(lifts(2, 2, <DualF<8, 36> as PkNum>::WORK));
    // Past the widest bucket there is no lift at all.
    assert_eq!(super::pk_space_work(2, 9), None);
    assert_eq!(super::pk_space_work(4, 7), None);
    // The costs themselves: `1 + p + p(p+1)/2` at the bucket, and the Taylor product.
    assert_eq!(super::pk_space_work(2, 5), Some(28));
    assert_eq!(super::pk_space_work(3, 4), Some(165));
}

/// The exact outer gradient of a subject carrying non-Gaussian records: the term `q`
/// folded into the FOCE/FOCEI objective (`2q` in the data, `∇²q` in `H̃`), its third-order
/// η-jets in the mode system, its θ-derivatives contracted with the tape's adjoints. Against
/// the Richardson-extrapolated, EBE-reconverged difference of the production objective
/// (`fd_oracle`, which shares no derivative code with the adjoint), under both interaction
/// settings, on a drug-driven (ODE-accumulated) hazard and on an analytic one.
#[cfg(feature = "survival")]
#[test]
fn the_outer_gradient_is_exact_with_a_non_gaussian_term() {
    let root = env!("CARGO_MANIFEST_DIR");
    let mut worst = 0.0f64;
    for (m, d) in [
        ("examples/pktte_joint.ferx", "data/pktte_joint.csv"),
        ("examples/tte_exponential.ferx", "data/tte_exponential.csv"),
    ] {
        let mut prep =
            crate::api::prepare_run(&format!("{root}/{m}"), Some(&format!("{root}/{d}"))).unwrap();
        if let Some(ode) = prep.parsed.model.ode_spec.as_mut() {
            ode.solver_opts.reltol = 1e-12;
            ode.solver_opts.abstol = 1e-14;
        }
        let model = &prep.parsed.model;
        assert!(
            outer_model_in_scope(model),
            "{m}: the outer route is analytic"
        );
        let s = prep
            .population
            .subjects
            .iter()
            .find(|s| !s.obs_records.is_empty())
            .unwrap();
        for interaction in [true, false] {
            let w = fd_oracle(model, s, interaction);
            eprintln!("{m} interaction={interaction}: worst {w:.2e}");
            worst = worst.max(w);
        }
    }
    // Measured: 5.3e-9 / 1.7e-8 (FOCEI / FOCE) on the joint PK-TTE model at reltol 1e-12,
    // 6.2e-13 on the analytic hazard — the reconverged oracle's own floor. A dropped
    // `∇²q` in `H̃`, a missing third-order jet or the conditional-for-frozen `∇l` mix-up
    // under FOCE (measured 2.3e-2 before it was fixed) misses by orders more.
    assert!(worst < 5e-7, "worst relative difference {worst:.2e}");
}

/// The exact outer gradient on binary and CTMM endpoints — endpoint-only and beside a PK
/// model — against the reconverged oracle, under both interaction settings.
#[cfg(all(feature = "survival", feature = "markov"))]
#[test]
fn the_outer_gradient_is_exact_on_binary_and_ctmm_endpoints() {
    use crate::types::ObsRecord;
    let states = |cmt: usize, rows: &[(f64, usize)]| -> Vec<ObsRecord> {
        rows.iter()
            .map(|&(time, state)| ObsRecord::DiscreteState {
                time,
                raw_time: time,
                state,
                cmt,
            })
            .collect()
    };
    let ctmm = parse_model_string(
        "[parameters]\n  theta LQ01(-0.7, -6.0, 3.0)\n  theta LQ10(-1.2, -6.0, 3.0)\n  \
         omega ETA_Q ~ 0.1\n[markov_model]\n  type   = ctmm\n  cmt    = 5\n  \
         states = [awake=0, asleep=1]\n  transition awake  -> asleep = exp(LQ01 + ETA_Q)\n  \
         transition asleep -> awake  = exp(LQ10 - 0.5 * ETA_Q)\n",
    )
    .unwrap();
    let ctmm_s = Subject {
        id: "1".into(),
        obs_records: states(
            5,
            &[(0.0, 0), (1.0, 0), (2.3, 1), (3.1, 1), (5.0, 0), (7.5, 1)],
        ),
        ..Default::default()
    };
    let binary = parse_model_string(
        "[parameters]\n  theta TH0(-0.3, -10.0, 10.0)\n  theta THT(0.2, -10.0, 10.0)\n  \
         omega ETA_I ~ 0.3\n[binary_model]\n  cmt   = 3\n  \
         logit = TH0 + ETA_I + THT * TIME * exp(0.5 * ETA_I)\n",
    )
    .unwrap();
    let binary_s = Subject {
        id: "1".into(),
        obs_records: states(3, &[(0.0, 0), (1.0, 1), (2.0, 1), (4.0, 0), (6.0, 1)]),
        ..Default::default()
    };
    // No random effects at all (`examples/binary_logistic.ferx`'s shape): the term's jets
    // have width 0, the value alone.
    let fixed = parse_model_string(
        "[parameters]\n  theta TH0(-0.3, -10.0, 10.0)\n  theta THT(0.2, -10.0, 10.0)\n\
         [binary_model]\n  cmt   = 3\n  logit = TH0 + THT * TIME\n",
    )
    .unwrap();
    assert_eq!(fixed.n_eta, 0);
    let mut worst = 0.0f64;
    for (tag, model, s) in [
        ("ctmm", &ctmm, &ctmm_s),
        ("binary", &binary, &binary_s),
        ("binary, no η", &fixed, &binary_s),
    ] {
        assert!(
            outer_model_in_scope(model),
            "{tag}: the outer route is analytic"
        );
        for interaction in [true, false] {
            let w = fd_oracle(model, s, interaction);
            eprintln!("{tag} interaction={interaction}: worst {w:.2e}");
            worst = worst.max(w);
        }
    }
    eprintln!("binary / CTMM outer gradient vs the reconverged oracle: worst {worst:.2e}");
    // Measured 1.5e-12: the oracle's floor on these closed-form terms.
    assert!(worst < 5e-7, "{worst:.2e}");
}

/// The exact Laplace / AGQ gradient of a subject carrying non-Gaussian records: the term in
/// the exact anchor (`∇²q`), at `η̂` (`q`, `∇q`, third-order jets for the mode system) and at
/// every node (its value, linearised in `b` there). Against the Richardson difference of
/// production's AGQ objective with the EBE re-solved at every point (`fd_oracle_agq`),
/// at one node (Laplace) and three, on an ODE hazard beside PK and on an analytic hazard.
#[cfg(feature = "survival")]
#[test]
fn the_agq_gradient_is_exact_with_a_non_gaussian_term() {
    let root = env!("CARGO_MANIFEST_DIR");
    let mut worst = 0.0f64;
    for (m, d) in [
        ("examples/pktte_joint.ferx", "data/pktte_joint.csv"),
        ("examples/tte_exponential.ferx", "data/tte_exponential.csv"),
    ] {
        let mut prep =
            crate::api::prepare_run(&format!("{root}/{m}"), Some(&format!("{root}/{d}"))).unwrap();
        if let Some(ode) = prep.parsed.model.ode_spec.as_mut() {
            ode.solver_opts.reltol = 1e-12;
            ode.solver_opts.abstol = 1e-14;
        }
        let model = &prep.parsed.model;
        let s = prep
            .population
            .subjects
            .iter()
            .find(|s| !s.obs_records.is_empty())
            .unwrap();
        // Both anchors: the Gauss–Newton one carries the term's `∇²q` in `H̃`, as
        // production's `exact_htilde` does.
        for (n_agq, exact) in [(1, true), (3, true), (3, false)] {
            let w = fd_oracle_agq(model, s, n_agq, exact);
            eprintln!("{m} n_agq={n_agq} exact={exact}: worst {w:.2e}");
            worst = worst.max(w);
        }
    }
    eprintln!("non-Gaussian AGQ gradient: worst {worst:.2e}");
    assert!(worst < 5e-7, "worst relative difference {worst:.2e}");
}

/// The covariance step's exact Hessian on subjects carrying non-Gaussian records — a
/// drug-driven (ODE) hazard beside PK, an analytic hazard, a binary and a CTMM endpoint —
/// against Richardson differences of the exact gradient with the EBE re-solved at every
/// point ([`hessian_gaps`]), under both interaction settings. The term's `q`, `∇q`, `∇²q`
/// ride the pass's tape, so the profile Hessian sees their θ-, η- and mode-response
/// derivatives; dropping any of them (or `objective_ng`'s FOCE frozen `∇l`) moves the gap
/// to the size of that term's curvature.
#[cfg(all(feature = "survival", feature = "markov"))]
#[test]
fn the_cov_hessian_is_exact_with_a_non_gaussian_term() {
    use crate::types::ObsRecord;
    let root = env!("CARGO_MANIFEST_DIR");
    let mut cases: Vec<(String, CompiledModel, Subject)> = Vec::new();
    for (m, d) in [
        ("examples/pktte_joint.ferx", "data/pktte_joint.csv"),
        ("examples/tte_exponential.ferx", "data/tte_exponential.csv"),
    ] {
        let mut prep =
            crate::api::prepare_run(&format!("{root}/{m}"), Some(&format!("{root}/{d}"))).unwrap();
        if let Some(ode) = prep.parsed.model.ode_spec.as_mut() {
            ode.solver_opts.reltol = 1e-12;
            ode.solver_opts.abstol = 1e-14;
        }
        let s = prep
            .population
            .subjects
            .iter()
            .find(|s| !s.obs_records.is_empty())
            .unwrap()
            .clone();
        cases.push((m.to_string(), prep.parsed.model, s));
    }
    let states = |cmt: usize, rows: &[(f64, usize)]| -> Vec<ObsRecord> {
        rows.iter()
            .map(|&(time, state)| ObsRecord::DiscreteState {
                time,
                raw_time: time,
                state,
                cmt,
            })
            .collect()
    };
    let ctmm = parse_model_string(
        "[parameters]\n  theta LQ01(-0.7, -6.0, 3.0)\n  theta LQ10(-1.2, -6.0, 3.0)\n  \
         omega ETA_Q ~ 0.1\n[markov_model]\n  type   = ctmm\n  cmt    = 5\n  \
         states = [awake=0, asleep=1]\n  transition awake  -> asleep = exp(LQ01 + ETA_Q)\n  \
         transition asleep -> awake  = exp(LQ10 - 0.5 * ETA_Q)\n",
    )
    .unwrap();
    let ctmm_s = Subject {
        id: "1".into(),
        obs_records: states(
            5,
            &[(0.0, 0), (1.0, 0), (2.3, 1), (3.1, 1), (5.0, 0), (7.5, 1)],
        ),
        ..Default::default()
    };
    let binary = parse_model_string(
        "[parameters]\n  theta TH0(-0.3, -10.0, 10.0)\n  theta THT(0.2, -10.0, 10.0)\n  \
         omega ETA_I ~ 0.3\n[binary_model]\n  cmt   = 3\n  \
         logit = TH0 + ETA_I + THT * TIME * exp(0.5 * ETA_I)\n",
    )
    .unwrap();
    let binary_s = Subject {
        id: "1".into(),
        obs_records: states(3, &[(0.0, 0), (1.0, 1), (2.0, 1), (4.0, 0), (6.0, 1)]),
        ..Default::default()
    };
    cases.push(("ctmm".into(), ctmm, ctmm_s));
    cases.push(("binary".into(), binary, binary_s));
    let mut worst = 0.0f64;
    for (tag, model, s) in &cases {
        for objective in [
            Objective::Focei,
            Objective::Foce,
            Objective::Agq {
                n_agq: 1,
                exact: true,
            },
            Objective::Agq {
                n_agq: 3,
                exact: true,
            },
            Objective::Agq {
                n_agq: 3,
                exact: false,
            },
        ] {
            assert!(
                super::cov_non_gaussian_served(model, objective),
                "{tag}: the model-level gate serves it"
            );
            let (gap, _) = hessian_gaps_for(model, s, objective);
            eprintln!("{tag} {objective:?}: cov Hessian vs FD of the exact gradient {gap:.2e}");
            worst = worst.max(gap);
        }
    }
    eprintln!("non-Gaussian cov Hessian: worst {worst:.2e}");
    assert!(worst < 1e-5, "worst relative gap {worst:.2e}");
}

/// `factor_anchor` is production's `agq_proposal` rule on the tape's number type (#2538):
/// the same branch and the same `log|·|` on an anchor that factorises (no jitter), on one
/// only the jittered anchor factorises, and on one only `Ω⁻¹` does.
#[test]
fn the_tape_anchor_factor_is_productions() {
    use nalgebra::DMatrix;
    let omega_inv = DMatrix::from_row_slice(2, 2, &[4.0, 0.0, 0.0, 9.0]);
    let rows = |m: &DMatrix<f64>| -> Vec<Vec<f64>> {
        (0..m.nrows())
            .map(|r| (0..m.ncols()).map(|c| m[(r, c)]).collect())
            .collect()
    };
    let cases = [
        // Positive-definite: `log|H|` itself.
        (
            DMatrix::from_row_slice(2, 2, &[2.0, 0.5, 0.5, 3.0]),
            (5.75f64).ln(),
        ),
        // Singular PSD: only `H + jitter` factorises.
        (
            DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]),
            (1.0f64 + 1e-6).ln() + (1e-10f64).ln(),
        ),
        // Indefinite: the prior precision.
        (
            DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, -1.0]),
            (36.0f64).ln(),
        ),
    ];
    for (h, want) in cases {
        let prod = crate::estimation::agq::agq_proposal(&h, &omega_inv, 2)
            .expect("a factor")
            .log_det_inv_scale;
        let mut ht = rows(&h);
        let l = factor_anchor(&mut ht, &rows(&omega_inv)).expect("a factor");
        let tape = crate::sens::focei_kernel::chol_logdet(&l);
        assert!(
            (prod - want).abs() < 1e-12,
            "production {prod} vs {want} on {h}"
        );
        assert!((tape - want).abs() < 1e-12, "tape {tape} vs {want} on {h}");
    }
}
