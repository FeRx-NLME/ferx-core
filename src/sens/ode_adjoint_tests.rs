//! The adjoint gradient, covariance Hessian and exact η-quantities against external
//! oracles: Richardson differences of the production objective with the EBE re-solved at
//! every point (which shares no derivative code with the adjoint), and of the production
//! predictor.

use super::*;
use crate::estimation::inner_optimizer::find_ebe;
use crate::estimation::parameterization::pack_params;
use crate::parser::model_parser::parse_model_string;
use crate::sens::dual2::Dual2;
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
        id: "1".to_string(),
        doses,
        obs_times: times.to_vec(),
        observations: vec![0.0; n],
        obs_cmts: vec![1; n],
        cens: vec![0; n],
        occasions: vec![1; n],
        ..Default::default()
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

/// LTBS with a natural-unit additive part (#2729): `Var(log y) = PROP² + ADD²/F²`. The
/// concentrations fall from ~9 to ~0.1 over the record, so `ADD/F` runs from ~0.03 (below
/// `PROP`) to ~3 (far above it): both terms are live, and `∂V/∂f` is non-zero on every
/// row — the first LTBS model where the FOCEI interaction term is. `{MAG}` is spliced
/// with a magnitude on the proportional slot.
const ONE_CPT_ORAL_LTBS_COMBINED: &str = r#"
[parameters]
  theta TVCL(2.0, 0.01, 50.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta PROP_EARLY(1.8, 0.1, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP_ERR ~ 0.1 (sd)
  sigma ADD_ERR ~ 0.3 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  log(DV) ~ combined(PROP_ERR{MAG}, ADD_ERR)
"#;

fn ltbs_combined_fixture(mag: &str) -> (CompiledModel, Subject) {
    let model = parse_model_string(&ONE_CPT_ORAL_LTBS_COMBINED.replace("{MAG}", mag)).unwrap();
    assert_eq!(model.error_model, crate::types::ErrorModel::LogCombined);
    assert!(model.log_transform && !model.dv_pre_logged);
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let times = [0.5, 1.0, 2.0, 4.0, 8.0, 11.5, 13.0, 16.0, 20.0, 30.0];
    let s = subject(&model, doses, &times, &[0.1, -0.05]);
    // Both terms live: the natural-scale predictions straddle `ADD/PROP = 3`.
    let preds = crate::pk::compute_predictions_with_tv(
        &model,
        &s,
        &model.default_params.theta,
        &[0.1, -0.05],
    );
    let nat: Vec<f64> = preds.iter().map(|g| g.exp()).collect();
    assert!(
        nat.iter().any(|&f| f > 6.0) && nat.iter().any(|&f| f < 0.5),
        "the fixture must put rows on both sides of ADD/PROP: {nat:?}"
    );
    (model, s)
}

#[test]
fn analytic_adjoint_matches_reconverged_fd_ltbs_combined() {
    let (model, s) = ltbs_combined_fixture("");
    assert!(
        in_scope(&model, &s),
        "LTBS combined must be served by the closed-form engine, not FD"
    );
    let worst = parity(&model, &s);
    eprintln!("LTBS combined FOCEI vs reconverged FD {worst:.2e}");
    // Measured 4.7e-10 (Linux x86_64); 5× headroom.
    assert!(
        worst < 2.5e-9,
        "FOCEI worst relative difference {worst:.2e}"
    );
    let worst = parity_foce(&model, &s);
    eprintln!("LTBS combined FOCE vs reconverged FD {worst:.2e}");
    // Measured 3.5e-9; 5× headroom.
    assert!(worst < 1.8e-8, "FOCE worst relative difference {worst:.2e}");
}

/// The same with a magnitude on the proportional slot only — the motivating model's
/// `PROP = THETA(6)*THETA(12)` at early times: the multiplier rides the constant log-scale
/// loading, not the `exp(−f)` one.
#[test]
fn analytic_adjoint_matches_reconverged_fd_ltbs_combined_magnitude() {
    let (model, s) = ltbs_combined_fixture(" * (if (TIME < 3) PROP_EARLY else 1)");
    assert!(model.has_custom_ruv_magnitude());
    assert!(in_scope(&model, &s), "served by the closed-form engine");
    let worst = parity(&model, &s);
    eprintln!("LTBS combined + magnitude FOCEI vs reconverged FD {worst:.2e}");
    // Measured 3.3e-10; 5× headroom.
    assert!(worst < 1.7e-9, "worst relative difference {worst:.2e}");
}

/// The kernel's objective is production's, and its covariance R-matrix is the
/// reconverged-FD Hessian of it.
#[test]
fn ltbs_combined_kernel_objective_and_hessian_match_production() {
    let (model, s) = ltbs_combined_fixture("");
    let g = objective_gaps(&model, &s, &model.default_params, &[0.1, -0.05]);
    eprintln!("LTBS combined objective gaps {g:?}");
    assert!(g.0 < 1e-12 && g.1 < 1e-12, "kernel vs production: {g:?}");
    let (fd, _) = hessian_gaps_for(&model, &s, Objective::Focei);
    eprintln!("LTBS combined FOCEI Hessian vs reconverged FD {fd:.2e}");
    // Measured 9.1e-12; 5× headroom.
    assert!(fd < 5e-11, "FOCEI Hessian vs FD {fd:.2e}");
}

/// Per-endpoint LTBS combined error on the ODE engine (#2729): plasma (`CMT=2`) and a
/// cumulative urine amount (`CMT=3`), each with its own `PROP`/`ADD`.
const ODE_LTBS_COMBINED_PER_CMT: &str = r#"
[parameters]
  theta TVCL(2.0, 0.01, 50.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP_P ~ 0.1 (sd)
  sigma ADD_P ~ 0.3 (sd)
  sigma PROP_U ~ 0.2 (sd)
  sigma ADD_U ~ 2.0 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA
[structural_model]
  ode(states=[depot, central, urine])
[odes]
  d/dt(depot) = -KA * depot
  d/dt(central) = KA * depot - (CL/V) * central
  d/dt(urine) = 0.4 * (CL/V) * central
[scaling]
  y[CMT=2] = central / V
  y[CMT=3] = urine
[error_model]
  CMT=2: log(DV) ~ combined(PROP_P, ADD_P)
  CMT=3: log(DV) ~ combined(PROP_U, ADD_U)
[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

#[test]
fn analytic_adjoint_matches_reconverged_fd_ltbs_combined_per_cmt() {
    let model = parse_model_string(ODE_LTBS_COMBINED_PER_CMT).unwrap();
    ltbs_combined_per_cmt_parity(model, LTBS_PER_CMT_BOUND);
}

/// The same with the motivating model's early-time factor on the plasma `PROP` only,
/// written on the per-CMT argument (#2729 review, through #2731's per-endpoint
/// magnitude tables): a θ-dependent magnitude on one endpoint's constant log-scale
/// loading.
#[test]
fn analytic_adjoint_matches_reconverged_fd_ltbs_combined_per_cmt_magnitude() {
    let src = ODE_LTBS_COMBINED_PER_CMT
        .replace(
            "CMT=2: log(DV) ~ combined(PROP_P, ADD_P)",
            "CMT=2: log(DV) ~ combined(PROP_P * (if (TAD < 3) FAC else 1), ADD_P)",
        )
        .replace(
            "  sigma PROP_P",
            "  theta FAC(1.8, 0.1, 10.0)\n  sigma PROP_P",
        );
    let model = parse_model_string(&src).unwrap();
    assert!(model.has_custom_ruv_magnitude());
    ltbs_combined_per_cmt_parity(model, LTBS_PER_CMT_BOUND);
}

/// Measured on Padé 2.0e-9 / 2.2e-9 (plain / magnitude, Linux x86_64); on the spectral path
/// (#2706 follow-up) 1.1e-8 / 1.3e-8 (CI, release semantics). The analytic gradient is the
/// same on both engines to the 11 printed digits; what moves is the reconverged-FD oracle's
/// objective: the spectral propagation's values carry ~10× Padé's rounding (a sum of modes;
/// `spectral_is_exact_on_a_stiff_spectrum` measures 1.6e-12 against Padé's 1.0e-13 on an
/// exact solution), which the oracle's re-solved η̂ and its Richardson step amplify.
/// ~5× over the worst.
const LTBS_PER_CMT_BOUND: f64 = 6e-8;

fn ltbs_combined_per_cmt_parity(model: CompiledModel, bound: f64) {
    let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    let base = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0];
    let times: Vec<f64> = base.iter().flat_map(|&t| [t, t]).collect();
    let mut s = subject(&model, doses, &times, &[0.1, -0.05]);
    s.obs_cmts = (0..times.len()).map(|j| 2 + j % 2).collect();
    // `subject` built the observations against CMT 1; rebuild them per endpoint.
    let preds = crate::pk::compute_predictions_with_tv(
        &model,
        &s,
        &model.default_params.theta,
        &[0.1, -0.05],
    );
    assert!(preds.iter().all(|p| p.is_finite()), "{preds:?}");
    s.observations = preds
        .iter()
        .enumerate()
        .map(|(j, g)| g + 0.15 * (1.3 * j as f64).sin())
        .collect();
    // Both terms live on both endpoints: ADD/F straddles PROP on each.
    let nat: Vec<f64> = preds.iter().map(|g| g.exp()).collect();
    // (The magnitude variant scales PROP_P by 1.8 before t = 3, which keeps the
    // straddle: its early rows have F well above ADD/PROP.)
    for (cmt, prop, add) in [(2usize, 0.1, 0.3), (3, 0.2, 2.0)] {
        let r: Vec<f64> = (0..nat.len())
            .filter(|&j| s.obs_cmts[j] == cmt)
            .map(|j| add / nat[j] / prop)
            .collect();
        assert!(
            r.iter().any(|&x| x > 1.0) && r.iter().any(|&x| x < 1.0),
            "CMT={cmt}: ADD/(F·PROP) {r:?}"
        );
    }
    assert!(
        in_scope(&model, &s),
        "per-CMT LTBS combined must be served by the ODE adjoint, not FD"
    );
    let g = objective_gaps(&model, &s, &model.default_params, &[0.1, -0.05]);
    eprintln!("per-CMT LTBS combined objective gaps {g:?}");
    assert!(g.0 < 1e-12 && g.1 < 1e-12, "kernel vs production: {g:?}");
    let worst = parity(&model, &s);
    eprintln!("per-CMT LTBS combined FOCEI vs reconverged FD {worst:.2e}");
    assert!(worst < bound, "FOCEI worst relative difference {worst:.2e}");
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

/// A Friberg feedback loop: the `if` switches where `r` crosses `C0`, a time every η moves.
const ODE_MOVING_CONTINUOUS_SWITCH: &str = r#"
[parameters]
  theta TVK(0.3, 0.01, 5.0)
  theta TVSL(0.6, 0.01, 5.0)
  theta TVC(4.0, 0.1, 20.0)
  theta TVG(0.5, 0.01, 2.0)
  omega ETA_K ~ 0.1
  omega ETA_C ~ 0.1
  sigma ADD ~ 0.2 (sd)
[individual_parameters]
  K = TVK * exp(ETA_K)
  SL = TVSL
  C0 = TVC * exp(ETA_C)
  G = TVG
[structural_model]
  ode(states=[d, p, r])
[odes]
  init(p) = C0
  init(r) = C0
  d/dt(d) = -0.5 * d
  d/dt(p) = K * p * (1.0 - SL * d) * (if (r < C0) (C0 / r)^G else 1.0) - K * p
  d/dt(r) = K * p - K * r
[scaling]
  y = r
[error_model]
  DV ~ additive(ADD)
"#;

/// The packed gradient of `ODE_MOVING_CONTINUOUS_SWITCH` with its feedback written as `form`,
/// at that form's own mode.
fn moving_switch_gradient(form: &str) -> Vec<f64> {
    let src = ODE_MOVING_CONTINUOUS_SWITCH.replace("(if (r < C0) (C0 / r)^G else 1.0)", form);
    let model = tight(parse_model_string(&src).unwrap());
    let s = moving_switch_subject(&model);
    assert!(in_scope(&model, &s), "`{form}` must be served");
    let params = moved_params(&model);
    let x = pack_params(&params);
    let eta = ebe_for(&model, &s, &params, 1e-10, true);
    subject_adjoint_packed_gradient(&model, &s, &params, &x, eta.as_slice(), true)
        .expect("in adjoint scope")
}

/// Two doses, so the second lands on a depressed, rebounding `r`.
fn moving_switch_subject(model: &CompiledModel) -> Subject {
    subject(
        model,
        vec![
            DoseEvent::new(0.0, 4.0, 1, 0.0, false, 0.0),
            DoseEvent::new(24.0, 4.0, 1, 0.0, false, 0.0),
        ],
        &[2.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 44.0, 52.0, 60.0],
        &[0.1, -0.05],
    )
}

/// An `if` whose arms meet at the switch is served by the exact derivatives although every η
/// moves the switching time, so it must be the same function as its `max` twin. They agree to
/// roundoff, not bit for bit: `max` raises `1.0` to `G` where the `if` writes `1.0`.
///
/// The reference for the value is the smoothed kink, not FD: central FD of the objective across
/// this kink reads -0.187 for `if` and -0.117 for the `abs` spelling of one function at `x[0]`,
/// more than the gap below, while the smoothed adjoint moves by under 1e-3 of `1 + |g|` from
/// `eps = 3e-3` to `1e-4`, and matched its own FD to 3e-7 at `eps = 0.1`. Against it the gap is
/// 9.5e-3 of `1 + |g|`, because a kink is not exact to second order and FOCEI's gradient reads
/// second derivatives. That inexactness is shared with `max` and `abs`, and is measured here
/// so it cannot widen unseen.
///
/// Regression caught: the twins parting (a change to one arm's differentiation), or the gap
/// widening beyond the measured value.
#[test]
fn ode_continuous_moving_switch_matches_its_max_twin_and_the_smoothed_limit() {
    let model = tight(parse_model_string(ODE_MOVING_CONTINUOUS_SWITCH).unwrap());
    let s = moving_switch_subject(&model);
    let theta = &model.default_params.theta;
    let eta = [0.1, -0.05];
    let preds = crate::pk::compute_predictions_with_tv(&model, &s, theta, &eta);
    // The fixture is live: the arm changes between records, in both directions.
    let c0 = theta[2] * eta[1].exp();
    let below: Vec<bool> = preds.iter().map(|&r| r < c0).collect();
    let flips = below.windows(2).filter(|w| w[0] != w[1]).count();
    assert!(flips >= 2, "r = {preds:?} against C0 = {c0}");

    let got = moving_switch_gradient("(if (r < C0) (C0 / r)^G else 1.0)");
    let twin = moving_switch_gradient("max(C0 / r, 1.0)^G");
    // Four θ, two ω and σ: a shorter gradient would leave the loops below with less to check.
    assert_eq!(got.len(), 7, "{got:?}");
    assert_eq!(twin.len(), 7, "{twin:?}");
    for (a, b) in got.iter().zip(&twin) {
        assert!((a - b).abs() <= 1e-13 * (1.0 + b.abs()), "{got:?} {twin:?}");
    }

    let smooth =
        moving_switch_gradient("(0.5 * (C0 / r + 1.0 + sqrt((C0 / r - 1.0)^2 + 1e-3 * 1e-3)))^G");
    assert_eq!(smooth.len(), 7, "{smooth:?}");
    let mut worst = 0.0f64;
    for (a, b) in got.iter().zip(&smooth) {
        assert!(a.is_finite() && b.is_finite(), "{got:?} {smooth:?}");
        worst = worst.max((a - b).abs() / (1.0 + b.abs()));
    }
    assert!(worst < 2e-2, "worst relative difference {worst:.2e}");
}

/// The gradient on the route the exemption keeps, against reconverged FD. A kink defeats FD, so
/// this fixture's arms also meet in value, slope and curvature: the `if` is `C²` across a
/// switch every η moves, and FD is an oracle for it. The test above cannot see a wrong gradient
/// common to its three spellings; this one can.
///
/// Regression caught: the served derivatives parting from the function.
#[test]
fn ode_smooth_moving_switch_matches_reconverged_fd() {
    let src = ODE_MOVING_CONTINUOUS_SWITCH.replace(
        "(if (r < C0) (C0 / r)^G else 1.0)",
        "(if (r < C0) 1.0 + 20.0 * G * ((C0 - r) / C0)^3 else 1.0)",
    );
    let model = tight(parse_model_string(&src).unwrap());
    let s = moving_switch_subject(&model);
    assert!(in_scope(&model, &s), "the `if` must be served");
    let theta = &model.default_params.theta;
    let eta = [0.1, -0.05];
    let preds = crate::pk::compute_predictions_with_tv(&model, &s, theta, &eta);
    let c0 = theta[2] * eta[1].exp();
    let below: Vec<bool> = preds.iter().map(|&r| r < c0).collect();
    let flips = below.windows(2).filter(|w| w[0] != w[1]).count();
    assert!(flips >= 2, "r = {preds:?} against C0 = {c0}");
    let worst = fd_oracle(&model, &s, true);
    // Measured worst 1.3e-7 on macOS, at the Richardson FD noise floor; about 15× headroom.
    assert!(worst < 2e-6, "worst relative difference {worst:.2e}");
}

/// Elimination that doubles above a concentration threshold every η and θ moves: a switch
/// whose arms do **not** meet, on the observed state itself (#2649).
const ODE_MOVING_JUMP_SWITCH: &str = r#"
[parameters]
  theta TVCL(2.0, 0.1, 20.0)
  theta TVV(10.0, 1.0, 100.0)
  theta TVCT(3.0, 0.001, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_CT ~ 0.05
  sigma ADD ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  CT = TVCT * exp(ETA_CT)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(if (central / V > CT) 2.0 * CL else CL) / V * central
[error_model]
  DV ~ additive(ADD)
"#;

/// [`ODE_MOVING_JUMP_SWITCH`] with the threshold's typical value `ct`, and its subject: two
/// doses, the second landing on residual drug below the threshold, so the switch is crossed
/// on the way down twice, once from a state each dose built. No record sits where a crossing
/// can fall (about 2–5 h after each dose): a prediction read exactly at the crossing is
/// continuous but not differentiable in the parameters (its sensitivity jumps there), and an
/// earlier fixture with a record at 4 h had its EBE settle on that kink, crossing at
/// 4.0000000001, where central FD averages two one-sided slopes.
fn moving_jump_fixture(ct: f64) -> (CompiledModel, Subject) {
    let src = ODE_MOVING_JUMP_SWITCH.replace("TVCT(3.0,", &format!("TVCT({ct},"));
    // At reltol 1e-12: the oracle's gap measured 5.9e-4, 1.9e-5 and 3.2e-7 at reltol 1e-8,
    // 1e-10 and 1e-12, the integrator's error, falling with its tolerance to the FD floor.
    let model = with_tol(parse_model_string(&src).unwrap(), 1e-12, 1e-14);
    let s = subject(
        &model,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &[
            0.5, 1.0, 1.5, 6.0, 8.0, 10.0, 12.0, 12.5, 13.0, 19.0, 22.0, 24.0,
        ],
        &[0.1, -0.05],
    );
    (model, s)
}

/// How many times the concentration crosses the threshold between records at the moved
/// parameters: the fixture's straddle, read off the production predictor.
fn threshold_crossings(model: &CompiledModel, s: &Subject) -> usize {
    let p = moved_params(model);
    let eta = [0.1, -0.05];
    let preds = crate::pk::compute_predictions_with_tv(model, s, &p.theta, &eta);
    let ct = p.theta[2] * eta[1].exp();
    let above: Vec<bool> = preds.iter().map(|&a| a / p.theta[1] > ct).collect();
    above.windows(2).filter(|w| w[0] != w[1]).count()
}

/// The exact gradient across a moving switch whose arms jump, against reconverged FD at tight
/// solver tolerances: the integrator ends a step on the crossing and the recorded jump carries
/// every lane across it (#2649). Before, the model was routed to finite differences, and the
/// branch-wise derivative it would otherwise have had misses the whole saltation `Δf·dt*/dp`.
///
/// A differential pair straddles the switch: at `TVCT = 3` the concentration crosses the
/// threshold within the horizon (down, up at the second dose, down again), at `TVCT = 0.001` it
/// never does and no jump fires. Both must match FD; the straddle is asserted, so neither half
/// can silently become the other.
///
/// Regression caught: the jump dropped or mis-signed (the crossing half parts from FD by the
/// saltation), or event location lost, while the control stays green.
#[test]
fn ode_jumping_moving_switch_on_an_observed_state_matches_reconverged_fd() {
    let (model, s) = moving_jump_fixture(3.0);
    let program = model
        .ode_spec
        .as_ref()
        .unwrap()
        .rhs_program
        .as_ref()
        .unwrap();
    assert_eq!(program.switch_count(), 1);
    assert!(
        threshold_crossings(&model, &s) >= 3,
        "the switch is not live"
    );
    assert!(in_scope(&model, &s), "the exact gradient must serve it");
    let worst = fd_oracle(&model, &s, true);
    eprintln!("jumping switch, crossing: worst {worst:.3e}");
    // Measured 1.2e-7 on Linux x86_64, at the Richardson floor; about 8× headroom. Mutations:
    // no jump 2.9e2, no event location 1.15, one chord step 3.1e-3, the forced arm dropped from
    // the locating steps 1.7e-4.
    assert!(worst < 1e-6, "worst relative difference {worst:.2e}");

    let (control, cs) = moving_jump_fixture(0.001);
    assert_eq!(threshold_crossings(&control, &cs), 0, "the control crosses");
    let worst = fd_oracle(&control, &cs, true);
    eprintln!("jumping switch, control: worst {worst:.3e}");
    // Measured 1.5e-8.
    assert!(worst < 1e-6, "worst relative difference {worst:.2e}");
}

/// The second order across the jump: the covariance step's exact Hessian of the FOCEI objective
/// against reconverged FD of the exact gradient. The jump's second-order lanes are not the
/// first-order saltation differentiated once more with a frozen crossing time; they carry the
/// crossing time's curvature and both flows' (`jump_tr`), which this reads.
///
/// Regression caught: a jump exact to first order only (one Newton step for the crossing time,
/// or a linear flow), which leaves the gradient test above green.
#[test]
fn ode_jumping_moving_switch_hessian_matches_fd_of_the_gradient() {
    let (model, s) = moving_jump_fixture(3.0);
    // Measured 6.4e-8 (FOCEI) on Linux x86_64; about 15× headroom. Mutations: one chord step
    // 1.1e-4, two 1.7e-4, the forced arm dropped from the locating steps 2.4e-5.
    let (vs_fd, _) = hessian_gaps_for(&model, &s, Objective::Focei);
    assert!(
        vs_fd < 1e-6,
        "FOCEI Hessian vs FD of the gradient {vs_fd:.2e}"
    );
    // Laplace's exact anchor reads `∂²f/∂η²` in the objective, so its Hessian reads one order
    // more of the jump than FOCEI's: measured 6.4e-8 with four chord steps, 6.6e-7 with three
    // (which FOCEI's above does not see), so the bound pins the fourth.
    let laplace = Objective::Agq {
        n_agq: 1,
        exact: true,
    };
    let (vs_fd, _) = hessian_gaps_for(&model, &s, laplace);
    eprintln!("jumping switch, Laplace Hessian {vs_fd:.3e}");
    assert!(
        vs_fd < 3e-7,
        "Laplace Hessian vs FD of the gradient {vs_fd:.2e}"
    );
}

/// #2124: the static dual walk (`integrate_g`) reads a record one ULP before a co-timed
/// dose — where the reader puts an observation listed before the dose — pre-dose, as the
/// production predictor does. Only `t = 0` is the live cell: there the record and the dose
/// share the subject's first break, where the dose is applied before anything is read.
/// `t = 4` is the control — its ULP is under the 1e-15 dedup too, but the record at
/// `4.next_down()` is neither the first nor the last break, so it is read at the end of the
/// previous segment before the dose, and passes without the fix as well. `init(...)` puts
/// drug in the state before the first dose, so both sides of each dose are non-zero.
///
/// Regression caught: the dual walk recording the post-dose state over the break's whole
/// band (either `record_at` or `record_at_break` ignoring `pre`) — the values part by a
/// dose from production, and the adjoint gradient is then the derivative of a different
/// objective from the one FD differences.
#[test]
fn ode_static_walk_reads_a_ulp_earlier_record_pre_dose_like_production() {
    let model = tight(parse_model_string(ODE_INIT).unwrap());
    let times = [0.0_f64.next_down(), 0.0, 2.0, 4.0_f64.next_down(), 4.0, 6.0];
    let s = subject(
        &model,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(4.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &times,
        &[0.1, -0.05],
    );
    assert!(
        in_scope(&model, &s),
        "the static walk must serve this subject"
    );
    let theta = &model.default_params.theta;
    let eta = [0.1, -0.05];
    let preds = crate::pk::compute_predictions_with_tv(&model, &s, theta, &eta);
    // The fixture is live: each pre-dose row sits a whole dose (100 / V = 10) below its
    // post-dose twin on production, so a walk reading post-dose there cannot agree.
    for (pre, post) in [(0, 1), (3, 4)] {
        assert!(
            preds[post] - preds[pre] > 9.0,
            "rows {pre}/{post}: production {} / {}",
            preds[pre],
            preds[post]
        );
    }
    let rows = exact_obs_grads(&model, &s, theta, &eta).expect("in recorder scope");
    for (j, (row, want)) in rows.iter().zip(&preds).enumerate() {
        assert!(
            (row.f - want).abs() <= 1e-9 * want.abs(),
            "row {j} (t = {:e}): dual walk {} vs production {want}",
            times[j],
            row.f
        );
    }
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

/// The implied dose (`doses[1]`) of `s` carries its parent's lag in slot `lag`, **and** that
/// differs from the governing record's (#1909, #2367). The governing value is computed, not
/// assumed: on a twin whose dose 1 is an ordinary dose (no parent), the same snapshot
/// builder reads dose 1's lag at its own inputs — the governing record's, which is what the
/// implied dose read before the copy. So the pair straddles the copy under the old
/// behaviour: dropping the copy fails the first assertion, and a fixture edit that made the
/// two lags equal fails the second.
fn assert_implied_lag_is_the_parents(
    model: &CompiledModel,
    s: &Subject,
    theta: &[f64],
    lag: usize,
    label: &str,
) {
    let mut dose = Vec::new();
    crate::pk::compute_dose_pk_params_into(model, s, theta, &[0.0, 0.0], &mut dose);
    let parent = dose[0].values[lag];
    assert_eq!(
        dose[1].values[lag], parent,
        "{label}: the implied dose takes the parent's lag"
    );
    let mut own = s.clone();
    own.doses[1].implied = None;
    let mut own_dose = Vec::new();
    crate::pk::compute_dose_pk_params_into(model, &own, theta, &[0.0, 0.0], &mut own_dose);
    let governing = own_dose[1].values[lag];
    assert!(
        governing.is_finite() && (governing - parent).abs() > 0.1 * parent.abs(),
        "{label}: the governing record's lag {governing} must differ from the parent's \
         {parent}, or the copy assertion cannot fail"
    );
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
            let theta = &model.default_params.theta;
            assert_implied_lag_is_the_parents(&model, &s, theta, lag, &label);
            let params = moved_params(&model);
            let (gf, gl) = objective_gaps(&model, &s, &params, &[0.1, -0.05]);
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

/// [`addl_lag_src`]'s ODE model with the lag written as the route's own (#2367): the dose
/// into central is absorbed by `first_order(ka=KA, lag=LAGR)`, with no compartment lag.
fn addl_route_lag_src(iov: bool) -> String {
    addl_lag_src(true, iov)
        .replace("LAGTIME =", "LAGR =")
        .replace(
            "ode(states=[depot, central])\n[odes]\n  d/dt(depot) = -KA * depot\n  \
             d/dt(central) = KA * depot - (CL/V) * central",
            "ode(states=[central])\n[odes]\n  \
             d/dt(central) = first_order(ka=KA, lag=LAGR) - (CL/V) * central",
        )
}

#[test]
fn an_addl_implied_dose_route_lag_from_its_parent_matches_reconverged_fd() {
    // (IOV, time-varying X, bound): the ODE rows of the test above, with the lag on the
    // route. Measured vs reconverged FD on macOS arm64: 2.6e-7, 2.7e-7 and 2.1e-7. Each
    // bound is 5× that. If the walks' seed vectors do not copy the route lag while the
    // predictor's snapshots do, the gaps are 52, 1.1 and 2.0.
    let cases = [
        (false, true, 1.3e-6),
        (true, true, 1.4e-6),
        (true, false, 1.1e-6),
    ];
    for (iov, tv, bound) in cases {
        let label = format!("route lag, iov {iov}, tv {tv}");
        let src = addl_route_lag_src(iov);
        let model = with_tol(parse_model_string(&src).unwrap(), 1e-8, 1e-10);
        let ode = model.ode_spec.as_ref().expect("ode model");
        assert!(ode.n_states == 1 && !model.has_lagtime(), "{label}: {src}");
        let lag = ode.input_rate[0].lag_slot.expect("a lagged route");
        let s = addl_implied_lag_subject(&model, tv);
        if !iov {
            // The copy is live: the parent's route lag is half the governing record's.
            let theta = &model.default_params.theta;
            assert_implied_lag_is_the_parents(&model, &s, theta, lag, &label);
            let params = moved_params(&model);
            let (gf, gl) = objective_gaps(&model, &s, &params, &[0.1, -0.05]);
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

/// A closed-form model whose clearance reads `D` through `+ 0 * D`, which is `+0` for
/// any finite `D` and leaves `CL` unchanged bit for bit, value and jets alike. `D` is
/// still an input the sharing key sees, so a subject giving each record its own `D`
/// shares nothing while computing the same numbers. Under `iov` the clearance carries a κ.
fn shared_inputs_src(iov: bool) -> String {
    let (kappa, kappa_term, fit) = if iov {
        (
            "  kappa KAPPA_CL ~ 0.04\n",
            " + KAPPA_CL",
            "[fit_options]\n  iov_column = OCC\n",
        )
    } else {
        ("", "", "")
    };
    format!(
        "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  theta TVV(30.0, 1.0, 500.0)\n  \
         theta TVKA(1.2, 0.05, 20.0)\n  theta TVLAG(0.4, 0.01, 5.0)\n  omega ETA_CL ~ 0.09\n  \
         omega ETA_LAG ~ 0.1\n{kappa}  sigma PROP_ERR ~ 0.1 (sd)\n[individual_parameters]\n  \
         CL = TVCL * sqrt(X) * exp(ETA_CL{kappa_term}) + 0 * D\n  V = TVV\n  KA = TVKA\n  \
         LAGTIME = TVLAG * X * exp(ETA_LAG)\n[structural_model]\n  \
         pk one_cpt_oral(cl=CL, v=V, ka=KA, lagtime=LAGTIME)\n\
         [error_model]\n  DV ~ proportional(PROP_ERR)\n{fit}"
    )
}

/// Doses at 0 (X = 1) and 6 (X = 2), and one at 12 implied by the first and governed by
/// a record with X = 2, so it reads the dose at 6's inputs but its parent's lag (#1909).
/// X returns to 1 after 13, so observations also share with records that are not their
/// neighbours. `distinct(i)` gives record `i` its own `D`, the doses being `0..3` and the
/// observations `10 + j`. Under IOV the records before 12 sit on occasion 1 and the rest on
/// occasion 2, so the X = 2 records straddle a κ.
fn shared_inputs_subject(model: &CompiledModel, distinct: impl Fn(usize) -> bool) -> Subject {
    let mut implied = DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0);
    implied.implied = Some(crate::types::ImpliedDose {
        parent: 0,
        param_time: 13.0,
        gov_time: 13.0,
    });
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(6.0, 100.0, 1, 0.0, false, 0.0),
        implied,
    ];
    let times = [1.0, 2.0, 4.0, 7.0, 8.0, 13.0, 14.0, 18.0, 24.0];
    let xs = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0];
    let cov = |x: f64, d: f64| HashMap::from([("X".to_string(), x), ("D".to_string(), d)]);
    let d = |i: usize| {
        if distinct(i) {
            1.0 + 0.125 * i as f64
        } else {
            1.0
        }
    };
    let mut s = subject_shell(doses, &times);
    s.covariates = cov(1.0, 1.0);
    s.dose_covariates = [1.0, 2.0, 2.0]
        .iter()
        .enumerate()
        .map(|(k, &x)| cov(x, d(k)))
        .collect();
    s.obs_covariates = xs
        .iter()
        .enumerate()
        .map(|(j, &x)| cov(x, d(10 + j)))
        .collect();
    if model.n_kappa > 0 {
        s.occasions = times
            .iter()
            .map(|&t| if t < 12.0 { 1 } else { 2 })
            .collect();
        s.dose_occasions = vec![1, 1, 2];
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
    s
}

/// The closed-form recorder evaluates `$PK` once per distinct record input on its untaped
/// types (`f64`, the inner loop's forward jets) and copies it to the rest. That must change no
/// bit of the predictions, the η-Jacobian or the outer gradient, so the subject whose
/// records share is compared against its twin whose records share nothing.
///
/// Regressions each comparison catches, run as mutations: sharing records of different κ
/// groups (IOV); writing an implied dose's parent lag into an evaluation the dose at 6
/// shares, both when other records read it too and when the two doses are its only
/// readers; copying from the wrong record; and letting the taped recording share (the
/// gradient row).
#[test]
fn records_sharing_an_evaluation_match_their_unshared_twin_bitwise() {
    for iov in [false, true] {
        let model = parse_model_string(&shared_inputs_src(iov)).unwrap();
        assert!(
            model.referenced_covariates.iter().any(|c| c == "D"),
            "iov {iov}: `0 * D` must keep D an input, or the twin shares too"
        );
        let shared = shared_inputs_subject(&model, |_| false);
        let twin = shared_inputs_subject(&model, |_| true);
        // The X = 2 observations each their own, so without IOV the dose at 6 and the
        // implied dose are the only two records reading their evaluation.
        let dose_pair = shared_inputs_subject(&model, |i| (13..16).contains(&i));
        assert!(
            shared.has_tv_covariates(),
            "iov {iov}: per-record evaluation"
        );
        // The straddle: the subject shares, its twin does not, and the dose at 6 shares
        // with the implied dose at 12, so the parent-lag copy lands in a shared evaluation.
        let group = |s: &Subject, occ: u32| {
            (model.n_kappa > 0)
                .then(|| crate::stats::stacked_layout::StackedLayout::sens_of(&model, s))
                .flatten()
                .and_then(|l| l.group_of(occ))
        };
        // The recorder's own record list, so this asserts the sharing production builds.
        let first = |s: &Subject| {
            let records = crate::pk::record_pk_inputs(s);
            crate::pk::first_record_with_same_pk_inputs(
                &model,
                records.iter().map(|&(occ, cov, _)| (group(s, occ), cov)),
            )
            .expect("a parsed model that does not read TIME has a sharing index")
        };
        let (fs, ft) = (first(&shared), first(&twin));
        assert!(
            ft.iter().enumerate().all(|(i, &f)| f == i),
            "iov {iov}: the twin must share nothing: {ft:?}"
        );
        // Doses 0..3, then the observations at 3 + j.
        let want = if iov {
            // The X = 2 records split at 12 by occasion, and so do the X = 1 ones.
            [0, 1, 2, 0, 0, 0, 1, 1, 2, 9, 9, 9]
        } else {
            [0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0]
        };
        assert_eq!(fs, want, "iov {iov}: the subject's sharing");
        let want_pair = if iov {
            [0, 1, 2, 0, 0, 0, 6, 7, 8, 9, 9, 9]
        } else {
            [0, 1, 1, 0, 0, 0, 6, 7, 8, 0, 0, 0]
        };
        assert_eq!(
            first(&dose_pair),
            want_pair,
            "iov {iov}: the dose pair's sharing"
        );

        let theta = &model.default_params.theta;
        let w = subject_width(&model, &shared).unwrap();
        assert_eq!(w, subject_width(&model, &twin).unwrap());
        let eta: Vec<f64> = (0..w).map(|k| 0.1 - 0.07 * k as f64).collect();
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let pred = |s: &Subject| crate::pk::compute_predictions_with_tv(&model, s, theta, &eta);
        let sharing = [("shared", &shared), ("dose pair", &dose_pair)];
        for (name, s) in sharing {
            assert_eq!(
                bits(&pred(s)),
                bits(&pred(&twin)),
                "iov {iov}, {name}: predictions"
            );
        }
        let jac = |s: &Subject| {
            exact_obs_grads(&model, s, theta, &eta)
                .expect("the closed-form recorder serves the subject")
                .iter()
                .flat_map(|r| std::iter::once(r.f).chain(r.df_deta.iter().copied()))
                .collect::<Vec<_>>()
        };
        for (name, s) in sharing {
            assert_eq!(
                bits(&jac(s)),
                bits(&jac(&twin)),
                "iov {iov}, {name}: the jet values and η-Jacobian"
            );
        }
        let p = moved_params(&model);
        let grad = |s: &Subject| {
            subject_adjoint_packed_gradient(&model, s, &p, &pack_params(&p), &eta, true)
                .expect("the adjoint serves the subject")
        };
        for (name, s) in sharing {
            assert_eq!(
                bits(&grad(s)),
                bits(&grad(&twin)),
                "iov {iov}, {name}: the taped outer gradient"
            );
        }
    }
}

/// Inside a `SharingScope` the recorder's second call reads the index its first call
/// kept (#2558), and its output must be the bits an unscoped call gives, on `f64` and on
/// the inner loop's forward jets. The subject's records do share (asserted), so the index a hit
/// hands back decides which evaluation each record reads, and a wrong or stale one moves
/// a prediction. The bits match the twin that shares nothing too, so what is compared is
/// the right answer, not only the unscoped path's. Catches a hit that hands the recorder
/// an index other than the one a build gives, which the f64 materializer's bit test and
/// the parity tests, run outside any scope, cannot see.
#[test]
fn the_recorder_reads_the_same_bits_from_a_kept_sharing_index() {
    use crate::pk::sharing_scope::{builds, SharingScope};
    for iov in [false, true] {
        let model = parse_model_string(&shared_inputs_src(iov)).unwrap();
        let shared = shared_inputs_subject(&model, |_| false);
        let twin = shared_inputs_subject(&model, |_| true);
        let theta = &model.default_params.theta;
        let w = subject_width(&model, &shared).unwrap();
        let eta: Vec<f64> = (0..w).map(|k| 0.1 - 0.07 * k as f64).collect();
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let values = |s: &Subject| {
            let f = record::<f64>(
                &model,
                s,
                None,
                theta,
                &eta,
                &ThetaLeaves::theta(theta),
                &eta,
            )
            .expect("the closed-form recorder serves the subject");
            bits(&f)
        };
        let jets = |s: &Subject| {
            let rows = exact_obs_grads(&model, s, theta, &eta)
                .expect("the closed-form recorder serves the subject");
            let flat: Vec<f64> = rows
                .iter()
                .flat_map(|r| std::iter::once(r.f).chain(r.df_deta.iter().copied()))
                .collect();
            bits(&flat)
        };
        let records = crate::pk::record_pk_inputs(&shared);
        let group = |occ: u32| {
            crate::stats::stacked_layout::StackedLayout::sens_of(&model, &shared)
                .and_then(|l| l.group_of(occ))
        };
        let first = crate::pk::first_record_with_same_pk_inputs(
            &model,
            records.iter().map(|&(occ, cov, _)| (group(occ), cov)),
        )
        .expect("a parsed model that does not read TIME has a sharing index");
        assert!(
            first.iter().enumerate().any(|(i, &f)| f != i),
            "iov {iov}: the records must share, or a hit's index decides nothing: {first:?}"
        );

        let (want_f, want_jets) = (values(&twin), jets(&twin));
        assert_eq!(values(&shared), want_f, "iov {iov}: unscoped f64");
        assert_eq!(jets(&shared), want_jets, "iov {iov}: unscoped jets");
        let _share = SharingScope::enter(&model, &shared);
        let b = builds();
        assert_eq!(
            values(&shared),
            want_f,
            "iov {iov}: f64, the call that builds"
        );
        assert_eq!(
            builds(),
            b + 1,
            "iov {iov}: the recorder builds its index once"
        );
        assert_eq!(values(&shared), want_f, "iov {iov}: f64, a hit");
        assert_eq!(jets(&shared), want_jets, "iov {iov}: jets, a hit");
        assert_eq!(
            builds(),
            b + 1,
            "iov {iov}: every later call is a hit, or this test compares builds"
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

/// A parent drained into two gated accumulators, one per collection interval: the shape
/// whose dormant states #2760 leaves out of the exponential. The observation reads
/// both accumulators, so one that is dormant but holds drug still moves the prediction.
/// `FE` has no random effect, so the outer gradient reads it off the tape (a θ with an η
/// is carried by the η-jets instead, #2706): the gated terms are then taped block outputs,
/// which only the recording's own structural-zero test can return as zeros.
const GATED_ACCUMULATORS_ODE: &str = r#"
[parameters]
  theta TVCL(1.2, 0.01, 50.0)
  theta TVV(15.0, 0.5, 500.0)
  theta TVFE(0.4, 0.01, 1.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP_ERR ~ 0.02
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV * exp(ETA_V)
  FE = TVFE
  G1 = if (IDX == 1) 1 else 0
  G2 = if (IDX == 2) 1 else 0
[structural_model]
  ode(states=[central, u1, u2])
[odes]
  d/dt(central) = -(CL / V) * central
  d/dt(u1) = G1 * FE * (CL / V) * central
  d/dt(u2) = G2 * FE * (CL / V) * central
[scaling]
  y = central / V + 0.01 * (u1 + u2)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// #2760: the adjoint on the exact linear path with gated accumulators matches
/// reconverged FD. Three intervals, the gate `IDX` set at each dose: interval 1 fills `u1`
/// (`u2` dormant and empty), interval 2 fills `u2` (`u1` dormant and holding drug), interval
/// 3 fills neither (both dormant, both holding). The recording must take the exact path,
/// and the dormant states must be left out on the taped recording itself: with the block
/// keeping its structural-zero outputs (`RHS_DEAD_KEPT`, the recording before #2760)
/// fewer states go dormant, since a taped zero is not a structural one. That straddle is
/// what makes the parity a test of the reduced recording rather than of the full one.
#[test]
fn adjoint_with_dormant_gated_accumulators_matches_reconverged_fd() {
    // Measured (Linux x86_64): dropped vs kept 7.0e-17, parity 3.4e-9. Hessian vs reconverged
    // FD 5.46e-10 (Linux aarch64, glibc 2.41; the same on macOS arm64), 10× headroom.
    dormant_accumulators_match_reconverged_fd(GATED_ACCUMULATORS_ODE, 1e-14, 3e-8, 5e-9);
}

/// [`GATED_ACCUMULATORS_ODE`] with a peripheral compartment, and the accumulators between
/// the live states: `[u1, central, u2, periph]`. The three intervals leave `{u2}`, `{u1}`
/// and `{u1, u2}` dormant, so a live state sits on both sides of a dormant one and the
/// reduced block's index `lj` differs from the state index `j` it reads (#2761 review): the
/// taped type's `dot_fold` branch builds its pairs through that map.
const INTERLEAVED_ACCUMULATORS_ODE: &str = r#"
[parameters]
  theta TVCL(1.2, 0.01, 50.0)
  theta TVV(15.0, 0.5, 500.0)
  theta TVFE(0.4, 0.01, 1.0)
  theta TVQ(2.0, 0.01, 50.0)
  theta TVV2(30.0, 0.5, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP_ERR ~ 0.02
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV * exp(ETA_V)
  FE = TVFE
  Q  = TVQ
  V2 = TVV2
  G1 = if (IDX == 1) 1 else 0
  G2 = if (IDX == 2) 1 else 0
[structural_model]
  ode(states=[u1, central, u2, periph])
[odes]
  d/dt(u1) = G1 * FE * (CL / V) * central
  d/dt(central) = -(CL / V) * central - (Q / V) * central + (Q / V2) * periph
  d/dt(u2) = G2 * FE * (CL / V) * central
  d/dt(periph) = (Q / V) * central - (Q / V2) * periph
[scaling]
  y = central / V + 0.01 * (u1 + u2)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

#[test]
fn adjoint_with_interleaved_dormant_accumulators_matches_reconverged_fd() {
    // Measured (Linux x86_64): dropped vs kept 4.2e-16, parity 4.9e-9. Hessian vs reconverged
    // FD 5.12e-11 (Linux aarch64, glibc 2.41; the same on macOS arm64), 10× headroom.
    dormant_accumulators_match_reconverged_fd(INTERLEAVED_ACCUMULATORS_ODE, 5e-14, 5e-8, 5e-10);
}

/// The body of the two tests above: `src` is a gated-accumulator model on an `IDX` gate
/// (dose 1 fills `u1`, dose 2 `u2`, dose 3 neither), dosed into the state named `central`.
/// Asserts the exact path, more dormant states on the recording that drops its structural
/// zeros than on the one that keeps them (`RHS_DEAD_KEPT`, the straddle), the two gradients
/// equal to `gap_tol`, and reconverged-FD parity to `parity_tol`.
fn dormant_accumulators_match_reconverged_fd(
    src: &str,
    gap_tol: f64,
    parity_tol: f64,
    hessian_tol: f64,
) {
    use crate::ode::linear::DORMANT_STATES;
    use crate::ode::solver::SolverStatsScope;
    let model = parse_model_string(src).unwrap();
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
    // The doses go into the state named `central` (1-based), wherever the model puts it.
    let cmt = 1 + model
        .ode_spec
        .as_ref()
        .unwrap()
        .state_names
        .iter()
        .position(|n| n == "central")
        .expect("a `central` state");
    let doses = vec![
        DoseEvent::new(0.0, 100.0, cmt, 0.0, false, 0.0),
        DoseEvent::new(24.0, 100.0, cmt, 0.0, false, 0.0),
        DoseEvent::new(48.0, 100.0, cmt, 0.0, false, 0.0),
    ];
    let times = [1.0, 6.0, 23.0, 25.0, 30.0, 47.0, 49.0, 60.0, 71.0];
    let idx_at = |t: f64| {
        if t < 24.0 {
            1.0
        } else if t < 48.0 {
            2.0
        } else {
            3.0
        }
    };
    let mut s = subject_shell(doses.clone(), &times);
    s.covariates.insert("IDX".into(), 1.0);
    s.obs_covariates = times
        .iter()
        .map(|&t| HashMap::from([("IDX".to_string(), idx_at(t))]))
        .collect();
    s.dose_covariates = doses
        .iter()
        .map(|d| HashMap::from([("IDX".to_string(), idx_at(d.time))]))
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
        .map(|(j, p)| p * (1.0 + 0.12 * (1.7 * j as f64).sin()))
        .collect();
    assert!(s.has_tv_covariates());
    assert!(in_scope(&model, &s), "the subject must be in adjoint scope");
    // One adjoint gradient, with the block dropping its structural zeros or keeping them:
    // the gradient, the dormant states it left out, and the solver's step/decline counts.
    let gradient = |keep: bool| {
        let params = moved_params(&model);
        let x = pack_params(&params);
        let eta = ebe_for(&model, &s, &params, 1e-10, true);
        // A fresh recording: the reduction is decided where the plan is recorded.
        crate::parser::model_parser::rhs_regcode::kernel_jit::reset();
        crate::sens::linsolve::RECORDED_ZEROS_KEPT.with(|c| c.set(keep));
        DORMANT_STATES.with(|c| c.set(0));
        let scope = SolverStatsScope::enter();
        let g = subject_adjoint_packed_gradient(&model, &s, &params, &x, eta.as_slice(), true)
            .expect("in scope");
        let st = scope.collected();
        crate::sens::linsolve::RECORDED_ZEROS_KEPT.with(|c| c.set(false));
        let dormant = DORMANT_STATES.with(|c| c.get());
        (
            g,
            dormant,
            (st.attempted_steps, st.linear_declined_segments),
        )
    };
    let (g, dormant, steps) = gradient(false);
    let (g_kept, dormant_kept, steps_kept) = gradient(true);
    assert_eq!(steps, (0, 0), "the exact path stepped or declined");
    assert_eq!(steps_kept, (0, 0), "the kept recording stepped or declined");
    assert!(
        dormant > dormant_kept,
        "the taped recording left out no more dormant states ({dormant} vs {dormant_kept} kept)"
    );
    // The same gradient whether the taped zeros are dropped or kept, to rounding: the tape
    // records fewer nodes, which moves what its CSE table merges and so the order a sweep
    // sums in. The callers' bounds are their measurements with 100× headroom.
    let mut gap = 0.0f64;
    for (a, b) in g.iter().zip(&g_kept) {
        assert!(a.is_finite() && b.is_finite(), "{a} vs {b}");
        gap = gap.max((a - b).abs() / (1.0 + b.abs()));
    }
    eprintln!("dormant {dormant} (kept {dormant_kept}); dropped vs kept gradient: {gap:.2e}");
    assert!(gap <= gap_tol, "dropped vs kept gradient {gap:.2e}");
    let worst = parity(&model, &s);
    eprintln!("gated accumulators vs reconverged FD: worst {worst:.2e}");
    // Reconverged FD (Richardson, EBE re-solved per point); the callers' bounds are their
    // measurements with 10× headroom.
    assert!(worst < parity_tol, "worst relative difference {worst:.2e}");
    // The covariance step's R-matrix on the same subject (#2761 review round 1): its nested
    // duals (`sens::rtape`) run the same reduced propagation, on a recording of their own.
    // First that the reduction ran there, counted on the Hessian's thread over those types
    // alone; then the Hessian against Richardson differences of the exact gradient with the
    // EBE re-solved at every point.
    let params = moved_params(&model);
    let x = pack_params(&params);
    let eta = ebe_for(&model, &s, &params, 1e-12, true);
    let dormant_cov = std::thread::scope(|scope| {
        std::thread::Builder::new()
            .stack_size(crate::FIT_RAYON_STACK_SIZE)
            .spawn_scoped(scope, || {
                use crate::ode::linear::DORMANT_STATES as DORMANT_STATES_COV;
                crate::parser::model_parser::rhs_regcode::kernel_jit::reset();
                DORMANT_STATES_COV.with(|c| c.set(0));
                subject_adjoint_cov_hessian(
                    &model,
                    &s,
                    &params,
                    &x,
                    eta.as_slice(),
                    Objective::Focei,
                )
                .expect("in scope");
                DORMANT_STATES_COV.with(|c| c.get())
            })
            .expect("spawn")
            .join()
            .expect("join")
    });
    assert!(
        dormant_cov > 0,
        "the covariance recording left out no dormant state"
    );
    let (hess, _) = hessian_gaps_for(&model, &s, Objective::Focei);
    eprintln!(
        "dormant on the covariance types {dormant_cov}; Hessian vs reconverged FD {hess:.2e}"
    );
    assert!(hess < hessian_tol, "Hessian vs reconverged FD {hess:.2e}");
}

/// A chain `depot → central → effect` with η-free rate constants: `A` is lower-triangular,
/// so every compartment is acyclic, and its rates are distinct, so the spectral path takes it.
const TRIANGULAR_CHAIN_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVKA(1.0, 0.05, 20.0)
  theta TVKE0(0.3, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.02
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV * exp(ETA_V)
  KA  = TVKA
  KE0 = TVKE0
[structural_model]
  ode(states=[depot, central, effect])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - CL / V * central
  d/dt(effect)  =  KE0 * (central / V - effect)
[scaling]
  y = central / V + effect
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// #2761 review row 5: the outer gradient's taped recording builds `A`'s coupling graph
/// exactly as the `f64` objective does. Before #2760 the recorded block kept `A`'s
/// structural zeros as tape nodes, so on the tape a triangular `A` was one cyclic block
/// (eigensolver eigenvalues, no masks) while the objective took the acyclic form. That
/// moved `emax_pkpd`'s start gradient by 7.9e-13 relative and its optimizer path by 8
/// iterations, a gap far inside every FD-parity bound, so the construction itself is
/// what is asserted.
///
/// Dies on: structural-zero outputs kept on the tape in `vtape::record_rhs` (the
/// `RHS_DEAD_KEPT` gate forced on); `DualV` reading a taped zero as an edge.
#[test]
fn taped_gradient_builds_a_triangular_generators_graph_like_the_objective() {
    taped_graphs_match_the_objective(TRIANGULAR_CHAIN_ODE);
}

/// The same on a generator that is genuinely cyclic (`central ⇄ peripheral`, fed by an
/// acyclic `depot`), where "some compartment on a cycle" reads `true` with the zeros
/// dropped or kept: only the reachability tells the two recordings apart (#2953 review
/// row 2).
#[test]
fn taped_gradient_builds_a_cyclic_generators_graph_like_the_objective() {
    taped_graphs_match_the_objective(TWOCPT_DEPOT_LINEAR_ODE);
}

/// The body of the two tests above: `src` is a linear model dosed into state 1. Records
/// every `spectral` call's reachability (`SPECTRAL_GRAPHS`) for the `f64` predictions and
/// for one adjoint gradient at the same θ and η, and asserts that every taped graph is one
/// the objective built. Straddle: with the block keeping its zeros (`RHS_DEAD_KEPT`, the
/// recording before #2760) some taped graph is not, so a fixture whose graph the kept
/// zeros cannot change, or that never reaches the taped spectral call, cannot pass.
fn taped_graphs_match_the_objective(src: &str) {
    use crate::ode::linear::SPECTRAL_GRAPHS;
    let model = parse_model_string(src).unwrap();
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
    // Two doses: two segments, so two `spectral` calls per recording (the graph comes from
    // `A` alone, not from the state).
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let times = [1.0, 3.0, 8.0, 23.0, 25.0, 30.0, 47.0];
    let mut s = subject_shell(doses, &times);
    let preds = crate::pk::compute_predictions_with_tv(
        &model,
        &s,
        &model.default_params.theta,
        &vec![0.05; model.n_eta],
    );
    s.observations = preds
        .iter()
        .enumerate()
        .map(|(j, p)| p * (1.0 + 0.12 * (1.7 * j as f64).sin()))
        .collect();
    assert!(in_scope(&model, &s), "the subject must be in adjoint scope");
    let params = moved_params(&model);
    let x = pack_params(&params);
    let eta = ebe_for(&model, &s, &params, 1e-10, true);
    // The graphs one closure call built, split into (taped, `f64`).
    let take = || {
        let all = SPECTRAL_GRAPHS.with(|g| std::mem::take(&mut *g.borrow_mut()));
        let (t, f): (Vec<_>, Vec<_>) = all.into_iter().partition(|(taped, _)| *taped);
        let strip = |v: Vec<(bool, [u8; 6])>| v.into_iter().map(|(_, d)| d).collect::<Vec<_>>();
        (strip(t), strip(f))
    };
    SPECTRAL_GRAPHS.with(|g| g.borrow_mut().clear());
    crate::pk::compute_predictions_with_tv(&model, &s, &params.theta, eta.as_slice());
    let (_, objective) = take();
    let taped = |keep: bool| {
        // A fresh recording: a cached plan would not run the recorder again.
        crate::parser::model_parser::rhs_regcode::kernel_jit::reset();
        crate::sens::linsolve::RECORDED_ZEROS_KEPT.with(|c| c.set(keep));
        let g = subject_adjoint_packed_gradient(&model, &s, &params, &x, eta.as_slice(), true);
        crate::sens::linsolve::RECORDED_ZEROS_KEPT.with(|c| c.set(false));
        let g = g.expect("in scope");
        assert!(g.iter().all(|v| v.is_finite()), "{g:?}");
        take().0
    };
    let dropped = taped(false);
    let kept = taped(true);
    let foreign = |v: &[[u8; 6]]| v.iter().filter(|d| !objective.contains(d)).count();
    eprintln!(
        "objective graphs {objective:?}; taped {dropped:?} ({} foreign); kept {kept:?} ({} foreign)",
        foreign(&dropped),
        foreign(&kept)
    );
    assert!(
        !objective.is_empty(),
        "the f64 objective never reached the spectral path"
    );
    assert!(
        !dropped.is_empty(),
        "the taped type never reached the spectral path"
    );
    assert_eq!(
        foreign(&dropped),
        0,
        "the taped recording built a graph the objective did not: {dropped:?} vs {objective:?}"
    );
    assert!(
        foreign(&kept) > 0,
        "the straddle: keeping the zeros on the tape must change the graph: {kept:?} vs \
         {objective:?}"
    );
}

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
        id: "1".to_string(),
        doses,
        obs_times: times.to_vec(),
        observations: vec![0.0; n],
        obs_cmts: vec![1; n],
        cens: vec![0; n],
        occasions: vec![1; n],
        ..Default::default()
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

// ---- IIV on the residual with a dense `R` (#2656) ---------------------------------------

/// The #2656 anchor twin named `twin`, with every `FIX` dropped so each coordinate is live,
/// and one of its subjects, its residual blocks assigned. Real cross-row `R`: `pair` is a
/// `block_sigma` across a total/unbound pair, `triple` a `correlation` over three rows, and
/// the data carry a clear per-subject residual scale.
fn ruv_dense_twin(
    twin: &str,
    data: &str,
    edit: impl Fn(String) -> String,
) -> (CompiledModel, Subject) {
    let root = env!("CARGO_MANIFEST_DIR");
    let dir = format!("{root}/nonmem_anchor/iiv_ruv_dense");
    let src = std::fs::read_to_string(format!("{dir}/{twin}.ferx")).unwrap();
    let parsed =
        crate::parser::model_parser::parse_full_model(&edit(src.replace(" FIX", ""))).unwrap();
    let (pop, _) = crate::api::read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        &format!("{dir}/{data}.csv"),
        None,
        None,
        None,
        &parsed.column_map,
    )
    .unwrap();
    let mut s = pop.subjects[1].clone();
    s.obs_rgroup = crate::stats::residual_error::residual_groups(&parsed.model, &s);
    assert!(parsed.model.residual_error_eta.is_some(), "{twin}");
    assert!(
        crate::stats::residual_error::has_cross_observation_residual(
            &parsed.model.error_spec,
            &s,
            &parsed.model.default_params.sigma.values,
            parsed.model.residual_cov_at(&parsed.model.default_params),
        ),
        "{twin}: rows must be coupled"
    );
    (parsed.model, s)
}

/// Every dense shape the anchor pins: `block_sigma` across rows, and each serial kind.
fn ruv_dense_fixtures() -> Vec<(&'static str, CompiledModel, Subject)> {
    let mut out = Vec::new();
    let (m, s) = ruv_dense_twin("pair", "pair", |t| t);
    out.push(("block_sigma", m, s));
    for kind in ["cs", "ar1", "car1"] {
        let line = format!("correlation {kind}(RHO)");
        let (m, s) = ruv_dense_twin("triple", "triple", |t| {
            t.replace("correlation cs(RHO)", &line)
        });
        out.push((kind, m, s));
    }
    out
}

/// `F` and `l` of the kernel against production's dense-`R` marginal and EBE objective, at a
/// point where `η_r` is far from 0. Kills: either side dropping `exp(2η_r)`.
#[test]
fn dense_r_with_iiv_on_ruv_kernel_objective_matches_production() {
    for (name, m, s) in ruv_dense_fixtures() {
        let g = objective_gaps(&m, &s, &m.default_params, &[0.08, -0.06, 0.45]);
        eprintln!("{name}: {g:?}");
        assert!(g.0 < 1e-10 && g.1 < 1e-10, "{name}: {g:?}");
    }
}

/// The exact outer gradient under `iiv_on_ruv` × a dense `R` (`Ψ = F − wᵀ∇_η l`, every θ, Ω,
/// σ and ρ coordinate) against the reconverged FD oracle, FOCEI.
#[test]
fn dense_r_with_iiv_on_ruv_gradient_matches_reconverged_fd() {
    for (name, m, s) in ruv_dense_fixtures() {
        assert!(in_scope(&m, &s), "{name} must be served");
        let worst = parity(&m, &s);
        eprintln!("{name}: FOCEI gradient {worst:.2e}");
        // Measured worst 2.1e-9 (car1, macOS arm64); ~10×.
        assert!(worst < 2e-8, "{name}: {worst:.2e}");
    }
}

/// The same under Laplace and AGQ (exact and Gauss–Newton anchors), and the FOCEI R-matrix.
#[test]
fn dense_r_with_iiv_on_ruv_agq_and_hessian_match_fd() {
    for (name, m, s) in ruv_dense_fixtures() {
        for (n_agq, exact) in [(1, true), (3, false)] {
            let worst = fd_oracle_agq(&m, &s, n_agq, exact);
            eprintln!("{name}: AGQ n={n_agq} exact={exact} gradient {worst:.2e}");
            // Measured worst 4.3e-11 (cs, macOS arm64); ~10×.
            assert!(worst < 5e-10, "{name} n={n_agq}: {worst:.2e}");
        }
        let (h, _) = hessian_gaps(&m, &s, true);
        eprintln!("{name}: FOCEI hessian {h:.2e}");
        // Measured worst 8.9e-11 (block_sigma, macOS arm64). The reference's own floor on
        // the other dense fixtures reaches 1e-7 (`dense_residual_agq_gradient_and_hessians_match_fd`).
        assert!(h < 3e-7, "{name}: hessian {h:.2e}");
    }
}

/// With a κ the marginal runs over the stacked `[η, κ]` system and the EBE search takes the
/// kernel's gradient (`iov_inner_residual_correlation`), so `η_r` is one stacked coordinate
/// among several, and it must still be the one `R` reads.
#[test]
fn dense_r_with_iiv_on_ruv_and_kappa_matches_reconverged_fd() {
    let src = ODE_IOV_BLOCK_SIGMA
        .replace(
            "  block_sigma (PROP_TOT, PROP_UNB) = [0.01, 0.006, 0.0225]\n",
            "  sigma PROP_TOT ~ 0.1 (sd)\n",
        )
        .replace(
            "  omega ETA_V ~ 0.06\n",
            "  omega ETA_V ~ 0.06\n  omega ETA_RUV ~ 0.2\n",
        )
        .replace(
            "  theta FU(0.3, 0.01, 1.0)\n",
            "  theta FU(0.3, 0.01, 1.0)\n  theta RHO(0.45, -0.4, 0.95)\n",
        )
        .replace(
            "  CMT=1: DV ~ proportional(PROP_TOT)\n  CMT=2: DV ~ proportional(PROP_UNB)\n",
            "  DV ~ proportional(PROP_TOT)\n  correlation cs(RHO) by ARM\n  iiv_on_ruv = ETA_RUV\n",
        );
    let model = tight(parse_model_string(&src).unwrap());
    assert!(model.n_kappa == 1 && model.residual_error_eta == Some(2));
    let mut s = iov_two_endpoint_subject(&model);
    s.obs_rgroup = (0..s.obs_times.len()).map(|j| 1 + (j / 2) as u32).collect();
    assert!(crate::estimation::inner_optimizer::iov_inner_residual_correlation(&model, &s));
    let worst = iov_parity(&model, &s, true);
    eprintln!("κ × cs × iiv_on_ruv: {worst:.2e}");
    // Measured 2.5e-6 (macOS arm64); ~8×.
    assert!(worst < 2e-5, "{worst:.2e}");
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

// ---- FREM with κ on an ODE (#2786) ------------------------------------------------------

/// `ODE_IOV` with a FREM weight pseudo-covariate beside the two BSV η.
fn frem_iov_ode() -> CompiledModel {
    let src = ODE_IOV
        .replace(
            "  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.06\n",
            "  theta TV_WT(72.0, 1.0, 500.0)\n  block_omega (ETA_CL, ETA_V, ETA_WT_FREM) = [\n    0.09,\n    0.0, 0.06,\n    0.0, 0.0, 111.56\n  ]\n",
        )
        .replace(
            "  sigma PROP_ERR ~ 0.1 (sd)\n",
            "  sigma PROP_ERR ~ 0.1 (sd)\n  sigma EPSCOV ~ 0.30\n",
        )
        .replace(
            "[fit_options]\n  iov_column = OCC\n",
            "[fit_options]\n  iov_column = OCC\n  frem_predictions = TV_WT/ETA_WT_FREM:100\n  frem_sigma = EPSCOV\n",
        );
    let m = tight(parse_model_string(&src).unwrap());
    assert!(m.ode_spec.is_some() && m.n_kappa == 2 && m.frem_config.is_some());
    m
}

/// A model with FREM and κ on an ODE is inside the recorders' scope, and each subject is
/// served exactly. A subject without FREM rows is an ordinary IOV subject (reconverged-FD
/// parity, FOCEI and FOCE). A subject with FREM rows is the one the IOV marginal refuses:
/// its objective is the constant `IOV_FREM_REFUSAL_NLL`, so the exact gradient and Hessian
/// are zero — checked against the production objective at the reconverged points, which
/// is that constant at every one of them (the premise of the zero).
#[test]
fn frem_with_kappa_on_an_ode_is_served_exactly() {
    use crate::stats::likelihood::IOV_FREM_REFUSAL_NLL;
    let model = frem_iov_ode();
    assert!(recorder_in_scope(&model) && outer_model_in_scope(&model));
    // `iov_subject` draws its data at a two-η point; this model has the FREM η as a third.
    let mut plain = iov_subject(&model);
    plain.fremtype = vec![0; plain.observations.len()];
    let kappas = vec![vec![0.1, -0.05], vec![-0.08, 0.04], vec![0.05, 0.06]];
    let preds = crate::pk::predict_iov(
        &model,
        &plain,
        &model.default_params.theta,
        &[0.1, -0.05, 0.0],
        &kappas,
    );
    plain.observations = preds
        .iter()
        .enumerate()
        .map(|(j, p)| p * (1.0 + 0.12 * (1.7 * j as f64).sin()))
        .collect();
    for interaction in [true, false] {
        let worst = iov_parity(&model, &plain, interaction);
        eprintln!("FREM + κ, no FREM rows, interaction={interaction}: {worst:.2e}");
        // Measured 4.8e-9 (FOCEI) / 3.4e-7 (FOCE); ~30x headroom on the worse.
        assert!(worst < 1e-5, "interaction={interaction}: {worst:.2e}");
    }
    // The same subject with its last observation a FREM weight row.
    let mut refused = plain.clone();
    let last = refused.observations.len() - 1;
    refused.fremtype[last] = 100;
    refused.observations[last] = 80.0;
    assert!(crate::stats::likelihood::iov_refuses_frem(&model, &refused));
    let params = moved_params(&model);
    let x = pack_params(&params);
    for interaction in [true, false] {
        let eta = ebe_for(&model, &refused, &params, 1e-10, interaction);
        let g = subject_adjoint_packed_gradient(
            &model,
            &refused,
            &params,
            &x,
            eta.as_slice(),
            interaction,
        )
        .expect("served, not declined to finite differences");
        assert_eq!(g, vec![0.0; x.len()]);
        // The oracle: the production objective at reconverged points is the constant.
        for k in 0..x.len() {
            let mut xp = x.clone();
            xp[k] += 1e-3;
            assert_eq!(
                reconverged_f_for(&model, &refused, &params, &xp, interaction),
                IOV_FREM_REFUSAL_NLL,
                "x[{k}]"
            );
        }
        let h = crate::sens::ode_provider::adjoint::subject_adjoint_cov_hessian(
            &model,
            &refused,
            &params,
            &x,
            eta.as_slice(),
            if interaction {
                Objective::Focei
            } else {
                Objective::Foce
            },
        )
        .expect("served");
        assert!(h.nrows() == x.len() && h.iter().all(|v| *v == 0.0));
        // Control: the unrefused subject's gradient is live.
        let g0 = subject_adjoint_packed_gradient(
            &model,
            &plain,
            &params,
            &x,
            eta.as_slice(),
            interaction,
        );
        assert!(g0.is_some_and(|g| g.iter().any(|v| v.abs() > 1e-2)));
    }
}

/// The route the inner-solve report gives an IOV + FREM subject is what the search runs. The
/// IOV marginal refuses a FREM subject's rows, so its search takes finite differences
/// (`analytic_iov_inner_supported`); the report (`iov_inner_subject_route`) must say so rather
/// than call it analytic. The control, the same subject without FREM rows, is analytic on
/// both. Mutation: drop `!iov_refuses_frem` from `iov_inner_rule` -> the refused subject's
/// route and search both read analytic and the `is_none` assertion dies; read the route
/// without the shared rule -> the two disagree on the refused subject.
#[test]
fn an_iov_frem_subject_is_reported_the_route_its_search_runs() {
    use crate::estimation::inner_optimizer::{
        analytic_iov_inner_supported, iov_inner_subject_route,
    };
    let model = frem_iov_ode();
    let params = &model.default_params;
    let mut plain = iov_subject(&model);
    plain.fremtype = vec![0; plain.observations.len()];
    let mut refused = plain.clone();
    let last = refused.observations.len() - 1;
    refused.fremtype[last] = 100;
    assert!(crate::stats::likelihood::iov_refuses_frem(&model, &refused));
    assert!(!crate::stats::likelihood::iov_refuses_frem(&model, &plain));
    for (name, s, analytic) in [("plain", &plain, true), ("refused", &refused, false)] {
        assert_eq!(
            analytic_iov_inner_supported(&model, s, params),
            analytic,
            "{name}: the search"
        );
        assert_eq!(
            iov_inner_subject_route(&model, s, &params.theta).is_some(),
            analytic,
            "{name}: the route report"
        );
    }
}

/// The FD reason an IOV + FREM subject is reported under is the FREM refusal, and the
/// non-Gaussian cause is named for what it covers. Discriminator and negatives: the refused
/// subject reads the FREM reason and neither the generic scope reason nor the non-Gaussian
/// one. Mutation: drop the FREM clause from `iov_fd_reason` -> the refused subject reads
/// "subject outside IOV analytic scope" and the equality dies.
#[test]
fn an_iov_frem_subject_is_reported_its_true_fd_reason() {
    use crate::estimation::inner_optimizer::iov_fd_reason;
    let model = frem_iov_ode();
    let (_, refused) = frem_iov_pair(&model);
    let reason = iov_fd_reason(&model, &refused);
    assert_eq!(reason, "FREM rows (the IOV marginal refuses them)");
    assert!(
        !reason.contains("outside IOV analytic scope") && !reason.contains("non-Gaussian"),
        "{reason}"
    );
}

/// The `frem_iov_ode` subject pair: an ordinary IOV subject and the same one with its last
/// observation a FREM weight row (the one the IOV marginal refuses).
fn frem_iov_pair(model: &CompiledModel) -> (Subject, Subject) {
    let mut plain = iov_subject(model);
    plain.fremtype = vec![0; plain.observations.len()];
    let mut refused = plain.clone();
    let last = refused.observations.len() - 1;
    refused.fremtype[last] = 100;
    refused.observations[last] = 80.0;
    (plain, refused)
}

/// A FOCE / FOCEI stage does not search a refused subject (`InnerSolvePolicy::skip_refused`):
/// no gradient of either kind runs (`IOV_AGRAD_ROUTE` stays `(0, 0)`), it reports η̂ = κ̂ = 0
/// and the refusal objective. Controls on the same subject and thread: the default policy
/// still solves it (the counter moves, η̂ ≠ 0), a stage that scores the subject itself
/// (AGQ, SAEM) does too, and an unrefused subject under the stage policy is searched.
/// Mutations: drop the skip in `find_ebe_impl` -> the counter moves; make
/// `skips_refused_subjects` true for AGQ -> the AGQ control reads zero.
#[test]
fn a_refused_subjects_inner_search_is_skipped_for_the_foce_family_only() {
    use crate::estimation::inner_optimizer::{find_ebe_for_stage, IOV_AGRAD_ROUTE};
    use crate::types::{EstimationMethod, FitOptions};
    let model = frem_iov_ode();
    let (plain, refused) = frem_iov_pair(&model);
    let params = moved_params(&model);
    let stage = |method: EstimationMethod, interaction: bool| FitOptions {
        method,
        interaction,
        ..FitOptions::default()
    };
    let run = |s: &Subject, o: &FitOptions| {
        IOV_AGRAD_ROUTE.with(|c| c.set((0, 0)));
        let r = find_ebe_for_stage(&model, s, &params, 200, 1e-8, None, None, 0, o);
        (r, IOV_AGRAD_ROUTE.with(|c| c.get()))
    };
    for (method, interaction) in [
        (EstimationMethod::Foce, false),
        (EstimationMethod::FoceI, true),
        (EstimationMethod::FoceGn, true),
    ] {
        let (r, route) = run(&refused, &stage(method, interaction));
        assert_eq!(route, (0, 0), "{method:?}: no gradient of either kind");
        assert!(r.eta.iter().all(|v| *v == 0.0), "{method:?}");
        assert_eq!(
            r.kappas.len(),
            crate::stats::stacked_layout::group_count(&refused)
        );
        assert!(r
            .kappas
            .iter()
            .all(|k| k.len() == model.n_kappa && k.iter().all(|v| *v == 0.0)));
        assert_eq!(r.nll, crate::stats::likelihood::IOV_FREM_REFUSAL_NLL);
        assert!(r.converged && !r.hard_reject);
        assert_eq!(r.h_matrix.nrows(), refused.obs_times.len());
        assert_eq!(r.h_matrix.ncols(), model.n_eta);
        // Control: an unrefused subject under the same stage is searched.
        let (rp, route) = run(&plain, &stage(method, interaction));
        assert!(
            route.0 + route.1 > 0,
            "{method:?}: the plain subject is searched"
        );
        assert!(rp.eta.iter().any(|v| *v != 0.0));
    }
    // Controls: every other policy solves the refused subject.
    let mut quad = stage(EstimationMethod::FoceI, true);
    quad.n_agq = 3;
    for (name, o) in [("agq", quad), ("saem", stage(EstimationMethod::Saem, true))] {
        let (r, route) = run(&refused, &o);
        assert!(route.0 + route.1 > 0, "{name}: solved");
        assert!(
            r.eta
                .iter()
                .chain(r.kappas.iter().flatten())
                .any(|v| *v != 0.0),
            "{name}"
        );
    }
    IOV_AGRAD_ROUTE.with(|c| c.set((0, 0)));
    let r = crate::estimation::inner_optimizer::find_ebe(
        &model, &refused, &params, 200, 1e-8, None, None, 0,
    );
    let route = IOV_AGRAD_ROUTE.with(|c| c.get());
    assert!(route.0 + route.1 > 0);
    assert!(r
        .eta
        .iter()
        .chain(r.kappas.iter().flatten())
        .any(|v| *v != 0.0));
}

/// A FOCE fit of a FREM + κ ODE population with one refused subject reports no
/// finite-difference inner subjects and no `W_PLAN_FD_FALLBACK`: the refused subject was the
/// only FD route under κ. Its objective is the constant, the others' are live.
/// Mutation: the warning's `refused_skipped` flag forced `false` -> "1 of 3 subjects use
/// finite-difference inner gradients".
#[test]
fn a_foce_fit_with_a_refused_frem_subject_reports_no_fd_inner_subjects() {
    use crate::types::{EstimationMethod, FitOptions, Population};
    let model = frem_iov_ode();
    let (plain, refused) = frem_iov_pair(&model);
    let mut other = plain.clone();
    other.id = "2".to_string();
    other.observations.iter_mut().for_each(|v| *v *= 1.07);
    let mut refused = refused;
    refused.id = "3".to_string();
    let pop = Population {
        subjects: vec![plain, other, refused],
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    for (method, interaction) in [
        (EstimationMethod::Foce, false),
        (EstimationMethod::FoceI, true),
    ] {
        let opts = FitOptions {
            method,
            interaction,
            outer_maxiter: 3,
            run_covariance_step: false,
            ..FitOptions::default()
        };
        let r = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("fit");
        let bad: Vec<_> = r
            .warnings
            .iter()
            .filter(|w| w.contains("W_PLAN_FD_FALLBACK") || w.contains("finite-difference inner"))
            .collect();
        assert!(bad.is_empty(), "{method:?}: {bad:?}");
        assert!(
            r.ofv >= crate::stats::likelihood::IOV_FREM_REFUSAL_NLL,
            "{method:?}: the constant is in the objective"
        );
        let sr = r.subjects.iter().find(|s| s.id == "3").expect("subject 3");
        assert!(sr.eta.iter().all(|v| *v == 0.0));
    }
}

// ---- the f64 kernel IS the production objective ------------------------------------------

/// Kernel `F` and `l` at `(params, η)` against production's `foce_subject_nll` / inner
/// objective, on the production predictions and η-Jacobian. Returns `(|ΔF|, |Δl|)`, each
/// relative to `1 + |F|`.
pub(super) fn objective_gaps(
    model: &CompiledModel,
    s: &Subject,
    params: &ModelParameters,
    eta: &[f64],
) -> (f64, f64) {
    objective_gaps_for(model, s, params, eta, true)
}

fn objective_gaps_for(
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
        crate::stats::likelihood::population_variance_preds(model, s, &params.theta, eta.len())
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
    let g: Vec<Vec<f64>> = (0..hm.nrows())
        .map(|j| (0..eta.len()).map(|k| hm[(j, k)]).collect())
        .collect();
    let fk = if interaction {
        crate::sens::focei_kernel::focei_f_grad(&ctx, &preds, &g, eta, &p).map(|r| r.0)
    } else {
        crate::sens::focei_kernel::foce_f(&ctx, &preds, &g, eta, frozen.as_deref(), &p)
    }
    .expect("kernel F");
    let lk = crate::sens::focei_kernel::inner_l(&ctx, &preds, eta, &p, frozen.as_deref())
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
    let g = objective_gaps(&m3, &s3, &m3.default_params, &[0.08, -0.06, 0.1]);
    assert!(g.0 < 1e-12 && g.1 < 1e-12, "diagonal + M3: {g:?}");

    // 2. block_sigma + M3 rows (row-wise for both F and l, live ρ)
    let mut bs = parse_model_string(&dense_src("", true)).unwrap();
    bs.bloq_method = crate::types::BloqMethod::M3;
    let sb = censor(dense_subject(&bs), &[0, 7]);
    let mut pb = bs.default_params.clone();
    // A live ρ different from the declaration: the inner loop must read the live one.
    pb.residual_correlations[0].rho += 0.07;
    let g = objective_gaps(&bs, &sb, &pb, &eta);
    assert!(g.0 < 1e-10 && g.1 < 1e-10, "block_sigma + M3: {g:?}");

    // 3. serial correlation, dense both ways
    let sm = parse_model_string(&dense_src("  correlation ar1(RHO) by ARM\n", false)).unwrap();
    let ss = dense_subject(&sm);
    let g = objective_gaps(&sm, &ss, &sm.default_params, &eta);
    assert!(g.0 < 1e-10 && g.1 < 1e-10, "serial: {g:?}");

    // 4. block_sigma + IIV on the residual: F scales R by exp(2η_r) like l does. The
    // cross-row and serial forms are in `dense_r_with_iiv_on_ruv_*` below (#2656).
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
    let g = objective_gaps(&sr, &srs, &sr.default_params, &[0.08, -0.06, 0.12]);
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
    let g = objective_gaps(&sm, &sf, &sm.default_params, &eta);
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
    let g = objective_gaps_for(&one, &s1, &one.default_params, &[0.08, -0.06, 0.1], false);
    assert!(g.0 < 1e-12 && g.1 < 1e-12, "diagonal: {g:?}");

    let mut m3 = parse_model_string(ONE_CPT_ORAL).unwrap();
    m3.bloq_method = crate::types::BloqMethod::M3;
    let s3 = censor(s1.clone(), &[0, 8]);
    let g = objective_gaps_for(&m3, &s3, &m3.default_params, &[0.08, -0.06, 0.1], false);
    assert!(g.0 < 1e-12 && g.1 < 1e-12, "diagonal + M3: {g:?}");

    let bs = parse_model_string(&dense_src("", true)).unwrap();
    let mut pb = bs.default_params.clone();
    pb.residual_correlations[0].rho += 0.07;
    let g = objective_gaps_for(&bs, &dense_subject(&bs), &pb, &eta, false);
    assert!(g.0 < 1e-10 && g.1 < 1e-10, "block_sigma: {g:?}");

    let sm = parse_model_string(&dense_src("  correlation ar1(RHO) by ARM\n", false)).unwrap();
    let g = objective_gaps_for(&sm, &dense_subject(&sm), &sm.default_params, &eta, false);
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
    // Measured: Jacobian 5.0e-13 (macOS arm64), 4.3e-13 (Linux); FOCEI 3.1e-9 on Padé, 3.8e-8
    // on the spectral path (#2706 follow-up). The analytic gradient is the same on both
    // engines (to the 11 printed digits); the gap is the oracle's. Its inner objective is flat
    // to 4e-15 along one direction at `x[22] + 1e-3`, so the re-solved η̂ lands 1e-10 apart on
    // the two engines, and the FOCEI objective, not stationary in η, moves 9e-11 there — a kink
    // a Richardson step of 1e-3 reads as 3.8e-8. A tighter EBE tol (1e-15) changes neither
    // point. 10× headroom on the Jacobian, 5× on the FOCEI gap.
    assert_wide_iov_parity("ODE", &model, &s, (5e-12, 2e-7));
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

pub(super) const CF_IOV: &str = r#"
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
    // walk's `ode_iov_walk_scope`): against the reconverged objective.
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

/// The kernel's Laplace objective at `η`, from the recorder's exact `(f, G, Q)` and the
/// generic definition of the inner Hessian ([`inner_hessian_generic`](crate::sens::focei_kernel::inner_hessian_generic)).
fn laplace_kernel(
    model: &CompiledModel,
    s: &Subject,
    params: &ModelParameters,
    eta: &[f64],
) -> f64 {
    let jets = canonical_jets(model, s, None, &params.theta, None, eta).expect("recorded η-jets");
    let (f, g) = jets.f_g();
    let q: Vec<Vec<f64>> = (0..jets.rows()).map(|j| jets.row_q(j).to_vec()).collect();
    let ctx = crate::sens::focei_kernel::LikCtx::new(model, s, params);
    let p = crate::sens::focei_kernel::ParT::of(model, s, params);
    let l = crate::sens::focei_kernel::inner_l(&ctx, &f, eta, &p, None).unwrap();
    let h = crate::sens::focei_kernel::inner_hessian_generic(&ctx, &f, &g, &q, eta, &p).unwrap();
    // The Laplace approximation's own `½log|H|`, no jitter (#2538): this fixture's
    // Hessian is positive-definite, where production factors it as is.
    let h = nalgebra::DMatrix::from_fn(eta.len(), eta.len(), |a, b| h[a][b]);
    l + 0.5
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
    for n_agq in [1usize, 3] {
        runtime_declined_subject_is_served_exactly(n_agq);
    }
}

fn runtime_declined_subject_is_served_exactly(n_agq: usize) {
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
    let _off = super::PolishOff::new();
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
    let prod = agq_prod(&one, &s, params, &eta, n_agq, true);
    assert!(
        prod.is_finite() && prod < 1e19,
        "production falls back: {prod}"
    );
    let laplace = Objective::Agq { n_agq, exact: true };
    assert!(
        subject_adjoint_gradient(&one, &s, params, &x, &eta, laplace).is_none(),
        "off the mode, an indefinite H must decline rather than drop wᵀ∇_η l"
    );
    // The old salvage for that subject, the held-EBE score `∂A/∂x` at this fixed `η` (FD of
    // production's objective with `η` held, not reconverged): the straddle's far side.
    let held = subject_adjoint_held_ebe_score(&one, &s, params, &x, &eta, laplace)
        .expect("the held-EBE score serves the declined subject");
    let mut worst = 0.0f64;
    for k in 0..x.len() {
        let h = 1e-5 * (1.0 + x[k].abs());
        let at = |d: f64| {
            let mut xv = x.clone();
            xv[k] += d;
            let p = crate::estimation::parameterization::unpack_params(&xv, params);
            agq_prod(&one, &s, &p, &eta, n_agq, true)
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
    // Measured 4.794e-10 (n_agq = 1), 7.780e-9 (n_agq = 3) (central FD, h = 1e-5); ~10× headroom.
    assert!(
        worst < 1e-7,
        "held-EBE score off the fixed-η FD by {worst:.3e}"
    );
    // FOCE/FOCEI have their own salvage in the outer optimizer: no held route here.
    assert!(subject_adjoint_held_ebe_score(&one, &s, params, &x, &eta, Objective::Focei).is_none());

    // The population gradient (the optimizer's) re-solves the subject's EBE at the trial point
    // and takes the exact score there: it matches the reconverged-FD oracle, which the held
    // score above does not (the straddle), and the warning says so.
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
        n_agq,
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
        None,
    )
    .expect("population gradient");
    let f_at = |xv: &[f64]| {
        let p = crate::estimation::parameterization::unpack_params(xv, params);
        let e = ebe_for(&one, &s, &p, 1e-12, true);
        agq_prod(&one, &s, &p, e.as_slice(), n_agq, true)
    };
    let (mut served, mut old) = (0.0f64, 0.0f64);
    for k in 0..x.len() {
        let fd = |h: f64| {
            let (mut xp, mut xm) = (x.clone(), x.clone());
            xp[k] += h;
            xm[k] -= h;
            (f_at(&xp) - f_at(&xm)) / (2.0 * h)
        };
        let rich = (4.0 * fd(1e-3) - fd(2e-3)) / 3.0;
        assert!(
            rich.is_finite() && g[k].is_finite(),
            "x[{k}]: {} {rich}",
            g[k]
        );
        // g is the OFV gradient, 2·F.
        let e_served = (g[k] - 2.0 * rich).abs() / (1.0 + (2.0 * rich).abs());
        let e_old = (2.0 * held[k] - 2.0 * rich).abs() / (1.0 + (2.0 * rich).abs());
        assert!(e_served.is_finite() && e_old.is_finite(), "x[{k}]");
        served = served.max(e_served);
        old = old.max(e_old);
    }
    eprintln!("n_agq = {n_agq}: served vs reconverged FD {served:.3e}, held vs FD {old:.3e}");
    assert!(
        served < SERVED_TOL,
        "n_agq = {n_agq}: served score off the oracle by {served:.3e}"
    );
    assert!(
        old > 100.0 * served && old > 1e-2,
        "n_agq = {n_agq}: the held score must be far off the oracle ({old:.3e} vs {served:.3e})"
    );
    assert!(
        log.all_declines_exact() && outer_fd_fallback_warning(&population, &log).is_none(),
        "the decline was served exactly at the re-solved EBE: no fallback to warn about"
    );
    assert!(g.iter().any(|v| *v != 0.0));
    // The route and the mode: the served subject reports `ResolvedExact` and hands the
    // re-solved EBE back as polished, so the objective is evaluated where the gradient was
    // taken; without the optimizer's salvage the same decline is a runtime one (a recorder
    // serves the subject), reconverged and not structural.
    {
        use crate::estimation::agq::{ScoreRoute, SubjectScoreContext};
        let bounds = crate::estimation::parameterization::compute_bounds(params);
        let ctx = SubjectScoreContext::new(&one, params, &x, &options, &bounds, false);
        let mut mode = ModeOut::default();
        let (_, route) = ctx
            .score_routed_with_mode(None, &s, &eta, &[], true, &mut mode)
            .expect("score");
        assert_eq!(route, ScoreRoute::ResolvedExact, "n_agq = {n_agq}");
        let ebe = ebe_for(&one, &s, params, 1e-12, true);
        let polished = mode
            .eta_polished
            .expect("the re-solved mode is handed back");
        let gap = polished
            .iter()
            .zip(ebe.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, |m, d| {
                assert!(d.is_finite());
                m.max(d)
            });
        assert!(gap < 1e-6, "polished mode off the EBE by {gap:.2e}");
        let (_, route) = ctx.score_routed(&s, &eta, &[], false).expect("score");
        assert_eq!(
            route,
            ScoreRoute::Reconverged { structural: false },
            "n_agq = {n_agq}"
        );
        // The structural probe (an adjoint recording) is asked only where its answer is used,
        // and once per fit: the covariance step's form (`resolve_ok = false`) never asks, the
        // optimizer's asks once for a subject whose decline the log has seen answered "no", and a
        // context without the log asks at every call. Mutations: probe unconditionally (the first
        // count), drop the log's kept answer (the second), or the log (the third).
        use std::sync::atomic::Ordering::Relaxed;
        let fit_log = OuterFdDeclineLog::new(1);
        let logged = SubjectScoreContext::new(&one, params, &x, &options, &bounds, false)
            .with_declines(Some(&fit_log));
        logged.score_routed(&s, &eta, &[], false).expect("score");
        logged.score(&s, &eta, &[]).expect("score");
        assert_eq!(logged.probes.load(Relaxed), 0, "n_agq = {n_agq}: no probe");
        for _ in 0..3 {
            let (_, route) = logged
                .score_routed_with_mode(Some(0), &s, &eta, &[], true, &mut ModeOut::default())
                .expect("score");
            assert_eq!(route, ScoreRoute::ResolvedExact, "n_agq = {n_agq}");
        }
        assert_eq!(
            logged.probes.load(Relaxed),
            1,
            "n_agq = {n_agq}: probed once"
        );
        for _ in 0..2 {
            ctx.score_routed_with_mode(Some(0), &s, &eta, &[], true, &mut ModeOut::default())
                .expect("score");
        }
        assert_eq!(
            ctx.probes.load(Relaxed),
            3,
            "n_agq = {n_agq}: without the log every call probes"
        );
        // The fused outer-loop evaluation (`agq_population_evaluate`, what the NLopt objective
        // runs) takes the same log: three evaluations of the declined subject ask the probe once.
        // Mutation: pass `None` for the log there -> the log is never asked, 0 probes.
        let loop_log = OuterFdDeclineLog::new(1);
        let anchor = options.hessian_anchor();
        for _ in 0..3 {
            crate::estimation::agq::agq_population_evaluate(
                &one,
                &population,
                params,
                params,
                &x,
                &[DVector::from_column_slice(&eta)],
                &[Vec::new()],
                &bounds,
                &options,
                n_agq,
                anchor,
                false,
                Some(&loop_log),
            );
        }
        assert_eq!(
            loop_log.structural_probes.load(Relaxed),
            1,
            "n_agq = {n_agq}: the fused loop probes once per fit"
        );
    }
}

const SERVED_TOL: f64 = 1e-2;

/// #2713: a FOCEI gradient declined at its mode system says so. Off the mode, on a grid
/// point where the exact `H` is indefinite (`laplace_serves_the_prior_precision_anchor_fallback`'s
/// search), `mode_solve` declines and the pass sets `mode_system_declined`, with the mode
/// response asked for too. The straddle: at the EBE the same subject is served, and the
/// flag left on the reused `ModeOut` is cleared. Mutations that redden it: dropping the
/// flag in `run`, or not resetting it at the start of a pass.
#[test]
fn a_mode_system_decline_is_named() {
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
    let pt = PackedPoint::new(params, &x).expect("packed point");
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
        .find(|e| {
            exact_inner_hessian(&one, &s, params, e)
                .expect("recorded")
                .cholesky()
                .is_none()
        })
        .expect("a point where the exact H is indefinite");
    let ebe = find_ebe(&one, &s, params, 500, 1e-11, None, None, 0).eta;
    for want_dx in [false, true] {
        let mut out = ModeOut {
            want_dx,
            ..ModeOut::default()
        };
        // The decline itself, with the polish off.
        let off = super::PolishOff::new();
        let g =
            subject_adjoint_gradient_and_mode_at(&one, &s, &pt, &eta, Objective::Focei, &mut out);
        drop(off);
        assert!(
            g.is_none() && out.mode_system_declined && out.polish.is_some(),
            "want_dx = {want_dx}: off the mode, H indefinite must decline at the mode system"
        );
        let g = subject_adjoint_gradient_and_mode_at(
            &one,
            &s,
            &pt,
            ebe.as_slice(),
            Objective::Focei,
            &mut out,
        );
        assert!(
            g.is_some() && !out.mode_system_declined,
            "want_dx = {want_dx}: the EBE is served, and the reused flag is cleared"
        );
    }
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
        laplace_kernel(&one, &s, &one.default_params, &eta),
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

/// One subject's 3-node AGQ gradient against the reconverged-FD oracle, for a model some of
/// whose quadrature nodes the value path scores as the sentinel (zero weight in the
/// `logsumexp`, `agq::agq_terms`). The subject must be served, over the nodes that remain,
/// and the oracle scores the same nodes the same way.
fn dropped_node_parity(model: &CompiledModel, etas: &[f64]) -> f64 {
    let s = subject(
        model,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &TIMES,
        etas,
    );
    let mut worst = 0.0f64;
    for exact in [true, false] {
        let w = fd_oracle_agq(model, &s, 3, exact);
        eprintln!("dropped nodes, exact = {exact}: worst {w:.2e}");
        assert!(w.is_finite());
        worst = worst.max(w);
    }
    worst
}

/// A quadrature node whose predictions are `NaN` is dropped, as the objective drops it: the
/// subject is served exactly, not declined to finite differences. `CL = 16*TVCL*e^eta*(c -
/// eta^3)^(1/2)` (the parser's `sqrt` clamps at 0; `^` does not) is `NaN` for `eta_CL^3 > c`,
/// `c = 0.008`: the mode `eta_CL ~ 0.01` and the lower nodes are finite, and 9 of the 27 nodes
/// (the upper `eta_CL` third) are not. Those nodes' forwards record, with `NaN` predictions,
/// so the first gradient does not form and the retry over the 18 live nodes serves it.
/// Mutations: never retrying (`return None` after the first run), or keeping every node in
/// the retry, each kill it.
#[test]
fn a_quadrature_node_with_nan_predictions_is_dropped_not_declined() {
    let src = ONE_CPT_ORAL.replace(
        "CL = TVCL * exp(ETA_CL)",
        "CL = TVCL * exp(ETA_CL) * 16.0 * (0.008 - (ETA_CL ^ 3)) ^ 0.5",
    );
    let model = parse_model_string(&src).unwrap();
    let worst = dropped_node_parity(&model, &[0.1, -0.05, 0.2]);
    // Measured 2.6e-8 (Linux); ~20x headroom.
    assert!(worst < 6e-7, "{worst:.2e}");
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
        per_subject_packed_gradients, subject_packed_gradient,
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
    use crate::sens::jet::DualT;
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
        let g = objective_gaps(&model, &s, &model.default_params, &[0.08, -0.06, 0.1]);
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

// ---- magnitude expressions per endpoint (#2731) -----------------------------------------

/// Two endpoints on one sigma each, a θ factor gated on `TAD` on the first and a θ factor
/// on the second's own sigma, so the direct-θ channel is live on both endpoints and differs
/// between them.
const ODE_ENDPOINT_MAGNITUDE: &str = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(30.0, 1.0, 500.0)
  theta FU(0.3, 0.01, 1.0)
  theta TH_M(1.4, 0.1, 10.0)
  theta TH_A(0.7, 0.1, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.06
  sigma PROP_TOT ~ 0.1 (sd)
  sigma PROP_UNB ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV * exp(ETA_V)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y[CMT=1] = central / V
  y[CMT=2] = FU * central / V
[error_model]
  CMT=1: DV ~ proportional(PROP_TOT * (if (TAD < 3.0) TH_M else 1.0))
  CMT=2: DV ~ proportional(PROP_UNB * TH_A)
"#;

/// Two doses and a total and an unbound row at every time, on [`ODE_ENDPOINT_MAGNITUDE`]
/// and its variants.
fn endpoint_magnitude_subject(model: &CompiledModel) -> Subject {
    let base = [1.0, 2.0, 5.0, 9.0, 13.0, 14.0, 18.0, 23.0];
    let times: Vec<f64> = base.iter().flat_map(|&t| [t, t]).collect();
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let mut s = subject_shell(doses, &times);
    s.obs_cmts = (0..times.len()).map(|j| 1 + j % 2).collect();
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
    s
}

/// The factor differs across rows of one subject, or the fixture cannot tell a row's
/// endpoint from any other.
fn assert_rows_differ(model: &CompiledModel, s: &Subject) {
    assert!(
        model
            .ruv_magnitude
            .as_ref()
            .is_some_and(|rm| rm.has_endpoint_magnitude() && rm.theta_dependent),
        "fixture must carry a θ-dependent endpoint magnitude"
    );
    let rows = model.ruv_obs_mult(s, &model.default_params.theta).unwrap();
    assert!(rows.iter().any(|r| *r != rows[0]), "{rows:?}");
}

fn assert_endpoint_magnitude_parity(model: &CompiledModel, s: &Subject) {
    assert_rows_differ(model, s);
    let g = objective_gaps(model, s, &model.default_params, &[0.08, -0.06]);
    eprintln!("objective gaps {g:?}");
    assert!(g.0 < 1e-12 && g.1 < 1e-12, "objective {g:?}");
    for (label, worst) in [
        ("FOCEI", fd_oracle(model, s, true)),
        ("FOCE", fd_oracle(model, s, false)),
        ("Laplace", fd_oracle_laplace(model, s)),
    ] {
        eprintln!("{label}: worst {worst:.2e}");
        // Measured worst 4.2e-9 over the four fixtures (macOS arm64); ~25×.
        assert!(worst < 1e-7, "{label}: {worst:.2e}");
    }
}

/// #2731: every θ and σ against the reconverged FD oracle, on `CMT=N:` lines. Kills the
/// adjoint reading a row's magnitude from a table other than the production row's: the
/// objective gap then opens, since production keys `ruv_obs_mult` on the endpoint.
#[test]
fn cmt_line_magnitude_objective_and_gradients() {
    let model = tight(parse_model_string(ODE_ENDPOINT_MAGNITUDE).unwrap());
    let s = endpoint_magnitude_subject(&model);
    assert_endpoint_magnitude_parity(&model, &s);
}

/// One sigma on both endpoints under different factors: the shape a per-slot multiplier
/// cannot represent, and where the σ column sums both endpoints' rows.
#[test]
fn shared_sigma_cmt_magnitude_objective_and_gradients() {
    let src = ODE_ENDPOINT_MAGNITUDE
        .replace("  sigma PROP_UNB ~ 0.15 (sd)\n", "")
        .replace(
            "proportional(PROP_UNB * TH_A)",
            "proportional(PROP_TOT * TH_A)",
        );
    let model = tight(parse_model_string(&src).unwrap());
    let s = endpoint_magnitude_subject(&model);
    assert_endpoint_magnitude_parity(&model, &s);
}

/// Cross-endpoint `block_sigma` pairs: each row of a pair loads its own endpoint's factor,
/// so `R`'s off-diagonal carries the product of two different multipliers.
#[test]
fn block_sigma_cmt_magnitude_objective_and_gradients() {
    let src = ODE_ENDPOINT_MAGNITUDE.replace(
        "  sigma PROP_TOT ~ 0.1 (sd)\n  sigma PROP_UNB ~ 0.15 (sd)\n",
        "  block_sigma (PROP_TOT, PROP_UNB) = [0.01, 0.006, 0.0225]\n",
    );
    let model = tight(parse_model_string(&src).unwrap());
    let s = endpoint_magnitude_subject(&model);
    assert!(model.residual_cov_at(&model.default_params).is_dense());
    assert_endpoint_magnitude_parity(&model, &s);
}

/// `if/else` branches on the closed-form engine, sharing a sigma with a factor on one
/// branch and a `combined` on the other, with `EARLY` changing inside the subject. The
/// `Selected` key is a branch index, not a `CMT`.
fn selected_magnitude_fixture() -> (CompiledModel, Subject) {
    let src = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta TH_M(1.4, 0.1, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.04
  sigma PROP_ERR ~ 0.1 (sd)
  sigma ADD_ERR ~ 0.05 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV * exp(ETA_V)
  KA = TVKA
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[covariates]
  EARLY continuous
[error_model]
  if (EARLY > 0.5) {
    DV ~ proportional(PROP_ERR * TH_M)
  } else {
    DV ~ combined(PROP_ERR, ADD_ERR)
  }
"#;
    let model = tight(parse_model_string(src).unwrap());
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let times = [1.0, 2.0, 4.0, 8.0, 13.0, 14.0, 18.0, 24.0];
    let mut s = subject_shell(doses, &times);
    s.covariates.insert("EARLY".into(), 0.0);
    s.obs_covariates = (0..times.len())
        .map(|j| HashMap::from([("EARLY".to_string(), (j % 2) as f64)]))
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
    (model, s)
}

#[test]
fn if_else_magnitude_objective_and_gradients() {
    let (model, s) = selected_magnitude_fixture();
    assert_endpoint_magnitude_parity(&model, &s);
}

/// The covariance step's R-matrix reads the row through its own call site
/// (`ode_adjoint_hessian`), which the gradient oracles above never reach.
#[test]
fn endpoint_magnitude_cov_hessian_matches_fd() {
    let (model, s) = selected_magnitude_fixture();
    let (fd, _) = hessian_gaps(&model, &s, true);
    // Measured 7.7e-12 (macOS arm64); ~13×.
    assert!(fd < 1e-10, "{fd:.2e}");
}

// ---- closed-form analytic `init(...)` impulses -----------------------------------------

pub(super) const CF_INIT: &str = r#"
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

/// The worst `|got − want|` relative to `want`'s largest entry. Every entry (and the scale it
/// divides by) is asserted finite first: a `max` fold drops a `NaN`, so a diverged entry would
/// otherwise pass green.
pub(super) fn rel_gap(got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let mut scale = 0.0f64;
    for &v in want {
        assert!(v.is_finite(), "non-finite reference entry {v}");
        scale = scale.max(v.abs());
    }
    let mut worst = 0.0f64;
    for (&a, &b) in got.iter().zip(want) {
        assert!(
            a.is_finite() && b.is_finite() && scale.is_finite(),
            "non-finite entry {a} vs {b}"
        );
        worst = worst.max((a - b).abs() / scale.max(1e-300));
    }
    worst
}

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
    // The conditional mode to the precision the exact derivatives allow: an ODE inner search
    // stops on an objective stall (#555), where the mode is only good to ~1e-9 and where it
    // lands depends on the path the line search took — on the SDE fixture a trial step out
    // at `KA ≈ 500`, integrated stiffly under `auto` and explicitly under `rk45`, moved it
    // 4.5e-9, and the difference quotient over `h = 1e-3` amplified that to a 1.7e-6 gap. A
    // few Newton steps on the exact inner gradient and Hessian take any such start to the
    // same mode. (FOCE's frozen-variance mode, and a subject with a non-Gaussian term the
    // kernel's inner gradient does not carry, are left as found.)
    let polish = interaction && s.obs_records.is_empty();
    let ebe = |p: &ModelParameters| {
        let mut e = ebe_for(model, s, p, 1e-12, interaction);
        for _ in 0..3 * usize::from(polish) {
            let (Some(g), Some(h)) = (
                exact_inner_gradient(model, s, p, e.as_slice(), None),
                exact_inner_hessian(model, s, p, e.as_slice()),
            ) else {
                break;
            };
            // A nested-levels subject's exact inner derivatives are over its own resolved
            // cells, not the full stacked layout `ebe_for` returns: no polish there.
            if g.len() != e.len() {
                break;
            }
            match h.cholesky() {
                Some(c) => e -= c.solve(&nalgebra::DVector::from_vec(g)),
                None => break,
            }
        }
        e
    };
    let eta = ebe(&params);
    // On a thread with the fit pool's worker stack (`FIT_RAYON_STACK_SIZE`), where
    // production runs the nested-dual passes; a test thread's default 2 MB is too small.
    // `(reverse per direction, every plan along the directions, no mu fold)`; returns the
    // Hessian and how many plans the pass assembled from a basis.
    // `(…, every pass through the recorded likelihood kernel)`; also how many passes took the
    // structured kernel.
    let hessian_in4 = |per_direction: bool, plan_dirs: bool, fold_off: bool, struct_off: bool| {
        std::thread::scope(|scope| {
            std::thread::Builder::new()
                .stack_size(crate::FIT_RAYON_STACK_SIZE)
                .spawn_scoped(scope, || {
                    use super::hessian::basis::{BASIS_PLANS, MU_FOLD_OFF, PLAN_PER_DIRECTION};
                    use super::hessian::st::{STRUCT_OFF, STRUCT_PASSES};
                    super::hessian::REVERSE_PER_DIRECTION.with(|c| c.set(per_direction));
                    PLAN_PER_DIRECTION.with(|c| c.set(plan_dirs));
                    MU_FOLD_OFF.with(|c| c.set(fold_off));
                    STRUCT_OFF.with(|c| c.set(struct_off));
                    BASIS_PLANS.with(|c| c.set(0));
                    STRUCT_PASSES.with(|c| c.set(0));
                    let h = subject_adjoint_cov_hessian(
                        model,
                        s,
                        &params,
                        &x,
                        eta.as_slice(),
                        objective,
                    );
                    (
                        (h, BASIS_PLANS.with(std::cell::Cell::get)),
                        STRUCT_PASSES.with(std::cell::Cell::get),
                    )
                })
                .expect("spawn")
                .join()
                .expect("join")
        })
    };
    let hessian_in = |per_direction: bool, plan_dirs: bool, fold_off: bool| {
        hessian_in4(per_direction, plan_dirs, fold_off, false).0
    };
    let hessian = |per_direction: bool| {
        hessian_in(per_direction, false, false)
            .0
            .expect("subject must be in adjoint scope")
    };
    // The plans along a basis of their inputs (#2980) against every plan along the directions,
    // and the mu fold against none: the same derivatives assembled by linearity (and, folded,
    // through `∂μ/∂θ`), so they differ by rounding only.
    {
        let (h_basis, n_basis) = hessian_in(false, false, false);
        let (h_dirs, _) = hessian_in(false, true, false);
        let (h_nofold, _) = hessian_in(false, false, true);
        let (h_basis, h_dirs, h_nofold) = (
            h_basis.expect("basis: in scope"),
            h_dirs.expect("directions: in scope"),
            h_nofold.expect("no fold: in scope"),
        );
        let (g_dirs, g_fold) = (
            rel_gap(h_basis.as_slice(), h_dirs.as_slice()),
            rel_gap(h_basis.as_slice(), h_nofold.as_slice()),
        );
        eprintln!(
            "hessian: {n_basis} basis plans; basis vs directions {g_dirs:.2e}, fold vs none {g_fold:.2e}"
        );
        assert!(g_dirs < 1e-12, "basis vs directions: {g_dirs:.2e}");
        assert!(g_fold < 1e-12, "fold vs none: {g_fold:.2e}");
        // The structured FOCEI kernel (#2980) against the recorded kernel's sweep.
        let ((h_st, _), n_st) = hessian_in4(false, false, false, false);
        let ((h_rec, _), n_rec) = hessian_in4(false, false, false, true);
        let (h_st, h_rec) = (
            h_st.expect("structured: in scope"),
            h_rec.expect("recorded: in scope"),
        );
        let g_st = rel_gap(h_st.as_slice(), h_rec.as_slice());
        eprintln!("hessian: {n_st} structured passes ({n_rec} with it off); structured vs recorded {g_st:.2e}");
        assert_eq!(
            n_rec, 0,
            "the switch must take every pass through the recorded kernel"
        );
        assert!(g_st < 1e-12, "structured vs recorded kernel: {g_st:.2e}");
    }
    // Forward over reverse (production) against reverse over forward, one reverse per
    // direction (#2980): the same second derivatives of the same programs, so they differ by
    // the accumulation order only. Every fixture of this oracle is also that parity check.
    // Measured over its 63 fixture runs (closed form, ODE, SDE, IOV, NN, steady state,
    // non-Gaussian terms; FOCE, FOCEI, Laplace, AGQ): worst 2.1e-14 relative to the largest
    // entry, so `1e-12` leaves ~50x headroom. A dropped lane-glue term (θ's `d²θ/dx²`, a value
    // lane read for a tangent) moves an entry by O(1) of the matrix.
    let h = hessian(false);
    let h_rof = hessian(true);
    let worst_for = rel_gap(h.as_slice(), h_rof.as_slice());
    eprintln!("hessian: forward over reverse vs a reverse per direction {worst_for:.2e}");
    assert!(
        worst_for < 1e-12,
        "forward over reverse vs a reverse per direction: {worst_for:.2e}"
    );
    let n = x.len();
    assert_eq!((h.nrows(), h.ncols()), (n, n));
    let mut fd = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        let grad_at = |d: f64| {
            let mut xp = x.clone();
            xp[i] += d;
            let pp = unpack_params(&xp, &params);
            let e = ebe(&pp);
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
    let peak = h.iter().map(|v| v.abs()).fold(0.0, f64::max); // nan-ok: the fold is asserted finite on the next line
    assert!(peak.is_finite(), "a non-finite Hessian entry");
    assert!(peak > 1e-1, "a vanishing Hessian tests nothing");
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

/// The covariance Hessian has no width rule: nine effects under Laplace's exact anchor (it
/// once served at most 8) and under FOCEI (24) are both served, as three are.
#[test]
fn the_covariance_hessian_serves_every_width() {
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
    for (label, m, s, objective) in [
        ("3 η, Laplace", &three, &s3, laplace),
        ("9 η, Laplace", &nine, &s9, laplace),
        ("9 η, FOCEI", &nine, &s9, Objective::Focei),
    ] {
        assert!(
            served(m, s, objective),
            "{label}: the Hessian must be served"
        );
    }
}

/// A subject with 50 random effects, past the old 48-wide dispatch ceiling, gets an exact
/// gradient with no width gate: FOCEI, FOCE and Laplace against reconverged FD of the
/// production objective, and its covariance R-matrix is served (it declined past 8 under
/// Laplace's exact anchor and 24 otherwise). Nine observations on a one-compartment oral
/// model; the 50 effects add into CL, V and KA in turn, so only their sums reach the data and
/// the prior alone identifies each, a well-conditioned `H`.
/// Mutation, run: restoring a width cap in `likelihood_supported` fails the `expect` in
/// `fd_oracle` ("subject must be in adjoint scope").
#[test]
fn a_subject_with_fifty_random_effects_has_an_exact_gradient() {
    let m = parse_model_string(&one_cpt_oral_n_eta(50)).unwrap();
    assert_eq!(m.n_eta, 50);
    let s = subject(
        &m,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &TIMES,
        &(0..50)
            .map(|k| 0.1 * ((k as f64) * 0.9).sin())
            .collect::<Vec<_>>(),
    );
    for (name, interaction) in [("FOCEI", true), ("FOCE", false)] {
        let g = fd_oracle(&m, &s, interaction);
        eprintln!("50 eta {name}: gradient vs reconverged FD {g:.2e}");
        // Measured 5.7e-9 (FOCEI) / 6.8e-9 (FOCE) on Linux; ~10× headroom.
        assert!(g < 7e-8, "{name}: {g:.2e}");
    }
    let g = fd_oracle_laplace(&m, &s);
    eprintln!("50 eta Laplace: gradient vs reconverged FD {g:.2e}");
    // Measured 6.7e-11 on Linux; ~10× headroom.
    assert!(g < 7e-10, "Laplace: {g:.2e}");
    let params = moved_params(&m);
    let x = pack_params(&params);
    let eta = ebe_for(&m, &s, &params, 1e-10, true);
    let laplace = Objective::Agq {
        n_agq: 1,
        exact: true,
    };
    for objective in [Objective::Focei, laplace] {
        let h = with_fit_stack(|| {
            subject_adjoint_cov_hessian(&m, &s, &params, &x, eta.as_slice(), objective)
        })
        .expect("a 50-effect R-matrix is served");
        assert_eq!(h.nrows(), x.len());
        assert!(h.iter().all(|v| v.is_finite()), "{objective:?}");
    }
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
    // Equal to rounding, not bit for bit: the plans' bases are built along the axes the
    // request's directions move (#2980), so a block's columns combine a different basis than
    // the full Hessian's. Measured 6.3e-16 here; a wrong column is O(1).
    assert!(worst < 1e-13, "{worst:.2e}");
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
    // Equal to rounding, not bit for bit: the plans' bases are built along the axes the
    // request's directions move (#2980), so a block's columns combine a different basis than
    // the full Hessian's. Measured 1.8e-15 here; a wrong column is O(1).
    assert!(worst < 1e-13, "{worst:.2e}");
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
    // Equal to rounding, not bit for bit: the plans' bases are built along the axes the
    // request's directions move (#2980), so a block's columns combine a different basis than
    // the full Hessian's. Measured 4.2e-15 here; a wrong column is O(1).
    assert!(worst < 1e-13, "{worst:.2e}");
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

/// The covariance Hessian's production passes (structured: one pass of every direction;
/// recorded: one per direction) equal passes of exactly one and of two directions
/// (`PASS_WIDTH`), on a plain subject, an AGQ one (`n_agq = 3`, non-exact anchor), an IOV one
/// (stacked η/κ widths, per-occasion tangents) and, with `nn`, a reduced-basis `[covariate_nn]`
/// one. Width 2 over 7 packed axes leaves a ragged last pass, so an offset taken from the pass's
/// own width rather than the planned one misplaces columns; width 1 has none, and is the
/// reference. Measured: equal to rounding (a column's arithmetic does not see the other
/// tangent); a misplaced column moves the matrix by O(1) of its largest entry.
fn narrow_equals_production(label: &str, model: &CompiledModel, s: &Subject, o: Objective) {
    use super::hessian::PASS_WIDTH;
    let params = moved_params(model);
    let x = pack_params(&params);
    let eta = ebe_for(model, s, &params, 1e-12, o != Objective::Foce);
    let run = |w: Option<usize>| {
        with_fit_stack(|| {
            PASS_WIDTH.with(|c| c.set(w));
            let h = subject_adjoint_cov_hessian(model, s, &params, &x, eta.as_slice(), o);
            PASS_WIDTH.with(|c| c.set(None));
            h
        })
        .unwrap_or_else(|| panic!("{label}: in scope"))
    };
    let narrow = run(Some(1));
    for (name, h) in [("width 2", run(Some(2))), ("production", run(None))] {
        assert_eq!(narrow.shape(), h.shape(), "{label}: shape");
        let worst = rel_gap(narrow.as_slice(), h.as_slice());
        eprintln!("{label}: width 1 vs {name}: {worst:.2e}");
        assert!(worst < 1e-12, "{label}: width 1 vs {name}: {worst:.2e}");
    }
}

#[test]
fn cov_hessian_narrow_passes_equal_the_production_ones() {
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
    assert_eq!(
        pack_params(&moved_params(&one)).len(),
        7,
        "the fixture must leave a ragged last pass at width 2"
    );
    narrow_equals_production("FOCEI", &one, &s, Objective::Focei);
    narrow_equals_production(
        "AGQ-3",
        &one,
        &s,
        Objective::Agq {
            n_agq: 3,
            exact: false,
        },
    );
    let iov = parse_model_string(ODE_IOV).unwrap();
    let s_iov = iov_subject(&iov);
    narrow_equals_production("IOV", &iov, &s_iov, Objective::Focei);
    #[cfg(feature = "nn")]
    {
        let (m, s) = covariate_nn_fixture();
        narrow_equals_production("covariate NN", &m, &s, Objective::Focei);
    }
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
                n_agq: 1,
                exact: false,
            },
            "Laplace H̃",
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

/// #2720: a circadian baseline with the phase random effect inside `cos` and `sin`.
const CIRCADIAN: &str = r#"
[parameters]
  theta TVBASE(120.0, 1.0, 1000.0)
  theta TVAMP1(0.12, 0.001, 1.0)
  theta PHS1(0.9, -5.0, 5.0)
  theta TVAMP2(0.04, 0.001, 1.0)
  omega ETA_BASE ~ 0.03
  omega ETA_AMP1 ~ 0.08
  omega ETA_SHIFT ~ 0.5
  sigma PROP ~ 0.03 (sd)
[individual_parameters]
  BASE  = TVBASE * exp(ETA_BASE)
  AMP1  = TVAMP1 * exp(ETA_AMP1)
  SHIFT = ETA_SHIFT
[structural_model]
  CIRC = AMP1 * cos(2 * 3.141592653589793 * (TIME + SHIFT) / 24 - PHS1) + TVAMP2 * sin(4 * 3.141592653589793 * (TIME + SHIFT) / 24 - 0.5)
  y = BASE * (1 + CIRC)
[error_model]
  DV ~ proportional(PROP)
"#;

/// #2720: `sin`/`cos` with a random effect in the argument, through every order the
/// recorders read. The FOCE/FOCEI gradient differentiates the lifted `φ′`, the Laplace
/// exact anchor's taped-Hessian sweep `φ″` (`lift3`), and the covariance step's
/// nested dual `φ‴`/`φ⁗` (`DualT::lift3`, `lift4`) — each against the reconverged
/// FD of the production objective, which shares no derivative code with the adjoint.
///
/// `ω(SHIFT)` is kept small on purpose. The inner objective is periodic in a phase η,
/// so it can have several modes, and at `ω = 3 h²` the FOCE oracle's re-solve at
/// `x[0] + 1e-3` landed on a second one (`η̂_SHIFT` 1.71 → 6.11, measured): a 31% "gap"
/// that was the oracle switching basins, not the adjoint. At `ω = 0.5` the mode is
/// unique over the stencil.
#[test]
fn circadian_sin_cos_matches_reconverged_fd() {
    let model = crate::parser::model_parser::parse_full_model(CIRCADIAN)
        .unwrap()
        .model;
    assert!(model.is_algebraic());
    let s = subject(&model, vec![], &TIMES, &[0.1, -0.2, 0.6]);
    assert!(
        in_scope(&model, &s),
        "the compartment-free recorder must serve a sin/cos subject"
    );
    // Measured (Linux): FOCEI 6.7e-9, FOCE 7.4e-9, Laplace 3.3e-11, Hessians FOCEI 8.8e-11
    // and Laplace 1.3e-10 — the oracles' floor; the bounds are this file's shared ones.
    let g = fd_oracle(&model, &s, true);
    eprintln!("circadian FOCEI gradient vs reconverged FD {g:.2e}");
    assert!(g < 1e-5, "FOCEI gradient vs FD {g:.2e}");
    let g = fd_oracle(&model, &s, false);
    eprintln!("circadian FOCE gradient vs reconverged FD {g:.2e}");
    assert!(g < FOCE_FD_TOL, "FOCE gradient vs FD {g:.2e}");
    let g = fd_oracle_agq(&model, &s, 1, true);
    eprintln!("circadian Laplace gradient vs reconverged FD {g:.2e}");
    assert!(g < 1e-5, "Laplace gradient vs FD {g:.2e}");
    for (objective, name) in [
        (Objective::Focei, "FOCEI"),
        (
            Objective::Agq {
                n_agq: 1,
                exact: true,
            },
            "Laplace",
        ),
    ] {
        let (fd, _) = hessian_gaps_for(&model, &s, objective);
        eprintln!("circadian {name} Hessian vs reconverged FD {fd:.2e}");
        assert!(fd < 1e-5, "{name} Hessian vs FD {fd:.2e}");
    }
}

/// #2603: a bounded Gaussian endpoint read through `pnorm`, the random effect inside the
/// link. The FOCE/FOCEI gradient differentiates the lifted `φ`, the Laplace exact anchor's
/// taped Hessian `φ′`, and the covariance step's nested dual `φ″`/`φ‴` — each against the
/// reconverged FD of the production objective.
#[test]
fn pnorm_link_matches_reconverged_fd() {
    let model = crate::parser::model_parser::parse_full_model(
        r#"
[parameters]
  theta TVL(-0.2, -5.0, 5.0)
  theta SLOPE(0.08, -1.0, 1.0)
  omega ETA_L ~ 0.3
  omega ETA_S ~ 0.05
  sigma ADD ~ 0.05 (sd)
[individual_parameters]
  LP = TVL + ETA_L + SLOPE * exp(ETA_S)
[structural_model]
  y = pnorm(LP + SLOPE * TIME * exp(ETA_S))
[error_model]
  DV ~ additive(ADD)
"#,
    )
    .unwrap()
    .model;
    assert!(model.is_algebraic());
    let s = subject(&model, vec![], &TIMES, &[0.2, -0.1]);
    assert!(
        in_scope(&model, &s),
        "the compartment-free recorder must serve a pnorm subject"
    );
    let g = fd_oracle(&model, &s, true);
    eprintln!("pnorm FOCEI gradient vs reconverged FD {g:.2e}");
    assert!(g < 1e-5, "FOCEI gradient vs FD {g:.2e}");
    let g = fd_oracle(&model, &s, false);
    eprintln!("pnorm FOCE gradient vs reconverged FD {g:.2e}");
    assert!(g < FOCE_FD_TOL, "FOCE gradient vs FD {g:.2e}");
    let g = fd_oracle_agq(&model, &s, 1, true);
    eprintln!("pnorm Laplace gradient vs reconverged FD {g:.2e}");
    assert!(g < 1e-5, "Laplace gradient vs FD {g:.2e}");
    for (objective, name) in [
        (Objective::Focei, "FOCEI"),
        (
            Objective::Agq {
                n_agq: 1,
                exact: true,
            },
            "Laplace",
        ),
    ] {
        let (fd, _) = hessian_gaps_for(&model, &s, objective);
        eprintln!("pnorm {name} Hessian vs reconverged FD {fd:.2e}");
        assert!(fd < 1e-5, "{name} Hessian vs FD {fd:.2e}");
    }
}

/// #2603: the `[binary_model]` probit term on every adjoint route — the outer gradient
/// under FOCEI and FOCE, the Laplace and AGQ(3) gradients, and the covariance step's
/// Hessian under each objective — against the reconverged FD of the production objective.
/// The predictor carries the random effect nonlinearly (`exp(0.5·η)` on the time slope), so
/// every order of the lifted `−log Φ` is live: `g′` in the gradients, `g″` in the Laplace
/// log-determinant, `g‴`/`g⁗` in its gradient and the R-matrix. The model-level gates are
/// asserted first, so the test cannot pass on a finite-difference fallback.
///
/// The records put `y = 1` at a strongly negative predictor and `y = 0` at a positive one,
/// so `s = 2y − 1` takes both signs on records where `m` is far from its `lp = 0` value.
#[cfg(feature = "survival")]
#[test]
fn the_probit_binary_term_is_exact_on_every_adjoint_route() {
    use crate::types::ObsRecord;
    let model = parse_model_string(
        "[parameters]\n  theta TH0(-0.3, -10.0, 10.0)\n  theta THT(0.2, -10.0, 10.0)\n  \
         omega ETA_I ~ 0.3\n[binary_model]\n  cmt    = 3\n  link   = probit\n  \
         probit = TH0 + ETA_I + THT * TIME * exp(0.5 * ETA_I)\n",
    )
    .unwrap();
    let rows = [
        (0.0, 0),
        (1.0, 1),
        (2.0, 1),
        (4.0, 0),
        (6.0, 1),
        (9.0, 0),
        (-8.0, 1),
    ];
    let s = Subject {
        id: "1".into(),
        obs_records: rows
            .iter()
            .map(|&(time, state)| ObsRecord::DiscreteState {
                time,
                raw_time: time,
                state,
                cmt: 3,
            })
            .collect(),
        ..Default::default()
    };
    assert!(outer_model_in_scope(&model), "the outer route is analytic");
    let mut gaps = Vec::new();
    for interaction in [true, false] {
        gaps.push((
            format!("outer interaction={interaction}"),
            fd_oracle(&model, &s, interaction),
        ));
    }
    for (n_agq, exact) in [(1, true), (3, true), (3, false)] {
        gaps.push((
            format!("AGQ n_agq={n_agq} exact={exact}"),
            fd_oracle_agq(&model, &s, n_agq, exact),
        ));
    }
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
    ] {
        assert!(
            super::cov_non_gaussian_served(&model, objective),
            "{objective:?}: the model-level gate serves the probit term"
        );
        gaps.push((
            format!("cov {objective:?}"),
            hessian_gaps_for(&model, &s, objective).0,
        ));
    }
    for (what, gap) in &gaps {
        eprintln!("probit {what}: {gap:.2e}");
        assert!(gap.is_finite(), "{what}: {gap}");
    }
    // Measured, macOS arm64 and Linux aarch64 alike: every gradient 1.9e-11, every R-matrix
    // 3.8e-11 — the oracles' floor. 1e-9 keeps 26–50× headroom over it (review r1 #6); the
    // file's shared bounds (5e-7, 1e-5) would pass a 1e-4 relative error in `g‴` / `g⁗`.
    for (what, gap) in &gaps {
        let bound = 1e-9;
        assert!(*gap < bound, "{what}: {gap:.2e}");
    }
}

/// #2603: the inner loop's exact η-gradient and η-Hessian of the probit term
/// (`discrete_subject_eta_grad` / `discrete_subject_eta_hessian`) against central
/// differences of the value path `discrete_subject_nll`, which shares no derivative code
/// with them. Mutation: the probit arm's `g″` zeroed (the Hessian), or `s` dropped (both).
#[cfg(feature = "survival")]
#[test]
fn the_probit_inner_eta_derivatives_match_fd_of_the_value() {
    use crate::categorical::{
        discrete_subject_eta_grad, discrete_subject_eta_hessian, discrete_subject_nll,
    };
    use crate::types::ObsRecord;
    let model = parse_model_string(
        "[parameters]\n  theta TH0(-0.3, -10.0, 10.0)\n  theta THT(0.2, -10.0, 10.0)\n  \
         omega ETA_I ~ 0.3\n  omega ETA_T ~ 0.1\n[binary_model]\n  cmt    = 3\n  \
         probit = TH0 + ETA_I + THT * TIME * exp(ETA_T)\n",
    )
    .unwrap();
    let s = Subject {
        id: "1".into(),
        obs_records: [(0.0, 0), (1.0, 1), (2.0, 1), (4.0, 0), (6.0, 1)]
            .iter()
            .map(|&(time, state)| ObsRecord::DiscreteState {
                time,
                raw_time: time,
                state,
                cmt: 3,
            })
            .collect(),
        ..Default::default()
    };
    let theta = [-0.3, 0.2];
    let eta = [0.4, -0.3];
    let f = |e: &[f64]| discrete_subject_nll(&model, &s, &theta, e);
    let (v, g) = discrete_subject_eta_grad(&model, &s, &theta, &eta).expect("exact gradient");
    let (vh, h) = discrete_subject_eta_hessian(&model, &s, &theta, &eta).expect("exact Hessian");
    assert_eq!(v.to_bits(), f(&eta).to_bits());
    assert_eq!(vh.to_bits(), f(&eta).to_bits());
    let st = 1e-5;
    let at = |i: usize, d: f64| {
        let mut e = eta;
        e[i] += d;
        e
    };
    for i in 0..2 {
        let fd = (f(&at(i, st)) - f(&at(i, -st))) / (2.0 * st);
        assert!(g[i].is_finite() && fd.is_finite());
        assert!((g[i] - fd).abs() < 1e-8, "∂/∂η{i}: {} vs FD {fd}", g[i]);
        let gp = discrete_subject_eta_grad(&model, &s, &theta, &at(i, st))
            .unwrap()
            .1;
        let gm = discrete_subject_eta_grad(&model, &s, &theta, &at(i, -st))
            .unwrap()
            .1;
        for j in 0..2 {
            let fd = (gp[j] - gm[j]) / (2.0 * st);
            assert!(h[(i, j)].is_finite() && fd.is_finite());
            assert!(
                (h[(i, j)] - fd).abs() < 1e-7,
                "H[{i},{j}]: {} vs FD {fd}",
                h[(i, j)]
            );
        }
    }
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
/// Against the reconverged FD of production's objective.
#[cfg(feature = "nn")]
#[test]
fn a_condition_on_a_network_output_takes_the_production_branch() {
    let src = COVARIATE_NN_IV.replace(
        "  CL = TYPICAL_PK.CL * exp(ETA_CL)\n",
        "  if (TYPICAL_PK.CL > 1.0) {\n    CL = TYPICAL_PK.CL * exp(ETA_CL)\n  } else {\n    \
         CL = 2.0 * exp(ETA_CL)\n  }\n",
    );
    assert_ne!(src, COVARIATE_NN_IV, "the fixture must carry the branch");
    let (model, s) = covariate_nn_fixture_from(&src);
    let worst = fd_oracle(&model, &s, true);
    eprintln!("NN-output condition vs reconverged FD {worst:.2e}");
    // Measured worst 3.1e-7 on Linux (the FD's own noise; the same as the branch-free
    // fixture's); ~10× headroom. The `else` arm misses by O(1).
    assert!(worst < 3e-6, "NN-output condition vs FD {worst:.2e}");
}

/// [`COVARIATE_NN_IV`] as an SDE: the two-compartment system as `[odes]` over amounts, with
/// process noise on `central` (the EKF variance rows, whose plan reads the network outputs as
/// parameter slots under the reduced basis).
#[cfg(feature = "nn")]
const COVARIATE_NN_SDE: &str = r#"
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
  ode(obs_cmt=central, states=[central, periph])

[odes]
  d/dt(central) = -(CL / V1) * central - (Q / V1) * central + (Q / V2) * periph
  d/dt(periph)  = (Q / V1) * central - (Q / V2) * periph

[diffusion]
  central ~ 400

[error_model]
  DV ~ proportional(PROP)
"#;

/// The covariance Hessian of a covariate-NN subject is assembled in the networks' output
/// basis (`NnReduction`: 1 + 3 + 2 directions here against 21 weights), and must equal the
/// one assembled along every packed axis, under each objective — on the closed form and on
/// an SDE, whose EKF variance plan takes the outputs as slots too. Dropping the weights' own
/// curvature (`Σ ∂Ψ/∂z_k ∇²z_k`) or the η part of a weight's direction moves the weight
/// block, and on the SDE so does reading the variance plan's σ slots as outputs or its
/// outputs as σ; the counter shows the reduction ran, the switch that it stood aside.
#[cfg(feature = "nn")]
#[test]
fn covariate_nn_reduced_hessian_equals_the_full_one() {
    for src in [COVARIATE_NN_IV, COVARIATE_NN_SDE] {
        let (model, s) = covariate_nn_fixture_from(src);
        assert_eq!(model.is_sde(), src == COVARIATE_NN_SDE);
        reduced_hessian_equals_the_full_one(&model, &s);
    }
}

#[cfg(feature = "nn")]
fn reduced_hessian_equals_the_full_one(model: &CompiledModel, s: &Subject) {
    use super::hessian::{NN_REDUCED_HITS, NN_REDUCE_OFF};
    let sde = if model.is_sde() { "SDE " } else { "" };
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
            "{sde}{name}: the switch must turn the reduction off"
        );
        assert_eq!(hits, 1, "{sde}{name}: the reduction did not run");
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
        eprintln!("covariate NN {sde}{name}: reduced vs full Hessian {worst:.2e}");
        assert!(worst < 1e-9, "{name}: reduced vs full Hessian {worst:.2e}");
    }
}

/// The exact anchor through a `tanh` network: its taped-Hessian sweep differentiates the
/// activation's `φ″` (`lift3`), and the covariance step's nested dual its `φ‴`
/// (`lift4`). With either order missing the sweep records `NaN` and the subject
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
    let w = super::mode_solve(&h, &b, &stationary, || None).expect("nonsingular");
    let r = &h * &w - &b;
    assert!(r.norm() < 1e-14, "residual {:.2e}", r.norm());
    // A strict mode keeps the Cholesky solve.
    let spd = DMatrix::from_row_slice(2, 2, &[2.0, 0.5, 0.5, 1.0]);
    let w = super::mode_solve(&spd, &b, &stationary, || None).expect("definite");
    assert!((&spd * &w - &b).norm() < 1e-14);
    // Singular at a stationary point: the pseudo-inverse response, not a non-finite `w`.
    // `H = [[1,1],[1,1]]` has eigenpair (2, (1,1)/sqrt 2), so `H+ b = (1,1)(b0+b1)/4`.
    let sing = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]);
    let w = super::mode_solve(&sing, &b, &stationary, || None).expect("pseudo-inverse");
    assert!(
        (w[0] + 0.5).abs() < 1e-14 && (w[1] + 0.5).abs() < 1e-14,
        "{w}"
    );
    // Nearly singular (rcond 1e-14): the plain solve is rounding, the range part is kept.
    let near = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0 - 1e-14]);
    let w = super::mode_solve(&near, &b, &stationary, || None).expect("pseudo-inverse");
    assert!(w.norm() < 1.0, "the null direction is not amplified: {w}");
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
    assert!(super::mode_solve(&h, &b, &[0.0, 5e-5], || None).is_some());
    assert!(super::mode_solve(&h, &b, &[1e-2, 0.0], || None).is_none());
    assert!(super::mode_solve(&h, &b, &[f64::NAN, 0.0], || None).is_none());
    let spd = DMatrix::from_row_slice(2, 2, &[2.0, 0.5, 0.5, 1.0]);
    assert!(super::mode_solve(&spd, &b, &[1e-2, 0.0], || None).is_some());
}

/// #2557: `MODE_STATIONARY_TOL` is a gradient gate of its own, and a point the inner
/// search's certificate accepts can fail it. On a direction of curvature 1e4,
/// `FallbackCertifier` certifies `η = (2e-8, 0)` by the Newton-step arm (`‖H⁻¹g‖ = 2e-8 <
/// tol`) without moving it, at `‖g‖∞ = 2e-4`, twice the gate. That gradient on an indefinite
/// `H` declines. Both halves are asserted, so the constant's doc (the certificate does not
/// imply the gate) cannot silently go false in either direction: a certificate that starts
/// bounding the gradient fails the first half, and a gate that starts accepting certified
/// points fails the second. The control, the same `H` at a gradient under the gate, is taken.
#[test]
fn a_certified_point_can_fail_the_indefinite_stationarity_gate() {
    use crate::estimation::inner_optimizer::FallbackCertifier;
    let c = 1e4;
    let obj = |x: &[f64]| 0.5 * (c * x[0] * x[0] + x[1] * x[1]);
    let grad = |x: &[f64]| vec![c * x[0], x[1]];
    let never = |_: &[f64]| false;
    let cert = FallbackCertifier {
        grad: &grad,
        precond: None,
        stop_precond: None,
        stall_at: &never,
        n: 2,
        max_iter: 50,
        tol: 1e-5,
    };
    let start = vec![2e-8, 0.0];
    let mut x = start.clone();
    assert!(
        cert.certify(&obj, &mut x),
        "the Newton-step arm certifies it"
    );
    assert_eq!(x, start, "certified in place, not polished");
    let g = grad(&x);
    let g_inf = g.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(
        g_inf > super::MODE_STATIONARY_TOL,
        "the certified point is above the gate: {g_inf:.2e}"
    );
    let indefinite = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
    let b = DVector::from_column_slice(&[1.0, -3.0]);
    assert!(
        super::mode_solve(&indefinite, &b, &g, || None).is_none(),
        "a certified point above the gate declines"
    );
    let under: Vec<f64> = g.iter().map(|v| v * 0.4).collect();
    assert!(super::mode_solve(&indefinite, &b, &under, || None).is_some());
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
            Some(g.to_vec())
        }
    };
    // The recording reads 1e-2, the search 5e-5: taken, by the search's gradient.
    let w = super::mode_solve(&indefinite, &b, &[1e-2, 0.0], inner([5e-5, 0.0]));
    assert!(w.is_some());
    assert_eq!(calls.get(), 1);
    // Both non-stationary: declined.
    assert!(super::mode_solve(&indefinite, &b, &[1e-2, 0.0], inner([1e-2, 0.0])).is_none());
    // No inner gradient: declined.
    assert!(super::mode_solve(&indefinite, &b, &[1e-2, 0.0], || None).is_none());
    // A definite Hessian and a stationary recording never ask.
    calls.set(0);
    assert!(super::mode_solve(&spd, &b, &[1e-2, 0.0], inner([1.0, 1.0])).is_some());
    assert!(super::mode_solve(&indefinite, &b, &[0.0, 5e-5], inner([1.0, 1.0])).is_some());
    // A singular `H` is judged by the same gate (its pseudo-inverse response is the limit at
    // a stationary point only): off the mode it declines to the polish.
    let singular = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]);
    assert!(super::mode_solve(&singular, &b, &[1e-2, 0.0], || None).is_none());
    assert!(super::mode_solve(&singular, &b, &[1e-2, 0.0], inner([5e-5, 0.0])).is_some());
    assert!(super::mode_solve(&singular, &b, &[0.0, 0.0], || None).is_some());
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
        super::stacked_omega(&model, &s, &params, &params, 3).chol,
    )
    .unwrap();
    let jets = canonical_jets1(&model, &s, None, &params.theta, None, &[0.0; 3]).unwrap();
    let fpop: Vec<f64> = (0..jets.rows()).map(|j| jets.v(j)).collect();
    for frozen in [None, Some(fpop.as_slice())] {
        let got =
            super::inner_search_gradient(&model, &s, None, &params.theta, &eta, &ctx, &pv, frozen)
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
    // inner problem (`subject_adjoint_eta_dx` takes the fit's `interaction`).
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
        // in the removed `population_eta_dx`) misses it by 6.1e-2, so the FOCE bound straddles that defect.
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
    let gated = objective_gaps_for(&model, &s, &p, eta.as_slice(), false);
    FORCE_WOODBURY.with(|c| c.set(true));
    let ungated = objective_gaps_for(&model, &s, &p, eta.as_slice(), false);
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

/// Arbitrary but live prediction jets for `n_obs` rows over `n` random effects: `f` stays
/// positive (a proportional variance is defined), `G` and `Q` carry no zero entries.
#[allow(clippy::type_complexity)]
fn live_jets(n_obs: usize, n: usize) -> (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let f = (0..n_obs)
        .map(|j| 0.6 + 0.4 * (1.7 * j as f64).sin().abs())
        .collect();
    let g = (0..n_obs)
        .map(|j| {
            (0..n)
                .map(|k| 0.3 + 0.4 * (1.3 * j as f64 + 0.7 * k as f64).sin())
                .collect()
        })
        .collect();
    let q = (0..n_obs)
        .map(|j| {
            (0..n * (n + 1) / 2)
                .map(|t| 0.25 + 0.2 * (0.9 * j as f64 + 1.1 * t as f64).cos())
                .collect()
        })
        .collect();
    (f, g, q)
}

/// The worst relative entry difference between the closed form and the generic definition
/// (`inner_l` over mixed second-order duals) on the same jets.
fn closed_vs_generic(
    model: &CompiledModel,
    s: &Subject,
    eta: &[f64],
    f: &[f64],
    g: &[Vec<f64>],
    q: &[Vec<f64>],
) -> f64 {
    use crate::sens::focei_kernel::{inner_hessian_generic, inner_hessian_through_predictions};
    let params = &model.default_params;
    let ctx = crate::sens::focei_kernel::LikCtx::new(model, s, params);
    let p = crate::sens::focei_kernel::ParT::of(model, s, params);
    let closed = inner_hessian_through_predictions(&ctx, f, g, q, &p).expect("closed form");
    let generic = inner_hessian_generic(&ctx, f, g, q, eta, &p).expect("generic");
    let mut worst = 0.0f64;
    for a in 0..eta.len() {
        for b in 0..eta.len() {
            let (c, d) = (closed[a][b], generic[a][b]);
            assert!(c.is_finite() && d.is_finite(), "({a}, {b}): {c} vs {d}");
            worst = worst.max((c - d).abs() / (1.0 + d.abs()));
        }
    }
    worst
}

/// The closed form declines, in every build, when its row sensitivities are not as wide as
/// Ω⁻¹ (a `ParT` from `map_rows` carries no Ω, so its width would be 0 and the Hessian
/// silently empty); a `ParT` with Ω answers. Mutation: drop the width check in
/// `inner_hessian_through_predictions` → the no-Ω call returns `Some` of an empty matrix.
#[test]
fn the_inner_hessian_through_predictions_declines_a_width_it_cannot_assemble() {
    use crate::sens::focei_kernel::{inner_hessian_through_predictions, LikCtx, ParT};
    let model = parse_model_string(ONE_CPT_ORAL).unwrap();
    let s = subject(
        &model,
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &TIMES,
        &[0.1, -0.05, 0.2],
    );
    let params = &model.default_params;
    let ctx = LikCtx::new(&model, &s, params);
    let (f, g, q) = live_jets(TIMES.len(), 3);
    let full = ParT::of(&model, &s, params);
    let h = inner_hessian_through_predictions(&ctx, &f, &g, &q, &full).expect("with Ω");
    assert_eq!((h.len(), h[0].len()), (3, 3));
    let rows = full.map_rows(|x| x);
    assert!(
        inner_hessian_through_predictions(&ctx, &f, &g, &q, &rows).is_none(),
        "a ParT without Ω declines"
    );
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
    let (f3, g3, q3) = live_jets(TIMES.len(), 3);
    let (f2, g2, q2) = live_jets(TIMES.len(), 2);
    let cases = [
        (
            "proportional",
            closed_vs_generic(
                &one,
                &subject(&one, dose(), &TIMES, &[0.1, -0.05, 0.2]),
                &[0.1, -0.05, 0.2],
                &f3,
                &g3,
                &q3,
            ),
        ),
        (
            "combined, block Ω",
            closed_vs_generic(
                &combined,
                &subject(&combined, dose(), &TIMES, &[0.1, -0.05]),
                &[0.1, -0.05],
                &f2,
                &g2,
                &q2,
            ),
        ),
        (
            "power",
            closed_vs_generic(
                &power,
                &subject(&power, dose(), &TIMES, &[0.1, -0.05, 0.2]),
                &[0.1, -0.05, 0.2],
                &f3,
                &g3,
                &q3,
            ),
        ),
        (
            "M3-censored rows",
            closed_vs_generic(
                &m3,
                &censor(subject(&m3, dose(), &TIMES, &[0.1, -0.05, 0.2]), &[0, 7, 8]),
                &[0.1, -0.05, 0.2],
                &f3,
                &g3,
                &q3,
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
    // The covariance step's structured kernel forms the exact anchor from its row aggregates
    // (#2980) and calls neither routine; with it off, the recorded kernel takes the closed form.
    // Never the generic route: the straddle shows the count is the route's, not a constant.
    let cov_route = |off: bool| {
        super::hessian::st::STRUCT_OFF.with(|c| c.set(off));
        let r = hessian_route(&s, &|| {
            subject_adjoint_cov_hessian(&one, &s, &params, &x, eta.as_slice(), laplace)
                .expect("served");
        });
        super::hessian::st::STRUCT_OFF.with(|c| c.set(false));
        r
    };
    assert_eq!(
        cov_route(false),
        (0, 0),
        "Laplace covariance step, structured"
    );
    let r = cov_route(true);
    assert!(
        r.0 >= 1 && r.1 == 0,
        "Laplace covariance step, recorded: {r:?}"
    );
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
    fn check(label: &str, model: &CompiledModel, s: &Subject, eta: &[f64]) {
        let params = &model.default_params;
        let ctx = crate::sens::focei_kernel::LikCtx::new(model, s, params);
        let p = crate::sens::focei_kernel::ParT::of(model, s, params);
        let (f, g, q) = live_jets(s.observations.len(), eta.len());
        let got = std::cell::RefCell::new(None);
        let r = hessian_route(s, &|| {
            *got.borrow_mut() = exact_inner_hessian_kernel(&ctx, &f, &g, &q, eta, &p)
        });
        let got = got.into_inner();
        assert_eq!(r, (0, 1), "{label}: route {r:?}");
        assert_eq!(
            got,
            inner_hessian_generic(&ctx, &f, &g, &q, eta, &p),
            "{label}"
        );
    }
    let frem = parse_model_string(FREM_MODEL).unwrap();
    let mut fs = subject_shell(dose(), &[1.0, 6.0, 24.0, 0.0]);
    fs.fremtype = vec![0, 0, 0, 100];
    fs.observations = vec![0.8, 0.6, 0.3, 75.0];
    check("a FREM row", &frem, &fs, &[0.1, -0.05, 0.15, 0.2]);
    let ruv = parse_model_string(TWO_CPT_ODE_RUV).unwrap();
    check(
        "the RUV-IIV scale",
        &ruv,
        &subject(&ruv, dose(), &TIMES, &[0.1, -0.05, 0.1]),
        &[0.1, -0.05, 0.1],
    );
    let dense = parse_model_string(&dense_src("", true)).unwrap();
    check(
        "a dense residual covariance",
        &dense,
        &dense_subject(&dense),
        &[0.1, -0.15],
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
    let (mut f, g, q) = live_jets(TIMES.len(), 3);
    f[4] = f64::INFINITY;
    let got = std::cell::RefCell::new(Some(Vec::new()));
    let r = hessian_route(&s, &|| {
        *got.borrow_mut() = exact_inner_hessian_kernel(&ctx, &f, &g, &q, &[0.1, -0.05, 0.2], &p)
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
/// tangent (`Dual2<1>`) — the same recorder, differentiated forwards, sharing no
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
            let th: Vec<Dual2<1>> = theta
                .iter()
                .enumerate()
                .map(|(k, &v)| {
                    if k == m {
                        Dual2::var(v, 0)
                    } else {
                        Dual2::constant(v)
                    }
                })
                .collect();
            let et: Vec<Dual2<1>> = eta.iter().map(|&e| Dual2::constant(e)).collect();
            record(model, s, None, theta, eta, &ThetaLeaves::theta(&th), &et)
                .expect("in scope")
                .iter()
                .map(|q| q.grad[0])
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
            bolus.clone(),
            vec![0.1, -0.05, 0.08, -0.03],
        ),
        // An SDE model (#3004): `want_jac` without σ is the predictions' own `∂f_j/∂θ`, served
        // rather than declined to a forward difference per θ.
        (
            "warfarin_sde",
            parse_model_string(
                &std::fs::read_to_string(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/examples/warfarin_sde.ferx"
                ))
                .unwrap(),
            )
            .unwrap(),
            bolus,
            vec![0.1, -0.05, 0.08],
        ),
    ];
    let mut worst_all = 0.0f64;
    // Both routes of the Jacobian: the forward tangents (one pass, a direction per θ, taken
    // where there are no more θ than observations) and one reverse seed per observation, each
    // against the independent `Dual2` forward oracle.
    for (force_reverse, (name, model, doses, eta)) in [false, true]
        .into_iter()
        .flat_map(|r| cases.iter().map(move |c| (r, c)))
    {
        JAC_REVERSE_FORCE.with(|f| f.set(force_reverse));
        let (doses, mut worst) = (doses.clone(), 0.0f64);
        let width = model.n_eta + model.n_kappa;
        let mut eta = eta[..width.min(eta.len())].to_vec();
        eta.resize(width, 0.05);
        let s = subject(model, doses, &TIMES, &eta[..model.n_eta]);
        let theta = model.default_params.theta.clone();
        let weights: Vec<f64> = (0..s.observations.len())
            .map(|j| (0.7 * j as f64 + 0.3).sin())
            .collect();
        let tan0 = TANGENT_JACS.with(|c| c.get());
        let x = fixed_eta_theta_adjoint(model, &s, &theta, &eta, true, &weights)
            .unwrap_or_else(|| panic!("{name}: out of the recorders' scope"));
        let took_tangents = TANGENT_JACS.with(|c| c.get()) - tan0;
        assert_eq!(
            took_tangents,
            usize::from(!force_reverse && theta.len() <= s.observations.len()),
            "{name} (force_reverse {force_reverse}): the route taken ({} θ, {} observations)",
            theta.len(),
            s.observations.len()
        );
        let n_axis = theta.len();
        let fwd = forward_theta_jacobian(model, &s, &theta, &eta, n_axis);
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
        // `additive_four_eta` is a linear ODE served by the spectral path (#2706 follow-up),
        // whose propagation is a sum of modes: measured 1.9e-14 there; 1e-13 keeps ~5×.
        let bound = if name.contains("rodas") {
            1e-9
        } else if *name == "additive_four_eta" {
            1e-13
        } else if *name == "warfarin_sde" {
            1e-12
        } else {
            1e-14
        };
        assert!(worst < bound, "{name}: worst relative difference {worst:e}");
        worst_all = worst_all.max(worst);
    }
    JAC_REVERSE_FORCE.with(|f| f.set(false));
    eprintln!("fixed-η θ adjoint vs forward tangents: worst relative difference {worst_all:e}");
}

/// The Jacobian over a subset of the θ coordinates (the free ones of an M-step that holds the
/// rest pinned) is the full Jacobian's columns, by forward tangents, and the gradient the
/// subset's contraction; routing is by direction count against observation count.
#[test]
fn the_tangent_jacobian_over_a_column_subset_is_the_full_jacobians_columns() {
    let model = parse_model_string(TWO_CPT_INF_COMBINED_BLOCK).unwrap();
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0),
        DoseEvent::new(8.0, 100.0, 1, 50.0, false, 0.0),
    ];
    let eta = vec![0.1, -0.05, 0.08];
    let s = subject(&model, doses, &TIMES, &eta);
    let theta = model.default_params.theta.clone();
    let n_theta = theta.len();
    let weights: Vec<f64> = (0..s.observations.len())
        .map(|j| (0.7 * j as f64 + 0.3).sin())
        .collect();
    let full = fixed_eta_theta_adjoint(&model, &s, &theta, &eta, true, &weights).unwrap();
    let cols = vec![n_theta - 1, 1];
    let tan0 = TANGENT_JACS.with(|c| c.get());
    let sub =
        fixed_eta_theta_adjoint_cols(&model, &s, &theta, &eta, Some(&cols), &weights).unwrap();
    assert_eq!(
        TANGENT_JACS.with(|c| c.get()) - tan0,
        1,
        "the subset took tangents"
    );
    let (fj, sj) = (full.jac.as_ref().unwrap(), sub.jac.as_ref().unwrap());
    let mut live = 0;
    for j in 0..s.observations.len() {
        for m in 0..n_theta {
            let (a, b) = (fj[j * n_theta + m], sj[j * n_theta + m]);
            if cols.contains(&m) {
                assert!(a.is_finite() && b.is_finite());
                assert!(
                    (a - b).abs() <= 1e-12 * (1.0 + a.abs()),
                    "∂f{j}/∂θ{m}: {a} vs {b}"
                );
                live += (a != 0.0) as usize;
            } else {
                assert_eq!(b, 0.0, "an unwanted column is not computed");
            }
        }
    }
    assert!(live > s.observations.len(), "the columns are live: {live}");
    for &m in &cols {
        let g = (full.grad[m] - sub.grad[m]).abs();
        assert!(g <= 1e-12 * (1.0 + full.grad[m].abs()), "grad θ{m}: {g:e}");
    }
    // Routing: a direction per column against a seed per observation.
    assert!(tangent_jac_wins(1, 9) && tangent_jac_wins(9, 9));
    assert!(!tangent_jac_wins(0, 9) && !tangent_jac_wins(10, 9) && !tangent_jac_wins(49, 99));
    JAC_REVERSE_FORCE.with(|f| f.set(true));
    assert!(!tangent_jac_wins(1, 9));
    JAC_REVERSE_FORCE.with(|f| f.set(false));
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

/// The exact outer gradient on binary, CTMM and DTMM endpoints — endpoint-only and beside a PK
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
    // The discrete-time Markov endpoint (#2473) on the same chain: per-record transition
    // probabilities, a stay step and a `TIME` term, through the discrete seam's term.
    let dtmm = parse_model_string(
        "[parameters]\n  theta A(-0.8, -6.0, 3.0)\n  theta B(-1.3, -6.0, 3.0)\n  \
         omega ETA_Q ~ 0.1\n[markov_model]\n  type   = dtmm\n  cmt    = 5\n  \
         states = [awake=0, asleep=1]\n  transition awake  -> asleep = inv_logit(A + ETA_Q)\n  \
         transition asleep -> awake  = inv_logit(B - 0.5 * ETA_Q + 0.1 * TIME)\n",
    )
    .unwrap();
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
        ("dtmm", &dtmm, &ctmm_s),
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

/// A closed-form joint model (PK + a binary endpoint sharing `ETA_CL`) and a subject whose
/// every observation is on the endpoint's CMT, so `subject_feeds_analytical_pk` is false:
/// the data reader routes such rows to `obs_records` and leaves `observations` empty, so the
/// recorders have no Gaussian row to record and the subject is served by the term alone.
/// Against the reconverged oracle, with the mixed subject of the same model as the control.
#[cfg(all(feature = "survival", feature = "markov"))]
#[test]
fn a_subject_with_only_endpoint_observations_on_a_closed_form_joint_model_is_served() {
    use crate::types::ObsRecord;
    let model = parse_model_string(
        "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  theta TVV(30.0, 1.0, 500.0)\n  \
         theta TVKA(1.2, 0.05, 20.0)\n  theta TH0(-0.3, -10.0, 10.0)\n  \
         omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.06\n  sigma PROP_ERR ~ 0.1 (sd)\n\
         [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n  \
         KA = TVKA\n[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n\
         [error_model]\n  DV ~ proportional(PROP_ERR)\n\
         [binary_model]\n  cmt   = 3\n  logit = TH0 + 2.0 * ETA_CL\n",
    )
    .unwrap();
    assert!(model.ode_spec.is_none() && !model.is_algebraic() && model.has_non_gaussian());
    let records = [(0.0, 0), (1.0, 1), (2.0, 1), (4.0, 0), (6.0, 1)]
        .iter()
        .map(|&(time, state)| ObsRecord::DiscreteState {
            time,
            raw_time: time,
            state,
            cmt: 3,
        })
        .collect::<Vec<_>>();
    let mut only = subject(
        &model,
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &[],
        &[0.0, 0.0],
    );
    only.obs_records = records.clone();
    assert!(only.observations.is_empty() && only.obs_cmts.is_empty());
    assert!(
        !crate::pk::subject_feeds_analytical_pk(&model, &only),
        "the premise: the predictor feeds nothing for this subject"
    );
    let mut mixed = subject(
        &model,
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &[1.0, 4.0, 12.0],
        &[0.1, -0.05],
    );
    mixed.obs_records = records;
    assert!(crate::pk::subject_feeds_analytical_pk(&model, &mixed));
    assert!(outer_model_in_scope(&model));
    for (tag, s) in [("endpoint observations only", &only), ("mixed", &mixed)] {
        for interaction in [true, false] {
            let w = fd_oracle(&model, s, interaction);
            eprintln!("{tag} interaction={interaction}: worst {w:.2e}");
            assert!(w < 5e-7, "{tag} interaction={interaction}: {w:.2e}");
        }
    }
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

/// A quadrature node whose non-Gaussian endpoint term is not usable (`NaN`, or beyond the
/// sentinel) while its Gaussian term `l` is finite is left out of the sum by the objective
/// (`agq::agq_terms` scores `l + q`), and the adjoint serves the subject over the same nodes:
/// mixed with PK rows and with the endpoint alone, against the reconverged-FD oracle. Two
/// predictors on `ETA_CL`: `exp(200*ETA_CL)` is `~1e30` at the upper nodes (finite, over
/// `1e20`, small below), and `16*(0.008 - ETA_CL^3)^0.5` is `NaN` there (the parser's `^` does
/// not clamp). Measured: the subject is served *without* the live-node filter reaching these
/// nodes (the first kernel run forms, since `q` enters the reverse as numbers and `0 * NaN` never
/// arises through it), and swapping the filter's `l + q` for `l` alone leaves this test green, so
/// it pins the parity of the served gradient, not the filter's `q` term (the review's reading of
/// that term as a defect has no observable effect; see the filter's comment).
#[cfg(all(feature = "survival", feature = "markov"))]
#[test]
fn an_agq_node_with_an_unusable_endpoint_term_is_served_exactly() {
    use crate::types::ObsRecord;
    let mut worst = 0.0f64;
    for (case, logit, nan) in [
        ("sentinel", "exp(200.0 * ETA_CL)", false),
        ("nan", "16.0 * (0.008 - ETA_CL ^ 3) ^ 0.5", true),
    ] {
        let model = parse_model_string(&format!(
            "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  theta TVV(30.0, 1.0, 500.0)\n  \
             theta TVKA(1.2, 0.05, 20.0)\n  theta TH0(-0.3, -10.0, 10.0)\n  \
             omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.06\n  sigma PROP_ERR ~ 0.1 (sd)\n\
             [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n  \
             KA = TVKA\n[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n\
             [error_model]\n  DV ~ proportional(PROP_ERR)\n\
             [binary_model]\n  cmt   = 3\n  logit = TH0 + 2.0 * ETA_CL + {logit}\n"
        ))
        .unwrap();
        let records: Vec<ObsRecord> = [(0.0, 0), (1.0, 0), (2.0, 1), (4.0, 0)]
            .iter()
            .map(|&(time, state)| ObsRecord::DiscreteState {
                time,
                raw_time: time,
                state,
                cmt: 3,
            })
            .collect();
        let mut only = subject(
            &model,
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            &[],
            &[0.0, 0.0],
        );
        only.obs_records = records.clone();
        assert!(only.observations.is_empty());
        let mut mixed = subject(
            &model,
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            &[1.0, 4.0, 12.0],
            &[0.1, -0.05],
        );
        mixed.obs_records = records;
        assert!(crate::pk::subject_feeds_analytical_pk(&model, &mixed));
        for (tag, s) in [("endpoint only", &only), ("mixed", &mixed)] {
            // The premise: the term is usable at the mode and below, and not at the upper node
            // (the nodes sit at the mode, ±~0.5 in `ETA_CL`).
            let params = moved_params(&model);
            let eta = ebe_for(&model, s, &params, 1e-12, true);
            let q: Vec<f64> = [-0.5, 0.0, 0.5]
                .iter()
                .map(|d| {
                    let e = [eta[0] + d, eta[1]];
                    super::ng_value_grad(&model, s, &params.theta, &e).map_or(f64::NAN, |r| r.0)
                })
                .collect();
            let unusable = |v: f64| !crate::stats::likelihood::is_usable_subject_nll(v);
            assert!(
                !unusable(q[0]) && !unusable(q[1]) && unusable(q[2]) && q[2].is_nan() == nan,
                "{case}/{tag}: the premise, a usable term below and an unusable one at the upper \
                 node: {q:?}"
            );
            for (n_agq, exact) in [(3, true), (3, false)] {
                let w = fd_oracle_agq(&model, s, n_agq, exact);
                eprintln!("{case}/{tag} n_agq={n_agq} exact={exact}: worst {w:.2e}");
                assert!(w.is_finite());
                worst = worst.max(w);
            }
        }
    }
    // Measured on Linux: 2.5e-6 on the `sentinel` mixed subject (the `exp(200*ETA_CL)` slope
    // makes the Richardson oracle's own error that large on the small TVCL component).
    assert!(worst < 1e-5, "{worst:.2e}");
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
    // The discrete-time Markov endpoint (#2473) on the same chain: per-record transition
    // probabilities, a stay step and a `TIME` term, through the discrete seam's term.
    let dtmm = parse_model_string(
        "[parameters]\n  theta A(-0.8, -6.0, 3.0)\n  theta B(-1.3, -6.0, 3.0)\n  \
         omega ETA_Q ~ 0.1\n[markov_model]\n  type   = dtmm\n  cmt    = 5\n  \
         states = [awake=0, asleep=1]\n  transition awake  -> asleep = inv_logit(A + ETA_Q)\n  \
         transition asleep -> awake  = inv_logit(B - 0.5 * ETA_Q + 0.1 * TIME)\n",
    )
    .unwrap();
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
    cases.push(("dtmm".into(), dtmm, ctmm_s.clone()));
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

/// #2549: the mode response fused into the outer gradient's own recording
/// ([`subject_adjoint_gradient_with_eta_dx_at`]) is the standalone exact one
/// ([`subject_adjoint_eta_dx`], itself pinned to the re-solved EBE above), on a Gaussian
/// five-effect subject (width-6 bucket, so one padded lane and two sweep chunks) and on
/// non-Gaussian subjects, under both interaction settings — and the gradient it rides on is
/// unchanged bit for bit.
#[test]
fn the_fused_mode_response_is_the_standalone_one() {
    let root = env!("CARGO_MANIFEST_DIR");
    let five = parse_model_string(&one_cpt_oral_n_eta(5)).unwrap();
    let s5 = subject(
        &five,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &TIMES,
        &[0.1, -0.05, 0.2, 0.03, -0.02],
    );
    #[allow(unused_mut)]
    let mut cases: Vec<(String, CompiledModel, Subject)> = vec![("five".into(), five, s5)];
    #[cfg(feature = "survival")]
    {
        let mut endpoint_cases = vec![
            ("examples/pktte_joint.ferx", "data/pktte_joint.csv"),
            ("examples/tte_exponential.ferx", "data/tte_exponential.csv"),
            ("examples/binary_logistic.ferx", "data/binary_logistic.csv"),
            (
                "examples/tte_competing_risks.ferx",
                "data/tte_competing_risks.csv",
            ),
        ];
        if cfg!(feature = "markov") {
            endpoint_cases.push(("examples/ctmm_pd_2state.ferx", "data/ctmm_pd_2state.csv"));
        }
        for (m, d) in endpoint_cases {
            let prep =
                crate::api::prepare_run(&format!("{root}/{m}"), Some(&format!("{root}/{d}")))
                    .unwrap();
            let s = prep
                .population
                .subjects
                .iter()
                .find(|s| !s.obs_records.is_empty())
                .unwrap()
                .clone();
            cases.push((m.into(), prep.parsed.model, s));
        }
    }
    // A `[covariate_nn]` model: the network weights reach the response through the
    // per-lane `add_nn_weight_adjoint`, as they reach the gradient through Ψ's.
    #[cfg(feature = "nn")]
    {
        let prep = crate::api::prepare_run(
            &format!("{root}/examples/warfarin_dcm.ferx"),
            Some(&format!("{root}/data/warfarin_dcm.csv")),
        )
        .unwrap();
        let s = prep.population.subjects[0].clone();
        cases.push(("warfarin_dcm".into(), prep.parsed.model, s));
    }
    let _ = root;
    let mut worst = 0.0f64;
    let mut live = 0.0f64;
    for (name, model, s) in &cases {
        let params = moved_params(model);
        let x = pack_params(&params);
        for interaction in [true, false] {
            // The NN example is a FOCEI model; its FOCE adjoint declines the subject. A model
            // without random effects has no mode to respond.
            if (name == "warfarin_dcm" && !interaction) || model.n_eta == 0 {
                continue;
            }
            let eta = ebe_for(model, s, &params, 1e-12, interaction);
            let objective = if interaction {
                Objective::Focei
            } else {
                Objective::Foce
            };
            let (plain, (fused_g, fused), want) = std::thread::scope(|sc| {
                std::thread::Builder::new()
                    .stack_size(crate::FIT_RAYON_STACK_SIZE)
                    .spawn_scoped(sc, || {
                        let g = |_: ()| {
                            subject_adjoint_gradient(
                                model,
                                s,
                                &params,
                                &x,
                                eta.as_slice(),
                                objective,
                            )
                        };
                        let plain = g(());
                        let pt = PackedPoint::new(&params, &x).unwrap();
                        let fused = subject_adjoint_gradient_with_eta_dx_at(
                            model,
                            s,
                            &pt,
                            eta.as_slice(),
                            interaction,
                        );
                        let want = subject_adjoint_eta_dx(
                            model,
                            s,
                            &params,
                            &x,
                            eta.as_slice(),
                            interaction,
                        );
                        (plain, fused, want)
                    })
                    .unwrap()
                    .join()
                    .unwrap()
            });
            let (plain, fused_g) = (plain.unwrap(), fused_g.unwrap());
            for (a, b) in plain.iter().zip(&fused_g) {
                assert_eq!(a.to_bits(), b.to_bits(), "{name}: the gradient moved");
            }
            let got = fused.unwrap_or_else(|| panic!("{name} {interaction}: no fused response"));
            let want = want.unwrap();
            assert_eq!(got.len(), want.len(), "{name}");
            for (gk, wk) in got.iter().zip(&want) {
                assert_eq!(gk.len(), wk.len(), "{name}");
                for (a, b) in gk.iter().zip(wk.iter()) {
                    assert!(a.is_finite() && b.is_finite(), "{name}");
                    worst = worst.max((a - b).abs() / (1.0 + b.abs()));
                    live = live.max(b.abs());
                }
            }
            eprintln!("{name} interaction={interaction}: fused vs standalone {worst:.2e}");
        }
    }
    assert!(live > 1e-2, "the mode does not move: the test sees nothing");
    // Measured 1.03e-15 (Linux x86_64): the two are one formula, swept in a different
    // order; 1e-12 leaves three decades for another platform's rounding.
    assert!(
        worst < 1e-12,
        "fused vs standalone mode response {worst:.2e}"
    );
}

/// #2549: the fused mode response `dη̂/dx = −H⁻¹·∂(∇_η l)/∂x` on a `[covariate_nn]` model,
/// against the same formula with both factors taken by mixed central differences of the
/// conditional inner objective `l(η; x)` (`individual_nll`) at the same fixed `η`. No EBE is
/// re-solved, so the oracle sees the network-weight coordinates smoothly — re-solved EBEs on
/// this model jump by orders of magnitude between step sizes — and shares no derivative code
/// with the response.
#[cfg(feature = "nn")]
#[test]
fn the_fused_mode_response_matches_fd_of_the_inner_objective_on_a_covariate_nn_model() {
    use crate::estimation::parameterization::unpack_params;
    let root = env!("CARGO_MANIFEST_DIR");
    let prep = crate::api::prepare_run(
        &format!("{root}/examples/warfarin_dcm.ferx"),
        Some(&format!("{root}/data/warfarin_dcm.csv")),
    )
    .unwrap();
    let model = &prep.parsed.model;
    let params = moved_params(model);
    let x = pack_params(&params);
    let n = model.n_eta;
    let s = &prep.population.subjects[0];
    let eta = ebe_for(model, s, &params, 1e-12, true);
    let pt = PackedPoint::new(&params, &x).unwrap();
    let (_, dx) = std::thread::scope(|sc| {
        std::thread::Builder::new()
            .stack_size(crate::FIT_RAYON_STACK_SIZE)
            .spawn_scoped(sc, || {
                subject_adjoint_gradient_with_eta_dx_at(model, s, &pt, eta.as_slice(), true)
            })
            .unwrap()
            .join()
            .unwrap()
    });
    let dx = dx.expect("an NN subject is served");
    let l = |xv: &[f64], e: &[f64]| {
        let p = unpack_params(xv, &params);
        crate::stats::likelihood::individual_nll(model, s, &p.theta, e, &p.omega, &p.sigma.values)
    };
    let de = 1e-4;
    let shift = |e: &[f64], j: usize, d: f64| {
        let mut v = e.to_vec();
        v[j] += d;
        v
    };
    let h = nalgebra::DMatrix::from_fn(n, n, |a, b| {
        let e = eta.as_slice();
        (l(&x, &shift(&shift(e, a, de), b, de))
            - l(&x, &shift(&shift(e, a, de), b, -de))
            - l(&x, &shift(&shift(e, a, -de), b, de))
            + l(&x, &shift(&shift(e, a, -de), b, -de)))
            / (4.0 * de * de)
    });
    let hinv = h.clone().try_inverse().expect("H");
    let mut worst = 0.0f64;
    for k in 0..x.len() {
        let hx = 1e-5 * (1.0 + x[k].abs());
        let g = nalgebra::DVector::from_fn(n, |j, _| {
            let e = eta.as_slice();
            let xp = shift(&x, k, hx);
            let xm = shift(&x, k, -hx);
            (l(&xp, &shift(e, j, de)) - l(&xp, &shift(e, j, -de)) - l(&xm, &shift(e, j, de))
                + l(&xm, &shift(e, j, -de)))
                / (4.0 * de * hx)
        });
        let want = -(&hinv * g);
        for j in 0..n {
            // Finite before the fold: `f64::max` drops a `NaN`, so a response returning one
            // would otherwise pass on the strength of the finite entries.
            assert!(
                dx[k][j].is_finite() && want[j].is_finite(),
                "x[{k}] η[{j}]: response {} vs FD {}",
                dx[k][j],
                want[j]
            );
            worst = worst.max((dx[k][j] - want[j]).abs() / (1.0 + want[j].abs()));
        }
    }
    eprintln!("NN response vs FD of the inner objective: {worst:.2e}");
    // Measured 5.12e-6 (Linux x86_64): the FD side is a nested central difference
    // (`hx = 1e-5`, `de` above) of an objective with ~1e-13 rounding, so its own error is of
    // that order; 1e-4 leaves 20× headroom for another platform's rounding.
    assert!(
        worst < 1e-4,
        "NN mode response vs FD of the inner objective {worst:.2e}"
    );
}

/// A failed mode response leaves the gradient alone (#2549 review r2 #2). The response
/// rows share the gradient's sweeps, and a failing lane fails the fused pass; the pass is
/// then re-run as the gradient alone. Forced here by `FAIL_RESPONSE_LANES`: the fused call
/// returns the plain gradient bit for bit and no response, where without the re-run it
/// returned no gradient at all.
#[test]
fn a_failed_mode_response_keeps_the_gradient() {
    let model = parse_model_string(&one_cpt_oral_n_eta(3)).unwrap();
    let s = subject(
        &model,
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &TIMES,
        &[0.1, -0.05, 0.2],
    );
    let params = moved_params(&model);
    let x = pack_params(&params);
    let eta = ebe_for(&model, &s, &params, 1e-12, true);
    let (plain, fused, fused_resp) = std::thread::scope(|sc| {
        std::thread::Builder::new()
            .stack_size(crate::FIT_RAYON_STACK_SIZE)
            .spawn_scoped(sc, || {
                let plain = subject_adjoint_gradient(
                    &model,
                    &s,
                    &params,
                    &x,
                    eta.as_slice(),
                    Objective::Focei,
                );
                let pt = PackedPoint::new(&params, &x).unwrap();
                // The straddle: without the hook the fused pass carries a response.
                let (_, live) =
                    subject_adjoint_gradient_with_eta_dx_at(&model, &s, &pt, eta.as_slice(), true);
                assert!(live.is_some(), "the fixture's fused pass forms a response");
                super::FAIL_RESPONSE_LANES.with(|f| f.set(true));
                let (g, r) =
                    subject_adjoint_gradient_with_eta_dx_at(&model, &s, &pt, eta.as_slice(), true);
                super::FAIL_RESPONSE_LANES.with(|f| f.set(false));
                (plain, g, r)
            })
            .unwrap()
            .join()
            .unwrap()
    });
    let (plain, fused) = (
        plain.unwrap(),
        fused.expect("the gradient survives the failed response"),
    );
    assert!(
        fused_resp.is_none(),
        "a failed response is reported as none"
    );
    assert_eq!(plain.len(), fused.len());
    for (a, b) in plain.iter().zip(&fused) {
        assert!(a.is_finite(), "the gradient is finite");
        assert_eq!(a.to_bits(), b.to_bits(), "the gradient moved");
    }
}

/// Laplace/AGQ's mode response (#2549) is the FOCEI one: the quadrature is centred on the
/// conditional mode `η̂ = argmin l`, the same inner problem FOCEI solves, so
/// `dη̂/dx = −(∇²_η l)⁻¹ ∂(∇_η l)/∂x` with the exact inner Hessian under either anchor (the
/// FOCEI anchor's `H̃` places the nodes; it is not the mode's derivative). The two come from
/// different recordings — FOCEI's `run` and the quadrature's `agq_taped` — so their
/// agreement checks each against the other; FOCEI's is pinned to the standalone response
/// (`the_fused_mode_response_is_the_standalone_one`). Asking for the response leaves the
/// quadrature gradient bit for bit.
#[test]
fn the_quadrature_mode_response_is_the_focei_one() {
    let root = env!("CARGO_MANIFEST_DIR");
    let three = parse_model_string(&one_cpt_oral_n_eta(3)).unwrap();
    let s3 = subject(
        &three,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &TIMES,
        &[0.1, -0.05, 0.2],
    );
    #[allow(unused_mut)]
    let mut cases: Vec<(String, CompiledModel, Subject)> = vec![("three".into(), three, s3)];
    // An IOV subject: the stacked `[η, κ]` solve, whose exact Hessian the fused pass and
    // `exact_inner_hessian` must also agree on (#2585 review #4).
    {
        let prep = crate::api::prepare_run(
            &format!("{root}/examples/warfarin_iov.ferx"),
            Some(&format!("{root}/data/warfarin_iov.csv")),
        )
        .unwrap();
        let s = prep
            .population
            .subjects
            .iter()
            .find(|s| !s.occasions.is_empty())
            .expect("an IOV subject")
            .clone();
        assert!(prep.parsed.model.n_kappa > 0, "premise: the model has IOV");
        cases.push(("examples/warfarin_iov.ferx".into(), prep.parsed.model, s));
    }
    #[cfg(feature = "survival")]
    for (m, d) in [
        ("examples/pktte_joint.ferx", "data/pktte_joint.csv"),
        (
            "examples/tte_competing_risks.ferx",
            "data/tte_competing_risks.csv",
        ),
    ] {
        let prep =
            crate::api::prepare_run(&format!("{root}/{m}"), Some(&format!("{root}/{d}"))).unwrap();
        let s = prep
            .population
            .subjects
            .iter()
            .find(|s| !s.obs_records.is_empty())
            .unwrap()
            .clone();
        cases.push((m.into(), prep.parsed.model, s));
    }
    let _ = root;
    let mut worst = 0.0f64;
    let mut live = 0.0f64;
    let mut n_checked = 0;
    for (name, model, s) in &cases {
        assert!(
            model.n_eta > 0,
            "{name}: a model with random effects has a mode to respond"
        );
        let params = moved_params(model);
        let x = pack_params(&params);
        let eta = ebe_for(model, s, &params, 1e-12, true);
        for (n_agq, exact) in [(1, true), (3, true), (1, false), (3, false)] {
            let objective = Objective::Agq { n_agq, exact };
            let (focei, (plain, (fused, dx))) = std::thread::scope(|sc| {
                std::thread::Builder::new()
                    .stack_size(crate::FIT_RAYON_STACK_SIZE)
                    .spawn_scoped(sc, || {
                        let pt = PackedPoint::new(&params, &x).unwrap();
                        let focei = subject_adjoint_gradient_with_eta_dx_at(
                            model,
                            s,
                            &pt,
                            eta.as_slice(),
                            true,
                        )
                        .1;
                        let plain =
                            subject_adjoint_gradient_at(model, s, &pt, eta.as_slice(), objective);
                        let fused = subject_adjoint_gradient_and_eta_dx_at(
                            model,
                            s,
                            &pt,
                            eta.as_slice(),
                            objective,
                        );
                        (focei, (plain, fused))
                    })
                    .unwrap()
                    .join()
                    .unwrap()
            });
            let label = format!("{name} n_agq={n_agq} exact={exact}");
            let (plain, fused) = (
                plain.unwrap_or_else(|| panic!("{label}: no gradient")),
                fused.unwrap_or_else(|| panic!("{label}: no fused gradient")),
            );
            // Not bitwise: with the response formed, Ψ's θ adjoint is read off A's and the
            // rows' instead of its own sweep (`agq_taped`), the same quantity summed in
            // another order: measured worst 4.1e-16 over these fixtures, so 1e-14 keeps ~25×
            // headroom. A response that corrupted the gradient moves it by far more.
            for (a, b) in plain.iter().zip(&fused) {
                assert!(a.is_finite() && b.is_finite(), "{label}: {a} vs {b}");
                let gap = (a - b).abs() / (1.0 + b.abs());
                assert!(gap < 1e-14, "{label}: the gradient moved by {gap:.2e}");
            }
            let want = focei.unwrap_or_else(|| panic!("{label}: no FOCEI response"));
            let got = dx.unwrap_or_else(|| panic!("{label}: no quadrature response"));
            assert_eq!(got.len(), want.len(), "{label}");
            for (gk, wk) in got.iter().zip(&want) {
                assert_eq!(gk.len(), wk.len(), "{label}");
                for (a, b) in gk.iter().zip(wk.iter()) {
                    assert!(a.is_finite() && b.is_finite(), "{label}: {a} vs {b}");
                    worst = worst.max((a - b).abs() / (1.0 + b.abs()));
                    live = live.max(b.abs());
                }
            }
            n_checked += 1;
            eprintln!("{label}: quadrature vs FOCEI mode response {worst:.2e}");
        }
    }
    assert!(n_checked >= 4, "every anchor and node count was checked");
    assert!(live > 1e-2, "the mode does not move: the test sees nothing");
    assert!(
        worst < 1e-10,
        "quadrature vs FOCEI mode response {worst:.2e}"
    );
}

/// The exact inner Hessian a quadrature gradient pass hands back (`ModeOut::hessian`, #2549)
/// is the one `exact_inner_hessian` forms: the quadrature objective's exact anchor now takes
/// it from the score's pass instead of recomputing it, and the next warm inner solve takes it
/// as its seed. Gaussian, joint PK–TTE and competing-risks subjects, both anchors, 1 and 3
/// nodes; asked for without the response too.
#[test]
fn the_quadrature_mode_hessian_is_the_exact_inner_hessian() {
    let root = env!("CARGO_MANIFEST_DIR");
    let three = parse_model_string(&one_cpt_oral_n_eta(3)).unwrap();
    let s3 = subject(
        &three,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &TIMES,
        &[0.1, -0.05, 0.2],
    );
    #[allow(unused_mut)]
    let mut cases: Vec<(String, CompiledModel, Subject)> = vec![("three".into(), three, s3)];
    // An IOV subject: the stacked `[η, κ]` solve, whose exact Hessian the fused pass and
    // `exact_inner_hessian` must also agree on (#2585 review #4).
    {
        let prep = crate::api::prepare_run(
            &format!("{root}/examples/warfarin_iov.ferx"),
            Some(&format!("{root}/data/warfarin_iov.csv")),
        )
        .unwrap();
        let s = prep
            .population
            .subjects
            .iter()
            .find(|s| !s.occasions.is_empty())
            .expect("an IOV subject")
            .clone();
        assert!(prep.parsed.model.n_kappa > 0, "premise: the model has IOV");
        cases.push(("examples/warfarin_iov.ferx".into(), prep.parsed.model, s));
    }
    #[cfg(feature = "survival")]
    for (m, d) in [
        ("examples/pktte_joint.ferx", "data/pktte_joint.csv"),
        (
            "examples/tte_competing_risks.ferx",
            "data/tte_competing_risks.csv",
        ),
    ] {
        let prep =
            crate::api::prepare_run(&format!("{root}/{m}"), Some(&format!("{root}/{d}"))).unwrap();
        let s = prep
            .population
            .subjects
            .iter()
            .find(|s| !s.obs_records.is_empty())
            .unwrap()
            .clone();
        cases.push((m.into(), prep.parsed.model, s));
    }
    let _ = root;
    let mut worst = 0.0f64;
    let mut n_checked = 0;
    for (name, model, s) in &cases {
        let params = moved_params(model);
        let x = pack_params(&params);
        let eta = ebe_for(model, s, &params, 1e-12, true);
        let want = exact_inner_hessian(model, s, &params, eta.as_slice())
            .unwrap_or_else(|| panic!("{name}: exact_inner_hessian"));
        for (n_agq, exact) in [(1, true), (3, true), (1, false), (3, false)] {
            for want_dx in [false, true] {
                let objective = Objective::Agq { n_agq, exact };
                let got = std::thread::scope(|sc| {
                    std::thread::Builder::new()
                        .stack_size(crate::FIT_RAYON_STACK_SIZE)
                        .spawn_scoped(sc, || {
                            let pt = PackedPoint::new(&params, &x).unwrap();
                            let mut out = ModeOut {
                                want_dx,
                                want_hessian: true,
                                ..ModeOut::default()
                            };
                            let g = subject_adjoint_gradient_and_mode_at(
                                model,
                                s,
                                &pt,
                                eta.as_slice(),
                                objective,
                                &mut out,
                            );
                            assert!(g.is_some(), "{name}: the gradient");
                            assert_eq!(out.eta_dx.is_some(), want_dx, "{name}: the response");
                            out.hessian
                        })
                        .unwrap()
                        .join()
                        .unwrap()
                });
                let label = format!("{name} n_agq={n_agq} exact={exact} dx={want_dx}");
                let got = got.unwrap_or_else(|| panic!("{label}: no mode Hessian"));
                assert_eq!(got.shape(), want.shape(), "{label}");
                for (a, b) in got.iter().zip(want.iter()) {
                    assert!(a.is_finite() && b.is_finite(), "{label}: {a} vs {b}");
                    worst = worst.max((a - b).abs() / (1.0 + b.abs()));
                }
                n_checked += 1;
            }
        }
    }
    eprintln!("mode Hessian vs exact_inner_hessian: {worst:.2e} over {n_checked} cases");
    // Measured 4.34e-16 over the 32 cases, the IOV ones included (Linux x86_64): the same
    // kernel on the same jets, differing in rounding only; 1e-12 leaves four decades.
    assert!(
        worst < 1e-12,
        "mode Hessian vs exact_inner_hessian {worst:.2e}"
    );
}

// ---- compartment on/off records (#2732) ------------------------------------------------

/// Two compartments with back-transfer, read from `central`: what the off compartment
/// holds reaches the observations, so a wrong jet across a toggle shows in the gradient.
const TWO_CPT_TOGGLE_ODE: &str = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(30.0, 1.0, 500.0)
  theta TVQ(6.0, 0.1, 100.0)
  theta TVV2(40.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.06
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV * exp(ETA_V)
  Q = TVQ
  V2 = TVV2
[structural_model]
  ode(states=[central, peri])
[odes]
  d/dt(central) = -(CL/V) * central - (Q/V) * central + (Q/V2) * peri
  d/dt(peri) = (Q/V) * central - (Q/V2) * peri
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// `s` with on/off rows at `rows` = `(time, state, on)`, filled in as a caller outside the
/// crate has to: `DoseEvent::off_at` is left alone, so the walks derive it.
fn with_toggles(mut s: Subject, rows: &[(f64, usize, bool)]) -> Subject {
    for (m, &(t, state, on)) in rows.iter().enumerate() {
        s.pk_only_times.push(t);
        s.pk_only_raw_times.push(t);
        s.cmt_toggles.push(crate::types::CmtToggle {
            pk_only: m,
            state,
            on,
        });
    }
    s
}

fn toggle_model() -> CompiledModel {
    with_tol(
        parse_model_string(TWO_CPT_TOGGLE_ODE).unwrap(),
        1e-12,
        1e-14,
    )
}

/// Measured worst 5.7e-9 on macOS arm64 (the infusion stopped by a toggle), the Richardson
/// FD noise floor at these tolerances. About 17× headroom.
const TOGGLE_BOUND: f64 = 1e-7;

const TOGGLE_TIMES: [f64; 8] = [1.0, 2.0, 3.5, 5.0, 6.5, 8.0, 10.0, 14.0];

/// The fixture's toggle must move the prediction, or a parity pass says nothing about it.
fn assert_toggle_is_live(model: &CompiledModel, with: &Subject, without: &Subject) {
    let theta = &model.default_params.theta;
    let eta = vec![0.0; model.n_eta];
    let a = crate::pk::compute_predictions_with_tv(model, with, theta, &eta);
    let b = crate::pk::compute_predictions_with_tv(model, without, theta, &eta);
    let gap = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum::<f64>();
    assert!(a.iter().all(|x| x.is_finite()), "{a:?}");
    assert!(gap > 1e-2, "the toggle changes nothing: {a:?}");
}

#[test]
fn adjoint_matches_fd_across_a_toggle_with_residual_on_both_sides() {
    let model = toggle_model();
    let plain = subject(
        &model,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(6.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &TOGGLE_TIMES,
        &[0.1, -0.05],
    );
    let s = with_toggles(
        plain.clone(),
        &[(3.0, 1, false), (4.5, 1, true), (9.0, 1, false)],
    );
    assert_toggle_is_live(&model, &s, &plain);
    let worst = fd_oracle(&model, &s, true);
    assert!(
        worst < TOGGLE_BOUND,
        "worst relative difference {worst:.2e}"
    );
}

#[test]
fn adjoint_matches_fd_when_the_main_compartment_is_off() {
    let model = toggle_model();
    let plain = subject(
        &model,
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &[1.0, 2.0, 5.0, 6.5, 8.0, 10.0],
        &[0.1, -0.05],
    );
    // `central` off from 3 to 4 reads nothing on the observations in that window, so the
    // observations after it are driven entirely by what `peri` kept.
    let s = with_toggles(plain.clone(), &[(3.0, 0, false), (4.0, 0, true)]);
    assert_toggle_is_live(&model, &s, &plain);
    let worst = fd_oracle(&model, &s, true);
    assert!(
        worst < TOGGLE_BOUND,
        "worst relative difference {worst:.2e}"
    );
}

#[test]
fn adjoint_matches_fd_for_an_infusion_stopped_by_a_toggle() {
    let model = toggle_model();
    let plain = subject(
        &model,
        vec![
            DoseEvent::new(0.0, 50.0, 1, 0.0, false, 0.0),
            DoseEvent::new(1.0, 80.0, 2, 10.0, false, 0.0),
        ],
        &TOGGLE_TIMES,
        &[0.1, -0.05],
    );
    let s = with_toggles(plain.clone(), &[(4.0, 1, false), (6.0, 1, true)]);
    assert_eq!(
        s.doses[1].off_at,
        f64::INFINITY,
        "derived by the walk, not here"
    );
    assert_toggle_is_live(&model, &s, &plain);
    let worst = fd_oracle(&model, &s, true);
    assert!(
        worst < TOGGLE_BOUND,
        "worst relative difference {worst:.2e}"
    );
}

#[test]
fn adjoint_matches_fd_for_a_dose_that_turns_its_compartment_on() {
    let model = toggle_model();
    let plain = subject(
        &model,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(5.5, 60.0, 2, 0.0, false, 0.0),
        ],
        &TOGGLE_TIMES,
        &[0.1, -0.05],
    );
    let s = with_toggles(plain.clone(), &[(3.0, 1, false)]);
    assert_toggle_is_live(&model, &s, &plain);
    let worst = fd_oracle(&model, &s, true);
    assert!(
        worst < TOGGLE_BOUND,
        "worst relative difference {worst:.2e}"
    );
}

#[test]
fn adjoint_matches_fd_for_a_reset_that_turns_a_compartment_on() {
    let model = toggle_model();
    let mut plain = subject(
        &model,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(6.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &TOGGLE_TIMES,
        &[0.1, -0.05],
    );
    plain.reset_times = vec![5.5];
    let s = with_toggles(plain.clone(), &[(3.0, 1, false)]);
    let worst = fd_oracle(&model, &s, true);
    assert!(
        worst < TOGGLE_BOUND,
        "worst relative difference {worst:.2e}"
    );
}

/// A lag on a compartment nothing switches is a moving boundary with a held state beside
/// it: the walk serves it, and the arrival's saltation has to see the held row.
#[test]
fn adjoint_matches_fd_for_a_lagged_arrival_while_another_compartment_is_off() {
    let src = TWO_CPT_TOGGLE_ODE
        .replace(
            "  theta TVV2(40.0, 1.0, 500.0)\n",
            "  theta TVV2(40.0, 1.0, 500.0)\n  theta TVLAG(0.7, 0.01, 5.0)\n",
        )
        .replace("  V2 = TVV2\n", "  V2 = TVV2\n  ALAG1 = TVLAG\n");
    let model = with_tol(parse_model_string(&src).unwrap(), 1e-12, 1e-14);
    let plain = subject(
        &model,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(4.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &TOGGLE_TIMES,
        &[0.1, -0.05],
    );
    let s = with_toggles(plain.clone(), &[(3.0, 1, false), (6.0, 1, true)]);
    assert_toggle_is_live(&model, &s, &plain);
    assert!(in_scope(&model, &s), "the lag is on another compartment");
    let worst = fd_oracle(&model, &s, true);
    assert!(
        worst < TOGGLE_BOUND,
        "worst relative difference {worst:.2e}"
    );
}

/// The arrival of a lagged dose into a compartment that is off switches its right-hand side at
/// a moving time: the pre side of that crossing keeps the compartment's rate off, the post side
/// has it on. Served and exact. Two fixtures straddle the `was_off` gate: `peri` is off when the
/// lagged dose arrives (live crossing), and `peri` was switched back on before the arrival (the
/// arrival's mask is a no-op). A bolus lands on residual drug in `central` in both, and `peri`
/// feeds `central`, which is observed, so the arrival time moves the observations.
#[test]
fn a_lagged_dose_into_a_switched_compartment_matches_fd() {
    let src = TWO_CPT_TOGGLE_ODE
        .replace(
            "  theta TVV2(40.0, 1.0, 500.0)\n",
            "  theta TVV2(40.0, 1.0, 500.0)\n  theta TVLAG(0.7, 0.01, 5.0)\n",
        )
        .replace("  V2 = TVV2\n", "  V2 = TVV2\n  ALAG2 = TVLAG\n");
    let model = with_tol(parse_model_string(&src).unwrap(), 1e-12, 1e-14);
    let plain = subject(
        &model,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(4.0, 50.0, 2, 0.0, false, 0.0),
        ],
        &TOGGLE_TIMES,
        &[0.1, -0.05],
    );
    assert!(in_scope(&model, &plain), "without the toggle it is served");
    for (rows, label) in [
        (vec![(3.0, 1usize, false)], "off at the arrival"),
        (vec![(3.0, 1, false), (3.5, 1, true)], "back on before it"),
    ] {
        let s = with_toggles(plain.clone(), &rows);
        assert_toggle_is_live(&model, &s, &plain);
        assert!(in_scope(&model, &s), "{label}: must be served");
        let worst = fd_oracle(&model, &s, true);
        eprintln!("lagged dose into switched cmt, {label}: {worst:.2e}");
        assert!(worst < TOGGLE_BOUND, "{label}: {worst:.2e}");
    }
}

/// A toggle is an event only the event-driven walk has, on the value path and the dual
/// one, so a subject with nothing else to send it there must not take the static walk.
#[test]
fn a_toggle_alone_takes_the_event_driven_walk() {
    let model = toggle_model();
    let plain = subject(
        &model,
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &TOGGLE_TIMES,
        &[0.1, -0.05],
    );
    let s = with_toggles(plain.clone(), &[(3.0, 1, false)]);
    use crate::sens::ode_provider::{event_walk_subject_supported, static_walk_subject_supported};
    assert!(static_walk_subject_supported(&model, &plain));
    assert!(!event_walk_subject_supported(&model, &plain));
    assert!(!static_walk_subject_supported(&model, &s));
    assert!(event_walk_subject_supported(&model, &s));
}

/// `examples/warfarin_ode_time.ferx` switches `d/dt(TAM_INT)` on a state (`central / V`) and a
/// clock, so the switching time moves with CL, V and every eta. The integration locates it and
/// jumps `TAM_INT`'s sensitivity there (#2649); nothing observed reads `TAM_INT`, so the
/// gradient must equal reconverged FD with or without the jump, and the locating must not
/// disturb the observed `central`. The fixture is live: the concentration crosses 0.5 mg/L
/// within a dosing interval, in both directions.
///
/// Regression caught: the model declined, or the served gradient parting from the function.
#[test]
fn ode_switch_in_an_unobserved_state_is_served_and_matches_reconverged_fd() {
    let model =
        tight(parse_model_string(include_str!("../../examples/warfarin_ode_time.ferx")).unwrap());
    let ode = model.ode_spec.as_ref().unwrap();
    // The switch moves with a parameter: the integration locates it (#2649).
    assert_eq!(ode.rhs_program.as_ref().unwrap().switch_count(), 1);

    let eta = [0.1, -0.05, 0.08];
    let s = subject(
        &model,
        vec![
            DoseEvent::new(0.0, 6.0, 1, 0.0, false, 0.0),
            DoseEvent::new(24.0, 6.0, 1, 0.0, false, 0.0),
        ],
        &[
            0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.5, 26.0, 30.0, 36.0, 44.0, 48.0,
        ],
        &eta,
    );
    let conc =
        crate::pk::compute_predictions_with_tv(&model, &s, &model.default_params.theta, &eta);
    let above: Vec<bool> = conc.iter().map(|&c| c > 0.5).collect();
    let flips = above.windows(2).filter(|w| w[0] != w[1]).count();
    assert!(flips >= 3, "the switch is not live: {conc:?}");
    assert!(in_scope(&model, &s), "the exact gradient must serve it");
    let worst = fd_oracle(&model, &s, true);
    eprintln!("warfarin_ode_time worst {worst:.3e}");
    // Measured worst 6.1e-7 on Linux x86_64 (1.4e-7 before the switch was located, #2649:
    // the steps now end on each crossing, at the FD noise floor either way); about 3x headroom.
    assert!(worst < 2e-6, "worst relative difference {worst:.2e}");
}

// ---- SDE models (EKF process-noise variance as kernel rows) ---------------------------

const SDE_ONE_CPT: &str = r#"
[parameters]
  theta TVCL(0.134, 0.001, 10.0)
  theta TVV(8.1, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  omega ETA_CL ~ 0.07
  omega ETA_V ~ 0.02
  omega ETA_KA ~ 0.40
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot) = -KA * depot
  d/dt(central) = KA * depot / V - (CL / V) * central
[diffusion]
  central ~ 0.02
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

fn sde_fixture() -> (CompiledModel, Subject) {
    let model = with_tol(parse_model_string(SDE_ONE_CPT).unwrap(), 1e-11, 1e-13);
    assert!(model.is_sde());
    let s = subject(
        &model,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &TIMES,
        &[0.1, -0.05, 0.2],
    );
    (model, s)
}

/// Bound for the SDE parity test, set from the measured worst (printed by the test) with
/// headroom.
const SDE_PARITY_BOUND: f64 = 1e-5;

/// An SDE subject is served by the adjoint, and its packed FOCEI and FOCE gradients (θ,
/// including the diffusion variance, Ω and σ — σ through the Kalman `R`) match the
/// reconverged finite-difference oracle of the production objective, which evaluates
/// `ekf_p_obs` in f64. Dies when the variance rows are dropped (the gradient then misses
/// the diffusion θ and σ's `p_obs` path) or when σ is held constant in the filter.
#[test]
fn sde_adjoint_matches_reconverged_finite_differences() {
    let (model, s) = sde_fixture();
    assert!(in_scope(&model, &s));
    for interaction in [true, false] {
        let worst = fd_oracle(&model, &s, interaction);
        eprintln!("sde interaction={interaction} worst {worst:.3e}");
        assert!(
            worst < SDE_PARITY_BOUND,
            "interaction={interaction}: {worst:.2e}"
        );
    }
}

/// The inner η-gradient and the exact inner Hessian of an SDE subject — both carry the
/// η-dependence of `p_obs` — against central differences of the production inner objective
/// (`individual_nll`, which recomputes `ekf_p_obs` at each η).
#[test]
fn sde_inner_gradient_and_hessian_match_fd_of_the_inner_objective() {
    let (model, s) = sde_fixture();
    let p = &model.default_params;
    let eta = [0.08, -0.06, 0.1];
    let l = |e: &[f64]| {
        crate::stats::likelihood::individual_nll(&model, &s, &p.theta, e, &p.omega, &p.sigma.values)
    };
    let g = exact_inner_gradient(&model, &s, p, &eta, None).expect("served");
    let h = exact_inner_hessian(&model, &s, p, &eta).expect("served");
    let (mut wg, mut wh) = (0.0f64, 0.0f64);
    for a in 0..3 {
        let d1 = |st: f64| {
            let (mut ep, mut em) = (eta, eta);
            ep[a] += st;
            em[a] -= st;
            (l(&ep) - l(&em)) / (2.0 * st)
        };
        let fd = (4.0 * d1(1e-4) - d1(2e-4)) / 3.0;
        assert!(fd.is_finite() && g[a].is_finite());
        wg = wg.max((g[a] - fd).abs() / (1.0 + fd.abs()));
        for b in 0..3 {
            let d2 = |st: f64| {
                let at = |sa: f64, sb: f64| {
                    let mut e = eta;
                    e[a] += sa;
                    e[b] += sb;
                    l(&e)
                };
                (at(st, st) - at(st, -st) - at(-st, st) + at(-st, -st)) / (4.0 * st * st)
            };
            let fd = (4.0 * d2(1e-3) - d2(2e-3)) / 3.0;
            assert!(fd.is_finite() && h[(a, b)].is_finite());
            wh = wh.max((h[(a, b)] - fd).abs() / (1.0 + fd.abs()));
        }
    }
    eprintln!("sde inner gradient worst {wg:.3e}, hessian worst {wh:.3e}");
    assert!(
        wg < 1e-6 && wh < 1e-4,
        "gradient {wg:.2e}, hessian {wh:.2e}"
    );
}

/// An SDE model over amounts, `variant` shaping what each route must mirror: `""` plain,
/// `"iov"` a κ on CL, `"scalar"` a constant output scale (the parser refuses `[scaling]` on
/// an SDE model, so [`sde_variant`] sets it on the compiled model), `"tv"` a time-varying
/// weight on CL.
fn sde_variant_src(variant: &str) -> String {
    let (kappa, cl, fit, scaling) = match variant {
        "iov" => (
            "  kappa KAPPA_CL ~ 0.04\n",
            "TVCL * exp(ETA_CL + KAPPA_CL)",
            "[fit_options]\n  iov_column = OCC\n",
            "",
        ),
        "tv" => ("", "TVCL * exp(ETA_CL) * WT / 70", "", ""),
        _ => ("", "TVCL * exp(ETA_CL)", "", ""),
    };
    let diff = if variant == "scalar" { "0.5" } else { "0.02" };
    format!(
        "[parameters]\n  theta TVCL(0.134, 0.001, 10.0)\n  theta TVV(8.1, 0.1, 500.0)\n  \
         theta TVKA(1.0, 0.01, 50.0)\n  omega ETA_CL ~ 0.07\n  omega ETA_V ~ 0.02\n  \
         omega ETA_KA ~ 0.40\n{kappa}  sigma PROP_ERR ~ 0.1 (sd)\n[individual_parameters]\n  \
         CL = {cl}\n  V = TVV * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n[structural_model]\n  \
         ode(obs_cmt=central, states=[depot, central])\n[odes]\n  d/dt(depot) = -KA * depot\n  \
         d/dt(central) = KA * depot - (CL / V) * central\n[diffusion]\n  central ~ {diff}\n\
         [error_model]\n  DV ~ proportional(PROP_ERR)\n{scaling}{fit}"
    )
}

fn sde_variant(variant: &str) -> (CompiledModel, Subject) {
    let mut model = with_tol(
        parse_model_string(&sde_variant_src(variant)).unwrap(),
        1e-11,
        1e-13,
    );
    if variant == "scalar" {
        model.scaling = crate::types::ScalingSpec::ScalarScale(10.0);
    }
    assert!(model.is_sde(), "{variant}");
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let mut s = subject(&model, doses, &TIMES, &[0.1, -0.05, 0.2]);
    if variant == "iov" {
        s.occasions = TIMES
            .iter()
            .map(|&t| if t < 12.0 { 1 } else { 2 })
            .collect();
        s.dose_occasions = vec![1, 2];
    }
    if variant == "tv" {
        let w = |v: f64| HashMap::from([("WT".to_string(), v)]);
        s.covariates.insert("WT".into(), 70.0);
        s.dose_covariates = vec![w(70.0), w(80.0)];
        s.obs_covariates = TIMES
            .iter()
            .map(|&t| w(if t < 12.0 { 70.0 } else { 80.0 }))
            .collect();
        assert!(s.has_tv_covariates());
    }
    // A data set the model fits only roughly: the predictions at the reference η, perturbed.
    let preds = crate::pk::compute_predictions_with_tv(
        &model,
        &s,
        &model.default_params.theta,
        &[0.1, -0.05, 0.2],
    );
    s.observations = preds
        .iter()
        .enumerate()
        .map(|(j, p)| p * (1.0 + 0.15 * (1.3 * j as f64).sin()))
        .collect();
    (model, s)
}

/// Laplace and AGQ (`n = 3`, both anchors): the node predictions carry the EKF variance
/// rows, so the quadrature gradient matches the reconverged finite-difference oracle of
/// production's AGQ objective. Dies when a node's variance rows are dropped or held.
#[test]
fn sde_laplace_and_agq_match_reconverged_fd() {
    let (model, s) = sde_fixture();
    for (n_agq, exact) in [(1, true), (3, true), (3, false)] {
        let worst = fd_oracle_agq(&model, &s, n_agq, exact);
        eprintln!("sde n_agq={n_agq} exact={exact}: worst {worst:.3e}");
        // Measured 2.8e-7 (Laplace), 3.8e-9, 3.2e-9; bound 1e-5 is SDE_PARITY_BOUND.
        assert!(
            worst < SDE_PARITY_BOUND,
            "n_agq={n_agq} exact={exact}: {worst:.2e}"
        );
    }
}

/// The covariance R-matrix of an SDE subject (the variance rows as one more stage of the
/// covariance plan) against central differences of the exact gradient, EBE re-found — under
/// the default `auto` stepper. The oracle's EBE is Newton-polished on the exact inner
/// derivatives (`hessian_gaps_for`): an ODE inner search stops on an objective stall, and
/// where it stops depends on its path — a line-search trial at `KA ≈ 500` is genuinely stiff
/// (`|λ| = KA`) and integrated by Rosenbrock under `auto`, by RK45 when pinned — which moved
/// the mode 4.5e-9 and the FOCEI gap to 1.7e-6 (3.5e-7 pinned), with the exact matrix and
/// gradient bit-identical between the two at any one mode.
#[test]
fn sde_covariance_hessian_matches_fd_of_the_gradient() {
    let (model, s) = sde_fixture();
    for (objective, bound) in [
        (Objective::Foce, 3e-7),
        (Objective::Focei, 1e-9),
        (
            Objective::Agq {
                n_agq: 3,
                exact: true,
            },
            1e-9,
        ),
    ] {
        let (a, _) = hessian_gaps_for(&model, &s, objective);
        eprintln!("sde hessian {objective:?}: gap {a:.3e}");
        // Measured on Linux under `auto`: 5.8e-8 (FOCE, its frozen-variance mode unpolished),
        // 7.9e-11 (FOCEI), 4.1e-11 (AGQ n=3 exact).
        assert!(a < bound, "{objective:?}: {a:.2e}");
    }
}

/// Central difference of `l` along `e[a]`, Richardson-extrapolated from steps `h` and `2h`.
fn richardson(l: &dyn Fn(&[f64]) -> f64, e: &[f64], a: usize, h: f64) -> f64 {
    let d = |st: f64| {
        let (mut ep, mut em) = (e.to_vec(), e.to_vec());
        ep[a] += st;
        em[a] -= st;
        (l(&ep) - l(&em)) / (2.0 * st)
    };
    (4.0 * d(h) - d(2.0 * h)) / 3.0
}

/// The SDE fixture with a correlated combined residual (`block_sigma`): the inner gradient's
/// dense-`R` branch.
fn sde_dense_fixture() -> (CompiledModel, Subject) {
    let src = SDE_ONE_CPT
        .replace(
            "  sigma PROP_ERR ~ 0.1 (sd)\n",
            "  block_sigma (PROP_ERR, ADD_ERR) = [0.01, 0.004, 0.04]\n",
        )
        .replace("proportional(PROP_ERR)", "combined(PROP_ERR, ADD_ERR)");
    let model = with_tol(parse_model_string(&src).unwrap(), 1e-11, 1e-13);
    let (_, s) = sde_fixture();
    (model, s)
}

/// The per-row inner assemblers on an SDE subject read the EKF variance rows beside the
/// predictions (`exact_obs_grads_into` with σ): `analytic_eta_nll_value_gradient_with_schedule`
/// (the inner search, HMC, the IMH refresh) on its diagonal and its dense-`R` (`block_sigma`)
/// branch, and the IOV `analytic_eta_nll_gradient_iov`. Each gradient matches central
/// differences of the production objective it serves (`individual_nll`, `individual_nll_iov`,
/// which run `ekf_p_obs`), and the fused value is `individual_nll` itself. Dies when a variance
/// row's η-jet or its value in `V` is dropped, on any branch: the variance term is live (the
/// gradient without it misses by more than the bound).
#[test]
fn sde_per_row_inner_gradients_match_fd_of_the_production_objective() {
    use crate::estimation::inner_optimizer::{
        analytic_eta_nll_gradient, analytic_eta_nll_gradient_iov,
        analytic_eta_nll_value_gradient_with_schedule,
    };
    use crate::stats::likelihood::{individual_nll, individual_nll_iov};
    let eta = [0.08, -0.06, 0.1];
    let mut worst = 0.0f64;
    for (name, (model, s)) in [("diagonal", sde_fixture()), ("dense", sde_dense_fixture())] {
        let p = &model.default_params;
        assert_eq!(
            model.residual_cov_theta(&p.theta).is_dense(),
            name == "dense"
        );
        let l = |e: &[f64]| individual_nll(&model, &s, &p.theta, e, &p.omega, &p.sigma.values);
        let g = analytic_eta_nll_gradient(&model, &s, &p.theta, &eta, &p.omega, &p.sigma.values)
            .expect("served");
        // The same gradient without the variance rows' η-jets: the term the test exists for.
        let fd: Vec<f64> = (0..3).map(|a| richardson(&l, &eta, a, 1e-4)).collect();
        let w = (0..3).fold(0.0f64, |w, a| {
            assert!(
                fd[a].is_finite() && g[a].is_finite(),
                "{name}: {} vs {}",
                g[a],
                fd[a]
            );
            w.max((g[a] - fd[a]).abs() / (1.0 + fd[a].abs()))
        });
        eprintln!("sde {name}: per-row inner gradient vs FD {w:.3e}");
        assert!(w.is_finite(), "{name}: {w}");
        worst = worst.max(w);
        if name == "diagonal" {
            let mut value = None;
            let keys = model.error_spec.obs_keys(&s);
            analytic_eta_nll_value_gradient_with_schedule(
                &model,
                &s,
                &p.theta,
                &eta,
                &p.omega,
                &p.sigma.values,
                model.residual_cov_theta(&p.theta),
                None,
                None,
                keys.as_ref(),
                &mut Vec::new(),
                &mut nalgebra::DVector::zeros(3),
                &mut nalgebra::DVector::zeros(3),
                None,
                Some(&mut value),
            )
            .expect("served");
            let (v, want) = (value.expect("the value is fused"), l(&eta));
            eprintln!("sde fused value {v} vs {want}");
            // Measured equal to the last bit.
            assert!(
                (v - want).abs() < 1e-12 * want.abs().max(1.0),
                "{v} vs {want}"
            );
        }
    }
    // IOV: the BSV-only filter's rows over the stacked `[η, κ₁, κ₂]`.
    let (model, s) = sde_variant("iov");
    let p = &model.default_params;
    let oi = p.omega_iov.as_ref().expect("Ω_IOV");
    let stacked = [0.08, -0.06, 0.1, 0.05, -0.04];
    let l = |e: &[f64]| {
        let kappas = vec![vec![e[3]], vec![e[4]]];
        individual_nll_iov(
            &model,
            &s,
            &p.theta,
            &e[..3],
            &kappas,
            &p.omega,
            Some(oi),
            &p.sigma.values,
        )
    };
    let g = analytic_eta_nll_gradient_iov(
        &model,
        &s,
        &p.theta,
        &stacked,
        &p.omega,
        oi,
        &p.sigma.values,
        None,
    )
    .expect("served");
    let w = (0..5).fold(0.0f64, |w, a| {
        let fd = richardson(&l, &stacked, a, 1e-4);
        assert!(fd.is_finite() && g[a].is_finite(), "iov: {} vs {fd}", g[a]);
        w.max((g[a] - fd).abs() / (1.0 + fd.abs()))
    });
    eprintln!("sde iov: per-row inner gradient vs FD {w:.3e}");
    assert!(w.is_finite(), "iov: {w}");
    worst = worst.max(w);
    // Measured 1.1e-10 (diagonal), 1.1e-10 (dense), 1.6e-10 (IOV): the differences' own error.
    assert!(worst < 2e-9, "{worst:.2e}");
}

/// The record-level shapes `record_sde_p` used to decline — IOV (BSV-only filter, κ = 0), an
/// output scale (the Kalman `R` reads the prediction before it), a time-varying covariate
/// (production's filter reads the subject's static ones) — are served, and match the
/// reconverged FD of production's objective.
#[test]
fn sde_iov_scaled_and_tv_subjects_match_reconverged_fd() {
    for variant in ["iov", "scalar", "tv"] {
        let (model, s) = sde_variant(variant);
        if variant != "iov" {
            assert!(in_scope(&model, &s), "{variant}: must be served");
        }
        for interaction in [true, false] {
            let worst = fd_oracle(&model, &s, interaction);
            eprintln!("sde {variant} interaction={interaction}: worst {worst:.3e}");
            // Measured worst 7.9e-8 (iov FOCE), 1.4e-8 (scalar), 3.4e-8 (tv); 12× headroom.
            assert!(
                worst < 1e-6,
                "{variant} interaction={interaction}: {worst:.2e}"
            );
        }
    }
}

/// An SDE subject's fixed-η data score and data η-gradient (the SAEM / VI building blocks,
/// with the variance rows' θ and σ reverse) against central differences of production's
/// individual objective.
#[test]
fn sde_fixed_eta_scores_match_fd() {
    let (model, s) = sde_fixture();
    let p = moved_params(&model);
    let eta = [0.08, -0.06, 0.1];
    let l = |th: &[f64], sg: &[f64], e: &[f64]| {
        crate::stats::likelihood::individual_nll(&model, &s, th, e, &p.omega, sg)
    };
    let (_, dt, ds) =
        fixed_eta_data_score(&model, &s, &p.theta, &p.sigma.values, &eta).expect("served");
    let mut worst = 0.0f64;
    for k in 0..p.theta.len() {
        let st = 1e-5 * p.theta[k];
        let (mut a, mut b) = (p.theta.clone(), p.theta.clone());
        a[k] += st;
        b[k] -= st;
        let fd = (l(&a, &p.sigma.values, &eta) - l(&b, &p.sigma.values, &eta)) / (2.0 * st);
        assert!(dt[k].is_finite() && fd.is_finite(), "theta {k}");
        worst = worst.max((dt[k] - fd).abs() / (1.0 + fd.abs()));
    }
    for k in 0..p.sigma.values.len() {
        let st = 1e-5 * p.sigma.values[k];
        let (mut a, mut b) = (p.sigma.values.clone(), p.sigma.values.clone());
        a[k] += st;
        b[k] -= st;
        let fd = (l(&p.theta, &a, &eta) - l(&p.theta, &b, &eta)) / (2.0 * st);
        assert!(ds[k].is_finite() && fd.is_finite(), "sigma {k}");
        worst = worst.max((ds[k] - fd).abs() / (1.0 + fd.abs()));
    }
    let g = fixed_theta_data_eta_grad(&model, &s, &p.theta, &p.sigma.values, &eta).expect("served");
    let prior = &p.omega.inv * nalgebra::DVector::from_column_slice(&eta);
    for k in 0..3 {
        let st = 1e-5;
        let (mut a, mut b) = (eta, eta);
        a[k] += st;
        b[k] -= st;
        let fd = (l(&p.theta, &p.sigma.values, &a) - l(&p.theta, &p.sigma.values, &b)) / (2.0 * st)
            - prior[k];
        assert!(g[k].is_finite() && fd.is_finite(), "eta {k}");
        worst = worst.max((g[k] - fd).abs() / (1.0 + fd.abs()));
    }
    eprintln!("sde fixed-eta scores worst {worst:.3e}");
    // Measured 4.0e-9 (the difference quotients' own error); 25× headroom.
    assert!(worst < 1e-7, "{worst:.2e}");
}

/// An `SS=1` dose on an SDE subject: the filter equilibrates its `(mean, covariance)` pair
/// (`equilibrate_ss_ekf`), and the plain prediction feeding the Kalman `R` takes the
/// event-driven walk's dual steady state. The packed FOCEI and FOCE gradients match the
/// reconverged-FD oracle; the steady state is live (the first prediction is far above the
/// same subject's non-SS one, and so is `p_obs`' diffusion path through the carried
/// covariance).
#[test]
fn sde_steady_state_dose_matches_reconverged_finite_differences() {
    let (model, plain) = sde_fixture();
    let mut s = plain.clone();
    s.doses[0].ss = true;
    s.doses[0].ii = 12.0;
    let theta = &model.default_params.theta;
    let eta = [0.1, -0.05, 0.2];
    s.observations = crate::pk::compute_predictions_with_tv(&model, &s, theta, &eta)
        .iter()
        .enumerate()
        .map(|(j, p)| p * (1.0 + 0.15 * (1.3 * j as f64).sin()))
        .collect();
    let without = crate::pk::compute_predictions_with_tv(&model, &plain, theta, &eta);
    let with = crate::pk::compute_predictions_with_tv(&model, &s, theta, &eta);
    assert!(
        with[0] > 2.0 * without[0],
        "the steady state is not live: {with:?}"
    );
    assert!(in_scope(&model, &s), "the exact gradient must serve it");
    for interaction in [true, false] {
        let worst = fd_oracle(&model, &s, interaction);
        eprintln!("sde SS interaction={interaction} worst {worst:.3e}");
        // Measured on Linux: 4.9e-8 (FOCEI) and 5.5e-8 (FOCE); about 18x headroom.
        assert!(worst < 1e-6, "interaction={interaction}: {worst:.2e}");
    }
    // Laplace and AGQ through the same plans: the steady state reaches every node.
    for (n_agq, exact) in [(1, true), (3, true), (3, false)] {
        let worst = fd_oracle_agq(&model, &s, n_agq, exact);
        eprintln!("sde SS n_agq={n_agq} exact={exact}: worst {worst:.3e}");
        assert!(
            worst < SDE_PARITY_BOUND,
            "n_agq={n_agq} exact={exact}: {worst:.2e}"
        );
    }
    // Routing: the AGQ population gradient takes the exact route for this subject — no
    // decline, no plan failure. The nodes at high `ETA_KA` (`KA·Δt > 1`) run the capped SS
    // train backwards through every Riccati sub-step, which overflowed to `NaN` and declined
    // under the explicit Euler step.
    use crate::estimation::outer_optimizer::OuterFdDeclineLog;
    use crate::types::{EstimationMethod, FitOptions, Population};
    let params = moved_params(&model);
    let x = pack_params(&params);
    let eta = ebe_for(&model, &s, &params, 1e-12, true);
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
        n_agq: 3,
        ..FitOptions::default()
    };
    let log = OuterFdDeclineLog::new(1);
    crate::estimation::agq::population_gradient_mixed(
        &model,
        &population,
        &params,
        &x,
        &[eta],
        &[Vec::new()],
        &crate::estimation::parameterization::compute_bounds(&params),
        &options,
        false,
        None,
        Some(&log),
        None,
    )
    .expect("population gradient");
    assert!(
        log.declined_indices().is_empty(),
        "the AGQ gradient declined"
    );
    assert!(
        crate::parser::model_parser::rhs_regcode::kernel_jit::plan_failures(&model, &[s]).is_none(),
        "a plan failed"
    );
}

/// The SDE fixture with its depot dosed by an infusion whose window end moves with the
/// parameters: `theta`, `omega` and `indiv` declare the moving quantity (`D1`, `R1` or `F1`),
/// `central` is the central compartment's right-hand side.
fn sde_window_model(theta: &str, omega: &str, indiv: &str, central: &str) -> CompiledModel {
    let src = SDE_ONE_CPT
        .replace("  omega ETA_CL", &format!("  {theta}\n  omega ETA_CL"))
        .replace("  sigma PROP_ERR", &format!("  {omega}\n  sigma PROP_ERR"))
        .replace(
            "  KA = TVKA * exp(ETA_KA)",
            &format!("  KA = TVKA * exp(ETA_KA)\n  {indiv}"),
        )
        .replace(
            "  d/dt(central) = KA * depot / V - (CL / V) * central",
            &format!("  d/dt(central) = {central}"),
        );
    with_tol(parse_model_string(&src).unwrap(), 1e-11, 1e-13)
}

const SDE_CENTRAL: &str = "KA * depot / V - (CL / V) * central";

/// Records both sides of each window end (about 2.25 h after the doses at 0 and 12), every
/// 0.4 h across the range the window end takes at the EBE and the AGQ nodes, so the window
/// end's Riccati sub-intervals stay below `RICCATI_DT_MAX` and the `f64` filter's sub-step
/// count — a jump in the objective — does not change inside a difference quotient.
const SDE_WINDOW_TIMES: [f64; 17] = [
    0.25, 0.5, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 6.0, 12.5, 13.2, 13.6, 14.0, 14.4, 14.8, 15.2, 24.0,
];

/// The three moving windows of an SDE subject — a modeled `D1` (`RATE=-2`), a modeled `R1`
/// (`RATE=-1`) and a rate-defined infusion under an estimated `F1` — each with the θ index of
/// the moving quantity. The second dose lands on residual drug. (The filter's steady-state
/// cycle over a moving window is pinned at Tier 1 on the filter alone:
/// `ode::ekf::tests::recorded_filter_follows_a_moving_infusion_window`.)
fn sde_window_case(case: &str, ss: bool) -> (CompiledModel, Subject, usize) {
    use crate::types::RateMode::{ModeledDuration, ModeledRate};
    let d1 = (
        "theta TVD1(2.47, 0.1, 24.0)",
        "omega ETA_D1 ~ 0.04",
        "D1 = TVD1 * exp(ETA_D1)",
    );
    let (model, mut doses, name) = match case {
        "D1" => {
            let m = sde_window_model(d1.0, d1.1, d1.2, SDE_CENTRAL);
            let d = |t| DoseEvent::modeled(t, 100.0, 1, false, 0.0, ModeledDuration);
            (m, vec![d(0.0), d(12.0)], "TVD1")
        }
        "R1" => {
            let m = sde_window_model(
                "theta TVR1(40.5, 1.0, 500.0)",
                "omega ETA_R1 ~ 0.04",
                "R1 = TVR1 * exp(ETA_R1)",
                SDE_CENTRAL,
            );
            let d = |t| DoseEvent::modeled(t, 100.0, 1, false, 0.0, ModeledRate);
            (m, vec![d(0.0), d(12.0)], "TVR1")
        }
        "F1" => {
            let m = sde_window_model(
                "theta TVF1(0.9, 0.05, 1.0)",
                "omega ETA_F1 ~ 0.04",
                "F1 = TVF1 * exp(ETA_F1)",
                SDE_CENTRAL,
            );
            let d = |t| DoseEvent::new(t, 100.0, 1, 40.0, false, 0.0);
            (m, vec![d(0.0), d(12.0)], "TVF1")
        }
        _ => unreachable!(),
    };
    assert!(model.is_sde());
    if ss {
        doses[0].ss = true;
        doses[0].ii = 12.0;
    }
    let k = model.theta_names.iter().position(|n| n == name).unwrap();
    let s = subject(&model, doses, &SDE_WINDOW_TIMES, &[0.1, -0.05, 0.2, 0.15]);
    (model, s, k)
}

/// The moving window end is live in the filter's covariance: scaling its θ moves production's
/// `f64` `p_obs` at the records after the first window end.
fn assert_window_moves_p_obs(model: &CompiledModel, s: &Subject, k: usize) {
    let p = moved_params(model);
    let eta = [0.05, 0.0, -0.05, 0.1];
    let at = |th: &[f64]| crate::stats::likelihood::ekf_p_obs(model, s, th, &eta, &p.sigma.values);
    let mut th = p.theta.clone();
    th[k] *= 0.8;
    let (a, b) = (at(&p.theta), at(&th));
    assert!(a.iter().chain(&b).all(|v| v.is_finite()), "{a:?} {b:?}");
    let moved = a
        .iter()
        .zip(&b)
        .skip(4)
        .map(|(x, y)| (x - y).abs() / x.abs())
        .fold(0.0, f64::max); // nan-ok: finite asserted above
    assert!(moved > 1e-3, "the window does not reach p_obs: {moved:.2e}");
}

/// An SDE subject whose infusion window end moves with the parameters is served by the exact
/// gradient: the filter's breaks are numbers of the recording type, so the mean solve's span
/// and the Riccati quadrature's sub-step lengths and midpoints carry the window's derivatives
/// (`solve_ekf_g`) — the discrete filter differentiated as it is computed. The packed FOCEI
/// gradient and AGQ (n = 3) match the reconverged-FD oracle of the production objective.
fn sde_moving_window_parity(case: &str, focei_bound: f64, agq_bound: f64) {
    let (model, s, k) = sde_window_case(case, false);
    assert_window_moves_p_obs(&model, &s, k);
    assert!(
        in_scope(&model, &s),
        "{case}: the exact gradient must serve it"
    );
    let focei = fd_oracle(&model, &s, true);
    let agq = fd_oracle_agq(&model, &s, 3, true);
    eprintln!("sde moving window {case}: FOCEI worst {focei:.3e}, AGQ n=3 worst {agq:.3e}");
    assert!(focei < focei_bound, "{case} FOCEI: {focei:.2e}");
    assert!(agq < agq_bound, "{case} AGQ: {agq:.2e}");
}

#[test]
fn sde_modeled_duration_matches_reconverged_finite_differences() {
    // Measured on Linux: FOCEI 4.5e-8, AGQ 3.6e-10; about 10× headroom on the larger.
    sde_moving_window_parity("D1", 5e-7, 5e-7);
}

#[test]
fn sde_modeled_rate_matches_reconverged_finite_differences() {
    // Measured on Linux: FOCEI 2.1e-8, AGQ 3.9e-9.
    sde_moving_window_parity("R1", 5e-7, 5e-7);
}

#[test]
fn sde_rate_defined_infusion_under_f_matches_reconverged_finite_differences() {
    // Measured on Linux: FOCEI 2.1e-8, AGQ 1.7e-9.
    sde_moving_window_parity("F1", 5e-7, 5e-7);
}

/// The three moving windows under an `SS=1` first dose: the filter's periodic steady state
/// (`equilibrate_ss_ekf`) cycles over the moving window, its mean and covariance both, the
/// Kalman `R`'s prediction takes the event walk's steady state over the same window, and the
/// second dose lands on that trough. FOCEI and FOCE against the reconverged-FD oracle.
///
/// `D1` starts from `TVD1 = 2.3` here: at 2.47 the FOCEI EBE puts the window end at 2.4037,
/// 0.004 h from the record at 2.4, where the objective's `H` has a kink inside the oracle's
/// difference quotient (measured: 6.0e-4 at 2.47, 1.1e-8 at 2.3, 2.5e-8 at 2.62).
#[test]
fn sde_steady_state_moving_windows_match_reconverged_finite_differences() {
    for case in ["D1", "R1", "F1"] {
        let (mut model, s, k) = sde_window_case(case, true);
        if case == "D1" {
            model.default_params.theta[k] = 2.3;
        }
        assert!(s.doses[0].ss && s.doses[0].ii == 12.0);
        assert_window_moves_p_obs(&model, &s, k);
        assert!(
            in_scope(&model, &s),
            "{case}: the exact gradient must serve it"
        );
        for interaction in [true, false] {
            let w = fd_oracle(&model, &s, interaction);
            eprintln!("sde SS moving window {case} interaction={interaction}: worst {w:.3e}");
            assert!(w < 5e-7, "{case} interaction={interaction}: {w:.2e}");
        }
    }
}

/// Subject `b`'s outer gradient — its prediction, kernel and term plans — served from `a`'s
/// templates (`kj::Template`), against `b` planned fresh: bit for bit. `(planned, of them from a
/// template)` over `b`'s served pass. Dies when a served plan keeps the template's literals or
/// loop calls (`a`'s data) instead of `b`'s.
pub(super) fn templated_twin(model: &CompiledModel, a: &Subject, b: &Subject) -> (usize, usize) {
    use crate::parser::model_parser::rhs_regcode::kernel_jit as kj;
    let params = moved_params(model);
    let x = pack_params(&params);
    let grad = |s: &Subject, eta: &[f64]| {
        subject_adjoint_packed_gradient(model, s, &params, &x, eta, true).expect("served")
    };
    let probe = |off: bool| kj::TEMPLATE_PROBE.with(|p| p.replace((off, 0, 0)));
    let replays = || kj::REPLAYS.with(|c| c.replace(0));
    // Both at `a`'s EBE: a recording's guards read η (a matrix exponential's squaring count),
    // so a point of `b`'s own may take a branch `a` never recorded.
    let eta = ebe_for(model, a, &params, 1e-10, true);
    let eta = eta.as_slice();
    // `b` leaves no template of its own: what serves it below is `a`'s.
    probe(true);
    kj::reset();
    let fresh = grad(b, eta);
    probe(false);
    grad(a, eta);
    kj::reset();
    probe(false);
    replays();
    let served = grad(b, eta);
    let (_, n, k) = probe(false);
    REPLAYED.with(|c| c.set(replays()));
    assert_eq!(served.len(), fresh.len());
    for (i, (s, f)) in served.iter().zip(&fresh).enumerate() {
        assert!(f.is_finite(), "coordinate {i}: {f}");
        assert_eq!(
            s.to_bits(),
            f.to_bits(),
            "coordinate {i}: served {s} vs fresh {f}"
        );
    }
    (n, k)
}

thread_local! {
    /// [`templated_twin`]'s served pass: how many of `b`'s recordings replayed `a`'s template.
    static REPLAYED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// `s` with other data and the same events: slightly later times (a matrix exponential's squaring
/// count follows the spans, so a large shift is a structure of its own), larger doses, other
/// observations.
pub(super) fn other_data(s: &Subject) -> Subject {
    let mut b = s.clone();
    b.obs_times.iter_mut().for_each(|t| *t *= 1.001);
    b.doses.iter_mut().for_each(|d| d.amt *= 1.3);
    for (j, y) in b.observations.iter_mut().enumerate() {
        *y *= 1.1 + 0.02 * j as f64;
    }
    b
}

/// One plan per event structure (#2786): a subject with another's events and its own data is
/// planned from that subject's templates, exactly — closed-form multi-dose, ODE, IOV — and a
/// subject whose data put a dose after every observation (an event-order branch) records a
/// structure of its own, still exact.
#[test]
fn a_subject_served_from_anothers_template_matches_its_fresh_plan_bitwise() {
    let doses = |at: &[f64]| -> Vec<DoseEvent> {
        at.iter()
            .map(|&t| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0))
            .collect()
    };
    let times = [1.1, 4.2, 8.3, 14.4, 20.5];
    let cf = parse_model_string(ONE_CPT_ORAL).unwrap();
    let a = subject(&cf, doses(&[0.0, 12.0]), &times, &[0.1, -0.1, 0.2]);
    let (n, k) = templated_twin(&cf, &a, &other_data(&a));
    assert!(
        k > 0 && k == n,
        "closed form: {k} of {n} plans from a template"
    );
    // Recorded by replaying `a`'s template (its pool and loop calls `b`'s own), not afresh.
    let replayed = |what: &str| {
        let r = REPLAYED.with(std::cell::Cell::get);
        assert!(r > 0, "{what}: no recording replayed its family's template");
    };
    replayed("closed form");
    let flip = subject(&cf, doses(&[0.0, 30.0]), &times, &[0.1, -0.1, 0.2]);
    let (n, k) = templated_twin(&cf, &a, &flip);
    assert!(
        k < n,
        "a dose after every observation is a structure of its own: {k} of {n}"
    );
    // The family now holds both structures, `flip`'s first: `a`'s twin is replayed from `a`'s
    // template, the replay moving to it at the first decision the two structures make apart.
    let (n, k) = templated_twin(&cf, &a, &other_data(&a));
    assert!(k > 0 && k == n, "second structure: {k} of {n}");
    replayed("closed form, its template second in the family");
    let ode = tight(parse_model_string(ONE_CPT_ORAL_ODE).unwrap());
    let a = subject(&ode, doses(&[0.0, 12.0]), &times, &[0.1, -0.1]);
    let (n, k) = templated_twin(&ode, &a, &other_data(&a));
    assert!(k > 0 && k == n, "ODE: {k} of {n} plans from a template");
    replayed("ODE");
    let iov = tight(parse_model_string(ODE_IOV).unwrap());
    // Its times kept: the fixture's round times and the per-occasion times derived from them
    // coincide (`16 − 12 = 4`), and equal literals share a register, so shifting them is a
    // structure of its own. Its doses and observations differ.
    let a = iov_subject(&iov);
    let mut b = a.clone();
    b.doses.iter_mut().for_each(|d| d.amt *= 1.3);
    b.observations.iter_mut().for_each(|y| *y *= 1.2);
    let (n, k) = templated_twin(&iov, &a, &b);
    assert!(k > 0 && k == n, "IOV: {k} of {n} plans from a template");
    replayed("IOV");
}

/// [`a_subject_served_from_anothers_template_matches_its_fresh_plan_bitwise`] with every key
/// descriptor and template shape hashed to `0` (`kj::COLLIDE`): the subjects and both
/// structures share one bucket, and the exact comparison alone serves the twin from the
/// template and gives the other structure a plan of its own, bit for bit.
#[test]
fn templates_in_one_hash_bucket_are_told_apart_by_their_content() {
    use crate::parser::model_parser::rhs_regcode::kernel_jit as kj;
    kj::COLLIDE.with(|c| c.set(true));
    let doses = |at: &[f64]| -> Vec<DoseEvent> {
        at.iter()
            .map(|&t| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0))
            .collect()
    };
    let times = [1.1, 4.2, 8.3, 14.4, 20.5];
    let cf = parse_model_string(ONE_CPT_ORAL).unwrap();
    let a = subject(&cf, doses(&[0.0, 12.0]), &times, &[0.1, -0.1, 0.2]);
    let flip = subject(&cf, doses(&[0.0, 30.0]), &times, &[0.1, -0.1, 0.2]);
    let (n, k) = templated_twin(&cf, &a, &flip);
    assert!(k < n, "the other structure is planned: {k} of {n}");
    let (n, k) = templated_twin(&cf, &a, &other_data(&a));
    assert!(k > 0 && k == n, "the twin is templated: {k} of {n}");
    kj::COLLIDE.with(|c| c.set(false));
}

// ---- value-only objective through the plan (#2786, A1/A2) ---------------------------------

/// The Michaelis–Menten (nonlinear) ODE of the fit benchmarks: no exact linear path, so its
/// plan drives a solver loop.
const MM_ODE: &str = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(40.0, 1.0, 500.0)
  theta TVKA(1.0, 0.01, 10.0)
  theta TVKM(5.0, 0.0, 100.0)
  omega ETA_CL ~ 0.15
  omega ETA_V ~ 0.15
  omega ETA_KA ~ 0.20
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
  KM = TVKM
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot) = -KA * depot
  d/dt(central) = KA * depot - ((CL * KM / (KM + central / V)) / V) * central
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// The plan's value-only predictions at several `eta` against the native predictor's, bit for
/// bit; returns whether the plan served every point (`plan_values`). `native` is the
/// production predictor at `(theta, eta)`.
fn plan_values_vs_native(
    model: &CompiledModel,
    s: &Subject,
    layout: Option<&crate::stats::stacked_layout::StackedLayout>,
    etas: &[Vec<f64>],
    native: &dyn Fn(&[f64]) -> Vec<f64>,
) -> bool {
    let (served, worst) = plan_values_ulps(model, s, layout, etas, native);
    assert_eq!(worst, 0, "plan and native differ by {worst} ulp");
    served
}

/// [`plan_values_vs_native`]'s measurement: whether the plan served every point, and the
/// largest gap in ulps over every served row (a NaN or infinite row is a failure here).
fn plan_values_ulps(
    model: &CompiledModel,
    s: &Subject,
    layout: Option<&crate::stats::stacked_layout::StackedLayout>,
    etas: &[Vec<f64>],
    native: &dyn Fn(&[f64]) -> Vec<f64>,
) -> (bool, i64) {
    let theta = &moved_params(model).theta;
    let (mut served, mut worst) = (true, 0i64);
    for eta in etas {
        let mut got = Vec::new();
        let ok = plan_values(model, s, layout, theta, eta, &mut got);
        served &= ok;
        if !ok {
            continue;
        }
        let want = native(eta);
        assert_eq!(got.len(), want.len());
        for (j, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!(a.is_finite() && b.is_finite(), "row {j}: {a} vs {b}");
            worst = worst.max((a.to_bits() as i64 - b.to_bits() as i64).abs()); // nan-ok: integer ULP distance, no NaN
        }
    }
    (served, worst)
}

fn multidose_subject(model: &CompiledModel, times: &[f64], n_eta: usize) -> Subject {
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
    ];
    subject(model, doses, times, &vec![0.05; n_eta])
}

/// Which model classes the plan's value pass serves, and that where it serves it is the native
/// predictor's bits (multi-dose subjects, so a later dose lands on residual drug): a closed form
/// and a closed-form IOV subject with κ ≠ 0 are bit-equal; every `[odes]` model stays native
/// (`plan_values` declines it) — a linear system's recorded matrix exponential measured 1–2 ulp
/// off the native path, a nonlinear one's values differ more, and a linear one forced onto RK45
/// drives a solver loop. The objective's route (`plan_value_route`) is the closed forms.
#[test]
fn plan_values_are_bit_equal_to_the_native_predictor() {
    use crate::stats::likelihood::plan_value_route;
    let etas3 = vec![
        vec![0.0, 0.0, 0.0],
        vec![0.1, -0.05, 0.08],
        vec![-0.2, 0.15, -0.1],
    ];
    let times = [0.5, 1.0, 3.0, 12.6, 13.0, 18.0, 30.0];
    // Closed form: bit-equal and routed.
    let cf = parse_model_string(ONE_CPT_ORAL).unwrap();
    let s = multidose_subject(&cf, &TIMES, 3);
    let theta = moved_params(&cf).theta;
    let native = |e: &[f64]| crate::pk::compute_predictions_with_tv(&cf, &s, &theta, e);
    assert!(
        plan_values_vs_native(&cf, &s, None, &etas3, &native),
        "closed form served"
    );
    assert!(plan_value_route(&cf));
    // A linear `[odes]` model is served and routed (bit-equal; its shapes are pinned by
    // `linear_ode_plan_values_are_bit_equal_to_the_native_predictor`); a nonlinear one and a
    // linear one forced onto RK45 drive a solver loop, so the plan declines and they are not routed.
    let mut rk = parse_model_string(&lin_3state()).unwrap();
    rk.ode_spec.as_mut().unwrap().solver_opts.method = crate::ode::OdeMethod::Rk45;
    for (label, src, served) in [
        ("linear", parse_model_string(&lin_3state()).unwrap(), true),
        ("rk45", rk, false),
        ("mm", parse_model_string(MM_ODE).unwrap(), false),
    ] {
        let s = multidose_subject(&src, &times, if label == "mm" { 3 } else { 2 });
        let theta = moved_params(&src).theta;
        let native = |e: &[f64]| crate::pk::compute_predictions_with_tv(&src, &s, &theta, e);
        let etas = if label == "mm" {
            etas3.clone()
        } else {
            vec![vec![0.0; 2], vec![0.1, -0.05]]
        };
        assert_eq!(
            plan_values_vs_native(&src, &s, None, &etas, &native),
            served,
            "{label} served: {served}"
        );
        assert_eq!(plan_value_route(&src), served, "{label} routed: {served}");
    }
    // The lagged linear model is not routed whatever the plan serves.
    assert!(!plan_value_route(
        &parse_model_string(TWOCPT_DEPOT_LINEAR_ODE).unwrap()
    ));
    // IOV: stacked [η, κ₁, κ₂, κ₃], κ ≠ 0, doses landing on residual drug across occasions.
    for (src, served) in [(CF_IOV, true), (ODE_IOV, true)] {
        let m = parse_model_string(src).unwrap();
        let s = iov_subject(&m);
        let lay = crate::stats::stacked_layout::StackedLayout::sens_of(&m, &s).unwrap();
        let theta = moved_params(&m).theta;
        let stacked = vec![0.1, -0.05, 0.1, -0.05, -0.08, 0.04, 0.05, 0.06];
        let native = |e: &[f64]| {
            let k = vec![e[2..4].to_vec(), e[4..6].to_vec(), e[6..8].to_vec()];
            crate::pk::predict_iov(&m, &s, &theta, &e[..2], &k)
        };
        let etas = vec![stacked, vec![0.0; 8]];
        assert_eq!(
            plan_values_vs_native(&m, &s, Some(&lay), &etas, &native),
            served,
            "IOV model served: {served}"
        );
        assert_eq!(plan_value_route(&m), served);
    }
}

/// The IOV objective scores the plan's values: its value is the same bits as with the plan
/// declined (a recording open on this thread makes `plan_values` decline, so the native walk
/// answers), and the plan did serve the first.
#[test]
fn the_iov_objective_scores_plan_values_and_falls_back_to_native() {
    let m = parse_model_string(CF_IOV).unwrap();
    let s = iov_subject(&m);
    let params = moved_params(&m);
    let kappas = vec![vec![0.1, -0.05], vec![-0.08, 0.04], vec![0.05, 0.06]];
    let score = || {
        crate::stats::likelihood::individual_nll_iov(
            &m,
            &s,
            &params.theta,
            &[0.1, -0.05],
            &kappas,
            &params.omega,
            params.omega_iov.as_ref(),
            &params.sigma.values,
        )
    };
    let runs = kj_fwd::value_runs();
    let via_plan = score();
    assert!(
        kj_fwd::value_runs() > runs,
        "the plan served the value pass"
    );
    kj_fwd::begin();
    let native = score();
    let _ = kj_fwd::end(None);
    assert!(via_plan.is_finite());
    assert_eq!(via_plan.to_bits(), native.to_bits());
}

/// The outer gradient reads the forward the subject's inner solve ended on instead of running
/// the plan again at `η̂` (A5): adopted exactly once (a counter), the gradient is the same bits
/// as the one from a fresh forward, and a plan that drives a solver loop (no run logs kept) or a
/// point off the kept one (`θ` moved a ulp) is not adopted.
#[test]
fn the_outer_gradient_adopts_the_inner_solves_last_forward_bitwise() {
    use crate::parser::model_parser::rhs_regcode::kernel_jit as kj;
    let lin = parse_model_string(TWOCPT_DEPOT_LINEAR_ODE).unwrap();
    let s = multidose_subject(&lin, &[0.5, 1.0, 3.0, 12.6, 13.0, 18.0, 30.0], 3);
    let template = moved_params(&lin);
    let x = pack_params(&template);
    // The point the outer gradient unpacks, which is the one the fit's inner solve ran at.
    let params = unpack_params(&x, &template);
    let gradient = |eta: &[f64]| {
        subject_adjoint_gradient(&lin, &s, &template, &x, eta, Objective::Focei).expect("served")
    };
    let eta = ebe_for(&lin, &s, &params, 1e-10, true);
    let before = kj::adopts();
    let adopted = gradient(eta.as_slice());
    assert_eq!(kj::adopts(), before + 1, "the kept forward is adopted");
    // Taken: the second gradient at the same point runs the forward.
    let fresh = gradient(eta.as_slice());
    assert_eq!(kj::adopts(), before + 1, "a kept forward is read once");
    assert_eq!(adopted.len(), fresh.len());
    for (i, (a, b)) in adopted.iter().zip(&fresh).enumerate() {
        assert!(a.is_finite() && b.is_finite());
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "coordinate {i}: adopted {a} fresh {b}"
        );
    }
    // A point off the kept one (η̂ a ulp away) is not the kept forward.
    let _ = ebe_for(&lin, &s, &params, 1e-10, true);
    let mut off = eta.clone();
    off[0] = f64::from_bits(off[0].to_bits() + 1);
    let _ = gradient(off.as_slice());
    assert_eq!(kj::adopts(), before + 1, "a different η̂ is not adopted");
    // A solver-loop plan keeps no run logs, so its forward is never kept.
    let mm = parse_model_string(MM_ODE).unwrap();
    let sm = multidose_subject(&mm, &TIMES, 3);
    let tm = moved_params(&mm);
    let xm = pack_params(&tm);
    let pm = unpack_params(&xm, &tm);
    let em = ebe_for(&mm, &sm, &pm, 1e-10, true);
    let b = kj::adopts();
    subject_adjoint_gradient(&mm, &sm, &tm, &xm, em.as_slice(), Objective::Focei).expect("served");
    assert_eq!(kj::adopts(), b, "a loop plan is not adopted");
}

/// Every closed-form class the objective's plan route serves is the native predictor's bits:
/// multi-dose, LTBS, a bolus and an infusion on a two-compartment model with a correlated block,
/// a time-varying covariate, an estimated bioavailability under a rate-defined dose, and a
/// steady-state dose, each at η = 0 and at two moved points. The lagged models are the straddle:
/// the plan serves them but a few ulp off the native walk, so they are not routed.
#[test]
fn plan_values_are_bit_equal_across_the_closed_form_classes() {
    let dose = |t, amt, cmt, rate, ss, ii| DoseEvent::new(t, amt, cmt, rate, ss, ii);
    let two = vec![
        dose(0.0, 100.0, 1, 0.0, false, 0.0),
        dose(12.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let infusion = vec![
        dose(0.0, 100.0, 1, 20.0, false, 0.0),
        dose(8.0, 100.0, 1, 20.0, false, 0.0),
    ];
    let steady = vec![
        dose(0.0, 100.0, 1, 0.0, true, 12.0),
        dose(24.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let times = [0.5, 1.0, 3.0, 12.6, 13.0, 18.0, 30.0];
    let lag_only =
        ONE_CPT_ORAL_LAG_LTBS.replace("log(DV) ~ additive(ADD_ERR)", "DV ~ additive(ADD_ERR)");
    let ltbs_only = ONE_CPT_ORAL.replace(
        "DV ~ proportional(PROP_ERR)",
        "log(DV) ~ additive(PROP_ERR)",
    );
    let cases: Vec<(&str, &str, Option<Vec<DoseEvent>>, bool)> = vec![
        ("oral", ONE_CPT_ORAL, Some(two.clone()), true),
        ("lag+ltbs", ONE_CPT_ORAL_LAG_LTBS, Some(two.clone()), false),
        ("lag", &lag_only, Some(two.clone()), false),
        ("ltbs", &ltbs_only, Some(two.clone()), true),
        ("2cpt bolus", TWO_CPT_IV_COMBINED, Some(two.clone()), true),
        (
            "2cpt infusion",
            TWO_CPT_IV_COMBINED,
            Some(infusion.clone()),
            true,
        ),
        ("tv covariate", CF_TVCOV, None, true),
        ("F + rate", CF_ORAL_F, Some(infusion), true),
        ("steady state", ONE_CPT_ORAL, Some(steady), true),
    ];
    for (label, src, doses, routed) in cases {
        let m = parse_model_string(src).unwrap();
        let n = m.n_eta;
        let s = match doses {
            Some(d) => subject(&m, d, &times, &vec![0.05; n]),
            None => cf_tvcov_subject(&m),
        };
        let theta = moved_params(&m).theta;
        let native = |e: &[f64]| crate::pk::compute_predictions_with_tv(&m, &s, &theta, e);
        let etas = vec![
            vec![0.0; n],
            vec![0.1; n],
            (0..n).map(|k| 0.15 - 0.1 * k as f64).collect(),
        ];
        let (served, worst) = plan_values_ulps(&m, &s, None, &etas, &native);
        if routed {
            assert!(
                served && worst == 0,
                "{label}: served {served}, {worst} ulp"
            );
        } else {
            // The straddle: the lagged models are served by the plan but not bit-equal, so the
            // route must keep them native.
            assert!(worst > 0, "{label}: {worst} ulp; route it if it is equal");
            assert!(!crate::stats::likelihood::plan_value_route(&m));
        }
    }
}

// ---- linear `[odes]` plan values are the native path's bits (#2786) -----------------------

const LIN_1STATE: &str = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(30.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.06
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV * exp(ETA_V)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// [`TWOCPT_DEPOT_LINEAR_ODE`] without the lagtime (a lagged model is not routed).
fn lin_3state() -> String {
    TWOCPT_DEPOT_LINEAR_ODE
        .replace("  theta TVLAG(0.4, 0.01, 5.0)\n", "")
        .replace("  omega ETA_LAG ~ 0.05\n", "")
        .replace("  LAGTIME = TVLAG * exp(ETA_LAG)\n", "")
}

/// The plan's value-only predictions of every linear `[odes]` shape (1-, 2-, 3-state, infusion,
/// multi-dose, a time-varying covariate, IOV) against the native predictor's bits, at η = 0 and
/// at moved points, on the spectral path and on the Padé path.
#[test]
fn linear_ode_plan_values_are_bit_equal_to_the_native_predictor() {
    use crate::ode::linear::{SPECTRAL_HITS, SPECTRAL_OFF};
    use crate::parser::model_parser::rhs_regcode::kernel_jit as kj;
    let times = [0.5, 1.0, 3.0, 12.6, 13.0, 18.0, 30.0];
    let bolus = |cmt: usize| {
        vec![
            DoseEvent::new(0.0, 100.0, cmt, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, cmt, 0.0, false, 0.0),
        ]
    };
    let infusion = vec![
        DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0),
        DoseEvent::new(8.0, 100.0, 1, 50.0, false, 0.0),
        DoseEvent::new(16.0, 100.0, 1, 50.0, false, 0.0),
    ];
    let wt_at = |t: f64| 60.0 + 1.5 * t;
    let mut cases: Vec<(&str, CompiledModel, Subject, usize)> = Vec::new();
    let m = parse_model_string(LIN_1STATE).unwrap();
    let s = multidose_subject(&m, &times, 2);
    cases.push(("1-state", m, s, 2));
    let m = parse_model_string(ONE_CPT_ORAL_ODE).unwrap();
    let s = multidose_subject(&m, &times, 2);
    cases.push(("2-state", m, s, 2));
    let m = parse_model_string(&lin_3state()).unwrap();
    let s = multidose_subject(&m, &times, 2);
    cases.push(("3-state", m, s, 2));
    let m = parse_model_string(TWO_CPT_INF_COMBINED_BLOCK).unwrap();
    let s = subject(
        &m,
        infusion,
        &[1.0, 2.0, 4.0, 7.9, 9.0, 12.0, 17.0, 20.0, 30.0],
        &[0.05; 3],
    );
    cases.push(("infusion", m, s, 3));
    let m = parse_model_string(ODE_TVCOV).unwrap();
    let mut s = subject_shell(bolus(1), &times);
    s.covariates.insert("WT".into(), 70.0);
    s.obs_covariates = times
        .iter()
        .map(|&t| HashMap::from([("WT".to_string(), wt_at(t))]))
        .collect();
    s.dose_covariates = [0.0, 12.0]
        .iter()
        .map(|&t| HashMap::from([("WT".to_string(), wt_at(t))]))
        .collect();
    cases.push(("tvcov", m, s, 2));
    for pade in [false, true] {
        SPECTRAL_OFF.with(|c| c.set(pade));
        // A plan recorded under the other propagation would serve this one's points.
        kj::reset();
        for (label, m, s, ne) in &cases {
            let theta = moved_params(m).theta;
            let native = |e: &[f64]| crate::pk::compute_predictions_with_tv(m, s, &theta, e);
            let etas: Vec<Vec<f64>> = vec![
                vec![0.0; *ne],
                (0..*ne).map(|i| 0.1 - 0.07 * i as f64).collect(),
                (0..*ne).map(|i| -0.2 + 0.11 * i as f64).collect(),
            ];
            let hits = SPECTRAL_HITS.with(|c| c.get());
            let (served, worst) = plan_values_ulps(m, s, None, &etas, &native);
            let hits = SPECTRAL_HITS.with(|c| c.get()) - hits;
            eprintln!(
                "{label} pade={pade}: served {served} worst {worst} ulp (spectral segments {hits})"
            );
            // Forced onto Padé the plan is not a straight-line chunk and declines; the native
            // walk then answers (bit-equal by construction), which is the safe outcome.
            assert!(served || pade, "{label} pade={pade}: plan serves");
            assert_eq!(worst, 0, "{label} pade={pade}: {worst} ulp");
        }
    }
    SPECTRAL_OFF.with(|c| c.set(false));
    // KA = CL/V exactly and within the spectral gap bound (1e-4): the native path takes Padé
    // there, so a plan recorded on the spectral path must decline (its guard on the analysis's
    // verdict fails) rather than serve a spectral value; where it serves, it is the native bits.
    kj::reset();
    let (_, m, s, _) = &cases[1];
    let theta = moved_params(m).theta;
    let native = |e: &[f64]| crate::pk::compute_predictions_with_tv(m, s, &theta, e);
    let r = (theta[2] * theta[1] / theta[0]).ln();
    let mut served_near = Vec::new();
    for eta in [
        [0.0, 0.0],
        [0.1, -0.05],
        [r, 0.0],
        [r + 3e-5, 0.0],
        [r + 1e-3, 0.0],
    ] {
        let (served, worst) = plan_values_ulps(m, s, None, &[eta.to_vec()], &native);
        eprintln!("near-repeat {eta:?}: served {served} worst {worst} ulp");
        assert_eq!(worst, 0, "near-repeat {eta:?}: {worst} ulp");
        served_near.push(served);
    }
    assert!(
        served_near[0] && served_near[1],
        "the well-separated points are served"
    );
    // IOV: stacked [η, κ₁, κ₂, κ₃].
    let m = parse_model_string(ODE_IOV).unwrap();
    let s = iov_subject(&m);
    let lay = crate::stats::stacked_layout::StackedLayout::sens_of(&m, &s).unwrap();
    let theta = moved_params(&m).theta;
    let native = |e: &[f64]| {
        let k = vec![e[2..4].to_vec(), e[4..6].to_vec(), e[6..8].to_vec()];
        crate::pk::predict_iov(&m, &s, &theta, &e[..2], &k)
    };
    let etas = vec![
        vec![0.1, -0.05, 0.1, -0.05, -0.08, 0.04, 0.05, 0.06],
        vec![0.0; 8],
    ];
    let (served, worst) = plan_values_ulps(&m, &s, Some(&lay), &etas, &native);
    eprintln!("iov: served {served} worst {worst} ulp");
    assert!(served, "iov: plan serves");
    assert_eq!(worst, 0, "iov: {worst} ulp");
}

/// A weighted kappa (`weight = W`, #1031) is `exp(ETA + KAPPA/sqrt(W))`; its exact outer gradient
/// must match the reconverged-FD oracle at a non-unit weight as at `W = 1`, where the spelling
/// is also the plain `κ` (bit for bit, below).
fn weighted_kappa_src(weight: &str) -> String {
    format!(
        "[parameters]\n  theta TVCL(0.2, 0.001, 10.0)\n  theta TVV(10.0, 0.1, 500.0)\n  \
         theta TVKA(1.5, 0.01, 50.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  \
         kappa KAPPA_CL ~ 0.05{weight}\n  sigma PROP_ERR ~ 0.04\n[individual_parameters]\n  \
         CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n  KA = TVKA\n\
         [structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n\
         [error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  iov_column = OCC\n"
    )
}

fn weighted_kappa_subject(model: &CompiledModel) -> Subject {
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.5, 14.0, 18.0, 24.0];
    let mut s = subject_shell(doses, &times);
    s.occasions = times
        .iter()
        .map(|&t| if t < 12.0 { 1 } else { 2 })
        .collect();
    s.dose_occasions = vec![1, 2];
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
    s
}

#[test]
fn weighted_kappa_gradient_matches_reconverged_fd() {
    for weight in [" weight = 1", " weight = 2.5"] {
        let model = parse_model_string(&weighted_kappa_src(weight)).unwrap();
        assert!(model.has_weighted_kappa());
        let s = weighted_kappa_subject(&model);
        for interaction in [false, true] {
            let worst = parity_for(&model, &s, interaction);
            // Measured 2.8e-12 worst (Richardson reconverged FD); bound 1e-10 leaves ~35x.
            assert!(
                worst < 1e-10,
                "{weight}, interaction {interaction}: worst relative difference {worst:.2e}"
            );
        }
    }
}

/// `weight = 1` is the degenerate weight: the plan it records is the plain `κ`'s, so the exact
/// outer gradient is the same bits, not merely the same to rounding (the #1082 oracle, where one
/// L-BFGS step turns a 1-ulp gradient difference into a different iterate).
#[test]
fn unit_weight_gradient_is_the_plain_kappas_bit_for_bit() {
    let plain = parse_model_string(&weighted_kappa_src("")).unwrap();
    let unit = parse_model_string(&weighted_kappa_src(" weight = 1")).unwrap();
    let (sp, su) = (
        weighted_kappa_subject(&plain),
        weighted_kappa_subject(&unit),
    );
    for interaction in [false, true] {
        let grad = |m: &CompiledModel, s: &Subject| {
            let p = moved_params(m);
            let eta = ebe_for(m, s, &p, 1e-11, interaction);
            subject_adjoint_packed_gradient(m, s, &p, &pack_params(&p), eta.as_slice(), interaction)
                .expect("in scope")
        };
        let bits = |v: Vec<f64>| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(grad(&plain, &sp)),
            bits(grad(&unit, &su)),
            "interaction {interaction}"
        );
    }
}

const IV_LAG: &str = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(30.0, 1.0, 500.0)
  theta TVLAG(0.4, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_LAG ~ 0.1
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  ALAG = TVLAG * exp(ETA_LAG)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V, lagtime=ALAG)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// One plan of a lagged steady-state infusion, replayed at other lags (#2786 review #13). The
/// seed freezes `phase_v = phase.seen_value()` into the plan, with only the `phase > r` branch
/// (whether the newest window closes before the record) guarded and the `dur` break not moving
/// with the lag, so the replay has to carry the lag's shift by the arrival alone. `II = 3`,
/// a 2 h infusion (`r = 2`): `phase = 3 − lag` is above `r` for `lag < 1` and below it past.
/// The plan recorded at the first lag serves the second lag on its side of the branch and a
/// third lag across it (where its guard must decline and the walk re-record, never serve the
/// other side's shape): values against the native predictor, and the adjoint gradient against a
/// fresh recording on a re-parse (own plan identity) of the model at the same point. The value
/// pass records nothing (#3010), so its values are checked at every lag once the gradient has
/// recorded that lag's structure. Dies when the replay's phase or window end stops following the
/// lag.
#[test]
fn a_lagged_ss_infusion_plan_replays_at_other_lags_on_both_sides_of_the_window_end() {
    let model = parse_model_string(IV_LAG).unwrap();
    let params = moved_params(&model);
    let lag_of = |eta: &[f64]| params.theta[2] * eta[1].exp();
    // Same side (phase > r) twice, then across the branch.
    let etas = vec![
        vec![0.1, 0.1],
        vec![-0.05, 0.4],
        vec![0.1, 1.0],
        vec![0.1, 0.1],
    ];
    let lags: Vec<f64> = etas.iter().map(|e| lag_of(e)).collect();
    assert!(
        lags[0] < 1.0 && lags[1] < 1.0 && lags[2] > 1.0 && lags[2] < 3.0,
        "the straddle of phase > r, lag < II: {lags:?}"
    );
    let doses = vec![DoseEvent::new(0.0, 100.0, 1, 50.0, true, 3.0)];
    let s = subject(&model, doses, &[0.5, 1.0, 2.0, 3.5, 5.0], &etas[0]);
    // Values: the plan recorded at the first point is the native predictor's bits there; a point
    // its guard does not hold at is declined — never served the other side's shape — and the
    // value pass leaves it to the native predictor without recording (#3010).
    let theta = params.theta.clone();
    let native = |e: &[f64]| crate::pk::compute_predictions_with_tv(&model, &s, &theta, e);
    let (served, worst) = plan_values_ulps(&model, &s, None, &etas[..1], &native);
    assert!(served, "the first point is served");
    assert_eq!(worst, 0, "values: {worst} ulp from the native predictor");
    let (served, _) = plan_values_ulps(&model, &s, None, &etas[2..3], &native);
    assert!(!served, "across the branch the value pass declines");
    // Gradients: replay on the recording model, fresh on a re-parse.
    let x = pack_params(&params);
    let grad = |m: &CompiledModel, e: &[f64]| {
        subject_adjoint_packed_gradient(m, &s, &params, &x, e, true).expect("in scope")
    };
    let _ = grad(&model, &etas[0]);
    let mut gap = 0.0f64;
    for e in &etas[1..] {
        let replayed = grad(&model, e);
        let fresh = grad(&parse_model_string(IV_LAG).unwrap(), e);
        for (k, (a, b)) in replayed.iter().zip(&fresh).enumerate() {
            assert!(
                a.is_finite() && b.is_finite(),
                "x[{k}] at {e:?}: {a} vs {b}"
            );
            gap = gap.max((a - b).abs() / (1.0 + b.abs()));
        }
    }
    eprintln!("lagged SS infusion replay: worst gradient gap {gap:.3e}");
    // Measured 0.0 (bit-identical to a fresh recording); 1e-12 leaves room for another libm.
    assert!(
        gap < 1e-12,
        "replayed gradient off a fresh recording by {gap:e}"
    );
    // The gradient recorded every lag's structure, both sides of the branch: now the value pass
    // serves each lag from the plans, the native predictor's bits at every one.
    let (served, worst) = plan_values_ulps(&model, &s, None, &etas, &native);
    eprintln!("lagged SS infusion replay: served {served}, worst {worst} ulp");
    assert!(served, "every lag served once its structure is recorded");
    // Measured 0 ulp (Linux x86_64): the replay is the native predictor's bits at every lag.
    assert_eq!(worst, 0, "values: {worst} ulp from the native predictor");
}

/// A subject's plan is keyed by its design, `DoseEvent::implied` included (#2786 review #4).
/// An `ADDL`-implied dose takes its lag from its parent record and its `TIME` from the record
/// that governs it, so the same dose list with `implied` cleared is another subject whose
/// parameters, lag and predictions differ. The key once hashed everything of a dose but that
/// field, so the second of the pair was served the first's plan: a cache hit across subjects
/// that predict differently. Reachable, as measured: the explicit twin's plan values were
/// those of the implied dose's lag. Dies when `implied` leaves the key (the twin, served the
/// implied subject's plan, is off the native predictor by far more than an ulp).
#[test]
fn an_addl_implied_dose_and_its_explicit_twin_do_not_share_a_plan() {
    let model = parse_model_string(&addl_lag_src(false, false)).unwrap();
    let implied = addl_implied_lag_subject(&model, true);
    let mut explicit = implied.clone();
    explicit.doses[1].implied = None;
    let theta = moved_params(&model).theta;
    let etas = vec![vec![0.1, -0.05]];
    let mut worst = Vec::new();
    // The implied subject records and serves first, then its twin.
    for (label, s) in [("implied", &implied), ("explicit", &explicit)] {
        let native = |e: &[f64]| crate::pk::compute_predictions_with_tv(&model, s, &theta, e);
        let (served, ulps) = plan_values_ulps(&model, s, None, &etas, &native);
        assert!(served, "{label}: the plan serves");
        worst.push((label, ulps));
    }
    eprintln!("implied/explicit twins, plan vs native: {worst:?}");
    for (label, ulps) in worst {
        assert_eq!(ulps, 0, "{label}: served a plan that is not its own");
    }
    // The twins really do predict differently (else the pair cannot straddle the key).
    let p = |s: &Subject| crate::pk::compute_predictions_with_tv(&model, s, &theta, &etas[0]);
    let (a, b) = (p(&implied), p(&explicit));
    assert!(
        a.iter()
            .zip(&b)
            .any(|(x, y)| (x - y).abs() > 1e-3 * x.abs()),
        "the twins predict apart: {a:?} vs {b:?}"
    );
}

// ---- a recording at an extreme η is served, never a reason to give the subject up -----------

/// One-compartment oral models whose `[scaling]` divisor reads an individual parameter
/// (`obs_scale = V`, Form B): the ODE form, the closed form, and each with a κ on `CL`. At an
/// extreme η the divisor underflows (`V = TVV·exp(η) = 0`), which production reads as a `NaN`
/// prediction (`divide_by_obs_scale`).
fn scaled_src(ode: bool, iov: bool) -> String {
    let (cl, kappa) = if iov {
        ("exp(ETA_CL + KAPPA_CL)", "  kappa KAPPA_CL ~ 0.05\n")
    } else {
        ("exp(ETA_CL)", "")
    };
    let (structural, odes, scale) = if ode {
        (
            "ode(obs_cmt=central, states=[depot, central])",
            "[odes]\n  d/dt(depot) = -KA * depot\n  d/dt(central) = KA * depot - (CL/V) * central\n",
            "V",
        )
    } else {
        ("pk one_cpt_oral(cl=CL, v=V, ka=KA)", "", "V * 1.5")
    };
    format!(
        "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  theta TVV(30.0, 1.0, 500.0)\n  \
         theta TVKA(1.2, 0.05, 20.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.06\n{kappa}  \
         sigma PROP_ERR ~ 0.1 (sd)\n[individual_parameters]\n  CL = TVCL * {cl}\n  \
         V = TVV * exp(ETA_V)\n  KA = TVKA\n[structural_model]\n  {structural}\n{odes}\
         [scaling]\n  obs_scale = {scale}\n[error_model]\n  DV ~ proportional(PROP_ERR)\n"
    )
}

/// A subject whose first plan is recorded at an extreme η (the divisor `V` is 0 there) is not
/// condemned to finite differences: the plan records the `NaN` predictions as the branch it is,
/// and the same subject is then served at an ordinary η, matching the reconverged FD of the
/// objective. Mutation: restore `return None` where the divisor fails `0 < s < ∞` -> the
/// extreme call marks the subject bad (`W_PLAN_FD_FALLBACK`, "the recorder returned None") and
/// the parity below panics with "subject must be in adjoint scope". Four sites, one per engine
/// that applies an expression scale (ODE, ODE with κ, closed form, closed form with κ).
#[test]
fn a_scale_that_underflows_at_an_extreme_eta_does_not_condemn_the_subject() {
    use crate::parser::model_parser::rhs_regcode::kernel_jit as kj;
    for (ode, iov) in [(true, false), (true, true), (false, false), (false, true)] {
        let label = format!("ode={ode} iov={iov}");
        let model = tight(parse_model_string(&scaled_src(ode, iov)).unwrap());
        let n_stacked = model.n_eta + model.n_kappa;
        let eta_ref: Vec<f64> = (0..n_stacked).map(|k| 0.1 - 0.07 * k as f64).collect();
        let s = subject(
            &model,
            vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
            ],
            &TIMES,
            &eta_ref,
        );
        let params = moved_params(&model);
        let x = pack_params(&params);
        // `V = TVV·exp(-800)` is 0 and `CL` is denormal: the divisor is not a positive number.
        let extreme = vec![-800.0; n_stacked];
        let _ = subject_adjoint_packed_gradient(&model, &s, &params, &x, &extreme, true);
        let failed = kj::plan_failures(&model, std::slice::from_ref(&s));
        assert!(
            failed.is_none(),
            "{label}: the extreme η condemned the subject: {failed:?}"
        );
        // The same subject, at ordinary points, is served exactly. Measured below.
        let worst = parity(&model, &s);
        eprintln!("{label}: scaled-divisor parity {worst:.3e}");
        assert!(
            worst < 1e-8,
            "{label}: worst relative difference {worst:.2e}"
        );
        let failed = kj::plan_failures(&model, std::slice::from_ref(&s));
        assert!(failed.is_none(), "{label}: a plan failed later: {failed:?}");
    }
}

/// An `SS=1` dose recorded first at an extreme η — `CL = TVCL·exp(800) = ∞`, no finite trough, the
/// steady-state search fails at record time — is that point's failure, not the subject's: the
/// call is declined, no plan failure is noted, and the same subject is served at ordinary points
/// matching the reconverged FD. Mutation: make `fail_here` fail with `FailScope::Subject` in
/// `kjit_record` -> the extreme call marks the subject bad ("a steady-state call's record-time
/// search failed") and the parity panics with "subject must be in adjoint scope". The control
/// (`+800` on `CL` and `-800` on `V`) is a finite value path with a failing recording, which
/// stays a plan failure and is not asserted here.
#[test]
fn a_steady_state_with_no_finite_trough_at_an_extreme_eta_does_not_condemn_the_subject() {
    use crate::parser::model_parser::rhs_regcode::kernel_jit as kj;
    let model = tight(parse_model_string(ONE_CPT_ORAL_ODE).unwrap());
    let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)];
    let times = [0.5, 1.0, 2.0, 4.0, 8.0, 11.0];
    let s = subject(&model, doses, &times, &[0.1, -0.05]);
    let params = moved_params(&model);
    let x = pack_params(&params);
    let extreme = [800.0, 800.0];
    assert!(
        (model.pk_param_fn)(&params.theta, &extreme, &s.covariates, 0.0)
            .values
            .iter()
            .any(|v| !v.is_finite()),
        "the extreme point must be one with a non-finite individual parameter"
    );
    let _ = subject_adjoint_packed_gradient(&model, &s, &params, &x, &extreme, true);
    let failed = kj::plan_failures(&model, std::slice::from_ref(&s));
    assert!(
        failed.is_none(),
        "the extreme η condemned the subject: {failed:?}"
    );
    let worst = parity(&model, &s);
    eprintln!("ss after an extreme η: parity {worst:.3e}");
    // Measured 1.6e-9; 6x headroom.
    assert!(worst < 1e-8, "worst relative difference {worst:.2e}");
    let failed = kj::plan_failures(&model, std::slice::from_ref(&s));
    assert!(failed.is_none(), "a plan failed later: {failed:?}");
}

/// A line search overshoots to a point whose individual parameters are finite but whose
/// predictions overflow (`CL = TVCL·e^700`, `V = TVV·e^-30`: `CL` is `~1e304`, the
/// recording's predictions are `NaN`, the objective there the repel sentinel): the plan recorded
/// there does not serve at its own point, and that is the point's failure, not the subject's.
/// The call is declined, no plan failure is noted, and the next sane point is recorded afresh
/// and served by the plan (matching the reconverged FD) instead of the finite differences the
/// subject would be condemned to for the rest of the fit. Control: a plan that fails to serve at
/// a sane point is still condemned (`mark_bad` stays for the structural case).
/// Mutation: drop `sane &&` from the `mark_bad` condition in `with_plan_in` -> the extreme call
/// marks the subject bad and the sane point after it is declined.
#[test]
fn a_variant_that_overflows_at_a_trial_point_does_not_condemn_the_subject() {
    use crate::parser::model_parser::rhs_regcode::kernel_jit as kj;
    let model = tight(parse_model_string(ONE_CPT_ORAL_ODE).unwrap());
    let doses = || vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    let s = subject(&model, doses(), &TIMES, &[0.1, -0.05]);
    let params = moved_params(&model);
    let extreme = [EXTREME.0, EXTREME.1];
    assert!(
        (model.pk_param_fn)(&params.theta, &extreme, &s.covariates, 0.0)
            .values
            .iter()
            .all(|v| v.is_finite()),
        "fixture: the individual parameters are finite at the trial point"
    );
    // A serve that declines wherever the predictions are not finite, as every consumer does.
    let declines = &mut |_: &PlanArc| None::<()>;
    assert!(super::with_plan(&model, &s, None, &params.theta, &extreme, declines).is_none());
    let failed = kj::plan_failures(&model, std::slice::from_ref(&s));
    assert!(
        failed.is_none(),
        "the overflowing trial point condemned the subject: {failed:?}"
    );
    // The next point of the line search, sane: served by a plan.
    let worst = parity(&model, &s);
    eprintln!("after an overflowing trial point: parity {worst:.3e}");
    assert!(worst < 1e-6, "worst relative difference {worst:.2e}");
    let failed = kj::plan_failures(&model, std::slice::from_ref(&s));
    assert!(failed.is_none(), "a plan failed later: {failed:?}");
    // Control: the same declining serve at a sane point is a plan that never serves.
    let t = subject(
        &model,
        vec![DoseEvent::new(0.0, 101.0, 1, 0.0, false, 0.0)],
        &TIMES,
        &[0.1, -0.05],
    );
    let sane = [0.1, -0.05];
    assert!(super::with_plan(&model, &t, None, &params.theta, &sane, declines).is_none());
    assert!(
        kj::plan_failures(&model, std::slice::from_ref(&t)).is_some(),
        "a plan that fails to serve at a sane point is condemned"
    );
}

/// The trial point of the test above, set by what the exploration found to overflow.
const EXTREME: (f64, f64) = (700.0, -30.0);

// ---- Plan failures at a trial point, and the variant cache (#2786 review) ----------------

/// The key `with_plan` files a subject's plan under, at `n` stacked random effects.
fn plan_key(
    model: &CompiledModel,
    s: &Subject,
    n: usize,
    n_theta: usize,
) -> crate::parser::model_parser::rhs_regcode::kernel_jit::SubjectKey {
    crate::parser::model_parser::rhs_regcode::kernel_jit::SubjectKey::new(model, s, n, n_theta)
        .with_layout(None)
}

/// A recording that fails at one finite trial point — here an `SS=1` dose with
/// `KA = TVKA·e^700 ≈ 1e304`, whose steady-state search finds no state — is that
/// point's failure, not the subject's: `fail_here` carries its scope, so the call is declined
/// with nothing cached and nothing marked, and the next sane point is served by a plan.
/// Mutation: make `fail_here` fail with `FailScope::Subject` -> the trial point marks the
/// subject bad (the key is `Bad`, not `Missing`) and the sane call after it is declined.
#[test]
fn a_finite_trial_eta_with_no_steady_state_declines_the_point_not_the_subject() {
    use crate::parser::model_parser::rhs_regcode::kernel_jit as kj;
    let model = tight(parse_model_string(MM_ODE).unwrap());
    let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)];
    let times = [0.5, 1.0, 2.0, 4.0, 8.0, 11.0];
    let s = subject(&model, doses, &times, &[0.1, -0.05, 0.02]);
    let params = moved_params(&model);
    let trial = [0.0, 0.0, 700.0];
    assert!(
        (model.pk_param_fn)(&params.theta, &trial, &s.covariates, 0.0)
            .values
            .iter()
            .all(|v| v.is_finite()),
        "fixture: the individual parameters are finite at the trial point"
    );
    let key = plan_key(&model, &s, 3, params.theta.len());
    let served = &mut |_: &PlanArc| Some(());
    assert!(super::with_plan(&model, &s, None, &params.theta, &trial, served).is_none());
    assert!(
        matches!(kj::lookup(&key), kj::Lookup::Missing),
        "the trial point recorded nothing and marked nothing"
    );
    assert!(kj::plan_failures(&model, std::slice::from_ref(&s)).is_none());
    // The next, sane point is recorded and served by a plan.
    let sane = [0.1, -0.05, 0.02];
    assert!(super::with_plan(&model, &s, None, &params.theta, &sane, served).is_some());
    assert!(matches!(kj::lookup(&key), kj::Lookup::Found(_)));
    assert!(kj::plan_failures(&model, std::slice::from_ref(&s)).is_none());
}

/// The same for an overflowing κ: a trial point whose κ of one occasion is 800 (`exp(κ) = ∞`)
/// failed the recording while a check on η and the baseline covariates alone read the
/// parameters as finite, and condemned the subject. The scope the failure carries decides, not
/// a re-evaluation of the parameters over part of the stacked vector.
/// Mutation: make `fail_here` fail with `FailScope::Subject` -> the stacked trial point marks
/// the subject bad.
#[test]
fn an_overflowing_kappa_declines_the_point_not_the_subject() {
    use crate::parser::model_parser::rhs_regcode::kernel_jit as kj;
    let model = tight(parse_model_string(ODE_IOV).unwrap());
    let mut s = iov_subject(&model);
    s.doses[0] = DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0);
    let params = moved_params(&model);
    // [η (2), κ of occasions 1..3 (2 each)].
    let trial = [0.1, -0.05, 800.0, 800.0, 0.0, 0.0, 0.0, 0.0];
    let key = plan_key(&model, &s, trial.len(), params.theta.len());
    let served = &mut |_: &PlanArc| Some(());
    assert!(super::with_plan(&model, &s, None, &params.theta, &trial, served).is_none());
    assert!(
        matches!(kj::lookup(&key), kj::Lookup::Missing),
        "the trial point recorded nothing and marked nothing"
    );
    assert!(kj::plan_failures(&model, std::slice::from_ref(&s)).is_none());
    let sane = [0.1, -0.05, 0.1, -0.05, -0.08, 0.04, 0.05, 0.06];
    assert!(super::with_plan(&model, &s, None, &params.theta, &sane, served).is_some());
    assert!(kj::plan_failures(&model, std::slice::from_ref(&s)).is_none());
}

/// Trial points whose predictions overflow record variants that cannot serve, and such a
/// variant is not cached: the serving variant stays the subject's, however many insane probes
/// come (more than `MAX_VARIANTS`, which would have evicted it).
/// Mutation: add the variant before `serves` has run (as `kjit_record` did) -> the cache holds
/// `MAX_VARIANTS` non-serving variants and the serving one is gone.
#[test]
fn insane_probes_do_not_evict_the_serving_variant() {
    use crate::parser::model_parser::rhs_regcode::kernel_jit as kj;
    let model = tight(parse_model_string(ONE_CPT_ORAL_ODE).unwrap());
    let s = subject(
        &model,
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &TIMES,
        &[0.1, -0.05],
    );
    let params = moved_params(&model);
    let key = plan_key(&model, &s, 2, params.theta.len());
    let sane = [0.1, -0.05];
    assert!(super::with_plan(&model, &s, None, &params.theta, &sane, &mut |_| Some(())).is_some());
    let serving = match kj::lookup(&key) {
        kj::Lookup::Found(ks) => ks[0].clone(),
        _ => panic!("fixture: the sane point cached a variant"),
    };
    let extreme = [EXTREME.0, EXTREME.1];
    for _ in 0..(kj::MAX_VARIANTS + 3) {
        // A serve that declines wherever the predictions are not finite, as every consumer does.
        let r = super::with_plan(
            &model,
            &s,
            None,
            &params.theta,
            &extreme,
            &mut |_| None::<()>,
        );
        assert!(r.is_none());
    }
    match kj::lookup(&key) {
        kj::Lookup::Found(ks) => {
            assert_eq!(ks.len(), 1, "no insane variant was cached");
            assert!(std::sync::Arc::ptr_eq(&ks[0], &serving));
        }
        _ => panic!("the subject was condemned"),
    }
}

/// The reduced basis's plan takes the same lookup as every other: a model-level bad mark
/// declines it (and marks the subject) instead of recording again.
/// Mutation: drop the `model_bad` arm from `plan_or_record` -> the plan is recorded and served.
#[test]
fn the_reduced_basis_plan_honours_a_model_level_bad_mark() {
    use crate::parser::model_parser::rhs_regcode::kernel_jit as kj;
    let model = tight(parse_model_string(ONE_CPT_ORAL_ODE).unwrap());
    let s = subject(
        &model,
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &TIMES,
        &[0.1, -0.05],
    );
    let params = moved_params(&model);
    let eta = [0.1, -0.05];
    let _g = kj::register(&model, std::slice::from_ref(&s));
    let shape = kernel::variant_of(&(Vec::<usize>::new(), None::<usize>));
    let key = kj::SubjectKey::new(&model, &s, 2, params.theta.len())
        .with_kind(9)
        .with_variant(shape);
    kj::mark_model_bad(&key, "a code path the recorder cannot take");
    let mut called = false;
    let r = super::with_plan_nn(&model, &s, &params.theta, &eta, &[], None, &mut |_| {
        called = true;
        Some(())
    });
    assert!(r.is_none() && !called, "declined without recording");
    assert!(matches!(kj::lookup(&key), kj::Lookup::Bad));
}

/// An overflowing η (`exp(800) = ∞` on both parameters) never condemns the subject, on a plain
/// model or a steady-state one, and whether the predictions there happen to be finite or not
/// (#2786 round 2, finding 2): with `CL = ∞` the predictions at `t > 0` are exactly `0`, so a
/// check on the final predictions alone calls the point sane, and the guard's own `inf − inf`
/// (or a predictor that bails on the overflowed parameter) then condemned the subject. The
/// sanity is read off the recording's own values (`Recording::finite`, `FailScope::Repelled`).
/// Afterwards a sane point is served by a plan and nothing is noted. Mutation: take `sane` from
/// the predictions alone in `kjit_record` (drop `&& finite`) and the `nonfinite` -> `Repelled`
/// mapping in `end` -> the plain row marks the subject bad.
#[test]
fn an_overflowing_eta_does_not_condemn_a_plain_or_a_steady_state_subject() {
    use crate::parser::model_parser::rhs_regcode::kernel_jit as kj;
    let mut n = 0.0;
    // Overflow on CL, on V, on both, and the opposite signs: predictions are `NaN`, `0` or
    // finite, depending on which.
    for extreme in [
        [800.0, 800.0],
        [800.0, 0.0],
        [0.0, 800.0],
        [800.0, -800.0],
        [-800.0, 800.0],
    ] {
        for (label, ss) in [("plain", false), ("steady state", true)] {
            n += 1.0;
            let model = tight(parse_model_string(ONE_CPT_ORAL_ODE).unwrap());
            // Each subject its own (the dose differs), so a condemnation names its point.
            let amt = 100.0 + n;
            let dose = if ss {
                DoseEvent::new(0.0, amt, 1, 0.0, true, 12.0)
            } else {
                DoseEvent::new(0.0, amt, 1, 0.0, false, 0.0)
            };
            let times = [0.5, 1.0, 2.0, 4.0, 8.0, 11.0];
            let s = subject(&model, vec![dose], &times, &[0.1, -0.05]);
            let params = moved_params(&model);
            let label = format!("{label} at {extreme:?}");
            // Declined whichever way the consumer scores it there.
            for serves in [
                &mut (|_: &PlanArc| None::<()>) as &mut dyn FnMut(&PlanArc) -> Option<()>,
                &mut |_: &PlanArc| Some(()),
            ] {
                let _ = super::with_plan(&model, &s, None, &params.theta, &extreme, serves);
            }
            let key = plan_key(&model, &s, 2, params.theta.len());
            assert!(
                !matches!(kj::lookup(&key), kj::Lookup::Bad),
                "{label}: the overflow condemned the subject"
            );
            // The next, sane point is served by a plan and nothing is condemned.
            let worst = parity(&model, &s);
            assert!(
                worst < 1e-6,
                "{label}: worst relative difference {worst:.2e}"
            );
            let failed = kj::plan_failures(&model, std::slice::from_ref(&s));
            assert!(failed.is_none(), "{label}: a plan failed later: {failed:?}");
        }
    }
}

/// G11: a pass that finds an indefinite `H` off the mode no longer declines to the held-EBE or
/// reconverged-FD score: it walks `eta` to the mode with damped Newton steps on its own exact
/// gradient and `H`, and serves the exact gradient there. Fixture: the grid point of
/// `a_mode_system_decline_is_named` (exact `H` indefinite). FOCEI is compared with the
/// reconverged-FD gradient, and FOCEI and Laplace with the gradient from the converged EBE.
/// Mutation: with the polish budget at 0 the first assertion fails (the pass declines).
#[test]
fn an_off_mode_indefinite_point_is_polished_and_served() {
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
    let grid = [-2.5, -1.5, -0.75, 0.75, 1.5, 2.5];
    let mut eta = None;
    'find: for &a in &grid {
        for &b in &grid {
            for &c in &grid {
                let e = [a, b, c];
                let h = exact_inner_hessian(&one, &s, params, &e).expect("recorded");
                if h.cholesky().is_none() {
                    eta = Some(e);
                    break 'find;
                }
            }
        }
    }
    let eta = eta.expect("a point where the exact H is indefinite");
    let ebe = find_ebe(&one, &s, params, 500, 1e-11, None, None, 0).eta;
    let laplace = Objective::Agq {
        n_agq: 1,
        exact: true,
    };
    for objective in [Objective::Focei, laplace] {
        let polished = subject_adjoint_gradient(&one, &s, params, &x, &eta, objective)
            .expect("the off-mode indefinite point is served, not declined");
        let truth = subject_adjoint_gradient(&one, &s, params, &x, ebe.as_slice(), objective)
            .expect("served at the EBE");
        let mut worst = 0.0_f64;
        for (a, b) in polished.iter().zip(&truth) {
            assert!(a.is_finite() && b.is_finite());
            worst = worst.max((a - b).abs() / (1.0 + b.abs())); // nan-ok: finite asserted above
        }
        eprintln!("{objective:?}: polished vs EBE gradient, worst rel {worst:.3e}");
        assert!(worst < GAP_POLISHED_VS_EBE, "{objective:?}: {worst:.3e}");
    }
    // FOCEI against the reconverged finite difference of the objective.
    let polished = subject_adjoint_gradient(&one, &s, params, &x, &eta, Objective::Focei).unwrap();
    let mut worst = 0.0_f64;
    for k in 0..x.len() {
        let fd = |h: f64| {
            let (mut xp, mut xm) = (x.clone(), x.clone());
            xp[k] += h;
            xm[k] -= h;
            (reconverged_f_for(&one, &s, params, &xp, true)
                - reconverged_f_for(&one, &s, params, &xm, true))
                / (2.0 * h)
        };
        let rich = (4.0 * fd(1e-3) - fd(2e-3)) / 3.0;
        assert!(rich.is_finite() && polished[k].is_finite());
        worst = worst.max((polished[k] - rich).abs() / (1.0 + rich.abs())); // nan-ok: finite asserted above
    }
    eprintln!("FOCEI polished vs reconverged FD, worst rel {worst:.3e}");
    assert!(worst < GAP_POLISHED_VS_FD, "{worst:.3e}");
    // The mode-system flag is clear when the polish serves the subject.
    let mut out = ModeOut::default();
    let pt = PackedPoint::new(params, &x).expect("packed point");
    let g = subject_adjoint_gradient_and_mode_at(&one, &s, &pt, &eta, Objective::Focei, &mut out);
    assert!(g.is_some() && !out.mode_system_declined && out.polish.is_none());
    // The gradient and the objective are of the same `η̂`: the pass hands back the `η̂` it
    // differentiated at, and the objective it differentiated there is the production
    // objective at that `η̂` (not at the off-mode one it was handed).
    let moved = out
        .eta_polished
        .clone()
        .expect("the polished eta is handed back");
    assert!(
        moved.iter().zip(&eta).any(|(a, b)| (a - b).abs() > 0.1),
        "the polish moved eta"
    );
    let at = |e: &[f64]| {
        let e = DVector::from_column_slice(e);
        let hm = oracle_jacobian(&one, &s, &params.theta, e.as_slice());
        crate::stats::likelihood::foce_subject_nll_cov(
            &one,
            &s,
            &params.theta,
            &e,
            &hm,
            &params.omega,
            &params.sigma.values,
            one.residual_cov_with(&params.residual_correlations, &params.theta),
            true,
        )
    };
    let f_pass = out.objective.expect("the pass objective");
    let f_polished = at(&moved);
    let f_handed = at(&eta);
    eprintln!(
        "F pass {f_pass:.12e}, production at polished {f_polished:.12e}, at handed {f_handed:.12e}"
    );
    assert!(
        (f_pass - f_polished).abs() <= 1e-9 * (1.0 + f_polished.abs()),
        "pass objective {f_pass} vs production at the polished eta {f_polished}"
    );
    assert!(
        (f_handed - f_polished).abs() > 1e-3,
        "the handed-in eta is a different objective value, so the check discriminates"
    );
}

// Measured 9.6e-11 (Laplace; FOCEI 2.0e-11): ten times that.
const GAP_POLISHED_VS_EBE: f64 = 1e-9;
// Measured 1.0e-9 (Richardson central FD, reconverged EBE): ten times that.
const GAP_POLISHED_VS_FD: f64 = 1e-8;

/// G14: the polish step is clipped to `POLISH_RADIUS`. On `H = diag(1, 0)` (a flat direction)
/// the reflected Newton step along it is `g / floor = 1e8`; clipped, it keeps its direction at
/// norm 1. Mutation: delete the clip in `newton_polish_step` -> the norm assertion fails.
#[test]
fn the_polish_step_is_clipped_to_the_trust_radius() {
    let h = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 0.0]));
    let PolishStep { dir, gn, .. } = polish_proposal(&h, &[0.5, 1.0]).expect("finite");
    let norm = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt();
    assert!((norm - POLISH_RADIUS).abs() < 1e-12, "norm {norm}");
    assert_eq!(gn, 1.0);
    // Direction kept: the unclipped step is (-0.5, -1e8).
    assert!((dir[0] / dir[1] - 0.5e-8).abs() < 1e-12, "{dir:?}");
    // Below the radius the step is the plain damped Newton one.
    let small = polish_proposal(&h, &[0.5, 0.0]).unwrap().dir;
    assert_eq!(small, vec![-0.5, 0.0]);
}

/// G14: a far-off start that diverges without the radius. The synthetic pass serves only
/// inside `‖η‖ < 10` (as an overflowing model does) and proposes the damped Newton step of a
/// landscape that is linear along a flat direction. With the clip the walk takes unit steps
/// and ends on the last point it served; without it the first step is `1e8` long, the pass
/// declines there, and the polish returns `None`.
/// Mutation: delete the clip -> `polish_mode` returns `None`.
#[test]
fn a_far_off_polish_start_needs_the_trust_radius() {
    let h = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 0.0]));
    let grad = |at: &[f64]| vec![at[0] + 0.5, 1.0];
    let eta0 = [0.0, 0.0];
    let mut out = ModeOut {
        polish: polish_proposal(&h, &grad(&eta0)),
        ..ModeOut::default()
    };
    let r = polish_mode(&eta0, &mut out, POLISH_STEPS, |at, out| {
        if at.iter().map(|v| v * v).sum::<f64>().sqrt() > 10.0 {
            return None;
        }
        out.polish = polish_proposal(&h, &grad(at));
        out.objective = Some(0.0);
        Some(vec![0.0])
    });
    assert!(r.is_some(), "the walk served a point inside the map");
    let at = out.eta_polished.expect("the last served eta");
    assert!(
        at.iter().map(|v| v * v).sum::<f64>().sqrt() <= 10.0,
        "{at:?}"
    );
}

/// G15: the walk leaves a saddle region. `l(x) = x^4/4 - x^2/2` has `g = x^3 - x`, `H = 3x^2 - 1`;
/// from `x = 0.3` (`H < 0`) the reflected Newton step goes downhill in `l` toward the mode at
/// `x = 1`, and `|g|` grows along it (to `0.385` at `x = 1/sqrt 3`) however short the step, so a
/// line search on `|g|` alone halves six times and gives the subject up. The directional merit
/// (`descends`) accepts it. The pass declines (`None`, carrying the proposal) until `|g|` is at the
/// target, as a mode-system decline does.
/// Mutation: drop `descends` from `polish_mode`'s acceptance -> `polish_mode` returns `None`.
#[test]
fn the_polish_leaves_a_saddle_region_where_the_gradient_norm_grows() {
    let grad = |x: f64| vec![x * x * x - x];
    let hess = |x: f64| DMatrix::from_element(1, 1, 3.0 * x * x - 1.0);
    let eta0 = [0.3];
    let mut out = ModeOut {
        polish: polish_proposal(&hess(0.3), &grad(0.3)),
        ..ModeOut::default()
    };
    let r = polish_mode(&eta0, &mut out, POLISH_STEPS, |at, out| {
        let x = at[0];
        out.polish = polish_proposal(&hess(x), &grad(x)).filter(|p| !(p.gn <= POLISH_TARGET_TOL));
        out.objective = Some(0.0);
        out.polish.is_none().then(|| vec![0.0])
    });
    assert!(r.is_some(), "the walk reached the mode");
    let at = out.eta_polished.expect("the mode");
    assert!((at[0] - 1.0).abs() < 1e-6, "{at:?}");
    // The straddle: `|g|` really does grow along the first step, so the old merit refused it.
    let step = polish_proposal(&hess(0.3), &grad(0.3)).unwrap().dir[0];
    assert!(grad(0.3 + 0.01 * step)[0].abs() > grad(0.3)[0].abs());
}

// ---- G14: the exact inner Hessian's declines, and the FD Hessians they fed (deleted) ----------

/// `exact_inner_hessian_or_decline`'s reason on a width that is not the subject's: the one shape
/// no consumer builds (the quadrature's mode is `stack_mode(η̂, κ̂)`, the subject's own stacked
/// width; nested levels, whose compact axes differ, are refused for every quadrature method by
/// `E_NESTED_LEVEL_METHOD_UNSUPPORTED`). Mutation: drop the `Width` return -> panics on index.
#[test]
fn a_wrong_width_eta_has_no_exact_inner_hessian() {
    let m = parse_model_string(ONE_CPT_ORAL).unwrap();
    let s = subject(
        &m,
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &TIMES,
        &[0.1, -0.05, 0.2],
    );
    let p = &m.default_params;
    assert_eq!(
        exact_inner_hessian_or_decline(&m, &s, p, &[0.0; 4]).err(),
        Some(HessianDecline::Width)
    );
    assert!(exact_inner_hessian_or_decline(&m, &s, p, &[0.0; 3]).is_ok());
}

/// The property that makes a deleted FD Hessian safe: wherever the exact Hessian declines, the
/// value path does not score the point either (non-finite, or the sentinel). Checked at an
/// overflowing η on a closed form and at a zero Ω (the parser accepts `omega ~ 0`, whose
/// Cholesky diagonal is the `Omega` decline). Mutation: decline at a point the value path
/// scores -> the assertion names it.
#[test]
fn where_the_exact_hessian_declines_the_objective_repels() {
    let m = parse_model_string(ONE_CPT_ORAL).unwrap();
    let s = subject(
        &m,
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &TIMES,
        &[0.1, -0.05, 0.2],
    );
    let usable = |m: &CompiledModel, s: &Subject, p: &ModelParameters, eta: &[f64]| {
        crate::stats::likelihood::is_usable_subject_nll(crate::stats::likelihood::individual_nll(
            m,
            s,
            &p.theta,
            eta,
            &p.omega,
            &p.sigma.values,
        ))
    };
    let p = &m.default_params;
    let mut n_declined = 0;
    for extreme in [
        [800.0, 800.0, 800.0],
        [-800.0, 800.0, 0.0],
        [0.0, 0.0, 800.0],
    ] {
        let why = exact_inner_hessian_or_decline(&m, &s, p, &extreme);
        let ok = usable(&m, &s, p, &extreme);
        eprintln!(
            "eta {extreme:?}: hessian declined = {}, objective usable = {ok}",
            why.is_err()
        );
        if why.is_err() {
            n_declined += 1;
            assert!(
                !ok,
                "{extreme:?}: the Hessian declines ({why:?}) where the objective scores"
            );
        }
    }
    eprintln!("{n_declined} of 3 overflow points decline");
    // A zero variance: the parser takes it but floors it (the Cholesky diagonal stays positive),
    // so the `Omega` decline is not reachable from a parsed model; a hand-built zero diagonal
    // is the only way to it.
    let zero = ONE_CPT_ORAL.replacen("omega ETA_CL ~ 0.09", "omega ETA_CL ~ 0.0", 1);
    assert_ne!(zero, ONE_CPT_ORAL, "the fixture names ETA_CL");
    let mz = parse_model_string(&zero).expect("a zero omega parses");
    assert!(
        mz.default_params.omega.chol[(0, 0)] > 0.0,
        "floored by the parser"
    );
    assert!(exact_inner_hessian_or_decline(&mz, &s, &mz.default_params, &[0.0; 3]).is_ok());
    let mut hand = m.default_params.clone();
    hand.omega.chol[(0, 0)] = 0.0;
    assert_eq!(
        exact_inner_hessian_or_decline(&m, &s, &hand, &[0.0; 3]).err(),
        Some(HessianDecline::Omega)
    );
}

/// G14: FREM rows under κ are served by the exact inner Hessian, as the conditional objective
/// the quadrature integrates scores them. Fixture: `frem_iov_ode`'s refused subject (a FREM
/// weight row on an ODE + IOV model) at its quadrature mode, against central second differences
/// of `individual_nll_iov_with_scratch` (the integrand `Stack::nll_at` evaluates), Richardson in
/// the step. Mutation: restore the `iov_frem_mismatch` decline in
/// `exact_inner_hessian_or_decline` -> the `expect` fails.
#[test]
fn iov_frem_rows_have_the_exact_inner_hessian_of_the_quadrature_integrand() {
    use crate::estimation::inner_optimizer::find_ebe_for_stage;
    use crate::types::{EstimationMethod, FitOptions};
    let model = frem_iov_ode();
    let (_, refused) = frem_iov_pair(&model);
    let params = moved_params(&model);
    let mut o = FitOptions {
        method: EstimationMethod::Laplace,
        ..FitOptions::default()
    };
    o.n_agq = 1;
    let r = find_ebe_for_stage(&model, &refused, &params, 200, 1e-10, None, None, 0, &o);
    let mut b: Vec<f64> = r.eta.iter().copied().collect();
    for k in &r.kappas {
        b.extend(k.iter().copied());
    }
    let h = exact_inner_hessian_or_decline(&model, &refused, &params, &b)
        .expect("FREM rows under kappa are served");
    let n_eta = model.n_eta;
    let nk = model.n_kappa;
    let nll = |b: &[f64]| {
        let kappas: Vec<&[f64]> = b[n_eta..].chunks(nk).collect();
        crate::stats::likelihood::individual_nll_iov_with_scratch(
            &model,
            &refused,
            &params.theta,
            &b[..n_eta],
            &kappas,
            &params.omega,
            params.omega_iov.as_ref(),
            &params.sigma.values,
            &mut crate::pk::EventPkParams::default(),
        )
    };
    let d = b.len();
    let second = |i: usize, j: usize, hh: f64| {
        let at = |si: f64, sj: f64| {
            let mut x = b.clone();
            x[i] += si * hh;
            x[j] += sj * hh;
            nll(&x)
        };
        (at(1.0, 1.0) - at(1.0, -1.0) - at(-1.0, 1.0) + at(-1.0, -1.0)) / (4.0 * hh * hh)
    };
    let mut worst = 0.0_f64;
    for i in 0..d {
        for j in 0..d {
            let rich = (4.0 * second(i, j, 1e-3) - second(i, j, 2e-3)) / 3.0;
            assert!(rich.is_finite() && h[(i, j)].is_finite());
            worst = worst.max((h[(i, j)] - rich).abs() / (1.0 + rich.abs())); // nan-ok: finite asserted above
        }
    }
    eprintln!("IOV + FREM exact inner Hessian vs central FD: {worst:.3e}");
    assert!(worst < 1e-5, "{worst:.3e}");
}

/// A TTE term at an overflowing η repels by its value (the sentinel or a non-finite NLL) and
/// has no exact Hessian there: `NonGaussianTerm` is the same point as the objective's repel, so
/// the Laplace objective reads the sentinel and `foce_subject_nll_cov` adds
/// `SUBJECT_SENTINEL_NLL` — never a finite-difference curvature. At the subject's own EBE the
/// term is served. Mutation: map the `jets2` decline to `Ok` -> the first assertion fails.
#[cfg(feature = "survival")]
#[test]
fn a_repelled_tte_term_declines_the_hessian_where_the_objective_repels() {
    let root = env!("CARGO_MANIFEST_DIR");
    let prep = crate::api::prepare_run(
        &format!("{root}/examples/tte_exponential.ferx"),
        Some(&format!("{root}/data/tte_exponential.csv")),
    )
    .unwrap();
    let model = &prep.parsed.model;
    let s = prep
        .population
        .subjects
        .iter()
        .find(|s| !s.obs_records.is_empty())
        .expect("a TTE subject")
        .clone();
    let p = &model.default_params;
    let ok = exact_inner_hessian_or_decline(model, &s, p, &vec![0.1; model.n_eta]);
    assert!(ok.is_ok(), "served at an ordinary point: {:?}", ok.err());
    let eta = vec![800.0; model.n_eta];
    assert_eq!(
        exact_inner_hessian_or_decline(model, &s, p, &eta).err(),
        Some(HessianDecline::NonGaussianTerm)
    );
    let mut acc = 0.0;
    crate::stats::likelihood::accumulate_non_gaussian_nll(
        model, &s, &p.theta, &eta, None, 1.0, &mut acc,
    );
    assert!(
        !acc.is_finite() || acc >= crate::survival::REPEL_SENTINEL_FLOOR,
        "the value path scores {acc} where the Hessian declines"
    );
}

/// G14 sweep: at its own EBE, every subject of every committed model that has a dataset has an
/// exact inner Hessian (no `HessianDecline`), so the quadrature's exact anchor and the non-Gaussian
/// curvature never reach a point the value path scores and the Hessian cannot. First two
/// subjects with data of each model; the stacked `[η̂, κ̂]` under IOV. Tier 3 (it solves ~100 EBEs).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: solves EBEs of every committed model; opt in with --features slow-tests"
)]
fn every_committed_subject_has_an_exact_inner_hessian_at_its_ebe() {
    let root = env!("CARGO_MANIFEST_DIR");
    let mut paths: Vec<_> = std::fs::read_dir(format!("{root}/examples"))
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "ferx"))
        .collect();
    paths.sort();
    let (mut ran, mut declined) = (0usize, Vec::new());
    for p in paths {
        let stem = p.file_stem().unwrap().to_str().unwrap().to_string();
        let data = format!("{root}/data/{stem}.csv");
        if !std::path::Path::new(&data).is_file() {
            continue;
        }
        let Ok(prep) = crate::api::prepare_run(p.to_str().unwrap(), Some(&data)) else {
            continue;
        };
        let model = &prep.parsed.model;
        if model.n_eta + model.n_kappa == 0 || model.is_sde() || model.mixture.is_some() {
            continue;
        }
        let params = &model.default_params;
        for s in prep
            .population
            .subjects
            .iter()
            .filter(|s| !s.observations.is_empty() || !s.obs_records.is_empty())
            .take(2)
        {
            let r = find_ebe(model, s, params, 300, 1e-8, None, None, 0);
            if r.hard_reject || !r.nll.is_finite() {
                continue;
            }
            let mut b: Vec<f64> = r.eta.iter().copied().collect();
            for k in &r.kappas {
                b.extend(k.iter().copied());
            }
            ran += 1;
            if let Err(why) = exact_inner_hessian_or_decline(model, s, params, &b) {
                declined.push(format!("{stem} / {}: {why:?}", s.id));
            }
        }
    }
    eprintln!(
        "exact inner Hessian sweep: {ran} subjects, {} declined",
        declined.len()
    );
    assert!(ran >= 60, "only {ran} subjects ran");
    assert!(declined.is_empty(), "declined:\n{}", declined.join("\n"));
}

/// A linear `[odes]` model whose infusion end moves with η (`D1`, `RATE=-2`).
const ONE_CPT_D1_ODE: &str = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(30.0, 1.0, 500.0)
  theta TVD1(2.0, 0.1, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_D1 ~ 0.09
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  D1 = TVD1 * exp(ETA_D1)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// A point no plan of the subject serves is the native predictor's in the objective's value
/// pass: `plan_values` declines it without recording (#3010). Recording it there cost a whole
/// recording (a few hundred thousand instructions) per objective value of an EBE search. The
/// gradient's lookup (`with_plan`) still records the point, after which the value pass serves
/// it. Since an infusion end that moves across an observation no longer changes a plan's
/// structure (`a_plan_serves_both_sides_of_a_moving_infusion_end`), the moved point here moves
/// the first dose's end past the second dose's arrival, the one check a walked end keeps.
/// Mutation: `record_miss = true` in `plan_values` -> the moved point is served, by a recording;
/// drop the walked end's check (`end < until`) -> the first plan serves it, wrongly.
#[test]
fn plan_values_declines_a_point_no_plan_serves_without_recording() {
    use crate::parser::model_parser::rhs_regcode::kernel_jit as kj;
    use crate::stats::likelihood::plan_value_route;
    use crate::types::RateMode::ModeledDuration;
    let model = parse_model_string(ONE_CPT_D1_ODE).unwrap();
    assert!(plan_value_route(&model));
    let d = |t| DoseEvent::modeled(t, 100.0, 1, false, 0.0, ModeledDuration);
    let times = [1.0, 2.1, 2.5, 6.0, 12.5, 14.1, 14.5, 20.0];
    let s = subject(&model, vec![d(0.0), d(12.0)], &times, &[0.0; 2]);
    let theta = moved_params(&model).theta;
    let end = |e: &[f64]| theta[2] * e[1].exp();
    let (e0, e1) = (vec![0.0, 0.0], vec![0.0, (13.0 / theta[2]).ln()]);
    // The straddle: the first window ends before the second dose arrives at `e0`, after it at
    // `e1`.
    assert!(
        end(&e0) < 12.0 && end(&e1) > 12.0,
        "{} {}",
        end(&e0),
        end(&e1)
    );
    let native = |e: &[f64]| crate::pk::compute_predictions_with_tv(&model, &s, &theta, e);
    let planned = || kj::TEMPLATE_PROBE.with(|p| p.replace((false, 0, 0)).1);
    let values = |e: &[f64]| {
        let mut out = Vec::new();
        plan_values(&model, &s, None, &theta, e, &mut out).then_some(out)
    };
    planned();
    // The first point records: the subject has no plan yet.
    let got = values(&e0).expect("the first point is served");
    assert_eq!(planned(), 1, "the subject's first plan is recorded");
    assert_eq!(got, native(&e0));
    // The moved point: no plan serves it, and the value pass does not record one.
    assert!(values(&e1).is_none(), "the moved point is declined");
    assert_eq!(planned(), 0, "declined without a recording");
    // The gradient's lookup records it; then the value pass serves it.
    let n = s.observations.len();
    with_plan(&model, &s, None, &theta, &e1, &mut |p| {
        kjit_run(p, &theta, &e1, n)
    })
    .expect("the gradient's lookup serves the moved point");
    assert_eq!(
        planned(),
        1,
        "the gradient's lookup records the moved point"
    );
    let got = values(&e1).expect("the recorded variant serves the moved point");
    assert_eq!(planned(), 0);
    assert_eq!(got, native(&e1));
}

/// One plan serves an infusion end on both sides of an observation (#3010): the end of a
/// modeled `D1` (`RATE=-2`) is walked without a timeline event (`integrate_tvcov_g`'s `soft`),
/// each segment of its window cut where it falls (`AffinePieces`), so moving it past the
/// observation at 2.5 h changes no structure. Before, the end was an event sorted against the
/// observations, the recording kept that order as a check, and every crossing recorded the
/// subject again — on #3010's 156-subject model, 2084 recordings and 7011 compiled kernels.
///
/// Three pairs, the recording made at the first point and run at the second: the end moves
/// later across the observation, earlier, and with the other η moving too. At the second
/// point: no recording, values bit-equal to the native predictor and to a fresh recording
/// there, and each `∂f/∂η` against central differences of the native predictor (worst
/// measured 1.3e-9 relative, the differences' own error at `h = 1e-6`; bound 2e-8).
///
/// Mutations, each run: the walked ends off (`soft` empty) → "no new recording at the moved
/// point"; the cut without its lower clamp → a piece of negative length fails its check and the
/// moved point records again; without its upper clamp → "a fresh recording at the moved point"
/// disagrees; the rate left off the first piece → "value 0 against native"; the cut at the end's
/// value only (`w.end.value_only()`) → `∂f/∂η_D1` off by 5.1 relative (the boundary term).
#[test]
fn a_plan_serves_both_sides_of_a_moving_infusion_end() {
    use crate::parser::model_parser::rhs_regcode::kernel_jit as kj;
    use crate::types::RateMode::ModeledDuration;
    let model = parse_model_string(ONE_CPT_D1_ODE).unwrap();
    let d = |t| DoseEvent::modeled(t, 100.0, 1, false, 0.0, ModeledDuration);
    let times = [1.0, 2.1, 2.5, 6.0, 12.5, 14.1, 14.5, 20.0];
    let s = subject(&model, vec![d(0.0), d(12.0)], &times, &[0.0; 2]);
    let mut fresh_s = s.clone();
    fresh_s.id = format!("{}-fresh", s.id);
    let theta = moved_params(&model).theta;
    let n = s.observations.len();
    let end = |e: &[f64]| theta[2] * e[1].exp();
    let planned = || kj::TEMPLATE_PROBE.with(|p| p.replace((false, 0, 0)).1);
    let native = |e: &[f64]| crate::pk::compute_predictions_with_tv(&model, &s, &theta, e);
    let run = |s: &Subject, e: &[f64]| {
        with_plan(&model, s, None, &theta, e, &mut |p| {
            kjit_run(p, &theta, e, n)
        })
        .expect("served")
        .du
    };
    let pairs = [
        (vec![0.0, 0.0], vec![0.0, 0.2]),
        (vec![0.0, 0.2], vec![0.0, 0.0]),
        (vec![0.1, -0.3], vec![-0.1, 0.25]),
    ];
    for (i, (e0, e1)) in pairs.iter().enumerate() {
        // The straddle: the first window's end on opposite sides of the observation at 2.5,
        // and before the second dose at both points.
        assert!(
            (end(e0) < 2.5) != (end(e1) < 2.5) && end(e0).max(end(e1)) < 12.0,
            "pair {i}: {} {}",
            end(e0),
            end(e1)
        );
        planned();
        run(&s, e0);
        // The first pair records the subject's plan; the later ones find it.
        assert_eq!(planned(), usize::from(i == 0), "pair {i}: the first point");
        let got = run(&s, e1);
        assert_eq!(
            planned(),
            0,
            "pair {i}: no new recording at the moved point"
        );
        let fresh = run(&fresh_s, e1);
        assert_eq!(got, fresh, "pair {i}: a fresh recording at the moved point");
        let nat = native(e1);
        let l = 1 + e1.len();
        let mut worst = 0.0f64;
        for j in 0..n {
            assert_eq!(got[j * l], nat[j], "pair {i}: value {j} against native");
            for k in 0..e1.len() {
                let h = 1e-6;
                let (mut ep, mut em) = (e1.clone(), e1.clone());
                ep[k] += h;
                em[k] -= h;
                let fd = (native(&ep)[j] - native(&em)[j]) / (2.0 * h);
                let gap = (got[j * l + 1 + k] - fd).abs() / (1e-6 + fd.abs());
                assert!(gap.is_finite(), "pair {i}: obs {j}, eta {k}: {gap}");
                worst = worst.max(gap);
            }
        }
        eprintln!("pair {i}: worst d/deta gap {worst:e}");
        assert!(
            worst < 2e-8,
            "pair {i}: d/deta against central differences: {worst:e}"
        );
    }
}

/// The structured kernel's reduce is one plan shared by every subject (`kernel_run_shared`,
/// #2980). Two subjects with different doses and data on one thread — the second runs the
/// first's cached plan — each against the recorded kernel, so a reduce keyed too loosely (one
/// subject's recording served where it does not hold) or baking a subject's data shows.
/// Mutation: key the shared plan without the parameter literals (or reuse one plan for every
/// variant) → the second subject's Hessian moves.
#[test]
fn structured_reduce_is_shared_across_subjects_exactly() {
    use super::hessian::st::{STRUCT_OFF, STRUCT_PASSES};
    let model = parse_model_string(ONE_CPT_ORAL).unwrap();
    let s1 = subject(
        &model,
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &TIMES,
        &[0.1, -0.05, 0.2],
    );
    let s2 = subject(
        &model,
        vec![
            DoseEvent::new(0.0, 250.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &TIMES,
        &[-0.2, 0.15, 0.05],
    );
    let params = moved_params(&model);
    let x = pack_params(&params);
    with_fit_stack(|| {
        let h = |s: &Subject, off: bool| {
            STRUCT_OFF.with(|c| c.set(off));
            let eta = ebe_for(&model, s, &params, 1e-12, true);
            let h = subject_adjoint_cov_hessian(
                &model,
                s,
                &params,
                &x,
                eta.as_slice(),
                Objective::Focei,
            )
            .expect("in scope");
            STRUCT_OFF.with(|c| c.set(false));
            h
        };
        STRUCT_PASSES.with(|c| c.set(0));
        let (a1, a2) = (h(&s1, false), h(&s2, false));
        assert!(
            STRUCT_PASSES.with(std::cell::Cell::get) > 0,
            "the fixture must take the structured kernel"
        );
        let (b1, b2) = (h(&s1, true), h(&s2, true));
        let (g1, g2) = (
            rel_gap(a1.as_slice(), b1.as_slice()),
            rel_gap(a2.as_slice(), b2.as_slice()),
        );
        eprintln!("structured vs recorded across subjects: {g1:.2e}, {g2:.2e}");
        assert!(g1 < 1e-12 && g2 < 1e-12, "{g1:.2e} {g2:.2e}");
    });
}

/// A FOCE subject past the Woodbury gate (`1 + tr(LᵀSL) > 1e6`, where production factors `R̃`
/// itself) is outside the structured kernel's form, so its passes take the recorded kernel
/// rather than declining the subject (#2980 review). The pair straddles the gate: at the
/// fixture's σ the subject takes the structured kernel, at a σ shrunk `1e-5`× (the trace far
/// past `1e6`) none of its passes does — both asserted — and each Hessian matches the recorded
/// kernel's (measured: 3.4e-15 structured, exactly 0 on the recorded route; bound as
/// `structured_reduce_is_shared_across_subjects_exactly`'s).
/// Mutation: drop `struct_rows_hold` from `CovPlan::run`'s gate → the shrunk-σ subject's
/// Hessian is `None` (its structured sweep cannot form the rows).
#[test]
fn foce_past_the_woodbury_gate_takes_the_recorded_kernel() {
    use super::hessian::st::{STRUCT_OFF, STRUCT_PASSES};
    let model = parse_model_string(ONE_CPT_ORAL).unwrap();
    let s = subject(
        &model,
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &TIMES,
        &[0.1, -0.05, 0.2],
    );
    let base = moved_params(&model);
    let mut tight = base.clone();
    for v in tight.sigma.values.iter_mut() {
        *v *= 1e-5;
    }
    with_fit_stack(|| {
        for (params, want_structured) in [(&base, true), (&tight, false)] {
            let x = pack_params(params);
            let eta = ebe_for(&model, &s, params, 1e-12, false);
            let h = |off: bool| {
                STRUCT_OFF.with(|c| c.set(off));
                let h = subject_adjoint_cov_hessian(
                    &model,
                    &s,
                    params,
                    &x,
                    eta.as_slice(),
                    Objective::Foce,
                );
                STRUCT_OFF.with(|c| c.set(false));
                h
            };
            STRUCT_PASSES.with(|c| c.set(0));
            let got = h(false).expect("the exact Hessian, structured or recorded");
            let structured = STRUCT_PASSES.with(std::cell::Cell::get) > 0;
            assert_eq!(
                structured, want_structured,
                "structured passes: {structured}, want {want_structured}"
            );
            let want = h(true).expect("the recorded kernel's Hessian");
            let gap = rel_gap(got.as_slice(), want.as_slice());
            eprintln!("Woodbury straddle, structured {structured}: gap {gap:.2e}");
            assert!(gap < 1e-12, "{gap:.2e}");
        }
    });
}

/// A shared kernel (`kernel_run_shared`) serves a point on the other side of a branch its
/// first recording took: the recording carries that branch as a guard, so the second point
/// fails it and is served by a plan recorded there — the Laplace reduce factors `H` or
/// jitters it per subject (`factor_anchor`), on one shared key. The straddle is asserted:
/// both points must take the same key and opposite branches, and each value is the branch's
/// own closed form. Mutation: serve only the first plan under a key → the second point
/// declines (`None`); serve it without its guard → the second point gets `x²` (= 9, not 6).
#[test]
fn shared_kernel_serves_both_sides_of_a_recorded_branch() {
    use crate::parser::model_parser::rhs_regcode::kernel_jit as kj;
    use crate::sens::num::vm::Layout;
    use crate::sens::num::{BranchCmp, PkNum};
    let model = parse_model_string(ONE_CPT_ORAL).unwrap();
    let s = subject(
        &model,
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &TIMES,
        &[0.1, -0.05, 0.2],
    );
    let taken = std::cell::RefCell::new(Vec::new());
    let f = |lv: &[kj::Tr]| -> Option<Vec<kj::Tr>> {
        let x = lv[0];
        let pos = x.br_cmp_c(BranchCmp::Gt, 0.0);
        taken.borrow_mut().push(pos);
        Some(vec![if pos {
            x * x
        } else {
            x * <kj::Tr as PkNum>::from_f64(-3.0)
        }])
    };
    let variant = super::kernel::variant_of(&("branch test", 1u8));
    let run = |v: f64| {
        super::kernel::kernel_run_shared(&model, &s, variant.clone(), Layout::F64, &[v], 1, &f)
            .map(|k| k.value(0))
    };
    let a = run(2.0).expect("first point records");
    let b = run(-2.0).expect("the other branch is served by its own recording");
    let c = run(3.0).expect("the first branch is served again");
    // A family-template replay may run the recorder once more on the new branch before it
    // records afresh; what matters is that both branches were recorded and nothing after.
    let t = taken.borrow();
    assert!(
        t[0] && t[1..].iter().all(|p| !p) && t.len() >= 2,
        "the straddle: the first point positive, the second recorded on the other branch: {t:?}"
    );
    assert_eq!((a, b, c), (4.0, 6.0, 9.0));
}

/// The covariance Hessian's bases take the shared-primal kernels (#2980): on closed-form and
/// linear-ODE subjects the rows forward and the rows forward-over-reverse run, and on a
/// nonlinear-ODE subject the ODE step loop runs as rows too — each counted where it completes, so a silent fallback to the
/// per-axis runs (correct, and invisible to every oracle) fails here. The Hessians themselves
/// are pinned by the oracles (`hessian_gaps_for`); this pins which engine produced them.
#[test]
fn the_covariance_bases_take_the_shared_primal_kernels() {
    use crate::parser::model_parser::rhs_regcode::kernel_jit::ROWS_RUNS;
    // Closed form; a linear ODE (its matrix exponential, no step loop); a nonlinear one (the
    // step loop).
    for (src, ode) in [
        (ONE_CPT_ORAL, false),
        (ONE_CPT_ORAL_ODE, false),
        (MM_ODE, true),
    ] {
        let model = parse_model_string(src).unwrap();
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
        let x = pack_params(&params);
        let eta = ebe_for(&model, &s, &params, 1e-12, true);
        // Counted on the thread that runs the step (`with_fit_stack` runs it on its own).
        let [fwd, rev, loops] = with_fit_stack(|| {
            ROWS_RUNS.with(|c| c.set([0; 3]));
            subject_adjoint_cov_hessian(&model, &s, &params, &x, eta.as_slice(), Objective::Focei)
                .expect("in scope");
            ROWS_RUNS.with(std::cell::Cell::get)
        });
        eprintln!("ode {ode}: rows forwards {fwd}, reverses {rev}, loop runs {loops}");
        assert!(
            fwd > 0 && rev > 0,
            "ode {ode}: rows forward {fwd}, reverse {rev}"
        );
        assert_eq!(loops > 0, ode, "ode {ode}: rows loop runs {loops}");
    }
}

/// A walked infusion end's segment that the exact pieces decline (a type that does not record
/// probes `A`, and its self-check can refuse a finite segment) is integrated on the values, as
/// the end's event would have been — not filled with `NaN` (#3010 review). Forced
/// (`FORCE_LINEAR_DECLINE`) on the `Dual2` walk, with the end inside a segment (2.47 h, between
/// the 2.1 and 2.5 h samples) and outside every segment's interior (2.24 h): values and
/// `∂f/∂η` against the exact pieces, the fallback's adaptive solver at 1e-12: measured worst
/// 9.8e-13 relative (4.0e-5 at the default tolerance, the solver's own error); bound 1e-10.
/// Mutations: the decline filled with `NaN` again → "finite"; the fallback's rate-off saltation
/// dropped → `∂f/∂η_D1` past the end.
#[test]
fn a_declined_walked_segment_is_integrated_on_its_values() {
    use crate::types::RateMode::ModeledDuration;
    let mut model = parse_model_string(ONE_CPT_D1_ODE).unwrap();
    // The fallback integrates adaptively: tight enough that its error is far below a wrong
    // crossing's.
    let ode = model.ode_spec.as_mut().unwrap();
    ode.solver_opts.reltol = 1e-12;
    ode.solver_opts.abstol = 1e-12;
    let d = |t| DoseEvent::modeled(t, 100.0, 1, false, 0.0, ModeledDuration);
    let times = [1.0, 2.1, 2.5, 6.0, 12.5, 14.1, 14.5, 20.0];
    let s = subject(&model, vec![d(0.0), d(12.0)], &times, &[0.0; 2]);
    let theta = moved_params(&model).theta;
    let th: Vec<Dual2<2>> = theta.iter().map(|&v| Dual2::constant(v)).collect();
    let walk = |eta: &[f64]| -> Vec<Dual2<2>> {
        let et: Vec<Dual2<2>> = eta
            .iter()
            .enumerate()
            .map(|(k, &v)| Dual2::var(v, k))
            .collect();
        record(&model, &s, None, &theta, eta, &ThetaLeaves::theta(&th), &et).expect("in scope")
    };
    for eta in [[0.05, 0.1], [-0.05, 0.0]] {
        let exact = walk(&eta);
        let declined = {
            let prev = crate::ode::solver::FORCE_LINEAR_DECLINE.replace(true);
            let r = walk(&eta);
            crate::ode::solver::FORCE_LINEAR_DECLINE.set(prev);
            r
        };
        let mut worst = 0.0f64;
        for (j, (a, b)) in exact.iter().zip(&declined).enumerate() {
            let pairs = std::iter::once((a.value, b.value))
                .chain(a.grad.iter().copied().zip(b.grad.iter().copied()));
            for (x, y) in pairs {
                assert!(
                    x.is_finite() && y.is_finite(),
                    "eta {eta:?}, obs {j}: finite ({x}, {y})"
                );
                worst = worst.max((x - y).abs() / (1e-6 + x.abs()));
            }
        }
        eprintln!("eta {eta:?}: declined vs exact {worst:e}");
        assert!(worst < 1e-10, "eta {eta:?}: {worst:e}");
    }
}

/// [`ONE_CPT_D1_ODE`] on the closed form (`pk one_cpt_iv`): the same modeled `D1`, walked by
/// the analytic event walk.
fn one_cpt_d1_closed_form() -> CompiledModel {
    let ode = "  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n";
    assert!(ONE_CPT_D1_ODE.contains(ode));
    let model =
        parse_model_string(&ONE_CPT_D1_ODE.replace(ode, "  pk one_cpt_iv(cl=CL, v=V)\n")).unwrap();
    assert!(model.ode_spec.is_none(), "the closed form");
    model
}

/// A subject of [`one_cpt_d1_closed_form`]: 100 mg doses at `doses` with a modeled `D1`
/// (`RATE=-2`), sampled at `times`.
fn d1_subject(model: &CompiledModel, doses: &[f64], times: &[f64]) -> Subject {
    use crate::types::RateMode::ModeledDuration;
    let d = |t| DoseEvent::modeled(t, 100.0, 1, false, 0.0, ModeledDuration);
    subject(
        model,
        doses.iter().map(|&t| d(t)).collect(),
        times,
        &[0.0; 2],
    )
}

/// The served plan's `[f, ∂f/∂η…]` per observation at `eta`, and how many plans the call
/// recorded.
fn served_and_recorded(
    model: &CompiledModel,
    s: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> (Vec<f64>, usize) {
    use crate::parser::model_parser::rhs_regcode::kernel_jit as kj;
    kj::TEMPLATE_PROBE.with(|p| p.replace((false, 0, 0)));
    let n = s.observations.len();
    let du = with_plan(model, s, None, theta, eta, &mut |p| {
        kjit_run(p, theta, eta, n)
    })
    .expect("served")
    .du;
    (du, kj::TEMPLATE_PROBE.with(|p| p.replace((false, 0, 0)).1))
}

/// The worst relative gap of a plan's values (`got`, laid out as [`served_and_recorded`]'s)
/// against the native predictor at `eta`, and of its `∂f/∂η` against the native predictor's
/// differences, central (`h = 1e-6`) except at the observations in `one_sided`, which take the
/// forward difference (`h = 1e-7`).
fn gaps_against_native(
    model: &CompiledModel,
    s: &Subject,
    theta: &[f64],
    eta: &[f64],
    got: &[f64],
    one_sided: &[usize],
) -> (f64, f64) {
    let native = |e: &[f64]| crate::pk::compute_predictions_with_tv(model, s, theta, e);
    let nat = native(eta);
    let l = 1 + eta.len();
    let (mut value, mut grad) = (0.0f64, 0.0f64);
    for j in 0..s.observations.len() {
        let gap = (got[j * l] - nat[j]).abs() / nat[j].abs();
        assert!(gap.is_finite(), "value {j}: {} vs {}", got[j * l], nat[j]);
        value = value.max(gap);
        for k in 0..eta.len() {
            let (mut ep, mut em) = (eta.to_vec(), eta.to_vec());
            let fd = if one_sided.contains(&j) {
                let h = 1e-7;
                ep[k] += h;
                (native(&ep)[j] - nat[j]) / h
            } else {
                let h = 1e-6;
                ep[k] += h;
                em[k] -= h;
                (native(&ep)[j] - native(&em)[j]) / (2.0 * h)
            };
            let gap = (got[j * l + 1 + k] - fd).abs() / (1e-6 + fd.abs());
            assert!(
                gap.is_finite(),
                "obs {j}, eta {k}: {} vs {fd}",
                got[j * l + 1 + k]
            );
            grad = grad.max(gap);
        }
    }
    (value, grad)
}

/// The closed-form twin of `a_plan_serves_both_sides_of_a_moving_infusion_end` (#3033): one
/// plan serves the end of a modeled `D1` (`RATE=-2`) on both sides of the observation at 2.5 h.
/// The analytic event walk (`sens::propagate`'s `walked_ends`) walks such an end without a break:
/// every sub-interval of its window is cut where the end falls (`walk_cut`), chosen on every
/// evaluation. Before, the end was a break sorted against the observations, the recording kept
/// that order as a guard, and every crossing recorded the subject again.
///
/// Three pairs, the recording made at the first point and run at the second: the end moves
/// later across the observation, earlier, and with the other η moving too. At the second point:
/// no recording, values bit-equal to a fresh recording there and equal to the native predictor
/// (measured worst 3.3e-16 relative — the plan and the native walk round differently, by an ULP
/// or two, at every point, recorded before #3033 too; bound 1e-14), and each `∂f/∂η` against
/// central differences of the native predictor (measured worst 7.9e-10, the differences' own
/// error at `h = 1e-6`; bound 2e-8).
///
/// Mutations, each run: no window walked (`walked_ends` empty) → "no new recording at the moved
/// point"; the window's rate left off the cut's first piece → the values; the cut at the end's
/// value only (`we.end.value_only()`) → `∂f/∂η_D1`.
#[test]
fn a_closed_form_plan_serves_both_sides_of_a_moving_infusion_end() {
    let model = one_cpt_d1_closed_form();
    let times = [1.0, 2.1, 2.5, 6.0, 12.5, 14.1, 14.5, 20.0];
    let s = d1_subject(&model, &[0.0, 12.0], &times);
    let mut fresh_s = s.clone();
    fresh_s.id = format!("{}-fresh", s.id);
    let theta = moved_params(&model).theta;
    let end = |e: &[f64]| theta[2] * e[1].exp();
    let pairs = [
        (vec![0.0, 0.0], vec![0.0, 0.2]),
        (vec![0.0, 0.2], vec![0.0, 0.0]),
        (vec![0.1, -0.3], vec![-0.1, 0.25]),
    ];
    for (i, (e0, e1)) in pairs.iter().enumerate() {
        // The straddle: the first window's end on opposite sides of the observation at 2.5,
        // and before the second dose at both points.
        assert!(
            (end(e0) < 2.5) != (end(e1) < 2.5) && end(e0).max(end(e1)) < 12.0,
            "pair {i}: {} {}",
            end(e0),
            end(e1)
        );
        let (_, first) = served_and_recorded(&model, &s, &theta, e0);
        // The first pair records the subject's plan; the later ones find it.
        assert_eq!(first, usize::from(i == 0), "pair {i}: the first point");
        let (got, moved) = served_and_recorded(&model, &s, &theta, e1);
        assert_eq!(moved, 0, "pair {i}: no new recording at the moved point");
        let (fresh, _) = served_and_recorded(&model, &fresh_s, &theta, e1);
        assert_eq!(got, fresh, "pair {i}: a fresh recording at the moved point");
        let (value, grad) = gaps_against_native(&model, &s, &theta, e1, &got, &[]);
        eprintln!("pair {i}: worst value gap {value:e}, d/deta gap {grad:e}");
        assert!(value < 1e-14, "pair {i}: values against native: {value:e}");
        assert!(
            grad < 2e-8,
            "pair {i}: d/deta against central differences: {grad:e}"
        );
    }
}

/// The one check a walked window keeps (#3033, `walked_ends`): its end comes before the next
/// infusion's arrival, recorded as a guard. Moving the first dose's end past the second dose's
/// arrival at 12 h fails it, so the moved point is recorded again — the end is a break there, as
/// before — and its values and `∂f/∂η` are the native predictor's (measured worst 4.4e-16 and
/// 8.0e-10; bounds 1e-14 and 2e-8). Mutation: the check dropped (`end < until`) → the first
/// plan serves the moved point, with the first window's rate cut off at 12 h ("a new recording").
#[test]
fn a_walked_end_past_the_next_infusion_records_again() {
    let model = one_cpt_d1_closed_form();
    let times = [1.0, 2.1, 2.5, 6.0, 12.5, 14.1, 14.5, 20.0];
    let s = d1_subject(&model, &[0.0, 12.0], &times);
    let theta = moved_params(&model).theta;
    let end = |e: &[f64]| theta[2] * e[1].exp();
    let (e0, e1) = (vec![0.0, 0.0], vec![0.0, (13.0 / theta[2]).ln()]);
    assert!(
        end(&e0) < 12.0 && end(&e1) > 12.0,
        "{} {}",
        end(&e0),
        end(&e1)
    );
    let (_, first) = served_and_recorded(&model, &s, &theta, &e0);
    assert_eq!(first, 1, "the first point");
    let (got, moved) = served_and_recorded(&model, &s, &theta, &e1);
    assert_eq!(moved, 1, "a new recording at the moved point");
    let (value, grad) = gaps_against_native(&model, &s, &theta, &e1, &got, &[]);
    eprintln!("worst value gap {value:e}, d/deta gap {grad:e}");
    assert!(value < 1e-14, "values against native: {value:e}");
    assert!(grad < 2e-8, "d/deta against central differences: {grad:e}");
}

/// A walked end exactly on a sample belongs to the later sub-interval (#3033, `walk_cut`): the
/// window still contributes to the sample, production's convention for a window end on a sample
/// (the end sorts after the observation), and the sample's `∂f/∂η_D1` is the one-sided
/// derivative of that branch — the end moving later. `TVD1 = 2.5` and `η_D1 = 0` put the end on
/// the 2.5 h sample exactly. The two one-sided differences there differ by the rate-off term
/// (asserted, so the test can tell the sides apart); the plan matches the forward one (measured
/// worst 7.3e-8 relative, the forward difference's own error at `h = 1e-7`; bound 1e-6), and
/// every other sample's central difference (smooth there). Mutations: the whole-on arm taken only
/// past `b` (`Gt` for `Ge`) → the 2.5 h sample takes the backward side; the whole-off arm from
/// `a` on (`Le` for `Lt`) → the end's derivative is lost from the sub-interval after it.
#[test]
fn a_walked_end_on_a_sample_leaves_the_window_on_for_it() {
    let model = one_cpt_d1_closed_form();
    let times = [1.0, 2.1, 2.5, 6.0];
    let s = d1_subject(&model, &[0.0], &times);
    let mut theta = moved_params(&model).theta;
    theta[2] = 2.5;
    let eta = [0.05f64, 0.0];
    assert_eq!(theta[2] * eta[1].exp(), 2.5, "the end on the sample");
    let native = |e: &[f64]| crate::pk::compute_predictions_with_tv(&model, &s, &theta, e)[2];
    let h = 1e-7;
    let (fwd, bwd) = (
        (native(&[eta[0], h]) - native(&eta)) / h,
        (native(&eta) - native(&[eta[0], -h])) / h,
    );
    assert!(
        (fwd - bwd).abs() > 1e-2 * fwd.abs(),
        "the two sides apart: {fwd} {bwd}"
    );
    let (got, _) = served_and_recorded(&model, &s, &theta, &eta);
    let (value, grad) = gaps_against_native(&model, &s, &theta, &eta, &got, &[2]);
    eprintln!("worst value gap {value:e}, d/deta gap {grad:e} (forward {fwd}, backward {bwd})");
    assert!(value < 1e-14, "values against native: {value:e}");
    assert!(
        grad < 1e-6,
        "d/deta against the forward difference: {grad:e}"
    );
}

/// A sub-interval far from a walked end stays finite (#3033, `walk_cut`): the cut's point is
/// clamped into the sub-interval, so the arm not chosen never steps backwards — a backward
/// closed-form step grows as `e^{ke·|dt|}`, and `ke·|dt| ≈ 800` overflows it. The forward
/// tangents select past such an arm; the reverse passes take it with a zero adjoint, and
/// `0 · ∞` is `NaN`. Two subjects: the end long before a sub-interval (a bolus 6000 h after the
/// window, sampled after it), and long after one (a 7000 h window, sampled in its first hours).
/// The plan's values and `∂f/∂η`, and the FOCEI adjoint gradient, finite. Mutations, each run:
/// either clamp dropped → this test fails (the plan's forward values stay finite without them;
/// only the reverse passes see it).
#[test]
fn a_walked_cut_far_from_the_end_stays_finite() {
    use crate::types::RateMode::ModeledDuration;
    let model = one_cpt_d1_closed_form();
    let eta = [0.05, 0.1];
    let template = moved_params(&model);
    let d1 = template.theta[2];
    let infusion = DoseEvent::modeled(0.0, 100.0, 1, false, 0.0, ModeledDuration);
    for (label, tvd1, doses, times) in [
        (
            "end long before",
            d1,
            vec![
                infusion.clone(),
                DoseEvent::new(6000.0, 100.0, 1, 0.0, false, 0.0),
            ],
            vec![1.0, 2.5, 6001.0, 6002.0],
        ),
        (
            "end long after",
            7000.0,
            vec![infusion.clone()],
            vec![1.0, 2.0],
        ),
    ] {
        let mut params = template.clone();
        params.theta[2] = tvd1;
        let mut s = subject(&model, doses, &times, &[0.0; 2]);
        s.id = label.to_string();
        s.observations = vec![1.0; times.len()];
        let (got, _) = served_and_recorded(&model, &s, &params.theta, &eta);
        assert!(got.iter().all(|x| x.is_finite()), "{label}: plan {got:?}");
        let x = pack_params(&params);
        let grad = subject_adjoint_packed_gradient(&model, &s, &params, &x, &eta, true)
            .expect("in adjoint scope");
        assert!(
            grad.iter().all(|x| x.is_finite()),
            "{label}: gradient {grad:?}"
        );
    }
}

/// A plan recorded where a modeled window overflows does not serve a finite one (#3033). At
/// `η_D1 = 829.3` (a runaway step of an EBE search on the `ADVAN1` anchor's subject 19) `D` is
/// `∞`, the dose resolves to `rate = amt/∞ = 0` and so to no infusion — a plain value the
/// recording kept as its structure with no guard, so the plan then served every finite `D` as no
/// infusion either: predictions zero, an objective of 8.6e13 at every point, and the inner search
/// stuck at `η = 0` for 41 of the anchor's 60 subjects. Now the same decision on the dual window is
/// a guard: the finite point records again, with the native predictor's values (measured worst
/// 4.3e-16; bound 1e-14). `η_D1 = 300` (`D` huge but finite) records again too — the control that
/// it is the overflow, not the size. Mutation: the guard dropped → "a new recording" (1.0 off).
#[test]
fn a_plan_recorded_at_an_overflowed_window_does_not_serve_a_finite_one() {
    let model = one_cpt_d1_closed_form();
    let theta = moved_params(&model).theta;
    let good = [0.04, 0.25];
    for (k, bad) in [[0.0, 829.3], [0.0, 300.0]].iter().enumerate() {
        let mut s = d1_subject(
            &model,
            &[0.0, 12.0],
            // Its own design, so the two cases share no template.
            &[1.0 + 0.1 * k as f64, 2.0, 2.5, 6.0, 13.0, 14.5, 20.0],
        );
        s.id = format!("overflow-{k}");
        let (_, first) = served_and_recorded(&model, &s, &theta, bad);
        assert_eq!(first, 1, "{bad:?}: the first point");
        let (got, again) = served_and_recorded(&model, &s, &theta, &good);
        assert_eq!(again, 1, "{bad:?}: a new recording at the finite window");
        let (value, _) = gaps_against_native(&model, &s, &theta, &good, &got, &[]);
        eprintln!("{bad:?}: worst value gap {value:e}");
        assert!(value < 1e-14, "{bad:?}: values against native: {value:e}");
    }
}

/// A FREM covariate pseudo-observation (first row, at `TIME = 0.5` so that the plan's PK prediction
/// there, which production discards, is live and θ-dependent) predicts `θ_i + η_e`, so its row of the Jacobian is the
/// unit vector on `θ_i` and its weight is that θ's whole share of the score from the row —
/// by forward tangents as by reverse seeds (the plan returns the PK prediction on that row,
/// whose own tangents are discarded). The PK predictions do not read `TV_WT`, so its
/// gradient entry is the row's weight exactly, whichever route computed it.
#[test]
fn the_tangent_jacobian_patches_a_frem_row_like_the_reverse_one() {
    let model = parse_model_string(FREM_MODEL).unwrap();
    let fc = model.frem_config.as_ref().expect("fixture must be FREM");
    let (ti, _) = fc.fremtype_to_indices[&100u16];
    let mut s = subject_shell(
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &[0.5, 1.0, 6.0, 24.0],
    );
    s.fremtype = vec![100, 0, 0, 0];
    let theta = model.default_params.theta.clone();
    let eta = vec![0.1, -0.05, 0.15, 0.7];
    let n_theta = theta.len();
    assert!(
        n_theta <= s.observations.len(),
        "fixture: the tangents serve it"
    );
    let weights = vec![1.7, 0.9, -0.4, 0.6];
    let tan0 = TANGENT_JACS.with(|c| c.get());
    let by_tangents = fixed_eta_theta_adjoint(&model, &s, &theta, &eta, true, &weights).unwrap();
    assert_eq!(TANGENT_JACS.with(|c| c.get()) - tan0, 1);
    JAC_REVERSE_FORCE.with(|f| f.set(true));
    let by_reverse = fixed_eta_theta_adjoint(&model, &s, &theta, &eta, true, &weights).unwrap();
    JAC_REVERSE_FORCE.with(|f| f.set(false));
    assert_eq!(
        TANGENT_JACS.with(|c| c.get()) - tan0,
        1,
        "the control took reverse seeds"
    );
    let (jt, jr) = (by_tangents.jac.unwrap(), by_reverse.jac.unwrap());
    for (k, (a, b)) in jt.iter().zip(&jr).enumerate() {
        assert!(
            (a - b).abs() <= 1e-12 * (1.0 + b.abs()),
            "jac[{k}]: {a} vs {b}"
        );
    }
    for m in 0..n_theta {
        let (a, b) = (by_tangents.grad[m], by_reverse.grad[m]);
        assert!(
            (a - b).abs() <= 1e-12 * (1.0 + b.abs()),
            "grad[{m}]: {a} vs {b}"
        );
    }
    for m in 0..n_theta {
        assert_eq!(jt[m], f64::from(u8::from(m == ti)), "FREM row, θ{m}");
    }
    assert_eq!(
        by_tangents.grad[ti], weights[0],
        "TV_WT's score is the FREM row's weight"
    );
    assert!(
        (1..4).any(|j| jt[j * n_theta] != 0.0),
        "fixture: the PK rows depend on θ"
    );
}
