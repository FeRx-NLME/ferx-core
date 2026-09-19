//! #1458 — the numerical θ/σ M-step's two non-default estimators.
//!
//! The defect these exist for is not a wrong formula but a wrong *functional*:
//! the historical M-step blends in the **maximiser** of the frozen-η objective
//! at one η draw, and a maximiser is a nonlinear function of the draw, so its
//! Robbins-Monro average converges to `E[θ*(η)]` rather than to the `θ` that
//! maximises `E[Q(θ, η)]`. The two are not the same point, and the gap grows
//! with the dispersion of the draws — which is why better E-step mixing makes
//! the busulfan `TVQ` bias *worse*.
//!
//! Everything here is deterministic: a fixed list of η draws stands in for the
//! E-step, so the two estimators can be compared as recursions rather than as
//! fits, and the bias is exhibited as an exact number instead of a seed average.

use super::*;
use crate::parser::model_parser::parse_model_string;

// ── fixture ────────────────────────────────────────────────────────────────

/// Two-compartment IV with IIV on clearance only, so `TVQ` and `TVV2` have **no
/// ETA** and can move only through the numerical M-step — the busulfan B6 shape
/// the issue is written about, shrunk to six subjects.
fn no_eta_theta_model() -> CompiledModel {
    parse_model_string(
        r"
[parameters]
  theta TVCL(1.0, 0.05, 20.0)
  theta TVV(10.0, 1.0, 200.0)
  theta TVQ(2.0, 0.05, 50.0)
  theta TVV2(20.0, 1.0, 400.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.10 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  Q  = TVQ
  V2 = TVV2
[structural_model]
  pk two_cpt_iv(cl=CL, v1=V, q=Q, v2=V2)
[error_model]
  DV ~ proportional(PROP_ERR)
",
    )
    .expect("#1458 fixture parses")
}

/// Six subjects, rich sampling, observations simulated noise-free at a θ away
/// from the starting values so the M-step has somewhere to go.
fn no_eta_population(model: &CompiledModel, theta_true: &[f64]) -> Population {
    use crate::types::{DoseEvent, Population, Subject};
    let times = [0.25f64, 0.75, 1.5, 3.0, 6.0, 12.0, 24.0];
    let etas_true = [0.30f64, -0.25, 0.12, -0.05, 0.22, -0.34];
    let mut scratch = EventPkParams::default();
    let subjects: Vec<Subject> = etas_true
        .iter()
        .enumerate()
        .map(|(i, &e)| {
            let mut s = Subject {
                id: format!("{}", i + 1),
                doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                obs_times: times.to_vec(),
                observations: vec![0.0; times.len()],
                obs_cmts: vec![1; times.len()],
                cens: vec![0; times.len()],
                ..Default::default()
            };
            let preds = crate::pk::compute_predictions_with_tv_into(
                model,
                &s,
                theta_true,
                &[e],
                &mut scratch,
            );
            s.observations = preds.to_vec();
            s
        })
        .collect();
    Population {
        subjects,
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

/// A deterministic stand-in for the E-step: eight η draw-sets, each one η per
/// subject, dispersed around the values the data were simulated at. Dispersion
/// is the lever the Jensen bias scales with, so it is spelled out here rather
/// than sampled.
fn eta_draws() -> Vec<Vec<Vec<f64>>> {
    let base = [0.30f64, -0.25, 0.12, -0.05, 0.22, -0.34];
    let jitter = [
        [0.35f64, -0.10, 0.05, 0.40, -0.30, 0.15],
        [-0.40, 0.30, -0.20, -0.15, 0.35, -0.05],
        [0.10, 0.45, 0.30, -0.35, -0.10, 0.40],
        [-0.20, -0.35, 0.40, 0.20, 0.05, -0.25],
        [0.45, 0.15, -0.30, -0.05, 0.40, 0.10],
        [-0.05, -0.20, 0.15, 0.35, -0.40, 0.30],
        [0.25, 0.40, -0.10, -0.30, 0.15, -0.35],
        [-0.30, 0.05, 0.35, 0.10, -0.25, 0.45],
    ];
    jitter
        .iter()
        .map(|j| {
            base.iter()
                .zip(j.iter())
                .map(|(b, d)| vec![b + d])
                .collect()
        })
        .collect()
}

struct Packed {
    log_theta: Vec<f64>,
    log_sigma: Vec<f64>,
    theta_lower: Vec<f64>,
    theta_upper: Vec<f64>,
    sigma_lower: Vec<f64>,
    sigma_upper: Vec<f64>,
    mask: Vec<bool>,
}

fn packed_start() -> Packed {
    let theta = [1.0f64, 10.0, 2.0, 20.0];
    let sigma = [0.10f64];
    Packed {
        log_theta: theta.iter().map(|v| v.ln()).collect(),
        log_sigma: sigma.iter().map(|v| v.ln()).collect(),
        theta_lower: vec![0.05f64.ln(), 1.0f64.ln(), 0.05f64.ln(), 1.0f64.ln()],
        theta_upper: vec![20.0f64.ln(), 200.0f64.ln(), 50.0f64.ln(), 400.0f64.ln()],
        sigma_lower: vec![(-8.0f64)],
        sigma_upper: vec![2.0f64],
        mask: vec![true; 4],
    }
}

/// Converged single-draw maximiser of the frozen-η objective at `etas`.
fn maximiser_at(
    model: &CompiledModel,
    population: &Population,
    p: &Packed,
    etas: &[Vec<f64>],
    extra: &[Vec<Vec<f64>>],
) -> (Vec<f64>, Vec<f64>) {
    theta_sigma_mstep_light(
        model,
        population,
        etas,
        None,
        &p.log_theta,
        &p.log_sigma,
        &p.theta_lower,
        &p.theta_upper,
        &p.sigma_lower,
        &p.sigma_upper,
        4,
        1,
        400,
        false,
        &p.mask,
        None,
        &[],
        extra,
    )
}

// ── the K-draw objective ───────────────────────────────────────────────────

/// Averaging a draw with **itself** must be the identity, bit for bit: the mean
/// of `x` and `x` is `x` in IEEE arithmetic, so a K-draw objective that changed
/// the answer here would be averaging the wrong thing (or dividing twice).
///
/// This is the control for the test below it — without it, "the answer moved"
/// could be the averaging machinery rather than the second draw.
#[test]
fn k_draw_objective_with_a_duplicated_draw_is_the_single_draw_answer() {
    let model = no_eta_theta_model();
    let population = no_eta_population(&model, &[1.3, 12.0, 2.6, 26.0]);
    let p = packed_start();
    let draws = eta_draws();

    let (t1, s1) = maximiser_at(&model, &population, &p, &draws[0], &[]);
    let (t2, s2) = maximiser_at(&model, &population, &p, &draws[0], &[draws[0].clone()]);
    assert_eq!(t1, t2, "duplicating a draw changed theta");
    assert_eq!(s1, s2, "duplicating a draw changed sigma");
}

/// …and a genuinely different second draw must move it. Without this the test
/// above is satisfied by an implementation that ignores `extra_eta_draws`
/// entirely.
#[test]
fn k_draw_objective_moves_when_the_second_draw_differs() {
    let model = no_eta_theta_model();
    let population = no_eta_population(&model, &[1.3, 12.0, 2.6, 26.0]);
    let p = packed_start();
    let draws = eta_draws();

    let (t1, _) = maximiser_at(&model, &population, &p, &draws[0], &[]);
    let (t2, _) = maximiser_at(&model, &population, &p, &draws[0], &[draws[1].clone()]);
    let moved = t1
        .iter()
        .zip(t2.iter())
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        moved > 1e-3,
        "the second draw did not reach the objective: worst |Δ log θ| = {moved:.3e}"
    );
}

// ── the Jensen bias, and its removal ───────────────────────────────────────

/// The defect, exhibited as a number: the **mean of the per-draw maximisers**
/// is not the **maximiser of the mean objective**.
///
/// The historical M-step converges to the first of those (it blends in one
/// draw's maximiser each iteration); SAEM's target is the second. On this
/// fixture the no-ETA `TVQ` differs by 0.049 log units between them while the
/// mu-referenced `TVCL` differs by 0.0016 — the same ordering the busulfan
/// benchmark reports, and the reason the issue is about no-ETA θ specifically.
#[test]
fn averaging_maximisers_is_not_maximising_the_average() {
    let model = no_eta_theta_model();
    let population = no_eta_population(&model, &[1.3, 12.0, 2.6, 26.0]);
    let p = packed_start();
    let draws = eta_draws();

    let mut mean_of_maximisers = vec![0.0f64; 4];
    for d in &draws {
        let (t, _) = maximiser_at(&model, &population, &p, d, &[]);
        for (a, b) in mean_of_maximisers.iter_mut().zip(t.iter()) {
            assert!(b.is_finite(), "a per-draw maximiser was not finite");
            *a += b / draws.len() as f64;
        }
    }
    let (maximiser_of_mean, _) =
        maximiser_at(&model, &population, &p, &draws[0], &draws[1..].to_vec());

    let q = 2usize; // TVQ, the coordinate with no ETA
    let cl = 0usize; // TVCL, mu-referenceable
    let gap_q = (mean_of_maximisers[q] - maximiser_of_mean[q]).abs();
    let gap_cl = (mean_of_maximisers[cl] - maximiser_of_mean[cl]).abs();
    assert!(
        gap_q > 1e-2,
        "the Jensen gap on TVQ has vanished ({gap_q:.4e}) — this fixture can no longer \
         exhibit the defect #1458 is about, so every test built on it is vacuous"
    );
    assert!(
        gap_q > 5.0 * gap_cl,
        "TVQ gap {gap_q:.4e} is not clearly larger than TVCL's {gap_cl:.4e}"
    );
}

/// Run the score/information SA recursion over the fixed draw list until the
/// packed vector stops moving, and return it.
fn run_score_sa(
    model: &CompiledModel,
    population: &Population,
    p: &Packed,
    draws: &[Vec<Vec<f64>>],
    n_pass: usize,
) -> (MstepScoreSa, Vec<f64>, Vec<f64>) {
    let mut sa = MstepScoreSa::new(4, 1);
    let mut lt = p.log_theta.clone();
    let mut ls = p.log_sigma.clone();
    let mut k = 0usize;
    for _ in 0..n_pass {
        for d in draws {
            k += 1;
            // The production schedule: γ = 1 while exploring, then 1/(k − k1).
            let gamma = if k <= draws.len() {
                1.0
            } else {
                1.0 / (k - draws.len()) as f64
            };
            sa.step(
                model,
                population,
                d,
                &mut lt,
                &mut ls,
                &p.theta_lower,
                &p.theta_upper,
                &p.sigma_lower,
                &p.sigma_upper,
                &p.mask,
                gamma,
                &[],
            );
        }
    }
    (sa, lt, ls)
}

/// The fix: stochastic approximation on the score and information converges to
/// the maximiser of the **averaged** objective, not to the average of the
/// maximisers — so the gap the test above measures is closed.
///
/// Measured on this fixture: the SA fixed point sits 0.0043 log units from the
/// averaged-objective maximiser on `TVQ`, against the 0.049 the mean of
/// maximisers sits away from it — an 11× reduction. The bound is set from those
/// numbers with headroom, and the *comparison* against the biased estimator is
/// the assertion, not an absolute tolerance that a drifting fixture could
/// satisfy by accident.
#[test]
fn score_sa_converges_to_the_maximiser_of_the_averaged_objective() {
    let model = no_eta_theta_model();
    let population = no_eta_population(&model, &[1.3, 12.0, 2.6, 26.0]);
    let p = packed_start();
    let draws = eta_draws();

    let (maximiser_of_mean, _) =
        maximiser_at(&model, &population, &p, &draws[0], &draws[1..].to_vec());
    let mut mean_of_maximisers = vec![0.0f64; 4];
    for d in &draws {
        let (t, _) = maximiser_at(&model, &population, &p, d, &[]);
        for (a, b) in mean_of_maximisers.iter_mut().zip(t.iter()) {
            *a += b / draws.len() as f64;
        }
    }

    let (sa, lt, _) = run_score_sa(&model, &population, &p, &draws, 12);
    let (rejected, out_of_scope) = sa.counters();
    assert_eq!(out_of_scope, 0, "the fixture is inside the Gaussian scope");
    assert!(
        rejected * 4 < (12 * draws.len()) as u64,
        "the EM guard rejected {rejected} of {} steps — the recursion is not running",
        12 * draws.len()
    );

    let q = 2usize;
    assert!(lt[q].is_finite(), "TVQ left the real line");
    let sa_gap = (lt[q] - maximiser_of_mean[q]).abs();
    let biased_gap = (mean_of_maximisers[q] - maximiser_of_mean[q]).abs();
    assert!(
        sa_gap * 3.0 < biased_gap,
        "score_sa did not close the Jensen gap on TVQ: it sits {sa_gap:.4e} from the \
         averaged-objective maximiser while the average of maximisers sits {biased_gap:.4e} \
         away"
    );
}

/// The accumulators are a Robbins-Monro blend of quantities that are *linear*
/// in the draw — the property the whole approach rests on. Mutating the blend
/// to an assignment (`s_k = ∇_k`) or to a blend of the wrong sign fails here.
#[test]
fn score_sa_accumulators_are_a_robbins_monro_blend() {
    let model = no_eta_theta_model();
    let population = no_eta_population(&model, &[1.3, 12.0, 2.6, 26.0]);
    let p = packed_start();
    let draws = eta_draws();

    // One step at γ = 1 fixes `s_1 = ∇_1`, `I_1 = I_1`.
    let mut sa = MstepScoreSa::new(4, 1);
    let mut lt = p.log_theta.clone();
    let mut ls = p.log_sigma.clone();
    sa.step(
        &model,
        &population,
        &draws[0],
        &mut lt,
        &mut ls,
        &p.theta_lower,
        &p.theta_upper,
        &p.sigma_lower,
        &p.sigma_upper,
        &p.mask,
        1.0,
        &[],
    );
    let s1 = sa.score.clone();
    let i1 = sa.info.clone();
    assert!(
        s1.iter().any(|v| v.abs() > 1e-6),
        "the first step left an all-zero score — nothing below can fail"
    );

    // A second step at γ = 0.5, from the SAME packed point, so the fresh
    // contribution is computable independently.
    let mut sa_b = MstepScoreSa::new(4, 1);
    let mut lt_b = lt.clone();
    let mut ls_b = ls.clone();
    sa_b.step(
        &model,
        &population,
        &draws[1],
        &mut lt_b,
        &mut ls_b,
        &p.theta_lower,
        &p.theta_upper,
        &p.sigma_lower,
        &p.sigma_upper,
        &p.mask,
        1.0,
        &[],
    );
    let fresh_score = sa_b.score.clone();
    let fresh_info = sa_b.info.clone();

    let mut lt_c = lt.clone();
    let mut ls_c = ls.clone();
    sa.step(
        &model,
        &population,
        &draws[1],
        &mut lt_c,
        &mut ls_c,
        &p.theta_lower,
        &p.theta_upper,
        &p.sigma_lower,
        &p.sigma_upper,
        &p.mask,
        0.5,
        &[],
    );

    for a in 0..s1.len() {
        let want = s1[a] + 0.5 * (fresh_score[a] - s1[a]);
        assert!(
            (sa.score[a] - want).abs() <= 1e-9 * want.abs().max(1.0),
            "score[{a}] is not the Robbins-Monro blend: {} vs {want}",
            sa.score[a]
        );
    }
    for a in 0..i1.len() {
        let want = i1[a] + 0.5 * (fresh_info[a] - i1[a]);
        assert!(
            (sa.info[a] - want).abs() <= 1e-9 * want.abs().max(1.0),
            "info[{a}] is not the Robbins-Monro blend: {} vs {want}",
            sa.info[a]
        );
    }
}

/// A pinned coordinate — a `FIX`, or a θ the closed-form mu-reference shift has
/// already placed — must come back untouched, because the numerical M-step is
/// not the thing that moves it.
#[test]
fn score_sa_leaves_a_pinned_coordinate_alone() {
    let model = no_eta_theta_model();
    let population = no_eta_population(&model, &[1.3, 12.0, 2.6, 26.0]);
    let mut p = packed_start();
    // Pin TVQ.
    p.theta_lower[2] = p.log_theta[2];
    p.theta_upper[2] = p.log_theta[2];
    let draws = eta_draws();
    let (_, lt, _) = run_score_sa(&model, &population, &p, &draws, 3);
    assert_eq!(
        lt[2], p.log_theta[2],
        "a pinned TVQ moved under the score/information step"
    );
    assert!(
        (lt[0] - p.log_theta[0]).abs() > 1e-3,
        "nothing moved at all — the pin assertion above is vacuous"
    );
}

// ── the scope gate ─────────────────────────────────────────────────────────

#[test]
fn numerical_mstep_scope_gap_admits_a_plain_gaussian_model() {
    let model = no_eta_theta_model();
    assert_eq!(numerical_mstep_scope_gap(&model, 0, false), None);
}

#[test]
fn numerical_mstep_scope_gap_refuses_iov_and_mixtures() {
    let model = no_eta_theta_model();
    assert!(numerical_mstep_scope_gap(&model, 2, false).is_some());
    assert!(numerical_mstep_scope_gap(&model, 0, true).is_some());
}

#[test]
fn numerical_mstep_scope_gap_refuses_a_correlated_residual() {
    let model = parse_model_string(
        r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.04
  block_sigma (PROP1, PROP2) = [
0.04,
0.01, 0.09
  ]
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -CL/V * central
[scaling]
  y[CMT=1] = central / V
  y[CMT=2] = 2.0 * central / V
[error_model]
  CMT=1: DV ~ proportional(PROP1)
  CMT=2: DV ~ proportional(PROP2)
",
    )
    .expect("block_sigma fixture parses");
    assert!(numerical_mstep_scope_gap(&model, 0, false).is_some());
}

#[test]
fn numerical_mstep_scope_gap_refuses_a_residual_magnitude() {
    let model = parse_model_string(
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
    assert!(numerical_mstep_scope_gap(&model, 0, false).is_some());
}

// ── stored-draw re-centring ────────────────────────────────────────────────

/// A closed-form mu-reference shift moves `log TVP` and every subject's η
/// together. A stored draw that is not moved with them stops describing the
/// individual parameters it was drawn for, and the K-draw objective would then
/// average two different parameterisations.
#[test]
fn stored_draws_follow_the_mu_reference_shift() {
    let mut draws = vec![
        vec![vec![0.10f64, 1.0], vec![-0.20, 2.0]],
        vec![vec![0.30f64, 3.0], vec![0.05, 4.0]],
    ];
    recentre_eta_draws(&mut draws, 0, 0.25);
    assert_eq!(draws[0][0][0], 0.10 - 0.25);
    assert_eq!(draws[1][1][0], 0.05 - 0.25);
    // The other η coordinate is untouched.
    assert_eq!(draws[0][0][1], 1.0);
    assert_eq!(draws[1][1][1], 4.0);

    // A zero shift is a no-op, and an out-of-range η index cannot panic.
    let before = draws.clone();
    recentre_eta_draws(&mut draws, 0, 0.0);
    recentre_eta_draws(&mut draws, 7, 1.0);
    assert_eq!(draws, before);
}

#[test]
fn stored_draws_follow_a_per_subject_mu_reference_shift() {
    let mut draws = vec![vec![vec![0.10f64], vec![-0.20], vec![0.40]]];
    recentre_eta_draws_per_subject(&mut draws, 0, &[0.05, -0.10, f64::NAN]);
    assert_eq!(draws[0][0][0], 0.10 - 0.05);
    assert_eq!(draws[0][1][0], -0.20 + 0.10);
    // A non-finite shift is skipped rather than poisoning the stored draw —
    // the same rule the live η re-centring uses.
    assert_eq!(draws[0][2][0], 0.40);
}
