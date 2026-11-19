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

/// A deterministic uniform-ish stream in `[-1, 1)`, so the fixture has no RNG
/// dependency and no seed to drift.
fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// Number of subjects in the fixture. Large enough that each per-draw maximiser
/// is well determined — at six subjects the `TVQ`/`TVV2` ridge is so flat that
/// the reference maximiser itself is unreliable, and the measured "Jensen gap"
/// moved the *wrong* way when the draws were made more dispersed, which is the
/// signature of a reference that is noise rather than a point.
const N_SUBJ: usize = 24;

/// True η per subject, and the SD the draws are dispersed by.
fn etas_true() -> Vec<f64> {
    let mut st = 0x1458_u64;
    (0..N_SUBJ).map(|_| 0.30 * lcg(&mut st)).collect()
}

/// `N_SUBJ` subjects, six samples each spanning the distribution and the
/// terminal phase, simulated noise-free at a θ away from the starting values so
/// the M-step has somewhere to go.
fn no_eta_population(model: &CompiledModel, theta_true: &[f64]) -> Population {
    use crate::types::{DoseEvent, Population, Subject};
    let times = [0.25f64, 0.75, 2.0, 5.0, 12.0, 24.0];
    let mut scratch = EventPkParams::default();
    let subjects: Vec<Subject> = etas_true()
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

/// A deterministic stand-in for the E-step: `K_DRAWS` η draw-sets, dispersed
/// around the values the data were simulated at. Dispersion is the lever the
/// Jensen bias scales with, so it is generated here rather than sampled by the
/// estimator.
const K_DRAWS: usize = 8;

fn eta_draws() -> Vec<Vec<Vec<f64>>> {
    let base = etas_true();
    let mut st = 0x9E37_79B9_u64;
    (0..K_DRAWS)
        .map(|_| base.iter().map(|b| vec![b + 0.45 * lcg(&mut st)]).collect())
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
    maximiser_with(
        model,
        population,
        p,
        etas,
        extra,
        crate::types::MstepOptimizer::Bobyqa,
        400,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn maximiser_with(
    model: &CompiledModel,
    population: &Population,
    p: &Packed,
    etas: &[Vec<f64>],
    extra: &[Vec<Vec<f64>>],
    optimizer: crate::types::MstepOptimizer,
    maxiter: u32,
    scale_params: bool,
) -> (Vec<f64>, Vec<f64>) {
    maximiser_tallied(
        model,
        population,
        p,
        etas,
        extra,
        optimizer,
        maxiter,
        scale_params,
        None,
    )
}

/// [`maximiser_with`] recording what its Newton solve did in `tally`.
#[allow(clippy::too_many_arguments)]
fn maximiser_tallied(
    model: &CompiledModel,
    population: &Population,
    p: &Packed,
    etas: &[Vec<f64>],
    extra: &[Vec<Vec<f64>>],
    optimizer: crate::types::MstepOptimizer,
    maxiter: u32,
    scale_params: bool,
    tally: Option<&crate::estimation::mstep_newton::NewtonTally>,
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
        maxiter,
        scale_params,
        &p.mask,
        None,
        &[],
        extra,
        optimizer,
        tally,
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

/// Worst absolute difference over the four θ coordinates, in packed (log) units.
fn worst_theta_gap(a: &[f64], b: &[f64]) -> (usize, f64) {
    let mut worst = (0usize, 0.0f64);
    for i in 0..4 {
        assert!(
            a[i].is_finite() && b[i].is_finite(),
            "coordinate {i} is not finite: {} vs {}",
            a[i],
            b[i]
        );
        let d = (a[i] - b[i]).abs();
        if d > worst.1 {
            worst = (i, d);
        }
    }
    worst
}

/// The mean of the per-draw maximisers, which is what the historical M-step
/// converges to.
fn mean_of_maximisers(
    model: &CompiledModel,
    population: &Population,
    p: &Packed,
    draws: &[Vec<Vec<f64>>],
) -> Vec<f64> {
    let mut acc = vec![0.0f64; 4];
    for d in draws {
        let (t, _) = maximiser_at(model, population, p, d, &[]);
        for (a, b) in acc.iter_mut().zip(t.iter()) {
            assert!(b.is_finite(), "a per-draw maximiser was not finite");
            *a += b / draws.len() as f64;
        }
    }
    acc
}

/// The defect, exhibited as a number: the **mean of the per-draw maximisers**
/// is not the **maximiser of the mean objective**.
///
/// The historical M-step converges to the first of those (it adopts one draw's
/// maximiser each iteration); SAEM's target is the second. Realised on this
/// fixture, in log units: `TVCL` **3.73e-2**, `TVV2` 2.46e-2, `TVQ` 5.31e-3,
/// `TVV` below 1e-3. The bound below is half the worst realised value, and the
/// message names the coordinate so a fixture that stops exhibiting the defect
/// says which one went quiet.
///
/// Every test after this one measures against `maximiser_of_mean`, so if this
/// gap collapses they all become vacuous — which is why it is asserted here and
/// not merely assumed.
#[test]
fn averaging_maximisers_is_not_maximising_the_average() {
    let model = no_eta_theta_model();
    let population = no_eta_population(&model, &[1.3, 12.0, 2.6, 26.0]);
    let p = packed_start();
    let draws = eta_draws();

    let biased = mean_of_maximisers(&model, &population, &p, &draws);
    let (target, _) = maximiser_at(&model, &population, &p, &draws[0], &draws[1..].to_vec());

    let (coord, gap) = worst_theta_gap(&biased, &target);
    assert!(
        gap > 1.8e-2,
        "the Jensen gap has collapsed to {gap:.4e} (worst coordinate {coord}) — this fixture \
         can no longer exhibit the defect #1458 is about, so every test built on it is \
         vacuous. Realised when written: 3.73e-2 on TVCL."
    );
}

/// Run the score recursion over the fixed draw list and return the final packed
/// vector, plus the accumulator for its counters.
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
            // The production schedule: γ = 1 while exploring, then 1/(k − k1),
            // with the first pass standing in for the exploration phase.
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
                // `mstep_damping` off, which is the shipped default, so γ_σ is
                // `min(γ, SIGMA_SA_MAX_STEP)` exactly as in a production fit.
                &[sigma_mstep_sa_step(gamma, 1.0)],
                &[],
            );
        }
    }
    (sa, lt, ls)
}

/// The fix: the score recursion converges to the maximiser of the **averaged**
/// objective, where the average of maximisers does not.
///
/// Measured on this fixture, worst θ distance from `maximiser_of_mean` in log
/// units, against the 3.73e-2 the average of maximisers sits away:
///
/// | passes over the draw list | worst gap | ratio |
/// |---|---|---|
/// | 4  | 8.6e-4 | 43× |
/// | 12 | 7.4e-4 | 50× |
/// | 30 | 1.3e-4 | 287× |
///
/// The assertion is the **ratio**, not an absolute tolerance: an absolute bound
/// would be satisfied by a fixture whose Jensen gap had quietly shrunk, which is
/// exactly the failure `averaging_maximisers_is_not_maximising_the_average`
/// exists to catch. 10× is a quarter of the realised 43× at the pass count used
/// here.
///
/// This is also the test that dies if the recursion goes back to the form #1458
/// proposes (accumulate the score, take a **full** Newton step): that form is
/// marginally stable and realised 1.09e-2 here — a ratio of 3.4×, under the
/// bound. See `MstepScoreSa`'s docs for why.
#[test]
fn score_sa_converges_to_the_maximiser_of_the_averaged_objective() {
    let model = no_eta_theta_model();
    let population = no_eta_population(&model, &[1.3, 12.0, 2.6, 26.0]);
    let p = packed_start();
    let draws = eta_draws();

    let (target, _) = maximiser_at(&model, &population, &p, &draws[0], &draws[1..].to_vec());
    let biased = mean_of_maximisers(&model, &population, &p, &draws);
    let (_, biased_gap) = worst_theta_gap(&biased, &target);

    let (sa, lt, _) = run_score_sa(&model, &population, &p, &draws, 12);
    // Every step must have moved theta/sigma: this fixture is inside the
    // Gaussian scope, its information is well conditioned, and nothing should
    // land on a non-finite objective. A non-zero count in ANY slot would mean
    // the recursion below ran on fewer steps than it claims.
    assert_eq!(
        sa.failure_total(),
        0,
        "{} of {} steps did not move theta/sigma: {:?}",
        sa.failure_total(),
        12 * draws.len(),
        sa.failures()
    );

    let (coord, sa_gap) = worst_theta_gap(&lt, &target);
    assert!(
        sa_gap * 10.0 < biased_gap,
        "score_sa did not close the Jensen gap: its worst theta (coordinate {coord}) sits \
         {sa_gap:.4e} from the averaged-objective maximiser while the average of maximisers \
         sits {biased_gap:.4e} away — a ratio of {:.1}x, against a realised 50x",
        biased_gap / sa_gap.max(f64::MIN_POSITIVE)
    );
}

/// The K-draw objective shrinks the same gap, and the rate is what makes it a
/// *partial* remedy rather than a fix: the Jensen bias of a maximiser is second
/// order in the dispersion of what is maximised, so averaging `K` draws cuts it
/// by roughly `1/K` — it never removes it.
///
/// The comparison has to be like for like. One `K = 3` maximiser against the
/// mean of eight `K = 1` maximisers is not: the first is a single realisation
/// and the second is an eight-fold average, so their difference is dominated by
/// sampling noise (measured: 3.50e-2 against 3.07e-2 at `K = 3` and `K = 2`,
/// i.e. the wrong order, from noise alone). What is comparable is the **mean of
/// the `8/K` disjoint `K`-draw maximisers** at each `K`, each using all eight
/// draws exactly once.
///
/// Realised worst-θ distance from the eight-draw `maximiser_of_mean`, in log
/// units: `K = 1` → 3.73e-2, `K = 2` → see the assertion, `K = 4` → see the
/// assertion. The bound is monotone shrinkage with `K`, which a `K` that used
/// only its first extra draw would fail.
#[test]
fn the_k_draw_objective_shrinks_the_gap_as_k_grows() {
    let model = no_eta_theta_model();
    let population = no_eta_population(&model, &[1.3, 12.0, 2.6, 26.0]);
    let p = packed_start();
    let draws = eta_draws();
    assert_eq!(
        draws.len(),
        8,
        "the disjoint blocks below assume eight draws"
    );

    let (target, _) = maximiser_at(&model, &population, &p, &draws[0], &draws[1..].to_vec());

    // Mean over the `8/k` disjoint blocks of `k` draws.
    let mean_block_maximiser = |k: usize| -> Vec<f64> {
        let n_block = draws.len() / k;
        let mut acc = vec![0.0f64; 4];
        for b in 0..n_block {
            let block = &draws[b * k..(b + 1) * k];
            let (t, _) = maximiser_at(&model, &population, &p, &block[0], &block[1..].to_vec());
            for (a, v) in acc.iter_mut().zip(t.iter()) {
                assert!(v.is_finite(), "a K = {k} block maximiser was not finite");
                *a += v / n_block as f64;
            }
        }
        acc
    };

    let (_, g1) = worst_theta_gap(&mean_block_maximiser(1), &target);
    let (_, g2) = worst_theta_gap(&mean_block_maximiser(2), &target);
    let (_, g4) = worst_theta_gap(&mean_block_maximiser(4), &target);

    assert!(
        g2 < g1 && g4 < g2,
        "the K-draw gap must shrink with K: K=1 {g1:.4e}, K=2 {g2:.4e}, K=4 {g4:.4e}"
    );
    // …and it must still be there at K = 4, or the fixture has stopped being
    // able to show that K-draw averaging is a mitigation and not a cure.
    assert!(
        g4 > 1e-3,
        "K = 4 already removed the gap ({g4:.4e}) — then this fixture cannot distinguish a \
         partial remedy from a fix"
    );
}

/// The information accumulator is a Robbins-Monro blend of a quantity that is
/// *linear* in the draw — the matrix gain the score recursion is preconditioned
/// by. Mutating the blend to an assignment (`I_k = I_k(x_k, eta_k)`) or to a
/// blend of the wrong sign fails here.
#[test]
fn score_sa_information_is_a_robbins_monro_blend() {
    let model = no_eta_theta_model();
    let population = no_eta_population(&model, &[1.3, 12.0, 2.6, 26.0]);
    let p = packed_start();
    let draws = eta_draws();

    // One step at γ = 1 fixes `I_1 = I(x_1, η_1)`.
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
        &[sigma_mstep_sa_step(1.0, 1.0)],
        &[],
    );
    let i1 = sa.info.clone();
    assert!(
        i1.iter().any(|v| v.abs() > 1e-6),
        "the first step left an all-zero information — nothing below can fail"
    );

    // A second step at γ = 0.5, from the SAME packed point, so the fresh
    // contribution is computable independently by a fresh accumulator at γ = 1.
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
        &[sigma_mstep_sa_step(1.0, 1.0)],
        &[],
    );
    let fresh = sa_b.info.clone();

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
        &[sigma_mstep_sa_step(0.5, 1.0)],
        &[],
    );

    let mut moved = 0.0f64;
    for a in 0..i1.len() {
        let want = i1[a] + 0.5 * (fresh[a] - i1[a]);
        assert!(
            (sa.info[a] - want).abs() <= 1e-9 * want.abs().max(1.0),
            "info[{a}] is not the Robbins-Monro blend: {} vs {want}",
            sa.info[a]
        );
        moved = moved.max((fresh[a] - i1[a]).abs());
    }
    // If the two draws produced the same information, the blend is the identity
    // and the assertion above cannot tell a blend from an assignment.
    assert!(
        moved > 1e-6,
        "the two draws gave the same information ({moved:.3e}) — the blend is untested"
    );
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

fn correlated_residual_model() -> CompiledModel {
    parse_model_string(
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
    .expect("block_sigma fixture parses")
}

#[test]
fn numerical_mstep_scope_gap_refuses_a_correlated_residual() {
    assert!(numerical_mstep_scope_gap(&correlated_residual_model(), 0, false).is_some());
}

fn magnitude_model() -> CompiledModel {
    parse_model_string(
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
    .expect("magnitude fixture parses")
}

#[test]
fn numerical_mstep_scope_gap_refuses_a_residual_magnitude() {
    assert!(numerical_mstep_scope_gap(&magnitude_model(), 0, false).is_some());
}

/// #1421: a FREM model is in scope — its covariate rows' score and information
/// are exact since `obs_dvar_dlogsigma` (see the FD-parity tests in
/// `fixed_eta_gradient_tests.rs`), so `score_sa` and `mstep_draws` need not
/// refuse it any more.
fn frem_model() -> CompiledModel {
    parse_model_string(
        r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  theta TV_WT(70.0, FIX)
  omega ETA_CL ~ 0.04
  omega ETA_WT_FREM ~ 100.0
  sigma PROP_ERR ~ 0.10 (sd)
  sigma EPSCOV ~ 1e-6 FIX
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  frem_predictions = TV_WT/ETA_WT_FREM:100
  frem_sigma = EPSCOV
",
    )
    .expect("FREM fixture parses")
}

#[test]
fn numerical_mstep_scope_gap_admits_a_frem_model() {
    let model = frem_model();
    assert!(model.frem_config.is_some(), "the fixture must be FREM");
    assert_eq!(numerical_mstep_scope_gap(&model, 0, false), None);
}

/// One-compartment IV with IOV on clearance (κ live on a later dose).
fn iov_model() -> CompiledModel {
    parse_model_string(
        r"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  kappa KAPPA_CL ~ 0.04
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
",
    )
    .expect("IOV fixture parses")
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

/// #2548 item 2: the closed-form shift keeps every subject's individual parameter where it
/// was, `φᵢ = mu + ηᵢ` on the mu scale, even when the clamp bites. Each case drives θ into
/// its upper bound (`γ·mean(η)` would carry it past) and starts one already on it, for an
/// additive anchor (`P = θ + η`, natural scale) packed both ways and a log anchor
/// (`log P = log θ + η`). The re-centre must use the change the clamp let through, so a
/// mutation re-centring by the unclamped `γ·mean(η)` moves every φᵢ (by up to 2.0 here). The
/// straddle is asserted per case: the clamp held θ on the bound and the realised change
/// differs from `γ·mean(η)`, so the fixture can tell the two re-centres apart. The extra
/// stored draws are checked the same way.
#[test]
fn the_mu_shift_keeps_phi_at_a_bound() {
    let gamma = 1.0;
    // Additive: natural θ, upper bound 10. Log: log θ, upper bound log 10.
    for (case, additive, packs_log) in [
        ("additive, log-packed", true, true),
        ("additive, identity-packed", true, false),
        ("log", false, true),
    ] {
        let to_packed = |t: f64| if packs_log { t.ln() } else { t };
        let to_mu = |packed: f64| {
            if additive && packs_log {
                packed.exp()
            } else {
                packed
            }
        };
        let bounds = (to_packed(0.5), to_packed(10.0));
        // mean η 2.0 (additive) / 1.0 (log): both carry θ past the bound from 9.5. Realised
        // change from 9.5 is 0.5 (additive) / ln(10/9.5) = 0.051 (log), so the straddle
        // below clears its 0.4 bar by ≥ 0.55 in every case.
        let eta_mean = if additive { 2.0 } else { 1.0 };
        for start in [9.5, 10.0] {
            let mut packed = to_packed(start);
            let mut etas: Vec<Vec<f64>> = [-1.0, 0.0, 1.0]
                .iter()
                .map(|d| vec![eta_mean + 0.3 * d, 7.0])
                .collect();
            let mut draws: Vec<Vec<Vec<f64>>> = vec![etas.clone(), etas.clone()];
            let mu_before = to_mu(packed);
            let phi_before: Vec<f64> = etas.iter().map(|e| mu_before + e[0]).collect();
            closed_form_mu_shift(
                &mut packed,
                bounds,
                packs_log,
                additive,
                gamma,
                &mut etas,
                &mut draws,
                0,
            );
            let mu_after = to_mu(packed);
            let what = format!("{case}, θ from {start}");
            assert!(
                (mu_after - to_mu(bounds.1)).abs() < 1e-12,
                "{what}: the clamp holds θ on its bound, got {mu_after}"
            );
            assert!(
                ((mu_after - mu_before) - gamma * eta_mean).abs() > 0.4,
                "{what}: the fixture cannot tell the realised change from γ·mean(η)"
            );
            for (d, set) in std::iter::once(&etas).chain(draws.iter()).enumerate() {
                for (i, e) in set.iter().enumerate() {
                    let gap = (mu_after + e[0]) - phi_before[i];
                    assert!(
                        gap.is_finite() && gap.abs() < 1e-12,
                        "{what}: draw {d}, subject {i} moved by {gap:.3e}"
                    );
                    assert_eq!(e[1], 7.0, "{what}: the other η is untouched");
                }
            }
        }
    }
}

/// Every early return from `MstepScoreSa::step` must be **counted**, or a run
/// can lose its numerical M-step on every iteration and report nothing — the
/// objective is fine and the parameters simply stop being estimated, which is
/// invisible in the trace.
///
/// The callers ignore `step`'s return value on purpose (a held θ/σ is not a
/// fatal condition), so the counter *is* the only channel, and this test is
/// what stops a new early return being added without one. It drives three of
/// the six routes directly; the gate-bug route (`OutOfScope`) is unreachable
/// from an in-scope fixture by construction, and `NonFiniteTerms` /
/// `NonFiniteDirection` are asserted to be reachable-but-unfired here.
#[test]
fn every_failure_route_is_counted() {
    let model = no_eta_theta_model();
    let population = no_eta_population(&model, &[1.3, 12.0, 2.6, 26.0]);
    let draws = eta_draws();

    // Route: no free coordinate. Pin every theta AND the sigma, so the free
    // set is empty and there is nothing to solve for.
    let mut p = packed_start();
    for i in 0..4 {
        p.theta_lower[i] = p.log_theta[i];
        p.theta_upper[i] = p.log_theta[i];
    }
    p.sigma_lower[0] = p.log_sigma[0];
    p.sigma_upper[0] = p.log_sigma[0];

    let mut sa = MstepScoreSa::new(4, 1);
    let mut lt = p.log_theta.clone();
    let mut ls = p.log_sigma.clone();
    let moved = sa.step(
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
        &[sigma_mstep_sa_step(1.0, 1.0)],
        &[],
    );
    assert!(
        !moved,
        "a fully pinned step must report that it did not move"
    );
    assert_eq!(
        sa.failures()[ScoreSaFailure::NoFreeCoordinate as usize],
        1,
        "the fully-pinned route was not counted: {:?}",
        sa.failures()
    );
    assert_eq!(sa.failure_total(), 1, "exactly one route should have fired");
    assert_eq!(lt, p.log_theta, "a failed step must leave theta untouched");
    assert_eq!(ls, p.log_sigma, "a failed step must leave sigma untouched");

    // …and the same accumulator keeps counting rather than latching.
    let mut lt2 = p.log_theta.clone();
    let mut ls2 = p.log_sigma.clone();
    sa.step(
        &model,
        &population,
        &draws[1],
        &mut lt2,
        &mut ls2,
        &p.theta_lower,
        &p.theta_upper,
        &p.sigma_lower,
        &p.sigma_upper,
        &p.mask,
        1.0,
        &[sigma_mstep_sa_step(1.0, 1.0)],
        &[],
    );
    assert_eq!(sa.failure_total(), 2, "the counter must accumulate");

    // Control: the SAME model with the coordinates free must succeed and count
    // nothing, so the assertions above are about the pin and not about the
    // fixture being broken.
    let q = packed_start();
    let mut sa_ok = MstepScoreSa::new(4, 1);
    let mut lt3 = q.log_theta.clone();
    let mut ls3 = q.log_sigma.clone();
    let moved_ok = sa_ok.step(
        &model,
        &population,
        &draws[0],
        &mut lt3,
        &mut ls3,
        &q.theta_lower,
        &q.theta_upper,
        &q.sigma_lower,
        &q.sigma_upper,
        &q.mask,
        1.0,
        &[sigma_mstep_sa_step(1.0, 1.0)],
        &[],
    );
    assert!(moved_ok, "the free fixture must take a step");
    assert_eq!(
        sa_ok.failure_total(),
        0,
        "the free fixture counted a failure: {:?}",
        sa_ok.failures()
    );
    assert_ne!(lt3, q.log_theta, "the free fixture did not move theta");
}

/// The failure names are indexed by the enum discriminant, so a new variant
/// added without a name would print the wrong reason — or panic.
#[test]
fn every_failure_reason_has_a_name() {
    for (i, name) in MSTEP_SA_FAILURE_NAMES.iter().enumerate() {
        assert!(!name.is_empty(), "reason {i} has no name");
    }
    assert_eq!(MSTEP_SA_FAILURE_NAMES.len(), MSTEP_SA_FAILURE_KINDS);
    // Discriminants must be dense and in order, since they index the array.
    assert_eq!(ScoreSaFailure::OutOfScope as usize, 0);
    assert_eq!(ScoreSaFailure::NonFiniteTerms as usize, 1);
    assert_eq!(ScoreSaFailure::NoFreeCoordinate as usize, 2);
    assert_eq!(ScoreSaFailure::NotPositiveDefinite as usize, 3);
    assert_eq!(ScoreSaFailure::NonFiniteDirection as usize, 4);
    assert_eq!(
        ScoreSaFailure::NoFiniteObjective as usize,
        MSTEP_SA_FAILURE_KINDS - 1
    );
}

// ── option resolution and the end-of-run report ────────────────────────────
//
// Both are pure functions precisely so they can be tested here: the logic
// otherwise lives inside `run_saem` and is reachable only by a full fit, which
// means the four warning strings would be exercised by nothing that runs on a
// PR (slow-gated tests contribute no patch coverage).

/// In scope, `score_sa` is adopted silently and `mstep_draws` passes through.
#[test]
fn resolve_mstep_options_adopts_both_in_scope() {
    let (sa, k, w) = resolve_mstep_options(SaemMstepSolver::ScoreSa, 1, None);
    assert!(sa, "score_sa must be adopted in scope");
    assert_eq!(k, 1);
    assert!(w.is_empty(), "no warning expected in scope: {w:?}");

    let (sa, k, w) = resolve_mstep_options(SaemMstepSolver::Bobyqa, 3, None);
    assert!(!sa);
    assert_eq!(k, 3, "mstep_draws must pass through in scope");
    assert!(w.is_empty(), "{w:?}");
}

/// The default is the historical solver with one draw, and says nothing.
#[test]
fn resolve_mstep_options_default_is_silent_and_historical() {
    let (sa, k, w) = resolve_mstep_options(SaemMstepSolver::Bobyqa, 1, None);
    assert!(!sa);
    assert_eq!(k, 1);
    assert!(w.is_empty(), "the default must not warn: {w:?}");
    // …and a scope gap changes nothing when neither option was asked for.
    let (sa, k, w) = resolve_mstep_options(SaemMstepSolver::Bobyqa, 1, Some("the model has IOV"));
    assert!(!sa);
    assert_eq!(k, 1);
    assert!(w.is_empty(), "an unused option must not warn: {w:?}");
}

/// Out of scope, each option is declined **by name**, and the reason is
/// repeated so the user can act on it.
#[test]
fn resolve_mstep_options_declines_out_of_scope_by_name() {
    let reason = "the model has IOV";
    let (sa, _, w) = resolve_mstep_options(SaemMstepSolver::ScoreSa, 1, Some(reason));
    assert!(!sa, "score_sa must not be adopted out of scope");
    assert_eq!(w.len(), 1, "{w:?}");
    assert!(
        w[0].contains("score_sa") && w[0].contains(reason),
        "{}",
        w[0]
    );

    let (_, k, w) = resolve_mstep_options(SaemMstepSolver::Bobyqa, 3, Some(reason));
    assert_eq!(k, 1, "mstep_draws must fall back to one draw");
    assert_eq!(w.len(), 1, "{w:?}");
    assert!(
        w[0].contains("mstep_draws") && w[0].contains(reason),
        "{}",
        w[0]
    );
}

/// The two never stack: `mstep_draws` is ignored under `score_sa`, and said so.
#[test]
fn resolve_mstep_options_does_not_stack_the_two() {
    let (sa, k, w) = resolve_mstep_options(SaemMstepSolver::ScoreSa, 4, None);
    assert!(sa);
    assert_eq!(k, 1, "mstep_draws must be neutralised under score_sa");
    assert_eq!(w.len(), 1, "{w:?}");
    assert!(
        w[0].contains("ignored") && w[0].contains("mstep_draws"),
        "{}",
        w[0]
    );
}

/// `mstep_draws = 0` is a `FitOptions` a Rust caller can build directly (the
/// parser rejects it, `FitOptions` is public), and must not produce a zero-draw
/// objective.
#[test]
fn resolve_mstep_options_floors_zero_draws_at_one() {
    let (_, k, w) = resolve_mstep_options(SaemMstepSolver::Bobyqa, 0, None);
    assert_eq!(k, 1, "zero draws must floor at one");
    assert!(w.is_empty(), "flooring is silent: {w:?}");
}

/// No failures, no warning — otherwise every clean `score_sa` fit would carry
/// a spurious one.
#[test]
fn score_sa_failure_warning_is_silent_when_every_step_moved() {
    assert!(score_sa_failure_warning([0; MSTEP_SA_FAILURE_KINDS], 400).is_none());
}

/// Every route that fired is named with its count; routes that did not fire are
/// left out. This is the assertion that stops a new `ScoreSaFailure` variant
/// being added without a reason string, and stops the report collapsing to a
/// bare total.
#[test]
fn score_sa_failure_warning_names_each_route_that_fired() {
    let mut f = [0u64; MSTEP_SA_FAILURE_KINDS];
    f[ScoreSaFailure::NotPositiveDefinite as usize] = 7;
    f[ScoreSaFailure::NoFiniteObjective as usize] = 2;
    let w = score_sa_failure_warning(f, 400).expect("a failure must be reported");

    assert!(w.contains("9 of 400"), "the total must be the sum: {w}");
    assert!(
        w.contains("7x")
            && w.contains(MSTEP_SA_FAILURE_NAMES[ScoreSaFailure::NotPositiveDefinite as usize]),
        "{w}"
    );
    assert!(
        w.contains("2x")
            && w.contains(MSTEP_SA_FAILURE_NAMES[ScoreSaFailure::NoFiniteObjective as usize]),
        "{w}"
    );
    // A route that did not fire must not be mentioned — a report that names
    // every reason every time tells the user nothing.
    assert!(
        !w.contains(MSTEP_SA_FAILURE_NAMES[ScoreSaFailure::OutOfScope as usize]),
        "a route that did not fire was named: {w}"
    );
    assert!(w.contains("score_sa") && w.contains("#1458"), "{w}");
}

/// The gate-bug route is reportable too — it is the one that should never fire,
/// so it is the one most worth naming if it does.
#[test]
fn score_sa_failure_warning_reports_the_gate_bug_route() {
    let mut f = [0u64; MSTEP_SA_FAILURE_KINDS];
    f[ScoreSaFailure::OutOfScope as usize] = 1;
    let w = score_sa_failure_warning(f, 10).expect("reported");
    assert!(w.contains("1 of 10"), "{w}");
    assert!(
        w.contains("report it"),
        "the gate-bug route must ask for a report: {w}"
    );
}

// ── #1480: σ takes the #1445 policy, θ does not ────────────────────────────

/// One `step` from a fixed point, returning `(log θ, log σ)`.
fn one_step(gamma: f64, gamma_mstep: f64) -> (Vec<f64>, Vec<f64>) {
    let model = no_eta_theta_model();
    let population = no_eta_population(&model, &[1.3, 12.0, 2.6, 26.0]);
    let p = packed_start();
    let draws = eta_draws();
    let mut sa = MstepScoreSa::new(4, 1);
    let mut lt = p.log_theta.clone();
    let mut ls = p.log_sigma.clone();
    let moved = sa.step(
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
        gamma,
        &[sigma_mstep_sa_step(gamma, gamma_mstep)],
        &[],
    );
    assert!(
        moved,
        "the fixture must take a step at γ = {gamma}, γ_mstep = {gamma_mstep}; \
         failures: {:?}",
        sa.failures()
    );
    (lt, ls)
}

/// σ's step size is [`sigma_mstep_sa_step`]'s, θ's is the shared γ (#1480).
///
/// **The regression this exists to catch** is the shipped #1462 form: σ scaled
/// by the same `g_eff` as θ. During exploration `γ = 1`, so that is a *full*
/// Newton step to one draw's score root — no Robbins-Monro averaging at all —
/// and it re-opened the #1445 additive-σ collapse on #1445's own fixture
/// (`ADD_ERR` 0.84 / 0.52 / 1.13 against a truth of 1.8). The smallest edit that
/// restores it is replacing `g_sigma` with `g_eff` in the apply loop of
/// [`MstepScoreSa::step`]; that edit makes both arms below identical and kills
/// the ratio assertion.
///
/// **Why a ratio and not a value.** The two arms differ *only* in `gamma_mstep`,
/// which reaches nothing but γ_σ: the per-subject score and information, the
/// accumulator blend (`g_info` is γ, not γ_σ) and the Newton direction `d` are
/// bit-identical, so the θ half must come back bit-identical and the σ half must
/// differ by exactly the ratio of the two γ_σ. `sigma_mstep_sa_step(1, 1) = 0.2`
/// and `sigma_mstep_sa_step(1, 0.05) = 0.05`, a ratio of 4 — on σ², since #1445's
/// blend is on the variance scale, which is the second half of the fix and is
/// what the closed form below reads.
///
/// The no-clamp preconditions are asserted rather than assumed: a σ target that
/// hit [`SCORE_SA_MAX_STEP`] or a packed bound would make the closed form false
/// and the test would then be measuring the clamp.
#[test]
fn score_sa_steps_sigma_on_the_1445_schedule_and_theta_on_gamma() {
    let p = packed_start();
    let (lt_a, ls_a) = one_step(1.0, 1.0); // γ_σ = min(1, 0.2, 1)    = 0.2
    let (lt_b, ls_b) = one_step(1.0, 0.05); // γ_σ = min(1, 0.2, 0.05) = 0.05

    // θ is untouched by γ_σ, bit-for-bit.
    assert_eq!(
        lt_a, lt_b,
        "γ_mstep must not reach θ under score_sa: {lt_a:?} vs {lt_b:?}"
    );
    // ...and θ actually moved, or the equality above is an equality of two
    // frozen vectors and cannot fail.
    let theta_moved = worst_theta_gap(&lt_a, &p.log_theta).1;
    assert!(
        theta_moved > 1e-3,
        "θ did not move ({theta_moved:.3e}) — the θ half of this test is vacuous"
    );

    // σ moved, in both arms, and by different amounts.
    let (s0, sa_, sb) = (p.log_sigma[0].exp(), ls_a[0].exp(), ls_b[0].exp());
    assert!(
        sa_.is_finite() && sb.is_finite(),
        "σ must stay finite: {sa_} / {sb}"
    );
    assert!(
        (sa_ - s0).abs() > 1e-6 && (sb - s0).abs() > 1e-6,
        "σ did not move from {s0}: {sa_} / {sb} — a frozen σ satisfies the \
         ratio below trivially"
    );

    // The closed form. Both arms blend the SAME target σ_t on the variance
    // scale, so σ² − σ₀² is linear in γ_σ and the ratio is exactly 0.2 / 0.05.
    let num = sa_ * sa_ - s0 * s0;
    let den = sb * sb - s0 * s0;
    let ratio = num / den;
    assert!(
        (ratio - 4.0).abs() < 1e-9,
        "σ² must move at γ_σ = min(γ, {SIGMA_SA_MAX_STEP}, γ_mstep): realised ratio \
         {ratio:.9} against the closed-form 4.0 (σ₀ = {s0:.6}, γ_σ = 0.2 → {sa_:.6}, \
         γ_σ = 0.05 → {sb:.6}). A ratio of 1 is σ riding θ's γ, which is the \
         #1462 form this test exists to reject."
    );

    // Preconditions for that closed form: the shared target is inside the trust
    // region and inside the packed bounds, so neither clamp bound it.
    let target = s0 * s0 + (sa_ * sa_ - s0 * s0) / 0.2;
    let log_target = 0.5 * target.ln();
    assert!(
        (log_target - p.log_sigma[0]).abs() < SCORE_SA_MAX_STEP,
        "the σ target moved {:.4} log units, at or past the {SCORE_SA_MAX_STEP} trust \
         region — the ratio above would then be measuring the clamp",
        (log_target - p.log_sigma[0]).abs()
    );
    assert!(
        log_target > p.sigma_lower[0] && log_target < p.sigma_upper[0],
        "the σ target {log_target} is on a packed bound ({}, {})",
        p.sigma_lower[0],
        p.sigma_upper[0]
    );
}

/// σ is blended on the **variance** scale under `score_sa` too (#1480), which is
/// the larger half of the fix: #1445 measured the packed-log spelling at
/// `ADD_ERR` 0.277 against a truth of 1.8, where the schedule alone left seed 1
/// at 0.87, under the 0.9 gate.
///
/// **The regression this exists to catch**: re-spelling the σ half of the step
/// as an in-place `ls[j] += γ_σ · step`, i.e. the pre-#1445
/// `damp_mstep(&mut log_sigma, …)` written as a step. That edit leaves
/// `score_sa_steps_sigma_on_the_1445_schedule_and_theta_on_gamma` **green** —
/// the log blend is linear in γ_σ too, so that test's ratio would come back 4.0
/// on `log σ` rather than on `σ²` — which is why the scale needs its own test.
///
/// **What is asserted, and why it is not the closed form.** Recovering the
/// target from one arm and re-deriving σ with the same blend is an identity: it
/// comes back green under either spelling, and under *any* spelling. So this
/// test never names the target. It takes three arms whose only difference is
/// γ_σ (0.05 / 0.10 / 0.15, set through `gamma_mstep`; everything upstream —
/// score, information, Newton direction, target — is bit-identical) and asserts
/// that **σ² is affine in γ_σ**, which is what a variance-scale blend
/// `σ² = (1−γ)σ₀² + γσ_t²` is and what a log-scale blend
/// `σ² = σ₀²·(σ_t²/σ₀²)^γ` is not. Equally spaced γ, so the statement is a zero
/// second difference — analytically exact, so the bound is floating-point only.
///
/// Realised when written: σ = 0.104207 / 0.108251 / 0.112149 (σ starts at 0.10
/// and the score pushes it *up* on this noise-free fixture), second difference
/// **−1.041e-17** on a σ² of order 1.2e-2, against **6.307e-5** for the log
/// spelling at the same two endpoints — a discriminator of 6e12×. The bound is
/// 1e-9: 1e8× above the realised value and 6.3e4× below the spelling it
/// rejects. The spread guard is what stops a frozen σ (every arm equal, second
/// difference trivially zero) reading as a pass; that is exactly the state the
/// `g_eff`-for-`g_sigma` mutation produces, since γ ≥ 1 makes
/// `damp_mstep_sigma_variance` assign and the three arms collapse to one value
/// — so this test dies under **both** halves of the #1462 form, and the
/// realised spread is 7.6e-2 against the 1e-3 guard.
#[test]
fn score_sa_blends_sigma_on_the_variance_scale() {
    // γ_σ = min(1, SIGMA_SA_MAX_STEP, γ_mstep) = γ_mstep for these three.
    let gs = [0.05_f64, 0.10, 0.15];
    let sig: Vec<f64> = gs.iter().map(|&g| one_step(1.0, g).1[0].exp()).collect();
    for (g, s) in gs.iter().zip(sig.iter()) {
        assert!(s.is_finite() && *s > 0.0, "σ at γ_σ = {g} is {s}");
    }

    // Vacuity guard: σ must actually move across the three arms, or a zero
    // second difference says nothing. (A frozen σ is what the schedule mutation
    // produces — γ ≥ 1 assigns, so all three arms would be the same number.)
    let spread = (sig[2] - sig[0]).abs() / sig[0];
    assert!(
        spread > 1e-3,
        "σ barely moved across γ_σ = {gs:?}: {sig:?} (relative spread {spread:.2e}) — the \
         second difference below would be zero for either spelling"
    );

    // σ² is affine in γ_σ ⇔ its second difference over equally spaced γ_σ is
    // zero. Exact for the variance blend; the log blend is exponential in γ_σ.
    let v: Vec<f64> = sig.iter().map(|s| s * s).collect();
    let second = v[0] - 2.0 * v[1] + v[2];
    // Floating-point only: the statement is exact.
    const SIGMA_AFFINE_TOL: f64 = 1e-9;
    // What the packed-log spelling would have produced at these same three
    // points, from the same endpoints: σ²(γ) = v0·(v2/v0)^((γ−γ0)/(γ2−γ0)).
    let log_mid = v[0] * (v[2] / v[0]).sqrt();
    let log_second = v[0] - 2.0 * log_mid + v[2];
    assert!(
        log_second.abs() > 1e3 * SIGMA_AFFINE_TOL,
        "the two spellings are indistinguishable on this fixture (log-spelling second \
         difference {log_second:.3e}) — this test is vacuous"
    );
    assert!(
        second.abs() < SIGMA_AFFINE_TOL,
        "σ² must be affine in γ_σ (a variance-scale blend): realised second difference \
         {second:.3e} over γ_σ = {gs:?}, σ = {sig:?}. The packed-log spelling this test \
         rejects would give {log_second:.3e} at the same points."
    );
}

/// Score-SA's acceptance gate reads `obs_nll_sum`, whose per-subject scorers return
/// the shared `SUBJECT_SENTINEL_NLL` for a subject that cannot be evaluated (#2269).
/// One such subject (a NaN η makes its prediction NaN) puts the sum on the rail, and
/// the gate must refuse it; the same population with a finite η must pass, else the
/// refusal proves nothing. A gate with its own threshold below the sum would accept
/// the failed point, and one above a lowered sentinel would too — the shared
/// predicate is what keeps the two from drifting.
#[test]
fn score_sa_gate_refuses_a_population_holding_a_sentinel_subject() {
    use crate::stats::likelihood::SUBJECT_SENTINEL_NLL;
    let model = no_eta_theta_model();
    let population = no_eta_population(&model, &[1.3, 12.0, 2.6, 26.0]);
    let p = packed_start();
    let theta: Vec<f64> = p.log_theta.iter().map(|v| v.exp()).collect();
    let sigma: Vec<f64> = p.log_sigma.iter().map(|v| v.exp()).collect();

    let ok_draws = eta_draws().remove(0);
    let ok = obs_nll_sum(&model, &population, &theta, &sigma, &ok_draws, &[]);
    assert!(
        score_sa_objective_usable(ok),
        "a finite population must pass: {ok}"
    );

    let mut bad_draws = ok_draws.clone();
    bad_draws[0][0] = f64::NAN;
    let bad = obs_nll_sum(&model, &population, &theta, &sigma, &bad_draws, &[]);
    assert!(
        bad >= SUBJECT_SENTINEL_NLL,
        "one unevaluable subject must put the sum on the rail: {bad}"
    );
    assert!(
        !score_sa_objective_usable(bad),
        "the gate accepted a population holding a sentinel subject: {bad}"
    );
}

// ── `mstep_optimizer = lbfgs` ──────────────────────────────────────────────
//
// The L-BFGS arm drives the same objective through its exact fixed-η score
// (`obs_nll_subject_grad`: adjoint θ score, analytic σ score), so a wrong
// assembled gradient (the K-draw mean, the `scale` chain rule, a pinned
// coordinate) lands L-BFGS on a different point than BOBYQA's converged one.
// The two solvers share nothing but the objective, which is what makes the
// agreement an oracle for the gradient the way a finite-difference parity is.

/// Worst packed-space gap between two `(θ, σ)` results, over every coordinate.
fn worst_gap(a: &(Vec<f64>, Vec<f64>), b: &(Vec<f64>, Vec<f64>)) -> f64 {
    let mut worst = 0.0f64;
    for (x, y) in
        a.0.iter()
            .chain(a.1.iter())
            .zip(b.0.iter().chain(b.1.iter()))
    {
        assert!(
            x.is_finite() && y.is_finite(),
            "non-finite coordinate {x} vs {y}"
        );
        worst = worst.max((x - y).abs()); // nan-ok: both operands asserted finite above
    }
    worst
}

fn conditional_objective(
    model: &CompiledModel,
    population: &Population,
    etas: &[Vec<f64>],
    extra: &[Vec<Vec<f64>>],
    r: &(Vec<f64>, Vec<f64>),
) -> f64 {
    let th: Vec<f64> = r.0.iter().map(|x| x.exp()).collect();
    let sg: Vec<f64> = r.1.iter().map(|x| x.exp()).collect();
    let mut v = obs_nll_sum(model, population, &th, &sg, etas, &[]);
    for d in extra {
        v += obs_nll_sum(model, population, &th, &sg, d, &[]);
    }
    v / (1 + extra.len()) as f64
}

/// Converged L-BFGS = converged BOBYQA, single draw, with and without the
/// per-coordinate magnitude scaling (whose `g * scale` chain rule only the
/// gradient arm exercises).
#[test]
fn lbfgs_mstep_reaches_the_bobyqa_maximiser() {
    use crate::types::MstepOptimizer::{Bobyqa, Lbfgs};
    let model = no_eta_theta_model();
    let population = no_eta_population(&model, &[1.3, 12.0, 2.6, 26.0]);
    let p = packed_start();
    let draws = eta_draws();
    let reference = maximiser_with(&model, &population, &p, &draws[0], &[], Bobyqa, 900, false);
    let f_ref = conditional_objective(&model, &population, &draws[0], &[], &reference);
    for scale in [false, true] {
        let got = maximiser_with(&model, &population, &p, &draws[0], &[], Lbfgs, 400, scale);
        let f = conditional_objective(&model, &population, &draws[0], &[], &got);
        let gap = worst_gap(&got, &reference);
        eprintln!(
            "lbfgs scale={scale}: worst |dx| = {gap:.3e}, dObj = {:.3e}",
            f - f_ref
        );
        assert!(
            f - f_ref < 1e-5,
            "scale={scale}: lbfgs objective {f} vs bobyqa {f_ref}"
        );
        assert!(
            gap < 5e-3,
            "scale={scale}: lbfgs {got:?} vs bobyqa {reference:?}"
        );
    }
}

/// The K-draw mean objective's gradient (`mstep_draws > 1`) is the mean of the
/// per-draw gradients: L-BFGS on it must land on BOBYQA's maximiser of the mean.
#[test]
fn lbfgs_mstep_reaches_the_bobyqa_maximiser_of_the_k_draw_mean() {
    use crate::types::MstepOptimizer::{Bobyqa, Lbfgs};
    let model = no_eta_theta_model();
    let population = no_eta_population(&model, &[1.3, 12.0, 2.6, 26.0]);
    let p = packed_start();
    let draws = eta_draws();
    let extra = draws[1..4].to_vec();
    let reference = maximiser_with(
        &model,
        &population,
        &p,
        &draws[0],
        &extra,
        Bobyqa,
        900,
        false,
    );
    let got = maximiser_with(
        &model,
        &population,
        &p,
        &draws[0],
        &extra,
        Lbfgs,
        400,
        false,
    );
    // Control: the single-draw maximiser is elsewhere, so a K-draw gradient that
    // ignored the extra draws would not pass.
    let single = maximiser_with(&model, &population, &p, &draws[0], &[], Bobyqa, 900, false);
    let gap = worst_gap(&got, &reference);
    let apart = worst_gap(&single, &reference);
    eprintln!("lbfgs K-draw: worst |dx| = {gap:.3e}; single-draw maximiser is {apart:.3e} away");
    assert!(
        apart > 10.0 * gap.max(1e-4),
        "fixture: the extra draws must move the maximiser"
    );
    assert!(gap < 5e-3, "lbfgs {got:?} vs bobyqa {reference:?}");
}

/// A pinned coordinate (`lower == upper`, a closed-form-owned or FIX theta) stays put
/// under L-BFGS and the free ones still reach the optimum of the restricted problem.
#[test]
fn lbfgs_mstep_holds_a_pinned_coordinate() {
    use crate::types::MstepOptimizer::{Bobyqa, Lbfgs};
    let model = no_eta_theta_model();
    let population = no_eta_population(&model, &[1.3, 12.0, 2.6, 26.0]);
    let mut p = packed_start();
    let pin = p.log_theta[1];
    p.theta_lower[1] = pin;
    p.theta_upper[1] = pin;
    let draws = eta_draws();
    let reference = maximiser_with(&model, &population, &p, &draws[0], &[], Bobyqa, 900, false);
    let got = maximiser_with(&model, &population, &p, &draws[0], &[], Lbfgs, 400, false);
    assert_eq!(got.0[1], pin, "the pinned theta moved");
    let gap = worst_gap(&got, &reference);
    eprintln!("lbfgs pinned: worst |dx| = {gap:.3e}");
    assert!(gap < 5e-3, "lbfgs {got:?} vs bobyqa {reference:?}");
}

/// At the production budget (a handful of NLopt evaluations) the L-BFGS solve
/// must still capture the gain of a converged solve, from the same 0.05-log start
/// the BOBYQA budget test (`budgeted_mstep_returns_the_converged_conditional_maximiser`)
/// uses.
#[test]
fn lbfgs_mstep_at_the_production_budget_captures_the_converged_gain() {
    use crate::types::MstepOptimizer::{Bobyqa, Lbfgs};
    let model = no_eta_theta_model();
    let population = no_eta_population(&model, &[1.3, 12.0, 2.6, 26.0]);
    let draws = eta_draws();
    let mut p = packed_start();
    let reference = maximiser_with(&model, &population, &p, &draws[0], &[], Bobyqa, 900, false);
    for (v, r) in p.log_theta.iter_mut().zip(reference.0.iter()) {
        *v = r + 0.05;
    }
    p.log_sigma[0] = reference.1[0] + 0.05;
    let f_ref = conditional_objective(&model, &population, &draws[0], &[], &reference);
    let f_start = conditional_objective(
        &model,
        &population,
        &draws[0],
        &[],
        &(p.log_theta.clone(), p.log_sigma.clone()),
    );
    assert!(
        f_start > f_ref + 0.5,
        "fixture: the start must be off the optimum"
    );
    for maxiter in [3u32, 5] {
        let got = maximiser_with(
            &model,
            &population,
            &p,
            &draws[0],
            &[],
            Lbfgs,
            maxiter,
            false,
        );
        let f = conditional_objective(&model, &population, &draws[0], &[], &got);
        let captured = (f_start - f) / (f_start - f_ref);
        eprintln!("lbfgs maxiter {maxiter}: captured {captured:.4} of the converged gain");
        assert!(captured > 0.9, "maxiter {maxiter}: captured {captured}");
    }
}

/// The advisory for `mstep_optimizer = lbfgs` names each way the option can fail to do what
/// it says, and only those: silent under the default, silent where the θ score is exact, one
/// message per cause otherwise. Each assertion carries its cause's own discriminator and the
/// other's negative, so a message that names both cannot pass.
#[test]
fn the_lbfgs_advisory_names_its_cause() {
    use crate::types::MstepOptimizer::{Bobyqa, Lbfgs};
    let advisory =
        |o, sa, exact, label| resolve_mstep_optimizer_for(o, sa, exact, None, true, false, label).1;
    assert!(advisory(Bobyqa, false, false, "SAEM").is_none());
    assert!(advisory(Bobyqa, true, false, "SAEM").is_none());
    assert!(advisory(Lbfgs, false, true, "SAEM").is_none());
    let sa = advisory(Lbfgs, true, true, "SAEM").expect("score_sa");
    assert!(
        sa.contains("mstep_solver = score_sa") && !sa.contains("forward difference"),
        "{sa}"
    );
    let fd = advisory(Lbfgs, false, false, "IMPMAP").expect("out of scope");
    assert!(fd.starts_with("IMPMAP:"), "{fd}");
    assert!(
        fd.contains("forward difference") && !fd.contains("score_sa"),
        "{fd}"
    );
    // The production wrapper reads the adjoint's scope: an analytic two-compartment model is in it.
    let model = no_eta_theta_model();
    let prod = |sa| resolve_mstep_optimizer(&model, Lbfgs, sa, false, true, false, "SAEM");
    assert_eq!(prod(false), (Lbfgs, None));
    assert!(prod(true).1.is_some());
}

/// Converged Fisher scoring = converged BOBYQA, single draw: same objective, same optimum
/// (the information is a curvature, not a different problem).
#[test]
fn newton_mstep_reaches_the_bobyqa_maximiser() {
    use crate::types::MstepOptimizer::{Bobyqa, Newton};
    let model = no_eta_theta_model();
    let population = no_eta_population(&model, &[1.3, 12.0, 2.6, 26.0]);
    let p = packed_start();
    let draws = eta_draws();
    let reference = maximiser_with(&model, &population, &p, &draws[0], &[], Bobyqa, 900, false);
    let f_ref = conditional_objective(&model, &population, &draws[0], &[], &reference);
    let got = maximiser_with(&model, &population, &p, &draws[0], &[], Newton, 400, false);
    let f = conditional_objective(&model, &population, &draws[0], &[], &got);
    let gap = worst_gap(&got, &reference);
    eprintln!("newton: worst |dx| = {gap:.3e}, dObj = {:.3e}", f - f_ref);
    assert!(f - f_ref < 1e-5, "newton objective {f} vs bobyqa {f_ref}");
    assert!(gap < 5e-3, "newton {got:?} vs bobyqa {reference:?}");
    // Non-vacuity: the solve moved off the start (a no-op would also "not be worse").
    assert!(worst_gap(&got, &(p.log_theta.clone(), p.log_sigma.clone())) > 1e-3);
}

/// The K-draw mean: the averaged value, score and information must land on BOBYQA's maximiser
/// of the mean, away from the single-draw one.
#[test]
fn newton_mstep_reaches_the_bobyqa_maximiser_of_the_k_draw_mean() {
    use crate::types::MstepOptimizer::{Bobyqa, Newton};
    let model = no_eta_theta_model();
    let population = no_eta_population(&model, &[1.3, 12.0, 2.6, 26.0]);
    let p = packed_start();
    let draws = eta_draws();
    let extra = draws[1..4].to_vec();
    let reference = maximiser_with(
        &model,
        &population,
        &p,
        &draws[0],
        &extra,
        Bobyqa,
        900,
        false,
    );
    let got = maximiser_with(
        &model,
        &population,
        &p,
        &draws[0],
        &extra,
        Newton,
        400,
        false,
    );
    let single = maximiser_with(&model, &population, &p, &draws[0], &[], Bobyqa, 900, false);
    let gap = worst_gap(&got, &reference);
    let apart = worst_gap(&single, &reference);
    eprintln!("newton K-draw: worst |dx| = {gap:.3e}; single-draw maximiser is {apart:.3e} away");
    assert!(
        apart > 10.0 * gap.max(1e-4),
        "fixture: the extra draws must move it"
    );
    assert!(gap < 5e-3, "newton {got:?} vs bobyqa {reference:?}");
}

/// A pinned coordinate stays put and the free ones reach the restricted optimum.
#[test]
fn newton_mstep_holds_a_pinned_coordinate() {
    use crate::types::MstepOptimizer::{Bobyqa, Newton};
    let model = no_eta_theta_model();
    let population = no_eta_population(&model, &[1.3, 12.0, 2.6, 26.0]);
    let mut p = packed_start();
    let pin = p.log_theta[1];
    p.theta_lower[1] = pin;
    p.theta_upper[1] = pin;
    let draws = eta_draws();
    let reference = maximiser_with(&model, &population, &p, &draws[0], &[], Bobyqa, 900, false);
    let got = maximiser_with(&model, &population, &p, &draws[0], &[], Newton, 400, false);
    assert_eq!(got.0[1], pin, "the pinned theta moved");
    let gap = worst_gap(&got, &reference);
    eprintln!("newton pinned: worst |dx| = {gap:.3e}");
    assert!(gap < 5e-3, "newton {got:?} vs bobyqa {reference:?}");
}

/// At the production budget the Fisher-scoring solve captures the converged gain from the
/// same 0.05-log start the BOBYQA and L-BFGS budget tests use, and does it in fewer fused
/// evaluations than BOBYQA's budget (the count is the point of the option).
#[test]
fn newton_mstep_at_the_production_budget_captures_the_converged_gain() {
    use crate::types::MstepOptimizer::{Bobyqa, Newton};
    let model = no_eta_theta_model();
    let population = no_eta_population(&model, &[1.3, 12.0, 2.6, 26.0]);
    let draws = eta_draws();
    let mut p = packed_start();
    let reference = maximiser_with(&model, &population, &p, &draws[0], &[], Bobyqa, 900, false);
    for (v, r) in p.log_theta.iter_mut().zip(reference.0.iter()) {
        *v = r + 0.05;
    }
    p.log_sigma[0] = reference.1[0] + 0.05;
    let f_ref = conditional_objective(&model, &population, &draws[0], &[], &reference);
    let f_start = conditional_objective(
        &model,
        &population,
        &draws[0],
        &[],
        &(p.log_theta.clone(), p.log_sigma.clone()),
    );
    assert!(
        f_start > f_ref + 0.5,
        "fixture: the start must be off the optimum"
    );
    for maxiter in [1u32, 3] {
        let got = maximiser_with(
            &model,
            &population,
            &p,
            &draws[0],
            &[],
            Newton,
            maxiter,
            false,
        );
        let f = conditional_objective(&model, &population, &draws[0], &[], &got);
        let captured = (f_start - f) / (f_start - f_ref);
        eprintln!("newton maxiter {maxiter}: captured {captured:.4} of the converged gain");
        assert!(captured > 0.9, "maxiter {maxiter}: captured {captured}");
    }
}

/// A model the adjoint's recorders decline (a Form C readout without its compiled program),
/// otherwise a plain Gaussian one-compartment ODE.
fn out_of_adjoint_model() -> CompiledModel {
    let mut m = parse_model_string(
        r"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
",
    )
    .expect("ODE fixture parses");
    assert!(
        crate::sens::ode_provider::adjoint::recorder_in_scope(&m),
        "premise: the adjoint serves it with its program"
    );
    m.ode_spec.as_mut().unwrap().readout_program = None;
    m
}

/// The `newton` scope is ONE decision, [`newton_scope_gap`], read by the advisory and the
/// solve alike: `newton` runs on a model with the closed-form information and runs `bobyqa`
/// — named, with the property that put it there — on every one without. Each reason's
/// assertion carries its own discriminator and the others' negatives, so a message that names
/// two cannot pass; the effective optimizer is `bobyqa` in each, never silently `newton`.
#[test]
fn the_newton_scope_is_one_decision_with_a_named_reason() {
    use crate::types::BloqMethod;
    use crate::types::MstepOptimizer::{Bobyqa, Newton};
    let reasons = [
        "mixture",
        "M3 censoring",
        "block_sigma",
        "magnitude term",
        "outside the adjoint's scope",
    ];
    let plain = no_eta_theta_model();
    assert_eq!(newton_scope_gap(&plain, false), None);
    let mut m3 = no_eta_theta_model();
    m3.bloq_method = BloqMethod::M3;
    let ode_out = out_of_adjoint_model();
    for (label, model, mixture, want) in [
        ("mixture", &plain, true, 0usize),
        ("M3", &m3, false, 1),
        ("out of the adjoint", &ode_out, false, 4),
    ] {
        let gap = newton_scope_gap(model, mixture).unwrap_or_else(|| panic!("{label}: no gap"));
        assert!(gap.contains(reasons[want]), "{label}: {gap}");
        let (opt, warning) =
            resolve_mstep_optimizer(model, Newton, false, mixture, true, false, "IMP");
        let w = warning.unwrap_or_else(|| panic!("{label}: silent"));
        assert_eq!(opt, Bobyqa, "{label}");
        assert!(
            w.starts_with("IMP:") && w.contains("runs `bobyqa`"),
            "{label}: {w}"
        );
        for (k, other) in reasons.iter().enumerate() {
            assert_eq!(w.contains(other), k == want, "{label}: {other} in {w}");
        }
        assert!(!w.contains("score_sa"), "{label}: {w}");
    }
    // The model-property reasons of the Gaussian-scope list are the same list `score_sa` reads.
    for (model, k) in [(correlated_residual_model(), 2), (magnitude_model(), 3)] {
        let gap = newton_scope_gap(&model, false).expect("a gap");
        assert!(gap.contains(reasons[k]), "{gap}");
        assert_eq!(Some(gap), numerical_mstep_scope_gap(&model, 0, false));
    }
    // Inside it — a closed form, IOV (the adjoint records the stacked `[η | κ]`), FREM — newton
    // is the optimizer, with no warning.
    assert_eq!(
        resolve_mstep_optimizer(&plain, Newton, false, false, true, false, "SAEM"),
        (Newton, None)
    );
    assert_eq!(newton_scope_gap(&iov_model(), false), None);
    assert_eq!(newton_scope_gap(&frem_model(), false), None);
    // `score_sa` replaces the numerical solve: the option has nothing to act on, and says that
    // (not a scope reason).
    let (opt, w) = resolve_mstep_optimizer(&plain, Newton, true, false, true, false, "SAEM");
    let w = w.expect("score_sa");
    assert!(
        opt == Bobyqa
            && w.contains("mstep_solver = score_sa")
            && !w.contains("adjoint")
            && !w.contains("runs `bobyqa`"),
        "{w}"
    );
    // Bobyqa and lbfgs are not Newton's to change.
    assert_eq!(
        resolve_mstep_optimizer(&m3, Bobyqa, false, true, true, false, "SAEM"),
        (Bobyqa, None)
    );
    assert_eq!(
        resolve_mstep_optimizer(
            &plain,
            crate::types::MstepOptimizer::Lbfgs,
            false,
            true,
            true,
            false,
            "SAEM"
        )
        .0,
        crate::types::MstepOptimizer::Lbfgs
    );
}

/// `auto` (the default, #3001) is the scope rule itself: `newton` exactly where
/// [`newton_scope_gap`] is `None`, `bobyqa` everywhere else, and silent in both — an `auto` is
/// not a request the model fails, unlike an explicit `newton` (which warns, above). Each
/// out-of-scope input is its own row, so a rule keyed on one reason only fails the others; the
/// explicit-`newton` warning is asserted still present beside it.
#[test]
fn auto_is_newton_inside_the_scope_and_silent_bobyqa_outside() {
    use crate::types::BloqMethod;
    use crate::types::MstepOptimizer::{Auto, Bobyqa, Newton};
    assert_eq!(crate::types::FitOptions::default().mstep_optimizer, Auto);
    let plain = no_eta_theta_model();
    let mut m3 = no_eta_theta_model();
    m3.bloq_method = BloqMethod::M3;
    let ode_out = out_of_adjoint_model();
    for (label, model, sa, mixture, want) in [
        ("plain", &plain, false, false, Newton),
        ("iov", &iov_model(), false, false, Newton),
        ("frem", &frem_model(), false, false, Newton),
        ("mixture", &plain, false, true, Bobyqa),
        ("M3", &m3, false, false, Bobyqa),
        ("out of the adjoint", &ode_out, false, false, Bobyqa),
        (
            "correlated residual",
            &correlated_residual_model(),
            false,
            false,
            Bobyqa,
        ),
        ("score_sa", &plain, true, false, Bobyqa),
    ] {
        assert_eq!(
            resolve_mstep_optimizer(model, Auto, sa, mixture, true, false, "SAEM"),
            (want, None),
            "{label}"
        );
        assert_eq!(
            want == Newton,
            newton_scope_gap(model, mixture).is_none() && !sa,
            "{label}: auto is the scope rule"
        );
    }
    // The explicit request out of scope still says why.
    assert!(
        resolve_mstep_optimizer(&m3, Newton, false, false, true, false, "SAEM")
            .1
            .is_some()
    );
    assert_eq!(Auto.as_str(), "auto");
}

/// A Newton solve whose fused pass loses its information after one accepted step (a stall) is
/// not a solved one: the route is `bobyqa`, and the solve reaches BOBYQA's maximiser — the
/// start-point stall (`0` passes: declined at the start) takes the same route, so the two
/// differ only by where BOBYQA starts.
#[test]
fn a_stalled_newton_mstep_continues_with_bobyqa() {
    use crate::estimation::mstep_newton::FUSED_PASSES_ALLOWED;
    use crate::types::MstepOptimizer::{Bobyqa, Newton};
    let model = no_eta_theta_model();
    let population = no_eta_population(&model, &[1.3, 12.0, 2.6, 26.0]);
    let p = packed_start();
    let draws = eta_draws();
    let reference = maximiser_with(&model, &population, &p, &draws[0], &[], Bobyqa, 900, false);
    let route = |passes: Option<u32>| {
        MSTEP_ROUTE.with(|c| c.set([0; 3]));
        FUSED_PASSES_ALLOWED.set(passes);
        let got = maximiser_with(&model, &population, &p, &draws[0], &[], Newton, 400, false);
        FUSED_PASSES_ALLOWED.set(None);
        (MSTEP_ROUTE.with(|c| c.get()), got)
    };
    // Unlimited: Newton solved it.
    assert_eq!(route(None).0, [0, 0, 1]);
    for passes in [0u32, 1] {
        let (used, got) = route(Some(passes));
        assert_eq!(used, [1, 0, 0], "{passes} fused pass(es): BOBYQA ran");
        let gap = worst_gap(&got, &reference);
        assert!(
            gap < 5e-3,
            "{passes} pass(es): {gap:.3e} from the maximiser"
        );
    }
}

/// One budget, and the continuation is a continuation. A Newton solve that stalls after one
/// accepted step hands BOBYQA its point (`x.copy_from_slice(&xn)`) *and* the evaluations it
/// did not spend, so the whole M-step costs [`mstep_budget`] and no more. Newton spends three
/// passes (fused, value, the fused pass that declines) and BOBYQA the rest: the evaluation
/// count stays inside the budget and — BOBYQA being from Newton's better point — below what
/// BOBYQA alone needs from the start; the continuation is no further from the maximiser.
#[test]
fn a_stalled_newton_mstep_continues_from_its_point_within_one_budget() {
    use crate::estimation::mstep_newton::FUSED_PASSES_ALLOWED;
    use crate::estimation::saem::MSTEP_EVALS;
    use crate::types::MstepOptimizer::{Bobyqa, Newton};
    let model = no_eta_theta_model();
    let population = no_eta_population(&model, &[1.3, 12.0, 2.6, 26.0]);
    let p = packed_start();
    let draws = eta_draws();
    let reference = maximiser_with(&model, &population, &p, &draws[0], &[], Bobyqa, 900, false);
    let start = (p.log_theta.clone(), p.log_sigma.clone());
    let budget = crate::estimation::saem::mstep_budget(1, 5, 5, Bobyqa, false);
    let run = |optimizer, passes: Option<u32>| {
        MSTEP_ROUTE.with(|c| c.set([0; 3]));
        MSTEP_EVALS.with(|c| c.set(0));
        FUSED_PASSES_ALLOWED.set(passes);
        let got = maximiser_with(&model, &population, &p, &draws[0], &[], optimizer, 1, false);
        FUSED_PASSES_ALLOWED.set(None);
        (
            MSTEP_ROUTE.with(|c| c.get()),
            MSTEP_EVALS.with(|c| c.get()),
            got,
        )
    };
    let (route, evals_stalled, stalled) = run(Newton, Some(1));
    assert_eq!(route, [1, 0, 0], "BOBYQA continued");
    assert!(
        evals_stalled <= budget,
        "Newton and its continuation spent {evals_stalled} evaluations of a budget of {budget}"
    );
    let (route, evals_alone, alone) = run(Bobyqa, None);
    assert_eq!(route, [1, 0, 0]);
    assert!(evals_alone > 0 && evals_alone <= budget, "{evals_alone}");
    let f_of =
        |x: &(Vec<f64>, Vec<f64>)| conditional_objective(&model, &population, &draws[0], &[], x);
    let (f_start, f_ref) = (f_of(&start), f_of(&reference));
    let captured = |x: &(Vec<f64>, Vec<f64>)| (f_start - f_of(x)) / (f_start - f_ref);
    let (c_stalled, c_alone) = (captured(&stalled), captured(&alone));
    eprintln!(
        "stalled newton + bobyqa: {evals_stalled} evaluations, captured {c_stalled:.6}; bobyqa \
         alone: {evals_alone}, {c_alone:.6}; budget {budget}"
    );
    // Both converge now, so the continuation's start shows in what it cost: BOBYQA from the
    // point of Newton's accepted step needs fewer evaluations than from the start.
    assert!(
        evals_stalled < evals_alone,
        "the continuation ({evals_stalled}) must start from Newton's point, not the start \
         ({evals_alone})"
    );
    assert!(
        c_stalled > 0.999 && c_alone > 0.999,
        "{c_stalled} {c_alone}"
    );
}

/// The default M-step solve is run to its tolerance: `maxiter` does not cut it off. BOBYQA
/// spends `2n + 1` evaluations on its first design, so the old budget `maxiter·(n + 1)` left a
/// search of 7 evaluations to `maxiter = 3` on this fixture's five free coordinates — a
/// fraction of the gain, and on a larger model (nine θ and two σ: 293 of 304 M-steps ended on
/// the budget) a fit −130 in −2 log L from the mu-referenced one. Now the result neither
/// depends on `maxiter` nor falls short of the converged maximiser; the budget a pinned
/// coordinate does not enlarge, `lbfgs` and the mixture arm keep `maxiter·(n + 1)`.
#[test]
fn a_bobyqa_mstep_is_run_to_its_tolerance_not_cut_off_by_maxiter() {
    use crate::estimation::saem::{mstep_budget, MSTEP_EVALS_PER_POINT};
    use crate::types::MstepOptimizer::{Bobyqa, Lbfgs, Newton};
    let model = no_eta_theta_model();
    let population = no_eta_population(&model, &[1.3, 12.0, 2.6, 26.0]);
    let p = packed_start();
    let draws = eta_draws();
    let start = (p.log_theta.clone(), p.log_sigma.clone());
    let at = |maxiter| {
        maximiser_with(
            &model,
            &population,
            &p,
            &draws[0],
            &[],
            Bobyqa,
            maxiter,
            false,
        )
    };
    let (cut_off, long) = (at(3), at(900));
    let f_of =
        |x: &(Vec<f64>, Vec<f64>)| conditional_objective(&model, &population, &draws[0], &[], x);
    let (f_start, f_ref) = (f_of(&start), f_of(&long));
    assert!(
        f_start - f_ref > 1.0,
        "a non-degenerate fixture: {}",
        f_start - f_ref
    );
    let captured = (f_start - f_of(&cut_off)) / (f_start - f_ref);
    eprintln!("maxiter 3 captures {captured:.6} of the converged gain");
    assert!(
        captured > 0.9999,
        "maxiter 3 reaches {captured} of the converged gain: the budget is cutting the solve off"
    );
    assert_eq!(cut_off, long, "the budget does not depend on maxiter");

    // The budget itself, by arm.
    let per = MSTEP_EVALS_PER_POINT;
    assert_eq!(mstep_budget(3, 11, 11, Bobyqa, false), per * 23);
    assert_eq!(
        mstep_budget(5, 11, 11, Bobyqa, false),
        per * 23,
        "not by maxiter"
    );
    assert_eq!(
        mstep_budget(3, 11, 11, Newton, false),
        per * 23,
        "its continuation's too"
    );
    assert_eq!(
        mstep_budget(3, 11, 4, Bobyqa, false),
        per * 9,
        "pinned coordinates cost nothing"
    );
    assert_eq!(
        mstep_budget(3, 11, 11, Lbfgs, false),
        36,
        "a gradient method's accounting"
    );
    assert_eq!(
        mstep_budget(3, 11, 11, Bobyqa, true),
        36,
        "the mixture arm keeps its budget"
    );
    // It is a guard, not a target: more than a converged solve needed on the largest model
    // measured (650 evaluations at n = 11).
    assert!(mstep_budget(3, 11, 11, Bobyqa, false) > 650);
}

/// What a fit's Newton solves did reaches its warnings: a solve that loses the information (the
/// hook spends the fused passes at the start) is counted and named, with the other cause's
/// negative; solves that converge leave no warning.
#[test]
fn a_newton_solve_that_loses_its_information_is_reported_by_the_fit() {
    use crate::estimation::mstep_newton::{NewtonTally, FUSED_PASSES_ALLOWED};
    use crate::types::MstepOptimizer::Newton;
    let model = no_eta_theta_model();
    let population = no_eta_population(&model, &[1.3, 12.0, 2.6, 26.0]);
    let p = packed_start();
    let draws = eta_draws();
    let tally = NewtonTally::default();
    let _ = maximiser_tallied(
        &model,
        &population,
        &p,
        &draws[0],
        &[],
        Newton,
        10,
        false,
        Some(&tally),
    );
    assert!(tally.warning("SAEM").is_none(), "converged: nothing to say");
    FUSED_PASSES_ALLOWED.set(Some(0));
    let _ = maximiser_tallied(
        &model,
        &population,
        &p,
        &draws[0],
        &[],
        Newton,
        10,
        false,
        Some(&tally),
    );
    FUSED_PASSES_ALLOWED.set(None);
    let w = tally
        .warning("SAEM")
        .expect("one solve lost its information");
    assert!(
        w.contains("handed 1 of 2 numerical M-steps")
            && w.contains("lost the expected information"),
        "{w}"
    );
    assert!(!w.contains("no admissible"), "{w}");
    // The fit reports it: every Newton solve of a SAEM fit declines under the hook.
    let mut opts = crate::types::FitOptions::default();
    opts.method = crate::types::EstimationMethod::Saem;
    opts.saem_n_exploration = 3;
    opts.saem_n_convergence = 2;
    opts.saem_seed = Some(7);
    opts.run_covariance_step = false;
    opts.mstep_optimizer = Newton;
    let fit = |hook: Option<u32>| {
        FUSED_PASSES_ALLOWED.set(hook);
        let res = run_saem(&model, &population, &model.default_params, &opts);
        FUSED_PASSES_ALLOWED.set(None);
        res.expect("SAEM fit").warnings
    };
    let declined = fit(Some(0));
    assert!(
        declined
            .iter()
            .any(|w| w.starts_with("SAEM:") && w.contains("lost the expected information")),
        "{declined:?}"
    );
    let clean = fit(None);
    assert!(
        !clean
            .iter()
            .any(|w| w.contains("lost the expected information")),
        "{clean:?}"
    );
}

/// An ODE one-compartment fixture the adjoint declines at run time (no compiled readout
/// program), so the θ score of a fit on it is a forward difference. `v_expr` makes V either
/// mu-referenced (`TVV * exp(ETA_V)`: every θ is the closed-form step's, a σ-only numerical
/// M-step) or a plain θ the numerical M-step must estimate (`TVV`).
fn declined_ode_model(v_expr: &str) -> CompiledModel {
    let src = format!(
        "[parameters]\n  theta TVCL(1.0, 0.01, 100.0)\n  theta TVV(10.0, 0.1, 1000.0)\n  \
         omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  sigma EPS ~ 0.04\n\
         [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = {v_expr}\n\
         [structural_model]\n  ode(states=[central])\n[odes]\n  \
         d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n\
         [error_model]\n  DV ~ proportional(EPS)\n"
    );
    let mut model = parse_model_string(&src).expect("ode fixture parses");
    model.ode_spec.as_mut().unwrap().readout_program = None;
    assert!(
        !crate::sens::ode_provider::adjoint::recorder_in_scope(&model),
        "premise: the adjoint declines this model"
    );
    model
}

fn declined_ode_population(n: usize) -> Population {
    use crate::types::{DoseEvent, Subject};
    let times = [0.5f64, 1.5, 4.0, 9.0, 20.0];
    let mut st = 0x2931_u64;
    let subjects: Vec<Subject> = (0..n)
        .map(|i| {
            let cl = 1.0 * (0.3 * lcg(&mut st)).exp();
            let v = 10.0 * (0.2 * lcg(&mut st)).exp();
            let obs: Vec<f64> = times
                .iter()
                .map(|t| 100.0 / v * (-(cl / v) * t).exp() * (1.0 + 0.04 * lcg(&mut st)))
                .collect();
            Subject {
                id: format!("{}", i + 1),
                doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                dose_occasions: vec![1],
                occasions: vec![1; times.len()],
                obs_times: times.to_vec(),
                observations: obs,
                obs_cmts: vec![1; times.len()],
                cens: vec![0; times.len()],
                ..Default::default()
            }
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

/// The `lbfgs` forward-difference advisory is for a solve that has a θ to differentiate: a
/// model the adjoint declines, fitted with every θ owned by the closed-form step (a σ-only
/// numerical M-step, whose σ score is exact), has nothing to say; the same model with a θ the
/// closed form does not own is the case the advisory is for. Each side of the gate, on SAEM
/// and on IMP (whose gate reads its own closed-form set), with the other's negative.
#[test]
fn the_lbfgs_advisory_needs_a_theta_for_the_solve_to_differentiate() {
    use crate::types::EstimationMethod::{Imp, Saem};
    use crate::types::MstepOptimizer::Lbfgs;
    let pop = declined_ode_population(8);
    let advisory = |method, v_expr: &str| -> bool {
        let model = declined_ode_model(v_expr);
        let mut opts = crate::types::FitOptions::default();
        opts.method = method;
        opts.saem_n_exploration = 2;
        opts.saem_n_convergence = 1;
        opts.saem_seed = Some(3);
        opts.imp_iterations = 2;
        opts.imp_samples = 20;
        opts.imp_auto = false;
        opts.imp_seed = Some(3);
        opts.run_covariance_step = false;
        opts.mstep_optimizer = Lbfgs;
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts)
            .unwrap_or_else(|e| panic!("{method:?} {v_expr}: {e}"));
        res.warnings
            .iter()
            .any(|w| w.contains("running without an exact θ gradient"))
    };
    for method in [Saem, Imp] {
        assert!(
            !advisory(method, "TVV * exp(ETA_V)"),
            "{method:?}: a σ-only solve has no θ score to be a forward difference"
        );
        assert!(
            advisory(method, "TVV"),
            "{method:?}: a θ the numerical M-step estimates is differenced"
        );
    }
}

/// The unit cases of the same gate: `theta_free` is the only thing that silences the advisory
/// for a model out of the adjoint's scope, and an in-scope model is silent either way.
#[test]
fn the_lbfgs_advisory_gate_and_the_free_theta_predicate() {
    use crate::types::MstepOptimizer::Lbfgs;
    let advisory =
        |exact, free| resolve_mstep_optimizer_for(Lbfgs, false, exact, None, free, false, "IMP").1;
    assert!(advisory(false, true).is_some());
    assert!(
        advisory(false, false).is_none(),
        "no θ free: nothing differenced"
    );
    assert!(advisory(true, true).is_none());
    assert!(advisory(true, false).is_none());
    // Free: not FIXed and not owned by a closed-form step.
    assert!(mstep_has_free_theta(3, &[false; 3], &[0, 1]));
    assert!(!mstep_has_free_theta(3, &[false; 3], &[0, 1, 2]));
    assert!(!mstep_has_free_theta(3, &[false, false, true], &[0, 1]));
    assert!(mstep_has_free_theta(3, &[false, false, true], &[0]));
    assert!(!mstep_has_free_theta(0, &[], &[]));
}

/// The value path with a prediction memo is the memo-free `obs_nll_sum` bit for bit, serves an
/// unchanged θ from the memo (a stale draw proves it), gives each draw its own sample, and a
/// tracking memo replaces the predictions when θ moves while a pinned-θ one declines the move.
#[test]
fn the_memoised_value_path_is_obs_nll_sum_bit_for_bit() {
    use crate::estimation::pred_memo::PredMemo;
    let model = no_eta_theta_model();
    let population = no_eta_population(&model, &[1.3, 12.0, 2.6, 26.0]);
    let draws = eta_draws();
    let n = population.subjects.len();
    let th: Vec<f64> = packed_start().log_theta.iter().map(|x| x.exp()).collect();
    let th2: Vec<f64> = th.iter().map(|x| x * 1.01).collect();
    // Two samples (draws) per subject; θ free under Newton, so the memo follows θ.
    let memo = PredMemo::for_solve(vec![2usize; n], true, true).unwrap();
    let ev = memo.begin(&th).unwrap();
    for sg in [[0.1f64], [0.2], [0.35]] {
        let plain = obs_nll_sum(&model, &population, &th, &sg, &draws[0], &[]);
        let with = obs_nll_sum_memo(&model, &population, &th, &sg, &draws[0], &[], Some((ev, 0)));
        assert_eq!(plain, with, "σ = {sg:?}");
    }
    // Stale: sample 0 holds draw 0's predictions for `th`; draw 1 through it scores draw 0.
    let sg = [0.2f64];
    let d0 = obs_nll_sum(&model, &population, &th, &sg, &draws[0], &[]);
    let d1 = obs_nll_sum(&model, &population, &th, &sg, &draws[1], &[]);
    assert!(d0 != d1, "fixture: the draws must differ");
    let stale = obs_nll_sum_memo(&model, &population, &th, &sg, &draws[1], &[], Some((ev, 0)));
    assert_eq!(stale, d0, "served from the memo");
    // Its own sample (1) computes draw 1 itself.
    let own = obs_nll_sum_memo(&model, &population, &th, &sg, &draws[1], &[], Some((ev, 1)));
    assert_eq!(own, d1);
    // A moved θ opens a new evaluation of a tracking memo: recomputed through sample 0.
    let ev2 = memo.begin(&th2).unwrap();
    let moved = obs_nll_sum_memo(
        &model,
        &population,
        &th2,
        &sg,
        &draws[1],
        &[],
        Some((ev2, 0)),
    );
    assert_eq!(
        moved,
        obs_nll_sum(&model, &population, &th2, &sg, &draws[1], &[])
    );
    // A pinned-θ memo holds the first θ: a different one is declined, not served from.
    let fixed = PredMemo::for_solve(vec![2usize; n], false, false).unwrap();
    assert!(fixed.begin(&th).is_some() && fixed.begin(&th2).is_none());
}

/// The optimizer's `scale_params` packing returns a pinned coordinate bit for bit: the scale
/// is taken from the coordinate itself (`|v|`, or 1 below 0.1), so `(v / scale) · scale` is `v`
/// exactly — `±1 · |v|`, or `v / 1 · 1`. That is what lets a σ-only (pinned-θ) memo, whose θ key
/// is compared bit for bit, keep serving a BOBYQA continuation that evaluates at the scaled
/// packing's `xs · scale` after a Newton solve evaluated at `x`. If the scale rule changes to
/// anything that is not the coordinate's own magnitude this fails, and the memo needs a re-key
/// at the hand-over.
#[test]
fn the_scaled_packing_returns_a_pinned_coordinate_bit_for_bit() {
    let mut st = 0x1_0001_u64;
    let mut values: Vec<f64> = vec![
        0.0,
        0.1,
        -0.1,
        0.1 + 1e-17,
        1.0,
        -1.0,
        2.0f64.ln(),
        400.0f64.ln(),
    ];
    values.extend((0..2000).map(|_| 12.0 * lcg(&mut st)));
    values.extend((0..2000).map(|_| 0.2 * lcg(&mut st)));
    for v in values {
        let scale = crate::estimation::parameterization::compute_scale(&[v])[0];
        assert_eq!(((v / scale) * scale).to_bits(), v.to_bits(), "v = {v:e}");
    }
}

/// What the prediction memo does after a Newton solve stalls and BOBYQA takes over. A solve
/// with every θ pinned (σ-only) keeps serving — under `scale_params`, whose packing is
/// bit-exact for a pinned coordinate (above) — and declines none of the continuation's
/// evaluations; a solve with free θ has retired its tracking memo (BOBYQA evaluates each point
/// once, so every evaluation would only copy predictions in): the memo took part in Newton's
/// evaluations and in none of the continuation's.
#[test]
fn a_stalled_newton_mstep_keeps_a_pinned_theta_memo_and_retires_a_tracking_one() {
    use crate::estimation::mstep_newton::FUSED_PASSES_ALLOWED;
    use crate::estimation::pred_memo::LAST_SOLVE;
    use crate::estimation::saem::MSTEP_EVALS;
    use crate::types::MstepOptimizer::Newton;
    let model = no_eta_theta_model();
    let population = no_eta_population(&model, &[1.3, 12.0, 2.6, 26.0]);
    let draws = eta_draws();
    let run = |pinned: bool, scale_params: bool| {
        let mut p = packed_start();
        if pinned {
            p.theta_lower = p.log_theta.clone();
            p.theta_upper = p.log_theta.clone();
        }
        MSTEP_ROUTE.with(|c| c.set([0; 3]));
        MSTEP_EVALS.with(|c| c.set(0));
        FUSED_PASSES_ALLOWED.set(Some(1));
        let _ = maximiser_with(
            &model,
            &population,
            &p,
            &draws[0],
            &[],
            Newton,
            1,
            scale_params,
        );
        FUSED_PASSES_ALLOWED.set(None);
        assert_eq!(
            MSTEP_ROUTE.with(|c| c.get()),
            [1, 0, 0],
            "premise: Newton stalled"
        );
        (MSTEP_EVALS.with(|c| c.get()), LAST_SOLVE.with(|c| c.get()))
    };
    for scale_params in [false, true] {
        let (evals, (opened, declined)) = run(true, scale_params);
        assert!(
            evals > 8,
            "premise: BOBYQA ran after Newton ({evals} evaluations)"
        );
        assert_eq!(
            (opened, declined),
            (evals, 0),
            "σ-only, scale_params = {scale_params}: every evaluation of the continuation is served"
        );
    }
    for scale_params in [false, true] {
        let (evals, (opened, declined)) = run(false, scale_params);
        assert!(
            evals > 8,
            "premise: BOBYQA ran after Newton ({evals} evaluations)"
        );
        assert!(
            opened <= 3 && declined == evals - opened,
            "free θ, scale_params = {scale_params}: the memo took part in {opened} of {evals} \
             evaluations (Newton's: a fused pass, a trial's value) and the continuation's \
             {declined} found it retired"
        );
    }
}

/// `auto` keeps BOBYQA for a SAEM solve with no free θ and Newton for one with — where Newton
/// is in scope at all — and keeps Newton for IMP/IMPMAP either way (`sigma_only_bobyqa`): the
/// rule is structural (free θ × who owns the memo), never a size. An explicit `newton` is the
/// user's, σ-only included, in both. And the fits report it: a SAEM fit with every θ
/// mu-referenced runs `bobyqa`, an IMP fit of the same model `newton`.
#[test]
fn auto_is_bobyqa_for_a_sigma_only_saem_solve_and_newton_for_imp() {
    use crate::types::MstepOptimizer::{Auto, Bobyqa, Newton};
    let plain = no_eta_theta_model();
    for (theta_free, sigma_only_bobyqa, want) in [
        (true, true, Newton),
        (true, false, Newton),
        (false, true, Bobyqa),
        (false, false, Newton),
    ] {
        assert_eq!(
            resolve_mstep_optimizer(
                &plain,
                Auto,
                false,
                false,
                theta_free,
                sigma_only_bobyqa,
                "X"
            ),
            (want, None),
            "θ free {theta_free}, σ-only bobyqa {sigma_only_bobyqa}"
        );
        assert_eq!(
            resolve_mstep_optimizer(
                &plain,
                Newton,
                false,
                false,
                theta_free,
                sigma_only_bobyqa,
                "X"
            ),
            (Newton, None),
            "an explicit newton is the request"
        );
    }
    use crate::types::EstimationMethod::{Imp, Saem};
    let pop = declined_ode_population(8);
    let model = |error: &str, extra_sigma: &str| {
        let src = format!(
            "[parameters]\n  theta TVCL(1.0, 0.01, 100.0)\n  theta TVV(10.0, 0.1, 1000.0)\n  \
             omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  sigma EPS ~ 0.04\n{extra_sigma}\
             [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n\
             [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n\
             [error_model]\n  {error}\n"
        );
        parse_model_string(&src).expect("parses")
    };
    let used = |model: &crate::types::CompiledModel, method, optimizer| {
        let mut opts = crate::types::FitOptions::default();
        opts.method = method;
        opts.mstep_optimizer = optimizer;
        opts.saem_n_exploration = 2;
        opts.saem_n_convergence = 1;
        opts.saem_seed = Some(3);
        opts.imp_iterations = 2;
        opts.imp_samples = 20;
        opts.imp_auto = false;
        opts.imp_seed = Some(3);
        opts.run_covariance_step = false;
        crate::api::fit(model, &pop, &model.default_params, &opts)
            .expect("fits")
            .mstep_optimizer_used
    };
    // One σ, additive or proportional, every θ mu-referenced: SAEM updates σ in closed form
    // and runs no numerical solve, so the fit reports none — whatever was asked (an explicit
    // `newton` included), not the optimizer the unused option resolved to.
    let closed = model("DV ~ proportional(EPS)", "");
    for optimizer in [Auto, Bobyqa, Newton] {
        assert_eq!(
            used(&closed, Saem, optimizer),
            None,
            "closed-form σ, {optimizer:?}: no numerical solve ran"
        );
    }
    // IMP/IMPMAP have no closed-form σ: the same model runs the numerical solve.
    assert_eq!(used(&closed, Imp, Auto).as_deref(), Some("newton"));
    // Two σ (combined): SAEM's σ-only numerical solve *does* run — `auto` keeps BOBYQA for it
    // (its evaluations are memo-served), an explicit `newton` is run as asked.
    let numerical = model("DV ~ combined(EPS, ADD)", "  sigma ADD ~ 0.1\n");
    assert_eq!(used(&numerical, Saem, Auto).as_deref(), Some("bobyqa"));
    assert_eq!(used(&numerical, Saem, Newton).as_deref(), Some("newton"));
    assert_eq!(used(&numerical, Imp, Auto).as_deref(), Some("newton"));
}

/// The BOBYQA solve stops on NLopt's absolute floor too ([`MSTEP_FTOL_ABS`], the same number as
/// the relative tolerance): at a zero-residual optimum with a pinned unit σ the objective is
/// `Σ r²/2 → 0`, and a purely relative tolerance (`1e-7 · mean|f|`) shrinks with it, so the
/// solve would run on until its trust radius collapses. With the floor it ends at the first
/// iteration that gains less than `1e-7`: measured 28 evaluations (f = 2e-14) against 194
/// without it (budget 500); the bound of 60 sits between, 2× over the floored count.
#[test]
fn a_bobyqa_mstep_with_a_zero_residual_optimum_stops_on_the_absolute_floor() {
    use crate::estimation::saem::{mstep_budget, MSTEP_EVALS};
    use crate::types::MstepOptimizer::Bobyqa;
    let model = parse_model_string(
        r"
[parameters]
  theta TVCL(1.0, 0.05, 20.0)
  theta TVV(10.0, 1.0, 200.0)
  omega ETA_CL ~ 0.09
  sigma ADD_ERR ~ 1.0 (sd) FIX
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ additive(ADD_ERR)
",
    )
    .expect("fixture parses");
    let population = no_eta_population(&model, &[1.3, 12.0]);
    let etas: Vec<Vec<f64>> = etas_true().iter().map(|&e| vec![e]).collect();
    let p = Packed {
        log_theta: vec![1.0f64.ln(), 10.0f64.ln()],
        log_sigma: vec![0.0],
        theta_lower: vec![0.05f64.ln(), 1.0f64.ln()],
        theta_upper: vec![20.0f64.ln(), 200.0f64.ln()],
        sigma_lower: vec![0.0],
        sigma_upper: vec![0.0],
        mask: vec![true; 2],
    };
    MSTEP_EVALS.with(|c| c.set(0));
    let got = theta_sigma_mstep_light(
        &model,
        &population,
        &etas,
        None,
        &p.log_theta,
        &p.log_sigma,
        &p.theta_lower,
        &p.theta_upper,
        &p.sigma_lower,
        &p.sigma_upper,
        2,
        1,
        1,
        false,
        &p.mask,
        None,
        &[],
        &[],
        Bobyqa,
        None,
    );
    let evals = MSTEP_EVALS.with(|c| c.get());
    let budget = mstep_budget(1, 3, 2, Bobyqa, false);
    let f = conditional_objective(&model, &population, &etas, &[], &got);
    eprintln!("zero-residual BOBYQA: {evals} evaluations of {budget}, f = {f:.3e}");
    assert!(f < 1e-3, "converged to the zero-residual optimum: {f}");
    assert!(
        evals < 60,
        "{evals} evaluations: the absolute floor ends the solve, not the budget {budget}"
    );
}
