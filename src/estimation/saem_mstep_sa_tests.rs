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
                1.0,
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
        1.0,
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
        1.0,
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
        1.0,
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
        1.0,
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
        1.0,
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
        1.0,
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
        gamma_mstep,
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
