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
    let (rejected, out_of_scope) = sa.counters();
    assert_eq!(out_of_scope, 0, "the fixture is inside the Gaussian scope");
    // Nothing here should land on a non-finite objective; a non-zero count means
    // the trust region is letting the step leave the model's domain.
    assert_eq!(
        rejected,
        0,
        "{rejected} of {} steps never found a finite objective",
        12 * draws.len()
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
