//! The weighted θ/σ M-step's solver and the gate on what it did (#2718).

use super::*;
use crate::estimation::importance_sampling::SubjectDraws;
use crate::parser::model_parser::parse_model_string;
use std::io::Write as _;

/// 14 coordinates, the size of the reported freeze: 8 identity-packed θ on wide bounds,
/// none mu-referenced, and three strata of combined error.
const WIDE_MODEL: &str = r"
[parameters]
  theta LOG_CL(1.3, -10, 10)
  theta LOG_V1(3.1, -10, 10)
  theta LOG_Q(1.8, -10, 10)
  theta LOG_V2(3.8, -10, 10)
  theta LOG_KA(-0.1, -10, 10)
  theta B_CLWT(0.4, -5, 5)
  theta B_VWT(0.5, -5, 5)
  theta B_CLSEX(0.05, -5, 5)
  sigma P1 ~ 0.2 (sd)
  sigma A1 ~ 0.15 (sd)
  sigma P2 ~ 0.25 (sd)
  sigma A2 ~ 0.25 (sd)
  sigma P3 ~ 0.3 (sd)
  sigma A3 ~ 0.1 (sd)

[individual_parameters]
  CL = exp(LOG_CL + B_CLWT*LWT + B_CLSEX*SEX)
  V1 = exp(LOG_V1 + B_VWT*LWT)
  Q  = exp(LOG_Q)
  V2 = exp(LOG_V2)
  KA = exp(LOG_KA)

[structural_model]
  pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)

[error_model]
  if (ASSAY == 1) {
    DV ~ combined(P1, A1)
  } else if (ASSAY == 2) {
    DV ~ combined(P2, A2)
  } else {
    DV ~ combined(P3, A3)
  }
";

/// Oral two-compartment concentration at `t` after a single dose of 100, by the
/// triexponential form, so the data do not come from the code under test.
fn two_cpt_oral_conc(t: f64, cl: f64, v1: f64, q: f64, v2: f64, ka: f64) -> f64 {
    let (k, k12, k21) = (cl / v1, q / v1, q / v2);
    let s = k + k12 + k21;
    let b = (s - (s * s - 4.0 * k * k21).sqrt()) / 2.0;
    let a = k * k21 / b;
    let term = |r: f64, o: f64, p: f64| ka * (k21 - r) / ((o - r) * (p - r)) * (-r * t).exp();
    100.0 / v1 * (term(a, ka, b) + term(b, ka, a) + term(ka, a, b))
}

/// 30 subjects at `CL = 5, V1 = 30, Q = 8, V2 = 60, KA = 1.2` with weight and sex effects,
/// and a fixed residual pattern sized to each stratum, so every σ has a maximiser.
fn wide_pop() -> Population {
    let mut csv = String::from("ID,TIME,DV,AMT,EVID,CMT,LWT,SEX,ASSAY\n");
    for id in 1..=30_usize {
        let lwt = (id % 5) as f64 * 0.15 - 0.3;
        let sex = (id % 2) as f64;
        let assay = id % 3 + 1;
        let prop = [0.1, 0.15, 0.2][assay - 1];
        csv.push_str(&format!("{id},0,0,100,1,1,{lwt},{sex},{assay}\n"));
        for (j, t) in [0.5_f64, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0].iter().enumerate() {
            let cl = (5.0_f64.ln() + 0.75 * lwt - 0.25 * sex).exp();
            let v1 = (30.0_f64.ln() + 0.9 * lwt).exp();
            let c = two_cpt_oral_conc(*t, cl, v1, 8.0, 60.0, 1.2);
            let bump = if (id + j) % 2 == 0 { 1.0 + prop } else { 1.0 - prop };
            csv.push_str(&format!("{id},{t},{:.6},0,0,1,{lwt},{sex},{assay}\n", c * bump));
        }
    }
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(csv.as_bytes()).unwrap();
    crate::read_nonmem_csv(f.path(), Some(&["LWT", "SEX", "ASSAY"]), None).unwrap()
}

/// The inputs of one solve from `WIDE_MODEL`'s start, with one empty draw per subject.
struct WideSolve {
    model: CompiledModel,
    pop: Population,
    draws: Vec<SubjectDraws>,
    schedules: Vec<Option<EventSchedule>>,
    theta: Vec<f64>,
    log_sigma: Vec<f64>,
    theta_lower: Vec<f64>,
    theta_upper: Vec<f64>,
}

impl WideSolve {
    fn new() -> Self {
        let model = parse_model_string(WIDE_MODEL).unwrap();
        let pop = wide_pop();
        let draws = pop
            .subjects
            .iter()
            .map(|_| {
                let mut d = SubjectDraws::cancelled(0);
                d.etas = vec![Vec::new()];
                d.weights = vec![1.0];
                d
            })
            .collect();
        let params = &model.default_params;
        let theta = params.theta.clone();
        let log_sigma = params.sigma.values.iter().map(|s| s.ln()).collect();
        let schedules = vec![None; pop.subjects.len()];
        let (theta_lower, theta_upper) = (params.theta_lower.clone(), params.theta_upper.clone());
        WideSolve { model, pop, draws, schedules, theta, log_sigma, theta_lower, theta_upper }
    }

    fn solve(&self, maxiter: u32) -> WeightedMstep {
        self.solve_with(maxiter, crate::types::MstepOptimizer::Bobyqa)
    }

    fn solve_with(&self, maxiter: u32, optimizer: crate::types::MstepOptimizer) -> WeightedMstep {
        let n_sigma = self.log_sigma.len();
        theta_sigma_weighted_mstep(
            &self.model,
            &self.pop,
            &self.draws,
            &self.schedules,
            &self.theta,
            &self.log_sigma,
            &self.theta_lower,
            &self.theta_upper,
            &vec![-8.0; n_sigma],
            &vec![5.0; n_sigma],
            self.theta.len(),
            n_sigma,
            maxiter,
            &vec![false; self.theta.len()],
            optimizer,
            None,
        )
    }

    fn objective(&self, log_theta: &[f64], log_sigma: &[f64]) -> f64 {
        let sigma: Vec<f64> = log_sigma.iter().map(|s| s.exp()).collect();
        weighted_mstep_objective(
            &self.model,
            &self.pop,
            &self.draws,
            &self.schedules,
            log_theta,
            &sigma,
        )
    }

    /// Coordinates of `step` more than [`MSTEP_MOVED_TOL`] from the start.
    fn moved(&self, step: &WeightedMstep) -> usize {
        let start = self.theta.iter().chain(&self.log_sigma);
        let end = step.log_theta.iter().chain(&step.log_sigma);
        start.zip(end).filter(|(a, b)| (*a - *b).abs() > MSTEP_MOVED_TOL).count()
    }
}

/// #2718: one budgeted solve on 14 coordinates has to move them, and has to capture most
/// of what a converged solve gains.
#[test]
fn a_budgeted_solve_moves_a_many_coordinate_problem() {
    let w = WideSolve::new();
    let f_start = w.objective(&w.theta, &w.log_sigma);
    let reference = w.solve(400);
    let f_ref = w.objective(&reference.log_theta, &reference.log_sigma);
    assert!(f_start.is_finite() && f_ref.is_finite());
    assert!(f_ref < f_start - 100.0, "{f_ref} vs {f_start}");
    for (maxiter, floor) in [(4_u32, 0.5), (8, 0.8)] {
        let step = w.solve(maxiter);
        let f = w.objective(&step.log_theta, &step.log_sigma);
        assert!(f.is_finite());
        let captured = (f_start - f) / (f_start - f_ref);
        eprintln!(
            "maxiter {maxiter}: moved {} of 14, start {f_start:.2}, f {f:.2}, ref {f_ref:.2}, \
             captured {captured:.3}",
            w.moved(&step)
        );
        assert!(step.failure.is_none(), "{:?}", step.failure);
        assert_eq!(w.moved(&step), 14, "maxiter {maxiter}");
        assert!(captured > floor, "maxiter {maxiter}: captured {captured}");
    }
}

/// The weighted M-step's budget: BOBYQA runs to its tolerance under a runaway guard of 100
/// evaluations per interpolation point, and the **mixture** arm keeps the pre-#3001
/// `maxiter·(n + 1)` as SAEM's mixture arm does (the changelog says mixtures keep the old
/// budget). 14 coordinates, `maxiter = 4`: 2900 against 60. Mutation: the mixture call passing
/// `mixture = false` to `mstep_budget` fails the second assertion.
#[test]
fn the_mixture_weighted_mstep_keeps_its_old_budget() {
    let w = WideSolve::new();
    let n_sigma = w.log_sigma.len();
    let n = w.theta.len() + n_sigma;
    assert_eq!(n, 14, "premise: the fixture's coordinate count");
    w.solve(4);
    assert_eq!(LAST_WEIGHTED_MAXEVAL.get(), 100 * (2 * 14 + 1));
    theta_sigma_weighted_mstep_mixture(
        &w.model,
        &w.pop,
        std::slice::from_ref(&w.draws),
        &w.schedules,
        &vec![vec![1.0]; w.pop.subjects.len()],
        &w.theta,
        &w.log_sigma,
        &w.theta_lower,
        &w.theta_upper,
        &vec![-8.0; n_sigma],
        &vec![5.0; n_sigma],
        w.theta.len(),
        n_sigma,
        4,
        &vec![false; w.theta.len()],
        crate::types::MstepOptimizer::Bobyqa,
    );
    assert_eq!(LAST_WEIGHTED_MAXEVAL.get(), 4 * (14 + 1));
}

/// #2718: the mixture M-step runs the same solver, so one class with all the
/// responsibility reproduces the single-population solve exactly. The budgets differ
/// (the mixture's is `maxiter·(n + 1)`, [`the_mixture_weighted_mstep_keeps_its_old_budget`]), so
/// `maxiter` is large enough that neither binds: both stop on the tolerance.
#[test]
fn the_mixture_mstep_shares_the_solver() {
    let w = WideSolve::new();
    let n_sigma = w.log_sigma.len();
    let pmix = vec![vec![1.0]; w.pop.subjects.len()];
    let mixture = theta_sigma_weighted_mstep_mixture(
        &w.model,
        &w.pop,
        std::slice::from_ref(&w.draws),
        &w.schedules,
        &pmix,
        &w.theta,
        &w.log_sigma,
        &w.theta_lower,
        &w.theta_upper,
        &vec![-8.0; n_sigma],
        &vec![5.0; n_sigma],
        w.theta.len(),
        n_sigma,
        100,
        &vec![false; w.theta.len()],
        crate::types::MstepOptimizer::Bobyqa,
    );
    let single = w.solve(100);
    assert_eq!(mixture.log_theta, single.log_theta);
    assert_eq!(mixture.log_sigma, single.log_sigma);
    assert_eq!(w.moved(&mixture), 14);
}

/// #2718: a θ on an interval narrower than twice the nominal step must not make BOBYQA
/// reject the whole problem, which would hold every coordinate at its start.
#[test]
fn a_narrow_interval_does_not_cost_the_solve() {
    let mut w = WideSolve::new();
    let (lo, hi) = (0.0, 0.1);
    assert!(hi - lo < 2.0 * crate::estimation::saem::MSTEP_INITIAL_STEP);
    w.theta[7] = 0.05;
    w.theta_lower[7] = lo;
    w.theta_upper[7] = hi;
    let f_start = w.objective(&w.theta, &w.log_sigma);
    let step = w.solve(4);
    assert!(step.failure.is_none(), "{:?}", step.failure);
    assert!(w.objective(&step.log_theta, &step.log_sigma) < f_start - 10.0);
    assert!(w.moved(&step) >= 13, "moved {}", w.moved(&step));
    assert!((lo..=hi).contains(&step.log_theta[7]));
}

/// A solve that returned `after`, θ and σ together in `log_theta`.
fn step_to(after: &[f64], failure: Option<nlopt::FailState>) -> WeightedMstep {
    WeightedMstep {
        log_theta: after.to_vec(),
        log_sigma: Vec::new(),
        failure,
        improvable: vec![true; after.len()],
    }
}

const WIDE: ([f64; 2], [f64; 2]) = ([-10.0, -10.0], [10.0, 10.0]);

#[test]
fn motion_finds_the_coordinate_that_never_moved() {
    let (lo, hi) = WIDE;
    let mut m = MstepMotion::new(2);
    m.record(&[1.0, 2.0], &lo, &hi, &step_to(&[1.5, 2.0], None));
    m.record(&[1.5, 2.0], &lo, &hi, &step_to(&[1.5, 2.0], None));
    assert_eq!(m.frozen(), vec![1]);
    assert_eq!(m.solves, 2);
}

/// Measured from the start and not per solve, so a coordinate creeping below the
/// tolerance each iteration still counts once it has gone anywhere, and one that comes
/// back to its start still moved.
#[test]
fn motion_is_measured_from_the_start() {
    let (lo, hi) = WIDE;
    let mut m = MstepMotion::new(2);
    let mut x = 1.0;
    for _ in 0..4 {
        let next = x + 0.4 * MSTEP_MOVED_TOL;
        m.record(&[x, 2.0], &lo, &hi, &step_to(&[next, 2.0 + 1.0], None));
        x = next;
    }
    m.record(&[x, 3.0], &lo, &hi, &step_to(&[x, 2.0], None));
    assert!(m.frozen().is_empty(), "{:?}", m.frozen());

    let mut m = MstepMotion::new(1);
    let (lo, hi) = (&lo[..1], &hi[..1]);
    m.record(&[1.0], lo, hi, &step_to(&[1.0 + 0.5 * MSTEP_MOVED_TOL], None));
    assert_eq!(m.frozen(), vec![0], "below the tolerance is not a move");
}

/// A pinned coordinate belongs to a closed-form step or to `FIX`, so holding it is the
/// solve doing its job. If it is released later, its record starts there.
#[test]
fn motion_ignores_a_coordinate_while_it_is_pinned() {
    let mut m = MstepMotion::new(2);
    let (lo, hi) = ([1.0, -10.0], [1.0, 10.0]);
    m.record(&[1.0, 2.0], &lo, &hi, &step_to(&[1.0, 3.0], None));
    assert!(m.frozen().is_empty(), "{:?}", m.frozen());

    let (lo, hi) = WIDE;
    m.record(&[7.0, 3.0], &lo, &hi, &step_to(&[7.0, 4.0], None));
    assert_eq!(m.frozen(), vec![0], "owned from here, and it stayed at 7");
}

/// Not moving is only half of it: a coordinate no step would improve stays out, and the
/// latest solve decides.
#[test]
fn motion_needs_the_coordinate_to_be_improvable() {
    let (lo, hi) = WIDE;
    let mut m = MstepMotion::new(2);
    let mut step = step_to(&[1.0, 2.0], None);
    m.record(&[1.0, 2.0], &lo, &hi, &step);
    assert_eq!(m.frozen(), vec![0, 1]);
    step.improvable = vec![false, true];
    m.record(&[1.0, 2.0], &lo, &hi, &step);
    assert_eq!(m.frozen(), vec![1]);
}

/// A bowl with its minimum at the origin.
fn bowl(x: &[f64], _: Option<&mut [f64]>, _: &mut ()) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>() * 1e3
}

#[test]
fn improvable_reads_only_what_a_step_would_gain() {
    let (lo, hi) = ([-10.0; 3], [10.0; 3]);
    // At its optimum, off it, and a coordinate that moved in the solve.
    let start = [0.0, 2.0, 5.0];
    let x = [0.0, 2.0, 1.0];
    assert_eq!(improvable_coordinates(&bowl, &start, &x, &lo, &hi), vec![false, true, false]);
    // Within half a step of the optimum a step overshoots, so it reads as there.
    let x = [0.04, 0.0, 0.0];
    assert_eq!(improvable_coordinates(&bowl, &x, &x, &lo, &hi), vec![false; 3]);
    // Off its optimum, but pinned.
    let x = [3.0, 0.0, 0.0];
    assert_eq!(
        improvable_coordinates(&bowl, &x, &x, &[3.0, -10.0, -10.0], &[3.0, 10.0, 10.0]),
        vec![false; 3]
    );
}

/// On a bound the only step is inward, so it pays only if the optimum is inside.
#[test]
fn improvable_steps_inside_the_bounds() {
    let x = [1.0, 0.0];
    let optimum_outside = improvable_coordinates(&bowl, &x, &x, &[1.0, -1.0], &[10.0, 1.0]);
    assert_eq!(optimum_outside, vec![false, false]);
    let optimum_inside = improvable_coordinates(&bowl, &x, &x, &[-1.0, -1.0], &[1.0, 1.0]);
    assert_eq!(optimum_inside, vec![true, false]);
}

/// A flat objective gives no coordinate a reason to move.
#[test]
fn improvable_is_false_on_a_flat_objective() {
    let flat = |_: &[f64], _: Option<&mut [f64]>, _: &mut ()| 5.0;
    let x = [1.0, 2.0];
    let (lo, hi) = WIDE;
    assert_eq!(improvable_coordinates(&flat, &x, &x, &lo, &hi), vec![false; 2]);
}

/// The bar is one likelihood-ratio step on −2 log L, so 1.92 on −log L, either side of it.
#[test]
fn improvable_needs_a_likelihood_ratio_step() {
    let (lo, hi) = (&WIDE.0[..1], &WIDE.1[..1]);
    // A step of 0.1 down this slope gains `slope / 10` on −log L.
    let slope = |g: f64| move |x: &[f64], _: Option<&mut [f64]>, _: &mut ()| g * x[0];
    let x = [1.0];
    assert_eq!(improvable_coordinates(&slope(19.0), &x, &x, lo, hi), vec![false]);
    assert_eq!(improvable_coordinates(&slope(20.0), &x, &x, lo, hi), vec![true]);
}

#[test]
fn motion_counts_failed_solves() {
    let (lo, hi) = WIDE;
    let mut m = MstepMotion::new(2);
    m.record(&[1.0, 2.0], &lo, &hi, &step_to(&[1.5, 2.5], None));
    let rejected = Some(nlopt::FailState::InvalidArgs);
    m.record(&[1.5, 2.5], &lo, &hi, &step_to(&[1.5, 2.5], rejected));
    assert_eq!((m.solves, m.failed), (2, 1));
    assert!(matches!(m.last_failure, Some(nlopt::FailState::InvalidArgs)));
}

#[test]
fn roundoff_is_not_a_failed_solve() {
    use nlopt::{FailState, SuccessState};
    assert!(mstep_failure(&Ok((SuccessState::FtolReached, 0.0))).is_none());
    assert!(mstep_failure(&Err((FailState::RoundoffLimited, 0.0))).is_none());
    assert!(matches!(
        mstep_failure(&Err((FailState::InvalidArgs, 0.0))),
        Some(FailState::InvalidArgs)
    ));
    assert!(matches!(mstep_failure(&Err((FailState::Failure, 0.0))), Some(FailState::Failure)));
}

/// Every clause of the message, and a σ's name as well as a θ's: the index runs across
/// both.
#[test]
fn the_gate_demotes_and_names_what_froze() {
    let model = parse_model_string(WIDE_MODEL).unwrap();
    let n = 14;
    let (lo, hi) = (vec![-20.0; n], vec![20.0; n]);
    let mut after = vec![1.0; n];
    after[1] = 0.0;
    after[9] = 0.0;
    let mut m = MstepMotion::new(n);
    m.record(&vec![0.0; n], &lo, &hi, &step_to(&after, None));
    m.record(&after, &lo, &hi, &step_to(&after, None));

    let mut converged = true;
    let w = gate_converged_on_mstep_motion(&mut converged, "IMPMAP", &m, &model).unwrap();
    assert!(!converged);
    assert!(w.starts_with("W_MCEM_MSTEP_FROZEN: IMPMAP: [LOG_V1, A1] never left"), "{w}");
    assert!(w.contains("in 2 weighted M-step(s)"), "{w}");
    assert!(w.contains("still lowers −2 log L by more than 3.84"), "{w}");
    assert!(w.contains("reported converged: false"), "{w}");
    assert!(w.contains("starting values and not estimates"), "{w}");
    assert!(w.contains("mu-reference them"), "{w}");
    assert!(!w.contains("NLopt"), "{w}");
}

/// #2718 review: every coordinate moved on the first solve, and the failures after it
/// still have to demote the run. The motion gate has nothing to say here, and a stopped
/// run is told too.
#[test]
fn a_failure_demotes_a_run_whose_coordinates_all_moved() {
    let model = parse_model_string(WIDE_MODEL).unwrap();
    let (lo, hi) = WIDE;
    let mut m = MstepMotion::new(2);
    m.record(&[1.0, 2.0], &lo, &hi, &step_to(&[1.5, 2.5], None));
    let rejected = Some(nlopt::FailState::InvalidArgs);
    m.record(&[1.5, 2.5], &lo, &hi, &step_to(&[1.5, 2.5], rejected));
    m.record(&[1.5, 2.5], &lo, &hi, &step_to(&[1.5, 2.5], rejected));
    assert!(m.frozen().is_empty());

    for stopped in [false, true] {
        let mut converged = true;
        let w = mstep_warnings(&mut converged, "IMP", &m, &model, stopped);
        assert!(!converged);
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(w[0].starts_with("W_MCEM_MSTEP_FAILED: IMP: NLopt failed 2 of 3"), "{w:?}");
        assert!(w[0].contains("(InvalidArgs)"), "{w:?}");
        assert!(w[0].contains("reported converged: false"), "{w:?}");
        assert!(w[0].contains("leave θ and σ where they were"), "{w:?}");
        assert!(w[0].contains("solver fault"), "{w:?}");
    }
}

/// A run `max_time` stopped is not judged on motion, and one that finished is.
#[test]
fn a_stopped_run_is_not_judged_on_motion() {
    let model = parse_model_string(WIDE_MODEL).unwrap();
    let (lo, hi) = WIDE;
    let mut m = MstepMotion::new(2);
    m.record(&[1.0, 2.0], &lo, &hi, &step_to(&[1.5, 2.0], None));
    let mut converged = true;
    assert!(mstep_warnings(&mut converged, "IMP", &m, &model, true).is_empty());
    assert!(converged);
    let w = mstep_warnings(&mut converged, "IMP", &m, &model, false);
    assert_eq!(w.len(), 1, "{w:?}");
    assert!(!converged);
}

/// #2718 review, the differential pair. No coordinate moves in either arm, because every
/// solve fails, so the only difference is where they start. From a converged solve's
/// result nothing is improvable; from `WIDE_MODEL`'s start the same non-movement is a
/// freeze. The objective repeats between solves, as it does for a coordinate the draws do
/// not reach.
#[test]
fn an_unmoved_coordinate_at_its_optimum_is_not_frozen() {
    let mut w = WideSolve::new();
    let motion_of = |w: &WideSolve| {
        let n = w.theta.len() + w.log_sigma.len();
        let (mut lo, mut hi) = (w.theta_lower.clone(), w.theta_upper.clone());
        lo.extend(vec![-8.0; w.log_sigma.len()]);
        hi.extend(vec![5.0; w.log_sigma.len()]);
        let before = [w.theta.as_slice(), w.log_sigma.as_slice()].concat();
        let mut m = MstepMotion::new(n);
        FAIL_MSTEP_SOLVES_AFTER.set(Some(0));
        for _ in 0..3 {
            let step = w.solve(4);
            assert_eq!(w.moved(&step), 0);
            m.record(&before, &lo, &hi, &step);
        }
        FAIL_MSTEP_SOLVES_AFTER.set(None);
        m
    };
    let cold = motion_of(&w).frozen();
    // Realised: LOG_CL, LOG_V1, LOG_Q, LOG_V2, LOG_KA, B_CLSEX and A2. The other seven
    // are within what a likelihood-ratio step can see from this start.
    assert_eq!(cold, vec![0, 1, 2, 3, 4, 7, 11]);

    let optimum = w.solve(400);
    w.theta = optimum.log_theta;
    w.log_sigma = optimum.log_sigma;
    assert_eq!(motion_of(&w).frozen(), Vec::<usize>::new());
}

/// The verdict has to read the same on the structured surface, for both messages and
/// behind a chain prefix. Without its token arm the frozen one reads as `Info` under
/// `mu_referencing`, because it says "mu-reference", and the failed one as `Warning` under
/// `general`, beside a demoted `converged`.
#[test]
fn the_gate_is_classified_as_a_critical_convergence_warning() {
    use crate::types::{classify_warning, WarningSeverity};
    let model = parse_model_string(WIDE_MODEL).unwrap();
    let (lo, hi) = WIDE;
    let mut m = MstepMotion::new(2);
    m.record(&[1.0, 2.0], &lo, &hi, &step_to(&[1.5, 2.0], None));
    let mut converged = true;
    let stopped = gate_converged_on_mstep_motion(&mut converged, "IMP", &m, &model).unwrap();
    let rejected = Some(nlopt::FailState::InvalidArgs);
    m.record(&[1.5, 2.0], &lo, &hi, &step_to(&[1.5, 2.0], rejected));
    let failed = gate_converged_on_mstep_failures(&mut converged, "IMP", &m).unwrap();
    assert!(stopped.contains("mu-reference") && !stopped.contains("NLopt"));
    assert!(failed.contains("NLopt") && !failed.contains("mu-reference"));
    let one_cpt = parse_model_string(ONE_CPT).unwrap();
    let opts = FitOptions {
        methods: vec![EstimationMethod::Imp],
        imp_iterations: 1,
        imp_samples: 50,
        imp_auto: false,
        imp_seed: Some(11),
        run_covariance_step: false,
        ..Default::default()
    };
    let base = crate::api::fit(&one_cpt, &one_cpt_pop(), &one_cpt.default_params, &opts).unwrap();
    for msg in [stopped, failed] {
        for raw in [msg.clone(), format!("[IMP] {msg}")] {
            let w = classify_warning(&raw);
            assert_eq!(
                (w.severity, w.category.as_str()),
                (WarningSeverity::Critical, "convergence"),
                "{raw}"
            );
            assert_eq!(w.message, msg);
        }
        // The path `fit()` publishes through.
        let mut fit = base.clone();
        fit.warnings.push(msg.clone());
        crate::api::rebuild_warnings_structured(&mut fit);
        let entry = fit
            .warnings_structured
            .iter()
            .find(|e| e.message == msg)
            .unwrap_or_else(|| panic!("{:?}", fit.warnings_structured));
        assert_eq!(entry.severity, WarningSeverity::Critical);
        assert_eq!(entry.category.as_str(), "convergence");
    }
}

#[test]
fn the_gate_is_silent_when_everything_moved() {
    let model = parse_model_string(WIDE_MODEL).unwrap();
    let (lo, hi) = WIDE;
    let mut m = MstepMotion::new(2);
    m.record(&[1.0, 2.0], &lo, &hi, &step_to(&[1.5, 2.5], None));
    let mut converged = true;
    assert!(gate_converged_on_mstep_motion(&mut converged, "IMP", &m, &model).is_none());
    assert!(converged);
}

const ONE_CPT: &str = r"
[parameters]
  theta TVCL(1.5, 0.01, 100.0)
  theta TVV(12.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.09
  sigma EPS ~ 0.2 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

const MIX_ONE_CPT: &str = r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(3.0, 0.01, 100.0)
  theta TVV(12.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma EPS ~ 0.2 (sd)

[mixture]
  nsub = 2
  logit(1) = MIXL

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

fn one_cpt_pop() -> Population {
    let mut csv = String::from("ID,TIME,DV,AMT,EVID,CMT\n");
    for id in 1..=8_usize {
        let cl = if id % 2 == 0 { 1.0 } else { 3.0 };
        csv.push_str(&format!("{id},0,0,100,1,1\n"));
        for (j, t) in [0.5_f64, 1.0, 2.0, 4.0, 8.0].iter().enumerate() {
            let c = 10.0 * (-(cl / 10.0) * t).exp();
            let bump = if (id + j) % 2 == 0 { 1.1 } else { 0.9 };
            csv.push_str(&format!("{id},{t},{:.6},0,0,1\n", c * bump));
        }
    }
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(csv.as_bytes()).unwrap();
    crate::read_nonmem_csv(f.path(), None, None).unwrap()
}

/// End to end on both methods and both drivers: with its solves failing the run reports
/// what it left at the start, and with them working it reports nothing. The estimates
/// the failed run returns for those coordinates are its starting values.
#[test]
fn a_run_reports_a_parameter_its_mstep_never_moved() {
    let pop = one_cpt_pop();
    let mut opts = FitOptions::default();
    opts.impmap_iterations = 3;
    opts.impmap_samples = 100;
    opts.impmap_seed = Some(11);
    opts.imp_iterations = 3;
    opts.imp_samples = 100;
    opts.imp_seed = Some(11);
    opts.run_covariance_step = false;
    let frozen = |r: &OuterResult| {
        r.warnings.iter().find(|w| w.starts_with(MCEM_MSTEP_FROZEN_TOKEN)).cloned()
    };
    let failed = |r: &OuterResult| {
        r.warnings.iter().find(|w| w.starts_with(MCEM_MSTEP_FAILED_TOKEN)).cloned()
    };
    for (source, label) in [(ONE_CPT, "plain"), (MIX_ONE_CPT, "mixture")] {
        let model = parse_model_string(source).unwrap();
        let p = &model.default_params;
        for imp in [false, true] {
            let run = || {
                if imp {
                    run_imp(&model, &pop, p, None, &opts).expect("imp")
                } else {
                    run_impmap(&model, &pop, p, None, &opts).expect("impmap")
                }
            };
            let healthy = run();
            assert_eq!(frozen(&healthy), None, "{label}, imp = {imp}");
            assert_eq!(failed(&healthy), None, "{label}, imp = {imp}");
            assert!(healthy.converged, "{label}, imp = {imp}");

            FAIL_MSTEP_SOLVES_AFTER.set(Some(0));
            let res = run();
            FAIL_MSTEP_SOLVES_AFTER.set(None);
            let w = frozen(&res).unwrap_or_else(|| panic!("{label}: {:?}", res.warnings));
            assert!(w.contains("TVV"), "{label}: {w}");
            let w = failed(&res).unwrap_or_else(|| panic!("{label}: {:?}", res.warnings));
            assert!(w.contains("NLopt failed 3 of 3 weighted"), "{w}");
            assert!(!res.converged, "{label}, imp = {imp}");
            let sigma_drift = (res.params.sigma.values[0] - p.sigma.values[0]).abs();
            assert!(sigma_drift < 1e-12, "{label}: {sigma_drift}");

            // The first solve works and the rest fail: everything moved, so only the
            // failure gate can catch it.
            FAIL_MSTEP_SOLVES_AFTER.set(Some(1));
            let res = run();
            FAIL_MSTEP_SOLVES_AFTER.set(None);
            assert_eq!(frozen(&res), None, "{label}, imp = {imp}");
            let w = failed(&res).unwrap_or_else(|| panic!("{label}: {:?}", res.warnings));
            assert!(w.contains("NLopt failed 2 of 3 weighted"), "{w}");
            assert!(!res.converged, "{label}, imp = {imp}");
        }
    }
}

// ── `mstep_optimizer = lbfgs` ──────────────────────────────────────────────

/// Three deterministic importance samples per subject at `scale`, stacked `[η | κ…]` for an
/// IOV subject, with unequal weights summing to 1.
fn sampled_draws(model: &CompiledModel, pop: &Population, scale: f64) -> Vec<SubjectDraws> {
    let n_eta = model.n_eta;
    pop.subjects
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let n_occ =
                if model.n_kappa > 0 { crate::stats::stacked_layout::group_count(s) } else { 0 };
            let width = n_eta + n_occ * model.n_kappa;
            let mut d = SubjectDraws::cancelled(n_eta);
            d.etas = (0..3)
                .map(|k| {
                    (0..width)
                        .map(|j| scale * (((i + 2 * j + 3 * k) % 7) as f64 - 3.0) / 3.0)
                        .collect()
                })
                .collect();
            d.weights = vec![0.2, 0.5, 0.3];
            d
        })
        .collect()
}

/// The packed `[log θ | log σ]` start, shifted off the model's defaults so no score is zero.
fn shifted_start(model: &CompiledModel) -> (Vec<f64>, Vec<bool>, Vec<f64>, Vec<f64>, usize, usize) {
    let p = &model.default_params;
    let mask: Vec<bool> = p.theta_lower.iter().map(|&l| l > 0.0).collect();
    let mut x: Vec<f64> = p
        .theta
        .iter()
        .zip(&mask)
        .enumerate()
        .map(|(i, (&t, &m))| {
            let v = t * (1.0 + 0.05 * (i % 3) as f64 + 0.1);
            if m {
                v.ln()
            } else {
                v + 0.07
            }
        })
        .collect();
    x.extend(p.sigma.values.iter().map(|s| (1.2 * s).ln()));
    let lower = vec![-50.0; x.len()];
    let upper = vec![50.0; x.len()];
    (x, mask, lower, upper, p.theta.len(), p.sigma.values.len())
}

/// Worst `|g - fd| / max(1, |g|)` of [`weighted_mstep_score`] against central differences
/// of [`weighted_mstep_objective`] over the packed vector, with the value checked on the way.
fn weighted_gradient_gap(model: &CompiledModel, pop: &Population, scale: f64, h: f64) -> f64 {
    let draws = sampled_draws(model, pop, scale);
    let schedules = vec![None; pop.subjects.len()];
    let (x, mask, lower, upper, n_theta, n_sigma) = shifted_start(model);
    let natural = |x: &[f64]| -> (Vec<f64>, Vec<f64>) {
        let th = (0..n_theta).map(|i| if mask[i] { x[i].exp() } else { x[i] }).collect();
        (th, x[n_theta..].iter().map(|v| v.exp()).collect())
    };
    let value = |x: &[f64]| {
        let (th, sg) = natural(x);
        weighted_mstep_objective(model, pop, &draws, &schedules, &th, &sg)
    };
    let (th, sg) = natural(&x);
    let score = weighted_mstep_score(
        model, pop, &draws, &schedules, &th, &sg, &mask, &lower, &upper, n_theta, n_sigma, false,
        None,
    )
    .expect("a score without the information always exists");
    let (v, g) = (score.val, score.grad);
    let v0 = value(&x);
    assert!(v0.is_finite() && v0 < 1e19, "objective {v0}");
    assert!(
        (v - v0).abs() < 1e-9 * v0.abs().max(1.0),
        "the gradient pass's value {v} is not the objective {v0}"
    );
    let mut worst = 0.0f64;
    let mut live = 0usize;
    for i in 0..x.len() {
        let (mut xp, mut xm) = (x.clone(), x.clone());
        xp[i] += h;
        xm[i] -= h;
        let fd = (value(&xp) - value(&xm)) / (2.0 * h);
        assert!(g[i].is_finite() && fd.is_finite(), "coordinate {i}: {} vs {fd}", g[i]);
        let gap = (g[i] - fd).abs() / g[i].abs().max(1.0);
        eprintln!("  coord {i}: grad {:.6e}  fd {fd:.6e}  gap {gap:.2e}", g[i]);
        live += usize::from(g[i].abs() > 1e-3);
        worst = worst.max(gap); // nan-ok: g[i] and fd asserted finite above
    }
    assert!(live >= x.len() / 2, "fixture: only {live} of {} scores are live", x.len());
    worst
}

/// The weighted gradient is the weighted sum of the per-sample fixed-η scores, checked against
/// central differences of the objective, on each class the M-step reaches: a closed-form
/// two-compartment model with covariate-carrying thetas and three combined-error strata (no
/// η), a one-compartment closed form with η, its ODE twin, and an IOV model whose samples
/// carry κ.
#[test]
fn the_weighted_gradient_matches_central_fd_of_the_objective() {
    let wide = WideSolve::new();
    let g = weighted_gradient_gap(&wide.model, &wide.pop, 0.0, 1e-5);
    eprintln!("wide (closed form, combined error, covariates): {g:.3e}");
    assert!(g < 1e-7, "wide: {g:.3e}");

    let warf = |path: &str| {
        let model = crate::parser::model_parser::parse_model_file(std::path::Path::new(path))
            .expect("parses");
        let pop = crate::read_nonmem_csv(std::path::Path::new("data/warfarin.csv"), None, None)
            .expect("reads");
        (model, pop)
    };
    let (model, pop) = warf("examples/warfarin.ferx");
    let g = weighted_gradient_gap(&model, &pop, 0.3, 1e-5);
    eprintln!("warfarin (closed form, eta): {g:.3e}");
    assert!(g < 1e-7, "warfarin: {g:.3e}");

    let (model, pop) = warf("examples/warfarin_ode.ferx");
    let g = weighted_gradient_gap(&model, &pop, 0.3, 1e-4);
    eprintln!("warfarin ODE: {g:.3e}");
    assert!(g < 2e-6, "warfarin ODE: {g:.3e}");

    let model = crate::parser::model_parser::parse_full_model(include_str!(
        "../../examples/warfarin_iov.ferx"
    ))
    .expect("parses")
    .model;
    let pop =
        crate::read_nonmem_csv(std::path::Path::new("data/warfarin_iov.csv"), None, Some("OCC"))
            .expect("reads");
    assert!(model.n_kappa > 0);
    let g = weighted_gradient_gap(&model, &pop, 0.3, 1e-5);
    eprintln!("warfarin IOV (kappa samples): {g:.3e}");
    assert!(g < 1e-7, "warfarin IOV: {g:.3e}");
}

/// Converged L-BFGS lands on converged BOBYQA's point on the 14-coordinate fixture, and a
/// budgeted L-BFGS solve captures what the BOBYQA budget test requires of BOBYQA.
#[test]
fn lbfgs_reaches_the_bobyqa_optimum_of_the_weighted_objective() {
    use crate::types::MstepOptimizer::Lbfgs;
    let w = WideSolve::new();
    let f_start = w.objective(&w.theta, &w.log_sigma);
    let reference = w.solve(400);
    let f_ref = w.objective(&reference.log_theta, &reference.log_sigma);
    let got = w.solve_with(400, Lbfgs);
    let f = w.objective(&got.log_theta, &got.log_sigma);
    eprintln!("converged: lbfgs {f:.6} vs bobyqa {f_ref:.6} (start {f_start:.2})");
    assert!(got.failure.is_none(), "{:?}", got.failure);
    assert!(f < f_ref + 1e-3, "lbfgs {f} vs bobyqa {f_ref}");
    for (maxiter, floor) in [(4_u32, 0.5), (8, 0.8)] {
        let step = w.solve_with(maxiter, Lbfgs);
        let f = w.objective(&step.log_theta, &step.log_sigma);
        let captured = (f_start - f) / (f_start - f_ref);
        eprintln!(
            "lbfgs maxiter {maxiter}: moved {} of 14, captured {captured:.4}, failure {:?}",
            w.moved(&step),
            step.failure
        );
        assert!(step.failure.is_none(), "{:?}", step.failure);
        assert_eq!(w.moved(&step), 14, "maxiter {maxiter}");
        assert!(captured > floor, "maxiter {maxiter}: captured {captured}");
    }
}

/// The mixture arm's gradient: one class holding all the responsibility is the
/// single-population solve, and two classes sharing the same draws with split
/// responsibility (a sum that is the same objective) land on it too.
#[test]
fn the_mixture_gradient_is_the_single_population_gradient() {
    use crate::types::MstepOptimizer::Lbfgs;
    let w = WideSolve::new();
    let n_sigma = w.log_sigma.len();
    let solve = |draws_by_class: &[Vec<SubjectDraws>], pmix: &[Vec<f64>]| {
        theta_sigma_weighted_mstep_mixture(
            &w.model,
            &w.pop,
            draws_by_class,
            &w.schedules,
            pmix,
            &w.theta,
            &w.log_sigma,
            &w.theta_lower,
            &w.theta_upper,
            &vec![-8.0; n_sigma],
            &vec![5.0; n_sigma],
            w.theta.len(),
            n_sigma,
            8,
            &vec![false; w.theta.len()],
            Lbfgs,
        )
    };
    let single = w.solve_with(8, Lbfgs);
    let one = solve(std::slice::from_ref(&w.draws), &vec![vec![1.0]; w.pop.subjects.len()]);
    assert_eq!(one.log_theta, single.log_theta);
    assert_eq!(one.log_sigma, single.log_sigma);

    let two = solve(
        &[WideSolve::new().draws, WideSolve::new().draws],
        &vec![vec![0.3, 0.7]; w.pop.subjects.len()],
    );
    let gap = two
        .log_theta
        .iter()
        .chain(&two.log_sigma)
        .zip(single.log_theta.iter().chain(&single.log_sigma))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, |m, d| if d > m { d } else { m });
    eprintln!("two-class split responsibility vs single: worst |dx| = {gap:.3e}");
    assert!(gap < 1e-6, "{gap:.3e}");
}

/// Fisher scoring on the weighted objective lands on converged BOBYQA's point on the
/// 14-coordinate fixture, and a one-step budget already moves every coordinate.
#[test]
fn newton_reaches_the_bobyqa_optimum_of_the_weighted_objective() {
    use crate::types::MstepOptimizer::Newton;
    let w = WideSolve::new();
    let f_start = w.objective(&w.theta, &w.log_sigma);
    let reference = w.solve(400);
    let f_ref = w.objective(&reference.log_theta, &reference.log_sigma);
    let got = w.solve_with(400, Newton);
    let f = w.objective(&got.log_theta, &got.log_sigma);
    eprintln!("converged: newton {f:.6} vs bobyqa {f_ref:.6} (start {f_start:.2})");
    assert!(got.failure.is_none(), "{:?}", got.failure);
    assert!(f < f_ref + 1e-3, "newton {f} vs bobyqa {f_ref}");
    assert!(f < f_start - 0.5 * (f_start - f_ref), "must move toward the optimum");
    assert_eq!(w.moved(&got), 14);
}

/// The weighted objective and score with the solve's prediction memo are the memo-free ones bit
/// for bit across a σ-only sequence and a θ move, in both modes: a tracking memo (θ free,
/// Newton) replaces its predictions when θ moves and serves the same θ again,
/// and a pinned-θ memo declines a θ it does not hold.
#[test]
fn the_memoised_weighted_objective_is_the_plain_one_bit_for_bit() {
    let w = WideSolve::new();
    let sizes = || w.draws.iter().map(|d| d.etas.len());
    let tracking = PredMemo::for_solve(sizes(), true, true).unwrap();
    let eval = |memo: &PredMemo, draws: &[SubjectDraws], theta: &[f64], ls: &[f64], on: bool| {
        let sg: Vec<f64> = ls.iter().map(|s| s.exp()).collect();
        weighted_mstep_objective_memo(
            &w.model,
            &w.pop,
            draws,
            &w.schedules,
            theta,
            &sg,
            on.then(|| memo.begin(theta)).flatten(),
        )
    };
    for bump in [0.0, 0.2, -0.3] {
        let ls: Vec<f64> = w.log_sigma.iter().map(|s| s + bump).collect();
        assert_eq!(
            eval(&tracking, &w.draws, &w.theta, &ls, false),
            eval(&tracking, &w.draws, &w.theta, &ls, true),
            "σ bump {bump}"
        );
    }
    let moved: Vec<f64> = w.theta.iter().map(|t| t + 0.05).collect();
    assert_eq!(
        eval(&tracking, &w.draws, &moved, &w.log_sigma, false),
        eval(&tracking, &w.draws, &moved, &w.log_sigma, true)
    );
    // Back at the first θ: the memo was replaced by the moved one, so this recomputes.
    assert_eq!(
        eval(&tracking, &w.draws, &w.theta, &w.log_sigma, false),
        eval(&tracking, &w.draws, &w.theta, &w.log_sigma, true)
    );
    // The score and its information through a memo are the memo-free ones, at one θ twice
    // (the value pass, then the fused pass at the accepted point).
    let (x, mask, lower, upper, n_theta, n_sigma) = shifted_start(&w.model);
    let th: Vec<f64> = (0..n_theta).map(|i| if mask[i] { x[i].exp() } else { x[i] }).collect();
    let sg: Vec<f64> = x[n_theta..].iter().map(|v| v.exp()).collect();
    let score = |want_fisher: bool, memo: Option<EvalMemo<'_>>| {
        weighted_mstep_score(
            &w.model,
            &w.pop,
            &w.draws,
            &w.schedules,
            &th,
            &sg,
            &mask,
            &lower,
            &upper,
            n_theta,
            n_sigma,
            want_fisher,
            memo,
        )
        .expect("in scope")
    };
    let fresh = PredMemo::for_solve(sizes(), true, true).unwrap();
    let value_pass = weighted_mstep_objective_memo(
        &w.model,
        &w.pop,
        &w.draws,
        &w.schedules,
        &th,
        &sg,
        fresh.begin(&th),
    );
    let (plain, memoised) = (score(true, None), score(true, fresh.begin(&th)));
    assert_eq!(value_pass, plain.val, "the value pass is the fused pass's value");
    assert_eq!(
        (plain.val, &plain.grad, &plain.info),
        (memoised.val, &memoised.grad, &memoised.info)
    );
    assert_eq!(plain.info.len(), (n_theta + n_sigma).pow(2));
    // A pinned-θ memo holds the first θ and declines another.
    let fixed = PredMemo::for_solve(sizes(), false, false).unwrap();
    assert!(fixed.begin(&th).is_some() && fixed.begin(&moved).is_none());
}

/// `fisher_scoring` is handed the solve's own state and reports one of three outcomes, and the
/// weighted solve acts on each: `Converged` is the answer (route `newton`); a start without
/// information (`Declined`) and a solve that loses it after one accepted step (`Stalled`) both
/// continue with BOBYQA — from the start, and from the point Newton reached — and the
/// continuation is exactly BOBYQA's solve from there, with **what Newton left of the one
/// evaluation budget**, not the start-point `x` Newton left nor a fresh budget.
#[test]
fn a_newton_solve_that_does_not_converge_continues_with_bobyqa_from_where_it_stopped() {
    use crate::estimation::mstep_newton::{NewtonOutcome, FUSED_PASSES_ALLOWED};
    use crate::estimation::saem::MSTEP_ROUTE;
    use crate::types::MstepOptimizer::{Bobyqa, Newton};
    let w = WideSolve::new();
    let n_sigma = w.log_sigma.len();
    let n_theta = w.theta.len();
    let n = n_theta + n_sigma;
    let budget = 4 * (n as u32 + 1);
    let x0: Vec<f64> = w.theta.iter().chain(&w.log_sigma).copied().collect();
    let lower: Vec<f64> = w.theta_lower.iter().copied().chain(vec![-8.0; n_sigma]).collect();
    let upper: Vec<f64> = w.theta_upper.iter().copied().chain(vec![5.0; n_sigma]).collect();
    let obj = |xv: &[f64], _g: Option<&mut [f64]>, _: &mut ()| {
        w.objective(&xv[..n_theta], &xv[n_theta..])
    };
    let mask = vec![false; n_theta];
    let fused = |xv: &[f64]| {
        let sg: Vec<f64> = xv[n_theta..].iter().map(|v| v.exp()).collect();
        weighted_mstep_score(
            &w.model,
            &w.pop,
            &w.draws,
            &w.schedules,
            &xv[..n_theta],
            &sg,
            &mask,
            &lower,
            &upper,
            n_theta,
            n_sigma,
            true,
            None,
        )
    };
    let handovers = std::cell::Cell::new(0u32);
    let solve = |from: &[f64], optimizer, passes: Option<u32>, maxeval: u32| {
        MSTEP_ROUTE.with(|c| c.set([0; 3]));
        handovers.set(0);
        FUSED_PASSES_ALLOWED.set(passes);
        let f: &dyn Fn(&[f64]) -> Option<ValueGradInfo> = &fused;
        let step = solve_weighted_mstep(
            obj,
            from.to_vec(),
            &lower,
            &upper,
            n_theta,
            maxeval,
            optimizer,
            (optimizer == Newton).then_some(f),
            &|| handovers.set(handovers.get() + 1),
            None,
        );
        FUSED_PASSES_ALLOWED.set(None);
        (MSTEP_ROUTE.with(|c| c.get()), step)
    };
    let packed = |s: &WeightedMstep| -> Vec<f64> {
        s.log_theta.iter().chain(&s.log_sigma).copied().collect()
    };
    // Converged: Newton's own answer, and not BOBYQA's.
    let (route, newton) = solve(&x0, Newton, None, budget);
    assert_eq!(route, [0, 0, 1]);
    assert_eq!(handovers.get(), 0, "no continuation, no hand-over");
    // Declined at the start (no fused pass): BOBYQA from `x0`, exactly, with the whole budget
    // (a declined pass spent none).
    let (route, declined) = solve(&x0, Newton, Some(0), budget);
    assert_eq!(handovers.get(), 1, "the memo is told BOBYQA takes over, once");
    let (_, bobyqa0) = solve(&x0, Bobyqa, None, budget);
    assert_eq!(handovers.get(), 0, "an explicit bobyqa solve is not a hand-over");
    assert_eq!(route, [1, 0, 0]);
    assert_eq!(packed(&declined), packed(&bobyqa0));
    assert!(packed(&declined) != packed(&newton), "fixture: the routes must differ");
    // Stalled after one accepted step: BOBYQA from the point Newton reached, with the
    // evaluations it did not spend.
    let mut x1 = x0.clone();
    FUSED_PASSES_ALLOWED.set(Some(1));
    let (outcome, spent) = crate::estimation::mstep_newton::fisher_scoring(
        &mut x1,
        &lower,
        &upper,
        budget,
        MSTEP_FTOL_REL,
        |xv| obj(xv, None, &mut ()),
        &fused,
    );
    FUSED_PASSES_ALLOWED.set(None);
    assert_eq!(outcome, NewtonOutcome::Lost);
    assert!(x1 != x0, "Newton took a step before it stalled");
    assert!(spent > 0 && spent < budget, "fixture: Newton spent {spent} of {budget}");
    let (route, stalled) = solve(&x0, Newton, Some(1), budget);
    assert_eq!(handovers.get(), 1, "a stall after one step hands over too");
    let (_, bobyqa1) = solve(&x1, Bobyqa, None, budget - spent);
    assert_eq!(route, [1, 0, 0]);
    assert_eq!(
        packed(&stalled),
        packed(&bobyqa1),
        "BOBYQA continued from Newton's point with the remainder of the budget"
    );
    assert!(packed(&stalled) != packed(&bobyqa0), "and not from the start");
    // Non-vacuity of the budget: a fresh budget from the same point is another solve (BOBYQA
    // is far from converged at this size), so the remainder is what was used.
    let (_, fresh) = solve(&x1, Bobyqa, None, budget);
    assert!(
        packed(&stalled) != packed(&fresh),
        "fixture: the remainder must differ from a fresh budget"
    );
    // A budget Newton spends whole leaves BOBYQA none (NLopt reads 0 as no limit): the solve
    // ends at Newton's point, whatever it is.
    let (route, spent_all) = solve(&x0, Newton, Some(1), spent);
    assert_eq!(route, [1, 0, 0]);
    assert_eq!(packed(&spent_all), x1, "no budget left: Newton's point stands");
}

/// What a weighted M-step keeps in memory is decided by the optimizer, not only by the solve's
/// structure: the memo is `subjects × samples × n_obs` numbers (`imp_samples` is 1000 by
/// default), so the derivative-free default keeps none even where every evaluation could be
/// served from one (a σ-only solve, the pinned θ here). The opt-in gradient solvers keep it
/// where an evaluation can repeat a θ: `newton` for the pinned-θ solve and for the free-θ one
/// (whose value pass and fused pass share a point), `lbfgs` for the pinned-θ solve only — with
/// a free θ it never sees one twice.
#[test]
fn only_an_opt_in_gradient_weighted_mstep_keeps_predictions() {
    use crate::estimation::pred_memo::OPENED;
    use crate::types::MstepOptimizer::{Bobyqa, Lbfgs, Newton};
    let mut pinned = WideSolve::new();
    pinned.theta_lower = pinned.theta.clone();
    pinned.theta_upper = pinned.theta.clone();
    let free = WideSolve::new();
    let opened = |w: &WideSolve, optimizer| {
        OPENED.with(|c| c.set(0));
        w.solve_with(1, optimizer);
        OPENED.with(|c| c.get())
    };
    assert_eq!(opened(&pinned, Bobyqa), 0, "bobyqa, σ-only");
    assert_eq!(opened(&free, Bobyqa), 0, "bobyqa, free θ");
    assert_eq!(opened(&pinned, Lbfgs), 1, "lbfgs, σ-only");
    assert_eq!(opened(&free, Lbfgs), 0, "lbfgs, free θ visits no θ twice");
    assert_eq!(opened(&pinned, Newton), 1, "newton, σ-only");
    assert_eq!(opened(&free, Newton), 1, "newton, free θ");
}

/// The test hook counts a weighted M-step once, however many branches of the solve read it:
/// `FAIL_MSTEP_SOLVES_AFTER = 1` lets the first solve work and fails the second, on `bobyqa`
/// and on a `newton` request alike (which a failed solve must not run).
#[test]
fn a_solve_consumes_one_forced_failure_count() {
    use crate::estimation::saem::MSTEP_ROUTE;
    use crate::types::MstepOptimizer::{Bobyqa, Newton};
    let w = WideSolve::new();
    for optimizer in [Bobyqa, Newton] {
        MSTEP_ROUTE.with(|c| c.set([0; 3]));
        FAIL_MSTEP_SOLVES_AFTER.set(Some(1));
        let first = w.solve_with(4, optimizer);
        let second = w.solve_with(4, optimizer);
        FAIL_MSTEP_SOLVES_AFTER.set(None);
        assert!(first.failure.is_none(), "{optimizer:?}: the first solve works");
        assert!(
            matches!(second.failure, Some(nlopt::FailState::Failure)),
            "{optimizer:?}: the second fails"
        );
        let used = MSTEP_ROUTE.with(|c| c.get());
        assert_eq!(
            used[2],
            u32::from(optimizer == Newton),
            "{optimizer:?}: a forced failure runs no Newton"
        );
    }
}

/// End to end, the `newton` request is one decision per fit: where the closed-form information
/// exists it runs Newton, silently; on a mixture it runs `bobyqa` and the fit says why — for
/// IMP and IMPMAP alike.
#[test]
fn newton_runs_where_its_scope_holds_and_says_so_where_it_does_not() {
    use crate::estimation::saem::MSTEP_ROUTE;
    let pop = one_cpt_pop();
    let mut opts = FitOptions::default();
    opts.impmap_iterations = 3;
    opts.impmap_samples = 100;
    opts.impmap_seed = Some(11);
    opts.imp_iterations = 3;
    opts.imp_samples = 100;
    opts.imp_seed = Some(11);
    opts.run_covariance_step = false;
    opts.mstep_optimizer = crate::types::MstepOptimizer::Newton;
    for (source, in_scope) in [(ONE_CPT, true), (MIX_ONE_CPT, false)] {
        let model = parse_model_string(source).unwrap();
        let p = &model.default_params;
        for imp in [false, true] {
            MSTEP_ROUTE.with(|c| c.set([0; 3]));
            let res = if imp {
                run_imp(&model, &pop, p, None, &opts).expect("imp")
            } else {
                run_impmap(&model, &pop, p, None, &opts).expect("impmap")
            };
            let route = MSTEP_ROUTE.with(|c| c.get());
            let warning = res.warnings.iter().find(|w| w.contains("`mstep_optimizer = newton`"));
            let what = format!("in_scope = {in_scope}, imp = {imp}, route {route:?}");
            if in_scope {
                assert_eq!(warning, None, "{what}");
                assert!(route[2] > 0 && route[0] == 0, "{what}");
            } else {
                let w = warning.unwrap_or_else(|| panic!("{what}: {:?}", res.warnings));
                assert!(
                    w.contains("the model is a mixture") && w.contains("runs `bobyqa`"),
                    "{what}: {w}"
                );
                assert!(route[2] == 0 && route[0] > 0, "{what}");
            }
        }
    }
}
