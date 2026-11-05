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
            let bump = if (id + j) % 2 == 0 {
                1.0 + prop
            } else {
                1.0 - prop
            };
            csv.push_str(&format!(
                "{id},{t},{:.6},0,0,1,{lwt},{sex},{assay}\n",
                c * bump
            ));
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
        WideSolve {
            model,
            pop,
            draws,
            schedules,
            theta,
            log_sigma,
            theta_lower,
            theta_upper,
        }
    }

    fn solve(&self, maxiter: u32) -> WeightedMstep {
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
        start
            .zip(end)
            .filter(|(a, b)| (*a - *b).abs() > MSTEP_MOVED_TOL)
            .count()
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

/// #2718: the mixture M-step runs the same solver, so one class with all the
/// responsibility reproduces the single-population solve exactly.
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
        4,
        &vec![false; w.theta.len()],
    );
    let single = w.solve(4);
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
    m.record(
        &[1.0],
        lo,
        hi,
        &step_to(&[1.0 + 0.5 * MSTEP_MOVED_TOL], None),
    );
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

/// Staying on a bound is an answer: the optimum is outside the box. Both bounds, and off
/// the bound the same coordinate is frozen, so the exemption is what clears it.
#[test]
fn motion_exempts_a_coordinate_held_on_a_bound() {
    for at in [-10.0, 10.0] {
        let (lo, hi) = WIDE;
        let mut m = MstepMotion::new(2);
        m.record(&[at, 0.0], &lo, &hi, &step_to(&[at, 0.0], None));
        assert_eq!(m.frozen(), vec![1], "bound {at}");
    }
}

#[test]
fn motion_counts_failed_solves() {
    let (lo, hi) = WIDE;
    let mut m = MstepMotion::new(2);
    m.record(&[1.0, 2.0], &lo, &hi, &step_to(&[1.5, 2.5], None));
    let rejected = Some(nlopt::FailState::InvalidArgs);
    m.record(&[1.5, 2.5], &lo, &hi, &step_to(&[1.5, 2.5], rejected));
    assert_eq!((m.solves, m.failed), (2, 1));
    assert!(matches!(
        m.last_failure,
        Some(nlopt::FailState::InvalidArgs)
    ));
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
    assert!(matches!(
        mstep_failure(&Err((FailState::Failure, 0.0))),
        Some(FailState::Failure)
    ));
}

/// Every clause of the message, on both sides of the one conditional sentence, and a
/// σ's name as well as a θ's: the index runs across both.
#[test]
fn the_gate_demotes_and_names_what_froze() {
    let model = parse_model_string(WIDE_MODEL).unwrap();
    let n = 14;
    let (lo, hi) = (vec![-20.0; n], vec![20.0; n]);
    let mut after = vec![0.0; n];
    for x in after.iter_mut() {
        *x = 1.0;
    }
    after[1] = 0.0;
    after[9] = 0.0;
    let mut m = MstepMotion::new(n);
    m.record(&vec![0.0; n], &lo, &hi, &step_to(&after, None));
    m.record(&after, &lo, &hi, &step_to(&after, None));

    let mut converged = true;
    let w = gate_converged_on_mstep_motion(&mut converged, "IMPMAP", &m, &model).unwrap();
    assert!(!converged);
    assert!(
        w.starts_with("W_MCEM_MSTEP_FROZEN: IMPMAP: [LOG_V1, A1] never left"),
        "{w}"
    );
    assert!(w.contains("in 2 weighted M-step(s)"), "{w}");
    assert!(w.contains("reported converged: false"), "{w}");
    assert!(w.contains("starting values and not estimates"), "{w}");
    assert!(w.contains("stopped before reaching them"), "{w}");
    assert!(w.contains("mu-reference them"), "{w}");
    assert!(!w.contains("NLopt"), "{w}");

    let rejected = Some(nlopt::FailState::InvalidArgs);
    m.record(&after, &lo, &hi, &step_to(&after, rejected));
    let mut converged = true;
    let w = gate_converged_on_mstep_motion(&mut converged, "IMP", &m, &model).unwrap();
    assert!(!converged);
    assert!(
        w.contains("NLopt failed 1 of those solves (InvalidArgs)"),
        "{w}"
    );
    assert!(w.contains("solver fault"), "{w}");
    assert!(!w.contains("stopped before reaching them"), "{w}");
}

/// The verdict has to read the same on the structured surface, for both messages and
/// behind a chain prefix. Without its token arm the first reads as `Info` under
/// `mu_referencing`, because it says "mu-reference", and the second as `Warning` under
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
    let failed = gate_converged_on_mstep_motion(&mut converged, "IMP", &m, &model).unwrap();
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
        r.warnings
            .iter()
            .find(|w| w.starts_with(MCEM_MSTEP_FROZEN_TOKEN))
            .cloned()
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
            assert!(healthy.converged, "{label}, imp = {imp}");

            FAIL_MSTEP_SOLVES.set(true);
            let res = run();
            FAIL_MSTEP_SOLVES.set(false);
            let w = frozen(&res).unwrap_or_else(|| panic!("{label}: {:?}", res.warnings));
            assert!(w.contains("TVV"), "{label}: {w}");
            assert!(w.contains("EPS] never left"), "{label}: {w}");
            assert!(
                w.contains("NLopt failed 3 of those solves (Failure)"),
                "{w}"
            );
            assert!(!res.converged, "{label}, imp = {imp}");
            let sigma_drift = (res.params.sigma.values[0] - p.sigma.values[0]).abs();
            assert!(sigma_drift < 1e-12, "{label}: {sigma_drift}");
        }
    }
}
