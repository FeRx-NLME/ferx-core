//! Tier-1 tests for #2778: `fit()` scores each subject only up to its last scored record.
//!
//! The fit-level tests compare three fits of one model, bit for bit: the dataset with its
//! trailing doses as `fit()` now runs it, the same dataset with the view switched off
//! (`SCORE_ALL_RECORDS`, what `fit()` did before), and the dataset with those rows deleted.
//! The structural tests pin the horizon itself, one record kind at a time.

use super::*;
use crate::api::fit;
use crate::types::{CmtToggle, DoseEvent, EstimationMethod, FitOptions, FitResult, Optimizer};
use std::io::Write;

// ── dataset fixtures ─────────────────────────────────────────────────────────

/// The last observation of every subject in [`dataset`].
const LAST_OBS: f64 = 50.0;
const OBS_TIMES: [f64; 5] = [1.0, 4.0, 25.0, 30.0, LAST_OBS];
/// The first explicit trailing dose row of [`dataset`].
const TAIL_TIME: f64 = 240.0;

/// What [`dataset`] writes after the last observation.
#[derive(Clone, Copy, PartialEq)]
enum Tail {
    /// Nothing: the `ADDL` train stops at the last dose at or before the last observation.
    None,
    /// The train runs to day 10, and two explicit dose rows follow it.
    Doses,
}

/// A daily oral regimen sampled over two days, with `WT` rising at each observation.
///
/// `tail_wt` and `tail_occ` are the `WT` and `OCC` of the trailing rows. The train's first
/// row carries the `ADDL`, so it straddles the horizon.
fn dataset(tail: Tail, tail_wt: f64, tail_occ: u32) -> String {
    dataset_with_wt_step(tail, tail_wt, tail_occ, 1.0)
}

/// [`dataset`] with `WT` rising by `wt_step` at each observation.
fn dataset_with_wt_step(tail: Tail, tail_wt: f64, tail_occ: u32, wt_step: f64) -> String {
    let mut csv = String::from("ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,II,ADDL,WT,OCC\n");
    for id in 1..=3 {
        let wt = 60.0 + 5.0 * id as f64;
        let addl = if tail == Tail::Doses { 9 } else { 2 };
        csv += &format!("{id},0,.,1,100,1,0,1,24,{addl},{wt},1\n");
        for (j, t) in OBS_TIMES.iter().enumerate() {
            let dv = 4.0 + (id as f64 * 1.7 + j as f64).sin() + 0.1 * *t / 10.0;
            let occ = if *t < 24.0 { 1 } else { 2 };
            csv += &format!(
                "{id},{t},{dv:.5},0,.,1,0,0,0,0,{},{occ}\n",
                wt + wt_step * j as f64
            );
        }
        if tail == Tail::Doses {
            for t in [TAIL_TIME, TAIL_TIME + 24.0] {
                csv += &format!("{id},{t},.,1,100,1,0,1,0,0,{tail_wt},{tail_occ}\n");
            }
        }
    }
    csv
}

const PARAMS: &str = "
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TVKA(1.2, 0.05, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP ~ 0.2 (sd)
";

const ERROR_MODEL: &str = "
[error_model]
  DV ~ proportional(PROP)
";

const INDIVIDUAL: &str = "
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA
";

const CLOSED_FORM: &str = "
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
";

const ODE: &str = "
[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -KA*depot
  d/dt(central) = KA*depot/V - CL/V*central
";

const TRANSIT_INDIVIDUAL: &str = "
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV  * exp(ETA_V)
  KA  = TVKA
  MTT = 1.5
  NTR = 3.0
";

const TRANSIT: &str = "
[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = transit(n=NTR, mtt=MTT) - KA*depot
  d/dt(central) = KA*depot/V - CL/V*central
";

fn model_text(individual: &str, structural: &str) -> String {
    format!("{PARAMS}{individual}{structural}{ERROR_MODEL}")
}

/// One model file and one dataset, ready to fit.
fn prepared(model: &str, csv: &str) -> crate::api::PreparedRun {
    let mut model_file = tempfile::Builder::new().suffix(".ferx").tempfile().unwrap();
    model_file.write_all(model.as_bytes()).unwrap();
    let mut data_file = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    data_file.write_all(csv.as_bytes()).unwrap();
    crate::api::prepare_run(
        model_file.path().to_str().unwrap(),
        Some(data_file.path().to_str().unwrap()),
    )
    .unwrap_or_else(|e| panic!("prepare_run: {e}"))
}

/// A few outer iterations: enough for gradients and the inner loop to shape the trajectory,
/// so a difference anywhere on the path shows in the estimates.
fn few_iterations() -> FitOptions {
    FitOptions {
        method: EstimationMethod::Foce,
        interaction: true,
        optimizer: Optimizer::NloptLbfgs,
        outer_maxiter: 3,
        run_covariance_step: false,
        threads: Some(1),
        ..FitOptions::default()
    }
}

fn fitted(run: &crate::api::PreparedRun, options: &FitOptions) -> FitResult {
    fit(
        &run.parsed.model,
        &run.population,
        &run.init_params,
        options,
    )
    .unwrap_or_else(|e| panic!("fit: {e}"))
}

/// `fit()` as it ran before the view: every record scored.
fn fitted_on_every_record(run: &crate::api::PreparedRun, options: &FitOptions) -> FitResult {
    SCORE_ALL_RECORDS.with(|s| s.set(true));
    let result = fitted(run, options);
    SCORE_ALL_RECORDS.with(|s| s.set(false));
    result
}

/// What a fit reports that the likelihood decides, as bit patterns.
fn numbers(r: &FitResult) -> Vec<u64> {
    let mut v = vec![r.ofv.to_bits()];
    v.extend(r.theta.iter().map(|x| x.to_bits()));
    v.extend(r.sigma.iter().map(|x| x.to_bits()));
    v.extend(r.omega.iter().map(|x| x.to_bits()));
    for kappas in r.ebe_kappas.iter().flatten() {
        v.extend(kappas.iter().map(|x| x.to_bits()));
    }
    for s in &r.subjects {
        v.extend(s.eta.iter().map(|x| x.to_bits()));
        v.extend(s.ipred.iter().map(|x| x.to_bits()));
        v.extend(s.pred.iter().map(|x| x.to_bits()));
        v.extend(s.cwres.iter().map(|x| x.to_bits()));
        v.extend(s.npde.iter().map(|x| x.to_bits()));
    }
    v
}

/// The largest difference between two fits' [`numbers`], relative for a value above 1 and
/// absolute below. Asserts each is finite first, since `f64::max` would drop a `NaN`.
fn worst_relative_gap(a: &[u64], b: &[u64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut worst = 0.0_f64;
    for (&a, &b) in a.iter().zip(b) {
        let (a, b) = (f64::from_bits(a), f64::from_bits(b));
        assert!(a.is_finite() && b.is_finite());
        if a != b {
            worst = worst.max((a - b).abs() / a.abs().max(b.abs()).max(1.0));
        }
    }
    worst
}

/// Compare `fit()` on the trailing doses with the two references.
///
/// The fit must equal the rows deleted bit for bit, and must have run on the analytic
/// gradient, so the dual walks are what was compared. It must equal every record scored up
/// to `rounding`: 0 where the walk has no per-record parameters.
fn assert_tail_is_free(model: &str, tail_wt: f64, options: &FitOptions, rounding: f64) {
    let full = prepared(model, &dataset(Tail::Doses, tail_wt, 2));
    let trimmed = prepared(model, &dataset(Tail::None, 0.0, 0));

    let view = scored_population(&full.parsed.model, Cow::Borrowed(&full.population));
    assert_eq!(
        view.subjects[0].doses.len(),
        3,
        "doses at 0, 24 and 48 stay"
    );
    assert_eq!(full.population.subjects[0].doses.len(), 12);

    let new = fitted(&full, options);
    let old = fitted_on_every_record(&full, options);
    assert_eq!(new.gradient_method_inner, "analytic (Dual2)");
    assert_eq!(new.gradient_method_outer, "analytic (Dual2)");
    let start = FitOptions {
        outer_maxiter: 0,
        ..options.clone()
    };
    let at_start = fitted(&full, &start);
    assert_ne!(
        new.ofv.to_bits(),
        at_start.ofv.to_bits(),
        "the fit must move, or the comparison sees only the start"
    );
    // The objective and the EBEs carry no rounding difference on any route.
    assert_eq!(
        numbers(&at_start),
        numbers(&fitted_on_every_record(&full, &start))
    );
    assert_eq!(numbers(&new), numbers(&fitted(&trimmed, options)));
    let gap = worst_relative_gap(&numbers(&new), &numbers(&old));
    assert!(gap <= rounding, "against every record scored: {gap:e}");
}

// ── fits with and without trailing doses ─────────────────────────────────────

#[test]
fn trailing_doses_do_not_change_a_closed_form_fit() {
    assert_tail_is_free(
        &model_text(INDIVIDUAL, CLOSED_FORM),
        70.0,
        &few_iterations(),
        0.0,
    );
}

#[test]
fn trailing_doses_do_not_change_an_ode_fit() {
    assert_tail_is_free(&model_text(INDIVIDUAL, ODE), 70.0, &few_iterations(), 0.0);
}

/// A smooth-input walk through the trailing doses is what #2778 removes, and in a debug
/// build one fit of it takes over a minute, so this compares one subject at one point: the
/// inner solve and the dual outer gradient. `tests/trailing_doses_transit.rs` runs the whole
/// fit against the rows deleted.
#[test]
fn trailing_doses_do_not_change_a_transit_subject() {
    let run = prepared(
        &model_text(TRANSIT_INDIVIDUAL, TRANSIT),
        &dataset(Tail::Doses, 70.0, 2),
    );
    let model = &run.parsed.model;
    // Prepared as `fit()` prepares it: `WT` is unread here, and a subject still carrying its
    // snapshots takes the time-varying route, which differs at solver tolerance.
    let prepared_population = crate::api::fitted_population(
        model,
        &crate::types::IovOccasionRule::Column,
        false,
        Cow::Borrowed(&run.population),
        &mut Vec::new(),
    );
    let view = scored_population(model, Cow::Borrowed(&*prepared_population));
    let (every_record, cut) = (&prepared_population.subjects[0], &view.subjects[0]);
    assert_eq!(every_record.doses.len(), 12);
    assert_eq!(cut.doses.len(), 3);

    let params = &run.init_params;
    let ebe = |s: &Subject| {
        crate::estimation::inner_optimizer::find_ebe(model, s, params, 200, 1e-6, None, None, 0)
    };
    let (on_every_record, on_cut) = (ebe(every_record), ebe(cut));
    assert_eq!(on_every_record.nll.to_bits(), on_cut.nll.to_bits());
    let eta: Vec<f64> = on_cut.eta.iter().copied().collect();
    assert_eq!(on_every_record.eta.iter().copied().collect::<Vec<_>>(), eta);

    let x = crate::estimation::parameterization::pack_params(params);
    let gradient = |s: &Subject| {
        crate::estimation::sens_outer_gradient::subject_packed_gradient_foce(
            model, s, params, &x, &eta,
        )
        .expect("the dual path, not the FD route")
    };
    let bits = |g: Vec<f64>| g.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    let on_cut = gradient(cut);
    assert!(on_cut.iter().all(|g| g.is_finite()));
    assert!(on_cut.iter().any(|&g| g != 0.0));
    assert_eq!(bits(gradient(every_record)), bits(on_cut));
}

const TV_INDIVIDUAL: &str = "
[individual_parameters]
  CL = TVCL * (WT/70)^0.75 * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA
";

#[test]
fn trailing_doses_do_not_change_a_time_varying_covariate_fit() {
    let model = model_text(TV_INDIVIDUAL, ODE);
    let full = prepared(&model, &dataset(Tail::Doses, 90.0, 2));
    assert!(
        full.population.subjects[0].has_tv_covariates(),
        "the fixture must take the time-varying route"
    );
    assert_tail_is_free(&model, 90.0, &few_iterations(), PER_RECORD_ROUNDING);
}

/// A nonlinear [`ODE`]: certified linear, that system is propagated exactly on both routes,
/// and they agree to rounding.
const SATURABLE: &str = "
[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -KA*depot
  d/dt(central) = KA*depot/V - CL/V*central/(1 + central/100)
";

/// The gap between a few iterations of a fit and of the same fit on every record, where a
/// walk's parameters are per record (time-varying covariates, IOV).
///
/// The objective and EBEs agree bit for bit at any one point there and the outer gradient
/// only to rounding ([`a_per_record_gradient_differs_only_by_rounding`]), and the optimizer
/// amplifies that. The amplification depends on platform and build and does not grow
/// steadily with iterations, so this is not a rounding bound: worst measured 1.2e-10 in Linux
/// CI, 2.4e-10 in `tools/linux-test.sh`, 5.1e-14 on macOS. The bound leaves 40x over Linux
/// and stays 170x under the smallest real effect here, the 1.7e-6 of a route change.
const PER_RECORD_ROUNDING: f64 = 1e-8;

/// One gradient per subject, every record against the cut ones, at the subject's mode. The
/// EBE must match exactly and each gradient entry to `1e-13` relative. Measured worst 3e-15
/// on macOS and 2e-16 on Linux, IOV exact there.
fn assert_gradient_rounding(run: &crate::api::PreparedRun) {
    use crate::estimation::{inner_optimizer::find_ebe, sens_outer_gradient as outer};
    let model = &run.parsed.model;
    let every_record = crate::api::fitted_population(
        model,
        &crate::types::IovOccasionRule::Column,
        false,
        Cow::Borrowed(&run.population),
        &mut Vec::new(),
    );
    let view = scored_population(model, Cow::Borrowed(&*every_record));
    let mut params = run.init_params.clone();
    params.theta.iter_mut().for_each(|t| *t *= 1.0137);
    let x = crate::estimation::parameterization::pack_params(&params);
    for (full, cut) in every_record.subjects.iter().zip(&view.subjects) {
        assert!(cut.doses.len() < full.doses.len());
        let mode = find_ebe(model, full, &params, 500, 1e-10, None, None, 0);
        let on_cut = find_ebe(model, cut, &params, 500, 1e-10, None, None, 0);
        assert_eq!(mode.nll.to_bits(), on_cut.nll.to_bits());
        assert_eq!(mode.eta, on_cut.eta);
        let mut stacked: Vec<f64> = mode.eta.iter().copied().collect();
        stacked.extend(mode.kappas.iter().flatten().copied());
        let gradient = if model.n_kappa > 0 {
            outer::subject_packed_gradient_foce_iov
        } else {
            outer::subject_packed_gradient_foce
        };
        let on_full = gradient(model, full, &params, &x, &stacked).expect("the dual path");
        let on_cut = gradient(model, cut, &params, &x, &stacked).expect("the dual path");
        for (a, b) in on_full.iter().zip(&on_cut) {
            assert!(a.is_finite() && b.is_finite());
            assert!((a - b).abs() <= 1e-13 * a.abs().max(b.abs()), "{a:e} {b:e}");
        }
    }
}

#[test]
fn a_per_record_gradient_differs_only_by_rounding() {
    assert_gradient_rounding(&prepared(
        &model_text(TV_INDIVIDUAL, ODE),
        &dataset(Tail::Doses, 90.0, 2),
    ));
    assert_gradient_rounding(&prepared(&iov_model(), &dataset(Tail::Doses, 70.0, 2)));
}

/// A read covariate that varies only on trailing rows made the subject time-varying. On the
/// view it no longer does, so the subject leaves that route for the constant one, as it
/// would on the data without those rows. The two routes break the walk at different
/// records, so this is the one case that moves by solver tolerance rather than rounding.
#[test]
fn a_subject_time_varying_only_on_trailing_rows_takes_the_constant_route() {
    let model = model_text(TV_INDIVIDUAL, SATURABLE);
    let options = FitOptions {
        outer_maxiter: 0,
        ..few_iterations()
    };
    let full = prepared(&model, &dataset_with_wt_step(Tail::Doses, 95.0, 2, 0.0));
    let trimmed = prepared(&model, &dataset_with_wt_step(Tail::None, 0.0, 0, 0.0));
    assert!(full.population.subjects[0].has_tv_covariates());

    let new = fitted(&full, &options);
    assert_eq!(numbers(&new), numbers(&fitted(&trimmed, &options)));
    let old = fitted_on_every_record(&full, &options);
    let gap = worst_relative_gap(&numbers(&new), &numbers(&old));
    // Measured 1.7e-6 at the default `reltol = 1e-4` (objective 153.89223 to 153.89225);
    // the bound leaves 12x. Zero would mean the fixture never changed route.
    assert!(gap > 0.0 && gap <= 2e-5, "{gap:e}");
}

const IOV_PARAMS: &str = "
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TVKA(1.2, 0.05, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.02
  sigma PROP ~ 0.2 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA
";

fn iov_model() -> String {
    format!("{IOV_PARAMS}{CLOSED_FORM}{ERROR_MODEL}\n[fit_options]\n  iov_column = OCC\n")
}

fn iov_options() -> FitOptions {
    FitOptions {
        iov_column: Some("OCC".to_string()),
        ..few_iterations()
    }
}

#[test]
fn trailing_doses_do_not_change_an_iov_fit() {
    let options = iov_options();
    let full = prepared(&iov_model(), &dataset(Tail::Doses, 70.0, 2));
    assert_eq!(fitted(&full, &options).ebe_kappas[0].len(), 2);
    assert_tail_is_free(&iov_model(), 70.0, &options, PER_RECORD_ROUNDING);
}

/// An occasion only trailing rows carry keeps its κ block, so `kappas` stays laid out for
/// the population the caller holds and the fit is the one every record gives. Deleting the
/// rows would drop the block, which is why this compares against the switch alone.
#[test]
fn an_occasion_only_trailing_rows_carry_keeps_its_kappa_block() {
    let options = iov_options();
    let full = prepared(&iov_model(), &dataset(Tail::Doses, 70.0, 3));
    let view = scored_population(&full.parsed.model, Cow::Borrowed(&full.population));
    let groups = |s: &Subject| {
        crate::stats::likelihood::iov_occasion_groups(s)
            .into_iter()
            .map(|(occ, _)| occ)
            .collect::<Vec<_>>()
    };
    assert_eq!(groups(&view.subjects[0]), vec![1, 2, 3]);
    assert_eq!(
        groups(&view.subjects[0]),
        groups(&full.population.subjects[0])
    );
    assert_eq!(
        crate::stats::likelihood::iov_group_informed(&view.subjects[0], true, false),
        crate::stats::likelihood::iov_group_informed(&full.population.subjects[0], true, false),
    );

    let new = fitted(&full, &options);
    let old = fitted_on_every_record(&full, &options);
    assert_eq!(new.ebe_kappas[0].len(), 3);
    assert_eq!(old.ebe_kappas[0].len(), 3);
    let gap = worst_relative_gap(&numbers(&new), &numbers(&old));
    assert!(gap <= PER_RECORD_ROUNDING, "{gap:e}");
}

/// #2778's reason for not trimming in the reader: a data-derived `center` counts every
/// record. The trailing rows move the median, so deleting them changes the fit, and the
/// view must not.
#[test]
fn a_center_moved_by_trailing_rows_survives() {
    let individual = "
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA

[covariates]
  WT continuous

[covariate_model]
  CL ~ WT power(center = median) => THETA_CL_WT(0.75, 0.01, 3.0)
";
    let model = model_text(individual, CLOSED_FORM);
    let options = FitOptions {
        outer_maxiter: 0,
        ..few_iterations()
    };
    let full = prepared(&model, &dataset(Tail::Doses, 150.0, 2));
    let trimmed = prepared(&model, &dataset(Tail::None, 0.0, 0));
    let new = fitted(&full, &options);
    assert_eq!(
        numbers(&new),
        numbers(&fitted_on_every_record(&full, &options))
    );
    assert_ne!(
        new.ofv.to_bits(),
        fitted(&trimmed, &options).ofv.to_bits(),
        "the rows must move the centre, or this cannot tell a reader-level trim apart"
    );
}

/// NPDE simulates each subject from its doses, so it runs on the view too.
#[test]
fn trailing_doses_do_not_change_npde() {
    let full = prepared(
        &model_text(INDIVIDUAL, CLOSED_FORM),
        &dataset(Tail::Doses, 70.0, 2),
    );
    let options = FitOptions {
        outer_maxiter: 0,
        npde_nsim: 20,
        ..few_iterations()
    };
    let new = fitted(&full, &options);
    assert!(new.subjects[0].npde.iter().all(|x| x.is_finite()));
    assert!(!new.subjects[0].npde.is_empty());
    assert_eq!(
        numbers(&new),
        numbers(&fitted_on_every_record(&full, &options))
    );
}

// ── the horizon, at the objective ────────────────────────────────────────────

/// The objective of `fit()` at the start on one IV-bolus subject sampled at `LAST_OBS`,
/// with one more dose at `extra_dose`.
fn start_ofv(extra_dose: Option<f64>) -> f64 {
    let model = format!(
        "{PARAMS}{INDIVIDUAL}
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
{ERROR_MODEL}"
    );
    // `(time, file order at a tie, row)`: a dose goes ahead of a co-timed observation, so
    // the observation reads it. Listed after, the reader moves the observation below it.
    let mut rows = vec![
        (0.0, 0, "1,0,.,1,100,1,0,1".to_string()),
        (10.0, 1, "1,10,1.2,0,.,1,0,0".to_string()),
        (LAST_OBS, 1, format!("1,{LAST_OBS},0.9,0,.,1,0,0")),
    ];
    if let Some(t) = extra_dose {
        rows.push((t, 0, format!("1,{t},.,1,100,1,0,1")));
    }
    rows.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut csv = String::from("ID,TIME,DV,EVID,AMT,CMT,RATE,MDV\n");
    for (_, _, row) in rows {
        csv += &row;
        csv.push('\n');
    }
    let run = prepared(&model, &csv);
    let options = FitOptions {
        outer_maxiter: 0,
        ..few_iterations()
    };
    fitted(&run, &options).ofv
}

#[test]
fn a_dose_before_the_horizon_moves_the_objective_and_one_after_does_not() {
    let base = start_ofv(None);
    let before = (start_ofv(Some(LAST_OBS - 0.5)) - base).abs();
    assert!(before > 1.0, "{before}");
    assert_eq!(start_ofv(Some(LAST_OBS + 0.5)).to_bits(), base.to_bits());
}

/// A dose listed ahead of the last observation at its `TIME` reaches it. The bound is well
/// above the shift a one-ULP move of the observation makes, which alone would pass `!=`.
#[test]
fn a_dose_tied_with_the_last_observation_moves_the_objective() {
    let gap = (start_ofv(Some(LAST_OBS)) - start_ofv(None)).abs();
    assert!(gap > 1.0, "{gap}");
}

// ── what still reads the whole dataset ───────────────────────────────────────

/// `[derived]` grid integrals run past the last observation, so the trailing doses reach
/// them, and the caller's population comes back whole for `predict()` and `simulate()`.
#[test]
fn outputs_past_the_horizon_still_see_every_dose() {
    let derived = "
[derived]
  AUC_LATE = integral(compartments[1], from=230, to=280, step=0.5)
";
    let model = format!("{}{derived}", model_text(INDIVIDUAL, ODE));
    let options = FitOptions {
        outer_maxiter: 0,
        ..few_iterations()
    };
    let full = prepared(&model, &dataset(Tail::Doses, 70.0, 2));
    let trimmed = prepared(&model, &dataset(Tail::None, 0.0, 0));
    let before = crate::types::PopulationFingerprint::of(&full.population);

    let auc = |r: &FitResult| {
        let (_, values) = r.subjects[0]
            .extra_columns
            .iter()
            .find(|(name, _)| name == "AUC_LATE")
            .expect("AUC_LATE column");
        values[0]
    };
    let new = fitted(&full, &options);
    let without = fitted(&trimmed, &options);
    assert!(auc(&new).is_finite() && auc(&new) > 0.0);
    assert_eq!(
        auc(&new).to_bits(),
        auc(&fitted_on_every_record(&full, &options)).to_bits()
    );
    assert!(
        auc(&new) > 2.0 * auc(&without),
        "the late doses must reach the integral: {} against {}",
        auc(&new),
        auc(&without)
    );
    assert_eq!(
        before,
        crate::types::PopulationFingerprint::of(&full.population)
    );
}

/// The post-hoc steps resolve the view, past the fingerprint check of the full population,
/// and `run_covariance` repeats the inline step on it.
#[test]
fn run_covariance_scores_the_records_the_fit_scored() {
    let run = prepared(
        &model_text(TV_INDIVIDUAL, ODE),
        &dataset(Tail::Doses, 90.0, 2),
    );
    let options = FitOptions {
        run_covariance_step: true,
        ..few_iterations()
    };
    let inline = fitted(&run, &options);
    let posthoc = crate::run_covariance(
        &inline,
        Some(&run.parsed.model),
        Some(&run.population),
        &options,
    )
    .unwrap_or_else(|e| panic!("run_covariance: {e}"));
    let bits = |r: &FitResult| {
        let se = r.se_theta.as_ref().expect("standard errors");
        se.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    };
    assert_eq!(bits(&inline), bits(&posthoc));
    let inputs = crate::estimation::fit_inputs::resolve_fit_inputs(
        &inline,
        Some(&run.parsed.model),
        Some(&run.population),
        "test",
    )
    .unwrap_or_else(|e| panic!("resolve_fit_inputs: {e}"));
    assert_eq!(inputs.population().subjects[0].doses.len(), 3);
    assert_eq!(run.population.subjects[0].doses.len(), 12);
}

// ── the cut, one record kind at a time ───────────────────────────────────────

fn dose(t: f64) -> DoseEvent {
    DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0)
}

/// A subject observed at 1 and 5, dosed at 0, 5 and 9.
fn subject() -> Subject {
    Subject {
        id: "1".into(),
        doses: vec![dose(0.0), dose(5.0), dose(9.0)],
        obs_times: vec![1.0, 5.0],
        observations: vec![1.0, 1.0],
        obs_cmts: vec![1, 1],
        cens: vec![0, 0],
        ..Subject::default()
    }
}

fn kept_dose_times(s: &Subject) -> Vec<f64> {
    let mut s = s.clone();
    if let Some(kept) = kept_records(&s) {
        cut_subject(&mut s, kept);
    }
    s.doses.iter().map(|d| d.time).collect()
}

#[test]
fn a_dose_at_the_horizon_stays_and_one_after_goes() {
    assert_eq!(kept_dose_times(&subject()), vec![0.0, 5.0]);
}

#[test]
fn a_censored_row_is_a_scored_record() {
    let mut s = subject();
    s.obs_times.push(9.5);
    s.observations.push(0.1);
    s.obs_cmts.push(1);
    s.cens.push(1);
    assert_eq!(kept_dose_times(&s), vec![0.0, 5.0, 9.0]);
}

#[test]
fn the_first_dose_stays_when_every_observation_precedes_it() {
    let mut s = subject();
    s.doses = vec![dose(20.0), dose(30.0)];
    assert_eq!(kept_dose_times(&s), vec![20.0]);
}

#[test]
fn a_subject_with_no_scored_record_is_left_alone() {
    let mut s = subject();
    s.obs_times.clear();
    s.observations.clear();
    s.obs_cmts.clear();
    s.cens.clear();
    assert_eq!(kept_records(&s), None);
}

#[test]
fn the_kept_doses_are_a_prefix_on_an_unsorted_list() {
    let mut s = subject();
    s.doses = vec![dose(0.0), dose(9.0), dose(5.0), dose(12.0)];
    assert_eq!(kept_dose_times(&s), vec![0.0, 9.0, 5.0]);
}

#[test]
fn a_nan_dose_time_stays_for_the_engines_to_refuse() {
    let mut s = subject();
    s.doses.push(dose(f64::NAN));
    assert_eq!(kept_records(&s), None);
}

#[test]
fn every_list_parallel_to_a_cut_record_kind_is_cut_with_it() {
    let snapshot = |w: f64| std::collections::HashMap::from([("WT".to_string(), w)]);
    let mut s = subject();
    s.dose_covariates = vec![snapshot(1.0), snapshot(2.0), snapshot(3.0)];
    s.dose_occasions = vec![1, 1, 1];
    s.pk_only_times = vec![2.0, 7.0];
    s.pk_only_raw_times = vec![2.0, 7.0];
    s.pk_only_covariates = vec![snapshot(1.0), snapshot(2.0)];
    s.pk_only_occasions = vec![1, 1];
    s.cmt_toggles = vec![
        CmtToggle {
            pk_only: 0,
            state: 0,
            on: false,
        },
        CmtToggle {
            pk_only: 1,
            state: 0,
            on: true,
        },
    ];
    s.reset_times = vec![3.0, 8.0];
    s.reset_covariates = vec![snapshot(1.0), snapshot(2.0)];
    s.reset_occasions = vec![1, 1];

    let kept = kept_records(&s).expect("a tail");
    assert_eq!(
        kept,
        Kept {
            doses: 2,
            pk_only: 1,
            resets: 1
        }
    );
    cut_subject(&mut s, kept);
    assert_eq!(s.dose_covariates.len(), 2);
    assert_eq!(s.dose_occasions, vec![1, 1]);
    assert_eq!(s.pk_only_times, vec![2.0]);
    assert_eq!(s.pk_only_raw_times, vec![2.0]);
    assert_eq!(s.pk_only_covariates.len(), 1);
    assert_eq!(s.pk_only_occasions, vec![1]);
    assert_eq!(s.cmt_toggles.len(), 1);
    assert_eq!(s.cmt_toggles[0].pk_only, 0);
    assert_eq!(s.reset_times, vec![3.0]);
    assert_eq!(s.reset_covariates.len(), 1);
    assert_eq!(s.reset_occasions, vec![1]);
}

#[test]
fn a_reset_at_the_horizon_stays_and_one_after_goes_with_its_dose() {
    let mut s = subject();
    s.reset_times = vec![5.0, 9.0];
    let kept = kept_records(&s).expect("a tail");
    assert_eq!(kept.resets, 1);
    assert_eq!(kept.doses, 2);
}

#[test]
fn cut_labels_come_back_once_each_in_first_seen_order() {
    let mut labels = vec![1, 2, 5, 3, 5, 2, 4, 3];
    cut_occasions(&mut labels, 2);
    assert_eq!(labels, vec![1, 2, 5, 3, 4]);
}

#[test]
fn a_cut_list_with_no_new_label_stays_parallel() {
    let mut labels = vec![1, 2, 2, 1];
    cut_occasions(&mut labels, 2);
    assert_eq!(labels, vec![1, 2]);
}

#[test]
fn a_population_with_nothing_past_the_horizon_is_not_copied() {
    let run = prepared(
        &model_text(INDIVIDUAL, CLOSED_FORM),
        &dataset(Tail::None, 0.0, 0),
    );
    let view = scored_population(&run.parsed.model, Cow::Borrowed(&run.population));
    assert!(matches!(view, Cow::Borrowed(_)));
}

/// A nested model's cells must stay parallel to the records (`cells_ready`), so its subjects
/// keep them all.
#[test]
fn a_nested_level_model_keeps_every_record() {
    let model = crate::parser::model_parser::parse_model_string(
        "[parameters]\n  theta TVE0(10.0)\n  omega ETA_SP ~ 0.09\n  \
         level STUDY by STUDY in ID\n  level ARM by ARM in STUDY\n  \
         kappa K_ST ~ 0.04 at STUDY\n  kappa K_ARM ~ 4.0 at ARM\n  sigma ADD ~ 0.25\n\
         [individual_parameters]\n  E0 = TVE0 * exp(ETA_SP + K_ST + K_ARM)\n\
         [structural_model]\n  y = E0\n[error_model]\n  DV ~ additive(ADD)\n",
    )
    .expect("parse");
    assert!(model.has_nested_levels());
    let population = Population {
        subjects: vec![subject()],
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    assert!(kept_records(&population.subjects[0]).is_some());
    let view = scored_population(&model, Cow::Borrowed(&population));
    assert!(matches!(view, Cow::Borrowed(_)));
}

/// A covariate that varied only on a removed row leaves the subject on the fast route.
#[test]
fn a_covariate_varying_only_on_cut_rows_stops_being_time_varying() {
    let model = "
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.2 (sd)

[individual_parameters]
  CL = TVCL * (WT/70)^0.75 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)
";
    let csv = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,WT\n\
               1,0,.,1,100,1,0,1,70\n\
               1,2,1.5,0,.,1,0,0,70\n\
               1,9,.,1,100,1,0,1,95\n";
    let run = prepared(model, csv);
    assert!(run.population.subjects[0].has_tv_covariates());
    let view = scored_population(&run.parsed.model, Cow::Borrowed(&run.population));
    assert!(!view.subjects[0].has_tv_covariates());
}

#[cfg(feature = "survival")]
mod non_gaussian {
    use super::*;
    use crate::types::{EventType, ObsRecord};

    fn event(time: f64, event_type: EventType) -> ObsRecord {
        ObsRecord::Event {
            time,
            event_type,
            entry_time: 0.0,
            cmt: 2,
        }
    }

    /// `data/pktte_joint.csv`'s first subjects, with two doses after every subject's last
    /// row when `tail` is set. The hazard reads the drug, and each event follows the last
    /// PK sample, so the TTE row sets the horizon.
    fn joint_dataset(tail: bool) -> String {
        let csv = std::fs::read_to_string("data/pktte_joint.csv").expect("joint dataset");
        let mut lines = csv.lines();
        let mut out = format!("{}\n", lines.next().expect("header"));
        let rows: Vec<&str> = lines.collect();
        for id in 1..=3 {
            let prefix = format!("{id},");
            for row in rows.iter().filter(|r| r.starts_with(&prefix)) {
                out += row;
                out.push('\n');
            }
            if tail {
                for t in [100, 124] {
                    out += &format!("{id},{t},.,1,100,1,0,1\n");
                }
            }
        }
        out
    }

    #[test]
    fn trailing_doses_do_not_change_a_joint_pk_tte_fit() {
        let model = include_str!("../../examples/pktte_joint.ferx");
        let options = few_iterations();
        let full = prepared(model, &joint_dataset(true));
        let trimmed = prepared(model, &joint_dataset(false));
        let view = scored_population(&full.parsed.model, Cow::Borrowed(&full.population));
        assert_eq!(view.subjects[0].doses.len(), 2, "the next record stays");
        assert_eq!(full.population.subjects[0].doses.len(), 3);

        let new = fitted(&full, &options);
        assert_eq!(new.gradient_method_inner, "analytic (Dual2)");
        assert_eq!(new.gradient_method_outer, "analytic (Dual2)");
        assert_eq!(
            numbers(&new),
            numbers(&fitted_on_every_record(&full, &options))
        );
        assert_eq!(numbers(&new), numbers(&fitted(&trimmed, &options)));
    }

    #[test]
    fn a_tte_row_past_the_observations_extends_the_horizon() {
        let mut s = subject();
        s.doses.push(dose(11.0));
        s.doses.push(dose(13.0));
        s.obs_records = vec![event(10.0, EventType::RightCensored)];
        assert_eq!(last_scored_time(&s), Some(10.0));
        // The dose at 11 closes the interval holding the event, so it stays too.
        assert_eq!(kept_dose_times(&s), vec![0.0, 5.0, 9.0, 11.0]);
    }

    #[test]
    fn an_interval_censored_event_reaches_its_right_bound() {
        let mut s = subject();
        s.obs_records = vec![event(
            6.0,
            EventType::IntervalCensored {
                left: 6.0,
                right: 12.0,
            },
        )];
        assert_eq!(last_scored_time(&s), Some(12.0));
    }

    #[test]
    fn a_discrete_or_count_row_is_a_scored_record() {
        let mut s = subject();
        s.obs_records = vec![
            ObsRecord::DiscreteState {
                time: 7.0,
                raw_time: 7.0,
                state: 0,
                cmt: 2,
            },
            ObsRecord::Count {
                time: 8.0,
                raw_time: 8.0,
                count: 1,
                cmt: 3,
            },
        ];
        assert_eq!(last_scored_time(&s), Some(8.0));
    }
}
