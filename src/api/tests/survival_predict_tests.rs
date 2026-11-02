use super::grid_median_from_cumhaz;
use super::{predict_survival, predict_survival_diag, read_population_for};
use crate::parser::model_parser::parse_model_string;
use crate::types::{CompiledModel, Population};
use std::io::Write;

#[test]
fn grid_median_interpolates_and_handles_no_crossing() {
    let ln2 = std::f64::consts::LN_2;
    let grid = [0.0, 1.0, 2.0, 3.0];
    // H crosses ln2 between t=1 and t=2; linear interp puts the median at t=1.5
    // (frac = (ln2 − 0.5·ln2)/(1.5·ln2 − 0.5·ln2) = 0.5).
    let cum = [0.0, ln2 * 0.5, ln2 * 1.5, ln2 * 3.0];
    let m = grid_median_from_cumhaz(&grid, &cum);
    assert!(
        (m - 1.5).abs() < 1e-9,
        "median should interpolate to 1.5; got {m}"
    );
    // H never reaches ln2 on the grid → NaN.
    let never = grid_median_from_cumhaz(&grid, &[0.0, 0.1, 0.2, 0.3]);
    assert!(never.is_nan(), "no crossing → NaN; got {never}");
    // Median at/before the first grid point (H(grid[0]) ≥ ln2): interpolated from the
    // origin (0,0). grid[0]=2, H=2·ln2 ⇒ median = 2·ln2 / (2·ln2) = 1.0.
    let early = grid_median_from_cumhaz(&[2.0, 4.0], &[ln2 * 2.0, ln2 * 4.0]);
    assert!(
        (early - 1.0).abs() < 1e-9,
        "median before the first grid point interpolates from the origin; got {early}"
    );
}

// ── predict_survival_diag (#2164) ────────────────────────────────────────────

/// A compact pure-TTE Weibull model (no PK, no `[error_model]`) with its endpoint on `cmt`,
/// plus `extra` lines in `[parameters]`.
fn pure_tte(cmt: usize, extra: &str) -> CompiledModel {
    parse_model_string(&format!(
        "[parameters]\n  theta TVSCALE(20.0, 0.1, 500.0)\n  theta TVSHAPE(1.5, 0.1, 10.0)\n\
         {extra}  omega ETA_SCALE ~ 0.04\n[event_model]\n  cmt = {cmt}\n  family = weibull\n  \
         scale = TVSCALE * exp(ETA_SCALE)\n  shape = TVSHAPE\n"
    ))
    .expect("parse")
}

/// Read `csv` through the model-aware reader the R wrapper and `fit()` use, so the
/// population carries the reader's own findings rather than injected strings.
fn read(m: &CompiledModel, csv: &str) -> Population {
    let mut f = tempfile::NamedTempFile::new().expect("temp csv");
    f.write_all(csv.as_bytes()).expect("write csv");
    read_population_for(m, &None, f.path().to_str().unwrap(), None, None, None, &[])
        .expect("data must load")
        .0
}

/// Two subjects, one exact event and one right-censored, with an explicit `CMT` column.
fn clean_csv(cmt: usize) -> String {
    format!("ID,TIME,DV,EVID,AMT,CMT,MDV\n1,7,1,0,0,{cmt},0\n2,30,0,0,0,{cmt},0\n")
}

const GRID: [f64; 4] = [1.0, 5.0, 10.0, 30.0];

fn has(warnings: &[String], token: &str) -> bool {
    warnings.iter().any(|w| w.contains(token))
}

/// `warnings` without the process-global SS non-convergence note, which a concurrent lib test
/// can leave in the sink this entry point drains (see `non_fit_diagnostics_tests`).
fn without_ss_sink_note(warnings: &[String]) -> Vec<&String> {
    warnings
        .iter()
        .filter(|w| !w.contains("Steady-state (SS=1) equilibration"))
        .collect()
}

/// A clean pure-TTE run is silent, and the wrapper's rows are the `_diag` form's bit for bit.
///
/// Regression this catches: the split into `predict_survival_rows` reordering, re-sorting or
/// recomputing a row; and a bundle that pushes something unconditionally (which the silence
/// arm pins, and which would make every positive test below vacuous).
#[test]
fn predict_survival_is_its_diag_forms_rows_and_a_clean_run_is_silent() {
    let m = pure_tte(2, "");
    let p = read(&m, &clean_csv(2));
    assert!(
        p.warnings.is_empty(),
        "fixture must read clean: {:?}",
        p.warnings
    );
    let out = predict_survival_diag(&m, &p, &m.default_params, &GRID).unwrap();
    assert!(
        without_ss_sink_note(&out.warnings).is_empty(),
        "a clean TTE run must come back silent: {:?}",
        out.warnings
    );
    let rows = predict_survival(&m, &p, &m.default_params, &GRID).unwrap();
    assert_eq!(rows.len(), 2 * GRID.len());
    assert_eq!(rows.len(), out.results.len());
    for (a, b) in rows.iter().zip(&out.results) {
        assert!(a.survival.is_finite() && a.survival < 1.0, "{a:?}");
        let bits = |r: &super::SurvivalPredictionResult| {
            (
                r.id.clone(),
                r.cmt,
                [
                    r.time,
                    r.survival,
                    r.cum_hazard,
                    r.hazard,
                    r.cif,
                    r.survival_all,
                    r.median_survival,
                    r.mean_survival,
                ]
                .map(f64::to_bits),
            )
        };
        assert_eq!(bits(a), bits(b));
    }
}

/// A parse warning (an unreferenced θ) and a data-reader finding (`W_MISSING_DV`, from a
/// stray `CMT = 1` observation row with no DV beside the `cmt = 2` endpoint's rows) both reach
/// `warnings` — the two sources ferx-r#433 was filed about — and neither appears on the clean
/// twin above.
///
/// Mutation: drop `parse_warnings` or the `Population::warnings` relay from
/// `postfit::non_fit_diagnostics`, or return `Vec::new()` from `predict_survival_diag`, and
/// the matching assertion fails.
#[test]
fn predict_survival_diag_carries_parse_and_reader_findings() {
    let m = pure_tte(2, "  theta UNUSED(1.0, 0.1, 10.0)\n");
    assert!(
        !m.parse_warnings.is_empty(),
        "fixture must carry a parse warning"
    );
    let p = read(
        &m,
        "ID,TIME,DV,EVID,CMT\n1,5,.,0,1\n1,7,1,0,2\n2,30,0,0,2\n",
    );
    assert!(has(&p.warnings, "W_MISSING_DV"), "{:?}", p.warnings);
    let out = predict_survival_diag(&m, &p, &m.default_params, &GRID).unwrap();
    assert!(
        has(&out.warnings, "theta 'UNUSED'"),
        "a parse warning must reach predict_survival_diag: {:?}",
        out.warnings
    );
    assert!(
        has(&out.warnings, "W_MISSING_DV"),
        "a data-reader finding must reach predict_survival_diag: {:?}",
        out.warnings
    );
}

/// The reader findings go through core's one filter, not raw (#1645): a pure-TTE model whose
/// only endpoint is the default compartment has one provably-right answer for a missing `CMT`
/// column, so the `W_CMT_DEFAULTED` the reader raised is withheld — and no `W_NO_DOSES` is
/// reported on a dose-free endpoint-only dataset.
///
/// The straddle: the same data on the same model with its endpoint at `cmt = 2` reports
/// `W_CMT_DEFAULTED`, since there the defaulted rows land on no endpoint. Without it, the
/// first arm passes on an implementation that drops reader warnings wholesale; without the
/// first, the second passes on a raw relay of `Population::warnings` (the pattern ferx-r#426
/// removed from the R glue).
#[test]
fn predict_survival_diag_filters_reader_findings_like_fit() {
    const NO_CMT: &str = "ID,TIME,DV,EVID,AMT\n1,7,1,0,0\n2,30,0,0,0\n";
    let at_default = pure_tte(1, "");
    let p = read(&at_default, NO_CMT);
    assert!(
        has(&p.warnings, "W_CMT_DEFAULTED"),
        "the reader must raise it, or the filter is not what withholds it: {:?}",
        p.warnings
    );
    let out = predict_survival_diag(&at_default, &p, &at_default.default_params, &GRID).unwrap();
    assert!(
        !has(&out.warnings, "W_CMT_DEFAULTED") && !has(&out.warnings, "W_NO_DOSES"),
        "a compartment-free TTE model must not be told about compartments or doses: {:?}",
        out.warnings
    );

    let elsewhere = pure_tte(2, "");
    let p = read(&elsewhere, NO_CMT);
    let out = predict_survival_diag(&elsewhere, &p, &elsewhere.default_params, &GRID).unwrap();
    assert!(
        has(&out.warnings, "W_CMT_DEFAULTED"),
        "…while a defaulted CMT that misses the endpoint is a real finding: {:?}",
        out.warnings
    );
}

/// A joint PK-TTE model whose hazard reads a stiff two-state `[odes]` system.
fn joint_pk_tte(kfast: f64, max_steps: usize) -> CompiledModel {
    parse_model_string(&format!(
        "[parameters]\n  theta TVCL(1.0, 0.1, 50.0)\n  theta TVV(10.0, 1.0, 500.0)\n  \
         theta KFAST({kfast}, 1e-6, 1e8)\n  omega ETA_CL ~ 0.04\n  sigma PROP ~ 0.04\n\
         [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV\n  KF = KFAST\n\
         [structural_model]\n  ode(obs_cmt=central, states=[central, periph])\n\
         [odes]\n  d/dt(central) = -(CL / V) * central - KF * central + KF * periph\n  \
         d/dt(periph) = KF * central - KF * periph\n\
         [error_model]\n  DV ~ proportional(PROP)\n\
         [event_model]\n  cmt = 3\n  hazard = 0.001 * central / V\n\
         [fit_options]\n  ode_method = rk45\n  ode_max_steps = {max_steps}\n"
    ))
    .expect("parse")
}

const JOINT_CSV: &str = "ID,TIME,DV,EVID,AMT,CMT,MDV\n1,0,.,1,100,1,1\n1,20,1,0,.,3,0\n\
                         2,0,.,1,100,1,1\n2,30,0,0,.,3,0\n";

/// An ODE-accumulated hazard's solver diagnostics reach `warnings`, worded for this entry
/// point: a budget-starved integration says so, and the identical model with a budget that
/// finishes says nothing.
///
/// Regression this catches: no solver-stats scope around the survival pass (the counters
/// would come back zero and the message absent), or the pass reported under another
/// phase's wording.
#[test]
fn predict_survival_diag_reports_the_ode_hazards_solver_diagnostics() {
    let starved = joint_pk_tte(1e5, 20);
    let p = read(&starved, JOINT_CSV);
    let out = predict_survival_diag(&starved, &p, &starved.default_params, &GRID).unwrap();
    let solver = out
        .warnings
        .iter()
        .find(|w| w.contains("W_ODE_SOLVER_DIAGNOSTICS"))
        .unwrap_or_else(|| panic!("a starved ODE hazard must be named: {:?}", out.warnings));
    assert!(
        solver.contains("this predict_survival() pass"),
        "the message must name this pass, not predict()'s or a fit's: {solver}"
    );

    let clean = joint_pk_tte(1.0, 10_000);
    let p = read(&clean, JOINT_CSV);
    let out = predict_survival_diag(&clean, &p, &clean.default_params, &GRID).unwrap();
    assert!(
        out.results
            .iter()
            .all(|r| r.cum_hazard.is_finite() && r.cum_hazard > 0.0),
        "the clean twin must integrate a live hazard: {:?}",
        out.results
    );
    assert!(
        !has(&out.warnings, "W_ODE_SOLVER_DIAGNOSTICS"),
        "a clean integration must not carry a solver diagnostic: {:?}",
        out.warnings
    );
}

/// The `Err` path is unchanged: the wrapper and the `_diag` form refuse the same input with
/// the same message.
#[test]
fn predict_survival_and_its_diag_form_refuse_alike() {
    let m = pure_tte(2, "");
    let p = read(&m, &clean_csv(2));
    let mut short = m.default_params.clone();
    short.theta.pop();
    let a = predict_survival(&m, &p, &short, &GRID).unwrap_err();
    let b = predict_survival_diag(&m, &p, &short, &GRID).unwrap_err();
    assert_eq!(a.to_string(), b.to_string());
    assert_eq!(a.code(), b.code());
}

// ── predict_survival_at_etas_diag (#1885) ────────────────────────────────────

use super::{predict_survival_at_etas_diag, SuppliedEtas, SurvivalPredictionResult};
use crate::survival::{tte_data_term, tte_nll_from_curves, MonoTol};
use crate::types::{EndpointLikelihood, EventType, HazardSpec, ObsRecord, TteRecurrence};

/// Supplied η named `names`, one `(id, row)` per entry, in the order given.
fn supplied(names: &[&str], rows: &[(&str, &[f64])]) -> SuppliedEtas {
    SuppliedEtas::from_table(
        rows.iter().map(|(id, _)| id.to_string()).collect(),
        names.iter().map(|n| n.to_string()).collect(),
        rows.iter().map(|(_, r)| r.to_vec()).collect(),
    )
    .expect("table")
}

fn rows_of<'a>(
    rows: &'a [SurvivalPredictionResult],
    id: &str,
    cmt: usize,
) -> Vec<&'a SurvivalPredictionResult> {
    rows.iter().filter(|r| r.id == id && r.cmt == cmt).collect()
}

fn all_bits(r: &SurvivalPredictionResult) -> [u64; 8] {
    [
        r.time,
        r.survival,
        r.cum_hazard,
        r.hazard,
        r.cif,
        r.survival_all,
        r.median_survival,
        r.mean_survival,
    ]
    .map(f64::to_bits)
}

fn event(time: f64, event_type: EventType, cmt: usize) -> ObsRecord {
    ObsRecord::Event {
        time,
        event_type,
        entry_time: 0.0,
        cmt,
    }
}

fn hazard_of(m: &CompiledModel, cmt: usize) -> &HazardSpec {
    match m.endpoints.get(&cmt) {
        Some(EndpointLikelihood::Tte { hazard, .. }) => hazard,
        _ => panic!("no TTE endpoint on CMT {cmt}"),
    }
}

/// T1: at every grid time, an analytic hazard's individual `cum_hazard` is the
/// cumulative hazard the likelihood scores at that subject's η̂, bit for bit — the
/// right-censored term `H(T)` read straight off the row, and the exact term
/// `H(T) − ln h(T)` rebuilt from the row's `H` and `h` through the scorer's own
/// `tte_nll_from_curves`. The grid holds both subjects' record times (7, 30).
///
/// The η rows are supplied in the reverse of population order, so mutation (b), a
/// zip of the rows in supplied order instead of a lookup by ID, gives subject '1'
/// subject '2''s η and fails here; mutation (a), the zero η, fails on every row.
#[test]
fn individual_cum_hazard_is_the_scored_one_at_the_ebe() {
    let m = pure_tte(2, "");
    let p = read(&m, &clean_csv(2));
    let ebes = [("2", -0.5), ("1", 0.6)];
    let etas = supplied(&["ETA_SCALE"], &[("2", &[-0.5]), ("1", &[0.6])]);
    let grid = [1.0, 5.0, 7.0, 10.0, 30.0];
    let out = predict_survival_at_etas_diag(&m, &p, &m.default_params, &etas, &grid).unwrap();
    assert_eq!(out.results.len(), 2 * grid.len());
    let hazard = hazard_of(&m, 2);
    let theta = &m.default_params.theta;
    for (id, eta) in ebes {
        let cov = &p.subjects.iter().find(|s| s.id == id).unwrap().covariates;
        let rows = rows_of(&out.results, id, 2);
        assert_eq!(rows.len(), grid.len());
        for r in rows {
            let t = r.time;
            let rc = [event(t, EventType::RightCensored, 2)];
            let scored = tte_data_term(&rc, hazard, TteRecurrence::Single, theta, &[eta], cov);
            assert!(scored.is_finite() && scored < 1e19, "{scored}");
            assert_eq!(
                r.cum_hazard.to_bits(),
                scored.to_bits(),
                "subject {id}, t = {t}: H {} vs scored {scored}",
                r.cum_hazard
            );
            let ex = [event(t, EventType::Exact, 2)];
            let scored = tte_data_term(&ex, hazard, TteRecurrence::Single, theta, &[eta], cov);
            let at = |u: f64, v: f64| if u == t { v } else { f64::NAN };
            let rebuilt = tte_nll_from_curves(
                &ex,
                |u| at(u, r.cum_hazard),
                |u| at(u, r.hazard),
                MonoTol::analytic(),
            );
            assert!(scored.is_finite() && scored < 1e19, "{scored}");
            assert_eq!(
                rebuilt.to_bits(),
                scored.to_bits(),
                "subject {id}, t = {t}: exact term {rebuilt} vs scored {scored}"
            );
        }
    }
}

/// T4: a differential pair in one population. Subject '2' is supplied η = 0 exactly
/// and gets `predict_survival_diag`'s rows bit for bit; subject '1' is supplied 0.6 and
/// its `H` drops by more than half at every grid time (Weibull shape 1.5:
/// `exp(−0.9) = 0.41`). Supplied out of population order.
///
/// Mutation (a), the zero η: '1' stops moving. Mutation (d), '1''s η applied to every
/// subject: '2' moves.
#[test]
fn a_zero_ebe_is_the_population_curve_and_a_live_one_moves() {
    let m = pure_tte(2, "");
    let p = read(&m, &clean_csv(2));
    let (eta_a, eta_b) = (0.6_f64, 0.0_f64);
    assert!(eta_a != 0.0 && eta_b == 0.0, "the pair must straddle η = 0");
    let etas = supplied(&["ETA_SCALE"], &[("2", &[eta_b]), ("1", &[eta_a])]);
    let ind = predict_survival_at_etas_diag(&m, &p, &m.default_params, &etas, &GRID).unwrap();
    let pop = predict_survival_diag(&m, &p, &m.default_params, &GRID).unwrap();

    let (b_ind, b_pop) = (rows_of(&ind.results, "2", 2), rows_of(&pop.results, "2", 2));
    assert_eq!(b_ind.len(), GRID.len());
    for (i, q) in b_ind.iter().zip(&b_pop) {
        assert_eq!(all_bits(i), all_bits(q), "η = 0 must be the population row");
    }
    let (a_ind, a_pop) = (rows_of(&ind.results, "1", 2), rows_of(&pop.results, "1", 2));
    assert_eq!(a_ind.len(), GRID.len());
    for (i, q) in a_ind.iter().zip(&a_pop) {
        assert!(i.time > 0.0 && i.cum_hazard.is_finite() && q.cum_hazard > 0.0);
        let rel = (q.cum_hazard - i.cum_hazard) / q.cum_hazard;
        assert!(rel >= 0.5, "t = {}: H moved by {rel} only", i.time);
    }
}

/// Two-cause competing risks with the η on cause A only.
const COMPETING: &str = "[parameters]\n  theta TVLAMBDA_A(0.10, 0.001, 10.0)\n  \
    theta TVLAMBDA_B(0.06, 0.001, 10.0)\n  omega ETA_F ~ 0.25\n\
    [event_model cause_a]\n  cmt = 2\n  family = exponential\n  scale = TVLAMBDA_A * exp(ETA_F)\n\
    [event_model cause_b]\n  cmt = 3\n  family = exponential\n  scale = TVLAMBDA_B\n";

/// T5: competing risks at η̂. Cause B's hazard reads no η, so its `H` stays the
/// population one bit for bit, yet its CIF moves — through `S_all`, which cause A's
/// η̂ lowers. Cause A's CIF moves too, and `Σ F_k + S_all = 1` holds at η̂.
///
/// Mutation: build the CIF from the η = 0 curves and neither CIF moves.
#[test]
fn competing_risks_cif_is_built_at_the_ebe() {
    let m = parse_model_string(COMPETING).expect("parse");
    let csv = "ID,TIME,DV,EVID,AMT,CMT,MDV\n1,4,1,0,0,2,0\n2,9,1,0,0,3,0\n";
    let p = read(&m, csv);
    let etas = supplied(&["ETA_F"], &[("2", &[0.0]), ("1", &[0.8])]);
    let ind = predict_survival_at_etas_diag(&m, &p, &m.default_params, &etas, &GRID).unwrap();
    let pop = predict_survival_diag(&m, &p, &m.default_params, &GRID).unwrap();
    let (a_i, a_p) = (rows_of(&ind.results, "1", 2), rows_of(&pop.results, "1", 2));
    let (b_i, b_p) = (rows_of(&ind.results, "1", 3), rows_of(&pop.results, "1", 3));
    assert_eq!((a_i.len(), b_i.len()), (GRID.len(), GRID.len()));
    for k in 0..GRID.len() {
        assert_eq!(
            b_i[k].cum_hazard.to_bits(),
            b_p[k].cum_hazard.to_bits(),
            "cause B reads no η: its H must not move"
        );
        assert!(
            a_i[k].cum_hazard > 2.0 * a_p[k].cum_hazard,
            "cause A must move"
        );
        assert!(a_i[k].cif > a_p[k].cif, "cause A's CIF must rise");
        assert!(
            b_i[k].cif < b_p[k].cif,
            "cause B's CIF must fall through S_all: {} vs {}",
            b_i[k].cif,
            b_p[k].cif
        );
        let total = a_i[k].cif + b_i[k].cif + a_i[k].survival_all;
        assert!(total.is_finite() && (total - 1.0).abs() <= 1e-15, "{total}");
    }
}

/// The `tests/tte_smoke.rs` "IOV + TTE subjects" model: a κ on CL, the hazard on ETA_CL.
const IOV_TTE: &str =
    "[parameters]\n  theta TVCL(1.0, 0.1, 10.0)\n  theta TVV(10.0, 1.0, 100.0)\n  \
    theta TVLAMBDA(0.05, 0.001, 5.0)\n  omega ETA_CL ~ 0.09\n  kappa KAPPA_CL ~ 0.04\n  \
    sigma SIGMA_ADD ~ 0.1\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  \
    V  = TVV\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  \
    DV ~ additive(SIGMA_ADD)\n[event_model]\n  cmt    = 2\n  family = exponential\n  \
    scale  = TVLAMBDA * exp(ETA_CL)\n";

/// T6: the κ = 0 padding is exact because no hazard reads κ. The hazard's parameters
/// at κ = 0 and at κ = 1 are bit-identical, so the rows built from them are too.
///
/// Regression this catches: a parser change that lets a hazard read a κ, which would
/// make the padding an approximation.
#[test]
fn kappa_padding_is_exact_because_no_hazard_reads_kappa() {
    let m = parse_model_string(IOV_TTE).expect("parse");
    assert_eq!(
        (m.n_eta, m.n_kappa),
        (1, 1),
        "fixture carries one η and one κ"
    );
    let endpoint = m.endpoints.get(&2).expect("endpoint");
    let at = |eta: f64, kappa: f64| {
        crate::survival::tte_cause_params(
            endpoint,
            &m.default_params.theta,
            &[eta, kappa],
            &Default::default(),
        )
        .expect("analytic cause")
        .1
        .iter()
        .map(|v| v.to_bits())
        .collect::<Vec<_>>()
    };
    assert_eq!(at(0.3, 0.0), at(0.3, 1.0), "κ must not reach the hazard");
    // …and the η slot it does read is live, so the identity above is not vacuous.
    assert_ne!(at(0.3, 0.0), at(0.0, 0.0), "the hazard must read η");
}

/// T6, the entry itself on an IOV model (#2488 review, finding 2): the call succeeds
/// with the κ slots padded, and each subject's `H` is the closed form `λ·exp(η̂)·t` at its
/// own η̂, so it moves off the η = 0 curve.
///
/// Regression this catches: the entry refusing or panicking on an IOV + TTE population.
/// The κ padding itself stays an equivalent mutant while no hazard can read κ (above).
#[test]
fn ebe_survival_runs_on_an_iov_model() {
    let m = parse_model_string(IOV_TTE).expect("parse");
    let csv = "ID,TIME,DV,EVID,AMT,CMT,MDV,OCC\n1,0,.,1,100,1,1,1\n1,6,1,0,.,2,0,1\n\
               2,0,.,1,100,1,1,1\n2,20,0,0,.,2,0,1\n";
    let p = read(&m, csv);
    let etas = supplied(&["ETA_CL"], &[("2", &[-0.4]), ("1", &[0.5])]);
    let out = predict_survival_at_etas_diag(&m, &p, &m.default_params, &etas, &GRID)
        .expect("an IOV + TTE population is accepted");
    let lambda = m.default_params.theta[2];
    for (id, eta) in [("1", 0.5_f64), ("2", -0.4)] {
        let rows = rows_of(&out.results, id, 2);
        assert_eq!(rows.len(), GRID.len());
        for r in rows {
            let want = lambda * eta.exp() * r.time;
            let zero = lambda * r.time;
            assert!(r.cum_hazard.is_finite(), "{r:?}");
            assert!(
                ((r.cum_hazard - want) / want).abs() <= 4.0 * f64::EPSILON,
                "subject {id} t = {}: H {} vs λ·exp(η̂)·t {want}",
                r.time,
                r.cum_hazard
            );
            assert!((r.cum_hazard - zero).abs() > 0.3 * zero, "η̂ must move H");
        }
    }
}

fn refused(r: Result<super::SurvivalPredictionOutput, crate::diagnostics::EngineError>) -> String {
    let e = r.expect_err("refused");
    assert_eq!(e.code(), Some("E_SUPPLIED_ETAS"), "{e}");
    e.to_string()
}

/// T7: every `E_SUPPLIED_ETAS` refusal of this entry, by its full text, so deleting any
/// sentence of a message fails here, with the accepted side of each gate beside it.
#[test]
fn supplied_eta_survival_refusals_carry_their_text() {
    let m = pure_tte(2, "");
    let p = read(&m, &clean_csv(2));
    let run = |m: &CompiledModel, e: &SuppliedEtas| {
        predict_survival_at_etas_diag(m, &p, &m.default_params, e, &GRID)
    };

    // η names: the set…
    let renamed = supplied(&["ETA_V"], &[("1", &[0.1]), ("2", &[0.1])]);
    assert_eq!(
        refused(run(&m, &renamed)),
        "the supplied η are named [ETA_V] but the model's are [ETA_SCALE]; supply every η \
         of the model, in the model's order"
    );
    // …and the order.
    let two = pure_tte(2, "  omega ETA_X ~ 0.04\n");
    assert_eq!(two.eta_names, ["ETA_X", "ETA_SCALE"]);
    let swapped = supplied(
        &["ETA_SCALE", "ETA_X"],
        &[("1", &[0.1, 0.0]), ("2", &[0.1, 0.0])],
    );
    assert_eq!(
        refused(run(&two, &swapped)),
        "the supplied η are named [ETA_SCALE, ETA_X] but the model's are [ETA_X, ETA_SCALE]; \
         supply every η of the model, in the model's order"
    );
    let ordered = supplied(
        &["ETA_X", "ETA_SCALE"],
        &[("1", &[0.0, 0.1]), ("2", &[0.0, 0.1])],
    );
    assert!(run(&two, &ordered).is_ok(), "the model's order is accepted");

    // A population subject with no row is refused; an extra supplied ID ('9') is not.
    let only_2 = supplied(&["ETA_SCALE"], &[("2", &[0.1]), ("9", &[0.0])]);
    assert_eq!(
        refused(run(&m, &only_2)),
        "1 subject(s) in the population have no supplied η: '1'. An individual curve needs \
         each subject's η: supply theirs, drop them from the population, or call \
         `predict_survival_diag` for the population (η = 0) curve"
    );
    let full = supplied(
        &["ETA_SCALE"],
        &[("2", &[0.1]), ("9", &[0.0]), ("1", &[0.2])],
    );
    assert!(run(&m, &full).is_ok(), "an extra supplied ID is ignored");

    // The predict_survival_diag checks run first: a short θ is E_THETA_LENGTH even with
    // wrongly named η.
    let mut short = m.default_params.clone();
    short.theta.pop();
    let first = predict_survival_at_etas_diag(&m, &p, &short, &renamed, &GRID).unwrap_err();
    let diag = predict_survival_diag(&m, &p, &short, &GRID).unwrap_err();
    assert_eq!(
        (first.code(), first.to_string()),
        (diag.code(), diag.to_string())
    );
}

/// T7, the `[mixture]` cell: refused with this entry's wording, and the same η on the
/// class-free twin is accepted.
#[test]
fn supplied_eta_survival_on_a_mixture_model_is_refused() {
    let mix = parse_model_string(
        "[parameters]\n  theta TVSCALE(20.0, 0.1, 500.0)\n  theta TVSHAPE(1.5, 0.1, 10.0)\n  \
         theta MIXL(0.0, -5.0, 5.0)\n  omega ETA_SCALE ~ 0.04\n[mixture]\n  nsub = 2\n  \
         logit(1) = MIXL\n[event_model]\n  cmt = 2\n  family = weibull\n  \
         scale = TVSCALE * exp(ETA_SCALE)\n  shape = TVSHAPE\n",
    )
    .expect("parse mixture");
    assert!(mix.mixture.is_some(), "the fixture is a mixture model");
    let p = read(&mix, &clean_csv(2));
    let e = supplied(&["ETA_SCALE"], &[("1", &[0.1]), ("2", &[0.1])]);
    assert_eq!(
        refused(predict_survival_at_etas_diag(
            &mix,
            &p,
            &mix.default_params,
            &e,
            &GRID
        )),
        "supplied η are not supported on a [mixture] model: each subject's η was estimated \
         under its own mixture class, and `predict_survival_at_etas_diag` has no class to \
         evaluate it under"
    );
    let plain = pure_tte(2, "");
    assert!(predict_survival_at_etas_diag(&plain, &p, &plain.default_params, &e, &GRID).is_ok());
}

/// T7, the `n_eta = 0` cell: allowed, and the curve is the population one bit for bit.
#[test]
fn supplied_eta_survival_without_etas_is_the_population_curve() {
    let m = parse_model_string(
        "[parameters]\n  theta TVSCALE(20.0, 0.1, 500.0)\n  theta TVSHAPE(1.5, 0.1, 10.0)\n\
         [event_model]\n  cmt = 2\n  family = weibull\n  scale = TVSCALE\n  shape = TVSHAPE\n",
    )
    .expect("parse");
    assert_eq!(m.n_eta, 0);
    let p = read(&m, &clean_csv(2));
    let e = supplied(&[], &[("1", &[]), ("2", &[])]);
    let ind = predict_survival_at_etas_diag(&m, &p, &m.default_params, &e, &GRID).unwrap();
    let pop = predict_survival_diag(&m, &p, &m.default_params, &GRID).unwrap();
    assert_eq!(ind.results.len(), pop.results.len());
    assert!(!ind.results.is_empty());
    for (i, q) in ind.results.iter().zip(&pop.results) {
        assert_eq!(all_bits(i), all_bits(q));
    }
}
