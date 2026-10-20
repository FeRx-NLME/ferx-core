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
