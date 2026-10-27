//! A subject's first record of any type starts the ODE clock (#1809).
//!
//! NONMEM sets the system to its initial state at the subject's first event record,
//! whatever its type, and advances it to every later record. Before #1809 the reader
//! dropped an `EVID=2` row on a subject whose covariates were all constant, and an
//! `EVID=0, MDV=1` row on every subject. So `subject_integration_start` saw neither,
//! and the integration started at the first observation or dose. On any ODE model
//! whose state is not stationary over `[first record, first observation]`, every
//! prediction was wrong.
//!
//! These tests run the reader and the public `predict()` end to end, against the
//! **closed form** of the ODE. They do not compare against a second engine: the
//! dense and event-driven predictors, the analytic sensitivity walk and its FD
//! reference all read the same `Subject`, so a parity test agrees with the wrong
//! clock by construction. The values are those NONMEM 7.6.0 printed in the #1809
//! triage (`MAXEVAL=0`, `PRED` at η = 0), and they agree with the closed forms below
//! to the 4 printed decimals.

use super::predict;
use crate::io::datareader::read_nonmem_csv;
use crate::parser::model_parser::parse_model_string;
use crate::types::CompiledModel;
use std::io::Write;

/// Solver tolerances far below the closed-form comparison. At the defaults
/// (`ode_reltol = 1e-4`, `ode_abstol = 1e-6`) the dense route sits 1.2e-4 from the
/// closed form, so the oracle would measure the solver, not the clock.
const ODE_TOL: &str = "[fit_options]\n  ode_reltol = 1e-10\n  ode_abstol = 1e-12\n";

/// The #1809 model: one state, a zero initial state (no `init()`), no doses needed,
/// `TIME` in the RHS, and exposure read from a data column. `resp` stays positive, so
/// `clamp_negative_predictions` cannot hide a difference.
fn time_rhs_model(init: Option<&str>) -> CompiledModel {
    let init_line = init
        .map(|e| format!("  init(resp) = {e}\n"))
        .unwrap_or_default();
    let src = format!(
        "[parameters]\n  theta TVKOUT(0.25, 0.01, 5.0)\n  theta TVPLB(6.0, 0.1, 50.0)\n  \
         theta TVTF(0.15, 0.001, 2.0)\n  theta TVEMAX(8.0, 0.1, 50.0)\n  \
         theta TVEC50(2.0, 0.1, 50.0)\n  omega ETA_PLB ~ 0.04\n  sigma ADD ~ 0.5 (sd)\n\
         [individual_parameters]\n  KOUT = TVKOUT\n  PLB = TVPLB * exp(ETA_PLB)\n  \
         TF = TVTF\n  EFF = TVEMAX * EXPO / (TVEC50 + EXPO)\n\
         [structural_model]\n  ode(obs_cmt=resp, states=[resp])\n\
         [odes]\n{init_line}  d/dt(resp) = KOUT * ((PLB * exp(-TF * TIME) + EFF) - resp)\n\
         [error_model]\n  DV ~ additive(ADD)\n{ODE_TOL}"
    );
    parse_model_string(&src).expect("parse the #1809 TIME-in-RHS model")
}

/// The same model with a constant production term instead of `PLB * exp(-TF * TIME)`,
/// so the RHS does not read `TIME` and a constant-covariate subject takes the **dense**
/// `ode_predictions` route instead of the event-driven walk.
fn production_model() -> CompiledModel {
    let src = format!(
        "[parameters]\n  theta TVKOUT(0.25, 0.01, 5.0)\n  theta TVPLB(6.0, 0.1, 50.0)\n  \
         theta TVEMAX(8.0, 0.1, 50.0)\n  theta TVEC50(2.0, 0.1, 50.0)\n  \
         omega ETA_PLB ~ 0.04\n  sigma ADD ~ 0.5 (sd)\n\
         [individual_parameters]\n  KOUT = TVKOUT\n  PLB = TVPLB * exp(ETA_PLB)\n  \
         EFF = TVEMAX * EXPO / (TVEC50 + EXPO)\n\
         [structural_model]\n  ode(obs_cmt=resp, states=[resp])\n\
         [odes]\n  d/dt(resp) = KOUT * ((PLB + EFF) - resp)\n\
         [error_model]\n  DV ~ additive(ADD)\n{ODE_TOL}"
    );
    parse_model_string(&src).expect("parse the #1809 production-term model")
}

/// `PRED` (η = 0) for each subject of `csv`, in the reader's subject order.
fn preds(model: &CompiledModel, csv: &str) -> Vec<Vec<f64>> {
    preds_with(model, csv, false)
}

/// As [`preds`], optionally on the population `fit()` sees: `fitted_population` prunes
/// the snapshots of every covariate the model does not reference, and `predict()` does
/// not.
fn preds_with(model: &CompiledModel, csv: &str, prune: bool) -> Vec<Vec<f64>> {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(csv.as_bytes()).unwrap();
    let mut pop = read_nonmem_csv(f.path(), None, None).expect("read the fixture");
    if prune {
        pop.prune_irrelevant_tv_covariates(&model.referenced_covariates);
    }
    let rows = predict(model, &pop, &model.default_params).expect("predict");
    pop.subjects
        .iter()
        .map(|s| {
            rows.iter()
                .filter(|r| r.id == s.id)
                .map(|r| r.pred)
                .collect()
        })
        .collect()
}

/// Exact solution of `dy/dt = k (P e^{-a t} + E - y)` from `y(0) = y0`, with the clock
/// at `t = 0` (the first record of every fixture below).
fn time_rhs_closed_form(t: f64, y0: f64) -> f64 {
    let (k, a, p, e) = (0.25, 0.15, 6.0, 8.0 * 3.0 / (2.0 + 3.0));
    y0 * (-k * t).exp()
        + e * (1.0 - (-k * t).exp())
        + p * k / (k - a) * ((-a * t).exp() - (-k * t).exp())
}

/// Every value must be finite, then within `tol` of the closed form, with the worst
/// error named in the message so the bound stays a measured one.
fn assert_matches(label: &str, got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{label}: row count");
    assert!(
        got.iter().all(|v| v.is_finite()),
        "{label}: non-finite PRED {got:?}"
    );
    let worst = got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0, f64::max);
    assert!(
        worst < tol,
        "{label}: PRED {got:?} vs closed form {want:?} (worst |error| {worst:.3e}, tolerance {tol:.0e})"
    );
}

// Measured worst |PRED - closed form| at `ODE_TOL`, Linux x86_64: 1.24e-10 (the dense
// route, the largest), 9.3e-11 on the event-driven fixtures. The unpruned-vs-pruned
// comparison in the invariance test is 4.1e-12 at worst (dense vs event-driven). 1e-8 is
// ~80x the largest and nine orders below every defect these tests exist to catch: the
// dropped window puts the first observation at the initial state, an error of 3.9 here,
// and the smallest gap of #1809's sweep (0.05) is still 0.13.
const TOL: f64 = 1e-8;

#[test]
fn a_static_evid2_first_record_starts_the_event_driven_clock() {
    // #1809 case 1. Subject 1 has every column constant (the static reader path),
    // subject 2 has `EXPO` 0 on the EVID=2 row and 3 afterwards (the per-record path).
    // Both start the clock at TIME 0. Before #1809, subject 1 read 0 at its first
    // observation (0.0000 / 3.3809 / 5.7674 / 5.9733).
    let model = time_rhs_model(None);
    assert!(
        crate::pk::model_uses_time_anywhere(&model),
        "fixture must read TIME, so a static subject takes the event-driven walk"
    );
    let csv = "ID,TIME,DV,EVID,MDV,AMT,CMT,EXPO\n\
               1,0,.,2,1,0,1,3\n1,2,1,0,0,0,1,3\n1,4,1,0,0,0,1,3\n1,8,1,0,0,0,1,3\n1,12,1,0,0,0,1,3\n\
               2,0,.,2,1,0,1,0\n2,2,1,0,0,0,1,3\n2,4,1,0,0,0,1,3\n2,8,1,0,0,0,1,3\n2,12,1,0,0,0,1,3\n";
    let p = preds(&model, csv);
    let want: Vec<f64> = [2.0_f64, 4.0, 8.0, 12.0]
        .iter()
        .map(|&t| time_rhs_closed_form(t, 0.0))
        .collect();
    // NONMEM 7.6.0 (#1809 triage): 3.9030 / 5.7482 / 6.6383 for both subjects.
    assert_matches("static EVID=2 subject", &p[0], &want, TOL);
    assert_matches("per-record EVID=2 subject", &p[1], &want, TOL);
}

#[test]
fn a_static_evid2_first_record_starts_the_dense_clock() {
    // #1809 case 5: the RHS does not read TIME, so subject 1 (all columns constant)
    // takes the dense `ode_predictions` route, and subject 2 (EXPO varying) the
    // event-driven one. The closed form is `10.8 (1 - e^{-0.25 t})`, with
    // 10.8 = 6 + 8·3/5. Before #1809, subject 1 read 0.0000 / 4.2495 / 8.3901.
    let model = production_model();
    assert!(!crate::pk::model_uses_time_anywhere(&model));
    let csv = "ID,TIME,DV,EVID,MDV,AMT,CMT,EXPO\n\
               1,0,.,2,1,0,1,3\n1,2,1,0,0,0,1,3\n1,4,1,0,0,0,1,3\n1,8,1,0,0,0,1,3\n\
               2,0,.,2,1,0,1,0\n2,2,1,0,0,0,1,3\n2,4,1,0,0,0,1,3\n2,8,1,0,0,0,1,3\n";
    let p = preds(&model, csv);
    let want: Vec<f64> = [2.0_f64, 4.0, 8.0]
        .iter()
        .map(|&t| 10.8 * (1.0 - (-0.25 * t).exp()))
        .collect();
    // NONMEM 7.6.0 (#1809 triage): 4.2495 / 6.8269 / 9.3384.
    assert_matches("static subject, dense route", &p[0], &want, TOL);
    assert_matches("per-record subject, event-driven route", &p[1], &want, TOL);
}

#[test]
fn an_evid0_mdv1_first_record_starts_the_clock_on_both_paths() {
    // #1809 option A2, measured in the triage: NONMEM 7.6.0 starts the clock at an
    // EVID=0/MDV=1 record (subjects 3 and 4). Before #1809 the reader dropped it on both
    // paths, so subject 2 (EXPO varying) was wrong too: both read 0 at the first
    // observation.
    let model = time_rhs_model(None);
    let csv = "ID,TIME,DV,EVID,MDV,AMT,CMT,EXPO\n\
               1,0,.,0,1,0,1,3\n1,2,1,0,0,0,1,3\n1,4,1,0,0,0,1,3\n1,8,1,0,0,0,1,3\n\
               2,0,.,0,1,0,1,0\n2,2,1,0,0,0,1,3\n2,4,1,0,0,0,1,3\n2,8,1,0,0,0,1,3\n";
    let p = preds(&model, csv);
    let want: Vec<f64> = [2.0_f64, 4.0, 8.0]
        .iter()
        .map(|&t| time_rhs_closed_form(t, 0.0))
        .collect();
    assert_matches("static EVID=0/MDV=1 subject", &p[0], &want, TOL);
    assert_matches("per-record EVID=0/MDV=1 subject", &p[1], &want, TOL);
}

#[test]
fn a_later_dose_does_not_repair_the_clock() {
    // #1809 case 4 (triage subject 5): EVID=2 at 0, a bolus of 2 into `resp` at 2,
    // observations at 4 and 8. The state at the dose is already wrong when the clock
    // starts at the dose, so the dose cannot repair it. Before #1809 the static subject
    // read 4.5939 / 6.2136; NONMEM 7.6.0 gives 6.9612 / 7.0845.
    //
    // Closed form: linear, so the bolus adds `2 e^{-k (t - 2)}` to the dose-free curve.
    let model = time_rhs_model(None);
    let csv = "ID,TIME,DV,EVID,MDV,AMT,CMT,EXPO\n\
               1,0,.,2,1,0,1,3\n1,2,.,1,1,2,1,3\n1,4,1,0,0,0,1,3\n1,8,1,0,0,0,1,3\n";
    let p = preds(&model, csv);
    let want: Vec<f64> = [4.0_f64, 8.0]
        .iter()
        .map(|&t| time_rhs_closed_form(t, 0.0) + 2.0 * (-0.25 * (t - 2.0)).exp())
        .collect();
    assert_matches("static subject with a later dose", &p[0], &want, TOL);
}

#[test]
fn an_init_baseline_is_seeded_at_the_first_record() {
    // #1809 "init" row: `init(resp) = 5` is not an equilibrium of the dynamics, so seeding
    // it at the first observation instead of the first record is wrong too. Before #1809
    // the static subject read exactly 5 at its first observation (5.0000 / 6.4135 / 6.8830).
    let model = time_rhs_model(Some("5.0"));
    let csv = "ID,TIME,DV,EVID,MDV,AMT,CMT,EXPO\n\
               1,0,.,2,1,0,1,3\n1,2,1,0,0,0,1,3\n1,4,1,0,0,0,1,3\n1,8,1,0,0,0,1,3\n";
    let p = preds(&model, csv);
    let want: Vec<f64> = [2.0_f64, 4.0, 8.0]
        .iter()
        .map(|&t| time_rhs_closed_form(t, 5.0))
        .collect();
    assert_matches("static subject with init()", &p[0], &want, TOL);
}

#[test]
fn an_unreferenced_varying_column_does_not_change_any_prediction() {
    // #1809 consequence 2: before the fix, whether the clock started at the first record
    // depended on whether *some* column varied within the subject. Auto-detect counts
    // every non-standard column, including one no expression reads (a `DAY` counter), so
    // adding such a column moved the clock. The fix restores the invariant that a column
    // the model does not read cannot change a prediction. It is asserted across both ODE
    // routes, a dose-bearing subject, an `init()` baseline and an analytical model (which
    // has no state before its first dose, so it never moved), at two strengths:
    //
    // - Bit for bit on the population `fit()` sees. Pruning clears the unreferenced DAY
    //   snapshots, so the two populations differ in nothing a predictor reads. Before
    //   #1809 they differed in the clock: subject 1's first PRED was 0 without DAY and
    //   3.9030 with it.
    // - Within `TOL` on the raw `predict()` population. There DAY is a time-varying
    //   column, so the subject legitimately changes route (event-driven instead of dense
    //   or superposition), and two integrators agree only to their tolerances.
    let base = "ID,TIME,DV,EVID,MDV,AMT,CMT,EXPO\n\
                1,0,.,2,1,0,1,3\n1,2,1,0,0,0,1,3\n1,4,1,0,0,0,1,3\n1,8,1,0,0,0,1,3\n\
                2,0,.,0,1,0,1,3\n2,2,.,1,1,2,1,3\n2,4,1,0,0,0,1,3\n2,8,1,0,0,0,1,3\n";
    let with_day = "ID,TIME,DV,EVID,MDV,AMT,CMT,EXPO,DAY\n\
                    1,0,.,2,1,0,1,3,1\n1,2,1,0,0,0,1,3,1\n1,4,1,0,0,0,1,3,1\n1,8,1,0,0,0,1,3,2\n\
                    2,0,.,0,1,0,1,3,1\n2,2,.,1,1,2,1,3,1\n2,4,1,0,0,0,1,3,1\n2,8,1,0,0,0,1,3,2\n";
    let analytical = parse_model_string(
        "[parameters]\n  theta TVCL(2.0, 0.01, 50.0)\n  theta TVV(10.0, 0.1, 500.0)\n  \
         omega ETA_CL ~ 0.04\n  sigma PROP ~ 0.1\n[individual_parameters]\n  \
         CL = TVCL * exp(ETA_CL) * EXPO / 3\n  V = TVV\n[structural_model]\n  \
         pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(PROP)\n",
    )
    .expect("parse the analytical control");
    for (label, model) in [
        ("event-driven ODE", time_rhs_model(None)),
        ("event-driven ODE with init()", time_rhs_model(Some("5.0"))),
        ("dense ODE", production_model()),
        ("analytical", analytical),
    ] {
        let a = preds_with(&model, base, true);
        let b = preds_with(&model, with_day, true);
        assert!(
            a.iter().flatten().all(|v| v.is_finite()),
            "{label}: non-finite PRED {a:?}"
        );
        assert_eq!(
            a, b,
            "{label}: an unreferenced, within-subject-varying DAY column changed the \
             predictions of the fitted population"
        );
        let raw = preds(&model, with_day);
        for (s, (want, got)) in a.iter().zip(&raw).enumerate() {
            assert_matches(
                &format!("{label}, subject {}, unpruned", s + 1),
                got,
                want,
                TOL,
            );
        }
    }
}
