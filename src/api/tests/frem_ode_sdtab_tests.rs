//! sdtab IPRED / IWRES of an ODE FREM fit on the `FREMTYPE > 0` covariate rows (#2259).
//!
//! The post-fit diagnostics take a non-IOV subject's IPRED from
//! `pk::compute_predictions_with_states`, whose ODE branch ran `[scaling]` and the LTBS
//! log-wrap but not the FREM override, so a covariate row reported the ODE readout
//! instead of `θ_k + η_m`. The fit's objective was unaffected (it scores through
//! `compute_predictions_with_tv`). Validation: the state-returning predictor pinned bit
//! for bit against the full `pk` sink on every row, on both ODE routes, plus a closed
//! form (the FREM row is exactly `TV_WT + η_WT`) through `compute_subject_results`.
use super::postfit::compute_subject_results;
use super::*;
use crate::parser::model_parser::parse_model_string;
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

/// 1-cpt ODE, one FREM covariate `WT` (`TV_WT` + `ETA_WT` predict `FREMTYPE = 100`).
fn model(readout: &str) -> CompiledModel {
    let src = format!(
        r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TV_WT(70.0) FIX
  omega ETA_V ~ 0.09
  omega ETA_WT ~ 100.0
  sigma S ~ 0.04
  sigma EPSCOV ~ 0.0001 FIX
[individual_parameters]
  CL = TVCL
  V  = TVV * exp(ETA_V)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
{readout}
[error_model]
  DV ~ proportional(S)
[fit_options]
  frem_predictions = TV_WT/ETA_WT:100
  frem_sigma = EPSCOV
"#
    );
    let m = parse_model_string(&src).expect("parse");
    assert!(
        m.frem_config.is_some() && m.ode_spec.is_some(),
        "ODE FREM fixture"
    );
    m
}

/// Form A (a divisor `[scaling]` applies) and Form C (the readout runs in the engine).
const FORMS: [(&str, &str); 2] = [
    ("A: obs_scale = V", "  obs_scale = V"),
    ("C: y = central / V", "  y = central / V"),
];

const OBS: [f64; 5] = [0.5, 1.0, 4.0, 14.0, 20.0];
const FREMTYPE: [u16; 5] = [100, 0, 0, 100, 0];
const FREM_ROWS: [usize; 2] = [0, 3];
const DV: [f64; 5] = [80.0, 1.6, 1.2, 80.0, 0.3];
const ETA: [f64; 2] = [0.2, 11.5];

/// With `tv`, the subject carries per-record covariate snapshots, which routes
/// `compute_predictions_with_states` onto the event-driven ODE walk instead of the
/// single-pass one.
fn subject(tv: bool) -> Subject {
    let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    let n = OBS.len();
    let snapshot = |k: usize| vec![HashMap::from([("X".to_string(), 1.0)]); k];
    Subject {
        id: "1".to_string(),
        dose_covariates: if tv {
            snapshot(doses.len())
        } else {
            Vec::new()
        },
        obs_covariates: if tv { snapshot(n) } else { Vec::new() },
        doses,
        obs_times: OBS.to_vec(),
        observations: DV.to_vec(),
        obs_cmts: vec![1; n],
        cens: vec![0; n],
        fremtype: FREMTYPE.to_vec(),
        ..Default::default()
    }
}

/// The FREM rows sit two orders above the concentration, so a row left on the readout
/// cannot pass a `θ + η` assertion by accident.
fn assert_frem_rows_are_theta_plus_eta(m: &CompiledModel, rows: &[f64], label: &str) {
    let want = m.default_params.theta[2] + ETA[1];
    assert_eq!(want, 81.5, "TV_WT is 70");
    for j in FREM_ROWS {
        assert_eq!(rows[j], want, "{label}: FREM row {j}");
    }
    for (j, r) in rows.iter().enumerate() {
        if FREMTYPE[j] == 0 {
            assert!(
                r.is_finite() && *r > 0.0 && *r < 20.0,
                "{label}: PK row {j} = {r} is not a concentration"
            );
        }
    }
}

/// `compute_predictions_with_states`'s IPRED against the full `pk` sink
/// (`compute_predictions_with_tv`), bit for bit on every row, FREM rows included, on
/// both ODE routes (single-pass and event-driven) and under Form A and C.
///
/// Regression: drop the FREM step from the ODE branch of
/// `compute_predictions_with_states` → FREM row 0 is the readout (Form C: 1.5718
/// against 81.5, #2259) on both routes.
#[test]
fn ode_states_ipred_matches_the_full_pk_sink_on_frem_rows() {
    for (label, readout) in FORMS {
        let m = model(readout);
        let theta = &m.default_params.theta;
        for tv in [false, true] {
            let label = format!("{label}, event-driven {tv}");
            let s = subject(tv);
            assert_eq!(s.has_tv_covariates(), tv, "{label}: route straddle");
            let (got, states) = crate::pk::compute_predictions_with_states(&m, &s, theta, &ETA);
            assert_eq!(states.len(), OBS.len(), "{label}: one state per row");
            let want = crate::pk::compute_predictions_with_tv(&m, &s, theta, &ETA);
            assert_frem_rows_are_theta_plus_eta(&m, &want, &label);
            assert_eq!(got.len(), want.len(), "{label}: row count");
            for (j, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "{label}: row {j}: {g} vs sink {w}"
                );
            }
        }
    }
}

/// End to end through the post-fit diagnostics: the sdtab IPRED of a FREM row is
/// `TV_WT + η̂_WT`, and its IWRES is a residual against that IPRED. `DV = 80` sits
/// *below* `θ + η = 81.5` and far *above* the readout (~1.6), so the IWRES sign tells
/// the two apart whatever the row's residual SD is. (The SD IWRES divides a FREM row
/// by is the PK error model's, not `frem_sigma`'s; that is outside #2259 and is
/// deliberately not pinned here.)
///
/// Regression: the same deletion → row 0 IPRED is the readout, and (with the IPRED
/// assertion disabled) its IWRES is +249.49 under Form A (measured).
#[test]
fn postfit_ipred_and_iwres_of_an_ode_frem_row() {
    for (label, readout) in FORMS {
        let m = model(readout);
        let pop = Population {
            subjects: vec![subject(false)],
            covariate_names: Vec::new(),
            dv_column: "DV".to_string(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        };
        let eta = DVector::from_vec(ETA.to_vec());
        let h = DMatrix::zeros(OBS.len(), ETA.len());
        let (mut subjects, _) = compute_subject_results(
            &m,
            &pop,
            &m.default_params,
            std::slice::from_ref(&eta),
            std::slice::from_ref(&h),
            &[Vec::new()],
            true,
            None,
            false,
        );
        let r = subjects.pop().expect("one subject");
        assert_frem_rows_are_theta_plus_eta(&m, &r.ipred, label);
        for j in FREM_ROWS {
            assert!(
                r.iwres[j].is_finite() && r.iwres[j] < 0.0,
                "{label}: FREM row {j} IWRES {} is not a residual against θ + η",
                r.iwres[j]
            );
        }
    }
}
