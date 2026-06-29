//! Regression tests for off-zero TIME origins (#573).
//!
//! TIME stays on the raw data clock everywhere (model `TIME`/`T` builtin, sdtab,
//! predict/simulate). The off-zero start is handled purely inside the ODE
//! drivers, which begin integration at the subject's first event (NONMEM
//! semantics) instead of at a fixed `t = 0` — so a dataset whose TIME column
//! starts off-zero is *not* integrated over a phantom `[0, first_record]`
//! window, and is *not* silently re-based onto an elapsed clock.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{predict, read_nonmem_csv, Population};
use std::io::Write;
use tempfile::NamedTempFile;

const TIME_DEPENDENT_ODE: &str = r"
[parameters]
  theta TVRIN(1.0, 0.0, 10.0)
  omega ETA_RIN ~ 0.1
  sigma ADD_ERR ~ 1.0

[individual_parameters]
  RIN = TVRIN * exp(ETA_RIN)

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = if (TIME < 2.0) RIN else 0.0

[error_model]
  DV ~ additive(ADD_ERR)
";

fn read_csv(bytes: &[u8]) -> Population {
    let mut f = NamedTempFile::new().unwrap();
    f.write_all(bytes).unwrap();
    read_nonmem_csv(f.path(), None, None).expect("data reads")
}

#[test]
fn ode_time_builtin_uses_raw_clock_not_elapsed() {
    // Data on a clock that starts at 10: the `if (TIME < 2.0)` input is keyed on
    // the *raw* TIME, so it never fires (TIME >= 10 throughout). The compartment
    // stays empty — both because the predicate sees the raw clock and because
    // integration begins at the first record (t = 10), not at a phantom t = 0
    // where the input would have accumulated mass before the data starts.
    let model = parse_model_string(TIME_DEPENDENT_ODE).expect("model parses");
    let pop = read_csv(
        b"ID,TIME,DV,EVID,MDV,AMT,CMT\n\
          1,10,0,0,0,0,1\n\
          1,11,0,0,0,0,1\n\
          1,15,0,0,0,0,1\n",
    );
    let preds = predict(&model, &pop, &model.default_params);

    assert_eq!(preds.len(), 3);
    assert_eq!(preds[0].time, 10.0, "reported TIME is the raw data clock");
    assert_eq!(preds[2].time, 15.0, "reported TIME is the raw data clock");
    assert!(
        preds.iter().all(|p| p.pred.abs() < 1e-9),
        "raw TIME >= 10 so `if (TIME < 2.0)` never fires (and there is no phantom \
         [0, 10] window where it would); got {:?}",
        preds.iter().map(|p| p.pred).collect::<Vec<_>>(),
    );
}

#[test]
fn ode_time_builtin_fires_on_zero_origin_data() {
    // Same model, same elapsed sampling pattern, but a clock that starts at 0:
    // here `if (TIME < 2.0)` *does* fire over [0, 2], so the compartment fills.
    // This is the control for `ode_time_builtin_uses_raw_clock_not_elapsed`:
    // the difference between the two is exactly the raw-vs-elapsed TIME builtin.
    let model = parse_model_string(TIME_DEPENDENT_ODE).expect("model parses");
    let pop = read_csv(
        b"ID,TIME,DV,EVID,MDV,AMT,CMT\n\
          1,0,0,0,0,0,1\n\
          1,1,0,0,0,0,1\n\
          1,5,0,0,0,0,1\n",
    );
    let preds = predict(&model, &pop, &model.default_params);

    assert_eq!(preds.len(), 3);
    assert!(
        (preds[1].pred - 1.0).abs() < 1e-6,
        "at TIME=1 the input has run for 1 unit, expected ~1.0, got {}",
        preds[1].pred,
    );
    assert!(
        (preds[2].pred - 2.0).abs() < 3e-2,
        "input stops at TIME=2, so TIME=5 holds ~2.0, got {}",
        preds[2].pred,
    );
}

const TURNOVER_ODE: &str = r"
[parameters]
  theta TVKOUT(0.5, 0.0, 10.0)
  omega ETA ~ 0.1
  sigma ADD_ERR ~ 1.0

[individual_parameters]
  KOUT = TVKOUT * exp(ETA)

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  init(central) = 100.0
  d/dt(central) = -KOUT * central

[error_model]
  DV ~ additive(ADD_ERR)
";

#[test]
fn off_zero_origin_matches_zero_origin_for_time_independent_ode() {
    // A model whose RHS does not reference TIME must give identical predictions
    // whether the data clock starts at 0 or at 100, as long as the *elapsed*
    // sampling pattern is the same: integration starts at the first record, so
    // there is no phantom pre-data window to distort the initial condition.
    let model = parse_model_string(TURNOVER_ODE).expect("model parses");

    let zero = read_csv(
        b"ID,TIME,DV,EVID,MDV,AMT,CMT\n\
          1,0,0,0,0,0,1\n\
          1,2,0,0,0,0,1\n\
          1,6,0,0,0,0,1\n",
    );
    let off = read_csv(
        b"ID,TIME,DV,EVID,MDV,AMT,CMT\n\
          1,100,0,0,0,0,1\n\
          1,102,0,0,0,0,1\n\
          1,106,0,0,0,0,1\n",
    );

    let pz = predict(&model, &zero, &model.default_params);
    let po = predict(&model, &off, &model.default_params);

    assert_eq!(pz.len(), po.len());
    assert_eq!(pz.len(), 3);
    for (a, b) in pz.iter().zip(po.iter()) {
        assert!(
            (a.pred - b.pred).abs() < 1e-9,
            "off-zero origin changed a TIME-independent prediction: {} vs {}",
            a.pred,
            b.pred,
        );
    }
    // Sanity: the baseline actually decays, so the test is not trivially 0 == 0.
    assert!(pz[0].pred > pz[2].pred, "baseline should decay over time");
}

#[test]
fn single_obs_no_dose_off_zero_still_records() {
    // Degenerate timeline: one observation, no dose, off-zero. The prediction
    // must still be produced (read the initial state at the first record), not
    // left NaN by a collapsed [t0, t0] timeline.
    let model = parse_model_string(TURNOVER_ODE).expect("model parses");
    let pop = read_csv(
        b"ID,TIME,DV,EVID,MDV,AMT,CMT\n\
          1,50,0,0,0,0,1\n",
    );
    let preds = predict(&model, &pop, &model.default_params);
    assert_eq!(preds.len(), 1);
    assert!(
        (preds[0].pred - 100.0).abs() < 1e-9,
        "single off-zero obs should read the baseline initial state 100, got {}",
        preds[0].pred,
    );
}
