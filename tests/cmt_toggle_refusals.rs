//! #2732: what a compartment off row cannot do is refused, never run with the row ignored.
//!
//! Each refusal names its own cause, and each test asserts that cause and the negative
//! of the others, because the arms share a code or a shape.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{CompiledModel, Population};
use std::io::Write;

const ODE: &str = r#"
[parameters]
  theta TCL(1.0, 0.01, 100.0)
  theta TV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.04
  sigma S ~ 0.1 (sd)
[individual_parameters]
  CL = TCL * exp(ETA_CL)
  V  = TV
[structural_model]
  ode(states=[central, urine])
[odes]
  d/dt(central) = -CL / V * central
  d/dt(urine)   = CL / V * central
[scaling]
  y[CMT=1] = central / V
  y[CMT=2] = urine
[error_model]
  DV ~ additive(S)
"#;

const ANALYTICAL: &str = r#"
[parameters]
  theta TCL(1.0, 0.01, 100.0)
  theta TV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.04
  sigma S ~ 0.1 (sd)
[individual_parameters]
  CL = TCL * exp(ETA_CL)
  V  = TV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ additive(S)
"#;

fn model(source: &str) -> CompiledModel {
    parse_full_model(source).expect("the model parses").model
}

fn data(rows: &str) -> Population {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.csv");
    let mut f = std::fs::File::create(&path).unwrap();
    write!(f, "ID,TIME,DV,EVID,AMT,CMT,MDV,II,SS\n{rows}").unwrap();
    ferx_core::read_nonmem_csv(&path, None, None).expect("the dataset loads")
}

/// The refusal `predict` gives.
fn refusal(m: &CompiledModel, pop: &Population) -> String {
    match ferx_core::predict(m, pop, &m.default_params) {
        Ok(_) => panic!("the dataset must be refused"),
        Err(e) => format!("{e:?}"),
    }
}

const OFF_AT_3: &str = "1,0,.,1,100,1,1,0,0\n1,3,.,2,.,-2,1,0,0\n";

#[test]
fn an_observation_of_an_off_compartment_is_refused() {
    let pop = data(&format!("{OFF_AT_3}1,4,1,0,.,2,0,0,0\n"));
    let e = refusal(&model(ODE), &pop);
    assert!(e.contains("E_OBS_CMT_OFF"), "{e}");
    assert!(e.contains("time 4"), "{e}");
    assert!(e.contains("compartment 2 (`urine`)"), "{e}");
    assert!(!e.contains("`central`"), "{e}");
}

#[test]
fn an_observation_on_either_side_of_the_off_window_is_accepted() {
    let m = model(ODE);
    let before = data("1,0,.,1,100,1,1,0,0\n1,2,1,0,.,2,0,0,0\n1,3,.,2,.,-2,1,0,0\n");
    assert!(ferx_core::predict(&m, &before, &m.default_params).is_ok());
    let other = data(&format!("{OFF_AT_3}1,4,1,0,.,1,0,0,0\n"));
    assert!(ferx_core::predict(&m, &other, &m.default_params).is_ok());
    let back_on = data(&format!("{OFF_AT_3}1,4,.,2,.,2,1,0,0\n1,5,1,0,.,2,0,0,0\n"));
    assert!(ferx_core::predict(&m, &back_on, &m.default_params).is_ok());
    let dosed = data(&format!("{OFF_AT_3}1,4,.,1,5,2,1,0,0\n1,5,1,0,.,2,0,0,0\n"));
    assert!(ferx_core::predict(&m, &dosed, &m.default_params).is_ok());
    let reset = data(&format!("{OFF_AT_3}1,4,.,3,.,1,1,0,0\n1,5,1,0,.,2,0,0,0\n"));
    assert!(ferx_core::predict(&m, &reset, &m.default_params).is_ok());
}

#[test]
fn an_off_row_past_the_declared_states_is_refused() {
    let pop = data("1,0,.,1,100,1,1,0,0\n1,3,.,2,.,-3,1,0,0\n1,4,1,0,.,1,0,0,0\n");
    let e = refusal(&model(ODE), &pop);
    assert!(e.contains("E_CMT_TOGGLE_OUT_OF_RANGE"), "{e}");
    assert!(e.contains("`CMT=-3`"), "{e}");
    assert!(e.contains("only 2 state(s)"), "{e}");
}

/// An on row past the end is the no-op it always was: an old marker row, not a mistake.
#[test]
fn an_on_row_past_the_declared_states_changes_nothing() {
    let m = model(ODE);
    let pop = data(&format!(
        "{OFF_AT_3}1,3.5,.,2,.,3,1,0,0\n1,4,1,0,.,1,0,0,0\n"
    ));
    assert!(ferx_core::predict(&m, &pop, &m.default_params).is_ok());
}

#[test]
fn an_analytical_model_refuses_the_row() {
    let pop = data(&format!("{OFF_AT_3}1,4,1,0,.,1,0,0,0\n"));
    let e = refusal(&model(ANALYTICAL), &pop);
    assert!(e.contains("E_CMT_TOGGLE_UNSUPPORTED"), "{e}");
    assert!(e.contains("needs an `[odes]` model"), "{e}");
    assert!(!e.contains("SDE"), "{e}");
}

#[test]
fn a_steady_state_dose_while_a_compartment_is_off_is_refused() {
    let pop = data(&format!(
        "{OFF_AT_3}1,4,.,1,100,1,1,12,1\n1,5,1,0,.,1,0,0,0\n"
    ));
    let e = refusal(&model(ODE), &pop);
    assert!(e.contains("E_CMT_TOGGLE_SS"), "{e}");
    assert!(e.contains("while `urine` is off"), "{e}");
    assert!(
        e.contains("a lagged dose only turns a compartment on when it arrives"),
        "{e}"
    );
    assert!(
        e.contains("with an `EVID=2` row before the steady-state record"),
        "{e}"
    );
}

/// NONMEM's answer here is its own periodic solve of the compartment, which ferx does not
/// reproduce (`SS=1` into the off compartment gave 24765735.951 against 441356), so the
/// dose's own compartment must not count as on for this check.
#[test]
fn a_steady_state_dose_into_the_off_compartment_is_refused() {
    let pop = data(&format!(
        "{OFF_AT_3}1,4,.,1,100,2,1,12,1\n1,5,1,0,.,1,0,0,0\n"
    ));
    let e = refusal(&model(ODE), &pop);
    assert!(e.contains("E_CMT_TOGGLE_SS"), "{e}");
}

#[test]
fn a_steady_state_dose_with_everything_on_is_accepted() {
    let m = model(ODE);
    let pop = data(&format!(
        "{OFF_AT_3}1,3.5,.,2,.,2,1,0,0\n1,4,.,1,100,1,1,12,1\n1,5,1,0,.,1,0,0,0\n"
    ));
    assert!(ferx_core::predict(&m, &pop, &m.default_params).is_ok());
}

/// The only arm of the absorption check: a compartment with no amount of its own.
#[test]
fn a_compartment_fed_by_a_built_in_input_cannot_be_switched() {
    let source = ODE
        .replace(
            "d/dt(central) = -CL / V * central\n",
            "d/dt(central) = first_order(ka=KA) - CL / V * central\n",
        )
        .replace("  V  = TV\n", "  V  = TV\n  KA = 1.0\n");
    let pop = data("1,0,.,1,100,1,1,0,0\n1,3,.,2,.,-1,1,0,0\n1,4,1,0,.,2,0,0,0\n");
    let e = refusal(&model(&source), &pop);
    assert!(e.contains("E_CMT_TOGGLE_UNSUPPORTED"), "{e}");
    assert!(
        e.contains("`central`, which a built-in absorption input feeds"),
        "{e}"
    );
}

#[test]
fn an_sde_model_refuses_the_row() {
    let parsed = ferx_core::parser::model_parser::parse_full_model_file(std::path::Path::new(
        concat!(env!("CARGO_MANIFEST_DIR"), "/examples/warfarin_sde.ferx"),
    ))
    .expect("the example parses");
    let pop = data(&format!("{OFF_AT_3}1,4,1,0,.,1,0,0,0\n"));
    let e = refusal(&parsed.model, &pop);
    assert!(e.contains("E_CMT_TOGGLE_UNSUPPORTED"), "{e}");
    assert!(e.contains("an SDE model cannot honour"), "{e}");
    assert!(!e.contains("declared endpoint"), "{e}");
}

#[cfg(feature = "survival")]
#[test]
fn an_endpoint_model_refuses_the_row() {
    let parsed = ferx_core::parser::model_parser::parse_full_model_file(std::path::Path::new(
        concat!(env!("CARGO_MANIFEST_DIR"), "/examples/pktte_joint.ferx"),
    ))
    .expect("the example parses");
    let pop = data(&format!("{OFF_AT_3}1,4,1,0,.,1,0,0,0\n"));
    let e = refusal(&parsed.model, &pop);
    assert!(e.contains("E_CMT_TOGGLE_UNSUPPORTED"), "{e}");
    assert!(
        e.contains("a model with a declared endpoint cannot honour"),
        "{e}"
    );
    assert!(!e.contains("SDE"), "{e}");
}

const FIT_ROWS: &str = "ID,TIME,DV,EVID,AMT,CMT,MDV\n\
1,0,0,1,100,1,1\n1,2,17,0,0,2,0\n1,4,30,0,0,2,0\n1,4,0,2,0,-2,1\n1,6,0,2,0,2,1\n1,8,10,0,0,2,0\n\
2,0,0,1,100,1,1\n2,2,14,0,0,2,0\n2,4,26,0,0,2,0\n2,4,0,2,0,-2,1\n2,6,0,2,0,2,1\n2,8,9,0,0,2,0\n";

/// A short fit: the subjects stay on the analytic gradient, the per-observation amounts are
/// real numbers, and a grid integral, which only the dense walk could serve, reads `NaN`
/// rather than integrating through the off window.
#[test]
fn a_fit_runs_on_the_analytic_gradient_and_a_grid_output_reads_nan() {
    let dir = tempfile::tempdir().unwrap();
    let (model_path, data_path) = (dir.path().join("m.ferx"), dir.path().join("d.csv"));
    let source = format!(
        "{ODE}[derived]\n  AMOUNT = compartments[1]\n  AUC_U = integral(compartments[1], from=0, to=8)\n\
         [fit_options]\n  method = focei\n  maxiter = 2\n  covariance = false\n"
    );
    std::fs::write(&model_path, source).unwrap();
    std::fs::write(&data_path, FIT_ROWS).unwrap();
    let (result, _) = ferx_core::run_model_with_data(
        model_path.to_str().unwrap(),
        Some(data_path.to_str().unwrap()),
    )
    .expect("the fit runs");
    assert!(
        result.ofv.is_finite() && result.ofv < 1e6,
        "OFV {}",
        result.ofv
    );
    assert!(
        !result
            .warnings
            .iter()
            .any(|w| w.contains("finite-difference")),
        "{:?}",
        result.warnings
    );
    for sr in &result.subjects {
        let column = |name: &str| {
            &sr.extra_columns
                .iter()
                .find(|(n, _)| n == name)
                .unwrap_or_else(|| panic!("no column {name}"))
                .1
        };
        let amount = column("AMOUNT");
        assert!(amount.iter().all(|v| v.is_finite()), "{amount:?}");
        // Refilled from zero after the window: the t = 8 amount is under the t = 4 one.
        assert!(amount[2] < amount[1], "{amount:?}");
        assert!(
            column("AUC_U").iter().all(|v| v.is_nan()),
            "{:?}",
            column("AUC_U")
        );
    }
}

/// A subject built in memory, as the R glue and FREM build them, has no written times. The
/// refusal must still be a refusal, not an index panic.
#[test]
fn a_subject_built_in_memory_is_refused_at_its_engine_time() {
    use ferx_core::types::CmtToggle;
    let m = model(ODE);
    let mut pop = data(&format!("{OFF_AT_3}1,4,1,0,.,2,0,0,0\n"));
    let s = &mut pop.subjects[0];
    s.obs_raw_times.clear();
    s.pk_only_raw_times.clear();
    assert_eq!(
        s.cmt_toggles[0],
        CmtToggle {
            pk_only: 0,
            state: 1,
            on: false
        }
    );
    let e = refusal(&m, &pop);
    assert!(e.contains("E_OBS_CMT_OFF") && e.contains("time 4"), "{e}");
    pop.subjects[0].cmt_toggles[0].state = 5;
    let e = refusal(&m, &pop);
    assert!(
        e.contains("E_CMT_TOGGLE_OUT_OF_RANGE") && e.contains("time 3"),
        "{e}"
    );
}

/// NONMEM stops on this row too, but it asks for no prediction, so ferx reads it as the
/// pk-only record it is. Documented as a difference in `docs/data-format.qmd`.
#[test]
fn an_mdv_row_on_an_off_compartment_is_accepted() {
    let m = model(ODE);
    let pop = data(&format!("{OFF_AT_3}1,4,.,0,.,2,1,0,0\n1,5,1,0,.,1,0,0,0\n"));
    assert!(ferx_core::predict(&m, &pop, &m.default_params).is_ok());
}

#[test]
fn an_on_off_record_naming_no_row_is_refused() {
    let m = model(ODE);
    let mut pop = data(&format!("{OFF_AT_3}1,4,1,0,.,1,0,0,0\n"));
    pop.subjects[0].cmt_toggles[0].pk_only = 7;
    let e = refusal(&m, &pop);
    assert!(e.contains("E_CMT_TOGGLE_OUT_OF_RANGE"), "{e}");
    assert!(e.contains("pk-only row 7, but the subject has 1"), "{e}");
}

fn lagged_urine() -> CompiledModel {
    model(&ODE.replace("  V  = TV\n", "  V  = TV\n  ALAG2 = 1.0\n"))
}

const LAGGED_DOSE_AT_4: &str = "1,4,.,1,50,2,1,0,0\n";

/// A lagged dose clears an observation of its compartment at the record, but the walk holds
/// the compartment off to the arrival, so an `SS=1` record in between would refill it.
#[test]
fn a_steady_state_dose_before_a_lagged_arrival_is_refused() {
    let pop = data(&format!(
        "{OFF_AT_3}{LAGGED_DOSE_AT_4}1,4.5,.,1,100,1,1,12,1\n1,6,1,0,.,1,0,0,0\n"
    ));
    let e = refusal(&lagged_urine(), &pop);
    assert!(e.contains("E_CMT_TOGGLE_SS"), "{e}");
    assert!(e.contains("time 4.5"), "{e}");
    assert!(e.contains("`urine`"), "{e}");
}

/// The two halves of that status, each with the other held fixed. Same rows, no lag: the
/// dose turns the compartment on at its record. Same lag, an on row: nothing is held. Same
/// lag, no `SS=1`: the observation between record and arrival stays accepted.
#[test]
fn the_lag_alone_decides_the_steady_state_refusal() {
    let ss_rows =
        format!("{OFF_AT_3}{LAGGED_DOSE_AT_4}1,4.5,.,1,100,1,1,12,1\n1,6,1,0,.,1,0,0,0\n");
    let unlagged = model(ODE);
    assert!(ferx_core::predict(&unlagged, &data(&ss_rows), &unlagged.default_params).is_ok());

    let m = lagged_urine();
    let turned_on = data(&format!(
        "{OFF_AT_3}{LAGGED_DOSE_AT_4}1,4.2,.,2,.,2,1,0,0\n1,4.5,.,1,100,1,1,12,1\n1,6,1,0,.,1,0,0,0\n"
    ));
    assert!(ferx_core::predict(&m, &turned_on, &m.default_params).is_ok());

    let observed = data(&format!(
        "{OFF_AT_3}{LAGGED_DOSE_AT_4}1,4.5,1,0,.,2,0,0,0\n"
    ));
    assert!(ferx_core::predict(&m, &observed, &m.default_params).is_ok());
}

/// A reset turns every compartment on, so an `SS=1` dose after one has nothing held off.
#[test]
fn a_steady_state_dose_after_a_reset_is_accepted() {
    let m = model(ODE);
    let pop = data(&format!(
        "{OFF_AT_3}1,3.5,.,3,.,1,1,0,0\n1,4,.,1,100,1,1,12,1\n1,5,1,0,.,1,0,0,0\n"
    ));
    assert!(ferx_core::predict(&m, &pop, &m.default_params).is_ok());
}

/// A dataset with an `ADDL` column, for the trains below.
fn data_addl(rows: &str) -> Population {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.csv");
    std::fs::write(
        &path,
        format!("ID,TIME,DV,EVID,AMT,CMT,MDV,II,SS,ADDL\n{rows}"),
    )
    .unwrap();
    ferx_core::read_nonmem_csv(&path, None, None).expect("the dataset loads")
}

/// A train into `urine` at 0, 2, 4 and 6, cancelled by the off row at 3. The dose at 4 stays
/// in `Subject::doses` but never arrives, so it turns nothing on.
const CANCELLED_TRAIN: &str = "1,0,.,1,100,1,1,0,0,0\n1,0,.,1,10,2,1,2,0,3\n1,3,.,2,.,-2,1,0,0,0\n";

/// NONMEM 7.5.1 stops here too, on record 5: "COMPARTMENT ASSOCIATED WITH THE PREDICTION IS
/// OFF".
#[test]
fn an_observation_past_a_cancelled_train_is_refused() {
    let pop = data_addl(&format!("{CANCELLED_TRAIN}1,5,1,0,.,2,0,0,0,0\n"));
    let e = refusal(&model(ODE), &pop);
    assert!(e.contains("E_OBS_CMT_OFF"), "{e}");
    assert!(e.contains("time 5"), "{e}");
}

#[test]
fn a_steady_state_dose_past_a_cancelled_train_is_refused() {
    let pop = data_addl(&format!(
        "{CANCELLED_TRAIN}1,5,.,1,100,1,1,12,1,0\n1,6,1,0,.,1,0,0,0,0\n"
    ));
    let e = refusal(&model(ODE), &pop);
    assert!(e.contains("E_CMT_TOGGLE_SS"), "{e}");
    assert!(e.contains("time 5"), "{e}");
}

/// The control: a train that starts after the off row is live, and its first dose turns
/// the compartment on, so only the cancellation above can be what refuses.
#[test]
fn an_observation_after_a_live_train_is_accepted() {
    let m = model(ODE);
    let pop = data_addl(
        "1,0,.,1,100,1,1,0,0,0\n1,1,.,2,.,-2,1,0,0,0\n1,2,.,1,10,2,1,2,0,3\n1,5,1,0,.,2,0,0,0,0\n",
    );
    assert!(ferx_core::predict(&m, &pop, &m.default_params).is_ok());
}
