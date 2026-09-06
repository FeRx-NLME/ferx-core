//! `E_SS_LAGTIME_TAD_RHS` — a steady-state dose carrying a **lagtime** on an `[odes]`
//! model whose right-hand side reads `TAD` (#1139 / [#1126]).
//!
//! #1139 makes the steady-state run-in read `TAD` on the cycle-local clock, anchored per
//! window, so the un-lagged case now reproduces NONMEM (`tests/ss_model_time_nonmem_anchor.rs`).
//! A lagtime adds a *second* question that fix does not answer: what `TAD` refers to inside
//! `[t_dose, t_dose + ALAG)`, between the dose record and the lagged arrival. There the
//! walk's anchor falls back to the subject's first arrival, so `TAD` runs negative where
//! the periodic fiction says it is `t - (t_dose + ALAG - II)`.
//!
//! That is **not confined to the window**. Since #1121 the record-time seed *flows* to the
//! arrival instead of being re-equilibrated there, so the wrong anchor is carried into the
//! trough and multiplies into every later prediction. Measured on `CL = 1`, `V = 20`,
//! `AMT = 100`, `II = 12`, `ALAG = 3` and `-(CL/V)·A·(1 + 0.03·TAD)`: a uniform **2.733 %**
//! high at all four post-arrival observations and 3.67 % inside the window, against a
//! closed form computed outside the engine. Before #1139 the same model returned `NaN` and
//! a `W_ODE_SOLVER_DIAGNOSTICS` warning blaming the solver.
//!
//! A plausible number 2.7 % out is worse than the `NaN` it would replace, so the
//! combination is named and rejected rather than served — the resolution #1139 itself
//! proposed, and the same shape as the neighbouring `E_ABSORPTION_SS_LAG`. #1126 fixes the
//! pre-arrival referent and lifts this gate; both halves belong in that change, where a
//! mutation can show that neither alone suffices.
//!
//! Each test below removes exactly **one** conjunct of the gate, so no two of them reject
//! the same inputs and none could be deleted with the others still covering it.

mod common;

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{check_model_data, DoseEvent, Population};

/// 1-cpt IV whose `[odes]` RHS reads `TAD`, with `{LAG}` and `{TERM}` substituted per test.
fn model_src(lag_line: &str, term: &str) -> String {
    format!(
        r#"
[parameters]
  theta TVCL(1.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVLAG(3.0, 0.0, 10.0)

  omega ETA_CL ~ 0.09

  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
{lag_line}
[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -(CL/V) * central * (1.0 {term})

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP_ERR)
"#
    )
}

const LAG: &str = "  LAGTIME = TVLAG\n";
const NO_LAG: &str = "";

fn pop(ss: bool) -> Population {
    let dose = DoseEvent::new(480.0, 100.0, 1, 0.0, ss, if ss { 12.0 } else { 0.0 });
    Population {
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![common::subject(
            "1",
            vec![dose],
            vec![482.0, 485.0, 488.0],
            vec![1.0; 3],
            vec![1; 3],
        )],
    }
}

fn codes(lag_line: &str, term: &str, ss: bool) -> Vec<String> {
    let model = parse_full_model(&model_src(lag_line, term))
        .expect("model parses")
        .model;
    check_model_data(&model, &pop(ss))
        .iter()
        .map(|d| d.code.clone())
        .collect()
}

const CODE: &str = "E_SS_LAGTIME_TAD_RHS";

#[test]
fn ss_plus_lagtime_plus_a_tad_reading_rhs_is_rejected() {
    let got = codes(LAG, "+ 0.03*TAD", true);
    assert!(
        got.iter().any(|c| c == CODE),
        "all three conjuncts present must raise {CODE}, got: {got:?}"
    );
}

/// Drop the **lagtime**: this is the case #1139 fixes and NONMEM-anchors. It must pass.
#[test]
fn ss_plus_a_tad_reading_rhs_without_a_lagtime_is_accepted() {
    let got = codes(NO_LAG, "+ 0.03*TAD", true);
    assert!(
        !got.iter().any(|c| c == CODE),
        "a steady-state dose reading TAD *without* a lagtime is exactly what #1139 makes \
         correct and is NONMEM-anchored — it must not be rejected. Got: {got:?}"
    );
}

/// Drop the **`TAD` read**: SS + a lagtime on an autonomous RHS is #1121's case, anchored
/// against NONMEM (`nonmem_anchor/dose_form_lag_ss*`). It must pass.
#[test]
fn ss_plus_lagtime_on_an_autonomous_rhs_is_accepted() {
    let got = codes(LAG, "", true);
    assert!(
        !got.iter().any(|c| c == CODE),
        "SS + lagtime with no model-time read is #1121's anchored case, not this gate's. \
         Got: {got:?}"
    );
}

/// Drop the **steady state**: an ordinary lagged dose whose RHS reads `TAD` is #1073's
/// case, and its pre-arrival anchor is already finite and mesh-invariant. It must pass.
#[test]
fn a_lagged_non_ss_dose_reading_tad_is_accepted() {
    let got = codes(LAG, "+ 0.03*TAD", false);
    assert!(
        !got.iter().any(|c| c == CODE),
        "without a steady-state dose there is no periodic pre-arrival window and nothing \
         for this gate to catch. Got: {got:?}"
    );
}

/// The gate is **structural**, not typical-value: it asks whether a lagtime is *declared*,
/// not whether it is currently non-zero.
///
/// This is deliberate and is the one place the gate is knowingly wider than the defect. An
/// estimated `ALAG` moves during a fit, so a value gate would let a run start at `lag = 0`,
/// pass, and drift into the wrong regime with no diagnostic — the failure mode that makes
/// a value gate worse than a structural one here. `E_ABSORPTION_SS_LAG` is structural for
/// the same reason.
#[test]
fn a_declared_lagtime_is_enough_even_when_its_initial_value_is_zero() {
    let zero_lag_model = model_src(LAG, "+ 0.03*TAD").replace("TVLAG(3.0,", "TVLAG(0.0,");
    let model = parse_full_model(&zero_lag_model)
        .expect("model parses")
        .model;
    let got: Vec<String> = check_model_data(&model, &pop(true))
        .iter()
        .map(|d| d.code.clone())
        .collect();
    assert!(
        got.iter().any(|c| c == CODE),
        "a declared-but-currently-zero lagtime must still be rejected — the optimizer can \
         move it off zero mid-fit and there would be no second chance to notice. Got: {got:?}"
    );
}

/// `0.0*TAD` is rejected too, and that is the honest consequence of a structural gate: the
/// predicate is "does this RHS *read* `TAD`", which no coefficient can answer.
///
/// Recorded rather than worked around. #1139 complains that `0.0*TAD` returning `NaN` is
/// indefensible — a named error saying which combination is unsupported is strictly better
/// than that, and the un-lagged `0.0*TAD` case (the one the issue actually reproduces) is
/// accepted and correct.
#[test]
fn an_inert_tad_term_is_rejected_too_under_ss_plus_lagtime() {
    let got = codes(LAG, "+ 0.0*TAD", true);
    assert!(
        got.iter().any(|c| c == CODE),
        "the gate is structural, so an inert `0.0*TAD` under SS + lagtime is rejected as \
         well; got: {got:?}"
    );
    // …but only under a lagtime. Without one it is accepted and returns the autonomous
    // steady state — the case #1139 opens with.
    let unlagged = codes(NO_LAG, "+ 0.0*TAD", true);
    assert!(
        !unlagged.iter().any(|c| c == CODE),
        "`0.0*TAD` without a lagtime must be accepted; got: {unlagged:?}"
    );
}

/// `TAFD` under SS + a lagtime is **not** this gate's business: it has no pre-arrival
/// referent problem, because it has no per-dose referent at all. Sweeping it in would name
/// the wrong issue, and #1139's remaining half is what covers it.
#[test]
fn tafd_under_ss_plus_lagtime_is_not_swept_up() {
    let got = codes(LAG, "+ 0.003*TAFD", true);
    assert!(
        !got.iter().any(|c| c == CODE),
        "the gate must key on `reads_tad`, not on `reads_model_time` — TAFD/T/TIME under \
         SS are a separate question. Got: {got:?}"
    );
}
