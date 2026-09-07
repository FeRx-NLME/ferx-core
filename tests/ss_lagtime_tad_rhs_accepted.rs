//! A steady-state dose carrying a **lagtime** on an `[odes]` model whose right-hand side
//! reads `TAD` — [#1126](https://github.com/FeRx-NLME/ferx-core/issues/1126), batch-T step T4.
//!
//! # This file used to assert the opposite, and that history is the point
//!
//! Between #1139 and #1126 this combination was **rejected** with `E_SS_LAGTIME_TAD_RHS`,
//! and these seven cases each removed exactly one conjunct of that gate. The gate existed
//! because #1139 fixed the steady-state run-in for `TAD` but not the *lagged* case: `TAD`
//! had no referent inside `[t_dose, t_dose + ALAG)`, the walk's anchor fell back to the
//! subject's first arrival, and since #1121 the record-time seed *flows* to the arrival
//! rather than being re-equilibrated there — so the wrong value reached the steady-state
//! trough and shifted **every** prediction, not only those inside that window. Serving a
//! plausible number 2.7 % out was worse than the `NaN` it replaced, so it was named instead.
//!
//! #1126 supplies the referent (`crate::dosing::tad_referent`: the previous cycle's pulse at
//! `dose.time − ss_seed_phase(dose, lag)`) and stops the phase advance handing the walk a
//! `NaN` seed to carry. The gate is gone, so every case below is now an **acceptance**
//! assertion — inverted rather than deleted, because the enumeration is what shows no case
//! was quietly dropped along with the diagnostic, and because an absent gate is exactly the
//! kind of thing a later change re-introduces under another name.
//!
//! Asserting an absence is weak on its own, so the first test also runs the model and pins
//! the number: the anchored value from `tests/ss_prearrival_tad_nonmem_anchor.rs`, whose
//! oracles (an explicit 41-dose NONMEM train and a closed form outside both engines) live
//! there. A regression that re-broke the prediction while leaving `check_model_data` quiet
//! would pass the six absence checks and fail this one.

mod common;

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::pk::compute_predictions_with_tv;
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

[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#
    )
}

const LAG: &str = "  LAGTIME = TVLAG\n";
const NO_LAG: &str = "";

/// One `SS=1, II=12` record at t = 480 (or an ordinary bolus there when `ss` is false).
///
/// Observations at 481 / 482 are **inside** the pre-arrival window `[480, 483)` and 485 is
/// past the lagged arrival, so a fixture built from this exercises both sides of the
/// boundary rather than only the side a given defect happens to move.
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
            vec![481.0, 482.0, 485.0],
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

/// The retired diagnostic. Spelled out here on purpose: this file is the one place that
/// would notice the code coming back, under this name, for this combination.
const RETIRED_CODE: &str = "E_SS_LAGTIME_TAD_RHS";

/// All three conjuncts of the retired gate present — the configuration #1126 fixes.
///
/// Two assertions, and neither implies the other. `check_model_data` must be silent (the
/// gate is gone), **and** the model must predict the anchored value (the fix is real). The
/// realised errors are quoted from `tests/ss_prearrival_tad_nonmem_anchor.rs`, which owns
/// the oracles; the bound here is deliberately looser (1e-7 against a measured 1.7e-8 from
/// NONMEM's explicit lagged train) because this file's job is "is it right at all", not
/// "how right".
#[test]
fn ss_plus_lagtime_plus_a_tad_reading_rhs_is_accepted_and_predicts_the_anchor() {
    let got = codes(LAG, "+ 0.03*TAD", true);
    assert!(
        !got.iter().any(|c| c == RETIRED_CODE),
        "#1126 supplies the pre-arrival referent, so this combination must no longer be \
         rejected. Got: {got:?}"
    );
    // …and nothing else may sweep the combination up under a *different* name either. Scoped
    // to the SS/TAD/lagtime family rather than `got.is_empty()`: an unrelated future check
    // firing on a 1-cpt steady-state model would otherwise fail here with a message pointing
    // at #1126 instead of at the change that added it.
    let swept: Vec<&String> = got
        .iter()
        .filter(|c| c.contains("SS") || c.contains("TAD") || c.contains("LAG"))
        .collect();
    assert!(
        swept.is_empty(),
        "no SS / TAD / lagtime diagnostic may reject this combination. Got: {swept:?} \
         (all codes: {got:?})"
    );

    let model = parse_full_model(&model_src(LAG, "+ 0.03*TAD"))
        .expect("model parses")
        .model;
    let population = pop(true);
    let subject = &population.subjects[0];
    let preds = compute_predictions_with_tv(
        &model,
        subject,
        &model.default_params.theta,
        &vec![0.0; model.default_params.omega.dim()],
    );

    // NONMEM 7.6.0's explicit 41-dose *lagged* train (`nonmem_anchor/results/train_tadlag.tab`,
    // `ADVAN13 TOL=9`, cycle clock `MOD(T - 3 + 120, 12)`). NOT its `SS=1` record, which is
    // 1.34e-2 from its own train here and is a measured negative — see the anchor suite.
    let want = [
        (481.0, 5.5452941786123), // inside the pre-arrival window
        (482.0, 5.1924189731992), // inside
        (485.0, 8.8902010585959), // past the lagged arrival
    ];
    for (i, (t, w)) in want.iter().enumerate() {
        let got = preds[i];
        // Before the `is_finite` guard, not after: `f64::max` returns the non-NaN operand,
        // so a NaN folded into a running worst-case reads as "in range". The whole point of
        // this configuration is that it used to be NaN.
        assert!(got.is_finite(), "prediction at t={t} is {got}");
        let rel = (got - w).abs() / w.abs();
        assert!(
            rel < 1e-7,
            "t={t}: ferx {got:.13} vs NONMEM's lagged train {w:.13} — relative {rel:.3e}, \
             bound 1e-7 (realised in the anchor suite: 1.7e-8)"
        );
    }
}

/// Drop the **lagtime**: #1139's case, anchored against NONMEM in
/// `tests/ss_model_time_nonmem_anchor.rs`. It was accepted before this change and still is.
#[test]
fn ss_plus_a_tad_reading_rhs_without_a_lagtime_is_accepted() {
    let got = codes(NO_LAG, "+ 0.03*TAD", true);
    assert!(
        !got.iter().any(|c| c == RETIRED_CODE),
        "the un-lagged steady state is #1139's anchored case and was never this gate's. \
         Got: {got:?}"
    );
}

/// Drop the **`TAD` read**: SS + a lagtime on an autonomous RHS is #1121's case, anchored
/// against NONMEM (`nonmem_anchor/dose_form_lag_ss*`). Accepted before and after.
#[test]
fn ss_plus_lagtime_on_an_autonomous_rhs_is_accepted() {
    let got = codes(LAG, "", true);
    assert!(
        !got.iter().any(|c| c == RETIRED_CODE),
        "SS + lagtime with no model-time read is #1121's anchored case. Got: {got:?}"
    );
}

/// Drop the **steady state**: an ordinary lagged dose whose RHS reads `TAD` is #1073's case.
///
/// This one is not merely "still accepted" — it is the case #1126's own suggested shortcut
/// would have broken. Relaxing the post-arrival filter for *every* dose, rather than adding
/// a referent gated on `ss_seeded_at_record`, flips an ordinary lagged dose's `TAD` (0/24
/// dosing, `lag = 3`: `t = 25` goes 22.0 → −2.0). `tad_referent`'s non-SS arm keeps its
/// filter untouched, and `dosing`'s own unit tests pin that number.
#[test]
fn a_lagged_non_ss_dose_reading_tad_is_accepted() {
    let got = codes(LAG, "+ 0.03*TAD", false);
    assert!(
        !got.iter().any(|c| c == RETIRED_CODE),
        "without a steady-state dose there is no periodic pre-arrival window at all. \
         Got: {got:?}"
    );
}

/// The retired gate was **structural** — it asked whether a lagtime was *declared*, not
/// whether it was currently non-zero, so an estimated `ALAG` starting at 0 could not slip
/// through and drift into the wrong regime mid-fit.
///
/// That reasoning is exactly why the fix had to be a referent rather than a narrower gate:
/// the value the optimizer walks to changes which arm of `tad_referent` runs at every
/// observation, and `tad_referent` is correct on both. A declared-but-zero lagtime is now
/// accepted like any other.
#[test]
fn a_declared_lagtime_whose_initial_value_is_zero_is_accepted() {
    let zero_lag_model = model_src(LAG, "+ 0.03*TAD").replace("TVLAG(3.0,", "TVLAG(0.0,");
    let model = parse_full_model(&zero_lag_model)
        .expect("model parses")
        .model;
    let got: Vec<String> = check_model_data(&model, &pop(true))
        .iter()
        .map(|d| d.code.clone())
        .collect();
    assert!(
        !got.iter().any(|c| c == RETIRED_CODE),
        "a declared-but-currently-zero lagtime is served, not rejected. Got: {got:?}"
    );
}

/// `0.0*TAD` was rejected too under the structural gate — the honest consequence of a
/// predicate no coefficient can answer, and the case #1139 called indefensible.
///
/// It is now accepted **and** returns the autonomous steady state: the anchor suite pins
/// that `0.0*TAD` is bit-identical to the same model with the term removed, which is the
/// discriminator that the anchor is genuinely plumbed rather than accidentally cancelling.
#[test]
fn an_inert_tad_term_is_accepted_under_ss_plus_lagtime() {
    let got = codes(LAG, "+ 0.0*TAD", true);
    assert!(
        !got.iter().any(|c| c == RETIRED_CODE),
        "`0.0*TAD` under SS + lagtime is served now; got: {got:?}"
    );
}

/// `TAFD` under SS + a lagtime was never this gate's business — the retired code keyed on
/// `reads_tad`, not `reads_model_time`, and `TAFD` has no per-dose referent at all.
///
/// Still accepted, and still **not** anchored: `TAFD` under `SS=1` reads `NaN` by design
/// (T2 of #1258 — its dose train has no periodic limit for a run-in to converge to, measured
/// at 0.294 per doubling). T3 of #1258 has since landed and deliberately moved no value: the
/// combination is now *reported* — `W_STEADY_STATE_ABSOLUTE_TIME` — and still served.
///
/// This test is unaffected by that, and the reason is structural rather than lucky:
/// `codes()` reads `check_model_data`, the **fatal** bundle, while the new finding is a
/// warning and lives in `check_model_data_warnings`. So this file cannot observe it in
/// either direction, and asserting its presence here would be asserting against the wrong
/// bundle. It is pinned where it belongs — `src/api/tests/ss_absolute_time_tests.rs` for the
/// diagnostic and its severity, `tests/ss_model_time_nonmem_anchor.rs` for the same warning
/// arriving through `fit()` and through `ferx check`.
#[test]
fn tafd_under_ss_plus_lagtime_is_not_swept_up() {
    let got = codes(LAG, "+ 0.003*TAFD", true);
    assert!(
        !got.iter().any(|c| c == RETIRED_CODE),
        "TAFD/T/TIME under SS are reported, not rejected (T3 of #1258). Got: {got:?}"
    );
}
