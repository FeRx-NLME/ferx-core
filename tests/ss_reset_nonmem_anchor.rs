//! NONMEM cross-check for #1576: **an `SS=1` record resets the system at its record**.
//!
//! A dose that precedes an `SS=1` record in (time, row order) contributes nothing at or
//! after it, and the `SS=1` dose's implied pulse train contributes nothing before it.
//! Before #1576 two engines broke the rule where it is not their compartment state that
//! carries the dose: the analytic superposition summed every dose regardless (D4; row 41,
//! `SS=1` on every dose record, read +91 %), and the ODE absorption *forcing* let a later
//! `SS=1` record's past pulses leak backwards (D1, up to 4.7e3× on row 2) and an earlier
//! dose keep absorbing after the record (D2, row 5). An ODE model with an explicit depot
//! state was already right — the equilibration replaces the state at the record — and is
//! kept here as the control.
//!
//! The reference is NONMEM 7.6.0 (`nm3`, `anchor` build), `MAXEVAL=0`, `FORMAT=s1PE23.16`:
//!
//! * `nonmem_anchor/ss_reset.{ctl,csv}` — `ADVAN2 TRANS2`, `CL = 2, V = 20, KA = 0.15`,
//!   `II = 12`. IDs 1–6 (SS regimens with a later `SS=1` at 120 / 400 or a later non-SS
//!   dose; a bolus before a mid-timeline `SS=1`; `SS=1` alone), 41 (`SS=1` on every dose
//!   record), 51/52 (a bolus row before / after a co-timed `SS=1` row — row order decides)
//!   and 53 (`SS=1` alone). NONMEM equals the closed form of the rule to **2.7e-15** on
//!   every observation (measured with an independent Python sum, outside both engines).
//! * `nonmem_anchor/ss_reset_transit.{ctl,csv}` — `ADVAN13 TOL=12`, an explicit
//!   depot→3 transit→central chain equal to ferx `transit(n=3, mtt=6)`: a non-SS regimen
//!   in flight at a mid-interval `SS=1` record with a dose change, an `SS=1` dose change
//!   mid-interval, and a non-SS continuation after `SS=1` (the control). NONMEM equals the
//!   closed form to **2.2e-12**.
//!
//! The oracle is the committed NONMEM table itself (`nonmem_anchor/results/*.sdtab`), read
//! at run time, so no reference number is hand-copied. Evaluation only — not gated.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{predict, read_nonmem_csv};

const ANALYTIC: &str = r#"
[parameters]
  theta TVCL(2.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVKA(0.15, 0.005, 20.0)
  omega ETA_CL ~ 0.0
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// `first_order(ka)` as an absorption **forcing** into central: the engine D1/D2 lived in.
const ODE_FORCING: &str = r#"
[parameters]
  theta TVCL(2.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVKA(0.15, 0.005, 20.0)
  omega ETA_CL ~ 0.0
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = first_order(ka=KA) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-11
"#;

/// The control: the same kinetics with an explicit depot **state**, which the SS
/// equilibration already replaced at the record before #1576.
const ODE_DEPOT: &str = r#"
[parameters]
  theta TVCL(2.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVKA(0.15, 0.005, 20.0)
  omega ETA_CL ~ 0.0
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = -KA*depot
  d/dt(central) = KA*depot - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-11
"#;

const ODE_TRANSIT: &str = r#"
[parameters]
  theta TVCL(2.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVMTT(6.0, 0.1, 50.0)
  theta TVN(3.0, 1.0, 30.0)
  omega ETA_CL ~ 0.0
  sigma ADD_ERR ~ 0.1 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  MTT = TVMTT
  N   = TVN
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = transit(n=N, mtt=MTT) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ additive(ADD_ERR)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-11
"#;

/// `(ID, TIME, PRED)` for every observation record of a NONMEM `$TABLE ... NOAPPEND` file
/// whose columns are `ID TIME EVID <pred>`.
fn nonmem_obs(path: &str) -> Vec<(String, f64, f64)> {
    let text = std::fs::read_to_string(path).expect("NONMEM table");
    text.lines()
        .skip(2) // "TABLE NO.  1" and the header
        .map(|l| {
            let c: Vec<f64> = l
                .split_whitespace()
                .map(|x| x.parse().expect("number"))
                .collect();
            (c[0], c[1], c[2], c[3])
        })
        .filter(|&(_, _, evid, _)| evid == 0.0)
        .map(|(id, t, _, p)| (format!("{}", id as i64), t, p))
        .collect()
}

/// Assert `model` on `data` reproduces `table` to `bound` (relative) on every observation
/// outside `skip_ids`, listing every row that misses rather than stopping at the first.
/// Every prediction is asserted finite before it is folded (`f64::max` discards `NaN`).
fn assert_matches(
    label: &str,
    model_src: &str,
    data: &str,
    table: &str,
    bound: f64,
    skip_ids: &[&str],
) {
    let model = parse_full_model(model_src).expect("model parses").model;
    let pop = read_nonmem_csv(std::path::Path::new(data), None, None).expect("dataset loads");
    let preds = predict(&model, &pop, &model.default_params).expect("predict");
    let want = nonmem_obs(table);
    assert_eq!(preds.len(), want.len(), "{label}: observation count");
    let mut worst = 0.0_f64;
    let mut bad = Vec::new();
    let mut skipped = 0;
    for (p, (id, t, w)) in preds.iter().zip(&want) {
        assert!(
            p.id == *id && (p.time - t).abs() < 1e-12,
            "{label}: row order ({}, {}) vs NONMEM ({id}, {t})",
            p.id,
            p.time
        );
        if skip_ids.contains(&id.as_str()) {
            skipped += 1;
            continue;
        }
        assert!(p.pred.is_finite(), "{label}: ID {id} t={t} non-finite");
        let rel = (p.pred - w).abs() / w.abs();
        if !(rel < bound) {
            bad.push(format!(
                "ID {id} t={t}: ferx {} vs NONMEM {w} (rel {rel:.3e})",
                p.pred
            ));
        }
        worst = worst.max(rel);
    }
    assert_eq!(
        skipped > 0,
        !skip_ids.is_empty(),
        "{label}: a skipped ID is not in the data"
    );
    println!("#1576 {label}: worst rel vs NONMEM {worst:.3e} (bound {bound:.0e})");
    assert!(
        bad.is_empty(),
        "{label}: {} rows off:\n{}",
        bad.len(),
        bad.join("\n")
    );
}

const ADVAN2_DATA: &str = "nonmem_anchor/ss_reset.csv";
const ADVAN2_TABLE: &str = "nonmem_anchor/results/ss_reset.sdtab";

/// D4. Measured 2.7e-15 (the NONMEM-vs-closed-form floor itself); the bound carries ~40x.
#[test]
fn analytic_superposition_resets_at_an_ss_record_like_nonmem() {
    assert_matches(
        "analytic one_cpt_oral",
        ANALYTIC,
        ADVAN2_DATA,
        ADVAN2_TABLE,
        1e-13,
        &[],
    );
}

/// D1 + D2. Measured 2.1e-9 at `ode_reltol = 1e-11`; the bound carries ~5x.
#[test]
fn an_ode_absorption_forcing_resets_at_an_ss_record_like_nonmem() {
    assert_matches(
        "ODE first_order forcing",
        ODE_FORCING,
        ADVAN2_DATA,
        ADVAN2_TABLE,
        1e-8,
        &[],
    );
}

/// The control: an explicit depot state was already reset by the SS equilibration, and
/// #1576 does not touch it. ID 51 (a bolus row before a co-timed `SS=1` row) was excluded
/// until #1588: the static ODE walker applied that bolus on top of the steady state
/// (+9.4 % / +44 %). It now gates the bolus on the reset, so every ID is asserted.
/// Measured 3.1e-9, unchanged from `a1084260`; the bound carries ~3x.
#[test]
fn an_ode_depot_state_was_already_reset_and_still_is() {
    assert_matches(
        "ODE depot state (control)",
        ODE_DEPOT,
        ADVAN2_DATA,
        ADVAN2_TABLE,
        1e-8,
        &[],
    );
}

/// D1 + D2 through a transit chain (ADVAN13). Measured 9.6e-12; the bound carries ~10x.
#[test]
fn a_transit_forcing_resets_at_an_ss_record_like_nonmem_advan13() {
    assert_matches(
        "ODE transit(n=3) forcing",
        ODE_TRANSIT,
        "nonmem_anchor/ss_reset_transit.csv",
        "nonmem_anchor/results/ss_reset_transit.sdtab",
        1e-10,
        &[],
    );
}
