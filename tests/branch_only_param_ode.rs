//! Regression test for issue #357: an individual parameter assigned *only*
//! inside symmetric `if`/`else` branches in `[individual_parameters]` must be
//! a first-class individual parameter on an ODE model — it has to (a) parse
//! (the `[odes]` RHS name resolver previously rejected it as undefined), (b)
//! earn a PK slot, and (c) actually be written back by `pk_param_fn` so the
//! ODE RHS reads its real value rather than the silent-0 sentinel.
//!
//! The check is an equivalence: a model whose `CL` is written on *both*
//! branches of an `if`/`else` (with the two branches collapsing to the same
//! value for the covariate we feed it) must `predict()` identically to the
//! plain top-level-`CL` form. If the fix only silenced the parse error without
//! wiring the write-back, the branch-only model would predict with `CL = 0`
//! (a flat, non-eliminating trajectory) and diverge sharply from the twin.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::predict;
use ferx_core::types::{DoseEvent, Population, Subject};

mod common;

const ATOL: f64 = 1e-5;
const RTOL: f64 = 1e-4;

/// Top-level `CL` — the control. `WT` is referenced so both forms parse against
/// the identical covariate set.
const TOP_LEVEL: &str = r"
[parameters]
  theta TVCL(3.0, 0.01, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  omega ETA_CL ~ 0.1
  sigma PROP_ERR ~ 0.04

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -(CL/V) * central

[scaling]
  obs_scale = V

[error_model]
  DV ~ proportional(PROP_ERR)
";

/// `CL` assigned only inside `if`/`else`. Both branches reduce to the same
/// expression as `TOP_LEVEL` (the `* 1.0` factor keeps the value identical) so
/// predictions must match the control exactly, proving the branch-only `CL`
/// is computed into its PK slot and seen by the ODE RHS.
const BRANCH_ONLY: &str = r"
[parameters]
  theta TVCL(3.0, 0.01, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  omega ETA_CL ~ 0.1
  sigma PROP_ERR ~ 0.04

[individual_parameters]
  if (WT > 70) {
    CL = TVCL * 1.0 * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V = TVV

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -(CL/V) * central

[scaling]
  obs_scale = V

[error_model]
  DV ~ proportional(PROP_ERR)
";

fn population(wt: f64) -> Population {
    let obs_times = vec![0.5, 1.0, 2.0, 4.0, 8.0, 24.0];
    let n = obs_times.len();
    let mut s: Subject = common::subject(
        "1",
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times,
        vec![0.0; n],
        vec![1; n],
    );
    s.covariates.insert("WT".to_string(), wt);
    Population {
        covariate_names: vec!["WT".to_string()],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![s],
    }
}

fn assert_branch_matches_top_level(wt: f64) {
    let top = parse_full_model(TOP_LEVEL)
        .unwrap_or_else(|e| panic!("top-level model did not parse: {e}"))
        .model;
    // The crux of #357: this used to fail with
    // `[odes]: RHS references undefined name(s): CL`.
    let branch = parse_full_model(BRANCH_ONLY)
        .unwrap_or_else(|e| panic!("branch-only model did not parse (issue #357): {e}"))
        .model;

    let pop = population(wt);
    let pt = predict(&top, &pop, &top.default_params);
    let pb = predict(&branch, &pop, &branch.default_params);
    assert_eq!(pt.len(), pb.len(), "prediction count mismatch (WT={wt})");
    assert!(!pt.is_empty(), "no predictions produced (WT={wt})");

    // Every prediction must be a real, eliminating concentration (a silent
    // CL=0 would leave the central amount constant → a flat, much larger PRED).
    let mut any_positive = false;
    for (x, y) in pt.iter().zip(pb.iter()) {
        let tol = ATOL + RTOL * x.pred.abs();
        assert!(
            (x.pred - y.pred).abs() <= tol,
            "WT={wt} t={:.3}: top-level PRED {:.6} vs branch-only PRED {:.6} (|diff| {:.2e} > tol {:.2e})",
            wt,
            x.pred,
            y.pred,
            (x.pred - y.pred).abs(),
            tol
        );
        if x.pred.abs() > 0.0 {
            any_positive = true;
        }
    }
    assert!(
        any_positive,
        "all predictions were zero (WT={wt}) — model is degenerate"
    );
}

#[test]
fn branch_only_param_predicts_like_top_level_wt_high() {
    // WT > 70 → the `if` branch fires.
    assert_branch_matches_top_level(80.0);
}

#[test]
fn branch_only_param_predicts_like_top_level_wt_low() {
    // WT <= 70 → the `else` branch fires. Both must write CL to its slot.
    assert_branch_matches_top_level(60.0);
}

/// A branch-only helper named like a reserved PK parameter (`F`) that is NOT
/// referenced by any downstream block must NOT be promoted into the PK array —
/// otherwise it silently hijacks the engine's bioavailability slot and corrupts
/// dosing. Guards the over-promotion regression flagged in review of #357: the
/// fix promotes an all-branch name only when a downstream block consumes it.
const UNUSED_BRANCH_F: &str = r"
[parameters]
  theta TVCL(3.0, 0.01, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  omega ETA_CL ~ 0.1
  sigma PROP_ERR ~ 0.04

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  if (WT > 70) {
    F = 0.5
  } else {
    F = 0.5
  }

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -(CL/V) * central

[scaling]
  obs_scale = V

[error_model]
  DV ~ proportional(PROP_ERR)
";

#[test]
fn unused_branch_f_does_not_hijack_bioavailability() {
    // `F` is assigned on every branch but never referenced in [odes]/[scaling],
    // so it must stay branch-local and leave the engine's default F = 1.0 in
    // place. Predictions must equal the model without the F block. If `F` were
    // promoted it would land in PK_IDX_F = 0.5, halving the delivered dose and
    // halving every prediction.
    let top = parse_full_model(TOP_LEVEL)
        .expect("top-level model did not parse")
        .model;
    let with_f = parse_full_model(UNUSED_BRANCH_F)
        .expect("unused-F model did not parse")
        .model;

    let pop = population(80.0);
    let pt = predict(&top, &pop, &top.default_params);
    let pf = predict(&with_f, &pop, &with_f.default_params);
    assert_eq!(pt.len(), pf.len());
    assert!(!pt.is_empty());
    for (x, y) in pt.iter().zip(pf.iter()) {
        let tol = ATOL + RTOL * x.pred.abs();
        assert!(
            (x.pred - y.pred).abs() <= tol,
            "t={:.3}: baseline PRED {:.6} vs unused-F PRED {:.6} — F was promoted and hijacked the dosing slot",
            x.time,
            x.pred,
            y.pred
        );
    }
}
