//! `E_KAPPA_WEIGHT_NONPOSITIVE` / `W_KAPPA_WEIGHT_VARIES_WITHIN_OCCASION`
//! (#1031): a sample-size-weighted IOV kappa is applied as `κ/√W`, so a weight
//! that is zero, negative or non-finite divides an individual parameter by zero
//! — a non-finite prediction deep in the inner loop with no visible cause.
//! Caught up front instead, against the arm-size covariate the dataset carries.

use super::{check_model_data, check_model_data_warnings};
use crate::parser::model_parser::parse_model_string;
use crate::types::{Population, Subject};
use std::collections::HashMap;

fn weighted_kappa_model() -> crate::types::CompiledModel {
    parse_model_string(
        "[parameters]\n  theta TVCL(0.2)\n  theta TVV(10.0)\n  omega ETA_CL ~ 0.09\n  \
         kappa KAPPA_CL ~ 2.0 (sd) weight = NARM\n  sigma PROP_ERR ~ \
         0.04\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V  = \
         TVV\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ \
         proportional(PROP_ERR)\n[covariates]\n  NARM continuous\n[fit_options]\n  \
         iov_column = OCC\n",
    )
    .expect("parse")
}

/// One subject, one observation per (occasion, arm size) pair.
fn population_with_arm_sizes(rows: &[(u32, f64)]) -> Population {
    let snap = |n: f64| -> HashMap<String, f64> { [("NARM".to_string(), n)].into_iter().collect() };
    let subject = Subject {
        id: "1".to_string(),
        obs_times: (0..rows.len()).map(|j| j as f64 + 1.0).collect(),
        observations: vec![9.0; rows.len()],
        obs_cmts: vec![1; rows.len()],
        cens: vec![0; rows.len()],
        occasions: rows.iter().map(|&(occ, _)| occ).collect(),
        covariates: snap(rows[0].1),
        obs_covariates: rows.iter().map(|&(_, n)| snap(n)).collect(),
        ..Default::default()
    };
    Population {
        subjects: vec![subject],
        covariate_names: vec!["NARM".to_string()],
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

fn weight_error(pop: &Population) -> Option<String> {
    check_model_data(&weighted_kappa_model(), pop)
        .into_iter()
        .find(|d| d.code == "E_KAPPA_WEIGHT_NONPOSITIVE")
        .map(|d| d.message)
}

fn variation_warning(pop: &Population) -> Option<String> {
    let model = weighted_kappa_model();
    let init = model.default_params.clone();
    check_model_data_warnings(&model, pop, &init)
        .into_iter()
        .find(|d| d.code == "W_KAPPA_WEIGHT_VARIES_WITHIN_OCCASION")
        .map(|d| d.message)
}

#[test]
fn positive_arm_sizes_pass() {
    assert_eq!(
        weight_error(&population_with_arm_sizes(&[(1, 200.0), (2, 150.0)])),
        None
    );
}

#[test]
fn a_zero_arm_size_is_rejected() {
    // The concrete MBMA failure: one arm's N cell is blank, the covariate reads
    // 0, and `κ/√0` blows the individual parameter to infinity.
    let msg = weight_error(&population_with_arm_sizes(&[(1, 200.0), (2, 0.0)]))
        .expect("a zero weight must be an error");
    assert!(msg.contains("KAPPA_CL"), "got: {msg}");
    assert!(msg.contains("TIME 2"), "got: {msg}");
}

#[test]
fn a_negative_arm_size_is_rejected() {
    // `sqrt` of a negative is clamped to 0 by the evaluator, so this lands as
    // the same division by zero — with no hint of where it came from.
    assert!(weight_error(&population_with_arm_sizes(&[(1, -4.0)])).is_some());
}

#[test]
fn a_non_finite_arm_size_is_rejected() {
    assert!(weight_error(&population_with_arm_sizes(&[(1, f64::NAN)])).is_some());
}

#[test]
fn a_weight_constant_within_each_occasion_warns_about_nothing() {
    assert_eq!(
        variation_warning(&population_with_arm_sizes(&[
            (1, 200.0),
            (1, 200.0),
            (2, 150.0)
        ])),
        None
    );
}

#[test]
fn a_weight_that_moves_within_an_occasion_warns_once() {
    // κ is drawn once per occasion, so Ω_IOV/W is ambiguous when W moves
    // underneath it. Not fatal — a dropout-adjusted N may be intended.
    let pop = population_with_arm_sizes(&[(1, 200.0), (1, 180.0), (1, 160.0)]);
    let msg = variation_warning(&pop).expect("a moving weight must warn");
    assert!(msg.contains("not constant within occasion 1"), "got: {msg}");
    let model = weighted_kappa_model();
    let init = model.default_params.clone();
    let n = check_model_data_warnings(&model, &pop, &init)
        .iter()
        .filter(|d| d.code == "W_KAPPA_WEIGHT_VARIES_WITHIN_OCCASION")
        .count();
    assert_eq!(n, 1, "one warning per kappa, not one per record");
}

#[test]
fn a_bad_arm_size_on_a_dose_record_is_rejected() {
    // The individual parameters are rebuilt at every dose event, so a dose row
    // reads the weight too — and a dose-only occasion has no observation for
    // the loop above to have caught.
    let mut pop = population_with_arm_sizes(&[(1, 200.0)]);
    let subj = &mut pop.subjects[0];
    subj.doses = vec![crate::types::DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    subj.dose_occasions = vec![1];
    subj.dose_covariates = vec![[("NARM".to_string(), 0.0)].into_iter().collect()];
    let msg = weight_error(&pop).expect("a zero weight on a dose row must be an error");
    assert!(msg.contains("dose record"), "got: {msg}");
}

/// The reported "typical" arm is the median over subject-occasions — the
/// granularity κ is drawn at, not the observation count, so a heavily-sampled
/// arm does not outvote a sparse one.
#[test]
fn the_typical_arm_size_is_the_median_over_occasions() {
    let model = weighted_kappa_model();
    // Occasion 1 carries three rows at N = 400, occasions 2 and 3 one each at
    // 25 and 100. Per-occasion: [400, 25, 100] → median 100. Per-observation it
    // would have been 400.
    let pop =
        population_with_arm_sizes(&[(1, 400.0), (1, 400.0), (1, 400.0), (2, 25.0), (3, 100.0)]);
    let typicals = super::kappa_weight_typicals(&model, &pop, &model.default_params.theta);
    assert_eq!(typicals, vec![Some(100.0)]);
}

#[test]
fn an_unweighted_kappa_has_no_typical_arm_size() {
    let plain = parse_model_string(
        "[parameters]\n  theta TVCL(0.2)\n  theta TVV(10.0)\n  omega ETA_CL ~ 0.09\n  \
         kappa KAPPA_CL ~ 0.09\n  sigma PROP_ERR ~ 0.04\n[individual_parameters]\n  CL = \
         TVCL * exp(ETA_CL + KAPPA_CL)\n  V  = TVV\n[structural_model]\n  pk \
         one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ \
         proportional(PROP_ERR)\n[covariates]\n  NARM continuous\n[fit_options]\n  \
         iov_column = OCC\n",
    )
    .expect("parse");
    let pop = population_with_arm_sizes(&[(1, 200.0)]);
    assert!(
        super::kappa_weight_typicals(&plain, &pop, &plain.default_params.theta).is_empty(),
        "an unweighted model reports nothing, so the fit YAML is unchanged"
    );
}

#[test]
fn an_unweighted_kappa_is_not_checked() {
    let plain = parse_model_string(
        "[parameters]\n  theta TVCL(0.2)\n  theta TVV(10.0)\n  omega ETA_CL ~ 0.09\n  \
         kappa KAPPA_CL ~ 0.09\n  sigma PROP_ERR ~ 0.04\n[individual_parameters]\n  CL = \
         TVCL * exp(ETA_CL + KAPPA_CL)\n  V  = TVV\n[structural_model]\n  pk \
         one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ \
         proportional(PROP_ERR)\n[covariates]\n  NARM continuous\n[fit_options]\n  \
         iov_column = OCC\n",
    )
    .expect("parse");
    let pop = population_with_arm_sizes(&[(1, 0.0), (2, -1.0)]);
    assert!(check_model_data(&plain, &pop)
        .iter()
        .all(|d| d.code != "E_KAPPA_WEIGHT_NONPOSITIVE"));
}

/// `simulate()` runs its own (shorter) validation list, so the weight check has
/// to be wired there too: a blank arm-size cell in a simulation dataset — the
/// exact MBMA case the error exists for — otherwise divides an individual
/// parameter by zero and lands as NaN/Inf DV instead of an up-front error.
#[test]
fn a_zero_arm_size_is_rejected_on_the_simulate_path() {
    let model = weighted_kappa_model();
    let pop = population_with_arm_sizes(&[(1, 200.0), (2, 0.0)]);
    let err = crate::api::simulate_with_options(
        &model,
        &pop,
        &model.default_params,
        1,
        &crate::api::SimulateOptions::default(),
    )
    .expect_err("simulate must reject a non-positive kappa weight");
    assert!(
        err.contains("kappa `KAPPA_CL` weight `NARM` evaluates to 0"),
        "got: {err}"
    );
}

/// The same path must stay open for a dataset whose arm sizes are all positive.
#[test]
fn positive_arm_sizes_pass_on_the_simulate_path() {
    let model = weighted_kappa_model();
    let pop = population_with_arm_sizes(&[(1, 200.0), (2, 150.0)]);
    let out = crate::api::simulate_with_options(
        &model,
        &pop,
        &model.default_params,
        1,
        &crate::api::SimulateOptions::default(),
    );
    assert!(
        !matches!(&out, Err(e) if e.contains("weight `NARM`")),
        "a positive weight must not be rejected: {out:?}"
    );
}

// ── The Vec-returning entry points, and the degenerate oracle (#1083) ────────
//
// `κ/√W` with `W = 0` does *not* blow up: `BinOp::Div` returns `0.0` when its
// divisor underflows, so a blank arm-size cell removes the arm's between-arm
// variability and leaves finite, plottable rows behind. `check_kappa_weights`
// was therefore the entire defence — and it was absent from `simulate` /
// `simulate_with_seed`, which return a bare `Vec` and had no check at all.

#[test]
#[should_panic(expected = "weight `NARM` evaluates to 0")]
fn the_vec_returning_entry_point_panics_on_a_zero_arm_size() {
    let model = weighted_kappa_model();
    let pop = population_with_arm_sizes(&[(1, 200.0), (2, 0.0)]);
    let _ = crate::api::simulate_with_seed(&model, &pop, &model.default_params, 1, 42);
}

/// [`population_with_arm_sizes`] plus a bolus at t = 0, so the simulated rows
/// carry a *non-zero* prediction. Without a dose every IPRED is 0.0 and the two
/// tests below compare zeros to zeros — passing while measuring nothing.
fn dosed_population_with_arm_sizes(rows: &[(u32, f64)]) -> Population {
    let mut pop = population_with_arm_sizes(rows);
    let subj = &mut pop.subjects[0];
    subj.doses = vec![crate::types::DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    subj.dose_occasions = vec![rows[0].0];
    subj.dose_covariates = vec![subj.covariates.clone()];
    pop
}

/// The degenerate oracle for the weighted-κ simulate path: `weight = NARM` is
/// applied by desugaring every reference into `KAPPA_CL / sqrt(NARM)`, so a
/// dataset whose arm sizes are all exactly 1 must simulate **bit-identically** to
/// the same model with no `weight =` modifier at all. A scaling applied on the
/// wrong side, or applied twice, disagrees here and nowhere else.
#[test]
fn an_arm_size_of_one_simulates_bit_identically_to_an_unweighted_kappa() {
    let weighted = weighted_kappa_model();
    let plain = parse_model_string(
        "[parameters]\n  theta TVCL(0.2)\n  theta TVV(10.0)\n  omega ETA_CL ~ 0.09\n  \
         kappa KAPPA_CL ~ 2.0 (sd)\n  sigma PROP_ERR ~ 0.04\n[individual_parameters]\n  \
         CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V  = TVV\n[structural_model]\n  \
         pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ \
         proportional(PROP_ERR)\n[fit_options]\n  iov_column = OCC\n",
    )
    .expect("parse");
    let pop = dosed_population_with_arm_sizes(&[(1, 1.0), (2, 1.0), (2, 1.0)]);
    let opts = crate::api::SimulateOptions {
        seed: Some(11),
        ..Default::default()
    };

    let a = crate::api::simulate_with_options(&weighted, &pop, &weighted.default_params, 1, &opts)
        .expect("weighted simulates");
    let b = crate::api::simulate_with_options(&plain, &pop, &plain.default_params, 1, &opts)
        .expect("unweighted simulates");
    assert_eq!(a.len(), b.len());
    assert!(
        !a.is_empty(),
        "the oracle is vacuous with no simulated rows"
    );
    assert!(
        a.iter().any(|r| r.ipred.abs() > 1e-9),
        "the oracle is vacuous if every prediction is zero"
    );
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(
            x.ipred.to_bits(),
            y.ipred.to_bits(),
            "weight = 1 must be bit-identical to no weight at TIME {}",
            x.time
        );
    }
}

/// …and the weight must actually *reach* the draw. Same seed, so the κ draws are
/// shared: a larger arm divides κ down, so the arm's individual CL sits closer to
/// its typical value and the spread of IPRED across occasions shrinks.
#[test]
fn a_larger_arm_size_shrinks_the_between_arm_spread() {
    let model = weighted_kappa_model();
    let spread = |n: f64| -> f64 {
        let pop = dosed_population_with_arm_sizes(&[(1, n), (2, n), (3, n)]);
        let opts = crate::api::SimulateOptions {
            seed: Some(11),
            ..Default::default()
        };
        let rows = crate::api::simulate_with_options(&model, &pop, &model.default_params, 1, &opts)
            .expect("simulates");
        let ipreds: Vec<f64> = rows.iter().map(|r| r.ipred).collect();
        let hi = ipreds.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lo = ipreds.iter().cloned().fold(f64::INFINITY, f64::min);
        hi - lo
    };
    let (small_arms, large_arms) = (spread(4.0), spread(400.0));
    assert!(
        large_arms < small_arms,
        "κ ~ N(0, γ²/N): larger arms must vary less, got {small_arms} then {large_arms}"
    );
}
