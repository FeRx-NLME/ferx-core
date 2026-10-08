//! Tier-1 tests for the θ-length gate on every static `simulate*` entry (#1614).
//!
//! # What was measured on `main` (`a1cd1b5b`) before this
//!
//! A plain two-θ one-compartment IV model simulated with a θ vector of length 1 returned `Ok`
//! with every `ipred` exactly `0.0` — the bytecode VM reads a θ past the end as `0.0`. Length 3
//! returned `Ok` with the third value ignored; length 0 returned `Ok`, all zeros. No `simulate*`
//! entry compared θ against the model.
//!
//! # The contract pinned here
//!
//! | θ length | Must say | Must not say |
//! |---|---|---|
//! | short (1), long (3), empty (0) | both counts, "theta" | that the data is wrong, "missing" |
//! | correct (2) | nothing — rows bit-identical to the model's own θ | — |
//!
//! The level-block half of the message (naming the fit's level bindings, and no function since
//! #1623) is pinned in
//! `theta_levels_tests.rs`, next to the fixtures that bind a block; this file's plain model is
//! the side of that gate on which the hint must be absent. The two adaptive entries are pinned
//! in `adaptive_sim_tests.rs`, which owns their fixtures.
//!
//! # Engine
//!
//! Analytic one-compartment IV throughout; no gradient path is reached.

use super::*;
use crate::parser::model_parser::parse_model_string;
use crate::types::{DoseEvent, Population, Subject};

const ONE_CPT_IV: &str = "[parameters]\n  theta TVCL(1.0, 0.001, 100.0)\n  \
    theta TVV(20.0, 0.001, 500.0)\n  omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.1 (sd)\n\
    [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV\n[structural_model]\n  \
    pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(PROP)\n";

fn population() -> Population {
    let subjects = (0..2)
        .map(|i| Subject {
            id: format!("{}", i + 1),
            obs_times: vec![1.0, 4.0, 12.0],
            observations: vec![1.0, 0.8, 0.3],
            obs_cmts: vec![1; 3],
            cens: vec![0; 3],
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            ..Default::default()
        })
        .collect();
    Population {
        subjects,
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

/// `params` with θ resized to `n` — padded with `1.0` or truncated — and every θ-parallel
/// vector resized with it, so the only thing wrong with the result is its length against the
/// model (the draw machinery behind `simulate_with_uncertainty` packs all of them).
pub(super) fn with_theta_len(params: &ModelParameters, n: usize) -> ModelParameters {
    let mut p = params.clone();
    p.theta.resize(n, 1.0);
    let names: Vec<String> = (0..n)
        .map(|i| {
            params
                .theta_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("EXTRA{i}"))
        })
        .collect();
    p.theta_names = names;
    p.theta_lower.resize(n, 1e-3);
    p.theta_upper.resize(n, 1e3);
    p.theta_fixed.resize(n, false);
    p
}

/// The message for a θ of length `got` against the two-θ model, held to the contract table.
fn assert_counts_message(msg: &str, got: usize, entry: &str) {
    assert!(
        msg.contains(&format!(
            "the supplied theta has {got} values but this model has 2"
        )),
        "{entry}, θ len {got}: counts not named: {msg}"
    );
    assert!(
        msg.contains(
            "; the model reads theta by position, so these values would be read against the \
             wrong parameters"
        ),
        "{entry}, θ len {got}: why a length mismatch is refused, missing: {msg}"
    );
    for banned in ["missing", "data", "bind_theta_levels_from_fit"] {
        assert!(
            !msg.contains(banned),
            "{entry}, θ len {got}: {banned:?} in a plain-model θ-length message: {msg}"
        );
    }
}

/// Every static `simulate*` entry refuses a θ of the wrong length, short, long and empty.
///
/// Regression this catches: the silent all-zero / tail-dropping simulation measured above.
/// Mutation — delete `check_theta_length` from `check_simulate_preconditions` and the
/// `simulate`, `simulate_with_seed` and `simulate_with_options_diag` arms go `Ok` (each names
/// itself); `simulate_with_uncertainty` is pinned separately below, because it carries its own
/// up-front call.
#[test]
fn every_static_simulate_entry_refuses_a_theta_of_the_wrong_length() {
    let model = parse_model_string(ONE_CPT_IV).expect("parse");
    let pop = population();
    assert_eq!(model.default_params.theta.len(), 2);
    for n in [1usize, 3, 0] {
        let params = with_theta_len(&model.default_params, n);

        let e = simulate(&model, &pop, &params, 1).expect_err("simulate");
        assert_counts_message(&e.to_string(), n, "simulate");
        let e = simulate_with_seed(&model, &pop, &params, 1, 5).expect_err("simulate_with_seed");
        assert_counts_message(&e.to_string(), n, "simulate_with_seed");
        let opts = SimulateOptions {
            seed: Some(5),
            ..Default::default()
        };
        let e = simulate_with_options_diag(&model, &pop, &params, 1, &opts)
            .expect_err("simulate_with_options_diag");
        assert_counts_message(&e.to_string(), n, "simulate_with_options_diag");
        let e = simulate_with_options(&model, &pop, &params, 1, &opts)
            .expect_err("simulate_with_options");
        assert_counts_message(&e.to_string(), n, "simulate_with_options");
    }
}

/// `simulate_with_uncertainty` takes θ from a `FitResult` and refuses it before any draw.
///
/// Zero draws is the arm that pins the up-front call: with it deleted, the per-draw
/// chokepoint never runs and the run returns `Ok(empty)`. Two draws is the arm a caller hits;
/// it stays an `Err` with either call deleted, so it is a contract check, not a mutation probe.
#[test]
fn simulate_with_uncertainty_refuses_a_fit_theta_of_the_wrong_length() {
    let model = parse_model_string(ONE_CPT_IV).expect("parse");
    let pop = population();
    for n in [1usize, 3] {
        let fit_result = super::simulate_with_uncertainty_tests::synthetic_fit(&with_theta_len(
            &model.default_params,
            n,
        ));
        for draws in [0usize, 2] {
            let opts = SimulateUncertaintyOptions {
                n_uncertainty_draws: draws,
                n_sim_per_draw: 1,
                seed: Some(3),
                ..Default::default()
            };
            let e = simulate_with_uncertainty(&model, &pop, &fit_result, &opts)
                .expect_err("simulate_with_uncertainty");
            assert_counts_message(
                &e.to_string(),
                n,
                &format!("simulate_with_uncertainty ({draws} draws)"),
            );
        }
    }
}

/// The other side of the gate: a θ of the right length is let through and simulates exactly
/// what it did before — the same rows, bit for bit, as the model's own θ, which is the vector
/// every existing caller passes. A gate comparing against the wrong count (`n_theta` of a
/// differently bound model, an off-by-one) would refuse this.
///
/// The `to_bits` pairing is against a *copy* routed through `with_theta_len(.., 2)`, so a
/// gate that perturbed θ on the way through (rather than only reading its length) also dies.
#[test]
fn a_theta_of_the_right_length_simulates_bit_identically() {
    let model = parse_model_string(ONE_CPT_IV).expect("parse");
    let pop = population();
    let same = with_theta_len(&model.default_params, 2);
    assert_eq!(same.theta, model.default_params.theta);
    let a = simulate_with_seed(&model, &pop, &model.default_params, 2, 9).expect("accepted");
    let b = simulate_with_seed(&model, &pop, &same, 2, 9).expect("accepted");
    assert_eq!(a.len(), 12);
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(&b) {
        assert!(x.ipred.is_finite() && x.ipred > 0.0, "{}", x.ipred);
        assert_eq!(x.ipred.to_bits(), y.ipred.to_bits());
        let (SimOutcome::Continuous { value: vx }, SimOutcome::Continuous { value: vy }) =
            (&x.outcome, &y.outcome)
        else {
            panic!("continuous rows expected");
        };
        assert_eq!(vx.to_bits(), vy.to_bits());
    }
}

// ── the predict-side entry points (#1615) ────────────────────────────────────

/// Every predict-side entry point taking a parameter vector refuses a θ of the wrong length,
/// short, long and empty, with the same message the simulate paths give (#1615).
///
/// Measured on `main` (`a1cd1b5b`, the issue's probe): `predict` with a one-value θ returned
/// `Ok` with a first `pred` of exactly `0.0`; with three values the third was ignored.
///
/// Mutation — delete `check_theta_length` from any one entry point and that arm goes `Ok`,
/// naming itself: `predict_diag` (which also reddens `predict`, its wrapper) and
/// `compute_npde_npd` here; `predict_survival` and `predict_categorical` in
/// `the_survival_predictors_refuse_a_theta_of_the_wrong_length`, which runs them on a model
/// with their endpoint, so the refusal is not an empty-result fast path.
#[test]
fn every_predict_entry_refuses_a_theta_of_the_wrong_length() {
    let model = parse_model_string(ONE_CPT_IV).expect("parse");
    let pop = population();
    for n in [1usize, 3, 0] {
        let params = with_theta_len(&model.default_params, n);
        let e = predict(&model, &pop, &params).expect_err("predict");
        assert_counts_message(&e.to_string(), n, "predict");
        let e = predict_diag(&model, &pop, &params).expect_err("predict_diag");
        assert_counts_message(&e.to_string(), n, "predict_diag");
        let e = crate::stats::npde::compute_npde_npd(&model, &pop, &params, 20, Some(4))
            .expect_err("compute_npde_npd");
        assert_counts_message(&e.to_string(), n, "compute_npde_npd");
    }
}

/// The other side of the predict gate: a θ of the right length predicts exactly what it did
/// before, bit for bit — rows from `predict_diag`, and npd/npde from `compute_npde_npd` —
/// against a copy routed through `with_theta_len(.., 2)`, so a gate that perturbed θ (rather
/// than only reading its length) dies too, as does one comparing against the wrong count.
#[test]
fn a_theta_of_the_right_length_predicts_bit_identically() {
    let model = parse_model_string(ONE_CPT_IV).expect("parse");
    let pop = population();
    let same = with_theta_len(&model.default_params, 2);
    assert_eq!(same.theta, model.default_params.theta);
    let a = predict_diag(&model, &pop, &model.default_params).expect("accepted");
    let b = predict_diag(&model, &pop, &same).expect("accepted");
    assert_eq!(a.results.len(), 6);
    assert_eq!(a.results.len(), b.results.len());
    for (x, y) in a.results.iter().zip(&b.results) {
        assert!(x.pred.is_finite() && x.pred > 0.0, "{}", x.pred);
        assert_eq!(x.pred.to_bits(), y.pred.to_bits());
    }
    let na = crate::stats::npde::compute_npde_npd(&model, &pop, &model.default_params, 20, Some(4))
        .expect("accepted");
    let nb = crate::stats::npde::compute_npde_npd(&model, &pop, &same, 20, Some(4)).expect("ok");
    assert_eq!(na.len(), 2);
    for (x, y) in na.iter().zip(&nb) {
        assert_eq!(x.npd.len(), 3);
        for (p, q) in x.npd.iter().zip(&y.npd) {
            assert!(p.is_finite(), "{p}");
            assert_eq!(p.to_bits(), q.to_bits());
        }
    }
}

/// `predict_survival` and `predict_categorical`, on real fixtures carrying their endpoint
/// (`examples/pktte_joint.ferx`, `examples/binary_logistic.ferx`, each read model-aware),
/// refuse a θ of the wrong length and predict a θ of the right length bit-identically.
///
/// The fixtures matter: on a model without the endpoint both return `Ok(empty)`, so a gate
/// test there would hold only for a check placed before the empty result, and the control
/// would compare two empty vecs. Here the control is non-empty and finite.
///
/// Mutation — delete `check_theta_length` from `predict_survival` (resp.
/// `predict_categorical`) and its arm goes `Ok`, naming itself.
#[cfg(feature = "survival")]
#[test]
fn the_survival_predictors_refuse_a_theta_of_the_wrong_length() {
    use crate::parser::model_parser::parse_full_model;
    let read = |path: &str, data: &str| {
        let src = std::fs::read_to_string(path).expect("example");
        let m = parse_full_model(&src).expect("parses").model;
        let pop = crate::api::read_population_for(&m, &None, data, None, None, None, &[])
            .expect("routed read")
            .0;
        (m, pop)
    };
    let counts = |msg: &str, got: usize, want: usize, entry: &str| {
        assert!(
            msg.contains(&format!(
                "the supplied theta has {got} values but this model has {want}; the model \
                 reads theta by position"
            )),
            "{entry}, θ len {got}: {msg}"
        );
    };

    let (tte, tte_pop) = read("examples/pktte_joint.ferx", "data/pktte_joint.csv");
    let k = tte.default_params.theta.len();
    let grid = [1.0, 10.0, 50.0];
    for n in [k - 1, k + 1] {
        let e = predict_survival(
            &tte,
            &tte_pop,
            &with_theta_len(&tte.default_params, n),
            &grid,
        )
        .expect_err("predict_survival");
        counts(&e.to_string(), n, k, "predict_survival");
    }
    let a = predict_survival(&tte, &tte_pop, &tte.default_params, &grid).expect("accepted");
    let b = predict_survival(
        &tte,
        &tte_pop,
        &with_theta_len(&tte.default_params, k),
        &grid,
    )
    .expect("accepted");
    assert!(!a.is_empty());
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(&b) {
        assert!(x.survival.is_finite() && x.survival < 1.0, "{}", x.survival);
        assert_eq!(x.survival.to_bits(), y.survival.to_bits());
        assert_eq!(x.cum_hazard.to_bits(), y.cum_hazard.to_bits());
    }

    let (bin, bin_pop) = read("examples/binary_logistic.ferx", "data/binary_logistic.csv");
    let k = bin.default_params.theta.len();
    for n in [k - 1, k + 1] {
        let e = predict_categorical(&bin, &bin_pop, &with_theta_len(&bin.default_params, n))
            .expect_err("predict_categorical");
        counts(&e.to_string(), n, k, "predict_categorical");
    }
    // Off the all-zero initial estimates, so a misread θ would move every probability.
    let mut params = bin.default_params.clone();
    params.theta = vec![-0.4, 0.9, 0.5];
    let a = predict_categorical(&bin, &bin_pop, &params).expect("accepted");
    let b = predict_categorical(&bin, &bin_pop, &with_theta_len(&params, k)).expect("accepted");
    assert!(!a.is_empty());
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(&b) {
        let (
            crate::types::Prediction::CatProbs { probs: p },
            crate::types::Prediction::CatProbs { probs: q },
        ) = (&x.prediction, &y.prediction)
        else {
            panic!("category probabilities expected");
        };
        assert_eq!(p.len(), 2);
        for (u, v) in p.iter().zip(q) {
            assert!(u.is_finite() && *u > 0.0 && *u < 1.0, "{u}");
            assert_eq!(u.to_bits(), v.to_bits());
        }
    }
}

// ── #1763: the survival predictors and npde run predict_diag's level and covariate checks ──

/// `examples/<name>` with `edits` applied, parsed, and the population of `data` read
/// model-aware with the **original** model — `data` carries no column an edit adds.
#[cfg(feature = "survival")]
fn edited_example(
    name: &str,
    data: &str,
    edits: &[(&str, &str)],
) -> (String, crate::types::ParsedModel, Population) {
    use crate::parser::model_parser::parse_full_model;
    let src = std::fs::read_to_string(format!("examples/{name}")).expect("example");
    let original = parse_full_model(&src).expect("parses").model;
    let pop = crate::api::read_population_for(&original, &None, data, None, None, None, &[])
        .expect("routed read")
        .0;
    let mut text = src;
    for (from, to) in edits {
        assert!(text.contains(from), "{name}: `{from}` not found");
        text = text.replace(from, to);
    }
    let parsed = parse_full_model(&text).expect("edited example parses");
    (text, parsed, pop)
}

/// `STUDY = i % 3 + 1` on subject `i`, on the baseline and on every snapshot.
#[cfg(feature = "survival")]
fn with_study(mut pop: Population) -> Population {
    for (i, s) in pop.subjects.iter_mut().enumerate() {
        let v = (i % 3 + 1) as f64;
        s.covariates.insert("STUDY".to_string(), v);
        for m in s
            .obs_covariates
            .iter_mut()
            .chain(s.dose_covariates.iter_mut())
        {
            m.insert("STUDY".to_string(), v);
        }
    }
    pop.covariate_names.push("STUDY".to_string());
    pop
}

/// θ with every level effect of `EFF` moved off zero, so a misread level moves a value.
#[cfg(feature = "survival")]
fn eff_moved(model: &CompiledModel) -> ModelParameters {
    let mut params = model.default_params.clone();
    for (k, (name, t)) in model
        .theta_names
        .iter()
        .zip(params.theta.iter_mut())
        .enumerate()
    {
        if name.starts_with("EFF") {
            *t = 0.25 * k as f64 - 0.6;
        }
    }
    params
}

/// The unbound-block refusal, naming `entry` and no other entry (#1763).
fn assert_unbound_names(err: &crate::diagnostics::EngineError, entry: &str) {
    assert_eq!(err.code(), Some("E_THETA_LEVELS_UNBOUND"), "{entry}: {err}");
    let text = err.to_string();
    assert!(
        text.contains(&format!("on the population you pass to `{entry}`")),
        "{entry}: the population the entry reads: {text}"
    );
    assert!(
        text.contains(&format!(
            "and call `{entry}` with the model it re-parses into `parsed`."
        )),
        "{entry}: the model to call it with: {text}"
    );
    assert!(
        !text.contains("`predict`"),
        "{entry}: names `predict`: {text}"
    );
}

/// #1763, Q1–Q4b. `predict_survival` and `predict_categorical` returned `Ok` on three inputs
/// `predict_diag` refuses (measured at `cfc84253`): a covariate the data lacks (read as 0.0 —
/// categorical probabilities 0.599/0.401 instead of 0.363/0.637; every survival `S = 1.0`
/// under `H0 = TVH0 * WTX`), a bound level block on a population never bound for it, and an
/// unbound model (both: every value `NaN`). Each fixture carries the entry's endpoint, so no
/// refusal is an empty-result fast path, and each bound control is non-empty and finite.
///
/// Mutations — delete any one added call in either function: the matching arm goes `Ok` or
/// takes another code, naming itself (`unbound_level_refusal`: the covariate check then names
/// `__level_EFF`, `E_MISSING_COVARIATE`, not `E_THETA_LEVELS_UNBOUND`); name `predict` in the
/// unbound suggestion again: `assert_unbound_names` dies.
#[cfg(feature = "survival")]
#[test]
fn the_survival_predictors_refuse_what_predict_diag_refuses() {
    let grid = [1.0, 10.0, 50.0];

    // Q1: `binary_logistic` without its covariate `X`.
    let (_, bin, mut pop) = edited_example("binary_logistic.ferx", "data/binary_logistic.csv", &[]);
    for s in &mut pop.subjects {
        s.covariates.remove("X");
        for m in &mut s.obs_covariates {
            m.remove("X");
        }
    }
    pop.covariate_names.retain(|c| c != "X");
    let e = predict_categorical(&bin.model, &pop, &bin.model.default_params)
        .expect_err("predict_categorical, no X");
    assert_eq!(e.code(), Some("E_MISSING_COVARIATE"), "{e}");
    assert!(e.to_string().contains("(case-sensitive): X."), "{e}");

    // Q2: `pktte_joint` with `H0 = TVH0 * WTX` and no `WTX`.
    let (_, tte, pop) = edited_example(
        "pktte_joint.ferx",
        "data/pktte_joint.csv",
        &[("H0   = TVH0", "H0   = TVH0 * WTX")],
    );
    let e = predict_survival(&tte.model, &pop, &tte.model.default_params, &grid)
        .expect_err("predict_survival, no WTX");
    assert_eq!(e.code(), Some("E_MISSING_COVARIATE"), "{e}");
    assert!(e.to_string().contains("(case-sensitive): WTX."), "{e}");

    // Q3 / Q3b: a level block on the hazard.
    let (text, unbound, pop) = edited_example(
        "pktte_joint.ferx",
        "data/pktte_joint.csv",
        &[
            (
                "theta TVBETA(0.5, -10.0, 10.0)",
                "theta TVBETA(0.5, -10.0, 10.0)\n  theta EFF[STUDY](0.0, -5.0, 5.0)",
            ),
            ("H0   = TVH0", "H0   = TVH0 * exp(EFF)"),
        ],
    );
    let never = with_study(pop);
    let e = predict_survival(&unbound.model, &never, &unbound.model.default_params, &grid)
        .expect_err("predict_survival, unbound model");
    assert_unbound_names(&e, "predict_survival");
    let mut bound = crate::parser::model_parser::parse_full_model(&text).expect("parse");
    let mut bound_pop = never.clone();
    crate::api::bind_theta_levels(&mut bound, &text, &mut bound_pop).expect("bind");
    let params = eff_moved(&bound.model);
    let e = predict_survival(&bound.model, &never, &params, &grid)
        .expect_err("predict_survival, never-bound population");
    assert_eq!(e.code(), Some("E_THETA_LEVELS_DATA_UNBOUND"), "{e}");
    let rows = predict_survival(&bound.model, &bound_pop, &params, &grid).expect("bound control");
    assert!(!rows.is_empty());
    for r in &rows {
        assert!(r.survival.is_finite() && r.survival < 1.0, "{}", r.survival);
    }

    // Q4 / Q4b: a Gaussian + binary model whose log-odds carry a level block.
    let dir = tempfile::tempdir().expect("tempdir");
    let csv = dir.path().join("mixed.csv");
    let mut body = String::from("ID,TIME,DV,EVID,AMT,CMT,MDV,STUDY\n");
    for i in 1..=6 {
        let study = i % 3 + 1;
        body.push_str(&format!("{i},0,.,1,100,1,1,{study}\n"));
        for t in [1, 2, 4] {
            let conc = 10.0 * (-0.2 * t as f64).exp();
            body.push_str(&format!("{i},{t},{conc},0,.,1,0,{study}\n"));
            body.push_str(&format!("{i},{t},{},0,.,3,0,{study}\n", (i + t) % 2));
        }
    }
    std::fs::write(&csv, body).expect("write csv");
    let mixed = "[parameters]\n  theta TVCL(1.0, 0.01, 100.0)\n  theta TVV(10.0, 0.1, 500.0)\n  \
        theta TH0(-0.4, -10.0, 10.0)\n  theta EFF[STUDY](0.0, -5.0, 5.0)\n  \
        omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.1 (sd)\n[individual_parameters]\n  \
        CL = TVCL * exp(ETA_CL)\n  V  = TVV\n  LO = TH0 + EFF\n[structural_model]\n  \
        pk one_cpt_iv(cl=CL, v=V)\n[binary_model]\n  cmt   = 3\n  logit = LO\n\
        [error_model]\n  DV ~ proportional(PROP)\n";
    let unbound = crate::parser::model_parser::parse_full_model(mixed).expect("parse");
    let never = crate::api::read_population_for(
        &unbound.model,
        &None,
        csv.to_str().unwrap(),
        None,
        None,
        None,
        &[],
    )
    .expect("routed read")
    .0;
    let e = predict_categorical(&unbound.model, &never, &unbound.model.default_params)
        .expect_err("predict_categorical, unbound model");
    assert_unbound_names(&e, "predict_categorical");
    let mut bound = crate::parser::model_parser::parse_full_model(mixed).expect("parse");
    let mut bound_pop = never.clone();
    crate::api::bind_theta_levels(&mut bound, mixed, &mut bound_pop).expect("bind");
    let params = eff_moved(&bound.model);
    let e = predict_categorical(&bound.model, &never, &params)
        .expect_err("predict_categorical, never-bound population");
    assert_eq!(e.code(), Some("E_THETA_LEVELS_DATA_UNBOUND"), "{e}");
    let rows = predict_categorical(&bound.model, &bound_pop, &params).expect("bound control");
    assert_eq!(rows.len(), 18, "6 subjects x 3 binary records");
    for r in &rows {
        let crate::types::Prediction::CatProbs { probs } = &r.prediction else {
            panic!("category probabilities expected");
        };
        for p in probs {
            assert!(p.is_finite() && *p > 0.0 && *p < 1.0, "{p}");
        }
    }
}

/// #1763, R1 and an unbound model. `compute_npde_npd` ran no covariate check (a missing `WTX`
/// read as 0.0 and returned `Ok`, npd ±1.96) and no unbound-block check (every npd `NaN`).
///
/// Mutations — delete `check_covariates` from it: the `WTX` arm goes `Ok`; delete
/// `unbound_level_refusal`: the unbound arm takes `E_MISSING_COVARIATE` on `__level_EFF`.
#[test]
fn compute_npde_npd_refuses_a_missing_covariate_and_an_unbound_block() {
    let model =
        parse_model_string(&ONE_CPT_IV.replace("CL = TVCL *", "CL = TVCL * WTX *")).expect("parse");
    let e = crate::stats::npde::compute_npde_npd(
        &model,
        &population(),
        &model.default_params,
        20,
        Some(4),
    )
    .expect_err("compute_npde_npd, no WTX");
    assert_eq!(e.code(), Some("E_MISSING_COVARIATE"), "{e}");
    assert!(e.to_string().contains("(case-sensitive): WTX."), "{e}");

    let levels = crate::parser::model_parser::parse_full_model(
        &ONE_CPT_IV
            .replace(
                "theta TVV(20.0, 0.001, 500.0)",
                "theta TVV(20.0, 0.001, 500.0)\n  theta EFF[STUDY](0.0, -5.0, 5.0)",
            )
            .replace("V  = TVV", "V  = TVV * exp(EFF)"),
    )
    .expect("parse")
    .model;
    assert!(!levels.theta_blocks().unbound_level_blocks().is_empty());
    let mut pop = population();
    for s in &mut pop.subjects {
        s.covariates.insert("STUDY".to_string(), 1.0);
    }
    pop.covariate_names.push("STUDY".to_string());
    let e =
        crate::stats::npde::compute_npde_npd(&levels, &pop, &levels.default_params, 20, Some(4))
            .expect_err("compute_npde_npd, unbound model");
    assert_unbound_names(&e, "compute_npde_npd");
}
