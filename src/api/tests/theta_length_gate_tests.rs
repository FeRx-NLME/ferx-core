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
//! The level-block half of the message (naming `bind_theta_levels_from_fit`) is pinned in
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
        assert_counts_message(&e, n, "simulate");
        let e = simulate_with_seed(&model, &pop, &params, 1, 5).expect_err("simulate_with_seed");
        assert_counts_message(&e, n, "simulate_with_seed");
        let opts = SimulateOptions {
            seed: Some(5),
            ..Default::default()
        };
        let e = simulate_with_options_diag(&model, &pop, &params, 1, &opts)
            .expect_err("simulate_with_options_diag");
        assert_counts_message(&e, n, "simulate_with_options_diag");
        let e = simulate_with_options(&model, &pop, &params, 1, &opts)
            .expect_err("simulate_with_options");
        assert_counts_message(&e, n, "simulate_with_options");
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
            assert_counts_message(&e, n, &format!("simulate_with_uncertainty ({draws} draws)"));
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
