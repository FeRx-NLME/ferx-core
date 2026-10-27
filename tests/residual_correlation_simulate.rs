//! `simulate()` draws a `correlation cs|ar1|car1(RHO) by COL` model's residuals from the
//! dense `R` (#1081, T9).
//!
//! A model with no random effects and a constant prediction, so `DV − IPRED` is the
//! residual draw itself. One subject, two arms of three rows each, interleaved, so
//! a row's AR(1) lag is its distance within its arm, not in the file; the visits are
//! irregular (Δt 2, 3, 5 and 2.5, 5, 7.5), so `car1`'s `ρ^Δt` is not `ar1`'s `ρ^lag`. The population is handed to `simulate()` as built — no
//! residual blocks assigned — so the draw has to read them off the `ARM` column.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{simulate_with_seed, Population, SimOutcome, Subject};
use std::collections::HashMap;

/// (arm, time) per row.
const ROWS: [(f64, f64); 6] = [
    (1.0, 0.0),
    (2.0, 0.5),
    (1.0, 2.0),
    (2.0, 3.0),
    (1.0, 5.0),
    (2.0, 8.0),
];
/// Position within the arm in time order, by hand.
const POS: [i32; 6] = [0, 0, 1, 1, 2, 2];
const N_SIM: usize = 20_000;
const RHO: f64 = 0.6;

fn model(kind: &str) -> ferx_core::CompiledModel {
    let lower = if kind == "car1" { 0.0 } else { -0.4 };
    parse_model_string(&format!(
        "[parameters]\n  theta TVY(10.0, 1.0, 100.0)\n  theta RHO({RHO}, {lower}, 0.95)\n  sigma ADD_ERR ~ 1.0 (sd)\n[individual_parameters]\n  Y0 = TVY\n[structural_model]\n  y = Y0\n[error_model]\n  DV ~ additive(ADD_ERR)\n  correlation {kind}(RHO) by ARM\n"
    ))
    .expect("model parses")
}

fn population() -> Population {
    let subject = Subject {
        id: "1".into(),
        obs_times: ROWS.iter().map(|r| r.1).collect(),
        observations: vec![10.0; 6],
        obs_cmts: vec![1; 6],
        cens: vec![0; 6],
        covariates: HashMap::from([("ARM".to_string(), 1.0)]),
        obs_covariates: ROWS
            .iter()
            .map(|r| HashMap::from([("ARM".to_string(), r.0)]))
            .collect(),
        ..Default::default()
    };
    Population {
        subjects: vec![subject],
        covariate_names: vec!["ARM".into()],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

/// Sample correlation of rows `j` and `k` over the replicates.
fn corr(eps: &[[f64; 6]], j: usize, k: usize) -> f64 {
    let n = eps.len() as f64;
    let mean = |c: usize| eps.iter().map(|e| e[c]).sum::<f64>() / n;
    let (mj, mk) = (mean(j), mean(k));
    let cov = eps.iter().map(|e| (e[j] - mj) * (e[k] - mk)).sum::<f64>() / n;
    let var = |c: usize, m: f64| eps.iter().map(|e| (e[c] - m).powi(2)).sum::<f64>() / n;
    cov / (var(j, mj) * var(k, mk)).sqrt()
}

/// Every pair's sample correlation is within 4 standard errors of `C_jk` — `ρ`
/// (cs), `ρ^lag` (ar1, lag in time order) or `ρ^|Δt|` (car1) within an arm, 0 across
/// arms. Dies on an independent draw, a draw that ignores the arm, an AR(1) lag
/// counted in file rows rather than within the arm, and a `car1` lag counted in
/// visits rather than time.
#[test]
fn simulate_draws_the_serial_residual_correlation() {
    for kind in ["cs", "ar1", "car1"] {
        let model = model(kind);
        let sims = simulate_with_seed(&model, &population(), &model.default_params, N_SIM, 1081)
            .expect("simulate runs");
        let mut eps = vec![[0.0; 6]; N_SIM];
        for r in &sims {
            let SimOutcome::Continuous { value } = r.outcome else {
                panic!("a Gaussian row");
            };
            let j = ROWS
                .iter()
                .position(|row| row.1 == r.time)
                .expect("a fixture row");
            eps[r.sim - 1][j] = value - r.ipred;
        }
        assert!(eps.iter().flatten().all(|v| v.is_finite()));
        let mut live = 0;
        for j in 0..6 {
            for k in (j + 1)..6 {
                let want = if ROWS[j].0 != ROWS[k].0 {
                    0.0
                } else if kind == "cs" {
                    RHO
                } else if kind == "car1" {
                    RHO.powf((ROWS[j].1 - ROWS[k].1).abs())
                } else {
                    RHO.powi((POS[j] - POS[k]).abs())
                };
                let got = corr(&eps, j, k);
                // SE of a sample correlation ≈ (1 − ρ²)/√n.
                let se = (1.0 - want * want) / (N_SIM as f64).sqrt();
                assert!(
                    (got - want).abs() < 4.0 * se,
                    "{kind} rows {j},{k}: sample correlation {got:.4} vs {want:.4} (4·SE {:.4})",
                    4.0 * se
                );
                if want != 0.0 {
                    live += 1;
                }
            }
        }
        assert_eq!(live, 6, "two arms of three rows: six within-arm pairs");
    }
}

/// #1974 review r1 #1: `simulate()` runs no `fit()` validation, and `ρ^Δt` is NaN for
/// a negative `ρ` at a non-integer gap (arm 2 here: 2.5 and 5). Before the fix a `car1`
/// ρ of −0.2 returned `Ok` with NaN DVs in every row but the first. Both sides: −0.2
/// is refused with the range message, 0.3 simulates finite rows.
#[test]
fn simulate_refuses_a_negative_car1_rho() {
    let at = |rho: &str| {
        let m = parse_model_string(&format!(
            "[parameters]\n  theta TVY(10.0, 1.0, 100.0)\n  theta RHO({rho}) FIX\n  sigma ADD_ERR ~ 1.0 (sd)\n[individual_parameters]\n  Y0 = TVY\n[structural_model]\n  y = Y0\n[error_model]\n  DV ~ additive(ADD_ERR)\n  correlation car1(RHO) by ARM\n"
        ))
        .expect("model parses");
        simulate_with_seed(&m, &population(), &m.default_params, 2, 7)
    };
    let err = at("-0.2").expect_err("a negative car1 ρ is refused");
    assert_eq!(err.code(), Some("E_RESIDUAL_CORRELATION_RANGE"), "{err}");
    let msg = err.to_string();
    assert!(
        msg.contains(
            "`correlation car1(RHO) by ARM`: theta RHO is -0.2, but `car1` correlates two rows \
             by ρ^Δt, and ρ^Δt needs ρ ≥ 0 for a non-integer time gap Δt."
        ),
        "{msg}"
    );
    assert!(msg.contains("For a negative correlation between consecutive rows, use `ar1`."));
    let ok = at("0.3").expect("a positive car1 ρ simulates");
    assert_eq!(ok.len(), 12);
    for r in &ok {
        let SimOutcome::Continuous { value } = r.outcome else {
            panic!("a Gaussian row");
        };
        assert!(value.is_finite(), "row at TIME {}: {value}", r.time);
    }
}

// ── #1976: simulate refuses every ρ and every block fit() refuses ────────────

/// A `kind` model with ρ `FIX`ed at `rho`, so the value simulated at is `rho` whatever
/// its bounds.
fn fixed_model(kind: &str, rho: &str) -> ferx_core::CompiledModel {
    parse_model_string(&format!(
        "[parameters]\n  theta TVY(10.0, 1.0, 100.0)\n  theta RHO({rho}) FIX\n  sigma ADD_ERR ~ 1.0 (sd)\n[individual_parameters]\n  Y0 = TVY\n[structural_model]\n  y = Y0\n[error_model]\n  DV ~ additive(ADD_ERR)\n  correlation {kind}(RHO) by ARM\n"
    ))
    .expect("model parses")
}

/// One subject with rows `(ARM, TIME)`.
fn population_of(rows: &[(f64, f64)]) -> Population {
    let mut pop = population();
    let s = &mut pop.subjects[0];
    s.obs_times = rows.iter().map(|r| r.1).collect();
    s.observations = vec![10.0; rows.len()];
    s.obs_cmts = vec![1; rows.len()];
    s.cens = vec![0; rows.len()];
    s.obs_covariates = rows
        .iter()
        .map(|r| HashMap::from([("ARM".to_string(), r.0)]))
        .collect();
    pop
}

/// `simulate_with_seed(…, 2, 7)` succeeded with two replicates of finite rows.
fn assert_finite_rows(
    what: &str,
    sims: Result<Vec<ferx_core::SimulationResult>, ferx_core::EngineError>,
    n_rows: usize,
) {
    let sims = sims.unwrap_or_else(|e| panic!("{what}: simulates, got {e}"));
    assert_eq!(sims.len(), 2 * n_rows, "{what}");
    for r in &sims {
        let SimOutcome::Continuous { value } = r.outcome else {
            panic!("a Gaussian row");
        };
        assert!(value.is_finite(), "{what}: row at TIME {}: {value}", r.time);
    }
}

/// The error of a refused simulation, with its code checked.
fn refused(
    what: &str,
    sims: Result<Vec<ferx_core::SimulationResult>, ferx_core::EngineError>,
    code: &str,
) -> String {
    let err = sims.expect_err(what);
    assert_eq!(err.code(), Some(code), "{what}: {err}");
    err.to_string()
}

fn sim_at(
    kind: &str,
    rho: &str,
    pop: &Population,
) -> Result<Vec<ferx_core::SimulationResult>, ferx_core::EngineError> {
    let m = fixed_model(kind, rho);
    simulate_with_seed(&m, pop, &m.default_params, 2, 7)
}

/// #1976: `ar1` and `car1` at |ρ| ≥ 1. Before the fix ρ = 1.0 and 1.2 returned `Ok`
/// with finite rows drawn from the clamped eigen square root of an indefinite (or
/// singular) `R` — a matrix the model does not state. Refused with `fit()`'s `FIX`
/// message; a ρ just inside the edge simulates finite rows. A `car1` ρ above 1 is
/// refused for its magnitude, not by the negative-ρ rule.
#[test]
fn simulate_refuses_a_serial_rho_at_or_past_one() {
    let pop = population();
    for (kind, rhos, inside) in [
        ("car1", &["1.0", "1.2"][..], &["0.0", "0.9"][..]),
        (
            "ar1",
            &["1.0", "1.2", "-1.0", "-1.5"][..],
            &["-0.9", "0.9"][..],
        ),
    ] {
        for rho in rhos {
            let msg = refused(
                &format!("{kind} at {rho}"),
                sim_at(kind, rho, &pop),
                "E_RESIDUAL_CORRELATION_RANGE",
            );
            let v: f64 = rho.parse().unwrap();
            assert!(
                msg.contains(&format!(
                    "`correlation {kind}(RHO) by ARM`: theta RHO is {v}; a correlation must lie \
                     strictly between −1 and 1."
                )),
                "{msg}"
            );
            if kind == "car1" {
                assert!(!msg.contains("ρ^Δt needs ρ ≥ 0"), "{msg}");
            }
        }
        for rho in inside {
            assert_finite_rows(&format!("{kind} at {rho}"), sim_at(kind, rho, &pop), 6);
        }
    }
}

/// #1976: `cs` over the largest block of `n = 3` rows needs ρ in (−1/2, 1). At −0.5
/// (on the floor) and −0.7 it is refused with the block and the bound, every sentence
/// asserted; at 1.0 with the |ρ| < 1 message; at −0.45 and 0.9 it simulates finite rows.
#[test]
fn simulate_refuses_a_cs_rho_outside_the_block_range() {
    let pop = population();
    for rho in ["-0.5", "-0.7"] {
        let msg = refused(
            &format!("cs at {rho}"),
            sim_at("cs", rho, &pop),
            "E_RESIDUAL_CORRELATION_RANGE",
        );
        let v: f64 = rho.parse().unwrap();
        assert!(
            msg.contains(&format!(
                "`correlation cs(RHO) by ARM`: theta RHO is {v}, but subject 1 has 3 rows in the \
                 block ARM=1, and compound symmetry over 3 rows is a valid covariance only for ρ \
                 in (−1/(3−1), 1) = (-0.5000, 1)."
            )),
            "{msg}"
        );
        assert!(msg.contains("Keep ρ above -0.5000."), "{msg}");
    }
    let msg = refused(
        "cs at 1.0",
        sim_at("cs", "1.0", &pop),
        "E_RESIDUAL_CORRELATION_RANGE",
    );
    assert!(
        msg.contains(
            "`correlation cs(RHO) by ARM`: theta RHO is 1; a correlation must lie strictly \
             between −1 and 1."
        ),
        "{msg}"
    );
    for rho in ["-0.45", "0.9"] {
        assert_finite_rows(&format!("cs at {rho}"), sim_at("cs", rho, &pop), 6);
    }
}

/// #1976: tied TIMEs within one block under `ar1` / `car1` are refused by `simulate()`
/// as by `fit()`, with each kind's reason and every sentence asserted. Both sides: the
/// same tie split across two blocks simulates finite rows, and `cs` (no time read)
/// simulates the tied block.
#[test]
fn simulate_refuses_tied_times_within_a_block() {
    let tied = population_of(&[(1.0, 0.0), (1.0, 2.0), (1.0, 2.0)]);
    let split = population_of(&[(1.0, 0.0), (1.0, 2.0), (2.0, 2.0)]);
    for (kind, why) in [
        (
            "ar1",
            "`ar1` orders a block's rows by TIME, so tied rows have no order and no lag between \
             them.",
        ),
        (
            "car1",
            "`car1` correlates two rows by ρ^|Δt|, so tied rows correlate perfectly (ρ^0 = 1) \
             and R is singular.",
        ),
    ] {
        let msg = refused(
            kind,
            sim_at(kind, "0.3", &tied),
            "E_RESIDUAL_CORRELATION_TIES",
        );
        assert!(
            msg.contains(&format!(
                "`correlation {kind}(RHO) by ARM`: subject 1 has two rows at TIME 2 in the block \
                 ARM=1."
            )),
            "{msg}"
        );
        assert!(msg.contains(why), "{msg}");
        assert!(
            msg.contains("Give them distinct times, or use `cs`."),
            "{msg}"
        );
        assert_finite_rows(&format!("{kind} split"), sim_at(kind, "0.3", &split), 3);
    }
    assert_finite_rows("cs tied", sim_at("cs", "0.3", &tied), 3);
}

/// #1976: a block spanning a stacked reset occasion (written TIME 0, 2, 12 then 1, 3,
/// 13, shifted 100 later on the engine clock, as the datareader lays an EVID=3/4
/// TIME restart out) is refused under `ar1` / `car1`, every sentence asserted. Both
/// sides: ARM splitting the occasions simulates finite rows, and so does `cs`.
#[test]
fn simulate_refuses_a_block_across_a_time_reset() {
    let pop_with = |arms: [f64; 6]| {
        let mut pop = population_of(&arms.map(|a| (a, 0.0)));
        pop.subjects[0].obs_times = vec![0.0, 2.0, 12.0, 113.0, 115.0, 125.0];
        pop.subjects[0].obs_raw_times = vec![0.0, 2.0, 12.0, 1.0, 3.0, 13.0];
        pop
    };
    let spanning = pop_with([1.0; 6]);
    let split = pop_with([1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    for kind in ["ar1", "car1"] {
        let msg = refused(
            kind,
            sim_at(kind, "0.3", &spanning),
            "E_RESIDUAL_CORRELATION_RESET",
        );
        assert!(
            msg.contains(&format!(
                "`correlation {kind}(RHO) by ARM`: subject 1 has rows at TIME 0 and TIME 1 in \
                 the block ARM=1 on either side of a TIME reset (an EVID=3/4 record that \
                 restarts TIME)."
            )),
            "{msg}"
        );
        assert!(
            msg.contains(&format!(
                "`{kind}` reads the time between two rows of a block, and across a reset it is \
                 undefined."
            )),
            "{msg}"
        );
        assert!(
            msg.contains(
                "Add the occasion column to `by` so that each occasion is its own block, or use \
                 `cs`."
            ),
            "{msg}"
        );
        assert_finite_rows(&format!("{kind} split"), sim_at(kind, "0.3", &split), 6);
    }
    assert_finite_rows("cs spanning", sim_at("cs", "0.3", &spanning), 6);
}
