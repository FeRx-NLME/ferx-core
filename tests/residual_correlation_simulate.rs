//! `simulate()` draws a `correlation cs|ar1(RHO) by COL` model's residuals from the
//! dense `R` (#1081, T9).
//!
//! A model with no random effects and a constant prediction, so `DV − IPRED` is the
//! residual draw itself. One subject, two arms of three rows each, interleaved, so
//! a row's AR(1) lag is its distance within its arm, not in the file. The population is handed to `simulate()` as built — no
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
    parse_model_string(&format!(
        "[parameters]\n  theta TVY(10.0, 1.0, 100.0)\n  theta RHO({RHO}, -0.4, 0.95)\n  sigma ADD_ERR ~ 1.0 (sd)\n[individual_parameters]\n  Y0 = TVY\n[structural_model]\n  y = Y0\n[error_model]\n  DV ~ additive(ADD_ERR)\n  correlation {kind}(RHO) by ARM\n"
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
/// (cs) or `ρ^lag` (ar1, lag in time order) within an arm, 0 across arms. Dies on
/// an independent draw, a draw that ignores the arm, and an AR(1) lag counted in file
/// rows rather than within the arm.
#[test]
fn simulate_draws_the_serial_residual_correlation() {
    for kind in ["cs", "ar1"] {
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
