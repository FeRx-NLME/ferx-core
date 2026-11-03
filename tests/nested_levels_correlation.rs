//! Nested levels with a residual `correlation … by` line (#1082 PR5) on the entry points
//! that are not the fit: `simulate()` (V8), NPDE (V9) and the fit's outputs (V10).
//!
//! The simulation fixture: one subject, two studies × two arms × three rows, `ARM`
//! numbered 1, 2 inside each study (nested by key), the rows of the four arms interleaved
//! in TIME so a row's AR(1) lag is its distance inside its arm, not in the file. The κ
//! enter the prediction, so `DV − IPRED` (IPRED at the drawn η and κ) is the residual
//! draw. The population is handed to `simulate()` as built — no cells, no blocks — so
//! the draw has to assign both.

use ferx_core::api::read_population_for;
use ferx_core::io::output::{sdtab, write_estimates_yaml};
use ferx_core::parser::model_parser::{parse_full_model, parse_model_string};
use ferx_core::stats::npde::compute_npde_npd;
use ferx_core::{
    fit, simulate_with_seed, CompiledModel, EstimationMethod, FitOptions, FitResult, Population,
    SimOutcome, Subject,
};
use std::collections::HashMap;

/// `(STUDY, ARM, TIME)` per row, in file (TIME) order.
const ROWS: [(f64, f64, f64); 12] = [
    (1.0, 1.0, 0.0),
    (1.0, 2.0, 0.5),
    (2.0, 1.0, 1.0),
    (2.0, 2.0, 1.5),
    (1.0, 1.0, 2.0),
    (1.0, 2.0, 3.0),
    (2.0, 1.0, 4.0),
    (2.0, 2.0, 6.0),
    (1.0, 1.0, 5.0),
    (1.0, 2.0, 8.0),
    (2.0, 1.0, 9.0),
    (2.0, 2.0, 10.0),
];
const N_SIM: usize = 20_000;
const RHO: f64 = 0.6;

/// The fixture model; `line` is the correlation line, and the `RHO` θ is declared only
/// with one (an unread θ is a model error).
fn src(line: &str, rho: f64) -> String {
    let rho = if line.is_empty() {
        String::new()
    } else {
        format!("  theta RHO({rho}, -0.4, 0.95)\n")
    };
    format!(
        "[parameters]\n  theta TVY(10.0, 1.0, 100.0)\n{rho}  \
         omega ETA ~ 0.04\n  level STUDY by STUDY in ID\n  level ARM by ARM in STUDY\n  \
         kappa K_ST ~ 0.04 at STUDY\n  kappa K_ARM ~ 0.09 at ARM\n  sigma ADD_ERR ~ 1.0 (sd)\n\
         [individual_parameters]\n  Y0 = TVY * exp(ETA + K_ST + K_ARM)\n\
         [structural_model]\n  y = Y0\n[error_model]\n  DV ~ additive(ADD_ERR)\n{line}"
    )
}

fn model(line: &str, rho: f64) -> CompiledModel {
    parse_model_string(&src(line, rho)).expect("model parses")
}

fn population() -> Population {
    let snap = |s: f64, a: f64| HashMap::from([("STUDY".to_string(), s), ("ARM".to_string(), a)]);
    let mut rows = ROWS.to_vec();
    rows.sort_by(|a, b| a.2.total_cmp(&b.2));
    let subject = Subject {
        id: "1".into(),
        obs_times: rows.iter().map(|r| r.2).collect(),
        observations: rows.iter().map(|r| 10.0 + 0.1 * r.2).collect(),
        obs_cmts: vec![1; 12],
        cens: vec![0; 12],
        covariates: snap(1.0, 1.0),
        obs_covariates: rows.iter().map(|r| snap(r.0, r.1)).collect(),
        ..Default::default()
    };
    Population {
        subjects: vec![subject],
        covariate_names: vec!["STUDY".into(), "ARM".into()],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

/// Position of each row inside its arm, in TIME order, by hand.
fn positions(rows: &[(f64, f64, f64)]) -> Vec<i32> {
    rows.iter()
        .map(|r| {
            rows.iter()
                .filter(|o| o.0 == r.0 && o.1 == r.1 && o.2 < r.2)
                .count() as i32
        })
        .collect()
}

fn corr(eps: &[Vec<f64>], j: usize, k: usize) -> f64 {
    let n = eps.len() as f64;
    let mean = |c: usize| eps.iter().map(|e| e[c]).sum::<f64>() / n;
    let (mj, mk) = (mean(j), mean(k));
    let cov = eps.iter().map(|e| (e[j] - mj) * (e[k] - mk)).sum::<f64>() / n;
    let var = |c: usize, m: f64| eps.iter().map(|e| (e[c] - m).powi(2)).sum::<f64>() / n;
    cov / (var(j, mj) * var(k, mk)).sqrt()
}

fn values(rows: &[ferx_core::SimulationResult]) -> Vec<(f64, f64)> {
    rows.iter()
        .map(|r| {
            let SimOutcome::Continuous { value } = r.outcome else {
                panic!("a Gaussian row");
            };
            (r.time, value)
        })
        .collect()
}

/// V8: the draw is the nested, within-arm `R`. Every pair's sample correlation of
/// `DV − IPRED` is within 4 SE of `ρ^lag` (lag inside the arm) within an arm and of 0
/// across arms — arm 1 of study 1 against arm 1 of study 2 included, so a grouping by the
/// label alone (one block of six rows) fails, as do an independent draw and an AR lag in
/// file rows.
#[test]
fn simulate_draws_the_within_arm_correlation_nested_by_key() {
    let m = model("  correlation ar1(RHO) by ARM\n", RHO);
    let mut rows = ROWS.to_vec();
    rows.sort_by(|a, b| a.2.total_cmp(&b.2));
    let pos = positions(&rows);
    let sims = simulate_with_seed(&m, &population(), &m.default_params, N_SIM, 1082)
        .expect("a nested model with a correlation simulates");
    let mut eps = vec![vec![0.0; 12]; N_SIM];
    for r in &sims {
        let SimOutcome::Continuous { value } = r.outcome else {
            panic!("a Gaussian row");
        };
        let j = rows.iter().position(|row| row.2 == r.time).unwrap();
        eps[r.sim - 1][j] = value - r.ipred;
    }
    assert!(eps.iter().flatten().all(|v| v.is_finite()));
    let (mut live, mut same_label) = (0, 0);
    for j in 0..12 {
        for k in (j + 1)..12 {
            let same_arm = rows[j].0 == rows[k].0 && rows[j].1 == rows[k].1;
            let want = if same_arm {
                RHO.powi((pos[j] - pos[k]).abs())
            } else {
                0.0
            };
            let got = corr(&eps, j, k);
            let se = (1.0 - want * want) / (N_SIM as f64).sqrt();
            assert!(
                (got - want).abs() < 4.0 * se,
                "rows {j},{k} {:?} {:?}: sample correlation {got:.4} vs {want:.4} (4·SE {:.4})",
                rows[j],
                rows[k],
                4.0 * se
            );
            live += usize::from(want != 0.0);
            same_label += usize::from(!same_arm && rows[j].1 == rows[k].1);
        }
    }
    assert_eq!(live, 12, "four arms of three rows: twelve within-arm pairs");
    assert_eq!(
        same_label, 18,
        "pairs sharing an ARM label across studies, at ρ = 0 each"
    );
}

/// V8, ρ = 0: the line with a free ρ at 0 draws what the model without it draws, with one
/// seed. Measured bit-identical: at ρ = 0 the block's factor is the identity, so the
/// dense draw consumes the same normals in the same order.
#[test]
fn simulate_at_rho_zero_is_the_draw_without_the_line() {
    let with = model("  correlation ar1(RHO) by ARM\n", 0.0);
    let without = model("", 0.0);
    let pop = population();
    let a = values(&simulate_with_seed(&with, &pop, &with.default_params, 50, 7).unwrap());
    let b = values(&simulate_with_seed(&without, &pop, &without.default_params, 50, 7).unwrap());
    assert_eq!(a.len(), b.len());
    for ((ta, va), (tb, vb)) in a.iter().zip(&b) {
        assert_eq!(ta, tb);
        assert_eq!(va.to_bits(), vb.to_bits(), "TIME {ta}: {va} vs {vb}");
    }
    // The straddle: at ρ = 0.6 the same seed draws other values.
    let c = model("  correlation ar1(RHO) by ARM\n", RHO);
    let c = values(&simulate_with_seed(&c, &pop, &c.default_params, 50, 7).unwrap());
    assert!(c.iter().zip(&b).any(|(x, y)| x.1 != y.1));
}

/// V9: NPDE on a nested model with a correlation is finite, and at ρ = 0 it is the NPDE of
/// the model without the line (the reference draw is the simulate draw above).
#[test]
fn npde_on_a_nested_correlated_model() {
    let pop = population();
    let at = |line: &str, rho: f64| {
        let m = model(line, rho);
        compute_npde_npd(&m, &pop, &m.default_params, 200, Some(11), None).expect("npde")
    };
    let line = "  correlation ar1(RHO) by ARM\n";
    let corr = at(line, RHO);
    assert!(corr[0]
        .npde
        .iter()
        .chain(&corr[0].npd)
        .all(|v| v.is_finite()));
    let zero = at(line, 0.0);
    let none = at("", 0.0);
    for (a, b) in zero[0].npde.iter().zip(&none[0].npde) {
        assert_eq!(a.to_bits(), b.to_bits(), "{a} vs {b}");
    }
    assert!(corr[0].npde.iter().zip(&none[0].npde).any(|(a, b)| a != b));
}

fn exact3(rho: Option<f64>) -> (FitResult, Population) {
    let base = include_str!("../nonmem_anchor/nested_levels_3.ferx");
    let src = match rho {
        None => base.to_string(),
        Some(r) => base
            .replace(
                "  sigma ADD ~ 0.255164\n",
                &format!("  sigma ADD ~ 0.255164\n  theta RHO({r}, -0.9, 0.9)\n"),
            )
            .replace(
                "  DV ~ additive(ADD)\n",
                "  DV ~ additive(ADD)\n  correlation ar1(RHO) by ARM\n",
            ),
    };
    let parsed = parse_full_model(&src).expect("parses");
    let (pop, _) = read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        "nonmem_anchor/nested_levels_3.csv",
        None,
        parsed.fit_options.iov_column.as_deref(),
        None,
        &parsed.column_map,
    )
    .expect("loads");
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        outer_maxiter: 0,
        run_covariance_step: false,
        ..parsed.fit_options.clone()
    };
    let r = fit(&parsed.model, &pop, &parsed.model.default_params, &opts).expect("evaluates");
    (r, pop)
}

/// V10: the fit's estimates YAML carries the ρ row next to each level's KAPPA row, and the
/// sdtab carries each κ's column and CWRES. That the CWRES are the dense-`R` ones is
/// `the_reported_cwres_whiten_by_the_dense_r_tilde` (a lib test, which can build `R̃`).
#[test]
fn the_fit_outputs_carry_rho_and_every_level_s_kappa() {
    let (r, pop) = exact3(Some(0.5));
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fit.yaml");
    write_estimates_yaml(&r, path.to_str().unwrap()).expect("writes");
    let yaml = std::fs::read_to_string(&path).unwrap();
    for name in ["RHO", "K_ST", "K_ARM"] {
        assert!(yaml.contains(name), "{name} missing from\n{yaml}");
    }
    let table = sdtab(&r, &pop);
    let cols: Vec<&str> = table.iter().map(|(c, _)| c.as_str()).collect();
    for c in ["K_ST", "K_ARM", "CWRES"] {
        assert!(cols.contains(&c), "{c} not in {cols:?}");
    }
    let cwres = &table.iter().find(|(c, _)| c == "CWRES").unwrap().1;
    assert_eq!(cwres.len(), 240);
    assert!(cwres.iter().all(|v| v.is_finite()));
}
