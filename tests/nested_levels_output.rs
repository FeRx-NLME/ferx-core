//! The per-level output of a nested fit (#1082 PR4), on the exact3 anchor
//! (`nonmem_anchor/nested_levels_3.*`) evaluated once at NONMEM's optimum: each level's
//! κ̂ once per group (`ebe_kappas_by_level`), the sdtab column per kappa, and the
//! `.fitrx` round trip — an older bundle without the per-level entries still loads.

use ferx_core::api::read_population_for;
use ferx_core::io::fitrx::{load_fit, save_fit, SaveFitOptions};
use ferx_core::io::output::sdtab;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, EstimationMethod, FitOptions, FitResult, Population};

const MODEL: &str = include_str!("../nonmem_anchor/nested_levels_3.ferx");

fn eval() -> (FitResult, Population) {
    let parsed = parse_full_model(MODEL).expect("anchor model must parse");
    let (pop, _) = read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        "nonmem_anchor/nested_levels_3.csv",
        None,
        parsed.fit_options.iov_column.as_deref(),
        None,
        &parsed.column_map,
    )
    .expect("anchor data must load");
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        outer_maxiter: 0,
        run_covariance_step: false,
        ..parsed.fit_options.clone()
    };
    let r = fit(&parsed.model, &pop, &parsed.model.default_params, &opts).expect("evaluates");
    (r, pop)
}

/// 12 sponsors, each with two studies of two arms. The study κ̂ appears once per study
/// (not once per arm, as in `ebe_kappas`), the arm κ̂ once per arm, and each is the value
/// the per-cell `ebe_kappas` repeats: `[κ_study, κ_arm]` of the cell (study s, arm a).
#[test]
fn a_nested_fit_reports_each_level_s_kappa_once_per_group() {
    let (r, _) = eval();
    assert_eq!(r.kappa_levels, ["STUDY", "ARM"]);
    assert!(
        r.shrinkage_kappa_by_occ.is_empty(),
        "no per-slot shrinkage under nested levels"
    );
    assert!(r.shrinkage_kappa.iter().all(|s| s.is_finite()));
    let [study, arm] = r.ebe_kappas_by_level.as_slice() else {
        panic!("two levels: {:?}", r.ebe_kappas_by_level.len())
    };
    // Every arm has observations, so every group is informed: each κ's shrinkage is
    // the pool over its own level's groups — 24 studies, 48 arms — not over 48 cells,
    // which would count each study twice.
    let iov = r.omega_iov.as_ref().unwrap();
    for (k, level) in [study, arm].into_iter().enumerate() {
        let vals: Vec<f64> = level.groups.iter().flatten().map(|g| g.kappa[0]).collect();
        assert_eq!(vals.len(), [24, 48][k]);
        let ms = vals.iter().map(|v| v.powi(2)).sum::<f64>() / vals.len() as f64;
        let want = 1.0 - ms.sqrt() / iov[(k, k)].sqrt();
        assert!(
            (r.shrinkage_kappa[k] - want).abs() < 1e-14,
            "level {}: shrinkage {} vs {want} over its groups",
            level.level,
            r.shrinkage_kappa[k]
        );
    }
    assert_eq!((study.level.as_str(), arm.level.as_str()), ("STUDY", "ARM"));
    assert_eq!(study.columns, ["STUDY"]);
    assert_eq!(arm.columns, ["STUDY", "ARM"]);
    assert_eq!(study.kappa_names, ["K_ST"]);
    assert_eq!(arm.kappa_names, ["K_ARM"]);
    assert_eq!(study.groups.len(), 12);
    for i in 0..12 {
        assert_eq!(study.groups[i].len(), 2, "sponsor {i}: two studies");
        assert_eq!(arm.groups[i].len(), 4, "sponsor {i}: four arms");
        // Cells come in first-record order (1,1) (1,2) (2,1) (2,2), as do the arms.
        for (g, a) in arm.groups[i].iter().enumerate() {
            let cell = &r.ebe_kappas[i][g];
            assert_eq!(a.kappa, [cell[1]], "sponsor {i} arm {:?}", a.labels);
            let s = study.groups[i]
                .iter()
                .find(|s| s.labels[0] == a.labels[0])
                .expect("the arm's study");
            assert_eq!(s.kappa, [cell[0]], "sponsor {i} study {}", s.labels[0]);
        }
    }
}

/// The sdtab carries `K_ST` and `K_ARM`: each row's own study and arm κ̂, found by the
/// row's labels, so the read population (cells never assigned) gives the same columns.
#[test]
fn the_sdtab_carries_each_row_s_level_kappas() {
    let (r, pop) = eval();
    let table = sdtab(&r, &pop);
    let col = |name: &str| {
        table
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("sdtab has no {name} column"))
            .1
            .clone()
    };
    let (k_st, k_arm) = (col("K_ST"), col("K_ARM"));
    assert_eq!(k_st.len(), 240);
    let (study, arm) = (&r.ebe_kappas_by_level[0], &r.ebe_kappas_by_level[1]);
    let mut row = 0;
    for (i, s) in pop.subjects.iter().enumerate() {
        for j in 0..s.obs_times.len() {
            let cov = &s.obs_covariates[j];
            let (st, a) = (cov["STUDY"], cov["ARM"]);
            let want_st = study.groups[i]
                .iter()
                .find(|g| g.labels == [st])
                .unwrap()
                .kappa[0];
            let want_arm = arm.groups[i]
                .iter()
                .find(|g| g.labels == [st, a])
                .unwrap()
                .kappa[0];
            assert!(k_st[row].is_finite() && k_arm[row].is_finite(), "row {row}");
            assert_eq!(k_st[row], want_st, "row {row}: K_ST of study {st}");
            assert_eq!(
                k_arm[row], want_arm,
                "row {row}: K_ARM of study {st} arm {a}"
            );
            row += 1;
        }
    }
    assert_eq!(row, 240);
}

/// `simulate_with_uncertainty` runs a nested fit (#1082 PR4 lifts PR2's refusal): each
/// parameter draw simulates through the same per-level κ draw as `simulate()`, whose
/// oracles are in `nested_levels_simulate.rs`.
#[test]
fn a_nested_fit_simulates_with_uncertainty() {
    let parsed = parse_full_model(MODEL).unwrap();
    let (pop, _) = read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        "nonmem_anchor/nested_levels_3.csv",
        None,
        None,
        None,
        &parsed.column_map,
    )
    .unwrap();
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        outer_maxiter: 0,
        run_covariance_step: true,
        ..parsed.fit_options.clone()
    };
    let r = fit(&parsed.model, &pop, &parsed.model.default_params, &opts).unwrap();
    assert!(r.covariance_matrix.is_some(), "{:?}", r.covariance_status);
    let rows = ferx_core::api::simulate_with_uncertainty(
        &parsed.model,
        &pop,
        &r,
        &ferx_core::api::SimulateUncertaintyOptions {
            n_uncertainty_draws: 3,
            n_sim_per_draw: 1,
            seed: Some(1082),
            ..Default::default()
        },
    )
    .expect("a nested fit simulates with uncertainty");
    assert_eq!(rows.len(), 3 * 240);
    assert!(rows.iter().all(|r| r.ipred.is_finite()));
}

fn bundle_path(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("ferx-1082-pr4-{tag}-{}.fitrx", std::process::id()))
}

/// `.fitrx` writes one `ebes_kappa_<LEVEL>.csv` per level and reads it back (to the six
/// decimals the bundle's CSVs carry); `kappa_levels` travels in `fit.json`.
#[test]
fn a_nested_fit_round_trips_through_fitrx_with_its_levels() {
    let (r, pop) = eval();
    let path = bundle_path("roundtrip");
    save_fit(&r, &pop, MODEL, &path, SaveFitOptions::default()).expect("saves");
    let loaded = load_fit(&path).expect("loads");
    let _ = std::fs::remove_file(&path);
    let names = loaded.manifest.entries.clone();
    assert!(
        names.iter().any(|e| e == "ebes_kappa_STUDY.csv"),
        "{names:?}"
    );
    assert!(names.iter().any(|e| e == "ebes_kappa_ARM.csv"), "{names:?}");
    assert_eq!(loaded.fit.kappa_levels, r.kappa_levels);
    let got = &loaded.fit.ebe_kappas_by_level;
    assert_eq!(got.len(), 2);
    for (g, w) in got.iter().zip(&r.ebe_kappas_by_level) {
        assert_eq!(
            (&g.level, &g.columns, &g.kappa_names),
            (&w.level, &w.columns, &w.kappa_names)
        );
        assert_eq!(g.groups.len(), w.groups.len());
        for (gs, ws) in g.groups.iter().zip(&w.groups) {
            assert_eq!(gs.len(), ws.len());
            for (a, b) in gs.iter().zip(ws) {
                assert_eq!(a.labels, b.labels);
                // The bundle's CSVs print 6 decimals, as `ebes_kappa.csv` does: the
                // reload is the value rounded there, within half a unit of the 6th.
                for (x, y) in a.kappa.iter().zip(&b.kappa) {
                    assert!(x.is_finite() && (x - y).abs() <= 5e-7, "{x} vs {y}");
                }
            }
        }
    }
}

/// A bundle written before #1082 PR4 — a nested fit with no `kappa_levels`, no level
/// list and no per-level entries — still loads, with no per-level κ̂.
#[test]
fn an_older_nested_bundle_without_levels_still_loads() {
    let (mut r, pop) = eval();
    r.kappa_levels.clear();
    r.ebe_kappas_by_level.clear();
    let path = bundle_path("older");
    save_fit(&r, &pop, MODEL, &path, SaveFitOptions::default()).expect("saves");
    let loaded = load_fit(&path).expect("an older bundle loads");
    let _ = std::fs::remove_file(&path);
    assert!(!loaded
        .manifest
        .entries
        .iter()
        .any(|e| e.starts_with("ebes_kappa_")));
    assert!(loaded.fit.kappa_levels.is_empty());
    assert!(loaded.fit.ebe_kappas_by_level.is_empty());
    assert_eq!(loaded.fit.ebe_kappas.len(), 12, "the per-cell κ̂ still load");
}
