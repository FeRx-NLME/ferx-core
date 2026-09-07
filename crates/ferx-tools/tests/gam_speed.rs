//! Speed benchmark for GAM covariate screening (#1114).
//!
//! Ignored by default; run with:
//!   cargo test -p ferx-tools --release --test gam_speed -- --ignored --nocapture
//!
//! It lives here rather than in a `#[cfg(test)] mod tests` inside
//! `crates/ferx-tools/src/gam.rs` for a coverage reason: `codecov.yml` excludes
//! `crates/*/tests/**` from measurement but not `src/` test modules, so an
//! `#[ignore]`d benchmark in `src/` is compiled, measured, never run, and drags
//! the patch gate down by its own line count.
//!
//! Exercises a large synthetic dataset: 2 000 subjects, 8 ETAs, 15 covariates
//! (120 ETA×covariate pairs × 3 forms = 360 OLS fits per run). Runs sequential
//! and parallel (rayon over ETAs) so the parallelism gain is visible.

use ferx_core::CovariateKind;
use ferx_tools::gam::{gam_screen_raw, GamOptions};
use rayon::prelude::*;

const N: usize = 2_000;
const N_ETAS: usize = 8;
const N_COVS: usize = 15;

/// One synthetic covariate: name, per-subject values, declared kind.
type Covariate = (String, Vec<f64>, CovariateKind);

/// Deterministic synthetic ETAs and covariates, shaped like a real screen.
fn synthetic_data() -> (Vec<Vec<f64>>, Vec<Covariate>) {
    let eta_data: Vec<Vec<f64>> = (0..N_ETAS)
        .map(|e| {
            (0..N)
                .map(|i| ((i as f64 * (e as f64 * 0.7 + 1.1)) * 0.003_141).sin() * 0.35)
                .collect()
        })
        .collect();

    let cov_data: Vec<Covariate> = (0..N_COVS)
        .map(|c| {
            if c % 5 == 0 {
                let vals: Vec<f64> = (0..N)
                    .map(|i| if (i + c * 3) % 2 == 0 { 0.0 } else { 1.0 })
                    .collect();
                (format!("COV_{c:02}"), vals, CovariateKind::Categorical)
            } else {
                let centre = 30.0 + c as f64 * 5.0;
                let vals: Vec<f64> = (0..N)
                    .map(|i| {
                        centre + ((i as f64 * (c as f64 * 0.4 + 0.9)) * 0.002_718).cos() * 20.0
                    })
                    .collect();
                (format!("COV_{c:02}"), vals, CovariateKind::Continuous)
            }
        })
        .collect();

    (eta_data, cov_data)
}

#[test]
#[ignore = "speed benchmark — run with: cargo test -p ferx-tools --release --test gam_speed -- --ignored --nocapture"]
fn gam_speed_large_dataset() {
    use std::time::Instant;

    let (eta_data, cov_data) = synthetic_data();

    let cov_names: Vec<&str> = cov_data.iter().map(|(n, _, _)| n.as_str()).collect();
    let cov_cols: Vec<&[f64]> = cov_data.iter().map(|(_, v, _)| v.as_slice()).collect();
    let cov_kinds: Vec<CovariateKind> = cov_data.iter().map(|(_, _, k)| *k).collect();
    let opts = GamOptions::default();
    let total_pairs = N_ETAS * N_COVS;

    // One ETA at a time: `gam_screen_raw` parallelises over ETAs, so a
    // single-ETA call is the sequential unit.
    let run_one = |e: usize| {
        let _ = gam_screen_raw(
            &["ETA"],
            &[eta_data[e].as_slice()],
            &[0.05],
            &cov_names,
            &cov_cols,
            &cov_kinds,
            &opts,
        );
    };

    // Warm up (first-touch allocation, rayon pool spin-up).
    for e in 0..N_ETAS {
        run_one(e);
    }

    let t0 = Instant::now();
    let n_seq_runs = 5;
    for _ in 0..n_seq_runs {
        for e in 0..N_ETAS {
            run_one(e);
        }
    }
    let seq_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_seq_runs as f64;

    let eta_names: Vec<&str> = (0..N_ETAS).map(|_| "ETA").collect();
    let eta_cols: Vec<&[f64]> = eta_data.iter().map(|v| v.as_slice()).collect();
    let shrinkage = vec![0.05_f64; N_ETAS];
    let run_all = || {
        let _ = gam_screen_raw(
            &eta_names, &eta_cols, &shrinkage, &cov_names, &cov_cols, &cov_kinds, &opts,
        );
    };

    run_all();
    let t1 = Instant::now();
    let n_par_runs = 20;
    for _ in 0..n_par_runs {
        run_all();
    }
    let par_ms = t1.elapsed().as_secs_f64() * 1000.0 / n_par_runs as f64;

    println!(
        "gam_screen: {N} subjects × {N_ETAS} ETAs × {N_COVS} covariates \
         = {total_pairs} pairs ({} OLS fits)\n  \
         sequential: {seq_ms:.1} ms ({:.3} ms/pair)\n  \
         parallel:   {par_ms:.1} ms ({:.3} ms/pair)\n  \
         speed-up:   {:.2}×",
        total_pairs * 3,
        seq_ms / total_pairs as f64,
        par_ms / total_pairs as f64,
        seq_ms / par_ms,
    );
}

/// Guard the benchmark itself: it is `#[ignore]`d, so without this nothing
/// would catch it rotting against a `gam_screen_raw` signature change.
#[test]
fn benchmark_inputs_screen_without_warnings() {
    let (eta_data, cov_data) = synthetic_data();
    let cov_names: Vec<&str> = cov_data.iter().map(|(n, _, _)| n.as_str()).collect();
    let cov_cols: Vec<&[f64]> = cov_data.iter().map(|(_, v, _)| v.as_slice()).collect();
    let cov_kinds: Vec<CovariateKind> = cov_data.iter().map(|(_, _, k)| *k).collect();

    let result = gam_screen_raw(
        &["ETA_0"],
        &[eta_data[0].as_slice()],
        &[0.05],
        &cov_names,
        &cov_cols,
        &cov_kinds,
        &GamOptions::default(),
    );
    assert_eq!(result.eta_results.len(), 1);
    assert_eq!(result.eta_results[0].covariate_scores.len(), N_COVS);
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);

    // The rayon import is what the benchmark's parallel arm needs from the
    // dependency list; touch it so the import cannot silently rot.
    let checked: usize = (0..N_ETAS).into_par_iter().map(|e| eta_data[e].len()).sum();
    assert_eq!(checked, N * N_ETAS);
}
