//! #2798 PR B, row B6: the fit's Durbin–Watson statistic is taken within each
//! innermost **cell** of a nested-level model, read off the population the fit
//! prepared — not the caller's, which carries no cells.
//!
//! ## The fixture
//!
//! `nonmem_anchor/nested_levels_3.*` (the #1082 anchor kit): 12 sponsors (`ID`) × 2
//! studies × 2 arms × 5 visits, `level STUDY by STUDY in ID`, `level ARM by ARM in
//! STUDY`, evaluated at NONMEM's optimum with `outer_maxiter = 0`. `ARM` is numbered
//! 1, 2 inside each study, so the cells are nested by key: the raw `ARM` label within
//! `ID` would merge arm 1 of study 1 with arm 1 of study 2. The caller's population
//! has no occasions at all (measured: `Subject::occasions` is empty after the read,
//! `iov_column` is `None`); `assign_level_cells` writes the 48 cells into the
//! prepared one.
//!
//! ## The oracle
//!
//! A Python computation on the 240 IWRES this fit exports, outside ferx. It first
//! reproduces ferx's **classic** statistic (whole sponsor, `÷ Σ e²`, the kernel before
//! this PR) to 10 decimals — 2.1218650325 both — and then gives, pair-normalised:
//!
//! | series | DW |
//! |---|---|
//! | per cell `(ID, STUDY, ARM)` — what this PR computes | 2.2683323360085725 |
//! | whole sponsor (the public, ungrouped `iwres_autocorrelation`) | 2.2008166400092444 |
//! | raw `ARM` within `ID` (studies merged) | 2.2208639616 |
//! | outer level only `(ID, STUDY)` | 2.1888425710 |
//!
//! The IWRES were identical, bit for bit, under the old and the new kernel (the
//! statistic is post-fit). Realised `|ferx − oracle|` is printed below; see
//! [`ORACLE_TOL`] for the measured values and the bound.

use ferx_core::api::read_population_for;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::stats::residual_error::iwres_autocorrelation;
use ferx_core::{fit, EstimationMethod, FitOptions};

/// Pair-normalised DW over the 48 innermost cells, from the Python oracle.
const ORACLE_CELLS: f64 = 2.268_332_336_008_572_5;
/// The same IWRES, whole sponsor (no split), pair-normalised.
const ORACLE_WHOLE: f64 = 2.200_816_640_009_244_4;
/// Realised `|ferx − oracle|`: 0 for the cells on native macOS arm64 and on Linux
/// aarch64 (`tools/linux-test.sh`); for the whole series 0 on macOS and 8.9e-16 on
/// Linux (the oracle was computed from the macOS IWRES, printed to 17 significant
/// figures). The inner optimizer's η̂ is the one platform-dependent input, and a
/// bound of `1e-8` leaves it room while sitting 4.7e6× below the nearest wrong answer
/// (raw `ARM` within `ID`, 0.047 away).
const ORACLE_TOL: f64 = 1e-8;

/// B6. Mutation that must redden this: the post-fit statistic reading the caller's
/// (unprepared) population, or no population at all — `iwres_autocorrelation(&subjects)`
/// at the call site in `fit_inner`, which on this fixture is the whole-sponsor value.
#[test]
fn dw_of_a_nested_level_fit_is_taken_within_the_innermost_cells() {
    let src = include_str!("../nonmem_anchor/nested_levels_3.ferx");
    let parsed = parse_full_model(src).expect("anchor model parses");
    assert!(
        parsed.fit_options.iov_column.is_none(),
        "a nested model sets no iov_column"
    );
    let (pop, _) = read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        "nonmem_anchor/nested_levels_3.csv",
        None,
        parsed.fit_options.iov_column.as_deref(),
        None,
        &parsed.column_map,
    )
    .expect("anchor data loads");
    assert!(
        pop.subjects.iter().all(|s| s.occasions.is_empty()),
        "the caller's population carries no cells; only the prepared one does"
    );
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        outer_maxiter: 0,
        run_covariance_step: false,
        ..parsed.fit_options.clone()
    };
    let r = fit(&parsed.model, &pop, &parsed.model.default_params, &opts).expect("evaluation");
    let (_, whole) = iwres_autocorrelation(&r.subjects);
    assert!(
        r.dw_statistic.is_finite() && whole.is_finite(),
        "({}, {whole})",
        r.dw_statistic
    );
    let d_cells = (r.dw_statistic - ORACLE_CELLS).abs();
    let d_whole = (whole - ORACLE_WHOLE).abs();
    eprintln!(
        "ferx DW {:.16} oracle {ORACLE_CELLS:.16} |Δ| {d_cells:.3e}; \
         whole {whole:.16} oracle {ORACLE_WHOLE:.16} |Δ| {d_whole:.3e}",
        r.dw_statistic
    );
    assert!(
        d_cells < ORACLE_TOL,
        "cells: ferx {} vs oracle",
        r.dw_statistic
    );
    assert!(d_whole < ORACLE_TOL, "whole: ferx {whole} vs oracle");
    // The straddle, asserted so the pair cannot quietly agree: the fit's value is the
    // split one, the public ungrouped entry the whole one, and they differ.
    assert!(
        (r.dw_statistic - whole).abs() > 0.05,
        "split {} vs whole {whole}",
        r.dw_statistic
    );
}

/// Pair-normalised DW of `series`, written out here rather than called, so the test
/// does not grade the kernel with itself.
fn pair_normalised_dw(series: &[Vec<f64>]) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for s in series {
        for w in s.windows(2) {
            num += (w[1] - w[0]).powi(2);
            den += (w[0] * w[0] + w[1] * w[1]) / 2.0;
        }
    }
    num / den
}

/// The `iov_column` cell with **no kappa** (#2798 carry-over from #2815). There
/// `run::occasions_are_derived` is false, so `dw_advice` names the column (`by ARM`)
/// — and the statistic must split there too, or the advice and the number would
/// describe different groupings. Measured: the reader fills `Subject::occasions`
/// from `iov_column` whether or not the model has a kappa, so the split happens.
///
/// Mutation that must redden this: the occasions part of the split key dropped (or
/// gated on `n_kappa > 0`).
#[test]
fn iov_column_without_kappa_splits_at_the_column() {
    let src = include_str!("../nonmem_anchor/nested_levels_3.ferx")
        .replace("  level STUDY by STUDY in ID\n", "")
        .replace("  level ARM by ARM in STUDY\n", "")
        .replace("  kappa K_ST ~ 0.0904362 at STUDY\n", "")
        .replace("  kappa K_ARM ~ 3.75478 at ARM weight = NARM\n", "")
        .replace("ETA_SP + K_ST + K_ARM", "ETA_SP")
        .replace("[fit_options]\n", "[fit_options]\n  iov_column = ARM\n");
    let parsed = parse_full_model(&src).expect("twin parses");
    assert_eq!(parsed.model.n_kappa, 0, "the cell under test has no kappa");
    assert_eq!(parsed.fit_options.iov_column.as_deref(), Some("ARM"));
    let (pop, _) = read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        "nonmem_anchor/nested_levels_3.csv",
        None,
        parsed.fit_options.iov_column.as_deref(),
        None,
        &parsed.column_map,
    )
    .expect("data loads");
    assert!(
        pop.subjects
            .iter()
            .all(|s| s.occasions.len() == s.observations.len()),
        "the reader fills the occasions from iov_column without a kappa"
    );
    let opts = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        ..parsed.fit_options.clone()
    };
    let r = fit(&parsed.model, &pop, &parsed.model.default_params, &opts).expect("evaluation");
    let mut by_arm: Vec<Vec<f64>> = Vec::new();
    for (s, res) in pop.subjects.iter().zip(&r.subjects) {
        for arm in [1, 2] {
            by_arm.push(
                s.occasions
                    .iter()
                    .zip(&res.iwres)
                    .filter(|(o, _)| **o == arm)
                    .map(|(_, e)| *e)
                    .collect(),
            );
        }
    }
    assert!(by_arm.iter().flatten().all(|e| e.is_finite()));
    let want = pair_normalised_dw(&by_arm);
    let (_, whole) = iwres_autocorrelation(&r.subjects);
    eprintln!(
        "iov_column = ARM, no kappa: fit DW {:.10}, split by ARM {want:.10}, whole {whole:.10}",
        r.dw_statistic
    );
    assert!(r.dw_statistic.is_finite() && whole.is_finite());
    assert!(
        (r.dw_statistic - want).abs() < 1e-12,
        "fit DW {} vs split {want}",
        r.dw_statistic
    );
    assert!(
        (want - whole).abs() > 0.01,
        "the fixture must tell the split from the whole: {want} vs {whole}"
    );
}
