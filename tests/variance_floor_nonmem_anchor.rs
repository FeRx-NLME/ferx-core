//! NONMEM anchor for #2441: a residual variance below `1e-12` is scored as computed.
//!
//! Before #2441 every residual variance was raised to an absolute floor, `MIN_VARIANCE =
//! 1e-12`. Under proportional error `V = (f·σ)²`, so a prediction below `1e-5` at `σ = 0.1`
//! was scored with the wrong variance, and whether that happened depended on the units of DV.
//! NONMEM has no such floor.
//!
//! # The anchor
//!
//! `nonmem_anchor/variance_floor_prop_k1.ctl` is a one-compartment IV bolus with proportional
//! error, `σ = 0.1`, a dose of 0.001 and 12 subjects × 9 samples out to 48 h. The concentration
//! falls from about 1e-4 to about 1e-8, so 30 of the 108 samples sit below the old floor, with
//! variances down to about 1e-18.
//! `variance_floor_prop_k1e6.ctl` is the same stream on the same data in units a million times
//! larger (AMT and DV × 1e6), where no variance comes near it. Both are `MAXEVAL=0 POSTHOC` with
//! every parameter `FIX`: one evaluation at the declared values, not a fit.
//! `nonmem_anchor/variance_floor_prop_fit.ferx` is the ferx twin, run on either dataset;
//! `nonmem_anchor/make_variance_floor_prop.py` writes the data.
//!
//! The two arms straddle the gate under the old behaviour: at k = 1e6 the floor never applied,
//! so the old and new code agree there and with NONMEM; at k = 1 only the new code does.
//!
//! # Tier and gate
//!
//! Tier 2: `outer_maxiter = 0` is one evaluation and no convergence loop.

use ferx_core::types::FitOptions;
use ferx_core::{fit, read_nonmem_csv};
use std::path::Path;

const MODEL: &str = "nonmem_anchor/variance_floor_prop_fit.ferx";
const DATA_K1: &str = "nonmem_anchor/variance_floor_prop_k1.csv";
const DATA_K1E6: &str = "nonmem_anchor/variance_floor_prop_k1e6.csv";

/// Observation records in each dataset.
const N_OBS: usize = 108;

/// NONMEM 7.5.1, `$EST METHOD=1 INTERACTION MAXEVAL=0 POSTHOC`, from each stream's `.ext`
/// `-1000000000` row (`nonmem_anchor/results/variance_floor_prop_k1{,e6}.ext`).
const NM_OFV_K1: f64 = -2_644.731_273_940_770_7;
const NM_OFV_K1E6: f64 = 339.419_006_602_090_46;

/// NONMEM's `$TABLE` at the same evaluation: `IPRED` per record, read to check that the k = 1
/// arm really puts variances below the old floor and the k = 1e6 arm does not.
const NM_TAB_K1: &str = "nonmem_anchor/results/variance_floor_prop_k1.tab";
const NM_TAB_K1E6: &str = "nonmem_anchor/results/variance_floor_prop_k1e6.tab";

/// The declared proportional SD, `PROP_ERR` in the twin and `√$SIGMA` in the streams.
const SIGMA_SD: f64 = 0.1;

/// ferx before #2441 on the k = 1 data (`main` at `72e1c6ef`, macOS arm64), with every variance
/// raised to `1e-12`: 93.86 above the objective after the fix. Not a target: it is here so the
/// tolerance can be chosen against the failure it has to catch. At k = 1e6 the same commit gave
/// 339.419006597793, the value after the fix to every printed digit, since no variance there
/// was ever below the floor.
const PRE_FIX_OFV_K1: f64 = -2_550.873_426_278_432;

/// Realised |ferx − NONMEM|: 1.8e-8 at k = 1 and 4.3e-9 at k = 1e6, on Linux x86_64 (glibc
/// 2.28) and macOS arm64 alike — the two print the same ferx objective to 12 decimals. The bound
/// gives about 50× headroom over the larger, and sits 1e8 below the 93.86 the floor moved the
/// k = 1 objective.
const TOL: f64 = 1e-6;

/// The ferx FOCEI objective at the declared parameters: `outer_maxiter = 0` is ferx's
/// `MAXEVAL=0`. The model file's `[fit_options]` are not carried by the parse, so the method
/// and the budget are restated on the options passed to `fit()`.
fn ferx_ofv(data: &str) -> f64 {
    let model = ferx_core::parse_model_file(Path::new(MODEL)).expect("the anchor twin must parse");
    let population =
        read_nonmem_csv(Path::new(data), None, None).expect("the anchor dataset must load");
    let n_obs: usize = population
        .subjects
        .iter()
        .map(|s| s.observations.len())
        .sum();
    assert_eq!(
        n_obs, N_OBS,
        "{data}: the anchor scores all {N_OBS} observations"
    );
    let options = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        verbose: false,
        method: ferx_core::EstimationMethod::FoceI,
        ..FitOptions::default()
    };
    let result = fit(&model, &population, &model.default_params, &options)
        .expect("the anchor evaluation must succeed");
    assert!(result.ofv.is_finite(), "{data}: ferx OFV is not finite");
    eprintln!("[#2441] {data}: ferx OFV {:.12}", result.ofv);
    result.ofv
}

/// #2441: with residual variances down to about 1e-18, ferx reproduces NONMEM's objective.
#[test]
fn variances_below_the_old_floor_match_nonmem() {
    let ofv = ferx_ofv(DATA_K1);
    let err = (ofv - NM_OFV_K1).abs();
    assert!(
        err < TOL,
        "k = 1: ferx {ofv} vs NONMEM {NM_OFV_K1}, |Δ| = {err}"
    );
}

/// The same data in units a million times larger, where no variance comes near `1e-12`.
#[test]
fn the_same_data_in_larger_units_match_nonmem() {
    let ofv = ferx_ofv(DATA_K1E6);
    let err = (ofv - NM_OFV_K1E6).abs();
    assert!(
        err < TOL,
        "k = 1e6: ferx {ofv} vs NONMEM {NM_OFV_K1E6}, |Δ| = {err}"
    );
}

/// A change of units shifts every `−2 ln L` by `2·n·ln k` and nothing else, in both engines.
#[test]
fn a_change_of_units_shifts_the_objective_by_2_n_ln_k() {
    let want = 2.0 * N_OBS as f64 * 1e6_f64.ln();
    let ferx = ferx_ofv(DATA_K1E6) - ferx_ofv(DATA_K1);
    assert!(
        (ferx - want).abs() < 1e-6,
        "ferx: shift {ferx} vs 2·n·ln k = {want}"
    );
    // Realised: ferx 4.5e-13 (Linux x86_64 and macOS arm64), NONMEM 2.3e-8.
    let nonmem = NM_OFV_K1E6 - NM_OFV_K1;
    assert!(
        (nonmem - want).abs() < 1e-6,
        "NONMEM: shift {nonmem} vs 2·n·ln k = {want}"
    );
}

/// NONMEM's `IPRED` column for one table's observation records (`TIME > 0`; the doses are at 0).
fn nonmem_ipred(tab: &str) -> Vec<f64> {
    let text = std::fs::read_to_string(tab).expect("the NONMEM table must be committed");
    let mut lines = text.lines().skip(1);
    let header: Vec<&str> = lines
        .next()
        .expect("table header")
        .split_whitespace()
        .collect();
    let col = |name: &str| header.iter().position(|h| *h == name).expect(name);
    let (time, ipred) = (col("TIME"), col("IPRED"));
    lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: Vec<f64> = l
                .split_whitespace()
                .map(|x| x.parse().expect("numeric table field"))
                .collect();
            (v[time], v[ipred])
        })
        .filter(|(t, _)| *t > 0.0)
        .map(|(_, f)| f)
        .collect()
}

/// The two arms straddle the old floor in NONMEM's own predictions: 30 of the 108 variances
/// `(IPRED·σ)²` sit below `1e-12` at k = 1, down to 1.5e-18, and none at k = 1e6, whose smallest
/// is 1.5e-6. Without this, a regenerated dataset whose concentrations stay above `1e-5` would
/// keep both OFV tests green while no longer testing the floor at all.
#[test]
fn nonmem_puts_the_k1_arm_below_the_old_floor_and_the_k1e6_arm_above_it() {
    let below = |tab: &str| {
        let ipred = nonmem_ipred(tab);
        assert_eq!(ipred.len(), N_OBS, "{tab}: one IPRED per observation");
        assert!(
            ipred.iter().all(|f| f.is_finite() && *f > 0.0),
            "{tab}: every IPRED must be positive and finite"
        );
        ipred
            .iter()
            .filter(|f| (**f * SIGMA_SD).powi(2) < 1e-12)
            .count()
    };
    assert_eq!(below(NM_TAB_K1), 30, "k = 1: variances below the old floor");
    assert_eq!(
        below(NM_TAB_K1E6),
        0,
        "k = 1e6: variances below the old floor"
    );
}

/// The anchor is only evidence if the old floor would have failed it.
#[test]
fn the_pre_fix_objective_is_far_outside_the_anchor_tolerance() {
    assert!(
        (PRE_FIX_OFV_K1 - NM_OFV_K1).abs() > 100.0 * TOL,
        "the fixture no longer separates the floor from NONMEM — a green anchor above \
         would say nothing"
    );
    let ofv = ferx_ofv(DATA_K1);
    assert!(
        (ofv - PRE_FIX_OFV_K1).abs() > 100.0 * TOL,
        "ferx OFV {ofv} is within {} of the pre-#2441 value {PRE_FIX_OFV_K1}: the floor is \
         being applied again",
        100.0 * TOL
    );
}
