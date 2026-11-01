//! Shared test helpers for the integration tests.
//!
//! Integration tests live in separate crates and can only see the *public* API,
//! so this lives in `tests/common/` (a subdirectory, which Cargo does not treat
//! as its own test binary) and is pulled in with `mod common;`. Not every test
//! uses every helper, hence the crate-level `allow(dead_code)`.

#![allow(dead_code)]

use ferx_core::types::{DoseEvent, Subject};
use std::collections::HashMap;

/// Build a [`Subject`] from the fields tests usually vary, defaulting the
/// boilerplate vectors (covariates, occasions, …) to empty and `cens` to
/// all-uncensored (`vec![0; obs_times.len()]`). Set any remaining field on the
/// returned value, e.g.
///
/// ```ignore
/// let mut s = common::subject("1", doses, obs_times, observations, obs_cmts);
/// s.reset_times = vec![12.0];
/// ```
///
/// Centralizing the full-field literal means a new `Subject` field is added in
/// one place here rather than in every integration-test file.
pub fn subject(
    id: &str,
    doses: Vec<DoseEvent>,
    obs_times: Vec<f64>,
    observations: Vec<f64>,
    obs_cmts: Vec<usize>,
) -> Subject {
    let n = obs_times.len();
    Subject {
        obs_rgroup: Vec::new(),
        id: id.into(),
        doses,
        obs_times,
        obs_raw_times: vec![],
        observations,
        obs_cmts,
        covariates: HashMap::new(),
        dose_covariates: vec![],
        obs_covariates: vec![],
        pk_only_times: vec![],
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: vec![],
        reset_times: vec![],
        reset_covariates: vec![],
        cens: vec![0; n],
        occasions: vec![],
        obs_l2: Vec::new(),
        dose_occasions: vec![],
        reset_occasions: vec![],
        pk_only_occasions: Vec::new(),
        fremtype: vec![],
        obs_records: vec![],
        endpoint_covariates: Vec::new(),
        filled_cells: Vec::new(),
        simulated_mixnum: None,
    }
}

/// Build a TTE-only [`Population`](ferx_core::types::Population) from `(time, dv)`
/// pairs: `dv == 1` is an exact event, anything else is right-censored. Every row
/// is a single `Event` record on CMT 2 (entry time 0). Shared by the TTE smoke
/// (`tte_smoke.rs`) and convergence (`tte_convergence.rs`) integration tests so the
/// one-record-per-subject construction lives in a single place.
#[cfg(feature = "survival")]
pub fn tte_pop_from_pairs(data: &[(f64, u8)]) -> ferx_core::types::Population {
    use ferx_core::types::{EventType, ObsRecord, Population};
    let subjects = data
        .iter()
        .enumerate()
        .map(|(i, &(t, dv))| {
            let event_type = if dv == 1 {
                EventType::Exact
            } else {
                EventType::RightCensored
            };
            let mut s = subject(&format!("{}", i + 1), vec![], vec![], vec![], vec![]);
            s.obs_records = vec![ObsRecord::Event {
                time: t,
                event_type,
                entry_time: 0.0,
                cmt: 2,
            }];
            s
        })
        .collect();

    Population {
        covariate_names: vec![],
        dv_column: "DV".to_string(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects,
    }
}

/// Build a two-cause competing-risks TTE population on CMTs 2 and 3 from
/// `(time, cause)` rows. `cause == 2` or `3` is an exact event on that CMT with
/// the *other* cause right-censored at the same time (the cause-specific layout
/// of §3.6); `cause == 0` (or any other value) is right-censored on both. One
/// subject per row, entry time 0.
#[cfg(feature = "survival")]
pub fn tte_competing_pop(rows: &[(f64, u8)]) -> ferx_core::types::Population {
    use ferx_core::types::{EventType, ObsRecord, Population};
    let subjects = rows
        .iter()
        .enumerate()
        .map(|(i, &(t, cause))| {
            let rec = |cmt: usize| ObsRecord::Event {
                time: t,
                event_type: if cause == cmt as u8 {
                    EventType::Exact
                } else {
                    EventType::RightCensored
                },
                entry_time: 0.0,
                cmt,
            };
            let mut s = subject(&format!("{}", i + 1), vec![], vec![], vec![], vec![]);
            s.obs_records = vec![rec(2), rec(3)];
            s
        })
        .collect();

    Population {
        covariate_names: vec![],
        dv_column: "DV".to_string(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects,
    }
}

/// Build a binary / logistic population (#760): `subjects[i] = (x, &[(time, state)])`
/// is one subject with baseline covariate `X = x` and the given binary observations
/// on `cmt` (`state ∈ {0,1}`, though the state validation is exercised separately by
/// passing a `2`). Times feed the per-record `TIME` builtin in the logit predictor.
/// One `DiscreteState` `ObsRecord` per observation; no doses.
#[cfg(feature = "survival")]
pub fn binary_pop(subjects: &[(f64, Vec<(f64, u8)>)], cmt: usize) -> ferx_core::types::Population {
    use ferx_core::types::{ObsRecord, Population};
    let subjects = subjects
        .iter()
        .enumerate()
        .map(|(i, (x, obs))| {
            let mut s = subject(&format!("{}", i + 1), vec![], vec![], vec![], vec![]);
            s.covariates.insert("X".to_string(), *x);
            s.obs_records = obs
                .iter()
                .map(|&(t, st)| ObsRecord::DiscreteState {
                    time: t,
                    raw_time: t,
                    state: st as usize,
                    cmt,
                })
                .collect();
            s
        })
        .collect();

    Population {
        covariate_names: vec!["X".to_string()],
        dv_column: "DV".to_string(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects,
    }
}

/// Each window AUC of an adaptive run matches `refs[i]` to `tol` relative (#943).
///
/// `windows_in(band)` runs the scenario with `auc_target = band` and returns how many
/// realized windows fall in it — the only view of the window AUCs the public API gives.
/// The bands are first checked pairwise disjoint, so a count of 1 in band `i` is window `i`
/// and no other; tightening the reference spacing or widening the band past that point
/// fails here, by name, instead of quietly letting two windows swap.
pub fn assert_window_aucs_match(
    windows_in: &dyn Fn((f64, f64)) -> usize,
    refs: &[f64],
    tol: f64,
    label: &str,
) {
    let band_of = |r: f64| (r * (1.0 - tol), r * (1.0 + tol));
    for (i, &a) in refs.iter().enumerate() {
        for (j, &b) in refs.iter().enumerate().skip(i + 1) {
            let (ba, bb) = (band_of(a), band_of(b));
            assert!(
                ba.1 < bb.0 || bb.1 < ba.0,
                "{label}: windows {i} and {j}: references {a} and {b} are closer than the \
                 band ({tol:e}), so a count of 1 no longer identifies the window"
            );
        }
    }
    assert_eq!(
        windows_in((0.0, f64::MAX)),
        refs.len(),
        "{label}: number of realized windows"
    );
    for (i, &r) in refs.iter().enumerate() {
        assert_eq!(
            windows_in(band_of(r)),
            1,
            "{label}: window {i}: ferx's AUC24 is not within {tol:e} of mrgsolve's {r}"
        );
    }
}

/// The number of realized windows `res` scores in its `auc_target` band.
pub fn windows_in_band(res: &ferx_core::AdaptiveSimulationResult) -> usize {
    let n_windows = res.decisions.len() - 1;
    let attain = res.metrics[0]
        .auc_target_attainment
        .expect("auc_target declared");
    let n = attain * n_windows as f64;
    assert!(
        (n - n.round()).abs() < 1e-9,
        "attainment {attain} is k/{n_windows}"
    );
    n.round() as usize
}
