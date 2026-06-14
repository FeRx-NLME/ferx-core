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
        pk_only_covariates: vec![],
        reset_times: vec![],
        cens: vec![0; n],
        occasions: vec![],
        dose_occasions: vec![],
        #[cfg(feature = "survival")]
        obs_records: vec![],
    }
}
