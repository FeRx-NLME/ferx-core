//! The records of a population the likelihood can see (#2778).
//!
//! A dose after a subject's last scored record cannot reach any prediction that enters the
//! objective, but every walk pays for it: `break_times` holds each dose's arrival, the walk
//! integrates on through them past the last observation, and on a smooth-input model each
//! RHS evaluation of those extra segments loops over every dose again, so the cost grows
//! with the square of the trailing doses. Long-treatment datasets, whose dosing records run
//! to end of treatment while sampling stops early, are mostly such doses.
//!
//! [`scored_population`] gives the estimators a copy of the population cut at each subject's
//! horizon. It is a view, not a data filter: the reader, the data-derived covariate
//! statistics, validation, `simulate()`, `predict()` and the population a caller holds all
//! keep every record, and `fit()` hands the full records to the outputs that read past the
//! horizon (the `[derived]` grid integrals).
//!
//! Cutting here rather than stopping each walk at the horizon leaves the engines alone: a
//! subject on the view takes the route the same data with those rows deleted would, so one
//! tail record of an unsupported kind no longer sends the subject to finite differences.
//! That route change is the one way the view moves a result by more than rounding: two routes
//! break an integrated walk at different records, so they agree only to solver tolerance.

use std::borrow::Cow;

use crate::types::{CompiledModel, ObsRecord, Population, Subject};

#[cfg(test)]
thread_local! {
    /// Test switch: `fit()` on this thread scores every record, as it did before #2778, so
    /// a test can compare a fit with and without the view on one dataset.
    pub(super) static SCORE_ALL_RECORDS: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

/// The time of the last record of `subject` that enters the likelihood, of any endpoint.
/// `None` for a subject with no scored record, or a non-finite time.
///
/// Censored rows are observations here (`Subject::cens`), so they count. A non-Gaussian
/// record contributes its time; an interval-censored event also its right bound, which the
/// survival term reads the cumulative hazard at.
fn last_scored_time(subject: &Subject) -> Option<f64> {
    let record_times = subject.obs_records.iter().flat_map(|r| match r {
        #[cfg(feature = "survival")]
        ObsRecord::Event {
            time, event_type, ..
        } => {
            let right = match event_type {
                crate::types::EventType::IntervalCensored { right, .. } => *right,
                _ => *time,
            };
            [*time, right]
        }
        ObsRecord::DiscreteState { time, .. } | ObsRecord::Count { time, .. } => [*time, *time],
    });
    let mut last = None;
    for t in subject.obs_times.iter().copied().chain(record_times) {
        if !t.is_finite() {
            return None;
        }
        last = Some(last.map_or(t, |l: f64| l.max(t)));
    }
    last
}

/// The time past which `subject`'s records can go, or `None` when none can.
///
/// Normally the last scored record. A record at the horizon stays: listed ahead of the
/// observation it acts on it. One listed after it sits a ULP later, moved there by the
/// reader, so it goes, and the observation never read it.
///
/// A non-Gaussian record is not on the engines' record timeline. The parameters of the
/// interval holding it come from the record that closes the interval, the first record
/// after it, so a subject with such records keeps that one too.
fn cutoff(subject: &Subject) -> Option<f64> {
    let horizon = last_scored_time(subject)?;
    if subject.obs_records.is_empty() {
        return Some(horizon);
    }
    let next_record = subject
        .doses
        .iter()
        .map(|d| d.time)
        .chain(subject.pk_only_times.iter().copied())
        .chain(subject.reset_times.iter().copied())
        .filter(|&t| t > horizon)
        .fold(f64::INFINITY, f64::min);
    Some(if next_record.is_finite() {
        next_record
    } else {
        horizon
    })
}

/// How many leading entries of `times` to keep: through the last one at or before `cutoff`,
/// so the kept records are a prefix however the list is ordered, and an `ADDL`-implied
/// dose's `parent` index stays valid. A `NaN` time is kept for the engines to refuse.
fn kept_prefix(
    mut times: impl DoubleEndedIterator<Item = f64> + ExactSizeIterator,
    cutoff: f64,
) -> usize {
    times
        .rposition(|t| t <= cutoff || t.is_nan())
        .map_or(0, |i| i + 1)
}

/// Cut a per-record occasion list to its first `keep` entries, then add back each label
/// only the cut records carried.
///
/// A label no kept record carries still has a κ block in the subject's stacked layout
/// (`iov_occasion_groups`), which the fit's `kappas` and everything reading them are laid
/// out by. Its κ is uninformed (`iov_group_informed`), so the block costs no integration,
/// and keeping the label keeps the layout: the labels are appended past the records, where
/// the by-index readers never look, in first-seen order, which is the order groups are
/// built in. A list with no such label stays parallel to its records.
fn cut_occasions(occasions: &mut Vec<u32>, keep: usize) {
    if occasions.len() <= keep {
        return;
    }
    let tail = occasions.split_off(keep);
    let mut seen: std::collections::HashSet<u32> = occasions.iter().copied().collect();
    occasions.extend(tail.into_iter().filter(|&occ| seen.insert(occ)));
}

/// How many leading records of each kind a subject keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Kept {
    doses: usize,
    pk_only: usize,
    resets: usize,
}

/// What `subject` keeps of its records, or `None` when it keeps them all.
///
/// The first dose always stays: `TAD` and `TAFD` read it before it arrives
/// (`tad_anchor_for`, `earliest_dose_time`), and a subject with only pre-dose observations
/// would otherwise lose that anchor.
fn kept_records(subject: &Subject) -> Option<Kept> {
    let cutoff = cutoff(subject)?;
    let kept = Kept {
        doses: kept_prefix(subject.doses.iter().map(|d| d.time), cutoff).max(1),
        pk_only: kept_prefix(subject.pk_only_times.iter().copied(), cutoff),
        resets: kept_prefix(subject.reset_times.iter().copied(), cutoff),
    };
    let has_tail = kept.doses < subject.doses.len()
        || kept.pk_only < subject.pk_only_times.len()
        || kept.resets < subject.reset_times.len();
    has_tail.then_some(kept)
}

/// Remove `subject`'s records beyond `kept`, with everything parallel to them.
fn cut_subject(subject: &mut Subject, kept: Kept) {
    subject.doses.truncate(kept.doses);
    subject.dose_covariates.truncate(kept.doses);
    cut_occasions(&mut subject.dose_occasions, kept.doses);

    subject.pk_only_times.truncate(kept.pk_only);
    subject.pk_only_raw_times.truncate(kept.pk_only);
    subject.pk_only_covariates.truncate(kept.pk_only);
    cut_occasions(&mut subject.pk_only_occasions, kept.pk_only);
    // A toggle is a pk-only row, and turning a compartment off past the horizon only moves
    // a kept dose's `off_at` out of the scored span.
    subject.cmt_toggles.retain(|c| c.pk_only < kept.pk_only);

    subject.reset_times.truncate(kept.resets);
    subject.reset_covariates.truncate(kept.resets);
    cut_occasions(&mut subject.reset_occasions, kept.resets);
}

/// `population` with every subject's records past its horizon removed; see the module docs.
/// Copied only when a subject has such a record; an owned population is cut in place.
///
/// Runs on the output of [`crate::api::fitted_population`], so a derived `iov_occasion` rule
/// has labelled the whole timeline first. A covariate that varied only on a removed row no
/// longer makes its subject time-varying, so the pruning runs again and the subject takes
/// the route it would on the data without those rows.
///
/// A nested-level model is left whole: its cells are checked to be parallel to the records
/// (`cells_ready`), which [`cut_occasions`] does not promise.
pub(crate) fn scored_population<'a>(
    model: &CompiledModel,
    population: Cow<'a, Population>,
) -> Cow<'a, Population> {
    #[cfg(test)]
    if SCORE_ALL_RECORDS.with(std::cell::Cell::get) {
        return population;
    }
    if model.has_nested_levels() {
        return population;
    }
    let kept: Vec<Option<Kept>> = population.subjects.iter().map(kept_records).collect();
    if kept.iter().all(Option::is_none) {
        return population;
    }
    let mut population = population;
    let p = population.to_mut();
    for (subject, kept) in p.subjects.iter_mut().zip(kept) {
        if let Some(kept) = kept {
            cut_subject(subject, kept);
        }
    }
    p.prune_irrelevant_tv_covariates(&model.referenced_covariates);
    population
}

#[cfg(test)]
#[path = "scoring_horizon_tests.rs"]
mod tests;
