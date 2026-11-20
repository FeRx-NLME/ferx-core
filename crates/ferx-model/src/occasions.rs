//! Model-side IOV occasion derivation, shared by the fit / simulate entry points (`api`) and
//! the nested-level binding (`levels`). Crate-level so `levels` need not depend on `api`
//! (#3041).
use crate::types::*;

/// Derive each subject's IOV occasion labels from a model-side rule
/// ([`IovOccasionRule::PerDose`] or [`IovOccasionRule::TimeWindows`]), writing
/// `Subject::occasions` (parallel to `obs_times`), `Subject::dose_occasions`
/// (parallel to `doses`), `Subject::pk_only_occasions` (parallel to
/// `pk_only_times`, #1876) and `Subject::reset_occasions` (parallel to
/// `reset_times`, #1944). This is the DSL alternative to a dataset occasion
/// column; downstream grouping (`split_obs_by_occasion`) is identical either way.
///
/// When `had_column` is true the model *also* set `iov_column`; the DSL rule wins
/// and a one-time override warning is pushed. A `Column` rule must never reach
/// here (the caller only clones the population when a derived rule is active).
pub fn apply_iov_occasion_rule(
    population: &mut Population, rule: &IovOccasionRule, had_column: bool,
    warnings: &mut Vec<String>,
) {
    if had_column {
        warnings.push(
            "iov_occasion is set in [fit_options], so the model-side occasion rule is used and \
             the iov_column dataset column is ignored."
                .to_string(),
        );
    }
    // `Column` supplies no derived labels — nothing to do (the caller only clones
    // the population for a derived rule, so this is a defensive no-op).
    if *rule == IovOccasionRule::Column {
        return;
    }
    // Occasion of a scalar time under a set of strictly-increasing breakpoints:
    // the number of edges at or before `t` (so `time(24,48)` → 0 | 1 | 2).
    let window_occ =
        |edges: &[f64], t: f64| -> u32 { edges.iter().filter(|&&e| e <= t).count() as u32 };
    for s in population.subjects.iter_mut() {
        match rule {
            IovOccasionRule::Column => unreachable!("handled by the early return above"),
            IovOccasionRule::TimeWindows(edges) => {
                // Bucket observations and doses on the *same* internal event clock.
                // For reset-stacked subjects (EVID 3/4) that clock is monotonic and
                // already shifted so each reset segment sits past the previous one,
                // so the breakpoints separate periods correctly. Using the raw user
                // TIME for observations (as an earlier version did) is wrong on two
                // counts: it restarts each period — folding period 2 back into
                // occasion 0 — and it puts observations on a different clock than the
                // doses (whose only stored time is the internal one), so a dose could
                // land in a different occasion than its co-timed samples (#757).
                s.occasions = s.obs_times.iter().map(|&t| window_occ(edges, t)).collect();
                s.dose_occasions = s.doses.iter().map(|d| window_occ(edges, d.time)).collect();
                // A pk-only row on the same clock (#1876). One the reader moved a ULP below
                // an edge (#1810) lands in the earlier window, as a #614-moved observation
                // does: that window holds the whole interval the row governs.
                s.pk_only_occasions =
                    s.pk_only_times.iter().map(|&t| window_occ(edges, t)).collect();
                // A reset row is a record too, and `reset_times` is on the same clock.
                // Left alone, it kept the dataset's labels under `iov_column`, or stayed
                // empty and ran under κ = 0.
                s.reset_occasions = s.reset_times.iter().map(|&t| window_occ(edges, t)).collect();
            }
            IovOccasionRule::PerDose => {
                // Each distinct administration *time* opens a new occasion. Doses
                // that share a time (e.g. a simultaneous depot + central bolus, or a
                // split dual-route dose) share one occasion rather than each spawning
                // its own — indexing by dose position would leave the earlier
                // co-timed dose stranded in an empty occasion, since observations can
                // only map to one of them (#757). Doses are stored time-sorted, so a
                // running counter that ticks on each time change is the label.
                let mut dose_occ = Vec::with_capacity(s.doses.len());
                let mut occ = 0u32;
                let mut prev: Option<f64> = None;
                for d in &s.doses {
                    if let Some(p) = prev {
                        if d.time != p {
                            occ += 1;
                        }
                    }
                    dose_occ.push(occ);
                    prev = Some(d.time);
                }
                s.dose_occasions = dose_occ;
                // Distinct dose times, ascending (one per occasion label above).
                let mut dose_times: Vec<f64> = Vec::new();
                for d in &s.doses {
                    if dose_times.last() != Some(&d.time) {
                        dose_times.push(d.time);
                    }
                }
                // Observation occasion = most recent dose at or before its time
                // (dose-before-obs tie-break at equal TIME); pre-first-dose rows fold
                // into occasion 0. obs_times are ascending, so a single forward walk
                // over the distinct dose times is linear rather than O(obs*doses)
                // (#757 cleanup).
                let mut j = 0usize;
                s.occasions = s
                    .obs_times
                    .iter()
                    .map(|&t| {
                        while j < dose_times.len() && dose_times[j] <= t {
                            j += 1;
                        }
                        j.saturating_sub(1) as u32
                    })
                    .collect();
                // A pk-only row by the same rule (#1876). Searched rather than walked, so
                // an in-memory subject's unsorted `pk_only_times` still maps correctly.
                s.pk_only_occasions = s
                    .pk_only_times
                    .iter()
                    .map(|&t| dose_times.partition_point(|&d| d <= t).saturating_sub(1) as u32)
                    .collect();
                // A reset row by the same rule. At-or-before puts an EVID=4 row, or an
                // EVID=3 row co-timed with a dose, in the occasion that dose opens.
                s.reset_occasions = s
                    .reset_times
                    .iter()
                    .map(|&t| dose_times.partition_point(|&d| d <= t).saturating_sub(1) as u32)
                    .collect();
            }
        }
    }
}
