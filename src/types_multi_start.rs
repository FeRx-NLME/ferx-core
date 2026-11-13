//! What each start of a multi-start fit did (#2867): the per-start record kept on
//! [`FitResult::starts`](super::FitResult::starts), and the one summary of their spread that
//! the `multi_start` warning line and the fit YAML's `multi_start:` block both print.
//!
//! Reporting only. Selection is `api::fit::multistart_prefers` over the starts' own results;
//! nothing here is read by it.

use super::FitStatus;

/// One start of a multi-start fit (`n_starts > 1`): a row of
/// [`FitResult::starts`](super::FitResult::starts).
///
/// A start either ran (`error` is `None`) or was refused or failed before it returned a
/// result (`error` is `Some`, and `ofv`, `status` and `theta` are empty).
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq)]
pub struct StartRecord {
    /// Start index. `0` is the model file's initial values, unperturbed.
    pub start: usize,
    /// The start's reported OFV, as [`FitResult::ofv`](super::FitResult::ofv) would carry it.
    /// `None` for a failed start, and for a start whose OFV is not finite (a VI fit under the
    /// default `vi_final_ofv = none`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ofv: Option<f64>,
    /// The value the selection ranked this start on, when that is not `ofv`: under `nn_l2` /
    /// `nn_smooth` the starts are ranked on the penalized objective, `ofv` plus the penalty.
    /// `None` when no penalty applies, and wherever `ofv` is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective: Option<f64>,
    /// The start's own `converged`. `false` for a failed start.
    pub converged: bool,
    /// The start's own [`FitResult::status`](super::FitResult::status). `None` for a failed
    /// start.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<FitStatus>,
    /// Outer iterations the start ran.
    pub n_iterations: usize,
    /// Wall time of this start alone, in seconds. The starts run in parallel, so these do not
    /// add up to the fit's wall time.
    pub wall_time_secs: f64,
    /// The start's final θ, in [`FitResult::theta_names`](super::FitResult::theta_names)
    /// order, so two different optima can be told apart without a refit. Empty for a failed
    /// start.
    pub theta: Vec<f64>,
    /// `true` on the one start whose result the fit returned.
    pub selected: bool,
    /// The message of a start that failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// The failed start's diagnostic code (`EngineError::code`), when its refusal carries one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
}

/// Two starts whose ranked values differ by at most this much are reported as agreeing.
///
/// Fixed, not tied to a convergence tolerance (#2867): measured over 30 multi-start fits,
/// starts that stop on the same optimum sit at most 2.3e-6 apart under the gradient
/// optimizers and at most 0.33 under BOBYQA, and the closest different stop is 5.00 away.
pub(crate) const START_AGREE_OFV: f64 = 1.0;

/// How many strays the summary line names before it says "and N more".
const NAMED_STARTS: usize = 3;

impl StartRecord {
    /// The value this start was ranked on, when it is one the ranking treats as a real
    /// objective (`estimation::outer_optimizer::ofv_is_valid`).
    fn ranked(&self) -> Option<f64> {
        self.objective
            .or(self.ofv)
            .filter(|v| crate::estimation::outer_optimizer::ofv_is_valid(*v))
    }

    fn ran(&self) -> bool {
        self.error.is_none()
    }
}

/// The spread of a multi-start fit's starts: every number the summary line and the YAML
/// block print, computed once.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StartSpread {
    pub(crate) n_starts: usize,
    /// Starts that ran and report `converged`.
    pub(crate) n_converged: usize,
    /// Starts that ran and do not.
    pub(crate) n_not_converged: usize,
    /// Starts that returned no result.
    pub(crate) n_failed: usize,
    /// Starts that ran but have no value the ranking treats as an objective.
    pub(crate) n_unusable: usize,
    /// The returned start, and whether it converged.
    pub(crate) selected: Option<(usize, bool)>,
    /// The returned start's ranked value. `None` when it has none to compare.
    pub(crate) best: Option<f64>,
    /// The ranked values are penalized objectives, not OFVs.
    pub(crate) penalized: bool,
    /// Starts compared with the returned one: the converged starts with a usable value when
    /// the returned start converged, every start with a usable value when none did.
    pub(crate) n_compared: usize,
    /// Of those, the ones within [`START_AGREE_OFV`] of the returned start (itself included).
    pub(crate) n_within: usize,
    /// The largest distance above the returned start among the compared starts.
    pub(crate) range: Option<f64>,
    /// Compared starts more than [`START_AGREE_OFV`] above the returned one, with their
    /// distance, in start order.
    pub(crate) above: Vec<(usize, f64)>,
    /// Starts that did not converge yet stopped below the returned (converged) start, with
    /// their (negative) distance, in start order.
    pub(crate) lower_not_converged: Vec<(usize, f64)>,
}

impl StartSpread {
    pub(crate) fn of(starts: &[StartRecord]) -> Self {
        let ran = || starts.iter().filter(|s| s.ran());
        let selected = starts.iter().find(|s| s.selected);
        let best = selected.and_then(StartRecord::ranked);
        let selected_converged = selected.is_some_and(|s| s.converged);
        let mut spread = StartSpread {
            n_starts: starts.len(),
            n_converged: ran().filter(|s| s.converged).count(),
            n_not_converged: ran().filter(|s| !s.converged).count(),
            n_failed: starts.len() - ran().count(),
            n_unusable: ran().filter(|s| s.ranked().is_none()).count(),
            selected: selected.map(|s| (s.start, s.converged)),
            best,
            penalized: starts.iter().any(|s| s.objective.is_some()),
            n_compared: 0,
            n_within: 0,
            range: None,
            above: Vec::new(),
            lower_not_converged: Vec::new(),
        };
        let Some(best) = best else {
            return spread;
        };
        for s in ran() {
            let Some(v) = s.ranked() else { continue };
            let d = v - best;
            if s.converged == selected_converged {
                spread.n_compared += 1;
                if d <= START_AGREE_OFV {
                    spread.n_within += 1;
                } else {
                    spread.above.push((s.start, d));
                }
                spread.range = Some(spread.range.map_or(d, |r: f64| if d > r { d } else { r }));
            } else if selected_converged && d < 0.0 {
                spread.lower_not_converged.push((s.start, d));
            }
        }
        spread
    }

    /// What the ranked value is called.
    pub(crate) fn value_label(&self) -> &'static str {
        if self.penalized {
            "penalized objective"
        } else {
            "OFV"
        }
    }

    /// The one `multi_start` line a multi-start fit carries in its warnings.
    pub(crate) fn line(&self) -> String {
        let mut head = format!(
            "Multi-start: {} starts, {} converged",
            self.n_starts, self.n_converged
        );
        if self.n_not_converged > 0 {
            head.push_str(&format!(", {} not converged", self.n_not_converged));
        }
        if self.n_failed > 0 {
            head.push_str(&format!(", {} failed", self.n_failed));
        }
        let Some((k, converged)) = self.selected else {
            return head;
        };
        let label = self.value_label();
        let mut clauses: Vec<String> = Vec::new();
        match self.best {
            None => clauses.push(format!("no {label} to compare (start {k} returned)")),
            Some(best) => {
                let note = if converged { "" } else { ", not converged" };
                clauses.push(format!("best {label} {best:.4} (start {k}{note})"));
                // Which starts the count is over: see `n_compared`.
                let set = if converged {
                    "converged starts"
                } else {
                    "starts"
                };
                let range = self.range.unwrap_or(0.0);
                if self.n_compared <= 1 {
                    clauses.push(format!(
                        "no other {} to compare it with",
                        &set[..set.len() - 1]
                    ));
                } else if self.n_within == self.n_compared {
                    clauses.push(format!(
                        "all {} {set} within {START_AGREE_OFV:.1} of best (range {range:.4})",
                        self.n_compared
                    ));
                } else {
                    clauses.push(format!(
                        "{} of {} {set} within {START_AGREE_OFV:.1} of best",
                        self.n_within, self.n_compared
                    ));
                }
                if !self.above.is_empty() {
                    let kind = if converged {
                        "converged above it"
                    } else {
                        "above it"
                    };
                    clauses.push(format!(
                        "{} {kind}: {}",
                        self.above.len(),
                        named(&self.above)
                    ));
                }
                if !self.lower_not_converged.is_empty() {
                    clauses.push(format!(
                        "{} not converged at a lower {label}: {}",
                        self.lower_not_converged.len(),
                        named(&self.lower_not_converged)
                    ));
                }
            }
        }
        if self.best.is_some() && self.n_unusable > 0 {
            clauses.push(format!("{} without a usable {label}", self.n_unusable));
        }
        format!("{head}; {}", clauses.join("; "))
    }
}

/// `start 4 (+5.27), start 6 (+16.10)`, the first [`NAMED_STARTS`] and then a count.
fn named(starts: &[(usize, f64)]) -> String {
    let mut parts: Vec<String> = starts
        .iter()
        .take(NAMED_STARTS)
        .map(|(k, d)| format!("start {k} ({d:+.2})"))
        .collect();
    if starts.len() > NAMED_STARTS {
        parts.push(format!("and {} more", starts.len() - NAMED_STARTS));
    }
    parts.join(", ")
}

#[cfg(test)]
#[path = "types_multi_start_tests.rs"]
mod tests;
