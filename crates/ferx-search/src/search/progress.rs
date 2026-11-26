//! The per-candidate progress event every search tool emits (#2512).
//!
//! The step/layer/batch events each tool already has say when a group of fits
//! starts and which one won; a log tailing a long search wants a line per
//! candidate as it finishes. That event is the same shape in every tool, since
//! every tool fits its candidates through the one [`Runner`](super::Runner), so
//! it is defined once here and each tool's event enum wraps it in a
//! `CandidateFinished` variant.

use super::CandidateResult;

/// One candidate's outcome, reported as soon as it is known.
///
/// A fitted candidate is reported from the worker that fitted it, so on a
/// thread pool the order is completion order, not submission order. A
/// candidate that cost no fit — its outcome read back from the run's journal
/// or a [`reuse_from`](super::Runner::reuse_from) directory, or a duplicate of
/// another candidate's canonical text — is reported once the step's fits are
/// done. Either way every candidate of a step is reported exactly once, before
/// the step's own finished event, so a sink that counts these against the
/// step's `candidates` reaches it — except on a cancelled run, where the
/// candidates the run never reached are not reported at all.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CandidateFinished {
    /// The step the candidate belongs to: the directory a cached run journals
    /// it under (`base`, `forward-2`, `layer-1`, `gen-3`, …).
    pub step: String,
    /// [`Candidate::id`](super::Candidate::id).
    pub id: String,
    /// [`Candidate::parent`](super::Candidate::parent).
    pub parent: Option<String>,
    /// The candidate's features as `candidates.csv` renders them
    /// ([`FeatureVector::render`](super::FeatureVector::render)); empty for a
    /// candidate that carries none, such as a search's input model.
    pub description: String,
    /// `None` when there is no fit.
    pub ofv: Option<f64>,
    /// The search's ranking criterion on the fit; `None` when there is no fit
    /// to evaluate it on.
    pub criterion: Option<f64>,
    /// Whether the candidate can be ranked: it fitted and passed every enabled
    /// strictness gate ([`CandidateResult::eligible`]).
    pub passed: bool,
    /// Why it cannot be, when it cannot: the failed gates joined with `; `, or
    /// the compile / fit error.
    pub failed_gate: Option<String>,
    /// The outcome came from a journal or another search's directory rather
    /// than from a fit in this run.
    pub reused: bool,
    /// Set when the candidate rendered to the same model as an earlier one in
    /// the step: the id named is the one that was fitted.
    pub duplicate_of: Option<String>,
    /// Wall-clock seconds the fit took; `0.0` when no fit was run.
    pub seconds: f64,
}

impl CandidateFinished {
    pub(crate) fn of(step: &str, result: &CandidateResult) -> Self {
        let failed_gate = if result.eligible() {
            None
        } else if !result.verdict.failures.is_empty() {
            Some(result.verdict.failures.join("; "))
        } else if let Some(e) = &result.error {
            Some(e.message.clone())
        } else {
            // Passed every gate but has no finite criterion to rank on.
            Some("no finite criterion".to_string())
        };
        Self {
            step: step.to_string(),
            id: result.id.clone(),
            parent: result.parent.clone(),
            description: result.features.render(),
            ofv: result.ofv,
            criterion: result.criterion.is_finite().then_some(result.criterion),
            passed: result.eligible(),
            failed_gate,
            reused: result.reused,
            duplicate_of: result.duplicate_of.clone(),
            seconds: result.seconds,
        }
    }
}

/// A sink for [`CandidateFinished`], called from whichever thread finished the
/// candidate.
pub(crate) type CandidateFn<'a> = &'a (dyn Fn(CandidateFinished) + Sync);

#[cfg(test)]
mod tests {
    use ferx_core::StrictnessVerdict;

    use super::*;
    use crate::search::{CandidateError, FeatureVector};

    fn result(criterion: f64, failures: &[&str], error: Option<&str>) -> CandidateResult {
        CandidateResult {
            id: "c".into(),
            hash: "h".into(),
            parent: Some("base".into()),
            features: FeatureVector::new().with("CL-WT", "exp"),
            fit: None,
            ofv: criterion.is_finite().then_some(500.0),
            converged: Some(true),
            verdict: StrictnessVerdict {
                passed: failures.is_empty(),
                failures: failures.iter().map(|f| f.to_string()).collect(),
                skipped: Vec::new(),
            },
            criterion,
            seconds: 1.5,
            error: error.map(CandidateError::model),
            duplicate_of: None,
            reused: false,
        }
    }

    #[test]
    fn an_eligible_candidate_reports_its_numbers_and_no_gate() {
        let c = CandidateFinished::of("layer-2", &result(510.0, &[], None));
        assert_eq!(c.step, "layer-2");
        assert_eq!(c.parent.as_deref(), Some("base"));
        assert_eq!(c.description, "CL-WT=exp");
        assert_eq!((c.ofv, c.criterion), (Some(500.0), Some(510.0)));
        assert!(c.passed);
        assert_eq!(c.failed_gate, None);
        assert_eq!(c.seconds, 1.5);
    }

    #[test]
    fn a_gated_candidate_names_every_failed_gate() {
        let c = CandidateFinished::of("s", &result(510.0, &["did not converge", "stalled"], None));
        assert!(!c.passed);
        // The criterion is still reported: the gate says "do not rank this".
        assert_eq!(c.criterion, Some(510.0));
        assert_eq!(c.failed_gate.as_deref(), Some("did not converge; stalled"));
    }

    #[test]
    fn a_candidate_with_no_finite_criterion_says_so() {
        let error = CandidateFinished::of("s", &result(f64::NAN, &[], Some("no compile")));
        assert_eq!(error.criterion, None);
        assert_eq!(error.failed_gate.as_deref(), Some("no compile"));
        let nan = CandidateFinished::of("s", &result(f64::NAN, &[], None));
        assert!(!nan.passed);
        assert_eq!(nan.failed_gate.as_deref(), Some("no finite criterion"));
    }
}
