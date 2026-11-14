//! [`FitStatus`]: one usability verdict beside [`FitResult::converged`] (#2799).
//!
//! `converged` keeps its meaning — the outer optimizer's stop, verified — and nothing here
//! writes it. The status reads it together with the facts a reader would otherwise have to
//! gather from four places (the evaluation budget, the critical warnings, the covariance
//! step, and the coordinates held at ferx's internal guard), each from **one** typed source.
//! It is computed on read, never stored, so it cannot go stale when `run_covariance` changes
//! the step's outcome or a `.fitrx` written before it existed is loaded.

use super::{CovarianceStatus, FitResult, HeldReason, WarningCode, WarningSeverity};

/// Whether a fit can be used as it stands: [`FitResult::status`].
///
/// The rule is one line: `!converged` → [`NotConverged`](Self::NotConverged); `converged`
/// with no [`StatusReason`] → [`Converged`](Self::Converged); otherwise
/// [`ConvergedWithIssues`](Self::ConvergedWithIssues). The reasons are computed the same way
/// whatever `converged` says, so a not-converged fit lists them too.
#[non_exhaustive]
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FitStatus {
    /// The optimizer's stop was verified and nothing else qualifies it.
    Converged,
    /// The optimizer's stop was verified, but at least one reason says the result needs a
    /// second look before it is used.
    ConvergedWithIssues {
        /// In the fixed order of [`StatusReason`]'s variants.
        reasons: Vec<StatusReason>,
    },
    /// The optimizer did not report a verified stop. `reasons` says what else is known
    /// (empty for a plain non-convergence).
    NotConverged {
        /// In the fixed order of [`StatusReason`]'s variants.
        reasons: Vec<StatusReason>,
    },
}

/// One fact that qualifies a [`FitStatus`], listed in this order (the variants' order):
/// evaluation-only, budget spent, each critical warning code (first appearance), covariance
/// failed, collapsed at the guard.
#[non_exhaustive]
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StatusReason {
    /// No stage estimated: the run scored the given values by request
    /// ([`FitResult::evaluation_only`]: `maxiter = 0` on an estimator that honours it, or a
    /// standalone `imp_eval_only` / `agq_eval_only` chain).
    EvaluationOnly,
    /// The outer optimizer spent its evaluation budget ([`FitResult::eval_budget_spent`]).
    BudgetSpent,
    /// A `Critical` entry of [`FitResult::warnings_structured`], one per distinct code, in
    /// first-appearance order. Never `covariance_failed`, which is read from
    /// [`FitResult::covariance_status`] as [`StatusReason::CovarianceFailed`] instead. An
    /// earlier chain stage's entry counts.
    CriticalWarning(WarningCode),
    /// The covariance step ran and failed ([`CovarianceStatus::Failed`]).
    CovarianceFailed,
    /// Coordinates collapsed onto ferx's internal guard ([`HeldReason::CollapsedAtGuard`] in
    /// [`FitResult::held_at_bound`]) that a covariance step which ran did **not** hold — a
    /// failed step, a floored `R`, or an `S` fallback. A collapse the step held (marked in
    /// [`FitResult::cov_fixed`]) is a clean answer and is not listed; neither is one with no
    /// step to judge it, nor a profiled level on its declared box.
    CollapsedAtGuard {
        /// The parameters' names, in packed order.
        parameters: Vec<String>,
    },
}

impl FitStatus {
    /// The console label: `CONVERGED`, `CONVERGED WITH ISSUES`, `NOT CONVERGED`.
    pub fn label(&self) -> &'static str {
        match self {
            FitStatus::Converged => "CONVERGED",
            FitStatus::ConvergedWithIssues { .. } => "CONVERGED WITH ISSUES",
            FitStatus::NotConverged { .. } => "NOT CONVERGED",
        }
    }

    /// The snake_case token the YAML writes under `status:`: `converged`,
    /// `converged_with_issues`, `not_converged`.
    pub fn token(&self) -> &'static str {
        match self {
            FitStatus::Converged => "converged",
            FitStatus::ConvergedWithIssues { .. } => "converged_with_issues",
            FitStatus::NotConverged { .. } => "not_converged",
        }
    }

    /// The reasons, in their fixed order; empty for [`FitStatus::Converged`].
    pub fn reasons(&self) -> &[StatusReason] {
        match self {
            FitStatus::Converged => &[],
            FitStatus::ConvergedWithIssues { reasons } | FitStatus::NotConverged { reasons } => {
                reasons
            }
        }
    }
}

impl StatusReason {
    /// The printed form: `evaluation_only`, `budget_spent`, the warning code's token (e.g.
    /// `condition_number`), `covariance_failed`, `collapsed_at_guard: ETA_KA, ETA_CL`.
    pub fn label(&self) -> String {
        match self {
            StatusReason::EvaluationOnly => "evaluation_only".to_string(),
            StatusReason::BudgetSpent => "budget_spent".to_string(),
            StatusReason::CriticalWarning(code) => code.as_str().to_string(),
            StatusReason::CovarianceFailed => "covariance_failed".to_string(),
            StatusReason::CollapsedAtGuard { parameters } => {
                format!("collapsed_at_guard: {}", parameters.join(", "))
            }
        }
    }
}

impl FitResult {
    /// The usability verdict beside [`Self::converged`] (#2799): see [`FitStatus`].
    ///
    /// Computed from the result's own fields on every call, so it is current after
    /// `run_covariance` and available on a fit loaded from a `.fitrx` or JSON file. It never
    /// changes `converged`, and multi-start selection does not read it.
    pub fn status(&self) -> FitStatus {
        let reasons = status_reasons(self);
        if !self.converged {
            FitStatus::NotConverged { reasons }
        } else if reasons.is_empty() {
            FitStatus::Converged
        } else {
            FitStatus::ConvergedWithIssues { reasons }
        }
    }
}

/// Every [`StatusReason`] that applies to `r`, in the fixed order. Independent of
/// `r.converged`.
fn status_reasons(r: &FitResult) -> Vec<StatusReason> {
    let mut reasons = Vec::new();
    if r.evaluation_only {
        reasons.push(StatusReason::EvaluationOnly);
    }
    if r.eval_budget_spent {
        reasons.push(StatusReason::BudgetSpent);
    }
    // One source per reason: the covariance failure is read from its typed status below, so
    // its Critical warning is not counted a second time here — which also keeps a stale
    // `covariance_failed` entry from an earlier chain stage from contradicting a final
    // `Computed`.
    let mut codes: Vec<WarningCode> = Vec::new();
    for w in &r.warnings_structured {
        if w.severity == WarningSeverity::Critical
            && w.category != WarningCode::CovarianceFailed
            && !codes.contains(&w.category)
        {
            codes.push(w.category);
        }
    }
    reasons.extend(codes.into_iter().map(StatusReason::CriticalWarning));
    if r.covariance_status == CovarianceStatus::Failed {
        reasons.push(StatusReason::CovarianceFailed);
    }
    // A collapse is an issue only when a step ran and did not hold it. `cov_fixed` marks a
    // collapse only when the step held it (a clean `R`); read through `get(..)`, so a mask
    // left empty beside a matrix reads "not held" — the conservative side.
    if r.covariance_status != CovarianceStatus::NotRequested {
        let parameters: Vec<String> = r
            .held_at_bound
            .iter()
            .filter(|h| h.reason == HeldReason::CollapsedAtGuard)
            .filter(|h| !r.cov_fixed.get(h.packed_index).copied().unwrap_or(false))
            .map(|h| h.name.clone())
            .collect();
        if !parameters.is_empty() {
            reasons.push(StatusReason::CollapsedAtGuard { parameters });
        }
    }
    reasons
}

#[cfg(test)]
#[path = "types_fit_status_tests.rs"]
mod tests;
