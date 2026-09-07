//! What a search hands the runner, and what it gets back (#1178).
//!
//! A [`Candidate`] is a rendered model plus its provenance: the id the search
//! knows it by, the parent it was derived from, and the [`FeatureVector`]
//! describing *what makes it different* — the row a report shows next to its
//! criterion. A [`CandidateResult`] is the same identity carried back with the
//! fit, the [`StrictnessVerdict`] and the ranking criterion attached.

use std::collections::BTreeMap;

use ferx_core::{bic, BicType, FitResult, StrictnessVerdict};
use serde::{Deserialize, Serialize};

use ferx_core::edit::ModelText;

/// The search-space coordinates of one candidate: `ABSORPTION = FO`,
/// `CL-WT = pow`, `PERIPHERALS = 1`, …
///
/// The runner never interprets these — it stores them, journals them and puts
/// them in the per-candidate table. They exist so a report can say *which*
/// model won rather than only which id, and so a resumed run's table still
/// describes its rows. Ordering is by key (a `BTreeMap`), so two vectors with
/// the same entries render identically whatever order they were built in.
///
/// It is deliberately **not** the candidate's identity: two different feature
/// vectors can render to the same model text, and it is the canonical hash of
/// that text that decides whether a fit is reused (see
/// [`ModelText::canonical_hash`]).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FeatureVector {
    entries: BTreeMap<String, String>,
}

impl FeatureVector {
    /// An empty vector — a base model with nothing to say about itself.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder form of [`set`](Self::set), for `FeatureVector::new().with(…).with(…)`.
    pub fn with(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.set(key, value);
        self
    }

    /// Set one feature, returning the value it replaced.
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) -> Option<String> {
        self.entries.insert(key.into(), value.into())
    }

    /// The value of one feature, if present.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.entries.get(key).map(String::as_str)
    }

    /// Every `(key, value)` in key order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The one-cell rendering for the per-candidate table: `k=v` pairs joined
    /// by `;`, in key order.
    ///
    /// Display only — the journal round-trips the vector through serde, which
    /// has no separator to collide with, so nothing has to parse this back.
    pub fn render(&self) -> String {
        self.iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(";")
    }
}

impl<K: Into<String>, V: Into<String>> FromIterator<(K, V)> for FeatureVector {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        Self {
            entries: iter
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
        }
    }
}

/// One model a search wants fitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    /// How the search refers to this candidate. Must be unique within one
    /// [`run`](super::Runner::run) — the results are keyed by it and the table
    /// is written under it.
    pub id: String,
    /// The candidate's model source, as produced by the `ferx-core::edit`
    /// layer.
    pub model: ModelText,
    /// The candidate this one was derived from, for a step-history report.
    /// `None` for the base model or for an exhaustively enumerated candidate.
    pub parent: Option<String>,
    /// What distinguishes it — see [`FeatureVector`].
    pub features: FeatureVector,
}

impl Candidate {
    /// A candidate with no parent and no features.
    pub fn new(id: impl Into<String>, model: ModelText) -> Self {
        Self {
            id: id.into(),
            model,
            parent: None,
            features: FeatureVector::new(),
        }
    }

    /// Builder: record the candidate this one was derived from.
    pub fn parent(mut self, parent: impl Into<String>) -> Self {
        self.parent = Some(parent.into());
        self
    }

    /// Builder: attach the search-space coordinates.
    pub fn features(mut self, features: FeatureVector) -> Self {
        self.features = features;
        self
    }

    /// The candidate's identity: hex [`ModelText::canonical_hash`].
    ///
    /// This — not the id, not the features — is the dedup and cache key, so a
    /// model reached twice by two different step orders is fitted once.
    pub fn hash(&self) -> String {
        hex(&self.model.canonical_hash())
    }
}

/// Lower-case hex of a 32-byte digest.
pub(crate) fn hex(bytes: &[u8; 32]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::with_capacity(64), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// What a candidate is ranked on.
///
/// The BIC variants are `ferx_core::bic`, i.e. Pharmpy's four
/// `calculate_bic` conventions; see [`BicType`]. Every variant is
/// **lower-is-better**, which is what lets a search compare them without
/// knowing which one it was configured with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Criterion {
    /// The objective function value itself. Only comparable between nested
    /// models with the same parameter count, so a search using it supplies its
    /// own likelihood-ratio cutoff.
    Ofv,
    /// `FitResult::aic`.
    Aic,
    /// `ferx_core::bic(result, kind)`.
    Bic(BicType),
}

impl Default for Criterion {
    /// The mixed BIC — Pharmpy's default and the one its structural and IIV
    /// searches rank on.
    fn default() -> Self {
        Criterion::Bic(BicType::Mixed)
    }
}

impl Criterion {
    /// Evaluate the criterion on a finished fit.
    pub fn of(&self, result: &FitResult) -> f64 {
        match self {
            Criterion::Ofv => result.ofv,
            Criterion::Aic => result.aic,
            Criterion::Bic(kind) => bic(result, *kind),
        }
    }

    /// A short stable label for the table header and the run manifest.
    pub fn label(&self) -> &'static str {
        match self {
            Criterion::Ofv => "ofv",
            Criterion::Aic => "aic",
            Criterion::Bic(BicType::Mixed) => "bic_mixed",
            Criterion::Bic(BicType::Iiv) => "bic_iiv",
            Criterion::Bic(BicType::Random) => "bic_random",
            Criterion::Bic(BicType::Fixed) => "bic_fixed",
        }
    }
}

/// How the candidates in one [`run`](super::Runner::run) are fitted and judged.
#[derive(Debug, Clone)]
pub struct RunOptions {
    /// What [`CandidateResult::criterion`] holds.
    pub criterion: Criterion,
    /// The gate a fit must pass before its criterion is trusted. A candidate
    /// that fails is **kept**, with its reasons — see
    /// [`CandidateResult::verdict`].
    pub strictness: ferx_core::Strictness,
    /// `FitOptions::n_starts` for every candidate — the *retries* of the epic's
    /// search configuration, applied before the strictness gate. Clamped to at
    /// least 1.
    ///
    /// The default is 3, not 1: under automation an init stall (#751) or an
    /// inner-EBE mode (#864, #891) is a model-selection error, not a slow fit.
    pub n_starts: usize,
    /// Reuse the candidates already in the cache directory's journal instead of
    /// refitting them. Ignored when the runner has no cache directory.
    pub resume: bool,
    /// Fit settings for every candidate, replacing the ones in its own
    /// `[fit_options]` block.
    ///
    /// `None` — the default — lets each candidate carry its own settings, which
    /// is what a search over models edited from one base file wants: the base
    /// file's estimation method, tolerances and covariance step come along with
    /// the edit. `Some` is for a caller that needs one configuration across a
    /// space whose members disagree, or a cheaper one than the base file's.
    ///
    /// Either way the runner still applies its own overrides on top —
    /// [`quiet`](ferx_core::FitOptions::quiet), the [`PoolPlan`](ferx_core::PoolPlan)
    /// thread pin, `n_starts` and the cancellation flag — so those four are not
    /// settable here.
    pub fit_options: Option<ferx_core::FitOptions>,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            criterion: Criterion::default(),
            strictness: ferx_core::Strictness::default(),
            n_starts: 3,
            resume: false,
            fit_options: None,
        }
    }
}

/// Why a candidate produced no fit, and whether a later run should try again.
///
/// The flag is not cosmetic: a journalled outcome is what every subsequent
/// `resume: true` run believes without rechecking, and the two failures that
/// arrive here are not the same statement. A model that does not compile will
/// not compile on the next machine either — remembering that saves the parse
/// and, more importantly, keeps the candidate in the report. A fit that died
/// because `install_on_fit_pool` could not build a pool (`api/pool.rs`) says
/// nothing about the model at all, and writing *that* down as final would let
/// one bad minute mark a fittable model dead for the rest of the search's life.
///
/// From inside the runner the two are one `String`, which is why the
/// classification is made where the failure is raised rather than recovered
/// from the message afterwards.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateError {
    /// What went wrong, as shown in the report.
    pub message: String,
    /// `true` when the failure describes the run rather than the candidate, so
    /// a resumed run should fit it again instead of trusting the row.
    pub retryable: bool,
}

impl CandidateError {
    /// A failure that is a property of the candidate itself — a model that does
    /// not compile, or does not bind against this run's data. Deterministic, so
    /// a resume reuses it.
    pub fn model(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retryable: false,
        }
    }

    /// A failure raised by the run rather than by the candidate. A resume
    /// refits it.
    ///
    /// This is the default for anything `fit()` itself returns. Note that the
    /// *expected* bad outcome of a search candidate — a fit that finishes but
    /// does not converge — is not an error at all: it comes back `Ok` and is
    /// judged by [`check_strictness`](ferx_core::check_strictness), so it is
    /// journalled and reused like any other result. An `Err` out of `fit()` is
    /// the pathological case, and refitting it is cheap because it is rare.
    pub fn environment(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retryable: true,
        }
    }
}

impl std::fmt::Display for CandidateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// One candidate's outcome.
///
/// A candidate that failed its strictness gate, failed to compile or failed to
/// fit is still here, with the reason — nothing is dropped silently, because a
/// candidate missing from a search report is indistinguishable from one that
/// was never generated.
#[derive(Debug, Clone)]
pub struct CandidateResult {
    /// The [`Candidate::id`] this result belongs to.
    pub id: String,
    /// Hex canonical hash — the cache key the fit was stored under.
    pub hash: String,
    /// [`Candidate::parent`], carried through.
    pub parent: Option<String>,
    /// [`Candidate::features`], carried through.
    pub features: FeatureVector,
    /// The fit, when this run performed it.
    ///
    /// `None` in three cases, told apart by the other fields: the fit failed
    /// (`error` is set), the candidate duplicated one that *was* fitted
    /// (`duplicate_of` is set), or the outcome was read back from a journal
    /// whose cached fit is absent or unreadable (`reused`).
    pub fit: Option<FitResult>,
    /// The objective function value, mirroring [`FitResult::ofv`] whenever
    /// [`fit`](Self::fit) is present.
    ///
    /// Kept beside the fit rather than read out of it because the fit is the
    /// part a resume can lose: `fits/<hash>.json` is a cache, and a candidate
    /// whose cached fit is missing or corrupt still has its OFV in the journal
    /// row. Reading the table's `ofv` column off `fit` alone would blank it for
    /// exactly the degraded resume the journal is designed to survive.
    /// `None` when there is no fit at all.
    pub ofv: Option<f64>,
    /// Whether the fit converged, mirroring [`FitResult::converged`] — and kept
    /// beside it for the same reason as [`ofv`](Self::ofv). `None` when there
    /// is no fit at all, which is not the same statement as `Some(false)`.
    pub converged: Option<bool>,
    /// Every gate the fit failed, and every gate that could not be evaluated.
    /// A compile or fit failure yields a verdict with one failure naming it.
    pub verdict: StrictnessVerdict,
    /// [`RunOptions::criterion`] evaluated on the fit; `NaN` when there is no
    /// fit to evaluate it on.
    pub criterion: f64,
    /// Wall-clock seconds the fit took; `0.0` for a reused or duplicate result.
    pub seconds: f64,
    /// Why there is no fit, when that is the reason — and whether a resumed run
    /// will take the candidate's word for it. See [`CandidateError`].
    pub error: Option<CandidateError>,
    /// Set when this candidate rendered to the same canonical text as an
    /// earlier one in the same run: the named id is the one that was fitted,
    /// and this result carries its criterion and verdict.
    pub duplicate_of: Option<String>,
    /// Set when the outcome came from the journal rather than from a fit in
    /// this run.
    pub reused: bool,
}

impl CandidateResult {
    /// `true` when the fit exists and passed every enabled gate — the
    /// precondition for ranking on [`criterion`](Self::criterion).
    pub fn eligible(&self) -> bool {
        self.error.is_none() && self.verdict.passed && self.criterion.is_finite()
    }
}

#[cfg(test)]
#[path = "candidate_tests.rs"]
mod tests;
