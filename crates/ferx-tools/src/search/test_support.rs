//! Fixtures shared by the `search` unit tests.
//!
//! The one thing these tests cannot fake is a [`FitResult`]: it has 130-odd
//! fields and no constructor, so there is no way to build one field by field.
//! [`fixture_fit`] is therefore one genuine `outer_maxiter = 0` **evaluation**
//! of `examples/warfarin.ferx` — ferx's `MAXEVAL=0`, the same trick
//! `tests/bootstrap_end_to_end.rs` uses — computed once per test binary and
//! cloned. Tests that need a different fit mutate the clone.

use std::sync::OnceLock;

use ferx_core::edit::ModelText;
use ferx_core::{fit, prepare_run, FitResult, Population, Subject};

use super::candidate::Candidate;

pub(crate) const MODEL: &str = "../../examples/warfarin.ferx";
pub(crate) const DATA: &str = "../../data/warfarin.csv";

/// One real evaluation of the warfarin model, memoized.
///
/// The initialisation runs on a **dedicated thread the caller joins**, and that
/// is load-bearing rather than tidy. The tests call this from inside the
/// runner's Rayon pool, and the evaluation itself enters a nested pool; a Rayon
/// worker blocked on a nested `install` keeps stealing work, so it can re-enter
/// this function *in the middle of its own* `get_or_init` and deadlock against
/// the `OnceLock` it is already initialising. Observed, not theorised: a
/// two-candidate run on a one-worker pool hung there. Blocking on a plain
/// `JoinHandle` is not a Rayon latch, so the waiting worker steals nothing and
/// the initialisation cannot be re-entered.
pub(crate) fn fixture_fit() -> FitResult {
    static FIT: OnceLock<FitResult> = OnceLock::new();
    FIT.get_or_init(|| {
        std::thread::spawn(|| {
            let prepared = prepare_run(MODEL, Some(DATA)).expect("warfarin model + data load");
            let mut options = prepared.parsed.fit_options.clone();
            options.outer_maxiter = 0;
            options.run_covariance_step = false;
            options.verbose = false;
            options.threads = Some(1);
            options.checkpoint = false;
            fit(
                &prepared.parsed.model,
                &prepared.population,
                &prepared.init_params,
                &options,
            )
            .expect("warfarin evaluation")
        })
        .join()
        .expect("the fixture thread panicked")
    })
    .clone()
}

/// The fixture with `converged` set and a chosen OFV / AIC / BIC.
pub(crate) fn converged_fit(ofv: f64) -> FitResult {
    let mut result = fixture_fit();
    result.converged = true;
    result.ofv = ofv;
    result.aic = ofv + 2.0;
    result.bic = ofv + 5.0;
    result
}

/// The smallest population the runner will accept. It is only ever read for the
/// resume manifest's data fingerprint, since these tests inject the fitter.
pub(crate) fn population(ids: &[&str]) -> Population {
    Population {
        subjects: ids
            .iter()
            .map(|id| Subject {
                id: (*id).to_string(),
                ..Default::default()
            })
            .collect(),
        covariate_names: vec![],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

pub(crate) fn model_text(body: &str) -> ModelText {
    ModelText::parse(body).expect("fixture model text")
}

/// Two spellings of one model: different bytes, identical canonical form.
pub(crate) fn same_model_two_ways() -> (ModelText, ModelText) {
    (
        model_text("[parameters]\ntheta CL = 1\ntheta V = 10\n"),
        model_text("[parameters]   # a comment\n\ntheta CL = 1\n   theta V  =  10\n"),
    )
}

pub(crate) fn candidate(id: &str, body: &str) -> Candidate {
    Candidate::new(id, model_text(body))
}

/// A scripted [`StepFitter`](crate::search::fitter::StepFitter) for the
/// variability searches (#1183): OFVs keyed by a caller-chosen key of the
/// candidate — its structure description — or by id, a `fallback` for the
/// rest, ids or keys that fail the gate, ids that produce no fit, and a call
/// log. Every fit is [`converged_fit`] at the scripted OFV, so a test that
/// ranks on `Criterion::Ofv` reads the table straight back.
pub(crate) struct ScriptedFitter {
    key: fn(&Candidate) -> String,
    ofv: std::collections::HashMap<String, f64>,
    ofv_by_id: std::collections::HashMap<String, f64>,
    pub fallback: f64,
    pub failing: Vec<String>,
    pub erroring: Vec<String>,
    pub cancel_after: Option<usize>,
    /// The step (1-based) at and after which the report comes back
    /// **cancelled with no results at all** — the shape [`Runner`] really
    /// returns when the flag is flipped *during* a fit: a candidate whose
    /// `fit()` failed while the flag was set is dropped rather than
    /// journalled, because that failure cannot be told from the flag
    /// unwinding it (`search/runner.rs`). A search that requires a result
    /// there turns a cancellation into an error, which is what this exists
    /// to catch.
    pub cancel_empty_after: Option<usize>,
    calls: std::sync::Mutex<Vec<(String, Vec<Candidate>)>>,
}

impl ScriptedFitter {
    pub(crate) fn new(key: fn(&Candidate) -> String, table: &[(&str, f64)]) -> Self {
        ScriptedFitter {
            key,
            ofv: table.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            ofv_by_id: std::collections::HashMap::new(),
            fallback: 1000.0,
            failing: Vec::new(),
            erroring: Vec::new(),
            cancel_after: None,
            cancel_empty_after: None,
            calls: std::sync::Mutex::new(Vec::new()),
        }
    }

    pub(crate) fn by_id(mut self, table: &[(&str, f64)]) -> Self {
        self.ofv_by_id = table.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        self
    }

    /// The step directories fitted, in order.
    pub(crate) fn dirs(&self) -> Vec<String> {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .map(|(d, _)| d.clone())
            .collect()
    }

    /// The candidates of one step directory.
    pub(crate) fn candidates_in(&self, dir: &str) -> Vec<Candidate> {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .find(|(d, _)| d == dir)
            .map(|(_, c)| c.clone())
            .unwrap_or_default()
    }

    /// One candidate's model, by step directory and id.
    pub(crate) fn model_of(&self, dir: &str, id: &str) -> ModelText {
        self.candidates_in(dir)
            .into_iter()
            .find(|c| c.id == id)
            .map(|c| c.model)
            .unwrap_or_else(|| panic!("no candidate {id} in {dir}"))
    }
}

impl crate::search::fitter::StepFitter for ScriptedFitter {
    fn fit_step(
        &self,
        step_dir: &str,
        candidates: &[Candidate],
    ) -> Result<crate::search::RunReport, String> {
        use crate::search::{CandidateError, CandidateResult};
        use ferx_core::StrictnessVerdict;
        let n_calls = {
            let mut calls = self.calls.lock().unwrap();
            calls.push((step_dir.to_string(), candidates.to_vec()));
            calls.len()
        };
        if self.cancel_empty_after.is_some_and(|n| n_calls >= n) {
            return Ok(crate::search::RunReport {
                results: Vec::new(),
                cancelled: true,
                fitted: 0,
                reused: 0,
                deduped: 0,
                warnings: vec![],
            });
        }
        let mut results = Vec::new();
        for c in candidates {
            let key = (self.key)(c);
            let ofv = self
                .ofv_by_id
                .get(&c.id)
                .or_else(|| self.ofv.get(&key))
                .copied()
                .unwrap_or(self.fallback);
            if self.erroring.contains(&c.id) {
                results.push(CandidateResult {
                    id: c.id.clone(),
                    hash: c.hash(),
                    parent: c.parent.clone(),
                    features: c.features.clone(),
                    fit: None,
                    ofv: None,
                    converged: None,
                    verdict: StrictnessVerdict {
                        passed: false,
                        failures: vec!["no fit: does not compile".into()],
                        skipped: vec![],
                    },
                    criterion: f64::NAN,
                    seconds: 0.0,
                    error: Some(CandidateError::model("does not compile")),
                    duplicate_of: None,
                    reused: false,
                });
                continue;
            }
            let failing = self.failing.contains(&key) || self.failing.contains(&c.id);
            results.push(CandidateResult {
                id: c.id.clone(),
                hash: c.hash(),
                parent: c.parent.clone(),
                features: c.features.clone(),
                fit: Some(converged_fit(ofv)),
                ofv: Some(ofv),
                converged: Some(!failing),
                verdict: StrictnessVerdict {
                    passed: !failing,
                    failures: if failing {
                        vec!["stalled at the initial estimates (#751)".into()]
                    } else {
                        vec![]
                    },
                    skipped: vec![],
                },
                criterion: ofv,
                seconds: 1.5,
                error: None,
                duplicate_of: None,
                reused: false,
            });
        }
        Ok(crate::search::RunReport {
            results,
            cancelled: self.cancel_after.is_some_and(|n| n_calls >= n),
            fitted: candidates.len(),
            reused: 0,
            deduped: 0,
            warnings: vec![],
        })
    }
}
