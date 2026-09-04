//! Tier-2 end-to-end checks for the candidate runner (#1178), on a real model
//! and dataset but never to convergence: every fit here is an evaluation
//! (`outer_maxiter = 0`, ferx's `MAXEVAL=0`).
//!
//! The unit tests in `src/search/runner_tests.rs` inject the fitter, so they
//! test the orchestration and deliberately never compile a candidate. This file
//! is the other half: the path from `ModelText` through `parse_full_model`, the
//! data binders and `fit()`.
//!
//! Its oracle is **degenerate**: a candidate that is the base model unedited
//! must give the OFV `fit()` gives on the same model and population, bit for
//! bit. Anything the runner's own compile path drops — a covariate binding, the
//! file's `[fit_options]`, the initial estimates — moves that number, and
//! nothing else here would notice, because there is no second copy of the
//! compile path to disagree with it.

use std::path::Path;

use ferx_core::edit::{ModelEdit, ModelText};
use ferx_core::{fit, prepare_run, CancelFlag, FitOptions, PreparedRun, Strictness};
use ferx_tools::search::{Candidate, Criterion, FeatureVector, RunOptions, Runner};

const MODEL: &str = "../../examples/warfarin.ferx";
const DATA: &str = "../../data/warfarin.csv";

fn prepared() -> PreparedRun {
    prepare_run(MODEL, Some(DATA)).expect("warfarin model + data load")
}

fn base_text() -> ModelText {
    ModelText::parse(&std::fs::read_to_string(MODEL).expect("read warfarin.ferx"))
        .expect("parse warfarin.ferx")
}

/// An evaluation, not a fit.
fn eval_options(base: &FitOptions) -> FitOptions {
    let mut o = base.clone();
    o.outer_maxiter = 0;
    o.run_covariance_step = false;
    o.verbose = false;
    o.checkpoint = false;
    o
}

fn run_options(base: &FitOptions) -> RunOptions {
    RunOptions {
        criterion: Criterion::Ofv,
        // Every gate off: an `outer_maxiter = 0` evaluation *is* an init stall,
        // and this file is not testing the gate (the unit tests are).
        strictness: Strictness::none(),
        n_starts: 1,
        resume: false,
        fit_options: Some(eval_options(base)),
    }
}

/// The candidate list used by most tests: the base model, a byte-different but
/// canonically identical copy of it, and a genuinely different model (ETA_KA
/// dropped).
fn candidates() -> Vec<Candidate> {
    let base = base_text();
    let commented = ModelText::parse(&format!(
        "# a comment the canonical form must ignore\n{}",
        base.render()
    ))
    .expect("parse the commented copy");
    let mut no_ka_iiv = base.clone();
    no_ka_iiv
        .apply(ModelEdit::DropIiv {
            param: "KA".to_string(),
        })
        .expect("drop the KA IIV");

    vec![
        Candidate::new("base", base),
        Candidate::new("base-again", commented).parent("base"),
        Candidate::new("no-ka-iiv", no_ka_iiv)
            .parent("base")
            .features(FeatureVector::new().with("IIV-KA", "off")),
    ]
}

#[test]
fn an_unedited_candidate_reproduces_a_direct_fit_exactly() {
    let p = prepared();
    let options = run_options(&p.parsed.fit_options);
    let direct = fit(
        &p.parsed.model,
        &p.population,
        &p.init_params,
        &eval_options(&p.parsed.fit_options),
    )
    .expect("direct evaluation");

    let report = Runner::new()
        .threads(1)
        .run(&candidates()[..1], &p.population, &options)
        .expect("run");

    let base = report.results[0].fit.as_ref().expect("the base fit");
    // Bit-for-bit. "Close" is exactly what a dropped covariate binding or a
    // silently substituted set of initial estimates looks like at this scale.
    assert_eq!(
        base.ofv.to_bits(),
        direct.ofv.to_bits(),
        "runner OFV {} vs direct {}",
        base.ofv,
        direct.ofv
    );
    assert_eq!(base.n_obs, direct.n_obs);
    assert_eq!(base.n_subjects, direct.n_subjects);
    assert_eq!(report.results[0].criterion, direct.ofv);
}

#[test]
fn a_real_search_step_dedups_fits_and_ranks_what_is_left() {
    let p = prepared();
    let dir = tempfile::tempdir().expect("tempdir");
    let options = run_options(&p.parsed.fit_options);
    let report = Runner::new()
        .threads(2)
        .cache_dir(dir.path())
        .run(&candidates(), &p.population, &options)
        .expect("run");

    assert_eq!((report.fitted, report.deduped, report.reused), (2, 1, 0));
    assert_eq!(report.results.len(), 3);

    // The commented copy is the same model: same hash, no fit of its own, and
    // the base model's criterion.
    assert_eq!(report.results[1].hash, report.results[0].hash);
    assert_eq!(report.results[1].duplicate_of.as_deref(), Some("base"));
    assert_eq!(report.results[1].criterion, report.results[0].criterion);

    // Dropping an η is a different model, so it is a different hash and a
    // different objective — the edit reached the engine.
    assert_ne!(report.results[2].hash, report.results[0].hash);
    let no_ka = report.results[2].fit.as_ref().expect("the edited fit");
    assert_ne!(
        no_ka.ofv.to_bits(),
        report.results[0].fit.as_ref().unwrap().ofv.to_bits(),
        "dropping the KA IIV did not change the objective"
    );
    assert_eq!(no_ka.eta_names.len(), 2, "the η was not dropped");
    assert!(report.best().is_some());

    // Both artefacts are on disk, and the journal holds one row per *fit*, not
    // per candidate.
    let table = std::fs::read_to_string(dir.path().join("candidates.csv")).expect("table");
    assert_eq!(table.lines().count(), 4);
    assert!(table.contains("IIV-KA=off"));
    let journal =
        std::fs::read_to_string(dir.path().join("search_journal.jsonl")).expect("journal");
    assert_eq!(journal.lines().count(), 2);
    assert!(dir.path().join("search_run.json").exists());
}

#[test]
fn a_resumed_run_refits_nothing_and_reports_the_same_numbers() {
    let p = prepared();
    let dir = tempfile::tempdir().expect("tempdir");
    let options = run_options(&p.parsed.fit_options);
    let first = Runner::new()
        .threads(2)
        .cache_dir(dir.path())
        .run(&candidates(), &p.population, &options)
        .expect("first run");

    let resumed = Runner::new()
        .threads(2)
        .cache_dir(dir.path())
        .run(
            &candidates(),
            &p.population,
            &RunOptions {
                resume: true,
                ..run_options(&p.parsed.fit_options)
            },
        )
        .expect("resumed run");

    assert_eq!((resumed.fitted, resumed.reused), (0, 2));
    assert_eq!(resumed.results.len(), 3);
    for (before, after) in first.results.iter().zip(&resumed.results) {
        assert_eq!(before.id, after.id);
        assert_eq!(before.criterion, after.criterion);
        assert_eq!(
            before.fit.as_ref().map(|f| f.ofv.to_bits()),
            after.fit.as_ref().map(|f| f.ofv.to_bits()),
            "the cached fit of `{}` came back different",
            before.id
        );
    }
    assert!(resumed.results[0].reused);
}

#[test]
fn a_candidate_that_does_not_compile_is_reported_not_fatal() {
    let p = prepared();
    // A model whose `[individual_parameters]` reference a parameter that does
    // not exist: structurally a valid `.ferx` file, rejected by the compiler.
    let broken = ModelText::parse("[parameters]\n  theta TVCL(0.2, 0.001, 10.0)\n\n[individual_parameters]\n  CL = NOSUCHTHETA\n")
        .expect("the text still parses structurally");

    let mut list = candidates();
    list.push(Candidate::new("broken", broken));
    let report = Runner::new()
        .threads(1)
        .run(&list, &p.population, &run_options(&p.parsed.fit_options))
        .expect("a failing candidate must not fail the run");

    let failed = report
        .results
        .iter()
        .find(|r| r.id == "broken")
        .expect("the broken candidate is missing from the report");
    assert!(failed.fit.is_none());
    assert!(failed.error.is_some(), "no reason was recorded");
    assert!(!failed.eligible());
    // The rest of the run is unaffected.
    assert_eq!(report.fitted, 3);
    assert!(report.best().is_some());
}

#[test]
fn cancelling_a_run_returns_what_finished() {
    let p = prepared();
    let flag = CancelFlag::new();
    flag.cancel();
    let report = Runner::new()
        .threads(1)
        .cancel(flag)
        .run(
            &candidates(),
            &p.population,
            &run_options(&p.parsed.fit_options),
        )
        .expect("a cancelled run still reports");
    assert!(report.cancelled);
    assert_eq!(report.fitted, 0);
}

#[test]
fn the_candidates_own_fit_options_are_used_when_none_are_given() {
    // `fit_options: None` means "each candidate carries its own", which is what
    // a search over models edited from one base file wants. The base file says
    // `method = foce`; a candidate that sets `method = focei` must be fitted
    // that way rather than under the runner's or its parent's choice.
    let p = prepared();
    let mut focei = base_text();
    focei
        .apply(ModelEdit::SetFitOption {
            key: "method".to_string(),
            value: "focei".to_string(),
        })
        .expect("set the method");
    focei
        .apply(ModelEdit::SetFitOption {
            key: "maxiter".to_string(),
            value: "0".to_string(),
        })
        .expect("cap the iterations");
    let mut foce = base_text();
    foce.apply(ModelEdit::SetFitOption {
        key: "maxiter".to_string(),
        value: "0".to_string(),
    })
    .expect("cap the iterations");

    let report = Runner::new()
        .threads(1)
        .run(
            &[Candidate::new("foce", foce), Candidate::new("focei", focei)],
            &p.population,
            &RunOptions {
                criterion: Criterion::Ofv,
                strictness: Strictness::none(),
                n_starts: 1,
                resume: false,
                fit_options: None,
            },
        )
        .expect("run");

    let method_of = |id: &str| {
        report
            .results
            .iter()
            .find(|r| r.id == id)
            .and_then(|r| r.fit.as_ref())
            .map(|f| format!("{:?}", f.method))
            .unwrap_or_default()
    };
    assert_ne!(
        method_of("foce"),
        method_of("focei"),
        "both candidates were fitted with the same method, so the file's \
         `[fit_options]` were not read per candidate"
    );
}

#[test]
fn an_edit_that_changes_the_random_effects_changes_the_bic_it_is_ranked_on() {
    // The runner's criterion is computed from the candidate's own fit, so an
    // edit that removes a variance parameter must move the penalty — the
    // property `iivsearch` will rank on.
    let p = prepared();
    let list = candidates();
    let options = RunOptions {
        criterion: Criterion::Bic(ferx_core::BicType::Iiv),
        ..run_options(&p.parsed.fit_options)
    };
    let report = Runner::new()
        .threads(1)
        .run(&list, &p.population, &options)
        .expect("run");

    let base = &report.results[0];
    let no_ka = &report.results[2];
    assert!(base.criterion.is_finite() && no_ka.criterion.is_finite());
    assert_ne!(base.criterion, no_ka.criterion);
    assert_eq!(
        report.best().map(|r| r.id.as_str()),
        Some(if base.criterion <= no_ka.criterion {
            "base"
        } else {
            "no-ka-iiv"
        })
    );
}

#[test]
fn the_fixtures_exist() {
    // A wrong relative path would turn every test above into a `prepare_run`
    // failure that reads like an engine problem.
    assert!(Path::new(MODEL).exists(), "missing {MODEL}");
    assert!(Path::new(DATA).exists(), "missing {DATA}");
}
