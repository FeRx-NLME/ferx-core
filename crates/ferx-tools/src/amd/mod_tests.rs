use std::path::Path;
use std::sync::Mutex;

use super::*;
use crate::search::test_support::{converged_fit, MODEL};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// The AMD space every fixture config carries: one statement per step.
const SPACE: &str = "ABSORPTION([INST,FO]);PERIPHERALS(0..1);IIV?([CL,V],EXP);\
                     IOV?(CL,EXP);ALLOMETRY(WT,70);COVARIATE?(@IIV,@CONTINUOUS,pow)";

/// The warfarin model, so [`seed_from`](crate::search::seed::seed_from) has
/// real parameters to write into — a fixture model whose names the fit does
/// not carry would make the seeding assertions vacuous.
fn warfarin() -> ModelText {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(MODEL);
    ModelText::parse(&std::fs::read_to_string(path).expect("warfarin model")).expect("model text")
}

fn config_with(space: &str, extra: &str) -> SearchConfig {
    let text = format!(
        "base = \"warfarin.ferx\"\n\n[space]\nmfl = \"\"\"{space}\"\"\"\n\n[run]\nretries = \
         2\n{extra}"
    );
    SearchConfig::from_str(&text, Path::new(".")).expect("fixture config")
}

fn config() -> SearchConfig {
    config_with(SPACE, "")
}

/// A fit whose θ differs from the model file's, so a model that was seeded is
/// distinguishable from one that was not.
fn moved_fit(ofv: f64) -> FitResult {
    let mut fit = converged_fit(ofv);
    fit.theta[0] *= 1.5;
    fit
}

/// What one call on the scripted runner recorded.
#[derive(Debug, Clone, PartialEq)]
struct Call {
    kind: &'static str,
    dir: String,
    /// The rendered model the call was handed.
    model: String,
    /// The rendered subspace the call was handed.
    space: String,
    starts: usize,
    /// Whether the call was handed the init-stall gate.
    stall_gate: bool,
}

/// A [`StepRunner`] that fits nothing: it records what it was handed and hands
/// back a model derived from it.
///
/// Every decision this module makes is observable only through the calls it
/// makes and the models it passes between them, which is what this exists to
/// capture — see [`StepRunner`]'s own docs.
#[derive(Default)]
struct Scripted {
    calls: Mutex<Vec<Call>>,
    /// The OFV of the pipeline's start fit.
    start_ofv: f64,
    /// The step (by pipeline index) whose output comes back cancelled.
    cancel_at: Option<usize>,
    /// The start fit comes back with no fit at all.
    no_start_fit: bool,
    /// The step (by pipeline index) whose tool returns an error.
    fail_at: Option<usize>,
    /// The retries pass comes back with a fit the strictness gate rejected.
    retries_fail_gate: bool,
    /// What the retries pass scores relative to the model it was handed.
    retry_delta: f64,
}

impl Scripted {
    fn new() -> Self {
        Scripted {
            start_ofv: 1000.0,
            ..Default::default()
        }
    }

    fn calls(&self) -> Vec<Call> {
        self.calls.lock().unwrap().clone()
    }

    fn dirs(&self) -> Vec<String> {
        self.calls().into_iter().map(|c| c.dir).collect()
    }

    /// The OFV a step's own fit comes back with: one better than the last.
    fn step_ofv(&self, index: usize) -> f64 {
        self.start_ofv - 10.0 * index as f64
    }
}

/// The last OFV any call before this one reported, so the retries pass can be
/// scored relative to the model it was handed.
fn last_ofv(scripted: &Scripted) -> f64 {
    let steps = scripted.calls().iter().filter(|c| c.kind == "step").count();
    scripted.step_ofv(steps)
}

impl StepRunner for Scripted {
    fn run_step(
        &self,
        index: usize,
        step: Step,
        dir_name: &str,
        config: &SearchConfig,
        model: &ModelText,
    ) -> Result<StepOutput, String> {
        self.calls.lock().unwrap().push(Call {
            kind: "step",
            dir: dir_name.to_string(),
            model: model.render(),
            space: config.mfl.render(),
            starts: config.run.retries + 1,
            stall_gate: config.strictness.strictness().reject_init_stall,
        });
        if self.fail_at == Some(index) {
            return Err(format!(
                "{dir_name}: `WT` is not a covariate of the dataset"
            ));
        }
        let ofv = self.step_ofv(index);
        let derived = ModelText::parse(&format!("{}\n# selected by {dir_name}\n", model.render()))?;
        let rows = vec![
            CandidateRow {
                step: index,
                tool: step.tool().to_string(),
                id: "base".into(),
                parent: None,
                description: "the model as handed over".into(),
                criterion: "ofv",
                value: Some(ofv + 5.0),
                d_value: None,
                ofv: Some(ofv + 5.0),
                d_ofv: None,
                rank: Some(2),
                converged: Some(true),
                passed: true,
                failures: vec![],
                error: None,
                note: None,
                seconds: 1.0,
                selected: false,
            },
            CandidateRow {
                step: index,
                tool: step.tool().to_string(),
                id: "cand".into(),
                parent: Some("base".into()),
                description: format!("{} candidate", step.label()),
                criterion: "ofv",
                value: Some(ofv),
                d_value: Some(-5.0),
                ofv: Some(ofv),
                d_ofv: None,
                rank: Some(1),
                converged: Some(true),
                passed: true,
                failures: vec![],
                error: None,
                note: None,
                seconds: 2.0,
                selected: true,
            },
        ];
        Ok(StepOutput {
            model: derived,
            fit: Some(moved_fit(ofv)),
            criterion: Criterion::Ofv,
            value: Some(ofv),
            passed: true,
            rows,
            selected: vec![format!("{} choice", step.label())],
            notes: vec![],
            cancelled: self.cancel_at == Some(index),
        })
    }

    fn fit_one(
        &self,
        index: usize,
        tool: &str,
        dir_name: &str,
        config: &SearchConfig,
        model: &ModelText,
        n_starts: usize,
    ) -> Result<StepOutput, String> {
        self.calls.lock().unwrap().push(Call {
            kind: if tool == "start" { "start" } else { "retries" },
            dir: dir_name.to_string(),
            model: model.render(),
            space: config.mfl.render(),
            starts: n_starts,
            stall_gate: config.strictness.strictness().reject_init_stall,
        });
        let (ofv, fit) = if tool == "start" {
            if self.no_start_fit {
                (None, None)
            } else {
                (Some(self.start_ofv), Some(moved_fit(self.start_ofv)))
            }
        } else {
            let ofv = last_ofv(self) + self.retry_delta;
            (Some(ofv), Some(moved_fit(ofv)))
        };
        let converged = fit_converged(&fit);
        let has_fit = fit.is_some();
        Ok(StepOutput {
            model: model.clone(),
            fit,
            criterion: Criterion::Ofv,
            value: ofv,
            passed: has_fit && !self.retries_fail_gate,
            rows: vec![CandidateRow {
                step: index,
                tool: tool.to_string(),
                id: tool.to_string(),
                parent: None,
                description: format!("{n_starts} starts"),
                criterion: "ofv",
                value: ofv,
                d_value: None,
                ofv,
                d_ofv: None,
                rank: None,
                converged,
                passed: has_fit,
                failures: if has_fit {
                    vec![]
                } else {
                    vec!["no fit".into()]
                },
                error: None,
                note: None,
                seconds: 0.5,
                selected: true,
            }],
            selected: vec![],
            notes: vec![],
            cancelled: false,
        })
    }
}

fn fit_converged(fit: &Option<FitResult>) -> Option<bool> {
    fit.as_ref().map(|f| f.converged)
}

fn run<R: StepRunner>(scripted: &R, options: &AmdOptions, config: &SearchConfig) -> AmdResult {
    let ctx = Context {
        iov_column: Some("OCC".into()),
    };
    let plan = plan(options, &config.mfl, &ctx).expect("plan");
    drive(scripted, config, options, &plan, &warfarin(), None).expect("drive")
}

// ---------------------------------------------------------------------------
// The plan
// ---------------------------------------------------------------------------

/// Every strategy's ordering, verbatim from Pharmpy's `get_subtool_order`
/// (`pharmpy/tools/amd/run.py`). The pipeline's whole contract is that it runs
/// these components in this order, so the orders are pinned as data rather
/// than exercised through a run.
#[test]
fn strategy_orders_match_pharmpy() {
    use Step::*;
    assert_eq!(
        Strategy::Default.order(),
        vec![Structural, Iivsearch, Residual, Iovsearch, Allometry, Covariates]
    );
    assert_eq!(
        Strategy::Reevaluation.order(),
        vec![
            Structural, Iivsearch, Residual, Iovsearch, Allometry, Covariates, Iivsearch, Residual
        ]
    );
    assert_eq!(Strategy::Sir.order(), vec![Structural, Iivsearch, Residual]);
    assert_eq!(Strategy::Sri.order(), vec![Structural, Residual, Iivsearch]);
    assert_eq!(Strategy::Rsi.order(), vec![Residual, Structural, Iivsearch]);
}

/// The plan numbers the steps in order and marks the second occurrence of a
/// step as a rerun, so its directory cannot collide with the first's.
#[test]
fn the_plan_numbers_its_steps_and_marks_the_rerun() {
    let options = AmdOptions {
        strategy: Strategy::Reevaluation,
        ..Default::default()
    };
    let ctx = Context {
        iov_column: Some("OCC".into()),
    };
    let plan = plan(&options, &config().mfl, &ctx).unwrap();
    assert_eq!(
        plan.iter().map(|p| p.dir.clone()).collect::<Vec<_>>(),
        vec![
            "01-modelsearch",
            "02-iivsearch",
            "03-ruvsearch",
            "04-iovsearch",
            "05-allometry",
            "06-covsearch",
            "07-rerun-iivsearch",
            "08-rerun-ruvsearch",
        ]
    );
    assert!(!plan[1].rerun && plan[6].rerun);
    assert!(plan.iter().all(|p| p.skipped.is_none()));
}

/// A step the space says nothing about is skipped with its reason — and the
/// reason names the statements that would have run it, so the message is
/// actionable.
#[test]
fn a_step_the_space_is_silent_about_is_skipped_with_its_reason() {
    let config = config_with("ABSORPTION([INST,FO])", "");
    let ctx = Context {
        iov_column: Some("OCC".into()),
    };
    let plan = plan(&AmdOptions::default(), &config.mfl, &ctx).unwrap();
    let by_step = |step: Step| {
        plan.iter()
            .find(|p| p.step == step)
            .unwrap()
            .skipped
            .clone()
    };
    assert_eq!(by_step(Step::Structural), None);
    // The residual search has no space at all, so silence is not a skip.
    assert_eq!(by_step(Step::Residual), None);
    let iiv = by_step(Step::Iivsearch).expect("iivsearch skipped");
    assert!(iiv.contains("IIV / COVARIANCE(IIV"), "{iiv}");
    assert!(by_step(Step::Allometry).unwrap().contains("ALLOMETRY"));
    assert!(by_step(Step::Covariates).unwrap().contains("COVARIATE"));
}

/// `iovsearch` reads the occasions from the starting model's `iov_column`; a
/// model without one has a population that never read them, so the step is
/// skipped rather than run on nothing. With a column it runs even though the
/// space is silent, since it defaults to every parameter with a free η.
#[test]
fn iovsearch_needs_an_occasion_column_but_not_a_space() {
    let config = config_with("ABSORPTION([INST,FO])", "");
    let without = plan(&AmdOptions::default(), &config.mfl, &Context::default()).unwrap();
    let reason = without
        .iter()
        .find(|p| p.step == Step::Iovsearch)
        .unwrap()
        .skipped
        .clone()
        .expect("skipped without an occasion column");
    assert!(reason.contains("iov_column"), "{reason}");

    let with = plan(
        &AmdOptions::default(),
        &config.mfl,
        &Context {
            iov_column: Some("OCC".into()),
        },
    )
    .unwrap();
    assert_eq!(
        with.iter()
            .find(|p| p.step == Step::Iovsearch)
            .unwrap()
            .skipped,
        None
    );
}

/// `[amd] skip` leaves a step out and says so in the plan.
#[test]
fn skip_leaves_a_step_out_and_records_it() {
    let options = AmdOptions {
        skip: vec![Step::Covariates],
        ..Default::default()
    };
    let ctx = Context {
        iov_column: Some("OCC".into()),
    };
    let plan = plan(&options, &config().mfl, &ctx).unwrap();
    let reason = plan
        .iter()
        .find(|p| p.step == Step::Covariates)
        .unwrap()
        .skipped
        .clone()
        .unwrap();
    assert!(reason.contains("[amd] skip"), "{reason}");
}

/// A `skip` naming a step the strategy does not run is a typo, not a no-op —
/// and skipping everything leaves nothing to run.
#[test]
fn skip_is_checked_against_the_strategy() {
    let error = AmdOptions {
        strategy: Strategy::Sir,
        skip: vec![Step::Covariates],
        ..Default::default()
    }
    .validate()
    .unwrap_err();
    assert!(
        error.contains("covariates") && error.contains("SIR"),
        "{error}"
    );

    let error = AmdOptions {
        strategy: Strategy::Sir,
        skip: vec![Step::Structural, Step::Iivsearch, Step::Residual],
        ..Default::default()
    }
    .validate()
    .unwrap_err();
    assert!(error.contains("no steps to run"), "{error}");
}

/// `[amd]` is read from the file, and Pharmpy's own upper-case strategy
/// spellings are accepted so a call copied from a Pharmpy script means the
/// same thing.
#[test]
fn the_amd_section_is_read_from_the_file() {
    let config = config_with(
        SPACE,
        "\n[amd]\nstrategy = \"SIR\"\nretries = \"final\"\nskip = [\"residual\"]\n",
    );
    let options = AmdOptions::from_config(&config).unwrap();
    assert_eq!(options.strategy, Strategy::Sir);
    assert_eq!(options.retries, Retries::Final);
    assert_eq!(options.skip, vec![Step::Residual]);

    // The defaults, when the file has no [amd] at all.
    let bare = AmdOptions::from_config(&config_with(SPACE, "")).unwrap();
    assert_eq!(bare, AmdOptions::default());
    assert_eq!(bare.strategy, Strategy::Default);
    assert_eq!(bare.retries, Retries::AllFinal);

    // A misspelt key is refused rather than silently ignored.
    let bad = config_with(SPACE, "\n[amd]\nstrategie = \"SIR\"\n");
    assert!(AmdOptions::from_config(&bad).is_err());
}

// ---------------------------------------------------------------------------
// The pipeline
// ---------------------------------------------------------------------------

/// The steps run in the strategy's order, each in its own directory, with the
/// start fit first.
#[test]
fn the_pipeline_runs_its_steps_in_order() {
    let scripted = Scripted::new();
    let result = run(
        &scripted,
        &AmdOptions {
            retries: Retries::Skip,
            ..Default::default()
        },
        &config(),
    );
    assert_eq!(
        scripted.dirs(),
        vec![
            "00-start",
            "01-modelsearch",
            "02-iivsearch",
            "03-ruvsearch",
            "04-iovsearch",
            "05-allometry",
            "06-covsearch",
        ]
    );
    assert_eq!(result.steps.len(), 6);
    assert!(result.ran().count() == 6);
    assert!(!result.cancelled);
    // Every step's outcome carries the comparison the report is made of.
    let covariates = result.steps.last().unwrap();
    assert_eq!(covariates.ofv_before, Some(scripted.step_ofv(5)));
    assert_eq!(covariates.ofv_after, Some(scripted.step_ofv(6)));
    assert_eq!(covariates.selected, vec!["covariates choice"]);
    assert_eq!(result.d_ofv(), Some(scripted.step_ofv(6) - 1000.0));
}

/// Each step is handed the model the previous step selected, **seeded from
/// that step's estimates**.
///
/// The expected model is built here by the same two operations the pipeline
/// performs, so removing either the hand-over or the seeding reddens it: the
/// scripted fit moves θ₀ off the file's value, which a model that was not
/// seeded still carries.
#[test]
fn each_step_starts_from_the_previous_step_s_selection_seeded() {
    let scripted = Scripted::new();
    run(
        &scripted,
        &AmdOptions {
            retries: Retries::Skip,
            ..Default::default()
        },
        &config(),
    );
    let calls = scripted.calls();

    // Step 1 gets the input, seeded from the start fit.
    let mut expected = warfarin();
    crate::search::seed::seed_from(&mut expected, &moved_fit(scripted.start_ofv)).unwrap();
    assert_eq!(calls[1].model, expected.render());
    assert_ne!(
        calls[1].model,
        warfarin().render(),
        "the start fit's estimates were not seeded"
    );

    // Step 2 gets step 1's selection, seeded from step 1's fit.
    let mut expected = ModelText::parse(&format!(
        "{}\n# selected by 01-modelsearch\n",
        calls[1].model
    ))
    .unwrap();
    crate::search::seed::seed_from(&mut expected, &moved_fit(scripted.step_ofv(1))).unwrap();
    assert_eq!(calls[2].model, expected.render());
    assert!(
        calls[2].model.contains("# selected by 01-modelsearch"),
        "step 2 did not start from step 1's selection"
    );
}

/// Each step is handed only the statements its tool accepts — the whole reason
/// [`space`] exists, since every tool refuses a foreign statement by name.
#[test]
fn each_step_is_handed_only_its_own_subspace() {
    let scripted = Scripted::new();
    run(
        &scripted,
        &AmdOptions {
            retries: Retries::Skip,
            ..Default::default()
        },
        &config(),
    );
    let space_of = |dir: &str| {
        scripted
            .calls()
            .into_iter()
            .find(|c| c.dir == dir)
            .unwrap()
            .space
    };
    assert_eq!(
        space_of("01-modelsearch"),
        "ABSORPTION([INST,FO]);PERIPHERALS(0..1)"
    );
    assert_eq!(space_of("02-iivsearch"), "IIV?([CL,V],EXP)");
    assert_eq!(space_of("04-iovsearch"), "IOV?(CL,EXP)");
    assert_eq!(space_of("05-allometry"), "ALLOMETRY(WT,70)");
    assert_eq!(space_of("06-covsearch"), "COVARIATE?(@IIV,@CONTINUOUS,pow)");
    // The residual search's candidates are the error forms, not MFL features.
    assert_eq!(space_of("03-ruvsearch"), "");
}

/// `all_final` runs the perturbed-restart pass after every step; `final` only
/// after the last one; `skip` not at all. The pass is fitted with
/// `[run] retries + 1` starts, which is the knob it is implemented as.
#[test]
fn the_retries_policy_decides_which_selected_models_get_the_pass() {
    let retries_dirs = |policy: Retries| {
        let scripted = Scripted::new();
        run(
            &scripted,
            &AmdOptions {
                retries: policy,
                ..Default::default()
            },
            &config(),
        );
        let calls = scripted.calls();
        assert!(
            calls
                .iter()
                .filter(|c| c.kind == "retries")
                .all(|c| c.starts == 3),
            "the pass must take [run] retries + 1 starts"
        );
        calls
            .into_iter()
            .filter(|c| c.kind == "retries")
            .map(|c| c.dir)
            .collect::<Vec<_>>()
    };
    assert_eq!(
        retries_dirs(Retries::AllFinal),
        vec![
            "01-modelsearch-retries",
            "02-iivsearch-retries",
            "03-ruvsearch-retries",
            "04-iovsearch-retries",
            "05-allometry-retries",
            "06-covsearch-retries",
        ]
    );
    assert_eq!(retries_dirs(Retries::Final), vec!["retries"]);
    assert!(retries_dirs(Retries::Skip).is_empty());
}

/// The pass replaces the selected model's fit only when it lands lower, and
/// says so in the notes when it does not — a pass that silently kept a worse
/// optimum would be indistinguishable from one that never ran.
#[test]
fn the_retries_pass_keeps_the_better_of_the_two_fits() {
    let mut scripted = Scripted::new();
    scripted.retry_delta = -4.0;
    let improved = run(
        &scripted,
        &AmdOptions {
            retries: Retries::Final,
            ..Default::default()
        },
        &config(),
    );
    assert_eq!(
        improved.final_fit.as_ref().map(|f| f.ofv),
        Some(scripted.step_ofv(6) - 4.0)
    );
    assert!(!improved.notes.iter().any(|n| n.contains("did not improve")));

    let mut scripted = Scripted::new();
    scripted.retry_delta = 7.0;
    let kept = run(
        &scripted,
        &AmdOptions {
            retries: Retries::Final,
            ..Default::default()
        },
        &config(),
    );
    assert_eq!(
        kept.final_fit.as_ref().map(|f| f.ofv),
        Some(scripted.step_ofv(6))
    );
    let note = kept
        .notes
        .iter()
        .find(|n| n.contains("did not improve"))
        .expect("the kept model is recorded");
    assert!(note.contains("retries"), "{note}");
}

/// `[run] retries = 0` asks for a single start, which would refit the selected
/// model at its own estimates. The pass is not run, and the report says why
/// rather than showing a step that did nothing.
#[test]
fn a_single_start_makes_the_retries_pass_pointless_and_it_says_so() {
    let scripted = Scripted::new();
    let config = config_with(SPACE, "");
    let mut config = config;
    config.run.retries = 0;
    let result = run(
        &scripted,
        &AmdOptions {
            retries: Retries::Final,
            ..Default::default()
        },
        &config,
    );
    assert!(scripted.calls().iter().all(|c| c.kind != "retries"));
    assert!(
        result.notes.iter().any(|n| n.contains("retries = 0")),
        "{:?}",
        result.notes
    );
}

/// A skipped step still gets an outcome, with its reason and no candidates —
/// so a reader can tell a step that ran and found nothing from one that never
/// ran.
#[test]
fn a_skipped_step_is_reported_rather_than_dropped() {
    let scripted = Scripted::new();
    let config = config_with("ABSORPTION([INST,FO])", "");
    let ctx = Context::default();
    let plan = plan(&AmdOptions::default(), &config.mfl, &ctx).unwrap();
    let options = AmdOptions {
        retries: Retries::Skip,
        ..Default::default()
    };
    let result = drive(&scripted, &config, &options, &plan, &warfarin(), None).unwrap();
    assert_eq!(result.steps.len(), 6);
    assert_eq!(result.ran().count(), 2, "structural and residual only");
    let skipped: Vec<&StepOutcome> = result.steps.iter().filter(|s| !s.ran()).collect();
    assert_eq!(skipped.len(), 4);
    for s in skipped {
        assert!(s.skipped.is_some());
        assert_eq!(s.candidates, 0);
        assert!(s.selected.is_empty());
    }
    assert!(result
        .rows
        .iter()
        .all(|r| r.tool != "iivsearch" && r.tool != "covsearch"));
}

/// A cancelled step stops the pipeline where it is: the steps after it never
/// run, the result says it was cancelled, and what did run is still reported.
#[test]
fn a_cancelled_step_stops_the_pipeline() {
    let mut scripted = Scripted::new();
    scripted.cancel_at = Some(2);
    let result = run(&scripted, &AmdOptions::default(), &config());
    assert!(result.cancelled);
    assert_eq!(result.steps.len(), 2);
    assert_eq!(
        scripted.dirs(),
        vec![
            "00-start",
            "01-modelsearch",
            "01-modelsearch-retries",
            "02-iivsearch"
        ]
    );
    // No retries pass on the step that was cancelled.
    assert!(!scripted
        .dirs()
        .contains(&"02-iivsearch-retries".to_string()));
}

/// A start model that produces no fit is an error, not a pipeline that runs
/// with no baseline: every Δ in the report and every seeded start depends on
/// it.
#[test]
fn a_start_model_that_cannot_be_fitted_is_an_error() {
    let mut scripted = Scripted::new();
    scripted.no_start_fit = true;
    let config = config();
    let ctx = Context {
        iov_column: Some("OCC".into()),
    };
    let plan = plan(&AmdOptions::default(), &config.mfl, &ctx).unwrap();
    let error = drive(
        &scripted,
        &config,
        &AmdOptions::default(),
        &plan,
        &warfarin(),
        None,
    )
    .unwrap_err();
    assert!(error.contains("no baseline"), "{error}");
    assert_eq!(scripted.dirs(), vec!["00-start"], "no step ran");
}

/// The progress callback sees the plan, the start fit, every step and every
/// skip — the events a CLI prints.
#[test]
fn progress_reports_the_plan_and_every_step() {
    let scripted = Scripted::new();
    let seen: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let progress = |event: AmdEvent| {
        seen.lock().unwrap().push(match event {
            AmdEvent::Planned { steps } => format!("planned {}", steps.len()),
            AmdEvent::StartStarted => "start".into(),
            AmdEvent::StartFinished { .. } => "start done".into(),
            AmdEvent::StepStarted { step, .. } => format!("step {}", step.label()),
            AmdEvent::StepSkipped { step, .. } => format!("skip {}", step.label()),
            AmdEvent::StepFailed { step, .. } => format!("failed {}", step.label()),
            AmdEvent::StepFinished { step, .. } => format!("done {}", step.label()),
            AmdEvent::RetriesStarted { .. } => "retries".into(),
            AmdEvent::RetriesFinished { .. } => "retries done".into(),
        });
    };
    let config = config_with("ABSORPTION([INST,FO])", "");
    let plan = plan(&AmdOptions::default(), &config.mfl, &Context::default()).unwrap();
    let options = AmdOptions {
        retries: Retries::Final,
        ..Default::default()
    };
    drive(
        &scripted,
        &config,
        &options,
        &plan,
        &warfarin(),
        Some(&progress),
    )
    .unwrap();
    let seen = seen.lock().unwrap().clone();
    assert_eq!(seen[0], "planned 6");
    assert_eq!(&seen[1..3], ["start", "start done"]);
    assert!(seen.contains(&"step structural".to_string()));
    assert!(seen.contains(&"skip covariates".to_string()));
    assert_eq!(&seen[seen.len() - 2..], ["retries", "retries done"]);
}

// ---------------------------------------------------------------------------
// Rows
// ---------------------------------------------------------------------------

/// `ΔOFV` is derived from the candidate's parent **within its own step**, so a
/// row whose parent is in another step (or absent) is left blank rather than
/// compared against an unrelated model.
#[test]
fn d_ofv_is_filled_from_the_parent_in_the_same_step() {
    let row = |step: usize, id: &str, parent: Option<&str>, ofv: Option<f64>| CandidateRow {
        step,
        tool: "modelsearch".into(),
        id: id.into(),
        parent: parent.map(str::to_string),
        description: String::new(),
        criterion: "ofv",
        value: ofv,
        d_value: None,
        ofv,
        d_ofv: None,
        rank: None,
        converged: None,
        passed: true,
        failures: vec![],
        error: None,
        note: None,
        seconds: 0.0,
        selected: false,
    };
    let mut rows = vec![
        row(1, "base", None, Some(100.0)),
        row(1, "child", Some("base"), Some(94.0)),
        row(1, "orphan", Some("elsewhere"), Some(90.0)),
        row(1, "failed", Some("base"), None),
    ];
    fill_d_ofv(&mut rows);
    assert_eq!(rows[0].d_ofv, None, "a root has nothing to compare against");
    assert_eq!(rows[1].d_ofv, Some(-6.0));
    assert_eq!(rows[2].d_ofv, None, "an unknown parent is not a comparison");
    assert_eq!(rows[3].d_ofv, None, "a candidate with no fit has no ΔOFV");

    // A row that already carries its own ΔOFV — the tools that report against
    // a parent OFV rather than a parent row — is left alone.
    let mut own = vec![row(1, "x", Some("base"), Some(94.0))];
    own[0].d_ofv = Some(-1.0);
    fill_d_ofv(&mut own);
    assert_eq!(own[0].d_ofv, Some(-1.0));
}

/// `[rank]` is narrowed the way the space is: the criterion goes to the steps
/// that rank on it, and the two likelihood-ratio steps are left at their own
/// defaults — which they insist on, refusing any other `type` and any
/// `cutoff` outright.
///
/// Without this, a pipeline configured with Pharmpy's own default,
/// `[rank] type = "bic"`, fails its residual and covariate steps by
/// construction — measured on `examples/amd_start.ferxsearch`, where both came
/// back `failed` with exactly those two messages.
#[test]
fn rank_is_narrowed_to_the_steps_that_rank() {
    let config = config_with(SPACE, "\n[rank]\ntype = \"bic\"\ncutoff = 3.84\n");
    for step in [Step::Structural, Step::Iivsearch, Step::Iovsearch] {
        let sub = step_config(&config, step).unwrap();
        assert_eq!(sub.rank, config.rank, "{} lost the criterion", step.label());
    }
    for step in [Step::Residual, Step::Covariates] {
        let sub = step_config(&config, step).unwrap();
        assert_eq!(sub.rank.kind, None, "{} kept a rank type", step.label());
        assert_eq!(sub.rank.cutoff, None, "{} kept a cutoff", step.label());
        // The step's own reader accepts what it is handed, which is the
        // property this exists for — and would reject the file's `bic`.
        match step {
            Step::Residual => {
                crate::ruvsearch::RuvsearchOptions::from_config(&sub).unwrap();
            }
            _ => {
                crate::covsearch::CovsearchOptions::from_config(&sub).unwrap();
            }
        }
        assert!(
            crate::ruvsearch::RuvsearchOptions::from_config(&config)
                .unwrap_err()
                .contains("[rank] type"),
            "the file's own [rank] is what the narrowing exists to keep away"
        );
    }
}

/// The init-stall gate is off inside every step, and the reason is in every
/// run's notes.
///
/// Without it the pipeline fails by construction: every step refits the model
/// it was handed, that model starts at its own optimum, and #751's gate calls
/// a fit that does not move a stall. Measured on
/// `examples/amd_start.ferxsearch` before the fix — the residual and covariate
/// steps both came back `failed` on a model that had just converged.
#[test]
fn the_init_stall_gate_is_off_inside_a_step_and_said_so() {
    let config = config_with(SPACE, "\n[strictness]\nreject_init_stall = true\n");
    assert_eq!(config.strictness.reject_init_stall, Some(true));
    for step in Step::ALL {
        let sub = step_config(&config, step).unwrap();
        assert_eq!(
            sub.strictness.reject_init_stall,
            Some(false),
            "{} kept the init-stall gate",
            step.label()
        );
        // Nothing else about the gate moves.
        assert_eq!(
            sub.strictness.strictness().require_converged,
            config.strictness.strictness().require_converged
        );
        assert_eq!(
            sub.strictness.strictness().reject_on_boundary,
            config.strictness.strictness().reject_on_boundary
        );
    }
    // The pipeline's own start fit keeps it: it begins at the file's initial
    // estimates, which is exactly what #751 is about — so the fit is made from
    // the file's own config, not from a step's narrowed one.
    let scripted = Scripted::new();
    let result = run(
        &scripted,
        &AmdOptions {
            retries: Retries::Skip,
            ..Default::default()
        },
        &config,
    );
    assert_eq!(scripted.calls()[0].kind, "start");
    assert!(
        scripted.calls()[0].stall_gate,
        "the start fit lost the init-stall gate"
    );
    assert!(
        result.notes.iter().any(|n| n == INIT_STALL_NOTE),
        "the relaxation is not in the notes: {:?}",
        result.notes
    );
}

/// A starting model that does not pass the gate is reported — every Δ in the
/// report is measured against it.
#[test]
fn a_starting_model_that_fails_the_gate_is_reported() {
    struct BadStart(Scripted);
    impl StepRunner for BadStart {
        fn run_step(
            &self,
            index: usize,
            step: Step,
            dir_name: &str,
            config: &SearchConfig,
            model: &ModelText,
        ) -> Result<StepOutput, String> {
            self.0.run_step(index, step, dir_name, config, model)
        }
        fn fit_one(
            &self,
            index: usize,
            tool: &str,
            dir_name: &str,
            config: &SearchConfig,
            model: &ModelText,
            n_starts: usize,
        ) -> Result<StepOutput, String> {
            let mut out = self
                .0
                .fit_one(index, tool, dir_name, config, model, n_starts)?;
            if tool == "start" {
                out.passed = false;
                out.rows[0].passed = false;
                out.rows[0].failures = vec!["condition number 1e9 exceeds 1e3".into()];
            }
            Ok(out)
        }
    }
    let result = run(
        &BadStart(Scripted::new()),
        &AmdOptions {
            retries: Retries::Skip,
            ..Default::default()
        },
        &config(),
    );
    let note = result
        .notes
        .iter()
        .find(|n| n.contains("does not pass the strictness gate"))
        .expect("the start model's verdict is reported");
    assert!(note.contains("condition number"), "{note}");
    // And the pipeline still ran: a gate failure is not a missing fit.
    assert_eq!(result.ran().count(), 6);
}

/// The residual step is handed a space with no statements at all, whatever the
/// file's space said — its candidates are the error forms.
#[test]
fn the_residual_step_is_handed_no_space() {
    let sub = step_config(&config(), Step::Residual).unwrap();
    assert!(sub.mfl.statements.is_empty());
    assert_eq!(sub.mfl_source, "");
    // And the config is otherwise the file's own.
    assert_eq!(sub.run.retries, config().run.retries);
}

/// A step whose tool fails is reported as a failed step, and the pipeline
/// carries on from the model it was handed.
///
/// The alternative — propagating the error — throws away every step that
/// already ran to re-report one message, which on a pipeline whose steps are
/// full population fits is the most expensive way to deliver the least
/// information.
#[test]
fn a_failed_step_is_reported_and_the_pipeline_carries_on() {
    let mut scripted = Scripted::new();
    scripted.fail_at = Some(2);
    let result = run(
        &scripted,
        &AmdOptions {
            retries: Retries::Skip,
            ..Default::default()
        },
        &config(),
    );
    assert!(!result.cancelled);
    assert_eq!(result.steps.len(), 6, "every planned step is reported");
    let failed: Vec<&StepOutcome> = result.failures().collect();
    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0].step, Step::Iivsearch);
    assert_eq!(failed[0].status(), "failed");
    assert!(failed[0].reason().unwrap().contains("not a covariate"));
    assert!(!failed[0].ok() && failed[0].ran());
    assert!(result
        .notes
        .iter()
        .any(|n| n.contains("failed and was skipped over")));

    // The next step starts from the model the failed one was handed, not from
    // nothing.
    let calls = scripted.calls();
    let handed = |dir: &str| {
        calls
            .iter()
            .find(|c| c.dir == dir)
            .unwrap_or_else(|| panic!("{dir} never ran"))
            .model
            .clone()
    };
    assert_eq!(handed("02-iivsearch"), handed("03-ruvsearch"));
    // And the run still produced a final model and a fit.
    assert!(result.final_fit.is_some());
}

/// A retries pass that cannot run costs the pass, not the model it was
/// refining.
#[test]
fn a_failed_retries_pass_keeps_the_selected_model() {
    struct NoRetries(Scripted);
    impl StepRunner for NoRetries {
        fn run_step(
            &self,
            index: usize,
            step: Step,
            dir_name: &str,
            config: &SearchConfig,
            model: &ModelText,
        ) -> Result<StepOutput, String> {
            self.0.run_step(index, step, dir_name, config, model)
        }
        fn fit_one(
            &self,
            index: usize,
            tool: &str,
            dir_name: &str,
            config: &SearchConfig,
            model: &ModelText,
            n_starts: usize,
        ) -> Result<StepOutput, String> {
            if tool == "retries" {
                return Err("the run directory is not writable".into());
            }
            self.0
                .fit_one(index, tool, dir_name, config, model, n_starts)
        }
    }
    let runner = NoRetries(Scripted::new());
    let result = run(
        &runner,
        &AmdOptions {
            retries: Retries::Final,
            ..Default::default()
        },
        &config(),
    );
    assert_eq!(
        result.final_fit.as_ref().map(|f| f.ofv),
        Some(runner.0.step_ofv(6))
    );
    assert!(
        result
            .notes
            .iter()
            .any(|n| n.contains("could not be run") && n.contains("not writable")),
        "{:?}",
        result.notes
    );
}

/// A retries pass whose fit does not pass the strictness gate is **not**
/// adopted, however low its OFV.
///
/// The pass exists to find a better optimum, not a better number: a fit that
/// fails the gate is one whose OFV the search is not allowed to trust, which
/// is the whole reason the runner gates every candidate before ranking it.
#[test]
fn a_retries_pass_that_fails_the_gate_is_not_adopted() {
    let mut scripted = Scripted::new();
    scripted.retries_fail_gate = true;
    scripted.retry_delta = -50.0;
    let result = run(
        &scripted,
        &AmdOptions {
            retries: Retries::Final,
            ..Default::default()
        },
        &config(),
    );
    assert_eq!(
        result.final_fit.as_ref().map(|f| f.ofv),
        Some(scripted.step_ofv(6)),
        "a fit the gate rejected was adopted because its OFV was lower"
    );
    assert!(
        result
            .notes
            .iter()
            .any(|n| n.contains("did not pass the strictness gate")),
        "{:?}",
        result.notes
    );
    // The pass is still in the candidate table, with its own row.
    assert!(result.rows.iter().any(|r| r.tool == "retries"));
}
