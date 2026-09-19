//! `W_PER_CMT_UNMATCHED` (#1405): a **declared** per-CMT entry that no observation
//! matched.
//!
//! The defect these pin: `pk::validate_per_cmt_scaling` checks `observed ⊆ declared`
//! and nothing checks the other direction, so a model declaring `obs_scale[CMT=2]`
//! against a dataset with no CMT-2 row validates clean and the entry silently does
//! nothing. `{1} ⊆ {1, 2}`, so the fit runs — at an objective 1.7e11 times larger on
//! the issue's own reprex — with no diagnostic from `fit()` or `ferx check`.
//!
//! Three declared-side maps have the hole, and they are not three model classes
//! somebody listed: they are the three [`CmtConsumer`] variants that carry a
//! declared map, and the check walks the enum. One test per channel rather than one
//! table, so a walk that loses a channel reddens the channel it lost — the #1223
//! lesson, which here has real bite because all three channels share **one** loop
//! body and a single fixture could otherwise carry all three. Every fixture below
//! therefore asserts that its own channel is the *only* live one before asserting
//! anything about the warning.
use crate::api::check_model_data_warnings;
use crate::api::validation::CmtConsumer;
use crate::diagnostics::Diagnostic;
use crate::io::datareader::SelectionFilter;
use crate::parser::model_parser::parse_full_model;
use crate::read_nonmem_csv;
use crate::types::{CompiledModel, FitOptions, Population, WarningCode};
use std::io::Write;

const CODE: &str = "W_PER_CMT_UNMATCHED";

/// Observations on compartment 1 only — two subjects, one IV bolus each. The
/// declared `CMT=2` entry of every model below matches nothing here.
const OBS_CMT1_ONLY: &str = "\
ID,TIME,DV,EVID,AMT,CMT,MDV
1,0,.,1,100,1,1
1,1,8.0,0,.,1,0
1,4,5.0,0,.,1,0
2,0,.,1,100,1,1
2,1,7.5,0,.,1,0
2,4,4.8,0,.,1,0
";

/// Dose records only — subjects present, not one scored observation between them.
/// The *second* observation-free shape, and the one that distinguishes a guard on the
/// observed set from a guard on the subject list.
const DOSES_ONLY: &str = "\
ID,TIME,DV,EVID,AMT,CMT,MDV
1,0,.,1,100,1,1
2,0,.,1,100,1,1
";

/// The same data with compartment 2 also observed — the matched control. Written
/// with a `STUDY` column so one fixture can serve the `[data_selection]` case too.
const OBS_BOTH_CMTS: &str = "\
ID,TIME,DV,EVID,AMT,CMT,MDV,STUDY
1,0,.,1,100,1,1,1
1,1,8.0,0,.,1,0,1
1,4,5.0,0,.,1,0,1
1,1,0.9,0,.,2,0,2
1,4,0.6,0,.,2,0,2
2,0,.,1,100,1,1,1
2,1,7.5,0,.,1,0,1
2,4,4.8,0,.,1,0,1
2,1,0.8,0,.,2,0,2
2,4,0.5,0,.,2,0,2
";

fn csv(body: &str) -> tempfile::NamedTempFile {
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    write!(f, "{body}").unwrap();
    f.flush().unwrap();
    f
}

fn population(body: &str) -> Population {
    let f = csv(body);
    read_nonmem_csv(f.path(), None, None).expect("fixture CSV reads")
}

/// The same read with a `[data_selection]` clause applied, so `Population::exclusions`
/// is the reader's own record of what the filter removed rather than a hand-set field.
fn filtered_population(model: &CompiledModel, body: &str, ignore: &str) -> Population {
    let f = csv(body);
    let filter =
        SelectionFilter::from_opts(&[ignore.to_string()], &[], &[]).expect("clause parses");
    crate::api::read_population_for(
        model,
        &None,
        f.path().to_str().unwrap(),
        None,
        None,
        Some(&filter),
        &[],
    )
    .expect("filtered read")
    .0
}

fn model_of(src: &str) -> CompiledModel {
    parse_full_model(src).expect("fixture model parses").model
}

fn warnings_of(model: &CompiledModel, pop: &Population) -> Vec<Diagnostic> {
    check_model_data_warnings(model, pop, &model.default_params)
}

fn unmatched_messages(model: &CompiledModel, pop: &Population) -> Vec<String> {
    warnings_of(model, pop)
        .into_iter()
        .filter(|d| d.code == CODE)
        .map(|d| d.message)
        .collect()
}

/// The channel-isolation guard every fixture runs before it asserts anything.
///
/// Without it a per-CMT `[error_model]` fixture that also happens to carry a per-CMT
/// readout would satisfy the readout test as well, and deleting either channel from
/// the shared walk would leave both tests green — exactly the twin whose second leg
/// is never exercised.
fn assert_only_live_channel(model: &CompiledModel, want: CmtConsumer) {
    let opts = FitOptions::default();
    let live: Vec<CmtConsumer> = CmtConsumer::iter()
        .filter(|c| c.is_live(model, &opts))
        .collect();
    assert_eq!(
        live,
        vec![want],
        "fixture must make {want:?} the only live CMT consumer, else one fixture \
         carries several channels of the shared walk"
    );
}

// ---------------------------------------------------------------------------
// Fixtures — one model per declared-side channel
// ---------------------------------------------------------------------------

/// `[scaling] obs_scale[CMT=N]` (Forms A/B) on an analytical model, declaring an
/// entry for compartment 2. This is the issue's own reprex shape.
fn per_cmt_scaling_model(declared: &str) -> CompiledModel {
    model_of(&format!(
        "[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.1 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[scaling]
{declared}

[error_model]
  DV ~ proportional(PROP)
"
    ))
}

/// `[error_model] CMT=N:` on a one-state `[odes]` model whose readout is the plain
/// `obs_cmt` one, so the readout channel stays inert.
fn per_cmt_error_model(declared: &str) -> CompiledModel {
    model_of(&format!(
        "[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.1 (sd)
  sigma ADD ~ 0.1 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -CL/V * central

[scaling]
  obs_scale = V

[error_model]
{declared}
"
    ))
}

/// `[scaling] y[CMT=N]` (Form C per-CMT) on a one-state `[odes]` model with a single
/// plain error model, so the error-model channel stays inert.
fn per_cmt_readout_model(declared: &str) -> CompiledModel {
    model_of(&format!(
        "[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.1 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = -CL/V * central

[scaling]
{declared}

[error_model]
  DV ~ proportional(PROP)

[fit_options]
  gradient = fd
"
    ))
}

/// The **other** struct that carries `OdeReadout::PerCmt`: Form C per-CMT on an
/// analytical `pk` model (`AnalyticReadout::readout`, since #650).
///
/// `PerCmtReadout` unions two legs, and a fixture on the `[odes]` leg alone cannot
/// see the analytic one — deleting it would leave `unmatched_per_cmt_readout_entry_warns`
/// green. #1404 round 2 inspected only one of the two structs; the point of having
/// both fixtures is that each mutation names its own side.
fn per_cmt_analytic_readout_model(declared: &str) -> CompiledModel {
    model_of(&format!(
        "[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.1 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[scaling]
{declared}

[error_model]
  DV ~ proportional(PROP)
"
    ))
}

// ---------------------------------------------------------------------------
// Channel 1 — ScalingSpec::PerCmt
// ---------------------------------------------------------------------------

#[test]
fn unmatched_per_cmt_scaling_entry_warns() {
    // Regression: `validate_per_cmt_scaling` checks observed ⊆ declared only, so
    // `{1} ⊆ {1, 2}` passes and `obs_scale[CMT=2] = 1000` is dead configuration.
    // Measured on this shape in #1405: the same data spelled with an explicit
    // `CMT=2` on the observation rows fits at OFV 0.0357, and with the column
    // dropped at 6097015712.1246 — validation silent in both arms.
    let m = per_cmt_scaling_model("  obs_scale[CMT=1] = 1\n  obs_scale[CMT=2] = 1000");
    assert_only_live_channel(&m, CmtConsumer::PerCmtScaling);
    let pop = population(OBS_CMT1_ONLY);

    let msgs = unmatched_messages(&m, &pop);
    assert_eq!(msgs.len(), 1, "expected one finding, got {msgs:?}");
    let msg = &msgs[0];
    // The channel and the compartment, not just "something is unmatched": with one
    // code for three channels, a message that names neither cannot be acted on.
    assert!(
        msg.contains("obs_scale[CMT=N]"),
        "must name the syntax that spells the dead entry: {msg}"
    );
    // Both sets, spelled out. Asserted as the rendered phrases rather than as "contains
    // a 2", which the message satisfies through its own prose ("compartment 1") whichever
    // way round the two sets are: a difference computed backwards would pass that.
    assert!(
        msg.contains("declares compartment(s) 2 "),
        "must name 2 as the unmatched entry: {msg}"
    );
    assert!(
        msg.contains("(observed: 1)"),
        "must name 1 as what the data does carry: {msg}"
    );
    // No live `[data_selection]`, so the filter clause must not be offered as an
    // explanation — the message would otherwise send every user chasing a block
    // their model does not have.
    assert!(
        !msg.contains("data_selection"),
        "unfiltered read must not blame [data_selection]: {msg}"
    );
}

#[test]
fn matched_per_cmt_scaling_entries_are_silent() {
    // The false-positive direction, which is the dangerous one for a warning that
    // fires on every fit: a model whose declared entries are all exercised must say
    // nothing at all.
    let m = per_cmt_scaling_model("  obs_scale[CMT=1] = 1\n  obs_scale[CMT=2] = 1000");
    let pop = population(OBS_BOTH_CMTS);
    assert_eq!(unmatched_messages(&m, &pop), Vec::<String>::new());
}

// ---------------------------------------------------------------------------
// Channel 2 — ErrorSpec::PerCmt
// ---------------------------------------------------------------------------

#[test]
fn unmatched_per_cmt_error_model_entry_warns() {
    // Regression: `check_per_cmt_error_model` is the error-direction twin of
    // `validate_per_cmt_scaling` and has the same hole — a `CMT=2: DV ~ ...` line
    // with no CMT-2 row is accepted in silence.
    let m = per_cmt_error_model("  CMT=1: DV ~ proportional(PROP)\n  CMT=2: DV ~ additive(ADD)");
    assert_only_live_channel(&m, CmtConsumer::PerCmtErrorModel);
    let pop = population(OBS_CMT1_ONLY);

    let msgs = unmatched_messages(&m, &pop);
    assert_eq!(msgs.len(), 1, "expected one finding, got {msgs:?}");
    assert!(
        msgs[0].contains("[error_model] CMT=N:"),
        "must name the error-model channel: {}",
        msgs[0]
    );
    assert!(
        msgs[0].contains("declares compartment(s) 2 ") && msgs[0].contains("(observed: 1)"),
        "must name both sets the right way round: {}",
        msgs[0]
    );
}

#[test]
fn matched_per_cmt_error_model_entries_are_silent() {
    let m = per_cmt_error_model("  CMT=1: DV ~ proportional(PROP)\n  CMT=2: DV ~ additive(ADD)");
    let pop = population(OBS_BOTH_CMTS);
    assert_eq!(unmatched_messages(&m, &pop), Vec::<String>::new());
}

// ---------------------------------------------------------------------------
// Channel 3 — OdeReadout::PerCmt, on either readout struct
// ---------------------------------------------------------------------------

#[test]
fn unmatched_per_cmt_readout_entry_warns() {
    // Regression: the Form C per-CMT readout is the third declared-side map, and
    // #1404 round 2 inspected only the analytical one of the two structs that carry
    // it. Measured in #1405 on this shape: OFV 10028.0940 with the column against
    // 0.0357 without.
    let m = per_cmt_readout_model("  y[CMT=1] = central / V\n  y[CMT=2] = central / V * 1000");
    assert_only_live_channel(&m, CmtConsumer::PerCmtReadout);
    let pop = population(OBS_CMT1_ONLY);

    let msgs = unmatched_messages(&m, &pop);
    assert_eq!(msgs.len(), 1, "expected one finding, got {msgs:?}");
    assert!(
        msgs[0].contains("y[CMT=N]"),
        "must name the readout syntax: {}",
        msgs[0]
    );
    assert!(
        msgs[0].contains("declares compartment(s) 2 ") && msgs[0].contains("(observed: 1)"),
        "must name both sets the right way round: {}",
        msgs[0]
    );
}

#[test]
fn matched_per_cmt_readout_entries_are_silent() {
    let m = per_cmt_readout_model("  y[CMT=1] = central / V\n  y[CMT=2] = central / V * 1000");
    let pop = population(OBS_BOTH_CMTS);
    assert_eq!(unmatched_messages(&m, &pop), Vec::<String>::new());
}

#[test]
fn unmatched_per_cmt_readout_entry_warns_on_the_analytic_readout_too() {
    // The second leg of the `PerCmtReadout` union, asserted separately so each
    // mutation names its own side. The test above runs on `OdeSpec::readout`; this
    // one on `AnalyticReadout::readout`, which is a different struct reached through
    // a different `if let` — dropping either leg leaves the other test green, which
    // is precisely the twin-with-an-unexercised-leg shape #1223 was about.
    let m = per_cmt_analytic_readout_model("  y[CMT=1] = CENTRAL / V\n  y[CMT=2] = CENTRAL / V");
    assert!(
        m.ode_spec.is_none(),
        "fixture must carry NO ode_spec, or it exercises the other leg"
    );
    assert!(
        m.analytic_readout.is_some(),
        "fixture must carry an analytic readout"
    );
    assert_only_live_channel(&m, CmtConsumer::PerCmtReadout);
    let pop = population(OBS_CMT1_ONLY);

    let msgs = unmatched_messages(&m, &pop);
    assert_eq!(msgs.len(), 1, "expected one finding, got {msgs:?}");
    assert!(
        msgs[0].contains("y[CMT=N]")
            && msgs[0].contains("declares compartment(s) 2 ")
            && msgs[0].contains("(observed: 1)"),
        "must name the channel and both sets: {}",
        msgs[0]
    );
}

// ---------------------------------------------------------------------------
// The rendered list
// ---------------------------------------------------------------------------

#[test]
fn several_unmatched_entries_are_listed_in_one_finding() {
    // Regression: `join_cmts`. A dead block carried over from another model usually
    // has more than one dead entry, and `Vec`'s `Debug` would render `[2, 3]` — a
    // user pasting that back into `obs_scale[CMT=...]` gets a parse error. One
    // finding per channel, not per compartment, so the list has to be legible.
    let m = per_cmt_scaling_model(
        "  obs_scale[CMT=1] = 1\n  obs_scale[CMT=2] = 1000\n  obs_scale[CMT=3] = 500",
    );
    let pop = population(OBS_CMT1_ONLY);
    let msgs = unmatched_messages(&m, &pop);
    assert_eq!(msgs.len(), 1, "one finding per channel, got {msgs:?}");
    assert!(
        msgs[0].contains("declares compartment(s) 2, 3 "),
        "both dead entries, comma-joined and unbracketed: {}",
        msgs[0]
    );
}

// ---------------------------------------------------------------------------
// The two ways to ship a false positive
// ---------------------------------------------------------------------------

#[test]
fn an_observation_free_population_is_silent() {
    // `ferx check` without `--data` validates against an empty population. Without
    // the guard, the most common invocation of the check reports *every* declared
    // entry as dead — the check would be wrong on the path it is most used on.
    let m = per_cmt_scaling_model("  obs_scale[CMT=1] = 1\n  obs_scale[CMT=2] = 1000");
    let empty = Population {
        subjects: Vec::new(),
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    assert!(empty.n_obs() == 0, "fixture must carry no observations");
    assert_eq!(unmatched_messages(&m, &empty), Vec::<String>::new());
}

#[test]
fn a_dose_only_population_is_silent_too() {
    // The second observation-free shape, and the reason the first one is not enough:
    // a guard written `population.subjects.is_empty()` is satisfied by the test above
    // and wrong here. A design dataset read for `simulate()` — or a `--data` file whose
    // observations were all filtered or all MDV=1 — has subjects, doses and no scored
    // observation, and every declared entry would be reported as dead.
    let m = per_cmt_scaling_model("  obs_scale[CMT=1] = 1\n  obs_scale[CMT=2] = 1000");
    let pop = population(DOSES_ONLY);
    assert!(
        !pop.subjects.is_empty(),
        "fixture must carry subjects, or it is the previous test again"
    );
    assert_eq!(pop.n_obs(), 0, "fixture must carry no scored observation");
    assert!(
        pop.subjects.iter().all(|s| s.obs_cmts.is_empty()),
        "fixture must carry no observed compartment"
    );
    assert_eq!(unmatched_messages(&m, &pop), Vec::<String>::new());
}

#[test]
fn a_filtered_read_names_data_selection_as_a_cause() {
    // `[data_selection]` legitimately removes rows, so after a filter a live entry
    // looks dead. The decision taken in #1405 is to *name* the filter rather than
    // suppress the warning — suppressing it re-silences the case the issue is about.
    //
    // Straddle, asserted here so it cannot become a tautology: the same model and
    // the same file, read twice, differing only in whether the clause is applied.
    // The unfiltered read matches both compartments and must stay silent; the
    // filtered read must warn *and* mention the block.
    let m = per_cmt_scaling_model("  obs_scale[CMT=1] = 1\n  obs_scale[CMT=2] = 1000");
    let unfiltered = population(OBS_BOTH_CMTS);
    assert_eq!(
        unmatched_messages(&m, &unfiltered),
        Vec::<String>::new(),
        "the unfiltered side of the pair must be silent, or the pair proves nothing"
    );

    let pop = filtered_population(&m, OBS_BOTH_CMTS, "STUDY == 2");
    let excl = pop
        .exclusions
        .as_ref()
        .expect("a filtered read records its exclusions");
    assert!(
        excl.n_obs_excluded > 0,
        "the clause must actually remove observation rows: {excl:?}"
    );

    let msgs = unmatched_messages(&m, &pop);
    assert_eq!(msgs.len(), 1, "expected one finding, got {msgs:?}");
    assert!(
        msgs[0].contains("data_selection"),
        "a filtered read must offer the filter as a cause: {}",
        msgs[0]
    );
}

// ---------------------------------------------------------------------------
// The second taxonomy — `classify_warning`
// ---------------------------------------------------------------------------

#[test]
fn the_warning_classifies_by_its_own_token() {
    // `FitResult.warnings` carries message text only, not the diagnostic code, so
    // `classify_warning` matches on substrings and the chain is order sensitive.
    // Measured on the chain at this SHA: no arm matches "scaling" or "error model",
    // but a broad `parameters` arm sits near the end of it — so this warning carries
    // its own `W_` token and is claimed by a token arm ahead of every prose one.
    //
    // Straddle: the same message with the token removed must NOT land on
    // `DataQuality`. Without it, an arm re-keyed on prose (or a message that happens
    // to hit `DataQuality` by accident through `ss=1 dose` / `non-positive dv` /
    // `ltbs`) would pass — that arm serves six unrelated inputs, so asserting the
    // category alone asserts almost nothing.
    let m = per_cmt_scaling_model("  obs_scale[CMT=1] = 1\n  obs_scale[CMT=2] = 1000");
    let pop = population(OBS_CMT1_ONLY);
    let msgs = unmatched_messages(&m, &pop);
    assert_eq!(msgs.len(), 1, "expected one finding, got {msgs:?}");
    let msg = &msgs[0];

    assert!(
        msg.contains(CODE),
        "the message must carry its own token, since the code does not travel with it: {msg}"
    );
    let entry = crate::types::classify_warning(msg);
    assert_eq!(entry.category, WarningCode::DataQuality, "message: {msg}");

    let detokenised = msg.replace(CODE, "a per-CMT entry");
    assert_ne!(
        crate::types::classify_warning(&detokenised).category,
        WarningCode::DataQuality,
        "the token must be what claims this message, not its prose: {detokenised}"
    );
}

// ---------------------------------------------------------------------------
// The enumeration itself
// ---------------------------------------------------------------------------

#[test]
fn declared_cmts_is_some_exactly_when_the_channel_is_live() {
    // The coupling that replaces an `is_live` call inside the walk:
    // `check_model_data_warnings` has no `&FitOptions` to pass, so the check gates on
    // `declared_cmts` alone. That is only safe while the two agree — a channel that
    // declares compartments while inert would warn about a map nothing reads, and a
    // live channel with no declared set would never be checked.
    //
    // Asserted over every fixture below rather than once, so a mutation that makes a
    // non-per-CMT arm return `Some` (the "is_live forced true" probe) fails here
    // instead of silently widening the walk.
    let opts = FitOptions::default();
    // The shape the parser hands an endpoint-only model: `ErrorSpec::PerCmt` with an
    // **empty** map, which `is_live` deliberately excludes (it dispatches no error
    // model at all). Built by clearing the map rather than through a `[event_model]`,
    // so the case is reachable without the `survival` feature — and it is the fixture
    // that makes the non-empty guard in `declared_cmts` observable: without it, an arm
    // returning `Some(∅)` is behaviour-neutral and nothing here can fail.
    let mut empty_per_cmt_map =
        per_cmt_error_model("  CMT=1: DV ~ proportional(PROP)\n  CMT=2: DV ~ additive(ADD)");
    match &mut empty_per_cmt_map.error_spec {
        crate::types::ErrorSpec::PerCmt(map) => map.clear(),
        other => panic!("fixture must be ErrorSpec::PerCmt, got {other:?}"),
    }
    assert!(
        !CmtConsumer::PerCmtErrorModel.is_live(&empty_per_cmt_map, &opts),
        "an empty PerCmt map must read as inert, or this fixture proves nothing"
    );

    let fixtures: Vec<CompiledModel> = vec![
        per_cmt_scaling_model("  obs_scale[CMT=1] = 1\n  obs_scale[CMT=2] = 1000"),
        per_cmt_error_model("  CMT=1: DV ~ proportional(PROP)\n  CMT=2: DV ~ additive(ADD)"),
        per_cmt_readout_model("  y[CMT=1] = central / V\n  y[CMT=2] = central / V * 1000"),
        per_cmt_analytic_readout_model("  y[CMT=1] = CENTRAL / V\n  y[CMT=2] = CENTRAL / V"),
        empty_per_cmt_map,
        // A model with no per-CMT anything: every declared-side arm must be `None`.
        per_cmt_scaling_model("  obs_scale = V"),
    ];
    let mut any_some = false;
    for m in &fixtures {
        for c in CmtConsumer::iter() {
            if let Some((syntax, declared)) = c.declared_cmts(m) {
                any_some = true;
                assert!(
                    c.is_live(m, &opts),
                    "{c:?} declares {declared:?} ({syntax}) but reports itself inert"
                );
                assert!(
                    !declared.is_empty(),
                    "{c:?} must not report an empty declared set as Some"
                );
            }
        }
    }
    assert!(
        any_some,
        "no fixture produced a declared set — the loop asserted nothing"
    );
}

#[test]
fn every_channel_without_a_declared_map_says_why() {
    // The three channels with no declared-side map are not oversights, and a
    // reviewer should not have to infer that. Pinned as a list so a new
    // `cmt_consumers!` entry that lands in the `None` half is a deliberate choice
    // someone made here, next to the arm.
    let opts = FitOptions::default();
    let m = per_cmt_scaling_model("  obs_scale[CMT=1] = 1\n  obs_scale[CMT=2] = 1000");
    for c in [
        // A dose row names its own compartment; there is no declared list to be dead.
        CmtConsumer::DoseCompartment,
        // Already an *error*: `E_ENDPOINT_NO_RECORDS` reports a declared endpoint
        // with no routed row, so a warning here would double-report it.
        CmtConsumer::EndpointRouting,
        // Lives on `FitOptions`, not the model, and declares a predicate rather than
        // a set of compartments.
        CmtConsumer::DataSelectionFilter,
    ] {
        assert!(
            c.declared_cmts(&m).is_none(),
            "{c:?} must declare no per-CMT map"
        );
    }
    // Non-degeneracy: the model above really does make one of the *other* channels
    // live, so this test is not passing because nothing is configured.
    assert!(CmtConsumer::PerCmtScaling.is_live(&m, &opts));
}
