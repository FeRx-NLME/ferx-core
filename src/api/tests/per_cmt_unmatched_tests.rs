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
use crate::api::validation::CmtConsumer;
use crate::api::{check_model_data_warnings, predict_diag, simulate_with_options_diag};
use crate::diagnostics::Diagnostic;
use crate::io::datareader::SelectionFilter;
use crate::parser::model_parser::parse_full_model;
use crate::read_nonmem_csv;
use crate::types::{CompiledModel, FitOptions, Population, WarningCode};
use crate::SimulateOptions;
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

/// The same three records with **no `CMT` column at all** — the dataset
/// `W_CMT_DEFAULTED` exists for, and the only one the missing-column advice is true of.
///
/// Review r2's finding: every CSV in this file carried a `CMT` column, so the test named
/// `..._only_when_the_data_reads_column_less` never read a column-less dataset. It
/// straddled `observed == {1}` — a *symptom* of a defaulted column — between two datasets
/// that both have one.
const OBS_NO_CMT_COLUMN: &str = "\
ID,TIME,DV,EVID,AMT,MDV
1,0,.,1,100,1
1,1,8.0,0,.,0
1,4,5.0,0,.,0
2,0,.,1,100,1
2,1,7.5,0,.,0
2,4,4.8,0,.,0
";

/// Observations on compartment **2** only — the mirror of `OBS_CMT1_ONLY`, and the
/// set on which the "your CMT column is missing" advice is false: the column is
/// present, correct, and being read.
const OBS_CMT2_ONLY: &str = "\
ID,TIME,DV,EVID,AMT,CMT,MDV
1,0,.,1,100,1,1
1,1,0.9,0,.,2,0
1,4,0.6,0,.,2,0
2,0,.,1,100,1,1
2,1,0.8,0,.,2,0
2,4,0.5,0,.,2,0
";

/// Observations on compartments 1 and **3**, with nothing on 2. The dataset that
/// separates "the filter removed the rows that would have matched" from "the filter
/// removed rows that could not have matched anything declared".
const OBS_CMT1_AND_3: &str = "\
ID,TIME,DV,EVID,AMT,CMT,MDV
1,0,.,1,100,1,1
1,1,8.0,0,.,1,0
1,4,5.0,0,.,1,0
1,1,0.9,0,.,3,0
1,4,0.6,0,.,3,0
2,0,.,1,100,1,1
2,1,7.5,0,.,1,0
2,4,4.8,0,.,1,0
2,1,0.8,0,.,3,0
2,4,0.5,0,.,3,0
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
    assert_live_channels(model, &[want]);
}

/// The same guard for a fixture that is *deliberately* multi-channel: the exact live
/// set, not "at least these", so a fixture that quietly gains a third channel is a red
/// test rather than a silently broader assertion.
fn assert_live_channels(model: &CompiledModel, want: &[CmtConsumer]) {
    let opts = FitOptions::default();
    let live: Vec<CmtConsumer> = CmtConsumer::iter()
        .filter(|c| c.is_live(model, &opts))
        .collect();
    assert_eq!(
        live,
        want.to_vec(),
        "fixture must make exactly {want:?} live, else one fixture carries channels of \
         the shared walk it does not mean to"
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

/// Per-CMT scaling **and** a per-CMT `[error_model]` on one model — what a real
/// multi-analyte model looks like, and the shape `assert_only_live_channel` forbids
/// on every fixture above.
///
/// The single-channel fixtures pin that a walk which loses a channel reddens the test
/// for the channel it lost; this one pins the other half — that one model producing two
/// dead entries produces **two** findings, each carrying its own syntax and its own
/// `[block]`, rather than one merged report or the first channel only.
///
/// `[odes]` and not an analytical `pk` block, because the parser rejects the pairing
/// outright on one: "Per-CMT error models (`CMT=N: DV ~ ...`) require an ODE-based
/// [structural_model]". The readout stays the plain `obs_cmt` one so the third channel
/// is inert and the live set is exactly two.
fn two_live_per_cmt_channels_model() -> CompiledModel {
    model_of(
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
  obs_scale[CMT=1] = 1
  obs_scale[CMT=2] = 1000

[error_model]
  CMT=1: DV ~ proportional(PROP)
  CMT=2: DV ~ additive(ADD)
",
    )
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

#[test]
fn a_filter_that_removed_no_observation_is_not_blamed() {
    // The other side of the note's gate, and the one a mutation sweep found missing:
    // the test above fixes `exclusions = None` against `exclusions = Some(n > 0)`, so
    // relaxing the gate from `n_obs_excluded > 0` to "a filter was applied at all"
    // changed nothing on any fixture and survived.
    //
    // `Population::exclusions` is `Some` whenever a `[data_selection]` block exists,
    // fired or not. A clause that matched nothing — or one that only removed dose
    // records — leaves every observation in place, so it cannot be why an entry is
    // unmatched, and naming it sends the user to edit a block that is not the cause.
    let m = per_cmt_scaling_model("  obs_scale[CMT=1] = 1\n  obs_scale[CMT=2] = 1000");
    let pop = filtered_population(&m, OBS_CMT1_ONLY, "TIME == 999");
    let excl = pop
        .exclusions
        .as_ref()
        .expect("a filtered read records its exclusions even when no clause fires");
    assert_eq!(
        excl.n_obs_excluded, 0,
        "this clause must remove no observation row, or it is the previous test: {excl:?}"
    );

    let msgs = unmatched_messages(&m, &pop);
    assert_eq!(msgs.len(), 1, "the entry is still dead, got {msgs:?}");
    assert!(
        !msgs[0].contains("data_selection"),
        "a filter that removed no observation must not be offered as the cause: {}",
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
fn declared_cmts_is_some_only_when_the_channel_is_live() {
    // The coupling that replaces an `is_live` call inside the walk:
    // `check_model_data_warnings` has no `&FitOptions` to pass, so the check gates on
    // `declared_cmts` alone. The direction that matters for safety is the one asserted
    // here — `Some ⟹ is_live`, so the walk never warns about a map nothing reads.
    //
    // **Not** the converse, which is false by construction and deliberately so:
    // `PerCmtScaling` and `PerCmtReadout` are live on any `PerCmt` value, while their
    // `declared_cmts` arms carry a non-empty guard, so an empty map is live with no
    // declared set. That costs nothing — an empty map has an empty difference and
    // could never produce a finding anyway.
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
            if let Some((syntax, block, declared)) = c.declared_cmts(m) {
                any_some = true;
                assert!(
                    c.is_live(m, &opts),
                    "{c:?} declares {declared:?} ({syntax}) but reports itself inert"
                );
                assert!(
                    !declared.is_empty(),
                    "{c:?} must not report an empty declared set as Some"
                );
                // The block travels with the syntax (#1456 review r1, finding 4), so
                // a new channel cannot inherit `"scaling"` from a `_` arm in the
                // caller. Pinned as the containing block of the syntax it ships with.
                assert!(
                    syntax.starts_with(&format!("[{block}]")),
                    "{c:?} reports block `{block}` for syntax `{syntax}`"
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
    // reviewer should not have to infer that. What enforces that a *new* channel is
    // considered at all is the exhaustive `match` in `declared_cmts`, not this list —
    // a new variant landing in the `None` half does not redden anything here. What
    // this does pin is that these three stay `None`: flipping one to `Some` without
    // a fixture and a message is a red test.
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

// ---------------------------------------------------------------------------
// The advice half — gated on the observed set (#1456 review r1, finding 1)
// ---------------------------------------------------------------------------

#[test]
fn the_missing_column_advice_is_offered_only_when_the_data_reads_column_less() {
    // Review r1's first measured finding: the message ended "The usual cause is on the
    // data side: a missing or mis-mapped CMT column keys every observation to
    // compartment 1. Check the CMT column" on a dataset whose CMT column was present,
    // correct and read — the PR body's own *after* block was that case. `{1}` is what
    // the reader defaults a column-less dataset to (#1009), so it is the one observed
    // set for which the sentence is a live hypothesis.
    //
    // The straddle, asserted here and not split across two tests so it cannot quietly
    // become a tautology: the same model and the same declared entries against two
    // datasets that differ in the observed compartment and nothing else. One is on each
    // side of the gate, and the assertion on each side is the *negation* of the other's
    // — deleting the gate (either branch always taken) reddens one of the two halves.
    const COLUMN_ADVICE: &str = "missing or mis-mapped CMT column";
    let m = per_cmt_scaling_model("  obs_scale[CMT=1] = 1\n  obs_scale[CMT=2] = 1000");
    assert_only_live_channel(&m, CmtConsumer::PerCmtScaling);

    // The column really is absent, and the reader says so. Asserted rather than assumed:
    // review r2 measured that the old fixture pair straddled a *symptom* of a defaulted
    // column between two datasets that both had one, so the pair could not fail.
    let column_less = population(OBS_NO_CMT_COLUMN);
    assert!(
        column_less
            .warnings
            .iter()
            .any(|w| w.starts_with("W_CMT_DEFAULTED")),
        "this side of the pair must be the column-less case the reader reports: {:?}",
        column_less.warnings
    );
    let on = unmatched_messages(&m, &column_less);
    assert_eq!(on.len(), 1, "expected one finding, got {on:?}");
    assert!(
        on[0].contains("(observed: 1)"),
        "this side of the pair must observe {{1}}: {}",
        on[0]
    );
    assert!(
        on[0].contains(COLUMN_ADVICE),
        "a column-less dataset is the case the column advice is true of: {}",
        on[0]
    );

    // The case review r2 measured as a false positive, and the reason the gate moved
    // off the observed set: a **present, correct, all-`1`** column. `observed == {1}` and
    // `observed ∪ excluded == {1}` are both true here, so either earlier gate offers the
    // column advice — and this is the ordinary `predict()` shape and the PR's own stated
    // legitimate case, a shared `[scaling]` block declaring more than one dataset uses.
    let present_all_ones = population(OBS_CMT1_ONLY);
    assert!(
        !present_all_ones
            .warnings
            .iter()
            .any(|w| w.starts_with("W_CMT_DEFAULTED")),
        "this side must have a present, correct column: {:?}",
        present_all_ones.warnings
    );
    let present = unmatched_messages(&m, &present_all_ones);
    assert_eq!(present.len(), 1, "expected one finding, got {present:?}");
    assert!(
        present[0].contains("(observed: 1)"),
        "and must observe exactly {{1}}, or it is not the confusable case: {}",
        present[0]
    );
    assert!(
        !present[0].contains(COLUMN_ADVICE),
        "the column is present and correct — the advice must not be offered: {}",
        present[0]
    );
    // The straddle itself. These two observe the *same* set and differ only in whether
    // the column exists, so a gate keyed on the observed set makes them identical — which
    // is exactly what review r2 measured (`assert_eq!` passed).
    assert_ne!(
        on[0], present[0],
        "the column-less and column-present messages must differ; equal means the gate \
         is reading a symptom again"
    );

    // Observed {2} — same block, same declared set, a column that is being read.
    let off = unmatched_messages(&m, &population(OBS_CMT2_ONLY));
    assert_eq!(off.len(), 1, "expected one finding, got {off:?}");
    assert!(
        off[0].contains("declares compartment(s) 1 ") && off[0].contains("(observed: 2)"),
        "this side of the pair must observe {{2}} and leave entry 1 unmatched: {}",
        off[0]
    );
    assert!(
        !off[0].contains(COLUMN_ADVICE),
        "the column advice is false here — the column is present and correct: {}",
        off[0]
    );
    // And the message is still actionable rather than merely shorter: dropping the
    // whole advice half (the mutation that killed 0 of 15 tests in review r1) leaves
    // no instruction at all, and this is what fails then.
    assert!(
        off[0].contains("delete them") && off[0].contains("CMT column is being read"),
        "the other branch must still say what to do: {}",
        off[0]
    );
}

// ---------------------------------------------------------------------------
// Blaming `[data_selection]` — only for compartments it actually emptied
// (#1456 review r1, finding 3)
// ---------------------------------------------------------------------------

#[test]
fn a_filter_is_blamed_only_for_the_compartments_it_emptied() {
    // Review r1's third measured finding: the note gated on `n_obs_excluded > 0`, so
    // `ignore = CMT == 3` against a dataset observing {1, 3} named `[data_selection]`
    // as a possible cause of a dead `[CMT=2]` entry — a block that cannot be the
    // reason, since entry 2 had nothing to match before the filter ran either. The fix
    // is on the data side: `ExclusionSummary::obs_cmts_excluded` records *which*
    // compartments went, so the note can intersect.
    //
    // One clause, one dataset, one variable: the declared set. Entry 2 is unmatched for
    // a reason the filter had nothing to do with; entry 3 is unmatched exactly because
    // of it. Both halves land in the *same* message below, which is what makes this a
    // straddle rather than two unrelated assertions — the note must name 3 and must not
    // name 2, in one string.
    let m = per_cmt_scaling_model(
        "  obs_scale[CMT=1] = 1\n  obs_scale[CMT=2] = 1000\n  obs_scale[CMT=3] = 500",
    );
    // Non-degeneracy: unfiltered, only entry 2 is dead and no filter is mentioned.
    let unfiltered = unmatched_messages(&m, &population(OBS_CMT1_AND_3));
    assert_eq!(
        unfiltered.len(),
        1,
        "expected one finding, got {unfiltered:?}"
    );
    assert!(
        unfiltered[0].contains("declares compartment(s) 2 ")
            && !unfiltered[0].contains("data_selection"),
        "before the filter, entry 2 is dead and nothing is blamed for it: {}",
        unfiltered[0]
    );

    let pop = filtered_population(&m, OBS_CMT1_AND_3, "CMT == 3");
    let excl = pop
        .exclusions
        .as_ref()
        .expect("a filtered read records its exclusions");
    assert_eq!(
        excl.obs_cmts_excluded,
        vec![3],
        "the clause must remove compartment 3's observations and only those: {excl:?}"
    );

    let msgs = unmatched_messages(&m, &pop);
    assert_eq!(msgs.len(), 1, "expected one finding, got {msgs:?}");
    let msg = &msgs[0];
    assert!(
        msg.contains("declares compartment(s) 2, 3 "),
        "both entries are now unmatched: {msg}"
    );
    assert!(
        msg.contains("observations on compartment(s) 3, so check"),
        "the filter must be named for compartment 3, which it emptied: {msg}"
    );
    // The finding itself. A note naming "compartment(s) 2, 3" — the whole unmatched list
    // rather than the intersection — is the defect review r1 measured.
    assert!(
        !msg.contains("observations on compartment(s) 2, 3,"),
        "the filter must NOT be blamed for entry 2, which it never touched: {msg}"
    );
}

#[test]
fn a_filter_that_emptied_no_declared_compartment_is_not_blamed_at_all() {
    // The same clause and the same dataset as above with one entry removed from the
    // block, which is review r1's case verbatim: declared {1, 2}, observed {1, 3},
    // `ignore = CMT == 3` — the filter removed real observation rows, so the old
    // `n_obs_excluded > 0` gate fired, and not one of them could have matched anything
    // declared.
    //
    // This is also the mutation control the test above cannot be: with entry 3 gone
    // from the block, blaming `filtered_cmts` *without* intersecting it with `unmatched`
    // names compartment 3 — a compartment the model never declares — and the assertion
    // below is what sees it.
    let m = per_cmt_scaling_model("  obs_scale[CMT=1] = 1\n  obs_scale[CMT=2] = 1000");
    let pop = filtered_population(&m, OBS_CMT1_AND_3, "CMT == 3");
    let excl = pop
        .exclusions
        .as_ref()
        .expect("a filtered read records its exclusions");
    assert!(
        excl.n_obs_excluded > 0 && excl.obs_cmts_excluded == vec![3],
        "the clause must really have removed observation rows, all on compartment 3, \
         or this is the no-op-filter test again: {excl:?}"
    );

    let msgs = unmatched_messages(&m, &pop);
    assert_eq!(msgs.len(), 1, "the entry is still dead, got {msgs:?}");
    assert!(
        msgs[0].contains("declares compartment(s) 2 "),
        "entry 2 is the dead one: {}",
        msgs[0]
    );
    assert!(
        !msgs[0].contains("data_selection"),
        "a filter that emptied no declared compartment must not be offered as the \
         cause: {}",
        msgs[0]
    );
}

#[test]
fn the_missing_column_advice_stays_withdrawn_on_a_filtered_read_of_a_column_bearing_file() {
    // A regression pin on a gate that has been wrong twice, kept because it is the case
    // that breaks *both* discarded spellings while looking innocuous.
    //
    // History, since the name changed with the gate. The first gate tested
    // `observed == {1}`; this fixture defeats it, because after `ignore = CMT == 3` the
    // observed set really is `{1}` on a file whose column plainly carries a 3, and the
    // message read
    //
    //   "... A `[data_selection]` clause removed every scored observation on
    //    compartment(s) 3 ... The usual cause is ... a missing or mis-mapped CMT column
    //    keys every observation to compartment 1. Check the CMT column ..."
    //
    // — two sentences of one message contradicting each other. The second gate widened
    // to `observed ∪ excluded`, which this fixture passes.
    //
    // Review r2 then showed the second gate was the same mistake one step out: both are
    // *symptoms* of a defaulted `CMT` column rather than the column's state, and a
    // present, correct, all-`1` column satisfies either. The gate now reads the reader's
    // own `W_CMT_DEFAULTED`, and the pair that pins it lives in
    // `the_missing_column_advice_is_offered_only_when_the_data_reads_column_less`.
    //
    // What this test still pins, and why it survives its own rationale: a future gate
    // that goes back to keying on the observed set fails *here* and passes there, because
    // here `observed == {1}` arises from filtering rather than from the data.
    const COLUMN_ADVICE: &str = "missing or mis-mapped CMT column";
    let m = per_cmt_scaling_model(
        "  obs_scale[CMT=1] = 1\n  obs_scale[CMT=2] = 1000\n  obs_scale[CMT=3] = 500",
    );
    let pop = filtered_population(&m, OBS_CMT1_AND_3, "CMT == 3");
    let excl = pop
        .exclusions
        .as_ref()
        .expect("a filtered read records its exclusions");
    assert_eq!(
        excl.obs_cmts_excluded,
        vec![3],
        "the clause must have emptied compartment 3: {excl:?}"
    );
    assert!(
        pop.subjects
            .iter()
            .flat_map(|s| s.obs_cmts.iter())
            .all(|c| *c == 1),
        "post-filter the population must observe only compartment 1, or the gate on \
         `observed` alone would not have fired and this test proves nothing"
    );

    let msgs = unmatched_messages(&m, &pop);
    assert_eq!(msgs.len(), 1, "expected one finding, got {msgs:?}");
    let msg = &msgs[0];
    assert!(
        msg.contains("(observed: 1)"),
        "the observed set really is {{1}} here: {msg}"
    );
    assert!(
        msg.contains("observations on compartment(s) 3, so check"),
        "the filter note must still name compartment 3: {msg}"
    );
    assert!(
        !msg.contains(COLUMN_ADVICE),
        "the file has a CMT column and the reader did not default it, so the column \
         hypothesis must not be offered: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Two live channels on one model (#1456 review r1, finding 5)
// ---------------------------------------------------------------------------

#[test]
fn a_model_with_two_live_per_cmt_channels_reports_both() {
    // Every fixture above is single-channel by construction, which is what makes a
    // lost channel visible — but it left the real multi-analyte model unfixtured: a
    // per-CMT `[scaling]` block and a per-CMT `[error_model]` on the same model, both
    // declaring an endpoint the dataset does not carry.
    //
    // The property is that the shared walk reports the channels *independently*: two
    // findings, each with its own syntax and its own `[block]`, not one merged report
    // and not the first channel only. A `break` in the loop, or a `diags.push` moved
    // outside it, passes every single-channel test above.
    let m = two_live_per_cmt_channels_model();
    assert_live_channels(
        &m,
        &[CmtConsumer::PerCmtScaling, CmtConsumer::PerCmtErrorModel],
    );

    let diags: Vec<_> = warnings_of(&m, &population(OBS_CMT1_ONLY))
        .into_iter()
        .filter(|d| d.code == CODE)
        .collect();
    assert_eq!(
        diags.len(),
        2,
        "one finding per live channel, got {diags:?}"
    );

    let scaling = diags
        .iter()
        .find(|d| d.message.contains("obs_scale[CMT=N]"))
        .unwrap_or_else(|| panic!("no scaling finding in {diags:?}"));
    let errmodel = diags
        .iter()
        .find(|d| d.message.contains("[error_model] CMT=N:"))
        .unwrap_or_else(|| panic!("no error-model finding in {diags:?}"));
    // The `[block]` each finding is filed under, which `ferx check` prints as its
    // location. Before review r1 this came from a `_` wildcard over `CmtConsumer` in
    // the caller; it now travels with the syntax, and this is the only test that sees
    // both values at once.
    assert_eq!(scaling.block.as_deref(), Some("scaling"));
    assert_eq!(errmodel.block.as_deref(), Some("error_model"));
    for d in &diags {
        assert!(
            d.message.contains("declares compartment(s) 2 ") && d.message.contains("(observed: 1)"),
            "each finding names its own dead entry: {}",
            d.message
        );
    }
}

// ---------------------------------------------------------------------------
// The other entry points (#1456 review r1, finding 2)
// ---------------------------------------------------------------------------

#[test]
fn predict_and_simulate_report_it_too() {
    // Decided rather than inherited. `check_model_data_warnings` is item 3 of
    // `postfit::non_fit_diagnostics`, so this warning reaches `predict()` and
    // `simulate()` as well as `fit()` — that was true when the check landed and
    // untested, which is the finding.
    //
    // It is kept, for the reason that function's own doc gives: the bundle carries
    // findings about *the model and the data*, and this is one — the declared map and
    // the observed compartments are both properties of what the caller handed in, with
    // no fit and no optimizer in the statement. Dropping a code there because it reads
    // oddly outside a fit is the per-entry-point filtering #1280 was filed about. It is
    // also true on those paths in the strict sense: `predict()` dispatches through the
    // same per-CMT map, so an entry no row matches is exactly as inert there.
    //
    // What made it read wrong outside a fit was the advice half, and that is finding 1:
    // the gated message states the fact and, on a population that is legitimately
    // partial, no longer tells the caller their CMT column is broken.
    let m = per_cmt_scaling_model("  obs_scale[CMT=1] = 1\n  obs_scale[CMT=2] = 1000");
    let pop = population(OBS_CMT1_ONLY);

    let predicted = predict_diag(&m, &pop, &m.default_params).unwrap();
    assert!(
        predicted.warnings.iter().any(|w| w.contains(CODE)),
        "predict() must carry the finding: {:?}",
        predicted.warnings
    );

    let simulated = simulate_with_options_diag(
        &m,
        &pop,
        &m.default_params,
        1,
        &SimulateOptions {
            seed: Some(7),
            ..Default::default()
        },
    )
    .expect("simulate");
    assert!(
        simulated.warnings.iter().any(|w| w.contains(CODE)),
        "simulate() must carry the finding: {:?}",
        simulated.warnings
    );

    // The control both halves need: a model whose entries are all matched must be
    // silent on these paths too, or the assertions above are satisfied by a path that
    // reports the bundle unconditionally.
    let matched = population(OBS_BOTH_CMTS);
    assert!(
        !predict_diag(&m, &matched, &m.default_params)
            .unwrap()
            .warnings
            .iter()
            .any(|w| w.contains(CODE)),
        "predict() must stay silent when every entry is exercised"
    );
    assert!(
        !simulate_with_options_diag(
            &m,
            &matched,
            &m.default_params,
            1,
            &SimulateOptions {
                seed: Some(7),
                ..Default::default()
            },
        )
        .expect("simulate")
        .warnings
        .iter()
        .any(|w| w.contains(CODE)),
        "simulate() must stay silent when every entry is exercised"
    );
}

// ---------------------------------------------------------------------------
// What the filter note may claim (#1456 review r2, findings 2, 3 and 4)
// ---------------------------------------------------------------------------

/// `EVID=0, MDV=0` rows whose `DV` is `.` — classified as observations where the filter
/// runs, then dropped by `MissingDvPolicy::Skip` before they become any.
const DV_DOT_ON_CMT2: &str = "\
ID,TIME,DV,EVID,AMT,CMT,MDV
1,0,.,1,100,1,1
1,1,8.0,0,.,1,0
1,4,5.0,0,.,1,0
1,1,.,0,.,2,0
1,4,.,0,.,2,0
";

/// Observations on 1, 2 and 3 — the fixture that can put **two** compartments in `blamed`.
const OBS_CMT_1_2_3: &str = "\
ID,TIME,DV,EVID,AMT,CMT,MDV
1,0,.,1,100,1,1
1,1,8.0,0,.,1,0
1,1,0.9,0,.,2,0
1,4,0.6,0,.,3,0
2,0,.,1,100,1,1
2,1,7.5,0,.,1,0
2,1,0.8,0,.,2,0
2,4,0.5,0,.,3,0
";

/// `filtered_population` with more than one clause.
fn filtered_population_multi(model: &CompiledModel, body: &str, ignores: &[&str]) -> Population {
    let f = csv(body);
    let owned: Vec<String> = ignores.iter().map(|s| s.to_string()).collect();
    let filter = SelectionFilter::from_opts(&owned, &[], &[]).expect("clauses parse");
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

#[test]
fn the_filter_note_does_not_claim_the_removed_rows_would_have_been_scored() {
    // Review r2, finding 2. `obs_cmts_excluded` is filled where the filter runs, which is
    // before the reader decides whether a row becomes a scored observation. A row with
    // `EVID=0, MDV=0` and `DV = .` is counted there and then skipped, so the set can name
    // a compartment that carries no observation even unfiltered.
    //
    // The control is the unfiltered read of the same file, which is the refutation the
    // old wording printed next to itself: entry 2 is dead *without* the filter, so
    // "that entry is exercised by the unfiltered dataset" was false exactly where it was
    // load-bearing.
    let m = per_cmt_scaling_model("  obs_scale[CMT=1] = 1\n  obs_scale[CMT=2] = 1000");
    let unfiltered = unmatched_messages(&m, &population(DV_DOT_ON_CMT2));
    assert_eq!(
        unfiltered.len(),
        1,
        "expected one finding, got {unfiltered:?}"
    );
    assert!(
        unfiltered[0].contains("declares compartment(s) 2 "),
        "entry 2 must be dead even unfiltered, or this proves nothing: {}",
        unfiltered[0]
    );

    let pop = filtered_population(&m, DV_DOT_ON_CMT2, "CMT == 2");
    let excl = pop
        .exclusions
        .as_ref()
        .expect("a filtered read records its exclusions");
    assert_eq!(
        excl.obs_cmts_excluded,
        vec![2],
        "the clause must put compartment 2 in the excluded set — the whole point is that \
         it lands there although no observation was ever on it: {excl:?}"
    );

    let msgs = unmatched_messages(&m, &pop);
    assert_eq!(msgs.len(), 1, "expected one finding, got {msgs:?}");
    assert!(
        msgs[0].contains("removed record(s) the reader classed as observations"),
        "the note must say what the field actually records: {}",
        msgs[0]
    );
    assert!(
        !msgs[0].contains("is exercised by the unfiltered dataset")
            && !msgs[0].contains("are exercised by the unfiltered dataset"),
        "and must not assert the entry is live, which the unfiltered read above \
         disproves: {}",
        msgs[0]
    );
}

#[test]
fn the_filter_note_agrees_in_number_with_the_compartments_it_names() {
    // Review r2, finding 3, and the first defect found by the rule this PR adds to
    // CLAUDE.md: deleting the plural spelling killed no test, because both existing
    // filter tests put exactly one compartment in `blamed`. The old text read
    // "so those entry is exercised".
    //
    // Straddle: one clause and two clauses on the same file and the same block, so the
    // only thing that changes is how many compartments are blamed.
    let m = per_cmt_scaling_model(
        "  obs_scale[CMT=1] = 1\n  obs_scale[CMT=2] = 1000\n  obs_scale[CMT=3] = 500",
    );

    let one = filtered_population_multi(&m, OBS_CMT_1_2_3, &["CMT == 2"]);
    assert_eq!(
        one.exclusions.as_ref().unwrap().obs_cmts_excluded,
        vec![2],
        "singular side must blame exactly one compartment"
    );
    let singular = unmatched_messages(&m, &one);
    assert_eq!(singular.len(), 1, "expected one finding, got {singular:?}");
    assert!(
        singular[0].contains("compartment(s) 2, so check") && singular[0].contains("that entry"),
        "singular: {}",
        singular[0]
    );

    let two = filtered_population_multi(&m, OBS_CMT_1_2_3, &["CMT == 2", "CMT == 3"]);
    assert_eq!(
        two.exclusions.as_ref().unwrap().obs_cmts_excluded,
        vec![2, 3],
        "plural side must blame two compartments, or the branch is not reached"
    );
    let plural = unmatched_messages(&m, &two);
    assert_eq!(plural.len(), 1, "expected one finding, got {plural:?}");
    assert!(
        plural[0].contains("compartment(s) 2, 3, so check") && plural[0].contains("those entries"),
        "plural: {}",
        plural[0]
    );
    // The defect itself, spelled out so a regression cannot pass by being merely
    // different: no mixed number anywhere in either message.
    for msg in [&singular[0], &plural[0]] {
        assert!(
            !msg.contains("those entry") && !msg.contains("that entries"),
            "number must agree: {msg}"
        );
    }
}

#[test]
fn a_finding_the_filter_fully_explains_does_not_also_say_to_delete_the_entries() {
    // Review r2, finding 4. When every unmatched entry is one the filter emptied, the
    // finding is already fully accounted for — and the advice half ends "or delete them",
    // contradicting the sentence before it, which says to check the unfiltered dataset
    // first. `advice` is loop-invariant and cannot see `blamed`, so the suppression is
    // decided per channel.
    //
    // Straddle: the same file and the same clause against two blocks. With `[CMT=2]` and
    // `[CMT=3]` declared, the filter explains both and the advice is dropped. Add
    // `[CMT=4]`, which the dataset never carries and no clause touches, and the advice
    // returns — because now something is unexplained.
    let filtered = |declared: &str| {
        let m = per_cmt_scaling_model(declared);
        let pop = filtered_population_multi(&m, OBS_CMT_1_2_3, &["CMT == 2", "CMT == 3"]);
        let msgs = unmatched_messages(&m, &pop);
        assert_eq!(msgs.len(), 1, "expected one finding, got {msgs:?}");
        msgs[0].clone()
    };

    let fully =
        filtered("  obs_scale[CMT=1] = 1\n  obs_scale[CMT=2] = 1000\n  obs_scale[CMT=3] = 500");
    assert!(
        fully.contains("declares compartment(s) 2, 3 ")
            && fully.contains("compartment(s) 2, 3, so check"),
        "the filter must account for every unmatched entry on this side: {fully}"
    );
    assert!(
        !fully.contains("delete them") && !fully.contains("delete the unused entries"),
        "a fully explained finding must not also tell the user to delete the entries: \
         {fully}"
    );

    let partly = filtered(
        "  obs_scale[CMT=1] = 1\n  obs_scale[CMT=2] = 1000\n  obs_scale[CMT=3] = 500\n  obs_scale[CMT=4] = 250",
    );
    assert!(
        partly.contains("declares compartment(s) 2, 3, 4 ")
            && partly.contains("compartment(s) 2, 3, so check"),
        "the other side must leave entry 4 unexplained: {partly}"
    );
    assert!(
        partly.contains("delete them"),
        "and must still give the advice, or the suppression is unconditional: {partly}"
    );
}
