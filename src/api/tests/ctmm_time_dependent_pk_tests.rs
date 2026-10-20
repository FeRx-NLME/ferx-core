//! #1918 — a time-inhomogeneous (drug-driven) CTMM generator reads model states solved
//! under one `TIME = 0` `$PK` snapshot, so a `TIME`-dependent CL / V / `init()` that can
//! reach those states is refused rather than silently mis-fit. The CTMM arm of #1853.
//!
//! The gate is `check_ctmm_time_dependent_pk`, a function of the **model** alone, so these
//! are Tier 1. `fit()` and `ferx check` surfacing it is `tests/ctmm_time_dependent_pk.rs`.
//!
//! **The input space, enumerated:** `{ reads TIME } × { reaches a generator state through
//! the RHS, through an init() seed, through both, through neither } × { inhomogeneous
//! CTMM, homogeneous CTMM }`. The first three cells with an inhomogeneous CTMM refuse;
//! every other cell returns `None` and has a test below. The slice is a **union** over
//! every state the generator reads, so a generator reading two states has its own cell: a
//! parameter reaching only the second state must refuse.

use super::*;
use crate::parser::model_parser::parse_model_string;

/// A two-state oral ODE model with a CTMM on CMT 5. `[individual_parameters]` is `ip`,
/// the `[odes]` block is `odes`, and the `awake -> asleep` intensity is `driver`.
fn ctmm_model(ip: &str, odes: &str, driver: &str) -> CompiledModel {
    let m = format!(
        r"
[parameters]
  theta TVCL(1.0, FIX)
  theta TVV(10.0, FIX)
  theta TVKA(1.0, FIX)
  theta TVKT(0.05, FIX)
  theta TVBASE(1.0, FIX)
  theta LQ01(-0.7, -6.0, 3.0)
  theta LQ10(-1.2, -6.0, 3.0)
  theta SLOPE(0.1, -5.0, 5.0)
  sigma PROP ~ 0.05 (sd)
[individual_parameters]
{ip}
[structural_model]
  ode(obs_cmt=central, states=[depot, central, effect])
[odes]
{odes}
[error_model]
  DV ~ proportional(PROP)
[markov_model]
  type   = ctmm
  cmt    = 5
  states = [awake=0, asleep=1]
  transition awake  -> asleep = {driver}
  transition asleep -> awake  = exp(LQ10)
"
    );
    parse_model_string(&m).unwrap_or_else(|e| panic!("model compiles: {e}"))
}

/// `effect` is a separate state that `central` never reads, so a parameter feeding only
/// `effect` cannot reach a generator that reads only `central`.
const PLAIN_ODES: &str = "  d/dt(depot)   = -KA * depot\n  \
                          d/dt(central) =  KA * depot - (CL/V) * central\n  \
                          d/dt(effect)  = -KE * effect\n";

const TIME_FREE_IP: &str = "  CL = TVCL\n  V = TVV\n  KA = TVKA\n  KE = TVKA\n";

const TIME_CL_IP: &str = "  CL = TVCL * (1.0 + TVKT * TIME)\n  V = TVV\n  KA = TVKA\n  KE = TVKA\n";

const TIME_KE_IP: &str = "  CL = TVCL\n  V = TVV\n  KA = TVKA\n  KE = TVKA * (1.0 + TVKT * TIME)\n";

/// Reads `central` only.
const CENTRAL_DRIVER: &str = "exp(LQ01 + SLOPE * (central / V))";

/// Reads no state: a time-homogeneous CTMM.
const NO_STATE_DRIVER: &str = "exp(LQ01)";

/// Same shape check as the #1853 suite: no run of literal whitespace from a mis-escaped
/// `format!` continuation, one line, a printable length.
fn assert_message_is_well_formed(label: &str, msg: &str) {
    assert!(
        !msg.contains("  "),
        "{label}: literal whitespace run: {msg:?}"
    );
    assert!(
        !msg.contains('\n') && !msg.contains('\t'),
        "{label}: must be one line: {msg:?}"
    );
    assert!(
        (80..=900).contains(&msg.len()),
        "{label}: length {} outside 80..=900",
        msg.len()
    );
}

/// A `TIME`-dependent `CL` read by the RHS of the state the generator reads is refused, and
/// every sentence of the message is asserted — the endpoint, the mechanism, the parameter
/// and site, the reason, and the remedy.
#[test]
fn a_time_dependent_disposition_on_a_drug_driven_ctmm_is_refused() {
    let model = ctmm_model(TIME_CL_IP, PLAIN_ODES, CENTRAL_DRIVER);
    let msg = check_ctmm_time_dependent_pk(&model)
        .expect("a TIME-dependent CL on a drug-driven CTMM must be refused");
    assert_message_is_well_formed("the RHS refusal", &msg);
    for want in [
        "[markov_model] on CMT=5",
        "reads model states solved with the individual parameters frozen at their TIME=0 \
         values",
        "`CL` reads the `TIME` built-in and the [odes] RHS reads it",
        "not yet supported (#1918)",
        "the CTMM pass keeps the TIME=0 value",
        "Hold `CL` constant in TIME for now",
    ] {
        assert!(msg.contains(want), "missing {want:?} in: {msg}");
    }
}

/// The `init()` cell: `BASE` reads `TIME` and is read only by `init(central)`.
#[test]
fn a_time_dependent_init_seed_on_a_drug_driven_ctmm_is_refused() {
    let model = ctmm_model(
        "  CL = TVCL\n  V = TVV\n  KA = TVKA\n  KE = TVKA\n  BASE = TVBASE * (1.0 + TVKT * TIME)\n",
        &format!("  init(central) = BASE\n{PLAIN_ODES}"),
        CENTRAL_DRIVER,
    );
    let msg = check_ctmm_time_dependent_pk(&model).expect("a TIME-dependent init seed refuses");
    assert_message_is_well_formed("the init refusal", &msg);
    assert!(
        msg.contains("`BASE` reads the `TIME` built-in and an [odes] init(...) seed reads it"),
        "{msg}"
    );
}

/// The both-sites cell.
#[test]
fn a_parameter_read_by_both_the_rhs_and_the_init_names_both() {
    let model = ctmm_model(
        TIME_CL_IP,
        &format!("  init(central) = CL\n{PLAIN_ODES}"),
        CENTRAL_DRIVER,
    );
    let msg = check_ctmm_time_dependent_pk(&model).expect("both sites refuse");
    assert_message_is_well_formed("the both-sites refusal", &msg);
    assert!(
        msg.contains(
            "`CL` reads the `TIME` built-in and the [odes] RHS and an init(...) seed read it"
        ),
        "{msg}"
    );
}

/// **The neither cell, with its straddle.** `KE` reads `TIME` but feeds only `effect`,
/// which the `central`-reading generator never sees — accepted. The same model whose
/// generator reads `effect` must refuse, so the accept is the slice and not a gate stuck on
/// `None`.
#[test]
fn a_time_dependent_parameter_the_generator_cannot_see_is_not_refused() {
    let unseen = ctmm_model(TIME_KE_IP, PLAIN_ODES, CENTRAL_DRIVER);
    assert_eq!(
        check_ctmm_time_dependent_pk(&unseen),
        None,
        "KE feeds only `effect`, which the generator does not read"
    );
    let seen = ctmm_model(TIME_KE_IP, PLAIN_ODES, "exp(LQ01 + SLOPE * effect)");
    let msg = check_ctmm_time_dependent_pk(&seen).expect("a generator reading `effect` refuses");
    assert!(msg.contains("`KE` reads the `TIME` built-in"), "{msg}");
}

/// **The slice is a union over the generator's states.** A generator reading `central`
/// *and* `effect`, with `TIME` reaching only `effect`: a slice taken from the first state
/// alone would accept it.
#[test]
fn every_state_the_generator_reads_is_in_the_slice() {
    let model = ctmm_model(
        TIME_KE_IP,
        PLAIN_ODES,
        "exp(LQ01 + SLOPE * (central / V) + SLOPE * effect)",
    );
    let msg = check_ctmm_time_dependent_pk(&model)
        .expect("TIME reaching the second generator state must refuse");
    assert!(msg.contains("`KE` reads the `TIME` built-in"), "{msg}");
}

/// A **time-homogeneous** CTMM never solves the `[odes]` system, so a `TIME`-dependent
/// disposition on the same model cannot reach it. One intensity apart from the refusing arm.
#[test]
fn a_homogeneous_ctmm_is_not_refused() {
    let model = ctmm_model(TIME_CL_IP, PLAIN_ODES, NO_STATE_DRIVER);
    assert_eq!(check_ctmm_time_dependent_pk(&model), None);
    // The diag wrapper — what `fit()` calls — agrees.
    assert!(check_survival_time_dependent_pk_diags(&model).is_empty());
}

/// The time-free baseline is accepted.
#[test]
fn a_time_free_drug_driven_ctmm_is_not_refused() {
    let model = ctmm_model(TIME_FREE_IP, PLAIN_ODES, CENTRAL_DRIVER);
    assert_eq!(check_ctmm_time_dependent_pk(&model), None);
}

/// The diagnostic `fit()` and `ferx check` report: `E_ENDPOINT_TIME_DEPENDENT_PK` on block
/// `markov_model`, exactly once, carrying the check's text.
#[test]
fn the_diagnostic_is_coded_on_the_markov_model_block() {
    let model = ctmm_model(TIME_CL_IP, PLAIN_ODES, CENTRAL_DRIVER);
    let diags = check_survival_time_dependent_pk_diags(&model);
    assert_eq!(diags.len(), 1, "{diags:?}");
    assert_eq!(diags[0].code, "E_ENDPOINT_TIME_DEPENDENT_PK");
    assert!(diags[0].is_error());
    assert_eq!(diags[0].block.as_deref(), Some("markov_model"));
    assert_eq!(
        Some(diags[0].message.clone()),
        check_ctmm_time_dependent_pk(&model)
    );
}

/// Missing analysis metadata fails closed, each component independently, from a model that
/// is accepted intact; a homogeneous CTMM still exits before the metadata is asked for.
#[test]
fn missing_analysis_metadata_is_refused_rather_than_passed() {
    use crate::parser::model_parser::IndivParamPartials;

    let base = ctmm_model(TIME_FREE_IP, PLAIN_ODES, CENTRAL_DRIVER);
    assert_eq!(check_ctmm_time_dependent_pk(&base), None, "straddle");

    let mut no_ip = ctmm_model(TIME_FREE_IP, PLAIN_ODES, CENTRAL_DRIVER);
    no_ip.indiv_param_partials = IndivParamPartials::empty();
    let msg = check_ctmm_time_dependent_pk(&no_ip).expect("no $PK program must not pass");
    assert_message_is_well_formed("the missing-$PK-program refusal", &msg);
    for want in [
        "[markov_model] on CMT=5",
        "drug-driven (time-inhomogeneous) CTMM generator",
        "no compiled `[individual_parameters]` program",
        "cannot be determined (#1918)",
        "IndivParamPartials::empty()",
    ] {
        assert!(msg.contains(want), "missing {want:?} in: {msg}");
    }

    let mut no_rhs = ctmm_model(TIME_FREE_IP, PLAIN_ODES, CENTRAL_DRIVER);
    no_rhs.ode_spec.as_mut().expect("ODE model").rhs_program = None;
    let msg = check_ctmm_time_dependent_pk(&no_rhs).expect("no [odes] program must not pass");
    assert_message_is_well_formed("the missing-[odes]-program refusal", &msg);
    for want in [
        "no compiled `[odes]` program",
        "make the transition intensities state-free",
    ] {
        assert!(msg.contains(want), "missing {want:?} in: {msg}");
    }

    let mut homogeneous = ctmm_model(TIME_FREE_IP, PLAIN_ODES, NO_STATE_DRIVER);
    homogeneous.indiv_param_partials = IndivParamPartials::empty();
    assert_eq!(
        check_ctmm_time_dependent_pk(&homogeneous),
        None,
        "a homogeneous CTMM has no freeze to be undecided about"
    );
}

/// **The #1575 cell.** `ALAG1` reads `TIME`, but the CTMM pass reads every dose-record
/// quantity at its own dose record, so nothing about it is frozen. The block does read
/// `TIME` (asserted), so the `None` is the slice and not the fast exit.
#[test]
fn a_time_dependent_dose_record_quantity_is_not_refused() {
    let model = ctmm_model(
        "  CL = TVCL\n  V = TVV\n  KA = TVKA\n  KE = TVKA\n  ALAG1 = TVBASE + TVKT * TIME\n",
        PLAIN_ODES,
        CENTRAL_DRIVER,
    );
    assert!(
        crate::parser::model_parser::compiled_model_uses_time_builtin(&model),
        "straddle: the block must read TIME"
    );
    assert_eq!(check_ctmm_time_dependent_pk(&model), None);
}
