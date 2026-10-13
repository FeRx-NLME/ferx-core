//! #1853 — an ODE-accumulated hazard integrates its disposition at one `TIME = 0`
//! snapshot, so a `TIME`-dependent CL / V / `init()` is refused rather than silently
//! mis-fit. The `TIME` axis of #741's freeze, which refuses the covariate axis.
//!
//! The gate is `check_survival_time_dependent_pk`, which is a function of the **model**
//! alone — no population, no solve — so these are Tier 1. The end-to-end claim (every
//! public entry point surfaces it as an `Err`) is `tests/survival_time_dependent_pk.rs`.
//!
//! **The input space, enumerated** (the message is as much the deliverable as the
//! predicate, and a claim no test can kill is a claim nobody has checked):
//! `{ reads TIME } × { reaches the accumulator through the RHS, through an init() seed,
//! through both, through neither } × { OdeAccumulated hazard, analytic hazard }`. The
//! cells that must refuse are the first three with an `OdeAccumulated` hazard; every
//! other cell must return `None`, and each has a test below — including the four that
//! look like they should refuse and must not:
//!
//! * a `TIME`-reading **dose-record** quantity (#1575 reads those per dose record);
//! * a `TIME`-reading **`[odes]` RHS** (the integrator clock honours that, #610);
//! * a `TIME`-reading parameter the **hazard does not depend on** (`hazard = H0` with a
//!   time-dependent `CL`), and an `init()` of a state it does not depend on;
//! * no `OdeAccumulated` endpoint at all.
//!
//! The "reaches" half is a dependency slice, not a syntax scan, so two further cells have
//! their own tests: a parameter read **only by an `[odes]` `if` condition** must refuse
//! (control dependence), and on a competing-risks model the message must name the
//! endpoint whose own hazard is affected rather than the lowest CMT.

use super::*;
use crate::parser::model_parser::{compiled_model_uses_time_builtin, parse_model_string};

/// A joint PK-TTE ODE model: `[individual_parameters]` is `ip`, the `[odes]` block is
/// `odes`. Two states, so the injected `d/dt(__chz_3)` accumulator makes the hazard CMT 3.
fn joint_model(ip: &str, odes: &str) -> CompiledModel {
    joint_model_with_hazard(ip, odes, "H0 * exp(BETA * (central / V))")
}

/// [`joint_model`] with the `hazard =` expression spelled out — the arms that need an
/// accumulator the disposition does *not* reach.
fn joint_model_with_hazard(ip: &str, odes: &str, hazard: &str) -> CompiledModel {
    let m = format!(
        r"
[parameters]
  theta TVCL(1.0, FIX)
  theta TVV(10.0, FIX)
  theta TVKA(1.0, FIX)
  theta TVKT(0.05, FIX)
  theta TVMTT(1.0, FIX)
  theta TVN(3.0, FIX)
  theta TVLAG(0.5, FIX)
  theta TVH0(0.02, FIX)
  theta TVBETA(0.1, FIX)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.01
[individual_parameters]
{ip}
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
{odes}
[scaling]
  obs_scale = V
[event_model]
  cmt    = 3
  hazard = {hazard}
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
"
    );
    parse_model_string(&m).unwrap_or_else(|e| panic!("model compiles: {e}"))
}

/// The plain two-state disposition, with no absorption kernel.
const PLAIN_ODES: &str = "  d/dt(depot)   = -KA * depot\n  \
                          d/dt(central) =  KA * depot - (CL/V) * central\n";

/// The time-free `[individual_parameters]` baseline every refusing arm perturbs.
const TIME_FREE_IP: &str = "  CL   = TVCL * exp(ETA_CL)\n  \
                            V    = TVV\n  \
                            KA   = TVKA\n  \
                            H0   = TVH0\n  \
                            BETA = TVBETA\n";

/// A `TIME`-dependent **disposition** parameter the `[odes]` RHS reads is refused, and
/// the message names the parameter, the endpoint, the site and the remedy.
///
/// Every substring asserted here belongs to a different sentence of the message, so
/// deleting any one of them reddens this test — including the advice half, which is the
/// only part a user acts on.
#[test]
fn a_time_dependent_disposition_on_an_ode_hazard_is_refused() {
    let model = joint_model(
        "  CL   = TVCL * (1.0 + TVKT * TIME) * exp(ETA_CL)\n  \
         V    = TVV\n  \
         KA   = TVKA\n  \
         H0   = TVH0\n  \
         BETA = TVBETA\n",
        PLAIN_ODES,
    );
    let msg = check_survival_time_dependent_pk(&model)
        .expect("a TIME-dependent CL on an ODE-accumulated hazard must be refused");
    for want in [
        "CMT=3",
        "frozen at their TIME=0 values",
        "`CL` reads the `TIME` built-in and the [odes] RHS reads it",
        "A TIME-dependent disposition on a drug-driven hazard is not yet supported (#1853)",
        "evaluate `[individual_parameters]` at every record while the hazard pass keeps the \
         TIME=0 value",
        "Hold `CL` constant in TIME for now",
    ] {
        assert!(msg.contains(want), "missing {want:?} in: {msg}");
    }
}

/// The `init()` cell: `BASE` reads `TIME` and is read **only** by the `init(central)`
/// seed — the RHS never mentions it — so the gate still fires and the message points at
/// the init rather than the RHS. Without the init half this model fits silently with its
/// seed stuck at `TIME = 0`.
#[test]
fn a_time_dependent_init_seed_on_an_ode_hazard_is_refused() {
    let model = joint_model(
        "  CL   = TVCL * exp(ETA_CL)\n  \
         V    = TVV\n  \
         KA   = TVKA\n  \
         BASE = TVMTT * (1.0 + TVKT * TIME)\n  \
         H0   = TVH0\n  \
         BETA = TVBETA\n",
        "  init(central) = BASE\n  d/dt(depot)   = -KA * depot\n  \
         d/dt(central) =  KA * depot - (CL/V) * central\n",
    );
    let msg = check_survival_time_dependent_pk(&model)
        .expect("a TIME-dependent init(...) seed must be refused");
    assert!(
        msg.contains("`BASE` reads the `TIME` built-in and an [odes] init(...) seed reads it"),
        "the message must name the init site: {msg}"
    );
}

/// The both-sites cell: `CL` is read by the RHS *and* by the `init()` seed, so the
/// message says both. Three cells of `(rhs, init)` are reachable and each renders its own
/// sentence; this is the one a two-arm `if` would get wrong.
#[test]
fn a_parameter_read_by_both_the_rhs_and_the_init_names_both() {
    let model = joint_model(
        "  CL   = TVCL * (1.0 + TVKT * TIME) * exp(ETA_CL)\n  \
         V    = TVV\n  \
         KA   = TVKA\n  \
         H0   = TVH0\n  \
         BETA = TVBETA\n",
        "  init(central) = CL\n  d/dt(depot)   = -KA * depot\n  \
         d/dt(central) =  KA * depot - (CL/V) * central\n",
    );
    let msg = check_survival_time_dependent_pk(&model)
        .expect("a TIME-dependent CL read by both sites must be refused");
    assert!(
        msg.contains(
            "`CL` reads the `TIME` built-in and the [odes] RHS and an init(...) seed read it"
        ),
        "the message must name both sites: {msg}"
    );
}

/// **The #1575 cell, which must keep fitting.** `MTT` and `ALAG1` read `TIME`, but the
/// engines read every dose-record quantity at its own dose record, so nothing about them
/// is frozen. A coarse gate — the whole-block `uses_time_builtin` flag, which is `true`
/// here — would refuse this model and undo #1575.
///
/// The straddle is asserted, not assumed: the block *does* read `TIME` (so the fast exit
/// is not what returns `None`) and the model *does* carry an ODE-accumulated hazard.
#[test]
fn a_time_dependent_dose_record_quantity_is_not_refused() {
    let model = joint_model(
        "  CL    = TVCL * exp(ETA_CL)\n  \
         V     = TVV\n  \
         KA    = TVKA\n  \
         MTT   = TVMTT * (1.0 + TVKT * TIME)\n  \
         NTR   = TVN\n  \
         ALAG1 = TVLAG + TVKT * TIME\n  \
         H0    = TVH0\n  \
         BETA  = TVBETA\n",
        "  d/dt(depot)   = transit(n=NTR, mtt=MTT) - KA * depot\n  \
         d/dt(central) =  KA * depot - (CL/V) * central\n",
    );
    assert!(
        compiled_model_uses_time_builtin(&model),
        "straddle: this block must read TIME, or the fast exit is what returns None"
    );
    assert!(
        model
            .ode_spec
            .as_ref()
            .and_then(|o| o.rhs_program.as_ref())
            .is_some(),
        "straddle: the model must carry a compiled [odes] block"
    );
    assert_eq!(
        check_survival_time_dependent_pk(&model),
        None,
        "a TIME-reading dose-record quantity is read at each dose record (#1575) — \
         refusing it would undo that fix"
    );
}

/// A `TIME`-reading **`[odes]` RHS** with a time-free `$PK` is not refused: the integrator
/// clock resolves `TIME` at each step (#610), so nothing about it is frozen.
///
/// What delivers that is the per-parameter mask, which answers all-`false` here — asserted,
/// because it is the mechanism. The predicate the *routing* gates ask
/// (`pk::model_uses_time_anywhere`) is `true` on this model, so a gate that asked the wide
/// flag and then refused would reject it; this test is the behavioural claim that it is
/// accepted, whichever way the implementation gets there.
#[test]
fn a_time_reading_odes_rhs_is_not_refused() {
    let model = joint_model(
        TIME_FREE_IP,
        "  d/dt(depot)   = -KA * depot\n  \
         d/dt(central) =  KA * depot - (CL/V) * central * (1.0 + 0.001 * TIME)\n",
    );
    assert!(
        crate::pk::model_uses_time_anywhere(&model),
        "straddle: the model must really be non-autonomous"
    );
    assert!(
        !compiled_model_uses_time_builtin(&model),
        "straddle: and the TIME read must be in the [odes] RHS, not in $PK"
    );
    assert_eq!(
        check_survival_time_dependent_pk(&model),
        None,
        "a non-autonomous [odes] RHS is honoured by the integrator clock (#610)"
    );
}

/// No `TIME` anywhere in `$PK` ⇒ nothing to refuse. The baseline the refusing arms differ
/// from in one line.
#[test]
fn a_time_free_joint_model_is_not_refused() {
    let model = joint_model(TIME_FREE_IP, PLAIN_ODES);
    assert!(!compiled_model_uses_time_builtin(&model));
    assert_eq!(check_survival_time_dependent_pk(&model), None);
}

/// **A parameter the accumulator does not depend on is not refused** (round-1 review of
/// PR #1924). `CL` reads `TIME` and the `[odes]` RHS reads `CL`, but this model's hazard
/// integrates a constant `H0`, so `H` and `h` are the same whatever `$PK` snapshot the
/// hazard pass runs under — the model is handled correctly and refusing it would be a
/// false refusal.
///
/// The straddle is the identical model with the hazard reading `central`, which **must**
/// refuse. One line of `[event_model]` apart, so the arms cannot both be satisfied by a
/// gate stuck on either answer.
#[test]
fn a_time_dependent_parameter_the_hazard_does_not_depend_on_is_not_refused() {
    let drug_free = joint_model_with_hazard(
        "  CL   = TVCL * (1.0 + TVKT * TIME) * exp(ETA_CL)\n  \
         V    = TVV\n  \
         KA   = TVKA\n  \
         H0   = TVH0\n  \
         BETA = TVBETA\n",
        PLAIN_ODES,
        "H0 * 1.0",
    );
    assert!(
        compiled_model_uses_time_builtin(&drug_free),
        "straddle: the block must read TIME, or the taint half is what returns None"
    );
    assert_eq!(
        check_survival_time_dependent_pk(&drug_free),
        None,
        "freezing CL cannot move an accumulator that integrates a constant"
    );

    let drug_driven = joint_model(
        "  CL   = TVCL * (1.0 + TVKT * TIME) * exp(ETA_CL)\n  \
         V    = TVV\n  \
         KA   = TVKA\n  \
         H0   = TVH0\n  \
         BETA = TVBETA\n",
        PLAIN_ODES,
    );
    assert!(
        check_survival_time_dependent_pk(&drug_driven).is_some(),
        "the same model whose hazard reads `central` must refuse — otherwise the arm \
         above passes because nothing refuses anything"
    );
}

/// **Control dependence is part of the slice.** `SW` reads `TIME` and is read by nothing
/// but an `if` condition inside the `[odes]` block — no derivative mentions it. But that
/// condition decides which `KEFF` the integration runs under, and `KEFF` is what
/// `central`'s derivative reads, so freezing `SW` at `TIME = 0` pins the system to one
/// branch for the whole timeline while the Gaussian half switches branches at every
/// record.
///
/// A slice that only followed data reads would miss it and accept this model. It is the
/// one fixture whose `TIME`-reading parameter reaches the accumulator *only* through a
/// condition, so dropping `chz_dependency_slice`'s control-dependence arm has to redden
/// here.
#[test]
fn a_time_dependent_parameter_read_only_by_an_odes_condition_is_refused() {
    const M: &str = r"
[parameters]
  theta TVCL(1.0, FIX)
  theta TVV(10.0, FIX)
  theta TVKT(0.05, FIX)
  theta TVH0(0.02, FIX)
  theta TVBETA(0.1, FIX)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.01
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  SW   = TVKT * TIME
  H0   = TVH0
  BETA = TVBETA
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  if (SW > 0.5) {
    KEFF = CL / V
  } else {
    KEFF = 0.5 * CL / V
  }
  d/dt(central) = -KEFF * central
[scaling]
  obs_scale = V
[event_model]
  cmt    = 2
  hazard = H0 * exp(BETA * (central / V))
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
";
    let model = parse_model_string(M).expect("the branching [odes] model compiles");
    // The straddle: `SW` is an individual parameter of this model, `CL` is time-free, and
    // only `SW` reads `TIME` — so a refusal naming anything else would be a different
    // finding.
    assert!(
        model.indiv_param_names.iter().any(|n| n == "SW"),
        "fixture must declare SW: {:?}",
        model.indiv_param_names
    );
    let msg = check_survival_time_dependent_pk(&model)
        .expect("a TIME-dependent [odes] branch condition must be refused");
    assert!(
        msg.contains("`SW` reads the `TIME` built-in and the [odes] RHS reads it"),
        "the message must name SW: {msg}"
    );
}

/// **Control dependence is a prefix, not "all or nothing"** (round-2 review of PR #1924).
/// Branch `k` runs only when conditions `0..k` were all false and its own is true, so a
/// needed write inside it is governed by conditions `0..=k` — and by no later one.
///
/// Three arms, which is the smallest set that pins the rule from both sides. The fixture
/// is one `[odes]` block with a needed intermediate `K` (which `central`'s derivative
/// reads, and the hazard reads `central`) and an unneeded one `KOTHER` (which only
/// `other`'s derivative reads, and nothing reads `other`):
///
/// * **needed write in arm 0, `SW` on arm 1** — `SW` governs only the unneeded write, so
///   it is outside the slice and the model must be **accepted**. Adding every condition
///   whenever anything inside is needed refuses it.
/// * **needed write in arm 1, `SW` on arm 0** — arm 1 runs only if arm 0's condition was
///   false, so `SW` is in the slice and the model must be **refused**. Adding only the
///   needed arm's own condition accepts it, which is a false accept.
/// * **needed write in arm 0, `SW` on arm 0** — the plain case, **refused**, so the first
///   arm cannot pass by the slice simply never containing `SW`.
#[test]
fn control_dependence_reaches_the_conditions_up_to_the_needed_arm_and_no_further() {
    const M: &str = r"
[parameters]
  theta TVCL(1.0, FIX)
  theta TVV(10.0, FIX)
  theta TVKT(0.05, FIX)
  theta TVFLAG(1.0, FIX)
  theta TVH0(0.02, FIX)
  theta TVBETA(0.1, FIX)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.01
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  FLAG = TVFLAG
  SW   = TVKT * TIME
  H0   = TVH0
  BETA = TVBETA
[structural_model]
  ode(obs_cmt=central, states=[central, other])
[odes]
  K      = 1.0
  KOTHER = 0.1
BRANCHES
  d/dt(central) = -K * (CL/V) * central
  d/dt(other)   = -KOTHER * other
[scaling]
  obs_scale = V
[event_model]
  cmt    = 3
  hazard = H0 * exp(BETA * (central / V))
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
";
    // `K` is needed (the hazard reads `central`, whose derivative reads `K`); `KOTHER` is
    // not (nothing needed reads `other`). Only `SW` reads `TIME`.
    let needed_first = "  if (FLAG > 0.0) {\n    K = 2.0\n  } else if (SW > 0.5) {\n                            KOTHER = 0.2\n  }";
    let needed_second = "  if (SW > 0.5) {\n    KOTHER = 0.2\n  } else if (FLAG > 0.0) {\n                             K = 2.0\n  }";
    let needed_own = "  if (SW > 0.5) {\n    K = 2.0\n  } else if (FLAG > 0.0) {\n                          KOTHER = 0.2\n  }";
    for (arm, branches, refuse) in [
        ("needed write in arm 0, SW on arm 1", needed_first, false),
        ("needed write in arm 1, SW on arm 0", needed_second, true),
        ("needed write in arm 0, SW on arm 0", needed_own, true),
    ] {
        let model = parse_model_string(&M.replace("BRANCHES", branches))
            .unwrap_or_else(|e| panic!("{arm}: the branching [odes] model compiles: {e}"));
        // The straddle, per arm: `SW` is this model's only `TIME` reader, so a verdict
        // either way is about `SW` and nothing else.
        assert!(
            compiled_model_uses_time_builtin(&model),
            "{arm}: straddle — the block must read TIME"
        );
        let got = check_survival_time_dependent_pk(&model);
        if refuse {
            let msg = got.unwrap_or_else(|| {
                panic!("{arm}: SW governs a write the accumulator depends on — must refuse")
            });
            assert!(
                msg.contains("`SW` reads the `TIME` built-in"),
                "{arm}: the message must name SW: {msg}"
            );
        } else {
            assert_eq!(
                got, None,
                "{arm}: SW governs only a write nothing needed reads — must be accepted"
            );
        }
    }
}

/// **The message names the endpoint whose own hazard depends on the parameter**, not the
/// lowest-numbered one (round-1 review of PR #1924). Two ODE-accumulated causes: CMT 3
/// integrates a constant, CMT 4 reads `central`, and the `TIME`-dependent `CL` reaches
/// only the second — so the message must say `CMT=4` although 3 sorts first.
#[test]
fn the_message_names_the_endpoint_whose_hazard_depends_on_the_parameter() {
    const M: &str = r"
[parameters]
  theta TVCL(1.0, FIX)
  theta TVV(10.0, FIX)
  theta TVKT(0.05, FIX)
  theta TVH0(0.02, FIX)
  theta TVBETA(0.1, FIX)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.01
[individual_parameters]
  CL   = TVCL * (1.0 + TVKT * TIME) * exp(ETA_CL)
  V    = TVV
  H0   = TVH0
  BETA = TVBETA
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  obs_scale = V
[event_model flat]
  cmt    = 3
  hazard = H0 * 1.0
[event_model exposure]
  cmt    = 4
  hazard = H0 * exp(BETA * (central / V))
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
";
    let model = parse_model_string(M).expect("the competing-risks joint model compiles");
    // The straddle: two ODE-accumulated endpoints, and the flat one sorts first.
    let mut ode_cmts: Vec<usize> = model
        .endpoints
        .iter()
        .filter(|(_, e)| {
            matches!(
                e,
                crate::types::EndpointLikelihood::Tte {
                    hazard: crate::types::HazardSpec::OdeAccumulated { .. },
                    ..
                }
            )
        })
        .map(|(cmt, _)| *cmt)
        .collect();
    ode_cmts.sort_unstable();
    assert_eq!(
        ode_cmts,
        vec![3, 4],
        "fixture must carry two ODE-accumulated endpoints with the flat one first"
    );
    let msg = check_survival_time_dependent_pk(&model).expect("the exposure hazard is frozen");
    assert!(
        msg.contains("CMT=4"),
        "the message must name the endpoint whose hazard reads `central`: {msg}"
    );
    assert!(
        !msg.contains("CMT=3"),
        "CMT 3 integrates a constant and is unaffected: {msg}"
    );
}

/// **Missing analysis metadata fails closed** (round-3 review of PR #1924). Both halves
/// of the predicate read metadata the parser always attaches, but
/// `CompiledModel::indiv_param_partials` is a `pub` field with a public
/// `IndivParamPartials::empty()`, and `OdeSpec`'s fields are `pub` too — so a
/// hand-assembled or hand-edited model can present a working `pk_param_fn` and `[odes]`
/// RHS with the metadata gone. Returning `None` there would hand exactly that model the
/// silent freeze this check exists to stop.
///
/// Each component is removed independently, from a model that is otherwise **accepted**
/// (time-free `$PK`), so the refusal is the missing metadata and nothing else — and each
/// message names which component is missing, since the repair differs.
#[test]
fn missing_analysis_metadata_is_refused_rather_than_passed() {
    use crate::parser::model_parser::IndivParamPartials;

    // The straddle: intact, this model is accepted.
    let base = joint_model(TIME_FREE_IP, PLAIN_ODES);
    assert_eq!(
        check_survival_time_dependent_pk(&base),
        None,
        "straddle: the intact model must be accepted, or these arms prove nothing"
    );

    let mut no_ip = joint_model(TIME_FREE_IP, PLAIN_ODES);
    no_ip.indiv_param_partials = IndivParamPartials::empty();
    let msg = check_survival_time_dependent_pk(&no_ip)
        .expect("a model with no [individual_parameters] program must not be passed");
    for want in [
        "CMT=3",
        "no compiled `[individual_parameters]` program",
        "IndivParamPartials::empty()",
        "#1853",
    ] {
        assert!(msg.contains(want), "missing {want:?} in: {msg}");
    }

    let mut no_rhs = joint_model(TIME_FREE_IP, PLAIN_ODES);
    no_rhs.ode_spec.as_mut().expect("ODE model").rhs_program = None;
    let msg = check_survival_time_dependent_pk(&no_rhs)
        .expect("a model with no compiled [odes] program must not be passed");
    for want in [
        "CMT=3",
        "no compiled `[odes]` program",
        "drop the ODE-accumulated hazard",
        "#1853",
    ] {
        assert!(msg.contains(want), "missing {want:?} in: {msg}");
    }

    // And a model with **no** ODE-accumulated hazard still exits before the metadata is
    // asked for: stripping it there must stay silent, or every analytic-hazard model with
    // a hand-built `CompiledModel` would start erroring.
    let mut analytic = joint_model(TIME_FREE_IP, PLAIN_ODES);
    analytic.endpoints.clear();
    analytic.indiv_param_partials = IndivParamPartials::empty();
    assert_eq!(
        check_survival_time_dependent_pk(&analytic),
        None,
        "no ODE-accumulated endpoint: there is no freeze to be undecided about"
    );
}

/// **A pinned limitation, not a desired property** (round-3 review of PR #1924, tracked as
/// #1931). Both analyses are flow-insensitive with respect to *kills*: they union over
/// every write to a slot and never notice that a later unconditional write discards an
/// earlier one. So a model whose `TIME`-dependent definition is dead — overwritten before
/// anything reads it — is refused although its effective `$PK` is time-free.
///
/// Two shapes, one per analysis: a dead store in `[individual_parameters]` (the taint
/// mask) and one among the `[odes]` intermediates (the dependency slice). Each is paired
/// with its **live twin**, differing only in the dead line's absence, which must be
/// accepted or refused for the real reason — that is what keeps this from being a test
/// that passes because everything is refused.
///
/// The error direction is the safe one: a loud over-refusal whose remedy (delete the dead
/// line) is right for the model anyway, rather than a silent freeze. When #1931 lands,
/// the two `assert!(… .is_some())` below become `assert_eq!(…, None)`.
#[test]
fn the_time_mask_and_the_slice_ignore_a_killed_definition() {
    // `[individual_parameters]`: `CL`'s TIME-dependent definition is overwritten.
    let dead_indiv = joint_model(
        "  CL   = TVCL * (1.0 + TVKT * TIME) * exp(ETA_CL)\n  \
         CL   = TVCL * exp(ETA_CL)\n  \
         V    = TVV\n  \
         KA   = TVKA\n  \
         H0   = TVH0\n  \
         BETA = TVBETA\n",
        PLAIN_ODES,
    );
    assert!(
        check_survival_time_dependent_pk(&dead_indiv).is_some(),
        "#1931: the dead TIME-dependent definition of CL is still followed"
    );
    // Its live twin: the dead line gone, nothing else changed — accepted.
    assert_eq!(
        check_survival_time_dependent_pk(&joint_model(TIME_FREE_IP, PLAIN_ODES)),
        None,
        "the same model without the dead line must be accepted, or the arm above is \
         refusing for some other reason"
    );

    // `[odes]`: `KEFF`'s read of the TIME-dependent `CL` is overwritten before use.
    let dead_odes = joint_model(
        "  CL   = TVCL * (1.0 + TVKT * TIME) * exp(ETA_CL)\n  \
         V    = TVV\n  \
         KA   = TVKA\n  \
         H0   = TVH0\n  \
         BETA = TVBETA\n",
        "  KEFF = CL / V\n  KEFF = 0.1\n  \
         d/dt(depot)   = -KA * depot\n  \
         d/dt(central) =  KA * depot - KEFF * central\n",
    );
    assert!(
        check_survival_time_dependent_pk(&dead_odes).is_some(),
        "#1931: the dead read of CL in the [odes] intermediate is still followed"
    );
    // Its live twin: the same block with the kill removed, so `CL` genuinely reaches the
    // accumulator — refused, and for the right reason, which the message states.
    let live_odes = joint_model(
        "  CL   = TVCL * (1.0 + TVKT * TIME) * exp(ETA_CL)\n  \
         V    = TVV\n  \
         KA   = TVKA\n  \
         H0   = TVH0\n  \
         BETA = TVBETA\n",
        "  KEFF = CL / V\n  \
         d/dt(depot)   = -KA * depot\n  \
         d/dt(central) =  KA * depot - KEFF * central\n",
    );
    let msg = check_survival_time_dependent_pk(&live_odes)
        .expect("the live twin's CL does reach the accumulator");
    assert!(
        msg.contains("`CL` reads the `TIME` built-in and the [odes] RHS reads it"),
        "the live twin must be refused for CL: {msg}"
    );
}

/// Scope: the freeze this refuses is the **ODE-accumulated** hazard's single `$PK`
/// snapshot, so a model with no such endpoint is not this check's business however its
/// `$PK` reads `TIME`. (An analytic hazard has its own frozen-snapshot question — its
/// `HazardParamFn` takes no time argument — which this change does not touch.)
#[test]
fn a_time_dependent_pk_without_an_ode_hazard_is_not_this_checks_business() {
    const M: &str = r"
[parameters]
  theta TVCL(1.0, FIX)
  theta TVV(10.0, FIX)
  theta TVKT(0.05, FIX)
  theta TVLAMBDA(0.05, FIX)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.01
[individual_parameters]
  CL = TVCL * (1.0 + TVKT * TIME) * exp(ETA_CL)
  V1 = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v1=V1)
[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
";
    let model = parse_model_string(M).expect("TTE + analytic hazard model compiles");
    assert!(
        compiled_model_uses_time_builtin(&model),
        "straddle: the block must read TIME, so only the missing ODE hazard returns None"
    );
    assert_eq!(check_survival_time_dependent_pk(&model), None);
}
