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
//! `{ reads TIME } × { read by the RHS, by an init() seed, by both, by neither }
//!  × { OdeAccumulated hazard, analytic hazard }`. The cells that must refuse are the
//! first three with an `OdeAccumulated` hazard; every other cell must return `None`, and
//! each has a test below — including the two that look like they should refuse and must
//! not: a `TIME`-reading **dose-record** quantity (#1575 reads those per dose record) and
//! a `TIME`-reading **`[odes]` RHS** (the integrator clock honours that, #610).

use super::*;
use crate::parser::model_parser::{compiled_model_uses_time_builtin, parse_model_string};

/// A joint PK-TTE ODE model: `[individual_parameters]` is `ip`, the `[odes]` block is
/// `odes`. Two states, so the injected `d/dt(__chz_3)` accumulator makes the hazard CMT 3.
fn joint_model(ip: &str, odes: &str) -> CompiledModel {
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
  hazard = H0 * exp(BETA * (central / V))
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
