//! Unit tests for the `W_CMT_DEFAULTED` arm of [`reader_warning_suppressed`]
//! (#1009).
//!
//! The reader is model-blind: it counts every row whose compartment it had to
//! choose and says so. Whether choosing was a *guess* is a property of the model,
//! and this predicate is where `fit()` and `ferx check` agree on the answer.
//!
//! One test per model class rather than one table, so a predicate that loses an
//! arm reddens the arm it actually lost. Two rounds of review on PR #1404 each
//! found the same hole — a class nobody had listed — so the cases below are laid
//! out along the two channels the predicate now asks about separately: how many
//! compartments a **dose** can reach, and whether an **observation**'s CMT
//! dispatches anything.
use crate::api::validation::reader_warning_suppressed;
use crate::types::test_helpers::{analytical_model, ode_model};
use crate::types::{CompiledModel, GradientMethod, PkModel};

/// The exact shape the reader emits, so these tests break if the prefix moves.
const MSG: &str = "W_CMT_DEFAULTED: the dataset has no CMT column, so 10 dose row(s) and 110 \
                   observation row(s) were assigned compartment 1.";

/// A two-state `[odes]` model — `depot` + `central`, the shape the probe uses.
fn two_state_ode() -> CompiledModel {
    ode_model(GradientMethod::Fd)
}

/// The same model with a single addressable state.
fn one_state_ode() -> CompiledModel {
    let mut m = ode_model(GradientMethod::Fd);
    let spec = m.ode_spec.as_mut().expect("ode_model builds an ode_spec");
    spec.n_states = 1;
    spec.state_names = vec!["central".into()];
    m
}

/// An analytical `pk` model of the given topology.
fn analytical(pk_model: PkModel) -> CompiledModel {
    let mut m = analytical_model(GradientMethod::Fd);
    m.pk_model = pk_model;
    assert!(
        !m.is_algebraic(),
        "fixture must route to the topology branch, not the algebraic one"
    );
    m
}

/// A compartment-free (algebraic) model: no `ode_spec`, and an analytic readout
/// over no states at all — what `is_algebraic()` tests for.
fn algebraic() -> CompiledModel {
    let mut m = analytical_model(GradientMethod::Fd);
    m.analytic_readout = Some(crate::types::AnalyticReadout {
        readout: crate::ode::OdeReadout::ObsCmt(0),
        program: None,
        state_names: Vec::new(),
    });
    assert!(m.is_algebraic(), "fixture must actually be algebraic");
    m
}

/// A `PerCmt` readout over two compartments, for whichever struct carries it.
/// The two arms read the same state through a different scale, which is the
/// dispatch the defaulted CMT would silently pick between.
fn per_cmt_readout() -> crate::ode::OdeReadout {
    let mut map = std::collections::HashMap::new();
    map.insert(
        1usize,
        crate::ode::PerCmtReadout {
            out_fn: Box::new(|s: &[f64], _pk: &[f64], _t, _e, _c| s[0]),
            program: None,
        },
    );
    map.insert(
        2usize,
        crate::ode::PerCmtReadout {
            out_fn: Box::new(|s: &[f64], _pk: &[f64], _t, _e, _c| 1000.0 * s[0]),
            program: None,
        },
    );
    crate::ode::OdeReadout::PerCmt(map)
}

// ---------------------------------------------------------------------------
// Channel 1 — how many compartments a dose row's CMT can reach
// ---------------------------------------------------------------------------

#[test]
fn cmt_defaulted_is_reported_on_a_multi_state_ode_model() {
    // The live case. Two states means the dataset's silence really did decide
    // which one the drug went into — and on this model's own probe it decided
    // wrong by 154233 OFV units.
    assert!(
        !reader_warning_suppressed(&two_state_ode(), MSG),
        "a two-state [odes] model must see W_CMT_DEFAULTED"
    );
}

#[test]
fn cmt_defaulted_is_suppressed_on_a_single_state_ode_model() {
    // One state and a uniform readout: there is nothing to choose between, so the
    // count is true but the finding is not.
    assert!(
        reader_warning_suppressed(&one_state_ode(), MSG),
        "a one-state [odes] model has no second compartment to have missed"
    );
}

#[test]
fn cmt_defaulted_is_suppressed_on_a_one_compartment_iv_model() {
    // `ONE_CPT_IV` is `channels: [Some(Central)]` — the single analytical topology
    // with exactly one dose target. This is the control for the oral and 2-cpt
    // cases below: without it, "analytical models report" would pass for a
    // predicate that ignores the topology and reports every analytical model.
    assert!(
        reader_warning_suppressed(&analytical(PkModel::OneCptIv), MSG),
        "a 1-cpt IV model has exactly one compartment a dose can reach"
    );
}

#[test]
fn cmt_defaulted_is_reported_on_an_oral_model_whose_cmt_2_is_a_central_bolus() {
    // Review round 2, and a scope correction rather than a missed arm: the first
    // version of this predicate suppressed *every* analytical model, on the
    // argument that ferx's analytic numbering equals NONMEM's ADVAN numbering so a
    // CMT-less dataset doses what the fixed-DEFDOSE ADVANs dose. True, and beside
    // the point — `ONE_CPT_ORAL` is `channels: [Some(Depot), Some(Central)]`, so
    // `CMT=2` on an oral model is the documented depot-bypass central bolus
    // (#350), which `pk::dose_needs_event_walk` routes to the event walk and which
    // is NONMEM-anchored as ADVAN2. A dataset that meant that and lost its column
    // silently gets a depot bolus instead.
    assert!(
        !reader_warning_suppressed(&analytical(PkModel::OneCptOral), MSG),
        "CMT=2 on an oral model is a central bolus, so compartment 1 was a choice"
    );
}

#[test]
fn cmt_defaulted_is_reported_on_a_two_compartment_iv_model() {
    // The other analytical shape with a reachable second compartment, asserted
    // separately from the oral case: `TWO_CPT_IV` routes `CMT=2` to the peripheral
    // (ADVAN3 peripheral bolus), a different `Channel` variant from the oral
    // depot, so a predicate that hard-codes one topology reddens here.
    assert!(
        !reader_warning_suppressed(&analytical(PkModel::TwoCptIv), MSG),
        "CMT=2 on a 2-cpt IV model is a peripheral bolus"
    );
}

#[test]
fn cmt_defaulted_is_suppressed_on_a_transit_model_despite_two_states() {
    // The straddle that separates `addressable_dose_compartments` from `n_states`,
    // and the reason the predicate counts live channels instead of reading the
    // state count: `ONE_CPT_TRANSIT` is `n_states: 2` but `channels: &[]`, because
    // the transit closed form absorbs every dose through the depot and
    // `single_dose_concentration` never reads `dose.cmt`. Answering `n_states`
    // here would warn about a compartment no dose can be routed to.
    let transit = analytical(PkModel::OneCptTransit);
    assert_eq!(
        transit.pk_model.topology().n_states,
        2,
        "fixture must actually have more states than dose targets, or it is not a straddle"
    );
    assert!(
        reader_warning_suppressed(&transit, MSG),
        "a transit model's dose always absorbs through the depot, whatever CMT says"
    );
}

#[test]
fn cmt_defaulted_is_suppressed_on_a_compartment_free_model() {
    // No compartments at all (MBMA / algebraic). Same reasoning as #811's
    // `W_NO_DOSES` arm: the advice would be about a shape the model cannot have.
    assert!(
        reader_warning_suppressed(&algebraic(), MSG),
        "a compartment-free model has no compartment to have missed"
    );
}

// ---------------------------------------------------------------------------
// Channel 2 — whether an observation row's CMT dispatches anything
// ---------------------------------------------------------------------------

#[test]
fn cmt_defaulted_is_reported_on_an_analytical_model_with_per_cmt_scaling() {
    // Review round 1: the first version of this predicate tested only the *dose*
    // channel, and its doc comment claimed an analytical `pk` model "has no second
    // compartment to have missed". False on both counts — `CMT` also selects an
    // observation's scale. `[scaling] obs_scale[CMT=N]` parses on an analytical
    // model, and `pk::validate_per_cmt_scaling` only checks that the *observed*
    // CMTs have entries, so a CMT-less dataset keys every row to 1, `{1} ⊆ {1,2}`
    // passes, and nothing complains. Measured on `pk one_cpt_iv` with
    // `obs_scale[CMT=1] = 1000` / `obs_scale[CMT=2] = 1`: the same data spelled
    // with `CMT=2` gives OFV 0.0357, with the column dropped 6097015712.1246.
    //
    // Built on `OneCptIv` so the dose channel is *inert* (one live channel): this
    // must fail when the scaling arm is deleted, not pass on the strength of the
    // topology.
    let mut m = analytical(PkModel::OneCptIv);
    assert!(
        reader_warning_suppressed(&m, MSG),
        "control: without per-CMT scaling this one-target model is suppressed"
    );
    m.scaling = crate::types::ScalingSpec::PerCmt(std::collections::HashMap::from([
        (1usize, crate::types::ScalingSpec::ScalarScale(1000.0)),
        (2usize, crate::types::ScalingSpec::ScalarScale(1.0)),
    ]));
    assert!(
        !reader_warning_suppressed(&m, MSG),
        "an observation's CMT selects its scale, so a defaulted CMT can change the number"
    );
}

#[test]
fn cmt_defaulted_is_reported_on_a_one_state_ode_model_with_per_cmt_error_models() {
    // Review round 2, on the fixture rather than the arm: this test used to set
    // `ErrorSpec::PerCmt(HashMap::new())` on an analytical model, which is neither
    // shape it claimed to be. An **empty** map dispatches nothing, and a *non-empty*
    // per-CMT error model on an analytical model does not parse at all —
    // `model_parser.rs` rejects it with "Per-CMT error models (CMT=N: DV ~ ...)
    // require an ODE-based [structural_model]". So the arm was pinned by a fixture
    // that could only ever exercise the empty case.
    //
    // A one-state `[odes]` model is the shape that both parses and is not already
    // covered by the dose channel.
    let mut m = one_state_ode();
    assert!(
        reader_warning_suppressed(&m, MSG),
        "control: a one-state ODE model with a single error model is suppressed"
    );
    m.error_spec = crate::types::ErrorSpec::PerCmt(std::collections::HashMap::from([(
        1usize,
        crate::types::EndpointError {
            error_model: crate::types::ErrorModel::Additive,
            sigma_idx: vec![0],
        },
    )]));
    assert!(
        !reader_warning_suppressed(&m, MSG),
        "an observation's CMT selects its error model"
    );
}

#[test]
fn an_empty_per_cmt_error_map_still_reports_because_endpoints_route_by_cmt() {
    // A `!m.is_empty()` gate here was tried in review round 3 and **reverted**, and
    // this test is the record of why. `ErrorSpec::PerCmt({})` is what the parser
    // hands every model with no `[error_model]` block — TTE-only, binary,
    // categorical — and it dispatches no *error model*. But such a model routes its
    // rows to endpoints **by CMT**, so a compartment the reader invented still picks
    // the endpoint.
    //
    // The gate's justification was that `E_ENDPOINT_NO_RECORDS` already names the
    // missing column. That covers only the *absent-column* cause; `W_CMT_DEFAULTED`
    // also fires for a missing or unparseable **cell**, where every endpoint keeps
    // rows and that error's `routed.get(&cmt) == 0` condition cannot fire. Measured
    // on the competing-risks example with endpoints at `cmt = 1` / `cmt = 2` and one
    // event row spelled `x` instead of `2`: the event silently moves from `cause_b`
    // to `cause_a`, OFV 27.8497 against 28.3610, and under the gate `ferx check`
    // said `0 warning(s)` on both spellings.
    //
    // So the empty map reports, duplicating `E_ENDPOINT_NO_RECORDS` on the
    // absent-column case. Duplication is cheaper than silence.
    let mut m = analytical(PkModel::OneCptIv);
    assert!(
        reader_warning_suppressed(&m, MSG),
        "control: a 1-cpt IV model with a single error model is suppressed"
    );
    m.error_spec = crate::types::ErrorSpec::PerCmt(std::collections::HashMap::new());
    assert!(
        !reader_warning_suppressed(&m, MSG),
        "an endpoint model routes rows by CMT, so a defaulted CMT picks the endpoint"
    );
}

#[test]
fn cmt_defaulted_is_reported_on_an_analytical_model_with_a_per_cmt_analytic_readout() {
    // Review round 2, finding 2: this arm shipped untested, and deleting it left
    // the whole suite green. `AnalyticReadout::readout` is the analytical engine's
    // half of the per-CMT readout — `y[CMT=N]` on a `pk` model — and it selects
    // which expression an observation reads by its CMT.
    let mut m = analytical(PkModel::OneCptIv);
    assert!(reader_warning_suppressed(&m, MSG), "control");
    m.analytic_readout = Some(crate::types::AnalyticReadout {
        readout: per_cmt_readout(),
        program: None,
        state_names: vec!["central".into()],
    });
    assert!(
        !reader_warning_suppressed(&m, MSG),
        "an observation's CMT selects which analytic readout expression it reads"
    );
}

#[test]
fn cmt_defaulted_is_reported_on_a_one_state_ode_model_with_per_cmt_readouts() {
    // Review round 2, finding 1, and the same defect class as round 1's finding 1
    // left on the other engine: the predicate inspected `analytic_readout.readout`
    // and never `ode_spec.readout`, two different structs both holding an
    // `OdeReadout`. The multi-state arm hid it, so the gap was exactly a one-state
    // `[odes]` model with `y[CMT=N]`.
    //
    // Measured on `d/dt(central) = -CL/V*central` with `y[CMT=1] = central/V` and
    // `y[CMT=2] = central/V*1000`, the same rows spelled two ways: `CMT=2` gives
    // OFV 10028.0940, the column dropped gives 0.0357, and `ferx check` reported
    // "no errors (0 warning(s))" on both.
    //
    // `obs_cmts` is dispatched here by *two* engines — `ode::predictions` for the
    // f64 value and `sens::ode_provider` for the `Dual2` twin — so a fit on this
    // shape is wrong in the gradient as well as the objective.
    let mut m = one_state_ode();
    assert!(
        reader_warning_suppressed(&m, MSG),
        "control: the same model with a uniform readout is suppressed"
    );
    m.ode_spec
        .as_mut()
        .expect("one_state_ode builds an ode_spec")
        .readout = per_cmt_readout();
    assert!(
        !reader_warning_suppressed(&m, MSG),
        "an observation's CMT selects which ODE readout expression it reads"
    );
}

// ---------------------------------------------------------------------------
// The predicate's own shape
// ---------------------------------------------------------------------------

#[test]
fn the_cmt_arm_does_not_swallow_the_no_doses_arm() {
    // Control for the shape of the predicate itself: the `W_CMT_DEFAULTED` early
    // return must not change how `W_NO_DOSES` is decided (#811). An algebraic
    // model still suppresses `W_NO_DOSES`; a two-state ODE model still does not.
    let no_doses = "W_NO_DOSES: no dose records were parsed from the dataset.";
    assert!(reader_warning_suppressed(&algebraic(), no_doses));
    assert!(!reader_warning_suppressed(&two_state_ode(), no_doses));
    // And an unrelated reader warning is suppressed for nobody.
    let other = "W_MISSING_DV: 3 observation row(s) (EVID=0) had a missing DV.";
    assert!(!reader_warning_suppressed(&algebraic(), other));
    assert!(!reader_warning_suppressed(&two_state_ode(), other));
}
