//! Unit tests for the `W_CMT_DEFAULTED` arm of [`reader_warning_suppressed`]
//! (#1009).
//!
//! The reader is model-blind: it counts every row whose compartment it had to
//! choose and says so. Whether choosing was a *guess* is a property of the model,
//! and this predicate is where `fit()` and `ferx check` agree on the answer. Four
//! model classes, each its own test, so a predicate that is inverted (or whose
//! analytical arm is deleted) reddens the side it actually broke rather than one
//! test that could die for either reason.
use crate::api::validation::reader_warning_suppressed;
use crate::types::test_helpers::{analytical_model, ode_model};
use crate::types::{CompiledModel, GradientMethod};

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
fn cmt_defaulted_is_suppressed_on_an_analytical_pk_model() {
    // An analytical `pk` model resolves compartment 1 to its own default channel
    // (`dose_needs_event_walk`: depot on oral, central on IV), and that numbering
    // is ferx's own — never `$MODEL`'s — so a CMT-less dataset doses exactly what
    // NONMEM's fixed-DEFDOSE ADVANs dose. Measured: `examples/warfarin.ferx` is
    // bit-identical with and without the column (OFV -200.287841 both).
    assert!(
        reader_warning_suppressed(&analytical_model(GradientMethod::Fd), MSG),
        "an analytical pk model has no second compartment to have missed"
    );
}

#[test]
fn cmt_defaulted_is_suppressed_on_a_single_state_ode_model() {
    // One state: there is nothing to choose between, so the count is true but the
    // finding is not.
    assert!(
        reader_warning_suppressed(&one_state_ode(), MSG),
        "a one-state [odes] model has no second compartment to have missed"
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
