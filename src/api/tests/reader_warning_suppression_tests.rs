//! Unit tests for the `W_CMT_DEFAULTED` arm of [`reader_warning_suppressed`]
//! (#1009).
//!
//! The reader is model-blind: it counts every row whose compartment it had to
//! choose and says so. Whether choosing was a *guess* is a property of the model,
//! and this predicate is where `fit()` and `ferx check` agree on the answer.
//!
//! One test per model class rather than one table, so a predicate that loses an
//! arm reddens the arm it actually lost. Three rounds of review on PR #1404 each
//! found the same hole — a class nobody had listed — so the cases below are laid
//! out along the channels the predicate asks about separately: how many
//! compartments a **dose** can reach, whether an **observation**'s CMT dispatches
//! anything, and (since #1409) whether the `[data_selection]` filter reads the
//! compartment the reader chose.
//!
//! Since #1409 the predicate is an `any()` over [`CmtConsumer`], so the channels are
//! also tested *as an enumeration*: `cmt_consumer_all_is_generated_from_the_enum_itself`
//! pins the macro-generated list, and
//! `every_cmt_consumer_has_a_fixture_where_it_is_the_only_live_one` requires each
//! variant to carry a model on which **it alone** reads `CMT` — so a new consumer is
//! a compile error until someone builds the shape that makes it live, and no arm can
//! be pinned by a neighbour that happens to fire on the same fixture.
use crate::api::validation::{reader_warning_suppressed, CmtConsumer};
use crate::types::test_helpers::{analytical_model, ode_model};
use crate::types::{CompiledModel, FitOptions, GradientMethod, PkModel};

/// The exact shape the reader emits, so these tests break if the prefix moves.
const MSG: &str = "W_CMT_DEFAULTED: the dataset has no CMT column, so 10 dose row(s) and 110 \
                   observation row(s) were assigned compartment 1.";

/// Options with no `[data_selection]` clauses — the default for every case whose
/// subject is the *model* side of the predicate, so the filter channel is inert and
/// cannot make a model-side assertion pass for the wrong reason (#1409).
fn no_filter() -> FitOptions {
    let opts = FitOptions::default();
    assert!(
        opts.ignore_exprs.is_empty() && opts.accept_exprs.is_empty(),
        "the default options must carry no data-selection clauses, or every \
         model-side case below is contaminated by the filter channel"
    );
    opts
}

/// Options carrying one `[data_selection]` clause.
fn ignoring(expr: &str) -> FitOptions {
    FitOptions {
        ignore_exprs: vec![expr.to_string()],
        ..FitOptions::default()
    }
}

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
        !reader_warning_suppressed(&two_state_ode(), &no_filter(), MSG),
        "a two-state [odes] model must see W_CMT_DEFAULTED"
    );
}

#[test]
fn cmt_defaulted_is_suppressed_on_a_single_state_ode_model() {
    // One state and a uniform readout: there is nothing to choose between, so the
    // count is true but the finding is not.
    assert!(
        reader_warning_suppressed(&one_state_ode(), &no_filter(), MSG),
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
        reader_warning_suppressed(&analytical(PkModel::OneCptIv), &no_filter(), MSG),
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
        !reader_warning_suppressed(&analytical(PkModel::OneCptOral), &no_filter(), MSG),
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
        !reader_warning_suppressed(&analytical(PkModel::TwoCptIv), &no_filter(), MSG),
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
        reader_warning_suppressed(&transit, &no_filter(), MSG),
        "a transit model's dose always absorbs through the depot, whatever CMT says"
    );
}

#[test]
fn cmt_defaulted_is_suppressed_on_a_compartment_free_model() {
    // No compartments at all (MBMA / algebraic). Same reasoning as #811's
    // `W_NO_DOSES` arm: the advice would be about a shape the model cannot have.
    assert!(
        reader_warning_suppressed(&algebraic(), &no_filter(), MSG),
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
        reader_warning_suppressed(&m, &no_filter(), MSG),
        "control: without per-CMT scaling this one-target model is suppressed"
    );
    m.scaling = crate::types::ScalingSpec::PerCmt(std::collections::HashMap::from([
        (1usize, crate::types::ScalingSpec::ScalarScale(1000.0)),
        (2usize, crate::types::ScalingSpec::ScalarScale(1.0)),
    ]));
    assert!(
        !reader_warning_suppressed(&m, &no_filter(), MSG),
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
        reader_warning_suppressed(&m, &no_filter(), MSG),
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
        !reader_warning_suppressed(&m, &no_filter(), MSG),
        "an observation's CMT selects its error model"
    );
}

/// A declared TTE endpoint on `cmt`, the cheapest `EndpointLikelihood` to build.
/// Which *kind* of endpoint it is does not matter here — what matters is that
/// `api::run::obs_routing_for` puts its CMT into a routing set.
#[cfg(feature = "survival")]
fn with_tte_endpoint(mut m: CompiledModel, cmt: usize) -> CompiledModel {
    m.endpoints.insert(
        cmt,
        crate::types::EndpointLikelihood::Tte {
            hazard: crate::types::HazardSpec::OdeAccumulated { chz_state: 0 },
            recurrence: crate::types::TteRecurrence::Single,
            hazard_covariates: Vec::new(),
        },
    );
    m
}

#[cfg(feature = "survival")]
#[test]
fn an_endpoint_model_reports_because_its_rows_route_by_cmt() {
    // An endpoint model must report, and #1409 changed *why*. Until then the only
    // thing that reported it was `ErrorSpec::PerCmt` matching an **empty** map — the
    // no-`[error_model]` shape the parser hands every TTE-only / binary / categorical
    // model. That is an accident of a dispatch table being empty, and a
    // `!m.is_empty()` gate in #1404 round 3 removed it, silencing a measured
    // mis-routing: competing risks with endpoints at `cmt = 1` / `cmt = 2`, one event
    // row's cell spelled `x` instead of `2`, the event moves from `cause_b` to
    // `cause_a`, OFV 27.8497 against 28.3610, and `ferx check` said `0 warning(s)`
    // both ways.
    //
    // (`E_ENDPOINT_NO_RECORDS` does not cover this: it reasons about an *absent
    // column*, where an endpoint ends up with no rows at all. `W_CMT_DEFAULTED` also
    // fires for a missing or unparseable **cell**, where every endpoint keeps rows
    // and that error's `routed.get(&cmt) == 0` condition cannot fire.)
    //
    // The reason is now the routing set itself. Note the fixture keeps a *non-empty*
    // `ErrorSpec`, so the error-model arm is inert and this can only pass through
    // `CmtConsumer::EndpointRouting`.
    let m = analytical(PkModel::OneCptIv);
    assert!(
        reader_warning_suppressed(&m, &no_filter(), MSG),
        "control: a 1-cpt IV model with a single error model is suppressed"
    );
    let m = with_tte_endpoint(m, 2);
    assert!(
        matches!(m.error_spec, crate::types::ErrorSpec::Single(_)),
        "fixture must keep a non-endpoint error spec, or the error-model arm could \
         be what is passing"
    );
    assert!(
        !reader_warning_suppressed(&m, &no_filter(), MSG),
        "an endpoint model routes rows by CMT, so a defaulted CMT picks the endpoint"
    );
}

#[cfg(feature = "survival")]
#[test]
fn an_endpoint_at_the_default_compartment_still_reports_when_gaussian_rows_exist() {
    // Both halves of the `routes_only(DEFAULT_CMT)` suppression, pinned apart.
    //
    // A lone endpoint at `cmt = 1` on an **endpoint-only** model is the documented
    // false positive #1409 closed: the reader's fallback IS the endpoint, so the guess
    // is provably right. But the same endpoint on a model that also scores Gaussian
    // observations is a genuine ambiguity — a defaulted row picks between the endpoint
    // and the Gaussian grid — and must still report.
    //
    // Without this case the suppression can be written as `routes_only(1)` alone and
    // the whole suite stays green (verified by mutation), silently withdrawing the
    // warning from every `cmt = 1` endpoint model that has ordinary observations too.
    let gaussian_and_endpoint = with_tte_endpoint(analytical(PkModel::OneCptIv), 1);
    assert!(
        matches!(
            gaussian_and_endpoint.error_spec,
            crate::types::ErrorSpec::Single(_)
        ),
        "fixture must score Gaussian observations, or it is the endpoint-only case"
    );
    assert!(
        CmtConsumer::EndpointRouting.is_live(&gaussian_and_endpoint, &no_filter()),
        "a defaulted row picks between the cmt=1 endpoint and the Gaussian grid"
    );

    // The endpoint-only twin of the same shape: nothing else to fall into, so silent.
    let mut endpoint_only = with_tte_endpoint(analytical(PkModel::OneCptIv), 1);
    endpoint_only.error_spec = crate::types::ErrorSpec::PerCmt(std::collections::HashMap::new());
    assert!(
        !CmtConsumer::EndpointRouting.is_live(&endpoint_only, &no_filter()),
        "an endpoint-only model whose only endpoint is the default compartment has \
         one provably-right answer"
    );
}

#[cfg(feature = "survival")]
#[test]
fn the_empty_error_map_and_the_routing_set_are_the_same_models() {
    // The safety condition for `PerCmtErrorModel` to require a non-empty map. An
    // empty `ErrorSpec::PerCmt` is the parser's marker for "no `[error_model]`
    // block", which only an endpoint-only model has — so every model the old,
    // wider arm caught must still be caught, now by `EndpointRouting`. If the two
    // ever come apart, the narrowing is a silent loss of scope and this reddens.
    //
    // Both directions, on the same fixture: an endpoint with an empty error map
    // reports, and an empty error map *without* an endpoint is not a shape the
    // parser can produce — asserted here as the routing set being what answers.
    let mut m = analytical(PkModel::OneCptIv);
    m.error_spec = crate::types::ErrorSpec::PerCmt(std::collections::HashMap::new());
    assert!(
        !CmtConsumer::PerCmtErrorModel.is_live(&m, &no_filter()),
        "an empty dispatch map selects no error model, so this arm must stay quiet"
    );
    let m = with_tte_endpoint(m, 2);
    assert!(
        CmtConsumer::EndpointRouting.is_live(&m, &no_filter()),
        "the endpoint's CMT is in a routing set, so the endpoint arm must answer"
    );
    assert!(
        !reader_warning_suppressed(&m, &no_filter(), MSG),
        "…and the warning must survive the narrowing"
    );
}

#[test]
fn cmt_defaulted_is_reported_on_an_analytical_model_with_a_per_cmt_analytic_readout() {
    // Review round 2, finding 2: this arm shipped untested, and deleting it left
    // the whole suite green. `AnalyticReadout::readout` is the analytical engine's
    // half of the per-CMT readout — `y[CMT=N]` on a `pk` model — and it selects
    // which expression an observation reads by its CMT.
    let mut m = analytical(PkModel::OneCptIv);
    assert!(reader_warning_suppressed(&m, &no_filter(), MSG), "control");
    m.analytic_readout = Some(crate::types::AnalyticReadout {
        readout: per_cmt_readout(),
        program: None,
        state_names: vec!["central".into()],
    });
    assert!(
        !reader_warning_suppressed(&m, &no_filter(), MSG),
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
        reader_warning_suppressed(&m, &no_filter(), MSG),
        "control: the same model with a uniform readout is suppressed"
    );
    m.ode_spec
        .as_mut()
        .expect("one_state_ode builds an ode_spec")
        .readout = per_cmt_readout();
    assert!(
        !reader_warning_suppressed(&m, &no_filter(), MSG),
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
    assert!(reader_warning_suppressed(
        &algebraic(),
        &no_filter(),
        no_doses
    ));
    assert!(!reader_warning_suppressed(
        &two_state_ode(),
        &no_filter(),
        no_doses
    ));
    // And an unrelated reader warning is suppressed for nobody.
    let other = "W_MISSING_DV: 3 observation row(s) (EVID=0) had a missing DV.";
    assert!(!reader_warning_suppressed(
        &algebraic(),
        &no_filter(),
        other
    ));
    assert!(!reader_warning_suppressed(
        &two_state_ode(),
        &no_filter(),
        other
    ));
}

// ---------------------------------------------------------------------------
// Channel 3 — which rows the `[data_selection]` filter keeps (#1409)
// ---------------------------------------------------------------------------

#[test]
fn cmt_defaulted_is_reported_when_a_data_selection_clause_compares_cmt() {
    // The live defect #1409 was filed for. `resolve_row_cmt` feeds the *defaulted*
    // compartment to the filter's `RowContext`, so `ignore = CMT == 2` selects rows
    // on a value the reader invented. Measured on `pk one_cpt_iv` with one
    // observation cell spelled `2` against `x` and nothing else changed: 3 records
    // scored at −5.7650 against 4 at −5.2516 (the realised numbers of
    // `tests/cmt_data_selection_scope.rs`; #1409's reprex quotes −4.6162 / −6.9517 on
    // its own dataset), with `ferx check` reporting `ok — 0 warning(s)` on both.
    //
    // Built on `OneCptIv` with no per-CMT anything, so every model-side arm is inert
    // and this can only pass through `CmtConsumer::DataSelectionFilter`.
    let m = analytical(PkModel::OneCptIv);
    assert!(
        reader_warning_suppressed(&m, &no_filter(), MSG),
        "control: with no clauses this model reads CMT for nothing"
    );
    assert!(
        !reader_warning_suppressed(&m, &ignoring("CMT == 2"), MSG),
        "a clause comparing CMT selects rows on the compartment the reader chose"
    );
}

#[test]
fn a_data_selection_clause_on_another_column_is_not_a_cmt_consumer() {
    // The other side of the straddle, and the reason the arm asks *which* column a
    // clause reads rather than whether any clause exists: `ignore = DV < 0.001` never
    // touches the compartment, so it must not widen the warning. Without this, an arm
    // written as `!options.ignore_exprs.is_empty()` passes the test above.
    let m = analytical(PkModel::OneCptIv);
    for expr in ["DV < 0.001", "STUDY == 2", "EVID == 2 && TIME > 24"] {
        assert!(
            reader_warning_suppressed(&m, &ignoring(expr), MSG),
            "`{expr}` reads no compartment, so it cannot make a defaulted CMT matter"
        );
    }
}

#[test]
fn the_data_selection_arm_reads_the_clause_the_way_the_filter_does() {
    // Spellings the filter accepts and a substring scan for "cmt" would get wrong in
    // both directions. Asked of the parsed clauses, so these are answers rather than
    // coincidences:
    //
    // * case, and the `&&` split — the filter lowercases each sub-expression's column;
    // * an `accept` clause is as much a consumer as an `ignore` one;
    // * a covariate column whose *name contains* `cmt` is a different column.
    let m = analytical(PkModel::OneCptIv);
    for expr in ["cmt == 2", "TIME > 0 && CMT != 1", "Cmt >= 2"] {
        assert!(
            !reader_warning_suppressed(&m, &ignoring(expr), MSG),
            "`{expr}` compares the CMT column"
        );
    }
    let accept_only = FitOptions {
        accept_exprs: vec!["CMT == 1".to_string()],
        ..FitOptions::default()
    };
    assert!(
        !reader_warning_suppressed(&m, &accept_only, MSG),
        "an `accept` clause filters on the resolved CMT exactly as an `ignore` one does"
    );
    assert!(
        reader_warning_suppressed(&m, &ignoring("CMTX == 2"), MSG),
        "`CMTX` is a covariate column of its own, not the CMT column"
    );
    // `ignore_subjects` compares `Subject::id` and reads no row column, so it is not
    // a consumer however many entries it has.
    let by_subject = FitOptions {
        ignore_subjects: vec!["3".to_string()],
        ..FitOptions::default()
    };
    assert!(
        reader_warning_suppressed(&m, &by_subject, MSG),
        "dropping whole subjects by ID reads no compartment"
    );
}

#[test]
fn an_unparseable_data_selection_clause_is_answered_conservatively() {
    // A clause the filter cannot compile is about to fail the read outright, so an
    // extra warning can mask nothing — while answering `false` would silently narrow
    // the predicate on an expression nobody has inspected. Pinned so the choice is a
    // decision rather than a side-effect of `unwrap_or_default()`.
    let m = analytical(PkModel::OneCptIv);
    let bad = "CMT || 2";
    assert!(
        crate::io::datareader::SelectionFilter::from_opts(&[bad.to_string()], &[], &[]).is_err(),
        "fixture must actually fail to parse, or this tests the happy path"
    );
    assert!(
        !reader_warning_suppressed(&m, &ignoring(bad), MSG),
        "an unparseable clause is treated as reading CMT"
    );
}

// ---------------------------------------------------------------------------
// The enumeration itself (#1409)
// ---------------------------------------------------------------------------

#[test]
fn cmt_consumer_all_is_generated_from_the_enum_itself() {
    // The completeness guard, and what it now rests on. `CmtConsumer` and
    // `CmtConsumer::ALL` are expanded from a single `cmt_consumers!` token list, so a
    // variant cannot exist without being in `ALL` — there is no second list to forget.
    //
    // Two earlier spellings both had a hole, each found by the same probe (a 7th
    // variant with arms in every `match`): a hand-written `ALL: [CmtConsumer; 6]` left
    // the suite green because the array simply stayed at six, and a `next()` chain left
    // it green because a variant whose arm was `None` was unreachable from the
    // iterator. Under the macro the probe reddens
    // `every_cmt_consumer_has_a_fixture_where_it_is_the_only_live_one`, which is the
    // behaviour we want: a new channel is not done until it has a fixture.
    //
    // This test pins the cheap structural half — no duplicates, and every channel this
    // file builds a fixture for is present.
    let all: Vec<CmtConsumer> = CmtConsumer::iter().collect();
    for c in &all {
        assert_eq!(
            all.iter().filter(|x| *x == c).count(),
            1,
            "{c:?} appears more than once in CmtConsumer::ALL: {all:?}"
        );
    }
    assert_eq!(
        all.len(),
        CmtConsumer::ALL.len(),
        "iter() must visit every entry of ALL"
    );
    for expected in [
        CmtConsumer::DoseCompartment,
        CmtConsumer::PerCmtScaling,
        CmtConsumer::PerCmtErrorModel,
        CmtConsumer::PerCmtReadout,
        CmtConsumer::EndpointRouting,
        CmtConsumer::DataSelectionFilter,
    ] {
        assert!(
            all.contains(&expected),
            "{expected:?} is missing from CmtConsumer::ALL: {all:?}"
        );
    }
}

/// A model + options on which **exactly one** consumer reads `CMT`.
///
/// **Exhaustive on purpose.** A new [`CmtConsumer`] variant is a compile error here
/// until someone builds the shape that makes it live — which is the step every
/// #1404 review round skipped, each time by widening a condition in place and
/// shipping no fixture that could tell whether the widening had worked.
#[cfg(feature = "survival")]
fn only_live(c: CmtConsumer) -> (CompiledModel, FitOptions) {
    match c {
        // Two `[odes]` states: the dataset's silence really did pick one.
        CmtConsumer::DoseCompartment => (two_state_ode(), no_filter()),
        CmtConsumer::PerCmtScaling => {
            let mut m = analytical(PkModel::OneCptIv);
            m.scaling = crate::types::ScalingSpec::PerCmt(std::collections::HashMap::from([
                (1usize, crate::types::ScalingSpec::ScalarScale(1000.0)),
                (2usize, crate::types::ScalingSpec::ScalarScale(1.0)),
            ]));
            (m, no_filter())
        }
        // A non-empty per-CMT error map does not parse on an analytical model, so
        // this one is built on the single-state ODE shape, where the dose arm is
        // inert for a different reason (one state rather than one channel).
        CmtConsumer::PerCmtErrorModel => {
            let mut m = one_state_ode();
            m.error_spec = crate::types::ErrorSpec::PerCmt(std::collections::HashMap::from([(
                1usize,
                crate::types::EndpointError {
                    error_model: crate::types::ErrorModel::Additive,
                    sigma_idx: vec![0],
                },
            )]));
            (m, no_filter())
        }
        CmtConsumer::PerCmtReadout => {
            let mut m = one_state_ode();
            m.ode_spec
                .as_mut()
                .expect("one_state_ode builds an ode_spec")
                .readout = per_cmt_readout();
            (m, no_filter())
        }
        CmtConsumer::EndpointRouting => (
            with_tte_endpoint(analytical(PkModel::OneCptIv), 2),
            no_filter(),
        ),
        CmtConsumer::DataSelectionFilter => (analytical(PkModel::OneCptIv), ignoring("CMT == 2")),
    }
}

#[cfg(feature = "survival")]
#[test]
fn every_cmt_consumer_has_a_fixture_where_it_is_the_only_live_one() {
    // The property the enumeration exists for, and the one no #1404 round could
    // state: each channel is, on its own, a reason to report — and *only* that
    // channel is live on its fixture, so deleting that arm reddens this test rather
    // than being covered by a neighbour. (CLAUDE.md, "two redundant gates cover for
    // each other": an arm that never decides anything alone is untested however
    // green the suite is.)
    for c in CmtConsumer::iter() {
        let (model, options) = only_live(c);
        let live: Vec<CmtConsumer> = CmtConsumer::iter()
            .filter(|x| x.is_live(&model, &options))
            .collect();
        assert_eq!(
            live,
            vec![c],
            "{c:?}'s fixture must make exactly that channel live, else the arm is \
             pinned by whatever else is live on it"
        );
        assert!(
            !reader_warning_suppressed(&model, &options, MSG),
            "{c:?} alone must be enough to report W_CMT_DEFAULTED"
        );
    }
}

#[cfg(feature = "survival")]
#[test]
fn a_model_that_reads_cmt_for_nothing_is_suppressed_on_every_channel() {
    // The control for the loop above: with no channel live the warning is withheld.
    // Without it, every assertion there passes for a predicate hard-wired to `true`.
    let (model, options) = (analytical(PkModel::OneCptIv), no_filter());
    let live: Vec<CmtConsumer> = CmtConsumer::iter()
        .filter(|x| x.is_live(&model, &options))
        .collect();
    assert_eq!(
        live,
        Vec::new(),
        "the control fixture must read CMT nowhere"
    );
    assert!(reader_warning_suppressed(&model, &options, MSG));
}

#[cfg(feature = "survival")]
#[test]
fn an_endpoint_only_models_placeholder_pk_model_is_not_read_as_a_dose_channel() {
    // The `pk_model` of an endpoint-only model is a **placeholder** — no
    // `[structural_model]` block, so `model_parser.rs` stores `PkModel::OneCptIv`
    // for a model no closed form will ever serve, and `types.rs` warns it must never
    // be dispatched on (#1356). The predicate used to reach its topology for any
    // model that is not `is_algebraic()`, which is benign today only because that
    // parser arm happens to pick a one-channel model; the `[odes]` arm picks
    // `OneCptOral`, which has two.
    //
    // Asserted on the channel rather than on the warning on purpose: such a model
    // reports anyway through `EndpointRouting`, so a warning-level assertion would
    // pass with the gate deleted. Mutation-checked — removing
    // `analytical_closed_form_dispatched` from `addressable_dose_compartments`
    // reddens this and nothing else.
    let mut m = analytical(PkModel::TwoCptIv);
    m.error_spec = crate::types::ErrorSpec::PerCmt(std::collections::HashMap::new());
    let m = with_tte_endpoint(m, 2);
    assert_eq!(
        m.pk_model.topology().addressable_dose_compartments(),
        2,
        "fixture must carry a placeholder whose topology WOULD claim a second \
         channel, or the gate has nothing to suppress"
    );
    assert!(
        !CmtConsumer::DoseCompartment.is_live(&m, &no_filter()),
        "no closed form serves this model, so its `pk_model` addresses nothing"
    );
    assert!(
        CmtConsumer::EndpointRouting.is_live(&m, &no_filter()),
        "…and the reason it still reports is the endpoint routing, as it should be"
    );
}
