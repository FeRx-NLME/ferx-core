//! Unit tests for [`check_dose_compartments`] (#375) — the up-front guard that
//! every dose names a compartment the analytical engine can actually deliver it
//! into.
//!
//! Before this check the three analytical paths disagreed on an unroutable dose:
//! the dose-superposition path silently routed the infusion into central, the
//! event-driven walk `panic!`ed from inside the prediction loop, and the `sens`
//! gradient walk silently dropped it. These tests pin the rejection (so the
//! panic is unreachable from user data) and, just as importantly, the positive
//! controls — the compartments that *are* routable must stay accepted.
use super::check_dose_compartments;
use crate::types::test_helpers::{analytical_model, ode_model};
use crate::types::{CompiledModel, DoseEvent, GradientMethod, PkModel, Population, Subject};
use std::collections::HashMap;

/// A one-subject population carrying exactly the doses under test.
fn population_with_doses(doses: Vec<DoseEvent>) -> Population {
    Population {
        subjects: vec![Subject {
            obs_rgroup: Vec::new(),
            id: "1".into(),
            doses,
            obs_times: vec![1.0],
            obs_raw_times: Vec::new(),
            observations: vec![0.0],
            obs_cmts: vec![1],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_raw_times: Vec::new(),
            reset_period_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            reset_covariates: Vec::new(),
            cens: vec![0],
            occasions: Vec::new(),
            obs_l2: Vec::new(),
            dose_occasions: Vec::new(),
            reset_occasions: Vec::new(),
            pk_only_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
        }],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

fn model_with(pk_model: PkModel) -> CompiledModel {
    let mut model = analytical_model(GradientMethod::Fd);
    model.pk_model = pk_model;
    model
}

/// `RATE>0` into compartment `cmt`.
fn infusion(cmt: usize) -> DoseEvent {
    DoseEvent::new(0.0, 100.0, cmt, 10.0, false, 0.0)
}

/// A plain bolus into compartment `cmt`.
fn bolus(cmt: usize) -> DoseEvent {
    DoseEvent::new(0.0, 100.0, cmt, 0.0, false, 0.0)
}

fn codes(model: &CompiledModel, population: &Population) -> Vec<String> {
    check_dose_compartments(model, population)
        .into_iter()
        .map(|d| d.code)
        .collect()
}

// ── the reported case: infusion into a compartment the walk cannot route ──

/// The #375 repro — an infusion into an oral model's peripheral. It used to
/// `panic!` from inside the event-driven walk, then (mid-#375) was rejected, and
/// is now **supported**: the oral propagators gained the peripheral forcing term
/// by reusing the IV forced response. Validated against NONMEM `ADVAN4`/`ADVAN12`
/// and, exactly, against an ODE twin (`tests/oral_peripheral_infusion.rs`).
#[test]
fn infusion_into_oral_peripheral_is_supported() {
    for (pk_model, cmt) in [
        (PkModel::TwoCptOral, 3),
        (PkModel::ThreeCptOral, 3),
        (PkModel::ThreeCptOral, 4),
    ] {
        let model = model_with(pk_model);
        let population = population_with_doses(vec![infusion(cmt)]);
        assert!(
            codes(&model, &population).is_empty(),
            "{pk_model:?} cmt {cmt} infusion should be supported"
        );
    }
}

/// With every in-range compartment of every walk-supported model now infusable,
/// `E_DOSE_CMT_NOT_INFUSABLE` narrows to exactly one case: `CMT=0`, which has no
/// rate channel because an infusion has no "default compartment" fallback. Pinned
/// so the code isn't quietly dead.
#[test]
fn every_in_range_compartment_is_infusable_on_walk_models() {
    for pk_model in [
        PkModel::OneCptIv,
        PkModel::OneCptOral,
        PkModel::TwoCptIv,
        PkModel::TwoCptOral,
        PkModel::ThreeCptIv,
        PkModel::ThreeCptOral,
    ] {
        let n = pk_model.topology().state_layout().0;
        for cmt in 1..=n {
            let model = model_with(pk_model);
            let population = population_with_doses(vec![infusion(cmt)]);
            assert!(
                codes(&model, &population).is_empty(),
                "{pk_model:?} cmt {cmt} infusion should be supported"
            );
        }
    }
}

/// `one_cpt_iv` has a single compartment, so cmt 2 is neither infusable nor in
/// range. The **range** rule owns it — "the model has only 1 compartment" is a
/// more useful message than "that compartment is not infusable" — and it is
/// reported once, not twice.
#[test]
fn infusion_beyond_the_state_count_is_rejected_once_as_out_of_range() {
    let model = model_with(PkModel::OneCptIv);
    let population = population_with_doses(vec![infusion(2)]);
    assert_eq!(codes(&model, &population), vec!["E_DOSE_CMT_OUT_OF_RANGE"]);
}

/// The range rule must apply to infusions on transit/IG too. They skip the
/// *routing* rule (an infusion reroutes them to their ODE twin, #719 gap 2), but
/// they are still 2-/3-state models: without a range check an out-of-range
/// infusion passes validation and is then silently dropped by the twin's bounds
/// check — the silent-drop failure this whole check exists to remove.
#[test]
fn out_of_range_infusion_on_transit_is_still_rejected() {
    for pk_model in [PkModel::OneCptTransit, PkModel::OneCptIg] {
        let model = model_with(pk_model);
        let population = population_with_doses(vec![infusion(9)]);
        assert_eq!(
            codes(&model, &population),
            vec!["E_DOSE_CMT_OUT_OF_RANGE"],
            "{pk_model:?} out-of-range infusion"
        );
    }
}

/// An infusion has no "default compartment" fallback — `dose_channel(0)` is
/// `None`, so the walk panicked on `CMT=0` too. Rejected rather than silently
/// treated as central (which is what the superposition path did).
#[test]
fn infusion_into_cmt_zero_is_rejected() {
    let model = model_with(PkModel::OneCptIv);
    let population = population_with_doses(vec![infusion(0)]);
    assert_eq!(codes(&model, &population), vec!["E_DOSE_CMT_NOT_INFUSABLE"]);
}

/// The message must carry the subject/time context the deep-in-the-walk panic
/// never had, and name the compartments that *are* infusable.
#[test]
fn rejection_message_names_the_subject_time_and_infusable_set() {
    let model = model_with(PkModel::TwoCptOral);
    let population = population_with_doses(vec![infusion(0)]);
    let diags = check_dose_compartments(&model, &population);
    let msg = &diags[0].message;
    assert!(msg.contains("subject 1"), "{msg}");
    assert!(msg.contains("compartment 0"), "{msg}");
    assert!(msg.contains("two_cpt_oral"), "{msg}");
    assert!(msg.contains("[1, 2, 3]"), "{msg}");
    assert_eq!(diags[0].block.as_deref(), Some("structural_model"));
}

// ── positive controls: everything the closed forms genuinely route ──

/// Zero-order release into the oral depot is supported since #400 — this check
/// must not regress it.
#[test]
fn infusion_into_oral_depot_is_accepted() {
    for pk_model in [
        PkModel::OneCptOral,
        PkModel::TwoCptOral,
        PkModel::ThreeCptOral,
    ] {
        let model = model_with(pk_model);
        let population = population_with_doses(vec![infusion(1)]);
        assert!(
            codes(&model, &population).is_empty(),
            "{pk_model:?} depot infusion should be accepted"
        );
    }
}

/// Central is infusable for every model; the IV models also take their
/// peripheral(s).
#[test]
fn infusion_into_central_and_iv_peripherals_is_accepted() {
    let cases = [
        (PkModel::OneCptIv, 1),
        (PkModel::OneCptOral, 2),
        (PkModel::TwoCptIv, 1),
        (PkModel::TwoCptIv, 2),
        (PkModel::ThreeCptIv, 2),
        (PkModel::ThreeCptIv, 3),
        (PkModel::TwoCptOral, 2),
        (PkModel::ThreeCptOral, 2),
    ];
    for (pk_model, cmt) in cases {
        let model = model_with(pk_model);
        let population = population_with_doses(vec![infusion(cmt)]);
        assert!(
            codes(&model, &population).is_empty(),
            "{pk_model:?} cmt {cmt} infusion should be accepted"
        );
    }
}

/// A bolus into a compartment that exists is fine even when that compartment is
/// not *infusable* — the walk adds the amount to the state directly. Rejecting
/// these would break `two_cpt_oral` peripheral boluses the walk handles today.
#[test]
fn bolus_into_a_non_infusable_but_existing_compartment_is_accepted() {
    let model = model_with(PkModel::TwoCptOral);
    let population = population_with_doses(vec![bolus(3)]);
    assert!(codes(&model, &population).is_empty());
}

/// `CMT=0` on a bolus is NONMEM's "default dose compartment", and both
/// analytical paths already agree on it (the walk via `saturating_sub(1)`, the
/// superposition path by ignoring `cmt`). Not a silent mis-route, so not
/// rejected.
#[test]
fn bolus_into_cmt_zero_is_accepted() {
    let model = model_with(PkModel::OneCptIv);
    let population = population_with_doses(vec![bolus(0)]);
    assert!(codes(&model, &population).is_empty());
}

// ── the out-of-range bolus ──

/// The sibling panic: a bolus past the end of the state vector.
#[test]
fn bolus_beyond_the_state_count_is_rejected() {
    let model = model_with(PkModel::OneCptIv);
    let population = population_with_doses(vec![bolus(2)]);
    let diags = check_dose_compartments(&model, &population);
    assert_eq!(
        diags.iter().map(|d| d.code.as_str()).collect::<Vec<_>>(),
        vec!["E_DOSE_CMT_OUT_OF_RANGE"]
    );
    let msg = &diags[0].message;
    assert!(msg.contains("only 1 compartment(s)"), "{msg}");
    assert!(msg.contains("central"), "{msg}");
}

/// The boundary: `cmt == n_states` is the last valid compartment.
#[test]
fn bolus_into_the_last_compartment_is_accepted() {
    let model = model_with(PkModel::TwoCptOral); // depot, central, peripheral
    let population = population_with_doses(vec![bolus(3)]);
    assert!(codes(&model, &population).is_empty());
    let population = population_with_doses(vec![bolus(4)]);
    assert_eq!(codes(&model, &population), vec!["E_DOSE_CMT_OUT_OF_RANGE"]);
}

// ── scope: which engines and models this check owns ──

/// ODE models index the state vector directly, so every compartment the
/// `[odes]` block **declares** is addressable and the analytical routing table
/// must not fire on them — but "declares" is the operative word (#899). A `CMT`
/// past the end of the declared state vector used to fall through the engine's
/// `if cmt_idx < n { … }` guards and be silently dropped: the fit converged and
/// reported a finite OFV having ignored the dose entirely.
///
/// The helper model declares 2 states (`depot`, `central`).
#[test]
fn an_ode_dose_beyond_the_declared_state_count_is_rejected() {
    let model = ode_model(GradientMethod::Fd);
    // One report per (kind, compartment): a bolus and an infusion into the same
    // bad compartment are independent causes.
    let population = population_with_doses(vec![bolus(3), infusion(3)]);
    assert_eq!(
        codes(&model, &population),
        vec!["E_DOSE_CMT_OUT_OF_RANGE", "E_DOSE_CMT_OUT_OF_RANGE"]
    );
}

/// The message must name the states the `[odes]` block actually declares — the
/// whole point of routing this through the ODE spec rather than a `PkModel`
/// topology, which on an ODE model describes a system that is not being solved.
#[test]
fn the_ode_out_of_range_message_names_the_declared_states() {
    let model = ode_model(GradientMethod::Fd);
    let population = population_with_doses(vec![bolus(3)]);
    let diags = check_dose_compartments(&model, &population);
    let msg = &diags[0].message;
    assert!(
        msg.contains("depot"),
        "should name the declared states: {msg}"
    );
    assert!(
        msg.contains("central"),
        "should name the declared states: {msg}"
    );
    assert!(
        msg.contains("only 2 state(s)"),
        "should give the declared state count: {msg}"
    );
    assert_eq!(diags[0].block.as_deref(), Some("odes"));
}

/// Positive control — the check must not be over-broad. Every declared state
/// accepts both a bolus and an infusion: the ODE engine has no infusable subset
/// (it indexes the state vector directly), so there is no ODE analogue of
/// `PkModel::infusable_compartments` to narrow this.
#[test]
fn every_declared_ode_state_accepts_a_bolus_and_an_infusion() {
    let model = ode_model(GradientMethod::Fd);
    let n = model.ode_spec.as_ref().unwrap().n_states;
    assert_eq!(n, 2, "helper model shape assumed by this test");
    for cmt in 1..=n {
        for dose in [bolus(cmt), infusion(cmt)] {
            let population = population_with_doses(vec![dose]);
            assert!(
                codes(&model, &population).is_empty(),
                "ODE cmt {cmt} should be accepted"
            );
        }
    }
}

/// `CMT=0` splits exactly as it does on the analytical engine (#375): it is
/// NONMEM's default dose *compartment*, which resolves to compartment 1 for a
/// **bolus** (accepted — the ODE dose sites resolve it with `saturating_sub(1)`
/// since #899) but has no meaning for a zero-order **input** (rejected). Keeping
/// the split identical across engines is the point — an `[odes]` model and its
/// analytical twin must answer the same dataset the same way.
#[test]
fn ode_cmt_zero_is_a_bolus_default_but_not_an_infusion_target() {
    let model = ode_model(GradientMethod::Fd);
    let population = population_with_doses(vec![infusion(0), bolus(0)]);
    assert_eq!(codes(&model, &population), vec!["E_DOSE_CMT_NOT_INFUSABLE"]);
}

/// The zero-amount split carries over to ODE models too: an inert `AMT=0`
/// infusion opens no rate window and is exempt from the `CMT=0` rule, but the
/// **range** rule still applies to it — the dose sites bound-check the
/// compartment before the amount is read, so an `EVID=4, AMT=0` reset row with a
/// stale `CMT` is exactly as unroutable as a real dose.
#[test]
fn ode_zero_amount_doses_follow_the_same_split_as_analytical() {
    let model = ode_model(GradientMethod::Fd);
    let mut inert_inf = infusion(0);
    inert_inf.amt = 0.0;
    assert!(codes(&model, &population_with_doses(vec![inert_inf])).is_empty());

    let mut inert_bolus = bolus(7);
    inert_bolus.amt = 0.0;
    assert_eq!(
        codes(&model, &population_with_doses(vec![inert_bolus])),
        vec!["E_DOSE_CMT_OUT_OF_RANGE"]
    );
}

/// Transit/IG never take the event-driven walk, so its routing table does not
/// apply to them: an infusion reroutes the subject to the model's ODE twin
/// (`effective_for`, #719 gap 2), which addresses compartments directly.
/// Rejecting here would break a **supported** model — only the SS+infusion
/// combination is rejected, and `check_absorption_closed_form_support` owns
/// that.
#[test]
fn transit_and_ig_infusions_are_not_rejected_by_the_routing_check() {
    for pk_model in [
        PkModel::OneCptTransit,
        PkModel::TwoCptTransit,
        PkModel::OneCptIg,
        PkModel::TwoCptIg,
    ] {
        let model = model_with(pk_model);
        let population = population_with_doses(vec![infusion(1)]);
        assert!(
            codes(&model, &population).is_empty(),
            "{pk_model:?} infusion should defer to the absorption check"
        );
    }
}

/// …but their *bolus* range check still applies (they are 2-state models).
#[test]
fn transit_bolus_range_is_still_checked() {
    let model = model_with(PkModel::OneCptTransit);
    let population = population_with_doses(vec![bolus(3)]);
    assert_eq!(codes(&model, &population), vec!["E_DOSE_CMT_OUT_OF_RANGE"]);
}

// ── reporting shape ──

/// Many offending rows give one actionable error per cause, matching
/// `check_modeled_dose_rates`.
#[test]
fn errors_are_deduped_per_kind_and_compartment() {
    let model = model_with(PkModel::TwoCptOral);
    let population = population_with_doses(vec![
        infusion(0),
        infusion(0),
        infusion(0),
        bolus(9),
        bolus(9),
    ]);
    assert_eq!(
        codes(&model, &population),
        vec!["E_DOSE_CMT_NOT_INFUSABLE", "E_DOSE_CMT_OUT_OF_RANGE"]
    );
}

/// An infusion and a bolus into the same bad compartment are independent
/// causes — reporting only one would hide the other after the user fixes it.
/// Both are out of range on `one_cpt_iv`, so both take the range rule.
#[test]
fn infusion_and_bolus_into_the_same_compartment_report_separately() {
    let model = model_with(PkModel::OneCptIv);
    let population = population_with_doses(vec![infusion(2), bolus(2)]);
    assert_eq!(
        codes(&model, &population),
        vec!["E_DOSE_CMT_OUT_OF_RANGE", "E_DOSE_CMT_OUT_OF_RANGE"]
    );
}

/// …and where the two rules genuinely differ: `CMT=0` is the default dose
/// compartment for a **bolus** (accepted) but has no rate channel for an
/// **infusion** (rejected), so only the infusion is reported.
#[test]
fn the_two_rules_are_distinguished_on_cmt_zero() {
    let model = model_with(PkModel::TwoCptOral);
    let population = population_with_doses(vec![infusion(0), bolus(0)]);
    assert_eq!(codes(&model, &population), vec!["E_DOSE_CMT_NOT_INFUSABLE"]);
}

/// The common dataset — every dose routable — costs one scan and reports
/// nothing.
#[test]
fn a_clean_population_reports_nothing() {
    let model = model_with(PkModel::OneCptOral);
    let population = population_with_doses(vec![bolus(1), infusion(2)]);
    assert!(codes(&model, &population).is_empty());
}

/// A zero-amount dose is a no-op on every path — the walk's bolus branch adds
/// `F·0` and its infusion branch needs `duration > 0`, which `AMT=0` never has.
/// `is_infusion()` still reports `true` for it (it is `rate > 0 || !is_fixed()`,
/// broader than the walk's predicate), so without an explicit skip the check
/// would reject a dataset that previously ran fine and delivered nothing.
#[test]
fn a_zero_amount_dose_is_exempt_from_the_routing_rule_but_not_the_range_rule() {
    let model = model_with(PkModel::TwoCptOral);
    // Exempt: an in-range zero-amount infusion delivers nothing on any path
    // (`duration = AMT/RATE = 0`, and the walk's infusion branch needs
    // `duration > 0`), so failing a whole fit over an inert row would be
    // stricter than the bug being fixed.
    let mut inf = infusion(0); // cmt 0 has no channel for an infusion
    inf.amt = 0.0;
    assert!(codes(&model, &population_with_doses(vec![inf])).is_empty());

    // NOT exempt: both walks bound-check the compartment *before* the amount is
    // read (`if cmt_idx < n_states { … } else { panic! }`), so a zero-amount
    // dose with an out-of-range CMT reaches the panic exactly like a real one —
    // and since #375 routes any `cmt != 1` bolus to the walk, it gets there.
    // Exempting it here made `ferx check` bless the dataset and `fit()` abort.
    let mut bol = bolus(9); // cmt 9 is past the end of the 3-state model
    bol.amt = 0.0;
    assert_eq!(
        codes(&model, &population_with_doses(vec![bol])),
        vec!["E_DOSE_CMT_OUT_OF_RANGE"],
    );
}

// ── wiring ──

/// `fit()` reaches this through `check_model_data`, so the new codes must be in
/// that aggregate rather than only reachable directly.
#[test]
fn check_model_data_surfaces_the_new_error() {
    let model = model_with(PkModel::TwoCptOral);
    let population = population_with_doses(vec![infusion(0)]);
    let diags = crate::api::check_model_data(&model, &population);
    assert!(
        diags.iter().any(|d| d.code == "E_DOSE_CMT_NOT_INFUSABLE"),
        "check_model_data should carry the dose-compartment diagnostic: {diags:?}"
    );
    assert!(crate::diagnostics::first_error(&diags).is_err());
}

// ── #1189 item 1: the fit-init front door for a non-finite dose attribute ──
//
// A `NaN`/infinite `ALAG` or `F` makes `dose.time + lag` — and every break derived from
// it — non-finite, so the subject's integration timeline cannot be ordered. The engines
// return a typed non-finite subject for that (right for a mid-fit θ/η excursion), and
// `check_model_data` rejects it up front when the model is already broken at its starting
// point, naming the subject.
//
// How a lag actually goes non-finite: the DSL's arithmetic is domain-guarded in the obvious
// places (division by ~0 returns 0, `ln`/`sqrt` floor their argument — measured), but `exp`
// is not clamped, so an exponential covariate model on an unscaled covariate
// (`ALAG1 = TVLAG*exp(WT)` with `WT` in the hundreds — a real modelling mistake) overflows
// to `+inf`. Moved here from `tests/dose_attr_finiteness.rs` (#1236): it needs only a
// `CompiledModel` and a hand-built `Population`.

/// Oral 1-cpt ODE whose depot lag is `TVLAG · exp(WT)` — finite for a small `WT`,
/// `+inf` once `WT` is large enough to overflow the exponential.
const EXP_LAG_MODEL: &str = r#"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVLAG(0.3, 0.0, 12.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL    = TVCL * exp(ETA_CL)
  V     = TVV
  ALAG1 = TVLAG * exp(WT)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

fn exp_lag_model() -> CompiledModel {
    crate::parser::model_parser::parse_full_model(EXP_LAG_MODEL)
        .expect("the exp-lag model parses")
        .model
}

/// One subject with a bolus into the depot and covariate `WT`.
fn population_with_wt(wt: f64) -> Population {
    let mut pop = population_with_doses(vec![bolus(1)]);
    pop.subjects[0].covariates.insert("WT".to_string(), wt);
    pop.covariate_names = vec!["WT".to_string()];
    pop
}

/// A non-finite `ALAG` at typical values is rejected before the fit starts, naming the
/// subject — instead of the panic it used to be, or the all-`NaN` fit the engine guard
/// alone would produce.
///
/// Mutation: delete the `check_dose_attr_finiteness` call from `check_model_data` and no
/// diagnostic is raised.
#[test]
fn a_non_finite_lagtime_at_typical_values_is_rejected_at_fit_init() {
    let model = exp_lag_model();
    let pop = population_with_wt(1000.0);
    // Sanity: the fixture really is non-finite, so this cannot pass on a mis-built model.
    assert!(
        (0.3f64 * 1000.0f64.exp()).is_infinite(),
        "the fixture must overflow, or there is nothing to reject"
    );
    let diags = crate::api::check_model_data(&model, &pop);
    let hit = diags
        .iter()
        .find(|d| d.code == "E_DOSE_ATTR_NONFINITE")
        .unwrap_or_else(|| {
            panic!(
                "expected E_DOSE_ATTR_NONFINITE, got {:?}",
                diags.iter().map(|d| &d.code).collect::<Vec<_>>()
            )
        });
    assert!(
        hit.message.contains("subject 1"),
        "the diagnostic must name the offending subject: {}",
        hit.message
    );
    assert!(
        hit.message.contains("lag time"),
        "the diagnostic must name which attribute went non-finite: {}",
        hit.message
    );
}

/// The control: the *same model* with an ordinary covariate is accepted. Without this the
/// test above would also pass if the check fired on every model, which would break every
/// lagged fit in the repo.
#[test]
fn a_finite_lagtime_is_not_rejected() {
    let model = exp_lag_model();
    let pop = population_with_wt(0.5);
    let diags = crate::api::check_model_data(&model, &pop);
    assert!(
        !diags.iter().any(|d| d.code == "E_DOSE_ATTR_NONFINITE"),
        "a finite lag must not be rejected, got {:?}",
        diags.iter().map(|d| &d.code).collect::<Vec<_>>()
    );
}

// ── #2151: one consequence for every dose attribute, and the causes it names ──────────
//
// Since #1284 a non-finite modeled `D{n}`/`R{n}` repels the subject exactly like a lag or
// `F` (`DoseEvent::resolve_rate` returns a dose with a `NaN` time). The message used to
// say the opposite for `D`/`R` — that the value "never reaches you as a NaN" and the fit
// returns finite, silently wrong numbers — and named engine internals (`resolve_rate`,
// `DURATION_FLOOR`, `RATE_FLOOR`) a model author cannot act on.

/// 1-cpt ODE whose compartment-1 dose attribute `attr` (`D1`, `R1` or `ALAG1`) is `rhs`.
fn dose_attr_model(attr: &str, rhs: &str) -> CompiledModel {
    let src = format!(
        r#"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVA(2.0, 0.0, 120.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  {attr} = {rhs}
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#
    );
    crate::parser::model_parser::parse_full_model(&src)
        .unwrap_or_else(|e| panic!("the {attr} model parses: {e:?}"))
        .model
}

/// One subject with one dose into compartment 1 coded `mode`, and covariate `WT`.
fn population_with_coded_dose(mode: crate::types::RateMode, wt: f64) -> Population {
    let dose = DoseEvent::modeled(0.0, 100.0, 1, false, 0.0, mode);
    let mut pop = population_with_doses(vec![dose]);
    pop.subjects[0].covariates.insert("WT".to_string(), wt);
    pop.covariate_names = vec!["WT".to_string()];
    pop
}

fn dose_attr_nonfinite_message(model: &CompiledModel, pop: &Population) -> Option<String> {
    crate::api::check_model_data(model, pop)
        .into_iter()
        .find(|d| d.code == "E_DOSE_ATTR_NONFINITE")
        .map(|d| d.message)
}

/// The message's consequence and cause sentences, asserted for every attribute. Each
/// assertion is one sentence (or one clause of the cause list) — delete it from the
/// message and this dies naming it.
fn assert_consequence_and_causes(what: &str, msg: &str) {
    for (needle, why) in [
        (
            "makes the subject's whole integration timeline non-finite and every prediction NaN",
            "the consequence: the subject's predictions are NaN",
        ),
        (
            "the subject is repelled as a diverged solve",
            "the consequence: the subject is repelled",
        ),
        (
            "a covariate missing at that record (a missing value is NaN, and NaN passes \
             through arithmetic, `^` and `exp`)",
            "the missing-covariate cause",
        ),
        (
            "a fractional power of a negative value",
            "the negative-power cause",
        ),
        ("overflowing `exp(...)`", "the exp-overflow cause"),
        (
            "Check this subject's covariate values at that record",
            "the remedy",
        ),
    ] {
        assert!(
            msg.contains(needle),
            "{what}: message must state {why}: {msg}"
        );
    }
    for stale in [
        "never reaches you",
        "silently wrong",
        "resolve_rate",
        "DURATION_FLOOR",
        "RATE_FLOOR",
        "bolus",
    ] {
        assert!(
            !msg.contains(stale),
            "{what}: message must not carry the pre-#1284 claim / internal `{stale}`: {msg}"
        );
    }
}

/// A non-finite modeled `D1` behind `RATE=-2` and a non-finite `R1` behind `RATE=-1` are
/// each rejected, and the message states the repelled-subject consequence, not the
/// pre-#1284 "finite, silently wrong numbers" one (#2151).
///
/// Mutation: restore the old `CLAMPED_DURATION` / `CLAMPED_RATE` consequence on either
/// arm and that arm's case dies on "never reaches you".
#[test]
fn a_non_finite_modeled_duration_or_rate_names_the_repelled_subject() {
    use crate::types::RateMode;
    for (attr, mode, what) in [
        ("D1", RateMode::ModeledDuration, "modeled infusion duration"),
        ("R1", RateMode::ModeledRate, "modeled infusion rate"),
    ] {
        let model = dose_attr_model(attr, "TVA * exp(WT)");
        let msg = dose_attr_nonfinite_message(&model, &population_with_coded_dose(mode, 1000.0))
            .unwrap_or_else(|| panic!("{attr} = 2*exp(1000) = inf must be rejected"));
        assert!(msg.contains(what), "{attr}: must name the attribute: {msg}");
        assert!(msg.contains("is inf"), "{attr}: must name the value: {msg}");
        assert_consequence_and_causes(attr, &msg);
        // The control: the same model and dose with a finite value is accepted, so the
        // check above is not firing on every coded dose.
        assert!(
            dose_attr_nonfinite_message(&model, &population_with_coded_dose(mode, 0.5)).is_none(),
            "{attr}: a finite value must not be rejected"
        );
    }
}

/// The two causes that need no overflow — a missing covariate (`NaN`) and a fractional
/// power of a negative covariate — each reach the check on an allometric lag, the
/// everyday spelling. Together with the overflow case above this is the message's whole
/// cause list, each with a fixture that actually produces it.
#[test]
fn a_missing_or_negative_covariate_under_a_power_is_rejected_on_the_same_message() {
    let model = dose_attr_model("ALAG1", "TVA * (WT / 70)^0.75");
    for (wt, case) in [(f64::NAN, "missing WT"), (-70.0, "negative WT")] {
        assert!(
            (2.0 * (wt / 70.0f64).powf(0.75)).is_nan(),
            "{case}: the fixture must be NaN, or there is nothing to reject"
        );
        let msg = dose_attr_nonfinite_message(&model, &population_with_wt(wt))
            .unwrap_or_else(|| panic!("{case}: ALAG1 = NaN must be rejected"));
        assert!(
            msg.contains("lag time"),
            "{case}: must name the attribute: {msg}"
        );
        assert!(msg.contains("is NaN"), "{case}: must name the value: {msg}");
        assert_consequence_and_causes(case, &msg);
    }
    assert!(
        dose_attr_nonfinite_message(&model, &population_with_wt(70.0)).is_none(),
        "a positive WT gives a finite lag and must not be rejected"
    );
}

/// The bound on #1196's behaviour change. Unifying the two infusion resolvers on one
/// membership predicate changes what a `CMT=0` or out-of-range **infusion** contributes —
/// `active_infusions` used to admit both and `gated_infusions` dropped them. That cannot
/// affect any `fit()` / `predict()` result because validation rejects both compartments
/// (#899); the change is confined to hand-built `OdeSpec`s
/// (`residual_ss_infusion_window_obeys_the_membership_predicate` pins what it does there).
/// If this ever stops holding, the unification needs a user-facing note it does not
/// currently carry.
#[test]
fn validation_still_rejects_the_infusion_compartments_the_resolvers_used_to_differ_on() {
    let model = exp_lag_model();
    for (cmt, why) in [(0usize, "CMT=0"), (7usize, "cmt > n_states")] {
        let mut pop = population_with_wt(0.5);
        pop.subjects[0].doses = vec![DoseEvent::new(0.0, 100.0, cmt, 25.0, false, 0.0)];
        let diags = crate::api::check_model_data(&model, &pop);
        assert!(
            diags.iter().any(|d| d.severity == crate::Severity::Error),
            "an infusion with {why} must still be rejected, got {:?}",
            diags.iter().map(|d| &d.code).collect::<Vec<_>>()
        );
    }
}
