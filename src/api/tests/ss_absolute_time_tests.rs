//! `W_STEADY_STATE_ABSOLUTE_TIME` (#1139, batch-T step T3): an `SS=1` dose on an `[odes]`
//! right-hand side that reads an **absolute** clock — `TAFD`, `T`/`t`, or the bare `TIME`
//! built-in.
//!
//! The steady-state run-in expands the periodic train on a clock local to each cycle, so an
//! absolute clock has nothing to converge to. Measured at `0596fbb9` on a 1-cpt IV bolus
//! (`CL = 1`, `V = 20`, `AMT = 100`, `II = 12`, one `SS=1` record at `t = 480`, samples at
//! 482/485/488/491): `0.003*T` returns a finite `8.5679354785102` — matching NONMEM's own
//! steady-state routine, and 67 % from the same model's explicit 41-dose train — while
//! `0.003*TAFD` returns `NaN` at every observation and a `NaN` objective, `0.0*TAFD`
//! included. `0.03*TAD`, by contrast, reproduces its own train to 3.8e-13 (#1139 T2).
//!
//! Before this warning existed, nothing named the cause. The `TAFD` fit returned five
//! warnings, of which the only two about the failure were `W_ODE_SOLVER_DIAGNOSTICS`
//! (`14400 step(s) clamped at the minimum step size`, plus 225 freeze-padded segments) and
//! `Covariance step failed: base OFV is non-finite at convergence` — both of which blame the
//! solver rather than the run-in. The other three are routine operational notes that fire
//! regardless (finite-difference inner gradients, and a thread-count hint). `ferx check
//! --data` on the same model returned **exactly one** diagnostic once this landed and
//! **zero** before it. All measured, not recalled.
//!
//! # Tiering
//!
//! Tier 1. Every case here is one parse plus one pass over the check functions; nothing
//! integrates an ODE or calls `fit()`.

use super::{check_model_data, check_model_data_warnings};
use crate::parser::model_parser::parse_model_string;
use crate::types::{DoseEvent, Population, RateMode, Subject};

const CODE: &str = "W_STEADY_STATE_ABSOLUTE_TIME";

/// A 1-cpt `[odes]` model whose elimination carries `term`, e.g. `" + 0.003*TAFD"`.
fn ode_model(term: &str) -> crate::types::CompiledModel {
    parse_model_string(&format!(
        "[parameters]\n  theta TVCL(1.0, 0.001, 100.0)\n  theta TVV(20.0, 0.001, 500.0)\n  \
         omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.1 (sd)\n[individual_parameters]\n  CL = TVCL * \
         exp(ETA_CL)\n  V  = TVV\n[structural_model]\n  ode(obs_cmt=central, \
         states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central * (1.0{term})\n  \
         \n[error_model]\n  DV ~ proportional(PROP)\n"
    ))
    .expect("parse")
}

/// The analytical twin of [`ode_model`] — no `[odes]` block, so no RHS to read a clock.
fn analytical_model() -> crate::types::CompiledModel {
    parse_model_string(
        "[parameters]\n  theta TVCL(1.0, 0.001, 100.0)\n  theta TVV(20.0, 0.001, 500.0)\n  \
         omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.1 (sd)\n[individual_parameters]\n  CL = TVCL * \
         exp(ETA_CL)\n  V  = TVV\n[structural_model]\n  pk one_cpt_iv(cl=CL, \
         v=V)\n[error_model]\n  DV ~ proportional(PROP)\n",
    )
    .expect("parse")
}

/// `n` subjects, each carrying one dose at `t = 480` with the given `SS` flag and `II`.
fn population(n: usize, ss: bool, ii: f64) -> Population {
    let subjects = (0..n)
        .map(|i| Subject {
            id: format!("{}", i + 1),
            obs_times: vec![482.0, 485.0],
            observations: vec![5.0, 4.0],
            obs_cmts: vec![1, 1],
            cens: vec![0, 0],
            doses: vec![DoseEvent::new(480.0, 100.0, 1, 0.0, ss, ii)],
            ..Default::default()
        })
        .collect();
    Population {
        subjects,
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

/// One subject carrying one `SS=1, II=12` dose written as an infusion of the given `rate`
/// into the given (1-based) compartment. `rate = 0.0` is a bolus.
fn population_with(rate: f64, cmt: usize) -> Population {
    let mut pop = population(1, true, 12.0);
    pop.subjects[0].doses = vec![DoseEvent::new(480.0, 100.0, cmt, rate, true, 12.0)];
    pop
}

/// The `W_STEADY_STATE_ABSOLUTE_TIME` message, if the check raised one.
fn warning_for(model: &crate::types::CompiledModel, pop: &Population) -> Option<String> {
    let init = model.default_params.clone();
    check_model_data_warnings(model, pop, &init)
        .into_iter()
        .find(|d| d.code == CODE)
        .map(|d| d.message)
}

fn warning(term: &str) -> Option<String> {
    warning_for(&ode_model(term), &population(1, true, 12.0))
}

/// Whether the check raised the given code.
fn raised(model: &crate::types::CompiledModel, pop: &Population, code: &str) -> bool {
    let init = model.default_params.clone();
    check_model_data_warnings(model, pop, &init)
        .iter()
        .any(|d| d.code == code)
}

/// Every absolute-clock spelling fires, and the message says which failure the model has.
///
/// The two consequences are asserted **positively and negatively** — a message naming both
/// would otherwise pass either arm, and the sentence choice is the only thing distinguishing
/// them (#1255). `T`, `t` and `TIME` are all present because they do not reach the flag the
/// same way: `T`/`t` resolve through `time_slot`, a bare `TIME` only through `Op::PushTime`
/// (#1124), so a `T`-only fixture cannot see the second disjunct.
#[test]
fn every_absolute_clock_spelling_warns_and_names_its_own_consequence() {
    for term in [" + 0.003*T", " + 0.003*t", " + 0.003*TIME"] {
        let m = warning(term).unwrap_or_else(|| panic!("`{term}` must warn"));
        assert!(
            m.contains("`T`/`TIME` therefore return a finite number"),
            "`{term}` must report the finite-but-wrong consequence: {m}"
        );
        assert!(
            !m.contains("reads NaN"),
            "`{term}` must not report `TAFD`'s consequence: {m}"
        );
        assert!(
            m.contains("(`T`/`TIME`)"),
            "`{term}` names its spelling: {m}"
        );
    }

    let m = warning(" + 0.003*TAFD").expect("`TAFD` must warn");
    assert!(
        m.contains("`TAFD` has no referent inside the run-in"),
        "TAFD must report the NaN consequence: {m}"
    );
    assert!(
        !m.contains("matching NONMEM"),
        "TAFD must not claim NONMEM parity, which it does not have: {m}"
    );
    assert!(m.contains("(`TAFD`)"), "TAFD names its spelling: {m}");

    // Both spellings at once: both sentences, and the joint spelling list.
    let m = warning(" + 0.003*TAFD + 0.003*TIME").expect("both must warn");
    assert!(
        m.contains("`T`/`TIME` therefore return a finite number")
            && m.contains("`TAFD` has no referent inside the run-in"),
        "a model reading both spellings must report both consequences: {m}"
    );
    assert!(m.contains("(`TAFD` and `T`/`TIME`)"), "joint spelling: {m}");
}

/// **The discriminating negative.** `TAD` is model time but not an absolute clock: it is
/// bounded inside one dosing interval, so the run-in *does* have a periodic limit and #1139
/// T2 anchored it against NONMEM. A gate wired to `pk_reads_model_time` instead of
/// `pk_reads_absolute_time` passes every other test in this file and fails only this one.
///
/// `0.0*TAD` is included because it is the case #1139 opens with — before T2 merely
/// mentioning `TAD` returned `NaN` — and a warning appearing on it now would read as that
/// bug coming back under a new name.
#[test]
fn a_tad_reading_rhs_does_not_warn() {
    for term in [" + 0.03*TAD", " + 0.0*TAD"] {
        assert_eq!(
            warning(term),
            None,
            "`{term}` under SS=1 is anchored since #1139 T2 and must not be warned about"
        );
    }
    assert_eq!(warning(""), None, "an autonomous RHS must not warn");
    // …and mentioning `TAD` must not suppress an absolute read sitting beside it.
    assert!(
        warning(" + 0.03*TAD + 0.003*TIME").is_some(),
        "`TAD` alongside `TIME` must still warn — the flag is a union, not a classification"
    );
}

/// **The conjunct that `has_ss_doses` would not carry.** With `II <= 0` the steady-state
/// branch is never entered — the dose falls through to the single-dose path, which is what
/// `W_STEADY_STATE_II` reports — so there is no run-in whose clock could be wrong. Measured
/// at `0596fbb9`: the same `0.003*TAFD` model that returns a `NaN` objective under
/// `SS=1, II=12` returns a finite `112.3538` under `SS=1, II=0`.
///
/// This is the **only** input separating `has_periodic_ss_dose()` from `has_ss_doses()`.
/// Without it the SS conjunct could be weakened to the flag alone and nothing would notice.
#[test]
fn a_steady_state_dose_with_no_interval_does_not_warn() {
    let m = ode_model(" + 0.003*TAFD");
    assert_eq!(
        warning_for(&m, &population(1, true, 0.0)),
        None,
        "SS=1 with II = 0 never enters the run-in"
    );
    // The same subject *does* warn once the interval is positive — so the row above is a
    // statement about `II`, not about the fixture being inert.
    assert!(
        warning_for(&m, &population(1, true, 12.0)).is_some(),
        "the same model with II > 0 must warn"
    );
    // A dose that is not flagged `SS` at all, likewise.
    assert_eq!(
        warning_for(&m, &population(1, false, 12.0)),
        None,
        "II > 0 without the SS flag is an ordinary dose"
    );
}

/// **A dose that never reaches the run-in must not be reported**, because the message names
/// what the run-in does and would be false about it.
///
/// `equilibrate_ss_pk_state` returns before integrating anything for a dose whose compartment
/// index is outside the state vector (#899); the ordinary finite `TAFD` anchor stays in place.
///
/// An infusion whose `T_inf` exceeds its own `II` used to be the second such case (served as a
/// single non-SS infusion). Since #2786 an overlapping train equilibrates like any other SS
/// infusion (⌊T_inf/II⌋ always-running pulses plus one ordinary train), so its run-in reads the
/// clock and it MUST warn — that row is now the straddle's other half.
///
/// Both rows are straddles: the same dose on the model's one real compartment does warn, so the
/// out-of-state row cannot pass against a gate that has simply stopped firing on this fixture.
#[test]
fn a_dose_that_never_reaches_the_run_in_is_not_reported() {
    let m = ode_model(" + 0.003*TAFD");

    assert!(
        warning_for(&m, &population_with(5.0, 1)).is_some(),
        "T_inf = 20 > II = 12: an overlapping SS infusion equilibrates through the run-in (#2786), \
         so it reads the clock and must warn"
    );
    assert!(
        warning_for(&m, &population_with(20.0, 1)).is_some(),
        "T_inf = 5 <= II = 12 equilibrates normally and must warn"
    );

    // `ode_model` declares a single state, so `CMT = 2` is outside the state vector.
    assert_eq!(
        warning_for(&m, &population_with(0.0, 2)),
        None,
        "a dose outside the state vector returns the unequilibrated zero state (#899)"
    );
    assert!(
        warning_for(&m, &population_with(0.0, 1)).is_some(),
        "the same bolus on the one real compartment must warn"
    );
}

/// An analytical model has no `[odes]` right-hand side to read a clock, so the gate's first
/// conjunct excludes it structurally rather than by luck.
#[test]
fn an_analytical_model_does_not_warn() {
    assert_eq!(
        warning_for(&analytical_model(), &population(1, true, 12.0)),
        None,
        "no `[odes]` block, no run-in clock to get wrong"
    );
}

/// The message counts subjects, and it is a **warning**, not an error — `ferx check` must
/// still call the model valid and `fit()` must still run.
#[test]
fn the_finding_is_a_warning_that_counts_subjects() {
    let m = ode_model(" + 0.003*TAFD");
    let pop = population(3, true, 12.0);
    let msg = warning_for(&m, &pop).expect("must warn");
    assert!(
        msg.starts_with("3 subject(s) have an SS=1 dose"),
        "the count must be the number of subjects carrying a periodic SS dose: {msg}"
    );
    // Only two of the three carry one — and the *second* of those carries two, so the count
    // cannot be a dose count wearing a subject count's label. Without the extra dose both
    // readings give 2 and a `flat_map(|s| &s.doses).filter(...)` mutation survives.
    let mut mixed = population(3, true, 12.0);
    mixed.subjects[1]
        .doses
        .push(DoseEvent::new(492.0, 100.0, 1, 0.0, true, 12.0));
    mixed.subjects[2].doses[0] = DoseEvent::new(480.0, 100.0, 1, 0.0, false, 12.0);
    let msg = warning_for(&m, &mixed).expect("must warn");
    assert!(
        msg.starts_with("2 subject(s) "),
        "two subjects carry a periodic SS dose, one of them twice — three SS doses in all: \
         {msg}"
    );

    let init = m.default_params.clone();
    let d = check_model_data_warnings(&m, &pop, &init)
        .into_iter()
        .find(|d| d.code == CODE)
        .expect("must warn");
    assert!(!d.is_error(), "must be Warning severity, not Error");
    assert_eq!(d.block.as_deref(), Some("odes"));
    assert!(
        d.suggestion.is_some(),
        "must carry a remediation suggestion"
    );
    // And it rides the *warning* bundle, not the fatal one — `fit()` consumes that one
    // through `first_error` and would refuse the model.
    assert!(
        !check_model_data(&m, &pop).iter().any(|d| d.code == CODE),
        "this must not appear in the fatal bundle"
    );
}

/// **The message's classification is load-bearing and is not asserted by its category
/// alone.** `classify_warning`'s `DataQuality` arm serves six different substrings, so
/// `code == DataQuality` passes for every input that arm accepts (#1255). What has to hold
/// is that *this* message reaches it through `SS=1 dose` — the phrase the neighbouring
/// steady-state warnings use, and the category ferx-r already carries guidance for.
///
/// Also pinned: the message must **not** contain the literal
/// `Steady-state (SS=1) equilibration`, which `tests/ss_chz_nonmem_anchor.rs` filters
/// `FitResult.warnings` on to assert that a joint steady-state fit reports no
/// non-convergence. A phrasing collision there would redden a #1210 anchor for a reason
/// having nothing to do with #1210.
#[test]
fn the_message_reaches_the_data_quality_category_through_its_ss_phrase() {
    use crate::types::{classify_warning, WarningCode};

    let msg = warning(" + 0.003*TAFD").expect("must warn");
    assert_eq!(
        classify_warning(&msg).category,
        WarningCode::DataQuality,
        "message must classify as data_quality: {msg}"
    );

    // The discriminator: remove the routing phrase and the classification must change.
    // Without this the assertion above would hold for any message the arm happens to accept.
    let defanged = msg.replace("SS=1 dose", "steady-state record");
    assert_ne!(
        classify_warning(&defanged).category,
        WarningCode::DataQuality,
        "`SS=1 dose` is what routes this message; if it classifies without the phrase, this \
         test is not pinning the route it claims to"
    );

    assert!(
        !msg.contains("Steady-state (SS=1) equilibration"),
        "reserved phrase — `tests/ss_chz_nonmem_anchor.rs` filters on it: {msg}"
    );
}

/// **The #1210 case must stay silent.** A Gompertz or Weibull baseline hazard reads `TIME`
/// by definition, and `nonmem_anchor/ss_chz_tdep_fit.ferx` is exactly that model under
/// `SS=1`. The run-in masks the accumulator's derivative and restores its record value, so a
/// time-reading hazard cannot disturb the PK equilibration — warning here would be a false
/// positive on the standard joint PK-TTE model.
///
/// This is the row that dies if the gate is wired to `reads_model_time` (the whole augmented
/// program) rather than the PK-block-only predicate.
#[cfg(feature = "survival")]
#[test]
fn a_time_reading_hazard_alone_does_not_warn() {
    let m = joint_model("", "H0 * exp(0.05*TIME) * exp(BETA * (central / V))");
    assert_eq!(
        warning_for(&m, &population(1, true, 12.0)),
        None,
        "a time-dependent hazard leaves the PK block autonomous (#1166)"
    );
    // The straddle: the same joint model with the time read moved into the PK block *does*
    // warn. Without this the row above would also pass against a gate that never fires on a
    // joint model at all.
    let m = joint_model(" * (1.0 + 0.003*TIME)", "H0 * exp(BETA * (central / V))");
    assert!(
        warning_for(&m, &population(1, true, 12.0)).is_some(),
        "`TIME` in the PK RHS of a joint model is a genuine absolute-clock read"
    );
}

/// **The intermediate spelling of the #1210 case is silent too (#1279).** Factoring the
/// hazard's `TIME` read out into an `[odes]` intermediate (`TT = TIME` … `hazard = f(TT)`)
/// is not a property of the model, so it must agree with the inline spelling in
/// `a_time_reading_hazard_alone_does_not_warn` — together they are the straddle.
///
/// Until #1279 this test was `…_still_warns` and asserted the opposite: #1166's PK-row filter
/// dropped only the injected `d/dt(__chz_*)` lines, kept every intermediate, and so this
/// autonomous PK block was warned about. The view is now a dataflow cut, and the guard that
/// the cut did not go too far is the second half: the same intermediate read by the PK RHS
/// still warns.
#[cfg(feature = "survival")]
#[test]
fn a_time_reading_intermediate_used_only_by_the_hazard_does_not_warn() {
    let m =
        joint_model_with_intermediate("TT = TIME", "H0 * exp(0.05*TT) * exp(BETA * (central / V))");
    assert_eq!(
        warning_for(&m, &population(1, true, 12.0)),
        None,
        "an intermediate read only by the hazard leaves the PK block autonomous (#1279)"
    );
    let m = joint_src(
        " * (1.0 + 0.003*TT)",
        "TT = TIME",
        "H0 * exp(BETA * (central / V))",
    );
    assert!(
        warning_for(&m, &population(1, true, 12.0)).is_some(),
        "the same intermediate read by the PK RHS is a genuine absolute-clock read"
    );
}

#[cfg(feature = "survival")]
fn joint_model(pk: &str, haz: &str) -> crate::types::CompiledModel {
    joint_src(pk, "", haz)
}

/// The same 1-cpt `[odes]` model as [`ode_model`], with a bioavailability on compartment 1.
fn ode_model_with_f(term: &str, f: &str) -> crate::types::CompiledModel {
    parse_model_string(&format!(
        "[parameters]\n  theta TVCL(1.0, 0.001, 100.0)\n  theta TVV(20.0, 0.001, 500.0)\n  \
         omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.1 (sd)\n[individual_parameters]\n  CL = TVCL * \
         exp(ETA_CL)\n  V  = TVV\n  F1 = {f}\n[structural_model]\n  ode(obs_cmt=central, \
         states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central * (1.0{term})\n  \
         \n[error_model]\n  DV ~ proportional(PROP)\n"
    ))
    .expect("parse")
}

/// The same model with a **modeled duration** (`RATE = -2` → `D1`), so `F` reshapes the rate
/// and leaves the infusion's length alone (`InfusionDef::DurationDefined`, #419).
fn ode_model_with_d1(term: &str, f: &str, d1: &str) -> crate::types::CompiledModel {
    parse_model_string(&format!(
        "[parameters]\n  theta TVCL(1.0, 0.001, 100.0)\n  theta TVV(20.0, 0.001, 500.0)\n  \
         theta TVD1({d1}, 0.1, 100.0)\n  omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.1 \
         (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV\n  D1 = TVD1\n  \
         F1 = {f}\n[structural_model]\n  ode(obs_cmt=central, states=[central])\n[odes]\n  \
         d/dt(central) = -(CL/V) * central * (1.0{term})\n  \n[error_model]\n  DV ~ \
         proportional(PROP)\n"
    ))
    .expect("parse")
}

/// **Every `SS=1` infusion reaches the run-in, overlapping pulses (`T_inf > II`) included**,
/// so the absolute-clock warning counts it. The run-in used to bail on
/// `bioavailable_infusion(F).1 > II`, serving the record as a single non-SS infusion under a
/// `W_STEADY_STATE_INFUSION` warning, and this gate had to follow that bail through `F` (#1281):
/// on a 20-hour infusion into `II = 12` the `TAFD`-reading predictions were finite at `F1 = 1`
/// (`9.514379031181768`, `22.093161802029634`) and `NaN` at `F1 = 0.5`. Now the overlap is
/// equilibrated like any other steady state (`dosing::ss_overlap_split_g`), so both are `NaN` —
/// the run-in's cycle-local clock has no first dose — and the warning fires on both, on a
/// rate-defined window and on a duration-defined (`RATE = -2`) one alike.
///
/// Tier 1 in cost — two observations on one subject — but it does call
/// [`crate::api::predict`], because the property under test is *agreement with the
/// integrator*, and a check-pass-only assertion would pin the gate against itself.
#[test]
fn an_overlapping_steady_state_infusion_reaches_the_run_in() {
    // `AMT = 100, RATE = 5` is a 20-hour infusion into an `II = 12` interval.
    let pop = population_with(5.0, 1);
    let mut pop_d1 = population(1, true, 12.0);
    pop_d1.subjects[0].doses = vec![DoseEvent::modeled(
        480.0,
        100.0,
        1,
        true,
        12.0,
        RateMode::ModeledDuration,
    )];
    for (label, model, pop) in [
        ("F1 = 1", ode_model_with_f(" + 0.003*TAFD", "1.0"), &pop),
        ("F1 = 0.5", ode_model_with_f(" + 0.003*TAFD", "0.5"), &pop),
        (
            "RATE = -2, D1 = 20",
            ode_model_with_d1(" + 0.003*TAFD", "0.5", "20.0"),
            &pop_d1,
        ),
    ] {
        let preds = crate::api::predict(&model, pop, &model.default_params.clone()).unwrap();
        assert!(
            preds.iter().all(|p| p.pred.is_nan()),
            "{label}: the run-in runs, and TAFD has no referent in it: {:?}",
            preds.iter().map(|p| p.pred).collect::<Vec<_>>()
        );
        assert!(warning_for(&model, pop).is_some(), "{label}: must warn");
        assert!(
            !raised(&model, pop, "W_STEADY_STATE_INFUSION"),
            "{label}: the overlap warning is retired"
        );
    }
}

#[cfg(feature = "survival")]
fn joint_model_with_intermediate(pre: &str, haz: &str) -> crate::types::CompiledModel {
    joint_src("", pre, haz)
}

#[cfg(feature = "survival")]
fn joint_src(pk: &str, pre: &str, haz: &str) -> crate::types::CompiledModel {
    parse_model_string(&format!(
        "[parameters]\n  theta TVCL(1.0, 0.01, 100.0)\n  theta TVV(20.0, 0.1, 500.0)\n  theta \
         TVH0(0.02, 1e-5, 10.0)\n  theta TVBETA(0.5, -10.0, 10.0)\n  omega ETA_CL ~ 0.09\n  \
         sigma PROP ~ 0.1 (sd)\n[individual_parameters]\n  CL   = TVCL * exp(ETA_CL)\n  V    = \
         TVV\n  H0   = TVH0\n  BETA = TVBETA\n[structural_model]\n  ode(obs_cmt=central, \
         states=[central])\n[odes]\n  {pre}\n  d/dt(central) = -(CL/V) * \
         central{pk}\n[event_model]\n  cmt    = 3\n  hazard = {haz}\n[error_model]\n  DV ~ \
         proportional(PROP)\n"
    ))
    .expect("joint model must parse")
}
