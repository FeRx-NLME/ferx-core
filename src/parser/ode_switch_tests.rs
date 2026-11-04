use super::super::{
    assigned_vars_in_order, parse_block_statements, parse_model_string, ParseCtx, StatementMode,
};
use super::*;
use crate::sens::ode_provider::adjoint::{outer_model_in_scope, recorder_in_scope};

const STATES: &[&str] = &["a", "tau"];
const PARAMS: &[&str] = &["KE", "LAGT"];

/// Parse an `[odes]` body the way `build_ode_spec` does, then report its conditions.
fn reads(body: &str) -> Vec<ConditionReads> {
    let states: Vec<String> = STATES.iter().map(|s| s.to_string()).collect();
    let params: Vec<String> = PARAMS.iter().map(|s| s.to_string()).collect();
    let mut defined: Vec<String> = states.iter().chain(&params).cloned().collect();
    let pre = parse_block_statements(body, ParseCtx::ode(&defined), StatementMode::Ode)
        .expect("fixture parses");
    for v in assigned_vars_in_order(&pre) {
        if !defined.contains(&v) {
            defined.push(v);
        }
    }
    let stmts = parse_block_statements(body, ParseCtx::ode(&defined), StatementMode::Ode)
        .expect("fixture parses");
    conditions_reading_params(
        &stmts,
        &SwitchInputs {
            states: &states,
            params: &params,
            dose_params: Vec::new(),
            lag_params: Vec::new(),
            forcing_params: Vec::new(),
            inits: &[],
        },
    )
}

fn lagt(moves: bool) -> Vec<ConditionReads> {
    vec![ConditionReads {
        params: vec!["LAGT".to_string()],
        moves,
    }]
}

#[test]
fn a_condition_over_a_parameter_and_something_varying_moves() {
    for body in [
        "d/dt(a) = if (tau > LAGT) -KE * a else 0.0\nd/dt(tau) = 1.0",
        "d/dt(a) = if (TIME > LAGT) -KE * a else 0.0\nd/dt(tau) = 1.0",
        "d/dt(a) = if (T > LAGT) -KE * a else 0.0\nd/dt(tau) = 1.0",
        "d/dt(a) = if (TAD > LAGT) -KE * a else 0.0\nd/dt(tau) = 1.0",
        "d/dt(a) = if (TAFD > LAGT) -KE * a else 0.0\nd/dt(tau) = 1.0",
        // Names resolve case-insensitively, and the report keeps the declared one.
        "d/dt(a) = if (TAU > lagt) -KE * a else 0.0\nd/dt(tau) = 1.0",
        // The statement form, in a later arm as well as the first.
        "if (tau > LAGT) {\n  d/dt(a) = -KE * a\n} else {\n  d/dt(a) = 0.0\n}\nd/dt(tau) = 1.0",
        "if (tau > 99.0) {\n  d/dt(a) = 0.0\n} else if (tau > LAGT) {\n  d/dt(a) = -KE * a\n}\n\
         d/dt(tau) = 1.0",
        // Through an intermediate, on either side of the comparison.
        "L = 2.0 * LAGT\nd/dt(a) = if (tau > L) -KE * a else 0.0\nd/dt(tau) = 1.0",
        "X = tau - LAGT\nd/dt(a) = if (X > 0.0) -KE * a else 0.0\nd/dt(tau) = 1.0",
        // Read before its assignment: the previous evaluation's value.
        "d/dt(a) = if (tau > L) -KE * a else 0.0\nL = LAGT\nd/dt(tau) = 1.0",
        // Held by an intermediate only through the arm that assigned it.
        "if (LAGT > 1.0) {\n  G = 1.0\n} else {\n  G = 0.0\n}\n\
         d/dt(a) = if (tau > G) -KE * a else 0.0\nd/dt(tau) = 1.0",
        // Inside a `max`: the select is continuous, the `if` in its operand is not.
        "d/dt(a) = max(if (tau > LAGT) -KE * a else 0.0, -1.0)\nd/dt(tau) = 1.0",
    ] {
        let got = reads(body);
        assert!(
            got.contains(&lagt(true)[0]),
            "expected a moving LAGT condition in:\n{body}\ngot {got:?}"
        );
    }
}

/// `LAGT` alone decides the arm once per subject, so nothing switches mid-integration. The
/// report still names it, because a parameter reading `TIME` changes the answer.
#[test]
fn a_condition_over_parameters_alone_does_not_move() {
    let got = reads("d/dt(a) = if (LAGT > 2.0) -KE * a else 0.0\nd/dt(tau) = 1.0");
    assert_eq!(got, lagt(false));
}

#[test]
fn conditions_without_a_parameter_are_not_reported() {
    for body in [
        "d/dt(a) = if (tau > 2.0) -KE * a else 0.0\nd/dt(tau) = 1.0",
        "d/dt(a) = if (TIME > 2.0) -KE * a else 0.0\nd/dt(tau) = 1.0",
        // `a` has no parameter in its equation either.
        "if (a > 2.0) {\n  d/dt(a) = -1.0\n} else {\n  d/dt(a) = 0.0\n}\nd/dt(tau) = 1.0",
    ] {
        assert_eq!(reads(body), Vec::new(), "{body}");
    }
}

/// With `d/dt(tau) = LAGT`, `if (tau > 2)` switches at `2/LAGT`: no parameter in the condition.
#[test]
fn a_parameter_reaches_a_condition_through_a_state() {
    for body in [
        "d/dt(a) = if (tau > 2.0) -1.0 else 0.0\nd/dt(tau) = LAGT",
        // Whichever line comes first.
        "d/dt(tau) = LAGT\nd/dt(a) = if (tau > 2.0) -1.0 else 0.0",
        // Through a second state, and an intermediate.
        "d/dt(a) = LAGT\nd/dt(tau) = a\nX = tau\nif (X > 2.0) {\n  d/dt(a) = 0.0\n}",
        // The equation's arm is reached through a condition on the parameter.
        "d/dt(a) = if (tau > 2.0) -1.0 else 0.0\nd/dt(tau) = if (LAGT > 1.0) 1.0 else 2.0",
    ] {
        let got = reads(body);
        assert!(got.contains(&lagt(true)[0]), "{body}\ngot {got:?}");
    }
}

/// These desugar to conditionals over `LAGT`, but the value is continuous at the switch.
#[test]
fn continuous_selects_are_exempt() {
    for body in [
        "d/dt(a) = -KE * max(tau - LAGT, 0.0)\nd/dt(tau) = 1.0",
        "d/dt(a) = -KE * min(tau, LAGT)\nd/dt(tau) = 1.0",
        "d/dt(a) = -KE * clamp(tau - LAGT, 0.0, 9.0)\nd/dt(tau) = 1.0",
        // The upper bound only: its arm is a `min`, and the lower test `tau <= 0` reads no
        // parameter. Even if `LAGT < 0` jumped there, that switch stays at `tau = 0`.
        "d/dt(a) = -KE * clamp(tau, 0.0, LAGT)\nd/dt(tau) = 1.0",
        // The same shape written by hand is the same function.
        "d/dt(a) = -KE * (if (tau >= LAGT) tau else LAGT)\nd/dt(tau) = 1.0",
    ] {
        assert_eq!(reads(body), Vec::new(), "{body}");
    }
}

/// One arm from each continuous shape is not enough: these jump at the switch.
#[test]
fn a_near_miss_of_a_continuous_select_is_not_exempt() {
    for body in [
        "d/dt(a) = -KE * (if (tau >= LAGT) tau else 0.0)\nd/dt(tau) = 1.0",
        "d/dt(a) = -KE * (if (tau <= LAGT) LAGT else if (tau >= 9.0) 9.0 else 0.0)\n\
         d/dt(tau) = 1.0",
        // A parameter bound can invert the interval: `LAGT > 0.5` jumps from `LAGT` to 0.5.
        "d/dt(a) = -KE * clamp(tau, LAGT, 0.5)\nd/dt(tau) = 1.0",
        // `clamp`'s layout with an inner comparison pointing the other way: at the outer
        // switch the arms are 0 and 9.
        "X = tau - LAGT\nd/dt(a) = -KE * (if (X <= 0.0) 0.0 else if (X <= 9.0) 9.0 else X)\n\
         d/dt(tau) = 1.0",
        // And with the outer one reversed: above 0 it holds 0, below it `X`.
        "X = tau - LAGT\nd/dt(a) = -KE * (if (X >= 0.0) 0.0 else if (X >= 9.0) 9.0 else X)\n\
         d/dt(tau) = 1.0",
        // Literal bounds out of order, which only `clamp` itself refuses at parse time.
        "X = tau - LAGT\nd/dt(a) = -KE * (if (X <= 9.0) 9.0 else if (X >= 0.0) 0.0 else X)\n\
         d/dt(tau) = 1.0",
    ] {
        assert_eq!(reads(body), lagt(true), "{body}");
    }
}

/// The #2649 model with its `[individual_parameters]` and first equation swappable.
fn model(thetas: &str, indiv: &str, rhs: &str) -> crate::types::CompiledModel {
    let src = format!(
        "[parameters]
  theta TVKE(0.2, 0.01, 5.0)
  {thetas}
  omega ETA_KE ~ 0.2
  sigma ADD ~ 0.5 (sd)
[individual_parameters]
  KE = TVKE * exp(ETA_KE)
  {indiv}
[structural_model]
  ode(obs_cmt=a, states=[a, tau])
[odes]
  d/dt(a) = {rhs}
  d/dt(tau) = 1.0
[covariates]
  WT continuous
[error_model]
  DV ~ additive(ADD)
"
    );
    parse_model_string(&src).unwrap_or_else(|e| panic!("{src}\n{e}"))
}

const TLAG: &str = "theta TLAG(1.13, 0.1, 6.0)";
const SWITCH: &str = "if (tau > LAGT) -KE * a else 0.0";

fn switch_warning(m: &crate::types::CompiledModel) -> Option<&String> {
    m.parse_warnings
        .iter()
        .find(|w| w.contains("`if` condition"))
}

fn assert_declined(m: &crate::types::CompiledModel, param: &str, why: &str) {
    assert!(!recorder_in_scope(m), "{why}: the recorder must decline");
    assert!(!outer_model_in_scope(m), "{why}: the outer must decline");
    let w = switch_warning(m).unwrap_or_else(|| panic!("{why}: no warning"));
    assert!(w.contains(&format!("`{param}`")), "{why}: {w}");
    assert!(w.contains("finite-difference gradients"), "{why}: {w}");
}

fn assert_analytic(m: &crate::types::CompiledModel, why: &str) {
    assert!(recorder_in_scope(m), "{why}: the recorder must serve it");
    assert!(outer_model_in_scope(m), "{why}: the outer must serve it");
    assert_eq!(switch_warning(m), None, "{why}");
}

#[test]
fn a_switch_moved_by_an_estimated_parameter_routes_to_fd() {
    // The issue's model: a θ reaching the equations only through the condition.
    assert_declined(&model(TLAG, "LAGT = TLAG", SWITCH), "LAGT", "θ only");
    // Through an intermediate of `[individual_parameters]`.
    assert_declined(
        &model(TLAG, "TV = 2 * TLAG\n  LAGT = TV", SWITCH),
        "LAGT",
        "θ via an intermediate",
    );
    // A condition in the arms as well: its θ has a non-zero gradient that is still wrong.
    assert_declined(
        &model(
            TLAG,
            "LAGT = TLAG",
            "if (tau > LAGT) -KE * a * LAGT else 0.0",
        ),
        "LAGT",
        "θ in the condition and an arm",
    );
    // An η on the switch, with no θ behind it.
    assert_declined(
        &model("omega ETA_L ~ 0.1", "LAGT = 2 * exp(ETA_L)", SWITCH),
        "LAGT",
        "η only",
    );
    // FIX is the caller's to lift at `fit()`, which the scope cannot see.
    assert_declined(
        &model("theta TLAG(1.13, FIX)", "LAGT = TLAG", SWITCH),
        "LAGT",
        "a FIX θ",
    );
    // A FIX θ does not rescue a parameter another θ also moves.
    assert_declined(
        &model(
            "theta TFIX(1.0, FIX)\n  theta TLAG(1.13, 0.1, 6.0)",
            "LAGT = TFIX * TLAG",
            SWITCH,
        ),
        "LAGT",
        "a free θ beside a FIX one",
    );
    // No state or clock in the condition: `SW` itself varies with time.
    assert_declined(
        &model(TLAG, "SW = TLAG * TIME", "if (SW > 0.5) -KE * a else 0.0"),
        "SW",
        "a parameter reading TIME",
    );
}

/// A switch moved through a state: the condition itself names no parameter.
#[test]
fn a_switch_moved_through_a_state_routes_to_fd() {
    assert_declined(
        &model("", "", "if (a > 2.0) -KE * a else 0.0"),
        "KE",
        "the state's own equation",
    );
    let clocked = |rhs_tau: &str, extra: &str, indiv: &str| {
        let src = format!(
            "[parameters]
  theta TR(1.0, 0.1, 6.0)
  omega ETA_R ~ 0.2
  sigma ADD ~ 0.5 (sd)
[individual_parameters]
  R = TR * exp(ETA_R)
  {indiv}
[structural_model]
  ode(obs_cmt=a, states=[a, tau])
[odes]
  d/dt(a) = if (tau > 2.0) -0.1 * a else 0.0
  d/dt(tau) = {rhs_tau}
  {extra}
[error_model]
  DV ~ additive(ADD)
"
        );
        parse_model_string(&src).unwrap_or_else(|e| panic!("{src}\n{e}"))
    };
    assert_declined(&clocked("R", "", ""), "R", "the review's d/dt(tau) = RATE");
    assert_declined(&clocked("1.0", "init(tau) = R", ""), "R", "init(...)");
    assert_declined(
        &clocked("1.0 + first_order(ka=R)", "", ""),
        "R",
        "an input-rate term",
    );
    assert_declined(&clocked("1.0", "", "F = R"), "F", "a dose attribute");
    assert_declined(&clocked("1.0", "", "ALAG1 = R"), "ALAG1", "a lag");
    // The control for all five: the clock is free of `R`.
    assert_analytic(&clocked("1.0", "", ""), "a clock no parameter reaches");
}

/// A dose's arrival anchors `TAD` and `TAFD`, so a lag moves them. `TIME` it does not.
#[test]
fn a_lag_moves_the_dose_clocks_only() {
    let lagged = |clock: &str| {
        let src = format!(
            "[parameters]
  theta TLAG(1.0, 0.1, 6.0)
  sigma ADD ~ 0.5 (sd)
[individual_parameters]
  ALAG1 = TLAG
[structural_model]
  ode(obs_cmt=a, states=[a])
[odes]
  d/dt(a) = if ({clock} > 2.0) -0.1 else 0.0
[error_model]
  DV ~ additive(ADD)
"
        );
        parse_model_string(&src).unwrap_or_else(|e| panic!("{src}\n{e}"))
    };
    assert_declined(&lagged("TAD"), "ALAG1", "TAD");
    assert_declined(&lagged("TAFD"), "ALAG1", "TAFD");
    assert_analytic(&lagged("TIME"), "TIME");
}

/// The controls, each differing from the declined model in the one thing that decides it.
#[test]
fn a_switch_nothing_estimated_can_move_stays_analytic() {
    assert_analytic(
        &model("", "", "if (tau > 2.0) -KE * a else 0.0"),
        "a literal threshold on a clock",
    );
    assert_analytic(&model("", "LAGT = WT / 10", SWITCH), "covariates alone");
    assert_analytic(
        &model(TLAG, "LAGT = TLAG", "if (LAGT > 2.0) -KE * a else 0.0"),
        "a condition that cannot change during the integration",
    );
    assert_analytic(
        &model(TLAG, "LAGT = TLAG", "-KE * a * max(tau - LAGT, 0.0)"),
        "a continuous select",
    );
}
