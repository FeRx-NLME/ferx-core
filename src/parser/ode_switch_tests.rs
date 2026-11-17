use super::super::{
    assigned_vars_in_order, parse_block_statements, parse_model_string, ParseCtx, StatementMode,
};
use super::*;
use crate::sens::ode_provider::adjoint::{outer_model_in_scope, recorder_in_scope};
use crate::types::CompiledModel;

const STATES: &[&str] = &["a", "tau"];
const PARAMS: &[&str] = &["KE", "LAGT"];

/// Parse an `[odes]` body the way `build_ode_spec` does, then report its conditions.
fn reads(body: &str) -> Vec<ConditionReads> {
    let mut got = analyse(body);
    got.iter_mut().for_each(|c| c.leaves.clear());
    got
}

fn analyse(body: &str) -> Vec<ConditionReads> {
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
        leaves: Vec::new(),
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

/// An `if` whose arms meet at the switch has no sensitivity jump, whatever the arms are.
#[test]
fn arms_that_meet_at_the_switch_are_exempt() {
    for body in [
        // Friberg feedback, as NONMEM writes it: the arms are `(C/x)^g` and 1 at `x = C`.
        "d/dt(a) = -KE * (if (tau < LAGT) (LAGT / tau)^0.16 else 1.0)\nd/dt(tau) = 1.0",
        // A hinge, with the condition on either side.
        "d/dt(a) = -KE * (if (tau > LAGT) tau - LAGT else 0.0)\nd/dt(tau) = 1.0",
        "d/dt(a) = -KE * (if (LAGT >= tau) 0.0 else 2.0 * (tau - LAGT))\nd/dt(tau) = 1.0",
        // The clock is a coordinate of the surface too.
        "d/dt(a) = -KE * (if (TIME > LAGT) TIME - LAGT else 0.0)\nd/dt(tau) = 1.0",
        // Arms that read a third name still have to meet for every value of it.
        "d/dt(a) = -(if (tau > LAGT) KE * (tau - LAGT) else 0.0)\nd/dt(tau) = 1.0",
        // `<=` is a side like any other.
        "d/dt(a) = -(if (tau <= LAGT) 0.0 else tau - LAGT)\nd/dt(tau) = 1.0",
        // Scaled sides, with the arms swapped.
        "d/dt(a) = -(if (tau * 2.0 > LAGT) tau * 2.0 - LAGT else 0.0)\nd/dt(tau) = 1.0",
        "d/dt(a) = -(if (2.0 * tau - 1.0 > LAGT / 2.0) 2.0 * tau - 1.0 - LAGT / 2.0 else 0.0)\n\
         d/dt(tau) = 1.0",
        "d/dt(a) = -KE * (if (tau >= LAGT) 1.0 else (LAGT / tau)^0.16)\nd/dt(tau) = 1.0",
        // The functions that have no jump of their own to hide.
        "d/dt(a) = -(if (tau > LAGT) log(1.0 + tau * tau) - log(1.0 + LAGT * LAGT) else 0.0)\n\
         d/dt(tau) = 1.0",
        "d/dt(a) = -(if (tau > LAGT) ln(1.0 + tau * tau) - ln(1.0 + LAGT * LAGT) else 0.0)\n\
         d/dt(tau) = 1.0",
        "d/dt(a) = -(if (tau > LAGT) sqrt(1.0 + tau * tau) - sqrt(1.0 + LAGT * LAGT) else 0.0)\n\
         d/dt(tau) = 1.0",
        // One function spelled two ways: the arms differ by rounding, at every point.
        "d/dt(a) = -(if (tau > LAGT) tau * 3.0 * 0.1 else tau * 0.1 + tau * 0.1 + tau * 0.1)\n\
         d/dt(tau) = 1.0",
        // Large arms sharing an offset, meeting to within its rounding and no closer.
        "d/dt(a) = (if (tau > LAGT) 1e12 + tau - LAGT else 1e12) - 1e12\nd/dt(tau) = 1.0",
    ] {
        assert_eq!(reads(body), Vec::new(), "{body}");
    }
}

/// Each of these is one edit from a shape above, and each jumps at its switch.
#[test]
fn arms_that_miss_at_the_switch_are_not_exempt() {
    for body in [
        // Arms 1 and 2 apart everywhere on the surface.
        "d/dt(a) = -KE * (if (tau < LAGT) (LAGT / tau)^0.16 else 2.0)\nd/dt(tau) = 1.0",
        // A small jump is still a jump.
        "d/dt(a) = -KE * (if (tau > LAGT) tau - LAGT + 1e-10 else 0.0)\nd/dt(tau) = 1.0",
        // So is one between small arms, and one the arms' size would otherwise hide.
        "d/dt(a) = -KE * (if (tau > LAGT) 1e-9 else 0.0)\nd/dt(tau) = 1.0",
        "d/dt(a) = (if (tau > LAGT) 1000000000001.0 else 1000000000000.0) - 1e12\n\
         d/dt(tau) = 1.0",
        // An arm can be steep enough to drown the jump in the crossing's rounding.
        "d/dt(a) = -KE * (if (tau > LAGT) 1.0 + 1e300 * (tau - LAGT) else 0.0)\nd/dt(tau) = 1.0",
        // Arms whose sizes overflow when added, and one arm that is not finite at all.
        "d/dt(a) = 1e-308 * (if (tau > LAGT) 1e308 else 9e307)\nd/dt(tau) = 1.0",
        "d/dt(a) = -KE * (if (tau > LAGT) exp(1000.0) else 1.0)\nd/dt(tau) = 1.0",
        // `==` and `!=` read one arm on the switch and the other everywhere around it.
        "d/dt(a) = -(if (tau == LAGT) 2.0 * tau else tau + LAGT)\nd/dt(tau) = 1.0",
        "d/dt(a) = -(if (tau != LAGT) 2.0 * tau else tau + LAGT)\nd/dt(tau) = 1.0",
        // And one beside a steep arm, however it compares with the arm's own growth.
        "d/dt(a) = -KE * (if (tau > LAGT) 1e-9 + 1e3 * (tau - LAGT) else 0.0)\nd/dt(tau) = 1.0",
        "d/dt(a) = -KE * (if (tau > LAGT) 1e-5 + 1e10 * (tau - LAGT) else 0.0)\nd/dt(tau) = 1.0",
        "d/dt(a) = -KE * (if (tau > LAGT) 1.0 + 1e7 * (tau - LAGT) else 0.0)\nd/dt(tau) = 1.0",
        // The arms meet at one value of `KE`, not across the surface.
        "d/dt(a) = -(if (tau > LAGT) KE else 1.0)\nd/dt(tau) = 1.0",
        // `|x|` and `x` agree on half the surface: positive draws alone would pass this.
        "d/dt(a) = -KE * (if (tau > LAGT) sqrt(tau * tau) else LAGT)\nd/dt(tau) = 1.0",
        // The arms meet wherever `tau`, `KE` and `a` have an even count of negatives, which
        // is most of the surface: one crossing can pass, and only several reject it.
        "d/dt(a) = -(if (tau > LAGT) sqrt(tau * tau) * sqrt(KE * KE) * sqrt(a * a) \
         else LAGT * KE * a)\n\
         d/dt(tau) = 1.0",
        "d/dt(a) = -(if (LAGT < tau) sqrt(KE * KE) * sqrt(a * a) * sqrt(LAGT * LAGT) \
         else KE * a * tau)\n\
         d/dt(tau) = 1.0",
        // A step inside an arm can hide its jump above the sampled range.
        "d/dt(a) = -KE * (if (tau > LAGT) floor(sqrt(LAGT * LAGT) / 1e9) else 0.0)\nd/dt(tau) = 1.0",
        "d/dt(a) = -KE * (if (tau > LAGT) ceil(sqrt(LAGT * LAGT) / 1e9 - 1.0) else 0.0)\n\
         d/dt(tau) = 1.0",
        "d/dt(a) = -KE * (if (tau > LAGT) round(sqrt(LAGT * LAGT) / 1e9) else 0.0)\n\
         d/dt(tau) = 1.0",
        // So can a kink.
        "d/dt(a) = -(if (tau > LAGT) abs(LAGT - 1e9) + LAGT else 1e9)\nd/dt(tau) = 1.0",
        // So can a saturation: `tanh` rounds to -1 throughout the sampled range.
        "d/dt(a) = -(if (tau > LAGT) tanh(a - 1200.0) else -1.0)\nd/dt(tau) = 1.0",
        // As can `exp`, whose term rounds away against the 1 until `a` passes 1200.
        "d/dt(a) = -(if (tau > LAGT) 1.0 + exp(0.4 * (a - 1200.0)) else 1.0)\nd/dt(tau) = 1.0",
        // A nested conditional can too.
        "d/dt(a) = -KE * (if (tau > LAGT) (if (LAGT > 1e9) 1.0 else 0.0) else 0.0)\n\
         d/dt(tau) = 1.0",
        // The arms meet at `tau = LAGT`, the only part of this surface in the sampled range,
        // and miss at `tau = LAGT + 3000`.
        "d/dt(a) = -(if (0.0 < (tau - LAGT) * (tau - LAGT - 3000.0)) tau - LAGT else 0.0)\n\
         d/dt(tau) = 1.0",
        // A variable denominator flips the condition at its pole too, with the arms apart.
        "d/dt(a) = -(if (LAGT / tau > 1.0) LAGT - tau else 0.0)\nd/dt(tau) = -LAGT",
        // A pole above the sampled range never shows in the arms, but can in a trajectory.
        "d/dt(a) = -(if (tau / (LAGT - 1200.0) > 0.01) tau - 0.01 * (LAGT - 1200.0) else 0.0)\n\
         d/dt(tau) = 1.0",
        // A ratio against a name, and a product of names, are not one hyperplane either.
        "d/dt(a) = -(if (tau / LAGT > KE) tau / LAGT - KE else 0.0)\nd/dt(tau) = 1.0",
        "d/dt(a) = -(if (tau > LAGT / KE) tau - LAGT / KE else 0.0)\nd/dt(tau) = 1.0",
        "d/dt(a) = -(if (tau * KE > LAGT) tau * KE - LAGT else 0.0)\nd/dt(tau) = 1.0",
        // A compound condition has no single surface to bisect onto.
        "d/dt(a) = -KE * (if (tau > LAGT && a > 0.0) tau - LAGT else 0.0)\nd/dt(tau) = 1.0",
    ] {
        let got = reads(body);
        assert!(got.iter().any(|c| c.moves), "{body}\ngot {got:?}");
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

/// How many moving switches the model's integration locates.
fn switches(m: &CompiledModel) -> usize {
    let ode = m.ode_spec.as_ref().expect("an ODE model");
    ode.rhs_program.as_ref().expect("compiled").switch_count()
}

/// Served by the exact derivatives with `n` switches located (#2649): every model is, the
/// switches decide only whether its integration watches for crossings and jumps there.
fn assert_served(m: &CompiledModel, n: usize, why: &str) {
    assert!(recorder_in_scope(m), "{why}: the recorder must serve it");
    assert!(outer_model_in_scope(m), "{why}: the outer must serve it");
    assert_eq!(switches(m), n, "{why}");
    assert!(
        !m.parse_warnings
            .iter()
            .any(|w| w.contains("`if` condition")),
        "{why}: {:?}",
        m.parse_warnings
    );
}

/// A located switch: the integration watches it.
fn assert_located(m: &CompiledModel, why: &str) {
    assert_served(m, 1, why);
}

/// No switch to locate: the integration is the plain one.
fn assert_analytic(m: &CompiledModel, why: &str) {
    assert_served(m, 0, why);
}

#[test]
fn a_switch_moved_by_an_estimated_parameter_is_located() {
    // The issue's model: a θ reaching the equations only through the condition.
    assert_located(&model(TLAG, "LAGT = TLAG", SWITCH), "θ only");
    // Through an intermediate of `[individual_parameters]`.
    assert_located(
        &model(TLAG, "TV = 2 * TLAG\n  LAGT = TV", SWITCH),
        "θ via an intermediate",
    );
    // A condition in the arms as well.
    assert_located(
        &model(
            TLAG,
            "LAGT = TLAG",
            "if (tau > LAGT) -KE * a * LAGT else 0.0",
        ),
        "θ in the condition and an arm",
    );
    // An η on the switch, with no θ behind it.
    assert_located(
        &model("omega ETA_L ~ 0.1", "LAGT = 2 * exp(ETA_L)", SWITCH),
        "η only",
    );
    // FIX is the caller's to lift at `fit()`: the switch is located whatever is estimated.
    assert_located(
        &model("theta TLAG(1.13, FIX)", "LAGT = TLAG", SWITCH),
        "a FIX θ",
    );
    // Covariates alone place the switch per subject; locating it is exact, with no jump.
    assert_located(&model("", "LAGT = WT / 10", SWITCH), "covariates alone");
}

/// An `if` whose arms meet at its switch has no jump, so nothing is located.
#[test]
fn a_moving_switch_with_meeting_arms_stays_analytic() {
    assert_analytic(
        &model(
            TLAG,
            "LAGT = TLAG",
            "if (tau > LAGT) KE * (LAGT - tau) else 0.0",
        ),
        "a hinge at a moved switch",
    );
    assert_analytic(
        &model(
            TLAG,
            "LAGT = TLAG",
            "-KE * a * (if (tau < LAGT) (LAGT / tau)^0.16 else 1.0)",
        ),
        "a feedback ratio at a moved switch",
    );
}

/// A switch moved through a state: the condition itself names no parameter.
#[test]
fn a_switch_moved_through_a_state_is_located() {
    assert_located(
        &model("", "", "if (a > 2.0) -KE * a else 0.0"),
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
    assert_located(&clocked("R", "", ""), "the review's d/dt(tau) = RATE");
    assert_located(&clocked("1.0", "init(tau) = R", ""), "init(...)");
    assert_located(
        &clocked("1.0 + first_order(ka=R)", "", ""),
        "an input-rate term",
    );
    assert_located(&clocked("1.0", "", "F = R"), "a dose attribute");
    assert_located(&clocked("1.0", "", "ALAG1 = R"), "a lag");
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
    assert_located(&lagged("TAD"), "TAD");
    assert_located(&lagged("TAFD"), "TAFD");
    assert_analytic(&lagged("TIME"), "TIME");
}

/// The controls, each differing from a located model in the one thing that decides it.
#[test]
fn a_switch_nothing_can_move_is_not_located() {
    assert_analytic(
        &model("", "", "if (tau > 2.0) -KE * a else 0.0"),
        "a literal threshold on a clock",
    );
    assert_analytic(
        &model(TLAG, "LAGT = TLAG", "if (LAGT > 2.0) -KE * a else 0.0"),
        "a condition that cannot change during the integration",
    );
    // `SW` changes only where the walk re-evaluates `[individual_parameters]`, at records,
    // never inside a segment: its switch is at a data time no parameter moves.
    assert_analytic(
        &model(TLAG, "SW = TLAG * TIME", "if (SW > 0.5) -KE * a else 0.0"),
        "a parameter reading TIME",
    );
    assert_analytic(
        &model(TLAG, "LAGT = TLAG", "-KE * a * max(tau - LAGT, 0.0)"),
        "a continuous select",
    );
}

/// Each moving comparison is one switch, however often it is written, and only the moving
/// comparisons of a compound condition are: `TAD < 24` reads no parameter (no lag here).
///
/// Regression caught: a repeated comparison counted twice (two jumps at one crossing), or a
/// comparison no parameter reaches located at every dose.
#[test]
fn switches_are_counted_per_moving_comparison() {
    let rhs = "if (tau > LAGT && TAD < 24) -KE * a else 0.0 + 0.0 * (if (tau > LAGT) 1.0 else 2.0)";
    assert_located(&model(TLAG, "LAGT = TLAG", rhs), "one comparison, twice");
    let two = "if (tau > LAGT) -KE * a else (if (a < LAGT) 0.0 else -1.0)";
    assert_served(&model(TLAG, "LAGT = TLAG", two), 2, "two comparisons");
}

/// A model without a moving switch takes no new path (#2649): no switch program, its closure
/// is the one built before (served by the direct native call where it has one), and evaluating
/// it tells the integrator of no switch, so the driver never evaluates a switch function and
/// integrates bit for bit as before. The same model with a moving switch differs in each.
///
/// Regression caught: a switch-free model wrapped or noted (paying the closure route, or the
/// driver's switch evaluations), or a switch model that does not note its switches (its
/// crossings go unlocated on the prediction path while the recording locates them).
#[test]
fn a_switch_free_model_takes_no_new_path() {
    let eval_notes = |m: &CompiledModel| {
        let ode = m.ode_spec.as_ref().expect("an ODE model");
        let prog = ode.rhs_program.as_ref().expect("compiled");
        let p = vec![0.3; crate::types::MAX_PK_PARAMS + 2];
        let mut du = vec![0.0; ode.n_states];
        crate::ode::solver::take_switches();
        (ode.rhs)(&[1.0, 0.5], &p, 0.0, &mut du);
        let direct = prog.direct_f64(&*ode.rhs).is_some();
        (
            crate::ode::solver::take_switches(),
            direct,
            prog.direct_f64_native(),
        )
    };
    let free = model(TLAG, "LAGT = TLAG", "-KE * a");
    assert_eq!(switches(&free), 0);
    let (noted, direct, native) = eval_notes(&free);
    assert_eq!(noted, 0, "a switch-free model notes no switch");
    assert_eq!(
        direct, native,
        "a switch-free model keeps its direct native call"
    );
    assert!(
        native,
        "the fixture has a native function, so the direct path is live"
    );
    let moving = model(TLAG, "LAGT = TLAG", SWITCH);
    assert_eq!(switches(&moving), 1);
    let (noted, direct, _) = eval_notes(&moving);
    assert_eq!(noted, 1, "a switch model notes its switch");
    assert!(!direct, "a switch model runs its closure, which notes it");
}
