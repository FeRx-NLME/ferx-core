use super::super::{assigned_vars_in_order, parse_block_statements, ParseCtx, StatementMode};
use super::*;

/// Parse an `[odes]` body the way `build_ode_spec` does — two passes, so intermediates
/// assigned inside the block parse as variables — and classify it.
fn affine(states: &[&str], params: &[&str], body: &str) -> bool {
    let states: Vec<String> = states.iter().map(|s| s.to_string()).collect();
    let mut defined: Vec<String> = states
        .iter()
        .cloned()
        .chain(params.iter().map(|s| s.to_string()))
        .collect();
    let pre = parse_block_statements(body, ParseCtx::ode(&defined), StatementMode::Ode)
        .expect("fixture parses");
    for v in assigned_vars_in_order(&pre) {
        if !defined.contains(&v) {
            defined.push(v);
        }
    }
    let stmts = parse_block_statements(body, ParseCtx::ode(&defined), StatementMode::Ode)
        .expect("fixture parses");
    stmts_affine_in_states(&stmts, &states)
}

const ONE_CPT: &[&str] = &["central"];
const TWO_CPT: &[&str] = &["depot", "central", "periph"];

#[test]
fn accepts_linear_compartmental_systems() {
    let cases: &[(&str, &[&str], &[&str], &str)] = &[
        (
            "one-cpt",
            ONE_CPT,
            &["CL", "V"],
            "d/dt(central) = -CL/V*central",
        ),
        (
            "two-cpt with depot",
            TWO_CPT,
            &["KA", "CL", "V", "Q", "VP"],
            "d/dt(depot) = -KA*depot
             d/dt(central) = KA*depot - (CL/V)*central - (Q/V)*central + (Q/VP)*periph
             d/dt(periph) = (Q/V)*central - (Q/VP)*periph",
        ),
        (
            "intermediates carry the affine degree",
            ONE_CPT,
            &["CL", "V"],
            "K = CL/V
             C = central/V
             d/dt(central) = -CL*C + 0*K",
        ),
        (
            "unary minus, parentheses, state^1",
            ONE_CPT,
            &["CL", "V"],
            "d/dt(central) = -(CL/V)*(central^1)",
        ),
        (
            "an if on parameters picks between affine forms",
            ONE_CPT,
            &["CL", "V", "SEX"],
            "if (SEX > 0.5) {
               KE = 1.2*CL/V
             } else {
               KE = CL/V
             }
             d/dt(central) = -KE*central",
        ),
        (
            "transcendental of parameters is a coefficient",
            ONE_CPT,
            &["LCL", "V"],
            "d/dt(central) = -exp(LCL)/V*central + log(V)",
        ),
        (
            "case-insensitive state names",
            ONE_CPT,
            &["CL", "V"],
            "d/dt(central) = -CL/V*CENTRAL",
        ),
    ];
    for (name, states, params, body) in cases {
        assert!(
            affine(states, params, body),
            "{name} should classify linear"
        );
    }
}

#[test]
fn rejects_nonlinear_systems() {
    let cases: &[(&str, &[&str], &[&str], &str)] = &[
        (
            "Michaelis-Menten",
            ONE_CPT,
            &["VMAX", "KM"],
            "d/dt(central) = -VMAX*central/(KM + central)",
        ),
        (
            "MM through an intermediate",
            ONE_CPT,
            &["VMAX", "KM", "V"],
            "C = central/V
             E = VMAX*C/(KM + C)
             d/dt(central) = -E",
        ),
        (
            "state times state (TMDD binding)",
            &["central", "target", "complex"],
            &["KON", "KOFF", "KINT"],
            "d/dt(central) = -KON*central*target + KOFF*complex
             d/dt(target) = -KON*central*target + KOFF*complex
             d/dt(complex) = KON*central*target - (KOFF + KINT)*complex",
        ),
        (
            "state in a denominator",
            ONE_CPT,
            &["K"],
            "d/dt(central) = -K/central",
        ),
        (
            "transcendental of a state",
            ONE_CPT,
            &["K"],
            "d/dt(central) = -K*exp(central)",
        ),
        (
            "non-unit power of a state",
            ONE_CPT,
            &["K"],
            "d/dt(central) = -K*central^2",
        ),
        (
            "state power by a parameter",
            ONE_CPT,
            &["K", "G"],
            "d/dt(central) = -K*central^G",
        ),
        (
            "state in an if condition",
            ONE_CPT,
            &["K1", "K2"],
            "if (central > 10) {
               KE = K1
             } else {
               KE = K2
             }
             d/dt(central) = -KE*central",
        ),
        (
            "state in an inline conditional's test",
            ONE_CPT,
            &["K1", "K2"],
            "d/dt(central) = -(if (central > 10) K1 else K2)*central",
        ),
        (
            "nonlinear branch of an if on parameters",
            ONE_CPT,
            &["K", "FLAG"],
            "if (FLAG > 0.5) {
               R = K*central*central
             } else {
               R = K*central
             }
             d/dt(central) = -R",
        ),
        (
            "intermediate read before it is assigned",
            ONE_CPT,
            &["K"],
            "d/dt(central) = -K*X
             X = central*central",
        ),
    ];
    for (name, states, params, body) in cases {
        assert!(
            !affine(states, params, body),
            "{name} must not classify linear"
        );
    }
}

#[test]
fn rejects_a_resolved_program() {
    // The classifier needs names; a slot-resolved program is reported as not linear rather
    // than guessed at.
    let stmts = vec![Statement::DiffEqIdx(0, Expression::VariableIdx(0))];
    assert!(!stmts_affine_in_states(&stmts, &["central".to_string()]));
}
