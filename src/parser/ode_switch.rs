//! Which `[odes]` `if` conditions read an individual parameter (#2649).
//!
//! The exact derivatives differentiate the arm an `if` selects; the condition itself carries
//! no derivative. That is right when the arm is fixed for the whole integration, or when the
//! switch sits at a time no parameter can move. It is wrong for `if (tau > LAGT)`: the
//! switching time moves with `LAGT`, the state's sensitivity jumps there, and the branch-wise
//! right-hand side does not contain that jump. A θ reaching the equations only through such a
//! condition gets an analytic gradient of exactly zero, so the fit never moves it.
//!
//! This module reports, for each condition, the individual parameters it reads and whether it
//! also reads something that varies during the integration (a state or a clock). The parser
//! then decides, against `[individual_parameters]`, which conditions depend on an estimated
//! parameter; `OdeRhsProgram::moving_switch` carries that verdict to the scope gates.
//!
//! Intermediates carry what they were assigned, so `L = 2*LAGT` then `if (tau > L)` is seen.
//! `min`, `max` and `clamp` desugar to conditionals but are continuous across their switch,
//! where the branch-wise derivative is exact, so [`is_continuous_select`] exempts them.
//!
//! What this module does **not** decide: a condition reading only states and literals, such as
//! `if (central > 20)`. Its switching time moves too, through the state, but no parameter is
//! read and no parameter is frozen by it. Those models stay on the analytic route.

use super::{Condition, Expression, Statement};
use std::collections::{BTreeSet, HashMap};

/// What an expression or condition reads, as far as a switch is concerned.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Reads {
    /// Individual parameters, by their declared name.
    params: BTreeSet<String>,
    /// A state or clock, directly or through an intermediate.
    moves: bool,
}

impl Reads {
    fn merge(&mut self, other: &Reads) {
        self.params.extend(other.params.iter().cloned());
        self.moves |= other.moves;
    }
}

/// One `[odes]` condition that reads at least one individual parameter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ConditionReads {
    /// The individual parameters it reads, directly or through an intermediate.
    pub(super) params: Vec<String>,
    /// Whether it also reads a state or clock. Without one, the condition is decided by the
    /// parameters alone unless a parameter itself reads `TIME`, which only the caller knows.
    pub(super) moves: bool,
}

/// What each intermediate holds, by upper-cased name.
type Scope = HashMap<String, Reads>;

struct Walker<'a> {
    /// Names resolve case-insensitively in `[odes]`, as in `ode_linearity`.
    states: &'a [String],
    params: &'a [String],
    scope: Scope,
    found: Vec<ConditionReads>,
}

impl Walker<'_> {
    fn name(&self, n: &str, out: &mut Reads) {
        if let Some(r) = self.scope.get(&n.to_ascii_uppercase()) {
            out.merge(r);
        } else if self.states.iter().any(|s| s.eq_ignore_ascii_case(n))
            || ["T", "TIME", "TAD", "TAFD"]
                .iter()
                .any(|c| c.eq_ignore_ascii_case(n))
        {
            out.moves = true;
        } else if let Some(p) = self.params.iter().find(|p| p.eq_ignore_ascii_case(n)) {
            out.params.insert(p.clone());
        }
    }

    fn expr_reads(&self, e: &Expression) -> Reads {
        let mut out = Reads::default();
        super::visit_expr_nodes(e, &mut |n| match n {
            Expression::Variable(v) => self.name(v, &mut out),
            Expression::Time => out.moves = true,
            _ => {}
        });
        out
    }

    fn cond_reads(&self, c: &Condition) -> Reads {
        let mut out = Reads::default();
        super::visit_condition_nodes(c, &mut |n| match n {
            Expression::Variable(v) => self.name(v, &mut out),
            Expression::Time => out.moves = true,
            _ => {}
        });
        out
    }

    fn record(&mut self, r: &Reads) {
        if r.params.is_empty() {
            return;
        }
        let c = ConditionReads {
            params: r.params.iter().cloned().collect(),
            moves: r.moves,
        };
        if !self.found.contains(&c) {
            self.found.push(c);
        }
    }

    /// Record every inline conditional inside `e`.
    fn inline_conditions(&mut self, e: &Expression) {
        let mut conds: Vec<Reads> = Vec::new();
        super::visit_expr_nodes(e, &mut |n| {
            if let Expression::Conditional(c, t, f) = n {
                if !is_continuous_select(c, t, f) {
                    conds.push(self.cond_reads(c));
                }
            }
        });
        for c in &conds {
            self.record(c);
        }
    }

    fn stmts(&mut self, stmts: &[Statement], ctx: &Reads) {
        for s in stmts {
            match s {
                Statement::Assign(name, e) => {
                    self.inline_conditions(e);
                    let mut r = self.expr_reads(e);
                    r.merge(ctx);
                    self.scope
                        .entry(name.to_ascii_uppercase())
                        .or_default()
                        .merge(&r);
                }
                Statement::DiffEq(_, e) => self.inline_conditions(e),
                Statement::If {
                    branches,
                    else_body,
                } => {
                    // Each arm is reached through every condition before it.
                    let mut arm_ctx = ctx.clone();
                    for (cond, body) in branches {
                        let r = self.cond_reads(cond);
                        self.record(&r);
                        arm_ctx.merge(&r);
                        self.stmts(body, &arm_ctx);
                    }
                    if let Some(body) = else_body {
                        self.stmts(body, &arm_ctx);
                    }
                }
                // Only the unresolved program reaches this module.
                Statement::AssignIdx(..)
                | Statement::DiffEqIdx(..)
                | Statement::AssignBc(..)
                | Statement::DiffEqBc(..) => {}
            }
        }
    }
}

/// `true` for the shapes `min`, `max` and `clamp` desugar to. Each arm equals the other on the
/// switching surface, so the value is continuous there and no sensitivity jump exists.
fn is_continuous_select(c: &Condition, then_e: &Expression, else_e: &Expression) -> bool {
    let same = |a: &Expression, b: &Expression| format!("{a:?}") == format!("{b:?}");
    let Condition::Compare(l, _, r) = c else {
        return false;
    };
    // `if (a >= b) a else b`, and the mirror image.
    if (same(then_e, l) && same(else_e, r)) || (same(then_e, r) && same(else_e, l)) {
        return true;
    }
    // `if (x <= lo) lo else if (x >= hi) hi else x`.
    let Expression::Conditional(inner, inner_then, inner_else) = else_e else {
        return false;
    };
    let Condition::Compare(inner_l, _, inner_r) = inner.as_ref() else {
        return false;
    };
    same(then_e, r) && same(inner_l, l) && same(inner_then, inner_r) && same(inner_else, l)
}

/// Every condition in the **unresolved** `[odes]` program `stmts` that reads an individual
/// parameter. A resolved program reports nothing, so call this before the slot rewrite.
pub(super) fn conditions_reading_params(
    stmts: &[Statement],
    states: &[String],
    params: &[String],
) -> Vec<ConditionReads> {
    let mut w = Walker {
        states,
        params,
        scope: Scope::new(),
        found: Vec::new(),
    };
    // Iterate so an intermediate read before its assignment (it then holds the previous
    // evaluation's value) is charged what it is eventually assigned. Reads only grow, over a
    // finite set, so this terminates; the cap is a backstop.
    for _ in 0..8 {
        let before = w.scope.clone();
        w.stmts(stmts, &Reads::default());
        if w.scope == before {
            break;
        }
    }
    w.found
}

#[cfg(test)]
#[path = "ode_switch_tests.rs"]
mod tests;
