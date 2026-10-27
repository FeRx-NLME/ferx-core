//! Structural linearity classifier for an `[odes]` block (#852).
//!
//! Decides — from the parsed expressions, before any number is evaluated — whether every
//! `d/dt(state) = …` is **affine in the states**: `Σ_j c_j·state_j + c_0`, every coefficient
//! free of states. Such a system is `du/dt = A·u + b`, which `crate::ode::linear` propagates
//! exactly by the matrix exponential instead of integrating.
//!
//! It is structural on purpose. A numerical probe of the Jacobian cannot certify linearity: a
//! Michaelis–Menten term evaluated far below `Km` is linear to every digit a probe can see, and
//! a model certified that way returns a plausible objective, fast, that is wrong as soon as a
//! subject's concentration reaches `Km`. Here a state under a division, a transcendental, a
//! non-unit power, a product with another state, or inside an `if` condition rejects the model
//! outright, whatever the values happen to be.
//!
//! Each expression gets a [`Degree`] in the states. Intermediates carry the degree of what
//! they were last assigned, so `C = central/V` then `d/dt(central) = -CL*C` is affine, while
//! `E = EMAX*C/(EC50 + C)` taints everything that reads `E`.
//!
//! What this module does **not** decide: whether the system is *time-invariant* (reading
//! `TIME`/`T`/`TAFD`/`TAD` is gated by the existing time-read flags on `OdeRhsProgram`), and
//! whether the absorption forcings (`first_order(...)` etc., already lifted out of the
//! equations by the parser) keep the input constant over a segment — `OdeSpec` gates those.

use super::{BinOp, Condition, Expression, Statement};
use std::collections::HashMap;

/// Degree of an expression in the ODE states, ordered so `max` is the degree of a sum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Degree {
    /// Free of states — a coefficient.
    Const,
    /// `Σ c_j·state_j + c_0` with state-free `c`.
    Affine,
    /// Anything else.
    Nonlinear,
}

/// The degree of each intermediate, by upper-cased name.
type Scope = HashMap<String, Degree>;

struct Classifier<'a> {
    /// Upper-cased state names. Names resolve case-insensitively in `[odes]`
    /// (`build_ode_spec` registers every case variant on one slot), so the match here does too.
    states: &'a [String],
    /// Degrees reached by the previous pass, read for an intermediate that is used before
    /// this pass has assigned it (a read of the previous RHS call's value, or of 0).
    prior: &'a Scope,
    /// Set as soon as a state reaches an `if` condition — a branch selected by the state
    /// is a piecewise system, not a linear one, whatever each branch contains.
    state_in_condition: bool,
}

impl Classifier<'_> {
    fn is_state(&self, name: &str) -> bool {
        self.states.iter().any(|s| s.eq_ignore_ascii_case(name))
    }

    fn expr(&mut self, e: &Expression, scope: &Scope) -> Degree {
        use Degree::*;
        match e {
            Expression::Literal(_)
            | Expression::Theta(_)
            | Expression::Eta(_)
            | Expression::Time
            | Expression::MixNum
            | Expression::Covariate(_)
            | Expression::CovariateIdx(_)
            | Expression::NnOutput { .. } => Const,
            Expression::Variable(name) => {
                if self.is_state(name) {
                    return Affine;
                }
                let key = name.to_ascii_uppercase();
                scope
                    .get(&key)
                    .or_else(|| self.prior.get(&key))
                    .copied()
                    .unwrap_or(Const)
            }
            // Only produced by `resolve_variable_indices`, which runs after this pass. Met here
            // it means the caller handed in a resolved program, whose names are gone — so the
            // slot could be a state, and the honest answer is "cannot certify".
            Expression::VariableIdx(_) => Nonlinear,
            Expression::BinOp(l, op, r) => {
                let (l, r) = (self.expr(l, scope), self.expr(r, scope));
                match op {
                    BinOp::Add | BinOp::Sub => l.max(r),
                    BinOp::Mul if l >= Affine && r >= Affine => Nonlinear,
                    BinOp::Mul => l.max(r),
                    BinOp::Div if r >= Affine => Nonlinear,
                    BinOp::Div => l,
                    BinOp::Mod if l.max(r) >= Affine => Nonlinear,
                    BinOp::Mod => Const,
                }
            }
            Expression::UnaryFn(_, x) => match self.expr(x, scope) {
                Const => Const,
                _ => Nonlinear,
            },
            Expression::Power(base, exp) => {
                let (b, x) = (self.expr(base, scope), self.expr(exp, scope));
                match (b, x) {
                    (Const, Const) => Const,
                    // `state^1` is the only power that stays affine.
                    (b, Const) if matches!(**exp, Expression::Literal(v) if v == 1.0) => b,
                    _ => Nonlinear,
                }
            }
            Expression::Conditional(c, a, b) => {
                self.condition(c, scope);
                self.expr(a, scope).max(self.expr(b, scope))
            }
            Expression::ThetaGather { idx, .. } => match self.expr(idx, scope) {
                Const => Const,
                _ => Nonlinear,
            },
        }
    }

    fn condition(&mut self, c: &Condition, scope: &Scope) {
        let deg = match c {
            Condition::Compare(a, _, b) => self.expr(a, scope).max(self.expr(b, scope)),
            Condition::And(a, b) | Condition::Or(a, b) => {
                self.condition(a, scope);
                self.condition(b, scope);
                Degree::Const
            }
            Condition::Not(a) => {
                self.condition(a, scope);
                Degree::Const
            }
            Condition::Present(e) => self.expr(e, scope),
        };
        if deg != Degree::Const {
            self.state_in_condition = true;
        }
    }

    /// Walk `stmts` in order, updating `scope`; returns the highest degree any `d/dt` reached.
    fn stmts(&mut self, stmts: &[Statement], scope: &mut Scope) -> Degree {
        let mut worst = Degree::Const;
        for s in stmts {
            match s {
                Statement::Assign(name, e) => {
                    let d = self.expr(e, scope);
                    scope.insert(name.to_ascii_uppercase(), d);
                }
                Statement::DiffEq(_, e) => worst = worst.max(self.expr(e, scope)),
                Statement::If {
                    branches,
                    else_body,
                } => {
                    // Each arm sees the scope as it was before the `if`; afterwards a variable
                    // has the worst degree any path can leave it with — including the
                    // fall-through path when there is no `else`.
                    let entry = scope.clone();
                    let mut merged = entry.clone();
                    let mut arms: Vec<&[Statement]> =
                        branches.iter().map(|(_, b)| b.as_slice()).collect();
                    for (c, _) in branches {
                        self.condition(c, &entry);
                    }
                    if let Some(e) = else_body {
                        arms.push(e);
                    }
                    for body in arms {
                        let mut arm_scope = entry.clone();
                        worst = worst.max(self.stmts(body, &mut arm_scope));
                        for (k, d) in arm_scope {
                            let slot = merged.entry(k).or_insert(Degree::Const);
                            *slot = (*slot).max(d);
                        }
                    }
                    *scope = merged;
                }
                // The resolved forms carry slot indices, not names: same reasoning as
                // `Expression::VariableIdx`.
                Statement::AssignIdx(..)
                | Statement::DiffEqIdx(..)
                | Statement::AssignBc(..)
                | Statement::DiffEqBc(..) => worst = Degree::Nonlinear,
            }
        }
        worst
    }
}

/// `true` when every `d/dt(...)` in `stmts` is affine in `states` and no state reaches an `if`
/// condition. `stmts` must be the **unresolved** `[odes]` program (names, not slots); a
/// resolved one is conservatively reported as not linear.
pub(super) fn stmts_affine_in_states(stmts: &[Statement], states: &[String]) -> bool {
    // Iterate to a fixed point so an intermediate read before its assignment (it then holds
    // the previous evaluation's value) is charged the degree it is eventually assigned.
    // Degrees only rise, over a finite lattice, so this terminates; the cap is a backstop.
    let mut prior = Scope::new();
    for _ in 0..8 {
        let mut c = Classifier {
            states,
            prior: &prior,
            state_in_condition: false,
        };
        let mut scope = Scope::new();
        let worst = c.stmts(stmts, &mut scope);
        if c.state_in_condition || worst == Degree::Nonlinear {
            return false;
        }
        if scope == prior {
            return true;
        }
        prior = scope;
    }
    false
}

#[cfg(test)]
#[path = "ode_linearity_tests.rs"]
mod tests;
