//! Which `[odes]` `if` conditions read an individual parameter (#2649).
//!
//! The exact derivatives differentiate the arm an `if` selects; the condition itself carries
//! no derivative. That is right when the arm is fixed for the whole integration, or when the
//! switch sits at a time no parameter can move. It is wrong for `if (tau > LAGT)`: the
//! switching time moves with `LAGT`, the state's sensitivity jumps there, and the branch-wise
//! right-hand side does not contain that jump. A θ reaching the equations only through such a
//! condition gets an analytic gradient of exactly zero, so the fit never moves it.
//!
//! This module reports, for each condition, the individual parameters that reach it and whether
//! it also reads something that varies during the integration (a state or a clock). The parser
//! then decides, against `[individual_parameters]`, which conditions depend on an estimated
//! parameter; `OdeRhsProgram::moving_switch` carries that verdict to the scope gates.
//!
//! A parameter reaches a condition directly, through an intermediate (`L = 2*LAGT` then
//! `if (tau > L)`), or through a state: with `d/dt(tau) = RATE`, `if (tau > 2)` switches at
//! `2/RATE`. A state carries what its equation reads, its `init(...)`, its built-in input-rate
//! terms and the dose attributes, since which compartment is dosed is a property of the data.
//! `TAD` and `TAFD` carry the lag parameters, since a dose's arrival anchors them.
//!
//! `min` and `max` desugar to conditionals but are continuous across their switch, where the
//! branch-wise derivative is exact, so [`is_continuous_select`] exempts them, and `clamp` with
//! it when its bounds are ordered literals. [`arms_agree_on_switch`] extends that to any `if`
//! whose arms meet where its condition flips, like `if (rb < C) (C / rb)^0.16 else 1.0`: the
//! jump in the sensitivity is `(f⁻ − f⁺)·∂τ/∂p`, and a continuous `f` has none. Anything
//! neither can prove continuous is reported.

use super::{eval_expression, with_model_time, BinOp, CmpOp, Condition, Expression, Statement};
use rand::{rngs::StdRng, RngExt, SeedableRng};
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

/// One `[odes]` condition that at least one individual parameter reaches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ConditionReads {
    /// The individual parameters that reach it.
    pub(super) params: Vec<String>,
    /// Whether it also reads a state or clock. Without one, the condition is decided by the
    /// parameters alone unless a parameter itself reads `TIME`, which only the caller knows.
    pub(super) moves: bool,
}

/// The names a condition can read, and what reaches the states from outside the equations.
pub(super) struct SwitchInputs<'a> {
    pub(super) states: &'a [String],
    pub(super) params: &'a [String],
    /// Parameters a dose applies (`F`, `ALAG`, `D{n}`, …), which reach every state.
    pub(super) dose_params: Vec<String>,
    /// The lag parameters among [`Self::dose_params`], which also reach `TAD` and `TAFD`.
    pub(super) lag_params: Vec<String>,
    /// Per state slot, the parameters its input-rate terms read.
    pub(super) forcing_params: Vec<(usize, Vec<String>)>,
    /// The `init(state) = <expr>` directives, by state slot.
    pub(super) inits: &'a [(usize, Expression)],
}

/// What each intermediate or state holds, by upper-cased name.
type Scope = HashMap<String, Reads>;

struct Walker<'a> {
    /// Names resolve case-insensitively in `[odes]`, as in `ode_linearity`.
    inputs: &'a SwitchInputs<'a>,
    scope: Scope,
    /// The parameters reaching each state; seeded, then grown from the equations.
    state_reads: Scope,
    found: Vec<ConditionReads>,
}

impl Walker<'_> {
    fn name(&self, n: &str, out: &mut Reads) {
        let key = n.to_ascii_uppercase();
        if let Some(r) = self.scope.get(&key) {
            out.merge(r);
        } else if let Some(r) = self.state_reads.get(&key) {
            out.merge(r);
            out.moves = true;
        } else if key == "T" || key == "TIME" {
            out.moves = true;
        } else if key == "TAD" || key == "TAFD" {
            out.moves = true;
            out.params.extend(self.inputs.lag_params.iter().cloned());
        } else if let Some(p) = self
            .inputs
            .params
            .iter()
            .find(|p| p.eq_ignore_ascii_case(n))
        {
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
                if !is_continuous_select(c, t, f) && !arms_agree_on_switch(c, t, f) {
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
                Statement::DiffEq(name, e) => {
                    self.inline_conditions(e);
                    let mut r = self.expr_reads(e);
                    r.merge(ctx);
                    self.state_reads
                        .entry(name.to_ascii_uppercase())
                        .or_default()
                        .merge(&r);
                }
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

/// `true` when the select is provably continuous at its switch, so no sensitivity jump exists.
///
/// `if (a ? b) a else b` and its mirror image, which `min` and `max` desugar to, switch only
/// where the arms are equal, whatever the comparison. `clamp` desugars to
/// `if (x <= lo) lo else if (x >= hi) hi else x`, which is continuous only for `lo <= hi`:
/// with `clamp(tau, LAGT, 0.5)` and `LAGT > 0.5` it jumps from `LAGT` to `0.5`. So the bounds
/// must be ordered literals, and the comparisons must point the way `clamp`'s do.
fn is_continuous_select(c: &Condition, then_e: &Expression, else_e: &Expression) -> bool {
    let same = |a: &Expression, b: &Expression| format!("{a:?}") == format!("{b:?}");
    let Condition::Compare(l, op, r) = c else {
        return false;
    };
    if (same(then_e, l) && same(else_e, r)) || (same(then_e, r) && same(else_e, l)) {
        return true;
    }
    let Expression::Conditional(inner, inner_then, inner_else) = else_e else {
        return false;
    };
    let Condition::Compare(inner_l, inner_op, inner_r) = inner.as_ref() else {
        return false;
    };
    let (Expression::Literal(lo), Expression::Literal(hi)) = (r, inner_r) else {
        return false;
    };
    lo <= hi
        && matches!(op, CmpOp::Le | CmpOp::Lt)
        && matches!(inner_op, CmpOp::Ge | CmpOp::Gt)
        && same(then_e, r)
        && same(inner_l, l)
        && same(inner_then, inner_r)
        && same(inner_else, l)
}

/// Switching-surface points [`arms_agree_on_switch`] requires before it trusts the arms.
const SWITCH_POINTS: usize = 16;
/// Pairs of random points it may try while looking for those crossings.
const SWITCH_PAIRS: usize = 2000;
/// Bisection steps per crossing, enough to reach `f64` resolution along the segment.
const SWITCH_BISECTIONS: usize = 80;
/// How close to zero `l - r` must get, relative to `1 + |l| + |r|`. A sign change that does
/// not reach this is a pole, such as `C / x` crossing `x = 0`, not a switch.
const SWITCH_ROOT_TOL: f64 = 1e-9;
/// The gap between the arms that is rounding alone, in units of `EPSILON·(|a| + |b|)`.
const SWITCH_ARM_ULPS: f64 = 16.0;
/// How far along the segment, as a fraction of it, to step off a crossing to read the gap there.
const SWITCH_STEP: f64 = 1e-8;
/// How much smaller the gap must be at the crossing than one [`SWITCH_STEP`] away. Arms that
/// meet shrink it to rounding, measured at 2.5e-8 at worst; a jump leaves it where it was. A
/// threshold on the gap itself could not tell these apart: it would pass a small jump between
/// small arms, and a large one between arms that share a large offset.
const SWITCH_ARM_SHRINK: f64 = 1e-6;

/// The names an expression reads, or `None` when it holds anything this proof cannot sample:
/// a nested conditional, a step function or `abs`, whose jump or kink could sit outside the
/// sampled range; and what has no free value here (θ, η, `MIXNUM`, a network output, a level
/// block).
fn sampled_names(e: &Expression, names: &mut BTreeSet<String>) -> Option<()> {
    let mut ok = true;
    super::visit_expr_nodes(e, &mut |n| match n {
        Expression::Variable(v) => {
            names.insert(v.clone());
        }
        Expression::Literal(_) | Expression::Time | Expression::Power(..) => {}
        Expression::BinOp(_, op, _) => ok &= !matches!(op, BinOp::Mod),
        Expression::UnaryFn(f, _) => {
            ok &= !matches!(f.as_str(), "floor" | "ceil" | "round" | "abs")
        }
        _ => ok = false,
    });
    ok.then_some(())
}

/// `true` when `e` is affine in the names it reads.
fn is_affine(e: &Expression) -> bool {
    let literal = |e: &Expression| matches!(e, Expression::Literal(_));
    match e {
        Expression::Literal(_) | Expression::Variable(_) | Expression::Time => true,
        Expression::BinOp(a, BinOp::Add | BinOp::Sub, b) => is_affine(a) && is_affine(b),
        Expression::BinOp(a, BinOp::Mul, b) => {
            (literal(a) && is_affine(b)) || (is_affine(a) && literal(b))
        }
        Expression::BinOp(a, BinOp::Div, b) => is_affine(a) && literal(b),
        _ => false,
    }
}

/// `true` when `l = r` is a single hyperplane: both sides affine, or an affine ratio against a
/// literal. Sampling sees only a bounded box, so a surface with several components, like
/// `x * (x - 3000) = 0`, could hide one whose arms miss beyond it. A hyperplane reaches the box
/// wherever it goes, and arms that agree on part of it agree on all of it if they are analytic,
/// which is why [`sampled_names`] refuses what is not.
fn is_one_surface(l: &Expression, r: &Expression) -> bool {
    let ratio = |e: &Expression| matches!(e, Expression::BinOp(a, BinOp::Div, b) if is_affine(a) && is_affine(b));
    let literal = |e: &Expression| matches!(e, Expression::Literal(_));
    (is_affine(l) && is_affine(r)) || (ratio(l) && literal(r)) || (literal(l) && ratio(r))
}

/// `true` when both arms of `if (l ? r)` agree wherever `l = r`, so the select is continuous
/// at its switch and no sensitivity jump exists there.
///
/// Evaluated rather than simplified, so it needs no algebra: it draws pairs of random points
/// over every name the select reads, bisects each pair that straddles the switch, and compares
/// the arms at the crossing. Two different analytic arms meet on a set of measure zero within
/// the surface, so [`SWITCH_POINTS`] crossings separate them. Each name is free, including an
/// intermediate: arms that agree for every value of it agree for the value it takes. So an
/// intermediate can only lose the proof, and a select it loses stays on finite differences.
fn arms_agree_on_switch(c: &Condition, then_e: &Expression, else_e: &Expression) -> bool {
    let Condition::Compare(l, _, r) = c else {
        return false;
    };
    if !is_one_surface(l, r) {
        return false;
    }
    let mut names = BTreeSet::new();
    for e in [l, r, then_e, else_e] {
        if sampled_names(e, &mut names).is_none() {
            return false;
        }
    }
    let names: Vec<String> = names.into_iter().collect();
    // One value per name, then the clock, which is not a name.
    let eval = |e: &Expression, x: &[f64]| {
        let vars: HashMap<String, f64> = names.iter().cloned().zip(x.iter().copied()).collect();
        with_model_time(x[names.len()], || {
            eval_expression(e, &[], &[], &vars, &vars, &[])
        })
    };
    let gap = |x: &[f64]| eval(l, x) - eval(r, x);
    let mut rng = StdRng::seed_from_u64(2649);
    // Log-uniform over about six decades, one value in four negative: an identity that holds
    // only for positive values must not pass.
    let mut draw = || -> Vec<f64> {
        (0..=names.len())
            .map(|_| {
                let v = rng.random_range(-7.0_f64..7.0).exp();
                if rng.random_bool(0.25) {
                    -v
                } else {
                    v
                }
            })
            .collect()
    };
    let mut crossings = 0;
    for _ in 0..SWITCH_PAIRS {
        let (a, b) = (draw(), draw());
        let (ga, gb) = (gap(&a), gap(&b));
        if !(ga * gb < 0.0) {
            continue;
        }
        let at = |t: f64| -> Vec<f64> { a.iter().zip(&b).map(|(p, q)| p + t * (q - p)).collect() };
        let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
        for _ in 0..SWITCH_BISECTIONS {
            let mid = 0.5 * (lo + hi);
            if gap(&at(mid)) * ga > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let t = 0.5 * (lo + hi);
        let x = at(t);
        let (lv, rv) = (eval(l, &x), eval(r, &x));
        if !((lv - rv).abs() <= SWITCH_ROOT_TOL * (1.0 + lv.abs() + rv.abs())) {
            continue;
        }
        let (tv, ev) = (eval(then_e, &x), eval(else_e, &x));
        if !tv.is_finite() && !ev.is_finite() {
            continue;
        }
        let miss = (tv - ev).abs();
        let x_beside = at(t + SWITCH_STEP);
        let beside = (eval(then_e, &x_beside) - eval(else_e, &x_beside)).abs();
        let rounding = SWITCH_ARM_ULPS * f64::EPSILON * (tv.abs() + ev.abs());
        if !(miss <= rounding || miss <= SWITCH_ARM_SHRINK * beside) {
            return false;
        }
        crossings += 1;
        if crossings == SWITCH_POINTS {
            return true;
        }
    }
    false
}

/// Every condition in the **unresolved** `[odes]` program `stmts` that an individual parameter
/// reaches. A resolved program reports nothing, so call this before the slot rewrite.
pub(super) fn conditions_reading_params(
    stmts: &[Statement],
    inputs: &SwitchInputs,
) -> Vec<ConditionReads> {
    let mut w = Walker {
        inputs,
        scope: Scope::new(),
        state_reads: Scope::new(),
        found: Vec::new(),
    };
    for s in inputs.states {
        let mut r = Reads::default();
        r.params.extend(inputs.dose_params.iter().cloned());
        w.state_reads.insert(s.to_ascii_uppercase(), r);
    }
    for (slot, params) in &inputs.forcing_params {
        if let Some(r) = inputs
            .states
            .get(*slot)
            .and_then(|s| w.state_reads.get_mut(&s.to_ascii_uppercase()))
        {
            r.params.extend(params.iter().cloned());
        }
    }
    for (slot, e) in inputs.inits {
        let r = w.expr_reads(e);
        if let Some(reads) = inputs
            .states
            .get(*slot)
            .and_then(|s| w.state_reads.get_mut(&s.to_ascii_uppercase()))
        {
            reads.merge(&r);
        }
    }
    // Iterate to a fixed point: a state reaches what reads it however the lines are ordered,
    // and an intermediate read before its assignment holds the previous evaluation's value.
    // Reads only grow, over a finite set, so this terminates, and only the last pass, run on
    // the settled scopes, reports.
    loop {
        let before = (w.scope.clone(), w.state_reads.clone());
        w.found.clear();
        w.stmts(stmts, &Reads::default());
        if (&w.scope, &w.state_reads) == (&before.0, &before.1) {
            return w.found;
        }
    }
}

#[cfg(test)]
#[path = "ode_switch_tests.rs"]
mod tests;
