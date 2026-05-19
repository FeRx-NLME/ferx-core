//! Shared expression / statement AST used by every model-file front-end.
//!
//! The ferx `.ferx` parser ([`crate::parser::model_parser`]) and the mrgsolve
//! front-end ([`crate::parser::mrgsolve`]) both lower their input into the
//! `Expression` / `Statement` tree defined here. Lowering to a `CompiledModel`
//! and the hot-path evaluators are also in this module so every front-end
//! gets the same indexing optimisation and the same evaluation semantics.
//!
//! Visibility is `pub(crate)` throughout — these types are deliberately not
//! part of the public crate API.

use std::collections::HashMap;

// ── AST ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub(crate) enum Expression {
    Literal(f64),
    Theta(usize),
    Eta(usize),
    Covariate(String),
    Variable(String),
    /// Same as `Variable(name)` but pre-resolved to a slot index. Produced
    /// by `resolve_variable_indices` for the `pk_param_fn` AST so the hot
    /// path doesn't pay HashMap-lookup overhead on every eval. `usize::MAX`
    /// is reserved for "unresolved" (defensive — eval treats it as 0.0).
    VariableIdx(usize),
    /// Same as `Covariate(name)` but pre-resolved to an index into a Vec
    /// aligned with `CompiledModel.referenced_covariates`. Built by
    /// `resolve_variable_indices`; the matching Vec is materialised once
    /// per call inside the `pk_param_fn` closure (`build_pk_param_fn`),
    /// reading from the caller-supplied covariate HashMap.
    CovariateIdx(usize),
    BinOp(Box<Expression>, BinOp, Box<Expression>),
    UnaryFn(String, Box<Expression>),
    Power(Box<Expression>, Box<Expression>),
    /// `if (cond) then_expr else else_expr` — value-producing inline conditional.
    Conditional(Box<Condition>, Box<Expression>, Box<Expression>),
}

#[derive(Debug, Clone)]
pub(crate) enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum CmpOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

#[derive(Debug, Clone)]
pub(crate) enum Condition {
    Compare(Expression, CmpOp, Expression),
    And(Box<Condition>, Box<Condition>),
    Or(Box<Condition>, Box<Condition>),
    Not(Box<Condition>),
}

/// A statement in a model block. Supports plain assignments, derivative
/// assignments (only valid in `[odes]`), and if/else-if/else blocks.
#[derive(Debug, Clone)]
pub(crate) enum Statement {
    Assign(String, Expression),
    /// Same as `Assign(name, expr)` but pre-resolved to a slot index for
    /// the `pk_param_fn` hot path. See `Expression::VariableIdx`.
    AssignIdx(usize, Expression),
    /// `d/dt(NAME) = expr` — only legal in `[odes]` blocks.
    DiffEq(String, Expression),
    /// One or more `if (cond) { ... }` arms followed by an optional `else { ... }`.
    /// Each arm in `branches` is `(condition, body)`.
    If {
        branches: Vec<(Condition, Vec<Statement>)>,
        else_body: Option<Vec<Statement>>,
    },
}

// ── Covariate collection ────────────────────────────────────────────────────

/// Walk an expression tree and accumulate every covariate name it references.
pub(crate) fn collect_covariates(expr: &Expression, out: &mut std::collections::HashSet<String>) {
    match expr {
        Expression::Covariate(name) => {
            out.insert(name.clone());
        }
        Expression::BinOp(lhs, _, rhs) => {
            collect_covariates(lhs, out);
            collect_covariates(rhs, out);
        }
        Expression::UnaryFn(_, arg) => collect_covariates(arg, out),
        Expression::Power(base, exp) => {
            collect_covariates(base, out);
            collect_covariates(exp, out);
        }
        Expression::Conditional(cond, t, e) => {
            collect_covariates_in_condition(cond, out);
            collect_covariates(t, out);
            collect_covariates(e, out);
        }
        _ => {}
    }
}

pub(crate) fn collect_covariates_in_condition(
    cond: &Condition,
    out: &mut std::collections::HashSet<String>,
) {
    match cond {
        Condition::Compare(l, _, r) => {
            collect_covariates(l, out);
            collect_covariates(r, out);
        }
        Condition::And(l, r) | Condition::Or(l, r) => {
            collect_covariates_in_condition(l, out);
            collect_covariates_in_condition(r, out);
        }
        Condition::Not(c) => collect_covariates_in_condition(c, out),
    }
}

/// Walk a list of statements (assignments and if-blocks) and accumulate every
/// covariate name they reference.
pub(crate) fn collect_covariates_in_stmts(
    stmts: &[Statement],
    out: &mut std::collections::HashSet<String>,
) {
    for s in stmts {
        match s {
            Statement::Assign(_, e)
            | Statement::AssignIdx(_, e)
            | Statement::DiffEq(_, e) => collect_covariates(e, out),
            Statement::If {
                branches,
                else_body,
            } => {
                for (cond, body) in branches {
                    collect_covariates_in_condition(cond, out);
                    collect_covariates_in_stmts(body, out);
                }
                if let Some(eb) = else_body {
                    collect_covariates_in_stmts(eb, out);
                }
            }
        }
    }
}

// ── HashMap-keyed evaluator (parse-time / ODE RHS path) ─────────────────────

pub(crate) fn eval_expression(
    expr: &Expression,
    theta: &[f64],
    eta: &[f64],
    covariates: &HashMap<String, f64>,
    vars: &HashMap<String, f64>,
) -> f64 {
    match expr {
        Expression::Literal(v) => *v,
        Expression::Theta(i) => theta[*i],
        Expression::Eta(i) => eta[*i],
        Expression::Covariate(name) => covariates.get(name).copied().unwrap_or(0.0),
        Expression::Variable(name) => vars.get(name).copied().unwrap_or(0.0),
        Expression::VariableIdx(_) | Expression::CovariateIdx(_) => {
            debug_assert!(false, "indexed expression reached unindexed eval_expression");
            0.0
        }
        Expression::BinOp(lhs, op, rhs) => {
            let l = eval_expression(lhs, theta, eta, covariates, vars);
            let r = eval_expression(rhs, theta, eta, covariates, vars);
            match op {
                BinOp::Add => l + r,
                BinOp::Sub => l - r,
                BinOp::Mul => l * r,
                BinOp::Div => {
                    if r.abs() < 1e-30 {
                        0.0
                    } else {
                        l / r
                    }
                }
            }
        }
        Expression::UnaryFn(name, arg) => {
            let v = eval_expression(arg, theta, eta, covariates, vars);
            match name.as_str() {
                "exp" => v.exp(),
                "log" | "ln" => v.max(1e-30).ln(),
                "sqrt" => v.max(0.0).sqrt(),
                "abs" => v.abs(),
                "inv_logit" | "expit" => {
                    // Numerically stable: avoid exp overflow for very negative v
                    if v >= 0.0 {
                        1.0 / (1.0 + (-v).exp())
                    } else {
                        let e = v.exp();
                        e / (1.0 + e)
                    }
                }
                "logit" => {
                    let clamped = v.clamp(1e-15, 1.0 - 1e-15);
                    (clamped / (1.0 - clamped)).ln()
                }
                _ => v,
            }
        }
        Expression::Power(base, exp) => {
            let b = eval_expression(base, theta, eta, covariates, vars);
            let e = eval_expression(exp, theta, eta, covariates, vars);
            b.powf(e)
        }
        Expression::Conditional(cond, t, e) => {
            if eval_condition(cond, theta, eta, covariates, vars) {
                eval_expression(t, theta, eta, covariates, vars)
            } else {
                eval_expression(e, theta, eta, covariates, vars)
            }
        }
    }
}

pub(crate) fn eval_condition(
    cond: &Condition,
    theta: &[f64],
    eta: &[f64],
    covariates: &HashMap<String, f64>,
    vars: &HashMap<String, f64>,
) -> bool {
    match cond {
        Condition::Compare(l, op, r) => {
            let lv = eval_expression(l, theta, eta, covariates, vars);
            let rv = eval_expression(r, theta, eta, covariates, vars);
            match op {
                CmpOp::Lt => lv < rv,
                CmpOp::Le => lv <= rv,
                CmpOp::Gt => lv > rv,
                CmpOp::Ge => lv >= rv,
                CmpOp::Eq => lv == rv,
                CmpOp::Ne => lv != rv,
            }
        }
        Condition::And(l, r) => {
            eval_condition(l, theta, eta, covariates, vars)
                && eval_condition(r, theta, eta, covariates, vars)
        }
        Condition::Or(l, r) => {
            eval_condition(l, theta, eta, covariates, vars)
                || eval_condition(r, theta, eta, covariates, vars)
        }
        Condition::Not(c) => !eval_condition(c, theta, eta, covariates, vars),
    }
}

// ── Indexed-form evaluator (hot path inside pk_param_fn) ────────────────────

/// Indexed-form evaluator: `vars` is a `Vec<f64>` indexed by parse-time
/// variable slot; `covariates` is a `Vec<f64>` aligned to
/// `CompiledModel.referenced_covariates`. Hot-path replacement for the
/// HashMap-keyed `eval_expression` — eliminates the per-call string hash
/// + probe in the `pk_param_fn` closure. Falls back to 0.0 for the
/// HashMap-keyed variants (Variable/Covariate) since callers running
/// the indexed path have already resolved every name.
pub(crate) fn eval_expression_indexed(
    expr: &Expression,
    theta: &[f64],
    eta: &[f64],
    covariates: &[f64],
    vars: &[f64],
) -> f64 {
    match expr {
        Expression::Literal(v) => *v,
        Expression::Theta(i) => theta[*i],
        Expression::Eta(i) => eta[*i],
        Expression::VariableIdx(i) => vars.get(*i).copied().unwrap_or(0.0),
        Expression::CovariateIdx(i) => covariates.get(*i).copied().unwrap_or(0.0),
        Expression::Covariate(_) | Expression::Variable(_) => {
            debug_assert!(false, "unresolved name reached eval_expression_indexed");
            0.0
        }
        Expression::BinOp(lhs, op, rhs) => {
            let l = eval_expression_indexed(lhs, theta, eta, covariates, vars);
            let r = eval_expression_indexed(rhs, theta, eta, covariates, vars);
            match op {
                BinOp::Add => l + r,
                BinOp::Sub => l - r,
                BinOp::Mul => l * r,
                BinOp::Div => {
                    if r.abs() < 1e-30 {
                        0.0
                    } else {
                        l / r
                    }
                }
            }
        }
        Expression::UnaryFn(name, arg) => {
            let v = eval_expression_indexed(arg, theta, eta, covariates, vars);
            match name.as_str() {
                "exp" => v.exp(),
                "log" | "ln" => v.max(1e-30).ln(),
                "sqrt" => v.max(0.0).sqrt(),
                "abs" => v.abs(),
                "inv_logit" | "expit" => {
                    if v >= 0.0 {
                        1.0 / (1.0 + (-v).exp())
                    } else {
                        let e = v.exp();
                        e / (1.0 + e)
                    }
                }
                "logit" => {
                    let clamped = v.clamp(1e-15, 1.0 - 1e-15);
                    (clamped / (1.0 - clamped)).ln()
                }
                _ => v,
            }
        }
        Expression::Power(base, exp) => {
            let b = eval_expression_indexed(base, theta, eta, covariates, vars);
            let e = eval_expression_indexed(exp, theta, eta, covariates, vars);
            b.powf(e)
        }
        Expression::Conditional(cond, t, e) => {
            if eval_condition_indexed(cond, theta, eta, covariates, vars) {
                eval_expression_indexed(t, theta, eta, covariates, vars)
            } else {
                eval_expression_indexed(e, theta, eta, covariates, vars)
            }
        }
    }
}

pub(crate) fn eval_condition_indexed(
    cond: &Condition,
    theta: &[f64],
    eta: &[f64],
    covariates: &[f64],
    vars: &[f64],
) -> bool {
    match cond {
        Condition::Compare(l, op, r) => {
            let lv = eval_expression_indexed(l, theta, eta, covariates, vars);
            let rv = eval_expression_indexed(r, theta, eta, covariates, vars);
            match op {
                CmpOp::Lt => lv < rv,
                CmpOp::Le => lv <= rv,
                CmpOp::Gt => lv > rv,
                CmpOp::Ge => lv >= rv,
                CmpOp::Eq => lv == rv,
                CmpOp::Ne => lv != rv,
            }
        }
        Condition::And(l, r) => {
            eval_condition_indexed(l, theta, eta, covariates, vars)
                && eval_condition_indexed(r, theta, eta, covariates, vars)
        }
        Condition::Or(l, r) => {
            eval_condition_indexed(l, theta, eta, covariates, vars)
                || eval_condition_indexed(r, theta, eta, covariates, vars)
        }
        Condition::Not(c) => !eval_condition_indexed(c, theta, eta, covariates, vars),
    }
}

/// Indexed-form statement executor; mirror of `eval_statements`. Only
/// handles `AssignIdx` + `If`; if any non-indexed variant slips through,
/// it falls back to a no-op (defensive — caller should ensure the AST
/// was resolved). `DiffEq` is not supported here because the indexed
/// path is only used by `pk_param_fn` (no derivatives).
pub(crate) fn eval_statements_indexed(
    stmts: &[Statement],
    theta: &[f64],
    eta: &[f64],
    covariates: &[f64],
    vars: &mut [f64],
) {
    for s in stmts {
        match s {
            Statement::AssignIdx(idx, expr) => {
                let v = eval_expression_indexed(expr, theta, eta, covariates, vars);
                if let Some(slot) = vars.get_mut(*idx) {
                    *slot = v;
                }
            }
            Statement::If {
                branches,
                else_body,
            } => {
                let mut taken = false;
                for (cond, body) in branches {
                    if eval_condition_indexed(cond, theta, eta, covariates, vars) {
                        eval_statements_indexed(body, theta, eta, covariates, vars);
                        taken = true;
                        break;
                    }
                }
                if !taken {
                    if let Some(eb) = else_body {
                        eval_statements_indexed(eb, theta, eta, covariates, vars);
                    }
                }
            }
            // Non-indexed Assign/DiffEq shouldn't appear in a resolved AST
            // for `pk_param_fn`. Silently skip.
            Statement::Assign(_, _) | Statement::DiffEq(_, _) => {}
        }
    }
}

// ── Index resolution (HashMap-keyed AST → indexed AST) ──────────────────────

/// Walk the AST in `stmts` and replace every `Statement::Assign(name, e)`
/// with `Statement::AssignIdx(idx, e)`, every `Expression::Variable(name)`
/// with `Expression::VariableIdx(idx)`, and every `Expression::Covariate(name)`
/// with `Expression::CovariateIdx(idx)`. Indices come from `var_idx` and
/// `cov_idx`. Variables not in `var_idx` get `usize::MAX` (eval returns 0.0).
///
/// Used by `build_pk_param_fn` so the hot-loop closure can run on
/// `Vec<f64>` slots instead of paying per-call HashMap-lookup overhead.
pub(crate) fn resolve_variable_indices(
    stmts: &mut [Statement],
    var_idx: &HashMap<String, usize>,
    cov_idx: &HashMap<String, usize>,
) {
    for s in stmts.iter_mut() {
        match s {
            Statement::Assign(name, expr) => {
                resolve_expr_indices(expr, var_idx, cov_idx);
                let i = var_idx.get(name).copied().unwrap_or(usize::MAX);
                let taken_expr = std::mem::replace(expr, Expression::Literal(0.0));
                *s = Statement::AssignIdx(i, taken_expr);
            }
            Statement::AssignIdx(_, expr) => {
                resolve_expr_indices(expr, var_idx, cov_idx);
            }
            Statement::DiffEq(_, expr) => {
                resolve_expr_indices(expr, var_idx, cov_idx);
            }
            Statement::If {
                branches,
                else_body,
            } => {
                for (cond, body) in branches.iter_mut() {
                    resolve_condition_indices(cond, var_idx, cov_idx);
                    resolve_variable_indices(body, var_idx, cov_idx);
                }
                if let Some(eb) = else_body {
                    resolve_variable_indices(eb, var_idx, cov_idx);
                }
            }
        }
    }
}

pub(crate) fn resolve_expr_indices(
    expr: &mut Expression,
    var_idx: &HashMap<String, usize>,
    cov_idx: &HashMap<String, usize>,
) {
    match expr {
        Expression::Variable(name) => {
            let i = var_idx.get(name).copied().unwrap_or(usize::MAX);
            *expr = Expression::VariableIdx(i);
        }
        Expression::Covariate(name) => {
            let i = cov_idx.get(name).copied().unwrap_or(usize::MAX);
            *expr = Expression::CovariateIdx(i);
        }
        Expression::BinOp(l, _, r) => {
            resolve_expr_indices(l, var_idx, cov_idx);
            resolve_expr_indices(r, var_idx, cov_idx);
        }
        Expression::UnaryFn(_, a) => resolve_expr_indices(a, var_idx, cov_idx),
        Expression::Power(b, e) => {
            resolve_expr_indices(b, var_idx, cov_idx);
            resolve_expr_indices(e, var_idx, cov_idx);
        }
        Expression::Conditional(cond, t, e) => {
            resolve_condition_indices(cond, var_idx, cov_idx);
            resolve_expr_indices(t, var_idx, cov_idx);
            resolve_expr_indices(e, var_idx, cov_idx);
        }
        Expression::Literal(_)
        | Expression::Theta(_)
        | Expression::Eta(_)
        | Expression::VariableIdx(_)
        | Expression::CovariateIdx(_) => {}
    }
}

pub(crate) fn resolve_condition_indices(
    cond: &mut Condition,
    var_idx: &HashMap<String, usize>,
    cov_idx: &HashMap<String, usize>,
) {
    match cond {
        Condition::Compare(l, _, r) => {
            resolve_expr_indices(l, var_idx, cov_idx);
            resolve_expr_indices(r, var_idx, cov_idx);
        }
        Condition::And(l, r) | Condition::Or(l, r) => {
            resolve_condition_indices(l, var_idx, cov_idx);
            resolve_condition_indices(r, var_idx, cov_idx);
        }
        Condition::Not(c) => resolve_condition_indices(c, var_idx, cov_idx),
    }
}

// ── HashMap-keyed statement executor (block-level / ODE path) ───────────────

/// Execute a list of statements, mutating `vars` with each assignment. ODE
/// `DiffEq` statements write into the optional `du` slice using the
/// `state_index` lookup. For `[individual_parameters]` callers, pass `None`
/// for `du` and `state_index` — encountering a `DiffEq` is then an error.
pub(crate) fn eval_statements(
    stmts: &[Statement],
    theta: &[f64],
    eta: &[f64],
    covariates: &HashMap<String, f64>,
    vars: &mut HashMap<String, f64>,
    du: Option<&mut [f64]>,
    state_index: Option<&HashMap<String, usize>>,
) {
    // The `du` borrow has to be re-passed into each recursive sub-block, so
    // shuttle it through an Option that we re-grab on each iteration.
    let mut du_opt = du;
    for s in stmts {
        match s {
            Statement::Assign(name, expr) => {
                let v = eval_expression(expr, theta, eta, covariates, vars);
                vars.insert(name.clone(), v);
            }
            Statement::AssignIdx(_, _) => {
                // Indexed-form statements are only produced by
                // `resolve_variable_indices` for the `pk_param_fn`
                // closure, which uses `eval_statements_indexed`
                // exclusively. They should never reach this evaluator;
                // silently skip if they do (defensive).
            }
            Statement::DiffEq(name, expr) => {
                let v = eval_expression(expr, theta, eta, covariates, vars);
                let idx = state_index
                    .and_then(|m| m.get(name).copied())
                    .expect("DiffEq encountered without state_index — internal error");
                if let Some(buf) = du_opt.as_deref_mut() {
                    buf[idx] = v;
                }
            }
            Statement::If {
                branches,
                else_body,
            } => {
                let mut taken = false;
                for (cond, body) in branches {
                    if eval_condition(cond, theta, eta, covariates, vars) {
                        eval_statements(
                            body,
                            theta,
                            eta,
                            covariates,
                            vars,
                            du_opt.as_deref_mut(),
                            state_index,
                        );
                        taken = true;
                        break;
                    }
                }
                if !taken {
                    if let Some(eb) = else_body {
                        eval_statements(
                            eb,
                            theta,
                            eta,
                            covariates,
                            vars,
                            du_opt.as_deref_mut(),
                            state_index,
                        );
                    }
                }
            }
        }
    }
}
