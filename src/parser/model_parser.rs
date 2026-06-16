use crate::types::*;
use regex::Regex;
use std::collections::HashMap;
use std::path::Path;
use std::sync::LazyLock;

static DIFFUSION_LINE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^(\w+)\s*~\s*([0-9eE.+-]+)(?:\s+(FIX)\b)?").unwrap());

// ── Mu-referencing pattern detection ────────────────────────────────────────

/// Anchor of a mu-referencing relationship — either a plain user-declared
/// theta (the classical case) or a `[covariate_nn]` output (Phase A M1+
/// "deep compartment model" extension; the per-individual typical value
/// comes out of an NN forward pass). Both shapes compose with eta the same
/// way: `param = anchor * exp(eta)` (lognormal) or `param = anchor + eta`
/// (additive).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MuRefAnchor {
    /// User-declared theta at this index in `theta_names`.
    Theta(usize),
    /// Output of a `[covariate_nn NAME]` block. `nn_idx` indexes the
    /// alphabetically-sorted NN list (same as `ParseCtx::nn_specs`);
    /// `output_idx` indexes the block's declared `outputs` list.
    #[allow(dead_code)]
    NnOutput { nn_idx: usize, output_idx: usize },
}

/// Walk a Mul-chain and collect direct anchor candidates — `Theta(i)` or
/// `NnOutput { nn_idx, output_idx }` — that sit at the top of the
/// multiplication tree (not inside any function call). Used by the
/// extended Pattern 1/4 detector to recognise both
/// `TVCL * exp(ETA)` (classical) and `TYPICAL_PK.CL * exp(ETA)` (DCM).
fn collect_mul_anchors(expr: &Expression, out: &mut Vec<MuRefAnchor>) {
    match expr {
        Expression::Theta(i) => out.push(MuRefAnchor::Theta(*i)),
        Expression::NnOutput { nn_idx, output_idx } => out.push(MuRefAnchor::NnOutput {
            nn_idx: *nn_idx,
            output_idx: *output_idx,
        }),
        Expression::BinOp(l, BinOp::Mul, r) => {
            collect_mul_anchors(l, out);
            collect_mul_anchors(r, out);
        }
        _ => {}
    }
}

/// Walk a Mul-chain and find the first `exp(Eta(j))` or `exp(Eta(a) + Eta(b))`,
/// returning the eta index. For the two-eta case (IIV+IOV combined pattern)
/// returns the **minimum** index; BSV etas are numbered `0..n_eta` and kappa etas
/// `n_eta..`, so min always selects the BSV eta regardless of expression order.
fn find_exp_eta_in_mul(expr: &Expression) -> Option<usize> {
    match expr {
        Expression::UnaryFn(name, arg) if name == "exp" => {
            if let Expression::Eta(j) = arg.as_ref() {
                return Some(*j);
            }
            // exp(ETA1 + ETA2) — IIV+IOV combined pattern.
            if let Expression::BinOp(l, BinOp::Add, r) = arg.as_ref() {
                let li = if let Expression::Eta(j) = l.as_ref() {
                    Some(*j)
                } else {
                    None
                };
                let ri = if let Expression::Eta(j) = r.as_ref() {
                    Some(*j)
                } else {
                    None
                };
                return match (li, ri) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    // One operand is not a bare Eta (e.g. exp(ETA + constant));
                    // return whichever index was found.
                    (a, b) => a.or(b),
                };
            }
            None
        }
        Expression::BinOp(l, BinOp::Mul, r) => {
            find_exp_eta_in_mul(l).or_else(|| find_exp_eta_in_mul(r))
        }
        _ => None,
    }
}

/// Returns `true` when `expr` contains an `Eta` node that is NOT shielded
/// inside an `exp(...)` call. Used to guard the log-normal product fallback
/// classifier: if all etas are inside exp(), the expression is log-normal.
fn has_bare_eta(expr: &Expression) -> bool {
    match expr {
        Expression::Eta(_) => true,
        // Etas inside exp() enter multiplicatively — they are not "bare".
        Expression::UnaryFn(name, _) if name == "exp" => false,
        Expression::BinOp(l, _, r) => has_bare_eta(l) || has_bare_eta(r),
        Expression::UnaryFn(_, arg) => has_bare_eta(arg),
        Expression::Power(b, e) => has_bare_eta(b) || has_bare_eta(e),
        _ => false,
    }
}

/// Collect all eta indices referenced by an expression (e.g. `Eta(2)` appears
/// inside `TVQ * exp(ETA_V2)` → `[2]`). Used to build the AD path's per-tv
/// eta-index map so parameters without etas (e.g. `Q = TVQ`) are handled
/// correctly — otherwise the AD loop would misalign `eta[i]` with `pk[i]`
/// and either apply the wrong eta or leave a pk slot at 0.
fn extract_eta_indices(expr: &Expression) -> Vec<usize> {
    let mut out = Vec::new();
    fn walk(e: &Expression, out: &mut Vec<usize>) {
        match e {
            Expression::Eta(i) => {
                if !out.contains(i) {
                    out.push(*i);
                }
            }
            Expression::BinOp(l, _, r) => {
                walk(l, out);
                walk(r, out);
            }
            Expression::UnaryFn(_, a) => walk(a, out),
            Expression::Power(b, e) => {
                walk(b, out);
                walk(e, out);
            }
            Expression::Conditional(cond, t, els) => {
                walk_eta_in_condition(cond, out);
                walk(t, out);
                walk(els, out);
            }
            _ => {}
        }
    }
    fn walk_eta_in_condition(cond: &Condition, out: &mut Vec<usize>) {
        match cond {
            Condition::Compare(l, _, r) => {
                walk(l, out);
                walk(r, out);
            }
            Condition::And(l, r) | Condition::Or(l, r) => {
                walk_eta_in_condition(l, out);
                walk_eta_in_condition(r, out);
            }
            Condition::Not(c) => walk_eta_in_condition(c, out),
        }
    }
    walk(expr, &mut out);
    out
}

/// First kappa name referenced anywhere in `expr` (as a `Variable` or
/// `Covariate` identifier), if any. Used to reject `KAPPA_*` references in a
/// Form C ODE output expression under IOV (issue #107): in the `[scaling]`
/// parse context the eta scope is BSV-only, so a kappa name there parses as an
/// unresolved identifier rather than `Eta(i)`. Walks the whole value-producing
/// tree, including conditional branches *and* the condition itself, so any
/// appearance of a kappa name (e.g. `if (KAPPA_CL > 0) ...`) is caught.
fn expr_references_kappa(expr: &Expression, kappa_names: &[String]) -> Option<String> {
    fn walk(e: &Expression, kappa: &[String]) -> Option<String> {
        match e {
            Expression::Variable(n) | Expression::Covariate(n) => {
                kappa.iter().find(|k| *k == n).cloned()
            }
            Expression::BinOp(l, _, r) => walk(l, kappa).or_else(|| walk(r, kappa)),
            Expression::UnaryFn(_, a) => walk(a, kappa),
            Expression::Power(b, e) => walk(b, kappa).or_else(|| walk(e, kappa)),
            Expression::Conditional(cond, t, els) => walk_cond(cond, kappa)
                .or_else(|| walk(t, kappa))
                .or_else(|| walk(els, kappa)),
            _ => None,
        }
    }
    fn walk_cond(c: &Condition, kappa: &[String]) -> Option<String> {
        match c {
            Condition::Compare(l, _, r) => walk(l, kappa).or_else(|| walk(r, kappa)),
            Condition::And(l, r) | Condition::Or(l, r) => {
                walk_cond(l, kappa).or_else(|| walk_cond(r, kappa))
            }
            Condition::Not(c) => walk_cond(c, kappa),
        }
    }
    walk(expr, kappa_names)
}

/// All variable names assigned anywhere in the statement tree, in
/// first-occurrence order, deduplicated. Used by `[individual_parameters]`
/// to enumerate "tv" parameters for the AD path and to populate the per-var
/// `vars` map for the ODE RHS. `DiffEq` statements are excluded since they
/// produce derivative outputs, not vars.
/// All variable names assigned anywhere in the statement tree (including
/// inside if-bodies), in first-occurrence order, deduplicated.  Used to
/// build the `defined_vars` set for `ParseCtx` so that forward references to
/// branch-local helpers resolve as `Variable` rather than `Covariate`.
/// **Do not use this for the TV output vector or pk_indices** — use
/// `top_level_assigned_vars` instead to avoid placing branch-local
/// temporaries in PK parameter slots.
fn assigned_vars_in_order(stmts: &[Statement]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    fn walk(stmts: &[Statement], out: &mut Vec<String>) {
        for s in stmts {
            match s {
                Statement::Assign(name, _) => {
                    if !out.iter().any(|n| n == name) {
                        out.push(name.clone());
                    }
                }
                Statement::AssignIdx(_, _)
                | Statement::DiffEqIdx(_, _)
                | Statement::AssignBc(_, _)
                | Statement::DiffEqBc(_, _) => {
                    // Indexed / bytecode variants only appear after
                    // `resolve_variable_indices`, which runs after all
                    // name-collecting helpers like this one. Treat as no-op
                    // for exhaustiveness.
                }
                Statement::DiffEq(_, _) => {}
                Statement::If {
                    branches,
                    else_body,
                } => {
                    for (_, body) in branches {
                        walk(body, out);
                    }
                    if let Some(eb) = else_body {
                        walk(eb, out);
                    }
                }
            }
        }
    }
    walk(stmts, &mut out);
    out
}

/// Variable names assigned at the TOP LEVEL of the statement list only —
/// not inside if-bodies.  Used to populate `indiv_var_names`, the ordered
/// vector that maps to PK parameter slots and the TV output array.
///
/// Branch-local helpers (e.g. `SCALE = ...` inside an `if` body) are
/// intentionally excluded: including them would corrupt the AD inner loop
/// by placing the helper in a PK slot (typically overwriting CL at slot 0).
///
/// These names are *always* individual parameters. `indiv_var_names` is this
/// set plus the subset of all-branch-assigned names that a downstream block
/// actually consumes (see `unconditionally_assigned_vars` and the call site in
/// `parse_full_model`, issue #357).
fn top_level_assigned_vars(stmts: &[Statement]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for s in stmts {
        if let Statement::Assign(name, _) = s {
            if !out.iter().any(|n| n == name) {
                out.push(name.clone());
            }
        }
    }
    out
}

/// Variable names that are *unconditionally* defined by the end of the block:
/// every top-level assignment, PLUS any name assigned on EVERY branch of an
/// `if`/`else` (recursively). Such a name has a definite value on all code
/// paths, so it *can* be a genuine individual parameter — unlike a branch-local
/// helper assigned in only some branches.
///
/// This is a SUPERSET of `top_level_assigned_vars`. The all-branch extras are
/// only promoted to actual individual parameters at the call site when a
/// downstream block consumes them (issue #357) — promoting *every* such name
/// would slot throwaway intermediates into the PK array (silently hijacking the
/// reserved F/lagtime slots, aliasing the CL slot in `pk_indices`, or
/// exhausting the 16-slot layout). See `parse_full_model`.
///
/// Motivating case: a PK parameter written only inside symmetric `if`/`else`
/// branches (the natural NONMEM-style `IF (cond) CL = ...` / `IF (!cond) CL =
/// ...` construction) must still get a PK slot, be written back by
/// `pk_param_fn`, and be visible to the `[odes]` RHS name resolver — but only
/// because `[odes]` references it.
///
/// First-occurrence order, deduplicated. An `if` with no `else`, or one where
/// some branch omits the name, does NOT contribute that name — it could be
/// undefined on the missing path, so it stays branch-local (matching
/// `top_level_assigned_vars` for that case).
fn unconditionally_assigned_vars(stmts: &[Statement]) -> Vec<String> {
    // `unconditional_names_in` already deduplicates (its `push` closure), so no
    // second pass is needed here.
    unconditional_names_in(stmts)
}

/// Recursive worker for `unconditionally_assigned_vars`. Returns the names
/// definitely assigned within `stmts`, in first-occurrence order: top-level
/// `Assign`s, plus the intersection of unconditional names across all branches
/// of any `if`/`else` (requires an `else`; the intersection preserves the
/// order in which names first appear in the leading branch).
fn unconditional_names_in(stmts: &[Statement]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let push = |name: &str, out: &mut Vec<String>| {
        if !out.iter().any(|n| n == name) {
            out.push(name.to_string());
        }
    };
    for s in stmts {
        match s {
            Statement::Assign(name, _) => push(name, &mut out),
            Statement::If {
                branches,
                else_body,
            } => {
                // Without an `else`, no name is guaranteed across all paths.
                if let Some(eb) = else_body {
                    let mut sets: Vec<Vec<String>> = branches
                        .iter()
                        .map(|(_, b)| unconditional_names_in(b))
                        .collect();
                    sets.push(unconditional_names_in(eb));
                    if let Some((first, rest)) = sets.split_first() {
                        for name in first {
                            if rest.iter().all(|s| s.iter().any(|n| n == name)) {
                                push(name, &mut out);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// Collect every identifier-like token (`[A-Za-z_][A-Za-z0-9_]*`) appearing in
/// `lines`, upper-cased, into `out`. Used to decide whether an all-branch
/// individual parameter is actually *consumed* by a downstream block (issue
/// #357): such a name is promoted to a PK-slotted individual parameter only if
/// some block outside `[individual_parameters]` references it. A purely
/// internal helper (used only to compute another param within
/// `[individual_parameters]`) stays branch-local, preserving the pre-#357
/// behaviour and avoiding spurious PK-slot allocation / reserved-slot hijack.
///
/// Deliberately crude (lexical, case-folded, no scope awareness): a false
/// positive only over-promotes a name that genuinely appears downstream — the
/// safe direction. Keywords / state names that happen to collide are harmless;
/// they would only matter if the user *also* named an individual parameter
/// identically, in which case promoting it is correct anyway. Matched
/// case-insensitively to mirror the ODE/analytical name resolvers, which alias
/// both cases.
fn collect_referenced_identifiers(lines: &[String], out: &mut std::collections::HashSet<String>) {
    for line in lines {
        let bytes = line.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'_' || bytes[i].is_ascii_alphabetic() {
                let start = i;
                i += 1;
                while i < bytes.len() && (bytes[i] == b'_' || bytes[i].is_ascii_alphanumeric()) {
                    i += 1;
                }
                out.insert(line[start..i].to_ascii_uppercase());
            } else {
                i += 1;
            }
        }
    }
}

/// Union of eta indices touched by every assignment to `var_name` anywhere in
/// the statement tree (top level OR nested inside if/else bodies). Used to
/// build the per-tv `eta_map` for the AD path; if-wrapped assignments
/// contribute their RHS eta references to the union.
fn extract_eta_indices_for_var(stmts: &[Statement], var_name: &str) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    fn walk(stmts: &[Statement], target: &str, out: &mut Vec<usize>) {
        for s in stmts {
            match s {
                Statement::Assign(name, expr) | Statement::DiffEq(name, expr) => {
                    if name == target {
                        for idx in extract_eta_indices(expr) {
                            if !out.contains(&idx) {
                                out.push(idx);
                            }
                        }
                    }
                }
                Statement::AssignIdx(_, _)
                | Statement::DiffEqIdx(_, _)
                | Statement::AssignBc(_, _)
                | Statement::DiffEqBc(_, _) => {
                    // Indexed / bytecode variants only appear after
                    // `resolve_variable_indices`; this helper runs on the
                    // pre-resolve AST.
                }
                Statement::If {
                    branches,
                    else_body,
                } => {
                    for (_, body) in branches {
                        walk(body, target, out);
                    }
                    if let Some(eb) = else_body {
                        walk(eb, target, out);
                    }
                }
            }
        }
    }
    walk(stmts, var_name, &mut out);
    out
}

/// Detect mu-referencing patterns in one assignment expression.
///
/// Returns `Some((eta_idx, anchor, log_transformed))` or `None`. The anchor
/// is either a plain `Theta(usize)` (classical mu-ref) or a `NnOutput`
/// reference (Phase A M1: the typical value is produced by a
/// `[covariate_nn]` forward pass).
///
/// Patterns recognised:
/// - Pattern 1: `THETA * exp(ETA)`              → lognormal
/// - Pattern 1-NN: `NN.OUTPUT * exp(ETA)`       → lognormal (DCM)
/// - Pattern 2: `exp(log(THETA) + ETA)`         → lognormal
/// - Pattern 3: `THETA + ETA` or `ETA + THETA`  → additive
/// - Pattern 4: `THETA * exp(ETA) * <const>`    → lognormal (multiplied
///   by a constant factor; the constant doesn't affect mu-ref detection
///   since `collect_mul_*` walks the whole product chain).
fn detect_pattern(expr: &Expression) -> Option<(usize, MuRefAnchor, bool)> {
    match expr {
        // Pattern 2: exp(log(THETA) + ETA)
        Expression::UnaryFn(name, inner) if name == "exp" => {
            if let Expression::BinOp(lhs, BinOp::Add, rhs) = inner.as_ref() {
                let try_log_theta_eta =
                    |a: &Expression, b: &Expression| -> Option<(usize, usize)> {
                        if let Expression::UnaryFn(fn_name, fn_arg) = a {
                            if fn_name == "log" || fn_name == "ln" {
                                if let Expression::Theta(ti) = fn_arg.as_ref() {
                                    if let Expression::Eta(ei) = b {
                                        return Some((*ei, *ti));
                                    }
                                }
                            }
                        }
                        None
                    };
                if let Some((ei, ti)) =
                    try_log_theta_eta(lhs, rhs).or_else(|| try_log_theta_eta(rhs, lhs))
                {
                    return Some((ei, MuRefAnchor::Theta(ti), true));
                }
            }
            None
        }
        // Pattern 3: THETA + ETA or ETA + THETA
        Expression::BinOp(lhs, BinOp::Add, rhs) => match (lhs.as_ref(), rhs.as_ref()) {
            (Expression::Theta(ti), Expression::Eta(ei)) => {
                Some((*ei, MuRefAnchor::Theta(*ti), false))
            }
            (Expression::Eta(ei), Expression::Theta(ti)) => {
                Some((*ei, MuRefAnchor::Theta(*ti), false))
            }
            _ => None,
        },
        // Pattern 1 / 1-NN / 4: product containing exactly one anchor
        // (Theta OR NnOutput) and `exp(Eta)` somewhere in the chain.
        _ => {
            let mut anchors = Vec::new();
            collect_mul_anchors(expr, &mut anchors);
            if anchors.len() == 1 {
                if let Some(ei) = find_exp_eta_in_mul(expr) {
                    return Some((ei, anchors[0], true));
                }
            }
            None
        }
    }
}

/// Analyse parsed `[individual_parameters]` statements and detect
/// mu-referencing relationships. Only top-level (unconditional) assignments
/// participate — a variable defined inside `if (...) { CL = ... }` cannot
/// participate in mu-referencing because the inner-loop re-centering only
/// holds when the relationship is unconditional.
///
/// `nn_specs` is the same `(name, output_names)` list passed to `ParseCtx`.
/// For NN-anchored mu-refs the produced `MuRef::theta_name` is the
/// structured `"<NN_NAME>.<OUTPUT_NAME>"` form. Downstream consumers that
/// only know about plain thetas (e.g. `compute_mu_k`) silently skip these
/// entries because the structured name doesn't appear in `theta_names`;
/// the full DCM-aware AD inner-loop fast path is planned for Phase A M2.
fn detect_mu_refs(
    stmts: &[Statement],
    theta_names: &[String],
    eta_names: &[String],
    nn_specs: &[(String, Vec<String>)],
) -> HashMap<String, MuRef> {
    let mut result = HashMap::new();
    for s in stmts {
        if let Statement::Assign(_, expr) = s {
            if let Some((eta_idx, anchor, log_transformed)) = detect_pattern(expr) {
                if eta_idx >= eta_names.len() {
                    continue;
                }
                let name = match anchor {
                    MuRefAnchor::Theta(ti) => {
                        if ti >= theta_names.len() {
                            continue;
                        }
                        theta_names[ti].clone()
                    }
                    MuRefAnchor::NnOutput { nn_idx, output_idx } => {
                        // Defensive: indices should be valid by construction
                        // (parse_atom built them against the same nn_specs),
                        // but skip silently rather than panic if anything's
                        // out of sync.
                        let Some((nn_name, outputs)) = nn_specs.get(nn_idx) else {
                            continue;
                        };
                        let Some(out_name) = outputs.get(output_idx) else {
                            continue;
                        };
                        format!("{nn_name}.{out_name}")
                    }
                };
                result.insert(
                    eta_names[eta_idx].clone(),
                    MuRef {
                        theta_name: name,
                        log_transformed,
                    },
                );
            }
        }
    }
    result
}

/// Intermediate result from classifying a single expression.
#[derive(Debug, Clone, PartialEq)]
struct ExprClass {
    eta_idx: usize,
    theta_idx: Option<usize>,
    param_type: crate::types::EtaParamType,
    /// Whether `theta_transform` should be updated for `theta_idx`.
    theta_transform: Option<crate::types::ThetaTransform>,
}

/// Classify a single expression into an `ExprClass`, or return `None` if no ETA
/// is present / no pattern recognised (caller handles `Custom` fallback).
fn classify_expr(expr: &Expression, n_theta: usize) -> Option<ExprClass> {
    use crate::types::{EtaParamType, ThetaTransform};

    // inv_logit(THETA + ETA) or inv_logit(logit(THETA) + ETA)
    if let Some((ei, ti, prob_scale)) = detect_logit_pattern(expr) {
        if ti < n_theta {
            let (tt, pt) = if prob_scale {
                (
                    ThetaTransform::LogitProbability,
                    EtaParamType::LogitProbability,
                )
            } else {
                (ThetaTransform::Logit, EtaParamType::Logit)
            };
            return Some(ExprClass {
                eta_idx: ei,
                theta_idx: Some(ti),
                param_type: pt,
                theta_transform: Some(tt),
            });
        }
    }

    // exp(THETA + ETA)
    if let Expression::UnaryFn(name, inner) = expr {
        if name == "exp" {
            if let Expression::BinOp(lhs, BinOp::Add, rhs) = inner.as_ref() {
                if let Some((ei, ti)) =
                    plain_theta_eta(lhs, rhs).or_else(|| plain_theta_eta(rhs, lhs))
                {
                    if ti < n_theta {
                        return Some(ExprClass {
                            eta_idx: ei,
                            theta_idx: Some(ti),
                            param_type: EtaParamType::LogNormal,
                            theta_transform: Some(ThetaTransform::Log),
                        });
                    }
                }
            }
        }
    }

    // TVCL * exp(ETA), exp(log(THETA) + ETA), TVCL + ETA, TYPICAL_PK.CL * exp(ETA).
    // NN-anchored variants are still classified as Lognormal/Additive — the
    // eta's *statistical* shape is the same; only the anchor differs.
    if let Some((ei, anchor, log_transformed)) = detect_pattern(expr) {
        let valid = match anchor {
            MuRefAnchor::Theta(ti) => ti < n_theta,
            MuRefAnchor::NnOutput { .. } => true,
        };
        if valid {
            let pt = if log_transformed {
                EtaParamType::LogNormal
            } else {
                EtaParamType::Additive
            };
            // Populate theta_idx so apply_class can set linked_theta in
            // EtaParamInfo. theta_transform stays None — the product/additive
            // pattern does not change how the theta is packed by the optimizer
            // (that is driven by theta_lower bounds, not by classification).
            let theta_idx = match anchor {
                MuRefAnchor::Theta(ti) if ti < n_theta => Some(ti),
                _ => None,
            };
            return Some(ExprClass {
                eta_idx: ei,
                theta_idx,
                param_type: pt,
                theta_transform: None,
            });
        }
    }

    // Fallback: BASE * exp(ETA[+KAPPA]) where BASE contains no bare eta
    // references. Handles derived intermediates like `KTR * exp(ETA_KA)`
    // where KTR is a variable defined earlier in [individual_parameters] and
    // collect_mul_anchors therefore finds no direct Theta anchor.
    // Only reached when detect_pattern returned None above.
    // detect_pattern already handles the Theta-anchored case, so this
    // only fires when anchors.len() != 1 but the expression is still
    // structurally log-normal.
    if let Some(ei) = find_exp_eta_in_mul(expr) {
        if !has_bare_eta(expr) {
            return Some(ExprClass {
                eta_idx: ei,
                theta_idx: None,
                param_type: EtaParamType::LogNormal,
                theta_transform: None,
            });
        }
    }

    None
}

/// Collect all expressions assigned to `param_name` across every branch of a
/// `Statement::If` (branches + else_body). Only looks one level deep (nested ifs
/// are not walked). Returns `None` if any branch body has no assignment for
/// `param_name` (meaning the parameter is only conditionally defined).
fn if_branch_exprs<'a>(stmt: &'a Statement, param_name: &str) -> Option<Vec<&'a Expression>> {
    if let Statement::If {
        branches,
        else_body,
    } = stmt
    {
        let mut exprs: Vec<&'a Expression> = Vec::new();
        for (_, body) in branches {
            let found: Vec<_> = body
                .iter()
                .filter_map(|s| {
                    if let Statement::Assign(n, e) = s {
                        if n == param_name {
                            Some(e)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect();
            if found.is_empty() {
                return None; // branch doesn't assign this param — incomplete
            }
            exprs.extend(found);
        }
        if let Some(eb) = else_body {
            let found: Vec<_> = eb
                .iter()
                .filter_map(|s| {
                    if let Statement::Assign(n, e) = s {
                        if n == param_name {
                            Some(e)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect();
            if found.is_empty() {
                return None;
            }
            exprs.extend(found);
        } else {
            // No else branch: parameter may be undefined for some subjects → incomplete.
            return None;
        }
        Some(exprs)
    } else {
        None
    }
}

/// Match `THETA + ETA` or `ETA + THETA` and return `(eta_idx, theta_idx)`.
/// Used by both `detect_logit_pattern` and the `exp(THETA+ETA)` arm of
/// `classify_indiv_params`.
fn plain_theta_eta(a: &Expression, b: &Expression) -> Option<(usize, usize)> {
    if let (Expression::Theta(ti), Expression::Eta(ei)) = (a, b) {
        return Some((*ei, *ti));
    }
    None
}

/// Detect logit-normal parameterisation patterns.
/// Returns `Some((eta_idx, theta_idx, prob_scale))` where `prob_scale` is
/// `true` for `inv_logit(logit(THETA) + ETA)` and `false` for `inv_logit(THETA + ETA)`.
///
/// Recognised forms:
///   - `inv_logit(THETA + ETA)`          — THETA on the logit scale
///   - `inv_logit(logit(THETA) + ETA)`   — THETA on the probability scale (0,1)
fn detect_logit_pattern(expr: &Expression) -> Option<(usize, usize, bool)> {
    if let Expression::UnaryFn(name, inner) = expr {
        if name == "inv_logit" || name == "expit" {
            if let Expression::BinOp(lhs, BinOp::Add, rhs) = inner.as_ref() {
                let try_logit_theta_eta = |a: &Expression,
                                           b: &Expression|
                 -> Option<(usize, usize, bool)> {
                    // Form 1: THETA + ETA  (THETA on logit scale)
                    if let Some((ei, ti)) = plain_theta_eta(a, b) {
                        return Some((ei, ti, false));
                    }
                    // Form 2: logit(THETA) + ETA  (THETA on probability scale)
                    if let (Expression::UnaryFn(fn_name, inner_arg), Expression::Eta(ei)) = (a, b) {
                        if fn_name == "logit" {
                            if let Expression::Theta(ti) = inner_arg.as_ref() {
                                return Some((*ei, *ti, true));
                            }
                        }
                    }
                    None
                };
                return try_logit_theta_eta(lhs, rhs).or_else(|| try_logit_theta_eta(rhs, lhs));
            }
        }
    }
    None
}

/// Classify each top-level [individual_parameters] assignment and return
/// `(eta_param_infos, theta_transforms)`.
///
/// `theta_transforms` is indexed parallel to `theta_names`; `eta_param_infos`
/// contains one entry per BSV ETA that could be classified.
///
/// Note: new metadata types (`EtaParamInfo`, `ThetaTransform`, `SigmaType`) are not yet
/// written to the fit YAML — `io/output.rs` will be updated alongside ferx#53.
fn classify_indiv_params(
    stmts: &[Statement],
    theta_names: &[String],
    eta_names: &[String],
) -> (
    Vec<crate::types::EtaParamInfo>,
    Vec<crate::types::ThetaTransform>,
) {
    use crate::types::{EtaParamInfo, EtaParamType, ThetaTransform};

    let n_theta = theta_names.len();
    let mut theta_transform = vec![ThetaTransform::Identity; n_theta];
    let mut eta_infos: Vec<EtaParamInfo> = Vec::new();

    for s in stmts {
        match s {
            Statement::Assign(param_name, expr) => {
                if let Some(c) = classify_expr(expr, n_theta) {
                    apply_class(
                        c,
                        param_name,
                        eta_names,
                        theta_names,
                        &mut theta_transform,
                        &mut eta_infos,
                    );
                } else {
                    // Unrecognised pattern → Custom for every ETA referenced.
                    // Note: multiple ETAs in one expression each get their own entry.
                    for ei in extract_eta_indices(expr) {
                        if ei < eta_names.len() {
                            eta_infos.push(EtaParamInfo {
                                eta_name: eta_names[ei].clone(),
                                param_type: EtaParamType::Custom,
                                linked_theta: None,
                                individual_param_name: param_name.clone(),
                            });
                        }
                    }
                }
            }
            Statement::If { .. } => {
                // For each individual parameter assigned inside this if/else block,
                // check whether every branch uses the same pattern. If so, emit that
                // classification; otherwise fall back to Custom.
                let candidate_names = collect_assigned_names_in_if(s);
                for param_name in &candidate_names {
                    if let Some(exprs) = if_branch_exprs(s, param_name) {
                        let classes: Vec<Option<ExprClass>> =
                            exprs.iter().map(|e| classify_expr(e, n_theta)).collect();
                        if classes.iter().all(|c| c.is_some()) {
                            let first = classes[0].as_ref().unwrap();
                            let unanimous = classes.iter().all(|c| {
                                let c = c.as_ref().unwrap();
                                c.param_type == first.param_type && c.eta_idx == first.eta_idx
                            });
                            if unanimous {
                                apply_class(
                                    first.clone(),
                                    param_name,
                                    eta_names,
                                    theta_names,
                                    &mut theta_transform,
                                    &mut eta_infos,
                                );
                                continue;
                            }
                        }
                        // Branches disagree or contain unrecognised patterns → Custom.
                        let all_etas: std::collections::HashSet<usize> = exprs
                            .iter()
                            .flat_map(|e| extract_eta_indices(e))
                            .filter(|&i| i < eta_names.len())
                            .collect();
                        for ei in all_etas {
                            eta_infos.push(EtaParamInfo {
                                eta_name: eta_names[ei].clone(),
                                param_type: EtaParamType::Custom,
                                linked_theta: None,
                                individual_param_name: param_name.clone(),
                            });
                        }
                    }
                    // if_branch_exprs returns None when a branch omits the param
                    // (no else arm, or incomplete coverage) — skip classification.
                }
            }
            _ => {}
        }
    }

    (eta_infos, theta_transform)
}

/// Apply a recognised `ExprClass` to the output vectors.
fn apply_class(
    c: ExprClass,
    param_name: &str,
    eta_names: &[String],
    theta_names: &[String],
    theta_transform: &mut Vec<crate::types::ThetaTransform>,
    eta_infos: &mut Vec<crate::types::EtaParamInfo>,
) {
    if c.eta_idx >= eta_names.len() {
        return;
    }
    if let (Some(ti), Some(tt)) = (c.theta_idx, c.theta_transform) {
        theta_transform[ti] = tt;
    }
    let linked = c.theta_idx.map(|ti| theta_names[ti].clone());
    eta_infos.push(crate::types::EtaParamInfo {
        eta_name: eta_names[c.eta_idx].clone(),
        param_type: c.param_type,
        linked_theta: linked,
        individual_param_name: param_name.to_owned(),
    });
}

/// Return the set of variable names assigned anywhere inside a `Statement::If`
/// (one level deep only — nested ifs are not walked).
fn collect_assigned_names_in_if(stmt: &Statement) -> std::collections::HashSet<String> {
    let mut names = std::collections::HashSet::new();
    if let Statement::If {
        branches,
        else_body,
    } = stmt
    {
        for (_, body) in branches {
            for s in body {
                if let Statement::Assign(n, _) = s {
                    names.insert(n.clone());
                }
            }
        }
        if let Some(eb) = else_body {
            for s in eb {
                if let Statement::Assign(n, _) = s {
                    names.insert(n.clone());
                }
            }
        }
    }
    names
}

/// Parse a model file (.ferx) and return a CompiledModel.
pub fn parse_model_file(path: &Path) -> Result<CompiledModel, String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("Failed to read model file: {}", e))?;
    parse_model_string(&content)
}

/// Parse a full model file including simulation spec, initial values, and fit options.
pub fn parse_full_model_file(path: &Path) -> Result<ParsedModel, String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("Failed to read model file: {}", e))?;
    parse_full_model(&content)
}

/// Parse a model string and return a CompiledModel (backward compatible).
pub fn parse_model_string(content: &str) -> Result<CompiledModel, String> {
    let parsed = parse_full_model(content)?;
    Ok(parsed.model)
}

/// Parse a full model string including all optional blocks.
pub fn parse_full_model(content: &str) -> Result<ParsedModel, String> {
    let mut extracted = extract_blocks(content)?;
    // `ode_template NAME(...)` desugaring (#322 Phase 0b): if [structural_model]
    // uses `ode_template`, rewrite it (and the [odes]/[scaling] blocks) into the
    // hand-written `ode(...)` form *before* anything else looks at the blocks, so
    // the rest of this function — including ODE detection below — sees a normal
    // ODE model with no special-casing.
    apply_ode_template(&mut extracted)?;
    // Keep the historical `blocks` binding for unnamed blocks so the rest of
    // this (large) function reads unchanged. Named blocks are pulled from
    // `extracted.named` directly where they're consumed below.
    let blocks = &extracted.unnamed;
    let name = extract_model_name(content);

    // ── Required blocks ──
    let param_lines = blocks
        .get("parameters")
        .ok_or("Missing [parameters] block")?;
    let (thetas, omegas, block_omegas, sigmas, eta_names_bsv, kappa_info) =
        parse_parameters(param_lines)?;

    // ── Optional [covariate_nn NAME] blocks (Phase A M1, behind `--features nn`)
    //
    // Parsed early so the auto-generated weight thetas land in `theta_names`
    // before [individual_parameters] is parsed — that way future PRs can
    // recognise `TYPICAL_PK.CL` as an NN-output reference during expression
    // parsing without re-walking the AST.
    #[cfg(feature = "nn")]
    let nn_specs: Vec<CovariateNnSpec> = {
        let mut specs = Vec::new();
        if let Some(map) = extracted.named.get("covariate_nn") {
            // Sort by name so theta-ordering is deterministic across runs
            // (HashMap iteration order is otherwise unstable).
            let mut entries: Vec<_> = map.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            for (instance, lines) in entries {
                specs.push(parse_covariate_nn_block(instance, lines)?);
            }
        }
        specs
    };
    #[cfg(not(feature = "nn"))]
    if extracted.named.contains_key("covariate_nn") {
        return Err(
            "[covariate_nn] blocks require building ferx-core with `--features nn`. \
             See plans/dcm-and-low-dim-node.md for the design and roadmap."
                .to_string(),
        );
    }

    // A model with [event_model] but no Gaussian PK blocks is valid (TTE-only).
    // Detect this case early so the three normally-required blocks can be omitted.
    #[cfg(feature = "survival")]
    let has_event_model_block =
        blocks.contains_key("event_model") || extracted.named.contains_key("event_model");
    #[cfg(not(feature = "survival"))]
    let has_event_model_block = false;

    let struct_lines_opt = blocks.get("structural_model");
    let error_lines_opt = blocks.get("error_model");
    let indiv_lines_opt = blocks.get("individual_parameters");
    // All three Gaussian blocks must be absent together for a valid TTE-only model.
    // Partial omission (e.g. [structural_model] present but no [individual_parameters])
    // would create an invalid mixed-model state.
    let is_tte_only = has_event_model_block
        && struct_lines_opt.is_none()
        && error_lines_opt.is_none()
        && indiv_lines_opt.is_none();
    if struct_lines_opt.is_none() && !is_tte_only {
        return Err("Missing [structural_model] block".to_string());
    }
    let struct_lines: &[String] = struct_lines_opt.map(Vec::as_slice).unwrap_or(&[]);

    if error_lines_opt.is_none() && !is_tte_only {
        return Err("Missing [error_model] block".to_string());
    }
    let (parsed_error_model, ltbs_flags) = if let Some(error_lines) = error_lines_opt {
        parse_error_model(error_lines)?
    } else {
        // TTE-only model: no Gaussian error model — empty per-CMT spec.
        (ParsedErrorModel::PerCmt(vec![]), LtbsFlags::default())
    };
    // LTBS log-transforms the structural prediction, which is incompatible with
    // the SDE/EKF measurement model (the extended Kalman filter assumes a
    // natural-scale additive/proportional observation). Reject the combination.
    if ltbs_flags.log_transform && blocks.contains_key("diffusion") {
        return Err(
            "[error_model] log-transform-both-sides is not supported with an SDE \
             ([diffusion]) model"
                .to_string(),
        );
    }

    if indiv_lines_opt.is_none() && !is_tte_only {
        return Err("Missing [individual_parameters] block".to_string());
    }
    let indiv_lines: &[String] = indiv_lines_opt.map(Vec::as_slice).unwrap_or(&[]);

    // theta_names is extended below after NN-weight and diffusion thetas are appended
    let mut theta_names: Vec<String> = thetas.iter().map(|t| t.name.clone()).collect();
    #[cfg(feature = "nn")]
    for spec in &nn_specs {
        theta_names.extend(spec.theta_names.iter().cloned());
    }
    let sigma_names: Vec<String> = sigmas.iter().map(|s| s.name.clone()).collect();
    let n_theta;
    let n_eta = eta_names_bsv.len(); // BSV-only count
    let n_kappa = kappa_info.names_ordered.len();
    let n_epsilon = sigma_names.len();

    // Extended eta context: BSV etas followed by kappa names.
    // This lets [individual_parameters] expressions like `ETA_CL + KAPPA_CL`
    // compile: KAPPA_CL becomes Eta(n_eta + kappa_idx) in the AST.
    let kappa_names: Vec<String> = kappa_info.names_ordered.clone();
    let eta_names: Vec<String> = eta_names_bsv
        .iter()
        .cloned()
        .chain(kappa_names.iter().cloned())
        .collect();

    // Parse the `[individual_parameters]` block into statements once. The block
    // may contain plain assignments AND multi-line `if (...) { ... } else { ... }`
    // constructs, so we reconstruct it as a single text buffer (newlines
    // separate statements) and run the recursive-descent statement parser.
    //
    // Two passes: the first resolves identifiers without local-var awareness,
    // just to discover every assigned name. The second re-parses with that
    // full set registered as defined_vars so any in-block reference (forward
    // or backward) resolves as Variable rather than Covariate.
    //
    // `indiv_var_names` — the names that map to PK slots and the TV output
    // vector. Two tiers (issue #357):
    //   1. Every top-level assignment is always an individual parameter.
    //   2. A name assigned on *all* branches of an if/else is promoted ONLY
    //      when a downstream block ([odes], [structural_model], [scaling],
    //      [derived]) actually references it — i.e. it is consumed as a PK
    //      parameter. This is what makes the NONMEM-style `IF (cond) CL = ...`
    //      / `IF (!cond) CL = ...` construction work: CL appears in [odes], so
    //      it earns a slot, is written back by pk_param_fn, and resolves in the
    //      ODE RHS.
    // Promoting *every* all-branch name (the naive fix) would slot throwaway
    // intermediates into the PK array — silently hijacking the reserved
    // F/lagtime slots, aliasing the CL slot in pk_indices, or exhausting the
    // 16-slot layout. So a branch-only helper used purely to compute another
    // param stays branch-local. The ParseCtx still receives the full set (via
    // assigned_vars_in_order) so such helpers parse as Variable, not Covariate.
    let indiv_text = indiv_lines.join("\n");
    // NN-output lookup table for `TYPICAL_PK.CL`-style dot-access in
    // [individual_parameters]. Always present (empty when no [covariate_nn]
    // block is in the model) so the same code path works with or without
    // --features nn.
    #[cfg(feature = "nn")]
    let nn_specs_for_ctx: Vec<(String, Vec<String>)> = nn_specs
        .iter()
        .map(|s| {
            use crate::nn::CovariateMapper;
            (s.name.clone(), s.mapper.output_names().to_vec())
        })
        .collect();
    #[cfg(not(feature = "nn"))]
    let nn_specs_for_ctx: Vec<(String, Vec<String>)> = Vec::new();

    let bare_ctx = ParseCtx::new(&theta_names, &eta_names, &[]).with_nn_specs(&nn_specs_for_ctx);
    let pre_stmts = parse_block_statements(&indiv_text, bare_ctx, StatementMode::Plain)?;
    let all_assigned = assigned_vars_in_order(&pre_stmts);
    // Tier 1 (always) + tier 2 (all-branch names referenced downstream). See
    // the comment above for the rationale behind the downstream-consumption gate.
    let top_level_names = top_level_assigned_vars(&pre_stmts);
    let mut downstream_refs: std::collections::HashSet<String> = std::collections::HashSet::new();
    for key in ["odes", "structural_model", "scaling", "derived"] {
        if let Some(lines) = blocks.get(key) {
            collect_referenced_identifiers(lines, &mut downstream_refs);
        }
    }
    let indiv_var_names: Vec<String> = unconditionally_assigned_vars(&pre_stmts)
        .into_iter()
        .filter(|n| {
            top_level_names.iter().any(|t| t == n)
                || downstream_refs.contains(&n.to_ascii_uppercase())
        })
        .collect();
    let indiv_ctx =
        ParseCtx::new(&theta_names, &eta_names, &all_assigned).with_nn_specs(&nn_specs_for_ctx);
    let indiv_stmts = parse_block_statements(&indiv_text, indiv_ctx, StatementMode::Plain)?;

    // Detect ODE vs analytical model
    let is_ode = struct_lines
        .iter()
        .any(|l| l.starts_with("ode(") || l.starts_with("ode "));

    // The error rule (#322 Phase 0b): an ODE-only absorption input-rate function
    // (`transit`/`igd`/`weibull`) has no closed form, so it cannot ride on an
    // analytical `pk ...` disposition. Reject the combination loudly, pointing at
    // `ode_template`, rather than silently ignoring the [odes] block. An
    // `ode_template`/`ode(...)` disposition sets `is_ode`, so this only fires for
    // an analytical `pk` model that also carries an ODE-only absorption term.
    if !is_ode {
        if let Some(fname) = ode_only_absorption_fn_in_odes(blocks.get("odes")) {
            return Err(format!(
                "[structural_model]: `{fname}(...)` absorption requires an ODE disposition, but \
                 the model uses an analytical `pk ...`. {fname}(...) has no closed form, so ferx \
                 will not silently turn the analytical model into an ODE. Replace `pk NAME(...)` \
                 with `ode_template NAME(...)` (ferx writes the disposition ODE) and keep \
                 `{fname}(...)` in [odes]."
            ));
        }
    }

    // For ODE models, map each individual parameter to a slot in the fixed
    // PkParams array (canonical names → their PK slot, others → free slots,
    // reserving PK_IDX_F / PK_IDX_LAGTIME). The RHS evaluator, the parameter
    // writer (`build_pk_param_fn`), and `pk_indices` all share this map.
    // Empty for analytical models, which route through `pk_param_map` instead.
    let ode_slot_map: Vec<usize> = if is_ode {
        ode_param_slots(&indiv_var_names)?
    } else {
        Vec::new()
    };

    let (
        pk_model,
        pk_param_map,
        ode_spec,
        diffusion_theta_names,
        diffusion_theta_inits,
        diffusion_theta_fixed,
        diffusion_state_indices,
    ) = if is_ode {
        let (state_names, obs_cmt_name) = parse_ode_structural(struct_lines)?;
        let ode_lines = blocks
            .get("odes")
            .ok_or("ODE model requires [odes] block")?;
        let mut ode_spec = build_ode_spec(
            ode_lines,
            &state_names,
            obs_cmt_name.as_deref(),
            &indiv_var_names,
            &ode_slot_map,
        )?;

        // Parse optional [diffusion] block
        let (diff_var, diff_names, diff_fixed, diff_state_idx) =
            if let Some(diff_lines) = blocks.get("diffusion") {
                let (variances, names, fixed) = parse_diffusion_block(diff_lines, &state_names)?;
                // Collect indices of states that actually have diffusion
                let state_idx: Vec<usize> = names
                    .iter()
                    .enumerate()
                    .filter_map(|(i, n)| n.as_ref().map(|_| i))
                    .collect();
                (variances, names, fixed, state_idx)
            } else {
                (
                    vec![0.0; state_names.len()],
                    vec![None; state_names.len()],
                    vec![false; state_names.len()],
                    Vec::new(),
                )
            };

        // Store initial diffusion variances in the ODE spec (non-zero states only)
        if !diff_state_idx.is_empty() {
            ode_spec.diffusion_var = diff_var.clone();
        }

        // Collect diffusion parameters that will become thetas (non-zero, non-fixed)
        let diff_theta_names: Vec<String> = diff_names.iter().filter_map(|n| n.clone()).collect();
        let diff_theta_inits: Vec<f64> = diff_state_idx.iter().map(|&i| diff_var[i]).collect();
        let diff_theta_fixed_vec: Vec<bool> =
            diff_state_idx.iter().map(|&i| diff_fixed[i]).collect();
        // PK model not used for ODE, but we need a placeholder + empty param map
        (
            PkModel::OneCptOral,
            HashMap::new(),
            Some(ode_spec),
            diff_theta_names,
            diff_theta_inits,
            diff_theta_fixed_vec,
            diff_state_idx,
        )
    } else {
        // [diffusion] outside an ODE model is an error
        if blocks.contains_key("diffusion") {
            return Err(
                "[diffusion] block requires an ODE model (use `ode(...)` in [structural_model] \
                 and define [odes]). Analytical PK models do not support SDE diffusion."
                    .to_string(),
            );
        }
        // TTE-only models have no [structural_model] block — supply a no-op placeholder.
        // The PK model is never invoked for pure-TTE subjects (no Gaussian observations).
        let (pk_model, pk_param_map) = if struct_lines.is_empty() {
            (PkModel::OneCptIv, HashMap::new())
        } else {
            parse_structural_model(struct_lines)?
        };
        (
            pk_model,
            pk_param_map,
            None,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
    };

    // Build the CovariateNn handles up front (before build_pk_param_fn, which
    // captures them in the pk_param_fn closure). The same vector is also
    // appended to CompiledModel.covariate_nns further down — see the diffusion
    // appendage block, which uses these to drive theta-value extension.
    #[cfg(feature = "nn")]
    let covariate_nns_for_closure: Vec<crate::nn::CovariateNn> = {
        let mut acc = Vec::with_capacity(nn_specs.len());
        let mut offset = thetas.len();
        for spec in &nn_specs {
            acc.push(crate::nn::CovariateNn {
                name: spec.name.clone(),
                mapper: spec.mapper.clone(),
                weights_offset: offset,
            });
            offset += spec.theta_inits.len();
        }
        acc
    };

    // Build pk_param_fn with the extended eta context (BSV + kappa names).
    // `n_theta_base` is the user-declared θ count — indiv params can only
    // reference these (NN-weight and diffusion θ are appended later and
    // aren't visible to user expressions). `n_eta_extended` matches the
    // `eta` slice the closure consumes (BSV η + kappa). Both feed the
    // Tier 4a milestone-2 partial-derivative builder.
    let n_eta_extended_for_partials = eta_names.len();
    let (pk_param_fn, referenced_covariates, indiv_param_partials) = build_pk_param_fn(
        indiv_stmts.clone(),
        &pk_param_map,
        &indiv_var_names,
        &ode_slot_map,
        thetas.len(),
        n_eta_extended_for_partials,
        #[cfg(feature = "nn")]
        &covariate_nns_for_closure,
    )?;

    // Reject an analytical model that omits a required PK parameter for its
    // structure (issue #309). Runs *after* build_pk_param_fn so the per-key
    // validation (unknown key / undefined reference, #308) reports first; every
    // surviving key is then a known PK name, so the canonical-slot set is exact
    // and the `v`/`v1`, `q`/`q2` aliases satisfy their slot. An unmapped required
    // slot would otherwise stay at `PkParams::default()` (0.0) and the fit would
    // silently "converge" to a structurally broken optimum.
    if !pk_param_map.is_empty() {
        let mapped_slots: std::collections::HashSet<usize> = pk_param_map
            .keys()
            .filter_map(|k| PkParams::name_to_index(k))
            .collect();
        let missing: Vec<&str> = pk_model
            .required_pk_params()
            .iter()
            .filter(|(slot, _)| !mapped_slots.contains(slot))
            .map(|(_, name)| *name)
            .collect();
        if !missing.is_empty() {
            let list = missing
                .iter()
                .map(|n| format!("`{n}`"))
                .collect::<Vec<_>>()
                .join(", ");
            let (verb, pron) = if missing.len() == 1 {
                ("is", "it")
            } else {
                ("are", "them")
            };
            let first = missing[0];
            let first_upper = first.to_uppercase();
            return Err(format!(
                "[structural_model] {} requires {list}, which {verb} not mapped. \
                 Add {pron}, e.g. `{first}={first_upper}`.",
                pk_model.canonical_name()
            ));
        }
    }

    // Append NN-weight thetas (Phase A M1), then diffusion variances. Both
    // sit at the tail of the theta vector so existing user-declared theta
    // indices are unaffected.
    //
    // Layout:
    //   [base thetas | NN-weight thetas | diffusion thetas]
    //
    // NN weights are identity-packed (lower = -∞, upper = +∞): they can be
    // any real number, the optimizer sees them unscaled. Initial values are
    // Glorot-style deterministic samples produced by `parse_covariate_nn_block`.
    let mut theta_values: Vec<f64> = thetas.iter().map(|t| t.init).collect();
    let mut theta_lower: Vec<f64> = thetas.iter().map(|t| t.lower).collect();
    let mut theta_upper: Vec<f64> = thetas.iter().map(|t| t.upper).collect();
    let mut theta_fixed: Vec<bool> = thetas.iter().map(|t| t.fixed).collect();

    // NN-weight thetas: same offsets as `covariate_nns_for_closure` built
    // above (both use `thetas.len()` as the first offset). We append values
    // / bounds / fixed flags here; the handles themselves live in
    // `covariate_nns_for_closure` and are reused below for `CompiledModel`.
    #[cfg(feature = "nn")]
    let covariate_nns: Vec<crate::nn::CovariateNn> = covariate_nns_for_closure.clone();
    #[cfg(feature = "nn")]
    for spec in &nn_specs {
        for &init in &spec.theta_inits {
            theta_values.push(init);
            theta_lower.push(f64::NEG_INFINITY);
            theta_upper.push(f64::INFINITY);
            theta_fixed.push(false);
        }
    }

    let diff_theta_start = theta_values.len(); // index of first diffusion theta
    for (i, &init) in diffusion_theta_inits.iter().enumerate() {
        let is_fixed = diffusion_theta_fixed[i];
        // Only clamp estimated params — a fixed 0.0 diffusion should stay 0.0
        let clamped = if is_fixed { init } else { init.max(1e-10) };
        theta_values.push(clamped);
        theta_names.push(diffusion_theta_names[i].clone());
        theta_lower.push(0.0);
        theta_upper.push(f64::INFINITY);
        theta_fixed.push(is_fixed);
    }
    // set here after diffusion thetas are appended above
    n_theta = theta_names.len();
    // A `[event_model]` (TTE) endpoint makes a fixed-effects model legitimate:
    // the hazard parameters can be pure theta/covariate, so n_eta = 0 (an empty
    // BSV Omega) is valid — even though `build_omega_matrix` rejects an empty
    // Omega for ordinary PK models. Detect the block from the raw parse so the
    // same `.ferx` parses identically with or without the `survival` feature
    // compiled in (the actual TTE endpoints are only built under `survival`).
    let has_event_model = blocks.get("event_model").is_some()
        || extracted
            .named
            .get("event_model")
            .is_some_and(|m| !m.is_empty());
    // BSV omega is built from the BSV-only eta names (no kappas)
    let omega = if eta_names_bsv.is_empty() && has_event_model {
        // 0×0 Omega — `from_matrix` handles the empty matrix (cholesky of a
        // 0-dim matrix is trivial, log|Ω| = 0); `build_omega_fixed` below
        // already returns an empty `Vec` for an empty eta list.
        OmegaMatrix::from_diagonal(&[], Vec::new())
    } else {
        build_omega_matrix(&omegas, &block_omegas, &eta_names_bsv)?
    };
    let omega_fixed = build_omega_fixed(&omegas, &block_omegas, &eta_names_bsv)?;
    // Per-eta SD-init flags, parallel to `eta_names_bsv`. Diagonal omega
    // declarations carry their `(sd)` flag from the parser; block-omega etas
    // are always `false` because block_omega is variance-only.
    let omega_init_as_sd: Vec<bool> = {
        let diag_lookup: std::collections::HashMap<&str, bool> = omegas
            .iter()
            .map(|o| (o.name.as_str(), o.init_as_sd))
            .collect();
        eta_names_bsv
            .iter()
            .map(|n| *diag_lookup.get(n.as_str()).unwrap_or(&false))
            .collect()
    };
    let sigma_values: Vec<f64> = sigmas.iter().map(|s| s.value).collect();
    let sigma_fixed: Vec<bool> = sigmas.iter().map(|s| s.fixed).collect();
    let sigma_init_as_sd: Vec<bool> = sigmas.iter().map(|s| s.init_as_sd).collect();
    // Resolve the error model now that the sigma vector ordering is known.
    // For per-CMT (multi-endpoint) models this maps each endpoint's sigma
    // names to indices into this flat vector and enforces the ODE-only
    // restriction.
    // Capture referenced sigma names before `parsed_error_model` is consumed.
    let used_sigmas_in_error = used_sigma_names(&parsed_error_model);
    let (error_model, error_spec) = build_error_spec(parsed_error_model, &sigma_names, is_ode)?;
    let sigma = SigmaVector {
        values: sigma_values,
        names: sigma_names,
    };

    // Per-kappa SD-init flags, parallel to `kappa_info.names_ordered`. Same
    // logic as omega: diagonal kappa declarations carry the `(sd)` flag;
    // block_kappa entries are variance-only and contribute `false`.
    let kappa_init_as_sd: Vec<bool> = {
        let diag_lookup: std::collections::HashMap<&str, bool> = kappa_info
            .diagonal
            .iter()
            .map(|k| (k.name.as_str(), k.init_as_sd))
            .collect();
        kappa_info
            .names_ordered
            .iter()
            .map(|n| *diag_lookup.get(n.as_str()).unwrap_or(&false))
            .collect()
    };

    // IOV omega: built from kappa (diagonal) and/or block_kappa specs.
    // When only diagonal kappas are present (Option A) the matrix is diagonal.
    // When block_kappa entries are present (Option B) the matrix is non-diagonal;
    // parameterization.rs uses the `diagonal` flag to choose Cholesky packing.
    let (omega_iov, kappa_fixed) = if kappa_info.diagonal.is_empty() && kappa_info.block.is_empty()
    {
        (None, Vec::new())
    } else {
        // Reuse build_omega_matrix by converting kappa specs to the omega spec types.
        let diag_as_omega: Vec<OmegaSpec> = kappa_info
            .diagonal
            .iter()
            .map(|k| OmegaSpec {
                name: k.name.clone(),
                variance: k.variance,
                fixed: k.fixed,
                init_as_sd: k.init_as_sd,
            })
            .collect();
        let block_as_omega: Vec<BlockOmegaSpec> = kappa_info
            .block
            .iter()
            .map(|bk| BlockOmegaSpec {
                names: bk.names.clone(),
                lower_triangle: bk.lower_triangle.clone(),
                fixed: bk.fixed,
            })
            .collect();
        let omega_iov = build_omega_matrix(&diag_as_omega, &block_as_omega, &kappa_names)?;
        let kappa_fixed = build_omega_fixed(&diag_as_omega, &block_as_omega, &kappa_names)?;
        (Some(omega_iov), kappa_fixed)
    };

    let default_params = ModelParameters {
        theta: theta_values,
        theta_names: theta_names.clone(),
        theta_lower,
        theta_upper,
        theta_fixed,
        omega,
        omega_fixed,
        sigma,
        sigma_fixed,
        omega_iov,
        kappa_fixed,
    };

    // Auto-generate tv_fn: evaluate individual parameters with eta=0
    // This gives covariate-adjusted typical values for the AD inner loop.
    // tv_fn uses the extended eta context (BSV + kappa) so KAPPA_* vars evaluate
    // to 0 at population-typical predictions, which is correct.
    let tv_eta_names = eta_names.clone(); // extended (BSV + kappa)
    let tv_fn: Option<Box<dyn Fn(&[f64], &HashMap<String, f64>) -> Vec<f64> + Send + Sync>> =
        if !is_ode {
            let stmts_for_tv = indiv_stmts.clone();
            let var_names_for_tv = indiv_var_names.clone();
            #[cfg(feature = "nn")]
            let tv_covariate_nns: Vec<crate::nn::CovariateNn> = covariate_nns_for_closure.clone();
            Some(Box::new(
                move |theta: &[f64], covariates: &HashMap<String, f64>| {
                    let zero_eta = vec![0.0; tv_eta_names.len()];
                    let mut vars: HashMap<String, f64> = HashMap::new();
                    // Pre-compute each NN's forward output once per call so
                    // `TYPICAL_PK.CL`-style references inside the eta=0
                    // expression evaluate consistently and share the work.
                    #[cfg(feature = "nn")]
                    let nn_outputs: Vec<Vec<f64>> = tv_covariate_nns
                        .iter()
                        .map(|nn| {
                            use crate::nn::CovariateMapper;
                            let n_w = nn.mapper.n_weights();
                            let weights = &theta[nn.weights_offset..nn.weights_offset + n_w];
                            nn.mapper.forward_raw(weights, covariates).expect(
                                "NN forward_raw failed in tv_fn: this indicates a \
                                 weight-offset/length wiring bug (missing covariates \
                                 are substituted with 0.0, not errored on)",
                            )
                        })
                        .collect();
                    #[cfg(not(feature = "nn"))]
                    let nn_outputs: Vec<Vec<f64>> = Vec::new();
                    eval_statements(
                        &stmts_for_tv,
                        theta,
                        &zero_eta,
                        covariates,
                        &mut vars,
                        None,
                        None,
                        &nn_outputs,
                    );
                    var_names_for_tv
                        .iter()
                        .map(|n| vars.get(n).copied().unwrap_or(0.0))
                        .collect()
                },
            ))
        } else {
            None
        };

    // Detect mu-referencing relationships from [individual_parameters].
    // Run detection over all eta names (BSV + kappa) so we can derive the
    // lognormal/additive flag for IOV kappas alongside the BSV etas.
    let all_eta_names: Vec<String> = eta_names_bsv
        .iter()
        .chain(kappa_names.iter())
        .cloned()
        .collect();
    let all_mu_refs = detect_mu_refs(
        &indiv_stmts,
        &theta_names,
        &all_eta_names,
        &nn_specs_for_ctx,
    );
    let kappa_set: std::collections::HashSet<&String> = kappa_names.iter().collect();
    let mu_refs: HashMap<String, MuRef> = all_mu_refs
        .iter()
        .filter(|(k, _)| !kappa_set.contains(k))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let kappa_mu_refs: HashMap<String, MuRef> = all_mu_refs
        .into_iter()
        .filter(|(k, _)| kappa_set.contains(k))
        .collect();

    // Build pk_indices: maps each individual parameter (by declaration order)
    // to its PK parameter index. Needed for AD to place values in correct slots.
    let pk_indices: Vec<usize> = if !pk_param_map.is_empty() {
        // Reverse the pk_param_map: variable_name → pk_param_name
        let var_to_pk: HashMap<String, String> = pk_param_map
            .iter()
            .map(|(pk_name, var_name)| (var_name.to_uppercase(), pk_name.clone()))
            .collect();
        indiv_var_names
            .iter()
            .map(|var_name| {
                var_to_pk
                    .get(&var_name.to_uppercase())
                    .and_then(|pk_name| PkParams::name_to_index(pk_name))
                    .unwrap_or(0)
            })
            .collect()
    } else {
        // ODE model: canonical slot map (parallel to indiv_var_names), so a
        // declared F lands at PK_IDX_F, lagtime at PK_IDX_LAGTIME, etc. — the
        // same map the RHS evaluator and pk_param_fn use.
        ode_slot_map.clone()
    };

    // Per-tv eta index: for each individual parameter, find which BSV eta its
    // assignment(s) reference (or -1 for none). Kappa indices (>= n_eta) map
    // to -1 here since the AD path is disabled when IOV is active. For
    // if-wrapped assignments we union the eta references across every branch
    // that targets the same var. Used only by the AD inner loop, which checks
    // n_kappa > 0 and falls back to FD automatically.
    //
    // If different branches reference different BSV etas for the same var
    // (e.g. `if (...) { CL = TVCL * exp(ETA_CL) } else { CL = TVCL * exp(ETA_X) }`),
    // we pick the first one in iteration order — the AD path's notion of "the
    // eta this tv depends on" is single-valued. This is harmless when the AD
    // path is unused (IOV) and uncommon enough in practice that we don't
    // diagnose it here; FD remains correct in either case.
    let eta_map: Vec<i32> = indiv_var_names
        .iter()
        .map(|var_name| {
            extract_eta_indices_for_var(&indiv_stmts, var_name)
                .into_iter()
                .find(|&i| i < n_eta)
                .map(|i| i as i32)
                .unwrap_or(-1)
        })
        .collect();

    let pk_idx_f64: Vec<f64> = pk_indices.iter().map(|&i| i as f64).collect();
    // sel_flat is n_tv × n_eta (BSV etas only). Kappa columns are intentionally
    // absent — the AD path reads BSV gradients only; when n_kappa > 0 the inner
    // loop forces FD for the full extended-eta gradient.
    let n_tv = eta_map.len();
    let mut sel_flat = vec![0.0f64; n_tv * n_eta];
    for (i, &em) in eta_map.iter().enumerate() {
        if em >= 0 && (em as usize) < n_eta {
            sel_flat[i * n_eta + em as usize] = 1.0;
        }
    }

    // Classify [individual_parameters] expressions for the R metadata layer.
    // Uses BSV-only eta names (no kappas).
    let (eta_param_info, theta_transform) =
        classify_indiv_params(&indiv_stmts, &theta_names, &eta_names_bsv);
    debug_assert_eq!(
        theta_transform.len(),
        theta_names.len(),
        "classify_indiv_params must return one ThetaTransform per theta"
    );

    let model = CompiledModel {
        name,
        pk_model,
        error_model,
        error_spec,
        pk_param_fn,
        n_theta,
        n_eta,
        n_kappa,
        n_epsilon,
        theta_names,
        eta_names: eta_names_bsv.clone(),
        kappa_names: kappa_names.clone(),
        indiv_param_names: indiv_var_names.clone(),
        indiv_param_partials,
        default_params,
        omega_init_as_sd,
        sigma_init_as_sd,
        kappa_init_as_sd,
        tv_fn,
        #[cfg(feature = "nn")]
        covariate_nns,
        pk_indices,
        eta_map,
        pk_idx_f64,
        sel_flat,
        ode_spec,
        diffusion_theta_start: if diffusion_state_indices.is_empty() {
            None
        } else {
            Some(diff_theta_start)
        },
        diffusion_state_indices,
        bloq_method: BloqMethod::Drop,
        mu_refs,
        kappa_mu_refs,
        referenced_covariates,
        gradient_method: GradientMethod::default(),
        parse_warnings: Vec::new(), // populated below
        has_conditional_eta_params: false,
        eta_param_info,
        theta_transform,
        scaling: ScalingSpec::None,
        log_transform: ltbs_flags.log_transform,
        dv_pre_logged: ltbs_flags.dv_pre_logged,
        derived_exprs: vec![],
        output_columns: vec![],
        #[cfg(feature = "survival")]
        endpoints: std::collections::HashMap::new(),
    };

    // ── Optional blocks ──
    let simulation = blocks
        .get("simulation")
        .map(|lines| parse_simulation_block(lines))
        .transpose()?;
    let mut fit_options = if let Some(lines) = blocks.get("fit_options") {
        parse_fit_options(lines)?
    } else {
        FitOptions::default()
    };

    // ── [data_selection] block ────────────────────────────────────────────────
    // Parsed after [fit_options] and merged into the same FitOptions so the
    // read-time filtering code has a single place to look.
    if let Some(lines) = blocks.get("data_selection") {
        for line in lines {
            let parts: Vec<&str> = line.splitn(2, '=').map(|s| s.trim()).collect();
            if parts.len() != 2 {
                continue;
            }
            let key = parts[0];
            let value = parts[1];
            if key != "ignore" && key != "accept" && key != "ignore_subjects" {
                return Err(format!(
                    "[data_selection]: unknown key `{key}` — valid keys are \
                     ignore, accept, ignore_subjects"
                ));
            }
            match apply_fit_option(&mut fit_options, key, value) {
                Ok(_) => {}
                Err(e) => return Err(e),
            }
        }
    }

    // Mirror fit-level BLOQ method onto the compiled model so the likelihood
    // functions can branch without threading bloq_method through every call.
    let mut model = model;
    model.bloq_method = fit_options.bloq_method;

    // Bake the configured ODE solver tolerances from [fit_options] onto the
    // OdeSpec so predict()/fit_from_files (which integrate the parsed spec
    // as-is) use the requested accuracy. Callers that merge call-time `settings`
    // into their own FitOptions (the R wrapper's ferx_fit) must re-apply
    // sync_ode_solver_opts on the owned model for those overrides to win;
    // ferx-core fit() takes &CompiledModel and does not. No-op for analytical.
    model.sync_ode_solver_opts(&fit_options);

    // ── [scaling] block ──
    // Parsed after `parse_fit_options` so we can validate the
    // `ExpressionScale + gradient = ad` combination (which is rejected for
    // Phase 1 — AD's `obs_scale` Const input is only wired to ScalarScale).
    if let Some(scaling_lines) = blocks.get("scaling") {
        let theta_names_for_scaling = model.theta_names.clone();
        let eta_names_for_scaling = model.eta_names.clone();
        let indiv_var_names_for_scaling = model.indiv_param_names.clone();
        let pk_indices_for_scaling = model.pk_indices.clone();
        let state_names_for_scaling = model
            .ode_spec
            .as_ref()
            .map(|s| s.state_names.clone())
            .unwrap_or_default();
        let is_ode_model = model.ode_spec.is_some();

        let (scaling, output_fn) = parse_scaling_block(
            scaling_lines,
            &theta_names_for_scaling,
            &eta_names_for_scaling,
            &indiv_var_names_for_scaling,
            &pk_indices_for_scaling,
            &state_names_for_scaling,
            is_ode_model,
            &model.kappa_names,
        )?;

        // AD compatibility check (Phase 2.5):
        //
        // ScalingSpec — None / ScalarScale / ExpressionScale / PerCmt —
        // all support AD now via the per-observation `obs_scale: &[f64]`
        // slice threaded into the four AD entry points and built by
        // `inner_optimizer::build_scale_array_for_ad`. The slice is
        // materialised once per gradient call from a subject-static pk
        // evaluation, so AD treats the scale as constant w.r.t. eta.
        // That's exact for an eta-independent scale (`WT/70`, `TVV/1000` -
        // covariates/thetas only). An eta-dependent scale (e.g.
        // `obs_scale = V` with `V = TVV*exp(ETA_V)`) is now auto-routed to
        // FD by `inner_optimizer::analytical_ad_unsupported`
        // (`ScalingSpec::breaks_ad_inner_gradient`), so the user gets a
        // correct gradient without having to set `gradient = fd` by hand.
        //
        // Form C readouts (`OdeReadout::Single` / `PerCmt`) STILL force
        // FD: they only exist on ODE models, and the AD path requires
        // `tv_fn.is_some()` which is only set for analytical models. The
        // runtime check would silently demote `gradient = ad` to FD; the
        // parse-time guard here surfaces it as a loud error so the user
        // knows AD isn't actually doing anything for their Form C model.
        let ad_explicit = fit_options.gradient_method == GradientMethod::Ad;
        let ad_auto_likely = fit_options.gradient_method == GradientMethod::Auto
            && model.tv_fn.is_some()
            && cfg!(feature = "autodiff");
        let readout_needs_fd = output_fn.as_ref().map(|r| r.requires_fd()).unwrap_or(false);
        if readout_needs_fd && (ad_explicit || ad_auto_likely) {
            let kind = match output_fn.as_ref() {
                Some(crate::ode::OdeReadout::PerCmt(_)) => "per-CMT `y[CMT=N]` (Form C)",
                Some(crate::ode::OdeReadout::Single(_)) => "`y = <expr>` (Form C)",
                _ => unreachable!("readout_needs_fd implies output_fn is Some(Single | PerCmt)"),
            };
            return Err(format!(
                "[scaling]: {} is not supported with AD gradients (Form C readouts only \
                 exist on ODE models, and AD requires the analytical PK path). \
                 Add `gradient = fd` to [fit_options].",
                kind
            ));
        }

        // Form C wiring: replace the ODE readout (which was set to the
        // `NEEDS_FORM_C = usize::MAX` sentinel by `build_ode_spec` if the
        // user omitted `obs_cmt=`) with the parsed Single/PerCmt readout.
        if let Some(new_readout) = output_fn {
            let ode_spec = model.ode_spec.as_mut().expect("guarded by is_ode_model");
            ode_spec.readout = new_readout;
        }

        model.scaling = scaling;
    }

    // ODE validation: the `NEEDS_FORM_C = usize::MAX` sentinel from
    // build_ode_spec must be replaced by parsed Form C readout. If it
    // survives, the user omitted `obs_cmt=` in `ode(states=[...])`
    // without providing `[scaling] y = ...`.
    const NEEDS_FORM_C: usize = usize::MAX;
    if let Some(ref ode_spec) = model.ode_spec {
        if matches!(
            ode_spec.readout,
            crate::ode::OdeReadout::ObsCmt(NEEDS_FORM_C)
        ) {
            return Err(
                "ODE model omitted `obs_cmt=...` in [structural_model] but no \
                 [scaling] y = <expr> was provided. Either add `obs_cmt=NAME` or supply Form C."
                    .into(),
            );
        }
        // SDE + scaling is rejected entirely in Phase 1:
        //   - Form C needs the obs_cmt index for the Kalman update.
        //   - Forms A/B post-multiply only the mean prediction; the EKF
        //     `p_obs` variance and the `r_obs` callback both run in the
        //     unscaled observation space. Correct EKF scaling needs to
        //     thread the factor into both (p_obs scales by 1/K^2; the
        //     residual_variance closure must see the scaled prediction).
        //     That's a wider change than Phase 1 covers — flag and defer.
        let sde_active = model.diffusion_theta_start.is_some();
        if sde_active {
            if !matches!(ode_spec.readout, crate::ode::OdeReadout::ObsCmt(_)) {
                return Err(
                    "[scaling] y = <expr> (Form C) is not supported on SDE models — the \
                     [diffusion] / EKF path requires a single observable compartment index."
                        .into(),
                );
            }
            if !matches!(model.scaling, ScalingSpec::None) {
                return Err(
                    "[scaling] is not yet supported on SDE / [diffusion] models — the EKF \
                     variance and `r_obs` paths run in the unscaled observation space and \
                     would need separate threading of the scale factor (Phase 1.5)."
                        .into(),
                );
            }
        }
    }

    // Warn when eta-referencing individual parameters are assigned inside
    // if-blocks and therefore excluded from mu-referencing. Users should
    // assign the typical-value (TV*) unconditionally and only apply the
    // conditional inside the individual parameter expression.
    {
        // Does `var` receive an eta-bearing assignment anywhere inside an
        // `if`/`else` body, at ANY nesting depth? A top-level (unconditional)
        // assignment does not count — only ones reached through a branch. This
        // recurses into nested ifs: a parameter assigned only inside a *nested*
        // branch must still disable mu-referencing and route the inner loop to
        // FD. The earlier single-level scan missed those, leaving such models on
        // the analytical AD kernel they can't represent faithfully (#278/#280).
        fn body_assigns_eta(body: &[Statement], var: &str, n_eta: usize) -> bool {
            for bs in body {
                match bs {
                    Statement::Assign(name, expr) => {
                        if name == var && extract_eta_indices(expr).iter().any(|&i| i < n_eta) {
                            return true;
                        }
                    }
                    Statement::If {
                        branches,
                        else_body,
                    } => {
                        for (_, b) in branches {
                            if body_assigns_eta(b, var, n_eta) {
                                return true;
                            }
                        }
                        if let Some(eb) = else_body {
                            if body_assigns_eta(eb, var, n_eta) {
                                return true;
                            }
                        }
                    }
                    _ => {}
                }
            }
            false
        }
        fn any_if_branch_assigns_eta(stmts: &[Statement], var: &str, n_eta: usize) -> bool {
            for s in stmts {
                if let Statement::If {
                    branches,
                    else_body,
                } = s
                {
                    for (_, body) in branches {
                        if body_assigns_eta(body, var, n_eta) {
                            return true;
                        }
                    }
                    if let Some(eb) = else_body {
                        if body_assigns_eta(eb, var, n_eta) {
                            return true;
                        }
                    }
                }
            }
            false
        }
        // `all_assigned` is the deduped union of top-level and if-only
        // assignments, so one pass flags both "assigned only inside an if" and
        // "unconditional default + conditional eta override" with no
        // double-counting. Order is the source-declaration order of
        // `all_assigned`.
        let mut mu_ref_disabled: Vec<String> = Vec::new();
        for var in &all_assigned {
            if any_if_branch_assigns_eta(&indiv_stmts, var, n_eta) {
                mu_ref_disabled.push(var.clone());
            }
        }
        if !mu_ref_disabled.is_empty() {
            // The analytical AD inner-gradient kernels can't represent an
            // if-branch that assigns an eta-bearing parameter, so flag the model
            // for `inner_optimizer::analytical_ad_unsupported` to route it to FD.
            model.has_conditional_eta_params = true;
            model.parse_warnings.push(format!(
                "Mu-referencing disabled for conditional parameter(s): {}. \
                 Assign TV* unconditionally and apply the if-block to the individual \
                 parameter expression to re-enable mu-referencing.",
                mu_ref_disabled.join(", ")
            ));
        }
    }

    // ── [derived] block ──
    if let Some(derived_lines) = blocks.get("derived") {
        let cov_names = model.referenced_covariates.clone();
        let mut derived_warnings = Vec::new();
        // For ODE models, ode_state_names drives uses_compartments detection.
        // For analytical models, use analytical_compartment_names() so that
        // named references like `central`, `depot`, `peripheral` in [derived]
        // also set uses_compartments=true and trigger W_DERIVED_CMT_* warnings.
        let ode_state_names: Vec<String> = model
            .ode_spec
            .as_ref()
            .map(|s| s.state_names.clone())
            .unwrap_or_else(|| model.analytical_compartment_names().to_vec());
        let derived_exprs = parse_derived_block(
            derived_lines,
            &model.theta_names.clone(),
            &model.eta_names.clone(),
            &model.indiv_param_names.clone(),
            &cov_names,
            &ode_state_names,
            &mut derived_warnings,
        )?;
        model.derived_exprs = derived_exprs;
        model.parse_warnings.extend(derived_warnings);
    }

    // ── [output] block ──
    if let Some(output_lines) = blocks.get("output") {
        model.output_columns = parse_output_block(output_lines);
    }

    // ── Optional [covariates] block ──
    // When present it is authoritative for the covariate *table* and typing:
    // only listed columns are tabled, and declared columns are read strictly.
    // A covariate used in [individual_parameters] or [event_model] but not declared
    // is still usable (read leniently) — we warn after all covariate sources have
    // been collected (including [event_model], parsed below).
    let covariate_decls = if let Some(lines) = blocks.get("covariates") {
        Some(parse_covariates_block(lines)?)
    } else {
        None
    };

    // ── [event_model] / [event_model NAME] blocks ──────────────────────────────
    // Unnamed: `[event_model]` — one TTE endpoint.
    // Named:   `[event_model LABEL]` — multiple TTE endpoints keyed by CMT.
    // Each block holds `cmt`, `family`, and family-specific parameter expressions.
    // Only compiled when the `survival` feature is enabled; the field
    // `model.endpoints` is always present (cfg-gated) and stays empty otherwise.
    //
    // Theta/eta indices used in event_model expressions are collected here so
    // that check_unused_parameters (below) can suppress false "not referenced"
    // warnings for parameters that only appear in [event_model].
    let mut event_model_used_thetas: std::collections::HashSet<usize> =
        std::collections::HashSet::new();
    let mut event_model_used_etas: std::collections::HashSet<usize> =
        std::collections::HashSet::new();
    #[cfg(feature = "survival")]
    {
        let theta_names = model.theta_names.clone();
        let eta_names = model.eta_names.clone();

        // Collect all [event_model] line-sets: unnamed (at most one) + named (any number).
        let mut event_blocks: Vec<&Vec<String>> = Vec::new();
        if let Some(lines) = blocks.get("event_model") {
            event_blocks.push(lines);
        }
        if let Some(named_map) = extracted.named.get("event_model") {
            for lines in named_map.values() {
                event_blocks.push(lines);
            }
        }

        for lines in event_blocks {
            let (cmt, endpoint, event_covs, blk_thetas, blk_etas) =
                parse_event_model_block(lines, &theta_names, &eta_names, &model.error_spec)?;
            if model.endpoints.contains_key(&cmt) {
                return Err(format!("[event_model]: CMT={cmt} declared more than once"));
            }
            model.endpoints.insert(cmt, endpoint);
            // Union covariate names from [event_model] expressions into the model's
            // covariate set so they appear in the covariate table and validation warnings.
            for cov in event_covs {
                if !model.referenced_covariates.contains(&cov) {
                    model.referenced_covariates.push(cov);
                }
            }
            event_model_used_thetas.extend(blk_thetas);
            event_model_used_etas.extend(blk_etas);
        }
        model.referenced_covariates.sort();
    }

    // Warn about declared-but-unused parameters. Runs here (after [event_model]
    // parsing) so that parameters used only in [event_model] are not falsely
    // reported as unused in mixed PK+TTE models.
    model.parse_warnings.extend(check_unused_parameters(
        &thetas,
        &eta_names_bsv,
        &kappa_names,
        n_eta,
        &model.default_params.sigma.names,
        &indiv_stmts,
        &used_sigmas_in_error,
        &event_model_used_thetas,
        &event_model_used_etas,
    ));

    // Warn about analytical PK parameters that are mapped but unused by the
    // chosen model — e.g. `ka` on an IV model (no absorption compartment), or
    // `q`/`v2` on a one-compartment model. They are set but have no effect
    // (#309). `PkModel::consumes_pk_slot` is the single source of truth for what
    // each model's closed form actually reads (`f` and `lagtime` apply to every
    // model — `f` scales IV bolus/infusion doses too since #327). Sibling to the
    // declared-but-unused check.
    if !pk_param_map.is_empty() {
        let mut unused: Vec<&str> = pk_param_map
            .iter()
            .filter_map(|(key, _)| {
                let slot = PkParams::name_to_index(key)?;
                (!pk_model.consumes_pk_slot(slot)).then_some(key.as_str())
            })
            .collect();
        unused.sort_unstable();
        if !unused.is_empty() {
            let list = unused
                .iter()
                .map(|k| format!("`{k}`"))
                .collect::<Vec<_>>()
                .join(", ");
            model.parse_warnings.push(format!(
                "[structural_model] {} does not use parameter(s) {list}; they are \
                 mapped but have no effect on this model. Remove them, or use a model \
                 that needs them.",
                pk_model.canonical_name()
            ));
        }
    }

    // Warn about an individual parameter that is computed but never used: it has
    // no effect because it is neither consumed by the structural model nor
    // referenced in any other block — e.g. an analytical oral model that declares
    // `F` to estimate bioavailability but forgets `f=F`, an ODE model that declares
    // `KE` but never uses it in the `[odes]` RHS, or an unused intermediate like
    // `ke = CL/V`. Implemented as a whole-identifier census over every model block:
    // a declared parameter whose name appears exactly once — its own
    // `[individual_parameters]` declaration — is dead. Over-counting (e.g. a name
    // in a comment) only ever makes a parameter look *used*, so this never
    // produces a false "unused" warning.
    //
    // Runs for analytical (`pk(...)`) and ODE models alike — the census already
    // tokenizes `[odes]` (an unnamed block), so a parameter used in the RHS counts
    // as used. The one ODE carve-out: parameters that `ode_param_slots` routes to
    // the engine-reserved slots `PK_IDX_F` / `PK_IDX_LAGTIME` (named `f`/`lagtime`/
    // `alag`) are applied to the dose by the engine without ever appearing in the
    // RHS, so their textual absence does not make them dead (#315). Analytical
    // models bind F/lagtime only via an explicit `pk(...)` mapping, which the
    // census counts, so they need no carve-out there. Pure-TTE models (no `pk(...)`,
    // no `[odes]`) are skipped: their params live in named `[event_model LABEL]`
    // blocks, which the census deliberately does not tokenize (see below).
    //
    // A raw-text census (rather than resolving against the parsed ASTs) is the
    // deliberate choice: it covers *every* block uniformly — including ones whose
    // references aren't retained as walkable ASTs at this point (`[output]`,
    // `[scaling]`, …) — so it cannot false-positive by overlooking a usage site.
    // It iterates `blocks.values()` = the *unnamed* blocks only; this is safe
    // because individual-parameter names are confined to unnamed blocks (named
    // `[event_model LABEL]` / `[covariate_nn NAME]` blocks reference thetas/etas/
    // covariates, never indiv params — so a param can't be "used" solely there).
    if !pk_param_map.is_empty() || is_ode {
        let mut token_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for lines in blocks.values() {
            for line in lines {
                for tok in line.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
                    if tok.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_') {
                        *token_counts.entry(tok.to_string()).or_insert(0) += 1;
                    }
                }
            }
        }
        let mut dead: Vec<String> = model
            .indiv_param_names
            .iter()
            .enumerate()
            .filter(|(_, name)| token_counts.get(name.as_str()).copied().unwrap_or(0) <= 1)
            // Exempt engine-applied slots on ODE models: bare `f`/`lagtime`/`alag`
            // (routed to RESERVED_PK_SLOTS) and the compartment-indexed dose
            // attributes `Fn`/`ALAGn` (issue #369) are applied to the dose by the
            // engine without a textual RHS reference, so absence from `[odes]` does
            // not make them dead. The bare case reuses `ode_param_slots`' own routing
            // (`ode_slot_map`, parallel to `indiv_param_names`) so the exemption can't
            // drift; the indexed case matches the same `from_indexed_name` predicate
            // that built `dose_attr_map`. No-op for analytical models (`is_ode` false;
            // their F/lagtime are bound via an explicit `pk(...)` mapping the census
            // already counts).
            .filter(|(i, name)| {
                !(is_ode
                    && (ode_slot_map
                        .get(*i)
                        .is_some_and(|slot| RESERVED_PK_SLOTS.contains(slot))
                        || crate::types::DoseAttr::from_indexed_name(name).is_some()))
            })
            .map(|(_, name)| name.clone())
            .collect();
        dead.sort_unstable();
        if !dead.is_empty() {
            let list = dead
                .iter()
                .map(|n| format!("`{n}`"))
                .collect::<Vec<_>>()
                .join(", ");
            let (verb, subj, obj, aux) = if dead.len() == 1 {
                ("is", "it", "it", "has")
            } else {
                ("are", "they", "them", "have")
            };
            // Shared scaffold; only the cause clause and the remediation hint
            // differ between analytical (`pk(...)`) and ODE models.
            let (cause, fix) = if is_ode {
                (
                    "not referenced in the [odes] RHS or any other block",
                    format!(
                        "Reference {obj} in [odes] (or [scaling]/[derived]/[output]) or remove {obj}."
                    ),
                )
            } else {
                (
                    "not mapped into the `pk(...)` model and not referenced in any other block",
                    format!("Map {obj} in [structural_model] (e.g. `f=F`) or remove {obj}."),
                )
            };
            model.parse_warnings.push(format!(
                "[individual_parameters] {list} {verb} computed but never used — {cause}, \
                 so {subj} {aux} no effect. {fix}"
            ));
        }
    }

    // Undeclared-covariate warning: checked here (after [event_model] parsing) so
    // that covariates used only in [event_model] expressions are included.
    if let Some(decls) = &covariate_decls {
        let declared: std::collections::HashSet<&str> =
            decls.iter().map(|d| d.name.as_str()).collect();
        let undeclared: Vec<&str> = model
            .referenced_covariates
            .iter()
            .filter(|c| !declared.contains(c.as_str()))
            .map(|s| s.as_str())
            .collect();
        if !undeclared.is_empty() {
            model.parse_warnings.push(format!(
                "Covariate(s) used in model expressions but not declared in [covariates]: \
                 {}. They are still usable, but declaring them (with `continuous`/`categorical`) \
                 lets ferx record their type and include them in the covariate table.",
                undeclared.join(", ")
            ));
        }
    }

    Ok(ParsedModel {
        model,
        simulation,
        fit_options,
        covariate_decls,
        block_lines: extracted.block_lines.clone(),
    })
}

// ── [derived] and [output] block parsers ────────────────────────────────────

/// Maximum `compartments[N]` index accepted at parse time.
///
/// `build_derived_vars` pre-seeds sentinel NaN entries for `__cmt_0` through
/// `__cmt_{MAX_CMT_INDEX}` (i.e. `MAX_CMT_INDEX + 1` entries total).  Any
/// index > MAX_CMT_INDEX is rejected at parse time so `eval_expression`'s
/// `.unwrap_or(0.0)` fallback can never silently return 0.0 for an
/// out-of-range access.  Both constants are derived from this single value so
/// they cannot drift apart.
pub(crate) const MAX_CMT_INDEX: usize = 255;

/// Built-in sdtab column names that [derived] names must not clash with.
const DERIVED_BUILTIN_NAMES: &[&str] = &[
    "ID", "TIME", "DV", "PRED", "IPRED", "CWRES", "IWRES", "NPDE", "NPD", "EBE_OFV", "N_OBS",
    "TAFD", "TAD", "CENS", "OCC", "CMT",
];

/// Split a token slice at top-level commas (depth 0 inside parentheses).
fn split_top_level_commas(tokens: &[Token]) -> Vec<&[Token]> {
    let mut result = Vec::new();
    let mut depth = 0usize;
    let mut start = 0;
    for (i, tok) in tokens.iter().enumerate() {
        match tok {
            Token::LParen => depth += 1,
            Token::RParen => {
                if depth > 0 {
                    depth -= 1;
                }
            }
            Token::Comma if depth == 0 => {
                result.push(&tokens[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    if start < tokens.len() {
        result.push(&tokens[start..]);
    }
    result
}

/// Returns true when a token slice starts with `IDENT =` (keyword arg pattern).
fn is_keyword_arg_tokens(tokens: &[Token]) -> bool {
    matches!(tokens, [Token::Ident(_), Token::Eq, ..])
}

/// Parse `IDENT = NUMBER` keyword arg, returning (key, value).
fn parse_keyword_float_arg(tokens: &[Token]) -> Result<(String, f64), String> {
    match tokens {
        [Token::Ident(k), Token::Eq, Token::Number(v)] => Ok((k.clone(), *v)),
        [Token::Ident(k), Token::Eq, ..] => Err(format!(
            "keyword arg `{k}=` must be followed by a numeric literal"
        )),
        _ => Err(format!(
            "expected keyword arg `name = value`, got unexpected tokens"
        )),
    }
}

/// Returns true when a token slice contains a comparison or logical operator
/// at parenthesis depth 0. Used to distinguish a condition arg from a
/// keyword-arg sequence in `integral(...)`.
fn tokens_contain_comparison(tokens: &[Token]) -> bool {
    let mut depth = 0usize;
    for tok in tokens {
        match tok {
            Token::LParen => depth += 1,
            Token::RParen => {
                if depth > 0 {
                    depth -= 1;
                }
            }
            Token::Lt
            | Token::Le
            | Token::Gt
            | Token::Ge
            | Token::EqEq
            | Token::Ne
            | Token::AndAnd
            | Token::OrOr
            | Token::Bang
                if depth == 0 =>
            {
                return true
            }
            _ => {}
        }
    }
    false
}

/// Parse a token slice as a fully-consuming arithmetic expression.
fn parse_derived_expr(tokens: &[Token], ctx: ParseCtx<'_>) -> Result<Expression, String> {
    if tokens.is_empty() {
        return Err("expected expression, got empty token list".into());
    }
    let (expr, pos) = parse_add_sub(tokens, 0, ctx)?;
    if pos < tokens.len() {
        return Err(format!(
            "unexpected token(s) after expression: {:?}",
            &tokens[pos..]
        ));
    }
    Ok(expr)
}

/// Parse a token slice as a fully-consuming boolean condition.
fn parse_derived_cond(tokens: &[Token], ctx: ParseCtx<'_>) -> Result<Condition, String> {
    if tokens.is_empty() {
        return Err("expected condition, got empty token list".into());
    }
    let (cond, pos) = parse_condition(tokens, 0, ctx)?;
    if pos < tokens.len() {
        return Err(format!(
            "unexpected token(s) after condition: {:?}",
            &tokens[pos..]
        ));
    }
    Ok(cond)
}

/// Build a DerivedEvalFn closure from a parsed Expression.
/// At evaluation time the closure assembles a `vars` map from the DerivedContext
/// fields and calls the AST-walking `eval_expression`.
fn build_derived_eval_fn(expr: Expression) -> DerivedEvalFn {
    let expr = std::sync::Arc::new(expr);
    Box::new(move |ctx: &DerivedContext<'_>| {
        let vars = build_derived_vars(ctx);
        // Covariates become Variable nodes (fallback_covariate=false at parse time),
        // so pass an empty covariates map and rely on vars for all name resolution.
        eval_expression(&expr, ctx.theta, ctx.eta, &HashMap::new(), &vars, &[])
    })
}

/// Build a DerivedFilterFn closure from a parsed Condition.
fn build_derived_filter_fn(cond: Condition) -> DerivedFilterFn {
    let cond = std::sync::Arc::new(cond);
    Box::new(move |ctx: &DerivedContext<'_>| {
        let vars = build_derived_vars(ctx);
        eval_condition(&cond, ctx.theta, ctx.eta, &HashMap::new(), &vars, &[])
    })
}

/// Build the variable map for derived-expression and derived-filter evaluation.
///
/// Keys for covariates, individual parameters, and prior derived columns are
/// inserted with their original case **and** both an uppercase and a lowercase
/// alias, so that an all-lowercase `wt` or all-uppercase `WT` expression resolves
/// regardless of the dataset header's case. (A header such as `WT` has an
/// uppercase form identical to itself, so the lowercase alias is what makes a
/// lowercase `wt` reference resolve.)
/// Built-in time variables (TIME, TAFD, TAD, IPRED, PRED, DV) are inserted in
/// both uppercase and lowercase for the same reason.
fn build_derived_vars(ctx: &DerivedContext<'_>) -> HashMap<String, f64> {
    let mut vars: HashMap<String, f64> = HashMap::new();

    // Index-based compartment keys.
    // Pre-seed indices 0..=MAX_CMT_INDEX with NaN so that:
    //   a) valid indices with empty compartment_states (IOV, analytical TV-covariate,
    //      analytical reset subjects) evaluate to NaN rather than 0.0, and
    //   b) out-of-range accesses (e.g. compartments[5] on a 1-cpt model) also
    //      produce NaN — 0.0 would silently look like an empty compartment.
    // Actual values then overwrite the NaN sentinels for valid indices.
    // MAX_CMT_INDEX is chosen to cover all practical PK/PBPK models
    // (largest published PBPK has ~30 compartments; 256 leaves generous headroom).
    // Any access compartments[i] for i > MAX_CMT_INDEX is rejected at parse time;
    // indices 0..=MAX_CMT_INDEX that are beyond the model's actual n_states return
    // NaN via this sentinel. Without the sentinel the HashMap miss returns 0.0,
    // which silently looks like an empty compartment.
    // MAX_CMT_INDEX is defined at module scope; the sentinel covers 0..=MAX_CMT_INDEX.
    for i in 0..=MAX_CMT_INDEX {
        vars.insert(format!("__cmt_{i}"), f64::NAN);
    }
    for (i, &v) in ctx.compartments.iter().enumerate() {
        vars.insert(format!("__cmt_{i}"), v); // overwrite sentinel with actual
    }

    // Insert each name with original case AND uppercase + lowercase aliases so
    // that case mismatches (e.g. `wt` vs. header `WT`, or `WT` vs. header `wt`)
    // resolve correctly. `eval_expression` looks names up verbatim, so every
    // case the user might type must be present as a key.
    let mut insert_ci = |k: &str, v: f64| {
        vars.insert(k.to_string(), v);
        let up = k.to_uppercase();
        if up != k {
            vars.insert(up, v);
        }
        let lo = k.to_lowercase();
        if lo != k {
            vars.insert(lo, v);
        }
    };

    // Named compartment access — lowest priority among `insert_ci` inserts.
    // Pre-insert NaN so unavailable states surface as NaN in [derived];
    // individual params, covariates, and built-ins below overwrite any clash.
    for name in ctx.compartment_names.iter() {
        insert_ci(name, f64::NAN);
    }
    for (name, &v) in ctx.compartment_names.iter().zip(ctx.compartments.iter()) {
        insert_ci(name, v); // overwrite sentinel with actual value
    }

    for (k, &v) in ctx.indiv_params.iter() {
        insert_ci(k, v);
    }
    for (k, &v) in ctx.covariates.iter() {
        insert_ci(k, v);
    }
    for (k, &v) in ctx.prev_derived.iter() {
        insert_ci(k, v);
    }
    // Built-in names: uppercase (canonical) + lowercase alias.
    for &(name, val) in &[
        ("IPRED", ctx.ipred),
        ("PRED", ctx.pred),
        ("DV", ctx.dv),
        ("TIME", ctx.time),
        ("TAFD", ctx.tafd),
        ("TAD", ctx.tad),
        ("MACHEPS", f64::EPSILON),
    ] {
        vars.insert(name.to_string(), val);
        vars.insert(name.to_lowercase(), val);
    }
    vars
}

/// Returns true if the expression subtree references the `DV` variable.
fn expr_refs_dv(expr: &Expression) -> bool {
    match expr {
        Expression::Variable(name) if name.eq_ignore_ascii_case("DV") => true,
        Expression::BinOp(l, _, r) => expr_refs_dv(l) || expr_refs_dv(r),
        Expression::UnaryFn(_, arg) => expr_refs_dv(arg),
        Expression::Power(b, e) => expr_refs_dv(b) || expr_refs_dv(e),
        Expression::Conditional(c, t, e) => cond_refs_dv(c) || expr_refs_dv(t) || expr_refs_dv(e),
        _ => false,
    }
}

fn cond_refs_dv(cond: &Condition) -> bool {
    match cond {
        Condition::Compare(l, _, r) => expr_refs_dv(l) || expr_refs_dv(r),
        Condition::And(l, r) | Condition::Or(l, r) => cond_refs_dv(l) || cond_refs_dv(r),
        Condition::Not(c) => cond_refs_dv(c),
    }
}

/// Returns true if the expression subtree references a compartment state.
/// Matches both `__cmt_N` (from `compartments[N]` syntax) and named ODE state
/// variables (e.g. `Ce`, `depot`). Named references are detected by checking
/// `ode_state_names`; an empty slice means only subscript-style access matches.
fn expr_refs_compartments(expr: &Expression, ode_state_names: &[String]) -> bool {
    match expr {
        Expression::Variable(s) => {
            s.starts_with("__cmt_") || ode_state_names.iter().any(|n| n.eq_ignore_ascii_case(s))
        }
        Expression::BinOp(l, _, r) => {
            expr_refs_compartments(l, ode_state_names) || expr_refs_compartments(r, ode_state_names)
        }
        Expression::UnaryFn(_, arg) => expr_refs_compartments(arg, ode_state_names),
        Expression::Power(b, e) => {
            expr_refs_compartments(b, ode_state_names) || expr_refs_compartments(e, ode_state_names)
        }
        Expression::Conditional(c, t, e) => {
            cond_refs_compartments(c, ode_state_names)
                || expr_refs_compartments(t, ode_state_names)
                || expr_refs_compartments(e, ode_state_names)
        }
        _ => false,
    }
}

fn cond_refs_compartments(cond: &Condition, ode_state_names: &[String]) -> bool {
    match cond {
        Condition::Compare(l, _, r) => {
            expr_refs_compartments(l, ode_state_names) || expr_refs_compartments(r, ode_state_names)
        }
        Condition::And(l, r) | Condition::Or(l, r) => {
            cond_refs_compartments(l, ode_state_names) || cond_refs_compartments(r, ode_state_names)
        }
        Condition::Not(c) => cond_refs_compartments(c, ode_state_names),
    }
}

/// Parse the interior args of `integral(...)` into a `DerivedKind::Integral`.
fn parse_integral_kind(
    args: &[&[Token]],
    ctx: ParseCtx<'_>,
    parse_warnings: &mut Vec<String>,
) -> Result<DerivedKind, String> {
    if args.is_empty() {
        return Err(
            "[derived] integral() requires at least one argument (the integrand expression)".into(),
        );
    }

    let integrand_expr = parse_derived_expr(args[0], ctx)?;
    let data_based = expr_refs_dv(&integrand_expr);
    // Also checked against condition below — must be `mut` so we can OR in
    // the condition's compartment references before the struct is built.
    let mut uses_compartments = expr_refs_compartments(&integrand_expr, ctx.ode_state_names);

    let mut condition: Option<DerivedFilterFn> = None;
    let mut from_val: Option<f64> = None;
    let mut to_val: Option<f64> = None;
    let mut window_val: Option<f64> = None;
    let mut anchor_val: f64 = 0.0;
    let mut step_val: Option<f64> = None;

    let mut i = 1;
    // Second positional arg is a condition if it contains comparison operators.
    if i < args.len() && !is_keyword_arg_tokens(args[i]) && tokens_contain_comparison(args[i]) {
        let cond = parse_derived_cond(args[i], ctx)?;
        // If the filter condition references a compartment (e.g.
        // `integral(IPRED, compartments[0] > threshold, from=0, to=24)`),
        // we must set uses_compartments so the grid path populates the state
        // vectors before the condition closure is evaluated — otherwise the
        // filter sees NaN for all __cmt_N keys. Matches max/min/tmax handling
        // at lines 2332–2335.
        if cond_refs_compartments(&cond, ctx.ode_state_names) {
            uses_compartments = true;
        }
        condition = Some(build_derived_filter_fn(cond));
        i += 1;
    }

    // Remaining args are keyword args.
    while i < args.len() {
        let (k, v) =
            parse_keyword_float_arg(args[i]).map_err(|e| format!("[derived] integral(): {e}"))?;
        match k.to_lowercase().as_str() {
            "from" => from_val = Some(v),
            "to" => to_val = Some(v),
            "window" => {
                if v <= 0.0 {
                    return Err(format!(
                        "[derived] integral(): `window=` must be positive, got {v}"
                    ));
                }
                window_val = Some(v);
            }
            "anchor" => anchor_val = v,
            "step" => {
                if v <= 0.0 {
                    return Err(format!(
                        "[derived] integral(): `step=` must be positive, got {v}"
                    ));
                }
                step_val = Some(v);
            }
            other => {
                return Err(format!(
                    "[derived] integral(): unknown keyword argument `{other}`"
                ))
            }
        }
        i += 1;
    }

    let int_window = if let Some(w) = window_val {
        if from_val.is_some() || to_val.is_some() {
            return Err(
                "[derived] integral(): cannot specify both `window=` and `from=`/`to=`".into(),
            );
        }
        IntegralWindow::Periodic {
            period: w,
            anchor: anchor_val,
        }
    } else {
        let f = from_val.ok_or("[derived] integral(): missing required argument `from=`")?;
        let t = to_val.ok_or("[derived] integral(): missing required argument `to=`")?;
        IntegralWindow::Explicit { from: f, to: t }
    };

    let int_step = if data_based {
        if step_val.is_some() {
            parse_warnings.push(
                "[derived] integral(): `step=` is ignored for DV-based integrands \
                 (W_DERIVED_STEP_IGNORED)"
                    .into(),
            );
        }
        IntegralStep::ObsTimes
    } else if let Some(s) = step_val {
        IntegralStep::Fixed(s)
    } else {
        IntegralStep::Auto
    };

    Ok(DerivedKind::Integral {
        integrand: build_derived_eval_fn(integrand_expr),
        condition,
        data_based,
        uses_compartments,
        window: int_window,
        step: int_step,
    })
}

/// Parse a `[derived]` block into a sequence of `DerivedExprSpec`s.
///
/// Each line must have the form `NAME = <rhs>` where RHS is one of:
/// - A plain arithmetic expression → `PerRow`
/// - `max(expr)` / `max(expr, cond)` → `Aggregate(Max)`
/// - `min(expr)` / `min(expr, cond)` → `Aggregate(Min)`
/// - `tmax(expr)` / `tmax(expr, cond)` → `Aggregate(Tmax)`
/// - `integral(expr, ...)` with keyword args → `Integral`
fn parse_derived_block(
    lines: &[String],
    theta_names: &[String],
    eta_names: &[String],
    indiv_param_names: &[String],
    covariate_names: &[String],
    ode_state_names: &[String],
    parse_warnings: &mut Vec<String>,
) -> Result<Vec<DerivedExprSpec>, String> {
    let mut specs: Vec<DerivedExprSpec> = Vec::new();
    let mut defined_derived: Vec<String> = Vec::new();

    // Names that resolve to Variable (not Theta/Eta) at parse time; these are
    // looked up in the vars HashMap at closure-call time.
    const BUILTIN_SPECIALS: &[&str] = &["IPRED", "PRED", "DV", "TIME", "TAFD", "TAD", "MACHEPS"];

    for raw_line in lines {
        let line = if let Some(idx) = raw_line.find('#') {
            &raw_line[..idx]
        } else {
            raw_line.as_str()
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Split at first `=` to separate NAME from RHS.
        let eq_pos = line
            .find('=')
            .ok_or_else(|| format!("[derived] line `{line}` must have the form `NAME = <expr>`"))?;
        let name = line[..eq_pos].trim().to_string();
        let rhs_str = line[eq_pos + 1..].trim().to_string();

        if name.is_empty() {
            return Err(format!(
                "[derived] line `{line}` has an empty name before `=`"
            ));
        }

        // ── Name conflict checks ──────────────────────────────────────────────
        if DERIVED_BUILTIN_NAMES
            .iter()
            .any(|b| b.eq_ignore_ascii_case(&name))
            || eta_names.iter().any(|e| e.eq_ignore_ascii_case(&name))
        {
            return Err(format!(
                "[derived] name `{name}` conflicts with a built-in sdtab column or eta name \
                 (E_DERIVED_NAME_CONFLICT)"
            ));
        }
        if theta_names.iter().any(|t| t.eq_ignore_ascii_case(&name))
            || indiv_param_names
                .iter()
                .any(|p| p.eq_ignore_ascii_case(&name))
        {
            return Err(format!(
                "[derived] name `{name}` conflicts with a theta or individual-parameter name \
                 (E_DERIVED_NAME_CONFLICT)"
            ));
        }
        if covariate_names
            .iter()
            .any(|c| c.eq_ignore_ascii_case(&name))
        {
            parse_warnings.push(format!(
                "[derived] name `{name}` shadows a covariate — this may be confusing \
                 (W_DERIVED_COVARIATE_SHADOW)"
            ));
        }

        // ── Build ParseCtx ────────────────────────────────────────────────────
        // Unknown identifiers become Variable (fallback_covariate = false), so
        // indiv params, covariates, and context fields (IPRED, DV, etc.) are all
        // resolved via the vars HashMap at closure-call time.
        let mut defined_vars: Vec<String> = indiv_param_names.to_vec();
        defined_vars.extend(BUILTIN_SPECIALS.iter().map(|s| s.to_string()));
        defined_vars.extend(covariate_names.iter().cloned());
        defined_vars.extend(defined_derived.iter().cloned());

        let ctx = ParseCtx {
            theta_names,
            eta_names,
            defined_vars: &defined_vars,
            fallback_covariate: false,
            nn_specs: &[],
            ode_state_names,
        };

        // ── Tokenize ──────────────────────────────────────────────────────────
        let mut tokens = tokenize(&rhs_str)?;
        tokens.retain(|t| !matches!(t, Token::Newline));

        if tokens.is_empty() {
            return Err(format!("[derived] `{name}` has an empty right-hand side"));
        }

        // ── Detect form ───────────────────────────────────────────────────────
        // `spec_uses_compartments` tracks whether this derived expression
        // references compartments[i] or named ODE state variables anywhere.
        // Propagated into `DerivedExprSpec::uses_compartments` so that the
        // post-fit warning logic can gate on "compartments actually requested"
        // rather than "any [derived] block exists".
        let (kind, spec_uses_compartments) = if let Token::Ident(func_name) = &tokens[0] {
            let fname_lc = func_name.to_lowercase();
            if matches!(fname_lc.as_str(), "max" | "min" | "tmax" | "integral")
                && tokens.get(1) == Some(&Token::LParen)
            {
                // Find the matching closing paren at depth 0 (after the opening LParen at [1]).
                let mut depth = 0usize;
                let mut close_idx = None;
                for (i, tok) in tokens[1..].iter().enumerate() {
                    match tok {
                        Token::LParen => depth += 1,
                        Token::RParen => {
                            depth -= 1;
                            if depth == 0 {
                                close_idx = Some(i + 1); // index in `tokens`
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                let close_idx = close_idx.ok_or_else(|| {
                    format!("[derived] `{name}`: `{fname_lc}(` missing closing `)`")
                })?;

                // tokens[2..close_idx] is the interior
                let interior = &tokens[2..close_idx];
                let args = split_top_level_commas(interior);

                if fname_lc == "integral" {
                    let kind = parse_integral_kind(&args, ctx, parse_warnings)?;
                    let uses = matches!(
                        kind,
                        DerivedKind::Integral {
                            uses_compartments: true,
                            ..
                        }
                    );
                    (kind, uses)
                } else {
                    // max / min / tmax
                    let agg_fn = match fname_lc.as_str() {
                        "max" => AggFunction::Max,
                        "min" => AggFunction::Min,
                        "tmax" => AggFunction::Tmax,
                        _ => unreachable!(),
                    };
                    let value_tokens = args.first().copied().unwrap_or(&[]);
                    let filter_tokens = if args.len() >= 2 { Some(args[1]) } else { None };
                    let value_expr = parse_derived_expr(value_tokens, ctx)?;
                    let value_uses = expr_refs_compartments(&value_expr, ctx.ode_state_names);
                    let (filter, filter_uses) = if let Some(ft) = filter_tokens {
                        let cond = parse_derived_cond(ft, ctx)?;
                        let uses = cond_refs_compartments(&cond, ctx.ode_state_names);
                        (Some(build_derived_filter_fn(cond)), uses)
                    } else {
                        (None, false)
                    };
                    let kind = DerivedKind::Aggregate {
                        func: agg_fn,
                        value: build_derived_eval_fn(value_expr),
                        filter,
                    };
                    (kind, value_uses || filter_uses)
                }
            } else {
                // Plain expression → PerRow
                let expr = parse_derived_expr(&tokens, ctx)?;
                let uses = expr_refs_compartments(&expr, ctx.ode_state_names);
                (
                    DerivedKind::PerRow {
                        eval: build_derived_eval_fn(expr),
                    },
                    uses,
                )
            }
        } else {
            // Starts with a literal or unary minus
            let expr = parse_derived_expr(&tokens, ctx)?;
            let uses = expr_refs_compartments(&expr, ctx.ode_state_names);
            (
                DerivedKind::PerRow {
                    eval: build_derived_eval_fn(expr),
                },
                uses,
            )
        };

        defined_derived.push(name.clone());
        specs.push(DerivedExprSpec {
            name,
            kind,
            uses_compartments: spec_uses_compartments,
        });
    }

    Ok(specs)
}

/// Parse an `[output]` block: whitespace-separated column names.
fn parse_output_block(lines: &[String]) -> Vec<String> {
    lines
        .iter()
        .flat_map(|line| {
            let line = if let Some(idx) = line.find('#') {
                &line[..idx]
            } else {
                line.as_str()
            };
            line.split_whitespace().map(|s| s.to_string())
        })
        .collect()
}

/// Map a `[covariates]` type token to a [`CovariateKind`]. Accepts the full
/// words and the `cont`/`cat` shorthands, case-insensitively.
fn parse_covariate_kind(token: &str) -> Option<CovariateKind> {
    match token.trim().to_lowercase().as_str() {
        "continuous" | "cont" => Some(CovariateKind::Continuous),
        "categorical" | "cat" => Some(CovariateKind::Categorical),
        _ => None,
    }
}

fn push_covariate_decl(
    decls: &mut Vec<CovariateDecl>,
    seen: &mut std::collections::HashSet<String>,
    name: &str,
    kind: CovariateKind,
) -> Result<(), String> {
    let name = name.trim();
    if name.is_empty() {
        return Err("[covariates]: empty covariate name".to_string());
    }
    // Names are case-sensitive (matching the data reader's covariate lookup).
    if !seen.insert(name.to_string()) {
        return Err(format!(
            "[covariates]: covariate '{}' is declared more than once",
            name
        ));
    }
    decls.push(CovariateDecl {
        name: name.to_string(),
        kind,
    });
    Ok(())
}

/// Parse the optional `[covariates]` block into ordered declarations.
///
/// Two line forms are accepted and may be mixed:
///   - `NAME TYPE`              e.g. `WT continuous`
///   - `TYPE: NAME, NAME, ...`  e.g. `continuous: WT, HT, CRCL`
///
/// where TYPE is `continuous`/`cont` or `categorical`/`cat` (case-insensitive).
/// Declaration order is preserved. Duplicate names and unknown types are errors.
fn parse_covariates_block(lines: &[String]) -> Result<Vec<CovariateDecl>, String> {
    let mut decls: Vec<CovariateDecl> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    for line in lines {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(colon) = line.find(':') {
            // `TYPE: NAME, NAME, ...`
            let (ty_str, rest) = line.split_at(colon);
            let kind = parse_covariate_kind(ty_str).ok_or_else(|| {
                format!(
                    "[covariates]: unknown covariate type '{}' (expected continuous/cont or \
                     categorical/cat)",
                    ty_str.trim()
                )
            })?;
            let names = &rest[1..]; // skip the ':'
            let mut any = false;
            for name in names.split(',') {
                let name = name.trim();
                if !name.is_empty() {
                    push_covariate_decl(&mut decls, &mut seen, name, kind)?;
                    any = true;
                }
            }
            if !any {
                return Err(format!(
                    "[covariates]: type '{}' declared with no covariate names",
                    ty_str.trim()
                ));
            }
        } else {
            // `NAME TYPE`
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 2 {
                return Err(format!(
                    "[covariates]: expected `NAME TYPE` (e.g. `WT continuous`) or \
                     `TYPE: NAME, ...`, got '{}'",
                    line
                ));
            }
            let kind = parse_covariate_kind(parts[1]).ok_or_else(|| {
                format!(
                    "[covariates]: unknown covariate type '{}' for '{}' (expected continuous/cont \
                     or categorical/cat)",
                    parts[1], parts[0]
                )
            })?;
            push_covariate_decl(&mut decls, &mut seen, parts[0], kind)?;
        }
    }

    Ok(decls)
}

// ── [event_model] block parser ─────────────────────────────────────────────

/// Parse one `[event_model]` (or `[event_model NAME]`) block.
///
/// Returns `(cmt, EndpointLikelihood::Tte { hazard })` ready to insert into
/// `CompiledModel::endpoints`.
///
/// Supported keys:
/// - `cmt`    — required; positive integer (data-file CMT column value)
/// - `family` — required; `exponential` | `weibull` | `gompertz`
/// - `scale`  — required for Exponential (= λ, the rate) and Weibull (= scale parameter)
/// - `rate`   — alias for `scale` (Exponential only; kept for user ergonomics)
/// - `shape`  — required for Weibull
/// - `alpha`  — required for Gompertz (baseline hazard at t=0)
/// - `gamma`  — required for Gompertz (hazard growth rate)
/// - `loghr`  — optional (all families); log-hazard-ratio covariate term Σ(β·x)
#[cfg(feature = "survival")]
fn parse_event_model_block(
    lines: &[String],
    theta_names: &[String],
    eta_names: &[String],
    error_spec: &ErrorSpec,
) -> Result<
    (
        usize,
        crate::types::EndpointLikelihood,
        Vec<String>,
        std::collections::HashSet<usize>,
        std::collections::HashSet<usize>,
    ),
    String,
> {
    use crate::types::{EndpointLikelihood, HazardFamily, HazardSpec};

    let ctx = ParseCtx::new(theta_names, eta_names, &[]);

    let mut cmt_opt: Option<usize> = None;
    let mut family_opt: Option<HazardFamily> = None;
    // Exponential / Weibull scale parameter (lambda for Exp).
    let mut scale_expr: Option<Expression> = None;
    // Weibull shape (p).
    let mut shape_expr: Option<Expression> = None;
    // Gompertz: α (baseline hazard at t=0).
    let mut alpha_expr: Option<Expression> = None;
    // Gompertz: γ (hazard growth rate).
    let mut gamma_expr: Option<Expression> = None;
    // Any family: Σ(β·covariate) added on the log-hazard scale.
    let mut loghr_expr: Option<Expression> = None;

    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = trimmed.splitn(2, '=').map(|s| s.trim()).collect();
        if parts.len() != 2 {
            return Err(format!(
                "[event_model]: invalid line `{trimmed}` — expected `key = value`"
            ));
        }
        let (key, value) = (parts[0], parts[1]);

        match key {
            "cmt" => {
                cmt_opt = Some(value.parse::<usize>().map_err(|_| {
                    format!("[event_model]: invalid cmt `{value}` — expected a positive integer")
                })?);
            }
            "family" => {
                family_opt = Some(match value {
                    "exponential" => HazardFamily::Exponential,
                    "weibull" => HazardFamily::Weibull,
                    "gompertz" => HazardFamily::Gompertz,
                    other => {
                        return Err(format!(
                            "[event_model]: unknown family `{other}` \
                             — valid: exponential, weibull, gompertz"
                        ))
                    }
                });
            }
            "scale" | "rate" => {
                scale_expr = Some(parse_scalar_expression(value, ctx)?);
            }
            "shape" => {
                shape_expr = Some(parse_scalar_expression(value, ctx)?);
            }
            "alpha" => {
                alpha_expr = Some(parse_scalar_expression(value, ctx)?);
            }
            "gamma" => {
                gamma_expr = Some(parse_scalar_expression(value, ctx)?);
            }
            "loghr" => {
                loghr_expr = Some(parse_scalar_expression(value, ctx)?);
            }
            other => {
                return Err(format!(
                    "[event_model]: unknown key `{other}` \
                     — valid keys: cmt, family, scale, rate, shape, alpha, gamma, loghr"
                ));
            }
        }
    }

    let cmt = cmt_opt.ok_or("[event_model]: missing required key `cmt`")?;
    let family = family_opt.ok_or("[event_model]: missing required key `family`")?;

    // Guard: same CMT can't be both Gaussian and TTE.
    match error_spec {
        ErrorSpec::PerCmt(cmt_map) => {
            if cmt_map.contains_key(&cmt) {
                return Err(format!(
                    "[event_model]: CMT={cmt} is already declared as a Gaussian endpoint \
                     in [error_model] — the same CMT cannot be both Gaussian and TTE"
                ));
            }
        }
        ErrorSpec::Single(_) => {
            // A Single error model has no CMT restriction — it applies to every Gaussian
            // observation regardless of the CMT column value.  We cannot detect a collision
            // at parse time because the error model carries no CMT information.  If the user
            // places both Gaussian and TTE observations on the same CMT value in their dataset,
            // the data reader's two-path routing (Gaussian → obs_times, TTE → obs_records)
            // prevents actual double-counting in the NLL; however, the Gaussian path would
            // silently consume those rows via the Single error model, which is almost certainly
            // unintended.  Use a per-CMT error model (`DV[CMT=N] ~ ...`) to get unambiguous
            // parse-time validation.
        }
    }

    // Validate family-specific keys: reject keys that do not belong to the chosen family.
    // Accepting and silently dropping them would mislead users into thinking they had an effect.
    match family {
        HazardFamily::Exponential => {
            if shape_expr.is_some() {
                return Err("[event_model] family=exponential does not accept `shape` \
                     — remove it or switch to `family = weibull`"
                    .into());
            }
            if alpha_expr.is_some() || gamma_expr.is_some() {
                return Err(
                    "[event_model] family=exponential does not accept `alpha` or `gamma` \
                     — use `family = gompertz` for the Gompertz family"
                        .into(),
                );
            }
        }
        HazardFamily::Weibull => {
            if alpha_expr.is_some() || gamma_expr.is_some() {
                return Err(
                    "[event_model] family=weibull does not accept `alpha` or `gamma` \
                     — use `family = gompertz` for the Gompertz family"
                        .into(),
                );
            }
        }
        HazardFamily::Gompertz => {
            if scale_expr.is_some() {
                return Err("[event_model] family=gompertz does not accept `scale` \
                     — use `alpha` (baseline hazard at t=0) and `gamma` (growth rate) instead"
                    .into());
            }
            if shape_expr.is_some() {
                return Err("[event_model] family=gompertz does not accept `shape` \
                     — use `family = weibull` for the Weibull family"
                    .into());
            }
        }
    }

    // Collect covariate/theta/eta references from all expressions BEFORE they are
    // moved into the param_fn closure.
    let event_model_covariates: Vec<String>;
    let event_model_thetas: std::collections::HashSet<usize>;
    let event_model_etas: std::collections::HashSet<usize>;
    {
        let mut cov_set = std::collections::HashSet::new();
        let mut theta_set = std::collections::HashSet::new();
        let mut eta_set = std::collections::HashSet::new();
        for expr_opt in [
            &scale_expr,
            &shape_expr,
            &alpha_expr,
            &gamma_expr,
            &loghr_expr,
        ] {
            if let Some(expr) = expr_opt {
                collect_covariates(expr, &mut cov_set);
                collect_theta_eta(expr, &mut theta_set, &mut eta_set);
            }
        }
        let mut v: Vec<String> = cov_set.into_iter().collect();
        v.sort();
        event_model_covariates = v;
        event_model_thetas = theta_set;
        event_model_etas = eta_set;
    }

    // Build the param_fn closure that evaluates hazard parameters from (θ, η, covariates).
    // Expression nodes hold only indices, so they're safe to move into the closure.
    // Parameter layout matches parametric.rs: [scale/alpha, (shape/gamma), loghr].
    let param_fn: crate::types::HazardParamFn = match family {
        HazardFamily::Exponential => {
            let scale = scale_expr
                .ok_or("[event_model] family=exponential requires `scale` (or `rate`)")?;
            Box::new(
                move |theta: &[f64], eta: &[f64], covariates: &HashMap<String, f64>| {
                    let lambda =
                        eval_expression(&scale, theta, eta, covariates, &HashMap::new(), &[]);
                    let lhr = loghr_expr.as_ref().map_or(0.0, |e| {
                        eval_expression(e, theta, eta, covariates, &HashMap::new(), &[])
                    });
                    vec![lambda, lhr]
                },
            )
        }
        HazardFamily::Weibull => {
            let scale = scale_expr.ok_or("[event_model] family=weibull requires `scale`")?;
            let shape = shape_expr.ok_or("[event_model] family=weibull requires `shape`")?;
            Box::new(
                move |theta: &[f64], eta: &[f64], covariates: &HashMap<String, f64>| {
                    let s = eval_expression(&scale, theta, eta, covariates, &HashMap::new(), &[]);
                    let p = eval_expression(&shape, theta, eta, covariates, &HashMap::new(), &[]);
                    let lhr = loghr_expr.as_ref().map_or(0.0, |e| {
                        eval_expression(e, theta, eta, covariates, &HashMap::new(), &[])
                    });
                    vec![s, p, lhr]
                },
            )
        }
        HazardFamily::Gompertz => {
            let alpha = alpha_expr.ok_or("[event_model] family=gompertz requires `alpha`")?;
            let gamma = gamma_expr.ok_or("[event_model] family=gompertz requires `gamma`")?;
            Box::new(
                move |theta: &[f64], eta: &[f64], covariates: &HashMap<String, f64>| {
                    let a = eval_expression(&alpha, theta, eta, covariates, &HashMap::new(), &[]);
                    let g = eval_expression(&gamma, theta, eta, covariates, &HashMap::new(), &[]);
                    let lhr = loghr_expr.as_ref().map_or(0.0, |e| {
                        eval_expression(e, theta, eta, covariates, &HashMap::new(), &[])
                    });
                    vec![a, g, lhr]
                },
            )
        }
    };

    Ok((
        cmt,
        EndpointLikelihood::Tte {
            hazard: HazardSpec::Analytic { family, param_fn },
        },
        event_model_covariates,
        event_model_thetas,
        event_model_etas,
    ))
}

// ── [simulation] block parser ───────────────────────────────────────────────

fn parse_simulation_block(lines: &[String]) -> Result<SimulationSpec, String> {
    let mut n_subjects = 10;
    let mut dose_amt = 100.0;
    let mut dose_cmt = 1;
    let mut obs_times = Vec::new();
    let mut seed = 42u64;

    for line in lines {
        let parts: Vec<&str> = line.splitn(2, '=').map(|s| s.trim()).collect();
        if parts.len() != 2 {
            continue;
        }
        match parts[0] {
            "subjects" => {
                n_subjects = parts[1]
                    .parse()
                    .map_err(|_| format!("Bad subjects: {}", line))?
            }
            "dose" => {
                dose_amt = parts[1]
                    .parse()
                    .map_err(|_| format!("Bad dose: {}", line))?
            }
            "cmt" => dose_cmt = parts[1].parse().map_err(|_| format!("Bad cmt: {}", line))?,
            "seed" => {
                seed = parts[1]
                    .parse()
                    .map_err(|_| format!("Bad seed: {}", line))?
            }
            "times" => obs_times = parse_float_array(parts[1])?,
            _ => {}
        }
    }
    if obs_times.is_empty() {
        return Err("[simulation] block requires 'times = [...]'".to_string());
    }

    Ok(SimulationSpec {
        n_subjects,
        dose_amt,
        dose_cmt,
        obs_times,
        seed,
        covariates: vec![],
    })
}

// ── [fit_options] block parser ──────────────────────────────────────────────

fn parse_method_token(token: &str) -> Result<EstimationMethod, String> {
    let val = token
        .trim()
        .trim_matches(|c| c == '"' || c == '\'')
        .to_lowercase();
    if val == "saem" {
        Ok(EstimationMethod::Saem)
    } else if val == "impmap"
        || val == "importance_sampling_map"
        || val == "importance-sampling-map"
    {
        Ok(EstimationMethod::Impmap)
    } else if val == "imp" || val == "importance_sampling" || val == "importance-sampling" {
        Ok(EstimationMethod::Imp)
    } else if val.contains("hybrid") || val == "gn_hybrid" || val == "gn-hybrid" {
        Ok(EstimationMethod::FoceGnHybrid)
    } else if val == "gn" || val.contains("gauss") {
        Ok(EstimationMethod::FoceGn)
    } else if val == "focei" || val == "foce-i" || val == "foce_i" || val.contains("interaction") {
        Ok(EstimationMethod::FoceI)
    } else if val == "foce" {
        Ok(EstimationMethod::Foce)
    } else {
        Err(format!("unknown estimation method: `{}`", token.trim()))
    }
}

fn parse_fit_options(lines: &[String]) -> Result<FitOptions, String> {
    let mut opts = FitOptions::default();
    for line in lines {
        let parts: Vec<&str> = line.splitn(2, '=').map(|s| s.trim()).collect();
        if parts.len() != 2 {
            continue;
        }
        if parts[0] == "method" {
            let raw = parts[1].trim();
            // List form: `method = [a, b, c]` — chain of stages.
            if raw.starts_with('[') {
                let inner = raw.trim_start_matches('[').trim_end_matches(']');
                let chain: Vec<EstimationMethod> = inner
                    .split(',')
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .map(parse_method_token)
                    .collect::<Result<_, _>>()?;
                if chain.is_empty() {
                    return Err("method = [] is empty; provide at least one method".into());
                }
                // Interaction flag follows the final stage of the chain.
                opts.interaction = *chain.last().unwrap() == EstimationMethod::FoceI;
                opts.method = *chain.last().unwrap();
                opts.methods = chain;
            } else {
                let m = parse_method_token(raw)?;
                opts.method = m;
                opts.methods.clear();
                if m == EstimationMethod::FoceI {
                    opts.interaction = true;
                }
            }
            opts.user_set_keys.push("method".to_string());
            continue;
        }
        // All other keys flow through the shared dispatch. Both `.ferx`
        // parsing and the R `settings` path are strict: unknown keys and
        // malformed values raise an error rather than silently defaulting.
        // A previous iteration of this parser used `.unwrap_or(default)` /
        // `== "true"` coercions that could silently flip behavior (e.g.
        // `covariance = TRUE` set `false`; `bloq_method = foo` landed on
        // the default `Drop`). Those traps are gone.
        match apply_fit_option(&mut opts, parts[0], parts[1]) {
            Ok(true) => {}
            Ok(false) => {
                return Err(format!("[fit_options]: unknown key `{}`", parts[0]));
            }
            Err(e) => return Err(format!("[fit_options]: {}", e)),
        }
    }
    Ok(opts)
}

/// Apply a single `key = value` pair to `FitOptions`.
///
/// Returns:
/// - `Ok(true)`  — key was recognized and applied.
/// - `Ok(false)` — key is not a known fit option.
/// - `Err(msg)`  — key is recognized but the value is malformed.
///
/// This is the single source of truth for the `[fit_options]` key grammar,
/// shared between `.ferx` parsing and the R wrapper's generic `settings`
/// list. Callers that want strict validation (e.g. the R wrapper) should
/// propagate `Err` and treat `Ok(false)` as "unknown setting".
///
/// Does NOT handle `method` (which has list-chain syntax) — that stays in
/// the block parser.
pub fn apply_fit_option(opts: &mut FitOptions, key: &str, value: &str) -> Result<bool, String> {
    let value = value.trim();

    let parse_usize = |name: &str| -> Result<usize, String> {
        value.parse::<usize>().map_err(|_| {
            format!("fit option `{name}`: expected non-negative integer, got `{value}`")
        })
    };
    let parse_bool = |name: &str| -> Result<bool, String> {
        match value.to_lowercase().as_str() {
            "true" | "t" | "yes" | "1" | "on" => Ok(true),
            "false" | "f" | "no" | "0" | "off" => Ok(false),
            _ => Err(format!(
                "fit option `{name}`: expected true/false, got `{value}`"
            )),
        }
    };
    let parse_u64_opt = |name: &str| -> Result<Option<u64>, String> {
        if value.is_empty()
            || value.eq_ignore_ascii_case("null")
            || value.eq_ignore_ascii_case("na")
        {
            Ok(None)
        } else {
            value.parse::<u64>().map(Some).map_err(|_| {
                format!("fit option `{name}`: expected non-negative integer, got `{value}`")
            })
        }
    };
    let parse_f64 = |name: &str| -> Result<f64, String> {
        value
            .parse::<f64>()
            .map_err(|_| format!("fit option `{name}`: expected number, got `{value}`"))
    };

    // Dispatch first, then record the key on success so we can later warn
    // when a key is set that the selected method does not consume. Malformed
    // values still return `Err` and don't get recorded.
    match key {
        "maxiter" => opts.outer_maxiter = parse_usize("maxiter")?,
        "inner_maxiter" => opts.inner_maxiter = parse_usize("inner_maxiter")?,
        "inner_tol" => opts.inner_tol = parse_f64("inner_tol")?,
        "ode_reltol" => {
            let v = parse_f64("ode_reltol")?;
            if v <= 0.0 || !v.is_finite() {
                return Err(format!(
                    "ode_reltol must be a positive finite value, got {v}"
                ));
            }
            opts.ode_reltol = v;
        }
        "ode_abstol" => {
            let v = parse_f64("ode_abstol")?;
            if v <= 0.0 || !v.is_finite() {
                return Err(format!(
                    "ode_abstol must be a positive finite value, got {v}"
                ));
            }
            opts.ode_abstol = v;
        }
        "ode_max_steps" => {
            let v = parse_usize("ode_max_steps")?;
            if v == 0 {
                return Err("ode_max_steps must be a positive integer".to_string());
            }
            opts.ode_max_steps = v;
        }
        "covariance" => opts.run_covariance_step = parse_bool("covariance")?,
        "covariance_fallback" => {
            opts.covariance_fallback = match value.to_lowercase().as_str() {
                "none" => crate::types::CovarianceFallback::None,
                "sir" => crate::types::CovarianceFallback::Sir,
                other => {
                    return Err(format!(
                        "fit option `covariance_fallback`: unknown value `{other}` — \
                         expected none/sir"
                    ));
                }
            };
        }
        "covariance_method" => {
            opts.covariance_method = match value.to_lowercase().as_str() {
                "r" | "hessian" => crate::types::CovarianceMethod::Hessian,
                "s" | "cross_product" => crate::types::CovarianceMethod::CrossProduct,
                "rsr" | "sandwich" => crate::types::CovarianceMethod::Sandwich,
                other => {
                    return Err(format!(
                        "fit option `covariance_method`: unknown value `{other}` — \
                         expected r/s/rsr"
                    ));
                }
            };
        }
        "fd_hessian_step" => {
            let v = parse_f64("fd_hessian_step")?;
            if v <= 0.0 || !v.is_finite() {
                return Err(format!(
                    "fd_hessian_step must be a positive finite value, got {}",
                    v
                ));
            }
            opts.fd_hessian_step = v;
        }
        "covariance_ofv_hessian" => {
            opts.covariance_ofv_hessian = parse_bool("covariance_ofv_hessian")?
        }
        "verbose" => opts.verbose = parse_bool("verbose")?,
        "optimizer" => {
            opts.optimizer = match value.to_lowercase().as_str() {
                "slsqp" => Optimizer::Slsqp,
                "lbfgs" | "nlopt_lbfgs" => Optimizer::NloptLbfgs,
                "mma" => Optimizer::Mma,
                "bfgs" => Optimizer::Bfgs,
                "bobyqa" => Optimizer::Bobyqa,
                "trust_region" | "newton_tr" => Optimizer::TrustRegion,
                other => {
                    return Err(format!(
                        "fit option `optimizer`: unknown value `{other}` — expected \
                         slsqp/lbfgs/nlopt_lbfgs/mma/bfgs/bobyqa/trust_region"
                    ));
                }
            };
        }
        "steihaug_max_iters" => opts.steihaug_max_iters = Some(parse_usize("steihaug_max_iters")?),
        "global_search" => opts.global_search = parse_bool("global_search")?,
        "global_maxeval" => opts.global_maxeval = parse_usize("global_maxeval")?,
        "n_exploration" => opts.saem_n_exploration = parse_usize("n_exploration")?,
        "n_convergence" => opts.saem_n_convergence = parse_usize("n_convergence")?,
        "n_mh_steps" => opts.saem_n_mh_steps = parse_usize("n_mh_steps")?,
        "n_leapfrog" | "saem_n_leapfrog" => opts.saem_n_leapfrog = parse_usize("n_leapfrog")?,
        "adapt_interval" => opts.saem_adapt_interval = parse_usize("adapt_interval")?,
        "omega_burnin" => opts.saem_omega_burnin = parse_usize("omega_burnin")?,
        "seed" | "saem_seed" => opts.saem_seed = parse_u64_opt("seed")?,
        "gn_lambda" => opts.gn_lambda = parse_f64("gn_lambda")?,
        "sir" => opts.sir = parse_bool("sir")?,
        "sir_samples" => opts.sir_samples = parse_usize("sir_samples")?,
        "sir_resamples" => opts.sir_resamples = parse_usize("sir_resamples")?,
        "sir_seed" => opts.sir_seed = parse_u64_opt("sir_seed")?,
        "sir_keep_samples" => opts.sir_keep_samples = parse_bool("sir_keep_samples")?,
        "sir_df" => {
            let v = parse_f64("sir_df")?;
            if v < 1.0 {
                return Err(format!("sir_df must be >= 1.0, got {v}"));
            }
            opts.sir_df = v;
        }
        "is_samples" => {
            let v = parse_usize("is_samples")?;
            if v < 2 {
                return Err(format!("is_samples must be >= 2, got {v}"));
            }
            opts.is_samples = v;
        }
        "is_proposal_df" => {
            let v = parse_f64("is_proposal_df")?;
            if v < 1.0 {
                return Err(format!("is_proposal_df must be >= 1.0, got {v}"));
            }
            opts.is_proposal_df = v;
        }
        "is_seed" => opts.is_seed = parse_u64_opt("is_seed")?,
        "is_low_ess_threshold" => {
            let v = parse_f64("is_low_ess_threshold")?;
            if !(0.0..=1.0).contains(&v) {
                return Err(format!(
                    "is_low_ess_threshold must be in [0.0, 1.0], got {v}"
                ));
            }
            opts.is_low_ess_threshold = v;
        }
        "impmap_iterations" => {
            let v = parse_usize("impmap_iterations")?;
            if v < 1 {
                return Err(format!("impmap_iterations must be >= 1, got {v}"));
            }
            opts.impmap_iterations = v;
        }
        "impmap_samples" => {
            let v = parse_usize("impmap_samples")?;
            if v < 2 {
                return Err(format!("impmap_samples must be >= 2, got {v}"));
            }
            opts.impmap_samples = v;
        }
        "impmap_proposal_df" => {
            // `normal` / `mvn` (or a very large df) select a multivariate-normal
            // proposal — NONMEM's IMPMAP default. A finite value gives Student-t.
            let tok = value.trim().trim_matches(|c| c == '"' || c == '\'');
            if tok.eq_ignore_ascii_case("normal") || tok.eq_ignore_ascii_case("mvn") {
                opts.impmap_proposal_df = f64::INFINITY;
            } else {
                let v = parse_f64("impmap_proposal_df")?;
                if v < 1.0 {
                    return Err(format!(
                        "impmap_proposal_df must be >= 1.0 or `normal`, got {v}"
                    ));
                }
                opts.impmap_proposal_df = v;
            }
        }
        "impmap_seed" => opts.impmap_seed = parse_u64_opt("impmap_seed")?,
        "impmap_averaging" => opts.impmap_averaging = parse_usize("impmap_averaging")?,
        "impmap_low_ess_threshold" => {
            let v = parse_f64("impmap_low_ess_threshold")?;
            if !(0.0..=1.0).contains(&v) {
                return Err(format!(
                    "impmap_low_ess_threshold must be in [0.0, 1.0], got {v}"
                ));
            }
            opts.impmap_low_ess_threshold = v;
        }
        "npde_nsim" => opts.npde_nsim = parse_usize("npde_nsim")?,
        "npde_seed" => opts.npde_seed = parse_u64_opt("npde_seed")?,
        "mu_referencing" => opts.mu_referencing = parse_bool("mu_referencing")?,
        "bloq_method" | "bloq" => {
            opts.bloq_method = match value.to_lowercase().as_str() {
                "m3" => BloqMethod::M3,
                "drop" | "none" | "ignore" => BloqMethod::Drop,
                other => {
                    return Err(format!(
                        "fit option `bloq_method`: unknown value `{other}` — expected 'm3' or 'drop'"
                    ));
                }
            };
        }
        "gradient" | "gradient_method" => {
            opts.gradient_method = match value.to_lowercase().as_str() {
                "auto" => GradientMethod::Auto,
                "ad" | "autodiff" => GradientMethod::Ad,
                "fd" | "finite" | "finite_difference" | "finite-difference" => GradientMethod::Fd,
                other => {
                    return Err(format!(
                        "fit option `gradient`: unknown value `{other}` — expected 'auto', 'ad', or 'fd'"
                    ));
                }
            };
        }
        "threads" => {
            if value.eq_ignore_ascii_case("auto") || value == "0" {
                opts.threads = None;
            } else {
                match value.parse::<usize>() {
                    Ok(n) if n > 0 => opts.threads = Some(n),
                    _ => {
                        return Err(format!(
                            "fit option `threads`: expected 'auto', 0, or a positive integer, got `{value}`"
                        ));
                    }
                }
            }
        }
        "n_starts" => match value.parse::<usize>() {
            Ok(n) if n >= 1 => opts.n_starts = n,
            _ => {
                return Err(format!(
                    "fit option `n_starts`: expected a positive integer, got `{value}`"
                ));
            }
        },
        "start_sigma" => opts.start_sigma = parse_f64("start_sigma")?,
        "multi_start_seed" => match value.parse::<u64>() {
            Ok(s) => opts.multi_start_seed = Some(s),
            _ => {
                return Err(format!(
                    "fit option `multi_start_seed`: expected a non-negative integer, got `{value}`"
                ));
            }
        },
        "iov_column" => {
            opts.iov_column = if value.is_empty()
                || value.eq_ignore_ascii_case("null")
                || value.eq_ignore_ascii_case("na")
                || value.eq_ignore_ascii_case("none")
            {
                None
            } else {
                Some(value.to_string())
            };
        }
        "optimizer_trace" => opts.optimizer_trace = parse_bool("optimizer_trace")?,
        "reconverge_gradient_interval" => {
            opts.reconverge_gradient_interval = parse_usize("reconverge_gradient_interval")?
        }
        "scale_params" => opts.scale_params = parse_bool("scale_params")?,
        "parameter_scaling" => {
            opts.parameter_scaling = match value.to_lowercase().as_str() {
                "auto" => ParameterScaling::Auto,
                "none" | "off" => ParameterScaling::None,
                "abs" => ParameterScaling::Abs,
                "rescale2" => ParameterScaling::Rescale2,
                other => {
                    return Err(format!(
                        "fit option `parameter_scaling`: unknown value `{other}` — \
                         expected auto/none/abs/rescale2"
                    ));
                }
            };
        }
        "max_unconverged_frac" => opts.max_unconverged_frac = parse_f64("max_unconverged_frac")?,
        "min_obs_for_convergence_check" => {
            opts.min_obs_for_convergence_check =
                parse_usize("min_obs_for_convergence_check")? as u32
        }
        "stagnation_guard" => opts.stagnation_guard = parse_bool("stagnation_guard")?,
        "inits_from_nca" => {
            use crate::suggest_start::NcaInit;
            opts.inits_from_nca = match value.to_lowercase().as_str() {
                "false" | "off" | "none" | "no" | "0" => None,
                "true" | "yes" | "1" | "nca_sweep" | "sweep" => Some(NcaInit::Sweep),
                "nca" => Some(NcaInit::Nca),
                "nca_ebe" | "ebe" => Some(NcaInit::Ebe),
                other => {
                    return Err(format!(
                        "fit option `inits_from_nca`: unknown value `{other}` — expected \
                         true/false or one of 'nca', 'nca_sweep', 'nca_ebe'"
                    ));
                }
            };
        }
        // ── [data_selection] keys ─────────────────────────────────────────────
        "ignore" => {
            // Validate the expression parses correctly; store verbatim for logging.
            crate::io::filter_expr::FilterClause::parse(value)
                .map_err(|e| format!("[data_selection] ignore: {e}"))?;
            push_unique_expr(&mut opts.ignore_exprs, value);
            // user_set_keys intentionally not pushed for selection keys — they
            // are not estimation options and should not trigger "unused key" warnings.
            return Ok(true);
        }
        "accept" => {
            crate::io::filter_expr::FilterClause::parse(value)
                .map_err(|e| format!("[data_selection] accept: {e}"))?;
            push_unique_expr(&mut opts.accept_exprs, value);
            return Ok(true);
        }
        "ignore_subjects" => {
            // Accept `[3, 17, 42]` or `3` (single value).
            let raw = value.trim().trim_start_matches('[').trim_end_matches(']');
            for part in raw.split(',') {
                let id = part.trim().to_string();
                if id.is_empty() {
                    continue;
                }
                // Validate it looks like a number or quoted string.
                let bare = id.trim_matches('"').trim_matches('\'');
                if bare.is_empty() {
                    return Err(format!(
                        "[data_selection] ignore_subjects: empty ID entry in '{value}'"
                    ));
                }
                push_unique_expr(&mut opts.ignore_subjects, bare);
            }
            return Ok(true);
        }
        _ => return Ok(false),
    }
    opts.user_set_keys.push(key.to_string());
    Ok(true)
}

/// Append `s` (trimmed) to `vec` only if not already present (case-sensitive).
/// Prevents duplicate conditions when the same expression appears in both the
/// `.ferx` file and the R call.
fn push_unique_expr(vec: &mut Vec<String>, s: &str) {
    let trimmed = s.trim().to_string();
    if !vec.iter().any(|e| e == &trimmed) {
        vec.push(trimmed);
    }
}

// ── [scaling] block parser ──────────────────────────────────────────────────

/// Parse a single scalar expression from `value` using the existing
/// recursive-descent expression parser. Used by `parse_scaling_block` for
/// `obs_scale = <expr>` and `y = <expr>` lines.
fn parse_scalar_expression(value: &str, ctx: ParseCtx<'_>) -> Result<Expression, String> {
    let toks = tokenize(value)?;
    let toks = strip_newlines_in_groups(toks);
    let pos = skip_newlines(&toks, 0);
    let (expr, p) = parse_add_sub(&toks, pos, ctx)?;
    let p = skip_newlines(&toks, p);
    if p != toks.len() {
        return Err(format!("trailing tokens after expression: `{}`", value));
    }
    Ok(expr)
}

/// Parse a `[scaling]` block key into `(base, optional_cmt)`.
///
/// Accepts:
///   - `obs_scale`           → ("obs_scale", None)
///   - `obs_scale[CMT=N]`    → ("obs_scale", Some(N))
///   - `y`                   → ("y", None)
///   - `y[CMT=N]`            → ("y", Some(N))
///
/// CMT must be a positive integer (1-based, matching the data file).
fn parse_scaling_key(key: &str) -> Result<(&str, Option<usize>), String> {
    match key.find('[') {
        None => Ok((key, None)),
        Some(open) => {
            let close = key
                .find(']')
                .ok_or_else(|| format!("[scaling]: malformed key `{}` (missing `]`)", key))?;
            if close < open {
                return Err(format!("[scaling]: malformed key `{}`", key));
            }
            let base = &key[..open];
            let inner = key[open + 1..close].trim();
            let trailing = key[close + 1..].trim();
            if !trailing.is_empty() {
                return Err(format!(
                    "[scaling]: unexpected text after `]` in key `{}`",
                    key
                ));
            }
            let cmt_parts: Vec<&str> = inner.splitn(2, '=').map(|s| s.trim()).collect();
            if cmt_parts.len() != 2 || cmt_parts[0] != "CMT" {
                return Err(format!(
                    "[scaling]: malformed key `{}` — expected `BASE[CMT=N]`",
                    key
                ));
            }
            let cmt: usize = cmt_parts[1].parse().map_err(|_| {
                format!(
                    "[scaling]: invalid CMT `{}` in key `{}` (expected positive integer)",
                    cmt_parts[1], key
                )
            })?;
            if cmt == 0 {
                return Err(format!(
                    "[scaling]: CMT must be ≥ 1 (1-based) in key `{}`",
                    key
                ));
            }
            Ok((base, Some(cmt)))
        }
    }
}

/// Build a `ScalingSpec` (None / ScalarScale / ExpressionScale) from one
/// `obs_scale[…] = value` line. Shared between the uniform and per-CMT paths.
fn build_obs_scale_spec(
    value: &str,
    theta_names: &[String],
    eta_names: &[String],
    indiv_var_names: &[String],
    pk_indices: &[usize],
) -> Result<ScalingSpec, String> {
    // Try scalar first (Form A). Otherwise parse as expression (Form B).
    if let Ok(k) = value.parse::<f64>() {
        // Divisor — strictly positive. A negative scale would flip every
        // prediction sign and bypass the upstream non-negativity clamps.
        if !(k > 0.0 && k.is_finite()) {
            return Err(format!(
                "[scaling]: obs_scale must be a strictly positive finite value, got `{}`",
                value
            ));
        }
        return Ok(ScalingSpec::ScalarScale(k));
    }
    let ctx = ParseCtx::new(theta_names, eta_names, indiv_var_names);
    let expr =
        parse_scalar_expression(value, ctx).map_err(|e| format!("[scaling] obs_scale: {}", e))?;
    // Pre-resolve indiv param name → PK slot so the closure can look up
    // `pk.values[slot]` for each `Expression::Variable(name)`. Mirrors
    // the analytical Form B path from Phase 1.5.
    let indiv_to_pk_slot: HashMap<String, usize> = indiv_var_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), pk_indices.get(i).copied().unwrap_or(i)))
        .collect();
    let scale_fn: ScaleFn = Box::new(
        move |theta: &[f64],
              eta: &[f64],
              covariates: &HashMap<String, f64>,
              pk: &PkParams|
              -> f64 {
            let mut vars: HashMap<String, f64> = HashMap::with_capacity(indiv_to_pk_slot.len());
            for (name, &slot) in &indiv_to_pk_slot {
                if slot < pk.values.len() {
                    vars.insert(name.clone(), pk.values[slot]);
                }
            }
            let empty_nn: Vec<Vec<f64>> = Vec::new();
            eval_expression(&expr, theta, eta, covariates, &vars, &empty_nn)
        },
    );
    Ok(ScalingSpec::ExpressionScale { scale_fn })
}

/// Build an `OdeOutputFn` from one `y[…] = value` line. Shared between
/// the uniform and per-CMT paths.
fn build_y_output_fn(
    value: &str,
    theta_names: &[String],
    eta_names: &[String],
    indiv_var_names: &[String],
    pk_indices: &[usize],
    state_names: &[String],
    kappa_names: &[String],
) -> Result<crate::ode::OdeOutputFn, String> {
    // Form C: expression may reference state names, individual params,
    // thetas, etas, and covariates. ParseCtx::new + theta/eta in scope.
    let mut defined: Vec<String> = state_names.to_vec();
    for n in indiv_var_names {
        if !defined.iter().any(|d| d == n) {
            defined.push(n.clone());
        }
    }
    let ctx = ParseCtx::new(theta_names, eta_names, &defined);
    let expr = parse_scalar_expression(value, ctx).map_err(|e| format!("[scaling] y: {}", e))?;

    // Reject KAPPA_* (IOV) references in a Form C ODE output expression: the
    // readout is evaluated once per observation with a single eta, so under IOV
    // it would silently see kappa = 0 (the per-occasion PK *dynamics* are still
    // correct — they flow through the per-event parameters — but a direct kappa
    // reference in the readout is not occasion-aware). The `[scaling]` eta scope
    // is BSV-only, so a kappa name parses as an unresolved identifier here; match
    // it by name. Fail fast rather than mislead. See issue #107; reference the
    // occasion-dependent structural parameter (e.g. CL) instead.
    if let Some(name) = expr_references_kappa(&expr, kappa_names) {
        return Err(format!(
            "[scaling] y: Form C output expressions cannot reference the IOV \
             parameter `{name}` — the ODE readout is evaluated per observation and \
             would see kappa = 0. Reference the occasion-dependent structural \
             parameter (e.g. CL) instead. See issue #107."
        ));
    }
    // Mirrors the ODE RHS port (PR #135): resolve the Form C expression to
    // the indexed evaluator at parse time so the closure can avoid per-call
    // `HashMap<String, f64>` allocation + string hashing on every observation.
    // The slow path was the dominant non-RHS HashMap cost in profiling — for a
    // model with N PK obs + M PD obs per integration, the readout closure was
    // doing O((N + M) · (n_states + n_indiv)) string-hash inserts.
    //
    // Layout: vars[0..n_states]                              = state values
    //         vars[n_states..n_states + n_indiv]             = indiv-param values
    //                                                          (read from pk_params_flat
    //                                                           via the same pk-slot plan
    //                                                           the old HashMap path used)
    //
    // `var_idx` uses last-writer-wins on collisions to preserve the prior
    // semantics: if a state and an indiv share a name, the old code did
    // `vars.insert(state)` then `vars.insert(indiv)` — indiv won. The
    // matching `HashMap::insert` (not `or_insert`) below keeps that.
    let n_states = state_names.len();
    let n_indiv = indiv_var_names.len();
    let mut var_idx: HashMap<String, usize> = HashMap::new();
    for (i, n) in state_names.iter().enumerate() {
        var_idx.insert(n.clone(), i);
    }
    for (i, n) in indiv_var_names.iter().enumerate() {
        var_idx.insert(n.clone(), n_states + i);
    }
    let n_vars_total = n_states + n_indiv;

    // Indiv-param → pk_params_flat slot plan (preserves the old HashMap
    // `indiv_idx` semantics).
    let indiv_to_pk: Vec<usize> = (0..n_indiv)
        .map(|i| pk_indices.get(i).copied().unwrap_or(i))
        .collect();

    // Covariate references in the expression are resolved against a small
    // `cov_idx` built from the names the expression actually references.
    // Most Form C readouts (including the experiment's Emax model) have
    // zero — when that's the case the per-call covariate-lookup loop is
    // a no-op.
    let mut cov_referenced: std::collections::HashSet<String> = std::collections::HashSet::new();
    collect_covariates(&expr, &mut cov_referenced);
    let cov_names: Vec<String> = {
        let mut v: Vec<String> = cov_referenced.into_iter().collect();
        v.sort();
        v
    };
    let cov_idx: HashMap<String, usize> = cov_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();

    // Rewrite Variable → VariableIdx and Covariate → CovariateIdx in place,
    // then compile to bytecode so the per-observation hot path is a tight
    // op-tag loop rather than a recursive `Box<Expression>` walk.
    let mut expr = expr;
    resolve_expr_indices(&mut expr, &var_idx, &cov_idx);
    let bc = compile_bytecode(&expr);

    // Per-thread scratch for the y readout vars + covariate slice + bytecode
    // f64 stack comes from the shared `FERX_SCRATCH` (see `FerxThreadScratch`).
    // The y-readout fields are kept separate from rhs_vars because the two are
    // sized for different layouts; the bc_stack is shared because the y readout
    // and ODE RHS never interleave on a single thread (readout runs
    // post-integration).
    let n_cov = cov_names.len();

    let out_fn: crate::ode::OdeOutputFn = Box::new(
        move |state: &[f64],
              pk_params_flat: &[f64],
              theta: &[f64],
              eta: &[f64],
              covariates: &HashMap<String, f64>|
              -> f64 {
            FERX_SCRATCH.with(|cell| {
                let mut s = cell.borrow_mut();
                let scratch = &mut *s;
                scratch.y_vars.clear();
                scratch.y_vars.resize(n_vars_total, 0.0);

                // State values from state[]; protect against a malformed
                // (state.len() < n_states) input the same way the old
                // HashMap path did (it skipped via `if i < state.len()`).
                let copy_n = n_states.min(state.len());
                scratch.y_vars[..copy_n].copy_from_slice(&state[..copy_n]);

                // Indiv params from pk_params_flat via the pre-computed slot
                // plan; OOB slots leave the var at 0.0.
                for (i, &pk_slot) in indiv_to_pk.iter().enumerate() {
                    if let (Some(dst), Some(&val)) = (
                        scratch.y_vars.get_mut(n_states + i),
                        pk_params_flat.get(pk_slot),
                    ) {
                        *dst = val;
                    }
                }

                scratch.y_cov.clear();
                if n_cov > 0 {
                    scratch.y_cov.resize(n_cov, 0.0);
                    for (i, name) in cov_names.iter().enumerate() {
                        if let Some(&v) = covariates.get(name) {
                            scratch.y_cov[i] = v;
                        }
                    }
                }

                let empty_nn: Vec<Vec<f64>> = Vec::new();
                eval_bytecode(
                    &bc,
                    theta,
                    eta,
                    &scratch.y_cov,
                    &scratch.y_vars,
                    &empty_nn,
                    &mut scratch.bc_stack,
                )
            })
        },
    );
    Ok(out_fn)
}

/// Parsed contents of a `[scaling]` block.
///
/// Returns `(ScalingSpec, Option<OdeReadout>)`:
/// - `ScalingSpec` populates `CompiledModel.scaling` and drives the
///   prediction-pipeline post-multiply. Variants: `None` /
///   `ScalarScale(K)` (Form A) / `ExpressionScale { scale_fn }` (Form B)
///   / `PerCmt(HashMap<usize, ScalingSpec>)` (Phase 2 multi-analyte
///   Forms A/B per CMT).
/// - `OdeReadout` populates `OdeSpec.readout` for Form C — the per-CMT
///   `y[CMT=N] = <expr>` and uniform `y = <expr>` syntaxes both go here
///   (`OdeReadout::PerCmt` vs `OdeReadout::Single` respectively), and
///   replace the default `OdeReadout::ObsCmt(idx)` state-index readout.
///
/// `is_ode = true` enables Form C and lets expressions reference state names.
/// `pk_indices` is parallel to `indiv_var_names`: `pk_indices[i]` is the
/// `PkParams.values` slot for the i-th individual parameter. Used by Form B
/// to look up indiv-param values (e.g. `V`) from a subject-static
/// `pk_param_fn` evaluation supplied to `scale_fn` at call time.
///
/// Validation:
/// - Unknown keys (anything other than `obs_scale[…]` / `y[…]`) → error.
/// - `y[…]` on an analytical model → error.
/// - Mixing uniform (`obs_scale = K`) with per-CMT (`obs_scale[CMT=N] = K`)
///   within the same group → error.
/// - Duplicate `[CMT=N]` keys → error.
#[allow(clippy::too_many_arguments)]
fn parse_scaling_block(
    lines: &[String],
    theta_names: &[String],
    eta_names: &[String],
    indiv_var_names: &[String],
    pk_indices: &[usize],
    state_names: &[String],
    is_ode: bool,
    kappa_names: &[String],
) -> Result<(ScalingSpec, Option<crate::ode::OdeReadout>), String> {
    // Accumulate uniform and per-CMT entries separately, then assemble at
    // the end. Mixing the two forms within the same group (obs_scale or y)
    // is rejected — keeps the semantic clean and matches NONMEM's
    // explicit-S1/S2 discipline.
    let mut obs_scale_uniform: Option<ScalingSpec> = None;
    let mut obs_scale_per_cmt: HashMap<usize, ScalingSpec> = HashMap::new();
    let mut y_uniform: Option<crate::ode::OdeOutputFn> = None;
    let mut y_per_cmt: HashMap<usize, crate::ode::OdeOutputFn> = HashMap::new();

    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Split on the first `=` OUTSIDE any `[...]` bracket. A naive
        // `splitn(2, '=')` would split inside `obs_scale[CMT=1]` and treat
        // `obs_scale[CMT` as the key. Walk the string tracking bracket
        // depth and split at the first depth-0 `=`.
        let mut depth: i32 = 0;
        let mut split_at: Option<usize> = None;
        for (i, ch) in trimmed.char_indices() {
            match ch {
                '[' => depth += 1,
                ']' => depth -= 1,
                '=' if depth == 0 => {
                    split_at = Some(i);
                    break;
                }
                _ => {}
            }
        }
        let split_at = match split_at {
            Some(p) => p,
            None => {
                return Err(format!(
                    "[scaling]: expected `key = value`, got: `{}`",
                    trimmed
                ));
            }
        };
        let key = trimmed[..split_at].trim();
        let value = trimmed[split_at + 1..].trim();
        let (base, cmt_opt) = parse_scaling_key(key)?;

        match base {
            "obs_scale" => {
                let spec = build_obs_scale_spec(
                    value,
                    theta_names,
                    eta_names,
                    indiv_var_names,
                    pk_indices,
                )?;
                match cmt_opt {
                    None => {
                        if obs_scale_uniform.is_some() {
                            return Err("[scaling]: duplicate `obs_scale` key".into());
                        }
                        if !obs_scale_per_cmt.is_empty() {
                            return Err("[scaling]: cannot mix uniform `obs_scale = ...` with \
                                 per-CMT `obs_scale[CMT=N] = ...` — choose one form."
                                .into());
                        }
                        obs_scale_uniform = Some(spec);
                    }
                    Some(cmt) => {
                        if obs_scale_uniform.is_some() {
                            return Err("[scaling]: cannot mix uniform `obs_scale = ...` with \
                                 per-CMT `obs_scale[CMT=N] = ...` — choose one form."
                                .into());
                        }
                        if obs_scale_per_cmt.contains_key(&cmt) {
                            return Err(format!(
                                "[scaling]: duplicate `obs_scale[CMT={}]` key",
                                cmt
                            ));
                        }
                        obs_scale_per_cmt.insert(cmt, spec);
                    }
                }
            }
            "y" => {
                if !is_ode {
                    return Err("[scaling]: `y = <expr>` (Form C) requires an ODE model — \
                         use `obs_scale = <expr>` for analytical PK"
                        .into());
                }
                let out_fn = build_y_output_fn(
                    value,
                    theta_names,
                    eta_names,
                    indiv_var_names,
                    pk_indices,
                    state_names,
                    kappa_names,
                )?;
                match cmt_opt {
                    None => {
                        if y_uniform.is_some() {
                            return Err("[scaling]: duplicate `y` key".into());
                        }
                        if !y_per_cmt.is_empty() {
                            return Err("[scaling]: cannot mix uniform `y = ...` with per-CMT \
                                 `y[CMT=N] = ...` — choose one form."
                                .into());
                        }
                        y_uniform = Some(out_fn);
                    }
                    Some(cmt) => {
                        if y_uniform.is_some() {
                            return Err("[scaling]: cannot mix uniform `y = ...` with per-CMT \
                                 `y[CMT=N] = ...` — choose one form."
                                .into());
                        }
                        if y_per_cmt.contains_key(&cmt) {
                            return Err(format!("[scaling]: duplicate `y[CMT={}]` key", cmt));
                        }
                        y_per_cmt.insert(cmt, out_fn);
                    }
                }
            }
            _ => {
                return Err(format!("[scaling]: unknown key `{}`", base));
            }
        }
    }

    let scaling = if let Some(s) = obs_scale_uniform {
        s
    } else if !obs_scale_per_cmt.is_empty() {
        ScalingSpec::PerCmt(obs_scale_per_cmt)
    } else {
        ScalingSpec::None
    };
    let readout = if let Some(f) = y_uniform {
        Some(crate::ode::OdeReadout::Single(f))
    } else if !y_per_cmt.is_empty() {
        Some(crate::ode::OdeReadout::PerCmt(y_per_cmt))
    } else {
        None
    };
    Ok((scaling, readout))
}

// ── ode_template desugaring + the analytical+ODE-only-absorption error rule ──

/// Built-in absorption input-rate functions with **no closed form**, which
/// therefore require an ODE disposition (the error rule, #322 Phase 0b). The
/// closed-form-capable functions (`first_order`, `zero_order`) are intentionally
/// excluded — they can ride on an analytical `pk` model.
///
/// This lists only the functions that are **actually implemented** as input
/// rates today (`transit`, #343). Each later absorption function adds its own
/// name here in the same PR that implements it — Phase 1 `igd` (#347), Phase 2
/// `weibull` — so the error rule never advertises a function the engine can't yet
/// run (which would send the user to `ode_template` for a dead end, Ron #363).
const ODE_ONLY_ABSORPTION_FNS: [&str; 1] = ["transit"];

/// Scan an `[odes]` block for an ODE-only absorption input-rate call, returning
/// the first such function name found. Drives the error rule that rejects an
/// analytical `pk` disposition combined with an ODE-only absorption term.
fn ode_only_absorption_fn_in_odes(odes: Option<&Vec<String>>) -> Option<&'static str> {
    let lines = odes?;
    for line in lines {
        for &f in ODE_ONLY_ABSORPTION_FNS.iter() {
            if find_word_call(line, f).is_some() {
                return Some(f);
            }
        }
    }
    None
}

/// Desugar an `ode_template NAME(...)` directive in `[structural_model]` into the
/// hand-written ODE form (#322 Phase 0b).
///
/// Generates the standard disposition ODE for the named model
/// (`crate::pk::ode_template::generate`), then rewrites the `structural_model`,
/// `odes`, and `scaling` blocks so the ordinary ODE pipeline takes over with no
/// special-casing. **Override semantics** (Ron, 2026-06-14): a `d/dt(X)` declared
/// by the user in `[odes]` *replaces* the generated equation for compartment `X`;
/// compartments the user leaves undeclared keep the generated RHS (no `+=`
/// append form). A no-op if `[structural_model]` has no `ode_template` line.
fn apply_ode_template(extracted: &mut ExtractedBlocks) -> Result<(), String> {
    // Cheap pre-check so the common (non-`ode_template`) model pays no regex
    // compilation: bail before any work unless `[structural_model]` actually
    // contains an `ode_template` line.
    let has_template = matches!(
        extracted.unnamed.get("structural_model"),
        Some(lines) if lines.iter().any(|l| l.starts_with("ode_template"))
    );
    if !has_template {
        return Ok(());
    }

    // Detect the `ode_template NAME(params)` line (and reject mixing it with a
    // `pk`/`ode(...)` disposition) in a scope that releases the immutable borrow
    // before the block rewrites below.
    let tmpl_re = Regex::new(r"^ode_template\s+(\w+)\s*\(([^)]*)\)\s*$").unwrap();
    let (model_name, params_str) = {
        let struct_lines = match extracted.unnamed.get("structural_model") {
            Some(l) => l,
            None => return Ok(()),
        };
        let mut found: Option<(String, String)> = None;
        let mut has_other_disposition = false;
        for line in struct_lines {
            if let Some(caps) = tmpl_re.captures(line) {
                if found.is_some() {
                    return Err(
                        "[structural_model]: more than one `ode_template` line; declare exactly one."
                            .to_string(),
                    );
                }
                found = Some((caps[1].to_string(), caps[2].to_string()));
            } else if line.starts_with("pk ")
                || line.starts_with("ode(")
                || line.starts_with("ode ")
            {
                has_other_disposition = true;
            }
        }
        match found {
            // `has_template` was true (some line starts with `ode_template`), so a
            // `None` here means that line did not match `ode_template NAME(...)` —
            // a malformed directive. Reject it explicitly rather than silently
            // falling through to a confusing "No PK model found" downstream (or,
            // with a `transit()` in [odes], the error rule telling the user to
            // "use ode_template" when they already are).
            None => {
                return Err(
                    "[structural_model]: malformed `ode_template` line — expected \
                     `ode_template NAME(role=VAR, ...)`, e.g. \
                     `ode_template two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)`."
                        .to_string(),
                );
            }
            Some(_) if has_other_disposition => {
                return Err(
                    "[structural_model]: `ode_template` cannot be combined with a `pk ...` or \
                     `ode(...)` disposition — choose one. `ode_template` already generates the \
                     full disposition ODE; add absorption / custom terms via override `d/dt(...)` \
                     lines in [odes]."
                        .to_string(),
                );
            }
            Some(x) => x,
        }
    };

    // Parse `role=VAR` pairs through the same strict helper as `pk NAME(...)`
    // (`parse_role_pairs`) so the two paths agree on malformed/duplicate handling.
    let params = parse_role_pairs(&params_str, &format!("ode_template {model_name}"))?;

    let generated = crate::pk::ode_template::generate(&model_name, &params)?;

    // `ode_template` produces a concentration readout via the injected
    // `obs_scale` (amount-based states / V), which the SDE/EKF path does not yet
    // support (it runs in the unscaled observation space). Without this guard the
    // injected `obs_scale` trips the generic SDE-scaling error downstream, which
    // blames a `[scaling]` block the user never wrote. Reject the combination
    // here with an accurate message instead.
    if extracted.unnamed.contains_key("diffusion") {
        return Err(
            "[structural_model]: `ode_template` is not supported with a `[diffusion]` (SDE/EKF) \
             model — it generates a concentration readout (`obs_scale`), which the EKF path does \
             not yet handle. Write the disposition by hand with `ode(...)` in amount space instead."
                .to_string(),
        );
    }

    // --- Override merge into [odes] -----------------------------------------
    // A **top-level** user `d/dt(X)` replaces the generated equation for X. Reuse
    // `diffeq_state` (whitespace-insensitive, matching `build_ode_spec`'s token
    // parser) so override detection can't drift from what the engine treats as a
    // state equation. Only top-level equations count: a `d/dt(X)` nested inside an
    // `if {...}` is a *conditional* tweak, so the generated unconditional equation
    // is kept (the two coexist — the conditional one wins when its branch fires,
    // the generated default applies otherwise). Suppressing the default for a
    // conditional override would silently leave X with no derivative outside the
    // branch. Brace depth is tracked across lines; an inline `if (..) { d/dt.. }`
    // line never starts with `d/dt`, so it is naturally excluded.
    let user_odes = extracted.unnamed.get("odes").cloned().unwrap_or_default();
    let mut overridden: Vec<String> = Vec::new();
    let mut brace_depth: i32 = 0;
    for line in &user_odes {
        if brace_depth == 0 {
            if let Some(cmt) = diffeq_state(line) {
                if !generated.states.iter().any(|s| s == &cmt) {
                    return Err(format!(
                        "[odes]: d/dt({cmt}) overrides a compartment not generated by \
                         `ode_template {model_name}` (generated states: {}). Override only a \
                         generated compartment, or use a hand-written `ode(...)` model instead.",
                        generated.states.join(", ")
                    ));
                }
                overridden.push(cmt);
            }
        }
        brace_depth += line.matches('{').count() as i32 - line.matches('}').count() as i32;
        if brace_depth < 0 {
            brace_depth = 0;
        }
    }
    // User lines first (they may define helper vars used by their overrides),
    // then the generated equations for every non-overridden compartment. The
    // duplicate-`d/dt` check in `build_ode_spec` is the backstop against a merge
    // bug ever letting two equations through for the same state.
    let mut merged = user_odes;
    for (state, line) in &generated.odes {
        if !overridden.iter().any(|s| s == state) {
            merged.push(line.clone());
        }
    }
    extracted.unnamed.insert("odes".to_string(), merged);

    // --- Rewrite [structural_model] to the synthesized ode(...) form --------
    extracted.unnamed.insert(
        "structural_model".to_string(),
        vec![format!(
            "ode(obs_cmt={}, states=[{}])",
            generated.obs_cmt,
            generated.states.join(", ")
        )],
    );

    // --- Supply obs_scale via [scaling] unless the user wrote their own -----
    // A user-provided [scaling] block wins (e.g. a Form C `y = <expr>` readout);
    // otherwise inject the generated central-volume scale.
    extracted
        .unnamed
        .entry("scaling".to_string())
        .or_insert_with(|| vec![format!("obs_scale = {}", generated.obs_scale)]);

    Ok(())
}

// ── [structural_model] ODE variant parser ───────────────────────────────────

fn parse_ode_structural(lines: &[String]) -> Result<(Vec<String>, Option<String>), String> {
    // ode(obs_cmt=central, states=[depot, central])   — classic form
    // ode(states=[depot, central])                    — Form C: requires [scaling] y = ...
    let with_obs =
        Regex::new(r"ode\(\s*obs_cmt\s*=\s*(\w+)\s*,\s*states\s*=\s*\[([^\]]+)\]\s*\)").unwrap();
    let without_obs = Regex::new(r"ode\(\s*states\s*=\s*\[([^\]]+)\]\s*\)").unwrap();
    for line in lines {
        if let Some(caps) = with_obs.captures(line) {
            let obs_cmt = caps[1].to_string();
            let states: Vec<String> = caps[2].split(',').map(|s| s.trim().to_string()).collect();
            return Ok((states, Some(obs_cmt)));
        }
        if let Some(caps) = without_obs.captures(line) {
            let states: Vec<String> = caps[1].split(',').map(|s| s.trim().to_string()).collect();
            return Ok((states, None));
        }
    }
    Err(
        "Could not parse ODE structural model. Expected: ode(obs_cmt=NAME, states=[...]) or \
         ode(states=[...]) with [scaling] y = <expr>"
            .to_string(),
    )
}

// ── [odes] block → OdeSpec ──────────────────────────────────────────────────

/// Parse a `[diffusion]` block into per-state initial variance values.
///
/// Expected syntax (one line per state):
///   `STATE_NAME ~ value`         — variance value, estimated
///   `STATE_NAME ~ value FIX`     — fixed variance
///
/// Returns `(diffusion_var, diffusion_names, diffusion_fixed)` where each vec
/// is aligned to `state_names` order (zero for states not mentioned).
/// States not listed default to 0 (no diffusion).
///
/// Also validates that all named states exist in `state_names`.
fn parse_diffusion_block(
    lines: &[String],
    state_names: &[String],
) -> Result<(Vec<f64>, Vec<Option<String>>, Vec<bool>), String> {
    let n = state_names.len();
    let mut variances = vec![0.0f64; n];
    let mut names: Vec<Option<String>> = vec![None; n];
    let mut fixed = vec![false; n];

    for line in lines {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let caps = DIFFUSION_LINE_RE.captures(line).ok_or_else(|| {
            format!(
                "[diffusion] invalid line (expected `STATE ~ value` or `STATE ~ value FIX`): `{}`",
                line
            )
        })?;
        let state = caps[1].to_string();
        let value: f64 = caps[2]
            .parse()
            .map_err(|_| format!("[diffusion] bad variance value in: `{}`", line))?;
        let is_fixed = caps.get(3).is_some();

        let idx = state_names
            .iter()
            .position(|s| s.eq_ignore_ascii_case(&state))
            .ok_or_else(|| {
                format!(
                    "[diffusion] state `{}` not defined in [odes] block. \
                     Known states: {}",
                    state,
                    state_names.join(", ")
                )
            })?;

        if value < 0.0 {
            return Err(format!(
                "[diffusion] variance for `{}` must be >= 0, got {}",
                state, value
            ));
        }
        variances[idx] = value;
        names[idx] = Some(format!("DIFF_{}", state.to_uppercase()));
        fixed[idx] = is_fixed;
    }

    Ok((variances, names, fixed))
}

// ── [covariate_nn NAME] block ────────────────────────────────────────────────
//
// Parses block bodies of the form:
//
//   [covariate_nn TYPICAL_PK]
//     inputs     = [WT, CRCL]
//     outputs    = [CL, V1, Q, V2, KA]
//     layers     = [16, 16]
//     activation = tanh           # hidden-layer activation
//     output     = softplus       # output-layer activation (optional, default identity)
//
// Produces a `NamedMlpMapper` plus a list of auto-generated weight-theta
// names. The names follow the convention `W_<NAME>_<l>_<i>_<j>` and
// `B_<NAME>_<l>_<i>` (1-indexed layers/units, all uppercase), matching the
// existing `TVCL` / `THETA_WT` theta-name style.
//
// Feature-gated behind `nn`. Without the feature the parser surfaces a
// clear error so users get told to rebuild with `--features nn`.

#[cfg(feature = "nn")]
#[derive(Debug, Clone)]
pub(crate) struct CovariateNnSpec {
    pub name: String,
    pub mapper: crate::nn::NamedMlpMapper,
    pub theta_names: Vec<String>,
    pub theta_inits: Vec<f64>,
}

#[cfg(feature = "nn")]
fn parse_covariate_nn_block(name: &str, lines: &[String]) -> Result<CovariateNnSpec, String> {
    use crate::nn::{Activation, CovariateMapper, MlpMapper, NamedMlpMapper};

    // Collect simple `key = value` pairs. Each value is either a list
    // `[a, b, c]` or a bare identifier / integer.
    let mut fields: HashMap<String, String> = HashMap::new();
    for line in lines {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(format!(
                "[covariate_nn {}] expected `key = value`, got: `{}`",
                name, line
            ));
        };
        let key = key.trim().to_ascii_lowercase();
        let value = value.trim().to_string();
        if fields.insert(key.clone(), value).is_some() {
            return Err(format!("[covariate_nn {}] duplicate key `{}`", name, key));
        }
    }

    let take_list = |field: &str| -> Result<Vec<String>, String> {
        let raw = fields
            .get(field)
            .ok_or_else(|| format!("[covariate_nn {}] missing required `{}`", name, field))?;
        let inner = raw
            .strip_prefix('[')
            .and_then(|s| s.strip_suffix(']'))
            .ok_or_else(|| {
                format!(
                    "[covariate_nn {}] `{}` must be a list like `[a, b, c]`, got `{}`",
                    name, field, raw
                )
            })?;
        Ok(inner
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect())
    };

    let inputs = take_list("inputs")?;
    let outputs = take_list("outputs")?;
    let layer_strs = take_list("layers")?;
    if inputs.is_empty() {
        return Err(format!("[covariate_nn {}] `inputs` is empty", name));
    }
    if outputs.is_empty() {
        return Err(format!("[covariate_nn {}] `outputs` is empty", name));
    }

    let hidden: Vec<usize> = layer_strs
        .iter()
        .map(|s| {
            s.parse::<usize>().map_err(|_| {
                format!(
                    "[covariate_nn {}] `layers` entries must be positive integers, got `{}`",
                    name, s
                )
            })
        })
        .collect::<Result<_, _>>()?;
    if hidden.is_empty() {
        return Err(format!(
            "[covariate_nn {}] `layers` must list at least one hidden width",
            name
        ));
    }
    if hidden.iter().any(|&h| h == 0) {
        return Err(format!(
            "[covariate_nn {}] hidden-layer width must be > 0",
            name
        ));
    }

    let parse_activation = |raw: &str, ctx: &str| -> Result<Activation, String> {
        match raw.to_ascii_lowercase().as_str() {
            "identity" | "linear" => Ok(Activation::Identity),
            "relu" => Ok(Activation::Relu),
            "softplus" => Ok(Activation::Softplus),
            "tanh" => Ok(Activation::Tanh),
            "sigmoid" => Ok(Activation::Sigmoid),
            "exp" => Ok(Activation::Exp),
            other => Err(format!(
                "[covariate_nn {}] unknown {} activation `{}` (expected one of: \
                 identity, relu, softplus, tanh, sigmoid, exp)",
                name, ctx, other
            )),
        }
    };

    let hidden_act_raw = fields
        .get("activation")
        .ok_or_else(|| format!("[covariate_nn {}] missing required `activation`", name))?;
    let hidden_activation = parse_activation(hidden_act_raw, "hidden")?;

    let output_activation = match fields.get("output") {
        Some(s) => parse_activation(s, "output")?,
        None => Activation::Identity,
    };

    // Reject any unknown keys so typos don't silently pass.
    const KNOWN: &[&str] = &["inputs", "outputs", "layers", "activation", "output"];
    for k in fields.keys() {
        if !KNOWN.contains(&k.as_str()) {
            return Err(format!(
                "[covariate_nn {}] unknown key `{}` (known: {})",
                name,
                k,
                KNOWN.join(", ")
            ));
        }
    }

    let mut layer_sizes = Vec::with_capacity(hidden.len() + 2);
    layer_sizes.push(inputs.len());
    layer_sizes.extend(hidden.iter().copied());
    layer_sizes.push(outputs.len());

    let mlp = MlpMapper::new(layer_sizes.clone(), hidden_activation, output_activation)
        .map_err(|e| format!("[covariate_nn {}] {}", name, e))?;
    let mapper = NamedMlpMapper::new(mlp, inputs, outputs)
        .map_err(|e| format!("[covariate_nn {}] {}", name, e))?;

    // Auto-generate weight-theta names + Glorot-style deterministic inits.
    // Names are uppercase to match the codebase convention (TVCL, THETA_WT,
    // DIFF_CENTRAL, …). Inits use a fixed PRNG (xorshift seeded by name) so
    // builds are reproducible without pulling `rand` into the parser.
    let upper = name.to_ascii_uppercase();
    let mut theta_names = Vec::new();
    let mut theta_inits = Vec::new();
    let mut state: u64 = {
        // Tiny string hash → seed. Deterministic across runs.
        let mut h: u64 = 0xcbf29ce484222325;
        for b in upper.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h | 1
    };
    let mut next_unit = || -> f64 {
        // xorshift64 → uniform(-0.5, 0.5)
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as f64 / u64::MAX as f64) - 0.5
    };
    for l in 1..layer_sizes.len() {
        let n_l = layer_sizes[l];
        let n_lm1 = layer_sizes[l - 1];
        // Glorot/Xavier scale: weights ~ U(-r, r), r = sqrt(6 / (fan_in + fan_out)).
        let r = (6.0 / (n_lm1 + n_l) as f64).sqrt();
        for i in 1..=n_l {
            for j in 1..=n_lm1 {
                theta_names.push(format!("W_{}_{}_{}_{}", upper, l, i, j));
                theta_inits.push(2.0 * r * next_unit());
            }
        }
        for i in 1..=n_l {
            theta_names.push(format!("B_{}_{}_{}", upper, l, i));
            // Biases initialised to 0 — standard for tanh/sigmoid; for ReLU
            // you'd want a small positive bias, but this module's intended
            // hidden activation for [covariate_nn] is `tanh` (softplus on the
            // output head), so 0 is fine.
            theta_inits.push(0.0);
        }
    }
    debug_assert_eq!(theta_names.len(), mapper.n_weights());

    Ok(CovariateNnSpec {
        name: name.to_string(),
        mapper,
        theta_names,
        theta_inits,
    })
}

/// Assign each ODE individual parameter (in declaration order) a slot in the
/// fixed `PkParams.values` array, returned parallel to `names`.
///
/// Parameters whose name maps to a canonical PK slot (`CL`, `V`, `KA`, `F`,
/// `LAGTIME`, …) take that slot, so the ODE engine — which reads bioavailability
/// from `PK_IDX_F` and absorption lag from `PK_IDX_LAGTIME` — finds them where
/// it expects. Every other ("structural") parameter takes the lowest free slot
/// that is neither already claimed by a canonical parameter nor one of the two
/// engine-reserved slots (`PK_IDX_F`, `PK_IDX_LAGTIME`). Reserving those two
/// keeps an *undeclared* F at its 1.0 default (and lagtime at 0) instead of
/// letting an unrelated structural parameter alias the slot and be silently
/// read as bioavailability — the issue #122 regression.
///
/// The RHS evaluator (`build_ode_spec`) and the parameter writer
/// (`build_pk_param_fn`) both consult this map, so a name binds to the same
/// slot whether it is being written or read.
fn ode_param_slots(names: &[String]) -> Result<Vec<usize>, String> {
    let reserved = RESERVED_PK_SLOTS;
    let mut taken = [false; MAX_PK_PARAMS];
    let mut slots = vec![usize::MAX; names.len()];

    // Pass 1: canonical names → their fixed PK slot.
    for (i, name) in names.iter().enumerate() {
        if let Some(s) = PkParams::name_to_index(&name.to_lowercase()) {
            if taken[s] {
                return Err(format!(
                    "ODE model declares two individual parameters that map to the same \
                     PK slot (at `{name}`); rename one."
                ));
            }
            taken[s] = true;
            slots[i] = s;
        }
    }

    // Pass 2: structural names → lowest free, non-reserved slot.
    for (i, name) in names.iter().enumerate() {
        if slots[i] != usize::MAX {
            continue;
        }
        match (0..MAX_PK_PARAMS).find(|s| !taken[*s] && !reserved.contains(s)) {
            Some(s) => {
                taken[s] = true;
                slots[i] = s;
            }
            None => {
                return Err(format!(
                    "ODE model has too many individual parameters for the {MAX_PK_PARAMS}-slot \
                     PK layout (at `{name}`). Slots {reserved:?} are reserved for F/lagtime, \
                     leaving room for {} other parameters.",
                    MAX_PK_PARAMS - reserved.len()
                ));
            }
        }
    }

    Ok(slots)
}

/// Find `name(` at a word boundary in `s` (ASCII), returning the index of `name`.
fn find_word_call(s: &str, name: &str) -> Option<usize> {
    let pat = format!("{name}(");
    let b = s.as_bytes();
    let mut from = 0;
    while let Some(rel) = s[from..].find(&pat) {
        let i = from + rel;
        let before_ok = i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_');
        if before_ok {
            return Some(i);
        }
        from = i + 1;
    }
    None
}

/// Given the index of a `(` in `s` (ASCII), return the inner text and the index
/// just past the matching `)`. `None` if unbalanced.
fn balanced_parens(s: &str, open: usize) -> Option<(String, usize)> {
    let b = s.as_bytes();
    if b.get(open) != Some(&b'(') {
        return None;
    }
    let mut depth = 0i32;
    for i in open..b.len() {
        match b[i] {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some((s[open + 1..i].to_string(), i + 1));
                }
            }
            _ => {}
        }
    }
    None
}

/// State name `X` if `line` is a `d/dt(X) = …` equation, else `None`.
///
/// Whitespace-insensitive, matching the **token-based** `[odes]` statement
/// parser ([`parse_statement`], which recognises `d/dt(NAME)` as the token
/// sequence `Ident("d") Slash Ident("dt") LParen Ident RParen` regardless of
/// spacing — so `d/dt (central)` and `d / dt(central)` are valid state
/// equations there). A literal `strip_prefix("d/dt(")` would diverge from that
/// and, in the `ode_template` override merge, miss a spaced override — leaving
/// both the generated and the user equation in place for a misleading
/// "duplicate d/dt" error. We collapse interior whitespace before matching so
/// the two definitions can never drift apart.
fn diffeq_state(line: &str) -> Option<String> {
    let compact: String = line.chars().filter(|c| !c.is_whitespace()).collect();
    let rest = compact.strip_prefix("d/dt(")?;
    let close = rest.find(')')?;
    Some(rest[..close].to_string())
}

/// True if the call spanning `[start, end)` in `line` (ASCII) is **not** a bare,
/// positively-signed, top-level additive term — i.e. it is scaled (`*` `/` `^`),
/// negated (a leading `-`), or grouped (`(` immediately before / `)` immediately
/// after, which can hide an outer scale such as `(transit(...))/V`). The
/// input-rate forcing is always injected as `+R_in`, unscaled, so only a bare
/// `+ transit(...)` term is faithful; any other context would silently drop the
/// sign or scale. Surrounding spaces are skipped; the faithful preceding chars
/// are `=` (RHS start) and `+`, the faithful following chars are end-of-line,
/// `+`, and `-` (each starts a new additive term).
fn call_is_scaled_or_signed(line: &str, start: usize, end: usize) -> bool {
    let b = line.as_bytes();
    let mut i = start;
    while i > 0 && b[i - 1] == b' ' {
        i -= 1;
    }
    let before_bad = i > 0 && matches!(b[i - 1], b'*' | b'/' | b'^' | b'-' | b'(');
    let mut j = end;
    while j < b.len() && b[j] == b' ' {
        j += 1;
    }
    let after_bad = j < b.len() && matches!(b[j], b'*' | b'/' | b'^' | b')');
    before_bad || after_bad
}

/// Split a string on top-level commas (commas outside nested parentheses).
fn split_args_on_commas(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut last = 0;
    for (i, &c) in s.as_bytes().iter().enumerate() {
        match c {
            b'(' => depth += 1,
            b')' => depth -= 1,
            b',' if depth == 0 => {
                parts.push(&s[last..i]);
                last = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&s[last..]);
    parts
}

/// Extract built-in absorption input-rate calls (currently `transit(...)`,
/// design A) from the `[odes]` RHS lines. For each
/// `d/dt(STATE) = … transit(n=P, mtt=Q) …`, records an [`InputRateForcing`]
/// (`cmt` ← STATE index; `n`/`mtt` resolved to individual-parameter slots) and
/// rewrites the call to `0` so the remaining RHS parses as a normal expression.
/// Returns the cleaned lines and the collected forcing terms.
///
/// Validation (negative-tested): `transit(...)` must be the input rate of a
/// top-level `d/dt(...)` equation; args are exactly `n` and `mtt`, each a
/// declared individual parameter; a scaled call (`FR*transit(...)`) and more than
/// one call per equation are rejected (parallel/biphasic comes with later models).
fn extract_input_rate_terms(
    rhs_lines: &[String],
    state_names: &[String],
    indiv_param_names: &[String],
    indiv_param_slots: &[usize],
) -> Result<(Vec<String>, Vec<crate::pk::absorption::InputRateForcing>), String> {
    use crate::pk::absorption::{InputRateForcing, InputRateKind};
    let resolve_slot = |val: &str, arg: &str| -> Result<usize, String> {
        indiv_param_names
            .iter()
            .position(|p| p == val)
            .map(|i| indiv_param_slots[i])
            .ok_or_else(|| {
                format!(
                    "[odes]: transit({arg}={val}): `{val}` is not a declared individual parameter"
                )
            })
    };

    let mut forcings = Vec::new();
    let mut cleaned = Vec::with_capacity(rhs_lines.len());
    for raw in rhs_lines {
        let Some(start) = find_word_call(raw, "transit") else {
            cleaned.push(raw.clone());
            continue;
        };
        let state = diffeq_state(raw).ok_or_else(|| {
            format!(
                "[odes]: transit(...) may only be the input rate of a `d/dt(...)` equation — \
                 found it in `{}`",
                raw.trim()
            )
        })?;
        let cmt = state_names
            .iter()
            .position(|s| s == &state)
            .ok_or_else(|| format!("[odes]: d/dt({state}): undeclared state"))?;

        let open = start + "transit".len();
        let (inner, end) = balanced_parens(raw, open).ok_or_else(|| {
            format!(
                "[odes]: transit(...): unbalanced parentheses in `{}`",
                raw.trim()
            )
        })?;
        if call_is_scaled_or_signed(raw, start, end) {
            return Err(
                "[odes]: transit(...) must be a standalone, positively-signed additive input \
                 rate — it cannot be scaled (`* / ^`), negated (a leading `-`), or wrapped in \
                 parentheses (e.g. `FR*transit(...)`, `-transit(...)`, `(transit(...))/V`), \
                 since these silently drop the sign/scale. Write it as a bare `+ transit(...)` \
                 term."
                    .to_string(),
            );
        }

        let mut n_slot = None;
        let mut mtt_slot = None;
        for part in split_args_on_commas(&inner) {
            let (name, val) = part.split_once('=').ok_or_else(|| {
                format!(
                    "[odes]: transit(...) arguments must be `name=parameter`, got `{}`",
                    part.trim()
                )
            })?;
            let (name, val) = (name.trim(), val.trim());
            match name {
                "n" => n_slot = Some(resolve_slot(val, "n")?),
                "mtt" => mtt_slot = Some(resolve_slot(val, "mtt")?),
                other => {
                    return Err(format!(
                        "[odes]: transit(...) has no argument `{other}` (expected `n` and `mtt`)"
                    ))
                }
            }
        }
        let n_slot = n_slot.ok_or("[odes]: transit(...) missing required argument `n`")?;
        let mtt_slot = mtt_slot.ok_or("[odes]: transit(...) missing required argument `mtt`")?;

        forcings.push(InputRateForcing {
            cmt,
            kind: InputRateKind::Transit,
            arg_slots: vec![n_slot, mtt_slot],
        });

        let new_line = format!("{}0{}", &raw[..start], &raw[end..]);
        if find_word_call(&new_line, "transit").is_some() {
            return Err("[odes]: at most one transit(...) per d/dt equation (Phase 0)".to_string());
        }
        cleaned.push(new_line);
    }
    Ok((cleaned, forcings))
}

fn build_ode_spec(
    lines: &[String],
    state_names: &[String],
    obs_cmt_name: Option<&str>,
    indiv_param_names: &[String],
    indiv_param_slots: &[usize],
) -> Result<crate::ode::OdeSpec, String> {
    let n_states = state_names.len();
    // When the user omitted `obs_cmt=` in `ode(states=[...])` they must
    // supply a `[scaling] y = <expr>` (Form C). At this point the
    // scaling block hasn't been parsed yet, so we encode "no obs_cmt
    // decided yet" as the sentinel `usize::MAX` on the ObsCmt readout.
    // `parse_full_model` then either replaces the readout with Form C
    // (Single/PerCmt) or errors when both the sentinel survives and no
    // override arrives.
    const NEEDS_FORM_C: usize = usize::MAX;
    let obs_cmt_idx = match obs_cmt_name {
        Some(name) => state_names.iter().position(|s| s == name).ok_or_else(|| {
            format!(
                "Observable compartment '{}' not in states {:?}",
                name, state_names
            )
        })?,
        None => NEEDS_FORM_C,
    };

    // Pull `init(state) = <expr>` directives out of the block first. The
    // statement parser only understands d/dt / assignment / if, so init lines
    // must be separated before parsing the RHS. Each init expression may
    // reference individual parameters and (defensively) state names — states
    // are bound to 0.0 at init time. Build the init_fn closure that seeds the
    // integrator from these expressions.
    let init_ctx_defined: Vec<String> = state_names
        .iter()
        .cloned()
        .chain(indiv_param_names.iter().cloned())
        .collect();
    // Names an init expression can resolve, mirroring the exact keys the
    // `init_fn` closure seeds below: states (original + lowercase, bound to 0 at
    // init time) and individual parameters (original + upper + lowercase). A
    // `Variable` outside this set (the MACHEPS builtin aside — handled in the
    // loop below) silently reads 0.0 via `eval_expression` (issue #314), so
    // reject it at parse time.
    let mut init_defined: HashMap<String, usize> = HashMap::new();
    for n in state_names {
        init_defined.insert(n.clone(), 0);
        init_defined.insert(n.to_lowercase(), 0);
    }
    for n in indiv_param_names {
        init_defined.insert(n.clone(), 0);
        init_defined.insert(n.to_uppercase(), 0);
        init_defined.insert(n.to_lowercase(), 0);
    }
    let mut init_specs: Vec<(usize, Expression)> = Vec::new();
    let mut seen_init: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut rhs_lines: Vec<String> = Vec::with_capacity(lines.len());
    for raw in lines {
        if let Some((name, expr_str)) = parse_init_line(raw) {
            let idx = state_names.iter().position(|s| s == &name).ok_or_else(|| {
                format!(
                    "[odes]: init({}) references unknown state. Declared states: {:?}",
                    name, state_names
                )
            })?;
            if !seen_init.insert(name.clone()) {
                return Err(format!(
                    "[odes]: duplicate init({}) — initial condition defined more than once",
                    name
                ));
            }
            let ctx = ParseCtx::ode(&init_ctx_defined);
            let expr = parse_scalar_expression(&expr_str, ctx)
                .map_err(|e| format!("[odes] init({}): {}", name, e))?;
            let mut undef: std::collections::HashSet<String> = std::collections::HashSet::new();
            collect_undefined_vars(&expr, &init_defined, &mut undef);
            // MACHEPS is a builtin constant that `eval_expression` resolves to
            // f64::EPSILON case-insensitively, so accept any casing here (the
            // exact-key `init_defined` carries only states/params, not MACHEPS).
            undef.retain(|n| !n.eq_ignore_ascii_case("MACHEPS"));
            if !undef.is_empty() {
                let mut names: Vec<String> = undef.into_iter().collect();
                names.sort();
                let mut defined = init_ctx_defined.clone();
                defined.sort();
                return Err(format!(
                    "[odes] init({}): references undefined name(s): {}. An init \
                     expression may only reference declared states (0 at init \
                     time), individual parameters, or the MACHEPS constant \
                     (defined: {}).",
                    name,
                    names.join(", "),
                    defined.join(", "),
                ));
            }
            init_specs.push((idx, expr));
        } else {
            rhs_lines.push(raw.clone());
        }
    }

    // Design A: pull built-in input-rate calls (transit(...)) out of each d/dt RHS
    // into forcing terms before expression parsing, so they never enter the
    // expression AST / bytecode / symbolic-AD core (each call is rewritten to `0`).
    let (rhs_lines, input_rate) = extract_input_rate_terms(
        &rhs_lines,
        state_names,
        indiv_param_names,
        indiv_param_slots,
    )?;

    // For ODE RHS expressions, states + individual params get injected into the
    // `vars` map at eval time, so every bare identifier should resolve to a
    // Variable (not a Covariate). ParseCtx::ode() flips the fallback accordingly.
    // Local intermediate vars assigned within the [odes] block (e.g. inside an
    // if-body) are also collected from a pre-pass below so they parse as
    // Variable too.
    let block_text = rhs_lines.join("\n");
    let pre_defined: Vec<String> = state_names
        .iter()
        .cloned()
        .chain(indiv_param_names.iter().cloned())
        .collect();
    let pre_ctx = ParseCtx::ode(&pre_defined);
    let pre_stmts = parse_block_statements(&block_text, pre_ctx, StatementMode::Ode)?;
    let local_vars = assigned_vars_in_order(&pre_stmts);

    let mut ode_defined = pre_defined.clone();
    for v in &local_vars {
        if !ode_defined.iter().any(|n| n == v) {
            ode_defined.push(v.clone());
        }
    }
    let ode_ctx = ParseCtx::ode(&ode_defined);
    let stmts = parse_block_statements(&block_text, ode_ctx, StatementMode::Ode)?;

    // Validate that every declared state has a d/dt assignment somewhere in
    // the block (top level OR inside an if-body). Whether the if-body actually
    // fires at run time is the user's problem — our job is to verify that the
    // block at least mentions each state.
    //
    // Also reject duplicate d/dt assignments within the same statement scope
    // (e.g. two `d/dt(central)` at top level or in the same branch body),
    // since the second write would silently win at runtime. Assignments to the
    // same state in *different* branches of an if/else are allowed.
    let mut diffeq_states: std::collections::HashSet<String> = std::collections::HashSet::new();
    fn collect_diffeqs(stmts: &[Statement], out: &mut std::collections::HashSet<String>) {
        for s in stmts {
            match s {
                Statement::DiffEq(name, _) => {
                    out.insert(name.clone());
                }
                Statement::If {
                    branches,
                    else_body,
                } => {
                    for (_, body) in branches {
                        collect_diffeqs(body, out);
                    }
                    if let Some(eb) = else_body {
                        collect_diffeqs(eb, out);
                    }
                }
                Statement::Assign(_, _)
                | Statement::AssignIdx(_, _)
                | Statement::DiffEqIdx(_, _)
                | Statement::AssignBc(_, _)
                | Statement::DiffEqBc(_, _) => {}
            }
        }
    }
    fn check_duplicate_diffeqs(stmts: &[Statement]) -> Result<(), String> {
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for s in stmts {
            match s {
                Statement::DiffEq(name, _) => {
                    if !seen.insert(name.clone()) {
                        return Err(format!(
                            "[odes]: duplicate d/dt({}) — state equation defined more than once in the same scope",
                            name
                        ));
                    }
                }
                Statement::If {
                    branches,
                    else_body,
                } => {
                    for (_, body) in branches {
                        check_duplicate_diffeqs(body)?;
                    }
                    if let Some(eb) = else_body {
                        check_duplicate_diffeqs(eb)?;
                    }
                }
                Statement::Assign(_, _)
                | Statement::AssignIdx(_, _)
                | Statement::DiffEqIdx(_, _)
                | Statement::AssignBc(_, _)
                | Statement::DiffEqBc(_, _) => {}
            }
        }
        Ok(())
    }
    collect_diffeqs(&stmts, &mut diffeq_states);
    check_duplicate_diffeqs(&stmts)?;
    for s in state_names {
        if !diffeq_states.contains(s) {
            return Err(format!("[odes]: missing d/dt({}) for declared state", s));
        }
    }

    let state_names_owned = state_names.to_vec();
    let indiv_names_owned = indiv_param_names.to_vec();
    // PkParams slot for each individual parameter (parallel to
    // `indiv_names_owned`). `pk_param_fn` writes each parameter's value here, so
    // the RHS must read it back from the same slot rather than from the
    // declaration position. See `ode_param_slots`.
    let indiv_slots_owned: Vec<usize> = indiv_param_slots.to_vec();
    let mut stmts_owned = stmts;
    let state_index: HashMap<String, usize> = state_names_owned
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();

    // ─── Build the flat var-slot layout once, at parse time ──────────────────
    //
    // The hot-path RHS used to allocate a fresh `HashMap<String, f64>` per call
    // and seed it with state names + individual-parameter names (in 2-3 case
    // variants each), then walk the AST doing string-keyed HashMap lookups
    // for every `Variable(name)` reference. For an ODE solve with N RK45
    // steps, that's ~6·N hash-map allocations per integration; for a typical
    // FOCEI fit on the Emax PK/PD model in the experiment, billions of
    // hash-map ops across the whole run.
    //
    // Switch to the indexed AST evaluator (`eval_statements_indexed`) that
    // the existing `pk_param_fn` already uses. Layout:
    //   vars[0..n_states]                                  → state values (read from u)
    //   vars[n_states..n_states+n_indiv]                   → indiv params  (read from params via indiv_slots)
    //   vars[n_states+n_indiv..n_states+n_indiv+n_inter]   → intermediates written by `Assign` in the ODE block
    //
    // The lookup map carries case-insensitive aliases (lower/upper/original)
    // to match the existing closure's behaviour. Name-collision (case-insensitive)
    // between any two of state names, indiv-param names, and intermediates is
    // rejected at parse time — without that check, the eager alias insertion
    // would silently route reads of one name through another's slot.
    let state_count = state_names_owned.len();
    let indiv_count = indiv_names_owned.len();

    // Reuse the existing top-level walker that already collects `Assign` LHS
    // names with the same dedup-and-recurse-through-If semantics we need.
    let intermediates: Vec<String> = assigned_vars_in_order(&stmts_owned);

    // Reject covariate references in ODE RHS expressions. Both the old
    // HashMap evaluator and the new indexed one silently resolve these to
    // 0.0 (empty covariate map / empty cov_idx) — which means models like
    // `d/dt(central) = -CL/V * central * (WT/70)^0.75` parse fine and fit
    // with the WT term collapsed to 0, producing silently-wrong dynamics.
    // Surface that as a parse-time error so users notice; see issue tracker
    // for the planned proper support.
    let mut ode_covariates: std::collections::HashSet<String> = std::collections::HashSet::new();
    collect_covariates_in_stmts(&stmts_owned, &mut ode_covariates);
    if !ode_covariates.is_empty() {
        let mut covs: Vec<String> = ode_covariates.into_iter().collect();
        covs.sort();
        return Err(format!(
            "[odes]: covariate reference(s) in ODE RHS not supported: {}. \
             Pre-compute the covariate-dependent term in [individual_parameters] \
             and reference that variable in the ODE block instead.",
            covs.join(", ")
        ));
    }

    // Reject name collisions (case-insensitive) across the three name spaces.
    // Without this, the eager alias insertion below would silently route reads
    // of one identifier through another's slot — pathological but real, and the
    // failure mode is mid-RHS-call slot-clobbering with no warning.
    let mut seen_lower: HashMap<String, &'static str> = HashMap::new();
    let mut check_collision = |name: &str, kind: &'static str| -> Result<(), String> {
        let lower = name.to_lowercase();
        if let Some(prev) = seen_lower.insert(lower, kind) {
            return Err(format!(
                "[odes]: name `{name}` collides with a previously-declared {prev} \
                 (case-insensitive); state, individual-parameter, and ODE-block \
                 intermediate names must all be distinct."
            ));
        }
        Ok(())
    };
    for n in &state_names_owned {
        check_collision(n, "state")?;
    }
    for n in &indiv_names_owned {
        check_collision(n, "individual parameter")?;
    }
    for n in &intermediates {
        check_collision(n, "ODE-block intermediate")?;
    }

    // Layout is now collision-free; build the slot map. `or_insert` is fine
    // because the only repeats are within one name's own case-variant set
    // (which we intentionally collapse to a single slot).
    let mut var_idx: HashMap<String, usize> = HashMap::new();
    let add_aliases = |map: &mut HashMap<String, usize>, name: &str, slot: usize| {
        map.entry(name.to_string()).or_insert(slot);
        map.entry(name.to_lowercase()).or_insert(slot);
        map.entry(name.to_uppercase()).or_insert(slot);
    };
    for (i, n) in state_names_owned.iter().enumerate() {
        add_aliases(&mut var_idx, n, i);
    }
    for (i, n) in indiv_names_owned.iter().enumerate() {
        add_aliases(&mut var_idx, n, state_count + i);
    }
    for (i, n) in intermediates.iter().enumerate() {
        add_aliases(&mut var_idx, n, state_count + indiv_count + i);
    }
    // Reserve extra slots for TIME/T, TAFD, TAD — always appended after intermediates.
    // These names are reserved: reject any ODE state, individual parameter, or
    // intermediate that collides with a reserved slot so the injected solver-time
    // values are always reachable.
    const RESERVED_ODE_NAMES: &[&str] = &["TIME", "T", "TAFD", "TAD", "MACHEPS"];
    for reserved in RESERVED_ODE_NAMES {
        let collides = state_names_owned
            .iter()
            .chain(indiv_names_owned.iter())
            .chain(intermediates.iter())
            .any(|n| n.eq_ignore_ascii_case(reserved));
        if collides {
            return Err(format!(
                "[odes] the name `{reserved}` is reserved for a solver-injected builtin \
                 (TIME/TAFD/TAD/MACHEPS) and cannot be used as a state, \
                 individual-parameter, or intermediate name"
            ));
        }
    }
    let n_base_vars = state_count + indiv_count + intermediates.len();
    let time_slot = n_base_vars;
    let tafd_slot = n_base_vars + 1;
    let tad_slot = n_base_vars + 2;
    let macheps_slot = n_base_vars + 3;
    // Forcefully assign reserved names (overwrite any accidental or_insert entry).
    var_idx.insert("TIME".to_string(), time_slot);
    var_idx.insert("time".to_string(), time_slot);
    var_idx.insert("T".to_string(), time_slot);
    var_idx.insert("t".to_string(), time_slot);
    var_idx.insert("TAFD".to_string(), tafd_slot);
    var_idx.insert("tafd".to_string(), tafd_slot);
    var_idx.insert("TAD".to_string(), tad_slot);
    var_idx.insert("tad".to_string(), tad_slot);
    // MACHEPS — machine epsilon, a compile-time constant. Matches its
    // availability in `[derived]` and `init(...)` expressions.
    var_idx.insert("MACHEPS".to_string(), macheps_slot);
    var_idx.insert("macheps".to_string(), macheps_slot);
    let n_vars_total = n_base_vars + 4;

    // Reject undefined identifiers in ODE RHS expressions (issue #314). In the
    // ODE parse context `fallback_covariate = false`, so a typo'd, omitted,
    // theta-only, or covariate name becomes a plain `Variable` that
    // `resolve_expr_indices` maps to the `usize::MAX` sentinel — the bytecode
    // read then silently returns 0.0, producing a structurally-broken fit with
    // no diagnostic. `var_idx` now holds every resolvable name (states,
    // individual parameters, ODE-block intermediates, and the reserved
    // TIME/TAFD/TAD slots), so any `Variable` whose name is not a key in it
    // cannot resolve. (The covariate guard above only matches `Covariate`
    // nodes, which this parse context never emits, so this is what actually
    // surfaces the bug.)
    let mut undefined_rhs: std::collections::HashSet<String> = std::collections::HashSet::new();
    collect_undefined_vars_in_stmts(&stmts_owned, &var_idx, &mut undefined_rhs);
    if !undefined_rhs.is_empty() {
        let mut names: Vec<String> = undefined_rhs.into_iter().collect();
        names.sort();
        let mut defined: Vec<String> = state_names_owned
            .iter()
            .chain(indiv_names_owned.iter())
            .chain(intermediates.iter())
            .cloned()
            .collect();
        defined.sort();
        return Err(format!(
            "[odes]: RHS references undefined name(s): {}. An ODE RHS may only \
             reference declared states, individual parameters, ODE-block \
             intermediates, or the reserved TIME/TAFD/TAD/MACHEPS variables \
             (defined: {}). If one of these is a covariate, pre-compute the \
             covariate-dependent term in [individual_parameters] and reference \
             that variable here instead.",
            names.join(", "),
            defined.join(", "),
        ));
    }

    // Rewrite the AST so the hot path walks `VariableIdx`/`AssignIdx`/`DiffEqIdx`
    // — pre-resolving names to slot indices. `cov_idx` stays empty (any
    // covariate reference would have been rejected above).
    let empty_cov_idx: HashMap<String, usize> = HashMap::new();
    resolve_variable_indices(
        &mut stmts_owned,
        &var_idx,
        &empty_cov_idx,
        Some(&state_index),
    );

    // Pre-build a `(vars_slot, params_slot)` plan for the indiv-param block.
    // `ode_param_slots` guarantees this Vec is exactly `indiv_count` long;
    // an `assert_eq!` makes that contract local. The previous fallback
    // (`unwrap_or(i)`) silently routed wrong PkParams values if the invariant
    // ever broke; the deeper fix is `unwrap_or(usize::MAX)` so a corrupted
    // plan reads 0 instead of garbage and the integrator surfaces the bug.
    assert_eq!(
        indiv_count,
        indiv_slots_owned.len(),
        "indiv_param_names and indiv_param_slots must be parallel"
    );
    let indiv_to_params_slot: Vec<usize> = (0..indiv_count)
        .map(|i| indiv_slots_owned.get(i).copied().unwrap_or(usize::MAX))
        .collect();

    // Per-thread scratch comes from the shared `FERX_SCRATCH` (see the
    // `FerxThreadScratch` declaration). The closure type is
    // `Box<dyn Fn(...) + Send + Sync>`, which forbids a captured `Cell` /
    // `RefCell`; thread-local storage sidesteps the `Sync` requirement and
    // amortises the per-call allocation across every RK45 stage on a thread.
    // `vec.clear(); vec.resize(n, 0.0)` re-zeros the buffer cheaply (no realloc
    // once the capacity grows), so intermediate slots in untaken if-branches
    // still read 0 just like the old per-call `vec![0.0; n]` path.
    let rhs: Box<dyn Fn(&[f64], &[f64], f64, &mut [f64]) + Send + Sync> =
        Box::new(move |u: &[f64], params: &[f64], t: f64, du: &mut [f64]| {
            // The integrator always passes a `u` whose length matches the
            // declared state count. The old closure index-panicked on
            // `u[i]` if that contract ever broke; preserve that signal here
            // via `debug_assert!` rather than silently truncating.
            debug_assert!(
                u.len() >= state_count,
                "ODE RHS: u.len() = {} < state_count = {}",
                u.len(),
                state_count,
            );
            let copy_n = state_count.min(u.len());

            FERX_SCRATCH.with(|cell| {
                let mut s = cell.borrow_mut();
                // Split-field borrows so eval_statements_indexed_with_stack
                // can take `vars` and `bc_stack` simultaneously without a
                // second TLS lookup. RefMut::deref_mut returns a `&mut`
                // to the struct; field reborrows are disjoint.
                let scratch = &mut *s;
                scratch.rhs_vars.clear();
                scratch.rhs_vars.resize(n_vars_total, 0.0);

                // State values from u[].
                scratch.rhs_vars[..copy_n].copy_from_slice(&u[..copy_n]);

                // Individual parameters from params[] via the pre-computed
                // slot plan. `usize::MAX` slots (impossible under the
                // assert_eq above) leave the var at 0.0.
                for (i, &slot) in indiv_to_params_slot.iter().enumerate() {
                    if let (Some(dst), Some(&val)) =
                        (scratch.rhs_vars.get_mut(state_count + i), params.get(slot))
                    {
                        *dst = val;
                    }
                }

                // TIME / T — solver time axis (same as dataset TIME column).
                if let Some(dst) = scratch.rhs_vars.get_mut(time_slot) {
                    *dst = t;
                }
                // TAFD — t minus first-dose time, injected via params[MAX_PK_PARAMS].
                if let Some(dst) = scratch.rhs_vars.get_mut(tafd_slot) {
                    *dst = params
                        .get(crate::types::MAX_PK_PARAMS)
                        .copied()
                        .filter(|v| v.is_finite())
                        .map_or(f64::NAN, |first| t - first);
                }
                // TAD — t minus last-effective-dose time, injected via params[MAX_PK_PARAMS+1].
                if let Some(dst) = scratch.rhs_vars.get_mut(tad_slot) {
                    *dst = params
                        .get(crate::types::MAX_PK_PARAMS + 1)
                        .copied()
                        .filter(|v| v.is_finite())
                        .map_or(f64::NAN, |last| t - last);
                }
                // MACHEPS — machine epsilon constant, available in every ODE
                // expression (parallels its `[derived]`/`init` availability).
                if let Some(dst) = scratch.rhs_vars.get_mut(macheps_slot) {
                    *dst = f64::EPSILON;
                }

                // Reset du so a state without a firing d/dt this iteration
                // (e.g. inside an untaken if-branch) gets 0.0 rather than
                // stale memory.
                for slot in du.iter_mut() {
                    *slot = 0.0;
                }

                let empty_theta: [f64; 0] = [];
                let empty_eta: [f64; 0] = [];
                let empty_cov: [f64; 0] = [];
                // `[covariate_nn]` outputs are routed via `pk_param_fn`, not
                // the ODE RHS, so this stays empty.
                let empty_nn_outputs: Vec<Vec<f64>> = Vec::new();
                eval_statements_indexed_with_stack(
                    &stmts_owned,
                    &empty_theta,
                    &empty_eta,
                    &empty_cov,
                    &mut scratch.rhs_vars,
                    Some(du),
                    &empty_nn_outputs,
                    &mut scratch.bc_stack,
                );
            });
        });

    // Build the init_fn closure from the extracted `init(state) = expr`
    // directives. It mirrors the RHS variable binding: individual parameters
    // by declaration order (PkParams.values slots), plus state names bound to
    // 0.0 (no drug present at init time). Returns the full n_states vector so
    // the caller can both seed the integrator and re-seed after a reset.
    let init_fn: Option<Box<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync>> = if init_specs.is_empty() {
        None
    } else {
        let n = n_states;
        let indiv = indiv_param_names.to_vec();
        let indiv_slots: Vec<usize> = indiv_param_slots.to_vec();
        let states = state_names.to_vec();
        let specs = init_specs;
        Some(Box::new(move |params: &[f64]| -> Vec<f64> {
            let mut vars: HashMap<String, f64> = HashMap::new();
            for name in &states {
                vars.insert(name.clone(), 0.0);
                vars.insert(name.to_lowercase(), 0.0);
            }
            for (i, name) in indiv.iter().enumerate() {
                let slot = indiv_slots.get(i).copied().unwrap_or(i);
                if slot < params.len() {
                    vars.insert(name.clone(), params[slot]);
                    vars.insert(name.to_uppercase(), params[slot]);
                    vars.insert(name.to_lowercase(), params[slot]);
                }
            }
            let empty_theta: [f64; 0] = [];
            let empty_eta: [f64; 0] = [];
            let empty_cov: HashMap<String, f64> = HashMap::new();
            let empty_nn: Vec<Vec<f64>> = Vec::new();
            let mut u0 = vec![0.0; n];
            for (idx, expr) in &specs {
                u0[*idx] =
                    eval_expression(expr, &empty_theta, &empty_eta, &empty_cov, &vars, &empty_nn);
            }
            u0
        }))
    };

    // Compartment-indexed dose attributes (NONMEM `Fn`/`ALAGn`; issue #369). An
    // individual parameter named `F{c}` / `ALAG{c}` / `LAGTIME{c}` binds the
    // bioavailability / lag for doses into compartment `c` (1-based); the bare
    // `F`/`lagtime` (at PK_IDX_F/PK_IDX_LAGTIME) remain the all-compartment
    // default, overridden per compartment by an indexed entry. The slot is the
    // one `ode_param_slots` already assigned the name (parallel to
    // `indiv_param_names`), so the RHS can still read the same value.
    let mut dose_attr_map = crate::types::DoseAttrMap::default();
    for (i, name) in indiv_param_names.iter().enumerate() {
        if let Some((attr, cmt)) = crate::types::DoseAttr::from_indexed_name(name) {
            if cmt > n_states {
                return Err(format!(
                    "[individual_parameters]: `{name}` is a compartment-indexed dose \
                     attribute for compartment {cmt}, but the model has only {n_states} \
                     compartment(s) {state_names:?}. Compartment indices are 1-based."
                ));
            }
            // `indiv_param_slots` is parallel to `indiv_param_names` (asserted
            // above), so the slot for `i` always exists — index directly, as the
            // input-rate extractor (`extract_input_rate_terms`) already does.
            let slot = indiv_param_slots[i];
            dose_attr_map.insert(attr, cmt, slot);
        }
    }

    Ok(crate::ode::OdeSpec {
        rhs,
        n_states,
        state_names: state_names.to_vec(),
        readout: crate::ode::OdeReadout::ObsCmt(obs_cmt_idx),
        diffusion_var: Vec::new(),
        init_fn,
        // Default tolerances; overwritten from [fit_options] / settings by
        // CompiledModel::sync_ode_solver_opts once fit options are merged.
        solver_opts: crate::ode::OdeSolverOptions::default(),
        // Built-in absorption forcing terms split out of the [odes] RHS above
        // (design A); empty for models with no transit()/etc. input-rate call.
        input_rate,
        dose_attr_map,
    })
}

/// Parse an `init(state) = <expr>` directive line. Returns `(state_name,
/// expr_str)` when `line` is such a directive, else `None` (so the caller
/// routes it to the d/dt statement parser). Tolerates whitespace variants
/// (`init (R)=...`, `init( R ) = ...`).
fn parse_init_line(line: &str) -> Option<(String, String)> {
    let rest = line.trim().strip_prefix("init")?.trim_start();
    let inner = rest.strip_prefix('(')?;
    let close = inner.find(')')?;
    let name = inner[..close].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let expr = inner[close + 1..].trim_start().strip_prefix('=')?.trim();
    if expr.is_empty() {
        return None;
    }
    Some((name, expr.to_string()))
}

// ── Helper: parse "[1.0, 2.0, 3.0]" → Vec<f64> ────────────────────────────

fn parse_float_array(s: &str) -> Result<Vec<f64>, String> {
    let s = s.trim().trim_start_matches('[').trim_end_matches(']');
    s.split(',')
        .map(|v| {
            v.trim()
                .parse::<f64>()
                .map_err(|_| format!("Bad float in array: '{}'", v.trim()))
        })
        .collect()
}

// --- Internal types ---

struct ThetaSpec {
    name: String,
    init: f64,
    lower: f64,
    upper: f64,
    fixed: bool,
}

struct OmegaSpec {
    name: String,
    variance: f64,
    fixed: bool,
    /// `true` when the user wrote `omega NAME ~ X (sd)` — i.e. they specified
    /// the initial value on the standard-deviation scale and the parser squared
    /// it. Purely display metadata; the stored `variance` is always on the
    /// variance scale.
    init_as_sd: bool,
}

/// Specifies a block (correlated) group of omegas.
/// The values are the lower triangle of the covariance matrix, row-wise:
/// e.g. for 2x2: [var1, cov12, var2]; for 3x3: [var1, cov12, var2, cov13, cov23, var3]
struct BlockOmegaSpec {
    names: Vec<String>,
    lower_triangle: Vec<f64>,
    fixed: bool,
}

struct SigmaSpec {
    name: String,
    /// Internal sigma value on the standard-deviation scale (the form the
    /// likelihood code consumes). The parser converts variance-scale input
    /// (the default since issue #56) to SD via `sqrt`.
    value: f64,
    fixed: bool,
    /// `true` when the user wrote `sigma NAME ~ X (sd)` — i.e. specified the
    /// initial value directly as a standard deviation. `false` for the default
    /// (variance) case. Purely display metadata.
    init_as_sd: bool,
}

/// Diagonal inter-occasion variability (kappa) specification.
struct KappaSpec {
    name: String,
    variance: f64,
    fixed: bool,
    /// Same semantics as `OmegaSpec::init_as_sd` — `true` when the user wrote
    /// `kappa NAME ~ X (sd)` and the parser squared the value.
    init_as_sd: bool,
}

/// Block (correlated) IOV kappa specification — mirrors `BlockOmegaSpec`.
struct BlockKappaSpec {
    names: Vec<String>,
    lower_triangle: Vec<f64>,
    fixed: bool,
}

/// All kappa-related data returned by `parse_parameters`.
struct ParsedKappas {
    diagonal: Vec<KappaSpec>,
    block: Vec<BlockKappaSpec>,
    /// All kappa names in declaration order (diagonal then block, interleaved).
    names_ordered: Vec<String>,
}

// --- Block extraction ---

fn extract_model_name(content: &str) -> String {
    let re = Regex::new(r"(?m)^\s*model\s+(\w+)").unwrap();
    re.captures(content)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .unwrap_or_else(|| "Unnamed".to_string())
}

/// Extracted-block state: the original (unnamed) block map plus a second
/// map for blocks with an instance name in the header — `[block_type NAME]`.
///
/// Unnamed blocks (e.g. `[parameters]`, `[odes]`) live in `unnamed` with the
/// block type as key. Each block type may appear at most once and is
/// represented as a flat `Vec<String>` of body lines.
///
/// Named blocks (currently `[covariate_nn NAME]`; later `[dynamics_nn NAME]`
/// in Phase B) live in `named[type][name]`. Multiple instances per type are
/// supported.
#[derive(Default, Debug)]
struct ExtractedBlocks {
    unnamed: HashMap<String, Vec<String>>,
    named: HashMap<String, HashMap<String, Vec<String>>>,
    /// 1-based source line of each unnamed block's header line, keyed by block
    /// type. First occurrence wins. Used to give `ferx check` diagnostics a
    /// block-level location.
    block_lines: HashMap<String, usize>,
}

fn extract_blocks(content: &str) -> Result<ExtractedBlocks, String> {
    let mut out = ExtractedBlocks::default();
    // Two header forms:
    //   `[block_type]`            — unnamed (existing)
    //   `[block_type INSTANCE]`   — named (e.g. `[covariate_nn TYPICAL_PK]`)
    // Anchor on the whole line so things like `states=[central]` inside an
    // ODE structural definition aren't misread as a block-tag opener.
    let block_re = Regex::new(r"^\[(\w+)(?:\s+(\w+))?\]$").unwrap();

    // current_target: either an unnamed block name, or a (block_type, instance) pair.
    enum BlockTarget {
        Unnamed(String),
        Named { ty: String, name: String },
    }
    let mut current: Option<BlockTarget> = None;

    for (idx, line) in content.lines().enumerate() {
        let without_comment = match line.find('#').into_iter().chain(line.find("//")).min() {
            Some(idx) => &line[..idx],
            None => line,
        };
        let trimmed = without_comment.trim();
        if trimmed.is_empty() {
            continue;
        }

        if let Some(caps) = block_re.captures(trimmed) {
            let ty = caps[1].to_lowercase();
            current = match caps.get(2) {
                Some(m) => Some(BlockTarget::Named {
                    ty,
                    name: m.as_str().to_string(),
                }),
                None => {
                    // Record the 1-based header line; first occurrence wins.
                    out.block_lines.entry(ty.clone()).or_insert(idx + 1);
                    Some(BlockTarget::Unnamed(ty))
                }
            };
            continue;
        }

        if trimmed.starts_with("model ") || trimmed == "end" {
            continue;
        }

        match current.as_ref() {
            Some(BlockTarget::Unnamed(block)) => {
                out.unnamed
                    .entry(block.clone())
                    .or_default()
                    .push(trimmed.to_string());
            }
            Some(BlockTarget::Named { ty, name }) => {
                out.named
                    .entry(ty.clone())
                    .or_default()
                    .entry(name.clone())
                    .or_default()
                    .push(trimmed.to_string());
            }
            None => { /* lines before any block header are ignored */ }
        }
    }

    Ok(out)
}

// --- Parameter parsing ---

/// Coalesce `block_omega`/`block_kappa` declarations whose lower-triangle list
/// spans several physical lines back into a single logical line.
///
/// `extract_blocks` hands `parse_parameters` one trimmed string per source
/// line, so a `block_omega`/`block_kappa` whose lower-triangle list is written
/// across multiple lines arrives split apart. Here we rejoin any run of lines
/// from the first unbalanced `[` through the matching `]` into one line (joined
/// with spaces) so the existing single-line regexes match unchanged. Lines with
/// no bracket, or with balanced brackets, pass through untouched. A trailing
/// bare `FIX` left on its own line after the closing `]` is folded back onto
/// the block line so the FIX flag isn't silently lost.
fn join_bracketed_lines(lines: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut depth: i32 = 0;
    for line in lines {
        if depth > 0 {
            buf.push(' ');
            buf.push_str(line);
        } else {
            buf = line.clone();
        }
        depth += line.matches('[').count() as i32;
        depth -= line.matches(']').count() as i32;
        if depth <= 0 {
            depth = 0;
            let logical = std::mem::take(&mut buf);
            // Fold a bare `FIX` line onto the block declaration just emitted.
            // Restricted to a previous line containing `]` so we never attach
            // it to a non-block parameter line.
            if logical.trim().eq_ignore_ascii_case("FIX")
                && out.last().is_some_and(|l: &String| l.contains(']'))
            {
                let last = out.last_mut().unwrap();
                last.push(' ');
                last.push_str(logical.trim());
            } else {
                out.push(logical);
            }
        }
    }
    // An unterminated `[` leaves text in the buffer; emit it so the downstream
    // regex fails to match and the user gets a clear "Bad block_omega" error
    // rather than the line silently vanishing.
    if !buf.is_empty() {
        out.push(buf);
    }
    out
}

fn parse_parameters(
    lines: &[String],
) -> Result<
    (
        Vec<ThetaSpec>,
        Vec<OmegaSpec>,
        Vec<BlockOmegaSpec>,
        Vec<SigmaSpec>,
        Vec<String>,  // BSV eta names in declaration order
        ParsedKappas, // IOV kappa specs (diagonal and/or block)
    ),
    String,
> {
    let mut thetas = Vec::new();
    let mut omegas = Vec::new();
    let mut block_omegas = Vec::new();
    let mut sigmas = Vec::new();
    let mut eta_names_ordered = Vec::new();
    let mut kappas: Vec<KappaSpec> = Vec::new();
    let mut block_kappas: Vec<BlockKappaSpec> = Vec::new();
    let mut kappa_names_ordered: Vec<String> = Vec::new();

    // theta NAME(init)  |  theta NAME(init, FIX)
    // theta NAME(init, lower, upper)  |  theta NAME(init, lower, upper, FIX)
    //
    // Whitespace between NAME and `(` is allowed (`theta TVCL (5, ...)`) — without
    // it the line silently falls through and TVCL is later misclassified as a
    // covariate.
    //
    // The `FIX` keyword is case-insensitive and must be the exact token —
    // the trailing `\b` rejects prefix matches like `FIXED`, which would
    // otherwise silently mark the parameter as fixed.
    // Accept FIX in any of these positions:
    //   theta NAME(init, FIX)                   — comma before FIX, inside parens
    //   theta NAME(init FIX)                    — no comma, inside parens
    //   theta NAME(init) FIX                    — after closing paren
    //   theta NAME(init, lower, FIX)            — lower only, FIX inside
    //   theta NAME(init, lower) FIX             — lower only, FIX outside
    //   theta NAME(init, lower, upper, FIX)
    //   theta NAME(init, lower, upper) FIX
    // Bounds: lower (group 3) is optional; upper (group 4) is optional within
    // the bounds sub-group, defaulting to 1e9 when absent.
    // Group 5 captures FIX inside the parens; group 6 captures FIX outside.
    // `fixed` is true when either group is present.
    let theta_re = Regex::new(
        r"(?i)theta\s+(\w+)\s*\(\s*([0-9eE.+-]+)\s*(?:,\s*([0-9eE.+-]+)(?:\s*,\s*([0-9eE.+-]+))?)?\s*(?:,?\s*(FIX)\b)?\s*\)(?:\s+(FIX)\b)?",
    )
    .unwrap();

    // omega NAME ~ value [FIX] [(sd|variance|var)] [FIX]
    //
    // Initial value defaults to the variance scale (matching how the optimizer
    // stores omega internally). Append `(sd)` to declare the value on the
    // standard-deviation scale — the parser squares it before storing. The
    // `(variance)` / `(var)` annotation is accepted as an explicit no-op for
    // symmetry with sigma.
    // FIX may appear before or after the scale annotation (group 3 = FIX
    // before annotation, group 4 = annotation, group 5 = FIX after).
    // Note: the first FIX group requires \s+ (at least one space) so that
    // `value(sd)` without a space between value and annotation still matches
    // correctly — the annotation group uses \s* (zero or more) intentionally.
    let omega_re = Regex::new(
        r"(?i)omega\s+(\w+)\s*~\s*([0-9eE.+-]+)(?:\s+(FIX)\b)?(?:\s*\((sd|variance|var)\))?(?:\s+(FIX)\b)?",
    )
    .unwrap();

    // block_omega (NAME1, NAME2, ...) = [lower_triangle_values]  |  ... FIX
    //
    // Block omegas are variance-scale only — the lower triangle mixes
    // variances and covariances, so a single `(sd)` flag would be ambiguous.
    let block_omega_re =
        Regex::new(r"(?i)block_omega\s*\(([^)]+)\)\s*=\s*\[([^\]]+)\](?:\s+(FIX)\b)?").unwrap();

    // sigma NAME ~ value [FIX] [(sd|variance|var)] [FIX]
    //
    // As of issue #56, sigma defaults to the variance scale (matching omega).
    // `(sd)` opts back into specifying a standard deviation directly. The
    // parser converts variance → internal SD via `sqrt` so the residual-error
    // and likelihood code (which work in SD) need no changes.
    // FIX may appear before or after the scale annotation (same group layout
    // as omega_re: group 3 = FIX before, group 4 = annotation, group 5 = FIX after).
    let sigma_re = Regex::new(
        r"(?i)sigma\s+(\w+)\s*~\s*([0-9eE.+-]+)(?:\s+(FIX)\b)?(?:\s*\((sd|variance|var)\))?(?:\s+(FIX)\b)?",
    )
    .unwrap();

    // kappa NAME ~ value [FIX] [(sd|variance|var)] [FIX]  (IOV diagonal variance)
    // Same group layout as omega_re.
    let kappa_re = Regex::new(
        r"(?i)kappa\s+(\w+)\s*~\s*([0-9eE.+-]+)(?:\s+(FIX)\b)?(?:\s*\((sd|variance|var)\))?(?:\s+(FIX)\b)?",
    )
    .unwrap();

    // block_kappa (NAME1, NAME2, ...) = [lower_triangle_values]  |  ... FIX
    let block_kappa_re =
        Regex::new(r"(?i)block_kappa\s*\(([^)]+)\)\s*=\s*\[([^\]]+)\](?:\s+(FIX)\b)?").unwrap();

    // Rejoin multi-line `block_omega`/`block_kappa` declarations before matching.
    let lines = join_bracketed_lines(lines);
    for line in &lines {
        if let Some(caps) = theta_re.captures(line) {
            let name = caps[1].to_string();
            let init: f64 = caps[2]
                .parse()
                .map_err(|_| format!("Bad theta init: {}", line))?;
            let lower: f64 = caps
                .get(3)
                .map(|m| m.as_str().parse().unwrap_or(1e-9))
                .unwrap_or(1e-9);
            let upper: f64 = caps
                .get(4)
                .map(|m| m.as_str().parse().unwrap_or(1e9))
                .unwrap_or(1e9);
            let fixed = caps.get(5).is_some() || caps.get(6).is_some();
            thetas.push(ThetaSpec {
                name,
                init,
                lower,
                upper,
                fixed,
            });
        } else if let Some(caps) = block_omega_re.captures(line) {
            let names: Vec<String> = caps[1].split(',').map(|s| s.trim().to_string()).collect();
            let values: Vec<f64> = caps[2]
                .split(',')
                .map(|s| {
                    s.trim()
                        .parse::<f64>()
                        .map_err(|_| format!("Bad block_omega value in: {}", line))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let n = names.len();
            let expected = n * (n + 1) / 2;
            if values.len() != expected {
                return Err(format!(
                    "block_omega with {} etas expects {} lower-triangle values, got {}: {}",
                    n,
                    expected,
                    values.len(),
                    line
                ));
            }
            for n in &names {
                eta_names_ordered.push(n.clone());
            }
            let fixed = caps.get(3).is_some();
            block_omegas.push(BlockOmegaSpec {
                names,
                lower_triangle: values,
                fixed,
            });
        } else if let Some(caps) = block_kappa_re.captures(line) {
            let names: Vec<String> = caps[1].split(',').map(|s| s.trim().to_string()).collect();
            let values: Vec<f64> = caps[2]
                .split(',')
                .map(|s| {
                    s.trim()
                        .parse::<f64>()
                        .map_err(|_| format!("Bad block_kappa value in: {}", line))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let n = names.len();
            let expected = n * (n + 1) / 2;
            if values.len() != expected {
                return Err(format!(
                    "block_kappa with {} kappas expects {} lower-triangle values, got {}: {}",
                    n,
                    expected,
                    values.len(),
                    line
                ));
            }
            for name in &names {
                kappa_names_ordered.push(name.clone());
            }
            let fixed = caps.get(3).is_some();
            block_kappas.push(BlockKappaSpec {
                names,
                lower_triangle: values,
                fixed,
            });
        } else if let Some(caps) = omega_re.captures(line) {
            let name = caps[1].to_string();
            let raw: f64 = caps[2]
                .parse()
                .map_err(|_| format!("Bad omega: {}", line))?;
            // Group 4 = scale annotation; groups 3 and 5 = FIX (before/after).
            let init_as_sd = caps
                .get(4)
                .map(|m| m.as_str().eq_ignore_ascii_case("sd"))
                .unwrap_or(false);
            // Negative values are nonsensical on either scale: variance ≥ 0
            // by definition, and SD ≥ 0 since SD = sqrt(variance). The
            // optimizer's Cholesky pack uses `l.max(1e-10).ln()` and would
            // silently clamp them — fail loudly here instead.
            if raw < 0.0 {
                let scale = if init_as_sd { "SD" } else { "variance" };
                return Err(format!(
                    "omega '{name}' has a negative initial {scale} ({raw}); both variance and SD must be non-negative"
                ));
            }
            let variance = if init_as_sd { raw * raw } else { raw };
            let fixed = caps.get(3).is_some() || caps.get(5).is_some();
            eta_names_ordered.push(name.clone());
            omegas.push(OmegaSpec {
                name,
                variance,
                fixed,
                init_as_sd,
            });
        } else if let Some(caps) = sigma_re.captures(line) {
            let name = caps[1].to_string();
            let raw: f64 = caps[2]
                .parse()
                .map_err(|_| format!("Bad sigma: {}", line))?;
            // Group 4 = scale annotation; groups 3 and 5 = FIX (before/after).
            let init_as_sd = caps
                .get(4)
                .map(|m| m.as_str().eq_ignore_ascii_case("sd"))
                .unwrap_or(false);
            // Reject negatives on both scales. On the default (variance)
            // path a negative would slip through `sqrt` as NaN; on the
            // (sd) path the optimizer's `s.max(1e-10).ln()` packing would
            // silently clamp it to 1e-10. Either is a hard-to-debug silent
            // failure — surface the bad input at parse time.
            if raw < 0.0 {
                let scale = if init_as_sd { "SD" } else { "variance" };
                return Err(format!(
                    "sigma '{name}' has a negative initial {scale} ({raw}); both variance and SD must be non-negative"
                ));
            }
            let value = if init_as_sd { raw } else { raw.sqrt() };
            let fixed = caps.get(3).is_some() || caps.get(5).is_some();
            sigmas.push(SigmaSpec {
                name,
                value,
                fixed,
                init_as_sd,
            });
        } else if let Some(caps) = kappa_re.captures(line) {
            let name = caps[1].to_string();
            let raw: f64 = caps[2]
                .parse()
                .map_err(|_| format!("Bad kappa: {}", line))?;
            // Group 4 = scale annotation; groups 3 and 5 = FIX (before/after).
            let init_as_sd = caps
                .get(4)
                .map(|m| m.as_str().eq_ignore_ascii_case("sd"))
                .unwrap_or(false);
            if raw < 0.0 {
                let scale = if init_as_sd { "SD" } else { "variance" };
                return Err(format!(
                    "kappa '{name}' has a negative initial {scale} ({raw}); both variance and SD must be non-negative"
                ));
            }
            let variance = if init_as_sd { raw * raw } else { raw };
            let fixed = caps.get(3).is_some() || caps.get(5).is_some();
            kappa_names_ordered.push(name.clone());
            kappas.push(KappaSpec {
                name,
                variance,
                fixed,
                init_as_sd,
            });
        }
    }

    // Reject names that appear in both kappa and block_kappa
    let diag_name_set: std::collections::HashSet<&str> =
        kappas.iter().map(|k| k.name.as_str()).collect();
    for bk in &block_kappas {
        for name in &bk.names {
            if diag_name_set.contains(name.as_str()) {
                return Err(format!(
                    "'{}' appears in both kappa and block_kappa declarations",
                    name
                ));
            }
        }
    }

    Ok((
        thetas,
        omegas,
        block_omegas,
        sigmas,
        eta_names_ordered,
        ParsedKappas {
            diagonal: kappas,
            block: block_kappas,
            names_ordered: kappa_names_ordered,
        },
    ))
}

// --- Build omega matrix from diagonal + block specs ---

/// Construct a full OmegaMatrix from diagonal omega specs and block omega specs.
/// The `eta_names` vector determines the matrix ordering (declaration order from
/// the model file). If any block omega is present, the matrix is non-diagonal.
fn build_omega_matrix(
    diag_omegas: &[OmegaSpec],
    block_omegas: &[BlockOmegaSpec],
    eta_names: &[String],
) -> Result<OmegaMatrix, String> {
    let n = eta_names.len();
    if n == 0 {
        return Err("No omega parameters defined".to_string());
    }

    // If no block omegas, use the simple diagonal path
    if block_omegas.is_empty() {
        let variances: Vec<f64> = diag_omegas.iter().map(|o| o.variance).collect();
        return Ok(OmegaMatrix::from_diagonal(&variances, eta_names.to_vec()));
    }

    // Build a name→index map
    let name_to_idx: std::collections::HashMap<&str, usize> = eta_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();

    // Start with a zero matrix, fill diagonal entries from diagonal specs
    let mut matrix = nalgebra::DMatrix::zeros(n, n);
    // free_mask: diagonal entries always free; off-diagonals free only when
    // both etas belong to the same block_omega declaration.
    let mut free_mask = nalgebra::DMatrix::from_element(n, n, false);
    for i in 0..n {
        free_mask[(i, i)] = true;
    }
    for spec in diag_omegas {
        if let Some(&idx) = name_to_idx.get(spec.name.as_str()) {
            matrix[(idx, idx)] = spec.variance;
        }
    }

    // Fill block entries from block specs (lower triangle, row-wise)
    for block in block_omegas {
        let block_n = block.names.len();
        let mut val_idx = 0;
        for row in 0..block_n {
            let i = *name_to_idx.get(block.names[row].as_str()).ok_or_else(|| {
                format!("block_omega references unknown eta '{}'", block.names[row])
            })?;
            for col in 0..=row {
                let j = *name_to_idx.get(block.names[col].as_str()).ok_or_else(|| {
                    format!("block_omega references unknown eta '{}'", block.names[col])
                })?;
                matrix[(i, j)] = block.lower_triangle[val_idx];
                matrix[(j, i)] = block.lower_triangle[val_idx]; // symmetric
                free_mask[(i, j)] = true;
                free_mask[(j, i)] = true;
                val_idx += 1;
            }
        }
    }

    Ok(OmegaMatrix::from_matrix_with_mask(
        matrix,
        eta_names.to_vec(),
        false,
        free_mask,
    ))
}

/// Build the per-eta `omega_fixed` flags from parsed diagonal + block specs.
///
/// Rules:
/// - `omega NAME ~ value FIX`: flag that eta as fixed.
/// - `block_omega (...) = [...] FIX`: flag every eta in the block.
/// - A diagonal omega FIX on an eta that is also listed in a (free) block is
///   rejected — you must fix the whole block instead.
fn build_omega_fixed(
    diag_omegas: &[OmegaSpec],
    block_omegas: &[BlockOmegaSpec],
    eta_names: &[String],
) -> Result<Vec<bool>, String> {
    let name_to_idx: std::collections::HashMap<&str, usize> = eta_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();

    let mut fixed = vec![false; eta_names.len()];

    for spec in diag_omegas {
        if spec.fixed {
            if let Some(&idx) = name_to_idx.get(spec.name.as_str()) {
                fixed[idx] = true;
            }
        }
    }

    for block in block_omegas {
        for name in &block.names {
            let idx = *name_to_idx
                .get(name.as_str())
                .ok_or_else(|| format!("block_omega references unknown eta '{}'", name))?;
            // If the eta was already marked FIX via a diagonal spec but the
            // block is not fully fixed, that's ambiguous.
            if fixed[idx] && !block.fixed {
                return Err(format!(
                    "'{}' is marked FIX but belongs to a non-FIX block_omega; \
                     fix the whole block instead",
                    name
                ));
            }
            if block.fixed {
                fixed[idx] = true;
            }
        }
    }

    Ok(fixed)
}

// --- Structural model parsing ---

/// Parse a `role=VAR, role=VAR, …` list from a `pk NAME(...)` / `ode_template
/// NAME(...)` parameter string into a `role(lowercased) → VAR` map.
///
/// **Strict and single-sourced** (Ron, #363): a pair that is not exactly
/// `role=VAR` with both sides non-empty is a hard error, and a duplicate role is
/// a hard error — for both the analytical `pk` and the `ode_template` paths, so
/// the two can't drift in strictness (they used to: `pk` silently dropped
/// malformed pairs and last-wins on duplicates). A trailing comma (empty pair) is
/// tolerated. `ctx` is the directive prefix used in error messages, e.g.
/// `"pk two_cpt_oral"` or `"ode_template two_cpt_oral"`.
fn parse_role_pairs(params_str: &str, ctx: &str) -> Result<HashMap<String, String>, String> {
    let mut map: HashMap<String, String> = HashMap::new();
    for pair in params_str.split(',') {
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }
        let parts: Vec<&str> = pair.split('=').map(str::trim).collect();
        if parts.len() != 2 || parts[0].is_empty() || parts[1].is_empty() {
            return Err(format!(
                "{ctx}: malformed parameter `{pair}` (expected `role=VARNAME`)."
            ));
        }
        if map
            .insert(parts[0].to_lowercase(), parts[1].to_string())
            .is_some()
        {
            return Err(format!("{ctx}: duplicate parameter `{}`.", parts[0]));
        }
    }
    Ok(map)
}

fn parse_structural_model(lines: &[String]) -> Result<(PkModel, HashMap<String, String>), String> {
    // pk model_name(param=VAR, param=VAR, ...)
    let pk_re = Regex::new(r"pk\s+(\w+)\(([^)]+)\)").unwrap();

    for line in lines {
        if let Some(caps) = pk_re.captures(line) {
            let model_name = &caps[1];
            // Name → model is resolved through the shared `PkModel::from_name`
            // (canonical + long-form aliases) so the `pk` and `ode_template` paths
            // accept exactly the same set; retired and unknown names are handled
            // here because they produce path-specific diagnostics, not a `PkModel`.
            let pk_model = match PkModel::from_name(model_name) {
                Some(m) => m,
                // Retired names (issue #176): bolus and infusion are no longer
                // separate model variants — the route is read per-dose from the
                // RATE column. Emit a migration error so users update their
                // model files explicitly rather than relying on a silent alias.
                None => match model_name {
                    retired @ ("one_cpt_iv_bolus"
                    | "one_compartment_iv_bolus"
                    | "one_cpt_infusion"
                    | "one_compartment_infusion"
                    | "two_cpt_iv_bolus"
                    | "two_compartment_iv_bolus"
                    | "two_cpt_infusion"
                    | "two_compartment_infusion"
                    | "three_cpt_iv_bolus"
                    | "three_compartment_iv_bolus"
                    | "three_cpt_infusion"
                    | "three_compartment_infusion") => {
                        let n = if retired.starts_with("one") {
                            "one"
                        } else if retired.starts_with("two") {
                            "two"
                        } else {
                            "three"
                        };
                        return Err(format!(
                            "`{retired}` was removed in #176; use `{n}_cpt_iv` instead. \
                             Bolus and infusion administration are now driven by the \
                             RATE column in the dataset (RATE=0 for bolus, RATE>0 for \
                             infusion), so a single `{n}_cpt_iv` model handles either \
                             or a mix of both within the same subject."
                        ));
                    }
                    other => return Err(format!("Unknown PK model: {}", other)),
                },
            };

            // Strict, shared with `ode_template` (`parse_role_pairs`): a malformed
            // or duplicate `role=VAR` pair is rejected rather than silently dropped.
            let param_map = parse_role_pairs(&caps[2], &format!("pk {model_name}"))?;

            // Structural-mapping validation (required params present, unused
            // params, undefined references) is deferred to `parse_full_model`,
            // which runs it *after* `build_pk_param_fn` so the per-key checks
            // (unknown key / undefined reference, #308) report first and the
            // required-completeness / unused checks (#309) layer cleanly on top.

            return Ok((pk_model, param_map));
        }
    }

    Err("No PK model found in [structural_model] block".to_string())
}

// --- Error model parsing ---

/// Parsed-but-unresolved `[error_model]` block. Sigma references are still
/// names here; `build_error_spec` resolves them to indices into the flat
/// global sigma vector once the `[parameters]` block has been parsed.
enum ParsedErrorModel {
    /// Single error model applied to all observations (no `CMT=` prefix).
    /// Carries the referenced sigma name(s) so `build_error_spec` can check
    /// they were declared in `[parameters]` (sigmas are then consumed
    /// positionally from the global sigma vector).
    Single(ErrorModel, Vec<String>),
    /// Per-CMT error models (every line prefixed `CMT=N:`). One entry per line,
    /// in source order; duplicates are rejected here.
    PerCmt(Vec<(usize, ErrorModel, Vec<String>)>),
}

/// Log-transform-both-sides (LTBS) flags extracted from the `[error_model]`
/// block. `log_transform` mirrors `CompiledModel::log_transform`; `dv_pre_logged`
/// mirrors `CompiledModel::dv_pre_logged`. Both default `false` (no LTBS).
#[derive(Clone, Copy, Default)]
struct LtbsFlags {
    log_transform: bool,
    dv_pre_logged: bool,
}

fn parse_error_model(lines: &[String]) -> Result<(ParsedErrorModel, LtbsFlags), String> {
    // Single-endpoint:
    //   DV ~ proportional(SIGMA_NAME)
    //   DV ~ additive(SIGMA_NAME)
    //   DV ~ combined(SIGMA1, SIGMA2)
    // Log-transform-both-sides (LTBS), additive error on the log scale:
    //   log(DV) ~ additive(SIGMA_NAME)   # natural-scale DV; engine logs DV + pred
    //   DV ~ log_additive(SIGMA_NAME)    # DV already log; engine logs the pred only
    // Multi-endpoint (per-CMT dispatch, ODE models only):
    //   CMT=2: DV ~ proportional(PROP_ERR_PK)
    //   CMT=3: DV ~ additive(ADD_ERR_PD)
    let re = Regex::new(r"(\w+)\s*~\s*(\w+)\(([^)]+)\)").unwrap();
    // LTBS LHS `log(DV) ~ TYPE(SIGMA)` — captures the logged data column,
    // the error type, and the sigma list.
    let log_lhs_re = Regex::new(r"^\s*log\s*\(\s*(\w+)\s*\)\s*~\s*(\w+)\(([^)]+)\)").unwrap();
    let cmt_re = Regex::new(r"^\s*CMT\s*=\s*(\d+)\s*:\s*(.*)$").unwrap();

    // singles carry the per-line LTBS flags so the chosen single can stamp them.
    let mut singles: Vec<(ErrorModel, Vec<String>, LtbsFlags)> = Vec::new();
    let mut per_cmt: Vec<(usize, ErrorModel, Vec<String>)> = Vec::new();

    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // Peel off an optional `CMT=N:` prefix.
        let (cmt_opt, body) = if let Some(c) = cmt_re.captures(trimmed) {
            let cmt: usize = c[1]
                .parse()
                .map_err(|_| format!("Invalid CMT index in [error_model] line: {}", trimmed))?;
            (Some(cmt), c[2].trim().to_string())
        } else {
            (None, trimmed.to_string())
        };

        // A `log(DV)` LHS (case 2) is detected first since the plain regex's
        // `\w+` LHS can't match the parenthesised form.
        let (lhs_logged, caps) = if let Some(c) = log_lhs_re.captures(&body) {
            (true, c)
        } else {
            match re.captures(&body) {
                Some(c) => (false, c),
                None => continue, // not an error-model statement
            }
        };
        let error_type = caps[2].to_lowercase();
        let sigma_names: Vec<String> = caps[3].split(',').map(|s| s.trim().to_string()).collect();
        // `log_additive` (case 1) is additive error whose prediction is logged
        // while DV is taken as-is (already log-transformed in the data).
        let type_is_log_additive = error_type == "log_additive";
        let error_model = match error_type.as_str() {
            "additive" | "log_additive" => ErrorModel::Additive,
            "proportional" => ErrorModel::Proportional,
            "combined" => ErrorModel::Combined,
            other => return Err(format!("Unknown error model: {}", other)),
        };
        if sigma_names.len() != error_model.n_sigma() {
            return Err(format!(
                "[error_model] {} model expects {} sigma(s) but {} given: {}",
                error_type,
                error_model.n_sigma(),
                sigma_names.len(),
                trimmed
            ));
        }

        // LTBS validation. `log(DV) ~ log_additive(...)` double-logs the data and
        // is rejected; LTBS only supports additive error on the log scale; and the
        // per-CMT (multi-endpoint) form is out of scope for LTBS.
        if lhs_logged && type_is_log_additive {
            return Err(format!(
                "[error_model] `{}` double-log-transforms DV: use `log(DV) ~ additive(...)` \
                 (engine logs DV) OR `DV ~ log_additive(...)` (DV already log), not both",
                trimmed
            ));
        }
        let log_transform = lhs_logged || type_is_log_additive;
        if log_transform && !matches!(error_model, ErrorModel::Additive) {
            return Err(format!(
                "[error_model] log-transform-both-sides supports only additive error on the \
                 log scale; use `log(DV) ~ additive(...)` or `DV ~ log_additive(...)`: {}",
                trimmed
            ));
        }
        if log_transform && cmt_opt.is_some() {
            return Err(
                "[error_model] log-transform-both-sides (`log(DV) ~ ...` / `log_additive`) \
                 is not supported with per-CMT (multi-endpoint) error models"
                    .to_string(),
            );
        }
        // The engine always log-transforms the `DV` column under LTBS — a
        // different LHS name (e.g. `log(CONC) ~ additive(...)`) would parse but
        // silently operate on `DV`, which is confusing. Reject it.
        if log_transform && !caps[1].eq_ignore_ascii_case("DV") {
            return Err(format!(
                "[error_model] log-transform-both-sides must reference the `DV` column \
                 (got `{}`): write `log(DV) ~ additive(...)` or `DV ~ log_additive(...)`",
                &caps[1]
            ));
        }
        let flags = LtbsFlags {
            log_transform,
            // `log(DV)` logs DV in the engine (data is natural scale);
            // `log_additive` trusts the data is already on the log scale.
            dv_pre_logged: type_is_log_additive,
        };

        match cmt_opt {
            Some(cmt) => per_cmt.push((cmt, error_model, sigma_names)),
            None => singles.push((error_model, sigma_names, flags)),
        }
    }

    if !singles.is_empty() && !per_cmt.is_empty() {
        return Err("[error_model] mixes a plain `DV ~ ...` line with `CMT=N:` \
                    per-compartment lines; use one style or the other"
            .to_string());
    }

    if !per_cmt.is_empty() {
        let mut seen = std::collections::HashSet::new();
        for (cmt, _, _) in &per_cmt {
            if !seen.insert(*cmt) {
                return Err(format!(
                    "[error_model] has more than one entry for CMT={}",
                    cmt
                ));
            }
        }
        return Ok((ParsedErrorModel::PerCmt(per_cmt), LtbsFlags::default()));
    }

    match singles.into_iter().next() {
        Some((model, names, flags)) => Ok((ParsedErrorModel::Single(model, names), flags)),
        None => Err("No error model found in [error_model] block".to_string()),
    }
}

/// Resolve a `ParsedErrorModel` into the `(representative ErrorModel, ErrorSpec)`
/// stored on `CompiledModel`. For `PerCmt`, sigma names are resolved to indices
/// into the flat global sigma vector (`sigma_names`, in `[parameters]` order)
/// and the feature is restricted to ODE models (Phase 1).
fn build_error_spec(
    parsed: ParsedErrorModel,
    sigma_names: &[String],
    is_ode: bool,
) -> Result<(ErrorModel, ErrorSpec), String> {
    match parsed {
        ParsedErrorModel::Single(model, names) => {
            // Single-endpoint sigmas are consumed positionally from the global
            // sigma vector, but a referenced name that was never declared in
            // [parameters] is a typo we should catch rather than silently bind
            // to sigma[0]. (Mirrors the strict resolution on the PerCmt path.)
            for nm in &names {
                if !sigma_names.iter().any(|s| s == nm) {
                    return Err(format!(
                        "[error_model] references unknown sigma '{}' \
                         (declare it in [parameters])",
                        nm
                    ));
                }
            }
            Ok((model, ErrorSpec::Single(model)))
        }
        ParsedErrorModel::PerCmt(entries) => {
            // An empty PerCmt arises for TTE-only models (no [error_model] block) —
            // allow it regardless of is_ode.  Non-empty PerCmt still requires ODE.
            if !entries.is_empty() && !is_ode {
                return Err(
                    "Per-CMT error models (`CMT=N: DV ~ ...`) require an ODE-based \
                     [structural_model]; analytical PK models support a single error \
                     model only."
                        .to_string(),
                );
            }
            let mut map = HashMap::new();
            let mut representative = None;
            for (cmt, em, names) in entries {
                let mut sigma_idx = Vec::with_capacity(names.len());
                for nm in &names {
                    let idx = sigma_names.iter().position(|s| s == nm).ok_or_else(|| {
                        format!(
                            "[error_model] CMT={}: references unknown sigma '{}' \
                             (declare it in [parameters])",
                            cmt, nm
                        )
                    })?;
                    sigma_idx.push(idx);
                }
                if representative.is_none() {
                    representative = Some(em);
                }
                map.insert(
                    cmt,
                    EndpointError {
                        error_model: em,
                        sigma_idx,
                    },
                );
            }
            Ok((
                representative.unwrap_or(ErrorModel::Additive),
                ErrorSpec::PerCmt(map),
            ))
        }
    }
}

// --- Individual parameter function builder ---

/// Build the PK parameter function from a parsed `[individual_parameters]`
/// statement list. The block may contain plain assignments, inline `if (...) ... else ...`
/// expressions, or full `if (...) { ... } else { ... }` statements.
///
/// `var_names` is the deduplicated list of all variables ever assigned in the
/// block (in first-occurrence order). For analytical PK models the assignment
/// order doubles as the slot ordering for `PkParams.values`.
/// Build the `pk_param_fn` closure used by every fit / simulate / predict
/// call site. When the `nn` feature is on and the model has any
/// `[covariate_nn]` blocks, `covariate_nns` carries each mapper plus the
/// offset of its weight block inside the optimizer's `theta` vector. The
/// closure pre-computes each NN's forward output once per call and exposes
/// them to the indexed evaluator via the `nn_outputs` slice.
fn build_pk_param_fn(
    stmts: Vec<Statement>,
    pk_param_map: &HashMap<String, String>,
    var_names: &[String],
    ode_slot_map: &[usize],
    n_theta_base: usize,
    n_eta_extended: usize,
    #[cfg(feature = "nn")] covariate_nns: &[crate::nn::CovariateNn],
) -> Result<(PkParamFn, Vec<String>, IndivParamPartials), String> {
    // Covariates referenced anywhere in the block (including inside if-bodies
    // and condition expressions). Sorted for deterministic error messages.
    let mut cov_set: std::collections::HashSet<String> = std::collections::HashSet::new();
    collect_covariates_in_stmts(&stmts, &mut cov_set);
    let mut referenced_covariates: Vec<String> = cov_set.into_iter().collect();
    referenced_covariates.sort();

    // Variable/covariate references switch from HashMap lookup to slot
    // index. Top-level vars come first so the ODE positional mapping
    // below stays valid; nested if-body vars get appended slots.
    let mut all_var_names: Vec<String> = var_names.to_vec();
    let nested_vars = assigned_vars_in_order(&stmts);
    for n in &nested_vars {
        if !all_var_names.iter().any(|m| m == n) {
            all_var_names.push(n.clone());
        }
    }
    let var_idx: HashMap<String, usize> = all_var_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();
    let cov_idx: HashMap<String, usize> = referenced_covariates
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();
    let n_vars = all_var_names.len();
    let n_cov = referenced_covariates.len();
    let mut stmts_resolved = stmts;

    // Compute symbolic partials BEFORE `resolve_variable_indices` consumes
    // the Expression nodes by bytecode-compiling them. `build_indiv_param_partials`
    // takes a `&[Statement]` and works on local clones (then runs
    // `resolve_expr_indices` itself); no need to defer the bytecode-compile
    // step or hold a parallel copy of the AST.
    let indiv_partials = build_indiv_param_partials(
        &stmts_resolved,
        &var_idx,
        &cov_idx,
        n_theta_base,
        n_eta_extended,
    );

    resolve_variable_indices(&mut stmts_resolved, &var_idx, &cov_idx, None);

    let stmts_owned = stmts_resolved;
    let vars_in_order = var_names.to_vec();

    // Pre-resolve pk_map → indexed (pk_slot, var_slot) pairs so the hot
    // loop is two array reads instead of two HashMap probes. Each structural
    // PK value is one of three things, in this precedence:
    //   1. a defined [individual_parameters] variable → bound by slot;
    //   2. a numeric literal (e.g. `ka=1.0`)           → bound as a constant;
    //   3. neither                                     → hard parse error.
    // The earlier `filter_map` silently `?`-dropped both the undefined-variable
    // (3) and the literal (2) cases, leaving the slot at `PkParams::default()`
    // (0.0 for everything but F). For an undefined reference that produced a
    // structurally-broken model that still "converged" with every prediction
    // floored to the log constant (#261); for a literal it silently meant the
    // value 0.0. Both now resolve correctly or error.
    //
    // Iterate in sorted key order so a model with several bad bindings always
    // reports the same one (HashMap iteration order is otherwise arbitrary).
    let mut pk_entries: Vec<(&String, &String)> = pk_param_map.iter().collect();
    pk_entries.sort_by(|a, b| a.0.cmp(b.0));
    let mut pk_assignment_mapping: Vec<(usize, usize)> = Vec::with_capacity(pk_entries.len());
    let mut pk_const_mapping: Vec<(usize, f64)> = Vec::new();
    for (pk_name, var_name) in pk_entries {
        let pk_slot = PkParams::name_to_index(pk_name).ok_or_else(|| {
            format!(
                "[structural_model] unknown PK parameter `{pk_name}`; valid names are \
                 cl, v/v1, q/q2, v2, ka, f, q3, v3, lagtime/alag"
            )
        })?;
        // Fall back to lowercase lookup — matches the previous
        // `vars.get(var_name.to_lowercase())` compat behaviour.
        let var_slot = var_idx
            .get(var_name)
            .copied()
            .or_else(|| var_idx.get(&var_name.to_lowercase()).copied());
        if let Some(var_slot) = var_slot {
            pk_assignment_mapping.push((pk_slot, var_slot));
        } else if let Ok(c) = var_name.parse::<f64>() {
            // A numeric literal binds the slot to a constant — but `f64::from_str`
            // also accepts `inf`/`nan`/`infinity`, which are never a meaningful PK
            // value. Reject them rather than binding a silently-degenerate
            // constant (the same silent-wrong default #261 set out to remove).
            if !c.is_finite() {
                return Err(format!(
                    "[structural_model] parameter `{pk_name}` has non-finite constant value \
                     `{var_name}`; use a finite number or a defined [individual_parameters] \
                     variable"
                ));
            }
            pk_const_mapping.push((pk_slot, c));
        } else {
            return Err(format!(
                "[structural_model] parameter `{pk_name}` references variable `{var_name}`, \
                 which is not defined in [individual_parameters] (defined: {}). \
                 Define it, e.g. `{var_name} = ...`.",
                var_names.join(", ")
            ));
        }
    }
    let is_analytical_pk = !pk_param_map.is_empty();

    // ODE branch counterpart of the analytical pre-resolution: map each
    // top-level individual parameter to (write_slot, var_slot). `write_slot`
    // is the parameter's `ode_param_slots` slot in `PkParams.values` (canonical
    // names → their PK slot, others → free non-reserved slots); `var_slot` is
    // where the evaluator leaves its value. The RHS reads back from the same
    // `write_slot`, so F lands at PK_IDX_F / lagtime at PK_IDX_LAGTIME with no
    // separate side-write and no risk of a structural parameter aliasing those
    // engine-reserved slots (issue #122). `vars_in_order` is parallel to
    // `ode_slot_map`.
    let ode_assignment_mapping: Vec<(usize, usize)> = vars_in_order
        .iter()
        .enumerate()
        .filter_map(|(i, var_name)| {
            let write_slot = ode_slot_map.get(i).copied()?;
            let var_slot = var_idx.get(var_name).copied()?;
            Some((write_slot, var_slot))
        })
        .collect();

    let cov_names_for_lookup = referenced_covariates.clone();

    // Snapshot the NN handles into the closure. Empty when no
    // `[covariate_nn]` blocks are present, in which case the per-call
    // forward-pass loop below is a no-op (just an empty `Vec<Vec<f64>>`
    // alloc — cheap enough to skip the branch).
    #[cfg(feature = "nn")]
    let covariate_nns_owned: Vec<crate::nn::CovariateNn> = covariate_nns.to_vec();

    let pk_param_fn: PkParamFn = Box::new(
        move |theta: &[f64], eta: &[f64], covariates: &HashMap<String, f64>| {
            // Materialise covariates into a Vec<f64> aligned with
            // `referenced_covariates`. For the typical 3-5 covariates
            // this is ~3-5 HashMap probes + one short alloc; cheaper
            // than the 10-20 probes the previous unresolved AST was
            // paying for both variables AND covariates.
            let mut cov_vec = vec![0.0_f64; n_cov];
            for (i, name) in cov_names_for_lookup.iter().enumerate() {
                cov_vec[i] = covariates.get(name).copied().unwrap_or(0.0);
            }
            let mut vars = vec![0.0_f64; n_vars];

            // Pre-compute each NN's forward output once per call. The
            // indexed evaluator reads `nn_outputs[nn_idx][output_idx]` for
            // every `Expression::NnOutput` it visits, so multiple `.CL`,
            // `.V1`, … references on the same NN share this single forward.
            #[cfg(feature = "nn")]
            let nn_outputs: Vec<Vec<f64>> = covariate_nns_owned
                .iter()
                .map(|nn| {
                    use crate::nn::CovariateMapper;
                    let n_w = nn.mapper.n_weights();
                    let weights = &theta[nn.weights_offset..nn.weights_offset + n_w];
                    nn.mapper.forward_raw(weights, covariates).expect(
                        "NN forward_raw failed in pk_param_fn: this indicates a \
                         weight-offset/length wiring bug (missing covariates \
                         are substituted with 0.0, not errored on)",
                    )
                })
                .collect();
            #[cfg(not(feature = "nn"))]
            let nn_outputs: Vec<Vec<f64>> = Vec::new();

            // pk_param_fn doesn't compute derivatives — no `du` to pass.
            eval_statements_indexed(
                &stmts_owned,
                theta,
                eta,
                &cov_vec,
                &mut vars,
                None,
                &nn_outputs,
            );

            let mut p = PkParams::default();
            if is_analytical_pk {
                for &(pk_slot, var_slot) in &pk_assignment_mapping {
                    p.values[pk_slot] = vars[var_slot];
                }
                // Literal-valued slots (e.g. `ka=1.0`) are constants — no
                // per-call evaluation, just write the parsed value.
                for &(pk_slot, c) in &pk_const_mapping {
                    p.values[pk_slot] = c;
                }
            } else {
                // ODE model: store each individual parameter at its
                // `ode_param_slots` slot (canonical names at their PK slot, F at
                // PK_IDX_F, lagtime at PK_IDX_LAGTIME, others at free slots).
                for &(slot, var_slot) in &ode_assignment_mapping {
                    p.values[slot] = vars[var_slot];
                }
            }
            p
        },
    );
    Ok((pk_param_fn, referenced_covariates, indiv_partials))
}

// --- Simple expression AST and evaluator ---
//
// Visibility note: `Expression` and its sibling enums (`BinOp`, `CmpOp`,
// `Condition`) are `pub(crate)` so the partial-derivative trees produced by
// the Tier 4a sensitivity work (`differentiate_with_chain`,
// `IndivParamPartials`) can be stored on `CompiledModel` (defined in
// `types.rs`). They remain unexported from the crate root — external users
// can't construct or pattern-match them.

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
    /// Dot-access on a `[covariate_nn NAME]` block's output, e.g.
    /// `TYPICAL_PK.CL`. `nn_idx` indexes into `CompiledModel.covariate_nns`
    /// (deterministic alphabetical order set at parse time); `output_idx`
    /// indexes into the block's declared `outputs` list. Eval-time dispatch
    /// reads the pre-computed forward output from a per-call cache in
    /// `build_pk_param_fn` so multiple references to outputs of the same
    /// NN share a single forward pass.
    NnOutput {
        nn_idx: usize,
        output_idx: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    /// Euclidean remainder — result always non-negative for positive divisor.
    Mod,
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
enum Statement {
    Assign(String, Expression),
    /// Same as `Assign(name, expr)` but pre-resolved to a slot index, before
    /// bytecode compilation. Currently transitional — `resolve_variable_indices`
    /// goes straight from `Assign` to `AssignBc` — kept around to keep the
    /// pre-/post-resolve pair symmetric with `DiffEq` / `DiffEqIdx` and so an
    /// older AST snapshot fed to `eval_statements_indexed` still evaluates
    /// correctly via the slower expression-tree fallback arm.
    #[allow(dead_code)]
    AssignIdx(usize, Expression),
    /// `d/dt(NAME) = expr` — only legal in `[odes]` blocks.
    DiffEq(String, Expression),
    /// Same as `DiffEq(name, expr)` but pre-resolved to the state's slot in the
    /// `du` array. Transitional like `AssignIdx` — kept for fallback evaluation
    /// of any unresolved AST snapshot.
    #[allow(dead_code)]
    DiffEqIdx(usize, Expression),
    /// Bytecode-compiled assignment. Emitted by `resolve_variable_indices`
    /// after `AssignIdx`'s expression has been compiled to `Bytecode` for
    /// the hot-path `eval_bytecode` interpreter.
    AssignBc(usize, Bytecode),
    /// Bytecode-compiled derivative assignment; sibling of `AssignBc`.
    DiffEqBc(usize, Bytecode),
    /// One or more `if (cond) { ... }` arms followed by an optional `else { ... }`.
    /// Each arm in `branches` is `(condition, body)`.
    If {
        branches: Vec<(Condition, Vec<Statement>)>,
        else_body: Option<Vec<Statement>>,
    },
}

/// Context threaded through the recursive-descent parser so that every bare
/// identifier can be classified as Theta / Eta / Variable / Covariate without
/// relying on casing heuristics.
#[derive(Clone, Copy)]
struct ParseCtx<'a> {
    theta_names: &'a [String],
    eta_names: &'a [String],
    /// Names previously assigned in the surrounding block (e.g. earlier lines
    /// of [individual_parameters]). These resolve to `Variable`.
    defined_vars: &'a [String],
    /// When `true` (the usual case), an unknown identifier is a covariate.
    /// Set to `false` for the ODE RHS parser, where state names and individual
    /// parameters are injected into the `vars` map at eval time instead.
    fallback_covariate: bool,
    /// Per-NN-block (name, output_names) pairs in alphabetical order.
    /// `nn_idx` in `Expression::NnOutput` indexes into this slice. Empty
    /// when no `[covariate_nn]` block is present in the model (the
    /// dot-access parser then errors on `FOO.BAR`).
    nn_specs: &'a [(String, Vec<String>)],
    /// ODE state variable names for detecting compartment references in
    /// integral integrands (`uses_compartments` flag). Empty for analytical models.
    ode_state_names: &'a [String],
}

impl<'a> ParseCtx<'a> {
    fn new(theta_names: &'a [String], eta_names: &'a [String], defined_vars: &'a [String]) -> Self {
        const EMPTY_NN: &[(String, Vec<String>)] = &[];
        const EMPTY: &[String] = &[];
        Self {
            theta_names,
            eta_names,
            defined_vars,
            fallback_covariate: true,
            nn_specs: EMPTY_NN,
            ode_state_names: EMPTY,
        }
    }

    fn ode(defined_vars: &'a [String]) -> Self {
        const EMPTY: &[String] = &[];
        const EMPTY_NN: &[(String, Vec<String>)] = &[];
        Self {
            theta_names: EMPTY,
            eta_names: EMPTY,
            defined_vars,
            fallback_covariate: false,
            nn_specs: EMPTY_NN,
            ode_state_names: EMPTY,
        }
    }

    fn with_nn_specs(mut self, nn_specs: &'a [(String, Vec<String>)]) -> Self {
        self.nn_specs = nn_specs;
        self
    }
}

/// Pre-order walk over every node of an expression tree, invoking `f` on each.
/// This single traversal backs the `collect_covariates`, `collect_undefined_vars`,
/// and `collect_theta_eta` families below — each is a thin wrapper supplying a
/// leaf-matching closure. Adding a name-bearing `Expression` variant therefore
/// only requires updating this walker (plus `visit_condition_nodes` /
/// `visit_stmt_nodes`), not every collector.
fn visit_expr_nodes(expr: &Expression, f: &mut dyn FnMut(&Expression)) {
    f(expr);
    match expr {
        Expression::BinOp(lhs, _, rhs) => {
            visit_expr_nodes(lhs, f);
            visit_expr_nodes(rhs, f);
        }
        Expression::UnaryFn(_, arg) => visit_expr_nodes(arg, f),
        Expression::Power(base, exp) => {
            visit_expr_nodes(base, f);
            visit_expr_nodes(exp, f);
        }
        Expression::Conditional(cond, t, e) => {
            visit_condition_nodes(cond, f);
            visit_expr_nodes(t, f);
            visit_expr_nodes(e, f);
        }
        _ => {}
    }
}

/// Walk every expression embedded in a condition (see `visit_expr_nodes`).
fn visit_condition_nodes(cond: &Condition, f: &mut dyn FnMut(&Expression)) {
    match cond {
        Condition::Compare(l, _, r) => {
            visit_expr_nodes(l, f);
            visit_expr_nodes(r, f);
        }
        Condition::And(l, r) | Condition::Or(l, r) => {
            visit_condition_nodes(l, f);
            visit_condition_nodes(r, f);
        }
        Condition::Not(c) => visit_condition_nodes(c, f),
    }
}

/// Walk every expression in a statement list — assignment/diff-eq RHSs and the
/// conditions + bodies of `if` blocks (see `visit_expr_nodes`). Bytecode
/// variants carry no tree to walk (they only appear after
/// `resolve_variable_indices`).
fn visit_stmt_nodes(stmts: &[Statement], f: &mut dyn FnMut(&Expression)) {
    for s in stmts {
        match s {
            Statement::Assign(_, e)
            | Statement::AssignIdx(_, e)
            | Statement::DiffEq(_, e)
            | Statement::DiffEqIdx(_, e) => visit_expr_nodes(e, f),
            Statement::AssignBc(_, _) | Statement::DiffEqBc(_, _) => {}
            Statement::If {
                branches,
                else_body,
            } => {
                for (cond, body) in branches {
                    visit_condition_nodes(cond, f);
                    visit_stmt_nodes(body, f);
                }
                if let Some(eb) = else_body {
                    visit_stmt_nodes(eb, f);
                }
            }
        }
    }
}

/// Accumulate every covariate name referenced in an expression.
fn collect_covariates(expr: &Expression, out: &mut std::collections::HashSet<String>) {
    visit_expr_nodes(expr, &mut |e: &Expression| {
        if let Expression::Covariate(name) = e {
            out.insert(name.clone());
        }
    });
}

/// Accumulate every covariate name referenced across a statement list.
fn collect_covariates_in_stmts(stmts: &[Statement], out: &mut std::collections::HashSet<String>) {
    visit_stmt_nodes(stmts, &mut |e: &Expression| {
        if let Expression::Covariate(name) = e {
            out.insert(name.clone());
        }
    });
}

/// Accumulate every `Variable(name)` in an expression whose name is not a key in
/// `defined` — i.e. a name that would resolve to the `usize::MAX` "reads 0.0"
/// sentinel in the ODE RHS bytecode (or `vars.get(name).unwrap_or(0.0)` in an
/// `init` expression). Used to reject undefined references in the `[odes]` block
/// before they silently corrupt the dynamics (issue #314). Membership is by
/// exact key: the ODE var maps already carry lower/upper/original aliases for
/// every resolvable name, so a name absent from `defined` genuinely cannot
/// resolve.
fn collect_undefined_vars(
    expr: &Expression,
    defined: &HashMap<String, usize>,
    out: &mut std::collections::HashSet<String>,
) {
    visit_expr_nodes(expr, &mut |e: &Expression| {
        if let Expression::Variable(name) = e {
            if !defined.contains_key(name) {
                out.insert(name.clone());
            }
        }
    });
}

/// Accumulate undefined `Variable` names (see `collect_undefined_vars`) across a
/// statement list — a d/dt RHS, an intermediate assignment, or an if-condition.
fn collect_undefined_vars_in_stmts(
    stmts: &[Statement],
    defined: &HashMap<String, usize>,
    out: &mut std::collections::HashSet<String>,
) {
    visit_stmt_nodes(stmts, &mut |e: &Expression| {
        if let Expression::Variable(name) = e {
            if !defined.contains_key(name) {
                out.insert(name.clone());
            }
        }
    });
}

/// Accumulate the theta and eta indices referenced in an expression. Only the
/// statement-level variant is used outside the `survival` feature, so this
/// expression-level entry is gated to its sole caller (`parse_event_model_block`).
#[cfg(feature = "survival")]
fn collect_theta_eta(
    expr: &Expression,
    thetas: &mut std::collections::HashSet<usize>,
    etas: &mut std::collections::HashSet<usize>,
) {
    visit_expr_nodes(expr, &mut |e: &Expression| match e {
        Expression::Theta(i) => {
            thetas.insert(*i);
        }
        Expression::Eta(i) => {
            etas.insert(*i);
        }
        _ => {}
    });
}

/// Accumulate the theta and eta indices referenced across a statement list.
fn collect_theta_eta_in_stmts(
    stmts: &[Statement],
    thetas: &mut std::collections::HashSet<usize>,
    etas: &mut std::collections::HashSet<usize>,
) {
    visit_stmt_nodes(stmts, &mut |e: &Expression| match e {
        Expression::Theta(i) => {
            thetas.insert(*i);
        }
        Expression::Eta(i) => {
            etas.insert(*i);
        }
        _ => {}
    });
}

/// Collect the sigma names referenced in a `ParsedErrorModel` (before index resolution).
fn used_sigma_names(parsed: &ParsedErrorModel) -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    match parsed {
        ParsedErrorModel::Single(_, names) => {
            for n in names {
                out.insert(n.clone());
            }
        }
        ParsedErrorModel::PerCmt(entries) => {
            for (_, _, names) in entries {
                for n in names {
                    out.insert(n.clone());
                }
            }
        }
    }
    out
}

/// Warn about parameters declared in `[parameters]` that are not referenced in
/// any expression. Returns one warning string per unused parameter; the caller
/// appends these to `CompiledModel.parse_warnings`.
///
/// Only user-declared thetas (indices `0..n_theta_user`) are checked; thetas
/// added automatically (NN weights, diffusion) are excluded.
///
/// Scanning `indiv_stmts` is sufficient for ODE models too: `build_ode_spec`
/// uses `ParseCtx::ode` which sets `theta_names = []`, so raw theta/eta names
/// cannot appear in `[odes]` RHS expressions — they must be routed through
/// `[individual_parameters]` first.
fn check_unused_parameters(
    thetas: &[ThetaSpec],
    eta_names_bsv: &[String],
    kappa_names: &[String],
    n_eta: usize,
    sigma_names: &[String],
    indiv_stmts: &[Statement],
    used_sigmas: &std::collections::HashSet<String>,
    event_model_thetas: &std::collections::HashSet<usize>,
    event_model_etas: &std::collections::HashSet<usize>,
) -> Vec<String> {
    let mut used_thetas = std::collections::HashSet::new();
    let mut used_etas = std::collections::HashSet::new();
    collect_theta_eta_in_stmts(indiv_stmts, &mut used_thetas, &mut used_etas);
    // Union in parameters used in [event_model] so mixed PK+TTE models do not
    // produce false "not referenced" warnings for hazard-model thetas/etas.
    used_thetas.extend(event_model_thetas.iter().copied());
    used_etas.extend(event_model_etas.iter().copied());

    let mut warnings = Vec::new();

    for (i, t) in thetas.iter().enumerate() {
        if !used_thetas.contains(&i) {
            warnings.push(format!(
                "theta '{}' is declared in [parameters] but not referenced in any \
                 model expression — it will not affect predictions or be \
                 meaningfully estimated",
                t.name
            ));
        }
    }
    for (i, name) in eta_names_bsv.iter().enumerate() {
        if !used_etas.contains(&i) {
            warnings.push(format!(
                "omega '{}' is declared in [parameters] but not referenced in any \
                 model expression — it will not affect predictions or be \
                 meaningfully estimated",
                name
            ));
        }
    }
    for (i, name) in kappa_names.iter().enumerate() {
        if !used_etas.contains(&(n_eta + i)) {
            warnings.push(format!(
                "kappa '{}' is declared in [parameters] but not referenced in any \
                 model expression — it will not affect predictions or be \
                 meaningfully estimated",
                name
            ));
        }
    }
    for name in sigma_names {
        if !used_sigmas.contains(name) {
            warnings.push(format!(
                "sigma '{}' is declared in [parameters] but not referenced in \
                 [error_model] — it will not affect predictions or be \
                 meaningfully estimated",
                name
            ));
        }
    }

    warnings
}

fn eval_expression(
    expr: &Expression,
    theta: &[f64],
    eta: &[f64],
    covariates: &HashMap<String, f64>,
    vars: &HashMap<String, f64>,
    nn_outputs: &[Vec<f64>],
) -> f64 {
    match expr {
        Expression::Literal(v) => *v,
        Expression::Theta(i) => theta[*i],
        Expression::Eta(i) => eta[*i],
        Expression::Covariate(name) => covariates.get(name).copied().unwrap_or(0.0),
        Expression::Variable(name) => {
            if name.eq_ignore_ascii_case("MACHEPS") {
                f64::EPSILON
            } else {
                vars.get(name).copied().unwrap_or(0.0)
            }
        }
        Expression::VariableIdx(_) | Expression::CovariateIdx(_) => {
            debug_assert!(
                false,
                "indexed expression reached unindexed eval_expression"
            );
            0.0
        }
        Expression::BinOp(lhs, op, rhs) => {
            let l = eval_expression(lhs, theta, eta, covariates, vars, nn_outputs);
            let r = eval_expression(rhs, theta, eta, covariates, vars, nn_outputs);
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
                BinOp::Mod => l.rem_euclid(r),
            }
        }
        Expression::UnaryFn(name, arg) => {
            let v = eval_expression(arg, theta, eta, covariates, vars, nn_outputs);
            match name.as_str() {
                "exp" => v.exp(),
                "log" | "ln" => v.max(1e-30).ln(),
                "sqrt" => v.max(0.0).sqrt(),
                "abs" => v.abs(),
                "floor" => v.floor(),
                "ceil" => v.ceil(),
                "round" => v.round(),
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
            let b = eval_expression(base, theta, eta, covariates, vars, nn_outputs);
            let e = eval_expression(exp, theta, eta, covariates, vars, nn_outputs);
            b.powf(e)
        }
        Expression::Conditional(cond, t, e) => {
            if eval_condition(cond, theta, eta, covariates, vars, nn_outputs) {
                eval_expression(t, theta, eta, covariates, vars, nn_outputs)
            } else {
                eval_expression(e, theta, eta, covariates, vars, nn_outputs)
            }
        }
        Expression::NnOutput { nn_idx, output_idx } => {
            // Same per-call cache as the indexed evaluator. Callers
            // populate `nn_outputs` from `NamedMlpMapper::forward_raw`
            // (see the `tv_fn` closure for the eta=0 path). Out-of-bounds
            // indices return 0.0 with a debug-assert so a logic bug
            // surfaces in tests but doesn't crash release builds.
            nn_outputs
                .get(*nn_idx)
                .and_then(|v| v.get(*output_idx))
                .copied()
                .unwrap_or_else(|| {
                    debug_assert!(
                        false,
                        "NnOutput nn_idx={nn_idx} output_idx={output_idx} out of bounds in unindexed eval"
                    );
                    0.0
                })
        }
    }
}

fn eval_condition(
    cond: &Condition,
    theta: &[f64],
    eta: &[f64],
    covariates: &HashMap<String, f64>,
    vars: &HashMap<String, f64>,
    nn_outputs: &[Vec<f64>],
) -> bool {
    match cond {
        Condition::Compare(l, op, r) => {
            let lv = eval_expression(l, theta, eta, covariates, vars, nn_outputs);
            let rv = eval_expression(r, theta, eta, covariates, vars, nn_outputs);
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
            eval_condition(l, theta, eta, covariates, vars, nn_outputs)
                && eval_condition(r, theta, eta, covariates, vars, nn_outputs)
        }
        Condition::Or(l, r) => {
            eval_condition(l, theta, eta, covariates, vars, nn_outputs)
                || eval_condition(r, theta, eta, covariates, vars, nn_outputs)
        }
        Condition::Not(c) => !eval_condition(c, theta, eta, covariates, vars, nn_outputs),
    }
}

// ─── Bytecode interpreter ───────────────────────────────────────────────────
//
// Flat-bytecode replacement for the recursive `eval_expression_indexed` AST
// walker. Profiling the experiment Emax FOCEI fit (#134) showed
// `eval_expression_indexed` was ~55% of self time after PR #135/#136 —
// dominated by `Box<Expression>` chasing and recursive function-call
// overhead. Lowering each parsed expression to a `Vec<Op>` + small f64
// stack flattens the walk into a tight match-on-tag loop with predictable
// branches and good cache locality.
//
// Compilation is post-resolve (every `Variable(name)` has already been
// turned into `VariableIdx(slot)` by `resolve_variable_indices`), so the
// Op variants carry slot indices directly. Constants live in a small
// pool so the Op enum stays compact (u32-indexed instead of an f64
// payload).
//
// Conditionals lower to forward jumps:
//   if (cond) { then } else { else }
//     [cond bytecode pushing 0.0 / 1.0]
//     JumpIfFalse(L_else)
//     [then bytecode]
//     Jump(L_end)
//   L_else: [else bytecode]
//   L_end:
//
// And/Or/Not are non-short-circuit (the underlying expressions never have
// side-effects worth short-circuiting: arithmetic only, with `Op::Div`'s
// `r.abs() < 1e-30 -> 0.0` guard and `Op::Ln`/`Op::Sqrt`'s clamps
// preserving the slow-path semantics).

#[derive(Debug, Clone, Copy)]
enum Op {
    PushConst(u32), // index into Bytecode.constants
    PushTheta(u32),
    PushEta(u32),
    PushVar(u32),
    PushCov(u32),
    PushNnOutput(u32, u32),
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Exp,
    Ln,   // matches eval's `v.max(1e-30).ln()` guard
    Sqrt, // matches eval's `v.max(0.0).sqrt()` guard
    Abs,
    InvLogit,
    Logit,
    CmpLt,
    CmpLe,
    CmpGt,
    CmpGe,
    CmpEq,
    CmpNe,
    // And/Or/Not consume 0.0 / 1.0 booleans on the stack.
    LogicAnd,
    LogicOr,
    LogicNot,
    JumpIfFalse(u32), // pops top; jumps to bytecode index if value == 0.0
    Jump(u32),
    Mod,   // rem_euclid (binary)
    Floor, // unary
    Ceil,  // unary
    Round, // unary
}

// ─── Consolidated per-thread scratch ───────────────────────────────────────
//
// All hot-path closures in this module need their own `Vec<f64>` scratch:
//   - `build_ode_rhs_fn`        : `rhs_vars` (state ‖ indiv params ‖ inters)
//   - `build_y_output_fn`       : `y_vars` + `y_cov` (Form C readout)
//   - `eval_statements_indexed` : `bc_stack` (bytecode f64 stack)
//   - `build_y_output_fn`       : also needs the bc_stack for its readout eval
//
// Each was originally its own `thread_local!`. A samply profile of the
// experiment Emax FOCEI fit after #135/#136/#137 attributed ~12% of self
// time to `std::thread::LocalKey::with` — each closure call paid 1-3 TLS
// lookups. Consolidating to a single TLS holding a struct with named
// fields means one `LocalKey::with` + one `RefCell::borrow_mut` per call,
// and the BC stack is shared between RHS and Y-readout calls (which never
// interleave on a single thread — the Y readout runs post-integration).
#[derive(Debug)]
struct FerxThreadScratch {
    rhs_vars: Vec<f64>,
    y_vars: Vec<f64>,
    y_cov: Vec<f64>,
    bc_stack: Vec<f64>,
}

impl FerxThreadScratch {
    const fn new_empty() -> Self {
        Self {
            rhs_vars: Vec::new(),
            y_vars: Vec::new(),
            y_cov: Vec::new(),
            bc_stack: Vec::new(),
        }
    }
}

thread_local! {
    static FERX_SCRATCH: std::cell::RefCell<FerxThreadScratch> =
        const { std::cell::RefCell::new(FerxThreadScratch::new_empty()) };
}

/// Compiled bytecode for a single expression tree. The `constants` pool
/// holds every literal value referenced by `PushConst(u32)` opcodes; this
/// avoids paying an 8-byte literal payload on every variant of `Op`. The
/// enum size is currently 12 bytes (`PushNnOutput(u32, u32)` carries 8
/// bytes of payload plus the tag — every variant pads to that size); the
/// pool still saves 4 bytes per push relative to a hypothetical
/// `PushLiteral(f64)` (which would force 16-byte alignment). `max_stack`
/// is computed at compile time so `eval_bytecode` can `reserve` once and
/// amortize the `Vec::push` bounds checks across all calls.
#[derive(Debug, Clone)]
struct Bytecode {
    ops: Vec<Op>,
    constants: Vec<f64>,
    max_stack: usize,
}

impl Bytecode {
    fn new() -> Self {
        Self {
            ops: Vec::new(),
            constants: Vec::new(),
            max_stack: 0,
        }
    }
    fn push_const(&mut self, v: f64) {
        let idx = self.constants.len() as u32;
        self.constants.push(v);
        self.ops.push(Op::PushConst(idx));
    }
}

/// Compile a single (post-resolve) `Expression` into `Bytecode`. Unresolved
/// `Variable` / `Covariate` nodes trip a `debug_assert!(false, …)` in debug
/// builds and lower to a constant `0.0` push in release; both are intended
/// to mirror the `eval_expression_indexed` fall-through semantics for
/// callers that bypass `resolve_expr_indices`.
fn compile_bytecode(expr: &Expression) -> Bytecode {
    let mut bc = Bytecode::new();
    compile_expr_into(&mut bc, expr);
    bc.max_stack = compute_max_stack(&bc.ops);
    bc
}

/// Compute a safe upper bound on the maximum f64 stack depth a bytecode
/// program reaches. Each Op has a known net stack delta:
///   Push*       :  +1
///   Pop2/Push1  :  -1  (arithmetic / compare / logic-binary)
///   Pop1/Push1  :   0  (unary fn / logic-not)
///   Jump        :   0
///   JumpIfFalse :  -1
///
/// A single linear scan ([`scan_stack_depth`]) returns an *upper bound* (not
/// the exact peak): `Conditional` emits both then- and else-branches inline
/// with a `Jump` between them, so the linear walk credits BOTH branches'
/// pushes against running depth even though execution only takes one branch at
/// runtime. That over-estimate is exactly what `eval_bytecode` wants for its
/// `stack.reserve(max_stack)` call — under-estimating here would let the
/// unchecked-write hot loop go OOB on the conservative-FD path. The
/// `depth >= 0` and balanced-end debug asserts catch a future opcode
/// addition that violates the invariant; in release builds the over-bound
/// is the only guard.
///
/// If backward jumps (e.g. for loops) are ever added, this linear-scan
/// algorithm no longer holds — fixed-point iteration would be required.
fn compute_max_stack(ops: &[Op]) -> usize {
    let (peak, depth) = scan_stack_depth(ops);
    // A well-formed *jump-free* expression leaves exactly one value on the
    // stack (the result); this catches off-by-one push/pop emissions in any
    // future `compile_expr_into` change. Bytecode with branches can't be
    // checked this way (the linear scan walks both arms — see
    // `bytecode_has_branch`), so the predicate exempts them. `peak` stays a
    // safe over-estimate either way.
    debug_assert!(
        ends_at_expected_depth(ops, depth),
        "compute_max_stack: bytecode ends at depth {depth}, expected 1",
    );
    peak.max(1) as usize
}

/// Single linear pass over `ops` returning `(peak, end_depth)`: the maximum
/// running f64-stack depth and the depth left after the final op. Split out
/// from [`compute_max_stack`] so the end-depth invariant can be checked on
/// *real* compiled bytecode from a unit test (see
/// `compute_max_stack_jumpfree_bytecode_ends_at_depth_one`) even under the
/// `ci-test` profile, where the `debug_assert!` that normally guards it is
/// compiled out. Returning `end_depth` — rather than only consuming it inside
/// that `debug_assert!` — is what lets a future `compile_expr_into` off-by-one
/// be caught in CI, not just under the local dev profile.
fn scan_stack_depth(ops: &[Op]) -> (i32, i32) {
    let mut depth: i32 = 0;
    let mut peak: i32 = 0;
    for op in ops {
        let delta: i32 = match op {
            Op::PushConst(_)
            | Op::PushTheta(_)
            | Op::PushEta(_)
            | Op::PushVar(_)
            | Op::PushCov(_)
            | Op::PushNnOutput(_, _) => 1,
            Op::Add
            | Op::Sub
            | Op::Mul
            | Op::Div
            | Op::Mod
            | Op::Pow
            | Op::CmpLt
            | Op::CmpLe
            | Op::CmpGt
            | Op::CmpGe
            | Op::CmpEq
            | Op::CmpNe
            | Op::LogicAnd
            | Op::LogicOr => -1,
            Op::Exp
            | Op::Ln
            | Op::Sqrt
            | Op::Abs
            | Op::InvLogit
            | Op::Logit
            | Op::LogicNot
            | Op::Floor
            | Op::Ceil
            | Op::Round => 0,
            Op::JumpIfFalse(_) => -1,
            Op::Jump(_) => 0,
        };
        depth += delta;
        // Liveness invariant: every Op's deltas must keep the running
        // (linear-scan) depth ≥ 0 — if not, the compiler emitted an
        // unbalanced sequence and `eval_bytecode`'s `pop!` macro would
        // underflow `len` (usize) into UB territory in release.
        debug_assert!(
            depth >= 0,
            "compute_max_stack: depth went negative at op {op:?}; bytecode is unbalanced",
        );
        if depth > peak {
            peak = depth;
        }
    }
    (peak, depth)
}

/// True if `ops` contains a branch (`Jump` / `JumpIfFalse`). The end-of-scan
/// depth invariant in [`compute_max_stack`] only holds for branch-free
/// bytecode: a `Conditional` emits both arms inline, so the linear walk ends
/// above depth 1 even though execution takes one arm. Extracted (and
/// unit-tested directly) so the branch check is exercised independently of the
/// debug-only assertion that consumes it.
fn bytecode_has_branch(ops: &[Op]) -> bool {
    ops.iter()
        .any(|op| matches!(op, Op::Jump(_) | Op::JumpIfFalse(_)))
}

/// Whether a completed linear scan of `ops` ending at `depth` satisfies the
/// well-formed end-depth invariant: a jump-free expression must leave exactly
/// one value on the stack. Branchy bytecode is exempt — a `Conditional` emits
/// both arms inline, so the linear walk ends above depth 1 even though
/// execution takes one arm (see [`bytecode_has_branch`]). The cheap `depth`
/// check is tried first so the common jump-free path skips the O(n) scan.
///
/// Returned rather than inlined into the `debug_assert!` so the invariant is
/// exercised even under the `ci-test` profile, where `debug_assert!` is
/// compiled out and the assertion itself never runs.
fn ends_at_expected_depth(ops: &[Op], depth: i32) -> bool {
    depth == 1 || ops.is_empty() || bytecode_has_branch(ops)
}

fn compile_expr_into(bc: &mut Bytecode, expr: &Expression) {
    match expr {
        Expression::Literal(v) => bc.push_const(*v),
        Expression::Theta(i) => bc.ops.push(Op::PushTheta(*i as u32)),
        Expression::Eta(i) => bc.ops.push(Op::PushEta(*i as u32)),
        Expression::VariableIdx(i) => bc.ops.push(Op::PushVar(*i as u32)),
        Expression::CovariateIdx(i) => bc.ops.push(Op::PushCov(*i as u32)),
        Expression::Variable(_) | Expression::Covariate(_) => {
            // Reached only if `resolve_expr_indices` was skipped — shouldn't
            // happen in practice; lower to 0.0 to preserve the
            // `eval_expression_indexed` fall-through behaviour.
            debug_assert!(false, "compile_bytecode: unresolved Variable/Covariate");
            bc.push_const(0.0);
        }
        Expression::NnOutput { nn_idx, output_idx } => bc
            .ops
            .push(Op::PushNnOutput(*nn_idx as u32, *output_idx as u32)),
        Expression::BinOp(lhs, op, rhs) => {
            compile_expr_into(bc, lhs);
            compile_expr_into(bc, rhs);
            bc.ops.push(match op {
                BinOp::Add => Op::Add,
                BinOp::Sub => Op::Sub,
                BinOp::Mul => Op::Mul,
                BinOp::Div => Op::Div,
                BinOp::Mod => Op::Mod,
            });
        }
        Expression::Power(base, exp) => {
            compile_expr_into(bc, base);
            compile_expr_into(bc, exp);
            bc.ops.push(Op::Pow);
        }
        Expression::UnaryFn(name, arg) => {
            compile_expr_into(bc, arg);
            // Names matched here mirror `eval_expression_indexed`'s UnaryFn
            // dispatch. Anything else becomes a no-op (the slow path returns
            // the argument unchanged); preserve that with `Op::Abs` of `Abs`
            // we can't — fall through to push the value as-is.
            match name.as_str() {
                "exp" => bc.ops.push(Op::Exp),
                "log" | "ln" => bc.ops.push(Op::Ln),
                "sqrt" => bc.ops.push(Op::Sqrt),
                "abs" => bc.ops.push(Op::Abs),
                "inv_logit" | "expit" => bc.ops.push(Op::InvLogit),
                "logit" => bc.ops.push(Op::Logit),
                "floor" => bc.ops.push(Op::Floor),
                "ceil" => bc.ops.push(Op::Ceil),
                "round" => bc.ops.push(Op::Round),
                _ => { /* unknown function → leave the argument on the stack */ }
            }
        }
        Expression::Conditional(cond, t_expr, e_expr) => {
            // Compile condition (leaves 0.0/1.0 on stack), then JumpIfFalse
            // to the else block, then the then-block + Jump-to-end, then
            // the else-block. The jumps are patched once we know the
            // target indices.
            compile_condition_into(bc, cond);
            let jif_idx = bc.ops.len();
            bc.ops.push(Op::JumpIfFalse(0)); // placeholder
            compile_expr_into(bc, t_expr);
            let jmp_idx = bc.ops.len();
            bc.ops.push(Op::Jump(0)); // placeholder
            let else_target = bc.ops.len() as u32;
            compile_expr_into(bc, e_expr);
            let end_target = bc.ops.len() as u32;
            bc.ops[jif_idx] = Op::JumpIfFalse(else_target);
            bc.ops[jmp_idx] = Op::Jump(end_target);
        }
    }
}

fn compile_condition_into(bc: &mut Bytecode, cond: &Condition) {
    match cond {
        Condition::Compare(l, op, r) => {
            compile_expr_into(bc, l);
            compile_expr_into(bc, r);
            bc.ops.push(match op {
                CmpOp::Lt => Op::CmpLt,
                CmpOp::Le => Op::CmpLe,
                CmpOp::Gt => Op::CmpGt,
                CmpOp::Ge => Op::CmpGe,
                CmpOp::Eq => Op::CmpEq,
                CmpOp::Ne => Op::CmpNe,
            });
        }
        Condition::And(l, r) => {
            compile_condition_into(bc, l);
            compile_condition_into(bc, r);
            bc.ops.push(Op::LogicAnd);
        }
        Condition::Or(l, r) => {
            compile_condition_into(bc, l);
            compile_condition_into(bc, r);
            bc.ops.push(Op::LogicOr);
        }
        Condition::Not(c) => {
            compile_condition_into(bc, c);
            bc.ops.push(Op::LogicNot);
        }
    }
}

/// Stack-machine bytecode evaluator. `stack` is reused across calls via the
/// caller's thread-local scratch; it's cleared at entry and the bytecode's
/// pre-computed `max_stack` is reserved up front so no `Vec::push` can
/// reallocate inside the hot loop.
///
/// We use a manual `len` cursor + raw pointer to the underlying buffer
/// instead of `Vec::push`/`pop` — `compute_max_stack` guarantees the
/// indices are in bounds, so the per-op bounds checks `Vec::push`/`pop`
/// emit are pure overhead here. The AST walker the bytecode replaces had no
/// such bounds checks (it just returned values from recursive calls), so
/// matching its overhead is the whole point.
fn eval_bytecode(
    bc: &Bytecode,
    theta: &[f64],
    eta: &[f64],
    covariates: &[f64],
    vars: &[f64],
    nn_outputs: &[Vec<f64>],
    stack: &mut Vec<f64>,
) -> f64 {
    stack.clear();
    stack.reserve(bc.max_stack);
    let mut pc: usize = 0;
    let ops = bc.ops.as_slice();
    let consts = bc.constants.as_slice();

    // SAFETY: `compute_max_stack` walks every op in compile order and the
    // compile-time stack-depth invariant holds for any well-formed Bytecode
    // (every variant we emit balances correctly). `stack.reserve(max_stack)`
    // guarantees the buffer is large enough for all subsequent unchecked
    // writes; `len` mirrors how many slots are actually live. Any malformed
    // Bytecode (e.g. produced by a future feature without a matching
    // `compute_max_stack` update) trips a debug_assert.
    let buf = stack.as_mut_ptr();
    let cap = stack.capacity();
    let mut len: usize = 0;
    macro_rules! push {
        ($v:expr) => {{
            debug_assert!(len < cap, "bytecode stack overflow at pc={pc}");
            unsafe {
                *buf.add(len) = $v;
            }
            len += 1;
        }};
    }
    macro_rules! pop {
        () => {{
            debug_assert!(len > 0, "bytecode stack underflow at pc={pc}");
            len -= 1;
            unsafe { *buf.add(len) }
        }};
    }

    while pc < ops.len() {
        match ops[pc] {
            Op::PushConst(i) => push!(consts[i as usize]),
            Op::PushTheta(i) => push!(theta.get(i as usize).copied().unwrap_or(0.0)),
            Op::PushEta(i) => push!(eta.get(i as usize).copied().unwrap_or(0.0)),
            Op::PushVar(i) => push!(vars.get(i as usize).copied().unwrap_or(0.0)),
            Op::PushCov(i) => push!(covariates.get(i as usize).copied().unwrap_or(0.0)),
            Op::PushNnOutput(nn_i, out_i) => {
                let v = nn_outputs
                    .get(nn_i as usize)
                    .and_then(|v| v.get(out_i as usize))
                    .copied()
                    .unwrap_or_else(|| {
                        debug_assert!(
                            false,
                            "Op::PushNnOutput nn_idx={nn_i} output_idx={out_i} out of bounds"
                        );
                        0.0
                    });
                push!(v);
            }
            Op::Add => {
                let b = pop!();
                let a = pop!();
                push!(a + b);
            }
            Op::Sub => {
                let b = pop!();
                let a = pop!();
                push!(a - b);
            }
            Op::Mul => {
                let b = pop!();
                let a = pop!();
                push!(a * b);
            }
            Op::Div => {
                let b = pop!();
                let a = pop!();
                push!(if b.abs() < 1e-30 { 0.0 } else { a / b });
            }
            Op::Pow => {
                let e = pop!();
                let b = pop!();
                push!(b.powf(e));
            }
            Op::Mod => {
                let b = pop!();
                let a = pop!();
                push!(if b.abs() < 1e-30 {
                    0.0
                } else {
                    a.rem_euclid(b)
                });
            }
            Op::Exp => {
                let v = pop!();
                push!(v.exp());
            }
            Op::Ln => {
                let v = pop!();
                let v = if v >= 1e-30 { v } else { 1e-30 };
                push!(v.ln());
            }
            Op::Sqrt => {
                let v = pop!();
                let v = if v >= 0.0 { v } else { 0.0 };
                push!(v.sqrt());
            }
            Op::Abs => {
                let v = pop!();
                push!(v.abs());
            }
            Op::Floor => {
                let v = pop!();
                push!(v.floor());
            }
            Op::Ceil => {
                let v = pop!();
                push!(v.ceil());
            }
            Op::Round => {
                let v = pop!();
                push!(v.round());
            }
            Op::InvLogit => {
                let v = pop!();
                let r = if v >= 0.0 {
                    1.0 / (1.0 + (-v).exp())
                } else {
                    let e = v.exp();
                    e / (1.0 + e)
                };
                push!(r);
            }
            Op::Logit => {
                let v = pop!();
                let clamped = v.clamp(1e-15, 1.0 - 1e-15);
                push!((clamped / (1.0 - clamped)).ln());
            }
            Op::CmpLt => {
                let r = pop!();
                let l = pop!();
                push!(if l < r { 1.0 } else { 0.0 });
            }
            Op::CmpLe => {
                let r = pop!();
                let l = pop!();
                push!(if l <= r { 1.0 } else { 0.0 });
            }
            Op::CmpGt => {
                let r = pop!();
                let l = pop!();
                push!(if l > r { 1.0 } else { 0.0 });
            }
            Op::CmpGe => {
                let r = pop!();
                let l = pop!();
                push!(if l >= r { 1.0 } else { 0.0 });
            }
            Op::CmpEq => {
                let r = pop!();
                let l = pop!();
                push!(if l == r { 1.0 } else { 0.0 });
            }
            Op::CmpNe => {
                let r = pop!();
                let l = pop!();
                push!(if l != r { 1.0 } else { 0.0 });
            }
            Op::LogicAnd => {
                let b = pop!();
                let a = pop!();
                push!(if a != 0.0 && b != 0.0 { 1.0 } else { 0.0 });
            }
            Op::LogicOr => {
                let b = pop!();
                let a = pop!();
                push!(if a != 0.0 || b != 0.0 { 1.0 } else { 0.0 });
            }
            Op::LogicNot => {
                let v = pop!();
                push!(if v == 0.0 { 1.0 } else { 0.0 });
            }
            Op::JumpIfFalse(target) => {
                let v = pop!();
                if v == 0.0 {
                    pc = target as usize;
                    continue;
                }
            }
            Op::Jump(target) => {
                pc = target as usize;
                continue;
            }
        }
        pc += 1;
    }
    // Balanced-bytecode invariant: every expression's compile produces
    // exactly one residual value on the stack. Catches off-by-one in any
    // future compile_expr_into change that compute_max_stack's end-of-scan
    // assert might miss (the two checks are belt-and-braces against the
    // same class of UB-on-pop bug).
    debug_assert!(
        len == 1,
        "eval_bytecode: bytecode finished at stack depth {len}, expected 1",
    );
    if len > 0 {
        unsafe { *buf.add(len - 1) }
    } else {
        0.0
    }
}

/// Indexed-form evaluator: `vars` is a `Vec<f64>` indexed by parse-time
/// variable slot; `covariates` is a `Vec<f64>` aligned to
/// `CompiledModel.referenced_covariates`. Hot-path replacement for the
/// HashMap-keyed `eval_expression` — eliminates the per-call string hash
/// + probe in the `pk_param_fn` closure. Falls back to 0.0 for the
/// HashMap-keyed variants (Variable/Covariate) since callers running
/// the indexed path have already resolved every name.
fn eval_expression_indexed(
    expr: &Expression,
    theta: &[f64],
    eta: &[f64],
    covariates: &[f64],
    vars: &[f64],
    nn_outputs: &[Vec<f64>],
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
            let l = eval_expression_indexed(lhs, theta, eta, covariates, vars, nn_outputs);
            let r = eval_expression_indexed(rhs, theta, eta, covariates, vars, nn_outputs);
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
                BinOp::Mod => l.rem_euclid(r),
            }
        }
        Expression::UnaryFn(name, arg) => {
            let v = eval_expression_indexed(arg, theta, eta, covariates, vars, nn_outputs);
            match name.as_str() {
                "exp" => v.exp(),
                "log" | "ln" => v.max(1e-30).ln(),
                "sqrt" => v.max(0.0).sqrt(),
                "abs" => v.abs(),
                "floor" => v.floor(),
                "ceil" => v.ceil(),
                "round" => v.round(),
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
            let b = eval_expression_indexed(base, theta, eta, covariates, vars, nn_outputs);
            let e = eval_expression_indexed(exp, theta, eta, covariates, vars, nn_outputs);
            b.powf(e)
        }
        Expression::Conditional(cond, t, e) => {
            if eval_condition_indexed(cond, theta, eta, covariates, vars, nn_outputs) {
                eval_expression_indexed(t, theta, eta, covariates, vars, nn_outputs)
            } else {
                eval_expression_indexed(e, theta, eta, covariates, vars, nn_outputs)
            }
        }
        Expression::NnOutput { nn_idx, output_idx } => {
            // Reads from the per-call cache that `build_pk_param_fn`
            // populates once per forward via `NamedMlpMapper::forward_raw`.
            // Multiple references to outputs of the same NN therefore share
            // a single forward pass.
            nn_outputs
                .get(*nn_idx)
                .and_then(|v| v.get(*output_idx))
                .copied()
                .unwrap_or_else(|| {
                    debug_assert!(
                        false,
                        "NnOutput nn_idx={nn_idx} output_idx={output_idx} out of bounds"
                    );
                    0.0
                })
        }
    }
}

fn eval_condition_indexed(
    cond: &Condition,
    theta: &[f64],
    eta: &[f64],
    covariates: &[f64],
    vars: &[f64],
    nn_outputs: &[Vec<f64>],
) -> bool {
    match cond {
        Condition::Compare(l, op, r) => {
            let lv = eval_expression_indexed(l, theta, eta, covariates, vars, nn_outputs);
            let rv = eval_expression_indexed(r, theta, eta, covariates, vars, nn_outputs);
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
            eval_condition_indexed(l, theta, eta, covariates, vars, nn_outputs)
                && eval_condition_indexed(r, theta, eta, covariates, vars, nn_outputs)
        }
        Condition::Or(l, r) => {
            eval_condition_indexed(l, theta, eta, covariates, vars, nn_outputs)
                || eval_condition_indexed(r, theta, eta, covariates, vars, nn_outputs)
        }
        Condition::Not(c) => !eval_condition_indexed(c, theta, eta, covariates, vars, nn_outputs),
    }
}

/// Indexed-form statement executor; mirror of `eval_statements`. Handles
/// `AssignIdx`, `DiffEqIdx`, and `If`; if any non-indexed `Assign` or
/// `DiffEq` slips through, it falls back to a no-op (defensive — caller
/// should ensure the AST has been run through `resolve_variable_indices`).
///
/// `du` carries the derivative buffer the `DiffEqIdx` arm writes into. ODE
/// RHS callers (`build_ode_rhs_fn`) pass `Some(du)`; non-derivative callers
/// (`pk_param_fn`) pass `None` and never construct a `DiffEqIdx`.
fn eval_statements_indexed(
    stmts: &[Statement],
    theta: &[f64],
    eta: &[f64],
    covariates: &[f64],
    vars: &mut [f64],
    du: Option<&mut [f64]>,
    nn_outputs: &[Vec<f64>],
) {
    // Acquire the bytecode-stack scratch ONCE per call from the shared
    // FERX_SCRATCH; the borrow is held for the lifetime of this statement
    // loop and threaded into recursive `If` arms. Callers that already
    // hold FERX_SCRATCH (`build_ode_rhs_fn` does, for the rhs_vars setup)
    // should bypass this wrapper and call eval_statements_indexed_with_stack
    // directly with the bc_stack field — re-entering the borrow_mut here
    // would panic.
    FERX_SCRATCH.with(|cell| {
        let mut s = cell.borrow_mut();
        eval_statements_indexed_with_stack(
            stmts,
            theta,
            eta,
            covariates,
            vars,
            du,
            nn_outputs,
            &mut s.bc_stack,
        );
    });
}

/// Inner statement evaluator threaded with a caller-owned bytecode stack.
/// Split from `eval_statements_indexed` so recursive `If` evaluation reuses
/// the same `Vec<f64>` scratch instead of re-acquiring the TLS borrow per
/// nested call.
#[allow(clippy::too_many_arguments)]
fn eval_statements_indexed_with_stack(
    stmts: &[Statement],
    theta: &[f64],
    eta: &[f64],
    covariates: &[f64],
    vars: &mut [f64],
    du: Option<&mut [f64]>,
    nn_outputs: &[Vec<f64>],
    bc_stack: &mut Vec<f64>,
) {
    let mut du_opt = du;
    for s in stmts {
        match s {
            Statement::AssignBc(idx, bc) => {
                let v = eval_bytecode(bc, theta, eta, covariates, vars, nn_outputs, bc_stack);
                if let Some(slot) = vars.get_mut(*idx) {
                    *slot = v;
                }
            }
            Statement::DiffEqBc(state_idx, bc) => {
                let v = eval_bytecode(bc, theta, eta, covariates, vars, nn_outputs, bc_stack);
                if let Some(buf) = du_opt.as_deref_mut() {
                    if let Some(slot) = buf.get_mut(*state_idx) {
                        *slot = v;
                    }
                }
            }
            Statement::AssignIdx(idx, expr) => {
                // Pre-bytecode path — kept for any caller that bypasses
                // `resolve_variable_indices`. Slightly slower (recursive AST
                // walk), correct.
                let v = eval_expression_indexed(expr, theta, eta, covariates, vars, nn_outputs);
                if let Some(slot) = vars.get_mut(*idx) {
                    *slot = v;
                }
            }
            Statement::DiffEqIdx(state_idx, expr) => {
                let v = eval_expression_indexed(expr, theta, eta, covariates, vars, nn_outputs);
                if let Some(buf) = du_opt.as_deref_mut() {
                    if let Some(slot) = buf.get_mut(*state_idx) {
                        *slot = v;
                    }
                }
            }
            Statement::If {
                branches,
                else_body,
            } => {
                let mut taken = false;
                for (cond, body) in branches {
                    if eval_condition_indexed(cond, theta, eta, covariates, vars, nn_outputs) {
                        eval_statements_indexed_with_stack(
                            body,
                            theta,
                            eta,
                            covariates,
                            vars,
                            du_opt.as_deref_mut(),
                            nn_outputs,
                            bc_stack,
                        );
                        taken = true;
                        break;
                    }
                }
                if !taken {
                    if let Some(eb) = else_body {
                        eval_statements_indexed_with_stack(
                            eb,
                            theta,
                            eta,
                            covariates,
                            vars,
                            du_opt.as_deref_mut(),
                            nn_outputs,
                            bc_stack,
                        );
                    }
                }
            }
            // Non-indexed Assign/DiffEq shouldn't appear in a resolved AST
            // for the ODE RHS or pk_param_fn — `resolve_variable_indices`
            // rewrites them. Silently skip if one slips through.
            Statement::Assign(_, _) | Statement::DiffEq(_, _) => {}
        }
    }
}

/// Walk the AST in `stmts` and replace every `Statement::Assign(name, e)`
/// with `Statement::AssignIdx(idx, e)`, every `Expression::Variable(name)`
/// with `Expression::VariableIdx(idx)`, and every `Expression::Covariate(name)`
/// with `Expression::CovariateIdx(idx)`. Indices come from `var_idx` and
/// `cov_idx`. Variables not in `var_idx` get `usize::MAX` (eval returns 0.0).
///
/// When `state_idx` is supplied (i.e. resolving an ODE RHS), `Statement::DiffEq`
/// is also rewritten to `Statement::DiffEqIdx(state_slot, expr)` so the hot
/// path can write directly into `du[state_slot]`. `pk_param_fn` calls this with
/// `state_idx = None` — no `d/dt(...)` statements are valid there anyway.
fn resolve_variable_indices(
    stmts: &mut [Statement],
    var_idx: &HashMap<String, usize>,
    cov_idx: &HashMap<String, usize>,
    state_idx: Option<&HashMap<String, usize>>,
) {
    for s in stmts.iter_mut() {
        match s {
            Statement::Assign(name, expr) => {
                resolve_expr_indices(expr, var_idx, cov_idx);
                let i = var_idx.get(name).copied().unwrap_or(usize::MAX);
                let taken_expr = std::mem::replace(expr, Expression::Literal(0.0));
                let bc = compile_bytecode(&taken_expr);
                *s = Statement::AssignBc(i, bc);
            }
            Statement::AssignIdx(idx, expr) => {
                // Already-resolved (only produced by intermediate parser passes
                // that didn't go through the bytecode compiler); compile here
                // so the hot-path evaluator never sees a tree node.
                resolve_expr_indices(expr, var_idx, cov_idx);
                let idx = *idx;
                let taken_expr = std::mem::replace(expr, Expression::Literal(0.0));
                let bc = compile_bytecode(&taken_expr);
                *s = Statement::AssignBc(idx, bc);
            }
            Statement::DiffEq(name, expr) => {
                resolve_expr_indices(expr, var_idx, cov_idx);
                if let Some(sidx) = state_idx {
                    // The parser's `[odes]: missing d/dt(...)` validator at
                    // `build_ode_rhs_fn` already enforces exact-string match
                    // between this `name` and a declared state, so the lookup
                    // cannot miss; if it ever does, we'd silently drop the
                    // derivative — assert in debug builds to catch that.
                    let slot = sidx.get(name).copied().unwrap_or(usize::MAX);
                    debug_assert!(
                        slot != usize::MAX,
                        "resolve_variable_indices: DiffEq state `{name}` not in state_index",
                    );
                    let taken_expr = std::mem::replace(expr, Expression::Literal(0.0));
                    let bc = compile_bytecode(&taken_expr);
                    *s = Statement::DiffEqBc(slot, bc);
                }
            }
            Statement::DiffEqIdx(slot, expr) => {
                resolve_expr_indices(expr, var_idx, cov_idx);
                let slot = *slot;
                let taken_expr = std::mem::replace(expr, Expression::Literal(0.0));
                let bc = compile_bytecode(&taken_expr);
                *s = Statement::DiffEqBc(slot, bc);
            }
            Statement::AssignBc(_, _) | Statement::DiffEqBc(_, _) => {
                // Already compiled — re-resolving is a no-op.
            }
            Statement::If {
                branches,
                else_body,
            } => {
                for (cond, body) in branches.iter_mut() {
                    resolve_condition_indices(cond, var_idx, cov_idx);
                    resolve_variable_indices(body, var_idx, cov_idx, state_idx);
                }
                if let Some(eb) = else_body {
                    resolve_variable_indices(eb, var_idx, cov_idx, state_idx);
                }
            }
        }
    }
}

fn resolve_expr_indices(
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
        | Expression::CovariateIdx(_)
        | Expression::NnOutput { .. } => {}
    }
}

fn resolve_condition_indices(
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

// ─── Symbolic AST differentiation ──────────────────────────────────────────
//
// Pure function `differentiate(expr, axis)` returning the partial derivative
// of `expr` with respect to one of three axes: a θ slot, an η slot, or an
// already-resolved variable slot (used when chain-ruling through ODE-block
// intermediates whose own sensitivities have been precomputed and live in
// the variable slot pool). This is milestone 1 of the Tier 4a sensitivity-
// ODE work tracked in issue #134 — the AST-level primitive everything
// else (param-block partials, augmented-ODE codegen, Form C readout
// sensitivities) will compose on top.
//
// The differentiator handles every Expression variant the parser emits:
//   Literal                : ∂c/∂x      = 0
//   Theta(k)               : ∂θ_k/∂θ_j  = δ_{k,j}; 0 against η / Var axis
//   Eta(k)                 : ∂η_k/∂η_j  = δ_{k,j}; 0 against θ / Var axis
//   VariableIdx(k)         : ∂v_k/∂v_j  = δ_{k,j}; 0 against θ / η axis
//   Covariate*, NnOutput   : treated as constants (0)
//   BinOp(+,-)             : linearity
//   BinOp(*)               : product rule  L'·R + L·R'
//   BinOp(/)               : quotient rule (L'·R − L·R') / R²
//   UnaryFn("exp", a)      : exp(a) · a'
//   UnaryFn("log"|"ln", a) : a' / a
//   UnaryFn("sqrt", a)     : a' / (2·sqrt(a))
//   UnaryFn("abs", a)      : if a ≥ 0 then a' else −a' (boundary undefined)
//   UnaryFn("inv_logit"|"expit", a) : inv_logit(a) · (1 − inv_logit(a)) · a'
//   UnaryFn("logit", a)    : a' / (a · (1 − a))
//   UnaryFn(unknown, a)    : a' (mirrors the slow path's identity fallthrough)
//   Power(b, e)            : b^e · (e'·ln(b) + e · b'/b)  (general; subsumes
//                            both constant-base and constant-exponent cases)
//   Conditional(c, t, e)   : Conditional(c, t', e')   (boundary discontinuity
//                            ignored — standard AD convention)
//
// Returned expressions are NOT simplified — the bytecode compiler's
// constant pool happily absorbs `0 + x` / `1 * x` slack, and keeping
// `differentiate` purely mechanical makes the result reviewable as the
// raw chain-rule output. A separate `simplify_expr` helper is provided
// for callers (e.g. test output formatting) that want a tidier tree.
//
// Unresolved `Variable(name)` / `Covariate(name)` nodes panic — every
// caller in the sensitivity pipeline runs `resolve_expr_indices` first.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DiffAxis {
    /// `θ_idx` — a fixed-effect (population) parameter slot.
    Theta(usize),
    /// `η_idx` — a random-effect slot.
    Eta(usize),
    /// `v_idx` — an already-resolved variable slot. Used for chain-ruling
    /// into ODE-block intermediates whose own sensitivities are precomputed
    /// and live in the variable pool.
    // No runtime consumer after #145 (the augmented-ODE-RHS path that
    // would have produced these v_idx leaves was reverted). Kept so the
    // axis type stays complete for a future symbolic-gradient consumer.
    #[allow(dead_code)]
    Variable(usize),
}

#[allow(dead_code)] // unit-test convenience wrapper for the no-chain case
fn differentiate(expr: &Expression, axis: DiffAxis) -> Expression {
    // No chain context — used by milestone-1 tests and any caller that doesn't
    // need to chain-rule through intermediates. Equivalent to
    // `differentiate_with_chain(expr, axis, &HashMap::new())`.
    static EMPTY: std::sync::OnceLock<HashMap<usize, Expression>> = std::sync::OnceLock::new();
    differentiate_with_chain(expr, axis, EMPTY.get_or_init(HashMap::new))
}

/// Symbolic differentiator with optional chain-rule substitution at
/// `VariableIdx(k)` leaves. When `chain` contains a partial expression for
/// slot `k`, the differentiator returns that expression instead of the
/// default Kronecker delta. This is how the milestone-2
/// `build_indiv_param_partials` pass chain-rules through intermediate
/// `[individual_parameters]` assignments; the ODE-block-intermediate use
/// case was wired by milestones 3-4 and reverted in #145.
///
/// Concrete example from milestone 2 — given
///   ka_mult = TVKAM * exp(ETA_KAM)
///   CL      = TVCL  * exp(ETA_CL) * ka_mult
/// computing ∂CL/∂η_KAM with `chain = { ka_mult_slot: ∂ka_mult/∂η_KAM }`
/// substitutes the precomputed partial at the `VariableIdx(ka_mult_slot)`
/// leaf rather than returning Literal(0.0); the product rule then carries
/// it through to the final answer `TVCL · exp(ETA_CL) · ∂ka_mult/∂η_KAM`.
///
/// When `chain` is empty this is identical to the milestone-1 `differentiate`
/// and the same FD-verified tests apply.
fn differentiate_with_chain(
    expr: &Expression,
    axis: DiffAxis,
    chain: &HashMap<usize, Expression>,
) -> Expression {
    // Local helpers — avoid `use BinOp::*` because `Expression::BinOp` is a
    // variant with the same name and would create a glob-import ambiguity.
    let mul =
        |l: Expression, r: Expression| Expression::BinOp(Box::new(l), BinOp::Mul, Box::new(r));
    let add =
        |l: Expression, r: Expression| Expression::BinOp(Box::new(l), BinOp::Add, Box::new(r));
    let sub =
        |l: Expression, r: Expression| Expression::BinOp(Box::new(l), BinOp::Sub, Box::new(r));
    let div =
        |l: Expression, r: Expression| Expression::BinOp(Box::new(l), BinOp::Div, Box::new(r));
    let kron = |slot: usize, target: usize| -> Expression {
        if slot == target {
            Expression::Literal(1.0)
        } else {
            Expression::Literal(0.0)
        }
    };
    match expr {
        Expression::Literal(_) => Expression::Literal(0.0),
        Expression::Theta(k) => match axis {
            DiffAxis::Theta(j) => kron(*k, j),
            _ => Expression::Literal(0.0),
        },
        Expression::Eta(k) => match axis {
            DiffAxis::Eta(j) => kron(*k, j),
            _ => Expression::Literal(0.0),
        },
        Expression::VariableIdx(k) => {
            // Chain-rule entry point: if `k` is an upstream intermediate
            // whose partial w.r.t. `axis` was precomputed, return that
            // expression. Otherwise fall back to the Kronecker delta — the
            // variable is either the differentiation target itself
            // (DiffAxis::Variable) or a base variable (state, dose-amount,
            // etc.) that is independent of θ/η.
            if let Some(partial) = chain.get(k) {
                return partial.clone();
            }
            match axis {
                DiffAxis::Variable(j) => kron(*k, j),
                _ => Expression::Literal(0.0),
            }
        }
        Expression::CovariateIdx(_) | Expression::NnOutput { .. } => Expression::Literal(0.0),
        Expression::Variable(name) | Expression::Covariate(name) => panic!(
            "differentiate: unresolved AST node `{name}` reached the \
             differentiator; resolve_expr_indices must run first",
        ),
        Expression::BinOp(l, op, r) => match op {
            BinOp::Add | BinOp::Sub => Expression::BinOp(
                Box::new(differentiate_with_chain(l, axis, chain)),
                *op,
                Box::new(differentiate_with_chain(r, axis, chain)),
            ),
            BinOp::Mul => {
                // L'·R + L·R'
                let dl = differentiate_with_chain(l, axis, chain);
                let dr = differentiate_with_chain(r, axis, chain);
                add(mul(dl, (**r).clone()), mul((**l).clone(), dr))
            }
            BinOp::Div => {
                // (L'·R − L·R') / R²
                let dl = differentiate_with_chain(l, axis, chain);
                let dr = differentiate_with_chain(r, axis, chain);
                let num = sub(mul(dl, (**r).clone()), mul((**l).clone(), dr));
                let denom = mul((**r).clone(), (**r).clone());
                div(num, denom)
            }
            BinOp::Mod => Expression::Literal(0.0), // mod is discontinuous; derivative is 0 a.e.
        },
        Expression::UnaryFn(name, arg) => {
            let da = differentiate_with_chain(arg, axis, chain);
            match name.as_str() {
                "exp" => {
                    // exp(a) · a'
                    mul(Expression::UnaryFn("exp".into(), arg.clone()), da)
                }
                "log" | "ln" => {
                    // a' / a
                    div(da, (**arg).clone())
                }
                "sqrt" => {
                    // a' / (2·sqrt(a))
                    let two_sqrt = mul(
                        Expression::Literal(2.0),
                        Expression::UnaryFn("sqrt".into(), arg.clone()),
                    );
                    div(da, two_sqrt)
                }
                "abs" => {
                    // if a ≥ 0 then a' else −a'
                    let neg_da = sub(Expression::Literal(0.0), da.clone());
                    Expression::Conditional(
                        Box::new(Condition::Compare(
                            (**arg).clone(),
                            CmpOp::Ge,
                            Expression::Literal(0.0),
                        )),
                        Box::new(da),
                        Box::new(neg_da),
                    )
                }
                "inv_logit" | "expit" => {
                    // s · (1 − s) · a'   where s = inv_logit(a)
                    let s = Expression::UnaryFn("inv_logit".into(), arg.clone());
                    let one_minus_s = sub(Expression::Literal(1.0), s.clone());
                    mul(mul(s, one_minus_s), da)
                }
                "logit" => {
                    // a' / (a · (1 − a))
                    let one_minus_a = sub(Expression::Literal(1.0), (**arg).clone());
                    let denom = mul((**arg).clone(), one_minus_a);
                    div(da, denom)
                }
                _ => {
                    // Unknown name — the slow path returns the argument
                    // unchanged, so the derivative is its argument's
                    // derivative. (See `eval_expression_indexed`'s
                    // `_ => v` fallthrough.)
                    da
                }
            }
        }
        Expression::Power(b, e) => {
            // d/dx (b^e) = b^e · (e' · ln(b) + e · b' / b)
            // This subsumes the constant-base and constant-exponent cases:
            // a Literal exponent has e' = 0 so the e'·ln(b) term vanishes,
            // leaving b^e · e · b'/b = e · b^(e-1) · b' after `simplify_expr`
            // (or bytecode constant folding); a Literal base has b' = 0 so
            // the second term vanishes.
            let db = differentiate_with_chain(b, axis, chain);
            let de = differentiate_with_chain(e, axis, chain);
            let pow = Expression::Power(b.clone(), e.clone());
            let term_e = mul(de, Expression::UnaryFn("ln".into(), b.clone()));
            let term_b = mul((**e).clone(), div(db, (**b).clone()));
            let bracket = add(term_e, term_b);
            mul(pow, bracket)
        }
        Expression::Conditional(c, t, e) => {
            // Boundary discontinuity ignored — standard AD convention.
            // Away from the discontinuity, the derivative is whichever
            // branch's derivative the condition selects.
            Expression::Conditional(
                c.clone(),
                Box::new(differentiate_with_chain(t, axis, chain)),
                Box::new(differentiate_with_chain(e, axis, chain)),
            )
        }
    }
}

/// Optional cosmetic simplification — drops obviously-zero terms in
/// `differentiate`'s output so the tree printed for debugging or compiled
/// into bytecode is tidy. NOT required for correctness: bytecode constant
/// folding (and the eventual `eval_bytecode` runtime arithmetic) would
/// produce the same value either way.
///
/// Applied rules: `0 + x` → `x`, `x + 0` → `x`, `0 - x` → `-x` (kept as
/// `0 - x`), `x - 0` → `x`, `0 * x` → `0`, `x * 0` → `0`, `1 * x` → `x`,
/// `x * 1` → `x`, `0 / x` → `0`. The simplifier is intentionally shallow —
/// no constant folding across nested BinOps, no algebraic identities like
/// `x * x` → `x^2`. The point is just to keep `differentiate`'s mechanical
/// output readable for tests; a full algebraic simplifier is out of scope.
// Exercised by milestone-2 partials tests only; the milestone 3-5
// runtime consumers were reverted in #145.
#[allow(dead_code)]
fn simplify_expr(expr: &Expression) -> Expression {
    let is_lit = |e: &Expression, v: f64| matches!(e, Expression::Literal(x) if *x == v);
    match expr {
        Expression::BinOp(l, op, r) => {
            let l = simplify_expr(l);
            let r = simplify_expr(r);
            match op {
                BinOp::Add => {
                    if is_lit(&l, 0.0) {
                        r
                    } else if is_lit(&r, 0.0) {
                        l
                    } else {
                        Expression::BinOp(Box::new(l), *op, Box::new(r))
                    }
                }
                BinOp::Sub => {
                    if is_lit(&r, 0.0) {
                        l
                    } else {
                        Expression::BinOp(Box::new(l), *op, Box::new(r))
                    }
                }
                BinOp::Mul => {
                    if is_lit(&l, 0.0) || is_lit(&r, 0.0) {
                        Expression::Literal(0.0)
                    } else if is_lit(&l, 1.0) {
                        r
                    } else if is_lit(&r, 1.0) {
                        l
                    } else {
                        Expression::BinOp(Box::new(l), *op, Box::new(r))
                    }
                }
                BinOp::Div => {
                    if is_lit(&l, 0.0) {
                        Expression::Literal(0.0)
                    } else if is_lit(&r, 1.0) {
                        l
                    } else {
                        Expression::BinOp(Box::new(l), *op, Box::new(r))
                    }
                }
                BinOp::Mod => Expression::BinOp(Box::new(l), *op, Box::new(r)),
            }
        }
        Expression::UnaryFn(name, arg) => {
            Expression::UnaryFn(name.clone(), Box::new(simplify_expr(arg)))
        }
        Expression::Power(b, e) => {
            Expression::Power(Box::new(simplify_expr(b)), Box::new(simplify_expr(e)))
        }
        Expression::Conditional(c, t, e) => Expression::Conditional(
            c.clone(),
            Box::new(simplify_expr(t)),
            Box::new(simplify_expr(e)),
        ),
        Expression::Literal(_)
        | Expression::Theta(_)
        | Expression::Eta(_)
        | Expression::Variable(_)
        | Expression::VariableIdx(_)
        | Expression::Covariate(_)
        | Expression::CovariateIdx(_)
        | Expression::NnOutput { .. } => expr.clone(),
    }
}

// --- Milestone 2: `[individual_parameters]` partial derivatives ---
//
// For each top-level assignment `P_i = expr_i` in the `[individual_parameters]`
// block, precompute the symbolic partial-derivative Expression trees
//   ∂P_i/∂θ_k  for k in 0..n_theta_base
//   ∂P_i/∂η_k  for k in 0..n_eta_extended  (BSV η followed by κ)
// using `differentiate_with_chain`. Each row threads a chain-context map
// keyed by variable slot so a later assignment whose RHS references an
// earlier intermediate gets chain-ruled correctly through the earlier
// partial — no need to inline the intermediate's expression and re-explode
// the tree.
//
// Storage shape: outer Vec indexed by indiv-param position (parallel to
// `CompiledModel.indiv_param_names`), inner Vec indexed by axis. The
// expressions are stored in resolved form (VariableIdx, not Variable) so
// a future symbolic-gradient consumer can compile them to Bytecode without
// re-running `resolve_expr_indices`. (The originally-planned milestones
// 3-5 — augmented RHS, Form C readout sensitivities, estimator wiring —
// were reverted in #145; see the field doc on `CompiledModel.indiv_param_partials`.)
//
// Top-level `If { … }` statements in `[individual_parameters]` are not
// differentiated — no in-tree user model uses them and handling them needs
// branch-specific Conditional handling for the assignment-existence
// boundary. They are silently skipped (their slot has no row in
// `IndivParamPartials`); a future milestone can lift this restriction if a
// real model needs it. `NnOutput` references in indiv params have ∂/∂θ = 0
// (NN weights are treated as a separate axis class, deferred to a later
// milestone) and the current differentiator correctly returns Literal(0)
// for them.

/// Precomputed symbolic partials of `[individual_parameters]` assignments,
/// produced by [`build_indiv_param_partials`]. Stored on
/// [`CompiledModel`](crate::types::CompiledModel) as a primitive for any
/// future analytical-η-gradient path. The originally-planned consumers —
/// Tier 4a milestones 3-5 (augmented ODE RHS, Form C readout sensitivities,
/// `gradient = sens` estimator wiring) — were reverted in #145; the
/// partials themselves still pass their own FD-vs-symbolic unit tests
/// and are kept on `CompiledModel` so the primitive is in place when a
/// future consumer lands.
///
/// Inner field types reference the parser's private `Expression` AST, so the
/// fields stay `pub(crate)`. External callers can construct an empty
/// placeholder via [`IndivParamPartials::empty`] — this is the only thing
/// they need for hand-built `CompiledModel` test fixtures and the
/// `generate_data` data-generation binary.
#[derive(Debug, Clone)]
#[allow(dead_code)] // no runtime consumer after #145; see struct doc.
pub struct IndivParamPartials {
    /// Indiv-param names parallel to `d_d_theta` / `d_d_eta` outer Vec, in
    /// `[individual_parameters]` source-declaration order — one row per
    /// top-level `Assign(name, _)`.
    ///
    /// CAUTION (issue #357): this is a *subset* of
    /// `CompiledModel.indiv_param_names`, not a positional twin. A parameter
    /// assigned only inside `if`/`else` branches and promoted because a
    /// downstream block references it (e.g. a conditional `CL`) appears in
    /// `indiv_param_names` but has NO row here — `build_indiv_param_partials`
    /// skips `Statement::If`, and a piecewise param has no single symbolic
    /// partial anyway (the AD inner loop falls back to FD; see the `eta_map`
    /// note in `parse_full_model`). A future AD consumer MUST look partials up
    /// by name, never zip them positionally against `indiv_param_names`.
    pub(crate) names: Vec<String>,
    /// `d_d_theta[i][k]` = ∂P_i/∂θ_k. Inner Vec length = `n_theta_base`
    /// (user-declared θ count, NOT including NN-weight or diffusion θ which
    /// indiv params don't reference). Each Expression is in resolved form.
    pub(crate) d_d_theta: Vec<Vec<Expression>>,
    /// `d_d_eta[i][k]` = ∂P_i/∂η_k. Inner Vec length = `n_eta_extended` —
    /// BSV η first (slots 0..n_eta_bsv), kappas appended (slots
    /// n_eta_bsv..n_eta_bsv+n_kappa). Matches the `eta` slice the
    /// `pk_param_fn` closure consumes.
    pub(crate) d_d_eta: Vec<Vec<Expression>>,
}

impl IndivParamPartials {
    /// Empty partials — used when there are no indiv params or as a
    /// placeholder when building a `CompiledModel` by hand (e.g. test
    /// fixtures, the `generate_data` binary). Fully populated instances are
    /// produced internally by the parser.
    pub fn empty() -> Self {
        Self {
            names: Vec::new(),
            d_d_theta: Vec::new(),
            d_d_eta: Vec::new(),
        }
    }
}

/// Compute partial-derivative expressions for every top-level `Assign` in
/// `stmts` (the `[individual_parameters]` block, BEFORE
/// `resolve_variable_indices` has bytecode-compiled it). Resolves each
/// statement's expression locally (without mutating `stmts`), differentiates
/// w.r.t. every θ/η axis, and threads a per-axis chain context so chained
/// intermediate references get chain-ruled correctly.
///
/// Top-level `If` and any non-`Assign` variant is skipped. Returns
/// `IndivParamPartials::empty()` if there are no top-level Assigns.
fn build_indiv_param_partials(
    stmts: &[Statement],
    var_idx: &HashMap<String, usize>,
    cov_idx: &HashMap<String, usize>,
    n_theta_base: usize,
    n_eta_extended: usize,
) -> IndivParamPartials {
    // Per-axis chain context: variable slot → precomputed partial Expression
    // for THIS axis. Indexed by axis index, populated incrementally as we
    // walk the assignments in source order.
    let mut chain_theta: Vec<HashMap<usize, Expression>> =
        (0..n_theta_base).map(|_| HashMap::new()).collect();
    let mut chain_eta: Vec<HashMap<usize, Expression>> =
        (0..n_eta_extended).map(|_| HashMap::new()).collect();

    let mut names: Vec<String> = Vec::new();
    let mut d_d_theta: Vec<Vec<Expression>> = Vec::new();
    let mut d_d_eta: Vec<Vec<Expression>> = Vec::new();

    for stmt in stmts {
        let Statement::Assign(name, raw_expr) = stmt else {
            // If-blocks and any already-resolved variant (defensive — at the
            // call site `stmts` is the parser's pristine pre-resolve output,
            // so only Assign/If can appear): skip. Indiv-params with
            // top-level conditionals get no partials in this milestone.
            continue;
        };
        // Differentiate on a resolved CLONE of the expression — the
        // caller (`build_pk_param_fn`) will run `resolve_variable_indices`
        // on `stmts` independently, which destroys the Expression by
        // compiling it to Bytecode. We can't share that work.
        let mut resolved = raw_expr.clone();
        resolve_expr_indices(&mut resolved, var_idx, cov_idx);

        // Slot for this indiv param. `usize::MAX` (the resolve fallback)
        // would mean the name isn't in `var_idx` — at this call site
        // that's a parser bug because `[individual_parameters]` names
        // populate `var_idx` unconditionally; debug-assert so a future
        // pipeline change doesn't silently drop chain entries.
        let slot = var_idx.get(name).copied().unwrap_or(usize::MAX);
        debug_assert!(
            slot != usize::MAX,
            "build_indiv_param_partials: indiv-param `{name}` missing from var_idx — \
             parser pipeline bug (var_idx must be built from the same name list before \
             differentiation)",
        );

        let mut row_theta: Vec<Expression> = Vec::with_capacity(n_theta_base);
        for k in 0..n_theta_base {
            let d = differentiate_with_chain(&resolved, DiffAxis::Theta(k), &chain_theta[k]);
            let s = simplify_expr(&d);
            // Store the simplified partial into the chain context BEFORE
            // pushing — any later assignment that references slot `k` will
            // chain-rule through this expression. Always insert (even
            // Literal(0)) so the chain map's "key present ⇔ slot is an
            // upstream indiv-param" invariant is robust to zero partials.
            if slot != usize::MAX {
                chain_theta[k].insert(slot, s.clone());
            }
            row_theta.push(s);
        }

        let mut row_eta: Vec<Expression> = Vec::with_capacity(n_eta_extended);
        for k in 0..n_eta_extended {
            let d = differentiate_with_chain(&resolved, DiffAxis::Eta(k), &chain_eta[k]);
            let s = simplify_expr(&d);
            if slot != usize::MAX {
                chain_eta[k].insert(slot, s.clone());
            }
            row_eta.push(s);
        }

        names.push(name.clone());
        d_d_theta.push(row_theta);
        d_d_eta.push(row_eta);
    }

    IndivParamPartials {
        names,
        d_d_theta,
        d_d_eta,
    }
}

/// Execute a list of statements, mutating `vars` with each assignment. ODE
/// `DiffEq` statements write into the optional `du` slice using the
/// `state_index` lookup. For `[individual_parameters]` callers, pass `None`
/// for `du` and `state_index` — encountering a `DiffEq` is then an error.
fn eval_statements(
    stmts: &[Statement],
    theta: &[f64],
    eta: &[f64],
    covariates: &HashMap<String, f64>,
    vars: &mut HashMap<String, f64>,
    du: Option<&mut [f64]>,
    state_index: Option<&HashMap<String, usize>>,
    nn_outputs: &[Vec<f64>],
) {
    // The `du` borrow has to be re-passed into each recursive sub-block, so
    // shuttle it through an Option that we re-grab on each iteration.
    let mut du_opt = du;
    for s in stmts {
        match s {
            Statement::Assign(name, expr) => {
                let v = eval_expression(expr, theta, eta, covariates, vars, nn_outputs);
                vars.insert(name.clone(), v);
            }
            Statement::AssignIdx(_, _)
            | Statement::DiffEqIdx(_, _)
            | Statement::AssignBc(_, _)
            | Statement::DiffEqBc(_, _) => {
                // Indexed / bytecode statements are only produced by
                // `resolve_variable_indices` for the `pk_param_fn` and ODE
                // RHS closures, both of which use `eval_statements_indexed`
                // exclusively. They should never reach this evaluator;
                // silently skip if they do (defensive).
            }
            Statement::DiffEq(name, expr) => {
                let v = eval_expression(expr, theta, eta, covariates, vars, nn_outputs);
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
                    if eval_condition(cond, theta, eta, covariates, vars, nn_outputs) {
                        eval_statements(
                            body,
                            theta,
                            eta,
                            covariates,
                            vars,
                            du_opt.as_deref_mut(),
                            state_index,
                            nn_outputs,
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
                            nn_outputs,
                        );
                    }
                }
            }
        }
    }
}

// --- Tokenizer ---

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    Ident(String),
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    /// `=` — used by the statement parser as the assignment operator. The
    /// expression parser never accepts it, so `==` (`Token::EqEq`) is the
    /// correct choice for equality comparisons inside conditions.
    Eq,
    EqEq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    AndAnd,
    OrOr,
    Bang,
    Comma,
    /// `.` — member-access operator, used today only by `[covariate_nn]`
    /// dot-access syntax (`TYPICAL_PK.CL`). Decimal-point dots inside number
    /// literals (e.g. `4.5`, `.5`) are absorbed by the number tokenizer
    /// before this token is produced.
    Dot,
    /// Logical line separator. Consumed only by the statement parser, where
    /// it acts as an "end of statement" marker. Newlines inside `(...)` and
    /// `[...]` are stripped by `strip_newlines_in_groups` so users can break
    /// long expressions across lines.
    Newline,
}

fn tokenize(s: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        match chars[i] {
            ' ' | '\t' | '\r' => i += 1,
            '\n' => {
                // Collapse adjacent newlines so the statement parser doesn't
                // see runs of empty lines as separate terminators.
                if !matches!(tokens.last(), Some(Token::Newline)) {
                    tokens.push(Token::Newline);
                }
                i += 1;
            }
            '#' => {
                // Line comment — skip to next newline. Block-level comments
                // were already stripped by extract_blocks, but inline `# ...`
                // tails inside a multi-line if-block (which we re-join with
                // newlines) need to be tolerated here too.
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
            }
            '{' => {
                tokens.push(Token::LBrace);
                i += 1;
            }
            '}' => {
                tokens.push(Token::RBrace);
                i += 1;
            }
            ',' => {
                tokens.push(Token::Comma);
                i += 1;
            }
            '<' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    tokens.push(Token::Le);
                    i += 2;
                } else {
                    tokens.push(Token::Lt);
                    i += 1;
                }
            }
            '>' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    tokens.push(Token::Ge);
                    i += 2;
                } else {
                    tokens.push(Token::Gt);
                    i += 1;
                }
            }
            '=' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    tokens.push(Token::EqEq);
                    i += 2;
                } else {
                    tokens.push(Token::Eq);
                    i += 1;
                }
            }
            '!' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    tokens.push(Token::Ne);
                    i += 2;
                } else {
                    tokens.push(Token::Bang);
                    i += 1;
                }
            }
            '&' => {
                if i + 1 < chars.len() && chars[i + 1] == '&' {
                    tokens.push(Token::AndAnd);
                    i += 2;
                } else {
                    return Err("Unexpected '&' (use '&&' for logical and)".to_string());
                }
            }
            '|' => {
                if i + 1 < chars.len() && chars[i + 1] == '|' {
                    tokens.push(Token::OrOr);
                    i += 2;
                } else {
                    return Err("Unexpected '|' (use '||' for logical or)".to_string());
                }
            }
            '(' => {
                tokens.push(Token::LParen);
                i += 1;
            }
            ')' => {
                tokens.push(Token::RParen);
                i += 1;
            }
            '[' => {
                tokens.push(Token::LBracket);
                i += 1;
            }
            ']' => {
                tokens.push(Token::RBracket);
                i += 1;
            }
            '+' => {
                tokens.push(Token::Plus);
                i += 1;
            }
            '-' => {
                // Check if this is a negative number (after operator or at start)
                let is_unary = tokens.is_empty()
                    || matches!(
                        tokens.last(),
                        Some(
                            Token::LParen
                                | Token::LBracket
                                | Token::LBrace
                                | Token::Plus
                                | Token::Minus
                                | Token::Star
                                | Token::Slash
                                | Token::Caret
                                | Token::Eq
                                | Token::EqEq
                                | Token::Ne
                                | Token::Lt
                                | Token::Le
                                | Token::Gt
                                | Token::Ge
                                | Token::AndAnd
                                | Token::OrOr
                                | Token::Bang
                                | Token::Comma
                                | Token::Newline
                        )
                    );
                if is_unary
                    && i + 1 < chars.len()
                    && (chars[i + 1].is_ascii_digit() || chars[i + 1] == '.')
                {
                    let start = i;
                    i += 1;
                    while i < chars.len()
                        && (chars[i].is_ascii_digit()
                            || chars[i] == '.'
                            || chars[i] == 'e'
                            || chars[i] == 'E')
                    {
                        i += 1;
                    }
                    // Handle optional sign in exponent: e.g. `-1e-10`, `-2.5E+3`.
                    if i > 0
                        && (chars[i - 1] == 'e' || chars[i - 1] == 'E')
                        && i < chars.len()
                        && (chars[i] == '+' || chars[i] == '-')
                    {
                        i += 1;
                        while i < chars.len() && chars[i].is_ascii_digit() {
                            i += 1;
                        }
                    }
                    let num_str: String = chars[start..i].iter().collect();
                    let num: f64 = num_str
                        .parse()
                        .map_err(|_| format!("Bad number: {}", num_str))?;
                    tokens.push(Token::Number(num));
                } else {
                    tokens.push(Token::Minus);
                    i += 1;
                }
            }
            '*' => {
                tokens.push(Token::Star);
                i += 1;
            }
            '/' => {
                tokens.push(Token::Slash);
                i += 1;
            }
            '^' => {
                tokens.push(Token::Caret);
                i += 1;
            }
            // A `.` followed by a digit starts a decimal-only literal like `.5`.
            // Standalone `.` (e.g. `TYPICAL_PK.CL`) is the member-access operator
            // and is handled by a separate arm below.
            '.' if i + 1 < chars.len() && chars[i + 1].is_ascii_digit() => {
                let start = i;
                while i < chars.len()
                    && (chars[i].is_ascii_digit()
                        || chars[i] == '.'
                        || chars[i] == 'e'
                        || chars[i] == 'E')
                {
                    i += 1;
                }
                // Handle optional sign in exponent: e.g. `1.5e-3`, `.5E+2`.
                if i > 0
                    && (chars[i - 1] == 'e' || chars[i - 1] == 'E')
                    && i < chars.len()
                    && (chars[i] == '+' || chars[i] == '-')
                {
                    i += 1;
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                let num_str: String = chars[start..i].iter().collect();
                let num: f64 = num_str
                    .parse()
                    .map_err(|_| format!("Bad number: {}", num_str))?;
                tokens.push(Token::Number(num));
            }
            '.' => {
                tokens.push(Token::Dot);
                i += 1;
            }
            c if c.is_ascii_digit() => {
                let start = i;
                while i < chars.len()
                    && (chars[i].is_ascii_digit()
                        || chars[i] == '.'
                        || chars[i] == 'e'
                        || chars[i] == 'E')
                {
                    i += 1;
                }
                // Handle optional sign in exponent: e.g. `1e-10`, `2.5E+3`.
                if i > 0
                    && (chars[i - 1] == 'e' || chars[i - 1] == 'E')
                    && i < chars.len()
                    && (chars[i] == '+' || chars[i] == '-')
                {
                    i += 1;
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                let num_str: String = chars[start..i].iter().collect();
                let num: f64 = num_str
                    .parse()
                    .map_err(|_| format!("Bad number: {}", num_str))?;
                tokens.push(Token::Number(num));
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let ident: String = chars[start..i].iter().collect();
                tokens.push(Token::Ident(ident));
            }
            other => return Err(format!("Unexpected character: {}", other)),
        }
    }

    Ok(tokens)
}

// --- Recursive descent parser ---

fn parse_add_sub(
    tokens: &[Token],
    pos: usize,
    ctx: ParseCtx<'_>,
) -> Result<(Expression, usize), String> {
    let (mut left, mut pos) = parse_mul_div(tokens, pos, ctx)?;

    while pos < tokens.len() {
        match &tokens[pos] {
            Token::Plus => {
                let (right, p) = parse_mul_div(tokens, pos + 1, ctx)?;
                left = Expression::BinOp(Box::new(left), BinOp::Add, Box::new(right));
                pos = p;
            }
            Token::Minus => {
                let (right, p) = parse_mul_div(tokens, pos + 1, ctx)?;
                left = Expression::BinOp(Box::new(left), BinOp::Sub, Box::new(right));
                pos = p;
            }
            _ => break,
        }
    }

    Ok((left, pos))
}

fn parse_mul_div(
    tokens: &[Token],
    pos: usize,
    ctx: ParseCtx<'_>,
) -> Result<(Expression, usize), String> {
    let (mut left, mut pos) = parse_power(tokens, pos, ctx)?;

    while pos < tokens.len() {
        match &tokens[pos] {
            Token::Star => {
                let (right, p) = parse_power(tokens, pos + 1, ctx)?;
                left = Expression::BinOp(Box::new(left), BinOp::Mul, Box::new(right));
                pos = p;
            }
            Token::Slash => {
                let (right, p) = parse_power(tokens, pos + 1, ctx)?;
                left = Expression::BinOp(Box::new(left), BinOp::Div, Box::new(right));
                pos = p;
            }
            Token::Ident(kw) if kw.eq_ignore_ascii_case("mod") => {
                let (right, p) = parse_power(tokens, pos + 1, ctx)?;
                left = Expression::BinOp(Box::new(left), BinOp::Mod, Box::new(right));
                pos = p;
            }
            _ => break,
        }
    }

    Ok((left, pos))
}

fn parse_power(
    tokens: &[Token],
    pos: usize,
    ctx: ParseCtx<'_>,
) -> Result<(Expression, usize), String> {
    let (base, mut pos) = parse_atom(tokens, pos, ctx)?;

    if pos < tokens.len() && tokens[pos] == Token::Caret {
        let (exp, p) = parse_atom(tokens, pos + 1, ctx)?;
        pos = p;
        return Ok((Expression::Power(Box::new(base), Box::new(exp)), pos));
    }

    Ok((base, pos))
}

fn parse_atom(
    tokens: &[Token],
    pos: usize,
    ctx: ParseCtx<'_>,
) -> Result<(Expression, usize), String> {
    if pos >= tokens.len() {
        return Err("Unexpected end of expression".to_string());
    }

    match &tokens[pos] {
        Token::Minus => {
            // Unary minus: -expr → 0 - expr
            let (expr, p) = parse_atom(tokens, pos + 1, ctx)?;
            Ok((
                Expression::BinOp(
                    Box::new(Expression::Literal(0.0)),
                    BinOp::Sub,
                    Box::new(expr),
                ),
                p,
            ))
        }
        Token::Number(n) => Ok((Expression::Literal(*n), pos + 1)),
        Token::LParen => {
            let (expr, p) = parse_add_sub(tokens, pos + 1, ctx)?;
            if p >= tokens.len() || tokens[p] != Token::RParen {
                return Err("Missing closing parenthesis".to_string());
            }
            Ok((expr, p + 1))
        }
        Token::Ident(name) => {
            // compartments[N] — subscript access into DerivedContext::compartments.
            // Emits Variable("__cmt_N") which build_derived_vars populates at eval time.
            // Only literal non-negative integer indices are supported.
            if name.eq_ignore_ascii_case("compartments")
                && pos + 1 < tokens.len()
                && tokens[pos + 1] == Token::LBracket
            {
                let n_tok = tokens
                    .get(pos + 2)
                    .ok_or_else(|| "[derived] `compartments[`: missing index".to_string())?;
                let n = match n_tok {
                    Token::Number(v) => {
                        if v.fract() != 0.0 || *v < 0.0 {
                            return Err(format!(
                                "[derived] `compartments[{v}]`: index must be a non-negative integer"
                            ));
                        }
                        *v as usize
                    }
                    _ => {
                        return Err(
                            "[derived] `compartments[...]`: only literal integer indices are \
                             supported in Phase 1 — use `compartments[0]`, `compartments[1]`, etc."
                                .to_string(),
                        )
                    }
                };
                // MAX_CMT_INDEX is defined at module scope (same value that
                // build_derived_vars uses to bound its sentinel loop).  Any index
                // beyond it is not in the sentinel map and would silently return 0.0
                // via eval_expression's .unwrap_or(0.0).  Reject at parse time.
                if n > MAX_CMT_INDEX {
                    return Err(format!(
                        "[derived] `compartments[{n}]`: index {n} exceeds the maximum \
                         supported value ({MAX_CMT_INDEX}). \
                         Use compartments[0] through compartments[{MAX_CMT_INDEX}]."
                    ));
                }
                if tokens.get(pos + 3) != Some(&Token::RBracket) {
                    return Err("[derived] `compartments[N`: missing closing `]`".to_string());
                }
                return Ok((Expression::Variable(format!("__cmt_{n}")), pos + 4));
            }

            // Inline conditional expression: `if (cond) then_expr else else_expr`
            // Used as a value (e.g. `CL = if (SEX == 1) TVCL * 1.2 else TVCL`).
            if name.eq_ignore_ascii_case("if") {
                let p = pos + 1;
                if p >= tokens.len() || tokens[p] != Token::LParen {
                    return Err("`if` must be followed by `(`".to_string());
                }
                let (cond, p) = parse_condition(tokens, p + 1, ctx)?;
                if p >= tokens.len() || tokens[p] != Token::RParen {
                    return Err("Missing closing `)` after if-condition".to_string());
                }
                let (then_expr, p) = parse_add_sub(tokens, p + 1, ctx)?;
                if p >= tokens.len() {
                    return Err("Inline `if` expression missing `else` branch".to_string());
                }
                match &tokens[p] {
                    Token::Ident(kw) if kw.eq_ignore_ascii_case("else") => {}
                    _ => {
                        return Err(
                            "Inline `if` expression must end with `else <expr>`".to_string()
                        );
                    }
                }
                let (else_expr, p) = parse_add_sub(tokens, p + 1, ctx)?;
                return Ok((
                    Expression::Conditional(
                        Box::new(cond),
                        Box::new(then_expr),
                        Box::new(else_expr),
                    ),
                    p,
                ));
            }

            // Check if it's a function call: name(expr)
            if pos + 1 < tokens.len() && tokens[pos + 1] == Token::LParen {
                let func_name = name.to_lowercase();
                let (arg, p) = parse_add_sub(tokens, pos + 2, ctx)?;
                if p >= tokens.len() || tokens[p] != Token::RParen {
                    return Err(format!("Missing closing parenthesis for function {}", name));
                }
                return Ok((Expression::UnaryFn(func_name, Box::new(arg)), p + 1));
            }

            // Check if it's an `[covariate_nn NAME]` output access: `NAME.OUTPUT`.
            // Resolved at parse time into `Expression::NnOutput { nn_idx, output_idx }`
            // so the evaluator can read the pre-computed forward output by index.
            if pos + 1 < tokens.len() && tokens[pos + 1] == Token::Dot {
                if let Some(nn_idx) = ctx.nn_specs.iter().position(|(n, _)| n == name) {
                    let output_tok = tokens.get(pos + 2).ok_or_else(|| {
                        format!("`{}.` is missing an output name after the dot", name)
                    })?;
                    let output_name = match output_tok {
                        Token::Ident(s) => s,
                        other => {
                            return Err(format!(
                                "`{}.` must be followed by an output name (identifier), got {:?}",
                                name, other
                            ));
                        }
                    };
                    let outputs = &ctx.nn_specs[nn_idx].1;
                    let output_idx =
                        outputs
                            .iter()
                            .position(|o| o == output_name)
                            .ok_or_else(|| {
                                format!(
                                    "`{name}.{output_name}` is not declared as an output of \
                                 [covariate_nn {name}]. Known outputs: {}",
                                    outputs.join(", ")
                                )
                            })?;
                    return Ok((Expression::NnOutput { nn_idx, output_idx }, pos + 3));
                }
                // `NAME.X` where NAME is not a known NN — fall through to the
                // identifier-classification chain. The `Dot` token will then
                // be the next unexpected token; the caller will surface that
                // as an expression-parse error.
            }

            // Check if it's a theta
            if let Some(idx) = ctx.theta_names.iter().position(|n| n == name) {
                return Ok((Expression::Theta(idx), pos + 1));
            }

            // Check if it's an eta
            if let Some(idx) = ctx.eta_names.iter().position(|n| n == name) {
                return Ok((Expression::Eta(idx), pos + 1));
            }

            // Previously-assigned local variable (e.g. earlier lines of
            // [individual_parameters], or a state/param name injected by the
            // ODE RHS harness).
            if ctx.defined_vars.iter().any(|n| n == name) {
                return Ok((Expression::Variable(name.clone()), pos + 1));
            }

            // Anything else is a covariate reference in the regular model
            // context. The ODE RHS context keeps it as a Variable so that the
            // eval-time `vars` map (which carries state + individual params)
            // can resolve it case-sensitively.
            if ctx.fallback_covariate {
                Ok((Expression::Covariate(name.clone()), pos + 1))
            } else {
                Ok((Expression::Variable(name.clone()), pos + 1))
            }
        }
        other => Err(format!("Unexpected token: {:?}", other)),
    }
}

// ── Condition parser ────────────────────────────────────────────────────────
//
// Precedence (lowest first):  ||  >  &&  >  !  >  comparison.
// The condition grammar lives in its own recursive-descent stack so that the
// arithmetic expression parser doesn't need to understand boolean operators.

fn parse_condition(
    tokens: &[Token],
    pos: usize,
    ctx: ParseCtx<'_>,
) -> Result<(Condition, usize), String> {
    parse_cond_or(tokens, pos, ctx)
}

fn parse_cond_or(
    tokens: &[Token],
    pos: usize,
    ctx: ParseCtx<'_>,
) -> Result<(Condition, usize), String> {
    let (mut left, mut pos) = parse_cond_and(tokens, pos, ctx)?;
    while pos < tokens.len() && tokens[pos] == Token::OrOr {
        let (right, p) = parse_cond_and(tokens, pos + 1, ctx)?;
        left = Condition::Or(Box::new(left), Box::new(right));
        pos = p;
    }
    Ok((left, pos))
}

fn parse_cond_and(
    tokens: &[Token],
    pos: usize,
    ctx: ParseCtx<'_>,
) -> Result<(Condition, usize), String> {
    let (mut left, mut pos) = parse_cond_not(tokens, pos, ctx)?;
    while pos < tokens.len() && tokens[pos] == Token::AndAnd {
        let (right, p) = parse_cond_not(tokens, pos + 1, ctx)?;
        left = Condition::And(Box::new(left), Box::new(right));
        pos = p;
    }
    Ok((left, pos))
}

fn parse_cond_not(
    tokens: &[Token],
    pos: usize,
    ctx: ParseCtx<'_>,
) -> Result<(Condition, usize), String> {
    if pos < tokens.len() && tokens[pos] == Token::Bang {
        let (inner, p) = parse_cond_not(tokens, pos + 1, ctx)?;
        return Ok((Condition::Not(Box::new(inner)), p));
    }
    parse_cond_atom(tokens, pos, ctx)
}

fn parse_cond_atom(
    tokens: &[Token],
    pos: usize,
    ctx: ParseCtx<'_>,
) -> Result<(Condition, usize), String> {
    // Parenthesised sub-condition: `(cond)`.  Always try to parse the
    // contents as a condition first.  This handles `(a < b)`, `(!(x == 1))`,
    // `((a > 0) && (b < 10))`, etc. without any lookahead heuristic.
    if pos < tokens.len() && tokens[pos] == Token::LParen {
        let (inner, p) = parse_condition(tokens, pos + 1, ctx)?;
        if p >= tokens.len() || tokens[p] != Token::RParen {
            return Err("Missing closing `)` in condition".to_string());
        }
        return Ok((inner, p + 1));
    }

    // comparison: expr <cmpop> expr
    let (lhs, p) = parse_add_sub(tokens, pos, ctx)?;
    if p >= tokens.len() {
        return Err("Expected comparison operator in condition".to_string());
    }
    let op = match tokens[p] {
        Token::Lt => CmpOp::Lt,
        Token::Le => CmpOp::Le,
        Token::Gt => CmpOp::Gt,
        Token::Ge => CmpOp::Ge,
        Token::EqEq => CmpOp::Eq,
        Token::Ne => CmpOp::Ne,
        ref other => {
            return Err(format!(
                "Expected comparison operator (<, <=, >, >=, ==, !=) in condition, got {:?}",
                other
            ));
        }
    };
    let (rhs, p) = parse_add_sub(tokens, p + 1, ctx)?;
    Ok((Condition::Compare(lhs, op, rhs), p))
}

// ── Statement parser ────────────────────────────────────────────────────────
//
// A "statement" is one of:
//   NAME = expr                       (Assign)
//   d/dt(NAME) = expr                 (DiffEq — only legal in [odes])
//   if (cond) { stmts } [else if (cond) { stmts }]* [else { stmts }]?
//
// Statements are separated by `Token::Newline`. Blank lines and inline `# ...`
// comments are tolerated by the tokenizer. Newlines inside `(...)` and `[...]`
// are stripped by `strip_newlines_in_groups` so users can split long
// expressions across lines (newlines inside `{...}` are preserved because they
// separate statements within a body).

#[derive(Debug, Clone, Copy, PartialEq)]
enum StatementMode {
    /// `[individual_parameters]` and similar — DiffEqs are forbidden.
    Plain,
    /// `[odes]` — `d/dt(NAME) = expr` is a DiffEq statement.
    Ode,
}

/// Strip `Token::Newline` tokens that occur inside `(...)` or `[...]` groups,
/// so they don't terminate an expression. Newlines inside `{...}` are kept
/// because they separate statements within a block body.
fn strip_newlines_in_groups(tokens: Vec<Token>) -> Vec<Token> {
    let mut out = Vec::with_capacity(tokens.len());
    let mut depth = 0i32;
    for t in tokens {
        match t {
            Token::LParen | Token::LBracket => {
                depth += 1;
                out.push(t);
            }
            Token::RParen | Token::RBracket => {
                depth -= 1;
                out.push(t);
            }
            Token::Newline if depth > 0 => {
                // drop
            }
            other => out.push(other),
        }
    }
    out
}

fn skip_newlines(tokens: &[Token], mut pos: usize) -> usize {
    while pos < tokens.len() && tokens[pos] == Token::Newline {
        pos += 1;
    }
    pos
}

/// Parse a complete block of statements from raw text. Used by both
/// `[individual_parameters]` (Plain mode) and `[odes]` (Ode mode).
fn parse_block_statements(
    s: &str,
    ctx: ParseCtx<'_>,
    mode: StatementMode,
) -> Result<Vec<Statement>, String> {
    let toks = tokenize(s)?;
    let toks = strip_newlines_in_groups(toks);
    let (stmts, p) = parse_statements_until(&toks, 0, ctx, mode, /*stop_on_rbrace=*/ false)?;
    let p = skip_newlines(&toks, p);
    if p != toks.len() {
        return Err(format!(
            "Unexpected token after statements: {:?}",
            tokens_pretty_tail(&toks, p)
        ));
    }
    Ok(stmts)
}

fn tokens_pretty_tail(tokens: &[Token], pos: usize) -> Vec<&Token> {
    tokens.iter().skip(pos).take(5).collect()
}

fn parse_statements_until(
    tokens: &[Token],
    pos: usize,
    ctx: ParseCtx<'_>,
    mode: StatementMode,
    stop_on_rbrace: bool,
) -> Result<(Vec<Statement>, usize), String> {
    let mut out: Vec<Statement> = Vec::new();
    let mut pos = skip_newlines(tokens, pos);
    while pos < tokens.len() {
        if stop_on_rbrace && tokens[pos] == Token::RBrace {
            return Ok((out, pos));
        }
        let (stmt, p) = parse_one_statement(tokens, pos, ctx, mode)?;
        out.push(stmt);
        pos = skip_newlines(tokens, p);
    }
    Ok((out, pos))
}

fn parse_one_statement(
    tokens: &[Token],
    pos: usize,
    ctx: ParseCtx<'_>,
    mode: StatementMode,
) -> Result<(Statement, usize), String> {
    if pos >= tokens.len() {
        return Err("Unexpected end of block while parsing statement".to_string());
    }
    // if-statement: `if (cond) { ... } [else if (cond) { ... }]* [else { ... }]?`
    if let Token::Ident(name) = &tokens[pos] {
        if name.eq_ignore_ascii_case("if") {
            return parse_if_statement(tokens, pos, ctx, mode);
        }
        // d/dt(NAME) = expr  →  DiffEq (ODE block only). Tokenizes as
        //   Ident("d") Slash Ident("dt") LParen Ident(name) RParen Eq ...
        if mode == StatementMode::Ode
            && name == "d"
            && pos + 5 < tokens.len()
            && tokens[pos + 1] == Token::Slash
            && matches!(&tokens[pos + 2], Token::Ident(s) if s == "dt")
            && tokens[pos + 3] == Token::LParen
        {
            let state_name = match &tokens[pos + 4] {
                Token::Ident(n) => n.clone(),
                other => {
                    return Err(format!("d/dt(...) expected an identifier, got {:?}", other));
                }
            };
            if tokens[pos + 5] != Token::RParen {
                return Err("d/dt(NAME): missing `)`".to_string());
            }
            let p = pos + 6;
            if p >= tokens.len() || tokens[p] != Token::Eq {
                return Err("d/dt(NAME): expected `=` after closing `)`".to_string());
            }
            let (expr, p) = parse_add_sub(tokens, p + 1, ctx)?;
            return Ok((Statement::DiffEq(state_name, expr), p));
        }
        // Plain assignment: `NAME = expr`
        if pos + 1 < tokens.len() && tokens[pos + 1] == Token::Eq {
            let var = name.clone();
            let (expr, p) = parse_add_sub(tokens, pos + 2, ctx)?;
            return Ok((Statement::Assign(var, expr), p));
        }
    }
    Err(format!(
        "Expected an assignment, an `if` block, or `d/dt(...)`, got {:?}",
        &tokens[pos]
    ))
}

fn parse_if_statement(
    tokens: &[Token],
    pos: usize,
    ctx: ParseCtx<'_>,
    mode: StatementMode,
) -> Result<(Statement, usize), String> {
    // pos points at the `if` Ident.
    let mut p = pos + 1;
    let mut branches: Vec<(Condition, Vec<Statement>)> = Vec::new();
    let mut else_body: Option<Vec<Statement>> = None;

    loop {
        // Expect `(` <cond> `)`
        if p >= tokens.len() || tokens[p] != Token::LParen {
            return Err("Expected `(` after `if`".to_string());
        }
        let (cond, p2) = parse_condition(tokens, p + 1, ctx)?;
        if p2 >= tokens.len() || tokens[p2] != Token::RParen {
            return Err("Missing `)` after if-condition".to_string());
        }
        // Expect `{` <body> `}`
        let p3 = skip_newlines(tokens, p2 + 1);
        if p3 >= tokens.len() || tokens[p3] != Token::LBrace {
            return Err("Expected `{` after if-condition".to_string());
        }
        let (body, p4) =
            parse_statements_until(tokens, p3 + 1, ctx, mode, /*stop_on_rbrace=*/ true)?;
        if p4 >= tokens.len() || tokens[p4] != Token::RBrace {
            return Err("Missing `}` at end of if-body".to_string());
        }
        branches.push((cond, body));
        p = p4 + 1;

        // Look for `else` (possibly across newlines)
        let look = skip_newlines(tokens, p);
        match tokens.get(look) {
            Some(Token::Ident(kw)) if kw.eq_ignore_ascii_case("else") => {
                let after_else = skip_newlines(tokens, look + 1);
                // `else if (...)` chains a new branch
                if let Some(Token::Ident(kw2)) = tokens.get(after_else) {
                    if kw2.eq_ignore_ascii_case("if") {
                        p = after_else + 1;
                        continue;
                    }
                }
                // `else { ... }` final block
                if tokens.get(after_else) == Some(&Token::LBrace) {
                    let (body, end) = parse_statements_until(
                        tokens,
                        after_else + 1,
                        ctx,
                        mode,
                        /*stop_on_rbrace=*/ true,
                    )?;
                    if end >= tokens.len() || tokens[end] != Token::RBrace {
                        return Err("Missing `}` at end of else-body".to_string());
                    }
                    else_body = Some(body);
                    p = end + 1;
                    break;
                }
                return Err("`else` must be followed by `if (...) {...}` or `{...}`".to_string());
            }
            _ => break,
        }
    }
    Ok((
        Statement::If {
            branches,
            else_body,
        },
        p,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ode_template desugaring + error rule (#322 Phase 0b) ─────────────────

    /// Parse `src`, asserting it fails; returns the error message. (`ParsedModel`
    /// is not `Debug`, so `unwrap_err()` can't be used directly.)
    fn parse_err(src: &str) -> String {
        match parse_full_model(src) {
            Ok(_) => panic!("expected a parse error, but the model parsed"),
            Err(e) => e,
        }
    }

    /// Two-cpt-oral parameter header (CL/V1/Q/V2/KA) plus optional extra
    /// individual params, used to build `ode_template` test models.
    fn two_cpt_oral_model(structural: &str, extra_indiv: &str, odes: &str) -> String {
        format!(
            "[parameters]\n\
             \x20 theta TVCL(3.0, 0.01, 100.0)\n\
             \x20 theta TVV1(15.0, 1.0, 500.0)\n\
             \x20 theta TVQ(3.0, 0.01, 100.0)\n\
             \x20 theta TVV2(30.0, 1.0, 500.0)\n\
             \x20 theta TVKA(1.1, 0.01, 50.0)\n\
             \x20 omega ETA_CL ~ 0.09\n\
             \x20 sigma PROP ~ 0.01 (sd)\n\n\
             [individual_parameters]\n\
             \x20 CL = TVCL * exp(ETA_CL)\n\
             \x20 V1 = TVV1\n\
             \x20 Q  = TVQ\n\
             \x20 V2 = TVV2\n\
             \x20 KA = TVKA\n{extra_indiv}\n\
             [structural_model]\n  {structural}\n\n{odes}\
             [error_model]\n  DV ~ proportional(PROP)\n"
        )
    }

    #[test]
    fn ode_template_desugars_to_ode_model() {
        let src = two_cpt_oral_model(
            "ode_template two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)",
            "",
            "",
        );
        let model = parse_full_model(&src)
            .unwrap_or_else(|e| panic!("ode_template should parse: {e}"))
            .model;
        let ode = model
            .ode_spec
            .expect("ode_template must produce an ODE model");
        assert_eq!(ode.state_names, vec!["depot", "central", "periph"]);
        // Pure disposition — no built-in input-rate forcing without an override.
        assert!(ode.input_rate.is_empty());
        // Observed compartment is central (state index 1).
        match ode.readout {
            crate::ode::OdeReadout::ObsCmt(idx) => assert_eq!(idx, 1),
            _ => panic!("expected an ObsCmt readout for ode_template"),
        }
    }

    #[test]
    fn ode_template_override_replaces_only_named_compartment() {
        // Override the generated depot with a transit input; central & periph
        // keep their generated equations (so the 3-state structure is intact and
        // exactly one transit forcing lands on the depot, compartment 0).
        let src = two_cpt_oral_model(
            "ode_template two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)",
            "  NTR = 3.0\n  MTT = 1.0\n",
            "[odes]\n  d/dt(depot) = transit(n=NTR, mtt=MTT) - KA*depot\n\n",
        );
        let model = parse_full_model(&src)
            .unwrap_or_else(|e| panic!("override should parse: {e}"))
            .model;
        let ode = model.ode_spec.expect("ODE model");
        assert_eq!(ode.state_names, vec!["depot", "central", "periph"]);
        assert_eq!(ode.input_rate.len(), 1, "exactly one transit forcing");
        assert_eq!(ode.input_rate[0].cmt, 0, "transit forces the depot (cmt 0)");
    }

    #[test]
    fn ode_template_override_unknown_compartment_errors() {
        let src = two_cpt_oral_model(
            "ode_template two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)",
            "",
            "[odes]\n  d/dt(gut) = -KA*gut\n\n",
        );
        let err = parse_err(&src);
        assert!(
            err.contains("d/dt(gut)") && err.contains("generated states"),
            "got: {err}"
        );
    }

    #[test]
    fn ode_template_missing_required_role_errors() {
        // Surfaced through the full-model parse, not just the generator.
        let src = two_cpt_oral_model(
            "ode_template two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2)",
            "",
            "",
        );
        let err = parse_err(&src);
        assert!(err.contains("requires `ka`"), "got: {err}");
    }

    #[test]
    fn ode_template_rejects_mixing_with_pk_or_ode() {
        let src = two_cpt_oral_model(
            "ode_template two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)\n  pk one_cpt_iv(cl=CL, v=V1)",
            "",
            "",
        );
        let err = parse_err(&src);
        assert!(err.contains("cannot be combined"), "got: {err}");
    }

    #[test]
    fn ode_template_malformed_arg_errors() {
        let src = two_cpt_oral_model("ode_template one_cpt_iv(cl)", "", "");
        let err = parse_err(&src);
        assert!(err.contains("malformed parameter"), "got: {err}");
    }

    #[test]
    fn ode_template_missing_parens_errors_clearly() {
        // A line that *looks* like an ode_template directive but doesn't match
        // `NAME(...)` must produce a clear "malformed ode_template" error — not
        // fall through to a confusing "No PK model found". Pair it with a
        // transit() in [odes] (the worst case: the error rule would otherwise
        // tell the user to "use ode_template" when they already are).
        let src = two_cpt_oral_model(
            "ode_template two_cpt_oral",
            "  NTR = 3.0\n  MTT = 1.0\n",
            "[odes]\n  d/dt(depot) = transit(n=NTR, mtt=MTT) - KA*depot\n\n",
        );
        let err = parse_err(&src);
        assert!(
            err.contains("malformed `ode_template`"),
            "expected a clear malformed-ode_template error, got: {err}"
        );
    }

    #[test]
    fn ode_template_duplicate_arg_errors() {
        let src = two_cpt_oral_model("ode_template one_cpt_iv(cl=CL, cl=V1)", "", "");
        let err = parse_err(&src);
        assert!(err.contains("duplicate parameter"), "got: {err}");
    }

    #[test]
    fn ode_template_multiple_lines_errors() {
        let src = two_cpt_oral_model(
            "ode_template one_cpt_iv(cl=CL, v=V1)\n  ode_template two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)",
            "",
            "",
        );
        let err = parse_err(&src);
        assert!(err.contains("more than one"), "got: {err}");
    }

    #[test]
    fn error_rule_analytical_pk_plus_transit() {
        // Analytical disposition + an ODE-only absorption function → hard error
        // pointing at ode_template, never a silent analytical→ODE swap.
        let src = two_cpt_oral_model(
            "pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)",
            "  NTR = 3.0\n  MTT = 1.0\n",
            "[odes]\n  d/dt(depot) = transit(n=NTR, mtt=MTT) - KA*depot\n\n",
        );
        let err = parse_err(&src);
        assert!(
            err.contains("ode_template"),
            "should point at ode_template: {err}"
        );
        assert!(err.contains("transit"), "should name the function: {err}");
    }

    #[test]
    fn ode_only_absorption_fn_detection() {
        let transit = vec!["d/dt(depot) = transit(n=N, mtt=M) - KA*depot".to_string()];
        assert_eq!(
            ode_only_absorption_fn_in_odes(Some(&transit)),
            Some("transit")
        );
        // The error rule only recognises functions that are actually implemented
        // (Ron #363): `igd`/`weibull` are not yet input rates, so they must NOT be
        // detected — otherwise the rule would point the user at `ode_template` for a
        // function the engine can't run. Each phase adds its name here when it lands.
        let igd = vec!["d/dt(central) = igd(mat=MAT, cv2=CV2) - (CL/V)*central".to_string()];
        assert_eq!(ode_only_absorption_fn_in_odes(Some(&igd)), None);
        let weibull = vec!["d/dt(central) = weibull(td=TD, beta=B) - (CL/V)*central".to_string()];
        assert_eq!(ode_only_absorption_fn_in_odes(Some(&weibull)), None);
        // A plain disposition ODE has no ODE-only absorption call.
        let plain = vec!["d/dt(central) = -(CL/V)*central".to_string()];
        assert_eq!(ode_only_absorption_fn_in_odes(Some(&plain)), None);
        assert_eq!(ode_only_absorption_fn_in_odes(None), None);
    }

    #[test]
    fn ode_template_rejects_mixing_with_ode() {
        // The `ode(...)` arm of the mixing guard (the `pk` arm is covered above).
        let src = two_cpt_oral_model(
            "ode_template two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)\n  \
             ode(obs_cmt=central, states=[depot, central, periph])",
            "",
            "",
        );
        let err = parse_err(&src);
        assert!(err.contains("cannot be combined"), "got: {err}");
    }

    #[test]
    fn ode_template_trailing_comma_ok() {
        // A trailing comma yields an empty `role=VAR` pair, which is skipped.
        let src = two_cpt_oral_model(
            "ode_template two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA, )",
            "",
            "",
        );
        assert!(
            parse_full_model(&src).is_ok(),
            "a trailing comma in the parameter list should be tolerated"
        );
    }

    #[test]
    fn ode_template_override_tolerates_whitespace_in_d_dt() {
        // `build_ode_spec` recognises `d/dt(NAME)` at the token level, so spacing
        // like `d/dt (central)` / `d / dt(central)` is a valid state equation
        // there. Override detection must agree — otherwise the override is missed,
        // the generated equation also survives, and the user gets a misleading
        // "duplicate d/dt(central)" error instead of their override taking effect.
        for lhs in ["d/dt (central)", "d / dt(central)", "d/dt(  central  )"] {
            let src = two_cpt_oral_model(
                "ode_template two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)",
                "",
                &format!("[odes]\n  {lhs} = KA*depot - (CL/V1 + Q/V1)*central + (Q/V2)*periph\n\n"),
            );
            let model = parse_full_model(&src)
                .unwrap_or_else(|e| panic!("spaced override `{lhs}` should parse: {e}"))
                .model;
            // Override consumed the generated central → still exactly 3 states,
            // and no duplicate-d/dt error fired.
            assert_eq!(
                model.ode_spec.expect("ODE model").state_names,
                vec!["depot", "central", "periph"],
                "override `{lhs}`"
            );
        }
    }

    #[test]
    fn ode_template_rejected_with_diffusion_block() {
        // ode_template injects obs_scale (a concentration readout), which the SDE
        // path can't carry. Reject with a message that names the real cause
        // (`ode_template` + `[diffusion]`), not the injected `[scaling]` block the
        // user never wrote.
        let src = two_cpt_oral_model(
            "ode_template two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)",
            "",
            "[diffusion]\n  central ~ 0.1\n\n",
        );
        let err = parse_err(&src);
        assert!(
            err.contains("ode_template") && err.contains("[diffusion]"),
            "diffusion error should name ode_template + [diffusion]: {err}"
        );
    }

    #[test]
    fn ode_template_transit_on_non_ddt_line_errors_with_pointer() {
        // Under ode_template, a `transit(...)` that is not the input rate of a
        // `d/dt(...)` equation (here a bare helper assignment) must not be silently
        // retained as a dead term: the input-rate extractor catches it and points
        // at the correct `d/dt(...)` form (Ron #363, finding 5 — confirmed already
        // guarded, pinned here so it can't regress).
        let src = two_cpt_oral_model(
            "ode_template two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)",
            "  NTR = 3.0\n  MTT = 1.0\n",
            "[odes]\n  rate = transit(n=NTR, mtt=MTT)\n\n",
        );
        let err = parse_err(&src);
        assert!(
            err.contains("d/dt"),
            "a transit() off a d/dt line should point at the d/dt form, got: {err}"
        );
    }

    // ── transit() input-rate parse-split (#322, design A) ────────────────────
    #[test]
    fn extract_transit_input_rate_basic() {
        let lines = vec![
            "d/dt(depot) = transit(n=NTR, mtt=MTT) - KA*depot".to_string(),
            "d/dt(central) = KA*depot - (CL/V)*central".to_string(),
        ];
        let states = vec!["depot".to_string(), "central".to_string()];
        let names = vec![
            "NTR".to_string(),
            "MTT".to_string(),
            "KA".to_string(),
            "CL".to_string(),
            "V".to_string(),
        ];
        let slots = vec![10, 11, 4, 0, 1];
        let (cleaned, forcings) =
            extract_input_rate_terms(&lines, &states, &names, &slots).unwrap();
        // transit(...) replaced by 0; the rest of the RHS is untouched.
        assert_eq!(cleaned[0], "d/dt(depot) = 0 - KA*depot");
        assert_eq!(cleaned[1], "d/dt(central) = KA*depot - (CL/V)*central");
        assert_eq!(forcings.len(), 1);
        assert_eq!(forcings[0].cmt, 0); // depot
        assert!(matches!(
            forcings[0].kind,
            crate::pk::absorption::InputRateKind::Transit
        ));
        assert_eq!(forcings[0].arg_slots, vec![10, 11]); // [n=NTR slot, mtt=MTT slot]
    }

    #[test]
    fn extract_transit_input_rate_rejects_bad_specs() {
        let states = vec!["depot".to_string()];
        let names = vec!["NTR".to_string(), "MTT".to_string()];
        let slots = vec![0, 1];
        let go =
            |line: &str| extract_input_rate_terms(&[line.to_string()], &states, &names, &slots);

        assert!(go("d/dt(depot) = transit(n=NTR, mtt=ZZZ)")
            .unwrap_err()
            .contains("not a declared individual parameter"));
        assert!(go("d/dt(depot) = transit(n=NTR, foo=MTT)")
            .unwrap_err()
            .contains("no argument `foo`"));
        assert!(go("d/dt(depot) = transit(n=NTR)")
            .unwrap_err()
            .contains("missing required argument `mtt`"));
        assert!(go("x = transit(n=NTR, mtt=MTT)")
            .unwrap_err()
            .contains("d/dt"));
        assert!(go("d/dt(depot) = FR*transit(n=NTR, mtt=MTT)")
            .unwrap_err()
            .contains("scaled"));
        // Leading unary minus: the forcing is injected `+R_in`, so a negated call
        // would silently flip the sign of the input rate. Must be rejected.
        assert!(go("d/dt(depot) = -transit(n=NTR, mtt=MTT) - KA*depot")
            .unwrap_err()
            .contains("standalone"));
        // Subtracted term (wrong sign) even without `*`/`/` scaling.
        assert!(go("d/dt(depot) = KA*depot - transit(n=NTR, mtt=MTT)")
            .unwrap_err()
            .contains("standalone"));
        // Parenthesised + scaled outside the group: the old adjacency check saw
        // only the flanking `(`/`)`, so the `/V` was silently dropped — now rejected.
        assert!(go("d/dt(depot) = (transit(n=NTR, mtt=MTT))/V")
            .unwrap_err()
            .contains("standalone"));
        assert!(go("d/dt(depot) = transit(n=NTR, mtt=MTT")
            .unwrap_err()
            .contains("unbalanced"));
        assert!(go("d/dt(depot) = transit(NTR, mtt=MTT)")
            .unwrap_err()
            .contains("name=parameter"));
        assert!(
            go("d/dt(depot) = transit(n=NTR, mtt=MTT) + transit(n=NTR, mtt=MTT)")
                .unwrap_err()
                .contains("at most one")
        );
        // Nested parens in an arg value are split correctly, then rejected as a
        // non-parameter (exercises the comma-splitter's paren-depth tracking).
        assert!(go("d/dt(depot) = transit(n=foo(a,b), mtt=MTT)")
            .unwrap_err()
            .contains("not a declared individual parameter"));
        // A word *ending* in `transit` (e.g. `xtransit(`) is not a transit() call:
        // left unchanged, no forcing recorded.
        let (kept, none) = go("d/dt(depot) = xtransit(n=NTR, mtt=MTT) - depot").unwrap();
        assert_eq!(kept[0], "d/dt(depot) = xtransit(n=NTR, mtt=MTT) - depot");
        assert!(none.is_empty());
        // No transit() → unchanged, no forcings.
        let (cleaned, f) = go("d/dt(depot) = -KA*depot").unwrap();
        assert_eq!(cleaned[0], "d/dt(depot) = -KA*depot");
        assert!(f.is_empty());

        // Positive controls for the tightened sign/scale guard: a bare additive
        // `transit(...)` is accepted whether it is the only term, the first term
        // before a `-` disposition term, or a later `+` term.
        assert!(go("d/dt(depot) = transit(n=NTR, mtt=MTT)").is_ok());
        assert!(go("d/dt(depot) = transit(n=NTR, mtt=MTT) - KA*depot").is_ok());
        assert!(go("d/dt(depot) = -KA*depot + transit(n=NTR, mtt=MTT)").is_ok());
    }

    #[test]
    fn pk_param_parser_is_strict_about_duplicates_and_malformed_pairs() {
        // The analytical `pk NAME(...)` parser now shares the strict
        // `parse_role_pairs` helper with `ode_template` (Ron #363, finding 2). It
        // previously silently dropped malformed pairs and let a duplicate role
        // last-win; both are now hard errors, single-sourced so the two paths can't
        // drift in strictness again.

        // Duplicate role → rejected (was: silent last-win, here `cl=V` would have
        // shadowed `cl=CL` with no warning).
        let dup = vec!["pk one_cpt_iv(cl=CL, cl=V)".to_string()];
        let err = parse_structural_model(&dup).unwrap_err();
        assert!(
            err.contains("duplicate parameter") && err.contains("cl"),
            "duplicate pk param should error: {err}"
        );

        // Malformed pairs (no `=`, double `=`, empty side) → rejected (was: silently
        // dropped, then surfaced only as a confusing missing-required error later).
        for bad in [
            "pk one_cpt_iv(cl, v=V)",
            "pk one_cpt_iv(cl=CL=X, v=V)",
            "pk one_cpt_iv(=CL, v=V)",
            "pk one_cpt_iv(cl=, v=V)",
        ] {
            let lines = vec![bad.to_string()];
            let err = parse_structural_model(&lines).unwrap_err();
            assert!(
                err.contains("malformed parameter"),
                "`{bad}` should be a malformed-parameter error, got: {err}"
            );
        }

        // A well-formed list (incl. a tolerated trailing comma) still parses.
        let ok = vec!["pk one_cpt_iv(cl=CL, v=V, )".to_string()];
        let (model, params) = parse_structural_model(&ok).expect("well-formed pk must parse");
        assert_eq!(model, PkModel::OneCptIv);
        assert_eq!(params.get("cl").map(String::as_str), Some("CL"));
        assert_eq!(params.get("v").map(String::as_str), Some("V"));
    }

    #[test]
    fn unknown_pk_model_name_errors() {
        // A model name that is neither a valid name/alias (`PkModel::from_name`)
        // nor a retired #176 spelling falls through to the generic "Unknown PK
        // model" error — the `None`/`other` arm of the name resolution.
        let lines = vec!["pk four_cpt_iv(cl=CL, v=V)".to_string()];
        let err = parse_structural_model(&lines).unwrap_err();
        assert!(
            err.contains("Unknown PK model") && err.contains("four_cpt_iv"),
            "got: {err}"
        );
    }

    // Issue #176 retired the split `*_iv_bolus` / `*_infusion` model names
    // in favour of a single `*_iv` per compartment count. The parser must
    // reject the old names with a migration message rather than silently
    // accept or emit a generic "unknown model" error.
    #[test]
    fn test_retired_iv_bolus_and_infusion_names_emit_migration_error() {
        let retired = [
            ("one_cpt_iv_bolus", "one"),
            ("one_compartment_iv_bolus", "one"),
            ("one_cpt_infusion", "one"),
            ("two_cpt_iv_bolus", "two"),
            ("two_compartment_infusion", "two"),
            ("three_cpt_iv_bolus", "three"),
            ("three_cpt_infusion", "three"),
        ];
        for (name, n) in retired {
            let lines = vec![format!("pk {}(cl=CL, v=V)", name)];
            let err = parse_structural_model(&lines)
                .expect_err(&format!("expected retired name `{name}` to error"));
            assert!(
                err.contains("#176") && err.contains(&format!("{n}_cpt_iv")),
                "missing migration hint for `{name}`: {err}"
            );
        }
    }

    #[test]
    fn test_unified_iv_names_parse_to_iv_variant() {
        // The new spelling must compile to the unified IV variant.
        let cases = [
            ("one_cpt_iv", PkModel::OneCptIv),
            ("one_compartment_iv", PkModel::OneCptIv),
            ("two_cpt_iv", PkModel::TwoCptIv),
            ("three_cpt_iv", PkModel::ThreeCptIv),
        ];
        for (name, expected) in cases {
            // `parse_structural_model` only resolves the model name → variant;
            // required-parameter completeness is checked later in `parse_full_model`.
            let lines = vec![format!("pk {}(cl=CL, v=V, q=Q, v2=V2, q2=Q2, v3=V3)", name)];
            let (pk_model, _) = parse_structural_model(&lines)
                .unwrap_or_else(|e| panic!("`{name}` failed to parse: {e}"));
            assert_eq!(pk_model, expected, "wrong variant for `{name}`");
        }
    }

    #[test]
    fn canonical_name_round_trips_through_parser() {
        // Every `PkModel::canonical_name()` must be a model name the parser
        // accepts and maps back to the same variant — guards `canonical_name`
        // (types.rs) against drifting from this name table.
        for model in [
            PkModel::OneCptIv,
            PkModel::OneCptOral,
            PkModel::TwoCptIv,
            PkModel::TwoCptOral,
            PkModel::ThreeCptIv,
            PkModel::ThreeCptOral,
        ] {
            let lines = vec![format!("pk {}(cl=CL, v=V)", model.canonical_name())];
            let (parsed, _) = parse_structural_model(&lines).unwrap_or_else(|e| {
                panic!(
                    "canonical_name `{}` did not parse: {e}",
                    model.canonical_name()
                )
            });
            assert_eq!(
                parsed,
                model,
                "canonical_name `{}` did not round-trip",
                model.canonical_name()
            );
        }
    }

    /// A complete analytical model wrapping `pk_line`, with every commonly
    /// referenced individual parameter defined so `build_pk_param_fn`'s per-key
    /// value validation passes and only the structural-mapping checks (required /
    /// unused, run in `parse_full_model`) are exercised. Surplus declared params
    /// just produce the usual declared-but-unused warnings, which don't affect
    /// whether parsing succeeds.
    fn structural_model_with(pk_line: &str) -> String {
        format!(
            "
[parameters]
  theta TVX(1.0, 0.001, 100.0)
  omega ETA ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL   = TVX * exp(ETA)
  V    = TVX
  V1   = TVX
  Q    = TVX
  Q2   = TVX
  V2   = TVX
  Q3   = TVX
  V3   = TVX
  KA   = TVX
  F    = TVX
  TLAG = TVX

[structural_model]
  {pk_line}

[error_model]
  DV ~ proportional(EPS)
"
        )
    }

    #[test]
    fn test_missing_required_pk_param_errors() {
        // Omitting a required structural parameter must be a hard parse error,
        // not a silent default-to-0.0 broken fit (issue #309).

        // Single missing param: exact message (matches the #309 acceptance text).
        let err = expect_parse_err(&structural_model_with("pk one_cpt_oral(cl=CL, v=V)"));
        assert_eq!(
            err,
            "[structural_model] one_cpt_oral requires `ka`, which is not mapped. \
             Add it, e.g. `ka=KA`."
        );

        // Multiple missing params: plural grammar, names all of them.
        let err = expect_parse_err(&structural_model_with("pk two_cpt_iv(cl=CL, v1=V1)"));
        assert!(err.contains("`q`"), "should name q: {err}");
        assert!(err.contains("`v2`"), "should name v2: {err}");
        assert!(
            err.contains("are not mapped") && err.contains("Add them"),
            "plural grammar expected: {err}"
        );

        // `ka` omitted on a three-cpt oral model (every other slot present).
        let err = expect_parse_err(&structural_model_with(
            "pk three_cpt_oral(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3)",
        ));
        assert!(err.contains("`ka`"), "should name ka: {err}");

        // Supplying an *optional* parameter (`f`) must not mask a *missing*
        // required one: `ka` is still reported even though `f` is present.
        let err = expect_parse_err(&structural_model_with("pk one_cpt_oral(cl=CL, v=V, f=F)"));
        assert!(err.contains("`ka`"), "should still name ka: {err}");
    }

    #[test]
    fn test_required_params_present_parses_ok() {
        // A model that maps every required slot parses without a structural error
        // — including via the `v`/`v1` and `q`/`q2` aliases (canonical-slot check)
        // and with optional `f`/`lagtime`/`alag` present. The required-
        // completeness check runs after `build_pk_param_fn` (issue #309).
        let ok_lines = [
            "pk one_cpt_iv(cl=CL, v=V)",
            "pk one_cpt_oral(cl=CL, v=V, ka=KA)",
            // optional f + lagtime present alongside the required params:
            "pk one_cpt_oral(cl=CL, v=V, ka=KA, f=F, lagtime=TLAG)",
            // `alag` alias for lagtime:
            "pk one_cpt_oral(cl=CL, v=V, ka=KA, alag=TLAG)",
            "pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)",
            // `q2` alias satisfies the `q` slot; `v` alias satisfies `v1`:
            "pk two_cpt_iv(cl=CL, v=V1, q2=Q, v2=V2)",
            "pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)",
            "pk three_cpt_iv(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3)",
            "pk three_cpt_oral(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3, ka=KA)",
        ];
        for line in ok_lines {
            assert!(
                super::parse_full_model(&structural_model_with(line)).is_ok(),
                "`{line}` should parse without a structural error"
            );
        }
    }

    #[test]
    fn test_unused_pk_param_warns() {
        // PK parameters mapped but not used by the chosen model parse Ok but warn
        // (#309): on an IV model `ka` (no absorption) is flagged. `f`
        // (bioavailability — applied to IV bolus/infusion since #327) and
        // `lagtime` (applied to every dose) are both used, so neither is flagged.
        let model_str = "
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta TVKA(1.0, 0.01, 100.0)
  theta TVF(1.0, 0.01, 10.0)
  theta TVLAG(0.5)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  KA  = TVKA
  F   = TVF
  LAG = TVLAG

[structural_model]
  pk one_cpt_iv(cl=CL, v=V, ka=KA, f=F, lagtime=LAG)

[error_model]
  DV ~ proportional(EPS)
";
        let parsed = super::parse_full_model(model_str)
            .expect("unused params are a warning, not a parse error");
        let warn = parsed
            .model
            .parse_warnings
            .iter()
            .find(|w| w.contains("does not use"))
            .unwrap_or_else(|| {
                panic!(
                    "expected an unused-param warning, got: {:?}",
                    parsed.model.parse_warnings
                )
            });
        assert!(
            warn.contains("`ka`"),
            "ka should be flagged unused on IV: {warn}"
        );
        assert!(
            !warn.contains("`f`"),
            "f is applied to IV bolus/infusion (#327) and must not be flagged: {warn}"
        );
        assert!(
            !warn.contains("`lagtime`"),
            "lagtime is applied to every dose and must not be flagged: {warn}"
        );
    }

    #[test]
    fn test_f_lagtime_warning_matrix() {
        // Pins the f/lagtime warning matrix (#309). Two distinct checks fire:
        //  - "does not use" (`consumes_pk_slot`): a param mapped in `pk(...)` but
        //    not consumed — `f` and `lagtime` are consumed by every model (#327),
        //    so neither warns; an unused structural slot (e.g. `ka` on IV) does;
        //  - "computed but never used": a param declared in [individual_parameters]
        //    but never mapped or referenced anywhere.
        // KA/F/LAG are literals so the helper declares no surplus thetas (which
        // would otherwise add unrelated unused-theta warnings).
        let model = |indiv: &str, pk: &str| -> String {
            format!(
                "
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
{indiv}

[structural_model]
  {pk}

[error_model]
  DV ~ proportional(EPS)
"
            )
        };
        let warns = |indiv: &str, pk: &str| -> Vec<String> {
            super::parse_full_model(&model(indiv, pk))
                .unwrap_or_else(|e| panic!("`{pk}` should parse: {e}"))
                .model
                .parse_warnings
        };
        let has = |ws: &[String], needle: &str| ws.iter().any(|w| w.contains(needle));
        let clv = "  CL = TVCL * exp(ETA_CL)\n  V = TVV";

        // (1) IV + `f` mapped → NOT flagged. F scales the bioavailable amount
        // on every route — IV bolus and infusion included (#327) — so it is
        // used, not inert, on IV models.
        let ws = warns(
            &format!("{clv}\n  F = 0.8"),
            "pk one_cpt_iv(cl=CL, v=V, f=F)",
        );
        assert!(
            !has(&ws, "does not use"),
            "IV + mapped f must not warn now that F applies to IV doses (#327): {ws:?}"
        );

        // (2) IV + `lagtime` mapped → NOT flagged (every model applies lagtime).
        let ws = warns(
            &format!("{clv}\n  LAG = 0.5"),
            "pk one_cpt_iv(cl=CL, v=V, lagtime=LAG)",
        );
        assert!(
            !has(&ws, "does not use"),
            "IV + lagtime must not warn: {ws:?}"
        );

        // (3) Oral + `f` mapped (defined) → no warning.
        let ws = warns(
            &format!("{clv}\n  KA = 1.0\n  F = 0.8"),
            "pk one_cpt_oral(cl=CL, v=V, ka=KA, f=F)",
        );
        assert!(
            !has(&ws, "does not use") && !has(&ws, "computed but never used"),
            "oral + used f must not warn: {ws:?}"
        );

        // (4) Oral + `f` mapped to an UNDEFINED variable → parse error (#308).
        assert!(
            super::parse_full_model(&model(
                "  CL = TVCL * exp(ETA_CL)\n  V = TVV\n  KA = 1.0",
                "pk one_cpt_oral(cl=CL, v=V, ka=KA, f=FNOPE)",
            ))
            .is_err(),
            "oral + undefined `f` reference must error"
        );

        // (5) Oral + `F` declared but NOT mapped → "computed but never used `F`".
        let ws = warns(
            &format!("{clv}\n  KA = 1.0\n  F = 0.8"),
            "pk one_cpt_oral(cl=CL, v=V, ka=KA)",
        );
        assert!(
            has(&ws, "computed but never used") && has(&ws, "`F`"),
            "oral + declared-not-mapped F should warn dead: {ws:?}"
        );

        // (6) IV + `F` declared but NOT mapped → "computed but never used `F`".
        let ws = warns(&format!("{clv}\n  F = 0.8"), "pk one_cpt_iv(cl=CL, v=V)");
        assert!(
            has(&ws, "computed but never used") && has(&ws, "`F`"),
            "IV + declared-not-mapped F should warn dead: {ws:?}"
        );
    }

    #[test]
    fn test_param_used_only_in_derived_not_flagged_dead() {
        // An individual parameter referenced only in [derived] — not mapped into
        // pk(...) and not used by another parameter — must NOT be flagged
        // "computed but never used": the census tokenizes every block (#309).
        let model = |derived: &str| -> String {
            format!(
                "
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  KA  = 1.0
  KEL = CL / V

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
{derived}
[error_model]
  DV ~ proportional(EPS)
"
            )
        };
        // KEL is referenced in [derived] → used → no dead warning.
        let p =
            super::parse_full_model(&model("\n[derived]\n  RATE = KEL * 2\n")).expect("parse ok");
        assert!(
            !p.model
                .parse_warnings
                .iter()
                .any(|w| w.contains("computed but never used")),
            "KEL is used in [derived] and must not be flagged dead: {:?}",
            p.model.parse_warnings
        );
        // Negative control: with no [derived] reference, KEL IS dead.
        let p2 = super::parse_full_model(&model("")).expect("parse ok");
        assert!(
            p2.model
                .parse_warnings
                .iter()
                .any(|w| w.contains("computed but never used") && w.contains("`KEL`")),
            "without the [derived] use KEL must be flagged dead: {:?}",
            p2.model.parse_warnings
        );
    }

    #[test]
    fn test_ode_dead_indiv_param_warns() {
        // #315: an ODE [individual_parameters] entry never referenced in the
        // [odes] RHS (and not engine-applied f/lagtime) is routed to a free slot,
        // computed every evaluation, but never read — silently inert. The #310
        // "computed but never used" census skipped ODE models; it now covers them.
        let content = r#"
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta TVKE(0.5, 0.001, 10.0)
  omega ETA_CL ~ 0.1
  omega ETA_KE ~ 0.09
  sigma EPS ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KE = TVKE * exp(ETA_KE)   # never referenced in [odes]

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -(CL/V) * central

[error_model]
  DV ~ proportional(EPS)
"#;
        let ws = super::parse_full_model(content)
            .expect("a dead ODE param is a warning, not an error")
            .model
            .parse_warnings;
        let dead: Vec<&String> = ws
            .iter()
            .filter(|w| w.contains("computed but never used"))
            .collect();
        assert_eq!(
            dead.len(),
            1,
            "exactly one dead-param warning expected, got: {ws:?}"
        );
        let w = dead[0];
        assert!(w.contains("`KE`"), "warning must name KE: {w}");
        // The params actually used in the RHS must not be flagged.
        assert!(
            !w.contains("`CL`") && !w.contains("`V`"),
            "used params must not be flagged dead: {w}"
        );
        // ODE-flavored guidance points at [odes], not the analytical pk(...) map.
        assert!(
            w.contains("[odes]") && !w.contains("pk(...)"),
            "ODE warning should reference [odes], not pk(...): {w}"
        );
    }

    #[test]
    fn test_ode_engine_applied_f_lagtime_not_flagged_dead() {
        // #315 carve-out: F and lagtime on an ODE model are routed to the
        // engine-reserved PK_IDX_F / PK_IDX_LAGTIME slots (`ode_param_slots`) and
        // applied to the dose by the engine without ever appearing in the [odes]
        // RHS, so their textual absence must NOT flag them dead. Mirrors
        // examples/bioavailability_ode.ferx and examples/warfarin_ode_lagtime.ferx.
        let content = r#"
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta TVKA(1.0, 0.001, 10.0)
  theta THETA_F(0.0, -10.0, 10.0)
  theta TVLAG(0.5, 0.001, 10.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL      = TVCL * exp(ETA_CL)
  V       = TVV
  KA      = TVKA
  F       = inv_logit(THETA_F)
  LAGTIME = TVLAG

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central

[error_model]
  DV ~ proportional(EPS)
"#;
        let ws = super::parse_full_model(content)
            .expect("engine-applied F/lagtime on an ODE model must parse")
            .model
            .parse_warnings;
        assert!(
            !ws.iter().any(|w| w.contains("computed but never used")),
            "engine-applied F/LAGTIME on an ODE model must not be flagged dead: {ws:?}"
        );
    }

    #[test]
    fn test_ode_multiple_dead_params_use_plural_message() {
        // #315: two+ dead ODE params share one warning and use the plural grammar
        // ("are … they … them") in the ODE-flavored message. Locks the plural
        // branch + name list for ODE (the singular case is covered above).
        let content = r#"
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta TVKE(0.5, 0.001, 10.0)
  theta TVKE2(0.2, 0.001, 10.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  KE  = TVKE      # never referenced in [odes]
  KE2 = TVKE2     # never referenced in [odes]

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -(CL/V) * central

[error_model]
  DV ~ proportional(EPS)
"#;
        let ws = super::parse_full_model(content)
            .expect("dead ODE params are a warning, not an error")
            .model
            .parse_warnings;
        let dead: Vec<&String> = ws
            .iter()
            .filter(|w| w.contains("computed but never used"))
            .collect();
        assert_eq!(
            dead.len(),
            1,
            "both dead params share a single warning, got: {ws:?}"
        );
        let w = dead[0];
        // Both names listed (sorted, comma-joined), neither used param flagged.
        assert!(
            w.contains("`KE`") && w.contains("`KE2`"),
            "both dead params must be named: {w}"
        );
        assert!(
            !w.contains("`CL`") && !w.contains("`V`"),
            "used params not flagged: {w}"
        );
        // Plural grammar + ODE-flavored guidance.
        assert!(
            w.contains("are computed but never used") && w.contains("remove them"),
            "plural ODE message expected: {w}"
        );
        assert!(
            w.contains("[odes]") && !w.contains("pk(...)"),
            "ODE plural warning should reference [odes], not pk(...): {w}"
        );
    }

    #[test]
    fn test_undeclared_name_in_derived_is_accepted_silently() {
        // [derived] uses `fallback_covariate = false`: an unknown identifier
        // becomes a Variable resolved at output time, not a covariate. So an
        // undeclared name used only in [derived] parses without error and without
        // any warning — unlike an ODE RHS (which rejects covariate references), or
        // [individual_parameters] when a [covariates] block is present (strict
        // mode, which warns on an undeclared covariate).
        let src = "
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = 1.0

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[derived]
  RATIO = MYSTERY_NAME / CL

[error_model]
  DV ~ proportional(EPS)
";
        let parsed =
            super::parse_full_model(src).expect("undeclared name in [derived] should parse");
        assert!(
            !parsed
                .model
                .parse_warnings
                .iter()
                .any(|w| w.contains("MYSTERY_NAME")),
            "undeclared name in [derived] is a silent Variable, no warning: {:?}",
            parsed.model.parse_warnings
        );
    }

    #[test]
    fn test_pk_key_relevance_matrix() {
        // Exhaustively pins every (model, PK key) cell, classified
        // R(equired) / O(ptional, used) / U(nused) — hardcoded independently of
        // `consumes_pk_slot`/`required_pk_params` so the test can't co-drift with
        // the code it checks:
        //   R → omitting the key is an error naming it;
        //   O → mapping the key is silently accepted (no "does not use" warning);
        //   U → mapping the key warns "does not use `key`".
        // Keys use each model's canonical slot name (`v`/`v1`, `q`/`q2`), so the
        // required-omission error (which echoes the canonical name) matches.
        let matrix: &[(&str, [(&str, char); 9])] = &[
            (
                "one_cpt_iv",
                [
                    ("cl", 'R'),
                    ("v", 'R'),
                    ("q", 'U'),
                    ("v2", 'U'),
                    ("ka", 'U'),
                    ("f", 'O'),
                    ("q3", 'U'),
                    ("v3", 'U'),
                    ("lagtime", 'O'),
                ],
            ),
            (
                "one_cpt_oral",
                [
                    ("cl", 'R'),
                    ("v", 'R'),
                    ("q", 'U'),
                    ("v2", 'U'),
                    ("ka", 'R'),
                    ("f", 'O'),
                    ("q3", 'U'),
                    ("v3", 'U'),
                    ("lagtime", 'O'),
                ],
            ),
            (
                "two_cpt_iv",
                [
                    ("cl", 'R'),
                    ("v1", 'R'),
                    ("q", 'R'),
                    ("v2", 'R'),
                    ("ka", 'U'),
                    ("f", 'O'),
                    ("q3", 'U'),
                    ("v3", 'U'),
                    ("lagtime", 'O'),
                ],
            ),
            (
                "two_cpt_oral",
                [
                    ("cl", 'R'),
                    ("v1", 'R'),
                    ("q", 'R'),
                    ("v2", 'R'),
                    ("ka", 'R'),
                    ("f", 'O'),
                    ("q3", 'U'),
                    ("v3", 'U'),
                    ("lagtime", 'O'),
                ],
            ),
            (
                "three_cpt_iv",
                [
                    ("cl", 'R'),
                    ("v1", 'R'),
                    ("q2", 'R'),
                    ("v2", 'R'),
                    ("ka", 'U'),
                    ("f", 'O'),
                    ("q3", 'R'),
                    ("v3", 'R'),
                    ("lagtime", 'O'),
                ],
            ),
            (
                "three_cpt_oral",
                [
                    ("cl", 'R'),
                    ("v1", 'R'),
                    ("q2", 'R'),
                    ("v2", 'R'),
                    ("ka", 'R'),
                    ("f", 'O'),
                    ("q3", 'R'),
                    ("v3", 'R'),
                    ("lagtime", 'O'),
                ],
            ),
        ];
        // `cl` carries the theta/eta so [parameters] has no unused declarations;
        // every other mapped key is a literal so there are no surplus declared
        // individual parameters to flag as dead.
        let build = |model: &str, keys: &[&str]| -> String {
            let indiv: String = keys
                .iter()
                .map(|k| {
                    let var = k.to_uppercase();
                    if *k == "cl" {
                        format!("  {var} = TVX * exp(ETA)\n")
                    } else {
                        format!("  {var} = 1.0\n")
                    }
                })
                .collect();
            let pk = keys
                .iter()
                .map(|k| format!("{k}={}", k.to_uppercase()))
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "
[parameters]
  theta TVX(1.0, 0.001, 100.0)
  omega ETA ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
{indiv}
[structural_model]
  pk {model}({pk})

[error_model]
  DV ~ proportional(EPS)
"
            )
        };
        for entry in matrix {
            let model = entry.0;
            let cells = &entry.1;
            let required: Vec<&str> = cells.iter().filter(|c| c.1 == 'R').map(|c| c.0).collect();
            for cell in cells {
                let key = cell.0;
                match cell.1 {
                    'R' => {
                        // Map every required key EXCEPT this one → error naming it.
                        let keys: Vec<&str> =
                            required.iter().copied().filter(|k| *k != key).collect();
                        let err = super::parse_full_model(&build(model, &keys))
                            .err()
                            .unwrap_or_else(|| {
                                panic!("{model}: omitting required `{key}` must error")
                            });
                        assert!(
                            err.contains(&format!("`{key}`")),
                            "{model}: omitting `{key}` should name it, got: {err}"
                        );
                    }
                    class @ ('O' | 'U') => {
                        // Map every required key PLUS this one → warn iff unused.
                        let mut keys: Vec<&str> = required.clone();
                        keys.push(key);
                        let parsed = super::parse_full_model(&build(model, &keys))
                            .unwrap_or_else(|e| panic!("{model} + `{key}` should parse: {e}"));
                        let warned =
                            parsed.model.parse_warnings.iter().any(|w| {
                                w.contains("does not use") && w.contains(&format!("`{key}`"))
                            });
                        assert_eq!(
                            warned,
                            class == 'U',
                            "{model} `{key}` ({class}): unexpected warning state, got: {:?}",
                            parsed.model.parse_warnings
                        );
                    }
                    other => panic!("bad classification `{other}`"),
                }
            }
        }
    }

    #[test]
    fn test_parse_method_single() {
        let opts = parse_fit_options(&["method = focei".to_string()]).unwrap();
        assert_eq!(opts.method, EstimationMethod::FoceI);
        assert!(opts.methods.is_empty());
        assert!(opts.interaction);
    }

    #[test]
    fn test_parse_inits_from_nca() {
        use crate::suggest_start::NcaInit;
        // Default is off.
        assert_eq!(FitOptions::default().inits_from_nca, None);
        // Boolean true maps to the sweep strategy.
        let opts = parse_fit_options(&["inits_from_nca = true".to_string()]).unwrap();
        assert_eq!(opts.inits_from_nca, Some(NcaInit::Sweep));
        // Boolean false disables.
        let opts = parse_fit_options(&["inits_from_nca = false".to_string()]).unwrap();
        assert_eq!(opts.inits_from_nca, None);
        // Explicit strategy names.
        let opts = parse_fit_options(&["inits_from_nca = nca".to_string()]).unwrap();
        assert_eq!(opts.inits_from_nca, Some(NcaInit::Nca));
        let opts = parse_fit_options(&["inits_from_nca = nca_sweep".to_string()]).unwrap();
        assert_eq!(opts.inits_from_nca, Some(NcaInit::Sweep));
        let opts = parse_fit_options(&["inits_from_nca = nca_ebe".to_string()]).unwrap();
        assert_eq!(opts.inits_from_nca, Some(NcaInit::Ebe));
        // Unknown value is rejected.
        assert!(parse_fit_options(&["inits_from_nca = bogus".to_string()]).is_err());
    }

    #[test]
    fn test_parse_method_chain() {
        let opts = parse_fit_options(&["method = [saem, focei]".to_string()]).unwrap();
        assert_eq!(
            opts.methods,
            vec![EstimationMethod::Saem, EstimationMethod::FoceI]
        );
        assert_eq!(opts.method, EstimationMethod::FoceI);
        assert!(opts.interaction);
    }

    #[test]
    fn test_parse_method_chain_final_foce() {
        let opts = parse_fit_options(&["method = [saem, foce]".to_string()]).unwrap();
        assert_eq!(opts.method, EstimationMethod::Foce);
        assert!(!opts.interaction);
    }

    #[test]
    fn test_parse_method_chain_empty_rejected() {
        assert!(parse_fit_options(&["method = []".to_string()]).is_err());
    }

    #[test]
    fn test_parse_method_unknown_rejected() {
        assert!(parse_fit_options(&["method = [foce, wibble]".to_string()]).is_err());
    }

    #[test]
    fn test_method_chain_helper_default() {
        let opts = FitOptions::default();
        assert_eq!(opts.method_chain(), vec![EstimationMethod::FoceI]);
    }

    #[test]
    fn test_method_chain_helper_populated() {
        let mut opts = FitOptions::default();
        opts.methods = vec![EstimationMethod::Saem, EstimationMethod::FoceI];
        assert_eq!(
            opts.method_chain(),
            vec![EstimationMethod::Saem, EstimationMethod::FoceI]
        );
    }

    #[test]
    fn test_parse_threads_positive() {
        let opts = parse_fit_options(&["threads = 4".to_string()]).unwrap();
        assert_eq!(opts.threads, Some(4));
    }

    #[test]
    fn test_parse_threads_auto() {
        let opts = parse_fit_options(&["threads = auto".to_string()]).unwrap();
        assert_eq!(opts.threads, None);
        // Case-insensitive.
        let opts = parse_fit_options(&["threads = AUTO".to_string()]).unwrap();
        assert_eq!(opts.threads, None);
    }

    #[test]
    fn test_parse_threads_zero_means_auto() {
        // `threads = 0` is treated as "leave rayon default alone",
        // matching the R binding's `threads <= 0` sentinel.
        let opts = parse_fit_options(&["threads = 0".to_string()]).unwrap();
        assert_eq!(opts.threads, None);
    }

    #[test]
    fn test_parse_threads_invalid_errors() {
        // Strict parsing: malformed threads values raise a parse error
        // rather than silently falling back to `None` (the pre-refactor
        // `.parse().ok().filter(...)` behavior was a typo trap).
        assert!(parse_fit_options(&["threads = -1".to_string()]).is_err());
        assert!(parse_fit_options(&["threads = wibble".to_string()]).is_err());
    }

    #[test]
    fn test_parse_threads_default_is_none() {
        // No `threads` line → None (rayon global pool, one worker per logical CPU).
        let opts = parse_fit_options(&["method = focei".to_string()]).unwrap();
        assert_eq!(opts.threads, None);
    }

    // ── mu_referencing fit option ────────────────────────────────────────

    #[test]
    fn test_parse_mu_referencing_default_true() {
        let opts = parse_fit_options(&["method = foce".to_string()]).unwrap();
        assert!(opts.mu_referencing);
    }

    #[test]
    fn test_parse_mu_referencing_false() {
        let opts = parse_fit_options(&["mu_referencing = false".to_string()]).unwrap();
        assert!(!opts.mu_referencing);
    }

    #[test]
    fn test_parse_mu_referencing_accepts_synonyms() {
        for raw in &["true", "TRUE", "1", "yes", "on"] {
            let opts = parse_fit_options(&[format!("mu_referencing = {}", raw)]).unwrap();
            assert!(opts.mu_referencing, "{} should enable", raw);
        }
        for raw in &["false", "FALSE", "0", "no", "off"] {
            let opts = parse_fit_options(&[format!("mu_referencing = {}", raw)]).unwrap();
            assert!(!opts.mu_referencing, "{} should disable", raw);
        }
    }

    #[test]
    fn test_parse_mu_referencing_invalid_rejected() {
        assert!(parse_fit_options(&["mu_referencing = wibble".to_string()]).is_err());
    }

    // ── apply_fit_option (shared dispatch used by the R wrapper's `settings`
    //    argument and by parse_fit_options) ────────────────────────────────

    #[test]
    fn test_apply_fit_option_known_applies() {
        let mut opts = FitOptions::default();
        assert_eq!(
            apply_fit_option(&mut opts, "n_exploration", "200"),
            Ok(true)
        );
        assert_eq!(opts.saem_n_exploration, 200);

        assert_eq!(
            apply_fit_option(&mut opts, "n_convergence", "400"),
            Ok(true)
        );
        assert_eq!(opts.saem_n_convergence, 400);

        assert_eq!(apply_fit_option(&mut opts, "omega_burnin", "30"), Ok(true));
        assert_eq!(opts.saem_omega_burnin, 30);
    }

    #[test]
    fn test_apply_fit_option_unknown_key_returns_false() {
        let mut opts = FitOptions::default();
        // Typo / unknown → Ok(false). Caller decides whether to error out.
        assert_eq!(
            apply_fit_option(&mut opts, "n_exploraton", "200"),
            Ok(false)
        );
        // `method` is deliberately excluded (list-chain syntax is handled
        // in the block parser); treat it as unknown here.
        assert_eq!(apply_fit_option(&mut opts, "method", "focei"), Ok(false));
    }

    #[test]
    fn test_apply_fit_option_malformed_value_errors() {
        let mut opts = FitOptions::default();
        assert!(apply_fit_option(&mut opts, "n_exploration", "oops").is_err());
        assert!(apply_fit_option(&mut opts, "covariance", "maybe").is_err());
        assert!(apply_fit_option(&mut opts, "gn_lambda", "x").is_err());
        assert!(apply_fit_option(&mut opts, "optimizer", "does_not_exist").is_err());
        assert!(apply_fit_option(&mut opts, "bloq_method", "nope").is_err());
        assert!(apply_fit_option(&mut opts, "threads", "-1").is_err());
        assert!(apply_fit_option(&mut opts, "sir_df", "0.0").is_err());
        assert!(apply_fit_option(&mut opts, "sir_df", "0.5").is_err());
        // Failed apply must not mutate — default preserved.
        assert_eq!(opts.saem_n_exploration, 150);
    }

    #[test]
    fn test_apply_fit_option_ode_solver_tolerances() {
        let mut opts = FitOptions::default();
        // Defaults match OdeSolverOptions::default() (engine default unchanged).
        assert_eq!(opts.ode_reltol, 1e-4);
        assert_eq!(opts.ode_abstol, 1e-6);
        assert_eq!(opts.ode_max_steps, 10_000);

        assert_eq!(apply_fit_option(&mut opts, "ode_reltol", "1e-10"), Ok(true));
        assert_eq!(apply_fit_option(&mut opts, "ode_abstol", "1e-12"), Ok(true));
        assert_eq!(
            apply_fit_option(&mut opts, "ode_max_steps", "200000"),
            Ok(true)
        );
        assert_eq!(opts.ode_reltol, 1e-10);
        assert_eq!(opts.ode_abstol, 1e-12);
        assert_eq!(opts.ode_max_steps, 200_000);

        // Non-positive / non-finite / zero are rejected, and a failed apply
        // must not mutate the previously-set value.
        assert!(apply_fit_option(&mut opts, "ode_reltol", "0").is_err());
        assert!(apply_fit_option(&mut opts, "ode_reltol", "-1e-9").is_err());
        assert!(apply_fit_option(&mut opts, "ode_abstol", "nan").is_err());
        assert!(apply_fit_option(&mut opts, "ode_max_steps", "0").is_err());
        assert!(apply_fit_option(&mut opts, "ode_max_steps", "x").is_err());
        assert_eq!(opts.ode_reltol, 1e-10);
        assert_eq!(opts.ode_max_steps, 200_000);
    }

    #[test]
    fn test_ode_reltol_from_fit_options_reaches_ode_spec() {
        // [fit_options] ODE solver tolerances must be baked onto
        // OdeSpec.solver_opts by the parser (via sync_ode_solver_opts) so the
        // integrator - including predict(), which receives no fit options -
        // uses the requested accuracy.
        let base = r#"
[parameters]
  theta TVCL(1.0, 0.01, 10.0)
  theta TVV(10.0, 0.1, 100.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        // Default: OdeSpec inherits OdeSolverOptions::default().
        let def = parse_full_model(base).unwrap();
        let s = def.model.ode_spec.as_ref().unwrap().solver_opts;
        assert_eq!(s.reltol, 1e-4);
        assert_eq!(s.abstol, 1e-6);
        assert_eq!(s.max_steps, 10_000);

        // Override via [fit_options].
        let with_opts = format!(
            "{base}\n[fit_options]\n  ode_reltol = 1e-9\n  ode_abstol = 1e-11\n  ode_max_steps = 50000\n"
        );
        let p = parse_full_model(&with_opts).unwrap();
        let s2 = p.model.ode_spec.as_ref().unwrap().solver_opts;
        assert_eq!(s2.reltol, 1e-9);
        assert_eq!(s2.abstol, 1e-11);
        assert_eq!(s2.max_steps, 50_000);
    }

    #[test]
    fn test_imp_method_and_is_options_parse() {
        // `methods = [focei, imp]` plus the four `is_*` keys must apply
        // cleanly and produce no `unsupported_keys_warnings`.
        let opts = parse_fit_options(&[
            "method = [focei, imp]".to_string(),
            "is_samples = 500".to_string(),
            "is_proposal_df = 4.0".to_string(),
            "is_seed = 99".to_string(),
            "is_low_ess_threshold = 0.2".to_string(),
            "covariance = false".to_string(),
            "verbose = false".to_string(),
        ])
        .expect("parse must succeed");
        assert_eq!(
            opts.methods,
            vec![EstimationMethod::FoceI, EstimationMethod::Imp]
        );
        assert_eq!(opts.is_samples, 500);
        assert_eq!(opts.is_proposal_df, 4.0);
        assert_eq!(opts.is_seed, Some(99));
        assert_eq!(opts.is_low_ess_threshold, 0.2);
        // No "ignored option" warnings — keys are method-specific to Imp,
        // and Imp is in the chain.
        assert!(opts.unsupported_keys_warnings().is_empty());
    }

    #[test]
    fn test_is_options_validate_ranges() {
        let mut opts = FitOptions::default();
        assert!(apply_fit_option(&mut opts, "is_samples", "1").is_err()); // < 2
        assert!(apply_fit_option(&mut opts, "is_proposal_df", "0.5").is_err()); // < 1
        assert!(apply_fit_option(&mut opts, "is_low_ess_threshold", "1.5").is_err()); // > 1
        assert!(apply_fit_option(&mut opts, "is_low_ess_threshold", "-0.1").is_err()); // < 0
                                                                                       // Defaults preserved after a failed apply.
        assert_eq!(opts.is_samples, 1000);
    }

    #[test]
    fn test_impmap_method_tokens_parse() {
        for tok in [
            "impmap",
            "importance_sampling_map",
            "importance-sampling-map",
        ] {
            let opts = parse_fit_options(&[format!("method = {tok}")]).expect("parse must succeed");
            assert_eq!(
                opts.method,
                EstimationMethod::Impmap,
                "token `{tok}` must map to Impmap"
            );
        }
        // `imp` must still resolve to the evaluation-only stage, not Impmap.
        let opts = parse_fit_options(&["method = imp".to_string()]).expect("parse");
        assert_eq!(opts.method, EstimationMethod::Imp);
    }

    #[test]
    fn test_impmap_options_parse_and_validate() {
        let opts = parse_fit_options(&[
            "method = importance_sampling_map".to_string(),
            "impmap_iterations = 80".to_string(),
            "impmap_samples = 250".to_string(),
            "impmap_proposal_df = 8.0".to_string(),
            "impmap_averaging = 30".to_string(),
            "impmap_seed = 77".to_string(),
            "impmap_low_ess_threshold = 0.2".to_string(),
            "covariance = false".to_string(),
        ])
        .expect("parse must succeed");
        assert_eq!(opts.method, EstimationMethod::Impmap);
        assert_eq!(opts.impmap_iterations, 80);
        assert_eq!(opts.impmap_samples, 250);
        assert_eq!(opts.impmap_proposal_df, 8.0);
        assert_eq!(opts.impmap_averaging, 30);
        assert_eq!(opts.impmap_seed, Some(77));
        assert_eq!(opts.impmap_low_ess_threshold, 0.2);
        // All keys are method-specific to Impmap and Impmap is selected.
        assert!(opts.unsupported_keys_warnings().is_empty());

        // `normal` / `mvn` select the multivariate-normal proposal (df = +inf).
        for kw in ["normal", "mvn", "NORMAL"] {
            let mut o = FitOptions::default();
            assert!(apply_fit_option(&mut o, "impmap_proposal_df", kw).is_ok());
            assert!(o.impmap_proposal_df.is_infinite());
        }

        // Range validation.
        let mut o = FitOptions::default();
        assert!(apply_fit_option(&mut o, "impmap_samples", "1").is_err()); // < 2
        assert!(apply_fit_option(&mut o, "impmap_iterations", "0").is_err()); // < 1
        assert!(apply_fit_option(&mut o, "impmap_proposal_df", "0.5").is_err()); // < 1
        assert!(apply_fit_option(&mut o, "impmap_low_ess_threshold", "1.5").is_err()); // > 1
                                                                                       // Default (MVN) preserved after the failed applies.
        assert!(o.impmap_proposal_df.is_infinite());
    }

    #[test]
    fn test_sir_df_valid_and_invalid() {
        let mut opts = FitOptions::default();
        assert!(apply_fit_option(&mut opts, "sir_df", "5.0").is_ok());
        assert_eq!(opts.sir_df, 5.0);
        assert!(apply_fit_option(&mut opts, "sir_df", "1.0").is_ok());
        assert!(apply_fit_option(&mut opts, "sir_df", "0.9").is_err());
        assert!(apply_fit_option(&mut opts, "sir_df", "0.0").is_err());
    }

    #[test]
    fn test_apply_fit_option_bool_variants() {
        let mut opts = FitOptions::default();
        for v in ["true", "True", "TRUE", "yes", "1", "t"] {
            opts.sir = false;
            assert_eq!(apply_fit_option(&mut opts, "sir", v), Ok(true));
            assert!(opts.sir, "value `{v}` should parse as true");
        }
        for v in ["false", "False", "no", "0", "f"] {
            opts.sir = true;
            assert_eq!(apply_fit_option(&mut opts, "sir", v), Ok(true));
            assert!(!opts.sir, "value `{v}` should parse as false");
        }
    }

    #[test]
    fn test_apply_fit_option_seed_null_clears() {
        let mut opts = FitOptions::default();
        opts.saem_seed = Some(7);
        // R sends NULL/NA through as the literal "null" / "na".
        assert_eq!(apply_fit_option(&mut opts, "seed", "null"), Ok(true));
        assert_eq!(opts.saem_seed, None);

        assert_eq!(apply_fit_option(&mut opts, "seed", "42"), Ok(true));
        assert_eq!(opts.saem_seed, Some(42));

        // `saem_seed` is accepted as an alias so R users can use either spelling.
        assert_eq!(apply_fit_option(&mut opts, "saem_seed", "99"), Ok(true));
        assert_eq!(opts.saem_seed, Some(99));
    }

    #[test]
    fn test_apply_fit_option_npde() {
        let mut opts = FitOptions::default();
        assert_eq!(opts.npde_nsim, 0, "NPDE is off by default");
        assert_eq!(apply_fit_option(&mut opts, "npde_nsim", "1000"), Ok(true));
        assert_eq!(opts.npde_nsim, 1000);
        assert_eq!(apply_fit_option(&mut opts, "npde_seed", "12345"), Ok(true));
        assert_eq!(opts.npde_seed, Some(12345));
        // NULL/NA from R clears the seed back to the default.
        assert_eq!(apply_fit_option(&mut opts, "npde_seed", "null"), Ok(true));
        assert_eq!(opts.npde_seed, None);
    }

    #[test]
    fn test_apply_fit_option_threads_variants() {
        let mut opts = FitOptions::default();
        assert_eq!(apply_fit_option(&mut opts, "threads", "4"), Ok(true));
        assert_eq!(opts.threads, Some(4));

        assert_eq!(apply_fit_option(&mut opts, "threads", "auto"), Ok(true));
        assert_eq!(opts.threads, None);

        opts.threads = Some(4);
        assert_eq!(apply_fit_option(&mut opts, "threads", "0"), Ok(true));
        assert_eq!(opts.threads, None);
    }

    #[test]
    fn test_apply_fit_option_optimizer_and_bloq() {
        let mut opts = FitOptions::default();
        assert_eq!(apply_fit_option(&mut opts, "optimizer", "lbfgs"), Ok(true));
        assert_eq!(opts.optimizer, Optimizer::NloptLbfgs);

        assert_eq!(apply_fit_option(&mut opts, "bloq", "m3"), Ok(true));
        assert_eq!(opts.bloq_method, BloqMethod::M3);
    }

    // ── Warn on options that don't apply to the selected estimation method.
    //    These fire from inside fit() via FitOptions::unsupported_keys_warnings,
    //    so we check the raw mechanism here without running a full fit. ────

    #[test]
    fn test_unsupported_saem_key_under_focei_warns() {
        let opts = parse_fit_options(&[
            "method = focei".to_string(),
            "n_convergence = 300".to_string(),
        ])
        .unwrap();
        let warnings = opts.unsupported_keys_warnings();
        assert_eq!(warnings.len(), 1, "got: {:?}", warnings);
        let w = &warnings[0];
        assert!(w.contains("n_convergence"), "got: {w}");
        assert!(w.contains("FOCEI"), "got: {w}");
        assert!(w.contains("will be ignored"), "got: {w}");
        // Mentions a FOCE-applicable key so the user can see what's available.
        assert!(w.contains("optimizer"), "got: {w}");
        // Does NOT suggest SAEM-specific keys as available.
        assert!(!w.contains("n_mh_steps"), "got: {w}");
    }

    #[test]
    fn test_unsupported_focei_key_under_saem_warns() {
        let opts =
            parse_fit_options(&["method = saem".to_string(), "optimizer = lbfgs".to_string()])
                .unwrap();
        let warnings = opts.unsupported_keys_warnings();
        assert_eq!(warnings.len(), 1, "got: {:?}", warnings);
        let w = &warnings[0];
        assert!(w.contains("optimizer"), "got: {w}");
        assert!(w.contains("SAEM"), "got: {w}");
        assert!(w.contains("n_exploration"), "got: {w}");
    }

    #[test]
    fn test_applicable_key_in_chain_no_warning() {
        // methods = [saem, focei]: n_convergence applies to SAEM, optimizer
        // applies to FOCEI, so neither should warn.
        let opts = parse_fit_options(&[
            "method = [saem, focei]".to_string(),
            "n_convergence = 300".to_string(),
            "optimizer = lbfgs".to_string(),
        ])
        .unwrap();
        assert!(opts.unsupported_keys_warnings().is_empty());
    }

    #[test]
    fn test_common_keys_never_warn() {
        // Covariance/verbose/sir/bloq/threads/mu_referencing apply to every
        // method — they must not produce a warning regardless of method.
        for method in ["foce", "focei", "gn", "gn_hybrid", "saem"] {
            let opts = parse_fit_options(&[
                format!("method = {method}"),
                "covariance = true".to_string(),
                // The whole covariance family is framework-wide — none of these
                // are method-specific, so none must warn under any method.
                "covariance_method = s".to_string(),
                "covariance_fallback = sir".to_string(),
                "covariance_ofv_hessian = true".to_string(),
                "verbose = false".to_string(),
                "sir = true".to_string(),
                "bloq_method = m3".to_string(),
                "threads = 2".to_string(),
                "mu_referencing = false".to_string(),
            ])
            .unwrap();
            let w = opts.unsupported_keys_warnings();
            assert!(
                w.is_empty(),
                "method={method} produced unexpected warnings: {:?}",
                w
            );
        }
    }

    #[test]
    fn test_unsupported_warning_omits_framework_keys() {
        // Framework-wide keys (covariance/verbose/sir/bloq/threads/mu_referencing)
        // are exposed as top-level wrapper args, not as method-specific settings.
        // The warning's "Method-specific options" list must not include them —
        // listing `covariance` next to `optimizer` would conflate the layers.
        let opts = parse_fit_options(&[
            "method = focei".to_string(),
            "n_convergence = 300".to_string(),
        ])
        .unwrap();
        let w = &opts.unsupported_keys_warnings()[0];
        for framework in [
            "covariance",
            "verbose",
            "sir",
            "sir_samples",
            "sir_resamples",
            "sir_seed",
            "sir_keep_samples",
            "bloq_method",
            "bloq",
            "threads",
            "mu_referencing",
        ] {
            assert!(
                !w.contains(framework),
                "framework key `{framework}` leaked into method-specific list: {w}"
            );
        }
        // And it uses the new phrasing, not the old "Available options".
        assert!(w.contains("Method-specific options"), "got: {w}");
    }

    #[test]
    fn test_gn_lambda_under_focei_warns() {
        let opts =
            parse_fit_options(&["method = focei".to_string(), "gn_lambda = 0.05".to_string()])
                .unwrap();
        let warnings = opts.unsupported_keys_warnings();
        assert_eq!(warnings.len(), 1, "got: {:?}", warnings);
        assert!(warnings[0].contains("gn_lambda"));
    }

    #[test]
    fn test_no_warning_when_no_keys_set() {
        // Bare default FitOptions (no parser path) must not conjure warnings.
        let opts = FitOptions::default();
        assert!(opts.unsupported_keys_warnings().is_empty());
    }

    #[test]
    fn test_stagnation_guard_recognized_by_nlopt_methods() {
        // `stagnation_guard` lives in the NLopt outer-loop code, so the
        // unused-key guard must accept it under FOCE/FOCEI and the FoceGn
        // hybrid (which has a FOCEI polish phase) — and must flag it under
        // pure GN and SAEM, which don't touch the NLopt path.
        for method in ["foce", "focei", "gn_hybrid"] {
            let opts = parse_fit_options(&[
                format!("method = {}", method),
                "stagnation_guard = false".to_string(),
            ])
            .unwrap();
            let warnings = opts.unsupported_keys_warnings();
            assert!(
                warnings.is_empty(),
                "method={method} should not warn for stagnation_guard: {:?}",
                warnings
            );
        }
        for method in ["gn", "saem"] {
            let opts = parse_fit_options(&[
                format!("method = {}", method),
                "stagnation_guard = false".to_string(),
            ])
            .unwrap();
            let warnings = opts.unsupported_keys_warnings();
            assert_eq!(
                warnings.len(),
                1,
                "method={method}: expected exactly one warning, got: {:?}",
                warnings
            );
            assert!(warnings[0].contains("stagnation_guard"));
        }
    }

    // ── parse_fit_options: strict parsing at the .ferx layer. Unknown
    //    keys and malformed values both raise an error — a typo like
    //    `covariance = maybe` or `bloq_method = nope` now fails loudly
    //    instead of silently landing on an unexpected default. ───────────

    #[test]
    fn test_parse_fit_options_unknown_key_errors() {
        let err = parse_fit_options(&["n_exploraton = 200".to_string()]).unwrap_err();
        assert!(err.contains("unknown key"), "got: {err}");
        assert!(err.contains("n_exploraton"), "got: {err}");
    }

    #[test]
    fn test_parse_fit_options_malformed_numeric_errors() {
        assert!(parse_fit_options(&["n_exploration = oops".to_string()]).is_err());
    }

    #[test]
    fn test_parse_fit_options_malformed_bool_errors() {
        // Pre-refactor, `covariance = maybe` silently coerced to `false`
        // via `== "true"`, flipping the default. Now it errors.
        assert!(parse_fit_options(&["covariance = maybe".to_string()]).is_err());
    }

    #[test]
    fn test_parse_fit_options_uppercase_bool_accepted() {
        // Pre-refactor, `covariance = TRUE` silently became `false`
        // because the inline check only matched lowercase "true". The
        // strict parser accepts common casing variants.
        let opts = parse_fit_options(&["covariance = TRUE".to_string()]).unwrap();
        assert!(opts.run_covariance_step);
    }

    #[test]
    fn test_parse_fit_options_bloq_method_typo_errors() {
        // `bloq_method` was already strict in the old inline parser; the
        // new strict dispatch must preserve that (not silently default).
        assert!(parse_fit_options(&["bloq_method = nope".to_string()]).is_err());
    }

    #[test]
    fn test_parse_fit_options_gradient_method() {
        // Accepted aliases resolve to the expected GradientMethod variant.
        for (input, expected) in [
            ("gradient = auto", GradientMethod::Auto),
            ("gradient = ad", GradientMethod::Ad),
            ("gradient = autodiff", GradientMethod::Ad),
            ("gradient = fd", GradientMethod::Fd),
            ("gradient = finite", GradientMethod::Fd),
            ("gradient_method = ad", GradientMethod::Ad),
        ] {
            let opts = parse_fit_options(&[input.to_string()]).unwrap();
            assert_eq!(opts.gradient_method, expected, "input: {input}");
        }

        // Unknown values must fail loudly — silently defaulting would hide
        // typos like `gradient = auo` that a user probably intended as `auto`.
        assert!(parse_fit_options(&["gradient = nope".to_string()]).is_err());
    }

    #[test]
    fn test_parse_fit_options_reconverge_gradient_interval() {
        // Defaults to 0 (off) so the cheap fixed-EBE path stays the non-IOV
        // default.
        assert_eq!(
            parse_fit_options(&[]).unwrap().reconverge_gradient_interval,
            0
        );

        // Parses as a non-negative integer and is recorded as user-set (so it
        // isn't flagged as ignored downstream).
        let opts = parse_fit_options(&["reconverge_gradient_interval = 5".to_string()]).unwrap();
        assert_eq!(opts.reconverge_gradient_interval, 5);
        assert!(opts
            .user_set_keys
            .iter()
            .any(|k| k == "reconverge_gradient_interval"));

        // 0 is a valid explicit value (disables reconverging), not an error.
        assert_eq!(
            parse_fit_options(&["reconverge_gradient_interval = 0".to_string()])
                .unwrap()
                .reconverge_gradient_interval,
            0
        );

        // Non-integer values fail loudly.
        assert!(parse_fit_options(&["reconverge_gradient_interval = lots".to_string()]).is_err());
    }

    #[test]
    fn test_parse_all_example_ferx_files() {
        // Smoke test: every checked-in example must parse under the strict
        // [fit_options] rules. Guards against accidentally tightening a key
        // in apply_fit_option in a way that breaks a shipped example.
        //
        // When `--features nn` is off, files that declare a `[covariate_nn]`
        // or `[dynamics_nn]` block are skipped — they're only valid under
        // the feature gate. A cheap pre-scan of file contents handles this
        // without splitting the example directory.
        let examples_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
        let mut seen = 0;
        for entry in std::fs::read_dir(&examples_dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|s| s.to_str()) != Some("ferx") {
                continue;
            }
            if !cfg!(feature = "nn") {
                let src = std::fs::read_to_string(&path).unwrap_or_default();
                if src.contains("[covariate_nn") || src.contains("[dynamics_nn") {
                    continue;
                }
            }
            // TTE-only files (no [structural_model] block) require the survival feature.
            // Use a line-start check so the comment "# Note: [structural_model] ..."
            // present in example file headers does not falsely count as a block.
            if !cfg!(feature = "survival") {
                let src = std::fs::read_to_string(&path).unwrap_or_default();
                let has_event_model = src.contains("[event_model");
                let has_struct_block = src
                    .lines()
                    .any(|l| l.trim_start().starts_with("[structural_model"));
                if has_event_model && !has_struct_block {
                    continue;
                }
            }
            seen += 1;
            if let Err(e) = parse_full_model_file(&path) {
                panic!("failed to parse {}: {}", path.display(), e);
            }
        }
        assert!(
            seen > 0,
            "no .ferx files found in {}",
            examples_dir.display()
        );
    }

    #[test]
    fn test_parse_fit_options_applies_known_keys() {
        let lines = vec![
            "method = saem".to_string(),
            "n_exploration = 200".to_string(),
            "n_convergence = 400".to_string(),
            "sir = true".to_string(),
            "sir_samples = 2000".to_string(),
        ];
        let opts = parse_fit_options(&lines).unwrap();
        assert_eq!(opts.method, EstimationMethod::Saem);
        assert_eq!(opts.saem_n_exploration, 200);
        assert_eq!(opts.saem_n_convergence, 400);
        assert!(opts.sir);
        assert_eq!(opts.sir_samples, 2000);
    }

    #[test]
    fn test_parse_covariance_fallback_none_and_sir() {
        use crate::types::CovarianceFallback;
        let opts = parse_fit_options(&["covariance_fallback = sir".to_string()]).unwrap();
        assert_eq!(opts.covariance_fallback, CovarianceFallback::Sir);

        let opts2 = parse_fit_options(&["covariance_fallback = none".to_string()]).unwrap();
        assert_eq!(opts2.covariance_fallback, CovarianceFallback::None);

        // Case-insensitive
        let opts3 = parse_fit_options(&["covariance_fallback = SIR".to_string()]).unwrap();
        assert_eq!(opts3.covariance_fallback, CovarianceFallback::Sir);

        // Invalid value returns error
        let err = parse_fit_options(&["covariance_fallback = hmc".to_string()]);
        assert!(err.is_err());
    }

    #[test]
    fn test_parse_covariance_method() {
        use crate::types::CovarianceMethod;
        // default
        let def = parse_fit_options(&[]).unwrap();
        assert_eq!(def.covariance_method, CovarianceMethod::Hessian);
        // r / s / rsr (and the long-form aliases), case-insensitive
        for (input, expected) in [
            ("r", CovarianceMethod::Hessian),
            ("hessian", CovarianceMethod::Hessian),
            ("S", CovarianceMethod::CrossProduct),
            ("cross_product", CovarianceMethod::CrossProduct),
            ("RSR", CovarianceMethod::Sandwich),
            ("sandwich", CovarianceMethod::Sandwich),
        ] {
            let opts = parse_fit_options(&[format!("covariance_method = {input}")]).unwrap();
            assert_eq!(opts.covariance_method, expected, "input `{input}`");
        }
        // invalid
        assert!(parse_fit_options(&["covariance_method = bhhh".to_string()]).is_err());
    }

    #[test]
    fn test_parse_parameter_scaling_and_ofv_hessian() {
        use crate::types::ParameterScaling;
        // Defaults: parameter_scaling = Auto, covariance_ofv_hessian = true.
        let def = parse_fit_options(&[]).unwrap();
        assert_eq!(def.parameter_scaling, ParameterScaling::Auto);
        assert!(def.covariance_ofv_hessian);
        // parameter_scaling keywords, case-insensitive.
        for (input, expected) in [
            ("auto", ParameterScaling::Auto),
            ("none", ParameterScaling::None),
            ("ABS", ParameterScaling::Abs),
            ("rescale2", ParameterScaling::Rescale2),
        ] {
            let opts = parse_fit_options(&[format!("parameter_scaling = {input}")]).unwrap();
            assert_eq!(opts.parameter_scaling, expected, "input `{input}`");
        }
        assert!(parse_fit_options(&["parameter_scaling = bogus".to_string()]).is_err());
        // covariance_ofv_hessian bool.
        let off = parse_fit_options(&["covariance_ofv_hessian = false".to_string()]).unwrap();
        assert!(!off.covariance_ofv_hessian);
    }

    // ── mu-referencing pattern detection ─────────────────────────────────

    fn detect_one(line: &str, theta_names: &[&str], eta_names: &[&str]) -> Option<MuRef> {
        let tn: Vec<String> = theta_names.iter().map(|s| s.to_string()).collect();
        let en: Vec<String> = eta_names.iter().map(|s| s.to_string()).collect();
        let ctx = ParseCtx::new(&tn, &en, &[]);
        let stmts = parse_block_statements(line, ctx, StatementMode::Plain).ok()?;
        let refs = detect_mu_refs(&stmts, &tn, &en, &[]);
        // Return the one detected mu-ref (if any). Tests assume a single line.
        refs.into_iter().next().map(|(_, v)| v)
    }

    #[test]
    fn test_detect_mu_ref_multiplicative_exp() {
        // Classic NONMEM pattern: CL = TVCL * exp(ETA_CL)
        let m = detect_one("CL = TVCL * exp(ETA_CL)", &["TVCL"], &["ETA_CL"])
            .expect("should detect mu-ref");
        assert_eq!(m.theta_name, "TVCL");
        assert!(m.log_transformed);
    }

    #[test]
    fn test_detect_mu_ref_exp_of_log_sum() {
        // Canonical mu-reference form: exp(log(THETA) + ETA)
        let m = detect_one("CL = exp(log(TVCL) + ETA_CL)", &["TVCL"], &["ETA_CL"])
            .expect("should detect mu-ref");
        assert_eq!(m.theta_name, "TVCL");
        assert!(m.log_transformed);
    }

    #[test]
    fn test_detect_mu_ref_exp_of_log_sum_reversed() {
        // ETA on the left: exp(ETA + log(THETA))
        let m = detect_one("CL = exp(ETA_CL + log(TVCL))", &["TVCL"], &["ETA_CL"])
            .expect("should detect mu-ref");
        assert_eq!(m.theta_name, "TVCL");
        assert!(m.log_transformed);
    }

    #[test]
    fn test_detect_mu_ref_additive() {
        // Additive eta: CL = TVCL + ETA_CL → mu = TVCL (not log-transformed)
        let m =
            detect_one("CL = TVCL + ETA_CL", &["TVCL"], &["ETA_CL"]).expect("should detect mu-ref");
        assert_eq!(m.theta_name, "TVCL");
        assert!(!m.log_transformed);
    }

    #[test]
    fn test_detect_mu_ref_additive_reversed() {
        // ETA first: CL = ETA_CL + TVCL
        let m =
            detect_one("CL = ETA_CL + TVCL", &["TVCL"], &["ETA_CL"]).expect("should detect mu-ref");
        assert_eq!(m.theta_name, "TVCL");
        assert!(!m.log_transformed);
    }

    #[test]
    fn test_detect_mu_ref_product_chain_with_covariate() {
        // Real covariate model: CL = TVCL * (WT/70)^0.75 * exp(ETA_CL).
        // The detector walks the Mul chain for the anchor theta and the
        // exp(eta) factor; the Power sub-expression is opaque (neither a
        // Theta nor an exp(Eta)), so it is simply skipped. As long as there
        // is exactly one bare Theta factor, detection still succeeds.
        let m = detect_one(
            "CL = TVCL * (WT/70)^0.75 * exp(ETA_CL)",
            &["TVCL"],
            &["ETA_CL"],
        )
        .expect("should still detect mu-ref through opaque covariate term");
        assert_eq!(m.theta_name, "TVCL");
        assert!(m.log_transformed);
    }

    #[test]
    fn test_detect_mu_ref_rejects_two_thetas() {
        // Two thetas in the product → ambiguous anchor, pattern rejected.
        let m = detect_one(
            "CL = TVCL * TVCL2 * exp(ETA_CL)",
            &["TVCL", "TVCL2"],
            &["ETA_CL"],
        );
        assert!(m.is_none());
    }

    #[test]
    fn test_detect_mu_ref_rejects_constant_only() {
        // No theta in the product → not a mu-ref.
        let m = detect_one("CL = 2.0 * exp(ETA_CL)", &["TVCL"], &["ETA_CL"]);
        assert!(m.is_none());
    }

    #[test]
    fn test_detect_mu_ref_rejects_compound_eta_expression() {
        // exp(ETA_CL + ETA_OCC) — IIV+IOV combined pattern. Since Fix 1,
        // find_exp_eta_in_mul recognises this and returns the min-index eta
        // (ETA_CL, index 0), so a mu-ref IS detected for ETA_CL → TVCL.
        let m = detect_one(
            "CL = TVCL * exp(ETA_CL + ETA_OCC)",
            &["TVCL"],
            &["ETA_CL", "ETA_OCC"],
        );
        let m = m.expect("IIV+IOV combined pattern should detect mu-ref for ETA_CL");
        assert_eq!(m.theta_name, "TVCL");
        assert!(m.log_transformed);
    }

    #[test]
    fn test_detect_mu_ref_rejects_no_eta() {
        // KM = TVKM — no eta, no mu-ref recorded.
        let m = detect_one("KM = TVKM", &["TVKM"], &[]);
        assert!(m.is_none());
    }

    #[test]
    fn test_detect_mu_ref_multiple_parameters() {
        // Detect across several lines; each eta maps to its own theta.
        let block = "CL = TVCL * exp(ETA_CL)\nV  = TVV  * exp(ETA_V)\nKA = TVKA * exp(ETA_KA)";
        let tn = vec!["TVCL".to_string(), "TVV".to_string(), "TVKA".to_string()];
        let en = vec![
            "ETA_CL".to_string(),
            "ETA_V".to_string(),
            "ETA_KA".to_string(),
        ];
        let ctx = ParseCtx::new(&tn, &en, &[]);
        let stmts = parse_block_statements(block, ctx, StatementMode::Plain).unwrap();
        let refs = detect_mu_refs(&stmts, &tn, &en, &[]);
        assert_eq!(refs.len(), 3);
        assert_eq!(refs["ETA_CL"].theta_name, "TVCL");
        assert_eq!(refs["ETA_V"].theta_name, "TVV");
        assert_eq!(refs["ETA_KA"].theta_name, "TVKA");
        assert!(refs.values().all(|m| m.log_transformed));
    }

    #[test]
    fn test_detect_mu_ref_full_model_parse() {
        // End-to-end: parse a minimal .ferx and verify mu_refs is populated.
        let content = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)

  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04

  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let parsed = parse_full_model(content).expect("model should parse");
        assert_eq!(parsed.model.mu_refs.len(), 2);
        let cl = parsed.model.mu_refs.get("ETA_CL").unwrap();
        assert_eq!(cl.theta_name, "TVCL");
        assert!(cl.log_transformed);
    }

    #[test]
    fn test_parse_diagonal_omega() {
        let lines = vec![
            "omega ETA_CL ~ 0.07".to_string(),
            "omega ETA_V  ~ 0.02".to_string(),
        ];
        let (_, omegas, block_omegas, _, _, _) = parse_parameters(&lines).unwrap();
        assert_eq!(omegas.len(), 2);
        assert_eq!(block_omegas.len(), 0);
        assert_eq!(omegas[0].name, "ETA_CL");
        assert!((omegas[0].variance - 0.07).abs() < 1e-10);
    }

    #[test]
    fn test_parse_block_omega() {
        let lines = vec!["block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.04]".to_string()];
        let (_, omegas, block_omegas, _, _, _) = parse_parameters(&lines).unwrap();
        assert_eq!(omegas.len(), 0);
        assert_eq!(block_omegas.len(), 1);
        assert_eq!(block_omegas[0].names, vec!["ETA_CL", "ETA_V"]);
        assert_eq!(block_omegas[0].lower_triangle, vec![0.09, 0.02, 0.04]);
    }

    #[test]
    fn test_parse_block_omega_multiline() {
        // Lower triangle spread across several lines, as extract_blocks would
        // hand them to parse_parameters (one trimmed string per source line).
        let lines = vec![
            "block_omega (ETA_CL, ETA_V) = [".to_string(),
            "0.09,".to_string(),
            "0.02, 0.04".to_string(),
            "]".to_string(),
        ];
        let (_, omegas, block_omegas, _, eta_names, _) = parse_parameters(&lines).unwrap();
        assert_eq!(omegas.len(), 0);
        assert_eq!(block_omegas.len(), 1);
        assert_eq!(block_omegas[0].names, vec!["ETA_CL", "ETA_V"]);
        assert_eq!(block_omegas[0].lower_triangle, vec![0.09, 0.02, 0.04]);
        assert!(!block_omegas[0].fixed);
        assert_eq!(eta_names, vec!["ETA_CL", "ETA_V"]);
    }

    #[test]
    fn test_parse_block_omega_multiline_fix() {
        // FIX keyword on the closing-bracket line must still be honored.
        let lines = vec![
            "block_omega (ETA_CL, ETA_V) = [0.09,".to_string(),
            "0.02, 0.04] FIX".to_string(),
        ];
        let (_, _, block_omegas, _, _, _) = parse_parameters(&lines).unwrap();
        assert_eq!(block_omegas.len(), 1);
        assert!(block_omegas[0].fixed);
    }

    #[test]
    fn test_parse_block_omega_multiline_fix_own_line() {
        // FIX on its own line after the closing bracket must still be honored.
        let lines = vec![
            "block_omega (ETA_CL, ETA_V) = [".to_string(),
            "0.09,".to_string(),
            "0.02, 0.04".to_string(),
            "]".to_string(),
            "FIX".to_string(),
        ];
        let (_, _, block_omegas, _, _, _) = parse_parameters(&lines).unwrap();
        assert_eq!(block_omegas.len(), 1);
        assert!(block_omegas[0].fixed);
    }

    #[test]
    fn test_parse_block_kappa_multiline() {
        let lines = vec![
            "block_kappa (KAPPA_CL, KAPPA_V) = [".to_string(),
            "0.05, 0.01, 0.03".to_string(),
            "]".to_string(),
        ];
        let (_, _, _, _, _, kappas) = parse_parameters(&lines).unwrap();
        assert_eq!(kappas.block.len(), 1);
        assert_eq!(kappas.block[0].names, vec!["KAPPA_CL", "KAPPA_V"]);
        assert_eq!(kappas.block[0].lower_triangle, vec![0.05, 0.01, 0.03]);
    }

    #[test]
    fn test_parse_block_kappa_multiline_fix_own_line() {
        // FIX on its own line after the closing bracket must be honored for
        // IOV blocks too (shared fold logic in join_bracketed_lines).
        let lines = vec![
            "block_kappa (KAPPA_CL, KAPPA_V) = [".to_string(),
            "0.05, 0.01, 0.03".to_string(),
            "]".to_string(),
            "FIX".to_string(),
        ];
        let (_, _, _, _, _, kappas) = parse_parameters(&lines).unwrap();
        assert_eq!(kappas.block.len(), 1);
        assert!(kappas.block[0].fixed);
    }

    #[test]
    fn test_parse_block_omega_3x3() {
        let lines = vec![
            "block_omega (ETA_CL, ETA_V, ETA_KA) = [0.09, 0.01, 0.04, 0.005, 0.002, 0.16]"
                .to_string(),
        ];
        let (_, _, block_omegas, _, _, _) = parse_parameters(&lines).unwrap();
        assert_eq!(block_omegas[0].names.len(), 3);
        assert_eq!(block_omegas[0].lower_triangle.len(), 6); // 3*(3+1)/2
    }

    #[test]
    fn test_parse_block_omega_wrong_count() {
        let lines = vec![
            "block_omega (ETA_CL, ETA_V) = [0.09, 0.02]".to_string(), // needs 3, got 2
        ];
        let result = parse_parameters(&lines);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_mixed_diagonal_and_block() {
        let lines = vec![
            "omega ETA_KA ~ 0.40".to_string(),
            "block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.04]".to_string(),
        ];
        let (_, omegas, block_omegas, _, eta_names, _) = parse_parameters(&lines).unwrap();
        assert_eq!(omegas.len(), 1);
        assert_eq!(block_omegas.len(), 1);
        // Declaration order preserved: ETA_KA first, then block (ETA_CL, ETA_V)
        assert_eq!(eta_names, vec!["ETA_KA", "ETA_CL", "ETA_V"]);
    }

    #[test]
    fn test_declaration_order_block_before_diagonal() {
        let lines = vec![
            "block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.04]".to_string(),
            "omega ETA_KA ~ 0.40".to_string(),
        ];
        let (_, _, _, _, eta_names, _) = parse_parameters(&lines).unwrap();
        // block_omega declared first, so ETA_CL, ETA_V come before ETA_KA
        assert_eq!(eta_names, vec!["ETA_CL", "ETA_V", "ETA_KA"]);
    }

    #[test]
    fn test_build_omega_matrix_diagonal_only() {
        let diag = vec![
            OmegaSpec {
                name: "ETA_CL".into(),
                variance: 0.09,
                fixed: false,
                init_as_sd: false,
            },
            OmegaSpec {
                name: "ETA_V".into(),
                variance: 0.04,
                fixed: false,
                init_as_sd: false,
            },
        ];
        let names = vec!["ETA_CL".into(), "ETA_V".into()];
        let omega = build_omega_matrix(&diag, &[], &names).unwrap();
        assert!(omega.diagonal);
        assert!((omega.matrix[(0, 0)] - 0.09).abs() < 1e-10);
        assert!((omega.matrix[(1, 1)] - 0.04).abs() < 1e-10);
        assert!((omega.matrix[(0, 1)]).abs() < 1e-10);
    }

    #[test]
    fn test_build_omega_matrix_block() {
        let block = vec![BlockOmegaSpec {
            names: vec!["ETA_CL".into(), "ETA_V".into()],
            lower_triangle: vec![0.09, 0.02, 0.04],
            fixed: false,
        }];
        let names = vec!["ETA_CL".into(), "ETA_V".into()];
        let omega = build_omega_matrix(&[], &block, &names).unwrap();
        assert!(!omega.diagonal);
        assert!((omega.matrix[(0, 0)] - 0.09).abs() < 1e-10);
        assert!((omega.matrix[(1, 1)] - 0.04).abs() < 1e-10);
        assert!((omega.matrix[(0, 1)] - 0.02).abs() < 1e-10);
        assert!((omega.matrix[(1, 0)] - 0.02).abs() < 1e-10);
    }

    #[test]
    fn test_build_omega_matrix_mixed() {
        let diag = vec![OmegaSpec {
            name: "ETA_KA".into(),
            variance: 0.16,
            fixed: false,
            init_as_sd: false,
        }];
        let block = vec![BlockOmegaSpec {
            names: vec!["ETA_CL".into(), "ETA_V".into()],
            lower_triangle: vec![0.09, 0.02, 0.04],
            fixed: false,
        }];
        let names = vec!["ETA_KA".into(), "ETA_CL".into(), "ETA_V".into()];
        let omega = build_omega_matrix(&diag, &block, &names).unwrap();
        assert!(!omega.diagonal);
        assert!((omega.matrix[(0, 0)] - 0.16).abs() < 1e-10); // ETA_KA
        assert!((omega.matrix[(1, 1)] - 0.09).abs() < 1e-10); // ETA_CL
        assert!((omega.matrix[(2, 2)] - 0.04).abs() < 1e-10); // ETA_V
        assert!((omega.matrix[(1, 2)] - 0.02).abs() < 1e-10); // cov(CL, V)
        assert!((omega.matrix[(0, 1)]).abs() < 1e-10); // no cov(KA, CL)
    }

    // ── FIX keyword ────────────────────────────────────────────────────────

    #[test]
    fn test_parse_theta_fix_without_bounds() {
        let lines = vec!["theta TVCL(0.1, FIX)".to_string()];
        let (thetas, _, _, _, _, _) = parse_parameters(&lines).unwrap();
        assert_eq!(thetas.len(), 1);
        assert!(thetas[0].fixed);
        assert!((thetas[0].init - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_parse_theta_fix_with_bounds() {
        let lines = vec!["theta TVCL(0.1, 0.01, 1.0, FIX)".to_string()];
        let (thetas, _, _, _, _, _) = parse_parameters(&lines).unwrap();
        assert!(thetas[0].fixed);
        assert!((thetas[0].lower - 0.01).abs() < 1e-12);
        assert!((thetas[0].upper - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_parse_theta_fix_no_comma_inside_parens() {
        // theta NAME(init FIX) — no comma before FIX
        let lines = vec!["theta TVCL(0.75 FIX)".to_string()];
        let (thetas, _, _, _, _, _) = parse_parameters(&lines).unwrap();
        assert_eq!(thetas.len(), 1);
        assert!(thetas[0].fixed);
        assert!((thetas[0].init - 0.75).abs() < 1e-12);
    }

    #[test]
    fn test_parse_theta_fix_after_paren() {
        // theta NAME(init) FIX — FIX outside closing paren
        let lines = vec!["theta TVCL(0.75) FIX".to_string()];
        let (thetas, _, _, _, _, _) = parse_parameters(&lines).unwrap();
        assert_eq!(thetas.len(), 1);
        assert!(thetas[0].fixed);
        assert!((thetas[0].init - 0.75).abs() < 1e-12);
    }

    #[test]
    fn test_parse_theta_fix_after_paren_with_bounds() {
        // theta NAME(init, lower, upper) FIX — bounds + FIX outside paren
        let lines = vec!["theta TVKA(1.0, 0.01, 10.0) FIX".to_string()];
        let (thetas, _, _, _, _, _) = parse_parameters(&lines).unwrap();
        assert_eq!(thetas.len(), 1);
        assert!(thetas[0].fixed);
        assert!((thetas[0].init - 1.0).abs() < 1e-12);
        assert!((thetas[0].lower - 0.01).abs() < 1e-12);
        assert!((thetas[0].upper - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_parse_theta_lower_bound_only() {
        // theta NAME(init, lower) — upper defaults to 1e9
        let lines = vec!["theta TVCL(1.0, 0.01)".to_string()];
        let (thetas, _, _, _, _, _) = parse_parameters(&lines).unwrap();
        assert_eq!(thetas.len(), 1);
        assert!(!thetas[0].fixed);
        assert!((thetas[0].init - 1.0).abs() < 1e-12);
        assert!((thetas[0].lower - 0.01).abs() < 1e-12);
        assert!((thetas[0].upper - 1e9).abs() < 1.0);
    }

    #[test]
    fn test_parse_theta_lower_bound_fix_inside() {
        // theta NAME(init, lower, FIX) — lower only + FIX inside parens
        let lines = vec!["theta TVCL(1.0, 0.01, FIX)".to_string()];
        let (thetas, _, _, _, _, _) = parse_parameters(&lines).unwrap();
        assert_eq!(thetas.len(), 1);
        assert!(thetas[0].fixed);
        assert!((thetas[0].lower - 0.01).abs() < 1e-12);
        assert!((thetas[0].upper - 1e9).abs() < 1.0);
    }

    #[test]
    fn test_parse_theta_lower_bound_fix_outside() {
        // theta NAME(init, lower) FIX — lower only + FIX after paren
        let lines = vec!["theta TVCL(1.0, 0.01) FIX".to_string()];
        let (thetas, _, _, _, _, _) = parse_parameters(&lines).unwrap();
        assert_eq!(thetas.len(), 1);
        assert!(thetas[0].fixed);
        assert!((thetas[0].lower - 0.01).abs() < 1e-12);
        assert!((thetas[0].upper - 1e9).abs() < 1.0);
    }

    #[test]
    fn test_parse_theta_unfixed_by_default() {
        let lines = vec!["theta TVCL(0.1, 0.01, 1.0)".to_string()];
        let (thetas, _, _, _, _, _) = parse_parameters(&lines).unwrap();
        assert!(!thetas[0].fixed);
    }

    #[test]
    fn test_parse_theta_allows_space_before_paren() {
        // Regression: `theta TVCL (5, ...)` (with whitespace before the paren)
        // used to silently fail to match, causing TVCL to be misclassified as
        // a covariate downstream.
        let lines = vec![
            "theta TVCL (5, 0.001, 100.0)".to_string(),
            "theta TVV  ( 10 )".to_string(),
            "theta TVKA\t(0.5, FIX)".to_string(),
        ];
        let (thetas, _, _, _, _, _) = parse_parameters(&lines).unwrap();
        assert_eq!(thetas.len(), 3);
        assert_eq!(thetas[0].name, "TVCL");
        assert!((thetas[0].init - 5.0).abs() < 1e-12);
        assert!((thetas[0].lower - 0.001).abs() < 1e-12);
        assert!((thetas[0].upper - 100.0).abs() < 1e-12);
        assert!(!thetas[0].fixed);
        assert_eq!(thetas[1].name, "TVV");
        assert!((thetas[1].init - 10.0).abs() < 1e-12);
        assert_eq!(thetas[2].name, "TVKA");
        assert!(thetas[2].fixed);
    }

    #[test]
    fn test_parse_omega_fix() {
        let lines = vec!["omega ETA_CL ~ 0.09 FIX".to_string()];
        let (_, omegas, _, _, _, _) = parse_parameters(&lines).unwrap();
        assert!(omegas[0].fixed);
    }

    #[test]
    fn test_omega_unfixed_no_annotation() {
        // Baseline: plain omega with no FIX and no annotation — confirms the
        // group-numbering shift (annotation moved 3→4) didn't regress the
        // common case.
        let lines = vec!["omega ETA_CL ~ 0.09".to_string()];
        let (_, omegas, _, _, _, _) = parse_parameters(&lines).unwrap();
        assert!(!omegas[0].fixed);
        assert!(!omegas[0].init_as_sd);
        assert!((omegas[0].variance - 0.09).abs() < 1e-12);
    }

    #[test]
    fn test_omega_double_fix_is_harmless() {
        // `FIX (sd) FIX` — both FIX groups fire; result must still be fixed
        // with SD squaring applied.
        let lines = vec!["omega ETA_CL ~ 0.30 FIX (sd) FIX".to_string()];
        let (_, omegas, _, _, _, _) = parse_parameters(&lines).unwrap();
        let expected = 0.30 * 0.30;
        assert!((omegas[0].variance - expected).abs() < 1e-12);
        assert!(omegas[0].fixed);
        assert!(omegas[0].init_as_sd);
    }

    #[test]
    fn test_parse_sigma_fix() {
        let lines = vec!["sigma PROP ~ 0.05 FIX".to_string()];
        let (_, _, _, sigmas, _, _) = parse_parameters(&lines).unwrap();
        assert!(sigmas[0].fixed);
    }

    #[test]
    fn test_parse_block_omega_fix() {
        let lines = vec!["block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.04] FIX".to_string()];
        let (_, _, blocks, _, _, _) = parse_parameters(&lines).unwrap();
        assert!(blocks[0].fixed);
    }

    #[test]
    fn test_fix_keyword_case_insensitive() {
        let lines = vec![
            "theta TVCL(0.1, fix)".to_string(),
            "omega ETA ~ 0.05 Fix".to_string(),
            "sigma S ~ 0.02 FIX".to_string(),
        ];
        let (thetas, omegas, _, sigmas, _, _) = parse_parameters(&lines).unwrap();
        assert!(thetas[0].fixed);
        assert!(omegas[0].fixed);
        assert!(sigmas[0].fixed);
    }

    // ── SD-init annotation (issue #5 + #56) ─────────────────────────────

    #[test]
    fn test_omega_default_is_variance() {
        // No annotation: value is stored verbatim as variance.
        let lines = vec!["omega ETA_CL ~ 0.07".to_string()];
        let (_, omegas, _, _, _, _) = parse_parameters(&lines).unwrap();
        assert!((omegas[0].variance - 0.07).abs() < 1e-12);
        assert!(!omegas[0].init_as_sd);
    }

    #[test]
    fn test_omega_sd_annotation_squares_value() {
        // `(sd)` → variance is the square of the raw value.
        let lines = vec!["omega ETA_CL ~ 0.265 (sd)".to_string()];
        let (_, omegas, _, _, _, _) = parse_parameters(&lines).unwrap();
        let expected = 0.265 * 0.265;
        assert!((omegas[0].variance - expected).abs() < 1e-12);
        assert!(omegas[0].init_as_sd);
    }

    #[test]
    fn test_omega_variance_annotation_is_noop() {
        // `(variance)` and `(var)` are explicit no-ops.
        let lines = vec![
            "omega ETA_CL ~ 0.07 (variance)".to_string(),
            "omega ETA_V  ~ 0.04 (var)".to_string(),
        ];
        let (_, omegas, _, _, _, _) = parse_parameters(&lines).unwrap();
        assert!((omegas[0].variance - 0.07).abs() < 1e-12);
        assert!(!omegas[0].init_as_sd);
        assert!((omegas[1].variance - 0.04).abs() < 1e-12);
        assert!(!omegas[1].init_as_sd);
    }

    #[test]
    fn test_omega_sd_annotation_with_fix() {
        // `(sd) FIX` — both annotations must be honored together.
        let lines = vec!["omega ETA_CL ~ 0.30 (sd) FIX".to_string()];
        let (_, omegas, _, _, _, _) = parse_parameters(&lines).unwrap();
        let expected = 0.30 * 0.30;
        assert!((omegas[0].variance - expected).abs() < 1e-12);
        assert!(omegas[0].fixed);
        assert!(omegas[0].init_as_sd);
    }

    #[test]
    fn test_omega_fix_before_sd_annotation() {
        // `FIX (sd)` — FIX before the scale annotation.
        let lines = vec!["omega ETA_CL ~ 0.30 FIX (sd)".to_string()];
        let (_, omegas, _, _, _, _) = parse_parameters(&lines).unwrap();
        let expected = 0.30 * 0.30;
        assert!((omegas[0].variance - expected).abs() < 1e-12);
        assert!(omegas[0].fixed);
        assert!(omegas[0].init_as_sd);
    }

    #[test]
    fn test_omega_fix_before_annotation_no_sd() {
        // `FIX` before a no-op annotation — fixed and variance-scale.
        let lines = vec!["omega ETA_CL ~ 0.09 FIX (variance)".to_string()];
        let (_, omegas, _, _, _, _) = parse_parameters(&lines).unwrap();
        assert!((omegas[0].variance - 0.09).abs() < 1e-12);
        assert!(omegas[0].fixed);
        assert!(!omegas[0].init_as_sd);
    }

    #[test]
    fn test_sigma_fix_before_sd_annotation() {
        // `FIX (sd)` — FIX before the scale annotation for sigma.
        let lines = vec!["sigma PROP ~ 0.30 FIX (sd)".to_string()];
        let (_, _, _, sigmas, _, _) = parse_parameters(&lines).unwrap();
        assert!(sigmas[0].fixed);
        assert!(sigmas[0].init_as_sd);
        assert!((sigmas[0].value - 0.30).abs() < 1e-12);
    }

    #[test]
    fn test_sigma_fix_after_sd_annotation() {
        // `(sd) FIX` — existing form still works.
        let lines = vec!["sigma PROP ~ 0.30 (sd) FIX".to_string()];
        let (_, _, _, sigmas, _, _) = parse_parameters(&lines).unwrap();
        assert!(sigmas[0].fixed);
        assert!(sigmas[0].init_as_sd);
    }

    #[test]
    fn test_sigma_unfixed_no_annotation() {
        // Baseline: plain sigma with no FIX and no annotation — confirms the
        // group-numbering shift didn't regress the common case.
        let lines = vec!["sigma PROP ~ 0.04".to_string()];
        let (_, _, _, sigmas, _, _) = parse_parameters(&lines).unwrap();
        assert!(!sigmas[0].fixed);
        assert!(!sigmas[0].init_as_sd);
        // Stored as SD internally: sqrt(0.04) = 0.2
        assert!((sigmas[0].value - 0.2).abs() < 1e-12);
    }

    #[test]
    fn test_sigma_default_is_variance() {
        // Since #56, the default sigma input is variance — the parser sqrt's
        // it into the internal SD representation that the likelihood uses.
        let lines = vec!["sigma PROP ~ 0.04".to_string()];
        let (_, _, _, sigmas, _, _) = parse_parameters(&lines).unwrap();
        // Stored value is SD = sqrt(variance) = sqrt(0.04) = 0.2.
        assert!((sigmas[0].value - 0.2).abs() < 1e-12);
        assert!(!sigmas[0].init_as_sd);
    }

    #[test]
    fn test_sigma_sd_annotation_stores_value_as_is() {
        // `(sd)` → the value is already on the SD scale, no transform.
        let lines = vec!["sigma PROP ~ 0.2 (sd)".to_string()];
        let (_, _, _, sigmas, _, _) = parse_parameters(&lines).unwrap();
        assert!((sigmas[0].value - 0.2).abs() < 1e-12);
        assert!(sigmas[0].init_as_sd);
    }

    #[test]
    fn test_sigma_default_and_sd_equivalent_initial_value() {
        // `sigma X ~ v²` (default variance) must produce the same internal
        // SD as `sigma X ~ v (sd)` (SD).
        let lines = vec![
            "sigma A ~ 0.0004".to_string(),    // variance 0.0004
            "sigma B ~ 0.02 (sd)".to_string(), // SD 0.02
        ];
        let (_, _, _, sigmas, _, _) = parse_parameters(&lines).unwrap();
        assert!((sigmas[0].value - sigmas[1].value).abs() < 1e-12);
    }

    #[test]
    fn test_sigma_negative_variance_rejected() {
        // A negative value on the (default) variance scale is meaningless —
        // sqrt would yield NaN and silently corrupt the fit. Reject up-front
        // with a clear error.
        let lines = vec!["sigma PROP ~ -0.1".to_string()];
        let res = parse_parameters(&lines);
        match res {
            Err(msg) => assert!(msg.contains("negative initial variance"), "got: {msg}"),
            Ok(_) => panic!("expected error for negative sigma variance"),
        }
    }

    #[test]
    fn test_sigma_negative_sd_rejected() {
        // Negative SD is just as nonsensical as negative variance, and the
        // optimizer's `s.max(1e-10).ln()` packing would silently clamp the
        // bad input rather than surface it. Reject at parse time, symmetric
        // with the negative-variance case.
        let lines = vec!["sigma PROP ~ -0.5 (sd)".to_string()];
        let res = parse_parameters(&lines);
        match res {
            Err(msg) => assert!(msg.contains("negative initial SD"), "got: {msg}"),
            Ok(_) => panic!("expected error for negative sigma SD"),
        }
    }

    #[test]
    fn test_omega_negative_value_rejected() {
        // Same rule applies to omega — variance must be ≥ 0, and SD ≥ 0.
        for line in [
            "omega ETA_CL ~ -0.04",
            "omega ETA_CL ~ -0.2 (sd)",
            "kappa KAPPA_CL ~ -0.03",
            "kappa KAPPA_CL ~ -0.1 (sd)",
        ] {
            let res = parse_parameters(&[line.to_string()]);
            assert!(res.is_err(), "expected negative `{line}` to be rejected");
        }
    }

    #[test]
    fn test_kappa_sd_annotation_squares_value() {
        let lines = vec!["kappa KAPPA_CL ~ 0.25 (sd)".to_string()];
        let (_, _, _, _, _, kappas) = parse_parameters(&lines).unwrap();
        let k = &kappas.diagonal[0];
        let expected = 0.25 * 0.25;
        assert!((k.variance - expected).abs() < 1e-12);
        assert!(k.init_as_sd);
    }

    #[test]
    fn test_sd_annotation_case_insensitive() {
        // `(SD)`, `(Sd)`, `(sd)` must all be accepted.
        let lines = vec![
            "omega ETA_A ~ 0.1 (SD)".to_string(),
            "omega ETA_B ~ 0.2 (Sd)".to_string(),
            "omega ETA_C ~ 0.3 (sd)".to_string(),
        ];
        let (_, omegas, _, _, _, _) = parse_parameters(&lines).unwrap();
        assert!(omegas.iter().all(|o| o.init_as_sd));
    }

    #[test]
    fn test_unknown_scale_tag_is_ignored_as_trailing_garbage() {
        // The omega regex is intentionally unanchored — it matches the
        // leading `omega NAME ~ value` and lets trailing tokens fall through.
        // An unrecognized tag like `(foo)` therefore doesn't fail the parse;
        // the value is taken as variance and `init_as_sd` stays `false`, just
        // as if the tag weren't there. (This matches how the `FIX` keyword's
        // prefix-match check works — only the exact, recognized tag changes
        // behavior; anything else is silently ignored, consistent with the
        // parser's existing FIXED-vs-FIX handling.)
        let lines = vec!["omega ETA_CL ~ 0.07 (foo)".to_string()];
        let (_, omegas, _, _, _, _) = parse_parameters(&lines).unwrap();
        assert_eq!(omegas.len(), 1);
        assert!((omegas[0].variance - 0.07).abs() < 1e-12);
        assert!(!omegas[0].init_as_sd);
    }

    #[test]
    fn test_parse_full_model_threads_init_as_sd_to_compiled_model() {
        // End-to-end: a `(sd)` annotation in the .ferx text must surface as
        // `true` in the matching CompiledModel.{omega,sigma}_init_as_sd slot.
        let content = r#"
[parameters]
  theta TVCL(0.2)
  theta TVV(10.0)
  theta TVKA(1.5)
  omega ETA_CL ~ 0.30 (sd)
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.50 (sd)
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let parsed = parse_full_model(content).unwrap();
        assert_eq!(
            parsed.model.omega_init_as_sd,
            vec![true, false, true],
            "omega_init_as_sd flags must thread through to CompiledModel"
        );
        assert_eq!(parsed.model.sigma_init_as_sd, vec![true]);
        // Verify the SD-coded omega was squared: 0.30² = 0.09.
        let omega = &parsed.model.default_params.omega.matrix;
        assert!((omega[(0, 0)] - 0.09).abs() < 1e-12);
        // And the variance-coded omega is stored verbatim.
        assert!((omega[(1, 1)] - 0.04).abs() < 1e-12);
        // Sigma stored as SD (input was already SD, no transform).
        let sigma = &parsed.model.default_params.sigma.values;
        assert!((sigma[0] - 0.02).abs() < 1e-12);
    }

    #[test]
    fn test_fix_keyword_rejects_prefix_match() {
        // `FIXED` must not be silently accepted as `FIX`. Any non-exact token
        // should leave the parameter as free (or fail to parse the line),
        // never flip `fixed = true`.
        let lines = vec![
            "omega ETA_CL ~ 0.09 FIXED".to_string(),
            "sigma PROP ~ 0.02 FIXED".to_string(),
            "block_omega (A, B) = [1.0, 0.0, 1.0] FIXED".to_string(),
        ];
        let (_, omegas, blocks, sigmas, _, _) = parse_parameters(&lines).unwrap();
        // omega/sigma still parse (trailing `FIXED` is ignored) but must NOT
        // be marked fixed.
        assert!(!omegas[0].fixed);
        assert!(!sigmas[0].fixed);
        assert!(!blocks[0].fixed);
    }

    #[test]
    fn test_build_omega_fixed_diagonal() {
        let diag = vec![
            OmegaSpec {
                name: "ETA_CL".into(),
                variance: 0.09,
                fixed: true,
                init_as_sd: false,
            },
            OmegaSpec {
                name: "ETA_V".into(),
                variance: 0.04,
                fixed: false,
                init_as_sd: false,
            },
        ];
        let names = vec!["ETA_CL".into(), "ETA_V".into()];
        let flags = build_omega_fixed(&diag, &[], &names).unwrap();
        assert_eq!(flags, vec![true, false]);
    }

    #[test]
    fn test_build_omega_fixed_block() {
        let block = vec![BlockOmegaSpec {
            names: vec!["ETA_CL".into(), "ETA_V".into()],
            lower_triangle: vec![0.09, 0.02, 0.04],
            fixed: true,
        }];
        let names = vec!["ETA_CL".into(), "ETA_V".into()];
        let flags = build_omega_fixed(&[], &block, &names).unwrap();
        assert_eq!(flags, vec![true, true]);
    }

    #[test]
    fn test_build_omega_fixed_rejects_diag_fix_inside_free_block() {
        // ETA_CL is in a non-FIX block but also declared FIX as a diagonal —
        // the parser must reject this as ambiguous.
        let diag = vec![OmegaSpec {
            name: "ETA_CL".into(),
            variance: 0.09,
            fixed: true,
            init_as_sd: false,
        }];
        let block = vec![BlockOmegaSpec {
            names: vec!["ETA_CL".into(), "ETA_V".into()],
            lower_triangle: vec![0.09, 0.02, 0.04],
            fixed: false,
        }];
        let names = vec!["ETA_CL".into(), "ETA_V".into()];
        let res = build_omega_fixed(&diag, &block, &names);
        assert!(res.is_err());
    }

    #[test]
    fn test_parse_full_model_with_fix() {
        let content = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, FIX)
  theta TVKA(1.5, 0.01, 50.0)

  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04 FIX
  omega ETA_KA ~ 0.30

  sigma PROP_ERR ~ 0.02 FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let parsed = parse_full_model(content).unwrap();
        let p = &parsed.model.default_params;
        assert_eq!(p.theta_fixed, vec![false, true, false]);
        assert_eq!(p.omega_fixed, vec![false, true, false]);
        assert_eq!(p.sigma_fixed, vec![true]);
        assert!(p.has_any_fixed());
    }

    /// Build a minimal one-cpt model, optionally with a `[covariates]` block and
    /// an optional covariate reference in the CL expression.
    fn model_with_covariates(cov_block: Option<&str>, cl_uses_wt: bool) -> String {
        let cl_expr = if cl_uses_wt {
            "CL = TVCL * (WT / 70.0) * exp(ETA_CL)"
        } else {
            "CL = TVCL * exp(ETA_CL)"
        };
        let cov = cov_block
            .map(|b| format!("\n[covariates]\n{}\n", b))
            .unwrap_or_default();
        format!(
            r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02
{cov}
[individual_parameters]
  {cl_expr}
  V  = TVV
  KA = TVKA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
"#
        )
    }

    #[test]
    fn test_covariates_absent_is_none() {
        let parsed = parse_full_model(&model_with_covariates(None, false)).unwrap();
        assert!(parsed.covariate_decls.is_none());
    }

    #[test]
    fn test_covariates_per_line_form() {
        let parsed = parse_full_model(&model_with_covariates(
            Some("WT continuous\nSEX categorical"),
            true,
        ))
        .unwrap();
        let decls = parsed.covariate_decls.expect("declared");
        assert_eq!(decls.len(), 2);
        assert_eq!(decls[0].name, "WT");
        assert_eq!(decls[0].kind, CovariateKind::Continuous);
        assert_eq!(decls[1].name, "SEX");
        assert_eq!(decls[1].kind, CovariateKind::Categorical);
    }

    #[test]
    fn test_covariates_colon_form_and_order() {
        let parsed = parse_full_model(&model_with_covariates(
            Some("continuous: WT, HT, CRCL\ncategorical: SEX"),
            true,
        ))
        .unwrap();
        let decls = parsed.covariate_decls.expect("declared");
        let names: Vec<&str> = decls.iter().map(|d| d.name.as_str()).collect();
        // Declaration order preserved across the two lines.
        assert_eq!(names, vec!["WT", "HT", "CRCL", "SEX"]);
        assert_eq!(decls[3].kind, CovariateKind::Categorical);
    }

    #[test]
    fn test_covariates_type_aliases() {
        let parsed = parse_full_model(&model_with_covariates(
            Some("WT cont\nSEX cat\ncont: HT"),
            true,
        ))
        .unwrap();
        let decls = parsed.covariate_decls.expect("declared");
        assert_eq!(decls[0].kind, CovariateKind::Continuous);
        assert_eq!(decls[1].kind, CovariateKind::Categorical);
        assert_eq!(decls[2].kind, CovariateKind::Continuous);
    }

    #[test]
    fn test_covariates_duplicate_errors() {
        let err = parse_full_model(&model_with_covariates(
            Some("WT continuous\nWT categorical"),
            true,
        ))
        .err()
        .unwrap();
        assert!(err.contains("more than once"), "got: {err}");
    }

    #[test]
    fn test_covariates_unknown_type_errors() {
        let err = parse_full_model(&model_with_covariates(Some("WT numeric"), true))
            .err()
            .unwrap();
        assert!(err.contains("unknown covariate type"), "got: {err}");
    }

    #[test]
    fn test_covariates_referenced_but_undeclared_warns_not_errors() {
        // CL uses WT, but only SEX is declared. This is allowed (WT is still
        // usable) — the parser warns rather than erroring.
        let parsed =
            parse_full_model(&model_with_covariates(Some("SEX categorical"), true)).unwrap();
        // WT is still declared as known? No — only SEX is. But the parse succeeds.
        let decls = parsed.covariate_decls.expect("declared");
        assert_eq!(decls.len(), 1);
        assert_eq!(decls[0].name, "SEX");
        // A warning names the undeclared covariate.
        assert!(
            parsed
                .model
                .parse_warnings
                .iter()
                .any(|w| w.contains("WT") && w.contains("not declared")),
            "expected a warning about undeclared WT, got: {:?}",
            parsed.model.parse_warnings
        );
    }

    #[test]
    fn test_covariates_declared_unreferenced_is_ok() {
        // Declaring a covariate the model doesn't use is the whole point
        // ("potentially available") — must not error.
        let parsed =
            parse_full_model(&model_with_covariates(Some("WT continuous"), false)).unwrap();
        assert_eq!(parsed.covariate_decls.expect("declared").len(), 1);
    }

    #[test]
    fn test_parse_full_model_with_block_omega() {
        let content = r#"
# Test model with block omega

[parameters]
  theta TVCL(0.134, 0.001, 10.0)
  theta TVV(8.1, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)

  block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.04]
  omega ETA_KA ~ 0.40

  sigma PROP_ERR ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let parsed = parse_full_model(content).unwrap();
        let omega = &parsed.model.default_params.omega;
        assert_eq!(omega.dim(), 3);
        assert!(!omega.diagonal);
        // Eta names preserve declaration order from the model file
        assert_eq!(omega.eta_names, vec!["ETA_CL", "ETA_V", "ETA_KA"]);
        // ETA_CL = index 0, ETA_V = index 1, ETA_KA = index 2
        assert!((omega.matrix[(0, 0)] - 0.09).abs() < 1e-10); // ETA_CL
        assert!((omega.matrix[(1, 1)] - 0.04).abs() < 1e-10); // ETA_V
        assert!((omega.matrix[(2, 2)] - 0.40).abs() < 1e-10); // ETA_KA
        assert!((omega.matrix[(0, 1)] - 0.02).abs() < 1e-10); // cov(CL, V)
    }

    // ── fit_options parsing: new optimizer choices ──────────────────────────

    fn minimal_model_with_fit_options(fit_opts: &str) -> String {
        format!(
            r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
{}
"#,
            fit_opts
        )
    }

    /// Build a 1-cpt oral model with a custom `[error_model]` block (and one
    /// extra sigma so combined/per-CMT cases have enough sigmas to reference).
    fn model_with_error_block(error_block: &str) -> Result<crate::types::CompiledModel, String> {
        let content = format!(
            r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  sigma SIG1 ~ 0.02
  sigma SIG2 ~ 0.1

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
{error_block}
"#
        );
        parse_model_string(&content)
    }

    #[test]
    fn test_ltbs_log_dv_additive_case2() {
        // `log(DV) ~ additive(...)`: engine logs DV (natural-scale data).
        let m = model_with_error_block("  log(DV) ~ additive(SIG1)").unwrap();
        assert!(
            m.log_transform,
            "log(DV) ~ additive should set log_transform"
        );
        assert!(
            !m.dv_pre_logged,
            "log(DV) ~ additive: engine logs DV, so dv_pre_logged must be false"
        );
        assert_eq!(m.error_model, ErrorModel::Additive);
    }

    #[test]
    fn test_ltbs_log_additive_case1() {
        // `DV ~ log_additive(...)`: DV already log; only the prediction is logged.
        let m = model_with_error_block("  DV ~ log_additive(SIG1)").unwrap();
        assert!(m.log_transform, "log_additive should set log_transform");
        assert!(
            m.dv_pre_logged,
            "log_additive: DV is already log, so dv_pre_logged must be true"
        );
        assert_eq!(m.error_model, ErrorModel::Additive);
    }

    #[test]
    fn test_non_ltbs_additive_unaffected() {
        let m = model_with_error_block("  DV ~ additive(SIG1)").unwrap();
        assert!(!m.log_transform);
        assert!(!m.dv_pre_logged);
    }

    #[test]
    fn test_ltbs_rejects_proportional() {
        let err = model_with_error_block("  log(DV) ~ proportional(SIG1)").unwrap_err();
        assert!(
            err.contains("additive"),
            "expected additive-only message, got: {err}"
        );
    }

    #[test]
    fn test_ltbs_rejects_double_log() {
        let err = model_with_error_block("  log(DV) ~ log_additive(SIG1)").unwrap_err();
        assert!(
            err.contains("double-log"),
            "expected double-log rejection, got: {err}"
        );
    }

    #[test]
    fn test_ltbs_rejects_non_dv_lhs() {
        // `log(<not DV>) ~ additive(...)` would parse silently but the engine
        // always log-transforms the `DV` column, so the LHS is misleading.
        let err = model_with_error_block("  log(CONC) ~ additive(SIG1)").unwrap_err();
        assert!(
            err.contains("DV"),
            "expected DV-required rejection, got: {err}"
        );
    }

    #[test]
    fn test_ltbs_rejects_per_cmt() {
        // LTBS is single-endpoint only.
        let err = model_with_error_block("  CMT=1: log(DV) ~ additive(SIG1)").unwrap_err();
        assert!(
            err.contains("per-CMT") || err.contains("multi-endpoint"),
            "expected per-CMT rejection, got: {err}"
        );
    }

    #[test]
    fn test_parse_optimizer_bobyqa() {
        let content = minimal_model_with_fit_options("  optimizer = bobyqa");
        let parsed = parse_full_model(&content).unwrap();
        assert_eq!(parsed.fit_options.optimizer, Optimizer::Bobyqa);
    }

    #[test]
    fn test_parse_optimizer_trust_region() {
        let content = minimal_model_with_fit_options("  optimizer = trust_region");
        let parsed = parse_full_model(&content).unwrap();
        assert_eq!(parsed.fit_options.optimizer, Optimizer::TrustRegion);
    }

    #[test]
    fn test_parse_optimizer_newton_tr_alias() {
        // newton_tr is an accepted alias for trust_region
        let content = minimal_model_with_fit_options("  optimizer = newton_tr");
        let parsed = parse_full_model(&content).unwrap();
        assert_eq!(parsed.fit_options.optimizer, Optimizer::TrustRegion);
    }

    #[test]
    fn test_parse_optimizer_case_insensitive() {
        // Parser lowercases the value, so mixed-case should map the same way.
        let content = minimal_model_with_fit_options("  optimizer = BOBYQA");
        let parsed = parse_full_model(&content).unwrap();
        assert_eq!(parsed.fit_options.optimizer, Optimizer::Bobyqa);

        let content2 = minimal_model_with_fit_options("  optimizer = Trust_Region");
        let parsed2 = parse_full_model(&content2).unwrap();
        assert_eq!(parsed2.fit_options.optimizer, Optimizer::TrustRegion);
    }

    #[test]
    fn test_parse_optimizer_defaults_to_bobyqa() {
        // `[fit_options]` block present but `optimizer` omitted → default is
        // BOBYQA (derivative-free trust-region). See `FitOptions::default` for
        // the rationale.
        let content = minimal_model_with_fit_options("  maxiter = 100");
        let parsed = parse_full_model(&content).unwrap();
        assert_eq!(parsed.fit_options.optimizer, Optimizer::Bobyqa);
    }

    #[test]
    fn test_parse_steihaug_max_iters() {
        let content =
            minimal_model_with_fit_options("  optimizer = trust_region\n  steihaug_max_iters = 30");
        let parsed = parse_full_model(&content).unwrap();
        assert_eq!(parsed.fit_options.optimizer, Optimizer::TrustRegion);
        assert_eq!(parsed.fit_options.steihaug_max_iters, Some(30));
    }

    #[test]
    fn test_steihaug_max_iters_default() {
        // Default is None (size-adaptive budget).
        let content = minimal_model_with_fit_options("  optimizer = trust_region");
        let parsed = parse_full_model(&content).unwrap();
        assert_eq!(parsed.fit_options.steihaug_max_iters, None);
    }

    #[test]
    fn test_parse_inner_maxiter_and_tol() {
        let content = minimal_model_with_fit_options("  inner_maxiter = 75\n  inner_tol = 1e-5");
        let parsed = parse_full_model(&content).unwrap();
        assert_eq!(parsed.fit_options.inner_maxiter, 75);
        assert!((parsed.fit_options.inner_tol - 1e-5).abs() < 1e-15);
    }

    #[test]
    fn test_fit_options_defaults() {
        // Guard against accidental drift in defaults — documented as:
        //   optimizer = bobyqa, inner_maxiter = 200, inner_tol = 1e-5,
        //   steihaug_max_iters = None (adaptive).
        let opts = FitOptions::default();
        assert_eq!(opts.optimizer, Optimizer::Bobyqa);
        assert_eq!(opts.inner_maxiter, 200);
        assert!((opts.inner_tol - 1e-5).abs() < 1e-20);
        assert_eq!(opts.steihaug_max_iters, None);
    }

    #[test]
    fn test_parse_example_warfarin_bobyqa_file() {
        // The example file is part of the user-visible surface; parsing it is
        // a lightweight smoke test that the key names match what the docs
        // and examples advertise.
        let content = include_str!("../../examples/warfarin_bobyqa.ferx");
        let parsed = parse_full_model(content).unwrap();
        assert_eq!(parsed.fit_options.optimizer, Optimizer::Bobyqa);
        assert_eq!(parsed.fit_options.outer_maxiter, 300);
        assert_eq!(parsed.fit_options.inner_maxiter, 100);
    }

    #[test]
    fn test_parse_example_warfarin_trust_region_file() {
        let content = include_str!("../../examples/warfarin_trust_region.ferx");
        let parsed = parse_full_model(content).unwrap();
        assert_eq!(parsed.fit_options.optimizer, Optimizer::TrustRegion);
        assert_eq!(parsed.fit_options.steihaug_max_iters, Some(30));
    }

    // ── apply_fit_option: coverage of the newly-added optimizer keys.
    //    The generic apply_fit_option tests (known/unknown/malformed/bool
    //    variants/threads/seed) live in the earlier test block — these
    //    only add the keys that are new on this branch.

    #[test]
    fn test_apply_fit_option_optimizer_bobyqa() {
        let mut opts = FitOptions::default();
        assert_eq!(apply_fit_option(&mut opts, "optimizer", "bobyqa"), Ok(true));
        assert_eq!(opts.optimizer, Optimizer::Bobyqa);
    }

    #[test]
    fn test_apply_fit_option_optimizer_trust_region() {
        let mut opts = FitOptions::default();
        assert_eq!(
            apply_fit_option(&mut opts, "optimizer", "trust_region"),
            Ok(true)
        );
        assert_eq!(opts.optimizer, Optimizer::TrustRegion);
    }

    #[test]
    fn test_apply_fit_option_steihaug_max_iters() {
        let mut opts = FitOptions::default();
        assert_eq!(
            apply_fit_option(&mut opts, "steihaug_max_iters", "30"),
            Ok(true)
        );
        assert_eq!(opts.steihaug_max_iters, Some(30));
        // Reject malformed (e.g. negative) value.
        assert!(apply_fit_option(&mut opts, "steihaug_max_iters", "-1").is_err());
    }

    #[test]
    fn test_apply_fit_option_inner_maxiter_and_tol() {
        let mut opts = FitOptions::default();
        assert_eq!(apply_fit_option(&mut opts, "inner_maxiter", "75"), Ok(true));
        assert_eq!(opts.inner_maxiter, 75);

        assert_eq!(apply_fit_option(&mut opts, "inner_tol", "1e-5"), Ok(true));
        assert!((opts.inner_tol - 1e-5).abs() < 1e-15);

        assert!(apply_fit_option(&mut opts, "inner_maxiter", "oops").is_err());
        assert!(apply_fit_option(&mut opts, "inner_tol", "not_a_num").is_err());
    }

    #[test]
    fn test_apply_fit_option_fd_hessian_step_valid() {
        let mut opts = FitOptions::default();
        assert_eq!(
            apply_fit_option(&mut opts, "fd_hessian_step", "0.05"),
            Ok(true)
        );
        assert!((opts.fd_hessian_step - 0.05).abs() < 1e-15);
    }

    #[test]
    fn test_apply_fit_option_fd_hessian_step_zero_rejected() {
        let mut opts = FitOptions::default();
        let err = apply_fit_option(&mut opts, "fd_hessian_step", "0").unwrap_err();
        assert!(
            err.contains("positive"),
            "error must mention 'positive': {err}"
        );
    }

    #[test]
    fn test_apply_fit_option_fd_hessian_step_negative_rejected() {
        let mut opts = FitOptions::default();
        let err = apply_fit_option(&mut opts, "fd_hessian_step", "-0.01").unwrap_err();
        assert!(
            err.contains("positive"),
            "error must mention 'positive': {err}"
        );
    }

    // ── IOV: kappa keyword and iov_column ──────────────────────────────────

    #[test]
    fn test_parse_kappa_keyword() {
        let lines = vec!["kappa KAPPA_CL ~ 0.01".to_string()];
        let (_, _, _, _, _, ki) = parse_parameters(&lines).unwrap();
        assert_eq!(ki.diagonal.len(), 1);
        assert_eq!(ki.diagonal[0].name, "KAPPA_CL");
        assert!((ki.diagonal[0].variance - 0.01).abs() < 1e-12);
        assert!(!ki.diagonal[0].fixed);
    }

    #[test]
    fn test_parse_kappa_fix() {
        let lines = vec!["kappa KAPPA_V ~ 0.05 FIX".to_string()];
        let (_, _, _, _, _, ki) = parse_parameters(&lines).unwrap();
        assert!(ki.diagonal[0].fixed);
    }

    #[test]
    fn test_kappa_unfixed_no_annotation() {
        // Baseline: plain kappa with no FIX and no annotation — confirms the
        // group-numbering shift didn't regress the common case.
        let lines = vec!["kappa KAPPA_V ~ 0.05".to_string()];
        let (_, _, _, _, _, ki) = parse_parameters(&lines).unwrap();
        assert!(!ki.diagonal[0].fixed);
        assert!(!ki.diagonal[0].init_as_sd);
        assert!((ki.diagonal[0].variance - 0.05).abs() < 1e-12);
    }

    #[test]
    fn test_kappa_fix_before_sd_annotation() {
        // `FIX (sd)` — FIX before the scale annotation for kappa.
        let lines = vec!["kappa KAPPA_V ~ 0.30 FIX (sd)".to_string()];
        let (_, _, _, _, _, ki) = parse_parameters(&lines).unwrap();
        let expected = 0.30 * 0.30;
        assert!((ki.diagonal[0].variance - expected).abs() < 1e-12);
        assert!(ki.diagonal[0].fixed);
        assert!(ki.diagonal[0].init_as_sd);
    }

    #[test]
    fn test_kappa_appended_after_bsv_etas() {
        // kappa names must NOT appear in the BSV eta_names list returned
        // as the 5th element; they only appear in the 6th (kappas) element.
        let lines = vec![
            "omega ETA_CL ~ 0.09".to_string(),
            "kappa KAPPA_CL ~ 0.01".to_string(),
        ];
        let (_, _, _, _, bsv_etas, ki) = parse_parameters(&lines).unwrap();
        assert_eq!(bsv_etas, vec!["ETA_CL"]);
        assert_eq!(ki.diagonal.len(), 1);
        assert_eq!(ki.diagonal[0].name, "KAPPA_CL");
    }

    #[test]
    fn test_parse_full_model_with_kappa() {
        let content = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let parsed = parse_full_model(content).unwrap();
        let m = &parsed.model;
        assert_eq!(m.n_eta, 2); // BSV only
        assert_eq!(m.n_kappa, 1);
        assert_eq!(m.kappa_names, vec!["KAPPA_CL"]);
        assert!(m.default_params.omega_iov.is_some());
        let iov = m.default_params.omega_iov.as_ref().unwrap();
        assert_eq!(iov.dim(), 1);
        assert!((iov.matrix[(0, 0)] - 0.01).abs() < 1e-12);
    }

    #[test]
    fn test_iov_column_fit_option() {
        let mut opts = FitOptions::default();
        assert_eq!(apply_fit_option(&mut opts, "iov_column", "OCC"), Ok(true));
        assert_eq!(opts.iov_column, Some("OCC".to_string()));
    }

    #[test]
    fn test_iov_column_none_values() {
        let mut opts = FitOptions::default();
        apply_fit_option(&mut opts, "iov_column", "OCC").unwrap();
        apply_fit_option(&mut opts, "iov_column", "none").unwrap();
        assert!(opts.iov_column.is_none());
    }

    #[test]
    fn test_iov_column_parsed_from_fit_options_block() {
        let content = minimal_model_with_fit_options("  iov_column = PERIOD");
        let parsed = parse_full_model(&content).unwrap();
        assert_eq!(parsed.fit_options.iov_column, Some("PERIOD".to_string()));
    }

    // ── block_kappa (Option B) ─────────────────────────────────────────────

    #[test]
    fn test_parse_block_kappa_syntax() {
        let lines = vec!["block_kappa (KAPPA_CL, KAPPA_V) = [0.01, 0.002, 0.005]".to_string()];
        let (_, _, _, _, _, ki) = parse_parameters(&lines).unwrap();
        assert_eq!(ki.diagonal.len(), 0);
        assert_eq!(ki.block.len(), 1);
        assert_eq!(ki.block[0].names, vec!["KAPPA_CL", "KAPPA_V"]);
        assert_eq!(ki.block[0].lower_triangle, vec![0.01, 0.002, 0.005]);
        assert!(!ki.block[0].fixed);
        assert_eq!(ki.names_ordered, vec!["KAPPA_CL", "KAPPA_V"]);
    }

    #[test]
    fn test_parse_block_kappa_fix() {
        let lines = vec!["block_kappa (KAPPA_CL, KAPPA_V) = [0.01, 0.002, 0.005] FIX".to_string()];
        let (_, _, _, _, _, ki) = parse_parameters(&lines).unwrap();
        assert!(ki.block[0].fixed);
    }

    #[test]
    fn test_parse_block_kappa_wrong_count_errors() {
        // 2 names → need 3 values, only 2 given
        let lines = vec!["block_kappa (KAPPA_CL, KAPPA_V) = [0.01, 0.002]".to_string()];
        assert!(parse_parameters(&lines).is_err());
    }

    #[test]
    fn test_parse_block_kappa_name_overlap_errors() {
        let lines = vec![
            "kappa KAPPA_CL ~ 0.01".to_string(),
            "block_kappa (KAPPA_CL, KAPPA_V) = [0.01, 0.002, 0.005]".to_string(),
        ];
        assert!(parse_parameters(&lines).is_err());
    }

    #[test]
    fn test_parse_full_model_with_block_kappa() {
        let content = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  block_kappa (KAPPA_CL, KAPPA_V) = [0.01, 0.002, 0.005]
  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V + KAPPA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let parsed = parse_full_model(content).unwrap();
        let m = &parsed.model;
        assert_eq!(m.n_eta, 2);
        assert_eq!(m.n_kappa, 2);
        assert_eq!(m.kappa_names, vec!["KAPPA_CL", "KAPPA_V"]);
        let iov = m.default_params.omega_iov.as_ref().unwrap();
        assert_eq!(iov.dim(), 2);
        assert!(
            !iov.diagonal,
            "block_kappa should produce non-diagonal omega_iov"
        );
        assert!((iov.matrix[(0, 0)] - 0.01).abs() < 1e-12);
        assert!((iov.matrix[(0, 1)] - 0.002).abs() < 1e-12);
        assert!((iov.matrix[(1, 0)] - 0.002).abs() < 1e-12);
        assert!((iov.matrix[(1, 1)] - 0.005).abs() < 1e-12);
    }

    // ── if-statement support ───────────────────────────────────────────────
    //
    // Tests the multi-line `if (cond) { ... } else if (...) { ... } else { ... }`
    // syntax and the inline `if (cond) expr else expr` ternary. Coverage:
    // tokenizer, statement parser, evaluator, integration with mu-ref
    // detection, and ODE bodies.

    fn empty_ctx() -> ParseCtx<'static> {
        const TN: &[String] = &[];
        const EN: &[String] = &[];
        const DV: &[String] = &[];
        ParseCtx::new(TN, EN, DV)
    }

    #[test]
    fn test_tokenize_comparison_and_logical_operators() {
        let toks = tokenize("a >= 1 && !(b == 2 || c <= 3)").unwrap();
        // We only assert that the two-character ops landed as single tokens —
        // the rest of the stream is asserted indirectly via parser tests.
        assert!(toks.contains(&Token::Ge));
        assert!(toks.contains(&Token::AndAnd));
        assert!(toks.contains(&Token::Bang));
        assert!(toks.contains(&Token::EqEq));
        assert!(toks.contains(&Token::OrOr));
        assert!(toks.contains(&Token::Le));
    }

    #[test]
    fn test_tokenize_newlines_and_braces() {
        let toks = tokenize("if (x > 0) {\n  y = 1\n}").unwrap();
        assert!(toks.contains(&Token::LBrace));
        assert!(toks.contains(&Token::RBrace));
        assert!(toks.contains(&Token::Newline));
        assert!(toks.contains(&Token::Eq));
        assert!(toks.contains(&Token::Gt));
    }

    #[test]
    fn test_strip_newlines_keeps_brace_separators() {
        let toks = tokenize("if (a >\n  1) {\n  y = 2\n}").unwrap();
        let stripped = strip_newlines_in_groups(toks);
        // Newline inside the parens is gone; newlines inside the braces stay
        // (statement separators). Two newlines are typed inside the brace body
        // (one after `{` and one after `y = 2`); both survive.
        let n_newlines = stripped.iter().filter(|t| **t == Token::Newline).count();
        assert_eq!(n_newlines, 2);
    }

    #[test]
    fn test_inline_if_expression_parses_and_evaluates() {
        let tn = vec!["TVCL".to_string()];
        let en: Vec<String> = vec![];
        let ctx = ParseCtx::new(&tn, &en, &[]);
        let stmts = parse_block_statements(
            "CL = if (WT > 70) TVCL * 1.2 else TVCL",
            ctx,
            StatementMode::Plain,
        )
        .unwrap();
        assert_eq!(stmts.len(), 1);

        let mut vars = HashMap::new();
        let theta = vec![5.0];
        let mut covs = HashMap::new();
        covs.insert("WT".to_string(), 80.0);
        eval_statements(&stmts, &theta, &[], &covs, &mut vars, None, None, &[]);
        assert!(
            (vars["CL"] - 6.0).abs() < 1e-12,
            "CL should pick the then-branch"
        );

        covs.insert("WT".to_string(), 60.0);
        let mut vars2 = HashMap::new();
        eval_statements(&stmts, &theta, &[], &covs, &mut vars2, None, None, &[]);
        assert!(
            (vars2["CL"] - 5.0).abs() < 1e-12,
            "CL should pick the else-branch"
        );
    }

    #[test]
    fn test_multiline_if_else_block_evaluates() {
        let tn = vec!["TVCL".to_string()];
        let block = "
if (WT > 70) {
  CL = TVCL * (WT/70)
} else {
  CL = TVCL
}
";
        let ctx = ParseCtx::new(&tn, &[], &[]);
        let stmts = parse_block_statements(block, ctx, StatementMode::Plain).unwrap();
        assert_eq!(stmts.len(), 1, "single top-level if-statement");

        let theta = vec![10.0];
        for (wt, expected) in [(80.0, 10.0 * (80.0 / 70.0)), (50.0, 10.0)] {
            let mut covs = HashMap::new();
            covs.insert("WT".to_string(), wt);
            let mut vars = HashMap::new();
            eval_statements(&stmts, &theta, &[], &covs, &mut vars, None, None, &[]);
            assert!(
                (vars["CL"] - expected).abs() < 1e-12,
                "WT={} → expected CL={}",
                wt,
                expected
            );
        }
    }

    #[test]
    fn test_else_if_chain_picks_first_match() {
        let tn = vec!["TVCL".to_string()];
        let block = "
if (X < 10) {
  CL = TVCL * 0.5
} else if (X < 20) {
  CL = TVCL
} else if (X < 30) {
  CL = TVCL * 1.5
} else {
  CL = TVCL * 2.0
}
";
        let ctx = ParseCtx::new(&tn, &[], &[]);
        let stmts = parse_block_statements(block, ctx, StatementMode::Plain).unwrap();
        let theta = vec![10.0];
        for (x, expected) in [(5.0, 5.0), (15.0, 10.0), (25.0, 15.0), (40.0, 20.0)] {
            let mut covs = HashMap::new();
            covs.insert("X".to_string(), x);
            let mut vars = HashMap::new();
            eval_statements(&stmts, &theta, &[], &covs, &mut vars, None, None, &[]);
            assert!(
                (vars["CL"] - expected).abs() < 1e-12,
                "X={x} should pick branch giving CL={expected}, got {}",
                vars["CL"]
            );
        }
    }

    #[test]
    fn test_logical_operators_in_condition() {
        let tn = vec!["TVCL".to_string()];
        let block = "CL = if ((SEX == 1 && WT > 70) || AGE >= 65) TVCL * 1.5 else TVCL";
        let ctx = ParseCtx::new(&tn, &[], &[]);
        let stmts = parse_block_statements(block, ctx, StatementMode::Plain).unwrap();
        let theta = vec![10.0];
        let cases = [
            // (sex, wt, age, expected)
            (1.0, 80.0, 30.0, 15.0), // && true → boost
            (1.0, 60.0, 30.0, 10.0), // && false, || false → no boost
            (0.0, 50.0, 70.0, 15.0), // age >= 65 → boost
            (0.0, 50.0, 64.999, 10.0),
        ];
        for (sex, wt, age, expected) in cases {
            let mut covs = HashMap::new();
            covs.insert("SEX".to_string(), sex);
            covs.insert("WT".to_string(), wt);
            covs.insert("AGE".to_string(), age);
            let mut vars = HashMap::new();
            eval_statements(&stmts, &theta, &[], &covs, &mut vars, None, None, &[]);
            assert!(
                (vars["CL"] - expected).abs() < 1e-12,
                "sex={sex} wt={wt} age={age} expected CL={expected}, got {}",
                vars["CL"]
            );
        }
    }

    #[test]
    fn test_if_statement_disables_mu_ref_for_var() {
        // CL is assigned only inside an if-block — it must NOT participate in
        // mu-referencing because the (eta, theta) relationship is conditional.
        // V is unconditional and SHOULD still mu-ref.
        let content = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)

  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04

  sigma PROP_ERR ~ 0.02

[individual_parameters]
  if (WT > 70) {
    CL = TVCL * (WT/70) * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let parsed = parse_full_model(content).expect("model should parse");
        // Only ETA_V participates in mu-referencing.
        assert!(parsed.model.mu_refs.contains_key("ETA_V"));
        assert!(!parsed.model.mu_refs.contains_key("ETA_CL"));
    }

    #[test]
    fn test_if_statement_collects_covariates_from_all_branches() {
        // Covariates referenced inside any branch (including the condition)
        // must show up in the model's referenced_covariates list.
        let content = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)

  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04

  sigma PROP_ERR ~ 0.02

[individual_parameters]
  if (SEX == 1) {
    CL = TVCL * (WT/70) * exp(ETA_CL)
  } else {
    CL = TVCL * (CRCL/100) * exp(ETA_CL)
  }
  V = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let parsed = parse_full_model(content).expect("model should parse");
        let covs = &parsed.model.referenced_covariates;
        assert!(covs.contains(&"SEX".to_string()));
        assert!(covs.contains(&"WT".to_string()));
        assert!(covs.contains(&"CRCL".to_string()));
    }

    #[test]
    fn test_pk_param_fn_runs_branch_at_runtime() {
        // End-to-end: parse a model with an if-block, run pk_param_fn at two
        // different covariate values, and confirm CL takes different values.
        let content = r#"
[parameters]
  theta TVCL(2.0, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)

  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04

  sigma PROP_ERR ~ 0.02

[individual_parameters]
  if (WT > 70) {
    CL = TVCL * 2.0
  } else {
    CL = TVCL
  }
  V = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let parsed = parse_full_model(content).unwrap();
        let theta = vec![2.0, 10.0];
        let eta = vec![0.0, 0.0];

        let mut covs_heavy = HashMap::new();
        covs_heavy.insert("WT".to_string(), 100.0);
        let p_heavy = (parsed.model.pk_param_fn)(&theta, &eta, &covs_heavy);
        assert!((p_heavy.values[0] - 4.0).abs() < 1e-12, "WT=100 → CL=4");

        let mut covs_light = HashMap::new();
        covs_light.insert("WT".to_string(), 50.0);
        let p_light = (parsed.model.pk_param_fn)(&theta, &eta, &covs_light);
        assert!((p_light.values[0] - 2.0).abs() < 1e-12, "WT=50 → CL=2");
    }

    #[test]
    fn test_ode_block_supports_if_statements() {
        // ODE block with an if-statement that switches the elimination term
        // depending on whether central is above a threshold (Michaelis-Menten
        // approximation toggle).
        let content = r#"
[parameters]
  theta TVCL(2.0, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)

  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04

  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot) = 0
  if (central > 0) {
    d/dt(central) = -CL/V * central
  } else {
    d/dt(central) = 0
  }

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let parsed = parse_full_model(content).expect("ODE model should parse");
        let ode = parsed.model.ode_spec.as_ref().expect("ode_spec present");
        // States are [depot, central]; params are [CL, V] (declaration order in
        // [individual_parameters]). du must match n_states.
        let params = vec![2.0, 10.0];

        // central > 0 → if-branch fires, du[central] = -CL/V * central
        let u_pos = vec![0.0, 5.0];
        let mut du_pos = vec![0.0, 0.0];
        (ode.rhs)(&u_pos, &params, 0.0, &mut du_pos);
        assert!((du_pos[1] - (-2.0 / 10.0 * 5.0)).abs() < 1e-12);

        // central == 0 → else-branch fires, du[central] = 0. Pre-seed with junk
        // so the test would fail if the branch silently no-op'd instead of
        // assigning zero.
        let u_zero = vec![0.0, 0.0];
        let mut du_zero = vec![999.0, 999.0];
        (ode.rhs)(&u_zero, &params, 0.0, &mut du_zero);
        assert!(
            du_zero[1].abs() < 1e-12,
            "else-branch should emit 0, got {}",
            du_zero[1]
        );
    }

    #[test]
    fn test_inline_if_in_parameter_assignment() {
        // Inline ternary form used directly as the RHS of a [individual_parameters]
        // assignment — this is the most concise form and should produce a
        // working pk_param_fn.
        let content = r#"
[parameters]
  theta TVCL(2.0, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)

  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04

  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = if (SEX == 1) TVCL * 1.5 else TVCL
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let parsed = parse_full_model(content).unwrap();
        let theta = vec![2.0, 10.0];
        let eta = vec![0.0, 0.0];

        let mut male = HashMap::new();
        male.insert("SEX".to_string(), 1.0);
        let p_male = (parsed.model.pk_param_fn)(&theta, &eta, &male);
        assert!((p_male.values[0] - 3.0).abs() < 1e-12);

        let mut female = HashMap::new();
        female.insert("SEX".to_string(), 0.0);
        let p_female = (parsed.model.pk_param_fn)(&theta, &eta, &female);
        assert!((p_female.values[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_missing_else_branch_in_inline_if_errors() {
        let ctx = empty_ctx();
        let res = parse_block_statements("CL = if (1 > 0) 5", ctx, StatementMode::Plain);
        assert!(res.is_err(), "inline if without else must error");
    }

    #[test]
    fn test_missing_brace_in_if_block_errors() {
        let ctx = empty_ctx();
        let res = parse_block_statements("if (1 > 0) CL = 5", ctx, StatementMode::Plain);
        assert!(res.is_err(), "block-form if without `{{` must error");
    }

    #[test]
    fn test_assigned_vars_in_order_includes_nested() {
        // Vars assigned inside if-bodies still count toward the ordered list
        // (so they're reachable as Variables in later expressions and end up
        // in the per-tv tables).
        let block = "
A = 1
if (1 > 0) {
  B = 2
  C = A + B
} else {
  D = 99
}
E = C + 1
";
        let ctx = empty_ctx();
        let stmts = parse_block_statements(block, ctx, StatementMode::Plain).unwrap();
        let names = assigned_vars_in_order(&stmts);
        assert_eq!(names, vec!["A", "B", "C", "D", "E"]);
    }

    // ── Regression tests for review-identified bugs ──────────────────────────

    #[test]
    fn test_parse_cond_atom_nested_parens_around_simple_compare() {
        // ((SEX == 1)) — double-parens around a comparison must parse correctly.
        // Previously the lookahead heuristic (only enter sub-condition path when
        // || or && is present) caused this to fail with "Missing closing )".
        let ctx = empty_ctx();
        let block = "X = if ((SEX == 1)) 1.0 else 2.0";
        assert!(
            parse_block_statements(block, ctx, StatementMode::Plain).is_ok(),
            "double-paren condition should parse"
        );
    }

    #[test]
    fn test_parse_cond_atom_nested_parens_around_negation() {
        // (!(SEX == 1)) — parens around a negation must parse.
        let ctx = empty_ctx();
        let block = "X = if (!(SEX == 1)) 1.0 else 2.0";
        assert!(
            parse_block_statements(block, ctx, StatementMode::Plain).is_ok(),
            "parenthesised negation should parse"
        );
    }

    #[test]
    fn test_top_level_assigned_vars_excludes_branch_locals() {
        // top_level_assigned_vars must NOT include variables assigned only
        // inside if-branches — those would corrupt the AD PK slot layout.
        let block = "
CL = 1.0
if (1 > 0) {
  SCALE = 2.0
  V = SCALE * 3.0
} else {
  V = 4.0
}
";
        let ctx = empty_ctx();
        let stmts = parse_block_statements(block, ctx, StatementMode::Plain).unwrap();
        let top = top_level_assigned_vars(&stmts);
        // SCALE is only assigned inside the if-body — must not appear.
        assert_eq!(top, vec!["CL"], "top-level vars should only contain CL");
        // But all_assigned_vars still sees SCALE and V.
        let all = assigned_vars_in_order(&stmts);
        assert!(all.contains(&"SCALE".to_string()));
        assert!(all.contains(&"V".to_string()));
    }

    #[test]
    fn test_unconditionally_assigned_vars_includes_all_branch_assignments() {
        // V is assigned on BOTH branches → unconditionally defined → promoted.
        // SCALE is assigned in only one branch → branch-local → excluded.
        // (Issue #357: a param written on every branch must earn a PK slot.)
        let block = "
CL = 1.0
if (1 > 0) {
  SCALE = 2.0
  V = SCALE * 3.0
} else {
  V = 4.0
}
";
        let ctx = empty_ctx();
        let stmts = parse_block_statements(block, ctx, StatementMode::Plain).unwrap();
        let uncond = unconditionally_assigned_vars(&stmts);
        assert_eq!(
            uncond,
            vec!["CL", "V"],
            "V is assigned on every branch and must be promoted; SCALE must not"
        );
        // top_level still excludes V (its only assignments are inside branches).
        let top = top_level_assigned_vars(&stmts);
        assert_eq!(top, vec!["CL"]);
    }

    #[test]
    fn test_unconditionally_assigned_vars_branch_only_param() {
        // The exact issue #357 shape: CL assigned only inside if/else, V at
        // top level. CL must come first (leading-branch order), then V.
        let block = "
if (WT > 70) {
  CL = TVCL * 1.5
} else {
  CL = TVCL
}
V = TVV
";
        let ctx = empty_ctx();
        let stmts = parse_block_statements(block, ctx, StatementMode::Plain).unwrap();
        let uncond = unconditionally_assigned_vars(&stmts);
        assert_eq!(uncond, vec!["CL", "V"]);
    }

    #[test]
    fn test_unconditionally_assigned_vars_if_without_else_excludes() {
        // No `else` → the name could be undefined on the fall-through path, so
        // it stays branch-local (not promoted).
        let block = "
CL = 1.0
if (WT > 70) {
  V = 2.0
}
";
        let ctx = empty_ctx();
        let stmts = parse_block_statements(block, ctx, StatementMode::Plain).unwrap();
        let uncond = unconditionally_assigned_vars(&stmts);
        assert_eq!(
            uncond,
            vec!["CL"],
            "V lacks an else branch — not unconditional"
        );
    }

    #[test]
    fn test_duplicate_diffeq_in_same_scope_errors() {
        // Two d/dt(central) at top level must be rejected.
        let block_text = "d/dt(central) = -0.1 * central\nd/dt(central) = -0.2 * central";
        let state_names = vec!["central".to_string()];
        let ctx = ParseCtx::ode(&state_names);
        let stmts = parse_block_statements(block_text, ctx, StatementMode::Ode).unwrap();
        let result = (|| -> Result<(), String> {
            fn check(stmts: &[Statement]) -> Result<(), String> {
                let mut seen = std::collections::HashSet::new();
                for s in stmts {
                    match s {
                        Statement::DiffEq(name, _) => {
                            if !seen.insert(name.clone()) {
                                return Err(format!("duplicate d/dt({})", name));
                            }
                        }
                        Statement::If {
                            branches,
                            else_body,
                        } => {
                            for (_, b) in branches {
                                check(b)?;
                            }
                            if let Some(eb) = else_body {
                                check(eb)?;
                            }
                        }
                        Statement::Assign(_, _)
                        | Statement::AssignIdx(_, _)
                        | Statement::DiffEqIdx(_, _)
                        | Statement::AssignBc(_, _)
                        | Statement::DiffEqBc(_, _) => {}
                    }
                }
                Ok(())
            }
            check(&stmts)
        })();
        assert!(result.is_err(), "duplicate d/dt in same scope must error");
    }

    #[test]
    fn test_duplicate_diffeq_in_different_branches_allowed() {
        // d/dt(central) in if-branch AND else-branch is legitimate.
        // build_ode_spec must accept it.
        let ode_lines: Vec<String> = vec![
            "if (1 > 0) {".into(),
            "  d/dt(central) = -0.1 * central".into(),
            "} else {".into(),
            "  d/dt(central) = -0.2 * central".into(),
            "}".into(),
        ];
        let state_names = vec!["central".to_string()];
        let result = build_ode_spec(&ode_lines, &state_names, Some("central"), &[], &[]);
        assert!(
            result.is_ok(),
            "same state in different branches must be allowed"
        );
    }

    #[test]
    fn test_ode_rhs_undefined_name_errors() {
        // #314: a name in an ODE RHS that is not a state, individual parameter,
        // intermediate, or reserved time var must error — it would otherwise
        // resolve to the `usize::MAX` sentinel and silently read 0.0, producing
        // a structurally-broken fit. Here `central` is a state and `CL` is an
        // individual parameter, but `V` is undeclared.
        let ode_lines: Vec<String> = vec!["d/dt(central) = -(CL/V) * central".into()];
        let state_names = vec!["central".to_string()];
        let result = build_ode_spec(
            &ode_lines,
            &state_names,
            Some("central"),
            &["CL".to_string()],
            &[0],
        );
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("undefined RHS name `V` must error"),
        };
        assert!(
            err.contains("undefined name(s): V."),
            "error should name the undefined `V`, got: {err}"
        );
        // The defined names are listed to make the fix obvious.
        assert!(
            err.contains("CL") && err.contains("central"),
            "error should list the defined names, got: {err}"
        );
    }

    #[test]
    fn test_ode_rhs_undefined_name_walks_all_nodes() {
        // Exercises every arm of the undefined-name walker: an intermediate
        // assignment RHS, a block `if` condition, `&&`/`||`/`!` boolean
        // operators, an inline conditional, `exp(...)`, and `^`. Every BAD*
        // name is undefined; `k` (intermediate), `central` (state), `CL`
        // (param), and `TIME` (reserved) must NOT be flagged.
        let ode_lines: Vec<String> = vec![
            "k = CL / V1".into(),
            "if (BADIF > 0) {".into(),
            "  d/dt(central) = exp(BADEXP) + BADPOW^2 - k * central + (if (BADAND1 > 0 && BADAND2 > 0) 1.0 else 0.0) + (if (BADOR1 > 0 || BADOR2 > 0) 1.0 else 0.0) + (if (!(BADNOT > 0)) 1.0 else 0.0)".into(),
            "} else {".into(),
            "  d/dt(central) = -k * central".into(),
            "}".into(),
        ];
        let state_names = vec!["central".to_string()];
        let result = build_ode_spec(
            &ode_lines,
            &state_names,
            Some("central"),
            &["CL".to_string()],
            &[0],
        );
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("undefined names in nested expressions must error"),
        };
        for bad in [
            "V1", "BADIF", "BADEXP", "BADPOW", "BADAND1", "BADAND2", "BADOR1", "BADOR2", "BADNOT",
        ] {
            assert!(err.contains(bad), "error should name `{bad}`, got: {err}");
        }
    }

    #[test]
    fn test_ode_rhs_defined_names_ok() {
        // Regression guard against false positives: an intermediate (`k`),
        // individual parameters (`CL`, `V`), a state (`central`), and the
        // reserved `TIME` variable must all resolve and parse cleanly.
        let ode_lines: Vec<String> = vec![
            "k = CL / V".into(),
            "d/dt(central) = if (TIME < 24.0) -k * central else 0.0".into(),
        ];
        let state_names = vec!["central".to_string()];
        let result = build_ode_spec(
            &ode_lines,
            &state_names,
            Some("central"),
            &["CL".to_string(), "V".to_string()],
            &[0, 1],
        );
        assert!(
            result.is_ok(),
            "intermediate + params + TIME must parse, got: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_ode_rhs_macheps_resolves_to_epsilon() {
        // MACHEPS is a reserved ODE builtin (machine epsilon): it must resolve
        // to f64::EPSILON in an ODE RHS — not be flagged as undefined, and not
        // silently read 0.0.
        let ode_lines: Vec<String> = vec!["d/dt(central) = MACHEPS".into()];
        let state_names = vec!["central".to_string()];
        let spec = match build_ode_spec(&ode_lines, &state_names, Some("central"), &[], &[]) {
            Ok(s) => s,
            Err(e) => panic!("MACHEPS in an ODE RHS must parse, got: {e}"),
        };
        let params = vec![0.0; crate::types::MAX_PK_PARAMS + 2];
        let mut du = vec![0.0_f64];
        (spec.rhs)(&[0.0], &params, 0.0, &mut du);
        assert_eq!(
            du[0],
            f64::EPSILON,
            "d/dt(central) = MACHEPS must evaluate to machine epsilon"
        );
    }

    #[test]
    fn test_ode_reserved_builtin_name_collision_errors() {
        // A state / individual parameter / intermediate may not reuse a reserved
        // builtin name — `MACHEPS` is now reserved alongside TIME/TAFD/TAD.
        let ode_lines: Vec<String> = vec!["d/dt(MACHEPS) = -MACHEPS".into()];
        let state_names = vec!["MACHEPS".to_string()];
        let result = build_ode_spec(&ode_lines, &state_names, Some("MACHEPS"), &[], &[]);
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("a state named MACHEPS must collide with the reserved builtin"),
        };
        assert!(
            err.contains("MACHEPS") && err.contains("reserved"),
            "expected a reserved-name collision error, got: {err}"
        );
    }

    #[test]
    fn test_mu_ref_warning_for_conditional_param() {
        // A model where CL is assigned only inside an if-block should emit a
        // parse_warning about mu-referencing being disabled.
        let model_str = "
[parameters]
  theta TVCL(1.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  if (1 > 0) {
    CL = TVCL * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }

[structural_model]
  pk one_cpt_oral(cl=CL, v=1.0, ka=1.0)

[error_model]
  DV ~ proportional(EPS)
";
        let parsed = super::parse_full_model(model_str).unwrap();
        assert!(
            !parsed.model.parse_warnings.is_empty(),
            "expected a parse warning about mu-ref disabled; got: {:?}",
            parsed.model.parse_warnings
        );
        let w = &parsed.model.parse_warnings[0];
        assert!(w.contains("CL"), "warning should mention CL");
        assert!(
            w.contains("Mu-referencing disabled"),
            "warning should mention mu-referencing"
        );
    }

    #[test]
    fn nested_if_eta_param_sets_conditional_flag() {
        // Regression for #278/#280: an eta-bearing parameter assigned inside a
        // *nested* if-branch must still set `has_conditional_eta_params`, so the
        // inner loop routes to FD instead of the analytical AD kernel (which
        // cannot represent the branch). Detection previously looked only one
        // level deep and silently missed nested conditionals, leaving the model
        // on a wrong AD gradient — the exact failure class this gate prevents.
        let plain = minimal_model_with_indiv(
            "  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)",
        );
        assert!(
            !plain.has_conditional_eta_params,
            "an unconditional model must not be flagged conditional"
        );

        let nested = minimal_model_with_indiv(
            "  CL = TVCL
  V  = TVV * exp(ETA_V)
  if (1 > 0) {
    if (1 > 0) {
      CL = TVCL * exp(ETA_CL)
    } else {
      CL = TVCL * exp(ETA_CL)
    }
  }",
        );
        assert!(
            nested.has_conditional_eta_params,
            "eta-bearing param assigned only inside a nested if must set \
             has_conditional_eta_params"
        );
    }

    #[test]
    fn test_indiv_param_names_populated_in_declaration_order() {
        // CompiledModel.indiv_param_names must hold every top-level
        // [individual_parameters] assignment, in source-declaration order.
        // Downstream consumers (the R FFI's per-subject EBE table) rely on
        // this list to label the columns of `individual_estimates`, and on
        // its alignment with `pk_indices` to read each value out of the
        // PkParams slot.
        let model_str = "
[parameters]
  theta TVCL(1.0)
  theta TVV(10.0)
  theta TVKA(2.0)
  omega ETA_CL ~ 0.1
  omega ETA_V  ~ 0.1
  omega ETA_KA ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(EPS)
";
        let parsed = super::parse_full_model(model_str).unwrap();
        assert_eq!(
            parsed.model.indiv_param_names,
            vec!["CL".to_string(), "V".to_string(), "KA".to_string()]
        );
        // The list must be parallel to pk_indices so the FFI can route each
        // name to its PkParams slot for analytical models.
        assert_eq!(
            parsed.model.indiv_param_names.len(),
            parsed.model.pk_indices.len()
        );
    }

    fn minimal_model_with_indiv(indiv_block: &str) -> crate::types::CompiledModel {
        let model_str = format!(
            r"
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.1
  omega ETA_V  ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
{}

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
",
            indiv_block
        );
        super::parse_full_model(&model_str).unwrap().model
    }

    fn minimal_logit_model() -> crate::types::CompiledModel {
        let model_str = r"
[parameters]
  theta THETA_F(0.0, -10.0, 10.0)
  sigma EPS ~ 0.01
  omega ETA_F  ~ 0.1

[individual_parameters]
  F = inv_logit(THETA_F + ETA_F)

[structural_model]
  pk one_cpt_iv(cl=1, v=1)

[error_model]
  DV ~ proportional(EPS)
";
        super::parse_full_model(model_str).unwrap().model
    }

    #[test]
    fn test_inv_logit_evaluates() {
        let model = minimal_logit_model();
        let tv = model.tv_fn.as_ref().unwrap()(&[0.0], &Default::default());
        let expected = 0.5_f64; // inv_logit(0) = 0.5
        assert!(
            (tv[0] - expected).abs() < 1e-10,
            "inv_logit(0) should be 0.5, got {}",
            tv[0]
        );
    }

    #[test]
    fn test_classify_lognormal_multiplicative() {
        use crate::types::{EtaParamType, ThetaTransform};
        // CL = TVCL * exp(ETA_CL)
        let model = minimal_model_with_indiv("  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)");
        assert_eq!(model.eta_param_info.len(), 2);
        let cl_info = model
            .eta_param_info
            .iter()
            .find(|i| i.eta_name == "ETA_CL")
            .unwrap();
        assert_eq!(cl_info.param_type, EtaParamType::LogNormal);
        // Anchor is the theta TVCL — linked_theta is now populated.
        assert_eq!(cl_info.linked_theta, Some("TVCL".to_string()));
        // theta_transform for TVCL (theta index 0) stays Identity — the
        // product pattern does not imply the theta is on the log scale.
        assert_eq!(model.theta_transform[0], ThetaTransform::Identity);
    }

    /// `TVCL * exp(ETA_CL + KAPPA_CL)` — IIV and IOV on the same parameter.
    /// ETA_CL (BSV) should be LogNormal with linked_theta TVCL.
    /// Also verifies that detect_mu_refs sets mu_ref for ETA_CL → TVCL.
    #[test]
    fn test_classify_iov_combined() {
        use crate::types::EtaParamType;
        let src = r"
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma EPS ~ 0.01
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(EPS)
[fit_options]
  iov_column = OCC
";
        let model = super::parse_full_model(src).unwrap().model;
        let cl_info = model
            .eta_param_info
            .iter()
            .find(|i| i.eta_name == "ETA_CL")
            .expect("ETA_CL must be classified");
        assert_eq!(cl_info.param_type, EtaParamType::LogNormal);
        assert_eq!(cl_info.linked_theta, Some("TVCL".to_string()));
        // mu_ref must be detected so SAEM can initialise ETA_CL at ln(TVCL).
        let mu = model.mu_refs.get("ETA_CL").expect("mu_ref for ETA_CL");
        assert_eq!(mu.theta_name, "TVCL");
        assert!(mu.log_transformed);
    }

    /// `KTR = 4.0 / TVMTT; KA = KTR * exp(ETA_KA)` — the base is a derived
    /// intermediate, not a raw theta. ETA_KA should still be LogNormal (CV%
    /// can be computed from the omega). linked_theta is None because no direct
    /// theta anchor is visible in the KA expression.
    #[test]
    fn test_classify_lognormal_derived_base() {
        use crate::types::EtaParamType;
        // Reuse minimal_model_with_indiv which has TVCL and TVV as thetas.
        let model = minimal_model_with_indiv(
            "  KTR = 4.0 / TVCL\n  V = TVV * exp(ETA_V)\n  CL = KTR * exp(ETA_CL)",
        );
        let cl_info = model
            .eta_param_info
            .iter()
            .find(|i| i.eta_name == "ETA_CL")
            .expect("ETA_CL must be classified");
        assert_eq!(
            cl_info.param_type,
            EtaParamType::LogNormal,
            "derived base should still yield LogNormal, not Custom"
        );
        // No direct theta is visible in `KTR * exp(ETA_CL)`, so linked_theta is None.
        assert!(cl_info.linked_theta.is_none());
    }

    /// Confirms that the covariate-product form `TVCL * (WT/70)^0.75 * exp(ETA_CL)`
    /// (ferx-r #54) is already classified as LogNormal with linked_theta = TVCL.
    #[test]
    fn test_classify_covariate_product() {
        use crate::types::EtaParamType;
        let src = r"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma EPS ~ 0.01
[individual_parameters]
  CL = TVCL * (WT / 70)^0.75 * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(EPS)
";
        let model = super::parse_full_model(src).unwrap().model;
        let cl_info = model
            .eta_param_info
            .iter()
            .find(|i| i.eta_name == "ETA_CL")
            .expect("ETA_CL must be classified");
        assert_eq!(cl_info.param_type, EtaParamType::LogNormal);
        assert_eq!(cl_info.linked_theta, Some("TVCL".to_string()));
    }

    #[test]
    fn test_classify_lognormal_log_scale() {
        use crate::types::{EtaParamType, ThetaTransform};
        // CL = exp(TVCL + ETA_CL)  — theta is on log scale
        let model = minimal_model_with_indiv("  CL = exp(TVCL + ETA_CL)\n  V = TVV * exp(ETA_V)");
        let cl_info = model
            .eta_param_info
            .iter()
            .find(|i| i.eta_name == "ETA_CL")
            .unwrap();
        assert_eq!(cl_info.param_type, EtaParamType::LogNormal);
        assert_eq!(cl_info.linked_theta, Some("TVCL".to_string()));
        assert_eq!(model.theta_transform[0], ThetaTransform::Log);
    }

    #[test]
    fn test_classify_additive() {
        use crate::types::{EtaParamType, ThetaTransform};
        // CL = TVCL + ETA_CL  — additive
        let model = minimal_model_with_indiv("  CL = TVCL + ETA_CL\n  V = TVV * exp(ETA_V)");
        let cl_info = model
            .eta_param_info
            .iter()
            .find(|i| i.eta_name == "ETA_CL")
            .unwrap();
        assert_eq!(cl_info.param_type, EtaParamType::Additive);
        assert_eq!(model.theta_transform[0], ThetaTransform::Identity);
    }

    #[test]
    fn test_classify_logit_scale() {
        // inv_logit(THETA_F + ETA_F) — THETA_F on logit scale
        use crate::types::{EtaParamType, ThetaTransform};
        let model = minimal_logit_model();
        let f_info = model
            .eta_param_info
            .iter()
            .find(|i| i.eta_name == "ETA_F")
            .unwrap();
        assert_eq!(f_info.param_type, EtaParamType::Logit);
        assert_eq!(f_info.linked_theta, Some("THETA_F".to_string()));
        assert_eq!(model.theta_transform[0], ThetaTransform::Logit);
    }

    #[test]
    fn test_classify_logit_probability_scale() {
        // inv_logit(logit(THETA_F) + ETA_F) — THETA_F on probability scale (0,1)
        use crate::types::{EtaParamType, ThetaTransform};
        let model_str = r"
[parameters]
  theta THETA_F(0.70, 0.001, 0.999)
  sigma EPS ~ 0.01
  omega ETA_F  ~ 0.1

[individual_parameters]
  F = inv_logit(logit(THETA_F) + ETA_F)

[structural_model]
  pk one_cpt_iv(cl=1, v=1)

[error_model]
  DV ~ proportional(EPS)
";
        let model = super::parse_full_model(model_str).unwrap().model;
        let f_info = model
            .eta_param_info
            .iter()
            .find(|i| i.eta_name == "ETA_F")
            .unwrap();
        assert_eq!(f_info.param_type, EtaParamType::LogitProbability);
        assert_eq!(f_info.linked_theta, Some("THETA_F".to_string()));
        assert_eq!(model.theta_transform[0], ThetaTransform::LogitProbability);
    }

    #[test]
    fn test_inv_logit_logit_theta_evaluates() {
        // inv_logit(logit(0.70) + 0) should equal 0.70
        let model_str = r"
[parameters]
  theta THETA_F(0.70, 0.001, 0.999)
  sigma EPS ~ 0.01
  omega ETA_F  ~ 0.1

[individual_parameters]
  F = inv_logit(logit(THETA_F) + ETA_F)

[structural_model]
  pk one_cpt_iv(cl=1, v=1)

[error_model]
  DV ~ proportional(EPS)
";
        let model = super::parse_full_model(model_str).unwrap().model;
        let tv = model.tv_fn.as_ref().unwrap()(&[0.70], &Default::default());
        assert!(
            (tv[0] - 0.70).abs() < 1e-10,
            "inv_logit(logit(0.70)) should be 0.70, got {}",
            tv[0]
        );
    }

    #[test]
    fn test_sigma_types_proportional() {
        use crate::types::SigmaType;
        let model = minimal_logit_model();
        // sigma_types is on FitResult, not CompiledModel — verify via ErrorModel::sigma_types().
        assert_eq!(
            model.error_model.sigma_types(),
            vec![SigmaType::Proportional]
        );
    }

    // ── Per-CMT (multi-endpoint) error models (issue #14) ───────────────────

    /// Minimal 2-endpoint ODE PK/PD model: PK readout at CMT=1 (proportional),
    /// PD readout at CMT=2 (additive). `error_block` overrides the
    /// `[error_model]` body so negative tests can vary just that block.
    /// Parse, expecting failure; returns the error string (ParsedModel isn't Debug).
    fn expect_parse_err(model_str: &str) -> String {
        match parse_full_model(model_str) {
            Ok(_) => panic!("expected parse error, got Ok"),
            Err(e) => e,
        }
    }

    fn pkpd_model_str(error_block: &str) -> String {
        format!(
            r"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR_PK ~ 0.10 (sd)
  sigma ADD_ERR_PD  ~ 1.00 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(states=[central, effect])

[odes]
  d/dt(central) = -CL/V * central
  d/dt(effect)  =  central/V - effect

[scaling]
  y[CMT=1] = central / V
  y[CMT=2] = effect

[error_model]
{}
",
            error_block
        )
    }

    #[test]
    fn test_per_cmt_error_model_parses() {
        let model = parse_full_model(&pkpd_model_str(
            "  CMT=1: DV ~ proportional(PROP_ERR_PK)\n  CMT=2: DV ~ additive(ADD_ERR_PD)",
        ))
        .unwrap()
        .model;

        match &model.error_spec {
            ErrorSpec::PerCmt(map) => {
                assert_eq!(map.len(), 2);
                // CMT=1 → proportional on the first declared sigma (idx 0).
                let pk = map.get(&1).expect("CMT=1 endpoint present");
                assert_eq!(pk.error_model, ErrorModel::Proportional);
                assert_eq!(pk.sigma_idx, vec![0]);
                // CMT=2 → additive on the second declared sigma (idx 1).
                let pd = map.get(&2).expect("CMT=2 endpoint present");
                assert_eq!(pd.error_model, ErrorModel::Additive);
                assert_eq!(pd.sigma_idx, vec![1]);
            }
            other => panic!("expected PerCmt, got {:?}", other),
        }
    }

    #[test]
    fn test_single_error_model_stays_single() {
        // A plain (unprefixed) line must still yield ErrorSpec::Single.
        let model = parse_full_model(&pkpd_model_str("  DV ~ proportional(PROP_ERR_PK)"))
            .unwrap()
            .model;
        assert!(matches!(
            model.error_spec,
            ErrorSpec::Single(ErrorModel::Proportional)
        ));
    }

    #[test]
    fn test_per_cmt_error_unknown_sigma_rejected() {
        let err = expect_parse_err(&pkpd_model_str(
            "  CMT=1: DV ~ proportional(NOPE)\n  CMT=2: DV ~ additive(ADD_ERR_PD)",
        ));
        assert!(err.contains("unknown sigma"), "got: {err}");
    }

    #[test]
    fn test_per_cmt_error_duplicate_cmt_rejected() {
        let err = expect_parse_err(&pkpd_model_str(
            "  CMT=1: DV ~ proportional(PROP_ERR_PK)\n  CMT=1: DV ~ additive(ADD_ERR_PD)",
        ));
        assert!(err.contains("CMT=1"), "got: {err}");
    }

    #[test]
    fn test_per_cmt_error_mixed_styles_rejected() {
        let err = expect_parse_err(&pkpd_model_str(
            "  DV ~ proportional(PROP_ERR_PK)\n  CMT=2: DV ~ additive(ADD_ERR_PD)",
        ));
        assert!(err.contains("mixes"), "got: {err}");
    }

    /// A `combined` (two-sigma) endpoint resolves both sigma indices, and they
    /// can interleave with other endpoints' sigmas in [parameters] order.
    fn pkpd_3sigma_model_str(error_block: &str) -> String {
        format!(
            r"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma S_PROP ~ 0.10 (sd)
  sigma S_ADD  ~ 1.00 (sd)
  sigma S_PD   ~ 0.50 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(states=[central, effect])

[odes]
  d/dt(central) = -CL/V * central
  d/dt(effect)  =  central/V - effect

[scaling]
  y[CMT=1] = central / V
  y[CMT=2] = effect

[error_model]
{}
",
            error_block
        )
    }

    #[test]
    fn test_per_cmt_combined_endpoint_resolves_two_sigma_indices() {
        let model = parse_full_model(&pkpd_3sigma_model_str(
            "  CMT=1: DV ~ combined(S_PROP, S_ADD)\n  CMT=2: DV ~ additive(S_PD)",
        ))
        .unwrap()
        .model;

        match &model.error_spec {
            ErrorSpec::PerCmt(map) => {
                let c1 = map.get(&1).expect("CMT=1 present");
                assert_eq!(c1.error_model, ErrorModel::Combined);
                assert_eq!(c1.sigma_idx, vec![0, 1]); // S_PROP, S_ADD
                let c2 = map.get(&2).expect("CMT=2 present");
                assert_eq!(c2.error_model, ErrorModel::Additive);
                assert_eq!(c2.sigma_idx, vec![2]); // S_PD
            }
            other => panic!("expected PerCmt, got {other:?}"),
        }
    }

    #[test]
    fn test_per_cmt_endpoint_sigma_count_mismatch_rejected() {
        // `combined` needs two sigmas; giving one must error at parse, not
        // silently propagate NaN into the likelihood.
        let err = expect_parse_err(&pkpd_3sigma_model_str(
            "  CMT=1: DV ~ combined(S_PROP)\n  CMT=2: DV ~ additive(S_PD)",
        ));
        assert!(
            err.contains("expects 2 sigma") || err.contains("2 sigma(s)"),
            "got: {err}"
        );
    }

    #[test]
    fn test_single_error_unknown_sigma_rejected() {
        // A single (unprefixed) error line that references a sigma not declared
        // in [parameters] is a typo, not a silent bind to sigma[0].
        let model_str = r"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.10 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(NO_SUCH_SIGMA)
";
        let err = expect_parse_err(model_str);
        assert!(err.contains("unknown sigma"), "got: {err}");
    }

    #[test]
    fn test_per_cmt_error_on_analytical_model_rejected() {
        let model_str = r"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR_PK ~ 0.10 (sd)
  sigma ADD_ERR_PD  ~ 1.00 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  CMT=1: DV ~ proportional(PROP_ERR_PK)
  CMT=2: DV ~ additive(ADD_ERR_PD)
";
        let err = expect_parse_err(model_str);
        assert!(err.contains("ODE"), "got: {err}");
    }

    // ── Issue 3: if/else classification ─────────────────────────────────────

    #[test]
    fn test_classify_if_else_unanimous_lognormal() {
        // Both branches use TVCL * exp(ETA_CL) — should classify as LogNormal.
        use crate::types::EtaParamType;
        let model = minimal_model_with_indiv(
            "  if (TVCL > 1) {\n    CL = TVCL * exp(ETA_CL)\n  } else {\n    CL = TVCL * exp(ETA_CL)\n  }\n  V = TVV * exp(ETA_V)",
        );
        let cl_info = model
            .eta_param_info
            .iter()
            .find(|i| i.eta_name == "ETA_CL")
            .unwrap();
        assert_eq!(cl_info.param_type, EtaParamType::LogNormal);
        assert_eq!(cl_info.individual_param_name, "CL");
    }

    #[test]
    fn test_classify_if_else_disagreement_custom() {
        // Branches use different patterns — should fall back to Custom.
        use crate::types::EtaParamType;
        let model = minimal_model_with_indiv(
            "  if (TVCL > 1) {\n    CL = TVCL * exp(ETA_CL)\n  } else {\n    CL = TVCL + ETA_CL\n  }\n  V = TVV * exp(ETA_V)",
        );
        let cl_info = model
            .eta_param_info
            .iter()
            .find(|i| i.eta_name == "ETA_CL")
            .unwrap();
        assert_eq!(cl_info.param_type, EtaParamType::Custom);
    }

    #[test]
    fn test_classify_if_no_else_skipped() {
        // No else arm → partially defined → classification skipped entirely for V.
        let model = minimal_model_with_indiv(
            "  CL = TVCL * exp(ETA_CL)\n  if (TVV > 1) {\n    V = TVV * exp(ETA_V)\n  }",
        );
        // CL should be classified; V (if-only, no else) should be absent.
        assert!(model.eta_param_info.iter().any(|i| i.eta_name == "ETA_CL"));
        assert!(!model.eta_param_info.iter().any(|i| i.eta_name == "ETA_V"));
    }

    #[test]
    fn test_classify_multi_eta_custom() {
        // Expression with two ETAs (unusual) — both get their own Custom entry.
        use crate::types::EtaParamType;
        let model = minimal_model_with_indiv("  CL = TVCL + ETA_CL + ETA_V\n  V = 10.0");
        let customs: Vec<_> = model
            .eta_param_info
            .iter()
            .filter(|i| i.param_type == EtaParamType::Custom)
            .collect();
        assert_eq!(
            customs.len(),
            2,
            "both ETAs in the expression should be Custom"
        );
    }

    #[test]
    fn test_lagtime_in_structural_model_block() {
        // The DSL key `lagtime=LAGTIME` on the structural_model line must
        // route LAGTIME's value into PK_IDX_LAGTIME (8). Verifies the
        // parser → name_to_index → PkParams pipeline end-to-end.
        let model_str = "
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta TVKA(2.0)
  theta TVLAGTIME(0.5)
  omega ETA_CL ~ 0.1
  omega ETA_V  ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL      = TVCL * exp(ETA_CL)
  V       = TVV  * exp(ETA_V)
  KA      = TVKA
  LAGTIME = TVLAGTIME

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, lagtime=LAGTIME)

[error_model]
  DV ~ proportional(EPS)
";
        let parsed = super::parse_full_model(model_str).unwrap();
        let pk_indices = &parsed.model.pk_indices;
        // LAGTIME should map to slot 8.
        assert!(
            pk_indices.contains(&crate::types::PK_IDX_LAGTIME),
            "pk_indices missing PK_IDX_LAGTIME: {:?}",
            pk_indices
        );

        // Evaluate pk_param_fn with default theta to confirm the value
        // flows through to the slot.
        let theta: Vec<f64> = parsed.model.default_params.theta.clone();
        let eta: Vec<f64> = vec![0.0; parsed.model.n_eta];
        let pk = (parsed.model.pk_param_fn)(&theta, &eta, &std::collections::HashMap::new());
        assert_eq!(pk.lagtime(), 0.5);
    }

    #[test]
    fn test_alag_alias_in_structural_model_block() {
        // For NONMEM-user familiarity, `alag=` is accepted as an alias
        // for `lagtime=`. Same target slot (PK_IDX_LAGTIME).
        let model_str = "
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta TVKA(2.0)
  theta TVALAG(0.75)
  omega ETA_CL ~ 0.1
  omega ETA_V  ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV  * exp(ETA_V)
  KA   = TVKA
  ALAG = TVALAG

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, alag=ALAG)

[error_model]
  DV ~ proportional(EPS)
";
        let parsed = super::parse_full_model(model_str).unwrap();
        assert!(parsed
            .model
            .pk_indices
            .contains(&crate::types::PK_IDX_LAGTIME));

        let theta: Vec<f64> = parsed.model.default_params.theta.clone();
        let eta: Vec<f64> = vec![0.0; parsed.model.n_eta];
        let pk = (parsed.model.pk_param_fn)(&theta, &eta, &std::collections::HashMap::new());
        assert_eq!(pk.lagtime(), 0.75);
    }

    #[test]
    fn test_undefined_structural_param_errors() {
        // #261: a [structural_model] PK value that names a variable not defined
        // in [individual_parameters] must be a hard parse error, not silently
        // defaulted to 0.0 (which yields a "converged" but structurally broken
        // fit). Here `cl=CL` but only V, KE, KA are defined.
        let model_str = "
[parameters]
  theta TVV(10.0, 0.1, 1000.0)
  theta TVKE(0.1, 0.001, 10.0)
  theta TVKA(1.0, 0.01, 100.0)
  omega ETA_V ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  V  = TVV * exp(ETA_V)
  KE = TVKE
  KA = TVKA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(EPS)
";
        let err = expect_parse_err(model_str);
        assert!(
            err.contains("CL") && err.contains("not defined"),
            "expected an undefined-variable error naming CL, got: {err}"
        );
        // The message should list the defined parameters to make the fix obvious.
        assert!(
            err.contains("KE"),
            "error should list defined params: {err}"
        );
    }

    #[test]
    fn test_unknown_pk_param_key_errors() {
        // Audit of the same silent-drop pattern on the key side: an unrecognized
        // PK-parameter name (`clx` here, a typo for `cl`) previously dropped the
        // binding, leaving cl at its 0.0 default. It must now error.
        let model_str = "
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(clx=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";
        let err = expect_parse_err(model_str);
        assert!(
            err.contains("clx") && err.contains("unknown PK parameter"),
            "expected an unknown-PK-parameter error naming clx, got: {err}"
        );
    }

    #[test]
    fn test_literal_pk_param_binds_constant() {
        // A numeric literal value (`ka=2.5`) is bound as a constant rather than
        // silently dropped to 0.0. Companion to #261: the same filter_map that
        // dropped undefined references also dropped literals. `cl=CL` (a defined
        // variable) must still bind, confirming the two paths coexist.
        let model_str = "
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=2.5)

[error_model]
  DV ~ proportional(EPS)
";
        let parsed = super::parse_full_model(model_str).unwrap();
        let theta: Vec<f64> = parsed.model.default_params.theta.clone();
        let eta: Vec<f64> = vec![0.0; parsed.model.n_eta];
        let pk = (parsed.model.pk_param_fn)(&theta, &eta, &std::collections::HashMap::new());
        assert_eq!(pk.ka(), 2.5, "literal ka should bind as the constant 2.5");
        assert!(
            pk.cl() > 0.0,
            "cl should still bind to the defined variable, got {}",
            pk.cl()
        );
    }

    #[test]
    fn test_valid_structural_param_still_parses() {
        // #261 acceptance: the repro fixed by defining CL (= KE * V) parses
        // without error and binds cl to a positive value.
        let model_str = "
[parameters]
  theta TVV(10.0, 0.1, 1000.0)
  theta TVKE(0.1, 0.001, 10.0)
  theta TVKA(1.0, 0.01, 100.0)
  omega ETA_V ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  V  = TVV * exp(ETA_V)
  KE = TVKE
  KA = TVKA
  CL = KE * V

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(EPS)
";
        let parsed =
            super::parse_full_model(model_str).expect("model with CL defined should parse");
        let theta: Vec<f64> = parsed.model.default_params.theta.clone();
        let eta: Vec<f64> = vec![0.0; parsed.model.n_eta];
        let pk = (parsed.model.pk_param_fn)(&theta, &eta, &std::collections::HashMap::new());
        assert!(pk.cl() > 0.0, "cl should bind to KE * V, got {}", pk.cl());
    }

    #[test]
    fn test_non_finite_literal_pk_param_errors() {
        // `f64::from_str` accepts "inf"/"nan", so a non-finite literal value
        // must be rejected rather than silently bound as a degenerate constant
        // (the same silent-wrong default #261 removes for undefined references).
        let model_str = "
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=inf)

[error_model]
  DV ~ proportional(EPS)
";
        let err = expect_parse_err(model_str);
        assert!(
            err.contains("non-finite") && err.contains("ka"),
            "expected a non-finite-constant error naming ka, got: {err}"
        );
    }

    #[test]
    fn test_undefined_optional_pk_param_errors() {
        // The undefined-reference guard (#261/#308) applies uniformly to the
        // *optional* params too — `f`, `lagtime`, and the `alag` alias — not just
        // the required ones. A typo'd / undefined optional reference must error,
        // never silently default the slot. (#309 keeps `f`/`lagtime` *optional*,
        // i.e. omitting them is fine; this is the orthogonal value-validation:
        // if you DO map them, the referenced variable must exist.) All required
        // params (cl/v/ka) are defined here so the model reaches value resolution.
        let header = "
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta TVKA(1.0, 0.01, 100.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, ";
        let footer = ")

[error_model]
  DV ~ proportional(EPS)
";
        // (pk key, undefined variable) — one per optional slot, incl. the alias.
        for (key, badvar) in [("f", "BADF"), ("lagtime", "BADLAG"), ("alag", "BADALAG")] {
            let model_str = format!("{header}{key}={badvar}{footer}");
            let err = expect_parse_err(&model_str);
            assert!(
                err.contains(badvar) && err.contains("not defined"),
                "optional `{key}={badvar}` must error as an undefined reference, got: {err}"
            );
        }
    }

    #[test]
    fn test_unknown_key_precedes_missing_required() {
        // Precedence: when a key is unrecognized (a typo like `vx` for `v`) the
        // error must name that bad key (#308), not report the now-unmapped slot
        // as a missing required param (#309). The required-param check defers to
        // the unknown-key check so the message points at the actual mistake.
        let model_str = "
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta TVKA(1.0, 0.01, 100.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA

[structural_model]
  pk one_cpt_oral(cl=CL, vx=V, ka=KA)

[error_model]
  DV ~ proportional(EPS)
";
        let err = expect_parse_err(model_str);
        assert!(
            err.contains("vx") && err.contains("unknown PK parameter"),
            "unknown key `vx` must be reported, not a missing-required error, got: {err}"
        );
    }

    #[test]
    fn test_lagtime_in_ode_model_routes_to_canonical_slot() {
        // Regression for the ODE-with-lagtime path. For ODE models there is
        // no [structural_model] pk= line, so pk_param_map is empty and
        // pk_param_fn's ODE branch writes individual parameters by
        // declaration order. LAGTIME (and ALAG) must also land at the
        // canonical PK_IDX_LAGTIME slot so `ode_predictions` (which reads
        // `pk_params_flat[PK_IDX_LAGTIME]`) sees it. `has_lagtime()` must
        // likewise return true via the indiv_param_names fallback so the
        // SS/negative-lagtime warning gating fires for ODE users.
        let model_str = "
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta TVLAGTIME(0.5)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL      = TVCL * exp(ETA_CL)
  V       = TVV
  LAGTIME = TVLAGTIME

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -CL/V * central

[error_model]
  DV ~ proportional(EPS)
";
        let parsed = super::parse_full_model(model_str).unwrap();
        // ODE models must report has_lagtime() via the indiv_param_names
        // fallback even when pk_indices doesn't contain PK_IDX_LAGTIME.
        assert!(
            parsed.model.has_lagtime(),
            "has_lagtime() must return true for an ODE model declaring LAGTIME"
        );

        let theta: Vec<f64> = parsed.model.default_params.theta.clone();
        let eta: Vec<f64> = vec![0.0; parsed.model.n_eta];
        let pk = (parsed.model.pk_param_fn)(&theta, &eta, &std::collections::HashMap::new());
        assert_eq!(
            pk.lagtime(),
            0.5,
            "LAGTIME must be routed to PK_IDX_LAGTIME for ODE models"
        );
    }

    #[test]
    fn test_bioavailability_in_ode_model_routes_to_canonical_slot() {
        // Issue #122: an ODE model that declares an `F` individual parameter
        // must route its value to the canonical PK_IDX_F slot so the ODE
        // engine (`ode_predictions`, which reads `pk_params_flat[PK_IDX_F]`)
        // loads the dosing compartment with F·AMT — NONMEM's convention.
        // F is declared third here; `ode_param_slots` maps the name `F` to
        // PK_IDX_F (5) regardless of declaration position.
        let model_str = "
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta TVF(0.7, 0.001, 0.999)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  F  = TVF

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -1.0 * depot
  d/dt(central) = depot - CL/V * central

[error_model]
  DV ~ proportional(EPS)
";
        let parsed = super::parse_full_model(model_str).unwrap();
        let theta: Vec<f64> = parsed.model.default_params.theta.clone();
        let eta: Vec<f64> = vec![0.0; parsed.model.n_eta];
        let pk = (parsed.model.pk_param_fn)(&theta, &eta, &std::collections::HashMap::new());
        assert_eq!(
            pk.f_bio(),
            0.7,
            "F must be routed to PK_IDX_F for ODE models"
        );
    }

    #[test]
    fn test_ode_structural_param_does_not_alias_bioavailability_slot() {
        // Issue #122 regression: an ODE model with ≥6 individual parameters and
        // NO F declared must NOT let a structural parameter land in PK_IDX_F
        // (slot 5) and be silently read as bioavailability. Before the
        // `ode_param_slots` fix, the 6th-declared param (here KE0=7.0) was
        // written positionally into slot 5, so `f_bio()` returned 7.0 and the
        // engine scaled every dose by 7×. With the canonical slot map, F is
        // reserved and undeclared → defaults to 1.0.
        let model_str = "
[parameters]
  theta TCL(1.0)
  theta TV(10.0)
  theta TQ(2.0)
  theta TV2(20.0)
  theta TKA(1.5)
  theta TKE0(7.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL  = TCL * exp(ETA_CL)
  V   = TV
  Q   = TQ
  V2  = TV2
  KA  = TKA
  KE0 = TKE0

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) = KA * depot - CL/V * central

[error_model]
  DV ~ proportional(EPS)
";
        let parsed = super::parse_full_model(model_str).unwrap();
        let theta: Vec<f64> = parsed.model.default_params.theta.clone();
        let eta: Vec<f64> = vec![0.0; parsed.model.n_eta];
        let pk = (parsed.model.pk_param_fn)(&theta, &eta, &std::collections::HashMap::new());
        assert_eq!(
            pk.f_bio(),
            1.0,
            "undeclared F must default to 1.0, not alias a structural parameter"
        );
        assert_eq!(pk.lagtime(), 0.0, "undeclared lagtime must default to 0.0");

        // The structural parameter KE0 must still round-trip: ode_param_slots
        // assigns it a free, non-reserved slot, pk_param_fn writes its value
        // there, and the RHS reads it back from the same slot. Assert it lands
        // off the engine-reserved F/lagtime slots and carries its value (7.0).
        let ke0_pos = parsed
            .model
            .indiv_param_names
            .iter()
            .position(|n| n == "KE0")
            .expect("KE0 declared");
        let ke0_slot = parsed.model.pk_indices[ke0_pos];
        assert_ne!(
            ke0_slot,
            crate::types::PK_IDX_F,
            "KE0 must not alias F slot"
        );
        assert_ne!(
            ke0_slot,
            crate::types::PK_IDX_LAGTIME,
            "KE0 must not alias lagtime slot"
        );
        assert_eq!(
            pk.values[ke0_slot], 7.0,
            "KE0 value must round-trip to its assigned slot"
        );
    }

    // ── [diffusion] block parsing ─────────────────────────────────────────

    fn minimal_ode_model_with_diffusion(diffusion_lines: &str) -> String {
        format!(
            r#"
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma ADD ~ 0.1

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -(CL/V) * central

[diffusion]
{diffusion_lines}

[error_model]
  DV ~ additive(ADD)

[fit_options]
  method = foce
"#,
            diffusion_lines = diffusion_lines
        )
    }

    // ── [odes] init(state) = expr ─────────────────────────────────────────

    /// Turnover model `d/dt(response) = KIN - KOUT*response` with a baseline
    /// initial condition. `init_lines` is spliced into the `[odes]` block.
    fn turnover_ode_model(init_lines: &str) -> String {
        format!(
            r#"
[parameters]
  theta TVKIN(10.0, 0.001, 100.0)
  theta TVKOUT(2.0, 0.001, 100.0)
  omega ETA_KIN ~ 0.09
  sigma ADD ~ 0.1

[individual_parameters]
  KIN  = TVKIN * exp(ETA_KIN)
  KOUT = TVKOUT

[structural_model]
  ode(obs_cmt=response, states=[response])

[odes]
{init_lines}
  d/dt(response) = KIN - KOUT * response

[error_model]
  DV ~ additive(ADD)

[fit_options]
  method = foce
"#,
            init_lines = init_lines
        )
    }

    #[test]
    fn test_init_directive_builds_init_fn() {
        let src = turnover_ode_model("  init(response) = KIN / KOUT");
        let parsed = parse_full_model(&src).unwrap();
        let ode = parsed.model.ode_spec.as_ref().expect("ODE spec");
        assert!(ode.init_fn.is_some(), "init_fn should be populated");

        // Evaluate at typical values (eta = 0): KIN = 10, KOUT = 2 → 5.0.
        // For an ODE model, individual params occupy PkParams slots in
        // declaration order: KIN @ 0, KOUT @ 1.
        let mut params = [0.0; crate::types::MAX_PK_PARAMS];
        params[0] = 10.0;
        params[1] = 2.0;
        let u0 = ode.initial_state(&params);
        assert_eq!(u0.len(), 1);
        assert!(
            (u0[0] - 5.0).abs() < 1e-9,
            "init(response) = KIN/KOUT = 5, got {}",
            u0[0]
        );
    }

    #[test]
    fn test_no_init_directive_seeds_zero() {
        let src = turnover_ode_model("");
        let parsed = parse_full_model(&src).unwrap();
        let ode = parsed.model.ode_spec.as_ref().unwrap();
        assert!(ode.init_fn.is_none());
        // initial_state falls back to zeros.
        assert_eq!(ode.initial_state(&[10.0, 2.0]), vec![0.0]);
    }

    #[test]
    fn test_init_unknown_state_errors() {
        let src = turnover_ode_model("  init(nonexistent) = 1.0");
        let err = match parse_full_model(&src) {
            Err(e) => e,
            Ok(_) => panic!("expected unknown-state error"),
        };
        assert!(
            err.contains("init(nonexistent)") && err.contains("unknown state"),
            "expected unknown-state error, got: {}",
            err
        );
    }

    #[test]
    fn test_duplicate_init_errors() {
        let src = turnover_ode_model("  init(response) = KIN / KOUT\n  init(response) = 0.0");
        let err = match parse_full_model(&src) {
            Err(e) => e,
            Ok(_) => panic!("expected duplicate-init error"),
        };
        assert!(
            err.contains("duplicate init(response)"),
            "expected duplicate-init error, got: {}",
            err
        );
    }

    #[test]
    fn test_init_literal_value() {
        let src = turnover_ode_model("  init(response) = 7.5");
        let parsed = parse_full_model(&src).unwrap();
        let ode = parsed.model.ode_spec.as_ref().unwrap();
        assert_eq!(ode.initial_state(&[10.0, 2.0]), vec![7.5]);
    }

    #[test]
    fn test_init_undefined_name_errors() {
        // #314: an init expression referencing a name that is neither a state
        // nor an individual parameter must error rather than silently reading
        // 0.0 via `eval_expression`. `KIN` is a declared parameter (must not be
        // flagged); `BASE` is undeclared.
        let src = turnover_ode_model("  init(response) = BASE * KIN");
        let err = match parse_full_model(&src) {
            Err(e) => e,
            Ok(_) => panic!("expected undefined-name error for init(response)"),
        };
        assert!(
            err.contains("init(response)"),
            "error should reference init(response), got: {err}"
        );
        // Only BASE is flagged — KIN resolves to the declared parameter.
        assert!(
            err.contains("undefined name(s): BASE."),
            "error should flag only BASE as undefined, got: {err}"
        );
    }

    #[test]
    fn test_init_macheps_resolves_to_epsilon() {
        // MACHEPS is available in init expressions (eval_expression resolves it)
        // — the undefined-name check must accept it, and it evaluates to EPSILON.
        let src = turnover_ode_model("  init(response) = MACHEPS");
        let parsed = parse_full_model(&src).unwrap();
        let ode = parsed.model.ode_spec.as_ref().unwrap();
        assert_eq!(ode.initial_state(&[10.0, 2.0]), vec![f64::EPSILON]);
    }

    #[test]
    fn test_init_macheps_is_case_insensitive() {
        // `eval_expression` resolves MACHEPS case-insensitively, so a mixed-case
        // spelling must not be rejected by the undefined-name check (regression:
        // exact-key matching against init_defined would have flagged it).
        let src = turnover_ode_model("  init(response) = MachEps");
        let parsed = parse_full_model(&src).unwrap();
        let ode = parsed.model.ode_spec.as_ref().unwrap();
        assert_eq!(ode.initial_state(&[10.0, 2.0]), vec![f64::EPSILON]);
    }

    #[test]
    fn test_diffusion_block_parsed_into_theta() {
        let src = minimal_ode_model_with_diffusion("  central ~ 0.05");
        let parsed = parse_full_model(&src).unwrap();
        let m = &parsed.model;
        // DIFF_CENTRAL should be appended as the last theta
        assert!(
            m.theta_names.iter().any(|n| n == "DIFF_CENTRAL"),
            "expected DIFF_CENTRAL in theta_names, got {:?}",
            m.theta_names
        );
        let idx = m
            .theta_names
            .iter()
            .position(|n| n == "DIFF_CENTRAL")
            .unwrap();
        assert!(
            (m.default_params.theta[idx] - 0.05).abs() < 1e-9,
            "initial diffusion variance should be 0.05"
        );
    }

    #[test]
    fn test_diffusion_block_sets_diffusion_theta_start() {
        let src = minimal_ode_model_with_diffusion("  central ~ 0.01");
        let parsed = parse_full_model(&src).unwrap();
        let m = &parsed.model;
        assert!(
            m.diffusion_theta_start.is_some(),
            "diffusion_theta_start must be set"
        );
        assert_eq!(m.diffusion_state_indices.len(), 1);
        assert!(m.is_sde());
    }

    #[test]
    fn test_diffusion_block_fix_flag() {
        let src = minimal_ode_model_with_diffusion("  central ~ 0.02 FIX");
        let parsed = parse_full_model(&src).unwrap();
        let m = &parsed.model;
        let idx = m
            .theta_names
            .iter()
            .position(|n| n == "DIFF_CENTRAL")
            .unwrap();
        assert!(
            m.default_params.theta_fixed[idx],
            "DIFF_CENTRAL should be FIX"
        );
    }

    #[test]
    fn test_diffusion_block_unknown_state_error() {
        let src = minimal_ode_model_with_diffusion("  depot ~ 0.01");
        assert!(
            parse_full_model(&src).is_err(),
            "unknown state in [diffusion] must be an error"
        );
    }

    #[test]
    fn test_diffusion_on_analytical_model_is_error() {
        let src = r#"
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma ADD ~ 0.1
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[diffusion]
  central ~ 0.01
[error_model]
  additive
"#;
        assert!(
            parse_full_model(src).is_err(),
            "[diffusion] on an analytical model must be an error"
        );
    }

    // ─── [covariate_nn NAME] block parsing ──────────────────────────────

    /// Minimal full model with a `[covariate_nn]` block. Re-used across tests.
    /// The base [parameters]/[individual_parameters] still use the analytical
    /// form — this PR only registers the NN-weight thetas and stores the
    /// mapper handle; it does not yet route PK params through the NN.
    fn covariate_nn_model_src(nn_block: &str) -> String {
        format!(
            r#"
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma ADD ~ 0.1

{nn_block}

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ additive(ADD)
"#
        )
    }

    #[cfg(feature = "nn")]
    #[test]
    fn test_covariate_nn_block_parses_and_appends_thetas() {
        let src = covariate_nn_model_src(
            "[covariate_nn TYPICAL_PK]\n  inputs = [WT, CRCL]\n  outputs = [CL, V]\n  layers = [4]\n  activation = tanh\n  output = softplus\n",
        );
        let model = parse_model_string(&src).expect("model must parse with --features nn");

        // Mapper present and named correctly.
        assert_eq!(model.covariate_nns.len(), 1);
        let nn = &model.covariate_nns[0];
        assert_eq!(nn.name, "TYPICAL_PK");
        // 2 inputs -> 4 hidden -> 2 outputs:
        // W_1: 4*2 = 8, b_1: 4, W_2: 2*4 = 8, b_2: 2 → total 22 weights.
        let n_weights = 4 * 2 + 4 + 2 * 4 + 2;
        use crate::nn::CovariateMapper;
        assert_eq!(nn.mapper.n_weights(), n_weights);

        // Auto-generated thetas land at the tail of the theta vector.
        // Base thetas: TVCL, TVV. So weights_offset == 2.
        assert_eq!(nn.weights_offset, 2);
        assert_eq!(model.n_theta, 2 + n_weights);

        // Theta-name convention is W_<NAME>_<l>_<i>_<j> / B_<NAME>_<l>_<i>.
        let names: &[String] = &model.theta_names;
        assert_eq!(names[0], "TVCL");
        assert_eq!(names[1], "TVV");
        assert_eq!(names[2], "W_TYPICAL_PK_1_1_1");
        // Last name is the last bias of layer 2 (output layer, 2 outputs).
        assert_eq!(names[names.len() - 1], "B_TYPICAL_PK_2_2");
    }

    #[cfg(feature = "nn")]
    #[test]
    fn test_covariate_nn_block_rejects_unknown_pk_output() {
        let src = covariate_nn_model_src(
            "[covariate_nn FOO]\n  inputs = [WT]\n  outputs = [NOT_A_PK_PARAM]\n  layers = [3]\n  activation = relu\n",
        );
        let err = parse_model_string(&src).expect_err("unknown PK output must error");
        assert!(
            err.contains("NOT_A_PK_PARAM"),
            "error should name the bad output, got: {err}"
        );
    }

    #[cfg(feature = "nn")]
    #[test]
    fn test_covariate_nn_block_rejects_unknown_activation() {
        let src = covariate_nn_model_src(
            "[covariate_nn FOO]\n  inputs = [WT]\n  outputs = [CL]\n  layers = [3]\n  activation = quack\n",
        );
        let err = parse_model_string(&src).expect_err("bad activation must error");
        assert!(
            err.contains("quack"),
            "error should name the bad activation, got: {err}"
        );
    }

    #[cfg(feature = "nn")]
    #[test]
    fn test_covariate_nn_block_rejects_unknown_key() {
        let src = covariate_nn_model_src(
            "[covariate_nn FOO]\n  inputs = [WT]\n  outputs = [CL]\n  layers = [3]\n  activation = relu\n  not_a_real_key = 42\n",
        );
        let err = parse_model_string(&src).expect_err("unknown key must error");
        assert!(
            err.contains("not_a_real_key"),
            "error should name the bad key, got: {err}"
        );
    }

    #[cfg(feature = "nn")]
    #[test]
    fn test_covariate_nn_block_missing_required_field() {
        // Missing `layers`.
        let src = covariate_nn_model_src(
            "[covariate_nn FOO]\n  inputs = [WT]\n  outputs = [CL]\n  activation = relu\n",
        );
        let err = parse_model_string(&src).expect_err("missing layers must error");
        assert!(
            err.contains("layers"),
            "error should mention `layers`, got: {err}"
        );
    }

    #[cfg(feature = "nn")]
    #[test]
    fn test_multiple_covariate_nn_blocks_appear_in_sorted_order() {
        // Two blocks; the second-declared one should still come first by
        // alphabetical sort to keep theta-ordering reproducible.
        let src = covariate_nn_model_src(
            "[covariate_nn ZETA]\n  inputs = [WT]\n  outputs = [CL]\n  layers = [2]\n  activation = tanh\n\n\
             [covariate_nn ALPHA]\n  inputs = [WT]\n  outputs = [V]\n  layers = [2]\n  activation = tanh\n",
        );
        let model = parse_model_string(&src).expect("two NN blocks must parse");
        assert_eq!(model.covariate_nns.len(), 2);
        assert_eq!(model.covariate_nns[0].name, "ALPHA");
        assert_eq!(model.covariate_nns[1].name, "ZETA");
        // ALPHA's weights are first; ZETA's start where ALPHA's end.
        use crate::nn::CovariateMapper;
        let alpha_weights = model.covariate_nns[0].mapper.n_weights();
        assert_eq!(model.covariate_nns[1].weights_offset, 2 + alpha_weights);
    }

    /// When ferx-core is built without `--features nn`, a `[covariate_nn]`
    /// block must produce a clear feature-gate error rather than being
    /// silently ignored.
    #[cfg(not(feature = "nn"))]
    #[test]
    fn test_covariate_nn_block_without_nn_feature_errors() {
        let src = covariate_nn_model_src(
            "[covariate_nn FOO]\n  inputs = [WT]\n  outputs = [CL]\n  layers = [2]\n  activation = tanh\n",
        );
        let err = parse_model_string(&src).expect_err("must error without --features nn");
        assert!(
            err.contains("nn"),
            "error should mention the nn feature, got: {err}"
        );
    }

    /// Sanity for the named-block parser extension itself (independent of the
    /// NN feature). `[block_type NAME]` should be recognised and parsed.
    #[test]
    fn test_extract_blocks_recognizes_named_block_form() {
        let src = "
[parameters]
  theta T1(1.0, 0.001, 10.0)

[some_named_block FOO]
  key = value
";
        let extracted = extract_blocks(src).unwrap();
        // Unnamed block intact.
        assert!(extracted.unnamed.contains_key("parameters"));
        // Named block captured by type + instance.
        let by_inst = extracted
            .named
            .get("some_named_block")
            .expect("named block extracted");
        let lines = by_inst.get("FOO").expect("instance FOO present");
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0], "key = value");
    }

    // ─── [covariate_nn] dot-access + pk_param_fn dispatch (Phase A M1 step 3) ────

    #[cfg(feature = "nn")]
    fn covariate_nn_dotted_model_src() -> String {
        // Two-output NN feeding both CL and V. Etas on the final PK params,
        // matching the Phase A M2 mu-ref form. The fit-objective surface
        // (method = nn_mse) isn't wired yet, so we only exercise the
        // pk_param_fn closure here — that's the simulate path.
        r#"
[parameters]
  theta TVKA(1.0, 0.001, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma ADD ~ 0.1

[covariate_nn TYPICAL_PK]
  inputs = [WT, CRCL]
  outputs = [CL, V]
  layers = [4]
  activation = tanh
  output = softplus

[individual_parameters]
  CL = TYPICAL_PK.CL * exp(ETA_CL)
  V  = TYPICAL_PK.V  * exp(ETA_V)
  KA = TVKA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ additive(ADD)
"#
        .to_string()
    }

    #[cfg(feature = "nn")]
    #[test]
    fn test_dot_access_parses_into_nn_output_node() {
        let src = covariate_nn_dotted_model_src();
        let model = parse_model_string(&src).expect("dot-access must parse");
        // 1 base theta (TVKA) + NN weights for a 2->4->2 net:
        //   W_1: 4*2 = 8, b_1: 4, W_2: 2*4 = 8, b_2: 2 → 22 weights.
        assert_eq!(model.n_theta, 1 + 22);
        assert_eq!(model.covariate_nns.len(), 1);
        assert_eq!(model.covariate_nns[0].name, "TYPICAL_PK");
        // Weights start right after the user-declared base thetas.
        assert_eq!(model.covariate_nns[0].weights_offset, 1);
    }

    /// Round-trip: call the closure with known theta values (including known
    /// NN weights) and confirm the PK params it produces match what
    /// `NamedMlpMapper::forward_raw` returns directly.
    #[cfg(feature = "nn")]
    #[test]
    fn test_pk_param_fn_dispatches_through_nn() {
        use crate::nn::CovariateMapper;
        use crate::types::{PK_IDX_CL, PK_IDX_V};

        let src = covariate_nn_dotted_model_src();
        let model = parse_model_string(&src).expect("model parses");
        let theta: Vec<f64> = model.default_params.theta.clone();
        let eta = vec![0.0_f64, 0.0_f64]; // zero etas → PK params == NN typical values.
        let mut cov = HashMap::new();
        cov.insert("WT".to_string(), 70.0);
        cov.insert("CRCL".to_string(), 95.0);

        let pk = (model.pk_param_fn)(&theta, &eta, &cov);

        // What the NN itself would emit, sliced from the same theta vector.
        let nn = &model.covariate_nns[0];
        let weights = &theta[nn.weights_offset..nn.weights_offset + nn.mapper.n_weights()];
        let nn_outputs = nn.mapper.forward_raw(weights, &cov).unwrap();

        // pk_param_fn writes to PkParams by name: output[0] = CL, output[1] = V.
        assert!((pk.values[PK_IDX_CL] - nn_outputs[0]).abs() < 1e-12);
        assert!((pk.values[PK_IDX_V] - nn_outputs[1]).abs() < 1e-12);
        // KA is a plain theta-only path.
        assert!((pk.values[crate::types::PK_IDX_KA] - theta[0]).abs() < 1e-12);
    }

    /// Non-zero etas: the mu-ref composition `TYPICAL_PK.CL * exp(ETA_CL)`
    /// must apply log-normal IIV on top of the NN typical value.
    #[cfg(feature = "nn")]
    #[test]
    fn test_pk_param_fn_composes_eta_on_top_of_nn_output() {
        use crate::nn::CovariateMapper;
        use crate::types::PK_IDX_CL;

        let src = covariate_nn_dotted_model_src();
        let model = parse_model_string(&src).expect("model parses");
        let theta = model.default_params.theta.clone();
        let mut cov = HashMap::new();
        cov.insert("WT".to_string(), 70.0);
        cov.insert("CRCL".to_string(), 95.0);

        let nn = &model.covariate_nns[0];
        let weights = &theta[nn.weights_offset..nn.weights_offset + nn.mapper.n_weights()];
        let nn_outputs = nn.mapper.forward_raw(weights, &cov).unwrap();
        let tv_cl = nn_outputs[0];

        // eta = +0.3 → CL should be tv_cl * exp(0.3).
        let pk = (model.pk_param_fn)(&theta, &[0.3_f64, 0.0_f64], &cov);
        let expected = tv_cl * 0.3_f64.exp();
        assert!(
            (pk.values[PK_IDX_CL] - expected).abs() < 1e-10,
            "expected {expected}, got {}",
            pk.values[PK_IDX_CL]
        );
    }

    #[cfg(feature = "nn")]
    #[test]
    fn test_dot_access_rejects_unknown_output() {
        // GARBAGE is not in `outputs = [CL, V]`.
        let src = r#"
[parameters]
  theta TVKA(1.0, 0.001, 100.0)
  omega ETA_CL ~ 0.09
  sigma ADD ~ 0.1

[covariate_nn TYPICAL_PK]
  inputs = [WT]
  outputs = [CL, V]
  layers = [3]
  activation = tanh

[individual_parameters]
  CL = TYPICAL_PK.GARBAGE * exp(ETA_CL)
  V  = 50
  KA = TVKA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ additive(ADD)
"#;
        let err = parse_model_string(src).expect_err("unknown output must error");
        assert!(
            err.contains("GARBAGE"),
            "error should name the bad output, got: {err}"
        );
    }

    #[cfg(feature = "nn")]
    #[test]
    fn test_dot_access_on_unknown_nn_name_falls_back_to_parse_error() {
        // FOO is not a declared [covariate_nn] block. `FOO.CL` must error
        // (the parser stops at the `.` it can't classify).
        let src = r#"
[parameters]
  theta TVKA(1.0, 0.001, 100.0)
  omega ETA_CL ~ 0.09
  sigma ADD ~ 0.1

[individual_parameters]
  CL = FOO.CL
  V  = 50
  KA = TVKA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ additive(ADD)
"#;
        assert!(parse_model_string(src).is_err());
    }

    // ─── Fix 1 + fix 2: NN-anchored mu-ref detection + tv_fn parity ─────

    /// Fix 1: `TYPICAL_PK.CL * exp(ETA_CL)` should be recognised as a
    /// lognormal mu-ref, with `MuRef.theta_name = "TYPICAL_PK.CL"` (the
    /// structured form). Downstream consumers that only know about plain
    /// thetas silently fall back to FD; the M2 AD-aware consumer is a
    /// follow-up.
    #[cfg(feature = "nn")]
    #[test]
    fn test_detect_mu_refs_recognises_nn_anchored_lognormal() {
        let src = covariate_nn_dotted_model_src();
        let model = parse_model_string(&src).expect("model parses");
        let cl_ref = model
            .mu_refs
            .get("ETA_CL")
            .expect("ETA_CL must be detected as mu-referenced");
        assert_eq!(cl_ref.theta_name, "TYPICAL_PK.CL");
        assert!(
            cl_ref.log_transformed,
            "TYPICAL_PK.CL * exp(ETA_CL) is lognormal"
        );
        let v_ref = model.mu_refs.get("ETA_V").expect("ETA_V mu-referenced");
        assert_eq!(v_ref.theta_name, "TYPICAL_PK.V");
        assert!(v_ref.log_transformed);
    }

    /// Fix 1 follow-on: `eta_param_info` still classifies NN-anchored
    /// patterns as `LogNormal` — the eta's statistical shape is unchanged;
    /// only the anchor differs.
    #[cfg(feature = "nn")]
    #[test]
    fn test_eta_param_info_classifies_nn_anchored_lognormal() {
        use crate::types::EtaParamType;
        let src = covariate_nn_dotted_model_src();
        let model = parse_model_string(&src).expect("model parses");
        let cl = model
            .eta_param_info
            .iter()
            .find(|i| i.eta_name == "ETA_CL")
            .expect("ETA_CL classification present");
        assert_eq!(cl.param_type, EtaParamType::LogNormal);
        let v = model
            .eta_param_info
            .iter()
            .find(|i| i.eta_name == "ETA_V")
            .expect("ETA_V classification present");
        assert_eq!(v.param_type, EtaParamType::LogNormal);
    }

    /// Fix 2: `tv_fn` (the eta=0 typical-value closure) must produce the
    /// same values as the NN forward pass. Previously the unindexed
    /// evaluator returned 0.0 for `NnOutput`, so `tv_fn` would have
    /// silently produced zero TVs for NN-bearing models — a footgun for
    /// the AD fast path the M2 PR will lean on.
    #[cfg(feature = "nn")]
    #[test]
    fn test_tv_fn_dispatches_through_nn() {
        use crate::nn::CovariateMapper;
        let src = covariate_nn_dotted_model_src();
        let model = parse_model_string(&src).expect("model parses");
        let theta = model.default_params.theta.clone();
        let mut cov = HashMap::new();
        cov.insert("WT".to_string(), 70.0);
        cov.insert("CRCL".to_string(), 95.0);

        let tv_fn = model
            .tv_fn
            .as_ref()
            .expect("analytical model -> Some(tv_fn)");
        let tvs = tv_fn(&theta, &cov);

        // tv_fn returns values in `indiv_param_names` declaration order:
        // CL, V, KA. At eta=0 the lognormal mu-ref `tv * exp(0)` collapses
        // to tv, which equals the NN's raw output for CL and V.
        let nn = &model.covariate_nns[0];
        let weights = &theta[nn.weights_offset..nn.weights_offset + nn.mapper.n_weights()];
        let nn_outputs = nn.mapper.forward_raw(weights, &cov).unwrap();

        assert!(
            (tvs[0] - nn_outputs[0]).abs() < 1e-12,
            "TV[CL] from tv_fn = {} vs NN output = {}",
            tvs[0],
            nn_outputs[0]
        );
        assert!(
            (tvs[1] - nn_outputs[1]).abs() < 1e-12,
            "TV[V] from tv_fn = {} vs NN output = {}",
            tvs[1],
            nn_outputs[1]
        );
        // KA = TVKA (plain theta path, unchanged).
        assert!((tvs[2] - theta[0]).abs() < 1e-12);
    }

    // ── [scaling] block parser tests ────────────────────────────────────────

    /// Analytical 1-cpt IV bolus model template — small enough to compose
    /// scaling-block variants on top.
    fn analytical_model_with_scaling(scaling_block: Option<&str>) -> String {
        let mut s = String::from(
            "\
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(50.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.05 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method  = focei
  maxiter = 10
  gradient = fd
",
        );
        if let Some(block) = scaling_block {
            s.push_str("\n[scaling]\n");
            s.push_str(block);
        }
        s
    }

    /// ODE 1-cpt oral template — small enough to compose Form C variants.
    fn ode_model_with_scaling(struct_line: &str, scaling_block: Option<&str>) -> String {
        let mut s = format!(
            "\
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(50.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.05 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA

[structural_model]
  {struct_line}

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) = KA * depot - CL/V * central

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method  = focei
  maxiter = 10
  gradient = fd
",
            struct_line = struct_line
        );
        if let Some(block) = scaling_block {
            s.push_str("\n[scaling]\n");
            s.push_str(block);
        }
        s
    }

    #[test]
    fn test_parse_scaling_none() {
        let src = analytical_model_with_scaling(None);
        let model = parse_model_string(&src).expect("base model parses");
        assert!(matches!(model.scaling, ScalingSpec::None));
    }

    #[test]
    fn test_parse_scaling_scalar() {
        let src = analytical_model_with_scaling(Some("  obs_scale = 1000\n"));
        let model = parse_model_string(&src).expect("scalar scaling parses");
        match model.scaling {
            ScalingSpec::ScalarScale(k) => assert!((k - 1000.0).abs() < 1e-12),
            other => panic!("expected ScalarScale, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_scaling_scalar_rejects_zero() {
        let src = analytical_model_with_scaling(Some("  obs_scale = 0\n"));
        let err = parse_model_string(&src).expect_err("obs_scale = 0 must be rejected");
        assert!(
            err.to_lowercase().contains("strictly positive"),
            "expected strictly-positive error, got: {}",
            err
        );
    }

    #[test]
    fn test_parse_scaling_scalar_rejects_negative() {
        let src = analytical_model_with_scaling(Some("  obs_scale = -1000\n"));
        let err = parse_model_string(&src).expect_err("negative obs_scale must be rejected");
        assert!(
            err.to_lowercase().contains("strictly positive"),
            "expected strictly-positive error, got: {}",
            err
        );
    }

    #[test]
    fn test_parse_scaling_expression_with_theta_evaluates() {
        // `obs_scale = TVV / 10` — references a theta, not an indiv param.
        let src = analytical_model_with_scaling(Some("  obs_scale = TVV / 10\n"));
        let model = parse_model_string(&src).expect("expression scaling parses");
        match model.scaling {
            ScalingSpec::ExpressionScale { ref scale_fn } => {
                // TVV = 50 (the parsed default), so scale = 50/10 = 5.
                let theta = vec![1.0, 50.0]; // [TVCL, TVV]
                let eta = vec![0.0];
                let cov = HashMap::new();
                let pk = PkParams::default();
                let s = scale_fn(&theta, &eta, &cov, &pk);
                assert!((s - 5.0).abs() < 1e-12, "expected 5.0, got {}", s);
            }
            other => panic!("expected ExpressionScale, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_scaling_expression_uses_covariate() {
        let src = analytical_model_with_scaling(Some("  obs_scale = WT / 70\n"));
        let model = parse_model_string(&src).expect("covariate scaling parses");
        match model.scaling {
            ScalingSpec::ExpressionScale { ref scale_fn } => {
                let theta = vec![1.0, 50.0];
                let eta = vec![0.0];
                let mut cov = HashMap::new();
                cov.insert("WT".to_string(), 84.0);
                let pk = PkParams::default();
                let s = scale_fn(&theta, &eta, &cov, &pk);
                assert!((s - 1.2).abs() < 1e-12, "expected 1.2, got {}", s);
            }
            other => panic!("expected ExpressionScale, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_scaling_expression_uses_indiv_param() {
        // Phase 1.5: `V` is an individual parameter — must resolve via the
        // PK slot in the pk_params snapshot passed to scale_fn. The
        // `analytical_model_with_scaling` template defines `V = TVV` (an
        // unattenuated theta passthrough), so V's runtime value equals
        // theta[1] = 50, and `obs_scale = 1000 / V` evaluates to 20.
        let src = analytical_model_with_scaling(Some("  obs_scale = 1000 / V\n"));
        let model = parse_model_string(&src).expect("indiv-param ref in obs_scale parses");
        match model.scaling {
            ScalingSpec::ExpressionScale { ref scale_fn } => {
                let theta = vec![1.0, 50.0]; // [TVCL, TVV]
                let eta = vec![0.0];
                let cov = HashMap::new();
                // Mimic what apply_scaling does at runtime: evaluate pk_param_fn
                // with the subject's theta/eta/covariates to materialize V.
                let pk = (model.pk_param_fn)(&theta, &eta, &cov);
                let s = scale_fn(&theta, &eta, &cov, &pk);
                assert!(
                    (s - 20.0).abs() < 1e-12,
                    "expected 20.0 (= 1000/50), got {}",
                    s
                );
            }
            other => panic!("expected ExpressionScale, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_scaling_y_on_analytical_errors() {
        let src = analytical_model_with_scaling(Some("  y = 1000\n"));
        let err = parse_model_string(&src).expect_err("y on analytical must be rejected");
        assert!(
            err.contains("Form C") || err.contains("ODE model"),
            "expected Form-C-requires-ODE error, got: {}",
            err
        );
    }

    #[test]
    fn test_parse_scaling_unknown_key_errors() {
        let src = analytical_model_with_scaling(Some("  foo = 1000\n"));
        let err = parse_model_string(&src).expect_err("unknown scaling key must be rejected");
        assert!(
            err.contains("unknown key") && err.contains("foo"),
            "expected unknown-key error, got: {}",
            err
        );
    }

    #[test]
    fn test_parse_scaling_y_form_c_on_ode() {
        // ODE form C: ode(states=...) without obs_cmt, and `y = central / V`
        // replaces the readout. Verifies the parse path completes and
        // output_fn evaluates correctly.
        let src =
            ode_model_with_scaling("ode(states=[depot, central])", Some("  y = central / V\n"));
        let model = parse_model_string(&src).expect("Form C parses");
        let ode = model.ode_spec.as_ref().expect("ODE spec present");
        let out_fn = match &ode.readout {
            crate::ode::OdeReadout::Single(f) => f,
            other => panic!(
                "Form C must set OdeReadout::Single, got {:?}",
                match other {
                    crate::ode::OdeReadout::ObsCmt(i) => format!("ObsCmt({})", i),
                    crate::ode::OdeReadout::Single(_) => "Single(..)".into(),
                    crate::ode::OdeReadout::PerCmt(_) => "PerCmt(..)".into(),
                }
            ),
        };

        // ODE writes indiv params sequentially into pk_params_flat[0..n] in
        // declaration order: [CL, V, KA] -> pk[0..3]. State order: [depot,
        // central] -> state[0..2]. So y = central / V = state[1] / pk[1].
        let state = vec![0.0, 100.0]; // depot=0, central=100
        let mut pk = vec![0.0f64; crate::types::MAX_PK_PARAMS];
        pk[0] = 1.0; // CL
        pk[1] = 50.0; // V
        pk[2] = 1.0; // KA
        let cov = HashMap::new();
        let y = out_fn(&state, &pk, &[], &[], &cov);
        assert!((y - 2.0).abs() < 1e-12, "expected 100/50 = 2, got {}", y);
    }

    #[test]
    fn test_parse_scaling_y_requires_scaling_block_for_missing_obs_cmt() {
        // ODE without obs_cmt and no [scaling] y = ... must error.
        let src = ode_model_with_scaling("ode(states=[depot, central])", None);
        let err = parse_model_string(&src)
            .expect_err("ODE without obs_cmt and without Form C must error");
        assert!(
            err.contains("obs_cmt") || err.contains("Form C"),
            "expected validation error, got: {}",
            err
        );
    }

    #[test]
    fn test_parse_scaling_scalar_on_ode_keeps_obs_cmt() {
        // Form A on an ODE model should preserve obs_cmt and apply post-multiply.
        let src = ode_model_with_scaling(
            "ode(obs_cmt=central, states=[depot, central])",
            Some("  obs_scale = 1000\n"),
        );
        let model = parse_model_string(&src).expect("Form A on ODE parses");
        let ode = model.ode_spec.as_ref().expect("ODE spec present");
        assert!(matches!(ode.readout, crate::ode::OdeReadout::ObsCmt(1)));
        assert!(matches!(model.scaling, ScalingSpec::ScalarScale(k) if (k - 1000.0).abs() < 1e-12));
    }

    #[test]
    fn test_parse_scaling_y_form_c_resolves_covariates() {
        // Regression for Copilot review: Form C parsed with ParseCtx::ode
        // silently turned `WT` into a `Variable` and returned 0.0. With the
        // fix (ParseCtx::new), unknown identifiers resolve as Covariate and
        // get looked up in the per-call covariate map.
        let src = ode_model_with_scaling(
            "ode(states=[depot, central])",
            Some("  y = central / V * WT\n"),
        );
        let model = parse_model_string(&src).expect("Form C with covariate parses");
        let ode = model.ode_spec.as_ref().unwrap();
        let out_fn = match &ode.readout {
            crate::ode::OdeReadout::Single(f) => f,
            _ => panic!("Form C must set OdeReadout::Single"),
        };

        let state = vec![0.0, 50.0]; // depot=0, central=50
        let mut pk = vec![0.0f64; crate::types::MAX_PK_PARAMS];
        pk[0] = 1.0; // CL
        pk[1] = 50.0; // V
        pk[2] = 1.0; // KA
        let mut cov = HashMap::new();
        cov.insert("WT".to_string(), 70.0);
        let y = out_fn(&state, &pk, &[], &[], &cov);
        // central/V * WT = 50/50 * 70 = 70. With the bug, WT was treated as
        // a Variable and read 0 from the empty vars map → y = 0.
        assert!(
            (y - 70.0).abs() < 1e-12,
            "expected 70.0 (covariate WT must resolve), got {}",
            y
        );
    }

    #[test]
    fn test_parse_scaling_y_form_c_resolves_theta() {
        // Follow-up after Copilot's Fix #1: switching Form C's ParseCtx to
        // ParseCtx::ode silently zeroed covariate refs; Copilot suggested
        // ParseCtx::new(EMPTY, EMPTY, ...) which fixed covariates but
        // introduced the symmetric bug for theta refs (TVCL became
        // Covariate("TVCL") → 0.0). Phase 1.5 fix: pass theta_names/eta_names
        // into ParseCtx so identifiers resolve as Theta(i) / Eta(i) and
        // thread theta/eta through OdeOutputFn at runtime.
        let src = ode_model_with_scaling(
            "ode(states=[depot, central])",
            Some("  y = central / V * TVCL\n"),
        );
        let model = parse_model_string(&src).expect("Form C with theta ref parses");
        let ode = model.ode_spec.as_ref().unwrap();
        let out_fn = match &ode.readout {
            crate::ode::OdeReadout::Single(f) => f,
            _ => panic!("Form C must set OdeReadout::Single"),
        };

        let state = vec![0.0, 50.0];
        let mut pk = vec![0.0f64; crate::types::MAX_PK_PARAMS];
        pk[0] = 1.0;
        pk[1] = 50.0;
        pk[2] = 1.0;
        let theta = vec![3.0, 50.0, 1.0]; // [TVCL=3, TVV=50, TVKA=1]
        let eta = vec![0.0];
        let cov = HashMap::new();
        let y = out_fn(&state, &pk, &theta, &eta, &cov);
        // central/V * TVCL = 50/50 * 3 = 3.
        assert!(
            (y - 3.0).abs() < 1e-12,
            "expected 3.0 (TVCL must resolve to theta[0]=3), got {}",
            y
        );
    }

    /// Issue #107: a Form C ODE output expression that references KAPPA_*
    /// directly would be evaluated per-observation with kappa=0, so it is
    /// rejected at parse time for IOV models.
    fn iov_ode_model_with_y(y_expr: &str) -> String {
        format!(
            "\
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(50.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.05 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV
  KA = TVKA

[structural_model]
  ode(states=[depot, central])

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) = KA * depot - CL/V * central

[error_model]
  DV ~ proportional(PROP_ERR)

[scaling]
  {y_expr}

[fit_options]
  method = focei
  iov_column = OCC
  gradient = fd
"
        )
    }

    #[test]
    fn ode_form_c_output_referencing_kappa_is_rejected_under_iov() {
        let src = iov_ode_model_with_y("y = central / V * exp(KAPPA_CL)");
        let err = parse_model_string(&src)
            .expect_err("Form C output referencing KAPPA_* must be rejected under IOV");
        assert!(
            err.contains("KAPPA") && err.contains("#107"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn ode_form_c_output_without_kappa_ref_is_allowed_under_iov() {
        // The readout reads only the structural state/param; the occasion
        // dependence enters through CL in the ODE rhs, so this must parse.
        let src = iov_ode_model_with_y("y = central / V");
        parse_model_string(&src)
            .expect("Form C output without a direct KAPPA_* reference must parse under IOV");
    }

    #[test]
    fn ode_form_c_output_referencing_kappa_in_condition_is_rejected_under_iov() {
        // KAPPA_* inside a conditional *condition* (not just a branch) must also
        // be rejected — the guard walks the condition tree (Copilot review #108).
        let src = iov_ode_model_with_y("y = if (KAPPA_CL > 0) central / V else central / V");
        let err = parse_model_string(&src)
            .expect_err("KAPPA_* in a Form C output condition must be rejected under IOV");
        assert!(
            err.contains("KAPPA") && err.contains("#107"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_parse_scaling_rejected_on_sde_model() {
        // Regression for Copilot review: EKF p_obs / r_obs run in unscaled
        // observation space, so Forms A/B Phase 1 scaling on SDE models
        // would produce mis-scaled variance. Parser must reject the
        // combination entirely (Form C was already rejected).
        let src = format!(
            "\
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(50.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.05 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -CL/V * central

[diffusion]
  central ~ 0.01

[error_model]
  DV ~ proportional(PROP_ERR)

[scaling]
  obs_scale = 1000

[fit_options]
  method  = focei
  maxiter = 5
  gradient = fd
"
        );
        let err =
            parse_model_string(&src).expect_err("SDE + [scaling] must be rejected in Phase 1");
        assert!(
            err.to_lowercase().contains("sde") || err.contains("[diffusion]"),
            "expected SDE rejection error, got: {}",
            err
        );
    }

    #[test]
    fn test_parse_scaling_expression_accepts_ad_gradient() {
        // Phase 2.5: Form B (`obs_scale = <expr>`) + `gradient = ad` is
        // now allowed. The AD path receives a per-observation scale
        // array materialised from a subject-static `pk_param_fn`
        // evaluation; the gradient treats the scale as constant w.r.t.
        // eta, which is exact for eta-independent scales (the common
        // case) and a documented approximation otherwise.
        let base = analytical_model_with_scaling(Some("  obs_scale = TVV / 10\n"));
        let src = base.replace("gradient = fd", "gradient = ad");
        parse_model_string(&src).expect("ExpressionScale + gradient = ad now parses");
    }

    // ── Phase 2: multi-analyte / per-CMT scaling ────────────────────────────

    #[test]
    fn test_parse_scaling_key_uniform() {
        let (base, cmt) = parse_scaling_key("obs_scale").unwrap();
        assert_eq!(base, "obs_scale");
        assert_eq!(cmt, None);
    }

    #[test]
    fn test_parse_scaling_key_per_cmt() {
        let (base, cmt) = parse_scaling_key("obs_scale[CMT=2]").unwrap();
        assert_eq!(base, "obs_scale");
        assert_eq!(cmt, Some(2));

        let (base, cmt) = parse_scaling_key("y[CMT=1]").unwrap();
        assert_eq!(base, "y");
        assert_eq!(cmt, Some(1));
    }

    #[test]
    fn test_parse_scaling_key_rejects_malformed() {
        assert!(parse_scaling_key("obs_scale[").is_err());
        assert!(parse_scaling_key("obs_scale[CMT=]").is_err());
        assert!(parse_scaling_key("obs_scale[CMT=abc]").is_err());
        assert!(parse_scaling_key("obs_scale[FOO=1]").is_err());
        assert!(parse_scaling_key("obs_scale[CMT=0]").is_err()); // 1-based
        assert!(parse_scaling_key("obs_scale[CMT=1]extra").is_err());
    }

    #[test]
    fn test_parse_scaling_per_cmt_scalar() {
        // Use the analytical template but layer a per-CMT scaling block on
        // top. Even though one_cpt_iv only emits CMT=1 observations,
        // the parser doesn't validate coverage (that happens at fit time),
        // so this exercises the parse path cleanly.
        let mut src = String::from(
            "\
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(50.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.05 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method  = focei
  maxiter = 5
  gradient = fd

[scaling]
  obs_scale[CMT=1] = 1000
  obs_scale[CMT=2] = 1
",
        );
        let _ = &mut src; // silence unused-mut on platforms without mods
        let model = parse_model_string(&src).expect("per-CMT obs_scale parses");
        match &model.scaling {
            ScalingSpec::PerCmt(map) => {
                assert_eq!(map.len(), 2);
                assert!(
                    matches!(map.get(&1), Some(ScalingSpec::ScalarScale(k)) if (*k - 1000.0).abs() < 1e-12)
                );
                assert!(
                    matches!(map.get(&2), Some(ScalingSpec::ScalarScale(k)) if (*k - 1.0).abs() < 1e-12)
                );
            }
            other => panic!("expected PerCmt, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_scaling_per_cmt_mixed_forms() {
        // CMT=1 uses scalar, CMT=2 uses expression. Both are valid within
        // the same PerCmt map.
        let src = analytical_model_with_scaling(Some(
            "  obs_scale[CMT=1] = 1000\n  obs_scale[CMT=2] = TVV / 10\n",
        ));
        let model = parse_model_string(&src).expect("mixed PerCmt parses");
        match &model.scaling {
            ScalingSpec::PerCmt(map) => {
                assert_eq!(map.len(), 2);
                assert!(matches!(map.get(&1), Some(ScalingSpec::ScalarScale(_))));
                assert!(matches!(
                    map.get(&2),
                    Some(ScalingSpec::ExpressionScale { .. })
                ));
            }
            other => panic!("expected PerCmt, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_scaling_rejects_mixing_uniform_and_per_cmt() {
        let src =
            analytical_model_with_scaling(Some("  obs_scale = 1000\n  obs_scale[CMT=2] = 1\n"));
        let err = parse_model_string(&src)
            .expect_err("mixing uniform + per-CMT obs_scale must be rejected");
        assert!(
            err.to_lowercase().contains("cannot mix"),
            "expected cannot-mix error, got: {}",
            err
        );
    }

    #[test]
    fn test_parse_scaling_rejects_duplicate_cmt() {
        let src = analytical_model_with_scaling(Some(
            "  obs_scale[CMT=1] = 1000\n  obs_scale[CMT=1] = 2000\n",
        ));
        let err = parse_model_string(&src).expect_err("duplicate CMT key must be rejected");
        assert!(
            err.contains("duplicate") && err.contains("CMT=1"),
            "expected duplicate-CMT-key error, got: {}",
            err
        );
    }

    #[test]
    fn test_parse_scaling_per_cmt_accepts_ad() {
        // Phase 2.5: per-CMT `obs_scale[CMT=N]` + `gradient = ad` is now
        // allowed. The AD path builds a per-observation scale array
        // dispatching the right per-CMT inner spec to each observation
        // via `inner_optimizer::build_scale_array_for_ad`.
        let base = analytical_model_with_scaling(Some(
            "  obs_scale[CMT=1] = 1000\n  obs_scale[CMT=2] = 1\n",
        ));
        let src = base.replace("gradient = fd", "gradient = ad");
        parse_model_string(&src).expect("per-CMT obs_scale + gradient = ad now parses");
    }

    #[test]
    fn test_parse_scaling_y_per_cmt_form_c() {
        // Per-CMT Form C on an ODE model. Build_y_output_fn picks up each
        // entry; the parser assembles them into OdeReadout::PerCmt.
        let src = ode_model_with_scaling(
            "ode(states=[depot, central])",
            Some("  y[CMT=1] = central / V\n  y[CMT=2] = central / V * 1000\n"),
        );
        let model = parse_model_string(&src).expect("per-CMT y Form C parses");
        let ode = model.ode_spec.as_ref().expect("ODE spec present");
        match &ode.readout {
            crate::ode::OdeReadout::PerCmt(map) => {
                assert_eq!(map.len(), 2);
                assert!(map.contains_key(&1));
                assert!(map.contains_key(&2));
            }
            _ => panic!("expected OdeReadout::PerCmt"),
        }
    }

    #[test]
    fn test_parse_scaling_y_form_c_rejects_ad() {
        // Regression for Copilot review on PR #84: the original guard only
        // checked `ScalingSpec::requires_fd()`, missing Form C readouts
        // (which live on `OdeSpec.readout`, not `model.scaling`). A Form C
        // model with `gradient = ad` silently fell back to FD via
        // `model.tv_fn.is_none()` at runtime; now the parser errors loudly.
        let src_per_cmt = ode_model_with_scaling(
            "ode(states=[depot, central])",
            Some("  y[CMT=1] = central / V\n  y[CMT=2] = central / V * 1000\n"),
        )
        .replace("gradient = fd", "gradient = ad");
        let err = parse_model_string(&src_per_cmt)
            .expect_err("per-CMT y + gradient = ad must be rejected");
        assert!(
            err.contains("per-CMT `y[CMT=N]`") && err.contains("gradient = fd"),
            "expected per-CMT Form C + AD rejection, got: {}",
            err
        );

        // Single Form C (uniform `y = <expr>`) gets the same treatment.
        let src_single =
            ode_model_with_scaling("ode(states=[depot, central])", Some("  y = central / V\n"))
                .replace("gradient = fd", "gradient = ad");
        let err = parse_model_string(&src_single)
            .expect_err("single Form C + gradient = ad must be rejected");
        assert!(
            err.contains("`y = <expr>` (Form C)") && err.contains("gradient = fd"),
            "expected single Form C + AD rejection, got: {}",
            err
        );
    }

    // ── Bytecode ↔ AST evaluator equivalence ────────────────────────────────
    //
    // The bytecode interpreter is a second evaluator for every
    // Expression/Condition shape ferx supports. These tests pin its
    // results to `eval_expression_indexed` so future opcode changes can't
    // silently diverge — Copilot review on #137 flagged the gap.

    fn bc_vs_ast(expr: Expression, vars: &[f64], theta: &[f64], eta: &[f64], covs: &[f64]) {
        let nn: Vec<Vec<f64>> = Vec::new();
        let ast = eval_expression_indexed(&expr, theta, eta, covs, vars, &nn);
        let bc = compile_bytecode(&expr);
        let mut stack: Vec<f64> = Vec::new();
        let by = eval_bytecode(&bc, theta, eta, covs, vars, &nn, &mut stack);
        assert_eq!(
            ast.to_bits(),
            by.to_bits(),
            "bit-identical mismatch: AST = {ast:?}, bytecode = {by:?}, expr = {expr:?}",
        );
    }

    fn lit(v: f64) -> Expression {
        Expression::Literal(v)
    }
    fn binop(op: BinOp, l: Expression, r: Expression) -> Expression {
        Expression::BinOp(Box::new(l), op, Box::new(r))
    }
    fn unary(name: &str, arg: Expression) -> Expression {
        Expression::UnaryFn(name.into(), Box::new(arg))
    }
    fn cond(c: Condition, t: Expression, e: Expression) -> Expression {
        Expression::Conditional(Box::new(c), Box::new(t), Box::new(e))
    }
    fn cmp(l: Expression, op: CmpOp, r: Expression) -> Condition {
        Condition::Compare(l, op, r)
    }

    #[test]
    fn compute_max_stack_allows_branching_bytecode() {
        // Regression: a `Conditional` compiles to
        //   cond; JumpIfFalse(else); then; Jump(end); else
        // so `compute_max_stack`'s linear scan walks BOTH arms and ends above
        // depth 1 (one extra per branch). The end-depth assertion must be
        // skipped when jumps are present; before the fix, `compile_bytecode`
        // panicked here under debug-assertions ("bytecode ends at depth 2").
        let expr = cond(cmp(lit(2.0), CmpOp::Lt, lit(3.0)), lit(1.0), lit(-1.0));
        let bc = compile_bytecode(&expr); // would panic pre-fix
        assert!(
            bc.max_stack >= 1,
            "max_stack must be >= 1, got {}",
            bc.max_stack
        );

        // The reserved size must actually be sufficient to evaluate correctly:
        // 2 < 3 ⇒ the `then` arm (1.0).
        let nn: Vec<Vec<f64>> = Vec::new();
        let mut stack: Vec<f64> = Vec::new();
        let by = eval_bytecode(&bc, &[], &[], &[], &[], &nn, &mut stack);
        assert_eq!(by.to_bits(), 1.0_f64.to_bits());

        // Nested conditionals (deeper branch nesting) must also not trip it,
        // and a deeper `peak` must still be returned.
        let nested = cond(
            cmp(lit(1.0), CmpOp::Gt, lit(0.0)),
            cond(cmp(lit(2.0), CmpOp::Lt, lit(5.0)), lit(10.0), lit(20.0)),
            lit(-1.0),
        );
        assert!(compute_max_stack(&compile_bytecode(&nested).ops) >= 1);

        // The jump-free path is unchanged: `1 + 2` pushes two operands before
        // `Add`, so the peak (the returned size) is 2 — and its end-depth of 1
        // still satisfies the (retained) jump-free assertion.
        assert_eq!(
            compute_max_stack(&compile_bytecode(&binop(BinOp::Add, lit(1.0), lit(2.0))).ops),
            2
        );

        // `bytecode_has_branch` (which gates the assertion) — true for
        // conditional bytecode, false for a jump-free arithmetic expression.
        assert!(bytecode_has_branch(&compile_bytecode(&expr).ops));
        assert!(!bytecode_has_branch(
            &compile_bytecode(&binop(BinOp::Add, lit(1.0), lit(2.0))).ops
        ));

        // `ends_at_expected_depth` — the returnable predicate the `debug_assert!`
        // consumes. Exercised directly (with hard-coded depths) so its branching
        // logic is covered even under the `ci-test` profile (debug-assertions
        // off), where the assertion is compiled out and never runs. Branchy
        // bytecode is exempt at any end depth; jump-free bytecode must end at
        // depth 1. The depth real bytecode actually compiles to is pinned
        // separately in `compute_max_stack_jumpfree_bytecode_ends_at_depth_one`.
        let cond_ops = compile_bytecode(&expr).ops;
        let addops = compile_bytecode(&binop(BinOp::Add, lit(1.0), lit(2.0))).ops;
        assert!(ends_at_expected_depth(&cond_ops, 2)); // branchy: exempt even at depth 2
        assert!(ends_at_expected_depth(&addops, 1)); // jump-free: depth 1 ok
        assert!(!ends_at_expected_depth(&addops, 2)); // jump-free: depth 2 rejected
        assert!(ends_at_expected_depth(&[], 0)); // empty: ok
    }

    #[test]
    fn compute_max_stack_jumpfree_bytecode_ends_at_depth_one() {
        // Closes the gap the sibling test leaves open. That one feeds
        // `ends_at_expected_depth` HARD-CODED depths, so it pins the predicate's
        // branching logic but never the depth a real expression compiles to — a
        // future `compile_expr_into` off-by-one would slip past it. Here we scan
        // REAL bytecode and assert the end-depth invariant directly.
        //
        // `scan_stack_depth` *returns* the end depth (rather than only feeding it
        // to the `debug_assert!` in `compute_max_stack`), so these are plain
        // value assertions that hold under every profile — including `ci-test`,
        // where debug-assertions are off. That is what makes such a regression
        // catchable in CI, not only under the local dev profile.
        for expr in [
            lit(3.0),
            binop(BinOp::Add, lit(1.0), lit(2.0)),
            binop(BinOp::Sub, binop(BinOp::Mul, lit(2.0), lit(3.0)), lit(4.0)),
            unary("exp", lit(0.5)),
            unary("ln", binop(BinOp::Div, lit(6.0), lit(2.0))),
        ] {
            let ops = compile_bytecode(&expr).ops;
            assert!(
                !bytecode_has_branch(&ops),
                "expr must compile jump-free for this invariant: {ops:?}"
            );
            let (peak, end_depth) = scan_stack_depth(&ops);
            // A well-formed expression leaves exactly one result on the stack.
            assert_eq!(
                end_depth, 1,
                "jump-free expr must end at depth 1, got {end_depth}: {ops:?}"
            );
            // `peak` (the reserved stack size) never under-counts the end depth,
            // and `compute_max_stack` returns exactly `peak.max(1)`.
            assert!(
                peak >= end_depth,
                "peak {peak} < end_depth {end_depth}: {ops:?}"
            );
            assert_eq!(compute_max_stack(&ops), peak.max(1) as usize);
        }

        // Counterpart to the exemption: branchy bytecode genuinely ends ABOVE
        // depth 1 (the linear scan walks both arms), which is exactly why
        // `compute_max_stack` skips the end-depth assert when jumps are present.
        let cond_ops = compile_bytecode(&cond(
            cmp(lit(2.0), CmpOp::Lt, lit(3.0)),
            lit(1.0),
            lit(-1.0),
        ))
        .ops;
        assert!(bytecode_has_branch(&cond_ops));
        assert!(
            scan_stack_depth(&cond_ops).1 > 1,
            "conditional should end above depth 1 (both arms walked): {cond_ops:?}"
        );
    }

    #[test]
    fn bytecode_matches_ast_on_arithmetic_and_literals() {
        let v = &[3.5, -2.0, 0.0];
        let t = &[10.0, 0.1];
        let e = &[0.5];
        let c = &[];
        // Lits and slot pushes
        bc_vs_ast(lit(7.5), v, t, e, c);
        bc_vs_ast(Expression::Theta(0), v, t, e, c);
        bc_vs_ast(Expression::Eta(0), v, t, e, c);
        bc_vs_ast(Expression::VariableIdx(1), v, t, e, c);
        // Binary arithmetic — left-to-right evaluation
        bc_vs_ast(binop(BinOp::Add, lit(2.0), lit(3.0)), v, t, e, c);
        bc_vs_ast(binop(BinOp::Sub, lit(2.0), lit(3.0)), v, t, e, c);
        bc_vs_ast(binop(BinOp::Mul, lit(2.0), lit(3.0)), v, t, e, c);
        bc_vs_ast(binop(BinOp::Div, lit(7.0), lit(2.0)), v, t, e, c);
        // Div-by-tiny guard (both paths clamp `r.abs() < 1e-30 -> 0.0`)
        bc_vs_ast(binop(BinOp::Div, lit(1.0), lit(1e-40)), v, t, e, c);
        bc_vs_ast(binop(BinOp::Div, lit(1.0), lit(0.0)), v, t, e, c);
        // Power
        bc_vs_ast(
            Expression::Power(Box::new(lit(2.0)), Box::new(lit(3.0))),
            v,
            t,
            e,
            c,
        );
        bc_vs_ast(
            Expression::Power(Box::new(lit(4.0)), Box::new(lit(0.5))),
            v,
            t,
            e,
            c,
        );
    }

    #[test]
    fn bytecode_matches_ast_on_unary_guards() {
        let v = &[];
        let t = &[];
        let e = &[];
        let c = &[];
        // exp / abs — no guards
        bc_vs_ast(unary("exp", lit(0.5)), v, t, e, c);
        bc_vs_ast(unary("abs", lit(-3.0)), v, t, e, c);
        // ln / log clamp `v.max(1e-30).ln()` — exercise negative and zero
        bc_vs_ast(unary("ln", lit(2.0)), v, t, e, c);
        bc_vs_ast(unary("ln", lit(0.0)), v, t, e, c);
        bc_vs_ast(unary("ln", lit(-1.0)), v, t, e, c);
        bc_vs_ast(unary("log", lit(1e-40)), v, t, e, c);
        // sqrt clamp `v.max(0.0).sqrt()`
        bc_vs_ast(unary("sqrt", lit(9.0)), v, t, e, c);
        bc_vs_ast(unary("sqrt", lit(-4.0)), v, t, e, c);
        // inv_logit numerically-stable branch (v ≥ 0 vs v < 0)
        bc_vs_ast(unary("inv_logit", lit(2.5)), v, t, e, c);
        bc_vs_ast(unary("inv_logit", lit(-2.5)), v, t, e, c);
        bc_vs_ast(unary("expit", lit(0.0)), v, t, e, c);
        // logit clamp `v.clamp(1e-15, 1.0 - 1e-15)`
        bc_vs_ast(unary("logit", lit(0.3)), v, t, e, c);
        bc_vs_ast(unary("logit", lit(0.0)), v, t, e, c);
        bc_vs_ast(unary("logit", lit(1.0)), v, t, e, c);
        // Unknown unary name — slow path returns the argument unchanged;
        // the bytecode no-op'd UnaryFn arm must too.
        bc_vs_ast(unary("expn", lit(42.0)), v, t, e, c);
    }

    #[test]
    fn bytecode_matches_ast_on_conditional_and_logic() {
        let v = &[];
        let t = &[];
        let e = &[];
        let c = &[];
        // Simple compare arms
        for op in [
            CmpOp::Lt,
            CmpOp::Le,
            CmpOp::Gt,
            CmpOp::Ge,
            CmpOp::Eq,
            CmpOp::Ne,
        ] {
            bc_vs_ast(
                cond(cmp(lit(2.0), op, lit(3.0)), lit(1.0), lit(-1.0)),
                v,
                t,
                e,
                c,
            );
            bc_vs_ast(
                cond(cmp(lit(3.0), op, lit(3.0)), lit(1.0), lit(-1.0)),
                v,
                t,
                e,
                c,
            );
        }
        // NaN-comparison sanity (NaN compares as false everywhere)
        bc_vs_ast(
            cond(
                cmp(lit(f64::NAN), CmpOp::Eq, lit(f64::NAN)),
                lit(1.0),
                lit(-1.0),
            ),
            v,
            t,
            e,
            c,
        );
        // And / Or / Not over compound conditions
        bc_vs_ast(
            cond(
                Condition::And(
                    Box::new(cmp(lit(1.0), CmpOp::Lt, lit(2.0))),
                    Box::new(cmp(lit(3.0), CmpOp::Lt, lit(4.0))),
                ),
                lit(42.0),
                lit(0.0),
            ),
            v,
            t,
            e,
            c,
        );
        bc_vs_ast(
            cond(
                Condition::Or(
                    Box::new(cmp(lit(1.0), CmpOp::Gt, lit(2.0))),
                    Box::new(cmp(lit(3.0), CmpOp::Lt, lit(4.0))),
                ),
                lit(7.0),
                lit(0.0),
            ),
            v,
            t,
            e,
            c,
        );
        bc_vs_ast(
            cond(
                Condition::Not(Box::new(cmp(lit(1.0), CmpOp::Lt, lit(2.0)))),
                lit(7.0),
                lit(0.0),
            ),
            v,
            t,
            e,
            c,
        );
        // Nested Conditional — exercises both then- and else-branch bytecode
        // bookkeeping (the compute_max_stack-over-counts case).
        let inner = cond(cmp(lit(1.0), CmpOp::Lt, lit(0.5)), lit(11.0), lit(22.0));
        let outer = cond(
            cmp(lit(0.0), CmpOp::Lt, lit(1.0)),
            inner.clone(),
            binop(BinOp::Add, lit(1.0), inner),
        );
        bc_vs_ast(outer, v, t, e, c);
    }

    #[test]
    fn bytecode_matches_ast_on_nn_output_fallback() {
        // PushNnOutput out-of-bounds falls back to 0.0 in both paths.
        let nn_outputs: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0]];
        let theta = &[];
        let eta = &[];
        let covs = &[];
        let vars = &[];
        // Valid index
        let ast_ok = eval_expression_indexed(
            &Expression::NnOutput {
                nn_idx: 0,
                output_idx: 1,
            },
            theta,
            eta,
            covs,
            vars,
            &nn_outputs,
        );
        let bc_ok = {
            let bc = compile_bytecode(&Expression::NnOutput {
                nn_idx: 0,
                output_idx: 1,
            });
            let mut stack = Vec::new();
            eval_bytecode(&bc, theta, eta, covs, vars, &nn_outputs, &mut stack)
        };
        assert_eq!(ast_ok, bc_ok);
        assert_eq!(bc_ok, 2.0);
    }

    // ── Symbolic AST differentiator ─────────────────────────────────────────
    //
    // Verifies `differentiate(expr, axis)` against central finite differences
    // on `eval_expression_indexed`. The tolerance reflects FD's noise floor:
    // central FD with h = 1e-5 on a smooth expression gives ~h² ≈ 1e-10
    // truncation error, so any disagreement larger than ~1e-7 relative
    // (multiplied by max(|f(x+h)|, |f(x-h)|, 1) for scale) is a real bug.

    /// Evaluate an Expression at the given (theta, eta, vars, covs) point.
    fn eval_at(expr: &Expression, theta: &[f64], eta: &[f64], vars: &[f64], covs: &[f64]) -> f64 {
        let nn: Vec<Vec<f64>> = Vec::new();
        eval_expression_indexed(expr, theta, eta, covs, vars, &nn)
    }

    /// Central finite-difference of `expr` along `axis` at the given point.
    /// Caller mutates the point's slot for the axis to compute f(x±h); we
    /// take a vec for each slice and indirect the mutation.
    fn fd_along(
        expr: &Expression,
        axis: DiffAxis,
        theta: &[f64],
        eta: &[f64],
        vars: &[f64],
        covs: &[f64],
    ) -> f64 {
        let h = 1e-5;
        let plus = |slot: usize, base: &[f64]| -> Vec<f64> {
            let mut v = base.to_vec();
            v[slot] += h;
            v
        };
        let minus = |slot: usize, base: &[f64]| -> Vec<f64> {
            let mut v = base.to_vec();
            v[slot] -= h;
            v
        };
        let (fp, fm) = match axis {
            DiffAxis::Theta(k) => {
                let tp = plus(k, theta);
                let tm = minus(k, theta);
                (
                    eval_at(expr, &tp, eta, vars, covs),
                    eval_at(expr, &tm, eta, vars, covs),
                )
            }
            DiffAxis::Eta(k) => {
                let ep = plus(k, eta);
                let em = minus(k, eta);
                (
                    eval_at(expr, theta, &ep, vars, covs),
                    eval_at(expr, theta, &em, vars, covs),
                )
            }
            DiffAxis::Variable(k) => {
                let vp = plus(k, vars);
                let vm = minus(k, vars);
                (
                    eval_at(expr, theta, eta, &vp, covs),
                    eval_at(expr, theta, eta, &vm, covs),
                )
            }
        };
        (fp - fm) / (2.0 * h)
    }

    /// Assert that the symbolic derivative matches central FD to ~FD-noise
    /// tolerance.
    fn assert_diff_matches_fd(
        expr: Expression,
        axis: DiffAxis,
        theta: &[f64],
        eta: &[f64],
        vars: &[f64],
        covs: &[f64],
    ) {
        let dexpr = differentiate(&expr, axis);
        let sym = eval_at(&dexpr, theta, eta, vars, covs);
        let num = fd_along(&expr, axis, theta, eta, vars, covs);
        let scale = sym.abs().max(num.abs()).max(1.0);
        let rel = (sym - num).abs() / scale;
        assert!(
            rel < 1e-6,
            "symbolic ≠ FD: sym = {sym}, fd = {num}, axis = {axis:?}, \n  expr = {expr:?}",
        );
    }

    // `lit`, `binop`, `unary` are already defined above in the bytecode
    // equivalence tests; reuse those. Add only `power` (not needed there).
    fn power(b: Expression, e: Expression) -> Expression {
        Expression::Power(Box::new(b), Box::new(e))
    }

    #[test]
    fn differentiate_constants_zero_against_every_axis() {
        let theta = &[1.0, 2.0];
        let eta = &[0.5];
        let vars = &[3.0];
        let covs = &[];
        for axis in [DiffAxis::Theta(0), DiffAxis::Eta(0), DiffAxis::Variable(0)] {
            let d = differentiate(&lit(7.5), axis);
            assert_eq!(eval_at(&d, theta, eta, vars, covs), 0.0);
            let d2 = differentiate(&Expression::CovariateIdx(0), axis);
            assert_eq!(eval_at(&d2, theta, eta, vars, covs), 0.0);
            let d3 = differentiate(
                &Expression::NnOutput {
                    nn_idx: 0,
                    output_idx: 0,
                },
                axis,
            );
            assert_eq!(eval_at(&d3, theta, eta, vars, covs), 0.0);
        }
    }

    #[test]
    fn differentiate_slot_pushes_kronecker_delta() {
        let theta = &[1.5, 2.5, 3.5];
        let eta = &[0.1, 0.2];
        let vars = &[4.0, 5.0];
        let covs = &[];
        // ∂θ_1 / ∂θ_j = δ_{1,j}
        for j in 0..3 {
            let d = differentiate(&Expression::Theta(1), DiffAxis::Theta(j));
            let want = if j == 1 { 1.0 } else { 0.0 };
            assert_eq!(eval_at(&d, theta, eta, vars, covs), want, "θ axis j={j}");
        }
        // Cross-axis: ∂θ_1 / ∂η_0 = 0
        let d = differentiate(&Expression::Theta(1), DiffAxis::Eta(0));
        assert_eq!(eval_at(&d, theta, eta, vars, covs), 0.0);
        // ∂η_1 / ∂η_j = δ_{1,j}
        for j in 0..2 {
            let d = differentiate(&Expression::Eta(1), DiffAxis::Eta(j));
            let want = if j == 1 { 1.0 } else { 0.0 };
            assert_eq!(eval_at(&d, theta, eta, vars, covs), want, "η axis j={j}");
        }
        // ∂v_0 / ∂v_0 = 1, ∂v_0 / ∂v_1 = 0
        for j in 0..2 {
            let d = differentiate(&Expression::VariableIdx(0), DiffAxis::Variable(j));
            let want = if j == 0 { 1.0 } else { 0.0 };
            assert_eq!(eval_at(&d, theta, eta, vars, covs), want, "var axis j={j}");
        }
    }

    #[test]
    fn differentiate_arithmetic_matches_fd() {
        // expr = θ_0 + θ_1*η_0 − v_0/θ_0
        let theta = &[1.5, 2.5];
        let eta = &[0.7];
        let vars = &[3.0];
        let covs = &[];
        let expr = binop(
            BinOp::Sub,
            binop(
                BinOp::Add,
                Expression::Theta(0),
                binop(BinOp::Mul, Expression::Theta(1), Expression::Eta(0)),
            ),
            binop(BinOp::Div, Expression::VariableIdx(0), Expression::Theta(0)),
        );
        for axis in [
            DiffAxis::Theta(0),
            DiffAxis::Theta(1),
            DiffAxis::Eta(0),
            DiffAxis::Variable(0),
        ] {
            assert_diff_matches_fd(expr.clone(), axis, theta, eta, vars, covs);
        }
    }

    #[test]
    fn differentiate_unary_fn_matches_fd() {
        let theta = &[1.5, 0.8];
        let eta = &[0.3];
        let vars = &[2.0];
        let covs = &[];
        // exp(θ_0)
        assert_diff_matches_fd(
            unary("exp", Expression::Theta(0)),
            DiffAxis::Theta(0),
            theta,
            eta,
            vars,
            covs,
        );
        // ln(θ_0 * v_0)
        assert_diff_matches_fd(
            unary(
                "ln",
                binop(BinOp::Mul, Expression::Theta(0), Expression::VariableIdx(0)),
            ),
            DiffAxis::Theta(0),
            theta,
            eta,
            vars,
            covs,
        );
        assert_diff_matches_fd(
            unary(
                "ln",
                binop(BinOp::Mul, Expression::Theta(0), Expression::VariableIdx(0)),
            ),
            DiffAxis::Variable(0),
            theta,
            eta,
            vars,
            covs,
        );
        // sqrt(θ_0² + v_0²)  — exercises non-trivial chain
        assert_diff_matches_fd(
            unary(
                "sqrt",
                binop(
                    BinOp::Add,
                    power(Expression::Theta(0), lit(2.0)),
                    power(Expression::VariableIdx(0), lit(2.0)),
                ),
            ),
            DiffAxis::Theta(0),
            theta,
            eta,
            vars,
            covs,
        );
        // inv_logit(η_0 + θ_1)
        assert_diff_matches_fd(
            unary(
                "inv_logit",
                binop(BinOp::Add, Expression::Eta(0), Expression::Theta(1)),
            ),
            DiffAxis::Eta(0),
            theta,
            eta,
            vars,
            covs,
        );
        assert_diff_matches_fd(
            unary(
                "expit",
                binop(BinOp::Add, Expression::Eta(0), Expression::Theta(1)),
            ),
            DiffAxis::Theta(1),
            theta,
            eta,
            vars,
            covs,
        );
        // logit needs a point strictly in (0, 1)
        assert_diff_matches_fd(
            unary("logit", binop(BinOp::Mul, lit(0.3), Expression::Theta(0))),
            DiffAxis::Theta(0),
            &[1.0, 0.0],
            eta,
            vars,
            covs,
        );
    }

    #[test]
    fn differentiate_abs_branches_on_sign() {
        // abs(θ_0 − 1) at θ_0 = 1.7  → derivative is +1 (positive branch)
        // abs(θ_0 − 1) at θ_0 = 0.3  → derivative is −1 (negative branch)
        let eta = &[];
        let vars = &[];
        let covs = &[];
        let expr = unary("abs", binop(BinOp::Sub, Expression::Theta(0), lit(1.0)));
        assert_diff_matches_fd(expr.clone(), DiffAxis::Theta(0), &[1.7], eta, vars, covs);
        assert_diff_matches_fd(expr, DiffAxis::Theta(0), &[0.3], eta, vars, covs);
    }

    #[test]
    fn differentiate_unknown_unary_passes_through() {
        // The slow path returns the argument unchanged for unknown names;
        // the differentiator must return the argument's derivative.
        let theta = &[2.0];
        let eta = &[];
        let vars = &[];
        let covs = &[];
        let expr = unary("expn", binop(BinOp::Mul, Expression::Theta(0), lit(3.0)));
        assert_diff_matches_fd(expr, DiffAxis::Theta(0), theta, eta, vars, covs);
    }

    #[test]
    fn differentiate_power_matches_fd_for_constant_and_variable_exponents() {
        let theta = &[2.5, 1.5];
        let eta = &[];
        let vars = &[];
        let covs = &[];
        // Constant exponent: θ_0² (the canonical case)
        assert_diff_matches_fd(
            power(Expression::Theta(0), lit(2.0)),
            DiffAxis::Theta(0),
            theta,
            eta,
            vars,
            covs,
        );
        // Constant exponent, non-integer: θ_0^0.5
        assert_diff_matches_fd(
            power(Expression::Theta(0), lit(0.5)),
            DiffAxis::Theta(0),
            theta,
            eta,
            vars,
            covs,
        );
        // Variable exponent (rare in PK/PD, but common via the Hill term):
        // θ_0^θ_1 — exercises the e' · ln(b) path
        assert_diff_matches_fd(
            power(Expression::Theta(0), Expression::Theta(1)),
            DiffAxis::Theta(0),
            theta,
            eta,
            vars,
            covs,
        );
        assert_diff_matches_fd(
            power(Expression::Theta(0), Expression::Theta(1)),
            DiffAxis::Theta(1),
            theta,
            eta,
            vars,
            covs,
        );
    }

    #[test]
    fn differentiate_conditional_picks_taken_branch() {
        // Conditional(θ_0 > 0, 2·θ_0, 3·θ_0)
        // At θ_0 = 1.5: derivative is 2; at θ_0 = -1.5: derivative is 3
        let eta = &[];
        let vars = &[];
        let covs = &[];
        let expr = Expression::Conditional(
            Box::new(Condition::Compare(
                Expression::Theta(0),
                CmpOp::Gt,
                lit(0.0),
            )),
            Box::new(binop(BinOp::Mul, lit(2.0), Expression::Theta(0))),
            Box::new(binop(BinOp::Mul, lit(3.0), Expression::Theta(0))),
        );
        assert_diff_matches_fd(expr.clone(), DiffAxis::Theta(0), &[1.5], eta, vars, covs);
        assert_diff_matches_fd(expr, DiffAxis::Theta(0), &[-1.5], eta, vars, covs);
    }

    #[test]
    fn differentiate_emax_pkpd_readout_matches_fd() {
        // y = E0 + EMAX · effect^γ / (EC50^γ + effect^γ)
        // Mapped to slots: θ_0=E0, θ_1=EMAX, θ_2=EC50, θ_3=γ; var_0=effect.
        // This is the Emax PK/PD readout shape the Tier 4a Form C codegen
        // (milestone 4) would have chain-ruled through. The codegen was
        // reverted in #145, but the differentiator end-to-end behaviour
        // on this shape is still worth pinning for any future consumer.
        let theta = &[3.0, 60.0, 7.5, 1.5];
        let eta = &[];
        let vars = &[2.0];
        let covs = &[];
        let eff = Expression::VariableIdx(0);
        let e0 = Expression::Theta(0);
        let emax = Expression::Theta(1);
        let ec50 = Expression::Theta(2);
        let gamma = Expression::Theta(3);
        let eff_g = power(eff.clone(), gamma.clone());
        let ec_g = power(ec50.clone(), gamma);
        let denom = binop(BinOp::Add, ec_g, eff_g.clone());
        let frac = binop(BinOp::Div, eff_g, denom);
        let expr = binop(BinOp::Add, e0, binop(BinOp::Mul, emax, frac));
        for axis in [
            DiffAxis::Theta(0),    // ∂y/∂E0 = 1
            DiffAxis::Theta(1),    // ∂y/∂EMAX = frac
            DiffAxis::Theta(2),    // ∂y/∂EC50 — the awkward one
            DiffAxis::Theta(3),    // ∂y/∂γ
            DiffAxis::Variable(0), // ∂y/∂effect — the sensitivity-ODE input
        ] {
            assert_diff_matches_fd(expr.clone(), axis, theta, eta, vars, covs);
        }
    }

    #[test]
    fn simplify_collapses_zero_and_one() {
        // 0 + (1 * x) → x
        let raw = binop(
            BinOp::Add,
            lit(0.0),
            binop(BinOp::Mul, lit(1.0), Expression::Theta(0)),
        );
        let simp = simplify_expr(&raw);
        assert!(matches!(simp, Expression::Theta(0)));
        // x * 0 → 0
        let raw = binop(BinOp::Mul, Expression::VariableIdx(2), lit(0.0));
        assert!(matches!(simplify_expr(&raw), Expression::Literal(v) if v == 0.0));
        // x - 0 → x
        let raw = binop(BinOp::Sub, Expression::Eta(0), lit(0.0));
        assert!(matches!(simplify_expr(&raw), Expression::Eta(0)));
        // 0 / x → 0
        let raw = binop(BinOp::Div, lit(0.0), Expression::Theta(1));
        assert!(matches!(simplify_expr(&raw), Expression::Literal(v) if v == 0.0));
    }

    // --- Milestone 2: `[individual_parameters]` partials ---
    //
    // Each test parses a small model and asserts the precomputed
    // `IndivParamPartials` rows agree with central FD on the pk_param_fn
    // output. We use the parser's `parse_full_model` entry point so the test
    // exercises the full pipeline: parsing → resolve → differentiate →
    // simplify → store on CompiledModel.

    /// Numerically evaluate a stored partial Expression with the same inputs
    /// the pk_param_fn closure sees. Returns the partial's value at
    /// (theta, eta, covariates), with `vars` cleared to zero (intermediates
    /// aren't realized — they show up in the partial expression through
    /// chain-rule substitution at differentiation time, so the eval here
    /// doesn't need to recompute them).
    fn eval_partial(
        partial: &Expression,
        theta: &[f64],
        eta: &[f64],
        cov: &HashMap<String, f64>,
        var_count: usize,
    ) -> f64 {
        let mut cov_vec = vec![0.0_f64; cov.len()];
        // Build a deterministic cov ordering matching the partial's
        // CovariateIdx references. For these tests we use NO covariates.
        let mut keys: Vec<&String> = cov.keys().collect();
        keys.sort();
        for (i, k) in keys.iter().enumerate() {
            cov_vec[i] = cov[k.as_str()];
        }
        let mut vars = vec![0.0_f64; var_count];
        let nn_outputs: Vec<Vec<f64>> = Vec::new();
        eval_expression_indexed(partial, theta, eta, &cov_vec, &mut vars, &nn_outputs)
    }

    /// Central-FD reference for ∂(pk_param_fn output at slot i)/∂θ_k.
    fn fd_d_theta(
        model: &crate::types::CompiledModel,
        i: usize,
        k: usize,
        theta: &[f64],
        eta: &[f64],
        cov: &HashMap<String, f64>,
    ) -> f64 {
        let h = 1e-5 * theta[k].abs().max(1.0);
        let mut tp = theta.to_vec();
        let mut tm = theta.to_vec();
        tp[k] += h;
        tm[k] -= h;
        let pk_plus = (model.pk_param_fn)(&tp, eta, cov);
        let pk_minus = (model.pk_param_fn)(&tm, eta, cov);
        let slot = model.pk_indices[i];
        (pk_plus.values[slot] - pk_minus.values[slot]) / (2.0 * h)
    }

    /// Central-FD reference for ∂(pk_param_fn output at slot i)/∂η_k.
    fn fd_d_eta(
        model: &crate::types::CompiledModel,
        i: usize,
        k: usize,
        theta: &[f64],
        eta: &[f64],
        cov: &HashMap<String, f64>,
    ) -> f64 {
        let h = 1e-5_f64.max(eta[k].abs() * 1e-5);
        let mut ep = eta.to_vec();
        let mut em = eta.to_vec();
        ep[k] += h;
        em[k] -= h;
        let pk_plus = (model.pk_param_fn)(theta, &ep, cov);
        let pk_minus = (model.pk_param_fn)(theta, &em, cov);
        let slot = model.pk_indices[i];
        (pk_plus.values[slot] - pk_minus.values[slot]) / (2.0 * h)
    }

    #[test]
    fn indiv_partials_populated_for_1cpt_oral() {
        // CL = TVCL * exp(ETA_CL), V = TVV * exp(ETA_V), KA = TVKA * exp(ETA_KA)
        // Three indiv params, three θ, three η, all flat (no chain).
        let model_str = "
[parameters]
  theta TVCL(7.5)
  theta TVV(75.0)
  theta TVKA(1.05)
  omega ETA_CL ~ 0.135
  omega ETA_V  ~ 0.135
  omega ETA_KA ~ 0.225
  sigma EPS ~ 0.245

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(EPS)
";
        let parsed = super::parse_full_model(model_str).unwrap();
        let m = &parsed.model;
        let partials = &m.indiv_param_partials;
        assert_eq!(partials.names, vec!["CL", "V", "KA"]);
        assert_eq!(partials.d_d_theta.len(), 3);
        assert_eq!(partials.d_d_eta.len(), 3);
        // Each row has length n_theta_base (3) and n_eta_extended (3).
        for row in &partials.d_d_theta {
            assert_eq!(row.len(), 3);
        }
        for row in &partials.d_d_eta {
            assert_eq!(row.len(), 3);
        }

        // FD verification at off-zero η so exp(ETA) ≠ 1 and we exercise
        // the chain through exp.
        let theta = &[7.5, 75.0, 1.05];
        let eta = &[0.1, -0.2, 0.05];
        let cov: HashMap<String, f64> = HashMap::new();
        for i in 0..3 {
            for k in 0..3 {
                let sym = eval_partial(&partials.d_d_theta[i][k], theta, eta, &cov, 8);
                let fd = fd_d_theta(m, i, k, theta, eta, &cov);
                assert!(
                    (sym - fd).abs() < 1e-6 * sym.abs().max(fd.abs()).max(1e-8),
                    "∂{}/∂θ_{}: sym={sym}, fd={fd}",
                    partials.names[i],
                    k,
                );
                let sym = eval_partial(&partials.d_d_eta[i][k], theta, eta, &cov, 8);
                let fd = fd_d_eta(m, i, k, theta, eta, &cov);
                assert!(
                    (sym - fd).abs() < 1e-6 * sym.abs().max(fd.abs()).max(1e-8),
                    "∂{}/∂η_{}: sym={sym}, fd={fd}",
                    partials.names[i],
                    k,
                );
            }
        }
    }

    #[test]
    fn indiv_partials_zero_for_cross_axis() {
        // ∂CL/∂η_V = 0, ∂V/∂η_CL = 0. The simplified expressions should be
        // Literal(0.0) (or a tree that evaluates to 0).
        let model_str = "
[parameters]
  theta TVCL(7.5)
  theta TVV(75.0)
  theta TVKA(1.05)
  omega ETA_CL ~ 0.135
  omega ETA_V  ~ 0.135
  omega ETA_KA ~ 0.225
  sigma EPS ~ 0.245

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(EPS)
";
        let parsed = super::parse_full_model(model_str).unwrap();
        let partials = &parsed.model.indiv_param_partials;
        // After simplify, ∂CL/∂η_V is exactly Literal(0). Even if a future
        // simplifier rule leaves it as a tree, eval must produce 0.0.
        let theta = &[7.5, 75.0, 1.05];
        let eta = &[0.3, -0.1, 0.2];
        let cov: HashMap<String, f64> = HashMap::new();
        // Cross-η partials (indiv i, η k ≠ i).
        for (i, k) in [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)] {
            let v = eval_partial(&partials.d_d_eta[i][k], theta, eta, &cov, 8);
            assert!(
                v.abs() < 1e-12,
                "∂{}/∂η_{} should be 0, got {v}",
                partials.names[i],
                k
            );
        }
        // Cross-θ partials.
        for (i, k) in [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)] {
            let v = eval_partial(&partials.d_d_theta[i][k], theta, eta, &cov, 8);
            assert!(
                v.abs() < 1e-12,
                "∂{}/∂θ_{} should be 0, got {v}",
                partials.names[i],
                k
            );
        }
    }

    #[test]
    fn indiv_partials_logit_inv_logit_chain() {
        // F = inv_logit(logit(THETA_F) + ETA_F)
        // Exercises the inv_logit/logit derivative rules end-to-end.
        let model_str = "
[parameters]
  theta TVCL(7.5)
  theta TVV(75.0)
  theta TVKA(1.05)
  theta THETA_F(0.6)
  omega ETA_CL ~ 0.135
  omega ETA_V  ~ 0.135
  omega ETA_KA ~ 0.225
  omega ETA_F  ~ 0.1
  sigma EPS ~ 0.245

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
  F  = inv_logit(logit(THETA_F) + ETA_F)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, f=F)

[error_model]
  DV ~ proportional(EPS)
";
        let parsed = super::parse_full_model(model_str).unwrap();
        let m = &parsed.model;
        let partials = &m.indiv_param_partials;
        assert_eq!(partials.names, vec!["CL", "V", "KA", "F"]);
        let theta = &[7.5, 75.0, 1.05, 0.6];
        let eta = &[0.05, -0.1, 0.15, 0.2];
        let cov: HashMap<String, f64> = HashMap::new();
        // F's row: position 3. Verify ∂F/∂η_F (index 3) and ∂F/∂THETA_F (index 3).
        let sym = eval_partial(&partials.d_d_eta[3][3], theta, eta, &cov, 8);
        let fd = fd_d_eta(m, 3, 3, theta, eta, &cov);
        assert!(
            (sym - fd).abs() < 1e-6 * sym.abs().max(fd.abs()).max(1e-8),
            "∂F/∂η_F: sym={sym}, fd={fd}",
        );
        let sym = eval_partial(&partials.d_d_theta[3][3], theta, eta, &cov, 8);
        let fd = fd_d_theta(m, 3, 3, theta, eta, &cov);
        assert!(
            (sym - fd).abs() < 1e-6 * sym.abs().max(fd.abs()).max(1e-8),
            "∂F/∂THETA_F: sym={sym}, fd={fd}",
        );
        // F doesn't depend on ETA_CL/V/KA — those partials must evaluate to 0.
        for k in 0..3 {
            let v = eval_partial(&partials.d_d_eta[3][k], theta, eta, &cov, 8);
            assert!(v.abs() < 1e-12, "∂F/∂η_{k} should be 0, got {v}");
        }
    }

    #[test]
    fn indiv_partials_chain_rule_through_intermediate() {
        // Synthetic model exercising the chain-rule substitution at a
        // VariableIdx leaf — `KA` references `ka_mult` (an earlier
        // intermediate) by name. The differentiator must chain-rule through
        // ka_mult's η_KAM dependence into KA's row.
        //
        // ka_mult = TVKAM * exp(ETA_KAM)
        // KA      = TVKA  * ka_mult     (depends on η_KAM via ka_mult)
        //
        // ∂KA/∂η_KAM = TVKA * ∂ka_mult/∂η_KAM = TVKA * TVKAM * exp(ETA_KAM)
        // ∂KA/∂η_KA  = 0
        // ∂KA/∂TVKAM = TVKA * exp(ETA_KAM)
        let model_str = "
[parameters]
  theta TVCL(7.5)
  theta TVV(75.0)
  theta TVKA(1.05)
  theta TVKAM(2.0)
  omega ETA_CL  ~ 0.135
  omega ETA_V   ~ 0.135
  omega ETA_KAM ~ 0.1
  sigma EPS ~ 0.245

[individual_parameters]
  CL      = TVCL * exp(ETA_CL)
  V       = TVV  * exp(ETA_V)
  ka_mult = TVKAM * exp(ETA_KAM)
  KA      = TVKA * ka_mult

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(EPS)
";
        let parsed = super::parse_full_model(model_str).unwrap();
        let m = &parsed.model;
        let partials = &m.indiv_param_partials;
        // ka_mult IS a top-level Assign so it has a row; ordering must match
        // declaration order.
        assert_eq!(partials.names, vec!["CL", "V", "ka_mult", "KA"]);
        let theta = &[7.5, 75.0, 1.05, 2.0];
        let eta = &[0.0, 0.0, 0.3]; // η_CL=0, η_V=0, η_KAM=0.3
        let cov: HashMap<String, f64> = HashMap::new();
        // KA is at indiv-param index 3, η_KAM is at eta index 2.
        let sym = eval_partial(&partials.d_d_eta[3][2], theta, eta, &cov, 8);
        let expected = theta[2] * theta[3] * eta[2].exp(); // TVKA * TVKAM * exp(ETA_KAM)
        assert!(
            (sym - expected).abs() < 1e-12 * expected.abs().max(1.0),
            "∂KA/∂η_KAM (chain rule): sym={sym}, expected={expected}",
        );
        // FD cross-check via pk_param_fn:
        let fd = fd_d_eta(m, 3, 2, theta, eta, &cov);
        assert!(
            (sym - fd).abs() < 1e-6 * sym.abs().max(fd.abs()).max(1e-8),
            "∂KA/∂η_KAM: sym={sym}, fd={fd}",
        );
        // ∂KA/∂η_CL = 0 (no chain to ETA_CL).
        let v = eval_partial(&partials.d_d_eta[3][0], theta, eta, &cov, 8);
        assert!(v.abs() < 1e-12, "∂KA/∂η_CL should be 0, got {v}");
        // ∂KA/∂TVKAM (θ index 3): TVKA * exp(ETA_KAM).
        let sym = eval_partial(&partials.d_d_theta[3][3], theta, eta, &cov, 8);
        let expected = theta[2] * eta[2].exp();
        assert!(
            (sym - expected).abs() < 1e-12 * expected.abs().max(1.0),
            "∂KA/∂TVKAM: sym={sym}, expected={expected}",
        );
    }

    // ── unused-parameter warning tests ──────────────────────────────────────

    fn minimal_model(parameters: &str, individual_parameters: &str, error_model: &str) -> String {
        format!(
            r#"
[parameters]
{parameters}

[individual_parameters]
{individual_parameters}

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
DV ~ proportional({error_model})

[fit_options]
method = foce
"#
        )
    }

    #[test]
    fn test_all_used_no_warnings() {
        let src = minimal_model(
            "theta TVCL(0.1)\nomega ETA_CL ~ 0.09\nsigma PROP ~ 0.01",
            "CL = TVCL * exp(ETA_CL)\nV = 10.0\nKA = 1.0",
            "PROP",
        );
        let parsed = parse_full_model(&src).expect("parse ok");
        assert!(
            parsed.model.parse_warnings.is_empty(),
            "unexpected warnings: {:?}",
            parsed.model.parse_warnings
        );
    }

    #[test]
    fn test_unused_theta_warns() {
        let src = minimal_model(
            "theta TVCL(0.1)\ntheta UNUSED(0.5)\nomega ETA_CL ~ 0.09\nsigma PROP ~ 0.01",
            "CL = TVCL * exp(ETA_CL)\nV = 10.0\nKA = 1.0",
            "PROP",
        );
        let parsed = parse_full_model(&src).expect("parse ok");
        let warns: Vec<_> = parsed
            .model
            .parse_warnings
            .iter()
            .filter(|w| w.contains("UNUSED"))
            .collect();
        assert_eq!(
            warns.len(),
            1,
            "expected exactly one warning for UNUSED theta"
        );
        assert!(warns[0].contains("theta"), "warning should mention 'theta'");
    }

    #[test]
    fn test_unused_omega_warns() {
        let src = minimal_model(
            "theta TVCL(0.1)\nomega ETA_CL ~ 0.09\nomega ETA_UNUSED ~ 0.04\nsigma PROP ~ 0.01",
            "CL = TVCL * exp(ETA_CL)\nV = 10.0\nKA = 1.0",
            "PROP",
        );
        let parsed = parse_full_model(&src).expect("parse ok");
        let warns: Vec<_> = parsed
            .model
            .parse_warnings
            .iter()
            .filter(|w| w.contains("ETA_UNUSED"))
            .collect();
        assert_eq!(
            warns.len(),
            1,
            "expected exactly one warning for ETA_UNUSED omega"
        );
        assert!(warns[0].contains("omega"), "warning should mention 'omega'");
    }

    #[test]
    fn test_unused_sigma_warns() {
        let src = minimal_model(
            "theta TVCL(0.1)\nomega ETA_CL ~ 0.09\nsigma PROP ~ 0.01\nsigma ADD_UNUSED ~ 0.01",
            "CL = TVCL * exp(ETA_CL)\nV = 10.0\nKA = 1.0",
            "PROP",
        );
        let parsed = parse_full_model(&src).expect("parse ok");
        let warns: Vec<_> = parsed
            .model
            .parse_warnings
            .iter()
            .filter(|w| w.contains("ADD_UNUSED"))
            .collect();
        assert_eq!(
            warns.len(),
            1,
            "expected exactly one warning for ADD_UNUSED sigma"
        );
        assert!(warns[0].contains("sigma"), "warning should mention 'sigma'");
    }

    #[test]
    fn test_commented_out_usage_warns() {
        // Simulates: CL = TVCL #* exp(ETA_CL) — ETA_CL is commented away
        let src = minimal_model(
            "theta TVCL(0.1)\nomega ETA_CL ~ 0.09\nsigma PROP ~ 0.01",
            "CL = TVCL\nV = 10.0\nKA = 1.0",
            "PROP",
        );
        let parsed = parse_full_model(&src).expect("parse ok");
        let warns: Vec<_> = parsed
            .model
            .parse_warnings
            .iter()
            .filter(|w| w.contains("ETA_CL"))
            .collect();
        assert_eq!(warns.len(), 1, "ETA_CL not used → should warn");
    }

    #[test]
    fn test_multiple_unused_all_reported() {
        let src = minimal_model(
            "theta TVCL(0.1)\ntheta TH_UNUSED(0.5)\nomega ETA_CL ~ 0.09\nomega ETA_UNUSED ~ 0.04\nsigma PROP ~ 0.01\nsigma SIG_UNUSED ~ 0.01",
            "CL = TVCL * exp(ETA_CL)\nV = 10.0\nKA = 1.0",
            "PROP",
        );
        let parsed = parse_full_model(&src).expect("parse ok");
        let unused: Vec<_> = parsed
            .model
            .parse_warnings
            .iter()
            .filter(|w| w.contains("UNUSED"))
            .collect();
        assert_eq!(
            unused.len(),
            3,
            "expected warnings for TH_UNUSED, ETA_UNUSED, SIG_UNUSED; got: {:?}",
            unused
        );
    }

    #[test]
    fn test_unused_kappa_warns() {
        // KAPPA_CL declared but not used in any expression (index arithmetic:
        // Eta(n_eta + i), distinct from BSV etas).
        let src = format!(
            r#"
[parameters]
  theta TVCL(0.1)
  omega ETA_CL ~ 0.09
  kappa KAPPA_CL ~ 0.01
  sigma PROP ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = 10.0
  KA = 1.0

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP)

[fit_options]
  method = foce
  iov_column = OCC
"#
        );
        let parsed = parse_full_model(&src).expect("parse ok");
        let warns: Vec<_> = parsed
            .model
            .parse_warnings
            .iter()
            .filter(|w| w.contains("KAPPA_CL"))
            .collect();
        assert_eq!(
            warns.len(),
            1,
            "expected one warning for unused KAPPA_CL; got: {:?}",
            warns
        );
        assert!(warns[0].contains("kappa"), "warning should mention 'kappa'");
    }

    #[test]
    fn test_derived_variable_no_false_positive() {
        // ke = CL / V is a derived variable — no theta/eta directly.
        // But CL = TVCL * exp(ETA_CL) and V = TVV * exp(ETA_V) ARE in
        // indiv_stmts and ARE walked, so TVCL/ETA_CL/TVV/ETA_V are found
        // through those statements. No spurious "unused" warnings expected.
        let src = format!(
            r#"
[parameters]
  theta TVCL(0.1)
  theta TVV(10.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.04
  sigma PROP ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  ke = CL / V
  KA = 1.0

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP)

[fit_options]
  method = foce
"#
        );
        let parsed = parse_full_model(&src).expect("parse ok");
        // `ke = CL/V` is itself an unused intermediate, now flagged by the
        // computed-but-unused check (#309). This test's point is narrower: the
        // thetas/etas it references (TVCL, TVV, ETA_CL, ETA_V) must NOT be falsely
        // reported as unused.
        assert!(
            !parsed
                .model
                .parse_warnings
                .iter()
                .any(|w| w.contains("not referenced in any model expression")),
            "thetas/etas used via the derived variable ke=CL/V must not be flagged \
             unused; got: {:?}",
            parsed.model.parse_warnings
        );
    }

    #[test]
    fn test_theta_used_only_in_conditional_branch_no_warn() {
        // WT_POW is only referenced inside an if-branch — it must still be
        // found because collect_theta_eta_in_stmts recurses into if-bodies.
        let src = format!(
            r#"
[parameters]
  theta TVCL(0.1)
  theta WT_POW(0.75)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.01

[individual_parameters]
  if (WT > 0) {{
    CL = TVCL * (WT / 70)^WT_POW * exp(ETA_CL)
  }} else {{
    CL = TVCL * exp(ETA_CL)
  }}
  V  = 10.0
  KA = 1.0

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP)

[fit_options]
  method = foce
"#
        );
        let parsed = parse_full_model(&src).expect("parse ok");
        // The conditional model legitimately triggers a mu-referencing warning
        // for CL — that is unrelated to our check. Assert only that no
        // unused-parameter warning is emitted for WT_POW or any other parameter.
        let unused: Vec<_> = parsed
            .model
            .parse_warnings
            .iter()
            .filter(|w| w.contains("declared in [parameters] but not referenced"))
            .collect();
        assert!(
            unused.is_empty(),
            "WT_POW used inside if-branch must not trigger unused-param warning; got: {:?}",
            unused
        );
    }

    #[test]
    fn test_block_omega_one_unused_warns() {
        // block_omega declares ETA_CL and ETA_V together; only ETA_CL is used.
        // ETA_V should produce a warning even though it's part of a block.
        let src = format!(
            r#"
[parameters]
  theta TVCL(0.1)
  block_omega (ETA_CL, ETA_V) = [0.09, 0.01, 0.04]
  sigma PROP ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = 10.0
  KA = 1.0

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP)

[fit_options]
  method = foce
"#
        );
        let parsed = parse_full_model(&src).expect("parse ok");
        let warns: Vec<_> = parsed
            .model
            .parse_warnings
            .iter()
            .filter(|w| w.contains("ETA_V"))
            .collect();
        assert_eq!(
            warns.len(),
            1,
            "ETA_V unused in block_omega should warn; got: {:?}",
            warns
        );
        assert!(warns[0].contains("omega"), "warning should mention 'omega'");
        // ETA_CL IS used — must not appear in warnings
        assert!(
            !parsed
                .model
                .parse_warnings
                .iter()
                .any(|w| w.contains("ETA_CL")),
            "ETA_CL is used and must not warn"
        );
    }

    #[test]
    fn test_block_omega_all_used_no_warn() {
        // Both etas in a block_omega are used — no warnings expected.
        let src = format!(
            r#"
[parameters]
  theta TVCL(0.1)
  theta TVV(10.0)
  block_omega (ETA_CL, ETA_V) = [0.09, 0.01, 0.04]
  sigma PROP ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = 1.0

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP)

[fit_options]
  method = foce
"#
        );
        let parsed = parse_full_model(&src).expect("parse ok");
        assert!(
            parsed.model.parse_warnings.is_empty(),
            "all block_omega etas used — no warnings expected; got: {:?}",
            parsed.model.parse_warnings
        );
    }

    // ── [derived] block parser unit tests ────────────────────────────────────

    fn minimal_model_with_derived(derived_block: &str) -> String {
        format!(
            r#"
[parameters]
  theta CL(1.0, 0, 100)
  theta V(10.0, 0, 1000)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP   ~ 0.01

[individual_parameters]
  CL = exp(log(CL) + ETA_CL)
  V  = exp(log(V)  + ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)

[derived]
{derived_block}
"#
        )
    }

    fn make_derived_ctx_simple() -> (
        Vec<f64>,
        Vec<f64>,
        std::collections::HashMap<String, f64>,
        std::collections::HashMap<String, f64>,
        std::collections::HashMap<String, f64>,
    ) {
        let theta = vec![1.0, 10.0];
        let eta = vec![0.0, 0.0];
        let mut indiv_params = std::collections::HashMap::new();
        indiv_params.insert("CL".to_string(), 1.0);
        indiv_params.insert("V".to_string(), 10.0);
        let covariates = std::collections::HashMap::new();
        let prev_derived = std::collections::HashMap::new();
        (theta, eta, indiv_params, covariates, prev_derived)
    }

    #[test]
    fn parse_derived_per_row() {
        let src = minimal_model_with_derived("KE = CL / V");
        let parsed = parse_full_model(&src).expect("parse ok");
        assert_eq!(parsed.model.derived_exprs.len(), 1);
        assert_eq!(parsed.model.derived_exprs[0].name, "KE");
        assert!(matches!(
            parsed.model.derived_exprs[0].kind,
            DerivedKind::PerRow { .. }
        ));

        // Evaluate the closure
        let (theta, eta, indiv_params, covariates, prev_derived) = make_derived_ctx_simple();
        if let DerivedKind::PerRow { eval } = &parsed.model.derived_exprs[0].kind {
            let ctx = DerivedContext {
                theta: &theta,
                eta: &eta,
                indiv_params: &indiv_params,
                covariates: &covariates,
                ipred: 0.0,
                pred: 0.0,
                dv: 0.0,
                time: 1.0,
                tafd: 1.0,
                tad: 1.0,
                prev_derived: &prev_derived,
                compartments: &[],
                compartment_names: &[],
            };
            let ke = eval(&ctx);
            assert!((ke - 0.1).abs() < 1e-10, "KE = CL/V = 1/10 = 0.1, got {ke}");
        } else {
            panic!("expected PerRow");
        }
    }

    #[test]
    fn parse_derived_sequential_reference() {
        let src = minimal_model_with_derived("KE = CL / V\nT_HALF = 0.693 / KE");
        let parsed = parse_full_model(&src).expect("parse ok");
        assert_eq!(parsed.model.derived_exprs.len(), 2);

        let (theta, eta, indiv_params, covariates, _) = make_derived_ctx_simple();
        let mut prev_derived = std::collections::HashMap::new();

        // Evaluate KE first
        if let DerivedKind::PerRow { eval } = &parsed.model.derived_exprs[0].kind {
            let ctx = DerivedContext {
                theta: &theta,
                eta: &eta,
                indiv_params: &indiv_params,
                covariates: &covariates,
                ipred: 0.0,
                pred: 0.0,
                dv: 0.0,
                time: 1.0,
                tafd: 1.0,
                tad: 1.0,
                prev_derived: &prev_derived,
                compartments: &[],
                compartment_names: &[],
            };
            let ke = eval(&ctx);
            prev_derived.insert("KE".to_string(), ke);
        }

        // Now evaluate T_HALF using prev_derived
        if let DerivedKind::PerRow { eval } = &parsed.model.derived_exprs[1].kind {
            let ctx = DerivedContext {
                theta: &theta,
                eta: &eta,
                indiv_params: &indiv_params,
                covariates: &covariates,
                ipred: 0.0,
                pred: 0.0,
                dv: 0.0,
                time: 1.0,
                tafd: 1.0,
                tad: 1.0,
                prev_derived: &prev_derived,
                compartments: &[],
                compartment_names: &[],
            };
            let t_half = eval(&ctx);
            let expected = 0.693 / 0.1;
            assert!(
                (t_half - expected).abs() < 1e-8,
                "T_HALF = 0.693/KE = {expected}, got {t_half}"
            );
        } else {
            panic!("expected PerRow for T_HALF");
        }
    }

    #[test]
    fn parse_derived_max_with_filter() {
        let src = minimal_model_with_derived("CMAX = max(IPRED, TIME < 24)");
        let parsed = parse_full_model(&src).expect("parse ok");
        assert!(matches!(
            parsed.model.derived_exprs[0].kind,
            DerivedKind::Aggregate {
                func: AggFunction::Max,
                filter: Some(_),
                ..
            }
        ));
    }

    #[test]
    fn parse_derived_min_no_filter() {
        let src = minimal_model_with_derived("CMIN = min(IPRED)");
        let parsed = parse_full_model(&src).expect("parse ok");
        assert!(matches!(
            parsed.model.derived_exprs[0].kind,
            DerivedKind::Aggregate {
                func: AggFunction::Min,
                filter: None,
                ..
            }
        ));
    }

    #[test]
    fn parse_derived_tmax() {
        let src = minimal_model_with_derived("TMAX = tmax(IPRED)");
        let parsed = parse_full_model(&src).expect("parse ok");
        assert!(matches!(
            parsed.model.derived_exprs[0].kind,
            DerivedKind::Aggregate {
                func: AggFunction::Tmax,
                filter: None,
                ..
            }
        ));
    }

    #[test]
    fn parse_derived_integral_explicit() {
        let src = minimal_model_with_derived("AUC = integral(IPRED, from=0, to=24)");
        let parsed = parse_full_model(&src).expect("parse ok");
        if let DerivedKind::Integral {
            data_based,
            window: IntegralWindow::Explicit { from, to },
            step: IntegralStep::Auto,
            ..
        } = &parsed.model.derived_exprs[0].kind
        {
            assert!(!data_based, "IPRED is not DV-based");
            assert!((from - 0.0).abs() < 1e-10);
            assert!((to - 24.0).abs() < 1e-10);
        } else {
            panic!(
                "expected Integral(Explicit, Auto), got {:?} kind",
                parsed.model.derived_exprs[0].name
            );
        }
    }

    #[test]
    fn parse_derived_integral_dv() {
        let src = minimal_model_with_derived("AUC_DV = integral(DV, from=0, to=24)");
        let parsed = parse_full_model(&src).expect("parse ok");
        if let DerivedKind::Integral {
            data_based,
            step: IntegralStep::ObsTimes,
            ..
        } = &parsed.model.derived_exprs[0].kind
        {
            assert!(*data_based, "DV is data_based");
        } else {
            panic!("expected Integral with ObsTimes step for DV integrand");
        }
    }

    #[test]
    fn parse_derived_integral_periodic() {
        let src = minimal_model_with_derived("AUC_TAU = integral(IPRED, window=24, anchor=0)");
        let parsed = parse_full_model(&src).expect("parse ok");
        if let DerivedKind::Integral {
            window: IntegralWindow::Periodic { period, anchor },
            ..
        } = &parsed.model.derived_exprs[0].kind
        {
            assert!((period - 24.0).abs() < 1e-10);
            assert!((anchor - 0.0).abs() < 1e-10);
        } else {
            panic!("expected Integral(Periodic)");
        }
    }

    #[test]
    fn parse_derived_name_conflict_error() {
        // IPRED is a built-in sdtab column — should error
        let src = minimal_model_with_derived("IPRED = CL / V");
        let result = parse_full_model(&src);
        assert!(
            result.is_err(),
            "expected parse error for IPRED name conflict"
        );
        let msg = match result {
            Err(e) => e,
            Ok(_) => panic!("expected Err"),
        };
        assert!(
            msg.contains("E_DERIVED_NAME_CONFLICT"),
            "expected E_DERIVED_NAME_CONFLICT in error, got: {msg}"
        );
    }

    #[test]
    fn parse_output_block() {
        let src = format!(
            r#"{}
[output]
CL V KA WT
"#,
            minimal_model_with_derived("")
        );
        let parsed = parse_full_model(&src).expect("parse ok");
        assert_eq!(
            parsed.model.output_columns,
            vec!["CL", "V", "KA", "WT"],
            "output_columns mismatch"
        );
    }

    #[test]
    fn parse_output_empty_block_ok() {
        let src = format!(
            r#"{}
[output]
"#,
            minimal_model_with_derived("")
        );
        let parsed = parse_full_model(&src).expect("parse ok");
        assert!(
            parsed.model.output_columns.is_empty(),
            "empty [output] block should produce empty output_columns"
        );
    }

    #[test]
    fn sci_notation_negative_exp() {
        let src = minimal_model_with_derived("FLAG = if (TAD < 1e-10) 1 else 0");
        let parsed = parse_full_model(&src).expect("parse ok — 1e-10 must tokenise correctly");
        assert_eq!(parsed.model.derived_exprs.len(), 1);
    }

    #[test]
    fn sci_notation_positive_exp() {
        let src = minimal_model_with_derived("FLAG = if (TAFD > 1.5E+3) 1 else 0");
        let parsed = parse_full_model(&src).expect("parse ok — 1.5E+3 must tokenise correctly");
        assert_eq!(parsed.model.derived_exprs.len(), 1);
    }

    #[test]
    fn mod_operator_euclidean() {
        // 5 mod 2 == 1, 7 mod 3 == 1, -1 mod 24 == 23
        let tests = [("5 mod 2", 1.0), ("7 mod 3", 1.0), ("-1 mod 24", 23.0)];
        for (expr_str, expected) in &tests {
            let expr_src = format!("VAL = {expr_str}");
            let src = minimal_model_with_derived(&expr_src);
            let parsed = parse_full_model(&src).expect("parse ok");
            let (theta, eta, indiv, cov, prev) = make_derived_ctx_simple();
            if let DerivedKind::PerRow { eval } = &parsed.model.derived_exprs[0].kind {
                let ctx = DerivedContext {
                    theta: &theta,
                    eta: &eta,
                    indiv_params: &indiv,
                    covariates: &cov,
                    ipred: 0.0,
                    pred: 0.0,
                    dv: 0.0,
                    time: 0.0,
                    tafd: 0.0,
                    tad: 0.0,
                    prev_derived: &prev,
                    compartments: &[],
                    compartment_names: &[],
                };
                let result = eval(&ctx);
                assert!(
                    (result - expected).abs() < 1e-10,
                    "{expr_str}: expected {expected}, got {result}"
                );
            }
        }
    }

    #[test]
    fn floor_ceil_round_functions() {
        let tests = [
            ("floor(-2.3)", -3.0),
            ("ceil(-2.3)", -2.0),
            ("round(2.5)", 3.0),
        ];
        for (expr_str, expected) in &tests {
            let src = minimal_model_with_derived(&format!("VAL = {expr_str}"));
            let parsed = parse_full_model(&src).expect("parse ok");
            let (theta, eta, indiv, cov, prev) = make_derived_ctx_simple();
            if let DerivedKind::PerRow { eval } = &parsed.model.derived_exprs[0].kind {
                let ctx = DerivedContext {
                    theta: &theta,
                    eta: &eta,
                    indiv_params: &indiv,
                    covariates: &cov,
                    ipred: 0.0,
                    pred: 0.0,
                    dv: 0.0,
                    time: 0.0,
                    tafd: 0.0,
                    tad: 0.0,
                    prev_derived: &prev,
                    compartments: &[],
                    compartment_names: &[],
                };
                let result = eval(&ctx);
                assert!(
                    (result - expected).abs() < 1e-10,
                    "{expr_str}: expected {expected}, got {result}"
                );
            }
        }
    }

    #[test]
    fn macheps_available_in_derived() {
        let src = minimal_model_with_derived("FLAG = if (TAD < MACHEPS) 1 else 0");
        let parsed = parse_full_model(&src).expect("parse ok");
        let (theta, eta, indiv, cov, _prev) = make_derived_ctx_simple();
        let prev = std::collections::HashMap::new();
        if let DerivedKind::PerRow { eval } = &parsed.model.derived_exprs[0].kind {
            // TAD = 0.0 < MACHEPS → flag = 1
            let ctx = DerivedContext {
                theta: &theta,
                eta: &eta,
                indiv_params: &indiv,
                covariates: &cov,
                ipred: 0.0,
                pred: 0.0,
                dv: 0.0,
                time: 1.0,
                tafd: 1.0,
                tad: 0.0, // zero
                prev_derived: &prev,
                compartments: &[],
                compartment_names: &[],
            };
            assert_eq!(eval(&ctx), 1.0, "TAD=0 should be < MACHEPS");
            // TAD = 1.0 >> MACHEPS → flag = 0
            let ctx2 = DerivedContext { tad: 1.0, ..ctx };
            assert_eq!(eval(&ctx2), 0.0, "TAD=1 should not be < MACHEPS");
        }
    }

    #[test]
    fn ode_compartment_indexed_dose_attrs_populate_map() {
        // `F2` / `ALAG2` in a 2-compartment ODE model must (a) parse without
        // tripping the dead-parameter census — they are engine-applied dose
        // attributes, not dead structural params — and (b) populate the
        // `dose_attr_map` and enable `has_lagtime()` (#369).
        let src = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TVF2(0.6, 0.01, 1.0)
  theta TVLAG2(0.4, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  F2 = TVF2
  ALAG2 = TVLAG2

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -CL/V * depot
  d/dt(central) =  CL/V * depot - CL/V * central

[error_model]
  DV ~ proportional(PROP)
"#;
        let parsed = parse_full_model(src).expect("parse ok");
        let map = &parsed
            .model
            .ode_spec
            .as_ref()
            .expect("ode spec")
            .dose_attr_map;
        assert!(
            map.indexed_slot(crate::types::DoseAttr::F, 2).is_some(),
            "F2 must map for compartment 2"
        );
        assert!(
            map.indexed_slot(crate::types::DoseAttr::Lag, 2).is_some(),
            "ALAG2 must map for compartment 2"
        );
        // No compartment-1 override was declared.
        assert!(map.indexed_slot(crate::types::DoseAttr::F, 1).is_none());
        assert!(
            parsed.model.has_lagtime(),
            "ALAG2 must enable has_lagtime() so downstream lag handling runs"
        );
    }

    #[test]
    fn ode_dose_attr_compartment_out_of_range_errors() {
        // `F3` references compartment 3, but the model has only 2 states — a loud
        // parse error, never a silently-ignored spare slot (#369).
        let src = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TVF3(0.6, 0.01, 1.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  F3 = TVF3

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -CL/V * depot
  d/dt(central) =  CL/V * depot - CL/V * central

[error_model]
  DV ~ proportional(PROP)
"#;
        let err = parse_full_model(src)
            .err()
            .expect("F3 with 2 compartments must error");
        assert!(
            err.contains("compartment 3") && err.contains("F3"),
            "error must name the attribute and compartment, got: {err}"
        );
    }

    #[test]
    fn derived_resolves_covariate_case_insensitively() {
        // Regression: a covariate carried in the dataset under an uppercase
        // header (`WT`) referenced as lowercase `wt` in a [derived] expression
        // must resolve to the covariate value, not silently evaluate to 0.
        // build_derived_vars must insert a lowercase alias because
        // `eval_expression` looks the name up verbatim.
        let src = minimal_model_with_derived("WT_DERIVED = wt * 2");
        let parsed = parse_full_model(&src).expect("parse ok");
        let (theta, eta, indiv, _cov, prev) = make_derived_ctx_simple();
        // Covariate stored under uppercase `WT`, as a NONMEM header would be.
        let mut cov = std::collections::HashMap::new();
        cov.insert("WT".to_string(), 70.0);
        if let DerivedKind::PerRow { eval } = &parsed.model.derived_exprs[0].kind {
            let ctx = DerivedContext {
                theta: &theta,
                eta: &eta,
                indiv_params: &indiv,
                covariates: &cov,
                ipred: 0.0,
                pred: 0.0,
                dv: 0.0,
                time: 0.0,
                tafd: 0.0,
                tad: 0.0,
                prev_derived: &prev,
                compartments: &[],
                compartment_names: &[],
            };
            assert_eq!(
                eval(&ctx),
                140.0,
                "lowercase `wt` must resolve to covariate header `WT` (=70)"
            );
        } else {
            panic!("expected PerRow derived kind");
        }
    }

    /// Regression: named analytical compartment references (e.g. `central`,
    /// `depot`) in a [derived] expression on an analytical model must set
    /// `uses_compartments = true` so that W_DERIVED_CMT_* warnings fire.
    ///
    /// Before the fix, `ode_state_names` was empty for analytical models, so
    /// `expr_refs_compartments` never detected named access — all four
    /// W_DERIVED_CMT_* guards were silently suppressed.
    #[test]
    fn analytical_model_named_compartment_sets_uses_compartments() {
        // one_cpt_iv has a single compartment named "central"
        let src = minimal_model_with_derived("C_CENTRAL = central");
        let parsed = parse_full_model(&src).expect("parse ok");
        let expr = &parsed.model.derived_exprs[0];
        assert_eq!(expr.name, "C_CENTRAL");
        assert!(
            expr.uses_compartments,
            "`central` in [derived] on a one_cpt_iv model must set uses_compartments=true"
        );

        // two_cpt_oral has depot, central, peripheral
        let src2 = r#"
[parameters]
  theta CL(1.0, 0, 100)
  theta V1(10.0, 0, 1000)
  theta Q(0.5, 0, 50)
  theta V2(5.0, 0, 500)
  theta KA(1.0, 0, 10)
  omega ETA_CL ~ 0.09
  sigma PROP   ~ 0.01
[individual_parameters]
  CL = exp(log(CL) + ETA_CL)
  V1 = V1
  Q  = Q
  V2 = V2
  KA = KA
[structural_model]
  pk two_cpt_oral(cl=CL, v=V1, q=Q, v2=V2, ka=KA)
[error_model]
  DV ~ proportional(PROP)
[derived]
  C_PERIPH = peripheral
"#;
        let parsed2 = parse_full_model(src2).expect("parse ok (two_cpt_oral)");
        let expr2 = &parsed2.model.derived_exprs[0];
        assert_eq!(expr2.name, "C_PERIPH");
        assert!(
            expr2.uses_compartments,
            "`peripheral` in [derived] on a two_cpt_oral model must set uses_compartments=true"
        );
    }

    /// Regression: `compartments[N]` with N > MAX_CMT_INDEX (255) must fail at
    /// parse time rather than silently evaluating to 0.0.
    ///
    /// MAX_CMT_INDEX is the module-level constant; `build_derived_vars` seeds
    /// `__cmt_0..=__cmt_{MAX_CMT_INDEX}` with NaN. Any index beyond that misses
    /// the map and `.unwrap_or(0.0)` would return 0.0 — a plausible drug
    /// concentration masking the out-of-range access.
    #[test]
    fn compartments_index_out_of_range_is_parse_error() {
        let src = minimal_model_with_derived("X = compartments[256]");
        match parse_full_model(&src) {
            Err(err) => assert!(
                err.contains("256") && err.contains("exceeds"),
                "error should mention the index and 'exceeds': {err}"
            ),
            Ok(_) => panic!("compartments[256] should be a parse error"),
        }

        // 255 is the last valid index — must succeed
        let src_ok = minimal_model_with_derived("X = compartments[255]");
        parse_full_model(&src_ok).expect("compartments[255] should parse ok");

        // 0 must still work
        let src_zero = minimal_model_with_derived("X = compartments[0]");
        parse_full_model(&src_zero).expect("compartments[0] should parse ok");
    }

    /// Regression: `uses_compartments` on `integral()` must also fire when the
    /// **condition** (filter) clause references a compartment, not just the
    /// integrand. Before the fix, `integral(IPRED, compartments[0] > 1, from=0,
    /// to=24)` had `uses_compartments = false` because only the integrand was
    /// checked — the filter closure silently received NaN for all `__cmt_N` keys,
    /// and the integral silently integrated over the wrong set of observations.
    #[test]
    fn integral_condition_referencing_compartment_sets_uses_compartments() {
        // Integrand is IPRED (no compartment ref); condition references compartments[0].
        let src = minimal_model_with_derived(
            "AUC_FILTERED = integral(IPRED, compartments[0] > 1.0, from=0, to=24)",
        );
        let parsed = parse_full_model(&src).expect("parse ok");
        let expr = &parsed.model.derived_exprs[0];
        assert_eq!(expr.name, "AUC_FILTERED");
        assert!(
            expr.uses_compartments,
            "integral() condition referencing compartments[0] must set uses_compartments=true"
        );

        // Sanity check: integrand-only reference still works
        let src2 = minimal_model_with_derived("AUC_CMT = integral(compartments[0], from=0, to=24)");
        let parsed2 = parse_full_model(&src2).expect("parse ok");
        assert!(
            parsed2.model.derived_exprs[0].uses_compartments,
            "integral() integrand referencing compartments[0] must still set uses_compartments"
        );

        // Sanity check: no compartment reference in either integrand or condition → false
        let src3 =
            minimal_model_with_derived("AUC_PLAIN = integral(IPRED, DV > 0.0, from=0, to=24)");
        let parsed3 = parse_full_model(&src3).expect("parse ok");
        assert!(
            !parsed3.model.derived_exprs[0].uses_compartments,
            "integral() with no compartment reference must leave uses_compartments=false"
        );
    }
}
