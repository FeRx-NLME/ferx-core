use crate::sim::adaptive::{
    validate_increasing_finite, AdaptiveAction, AdaptiveDosingSpec, AdaptiveRoute, AdaptiveRule,
    Comparison, DoseStep,
};
use crate::types::*;
use regex::Regex;
use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;
use std::sync::LazyLock;

static DIFFUSION_LINE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^(\w+)\s*~\s*([0-9eE.+-]+)(?:\s+(FIX)\b)?").unwrap());

/// Identifier scanner for `[error_model]` argument text. Hoisted out of the
/// functions that use it because compiling the regex costs far more than the
/// scan it performs, and it runs on every parse: `arg_sigma_name` calls it once
/// per argument (from both `build_error_spec` and `build_ruv_magnitude`) and
/// `used_sigma_names` scans every argument of every `[error_model]` form. Every
/// identifier scan over `[error_model]` text goes through this one static — add
/// new callers here rather than compiling a second copy.
static ARG_IDENT_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[A-Za-z_]\w*").unwrap());

/// Whether an `[error_model]` argument is a bare identifier (so an unresolved one
/// is a typo'd sigma name) rather than a #484 magnitude expression.
static BARE_IDENT_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[A-Za-z_]\w*$").unwrap());

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
    expr_references_any(expr, kappa_names)
}

/// First name in `names` referenced anywhere in `expr` (as a bare `Variable` or
/// `Covariate` leaf), in walk order — or `None`. The generic core behind
/// [`expr_references_kappa`] and the analytic Form C peripheral-rejection scan
/// (issue #650), so both agree on what "references a forbidden name" means.
fn expr_references_any(expr: &Expression, names: &[String]) -> Option<String> {
    fn walk(e: &Expression, names: &[String]) -> Option<String> {
        match e {
            Expression::Variable(n) | Expression::Covariate(n) => {
                names.iter().find(|k| *k == n).cloned()
            }
            Expression::BinOp(l, _, r) => walk(l, names).or_else(|| walk(r, names)),
            Expression::UnaryFn(_, a) => walk(a, names),
            Expression::Power(b, e) => walk(b, names).or_else(|| walk(e, names)),
            Expression::Conditional(cond, t, els) => walk_cond(cond, names)
                .or_else(|| walk(t, names))
                .or_else(|| walk(els, names)),
            _ => None,
        }
    }
    fn walk_cond(c: &Condition, names: &[String]) -> Option<String> {
        match c {
            Condition::Compare(l, _, r) => walk(l, names).or_else(|| walk(r, names)),
            Condition::And(l, r) | Condition::Or(l, r) => {
                walk_cond(l, names).or_else(|| walk_cond(r, names))
            }
            Condition::Not(c) => walk_cond(c, names),
        }
    }
    walk(expr, names)
}

/// Canonical compartment state names for an **analytic** Form C readout (issue
/// #650), and the peripheral/absorption names that are *rejected* in that scope.
///
/// Returns `(allowed, forbidden)`:
/// - `allowed` is the `state[]` layout the predictor reconstructs amounts into —
///   `["central"]` for IV models, `["depot", "central"]` for first-order oral
///   models (where the depot amount has the closed form `F·D·exp(-ka·t)`).
/// - `forbidden` lists compartment-ish names that have no closed-form amount with
///   cross-compartment sensitivity: every peripheral, plus `depot` on models that
///   are not first-order oral (IV models have none; the transit "depot" is a
///   lumped Gamma-convolution memory with no seeded-amount closed form — #386).
///   Referencing any of these fails the parse with a pointer to an ODE model,
///   mirroring the `[initial_conditions]` scope decision (`analytical_init_cmt`).
fn analytic_readout_state_names(pk_model: PkModel) -> (Vec<String>, Vec<String>) {
    let depot_ok = matches!(
        pk_model,
        PkModel::OneCptOral | PkModel::TwoCptOral | PkModel::ThreeCptOral
    );
    let allowed: Vec<String> = if depot_ok {
        vec!["depot".into(), "central".into()]
    } else {
        vec!["central".into()]
    };
    let mut forbidden: Vec<String> = vec![
        "peripheral".into(),
        "periph".into(),
        "peripheral1".into(),
        "periph1".into(),
        "peripheral2".into(),
        "periph2".into(),
    ];
    if !depot_ok {
        // `depot` is not a valid amount here (no closed form): reject it loudly
        // rather than let it fall through to a silent covariate lookup.
        forbidden.push("depot".into());
    }
    (allowed, forbidden)
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

/// Match a `MIXNUM == <k>` (or `<k> == MIXNUM`) condition, returning the
/// 1-based class index `k`. Any other condition — a different comparison
/// operator, a compound `and`/`or`/`not`, or a non-`MIXNUM` operand — returns
/// `None`, which makes the class-aware mu-ref detector reject the whole chain
/// and fall back to the numerical M-step (#996).
fn mixnum_eq_class(cond: &Condition) -> Option<usize> {
    let Condition::Compare(lhs, CmpOp::Eq, rhs) = cond else {
        return None;
    };
    let lit = match (lhs, rhs) {
        (Expression::MixNum, Expression::Literal(v)) => *v,
        (Expression::Literal(v), Expression::MixNum) => *v,
        _ => return None,
    };
    // MIXNUM is an integer index; a non-integral literal can never match.
    if lit.fract() != 0.0 || lit < 1.0 {
        return None;
    }
    Some(lit as usize)
}

/// Detect a `MIXNUM`-switched log-mu-reference (#996).
///
/// Recognises a chain of if-*expressions* whose conditions are all
/// `MIXNUM == k` and whose arms all match the same mu-ref pattern on the same
/// eta with the same log/additive flag, e.g.
///
/// ```text
/// CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
/// CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL)
///      else if (MIXNUM == 2) TVCL2 * exp(ETA_CL)
///      else TVCL3 * exp(ETA_CL)
/// ```
///
/// Returns `(eta_idx, theta_idx_per_class, log_transformed)` where
/// `theta_idx_per_class[c]` is the anchor for class `c + 1`. The trailing
/// `else` supplies every class the chain did not name explicitly, so a
/// three-class model whose chain names only class 1 collapses classes 2 and 3
/// onto the trailing-`else` theta. A class named twice keeps the *first* arm,
/// matching evaluation order.
///
/// Returns `None` — i.e. "no class-aware mu-ref, use the numerical M-step" —
/// for anything else: a non-`MIXNUM` condition, an out-of-range class index,
/// an arm that is not a mu-ref pattern, arms on different etas, arms mixing
/// log and additive forms, or an NN-anchored arm.
fn detect_mixture_pattern(
    expr: &Expression,
    n_classes: usize,
) -> Option<(usize, Vec<usize>, bool)> {
    // The expression must actually be a conditional — a plain `THETA*exp(ETA)`
    // is the classical (class-shared) mu-ref and is handled by `detect_mu_refs`.
    if !matches!(expr, Expression::Conditional(..)) {
        return None;
    }
    let mut by_class: Vec<Option<usize>> = vec![None; n_classes];
    let mut eta: Option<usize> = None;
    let mut log_t: Option<bool> = None;

    // Validate one arm and fold its (eta, log_transformed) into the running
    // consistency check. Returns the arm's anchor theta index.
    fn check_arm(
        arm: &Expression,
        eta: &mut Option<usize>,
        log_t: &mut Option<bool>,
    ) -> Option<usize> {
        let (ei, anchor, lt) = detect_pattern(arm)?;
        let MuRefAnchor::Theta(ti) = anchor else {
            // An NN-anchored class arm has no theta to shift.
            return None;
        };
        if *eta.get_or_insert(ei) != ei {
            return None;
        }
        if *log_t.get_or_insert(lt) != lt {
            return None;
        }
        Some(ti)
    }

    let mut cur = expr;
    loop {
        match cur {
            Expression::Conditional(cond, then_e, else_e) => {
                let k = mixnum_eq_class(cond)?;
                if k > n_classes {
                    return None;
                }
                let ti = check_arm(then_e.as_ref(), &mut eta, &mut log_t)?;
                // First arm naming a class wins, as at eval time.
                if by_class[k - 1].is_none() {
                    by_class[k - 1] = Some(ti);
                }
                cur = else_e.as_ref();
            }
            terminal => {
                let ti = check_arm(terminal, &mut eta, &mut log_t)?;
                // The trailing `else` serves every class the chain never named.
                for slot in by_class.iter_mut() {
                    if slot.is_none() {
                        *slot = Some(ti);
                    }
                }
                break;
            }
        }
    }
    let thetas: Vec<usize> = by_class.into_iter().collect::<Option<Vec<_>>>()?;
    Some((eta?, thetas, log_t?))
}

/// Scan `[individual_parameters]` for class-aware mu-references (#996).
///
/// Only top-level (unconditional) assignments participate, exactly as in
/// [`detect_mu_refs`] — the class switch lives in the *expression*, not in a
/// `Statement::If`. Non-log patterns are dropped here rather than downstream so
/// the returned list is directly consumable by the SAEM / IMP closed-form
/// shift, which is only valid on the log scale.
///
/// An eta assigned more than once keeps the **last** anchor, matching
/// `detect_mu_refs`' `result.insert`. The scan is over *both* pattern kinds, not
/// just the class-aware ones: an ordinary (non-switched) mu-ref written after a
/// class-aware one wins, and clears the class-aware entry. Without that,
/// `MixtureSpec::mu_refs` and `CompiledModel::mu_refs` could name different
/// anchors for one eta, and `get_mixture_mu_ref_pairs` — which prefers the spec
/// — would silently disagree with `compute_mu_k` and every other `mu_refs`
/// consumer (#996 review).
fn detect_mixture_mu_refs(
    stmts: &[Statement],
    theta_names: &[String],
    eta_names: &[String],
    n_classes: usize,
) -> Vec<crate::types::MixtureMuRef> {
    // `None` records "the last anchor for this eta was an ordinary mu-ref", which
    // `detect_mu_refs` already stores; the spec must then carry no entry for it.
    let mut latest: Vec<(String, Option<crate::types::MixtureMuRef>)> = Vec::new();
    let mut set = |eta: &str, entry: Option<crate::types::MixtureMuRef>| match latest
        .iter_mut()
        .find(|(name, _)| name == eta)
    {
        Some(slot) => slot.1 = entry,
        None => latest.push((eta.to_string(), entry)),
    };
    for s in stmts {
        let Statement::Assign(_, expr) = s else {
            continue;
        };
        if let Some((eta_idx, theta_idx, log_transformed)) = detect_mixture_pattern(expr, n_classes)
        {
            if !log_transformed || eta_idx >= eta_names.len() {
                continue;
            }
            if theta_idx.iter().any(|&t| t >= theta_names.len()) {
                continue;
            }
            let eta_name = eta_names[eta_idx].clone();
            let entry = crate::types::MixtureMuRef {
                eta_name: eta_name.clone(),
                theta_names: theta_idx.iter().map(|&t| theta_names[t].clone()).collect(),
                log_transformed,
            };
            set(&eta_name, Some(entry));
        } else if let Some((eta_idx, _anchor, _log)) = detect_pattern(expr) {
            // An ordinary mu-ref on the same eta, later in the block: it is what
            // `detect_mu_refs` will keep, so drop any class-aware entry.
            if eta_idx < eta_names.len() {
                let eta_name = eta_names[eta_idx].clone();
                set(&eta_name, None);
            }
        }
    }
    latest.into_iter().filter_map(|(_, entry)| entry).collect()
}

/// Whether a top-level `[individual_parameters]` assignment reads `MIXNUM`
/// anywhere in its right-hand side. Used to warn when a class-switched typical
/// value could *not* be class-aware mu-referenced (#996) — without this the
/// fallback to the purely numerical M-step is silent.
fn expr_uses_mixnum(expr: &Expression) -> bool {
    fn cond_uses(c: &Condition) -> bool {
        match c {
            Condition::Compare(a, _, b) => expr_uses_mixnum(a) || expr_uses_mixnum(b),
            Condition::And(a, b) | Condition::Or(a, b) => cond_uses(a) || cond_uses(b),
            Condition::Not(a) => cond_uses(a),
        }
    }
    match expr {
        Expression::MixNum => true,
        Expression::BinOp(a, _, b) | Expression::Power(a, b) => {
            expr_uses_mixnum(a) || expr_uses_mixnum(b)
        }
        Expression::UnaryFn(_, a) => expr_uses_mixnum(a),
        Expression::Conditional(c, t, e) => {
            cond_uses(c) || expr_uses_mixnum(t) || expr_uses_mixnum(e)
        }
        _ => false,
    }
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
                // Parser-internal readout parameters (#486) are not user parameterisations and
                // must not produce an `EtaParamInfo`. `__ferx_ro_eta{k} = ETA(k)` is a bare
                // `Expression::Eta`, which `classify_expr` has no pattern for, so it would fall
                // to the `Custom` arm below and surface in `FitResult.eta_param_info` (hence
                // ferx-r's parameter table) under a name the user never wrote. Worse, a single
                // `Custom` entry flips `resolve_param_derivs` / `subject_eta_grad`'s
                // "all LogNormal" fallback to `None`, dropping the whole model to FD whenever
                // the program path is unavailable. Applies to both engines: the ODE half of
                // this desugaring has appended such statements since #631, and
                // `classify_indiv_params` runs after both append sites (PR #950 review #4).
                if is_synthetic_readout_param(param_name) {
                    continue;
                }

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
    let mut parsed = parse_full_model(&content)?;
    // A relative `[data] path` is relative to the model file's own directory
    // (mirrors NONMEM's `$DATA` resolution against the control stream), not
    // the process's current working directory.
    if let Some(data_path) = &parsed.data_path {
        let p = std::path::Path::new(data_path);
        if p.is_relative() {
            if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
                parsed.data_path = Some(dir.join(p).to_string_lossy().into_owned());
            }
        }
    }
    Ok(parsed)
}

/// Parse a model string and return a CompiledModel (backward compatible).
pub fn parse_model_string(content: &str) -> Result<CompiledModel, String> {
    let parsed = parse_full_model(content)?;
    Ok(parsed.model)
}

/// Register covariate names as required data columns: push each name not
/// already present, then re-sort. Shared by the `[scaling]`, error-model
/// selector, and `[initial_conditions]` blocks so all three declare their
/// covariates identically — the divergence that let an unregistered init
/// covariate silently evaluate to 0 (issue #765).
fn register_referenced_covariates(
    referenced: &mut Vec<String>,
    covs: impl IntoIterator<Item = String>,
) {
    for cov in covs {
        if !referenced.contains(&cov) {
            referenced.push(cov);
        }
    }
    referenced.sort();
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
    // Prepare the absorption ODE-equivalent (#486; IG #790): the transit / IG closed forms
    // assume constant parameters over each absorption window, so they cannot serve a subject
    // whose parameters switch mid-profile (a `TIME`-dependent parameter or time-varying
    // covariates). For a plain-form transit / IG model we reconstruct its exact ODE
    // (`transit()` / `igd()`) equivalent *source* here; it is compiled at the attach site
    // below (once the primary's own blocks have parsed, so a broken primary reports its own
    // error first), and `CompiledModel::effective_for` routes only the subjects that need it
    // to this fallback. Non-absorption / out-of-scope forms yield `None`. (The equivalent is
    // a normal ODE model with no `pk one_cpt_transit`/`pk one_cpt_ig`/…, so compiling it does
    // not re-enter this branch.)
    let absorption_ode_equivalent_src: Option<String> =
        absorption_ode_equivalent_source(&extracted);
    // Keep the historical `blocks` binding for unnamed blocks so the rest of
    // this (large) function reads unchanged. Named blocks are pulled from
    // `extracted.named` directly where they're consumed below.
    let blocks = &extracted.unnamed;
    let name = extract_model_name(content);

    // ── Required blocks ──
    let param_lines = blocks
        .get("parameters")
        .ok_or("Missing [parameters] block")?;
    let (thetas, omegas, block_omegas, sigmas, block_sigmas, eta_names_bsv, kappa_info) =
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

    // A `[binary_model]` (Phase 4 categorical / logistic, #760) endpoint likewise makes
    // the Gaussian PK blocks optional — a logistic-only model needs none of them.
    #[cfg(feature = "survival")]
    let has_binary_model_block = blocks.contains_key("binary_model")
        || extracted
            .named
            .get("binary_model")
            .is_some_and(|m| !m.is_empty());
    #[cfg(not(feature = "survival"))]
    let has_binary_model_block = false;

    // A `[markov_model]` (Phase 5 CTMM, #759) endpoint is likewise a non-Gaussian
    // endpoint that makes the Gaussian PK blocks optional.
    #[cfg(feature = "markov")]
    let has_markov_model_block = blocks.contains_key("markov_model")
        || extracted
            .named
            .get("markov_model")
            .is_some_and(|m| !m.is_empty());
    #[cfg(not(feature = "markov"))]
    let has_markov_model_block = false;

    // Fold the CTMM block into the same "endpoint present" predicate the binary
    // block uses, so a CTMM-only model may omit the Gaussian PK blocks / empty Ω.
    let has_binary_model_block = has_binary_model_block || has_markov_model_block;

    let struct_lines_opt = blocks.get("structural_model");
    let error_lines_opt = blocks.get("error_model");
    let indiv_lines_opt = blocks.get("individual_parameters");
    // All three Gaussian blocks must be absent together for a valid TTE-only model.
    // Partial omission (e.g. [structural_model] present but no [individual_parameters])
    // would create an invalid mixed-model state.
    let is_endpoint_only = (has_event_model_block || has_binary_model_block)
        && struct_lines_opt.is_none()
        && error_lines_opt.is_none()
        && indiv_lines_opt.is_none();
    if struct_lines_opt.is_none() && !is_endpoint_only {
        return Err("Missing [structural_model] block".to_string());
    }
    let struct_lines: &[String] = struct_lines_opt.map(Vec::as_slice).unwrap_or(&[]);

    if error_lines_opt.is_none() && !is_endpoint_only {
        return Err("Missing [error_model] block".to_string());
    }
    let (parsed_error_model, ltbs_flags, iiv_on_ruv_name) =
        if let Some(error_lines) = error_lines_opt {
            parse_error_model(error_lines)?
        } else {
            // TTE-only model: no Gaussian error model — empty per-CMT spec.
            (ParsedErrorModel::PerCmt(vec![]), LtbsFlags::default(), None)
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

    if indiv_lines_opt.is_none() && !is_endpoint_only {
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
    let mut seen_sigma_names = std::collections::HashSet::new();
    for name in &sigma_names {
        if !seen_sigma_names.insert(name.as_str()) {
            return Err(format!(
                "sigma '{}' is declared more than once (including block_sigma entries)",
                name
            ));
        }
    }
    let n_theta;
    let n_eta = eta_names_bsv.len(); // BSV-only count
    let n_kappa = kappa_info.names_ordered.len();
    let n_epsilon = sigma_names.len();

    // Resolve `iiv_on_ruv = NAME` to a BSV eta index. The named eta must be a
    // declared `omega` (BSV), not a kappa (IOV residual scaling is out of scope).
    let residual_error_eta: Option<usize> = match &iiv_on_ruv_name {
        Some(name) => {
            let idx = eta_names_bsv
                .iter()
                .position(|e| e == name)
                .ok_or_else(|| {
                    format!(
                        "[error_model] iiv_on_ruv = {name}: no `omega {name} ~ ...` declared in \
                     [parameters] (the residual-error random effect must be a declared omega)"
                    )
                })?;
            Some(idx)
        }
        None => None,
    };

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
    let mut indiv_var_names: Vec<String> = unconditionally_assigned_vars(&pre_stmts)
        .into_iter()
        .filter(|n| {
            top_level_names.iter().any(|t| t == n)
                || downstream_refs.contains(&n.to_ascii_uppercase())
        })
        .collect();
    let indiv_ctx =
        ParseCtx::new(&theta_names, &eta_names, &all_assigned).with_nn_specs(&nn_specs_for_ctx);
    let mut indiv_stmts = parse_block_statements(&indiv_text, indiv_ctx, StatementMode::Plain)?;

    // Reject a forward reference — a name read before it is assigned in file
    // order — rather than let it silently resolve to 0.0 at eval time (#710).
    let in_block: HashSet<String> = all_assigned.iter().cloned().collect();
    let forward_refs = collect_forward_refs(&indiv_stmts, &in_block);
    if !forward_refs.is_empty() {
        let mut seen: HashSet<(String, String)> = HashSet::new();
        let mut msgs: Vec<String> = Vec::new();
        for (used, defining) in &forward_refs {
            if seen.insert((used.clone(), defining.clone())) {
                msgs.push(format!("`{used}` (used in `{defining}`)"));
            }
        }
        return Err(format!(
            "[individual_parameters]: forward reference(s) to name(s) declared later \
             in the block: {}. Statements are evaluated in file order, so a name must \
             be assigned before it is used; a forward reference would otherwise \
             silently read 0.0. Reorder the block so each name is declared before it \
             is referenced.",
            msgs.join(", "),
        ));
    }

    // ── Sample-size-weighted IOV (#1031) ────────────────────────────────────
    // `kappa K ~ γ² weight = W` declares `κ_ik ~ N(0, Ω_IOV / W_ik)` — the
    // between-treatment-arm variability of a longitudinal MBMA, whose arm-level
    // effect scales with the number of subjects behind the arm.
    //
    // The engine applies it by *reparameterisation*, not by scaling the prior:
    // every reference to a weighted kappa in `[individual_parameters]` is
    // rewritten `K` → `K / sqrt(W)`. With `κ ~ N(0, γ²)` the rewritten effect
    // `κ/√W` is distributed exactly `N(0, γ²/W)`, so the two formulations are
    // the same likelihood — and the rewritten model is precisely the
    // hand-written form this modifier replaces, which every estimator, analytic
    // sensitivity, and diagnostic already supports. Two things fall out of that
    // choice for free: the estimated Ω_IOV stays the *unweighted* γ² a published
    // analysis reports, and the κ̂ EBEs (and their shrinkage, which compares
    // Var(κ̂) against Ω_IOV) stay on Ω_IOV's own scale.
    //
    // The rewrite runs here — before the covariate collection, the pk/tv
    // closures, and the ODE/scaling compilers all consume `indiv_stmts` — so a
    // covariate named only by a weight expression is registered like any other
    // (the #484/#1029 time-varying-snapshot trap) with no extra plumbing.
    let kappa_weight_exprs: Vec<Option<Expression>> = {
        let mut out: Vec<Option<Expression>> = Vec::with_capacity(kappa_names.len());
        for (k, src) in kappa_info.weights.iter().enumerate() {
            let Some(src) = src else {
                out.push(None);
                continue;
            };
            let name = kappa_names.get(k).map(String::as_str).unwrap_or("<kappa>");
            // No `defined_vars`: a weight is evaluated at the point of *use*, so
            // letting it name an `[individual_parameters]` local would make its
            // value depend on statement order (and read 0.0 when the local is
            // assigned later). Unknown identifiers are covariates, checked
            // against the data by `check_covariates` like every other.
            let ctx = ParseCtx::new(&theta_names, &eta_names, &[]).with_nn_specs(&nn_specs_for_ctx);
            let expr = parse_scalar_expression(src, ctx)
                .map_err(|e| format!("[parameters] kappa `{name}` weight `{src}`: {e}"))?;
            if expr_uses_eta(&expr) {
                return Err(format!(
                    "[parameters] kappa `{name}` weight `{src}` references a random effect. \
                     A weight is the *known* precision of an occasion (an arm's sample size), \
                     so it may depend only on covariates, fixed thetas and TIME — a random \
                     effect there would make the variance of κ a function of κ itself."
                ));
            }
            // Same argument one step further out: an *estimated* theta makes the
            // weight move during the outer optimisation, so the up-front
            // positivity check (`E_KAPPA_WEIGHT_NONPOSITIVE`, evaluated at the
            // initial θ) certifies nothing — a weight that is positive at the
            // start and crosses zero mid-fit divides an individual parameter by
            // zero with no diagnostic. A FIXed theta is a known constant, so it
            // is as safe as a covariate column and stays allowed.
            let mut estimated: Vec<&str> = Vec::new();
            visit_expr_nodes(&expr, &mut |e: &Expression| {
                if let Expression::Theta(ti) = e {
                    if thetas.get(*ti).is_some_and(|t| !t.fixed) {
                        let tn = thetas[*ti].name.as_str();
                        if !estimated.contains(&tn) {
                            estimated.push(tn);
                        }
                    }
                }
            });
            if !estimated.is_empty() {
                return Err(format!(
                    "[parameters] kappa `{name}` weight `{src}` references estimated theta(s) {}. \
                     A weight is the *known* precision of an occasion, so it may depend only on \
                     covariates, FIXed thetas and TIME: an estimated theta moves the weight during \
                     the fit, so the up-front check that it stays strictly positive cannot certify \
                     it, and a weight that crosses zero divides an individual parameter by zero. \
                     Declare the constant `FIX`, or move it into a covariate column.",
                    estimated
                        .iter()
                        .map(|n| format!("`{n}`"))
                        .collect::<Vec<_>>()
                        .join(", "),
                ));
            }
            out.push(Some(expr));
        }
        out
    };
    let weighted_kappa_slots: HashMap<usize, Expression> = kappa_weight_exprs
        .iter()
        .enumerate()
        .filter_map(|(k, w)| w.as_ref().map(|w| (n_eta + k, w.clone())))
        .collect();
    if !weighted_kappa_slots.is_empty() {
        // A weighted kappa outside `[individual_parameters]` would read as the
        // *unweighted* κ — a plausible wrong answer rather than an error, which
        // is the exact failure this feature exists to remove. Every block that
        // can see a kappa name at all is scanned textually (the compiled forms
        // differ per block; the name does not).
        let weighted_names: Vec<&String> = kappa_weight_exprs
            .iter()
            .enumerate()
            .filter_map(|(k, w)| w.as_ref().and_then(|_| kappa_names.get(k)))
            .collect();
        // `extracted.named` is keyed block-type → instance-name, so the inner
        // key alone would print `[armA]` — the instance, not the block. Label
        // each with the pair the author actually wrote (`event_model armA`).
        let named_blocks = extracted
            .named
            .iter()
            .flat_map(|(ty, m)| m.keys().map(move |inst| (format!("{ty} {inst}"), &m[inst])));
        for (block, lines) in blocks
            .iter()
            .filter(|(b, _)| b.as_str() != "parameters" && b.as_str() != "individual_parameters")
            .map(|(b, l)| (b.clone(), l))
            .chain(named_blocks)
        {
            let mut refs: HashSet<String> = HashSet::new();
            collect_referenced_identifiers(lines, &mut refs);
            for name in &weighted_names {
                if refs.contains(&name.to_ascii_uppercase()) {
                    return Err(format!(
                        "[{block}] references `{name}`, a kappa declared with `weight = ...`. \
                         A weighted kappa is scaled where it is used, and only \
                         `[individual_parameters]` applies that scaling — a reference here \
                         would silently use the unweighted κ. Move the term into \
                         `[individual_parameters]` (assign it to an individual parameter and \
                         reference that name instead)."
                    ));
                }
            }
        }
    }
    if !weighted_kappa_slots.is_empty() {
        // Mu-reference detection (further down) runs on the *rewritten*
        // statements, deliberately: the declared form and the hand-written
        // `K / sqrt(W)` must be the same model everywhere, mu-refs included.
        // Detecting on a pre-rewrite snapshot would register a kappa mu-ref the
        // rewritten model does not satisfy (which `method = bayes` then
        // rejects) and, in `THETA * exp(KAPPA) * exp(ETA)`, would return the
        // kappa's index and *lose* the BSV eta's mu-ref that survives the
        // rewrite — costing SAEM/IMP the closed-form θ step it is meant to
        // protect. `exp(ETA + KAPPA/sqrt(W))` still resolves to `ETA` via the
        // `(a, b) => a.or(b)` arm, which already tolerates a non-bare operand.
        let rewritten = rewrite_weighted_kappas(&mut indiv_stmts, &weighted_kappa_slots);
        // A weight that scales nothing is not a no-op: `print_results` would
        // still report `→ SD = γ/√N` for a scaling that was never applied, and
        // a covariate named only by the weight expression never reaches
        // `referenced_covariates`, so the datareader skips the column and the
        // data check then blames a *present* column for evaluating to 0.
        let mut inert: Vec<&str> = weighted_kappa_slots
            .keys()
            .filter(|slot| !rewritten.contains(slot))
            .filter_map(|slot| kappa_names.get(slot - n_eta).map(String::as_str))
            .collect();
        if !inert.is_empty() {
            inert.sort_unstable();
            return Err(format!(
                "[parameters] kappa(s) {} are declared with `weight = ...` but never referenced \
                 in [individual_parameters]. The weight is applied where the kappa is *used* \
                 (`K` → `K / sqrt(W)`), so an unreferenced one scales nothing while the fit still \
                 reports the weighted SD. Reference the kappa in [individual_parameters], or drop \
                 the `weight = ...` modifier.",
                inert
                    .iter()
                    .map(|n| format!("`{n}`"))
                    .collect::<Vec<_>>()
                    .join(", "),
            ));
        }
    }

    // Detect ODE vs analytical model
    let is_ode = struct_lines
        .iter()
        .any(|l| l.starts_with("ode(") || l.starts_with("ode "));

    // #486 — desugar a bare θ/η in a Form-C `y = expr` readout into a synthetic
    // individual parameter (`__ferx_ro_th{i} = THETA(i)` / `…eta{k} = ETA(k)`). The
    // readout (rewritten in `parse_scaling_block` below) then references that
    // parameter, so its direct θ/η dependence rides the validated
    // individual-parameter sensitivity chain rather than dropping the subject to FD.
    // Must run *before* `ode_param_slots` / `build_ode_spec` / `build_pk_param_fn`
    // (which all consume `indiv_var_names` / `indiv_stmts`) so the synthetic params
    // get slots, values, and partials like any individual parameter. Only `y`
    // readouts are scanned; `obs_scale` θ/η stays with `ScaleDerivProgram`.
    // Collected for BOTH engines (#486). The two differ only in *where the slots come from*:
    // an ODE model routes every individual parameter through `ode_param_slots` (available a
    // few lines below), so its synthetics are sized and appended here; an analytical model
    // draws from the small spare region `allocate_readout_extra_slots` owns, and neither
    // `pk_model` nor `pk_param_map` exists yet at this point — so the analytical append is
    // deferred to that allocator (still before `build_pk_param_fn`, the real constraint).
    let mut readout_synth_params: Vec<ReadoutSynthParam> = match blocks.get("scaling") {
        Some(lines) => collect_readout_theta_eta_synth(lines, &theta_names, &eta_names)?,
        None => Vec::new(),
    };
    // #486 — the `__ferx_ro_` (Form-C readout) and `__ferx_pktime_` (direct `pk(...=TIME)`
    // mapping) prefixes are reserved for synthetic individual parameters. Reject a
    // user/generated parameter that carries either rather than silently misclassifying it
    // as internal (it would vanish from EBE/sdtab output and be rejected as an unknown
    // output column by `is_synthetic_readout_param`). This runs before either desugaring
    // appends its own synthetic params, so `all_assigned`/`theta_names`/`eta_names` hold
    // only user names here.
    if let Some(bad) = all_assigned
        .iter()
        .chain(theta_names.iter())
        .chain(eta_names.iter())
        .find(|n| is_synthetic_readout_param(n.as_str()))
    {
        let (prefix, reserved_for) = if bad.starts_with(PKTIME_SYNTH_PREFIX) {
            (
                PKTIME_SYNTH_PREFIX,
                "internal `pk(...=TIME)` mapping parameters",
            )
        } else {
            (READOUT_SYNTH_PREFIX, "internal Form-C readout parameters")
        };
        return Err(format!(
            "parameter name `{bad}` uses the reserved `{prefix}` prefix, which ferx \
             reserves for {reserved_for} (#486). Rename it."
        ));
    }
    // #486 slot headroom: each synthetic readout param consumes a PK slot. A model
    // that previously parsed via the FD fallback (where a direct θ/η in the readout
    // took no slot) must not start failing the fixed PK-slot layout now. If
    // the synthetics would overflow, keep the readout on the FD fallback (its prior
    // behaviour) instead of erroring, and note it.
    let mut readout_fd_fallback_note: Option<String> = None;
    if is_ode {
        if !readout_synth_params.is_empty() {
            let free = ode_free_slot_count(&indiv_var_names);
            if readout_synth_params.len() > free {
                readout_fd_fallback_note = Some(readout_synth_fd_note(
                    readout_synth_params.len() - free,
                    crate::types::MAX_PK_PARAMS,
                ));
                readout_synth_params.clear();
            }
        }
        for s in &readout_synth_params {
            append_readout_synth_param(s, &mut indiv_var_names, &mut indiv_stmts);
        }
    }

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

    // Joint PK-TTE (Slice 2.1, #564): cmt → index of the synthetic cumulative-hazard
    // ODE state appended for each ODE-driven `[event_model] hazard = ...`. Stays empty
    // for every model without an ODE-accumulated hazard.
    #[cfg(feature = "survival")]
    let mut chz_state_map: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();

    let (
        pk_model,
        mut pk_param_map,
        mut ode_spec,
        diffusion_theta_names,
        diffusion_theta_inits,
        diffusion_theta_fixed,
        diffusion_state_indices,
    ) = if is_ode {
        #[cfg_attr(not(feature = "survival"), allow(unused_mut))]
        let (mut state_names, obs_cmt_name) = parse_ode_structural(struct_lines)?;
        // Owned copy of the [odes] derivative lines so joint PK-TTE can append a
        // cumulative-hazard accumulator without mutating the shared block map.
        #[cfg_attr(not(feature = "survival"), allow(unused_mut))]
        let mut ode_lines: Vec<String> = blocks
            .get("odes")
            .ok_or("ODE model requires [odes] block")?
            .clone();
        // For each ODE-driven `[event_model] hazard = <expr>`, append a synthetic
        // accumulator state `__chz_<cmt>` with `d/dt(__chz_<cmt>) = <expr>`. It then
        // compiles in the [odes] RHS namespace (states + individual parameters) like
        // any derivative; the TTE likelihood reads H(t)/h(t) from this state, and the
        // endpoint records its index via `chz_state_map` (#564).
        #[cfg(feature = "survival")]
        for (cmt, hazard_expr) in prescan_ode_hazards(&extracted)? {
            let chz_idx = state_names.len();
            let state = format!("__chz_{cmt}");
            if state_names.contains(&state) {
                return Err(format!(
                    "[odes]: state `{state}` collides with the cumulative-hazard accumulator \
                     the parser appends for the [event_model] CMT={cmt} hazard (joint PK-TTE) \
                     — rename the ODE state (`__chz_*` names are reserved)."
                ));
            }
            ode_lines.push(format!("d/dt({state}) = {hazard_expr}"));
            state_names.push(state);
            chz_state_map.insert(cmt, chz_idx);
        }
        let mut ode_spec = build_ode_spec(
            &ode_lines,
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

    // #486 — desugar a direct `pk(...=TIME)` structural mapping into a synthetic
    // individual parameter (`__ferx_pktime_<pk_name> = TIME`), mirroring the Form-C
    // readout θ/η desugaring (#631). A bare `pk(v=TIME)` binding otherwise leaves the
    // mapped PK slot out of the individual-parameter program's `pk_slots`, so
    // `prog_covers_required_pk_slots` declines and the whole model routes to FD (#637
    // follow-up). Rewriting the binding to reference the synthetic parameter makes the
    // slot ride the validated per-event `TIME` sensitivity chain (value = event time via
    // `Op::PushTime`, ∂/∂θ = ∂/∂η = 0). Runs before `pk_indices` / `build_pk_param_fn`
    // consume `indiv_var_names` / `indiv_stmts`.
    {
        let mut time_bound: Vec<String> = pk_param_map
            .iter()
            .filter(|(_, v)| v.as_str() == "TIME" || v.as_str() == "time")
            .map(|(k, _)| k.clone())
            .collect();
        if !time_bound.is_empty() {
            // A user parameter carrying the reserved `__ferx_pktime_` prefix is already
            // rejected by the shared synthetic-prefix guard above (it runs before either
            // desugaring adds its own synthetic params, so it sees only user names).
            // Sorted for deterministic slot/var order across runs.
            time_bound.sort();
            for pk_name in time_bound {
                let synth = format!("{PKTIME_SYNTH_PREFIX}{}", pk_name.to_lowercase());
                indiv_var_names.push(synth.clone());
                indiv_stmts.push(Statement::Assign(synth.clone(), Expression::Time));
                pk_param_map.insert(pk_name, synth);
            }
        }
    }

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
    // Compartment-indexed modeled-dose attributes (`D{cmt}` for `RATE=-2`
    // duration, `R{cmt}` for `RATE=-1` rate; #324/#394) for ANALYTICAL models. ODE
    // models build their `dose_attr_map` inside `build_ode_spec` (routing
    // `D{cmt}`/`R{cmt}` through `ode_param_slots`); analytical models route only
    // canonical PK names into `PkParams`, so a `D{cmt}`/`R{cmt}` individual
    // parameter would otherwise be evaluated and discarded. Route each into a
    // free `PkParams` slot — the spare region above the canonical PK slots
    // (`PK_IDX_LAGTIME + 1 ..= MAX_PK_PARAMS - 1`), which analytical models never
    // touch — and record `(attr, cmt) -> slot` so the analytical predictor can
    // resolve the modeled dose via `DoseEvent::resolve_rate`, exactly as the ODE
    // path does. `analytical_modeled_slots` carries `(var_name, slot)` into
    // `build_pk_param_fn` so its closure writes the value alongside the canonical
    // PK assignments. Empty (and the map stays `Default`) for ODE models and for
    // the common analytical model with no `RATE=-1`/`-2` dosing.
    let mut analytical_dose_attr_map = crate::types::DoseAttrMap::default();
    let mut analytical_modeled_slots: Vec<(String, usize)> = Vec::new();
    if !is_ode {
        let mut next_slot = crate::types::PK_IDX_LAGTIME + 1;
        for name in &indiv_var_names {
            // Only the modeled-`RATE` attributes (`D{cmt}` duration, `R{cmt}`
            // rate) are routed here. `F{cmt}`/`ALAG{cmt}` collapse to the single
            // analytical dose route (the bare `PK_IDX_F` / `PK_IDX_LAGTIME`
            // slots), so although `from_indexed_name` recognises them they are not
            // routed for analytical models.
            let Some((attr, cmt)) = crate::types::DoseAttr::from_indexed_name(name) else {
                continue;
            };
            // NONMEM-faithful display of the attribute for diagnostics: the DSL
            // parameter prefix, the RATE code that drives it, and the noun.
            // Per-compartment `F{cmt}`/`ALAG{cmt}` are ODE-only (they route through
            // the ODE `dose_attr_map`); the analytical engine has a single dose
            // route reached via the bare `f=`/`lagtime=` mapping.
            let (param_prefix, rate_code, kind) = match attr {
                crate::types::DoseAttr::Duration => ("D", "-2", "duration"),
                crate::types::DoseAttr::Rate => ("R", "-1", "rate"),
                crate::types::DoseAttr::F | crate::types::DoseAttr::Lag => {
                    // A user may legitimately NAME their bioavailability / lag
                    // parameter `F1`/`ALAG1` (a natural NONMEM-porting choice) and
                    // bind it to the single analytical dose route via the bare
                    // `f=`/`lagtime=` mapping — `pk(..., f=F1)` routes `F1` into
                    // `PK_IDX_F` and applies it correctly. Accept that case (it is
                    // bit-identical to the canonical `f=F` idiom). Reject only an
                    // *unmapped* `F{cmt}`/`ALAG{cmt}`: with nothing binding it to the
                    // dose route, its value lands in an unused spare slot and is
                    // silently discarded — effective bioavailability stays 1 / lag
                    // stays 0 with no effect — the porting footgun this guard exists
                    // to catch. `alag`/`lagtime` both resolve to `PK_IDX_LAGTIME`
                    // via `name_to_index`, so either spelling of the mapping counts.
                    let target_slot = if matches!(attr, crate::types::DoseAttr::F) {
                        crate::types::PK_IDX_F
                    } else {
                        crate::types::PK_IDX_LAGTIME
                    };
                    let bound_to_dose_route = pk_param_map.iter().any(|(key, value)| {
                        value == name
                            && crate::types::PkParams::name_to_index(key) == Some(target_slot)
                    });
                    if bound_to_dose_route {
                        continue;
                    }
                    let (indexed, noun, bare) = if matches!(attr, crate::types::DoseAttr::F) {
                        ("F", "bioavailability", "f")
                    } else {
                        ("ALAG", "lag time", "lagtime")
                    };
                    return Err(format!(
                        "[individual_parameters]: `{name}` reads as a per-compartment {noun} \
                         (`{indexed}{cmt}`), an ODE-only dose attribute, but nothing binds it to \
                         the analytical `{}` model's single dose route — so its value would be \
                         silently discarded (effective {noun} unchanged). Bind it via the bare \
                         `{bare}=` argument in the `pk(...)` call (e.g. `{bare}={name}`), or use an \
                         `ode(...)` model for a genuinely compartment-specific value.",
                        pk_model.canonical_name(),
                    ));
                }
            };
            // Reject a `D{cmt}`/`R{cmt}` whose compartment the analytical engine
            // cannot infuse into (see `PkModel::infusable_compartments`): every
            // compartment of the six walk-supported models is infusable — central
            // for all, the oral depot since #400, the oral peripherals since #375
            // — so what this actually catches is a `D{cmt}` past the end of the
            // state vector, and any `D{cmt}` at all on a transit/IG model (whose
            // `infusable` is empty). A looser bound (e.g. raw compartment count)
            // would let such a coded `RATE` pass parse + the data gate, then
            // either silently route into central (no-TV superposition) or panic in
            // the event-driven walker — the silent/abrupt failure class this
            // feature exists to prevent. Caught here at parse time with an
            // actionable message.
            let infusable = pk_model.infusable_compartments();
            if !infusable.contains(&cmt) {
                let supported = infusable
                    .iter()
                    .map(|c| format!("`{param_prefix}{c}`"))
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(format!(
                    "[individual_parameters]: `{name}` is a modeled infusion {kind} \
                     (RATE={rate_code}) for compartment {cmt}, but the analytical `{}` model \
                     can only infuse into compartment(s) {:?} ({supported}). A zero-order \
                     input into a compartment the closed form does not have — or into any \
                     compartment of a transit / inverse-Gaussian absorption model — needs \
                     an `ode(...)` model.",
                    pk_model.canonical_name(),
                    infusable,
                ));
            }
            // `infusable_compartments()` has ≤ 4 entries (`three_cpt_oral`, since
            // #375) and at most one `D` and one `R` per compartment, so at most 8
            // distinct modeled-dose parameters are routed — comfortably inside
            // `MAX_PK_PARAMS`. A debug assertion documents that invariant without
            // a permanently-dead user-facing error arm.
            debug_assert!(
                next_slot < crate::types::MAX_PK_PARAMS,
                "modeled-dose slot {next_slot} exceeds MAX_PK_PARAMS; \
                 infusable_compartments() should have bounded the D{{cmt}}/R{{cmt}} count"
            );
            let slot = next_slot;
            next_slot += 1;
            analytical_dose_attr_map.insert(attr, cmt, slot);
            analytical_modeled_slots.push((name.clone(), slot));
        }
    }

    // #650: non-structural individual parameters referenced by an analytic Form C
    // readout (e.g. `BMAX`/`KD` in a saturable-binding total-conc readout) get free
    // differentiable PK slots so they are written by `pk_param_fn` and differentiated
    // by the individual-parameter program — instead of aliasing the `CL` slot. Empty
    // for ODE models and for models without such a readout.
    let structural_vars: std::collections::HashSet<String> =
        pk_param_map.values().cloned().collect();
    let readout_alloc = allocate_readout_extra_slots(
        blocks.get("scaling"),
        &theta_names,
        &eta_names,
        &indiv_var_names,
        &structural_vars,
        &analytical_modeled_slots,
        pk_model,
        is_ode,
        &readout_synth_params,
    )?;
    let readout_extra_slots = readout_alloc.slots;
    // #486: the analytical half of the direct-θ/η readout desugaring. The ODE path appended
    // its synthetics far upstream (its slot map was already known there); the analytical path
    // could not size them until now, so it appends the accepted ones here — still before
    // `build_pk_param_fn` below, which is what actually consumes `indiv_var_names`/`indiv_stmts`.
    // Dropping the rest keeps `readout_synth_params` in step with the slots that exist, so
    // `parse_scaling_block`'s rewrite only rewrites θ/η it can actually resolve; anything left
    // bare keeps `dual_evaluable = false` and the readout falls back to FD, as before #486.
    if !is_ode {
        for s in &readout_alloc.accepted_synths {
            append_readout_synth_param(s, &mut indiv_var_names, &mut indiv_stmts);
        }
        readout_synth_params = readout_alloc.accepted_synths;
        if let Some(note) = readout_alloc.fd_note {
            readout_fd_fallback_note = Some(note);
        }
    }

    let (pk_param_fn, referenced_covariates, mut indiv_param_partials, indiv_param_program) =
        build_pk_param_fn(
            indiv_stmts.clone(),
            &pk_param_map,
            &indiv_var_names,
            &ode_slot_map,
            &analytical_modeled_slots,
            &readout_extra_slots,
            thetas.len(),
            n_eta_extended_for_partials,
            #[cfg(feature = "nn")]
            &covariate_nns_for_closure,
        )?;

    // Attach the individual-parameter program to the ODE spec (if any) for the
    // analytic-sensitivity η/θ chain (issue #367). The analytical PK provider
    // reads its copy from `indiv_param_partials` (ODE models route to the ODE
    // provider, so the partials copy is unused there — a single parse-time clone).
    indiv_param_partials.indiv_param_program = Some(indiv_param_program.clone());
    if let Some(ode_spec) = ode_spec.as_mut() {
        ode_spec.indiv_param_program = Some(indiv_param_program);
    }

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
    // BSV omega is built from the BSV-only eta names (no kappas). An empty eta list
    // yields a 0×0 Omega — a fixed-effects-only (naive-pooled) model — which
    // `build_omega_matrix` handles directly; see its `n == 0` branch for why the
    // empty matrix is well-defined here.
    let omega = build_omega_matrix(&omegas, &block_omegas, &eta_names_bsv)?;
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
    let used_sigmas_in_error = used_sigma_names(&parsed_error_model, &sigma_names);
    // Capture the single-endpoint arguments before `parsed_error_model` is
    // consumed, so the custom residual-magnitude programs (#484) can be built
    // after the [covariates] block is parsed (covariate names are needed to
    // validate the magnitude expressions).
    let single_error_args: Option<(ErrorModel, Vec<String>)> = match &parsed_error_model {
        ParsedErrorModel::Single(em, args) => Some((*em, args.clone())),
        ParsedErrorModel::PerCmt(_) | ParsedErrorModel::Selected { .. } => None,
    };
    // Covariates the #658 error selector references become required data columns
    // (validated as E_MISSING_COVARIATE at fit setup). Capture before
    // `build_error_spec` consumes the parsed model.
    let selector_covariates: Vec<String> = match &parsed_error_model {
        ParsedErrorModel::Selected { covariates, .. } => covariates.clone(),
        _ => Vec::new(),
    };
    let (error_model, error_spec) = build_error_spec(parsed_error_model, &sigma_names, is_ode)?;
    let residual_correlations = build_residual_correlations(&block_sigmas, &sigma_names)?;
    validate_residual_correlations(&error_spec, &residual_correlations, &sigma_names)?;
    // Keep a copy of the flat sigma names for the custom residual-magnitude
    // build (#484), which runs after the [covariates] block is parsed.
    let ruv_sigma_names = sigma_names.clone();
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

    let mut default_params = ModelParameters {
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
        mixture: None,
    };

    // Auto-generate tv_fn: evaluate individual parameters with eta=0
    // This gives covariate-adjusted typical values for the AD inner loop.
    // tv_fn uses the extended eta context (BSV + kappa) so KAPPA_* vars evaluate
    // to 0 at population-typical predictions, which is correct.
    let tv_eta_names = eta_names.clone(); // extended (BSV + kappa)
    let tv_fn: Option<crate::types::TvFn> = if !is_ode {
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
    // Detection runs on the statements as they will actually be evaluated —
    // including the weighted-kappa rewrite (#1031), which is the point: the
    // declared `weight = W` form and the hand-written `K / sqrt(W)` must agree
    // on their mu-refs exactly as they agree on the objective. See the rewrite
    // site above.
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
        // #650: readout-referenced non-structural params get their allocated slot
        // (so the readout program's `indiv_to_pk` points at the value `pk_param_fn`
        // writes), instead of aliasing the `CL` slot via the `unwrap_or(0)` below.
        let extra_slot: HashMap<&str, usize> = readout_extra_slots
            .iter()
            .map(|(n, s)| (n.as_str(), *s))
            .collect();
        indiv_var_names
            .iter()
            .map(|var_name| {
                var_to_pk
                    .get(&var_name.to_uppercase())
                    .and_then(|pk_name| PkParams::name_to_index(pk_name))
                    .or_else(|| extra_slot.get(var_name.as_str()).copied())
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

    // ── [mixture] block (#977) ──
    // MIXNUM is reserved read-only: reject any assignment to it, and reject its
    // use in a model without a `[mixture]` block. When the block is present,
    // parse + validate it here (theta/eta/sigma names must still be in scope —
    // `theta_names`/`default_params` are moved into `CompiledModel` below).
    if stmts_assign_mixnum(&indiv_stmts) {
        return Err(
            "MIXNUM is a reserved read-only mixture index and may not be assigned".to_string(),
        );
    }
    let mixture_spec: Option<crate::types::MixtureSpec> =
        if let Some(mix_lines) = blocks.get("mixture") {
            Some(parse_mixture_block(
                mix_lines,
                &theta_names,
                &eta_names_bsv,
                &default_params.sigma.names,
            )?)
        } else {
            // MIXNUM is invalid anywhere in a model with no [mixture] block. The
            // AST check covers [individual_parameters]; scan the other
            // expression-bearing blocks ([error_model], [structural_model],
            // [derived], [odes], …) at the token level since they compile to
            // opaque closures below.
            let mixnum_elsewhere = blocks
                .iter()
                .filter(|(k, _)| k.as_str() != "mixture" && k.as_str() != "individual_parameters")
                .any(|(_, lines)| lines_use_mixnum(lines));
            if stmts_use_mixnum(&indiv_stmts) || mixnum_elsewhere {
                return Err(
                    "MIXNUM is only valid in a mixture model; add a [mixture] block".to_string(),
                );
            }
            None
        };

    // Build the numeric per-class Omega/Sigma into `default_params.mixture`
    // (#977 Phase 2). Uses the just-built base Omega/Sigma; a local avoids
    // borrowing `default_params` immutably and mutably at once.
    let mixture_params = mixture_spec
        .as_ref()
        .map(|spec| build_mixture_params(spec, &default_params.omega, &default_params.sigma))
        .transpose()?;
    default_params.mixture = mixture_params;

    // Per-kappa weight, carried for *reporting* and validation only (#1031) —
    // the scaling itself was desugared into `indiv_stmts` above. The evaluator
    // lets `fit()` report the effective SD (`γ/√W`) at a typical arm size, and
    // lets the up-front data check reject a weight that is zero, negative or
    // non-finite before it divides an individual parameter.
    let kappa_weights: Vec<Option<crate::types::KappaWeight>> = kappa_weight_exprs
        .into_iter()
        .zip(kappa_info.weights.iter())
        .map(|(expr, src)| {
            let expr = expr?;
            let src = src.clone().unwrap_or_default();
            let eval: crate::types::KappaWeightFn = Box::new(move |theta, cov, time| {
                let vars: HashMap<String, f64> = HashMap::new();
                with_model_time(time, || eval_expression(&expr, theta, &[], cov, &vars, &[]))
            });
            Some(crate::types::KappaWeight { expr: src, eval })
        })
        .collect();

    let model = CompiledModel {
        name,
        pk_model,
        error_model,
        error_spec,
        residual_correlations,
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
        kappa_weights,
        tv_fn,
        #[cfg(feature = "nn")]
        covariate_nns,
        pk_indices,
        eta_map,
        pk_idx_f64,
        sel_flat,
        ode_spec,
        // Analytical models carry their modeled-dose (`D{cmt}`) map here; ODE
        // models keep theirs on `ode_spec.dose_attr_map` and leave this empty.
        dose_attr_map: analytical_dose_attr_map,
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
        frem_config: None,
        residual_error_eta,
        // Populated below from the optional [initial_conditions] block (#521).
        analytical_init: Vec::new(),
        analytic_readout: None,
        // Populated below from the [error_model] magnitude expressions (#484).
        ruv_magnitude: None,
        absorption_ode_equivalent: None,
        mixture: None,
    };

    // ── Optional blocks ──
    let simulation = blocks
        .get("simulation")
        .map(|lines| parse_simulation_block(lines))
        .transpose()?;
    // Peek the `[covariates]` declarations (parsed for real further down, at the
    // block's own site) purely for name precedence: a declared `T` / `t` is a data
    // column, not the `[odes]` model-time alias, in `[scaling]` and in
    // `[adaptive_dosing] observe` alike (#1028). Peeked rather than hoisted so a
    // malformed `[covariates]` block still reports its own error at its own site, in
    // the existing order — hence the `.ok()`.
    let declared_covariate_names: Vec<String> = blocks
        .get("covariates")
        .and_then(|lines| parse_covariates_block(lines).ok())
        .map(|decls| decls.into_iter().map(|d| d.name).collect())
        .unwrap_or_default();
    // Declarative reactive-dosing controller (#391 S2). Parsed and validated here;
    // compiled to a controller and run by the adaptive simulate path in a later slice.
    let adaptive_dosing = blocks
        .get("adaptive_dosing")
        .map(|lines| parse_adaptive_dosing_block(lines, &declared_covariate_names))
        .transpose()?;
    let mut fit_options = if let Some(lines) = blocks.get("fit_options") {
        parse_fit_options(lines)?
    } else {
        FitOptions::default()
    };

    // `[data]` block (#690, NONMEM `$DATA` analogue). Only the raw `path =`
    // value is resolved here; `parse_full_model_file` resolves it relative to
    // the model file's directory afterwards (this function only sees the
    // model's text, not its path on disk).
    let (data_path, data_column_map) = match blocks.get("data") {
        Some(lines) => {
            let (p, m) = parse_data_block(lines)?;
            (Some(p), m)
        }
        None => (None, Vec::new()),
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
    // Class-aware mu-references (#996): detected here rather than in
    // `parse_mixture_block` because the scan needs both the parsed
    // `[individual_parameters]` statements and the mixture's class count.
    model.mixture = mixture_spec.map(|mut spec| {
        spec.mu_refs = detect_mixture_mu_refs(
            &indiv_stmts,
            &model.theta_names,
            &model.eta_names,
            spec.n_classes,
        );
        spec
    });
    // A `MIXNUM`-switched typical value that did *not* yield a class-aware
    // mu-ref falls back to the purely numerical M-step under SAEM/IMP, which is
    // the slowest-converging channel. Say so rather than degrading silently.
    //
    // Only eta-carrying expressions qualify: a class-switched typical value with
    // no IIV (`V = if (MIXNUM == 1) TVV1 else TVV2`) or an intermediate class flag
    // (`FLAG = if (MIXNUM == 1) 1 else 2`) is not a missed mu-ref, and advising
    // the user to add a random effect they deliberately omitted is wrong
    // (#996 review).
    if let Some(ref mix) = model.mixture {
        let unmatched: Vec<String> = indiv_stmts
            .iter()
            .filter_map(|st| match st {
                Statement::Assign(name, expr) if expr_uses_mixnum(expr) && expr_uses_eta(expr) => {
                    Some((name, expr))
                }
                _ => None,
            })
            .filter(|(_, expr)| detect_mixture_pattern(expr, mix.n_classes).is_none())
            .map(|(name, _)| name.clone())
            .collect();
        if !unmatched.is_empty() {
            model.parse_warnings.push(format!(
                "Class-aware mu-referencing not applied to {}: the MIXNUM-switched \
                 expression is not a chain of `MIXNUM == k` branches whose arms are all \
                 `THETA_k * exp(ETA)` (or `exp(log(THETA_k) + ETA)`) on the same ETA. Under \
                 SAEM/IMP the class typical values are estimated by the numerical M-step \
                 alone, which converges more slowly (#996).",
                unmatched.join(", ")
            ));
        }
    }

    // #993, the `[adaptive_dosing]` half. `observe` is compiled through the very
    // same `build_y_output_fn` as a Form-C `y` readout (`compile_observe`), so it
    // sees individual parameters — and it is the *controller's* signal, not a
    // reported number. A dose attribute read here is applied once at the dose and
    // once in the signal, so the titration logic compares a value that is wrong by
    // exactly that attribute and every dose it then emits inherits the error. The
    // `when` rules cannot reach a parameter (they compare the `signal` keyword to an
    // `f64` literal), so `observe` is the whole surface.
    //
    // Same split as `[scaling]`: the rejection runs on both engines — the ODE
    // side over `ode_slot_map` (#993), the analytical side over the explicit
    // `pk(..., f=F)`/`lagtime=` mapping (#1004) — as does the `D{n}`/`R{n}`
    // recording.
    if let Some(observe) = adaptive_dosing.as_ref().and_then(|s| s.observe.as_deref()) {
        let observe_reads = collect_indiv_param_reads(
            observe,
            &model.theta_names,
            &model.eta_names,
            &model.indiv_param_names,
            &[],
            "[adaptive_dosing] observe",
        )?;
        // The `T` / `t` model-time fold warns wherever it fires — but `observe` is
        // compiled at *simulate* time (`sim::adaptive_control::compile_observe`), long
        // after `parse_warnings` is sealed, and the adaptive result has no warnings
        // channel of its own. Emit the note here instead, from the same helper the
        // compiler will run, so the fold is not silent on this one path (#1028).
        warn_if_time_alias_folds(
            observe,
            "[adaptive_dosing] observe",
            &model.theta_names,
            &model.eta_names,
            &model.indiv_param_names,
            &declared_covariate_names,
            &mut model.parse_warnings,
        );
        if let Some(ode) = model.ode_spec.as_ref() {
            let n_states = ode.state_names.len();
            check_dose_attr_double_use(
                &model.indiv_param_names,
                &ode_slot_map,
                &observe_reads,
                n_states,
                "[adaptive_dosing]",
                "[adaptive_dosing]",
                None,
            )?;
        } else {
            let analytical_slots =
                analytical_dose_attr_slot_map(&pk_param_map, &model.indiv_param_names);
            check_dose_attr_double_use(
                &model.indiv_param_names,
                &analytical_slots,
                &observe_reads,
                usize::MAX,
                "[adaptive_dosing]",
                "[adaptive_dosing]",
                Some(&pk_param_map),
            )?;
        }
        // Cloned because `record_coded_rate_reads` needs `&mut model` for the map
        // while reading the name list off the same model; one short-string Vec per
        // parse. (`[scaling]` needs no clone — it already holds its own copy.)
        let adaptive_indiv_names = model.indiv_param_names.clone();
        record_coded_rate_reads(
            &mut model,
            &adaptive_indiv_names,
            &ode_slot_map,
            &observe_reads,
        );
    }

    // Register the mixing-expression covariates (logit(k) = … BWT*(WT−75) …) as
    // required data columns, mirroring the scaling / error-selector / init blocks
    // below. Without this a `[covariates]`-declared model never reads the column,
    // so a covariate used only in the mixing expression silently evaluates to 0
    // and the mixing degrades to intercept-only (the #765 trap).
    if let Some(ref mix) = model.mixture {
        let mix_covs = mix.logit_covariates.clone();
        register_referenced_covariates(&mut model.referenced_covariates, mix_covs);
    }

    // Build FremConfig from fit options when frem_predictions is present.
    // Format: "THETA_NAME/ETA_NAME:FREMTYPE, ..."
    // Example: "TV_WT/ETA_WT_FREM:100, TV_AGE/ETA_AGE_FREM:200"
    if let Some(ref preds_str) = fit_options.frem_predictions {
        let mut fremtype_to_indices = std::collections::HashMap::new();
        for pair in preds_str.split(',') {
            let pair = pair.trim();
            if pair.is_empty() {
                continue;
            }
            let parts: Vec<&str> = pair.split(':').collect();
            if parts.len() != 2 {
                return Err(format!(
                    "frem_predictions: expected 'THETA/ETA:FREMTYPE', got '{}'",
                    pair
                ));
            }
            let names_part = parts[0].trim();
            let ft_value: u16 = parts[1].trim().parse().map_err(|_| {
                format!(
                    "frem_predictions: expected integer FREMTYPE value, got '{}'",
                    parts[1].trim()
                )
            })?;
            let name_parts: Vec<&str> = names_part.split('/').collect();
            if name_parts.len() != 2 {
                return Err(format!(
                    "frem_predictions: expected 'THETA/ETA:FREMTYPE', got '{}'",
                    pair
                ));
            }
            let theta_name = name_parts[0].trim();
            let eta_name = name_parts[1].trim();
            let theta_idx = model
                .theta_names
                .iter()
                .position(|n| n == theta_name)
                .ok_or_else(|| {
                    format!(
                        "frem_predictions: theta '{}' not found (available: {:?})",
                        theta_name, model.theta_names
                    )
                })?;
            let eta_idx = model
                .eta_names
                .iter()
                .position(|n| n == eta_name)
                .ok_or_else(|| {
                    format!(
                        "frem_predictions: eta '{}' not found (available: {:?})",
                        eta_name, model.eta_names
                    )
                })?;
            fremtype_to_indices.insert(ft_value, (theta_idx, eta_idx));
        }
        // Find covariate sigma index.
        let sigma_name = fit_options.frem_sigma.as_deref().unwrap_or("EPSCOV");
        let covariate_sigma_index = model
            .default_params
            .sigma
            .names
            .iter()
            .position(|n| n == sigma_name)
            .ok_or_else(|| {
                format!(
                    "frem_sigma: sigma parameter '{}' not found (available: {:?})",
                    sigma_name, model.default_params.sigma.names
                )
            })?;
        model.frem_config = Some(crate::types::FremConfig {
            fremtype_to_indices,
            covariate_sigma_index,
        });
    }

    // Bake the configured ODE solver tolerances from [fit_options] onto the
    // OdeSpec so predict()/fit_from_files (which integrate the parsed spec
    // as-is) use the requested accuracy. Callers that merge call-time `settings`
    // into their own FitOptions (the R wrapper's ferx_fit) must re-apply
    // sync_ode_solver_opts on the owned model for those overrides to win;
    // ferx-core fit() takes &CompiledModel and does not. No-op for analytical.
    model.sync_ode_solver_opts(&fit_options);

    // ── [scaling] block ──
    // Parsed after `parse_fit_options` so option-dependent wiring (Form-C ODE readout,
    // estimator/gradient context) is available. (`gradient = ad` is retired and rejected
    // unconditionally by `check_model_options` — see the sibling note below; there is no
    // longer an `ExpressionScale + ad` combination to validate here.)
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
        // Individual-parameter name bound to this `pk <model>(...)` block's `v`/`v1` role
        // (`None` for `ode(...)`/`ode_template(...)`) — see `build_obs_scale_spec`'s doc (#712).
        let volume_indiv_name_for_scaling: Option<String> = if is_ode_model {
            None
        } else {
            pk_param_map
                .get("v")
                .or_else(|| pk_param_map.get("v1"))
                .cloned()
        };
        let mut scaling_parse_warnings: Vec<String> = Vec::new();

        let (scaling, output_fn, output_program, scaling_covariates) = parse_scaling_block(
            scaling_lines,
            &theta_names_for_scaling,
            &eta_names_for_scaling,
            &indiv_var_names_for_scaling,
            &pk_indices_for_scaling,
            &state_names_for_scaling,
            is_ode_model,
            model.pk_model,
            &model.kappa_names,
            &readout_synth_params,
            volume_indiv_name_for_scaling.as_deref(),
            &declared_covariate_names,
            &mut scaling_parse_warnings,
        )?;
        model.parse_warnings.extend(scaling_parse_warnings);

        // #993, the `[scaling]` half of the dose-attribute double-use rejection
        // (`[odes]`'s lives in `build_ode_spec`). A readout that divides by `F` gets
        // bioavailability applied once at the dose and once here — measured at
        // exactly `F` on the prediction, for `y` and `obs_scale` alike. Runs after
        // `parse_scaling_block` so a malformed block reports its own syntax error
        // first, and re-walks the entries because the parsed `ScalingSpec` keeps
        // compiled programs rather than the name references.
        //
        // The read-set is collected once and serves both engines: the rejection
        // below splits on the engine (ODE via `ode_slot_map`, analytical via the
        // `pk(...)` mapping — #1004), the `D{n}`/`R{n}` recording does not.
        let mut scaling_reads: std::collections::HashSet<String> = std::collections::HashSet::new();
        // Named intermediates are inlined first (#1030), so a readout that reaches
        // `F` through one is caught by the double-use rejection exactly as if it had
        // been written inline. Scanning the intermediate lines directly instead would
        // over-report — an intermediate no entry uses is rejected upstream, but one
        // used by `y` only would still be charged against `obs_scale`.
        let scaling_intermediates_for_reads = scaling_intermediates(scaling_lines)?;
        for line in scaling_lines {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let (key, value) = split_scaling_entry(trimmed)?;
            let (base, _cmt) = parse_scaling_key(key)?;
            if base != "y" && base != "obs_scale" {
                continue;
            }
            scaling_reads.extend(collect_indiv_param_reads(
                value,
                &theta_names_for_scaling,
                &eta_names_for_scaling,
                &indiv_var_names_for_scaling,
                &scaling_intermediates_for_reads,
                &format!("[scaling] {base}"),
            )?);
        }

        // Both engines. The ODE side (#993) keys off `ode_slot_map` — name-based
        // routing is what makes a bare `F` a dose attribute there. The analytical
        // side (#1004) keys off the explicit `pk(..., f=F)`/`lagtime=` mapping:
        // #1003's scope note argued that mapping made a second use "stated rather
        // than silent", but nothing in the model states the value is applied
        // twice, which is the same silence #993 closed — measured at exactly `F`
        // on the prediction on both ferx and NONMEM's analytical ADVAN2 routine
        // (`S2 = V/F1` with `F1` defined; see the anchor test).
        if is_ode_model {
            check_dose_attr_double_use(
                &indiv_var_names_for_scaling,
                &ode_slot_map,
                &scaling_reads,
                state_names_for_scaling.len(),
                "[scaling]",
                "[scaling]",
                None,
            )?;
        } else {
            let analytical_slots =
                analytical_dose_attr_slot_map(&pk_param_map, &indiv_var_names_for_scaling);
            check_dose_attr_double_use(
                &indiv_var_names_for_scaling,
                &analytical_slots,
                &scaling_reads,
                usize::MAX,
                "[scaling]",
                "[scaling]",
                Some(&pk_param_map),
            )?;
        }

        // Record a `D{n}`/`R{n}` read by the readout, on either engine, for the
        // data-aware check (#993 — see `mark_prediction_path_read`). The ODE RHS half
        // is recorded in `build_ode_spec`; this adds the readout half, which is the
        // *only* prediction path an analytical model has.
        record_coded_rate_reads(
            &mut model,
            &indiv_var_names_for_scaling,
            &ode_slot_map,
            &scaling_reads,
        );

        // AD compatibility check (Phase 2.5):
        //
        // (`gradient = ad` no longer needs a Form-C-specific guard here: it is
        // retired and rejected unconditionally by `check_model_options`.)

        // Form C wiring. ODE: replace the ODE readout (set to the
        // `NEEDS_FORM_C = usize::MAX` sentinel by `build_ode_spec` if the user
        // omitted `obs_cmt=`) with the parsed Single/PerCmt readout. Analytic
        // (#650): store the readout on `model.analytic_readout`, where the
        // closed-form predictor and the analytic sensitivity provider pick it up
        // (analytical models have `ode_spec == None`).
        if let Some(new_readout) = output_fn {
            if is_ode_model {
                let ode_spec = model.ode_spec.as_mut().expect("guarded by is_ode_model");
                ode_spec.readout = new_readout;
                // Form C sensitivity program (issue #367); `None` for per-CMT.
                ode_spec.readout_program = output_program;
            } else {
                let (state_names, _forbidden) = analytic_readout_state_names(model.pk_model);
                // Warn (not silent) when the readout can't ride the analytic Dual2
                // provider — it stays correct via FD of the readout-aware predictor,
                // but the user loses the analytic-gradient speedup (#650). The
                // dual-evaluable case (indiv-params/covariates only) is served
                // analytically by the static superposition path.
                let has_depot_slot = matches!(
                    model.pk_model,
                    PkModel::OneCptOral | PkModel::TwoCptOral | PkModel::ThreeCptOral
                );
                let dual_ok = model.analytical_init.is_empty()
                    && matches!(
                        (&new_readout, &output_program),
                        (crate::ode::OdeReadout::Single(_), Some(p))
                            if p.is_dual_evaluable() && !(has_depot_slot && p.references_state(0))
                    );
                if !dual_ok {
                    model.parse_warnings.push(
                        "[scaling] y: this analytic Form C readout (a per-CMT readout, a \
                         direct THETA/ETA reference, a neural-network output, or a model with \
                         [initial_conditions]) falls back to finite-difference gradients. The \
                         prediction is exact; only the analytic-gradient speedup is lost. Use \
                         individual-parameter / covariate references in the readout to keep it \
                         analytic. See issue #650."
                            .to_string(),
                    );
                }
                model.analytic_readout = Some(crate::types::AnalyticReadout {
                    readout: new_readout,
                    program: output_program,
                    state_names,
                });
            }
        }

        register_referenced_covariates(&mut model.referenced_covariates, scaling_covariates);
        model.scaling = scaling;
    }

    // Covariate-selected [error_model] (issue #658): the selector's covariates
    // are required data columns, so register them like scaling covariates.
    if !selector_covariates.is_empty() {
        register_referenced_covariates(&mut model.referenced_covariates, selector_covariates);
    }

    // ── [initial_conditions] block (issue #521) ──
    // Non-zero starting compartment amounts for analytical PK models. The ODE
    // path seeds state via `init(...)` in [odes]; here we record closed-form
    // init impulses layered onto the dose-driven prediction by
    // `pk::add_analytical_init`.
    if let Some(ic_lines) = blocks.get("initial_conditions") {
        let (analytical_init, init_covariates) = parse_initial_conditions_block(
            ic_lines,
            model.pk_model,
            model.ode_spec.is_some(),
            &model.theta_names,
            &model.eta_names,
            &model.indiv_param_names,
            &model.pk_indices,
            &model.kappa_names,
        )?;
        // Deliberately NOT a `check_dose_attr_double_use` site, unlike `[scaling]`
        // and `[adaptive_dosing] observe` (#1004). An analytical init amount is not
        // a dose: `pk::analytical_init_concentration_g` propagates it with `F = 1`
        // and no lag ("an initial condition is not an absorbed dose, so
        // bioavailability does not apply"), so `init(depot) = F * 500` — the
        // bioavailable residue of a pre-study dose — applies `F` exactly once, to a
        // term the engine never scales. `[scaling]` is a doubling because
        // `obs_scale` multiplies *every* prediction including the F-scaled dose
        // contribution; an init is one additive term beside it. For the same
        // reason a `D{n}`/`R{n}` read here is not a prediction-path read and is
        // not recorded for `check_modeled_dose_rates`.
        model.analytical_init = analytical_init;
        // Register init-expression covariates as required data columns, mirroring
        // the scaling (`scaling_covariates`) and error-selector blocks above. Without
        // this, a `[covariates]`-declared model never reads the column, and a miscased
        // or absent init covariate silently evaluates to 0 (issue #765).
        register_referenced_covariates(&mut model.referenced_covariates, init_covariates);
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
            // Covariate-selected error models (#658) are multi-endpoint: the EKF
            // `p_obs` measurement-noise closure (`stats/likelihood.rs`) binds a
            // single representative `error_model` (branch 0) and cannot switch per
            // row, so a `Selected` spec would silently score every observation with
            // branch 0's sigma. The `debug_assert!` there is a release no-op — reject
            // at parse instead. (Per-CMT/Form-C is already rejected via the ObsCmt
            // readout guard above; this closes the remaining multi-endpoint case.)
            if matches!(model.error_spec, ErrorSpec::Selected { .. }) {
                return Err(
                    "covariate-selected `[error_model]` (`if (COV …) { … } else { … }`) is not \
                     supported on SDE / [diffusion] models — the EKF measurement-noise path uses \
                     a single error model and cannot switch per observation. Use a single-endpoint \
                     error model, or drop the [diffusion] block."
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
            // The analytic inner-gradient kernels can't represent an if-branch that
            // assigns an eta-bearing parameter, so flag the model
            // (`has_conditional_eta_params`); the live inner gate
            // (`analytic_inner_grad_supported_model`) routes such models to FD.
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

    // ── Custom residual-error magnitude (#484) ──────────────────────────────
    // Compile per-sigma magnitude multipliers now that covariate names are
    // known (they validate the magnitude expressions). `None` for the common
    // case of bare sigmas — the legacy variance path is then taken verbatim.
    let ruv_magnitude_used_thetas: std::collections::HashSet<usize> = {
        let ruv_theta_names = model.theta_names.clone();
        let ruv_eta_names = model.eta_names.clone();
        let (rm, used_thetas, used_covs) = build_ruv_magnitude(
            &single_error_args,
            &ruv_sigma_names,
            &ruv_theta_names,
            &ruv_eta_names,
            &covariate_decls,
        )?;
        model.ruv_magnitude = rm;
        // A covariate that appears *only* in a magnitude / `weight =` expression
        // is still model-referenced: without registering it here,
        // `prune_irrelevant_tv_covariates` would drop the per-observation
        // snapshots for a subject whose only time-varying covariate is the
        // magnitude's, and every observation would silently be scored with the
        // subject's *first* value (#484 / #1029).
        register_referenced_covariates(&mut model.referenced_covariates, used_covs);
        used_thetas
    };

    // Compile and attach the ODE-equivalent sub-model whose source was reconstructed above
    // (`None` for non-transit / out-of-scope forms). The runtime dispatch reads it off the
    // model.
    //
    // The twin is built **here, at parse time**, and a build failure declines it rather than
    // propagating (#1008). The twin is an ODE model while this primary is analytical, so every
    // ODE-scoped parse check runs on the twin and not on the primary: a model this parser
    // accepts can reconstruct into a twin the parser rejects. Building lazily and `.expect()`ing
    // that parse turned each such case into an internal panic mid-fit, the first time a
    // TV-covariate / `TIME` / IOV / SS / infusion subject rerouted — invisible to the model
    // author and to the parse tests. Three point guards in `absorption_ode_equivalent_source`
    // were added one-per-incident for exactly this (`f=V1` slot collision, `[adaptive_dosing]`
    // re-emission, a dose-attribute param in a disposition role); this makes the class
    // structurally unreachable, so a *new* ODE-only check cannot re-arm it.
    //
    // Declining is the standing policy of the desugar, and it is not silent-wrong: without a
    // twin the model simply stays closed-form, and a subject that actually needs the fallback
    // is rejected up front with an actionable message — `check_absorption_closed_form_support`
    // (TV covariates / IOV / SS / infusion / resets) and `check_absorption_flip_flop_no_twin`,
    // both of which already key on `absorption_ode_equivalent.is_none()` and both of which run
    // on every entry point (`fit` as an `Err`, `predict`/`simulate` as their `assert_*`
    // wrappers). The warning carries the twin parser's own message so the reason is visible
    // without a debugger.
    if let Some(src) = absorption_ode_equivalent_src {
        match crate::types::AbsorptionOdeEquivalent::build(&src) {
            Ok(eq) => model.absorption_ode_equivalent = Some(eq),
            Err(e) => model
                .parse_warnings
                .push(crate::types::absorption_twin_declined_warning(&e)),
        }
    }

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
    #[cfg_attr(not(feature = "survival"), allow(unused_mut))]
    let mut event_model_used_thetas: std::collections::HashSet<usize> =
        std::collections::HashSet::new();
    #[cfg_attr(not(feature = "survival"), allow(unused_mut))]
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
            let (cmt, endpoint, event_covs, blk_thetas, blk_etas) = parse_event_model_block(
                lines,
                &theta_names,
                &eta_names,
                &indiv_stmts,
                &model.kappa_names,
                &model.error_spec,
                &chz_state_map,
            )?;
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

        // ── [binary_model] blocks (Phase 4 categorical / logistic, #760) ──────────
        // Unnamed `[binary_model]` and any named `[binary_model LABEL]`, each keyed by CMT.
        let mut binary_blocks: Vec<&Vec<String>> = Vec::new();
        if let Some(lines) = blocks.get("binary_model") {
            binary_blocks.push(lines);
        }
        if let Some(named_map) = extracted.named.get("binary_model") {
            for lines in named_map.values() {
                binary_blocks.push(lines);
            }
        }
        for lines in binary_blocks {
            let (cmt, endpoint, bin_covs, blk_thetas, blk_etas) = parse_binary_model_block(
                lines,
                &theta_names,
                &eta_names,
                &indiv_stmts,
                &model.kappa_names,
                &model.error_spec,
            )?;
            if model.endpoints.contains_key(&cmt) {
                return Err(format!(
                    "[binary_model]: CMT={cmt} already declared by another endpoint"
                ));
            }
            model.endpoints.insert(cmt, endpoint);
            for cov in bin_covs {
                if !model.referenced_covariates.contains(&cov) {
                    model.referenced_covariates.push(cov);
                }
            }
            event_model_used_thetas.extend(blk_thetas);
            event_model_used_etas.extend(blk_etas);
        }

        // ── [markov_model] blocks (Phase 5 CTMM, #759) ────────────────────────────
        // Unnamed `[markov_model]` and any named `[markov_model LABEL]`, each keyed by CMT.
        #[cfg(feature = "markov")]
        {
            let mut markov_blocks: Vec<&Vec<String>> = Vec::new();
            if let Some(lines) = blocks.get("markov_model") {
                markov_blocks.push(lines);
            }
            if let Some(named_map) = extracted.named.get("markov_model") {
                for lines in named_map.values() {
                    markov_blocks.push(lines);
                }
            }
            // ODE state names (in state-vector order) let an intensity reference a model
            // state for the time-inhomogeneous path (#817). `build_ode_spec` has already
            // run, so `model.ode_spec` is populated for an ODE model; analytic/no-ODE
            // models pass an empty slice (no state to drive Q, so any such reference stays
            // an unresolved identifier as before).
            let ode_state_names: &[String] = model
                .ode_spec
                .as_ref()
                .map(|s| s.state_names.as_slice())
                .unwrap_or(&[]);
            for lines in markov_blocks {
                let declared_cov_names: Vec<String> = covariate_decls
                    .as_ref()
                    .map(|d| d.iter().map(|c| c.name.clone()).collect())
                    .unwrap_or_default();
                let (cmt, endpoint, ctmm_covs, blk_thetas, blk_etas) = parse_markov_model_block(
                    lines,
                    &theta_names,
                    &eta_names,
                    &indiv_stmts,
                    &model.kappa_names,
                    &model.error_spec,
                    ode_state_names,
                    &declared_cov_names,
                )?;
                if model.endpoints.contains_key(&cmt) {
                    return Err(format!(
                        "[markov_model]: CMT={cmt} already declared by another endpoint"
                    ));
                }
                model.endpoints.insert(cmt, endpoint);
                for cov in ctmm_covs {
                    if !model.referenced_covariates.contains(&cov) {
                        model.referenced_covariates.push(cov);
                    }
                }
                event_model_used_thetas.extend(blk_thetas);
                event_model_used_etas.extend(blk_etas);
            }
        }

        model.referenced_covariates.sort();
    }

    // Warn about declared-but-unused parameters. Runs here (after [event_model]
    // parsing) so that parameters used only in [event_model] are not falsely
    // reported as unused in mixed PK+TTE models. A theta referenced only from a
    // custom RUV magnitude expression (#484/#576, e.g. `RUV_LATE`) is unioned in
    // too, the same way — it is meaningfully estimated (the analytic θ/σ gradient
    // now carries its direct-θ channel), just never seen by `indiv_stmts`.
    event_model_used_thetas.extend(ruv_magnitude_used_thetas);
    // #486 — surface the FD fallback when a direct-θ/η readout could not be
    // desugared because the PK-slot layout was full (see the headroom check above).
    if let Some(note) = readout_fd_fallback_note.take() {
        model.parse_warnings.push(note);
    }

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
        model.residual_error_eta,
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
            // Individual parameters the parser synthesized for a direct-θ/η Form-C
            // readout (`__ferx_ro_*`, #486) are referenced by the readout via the
            // post-census rewrite (`rewrite_readout_synth`), not in the raw block
            // text the census tokenizes — so they always look "dead" here. They are
            // load-bearing (the readout reads them), so exempt them.
            .filter(|(_, name)| !is_synthetic_readout_param(name))
            // Exempt parameters the *engine* applies to a dose without a textual
            // reference, so their absence from the RHS / `pk(...)` mapping does not
            // make them dead:
            //  * ODE models: bare `f`/`lagtime`/`alag` (routed to RESERVED_PK_SLOTS)
            //    and every compartment-indexed dose attribute `Fn`/`ALAGn`/`Dn`/`Rn`
            //    (#369/#324) — the bare case reuses `ode_param_slots`' own routing
            //    (`ode_slot_map`) so the exemption can't drift; the indexed case
            //    matches the same `from_indexed_name` predicate that built
            //    `dose_attr_map`.
            //  * analytical models: only `Dn`/`Rn` (modeled duration / rate,
            //    RATE=-2/-1), which the parser routes to spare `PkParams` slots and
            //    which are consulted solely via coded-`RATE` *data* — they have no
            //    textual reference, so the census would otherwise tell the user to
            //    delete a load-bearing modeled-infusion parameter. `Fn`/`ALAGn` are
            //    NOT exempt here: the analytical engine binds F/lag only via an
            //    explicit `pk(...)` mapping (which the census counts), so an
            //    unmapped `F1` really is dead.
            .filter(|(i, name)| {
                use crate::types::DoseAttr;
                let exempt = if is_ode {
                    ode_slot_map
                        .get(*i)
                        .is_some_and(|slot| RESERVED_PK_SLOTS.contains(slot))
                        || DoseAttr::from_indexed_name(name).is_some()
                } else {
                    matches!(
                        DoseAttr::from_indexed_name(name),
                        Some((DoseAttr::Duration | DoseAttr::Rate, _))
                    )
                };
                !exempt
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
        adaptive_dosing,
        fit_options,
        data_path,
        column_map: data_column_map,
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
        // A bare `TIME` parses to `Expression::Time`, resolved from the model-time
        // thread-local; set it to this row's event time so `[derived]` columns
        // honour the `TIME` built-in instead of reading the 0.0 default (#610).
        let _time_guard = ModelTimeGuard::enter(ctx.time);
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
        // Match `build_derived_eval_fn`: a `TIME`-keyed filter sees this row's
        // event time, not the 0.0 default (#610).
        let _time_guard = ModelTimeGuard::enter(ctx.time);
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
                } else if matches!(fname_lc.as_str(), "max" | "min")
                    && args.len() == 2
                    && !tokens_contain_comparison(args[1])
                {
                    // Two-argument numeric `min(a, b)` / `max(a, b)` (#1030) shares a
                    // spelling with the aggregate's `min(<value>, <row filter>)`.
                    // Disambiguate the way the rest of the parser already does: a
                    // second argument with no comparison operator cannot be a row
                    // filter, so this is the numeric form. Parsed as a plain expression
                    // over the *whole* statement, so it means the same thing at
                    // statement top level as it does nested one level in.
                    let expr = parse_derived_expr(&tokens, ctx)?;
                    let uses = expr_refs_compartments(&expr, ctx.ode_state_names);
                    (
                        DerivedKind::PerRow {
                            eval: build_derived_eval_fn(expr),
                        },
                        uses,
                    )
                } else {
                    // max / min / tmax
                    // Trailing tokens after the aggregate's closing `)` used to be
                    // silently dropped: `max(A1) * 2` computed `max(A1)`. No aggregate
                    // form continues into a larger expression, so say so.
                    if close_idx + 1 != tokens.len() {
                        return Err(format!(
                            "[derived] `{name}`: unexpected token(s) after `{fname_lc}(…)` — an \
                             aggregate over rows cannot be combined into a larger expression."
                        ));
                    }
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

/// Collect the names referenced as `Expression::Variable` in `expr` (i.e.
/// `[individual_parameters]` names an `[event_model]` expression depends on).
#[cfg(feature = "survival")]
fn collect_variable_names(expr: &Expression, out: &mut std::collections::HashSet<String>) {
    visit_expr_nodes(expr, &mut |e: &Expression| {
        if let Expression::Variable(name) = e {
            out.insert(name.clone());
        }
    });
}

/// Evaluate the hazard-reachable `[individual_parameters]` statements into a
/// name→value map for the given `(θ, η, covariates)`, so `[event_model]` hazard
/// expressions can reference individual-parameter names (e.g. a hazard driven by
/// an individual `CL`).
///
/// `stmts` is the reachability-filtered subset (`needed_indiv_stmts`) built in
/// `parse_event_model_block`. It is **empty** for a hazard that references no
/// individual parameter — the common TTE-only case — and this then returns an
/// empty map without allocating, so such hazards pay nothing for the feature.
///
/// Evaluation uses the tree-walking `eval_statements`, NOT the bytecode
/// bytecode indexed evaluators: those borrow the `FERX_SCRATCH` thread-local,
/// so this is safe to call from inside the likelihood's hazard `param_fn`, and
/// it transparently handles NONMEM-style `if (...) CL = ...` conditional
/// parameters. Kappa (IOV) and `[covariate_nn]`-output references in these
/// statements are rejected up front (see `needed_indiv_stmts`), so the BSV-only
/// `eta` slice never indexes out of bounds and the empty `nn_outputs` here is
/// never read.
#[cfg(feature = "survival")]
fn eval_indiv_param_vars(
    stmts: &[Statement],
    theta: &[f64],
    eta: &[f64],
    covariates: &HashMap<String, f64>,
) -> HashMap<String, f64> {
    if stmts.is_empty() {
        return HashMap::new();
    }
    let mut vars = HashMap::with_capacity(stmts.len());
    eval_statements(stmts, theta, eta, covariates, &mut vars, None, None, &[]);
    vars
}

/// Shared predictor validation + `[individual_parameters]` restriction for the
/// `[event_model]` and `[binary_model]` blocks.
///
/// Both blocks carry a linear predictor over θ/η/covariates/`[individual_parameters]`
/// — a hazard parameter expression, or the `logit` log-odds — and both must enforce
/// the same four invariants on it (previously duplicated verbatim in each block):
///
/// 1. reject a **direct** IOV-κ reference (#770): the predictor is evaluated once per
///    subject with BSV-only η, so a kappa name has no occasion context and would
///    otherwise fall back to a leniently-read `0.0` covariate, silently dropping it;
/// 2. restrict the individual-parameter statements to those the predictor references,
///    transitively (#442) — bounding per-eval work and scoping checks 3–4 to the
///    parameters the predictor actually depends on;
/// 3. reject an individual-parameter value that reaches an IOV-κ through η (#442);
/// 4. reject an individual-parameter value that reads a `[covariate_nn]` output
///    (#293, `nn`-gated) — it would silently resolve to `0.0` without the forward pass.
///
/// Returns the restricted statements (for the predictor closure) and the predictor's
/// covariate set — the **union** of covariates referenced directly and transitively
/// through the kept statements, so a time-varying covariate reached only via an
/// individual parameter cannot slip past the #741 TV-covariate guard silently frozen.
/// θ/η index collection stays with each caller (it returns those for its own
/// dead-parameter census). `block` names the block for error messages.
///
/// The two blocks previously each carried these four steps verbatim; keeping them here
/// once means a fix to any #770/#442/#741/#293 invariant lands for both endpoints.
#[cfg(feature = "survival")]
fn restrict_and_validate_indiv_stmts(
    exprs: &[&Expression],
    indiv_stmts: &[Statement],
    eta_names: &[String],
    kappa_names: &[String],
    block: &str,
) -> Result<(Vec<Statement>, Vec<String>), String> {
    // Block-appropriate noun for the predictor in error messages — `[event_model]` carries a
    // hazard, `[binary_model]` the logit — so a user sees the same wording as before the two
    // blocks shared this helper.
    let predictor = match block {
        "event_model" => "a hazard expression",
        "binary_model" => "the logit predictor",
        _ => "a predictor expression",
    };
    // (1) Direct IOV-κ reference — fail loud (#770).
    for expr in exprs {
        if let Some(k) = expr_references_kappa(expr, kappa_names) {
            return Err(format!(
                "[{block}]: {predictor} references the inter-occasion (IOV) random \
                 effect `{k}` directly. The predictor is evaluated once per subject with \
                 per-subject (BSV) η only, so an IOV kappa has no well-defined value here — write \
                 it in terms of θ/η, or reference an IOV-free parameter."
            ));
        }
    }

    // (2) Restrict `[individual_parameters]` to what the predictor references, transitively.
    // Walk declarations in reverse so a needed parameter pulls in the earlier-declared
    // parameters it depends on. Conditional (`if (...) CL = ...`) parameters are handled by
    // keying on every assigned name (`assigned_vars_in_order`) and pulling in every referenced
    // name (`visit_stmt_nodes`, covering RHS, conditions, and both branches).
    let needed_indiv_stmts: Vec<Statement> = {
        let mut needed: std::collections::HashSet<String> = std::collections::HashSet::new();
        for expr in exprs {
            collect_variable_names(expr, &mut needed);
        }
        let mut keep: Vec<Statement> = Vec::new();
        for s in indiv_stmts.iter().rev() {
            let stmt = std::slice::from_ref(s);
            if assigned_vars_in_order(stmt)
                .iter()
                .any(|n| needed.contains(n))
            {
                visit_stmt_nodes(stmt, &mut |e: &Expression| {
                    if let Expression::Variable(name) = e {
                        needed.insert(name.clone());
                    }
                });
                keep.push(s.clone());
            }
        }
        keep.reverse();
        keep
    };

    // (3) Covariate set: union of covariates referenced directly by the predictor and
    // transitively through the kept individual-parameter statements (#741).
    let covariates: Vec<String> = {
        let mut cov_set = std::collections::HashSet::new();
        for expr in exprs {
            collect_covariates(expr, &mut cov_set);
        }
        collect_covariates_in_stmts(&needed_indiv_stmts, &mut cov_set);
        let mut v: Vec<String> = cov_set.into_iter().collect();
        v.sort();
        v
    };

    // (4) An individual-parameter value that depends on an IOV-κ (BSV-only η here) — fail
    // loud (#442). `eta_names` is BSV-only, so any `Eta(i)` with `i >= n_eta` is the kappa at
    // position `i - n_eta` in `kappa_names`.
    {
        let n_eta = eta_names.len();
        let mut kappa_hit: Option<String> = None;
        visit_stmt_nodes(&needed_indiv_stmts, &mut |e: &Expression| {
            if let Expression::Eta(i) = e {
                if *i >= n_eta && kappa_hit.is_none() {
                    kappa_hit = Some(
                        kappa_names
                            .get(*i - n_eta)
                            .cloned()
                            .unwrap_or_else(|| "<kappa>".to_string()),
                    );
                }
            }
        });
        if let Some(k) = kappa_hit {
            return Err(format!(
                "[{block}]: {predictor} references an [individual_parameters] value \
                 that depends on the inter-occasion (IOV) random effect `{k}`. The predictor is \
                 evaluated with per-subject (BSV) η only — reference an IOV-free parameter, or \
                 write the predictor in terms of θ/η directly."
            ));
        }
    }

    // (5) An individual-parameter value whose definition reads a `[covariate_nn]` output would
    // silently resolve to `0.0` (the predictor evaluates these statements without the network
    // forward pass). Reject it — gated to `nn` like the rest: `Expression::NnOutput` nodes only
    // exist under the `nn` feature, so the survival coverage build (no `nn`) does not compile
    // this — a measurement gap, not missed coverage (#293) — and it is unreachable without `nn`.
    #[cfg(feature = "nn")]
    {
        let mut nn_hit = false;
        visit_stmt_nodes(&needed_indiv_stmts, &mut |e: &Expression| {
            if matches!(e, Expression::NnOutput { .. }) {
                nn_hit = true;
            }
        });
        if nn_hit {
            return Err(format!(
                "[{block}]: {predictor} references an [individual_parameters] value \
                 whose definition uses a [covariate_nn] output. Neural-network-driven individual \
                 parameters are not available to the predictor (it is evaluated without the \
                 network forward pass) — reference an NN-free parameter instead."
            ));
        }
    }

    Ok((needed_indiv_stmts, covariates))
}

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
/// - `type`   — optional; `tte` (default, single event) | `rtte` (repeated events)
/// - `clock`  — optional (RTTE only); `forward` (default, Andersen–Gill total time) | `reset` (gap time / renewal — hazard clock resets at each event)
#[cfg(feature = "survival")]
fn parse_event_model_block(
    lines: &[String],
    theta_names: &[String],
    eta_names: &[String],
    indiv_stmts: &[Statement],
    kappa_names: &[String],
    error_spec: &ErrorSpec,
    chz_state_map: &std::collections::HashMap<usize, usize>,
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

    // `[event_model]` hazard expressions may reference names defined in
    // `[individual_parameters]`. Register every assigned name — including those
    // inside `if (...) { ... }` branches (NONMEM-style conditional parameters) and
    // intermediate helpers — as `defined_vars`, so such a reference parses as
    // `Expression::Variable` and is resolved per subject from `needed_indiv_stmts`
    // (below) rather than silently falling back to a covariate, which would read
    // 0.0. Names that are neither θ/η nor an individual parameter still fall back
    // to covariates. `kappa_names` is used below to reject IOV references that the
    // per-subject hazard cannot evaluate.
    let indiv_param_names = assigned_vars_in_order(indiv_stmts);
    let ctx = ParseCtx::new(theta_names, eta_names, &indiv_param_names);

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
    // ODE-accumulated hazard (joint PK-TTE): `hazard = <expr>` is injected as a
    // cumulative-hazard ODE derivative before the ODE spec is built; here we only
    // record its presence and route to the OdeAccumulated endpoint below.
    let mut hazard_present = false;
    // Recurrence (RTTE): `type = tte | rtte` (default tte) and, for RTTE,
    // `clock = forward | reset` (default forward). Resolved into `TteRecurrence`
    // after the key loop. §3.3 of `plans/tte-survival-markov.md`.
    let mut type_opt: Option<String> = None;
    let mut clock_opt: Option<String> = None;

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
            "hazard" => {
                // The expression itself is injected as the `d/dt(__chz_<cmt>)` ODE
                // derivative by the parser pre-scan, so it is not compiled here.
                hazard_present = true;
            }
            "type" => {
                type_opt = Some(value.to_string());
            }
            "clock" => {
                clock_opt = Some(value.to_string());
            }
            other => {
                return Err(format!(
                    "[event_model]: unknown key `{other}` — valid keys: cmt, family, scale, \
                     rate, shape, alpha, gamma, loghr, hazard, type, clock"
                ));
            }
        }
    }

    let cmt = cmt_opt.ok_or("[event_model]: missing required key `cmt`")?;

    // Guard: same CMT can't be both Gaussian and TTE (applies to the analytic and
    // the ODE-accumulated `hazard =` endpoint paths alike).
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
        ErrorSpec::Selected { .. } => {
            // A covariate-selected error model (#658) keys its endpoints by the per-row
            // selector branch, not by CMT, so — like `Single` — it carries no CMT
            // information to collide against the TTE CMT at parse time.  The data reader's
            // two-path routing still keeps Gaussian and TTE observations from
            // double-counting; use a per-CMT error model for parse-time collision checks.
        }
    }

    // Resolve the recurrence: standard single-event TTE (default) vs. repeated TTE
    // (RTTE). For RTTE, `clock = forward` (Andersen–Gill total time, the default)
    // accumulates the hazard continuously; `clock = reset` (gap time, §3.3) restarts the
    // hazard clock at each event. `clock` is meaningless without `type = rtte`.
    let recurrence = {
        use crate::types::{RtteClock, TteRecurrence};
        match type_opt.as_deref().unwrap_or("tte") {
            "tte" | "single" => {
                if let Some(c) = clock_opt.as_deref() {
                    return Err(format!(
                        "[event_model]: CMT={cmt} sets `clock = {c}` on a single-event TTE \
                         endpoint. `clock` selects how time is measured between repeated events, \
                         so it is only valid with `type = rtte`."
                    ));
                }
                TteRecurrence::Single
            }
            "rtte" => {
                let clock = match clock_opt.as_deref().unwrap_or("forward") {
                    "forward" => RtteClock::Forward,
                    "reset" => RtteClock::Reset,
                    other => {
                        return Err(format!(
                            "[event_model]: CMT={cmt} unknown clock `{other}` — valid: forward, reset"
                        ));
                    }
                };
                TteRecurrence::Repeated { clock }
            }
            other => {
                return Err(format!(
                    "[event_model]: CMT={cmt} unknown type `{other}` — valid: tte, rtte"
                ));
            }
        }
    };

    // ODE-accumulated hazard path (joint PK-TTE, Slice 2.1): `hazard = <expr>` was
    // injected as a cumulative-hazard ODE derivative (`d/dt(__chz_<cmt>)`); the
    // endpoint only records that accumulator's state index. Mutually exclusive with
    // the analytic closed-form `family` keys.
    if hazard_present {
        // RTTE with a drug-driven ODE hazard (joint PK-RTTE) needs the repeated-event
        // root-finder loop AND, for clock-reset, the selective CHZ reset — a later
        // slice. Reject rather than silently fitting a single-event likelihood.
        if matches!(recurrence, crate::types::TteRecurrence::Repeated { .. }) {
            return Err(format!(
                "[event_model]: CMT={cmt} combines `type = rtte` with an ODE-accumulated \
                 `hazard = ...` (drug-driven joint PK-RTTE), which is not yet supported. Use an \
                 analytic `family` hazard for repeated events, or `type = tte` for a single-event \
                 drug-driven hazard."
            ));
        }
        if family_opt.is_some()
            || scale_expr.is_some()
            || shape_expr.is_some()
            || alpha_expr.is_some()
            || gamma_expr.is_some()
            || loghr_expr.is_some()
        {
            return Err(format!(
                "[event_model]: CMT={cmt} mixes `hazard = ...` (ODE-accumulated, joint \
                 PK-TTE) with analytic keys (family/scale/rate/shape/alpha/gamma/loghr). \
                 Use either a closed-form `family` or an ODE `hazard` expression, not both \
                 — put drug and covariate effects directly in the hazard expression."
            ));
        }
        if !kappa_names.is_empty() {
            return Err(format!(
                "[event_model]: CMT={cmt} declares an ODE-accumulated `hazard = ...` (joint \
                 PK-TTE) in a model with inter-occasion variability (IOV / kappa). The \
                 PK-driven hazard needs per-occasion random effects threaded into the ODE \
                 solve, which is not yet supported (Slice 2.1) — remove IOV, or use an \
                 analytic `family` hazard instead."
            ));
        }
        let chz_state = *chz_state_map.get(&cmt).ok_or_else(|| {
            format!(
                "[event_model]: CMT={cmt} declares `hazard = ...` but the model has no ODE \
                 system. ODE-accumulated hazard (joint PK-TTE) requires an ODE model — use \
                 `ode(...)` in [structural_model] and define [odes]."
            )
        })?;
        // The hazard compiles as the injected ODE derivative, so its θ/η/covariate
        // dependencies flow through [individual_parameters] / [odes] (already covered by
        // check_unused_parameters and the dead-parameter census) — none to report here.
        return Ok((
            cmt,
            EndpointLikelihood::Tte {
                hazard: HazardSpec::OdeAccumulated { chz_state },
                recurrence,
                // The ODE-accumulated hazard's covariate dependencies flow through the
                // injected `d/dt(__chz)` derivative; the whole-subject time-varying check
                // (the ODE solve freezes PK params at t=0) guards them, not this list.
                hazard_covariates: Vec::new(),
            },
            Vec::new(),
            std::collections::HashSet::new(),
            std::collections::HashSet::new(),
        ));
    }

    let family = family_opt.ok_or("[event_model]: missing required key `family`")?;

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

    // The hazard's parameter expressions in optimizer-parameter order. The covariate
    // union, `[individual_parameters]` restriction, and the direct/via-indiv IOV-κ
    // (#770/#442) and `[covariate_nn]` (#293) rejects are shared with `[binary_model]`
    // through `restrict_and_validate_indiv_stmts`; only the θ/η index sets (for the
    // dead-parameter census) are collected here, since each block returns its own.
    let hazard_param_exprs = [
        &scale_expr,
        &shape_expr,
        &alpha_expr,
        &gamma_expr,
        &loghr_expr,
    ];
    let hazard_exprs: Vec<&Expression> = hazard_param_exprs.into_iter().flatten().collect();
    let (event_model_thetas, event_model_etas) = {
        let mut theta_set = std::collections::HashSet::new();
        let mut eta_set = std::collections::HashSet::new();
        for expr in hazard_exprs.iter().copied() {
            collect_theta_eta(expr, &mut theta_set, &mut eta_set);
        }
        (theta_set, eta_set)
    };
    let (needed_indiv_stmts, event_model_covariates) = restrict_and_validate_indiv_stmts(
        &hazard_exprs,
        indiv_stmts,
        eta_names,
        kappa_names,
        "event_model",
    )?;

    // Build the param_fn closure that evaluates hazard parameters from (θ, η, covariates).
    // Expression nodes hold only indices, so they're safe to move into the closure.
    // Parameter layout matches parametric.rs: [scale/alpha, (shape/gamma), loghr].
    let param_fn: crate::types::HazardParamFn = match family {
        HazardFamily::Exponential => {
            let scale = scale_expr
                .ok_or("[event_model] family=exponential requires `scale` (or `rate`)")?;
            let indiv = needed_indiv_stmts.clone();
            Box::new(
                move |theta: &[f64], eta: &[f64], covariates: &HashMap<String, f64>| {
                    let vars = eval_indiv_param_vars(&indiv, theta, eta, covariates);
                    let lambda = eval_expression(&scale, theta, eta, covariates, &vars, &[]);
                    let lhr = loghr_expr.as_ref().map_or(0.0, |e| {
                        eval_expression(e, theta, eta, covariates, &vars, &[])
                    });
                    vec![lambda, lhr]
                },
            )
        }
        HazardFamily::Weibull => {
            let scale = scale_expr.ok_or("[event_model] family=weibull requires `scale`")?;
            let shape = shape_expr.ok_or("[event_model] family=weibull requires `shape`")?;
            let indiv = needed_indiv_stmts.clone();
            Box::new(
                move |theta: &[f64], eta: &[f64], covariates: &HashMap<String, f64>| {
                    let vars = eval_indiv_param_vars(&indiv, theta, eta, covariates);
                    let s = eval_expression(&scale, theta, eta, covariates, &vars, &[]);
                    let p = eval_expression(&shape, theta, eta, covariates, &vars, &[]);
                    let lhr = loghr_expr.as_ref().map_or(0.0, |e| {
                        eval_expression(e, theta, eta, covariates, &vars, &[])
                    });
                    vec![s, p, lhr]
                },
            )
        }
        HazardFamily::Gompertz => {
            let alpha = alpha_expr.ok_or("[event_model] family=gompertz requires `alpha`")?;
            let gamma = gamma_expr.ok_or("[event_model] family=gompertz requires `gamma`")?;
            let indiv = needed_indiv_stmts.clone();
            Box::new(
                move |theta: &[f64], eta: &[f64], covariates: &HashMap<String, f64>| {
                    let vars = eval_indiv_param_vars(&indiv, theta, eta, covariates);
                    let a = eval_expression(&alpha, theta, eta, covariates, &vars, &[]);
                    let g = eval_expression(&gamma, theta, eta, covariates, &vars, &[]);
                    let lhr = loghr_expr.as_ref().map_or(0.0, |e| {
                        eval_expression(e, theta, eta, covariates, &vars, &[])
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
            recurrence,
            hazard_covariates: event_model_covariates.clone(),
        },
        event_model_covariates,
        event_model_thetas,
        event_model_etas,
    ))
}

/// Parse one `[binary_model]` block — a binary / Bernoulli (logistic) endpoint
/// (Phase 4 categorical, Track C — #760). Mirrors [`parse_event_model_block`] but
/// carries a single linear-predictor expression instead of a hazard family.
///
/// Returns `(cmt, EndpointLikelihood::Binary { .. }, covariates, thetas, etas)` ready
/// to insert into `CompiledModel::endpoints`.
///
/// Supported keys:
/// - `cmt`   — required; positive integer (data-file CMT column value)
/// - `logit` — required; the linear predictor `lp` (log-odds of `P(Y=1)`) as a
///   θ/η/covariate/`[individual_parameters]` expression. `TIME` resolves per observation
///   (the data term sets the model-time guard per record).
/// - `link`  — optional; `logit` (default). `probit`/`cloglog` are not yet supported.
#[cfg(feature = "survival")]
fn parse_binary_model_block(
    lines: &[String],
    theta_names: &[String],
    eta_names: &[String],
    indiv_stmts: &[Statement],
    kappa_names: &[String],
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
    use crate::types::{EndpointLikelihood, LinkFn};

    // Like `[event_model]`, the `logit` predictor may reference `[individual_parameters]`
    // names (resolved per subject from `needed_indiv_stmts` below); anything else falls back
    // to a leniently-read covariate. `kappa_names` rejects IOV references the per-subject
    // predictor cannot evaluate.
    let indiv_param_names = assigned_vars_in_order(indiv_stmts);
    let ctx = ParseCtx::new(theta_names, eta_names, &indiv_param_names);

    let mut cmt_opt: Option<usize> = None;
    let mut logit_expr: Option<Expression> = None;
    let mut link = LinkFn::Logit;

    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = trimmed.splitn(2, '=').map(|s| s.trim()).collect();
        if parts.len() != 2 {
            return Err(format!(
                "[binary_model]: invalid line `{trimmed}` — expected `key = value`"
            ));
        }
        let (key, value) = (parts[0], parts[1]);
        match key {
            "cmt" => {
                cmt_opt = Some(value.parse::<usize>().map_err(|_| {
                    format!("[binary_model]: invalid cmt `{value}` — expected a positive integer")
                })?);
            }
            "logit" => {
                logit_expr = Some(parse_scalar_expression(value, ctx)?);
            }
            "link" => {
                link = match value {
                    "logit" => LinkFn::Logit,
                    other => {
                        return Err(format!(
                            "[binary_model]: link `{other}` is not yet supported — only `logit` \
                             (the default) is available; probit/cloglog are planned."
                        ))
                    }
                };
            }
            other => {
                return Err(format!(
                    "[binary_model]: unknown key `{other}` — valid keys: cmt, logit, link"
                ));
            }
        }
    }

    let cmt = cmt_opt.ok_or("[binary_model]: missing required key `cmt`")?;
    let logit_expr = logit_expr.ok_or("[binary_model]: missing required key `logit`")?;

    // Same CMT can't be both Gaussian and binary (parallels the event-model guard).
    if let ErrorSpec::PerCmt(cmt_map) = error_spec {
        if cmt_map.contains_key(&cmt) {
            return Err(format!(
                "[binary_model]: CMT={cmt} is already declared as a Gaussian endpoint in \
                 [error_model] — the same CMT cannot be both Gaussian and binary"
            ));
        }
    }

    // The covariate union, `[individual_parameters]` restriction, and the direct/via-indiv
    // IOV-κ (#770/#442) and `[covariate_nn]` (#293) rejects are shared with `[event_model]`
    // through `restrict_and_validate_indiv_stmts`; only the θ/η index sets (for the
    // dead-parameter census) are collected here, since each block returns its own.
    let lp_exprs: [&Expression; 1] = [&logit_expr];
    let (lp_thetas, lp_etas) = {
        let mut theta_set = std::collections::HashSet::new();
        let mut eta_set = std::collections::HashSet::new();
        collect_theta_eta(&logit_expr, &mut theta_set, &mut eta_set);
        (theta_set, eta_set)
    };
    let (needed_indiv_stmts, lp_covariates) = restrict_and_validate_indiv_stmts(
        &lp_exprs,
        indiv_stmts,
        eta_names,
        kappa_names,
        "binary_model",
    )?;

    // Build the lp_fn closure over (θ, η, covariates). Expression nodes hold only indices, so
    // they are safe to move; `TIME` resolves via the model-time guard `binary_data_term` sets
    // per record.
    let indiv = needed_indiv_stmts;
    let lp_fn: crate::types::LinearPredictorFn = Box::new(
        move |theta: &[f64], eta: &[f64], covariates: &HashMap<String, f64>| {
            let vars = eval_indiv_param_vars(&indiv, theta, eta, covariates);
            eval_expression(&logit_expr, theta, eta, covariates, &vars, &[])
        },
    );

    Ok((
        cmt,
        EndpointLikelihood::Binary {
            link,
            lp_fn,
            lp_covariates: lp_covariates.clone(),
        },
        lp_covariates,
        lp_thetas,
        lp_etas,
    ))
}

/// True if `key` is an Option-B matrix-rate key like `q12` (`q` + ≥2 digits).
/// Used only to steer the "not yet supported" error toward the Option-A form.
#[cfg(feature = "markov")]
fn is_matrix_rate_key(key: &str) -> bool {
    key.strip_prefix('q')
        .is_some_and(|rest| rest.len() >= 2 && rest.chars().all(|c| c.is_ascii_digit()))
}

/// Parse one `[markov_model]` block — a continuous-time Markov (CTMM) endpoint
/// (Phase 5, Track D — #759; §8.4 of `plans/tte-survival-markov.md`).
///
/// Returns `(cmt, EndpointLikelihood::Ctmm { .. }, generator_covariates, thetas, etas)`
/// ready to insert into `CompiledModel::endpoints`.
///
/// Supported lines (Option A — the ergonomic transition-list form):
/// - `type`   — optional; `ctmm` (default). `mctmm`/`dtmm` are not yet supported.
/// - `cmt`    — required; positive integer (data-file CMT column value).
/// - `states` — required; `[label=code, ...]` or bare `[code, ...]`. `code` is the
///   integer DV value that denotes the state; the optional `label` names it for the
///   transitions and output. States may be coded with any non-negative integers.
/// - `transition A -> B = <expr>` — one per allowed transition; `<expr>` is the
///   intensity `q_{A→B} ≥ 0` (θ/η/covariate/`[individual_parameters]` expression,
///   evaluated per subject at a baseline covariate snapshot). The generator's
///   diagonal `q_jj = −Σ_{k≠j} q_jk` is filled automatically.
///
/// Option B (the `q12 = …` matrix spelling), `init`, and `type = mctmm|dtmm` are
/// rejected with a "not yet supported" message so the surface stays honest.
#[cfg(feature = "markov")]
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
fn parse_markov_model_block(
    lines: &[String],
    theta_names: &[String],
    eta_names: &[String],
    indiv_stmts: &[Statement],
    kappa_names: &[String],
    error_spec: &ErrorSpec,
    // ODE state names in state-vector order (empty for a non-ODE / analytic model).
    // An intensity that references one of these is time-inhomogeneous (drug/PD-driven
    // Q(t), Phase 6 #817): the referenced state is threaded into the generator.
    ode_state_names: &[String],
    // Names declared in the optional `[covariates]` block (empty when absent). Used to
    // reject a name that is *both* an ODE state and a declared data covariate — that
    // collision would otherwise silently reinterpret the covariate column as the state.
    declared_covariates: &[String],
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
    use crate::types::EndpointLikelihood;
    use std::collections::{HashMap as StdHashMap, HashSet};

    // Like `[binary_model]`, transition intensities may reference
    // `[individual_parameters]` names (resolved per subject) plus θ/η/covariates.
    let indiv_param_names = assigned_vars_in_order(indiv_stmts);
    let ctx = ParseCtx::new(theta_names, eta_names, &indiv_param_names);

    let mut cmt_opt: Option<usize> = None;
    let mut states_seen = false;
    let mut states: Vec<(String, usize)> = Vec::new(); // ordered (label, DV code)
    let mut transitions_raw: Vec<(String, String, Expression)> = Vec::new();

    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // `transition A -> B = <intensity>`
        if let Some(rest) = trimmed.strip_prefix("transition") {
            if !rest.starts_with(char::is_whitespace) {
                return Err(format!(
                    "[markov_model]: unknown key in `{trimmed}` — a transition reads \
                     `transition A -> B = <intensity>`"
                ));
            }
            let (lhs, rhs) = rest.split_once('=').ok_or_else(|| {
                format!(
                    "[markov_model]: transition `{trimmed}` needs an intensity: \
                     `transition A -> B = <intensity>`"
                )
            })?;
            let (from, to) = lhs.split_once("->").ok_or_else(|| {
                format!("[markov_model]: transition `{trimmed}` needs `A -> B` before `=`")
            })?;
            let (from, to) = (from.trim().to_string(), to.trim().to_string());
            if from.is_empty() || to.is_empty() {
                return Err(format!(
                    "[markov_model]: transition `{trimmed}` has an empty state name"
                ));
            }
            let expr = parse_scalar_expression(rhs.trim(), ctx)?;
            transitions_raw.push((from, to, expr));
            continue;
        }

        // `key = value`
        let (key, value) = trimmed
            .split_once('=')
            .map(|(k, v)| (k.trim(), v.trim()))
            .ok_or_else(|| {
                format!(
                    "[markov_model]: invalid line `{trimmed}` — expected `key = value` or \
                     `transition A -> B = <intensity>`"
                )
            })?;
        match key {
            // `type` is optional and defaults to `ctmm` — the only endpoint family
            // implemented today. The default is explicit: a block with no `type` line
            // parses as a CTMM (see the required-key checks below, which do not include
            // `type`). When mCTMM/DTMM land they must NOT silently inherit this default —
            // dispatch on the value read here rather than widening the fall-through.
            "type" => match value {
                "ctmm" => {}
                "mctmm" | "dtmm" => {
                    return Err(format!(
                        "[markov_model]: type `{value}` is not yet supported — only `ctmm` \
                         (time-homogeneous CTMM) is available; mCTMM/DTMM are planned \
                         (Phase 4b/4c)."
                    ))
                }
                other => {
                    return Err(format!(
                        "[markov_model]: unknown type `{other}` — expected `ctmm`"
                    ))
                }
            },
            "cmt" => {
                let c = value.parse::<usize>().map_err(|_| {
                    format!("[markov_model]: invalid cmt `{value}` — expected a positive integer")
                })?;
                // CMT is 1-based everywhere in ferx (CMT=0 has special meaning), so reject
                // 0 here rather than silently route data to compartment index 0.
                if c == 0 {
                    return Err(
                        "[markov_model]: cmt = 0 is invalid — compartment indices are 1-based"
                            .to_string(),
                    );
                }
                cmt_opt = Some(c);
            }
            "states" => {
                states_seen = true;
                let inner = value
                    .strip_prefix('[')
                    .and_then(|s| s.strip_suffix(']'))
                    .ok_or_else(|| {
                        format!(
                            "[markov_model]: states `{value}` must be a bracketed list, e.g. \
                             [awake=0, asleep=1] or [0, 1]"
                        )
                    })?;
                for tok in inner.split(',') {
                    let tok = tok.trim();
                    if tok.is_empty() {
                        continue;
                    }
                    // `label=code` or bare `code` (label defaults to the code text).
                    let (label, code_str) = match tok.split_once('=') {
                        Some((l, c)) => (l.trim().to_string(), c.trim()),
                        None => (tok.to_string(), tok),
                    };
                    // Reject an empty label (`states = [=0, ...]`): a transition can never
                    // reference it (transition state names must be non-empty), so it would
                    // be an unreferenceable state. Point the user at the bare-numeric form.
                    if label.is_empty() {
                        return Err(format!(
                            "[markov_model]: state code `{code_str}` has an empty label — use \
                             `label={code_str}` or the bare numeric form (e.g. `states = [0, 1]`)"
                        ));
                    }
                    let code = code_str.parse::<usize>().map_err(|_| {
                        format!(
                            "[markov_model]: state code `{code_str}` must be a non-negative \
                             integer (the DV value that denotes the state)"
                        )
                    })?;
                    states.push((label, code));
                }
            }
            "init" => {
                return Err(
                    "[markov_model]: `init` (starting occupancy) is not yet supported — the CTMM \
                     fit conditions on each subject's first observed state; `init` will drive \
                     prediction/simulation in a later slice."
                        .to_string(),
                )
            }
            other => {
                if other == "n_states" || is_matrix_rate_key(other) {
                    return Err(format!(
                        "[markov_model]: the matrix spelling (`{other} = …`, Option B) is not yet \
                         supported — use the transition list (Option A): `states = [...]` plus \
                         `transition A -> B = <intensity>` lines."
                    ));
                }
                return Err(format!(
                    "[markov_model]: unknown key `{other}` — valid keys: type, cmt, states, and \
                     `transition A -> B = <intensity>` lines"
                ));
            }
        }
    }

    let cmt = cmt_opt.ok_or("[markov_model]: missing required key `cmt`")?;
    if !states_seen {
        return Err("[markov_model]: missing required key `states`".to_string());
    }
    if states.len() < 2 {
        return Err(format!(
            "[markov_model]: a CTMM needs at least 2 states; got {}",
            states.len()
        ));
    }
    if transitions_raw.is_empty() {
        return Err(
            "[markov_model]: no `transition` lines — declare at least one \
             `transition A -> B = <intensity>`"
                .to_string(),
        );
    }

    // Same CMT can't be both Gaussian and CTMM (parallels the binary/event guard).
    if let ErrorSpec::PerCmt(cmt_map) = error_spec {
        if cmt_map.contains_key(&cmt) {
            return Err(format!(
                "[markov_model]: CMT={cmt} is already declared as a Gaussian endpoint in \
                 [error_model] — the same CMT cannot be both Gaussian and CTMM"
            ));
        }
    }

    // Build the label→index map and the ordered DV-code table, rejecting duplicate
    // labels and duplicate codes (a duplicate code would make the DV→index mapping
    // ambiguous).
    let mut label_to_idx: StdHashMap<String, usize> = StdHashMap::new();
    let mut code_owner: StdHashMap<usize, String> = StdHashMap::new();
    let mut state_codes: Vec<usize> = Vec::with_capacity(states.len());
    for (i, (label, code)) in states.iter().enumerate() {
        if label_to_idx.insert(label.clone(), i).is_some() {
            return Err(format!("[markov_model]: duplicate state label `{label}`"));
        }
        if let Some(prev) = code_owner.insert(*code, label.clone()) {
            return Err(format!(
                "[markov_model]: duplicate state code {code} (states `{prev}` and `{label}`)"
            ));
        }
        state_codes.push(*code);
    }

    // Resolve each transition's labels to generator indices, rejecting unknown
    // states, self-transitions (a rate is off-diagonal), and duplicate pairs.
    let mut pair_seen: HashSet<(usize, usize)> = HashSet::new();
    let mut transitions: Vec<(usize, usize, Expression)> =
        Vec::with_capacity(transitions_raw.len());
    for (from, to, expr) in transitions_raw {
        let j = *label_to_idx.get(&from).ok_or_else(|| {
            format!("[markov_model]: transition references unknown state `{from}`")
        })?;
        let k = *label_to_idx
            .get(&to)
            .ok_or_else(|| format!("[markov_model]: transition references unknown state `{to}`"))?;
        if j == k {
            return Err(format!(
                "[markov_model]: self-transition `{from} -> {to}` is not allowed — an intensity is \
                 an off-diagonal generator entry"
            ));
        }
        if !pair_seen.insert((j, k)) {
            return Err(format!(
                "[markov_model]: transition `{from} -> {to}` is declared more than once"
            ));
        }
        transitions.push((j, k, expr));
    }

    // Collect covariate/theta/eta references across every intensity BEFORE the
    // expressions move into the generator closure.
    let mut cov_set: HashSet<String> = HashSet::new();
    let mut theta_set: HashSet<usize> = HashSet::new();
    let mut eta_set: HashSet<usize> = HashSet::new();
    for (_, _, expr) in &transitions {
        collect_covariates(expr, &mut cov_set);
        collect_theta_eta(expr, &mut theta_set, &mut eta_set);
    }

    // Reject a direct IOV-kappa reference: intensities are evaluated with BSV-only η,
    // so a kappa would silently read 0.0 (#770 analogue).
    for (_, _, expr) in &transitions {
        if let Some(k) = expr_references_kappa(expr, kappa_names) {
            return Err(format!(
                "[markov_model]: a transition intensity references the inter-occasion (IOV) random \
                 effect `{k}`. Intensities are evaluated with per-subject (BSV) η only — write the \
                 intensity in terms of θ/η, or reference an IOV-free parameter."
            ));
        }
    }

    // Restrict the `[individual_parameters]` statements to those any intensity
    // references, transitively (mirrors the binary/hazard path).
    let needed_indiv_stmts: Vec<Statement> = {
        let mut needed: HashSet<String> = HashSet::new();
        for (_, _, expr) in &transitions {
            collect_variable_names(expr, &mut needed);
        }
        let mut keep: Vec<Statement> = Vec::new();
        for s in indiv_stmts.iter().rev() {
            let stmt = std::slice::from_ref(s);
            if assigned_vars_in_order(stmt)
                .iter()
                .any(|n| needed.contains(n))
            {
                visit_stmt_nodes(stmt, &mut |e: &Expression| {
                    if let Expression::Variable(name) = e {
                        needed.insert(name.clone());
                    }
                });
                keep.push(s.clone());
            }
        }
        keep.reverse();
        keep
    };

    // Union covariates reached transitively through `[individual_parameters]` so a
    // time-varying covariate can't slip past the TV-covariate guard silently frozen.
    {
        let mut c2: HashSet<String> = HashSet::new();
        collect_covariates_in_stmts(&needed_indiv_stmts, &mut c2);
        cov_set.extend(c2);
    }

    // Reject an `[individual_parameters]` value that depends on an IOV kappa (BSV-only η here).
    {
        let n_eta = eta_names.len();
        let mut kappa_hit: Option<String> = None;
        visit_stmt_nodes(&needed_indiv_stmts, &mut |e: &Expression| {
            if let Expression::Eta(i) = e {
                if *i >= n_eta && kappa_hit.is_none() {
                    kappa_hit = Some(
                        kappa_names
                            .get(*i - n_eta)
                            .cloned()
                            .unwrap_or_else(|| "<kappa>".to_string()),
                    );
                }
            }
        });
        if let Some(k) = kappa_hit {
            return Err(format!(
                "[markov_model]: a transition intensity references an [individual_parameters] \
                 value that depends on the inter-occasion (IOV) random effect `{k}`. Intensities \
                 are evaluated with per-subject (BSV) η only — reference an IOV-free parameter."
            ));
        }
    }

    // An `[individual_parameters]` value reading a `[covariate_nn]` output would
    // silently resolve to 0.0 here (no network forward pass). Reject it — gated to
    // `nn` like the binary/hazard guard.
    #[cfg(feature = "nn")]
    {
        let mut nn_hit = false;
        visit_stmt_nodes(&needed_indiv_stmts, &mut |e: &Expression| {
            if matches!(e, Expression::NnOutput { .. }) {
                nn_hit = true;
            }
        });
        if nn_hit {
            return Err(
                "[markov_model]: a transition intensity references an [individual_parameters] value \
                 whose definition uses a [covariate_nn] output. Neural-network-driven individual \
                 parameters are not available to CTMM intensities — reference an NN-free parameter."
                    .to_string(),
            );
        }
    }

    // Reject a literal `TIME` reference in an intensity (or in an
    // `[individual_parameters]` value an intensity reaches). The CTMM is
    // time-homogeneous: the generator is built once per subject, *outside* any
    // model-time scope (unlike the binary logit, which is wrapped per record), so
    // `Expression::Time` would resolve to the thread-local default 0.0 and the fit
    // would silently drop the time term. The TV-covariate guard does not catch this —
    // `TIME` is not a covariate. A drug/time-driven Q(t) is Phase 6.
    {
        let mut time_hit = false;
        {
            let mut mark = |e: &Expression| {
                if matches!(e, Expression::Time) {
                    time_hit = true;
                }
            };
            for (_, _, expr) in &transitions {
                visit_expr_nodes(expr, &mut mark);
            }
            visit_stmt_nodes(&needed_indiv_stmts, &mut mark);
        }
        if time_hit {
            return Err(
                "[markov_model]: a transition intensity references TIME. The CTMM is \
                 time-homogeneous — the generator is built once per subject and evaluated \
                 outside any model-time scope, so TIME would silently resolve to 0. Write the \
                 intensity in terms of θ/η/covariates only; a time-varying (drug-driven) \
                 generator Q(t) is Phase 6."
                    .to_string(),
            );
        }
    }

    // ── Time-inhomogeneous (drug/PD-driven Q(t), Phase 6 #817) detection ──────────
    // An intensity may reference a model **ODE state** by name (e.g. a concentration
    // `central/V` or a PD response compartment), evaluated at the current time as the
    // state evolves. An identifier that is neither θ/η nor an `[individual_parameters]`
    // name parses as an `Expression::Covariate` leaf (resolved from the covariates map at
    // eval); a state name lands there too. Collect every Variable **and** Covariate leaf
    // used directly in the transitions, and keep the ones that name an ODE state (and are
    // not shadowed by an individual-parameter name — those resolve to the parameter).
    // `generator_states` pairs each with its ODE-state-vector index; non-empty ⇒ the
    // endpoint is inhomogeneous.
    let generator_states: Vec<(String, usize)> = {
        let mut referenced: HashSet<String> = HashSet::new();
        for (_, _, expr) in &transitions {
            visit_expr_nodes(expr, &mut |e: &Expression| {
                if let Expression::Variable(name) | Expression::Covariate(name) = e {
                    referenced.insert(name.clone());
                }
            });
        }
        let indiv_set: HashSet<&String> = indiv_param_names.iter().collect();
        ode_state_names
            .iter()
            .enumerate()
            .filter(|(_, name)| referenced.contains(*name) && !indiv_set.contains(name))
            .map(|(idx, name)| (name.clone(), idx))
            .collect()
    };

    // Reject a name that is simultaneously an ODE state and a referenced θ/η parameter:
    // such an identifier resolves to the *parameter* leaf (`Expression::Theta/Eta`,
    // indices assigned at parse), so it is invisible to the state-reference scan above
    // and the endpoint would silently score on the time-**homogeneous** path with the
    // constant parameter value instead of the evolving model state the user intended as
    // the driver. The covariate-collision guard below catches the [covariates] case; this
    // catches the parameter case symmetrically. `Theta/Eta` leaves carry only an index,
    // so map it back to a name via `theta_names`/`eta_names`.
    if !ode_state_names.is_empty() {
        let state_name_set: HashSet<&String> = ode_state_names.iter().collect();
        let mut collide: Option<String> = None;
        for (_, _, expr) in &transitions {
            visit_expr_nodes(expr, &mut |e: &Expression| {
                let nm = match e {
                    Expression::Theta(i) => theta_names.get(*i),
                    Expression::Eta(i) => eta_names.get(*i),
                    _ => None,
                };
                if let Some(name) = nm {
                    if state_name_set.contains(name) {
                        collide = Some(name.clone());
                    }
                }
            });
        }
        if let Some(name) = collide {
            return Err(format!(
                "[markov_model]: the transition-intensity identifier `{name}` names both an ODE \
                 model state and a θ/η parameter. This is ambiguous — the fit would use the \
                 constant parameter and ignore the (time-inhomogeneous) model state. Rename one \
                 of them so the intended driver is unambiguous."
            ));
        }
    }

    // Reject a name that is simultaneously an ODE state and a declared data covariate:
    // the intensity would resolve it to the model state (below), silently overwriting the
    // covariate column and stripping it from the required-column / TV-covariate checks —
    // a wrong driver with no error. Fail loud and ask the user to disambiguate.
    if !declared_covariates.is_empty() {
        let cov_set_decl: HashSet<&String> = declared_covariates.iter().collect();
        if let Some((name, _)) = generator_states
            .iter()
            .find(|(n, _)| cov_set_decl.contains(n))
        {
            return Err(format!(
                "[markov_model]: the transition-intensity identifier `{name}` names both an ODE \
                 model state and a declared [covariates] column. This is ambiguous — the fit \
                 would use the model state and ignore the data column. Rename one of them so the \
                 intended driver is unambiguous."
            ));
        }
    }

    // A referenced ODE state parsed as a `Covariate` leaf, so it was pooled into `cov_set`
    // by the covariate scan above. It is a model **state**, not a data covariate — drop it
    // so it is neither registered as a required data column nor caught by the TV-covariate
    // guard (its time variation is intrinsic and handled by the occupancy integration).
    let state_name_set: HashSet<&String> = generator_states.iter().map(|(n, _)| n).collect();
    cov_set.retain(|c| !state_name_set.contains(c));
    let mut generator_covariates: Vec<String> = cov_set.into_iter().collect();
    generator_covariates.sort();

    // A state reached only *transitively* through an `[individual_parameters]` value is
    // not supported yet: `eval_indiv_param_vars` evaluates those with no state channel,
    // so the state would silently resolve to 0. Reject fail-loud, pointing the user to
    // reference the state directly in the transition intensity.
    if !ode_state_names.is_empty() {
        let mut indiv_refs: HashSet<String> = HashSet::new();
        collect_covariates_in_stmts(&needed_indiv_stmts, &mut indiv_refs); // covariate leaves
        for s in &needed_indiv_stmts {
            visit_stmt_nodes(std::slice::from_ref(s), &mut |e: &Expression| {
                if let Expression::Variable(name) = e {
                    indiv_refs.insert(name.clone());
                }
            });
        }
        let indiv_set: HashSet<&String> = indiv_param_names.iter().collect();
        if let Some(name) = ode_state_names
            .iter()
            .find(|n| indiv_refs.contains(*n) && !indiv_set.contains(n))
        {
            return Err(format!(
                "[markov_model]: an [individual_parameters] value referenced by a transition \
                 intensity reads the model state `{name}`. A state-driven (time-inhomogeneous) \
                 intensity must reference the state directly in the `transition` line (e.g. \
                 `transition A -> B = exp(LQ + SLOPE * ({name} / V))`), not through an \
                 [individual_parameters] value."
            ));
        }
    }

    // Build the generator closure over (θ, η, covariates, state): place each intensity
    // at its off-diagonal entry and subtract it from the diagonal so every row sums to
    // zero — the caller always receives a valid generator. On the inhomogeneous path the
    // caller passes the ODE `state` at the current time; each referenced state name is
    // injected into the variable map so `eval_expression` resolves it. On the homogeneous
    // path `generator_states` is empty and `state` is ignored.
    let n_states = states.len();

    // Snapshot a resolved, **dual-evaluable** form of the intensities before they move into
    // the f64 closure below — the CTMM analogue of `IndivParamProgram` (#759). This is what
    // lets `markov::endpoint` produce an exact `∂Q/∂(θ,η)` (and, chained through the Van Loan
    // Fréchet derivative of `expm`, an exact likelihood gradient) instead of finite-
    // differencing the whole matrix-exponential likelihood.
    //
    // Built only for the **time-homogeneous** endpoint: an intensity that reads an ODE state
    // makes `Q` a functional of the PK trajectory, whose derivative needs the state's own
    // sensitivities (and whose likelihood is an occupancy ODE, not an `expm` — so Van Loan
    // does not apply). Those keep the FD path, so the program is `None` there.
    let generator_program = if generator_states.is_empty() {
        build_ctmm_generator_program(
            &transitions,
            &needed_indiv_stmts,
            &generator_covariates,
            n_states,
            theta_names.len(),
            eta_names.len(),
        )
    } else {
        None
    };

    let indiv = needed_indiv_stmts;
    let gen_states = generator_states.clone();
    let generator_fn: crate::types::GeneratorFn = Box::new(
        move |theta: &[f64], eta: &[f64], covariates: &HashMap<String, f64>, state: &[f64]| {
            // A referenced ODE state parses as a `Covariate` leaf, so `eval_expression`
            // resolves it from the covariates map. On the inhomogeneous path overlay the
            // current state values there; the homogeneous path leaves `gen_states` empty and
            // uses the caller's map unchanged (no clone).
            let cov_owned;
            let cov: &HashMap<String, f64> = if gen_states.is_empty() {
                covariates
            } else {
                let mut c = covariates.clone();
                for (name, idx) in &gen_states {
                    c.insert(name.clone(), state.get(*idx).copied().unwrap_or(0.0));
                }
                cov_owned = c;
                &cov_owned
            };
            let vars = eval_indiv_param_vars(&indiv, theta, eta, cov);
            let mut q = nalgebra::DMatrix::<f64>::zeros(n_states, n_states);
            for (j, k, expr) in &transitions {
                let rate = eval_expression(expr, theta, eta, cov, &vars, &[]);
                q[(*j, *k)] += rate;
                q[(*j, *j)] -= rate;
            }
            q
        },
    );

    Ok((
        cmt,
        EndpointLikelihood::Ctmm {
            n_states,
            state_codes,
            generator_fn,
            generator_program,
            generator_covariates: generator_covariates.clone(),
            generator_states,
        },
        generator_covariates,
        theta_set,
        eta_set,
    ))
}

/// Pre-scan `[event_model]` blocks for ODE-accumulated hazards (joint PK-TTE, #564).
///
/// Returns `(cmt, hazard_expr_text)` for every block that declares `hazard = <expr>`.
/// Runs *before* the ODE spec is built so each hazard can be appended to the ODE
/// system as a cumulative-hazard accumulator state (`d/dt(__chz_<cmt>) = <expr>`),
/// which then compiles in the `[odes]` RHS namespace. Only the `cmt` and `hazard`
/// keys are read here; full key/family validation and the `OdeAccumulated` endpoint
/// are produced later by [`parse_event_model_block`] (malformed lines are reported
/// there, so they are skipped rather than errored on in this lightweight pass).
#[cfg(feature = "survival")]
fn prescan_ode_hazards(extracted: &ExtractedBlocks) -> Result<Vec<(usize, String)>, String> {
    let mut event_blocks: Vec<&Vec<String>> = Vec::new();
    if let Some(lines) = extracted.unnamed.get("event_model") {
        event_blocks.push(lines);
    }
    if let Some(named_map) = extracted.named.get("event_model") {
        for lines in named_map.values() {
            event_blocks.push(lines);
        }
    }

    let mut out: Vec<(usize, String)> = Vec::new();
    for lines in event_blocks {
        let mut cmt: Option<usize> = None;
        let mut hazard: Option<String> = None;
        for line in lines {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let parts: Vec<&str> = trimmed.splitn(2, '=').map(|s| s.trim()).collect();
            if parts.len() != 2 {
                continue; // malformed `key = value` lines are reported downstream
            }
            match parts[0] {
                "cmt" => cmt = parts[1].parse::<usize>().ok(),
                "hazard" => hazard = Some(parts[1].to_string()),
                _ => {}
            }
        }
        if let Some(expr) = hazard {
            let cmt = cmt
                .ok_or("[event_model]: `hazard` requires `cmt` (the TTE endpoint's compartment)")?;
            out.push((cmt, expr));
        }
    }
    Ok(out)
}

// ── [simulation] block parser ───────────────────────────────────────────────

fn parse_simulation_block(lines: &[String]) -> Result<SimulationSpec, String> {
    let mut n_subjects = 10;
    let mut dose_amt = 100.0;
    let mut dose_cmt = 1;
    let mut obs_times = Vec::new();
    let mut seed = 42u64;
    let mut horizon: Option<f64> = None;

    for line in lines {
        let parts: Vec<&str> = line.splitn(2, '=').map(|s| s.trim()).collect();
        if parts.len() != 2 {
            // No `=` on a non-blank, non-comment line (comments/blanks are already
            // stripped by `extract_blocks`). Treat it as a hard error rather than
            // silently skipping — e.g. `n_subjects 5` must not fall back to the
            // default, which is the whole point of this block's strict parsing.
            return Err(format!(
                "[simulation]: malformed line (expected `key = value`): {}",
                line.trim()
            ));
        }
        match parts[0] {
            // `n_subjects` / `dose_amt` / `dose_cmt` are the canonical spellings
            // (they match the `SimulationSpec` fields and every `examples/*.ferx`);
            // the short `subjects` / `dose` / `cmt` forms are kept as back-compat
            // aliases. An unknown key is a hard error (mirrors [fit_options]) so a
            // typo like `n_subject` no longer silently falls back to the default.
            "subjects" | "n_subjects" => {
                n_subjects = parts[1]
                    .parse()
                    .map_err(|_| format!("[simulation]: bad {}: {}", parts[0], line))?
            }
            "dose" | "dose_amt" => {
                dose_amt = parts[1]
                    .parse()
                    .map_err(|_| format!("[simulation]: bad {}: {}", parts[0], line))?
            }
            "cmt" | "dose_cmt" => {
                dose_cmt = parts[1]
                    .parse()
                    .map_err(|_| format!("[simulation]: bad {}: {}", parts[0], line))?
            }
            "seed" => {
                seed = parts[1]
                    .parse()
                    .map_err(|_| format!("[simulation]: bad {}: {}", parts[0], line))?
            }
            "times" => {
                obs_times = parse_float_array(parts[1])
                    .map_err(|e| format!("[simulation]: bad times: {e}"))?
            }
            // Administrative censoring horizon for TTE endpoints (#522). Must be a
            // finite, strictly-positive time: it is the right-censoring window for
            // every simulated TTE cause and a non-positive / non-finite horizon
            // would censor every subject at or before entry (no events possible).
            "horizon" => {
                let t: f64 = parts[1]
                    .parse()
                    .map_err(|_| format!("[simulation]: bad horizon: {}", line))?;
                if !t.is_finite() || t <= 0.0 {
                    return Err(format!(
                        "[simulation]: horizon must be finite and > 0: {}",
                        line
                    ));
                }
                horizon = Some(t);
            }
            other => return Err(format!("[simulation]: unknown key `{}`", other)),
        }
    }
    // A synthetic design needs *something* to observe: continuous `times` for a
    // Gaussian model, or a `horizon` for a TTE model (#522). Require at least one;
    // the model-vs-spec check (e.g. "TTE endpoint needs a horizon") happens once
    // the endpoints are known, in `api::run_model_simulate`.
    if obs_times.is_empty() && horizon.is_none() {
        return Err(
            "[simulation] block requires 'times = [...]' (Gaussian) or 'horizon = <t>' (TTE)"
                .to_string(),
        );
    }

    Ok(SimulationSpec {
        n_subjects,
        dose_amt,
        dose_cmt,
        obs_times,
        seed,
        horizon,
        covariates: vec![],
    })
}

// ── [adaptive_dosing] block parser (#391 S2.1) ──────────────────────────────

/// Parse a declarative `[adaptive_dosing]` block into an [`AdaptiveDosingSpec`]
/// (#391 S2).
///
/// Pure syntax: every malformed, unknown, or ambiguous construct is a typed error
/// with no silent default (mirroring the strict `[simulation]` / `[fit_options]`
/// parsers). No controller is built and the `observe` expression is **not**
/// compiled here — that, and resolving which endpoint a bare `observe` maps to,
/// is the S2.2 step. The one ambiguity this parser *can* settle without the model
/// — `with_assay_error` on an expression `observe` with no designated endpoint —
/// is rejected here rather than left to guess a σ downstream.
fn parse_adaptive_dosing_block(
    lines: &[String],
    declared_covariates: &[String],
) -> Result<AdaptiveDosingSpec, String> {
    let mut observe: Option<String> = None;
    let mut with_assay_error = false;
    let mut assay_cmt: Option<usize> = None;
    let mut at: Option<Vec<f64>> = None;
    let mut start_dose: Option<f64> = None;
    let mut route: Option<AdaptiveRoute> = None;
    let mut dose_bounds: Option<(f64, f64)> = None;
    let mut confirm: u32 = 1;
    let mut levels: Option<Vec<f64>> = None;
    let mut target_window: Option<(f64, f64)> = None;
    let mut auc_target: Option<(f64, f64)> = None;
    let mut rules: Vec<AdaptiveRule> = Vec::new();

    for line in lines {
        let line = line.trim();
        // Rule lines (`when …`) carry no top-level `=`; everything else is `key = value`.
        if let Some(rest) = line.strip_prefix("when ") {
            rules.push(parse_adaptive_rule(rest)?);
            continue;
        }
        let parts: Vec<&str> = line.splitn(2, '=').map(|s| s.trim()).collect();
        if parts.len() != 2 {
            return Err(format!(
                "[adaptive_dosing]: malformed line (expected `key = value` or `when …`): {line}"
            ));
        }
        let (key, val) = (parts[0], parts[1]);
        match key {
            "observe" => {
                if val.is_empty() {
                    return Err("[adaptive_dosing]: `observe` requires an expression".to_string());
                }
                observe = Some(val.to_string());
            }
            "with_assay_error" => {
                with_assay_error = match val {
                    "true" => true,
                    "false" => false,
                    other => {
                        return Err(format!(
                            "[adaptive_dosing]: with_assay_error must be `true` or `false`, got `{other}`"
                        ))
                    }
                };
            }
            "assay_cmt" => {
                let c: usize = val.parse().map_err(|_| {
                    format!("[adaptive_dosing]: bad assay_cmt (expected a positive integer): {val}")
                })?;
                if c == 0 {
                    return Err(
                        "[adaptive_dosing]: assay_cmt is 1-based and must be >= 1".to_string()
                    );
                }
                assay_cmt = Some(c);
            }
            "at" => at = Some(parse_decision_schedule(val)?),
            "start_dose" => {
                let d: f64 = val
                    .parse()
                    .map_err(|_| format!("[adaptive_dosing]: bad start_dose: {val}"))?;
                if !d.is_finite() || d < 0.0 {
                    return Err(format!(
                        "[adaptive_dosing]: start_dose must be finite and >= 0: {val}"
                    ));
                }
                start_dose = Some(d);
            }
            "route" => route = Some(parse_adaptive_route(val)?),
            "dose_bounds" => {
                let b = parse_float_array(val)
                    .map_err(|e| format!("[adaptive_dosing]: bad dose_bounds: {e}"))?;
                if b.len() != 2 {
                    return Err(format!(
                        "[adaptive_dosing]: dose_bounds must be `[low, high]`: {val}"
                    ));
                }
                let (lo, hi) = (b[0], b[1]);
                if !lo.is_finite() || !hi.is_finite() || lo < 0.0 || hi < lo {
                    return Err(format!(
                        "[adaptive_dosing]: dose_bounds must be finite with 0 <= low <= high: {val}"
                    ));
                }
                dose_bounds = Some((lo, hi));
            }
            "confirm" => {
                let n: u32 = val.parse().map_err(|_| {
                    format!("[adaptive_dosing]: bad confirm (expected a positive integer): {val}")
                })?;
                if n == 0 {
                    return Err(
                        "[adaptive_dosing]: confirm must be >= 1 (1 acts on the first breach)"
                            .to_string(),
                    );
                }
                confirm = n;
            }
            "levels" => {
                let l = parse_float_array(val)
                    .map_err(|e| format!("[adaptive_dosing]: bad levels: {e}"))?;
                validate_increasing_finite(&l, "levels")?;
                levels = Some(l);
            }
            "target_window" => {
                // Therapeutic band `[low, high]` for the `pct_time_in_window` metric
                // only — it does not affect dosing. `low` must be finite; `high` may
                // be `inf` for a one-sided "at or above low" target. The same check
                // runs in `AdaptiveDosingSpec::validate`.
                let b = parse_float_array(val)
                    .map_err(|e| format!("[adaptive_dosing]: bad target_window: {e}"))?;
                if b.len() != 2 {
                    return Err(format!(
                        "[adaptive_dosing]: target_window must be `[low, high]`: {val}"
                    ));
                }
                let (lo, hi) = (b[0], b[1]);
                if !lo.is_finite() || hi.is_nan() || hi < lo {
                    return Err(format!(
                        "[adaptive_dosing]: target_window must be `[low, high]` with low finite \
                         and low <= high (high may be `inf`): {val}"
                    ));
                }
                target_window = Some((lo, hi));
            }
            "auc_target" => {
                // Exposure band `[low, high]` for the `auc_target_attainment` metric
                // only — it does not affect dosing. It is the AUC of a non-negative
                // signal, so `low` must be finite *and* `>= 0`; `high` may be `inf`
                // for a one-sided "at or above low" target. Mirrors
                // `AdaptiveDosingSpec::validate`.
                let b = parse_float_array(val)
                    .map_err(|e| format!("[adaptive_dosing]: bad auc_target: {e}"))?;
                if b.len() != 2 {
                    return Err(format!(
                        "[adaptive_dosing]: auc_target must be `[low, high]`: {val}"
                    ));
                }
                let (lo, hi) = (b[0], b[1]);
                if !lo.is_finite() || lo < 0.0 || hi.is_nan() || hi < lo {
                    return Err(format!(
                        "[adaptive_dosing]: auc_target must be `[low, high]` with low finite, \
                         0 <= low <= high (high may be `inf`): {val}"
                    ));
                }
                auc_target = Some((lo, hi));
            }
            other => return Err(format!("[adaptive_dosing]: unknown key `{other}`")),
        }
    }

    // ── Required keys ──
    // `observe` is *conditionally* required (a latent signal needs it; an
    // assay-noised signal uses the model output instead) — `validate()` enforces
    // the `observe` xor `with_assay_error` rule, so it stays an `Option` here.
    let at = at.ok_or("[adaptive_dosing]: missing required `at` (decision schedule)")?;
    let start_dose = start_dose.ok_or("[adaptive_dosing]: missing required `start_dose`")?;
    let route = route.ok_or("[adaptive_dosing]: missing required `route`")?;
    let dose_bounds = dose_bounds.ok_or("[adaptive_dosing]: missing required `dose_bounds`")?;
    // Assemble the spec, then enforce every cross-field invariant in one place
    // (`AdaptiveDosingSpec::validate`). The same validator runs again in
    // `compile_adaptive`, so a programmatically-built spec can never reach the
    // controller with a contradiction the parser would have rejected here.
    let spec = AdaptiveDosingSpec {
        observe,
        observe_declared_covariates: declared_covariates.to_vec(),
        with_assay_error,
        assay_cmt,
        at,
        start_dose,
        route,
        dose_bounds,
        confirm,
        levels,
        target_window,
        auc_target,
        rules,
    };
    spec.validate()?;
    Ok(spec)
}

/// Parse an `at =` decision schedule: either an explicit list `[t1, t2, …]` or an
/// arithmetic sequence `every <Δ> from <t0> to <t1>` (inclusive), expanded to
/// explicit times. The `to` bound is required (an open-ended schedule has no
/// finite decision list); times must be finite and strictly increasing.
fn parse_decision_schedule(val: &str) -> Result<Vec<f64>, String> {
    let v = val.trim();
    if v.starts_with('[') {
        if !v.ends_with(']') {
            return Err(format!(
                "[adaptive_dosing]: `at` list is missing its closing `]`: {val}"
            ));
        }
        let times = parse_float_array(v).map_err(|e| format!("[adaptive_dosing]: bad at: {e}"))?;
        validate_increasing_finite(&times, "`at` times")?;
        // Strictly increasing ⇒ the first element is the minimum; reject a
        // negative decision time (start_dose / dose_bounds reject < 0 too).
        if times[0] < 0.0 {
            return Err(format!("[adaptive_dosing]: `at` times must be >= 0: {val}"));
        }
        return Ok(times);
    }
    let toks: Vec<&str> = v.split_whitespace().collect();
    if toks.len() != 6 || toks[0] != "every" || toks[2] != "from" || toks[4] != "to" {
        return Err(format!(
            "[adaptive_dosing]: `at` must be a list `[t1, t2, …]` or \
             `every <Δ> from <t0> to <t1>`: {val}"
        ));
    }
    let step: f64 = toks[1]
        .parse()
        .map_err(|_| format!("[adaptive_dosing]: bad `at` interval: {}", toks[1]))?;
    let t0: f64 = toks[3]
        .parse()
        .map_err(|_| format!("[adaptive_dosing]: bad `at` start: {}", toks[3]))?;
    let t1: f64 = toks[5]
        .parse()
        .map_err(|_| format!("[adaptive_dosing]: bad `at` end: {}", toks[5]))?;
    if !step.is_finite() || step <= 0.0 {
        return Err(format!(
            "[adaptive_dosing]: `at` interval must be finite and > 0: {}",
            toks[1]
        ));
    }
    if !t0.is_finite() || !t1.is_finite() || t1 < t0 || t0 < 0.0 {
        return Err(format!(
            "[adaptive_dosing]: `at` range must be finite with 0 <= start <= end: {val}"
        ));
    }
    // Relative tolerance so float division error (e.g. 1.0 / 0.1 = 9.999…8) does
    // not drop the inclusive final time when `t1` is an exact multiple of `step`.
    let ratio = (t1 - t0) / step;
    let n_f = (ratio + 1e-9 * ratio.max(1.0)).floor();
    if n_f > 100_000.0 {
        // Hard parse-time expansion bound (avoids allocating a runaway `Vec` here);
        // distinct from, and far above, the runtime `max_decisions` cap. Tighten the
        // range or step rather than relying on either as a policy knob.
        return Err(
            "[adaptive_dosing]: `at` schedule expands to too many decision times (> 100000); \
             tighten the range or step"
                .to_string(),
        );
    }
    let n = n_f as usize;
    // Index-based expansion (t0 + i·Δ) avoids accumulating floating-point drift.
    let times: Vec<f64> = (0..=n).map(|i| t0 + i as f64 * step).collect();
    Ok(times)
}

/// Parse a `route =` value: `bolus(cmt = N)` or `infuse(cmt = N, over = T)`.
fn parse_adaptive_route(val: &str) -> Result<AdaptiveRoute, String> {
    let v = val.trim();
    let (kind, args) = v
        .strip_suffix(')')
        .and_then(|s| s.split_once('('))
        .ok_or_else(|| {
            format!(
                "[adaptive_dosing]: route must be `bolus(cmt=N)` or `infuse(cmt=N, over=T)`: {val}"
            )
        })?;
    let mut cmt: Option<usize> = None;
    let mut over: Option<f64> = None;
    for arg in args.split(',') {
        let arg = arg.trim();
        if arg.is_empty() {
            continue;
        }
        let (k, x) = arg.split_once('=').ok_or_else(|| {
            format!("[adaptive_dosing]: bad route argument `{arg}` (expected key = value)")
        })?;
        match (k.trim(), x.trim()) {
            ("cmt", x) => {
                let c: usize = x
                    .parse()
                    .map_err(|_| format!("[adaptive_dosing]: bad route cmt: {x}"))?;
                if c == 0 {
                    return Err(
                        "[adaptive_dosing]: route cmt is 1-based and must be >= 1".to_string()
                    );
                }
                cmt = Some(c);
            }
            ("over", x) => {
                let t: f64 = x
                    .parse()
                    .map_err(|_| format!("[adaptive_dosing]: bad route over: {x}"))?;
                if !t.is_finite() || t <= 0.0 {
                    return Err(format!(
                        "[adaptive_dosing]: route over (infusion duration) must be finite and > 0: {x}"
                    ));
                }
                over = Some(t);
            }
            (other, _) => {
                return Err(format!(
                    "[adaptive_dosing]: unknown route argument `{other}`"
                ))
            }
        }
    }
    let cmt = cmt.ok_or_else(|| {
        format!(
            "[adaptive_dosing]: route `{}` requires cmt = N",
            kind.trim()
        )
    })?;
    match kind.trim() {
        "bolus" => {
            if over.is_some() {
                return Err(
                    "[adaptive_dosing]: bolus route does not take `over` (use infuse)".to_string(),
                );
            }
            Ok(AdaptiveRoute::Bolus { cmt })
        }
        "infuse" => {
            let over = over
                .ok_or("[adaptive_dosing]: infuse route requires over = T (infusion duration)")?;
            Ok(AdaptiveRoute::Infuse { cmt, over })
        }
        other => Err(format!(
            "[adaptive_dosing]: unknown route `{other}` (expected bolus or infuse)"
        )),
    }
}

/// Parse the body of a `when …` rule line (everything after `when `):
/// `<signal-condition> : <action>`.
fn parse_adaptive_rule(rest: &str) -> Result<AdaptiveRule, String> {
    let (cond, action) = rest.split_once(':').ok_or_else(|| {
        format!(
            "[adaptive_dosing]: rule must be `when <signal> <op> <value> : <action>`: when {rest}"
        )
    })?;
    let (op, threshold) = parse_signal_condition(cond.trim())?;
    let action = parse_rule_action(action.trim())?;
    Ok(AdaptiveRule {
        op,
        threshold,
        action,
    })
}

/// Parse `signal <op> <value>` where `<op>` is one of `< <= > >= ==`.
fn parse_signal_condition(cond: &str) -> Result<(Comparison, f64), String> {
    let rest = cond.trim().strip_prefix("signal").ok_or_else(|| {
        format!("[adaptive_dosing]: rule condition must compare `signal`: {cond}")
    })?;
    let rest = rest.trim_start();
    // Two-character operators first so `<=`/`>=` aren't read as `<`/`>`.
    let (op, num) = if let Some(r) = rest.strip_prefix("<=") {
        (Comparison::Le, r)
    } else if let Some(r) = rest.strip_prefix(">=") {
        (Comparison::Ge, r)
    } else if let Some(r) = rest.strip_prefix("==") {
        (Comparison::Eq, r)
    } else if let Some(r) = rest.strip_prefix('<') {
        (Comparison::Lt, r)
    } else if let Some(r) = rest.strip_prefix('>') {
        (Comparison::Gt, r)
    } else {
        return Err(format!(
            "[adaptive_dosing]: rule needs a comparison operator (< <= > >= ==): {cond}"
        ));
    };
    let threshold: f64 = num
        .trim()
        .parse()
        .map_err(|_| format!("[adaptive_dosing]: bad rule threshold: {}", num.trim()))?;
    if !threshold.is_finite() {
        return Err(format!(
            "[adaptive_dosing]: rule threshold must be finite: {cond}"
        ));
    }
    Ok((op, threshold))
}

/// Parse a rule action: `increase N%` / `decrease N%` (continuous), bare
/// `increase` / `decrease` (one discrete level), `hold`, or `stop`.
fn parse_rule_action(action: &str) -> Result<AdaptiveAction, String> {
    let a = action.trim();
    match a {
        "hold" => return Ok(AdaptiveAction::Hold),
        "stop" => return Ok(AdaptiveAction::Stop),
        "" => return Err("[adaptive_dosing]: rule has no action after `:`".to_string()),
        _ => {}
    }
    let (verb, operand) = match a.split_once(char::is_whitespace) {
        Some((v, o)) => (v, o.trim()),
        None => (a, ""),
    };
    let step = if operand.is_empty() {
        DoseStep::Level
    } else {
        let pct = operand.strip_suffix('%').ok_or_else(|| {
            format!("[adaptive_dosing]: dose change must be a percentage like `25%` (or bare for a level step): {action}")
        })?;
        let p: f64 = pct
            .trim()
            .parse()
            .map_err(|_| format!("[adaptive_dosing]: bad percentage in action: {action}"))?;
        if !p.is_finite() || p < 0.0 {
            return Err(format!(
                "[adaptive_dosing]: percentage must be finite and >= 0: {action}"
            ));
        }
        DoseStep::Percent(p)
    };
    match verb {
        "increase" => Ok(AdaptiveAction::Increase(step)),
        "decrease" => Ok(AdaptiveAction::Decrease(step)),
        other => Err(format!(
            "[adaptive_dosing]: unknown action `{other}` (expected increase/decrease/hold/stop): {action}"
        )),
    }
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
    } else if val == "laplace" || val == "laplacian" {
        // NONMEM `$EST METHOD=1 LAPLACIAN` — the exact-anchor quadrature. `n_agq = 1`
        // (default) is Laplace; `n_agq > 1` is adaptive Gauss–Hermite quadrature.
        Ok(EstimationMethod::Laplace)
    } else if val == "agq"
        || val == "aghq"
        || val == "gauss_hermite"
        || val == "gauss-hermite"
        || val == "adaptive_gaussian_quadrature"
        || val == "adaptive-gaussian-quadrature"
    {
        // `method = agq` was removed (#251): adaptive quadrature is not a separate estimator,
        // it is `method = laplace` with `n_agq > 1` (exact Hessian anchor) — or
        // `method = focei` with `n_agq > 1` for the Gauss-Newton anchor. This arm MUST stay
        // above the `contains("gauss")` arm below, which would otherwise route
        // `gauss_hermite` into Gauss-*Newton*.
        Err(
            "method = agq has been removed: use method = laplace with n_agq = N \
             (n_agq = 1 is the Laplace approximation, n_agq > 1 is adaptive Gauss–Hermite \
             quadrature). For the Gauss-Newton-anchored quadrature use method = focei with \
             n_agq > 1."
                .to_string(),
        )
    } else if val == "gn" || val.contains("gauss") {
        Ok(EstimationMethod::FoceGn)
    } else if val == "focei" || val == "foce-i" || val == "foce_i" || val.contains("interaction") {
        Ok(EstimationMethod::FoceI)
    } else if val == "foce" {
        Ok(EstimationMethod::Foce)
    } else if val == "bayes" || val == "bayesian" || val == "mcmc" {
        Ok(EstimationMethod::Bayes)
    } else if val == "vi" {
        // Exactly one spelling. Deliberately *not* accepting `advi`: this is not
        // automatic differentiation variational inference — the model is not
        // differentiated wholesale, it uses the same hand-written `Dual2`
        // sensitivities FOCE does.
        Ok(EstimationMethod::Vi)
    } else {
        Err(format!("unknown estimation method: `{}`", token.trim()))
    }
}

/// Canonical column roles a `[data]` block may remap (#730, NONMEM
/// `$INPUT TIME=TAFD` analogue). A target naming one of these is upper-cased to
/// the standard header spelling; any other target is an arbitrary column rename
/// (#742) kept in its written case. Stays in sync with the standard columns
/// recognised by [`crate::io::datareader`].
const DATA_BLOCK_ROLES: &[&str] = &[
    "id", "time", "dv", "evid", "amt", "cmt", "rate", "mdv", "ii", "ss", "cens", "addl", "tentry",
    "fremtype",
];

/// Parse a `[data]` block (#690, NONMEM `$DATA` analogue). Returns the raw
/// `path` value as written (resolved relative to the model file's directory by
/// `parse_full_model_file`) plus any `target = actual` column remappings.
/// Each map entry is `(target_header, actual_header)`: the dataset's
/// `actual_header` column is renamed to `target_header` when the CSV is read.
///
/// A `target` that names a canonical role (#730, DATA_BLOCK_ROLES) is upper-cased
/// to the standard header spelling; any other target (#742 — arbitrary column /
/// covariate rename) keeps the exact case written, since covariate lookups are
/// case-sensitive. This lets a dataset's own `dv` column be renamed aside so a
/// different column can take the `DV` role, e.g.
///
/// ```text
/// [data]
///   ODV = dv       # keep the raw DV under a new name
///   DV  = lndv     # promote the log-DV column to the DV role
///   WT  = weight   # rename an arbitrary covariate
/// ```
///
/// Validated here: an empty `path`, empty target names, a target mapped twice,
/// and two targets mapped to the same actual header. Whether a mapped header
/// actually exists in the dataset — and whether a rename collides with a
/// surviving column — can only be checked when the CSV is read, so that lives in
/// the datareader.
fn parse_data_block(lines: &[String]) -> Result<(String, Vec<(String, String)>), String> {
    let mut path: Option<String> = None;
    let mut column_map: Vec<(String, String)> = Vec::new();
    for line in lines {
        let parts: Vec<&str> = line.splitn(2, '=').map(|s| s.trim()).collect();
        if parts.len() != 2 {
            continue;
        }
        let (key, value) = (parts[0], parts[1].trim_matches(|c| c == '"' || c == '\''));
        let key_lc = key.to_ascii_lowercase();
        if key_lc == "path" {
            if value.is_empty() {
                return Err("[data]: empty `path` — write `path = <path-to-csv>`".to_string());
            }
            path = Some(value.to_string());
            continue;
        }
        if value.is_empty() {
            return Err(format!(
                "[data]: empty column name for `{key}` — write `{key} = <header>`"
            ));
        }
        // Canonical roles use the standard upper-case spelling; arbitrary
        // targets (#742) keep their exact case for case-sensitive covariate
        // lookups.
        let target = if DATA_BLOCK_ROLES.contains(&key_lc.as_str()) {
            key_lc.to_ascii_uppercase()
        } else {
            key.to_string()
        };
        if column_map
            .iter()
            .any(|(t, _)| t.eq_ignore_ascii_case(&target))
        {
            return Err(format!("[data]: target `{target}` mapped more than once"));
        }
        column_map.push((target, value.to_string()));
    }
    // Reject two targets pointing at the same actual header (case-insensitive):
    // the reader would rename one header to two targets, silently dropping one.
    for i in 0..column_map.len() {
        for j in (i + 1)..column_map.len() {
            if column_map[i].1.eq_ignore_ascii_case(&column_map[j].1) {
                return Err(format!(
                    "[data]: targets `{}` and `{}` both map to column `{}`",
                    column_map[i].0, column_map[j].0, column_map[i].1
                ));
            }
        }
    }
    let path = path.ok_or_else(|| "[data]: missing required key `path`".to_string())?;
    Ok((path, column_map))
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
/// Parse the `iov_occasion` fit-option value into an [`IovOccasionRule`].
///
/// Accepts:
///   - `column` (or empty/none/null/na) → [`IovOccasionRule::Column`]
///   - `dose`                           → [`IovOccasionRule::PerDose`]
///   - `time(24, 48)`                   → [`IovOccasionRule::TimeWindows`]
///
/// Breakpoints must be finite, strictly increasing, and must not include `0`
/// (the first occasion already spans everything before the first breakpoint, so
/// a leading `0` is implied and would only carve out an empty leading occasion).
fn parse_iov_occasion(value: &str) -> Result<IovOccasionRule, String> {
    let v = value.trim();
    let lower = v.to_ascii_lowercase();
    if lower.is_empty() || lower == "column" || lower == "none" || lower == "null" || lower == "na"
    {
        return Ok(IovOccasionRule::Column);
    }
    if lower == "dose" {
        return Ok(IovOccasionRule::PerDose);
    }
    if let Some(rest) = lower.strip_prefix("time") {
        let inner = rest.trim();
        let inner = inner
            .strip_prefix('(')
            .and_then(|s| s.strip_suffix(')'))
            .ok_or_else(|| {
                format!(
                    "fit option `iov_occasion`: expected `time(edge1, edge2, ...)`, got `{value}`"
                )
            })?;
        let mut edges = Vec::new();
        for tok in inner.split(',') {
            let tok = tok.trim();
            if tok.is_empty() {
                continue;
            }
            let x = tok.parse::<f64>().map_err(|_| {
                format!("fit option `iov_occasion`: breakpoint `{tok}` is not a number")
            })?;
            if !x.is_finite() {
                return Err(format!(
                    "fit option `iov_occasion`: breakpoint `{tok}` must be finite"
                ));
            }
            edges.push(x);
        }
        if edges.is_empty() {
            return Err(
                "fit option `iov_occasion`: `time(...)` needs at least one breakpoint".to_string(),
            );
        }
        // A `0` breakpoint is redundant and wrong: the first occasion already
        // spans everything before the first breakpoint — `[-inf, edge1)` — so a
        // leading `0` only carves out an empty occasion for `t < 0` and shifts
        // every real occasion up by one. Reject it with a message that names the
        // fix (a common `c(0, 24, 48)`-style mistake).
        if edges.iter().any(|&e| e == 0.0) {
            return Err(format!(
                "fit option `iov_occasion`: `time(...)` breakpoints must not include `0`. The \
                 first occasion already covers everything before the first breakpoint (`[-inf, \
                 edge1)`), and the last covers everything at or after the last breakpoint (to \
                 `+inf`), so `0` is implied and does not need to be specified — a leading `0` \
                 would create an empty leading occasion. Write `time(24, 48)`, not \
                 `time(0, 24, 48)` (got `{value}`)."
            ));
        }
        if edges.windows(2).any(|w| w[1] <= w[0]) {
            return Err(format!(
                "fit option `iov_occasion`: `time(...)` breakpoints must be strictly increasing, \
                 got `{value}`"
            ));
        }
        return Ok(IovOccasionRule::TimeWindows(edges));
    }
    Err(format!(
        "fit option `iov_occasion`: unknown value `{value}` — expected `column`, `dose`, or \
         `time(edge1, edge2, ...)`"
    ))
}

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
    // Parse an f64 that must be strictly positive and finite.
    let parse_pos_finite = |name: &str| -> Result<f64, String> {
        let v = parse_f64(name)?;
        if v <= 0.0 || !v.is_finite() {
            return Err(format!("{name} must be a positive finite value, got {v}"));
        }
        Ok(v)
    };
    // Parse an f64 with an inclusive lower bound (`v >= min`). `min` is rendered
    // with Debug so integral bounds keep their trailing `.0` (e.g. `>= 1.0`).
    let parse_f64_min = |name: &str, min: f64| -> Result<f64, String> {
        let v = parse_f64(name)?;
        if v < min {
            return Err(format!("{name} must be >= {min:?}, got {v}"));
        }
        Ok(v)
    };
    // Parse an f64 constrained to the closed unit interval [0.0, 1.0].
    let parse_unit_interval = |name: &str| -> Result<f64, String> {
        let v = parse_f64(name)?;
        if !(0.0..=1.0).contains(&v) {
            return Err(format!("{name} must be in [0.0, 1.0], got {v}"));
        }
        Ok(v)
    };
    // Parse a Student-t proposal df: `normal`/`mvn` select a multivariate-normal
    // proposal (∞ df), otherwise a finite df `>= 1.0`. `strip_quotes` mirrors the
    // impmap site, which tolerates a quoted `"normal"` token.
    let parse_proposal_df = |name: &str, strip_quotes: bool| -> Result<f64, String> {
        let tok = if strip_quotes {
            value.trim_matches(|c| c == '"' || c == '\'')
        } else {
            value
        };
        if tok.eq_ignore_ascii_case("normal") || tok.eq_ignore_ascii_case("mvn") {
            Ok(f64::INFINITY)
        } else {
            let v = parse_f64(name)?;
            if v < 1.0 {
                return Err(format!("{name} must be >= 1.0 or `normal`, got {v}"));
            }
            Ok(v)
        }
    };

    // Dispatch first, then record the key on success so we can later warn
    // when a key is set that the selected method does not consume. Malformed
    // values still return `Err` and don't get recorded.
    match key {
        "maxiter" => opts.outer_maxiter = parse_usize("maxiter")?,
        "inner_maxiter" => opts.inner_maxiter = parse_usize("inner_maxiter")?,
        "inner_tol" => opts.inner_tol = parse_f64("inner_tol")?,
        "inner_restarts" => opts.inner_restarts = parse_usize("inner_restarts")?,
        "cov_inner_tol" => opts.cov_inner_tol = Some(parse_f64("cov_inner_tol")?),
        "outer_xtol" => opts.outer_xtol = parse_pos_finite("outer_xtol")?,
        "outer_ftol" => opts.outer_ftol = Some(parse_pos_finite("outer_ftol")?),
        "ode_reltol" => opts.ode_reltol = parse_pos_finite("ode_reltol")?,
        "ode_abstol" => opts.ode_abstol = parse_pos_finite("ode_abstol")?,
        "ode_max_steps" => {
            let v = parse_usize("ode_max_steps")?;
            if v == 0 {
                return Err("ode_max_steps must be a positive integer".to_string());
            }
            opts.ode_max_steps = v;
        }
        "ode_method" => {
            opts.ode_method = crate::ode::OdeMethod::parse(value).ok_or_else(|| {
                format!(
                    "fit option `ode_method`: unknown value `{value}` — expected one of \
                     rk45, vern7, rosenbrock23, rodas4, rodas5p (aliases: dopri5, verner7, \
                     ros23, ode23s, rodas5)"
                )
            })?;
        }
        "covariance" => opts.run_covariance_step = parse_bool("covariance")?,
        "analytic_cov_hessian" => opts.analytic_cov_hessian = parse_bool("analytic_cov_hessian")?,
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
        "fd_hessian_step" => opts.fd_hessian_step = parse_pos_finite("fd_hessian_step")?,
        "verbose" => opts.verbose = parse_bool("verbose")?,
        "optimizer" => {
            opts.optimizer = match value.to_lowercase().as_str() {
                // `auto` (the default) picks nlopt_lbfgs when the analytic
                // FOCE/FOCEI gradient is available, bobyqa otherwise (#490).
                "auto" => Optimizer::Auto,
                "slsqp" => Optimizer::Slsqp,
                // `lbfgs` and `bfgs` are deprecated aliases for `nlopt_lbfgs`: they now
                // select the NLopt L-BFGS (+ SLSQP polish) path. The hand-rolled
                // built-in BFGS/L-BFGS is strictly worse — 3–5× slower on
                // analytic-gradient models and prone to diverging or hanging on harder
                // problems (see #483) — so the keyword no longer reaches it. The
                // `Optimizer::Bfgs`/`Lbfgs` variants remain for a Rust caller that
                // constructs them directly, pending removal.
                "lbfgs" | "bfgs" | "nlopt_lbfgs" => Optimizer::NloptLbfgs,
                "mma" => Optimizer::Mma,
                "bobyqa" => Optimizer::Bobyqa,
                "trust_region" | "newton_tr" => Optimizer::TrustRegion,
                other => {
                    return Err(format!(
                        "fit option `optimizer`: unknown value `{other}` — expected \
                         auto/bobyqa/slsqp/nlopt_lbfgs/mma/trust_region (`lbfgs` and \
                         `bfgs` are accepted as deprecated aliases for `nlopt_lbfgs`)"
                    ));
                }
            };
        }
        "inner_optimizer" => {
            opts.inner_optimizer = match value.to_lowercase().as_str() {
                "auto" => crate::types::InnerOptimizer::Auto,
                "bfgs" => crate::types::InnerOptimizer::Bfgs,
                "lbfgs" => crate::types::InnerOptimizer::Lbfgs,
                "nelder_mead" | "neldermead" => crate::types::InnerOptimizer::NelderMead,
                other => {
                    return Err(format!(
                        "fit option `inner_optimizer`: unknown value `{other}` — expected \
                         auto/bfgs/lbfgs/nelder_mead"
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
        "mstep_damping" | "saem_mstep_damping" => {
            let v = parse_f64(key)?;
            // `1.0` is the documented "off" value (the pre-#1011 assignment), so
            // the range is half-open at the bottom and closed at the top.
            // Report back the spelling the user wrote, and the value they wrote,
            // the way the neighbouring range validators do.
            if !(v > 0.0 && v <= 1.0) {
                return Err(format!(
                    "fit option `{key}` must be in (0, 1] — smaller damps the SAEM numerical \
                     θ/σ M-step harder, 1.0 disables the damping (#1011), got {v}"
                ));
            }
            opts.saem_mstep_damping = Some(v);
        }
        "omega_burnin" => opts.saem_omega_burnin = parse_usize("omega_burnin")?,
        "conddist" | "saem_conddist" => opts.saem_conddist = parse_bool("conddist")?,
        "conddist_nsamp" => opts.saem_conddist_nsamp = parse_usize("conddist_nsamp")?,
        "conddist_burnin" => opts.saem_conddist_burnin = parse_usize("conddist_burnin")?,
        "conddist_keep_samples" => {
            opts.saem_conddist_keep_samples = parse_bool("conddist_keep_samples")?
        }
        "seed" | "saem_seed" => opts.saem_seed = parse_u64_opt("seed")?,
        "bayes_warmup" => opts.bayes_warmup = parse_usize("bayes_warmup")?,
        "bayes_iters" => opts.bayes_iters = parse_usize("bayes_iters")?,
        "bayes_chains" => opts.bayes_chains = parse_usize("bayes_chains")?,
        "bayes_thin" => opts.bayes_thin = parse_usize("bayes_thin")?,
        "bayes_seed" => opts.bayes_seed = parse_u64_opt("bayes_seed")?,
        "vi_iters" => {
            let v = parse_usize("vi_iters")?;
            if v < 1 {
                return Err("vi_iters must be >= 1".to_string());
            }
            opts.vi_iters = v;
        }
        "vi_mc_samples" => {
            let v = parse_usize("vi_mc_samples")?;
            if v < 1 {
                return Err("vi_mc_samples must be >= 1".to_string());
            }
            opts.vi_mc_samples = v;
        }
        "vi_lr" => {
            let v = parse_f64("vi_lr")?;
            if !(v > 0.0) {
                return Err(format!("vi_lr must be > 0, got {v}"));
            }
            opts.vi_lr = v;
        }
        "vi_family" => {
            opts.vi_family = match value.trim().to_lowercase().as_str() {
                "full_rank" | "fullrank" => crate::types::ViFamily::FullRank,
                "mean_field" | "meanfield" | "diagonal" => crate::types::ViFamily::MeanField,
                other => {
                    return Err(format!(
                        "vi_family must be `full_rank` or `mean_field`, got `{other}`"
                    ))
                }
            }
        }
        "vi_omega_update" => {
            opts.vi_omega_update = match value.trim().to_lowercase().as_str() {
                "closed_form" | "closedform" => crate::types::ViOmegaUpdate::ClosedForm,
                "adam" => crate::types::ViOmegaUpdate::Adam,
                other => {
                    return Err(format!(
                        "vi_omega_update must be `closed_form` or `adam`, got `{other}`"
                    ))
                }
            }
        }
        "vi_avg_last" => {
            let v = parse_usize("vi_avg_last")?;
            if v < 1 {
                return Err("vi_avg_last must be >= 1".to_string());
            }
            opts.vi_avg_last = Some(v);
        }
        "vi_eta_grad" => {
            opts.vi_eta_grad = match value.trim().to_lowercase().as_str() {
                "auto" => crate::types::ViEtaGrad::Auto,
                "analytic" => crate::types::ViEtaGrad::Analytic,
                "fd" => crate::types::ViEtaGrad::Fd,
                other => {
                    return Err(format!(
                        "vi_eta_grad must be `auto`, `analytic`, or `fd`, got `{other}`"
                    ))
                }
            }
        }
        "vi_final_ofv" => {
            opts.vi_final_ofv = match value.trim().to_lowercase().as_str() {
                "none" => crate::types::ViFinalOfv::None,
                "laplace" => crate::types::ViFinalOfv::Laplace,
                other => {
                    return Err(format!(
                        "vi_final_ofv must be `none` or `laplace`, got `{other}`. For an \
                         importance-sampling −2 log L, chain `methods = vi, imp` with \
                         `imp_eval_only = true`."
                    ))
                }
            }
        }
        "vi_seed" => opts.vi_seed = parse_u64_opt("vi_seed")?,
        "gn_lambda" => opts.gn_lambda = parse_f64("gn_lambda")?,
        "sir" => opts.sir = parse_bool("sir")?,
        "sir_samples" => opts.sir_samples = parse_usize("sir_samples")?,
        "sir_resamples" => opts.sir_resamples = parse_usize("sir_resamples")?,
        "sir_seed" => opts.sir_seed = parse_u64_opt("sir_seed")?,
        "sir_keep_samples" => opts.sir_keep_samples = parse_bool("sir_keep_samples")?,
        "sir_df" => opts.sir_df = parse_f64_min("sir_df", 1.0)?,
        "n_agq" => {
            let v = parse_usize("n_agq")?;
            if v < 1 {
                return Err("n_agq must be >= 1 (1 = the Laplace approximation)".to_string());
            }
            if v > crate::estimation::agq::MAX_AGQ_NODES {
                return Err(format!(
                    "n_agq must be <= {}, got {v}",
                    crate::estimation::agq::MAX_AGQ_NODES
                ));
            }
            opts.n_agq = v;
        }
        "imp_samples" => {
            let v = parse_usize("imp_samples")?;
            if v < 2 {
                return Err(format!("imp_samples must be >= 2, got {v}"));
            }
            opts.imp_samples = v;
        }
        "imp_proposal_df" => opts.imp_proposal_df = parse_proposal_df("imp_proposal_df", false)?,
        "imp_seed" => opts.imp_seed = parse_u64_opt("imp_seed")?,
        "imp_low_ess_threshold" => {
            opts.imp_low_ess_threshold = parse_unit_interval("imp_low_ess_threshold")?
        }
        "imp_iterations" => {
            let v = parse_usize("imp_iterations")?;
            if v < 1 {
                return Err(format!("imp_iterations must be >= 1, got {v}"));
            }
            opts.imp_iterations = v;
        }
        "imp_averaging" => opts.imp_averaging = parse_usize("imp_averaging")?,
        "imp_eval_only" => opts.imp_eval_only = parse_bool("imp_eval_only")?,
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
        // `normal` / `mvn` (or a very large df) select a multivariate-normal
        // proposal — NONMEM's IMPMAP default. A finite value gives Student-t.
        // `strip_quotes = true` tolerates a quoted `"normal"` token.
        "impmap_proposal_df" => {
            opts.impmap_proposal_df = parse_proposal_df("impmap_proposal_df", true)?
        }
        "impmap_seed" => opts.impmap_seed = parse_u64_opt("impmap_seed")?,
        "impmap_averaging" => opts.impmap_averaging = parse_usize("impmap_averaging")?,
        "impmap_low_ess_threshold" => {
            opts.impmap_low_ess_threshold = parse_unit_interval("impmap_low_ess_threshold")?
        }
        "impmap_trace" => opts.impmap_trace = parse_bool("impmap_trace")?,
        "impmap_mceta" => opts.impmap_mceta = parse_usize("impmap_mceta")?,
        "impmap_sobol" => opts.impmap_sobol = parse_bool("impmap_sobol")?,
        "frem_rao_blackwell" => opts.frem_rao_blackwell = parse_bool("frem_rao_blackwell")?,
        "imp_auto" => opts.imp_auto = parse_bool("imp_auto")?,
        "impmap_auto" => opts.impmap_auto = parse_bool("impmap_auto")?,
        "iscale_min" => {
            let v = parse_f64("iscale_min")?;
            if v <= 0.0 {
                return Err("fit option `iscale_min` must be > 0".to_string());
            }
            opts.iscale_min = v;
        }
        "iscale_max" => {
            let v = parse_f64("iscale_max")?;
            if v <= 0.0 {
                return Err("fit option `iscale_max` must be > 0".to_string());
            }
            opts.iscale_max = v;
        }
        // `impmap_defensive_alpha` is an alias for `imp_defensive_alpha`: the
        // defensive mixture is shared IMP/IMPMAP machinery, so both names write
        // the same field (issue #528). The alias keeps the `impmap_*` namespace
        // discoverable for IMPMAP users.
        "imp_defensive_alpha" | "impmap_defensive_alpha" => {
            let v = parse_f64(key)?;
            if !(0.0..1.0).contains(&v) {
                return Err(format!("fit option `{key}` must be in [0, 1), got {v}"));
            }
            opts.imp_defensive_alpha = v;
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
                // `ad` still *parses* so that requesting it reaches the engine's specific
                // "the Enzyme AD path was retired, use auto/fd" error (#428) rather than a
                // generic unknown-value one. It is not advertised below: listing it as a
                // valid value in the message shown for some *other* typo pointed users at a
                // setting no fit accepts (#958).
                "ad" | "autodiff" => GradientMethod::Ad,
                "fd" | "finite" | "finite_difference" | "finite-difference" => GradientMethod::Fd,
                other => {
                    return Err(format!(
                        "fit option `gradient`: unknown value `{other}` — expected 'auto' or 'fd'"
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
        "iov_occasion" => {
            opts.iov_occasion = parse_iov_occasion(value)?;
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
        "ebe_warm_start" => opts.ebe_warm_start = parse_bool("ebe_warm_start")?,
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
        "frem_predictions" => {
            opts.frem_predictions = if value.is_empty() {
                None
            } else {
                Some(value.to_string())
            };
        }
        "frem_sigma" => {
            opts.frem_sigma = if value.is_empty() {
                None
            } else {
                Some(value.to_string())
            };
        }
        "checkpoint" => opts.checkpoint = parse_bool("checkpoint")?,
        "checkpoint_interval_secs" => {
            opts.checkpoint_interval_secs = parse_usize("checkpoint_interval_secs")? as u64
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

/// Lower a parsed scalar expression to a `Dual2`-differentiable
/// [`ScaleDerivProgram`] over `(θ, η, individual-parameter vars, covariates)`.
///
/// Single source of truth for the `obs_scale` (issue #367) and `init` amount
/// (issue #524) programs — they share the same axis layout, so they must lower
/// identically or the analytic provider would differentiate the two on
/// inconsistent ABIs. A `Variable(name)` (an individual parameter) → var slot `i`
/// (flat PK slot `pk_indices[i]`); covariates → sorted cov slots; θ/η are read
/// directly from the seed duals. The input `expr` is cloned, so the caller keeps
/// its AST for the runtime closure.
fn compile_scale_deriv_program(
    expr: &Expression,
    theta_names: &[String],
    eta_names: &[String],
    indiv_var_names: &[String],
    pk_indices: &[usize],
) -> ScaleDerivProgram {
    let mut e = expr.clone();
    let var_idx: HashMap<String, usize> = indiv_var_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), i))
        .collect();
    let var_to_pk_slot: Vec<usize> = (0..indiv_var_names.len())
        .map(|i| pk_indices.get(i).copied().unwrap_or(i))
        .collect();
    let mut cov_set = std::collections::HashSet::new();
    collect_covariates(&e, &mut cov_set);
    let mut cov_names: Vec<String> = cov_set.into_iter().collect();
    cov_names.sort();
    let cov_idx: HashMap<String, usize> = cov_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();
    resolve_expr_indices(&mut e, &var_idx, &cov_idx);
    ScaleDerivProgram {
        bc: compile_bytecode(&e),
        n_theta: theta_names.len(),
        n_eta: eta_names.len(),
        var_to_pk_slot,
        cov_names,
    }
}

/// Build a `ScalingSpec` (None / ScalarScale / ExpressionScale) from one
/// `obs_scale[…] = value` line. Shared between the uniform and per-CMT paths.
///
/// `volume_indiv_name` is the individual-parameter name bound to the built-in
/// `pk <model>(...)` structural block's `v`/`v1` role (`None` for `ode(...)`/
/// `ode_template(...)`, where there's no such built-in concentration divisor to
/// collide with). Referencing that exact name in a Form B expression is a real,
/// supported feature (an intentional additional transform on top of the
/// built-in concentration output — see `docs/model-file/scaling.qmd` and
/// `examples/scaling_expression.ferx`), not rejected. But it is *also* the
/// signature of a common mistake: copying `obs_scale = V` from an `ode(...)`
/// translation of the same model, where it's required (raw output there is
/// amount, not concentration) — on a `pk` block it silently divides by `v` a
/// second time. Since ferx can't tell intent from mistake here, it warns
/// (`parse_warnings`) rather than erroring (#712).
#[allow(clippy::too_many_arguments)]
fn build_obs_scale_spec(
    value: &str,
    theta_names: &[String],
    eta_names: &[String],
    indiv_var_names: &[String],
    pk_indices: &[usize],
    volume_indiv_name: Option<&str>,
    declared_covariates: &[String],
    intermediates: &[(String, String)],
    parse_warnings: &mut Vec<String>,
) -> Result<(ScalingSpec, Vec<String>), String> {
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
        return Ok((ScalingSpec::ScalarScale(k), Vec::new()));
    }
    let ctx = ParseCtx::new(theta_names, eta_names, indiv_var_names);
    let mut expr =
        parse_scalar_expression(value, ctx).map_err(|e| format!("[scaling] obs_scale: {}", e))?;
    // Named intermediates are inlined before any of the checks below, so each one
    // sees the expression the user would have had to write by hand (#1030).
    inline_scaling_intermediates(&mut expr, intermediates, ctx)?;
    // `obs_scale` is a **subject-static** divisor: `apply_scaling` evaluates it once
    // per subject against the baseline covariate map and a `t = 0` `pk_param_fn`
    // snapshot, then divides the whole prediction vector by the result. A `TIME`
    // reference in it would therefore read whatever the model-time thread-local
    // happened to hold — in practice the `0.0` default — and silently scale every
    // observation by the t=0 value. Reject it rather than serve that (#1028); a
    // genuinely time-dependent readout is Form C's job, where `TIME` resolves per
    // observation.
    if expr_references_time_builtin(&expr, declared_covariates) {
        return Err(format!(
            "[scaling] obs_scale: `{}` references the `TIME` built-in (or its `T` alias), \
             but `obs_scale` is a subject-static divisor — it is evaluated once per \
             subject at t = 0 and applied to every observation, so `TIME` would always \
             read 0. Write the time-dependent readout as Form C instead \
             (`y = <expr>`, or `y[CMT=N] = <expr>`), where `TIME` resolves to each \
             observation's own time. See issue #1028.",
            value.trim()
        ));
    }
    if let Some(vname) = volume_indiv_name {
        let mut references_volume = false;
        visit_expr_nodes(&expr, &mut |e| {
            if let Expression::Variable(name) = e {
                if name == vname {
                    references_volume = true;
                }
            }
        });
        if references_volume {
            parse_warnings.push(format!(
                "[scaling]: obs_scale = `{value}` references `{vname}`, which is also this \
                 model's `pk` structural block volume parameter. A built-in analytical `pk` \
                 block already outputs concentration (divides by `{vname}` internally) — this \
                 `obs_scale` divides by `{vname}` again on top. If that's intentional (a \
                 deliberate additional transform), ignore this warning. If `obs_scale = {vname}` \
                 was copied from an `ode(...)` translation of this model (where it converts raw \
                 compartment amount to concentration and is required), remove `[scaling]` here — \
                 the `pk` block's built-in concentration output is already what that conversion \
                 was for."
            ));
        }
    }
    // Covariate leaves — every identifier the parse could not bind to a theta, an
    // eta, or an individual parameter. The `scale_fn` below reads them straight out
    // of the per-subject covariate map, so they are required data columns and must
    // be registered as such (issue #1028). Before this they were silently dropped:
    // a typo'd name (or a real covariate the data didn't carry) resolved to the
    // map's `0.0` default, and the divisive scale then turned every prediction into
    // `x / 0` → the `apply_scaling` NaN/zero path, with no diagnostic. The `y`
    // (Form C) half of `[scaling]` has registered its covariates since #540; this
    // closes the `obs_scale` half. Returned unsorted — `register_referenced_covariates`
    // pools and re-sorts.
    let mut cov_set: std::collections::HashSet<String> = std::collections::HashSet::new();
    collect_covariates(&expr, &mut cov_set);
    let covariates_ref: Vec<String> = cov_set.into_iter().collect();
    // Pre-resolve indiv param name → PK slot so the closure can look up
    // `pk.values[slot]` for each `Expression::Variable(name)`. Mirrors
    // the analytical Form B path from Phase 1.5.
    let indiv_to_pk_slot: HashMap<String, usize> = indiv_var_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), pk_indices.get(i).copied().unwrap_or(i)))
        .collect();
    // Differentiable scale program (issue #367): compile the same expression to
    // bytecode so the analytic sensitivity provider can differentiate `f / scale`
    // exactly instead of finite-differencing the opaque closure. Shared lowering
    // with the `init` amount program (issue #524) — see `compile_scale_deriv_program`.
    let deriv = Some(compile_scale_deriv_program(
        &expr,
        theta_names,
        eta_names,
        indiv_var_names,
        pk_indices,
    ));
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
    Ok((
        ScalingSpec::ExpressionScale { scale_fn, deriv },
        covariates_ref,
    ))
}

/// Resolve a `[initial_conditions] init(NAME)` compartment name to a 1-based
/// compartment index for an **analytical** `pk_model`, validating that an
/// initial amount in that compartment is supported (issue #521).
///
/// Accepted names: `central` (every model), `depot` (first-order oral models,
/// cmt 1), or a 1-based integer index. Peripheral compartments are rejected —
/// seeding a peripheral needs the cross-compartment Green's function, which the
/// closed forms don't expose; use an ODE model for that.
fn analytical_init_cmt(pk_model: PkModel, name: &str) -> Result<usize, String> {
    let is_oral = pk_model.is_oral();
    let central = if is_oral { 2 } else { 1 };
    // A pre-loaded *depot* amount is delivered to central by a first-order
    // absorption Green's function, which only the oral models expose. The analytic
    // transit models (`one_cpt_transit`, `two_cpt_transit`) are `is_oral()`
    // (absorption layout, central = cmt 2) but their "depot" is a lumped
    // Gamma-convolution memory with no closed form for a seeded amount, so a transit
    // depot init is rejected here — otherwise `analytical_init_concentration_g`'s
    // depot arm would silently drop it (#386).
    let depot_seedable = matches!(
        pk_model,
        PkModel::OneCptOral | PkModel::TwoCptOral | PkModel::ThreeCptOral
    );
    let lname = name.trim().to_lowercase();

    let cmt = if let Ok(n) = lname.parse::<usize>() {
        n
    } else {
        match lname.as_str() {
            "central" => central,
            "depot" if depot_seedable => 1,
            "depot"
                if matches!(
                    pk_model,
                    PkModel::OneCptTransit
                        | PkModel::TwoCptTransit
                        | PkModel::OneCptIg
                        | PkModel::TwoCptIg
                ) =>
            {
                return Err(format!(
                    "[initial_conditions]: `{}` (analytic absorption) does not support a `depot` \
                     initial amount — its absorption chain is a lumped convolution with no closed \
                     form for a pre-loaded depot. Use `central`, or an ODE absorption model \
                     (transit()/igd() forcing in [odes]) for a depot initial condition.",
                    pk_model.canonical_name()
                ));
            }
            "depot" => {
                return Err(format!(
                    "[initial_conditions]: `depot` is only valid for oral models; \
                     `{}` has no depot compartment.",
                    pk_model.canonical_name()
                ));
            }
            _ => {
                return Err(format!(
                    "[initial_conditions]: unknown compartment `{}`. Use `central`{}, \
                     or a 1-based CMT index.",
                    name,
                    if depot_seedable { " or `depot`" } else { "" }
                ));
            }
        }
    };

    if cmt == central || (depot_seedable && cmt == 1) {
        Ok(cmt)
    } else if matches!(
        pk_model,
        PkModel::OneCptTransit | PkModel::TwoCptTransit | PkModel::OneCptIg | PkModel::TwoCptIg
    ) && cmt == 1
    {
        // Numeric `init(1)` on transit / IG: same lumped-convolution limitation as
        // the named `depot` form above (#386/#790).
        Err(format!(
            "[initial_conditions]: `{}` (analytic absorption) does not support a depot (cmt 1) \
             initial amount — its absorption chain is a lumped convolution with no closed form \
             for a pre-loaded depot. Use `central` (cmt 2), or an ODE absorption model.",
            pk_model.canonical_name()
        ))
    } else {
        Err(format!(
            "[initial_conditions]: initial amounts are only supported in the central \
             compartment{} (got compartment {}). A peripheral-compartment initial amount \
             needs the cross-compartment impulse response, which the analytical closed \
             forms don't expose — use an ODE model with `init(...)` in [odes] (issue #521).",
            if depot_seedable {
                " or the oral depot"
            } else {
                ""
            },
            cmt
        ))
    }
}

/// Compile one `init(NAME) = <expr>` right-hand side into a [`ScaleFn`] that
/// evaluates the initial amount `A₀` from `(theta, eta, covariates, pk_params)`.
/// Mirrors the Form-B `obs_scale` expression closure in [`build_obs_scale_spec`]:
/// individual-parameter names resolve from the subject-static `pk_params`, so
/// `init(central) = CONC0 * V` reads `V` from its PK slot.
///
/// The third tuple element is the covariate names the expression references (an
/// identifier that is not a theta/eta/individual parameter parses as a
/// `Covariate` leaf). The caller registers these in
/// `CompiledModel.referenced_covariates` so they become required data columns —
/// otherwise a miscased or absent init covariate silently evaluates to `0` and
/// the baseline vanishes with no diagnostic (issue #765).
#[allow(clippy::type_complexity)]
fn build_init_amount_fn(
    value: &str,
    theta_names: &[String],
    eta_names: &[String],
    indiv_var_names: &[String],
    pk_indices: &[usize],
    kappa_names: &[String],
) -> Result<(ScaleFn, Option<ScaleDerivProgram>, Vec<String>), String> {
    // Build the differentiable `A₀` program from a (possibly resolved) expression
    // AST. Shares the `obs_scale` lowering (`compile_scale_deriv_program`) so the
    // init impulse and the scale divisor differentiate on the same ABI; this lets
    // the analytic sensitivity provider differentiate the init impulse exactly
    // (issue #524) instead of finite-differencing `amount_fn`.
    let build_deriv = |expr: &Expression| -> ScaleDerivProgram {
        compile_scale_deriv_program(expr, theta_names, eta_names, indiv_var_names, pk_indices)
    };

    // Constant fast path (e.g. `init(central) = 0.0`). Still build a (constant)
    // deriv program so a constant baseline keeps `gradient = auto` support.
    if let Ok(k) = value.parse::<f64>() {
        let deriv = build_deriv(&Expression::Literal(k));
        let scale_fn: ScaleFn =
            Box::new(move |_: &[f64], _: &[f64], _: &HashMap<String, f64>, _: &PkParams| k);
        return Ok((scale_fn, Some(deriv), Vec::new()));
    }
    let ctx = ParseCtx::new(theta_names, eta_names, indiv_var_names);
    let expr = parse_scalar_expression(value, ctx)
        .map_err(|e| format!("[initial_conditions] init: {}", e))?;

    // Same `TIME` rejection as the `[odes]` init directive (#994): the analytical
    // baseline amount is evaluated once per subject at t = 0, so an
    // `Expression::Time` here reads the model-time thread-local's 0.0 and the
    // reference silently contributes nothing. Only `TIME` is rejected on this
    // surface — `T` / `TAFD` / `TAD` / `MACHEPS` are *not* built-ins outside
    // `[odes]`, they parse as covariate leaves and become required data columns,
    // the same deliberate rule `[scaling]` follows (#1028). See
    // `ODE_INIT_REJECTED_BUILTINS`.
    if expr_references_time_node(&expr) {
        return Err(format!(
            "[initial_conditions] init: `{}` references the `TIME` built-in, but an \
             initial condition is evaluated at the time origin — `TIME` there is \
             always exactly 0, so the expression can only ever read as its t = 0 \
             value. Write that value directly; a time-dependent readout belongs in \
             `[scaling]` Form C (`y = <expr>`), where `TIME` resolves per \
             observation. See issue #994.",
            value.trim()
        ));
    }

    // Reject KAPPA_* (IOV) references, mirroring the Form C ODE readout guard
    // (`build_y_output_fn`, issue #107). The init expression's eta scope is
    // BSV-only, so a kappa name parses as an unresolved identifier and would
    // silently evaluate to 0 — giving a wrong baseline with no error. The
    // baseline amount is a t=0 quantity built from BSV parameters; reference the
    // occasion-dependent structural parameter (e.g. CL) if a κ-driven starting
    // amount is needed (issue #521 review).
    if let Some(name) = expr_references_kappa(&expr, kappa_names) {
        return Err(format!(
            "[initial_conditions] init: expression cannot reference the IOV \
             parameter `{name}` — the baseline amount is evaluated once with the \
             BSV eta and would see kappa = 0. Reference the occasion-dependent \
             structural parameter (e.g. CL) instead."
        ));
    }
    let deriv = build_deriv(&expr);
    // Covariate leaves (identifiers not resolved as theta/eta/individual param)
    // must become required data columns — see the fn doc (issue #765).
    let mut cov_set: std::collections::HashSet<String> = std::collections::HashSet::new();
    collect_covariates(&expr, &mut cov_set);
    // Order is irrelevant: the caller pools these into a HashSet and the final
    // `register_referenced_covariates` re-sorts, so don't sort here.
    let covariates_ref: Vec<String> = cov_set.into_iter().collect();
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
    Ok((scale_fn, Some(deriv), covariates_ref))
}

/// Parse the `[initial_conditions]` block (issue #521) into the analytical
/// model's [`AnalyticalInit`] list. Each `init(NAME) = <expr>` declares a
/// non-zero starting amount in compartment `NAME`. Analytical models only —
/// ODE models seed state via `init(...)` inside `[odes]`.
fn parse_initial_conditions_block(
    lines: &[String],
    pk_model: PkModel,
    is_ode: bool,
    theta_names: &[String],
    eta_names: &[String],
    indiv_var_names: &[String],
    pk_indices: &[usize],
    kappa_names: &[String],
) -> Result<(Vec<crate::types::AnalyticalInit>, Vec<String>), String> {
    if is_ode {
        return Err(
            "[initial_conditions]: this block is for analytical PK models; for an \
                    ODE model declare `init(state) = <expr>` inside the [odes] block instead."
                .into(),
        );
    }
    let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut out = Vec::new();
    // Covariate names referenced across all init expressions; the caller
    // registers these as required data columns (issue #765).
    let mut cov_refs: std::collections::HashSet<String> = std::collections::HashSet::new();
    for line in lines {
        let (name, expr) = parse_init_line(line).ok_or_else(|| {
            format!(
                "[initial_conditions]: expected `init(NAME) = <expr>`, got: `{}`",
                line.trim()
            )
        })?;
        let cmt = analytical_init_cmt(pk_model, &name)?;
        if !seen.insert(cmt) {
            return Err(format!(
                "[initial_conditions]: duplicate init for compartment `{}`",
                name
            ));
        }
        let (amount_fn, amount_deriv, covariates_ref) = build_init_amount_fn(
            &expr,
            theta_names,
            eta_names,
            indiv_var_names,
            pk_indices,
            kappa_names,
        )?;
        cov_refs.extend(covariates_ref);
        out.push(crate::types::AnalyticalInit {
            cmt,
            amount_fn,
            amount_deriv,
        });
    }
    // Return unsorted: the caller re-sorts via `register_referenced_covariates`.
    let init_covariates: Vec<String> = cov_refs.into_iter().collect();
    Ok((out, init_covariates))
}

/// Fold the `[odes]` model-time alias `T` / `t` into the same `Expression::Time`
/// node a bare `TIME` produces, so a `[scaling]` expression spells the built-in
/// the way `[odes]` does (issue #1028).
///
/// `[odes]` reserves `TIME`/`T`/`t` for the solver clock and rejects any state,
/// individual parameter, or intermediate that collides with them. `[scaling]`
/// parses with `fallback_covariate = true`, so `TIME` was already special-cased by
/// `parse_atom` but a bare `T` landed as `Covariate("T")` — a required data column
/// named `T` if the dataset happened to carry one, and otherwise (before the
/// undefined-name guard) a silent zero. Only `Covariate` leaves are folded:
/// a state or individual parameter genuinely named `T` resolves to `Variable("T")`
/// during the parse and keeps winning, matching the usual name-resolution
/// precedence (the ODE-model case can't arise — `[odes]` rejects that name).
///
/// Two further guards keep the fold from *taking* a name that means something
/// else. A `[scaling]` `y` covariate has been a required data column since #540,
/// so a dataset with a real column named `T` used to work and must keep working:
///
/// 1. a `T` / `t` listed in `declared_covariates` (the `[covariates]` block) is a
///    data column by explicit declaration and is left as `Covariate` — the same
///    precedence a bound state or individual parameter gets; and
/// 2. when the fold does fire it pushes a `parse_warnings` note naming that escape
///    hatch, so a model that meant the column — and did not declare it, which is
///    only a warning elsewhere — is told rather than silently re-pointed at the
///    clock. Spelling it `TIME` clears the warning.
fn rewrite_scaling_time_alias(
    expr: &mut Expression,
    context: &str,
    declared_covariates: &[String],
    parse_warnings: &mut Vec<String>,
) {
    let mut folded: Option<String> = None;
    visit_expr_nodes_mut(expr, &mut |e: &mut Expression| {
        if let Expression::Covariate(name) = e {
            if is_time_alias(name) && !declares_time_alias(declared_covariates) {
                if folded.is_none() {
                    folded = Some(name.clone());
                }
                *e = Expression::Time;
            }
        }
    });
    if let Some(name) = folded {
        parse_warnings.push(format!(
            "{context}: `{name}` resolved as the model-time built-in (the `[odes]` alias \
             for `TIME`), not as a data column. Spell it `TIME` to silence this; if your \
             dataset really has a column named `{name}`, declare it in `[covariates]` and \
             it will be read as the column instead. See issue #1028."
        ));
    }
}

/// Run [`rewrite_scaling_time_alias`] on a *throwaway* parse of `src` purely to
/// collect the warning it would emit, discarding the rewritten expression.
///
/// For `[adaptive_dosing] observe`, which is stored as a raw string and compiled at
/// simulate time — where `parse_warnings` is long sealed and the adaptive result has
/// no warnings channel — this is what keeps the fold from being silent on that path.
/// The expression is re-parsed rather than duplicating the alias test, so the note
/// can never disagree with the fold that actually happens (#1028). A parse failure is
/// ignored: the real compile reports it, with its own context.
#[allow(clippy::too_many_arguments)]
fn warn_if_time_alias_folds(
    src: &str,
    context: &str,
    theta_names: &[String],
    eta_names: &[String],
    indiv_var_names: &[String],
    declared_covariates: &[String],
    parse_warnings: &mut Vec<String>,
) {
    let ctx = ParseCtx::new(theta_names, eta_names, indiv_var_names);
    if let Ok(mut expr) = parse_scalar_expression(src, ctx) {
        rewrite_scaling_time_alias(&mut expr, context, declared_covariates, parse_warnings);
    }
}

/// Whether `expr` reads the model-time built-in, under either spelling: the
/// `Expression::Time` node a bare `TIME` parses to, or the `T` / `t` alias
/// [`rewrite_scaling_time_alias`] folds into it. Drives the `obs_scale`
/// rejection (#1028) — see [`build_obs_scale_spec`].
///
/// `declared_covariates` gets the same precedence it gets in the fold: a `T`
/// declared in `[covariates]` is a data column, not the clock, so it does not
/// trip the rejection.
fn expr_references_time_builtin(expr: &Expression, declared_covariates: &[String]) -> bool {
    let mut found = false;
    visit_expr_nodes(expr, &mut |e: &Expression| match e {
        Expression::Time => found = true,
        Expression::Covariate(n)
            if is_time_alias(n) && !declares_time_alias(declared_covariates) =>
        {
            found = true
        }
        _ => {}
    });
    found
}

/// Whether `name` is the `[odes]` model-time alias, i.e. `T` under either case.
/// The two spellings are one built-in, so every guard that keys on the alias must
/// treat them as one name.
fn is_time_alias(name: &str) -> bool {
    name == "T" || name == "t"
}

/// Whether `[covariates]` declares the model-time alias, in *either* case.
///
/// Deliberately case-insensitive while ordinary covariate resolution is
/// case-sensitive: since [`is_time_alias`] collapses `T` and `t` into one
/// built-in, a case-exact check would let `[covariates] T` fail to protect a
/// `y = ... t ...` reference (silently folding it to the clock even though the
/// user took the documented escape hatch) and, in `obs_scale`, would raise the
/// `TIME`-rejection error against a legitimately declared column. Matching the
/// declaration the same way the reference is matched keeps the escape hatch
/// working for both spellings (#1028).
fn declares_time_alias(declared_covariates: &[String]) -> bool {
    declared_covariates
        .iter()
        .any(|d| d.eq_ignore_ascii_case("T"))
}

/// Build an `OdeOutputFn` from one `y[…] = value` line. Shared between
/// the uniform and per-CMT paths.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_y_output_fn(
    value: &str,
    context: &str,
    theta_names: &[String],
    eta_names: &[String],
    indiv_var_names: &[String],
    pk_indices: &[usize],
    state_names: &[String],
    kappa_names: &[String],
    readout_synth: &[ReadoutSynthParam],
    forbidden_state_names: &[String],
    declared_covariates: &[String],
    intermediates: &[(String, String)],
    parse_warnings: &mut Vec<String>,
) -> Result<(crate::ode::OdeOutputFn, OdeOutputProgram, Vec<String>), String> {
    // Form C: expression may reference state names, individual params,
    // thetas, etas, and covariates. ParseCtx::new + theta/eta in scope.
    let mut defined: Vec<String> = state_names.to_vec();
    for n in indiv_var_names {
        if !defined.iter().any(|d| d == n) {
            defined.push(n.clone());
        }
    }
    let ctx = ParseCtx::new(theta_names, eta_names, &defined);
    let mut expr = parse_scalar_expression(value, ctx).map_err(|e| format!("{context}: {e}"))?;

    // Named intermediates are inlined first, so every check and rewrite below —
    // the `T` fold, the forbidden-compartment and kappa rejections, the #486
    // desugaring, the covariate scan — runs on the fully expanded readout (#1030).
    inline_scaling_intermediates(&mut expr, intermediates, ctx)?;

    // `T` / `t` is the `[odes]` spelling of the model-time built-in; fold it into
    // the `Expression::Time` node a bare `TIME` already parses to, so the two
    // blocks agree on the name (#1028). Runs before the covariate scan below so
    // the alias never registers as a required data column — unless `[covariates]`
    // declares it, in which case it stays the data column it was declared to be.
    rewrite_scaling_time_alias(&mut expr, context, declared_covariates, parse_warnings);

    // Analytic Form C (#650): reject a readout that references a peripheral (or a
    // depot/transit amount with no closed form) — the analytical solutions don't
    // expose those amounts with cross-compartment sensitivity. Empty for ODE
    // callers (any state name is integrated), so this is a no-op there. Point the
    // user at an ODE model, mirroring `[initial_conditions]`'s scope.
    if let Some(name) = expr_references_any(&expr, forbidden_state_names) {
        return Err(format!(
            "{context}: compartment `{name}` is not available in an analytic Form C \
             readout — the closed forms don't expose a peripheral (or transit/IV depot) \
             amount with cross-compartment sensitivity. Reference the central compartment \
             amount (`central`, plus the oral `depot` for first-order oral models), or use \
             an ODE model with `ode(states=[...])` in [odes]. See issue #650."
        ));
    }
    // #486: rewrite the bare θ/η references the parser desugared into synthetic
    // individual parameters (`__ferx_ro_*`, already appended to `indiv_var_names`)
    // back to `Variable(synthetic_name)`. The readout then compiles to `Op::PushVar`
    // and its θ/η dependence rides the individual-parameter sensitivity chain — so the
    // analytic provider serves it instead of dropping the subject to FD. Any θ/η left
    // un-desugared (none, in practice) keeps `dual_evaluable` false → FD fallback.
    rewrite_readout_synth(&mut expr, readout_synth);

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
            "{context}: Form C output expressions cannot reference the IOV \
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
    resolve_expr_indices(&mut expr, &var_idx, &cov_idx);
    let bc = compile_bytecode(&expr);

    // Snapshot the readout as an `OdeOutputProgram` for the analytic-sensitivity
    // path (issue #367): same bytecode + layout, evaluated over `Dual2<N>`. It is
    // dual-evaluable when the expression references only states / individual
    // parameters / covariates / constants — covariates thread in as constants
    // from the per-observation snapshot (#540), and a direct θ/η reference is
    // *normally* desugared above into a synthetic individual parameter (#486).
    //
    // The `PushTheta`/`PushEta` arms below are still load-bearing and must not be
    // removed: when the synthetic readout params overflow the fixed PK-slot layout,
    // the parser clears `readout_synth_params` and leaves the bare θ/η ops in the
    // bytecode (the documented FD fallback). Those arms are what keep `dual_evaluable`
    // `false` on that path; dropping them would wrongly route a slot-overflowed
    // direct-θ/η readout onto the analytic walk. `PushNnOutput` is the other
    // disqualifier (an NN output is never desugared).
    let output_dual_evaluable = !bc.ops.iter().any(|op| {
        matches!(
            op,
            Op::PushTheta(_) | Op::PushEta(_) | Op::PushNnOutput(_, _)
        )
    });
    let output_program = OdeOutputProgram {
        bc: bc.clone(),
        n_states,
        n_indiv,
        indiv_to_pk: indiv_to_pk.clone(),
        cov_names: cov_names.clone(),
        dual_evaluable: output_dual_evaluable,
    };

    // Per-thread scratch for the y readout vars + covariate slice + bytecode
    // f64 stack comes from the shared `FERX_SCRATCH` (see `FerxThreadScratch`).
    // The y-readout fields are kept separate from rhs_vars because the two are
    // sized for different layouts; the bc_stack is shared because the y readout
    // and ODE RHS never interleave on a single thread (readout runs
    // post-integration).
    let n_cov = cov_names.len();
    let cov_names_for_fn = cov_names.clone();

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
                    for (i, name) in cov_names_for_fn.iter().enumerate() {
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
    Ok((out_fn, output_program, cov_names))
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
/// - The trailing `Vec<String>` is every covariate name referenced across the
///   block — **both** the Form C `y` readouts and the Form B `obs_scale`
///   expressions (#1028). The caller registers them via
///   `register_referenced_covariates`, which is what makes an identifier the
///   parse could not bind to a theta / eta / individual parameter / state a
///   *required data column* rather than a silent `0.0`.
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
/// - `obs_scale` referencing the `TIME` built-in (or its `T` alias) → error: the
///   divisor is subject-static, so `TIME` would always read the t=0 default
///   (#1028). Form C `y = <expr>` is where a time-dependent readout belongs.
/// - `obs_scale` referencing the same individual parameter bound to a built-in
///   `pk <model>(...)` block's `v`/`v1` role → **warning** (`parse_warnings`),
///   not an error: it's a supported feature, but also the signature of a
///   common mistake (a leftover `obs_scale = V` from an `ode(...)`
///   translation) — see [`build_obs_scale_spec`] (#712).
///
/// `declared_covariates` is the `[covariates]` block's name list (empty when the
/// block is absent). It only affects name *precedence*: a `T` / `t` declared there
/// is a data column and is read as one, instead of being folded into the
/// model-time built-in (#1028).
#[allow(clippy::too_many_arguments)]
fn parse_scaling_block(
    lines: &[String],
    theta_names: &[String],
    eta_names: &[String],
    indiv_var_names: &[String],
    pk_indices: &[usize],
    state_names: &[String],
    is_ode: bool,
    pk_model: PkModel,
    kappa_names: &[String],
    readout_synth: &[ReadoutSynthParam],
    volume_indiv_name: Option<&str>,
    declared_covariates: &[String],
    parse_warnings: &mut Vec<String>,
) -> Result<
    (
        ScalingSpec,
        Option<crate::ode::OdeReadout>,
        Option<OdeOutputProgram>,
        Vec<String>,
    ),
    String,
> {
    // Accumulate uniform and per-CMT entries separately, then assemble at
    // the end. Mixing the two forms within the same group (obs_scale or y)
    // is rejected — keeps the semantic clean and matches NONMEM's
    // explicit-S1/S2 discipline.
    let mut obs_scale_uniform: Option<ScalingSpec> = None;
    let mut obs_scale_per_cmt: HashMap<usize, ScalingSpec> = HashMap::new();
    let mut y_uniform: Option<crate::ode::OdeOutputFn> = None;
    let mut y_per_cmt: HashMap<usize, crate::ode::PerCmtReadout> = HashMap::new();
    // Sensitivity program for the uniform `y = <expr>` readout (issue #367). The
    // per-CMT readouts carry their own program inside each `PerCmtReadout` (#439),
    // so the analytic-sensitivity provider can differentiate each endpoint.
    let mut y_uniform_program: Option<OdeOutputProgram> = None;
    let mut scaling_covariates: Vec<String> = Vec::new();

    // Named intermediates (#1030): every non-`obs_scale`/`y` key. Collected up
    // front — this is the one place with the full name scope in hand, so it owns
    // the shadowing rejection the pre-scans can't make.
    let intermediates = scaling_intermediates(lines)?;
    for (name, _) in &intermediates {
        // An intermediate whose name is already a θ / η / individual parameter /
        // state would never be read: the expression parser binds those before it
        // consults the intermediate table, so the binding would be silently dead.
        //
        // A declared covariate clashes the other way round — an unresolved
        // identifier in `[scaling]` parses as `Covariate(name)`, which the
        // substituter *does* rewrite, so the binding would silently shadow the data
        // column and drop it from the required-column set (#1030). Rejected either
        // way: one name, one meaning.
        let clash = theta_names
            .iter()
            .map(|n| (n, "a theta"))
            .chain(eta_names.iter().map(|n| (n, "an eta")))
            .chain(
                indiv_var_names
                    .iter()
                    .map(|n| (n, "an individual parameter")),
            )
            .chain(state_names.iter().map(|n| (n, "a compartment")))
            .chain(
                declared_covariates
                    .iter()
                    .map(|n| (n, "a declared covariate")),
            )
            .find(|(n, _)| *n == name);
        if let Some((_, what)) = clash {
            return Err(format!(
                "[scaling]: `{name}` is already {what}, so it cannot also be a named \
                 intermediate — one name cannot mean two things here. Rename the \
                 intermediate."
            ));
        }
    }
    check_scaling_intermediates_used(lines, &intermediates)?;

    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Split on the first `=` OUTSIDE any `[...]` bracket (a naive
        // `splitn(2, '=')` would split inside `obs_scale[CMT=1]`). Shared with the
        // #486 readout θ/η pre-scan (`collect_readout_theta_eta_synth`).
        let (key, value) = split_scaling_entry(trimmed)?;
        let (base, cmt_opt) = parse_scaling_key(key)?;

        match base {
            "obs_scale" => {
                let (spec, cov_names) = build_obs_scale_spec(
                    value,
                    theta_names,
                    eta_names,
                    indiv_var_names,
                    pk_indices,
                    volume_indiv_name,
                    declared_covariates,
                    &intermediates,
                    parse_warnings,
                )?;
                // Form B `obs_scale` covariate leaves are required data columns too
                // (#1028) — pooled into the same list the Form C `y` readout feeds,
                // so both halves of `[scaling]` reach `register_referenced_covariates`.
                for cov in cov_names {
                    if !scaling_covariates.contains(&cov) {
                        scaling_covariates.push(cov);
                    }
                }
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
                // Form C readout. On ODE models the state names come from
                // `ode(states=[...])` and any state is integrable (no forbidden
                // set). On analytical models (#650) we synthesize the canonical
                // compartment names the closed forms expose (`central`, plus the
                // oral `depot`) and forbid peripheral / no-closed-form amounts.
                let (y_state_names, forbidden): (Vec<String>, Vec<String>) = if is_ode {
                    (state_names.to_vec(), Vec::new())
                } else {
                    analytic_readout_state_names(pk_model)
                };
                let (out_fn, out_program, cov_names) = build_y_output_fn(
                    value,
                    "[scaling] y",
                    theta_names,
                    eta_names,
                    indiv_var_names,
                    pk_indices,
                    &y_state_names,
                    kappa_names,
                    readout_synth,
                    &forbidden,
                    declared_covariates,
                    &intermediates,
                    parse_warnings,
                )?;
                for cov in cov_names {
                    if !scaling_covariates.contains(&cov) {
                        scaling_covariates.push(cov);
                    }
                }
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
                        y_uniform_program = Some(out_program);
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
                        y_per_cmt.insert(
                            cmt,
                            crate::ode::PerCmtReadout {
                                out_fn,
                                program: Some(out_program),
                            },
                        );
                    }
                }
            }
            // A named intermediate (#1030). Already collected and validated above,
            // and inlined into the entries that reference it — nothing left to do
            // on its own line.
            _ => {}
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
    // The output program accompanies the uniform `Single` readout only.
    let readout_program = if readout.is_some() {
        y_uniform_program
    } else {
        None
    };

    // Analytic Form C replaces the built-in concentration readout; a divisive
    // `obs_scale` alongside it would double-apply on the same output. Reject the
    // combination on analytical models so intent is explicit (#650). ODE keeps
    // its historical behaviour.
    if !is_ode && readout.is_some() && !matches!(scaling, ScalingSpec::None) {
        return Err(
            "[scaling]: cannot combine a Form C `y = <expr>` readout with a \
             divisive `obs_scale` on an analytical model — the readout already replaces \
             the built-in concentration output. Fold any unit conversion into `y` \
             (e.g. `y = central / V / 1000`)."
                .into(),
        );
    }

    scaling_covariates.sort();
    Ok((scaling, readout, readout_program, scaling_covariates))
}

// ── ode_template desugaring + the analytical+ODE-only-absorption error rule ──

/// Built-in absorption input-rate functions that — **as currently implemented** —
/// require an ODE disposition, so combining them with an analytical `pk ...` is a
/// hard error pointing at `ode_template` (the error rule, #322 Phase 0b). This is
/// about what the engine can run *today*, not which models are closed-form-capable
/// in principle: it lists only the functions actually implemented as input rates
/// (`transit`, #343; `igd`, #347; `weibull`, Phase 2; `zero_order`, #504), so the
/// error rule never advertises a function the engine can't yet run nor silently
/// drops an `[odes]` absorption term on an analytical model. A slice (not a
/// fixed-size array) so a new entry needs no length-annotation bump, matching
/// [`INPUT_RATE_FNS`].
///
/// `weibull` stays here **permanently** — it has no elementary closed form with
/// linear disposition, so it always requires an explicit ODE disposition.
/// `transit`/`igd` will leave the list once their analytical (incomplete-gamma /
/// tilted-IG) closed forms land (#386 / Phase 3). `zero_order` and `first_order`
/// are both closed-form-able (a zero-order infusion / a Bateman superposition),
/// but #504 / #505 ship them on the **ODE forcing path only**; they leave the list
/// when that closed-form acceleration is added under the equivalence harness.
/// Until then, listing them keeps the no-surprises contract (a `pk ... +
/// zero_order(...)` / `pk ... + first_order(...)` model errors loudly instead of
/// dropping the absorption).
const ODE_ONLY_ABSORPTION_FNS: &[&str] =
    &["transit", "igd", "weibull", "zero_order", "first_order"];

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

/// The reserved F / lagtime PK slot a parameter or `pk(...)` role name routes to in the ODE
/// twin — `Some(PK_IDX_F)` / `Some(PK_IDX_LAGTIME)` — or `None` if it routes to a disposition /
/// structural slot. Derived from the real routing (`PkParams::name_to_index` filtered by
/// `RESERVED_PK_SLOTS`) so the twin-builder's F/lagtime allowlist, alias emission, and
/// shadow guard can never drift from `ode_param_slots`. #735.
fn reserved_flag_slot(name: &str) -> Option<usize> {
    crate::types::PkParams::name_to_index(&name.to_lowercase())
        .filter(|s| crate::types::RESERVED_PK_SLOTS.contains(s))
}

/// For a closed-form `pk one_cpt_transit(cl, v, n, mtt)` / `pk one_cpt_ig(cl, v, mat, cv2)`
/// model (and the 2-cpt forms), build the full `.ferx` source of its exact ODE equivalent
/// (`transit()` / `igd()` forcing) — otherwise `None`.
///
/// The transit / IG closed forms assume constant parameters over each absorption window, so
/// they cannot serve a subject whose parameters switch mid-profile (a `TIME`-dependent
/// structural parameter, or time-varying covariates). The ODE twin
/// `d/dt(central) = <forcing> − (CL/V)·central` with `obs_scale = V` (`<forcing>` =
/// `transit(n, mtt)` / `igd(mat, cv2)`) is the exact numerical equivalent (validated in
/// `tests/transit_analytic_equivalence.rs` / `tests/ig_analytic_equivalence.rs`) and carries
/// those per-event through the event-driven walk (analytically for `TIME` since #664). The
/// parser compiles this source into a sub-model stored on
/// [`CompiledModel::absorption_ode_equivalent`]; the runtime dispatch
/// ([`CompiledModel::effective_for`]) uses it only for the subjects that need it, so a plain
/// transit / IG fit keeps its fast, exact closed form (#486, #790).
///
/// The equivalent shares the model's `[parameters]`/`[individual_parameters]`/… blocks
/// verbatim and swaps only the disposition, so its θ/η layout matches the closed-form model
/// exactly (the dispatch can hand it the same parameter vector). Scoped to the plain
/// `one_cpt_transit(cl,v,n,mtt)` / `one_cpt_ig(cl,v,mat,cv2)` (and 2-cpt) forms, optionally
/// with a `lagtime=`/`f=` (`alag=`) mapping carried into the twin as a reserved-name alias
/// (#735), and with no user `[odes]`/`[scaling]`/`[initial_conditions]` block. Anything else —
/// an unknown role, a parameter name that shadows or collides with a reserved F/lagtime slot,
/// or a user disposition block — returns `None` (and stays closed-form, still rejected up front
/// for the features it cannot serve).
fn absorption_ode_equivalent_source(extracted: &ExtractedBlocks) -> Option<String> {
    // Match a plain closed-form transit or IG disposition — 1-cpt `cl/v/<abs>` or 2-cpt
    // `cl/v1/q/v2/<abs>`, where `<abs>` is `n/mtt` (transit) or `mat/cv2` (IG). Any other
    // structural form stays closed-form.
    let structural = extracted.unnamed.get("structural_model")?;
    // `LazyLock` rather than per-call `Regex::new`: since #1008 the twin is compiled eagerly, so
    // every transit/IG model reaches this function twice — once for itself, once from the twin's
    // own parse, where all four patterns are compiled only to miss (the twin's structural model
    // is `ode(...)`). Compiling them per call cost ~0.9 ms of the parse, measured on #1027.
    static TRANSIT_ONE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"^pk\s+one_cpt_transit\s*\(([^)]*)\)\s*$").unwrap());
    static TRANSIT_TWO: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"^pk\s+two_cpt_transit\s*\(([^)]*)\)\s*$").unwrap());
    static IG_ONE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"^pk\s+one_cpt_ig\s*\(([^)]*)\)\s*$").unwrap());
    static IG_TWO: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"^pk\s+two_cpt_ig\s*\(([^)]*)\)\s*$").unwrap());
    // (args, is_two_cpt, is_ig, pk_label)
    let (args_str, is_two_cpt, is_ig, pk_label) = if let Some(caps) = structural
        .iter()
        .find_map(|l| TRANSIT_ONE.captures(l.trim()))
    {
        (
            caps.get(1)?.as_str().to_string(),
            false,
            false,
            "pk one_cpt_transit",
        )
    } else if let Some(caps) = structural
        .iter()
        .find_map(|l| TRANSIT_TWO.captures(l.trim()))
    {
        (
            caps.get(1)?.as_str().to_string(),
            true,
            false,
            "pk two_cpt_transit",
        )
    } else if let Some(caps) = structural.iter().find_map(|l| IG_ONE.captures(l.trim())) {
        (
            caps.get(1)?.as_str().to_string(),
            false,
            true,
            "pk one_cpt_ig",
        )
    } else if let Some(caps) = structural.iter().find_map(|l| IG_TWO.captures(l.trim())) {
        (
            caps.get(1)?.as_str().to_string(),
            true,
            true,
            "pk two_cpt_ig",
        )
    } else {
        return None;
    };
    // A user-written [odes]/[scaling] block is outside the plain-form scope. An
    // [initial_conditions] block is too: the equivalent's disposition states (central,
    // and periph for 2-cpt) cannot carry an absorption `depot` seed, and absorption-init has
    // its own parse-time validation on the primary model — building the equivalent would
    // pre-empt that with a confusing error.
    if extracted.unnamed.contains_key("odes")
        || extracted.unnamed.contains_key("scaling")
        || extracted.unnamed.contains_key("initial_conditions")
    {
        return None;
    }
    let roles = parse_role_pairs(&args_str, pk_label).ok()?;
    let disposition: &[&str] = match (is_two_cpt, is_ig) {
        (false, false) => &["cl", "v", "n", "mtt"],
        (true, false) => &["cl", "v1", "q", "v2", "n", "mtt"],
        (false, true) => &["cl", "v", "mat", "cv2"],
        (true, true) => &["cl", "v1", "q", "v2", "mat", "cv2"],
    };
    // #735: absorption `f=` / `lagtime=` (`alag=`) are now carried into the twin (alias
    // emission below); any *other* unrecognised role stays outside the plain-form scope.
    // `reserved_flag_slot` recognises exactly the role names that route to the F/lagtime slots
    // (derived from the real routing, so this can never drift from `ode_param_slots`).
    if roles
        .keys()
        .any(|k| !disposition.contains(&k.as_str()) && reserved_flag_slot(k).is_none())
    {
        return None; // an unknown mapping — outside the plain-form scope.
    }
    // The ODE twin routes F / lagtime by an individual parameter *named* `f` / `lagtime`
    // / `alag` (`ode_param_slots`), whereas the closed form binds them via this pk() role
    // map under arbitrary names. Bridge with reserved-name alias lines — but only when the
    // mapped parameter does not already self-route (a param literally named `F` / `LAGTIME`
    // / `ALAG` routes to the canonical slot already, so an alias would double-bind it and
    // `ode_param_slots` would reject). #735.
    let mut alias_lines: Vec<String> = Vec::new();
    let mut alias_names: Vec<&str> = Vec::new();
    if let Some(fp) = roles.get("f") {
        if reserved_flag_slot(fp) != Some(crate::types::PK_IDX_F) {
            alias_lines.push(format!("  f = {fp}"));
            alias_names.push("f");
        }
    }
    if let Some(lp) = roles.get("lagtime").or_else(|| roles.get("alag")) {
        if reserved_flag_slot(lp) != Some(crate::types::PK_IDX_LAGTIME) {
            alias_lines.push(format!("  lagtime = {lp}"));
            alias_names.push("lagtime");
        }
    }
    // Silent-divergence guard (#735): the ODE twin routes *any* individual parameter whose
    // name routes to a reserved slot (`f` / `lagtime` / `alag`) into that F/lagtime slot
    // (`ode_param_slots`), even one the closed form never treated as F/lag. A param whose name
    // routes to a reserved slot is allowed only when it IS the intended mapping for *that
    // specific* slot: a param named `f` must be the `f=` mapping; a param named `lagtime`/`alag`
    // must be the `lagtime=`/`alag=` mapping. Anything else declines the twin — a disposition
    // role bound to such a param (`mtt=ALAG`), a stray covariate/derived param literally named
    // `f`, OR a cross-role mapping whose param name routes to a *different* reserved slot than
    // its role (`f=LAGTIME`: the `LAGTIME` param name-routes to the lagtime slot the closed form
    // never used it for, so the twin would apply it as both F and lag). Declining keeps the model
    // closed-form (matching the closed form, which never applied that param as the shadowed
    // F/lag) and, if it is also flip-flop, rejected up front — rather than the twin silently
    // applying an extra F/lagtime the closed form did not.
    //
    // The same pass records the twin's individual-parameter names for the collision guard below.
    let f_param = roles.get("f").map(String::as_str);
    let lag_param = roles
        .get("lagtime")
        .or_else(|| roles.get("alag"))
        .map(String::as_str);
    // #993 companion to the guard above. That one asks "is this reserved-name param the
    // intended mapping for its slot?"; it never asks whether the same param *also* fills a
    // disposition role. `pk one_cpt_transit(cl=CL, v=F, n=N, mtt=MTT, f=F)` passes it — `F`
    // is the `f=` mapping — but the twin then emits `d/dt(central) = … − (CL/F) * central`
    // and `obs_scale = F`, reading a name `ode_param_slots` routes to the F slot. That is a
    // dose-attribute double use, so the twin's own parse rejects it. Decline here, per this
    // function's standing policy: keep the model closed-form. (Since #1008 an unguarded case
    // like this no longer panics — the attach site declines the twin and warns — but naming
    // the case here keeps the *reason* for the decline specific instead of generic.)
    if disposition
        .iter()
        .filter_map(|role| roles.get(*role))
        .any(|p| Some(p.as_str()) == f_param || Some(p.as_str()) == lag_param)
    {
        return None;
    }
    let mut twin_param_names: Vec<String> = Vec::new();
    if let Some(ip_lines) = extracted.unnamed.get("individual_parameters") {
        for line in ip_lines {
            // `extract_blocks` has already stripped `#`/`//` comments, so the LHS is the text
            // before the first `=`; the alphanumeric check skips `if`/`else`/brace lines.
            let lhs = line.split('=').next().unwrap_or("").trim();
            if lhs.is_empty() || !lhs.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                continue;
            }
            let shadows = match reserved_flag_slot(lhs) {
                Some(slot) if slot == crate::types::PK_IDX_F => f_param != Some(lhs),
                Some(_) => lag_param != Some(lhs), // PK_IDX_LAGTIME
                None => false,
            };
            if shadows {
                return None; // shadows a reserved F/lagtime slot — decline (see above).
            }
            twin_param_names.push(lhs.to_string());
        }
    }
    // Collision guard (#735): a `f=`/`lagtime=` value that is an individual parameter whose
    // *name* routes to an already-taken *disposition* slot (e.g. `f=V1`, where `V1` name-routes
    // to the V slot held by `v=V`) is not a reserved-name shadow, so it passes the guard above —
    // but the generated twin's `ode_param_slots` would reject it as two parameters mapping to one
    // slot (which, before #1008, `AbsorptionOdeEquivalent::get_or_build` surfaced as a *panic*).
    // The closed form binds F/lagtime by role, so the collision is harmless there. Dry-run the real slot
    // assignment on the twin's projected parameter names. This list is a superset of the twin's
    // real (filtered) parameter set, so it catches every real collision — at worst it declines an
    // *unused* colliding name the real twin would not hit, which is harmless (the model stays
    // closed-form). If it fails, decline the twin so the model stays closed-form — and, if
    // flip-flop, is rejected up front with a clean error rather than panicking at eval time.
    twin_param_names.extend(alias_names.iter().map(|s| s.to_string()));
    // A parameter assigned in several `if`/`else` branches appears once in the twin's real
    // parameter list (`unconditionally_assigned_vars` de-duplicates), so counting each raw LHS
    // occurrence would be a spurious self-collision (`CL` in both branches of a `TIME` switch).
    // De-duplicate by exact name, preserving first-occurrence order, before the slot check.
    let mut seen = std::collections::HashSet::new();
    twin_param_names.retain(|n| seen.insert(n.clone()));
    if ode_param_slots(&twin_param_names).is_err() {
        return None;
    }
    // The absorption forcing term, `transit(n, mtt)` or `igd(mat, cv2)`.
    let forcing = if is_ig {
        format!("igd(mat={}, cv2={})", roles.get("mat")?, roles.get("cv2")?)
    } else {
        format!("transit(n={}, mtt={})", roles.get("n")?, roles.get("mtt")?)
    };
    // Re-emit every block verbatim except the disposition (deterministic order), then append
    // the ODE twin. Parsing this source yields a normal ODE model — it contains no
    // `pk one_cpt_transit`/`pk two_cpt_transit`/`pk one_cpt_ig`/`pk two_cpt_ig`, so it does
    // not recurse into this desugar.
    //
    // `adaptive_dosing` is dropped rather than carried (#993). Two independent reasons.
    // First, it would cost the model its twin for no gain: the twin *is* an ODE model, so a
    // re-emitted `observe` that reads a dose attribute hits `check_dose_attr_double_use` and
    // the twin's parse fails. (Before #1008 that failure was a panic mid-fit, through
    // `get_or_build`'s `.expect()`, on the plain `predict`/`fit` path that reroutes
    // TV-covariate / `TIME` / IOV subjects here; it now declines the twin at parse time,
    // which is survivable but still needlessly loses the fallback.) Blocks that would
    // otherwise reach an ODE-scoped check (`odes`, `scaling`) decline the twin outright
    // above; `adaptive_dosing` does not, so it has to be dropped here. Since #1004 the
    // analytical primary rejects that shape itself, so the twin can no longer be *reached*
    // with it — which is why `adaptive_observe_does_not_reach_the_absorption_twin` asserts
    // on the re-emitted source directly. "The twin still builds" no longer implies the drop
    // once no reachable controller carries something the twin's parse would reject.
    // Second, the twin is reached only through `CompiledModel::effective_for` on the
    // prediction path (`pk/mod.rs`), and nothing there reads a controller spec —
    // `simulate_adaptive_from_spec` takes the spec as an argument from the primary — so
    // carrying it would only give the twin a stale block it can never act on.
    let mut src = String::new();
    let mut names: Vec<&String> = extracted.unnamed.keys().collect();
    names.sort();
    for name in names {
        if matches!(
            name.as_str(),
            "structural_model" | "odes" | "scaling" | "adaptive_dosing"
        ) {
            continue;
        }
        src.push_str(&format!("[{name}]\n"));
        for line in &extracted.unnamed[name] {
            src.push_str(line);
            src.push('\n');
        }
        // #735: append the reserved-name F/lagtime aliases to [individual_parameters]
        // so the ODE twin applies them (the closed form's pk() role map has no ODE
        // analogue — see `ode_param_slots`).
        if name == "individual_parameters" {
            for alias in &alias_lines {
                src.push_str(alias);
                src.push('\n');
            }
        }
        src.push('\n');
    }
    let mut named: Vec<(&String, &String, &Vec<String>)> = extracted
        .named
        .iter()
        .flat_map(|(block, insts)| insts.iter().map(move |(inst, lines)| (block, inst, lines)))
        .collect();
    named.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    for (block, inst, lines) in named {
        src.push_str(&format!("[{block} {inst}]\n"));
        for line in lines {
            src.push_str(line);
            src.push('\n');
        }
        src.push('\n');
    }
    // Disposition twin. The micro-constant flux mirrors `ode_template::generate`'s
    // matching case exactly — 1-cpt: `ke = cl/v`; 2-cpt: `k10 = cl/v1`, `k12 = q/v1`,
    // `k21 = q/v2`, observed on `central` — so the ODE twin describes the same linear
    // system as the exponential-tilting closed form (pinned by the closed-form↔ODE
    // equivalence tests, `tests/transit_analytic_equivalence.rs` / `ig_analytic_equivalence.rs`).
    // `{forcing}` is the absorption input (`transit(n, mtt)` / `igd(mat, cv2)`).
    if is_two_cpt {
        let (cl, v1, q, v2) = (
            roles.get("cl")?,
            roles.get("v1")?,
            roles.get("q")?,
            roles.get("v2")?,
        );
        src.push_str("[structural_model]\n  ode(obs_cmt=central, states=[central, periph])\n\n");
        src.push_str(&format!(
            "[odes]\n  d/dt(central) = {forcing} - ({cl}/{v1} + {q}/{v1}) * central + ({q}/{v2}) * periph\n  d/dt(periph) = ({q}/{v1}) * central - ({q}/{v2}) * periph\n\n"
        ));
        src.push_str(&format!("[scaling]\n  obs_scale = {v1}\n\n"));
    } else {
        let (cl, v) = (roles.get("cl")?, roles.get("v")?);
        src.push_str("[structural_model]\n  ode(obs_cmt=central, states=[central])\n\n");
        src.push_str(&format!(
            "[odes]\n  d/dt(central) = {forcing} - ({cl}/{v}) * central\n\n"
        ));
        src.push_str(&format!("[scaling]\n  obs_scale = {v}\n\n"));
    }
    Some(src)
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

/// Number of additional individual parameters that can still be assigned a
/// non-reserved PK slot, given the already-declared `names`. Returns 0 if `names`
/// itself does not fit (the real error then surfaces in `ode_param_slots`).
/// Reuses `ode_param_slots` so the headroom can never drift from the real
/// assignment. Used by the #486 readout desugaring to fall back to FD rather than
/// fail a model that previously parsed, when synthetic readout params would
/// overflow the layout.
fn ode_free_slot_count(names: &[String]) -> usize {
    let slots = match ode_param_slots(names) {
        Ok(s) => s,
        Err(_) => return 0,
    };
    let mut taken = [false; MAX_PK_PARAMS];
    for s in slots {
        taken[s] = true;
    }
    (0..MAX_PK_PARAMS)
        .filter(|s| !taken[*s] && !RESERVED_PK_SLOTS.contains(s))
        .count()
}

/// Reject an individual parameter that the engine applies to the **dose** and the
/// model *also* reads on the **prediction path** — its value is then applied twice,
/// silently (#993).
///
/// `F` (and `LAGTIME`/`ALAG`, and the compartment-indexed `F{n}`/`ALAG{n}`) never
/// appear in the `[odes]` RHS or the `[scaling]` readout of a correct model: the
/// engine consumes them at the dose event. A model that reads one anyway gets it
/// applied once by the engine and once where it is read — measured at exactly `F`
/// on the prediction, and at `exp(LAGTIME·ke)` for lag. `docs/model-file/
/// ode-models.qmd` has said so in prose since the dose-entry migration; this is the
/// enforcement, so the legacy `F`-in-the-flux models that prose was written for
/// fail loudly instead of quietly computing `F²`.
///
/// Scoped to [`DoseAttrConsumption::EveryDose`]. `D{n}`/`R{n}` are consumed only by
/// a coded-`RATE` dose, so an `R1` that is really a rate constant is a correct model
/// on ordinary data and cannot be judged here — that pair is checked against the
/// dataset in `api::validation::check_modeled_dose_rates`.
///
/// `reads` holds the names actually referenced in `block`, upper-cased: the ODE
/// var-slot map aliases each parameter's case variants onto one slot, so a `f` in
/// the RHS reads the same value as `F` and must diagnose the same.
///
/// `n_states` bounds the compartment indices this check will claim. An `F{c}` with
/// `c` past the last state is a *different* defect, and the `dose_attr_map` build
/// below reports it precisely ("the model has only N compartment(s)"); telling such
/// a user to rename `F5` would send them after the wrong thing, so out-of-range
/// indices are left for that check.
///
/// Runs on **both engines** (#993 shipped the ODE half; #1004 the analytical).
/// `indiv_param_slots` is what differs: the ODE caller passes `ode_slot_map`
/// (name-based routing is what makes a bare `F` a dose attribute there), the
/// analytical caller passes [`analytical_dose_attr_slot_map`] (the explicit
/// `pk(..., f=F)` / `lagtime=` mapping is what binds it there — measured on
/// NONMEM's own analytical routine, ADVAN2 with `F1` defined and `S2 = V/F1`:
/// predictions shift by exactly `F1` with no diagnostic, see
/// `tests/dose_attr_double_use_nonmem_anchor.rs`). An analytical caller passes
/// `usize::MAX` for `n_states`: a compartment-indexed `EveryDose` attribute
/// cannot reach this check there — an *unmapped* `F{n}`/`ALAG{n}` is already a
/// parse error ("nothing binds it to the dose route"), and a mapped one resolves
/// through the bare-slot arm of `DoseAttrConsumption::of`, whose `cmt` is `None`.
///
/// `analytical_pk_map` is `None` for the ODE callers and `Some(pk_param_map)` for
/// the analytical ones. It selects the remediation clause (rename vs drop the
/// mapping) and, via [`analytical_role_binding`], supplies the `role=value` pair
/// **as the user spelled it**, so an `alag=` model is not told to look for a
/// `lagtime=` argument it never wrote.
///
/// Scope gap on the analytical side: the caller iterates `indiv_param_names`,
/// which does not carry a parameter assigned only inside an `if` body — while
/// `build_pk_param_fn` resolves `pk(..., f=F)` against those nested names too, so
/// such a parameter *is* applied at the dose and escapes this check. That model
/// has a wider defect anyway (#1026: reading the name off the prediction path
/// yields `NaN` with a clean parse); fixing the registration there closes this
/// hole with it.
///
/// `[initial_conditions]` is deliberately **not** a call site. An analytical init
/// amount is propagated by `pk::analytical_init_concentration_g` with `F = 1` and
/// no lag, so reading a mapped dose attribute in an init expression applies it
/// once, not twice — see the note at that block.
/// `read_site` names the surface the `reads` set was actually collected from, for the
/// "read in …" and "remove it from …" clauses. It is usually the same as `block`, but
/// not always: since #1046 only the **RHS** of `[odes]` is a prediction-path read —
/// an `init(...)` seed is not — so telling an `[odes]` user to "remove it from
/// `[odes]`" would send them to delete a read that is legal and whose removal does not
/// clear the error. `block` stays the `"{block}:"` prefix every block-scoped parse
/// error carries (and that `parse_error_to_diagnostic` / the tests key on).
fn check_dose_attr_double_use(
    indiv_param_names: &[String],
    indiv_param_slots: &[usize],
    reads: &std::collections::HashSet<String>,
    n_states: usize,
    block: &str,
    read_site: &str,
    analytical_pk_map: Option<&HashMap<String, String>>,
) -> Result<(), String> {
    for (i, name) in indiv_param_names.iter().enumerate() {
        let slot = indiv_param_slots.get(i).copied();
        let Some((attr, when, cmt)) = crate::types::DoseAttrConsumption::of(name, slot) else {
            continue;
        };
        if when != crate::types::DoseAttrConsumption::EveryDose {
            continue;
        }
        if cmt.is_some_and(|c| c > n_states) {
            continue;
        }
        if !reads.contains(&name.to_ascii_uppercase()) {
            continue;
        }
        let scope = match cmt {
            Some(c) => format!(" for doses into compartment {c}"),
            None => String::new(),
        };
        // The remedy for "it is meant to be an ordinary parameter" differs by
        // engine. On an ODE model the *name* is what routes a bare `F`/`LAGTIME`
        // (or an indexed `F{n}`/`ALAG{n}`) onto the dose, so renaming un-routes
        // it. On an analytical model the name is inert; the explicit
        // `pk(..., f=NAME)` / `lagtime=NAME` mapping is what binds it, so the
        // mapping — not the name — is what has to go (#1004; renaming alone
        // changes nothing, the mapping follows the parameter).
        //
        // The analytical clause quotes the mapping **as the user spelled it** —
        // `alag=` is a legal alias for `lagtime=`, and the mapped value may differ
        // in case from the declaration (`alag=TLAG` binding a declared `tlag`
        // through the lowercase compat lookup). Naming an argument the model file
        // does not contain sends the reader looking for the wrong text, so the
        // role/value pair comes from the entry that actually did the binding.
        let (ordinary_remedy, issue) = match analytical_pk_map {
            Some(map) => {
                let (role, bound) = analytical_role_binding(map, name, attr)
                    .map(|(k, v)| (k.clone(), v.clone()))
                    // Unreachable: `analytical_dose_attr_slot_map` set this
                    // parameter's slot from exactly such an entry, under the same
                    // resolution. Name the canonical role rather than panicking if
                    // that ever drifts.
                    .unwrap_or_else(|| {
                        let role = match attr {
                            crate::types::DoseAttr::F => "f",
                            _ => "lagtime",
                        };
                        (role.to_string(), name.to_string())
                    });
                (
                    format!(
                        "remove the `{role}={bound}` mapping from the `pk(...)` call — the \
                         mapping, not the name, is what makes `{name}` a dose attribute"
                    ),
                    "#1004",
                )
            }
            None => (
                format!("rename it — `{name}` is a reserved dose-attribute name"),
                "#993",
            ),
        };
        return Err(format!(
            "{block}: `{name}` is this model's {noun}{scope} — the engine already \
             {applied} — but it is also read in {read_site}, so the value is applied \
             twice: once at the dose, once where you read it ({issue}). If `{name}` is \
             meant to be {noun}, remove it from {read_site}; if it is meant to be an \
             ordinary parameter, {ordinary_remedy}.",
            noun = attr.noun(),
            applied = attr.applied_as(),
        ));
    }
    Ok(())
}

/// The `indiv_param_slots` argument of [`check_dose_attr_double_use`] for an
/// **analytical** model (#1004): a vector parallel to `indiv_var_names` carrying
/// [`crate::types::PK_IDX_F`] / [`crate::types::PK_IDX_LAGTIME`] for the
/// parameter(s) the `pk(...)` call maps onto the dose route via `f=` /
/// `lagtime=`/`alag=`, and a `usize::MAX` sentinel everywhere else.
///
/// The sentinel makes `DoseAttrConsumption::of` fall through its bare-slot arm
/// to name-based recognition, exactly as the ODE caller's unrelated slots do —
/// so an analytical `D{n}`/`R{n}` still classifies as `CodedRateOnly` (skipped
/// here, judged with the data in `check_modeled_dose_rates`) and everything else
/// stays an ordinary parameter.
///
/// Binding resolution mirrors `build_pk_param_fn`'s, which is what actually
/// wires the value: the mapped name matches a declared parameter exactly, or by
/// its lowercase form (the legacy `vars.get(to_lowercase())` compat lookup). A
/// value that resolves to neither is a numeric literal, `TIME` (desugared to a
/// synthetic parameter upstream), or an undefined name that
/// `build_pk_param_fn` rejects with its own error — none of which can collide
/// with a read.
fn analytical_dose_attr_slot_map(
    pk_param_map: &HashMap<String, String>,
    indiv_var_names: &[String],
) -> Vec<usize> {
    let mut slots = vec![usize::MAX; indiv_var_names.len()];
    // Iterate in sorted key order. `pk_param_map` is a `HashMap`, and a parameter
    // bound to *both* roles (`pk(..., f=X, lagtime=X)`) is written twice — so with
    // arbitrary iteration order the surviving slot, and therefore the diagnostic
    // `check_dose_attr_double_use` emits, differed between runs of the identical
    // model. Same reason `build_pk_param_fn` sorts its `pk_entries`.
    let mut entries: Vec<(&String, &String)> = pk_param_map.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    for (key, value) in entries {
        let target = match PkParams::name_to_index(key) {
            Some(s @ (crate::types::PK_IDX_F | crate::types::PK_IDX_LAGTIME)) => s,
            _ => continue,
        };
        // Exact first, then the lowercase compat form — `build_pk_param_fn`'s own
        // order. The conversion is hoisted out of the `position` closure so it
        // costs one allocation per mapping rather than one per candidate name.
        let idx = indiv_var_names.iter().position(|n| n == value).or_else(|| {
            let lowered = value.to_lowercase();
            indiv_var_names.iter().position(|n| *n == lowered)
        });
        if let Some(i) = idx {
            slots[i] = target;
        }
    }
    slots
}

/// The `pk(...)` entry that binds `name` to dose attribute `attr` on an analytical
/// model, as `(role key, mapped value)` — both **as the user spelled them**, so the
/// remediation clause in [`check_dose_attr_double_use`] quotes text that is
/// actually in the model file (`alag=TLAG`, not the canonical `lagtime=tlag`).
///
/// Resolution mirrors [`analytical_dose_attr_slot_map`]'s, which is what assigned
/// the slot in the first place: the role key must route to `attr`'s reserved slot,
/// and the mapped value must match the declared parameter exactly or through the
/// legacy lowercase compat lookup. Filtering on `attr` is what keeps the noun and
/// the quoted role in agreement when one parameter fills both roles; sorting picks
/// one deterministically in that same case.
fn analytical_role_binding<'a>(
    pk_param_map: &'a HashMap<String, String>,
    name: &str,
    attr: crate::types::DoseAttr,
) -> Option<(&'a String, &'a String)> {
    let target = match attr {
        crate::types::DoseAttr::F => crate::types::PK_IDX_F,
        _ => crate::types::PK_IDX_LAGTIME,
    };
    let mut hit: Option<(&String, &String)> = None;
    for (key, value) in pk_param_map {
        if PkParams::name_to_index(key) != Some(target) {
            continue;
        }
        if value.as_str() != name && value.to_lowercase() != name {
            continue;
        }
        // Keep the LARGEST role key, matching which mapping actually binds: both
        // `analytical_dose_attr_slot_map` and `build_pk_param_fn` iterate the map
        // in ascending key order and let the last write win, so with `lagtime=X`
        // and `alag=X` both present it is `lagtime=` that reaches the slot. Quoting
        // the other one would name a mapping whose removal changes nothing.
        match hit {
            Some((k, _)) if k >= key => {}
            _ => hit = Some((key, value)),
        }
    }
    hit
}

/// The individual-parameter names an expression *source* references, upper-cased,
/// for [`check_dose_attr_double_use`] (#993).
///
/// The block parsers keep compiled programs rather than the name references, so the
/// source is re-walked here. `ParseCtx::new` is deliberately the same constructor
/// `build_y_output_fn` / `build_obs_scale_spec` / `compile_observe` use, minus the
/// state names: with `fallback_covariate`, a state resolves to `Covariate` here and
/// is ignored, which is what we want — the check is about *parameters*. Dropping the
/// state names can only *lose* a `Variable`, never invent one, so this cannot
/// manufacture a rejection the real parse would not have seen.
///
/// For `[scaling]` this re-walks an expression `parse_scaling_block` already parsed,
/// so it cannot fail. For `[adaptive_dosing] observe` it is the **first** parse —
/// the block parser stores the raw string and `compile_observe` only runs at
/// simulate time — so a syntactically malformed `observe` now surfaces here, at
/// parse time, instead of at the first `simulate()`. Earlier and with the same
/// message; nothing that used to compile stops compiling.
///
/// One shared collector rather than one per block: the boundary that matters is
/// "an expression compiled against the individual parameters", not any particular
/// block heading, and this rule already had to be extended to a third such
/// expression once (`[adaptive_dosing] observe`).
fn collect_indiv_param_reads(
    src: &str,
    theta_names: &[String],
    eta_names: &[String],
    indiv_var_names: &[String],
    intermediates: &[(String, String)],
    context: &str,
) -> Result<std::collections::HashSet<String>, String> {
    let ctx = ParseCtx::new(theta_names, eta_names, indiv_var_names);
    let mut expr = parse_scalar_expression(src, ctx).map_err(|e| format!("{context}: {e}"))?;
    // `[scaling]` named intermediates (#1030); empty for every other caller.
    inline_scaling_intermediates(&mut expr, intermediates, ctx)?;
    let mut reads: std::collections::HashSet<String> = std::collections::HashSet::new();
    visit_expr_nodes(&expr, &mut |e: &Expression| {
        if let Expression::Variable(name) = e {
            reads.insert(name.to_ascii_uppercase());
        }
    });
    Ok(reads)
}

/// Record every `D{n}`/`R{n}` in `reads` on the model's active dose-attribute map,
/// so `check_modeled_dose_rates` can report the double use once a dataset is in hand
/// (#993). Unlike its `EveryDose` siblings these are inert until a dose codes
/// `RATE=-2`/`-1`, so nothing is rejected here — see [`DoseAttrConsumption`].
///
/// `slot_map` is parallel to `indiv_var_names` by position but only as a **prefix**:
/// the `__ferx_pktime_` desugaring appends names after the slot map was built. `get(i)`
/// is therefore the correct lookup (and matches [`check_dose_attr_double_use`]) — a
/// real dose attribute is always a user name, hence inside the prefix, while the
/// synthetic tail is never one. Empty for analytical models, whose indexed
/// `D{n}`/`R{n}` are recognised by name alone.
fn record_coded_rate_reads(
    model: &mut CompiledModel,
    indiv_var_names: &[String],
    slot_map: &[usize],
    reads: &std::collections::HashSet<String>,
) {
    for (i, name) in indiv_var_names.iter().enumerate() {
        if !reads.contains(&name.to_ascii_uppercase()) {
            continue;
        }
        let slot = slot_map.get(i).copied();
        if let Some((attr, crate::types::DoseAttrConsumption::CodedRateOnly, Some(cmt))) =
            crate::types::DoseAttrConsumption::of(name, slot)
        {
            model
                .active_dose_attr_map_mut()
                .mark_prediction_path_read(attr, cmt, name);
        }
    }
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

/// Inspect the operator context around an input-rate call spanning `[start, end)`
/// in `line` (ASCII) and resolve an optional **pathway-fraction multiplier**
/// (#388). The forcing is injected additively as `+frac·R_in`, so the only
/// faithful contexts are a bare `+ fn(...)` term or a single
/// declared-individual-parameter multiplier `FR * fn(...)`. Returns:
///
/// - `Ok((None, start))` — a bare, positively-signed additive term (no fraction);
///   the rewrite-to-`0` starts at `start`.
/// - `Ok((Some(slot), prefix_start))` — `FR*fn(...)` where `FR` is a declared
///   individual parameter standing **alone** as the leading factor; `slot` is its
///   parameter slot and `prefix_start` is where `FR` begins, so the rewrite-to-`0`
///   swallows the `FR*` prefix too.
/// - `Err(msg)` — any other scaling (`* / ^`), a leading `-`, grouping (`(`
///   before / `)` after, e.g. `(fn(...))/V`), or a multiplier that is not a single
///   declared parameter (`(1-FR)*`, `2.5*`, `K*FR*`) — each silently drops the
///   sign/scale, so it is rejected loudly.
///
/// Requiring a bare declared parameter (not an arbitrary `(1-FR)` expression)
/// keeps the fraction bound to one slot; a two-pathway split is two terms whose
/// declared fractions sum to 1 (validated at fit-init).
fn parse_input_rate_prefix(
    line: &str,
    start: usize,
    end: usize,
    fname: &str,
    indiv_param_names: &[String],
    indiv_param_slots: &[usize],
) -> Result<(Option<usize>, usize), String> {
    let scaled_err = || {
        format!(
            "[odes]: {fname}(...) must be a bare additive input rate (`+ {fname}(...)`) or a \
             single declared-parameter pathway fraction (`FR*{fname}(...)`) — it cannot be \
             otherwise scaled (`* / ^`), negated (a leading `-`), or wrapped in parentheses \
             (e.g. `-{fname}(...)`, `({fname}(...))/V`), since these silently drop the sign/scale."
        )
    };
    let b = line.as_bytes();
    // Trailing context: a `* / ^` or a closing `)` right after the call hides an
    // outer scale/group (e.g. `fn(...)/V`, `(fn(...))*K`). Always rejected.
    let mut j = end;
    while j < b.len() && b[j] == b' ' {
        j += 1;
    }
    if j < b.len() && matches!(b[j], b'*' | b'/' | b'^' | b')') {
        return Err(scaled_err());
    }
    // Leading context: first non-space char to the left of the call.
    let mut i = start;
    while i > 0 && b[i - 1] == b' ' {
        i -= 1;
    }
    if i == 0 {
        return Ok((None, start)); // start of line (defensive; real RHS lines begin `d/dt(X)=`)
    }
    match b[i - 1] {
        // Bare additive term: `= fn(...)` (RHS start) or `+ fn(...)`.
        b'+' | b'=' => Ok((None, start)),
        // A multiplier: it must be a single declared individual parameter `FR *`.
        b'*' => {
            let mut k = i - 1; // index of the `*`
            while k > 0 && b[k - 1] == b' ' {
                k -= 1;
            }
            // Identifier token immediately left of the `*`.
            let mut id_start = k;
            while id_start > 0
                && (b[id_start - 1].is_ascii_alphanumeric() || b[id_start - 1] == b'_')
            {
                id_start -= 1;
            }
            if id_start == k {
                return Err(scaled_err()); // no identifier (e.g. `)*fn`, numeric `5*fn`)
            }
            // The identifier must stand alone as the leading factor: the char before
            // it (skipping spaces) is `+`/`=`/start. Anything else is a compound
            // expression (`(1-FR)*`, `K*FR*`, `1-FR*`) or a negated/subtracted term
            // (`- FR*`) — the sign/scale would be silently dropped, so this is the
            // `scaled_err` case, *not* a "not a declared parameter" one (the name may
            // well be declared; the problem is its surrounding context).
            let mut p = id_start;
            while p > 0 && b[p - 1] == b' ' {
                p -= 1;
            }
            let name = &line[id_start..k];
            // A leading digit means the "multiplier" is a numeric literal (`5*igd(...)`,
            // or the `5` tail of `2.5*igd(...)`), not a parameter identifier — it drops
            // a scale, so report the sign/scale rejection rather than the misleading
            // "`5` is not a declared parameter".
            if name.as_bytes().first().is_some_and(u8::is_ascii_digit) {
                return Err(scaled_err());
            }
            if p != 0 && !matches!(b[p - 1], b'+' | b'=') {
                return Err(scaled_err());
            }
            let slot = indiv_param_names
                .iter()
                .position(|q| q == name)
                .map(|idx| indiv_param_slots[idx])
                .ok_or_else(|| frac_not_param_err(fname, name))?;
            Ok((Some(slot), id_start))
        }
        // Scaled (`/ ^`), negated (`-`), or grouped (`(`): all drop the sign/scale.
        _ => Err(scaled_err()),
    }
}

/// Error for a pathway-fraction multiplier that is not a single declared
/// individual parameter (#388) — e.g. `(1-FR)*fn(...)` or `2.5*fn(...)`.
fn frac_not_param_err(fname: &str, name: &str) -> String {
    format!(
        "[odes]: the pathway-fraction multiplier `{name}` on `{fname}(...)` must be a single \
         declared individual parameter (e.g. `FR*{fname}(...)` with `FR` in \
         [individual_parameters]). An expression like `(1-FR)` is not allowed — declare a \
         complementary parameter whose fractions sum to 1."
    )
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

/// Built-in absorption input-rate functions recognised in `[odes]` RHS lines
/// (design A), each with its [`InputRateKind`] and the **ordered** named
/// arguments it requires. Arguments are resolved to individual-parameter slots
/// and stored in `arg_slots` in this order, so `src/pk/absorption.rs` reads them
/// positionally. Each new model adds one row here in the PR that implements it
/// (`transit`, #343; `igd`, #347; `weibull`, Phase 2; `zero_order`, #504).
const INPUT_RATE_FNS: &[(&str, crate::pk::absorption::InputRateKind, &[&str])] = &[
    (
        "transit",
        crate::pk::absorption::InputRateKind::Transit,
        &["n", "mtt"],
    ),
    (
        "igd",
        crate::pk::absorption::InputRateKind::InverseGaussian,
        &["mat", "cv2"],
    ),
    (
        "weibull",
        crate::pk::absorption::InputRateKind::Weibull,
        &["td", "beta"],
    ),
    (
        "zero_order",
        crate::pk::absorption::InputRateKind::ZeroOrder,
        &["dur"],
    ),
    (
        "first_order",
        crate::pk::absorption::InputRateKind::FirstOrder,
        &["ka"],
    ),
];

/// Extract built-in absorption input-rate calls ([`INPUT_RATE_FNS`] —
/// `transit(...)`, `igd(...)`, `weibull(...)`, `zero_order(...)`) from the `[odes]`
/// RHS lines. For each `d/dt(STATE) = … [FR*]fn(arg=P, …) …`, records an
/// [`InputRateForcing`] (`cmt` ← STATE index; named args resolved to
/// individual-parameter slots in declared order; an optional leading declared
/// parameter `FR` resolved to `frac_slot`) and rewrites the `[FR*]fn(...)` span to
/// `0` so the remaining RHS parses as a normal expression. A line may carry **more
/// than one** input-rate term (parallel / biphasic absorption, #388). Returns the
/// cleaned lines and the forcings.
///
/// Validation (negative-tested): an input-rate call may appear only in a top-level
/// `d/dt(...)` equation; its args are the function's declared names plus the
/// universal optional `lag` (per-route absorption delay), each a declared
/// individual parameter; a non-fraction scaling / sign / grouping is rejected
/// ([`parse_input_rate_prefix`]); a fraction multiplier must be a single declared
/// parameter; and a fraction on `zero_order(...)` is rejected pending #505.
fn extract_input_rate_terms(
    rhs_lines: &[String],
    state_names: &[String],
    indiv_param_names: &[String],
    indiv_param_slots: &[usize],
) -> Result<(Vec<String>, Vec<crate::pk::absorption::InputRateForcing>), String> {
    use crate::pk::absorption::InputRateForcing;
    let resolve_slot = |fname: &str, val: &str, arg: &str| -> Result<usize, String> {
        indiv_param_names
            .iter()
            .position(|p| p == val)
            .map(|i| indiv_param_slots[i])
            .ok_or_else(|| {
                format!(
                    "[odes]: {fname}({arg}={val}): `{val}` is not a declared individual parameter"
                )
            })
    };
    // "`a` and `b`" — for the unknown-/expected-argument error messages.
    let arg_list = |args: &[&str]| -> String {
        args.iter()
            .map(|a| format!("`{a}`"))
            .collect::<Vec<_>>()
            .join(" and ")
    };

    let mut forcings = Vec::new();
    let mut cleaned = Vec::with_capacity(rhs_lines.len());
    for raw in rhs_lines {
        // Lines with no built-in input-rate call pass through untouched.
        if !INPUT_RATE_FNS
            .iter()
            .any(|&(f, _, _)| find_word_call(raw, f).is_some())
        {
            cleaned.push(raw.clone());
            continue;
        }

        // An input-rate call may appear only in a `d/dt(STATE)` equation; the state
        // (hence compartment) is shared by every input-rate term on the line.
        let state = diffeq_state(raw).ok_or_else(|| {
            format!(
                "[odes]: an absorption input-rate function may only be the input rate of a \
                 `d/dt(...)` equation — found one in `{}`",
                raw.trim()
            )
        })?;
        let cmt = state_names
            .iter()
            .position(|s| s == &state)
            .ok_or_else(|| format!("[odes]: d/dt({state}): undeclared state"))?;

        // Extract every input-rate call on the line — parallel / biphasic absorption
        // splits the dose across ≥2 terms (#388) — rewriting each (with its optional
        // `FR*` fraction prefix) to `0` so the remaining RHS parses as a normal
        // expression. `work` shrinks each pass; indices below are into `work`.
        let mut work = raw.clone();
        loop {
            let Some((fname, kind, arg_names, start)) = INPUT_RATE_FNS
                .iter()
                .find_map(|&(f, k, a)| find_word_call(&work, f).map(|s| (f, k, a, s)))
            else {
                break;
            };

            let open = start + fname.len();
            let (inner, end) = balanced_parens(&work, open).ok_or_else(|| {
                format!(
                    "[odes]: {fname}(...): unbalanced parentheses in `{}`",
                    raw.trim()
                )
            })?;

            // Operator context: resolve an optional `FR*` pathway fraction and reject
            // any other scaling / sign / grouping. `rewrite_start` ≤ `start` swallows
            // the `FR*` prefix (if any) when the term is replaced by `0`.
            let (frac_slot, rewrite_start) = parse_input_rate_prefix(
                &work,
                start,
                end,
                fname,
                indiv_param_names,
                indiv_param_slots,
            )?;

            // A `FR*zero_order(...)` term (the `mixed` model, #505) is now accepted:
            // the per-segment zero-order channel (`zero_order_windows`) scales its
            // window rate by the pathway fraction. Multiple zero-order terms on one
            // compartment (biphasic zero-order) are rejected by the structural check
            // in `build_ode_spec` (after the full forcing list is assembled), since
            // the single-window-per-dose channel would under-deliver them.

            // Resolve each named arg into its declared slot position. `lag` is a
            // **universal optional** argument on every input-rate function:
            // `fn(..., lag=L)` gives that pathway its own absorption delay, added on
            // top of any compartment lagtime, so parallel / mixed routes can switch
            // on at different times. It is not one of the kind's required args, so it
            // is captured separately into `lag_slot` rather than `slots`.
            let mut slots: Vec<Option<usize>> = vec![None; arg_names.len()];
            let mut lag_slot: Option<usize> = None;
            for part in split_args_on_commas(&inner) {
                let (name, val) = part.split_once('=').ok_or_else(|| {
                    format!(
                        "[odes]: {fname}(...) arguments must be `name=parameter`, got `{}`",
                        part.trim()
                    )
                })?;
                let (name, val) = (name.trim(), val.trim());
                if name == "lag" {
                    if lag_slot.is_some() {
                        return Err(format!(
                            "[odes]: {fname}(...) has a duplicate `lag` argument"
                        ));
                    }
                    lag_slot = Some(resolve_slot(fname, val, name)?);
                    continue;
                }
                match arg_names.iter().position(|a| *a == name) {
                    Some(i) => slots[i] = Some(resolve_slot(fname, val, name)?),
                    None => {
                        return Err(format!(
                            "[odes]: {fname}(...) has no argument `{name}` (expected {} \
                             or the optional `lag`)",
                            arg_list(arg_names)
                        ))
                    }
                }
            }
            let mut arg_slots = Vec::with_capacity(arg_names.len());
            for (i, slot) in slots.into_iter().enumerate() {
                arg_slots.push(slot.ok_or_else(|| {
                    format!(
                        "[odes]: {fname}(...) missing required argument `{}`",
                        arg_names[i]
                    )
                })?);
            }

            forcings.push(InputRateForcing {
                cmt,
                kind,
                arg_slots,
                frac_slot,
                lag_slot,
            });

            // Rewrite the `[FR*]fn(...)` span to `0`, then loop to find the next term.
            work = format!("{}0{}", &work[..rewrite_start], &work[end..]);
        }
        cleaned.push(work);
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
            // The init built-ins are constants/indices that `eval_expression`
            // resolves case-insensitively, so accept any casing here (the
            // exact-key `init_defined` carries only states/params). `MIXNUM`
            // never reaches `undef` — it parses to its own `Expression::MixNum`
            // node — but is filtered anyway so this list is the single place the
            // accepted built-ins are named.
            undef.retain(|n| {
                !ODE_INIT_SCOPE_BUILTINS
                    .iter()
                    .any(|b| n.eq_ignore_ascii_case(b))
            });
            // Split the out-of-scope names into the time clocks and everything
            // else, so each gets the diagnostic that explains it (#994). The
            // clocks reach `undef` under any casing (`T`, `t`, `Tafd`, …), and a
            // bare `TIME`/`time` reaches none of it at all — it parses to
            // `Expression::Time`, a dedicated node rather than a `Variable`,
            // which is exactly how it evaded this check and became the bug. Both
            // routes converge here so all spellings of all four clocks get one
            // message: previously `TIME` was accepted, `time` accepted, and
            // `Time` reported as a plain undefined name.
            let mut clocks: Vec<String> = Vec::new();
            if expr_references_time_node(&expr) {
                clocks.push("TIME".to_string());
            }
            undef.retain(|n| {
                if ODE_INIT_REJECTED_BUILTINS
                    .iter()
                    .any(|b| n.eq_ignore_ascii_case(b))
                {
                    clocks.push(n.clone());
                    false
                } else {
                    true
                }
            });
            clocks.sort();
            clocks.dedup();
            let mut names: Vec<String> = undef.into_iter().collect();
            names.sort();
            // Report both problems from one parse — an expression can carry a
            // clock and an undefined name at once, and fixing them one error per
            // parse is two round-trips for one line.
            if !clocks.is_empty() {
                let also = if names.is_empty() {
                    String::new()
                } else {
                    format!(
                        " It also references undefined name(s): {}.",
                        names.join(", ")
                    )
                };
                return Err(format!(
                    "[odes] init({}): references the time built-in(s): {}. An initial \
                     condition is evaluated at the time origin, so a clock there is \
                     always exactly 0 and the expression can only ever read as its \
                     t = 0 value. Write that value directly, or move the time \
                     dependence into the `d/dt(...)` RHS, where `TIME` resolves to \
                     the integrator's current time.{} See issue #994.",
                    name,
                    clocks.join(", "),
                    also,
                ));
            }
            if !names.is_empty() {
                let mut defined = init_ctx_defined.clone();
                defined.sort();
                return Err(format!(
                    "[odes] init({}): references undefined name(s): {}. An init \
                     expression may only reference declared states (0 at init \
                     time), individual parameters, or the {} built-ins \
                     (defined: {}). The time built-ins ({}) are out of scope — an \
                     initial condition is evaluated at the time origin.",
                    name,
                    names.join(", "),
                    ODE_INIT_SCOPE_BUILTINS.join(" / "),
                    defined.join(", "),
                    ODE_INIT_REJECTED_BUILTINS.join(", "),
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

    // Structural invariant (compile-time): **at most one zero-order forcing per
    // compartment**. The per-segment zero-order channel
    // (`predictions::zero_order_windows`) builds one window per dose and
    // `zero_order_dur_and_frac_for_dose` resolves a *single* zero-order forcing per
    // dose-compartment via `find_map`, so a second `zero_order(...)` on the same
    // compartment would be silently under-delivered (#505). A `mixed` model pairs
    // one zero-order with a `first_order(...)` (a freely-superposed Channel-A
    // pointwise term) — that is allowed; only ≥2 *zero-order* terms on a
    // compartment are rejected. Enforced here, in the parser, rather than in the
    // data-level `api::check_absorption_dosing` so the `simulate()` / `predict()`
    // paths — which never run that fit-init check — are guarded too (every entry
    // point parses the model first).
    {
        let mut zero_order_per_cmt: std::collections::BTreeMap<usize, usize> =
            std::collections::BTreeMap::new();
        for f in &input_rate {
            if f.kind == crate::pk::absorption::InputRateKind::ZeroOrder {
                *zero_order_per_cmt.entry(f.cmt).or_insert(0) += 1;
            }
        }
        if let Some((&cmt, &n)) = zero_order_per_cmt.iter().find(|&(_, &n)| n >= 2) {
            return Err(format!(
                "[odes]: compartment {} has {n} zero-order input-rate terms \
                 (`zero_order(...)`). Multiple zero-order pathways on one compartment \
                 (biphasic zero-order) are not supported — use at most one \
                 `zero_order(...)` per compartment (a `mixed` model pairs it with a \
                 `first_order(...)` term).",
                cmt + 1
            ));
        }
    }

    // Structural invariant (compile-time, #388/#588): built-in absorption **pathway
    // fractions must partition the dose on a compartment**. When ≥2 input-rate terms
    // feed one compartment they must *each* carry a fraction (`FR*fn(...)`) — a bare
    // term alongside a fractioned one (or two bare terms) would deliver the full
    // `F·Dose` more than once. And a *lone* fractioned term is rejected: a fraction
    // only partitions across ≥2 terms, so a single pathway is written bare (`F`
    // handles partial absorption). Enforced here, in the parser (mirroring the
    // at-most-one-zero-order rule above), rather than only in the data-level
    // `api::check_absorption_dosing`, so the `simulate()` / `predict()` paths — which
    // never run that fit-init check — reject a malformed multi-pathway model too
    // (every entry point parses the model first). The companion **value** checks
    // (each fraction in (0, 1], Σ ≈ 1) depend on typical parameter values, so they
    // stay data-level; the simulate/predict paths reach them via
    // `assert_absorption_dosing_supported`.
    {
        let mut frac_count: std::collections::BTreeMap<usize, (usize, usize)> =
            std::collections::BTreeMap::new(); // cmt -> (total, fractioned)
        for f in &input_rate {
            let e = frac_count.entry(f.cmt).or_insert((0, 0));
            e.0 += 1;
            if f.frac_slot.is_some() {
                e.1 += 1;
            }
        }
        for (&cmt, &(total, fractioned)) in &frac_count {
            if total >= 2 && fractioned != total {
                return Err(format!(
                    "[odes]: compartment {} has {total} built-in absorption input-rate terms \
                     but only {fractioned} carry a pathway fraction. When more than one term \
                     feeds a compartment, each must be written `FR*fn(...)` with the fractions \
                     summing to 1 — otherwise the dose mass is counted more than once.",
                    cmt + 1
                ));
            }
            if total == 1 && fractioned == 1 {
                return Err(format!(
                    "[odes]: compartment {} has a single input-rate term carrying a pathway \
                     fraction (`FR*fn(...)`). A pathway fraction only applies when ≥2 terms \
                     split the dose across a compartment — write a single pathway as a bare \
                     `+ fn(...)`, and use bioavailability `F` for partial absorption.",
                    cmt + 1
                ));
            }
        }
    }

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
    // Switch to the indexed AST evaluator (`eval_statements_indexed_with_stack`) that
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

    // Reject a dose attribute (`F`, `LAGTIME`/`ALAG`, `F{n}`, `ALAG{n}`) that the
    // engine applies at the dose event *and* the RHS reads — it would be applied
    // twice, silently (#993). Sibling of the covariate rejection above: same class
    // of defect (a model that parses clean and computes the wrong number), same
    // remedy (say so at parse time). Read off the resolved statement tree rather
    // than the block text, so it sees exactly what the evaluator will: comments are
    // already stripped, `d/dt(X)` / `Assign` left-hand sides are not reads, and the
    // `ode_template`-generated equations plus the injected joint-PK-TTE
    // `d/dt(__chz_n)` hazard lines are all in `stmts_owned` by now.
    //
    // `init(state) = <expr>` reads are deliberately NOT folded in (#1046), mirroring
    // the analytical `[initial_conditions]` carve-out (#1004/#1035). An initial
    // condition is not a dose: `OdeSpec::initial_state` seeds the state vector with
    // the raw expression value, and `F` / lag are resolved through `DoseAttrMap` at
    // dose events only — a path the seed never takes. So `init(central) = F * 100`
    // (the bioavailable residue of a pre-study dose) applies `F` exactly once, to a
    // quantity the engine never scales, and rejecting it would make a correct model
    // unwritable while advising the wrong repair — renaming the very parameter whose
    // meaning *is* bioavailability. The RHS is a genuine doubling by contrast:
    // `d/dt(central) = … F …` folds `F` into the flux, so every gram that ever
    // entered the compartment is multiplied by it a second time (the `F²` defect
    // #993 exists for), which is why the RHS walk below stays.
    //
    // Anchored on NONMEM 7.6.0 (`nonmem_anchor/odes_init_dose_attr_{f,lag}_{A,B}.ctl`):
    // `A_0(1) = F1*100` with `F1 = 0.5` seeds 50, not 25, and `A_0(1) = ALAG1*100`
    // seeds at t=0 unshifted — each table byte-identical to its twin seeding from an
    // ordinary parameter of the same value, so the reference engine likewise treats a
    // dose-attribute name as an ordinary value in an initial condition.
    //
    // For the same reason a `D{n}`/`R{n}` read in an init is not a prediction-path
    // read: the marking loop below is driven by this same `rhs_reads` set, so leaving
    // init out keeps it from recording one.
    let mut rhs_reads: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut collect_reads = |e: &Expression| {
        if let Expression::Variable(name) = e {
            rhs_reads.insert(name.to_ascii_uppercase());
        }
    };
    visit_stmt_nodes(&stmts_owned, &mut collect_reads);
    check_dose_attr_double_use(
        &indiv_names_owned,
        &indiv_slots_owned,
        &rhs_reads,
        n_states,
        "[odes]",
        "the [odes] RHS",
        None,
    )?;

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
    // Snapshot the resolved RHS program for the analytic-sensitivity path
    // (issue #367) before the f64 closure moves `stmts_owned`. Same statements,
    // same var layout — evaluated over a dual type by `eval_rhs_g`.
    let uses_time_vars = stmts_read_slots(&stmts_owned, &[time_slot, tafd_slot, tad_slot]);
    let rhs_program = OdeRhsProgram {
        stmts: stmts_owned.clone(),
        n_vars_total,
        state_count,
        indiv_to_params_slot: indiv_to_params_slot.clone(),
        time_slot,
        tafd_slot,
        tad_slot,
        macheps_slot,
        uses_time_vars,
    };

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
                // A bare `TIME` in the RHS parses to `Expression::Time`
                // (`Op::PushTime`), which resolves from the model-time
                // thread-local — not the `time_slot` the `T`/`t` aliases use.
                // Set it to the integrator clock so `TIME` matches `T` (#610).
                let _time_guard = ModelTimeGuard::enter(t);
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
            // #993: a `D{cmt}`/`R{cmt}` that the RHS *also* reads is a double use —
            // but only for a dose that actually codes `RATE=-2`/`-1`, which is a
            // property of the data, not the model. (`F{cmt}`/`ALAG{cmt}` in the same
            // position were already rejected above; they apply to every dose.) Record
            // it so `check_modeled_dose_rates` can report it once the dataset is in
            // hand, and leave a model whose data never codes `RATE` untouched.
            if rhs_reads.contains(&name.to_ascii_uppercase()) {
                dose_attr_map.mark_prediction_path_read(attr, cmt, name);
            }
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
        rhs_program: Some(rhs_program),
        // Form C readout + individual-parameter programs are attached later, when
        // `[scaling]` / `[individual_parameters]` are parsed (see the wiring).
        readout_program: None,
        indiv_param_program: None,
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

/// Specifies a correlated residual-error block.
///
/// Values are lower-triangle covariance-scale entries, matching NONMEM
/// `$SIGMA BLOCK(n)`: `[var1, cov21, var2, cov31, cov32, var3, ...]`.
struct BlockSigmaSpec {
    names: Vec<String>,
    lower_triangle: Vec<f64>,
    fixed: bool,
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
    /// Parallel to `names_ordered`: the `weight = <expr>` source text (#1031),
    /// or `None` for an unweighted kappa. Always `None` for a `block_kappa`
    /// member, which the parser rejects a weight on.
    weights: Vec<Option<String>>,
}

// --- Block extraction ---

/// Which header form a `[block]` accepts.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BlockForm {
    /// `[name]` only.
    Unnamed,
    /// `[name]` or `[name INSTANCE]` — several instances may coexist.
    Either,
    /// `[name INSTANCE]` only.
    Named,
}

/// Every `[block]` name the parser reads, with the header form it accepts and
/// the cargo feature it needs (`None` = always compiled in).
///
/// This is the closed-world registry behind `E_UNKNOWN_BLOCK` (#1040). Blocks
/// are consumed by *name lookup* — `blocks.get("fit_options")` and friends —
/// never by an exhaustive match, so without this table a block the parser does
/// not know is simply never read: a misspelled `[fit_option]` used to leave
/// `ferx check` reporting `valid: true` while the fit silently ran with the
/// default method and no covariance step. Keys inside a block are already
/// closed-world (`[event_model]: unknown key ...`); this closes the same door
/// on the block names themselves.
///
/// **Teaching the parser a new block means adding it here in the same PR** —
/// otherwise the new block is rejected as unknown.
///
/// `covariate_nn` carries no feature here on purpose: it has its own, more
/// specific `E_NN_FEATURE_DISABLED` error (which points at the design doc)
/// raised further down in `parse_full_model`.
const BLOCK_REGISTRY: &[(&str, BlockForm, Option<&str>)] = &[
    ("adaptive_dosing", BlockForm::Unnamed, None),
    ("binary_model", BlockForm::Either, Some("survival")),
    ("covariate_nn", BlockForm::Named, None),
    ("covariates", BlockForm::Unnamed, None),
    ("data", BlockForm::Unnamed, None),
    ("data_selection", BlockForm::Unnamed, None),
    ("derived", BlockForm::Unnamed, None),
    ("diffusion", BlockForm::Unnamed, None),
    ("error_model", BlockForm::Unnamed, None),
    ("event_model", BlockForm::Either, Some("survival")),
    ("fit_options", BlockForm::Unnamed, None),
    ("individual_parameters", BlockForm::Unnamed, None),
    ("initial_conditions", BlockForm::Unnamed, None),
    ("markov_model", BlockForm::Either, Some("markov")),
    ("mixture", BlockForm::Unnamed, None),
    ("odes", BlockForm::Unnamed, None),
    ("output", BlockForm::Unnamed, None),
    ("parameters", BlockForm::Unnamed, None),
    ("scaling", BlockForm::Unnamed, None),
    ("simulation", BlockForm::Unnamed, None),
    ("structural_model", BlockForm::Unnamed, None),
];

/// Blocks that *were* ferx syntax and are no longer read, with the remediation
/// to print. Kept separate from [`BLOCK_REGISTRY`] because they are neither
/// valid (they must not appear in the "valid blocks" enumeration) nor unknown
/// (a bare "unknown block, did you mean …" is unhelpful for something that was
/// once the documented spelling, and the nearest valid name is often far enough
/// away that no did-you-mean fires at all).
///
/// `initial_values` is the one live case: initial estimates moved inline into
/// `[parameters]`, the parser stopped reading the block, and — because unknown
/// names were silently dropped — nothing ever said so. `ferx-r` still lists it
/// in its section docs and three of its checked-in example models still carry
/// one, so this fires for real files (#1040).
const DEPRECATED_BLOCKS: &[(&str, &str)] = &[(
    "initial_values",
    "initial estimates are declared inline in `[parameters]` \
     (`theta NAME(init, lower, upper)`, `omega NAME ~ variance`); \
     the block has not been read for several releases — delete it",
)];

/// The name of every `[block]` **this build** of the parser will accept, in the
/// order they appear in the error message (alphabetical).
///
/// Feature-gated blocks are filtered out when their feature is off: a default
/// build cannot use `[event_model]`, so listing it as valid would advertise a
/// name the same binary then rejects with `E_BLOCK_FEATURE_DISABLED`. The
/// registry itself still *recognises* those names in every build — that is what
/// makes the rejection a specific "build with `--features …`" error rather than
/// a bare "unknown block".
///
/// Exposed so downstream consumers (`ferx-r`'s `ferx_model_validate()`,
/// `ferxtranslate`) can source the block list from the engine instead of
/// keeping their own copy — the duplicated R-side list had already drifted
/// from this one (#1040). Because the result is build-dependent, a consumer
/// must read it from the same binary it will parse with.
pub fn known_block_names() -> Vec<&'static str> {
    BLOCK_REGISTRY
        .iter()
        .filter(|(_, _, feature)| feature.is_none_or(block_feature_enabled))
        .map(|(n, _, _)| *n)
        .collect()
}

/// Whether the cargo feature a registry entry names is enabled in this build.
fn block_feature_enabled(feature: &str) -> bool {
    match feature {
        "survival" => cfg!(feature = "survival"),
        "markov" => cfg!(feature = "markov"),
        "nn" => cfg!(feature = "nn"),
        // An unrecognised feature name can only be a registry typo; treating it
        // as enabled keeps a mistake here from rejecting a valid model.
        _ => true,
    }
}

/// Levenshtein distance, used only to offer a did-you-mean for a misspelled
/// block name.
fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, ca) in a.chars().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = usize::from(ca != *cb);
            cur[j + 1] = (prev[j] + cost).min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// The registry name closest to `name`, when one is close enough to be worth
/// suggesting: an edit distance of at most 2, or a plain prefix relationship
/// (which catches `fit_option` → `fit_options` and `param` → `parameters`).
pub(crate) fn nearest_block_name(name: &str) -> Option<&'static str> {
    BLOCK_REGISTRY
        .iter()
        .map(|(n, _, _)| *n)
        .filter(|n| n.starts_with(name) || name.starts_with(*n) || edit_distance(name, n) <= 2)
        .min_by_key(|n| edit_distance(name, n))
}

/// Reject any `[block]` header the parser would otherwise drop on the floor.
///
/// `headers` is every header seen, in source order, as
/// `(type, instance, 1-based line)`. Four silent-drop shapes are caught, in
/// the order a user is most likely to hit them:
///
/// 1. a block that was ferx syntax and is no longer read
///    (`E_DEPRECATED_BLOCK`),
/// 2. an unrecognised block name (`E_UNKNOWN_BLOCK`),
/// 3. a known block written in the wrong header form — an instance name on a
///    block that takes none, or a missing one on `[covariate_nn NAME]`
///    (`E_BLOCK_INSTANCE_NAME`),
/// 4. a known block whose cargo feature this binary was not built with
///    (`E_BLOCK_FEATURE_DISABLED`).
///
/// All findings of the first non-empty class are reported together, so an
/// author fixing a batch of typos sees them in one pass.
fn check_block_names(headers: &[(String, Option<String>, usize)]) -> Result<(), String> {
    let mut deprecated: Vec<String> = Vec::new();
    let mut unknown: Vec<String> = Vec::new();
    let mut wrong_form: Vec<String> = Vec::new();
    let mut disabled: Vec<String> = Vec::new();
    let mut seen: Vec<(&str, Option<&str>)> = Vec::new();

    for (ty, instance, line) in headers {
        // Report each distinct header once, however many times it is repeated.
        // Keyed on the instance name too, so `[foo A]` and `[foo B]` are two
        // findings rather than one entry that names only the first.
        let key = (ty.as_str(), instance.as_deref());
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        // Echo the header exactly as written, instance name included, so the
        // text is greppable in the user's own file.
        let header = match instance {
            Some(inst) => format!("[{ty} {inst}]"),
            None => format!("[{ty}]"),
        };
        if let Some((name, remedy)) = DEPRECATED_BLOCKS.iter().find(|(n, _)| *n == ty) {
            deprecated.push(format!(
                "`[{name}]` (line {line}) is no longer read: {remedy}"
            ));
            continue;
        }
        let Some((name, form, feature)) = BLOCK_REGISTRY.iter().find(|(n, _, _)| n == ty) else {
            let hint = match nearest_block_name(ty) {
                Some(near) => format!(" — did you mean `[{near}]`"),
                None => String::new(),
            };
            unknown.push(format!("`{header}` (line {line}){hint}"));
            continue;
        };
        match (form, instance) {
            (BlockForm::Unnamed, Some(inst)) => wrong_form.push(format!(
                "`[{name} {inst}]` (line {line}) does not take an instance name — write `[{name}]`"
            )),
            (BlockForm::Named, None) => wrong_form.push(format!(
                "`[{name}]` (line {line}) requires an instance name — write `[{name} NAME]`"
            )),
            _ => {}
        }
        if let Some(feature) = feature {
            if !block_feature_enabled(feature) {
                disabled.push(format!(
                    "`[{name}]` (line {line}) requires building ferx-core with `--features {feature}`"
                ));
            }
        }
    }

    if !deprecated.is_empty() {
        return Err(format!("Deprecated block {}.", deprecated.join("; ")));
    }
    if !unknown.is_empty() {
        return Err(format!(
            "Unknown block {}. Valid blocks: {}.",
            unknown.join("; "),
            known_block_names().join(", ")
        ));
    }
    if !wrong_form.is_empty() {
        return Err(format!("Block {}.", wrong_form.join("; ")));
    }
    if !disabled.is_empty() {
        return Err(format!("Block {}.", disabled.join("; ")));
    }
    Ok(())
}

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
    // Every header seen, in source order, as `(type, instance, 1-based line)`.
    // Validated against `BLOCK_REGISTRY` once the whole file has been scanned
    // (#1040) so a misspelled block name is an error rather than a silent drop.
    let mut headers: Vec<(String, Option<String>, usize)> = Vec::new();

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
            headers.push((
                ty.clone(),
                caps.get(2).map(|m| m.as_str().to_string()),
                idx + 1,
            ));
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

    check_block_names(&headers)?;

    // Fold operator continuation lines into single logical lines before anyone
    // reads the block (#1030). Done here rather than in each block's parser so the
    // token-stream consumers (`[individual_parameters]`, `[odes]`) and the
    // line-oriented ones (`[scaling]`) can't drift on what a "line" is.
    for ty in CONTINUATION_BLOCKS {
        if let Some(lines) = out.unnamed.get_mut(*ty) {
            *lines = join_continuation_lines(lines);
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

/// Block types whose lines are expression statements, and therefore accept
/// operator continuation lines (#1030). Everything else (`[parameters]`,
/// `[fit_options]`, …) keeps its strict one-declaration-per-line grammar.
///
/// The entry criterion is that a *statement* in the block can never begin with a
/// binary operator — see [`join_continuation_lines`] for why that is what makes
/// joining a strict widening rather than a reinterpretation.
const CONTINUATION_BLOCKS: &[&str] = &[
    "individual_parameters",
    "odes",
    "scaling",
    "derived",
    // `init(NAME) = <expr>` — line-oriented like `[scaling]`, and every line must
    // start with `init(`, so a line opening with an operator is unreachable here too.
    "initial_conditions",
];

/// Leading characters that mark a line as the continuation of the previous one.
/// Every statement in a continuation block starts with an identifier, `d/dt(`,
/// `if`, or a brace — never with a binary operator — so a line opening with one
/// of these can only be a continuation, and joining it strictly widens the
/// accepted grammar rather than reinterpreting anything that parsed before.
const CONTINUATION_LEADERS: &[char] = &['+', '-', '*', '/', '^'];

/// Trailing characters that mark a line as unfinished, so the *next* line
/// continues it. `=` is included so `LEMAX =` followed by an indented expression
/// works; `,` so a multi-line argument list does.
const CONTINUATION_TRAILERS: &[char] = &['+', '-', '*', '/', '^', ',', '='];

/// Join operator continuation lines into single logical lines (#1030).
///
/// A long `[individual_parameters]` expression — the additive-on-logit covariate
/// model, one covariate per line — is unreadable collapsed onto one line, but the
/// statement parser treats a newline as an end-of-statement marker, so the natural
/// layout used to fail with ``Expected an assignment, an `if` block, or
/// `d/dt(...)`, got Plus``. Two spellings are accepted, both unambiguous because
/// no statement can begin with a binary operator:
///
/// ```text
///   LEMAX = LEMAX0                     LEMAX = LEMAX0 +
///           + log(OR_ABATA) * ABATA            log(OR_ABATA) * ABATA
/// ```
///
/// Newlines inside `(...)` / `[...]` are already dropped by
/// [`strip_newlines_in_groups`], so this only has to cover the top level. It runs
/// on the comment-stripped, trimmed lines [`extract_blocks`] produces, and is
/// applied there so every consumer of a block's lines — including the
/// line-oriented `[scaling]` scans, which never see the token stream — gets the
/// joined form.
fn join_continuation_lines(lines: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let starts_continuation = trimmed
            .chars()
            .next()
            .is_some_and(|c| CONTINUATION_LEADERS.contains(&c));
        let prev_unfinished = out
            .last()
            .and_then(|l: &String| l.trim_end().chars().last())
            .is_some_and(|c| CONTINUATION_TRAILERS.contains(&c));
        match out.last_mut() {
            Some(last) if starts_continuation || prev_unfinished => {
                last.push(' ');
                last.push_str(trimmed);
            }
            // A block's first line can't continue anything; leave it alone so the
            // downstream parser reports its own error.
            _ => out.push(trimmed.to_string()),
        }
    }
    out
}

/// Whether any node of an expression references a random effect (`eta`).
/// Mixing-probability expressions must be eta-free (#977).
fn expr_uses_eta(expr: &Expression) -> bool {
    let mut found = false;
    visit_expr_nodes(expr, &mut |e| {
        if matches!(e, Expression::Eta(_)) {
            found = true;
        }
    });
    found
}

/// Parse a `[mixture]` block into a [`crate::types::MixtureSpec`] (#977).
///
/// Grammar (one directive per line, `#` starts a comment):
/// ```text
/// nsub = K                    # number of subpopulations, K >= 2
/// logit(k) = <expr>           # class-k logit (softmax); k in 1..=K-1
/// p(k)     = <expr>           # class-k probability (alternative form)
/// omega(k) NAME ~ var [(sd)] [FIX]   # per-class Omega override (k in 2..=K)
/// sigma(k) NAME ~ var [(sd)] [FIX]   # per-class Sigma override (k in 2..=K)
/// ```
/// `theta_names` / `eta_names` / `sigma_names` are the base declarations, used
/// to resolve theta references in the mixing expressions and to validate that an
/// override names an existing random-effect / residual term.
fn parse_mixture_block(
    lines: &[String],
    theta_names: &[String],
    eta_names: &[String],
    sigma_names: &[String],
) -> Result<crate::types::MixtureSpec, String> {
    use crate::types::{MixingExpr, MixtureClassOverride, MixtureSpec};

    let nsub_re = Regex::new(r"(?i)^nsub\s*=\s*(\d+)$").unwrap();
    let mix_re = Regex::new(r"(?i)^(logit|p)\s*\(\s*(\d+)\s*\)\s*=\s*(.+)$").unwrap();
    // `omega(k) NAME ~ var [FIX] [(sd|var)] [FIX]` — mirrors the base omega/sigma
    // regexes in `parse_parameters`, prefixed with the `(k)` class selector.
    let ov_re = Regex::new(
        r"(?i)^(omega|sigma)\s*\(\s*(\d+)\s*\)\s+(\w+)\s*~\s*([0-9eE.+-]+)(?:\s+(FIX)\b)?(?:\s*\((sd|variance|var)\))?(?:\s+(FIX)\b)?$",
    )
    .unwrap();

    let mut nsub: Option<usize> = None;
    let mut mixing: Vec<MixingExpr> = Vec::new();
    let mut omega_overrides: Vec<MixtureClassOverride> = Vec::new();
    let mut sigma_overrides: Vec<MixtureClassOverride> = Vec::new();

    for raw in lines {
        // Strip inline `#` comments and surrounding whitespace.
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }

        if let Some(c) = nsub_re.captures(line) {
            if nsub.is_some() {
                return Err("[mixture] duplicate `nsub` declaration".to_string());
            }
            nsub = Some(
                c[1].parse::<usize>()
                    .map_err(|_| "[mixture] bad nsub value")?,
            );
            continue;
        }

        if let Some(c) = ov_re.captures(line) {
            let is_omega = c[1].eq_ignore_ascii_case("omega");
            let class = c[2]
                .parse::<usize>()
                .map_err(|_| "[mixture] bad class index")?;
            let name = c[3].to_string();
            let raw_val: f64 = c[4]
                .parse()
                .map_err(|_| format!("[mixture] bad numeric value in `{line}`"))?;
            let as_sd = c
                .get(6)
                .is_some_and(|m| m.as_str().eq_ignore_ascii_case("sd"));
            // Reject negatives on both scales, mirroring the base omega/sigma
            // parsing — a negative variance/SD would `sqrt` to NaN or pack to a
            // clamped garbage value.
            if raw_val < 0.0 {
                let kind = if is_omega { "omega" } else { "sigma" };
                return Err(format!(
                    "[mixture] {kind}({class}) {} has a negative initial value ({raw_val}); \
                     both variance and SD must be non-negative",
                    c[3].to_string()
                ));
            }
            let fixed = c.get(5).is_some() || c.get(7).is_some();
            let ov = MixtureClassOverride {
                class,
                name,
                init: raw_val,
                as_sd,
                fixed,
            };
            if is_omega {
                omega_overrides.push(ov);
            } else {
                sigma_overrides.push(ov);
            }
            continue;
        }

        if let Some(c) = mix_re.captures(line) {
            let is_logit = c[1].eq_ignore_ascii_case("logit");
            let class = c[2]
                .parse::<usize>()
                .map_err(|_| "[mixture] bad class index")?;
            let expr_src = c[3].trim();
            let ctx = ParseCtx::new(theta_names, eta_names, &[]);
            let toks = tokenize(expr_src)?;
            let (expr, consumed) = parse_add_sub(&toks, 0, ctx)?;
            if consumed != toks.len() {
                return Err(format!(
                    "[mixture] could not fully parse mixing expression `{expr_src}`"
                ));
            }
            if expr_uses_eta(&expr) {
                return Err(
                    "[mixture] mixing-probability expression may not depend on a random effect (eta)"
                        .to_string(),
                );
            }
            mixing.push(MixingExpr {
                class,
                is_logit,
                expr,
            });
            continue;
        }

        return Err(format!("[mixture] unrecognized directive: `{line}`"));
    }

    let n_classes = nsub.ok_or("[mixture] block requires `nsub = K` (K >= 2)")?;
    if n_classes < 2 {
        return Err(format!("[mixture] nsub must be >= 2 (got {n_classes})"));
    }

    // Mixing-expression coverage: forms may not be mixed, and exactly the
    // classes 1..=K-1 must each be scored once (the last class is the softmax
    // reference / probability complement).
    if mixing.is_empty() {
        return Err(
            "[mixture] block requires a `logit(k)` or `p(k)` for each class 1..=nsub-1".to_string(),
        );
    }
    let all_logit = mixing.iter().all(|m| m.is_logit);
    let all_prob = mixing.iter().all(|m| !m.is_logit);
    if !(all_logit || all_prob) {
        return Err("[mixture] cannot mix `logit(k)` and `p(k)` forms in one block".to_string());
    }
    let mut seen = vec![false; n_classes]; // index 0 => class 1
    for m in &mixing {
        if m.class < 1 || m.class >= n_classes {
            return Err(format!(
                "[mixture] logit/p({}) out of range: expected class in 1..={}",
                m.class,
                n_classes - 1
            ));
        }
        if seen[m.class - 1] {
            return Err(format!(
                "[mixture] duplicate mixing expression for class {}",
                m.class
            ));
        }
        seen[m.class - 1] = true;
    }
    for k in 1..n_classes {
        if !seen[k - 1] {
            return Err(format!(
                "[mixture] missing mixing expression for class {k} (need one per class 1..={})",
                n_classes - 1
            ));
        }
    }

    // Overrides: class in 2..=K, and the name must exist as a base eta/sigma.
    let check_override = |ov: &MixtureClassOverride,
                          names: &[String],
                          kind: &str|
     -> Result<(), String> {
        if ov.class < 2 || ov.class > n_classes {
            return Err(format!(
                "[mixture] {kind}({}) out of range: per-class overrides use class 2..={n_classes} (class 1 is the base)",
                ov.class
            ));
        }
        if !names.contains(&ov.name) {
            return Err(format!(
                "[mixture] {kind}({}) {} does not name a base {kind} declaration",
                ov.class, ov.name
            ));
        }
        Ok(())
    };
    // Reject a duplicate (class, name) override: two declarations for the same
    // entry would emit two packed coordinates for one variance with last-wins
    // semantics, breaking the pack/unpack round-trip.
    let check_dupes = |overrides: &[MixtureClassOverride], kind: &str| -> Result<(), String> {
        let mut seen = std::collections::HashSet::new();
        for ov in overrides {
            if !seen.insert((ov.class, ov.name.clone())) {
                return Err(format!(
                    "[mixture] duplicate {kind}({}) override for {}",
                    ov.class, ov.name
                ));
            }
        }
        Ok(())
    };
    for ov in &omega_overrides {
        check_override(ov, eta_names, "omega")?;
    }
    for ov in &sigma_overrides {
        check_override(ov, sigma_names, "sigma")?;
    }
    check_dupes(&omega_overrides, "omega")?;
    check_dupes(&sigma_overrides, "sigma")?;

    // Covariates referenced by the mixing expressions, for data validation.
    let mut cov_set = std::collections::HashSet::new();
    for m in &mixing {
        visit_expr_nodes(&m.expr, &mut |e| {
            if let Expression::Covariate(name) = e {
                cov_set.insert(name.clone());
            }
        });
    }
    let mut logit_covariates: Vec<String> = cov_set.into_iter().collect();
    logit_covariates.sort();

    Ok(MixtureSpec {
        n_classes,
        mixing,
        logit_covariates,
        omega_overrides,
        sigma_overrides,
        // Filled in by `parse_model_string` once `[individual_parameters]` has
        // been parsed (this block is parsed before the mu-ref scan can run).
        mu_refs: Vec::new(),
    })
}

/// Build the numeric per-class Omega/Sigma (`MixtureParams`) from a `MixtureSpec`
/// and the base (class-1) Omega/Sigma (#977 Phase 2).
///
/// Each class starts as a copy of the base; every `omega(k)` / `sigma(k)`
/// override replaces one diagonal entry, converting from the declared scale
/// (`(sd)` → square for Omega variance, plain → square-root for the SD-scale
/// Sigma) exactly as the base declarations do. The override addresses drive the
/// per-class packed segment in `estimation::parameterization`. Omega overrides
/// require a diagonal base Omega (a block base has no well-defined per-eta
/// variance to swap).
fn build_mixture_params(
    spec: &crate::types::MixtureSpec,
    base_omega: &crate::types::OmegaMatrix,
    base_sigma: &crate::types::SigmaVector,
) -> Result<crate::types::MixtureParams, String> {
    use crate::types::{MixtureParams, OmegaMatrix, SigmaVector};

    if !spec.omega_overrides.is_empty() && !base_omega.diagonal {
        return Err(
            "[mixture] per-class `omega(k)` overrides require a diagonal base omega (a block \
             omega has no single per-eta variance to override)"
                .to_string(),
        );
    }

    let k = spec.n_classes;
    // Working matrices/vectors, one per class, seeded from the base.
    let mut work_omega: Vec<nalgebra::DMatrix<f64>> = vec![base_omega.matrix.clone(); k];
    let mut work_sigma: Vec<Vec<f64>> = vec![base_sigma.values.clone(); k];

    let mut omega_override_addr: Vec<(usize, usize)> = Vec::new();
    let mut omega_override_fixed: Vec<bool> = Vec::new();
    for ov in &spec.omega_overrides {
        let e = base_omega
            .eta_names
            .iter()
            .position(|n| n == &ov.name)
            .ok_or_else(|| format!("[mixture] omega({}) {} not a base eta", ov.class, ov.name))?;
        let c = ov.class - 1; // class is 1-based, validated >= 2 in Phase 1
        let variance = if ov.as_sd { ov.init * ov.init } else { ov.init };
        work_omega[c][(e, e)] = variance;
        omega_override_addr.push((c, e));
        omega_override_fixed.push(ov.fixed);
    }

    let mut sigma_override_addr: Vec<(usize, usize)> = Vec::new();
    let mut sigma_override_fixed: Vec<bool> = Vec::new();
    for ov in &spec.sigma_overrides {
        let s = base_sigma
            .names
            .iter()
            .position(|n| n == &ov.name)
            .ok_or_else(|| format!("[mixture] sigma({}) {} not a base sigma", ov.class, ov.name))?;
        let c = ov.class - 1;
        // Sigma is stored on the SD scale: `(sd)` keeps it, plain input is sqrt'd.
        let value = if ov.as_sd { ov.init } else { ov.init.sqrt() };
        work_sigma[c][s] = value;
        sigma_override_addr.push((c, s));
        sigma_override_fixed.push(ov.fixed);
    }

    let omega = (0..k)
        .map(|c| {
            OmegaMatrix::from_matrix_with_mask(
                work_omega[c].clone(),
                base_omega.eta_names.clone(),
                base_omega.diagonal,
                base_omega.free_mask.clone(),
            )
        })
        .collect();
    let sigma = (0..k)
        .map(|c| SigmaVector {
            values: work_sigma[c].clone(),
            names: base_sigma.names.clone(),
        })
        .collect();

    Ok(MixtureParams {
        omega,
        sigma,
        omega_override_addr,
        omega_override_fixed,
        sigma_override_addr,
        sigma_override_fixed,
    })
}

/// Re-add the RAII guard for the mixture-class thread-local (#977 Phase 3). The
/// mixture objective sets `MIXTURE_CLASS = k` around each class's inner EBE solve
/// and NLL so `MIXNUM` branches in `[individual_parameters]` see class `k`, then
/// restores the prior value on drop (panic-safe). Must be set on the *same*
/// thread that runs the prediction (the objective drives the per-class solve
/// serially for exactly this reason — a thread-local set on the outer thread
/// would not reach rayon workers).
pub(crate) struct MixtureClassGuard(usize);

impl MixtureClassGuard {
    /// Set `MIXTURE_CLASS` to `class` (1-based), restoring the prior value on drop.
    pub(crate) fn enter(class: usize) -> Self {
        let prev = MIXTURE_CLASS.with(|cell| cell.replace(class));
        MixtureClassGuard(prev)
    }
}

impl Drop for MixtureClassGuard {
    fn drop(&mut self) {
        MIXTURE_CLASS.with(|cell| cell.set(self.0));
    }
}

/// Evaluate the per-class mixing log-probabilities `ln p_ik` for one subject
/// (#977 Phase 3). Mixing expressions depend only on theta + (baseline)
/// covariates — never eta (validated at parse time) — so this takes a plain
/// covariate map and no random effects.
///
/// - `logit(k)` form: scores `1..=K-1`, reference class `K` has logit 0;
///   `p = softmax(logits)`.
/// - `p(k)` form: probabilities for `1..=K-1`, last class is the complement;
///   renormalized defensively.
pub(crate) fn eval_mixing_log_probs(
    spec: &crate::types::MixtureSpec,
    theta: &[f64],
    covariates: &HashMap<String, f64>,
) -> Vec<f64> {
    let k = spec.n_classes;
    let empty_vars: HashMap<String, f64> = HashMap::new();
    let env = MapEnv {
        covariates,
        vars: &empty_vars,
    };
    let no_eta: [f64; 0] = [];

    if spec.mixing.iter().all(|m| m.is_logit) {
        let mut logits = vec![0.0f64; k]; // reference class K stays 0
        for m in &spec.mixing {
            logits[m.class - 1] = eval_expr(&m.expr, theta, &no_eta, &env, &[]);
        }
        // ln p_k = logit_k − logsumexp(logits)
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lse = max + logits.iter().map(|l| (l - max).exp()).sum::<f64>().ln();
        logits.iter().map(|l| l - lse).collect()
    } else {
        let mut p = vec![0.0f64; k];
        let mut sum = 0.0;
        for m in &spec.mixing {
            let v = eval_expr(&m.expr, theta, &no_eta, &env, &[]).clamp(1e-12, 1.0 - 1e-12);
            p[m.class - 1] = v;
            sum += v;
        }
        p[k - 1] = (1.0 - sum).clamp(1e-12, 1.0);
        let tot: f64 = p.iter().sum();
        p.iter().map(|v| (v / tot).ln()).collect()
    }
}

/// Analytic `∂ ln p_ik / ∂ θ_j` for the mixing probabilities (#977 Phase 4),
/// returned as `K × n_theta`. Only the **logit** form is differentiated here
/// (`p = softmax(g)`), which is the general covariate-dependent case; the `p`
/// (direct-probability) form returns `None` so the caller routes to FD.
///
/// For the softmax: `∂ ln p_k/∂θ_j = ∂g_k/∂θ_j − Σ_m p_m ∂g_m/∂θ_j`, where the
/// per-class scores `g_k` are the mixing expressions (reference class `K` has
/// `g ≡ 0 ⇒ ∂ = 0`) and `∂g/∂θ_j` comes from the symbolic differentiator.
pub(crate) fn mixing_logp_grad(
    spec: &crate::types::MixtureSpec,
    theta: &[f64],
    covariates: &HashMap<String, f64>,
) -> Option<Vec<Vec<f64>>> {
    if !spec.mixing.iter().all(|m| m.is_logit) {
        return None; // p-form → FD fallback
    }
    let k = spec.n_classes;
    let nt = theta.len();
    let empty_vars: HashMap<String, f64> = HashMap::new();
    let env = MapEnv {
        covariates,
        vars: &empty_vars,
    };
    let no_eta: [f64; 0] = [];

    // Per-class logit value and its θ-gradient (reference class K stays 0). The
    // mixing expression is a cheap closed form over θ + covariates (no inner
    // solve, no random effects), so a central difference over θ is machine-exact
    // — and it avoids the symbolic differentiator, which expects resolved
    // covariate-index nodes the mixing AST never carries.
    let mut logit = vec![0.0f64; k];
    let mut dlogit = vec![vec![0.0f64; nt]; k];
    let mut tperturb = theta.to_vec();
    for m in &spec.mixing {
        let c = m.class - 1;
        logit[c] = eval_expr(&m.expr, theta, &no_eta, &env, &[]);
        for j in 0..nt {
            let h = 1e-6 * theta[j].abs().max(1.0);
            tperturb[j] = theta[j] + h;
            let fp = eval_expr(&m.expr, &tperturb, &no_eta, &env, &[]);
            tperturb[j] = theta[j] - h;
            let fm = eval_expr(&m.expr, &tperturb, &no_eta, &env, &[]);
            tperturb[j] = theta[j];
            dlogit[c][j] = (fp - fm) / (2.0 * h);
        }
    }
    // p = softmax(logit)
    let max = logit.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ex: Vec<f64> = logit.iter().map(|l| (l - max).exp()).collect();
    let s: f64 = ex.iter().sum();
    let p: Vec<f64> = ex.iter().map(|e| e / s).collect();

    // ∂ ln p_c/∂θ_j = dlogit[c][j] − Σ_m p_m dlogit[m][j]
    let mut out = vec![vec![0.0f64; nt]; k];
    for j in 0..nt {
        let wsum: f64 = (0..k).map(|m| p[m] * dlogit[m][j]).sum();
        for c in 0..k {
            out[c][j] = dlogit[c][j] - wsum;
        }
    }
    Some(out)
}

fn parse_parameters(
    lines: &[String],
) -> Result<
    (
        Vec<ThetaSpec>,
        Vec<OmegaSpec>,
        Vec<BlockOmegaSpec>,
        Vec<SigmaSpec>,
        Vec<BlockSigmaSpec>,
        Vec<String>,  // BSV eta names in declaration order
        ParsedKappas, // IOV kappa specs (diagonal and/or block)
    ),
    String,
> {
    let mut thetas = Vec::new();
    let mut omegas = Vec::new();
    let mut block_omegas = Vec::new();
    let mut sigmas = Vec::new();
    let mut block_sigmas = Vec::new();
    let mut eta_names_ordered = Vec::new();
    let mut kappas: Vec<KappaSpec> = Vec::new();
    let mut block_kappas: Vec<BlockKappaSpec> = Vec::new();
    let mut kappa_names_ordered: Vec<String> = Vec::new();
    let mut kappa_weights_ordered: Vec<Option<String>> = Vec::new();

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

    // block_sigma (NAME1, NAME2, ...) = [lower_triangle_covariances] | ... FIX
    //
    // Diagonal entries initialise the named sigma SDs; off-diagonal entries are
    // converted to fixed correlations so the existing positive sigma
    // parameterization remains intact.
    let block_sigma_re =
        Regex::new(r"(?i)block_sigma\s*\(([^)]+)\)\s*=\s*\[([^\]]+)\](?:\s+(FIX)\b)?").unwrap();

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

    // Rejoin multi-line `block_*` declarations before matching.
    let lines = join_bracketed_lines(lines);
    for line in &lines {
        // #1031: a `kappa` declaration may carry a trailing `weight = <expr>`
        // (sample-size-weighted IOV). It is peeled off before the declaration
        // regexes run because every one of them is unanchored — the modifier
        // would otherwise sit past the end of a successful match and be
        // silently dropped, which is the specific failure mode this feature
        // exists to remove. A weight left on any other declaration is rejected
        // below rather than ignored, for the same reason.
        let (line, weight_expr) =
            split_weight_modifier(line, "parameters", "a `kappa NAME ~ VALUE` declaration")?;
        let line = &line;
        let mut weight_consumed = false;
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
        } else if let Some(caps) = block_sigma_re.captures(line) {
            let names: Vec<String> = caps[1].split(',').map(|s| s.trim().to_string()).collect();
            let values: Vec<f64> = caps[2]
                .split(',')
                .map(|s| {
                    s.trim()
                        .parse::<f64>()
                        .map_err(|_| format!("Bad block_sigma value in: {}", line))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let n = names.len();
            let expected = n * (n + 1) / 2;
            if values.len() != expected {
                return Err(format!(
                    "block_sigma with {} sigmas expects {} lower-triangle values, got {}: {}",
                    n,
                    expected,
                    values.len(),
                    line
                ));
            }
            let fixed = caps.get(3).is_some();
            let mut val_idx = 0;
            for row in 0..n {
                for col in 0..=row {
                    let value = values[val_idx];
                    if row == col {
                        if value < 0.0 {
                            return Err(format!(
                                "block_sigma '{}' has a negative initial variance ({value}); variances must be non-negative",
                                names[row]
                            ));
                        }
                        sigmas.push(SigmaSpec {
                            name: names[row].clone(),
                            value: value.sqrt(),
                            fixed,
                            init_as_sd: false,
                        });
                    } else if !value.is_finite() {
                        return Err(format!("block_sigma covariance is non-finite in: {}", line));
                    }
                    val_idx += 1;
                }
            }
            block_sigmas.push(BlockSigmaSpec {
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
                kappa_weights_ordered.push(None);
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
            kappa_weights_ordered.push(weight_expr.clone());
            weight_consumed = true;
            kappas.push(KappaSpec {
                name,
                variance,
                fixed,
                init_as_sd,
            });
        }
        if let Some(w) = &weight_expr {
            if !weight_consumed {
                return Err(format!(
                    "[parameters] `weight = {w}` is only supported on a `kappa NAME ~ VALUE` \
                     declaration (it declares κ ~ N(0, Ω_IOV / W), the sample-size-weighted \
                     IOV of an MBMA arm effect). It has no meaning on `{}`.{} For a weighted \
                     residual error write the modifier on the `[error_model]` statement \
                     instead.",
                    line.trim(),
                    if block_kappa_re.is_match(line) {
                        " A `block_kappa` cannot carry one: the weight scales a single \
                         kappa's variance, and a block's off-diagonal covariances would \
                         need the same divisor to stay a valid covariance matrix — declare \
                         the weighted kappa as a diagonal `kappa` instead."
                    } else {
                        ""
                    }
                ));
            }
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
        block_sigmas,
        eta_names_ordered,
        ParsedKappas {
            diagonal: kappas,
            block: block_kappas,
            names_ordered: kappa_names_ordered,
            weights: kappa_weights_ordered,
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
        // Fixed-effects-only (naive-pooled) model: no random effects at all. The
        // 0×0 Omega is well-defined — the Cholesky of a 0-dim matrix is trivial and
        // log|Ω| = 0 — so FOCE/FOCEI collapse to the plain ML objective with no inner
        // optimisation (#989). This is the same path a `[event_model]` / `[binary_model]`
        // endpoint has always taken; it is not restricted to non-Gaussian endpoints.
        // `build_omega_fixed` likewise returns an empty `Vec` for an empty eta list.
        return Ok(OmegaMatrix::from_diagonal(&[], Vec::new()));
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

fn build_residual_correlations(
    block_sigmas: &[BlockSigmaSpec],
    sigma_names: &[String],
) -> Result<Vec<ResidualCorrelation>, String> {
    let name_to_idx: std::collections::HashMap<&str, usize> = sigma_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();
    let mut out = Vec::new();
    for block in block_sigmas {
        let n = block.names.len();
        let mut variances = vec![0.0; n];
        let mut pos = 0;
        for row in 0..n {
            for col in 0..=row {
                if row == col {
                    variances[row] = block.lower_triangle[pos];
                }
                pos += 1;
            }
        }
        pos = 0;
        for row in 0..n {
            for col in 0..=row {
                let value = block.lower_triangle[pos];
                if row != col && value != 0.0 {
                    let vi = variances[row];
                    let vj = variances[col];
                    if vi <= 0.0 || vj <= 0.0 {
                        return Err(format!(
                            "block_sigma covariance between '{}' and '{}' requires positive diagonal variances",
                            block.names[row], block.names[col]
                        ));
                    }
                    let rho = value / (vi.sqrt() * vj.sqrt());
                    if !(rho.is_finite() && (-1.0..=1.0).contains(&rho)) {
                        return Err(format!(
                            "block_sigma covariance between '{}' and '{}' implies invalid correlation {}",
                            block.names[row], block.names[col], rho
                        ));
                    }
                    let i = *name_to_idx.get(block.names[row].as_str()).ok_or_else(|| {
                        format!(
                            "block_sigma references unknown sigma '{}'",
                            block.names[row]
                        )
                    })?;
                    let j = *name_to_idx.get(block.names[col].as_str()).ok_or_else(|| {
                        format!(
                            "block_sigma references unknown sigma '{}'",
                            block.names[col]
                        )
                    })?;
                    out.push(ResidualCorrelation {
                        sigma_i: i,
                        sigma_j: j,
                        rho,
                    });
                }
                pos += 1;
            }
        }
        let _fixed = block.fixed;
    }
    Ok(out)
}

fn validate_residual_correlations(
    error_spec: &ErrorSpec,
    correlations: &[ResidualCorrelation],
    sigma_names: &[String],
) -> Result<(), String> {
    if correlations.is_empty() {
        return Ok(());
    }

    // Covariate-selected residual error (#658) with block_sigma correlated
    // residuals (#669): the pairing rule is the same as `PerCmt`. Each
    // observation resolves to an endpoint per-row via the covariate selector
    // (`ErrorSpec::obs_keys`), loads that endpoint's sigmas, and rows sharing a
    // subject time+occasion receive the `block_sigma` cross covariance from
    // `cross_observation_covariance`. Two paired rows resolving to *different*
    // branches get the honest cross-branch covariance `rho·σ_i·σ_j` built from
    // each row's own loadings — exactly the total/unbound cross-endpoint case
    // the `PerCmt` path already handles. The whole residual runtime is generic
    // over the endpoint key, so `Selected` needs no special casing here; the
    // reference-validity check below already dispatches through the shared
    // `PerCmt | Selected { endpoints: map, .. }` arm.
    let endpoint_loadings: Vec<Vec<usize>> = match error_spec {
        ErrorSpec::Single(_) => vec![error_spec
            .sigma_loadings(0, 1.0, sigma_names.len())
            .into_iter()
            .map(|(idx, _)| idx)
            .collect()],
        ErrorSpec::PerCmt(map) | ErrorSpec::Selected { endpoints: map, .. } => map
            .keys()
            .map(|&cmt| {
                error_spec
                    .sigma_loadings(cmt, 1.0, sigma_names.len())
                    .into_iter()
                    .map(|(idx, _)| idx)
                    .collect()
            })
            .collect(),
    };

    for corr in correlations {
        let left_used = endpoint_loadings
            .iter()
            .any(|loadings| loadings.contains(&corr.sigma_i));
        let right_used = endpoint_loadings
            .iter()
            .any(|loadings| loadings.contains(&corr.sigma_j));
        if !(left_used && right_used) {
            let left = sigma_names
                .get(corr.sigma_i)
                .map(String::as_str)
                .unwrap_or("<unknown>");
            let right = sigma_names
                .get(corr.sigma_j)
                .map(String::as_str)
                .unwrap_or("<unknown>");
            return Err(format!(
                "block_sigma covariance between '{}' and '{}' references a sigma \
                 that is not used by [error_model]",
                left, right
            ));
        }
    }
    Ok(())
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
    /// Covariate-selected error models (issue #658): an `if (cov …) { DV ~ … }
    /// [else if …]* else { DV ~ … }` block. `branches` holds the conditional
    /// endpoints in source order; `default` is the mandatory final `else`
    /// endpoint. `covariates` lists the covariate names the conditions
    /// reference (added to `referenced_covariates` so they become required data
    /// columns). Sigma references are still names here; `build_error_spec`
    /// resolves them to flat-sigma indices.
    Selected {
        branches: Vec<(Condition, ErrorModel, Vec<String>)>,
        default: (ErrorModel, Vec<String>),
        covariates: Vec<String>,
    },
}

/// Log-transform-both-sides (LTBS) flags extracted from the `[error_model]`
/// block. `log_transform` mirrors `CompiledModel::log_transform`; `dv_pre_logged`
/// mirrors `CompiledModel::dv_pre_logged`. Both default `false` (no LTBS).
#[derive(Clone, Copy, Default)]
struct LtbsFlags {
    log_transform: bool,
    dv_pre_logged: bool,
}

/// Split an `[error_model]` argument list on commas that sit at paren depth 0,
/// so a magnitude expression's own parenthesised commas stay within one
/// argument. Each argument is trimmed; empty fragments are dropped.
fn split_top_level_args(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth: i32 = 0;
    let mut start = 0usize;
    for (i, ch) in s.char_indices() {
        match ch {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                let part = s[start..i].trim();
                if !part.is_empty() {
                    out.push(part.to_string());
                }
                start = i + 1;
            }
            _ => {}
        }
    }
    let last = s[start..].trim();
    if !last.is_empty() {
        out.push(last.to_string());
    }
    out
}

/// Peel an optional trailing `weight = <expr>` modifier off a statement that
/// accepts one — an `[error_model]` residual (#1029) or a `[parameters]` kappa
/// declaration (#1031):
///
/// ```text
/// DV ~ additive(ADD_ERR) weight = WPSE
/// DV ~ combined(PROP_ERR, ADD_ERR) weight = 1 / sqrt(NARM)
/// kappa KAPPA_EMAX ~ 2.0 (sd) weight = NARM
/// ```
///
/// The `weight` keyword is recognised only at bracket depth 0 and only when
/// followed by a single `=`, so a weight expression carrying its own parens
/// survives intact and a covariate named `weight` *inside* a magnitude
/// expression (#484) is not mistaken for the modifier. Returns
/// `(statement_without_modifier, Some(weight_expr))`, or `(statement, None)`
/// when no modifier is present. `block` and `stmt_desc` name the block and the
/// statement form the modifier attaches to, so each caller's diagnostics read
/// as its own.
fn split_weight_modifier(
    body: &str,
    block: &str,
    stmt_desc: &str,
) -> Result<(String, Option<String>), String> {
    const KW: &[u8] = b"weight";
    let bytes = body.as_bytes();
    let mut depth: i32 = 0;
    let mut found: Option<usize> = None;
    for i in 0..bytes.len() {
        match bytes[i] {
            b'(' | b'[' => {
                depth += 1;
                continue;
            }
            b')' | b']' => {
                depth -= 1;
                continue;
            }
            _ => {}
        }
        if depth != 0 || i + KW.len() > bytes.len() {
            continue;
        }
        if !bytes[i..i + KW.len()].eq_ignore_ascii_case(KW) {
            continue;
        }
        // Whole-word match on both sides, so `WEIGHTED` / `X_weight` are not it.
        let is_ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
        if i > 0 && is_ident(bytes[i - 1]) {
            continue;
        }
        if bytes.get(i + KW.len()).copied().is_some_and(is_ident) {
            continue;
        }
        let eq = skip_ascii_ws(body, i + KW.len());
        // `==` is a comparison, not the modifier's assignment.
        if bytes.get(eq) != Some(&b'=') || bytes.get(eq + 1) == Some(&b'=') {
            continue;
        }
        found = Some(i);
        break;
    }
    let Some(i) = found else {
        return Ok((body.to_string(), None));
    };
    let eq = skip_ascii_ws(body, i + KW.len());
    let expr = body[eq + 1..].trim();
    if expr.is_empty() {
        return Err(format!(
            "[{block}] `weight =` has no expression on its right-hand side: `{}`",
            body.trim()
        ));
    }
    let stmt = body[..i].trim();
    if stmt.is_empty() {
        return Err(format!(
            "[{block}] `weight = {}` must follow {stmt_desc} on the same line",
            expr
        ));
    }
    Ok((stmt.to_string(), Some(expr.to_string())))
}

/// Does `s[i..]` begin with the keyword `kw`, terminated by a non-identifier
/// character (so `if` matches but `iffy` does not)?
fn starts_with_keyword(s: &str, i: usize, kw: &str) -> bool {
    let rest = &s[i..];
    if !rest.starts_with(kw) {
        return false;
    }
    match rest[kw.len()..].chars().next() {
        Some(c) => !(c.is_alphanumeric() || c == '_'),
        None => true,
    }
}

/// Advance past ASCII whitespace (spaces, tabs, newlines) from byte offset `i`.
fn skip_ascii_ws(s: &str, i: usize) -> usize {
    let bytes = s.as_bytes();
    let mut i = i;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    i
}

/// Given `s` where byte `open` is an opening delimiter `open_ch`, return the
/// byte index of the matching closing delimiter `close_ch`, honouring nesting.
fn match_delim(s: &str, open: usize, open_ch: u8, close_ch: u8) -> Result<usize, String> {
    let bytes = s.as_bytes();
    debug_assert_eq!(bytes[open], open_ch);
    let mut depth = 0i32;
    let mut i = open;
    while i < bytes.len() {
        let b = bytes[i];
        if b == open_ch {
            depth += 1;
        } else if b == close_ch {
            depth -= 1;
            if depth == 0 {
                return Ok(i);
            }
        }
        i += 1;
    }
    Err(format!(
        "[error_model] unbalanced `{}` … `{}` in if/else block",
        open_ch as char, close_ch as char
    ))
}

/// Parse one `DV ~ TYPE(sigmas)` statement that forms an if/else branch body
/// into `(ErrorModel, sigma_names)`. Rejects LTBS forms (`log(DV) ~ …` /
/// `log_additive`) and multi-statement bodies — a covariate-selected branch is
/// a single ordinary error model (LTBS + covariate selection is out of scope).
fn parse_error_endpoint_body(body: &str) -> Result<(ErrorModel, Vec<String>), String> {
    let body = body.trim();
    // #1029: `weight = …` would otherwise be swallowed by the statement regex's
    // failure path and silently ignored inside a branch.
    if split_weight_modifier(body, "error_model", "a `DV ~ TYPE(...)` statement")?
        .1
        .is_some()
    {
        return Err(
            "[error_model] `weight = …` is not supported inside a covariate-selected if/else \
             block"
                .to_string(),
        );
    }
    let log_lhs_re = Regex::new(r"^\s*log\s*\(\s*\w+\s*\)\s*~").unwrap();
    if log_lhs_re.is_match(body) {
        return Err(
            "[error_model] log-transform-both-sides (`log(DV) ~ …`) is not supported inside a \
             covariate-selected if/else block"
                .to_string(),
        );
    }
    let re = Regex::new(r"^(\w+)\s*~\s*(\w+)\s*\((.+)\)$").unwrap();
    let caps = re.captures(body).ok_or_else(|| {
        format!(
            "[error_model] expected a single `DV ~ TYPE(...)` statement inside an if/else branch, \
             got `{}`",
            body
        )
    })?;
    let error_type = caps[2].to_lowercase();
    if error_type == "log_additive" {
        return Err(
            "[error_model] `log_additive` (log-transform-both-sides) is not supported inside a \
             covariate-selected if/else block"
                .to_string(),
        );
    }
    let error_model = match error_type.as_str() {
        "additive" => ErrorModel::Additive,
        "proportional" => ErrorModel::Proportional,
        "combined" => ErrorModel::Combined,
        other => return Err(format!("Unknown error model: {}", other)),
    };
    let sigma_names = split_top_level_args(&caps[3]);
    if sigma_names.len() != error_model.n_sigma() {
        return Err(format!(
            "[error_model] {} model expects {} sigma(s) but {} given: {}",
            error_type,
            error_model.n_sigma(),
            sigma_names.len(),
            body
        ));
    }
    Ok((error_model, sigma_names))
}

/// Detect and parse a covariate-selected `[error_model]` (issue #658):
///
/// ```text
/// if (FREE == 0) { DV ~ proportional(PROP_TOTAL) }
/// else           { DV ~ proportional(PROP_UNBOUND) }
/// ```
///
/// Returns `Ok(None)` when the block is not an `if/else` form (the caller then
/// falls back to the line-based single / per-CMT parsing). The condition uses
/// the shared `parse_condition`; each branch body reuses the ordinary
/// `DV ~ TYPE(...)` grammar via [`parse_error_endpoint_body`]. A final `else`
/// is required so every observation resolves to a declared endpoint.
#[allow(clippy::type_complexity)]
fn parse_selected_error_model(
    lines: &[String],
) -> Result<
    Option<(
        Vec<(Condition, ErrorModel, Vec<String>)>,
        (ErrorModel, Vec<String>),
        Vec<String>,
    )>,
    String,
> {
    // Join comment-stripped lines; the if/else spans multiple physical lines.
    let joined: String = lines
        .iter()
        .map(|l| l.split('#').next().unwrap_or("").trim_end())
        .collect::<Vec<_>>()
        .join("\n");
    let s = joined.trim().to_string();
    let start = skip_ascii_ws(&s, 0);
    if !starts_with_keyword(&s, start, "if") {
        return Ok(None);
    }

    // Covariate-only condition context: every bare identifier is a covariate.
    let no_names: [String; 0] = [];
    let no_nn: [(String, Vec<String>); 0] = [];
    let cond_ctx = ParseCtx {
        theta_names: &no_names,
        eta_names: &no_names,
        defined_vars: &no_names,
        fallback_covariate: true,
        nn_specs: &no_nn,
        ode_state_names: &no_names,
    };

    let bytes = s.as_bytes();
    let mut i = start;
    let mut branches: Vec<(Condition, ErrorModel, Vec<String>)> = Vec::new();
    let default: (ErrorModel, Vec<String>);

    loop {
        // `if`
        i = skip_ascii_ws(&s, i);
        if !starts_with_keyword(&s, i, "if") {
            return Err("[error_model] expected `if` in covariate-selected block".to_string());
        }
        i += 2;
        // `( cond )`
        i = skip_ascii_ws(&s, i);
        if i >= bytes.len() || bytes[i] != b'(' {
            return Err("[error_model] expected `(` after `if`".to_string());
        }
        let close = match_delim(&s, i, b'(', b')')?;
        let cond_str = &s[i + 1..close];
        let toks = tokenize(cond_str)?;
        let (cond, cp) = parse_condition(&toks, 0, cond_ctx)?;
        if skip_newlines(&toks, cp) != toks.len() {
            return Err(format!(
                "[error_model] trailing tokens in if-condition `{}`",
                cond_str
            ));
        }
        i = close + 1;
        // `{ body }`
        i = skip_ascii_ws(&s, i);
        if i >= bytes.len() || bytes[i] != b'{' {
            return Err("[error_model] expected `{` after if-condition".to_string());
        }
        let bclose = match_delim(&s, i, b'{', b'}')?;
        let (em, names) = parse_error_endpoint_body(&s[i + 1..bclose])?;
        branches.push((cond, em, names));
        i = bclose + 1;

        // `else` (required — either `else if` chains or `else { … }` terminates)
        i = skip_ascii_ws(&s, i);
        if !starts_with_keyword(&s, i, "else") {
            return Err(
                "[error_model] a covariate-selected if/else must end with a final `else { DV ~ … }` \
                 so every observation maps to an error model"
                    .to_string(),
            );
        }
        i += 4;
        i = skip_ascii_ws(&s, i);
        if starts_with_keyword(&s, i, "if") {
            continue; // else-if: parse another branch
        }
        if i >= bytes.len() || bytes[i] != b'{' {
            return Err(
                "[error_model] `else` must be followed by `if (...) {...}` or `{...}`".to_string(),
            );
        }
        let eclose = match_delim(&s, i, b'{', b'}')?;
        default = parse_error_endpoint_body(&s[i + 1..eclose])?;
        i = eclose + 1;
        break;
    }

    let tail = skip_ascii_ws(&s, i);
    if tail != s.len() {
        return Err(format!(
            "[error_model] unexpected content after covariate-selected if/else: `{}`",
            s[tail..].trim()
        ));
    }

    // Covariate names referenced by any branch condition become required data
    // columns (validated as E_MISSING_COVARIATE at fit setup).
    let mut cov_set: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (cond, _, _) in &branches {
        visit_condition_nodes(cond, &mut |e| {
            if let Expression::Covariate(name) = e {
                cov_set.insert(name.clone());
            }
        });
    }
    let mut covariates: Vec<String> = cov_set.into_iter().collect();
    covariates.sort();

    Ok(Some((branches, default, covariates)))
}

fn parse_error_model(
    lines: &[String],
) -> Result<(ParsedErrorModel, LtbsFlags, Option<String>), String> {
    // Covariate-selected if/else form (issue #658) — detected and parsed first;
    // falls through to the line-based single / per-CMT grammar when absent.
    if let Some((branches, default, covariates)) = parse_selected_error_model(lines)? {
        return Ok((
            ParsedErrorModel::Selected {
                branches,
                default,
                covariates,
            },
            LtbsFlags::default(),
            None,
        ));
    }
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
    // The argument list is captured greedily up to the final `)` so magnitude
    // expressions with their own nested parens — `combined(PROP*(if (TIME>24)
    // T else 1), ADD)` (#484) — survive intact; top-level commas are split out
    // below with `split_top_level_args`.
    let re = Regex::new(r"^(\w+)\s*~\s*(\w+)\s*\((.+)\)\s*$").unwrap();
    // LTBS LHS `log(DV) ~ TYPE(SIGMA)` — captures the logged data column,
    // the error type, and the sigma list.
    let log_lhs_re = Regex::new(r"^\s*log\s*\(\s*(\w+)\s*\)\s*~\s*(\w+)\s*\((.+)\)\s*$").unwrap();
    let cmt_re = Regex::new(r"^\s*CMT\s*=\s*(\d+)\s*:\s*(.*)$").unwrap();

    // singles carry the per-line LTBS flags so the chosen single can stamp them.
    let mut singles: Vec<(ErrorModel, Vec<String>, LtbsFlags)> = Vec::new();
    let mut per_cmt: Vec<(usize, ErrorModel, Vec<String>)> = Vec::new();
    // IIV on residual error: `iiv_on_ruv = ETA_NAME` (NONMEM `Y=IPRED+EPS*EXP(ETA)`).
    let iiv_re = Regex::new(r"(?i)^\s*iiv_on_ruv\s*=\s*(\w+)\s*$").unwrap();
    let mut iiv_on_ruv: Option<String> = None;

    for line in lines {
        // Strip inline `# ...` comments so the anchored statement regex sees a
        // clean `LHS ~ TYPE(args)` (a trailing comment would otherwise defeat
        // the `$` anchor and the statement would be silently ignored).
        let trimmed = line.split('#').next().unwrap_or("").trim();
        if trimmed.is_empty() {
            continue;
        }

        // `iiv_on_ruv = ETA_NAME` declares a random effect on the residual error.
        if let Some(c) = iiv_re.captures(trimmed) {
            if iiv_on_ruv.is_some() {
                return Err("[error_model] has more than one `iiv_on_ruv = ...` entry".to_string());
            }
            iiv_on_ruv = Some(c[1].to_string());
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

        // Inverse-variance weighting (#1029): `… weight = <expr>` is peeled off
        // before the statement regexes run — its trailing `= expr` would defeat
        // their `$` anchor and the whole line would be silently ignored.
        let (body, weight_expr) =
            split_weight_modifier(&body, "error_model", "a `DV ~ TYPE(...)` statement")?;

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
        // Split on top-level commas only, so a magnitude expression's own
        // commas (e.g. inside a future `min(a, b)`) don't fragment an argument.
        let mut sigma_names: Vec<String> = split_top_level_args(&caps[3]);
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

        // Inverse-variance weighting (#1029). `weight = W` declares that each
        // observation carries its own precision — the study-as-subject MBMA
        // construction where the observed quantity is a trial-level mean with a
        // known standard error. It is defined as "divide DV and the prediction
        // by `W`, score on that scale": with `y' = y/W` and `f' = f/W` the
        // natural-scale variance of the residual is `W² · Var(f/W)`, i.e. the
        // *additive* loading picks up a factor `W` while the proportional
        // loading is untouched (`W · (f/W) = f` — a common scale factor cancels
        // out of a CV). So the modifier desugars exactly into a #484
        // per-observation residual magnitude on the additive slot, reusing that
        // channel's compiler, θ-derivative program, and runtime plumbing
        // wholesale; DV, PRED, IPRED, CWRES and the VPC all stay on the natural
        // scale with no back-transformation.
        if let Some(w) = &weight_expr {
            if cmt_opt.is_some() {
                return Err(
                    "[error_model] `weight = …` is not supported with per-CMT (multi-endpoint) \
                     error models"
                        .to_string(),
                );
            }
            // Combined's argument order is (proportional, additive) — see
            // `ErrorModel::sigma_types` — so the weighted slot is the last one.
            let slot = match error_model {
                ErrorModel::Additive => 0,
                ErrorModel::Combined => 1,
                ErrorModel::Proportional => {
                    return Err(format!(
                        "[error_model] `weight = {}` has no effect on a purely proportional error \
                         model: weighting divides both DV and the prediction by the weight, and a \
                         common scale factor cancels out of a proportional (constant-CV) error. \
                         Use `additive(...)` or `combined(...)` — or drop the `weight =`.",
                        w
                    ));
                }
            };
            sigma_names[slot] = format!("({}) * ({})", sigma_names[slot], w);
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
        if iiv_on_ruv.is_some() {
            return Err("[error_model] `iiv_on_ruv` is not supported with per-CMT \
                        (multi-endpoint) error models"
                .to_string());
        }
        return Ok((
            ParsedErrorModel::PerCmt(per_cmt),
            LtbsFlags::default(),
            None,
        ));
    }

    match singles.into_iter().next() {
        Some((model, names, flags)) => {
            Ok((ParsedErrorModel::Single(model, names), flags, iiv_on_ruv))
        }
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
        ParsedErrorModel::Single(model, args) => {
            // Single-endpoint sigmas are consumed positionally from the global
            // sigma vector: `ErrorSpec::Single` carries no indices, so every
            // consumer (`sigma_loadings`, `residual_variance`, `sigma_types`,
            // the #484 magnitude multipliers, …) reads the *leading* slots.
            //
            // Each argument must therefore scale exactly one declared sigma —
            // `arg_sigma_name` resolves that sigma and rejects an unknown/typo'd
            // reference — **and** must be the sigma that actually occupies its
            // slot. Before #1001 the resolved name was discarded here, so
            // `proportional(S_SMALL)` with `S_BIG` declared first validated,
            // fitted against `S_BIG`, and reported nothing: the written name was
            // checked for existence and then ignored. The per-CMT and
            // covariate-selected arms below bind strictly by name, so without
            // this the same file meant two different things depending on which
            // `[error_model]` form it used.
            //
            // Rejecting rather than binding by name keeps one resolution rule
            // for the whole positional path (including `block_sigma`, whose
            // entries land in the same flat vector — this check generalises the
            // block_sigma-only guard it replaces).
            //
            // Two distinct defects, kept apart because their remedies are:
            // repeating a name is a *count* problem no permutation can fix,
            // while a genuine order mismatch is fixed by reordering one list.
            // `claimed` tracks the sigmas earlier arguments already took so the
            // repeat is caught first — otherwise `combined(S_A, S_A)` with two
            // sigmas declared falls into the order arm below and is told to
            // reorder lists that cannot be reordered into agreement.
            let mut claimed: Vec<String> = Vec::with_capacity(args.len());
            for (idx, arg) in args.iter().enumerate() {
                let named = arg_sigma_name(arg, sigma_names)?;
                // Duplicate `sigma` declarations are rejected upstream, so no
                // declaration order can put one name in two slots and no
                // argument order can either — say what is actually wrong, and
                // leave this on the generic `E_PARSE` rather than the
                // reorder-shaped `E_SIGMA_ORDER_MISMATCH` a consumer would act
                // on. Both reachable shapes (a second sigma declared or not)
                // report the same defect with the same code; the counts are
                // interpolated so the message stays honest if `args.len()` is
                // ever not pinned to the model's `n_sigma()`.
                if let Some(first) = claimed.iter().position(|c| *c == named) {
                    return Err(format!(
                        "[error_model] argument {} names sigma '{}', which argument \
                         {} already claims; each argument of a single-endpoint \
                         [error_model] must name a distinct declared sigma (this \
                         model needs {}, [parameters] declares {}). Give argument \
                         {} its own sigma.",
                        idx + 1,
                        named,
                        first + 1,
                        args.len(),
                        sigma_names.len(),
                        idx + 1
                    ));
                }
                match sigma_names.get(idx) {
                    Some(slot) if *slot == named => {}
                    // Both remedies are spelled out because both have a silent
                    // failure mode. `block_sigma` pushes its diagonals along the
                    // written name order, so reordering the names without
                    // permuting the lower triangle swaps the two variances and
                    // parses clean. And reordering the *arguments* of a
                    // `combined` model swaps which sigma is the proportional
                    // component — it makes the file honest about a model the
                    // user did not write, which is the #1001 failure mode again.
                    //
                    // `None` is folded in here rather than given its own arm.
                    // The repeat guard above leaves every `named` distinct and
                    // declared, so by argument `idx` at least `idx + 1` sigmas
                    // exist and `get(idx)` is always `Some` — an arm no input can
                    // reach is also an arm no test can cover, and it would sit in
                    // the diff as permanently-missed lines. Binding the whole
                    // `Option` keeps the impossible case a message rather than an
                    // `unreachable!()` panic in the parser.
                    slot => {
                        return Err(format!(
                            "[error_model] argument {} names sigma '{}', but a \
                             single-endpoint [error_model] has its sigmas consumed \
                             positionally from the [parameters] declaration order, \
                             which supplies '{}' in that position. Reorder the sigma \
                             declarations to match the [error_model] argument order \
                             — if a block_sigma supplies them, permute its lower \
                             triangle to match the new name order. Reordering the \
                             arguments instead also parses, but for `combined` it \
                             swaps which sigma is the proportional component, so it \
                             changes the model rather than fixing the spelling.",
                            idx + 1,
                            named,
                            slot.map(String::as_str).unwrap_or("<none>")
                        ));
                    }
                }
                claimed.push(named);
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
        ParsedErrorModel::Selected {
            branches, default, ..
        } => {
            // Resolve one endpoint's sigma names to flat-sigma indices.
            let resolve = |em: ErrorModel, names: &[String]| -> Result<EndpointError, String> {
                let mut sigma_idx = Vec::with_capacity(names.len());
                for nm in names {
                    let idx = sigma_names.iter().position(|s| s == nm).ok_or_else(|| {
                        format!(
                            "[error_model] references unknown sigma '{}' \
                             (declare it in [parameters])",
                            nm
                        )
                    })?;
                    sigma_idx.push(idx);
                }
                Ok(EndpointError {
                    error_model: em,
                    sigma_idx,
                })
            };
            // Assign 0-based keys in source order; the `else` endpoint is the
            // final key. The selector evaluates conditions in order and returns
            // the first match, falling back to the `else` key.
            let representative = branches.first().map(|(_, em, _)| *em).unwrap_or(default.0);
            let mut endpoints: HashMap<usize, EndpointError> = HashMap::new();
            let mut conds: Vec<(Condition, usize)> = Vec::with_capacity(branches.len());
            let mut labels: Vec<String> = Vec::with_capacity(branches.len() + 1);
            for (k, (cond, em, names)) in branches.into_iter().enumerate() {
                endpoints.insert(k, resolve(em, &names)?);
                labels.push(format!("branch{k}"));
                conds.push((cond, k));
            }
            let default_key = endpoints.len();
            endpoints.insert(default_key, resolve(default.0, &default.1)?);
            labels.push("else".to_string());
            let selector = crate::types::ErrorSelector {
                eval: Box::new(move |cov: &HashMap<String, f64>| {
                    for (cond, key) in &conds {
                        if eval_condition(cond, &[], &[], cov, &HashMap::new(), &[]) {
                            return *key;
                        }
                    }
                    default_key
                }),
                branch_labels: labels,
            };
            Ok((
                representative,
                ErrorSpec::Selected {
                    selector,
                    endpoints,
                },
            ))
        }
    }
}

/// Reserved built-in names a residual-magnitude expression (#484) may **not**
/// reference in Phase 1. `IPRED`/`PRED`/`DV` would make the magnitude depend on
/// the prediction beyond the built-in proportional loading; `TAD`/`TAFD` are
/// not yet plumbed per-observation. `TIME` is the one allowed built-in.
const RUV_FORBIDDEN_NAMES: &[&str] = &[
    "IPRED", "PRED", "DV", "TAD", "TAFD", "IRES", "IWRES", "CWRES",
];

/// Validate a parsed residual-magnitude expression (#484). The magnitude may
/// depend only on θ, the scaled sigma itself, `TIME`, and declared covariates —
/// never on η/EBE or the prediction. Unlike the rest of the parser, covariate
/// references here are **not** read leniently: an undeclared name silently
/// evaluates to `0.0` (`covariates.get(name).unwrap_or(0.0)`), which would
/// collapse the multiplier to a constant with no error — so any covariate the
/// expression names (other than `TIME`) must appear in `allowed_covs`. A
/// `allowed_covs == None` (no `[covariates]` block) therefore rejects every
/// covariate reference, forcing the user to declare it.
fn validate_ruv_expr(
    expr: &Expression,
    sigma_name: &str,
    allowed_covs: Option<&[String]>,
) -> Result<(), String> {
    let mut err: Option<String> = None;
    visit_expr_nodes(expr, &mut |e: &Expression| {
        if err.is_some() {
            return;
        }
        match e {
            Expression::Eta(_) => {
                err = Some(
                    "residual-magnitude expression may not depend on a random effect (eta)"
                        .to_string(),
                );
            }
            Expression::NnOutput { .. } => {
                err = Some(
                    "residual-magnitude expression may not reference a neural-network output"
                        .to_string(),
                );
            }
            Expression::Variable(name) => {
                // The only individual-scope variable allowed is the sigma the
                // expression scales; anything else (an individual parameter)
                // would make the magnitude eta-dependent.
                if !name.eq_ignore_ascii_case(sigma_name) && !name.eq_ignore_ascii_case("MACHEPS") {
                    err = Some(format!(
                        "residual-magnitude expression references `{}`, which is not \
                         the scaled sigma `{}`, a covariate, or TIME",
                        name, sigma_name
                    ));
                }
            }
            Expression::Covariate(name) => {
                let upper = name.to_uppercase();
                if RUV_FORBIDDEN_NAMES.contains(&upper.as_str()) {
                    err = Some(format!(
                        "residual-magnitude expression may not reference `{}` \
                         (only TIME, covariates, thetas, and the sigma are allowed)",
                        name
                    ));
                } else if !name.eq_ignore_ascii_case("TIME") {
                    // An undeclared covariate silently evaluates to 0.0 at every
                    // observation, collapsing the magnitude to a constant — so
                    // require declaration whether or not a [covariates] block
                    // exists (`allowed_covs == None` ⇒ nothing is declared ⇒
                    // reject). This also catches covariate-name typos.
                    let declared = allowed_covs
                        .is_some_and(|covs| covs.iter().any(|c| c.eq_ignore_ascii_case(name)));
                    if !declared {
                        err = Some(format!(
                            "residual-magnitude expression references undeclared covariate \
                             `{}` (declare it in [covariates]; an undeclared name silently \
                             evaluates to 0 and would make the magnitude a constant)",
                            name
                        ));
                    }
                }
            }
            _ => {}
        }
    });
    match err {
        Some(e) => Err(format!("[error_model] {}", e)),
        None => Ok(()),
    }
}

/// Lower a parsed magnitude expression to a `Dual1`-differentiable
/// [`RuvMagDerivProgram`] over `θ` only (#576/#486): the magnitude is validated
/// ([`validate_ruv_expr`]) to never reference `η`, an individual parameter, or an
/// NN output, so — unlike [`compile_scale_deriv_program`] — there is no `η` axis
/// and no `var_to_pk_slot` to feed in. The scaled sigma resolves to var slot `0`,
/// pinned to the constant `1.0` at eval time (mirroring the runtime closure's
/// `vars.insert(sigma_name, 1.0)`); covariates are sorted cov slots; `TIME` reads
/// the event-time built-in. The input `expr` is cloned, so the caller keeps its
/// AST for the runtime closure.
fn compile_ruv_mag_deriv_program(
    expr: &Expression,
    sigma_name: &str,
    n_theta: usize,
) -> RuvMagDerivProgram {
    let mut e = expr.clone();
    // Slot 0 = the pinned scaled sigma (bound to 1.0 at eval time). Slot 1 =
    // `MACHEPS`, which `validate_ruv_expr` explicitly permits inside a magnitude
    // expression alongside the scaled sigma. `eval_expression` (the runtime
    // f64 closure) special-cases the `MACHEPS` name string at eval time, but
    // bytecode compilation resolves every `Variable` name to a fixed slot up
    // front — so without a reserved slot here, `resolve_expr_indices` would map
    // `MACHEPS` to `VariableIdx(usize::MAX)`, which silently evaluates to `0.0`
    // instead of `f64::EPSILON` (mirrors the ODE var builder's `macheps_slot`).
    let var_idx: HashMap<String, usize> = [
        (sigma_name.to_string(), 0),
        ("MACHEPS".to_string(), 1),
        ("macheps".to_string(), 1),
    ]
    .into_iter()
    .collect();
    let mut cov_set = std::collections::HashSet::new();
    collect_covariates(&e, &mut cov_set);
    let mut cov_names: Vec<String> = cov_set.into_iter().collect();
    cov_names.sort();
    let cov_idx: HashMap<String, usize> = cov_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();
    resolve_expr_indices(&mut e, &var_idx, &cov_idx);
    RuvMagDerivProgram {
        bc: compile_bytecode(&e),
        n_theta,
        cov_names,
    }
}

/// Build the custom residual-error magnitude (#484) from the single-endpoint
/// `[error_model]` arguments, once covariate names are known. Returns `(None, {})`
/// when every argument is a bare sigma (the legacy variance path is then taken
/// verbatim). Each non-bare argument compiles to a per-observation multiplier
/// closure over `(theta, observation covariates, TIME)`, plus a `Dual1`
/// θ-derivative program (#576/#486) so the analytic outer θ/σ gradient can
/// differentiate the magnitude exactly instead of falling back to FD.
///
/// The second element is the set of theta indices any magnitude expression
/// references — a theta that appears *only* in `[error_model]` (e.g. `RUV_LATE`)
/// would otherwise trip `check_unused_parameters`'s "not referenced" warning,
/// mirroring how `[event_model]` thetas are unioned in.
fn build_ruv_magnitude(
    single_error_args: &Option<(ErrorModel, Vec<String>)>,
    sigma_names: &[String],
    theta_names: &[String],
    eta_names: &[String],
    covariate_decls: &Option<Vec<CovariateDecl>>,
) -> Result<
    (
        Option<RuvMagnitude>,
        std::collections::HashSet<usize>,
        Vec<String>,
    ),
    String,
> {
    let mut used_thetas = std::collections::HashSet::new();
    let mut used_covs: std::collections::HashSet<String> = std::collections::HashSet::new();
    let Some((_em, args)) = single_error_args else {
        return Ok((None, used_thetas, Vec::new()));
    };
    let allowed_covs: Option<Vec<String>> = covariate_decls
        .as_ref()
        .map(|d| d.iter().map(|c| c.name.clone()).collect());
    let mut per_sigma: Vec<Option<RuvMagFn>> = Vec::with_capacity(args.len());
    let mut per_sigma_deriv: Vec<Option<RuvMagDerivProgram>> = Vec::with_capacity(args.len());
    let mut any = false;
    for arg in args {
        let sigma_name = arg_sigma_name(arg, sigma_names)?;
        let trimmed = arg.trim();
        if trimmed == sigma_name {
            per_sigma.push(None);
            per_sigma_deriv.push(None);
            continue;
        }
        any = true;
        // The scaled sigma resolves to a `Variable` (bound to 1.0 at eval);
        // unknown identifiers fall back to model covariates. TIME/time parse
        // as event-time built-ins. `MACHEPS`/`macheps` must resolve to a
        // `Variable` too (not fall through to `Covariate`) — `validate_ruv_expr`'s
        // `Variable` arm already permits it, but without a `defined_vars` entry
        // the parser would classify it as an (undeclared, rejected) covariate
        // instead, since this context sets `fallback_covariate: true` (#486 review).
        let defined = [
            sigma_name.clone(),
            "MACHEPS".to_string(),
            "macheps".to_string(),
        ];
        let ctx = ParseCtx {
            theta_names,
            eta_names,
            defined_vars: &defined,
            fallback_covariate: true,
            nn_specs: &[],
            ode_state_names: &[],
        };
        let expr = parse_scalar_expression(trimmed, ctx)
            .map_err(|e| format!("[error_model] magnitude `{}`: {}", trimmed, e))?;
        validate_ruv_expr(&expr, &sigma_name, allowed_covs.as_deref())?;
        visit_expr_nodes(&expr, &mut |e: &Expression| {
            if let Expression::Theta(i) = e {
                used_thetas.insert(*i);
            }
        });
        collect_covariates(&expr, &mut used_covs);
        per_sigma_deriv.push(Some(compile_ruv_mag_deriv_program(
            &expr,
            &sigma_name,
            theta_names.len(),
        )));
        let sig = sigma_name.clone();
        let f: RuvMagFn = Box::new(move |theta, obs_cov, time| {
            let mut vars: HashMap<String, f64> = HashMap::new();
            vars.insert(sig.clone(), 1.0);
            // Preserve the legacy covariate-map injection for any pre-built
            // expression that still treats TIME as a covariate; parsed models
            // now use the event-time built-in via `with_model_time`.
            let mut cov = obs_cov.clone();
            cov.insert("TIME".to_string(), time);
            cov.insert("time".to_string(), time);
            with_model_time(time, || {
                eval_expression(&expr, theta, &[], &cov, &vars, &[])
            })
        });
        per_sigma.push(Some(f));
    }
    let rm = if any {
        Some(RuvMagnitude {
            per_sigma,
            per_sigma_deriv,
            theta_dependent: !used_thetas.is_empty(),
        })
    } else {
        None
    };
    let mut covs: Vec<String> = used_covs.into_iter().collect();
    covs.sort();
    Ok((rm, used_thetas, covs))
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
    // Analytical modeled-dose (`D{cmt}` RATE=-2 duration, `R{cmt}` RATE=-1 rate)
    // parameters as `(var_name, PkParams slot)`; the closure writes each value
    // into its reserved spare slot in the analytical arm. Empty for ODE models
    // and for analytical models with no `RATE=-1`/`-2` dosing. See #324.
    analytical_modeled_slots: &[(String, usize)],
    // Non-structural individual parameters an analytic Form C readout references,
    // as `(var_name, PkParams slot)` (#650). Unlike `analytical_modeled_slots`,
    // these are merged into `pk_assignment_mapping` so they are BOTH written by the
    // closure AND carried in the program's `pk_var_slots` — the readout needs their
    // exact `∂/∂(θ,η)`. Empty for ODE models and readout-free models.
    readout_extra_slots: &[(String, usize)],
    n_theta_base: usize,
    n_eta_extended: usize,
    #[cfg(feature = "nn")] covariate_nns: &[crate::nn::CovariateNn],
) -> Result<
    (
        PkParamFn,
        Vec<String>,
        IndivParamPartials,
        IndivParamProgram,
    ),
    String,
> {
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

    // Detect use of the `TIME` built-in BEFORE `resolve_variable_indices`
    // bytecode-compiles the statements: that pass replaces each `Expression::Time`
    // node with an `Op::PushTime` bytecode op and an `Expression::Literal(0.0)`
    // placeholder, after which `stmts_use_time_builtin` (an AST walker) can no
    // longer see it. Computing the flag here, on the intact AST, is what makes a
    // time-dependent individual parameter expressed as a conditional RHS
    // (`STEP = if (TIME > 5) 1 else 0`) route through the per-event evaluation
    // path. (A genuine `if`-statement block survives resolution and was detected
    // either way; the conditional-expression form regressed silently — #610.)
    let stmts_use_time = stmts_use_time_builtin(&stmts_resolved);

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
        // A direct `pk(...=TIME)` binding is desugared into a synthetic
        // `__ferx_pktime_*` individual parameter upstream in `parse_full_model` (#486),
        // so `var_name` is that synthetic name here (resolved via `var_idx` below), never
        // a literal `TIME`.
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

    // #650: merge readout-referenced non-structural individual parameters into the
    // structural mapping, so they are written by the closure below AND land in the
    // program's `pk_var_slots` (line 9142 clones `pk_assignment_mapping`) — giving
    // the analytic sensitivity provider their exact `∂/∂(θ,η)` for the readout.
    for (var_name, pk_slot) in readout_extra_slots {
        if let Some(var_slot) = var_idx
            .get(var_name)
            .copied()
            .or_else(|| var_idx.get(&var_name.to_lowercase()).copied())
        {
            pk_assignment_mapping.push((*pk_slot, var_slot));
        }
    }

    // Resolve each analytical modeled-dose parameter's value slot once, to
    // `(PkParams write slot, var slot)`, so the hot closure is two array reads.
    // Empty unless this is an analytical model with `D{cmt}`/`R{cmt}` (RATE=-2/-1)
    // declared. A name with no matching var slot is dropped (defensive — it would
    // have errored earlier as an undefined reference).
    let analytical_extra_mapping: Vec<(usize, usize)> = analytical_modeled_slots
        .iter()
        .filter_map(|(var_name, pk_slot)| {
            let var_slot = var_idx
                .get(var_name)
                .copied()
                .or_else(|| var_idx.get(&var_name.to_lowercase()).copied())?;
            Some((*pk_slot, var_slot))
        })
        .collect();

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

    // Snapshot the resolved individual-parameter program for the analytic
    // sensitivity chain (issue #367) before the f64 `pk_param_fn` closure moves
    // `stmts_owned` / the mappings. `pk_var_slots` is `(pk_slot, var_slot)` per
    // individual parameter; ODE models use `ode_assignment_mapping`, analytical
    // models `pk_assignment_mapping` (both are `(pk_slot, var_slot)`).
    // Classify cov-static var slots once (#485): the analytic η/θ sensitivity
    // walks fold these to `f64` constants instead of re-deriving the covariate
    // kernel under dual arithmetic on every inner/outer evaluation. Drop the mask
    // when nothing is foldable so the eval paths stay on the original branch.
    let cov_static_mask = {
        let m = compute_cov_static_mask(&stmts_owned, n_vars);
        if m.iter().any(|&b| b) {
            m
        } else {
            Vec::new()
        }
    };

    let pk_var_slots = if is_analytical_pk {
        // Structural PK params (CL, V, …) plus any modeled-dose `D{cmt}`/`R{cmt}`
        // slots (#486). The latter are not read by the closed-form kernels, but the
        // analytic-sensitivity chain needs their `∂/∂(θ,η)` (+ 2nd order) rows so the
        // event-driven walk can seed the modeled infusion rate/window dual
        // (`pk_slot_dual_outer` looks the slot up in `pk_slots`). Appended after the
        // structural slots so the row order stays stable; empty for a model with no
        // modeled dose, so this is a no-op there. A fixed-dose subject of such a model
        // routes to superposition, where the extra slot is seeded but unread (its
        // kernel derivative is 0), so it is harmless.
        let mut m = pk_assignment_mapping.clone();
        m.extend(analytical_extra_mapping.iter().copied());
        m
    } else {
        ode_assignment_mapping.clone()
    };
    let pk_slot_indices = pk_var_slots.iter().map(|&(slot, _)| slot).collect();
    // A direct `pk(...=TIME)` mapping is desugared into a `__ferx_pktime_* = TIME`
    // statement upstream (#486), so `stmts_use_time` already covers it — no separate
    // literal-TIME-binding term is needed.
    let uses_time_builtin = stmts_use_time;
    let indiv_param_program = IndivParamProgram {
        stmts: stmts_owned.clone(),
        n_vars,
        cov_static_mask,
        pk_var_slots,
        pk_slot_indices,
        n_theta: n_theta_base,
        n_eta: n_eta_extended,
        cov_names: cov_names_for_lookup.clone(),
        uses_time_builtin,
    };

    // Snapshot the NN handles into the closure. Empty when no
    // `[covariate_nn]` blocks are present, in which case the per-call
    // forward-pass loop below is a no-op (just an empty `Vec<Vec<f64>>`
    // alloc — cheap enough to skip the branch).
    #[cfg(feature = "nn")]
    let covariate_nns_owned: Vec<crate::nn::CovariateNn> = covariate_nns.to_vec();

    let pk_param_fn: PkParamFn = Box::new(
        move |theta: &[f64], eta: &[f64], covariates: &HashMap<String, f64>, time: f64| {
            // Resolve the `TIME` built-in from the event time the caller passed.
            // `Expression::Time` / `Op::PushTime` (including the desugared
            // `__ferx_pktime_* = TIME` parameters, #486) read the model-time
            // thread-local, so set it here for the duration of this evaluation. Gated
            // on `uses_time_builtin` so models that don't use the built-in (the common
            // case) pay nothing for the guard (#610).
            let _time_guard = uses_time_builtin.then(|| ModelTimeGuard::enter(time));
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

            let mut p = PkParams::default();
            FERX_SCRATCH.with(|cell| {
                let mut scratch = cell.borrow_mut();
                let FerxThreadScratch {
                    pk_cov,
                    pk_vars,
                    bc_stack,
                    ..
                } = &mut *scratch;

                // Materialise covariates into thread-local scratch aligned with
                // `referenced_covariates`. This runs millions of times in FOCE
                // inner loops, so keeping the buffers hot avoids short heap
                // allocations per event/line-search evaluation.
                pk_cov.resize(n_cov, 0.0);
                for (i, name) in cov_names_for_lookup.iter().enumerate() {
                    pk_cov[i] = covariates.get(name).copied().unwrap_or(0.0);
                }
                pk_vars.resize(n_vars, 0.0);
                pk_vars.fill(0.0);

                // pk_param_fn doesn't compute derivatives — no `du` to pass.
                // Call the stack-threaded evaluator directly while this scratch
                // borrow is live; the wrapper would re-enter FERX_SCRATCH.
                eval_statements_indexed_with_stack(
                    &stmts_owned,
                    theta,
                    eta,
                    pk_cov,
                    pk_vars,
                    None,
                    &nn_outputs,
                    bc_stack,
                );

                if is_analytical_pk {
                    for &(pk_slot, var_slot) in &pk_assignment_mapping {
                        p.values[pk_slot] = pk_vars[var_slot];
                    }
                    // Literal-valued slots (e.g. `ka=1.0`) are constants — no
                    // per-call evaluation, just write the parsed value.
                    for &(pk_slot, c) in &pk_const_mapping {
                        p.values[pk_slot] = c;
                    }
                    // Modeled infusion duration (`D{cmt}`, RATE=-2; #394): write
                    // each duration parameter into its reserved spare slot so the
                    // analytical dose-resolution step can read it.
                    for &(pk_slot, var_slot) in &analytical_extra_mapping {
                        p.values[pk_slot] = pk_vars[var_slot];
                    }
                } else {
                    // ODE model: store each individual parameter at its
                    // `ode_param_slots` slot (canonical names at their PK slot, F at
                    // PK_IDX_F, lagtime at PK_IDX_LAGTIME, others at free slots).
                    for &(slot, var_slot) in &ode_assignment_mapping {
                        p.values[slot] = pk_vars[var_slot];
                    }
                }
            });
            p
        },
    );
    Ok((
        pk_param_fn,
        referenced_covariates,
        indiv_partials,
        indiv_param_program,
    ))
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
    Time,
    /// Reserved read-only subpopulation index `MIXNUM` (1..=K) for `$MIXTURE`
    /// models (#977). Resolves from the mixture-class thread-local
    /// (`current_mixture_class`, default 1) at eval time — mirrors `Time`. It is
    /// constant within a single class evaluation (∂/∂θ = ∂/∂η = 0) but *varies
    /// across classes*, so it is treated as dynamic by the constant-fold guards
    /// (`bytecode_is_dynamic` / `expr_is_dynamic`) exactly like `Time`, to keep a
    /// `MIXNUM`-dependent slot from folding to class 1's value.
    MixNum,
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

thread_local! {
    static MODEL_TIME: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
}

/// RAII guard that sets the model-time thread-local for the duration of a
/// scope and restores the previous value on drop (panic-safe). Used at every
/// eval boundary that resolves `Expression::Time` / `Op::PushTime` so the
/// built-in `TIME` sees the event/integration time rather than the `0.0`
/// default. Hold it for the lifetime of an evaluation; let it drop to restore.
pub(crate) struct ModelTimeGuard(f64);

impl ModelTimeGuard {
    /// Set `MODEL_TIME` to `time`, returning a guard that restores the prior
    /// value when dropped.
    pub(crate) fn enter(time: f64) -> Self {
        let prev = MODEL_TIME.with(|cell| cell.replace(time));
        ModelTimeGuard(prev)
    }

    /// Enter the guard only when `cond` holds (typically `uses_time_builtin`),
    /// returning `Some(guard)` that restores on drop or `None` when the model does
    /// not read the `TIME` built-in. Collapses the repeated
    /// `cond.then(|| ModelTimeGuard::enter(time))` idiom at the per-event seed seams so a
    /// future widening of the gating condition (e.g. the direct `pk(...=TIME)` mapping)
    /// updates one call each and can't silently miss a seam (#637 review #7).
    #[inline]
    pub(crate) fn enter_if(cond: bool, time: f64) -> Option<Self> {
        cond.then(|| Self::enter(time))
    }
}

impl Drop for ModelTimeGuard {
    fn drop(&mut self) {
        MODEL_TIME.with(|cell| cell.set(self.0));
    }
}

pub(crate) fn with_model_time<T>(time: f64, f: impl FnOnce() -> T) -> T {
    let _guard = ModelTimeGuard::enter(time);
    f()
}

/// The model-time thread-local `Expression::Time` / `Op::PushTime` resolves
/// against. `pub(crate)` so a hand-built `OdeReadout::Single` test closure can
/// observe the same value a compiled readout would (the readout closure signature
/// carries no `time` argument — the guard is the channel).
pub(crate) fn current_model_time() -> f64 {
    MODEL_TIME.with(std::cell::Cell::get)
}

thread_local! {
    /// Active subpopulation index (1..=K) for `$MIXTURE` models (#977), read by
    /// `Expression::MixNum` / `Op::PushMixNum`. Defaults to 1 so non-mixture
    /// evaluation and any class-agnostic path (`predict`/`simulate`) see class 1.
    static MIXTURE_CLASS: std::cell::Cell<usize> = const { std::cell::Cell::new(1) };
}

// The RAII guard that sets `MIXTURE_CLASS` per (subject × class) around each
// class NLL lands with the mixture objective (#977 Phase 3). Until then the
// thread-local stays at its class-1 default and `MIXNUM` resolves to 1.

fn current_mixture_class() -> usize {
    MIXTURE_CLASS.with(std::cell::Cell::get)
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

/// Mutable twin of [`visit_expr_nodes`] — same pre-order shape, but `f` may
/// **replace** the node it is handed. Backs the in-place node rewrites
/// (`rewrite_scaling_time_alias`) that run between parse and index resolution.
/// Recursion happens after `f` returns, so a replacement node's children are
/// visited too; a rewrite that could re-fire on its own output must therefore be
/// idempotent (all current ones replace a leaf with a leaf).
fn visit_expr_nodes_mut(expr: &mut Expression, f: &mut dyn FnMut(&mut Expression)) {
    f(expr);
    match expr {
        Expression::BinOp(lhs, _, rhs) => {
            visit_expr_nodes_mut(lhs, f);
            visit_expr_nodes_mut(rhs, f);
        }
        Expression::UnaryFn(_, arg) => visit_expr_nodes_mut(arg, f),
        Expression::Power(base, exp) => {
            visit_expr_nodes_mut(base, f);
            visit_expr_nodes_mut(exp, f);
        }
        Expression::Conditional(cond, t, e) => {
            visit_condition_nodes_mut(cond, f);
            visit_expr_nodes_mut(t, f);
            visit_expr_nodes_mut(e, f);
        }
        _ => {}
    }
}

/// Condition companion of [`visit_expr_nodes_mut`].
fn visit_condition_nodes_mut(cond: &mut Condition, f: &mut dyn FnMut(&mut Expression)) {
    match cond {
        Condition::Compare(l, _, r) => {
            visit_expr_nodes_mut(l, f);
            visit_expr_nodes_mut(r, f);
        }
        Condition::And(l, r) | Condition::Or(l, r) => {
            visit_condition_nodes_mut(l, f);
            visit_condition_nodes_mut(r, f);
        }
        Condition::Not(c) => visit_condition_nodes_mut(c, f),
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

/// Rewrite every reference to a sample-size-weighted IOV kappa (#1031) into its
/// scaled form: `Eta(k)` → `Eta(k) / sqrt(W_k)`, where `weights` maps the
/// kappa's slot in the *extended* eta vector (`n_eta + kappa_idx`) to its
/// compiled weight expression.
///
/// This is the whole of "the engine applies the scaling": with `κ ~ N(0, γ²)`
/// the rewritten effect is distributed `N(0, γ²/W)`, so the declaration
/// `kappa K ~ γ² weight = W` and the hand-written `K / sqrt(W)` are the same
/// model — and every estimator, sensitivity and diagnostic downstream sees a
/// shape it already supports.
///
/// Not written on top of [`visit_expr_nodes_mut`]: that walker recurses *into*
/// a replacement node, so replacing `Eta(k)` with an expression that still
/// contains `Eta(k)` would re-fire forever. This walk descends only into the
/// nodes it did not create.
///
/// Returns the set of slots that were actually rewritten — i.e. the weighted
/// kappas the block really references. A weighted kappa missing from that set
/// carries a weight that does nothing, which the caller rejects rather than
/// letting the fit report a scaling it never applied.
fn rewrite_weighted_kappas(
    stmts: &mut [Statement],
    weights: &HashMap<usize, Expression>,
) -> HashSet<usize> {
    fn walk_expr(
        expr: &mut Expression,
        weights: &HashMap<usize, Expression>,
        hit: &mut HashSet<usize>,
    ) {
        if let Expression::Eta(i) = expr {
            if let Some(w) = weights.get(i) {
                hit.insert(*i);
                *expr = Expression::BinOp(
                    Box::new(Expression::Eta(*i)),
                    BinOp::Div,
                    Box::new(Expression::UnaryFn("sqrt".to_string(), Box::new(w.clone()))),
                );
            }
            return;
        }
        match expr {
            Expression::BinOp(l, _, r) => {
                walk_expr(l, weights, hit);
                walk_expr(r, weights, hit);
            }
            Expression::UnaryFn(_, a) => walk_expr(a, weights, hit),
            Expression::Power(b, e) => {
                walk_expr(b, weights, hit);
                walk_expr(e, weights, hit);
            }
            Expression::Conditional(c, t, e) => {
                walk_cond(c, weights, hit);
                walk_expr(t, weights, hit);
                walk_expr(e, weights, hit);
            }
            _ => {}
        }
    }
    fn walk_cond(
        cond: &mut Condition,
        weights: &HashMap<usize, Expression>,
        hit: &mut HashSet<usize>,
    ) {
        match cond {
            Condition::Compare(l, _, r) => {
                walk_expr(l, weights, hit);
                walk_expr(r, weights, hit);
            }
            Condition::And(l, r) | Condition::Or(l, r) => {
                walk_cond(l, weights, hit);
                walk_cond(r, weights, hit);
            }
            Condition::Not(c) => walk_cond(c, weights, hit),
        }
    }
    let mut hit: HashSet<usize> = HashSet::new();
    for s in stmts {
        match s {
            Statement::Assign(_, e)
            | Statement::AssignIdx(_, e)
            | Statement::DiffEq(_, e)
            | Statement::DiffEqIdx(_, e) => walk_expr(e, weights, &mut hit),
            // Bytecode forms only exist after `resolve_variable_indices`, which
            // runs long after this rewrite.
            Statement::AssignBc(_, _) | Statement::DiffEqBc(_, _) => {}
            Statement::If {
                branches,
                else_body,
            } => {
                for (cond, body) in branches {
                    walk_cond(cond, weights, &mut hit);
                    hit.extend(rewrite_weighted_kappas(body, weights));
                }
                if let Some(eb) = else_body {
                    hit.extend(rewrite_weighted_kappas(eb, weights));
                }
            }
        }
    }
    hit
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

/// Whether `expr` contains the model-time built-in node itself — the
/// `Expression::Time` a bare `TIME` / `time` parses to. Unlike
/// [`expr_references_time_builtin`] this does **not** also match the `T` / `t`
/// alias: the alias fold (`rewrite_scaling_time_alias`) never runs on an
/// `init(...)` RHS, so `T` there is an ordinary identifier that the surrounding
/// scope checks already handle. Drives the init-time rejection (#994).
fn expr_references_time_node(expr: &Expression) -> bool {
    let mut found = false;
    visit_expr_nodes(expr, &mut |e| {
        if matches!(e, Expression::Time) {
            found = true;
        }
    });
    found
}

fn stmts_use_time_builtin(stmts: &[Statement]) -> bool {
    let mut found = false;
    visit_stmt_nodes(stmts, &mut |e| {
        if matches!(e, Expression::Time) {
            found = true;
        }
    });
    found
}

/// Whether any expression in the statement list references the reserved
/// `MIXNUM` subpopulation index (#977). Used to reject `MIXNUM` in a
/// non-mixture model.
fn stmts_use_mixnum(stmts: &[Statement]) -> bool {
    let mut found = false;
    visit_stmt_nodes(stmts, &mut |e| {
        if matches!(e, Expression::MixNum) {
            found = true;
        }
    });
    found
}

/// Whether any raw block line references the reserved `MIXNUM` built-in (#977).
/// Used to reject `MIXNUM` **anywhere** in a model with no `[mixture]` block —
/// not just `[individual_parameters]` (which is caught at the AST level by
/// [`stmts_use_mixnum`]), but also `[error_model]`, `[structural_model]`,
/// `[derived]`, `[odes]`, etc., whose expressions are compiled to opaque closures
/// after this check. Tokenizes each line and matches the `MIXNUM` identifier
/// (case-insensitive), so a substring or a comment can't false-positive. A
/// tokenizer error is treated as "not found" — that block's own parse surfaces it.
fn lines_use_mixnum(lines: &[String]) -> bool {
    lines.iter().any(|line| {
        tokenize(line).is_ok_and(|toks| {
            toks.iter()
                .any(|t| matches!(t, Token::Ident(s) if s.eq_ignore_ascii_case("MIXNUM")))
        })
    })
}

/// Whether any statement assigns to `MIXNUM`. The index is reserved read-only
/// (#977), so an assignment to it is rejected. Recurses into `if` branches.
/// Called before `resolve_variable_indices`, so only `Assign` / `If` appear.
fn stmts_assign_mixnum(stmts: &[Statement]) -> bool {
    stmts.iter().any(|s| match s {
        Statement::Assign(name, _) => name.eq_ignore_ascii_case("MIXNUM"),
        Statement::If {
            branches,
            else_body,
        } => {
            branches.iter().any(|(_, body)| stmts_assign_mixnum(body))
                || else_body.as_deref().is_some_and(stmts_assign_mixnum)
        }
        _ => false,
    })
}

pub(crate) fn compiled_model_uses_time_builtin(model: &CompiledModel) -> bool {
    model
        .indiv_param_partials
        .indiv_param_program
        .as_ref()
        .is_some_and(|program| program.uses_time_builtin)
}

/// Built-in names an `[odes] init(state) = ...` expression may reference **in
/// addition to** the model's own declared states (bound to 0 at t = 0) and its
/// individual parameters. Matched case-insensitively, as both built-ins are.
///
/// Scoped to the `[odes]` surface on purpose. The analytical
/// `[initial_conditions] init(cmt) = ...` surface parses with `ParseCtx::new`,
/// where an unresolved identifier is an `Expression::Covariate` rather than an
/// `Expression::Variable` — so `MACHEPS` there is an ordinary (and therefore
/// required) data column, not machine epsilon, which only [`MapEnv::resolve`]'s
/// `Variable` arm supplies. `MIXNUM` is the one entry that does carry over: it
/// parses to its own `Expression::MixNum` node and resolves through
/// `current_mixture_class()` on both surfaces.
///
/// Public so a downstream generator (`ferxtranslate`) can mirror the init scope
/// rule instead of probing the engine one name at a time (issue #994). The
/// `[odes] init(...)` diagnostic is rendered from this list, so the message and
/// the guard cannot drift apart.
pub const ODE_INIT_SCOPE_BUILTINS: &[&str] = &["MACHEPS", "MIXNUM"];

/// Built-in names an `[odes] init(state) = ...` expression may **not**
/// reference. Every one of them is a time-since-something clock, and an initial
/// condition is evaluated at the time origin, so each would read a constant 0
/// and silently flatten the expression (issue #994). `TIME` is rejected by an
/// AST-node guard ([`expr_references_time_node`] — a bare `TIME` parses to a
/// dedicated `Expression::Time` node, not a variable, which is how it evaded the
/// undefined-name check); `T`, `TAFD` and `TAD` are ordinary identifiers in an
/// `[odes]` init RHS and are caught by the undefined-name check around it.
///
/// Only the first entry carries to the analytical `[initial_conditions]`
/// surface. `TIME` is rejected there too (same node guard, same reason), but
/// `T` / `TAFD` / `TAD` are ordinary covariates outside `[odes]` — the same
/// deliberate rule `[scaling]` follows, so a dataset that really carries a `TAD`
/// column can use it (see `ODE_ONLY_BUILTINS` in `api::validation`, #1028).
/// A consumer mirroring this list on an analytical model would reject a legal
/// covariate reference.
pub const ODE_INIT_REJECTED_BUILTINS: &[&str] = &["TIME", "T", "TAFD", "TAD"];

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

/// Detect a `[individual_parameters]` expression that reads an in-block name
/// *before* that name is assigned in file order (issue #710). Because the block
/// registers every assigned name as a defined variable — so a forward reference
/// parses as `Variable(name)` rather than `Covariate(name)` (see the two-pass
/// parse in `compile_model`) — such a reference would otherwise resolve to the
/// running var map's default `0.0` at eval time, silently collapsing the formula
/// (e.g. `exp(IMAX*…)` → `exp(0)`) with no diagnostic. This mirrors the `[odes]`
/// undefined-reference guard (#314, `collect_undefined_vars`) but keys on
/// declaration *order*, not *presence*: the name IS declared, just too late.
///
/// `in_block` is every name assigned anywhere in the block
/// (`assigned_vars_in_order`); only reads of those names are order-checked. A
/// `Variable(name)` whose name is outside `in_block` cannot occur in this parse
/// context, and covariate / theta / eta references are distinct node kinds that
/// are never flagged. Returns each `(used_name, defining_name)` violation in
/// encounter order (a read of a not-yet-assigned name inside the RHS of
/// `defining_name = …`). Statements execute top-to-bottom; each `if` arm is
/// checked against the names assigned before the `if`, and a name assigned in
/// any arm is treated as available afterwards (conservative — avoids
/// false-positive order errors on a name defined across branches).
fn collect_forward_refs(stmts: &[Statement], in_block: &HashSet<String>) -> Vec<(String, String)> {
    fn check_reads(
        assigned: &HashSet<String>,
        in_block: &HashSet<String>,
        defining: &str,
        out: &mut Vec<(String, String)>,
        visit: impl FnOnce(&mut dyn FnMut(&Expression)),
    ) {
        visit(&mut |e: &Expression| {
            if let Expression::Variable(name) = e {
                if in_block.contains(name) && !assigned.contains(name) {
                    out.push((name.clone(), defining.to_string()));
                }
            }
        });
    }
    fn walk(
        stmts: &[Statement],
        assigned: &mut HashSet<String>,
        in_block: &HashSet<String>,
        out: &mut Vec<(String, String)>,
    ) {
        for s in stmts {
            match s {
                Statement::Assign(name, expr) => {
                    check_reads(assigned, in_block, name, out, |f| visit_expr_nodes(expr, f));
                    assigned.insert(name.clone());
                }
                Statement::If {
                    branches,
                    else_body,
                } => {
                    // Each arm sees only names assigned before the `if`.
                    for (cond, body) in branches {
                        check_reads(assigned, in_block, "if-condition", out, |f| {
                            visit_condition_nodes(cond, f)
                        });
                        let mut branch_assigned = assigned.clone();
                        walk(body, &mut branch_assigned, in_block, out);
                    }
                    if let Some(eb) = else_body {
                        let mut branch_assigned = assigned.clone();
                        walk(eb, &mut branch_assigned, in_block, out);
                    }
                    // A name assigned in any arm is available afterwards.
                    let mut branch_names: Vec<String> = Vec::new();
                    for (_, body) in branches {
                        branch_names.extend(assigned_vars_in_order(body));
                    }
                    if let Some(eb) = else_body {
                        branch_names.extend(assigned_vars_in_order(eb));
                    }
                    assigned.extend(branch_names);
                }
                _ => {}
            }
        }
    }
    let mut assigned: HashSet<String> = HashSet::new();
    let mut out: Vec<(String, String)> = Vec::new();
    walk(stmts, &mut assigned, in_block, &mut out);
    out
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

/// Reserved name prefix for the individual parameters the parser synthesizes when a
/// Form-C readout `y = expr` references a θ/η **directly** (issue #486). A bare
/// `THETA(i)` / `ETA(k)` in the readout is desugared into an individual parameter
/// equal to it (`__ferx_ro_th{i} = THETA(i)`); the readout then references that
/// parameter instead, so its `∂y/∂θ` / `∂y/∂η` ride the validated
/// individual-parameter sensitivity chain (every analytic walk serves the readout
/// without a bespoke θ/η term). The prefix lets user-facing EBE / diagnostic output
/// suppress these internal parameters — they are not user-declared quantities.
pub(crate) const READOUT_SYNTH_PREFIX: &str = "__ferx_ro_";

/// Reserved prefix for the individual parameter synthesized by desugaring a direct
/// `pk(...=TIME)` structural mapping (`__ferx_pktime_<pk_name> = TIME`, #486). Like the
/// readout prefix, it lets EBE / diagnostic output suppress the internal parameter.
pub(crate) const PKTIME_SYNTH_PREFIX: &str = "__ferx_pktime_";

/// True for a ferx-internal synthetic individual parameter — either a Form-C readout θ/η
/// desugaring ([`READOUT_SYNTH_PREFIX`]) or a direct `pk(...=TIME)` mapping desugaring
/// ([`PKTIME_SYNTH_PREFIX`]). User-facing per-subject output skips these so the synthetic
/// parameters never surface as sdtab / fit-output columns.
pub(crate) fn is_synthetic_readout_param(name: &str) -> bool {
    name.starts_with(READOUT_SYNTH_PREFIX) || name.starts_with(PKTIME_SYNTH_PREFIX)
}

/// A θ/η reference desugared out of a Form-C readout into a synthetic individual
/// parameter (issue #486). `is_eta` selects the source axis (`Theta(idx)` when
/// false, `Eta(idx)` when true); `name` is the `READOUT_SYNTH_PREFIX`-prefixed
/// individual-parameter name that mirrors it.
#[derive(Debug, Clone)]
pub(crate) struct ReadoutSynthParam {
    name: String,
    is_eta: bool,
    idx: usize,
}

/// Split a `[scaling]` line `key = value` at the first `=` that lies OUTSIDE any
/// `[...]` bracket, so `obs_scale[CMT=1] = expr` splits at the top-level `=`, not the
/// one inside the bracket. Returns the trimmed key and value.
fn split_scaling_entry(trimmed: &str) -> Result<(&str, &str), String> {
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
    let split_at =
        split_at.ok_or_else(|| format!("[scaling]: expected `key = value`, got: `{trimmed}`"))?;
    Ok((trimmed[..split_at].trim(), trimmed[split_at + 1..].trim()))
}

// ── [scaling] named intermediates (#1030) ───────────────────────────────────
//
// `[scaling]` used to accept exactly two keys, so a bounded-endpoint readout had
// to repeat its guarded sub-expression once per occurrence — four times for the
// standard `min(max(x, 0.01), 0.99)` clamp, on a single ~200-character line that
// no reviewer can check. Any other key is now a *named intermediate*: a local
// binding usable by the `obs_scale` / `y` entries below it, exactly as
// `[individual_parameters]` already allows.
//
// Intermediates are **inlined into the AST** of each entry that references them,
// right after that entry is parsed and before anything else looks at it. That is
// deliberate: it means every downstream consumer — bytecode compilation, the
// `Dual2` sensitivity programs, the covariate/required-column scan (#1028), the
// dose-attribute double-use rejection (#993/#1004), the `T`-alias fold, the
// direct-θ/η desugaring (#486), the readout slot allocator (#650) — sees exactly
// the expression the user would have written by hand, with no new node type to
// teach any of them about. The cost is that a repeated intermediate is evaluated
// once per occurrence, which is what hand-expansion costs today.

/// Whether `base` is one of the two reserved `[scaling]` entry keys. Anything else
/// on the left of an `=` is a named intermediate.
fn is_scaling_entry_key(base: &str) -> bool {
    base == "y" || base == "obs_scale"
}

/// Whether `name` is a syntactically valid DSL identifier (the shape the
/// tokenizer produces for `Token::Ident`).
fn is_plain_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Names a `[scaling]` intermediate may not take: the built-ins the expression
/// parser resolves before it would ever consult the intermediate table, so a
/// binding under one of these names would silently never be read.
fn scaling_intermediate_reserved(name: &str) -> Option<&'static str> {
    if name == "TIME" || name == "time" {
        return Some("the model-time built-in");
    }
    if is_time_alias(name) {
        return Some("the `[odes]` model-time alias for `TIME`");
    }
    if name.eq_ignore_ascii_case("MIXNUM") {
        return Some("the mixture subpopulation index");
    }
    if name.eq_ignore_ascii_case("if") || name.eq_ignore_ascii_case("else") {
        return Some("an inline-conditional keyword");
    }
    if is_synthetic_readout_param(name) {
        return Some("a reserved internal parameter prefix");
    }
    if matches!(
        name.to_ascii_uppercase().as_str(),
        "TAD" | "TAFD" | "MACHEPS"
    ) {
        return Some("an eval-time built-in");
    }
    None
}

/// Collect the `[scaling]` block's named intermediates, in declaration order, as
/// `(name, source expression)` pairs.
///
/// Shared by every scan over `[scaling]` lines — `parse_scaling_block` itself and
/// the three pre-scans that run before it (`collect_readout_theta_eta_synth`,
/// `allocate_readout_extra_slots`, and the dose-attribute read scan in
/// `parse_model_file`) — so none of them can disagree about which lines are
/// entries and which are bindings.
fn scaling_intermediates(lines: &[String]) -> Result<Vec<(String, String)>, String> {
    let mut out: Vec<(String, String)> = Vec::new();
    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let (key, value) = split_scaling_entry(trimmed)?;
        let (base, cmt) = parse_scaling_key(key)?;
        if is_scaling_entry_key(base) {
            continue;
        }
        if cmt.is_some() {
            return Err(format!(
                "[scaling]: unknown key `{base}` — only `obs_scale` and `y` take a \
                 `[CMT=N]` subscript. A named intermediate is written `{base} = <expr>`, \
                 with no subscript."
            ));
        }
        if !is_plain_identifier(base) {
            return Err(format!("[scaling]: unknown key `{base}`"));
        }
        if let Some(what) = scaling_intermediate_reserved(base) {
            return Err(format!(
                "[scaling]: `{base}` cannot be used as a named intermediate — it is \
                 {what}. Pick another name."
            ));
        }
        if out.iter().any(|(n, _)| n == base) {
            return Err(format!(
                "[scaling]: duplicate named intermediate `{base}` — each one is assigned once."
            ));
        }
        if value.is_empty() {
            return Err(format!(
                "[scaling]: named intermediate `{base}` has no right-hand side."
            ));
        }
        out.push((base.to_string(), value.to_string()));
    }
    Ok(out)
}

/// Replace every leaf that names one of `expanded` with a clone of that
/// intermediate's (already fully expanded) expression.
///
/// Both leaf spellings are matched: `[scaling]` parses with
/// `fallback_covariate = true`, so an unresolved identifier arrives as
/// `Covariate(name)`, while `Variable(name)` is what it becomes in the contexts
/// that put states / individual parameters in scope.
///
/// [`visit_expr_nodes_mut`] descends into the node it just replaced, which is safe
/// here precisely because `expanded` holds *expanded* bodies: an intermediate can
/// only reference intermediates declared above it (enforced in
/// [`inline_scaling_intermediates`]), so a substituted body contains no
/// intermediate name and the walk cannot re-fire on its own output.
fn substitute_scaling_intermediates(expr: &mut Expression, expanded: &[(String, Expression)]) {
    if expanded.is_empty() {
        return;
    }
    visit_expr_nodes_mut(expr, &mut |e: &mut Expression| {
        let name = match e {
            Expression::Covariate(n) | Expression::Variable(n) => n.as_str(),
            _ => return,
        };
        if let Some((_, body)) = expanded.iter().find(|(n, _)| n == name) {
            *e = body.clone();
        }
    });
}

/// First identifier in `expr` that names one of `names`, if any.
fn first_scaling_intermediate_ref(expr: &Expression, names: &[(String, String)]) -> Option<String> {
    let mut found: Option<String> = None;
    visit_expr_nodes(expr, &mut |e: &Expression| {
        if found.is_some() {
            return;
        }
        let name = match e {
            Expression::Covariate(n) | Expression::Variable(n) => n.as_str(),
            _ => return,
        };
        if names.iter().any(|(m, _)| m == name) {
            found = Some(name.to_string());
        }
    });
    found
}

/// Inline the `[scaling]` named intermediates into a just-parsed entry expression.
///
/// `ctx` must be the same `ParseCtx` the entry itself was parsed with, so an
/// intermediate resolves its θ / η / individual-parameter / state names against the
/// scope of the entry that uses it. (`obs_scale` has no states in scope; `y` does —
/// an intermediate that reads a state is therefore legal in `y` and becomes a
/// required data column in `obs_scale`, the same as writing it inline.)
///
/// Forward and self references are rejected: intermediate `i` is expanded against
/// `0..i` only, which makes a reference cycle unrepresentable rather than a
/// recursion guard to get right.
fn inline_scaling_intermediates(
    expr: &mut Expression,
    intermediates: &[(String, String)],
    ctx: ParseCtx<'_>,
) -> Result<(), String> {
    if intermediates.is_empty() {
        return Ok(());
    }
    let mut expanded: Vec<(String, Expression)> = Vec::with_capacity(intermediates.len());
    for (i, (name, src)) in intermediates.iter().enumerate() {
        let mut body =
            parse_scalar_expression(src, ctx).map_err(|e| format!("[scaling] {name}: {e}"))?;
        substitute_scaling_intermediates(&mut body, &expanded);
        if let Some(later) = first_scaling_intermediate_ref(&body, &intermediates[i..]) {
            return Err(format!(
                "[scaling] {name}: references the named intermediate `{later}`, which is \
                 declared on or below this line. An intermediate may only use ones \
                 declared above it."
            ));
        }
        expanded.push((name.clone(), body));
    }
    substitute_scaling_intermediates(expr, &expanded);
    Ok(())
}

/// Every identifier appearing in a `[scaling]` source expression **in a position
/// the AST substituter can reach**, as raw tokens.
///
/// Used for the reachability scan behind the unused-intermediate rejection, which
/// needs names only and must not depend on how any particular scope resolves them.
/// The one thing it must agree with is [`substitute_scaling_intermediates`], which
/// only rewrites `Variable` / `Covariate` *leaves*: an identifier followed by `(`
/// parses as a function call and one followed by `.` as an `[covariate_nn]` output
/// access, so an intermediate named in either position is never inlined and must
/// still be reported as unused rather than counted as a read (#1030).
fn source_identifiers(src: &str) -> Vec<String> {
    // A malformed expression is reported with full context by the real parse;
    // contributing no names here just means it reaches nothing.
    let Ok(toks) = tokenize(src) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (i, tok) in toks.iter().enumerate() {
        let Token::Ident(name) = tok else { continue };
        if matches!(toks.get(i + 1), Some(Token::LParen) | Some(Token::Dot)) {
            continue;
        }
        out.push(name.clone());
    }
    out
}

/// Reject a named intermediate that no `obs_scale` / `y` entry reaches (#1030).
///
/// Before intermediates existed, every key other than `obs_scale` / `y` was an
/// error — which is what caught a typo like `obs_scal = V`. Now that any key is
/// syntactically legal, that typo would silently disable scaling instead, so the
/// same protection is re-established from the other side: a binding nothing reads
/// is a mistake, and is rejected with both readings named.
fn check_scaling_intermediates_used(
    lines: &[String],
    intermediates: &[(String, String)],
) -> Result<(), String> {
    if intermediates.is_empty() {
        return Ok(());
    }
    let mut reached: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut queue: Vec<String> = Vec::new();
    let mut declared_above: Vec<&str> = Vec::new();
    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let (key, value) = split_scaling_entry(trimmed)?;
        let (base, _cmt) = parse_scaling_key(key)?;
        if !is_scaling_entry_key(base) {
            declared_above.push(base);
            continue;
        }
        // Entries obey the same define-above-use-below rule the intermediates
        // themselves do. `inline_scaling_intermediates` substitutes an entry
        // against the whole table regardless of source order, so without this the
        // rule would be enforced on one side only (#1030).
        for name in source_identifiers(value) {
            if intermediates.iter().any(|(n, _)| *n == name)
                && !declared_above.contains(&name.as_str())
            {
                return Err(format!(
                    "[scaling] {key}: references the named intermediate `{name}`, which is \
                     declared on or below this line. An intermediate must be declared above \
                     every entry that uses it."
                ));
            }
        }
        queue.extend(source_identifiers(value));
    }
    while let Some(name) = queue.pop() {
        let Some((n, src)) = intermediates.iter().find(|(n, _)| *n == name) else {
            continue;
        };
        if reached.insert(n.as_str()) {
            queue.extend(source_identifiers(src));
        }
    }
    if let Some((unused, _)) = intermediates
        .iter()
        .find(|(n, _)| !reached.contains(n.as_str()))
    {
        return Err(format!(
            "[scaling]: `{unused}` is never used. If it is a named intermediate, \
             reference it from an `obs_scale` / `y` entry below it; if it is a \
             misspelled `obs_scale` / `y` key, fix the spelling — as written it does \
             nothing."
        ));
    }
    Ok(())
}

/// Outcome of [`allocate_readout_extra_slots`]: the `(indiv param name, PK slot)` pairs the
/// analytic Form-C readout needs, plus — for the analytical engine only — which synthetic
/// θ/η parameters (#486) actually got a slot and, if they did not, the FD-fallback note.
#[derive(Default)]
struct ReadoutExtraSlots {
    slots: Vec<(String, usize)>,
    accepted_synths: Vec<ReadoutSynthParam>,
    fd_note: Option<String>,
}

/// Allocate free `PkParams` slots to the **non-structural** individual parameters
/// an analytic Form C readout references (#650), making them first-class
/// differentiable parameters: `pk_param_fn` writes each value into its slot and the
/// individual-parameter program carries its `∂/∂(θ,η)`, exactly as for a structural
/// PK parameter. So a readout like `central/V + BMAX*C/(KD+C)` differentiates
/// `BMAX`/`KD` analytically instead of aliasing them onto the `CL` slot.
///
/// Slots are drawn from the differentiable slot space `0..=PK_IDX_MTT` (the range
/// `slot_to_dim` covers and `seed_dim` is sized for), restricted to slots the
/// model's solver does **not** read (`!consumes_pk_slot`) and not already claimed by
/// a modeled-dose parameter — so a readout param never collides with a value the
/// closed form consumes. Structural parameters (bound to a PK role) keep their
/// canonical slot; θ/η/covariate references are ignored. Returns `(name, slot)` in
/// reference order, or an error if the readout references more distinct
/// non-structural parameters than the free differentiable slots can hold.
///
/// ODE models get no entries here (their individual parameters already receive
/// slots via `ode_param_slots`), and a model with no `[scaling] y` readout returns
/// empty.
#[allow(clippy::too_many_arguments)]
fn allocate_readout_extra_slots(
    scaling_lines: Option<&Vec<String>>,
    theta_names: &[String],
    eta_names: &[String],
    indiv_var_names: &[String],
    structural_vars: &std::collections::HashSet<String>,
    modeled_slots: &[(String, usize)],
    pk_model: PkModel,
    is_ode: bool,
    synth_params: &[ReadoutSynthParam],
) -> Result<ReadoutExtraSlots, String> {
    if is_ode {
        return Ok(ReadoutExtraSlots::default());
    }
    let Some(lines) = scaling_lines else {
        return Ok(ReadoutExtraSlots::default());
    };
    let indiv_set: std::collections::HashSet<&str> =
        indiv_var_names.iter().map(|s| s.as_str()).collect();
    let mut referenced: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    // An individual parameter can be referenced only from a named intermediate the
    // readout uses (#1030); it still needs a slot, so inline before scanning.
    let intermediates = scaling_intermediates(lines)?;
    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let (key, value) = split_scaling_entry(trimmed)?;
        let (base, _cmt) = parse_scaling_key(key)?;
        if base != "y" {
            continue;
        }
        // Individual params in scope resolve to `Variable(name)`; θ/η/covariate
        // references resolve elsewhere and are ignored here.
        let ctx = ParseCtx::new(theta_names, eta_names, indiv_var_names);
        let mut expr =
            parse_scalar_expression(value, ctx).map_err(|e| format!("[scaling] y: {e}"))?;
        inline_scaling_intermediates(&mut expr, &intermediates, ctx)?;
        visit_expr_nodes(&expr, &mut |e: &Expression| {
            if let Expression::Variable(name) = e {
                if indiv_set.contains(name.as_str())
                    && !structural_vars.contains(name)
                    && seen.insert(name.clone())
                {
                    referenced.push(name.clone());
                }
            }
        });
    }
    if referenced.is_empty() && synth_params.is_empty() {
        return Ok(ReadoutExtraSlots::default());
    }
    let used: std::collections::HashSet<usize> = modeled_slots.iter().map(|&(_, s)| s).collect();
    let mut free: Vec<usize> = (0..=crate::types::PK_IDX_MTT)
        .filter(|&s| !pk_model.consumes_pk_slot(s) && !used.contains(&s))
        .collect();
    // The pool THIS engine draws from, for the FD-fallback note. Not `MAX_PK_PARAMS`: the
    // analytical allocator is confined to the differentiable spare region, so quoting the
    // ODE layout would misstate the user's headroom tenfold (PR #950 review #5).
    let layout_slots = free.len();
    // Text-referenced parameters have priority and keep their pre-#486 hard failure: the user
    // named them, so silently dropping the readout's analytic path would be the wrong answer.
    if referenced.len() > free.len() {
        return Err(format!(
            "[scaling] y: the analytic readout references {} non-structural individual \
             parameter(s) ({}), but only {} free differentiable PK slot(s) are available for \
             this model. Reduce the number of distinct non-PK parameters referenced in the \
             readout, or use an ODE model.",
            referenced.len(),
            referenced.join(", "),
            free.len(),
        ));
    }
    let remaining = free.split_off(referenced.len());
    let mut slots: Vec<(String, usize)> = referenced.into_iter().zip(free).collect();
    // Synthetic θ/η parameters are best-effort, mirroring the ODE sizing policy: a model that
    // parsed before #486 (where a direct θ/η in the readout took no slot at all) must not start
    // failing now, so an overflow drops them and the readout keeps its FD fallback.
    let mut accepted_synths: Vec<ReadoutSynthParam> = Vec::new();
    let mut fd_note: Option<String> = None;
    if synth_params.len() > remaining.len() {
        fd_note = Some(readout_synth_fd_note(
            synth_params.len() - remaining.len(),
            layout_slots,
        ));
    } else {
        for (s, slot) in synth_params.iter().zip(remaining) {
            slots.push((s.name.clone(), slot));
            accepted_synths.push(s.clone());
        }
    }
    Ok(ReadoutExtraSlots {
        slots,
        accepted_synths,
        fd_note,
    })
}

/// The documented FD-fallback warning when `short` synthetic readout parameters (#486) do
/// not fit the PK-slot layout. Shared by the ODE and analytical sizing sites so the two
/// cannot word the same condition differently.
///
/// `layout_slots` is the size of the pool the **calling engine** actually draws from, and the
/// two differ by an order of magnitude: an ODE model allocates from the full
/// [`MAX_PK_PARAMS`](crate::types::MAX_PK_PARAMS) (128) layout, while the analytical engine
/// draws only from the differentiable spare region `0..=PK_IDX_MTT` minus the slots the closed
/// form consumes — at most 11, typically 4–7. It is a parameter rather than a hard-coded
/// constant because quoting the ODE figure in a warning the analytical allocator emitted told
/// the user to free slots out of a layout their model never used (PR #950 review #5).
fn readout_synth_fd_note(short: usize, layout_slots: usize) -> String {
    format!(
        "[scaling] y: a direct THETA/ETA reference in the readout needs {short} more PK \
         slot(s) than the {layout_slots}-slot layout has free; the readout falls back to \
         finite-difference sensitivities. For analytic sensitivities, free up {short} \
         individual-parameter slot(s)."
    )
}

/// Append one synthetic readout parameter as a trailing individual parameter
/// (value = `θ_i` / `η_k`, so `∂p/∂θ_i = 1` / `∂p/∂η_k = 1`). The synthetic name carries the
/// reserved `__ferx_ro_` prefix (rejected for user params upstream) and is BTreeSet-deduped,
/// so it cannot collide. Shared by the ODE site and the analytical (deferred) one.
fn append_readout_synth_param(
    s: &ReadoutSynthParam,
    indiv_var_names: &mut Vec<String>,
    indiv_stmts: &mut Vec<Statement>,
) {
    indiv_var_names.push(s.name.clone());
    let rhs = if s.is_eta {
        Expression::Eta(s.idx)
    } else {
        Expression::Theta(s.idx)
    };
    indiv_stmts.push(Statement::Assign(s.name.clone(), rhs));
}

/// Scan a `[scaling]` block's `y = ...` readout entries for bare `THETA(i)` / `ETA(k)`
/// references and synthesize one individual parameter per distinct reference (#486).
/// Only `y` (Form-C readout) entries are scanned — `obs_scale` θ/η is differentiated
/// separately by `ScaleDerivProgram`. Identifiers other than θ/η (states, individual
/// parameters) parse as permissive covariate references here and are ignored; we only
/// collect the θ/η axes. Returns the synthetic descriptors in a stable
/// (θ-then-η, ascending index) order.
fn collect_readout_theta_eta_synth(
    scaling_lines: &[String],
    theta_names: &[String],
    eta_names: &[String],
) -> Result<Vec<ReadoutSynthParam>, String> {
    let mut thetas: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    let mut etas: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    // A θ/η reference can sit inside a named intermediate the readout uses (#1030);
    // inline them here too, or the desugaring would miss it and drop an otherwise
    // analytic readout to the FD fallback.
    let intermediates = scaling_intermediates(scaling_lines)?;
    for line in scaling_lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let (key, value) = split_scaling_entry(trimmed)?;
        let (base, _cmt) = parse_scaling_key(key)?;
        if base != "y" {
            continue;
        }
        // θ/η names resolve to `Theta`/`Eta`; every other identifier falls back to a
        // covariate (no `defined` set needed) since we only collect the θ/η axes.
        let ctx = ParseCtx::new(theta_names, eta_names, &[]);
        let mut expr =
            parse_scalar_expression(value, ctx).map_err(|e| format!("[scaling] y: {e}"))?;
        inline_scaling_intermediates(&mut expr, &intermediates, ctx)?;
        visit_expr_nodes(&expr, &mut |e: &Expression| match e {
            Expression::Theta(i) => {
                thetas.insert(*i);
            }
            Expression::Eta(i) => {
                etas.insert(*i);
            }
            _ => {}
        });
    }
    let mut out = Vec::with_capacity(thetas.len() + etas.len());
    for i in thetas {
        out.push(ReadoutSynthParam {
            name: format!("{READOUT_SYNTH_PREFIX}th{i}"),
            is_eta: false,
            idx: i,
        });
    }
    for k in etas {
        out.push(ReadoutSynthParam {
            name: format!("{READOUT_SYNTH_PREFIX}eta{k}"),
            is_eta: true,
            idx: k,
        });
    }
    Ok(out)
}

/// Rewrite the bare `THETA(i)` / `ETA(k)` nodes that the #486 desugaring covered into
/// `Variable(synthetic_name)`, so the readout compiles to `Op::PushVar` (an ordinary
/// individual-parameter reference) instead of `Op::PushTheta` / `Op::PushEta`. Any θ/η
/// not in `synth` is left intact (it then keeps the readout on the FD fallback via the
/// `dual_evaluable` gate). Mirrors [`resolve_expr_indices`]'s recursive shape.
fn rewrite_readout_synth(expr: &mut Expression, synth: &[ReadoutSynthParam]) {
    match expr {
        Expression::Theta(i) => {
            if let Some(s) = synth.iter().find(|s| !s.is_eta && s.idx == *i) {
                *expr = Expression::Variable(s.name.clone());
            }
        }
        Expression::Eta(i) => {
            if let Some(s) = synth.iter().find(|s| s.is_eta && s.idx == *i) {
                *expr = Expression::Variable(s.name.clone());
            }
        }
        Expression::BinOp(l, _, r) => {
            rewrite_readout_synth(l, synth);
            rewrite_readout_synth(r, synth);
        }
        Expression::UnaryFn(_, a) => rewrite_readout_synth(a, synth),
        Expression::Power(b, e) => {
            rewrite_readout_synth(b, synth);
            rewrite_readout_synth(e, synth);
        }
        Expression::Conditional(cond, t, e) => {
            rewrite_readout_synth_cond(cond, synth);
            rewrite_readout_synth(t, synth);
            rewrite_readout_synth(e, synth);
        }
        Expression::Literal(_)
        | Expression::Time
        | Expression::MixNum
        | Expression::Covariate(_)
        | Expression::Variable(_)
        | Expression::VariableIdx(_)
        | Expression::CovariateIdx(_)
        | Expression::NnOutput { .. } => {}
    }
}

/// Condition-tree companion of [`rewrite_readout_synth`].
fn rewrite_readout_synth_cond(cond: &mut Condition, synth: &[ReadoutSynthParam]) {
    match cond {
        Condition::Compare(l, _, r) => {
            rewrite_readout_synth(l, synth);
            rewrite_readout_synth(r, synth);
        }
        Condition::And(l, r) | Condition::Or(l, r) => {
            rewrite_readout_synth_cond(l, synth);
            rewrite_readout_synth_cond(r, synth);
        }
        Condition::Not(c) => rewrite_readout_synth_cond(c, synth),
    }
}

/// Collect the sigma names referenced in a `ParsedErrorModel` (before index resolution).
/// Collect the declared sigma names referenced anywhere in the `[error_model]`
/// arguments. An argument may be a bare sigma name or a magnitude expression
/// that scales one sigma (#484); either way we scan its identifiers and keep
/// those that match a declared sigma. Best-effort and infallible — strict
/// "exactly one sigma per argument" validation happens in `build_error_spec`.
fn used_sigma_names(
    parsed: &ParsedErrorModel,
    sigma_names: &[String],
) -> std::collections::HashSet<String> {
    let sigma_set: std::collections::HashSet<&str> =
        sigma_names.iter().map(|s| s.as_str()).collect();
    let mut out = std::collections::HashSet::new();
    let scan = |arg: &str, out: &mut std::collections::HashSet<String>| {
        for m in ARG_IDENT_RE.find_iter(arg) {
            if sigma_set.contains(m.as_str()) {
                out.insert(m.as_str().to_string());
            }
        }
    };
    match parsed {
        ParsedErrorModel::Single(_, args) => {
            for a in args {
                scan(a, &mut out);
            }
        }
        ParsedErrorModel::PerCmt(entries) => {
            for (_, _, args) in entries {
                for a in args {
                    scan(a, &mut out);
                }
            }
        }
        ParsedErrorModel::Selected {
            branches, default, ..
        } => {
            for (_, _, args) in branches {
                for a in args {
                    scan(a, &mut out);
                }
            }
            for a in &default.1 {
                scan(a, &mut out);
            }
        }
    }
    out
}

/// Extract the single declared sigma that a `[error_model]` argument scales.
///
/// A bare argument `PROP_ERR` returns `PROP_ERR`; a magnitude expression such
/// as `PROP_ERR * (if (TIME > 24) RUV else 1)` (#484) must reference exactly
/// one declared sigma, which is returned. Zero or multiple referenced sigmas
/// are errors — the sigma a magnitude scales must be unambiguous.
fn arg_sigma_name(arg: &str, sigma_names: &[String]) -> Result<String, String> {
    let sigma_set: std::collections::HashSet<&str> =
        sigma_names.iter().map(|s| s.as_str()).collect();
    let mut found: Vec<String> = Vec::new();
    for m in ARG_IDENT_RE.find_iter(arg) {
        let id = m.as_str();
        if sigma_set.contains(id) && !found.iter().any(|f| f == id) {
            found.push(id.to_string());
        }
    }
    match found.len() {
        1 => Ok(found.pop().unwrap()),
        0 => {
            // A bare identifier that isn't declared is a typo'd sigma name; keep
            // the historical "unknown sigma" wording. A non-trivial expression
            // that names no sigma is the #484-specific error.
            let trimmed = arg.trim();
            if BARE_IDENT_RE.is_match(trimmed) {
                Err(format!(
                    "[error_model] references unknown sigma '{}' \
                     (declare it in [parameters])",
                    trimmed
                ))
            } else {
                Err(format!(
                    "[error_model] argument `{}` references no declared sigma \
                     (each magnitude expression must scale exactly one sigma parameter)",
                    trimmed
                ))
            }
        }
        _ => Err(format!(
            "[error_model] argument `{}` references multiple sigmas {:?}; \
             a residual-magnitude expression may scale only one sigma",
            arg.trim(),
            found
        )),
    }
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
    residual_error_eta: Option<usize>,
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
        // The `iiv_on_ruv` residual-error eta is referenced from [error_model], not
        // any individual-parameter / [event_model] expression, but it *is* used (it
        // scales the residual variance) and *is* estimated — so don't flag it.
        if Some(i) == residual_error_eta {
            continue;
        }
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

// ─── Shared expression / condition evaluators ───────────────────────────────
//
// `eval_expression`/`eval_condition` (HashMap-keyed name lookup) and
// `eval_expression_indexed`/`eval_condition_indexed` (slice-index lookup) were
// byte-for-byte identical in every arithmetic/unary/power/conditional/literal
// arm; they differed ONLY in how the four "environment" variants
// (`Variable`/`Covariate` vs `VariableIdx`/`CovariateIdx`) resolve a name to an
// `f64`. That name resolution is factored into the `EvalEnv` trait, with a
// HashMap impl (`MapEnv`) and a slice impl (`SliceEnv`), so the arithmetic lives
// once in the generic `eval_expr`/`eval_cond`. Monomorphization keeps both call
// sites zero-cost. Every arithmetic/guard arm is copied verbatim from the
// originals (the `Div` 1e-30 guard, the unary fn set, the `NnOutput` cache
// lookup).

/// Name resolution for the four environment expression variants
/// (`Variable`/`Covariate`/`VariableIdx`/`CovariateIdx`). The generic
/// evaluator routes exactly those variants here; every other arm is handled
/// generically.
trait EvalEnv {
    fn resolve(&self, expr: &Expression) -> f64;
}

/// HashMap-keyed environment (the unindexed evaluator).
struct MapEnv<'a> {
    covariates: &'a HashMap<String, f64>,
    vars: &'a HashMap<String, f64>,
}

impl EvalEnv for MapEnv<'_> {
    #[inline]
    fn resolve(&self, expr: &Expression) -> f64 {
        match expr {
            Expression::Covariate(name) => self.covariates.get(name).copied().unwrap_or(0.0),
            Expression::Variable(name) => {
                if name.eq_ignore_ascii_case("MACHEPS") {
                    f64::EPSILON
                } else {
                    self.vars.get(name).copied().unwrap_or(0.0)
                }
            }
            _ => {
                debug_assert!(
                    false,
                    "indexed expression reached unindexed eval_expression"
                );
                0.0
            }
        }
    }
}

/// Slice-indexed environment (the resolved, hot indexed evaluator).
struct SliceEnv<'a> {
    covariates: &'a [f64],
    vars: &'a [f64],
}

impl EvalEnv for SliceEnv<'_> {
    #[inline]
    fn resolve(&self, expr: &Expression) -> f64 {
        match expr {
            Expression::VariableIdx(i) => self.vars.get(*i).copied().unwrap_or(0.0),
            Expression::CovariateIdx(i) => self.covariates.get(*i).copied().unwrap_or(0.0),
            _ => {
                debug_assert!(false, "unresolved name reached eval_expression_indexed");
                0.0
            }
        }
    }
}

/// Generic expression evaluator. All arithmetic/unary/power/conditional/literal
/// arms are shared; the environment variants delegate to `env.resolve`.
fn eval_expr<E: EvalEnv>(
    expr: &Expression,
    theta: &[f64],
    eta: &[f64],
    env: &E,
    nn_outputs: &[Vec<f64>],
) -> f64 {
    match expr {
        Expression::Literal(v) => *v,
        Expression::Theta(i) => theta[*i],
        Expression::Eta(i) => eta[*i],
        Expression::Time => current_model_time(),
        Expression::MixNum => current_mixture_class() as f64,
        Expression::Variable(_)
        | Expression::Covariate(_)
        | Expression::VariableIdx(_)
        | Expression::CovariateIdx(_) => env.resolve(expr),
        Expression::BinOp(lhs, op, rhs) => {
            let l = eval_expr(lhs, theta, eta, env, nn_outputs);
            let r = eval_expr(rhs, theta, eta, env, nn_outputs);
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
            let v = eval_expr(arg, theta, eta, env, nn_outputs);
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
            let b = eval_expr(base, theta, eta, env, nn_outputs);
            let e = eval_expr(exp, theta, eta, env, nn_outputs);
            b.powf(e)
        }
        Expression::Conditional(cond, t, e) => {
            if eval_cond(cond, theta, eta, env, nn_outputs) {
                eval_expr(t, theta, eta, env, nn_outputs)
            } else {
                eval_expr(e, theta, eta, env, nn_outputs)
            }
        }
        Expression::NnOutput { nn_idx, output_idx } => {
            // Per-call cache populated once per forward via
            // `NamedMlpMapper::forward_raw`. Multiple references to outputs of
            // the same NN therefore share a single forward pass. Out-of-bounds
            // indices return 0.0 with a debug-assert so a logic bug surfaces in
            // tests but doesn't crash release builds.
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

/// Generic condition evaluator (shared by the HashMap and slice entry points).
fn eval_cond<E: EvalEnv>(
    cond: &Condition,
    theta: &[f64],
    eta: &[f64],
    env: &E,
    nn_outputs: &[Vec<f64>],
) -> bool {
    match cond {
        Condition::Compare(l, op, r) => {
            let lv = eval_expr(l, theta, eta, env, nn_outputs);
            let rv = eval_expr(r, theta, eta, env, nn_outputs);
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
            eval_cond(l, theta, eta, env, nn_outputs) && eval_cond(r, theta, eta, env, nn_outputs)
        }
        Condition::Or(l, r) => {
            eval_cond(l, theta, eta, env, nn_outputs) || eval_cond(r, theta, eta, env, nn_outputs)
        }
        Condition::Not(c) => !eval_cond(c, theta, eta, env, nn_outputs),
    }
}

#[inline]
fn eval_expression(
    expr: &Expression,
    theta: &[f64],
    eta: &[f64],
    covariates: &HashMap<String, f64>,
    vars: &HashMap<String, f64>,
    nn_outputs: &[Vec<f64>],
) -> f64 {
    let env = MapEnv { covariates, vars };
    eval_expr(expr, theta, eta, &env, nn_outputs)
}

#[inline]
fn eval_condition(
    cond: &Condition,
    theta: &[f64],
    eta: &[f64],
    covariates: &HashMap<String, f64>,
    vars: &HashMap<String, f64>,
    nn_outputs: &[Vec<f64>],
) -> bool {
    let env = MapEnv { covariates, vars };
    eval_cond(cond, theta, eta, &env, nn_outputs)
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
    PushTime,
    PushMixNum,
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
//   - `build_pk_param_fn`       : `pk_cov` + `pk_vars` (individual params)
//   - `build_y_output_fn`       : `y_vars` + `y_cov` (Form C readout)
//   - indexed statement eval    : `bc_stack` (bytecode f64 stack)
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
    pk_cov: Vec<f64>,
    pk_vars: Vec<f64>,
    y_vars: Vec<f64>,
    y_cov: Vec<f64>,
    bc_stack: Vec<f64>,
}

impl FerxThreadScratch {
    const fn new_empty() -> Self {
        Self {
            rhs_vars: Vec::new(),
            pk_cov: Vec::new(),
            pk_vars: Vec::new(),
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
            | Op::PushTime
            | Op::PushMixNum
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
        Expression::Time => bc.ops.push(Op::PushTime),
        Expression::MixNum => bc.ops.push(Op::PushMixNum),
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
            Op::PushTime => push!(current_model_time()),
            Op::PushMixNum => push!(current_mixture_class() as f64),
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

/// Generic counterpart to [`eval_bytecode`] for the analytic-sensitivity path:
/// the same `Bytecode`, evaluated over any [`PkNum`] `T` so a single program
/// serves both the scalar value (`T = f64`) and its exact PK-parameter
/// derivatives (`T = Dual2<N>`). Only `vars` carries `T` — that slice holds the
/// ODE state and individual parameters, the things seeded as dual variables.
/// `theta`/`eta`/`covariates`/`nn_outputs` are lifted as constants (we do not
/// differentiate w.r.t. them here; η/θ enter through the provider's outer chain).
///
/// **This is a second evaluator and must stay semantically identical to
/// [`eval_bytecode`]'s `f64` path** — the smooth ops route through `PkNum`, the
/// guards (`Div`/`Mod`/`Ln`/`Sqrt`) and value-based ops (`Cmp*`/`Logic*`/`Mod`/
/// `Floor`/`Ceil`/`Round`/jumps) branch on `.val()` exactly as the scalar path
/// branches on the `f64`. The `bytecode_g_matches_f64_*` tests pin `T::val()` to
/// `eval_bytecode` so the two cannot drift.
// Wired into the `Dual2`-state ODE integrator in Phase 3 of #367; until then it
// has only test callers, so the non-test build sees it as unused.
#[allow(dead_code)]
fn eval_bytecode_g<T: crate::sens::num::PkNum>(
    bc: &Bytecode,
    theta: &[T],
    eta: &[T],
    covariates: &[f64],
    vars: &[T],
    nn_outputs: &[Vec<f64>],
    stack: &mut Vec<T>,
) -> T {
    stack.clear();
    stack.reserve(bc.max_stack);
    let mut pc: usize = 0;
    let ops = bc.ops.as_slice();
    let consts = bc.constants.as_slice();
    let k = T::from_f64;

    macro_rules! push {
        ($v:expr) => {
            stack.push($v)
        };
    }
    macro_rules! pop {
        () => {
            stack.pop().unwrap_or_else(|| {
                debug_assert!(false, "eval_bytecode_g stack underflow at pc={pc}");
                k(0.0)
            })
        };
    }

    while pc < ops.len() {
        match ops[pc] {
            Op::PushConst(i) => push!(k(consts[i as usize])),
            Op::PushTheta(i) => push!(theta.get(i as usize).copied().unwrap_or_else(|| k(0.0))),
            Op::PushEta(i) => push!(eta.get(i as usize).copied().unwrap_or_else(|| k(0.0))),
            Op::PushTime => push!(k(current_model_time())),
            Op::PushMixNum => push!(k(current_mixture_class() as f64)),
            Op::PushVar(i) => push!(*vars.get(i as usize).unwrap_or(&k(0.0))),
            Op::PushCov(i) => push!(k(covariates.get(i as usize).copied().unwrap_or(0.0))),
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
                push!(k(v));
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
                push!(if b.val().abs() < 1e-30 { k(0.0) } else { a / b });
            }
            Op::Pow => {
                let e = pop!();
                let b = pop!();
                push!(b.pow(e));
            }
            Op::Mod => {
                let b = pop!();
                let a = pop!();
                // Non-differentiable: compute on values, lift as a constant.
                push!(if b.val().abs() < 1e-30 {
                    k(0.0)
                } else {
                    k(a.val().rem_euclid(b.val()))
                });
            }
            Op::Exp => {
                let v = pop!();
                push!(v.exp());
            }
            Op::Ln => {
                let v = pop!();
                push!(v.guard_floor(1e-30).ln());
            }
            Op::Sqrt => {
                let v = pop!();
                push!(v.guard_floor(0.0).sqrt());
            }
            Op::Abs => {
                let v = pop!();
                push!(v.abs());
            }
            Op::Floor => {
                let v = pop!();
                push!(k(v.val().floor()));
            }
            Op::Ceil => {
                let v = pop!();
                push!(k(v.val().ceil()));
            }
            Op::Round => {
                let v = pop!();
                push!(k(v.val().round()));
            }
            Op::InvLogit => {
                let v = pop!();
                push!(v.inv_logit());
            }
            Op::Logit => {
                let v = pop!();
                // Match the scalar path's clamp to (0,1); the clamped region is
                // flat (zero jet for duals).
                let r = if v.val() <= 1e-15 {
                    k((1e-15_f64 / (1.0 - 1e-15)).ln())
                } else if v.val() >= 1.0 - 1e-15 {
                    k(((1.0 - 1e-15_f64) / 1e-15).ln())
                } else {
                    v.logit()
                };
                push!(r);
            }
            Op::CmpLt => {
                let r = pop!();
                let l = pop!();
                push!(k(if l.val() < r.val() { 1.0 } else { 0.0 }));
            }
            Op::CmpLe => {
                let r = pop!();
                let l = pop!();
                push!(k(if l.val() <= r.val() { 1.0 } else { 0.0 }));
            }
            Op::CmpGt => {
                let r = pop!();
                let l = pop!();
                push!(k(if l.val() > r.val() { 1.0 } else { 0.0 }));
            }
            Op::CmpGe => {
                let r = pop!();
                let l = pop!();
                push!(k(if l.val() >= r.val() { 1.0 } else { 0.0 }));
            }
            Op::CmpEq => {
                let r = pop!();
                let l = pop!();
                push!(k(if l.val() == r.val() { 1.0 } else { 0.0 }));
            }
            Op::CmpNe => {
                let r = pop!();
                let l = pop!();
                push!(k(if l.val() != r.val() { 1.0 } else { 0.0 }));
            }
            Op::LogicAnd => {
                let b = pop!();
                let a = pop!();
                push!(k(if a.val() != 0.0 && b.val() != 0.0 {
                    1.0
                } else {
                    0.0
                }));
            }
            Op::LogicOr => {
                let b = pop!();
                let a = pop!();
                push!(k(if a.val() != 0.0 || b.val() != 0.0 {
                    1.0
                } else {
                    0.0
                }));
            }
            Op::LogicNot => {
                let v = pop!();
                push!(k(if v.val() == 0.0 { 1.0 } else { 0.0 }));
            }
            Op::JumpIfFalse(target) => {
                let v = pop!();
                if v.val() == 0.0 {
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
    debug_assert!(
        stack.len() == 1,
        "eval_bytecode_g: bytecode finished at stack depth {}, expected 1",
        stack.len()
    );
    stack.pop().unwrap_or_else(|| k(0.0))
}

/// Indexed-form evaluator: `vars` is a `Vec<f64>` indexed by parse-time
/// variable slot; `covariates` is a `Vec<f64>` aligned to
/// `CompiledModel.referenced_covariates`. Hot-path replacement for the
/// HashMap-keyed `eval_expression` — eliminates the per-call string hash
/// + probe in the `pk_param_fn` closure. Falls back to 0.0 for the
/// HashMap-keyed variants (Variable/Covariate) since callers running
/// the indexed path have already resolved every name.
#[inline]
fn eval_expression_indexed(
    expr: &Expression,
    theta: &[f64],
    eta: &[f64],
    covariates: &[f64],
    vars: &[f64],
    nn_outputs: &[Vec<f64>],
) -> f64 {
    let env = SliceEnv { covariates, vars };
    eval_expr(expr, theta, eta, &env, nn_outputs)
}

#[inline]
fn eval_condition_indexed(
    cond: &Condition,
    theta: &[f64],
    eta: &[f64],
    covariates: &[f64],
    vars: &[f64],
    nn_outputs: &[Vec<f64>],
) -> bool {
    let env = SliceEnv { covariates, vars };
    eval_cond(cond, theta, eta, &env, nn_outputs)
}

/// Inner statement evaluator threaded with a caller-owned bytecode stack.
/// Recursive `If` evaluation reuses the same `Vec<f64>` scratch instead of
/// re-acquiring the TLS borrow per nested call.
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

/// [`eval_statements_indexed_with_stack`] generic over `T: PkNum`, for the ODE
/// sensitivity RHS (`T = Dual2<N>`). The resolved ODE RHS only ever contains
/// `AssignBc`/`DiffEqBc`/`If`, so the smooth assignments route through
/// [`eval_bytecode_g`] and `If` conditions — value-based branch decisions —
/// evaluate on a `.val()` view via the existing scalar [`eval_condition_indexed`]
/// (theta/eta/cov/nn are empty in the ODE RHS). Mirrors the f64 evaluator
/// statement-for-statement; the `eval_statements_g_*` tests pin it to that path.
// Wired into the `Dual2`-state ODE RHS in Phase 3 of #367; test-only caller so far.
#[allow(dead_code)]
fn eval_statements_g<T: crate::sens::num::PkNum>(
    stmts: &[Statement],
    theta: &[T],
    eta: &[T],
    cov: &[f64],
    vars: &mut [T],
    du: Option<&mut [T]>,
    bc_stack: &mut Vec<T>,
    // Per var-slot mask: when `skip[idx]` is `true`, the `AssignBc(idx, _)` for
    // that slot is NOT re-evaluated — its value is assumed already present in
    // `vars` (the cov-static constant-folding pre-seed, #485). Empty slice =
    // skip nothing (every other caller passes `&[]`). Conditions are still
    // evaluated and branches still descended, so control flow is identical.
    skip: &[bool],
) {
    let empty_nn: Vec<Vec<f64>> = Vec::new();
    let mut du_opt = du;
    for s in stmts {
        match s {
            Statement::AssignBc(idx, bc) => {
                if skip.get(*idx).copied().unwrap_or(false) {
                    continue;
                }
                let v = eval_bytecode_g::<T>(bc, theta, eta, cov, vars, &empty_nn, bc_stack);
                if let Some(slot) = vars.get_mut(*idx) {
                    *slot = v;
                }
            }
            Statement::DiffEqBc(state_idx, bc) => {
                let v = eval_bytecode_g::<T>(bc, theta, eta, cov, vars, &empty_nn, bc_stack);
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
                // Conditions are value-based; evaluate on `.val()` views.
                let theta_val: Vec<f64> = theta.iter().map(|v| v.val()).collect();
                let eta_val: Vec<f64> = eta.iter().map(|v| v.val()).collect();
                let vars_val: Vec<f64> = vars.iter().map(|v| v.val()).collect();
                let mut taken = false;
                for (cond, body) in branches {
                    if eval_condition_indexed(cond, &theta_val, &eta_val, cov, &vars_val, &empty_nn)
                    {
                        eval_statements_g::<T>(
                            body,
                            theta,
                            eta,
                            cov,
                            vars,
                            du_opt.as_deref_mut(),
                            bc_stack,
                            skip,
                        );
                        taken = true;
                        break;
                    }
                }
                if !taken {
                    if let Some(eb) = else_body {
                        eval_statements_g::<T>(
                            eb,
                            theta,
                            eta,
                            cov,
                            vars,
                            du_opt.as_deref_mut(),
                            bc_stack,
                            skip,
                        );
                    }
                }
            }
            Statement::AssignIdx(_, _)
            | Statement::DiffEqIdx(_, _)
            | Statement::Assign(_, _)
            | Statement::DiffEq(_, _) => {}
        }
    }
}

/// Compiled ODE RHS program + var layout, exposed so the analytic-sensitivity
/// provider can evaluate the same RHS over a dual type (issue #367, Option A).
/// Carries exactly what the f64 `rhs` closure binds, so [`eval_rhs_g`]
/// reproduces that binding but seeds the individual parameters as dual variables.
///
/// [`eval_rhs_g`]: OdeRhsProgram::eval_rhs_g
///
/// `pub` only so it can appear in the (public) [`OdeSpec`](crate::ode::OdeSpec)
/// field; all fields are private, so it is opaque outside the crate and can be
/// produced only by the parser.
/// Recursively test whether an expression reads any of `slots` (post-resolution, time/TAD
/// anchors appear as `VariableIdx`). Conservative — used to detect a non-autonomous RHS.
fn expr_reads_slots(e: &Expression, slots: &[usize]) -> bool {
    match e {
        Expression::VariableIdx(i) => slots.contains(i),
        Expression::BinOp(a, _, b) | Expression::Power(a, b) => {
            expr_reads_slots(a, slots) || expr_reads_slots(b, slots)
        }
        Expression::UnaryFn(_, a) => expr_reads_slots(a, slots),
        Expression::Conditional(c, t, f) => {
            cond_reads_slots(c, slots) || expr_reads_slots(t, slots) || expr_reads_slots(f, slots)
        }
        _ => false,
    }
}

fn cond_reads_slots(c: &Condition, slots: &[usize]) -> bool {
    match c {
        Condition::Compare(a, _, b) => expr_reads_slots(a, slots) || expr_reads_slots(b, slots),
        Condition::And(a, b) | Condition::Or(a, b) => {
            cond_reads_slots(a, slots) || cond_reads_slots(b, slots)
        }
        Condition::Not(a) => cond_reads_slots(a, slots),
    }
}

/// Whether the RHS statements read any of `slots` — scans both the compiled bytecode
/// (`PushVar`) and any `if`-condition / fallback expressions. Used at construction to set
/// [`OdeRhsProgram::uses_time_vars`].
fn stmts_read_slots(stmts: &[Statement], slots: &[usize]) -> bool {
    stmts.iter().any(|s| match s {
        Statement::AssignBc(_, bc) | Statement::DiffEqBc(_, bc) => bc
            .ops
            .iter()
            .any(|op| matches!(op, Op::PushVar(i) if slots.contains(&(*i as usize)))),
        Statement::Assign(_, e)
        | Statement::AssignIdx(_, e)
        | Statement::DiffEq(_, e)
        | Statement::DiffEqIdx(_, e) => expr_reads_slots(e, slots),
        Statement::If {
            branches,
            else_body,
        } => {
            branches
                .iter()
                .any(|(c, b)| cond_reads_slots(c, slots) || stmts_read_slots(b, slots))
                || else_body
                    .as_deref()
                    .is_some_and(|b| stmts_read_slots(b, slots))
        }
    })
}

pub struct OdeRhsProgram {
    stmts: Vec<Statement>,
    n_vars_total: usize,
    state_count: usize,
    /// Per individual parameter `i`, the slot in the flat `params` vector it is
    /// read from (its PK slot). Same plan the f64 `rhs` closure uses.
    indiv_to_params_slot: Vec<usize>,
    time_slot: usize,
    tafd_slot: usize,
    tad_slot: usize,
    macheps_slot: usize,
    /// True when the RHS reads any of the `TIME`/`TAFD`/`TAD` time-anchor slots — i.e. the
    /// system is **non-autonomous**. A steady-state (`SS=1`) dose assumes a time-invariant
    /// pulse train, so the SS dual equilibration's cycle recurrence breaks for such an RHS;
    /// the gate routes SS subjects on a non-autonomous RHS to FD (#473 review #1).
    uses_time_vars: bool,
}

impl OdeRhsProgram {
    /// See [`OdeRhsProgram::uses_time_vars`]. `true` ⇒ a steady-state dose must route to FD.
    pub(crate) fn uses_time_vars(&self) -> bool {
        self.uses_time_vars
    }

    /// Evaluate `du = f(u, p, t)` over a dual type, generic over [`PkNum`]
    /// (`Dual1<N>` for the light inner η-gradient, `Dual2<N>` for the full outer
    /// gradient). `u` is the current state; `params` is the flat PK-parameter vector
    /// with the differentiated slots already seeded as dual variables (so individual
    /// parameter `i`, read from `params[indiv_to_params_slot[i]]`, carries its
    /// derivative); `tafd`/`tad` are the time-after-first/last-dose anchors
    /// (constants w.r.t. the parameters, lifted as such). `vars`/`stack` are
    /// caller-owned scratch reused across RK stages. Writes `du` (length
    /// `state_count`). Mirrors the f64 binding in the `rhs` closure
    /// statement-for-statement (issue #410, inner η-gradient).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn eval_rhs_g<T: crate::sens::num::PkNum>(
        &self,
        u: &[T],
        params: &[T],
        t: f64,
        tafd: f64,
        tad: f64,
        du: &mut [T],
        vars: &mut Vec<T>,
        stack: &mut Vec<T>,
    ) {
        vars.clear();
        vars.resize(self.n_vars_total, T::from_f64(0.0));
        let copy_n = self.state_count.min(u.len());
        vars[..copy_n].copy_from_slice(&u[..copy_n]);
        for (i, &slot) in self.indiv_to_params_slot.iter().enumerate() {
            if let (Some(dst), Some(&val)) = (vars.get_mut(self.state_count + i), params.get(slot))
            {
                *dst = val;
            }
        }
        if let Some(d) = vars.get_mut(self.time_slot) {
            *d = T::from_f64(t);
        }
        if let Some(d) = vars.get_mut(self.tafd_slot) {
            *d = T::from_f64(tafd);
        }
        if let Some(d) = vars.get_mut(self.tad_slot) {
            *d = T::from_f64(tad);
        }
        if let Some(d) = vars.get_mut(self.macheps_slot) {
            *d = T::from_f64(f64::EPSILON);
        }
        for d in du.iter_mut() {
            *d = T::from_f64(0.0);
        }
        // The ODE RHS references states/indiv-params (in `vars`) only, not
        // θ/η/cov directly — so those are empty here.
        // A bare `TIME` resolves to `Op::PushTime` (the model-time
        // thread-local), not the `time_slot` used by the `T`/`t` aliases; set
        // it to the integrator clock so `TIME` matches `T` (#610).
        let _time_guard = ModelTimeGuard::enter(t);
        eval_statements_g::<T>(&self.stmts, &[], &[], &[], vars, Some(du), stack, &[]);
    }
}

/// Does this compiled expression's value vary across the dual axes the
/// sensitivity providers differentiate — i.e. does it read η, *any* θ, an NN
/// output, or a variable already known to be dynamic? Reading only covariates,
/// literals, and other cov-static variables makes it cov-static (foldable to a
/// constant). See [`compute_cov_static_mask`] (#485).
///
/// θ is treated as dynamic even when FIXED: the `Dual2` path seeds *every* θ
/// (FIXED included) as a variable, so a FIXED-θ-dependent slot still carries a
/// non-zero `∂/∂θ_fixed` column that folding to a constant would silently zero.
/// Restricting the fold to genuinely θ-free slots keeps it bit-identical to the
/// unfolded walk on all axes.
fn bytecode_is_dynamic(bc: &Bytecode, dyn_vars: &[bool]) -> bool {
    bc.ops.iter().any(|op| match op {
        Op::PushEta(_)
        | Op::PushTheta(_)
        | Op::PushTime
        | Op::PushMixNum
        | Op::PushNnOutput(_, _) => true,
        Op::PushVar(i) => dyn_vars.get(*i as usize).copied().unwrap_or(false),
        _ => false,
    })
}

/// Condition counterpart of [`bytecode_is_dynamic`], over the (resolved) AST a
/// branch condition still carries. An unresolved `Variable` is treated as
/// dynamic (conservative — it never appears in a resolved program).
fn expr_is_dynamic(e: &Expression, dyn_vars: &[bool]) -> bool {
    match e {
        Expression::Eta(_)
        | Expression::Theta(_)
        | Expression::Time
        | Expression::MixNum
        | Expression::NnOutput { .. } => true,
        Expression::Variable(_) => true,
        Expression::VariableIdx(i) => dyn_vars.get(*i).copied().unwrap_or(false),
        Expression::Literal(_) | Expression::Covariate(_) | Expression::CovariateIdx(_) => false,
        Expression::BinOp(a, _, b) | Expression::Power(a, b) => {
            expr_is_dynamic(a, dyn_vars) || expr_is_dynamic(b, dyn_vars)
        }
        Expression::UnaryFn(_, a) => expr_is_dynamic(a, dyn_vars),
        Expression::Conditional(c, a, b) => {
            cond_is_dynamic(c, dyn_vars)
                || expr_is_dynamic(a, dyn_vars)
                || expr_is_dynamic(b, dyn_vars)
        }
    }
}

fn cond_is_dynamic(c: &Condition, dyn_vars: &[bool]) -> bool {
    match c {
        Condition::Compare(l, _, r) => expr_is_dynamic(l, dyn_vars) || expr_is_dynamic(r, dyn_vars),
        Condition::And(a, b) | Condition::Or(a, b) => {
            cond_is_dynamic(a, dyn_vars) || cond_is_dynamic(b, dyn_vars)
        }
        Condition::Not(c) => cond_is_dynamic(c, dyn_vars),
    }
}

/// Classify each individual-parameter var slot as **cov-static** (`true`): its
/// value is constant across every (θ, η) axis the analytic sensitivity providers
/// differentiate, so the `Dual2`/`Dual1` walks can pre-seed it as a constant
/// computed once in plain `f64` instead of carrying it through dual arithmetic
/// (#485). A slot is cov-static iff every assignment to it reads only covariates,
/// literals, and other cov-static slots, AND it is never assigned under an `if`
/// whose governing condition is itself dynamic.
///
/// Computed by monotone fixpoint over the `dynamic` complement: a pass marks a
/// slot dynamic when its bytecode, an enclosing condition, or a referenced slot
/// is dynamic; passes repeat until no slot flips (var refs can be forward, so a
/// single pass is not enough). `n_vars` is small (tens), so this is cheap and
/// runs once at compile time. The returned mask has length `n_vars`.
fn compute_cov_static_mask(stmts: &[Statement], n_vars: usize) -> Vec<bool> {
    fn walk(stmts: &[Statement], ctx_dynamic: bool, dyn_vars: &mut [bool], changed: &mut bool) {
        for s in stmts {
            match s {
                Statement::AssignBc(idx, bc) => {
                    let already = dyn_vars.get(*idx).copied().unwrap_or(true);
                    if !already && (ctx_dynamic || bytecode_is_dynamic(bc, dyn_vars)) {
                        if let Some(slot) = dyn_vars.get_mut(*idx) {
                            *slot = true;
                            *changed = true;
                        }
                    }
                }
                Statement::If {
                    branches,
                    else_body,
                } => {
                    // The `else` arm fires only when *every* branch condition was
                    // false, so it is governed by all of them: dynamic if any is.
                    let mut any_cond_dynamic = ctx_dynamic;
                    for (cond, body) in branches {
                        let branch_dynamic = ctx_dynamic || cond_is_dynamic(cond, dyn_vars);
                        any_cond_dynamic |= branch_dynamic;
                        walk(body, branch_dynamic, dyn_vars, changed);
                    }
                    if let Some(eb) = else_body {
                        walk(eb, any_cond_dynamic, dyn_vars, changed);
                    }
                }
                _ => {}
            }
        }
    }

    let mut dyn_vars = vec![false; n_vars];
    loop {
        let mut changed = false;
        walk(stmts, false, &mut dyn_vars, &mut changed);
        if !changed {
            break;
        }
    }
    dyn_vars.iter().map(|&d| !d).collect()
}

/// Widest `(θ, η)` axis count the CTMM generator's dual dispatch specializes.
/// Above this, [`CtmmGeneratorProgram::eval_generator_duals`] returns `None` and the
/// endpoint keeps its finite-difference gradient — a fallback, never a wrong answer.
#[cfg(feature = "markov")]
pub(crate) const MAX_CTMM_AXES: usize = 24;

/// Compiled, **dual-evaluable** `[markov_model]` transition intensities — the CTMM
/// analogue of [`IndivParamProgram`] (#759).
///
/// The f64 [`GeneratorFn`](crate::types::GeneratorFn) tree-walks `Expression`s and can
/// only ever produce `Q` itself, which is why the CTMM likelihood has been finite-
/// differenced end-to-end (perturb η, rebuild `Q`, redo an `expm` per observation gap).
/// This snapshot carries the same intensities in *resolved* form — the shared
/// `Bytecode` the rest of the analytic-sensitivity path uses — so they can be replayed
/// over `Dual1<M>` with θ and η seeded, yielding an exact `∂Q/∂(θ,η)`.
///
/// Chained through the Van Loan Fréchet derivative of `expm`
/// ([`markov::matrix_exp_frechet`](crate::markov::matrix_exp_frechet)) this gives the exact
/// likelihood gradient. Note `∂Q/∂p` inherits the generator's **row-sum-zero constraint**
/// for free: every intensity is added at its off-diagonal entry and subtracted from the
/// diagonal, and differentiation is linear, so the derivative of a valid generator is a
/// valid *direction*. That is the constraint `generator_rate_direction` hand-builds for the
/// one-rate-at-a-time case, and the reason a naive "one gradient per `Q` entry" API is wrong
/// (see the `markov` module docs).
///
/// Time-**inhomogeneous** endpoints (an intensity reading an ODE state) get no program:
/// their likelihood is an occupancy ODE, not an `expm`, so Van Loan does not apply.
#[cfg(feature = "markov")]
#[derive(Debug, Clone)]
pub struct CtmmGeneratorProgram {
    /// Resolved `[individual_parameters]` statements the intensities read, in
    /// dependency order (already restricted to the transitively-referenced subset).
    stmts: Vec<Statement>,
    /// Var-slot count the statements above write into.
    n_vars: usize,
    /// `(from, to, intensity)` — one compiled off-diagonal rate per `transition` line.
    rates: Vec<(usize, usize, Bytecode)>,
    n_states: usize,
    /// Covariate names in the generator's own (sorted) order, for the cov slice.
    cov_names: Vec<String>,
    n_theta: usize,
    /// BSV η count. IOV kappas are rejected in an intensity at parse, so this is the
    /// full η width the intensities can read.
    n_eta: usize,
}

#[cfg(feature = "markov")]
impl CtmmGeneratorProgram {
    /// Dual width needed to seed every `(θ, η)` axis: `n_theta + n_eta`.
    pub(crate) fn n_axes(&self) -> usize {
        self.n_theta + self.n_eta
    }
    pub(crate) fn n_theta_axis(&self) -> usize {
        self.n_theta
    }

    /// `Q` and `∂Q/∂(θ, η)` at a covariate snapshot, evaluated over `Dual1<M>` with
    /// θ_m on axis `m` and η_k on axis `n_theta + k` (the same axis convention
    /// [`IndivParamProgram::eval_param_duals`] uses).
    ///
    /// Returns `(q, dq)` where `q` is the `S×S` generator and `dq[a]` is `∂Q/∂(axis a)`,
    /// length `n_axes()`. Both are built by placing each intensity — value and jet — at
    /// its off-diagonal entry and subtracting it from the diagonal, so each `dq[a]` has
    /// zero row sums by construction.
    ///
    /// `None` above the dispatch cap ([`MAX_CTMM_AXES`]): the endpoint then keeps its FD
    /// gradient rather than a wrong one.
    pub(crate) fn eval_generator_duals(
        &self,
        theta: &[f64],
        eta: &[f64],
        covariates: &HashMap<String, f64>,
    ) -> Option<(nalgebra::DMatrix<f64>, Vec<nalgebra::DMatrix<f64>>)> {
        macro_rules! disp {
            ($($m:literal),+) => {
                match self.n_axes() {
                    $($m => Some(self.eval_generator_duals_n::<$m>(theta, eta, covariates)),)+
                    _ => None,
                }
            };
        }
        disp!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)
    }

    fn eval_generator_duals_n<const M: usize>(
        &self,
        theta: &[f64],
        eta: &[f64],
        covariates: &HashMap<String, f64>,
    ) -> (nalgebra::DMatrix<f64>, Vec<nalgebra::DMatrix<f64>>) {
        use crate::sens::dual1::Dual1;
        debug_assert_eq!(M, self.n_axes());
        let theta_d: Vec<Dual1<M>> = theta
            .iter()
            .enumerate()
            .map(|(m, &v)| {
                if m < M {
                    Dual1::var(v, m)
                } else {
                    // A θ the intensities cannot reference anyway (the program is sized by
                    // the model's declared θ count, so this is unreachable in practice).
                    Dual1::constant(v)
                }
            })
            .collect();
        let eta_d: Vec<Dual1<M>> = eta
            .iter()
            .enumerate()
            .map(|(k, &v)| {
                let dim = self.n_theta + k;
                if dim < M {
                    Dual1::var(v, dim)
                } else {
                    // IOV kappas are appended past the BSV η block and are rejected in an
                    // intensity at parse, so they are correctly seeded as constants here.
                    Dual1::constant(v)
                }
            })
            .collect();
        let cov_vec: Vec<f64> = self
            .cov_names
            .iter()
            .map(|n| covariates.get(n).copied().unwrap_or(0.0))
            .collect();

        let mut vars = vec![Dual1::<M>::constant(0.0); self.n_vars];
        let mut stack: Vec<Dual1<M>> = Vec::new();
        eval_statements_g::<Dual1<M>>(
            &self.stmts,
            &theta_d,
            &eta_d,
            &cov_vec,
            &mut vars,
            None,
            &mut stack,
            &[],
        );

        let s = self.n_states;
        let n_axes = self.n_axes();
        let mut q = nalgebra::DMatrix::<f64>::zeros(s, s);
        let mut dq = vec![nalgebra::DMatrix::<f64>::zeros(s, s); n_axes];
        let empty_nn: Vec<Vec<f64>> = Vec::new();
        for (j, k, bc) in &self.rates {
            let rate = eval_bytecode_g::<Dual1<M>>(
                bc, &theta_d, &eta_d, &cov_vec, &vars, &empty_nn, &mut stack,
            );
            q[(*j, *k)] += rate.value;
            q[(*j, *j)] -= rate.value;
            for (a, dq_a) in dq.iter_mut().enumerate().take(n_axes) {
                let g = rate.grad[a];
                dq_a[(*j, *k)] += g;
                dq_a[(*j, *j)] -= g;
            }
        }
        (q, dq)
    }
}

/// Resolve + compile a `[markov_model]`'s intensities into a [`CtmmGeneratorProgram`].
/// `None` when the program cannot be represented exactly, in which case the endpoint keeps
/// its FD gradient: an intensity or an `[individual_parameters]` value that reads an
/// identifier we cannot bind (which would silently lower to a constant `0.0` push and thus
/// a silently-wrong derivative), or a `(θ, η)` width past the dual dispatch cap.
#[cfg(feature = "markov")]
fn build_ctmm_generator_program(
    transitions: &[(usize, usize, Expression)],
    needed_indiv_stmts: &[Statement],
    generator_covariates: &[String],
    n_states: usize,
    n_theta: usize,
    n_eta: usize,
) -> Option<CtmmGeneratorProgram> {
    if n_theta + n_eta == 0 || n_theta + n_eta > MAX_CTMM_AXES {
        return None;
    }

    // Var slots for the `[individual_parameters]` values the intensities read, in the
    // order they are assigned (so a later value may read an earlier one).
    let var_names = assigned_vars_in_order(needed_indiv_stmts);
    let var_idx: HashMap<String, usize> = var_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();
    let n_vars = var_names.len();
    let cov_idx: HashMap<String, usize> = generator_covariates
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();

    // An unresolved `Variable`/`Covariate` leaf lowers to a constant `0.0` in
    // `compile_bytecode` (its documented fall-through). That is survivable for a *value*
    // path that never sees one, but here it would hand back a confidently wrong gradient,
    // so refuse the program instead and let FD serve the endpoint.
    let resolvable = |e: &Expression| -> bool {
        let mut ok = true;
        visit_expr_nodes(e, &mut |node: &Expression| match node {
            Expression::Variable(name) => ok &= var_idx.contains_key(name),
            Expression::Covariate(name) => ok &= cov_idx.contains_key(name),
            _ => {}
        });
        ok
    };
    if !transitions.iter().all(|(_, _, e)| resolvable(e)) {
        return None;
    }
    let mut indiv_ok = true;
    visit_stmt_nodes(needed_indiv_stmts, &mut |e: &Expression| {
        indiv_ok &= resolvable(e);
    });
    if !indiv_ok {
        return None;
    }

    let mut stmts: Vec<Statement> = needed_indiv_stmts.to_vec();
    resolve_variable_indices(&mut stmts, &var_idx, &cov_idx, None);

    let rates: Vec<(usize, usize, Bytecode)> = transitions
        .iter()
        .map(|(j, k, expr)| {
            let mut e = expr.clone();
            resolve_expr_indices(&mut e, &var_idx, &cov_idx);
            (*j, *k, compile_bytecode(&e))
        })
        .collect();

    Some(CtmmGeneratorProgram {
        stmts,
        n_vars,
        rates,
        n_states,
        cov_names: generator_covariates.to_vec(),
        n_theta,
        n_eta,
    })
}

/// Compiled `[individual_parameters]` block + var layout, exposed so the
/// analytic-sensitivity provider can obtain `∂p/∂η`, `∂p/∂θ` (and second order)
/// **analytically** — by evaluating the same statements over `Dual2<M>` seeded on
/// (θ, η) — instead of finite-differencing `pk_param_fn` (issue #367). Mirrors the
/// f64 binding in the `pk_param_fn` closure.
#[derive(Debug, Clone)]
pub struct IndivParamProgram {
    stmts: Vec<Statement>,
    n_vars: usize,
    /// Per var-slot mask: `true` = the slot is cov-static (constant across every
    /// θ / η dual axis), so [`eval_param_duals`](Self::eval_param_duals) and
    /// [`eval_param_eta_grad`](Self::eval_param_eta_grad) compute it once in `f64`
    /// and seed it as a dual constant instead of re-running the (often pow/exp/log
    /// heavy) covariate kernel under dual arithmetic every call (#485). Empty when
    /// no slot is foldable, in which case both paths run exactly as before.
    cov_static_mask: Vec<bool>,
    /// `(pk_slot, var_slot)` per individual parameter, in declaration order
    /// (parallel to `CompiledModel.pk_indices`).
    pk_var_slots: Vec<(usize, usize)>,
    /// `pk_var_slots` projected to just the PK slots — the [`pk_slots`](Self::pk_slots)
    /// value, cached at construction so the hot analytic paths borrow it via
    /// [`pk_slots_ref`](Self::pk_slots_ref) instead of re-collecting a `Vec` on every
    /// per-gradient evaluation (#534 review #2).
    pk_slot_indices: Vec<usize>,
    /// User-declared θ count the individual parameters can reference.
    n_theta: usize,
    /// η count the `pk_param_fn` consumes (BSV + IOV kappa).
    n_eta: usize,
    /// Covariate names in `referenced_covariates` order (for the cov slice).
    cov_names: Vec<String>,
    /// Whether `[individual_parameters]` or a direct analytical `pk(...=TIME)`
    /// mapping reads the event-time built-in.
    uses_time_builtin: bool,
}

impl IndivParamProgram {
    /// Dual width needed to seed every (θ, η) axis: `n_theta + n_eta`.
    pub(crate) fn n_axes(&self) -> usize {
        self.n_theta + self.n_eta
    }
    /// User-declared θ count the individual parameters reference (the dual seed
    /// dimension of η axis `k` is `n_theta_axis() + k`).
    pub(crate) fn n_theta_axis(&self) -> usize {
        self.n_theta
    }
    /// η count the individual parameters reference (BSV + IOV kappa).
    pub(crate) fn n_eta_axis(&self) -> usize {
        self.n_eta
    }

    /// The PK slot each row of [`eval_param_duals`](Self::eval_param_duals) /
    /// [`pd_from_program`](crate::sens::ode_provider::pd_from_program) corresponds
    /// to, in the program's own (analytical: alphabetical-by-PK-name; ODE:
    /// declaration) order. The analytical provider uses this to pair each `∂p/∂·`
    /// row with the right slot of the 8-slot PK gradient — `pk_var_slots` order is
    /// NOT `CompiledModel.pk_indices` (declaration) order.
    pub(crate) fn pk_slots(&self) -> Vec<usize> {
        self.pk_var_slots.iter().map(|&(slot, _)| slot).collect()
    }

    /// Borrowing form of [`pk_slots`](Self::pk_slots): the PK-slot list cached at
    /// construction, for hot paths that only read it (no per-call `Vec`). Identical
    /// contents to `pk_slots()` (#534 review #2).
    pub(crate) fn pk_slots_ref(&self) -> &[usize] {
        &self.pk_slot_indices
    }

    /// Evaluate the individual parameters over `Dual2<M>` seeded on (θ, η):
    /// `θ_m → var(·, m)`, `η_k → var(·, n_theta + k)`. Returns one `Dual2<M>` per
    /// individual parameter (declaration order), whose `grad`/`hess` are the exact
    /// `∂p/∂(θ,η)` / `∂²p/∂(θ,η)²`. Requires `M ≥ n_axes()`.
    pub(crate) fn eval_param_duals<const M: usize>(
        &self,
        theta: &[f64],
        eta: &[f64],
        covariates: &HashMap<String, f64>,
    ) -> Vec<crate::sens::dual2::Dual2<M>> {
        use crate::sens::dual2::Dual2;
        let theta_d: Vec<Dual2<M>> = theta
            .iter()
            .enumerate()
            .map(|(m, &v)| {
                if m < M {
                    Dual2::var(v, m)
                } else {
                    Dual2::constant(v)
                }
            })
            .collect();
        let eta_d: Vec<Dual2<M>> = eta
            .iter()
            .enumerate()
            .map(|(k, &v)| {
                let dim = self.n_theta + k;
                if dim < M {
                    Dual2::var(v, dim)
                } else {
                    Dual2::constant(v)
                }
            })
            .collect();
        let cov_vec: Vec<f64> = self
            .cov_names
            .iter()
            .map(|n| covariates.get(n).copied().unwrap_or(0.0))
            .collect();
        let mut vars = vec![Dual2::<M>::constant(0.0); self.n_vars];
        let mut stack: Vec<Dual2<M>> = Vec::new();
        // Cov-static fold (#485): evaluate the constant-across-(θ,η) slots once in
        // plain f64, seed them as dual constants (zero gradient/Hessian — exactly
        // what they carry), and skip their (pow/exp/log heavy) re-derivation in the
        // Dual2 walk. `skip` empty ⇒ original full-dual path.
        let skip: &[bool] = &self.cov_static_mask;
        if !skip.is_empty() {
            let static_vals = self.eval_cov_static_f64(theta, &cov_vec);
            for (v, &is_static) in skip.iter().enumerate() {
                if is_static {
                    vars[v] = Dual2::constant(static_vals[v]);
                }
            }
        }
        eval_statements_g::<Dual2<M>>(
            &self.stmts,
            &theta_d,
            &eta_d,
            &cov_vec,
            &mut vars,
            None,
            &mut stack,
            skip,
        );
        self.pk_var_slots
            .iter()
            .map(|&(_, vs)| vars.get(vs).copied().unwrap_or(Dual2::constant(0.0)))
            .collect()
    }

    /// Compute the f64 values of the cov-static var slots (the slots flagged in
    /// `cov_static_mask`), evaluating only those assignments — the dynamic ones
    /// are skipped, so their slots stay `0.0` and must not be read. Branch
    /// conditions are still evaluated (cheap comparisons); cov-static slots are
    /// only ever governed by cov-static conditions, so the decisions — and hence
    /// the returned values — are independent of the η seed used here (#485).
    fn eval_cov_static_f64(&self, theta: &[f64], cov_vec: &[f64]) -> Vec<f64> {
        use crate::sens::dual2::Dual2;
        let dynamic: Vec<bool> = self.cov_static_mask.iter().map(|&s| !s).collect();
        // Evaluated over a zero-width `Dual2<0>` (no gradient/Hessian axes), NOT
        // native `f64`: the dual divides via `× recip` and raises powers via the
        // dual `powd`, both of which differ from `f64::/` / `f64::powf` by up to
        // 1 ULP. A division-bearing covariate kernel (CKD-EPI, Schwartz
        // `0.413·HEIGHT/CREAT`, FFM) would otherwise make a folded slot drift ~1
        // ULP from the unfolded walk. Both `Dual1<M>` and `Dual2<M>` share the
        // `× recip` division and the same `powd`/`exp`/`ln` value arithmetic, and
        // the `.value` field is computed independently of the axis count, so a
        // `Dual2<0>` `.value` is bit-for-bit equal to what either unfolded path
        // computes for the same slot — keeping the fold exactly identical (#485).
        let theta_d: Vec<Dual2<0>> = theta.iter().map(|&v| Dual2::constant(v)).collect();
        let eta_zero = vec![Dual2::<0>::constant(0.0); self.n_eta];
        let mut vars = vec![Dual2::<0>::constant(0.0); self.n_vars];
        let mut stack: Vec<Dual2<0>> = Vec::new();
        eval_statements_g::<Dual2<0>>(
            &self.stmts,
            &theta_d,
            &eta_zero,
            cov_vec,
            &mut vars,
            None,
            &mut stack,
            &dynamic,
        );
        vars.iter().map(|d| d.value).collect()
    }

    /// Evaluate the individual parameters over `Dual1<M>` seeded on **η only**
    /// (`η_k → var(·, k)`, θ held constant): the light first-order counterpart of
    /// [`eval_param_duals`](Self::eval_param_duals) for the inner η-gradient (#410).
    /// Returns one `Dual1<M>` per individual parameter (declaration order); its
    /// `grad` is the exact `∂p/∂η`. Avoids the θ-axes and the second-order Hessian
    /// the `Dual2` path computes. Requires `M ≥ n_eta_axis()`.
    pub(crate) fn eval_param_eta_grad<const M: usize>(
        &self,
        theta: &[f64],
        eta: &[f64],
        covariates: &HashMap<String, f64>,
    ) -> Vec<crate::sens::dual1::Dual1<M>> {
        use crate::sens::dual1::Dual1;
        let theta_d: Vec<Dual1<M>> = theta.iter().map(|&v| Dual1::constant(v)).collect();
        let eta_d: Vec<Dual1<M>> = eta
            .iter()
            .enumerate()
            .map(|(k, &v)| {
                if k < M {
                    Dual1::var(v, k)
                } else {
                    Dual1::constant(v)
                }
            })
            .collect();
        let cov_vec: Vec<f64> = self
            .cov_names
            .iter()
            .map(|n| covariates.get(n).copied().unwrap_or(0.0))
            .collect();
        let mut vars = vec![Dual1::<M>::constant(0.0); self.n_vars];
        let mut stack: Vec<Dual1<M>> = Vec::new();
        // Cov-static fold (#485): same as the Dual2 path — cov-static slots are a
        // fortiori η-independent, so seeding them as Dual1 constants is exact.
        let skip: &[bool] = &self.cov_static_mask;
        if !skip.is_empty() {
            let static_vals = self.eval_cov_static_f64(theta, &cov_vec);
            for (v, &is_static) in skip.iter().enumerate() {
                if is_static {
                    vars[v] = Dual1::constant(static_vals[v]);
                }
            }
        }
        eval_statements_g::<Dual1<M>>(
            &self.stmts,
            &theta_d,
            &eta_d,
            &cov_vec,
            &mut vars,
            None,
            &mut stack,
            skip,
        );
        self.pk_var_slots
            .iter()
            .map(|&(_, vs)| vars.get(vs).copied().unwrap_or(Dual1::constant(0.0)))
            .collect()
    }
}

/// Compiled Form C ODE output expression (`[scaling] y = <expr>`) + var layout,
/// exposed so the analytic-sensitivity provider can evaluate the readout
/// (e.g. `central / V1`) over `Dual2<N>` — turning the integrated dual state and
/// the dual PK parameters into the scaled observable with exact derivatives
/// (issue #367, Option A). Mirrors the f64 readout closure in `build_y_output_fn`.
pub struct OdeOutputProgram {
    bc: Bytecode,
    n_states: usize,
    n_indiv: usize,
    /// Per individual parameter `i`, its slot in the flat PK-parameter vector.
    indiv_to_pk: Vec<usize>,
    /// Covariate names the bytecode's `PushCov(i)` reads, in index order. The
    /// dual readout resolves these from the per-observation covariate snapshot
    /// and feeds them as constants — a covariate carries no parameter derivative
    /// in the individual-parameter dual basis the ODE provider seeds (#540).
    cov_names: Vec<String>,
    /// True when the expression references only states / individual parameters /
    /// covariates / constants — the case the dual readout can evaluate, threading
    /// the covariate snapshot in while leaving θ/η empty. A θ/η referenced
    /// *directly* in the readout is *normally* desugared at parse time into a
    /// synthetic individual parameter (`__ferx_ro_*`, #486), so it reaches the
    /// readout as an ordinary `PushVar` and rides the individual-parameter
    /// sensitivity chain. The exception is the slot-overflow FD fallback: when the
    /// synthetic params would exceed the PK-slot layout the parser leaves the bare
    /// θ/η ops in place, and the `PushTheta`/`PushEta` check at the construction
    /// site keeps this `false` so that readout stays on FD. An NN output
    /// (`PushNnOutput`) is the other, always-FD disqualifier.
    dual_evaluable: bool,
}

impl OdeOutputProgram {
    /// See [`OdeOutputProgram::dual_evaluable`].
    pub(crate) fn is_dual_evaluable(&self) -> bool {
        self.dual_evaluable
    }

    /// Whether the readout reads the model-time built-in (`Op::PushTime`, from a
    /// `TIME` / `T` / `t` reference). Such a readout is **not** a pure function of
    /// the compartment state: its value at a fixed state depends on which
    /// observation is being read.
    ///
    /// That breaks any consumer that characterises the readout by probing it at a
    /// synthetic state — the modified-release closed-form fast path
    /// (`pk::modified_release::recover_disp_params_g`) probes `readout(0)`,
    /// `readout(e_c)`, `readout(2·e_c)` to establish linearity in the central
    /// amount, and an additive `TIME` term evaluates to its *ambient* thread-local
    /// value there (`0.0` at the gate), so the probe would certify a linear readout
    /// and the fast path would then silently drop the whole time term. Callers that
    /// probe must decline on `true` (#1028).
    pub(crate) fn reads_time_builtin(&self) -> bool {
        self.bc.ops.iter().any(|op| matches!(op, Op::PushTime))
    }

    /// Number of compartment-state inputs in the readout's `vars[0..n_states]`
    /// layout. For an analytic readout this is `1` (IV: `["central"]`) or `2`
    /// (oral: `["depot", "central"]`).
    pub(crate) fn n_states(&self) -> usize {
        self.n_states
    }

    /// Whether the compiled readout actually references state slot `idx` (a
    /// `PushVar(idx)` with `idx < n_states`). The analytic Dual2 provider (#650)
    /// uses this to gate a readout that reads the oral `depot` amount — which the
    /// static superposition jet doesn't reconstruct — onto the FD path.
    pub(crate) fn references_state(&self, idx: usize) -> bool {
        idx < self.n_states
            && self
                .bc
                .ops
                .iter()
                .any(|op| matches!(op, Op::PushVar(i) if *i as usize == idx))
    }

    /// Evaluate the output expression over a dual type, generic over [`PkNum`]
    /// (`Dual1` light inner / `Dual2` full outer; #410) — the Form-C readout.
    /// `state` is the integrated dual state; `params` is the flat PK-parameter
    /// vector with the differentiated slots seeded as dual variables (so `V1`'s
    /// derivative flows into `central / V1`); `cov` is the per-observation
    /// covariate snapshot, whose values thread in as constants (#540). `vars`/
    /// `stack` are caller-owned scratch. Only valid when
    /// [`is_dual_evaluable`](Self::is_dual_evaluable) holds.
    pub(crate) fn eval_output_g<T: crate::sens::num::PkNum>(
        &self,
        state: &[T],
        params: &[T],
        cov: &HashMap<String, f64>,
        vars: &mut Vec<T>,
        stack: &mut Vec<T>,
    ) -> T {
        vars.clear();
        vars.resize(self.n_states + self.n_indiv, T::from_f64(0.0));
        let copy_n = self.n_states.min(state.len());
        vars[..copy_n].copy_from_slice(&state[..copy_n]);
        for (i, &slot) in self.indiv_to_pk.iter().enumerate() {
            if let (Some(dst), Some(&v)) = (vars.get_mut(self.n_states + i), params.get(slot)) {
                *dst = v;
            }
        }
        // Covariate values in `PushCov` index order (== `cov_names` order); a
        // missing covariate reads 0.0, mirroring the f64 readout `out_fn`.
        let cov_vec: Vec<f64> = self
            .cov_names
            .iter()
            .map(|n| cov.get(n).copied().unwrap_or(0.0))
            .collect();
        let empty_nn: Vec<Vec<f64>> = Vec::new();
        eval_bytecode_g::<T>(&self.bc, &[], &[], &cov_vec, vars, &empty_nn, stack)
    }
}

/// Differentiable form of an `obs_scale = <expr>` scaling expression (issue #367).
/// Holds the scale expression compiled to bytecode plus the layout to evaluate it
/// over `Dual2<M>` seeded on (θ, η): θ/η references read the seed duals directly;
/// individual-parameter references read `vars[i]`, where var slot `i` is fed by
/// the provider as a dual for flat PK slot `var_to_pk_slot[i]` (value + ∂p/∂(θ,η));
/// covariates are constants. This lets the analytic sensitivity provider
/// differentiate the scaled prediction `f / scale` exactly instead of falling back
/// to finite differences for `ExpressionScale` models. Mirrors [`OdeOutputProgram`].
pub struct ScaleDerivProgram {
    bc: Bytecode,
    n_theta: usize,
    n_eta: usize,
    /// `vars` slot `i` (the bytecode's `PushVar(i)`) → flat PK slot whose value
    /// and `∂/∂(θ,η)` the provider feeds in. Parallel to the individual-parameter
    /// declaration order.
    var_to_pk_slot: Vec<usize>,
    cov_names: Vec<String>,
}

impl ScaleDerivProgram {
    /// Dual width needed to seed every (θ, η) axis.
    pub(crate) fn n_axes(&self) -> usize {
        self.n_theta + self.n_eta
    }
    pub(crate) fn n_theta_axis(&self) -> usize {
        self.n_theta
    }
    pub(crate) fn n_eta_axis(&self) -> usize {
        self.n_eta
    }
    /// `vars` slot → flat PK slot (so the provider knows which PK params to seed).
    pub(crate) fn var_to_pk_slot(&self) -> &[usize] {
        &self.var_to_pk_slot
    }

    /// Evaluate the scale over `Dual2<M>` seeded on (θ, η): `θ_m → var(·, m)`,
    /// `η_k → var(·, n_theta + k)`. `var_duals[i]` is the dual for the individual
    /// parameter at PK slot `var_to_pk_slot[i]` (value + `∂/∂(θ,η)`). Returns the
    /// scale's value, gradient `∂scale/∂(θ,η)`, and Hessian. Requires `M ≥ n_axes()`.
    pub(crate) fn eval_scale_dual<const M: usize>(
        &self,
        theta: &[f64],
        eta: &[f64],
        cov: &HashMap<String, f64>,
        var_duals: &[crate::sens::dual2::Dual2<M>],
    ) -> crate::sens::dual2::Dual2<M> {
        use crate::sens::dual2::Dual2;
        let theta_d: Vec<Dual2<M>> = theta
            .iter()
            .enumerate()
            .map(|(m, &v)| {
                if m < M {
                    Dual2::var(v, m)
                } else {
                    Dual2::constant(v)
                }
            })
            .collect();
        let eta_d: Vec<Dual2<M>> = eta
            .iter()
            .enumerate()
            .map(|(k, &v)| {
                let dim = self.n_theta + k;
                if dim < M {
                    Dual2::var(v, dim)
                } else {
                    Dual2::constant(v)
                }
            })
            .collect();
        let cov_vec: Vec<f64> = self
            .cov_names
            .iter()
            .map(|n| cov.get(n).copied().unwrap_or(0.0))
            .collect();
        let empty_nn: Vec<Vec<f64>> = Vec::new();
        let mut stack: Vec<Dual2<M>> = Vec::new();
        eval_bytecode_g::<Dual2<M>>(
            &self.bc, &theta_d, &eta_d, &cov_vec, var_duals, &empty_nn, &mut stack,
        )
    }

    /// First-order η-only evaluation: returns `value` and `∂/∂η_k` on axis `k` over
    /// `Dual1<N>` (`N = n_eta`). θ and individual-param vars enter as constants
    /// except the seeded η axes, so this drops the θ axes and the `Dual2` Hessian
    /// that [`eval_scale_dual`](Self::eval_scale_dual) carries. Used by the inner
    /// init-amount gradient, whose EBE loop consumes only `∂A₀/∂η` (#524 follow-up).
    pub(crate) fn eval_scale_dual1<const N: usize>(
        &self,
        theta: &[f64],
        eta: &[f64],
        cov: &HashMap<String, f64>,
        var_duals: &[crate::sens::dual1::Dual1<N>],
    ) -> crate::sens::dual1::Dual1<N> {
        use crate::sens::dual1::Dual1;
        // θ and individual-param vars are constants on the inner path (no θ axes);
        // η_k seeds axis k directly (no `n_theta` offset). For any in-scope
        // `ExpressionScale` model the provider gate caps `n_theta + n_eta` at
        // `MAX_SCALE_AXES`, so both `theta` and `eta` fit a stack buffer — build them
        // there instead of per-call heap `Vec`s (#534 review #4). (`cov_vec` /
        // bytecode `stack` lengths are not axis-bounded, so they stay on the heap.)
        const CAP: usize = crate::sens::provider::MAX_SCALE_AXES;
        debug_assert!(
            theta.len() <= CAP && eta.len() <= CAP,
            "scale eval θ/η exceed MAX_SCALE_AXES; caller must be gate-checked"
        );
        let mut theta_buf = [Dual1::<N>::constant(0.0); CAP];
        for (d, &v) in theta_buf.iter_mut().zip(theta.iter()) {
            *d = Dual1::constant(v);
        }
        let theta_d = &theta_buf[..theta.len()];
        let mut eta_buf = [Dual1::<N>::constant(0.0); CAP];
        for (k, (d, &v)) in eta_buf.iter_mut().zip(eta.iter()).enumerate() {
            *d = if k < N {
                Dual1::var(v, k)
            } else {
                Dual1::constant(v)
            };
        }
        let eta_d = &eta_buf[..eta.len()];
        let cov_vec: Vec<f64> = self
            .cov_names
            .iter()
            .map(|n| cov.get(n).copied().unwrap_or(0.0))
            .collect();
        let empty_nn: Vec<Vec<f64>> = Vec::new();
        let mut stack: Vec<Dual1<N>> = Vec::new();
        eval_bytecode_g::<Dual1<N>>(
            &self.bc, theta_d, eta_d, &cov_vec, var_duals, &empty_nn, &mut stack,
        )
    }
}

/// Differentiable form of a custom residual-error magnitude slot (#484/#576/#486):
/// the magnitude expression compiled to bytecode, evaluable over `Dual1<M>` seeded
/// on `θ` only. Unlike [`ScaleDerivProgram`] the magnitude never depends on `η` or
/// an individual parameter (`validate_ruv_expr` rejects both), so there is a single
/// var slot — the scaled sigma, pinned to the constant `1.0` — and no
/// `var_to_pk_slot`. This lets the analytic outer θ/σ gradient differentiate
/// `mult(θ)` exactly (a new direct-θ term in `∂R/∂θ`) instead of falling back to
/// FD for a custom / time-varying σ magnitude.
pub struct RuvMagDerivProgram {
    bc: Bytecode,
    n_theta: usize,
    cov_names: Vec<String>,
}

/// Axis cap for the `Dual1<M>` dispatch table in [`RuvMagDerivProgram::theta_grad`]
/// (same const-generic-dispatch convention as `MAX_SCALE_AXES` / `MAX_ODE_AXES`,
/// but set higher — the magnitude program's bytecode evaluator is cheap to
/// monomorphize, and covering realistic large-θ population models keeps the
/// magnitude direct-θ channel analytic rather than dropping to FD, #486). A model
/// with more thetas than this falls back to FD for the magnitude's direct-θ
/// gradient channel — `analytic_outer_gradient_available` bounds `model.n_theta`
/// against this constant when a custom magnitude is active.
pub(crate) const MAX_RUV_MAG_AXES: usize = 32;

impl RuvMagDerivProgram {
    /// `∂(magnitude)/∂θ` at `theta` (with the observation's covariates and raw
    /// TIME), dispatching the const-generic `Dual1<M>` eval on `theta.len()`.
    /// `None` when `theta.len()` exceeds [`MAX_RUV_MAG_AXES`] or does not match
    /// the program's own `n_theta` (caller falls back to FD).
    pub(crate) fn theta_grad(
        &self,
        theta: &[f64],
        cov: &HashMap<String, f64>,
        time: f64,
    ) -> Option<Vec<f64>> {
        use crate::sens::dual1::Dual1;
        if theta.len() != self.n_theta {
            return None;
        }
        if theta.is_empty() {
            return Some(Vec::new());
        }
        // Mirror the runtime `RuvMagFn` closure's `TIME`/`time` covariate-map
        // injection: a legacy pre-built expression may still carry `TIME` as an
        // `Expression::Covariate` node (parsed models use the `Expression::Time`
        // built-in, read directly from `time` via `Op::PushTime`, and never reach
        // this branch) — without it, such a `cov_name` would look up the
        // observation's covariate map (which never carries a literal "TIME" key)
        // and silently default to `0.0` instead of the actual observation time.
        let cov_vec: Vec<f64> = self
            .cov_names
            .iter()
            .map(|n| {
                if n.eq_ignore_ascii_case("TIME") {
                    time
                } else {
                    cov.get(n).copied().unwrap_or(0.0)
                }
            })
            .collect();
        // `theta_d`/`prog_vars` are exactly `$n`-sized at each dispatch arm (unlike
        // `ScaleDerivProgram::eval_scale_dual1`'s padded-to-N buffers), so a plain
        // stack array avoids the sibling's per-call heap `Vec` for the axis-bounded
        // pieces (#486 review); `cov_vec`/`stack` stay on the heap, matching that
        // sibling's own precedent — they are not axis-bounded.
        macro_rules! run {
            ($n:literal) => {{
                let mut theta_d = [Dual1::<$n>::constant(0.0); $n];
                for (m, d) in theta_d.iter_mut().enumerate() {
                    *d = Dual1::var(theta[m], m);
                }
                // Slot 0 = the pinned scaled sigma (1.0); slot 1 = MACHEPS.
                let prog_vars = [
                    Dual1::<$n>::constant(1.0),
                    Dual1::<$n>::constant(f64::EPSILON),
                ];
                let empty_nn: Vec<Vec<f64>> = Vec::new();
                let mut stack: Vec<Dual1<$n>> = Vec::new();
                let d = with_model_time(time, || {
                    eval_bytecode_g::<Dual1<$n>>(
                        &self.bc,
                        &theta_d,
                        &[],
                        &cov_vec,
                        &prog_vars,
                        &empty_nn,
                        &mut stack,
                    )
                });
                d.grad.to_vec()
            }};
        }
        Some(match theta.len() {
            1 => run!(1),
            2 => run!(2),
            3 => run!(3),
            4 => run!(4),
            5 => run!(5),
            6 => run!(6),
            7 => run!(7),
            8 => run!(8),
            9 => run!(9),
            10 => run!(10),
            11 => run!(11),
            12 => run!(12),
            13 => run!(13),
            14 => run!(14),
            15 => run!(15),
            16 => run!(16),
            17 => run!(17),
            18 => run!(18),
            19 => run!(19),
            20 => run!(20),
            21 => run!(21),
            22 => run!(22),
            23 => run!(23),
            24 => run!(24),
            25 => run!(25),
            26 => run!(26),
            27 => run!(27),
            28 => run!(28),
            29 => run!(29),
            30 => run!(30),
            31 => run!(31),
            32 => run!(32),
            _ => return None,
        })
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
        | Expression::Time
        | Expression::MixNum
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
        Expression::Literal(_) | Expression::Time | Expression::MixNum => Expression::Literal(0.0),
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
        | Expression::Time
        | Expression::MixNum
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
    /// The compiled individual-parameter program, so the analytical PK
    /// sensitivity provider can obtain exact `∂p/∂(θ,η)` by evaluating it over
    /// `Dual2` seeded on (θ, η) — replacing the finite-difference `tv_theta_jacobian`
    /// θ chain and the log-normal-only `sel` η chain (issue #367). Unlike the
    /// `d_d_theta`/`d_d_eta` symbolic partials (reserved, no runtime consumer),
    /// this field IS consumed at runtime by `sens::provider`. `None` for
    /// hand-built fixtures and when no `[individual_parameters]` block exists;
    /// the ODE provider reads its own copy from `ode_spec`.
    pub(crate) indiv_param_program: Option<IndivParamProgram>,
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
            indiv_param_program: None,
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
        // Attached by the caller (`build_pk_param_fn` site) after the program is
        // compiled; the symbolic-partials builder itself doesn't produce it.
        indiv_param_program: None,
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
            if name == "TIME" || name == "time" {
                return Ok((Expression::Time, pos + 1));
            }

            // MIXNUM — reserved read-only subpopulation index (1..=K) for
            // `$MIXTURE` models (#977). Case-insensitive, like the `MACHEPS`
            // built-in. Resolves to `Expression::MixNum`; a non-mixture model
            // that references it is rejected later by `validate_mixture_spec`.
            if name.eq_ignore_ascii_case("MIXNUM") {
                return Ok((Expression::MixNum, pos + 1));
            }

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

            // Check if it's a function call: `name(expr)` — or, for `min` / `max`,
            // the two-argument `name(a, b)` form (#1030).
            if pos + 1 < tokens.len() && tokens[pos + 1] == Token::LParen {
                let func_name = name.to_lowercase();
                let is_min_max = matches!(func_name.as_str(), "min" | "max");
                let (arg, p) = parse_add_sub(tokens, pos + 2, ctx)?;
                if is_min_max && tokens.get(p) == Some(&Token::Comma) {
                    let (arg2, p) = parse_add_sub(tokens, p + 1, ctx)?;
                    if tokens.get(p) != Some(&Token::RParen) {
                        return Err(format!(
                            "Missing closing parenthesis for function {} — `{}` takes exactly \
                             two arguments, `{}(a, b)`.",
                            name, name, func_name
                        ));
                    }
                    // Desugar to the inline conditional rather than adding a
                    // two-argument AST node: `Expression::Conditional` is already
                    // evaluated, bytecode-compiled, `Dual2`-differentiated and
                    // index-resolved everywhere in the pipeline, so `min`/`max` gets
                    // all of that for free and cannot drift from the hand-written
                    // `if (a > b) a else b` it replaces. The guarded expression is
                    // duplicated in the tree, exactly as it is when written by hand;
                    // name it with an intermediate to keep the source readable.
                    let op = if func_name == "min" {
                        CmpOp::Le
                    } else {
                        CmpOp::Ge
                    };
                    let cond = Condition::Compare(arg.clone(), op, arg2.clone());
                    return Ok((
                        Expression::Conditional(Box::new(cond), Box::new(arg), Box::new(arg2)),
                        p + 1,
                    ));
                }
                if tokens.get(p) != Some(&Token::RParen) {
                    // Say what's actually wrong. A stray `,` used to be reported as a
                    // missing `)`, which sends the reader looking for a bracket bug
                    // that doesn't exist (#1030).
                    if tokens.get(p) == Some(&Token::Comma) {
                        return Err(format!(
                            "function `{}` takes 1 argument, but a `,` was found — only \
                             `min(a, b)` and `max(a, b)` take two.",
                            name
                        ));
                    }
                    return Err(format!("Missing closing parenthesis for function {}", name));
                }
                if is_min_max {
                    return Err(format!(
                        "`{}` takes exactly two arguments: `{}(a, b)`.",
                        name, func_name
                    ));
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
#[path = "model_parser_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "model_parser_adaptive_dosing_tests.rs"]
mod adaptive_dosing_tests;
