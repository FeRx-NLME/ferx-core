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

/// Walk a Mul-chain and find the first `exp(Eta(j))`, returning the eta index.
fn find_exp_eta_in_mul(expr: &Expression) -> Option<usize> {
    match expr {
        Expression::UnaryFn(name, arg) if name == "exp" => {
            if let Expression::Eta(j) = arg.as_ref() {
                return Some(*j);
            }
            None
        }
        Expression::BinOp(l, BinOp::Mul, r) => {
            find_exp_eta_in_mul(l).or_else(|| find_exp_eta_in_mul(r))
        }
        _ => None,
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
                Statement::AssignIdx(_, _) => {
                    // Should never appear in the AST walked by this
                    // helper — `resolve_variable_indices` is run only
                    // after all name-collecting helpers have finished.
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
                Statement::AssignIdx(_, _) => {
                    // Indexed-form statements only appear in the
                    // post-resolve AST owned by `pk_param_fn`; this
                    // helper is called on the pre-resolve AST.
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
            return Some(ExprClass {
                eta_idx: ei,
                theta_idx: None,
                param_type: pt,
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
    let extracted = extract_blocks(content)?;
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

    let struct_lines = blocks
        .get("structural_model")
        .ok_or("Missing [structural_model] block")?;

    let error_lines = blocks
        .get("error_model")
        .ok_or("Missing [error_model] block")?;
    let (error_model, _) = parse_error_model(error_lines)?;

    let indiv_lines = blocks
        .get("individual_parameters")
        .ok_or("Missing [individual_parameters] block")?;

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
    // `indiv_var_names` contains only TOP-LEVEL assignments — these are the
    // individual parameters that map to PK slots and the TV output vector.
    // Branch-local helpers (assigned only inside if-bodies) are intentionally
    // excluded to prevent them from corrupting the AD inner-loop slot layout.
    // The ParseCtx still receives the full set (via assigned_vars_in_order) so
    // branch-local names parse as Variable rather than Covariate.
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
    let indiv_var_names = top_level_assigned_vars(&pre_stmts);
    let indiv_ctx =
        ParseCtx::new(&theta_names, &eta_names, &all_assigned).with_nn_specs(&nn_specs_for_ctx);
    let indiv_stmts = parse_block_statements(&indiv_text, indiv_ctx, StatementMode::Plain)?;

    // Detect ODE vs analytical model
    let is_ode = struct_lines
        .iter()
        .any(|l| l.starts_with("ode(") || l.starts_with("ode "));

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
        let (pk_model, pk_param_map) = parse_structural_model(struct_lines)?;
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
    let (pk_param_fn, referenced_covariates) = build_pk_param_fn(
        indiv_stmts.clone(),
        &pk_param_map,
        &indiv_var_names,
        #[cfg(feature = "nn")]
        &covariate_nns_for_closure,
    )?;

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
    // BSV omega is built from the BSV-only eta names (no kappas)
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
        // ODE model: sequential indices
        (0..n_eta).collect()
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
        pk_param_fn,
        n_theta,
        n_eta,
        n_kappa,
        n_epsilon,
        theta_names,
        eta_names: eta_names_bsv,
        kappa_names,
        indiv_param_names: indiv_var_names.clone(),
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
        eta_param_info,
        theta_transform,
        scaling: ScalingSpec::None,
    };

    // ── Optional blocks ──
    let simulation = blocks
        .get("simulation")
        .map(|lines| parse_simulation_block(lines))
        .transpose()?;
    let fit_options = if let Some(lines) = blocks.get("fit_options") {
        parse_fit_options(lines)?
    } else {
        FitOptions::default()
    };

    // Mirror fit-level BLOQ method onto the compiled model so the likelihood
    // functions can branch without threading bloq_method through every call.
    let mut model = model;
    model.bloq_method = fit_options.bloq_method;

    // ── [scaling] block ──
    // Parsed after `parse_fit_options` so we can validate the
    // `ExpressionScale + gradient = ad` combination (which is rejected for
    // Phase 1 — AD's `obs_scale` Const input is only wired to ScalarScale).
    if let Some(scaling_lines) = blocks.get("scaling") {
        let theta_names_for_scaling = model.theta_names.clone();
        let eta_names_for_scaling = model.eta_names.clone();
        let indiv_var_names_for_scaling = model.indiv_param_names.clone();
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
            &state_names_for_scaling,
            is_ode_model,
        )?;

        // Form B + AD is rejected: the analytical AD path threads a single
        // f64 scale into `individual_nll_ad` as a `Const`, which can't
        // evaluate an expression that touches theta/eta. (gradient_method
        // resolution happens at fit time when `Auto` defaults to AD if
        // `tv_fn.is_some()`, so we conservatively reject both `Ad` and
        // `Auto` here when ExpressionScale is in play.)
        let ad_explicit = fit_options.gradient_method == GradientMethod::Ad;
        let ad_auto_likely = fit_options.gradient_method == GradientMethod::Auto
            && model.tv_fn.is_some()
            && cfg!(feature = "autodiff");
        if matches!(scaling, ScalingSpec::ExpressionScale { .. }) && (ad_explicit || ad_auto_likely)
        {
            return Err(
                "[scaling]: expression-form `obs_scale = <expr>` is not yet supported with AD \
                 gradients. Add `gradient = fd` to [fit_options], or rewrite as a scalar \
                 `obs_scale = <number>`."
                    .into(),
            );
        }

        // Form C wiring: store the output_fn on the ODE spec and clear
        // `obs_cmt_idx` so the readout dispatches through `output_fn`.
        if let Some(of) = output_fn {
            // `is_ode_model` guarantees `model.ode_spec.is_some()` here,
            // and parse_scaling_block already rejects Form C for analytical models.
            let ode_spec = model.ode_spec.as_mut().expect("guarded by is_ode_model");
            ode_spec.output_fn = Some(of);
            ode_spec.obs_cmt_idx = None;
        }

        model.scaling = scaling;
    }

    // ODE validation: exactly one of obs_cmt / output_fn must be set on
    // any compiled ODE model. `parse_ode_structural` accepts both forms;
    // the only invalid path is omitting obs_cmt without supplying
    // [scaling] y = <expr>.
    if let Some(ref ode_spec) = model.ode_spec {
        if ode_spec.obs_cmt_idx.is_none() && ode_spec.output_fn.is_none() {
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
            if ode_spec.obs_cmt_idx.is_none() {
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
        // Collect variables that appear inside if-branches but NOT at top level.
        let conditional_only: Vec<&String> = all_assigned
            .iter()
            .filter(|n| !indiv_var_names.contains(n))
            .collect();
        // Also collect top-level vars that are assigned ONLY inside if-blocks
        // (i.e. they appear in an If branch but have no top-level Assign).
        // Union: any var in an if-branch that references an eta → warn.
        fn any_if_branch_assigns_eta(stmts: &[Statement], var: &str, n_eta: usize) -> bool {
            for s in stmts {
                if let Statement::If {
                    branches,
                    else_body,
                } = s
                {
                    for (_, body) in branches {
                        for bs in body {
                            if let Statement::Assign(name, expr) = bs {
                                if name == var
                                    && extract_eta_indices(expr).iter().any(|&i| i < n_eta)
                                {
                                    return true;
                                }
                            }
                        }
                    }
                    if let Some(eb) = else_body {
                        for bs in eb {
                            if let Statement::Assign(name, expr) = bs {
                                if name == var
                                    && extract_eta_indices(expr).iter().any(|&i| i < n_eta)
                                {
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
            false
        }
        let mut mu_ref_disabled: Vec<String> = Vec::new();
        for var in &conditional_only {
            if any_if_branch_assigns_eta(&indiv_stmts, var, n_eta) {
                mu_ref_disabled.push((*var).clone());
            }
        }
        // Also catch top-level vars whose only eta-bearing assignment is
        // inside a nested if — these are in indiv_var_names but not mu_refs.
        for var in &indiv_var_names {
            if !model.mu_refs.contains_key(var)
                && any_if_branch_assigns_eta(&indiv_stmts, var, n_eta)
            {
                if !mu_ref_disabled.contains(var) {
                    mu_ref_disabled.push(var.clone());
                }
            }
        }
        if !mu_ref_disabled.is_empty() {
            model.parse_warnings.push(format!(
                "Mu-referencing disabled for conditional parameter(s): {}. \
                 Assign TV* unconditionally and apply the if-block to the individual \
                 parameter expression to re-enable mu-referencing.",
                mu_ref_disabled.join(", ")
            ));
        }
    }

    Ok(ParsedModel {
        model,
        simulation,
        fit_options,
    })
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
        "covariance" => opts.run_covariance_step = parse_bool("covariance")?,
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
        "adapt_interval" => opts.saem_adapt_interval = parse_usize("adapt_interval")?,
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
        "scale_params" => opts.scale_params = parse_bool("scale_params")?,
        "max_unconverged_frac" => opts.max_unconverged_frac = parse_f64("max_unconverged_frac")?,
        "min_obs_for_convergence_check" => {
            opts.min_obs_for_convergence_check =
                parse_usize("min_obs_for_convergence_check")? as u32
        }
        "stagnation_guard" => opts.stagnation_guard = parse_bool("stagnation_guard")?,
        _ => return Ok(false),
    }
    opts.user_set_keys.push(key.to_string());
    Ok(true)
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

/// Walk an expression tree and reject any `Variable` reference. Used to
/// enforce the Phase-1 restriction that `obs_scale = <expr>` (Form B) cannot
/// reference individual parameters — only theta/eta/covariates.
fn reject_variable_refs(expr: &Expression, where_msg: &str) -> Result<(), String> {
    match expr {
        Expression::Variable(name) => Err(format!(
            "[scaling]: `{}` references individual parameter `{}`. \
             Form B (obs_scale = <expr>) supports thetas, etas, and covariates only \
             in Phase 1; refer to the underlying theta (e.g. `TVV`) or use Form C \
             (`y = <expr>` on an ODE model) for individual-parameter access.",
            where_msg, name
        )),
        Expression::BinOp(l, _, r) => {
            reject_variable_refs(l, where_msg)?;
            reject_variable_refs(r, where_msg)
        }
        Expression::UnaryFn(_, a) => reject_variable_refs(a, where_msg),
        Expression::Power(b, e) => {
            reject_variable_refs(b, where_msg)?;
            reject_variable_refs(e, where_msg)
        }
        Expression::Conditional(_, t, e) => {
            reject_variable_refs(t, where_msg)?;
            reject_variable_refs(e, where_msg)
        }
        _ => Ok(()),
    }
}

/// Parsed contents of a `[scaling]` block.
///
/// Returns `(ScalingSpec, Option<OdeOutputFn>)`:
/// - `ScalingSpec` populates `CompiledModel.scaling` (Forms A/B post-multiply).
/// - `OdeOutputFn` populates `OdeSpec.output_fn` for Form C (`y = <expr>`),
///   replacing the default state-index readout inside the ODE timeline loop.
///
/// `is_ode = true` enables Form C and lets expressions reference state names.
/// Validation:
/// - Unknown keys are rejected.
/// - `y = ...` on an analytical model is rejected.
/// - Form B (expression `obs_scale`) cannot reference individual parameters
///   in Phase 1 — see `reject_variable_refs`.
#[allow(clippy::too_many_arguments)]
fn parse_scaling_block(
    lines: &[String],
    theta_names: &[String],
    eta_names: &[String],
    indiv_var_names: &[String],
    state_names: &[String],
    is_ode: bool,
) -> Result<(ScalingSpec, Option<crate::ode::OdeOutputFn>), String> {
    let mut scaling = ScalingSpec::None;
    let mut output_fn: Option<crate::ode::OdeOutputFn> = None;

    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = trimmed.splitn(2, '=').map(|s| s.trim()).collect();
        if parts.len() != 2 {
            return Err(format!(
                "[scaling]: expected `key = value`, got: `{}`",
                trimmed
            ));
        }
        let (key, value) = (parts[0], parts[1]);

        match key {
            "obs_scale" => {
                if !matches!(scaling, ScalingSpec::None) {
                    return Err("[scaling]: duplicate `obs_scale` key".into());
                }
                // Try scalar first (Form A). Otherwise parse as expression (Form B).
                if let Ok(k) = value.parse::<f64>() {
                    // Divisor — strictly positive. A negative scale would flip
                    // every prediction sign and bypass the upstream
                    // non-negativity clamps in the analytical / ODE paths.
                    if !(k > 0.0 && k.is_finite()) {
                        return Err(format!(
                            "[scaling]: obs_scale must be a strictly positive finite value, got `{}`",
                            value
                        ));
                    }
                    scaling = ScalingSpec::ScalarScale(k);
                } else {
                    let ctx = ParseCtx::new(theta_names, eta_names, indiv_var_names);
                    let expr = parse_scalar_expression(value, ctx)
                        .map_err(|e| format!("[scaling] obs_scale: {}", e))?;
                    // Phase 1 restriction: no individual-parameter refs in
                    // Form B — the closure has no path to evaluate them
                    // without re-running the pk_param_fn (deferred to phase 2).
                    reject_variable_refs(&expr, "obs_scale")?;
                    let scale_fn: ScaleFn = Box::new(
                        move |theta: &[f64],
                              eta: &[f64],
                              covariates: &HashMap<String, f64>|
                              -> f64 {
                            let empty_vars: HashMap<String, f64> = HashMap::new();
                            let empty_nn: Vec<Vec<f64>> = Vec::new();
                            eval_expression(&expr, theta, eta, covariates, &empty_vars, &empty_nn)
                        },
                    );
                    scaling = ScalingSpec::ExpressionScale { scale_fn };
                }
            }
            "y" => {
                if !is_ode {
                    return Err("[scaling]: `y = <expr>` (Form C) requires an ODE model — \
                         use `obs_scale = <expr>` for analytical PK"
                        .into());
                }
                if output_fn.is_some() {
                    return Err("[scaling]: duplicate `y` key".into());
                }
                // Form C: expression references state names + individual parameter
                // names. Thetas/etas are NOT in scope (they're folded into pk via
                // pk_param_fn before output_fn is called).
                let mut defined: Vec<String> = state_names.to_vec();
                for n in indiv_var_names {
                    if !defined.iter().any(|d| d == n) {
                        defined.push(n.clone());
                    }
                }
                // Use `ParseCtx::new` (fallback_covariate = true) rather than
                // `ParseCtx::ode` so unknown identifiers like `WT` resolve as
                // `Covariate(WT)` and are looked up in the covariate map at
                // eval time. With `ParseCtx::ode` they would silently parse
                // as `Variable(WT)` and read 0.0 from `vars` since the output_fn
                // vars map only carries states + indiv params.
                const EMPTY: &[String] = &[];
                let ctx = ParseCtx::new(EMPTY, EMPTY, &defined);
                let expr = parse_scalar_expression(value, ctx)
                    .map_err(|e| format!("[scaling] y: {}", e))?;
                // Pre-resolve state names → state indices and individual-param
                // names → positional pk_params_flat slots (ODE writes indiv
                // params sequentially starting at slot 0; see build_pk_param_fn).
                let state_idx: HashMap<String, usize> = state_names
                    .iter()
                    .enumerate()
                    .map(|(i, n)| (n.clone(), i))
                    .collect();
                let indiv_idx: HashMap<String, usize> = indiv_var_names
                    .iter()
                    .enumerate()
                    .map(|(i, n)| (n.clone(), i))
                    .collect();
                let out_fn: crate::ode::OdeOutputFn = Box::new(
                    move |state: &[f64],
                          pk_params_flat: &[f64],
                          covariates: &HashMap<String, f64>|
                          -> f64 {
                        let mut vars: HashMap<String, f64> =
                            HashMap::with_capacity(state_idx.len() + indiv_idx.len());
                        for (name, &i) in &state_idx {
                            if i < state.len() {
                                vars.insert(name.clone(), state[i]);
                            }
                        }
                        for (name, &i) in &indiv_idx {
                            if i < pk_params_flat.len() {
                                vars.insert(name.clone(), pk_params_flat[i]);
                            }
                        }
                        let empty_nn: Vec<Vec<f64>> = Vec::new();
                        let empty_theta: [f64; 0] = [];
                        let empty_eta: [f64; 0] = [];
                        eval_expression(
                            &expr,
                            &empty_theta,
                            &empty_eta,
                            covariates,
                            &vars,
                            &empty_nn,
                        )
                    },
                );
                output_fn = Some(out_fn);
            }
            _ => {
                return Err(format!("[scaling]: unknown key `{}`", key));
            }
        }
    }

    Ok((scaling, output_fn))
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

fn build_ode_spec(
    lines: &[String],
    state_names: &[String],
    obs_cmt_name: Option<&str>,
    indiv_param_names: &[String],
) -> Result<crate::ode::OdeSpec, String> {
    let n_states = state_names.len();
    // `obs_cmt_idx = None` when the user omitted `obs_cmt` in the ODE form
    // and supplied a `[scaling] y = <expr>` (Form C) instead — wired in
    // `parse_full_model` after the scaling block parses.
    let obs_cmt_idx = match obs_cmt_name {
        Some(name) => Some(state_names.iter().position(|s| s == name).ok_or_else(|| {
            format!(
                "Observable compartment '{}' not in states {:?}",
                name, state_names
            )
        })?),
        None => None,
    };

    // For ODE RHS expressions, states + individual params get injected into the
    // `vars` map at eval time, so every bare identifier should resolve to a
    // Variable (not a Covariate). ParseCtx::ode() flips the fallback accordingly.
    // Local intermediate vars assigned within the [odes] block (e.g. inside an
    // if-body) are also collected from a pre-pass below so they parse as
    // Variable too.
    let block_text = lines.join("\n");
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
                Statement::Assign(_, _) | Statement::AssignIdx(_, _) => {}
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
                Statement::Assign(_, _) | Statement::AssignIdx(_, _) => {}
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
    let stmts_owned = stmts;
    let state_index: HashMap<String, usize> = state_names_owned
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();

    let rhs: Box<dyn Fn(&[f64], &[f64], f64, &mut [f64]) + Send + Sync> =
        Box::new(move |u: &[f64], params: &[f64], _t: f64, du: &mut [f64]| {
            let mut vars: HashMap<String, f64> = HashMap::new();

            // Inject state variables: state_name → u[i]
            for (i, name) in state_names_owned.iter().enumerate() {
                vars.insert(name.clone(), u[i]);
                vars.insert(name.to_lowercase(), u[i]);
            }

            // Inject individual parameters by name → params[i]
            // params = PkParams.values, where pk_param_fn stores individual params
            // by position matching the order in [individual_parameters] block
            for (i, name) in indiv_names_owned.iter().enumerate() {
                if i < params.len() {
                    vars.insert(name.clone(), params[i]);
                    vars.insert(name.to_uppercase(), params[i]);
                    vars.insert(name.to_lowercase(), params[i]);
                }
            }

            // Reset du so that a state without a firing d/dt this iteration
            // (e.g. inside an untaken if-branch) gets 0.0 rather than stale
            // memory.
            for slot in du.iter_mut() {
                *slot = 0.0;
            }

            let empty_theta: [f64; 0] = [];
            let empty_eta: [f64; 0] = [];
            let empty_cov: HashMap<String, f64> = HashMap::new();
            // ODE RHS evaluation runs with empty theta/eta/covariates here —
            // values are injected later via the `vars` map (state names, indiv
            // params). NN outputs aren't relevant: `[covariate_nn]` outputs are
            // routed via `pk_param_fn`, not the ODE RHS.
            let empty_nn_outputs: Vec<Vec<f64>> = Vec::new();
            eval_statements(
                &stmts_owned,
                &empty_theta,
                &empty_eta,
                &empty_cov,
                &mut vars,
                Some(du),
                Some(&state_index),
                &empty_nn_outputs,
            );
        });

    Ok(crate::ode::OdeSpec {
        rhs,
        n_states,
        state_names: state_names.to_vec(),
        obs_cmt_idx,
        output_fn: None,
        diffusion_var: Vec::new(),
    })
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

    for line in content.lines() {
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
                None => Some(BlockTarget::Unnamed(ty)),
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
    let theta_re = Regex::new(
        r"(?i)theta\s+(\w+)\s*\(\s*([0-9eE.+-]+)\s*(?:,\s*([0-9eE.+-]+)\s*,\s*([0-9eE.+-]+))?\s*(?:,\s*(FIX)\b)?\s*\)",
    )
    .unwrap();

    // omega NAME ~ value [(sd|variance|var)] [FIX]
    //
    // Initial value defaults to the variance scale (matching how the optimizer
    // stores omega internally). Append `(sd)` to declare the value on the
    // standard-deviation scale — the parser squares it before storing. The
    // `(variance)` / `(var)` annotation is accepted as an explicit no-op for
    // symmetry with sigma.
    let omega_re = Regex::new(
        r"(?i)omega\s+(\w+)\s*~\s*([0-9eE.+-]+)(?:\s*\((sd|variance|var)\))?(?:\s+(FIX)\b)?",
    )
    .unwrap();

    // block_omega (NAME1, NAME2, ...) = [lower_triangle_values]  |  ... FIX
    //
    // Block omegas are variance-scale only — the lower triangle mixes
    // variances and covariances, so a single `(sd)` flag would be ambiguous.
    let block_omega_re =
        Regex::new(r"(?i)block_omega\s*\(([^)]+)\)\s*=\s*\[([^\]]+)\](?:\s+(FIX)\b)?").unwrap();

    // sigma NAME ~ value [(sd|variance|var)] [FIX]
    //
    // As of issue #56, sigma defaults to the variance scale (matching omega).
    // `(sd)` opts back into specifying a standard deviation directly. The
    // parser converts variance → internal SD via `sqrt` so the residual-error
    // and likelihood code (which work in SD) need no changes.
    let sigma_re = Regex::new(
        r"(?i)sigma\s+(\w+)\s*~\s*([0-9eE.+-]+)(?:\s*\((sd|variance|var)\))?(?:\s+(FIX)\b)?",
    )
    .unwrap();

    // kappa NAME ~ value [(sd|variance|var)] [FIX]  (IOV diagonal variance)
    let kappa_re = Regex::new(
        r"(?i)kappa\s+(\w+)\s*~\s*([0-9eE.+-]+)(?:\s*\((sd|variance|var)\))?(?:\s+(FIX)\b)?",
    )
    .unwrap();

    // block_kappa (NAME1, NAME2, ...) = [lower_triangle_values]  |  ... FIX
    let block_kappa_re =
        Regex::new(r"(?i)block_kappa\s*\(([^)]+)\)\s*=\s*\[([^\]]+)\](?:\s+(FIX)\b)?").unwrap();

    for line in lines {
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
            let fixed = caps.get(5).is_some();
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
            let init_as_sd = caps
                .get(3)
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
            let fixed = caps.get(4).is_some();
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
            let init_as_sd = caps
                .get(3)
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
            let fixed = caps.get(4).is_some();
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
            let init_as_sd = caps
                .get(3)
                .map(|m| m.as_str().eq_ignore_ascii_case("sd"))
                .unwrap_or(false);
            if raw < 0.0 {
                let scale = if init_as_sd { "SD" } else { "variance" };
                return Err(format!(
                    "kappa '{name}' has a negative initial {scale} ({raw}); both variance and SD must be non-negative"
                ));
            }
            let variance = if init_as_sd { raw * raw } else { raw };
            let fixed = caps.get(4).is_some();
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

fn parse_structural_model(lines: &[String]) -> Result<(PkModel, HashMap<String, String>), String> {
    // pk model_name(param=VAR, param=VAR, ...)
    let pk_re = Regex::new(r"pk\s+(\w+)\(([^)]+)\)").unwrap();

    for line in lines {
        if let Some(caps) = pk_re.captures(line) {
            let model_name = &caps[1];
            let pk_model = match model_name {
                "one_cpt_iv_bolus" | "one_compartment_iv_bolus" => PkModel::OneCptIvBolus,
                "one_cpt_oral" | "one_compartment_oral" => PkModel::OneCptOral,
                "one_cpt_infusion" | "one_compartment_infusion" => PkModel::OneCptInfusion,
                "two_cpt_iv_bolus" | "two_compartment_iv_bolus" => PkModel::TwoCptIvBolus,
                "two_cpt_oral" | "two_compartment_oral" => PkModel::TwoCptOral,
                "two_cpt_infusion" | "two_compartment_infusion" => PkModel::TwoCptInfusion,
                "three_cpt_iv_bolus" | "three_compartment_iv_bolus" => PkModel::ThreeCptIvBolus,
                "three_cpt_oral" | "three_compartment_oral" => PkModel::ThreeCptOral,
                "three_cpt_infusion" | "three_compartment_infusion" => PkModel::ThreeCptInfusion,
                other => return Err(format!("Unknown PK model: {}", other)),
            };

            let params_str = &caps[2];
            let mut param_map = HashMap::new();
            for pair in params_str.split(',') {
                let parts: Vec<&str> = pair.split('=').map(|s| s.trim()).collect();
                if parts.len() == 2 {
                    param_map.insert(parts[0].to_lowercase(), parts[1].to_string());
                }
            }

            return Ok((pk_model, param_map));
        }
    }

    Err("No PK model found in [structural_model] block".to_string())
}

// --- Error model parsing ---

fn parse_error_model(lines: &[String]) -> Result<(ErrorModel, Vec<String>), String> {
    // DV ~ proportional(SIGMA_NAME)
    // DV ~ additive(SIGMA_NAME)
    // DV ~ combined(SIGMA1, SIGMA2)
    let re = Regex::new(r"(\w+)\s*~\s*(\w+)\(([^)]+)\)").unwrap();

    for line in lines {
        if let Some(caps) = re.captures(line) {
            let error_type = &caps[2];
            let sigma_names: Vec<String> =
                caps[3].split(',').map(|s| s.trim().to_string()).collect();

            let error_model = match error_type.to_lowercase().as_str() {
                "additive" => ErrorModel::Additive,
                "proportional" => ErrorModel::Proportional,
                "combined" => ErrorModel::Combined,
                other => return Err(format!("Unknown error model: {}", other)),
            };

            return Ok((error_model, sigma_names));
        }
    }

    Err("No error model found in [error_model] block".to_string())
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
    #[cfg(feature = "nn")] covariate_nns: &[crate::nn::CovariateNn],
) -> Result<(PkParamFn, Vec<String>), String> {
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
    resolve_variable_indices(&mut stmts_resolved, &var_idx, &cov_idx);

    let stmts_owned = stmts_resolved;
    let vars_in_order = var_names.to_vec();

    // Pre-resolve pk_map → indexed (pk_slot, var_slot) pairs so the hot
    // loop is two array reads instead of two HashMap probes.
    let pk_assignment_mapping: Vec<(usize, usize)> = pk_param_map
        .iter()
        .filter_map(|(pk_name, var_name)| {
            let pk_slot = PkParams::name_to_index(pk_name)?;
            let var_slot = var_idx.get(var_name).copied().or_else(|| {
                // Fall back to lowercase lookup — matches the previous
                // `vars.get(var_name.to_lowercase())` compat behaviour.
                var_idx.get(&var_name.to_lowercase()).copied()
            })?;
            Some((pk_slot, var_slot))
        })
        .collect();
    let is_analytical_pk = !pk_param_map.is_empty();

    // ODE branch counterpart of the same pre-resolution. Lagtime is
    // handled separately because it can live outside the first
    // MAX_PK_PARAMS positions.
    let ode_positional_slots: Vec<usize> = vars_in_order
        .iter()
        .enumerate()
        .filter_map(|(pos, var_name)| {
            if pos >= MAX_PK_PARAMS {
                return None;
            }
            var_idx.get(var_name).copied().map(|s| s)
        })
        .collect();
    let ode_lagtime_slot: Option<usize> = vars_in_order.iter().find_map(|var_name| {
        let upper = var_name.to_uppercase();
        if upper == "LAGTIME" || upper == "ALAG" {
            var_idx.get(var_name).copied()
        } else {
            None
        }
    });

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

            eval_statements_indexed(&stmts_owned, theta, eta, &cov_vec, &mut vars, &nn_outputs);

            let mut p = PkParams::default();
            if is_analytical_pk {
                for &(pk_slot, var_slot) in &pk_assignment_mapping {
                    p.values[pk_slot] = vars[var_slot];
                }
            } else {
                // ODE model: store individual params by declaration order.
                for (pos, &var_slot) in ode_positional_slots.iter().enumerate() {
                    p.values[pos] = vars[var_slot];
                }
                // Lagtime side-write (see original commentary).
                if let Some(lag_slot) = ode_lagtime_slot {
                    p.values[PK_IDX_LAGTIME] = vars[lag_slot];
                }
            }
            p
        },
    );
    Ok((pk_param_fn, referenced_covariates))
}

// --- Simple expression AST and evaluator ---

#[derive(Debug, Clone)]
enum Expression {
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

#[derive(Debug, Clone)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Copy)]
enum CmpOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

#[derive(Debug, Clone)]
enum Condition {
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
}

impl<'a> ParseCtx<'a> {
    fn new(theta_names: &'a [String], eta_names: &'a [String], defined_vars: &'a [String]) -> Self {
        const EMPTY_NN: &[(String, Vec<String>)] = &[];
        Self {
            theta_names,
            eta_names,
            defined_vars,
            fallback_covariate: true,
            nn_specs: EMPTY_NN,
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
        }
    }

    fn with_nn_specs(mut self, nn_specs: &'a [(String, Vec<String>)]) -> Self {
        self.nn_specs = nn_specs;
        self
    }
}

/// Walk an expression tree and accumulate every covariate name it references.
fn collect_covariates(expr: &Expression, out: &mut std::collections::HashSet<String>) {
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

fn collect_covariates_in_condition(cond: &Condition, out: &mut std::collections::HashSet<String>) {
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
fn collect_covariates_in_stmts(stmts: &[Statement], out: &mut std::collections::HashSet<String>) {
    for s in stmts {
        match s {
            Statement::Assign(_, e) | Statement::AssignIdx(_, e) | Statement::DiffEq(_, e) => {
                collect_covariates(e, out)
            }
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
        Expression::Variable(name) => vars.get(name).copied().unwrap_or(0.0),
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
            }
        }
        Expression::UnaryFn(name, arg) => {
            let v = eval_expression(arg, theta, eta, covariates, vars, nn_outputs);
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
            }
        }
        Expression::UnaryFn(name, arg) => {
            let v = eval_expression_indexed(arg, theta, eta, covariates, vars, nn_outputs);
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

/// Indexed-form statement executor; mirror of `eval_statements`. Only
/// handles `AssignIdx` + `If`; if any non-indexed variant slips through,
/// it falls back to a no-op (defensive — caller should ensure the AST
/// was resolved). `DiffEq` is not supported here because the indexed
/// path is only used by `pk_param_fn` (no derivatives).
fn eval_statements_indexed(
    stmts: &[Statement],
    theta: &[f64],
    eta: &[f64],
    covariates: &[f64],
    vars: &mut [f64],
    nn_outputs: &[Vec<f64>],
) {
    for s in stmts {
        match s {
            Statement::AssignIdx(idx, expr) => {
                let v = eval_expression_indexed(expr, theta, eta, covariates, vars, nn_outputs);
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
                    if eval_condition_indexed(cond, theta, eta, covariates, vars, nn_outputs) {
                        eval_statements_indexed(body, theta, eta, covariates, vars, nn_outputs);
                        taken = true;
                        break;
                    }
                }
                if !taken {
                    if let Some(eb) = else_body {
                        eval_statements_indexed(eb, theta, eta, covariates, vars, nn_outputs);
                    }
                }
            }
            // Non-indexed Assign/DiffEq shouldn't appear in a resolved AST
            // for `pk_param_fn`. Silently skip.
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
/// Used by `build_pk_param_fn` so the hot-loop closure can run on
/// `Vec<f64>` slots instead of paying per-call HashMap-lookup overhead.
fn resolve_variable_indices(
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
            Statement::AssignIdx(_, _) => {
                // Indexed-form statements are only produced by
                // `resolve_variable_indices` for the `pk_param_fn`
                // closure, which uses `eval_statements_indexed`
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
    let mut paren_depth = 0i32;
    for t in tokens {
        match t {
            Token::LParen => {
                paren_depth += 1;
                out.push(Token::LParen);
            }
            Token::RParen => {
                paren_depth -= 1;
                out.push(Token::RParen);
            }
            Token::Newline if paren_depth > 0 => {
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

    #[test]
    fn test_parse_method_single() {
        let opts = parse_fit_options(&["method = focei".to_string()]).unwrap();
        assert_eq!(opts.method, EstimationMethod::FoceI);
        assert!(opts.methods.is_empty());
        assert!(opts.interaction);
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
                "covariance = false".to_string(),
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
        // exp(ETA_CL + ETA_OCC) is not a bare exp(Eta) — rejected.
        let m = detect_one(
            "CL = TVCL * exp(ETA_CL + ETA_OCC)",
            &["TVCL"],
            &["ETA_CL", "ETA_OCC"],
        );
        assert!(m.is_none());
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
  pk one_cpt_iv_bolus(cl=CL, v=V)

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
    fn test_parse_optimizer_defaults_to_slsqp() {
        // No [fit_options] block → default optimizer.
        let content = minimal_model_with_fit_options("  maxiter = 100");
        let parsed = parse_full_model(&content).unwrap();
        assert_eq!(parsed.fit_options.optimizer, Optimizer::Slsqp);
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
        //   optimizer = slsqp, inner_maxiter = 200, inner_tol = 1e-4,
        //   steihaug_max_iters = None (adaptive).
        let opts = FitOptions::default();
        assert_eq!(opts.optimizer, Optimizer::Slsqp);
        assert_eq!(opts.inner_maxiter, 200);
        assert!((opts.inner_tol - 1e-4).abs() < 1e-20);
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
  pk one_cpt_iv_bolus(cl=CL, v=V)

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
  pk one_cpt_iv_bolus(cl=CL, v=V)

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
  pk one_cpt_iv_bolus(cl=CL, v=V)

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
  pk one_cpt_iv_bolus(cl=CL, v=V)

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
  pk one_cpt_iv_bolus(cl=CL, v=V)

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
  pk one_cpt_iv_bolus(cl=CL, v=V)

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
                        Statement::Assign(_, _) | Statement::AssignIdx(_, _) => {}
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
        let result = build_ode_spec(&ode_lines, &state_names, Some("central"), &[]);
        assert!(
            result.is_ok(),
            "same state in different branches must be allowed"
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
  pk one_cpt_iv_bolus(cl=CL, v=V)

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
  pk one_cpt_iv_bolus(cl=1, v=1)

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
        // TVCL * exp(ETA) pattern: theta is not on log scale (theta IS TVCL)
        assert!(cl_info.linked_theta.is_none());
        // theta_transform for TVCL (theta index 0) stays Identity
        assert_eq!(model.theta_transform[0], ThetaTransform::Identity);
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
  pk one_cpt_iv_bolus(cl=1, v=1)

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
  pk one_cpt_iv_bolus(cl=1, v=1)

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
  pk one_cpt_iv_bolus(cl=CL, v=V)
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
  pk one_cpt_iv_bolus(cl=CL, v=V)

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
  pk one_cpt_iv_bolus(cl=CL, v=V)

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
                let s = scale_fn(&theta, &eta, &cov);
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
                let s = scale_fn(&theta, &eta, &cov);
                assert!((s - 1.2).abs() < 1e-12, "expected 1.2, got {}", s);
            }
            other => panic!("expected ExpressionScale, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_scaling_expression_rejects_indiv_param_ref() {
        // Phase 1 restriction: `V` is an individual parameter — must be rejected
        // in Form B because the closure has no path to materialize it without
        // re-running pk_param_fn.
        let src = analytical_model_with_scaling(Some("  obs_scale = 1000 / V\n"));
        let err =
            parse_model_string(&src).expect_err("indiv param ref in obs_scale must be rejected");
        assert!(
            err.contains("individual parameter") && err.contains('V'),
            "expected indiv-param rejection error, got: {}",
            err
        );
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
        assert!(ode.obs_cmt_idx.is_none(), "Form C clears obs_cmt_idx");
        let out_fn = ode.output_fn.as_ref().expect("Form C sets output_fn");

        // ODE writes indiv params sequentially into pk_params_flat[0..n] in
        // declaration order: [CL, V, KA] -> pk[0..3]. State order: [depot,
        // central] -> state[0..2]. So y = central / V = state[1] / pk[1].
        let state = vec![0.0, 100.0]; // depot=0, central=100
        let mut pk = vec![0.0f64; crate::types::MAX_PK_PARAMS];
        pk[0] = 1.0; // CL
        pk[1] = 50.0; // V
        pk[2] = 1.0; // KA
        let cov = HashMap::new();
        let y = out_fn(&state, &pk, &cov);
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
        assert_eq!(ode.obs_cmt_idx, Some(1));
        assert!(ode.output_fn.is_none());
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
        let out_fn = ode.output_fn.as_ref().unwrap();

        let state = vec![0.0, 50.0]; // depot=0, central=50
        let mut pk = vec![0.0f64; crate::types::MAX_PK_PARAMS];
        pk[0] = 1.0; // CL
        pk[1] = 50.0; // V
        pk[2] = 1.0; // KA
        let mut cov = HashMap::new();
        cov.insert("WT".to_string(), 70.0);
        let y = out_fn(&state, &pk, &cov);
        // central/V * WT = 50/50 * 70 = 70. With the bug, WT was treated as
        // a Variable and read 0 from the empty vars map → y = 0.
        assert!(
            (y - 70.0).abs() < 1e-12,
            "expected 70.0 (covariate WT must resolve), got {}",
            y
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
    fn test_parse_scaling_expression_rejects_ad_gradient() {
        // Form B + gradient = ad must error with clear message.
        let base = analytical_model_with_scaling(Some("  obs_scale = TVV / 10\n"));
        // Replace `gradient = fd` with `gradient = ad`.
        let src = base.replace("gradient = fd", "gradient = ad");
        let err =
            parse_model_string(&src).expect_err("ExpressionScale + gradient = ad must be rejected");
        assert!(
            err.to_lowercase().contains("expression") && err.contains("gradient = fd"),
            "expected gradient=fd hint, got: {}",
            err
        );
    }
}
