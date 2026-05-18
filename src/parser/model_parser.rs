use crate::types::*;
use regex::Regex;
use std::collections::HashMap;
use std::path::Path;
use std::sync::LazyLock;

static DIFFUSION_LINE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^(\w+)\s*~\s*([0-9eE.+-]+)(?:\s+(FIX)\b)?").unwrap());

// ── Mu-referencing pattern detection ────────────────────────────────────────

/// Walk a Mul-chain and collect direct Theta indices (not inside any function).
fn collect_mul_thetas(expr: &Expression, out: &mut Vec<usize>) {
    match expr {
        Expression::Theta(i) => out.push(*i),
        Expression::BinOp(l, BinOp::Mul, r) => {
            collect_mul_thetas(l, out);
            collect_mul_thetas(r, out);
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
/// Returns `Some((eta_idx, theta_idx, log_transformed))` or `None`.
fn detect_pattern(expr: &Expression) -> Option<(usize, usize, bool)> {
    match expr {
        // Pattern 2: exp(log(THETA) + ETA)
        Expression::UnaryFn(name, inner) if name == "exp" => {
            // inner must be Add with log(Theta) and Eta in either order
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
                    return Some((ei, ti, true));
                }
            }
            None
        }
        // Pattern 3: THETA + ETA or ETA + THETA
        Expression::BinOp(lhs, BinOp::Add, rhs) => match (lhs.as_ref(), rhs.as_ref()) {
            (Expression::Theta(ti), Expression::Eta(ei)) => Some((*ei, *ti, false)),
            (Expression::Eta(ei), Expression::Theta(ti)) => Some((*ei, *ti, false)),
            _ => None,
        },
        // Pattern 1 / 4: product containing Theta and exp(Eta)
        _ => {
            let mut thetas = Vec::new();
            collect_mul_thetas(expr, &mut thetas);
            if thetas.len() == 1 {
                if let Some(ei) = find_exp_eta_in_mul(expr) {
                    return Some((ei, thetas[0], true));
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
fn detect_mu_refs(
    stmts: &[Statement],
    theta_names: &[String],
    eta_names: &[String],
) -> HashMap<String, MuRef> {
    let mut result = HashMap::new();
    for s in stmts {
        if let Statement::Assign(_, expr) = s {
            if let Some((eta_idx, theta_idx, log_transformed)) = detect_pattern(expr) {
                if eta_idx < eta_names.len() && theta_idx < theta_names.len() {
                    result.insert(
                        eta_names[eta_idx].clone(),
                        MuRef {
                            theta_name: theta_names[theta_idx].clone(),
                            log_transformed,
                        },
                    );
                }
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

    // TVCL * exp(ETA), exp(log(THETA) + ETA), TVCL + ETA
    if let Some((ei, ti, log_transformed)) = detect_pattern(expr) {
        if ti < n_theta {
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
    let blocks = extract_blocks(content)?;
    let name = extract_model_name(content);

    // ── Required blocks ──
    let param_lines = blocks
        .get("parameters")
        .ok_or("Missing [parameters] block")?;
    let (thetas, omegas, block_omegas, sigmas, eta_names_bsv, kappa_info) =
        parse_parameters(param_lines)?;

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

    // theta_names is extended below after diffusion thetas are appended
    let mut theta_names: Vec<String> = thetas.iter().map(|t| t.name.clone()).collect();
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
    let bare_ctx = ParseCtx::new(&theta_names, &eta_names, &[]);
    let pre_stmts = parse_block_statements(&indiv_text, bare_ctx, StatementMode::Plain)?;
    let all_assigned = assigned_vars_in_order(&pre_stmts);
    let indiv_var_names = top_level_assigned_vars(&pre_stmts);
    let indiv_ctx = ParseCtx::new(&theta_names, &eta_names, &all_assigned);
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
        let mut ode_spec =
            build_ode_spec(ode_lines, &state_names, &obs_cmt_name, &indiv_var_names)?;

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

    // Build pk_param_fn with the extended eta context (BSV + kappa names).
    let (pk_param_fn, referenced_covariates) =
        build_pk_param_fn(indiv_stmts.clone(), &pk_param_map, &indiv_var_names)?;

    // Append diffusion parameters to the theta list. They are treated as
    // positive variance parameters: log-transformed during optimisation,
    // lower-bounded at 0, no upper bound. They appear last in the theta vector
    // so existing theta indices are unaffected.
    let diff_theta_start = thetas.len(); // index of first diffusion theta
    let mut theta_values: Vec<f64> = thetas.iter().map(|t| t.init).collect();
    let mut theta_lower: Vec<f64> = thetas.iter().map(|t| t.lower).collect();
    let mut theta_upper: Vec<f64> = thetas.iter().map(|t| t.upper).collect();
    let mut theta_fixed: Vec<bool> = thetas.iter().map(|t| t.fixed).collect();
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
    n_theta = theta_names.len(); // set here after diffusion thetas are appended above
    // BSV omega is built from the BSV-only eta names (no kappas)
    let omega = build_omega_matrix(&omegas, &block_omegas, &eta_names_bsv)?;
    let omega_fixed = build_omega_fixed(&omegas, &block_omegas, &eta_names_bsv)?;
    let sigma_values: Vec<f64> = sigmas.iter().map(|s| s.value).collect();
    let sigma_fixed: Vec<bool> = sigmas.iter().map(|s| s.fixed).collect();
    let sigma = SigmaVector {
        values: sigma_values,
        names: sigma_names,
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
            Some(Box::new(
                move |theta: &[f64], covariates: &HashMap<String, f64>| {
                    let zero_eta = vec![0.0; tv_eta_names.len()];
                    let mut vars: HashMap<String, f64> = HashMap::new();
                    eval_statements(
                        &stmts_for_tv,
                        theta,
                        &zero_eta,
                        covariates,
                        &mut vars,
                        None,
                        None,
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
    let all_mu_refs = detect_mu_refs(&indiv_stmts, &theta_names, &all_eta_names);
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
        tv_fn,
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
        "steihaug_max_iters" => opts.steihaug_max_iters = parse_usize("steihaug_max_iters")?,
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
        _ => return Ok(false),
    }
    opts.user_set_keys.push(key.to_string());
    Ok(true)
}

// ── [structural_model] ODE variant parser ───────────────────────────────────

fn parse_ode_structural(lines: &[String]) -> Result<(Vec<String>, String), String> {
    // ode(obs_cmt=central, states=[depot, central])
    let re =
        Regex::new(r"ode\(\s*obs_cmt\s*=\s*(\w+)\s*,\s*states\s*=\s*\[([^\]]+)\]\s*\)").unwrap();
    for line in lines {
        if let Some(caps) = re.captures(line) {
            let obs_cmt = caps[1].to_string();
            let states: Vec<String> = caps[2].split(',').map(|s| s.trim().to_string()).collect();
            return Ok((states, obs_cmt));
        }
    }
    Err(
        "Could not parse ODE structural model. Expected: ode(obs_cmt=NAME, states=[...])"
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

fn build_ode_spec(
    lines: &[String],
    state_names: &[String],
    obs_cmt_name: &str,
    indiv_param_names: &[String],
) -> Result<crate::ode::OdeSpec, String> {
    let n_states = state_names.len();
    let obs_cmt_idx = state_names
        .iter()
        .position(|s| s == obs_cmt_name)
        .ok_or_else(|| {
            format!(
                "Observable compartment '{}' not in states {:?}",
                obs_cmt_name, state_names
            )
        })?;

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
                Statement::Assign(_, _) => {}
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
                Statement::Assign(_, _) => {}
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
            eval_statements(
                &stmts_owned,
                &empty_theta,
                &empty_eta,
                &empty_cov,
                &mut vars,
                Some(du),
                Some(&state_index),
            );
        });

    Ok(crate::ode::OdeSpec {
        rhs,
        n_states,
        state_names: state_names.to_vec(),
        obs_cmt_idx,
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
    value: f64,
    fixed: bool,
}

/// Diagonal inter-occasion variability (kappa) specification.
struct KappaSpec {
    name: String,
    variance: f64,
    fixed: bool,
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

fn extract_blocks(content: &str) -> Result<HashMap<String, Vec<String>>, String> {
    let mut blocks: HashMap<String, Vec<String>> = HashMap::new();
    // Anchor on the whole line so things like `states=[central]` inside an
    // ODE structural definition aren't misread as a block-tag opener.
    let block_re = Regex::new(r"^\[(\w+)\]$").unwrap();

    let mut current_block: Option<String> = None;

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
            current_block = Some(caps[1].to_lowercase());
            continue;
        }

        if trimmed.starts_with("model ") || trimmed == "end" {
            continue;
        }

        if let Some(ref block) = current_block {
            blocks
                .entry(block.clone())
                .or_default()
                .push(trimmed.to_string());
        }
    }

    Ok(blocks)
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

    // omega NAME ~ value  |  omega NAME ~ value FIX
    let omega_re = Regex::new(r"(?i)omega\s+(\w+)\s*~\s*([0-9eE.+-]+)(?:\s+(FIX)\b)?").unwrap();

    // block_omega (NAME1, NAME2, ...) = [lower_triangle_values]  |  ... FIX
    let block_omega_re =
        Regex::new(r"(?i)block_omega\s*\(([^)]+)\)\s*=\s*\[([^\]]+)\](?:\s+(FIX)\b)?").unwrap();

    // sigma NAME ~ value  |  sigma NAME ~ value FIX
    let sigma_re = Regex::new(r"(?i)sigma\s+(\w+)\s*~\s*([0-9eE.+-]+)(?:\s+(FIX)\b)?").unwrap();

    // kappa NAME ~ value  |  kappa NAME ~ value FIX  (IOV diagonal variance)
    let kappa_re = Regex::new(r"(?i)kappa\s+(\w+)\s*~\s*([0-9eE.+-]+)(?:\s+(FIX)\b)?").unwrap();

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
            let variance: f64 = caps[2]
                .parse()
                .map_err(|_| format!("Bad omega: {}", line))?;
            let fixed = caps.get(3).is_some();
            eta_names_ordered.push(name.clone());
            omegas.push(OmegaSpec {
                name,
                variance,
                fixed,
            });
        } else if let Some(caps) = sigma_re.captures(line) {
            let name = caps[1].to_string();
            let value: f64 = caps[2]
                .parse()
                .map_err(|_| format!("Bad sigma: {}", line))?;
            let fixed = caps.get(3).is_some();
            sigmas.push(SigmaSpec { name, value, fixed });
        } else if let Some(caps) = kappa_re.captures(line) {
            let name = caps[1].to_string();
            let variance: f64 = caps[2]
                .parse()
                .map_err(|_| format!("Bad kappa: {}", line))?;
            let fixed = caps.get(3).is_some();
            kappa_names_ordered.push(name.clone());
            kappas.push(KappaSpec {
                name,
                variance,
                fixed,
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
fn build_pk_param_fn(
    stmts: Vec<Statement>,
    pk_param_map: &HashMap<String, String>,
    var_names: &[String],
) -> Result<(PkParamFn, Vec<String>), String> {
    // Covariates referenced anywhere in the block (including inside if-bodies
    // and condition expressions). Sorted for deterministic error messages.
    let mut cov_set: std::collections::HashSet<String> = std::collections::HashSet::new();
    collect_covariates_in_stmts(&stmts, &mut cov_set);
    let mut referenced_covariates: Vec<String> = cov_set.into_iter().collect();
    referenced_covariates.sort();

    let pk_map: HashMap<String, String> = pk_param_map.clone();
    let stmts_owned = stmts;
    let vars_in_order = var_names.to_vec();

    let pk_param_fn: PkParamFn = Box::new(
        move |theta: &[f64], eta: &[f64], covariates: &HashMap<String, f64>| {
            let mut vars: HashMap<String, f64> = HashMap::new();
            eval_statements(&stmts_owned, theta, eta, covariates, &mut vars, None, None);

            let mut p = PkParams::default();

            if pk_map.is_empty() {
                // ODE model or no pk_param_map: store individual params by declaration order
                for (i, var_name) in vars_in_order.iter().enumerate() {
                    if i < MAX_PK_PARAMS {
                        if let Some(&val) = vars.get(var_name) {
                            p.values[i] = val;
                        }
                    }
                }
                // Lagtime is consumed by `ode_predictions` outside the user's
                // RHS, so a LAGTIME/ALAG-named individual parameter must land
                // at the canonical PK_IDX_LAGTIME slot regardless of its
                // declaration position. Side-write here AFTER the sequential
                // loop so this takes precedence over a positional value that
                // happens to land on slot 8 (only relevant when the user has
                // ≥9 individual parameters).
                for var_name in vars_in_order.iter() {
                    let upper = var_name.to_uppercase();
                    if upper == "LAGTIME" || upper == "ALAG" {
                        if let Some(&val) = vars.get(var_name) {
                            p.values[PK_IDX_LAGTIME] = val;
                        }
                    }
                }
            } else {
                // Analytical model: map pk_param_name → value via pk_param_map
                let mut named = HashMap::new();
                for (pk_name, var_name) in &pk_map {
                    if let Some(&val) = vars.get(var_name) {
                        named.insert(pk_name.clone(), val);
                    } else if let Some(&val) = vars.get(&var_name.to_lowercase()) {
                        named.insert(pk_name.clone(), val);
                    }
                }
                p = PkParams::from_hashmap(&named);
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
    BinOp(Box<Expression>, BinOp, Box<Expression>),
    UnaryFn(String, Box<Expression>),
    Power(Box<Expression>, Box<Expression>),
    /// `if (cond) then_expr else else_expr` — value-producing inline conditional.
    Conditional(Box<Condition>, Box<Expression>, Box<Expression>),
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
}

impl<'a> ParseCtx<'a> {
    fn new(theta_names: &'a [String], eta_names: &'a [String], defined_vars: &'a [String]) -> Self {
        Self {
            theta_names,
            eta_names,
            defined_vars,
            fallback_covariate: true,
        }
    }

    fn ode(defined_vars: &'a [String]) -> Self {
        const EMPTY: &[String] = &[];
        Self {
            theta_names: EMPTY,
            eta_names: EMPTY,
            defined_vars,
            fallback_covariate: false,
        }
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
            Statement::Assign(_, e) | Statement::DiffEq(_, e) => collect_covariates(e, out),
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
) -> f64 {
    match expr {
        Expression::Literal(v) => *v,
        Expression::Theta(i) => theta[*i],
        Expression::Eta(i) => eta[*i],
        Expression::Covariate(name) => covariates.get(name).copied().unwrap_or(0.0),
        Expression::Variable(name) => vars.get(name).copied().unwrap_or(0.0),
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

fn eval_condition(
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
            c if c.is_ascii_digit() || c == '.' => {
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
        // Failed apply must not mutate — default preserved.
        assert_eq!(opts.saem_n_exploration, 150);
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
        let examples_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
        let mut seen = 0;
        for entry in std::fs::read_dir(&examples_dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|s| s.to_str()) != Some("ferx") {
                continue;
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
        let refs = detect_mu_refs(&stmts, &tn, &en);
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
        let refs = detect_mu_refs(&stmts, &tn, &en);
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
            },
            OmegaSpec {
                name: "ETA_V".into(),
                variance: 0.04,
                fixed: false,
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
            },
            OmegaSpec {
                name: "ETA_V".into(),
                variance: 0.04,
                fixed: false,
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
    fn test_parse_optimizer_defaults_to_bobyqa() {
        // No [fit_options] block → default optimizer.
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
        assert_eq!(parsed.fit_options.steihaug_max_iters, 30);
    }

    #[test]
    fn test_steihaug_max_iters_default() {
        // Default must match the documented value (50).
        let content = minimal_model_with_fit_options("  optimizer = trust_region");
        let parsed = parse_full_model(&content).unwrap();
        assert_eq!(parsed.fit_options.steihaug_max_iters, 50);
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
        //   optimizer = bobyqa, inner_maxiter = 200, inner_tol = 1e-4,
        //   steihaug_max_iters = 50.
        let opts = FitOptions::default();
        assert_eq!(opts.optimizer, Optimizer::Bobyqa);
        assert_eq!(opts.inner_maxiter, 200);
        assert!((opts.inner_tol - 1e-4).abs() < 1e-20);
        assert_eq!(opts.steihaug_max_iters, 50);
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
        assert_eq!(parsed.fit_options.steihaug_max_iters, 30);
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
        assert_eq!(opts.steihaug_max_iters, 30);
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
        eval_statements(&stmts, &theta, &[], &covs, &mut vars, None, None);
        assert!(
            (vars["CL"] - 6.0).abs() < 1e-12,
            "CL should pick the then-branch"
        );

        covs.insert("WT".to_string(), 60.0);
        let mut vars2 = HashMap::new();
        eval_statements(&stmts, &theta, &[], &covs, &mut vars2, None, None);
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
            eval_statements(&stmts, &theta, &[], &covs, &mut vars, None, None);
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
            eval_statements(&stmts, &theta, &[], &covs, &mut vars, None, None);
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
            eval_statements(&stmts, &theta, &[], &covs, &mut vars, None, None);
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
                        Statement::Assign(_, _) => {}
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
        let result = build_ode_spec(&ode_lines, &state_names, "central", &[]);
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
}
